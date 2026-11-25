# -*- coding: utf-8 -*-
"""
阶段五典型场景定性可视化脚本（step5_visualize_bev_case.py）：
在 OPV2V 未见测试集中自动检索同时包含“本车遮挡盲区真目标（0 ego pts, >=8 cav pts）”的场景帧，
生成 4 宫格 BEV 直观对比图 fig5_qualitative_bev_case.png，用于预开题 PPT 直观展示：
(a) Ego-Only + 目标层光束可见度不确定性场（展示 25m 清晰空地 vs 真实遮挡阴影，以及盲区漏检）；
(b) STAMP/HEAL-AttFusion（遭受鬼影伪造攻击产生严重假阳性框）；
(c) ROBOSAC-VisCalib（因异构残差误杀良性弱配车导致盲区补盲丢失或漏放局部伪造）；
(d) Ours BUD-Fusion（精准拦截鬼影伪造，同时无损保留良性异构盲区真车检出）。
"""

import os
import sys
import copy
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OPENCOOD_ROOT = os.path.join(PROJECT_ROOT, "third_party", "OpenCOOD")
for p in [PROJECT_ROOT, OPENCOOD_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.open_hetero_env import (
    COLLAB_PROFILES,
    FrozenAnchorWrapper,
    load_opv2v_split_dataset,
    prepare_hetero_frame_bundle,
)
from src.hetero_sensors import count_points_in_3d_boxes
from src.protocol_models import HeteroAgentBranch
from src.uncertainty_fusion import (
    BUDFusionModule,
    fuse_protocol_standard,
    compute_robosac_ccloss_scores,
    fuse_with_agent_rejection,
)
from src.attacks import generate_ghost_spoofing_attack

CFG_PATH = "/root/autodl-tmp/ToolV2X_resources/cad_native_2026_10_05/AdvCollaborativePerception/models/OpenCOOD/pointpillar_attentive_fusion/config.yaml"
CKPT_PATH = "/root/autodl-tmp/ToolV2X_resources/cad_native_2026_10_05/AdvCollaborativePerception/models/OpenCOOD/pointpillar_attentive_fusion/latest.pth"


def draw_bev_boxes(ax, boxes_8x3, color, lw=1.6, ls="-", label=None):
    if boxes_8x3 is None or len(boxes_8x3) == 0:
        return
    arr = boxes_8x3.detach().cpu().numpy() if isinstance(boxes_8x3, torch.Tensor) else boxes_8x3
    for idx_b, corners in enumerate(arr):
        poly_xy = corners[[0, 1, 2, 3, 0], :2]
        ax.plot(poly_xy[:, 0], poly_xy[:, 1], color=color, linewidth=lw, linestyle=ls, label=label if idx_b == 0 else None)


def main():
    ckpt_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
    calib_path = os.path.join(PROJECT_ROOT, "outputs", "calibrated_thresholds.json")
    with open(calib_path, "r", encoding="utf-8") as f:
        calib_data = json.load(f)

    train_dir = os.path.join(PROJECT_ROOT, "data_links", "opv2v_train")
    test_dir = os.path.join(PROJECT_ROOT, "data_links", "opv2v_test")
    test_ds, _ = load_opv2v_split_dataset(CFG_PATH, test_dir, train_dir, is_train=False)
    anchor_wrapper = FrozenAnchorWrapper(CFG_PATH, CKPT_PATH, test_ds.pre_processor).cuda()
    cls_ref = anchor_wrapper.model.cls_head
    reg_ref = anchor_wrapper.model.reg_head

    branches_b, branches_u = {}, {}
    for prof in COLLAB_PROFILES:
        bb = HeteroAgentBranch(prof, bounded=True, lip_gamma=1.12).cuda().eval()
        bb.load_state_dict(torch.load(os.path.join(ckpt_dir, f"{prof}_bounded.pth"), map_location="cuda"), strict=True)
        branches_b[prof] = bb
        bu = HeteroAgentBranch(prof, bounded=False).cuda().eval()
        bu.load_state_dict(torch.load(os.path.join(ckpt_dir, f"{prof}_unconstrained.pth"), map_location="cuda"), strict=True)
        branches_u[prof] = bu

    bud = BUDFusionModule(proto_channels=384).cuda().eval()
    bud.load_state_dict(torch.load(os.path.join(ckpt_dir, "bud_fusion_duid.pth"), map_location="cuda"), strict=True)
    bud.calib_params = copy.deepcopy(calib_data["bud_calib_params"])
    th_het = calib_data["hetero_calibrated_95"]

    # 搜索拥有 >=2 辆协作者且存在本车盲区被遮挡目标的典型交叉口测试帧
    chosen_bundle = None
    for idx in range(0, len(test_ds), 9):
        b = prepare_hetero_frame_bundle(test_ds, idx, anchor_wrapper, include_teacher=False)
        if len(b["cav_entries"]) < 2:
            continue
        with torch.no_grad():
            out_e = anchor_wrapper.head_forward(b["z_ego"])
            _, _, gt_t = test_ds.post_process(b["batch_for_post"], {"ego": out_e})
        if gt_t is None or len(gt_t) < 4:
            continue
        gt_np = gt_t.cpu().numpy()
        in_roi = np.abs(gt_np.mean(axis=1)[:, 0]) < 62.0
        if np.sum(in_roi) < 4:
            continue
        ego_cnt = count_points_in_3d_boxes(b["ego_pts_np"], gt_np)
        cav_cnt = np.zeros_like(ego_cnt)
        for c in b["cav_entries"]:
            cav_cnt = np.maximum(cav_cnt, count_points_in_3d_boxes(c["pts_full64_np"], gt_np))
        if np.any((ego_cnt == 0) & (cav_cnt >= 8) & in_roi):
            chosen_bundle = b
            break

    if chosen_bundle is None:
        chosen_bundle = prepare_hetero_frame_bundle(test_ds, 25, anchor_wrapper, include_teacher=False)

    cavs = chosen_bundle["cav_entries"]
    z_ego = chosen_bundle["z_ego"]
    s_ego = chosen_bundle["sigma_alea_ego"]
    bp = chosen_bundle["batch_for_post"]

    with torch.no_grad():
        zb_l, zu_l, ug_l, dm_l, p_names = [z_ego], [z_ego], [], [], []
        for c in cavs:
            pr = c["profile"]
            zb_l.append(branches_b[pr](c["pts_hetero_t"])[0])
            zu_l.append(branches_u[pr](c["pts_hetero_t"])[0])
            ug_l.append(c["u_geom"].unsqueeze(0))
            dm_l.append(c["dist_map"].unsqueeze(0))
            p_names.append(pr)
        zb_clean = torch.cat(zb_l, dim=0)
        zu_clean = torch.cat(zu_l, dim=0)
        ug_c = torch.cat(ug_l, dim=0)
        dm_c = torch.cat(dm_l, dim=0)

    # 在最后 1 辆协作者上施加鬼影伪造攻击（保留第 1 辆良性异构协作者提供真实盲区补盲！）
    atk_idx = len(cavs)
    zb_adv = generate_ghost_spoofing_attack(zb_clean, atk_idx, cls_ref, eps=1.45, steps=8, num_spoofs=3, frame_seed=chosen_bundle["sample_idx"])
    zu_adv = generate_ghost_spoofing_attack(zu_clean, atk_idx, cls_ref, eps=1.45, steps=8, num_spoofs=3, frame_seed=chosen_bundle["sample_idx"])

    with torch.no_grad():
        pb_ego, _, gt_boxes = test_ds.post_process(bp, {"ego": anchor_wrapper.head_forward(z_ego)})
        pb_u, _, _ = test_ds.post_process(bp, {"ego": anchor_wrapper.head_forward(fuse_protocol_standard(zu_adv, cls_head_ref=cls_ref))})
        sc_rv = compute_robosac_ccloss_scores(zb_adv, cls_ref, reg_ref, s_ego, vis_calibrated=True)
        zf_rv, _ = fuse_with_agent_rejection(zb_adv, sc_rv, th_het["robosac_vis_calib"], cls_head_ref=cls_ref)
        pb_rv, _, _ = test_ds.post_process(bp, {"ego": anchor_wrapper.head_forward(zf_rv)})
        zf_ou, _ = bud.forward_fuse(zb_adv, s_ego, ug_c, dm_c, p_names, cls_ref)
        pb_ou, _, _ = test_ds.post_process(bp, {"ego": anchor_wrapper.head_forward(zf_ou)})

    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.6))
    ego_pts = chosen_bundle["ego_pts_np"]
    pts_sub = ego_pts[::4]

    titles = [
        "(a) Ego-Only + Target-Layer Ray Uncertainty Map",
        "(b) Unconstrained STAMP/HEAL-AttFusion (Under Ghost Spoof)",
        "(c) ROBOSAC-VisCalib Defense (Rigid Consensus)",
        "(d) Ours: BUD-Fusion (Decoupled Soft-Gated Fusion)",
    ]
    preds = [pb_ego, pb_u, pb_rv, pb_ou]

    s_map = s_ego[0, 0].cpu().numpy()
    for idx_ax, (ax, title, pred_b) in enumerate(zip(axes.flat, titles, preds)):
        if idx_ax == 0:
            im = ax.imshow(
                s_map,
                extent=[-140.8, 140.8, -40.0, 40.0],
                origin="lower",
                cmap="YlOrRd",
                alpha=0.38,
                vmin=0.0,
                vmax=1.0,
                aspect="auto",
            )
            cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            cbar.set_label("$\\sigma_{alea,0}^2(x,y)$", fontsize=8)
        ax.scatter(pts_sub[:, 0], pts_sub[:, 1], s=0.18, c="#525252", alpha=0.45)
        draw_bev_boxes(ax, gt_boxes, color="#00a65a", lw=2.0, ls="-", label="Ground Truth 3D Box")
        draw_bev_boxes(ax, pred_b, color="#e41a1c", lw=1.3, ls="--", label="Predicted 3D Box")
        ax.scatter([0.0], [0.0], c="#08519c", marker="*", s=110, zorder=6, label="Ego (64L PP)")
        for ic, c in enumerate(cavs):
            cxy = c["cav_xy"].cpu().numpy()
            role_lbl = "Malicious Hetero CAV" if (ic + 1 == atk_idx) else f"Benign {c['profile'].split('_')[-1]}"
            col_c = "#980043" if (ic + 1 == atk_idx) else "#2b8cbe"
            ax.scatter([cxy[0]], [cxy[1]], c=col_c, marker="^", s=75, zorder=6, label=role_lbl)

        ax.set_xlim(-72, 72)
        ax.set_ylim(-28, 28)
        ax.set_xlabel("X (m) in Ego Coordinate")
        ax.set_ylabel("Y (m)")
        ax.set_title(title, fontweight="bold", fontsize=10.5)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    save_p = os.path.join(PROJECT_ROOT, "outputs", "figures", "fig5_qualitative_bev_case.png")
    os.makedirs(os.path.dirname(save_p), exist_ok=True)
    plt.savefig(save_p)
    plt.close(fig)
    print(f"[Step 5 Complete] Saved qualitative BEV case study figure to {save_p}")


if __name__ == "__main__":
    main()
