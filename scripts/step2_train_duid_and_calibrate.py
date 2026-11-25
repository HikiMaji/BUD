# -*- coding: utf-8 -*-
"""
阶段二训练与统一门限校准脚本（step2_train_duid_and_calibrate.py）：
1. 严格零攻击样本训练（Attack-Agnostic Training）：仅在 OPV2V 训练集（20 条序列）的纯良性开放异构协议特征上，
   训练 BUDFusionModule 中的局部偶然不确定性预测头（以极坐标目标层光束退化真值为监督）
   与前景-高激活背景平衡采样的单位超球面正交原型码本（SphericalManifoldEpistemicScorer）；
2. 共同校准程序（严格落实历史教训第 10 条）：在独立的 OPV2V 验证集（6 条序列）纯良性开放异构数据流上，
   统一按固定 95% 良性分位数（5% 误报预算 alpha=0.05）一次性标定所有防御基线及本方法/消融变体的门限，
   固化写入 outputs/calibrated_thresholds.json，供测试集全程冻结调用！
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    build_foreground_gaussian_mask,
)
from src.protocol_models import HeteroAgentBranch
from src.uncertainty_fusion import (
    BUDFusionModule,
    extract_net_response,
    compute_global_residual_scores,
    compute_robosac_ccloss_scores,
)


CFG_PATH = "/root/autodl-tmp/ToolV2X_resources/cad_native_2026_10_05/AdvCollaborativePerception/models/OpenCOOD/pointpillar_attentive_fusion/config.yaml"
CKPT_PATH = "/root/autodl-tmp/ToolV2X_resources/cad_native_2026_10_05/AdvCollaborativePerception/models/OpenCOOD/pointpillar_attentive_fusion/latest.pth"


def set_seed(seed: int = 2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_trained_branches(ckpt_dir: str, bounded: bool = True) -> dict:
    suffix = "bounded" if bounded else "unconstrained"
    branches = {}
    for prof in COLLAB_PROFILES:
        b = HeteroAgentBranch(profile_name=prof, bounded=bounded, lip_gamma=1.12).cuda()
        path = os.path.join(ckpt_dir, f"{prof}_{suffix}.pth")
        b.load_state_dict(torch.load(path, map_location="cuda"), strict=True)
        b.eval()
        for p in b.parameters():
            p.requires_grad_(False)
        branches[prof] = b
    return branches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_frames", type=int, default=260)
    parser.add_argument("--duid_epochs", type=int, default=6)
    parser.add_argument("--val_calib_frames", type=int, default=240)
    parser.add_argument("--alpha_budget", type=float, default=0.05)
    args = parser.parse_args()

    set_seed(2026)
    ckpt_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
    train_dir = os.path.join(PROJECT_ROOT, "data_links", "opv2v_train")
    val_dir = os.path.join(PROJECT_ROOT, "data_links", "opv2v_val")

    train_ds, _ = load_opv2v_split_dataset(CFG_PATH, train_dir, train_dir, is_train=True)
    val_ds, _ = load_opv2v_split_dataset(CFG_PATH, val_dir, train_dir, is_train=False)

    anchor_wrapper = FrozenAnchorWrapper(CFG_PATH, CKPT_PATH, train_ds.pre_processor).cuda()
    branches_bounded = load_trained_branches(ckpt_dir, bounded=True)

    bud = BUDFusionModule(proto_channels=384).cuda()

    # ==========================================================================
    # 1. 在纯良性训练集上缓存开放异构协议特征，训练 DUID（偶然预测头 + 平衡球面正交码本）
    # ==========================================================================
    stride_tr = max(1, len(train_ds) // args.train_frames)
    tr_indices = list(range(0, len(train_ds), stride_tr))[: args.train_frames]
    print(f"[Step 2.1] Caching {len(tr_indices)} benign heterogeneous frames from OPV2V train...")

    duid_cache = []
    norm_99_samples = []
    hf_99_samples = []
    for idx in tr_indices:
        bundle = prepare_hetero_frame_bundle(train_ds, idx, anchor_wrapper, include_teacher=True)
        fg_m = build_foreground_gaussian_mask(bundle["gt_center_np"], bundle["gt_mask_np"], device="cuda")
        # 收集本车特征用于训练球面码本
        z_net_e, ego_nn = extract_net_response(bundle["z_ego"])
        hf_e = (z_net_e - F.avg_pool2d(z_net_e, 3, 1, 1)).norm(dim=1)
        norm_99_samples.append(float(torch.quantile(ego_nn.flatten(), 0.995).item()))
        hf_99_samples.append(float(torch.quantile(hf_e.flatten(), 0.995).item()))
        duid_cache.append({
            "z": bundle["z_ego"].cpu().half(),
            "u_geom": bundle["sigma_alea_ego"].squeeze(0).cpu().half(),
            "u_gt": bundle["sigma_alea_ego"].squeeze(0).cpu().half(),
            "fg": fg_m.cpu().half(),
        })
        for cav in bundle["cav_entries"]:
            prof = cav["profile"]
            with torch.no_grad():
                z_cav, _ = branches_bounded[prof](cav["pts_hetero_t"])
                z_net_c, cav_nn = extract_net_response(z_cav)
                hf_c = (z_net_c - F.avg_pool2d(z_net_c, 3, 1, 1)).norm(dim=1)
                norm_99_samples.append(float(torch.quantile(cav_nn.flatten(), 0.995).item()))
                hf_99_samples.append(float(torch.quantile(hf_c.flatten(), 0.995).item()))
            duid_cache.append({
                "z": z_cav.cpu().half(),
                "u_geom": cav["u_geom"].cpu().half(),
                "u_gt": cav["u_alea_gt"].cpu().half(),
                "fg": fg_m.cpu().half(),
            })

    # 自动标定良性净响应范数与局部高频能量 99.5% 包络上界
    calibrated_norm_bound = float(np.percentile(norm_99_samples, 98.0)) * 1.15
    calibrated_hf_bound = float(np.percentile(hf_99_samples, 98.0)) * 1.15
    bud.epis_scorer.benign_net_norm_99.fill_(max(calibrated_norm_bound, 6.0))
    bud.epis_scorer.benign_hf_99.fill_(max(calibrated_hf_bound, 1.2))
    print(f"  Calibrated benign_net_norm_99 = {float(bud.epis_scorer.benign_net_norm_99.item()):.3f}, benign_hf_99 = {float(bud.epis_scorer.benign_hf_99.item()):.3f}")

    opt_duid = torch.optim.AdamW(bud.parameters(), lr=2e-3, weight_decay=1e-4)
    bud.train()
    for ep in range(1, args.duid_epochs + 1):
        order = list(range(len(duid_cache)))
        random.shuffle(order)
        alea_losses, epis_losses = [], []
        for s_i in order:
            rec = duid_cache[s_i]
            z_b = rec["z"].cuda().float()
            ug_b = rec["u_geom"].unsqueeze(0).cuda().float()
            ugt_b = rec["u_gt"].unsqueeze(0).cuda().float()
            fg_b = rec["fg"].cuda().float()

            opt_duid.zero_grad()
            u_pred = bud.alea_head(z_b, ug_b)
            loss_a = F.smooth_l1_loss(u_pred, ugt_b)
            loss_e = bud.epis_scorer.compute_balanced_manifold_loss(z_b, fg_b)
            loss = loss_a + loss_e
            loss.backward()
            opt_duid.step()

            alea_losses.append(float(loss_a.item()))
            epis_losses.append(float(loss_e.item()))
        print(f"  [DUID Epoch {ep}/{args.duid_epochs}] AleaLoss={np.mean(alea_losses):.4f} | SphereLoss={np.mean(epis_losses):.4f}")

    bud.eval()

    # ==========================================================================
    # 2. 在独立验证集（validate, 6条序列）纯良性开放异构数据流上一次性标定所有方法门限
    # ==========================================================================
    stride_val = max(1, len(val_ds) // args.val_calib_frames)
    val_indices = list(range(0, len(val_ds), stride_val))[: args.val_calib_frames]
    print(f"\n[Step 2.2] Calibrating thresholds on {len(val_indices)} independent OPV2V validate frames (alpha={args.alpha_budget})...")

    # 步骤 2.2a：先统计各异构画像在共视清晰域上的良性域偏移基线方差 dom_var[profile]
    prof_mse_samples = {p: [] for p in COLLAB_PROFILES}
    val_bundles_light = []

    with torch.no_grad():
        for idx in val_indices:
            bundle = prepare_hetero_frame_bundle(val_ds, idx, anchor_wrapper, include_teacher=False)
            if len(bundle["cav_entries"]) == 0:
                continue
            z_list = [bundle["z_ego"]]
            ug_list, dm_list, p_names = [], [], []
            for cav in bundle["cav_entries"]:
                prof = cav["profile"]
                z_c, _ = branches_bounded[prof](cav["pts_hetero_t"])
                z_list.append(z_c)
                ug_list.append(cav["u_geom"].unsqueeze(0))
                dm_list.append(cav["dist_map"].unsqueeze(0))
                p_names.append(prof)

                # 在双边高可见清晰区测量良性跨架构残差基线
                clear_m = (bundle["sigma_alea_ego"] < 0.40) & (cav["u_geom"].unsqueeze(0) < 0.45)
                if clear_m.sum() > 30:
                    mse_c = ((bundle["z_ego"] - z_c) ** 2).mean(dim=1, keepdim=True)
                    prof_mse_samples[prof].append(float(mse_c[clear_m].quantile(0.85).item()))

            val_bundles_light.append({
                "z_all": torch.cat(z_list, dim=0).cpu().half(),
                "sigma_ego": bundle["sigma_alea_ego"].cpu().half(),
                "u_geom_cavs": torch.cat(ug_list, dim=0).cpu().half(),
                "dist_cavs": torch.cat(dm_list, dim=0).cpu().half(),
                "profiles": p_names,
            })

    for prof in COLLAB_PROFILES:
        if prof_mse_samples[prof]:
            bud.calib_params["dom_var"][prof] = round(float(np.median(prof_mse_samples[prof])), 5)
    print("  Calibrated benign domain variances:", bud.calib_params["dom_var"])

    # 步骤 2.2b：收集所有对比防御方法与本方法（含消融变体）在良性验证集上的分数分布，统一取 (1 - alpha) 分位数！
    q_target = 1.0 - args.alpha_budget  # 0.95
    scores_global_res = []
    scores_robosac_full = []
    scores_robosac_vis = []
    scores_ours_full = []
    scores_ours_no_alea = []
    scores_ours_no_epis = []
    scores_ours_quant95 = []
    grid_dcal_p99 = []
    grid_epis_p99 = []

    cls_ref = anchor_wrapper.model.cls_head
    reg_ref = anchor_wrapper.model.reg_head

    with torch.no_grad():
        for vb in val_bundles_light:
            z_all = vb["z_all"].cuda().float()
            s_ego = vb["sigma_ego"].cuda().float()
            ug_c = vb["u_geom_cavs"].cuda().float()
            dm_c = vb["dist_cavs"].cuda().float()
            p_names = vb["profiles"]

            scores_global_res.extend(compute_global_residual_scores(z_all).cpu().tolist())
            scores_robosac_full.extend(
                compute_robosac_ccloss_scores(z_all, cls_ref, reg_ref, s_ego, vis_calibrated=False).cpu().tolist()
            )
            scores_robosac_vis.extend(
                compute_robosac_ccloss_scores(z_all, cls_ref, reg_ref, s_ego, vis_calibrated=True).cpu().tolist()
            )

            d_cal, epis_c, _, _, sf_full, _, _ = bud.compute_verification_maps(
                z_all, s_ego, ug_c, dm_c, p_names,
                use_aleatoric_calib=True, use_epistemic_scorer=True, use_topk_cluster=True
            )
            scores_ours_full.extend(sf_full.cpu().tolist())
            grid_dcal_p99.append(float(torch.quantile(d_cal.flatten(), 0.995).item()))
            act_ep = epis_c[epis_c > 0.01]
            if act_ep.numel() > 10:
                grid_epis_p99.append(float(torch.quantile(act_ep, 0.95).item()))

            _, _, _, _, sf_no_alea, _, _ = bud.compute_verification_maps(
                z_all, s_ego, ug_c, dm_c, p_names,
                use_aleatoric_calib=False, use_epistemic_scorer=True, use_topk_cluster=True
            )
            scores_ours_no_alea.extend(sf_no_alea.cpu().tolist())

            _, _, _, _, sf_no_epis, _, _ = bud.compute_verification_maps(
                z_all, s_ego, ug_c, dm_c, p_names,
                use_aleatoric_calib=True, use_epistemic_scorer=False, use_topk_cluster=True
            )
            scores_ours_no_epis.extend(sf_no_epis.cpu().tolist())

            _, _, _, _, sf_q95, _, _ = bud.compute_verification_maps(
                z_all, s_ego, ug_c, dm_c, p_names,
                use_aleatoric_calib=True, use_epistemic_scorer=True, use_topk_cluster=False
            )
            scores_ours_quant95.extend(sf_q95.cpu().tolist())

    bud.calib_params["tau_frame"] = round(float(np.quantile(scores_ours_full, q_target)), 5)
    bud.calib_params["tau_grid"] = round(float(np.quantile(grid_dcal_p99, 0.90)), 5)
    bud.calib_params["tau_epis"] = round(float(np.quantile(grid_epis_p99, 0.90)), 5)

    # 同时采集一组同构 64线良性协作者分数（用于动机实验对比：若按同构 95% 门限部署到开放异构会发生多严重的误报崩溃！）
    homo_global_res, homo_robosac_full, homo_robosac_vis = [], [], []
    with torch.no_grad():
        for idx in val_indices[:80]:
            bundle = prepare_hetero_frame_bundle(val_ds, idx, anchor_wrapper, include_teacher=True)
            if not bundle["cav_entries"]:
                continue
            z_homo = torch.cat([bundle["z_ego"]] + [c["z_teacher"] for c in bundle["cav_entries"]], dim=0)
            s_ego = bundle["sigma_alea_ego"]
            homo_global_res.extend(compute_global_residual_scores(z_homo).cpu().tolist())
            homo_robosac_full.extend(
                compute_robosac_ccloss_scores(z_homo, cls_ref, reg_ref, s_ego, vis_calibrated=False).cpu().tolist()
            )
            homo_robosac_vis.extend(
                compute_robosac_ccloss_scores(z_homo, cls_ref, reg_ref, s_ego, vis_calibrated=True).cpu().tolist()
            )

    calib_export = {
        "alpha_budget": args.alpha_budget,
        "hetero_calibrated_95": {
            "global_residual": round(float(np.quantile(scores_global_res, q_target)), 5),
            "robosac_full": round(float(np.quantile(scores_robosac_full, q_target)), 5),
            "robosac_vis_calib": round(float(np.quantile(scores_robosac_vis, q_target)), 5),
            "ours_full": bud.calib_params["tau_frame"],
            "ours_no_alea": round(float(np.quantile(scores_ours_no_alea, q_target)), 5),
            "ours_no_epis": round(float(np.quantile(scores_ours_no_epis, q_target)), 5),
            "ours_quant95": round(float(np.quantile(scores_ours_quant95, q_target)), 5),
        },
        "homo_calibrated_95": {
            "global_residual": round(float(np.quantile(homo_global_res, q_target)), 5),
            "robosac_full": round(float(np.quantile(homo_robosac_full, q_target)), 5),
            "robosac_vis_calib": round(float(np.quantile(homo_robosac_vis, q_target)), 5),
        },
        "bud_calib_params": bud.calib_params,
    }

    torch.save(bud.state_dict(), os.path.join(ckpt_dir, "bud_fusion_duid.pth"))
    calib_json_path = os.path.join(PROJECT_ROOT, "outputs", "calibrated_thresholds.json")
    with open(calib_json_path, "w", encoding="utf-8") as f:
        json.dump(calib_export, f, indent=2, ensure_ascii=False)

    print("\n[Step 2 Complete] Saved DUID checkpoint & frozen thresholds:")
    print(json.dumps(calib_export, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
