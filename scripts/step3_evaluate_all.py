# -*- coding: utf-8 -*-
"""
阶段三全面终测与机理剖析脚本（step3_evaluate_all.py）：
严格在未见测试集 OPV2V test（全部 16 条独立序列）上冻结全部模型权重与验证集标定门限，一次性完成四大核心实验：
1. 【动机实验 (Motivation)】：同构标定门限 vs 开放异构分布混叠分析——量化同构防御在良性开放异构下的高误报崩溃（BFQR）及 ROC-AUC 对比；
2. 【主对比实验 (Main Benchmark)】：8 种对比方法 × 5 类威胁场景（Clean / PGD-Removal / Ghost-Spoofing / HetPoison-Transfer / WhiteBox-Adaptive），
   汇报 AP@0.5, AP@0.7, 良性异构误杀率 BFQR, 恶意攻击压制率 MSR 与单帧推理延迟 (ms)；
3. 【严格同底座消融实验 (Ablation Study)】：逐项验证 LB-FPA 谱有界适配器、极坐标目标层光束可见度（对比 v1 原始网格点数 C1 漏洞）、
   双边偶然白化校准、平衡球面流形评分器、3x3 连通簇 MeanTopK(16)（对比 v1 Quantile_0.95 C3 漏洞）及有界余弦防稀释门的真实贡献；
4. 【客观物理真值框三层评估 & 自适应攻击权衡曲线】：按本车原始激光雷达落入真值框点数 N_pts^ego 客观划分为
   Clear (>=15 pts) / Sparse (1~14 pts) / Blind-Spot Occluded (0 ego pts & >=5 cav pts) 三层统计召回率与精确率。
"""

import os
import sys
import copy
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OPENCOOD_ROOT = os.path.join(PROJECT_ROOT, "third_party", "OpenCOOD")
for p in [PROJECT_ROOT, OPENCOOD_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from opencood.utils import eval_utils, box_utils

from src.open_hetero_env import (
    COLLAB_PROFILES,
    FrozenAnchorWrapper,
    load_opv2v_split_dataset,
    prepare_hetero_frame_bundle,
)
from src.hetero_sensors import count_points_in_3d_boxes, build_bev_grid_xy
from src.protocol_models import HeteroAgentBranch
from src.uncertainty_fusion import (
    BUDFusionModule,
    fuse_protocol_standard,
    fuse_where2comm_hetero,
    compute_global_residual_scores,
    compute_robosac_ccloss_scores,
    fuse_with_agent_rejection,
)
from src.attacks import (
    generate_pgd_removal_attack,
    generate_ghost_spoofing_attack,
    generate_hetpoison_raw_transfer_attack,
    generate_whitebox_adaptive_attack,
)


CFG_PATH = "/root/autodl-tmp/ToolV2X_resources/cad_native_2026_10_05/AdvCollaborativePerception/models/OpenCOOD/pointpillar_attentive_fusion/config.yaml"
CKPT_PATH = "/root/autodl-tmp/ToolV2X_resources/cad_native_2026_10_05/AdvCollaborativePerception/models/OpenCOOD/pointpillar_attentive_fusion/latest.pth"


def init_ap_stat():
    return {
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }


def init_stratum_stat():
    return {
        "clear": {"hit": 0, "total": 0},
        "sparse": {"hit": 0, "total": 0},
        "blind_occluded": {"hit": 0, "total": 0},
        "total_preds": 0,
        "false_positives": 0,
    }


def update_stratum_recall_fp(
    pred_box_t,
    gt_box_t,
    ego_pts_np: np.ndarray,
    cav_pts_list: list,
    stratum_stat: dict,
    iou_thresh: float = 0.5,
):
    """
    按与被测模型完全无关的客观真值框本车点数划分三层统计命中率与假阳性数（严格落实历史教训第 1 条）：
    - Clear: N_ego >= 15
    - Sparse: 1 <= N_ego <= 14
    - Blind-Spot Occluded: N_ego == 0 且 max(N_cav) >= 5
    """
    pred_np = pred_box_t.detach().cpu().numpy() if pred_box_t is not None and len(pred_box_t) > 0 else np.zeros((0, 8, 3))
    gt_np = gt_box_t.detach().cpu().numpy() if gt_box_t is not None and len(gt_box_t) > 0 else np.zeros((0, 8, 3))

    stratum_stat["total_preds"] += int(len(pred_np))
    if len(gt_np) == 0:
        stratum_stat["false_positives"] += int(len(pred_np))
        return

    ego_counts = count_points_in_3d_boxes(ego_pts_np, gt_np)
    max_cav_counts = np.zeros_like(ego_counts)
    for c_pts in cav_pts_list:
        c_cnt = count_points_in_3d_boxes(c_pts, gt_np)
        max_cav_counts = np.maximum(max_cav_counts, c_cnt)

    # 使用 BEV 中心距离 + 快速旋转框 IoU 判定真值命中
    gt_centers = gt_np.mean(axis=1)[:, :2]
    pred_centers = pred_np.mean(axis=1)[:, :2] if len(pred_np) > 0 else np.zeros((0, 2))

    gt_matched = np.zeros((len(gt_np),), dtype=bool)
    pred_matched = np.zeros((len(pred_np),), dtype=bool)

    if len(pred_np) > 0:
        from opencood.utils import common_utils
        pred_poly = common_utils.convert_format(pred_np)
        gt_poly = common_utils.convert_format(gt_np)
        for p_idx, p_pl in enumerate(pred_poly):
            best_iou = 0.0
            best_g = -1
            for g_idx, g_pl in enumerate(gt_poly):
                if gt_matched[g_idx]:
                    continue
                if np.linalg.norm(pred_centers[p_idx] - gt_centers[g_idx]) > 4.5:
                    continue
                inter = p_pl.intersection(g_pl).area
                if inter <= 0.0:
                    continue
                iou = inter / max(p_pl.area + g_pl.area - inter, 1e-6)
                if iou > best_iou:
                    best_iou = iou
                    best_g = g_idx
            if best_iou >= iou_thresh and best_g >= 0:
                gt_matched[best_g] = True
                pred_matched[p_idx] = True

    stratum_stat["false_positives"] += int(np.sum(~pred_matched))

    for g_idx in range(len(gt_np)):
        n_e = int(ego_counts[g_idx])
        n_c = int(max_cav_counts[g_idx])
        hit = int(gt_matched[g_idx])
        if n_e >= 15:
            stratum_stat["clear"]["total"] += 1
            stratum_stat["clear"]["hit"] += hit
        elif n_e >= 1:
            stratum_stat["sparse"]["total"] += 1
            stratum_stat["sparse"]["hit"] += hit
        elif n_e == 0 and n_c >= 5:
            stratum_stat["blind_occluded"]["total"] += 1
            stratum_stat["blind_occluded"]["hit"] += hit


def compute_raw_grid_count_uncertainty(pts_np: np.ndarray, device="cuda") -> torch.Tensor:
    """v1 初稿被审查指出的 C1 退化写法对照：直接数 0.8m x 0.8m 网格落点数（导致 >20m 空旷车道被误判为盲区）"""
    counts = torch.zeros((100, 352), device=device, dtype=torch.float32)
    if len(pts_np) > 0:
        pts_t = torch.from_numpy(pts_np[:, :2]).to(device=device, dtype=torch.float32)
        ix = torch.clamp(((pts_t[:, 0] + 140.8) / 0.8).long(), 0, 351)
        iy = torch.clamp(((pts_t[:, 1] + 40.0) / 0.8).long(), 0, 99)
        lin = iy * 352 + ix
        ones = torch.ones_like(lin, dtype=torch.float32)
        counts.view(-1).scatter_add_(0, lin, ones)
    rho = torch.clamp(torch.log1p(counts) / np.log(33.0), 0.0, 1.0)
    return (1.0 - rho).view(1, 1, 100, 352)


def load_branch_group(ckpt_dir: str, suffix: str) -> dict:
    bounded = suffix == "bounded"
    group = {}
    for prof in COLLAB_PROFILES:
        b = HeteroAgentBranch(profile_name=prof, bounded=bounded, lip_gamma=1.12).cuda()
        b.load_state_dict(torch.load(os.path.join(ckpt_dir, f"{prof}_{suffix}.pth"), map_location="cuda"), strict=True)
        b.eval()
        for p in b.parameters():
            p.requires_grad_(False)
        group[prof] = b
    return group


def finalize_ap_dict(stat):
    ap50 = float(eval_utils.calculate_ap(stat, 0.5, True)[0])
    ap70 = float(eval_utils.calculate_ap(stat, 0.7, True)[0])
    return {"AP50": round(ap50, 4), "AP70": round(ap70, 4)}


def finalize_stratum_dict(st):
    out = {}
    for k in ["clear", "sparse", "blind_occluded"]:
        tot = max(st[k]["total"], 1)
        out[f"{k}_recall"] = round(st[k]["hit"] / tot, 4)
        out[f"{k}_gt"] = st[k]["total"]
    tot_p = max(st["total_preds"], 1)
    out["precision_proxy"] = round(1.0 - st["false_positives"] / tot_p, 4)
    out["false_positives"] = st["false_positives"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_test_frames", type=int, default=240)
    args = parser.parse_args()

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

    branches_bounded = load_branch_group(ckpt_dir, "bounded")
    branches_unconst = load_branch_group(ckpt_dir, "unconstrained")
    branches_surr = load_branch_group(ckpt_dir, "surrogate")

    bud = BUDFusionModule(proto_channels=384).cuda()
    bud.load_state_dict(torch.load(os.path.join(ckpt_dir, "bud_fusion_duid.pth"), map_location="cuda"), strict=True)
    bud.calib_params = copy.deepcopy(calib_data["bud_calib_params"])
    bud.eval()
    for p in bud.parameters():
        p.requires_grad_(False)

    th_het = calib_data["hetero_calibrated_95"]
    th_homo = calib_data["homo_calibrated_95"]

    total_test = len(test_ds)
    stride = max(1, total_test // args.max_test_frames)
    test_indices = list(range(0, total_test, stride))[: args.max_test_frames]
    print(f"[Step 3] Evaluating on {len(test_indices)} unseen OPV2V test frames across all 16 sequences (total={total_test})...")

    scenarios = ["clean", "pgd_removal", "ghost_spoof", "hetpoison_transfer", "whitebox_adaptive"]
    main_methods = [
        "ego_only",
        "homo_64_oracle",
        "unconstrained_att",
        "hetero_where2comm",
        "homo_def_global_res_homo_th",
        "homo_def_robosac_homo_th",
        "hetero_def_global_res",
        "hetero_def_robosac_full",
        "hetero_def_robosac_vis",
        "ours_bud_fusion",
    ]
    ablation_methods = [
        "ours_full",
        "ablation_unconstrained_adapter",
        "ablation_raw_grid_count_c1",
        "ablation_no_aleatoric_calib",
        "ablation_no_spherical_epis",
        "ablation_quantile95_c3",
        "ablation_unbounded_dot_att",
    ]

    ap_records = {sc: {m: init_ap_stat() for m in main_methods} for sc in scenarios}
    ablation_ap = {sc: {m: init_ap_stat() for m in ablation_methods} for sc in ["clean", "pgd_removal", "ghost_spoof", "hetpoison_transfer"]}
    stratum_records = {
        sc: {m: init_stratum_stat() for m in ["ego_only", "unconstrained_att", "hetero_def_robosac_vis", "ours_bud_fusion"]}
        for sc in ["clean", "ghost_spoof", "pgd_removal"]
    }

    # 统计误杀率 (BFQR)、攻击压制率 (MSR)、分数分布（供绘制 KDE / ROC 图）与推理延迟
    quarantine_stats = {
        m: {"benign_total": 0, "benign_quarantined": 0, "atk_total": 0, "atk_suppressed": 0}
        for m in [
            "homo_def_global_res_homo_th",
            "homo_def_robosac_homo_th",
            "hetero_def_global_res",
            "hetero_def_robosac_full",
            "hetero_def_robosac_vis",
            "ours_bud_fusion",
        ]
    }
    motivation_distributions = {
        "robosac_full": {"benign_homo": [], "benign_hetero": [], "malicious_removal": [], "malicious_spoof": []},
        "global_res": {"benign_homo": [], "benign_hetero": [], "malicious_removal": [], "malicious_spoof": []},
        "ours_s_frame": {"benign_homo": [], "benign_hetero": [], "malicious_removal": [], "malicious_spoof": []},
    }
    latency_ms = {m: [] for m in ["unconstrained_att", "hetero_def_robosac_vis", "ours_bud_fusion"]}

    t_start = time.time()
    for loop_i, idx in enumerate(test_indices):
        bundle = prepare_hetero_frame_bundle(test_ds, idx, anchor_wrapper, include_teacher=True)
        cavs = bundle["cav_entries"]
        z_ego = bundle["z_ego"]
        s_ego = bundle["sigma_alea_ego"]
        batch_post = bundle["batch_for_post"]
        cav_pts_list = [c["pts_full64_np"] for c in cavs]

        # 1. 提取本帧干净特征（Bounded 分支、Unconstrained 分支与 64线同构 Oracle）
        with torch.no_grad():
            out_ego = anchor_wrapper.head_forward(z_ego)
            pb_ego, ps_ego, gb_ego = test_ds.post_process(batch_post, {"ego": out_ego})

            if len(cavs) == 0:
                for sc in scenarios:
                    for m in main_methods:
                        eval_utils.caluclate_tp_fp(pb_ego, ps_ego, gb_ego, ap_records[sc][m], 0.5)
                        eval_utils.caluclate_tp_fp(pb_ego, ps_ego, gb_ego, ap_records[sc][m], 0.7)
                continue

            z_b_list, raw_b_list = [z_ego], []
            z_u_list, raw_u_list = [z_ego], []
            z_homo_list = [z_ego]
            ug_list, dm_list, p_names = [], [], []

            for c in cavs:
                prof = c["profile"]
                zb, rb = branches_bounded[prof](c["pts_hetero_t"])
                zu, ru = branches_unconst[prof](c["pts_hetero_t"])
                z_b_list.append(zb)
                raw_b_list.append(rb)
                z_u_list.append(zu)
                raw_u_list.append(ru)
                z_homo_list.append(c["z_teacher"])
                ug_list.append(c["u_geom"].unsqueeze(0))
                dm_list.append(c["dist_map"].unsqueeze(0))
                p_names.append(prof)

            z_clean_b = torch.cat(z_b_list, dim=0)
            z_clean_u = torch.cat(z_u_list, dim=0)
            z_clean_homo = torch.cat(z_homo_list, dim=0)
            ug_cavs = torch.cat(ug_list, dim=0)
            dm_cavs = torch.cat(dm_list, dim=0)

            # 同构上界参考
            z_homo_fused = fuse_protocol_standard(z_clean_homo, cls_head_ref=cls_ref)
            out_homo = anchor_wrapper.head_forward(z_homo_fused)
            pb_homo, ps_homo, gb_homo = test_ds.post_process(batch_post, {"ego": out_homo})

            # 记录动机实验良性同构分数
            if loop_i < 100:
                motivation_distributions["robosac_full"]["benign_homo"].extend(
                    compute_robosac_ccloss_scores(z_clean_homo, cls_ref, reg_ref, s_ego, False).cpu().tolist()
                )
                motivation_distributions["global_res"]["benign_homo"].extend(
                    compute_global_residual_scores(z_clean_homo).cpu().tolist()
                )
                _, _, _, _, sf_hm, _, _ = bud.compute_verification_maps(
                    z_clean_homo, s_ego, ug_cavs, dm_cavs, ["profile_a_pp64"] * len(cavs), cls_head_ref=cls_ref
                )
                motivation_distributions["ours_s_frame"]["benign_homo"].extend(sf_hm.cpu().tolist())

        # 确定性轮换攻击者索引（均衡覆盖不同异构角色与不同位置车辆，杜绝固定攻击者捷径！）
        atk_idx = 1 + (loop_i % len(cavs))
        atk_prof = p_names[atk_idx - 1]
        s_ego_raw_c1 = compute_raw_grid_count_uncertainty(bundle["ego_pts_np"], device="cuda")

        # 2 & 3. 逐场景流式生成威胁张量并完成全部基线与消融推理（测完单场景立即释放显存，峰值显存 < 2.0 GiB！）
        for sc in scenarios:
            if sc == "clean":
                zb_sc, zu_sc = z_clean_b, z_clean_u
            elif sc == "pgd_removal":
                zb_sc = generate_pgd_removal_attack(z_clean_b, atk_idx, cls_ref, eps=1.35, steps=8)
                zu_sc = generate_pgd_removal_attack(z_clean_u, atk_idx, cls_ref, eps=1.35, steps=8)
            elif sc == "ghost_spoof":
                zb_sc = generate_ghost_spoofing_attack(z_clean_b, atk_idx, cls_ref, eps=1.45, steps=8, num_spoofs=2, frame_seed=idx)
                zu_sc = generate_ghost_spoofing_attack(z_clean_u, atk_idx, cls_ref, eps=1.45, steps=8, num_spoofs=2, frame_seed=idx)
            elif sc == "hetpoison_transfer":
                zb_sc = z_clean_b.clone()
                zu_sc = z_clean_u.clone()
                zb_sc[atk_idx : atk_idx + 1] = generate_hetpoison_raw_transfer_attack(
                    raw_b_list[atk_idx - 1], branches_surr[atk_prof].adapter, cls_ref, branches_bounded[atk_prof].adapter, raw_eps=1.4
                )
                zu_sc[atk_idx : atk_idx + 1] = generate_hetpoison_raw_transfer_attack(
                    raw_u_list[atk_idx - 1], branches_surr[atk_prof].adapter, cls_ref, branches_unconst[atk_prof].adapter, raw_eps=1.4
                )
            elif sc == "whitebox_adaptive":
                zb_sc = generate_whitebox_adaptive_attack(
                    z_clean_b, atk_idx, bud, s_ego, ug_cavs, dm_cavs, p_names, cls_ref,
                    eps=1.35, steps=10, stealth_lambda=1.5, frame_seed=idx
                )
                zu_sc = generate_pgd_removal_attack(z_clean_u, atk_idx, cls_ref, eps=1.35, steps=10)

            with torch.no_grad():

                # 固定参考
                eval_utils.caluclate_tp_fp(pb_ego, ps_ego, gb_ego, ap_records[sc]["ego_only"], 0.5)
                eval_utils.caluclate_tp_fp(pb_ego, ps_ego, gb_ego, ap_records[sc]["ego_only"], 0.7)
                eval_utils.caluclate_tp_fp(pb_homo, ps_homo, gb_homo, ap_records[sc]["homo_64_oracle"], 0.5)
                eval_utils.caluclate_tp_fp(pb_homo, ps_homo, gb_homo, ap_records[sc]["homo_64_oracle"], 0.7)

                # 基线：Unconstrained-AttFusion (HEAL/STAMP-style)
                torch.cuda.synchronize()
                t_m0 = time.perf_counter()
                zf_u_att = fuse_protocol_standard(zu_sc, cls_head_ref=cls_ref)
                out_u_att = anchor_wrapper.head_forward(zf_u_att)
                torch.cuda.synchronize()
                if sc == "clean":
                    latency_ms["unconstrained_att"].append((time.perf_counter() - t_m0) * 1000.0)
                pb_ua, ps_ua, gb_ua = test_ds.post_process(batch_post, {"ego": out_u_att})
                eval_utils.caluclate_tp_fp(pb_ua, ps_ua, gb_ua, ap_records[sc]["unconstrained_att"], 0.5)
                eval_utils.caluclate_tp_fp(pb_ua, ps_ua, gb_ua, ap_records[sc]["unconstrained_att"], 0.7)

                # 基线：Hetero-Where2comm
                zf_w2c = fuse_where2comm_hetero(zb_sc, cls_ref)
                pb_w2, ps_w2, gb_w2 = test_ds.post_process(batch_post, {"ego": anchor_wrapper.head_forward(zf_w2c)})
                eval_utils.caluclate_tp_fp(pb_w2, ps_w2, gb_w2, ap_records[sc]["hetero_where2comm"], 0.5)
                eval_utils.caluclate_tp_fp(pb_w2, ps_w2, gb_w2, ap_records[sc]["hetero_where2comm"], 0.7)

                # 基线分数计算：Global Residual / ROBOSAC-Full / ROBOSAC-VisCalib（在相同 LB-FPA 协议底座上保证绝对公平对照！）
                sc_gres = compute_global_residual_scores(zb_sc)
                sc_rfull = compute_robosac_ccloss_scores(zb_sc, cls_ref, reg_ref, s_ego, vis_calibrated=False)

                torch.cuda.synchronize()
                t_rb = time.perf_counter()
                sc_rvis = compute_robosac_ccloss_scores(zb_sc, cls_ref, reg_ref, s_ego, vis_calibrated=True)
                zf_rvis, rej_rvis = fuse_with_agent_rejection(zb_sc, sc_rvis, th_het["robosac_vis_calib"], cls_head_ref=cls_ref)
                out_rvis = anchor_wrapper.head_forward(zf_rvis)
                torch.cuda.synchronize()
                if sc == "clean":
                    latency_ms["hetero_def_robosac_vis"].append((time.perf_counter() - t_rb) * 1000.0)

                zf_gres_homo, rej_gres_hm = fuse_with_agent_rejection(zb_sc, sc_gres, th_homo["global_residual"], cls_head_ref=cls_ref)
                zf_rfull_homo, rej_rf_hm = fuse_with_agent_rejection(zb_sc, sc_rfull, th_homo["robosac_full"], cls_head_ref=cls_ref)
                zf_gres_het, rej_gres_ht = fuse_with_agent_rejection(zb_sc, sc_gres, th_het["global_residual"], cls_head_ref=cls_ref)
                zf_rfull_het, rej_rf_ht = fuse_with_agent_rejection(zb_sc, sc_rfull, th_het["robosac_full"], cls_head_ref=cls_ref)

                # 同一场景下若不同硬过滤防御保留了完全一致的协作者子集，直接复用旋转框 NMS 后处理结果（100% 数值恒等无损加速）
                subset_post_cache = {tuple([True] * len(cavs)): (pb_ego, ps_ego, gb_ego)}
                pb_rv_key = tuple(rej_rvis.cpu().tolist())
                if pb_rv_key not in subset_post_cache:
                    subset_post_cache[pb_rv_key] = test_ds.post_process(batch_post, {"ego": out_rvis})
                pb_rv, ps_rv, gb_rv = subset_post_cache[pb_rv_key]

                for m_key, z_f, rej_t in [
                    ("homo_def_global_res_homo_th", zf_gres_homo, rej_gres_hm),
                    ("homo_def_robosac_homo_th", zf_rfull_homo, rej_rf_hm),
                    ("hetero_def_global_res", zf_gres_het, rej_gres_ht),
                    ("hetero_def_robosac_full", zf_rfull_het, rej_rf_ht),
                ]:
                    r_key = tuple(rej_t.cpu().tolist())
                    if r_key not in subset_post_cache:
                        subset_post_cache[r_key] = test_ds.post_process(batch_post, {"ego": anchor_wrapper.head_forward(z_f)})
                    pb_t, ps_t, gb_t = subset_post_cache[r_key]
                    eval_utils.caluclate_tp_fp(pb_t, ps_t, gb_t, ap_records[sc][m_key], 0.5)
                    eval_utils.caluclate_tp_fp(pb_t, ps_t, gb_t, ap_records[sc][m_key], 0.7)

                eval_utils.caluclate_tp_fp(pb_rv, ps_rv, gb_rv, ap_records[sc]["hetero_def_robosac_vis"], 0.5)
                eval_utils.caluclate_tp_fp(pb_rv, ps_rv, gb_rv, ap_records[sc]["hetero_def_robosac_vis"], 0.7)

                # 本文方法：Ours BUD-Fusion
                torch.cuda.synchronize()
                t_ours = time.perf_counter()
                zf_ours, diag_ours = bud.forward_fuse(
                    zb_sc, s_ego, ug_cavs, dm_cavs, p_names, cls_ref,
                    use_aleatoric_calib=True, use_epistemic_scorer=True, use_topk_cluster=True, use_bounded_cosine=True
                )
                out_ours = anchor_wrapper.head_forward(zf_ours)
                torch.cuda.synchronize()
                if sc == "clean":
                    latency_ms["ours_bud_fusion"].append((time.perf_counter() - t_ours) * 1000.0)

                pb_ou, ps_ou, gb_ou = test_ds.post_process(batch_post, {"ego": out_ours})
                eval_utils.caluclate_tp_fp(pb_ou, ps_ou, gb_ou, ap_records[sc]["ours_bud_fusion"], 0.5)
                eval_utils.caluclate_tp_fp(pb_ou, ps_ou, gb_ou, ap_records[sc]["ours_bud_fusion"], 0.7)

                # 记录动机分布、良性异构误杀率 BFQR 与攻击压制率 MSR
                rej_ours = diag_ours["eta"] < 0.5  # 等价于 s_frame > tau_frame
                rej_map = {
                    "homo_def_global_res_homo_th": rej_gres_hm,
                    "homo_def_robosac_homo_th": rej_rf_hm,
                    "hetero_def_global_res": rej_gres_ht,
                    "hetero_def_robosac_full": rej_rf_ht,
                    "hetero_def_robosac_vis": rej_rvis,
                    "ours_bud_fusion": rej_ours,
                }
                if sc == "clean":
                    for mk, r_tensor in rej_map.items():
                        quarantine_stats[mk]["benign_total"] += int(r_tensor.numel())
                        quarantine_stats[mk]["benign_quarantined"] += int(r_tensor.sum().item())
                    if loop_i < 120:
                        motivation_distributions["robosac_full"]["benign_hetero"].extend(sc_rfull.cpu().tolist())
                        motivation_distributions["global_res"]["benign_hetero"].extend(sc_gres.cpu().tolist())
                        motivation_distributions["ours_s_frame"]["benign_hetero"].extend(diag_ours["s_frame"].cpu().tolist())
                else:
                    sub_a = atk_idx - 1
                    for mk, r_tensor in rej_map.items():
                        quarantine_stats[mk]["atk_total"] += 1
                        quarantine_stats[mk]["atk_suppressed"] += int(bool(r_tensor[sub_a].item()))
                    if sc == "pgd_removal" and loop_i < 120:
                        motivation_distributions["robosac_full"]["malicious_removal"].append(float(sc_rfull[sub_a].item()))
                        motivation_distributions["global_res"]["malicious_removal"].append(float(sc_gres[sub_a].item()))
                        motivation_distributions["ours_s_frame"]["malicious_removal"].append(float(diag_ours["s_frame"][sub_a].item()))
                    elif sc == "ghost_spoof" and loop_i < 120:
                        motivation_distributions["robosac_full"]["malicious_spoof"].append(float(sc_rfull[sub_a].item()))
                        motivation_distributions["global_res"]["malicious_spoof"].append(float(sc_gres[sub_a].item()))
                        motivation_distributions["ours_s_frame"]["malicious_spoof"].append(float(diag_ours["s_frame"][sub_a].item()))

                # 记录客观物理真值框三层命中统计
                if sc in stratum_records:
                    update_stratum_recall_fp(pb_ego, gb_ego, bundle["ego_pts_np"], cav_pts_list, stratum_records[sc]["ego_only"])
                    update_stratum_recall_fp(pb_ua, gb_ua, bundle["ego_pts_np"], cav_pts_list, stratum_records[sc]["unconstrained_att"])
                    update_stratum_recall_fp(pb_rv, gb_rv, bundle["ego_pts_np"], cav_pts_list, stratum_records[sc]["hetero_def_robosac_vis"])
                    update_stratum_recall_fp(pb_ou, gb_ou, bundle["ego_pts_np"], cav_pts_list, stratum_records[sc]["ours_bud_fusion"])

                # 4. 运行逐项消融实验
                if sc in ablation_ap:
                    eval_utils.caluclate_tp_fp(pb_ou, ps_ou, gb_ou, ablation_ap[sc]["ours_full"], 0.5)
                    eval_utils.caluclate_tp_fp(pb_ou, ps_ou, gb_ou, ablation_ap[sc]["ours_full"], 0.7)

                    # Ablation 1: Unconstrained Adapter + BUD Fusion (检验 LB-FPA 独立贡献)
                    zf_ab1, _ = bud.forward_fuse(zu_sc, s_ego, ug_cavs, dm_cavs, p_names, cls_ref)
                    pb_1, ps_1, gb_1 = test_ds.post_process(batch_post, {"ego": anchor_wrapper.head_forward(zf_ab1)})
                    eval_utils.caluclate_tp_fp(pb_1, ps_1, gb_1, ablation_ap[sc]["ablation_unconstrained_adapter"], 0.5)
                    eval_utils.caluclate_tp_fp(pb_1, ps_1, gb_1, ablation_ap[sc]["ablation_unconstrained_adapter"], 0.7)

                    # Ablation 2: Raw Grid Point Count u_alea (检验修复 C1 空旷路面盲区塌陷的作用)
                    zf_ab2, _ = bud.forward_fuse(zb_sc, s_ego_raw_c1, ug_cavs, dm_cavs, p_names, cls_ref)
                    pb_2, ps_2, gb_2 = test_ds.post_process(batch_post, {"ego": anchor_wrapper.head_forward(zf_ab2)})
                    eval_utils.caluclate_tp_fp(pb_2, ps_2, gb_2, ablation_ap[sc]["ablation_raw_grid_count_c1"], 0.5)
                    eval_utils.caluclate_tp_fp(pb_2, ps_2, gb_2, ablation_ap[sc]["ablation_raw_grid_count_c1"], 0.7)

                    # Ablation 3: w/o Aleatoric Calibration (使用对应验证集标定的 ours_no_alea 门限)
                    orig_tf = bud.calib_params["tau_frame"]
                    bud.calib_params["tau_frame"] = th_het["ours_no_alea"]
                    zf_ab3, _ = bud.forward_fuse(zb_sc, s_ego, ug_cavs, dm_cavs, p_names, cls_ref, use_aleatoric_calib=False)
                    bud.calib_params["tau_frame"] = orig_tf
                    pb_3, ps_3, gb_3 = test_ds.post_process(batch_post, {"ego": anchor_wrapper.head_forward(zf_ab3)})
                    eval_utils.caluclate_tp_fp(pb_3, ps_3, gb_3, ablation_ap[sc]["ablation_no_aleatoric_calib"], 0.5)
                    eval_utils.caluclate_tp_fp(pb_3, ps_3, gb_3, ablation_ap[sc]["ablation_no_aleatoric_calib"], 0.7)

                    # Ablation 4: w/o Spherical Epistemic Scorer
                    bud.calib_params["tau_frame"] = th_het["ours_no_epis"]
                    zf_ab4, _ = bud.forward_fuse(zb_sc, s_ego, ug_cavs, dm_cavs, p_names, cls_ref, use_epistemic_scorer=False)
                    bud.calib_params["tau_frame"] = orig_tf
                    pb_4, ps_4, gb_4 = test_ds.post_process(batch_post, {"ego": anchor_wrapper.head_forward(zf_ab4)})
                    eval_utils.caluclate_tp_fp(pb_4, ps_4, gb_4, ablation_ap[sc]["ablation_no_spherical_epis"], 0.5)
                    eval_utils.caluclate_tp_fp(pb_4, ps_4, gb_4, ablation_ap[sc]["ablation_no_spherical_epis"], 0.7)

                    # Ablation 5: Quantile_0.95 instead of Top-16 Cluster Pooling (检验修复 C3 局部攻击漏检的作用)
                    bud.calib_params["tau_frame"] = th_het["ours_quant95"]
                    zf_ab5, _ = bud.forward_fuse(zb_sc, s_ego, ug_cavs, dm_cavs, p_names, cls_ref, use_topk_cluster=False)
                    bud.calib_params["tau_frame"] = orig_tf
                    pb_5, ps_5, gb_5 = test_ds.post_process(batch_post, {"ego": anchor_wrapper.head_forward(zf_ab5)})
                    eval_utils.caluclate_tp_fp(pb_5, ps_5, gb_5, ablation_ap[sc]["ablation_quantile95_c3"], 0.5)
                    eval_utils.caluclate_tp_fp(pb_5, ps_5, gb_5, ablation_ap[sc]["ablation_quantile95_c3"], 0.7)

                    # Ablation 6: Unbounded Dot-Product Attention
                    zf_ab6, _ = bud.forward_fuse(zb_sc, s_ego, ug_cavs, dm_cavs, p_names, cls_ref, use_bounded_cosine=False)
                    pb_6, ps_6, gb_6 = test_ds.post_process(batch_post, {"ego": anchor_wrapper.head_forward(zf_ab6)})
                    eval_utils.caluclate_tp_fp(pb_6, ps_6, gb_6, ablation_ap[sc]["ablation_unbounded_dot_att"], 0.5)
                    eval_utils.caluclate_tp_fp(pb_6, ps_6, gb_6, ablation_ap[sc]["ablation_unbounded_dot_att"], 0.7)

            if sc != "clean":
                del zb_sc, zu_sc

        del z_clean_b, z_clean_u, z_clean_homo, raw_b_list, raw_u_list, bundle
        if (loop_i + 1) % 3 == 0:
            torch.cuda.empty_cache()

        if (loop_i + 1) % 20 == 0 or (loop_i + 1) == len(test_indices):
            print(f"  Evaluated {loop_i + 1}/{len(test_indices)} test frames ({time.time() - t_start:.1f}s)")

    # 汇总计算全部评价指标
    main_summary = {sc: {m: finalize_ap_dict(ap_records[sc][m]) for m in main_methods} for sc in scenarios}
    ablation_summary = {sc: {m: finalize_ap_dict(ablation_ap[sc][m]) for m in ablation_methods} for sc in ablation_ap}
    stratum_summary = {sc: {m: finalize_stratum_dict(stratum_records[sc][m]) for sc in stratum_records for m in stratum_records[sc]} for sc in stratum_records}
    # 修正 stratum_summary 结构按 sc 分组
    stratum_summary = {
        sc: {m: finalize_stratum_dict(stratum_records[sc][m]) for m in stratum_records[sc]}
        for sc in stratum_records
    }

    bfqr_msr_summary = {}
    for mk, qv in quarantine_stats.items():
        bfqr = qv["benign_quarantined"] / max(qv["benign_total"], 1)
        msr = qv["atk_suppressed"] / max(qv["atk_total"], 1)
        bfqr_msr_summary[mk] = {
            "BFQR_benign_false_quarantine_rate": round(bfqr, 4),
            "MSR_malicious_suppression_rate": round(msr, 4),
            "benign_N": qv["benign_total"],
            "atk_N": qv["atk_total"],
        }

    auroc_summary = {}
    for det_name, dist_d in motivation_distributions.items():
        b_het = dist_d["benign_hetero"]
        m_all = dist_d["malicious_removal"] + dist_d["malicious_spoof"]
        if b_het and m_all:
            y_true = [0] * len(b_het) + [1] * len(m_all)
            y_score = b_het + m_all
            auroc_summary[det_name] = round(float(roc_auc_score(y_true, y_score)), 4)

    latency_summary = {k: round(float(np.mean(v[2:])), 2) if len(v) > 2 else 0.0 for k, v in latency_ms.items()}

    final_output = {
        "num_test_frames": len(test_indices),
        "main_benchmark_ap": main_summary,
        "quarantine_and_suppression": bfqr_msr_summary,
        "detector_hetero_auroc": auroc_summary,
        "ablation_study_ap": ablation_summary,
        "objective_point_strata": stratum_summary,
        "mean_fusion_head_latency_ms": latency_summary,
    }

    out_json = os.path.join(PROJECT_ROOT, "outputs", "step3_full_evaluation_results.json")
    dist_json = os.path.join(PROJECT_ROOT, "outputs", "step3_score_distributions.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    with open(dist_json, "w", encoding="utf-8") as f:
        json.dump(motivation_distributions, f, indent=2, ensure_ascii=False)

    print("\n================ FINAL BENCHMARK SUMMARY ================")
    print(json.dumps(final_output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
