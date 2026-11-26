# -*- coding: utf-8 -*-
"""
步骤 ⑤-3（核心立足点实证实验）：异构车队中统一判据的跨模态误报偏差与检出率权衡验证。

直接实证两大核心主张（严格仅在 Dev-320 8 条未见序列上评估，绝不触碰 opv2v_test）：
1. 【实验 A · 现有防御在异构车队统一判据下的跨模态失衡】：
   在干净 Dev-320 上，分别统计诚实 m1 (LiDAR) 与诚实 m2 (Camera) 协作车在以下防御中的分数分布与误报率：
   - HetShield (修复版 Shared 统一门限：原论文 τ=0.3、Fit-Tuned、等 5% 总 FPR)
   - LUCIA (官方原味：2 车与混合多车队下的 L1 距离与信任得分分布、低信任告警率)
   - ROBOSAC (官方原味，仅 >=3 车序列：all_m1 vs all_m2 vs mixed 的 Jaccard 分布与误拒率)
   - Tier ① 统一门限基线 (全模态共享标尺，不按传感器/距离/可见性分档)
2. 【实验 B · 等总误报率下：统一判据 vs 我们的按条件判据 (Tier ④)】：
   - 在相同总误报率 (2% / 5% / 10%) 及「压低 m2 误报至 <=5%」口径下，对比：
     (a) 诚实 m1、m2 各自误报率（含成对差值 2,000 次 Block-Bootstrap 95% CI）；
     (b) 攻击者分别为 m1、m2 时在 PGD 1.0σ、局部擦除 1.0σ、局部鬼影 2.0σ 下的检出率 TPR 与 AUROC
         （含成对差值 Block-Bootstrap 95% CI）。
"""

import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJ_ROOT = "/root/autodl-tmp/proj_dir2_hetero_trust"
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from src_v2_mm.baselines_robosac_lucia_heal import (
    ROBOSACHEALDefender,
    compute_box_set_jaccard_2d,
    cp_defense_lucia_heal,
)
from src_v2_mm.bud_fusion_heal import BUDFusionHEALAuditor
from src_v2_mm.eval_stats_utils import (
    build_seq_bootstrap_indices,
    paired_block_bootstrap_rate_diff,
)
from src_v2_mm.heal_attacks import (
    generate_local_ghost_spoofing_heal,
    generate_local_removal_heal,
    generate_scaled_pgd_removal_heal,
)
from src_v2_mm.heal_base_wrapper import HEALBaseWrapper
from src_v2_mm.hetshield_heal import HetShieldHEALDefender
from src_v2_mm.step4_1_v2_core import (
    PROTO_H,
    PROTO_W,
    TRAIN_FULL19_DIR,
    assign_dist5_vis3_indices_v2,
    build_bev_grid_xy_256,
    compute_auroc,
    compute_dev_maps_from_protocol,
    compute_ego_expected_pts_and_los_256,
    compute_raw_frame_score_from_dev,
    extract_frame_context_and_clean,
    find_exact_fpr_threshold,
    load_ego_lidar_np,
)

DEFAULT_EVAL_SPLIT = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_3_eval320_split.json")
DEFAULT_BASE_CALIB = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_1_v2_frozen_calibrator.json")
DEFAULT_FROZEN_GATING = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_4_bud_frozen_calibrator.json")
DEFAULT_HS_CALIB = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step5_1_hetshield_heal_calib.json")
DEFAULT_CACHE_DIR = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step5_3_seq_cache")
DEFAULT_OUT_JSON = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step5_3_hetero_unified_threshold_evidence.json")
DEFAULT_OUT_FIG = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "fig_step5_3_hetero_unified_vs_cond.png")

ATK_LIST = ["pgd_1.0sigma", "local_removal_1.0sigma", "local_ghost_2.0sigma"]
BOOT_N = 2000
BOOT_SEED = 20261120


def _atomic_write_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _dist_stats(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 5),
        "std": round(float(arr.std()), 5),
        "median": round(float(np.median(arr)), 5),
        "p90": round(float(np.percentile(arr, 90.0)), 5),
        "p95": round(float(np.percentile(arr, 95.0)), 5),
        "max": round(float(arr.max()), 5),
    }


def _prime_hs_prev_frame(hs_def, wrapper, ds, sample_seqs, seq_name: str, idx: int, mod: str):
    """对单帧 idx 仅预热一次真实紧邻前一帧 idx-1 隐状态，供 clean 与全部攻击态复用（消除 4x 重复读盘编码）。"""
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
    from opencood.tools import train_utils

    hs_def.reset_sequence(seq_name)
    ac = wrapper.model.pyramid_backbone.align_corners

    if idx > 0 and sample_seqs[idx - 1] == seq_name:
        raw_prev = train_utils.to_device(ds.collate_batch_test([ds[idx - 1]]), wrapper.device_obj)
        p_prev = wrapper.extract_protocol_features(raw_prev["ego"])
        t_prev = p_prev["affine_matrix"][0, 0, 1:2]
        fc_p = warp_affine_simple(p_prev["heter_feature_2d"][1:2], t_prev, (PROTO_H, PROTO_W), align_corners=ac)
        vc_p = warp_affine_simple(p_prev["cam_valid_mask"][1:2].float(), t_prev, (PROTO_H, PROTO_W), align_corners=ac)
        hs_def.step_frame(
            seq_name, idx - 1, p_prev["heter_feature_2d"][0:1],
            [{"cav_id": f"{seq_name}_{mod}_c1", "mod": mod, "feat_warped": fc_p, "valid_ego": (vc_p > 0.5).float()}],
            update_state=True,
        )
        del raw_prev, p_prev


def _score_hs_cur_frame(hs_def, wrapper, seq_name: str, idx: int, feat_cur: torch.Tensor, p_cur: dict, mod: str) -> dict:
    """复用已预热的 idx-1 隐状态，以 update_state=False 对当前状态打分（与逐状态重跑比特级等价）。"""
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

    ac = wrapper.model.pyramid_backbone.align_corners
    t_cur = p_cur["affine_matrix"][0, 0, 1:2]
    fc = warp_affine_simple(feat_cur[1:2], t_cur, (PROTO_H, PROTO_W), align_corners=ac)
    vc = warp_affine_simple(p_cur["cam_valid_mask"][1:2].float(), t_cur, (PROTO_H, PROTO_W), align_corners=ac)
    return hs_def.step_frame(
        seq_name, idx, feat_cur[0:1],
        [{"cav_id": f"{seq_name}_{mod}_c1", "mod": mod, "feat_warped": fc, "valid_ego": (vc > 0.5).float()}],
        update_state=False,
    )[0]


def _compute_lucia_pair_score(wrapper, feat_cur: torch.Tensor, p_cur: dict) -> dict:
    """在 2 车成对样本上运行官方 LUCIA，返回协作者 L1 距离与 trust 得分。"""
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

    ac = wrapper.model.pyramid_backbone.align_corners
    t_mat = p_cur["affine_matrix"][0, 0, :2]
    aligned = warp_affine_simple(feat_cur[:2], t_mat, (PROTO_H, PROTO_W), align_corners=ac)
    trust, l1_sc = cp_defense_lucia_heal(aligned, compression_ratio=32)
    return {
        "l1_dist": float(l1_sc[1].item()),
        "trust": float(trust[1].item()),
        "anom_score": float(1.0 - trust[1].item()),
    }


def _eval_threshold_regime(
    seq_order: list,
    boot_idx: np.ndarray,
    scores_by_seq: dict,
    thresh_m1: float,
    thresh_m2: float,
) -> dict:
    """给定 m1/m2 门限（统一判据时两者相等），按父序列 Block-Bootstrap 输出 m1/m2 误报率与各攻击检出率。"""
    out = OrderedDict()
    # 干净误报率（m1、m2 及 m2 - m1 成对差值 CI）
    fp_m1 = {s: sum(1 for v in scores_by_seq[s]["m1"]["clean"] if v > thresh_m1) for s in seq_order}
    fp_m2 = {s: sum(1 for v in scores_by_seq[s]["m2"]["clean"] if v > thresh_m2) for s in seq_order}
    tot_c = {s: len(scores_by_seq[s]["m1"]["clean"]) for s in seq_order}
    fpr_diff = paired_block_bootstrap_rate_diff(seq_order, fp_m2, fp_m1, tot_c, boot_indices=boot_idx)
    tot_all = sum(tot_c.values()) * 2
    out["clean_fpr_overall"] = round((sum(fp_m1.values()) + sum(fp_m2.values())) / max(tot_all, 1), 5)
    out["clean_fpr_m1"] = {"val": fpr_diff["val_b"], "ci95": fpr_diff["b_ci95"]}
    out["clean_fpr_m2"] = {"val": fpr_diff["val_a"], "ci95": fpr_diff["a_ci95"]}
    out["clean_fpr_diff_m2_minus_m1"] = {
        "diff": fpr_diff["diff"],
        "diff_ci95": fpr_diff["diff_ci95"],
        "significant_pos": fpr_diff["significant_pos"],
    }

    # 各攻击下，攻击者为 m1 与攻击者为 m2 时的 TPR 与 AUROC
    out["attack_detection"] = OrderedDict()
    for atk in ATK_LIST:
        tp_m1 = {s: sum(1 for v in scores_by_seq[s]["m1"][atk] if v > thresh_m1) for s in seq_order}
        tp_m2 = {s: sum(1 for v in scores_by_seq[s]["m2"][atk] if v > thresh_m2) for s in seq_order}
        d_tpr = paired_block_bootstrap_rate_diff(seq_order, tp_m1, tp_m2, tot_c, boot_indices=boot_idx)
        neg_1 = np.asarray([v for s in seq_order for v in scores_by_seq[s]["m1"]["clean"]], dtype=np.float64)
        pos_1 = np.asarray([v for s in seq_order for v in scores_by_seq[s]["m1"][atk]], dtype=np.float64)
        neg_2 = np.asarray([v for s in seq_order for v in scores_by_seq[s]["m2"]["clean"]], dtype=np.float64)
        pos_2 = np.asarray([v for s in seq_order for v in scores_by_seq[s]["m2"][atk]], dtype=np.float64)
        out["attack_detection"][atk] = {
            "attacker_m1_tpr": {"val": d_tpr["val_a"], "ci95": d_tpr["a_ci95"]},
            "attacker_m2_tpr": {"val": d_tpr["val_b"], "ci95": d_tpr["b_ci95"]},
            "attacker_m1_auroc": round(compute_auroc(neg_1, pos_1), 5),
            "attacker_m2_auroc": round(compute_auroc(neg_2, pos_2), 5),
            "overall_auroc": round(compute_auroc(np.concatenate([neg_1, neg_2]), np.concatenate([pos_1, pos_2])), 5),
        }
    return out


def _plot_motivation_figure(report: dict, out_fig: str):
    """绘制论文 Motivation 对比图：(左) 等 5% 总 FPR 下各方法诚实 m1 vs m2 误报率；(右) 统一判据 vs 本文按条件判据对 m1/m2 攻击者检出率。"""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), dpi=300)

    # 左图：等 5% 总 FPR 下的诚实 m1 vs m2 误报率失衡对比
    ax = axes[0]
    m_keys = [
        ("tier1_unified", "Tier1\n(Unified)"),
        ("hetshield_shared", "HetShield\n(Unified)"),
        ("lucia", "LUCIA\n(Unified)"),
        ("tier4_ours", "BUD-Fusion\n(Ours, Cond.)"),
    ]
    labels, m1_v, m2_v = [], [], []
    for k, lb in m_keys:
        reg = report["experiment_b_equal_fpr"]["fpr_5pct"][k]
        labels.append(lb)
        m1_v.append(reg["clean_fpr_m1"]["val"] * 100.0)
        m2_v.append(reg["clean_fpr_m2"]["val"] * 100.0)

    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w / 2, m1_v, w, color="#2B6CB0", label="Honest m1 (LiDAR)")
    b2 = ax.bar(x + w / 2, m2_v, w, color="#DD6B20", label="Honest m2 (Camera)")
    ax.axhline(5.0, color="#718096", linestyle="--", linewidth=1.2, label="Target Overall FPR (5%)")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.25, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Clean False Positive Rate (%)")
    ax.set_title("(a) Per-Sensor FPR at Equal 5% Overall FPR", fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, max(12.0, max(m1_v + m2_v) * 1.28))
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # 右图：压低 m2 误报 <= 5% 口径下，平均攻击检出率（m1 攻击者 vs m2 攻击者）
    ax2 = axes[1]
    reg_m2cap = report["experiment_b_equal_fpr"]["m2_fpr_capped_5pct"]
    m_keys2 = [
        ("tier1_unified", "Tier1\n(Unified)"),
        ("hetshield_shared", "HetShield\n(Unified)"),
        ("tier4_ours", "BUD-Fusion\n(Ours, Cond.)"),
    ]
    lb2, atk1_v, atk2_v = [], [], []
    for k, lb in m_keys2:
        ad = reg_m2cap[k]["attack_detection"]
        lb2.append(lb)
        atk1_v.append(float(np.mean([ad[a]["attacker_m1_tpr"]["val"] for a in ATK_LIST])) * 100.0)
        atk2_v.append(float(np.mean([ad[a]["attacker_m2_tpr"]["val"] for a in ATK_LIST])) * 100.0)
    x2 = np.arange(len(lb2))
    b3 = ax2.bar(x2 - w / 2, atk1_v, w, color="#2F855A", label="Attacker = m1 (LiDAR)")
    b4 = ax2.bar(x2 + w / 2, atk2_v, w, color="#C53030", label="Attacker = m2 (Camera)")
    for bars in (b3, b4):
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(lb2, fontsize=9)
    ax2.set_ylabel("Mean Attack Detection TPR (%)")
    ax2.set_title("(b) Attack TPR when Honest m2 FPR <= 5%", fontsize=10.5, fontweight="bold")
    ax2.set_ylim(0, 112.0)
    ax2.legend(fontsize=8.5, loc="lower right")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Why Unified Thresholds Fail in Heterogeneous LiDAR-Camera Fleets (Dev-320)", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="步骤 ⑤-3：异构车队统一判据跨模态误报与检出权衡实证实验")
    parser.add_argument("--data_dir", type=str, default=TRAIN_FULL19_DIR)
    parser.add_argument("--eval_split_json", type=str, default=DEFAULT_EVAL_SPLIT)
    parser.add_argument("--base_calib_json", type=str, default=DEFAULT_BASE_CALIB)
    parser.add_argument("--frozen_gating_json", type=str, default=DEFAULT_FROZEN_GATING)
    parser.add_argument("--hetshield_calib_json", type=str, default=DEFAULT_HS_CALIB)
    parser.add_argument("--seqs", nargs="+", default=None, help="可选：仅运行指定父序列子集（多进程并行分片加速用）")
    parser.add_argument("--worker_only", action="store_true", help="仅写序列缓存 JSON，不聚合最终总报告")
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out_json", type=str, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_fig", type=str, default=DEFAULT_OUT_FIG)
    args = parser.parse_args()

    t0 = time.time()
    with open(args.eval_split_json, "r", encoding="utf-8") as f:
        eval_split = json.load(f)
    with open(args.hetshield_calib_json, "r", encoding="utf-8") as f:
        hs_cfg = json.load(f)

    seq_order = list(eval_split["seq_order"])
    target_seqs = [s for s in seq_order if (args.seqs is None or s in set(args.seqs))]
    os.makedirs(args.cache_dir, exist_ok=True)

    methods_4 = ["tier1_unified", "tier4_ours", "hetshield_shared", "lucia"]
    states_4 = ["clean"] + ATK_LIST

    all_cached = (not args.worker_only) and all(
        os.path.exists(os.path.join(args.cache_dir, f"{s}.json")) for s in seq_order
    )

    wrapper = ds_m1 = ds_m2 = sample_seqs = gx = gy = auditor = vis_v2 = hs_shared = None
    ds_fleets = {}
    if not all_cached:
        wrapper = HEALBaseWrapper(device="cuda", use_cav=7)
        ds_m1 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m1")
        ds_m2 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m2")
        sample_seqs = wrapper.get_sample_sequences(ds_m2)

        gx, gy = build_bev_grid_xy_256(device="cuda", dtype=torch.float32)
        auditor = BUDFusionHEALAuditor(args.base_calib_json, gating_calib_path=args.frozen_gating_json)
        vis_v2 = (auditor.gating_calib or {}).get("visibility_v2", None)

        hs_shared = HetShieldHEALDefender(
            mode="shared", alpha=0.5, tau_th=0.3, temperature=0.1,
            energy_thresh=float(hs_cfg["meta"]["energy_thresh"]),
        ).cuda().eval()
        hs_shared.predictor.load_state_dict(torch.load(hs_cfg["meta"]["shared_ckpt"], map_location="cpu"))

        for fm, cm in (("all_m1", "m1"), ("all_m2", "m2"), ("mixed", "mixed")):
            ds_fleets[fm] = wrapper.build_dataset(
                data_dir=args.data_dir, use_cav=7, visualize=True, collab_modality=cm, max_cav=7
            )

    seq_caches = {}
    for seq_name in (target_seqs if args.worker_only else seq_order):
        cpath = os.path.join(args.cache_dir, f"{seq_name}.json")
        if os.path.exists(cpath):
            print(f"[断点复用] Step 5-3 | {seq_name}")
            with open(cpath, "r", encoding="utf-8") as f:
                seq_caches[seq_name] = json.load(f)
            continue

        idxs = eval_split["sequences"][seq_name]["indices"]
        print(f"[开始评测] Step 5-3 | {seq_name} ({len(idxs)} 帧)", flush=True)
        sc_seq = {mk: {m: {st: [] for st in states_4} for m in ("m1", "m2")} for mk in methods_4}
        luc_2c = {"m1": {"l1": [], "trust": []}, "m2": {"l1": [], "trust": []}}
        hs_cc = {"m1": {"tau_k": [], "sigma_k": [], "s_k": []}, "m2": {"tau_k": [], "sigma_k": [], "s_k": []}}
        robo_seq = {fm: {"jaccard": [], "rej_03": 0, "rej_05": 0, "total": 0} for fm in ("all_m1", "all_m2", "mixed")}
        luc_mix = {"m1": {"trust": [], "l1": []}, "m2": {"trust": [], "l1": []}}

        # Part 1: 2 车成对 (clean + 3 攻击 × m1/m2)
        for fi, idx in enumerate(idxs):
            c_tau = float(vis_v2["clear_pts_tau"]) if vis_v2 is not None else 73.853
            l_tau = float(vis_v2.get("los_occ_tau", 0.50)) if vis_v2 is not None else 0.50
            b = extract_frame_context_and_clean(wrapper, ds_m1, ds_m2, idx, gx, gy, clear_pts_tau=c_tau, los_occ_tau=l_tau)
            # 动机实验（表 2 跨模态校准对比）：显式保留共同视野 cov_ego 与 dev_clean_common，确保差异纯来自传感器本身
            cov_2d = b["cov_ego"][0, 0]
            d_idx = b["d_idx"]
            v_idx = b["v_idx"]

            for mod, p_dict, ds_ref in (("m1", b["p1"], ds_m1), ("m2", b["p2"], ds_m2)):
                feat = p_dict["heter_feature_2d"]
                rlen = p_dict["record_len"]
                amat = p_dict["affine_matrix"]
                mods_l = list(p_dict["agent_modality_list"])
                vmask = p_dict["cam_valid_mask"]
                preg = b["cov_local_m1"] if mod == "m1" else None

                f_p1, _ = generate_scaled_pgd_removal_heal(wrapper, feat, rlen, amat, mods_l, vmask, attacker_idx=1, sigma_mult=1.0, steps=8, perturb_region_mask=preg)
                f_lr, _ = generate_local_removal_heal(wrapper, feat, rlen, amat, mods_l, vmask, attacker_idx=1, num_targets=2, patch_radius=5, sigma_mult=1.0, steps=10, perturb_region_mask=preg)
                f_lg, _ = generate_local_ghost_spoofing_heal(wrapper, feat, rlen, amat, mods_l, vmask, attacker_idx=1, num_spoofs=2, patch_radius=5, sigma_mult=2.0, steps=10, frame_seed=idx, perturb_region_mask=preg)

                # 每帧仅预热一次 idx-1 隐状态，供 clean 与 3 种攻击态复用（消除 4x 重复读盘与编码）
                _prime_hs_prev_frame(hs_shared, wrapper, ds_ref, sample_seqs, seq_name, idx, mod)

                feat_map = {
                    "clean": feat,
                    "pgd_1.0sigma": f_p1,
                    "local_removal_1.0sigma": f_lr,
                    "local_ghost_2.0sigma": f_lg,
                }
                for st_name, f_cur in feat_map.items():
                    dev_4 = b["dev_clean_common"][mod] if st_name == "clean" else compute_dev_maps_from_protocol(wrapper, p_dict, f_cur, b["p_ego_cls"], b["cov_ego"])
                    sc_t1 = compute_raw_frame_score_from_dev(
                        dev_4, cov_2d, d_idx, v_idx, mod, auditor.stats, "tier1_unified", auditor.main_cfg, "main_ms_cos_clear_sem"
                    )
                    ar_t4 = auditor.audit_dev_maps(dev_4, cov_2d, d_idx, v_idx, mod, build_gate=False)
                    sc_t4 = float(ar_t4["score_t4"])
                    hs_r = _score_hs_cur_frame(hs_shared, wrapper, seq_name, idx, f_cur, p_dict, mod)
                    sc_hs = float(hs_r["s_k"])
                    luc_r = _compute_lucia_pair_score(wrapper, f_cur, p_dict)
                    sc_luc = float(luc_r["anom_score"])

                    sc_seq["tier1_unified"][mod][st_name].append(float(sc_t1))
                    sc_seq["tier4_ours"][mod][st_name].append(sc_t4)
                    sc_seq["hetshield_shared"][mod][st_name].append(sc_hs)
                    sc_seq["lucia"][mod][st_name].append(sc_luc)

                    if st_name == "clean":
                        luc_2c[mod]["l1"].append(luc_r["l1_dist"])
                        luc_2c[mod]["trust"].append(luc_r["trust"])
                        hs_cc[mod]["tau_k"].append(float(hs_r["tau_k"]))
                        hs_cc[mod]["sigma_k"].append(float(hs_r["sigma_k"]))
                        hs_cc[mod]["s_k"].append(sc_hs)

        # Part 2: 本序列 >=3 车 ROBOSAC 与混合车队 LUCIA
        from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
        from opencood.tools import train_utils
        from opencood.utils import common_utils

        ac = wrapper.model.pyramid_backbone.align_corners
        for fm in ("all_m1", "all_m2", "mixed"):
            ds_f = ds_fleets[fm]
            for idx in idxs:
                raw_b = train_utils.to_device(ds_f.collate_batch_test([ds_f[idx]]), wrapper.device_obj)
                p = wrapper.extract_protocol_features(raw_b["ego"])
                feat = p["heter_feature_2d"]
                n_tot = int(feat.shape[0])
                if n_tot < 3:
                    continue
                mods_l = list(p["agent_modality_list"])
                rlen = p["record_len"]
                amat = p["affine_matrix"]
                with torch.no_grad():
                    o_ego = wrapper.forward_single_agent(feat[0:1])
                    epb, _, _ = wrapper.decode_boxes(raw_b, o_ego, ds_f)
                    epb_np = common_utils.torch_tensor_to_numpy(epb) if (epb is not None and len(epb) > 0) else None
                    o_all = wrapper.forward_from_protocol(feat, rlen, amat, mods_l)
                    apb, _, _ = wrapper.decode_boxes(raw_b, o_all, ds_f)
                    apb_np = common_utils.torch_tensor_to_numpy(apb) if (apb is not None and len(apb) > 0) else None
                jac = compute_box_set_jaccard_2d(epb_np, apb_np, iou_thresh=0.3)
                robo_seq[fm]["jaccard"].append(float(jac))
                robo_seq[fm]["total"] += 1
                if jac < 0.3:
                    robo_seq[fm]["rej_03"] += 1
                if jac < 0.5:
                    robo_seq[fm]["rej_05"] += 1

                if fm == "mixed":
                    aligned = warp_affine_simple(feat, amat[0, 0, :n_tot], (PROTO_H, PROTO_W), align_corners=ac)
                    trust, l1_sc = cp_defense_lucia_heal(aligned, compression_ratio=32)
                    for ci in range(1, n_tot):
                        mc = mods_l[ci]
                        if mc in luc_mix:
                            luc_mix[mc]["trust"].append(float(trust[ci].item()))
                            luc_mix[mc]["l1"].append(float(l1_sc[ci].item()))

        rec = {
            "seq_name": seq_name,
            "scores_by_meth": sc_seq,
            "lucia_raw_clean": luc_2c,
            "hs_comp_clean": hs_cc,
            "robosac_clean": robo_seq,
            "lucia_mixed_clean": luc_mix,
        }
        _atomic_write_json(cpath, rec)
        seq_caches[seq_name] = rec
        print(f"  [Step 5-3] 完成序列 {seq_name} ({len(idxs)} 帧, 累计 {time.time() - t0:.1f}s)", flush=True)

    if args.worker_only:
        print(f"\n[Worker完成] Step 5-3 seqs={target_seqs} (耗时 {time.time() - t0:.1f}s)")
        return

    # 从 8 条序列缓存汇总全部统计
    scores_by_meth = {
        mk: {s: seq_caches[s]["scores_by_meth"][mk] for s in seq_order}
        for mk in methods_4
    }
    lucia_raw_clean = {
        m: {k: [v for s in seq_order for v in seq_caches[s]["lucia_raw_clean"][m][k]] for k in ("l1", "trust")}
        for m in ("m1", "m2")
    }
    hs_comp_clean = {
        m: {k: [v for s in seq_order for v in seq_caches[s]["hs_comp_clean"][m][k]] for k in ("tau_k", "sigma_k", "s_k")}
        for m in ("m1", "m2")
    }
    robosac_clean = {
        fm: {
            "jaccard": [v for s in seq_order for v in seq_caches[s]["robosac_clean"][fm]["jaccard"]],
            "rej_03": sum(int(seq_caches[s]["robosac_clean"][fm]["rej_03"]) for s in seq_order),
            "rej_05": sum(int(seq_caches[s]["robosac_clean"][fm]["rej_05"]) for s in seq_order),
            "total": sum(int(seq_caches[s]["robosac_clean"][fm]["total"]) for s in seq_order),
        }
        for fm in ("all_m1", "all_m2", "mixed")
    }
    lucia_mixed_clean = {
        m: {k: [v for s in seq_order for v in seq_caches[s]["lucia_mixed_clean"][m][k]] for k in ("trust", "l1")}
        for m in ("m1", "m2")
    }

    # 汇总实验 A 与实验 B 全部统计
    boot_idx = build_seq_bootstrap_indices(seq_order, n_boot=BOOT_N, seed=BOOT_SEED)

    exp_a = OrderedDict({
        "clean_score_distributions_2cav": OrderedDict({
            mk: {
                m: _dist_stats(np.asarray([v for s in seq_order for v in scores_by_meth[mk][s][m]["clean"]], dtype=np.float64))
                for m in ("m1", "m2")
            }
            for mk in methods_4
        }),
        "hetshield_components_clean": {
            m: {k: _dist_stats(np.asarray(v, dtype=np.float64)) for k, v in hs_comp_clean[m].items()}
            for m in ("m1", "m2")
        },
        "hetshield_fixed_thresholds": {
            "paper_tau_0.3": _eval_threshold_regime(seq_order, boot_idx, scores_by_meth["hetshield_shared"], 0.3, 0.3),
            "fit_tuned_tau_p95": _eval_threshold_regime(
                seq_order, boot_idx, scores_by_meth["hetshield_shared"],
                float(hs_cfg["fit_tuned_shared_tau_p95"]), float(hs_cfg["fit_tuned_shared_tau_p95"]),
            ),
        },
        "lucia_2cav_clean": {
            m: {k: _dist_stats(np.asarray(v, dtype=np.float64)) for k, v in lucia_raw_clean[m].items()}
            for m in ("m1", "m2")
        },
        "lucia_mixed_multicav_clean": {
            m: {k: _dist_stats(np.asarray(v, dtype=np.float64)) for k, v in lucia_mixed_clean[m].items()}
            for m in ("m1", "m2")
        },
        "robosac_ge3cav_clean": {
            fm: {
                "total_frames": robosac_clean[fm]["total"],
                "jaccard_dist": _dist_stats(np.asarray(robosac_clean[fm]["jaccard"], dtype=np.float64)),
                "reject_rate_thresh_0.3": round(robosac_clean[fm]["rej_03"] / max(robosac_clean[fm]["total"], 1), 5),
                "reject_rate_thresh_0.5": round(robosac_clean[fm]["rej_05"] / max(robosac_clean[fm]["total"], 1), 5),
            }
            for fm in ("all_m1", "all_m2", "mixed")
        },
    })

    # 实验 B：等总误报率 (2% / 5% / 10%) 及 压低 m2 误报 <= 5% 口径下的同台对比
    exp_b = OrderedDict()
    for fpr_tag, target_fpr in (("fpr_2pct", 0.02), ("fpr_5pct", 0.05), ("fpr_10pct", 0.10)):
        exp_b[fpr_tag] = OrderedDict()
        for mk in methods_4:
            all_ben = np.asarray(
                [v for s in seq_order for m in ("m1", "m2") for v in scores_by_meth[mk][s][m]["clean"]], dtype=np.float64
            )
            th_eq = find_exact_fpr_threshold(all_ben, target_fpr)
            exp_b[fpr_tag][mk] = _eval_threshold_regime(seq_order, boot_idx, scores_by_meth[mk], th_eq, th_eq)
            exp_b[fpr_tag][mk]["threshold_used"] = round(float(th_eq), 6)

    # 压低诚实 m2 误报至 <= 5% 口径（统一判据被迫放宽门限，展示对 m1 攻击者检出率崩跌 vs 本文按条件判据守住双模态）
    exp_b["m2_fpr_capped_5pct"] = OrderedDict()
    for mk in methods_4:
        ben_m1 = np.asarray([v for s in seq_order for v in scores_by_meth[mk][s]["m1"]["clean"]], dtype=np.float64)
        ben_m2 = np.asarray([v for s in seq_order for v in scores_by_meth[mk][s]["m2"]["clean"]], dtype=np.float64)
        th_m2 = find_exact_fpr_threshold(ben_m2, 0.05)
        if mk == "tier4_ours":
            # 本文按条件判据各模态独立按 5% 对齐（div_p95 本身已跨模态校准）
            th_m1 = find_exact_fpr_threshold(ben_m1, 0.05)
        else:
            # 统一判据必须使用同一门限，为了压低 m2 误报被迫放宽到 th_m2
            th_m1 = th_m2
        exp_b["m2_fpr_capped_5pct"][mk] = _eval_threshold_regime(seq_order, boot_idx, scores_by_meth[mk], th_m1, th_m2)
        exp_b["m2_fpr_capped_5pct"][mk]["threshold_m1"] = round(float(th_m1), 6)
        exp_b["m2_fpr_capped_5pct"][mk]["threshold_m2"] = round(float(th_m2), 6)

    report = OrderedDict({
        "meta": {
            "description": "步骤 ⑤-3 异构车队统一判据跨模态误报偏差与检出率权衡实证报告（Dev-320，成对 Bootstrap 95% CI）",
            "seq_order": seq_order,
            "total_frames": int(eval_split["total_frames"]),
            "bootstrap_iters": BOOT_N,
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "experiment_a_existing_defenses_bias": exp_a,
        "experiment_b_equal_fpr": exp_b,
    })
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    _plot_motivation_figure(report, args.out_fig)
    print(f"\n[完成] 立足点实验报告：{args.out_json}，对比图：{args.out_fig} (总耗时 {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
