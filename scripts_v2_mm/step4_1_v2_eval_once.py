# -*- coding: utf-8 -*-
"""
步骤 ④-1 修正版 · Phase 2：在 80 帧全新 Eval 盲测集上一次性评估主方法与消融方法并出图。

严格纪律：
1. 仅从 step4_1_v2_split_plan.json 与 step4_1_v2_frozen_calibrator.json 加载冻结配置与门限，
   严禁利用任何 Eval 数据调整特征组合、权重、K 值或校准曲线；
2. 逐帧在 m1 与 m2 上生成 6 类攻击（含 0~6m 局部攻击压力测试与仅遮挡区伪造能力边界），
   在相同 5.0% 总误报率与严格 Fit 冻结门限两个口径下同步报告总体及 m1/m2 各自误报率、检出率与 AUROC；
3. 生成 2 张防裁边高清图，保存完整评估报告 JSON。
"""

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

PROJ_ROOT = "/root/autodl-tmp/proj_dir2_hetero_trust"
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from src_v2_mm.heal_attacks import (
    generate_local_ghost_spoofing_heal,
    generate_local_near0_6m_attack_heal,
    generate_local_occluded_ghost_heal,
    generate_local_removal_heal,
    generate_scaled_pgd_removal_heal,
)
from src_v2_mm.heal_base_wrapper import HEALBaseWrapper
from src_v2_mm.step4_1_v2_core import (
    DIST_LABELS_5,
    PROJ_ROOT as CORE_ROOT,
    TIERS_4,
    TRAIN_FULL19_DIR,
    VIS_LABELS_3,
    apply_frame_calibration,
    build_bev_grid_xy_256,
    compute_auroc,
    compute_dev_maps_from_protocol,
    compute_raw_frame_score_from_dev,
    extract_frame_context_and_clean,
    find_exact_fpr_threshold,
)

OUT_DIR = os.path.join(CORE_ROOT, "results_rq1_v2_mm")
SPLIT_JSON = os.path.join(OUT_DIR, "step4_1_v2_split_plan.json")
CALIB_JSON = os.path.join(OUT_DIR, "step4_1_v2_frozen_calibrator.json")
OUT_REPORT = os.path.join(OUT_DIR, "step4_1_v2_eval_report.json")
OUT_FIG_HEAT = os.path.join(OUT_DIR, "fig_step4_1_v2_benign_mscos_heatmap.png")
OUT_FIG_4TIER = os.path.join(OUT_DIR, "fig_step4_1_v2_4tier_main_vs_ablation.png")

EVAL_ATTACKS = [
    "scaled_pgd_0.5sigma",
    "scaled_pgd_1.0sigma",
    "local_removal_1.0sigma",
    "local_ghost_2.0sigma",
    "local_near0_6m_removal_2.0sigma",
    "local_ghost_occluded_2.0sigma",
]


def _evaluate_method(scores_dict: dict, valid_flags: dict, frozen_entry: dict) -> dict:
    """对冻结方法计算相等 5.0% 总误报率及 Fit 冻结门限两大口径完整结果。"""
    method_out = OrderedDict()
    tier_calib = frozen_entry["tier_calib"]

    for tier in TIERS_4:
        ben_m1 = np.asarray(scores_dict[tier]["m1"]["benign"], dtype=np.float64)
        ben_m2 = np.asarray(scores_dict[tier]["m2"]["benign"], dtype=np.float64)
        ben_all = np.concatenate([ben_m1, ben_m2], axis=0)

        th_5pct = find_exact_fpr_threshold(ben_all, 0.05)
        th_fit = float(tier_calib["tier1_unified"]["th_fit"]) if tier == "tier1_unified" else 1.0

        res_t = OrderedDict({
            "equal_5pct_total_fpr": {
                "threshold": round(th_5pct, 6),
                "total_fpr": round(float(np.mean(ben_all > th_5pct)), 5),
                "m1_fpr": round(float(np.mean(ben_m1 > th_5pct)), 5),
                "m2_fpr": round(float(np.mean(ben_m2 > th_5pct)), 5),
            },
            "fit_frozen_threshold": {
                "threshold": round(th_fit, 6),
                "total_fpr": round(float(np.mean(ben_all > th_fit)), 5),
                "m1_fpr": round(float(np.mean(ben_m1 > th_fit)), 5),
                "m2_fpr": round(float(np.mean(ben_m2 > th_fit)), 5),
            },
            "attacks": OrderedDict(),
        })

        for aname in EVAL_ATTACKS:
            v1 = np.asarray(valid_flags["m1"][aname], dtype=bool)
            v2 = np.asarray(valid_flags["m2"][aname], dtype=bool)
            p1 = np.asarray(scores_dict[tier]["m1"][aname], dtype=np.float64)[v1]
            p2 = np.asarray(scores_dict[tier]["m2"][aname], dtype=np.float64)[v2]
            p_all = np.concatenate([p1, p2], axis=0)

            res_t["attacks"][aname] = {
                "valid_counts": {"m1": int(p1.size), "m2": int(p2.size), "total": int(p_all.size)},
                "tpr_at_5pct_total_fpr": {
                    "m1": round(float(np.mean(p1 > th_5pct)), 5) if p1.size else 0.0,
                    "m2": round(float(np.mean(p2 > th_5pct)), 5) if p2.size else 0.0,
                    "overall": round(float(np.mean(p_all > th_5pct)), 5) if p_all.size else 0.0,
                },
                "tpr_at_fit_frozen_thresh": {
                    "m1": round(float(np.mean(p1 > th_fit)), 5) if p1.size else 0.0,
                    "m2": round(float(np.mean(p2 > th_fit)), 5) if p2.size else 0.0,
                    "overall": round(float(np.mean(p_all > th_fit)), 5) if p_all.size else 0.0,
                },
                "auroc": {
                    "m1": round(compute_auroc(ben_m1, p1), 5) if p1.size else 0.5,
                    "m2": round(compute_auroc(ben_m2, p2), 5) if p2.size else 0.5,
                    "overall": round(compute_auroc(ben_all, p_all), 5) if p_all.size else 0.5,
                },
            }
        method_out[tier] = res_t
    return method_out


def plot_benign_mscos_heatmap(benign_summary: dict, out_png: str):
    """绘制图 1：m1/m2 在 5 距离(含 0-6m) × 3 可见性 下的多尺度余弦分布与子单元像素误报对比。"""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 10.8, "axes.titleweight": "bold"})

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.6), dpi=300)
    fig.subplots_adjust(wspace=0.28, hspace=0.34, top=0.90, bottom=0.07, left=0.07, right=0.95)
    fig.suptitle(
        "Fit-100 Multi-Scale Cosine Benign Deviation Breakdown (5 Dist Bins incl. 0-6m x 3 Visibility) & Sub-Cell Exceedance",
        fontsize=12.6, fontweight="bold", y=0.96,
    )

    ms_sum = benign_summary["ms_cos_equal_dist_vis_summary"]
    fpr_cmp = benign_summary["subcell_pixel_fpr_compare"]

    # 上排：(a) m1 与 (b) m2 均值/[P95] 热力图 (3 Vis x 5 Dist)
    for col, (mod, title, cmap) in enumerate([
        ("m1", "(a) m1 Coll. (64L LiDAR): MS-Cos Mean / [P95]", "Blues"),
        ("m2", "(b) m2 Coll. (Camera-LSS): MS-Cos Mean / [P95]", "YlOrRd"),
    ]):
        ax = axes[0, col]
        mean_mat = np.zeros((3, 5), dtype=np.float64)
        p95_mat = np.zeros((3, 5), dtype=np.float64)
        for vi, vl in enumerate(VIS_LABELS_3):
            for di, dl in enumerate(DIST_LABELS_5):
                st = ms_sum[mod][vl][dl]
                mean_mat[vi, di] = st["mean"]
                p95_mat[vi, di] = st["p95"]

        vmax = max(0.35, float(np.max(mean_mat)) * 1.15)
        im = ax.imshow(mean_mat, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels(DIST_LABELS_5)
        ax.set_yticks(np.arange(3))
        ax.set_yticklabels(VIS_LABELS_3)
        ax.set_xlabel("Distance from Collaborator Center (0-6m Isolated Bin)")
        ax.set_title(title)
        for vi in range(3):
            for di in range(5):
                col_txt = "white" if mean_mat[vi, di] > vmax * 0.55 else "#111111"
                ax.text(di, vi, f"{mean_mat[vi, di]:.3f}\n[{p95_mat[vi, di]:.2f}]", ha="center", va="center", color=col_txt, fontsize=8.4, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)

    # 下排：(c) m1 与 (d) m2 在关键子单元上 ② vs ③ vs ④ 像素级 P95 越界率（目标 5.0%）
    target_cells = [
        ("Clear", "0-6m"), ("Clear", "15-30m"), ("Clear", "30-45m"), ("Clear", "45m+"),
        ("Occluded", "0-6m"), ("Occluded", "45m+"),
    ]
    cell_labels = [f"{v[:3]}\n{d}" for v, d in target_cells]
    x = np.arange(len(target_cells))
    w = 0.25

    for col, (mod, title) in enumerate([
        ("m1", "(c) m1 Sub-Cell Benign Exceedance FPR (Target = 5%)"),
        ("m2", "(d) m2 Sub-Cell Benign Exceedance FPR (Target = 5%)"),
    ]):
        ax = axes[1, col]
        y_t2 = [fpr_cmp[mod][v][d]["tier2_sensor_only_exceed"] * 100.0 for v, d in target_cells]
        y_t3 = [fpr_cmp[mod][v][d]["tier3_sensor_dist_exceed"] * 100.0 for v, d in target_cells]
        y_t4 = [fpr_cmp[mod][v][d]["tier4_sensor_dist_vis_exceed"] * 100.0 for v, d in target_cells]

        ax.bar(x - w, y_t2, width=w, color="#7570b3", label="Tier 2: Sensor Only")
        ax.bar(x, y_t3, width=w, color="#d95f02", label="Tier 3: Sensor + Dist(5)")
        ax.bar(x + w, y_t4, width=w, color="#1b9e77", label="Tier 4: Sensor + Dist + Vis")
        ax.axhline(5.0, color="#222222", linestyle="--", linewidth=1.1, label="5% Target Budget")
        ax.set_xticks(x)
        ax.set_xticklabels(cell_labels, fontsize=8.8)
        ax.set_ylabel("Pixel Exceedance Rate (%)")
        ax.set_ylim(0.0, max(12.0, max(y_t2 + y_t3 + y_t4) * 1.25))
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.legend(loc="upper right", fontsize=8.2, frameon=True)

    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_4tier_main_vs_ablation(eval_methods: dict, out_png: str):
    """绘制图 2：主方法 vs 消融在相同 5% 总误报率下的 4 组门限 × 6 类攻击检出率对比图。"""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11.2, "axes.titleweight": "bold"})

    fig, axes = plt.subplots(2, 1, figsize=(15.5, 10.2), dpi=300)
    fig.subplots_adjust(hspace=0.38, top=0.90, bottom=0.09, left=0.06, right=0.98)
    fig.suptitle(
        "Eval-80 (Unseen Sequences) Detection TPR @ Equal 5.0% Total FPR: Main Method vs Ablation Across 4 Threshold Tiers",
        fontsize=12.8, fontweight="bold", y=0.96,
    )

    atk_labels = [
        "0.5σ Global\nPGD",
        "1.0σ Global\nPGD",
        "Local Removal\n(1.0σ, 1-2 Obj)",
        "Local Ghost\n(2.0σ, 1-2 Obj)",
        "0-6m Stress\n(2.0σ Removal)",
        "Occ-Only Ghost\n(Cap. Boundary)",
    ]
    tier_styles = [
        ("tier1_unified", "Tier 1: Unified", "#8c96c6"),
        ("tier2_sensor_only", "Tier 2: Sensor Only", "#74a9cf"),
        ("tier3_sensor_dist", "Tier 3: Sensor + Dist(5)", "#fe9929"),
        ("tier4_sensor_dist_vis", "Tier 4: Sensor + Dist + Vis", "#1b9e77"),
    ]
    x = np.arange(len(EVAL_ATTACKS))
    w = 0.19

    panels = [
        ("main_ms_cos_clear_sem", "(a) Main Method: Multi-Scale Cosine + Clear-Zone Semantic Consistency (MS-Cos + Clear-Sem)"),
        ("ablation_ms_cos", "(b) Ablation: Pure Multi-Scale Cosine (MS-Cos Only)"),
    ]

    for row, (mkey, title) in enumerate(panels):
        ax = axes[row]
        mres = eval_methods[mkey]
        for ti, (tname, tlabel, tcol) in enumerate(tier_styles):
            tres = mres[tname]
            fpr_info = tres["equal_5pct_total_fpr"]
            lbl_with_fpr = f"{tlabel}  [Eval FPR m1={fpr_info['m1_fpr']*100:.1f}%, m2={fpr_info['m2_fpr']*100:.1f}%]"
            ys = [tres["attacks"][a]["tpr_at_5pct_total_fpr"]["overall"] * 100.0 for a in EVAL_ATTACKS]
            offs = (ti - 1.5) * w
            bars = ax.bar(x + offs, ys, width=w, color=tcol, edgecolor="#222222", linewidth=0.5, label=lbl_with_fpr)
            for b, yv in zip(bars, ys):
                ax.text(b.get_x() + b.get_width() / 2.0, yv + 1.2, f"{yv:.0f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(atk_labels, fontsize=9.2)
        ax.set_ylabel("Overall Detection TPR (%)")
        ax.set_ylim(0.0, 114.0)
        ax.set_title(title)
        ax.axvline(3.5, color="#888888", linestyle=":", linewidth=1.0)
        ax.axvline(4.5, color="#cc4c02", linestyle="--", linewidth=1.0)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.legend(loc="upper left", ncol=2, fontsize=8.0, frameon=True)

    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main():
    t0 = time.time()
    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        split_plan = json.load(f)
    with open(CALIB_JSON, "r", encoding="utf-8") as f:
        calib = json.load(f)

    fit_set = set(split_plan["fit_indices"])
    eval_indices = list(split_plan["eval_indices"])
    assert set(eval_indices).isdisjoint(fit_set), "Eval 索引与 Fit 存在重叠！"
    print(f"[Phase 2] 开始一次性盲测 {len(eval_indices)} 帧全新 Eval 样本...")

    wrapper = HEALBaseWrapper(device="cuda")
    ds_m1 = wrapper.build_dataset(data_dir=TRAIN_FULL19_DIR, use_cav=2, visualize=True, collab_modality="m1")
    ds_m2 = wrapper.build_dataset(data_dir=TRAIN_FULL19_DIR, use_cav=2, visualize=True, collab_modality="m2")
    gx, gy = build_bev_grid_xy_256(device="cuda", dtype=torch.float32)

    stats = calib["channel_tier_stats"]
    frozen_methods = calib["frozen_methods"]
    method_names = ["main_ms_cos_clear_sem", "ablation_ms_cos"]

    scores = {
        mname: {t: {mod: {"benign": [], **{a: [] for a in EVAL_ATTACKS}} for mod in ("m1", "m2")} for t in TIERS_4}
        for mname in method_names
    }
    valid_flags = {mod: {a: [] for a in EVAL_ATTACKS} for mod in ("m1", "m2")}
    atk_gen_stats = {
        mod: {
            a: {"valid": 0, "total": 0, "linf": [], "prob_drop": [], "max_ghost": [], "in06m_ratio": [], "inocc_ratio": []}
            for a in EVAL_ATTACKS
        }
        for mod in ("m1", "m2")
    }

    for rank, idx in enumerate(eval_indices):
        b = extract_frame_context_and_clean(wrapper, ds_m1, ds_m2, idx, gx, gy)
        cov_2d = b["cov_ego"][0, 0]
        d_idx = b["d_idx"]
        v_idx = b["v_idx"]

        for mod, p_dict in (("m1", b["p1"]), ("m2", b["p2"])):
            # 1. 干净帧打分
            dev_clean = b["dev_clean"][mod]
            for mname in method_names:
                f_entry = frozen_methods[mname]
                cfg = f_entry["cfg"]
                tcal = f_entry["tier_calib"]
                for tier in TIERS_4:
                    raw = compute_raw_frame_score_from_dev(dev_clean, cov_2d, d_idx, v_idx, mod, stats, tier, cfg, mname)
                    norm_s = apply_frame_calibration(raw, mod, tier, cfg["frame_norm_mode"], tcal)
                    scores[mname][tier][mod]["benign"].append(norm_s)

            # 2. 生成 6 类 Eval 攻击并打分
            feat = p_dict["heter_feature_2d"]
            rlen = p_dict["record_len"]
            amat = p_dict["affine_matrix"]
            mods = p_dict["agent_modality_list"]
            vmask = p_dict["cam_valid_mask"]
            preg = b["cov_local_m1"] if mod == "m1" else None

            f_05, st_05 = generate_scaled_pgd_removal_heal(wrapper, feat, rlen, amat, mods, vmask, 1, 0.5, 10, preg)
            f_10, st_10 = generate_scaled_pgd_removal_heal(wrapper, feat, rlen, amat, mods, vmask, 1, 1.0, 10, preg)
            f_lr, st_lr = generate_local_removal_heal(wrapper, feat, rlen, amat, mods, vmask, 1, 2, 5, 1.0, 12, preg)
            f_lg, st_lg = generate_local_ghost_spoofing_heal(wrapper, feat, rlen, amat, mods, vmask, 1, 2, 5, 2.0, 12, idx, preg)
            f_n6, st_n6 = generate_local_near0_6m_attack_heal(wrapper, feat, rlen, amat, mods, vmask, 1, 2.0, 15, idx, preg, False)
            f_oc, st_oc = generate_local_occluded_ghost_heal(wrapper, feat, rlen, amat, mods, vmask, b["occ_mask_ego"], 1, 2, 5, 2.0, 12, idx, preg)

            with torch.no_grad():
                o_clean = wrapper.forward_from_protocol(feat, rlen, amat, mods)

                # 记录局部攻击实际破坏强度
                if st_lr.get("target_centers_hw"):
                    o_lr = wrapper.forward_from_protocol(f_lr, rlen, amat, mods)
                    drops = [
                        float(torch.sigmoid(o_clean["cls_preds"])[0, :, hc, wc].max() - torch.sigmoid(o_lr["cls_preds"])[0, :, hc, wc].max())
                        for hc, wc in st_lr["target_centers_hw"]
                    ]
                    atk_gen_stats[mod]["local_removal_1.0sigma"]["prob_drop"].append(float(np.mean(drops)))
                if st_lg.get("anchors_hwr"):
                    o_lg = wrapper.forward_from_protocol(f_lg, rlen, amat, mods)
                    gp = max(float(torch.sigmoid(o_lg["cls_preds"])[0, r, h, w]) for h, w, r in st_lg["anchors_hwr"])
                    atk_gen_stats[mod]["local_ghost_2.0sigma"]["max_ghost"].append(gp)
                if st_n6.get("valid", False):
                    atk_gen_stats[mod]["local_near0_6m_removal_2.0sigma"]["prob_drop"].append(float(st_n6.get("prob_drop", 0.0)))
                    atk_gen_stats[mod]["local_near0_6m_removal_2.0sigma"]["in06m_ratio"].append(float(st_n6.get("perturb_in_0_6m_ratio", 0.0)))
                if st_oc.get("valid", False) and st_oc.get("anchors_hwr"):
                    o_oc = wrapper.forward_from_protocol(f_oc, rlen, amat, mods)
                    gp_oc = max(float(torch.sigmoid(o_oc["cls_preds"])[0, r, h, w]) for h, w, r in st_oc["anchors_hwr"])
                    atk_gen_stats[mod]["local_ghost_occluded_2.0sigma"]["max_ghost"].append(gp_oc)
                    atk_gen_stats[mod]["local_ghost_occluded_2.0sigma"]["inocc_ratio"].append(float(st_oc.get("perturb_in_occ_ratio", 0.0)))

            for aname, f_adv, st in (
                ("scaled_pgd_0.5sigma", f_05, st_05),
                ("scaled_pgd_1.0sigma", f_10, st_10),
                ("local_removal_1.0sigma", f_lr, st_lr),
                ("local_ghost_2.0sigma", f_lg, st_lg),
                ("local_near0_6m_removal_2.0sigma", f_n6, st_n6),
                ("local_ghost_occluded_2.0sigma", f_oc, st_oc),
            ):
                if aname == "local_removal_1.0sigma":
                    is_val = bool(st.get("perturb_grids", 0) > 0)
                elif aname == "local_ghost_2.0sigma":
                    is_val = bool(st.get("num_spoof_anchors", 0) > 0 and st.get("perturb_grids", 0) > 0)
                elif aname in ("local_near0_6m_removal_2.0sigma", "local_ghost_occluded_2.0sigma"):
                    is_val = bool(st.get("valid", False))
                else:
                    is_val = True

                valid_flags[mod][aname].append(is_val)
                atk_gen_stats[mod][aname]["total"] += 1
                if is_val:
                    atk_gen_stats[mod][aname]["valid"] += 1
                    atk_gen_stats[mod][aname]["linf"].append(float(st.get("linf_max", 0.0)))

                dev_adv = compute_dev_maps_from_protocol(wrapper, p_dict, f_adv, b["p_ego_cls"], b["cov_ego"])
                for mname in method_names:
                    f_entry = frozen_methods[mname]
                    cfg = f_entry["cfg"]
                    tcal = f_entry["tier_calib"]
                    for tier in TIERS_4:
                        raw = compute_raw_frame_score_from_dev(dev_adv, cov_2d, d_idx, v_idx, mod, stats, tier, cfg, mname)
                        norm_s = apply_frame_calibration(raw, mod, tier, cfg["frame_norm_mode"], tcal)
                        scores[mname][tier][mod][aname].append(norm_s)

        if (rank + 1) % 10 == 0 or (rank + 1) == len(eval_indices):
            print(f"  已评估 Eval 样本 {rank + 1}/{len(eval_indices)} ({time.time() - t0:.1f}s)")

    eval_methods = OrderedDict()
    for mname in method_names:
        eval_methods[mname] = _evaluate_method(scores[mname], valid_flags, frozen_methods[mname])

    atk_summary = {
        mod: {
            aname: {
                "valid": v["valid"],
                "total": v["total"],
                "linf_mean": round(float(np.mean(v["linf"])) if v["linf"] else 0.0, 4),
                **({"mean_prob_drop": round(float(np.mean(v["prob_drop"])), 4)} if v["prob_drop"] else {}),
                **({"mean_max_ghost_prob": round(float(np.mean(v["max_ghost"])), 4)} if v["max_ghost"] else {}),
                **({"mean_in_0_6m_ratio": round(float(np.mean(v["in06m_ratio"])), 4)} if v["in06m_ratio"] else {}),
                **({"mean_in_occ_ratio": round(float(np.mean(v["inocc_ratio"])), 4)} if v["inocc_ratio"] else {}),
            }
            for aname, v in atk_gen_stats[mod].items()
        }
        for mod in ("m1", "m2")
    }

    report = OrderedDict({
        "meta": {
            "description": "步骤 ④-1 修正版 Eval-80 盲测一次性报告（严格零泄露）",
            "excluded_3cam_seq": split_plan["excluded_3cam_seq"],
            "fit_frames": split_plan["counts"]["fit_total"],
            "eval_frames": len(eval_indices),
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "frozen_configs": {mname: frozen_methods[mname]["cfg"] for mname in method_names},
        "eval_attack_generation_stats": atk_summary,
        "eval_results": eval_methods,
    })

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    plot_benign_mscos_heatmap(calib["benign_dist_vis_summary"], OUT_FIG_HEAT)
    plot_4tier_main_vs_ablation(eval_methods, OUT_FIG_4TIER)
    print(f"\n[Phase 2 完成] 报告与图表已生成：{OUT_REPORT} (耗时 {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
