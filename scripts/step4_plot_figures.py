# -*- coding: utf-8 -*-
"""
阶段四预开题汇报高分辨率图表生成脚本（step4_plot_figures.py）：
直接从 outputs/ 下的真实实验 JSON 数据绘制 4 张 300 DPI 出版级图表：
1. fig1_motivation_entanglement.png：问题动机实证——同构防御（ROBOSAC / Global-Residual）在开放异构下的分布混叠与高误报崩溃 vs 本文解耦分数的清晰可分性；
2. fig2_lipschitz_amplification.png：适配器理论与实测局部 Lipschitz 扰动放大率对比（Unconstrained Conv+BN vs 本文 LB-FPA）；
3. fig3_main_pareto_benchmark.png：OPV2V 未见测试集主实验对比（Clean AP@0.5 vs 四类攻击平均 Attacked AP@0.5 帕累托图 + 分场景柱状图）；
4. fig4_ablation_and_strata.png：逐模块消融增益与客观物理真值框三层（Clear / Sparse / Blind-Spot Occluded）召回率对比。
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def setup_style():
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.28
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["font.size"] = 10


def plot_fig1_motivation(dist_data: dict, eval_data: dict, calib_data: dict, save_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # 子图 A: ROBOSAC-Full 在良性同构、良性异构与恶意攻击上的分数重叠
    ax = axes[0]
    rb = dist_data["robosac_full"]
    b_hm = np.array(rb["benign_homo"])
    b_ht = np.array(rb["benign_hetero"])
    m_all = np.array(rb["malicious_removal"] + rb["malicious_spoof"])

    bins_a = np.linspace(0, max(np.percentile(m_all, 98), np.percentile(b_ht, 98)) * 1.08, 32)
    ax.hist(b_hm, bins=bins_a, density=True, alpha=0.55, color="#2b8cbe", label="Benign Homo (64L PP)")
    ax.hist(b_ht, bins=bins_a, density=True, alpha=0.55, color="#fdae61", label="Benign Hetero (16L/32L/SECOND)")
    ax.hist(m_all, bins=bins_a, density=True, alpha=0.50, color="#d7191c", label="Malicious Attackers")
    th_hm = calib_data["homo_calibrated_95"]["robosac_full"]
    bfqr_rb_hm = eval_data["quarantine_and_suppression"]["homo_def_robosac_homo_th"]["BFQR_benign_false_quarantine_rate"] * 100.0
    ax.axvline(th_hm, color="#08519c", linestyle="--", linewidth=1.8, label=f"Homo 95% Thresh ({th_hm:.2f})")
    ax.set_title(f"(a) Homo Consensus (ROBOSAC CCLoss)\nHomo-Thresh Benign False Quarantine = {bfqr_rb_hm:.1f}%", fontweight="bold")
    ax.set_xlabel("Consensus Discrepancy Score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, loc="upper right")

    # 子图 B: 本文 DUID + UCR-Fusion 校准帧级异常分数分布
    ax = axes[1]
    ou = dist_data["ours_s_frame"]
    o_hm = np.array(ou["benign_homo"])
    o_ht = np.array(ou["benign_hetero"])
    o_mal = np.array(ou["malicious_removal"] + ou["malicious_spoof"])

    max_x = max(np.percentile(o_mal, 96), np.percentile(o_ht, 99) * 2.0)
    bins_b = np.linspace(0, max_x, 34)
    ax.hist(o_hm, bins=bins_b, density=True, alpha=0.55, color="#2b8cbe", label="Benign Homo")
    ax.hist(o_ht, bins=bins_b, density=True, alpha=0.60, color="#2ca25f", label="Benign Hetero (Calibrated)")
    ax.hist(o_mal, bins=bins_b, density=True, alpha=0.55, color="#d7191c", label="Malicious Attackers")
    th_ou = calib_data["hetero_calibrated_95"]["ours_full"]
    ax.axvline(th_ou, color="#006d2c", linestyle="--", linewidth=1.8, label=f"Frozen Val 95% Thresh ({th_ou:.2f})")
    auroc_ou = eval_data["detector_hetero_auroc"].get("ours_s_frame", 0.0)
    ax.set_title(f"(b) Ours (Uncertainty-Decoupled Audit)\nHetero AUROC = {auroc_ou:.3f} (Clean Separation)", fontweight="bold")
    ax.set_xlabel("Calibrated Cluster Anomaly Score $s_i^{frame}$")
    ax.legend(fontsize=8, loc="upper right")

    # 子图 C: 良性异构误杀率 BFQR vs 恶意压制率 MSR 柱状对比
    ax = axes[2]
    q_dict = eval_data["quarantine_and_suppression"]
    labels = ["GlobalRes\n(Homo-Th)", "ROBOSAC\n(Homo-Th)", "GlobalRes\n(Het-Th)", "ROBOSAC-Vis\n(Het-Th)", "Ours\n(BUD-Fusion)"]
    keys = [
        "homo_def_global_res_homo_th",
        "homo_def_robosac_homo_th",
        "hetero_def_global_res",
        "hetero_def_robosac_vis",
        "ours_bud_fusion",
    ]
    bfqr = [q_dict[k]["BFQR_benign_false_quarantine_rate"] * 100.0 for k in keys]
    msr = [q_dict[k]["MSR_malicious_suppression_rate"] * 100.0 for k in keys]

    x = np.arange(len(labels))
    w = 0.36
    b1 = ax.bar(x - w / 2, bfqr, w, color="#fc8d59", edgecolor="black", linewidth=0.6, label="Benign False Quarantine (BFQR % ↓)")
    b2 = ax.bar(x + w / 2, msr, w, color="#3288bd", edgecolor="black", linewidth=0.6, label="Malicious Suppression (MSR % ↑)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("(c) False Quarantine vs. Attack Suppression", fontweight="bold")
    ax.legend(fontsize=8, loc="upper center")
    for rect in b1:
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 1.5, f"{rect.get_height():.1f}", ha="center", fontsize=7.5)
    for rect in b2:
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 1.5, f"{rect.get_height():.1f}", ha="center", fontsize=7.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig1_motivation_entanglement.png"))
    plt.close(fig)


def plot_fig2_lipschitz(step1_data: dict, save_dir: str):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    profs = ["profile_b_second64", "profile_c_coarse32", "profile_d_sparse16"]
    p_labels = ["SECOND-64\n(3D SparseConv)", "CoarsePillar-32\n(0.8m Grid)", "SparsePillar-16\n(16-Beam LiDAR)"]

    u_max = [step1_data[p]["Unconstrained"]["empirical_lip_max"] for p in profs]
    b_max = [step1_data[p]["LB_FPA"]["empirical_lip_max"] for p in profs]
    theo = step1_data[profs[0]]["LB_FPA"]["theoretical_lip_bound"]

    x = np.arange(len(profs))
    w = 0.34
    r1 = ax.bar(x - w / 2, u_max, w, color="#e34a33", edgecolor="black", linewidth=0.7, label="Unconstrained Adapter (Conv2d + BN)")
    r2 = ax.bar(x + w / 2, b_max, w, color="#2b8cbe", edgecolor="black", linewidth=0.7, label="Ours LB-FPA (Exact Conv2D Spectral + ConvexRes)")
    if isinstance(theo, (int, float)):
        ax.axhline(theo, color="#006d2c", linestyle="--", linewidth=1.6, label=f"Analytical Upper Bound ({theo:.2f})")

    ax.set_xticks(x)
    ax.set_xticklabels(p_labels, fontsize=9.5)
    ax.set_ylabel("Empirical Worst-Case Amplification $\\|\\Delta Z\\|_2 / \\|\\Delta F\\|_2$")
    ax.set_title("Cross-Domain Perturbation Amplification Across Heterogeneous Adapters", fontweight="bold")
    ax.legend(fontsize=8.5)
    for rect in list(r1) + list(r2):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.08, f"{rect.get_height():.2f}x", ha="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig2_lipschitz_amplification.png"))
    plt.close(fig)


def plot_fig3_main_benchmark(eval_data: dict, save_dir: str):
    mb = eval_data["main_benchmark_ap"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8))

    methods = [
        ("ego_only", "Ego-Only (No Collab)", "#7f7f7f", "s"),
        ("unconstrained_att", "STAMP/HEAL-AttFusion", "#d62728", "v"),
        ("hetero_where2comm", "Hetero-Where2comm", "#ff7f0e", "^"),
        ("homo_def_robosac_homo_th", "ROBOSAC (Homo-Thresh)", "#9467bd", "D"),
        ("hetero_def_global_res", "Global-Feat-Residual", "#8c564b", "P"),
        ("hetero_def_robosac_vis", "ROBOSAC-VisCalib", "#1f77b4", "o"),
        ("ours_bud_fusion", "Ours (BUD-Fusion)", "#2ca02c", "*"),
    ]
    atk_scens = ["pgd_removal", "ghost_spoof", "hetpoison_transfer", "whitebox_adaptive"]

    # 子图 A: Clean AP@0.5 vs Mean Attacked AP@0.5 帕累托前沿图
    ax = axes[0]
    for m_key, m_label, color, marker in methods:
        c_ap = mb["clean"][m_key]["AP50"] * 100.0
        a_ap = np.mean([mb[sc][m_key]["AP50"] for sc in atk_scens]) * 100.0
        sz = 230 if m_key == "ours_bud_fusion" else 110
        ax.scatter(c_ap, a_ap, s=sz, c=color, marker=marker, edgecolors="black", linewidths=0.8, label=m_label, zorder=4)
        if m_key == "hetero_def_robosac_vis":
            xy_off = (-58, 7)
        elif m_key == "hetero_def_global_res":
            xy_off = (5, -10)
        elif m_key == "unconstrained_att":
            xy_off = (-35, -12)
        else:
            xy_off = (5, 4)
        ax.annotate(m_label, (c_ap, a_ap), xytext=xy_off, textcoords="offset points", fontsize=8)

    ax.set_xlabel("Clean Open-Heterogeneous AP@0.5 (%) ↑")
    ax.set_ylabel("Mean Attacked AP@0.5 Across 4 Threats (%) ↑")
    ax.set_title("(a) Utility-Security Pareto Frontier on OPV2V Test", fontweight="bold")
    ax.margins(x=0.14, y=0.12)
    ax.legend(fontsize=7.8, loc="lower left", framealpha=0.92)

    # 子图 B: 五类场景分项 AP@0.7 柱状对比
    ax = axes[1]
    sel_methods = [
        ("ego_only", "Ego-Only", "#bdbdbd"),
        ("unconstrained_att", "STAMP-Att", "#fb6a4a"),
        ("hetero_where2comm", "Where2comm", "#fdae6b"),
        ("hetero_def_robosac_vis", "ROBOSAC-Vis", "#6baed6"),
        ("ours_bud_fusion", "Ours (BUD)", "#31a354"),
    ]
    scen_keys = ["clean", "pgd_removal", "ghost_spoof", "hetpoison_transfer", "whitebox_adaptive"]
    scen_labels = ["Clean\nHetero", "PGD\nRemoval", "Ghost\nSpoofing", "HetPoison\nTransfer", "WhiteBox\nAdaptive"]

    x = np.arange(len(scen_keys))
    w = 0.155
    for idx_m, (mk, ml, col) in enumerate(sel_methods):
        vals = [mb[sc][mk]["AP70"] * 100.0 for sc in scen_keys]
        offset = (idx_m - 2) * w
        ax.bar(x + offset, vals, w, color=col, edgecolor="black", linewidth=0.5, label=ml)

    ax.set_xticks(x)
    ax.set_xticklabels(scen_labels, fontsize=8.8)
    ax.set_ylabel("3D Detection AP@0.7 (%) ↑")
    ax.set_title("(b) Fine-Grained Localization Precision (AP@0.7) Under Attacks", fontweight="bold")
    ax.legend(fontsize=8, ncol=3, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig3_main_pareto_benchmark.png"))
    plt.close(fig)


def plot_fig4_ablation_and_strata(eval_data: dict, save_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.6))

    # 子图 A: 逐模块消融实验
    ax = axes[0]
    ab = eval_data["ablation_study_ap"]
    ab_items = [
        ("ours_full", "Full BUD-Fusion (Ours)"),
        ("ablation_unconstrained_adapter", "w/o LB-FPA (Unconstrained BN)"),
        ("ablation_raw_grid_count_c1", "w/o Target-Ray Vis (Raw Count)"),
        ("ablation_no_aleatoric_calib", "w/o Aleatoric Whitening"),
        ("ablation_no_spherical_epis", "w/o Spherical Epistemic"),
        ("ablation_quantile95_c3", "w/o Cluster-TopK (Quant95)"),
        ("ablation_unbounded_dot_att", "w/o Bounded Cosine Att"),
    ]
    y_pos = np.arange(len(ab_items))
    clean_v = [ab["clean"][k]["AP50"] * 100.0 for k, _ in ab_items]
    atk_v = [
        np.mean([ab[s][k]["AP50"] for s in ["pgd_removal", "ghost_spoof", "hetpoison_transfer"]]) * 100.0
        for k, _ in ab_items
    ]
    h = 0.36
    ax.barh(y_pos - h / 2, clean_v, h, color="#74c476", edgecolor="black", linewidth=0.6, label="Clean AP@0.5 (%)")
    ax.barh(y_pos + h / 2, atk_v, h, color="#2b8cbe", edgecolor="black", linewidth=0.6, label="Mean Attacked AP@0.5 (%)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([lbl for _, lbl in ab_items], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("AP@0.5 (%)")
    ax.set_title("(a) Controlled Component-Wise Ablation Study", fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")

    # 子图 B: 客观物理真值框三层召回率对比（Clean 盲区补盲保留 + Removal 攻击下保护）
    ax = axes[1]
    st_clean = eval_data["objective_point_strata"]["clean"]
    m_keys = [
        ("ego_only", "Ego-Only", "#bdbdbd"),
        ("unconstrained_att", "STAMP-Att", "#fd8d3c"),
        ("hetero_def_robosac_vis", "ROBOSAC-Vis", "#6baed6"),
        ("ours_bud_fusion", "Ours (BUD)", "#2ca25f"),
    ]
    strata = ["clear_recall", "sparse_recall", "blind_occluded_recall"]
    s_labels = ["Ego Clear\n(≥15 Ego Pts)", "Ego Sparse\n(1–14 Ego Pts)", "Ego Blind-Spot\n(0 Ego, ≥5 CAV Pts)"]
    x = np.arange(len(strata))
    w = 0.18
    for idx_m, (mk, ml, col) in enumerate(m_keys):
        vals = [st_clean[mk][sk] * 100.0 for sk in strata]
        ax.bar(x + (idx_m - 1.5) * w, vals, w, color=col, edgecolor="black", linewidth=0.6, label=ml)

    ax.set_xticks(x)
    ax.set_xticklabels(s_labels, fontsize=9)
    ax.set_ylabel("Object Recall@IoU=0.5 (%) ↑")
    ax.set_title("(b) Objective Ground-Truth Point-Strata Recall (Clean Hetero)", fontweight="bold")
    ax.legend(fontsize=8.2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig4_ablation_and_strata.png"))
    plt.close(fig)


def main():
    setup_style()
    out_dir = os.path.join(PROJECT_ROOT, "outputs")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    with open(os.path.join(out_dir, "step1_protocol_alignment_report.json"), "r", encoding="utf-8") as f:
        s1 = json.load(f)
    with open(os.path.join(out_dir, "calibrated_thresholds.json"), "r", encoding="utf-8") as f:
        cal = json.load(f)
    with open(os.path.join(out_dir, "step3_full_evaluation_results.json"), "r", encoding="utf-8") as f:
        ev = json.load(f)
    with open(os.path.join(out_dir, "step3_score_distributions.json"), "r", encoding="utf-8") as f:
        dist = json.load(f)

    plot_fig1_motivation(dist, ev, cal, fig_dir)
    plot_fig2_lipschitz(s1, fig_dir)
    plot_fig3_main_benchmark(ev, fig_dir)
    plot_fig4_ablation_and_strata(ev, fig_dir)
    print(f"[Step 4 Complete] Generated 4 high-resolution figures in {fig_dir}")


if __name__ == "__main__":
    main()
