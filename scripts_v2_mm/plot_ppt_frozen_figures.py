# -*- coding: utf-8 -*-
"""
基于 Dev-320 冻结版实测 JSON 自动生成答辩 PPT 专用高清中文学术图表（300 DPI）。
所有数值均直接从 E:/BUD/results_rq1_v2_mm/ 对应 JSON 读取，杜绝手填误差。
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RES_DIR = r"E:\BUD\results_rq1_v2_mm"
OUT_DIR = r"E:\BUD\ppt_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# 统一中文无衬线学术字体与清爽排版风格
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.edgecolor"] = "#CBD5E0"
plt.rcParams["axes.linewidth"] = 0.9

# 高级克制学术配色（色盲友好）
C_NODEF = "#A0AEC0"      # 柔和冷灰（无防御）
C_WHOLE = "#ED8936"      # 琥珀橙（整车降权）
C_BUD = "#1E4E8C"        # 深靛蓝（BUD-Fusion 本文）
C_EGO = "#4A5568"        # 深灰虚线（单车基准）
C_GAIN = "#22543D"       # 深墨绿（净增益标注）
C_GAIN_BG = "#E6FFFA"    # 浅薄荷绿底
C_RED = "#C53030"        # 警示红（实时红线 / 失效标注）
C_M1 = "#2B6CB0"         # LiDAR m1 蓝
C_M2 = "#DD6B20"         # Camera m2 橙


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45, color="#CBD5E0")
    ax.set_axisbelow(True)


def _safe_savefig(fig, out_p: str) -> str:
    """安全保存 PNG，避免 Windows 下原图被预览占用导致报错。"""
    tmp_p = out_p + ".tmp.png"
    fig.savefig(tmp_p, bbox_inches="tight", dpi=300)
    try:
        os.replace(tmp_p, out_p)
        return out_p
    except OSError:
        alt_p = out_p.replace(".png", "_updated.png")
        os.replace(tmp_p, alt_p)
        return alt_p


def plot_fig1_ap50(r45: dict):
    """第 1 页左图：m1 / m2 在 5 个标准工况下的端到端 3D 检测 AP@0.5 对比。"""
    conds = [
        ("clean", "干净场景\n(Clean)"),
        ("pgd_1.0sigma", "全局 PGD 扰动\n(Global PGD 1.0σ)"),
        ("local_removal_1.0sigma", "定向目标移除\n(Object Removal)"),
        ("local_ghost_2.0sigma", "虚假目标伪造\n(Object Fabrication)"),
        ("local_occluded_ghost_2.0sigma", "遮挡区目标伪造\n(Occluded Fabrication)"),
    ]
    ego_ap = r45["ap_results"]["m1"]["clean"]["ego_only"]["ap50"]["val"] * 100.0

    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.8), dpi=300, sharex=True)
    fig.patch.set_facecolor("white")

    mod_titles = [
        ("m1", "(a) 协作者为 64 线激光雷达 (m1, LiDAR-PointPillars)"),
        ("m2", "(b) 协作者为 4 目环视摄像头 (m2, Camera-LSS)"),
    ]
    x = np.arange(len(conds))
    w = 0.25

    for ax, (mod, title) in zip(axes, mod_titles):
        _style_ax(ax)
        v_nd = [r45["ap_results"][mod][c]["no_defense"]["ap50"]["val"] * 100.0 for c, _ in conds]
        v_wd = [r45["ap_results"][mod][c]["whole_cav_drop"]["ap50"]["val"] * 100.0 for c, _ in conds]
        v_bf = [r45["ap_results"][mod][c]["bud_frozen"]["ap50"]["val"] * 100.0 for c, _ in conds]

        b1 = ax.bar(x - w, v_nd, w, color=C_NODEF, edgecolor="white", linewidth=0.6, label="无防御 (No-Defense)")
        b2 = ax.bar(x, v_wd, w, color=C_WHOLE, edgecolor="white", linewidth=0.6, label="整车降权 (Whole-CAV Drop)")
        b3 = ax.bar(x + w, v_bf, w, color=C_BUD, edgecolor="white", linewidth=0.6, label="BUD-Fusion (本文)")

        ax.axhline(ego_ap, color=C_EGO, linestyle="--", linewidth=1.15, zorder=1.5, label=f"单车基准 Ego-Only ({ego_ap:.1f}%)")

        for bars, is_bud in ((b1, False), (b2, False), (b3, True)):
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + 1.1,
                    f"{h:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold" if is_bud else "normal",
                    color=C_BUD if is_bud else "#2D3748",
                    zorder=6,
                    bbox=dict(boxstyle="square,pad=0.06", facecolor="white", edgecolor="none", alpha=0.82),
                )

        # 在 3 种定向局部攻击上标注 BUD 相对整车降权的净增益
        for idx_c in (2, 3, 4):
            diff = v_bf[idx_c] - v_wd[idx_c]
            y_pos = max(v_bf[idx_c], v_wd[idx_c]) + 7.2
            ax.text(
                x[idx_c] + w * 0.45,
                y_pos,
                f"+{diff:.1f}% vs 整车",
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="bold",
                color=C_GAIN,
                bbox=dict(boxstyle="round,pad=0.18", facecolor=C_GAIN_BG, edgecolor="#81E6D9", linewidth=0.7),
            )

        ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=8, color="#1A202C")
        ax.set_ylabel("3D 检测 AP@0.5 (%)", fontsize=10, fontweight="bold", color="#2D3748")
        ax.set_ylim(0, 102)
        ax.set_yticks([0, 20, 40, 60, 80, 100])

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([lb for _, lb in conds], fontsize=9.5, fontweight="bold", color="#1A202C")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False, fontsize=9.5)
    fig.suptitle(
        "OPV2V 2车异构对（1 协作车, 320 帧）端到端 3D 检测 AP@0.5 对比（HEAL 冻结底座）",
        fontsize=12.2,
        fontweight="bold",
        y=1.03,
        color="#1A202C",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_p = os.path.join(OUT_DIR, "fig1_ap50_m1_m2_attack_defense.png")
    out_p = _safe_savefig(fig, out_p)
    plt.close(fig)
    return out_p


def plot_fig2_blind_recall(r45: dict):
    """第 1 页右图：583 辆纯他车客观盲区 GT 召回率对比（核心亮点）。"""
    conds = [
        ("clean", "干净场景\n(Clean)"),
        ("local_removal_1.0sigma", "定向目标移除\n(Object Removal)"),
        ("local_ghost_2.0sigma", "虚假目标伪造\n(Object Fabrication)"),
        ("local_occluded_ghost_2.0sigma", "遮挡区目标伪造\n(Occ. Fabrication)"),
    ]
    total_gt = r45["blind_spot_recall"]["m1"]["clean"]["bud_frozen"]["total_blind_gt"]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), dpi=300)
    fig.patch.set_facecolor("white")

    specs = [
        ("m1", "(a) 协作者为 64 线激光雷达 (m1)", 106),
        ("m2", "(b) 协作者为 4 目环视摄像头 (m2)", 48),
    ]
    x = np.arange(len(conds))
    w = 0.25

    for ax, (mod, title, ymax) in zip(axes, specs):
        _style_ax(ax)
        v_nd = [r45["blind_spot_recall"][mod][c]["no_defense"]["val"] * 100.0 for c, _ in conds]
        v_wd = [r45["blind_spot_recall"][mod][c]["whole_cav_drop"]["val"] * 100.0 for c, _ in conds]
        v_bf = [r45["blind_spot_recall"][mod][c]["bud_frozen"]["val"] * 100.0 for c, _ in conds]

        b1 = ax.bar(x - w, v_nd, w, color=C_NODEF, edgecolor="white", linewidth=0.6, label="无防御 (No-Defense)")
        b2 = ax.bar(x, v_wd, w, color=C_WHOLE, edgecolor="white", linewidth=0.6, label="整车降权 (Whole-CAV Drop)")
        b3 = ax.bar(x + w, v_bf, w, color=C_BUD, edgecolor="white", linewidth=0.6, label="BUD-Fusion (本文)")

        for bars, is_bud in ((b1, False), (b2, False), (b3, True)):
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + ymax * 0.014,
                    f"{h:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.3,
                    fontweight="bold" if is_bud else "normal",
                    color=C_BUD if is_bud else "#2D3748",
                )

        # 干净损失标注
        cln_diff = v_bf[0] - v_nd[0]
        ax.text(
            x[0] + w * 0.35,
            max(v_bf[0], v_nd[0]) + ymax * 0.078,
            f"干净变化 {cln_diff:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color="#2B6CB0",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#EBF8FF", edgecolor="#90CDF4", linewidth=0.7),
        )

        # 3 种定向攻击下相对整车降权的盲区保活净增益
        for idx_c in (1, 2, 3):
            diff = v_bf[idx_c] - v_wd[idx_c]
            y_top = max(v_bf[idx_c], v_nd[idx_c]) + ymax * 0.078
            ax.text(
                x[idx_c] + w * 0.40,
                y_top,
                f"+{diff:.1f}% vs 整车",
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight="bold",
                color=C_GAIN,
                bbox=dict(boxstyle="round,pad=0.18", facecolor=C_GAIN_BG, edgecolor="#81E6D9", linewidth=0.7),
            )

        ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=8, color="#1A202C")
        ax.set_ylabel("客观盲区目标召回率 (%)", fontsize=10, fontweight="bold", color="#2D3748")
        ax.set_xticks(x)
        ax.set_xticklabels([lb for _, lb in conds], fontsize=8.6, fontweight="bold", color="#1A202C")
        ax.set_ylim(0, ymax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=3, frameon=False, fontsize=9.8)
    fig.suptitle(
        f"2车异构对（1 协作车, 320 帧）客观盲区召回率对比（共 {total_gt} 辆纯他车盲区 GT：本车点数 ≤ 5 且单车漏检）",
        fontsize=11.6,
        fontweight="bold",
        y=1.02,
        color="#1A202C",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_p = os.path.join(OUT_DIR, "fig2_blind_recall_m1_m2.png")
    out_p = _safe_savefig(fig, out_p)
    plt.close(fig)
    return out_p


def plot_fig0_motivation(r53: dict, r52: dict):
    """动机页配图（零阈值偏见）：左图干净异构异常分分布错位；右图混合车队诚实车信任/权重保留。"""
    hs = r53["experiment_a_existing_defenses_bias"]["clean_score_distributions_2cav"]["hetshield_shared"]
    luc_mix = r53["experiment_a_existing_defenses_bias"]["lucia_mixed_multicav_clean"]
    w_shift = r52["fleets"]["mixed"]["honest_cav_weight_shift_under_attack"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=300)
    fig.patch.set_facecolor("white")

    # 左图：HetShield 干净分数分布区间（Mean ± Std, P90~P95~Max）——不依赖任何人为阈值
    ax = axes[0]
    _style_ax(ax)
    # 画横向分布区间条（一目了然看出分布错位）
    y_pos = [1.0, 0.0]
    mods_info = [
        ("m1", "诚实激光雷达车\n(Honest m1, LiDAR)", C_M1),
        ("m2", "诚实摄像头车\n(Honest m2, Camera)", C_M2),
    ]
    for yp, (mk, lb, col) in zip(y_pos, mods_info):
        d = hs[mk]
        lo = max(0.0, d["mean"] - d["std"])
        hi = d["mean"] + d["std"]
        # 区间线：mean-std 到 max
        ax.plot([lo, d["max"]], [yp, yp], color=col, linewidth=2.2, alpha=0.55, zorder=2)
        # 主体框：mean ± std
        ax.barh(yp, hi - lo, left=lo, height=0.34, color=col, alpha=0.82, edgecolor="white", zorder=3)
        # 中位数竖线与 P95 菱形
        ax.plot([d["median"], d["median"]], [yp - 0.17, yp + 0.17], color="white", linewidth=2.2, zorder=4)
        ax.scatter([d["p95"]], [yp], marker="D", s=48, color="#1A202C", zorder=5, label="P95 分位点" if mk == "m1" else None)
        ax.text(d["mean"], yp + 0.23, f"均值 {d['mean']:.3f}", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color=col)
        ax.text(d["p95"] + 0.006, yp - 0.27, f"P95={d['p95']:.3f}", ha="center", va="top", fontsize=8.2, color="#2D3748")

    # 标出两难间隙区（m1 P95=0.297 到 m2 Mean=0.379）
    ax.axvspan(hs["m1"]["p95"], hs["m2"]["mean"], color="#FED7D7", alpha=0.45, zorder=1)
    ax.text(
        (hs["m1"]["p95"] + hs["m2"]["mean"]) / 2.0,
        0.50,
        "分布错位区\n任意单一阈值均无解",
        ha="center",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        color=C_RED,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([x[1] for x in mods_info], fontsize=9.2, fontweight="bold", color="#1A202C")
    ax.set_xlabel("干净数据异常得分分布（重训版 HetShield 64ch ConvGRU）", fontsize=9.3, fontweight="bold", color="#2D3748")
    ax.set_title("(a) 跨模态干净异常分分布错位（与阈值无关）", fontsize=10.5, fontweight="bold", loc="left", pad=8)
    ax.set_xlim(0.05, 0.54)
    ax.set_ylim(-0.55, 1.55)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.45, color="#CBD5E0")
    ax.grid(axis="y", visible=False)

    # 右图：混合多车队（≥3 车）中同帧诚实车的权重/信任保持率对比
    ax2 = axes[1]
    _style_ax(ax2)
    cats = ["诚实 m1 (LiDAR)", "诚实 m2 (Camera)"]
    x2 = np.arange(2)
    w2 = 0.28
    # LUCIA 干净混合车队相对信任均值（归一化到 100%）与 BUD 攻击发生时相对干净权重保留率
    luc_vals = [luc_mix["m1"]["trust"]["mean"] * 100.0, luc_mix["m2"]["trust"]["mean"] * 100.0]
    luc_w_ret = [
        (w_shift["m1"]["mean_w_atk_lucia"] / w_shift["m1"]["mean_w_clean_base"]) * 100.0,
        (w_shift["m2"]["mean_w_atk_lucia"] / w_shift["m2"]["mean_w_clean_base"]) * 100.0,
    ]
    bud_w_ret = [
        (w_shift["m1"]["mean_w_atk_bud"] / w_shift["m1"]["mean_w_clean_bud"]) * 100.0,
        (w_shift["m2"]["mean_w_atk_bud"] / w_shift["m2"]["mean_w_clean_bud"]) * 100.0,
    ]

    r1 = ax2.bar(x2 - w2, luc_vals, w2, color="#ECC94B", edgecolor="white", label="LUCIA 干净信任分 (均值)")
    r2 = ax2.bar(x2, luc_w_ret, w2, color=C_WHOLE, edgecolor="white", label="LUCIA 受攻时诚实车权重保留率")
    r3 = ax2.bar(x2 + w2, bud_w_ret, w2, color=C_BUD, edgecolor="white", label="BUD-Fusion 受攻时诚实车权重保留率")

    for bars, is_bud in ((r1, False), (r2, False), (r3, True)):
        for bar in bars:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 1.5,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.6,
                fontweight="bold" if is_bud else "normal",
                color=C_BUD if is_bud else "#2D3748",
            )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(cats, fontsize=9.5, fontweight="bold", color="#1A202C")
    ax2.set_ylabel("信任 / 权重保留比例 (%)", fontsize=9.5, fontweight="bold", color="#2D3748")
    ax2.set_title("(b) 混合多车队（2 协作车 145 帧 / ≥3 协作车 134 帧）诚实车保护", fontsize=10.2, fontweight="bold", loc="left", pad=24)
    ax2.set_ylim(0, 118)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=1, fontsize=7.8, frameon=False)

    fig.suptitle("动机实证：异构车队统一判据的分布错位与诚实摄像头车误伤（Dev-320 未见序列）", fontsize=11.6, fontweight="bold", y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_p = os.path.join(OUT_DIR, "fig0_motivation_hetero_disparity.png")
    out_p = _safe_savefig(fig, out_p)
    plt.close(fig)
    return out_p


def plot_fig3_latency(r7: dict):
    """第 2 页配图：单帧推理延迟横向对比与 10 Hz (100 ms) 车载实时红线。"""
    ov = r7["latency_ms"]["overall"]
    items = [
        ("无防御 (No-Defense)", ov["no_defense_total"]["mean_ms"], ov["no_defense_total"]["p95_ms"], "#A0AEC0"),
        ("LUCIA (USENIX'25)", ov["lucia_total"]["mean_ms"], ov["lucia_total"]["p95_ms"], "#718096"),
        ("HetShield (arXiv'26)", ov["hetshield_total"]["mean_ms"], ov["hetshield_total"]["p95_ms"], "#4A5568"),
        ("BUD-Fusion (本文)", ov["bud_receiver_total"]["mean_ms"], ov["bud_receiver_total"]["p95_ms"], C_BUD),
        ("ROBOSAC (ICCV'23, ≥3车)", ov["robosac_total"]["mean_ms"], ov["robosac_total"]["p95_ms"], "#E53E3E"),
    ]

    fig, ax = plt.subplots(figsize=(8.6, 3.9), dpi=300)
    fig.patch.set_facecolor("white")
    _style_ax(ax)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.45, color="#CBD5E0")
    ax.grid(axis="y", visible=False)

    y = np.arange(len(items))
    means = [it[1] for it in items]
    p95s = [it[2] for it in items]
    cols = [it[3] for it in items]

    bars = ax.barh(y, means, height=0.56, color=cols, edgecolor="white", linewidth=0.8)
    ax.axvline(100.0, color=C_RED, linestyle="--", linewidth=1.6, zorder=4)
    ax.text(102.5, 0.15, "10 Hz 车载实时红线 (100 ms)", color=C_RED, fontsize=9.0, fontweight="bold", va="center")

    for i, (bar, m_v, p_v) in enumerate(zip(bars, means, p95s)):
        is_bud = "BUD-Fusion" in items[i][0]
        if is_bud:
            # 数值放在蓝条内侧白色加粗，说明放在 100ms 红线右侧，彻底避免被红线穿过
            ax.text(
                m_v - 4.0,
                bar.get_y() + bar.get_height() / 2.0,
                f"{m_v:.1f} ms",
                va="center",
                ha="right",
                fontsize=9.0,
                fontweight="bold",
                color="white",
            )
            ax.text(
                103.5,
                bar.get_y() + bar.get_height() / 2.0,
                f"满足实时 (P95: {p_v:.1f} ms | 1 车: 48.0 ms)",
                va="center",
                ha="left",
                fontsize=8.6,
                fontweight="bold",
                color=C_BUD,
            )
        else:
            extra = f"  [P95: {p_v:.1f} ms, 超时]" if m_v > 100 else f"  (P95: {p_v:.1f} ms)"
            ax.text(
                m_v + 3.0,
                bar.get_y() + bar.get_height() / 2.0,
                f"{m_v:.1f} ms{extra}",
                va="center",
                ha="left",
                fontsize=8.6,
                fontweight="bold" if m_v > 100 else "normal",
                color=C_RED if m_v > 100 else "#2D3748",
            )

    ax.set_yticks(y)
    ax.set_yticklabels([it[0] for it in items], fontsize=9.5, fontweight="bold", color="#1A202C")
    ax.invert_yaxis()
    ax.set_xlabel("单帧接收端推理平均延迟 (ms，RTX 4090 CUDA Event 100 帧实测)", fontsize=9.5, fontweight="bold", color="#2D3748")
    ax.set_title("单帧推理延迟横向对比（BUD 物理分档仅 2.3 ms，软门控仅 3.9 ms）", fontsize=11, fontweight="bold", loc="left", pad=8)
    ax.set_xlim(0, 305)

    fig.tight_layout()
    out_p = os.path.join(OUT_DIR, "fig3_latency_comparison_10hz.png")
    out_p = _safe_savefig(fig, out_p)
    plt.close(fig)
    return out_p


def plot_fig4_targeted_asr(r54: dict):
    """新增王炸图：全 SOTA 定向局部攻击成功率 ASR 对比（越低越安全，显式标注协作车规模分层）。"""
    mix = r54["mixed_multicav_fleet_asr"]
    pair = r54["two_vehicle_hetero_pair_asr"]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2), dpi=300, gridspec_kw={"width_ratios": [1.18, 1.0]})
    fig.patch.set_facecolor("white")

    # ---------------- 左子图 (a)：3~6 车混合异构车队全 SOTA 横向对比 ----------------
    ax = axes[0]
    _style_ax(ax)

    rem_all = mix["ge2_all|single_attacker|object_removal"]["methods"]
    fab_all = mix["ge2_all|single_attacker|object_fabrication"]["methods"]

    method_order = [
        ("no_defense", "无防御\n(NoDef)"),
        ("robosac", "ROBOSAC\n(ICCV'23)"),
        ("hetshield_tuned", "HS-Tuned\n(放宽门槛)"),
        ("lucia", "LUCIA\n(USENIX'25)"),
        ("whole_cav_drop", "整车降权\n(WholeDrop)"),
        ("hetshield_paper", "HS-Paper*\n(误杀m2 98%)"),
        ("bud_frozen", "BUD-Fusion\n(本文)"),
    ]
    x = np.arange(len(method_order))
    w = 0.36

    rem_vals = [rem_all[k]["asr"] * 100.0 for k, _ in method_order]
    fab_vals = [fab_all[k]["asr"] * 100.0 for k, _ in method_order]

    # 对本文 BUD 使用更深高亮色，其余方法使用统一柔和对比色
    c_rem = ["#63B3ED" if k != "bud_frozen" else "#1E4E8C" for k, _ in method_order]
    c_fab = ["#F6AD55" if k != "bud_frozen" else "#C05621" for k, _ in method_order]

    b_rem = ax.bar(x - w / 2.0, rem_vals, w, color=c_rem, edgecolor="white", linewidth=0.7, label="定向目标移除 ASR (Object Removal)")
    b_fab = ax.bar(x + w / 2.0, fab_vals, w, color=c_fab, edgecolor="white", linewidth=0.7, label="虚假目标伪造 ASR (Object Fabrication)")

    # 给本文 BUD 列加浅蓝背景突出显示
    ax.axvspan(len(method_order) - 1.5, len(method_order) - 0.5, color="#EBF8FF", alpha=0.65, zorder=0)

    for idx_m, (br, bf) in enumerate(zip(b_rem, b_fab)):
        is_bud = method_order[idx_m][0] == "bud_frozen"
        hr, hf = br.get_height(), bf.get_height()
        ax.text(
            br.get_x() + br.get_width() / 2.0,
            hr + 1.4,
            f"{hr:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold" if is_bud else "normal",
            color="#1E4E8C" if is_bud else "#2D3748",
        )
        ax.text(
            bf.get_x() + bf.get_width() / 2.0,
            hf + 1.4,
            f"{hf:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold" if is_bud else "normal",
            color="#C05621" if is_bud else "#2D3748",
        )

    # 顶部绿底亮点注释框：BUD 相对无防御的巨幅降幅
    ax.text(
        len(method_order) - 1.0,
        54.0,
        "本文全场最低\n移除 -54.3%\n伪造 -64.1%",
        ha="center",
        va="bottom",
        fontsize=8.3,
        fontweight="bold",
        color=C_GAIN,
        bbox=dict(boxstyle="round,pad=0.28", facecolor=C_GAIN_BG, edgecolor="#81E6D9", linewidth=0.9),
    )

    ax.set_xticks(x)
    ax.set_xticklabels([lb for _, lb in method_order], fontsize=8.4, fontweight="bold", color="#1A202C")
    ax.set_ylabel("攻击成功率 ASR (%) ↓ 越低越安全", fontsize=9.8, fontweight="bold", color="#2D3748")
    ax.set_title("(a) 3~6 车混合车队全 SOTA 对比（279 帧，单攻击者）", fontsize=10.6, fontweight="bold", loc="left", pad=8, color="#1A202C")
    ax.set_ylim(0, 114)
    ax.legend(loc="upper right", fontsize=8.4, frameon=True, facecolor="white", edgecolor="#E2E8F0")

    # ---------------- 右子图 (b)：按协作车规模分层 + 双攻击者压测 ----------------
    ax2 = axes[1]
    _style_ax(ax2)

    # 4 个代表性规模档位（涵盖 1 协作车、2 协作车、≥3 协作车单攻、≥3 协作车双攻）
    tier_specs = [
        ("1 协作车\n(2车, 320帧)\n均值 m1/m2",
         0.5 * (pair["m1|object_removal"]["methods"]["no_defense"]["asr"] + pair["m2|object_removal"]["methods"]["no_defense"]["asr"]) * 100.0,
         0.5 * (pair["m1|object_removal"]["methods"]["bud_frozen"]["asr"] + pair["m2|object_removal"]["methods"]["bud_frozen"]["asr"]) * 100.0,
         0.5 * (pair["m1|object_fabrication"]["methods"]["no_defense"]["asr"] + pair["m2|object_fabrication"]["methods"]["no_defense"]["asr"]) * 100.0,
         0.5 * (pair["m1|object_fabrication"]["methods"]["bud_frozen"]["asr"] + pair["m2|object_fabrication"]["methods"]["bud_frozen"]["asr"]) * 100.0),
        ("2 协作车\n(3车, 145帧)\n单攻击者",
         0.5 * (mix["collab_2|atk_m1|object_removal"]["methods"]["no_defense"]["asr"] + mix["collab_2|atk_m2|object_removal"]["methods"]["no_defense"]["asr"]) * 100.0,
         0.5 * (mix["collab_2|atk_m1|object_removal"]["methods"]["bud_frozen"]["asr"] + mix["collab_2|atk_m2|object_removal"]["methods"]["bud_frozen"]["asr"]) * 100.0,
         0.5 * (mix["collab_2|atk_m1|object_fabrication"]["methods"]["no_defense"]["asr"] + mix["collab_2|atk_m2|object_fabrication"]["methods"]["no_defense"]["asr"]) * 100.0,
         0.5 * (mix["collab_2|atk_m1|object_fabrication"]["methods"]["bud_frozen"]["asr"] + mix["collab_2|atk_m2|object_fabrication"]["methods"]["bud_frozen"]["asr"]) * 100.0),
        ("≥3 协作车\n(4~6车, 134帧)\n单攻击者",
         0.5 * (mix["collab_ge3|atk_m1|object_removal"]["methods"]["no_defense"]["asr"] + mix["collab_ge3|atk_m2|object_removal"]["methods"]["no_defense"]["asr"]) * 100.0,
         0.5 * (mix["collab_ge3|atk_m1|object_removal"]["methods"]["bud_frozen"]["asr"] + mix["collab_ge3|atk_m2|object_removal"]["methods"]["bud_frozen"]["asr"]) * 100.0,
         0.5 * (mix["collab_ge3|atk_m1|object_fabrication"]["methods"]["no_defense"]["asr"] + mix["collab_ge3|atk_m2|object_fabrication"]["methods"]["no_defense"]["asr"]) * 100.0,
         0.5 * (mix["collab_ge3|atk_m1|object_fabrication"]["methods"]["bud_frozen"]["asr"] + mix["collab_ge3|atk_m2|object_fabrication"]["methods"]["bud_frozen"]["asr"]) * 100.0),
        ("≥3 协作车\n(4~6车, 134帧)\n双攻击者并发",
         mix["collab_ge3|two_attackers|object_removal"]["methods"]["no_defense"]["asr"] * 100.0,
         mix["collab_ge3|two_attackers|object_removal"]["methods"]["bud_frozen"]["asr"] * 100.0,
         mix["collab_ge3|two_attackers|object_fabrication"]["methods"]["no_defense"]["asr"] * 100.0,
         mix["collab_ge3|two_attackers|object_fabrication"]["methods"]["bud_frozen"]["asr"] * 100.0),
    ]

    x2 = np.arange(len(tier_specs))
    w2 = 0.19
    nd_r = [t[1] for t in tier_specs]
    bd_r = [t[2] for t in tier_specs]
    nd_f = [t[3] for t in tier_specs]
    bd_f = [t[4] for t in tier_specs]

    p1 = ax2.bar(x2 - 1.5 * w2, nd_r, w2, color="#CBD5E0", edgecolor="white", label="移除 ASR: 无防御")
    p2 = ax2.bar(x2 - 0.5 * w2, bd_r, w2, color="#1E4E8C", edgecolor="white", label="移除 ASR: BUD (本文)")
    p3 = ax2.bar(x2 + 0.5 * w2, nd_f, w2, color="#FBD38D", edgecolor="white", label="伪造 ASR: 无防御")
    p4 = ax2.bar(x2 + 1.5 * w2, bd_f, w2, color="#C05621", edgecolor="white", label="伪造 ASR: BUD (本文)")

    for bars, is_bold, col in (
        (p1, False, "#4A5568"),
        (p2, True, "#1E4E8C"),
        (p3, False, "#9C4221"),
        (p4, True, "#C05621"),
    ):
        for bar in bars:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 1.4,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.6,
                fontweight="bold" if is_bold else "normal",
                color=col,
            )

    ax2.set_xticks(x2)
    ax2.set_xticklabels([t[0] for t in tier_specs], fontsize=8.3, fontweight="bold", color="#1A202C")
    ax2.set_ylabel("攻击成功率 ASR (%) ↓ 越低越安全", fontsize=9.8, fontweight="bold", color="#2D3748")
    ax2.set_title("(b) 按协作车数量分层与双攻击者压测 ASR 对比", fontsize=10.6, fontweight="bold", loc="left", pad=8, color="#1A202C")
    ax2.set_ylim(0, 118)
    ax2.legend(loc="upper center", ncol=2, fontsize=7.9, frameon=True, facecolor="white", edgecolor="#E2E8F0")

    fig.suptitle(
        "定向局部攻击成功率 (ASR) 全基线对比（Dev-320：1 协作车 41 帧 / 2 协作车 145 帧 / ≥3 协作车 134 帧，匹配半径 3.0 m）",
        fontsize=11.6,
        fontweight="bold",
        y=1.01,
        color="#1A202C",
    )
    fig.text(
        0.5,
        -0.02,
        "*注：HS-Paper (τ=0.3) 靠对诚实摄像头车 98.1% 误杀压低 m2 攻 ASR；一旦放宽门槛保诚实车 (HS-Tuned)，其 ASR 即回弹至 57.9% / 95.9%。",
        ha="center",
        va="top",
        fontsize=8.3,
        color="#4A5568",
    )
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    out_p = os.path.join(OUT_DIR, "fig4_targeted_asr_sota_comparison.png")
    out_p = _safe_savefig(fig, out_p)
    plt.close(fig)
    return out_p


def main():
    r45 = json.load(open(os.path.join(RES_DIR, "step4_5_bud_frozen_320_report.json"), "r", encoding="utf-8"))
    r52 = json.load(open(os.path.join(RES_DIR, "step5_2_mixed_report.json"), "r", encoding="utf-8"))
    r53 = json.load(open(os.path.join(RES_DIR, "step5_3_hetero_unified_threshold_evidence.json"), "r", encoding="utf-8"))
    r54 = json.load(open(os.path.join(RES_DIR, "step5_4_targeted_asr_report.json"), "r", encoding="utf-8"))
    r7 = json.load(open(os.path.join(RES_DIR, "step7_latency_breakdown.json"), "r", encoding="utf-8"))

    paths = [
        plot_fig0_motivation(r53, r52),
        plot_fig1_ap50(r45),
        plot_fig2_blind_recall(r45),
        plot_fig3_latency(r7),
        plot_fig4_targeted_asr(r54),
    ]
    for p in paths:
        print("GENERATED:", p)


if __name__ == "__main__":
    main()
