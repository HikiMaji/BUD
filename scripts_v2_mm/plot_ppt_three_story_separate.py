# -*- coding: utf-8 -*-
"""
口径与配色 100% 统一的答辩极简高清独立图（300 DPI）：
- 前 3 张全部严格统一为「2 车异构对（1 协作车, 320 帧）」：
  1. fig_story1_2cav_clean.png ：无攻击场景检测精度对比（无防御 vs 本方案）
  2. fig_story2_2cav_asr.png   ：定向攻击成功率对比（无防御 vs 本方案）
  3. fig_story3_2cav_blind.png ：虚假目标伪造下真实车辆盲区召回率（无攻击上限 vs 整车拒绝 vs 本方案）
- 第 4 张单列「混合多车队（3~6 车, 279 帧，按协作车数量分层）」：
  4. fig_story4_mixed_fleet.png：3 车（2 协作车）与 4~6 车（≥3 协作车）定向攻击成功率对比（无防御 vs 本方案）

全局锁定颜色语义（4 张图完全一致，绝不串色）：
- 柔和冷灰 #94A3B8：无防御 / 无攻击上限（基线）
- 暖琥珀橙 #ED8936：整车拒绝
- 东南深蓝 #1E4E8C：本方案（BUD-Fusion）
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES_DIR = r"E:\BUD\results_rq1_v2_mm"
OUT_DIR = r"E:\BUD\ppt_figures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.edgecolor"] = "#CBD5E0"
plt.rcParams["axes.linewidth"] = 0.9

# 全局锁定统一配色
C_BASE = "#94A3B8"      # 无防御 / 无攻击上限
C_WHOLE = "#ED8936"     # 整车拒绝
C_BUD = "#1E4E8C"       # 本方案


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45, color="#CBD5E0")
    ax.set_axisbelow(True)


def _safe_savefig(fig, out_p: str) -> str:
    tmp_p = out_p + ".tmp.png"
    fig.savefig(tmp_p, bbox_inches="tight", dpi=300)
    try:
        os.replace(tmp_p, out_p)
        return out_p
    except OSError:
        alt_p = out_p.replace(".png", "_updated.png")
        os.replace(tmp_p, alt_p)
        return alt_p


def _label_bars(ax, bars, is_bud: bool):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 1.4,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10.2,
            fontweight="bold" if is_bud else "normal",
            color=C_BUD if is_bud else "#2D3748",
        )


def plot_story1_2cav_clean(r45: dict) -> str:
    """图 1（2 车异构对）：无攻击场景下，无防御 vs 本方案 AP@0.5 对比。"""
    groups = [
        (
            "雷达协作车\n(m1)",
            r45["ap_results"]["m1"]["clean"]["no_defense"]["ap50"]["val"] * 100.0,
            r45["ap_results"]["m1"]["clean"]["bud_frozen"]["ap50"]["val"] * 100.0,
        ),
        (
            "视觉协作车\n(m2)",
            r45["ap_results"]["m2"]["clean"]["no_defense"]["ap50"]["val"] * 100.0,
            r45["ap_results"]["m2"]["clean"]["bud_frozen"]["ap50"]["val"] * 100.0,
        ),
    ]

    fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=300)
    fig.patch.set_facecolor("white")
    _style_ax(ax)

    x = np.arange(len(groups))
    w = 0.28
    b1 = ax.bar(x - w / 2.0, [g[1] for g in groups], w, color=C_BASE, edgecolor="white", label="无防御")
    b2 = ax.bar(x + w / 2.0, [g[2] for g in groups], w, color=C_BUD, edgecolor="white", label="本方案")

    _label_bars(ax, b1, False)
    _label_bars(ax, b2, True)

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=10.5, fontweight="bold", color="#1A202C")
    ax.set_ylabel("3D 检测精度 AP@0.5 (%)", fontsize=10.5, fontweight="bold", color="#2D3748")
    ax.set_title("无攻击场景：检测精度对比（2 车异构对）", fontsize=12.0, fontweight="bold", pad=26, color="#1A202C")
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=2, frameon=False, fontsize=10.2)

    fig.tight_layout()
    out_p = _safe_savefig(fig, os.path.join(OUT_DIR, "fig_story1_2cav_clean.png"))
    plt.close(fig)
    return out_p


def plot_story2_2cav_asr(r54: dict) -> str:
    """图 2（2 车异构对）：定向攻击场景下，无防御 vs 本方案攻击成功率对比。"""
    pair = r54["two_vehicle_hetero_pair_asr"]
    groups = [
        (
            "雷达车\n目标移除",
            pair["m1|object_removal"]["methods"]["no_defense"]["asr"] * 100.0,
            pair["m1|object_removal"]["methods"]["bud_frozen"]["asr"] * 100.0,
        ),
        (
            "视觉车\n目标移除",
            pair["m2|object_removal"]["methods"]["no_defense"]["asr"] * 100.0,
            pair["m2|object_removal"]["methods"]["bud_frozen"]["asr"] * 100.0,
        ),
        (
            "雷达车\n目标伪造",
            pair["m1|object_fabrication"]["methods"]["no_defense"]["asr"] * 100.0,
            pair["m1|object_fabrication"]["methods"]["bud_frozen"]["asr"] * 100.0,
        ),
        (
            "视觉车\n目标伪造",
            pair["m2|object_fabrication"]["methods"]["no_defense"]["asr"] * 100.0,
            pair["m2|object_fabrication"]["methods"]["bud_frozen"]["asr"] * 100.0,
        ),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=300)
    fig.patch.set_facecolor("white")
    _style_ax(ax)

    x = np.arange(len(groups))
    w = 0.32
    b1 = ax.bar(x - w / 2.0, [g[1] for g in groups], w, color=C_BASE, edgecolor="white", label="无防御")
    b2 = ax.bar(x + w / 2.0, [g[2] for g in groups], w, color=C_BUD, edgecolor="white", label="本方案")

    _label_bars(ax, b1, False)
    _label_bars(ax, b2, True)

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=10.2, fontweight="bold", color="#1A202C")
    ax.set_ylabel("攻击成功率 ASR (%) ↓", fontsize=10.5, fontweight="bold", color="#2D3748")
    ax.set_title("定向攻击场景：攻击成功率对比（2 车异构对）", fontsize=12.0, fontweight="bold", pad=26, color="#1A202C")
    ax.set_ylim(0, 114)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=2, frameon=False, fontsize=10.2)

    fig.tight_layout()
    out_p = _safe_savefig(fig, os.path.join(OUT_DIR, "fig_story2_2cav_asr.png"))
    plt.close(fig)
    return out_p


def plot_story3_2cav_blind(r45: dict) -> str:
    """图 3（2 车异构对）：虚假目标伪造下，无攻击上限 vs 整车拒绝 vs 本方案真实盲区召回率。"""
    groups = [
        (
            "雷达车\n目标伪造",
            r45["blind_spot_recall"]["m1"]["clean"]["no_defense"]["val"] * 100.0,
            r45["blind_spot_recall"]["m1"]["local_ghost_2.0sigma"]["whole_cav_drop"]["val"] * 100.0,
            r45["blind_spot_recall"]["m1"]["local_ghost_2.0sigma"]["bud_frozen"]["val"] * 100.0,
        ),
        (
            "雷达车\n遮挡区伪造",
            r45["blind_spot_recall"]["m1"]["clean"]["no_defense"]["val"] * 100.0,
            r45["blind_spot_recall"]["m1"]["local_occluded_ghost_2.0sigma"]["whole_cav_drop"]["val"] * 100.0,
            r45["blind_spot_recall"]["m1"]["local_occluded_ghost_2.0sigma"]["bud_frozen"]["val"] * 100.0,
        ),
        (
            "视觉车\n目标伪造",
            r45["blind_spot_recall"]["m2"]["clean"]["no_defense"]["val"] * 100.0,
            r45["blind_spot_recall"]["m2"]["local_ghost_2.0sigma"]["whole_cav_drop"]["val"] * 100.0,
            r45["blind_spot_recall"]["m2"]["local_ghost_2.0sigma"]["bud_frozen"]["val"] * 100.0,
        ),
    ]

    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=300)
    fig.patch.set_facecolor("white")
    _style_ax(ax)

    x = np.arange(len(groups))
    w = 0.25
    b1 = ax.bar(x - w, [g[1] for g in groups], w, color=C_BASE, edgecolor="white", label="无攻击上限")
    b2 = ax.bar(x, [g[2] for g in groups], w, color=C_WHOLE, edgecolor="white", label="整车拒绝")
    b3 = ax.bar(x + w, [g[3] for g in groups], w, color=C_BUD, edgecolor="white", label="本方案")

    _label_bars(ax, b1, False)
    _label_bars(ax, b2, False)
    _label_bars(ax, b3, True)

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=10.2, fontweight="bold", color="#1A202C")
    ax.set_ylabel("真实车辆盲区召回率 (%)", fontsize=10.5, fontweight="bold", color="#2D3748")
    ax.set_title("虚假目标伪造下：真实盲区召回率（2 车异构对）", fontsize=12.0, fontweight="bold", pad=26, color="#1A202C")
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=3, frameon=False, fontsize=10.2)

    fig.tight_layout()
    out_p = _safe_savefig(fig, os.path.join(OUT_DIR, "fig_story3_2cav_blind.png"))
    plt.close(fig)
    return out_p


def _pooled_asr(entry_a: dict, entry_b: dict, method: str) -> float:
    """按有效目标数精确合并 m1/m2 单攻击者 ASR。"""
    succ = entry_a["methods"][method]["succ"] + entry_b["methods"][method]["succ"]
    tot = entry_a["total_valid_targets"] + entry_b["total_valid_targets"]
    return (float(succ) / float(tot)) * 100.0


def plot_story4_mixed_fleet(r54: dict) -> str:
    """图 4（单列混合多车队）：按协作车数量分层对比无防御 vs 本方案定向攻击成功率。"""
    mix = r54["mixed_multicav_fleet_asr"]
    groups = [
        (
            "3 车 (2 协作车)\n目标移除",
            _pooled_asr(mix["collab_2|atk_m1|object_removal"], mix["collab_2|atk_m2|object_removal"], "no_defense"),
            _pooled_asr(mix["collab_2|atk_m1|object_removal"], mix["collab_2|atk_m2|object_removal"], "bud_frozen"),
        ),
        (
            "3 车 (2 协作车)\n目标伪造",
            _pooled_asr(mix["collab_2|atk_m1|object_fabrication"], mix["collab_2|atk_m2|object_fabrication"], "no_defense"),
            _pooled_asr(mix["collab_2|atk_m1|object_fabrication"], mix["collab_2|atk_m2|object_fabrication"], "bud_frozen"),
        ),
        (
            "4~6 车 (≥3 协作车)\n目标移除",
            _pooled_asr(mix["collab_ge3|atk_m1|object_removal"], mix["collab_ge3|atk_m2|object_removal"], "no_defense"),
            _pooled_asr(mix["collab_ge3|atk_m1|object_removal"], mix["collab_ge3|atk_m2|object_removal"], "bud_frozen"),
        ),
        (
            "4~6 车 (≥3 协作车)\n目标伪造",
            _pooled_asr(mix["collab_ge3|atk_m1|object_fabrication"], mix["collab_ge3|atk_m2|object_fabrication"], "no_defense"),
            _pooled_asr(mix["collab_ge3|atk_m1|object_fabrication"], mix["collab_ge3|atk_m2|object_fabrication"], "bud_frozen"),
        ),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
    fig.patch.set_facecolor("white")
    _style_ax(ax)

    x = np.arange(len(groups))
    w = 0.32
    b1 = ax.bar(x - w / 2.0, [g[1] for g in groups], w, color=C_BASE, edgecolor="white", label="无防御")
    b2 = ax.bar(x + w / 2.0, [g[2] for g in groups], w, color=C_BUD, edgecolor="white", label="本方案")

    _label_bars(ax, b1, False)
    _label_bars(ax, b2, True)

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=9.8, fontweight="bold", color="#1A202C")
    ax.set_ylabel("攻击成功率 ASR (%) ↓", fontsize=10.5, fontweight="bold", color="#2D3748")
    ax.set_title("混合多车队（3~6 车）：定向攻击成功率对比", fontsize=12.0, fontweight="bold", pad=26, color="#1A202C")
    ax.set_ylim(0, 114)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=2, frameon=False, fontsize=10.2)

    fig.tight_layout()
    out_p = _safe_savefig(fig, os.path.join(OUT_DIR, "fig_story4_mixed_fleet.png"))
    plt.close(fig)
    return out_p


def main():
    r45 = json.load(open(os.path.join(RES_DIR, "step4_5_bud_frozen_320_report.json"), "r", encoding="utf-8"))
    r54 = json.load(open(os.path.join(RES_DIR, "step5_4_targeted_asr_report.json"), "r", encoding="utf-8"))

    for p in (
        plot_story1_2cav_clean(r45),
        plot_story2_2cav_asr(r54),
        plot_story3_2cav_blind(r45),
        plot_story4_mixed_fleet(r54),
    ):
        print("GENERATED:", p)


if __name__ == "__main__":
    main()
