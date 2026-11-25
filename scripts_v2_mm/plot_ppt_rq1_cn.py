# -*- coding: utf-8 -*-
"""
PPT 专用中文横排多联图绘制脚本（1:1 复刻学术答辩 PPT 清爽科研配色与排版）。

输入：本地已验证的 Dev-320 全维度评测 JSON（step4_3_bud_fusion_320_report.json），绝不硬编码任何数据。
输出（至 artifact 根目录，300 DPI 防裁边）：
  1. 16:9 横排 5 联总图：fig_ppt_rq1_5panel_cn.png
  2. 5 张独立高清小图：
     - fig_ppt_rq1_a_local_ap_cn.png
     - fig_ppt_rq1_b_blind_recall_cn.png
     - fig_ppt_rq1_c_gate_selectivity_cn.png
     - fig_ppt_rq1_d_blind_fpr_cn.png
     - fig_ppt_rq1_e_budget_curve_cn.png
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np

BASE_DIR = r"C:\Users\Hiki\.gemini\antigravity\brain\be230015-873c-49a8-9080-587e2bed03c6"
REPORT_JSON = os.path.join(BASE_DIR, "scratch", "step4_3_bud_fusion_320_report.json")
OUT_DIR = BASE_DIR

# 莫兰迪低饱和学术配色（参考答辩 PPT 清爽蓝/橙/绿/紫/红风格）
C_OURS = "#355C8C"     # 本文 (深科研蓝)
C_WCD = "#D88A3B"      # 整车降权 / Tier2 基线 (暖琥珀橙)
C_NODEF = "#4B936E"    # 无防御 / 保留基线 (松石绿)
C_EGO = "#8D6B94"      # 单车 / 灰紫
C_ATK = "#C35458"      # 攻击区 / 无防御(激光)强对照 (砖红)
C_LEG_GRAY = "#6E7681" # 实心/斜线图例通用中灰


def _setup_rc():
    """配置中文科研字体与极简坐标轴样式。"""
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "Noto Sans SC", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "hatch.linewidth": 0.75,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _draw_dual_bar(ax, x, v1, v2, color, bw=0.33, err1=None, err2=None):
    """在同一分类位置绘制「实心(m1 激光车) + 浅底同色斜线(m2 摄像头)」双柱。"""
    kw = dict(capsize=1.8, error_kw=dict(elinewidth=0.75, ecolor="#333333", capthick=0.75))
    b1 = ax.bar(
        x - bw * 0.54, v1, width=bw * 0.92,
        color=color, edgecolor=color, linewidth=0.7,
        yerr=err1, **(kw if err1 is not None else {}),
        zorder=3,
    )
    b2 = ax.bar(
        x + bw * 0.54, v2, width=bw * 0.92,
        facecolor=mcolors.to_rgba(color, 0.16), edgecolor=color, hatch="////", linewidth=0.8,
        yerr=err2, **(kw if err2 is not None else {}),
        zorder=3,
    )
    return b1[0], b2[0]


def _add_sensor_legend(ax, y_anchor=1.13):
    """参考图同款：用中性灰实心/斜线方块标注协同模态（实心=激光车 m1，斜线=摄像头 m2）。"""
    h = [
        Patch(facecolor=C_LEG_GRAY, edgecolor=C_LEG_GRAY, label="激光车(m1)"),
        Patch(facecolor=mcolors.to_rgba(C_LEG_GRAY, 0.16), edgecolor=C_LEG_GRAY, hatch="////", label="摄像头(m2)"),
    ]
    ax.legend(
        handles=h, loc="upper center", bbox_to_anchor=(0.5, y_anchor),
        ncol=2, frameon=False, handlelength=1.25, handleheight=0.85, columnspacing=1.0, fontsize=8.4,
    )


def _set_sub_title(ax, title: str):
    """使用轴归一化坐标固定 (a)~(e) 子图底部标题纵坐标，确保 5 联图严丝合缝水平齐平。"""
    ax.text(0.5, -0.25, title, transform=ax.transAxes, ha="center", va="top", fontsize=10.2, fontweight="bold", color="#111111")


def _plot_panel_a(ax, report: dict):
    """(a) 局部擦除攻击下 AP@0.5 (%) 对比（4 方法 × 双模态）。"""
    ap = report["ap_results"]
    st = "local_removal_1.0sigma"
    methods = [
        ("bud_fusion", "本文\n方法", C_OURS),
        ("whole_cav_drop", "整车\n降权", C_WCD),
        ("no_defense", "原生\n无防御", C_NODEF),
        ("ego_only", "单车\n检测", C_EGO),
    ]
    xs = np.arange(len(methods))
    for i, (mk, _, col) in enumerate(methods):
        v1 = ap["m1"][st][mk]["ap50"]["val"] * 100.0
        v2 = ap["m2"][st][mk]["ap50"]["val"] * 100.0
        b1, b2 = _draw_dual_bar(ax, xs[i], v1, v2, col)
        if mk == "bud_fusion":
            ax.text(b1.get_x() + b1.get_width() / 2 - 0.02, v1 + 1.8, f"{v1:.1f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color=C_OURS)
            ax.text(b2.get_x() + b2.get_width() / 2 + 0.02, v2 + 1.8, f"{v2:.1f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color=C_OURS)
        elif mk == "whole_cav_drop":
            ax.text(b1.get_x() + b1.get_width() / 2, v1 + 1.8, f"{v1:.1f}", ha="center", va="bottom", fontsize=7.4, color="#8C5214")

    ax.set_xticks(xs)
    ax.set_xticklabels([m[1] for m in methods], fontsize=8.5)
    ax.set_ylabel("AP@0.5 / %", labelpad=3)
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 40, 80, 100])
    _set_sub_title(ax, "(a) 局部擦除检测精度")
    _add_sensor_legend(ax)


def _plot_panel_b(ax, report: dict):
    """(b) 局部擦除攻击下本车盲区召回率 (%) 对比（突出 BUD-Fusion 保留率）。"""
    br = report["blind_spot_recall"]
    st = "local_removal_1.0sigma"
    methods = [
        ("bud_fusion", "本文\n方法", C_OURS),
        ("whole_cav_drop", "整车\n降权", C_WCD),
        ("no_defense", "原生\n无防御", C_NODEF),
        ("ego_only", "单车\n检测", C_EGO),
    ]
    xs = np.arange(len(methods))
    for i, (mk, _, col) in enumerate(methods):
        v1 = br["m1"][st][mk]["val"] * 100.0
        v2 = br["m2"][st][mk]["val"] * 100.0
        b1, b2 = _draw_dual_bar(ax, xs[i], v1, v2, col)
        if mk == "bud_fusion":
            ax.text(b1.get_x() + b1.get_width() / 2 - 0.02, v1 + 1.8, f"{v1:.1f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color=C_OURS)
            ax.text(b2.get_x() + b2.get_width() / 2 + 0.02, v2 + 1.8, f"{v2:.1f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color=C_OURS)
        elif mk == "whole_cav_drop":
            ax.text(b1.get_x() + b1.get_width() / 2 - 0.02, v1 + 1.8, f"{v1:.1f}", ha="center", va="bottom", fontsize=7.2, color="#8C5214")
            ax.text(b2.get_x() + b2.get_width() / 2 + 0.02, v2 + 1.8, f"{v2:.1f}", ha="center", va="bottom", fontsize=7.2, color="#8C5214")
        elif mk == "ego_only":
            ax.text(xs[i], 2.5, "0.0", ha="center", va="bottom", fontsize=7.2, color="#666666")

    ax.set_xticks(xs)
    ax.set_xticklabels([m[1] for m in methods], fontsize=8.5)
    ax.set_ylabel("盲区召回率 / %", labelpad=3)
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 40, 80, 100])
    _set_sub_title(ax, "(b) 局部攻击盲区召回率")
    _add_sensor_legend(ax)


def _plot_panel_c(ax, report: dict):
    """(c) BUD-Fusion 局部门控空间选择性（干净保留 vs 受攻击区精准降权）。"""
    sp = report["spatial_weight_selectivity"]
    groups = [
        ("clean_valid_gate", "干净\n全视野", C_OURS, True),
        ("local_removal_gate_clean_region", "擦除未\n攻击区", C_NODEF, False),
        ("local_removal_gate_attacked", "擦除\n攻击区", C_ATK, True),
        ("local_ghost_gate_attacked", "鬼影\n攻击区", C_WCD, True),
    ]
    xs = np.arange(len(groups))
    for i, (gk, _, col, show_txt) in enumerate(groups):
        v1 = sp["m1"][gk]["val"]
        v2 = sp["m2"][gk]["val"]
        b1, b2 = _draw_dual_bar(ax, xs[i], v1, v2, col)
        if show_txt:
            tc = C_OURS if i == 0 else ("#9B2B30" if i == 2 else "#8C5214")
            ax.text(b1.get_x() + b1.get_width() / 2 - 0.04, v1 + 0.025, f"{v1:.2f}", ha="center", va="bottom", fontsize=7.0, fontweight="bold" if i != 3 else "normal", color=tc)
            ax.text(b2.get_x() + b2.get_width() / 2 + 0.04, v2 + 0.025, f"{v2:.2f}", ha="center", va="bottom", fontsize=7.0, fontweight="bold" if i != 3 else "normal", color=tc)

    ax.set_xticks(xs)
    ax.set_xticklabels([g[1] for g in groups], fontsize=8.3)
    ax.set_ylabel("局部信任门控值", labelpad=3)
    ax.set_ylim(0, 1.20)
    ax.set_yticks([0.0, 0.5, 1.0])
    _set_sub_title(ax, "(c) 局部门控空间选择性")
    _add_sensor_legend(ax)


def _plot_panel_d(ax, report: dict):
    """(d) 诚实盲区补全>=2辆真车组干净误报率：本文 (Tier 4) vs 仅传感器 (Tier 2)。"""
    fpr_eq5 = report["clean_fpr_and_blind_groups"]["equal_5pct_overall"]
    t2 = fpr_eq5["tier2_sensor_only"]
    t4 = fpr_eq5["tier4_sensor_dist_vis"]
    scopes = [
        ("overall", "综合\n均值"),
        ("m1", "激光车\n(m1)"),
        ("m2", "摄像头\n(m2)"),
    ]
    xs = np.arange(len(scopes))
    bw = 0.34
    v_t4 = [t4[s]["ge2"]["val"] * 100.0 for s, _ in scopes]
    v_t2 = [t2[s]["ge2"]["val"] * 100.0 for s, _ in scopes]

    b4 = ax.bar(xs - bw * 0.53, v_t4, width=bw * 0.92, color=C_OURS, edgecolor=C_OURS, linewidth=0.7, label="本文(Tier④)", zorder=3)
    b2 = ax.bar(xs + bw * 0.53, v_t2, width=bw * 0.92, color=C_WCD, edgecolor=C_WCD, linewidth=0.7, label="仅传感器(Tier②)", zorder=3)
    ax.axhline(5.0, color="#666666", linestyle="--", linewidth=0.9, zorder=1)

    for bx, val in zip(b4, v_t4):
        ax.text(
            bx.get_x() + bx.get_width() / 2, val + 0.45, f"{val:.1f}",
            ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=C_OURS,
            bbox=dict(boxstyle="square,pad=0.06", facecolor="white", edgecolor="none", alpha=0.88),
            zorder=5,
        )
    for bx, val in zip(b2, v_t2):
        ax.text(bx.get_x() + bx.get_width() / 2, val + 0.45, f"{val:.1f}", ha="center", va="bottom", fontsize=7.6, color="#8C5214", zorder=5)

    ax.set_xticks(xs)
    ax.set_xticklabels([s[1] for s in scopes], fontsize=8.5)
    ax.set_ylabel("盲区≥2误报率 / %", labelpad=3)
    ax.set_ylim(0, 23.0)
    ax.set_yticks([0, 5, 10, 15, 20])
    _set_sub_title(ax, "(d) 多盲区补全误报对比")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.13),
        ncol=2, frameon=False, handlelength=1.1, columnspacing=0.8, fontsize=8.1,
    )


def _plot_panel_e(ax, report: dict):
    """(e) 不同攻击预算（干净 -> 0.5σ -> 1.0σ -> 2.0σ -> ε=4）AP@0.5 (%) 退化与防护曲线。"""
    ap = report["ap_results"]
    states = ["clean", "pgd_0.5sigma", "pgd_1.0sigma", "pgd_2.0sigma", "pgd_eps4.0"]
    s_labels = ["干净", "0.5σ", "1.0σ", "2.0σ", "ε=4"]
    xs = np.arange(len(states))

    curves = [
        ("m1", "bud_fusion", "本文(激光)", C_OURS, "-", "o", "full"),
        ("m2", "bud_fusion", "本文(相机)", C_OURS, "--", "^", "none"),
        ("m1", "no_defense", "无防御(激光)", C_ATK, ":", "D", "full"),
        ("m2", "no_defense", "无防御(相机)", C_WCD, "-.", "s", "full"),
    ]
    for mod, mk, lbl, col, ls, mkr, fill in curves:
        vals = [ap[mod][s][mk]["ap50"]["val"] * 100.0 for s in states]
        kw = dict(markerfacecolor="white", markeredgewidth=1.2) if fill == "none" else {}
        ax.plot(
            xs, vals, label=lbl, color=col, linestyle=ls, marker=mkr,
            linewidth=1.7, markersize=4.6, **kw, zorder=4 if mk == "bud_fusion" else 3,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(s_labels, fontsize=8.3)
    ax.set_ylabel("AP@0.5 / %", labelpad=3)
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 40, 80, 100])
    _set_sub_title(ax, "(e) 不同攻击预算 AP 曲线")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.19),
        ncol=2, frameon=False, handlelength=1.6, columnspacing=0.7, labelspacing=0.25, fontsize=7.8,
    )


def main():
    _setup_rc()
    with open(REPORT_JSON, "r", encoding="utf-8") as f:
        report = json.load(f)

    # 1. 生成 16:9 PPT 横排 5 联总图
    fig, axes = plt.subplots(1, 5, figsize=(17.6, 3.65))
    draw_fns = [_plot_panel_a, _plot_panel_b, _plot_panel_c, _plot_panel_d, _plot_panel_e]
    for ax, fn in zip(axes, draw_fns):
        fn(ax, report)

    fig.subplots_adjust(left=0.04, right=0.99, top=0.80, bottom=0.22, wspace=0.38)
    combined_path = os.path.join(OUT_DIR, "fig_ppt_rq1_5panel_cn.png")
    fig.savefig(combined_path, dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)

    # 2. 分别导出 5 张独立高清子图，方便 PPT 自由排版
    single_names = [
        "fig_ppt_rq1_a_local_ap_cn.png",
        "fig_ppt_rq1_b_blind_recall_cn.png",
        "fig_ppt_rq1_c_gate_selectivity_cn.png",
        "fig_ppt_rq1_d_blind_fpr_cn.png",
        "fig_ppt_rq1_e_budget_curve_cn.png",
    ]
    out_paths = [combined_path]
    for name, fn in zip(single_names, draw_fns):
        fig_s, ax_s = plt.subplots(figsize=(3.85, 3.50))
        fn(ax_s, report)
        fig_s.tight_layout()
        p_s = os.path.join(OUT_DIR, name)
        fig_s.savefig(p_s, dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor="white")
        plt.close(fig_s)
        out_paths.append(p_s)

    print("PPT_PLOTS_OK:")
    for p in out_paths:
        print(" ", p)


if __name__ == "__main__":
    main()
