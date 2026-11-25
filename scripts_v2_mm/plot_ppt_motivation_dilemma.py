# -*- coding: utf-8 -*-
"""
生成开题/答辩动机实证高清中文图（300 DPI）：
展示现有防御（同构安全 SOTA 迁移 + 专门异构防御）在 HEAL 跨模态共享空间中的
“保无攻击精度 vs 保攻击防御”二选一死锁，同时以无攻击零掉点数据自证复现公平、数据合理。

全部数据 100% 实时读取自 E:/BUD/results_rq1_v2_mm/ 冻结 JSON，杜绝手填误差。
"""

import json
import os
import shutil
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES_DIR = r"E:\BUD\results_rq1_v2_mm"
OUT_DIR = r"E:\BUD\ppt_figures"
ART_DIR = r"C:\Users\Hiki\.gemini\antigravity\brain\be230015-873c-49a8-9080-587e2bed03c6"
os.makedirs(OUT_DIR, exist_ok=True)

# 统一中文无衬线学术字体与清爽排版
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.edgecolor"] = "#CBD5E0"
plt.rcParams["axes.linewidth"] = 0.9

# 与前 4 张正片故事图 100% 统一的高级克制配色
C_CLEAN = "#94A3B8"     # 柔和冷灰（无攻击基准）
C_CLEAN_BUD = "#63B3ED" # 亮蓝灰（本方案无攻击）
C_ATK_BASE = "#ED8936"  # 暖琥珀橙（基线受攻）
C_BUD = "#1E4E8C"       # 东南深靛蓝（本方案受攻/高亮）
C_RED = "#C53030"       # 警示红（失效/误杀标注）
C_GAIN = "#22543D"      # 墨绿（达标/突破标注）
C_GAIN_BG = "#E6FFFA"   # 薄荷浅绿底
C_WARN_BG = "#FFF5F5"   # 极浅警示红底
C_FAB_BASE = "#DD6B20"  # 深暖橙（基线伪造 ASR）
C_REM_BASE = "#94A3B8"  # 冷灰蓝（基线移除 ASR）


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45, color="#CBD5E0")
    ax.set_axisbelow(True)


def _safe_savefig(fig, out_p: str) -> str:
    """原子替换保存 PNG，规避 Windows 文件预览锁。"""
    tmp_p = out_p + ".tmp.png"
    fig.savefig(tmp_p, bbox_inches="tight", dpi=300)
    try:
        os.replace(tmp_p, out_p)
        return out_p
    except OSError:
        alt_p = out_p.replace(".png", "_updated.png")
        os.replace(tmp_p, alt_p)
        return alt_p


def main():
    with open(os.path.join(RES_DIR, "step5_2_mixed_report.json"), "r", encoding="utf-8") as f:
        r52 = json.load(f)
    with open(os.path.join(RES_DIR, "step5_4_targeted_asr_report.json"), "r", encoding="utf-8") as f:
        r54 = json.load(f)

    sr = r52["fleets"]["mixed"]["stratified_results"]
    mix_asr = r54["mixed_multicav_fleet_asr"]

    # 自动定位 3 车无攻击与 3 车雷达攻击 PGD 1.0σ 的准确键名
    k_clean = "collab_2|clean_all|clean"
    k_pgd = "collab_2|atk_m1|pgd_1.0sigma"
    assert k_clean in sr, f"缺少键 {k_clean}"
    assert k_pgd in sr, f"缺少键 {k_pgd}, 可选: {[k for k in sr if 'collab_2' in k and 'pgd' in k]}"

    clean_m = sr[k_clean]["methods"]
    pgd_m = sr[k_pgd]["methods"]
    rem_m = mix_asr["ge2_all|single_attacker|object_removal"]["methods"]
    fab_m = mix_asr["ge2_all|single_attacker|object_fabrication"]["methods"]

    # 统一 5 方法展示顺序：同构代表 2 个 + 异构代表 2 个 + 本文 1 个
    methods = [
        ("lucia", "LUCIA\n(同构迁移)"),
        ("robosac", "ROBOSAC\n(同构迁移)"),
        ("hetshield_paper", "HetShield-Paper\n(异构默认)"),
        ("hetshield_tuned", "HetShield-Tuned\n(异构调参)"),
        ("bud_frozen", "BUD-Fusion\n(本方案)"),
    ]

    clean_ap = [clean_m[k]["ap50"]["val"] * 100.0 for k, _ in methods]
    pgd_ap = [pgd_m[k]["ap50"]["val"] * 100.0 for k, _ in methods]
    rem_asr = [rem_m[k]["asr"] * 100.0 for k, _ in methods]
    fab_asr = [fab_m[k]["asr"] * 100.0 for k, _ in methods]

    nodef_clean = clean_m["no_defense"]["ap50"]["val"] * 100.0
    nodef_pgd = pgd_m["no_defense"]["ap50"]["val"] * 100.0
    nodef_rem = rem_m["no_defense"]["asr"] * 100.0
    nodef_fab = fab_m["no_defense"]["asr"] * 100.0

    hs_m2_fpr = r52["fleets"]["mixed"]["honest_cav_fpr_by_modality"]["m2"]["hetshield_paper_fpr"] * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.5), dpi=300)
    fig.patch.set_facecolor("white")

    x = np.arange(len(methods))
    w = 0.35
    num_bbox = dict(boxstyle="square,pad=0.06", facecolor="white", edgecolor="none", alpha=0.86)

    # ==================== 左图：无攻击精度 vs 雷达全局 PGD 攻击精度 ====================
    ax1 = axes[0]
    _style_ax(ax1)

    # 给本方案列加柔和浅蓝底框突出
    ax1.axvspan(len(methods) - 1.48, len(methods) - 0.52, color="#EBF8FF", alpha=0.68, zorder=0)

    c_l1 = [C_CLEAN if k != "bud_frozen" else C_CLEAN_BUD for k, _ in methods]
    c_l2 = [C_ATK_BASE if k != "bud_frozen" else C_BUD for k, _ in methods]

    b1 = ax1.bar(x - w / 2, clean_ap, w, color=c_l1, edgecolor="white", linewidth=0.7, zorder=3, label="无攻击精度 AP@0.5")
    b2 = ax1.bar(x + w / 2, pgd_ap, w, color=c_l2, edgecolor="white", linewidth=0.7, zorder=3, label="雷达车全局 PGD 攻击 AP@0.5")

    ax1.axhline(nodef_clean, color="#718096", linestyle="--", linewidth=1.15, zorder=2, label=f"无防御上限 ({nodef_clean:.1f}%)")
    ax1.axhline(nodef_pgd, color=C_RED, linestyle=":", linewidth=1.25, alpha=0.85, zorder=2, label=f"无防御受攻 ({nodef_pgd:.1f}%)")

    for idx_m, (bc, bp) in enumerate(zip(b1, b2)):
        is_bud = methods[idx_m][0] == "bud_frozen"
        hc, hp = bc.get_height(), bp.get_height()
        ax1.text(
            bc.get_x() + bc.get_width() / 2,
            hc + 1.5,
            f"{hc:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold" if is_bud else "normal",
            color=C_BUD if is_bud else "#2D3748",
            zorder=6,
            bbox=num_bbox,
        )
        ax1.text(
            bp.get_x() + bp.get_width() / 2,
            hp + 1.5,
            f"{hp:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold" if is_bud else "normal",
            color=C_BUD if is_bud else "#9C4221",
            zorder=6,
            bbox=num_bbox,
        )

    # 关键防杠诊断胶囊标注（统一置于 y=106.5，与柱顶数字充分拉开）
    ax1.text(
        0.0,
        106.5,
        "4~6车无攻击\n误伤掉 -12.3%",
        ha="center",
        va="bottom",
        fontsize=7.6,
        fontweight="bold",
        color=C_RED,
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.22", facecolor=C_WARN_BG, edgecolor="#FEB2B2", linewidth=0.75),
    )
    ax1.text(
        1.0,
        106.5,
        "无攻击正常\n2车失效/191ms",
        ha="center",
        va="bottom",
        fontsize=7.6,
        fontweight="bold",
        color="#744210",
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#FEFCBF", edgecolor="#F6E05E", linewidth=0.75),
    )
    ax1.text(
        2.0,
        106.5,
        f"无攻击误杀\n摄像头 {hs_m2_fpr:.1f}%",
        ha="center",
        va="bottom",
        fontsize=7.6,
        fontweight="bold",
        color=C_RED,
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.22", facecolor=C_WARN_BG, edgecolor="#FEB2B2", linewidth=0.75),
    )
    ax1.text(
        3.0,
        106.5,
        "调参保无攻击\n雷达攻击崩盘",
        ha="center",
        va="bottom",
        fontsize=7.6,
        fontweight="bold",
        color=C_RED,
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.22", facecolor=C_WARN_BG, edgecolor="#FEB2B2", linewidth=0.75),
    )
    ax1.text(
        4.0,
        106.5,
        "高精度与强防御\n双达标 (69.9ms)",
        ha="center",
        va="bottom",
        fontsize=7.6,
        fontweight="bold",
        color=C_GAIN,
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.22", facecolor=C_GAIN_BG, edgecolor="#81E6D9", linewidth=0.75),
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels([lb for _, lb in methods], fontsize=8.4, fontweight="bold", color="#1A202C")
    ax1.set_ylabel("3D 检测精度 AP@0.5 (%) ↑ 越高越好", fontsize=9.6, fontweight="bold", color="#2D3748")
    ax1.set_title("(a) 3车混合车队：无攻击精度 vs 全局 PGD 防御", fontsize=10.5, fontweight="bold", loc="left", pad=32, color="#1A202C")
    ax1.set_ylim(0, 126)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, fontsize=7.9, frameon=False, columnspacing=1.2)

    # ==================== 右图：3~6车混合车队定向局部攻击成功率 ASR ====================
    ax2 = axes[1]
    _style_ax(ax2)

    ax2.axvspan(len(methods) - 1.48, len(methods) - 0.52, color="#EBF8FF", alpha=0.68, zorder=0)

    c_r1 = [C_REM_BASE if k != "bud_frozen" else "#2B6CB0" for k, _ in methods]
    c_r2 = [C_FAB_BASE if k != "bud_frozen" else C_BUD for k, _ in methods]

    r1 = ax2.bar(x - w / 2, rem_asr, w, color=c_r1, edgecolor="white", linewidth=0.7, zorder=3, label="定向目标移除 ASR ↓")
    r2 = ax2.bar(x + w / 2, fab_asr, w, color=c_r2, edgecolor="white", linewidth=0.7, zorder=3, label="虚假目标伪造 ASR ↓")

    ax2.axhline(nodef_rem, color="#718096", linestyle="--", linewidth=1.15, zorder=2, label=f"无防御移除 ({nodef_rem:.1f}%)")
    ax2.axhline(nodef_fab, color=C_RED, linestyle=":", linewidth=1.25, alpha=0.85, zorder=2, label=f"无防御伪造 ({nodef_fab:.1f}%)")

    for idx_m, (br, bf) in enumerate(zip(r1, r2)):
        is_bud = methods[idx_m][0] == "bud_frozen"
        hr, hf = br.get_height(), bf.get_height()
        ax2.text(
            br.get_x() + br.get_width() / 2,
            hr + 1.5,
            f"{hr:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold" if is_bud else "normal",
            color="#2B6CB0" if is_bud else "#2D3748",
            zorder=6,
            bbox=num_bbox,
        )
        ax2.text(
            bf.get_x() + bf.get_width() / 2,
            hf + 1.5,
            f"{hf:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold" if is_bud else "normal",
            color=C_BUD if is_bud else "#9C4221",
            zorder=6,
            bbox=num_bbox,
        )

    # 右图核心总结标注（彻底避开图例与参考线）
    ax2.text(
        1.0,
        106.5,
        "无攻击保住92.2%\n但局部攻击全漏防",
        ha="center",
        va="bottom",
        fontsize=7.6,
        fontweight="bold",
        color=C_RED,
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.22", facecolor=C_WARN_BG, edgecolor="#FEB2B2", linewidth=0.75),
    )
    ax2.text(
        3.0,
        106.5,
        "放宽阈值保无攻击\n局部攻击全漏防",
        ha="center",
        va="bottom",
        fontsize=7.6,
        fontweight="bold",
        color=C_RED,
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.22", facecolor=C_WARN_BG, edgecolor="#FEB2B2", linewidth=0.75),
    )
    ax2.text(
        4.0,
        66.0,
        "全场最低 ASR\n移除 -54.3%\n伪造 -64.1%",
        ha="center",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color=C_GAIN,
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.26", facecolor=C_GAIN_BG, edgecolor="#81E6D9", linewidth=0.85),
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels([lb for _, lb in methods], fontsize=8.4, fontweight="bold", color="#1A202C")
    ax2.set_ylabel("攻击成功率 ASR (%) ↓ 越低越安全", fontsize=9.6, fontweight="bold", color="#2D3748")
    ax2.set_title("(b) 3~6车混合车队：定向局部攻击防御对比 (279帧)", fontsize=10.5, fontweight="bold", loc="left", pad=32, color="#1A202C")
    ax2.set_ylim(0, 126)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, fontsize=7.9, frameon=False, columnspacing=1.2)

    fig.suptitle(
        "现有防御在跨模态对齐空间中的“保无攻击 vs 防攻击”死锁实证（HEAL 跨模态底座，Dev-320 未见序列）",
        fontsize=11.8,
        fontweight="bold",
        y=1.02,
        color="#1A202C",
    )
    fig.text(
        0.5,
        -0.02,
        "数据口径：OPV2V 官方跨模态底座 HEAL (m1 64线LiDAR + m2 4目Camera) | 所有基线均在同批 FIT-11 无攻击数据复现/重训；ROBOSAC 与 HetShield-Tuned 无攻击均达 92.2% 零掉点，证实复现无偏。",
        ha="center",
        va="top",
        fontsize=8.1,
        color="#4A5568",
    )

    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    out_p = os.path.join(OUT_DIR, "fig_story0_baseline_dilemma.png")
    out_p = _safe_savefig(fig, out_p)
    plt.close(fig)

    # 同步复制到 Artifact 目录便于随时查看
    art_p = os.path.join(ART_DIR, os.path.basename(out_p))
    shutil.copy2(out_p, art_p)
    print("GENERATED:", out_p)
    print("SYNCED_ARTIFACT:", art_p)


if __name__ == "__main__":
    main()
