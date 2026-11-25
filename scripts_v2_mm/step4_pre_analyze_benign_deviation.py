# -*- coding: utf-8 -*-
"""
步骤 ④ 前置验证脚本：在完整 Q95-444 训练集 100 帧上按距离分档分析并绘制
干净数据下 m1 (64线 LiDAR) 与 m2 (Camera-LSS) 协作车的良性特征偏差分布。

核心验证目的（支撑研究点一·创新点 1.1 前提）：
1. 在同一批 100 帧（覆盖 2/3/4 车场景，严格来自不含 part4 的 15 条完整 Q95-444 序列）、
   同一公共空间覆盖域 m2_cov_in_ego 与同一本车清晰视区 (sigma_alea_0 < 0.52) 下，
   按协作者传感器距离分 4 档 [0,15), [15,30), [30,45), [45,60]m 比较 m1 vs m2 的偏差；
2. 同步统计传输层 heter_feature_2d 的数值尺度 sigma，为下一步小预算攻击 (0.5σ/1σ/2σ) 提供标尺；
3. 计算未按传感器能力归一时的帧级 MeanTopK(12) 分布与对良性摄像头车的误报率，
   绘制论文级 2x2 分析图 fig_step4_pre_benign_deviation_by_dist.png。
"""

import json
import math
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

from src_v2_mm.heal_base_wrapper import HEALBaseWrapper

TRAIN_READY_DIR = os.path.join(PROJ_ROOT, "tmp_png_ab_test", "train_ready_444")
OUT_JSON = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_pre_benign_deviation_stats.json")
OUT_PNG = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "fig_step4_pre_benign_deviation_by_dist.png")

PROTO_H = 256
PROTO_W = 256
GRID_RES = 0.8
MIN_XY = -102.4
VERTICAL_FOV_RAD = math.radians(27.0)
TARGET_HEIGHT_M = 1.55
REF_BEAM_COUNT = 16.0

DIST_BINS = [(0.0, 15.0), (15.0, 30.0), (30.0, 45.0), (45.0, 60.0)]
DIST_LABELS = ["0-15m", "15-30m", "30-45m", "45-60m"]


def build_bev_grid_xy_256(device="cuda", dtype=torch.float32):
    """生成 HEAL [-102.4, 102.4]^2 (256x256, 0.8m) 网格中心坐标，H对应y轴，W对应x轴。"""
    xs = (torch.arange(PROTO_W, device=device, dtype=dtype) + 0.5) * GRID_RES + MIN_XY
    ys = (torch.arange(PROTO_H, device=device, dtype=dtype) + 0.5) * GRID_RES + MIN_XY
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return grid_x, grid_y


def load_ego_lidar_np(dataset, idx: int) -> np.ndarray:
    """仅读取本车 ego 唯一原始 .pcd 点云并过滤自车车体反射与越界点（绝不混入协作者点云）。"""
    from opencood.utils import pcd_utils
    from opencood.utils.pcd_utils import mask_ego_points, mask_points_by_range

    scenario_index = 0
    for i, ele in enumerate(dataset.len_record):
        if idx < ele:
            scenario_index = i
            break
    scenario_database = dataset.scenario_database[scenario_index]
    ts_index = idx if scenario_index == 0 else idx - dataset.len_record[scenario_index - 1]
    ts_key = dataset.return_timestamp_key(scenario_database, ts_index)

    for _, cav_content in scenario_database.items():
        if cav_content.get("ego", False):
            pts = pcd_utils.pcd_to_np(cav_content[ts_key]["lidar"])
            pts = mask_ego_points(pts)
            pts = mask_points_by_range(pts, [-102.4, -102.4, -3.0, 102.4, 102.4, 1.0])
            return pts
    raise RuntimeError(f"样本 {idx} 未找到 ego 车辆点云！")


def compute_ego_beam_visibility_256(
    ego_pts_np: np.ndarray,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    num_beams: int = 64,
    num_angle_bins: int = 360,
    occ_margin: float = 2.0,
) -> torch.Tensor:
    """
    计算 256x256 空间上的本车极坐标目标高度层有效光束偶然不确定性 sigma_{alea,0}^2 in [0.02, 1.0]。
    在立体障碍物高度层 z in [-1.30, +0.85]m（排除 r <= 2.4m 自车点）统计首达遮挡包络。
    """
    device = grid_x.device
    r_grid = torch.sqrt(grid_x * grid_x + grid_y * grid_y)
    theta_grid = torch.atan2(grid_y, grid_x)
    r_occ_bins = torch.full((num_angle_bins,), 160.0, device=device, dtype=torch.float32)

    if ego_pts_np is not None and ego_pts_np.shape[0] > 0:
        pts_t = torch.from_numpy(ego_pts_np[:, :3]).to(device=device, dtype=torch.float32)
        pr = torch.sqrt(pts_t[:, 0] ** 2 + pts_t[:, 1] ** 2)
        pz = pts_t[:, 2]
        obs_mask = (pz > -1.30) & (pz < 0.85) & (pr > 2.4)
        if obs_mask.any():
            obs_r = pr[obs_mask]
            obs_theta = torch.atan2(pts_t[obs_mask, 1], pts_t[obs_mask, 0])
            bin_idx = torch.clamp(
                ((obs_theta + math.pi) / (2.0 * math.pi) * num_angle_bins).long(),
                0,
                num_angle_bins - 1,
            )
            r_occ_bins.scatter_reduce_(0, bin_idx, obs_r, reduce="amin", include_self=True)
            r_pad = torch.cat([r_occ_bins[-1:], r_occ_bins, r_occ_bins[:1]], dim=0)
            r_occ_bins = F.max_pool1d(r_pad.view(1, 1, -1), kernel_size=3, stride=1).view(-1)

    grid_bin_idx = torch.clamp(
        ((theta_grid + math.pi) / (2.0 * math.pi) * num_angle_bins).long(),
        0,
        num_angle_bins - 1,
    )
    r_occ_map = r_occ_bins[grid_bin_idx]
    excess_dist = torch.clamp(r_grid - (r_occ_map + occ_margin), min=0.0)
    los_factor = torch.exp(-excess_dist / 3.0)

    safe_dist = torch.clamp(r_grid, min=2.0)
    n_beam_vis = float(num_beams) * (torch.atan(TARGET_HEIGHT_M / safe_dist) / VERTICAL_FOV_RAD) * los_factor
    vis_score = torch.clamp(torch.log1p(n_beam_vis) / math.log1p(REF_BEAM_COUNT), 0.0, 1.0)
    sigma_alea_sq = torch.clamp(1.0 - vis_score, min=0.02, max=1.0)
    return sigma_alea_sq.view(1, 1, PROTO_H, PROTO_W)


def select_100_train_indices(wrapper: HEALBaseWrapper, dataset, train_dir: str):
    """从 15 条完整 Q95-444 训练序列中跨车数（2/3/4 车）确定性均匀抽取 100 帧。"""
    sample_seqs = wrapper.get_sample_sequences(dataset)
    seq_to_idxs = OrderedDict()
    for i, s in enumerate(sample_seqs):
        seq_to_idxs.setdefault(s, []).append(i)

    # 按每条序列总帧数加权分配，保底每条 4 帧，余数按最大小数余量调整至精准 100 帧
    total_all = len(sample_seqs)
    seqs = list(seq_to_idxs.keys())
    raw_alloc = {s: max(4.0, 100.0 * len(seq_to_idxs[s]) / float(total_all)) for s in seqs}
    counts = {s: int(math.floor(raw_alloc[s])) for s in seqs}
    # 若超出 100 则从最大分配扣除，若不足 100 则补足至精准 100
    diff = 100 - sum(counts.values())
    if diff > 0:
        order = sorted(seqs, key=lambda s: raw_alloc[s] - counts[s], reverse=True)
        for s in order[:diff]:
            counts[s] += 1
    elif diff < 0:
        order = sorted(seqs, key=lambda s: counts[s], reverse=True)
        for s in order[:(-diff)]:
            counts[s] -= 1

    sampled = []
    plan = OrderedDict()
    for s in seqs:
        idxs = seq_to_idxs[s]
        n_pick = counts[s]
        pos = np.linspace(0, len(idxs) - 1, n_pick, dtype=int)
        picked = [int(idxs[p]) for p in pos]
        cav_dirs = [d for d in os.listdir(os.path.join(train_dir, s)) if os.path.isdir(os.path.join(train_dir, s, d))]
        for p_idx in picked:
            sampled.append((p_idx, s, len(cav_dirs)))
        plan[s] = {"raw_cavs": len(cav_dirs), "total_frames": len(idxs), "sampled_frames": n_pick, "indices": picked}
    return sampled, plan


def summarize_array(arr_list: list) -> dict:
    """对收集的分块 numpy 浮点数组计算严格描述性统计。"""
    if not arr_list:
        return {"count": 0, "mean": None, "std": None, "median": None, "p25": None, "p75": None, "p90": None, "p95": None}
    arr = np.concatenate(arr_list, axis=0).astype(np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": None, "std": None, "median": None, "p25": None, "p75": None, "p90": None, "p95": None}
    return {
        "count": int(arr.size),
        "mean": round(float(np.mean(arr)), 6),
        "std": round(float(np.std(arr)), 6),
        "median": round(float(np.median(arr)), 6),
        "p25": round(float(np.percentile(arr, 25)), 6),
        "p75": round(float(np.percentile(arr, 75)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "p95": round(float(np.percentile(arr, 95)), 6),
    }


def summarize_scalar_samples(vals: list) -> dict:
    """汇总 100 帧标量统计数组。"""
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": round(float(np.mean(arr)), 6),
        "std": round(float(np.std(arr)), 6),
        "median": round(float(np.median(arr)), 6),
        "p75": round(float(np.percentile(arr, 75)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "p95": round(float(np.percentile(arr, 95)), 6),
        "p99": round(float(np.percentile(arr, 99)), 6),
        "max": round(float(np.max(arr)), 6),
    }


def plot_analysis_figure(stats_dict: dict, out_png: str):
    """绘制 2x2 高质量学术对比图，展示按距离分档的 m1 vs m2 良性偏差及未归一化误报。"""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.8,
        "ytick.labelsize": 9.8,
        "legend.fontsize": 9.2,
    })

    c_m1 = "#1b6ca8"   # 深蓝：64线 LiDAR (m1)
    c_m2 = "#d95f02"   # 暖橙红：Camera-LSS (m2)
    x = np.arange(len(DIST_LABELS))

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.6), dpi=300)
    fig.subplots_adjust(hspace=0.33, wspace=0.23, top=0.90, bottom=0.07, left=0.07, right=0.97)
    fig.suptitle(
        "Benign Feature Deviation vs. Distance: m1 (64L LiDAR) vs. m2 (Camera-LSS) in HEAL pyramid_L0\n"
        "(100 Matched Clean Frames on OPV2V Q95-444, Shared Co-Visible Camera FOV & Ego Clear Sight)",
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )

    def _extract_curve(metric_key: str, mod_key: str, stat_key: str):
        return [stats_dict["by_distance_bin"][lbl][metric_key][mod_key][stat_key] for lbl in DIST_LABELS]

    # (a) 共视活跃区 MSE_per_ch (中位数 + IQR 误差棒 + 均值折线 + P95 点线)
    ax = axes[0, 0]
    for mod_key, lbl_mod, col, offset in [
        ("m1", "m1 Coll. (64L LiDAR)", c_m1, -0.16),
        ("m2", "m2 Coll. (Camera-LSS)", c_m2, +0.16),
    ]:
        med = np.array(_extract_curve("active_covis_mse", mod_key, "median"))
        p25 = np.array(_extract_curve("active_covis_mse", mod_key, "p25"))
        p75 = np.array(_extract_curve("active_covis_mse", mod_key, "p75"))
        mean = np.array(_extract_curve("active_covis_mse", mod_key, "mean"))
        p95 = np.array(_extract_curve("active_covis_mse", mod_key, "p95"))

        ax.bar(
            x + offset, p75 - p25, bottom=p25, width=0.28, color=col, alpha=0.26,
            edgecolor=col, linewidth=1.1, label=f"{lbl_mod} IQR [P25-P75]"
        )
        ax.hlines(med, x + offset - 0.13, x + offset + 0.13, colors=col, linewidth=2.4)
        ax.plot(x + offset, mean, marker="o", markersize=5.5, color=col, linewidth=2.0, label=f"{lbl_mod} Mean")
        ax.plot(x + offset, p95, marker="^", markersize=5.0, linestyle="--", color=col, linewidth=1.3, alpha=0.85, label=f"{lbl_mod} P95")

    ax.set_xticks(x)
    ax.set_xticklabels(DIST_LABELS)
    max_a = max(max(_extract_curve("active_covis_mse", "m1", "p95")), max(_extract_curve("active_covis_mse", "m2", "p95")))
    ax.set_ylim(0.0, max_a * 1.35)
    ax.set_ylabel("pyramid_L0 MSE per Channel")
    ax.set_xlabel("Distance from Collaborator Center d_i")
    ax.set_title("(a) Active Co-Visible Grids: Feature MSE vs. Distance")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.9)

    # (b) 共视前景车辆区 (cls_prob_ego > 0.20) MSE_per_ch
    ax = axes[0, 1]
    for mod_key, lbl_mod, col, mk in [
        ("m1", "m1 Coll. (64L LiDAR)", c_m1, "s"),
        ("m2", "m2 Coll. (Camera-LSS)", c_m2, "o"),
    ]:
        mean = np.array(_extract_curve("fg_covis_mse", mod_key, "mean"))
        med = np.array(_extract_curve("fg_covis_mse", mod_key, "median"))
        p95 = np.array(_extract_curve("fg_covis_mse", mod_key, "p95"))
        ax.plot(x, mean, marker=mk, markersize=6, linewidth=2.2, color=col, label=f"{lbl_mod} Mean")
        ax.plot(x, med, marker=mk, markersize=4.5, linewidth=1.4, linestyle="-.", color=col, alpha=0.85, label=f"{lbl_mod} Median")
        ax.plot(x, p95, marker="^", markersize=5.5, linewidth=1.5, linestyle="--", color=col, alpha=0.9, label=f"{lbl_mod} P95")
        for xi, yi in zip(x, mean):
            va = "bottom" if mod_key == "m2" else "top"
            dy = 0.28 if mod_key == "m2" else -0.28
            ax.text(xi, yi + dy, f"{yi:.2f}", ha="center", va=va, fontsize=8.5, color=col, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(DIST_LABELS)
    max_b = max(max(_extract_curve("fg_covis_mse", "m1", "p95")), max(_extract_curve("fg_covis_mse", "m2", "p95")))
    ax.set_ylim(0.0, max_b * 1.32)
    ax.set_ylabel("pyramid_L0 MSE per Channel")
    ax.set_xlabel("Distance from Collaborator Center d_i")
    ax.set_title("(b) Foreground Vehicle Grids (cls_ego > 0.20): MSE")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.9)

    # (c) 活跃重叠区 Cosine Deviation (1 - cos) vs. Distance
    ax = axes[1, 0]
    for mod_key, lbl_mod, col, mk in [
        ("m1", "m1 Coll. (64L LiDAR)", c_m1, "s"),
        ("m2", "m2 Coll. (Camera-LSS)", c_m2, "o"),
    ]:
        mean = np.array(_extract_curve("overlap_cos_dev", mod_key, "mean"))
        p25 = np.array(_extract_curve("overlap_cos_dev", mod_key, "p25"))
        p75 = np.array(_extract_curve("overlap_cos_dev", mod_key, "p75"))
        p95 = np.array(_extract_curve("overlap_cos_dev", mod_key, "p95"))
        ax.fill_between(x, p25, p75, color=col, alpha=0.16)
        ax.plot(x, mean, marker=mk, markersize=6, linewidth=2.2, color=col, label=f"{lbl_mod} Mean (IQR Band)")
        ax.plot(x, p95, marker="^", markersize=5.5, linewidth=1.5, linestyle="--", color=col, label=f"{lbl_mod} P95")
        for xi, yi in zip(x, mean):
            dy = 0.025 if mod_key == "m2" else -0.035
            ax.text(xi, yi + dy, f"{yi:.3f}", ha="center", va="center", fontsize=8.5, color=col, fontweight="bold")

    ax.axhline(0.30, color="#444444", linestyle=":", linewidth=1.5, label="HetShield Default tau=0.30")
    ax.set_xticks(x)
    ax.set_xticklabels(DIST_LABELS)
    ax.set_ylim(0.0, 0.92)
    ax.set_ylabel("Cosine Deviation (1 - cos)")
    ax.set_xlabel("Distance from Collaborator Center d_i")
    ax.set_title("(c) Active Overlap Grids: Cosine Deviation vs. Distance")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    # (d) 帧级未归一化 MeanTopK(12) 局部冲突分布与统一门限对 m2 的误报率
    ax = axes[1, 1]
    scores_m1 = np.asarray(stats_dict["frame_meantopk12_raw"]["m1_values"], dtype=np.float64)
    scores_m2 = np.asarray(stats_dict["frame_meantopk12_raw"]["m2_values"], dtype=np.float64)
    all_max = max(float(np.percentile(scores_m2, 99)), float(np.percentile(scores_m1, 99)), 1.0)
    bins = np.linspace(0.0, all_max * 1.05, 24)
    ax.hist(scores_m1, bins=bins, color=c_m1, alpha=0.52, edgecolor=c_m1, linewidth=0.9, label=f"m1 Benign (mean={scores_m1.mean():.2f})")
    ax.hist(scores_m2, bins=bins, color=c_m2, alpha=0.48, edgecolor=c_m2, linewidth=0.9, label=f"m2 Benign (mean={scores_m2.mean():.2f})")

    th_m1_95 = stats_dict["frame_meantopk12_raw"]["m1_p95_threshold"]
    fp_rate_m2 = stats_dict["frame_meantopk12_raw"]["m2_benign_fpr_at_m1_p95"] * 100.0
    ax.axvline(
        th_m1_95, color="#111111", linestyle="--", linewidth=1.8,
        label=f"m1-Calibrated 95% Thresh ({th_m1_95:.2f})\n=> Benign m2 FPR = {fp_rate_m2:.1f}%"
    )
    ax.set_xlabel("Frame-Level Uncalibrated MeanTopK(12) Local MSE")
    ax.set_ylabel("Frame Count (out of 100 Frames)")
    ax.set_title("(d) Uncalibrated MeanTopK(12): Homogeneous Threshold Fails m2")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wrapper = HEALBaseWrapper(data_dir=TRAIN_READY_DIR, device=str(device))
    ds_m2 = wrapper.build_dataset(data_dir=TRAIN_READY_DIR, use_cav=2, visualize=True, collab_modality="m2")
    ds_m1 = wrapper.build_dataset(data_dir=TRAIN_READY_DIR, use_cav=2, visualize=True, collab_modality="m1")
    assert len(ds_m2) == len(ds_m1), "m1 与 m2 数据集长度不一致！"

    sampled_list, sampling_plan = select_100_train_indices(wrapper, ds_m2, TRAIN_READY_DIR)
    print(f"[INFO] 完整训练集序列数={len(sampling_plan)}, 总帧={len(ds_m2)}, 抽样={len(sampled_list)} 帧", flush=True)

    from opencood.tools import train_utils
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

    grid_x, grid_y = build_bev_grid_xy_256(device=str(device))

    # 传输层特征尺度统计缓存
    trans_stats = {
        "m1_all": [],
        "m1_active_gt01": [],
        "m2_valid_box": [],
        "m2_active_gt01": [],
    }
    # 按距离分档聚合池
    dist_collect = {
        lbl: {
            "active_covis_mse": {"m1": [], "m2": []},
            "ego_act_covis_mse": {"m1": [], "m2": []},
            "all_clear_covis_mse": {"m1": [], "m2": []},
            "fg_covis_mse": {"m1": [], "m2": []},
            "overlap_cos_dev": {"m1": [], "m2": []},
        }
        for lbl in DIST_LABELS
    }
    frame_mtk_m1 = []
    frame_mtk_m2 = []

    for step_i, (sample_idx, seq, n_cav_raw) in enumerate(sampled_list, 1):
        b2 = train_utils.to_device(ds_m2.collate_batch_test([ds_m2[sample_idx]]), device)["ego"]
        b1 = train_utils.to_device(ds_m1.collate_batch_test([ds_m1[sample_idx]]), device)["ego"]

        p2 = wrapper.extract_protocol_features(b2)
        p1 = wrapper.extract_protocol_features(b1)
        assert p2["agent_modality_list"] == ["m1", "m2"], f"sample {sample_idx} p2 mods={p2['agent_modality_list']}"
        assert p1["agent_modality_list"] == ["m1", "m1"], f"sample {sample_idx} p1 mods={p1['agent_modality_list']}"

        ego_pts_np = load_ego_lidar_np(ds_m2, sample_idx)
        sigma_alea_0 = compute_ego_beam_visibility_256(ego_pts_np, grid_x, grid_y, num_beams=64)  # (1,1,256,256)
        ego_clear_vis = sigma_alea_0 < 0.52

        with torch.no_grad():
            # 1. 传输层尺度统计 (heter_feature_2d, 64x256x256)
            f_tr_m1_cav = p1["heter_feature_2d"][1]
            f_tr_m2_cav = p2["heter_feature_2d"][1]
            m2_mask_bool = p2["cam_valid_mask"][1, 0] > 0.5

            trans_stats["m1_all"].append(float(f_tr_m1_cav.std().item()))
            m1_act_vals = f_tr_m1_cav[:, f_tr_m1_cav.abs().sum(dim=0) > 1e-3]
            if m1_act_vals.numel() > 0:
                trans_stats["m1_active_gt01"].append(float(m1_act_vals.std().item()))

            m2_box_vals = f_tr_m2_cav[:, m2_mask_bool]
            trans_stats["m2_valid_box"].append(float(m2_box_vals.std().item()))
            m2_act_vals = f_tr_m2_cav[:, m2_mask_bool & (f_tr_m2_cav.norm(dim=0) > 2.0)]
            if m2_act_vals.numel() > 0:
                trans_stats["m2_active_gt01"].append(float(m2_act_vals.std().item()))

            # 2. 提取 pyramid_L0 (64x256x256 严格非负) 并对齐至本车坐标系
            l0_1 = wrapper.model.pyramid_backbone.get_multiscale_feature(p1["heter_feature_2d"])[0]
            l0_2 = wrapper.model.pyramid_backbone.get_multiscale_feature(p2["heter_feature_2d"])[0]
            align_corners = wrapper.model.pyramid_backbone.align_corners

            z0 = l0_1[0:1]
            z1_in_ego = warp_affine_simple(l0_1[1:2], p1["affine_matrix"][0, 0, 1:2], (PROTO_H, PROTO_W), align_corners=align_corners)
            z2_in_ego = warp_affine_simple(l0_2[1:2], p2["affine_matrix"][0, 0, 1:2], (PROTO_H, PROTO_W), align_corners=align_corners)

            # 同一公共覆盖域 m2_cov_in_ego 与同一物理距离图 d_i
            cov_m2 = warp_affine_simple(
                p2["cam_valid_mask"][1:2], p2["affine_matrix"][0, 0, 1:2], (PROTO_H, PROTO_W), align_corners=False
            ) > 0.5
            # pairwise_t_matrix[0, 1, 0][:2, 3] 已实测验证严格对应 T_{cav -> ego} 平移坐标 (cx, cy)
            cav_xy = b2["pairwise_t_matrix"][0, 1, 0, :2, 3]
            dist_cav = torch.sqrt((grid_x - cav_xy[0]) ** 2 + (grid_y - cav_xy[1]) ** 2).view(1, 1, PROTO_H, PROTO_W)

            # 本车前景概率图
            ego_out = wrapper.forward_single_agent(p1["heter_feature_2d"][0:1])
            cls_prob_ego = torch.sigmoid(ego_out["cls_preds"]).max(dim=1, keepdim=True)[0]

            # MSE_per_ch 与净响应 / 原始 L2 范数
            mse_m1 = ((z0 - z1_in_ego) ** 2).mean(dim=1, keepdim=True)
            mse_m2 = ((z0 - z2_in_ego) ** 2).mean(dim=1, keepdim=True)

            def _net_norm(z):
                zn = z - z.mean(dim=(2, 3), keepdim=True)
                return zn.norm(dim=1, keepdim=True)

            nn0 = _net_norm(z0)
            nn1 = _net_norm(z1_in_ego)
            nn2 = _net_norm(z2_in_ego)

            l2_0 = z0.norm(dim=1, keepdim=True)
            l2_1 = z1_in_ego.norm(dim=1, keepdim=True)
            l2_2 = z2_in_ego.norm(dim=1, keepdim=True)

            cos_m1_dev = 1.0 - F.cosine_similarity(z0, z1_in_ego, dim=1, eps=1e-6).unsqueeze(1)
            cos_m2_dev = 1.0 - F.cosine_similarity(z0, z2_in_ego, dim=1, eps=1e-6).unsqueeze(1)

            # 定义完全同像素的公共清晰可见与活跃区掩膜，保障对比绝对公平
            base_covis = cov_m2 & ego_clear_vis
            ego_act_mask = base_covis & (nn0 > 4.0)
            fg_mask = base_covis & (cls_prob_ego > 0.20)
            # 对双边活跃重叠：分别取净响应超阈值 (本车4.0 / 协作者阈值按实际物理活跃) 或 L2 范数 > 4.0
            act_m1_union = base_covis & ((nn0 > 4.0) | (nn1 > 4.0))
            act_m2_union = base_covis & ((nn0 > 4.0) | (nn2 > 5.8))
            ov_m1 = base_covis & (l2_0 > 4.0) & (l2_1 > 4.0)
            ov_m2 = base_covis & (l2_0 > 4.0) & (l2_2 > 4.0)

            for (d_lo, d_hi), lbl in zip(DIST_BINS, DIST_LABELS):
                d_mask = (dist_cav >= d_lo) & (dist_cav < d_hi if d_hi < 60.0 else dist_cav <= d_hi)
                # 主口径：本车清晰活跃同像素对同像素对比 (完全同一组网格！)
                m_same_act = d_mask & ego_act_mask
                if m_same_act.any():
                    dist_collect[lbl]["ego_act_covis_mse"]["m1"].append(mse_m1[m_same_act].detach().cpu().numpy())
                    dist_collect[lbl]["ego_act_covis_mse"]["m2"].append(mse_m2[m_same_act].detach().cpu().numpy())
                # 全清晰共视网格同像素对比
                m_clear = d_mask & base_covis
                if m_clear.any():
                    dist_collect[lbl]["all_clear_covis_mse"]["m1"].append(mse_m1[m_clear].detach().cpu().numpy())
                    dist_collect[lbl]["all_clear_covis_mse"]["m2"].append(mse_m2[m_clear].detach().cpu().numpy())
                # 双边活跃并集对比
                if (d_mask & act_m1_union).any():
                    dist_collect[lbl]["active_covis_mse"]["m1"].append(mse_m1[d_mask & act_m1_union].detach().cpu().numpy())
                if (d_mask & act_m2_union).any():
                    dist_collect[lbl]["active_covis_mse"]["m2"].append(mse_m2[d_mask & act_m2_union].detach().cpu().numpy())
                # 前景车辆区对比 (完全同一组网格！)
                m_fg = d_mask & fg_mask
                if m_fg.any():
                    dist_collect[lbl]["fg_covis_mse"]["m1"].append(mse_m1[m_fg].detach().cpu().numpy())
                    dist_collect[lbl]["fg_covis_mse"]["m2"].append(mse_m2[m_fg].detach().cpu().numpy())
                # 活跃重叠区余弦偏离 (1 - cos)
                if (d_mask & ov_m1).any():
                    dist_collect[lbl]["overlap_cos_dev"]["m1"].append(cos_m1_dev[d_mask & ov_m1].detach().cpu().numpy())
                if (d_mask & ov_m2).any():
                    dist_collect[lbl]["overlap_cos_dev"]["m2"].append(cos_m2_dev[d_mask & ov_m2].detach().cpu().numpy())

            # 3. 帧级未白化 3x3 MeanTopK(12) 局部冲突分布（在公共域 cov_m2 & ego_clear_vis 上）
            pool1 = F.avg_pool2d(mse_m1 * base_covis.float(), kernel_size=3, stride=1, padding=1)
            pool2 = F.avg_pool2d(mse_m2 * base_covis.float(), kernel_size=3, stride=1, padding=1)
            v1 = pool1[base_covis]
            v2 = pool2[base_covis]
            if v1.numel() >= 12:
                frame_mtk_m1.append(float(torch.topk(v1, k=12).values.mean().item()))
                frame_mtk_m2.append(float(torch.topk(v2, k=12).values.mean().item()))

        if step_i % 25 == 0 or step_i == len(sampled_list):
            print(f"  [进度 {step_i}/{len(sampled_list)}] 已耗时 {time.time() - t0:.1f}s ({seq})", flush=True)

    # 汇总所有统计指标
    by_distance = OrderedDict()
    for lbl in DIST_LABELS:
        by_distance[lbl] = OrderedDict()
        # 图 (a) 使用同像素公平对比 ego_act_covis_mse 作为 active_covis_mse 主视图，同时也保留双边并集供核查
        by_distance[lbl]["active_covis_mse"] = {
            "m1": summarize_array(dist_collect[lbl]["ego_act_covis_mse"]["m1"]),
            "m2": summarize_array(dist_collect[lbl]["ego_act_covis_mse"]["m2"]),
        }
        by_distance[lbl]["union_active_covis_mse"] = {
            "m1": summarize_array(dist_collect[lbl]["active_covis_mse"]["m1"]),
            "m2": summarize_array(dist_collect[lbl]["active_covis_mse"]["m2"]),
        }
        by_distance[lbl]["all_clear_covis_mse"] = {
            "m1": summarize_array(dist_collect[lbl]["all_clear_covis_mse"]["m1"]),
            "m2": summarize_array(dist_collect[lbl]["all_clear_covis_mse"]["m2"]),
        }
        by_distance[lbl]["fg_covis_mse"] = {
            "m1": summarize_array(dist_collect[lbl]["fg_covis_mse"]["m1"]),
            "m2": summarize_array(dist_collect[lbl]["fg_covis_mse"]["m2"]),
        }
        by_distance[lbl]["overlap_cos_dev"] = {
            "m1": summarize_array(dist_collect[lbl]["overlap_cos_dev"]["m1"]),
            "m2": summarize_array(dist_collect[lbl]["overlap_cos_dev"]["m2"]),
        }

    arr_m1 = np.asarray(frame_mtk_m1, dtype=np.float64)
    arr_m2 = np.asarray(frame_mtk_m2, dtype=np.float64)
    m1_p95 = float(np.percentile(arr_m1, 95))
    m2_fpr_at_m1_p95 = float(np.mean(arr_m2 > m1_p95))

    stats_out = {
        "meta": {
            "task": "Step4 Pre-Verification: Benign Deviation vs Distance (m1 vs m2 on Q95-444)",
            "source_dir": TRAIN_READY_DIR,
            "num_complete_sequences": len(sampling_plan),
            "num_sampled_frames": len(sampled_list),
            "total_elapsed_sec": round(time.time() - t0, 2),
        },
        "transport_layer_feature_scale_sigma": {
            "m1_all_grid_std": summarize_scalar_samples(trans_stats["m1_all"]),
            "m1_active_grid_std": summarize_scalar_samples(trans_stats["m1_active_gt01"]),
            "m2_valid_crop_std": summarize_scalar_samples(trans_stats["m2_valid_box"]),
            "m2_active_grid_std": summarize_scalar_samples(trans_stats["m2_active_gt01"]),
        },
        "by_distance_bin": by_distance,
        "frame_meantopk12_raw": {
            "m1_summary": summarize_scalar_samples(frame_mtk_m1),
            "m2_summary": summarize_scalar_samples(frame_mtk_m2),
            "m1_p95_threshold": round(m1_p95, 6),
            "m2_benign_fpr_at_m1_p95": round(m2_fpr_at_m1_p95, 6),
            "m1_values": [round(float(v), 5) for v in frame_mtk_m1],
            "m2_values": [round(float(v), 5) for v in frame_mtk_m2],
        },
        "sampling_plan": sampling_plan,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(stats_out, f, ensure_ascii=False, indent=2)
    plot_analysis_figure(stats_out, OUT_PNG)
    print(f"[DONE] 统计写入 {OUT_JSON}, 图表写入 {OUT_PNG}", flush=True)
    print(f"[SUMMARY] m1_p95_MeanTopK12={m1_p95:.4f} | benign m2 FPR under m1_p95={m2_fpr_at_m1_p95*100:.2f}%", flush=True)
    for lbl in DIST_LABELS:
        d = by_distance[lbl]
        print(
            f"  {lbl}: active_MSE(m1={d['active_covis_mse']['m1']['mean']:.3f}, m2={d['active_covis_mse']['m2']['mean']:.3f}) | "
            f"fg_MSE(m1={d['fg_covis_mse']['m1']['mean']:.3f}, m2={d['fg_covis_mse']['m2']['mean']:.3f}) | "
            f"cos_dev(m1={d['overlap_cos_dev']['m1']['mean']:.3f}, m2={d['overlap_cos_dev']['m2']['mean']:.3f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
