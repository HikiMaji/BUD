# -*- coding: utf-8 -*-
"""
步骤 ④-1 核心诊断与 4 组门限对比实验脚本：
1. 摄像头车 0~15m 偏差尖峰精确溯源（距离细档 + 位置归因 + 是否需要特殊处理对照）；
2. m1 与 m2 各自内部按「本车可见性（Clear/Sparse/Occluded）× 距离（4档）」统计良性偏差与子区域误报；
3. 按父序列互斥拆分 Fit (8条序列60帧) 与 Eval (7条序列60帧)，对比 4 组门限
   （① 统一门限 / ② 仅传感器 / ③ 传感器+距离 / ④ 传感器+距离+可见性）
   在相同总误报率 (5.0%) 下对 m1、m2 小预算 (0.5σ/1.0σ) 与局部 (1~2目标擦除/鬼影) 攻击的检出率。
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

from src_v2_mm.heal_attacks import (
    generate_local_ghost_spoofing_heal,
    generate_local_removal_heal,
    generate_scaled_pgd_removal_heal,
)
from src_v2_mm.heal_base_wrapper import HEALBaseWrapper

TRAIN_READY_DIR = os.path.join(PROJ_ROOT, "tmp_png_ab_test", "train_ready_444")
OUT_JSON = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_1_visibility_spike_4tier_report.json")
OUT_FIG_HEAT = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "fig_step4_1_visibility_distance_heatmap.png")
OUT_FIG_4TIER = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "fig_step4_1_4tier_threshold_comparison.png")

PROTO_H = 256
PROTO_W = 256
GRID_RES = 0.8
MIN_XY = -102.4
VERTICAL_FOV_RAD = math.radians(27.0)
TARGET_HEIGHT_M = 1.55
REF_BEAM_COUNT = 16.0

DIST_BINS = [(0.0, 15.0), (15.0, 30.0), (30.0, 45.0), (45.0, 999.0)]
DIST_LABELS = ["0-15m", "15-30m", "30-45m", "45m+"]
VIS_LABELS = ["Clear", "Sparse", "Occluded"]
SPIKE_SUB_BINS = [(0.0, 3.0), (3.0, 6.0), (6.0, 10.0), (10.0, 15.0)]
SPIKE_SUB_LABELS = ["0-3m", "3-6m", "6-10m", "10-15m"]


def build_bev_grid_xy_256(device="cuda", dtype=torch.float32):
    """生成 HEAL [-102.4, 102.4]^2 (256x256, 0.8m) 网格中心物理坐标。"""
    xs = (torch.arange(PROTO_W, device=device, dtype=dtype) + 0.5) * GRID_RES + MIN_XY
    ys = (torch.arange(PROTO_H, device=device, dtype=dtype) + 0.5) * GRID_RES + MIN_XY
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return grid_x, grid_y


def load_ego_lidar_np(dataset, idx: int) -> np.ndarray:
    """仅读取本车 ego 唯一原始 .pcd 点云并过滤自车外壳与越界点。"""
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
    """基于本车纯点云计算极坐标目标高度层有效光束偶然不确定性 sigma_{alea,0}^2 in [0.02, 1.0]。"""
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


def count_points_in_corners3d(points_xyz: np.ndarray, corners_3d: np.ndarray) -> np.ndarray:
    """客观统计落入每个 3D GT 框内部的本车原始激光点数量，用于 GT 可见性分层。"""
    n_boxes = corners_3d.shape[0]
    counts = np.zeros(n_boxes, dtype=np.int32)
    if n_boxes == 0 or points_xyz is None or points_xyz.shape[0] == 0:
        return counts
    pts = points_xyz[:, :3]
    for i in range(n_boxes):
        c = corners_3d[i]
        xmin, xmax = float(c[:, 0].min()), float(c[:, 0].max())
        ymin, ymax = float(c[:, 1].min()), float(c[:, 1].max())
        zmin, zmax = float(c[:, 2].min()), float(c[:, 2].max())
        aabb = (
            (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax)
            & (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)
            & (pts[:, 2] >= zmin - 0.1) & (pts[:, 2] <= zmax + 0.1)
        )
        if not np.any(aabb):
            continue
        psub = pts[aabb, :2]
        zmid = 0.5 * (zmin + zmax)
        bot = c[c[:, 2] <= zmid][:, :2]
        if bot.shape[0] != 4:
            bot = c[np.argsort(c[:, 2])[:4]][:, :2]
        center = bot.mean(axis=0)
        angles = np.arctan2(bot[:, 1] - center[1], bot[:, 0] - center[0])
        bot = bot[np.argsort(angles)]
        p0, p1, p3 = bot[0], bot[1], bot[3]
        u, v = p1 - p0, p3 - p0
        uu, vv = float(np.dot(u, u)) + 1e-8, float(np.dot(v, v)) + 1e-8
        rel = psub - p0
        pu, pv = rel @ u, rel @ v
        counts[i] = int(np.sum((pu >= 0.0) & (pu <= uu) & (pv >= 0.0) & (pv <= vv)))
    return counts


def split_fit_eval_sequences(wrapper: HEALBaseWrapper, dataset, train_dir: str):
    """按车数均衡交错将 15 条完整序列互斥划分为 Fit(8条) 与 Eval(7条)，各自抽 60 帧。"""
    sample_seqs = wrapper.get_sample_sequences(dataset)
    seq_to_idxs = OrderedDict()
    for i, s in enumerate(sample_seqs):
        seq_to_idxs.setdefault(s, []).append(i)

    by_cav = OrderedDict()
    for s in sorted(seq_to_idxs.keys()):
        cavs = len([d for d in os.listdir(os.path.join(train_dir, s)) if os.path.isdir(os.path.join(train_dir, s, d))])
        by_cav.setdefault(cavs, []).append(s)

    fit_seqs, eval_seqs = [], []
    for cavs, slist in sorted(by_cav.items()):
        for rank, s in enumerate(sorted(slist)):
            if rank % 2 == 0:
                fit_seqs.append((s, cavs))
            else:
                eval_seqs.append((s, cavs))

    def _alloc_frames(seq_pairs, target_n=60):
        total_f = sum(len(seq_to_idxs[s]) for s, _ in seq_pairs)
        raw = {s: max(4.0, target_n * len(seq_to_idxs[s]) / float(total_f)) for s, _ in seq_pairs}
        cnt = {s: int(math.floor(raw[s])) for s, _ in seq_pairs}
        diff = target_n - sum(cnt.values())
        if diff > 0:
            for s in sorted(cnt.keys(), key=lambda k: raw[k] - cnt[k], reverse=True)[:diff]:
                cnt[s] += 1
        elif diff < 0:
            for s in sorted(cnt.keys(), key=lambda k: cnt[k], reverse=True)[:(-diff)]:
                cnt[s] -= 1

        picked_list = []
        plan = OrderedDict()
        for s, cavs in seq_pairs:
            idxs = seq_to_idxs[s]
            n_p = cnt[s]
            pos = np.linspace(0, len(idxs) - 1, n_p, dtype=int)
            picked = [int(idxs[p]) for p in pos]
            for idx_v in picked:
                picked_list.append((idx_v, s, cavs))
            plan[s] = {"raw_cavs": cavs, "total_frames": len(idxs), "sampled": n_p, "indices": picked}
        return picked_list, plan

    fit_list, fit_plan = _alloc_frames(fit_seqs, 60)
    eval_list, eval_plan = _alloc_frames(eval_seqs, 60)
    return fit_list, fit_plan, eval_list, eval_plan


def summarize_np(arr: np.ndarray) -> dict:
    """对 1D float numpy 数组汇总统计量。"""
    if arr.size == 0:
        return {"count": 0, "mean": None, "std": None, "median": None, "p75": None, "p90": None, "p95": None, "p99": None}
    a = arr.astype(np.float64)
    return {
        "count": int(a.size),
        "mean": round(float(np.mean(a)), 5),
        "std": round(float(np.std(a)), 5),
        "median": round(float(np.median(a)), 5),
        "p75": round(float(np.percentile(a, 75)), 5),
        "p90": round(float(np.percentile(a, 90)), 5),
        "p95": round(float(np.percentile(a, 95)), 5),
        "p99": round(float(np.percentile(a, 99)), 5),
    }


def compute_auroc(neg_scores: np.ndarray, pos_scores: np.ndarray) -> float:
    """精确 Mann-Whitney U 秩和法计算 AUROC（无需 sklearn 依赖，正确处理并列值）。"""
    all_s = np.concatenate([neg_scores, pos_scores], axis=0)
    n0, n1 = neg_scores.size, pos_scores.size
    if n0 == 0 or n1 == 0:
        return 0.0
    order = np.argsort(all_s)
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    while i < all_s.size:
        j = i
        while j + 1 < all_s.size and all_s[order[j + 1]] == all_s[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    pos_rank_sum = ranks[n0:].sum()
    u1 = pos_rank_sum - n1 * (n1 + 1) / 2.0
    return float(u1 / (n0 * n1))


def extract_clean_frame_bundle(
    wrapper: HEALBaseWrapper,
    ds_m1,
    ds_m2,
    sample_idx: int,
    seq: str,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    device: torch.device,
):
    """提取单帧 m1+m1 与 m1+m2 的公共特征、对齐 L0 误差图、距离图、本车可见性图与样本元数据。"""
    from opencood.tools import train_utils
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

    b2_raw = ds_m2.collate_batch_test([ds_m2[sample_idx]])
    b1_raw = ds_m1.collate_batch_test([ds_m1[sample_idx]])
    b2 = train_utils.to_device(b2_raw, device)["ego"]
    b1 = train_utils.to_device(b1_raw, device)["ego"]

    p2 = wrapper.extract_protocol_features(b2)
    p1 = wrapper.extract_protocol_features(b1)
    assert p2["agent_modality_list"] == ["m1", "m2"]
    assert p1["agent_modality_list"] == ["m1", "m1"]

    ego_pts_np = load_ego_lidar_np(ds_m2, sample_idx)
    sigma_alea_0 = compute_ego_beam_visibility_256(ego_pts_np, grid_x, grid_y, num_beams=64)  # (1,1,256,256)

    with torch.no_grad():
        align_corners = wrapper.model.pyramid_backbone.align_corners
        l0_1 = wrapper.model.pyramid_backbone.get_multiscale_feature(p1["heter_feature_2d"])[0]
        l0_2 = wrapper.model.pyramid_backbone.get_multiscale_feature(p2["heter_feature_2d"])[0]

        z0 = l0_1[0:1]
        z1_ego = warp_affine_simple(l0_1[1:2], p1["affine_matrix"][0, 0, 1:2], (PROTO_H, PROTO_W), align_corners=align_corners)
        z2_ego = warp_affine_simple(l0_2[1:2], p2["affine_matrix"][0, 0, 1:2], (PROTO_H, PROTO_W), align_corners=align_corners)

        # 公共评估与攻击空间覆盖域（由 m2 有效中心传输区投到 ego）
        cov_m2 = warp_affine_simple(
            p2["cam_valid_mask"][1:2], p2["affine_matrix"][0, 0, 1:2], (PROTO_H, PROTO_W), align_corners=False
        ) > 0.5

        # 协作者传感中心距离 d_cav、本车中心距离 d_ego 与协作者局部坐标系角度
        cav_xy = b2["pairwise_t_matrix"][0, 1, 0, :2, 3]
        dist_cav = torch.sqrt((grid_x - cav_xy[0]) ** 2 + (grid_y - cav_xy[1]) ** 2).view(1, 1, PROTO_H, PROTO_W)
        dist_ego = torch.sqrt(grid_x ** 2 + grid_y ** 2).view(1, 1, PROTO_H, PROTO_W)

        # 计算网格中心在协作者局部坐标系的极角 (用于判断 4 路相机 45°/135° 拼接缝隙)
        t_e2c = b2["pairwise_t_matrix"][0, 0, 1]
        lx = t_e2c[0, 0] * grid_x + t_e2c[0, 1] * grid_y + t_e2c[0, 3]
        ly = t_e2c[1, 0] * grid_x + t_e2c[1, 1] * grid_y + t_e2c[1, 3]
        theta_cav_deg = torch.rad2deg(torch.atan2(ly, lx)).view(1, 1, PROTO_H, PROTO_W)

        ego_out = wrapper.forward_single_agent(p1["heter_feature_2d"][0:1])
        cls_prob_ego = torch.sigmoid(ego_out["cls_preds"]).max(dim=1, keepdim=True)[0]

        mse_m1 = ((z0 - z1_ego) ** 2).mean(dim=1, keepdim=True)
        mse_m2 = ((z0 - z2_ego) ** 2).mean(dim=1, keepdim=True)
        cos_m1 = 1.0 - F.cosine_similarity(z0, z1_ego, dim=1, eps=1e-6).unsqueeze(1)
        cos_m2 = 1.0 - F.cosine_similarity(z0, z2_ego, dim=1, eps=1e-6).unsqueeze(1)

        def _net_norm(z):
            return (z - z.mean(dim=(2, 3), keepdim=True)).norm(dim=1, keepdim=True)

        nn0, nn1, nn2 = _net_norm(z0), _net_norm(z1_ego), _net_norm(z2_ego)
        l2_0, l2_1, l2_2 = z0.norm(dim=1, keepdim=True), z1_ego.norm(dim=1, keepdim=True), z2_ego.norm(dim=1, keepdim=True)

        # GT 框及其在本车点云下的真实落点数
        out_gt = b2["object_bbx_center"][0]  # (max_num, 7)
        gt_mask = b2["object_bbx_mask"][0] > 0.5
        gt_valid_centers = out_gt[gt_mask].detach().cpu().numpy()
        gt_corners = None
        gt_pts = None
        if len(gt_valid_centers) > 0:
            from opencood.utils import box_utils
            gt_corners = box_utils.boxes_to_corners_3d(gt_valid_centers, order="hwl")
            gt_pts = count_points_in_corners3d(ego_pts_np, gt_corners)

    # 释放占用显存的原始图像与体素字典
    del b1_raw, b2_raw, b1, b2

    return {
        "sample_idx": sample_idx,
        "seq": seq,
        "p1": p1,
        "p2": p2,
        "z0": z0,
        "z1_ego": z1_ego,
        "z2_ego": z2_ego,
        "cov_m2": cov_m2,
        "sigma_alea_0": sigma_alea_0,
        "dist_cav": dist_cav,
        "dist_ego": dist_ego,
        "theta_cav_deg": theta_cav_deg,
        "cls_prob_ego": cls_prob_ego,
        "mse_m1": mse_m1,
        "mse_m2": mse_m2,
        "cos_m1": cos_m1,
        "cos_m2": cos_m2,
        "nn0": nn0,
        "nn1": nn1,
        "nn2": nn2,
        "l2_0": l2_0,
        "l2_1": l2_1,
        "l2_2": l2_2,
        "gt_centers_np": gt_valid_centers,
        "gt_pts": gt_pts,
    }


def _assign_bin_indices(dist_cav: torch.Tensor, sigma_alea_0: torch.Tensor):
    """返回距离分档索引 (0..3) 与本车可见性分档索引 (0:Clear, 1:Sparse, 2:Occluded)。"""
    d_idx = torch.zeros_like(dist_cav, dtype=torch.long)
    d_idx = torch.where(dist_cav >= 15.0, torch.ones_like(d_idx), d_idx)
    d_idx = torch.where(dist_cav >= 30.0, torch.full_like(d_idx, 2), d_idx)
    d_idx = torch.where(dist_cav >= 45.0, torch.full_like(d_idx, 3), d_idx)

    v_idx = torch.zeros_like(sigma_alea_0, dtype=torch.long)
    v_idx = torch.where(sigma_alea_0 >= 0.35, torch.ones_like(v_idx), v_idx)
    v_idx = torch.where(sigma_alea_0 >= 0.70, torch.full_like(v_idx, 2), v_idx)
    return d_idx, v_idx


def _compute_local_energy(mse_map: torch.Tensor, cov_mask: torch.Tensor) -> torch.Tensor:
    """在公共覆盖域内计算 3x3 局部均值冲突能量图。"""
    return F.avg_pool2d(mse_map * cov_mask.float(), kernel_size=3, stride=1, padding=1)


def _compute_meantopk_from_norm_map(norm_energy: torch.Tensor, cov_mask: torch.Tensor, k: int = 12) -> float:
    """在公共覆盖域内提取 MeanTopK(k) 帧级局部分数。"""
    vals = norm_energy[cov_mask]
    if vals.numel() == 0:
        return 0.0
    k_eff = min(k, int(vals.numel()))
    return float(torch.topk(vals, k=k_eff).values.mean().item())


def compute_adv_l0_mse(
    wrapper: HEALBaseWrapper,
    z0: torch.Tensor,
    feat_adv: torch.Tensor,
    affine_matrix: torch.Tensor,
) -> torch.Tensor:
    """从对抗性传输特征提取协作者 pyramid_L0 并对齐至 ego 算 MSE 图。"""
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

    with torch.no_grad():
        align_corners = wrapper.model.pyramid_backbone.align_corners
        l0_adv = wrapper.model.pyramid_backbone.get_multiscale_feature(feat_adv)[0]
        zcav_ego = warp_affine_simple(l0_adv[1:2], affine_matrix[0, 0, 1:2], (PROTO_H, PROTO_W), align_corners=align_corners)
        return ((z0 - zcav_ego) ** 2).mean(dim=1, keepdim=True)


class ModuleABAccumulator:
    """流式累积模块 A (0-15m 尖峰溯源) 与模块 B (m1/m2 可见性×距离分布)，避免显存堆积。"""

    def __init__(self):
        self.spike_sub = {lbl: [] for lbl in SPIKE_SUB_LABELS}
        self.spike_attr = {
            "total_spikes_gt25": 0,
            "near_cav_lt3.5m": 0,
            "near_ego_lt3.5m": 0,
            "cam_seam_pm10deg": 0,
            "ego_fg_gt0.20": 0,
            "z2_high_z0_low": 0,
            "z0_high": 0,
            "other": 0,
        }
        self.top_spikes = []
        self.ablation_0_15 = {k: [] for k in ["full", "excl_cav_3.5m", "excl_ego_3.5m", "excl_both_3.5m"]}
        self.mtk_0_15_check = {k: [] for k in ["full", "excl_cav_3.5m", "excl_ego_3.5m", "excl_both_3.5m"]}
        self.vis_dist = {
            m: {
                v: {d: {"all_mse": [], "act_mse": [], "cos": [], "e_pool": []} for d in DIST_LABELS}
                for v in VIS_LABELS
            }
            for m in ["m1", "m2"]
        }
        self.gt_layer_stats = {
            m: {k: {"mse": [], "cos": []} for k in ["Clear_pts15+", "Sparse_1_14", "Occluded_0"]}
            for m in ["m1", "m2"]
        }

    def update(self, b: dict):
        cov = b["cov_m2"]
        d_idx, v_idx = _assign_bin_indices(b["dist_cav"], b["sigma_alea_0"])
        e1 = _compute_local_energy(b["mse_m1"], cov)
        e2 = _compute_local_energy(b["mse_m2"], cov)

        d_cav = b["dist_cav"]
        d_ego = b["dist_ego"]
        theta_c = b["theta_cav_deg"]
        mse2 = b["mse_m2"]

        for (lo, hi), lbl in zip(SPIKE_SUB_BINS, SPIKE_SUB_LABELS):
            mk = cov & (d_cav >= lo) & (d_cav < hi)
            if mk.any():
                self.spike_sub[lbl].append(mse2[mk].detach().cpu().numpy())

        m015 = cov & (d_cav < 15.0)
        if m015.any():
            self.ablation_0_15["full"].append(mse2[m015].detach().cpu().numpy())
            m_nc = m015 & (d_cav >= 3.5)
            m_ne = m015 & (d_ego >= 3.5)
            m_nb = m015 & (d_cav >= 3.5) & (d_ego >= 3.5)
            if m_nc.any(): self.ablation_0_15["excl_cav_3.5m"].append(mse2[m_nc].detach().cpu().numpy())
            if m_ne.any(): self.ablation_0_15["excl_ego_3.5m"].append(mse2[m_ne].detach().cpu().numpy())
            if m_nb.any(): self.ablation_0_15["excl_both_3.5m"].append(mse2[m_nb].detach().cpu().numpy())

        for key_tag, sub_mk in [
            ("full", cov),
            ("excl_cav_3.5m", cov & (d_cav >= 3.5)),
            ("excl_ego_3.5m", cov & (d_ego >= 3.5)),
            ("excl_both_3.5m", cov & (d_cav >= 3.5) & (d_ego >= 3.5)),
        ]:
            e_sub = _compute_local_energy(b["mse_m2"], sub_mk)
            self.mtk_0_15_check[key_tag].append(_compute_meantopk_from_norm_map(e_sub, sub_mk, k=12))

        spk_mk = m015 & (mse2 > 25.0)
        if spk_mk.any():
            coords = torch.nonzero(spk_mk[0, 0], as_tuple=False)
            self.spike_attr["total_spikes_gt25"] += int(coords.shape[0])
            for rr, cc in coords.tolist():
                val = float(mse2[0, 0, rr, cc].item())
                dc = float(d_cav[0, 0, rr, cc].item())
                de = float(d_ego[0, 0, rr, cc].item())
                ang = float(theta_c[0, 0, rr, cc].item())
                seam_diff = min(abs(ang - s_ang) for s_ang in [-135.0, -45.0, 45.0, 135.0])
                is_seam = seam_diff <= 10.0
                fgp = float(b["cls_prob_ego"][0, 0, rr, cc].item())
                l0 = float(b["l2_0"][0, 0, rr, cc].item())
                l2v = float(b["l2_2"][0, 0, rr, cc].item())
                if dc < 3.5:
                    self.spike_attr["near_cav_lt3.5m"] += 1
                if de < 3.5:
                    self.spike_attr["near_ego_lt3.5m"] += 1
                if is_seam:
                    self.spike_attr["cam_seam_pm10deg"] += 1
                if fgp > 0.20:
                    self.spike_attr["ego_fg_gt0.20"] += 1
                if l2v > 15.0 and l0 < 10.0:
                    self.spike_attr["z2_high_z0_low"] += 1
                elif l0 > 15.0:
                    self.spike_attr["z0_high"] += 1
                if (dc >= 3.5) and (de >= 3.5) and (not is_seam) and (fgp <= 0.20) and (l0 <= 15.0) and (l2v <= 15.0):
                    self.spike_attr["other"] += 1

                self.top_spikes.append({
                    "sample_idx": b["sample_idx"],
                    "seq": b["seq"],
                    "hw": [rr, cc],
                    "mse": round(val, 2),
                    "d_cav": round(dc, 2),
                    "d_ego": round(de, 2),
                    "theta_cav_deg": round(ang, 1),
                    "seam_diff_deg": round(seam_diff, 1),
                    "l2_ego": round(l0, 2),
                    "l2_m2": round(l2v, 2),
                    "cls_ego": round(fgp, 3),
                    "sigma_alea_0": round(float(b["sigma_alea_0"][0, 0, rr, cc].item()), 3),
                })

        act_m1 = cov & ((b["nn0"] > 4.0) | (b["nn1"] > 4.0))
        act_m2 = cov & ((b["nn0"] > 4.0) | (b["nn2"] > 5.8))
        for vi, vlbl in enumerate(VIS_LABELS):
            for di, dlbl in enumerate(DIST_LABELS):
                cell = cov & (v_idx == vi) & (d_idx == di)
                if cell.any():
                    self.vis_dist["m1"][vlbl][dlbl]["all_mse"].append(b["mse_m1"][cell].detach().cpu().numpy())
                    self.vis_dist["m2"][vlbl][dlbl]["all_mse"].append(b["mse_m2"][cell].detach().cpu().numpy())
                    self.vis_dist["m1"][vlbl][dlbl]["cos"].append(b["cos_m1"][cell].detach().cpu().numpy())
                    self.vis_dist["m2"][vlbl][dlbl]["cos"].append(b["cos_m2"][cell].detach().cpu().numpy())
                    self.vis_dist["m1"][vlbl][dlbl]["e_pool"].append(e1[cell].detach().cpu().numpy())
                    self.vis_dist["m2"][vlbl][dlbl]["e_pool"].append(e2[cell].detach().cpu().numpy())
                c_a1 = act_m1 & (v_idx == vi) & (d_idx == di)
                if c_a1.any():
                    self.vis_dist["m1"][vlbl][dlbl]["act_mse"].append(b["mse_m1"][c_a1].detach().cpu().numpy())
                c_a2 = act_m2 & (v_idx == vi) & (d_idx == di)
                if c_a2.any():
                    self.vis_dist["m2"][vlbl][dlbl]["act_mse"].append(b["mse_m2"][c_a2].detach().cpu().numpy())

        if b["gt_pts"] is not None and len(b["gt_pts"]) > 0:
            for box_c, npts in zip(b["gt_centers_np"], b["gt_pts"]):
                gx, gy = float(box_c[0]), float(box_c[1])
                wc = int(math.floor((gx - MIN_XY) / GRID_RES))
                hc = int(math.floor((gy - MIN_XY) / GRID_RES))
                if not (0 <= hc < PROTO_H and 0 <= wc < PROTO_W):
                    continue
                if not bool(cov[0, 0, hc, wc].item()):
                    continue
                layer = "Clear_pts15+" if npts >= 15 else ("Sparse_1_14" if npts >= 1 else "Occluded_0")
                self.gt_layer_stats["m1"][layer]["mse"].append(float(b["mse_m1"][0, 0, hc, wc].item()))
                self.gt_layer_stats["m1"][layer]["cos"].append(float(b["cos_m1"][0, 0, hc, wc].item()))
                self.gt_layer_stats["m2"][layer]["mse"].append(float(b["mse_m2"][0, 0, hc, wc].item()))
                self.gt_layer_stats["m2"][layer]["cos"].append(float(b["cos_m2"][0, 0, hc, wc].item()))

    def finalize(self):
        top_spikes = sorted(self.top_spikes, key=lambda x: x["mse"], reverse=True)[:12]
        mod_a = {
            "sub_distance_0_15m": {
                lbl: {
                    **summarize_np(np.concatenate(self.spike_sub[lbl]) if self.spike_sub[lbl] else np.empty(0)),
                    "spike_frac_gt25": round(float(np.mean(np.concatenate(self.spike_sub[lbl]) > 25.0)), 6) if self.spike_sub[lbl] else 0.0,
                }
                for lbl in SPIKE_SUB_LABELS
            },
            "spike_attribution_counts": self.spike_attr,
            "top_12_extreme_spikes": top_spikes,
            "deadzone_exclusion_effect_0_15m": {
                k: summarize_np(np.concatenate(v) if v else np.empty(0)) for k, v in self.ablation_0_15.items()
            },
            "deadzone_exclusion_effect_frame_meantopk12_m2": {
                k: summarize_np(np.asarray(v, dtype=np.float64)) for k, v in self.mtk_0_15_check.items()
            },
        }

        mod_b = {"m1": OrderedDict(), "m2": OrderedDict(), "gt_box_visibility_strata": OrderedDict(), "subcell_pixel_fpr_analysis": OrderedDict()}
        for m in ["m1", "m2"]:
            all_e_m = []
            for vlbl in VIS_LABELS:
                mod_b[m][vlbl] = OrderedDict()
                for dlbl in DIST_LABELS:
                    cell = self.vis_dist[m][vlbl][dlbl]
                    arr_all = np.concatenate(cell["all_mse"]) if cell["all_mse"] else np.empty(0)
                    arr_act = np.concatenate(cell["act_mse"]) if cell["act_mse"] else np.empty(0)
                    arr_cos = np.concatenate(cell["cos"]) if cell["cos"] else np.empty(0)
                    arr_ep = np.concatenate(cell["e_pool"]) if cell["e_pool"] else np.empty(0)
                    if arr_ep.size > 0:
                        all_e_m.append(arr_ep)
                    mod_b[m][vlbl][dlbl] = {
                        "all_mse": summarize_np(arr_all),
                        "active_mse": summarize_np(arr_act),
                        "cos_dev": summarize_np(arr_cos),
                        "e_pool_3x3": summarize_np(arr_ep),
                    }
            e_cat = np.concatenate(all_e_m)
            sensor_p95 = float(np.percentile(e_cat, 95))
            cell_fpr_sensor_only = {}
            for vlbl in VIS_LABELS:
                for dlbl in DIST_LABELS:
                    arr_ep = np.concatenate(self.vis_dist[m][vlbl][dlbl]["e_pool"]) if self.vis_dist[m][vlbl][dlbl]["e_pool"] else np.empty(0)
                    cell_fpr_sensor_only[f"{vlbl}_{dlbl}"] = round(float(np.mean(arr_ep > sensor_p95)), 5) if arr_ep.size else None

            cell_fpr_dist_only = {}
            for dlbl in DIST_LABELS:
                d_ep = np.concatenate([
                    np.concatenate(self.vis_dist[m][vlbl][dlbl]["e_pool"])
                    for vlbl in VIS_LABELS if self.vis_dist[m][vlbl][dlbl]["e_pool"]
                ])
                d_p95 = float(np.percentile(d_ep, 95))
                for vlbl in VIS_LABELS:
                    arr_ep = np.concatenate(self.vis_dist[m][vlbl][dlbl]["e_pool"])
                    cell_fpr_dist_only[f"{vlbl}_{dlbl}"] = round(float(np.mean(arr_ep > d_p95)), 5)

            mod_b["subcell_pixel_fpr_analysis"][m] = {
                "sensor_global_e_pool_p95": round(sensor_p95, 4),
                "per_cell_fpr_under_sensor_only_p95": cell_fpr_sensor_only,
                "per_cell_fpr_under_sensor_dist_p95": cell_fpr_dist_only,
                "max_cell_fpr_sensor_only": round(max(v for v in cell_fpr_sensor_only.values() if v is not None), 5),
                "max_cell_fpr_sensor_dist": round(max(v for v in cell_fpr_dist_only.values() if v is not None), 5),
            }

        for m in ["m1", "m2"]:
            mod_b["gt_box_visibility_strata"][m] = {
                k: {
                    "mse": summarize_np(np.asarray(v["mse"], dtype=np.float64)),
                    "cos_dev": summarize_np(np.asarray(v["cos"], dtype=np.float64)),
                }
                for k, v in self.gt_layer_stats[m].items()
            }
        return mod_a, mod_b


def fit_4tier_calibrators(fit_bundles: list) -> dict:
    """严格仅在 Fit-Set 良性样本上拟合 ①/②/③/④ 四组门限的纯几何-可见性归一化参数。"""
    # 收集 Fit-Set 的局部能量分布
    grid_pool = {
        m: {
            "all": [],
            "d": {di: [] for di in range(4)},
            "dv": {(di, vi): [] for di in range(4) for vi in range(3)},
        }
        for m in ["m1", "m2"]
    }
    raw_frames = {m: [] for m in ["m1", "m2"]}

    for b in fit_bundles:
        cov = b["cov_m2"]
        d_idx, v_idx = _assign_bin_indices(b["dist_cav"], b["sigma_alea_0"])
        for m, mse_t in [("m1", b["mse_m1"]), ("m2", b["mse_m2"])]:
            e_map = _compute_local_energy(mse_t, cov)
            raw_frames[m].append(_compute_meantopk_from_norm_map(e_map, cov, k=12))
            if cov.any():
                grid_pool[m]["all"].append(e_map[cov].detach().cpu().numpy())
            for di in range(4):
                dm = cov & (d_idx == di)
                if dm.any():
                    grid_pool[m]["d"][di].append(e_map[dm].detach().cpu().numpy())
                for vi in range(3):
                    dvm = dm & (v_idx == vi)
                    if dvm.any():
                        grid_pool[m]["dv"][(di, vi)].append(e_map[dvm].detach().cpu().numpy())

    # 为 ③(m,d) 和 ④(m,d,v) 构建网格局部尺度表 (取分档 P95 局部尾尺度，含少样本平滑兜底)
    calib = {"sensor_frame_p95": {}, "dist_p95": {}, "dist_vis_p95": {}, "tier_frame_p95": {}}
    for m in ["m1", "m2"]:
        m_all = np.concatenate(grid_pool[m]["all"])
        m_p95 = max(float(np.percentile(m_all, 95)), 0.5)
        calib["sensor_frame_p95"][m] = round(float(np.percentile(raw_frames[m], 95)), 6)
        calib["dist_p95"][m] = []
        calib["dist_vis_p95"][m] = []
        for di in range(4):
            d_arr = np.concatenate(grid_pool[m]["d"][di]) if grid_pool[m]["d"][di] else np.empty(0)
            d_val = max(float(np.percentile(d_arr, 95)), 0.5) if d_arr.size >= 200 else m_p95
            calib["dist_p95"][m].append(round(d_val, 6))
            row = []
            for vi in range(3):
                dv_arr = np.concatenate(grid_pool[m]["dv"][(di, vi)]) if grid_pool[m]["dv"][(di, vi)] else np.empty(0)
                if dv_arr.size >= 200:
                    dv_val = max(float(np.percentile(dv_arr, 95)), 0.5)
                elif dv_arr.size >= 30:
                    w_blend = dv_arr.size / 200.0
                    dv_val = w_blend * max(float(np.percentile(dv_arr, 95)), 0.5) + (1.0 - w_blend) * d_val
                else:
                    dv_val = d_val
                row.append(round(dv_val, 6))
            calib["dist_vis_p95"][m].append(row)

    # 在 Fit-Set 上反算 ③ 与 ④ 在各自模态下的帧级 MeanTopK(12) 95% 分位，用于跨模态帧级对齐
    for tier_name in ["tier3_sensor_dist", "tier4_sensor_dist_vis"]:
        calib["tier_frame_p95"][tier_name] = {}
        for m in ["m1", "m2"]:
            fscores = []
            for b in fit_bundles:
                sc = score_frame_with_tier(b, m, b["mse_m1"] if m == "m1" else b["mse_m2"], calib, tier_name, apply_frame_norm=False)
                fscores.append(sc)
            calib["tier_frame_p95"][tier_name][m] = round(max(float(np.percentile(fscores, 95)), 1e-4), 6)

    return calib


def score_frame_with_tier(
    bundle: dict,
    mod: str,
    mse_map: torch.Tensor,
    calib: dict,
    tier_name: str,
    apply_frame_norm: bool = True,
) -> float:
    """按指定门限级别 (①/②/③/④) 计算单帧审计得分，严格保持分母与外部特征独立。"""
    cov = bundle["cov_m2"]
    e_map = _compute_local_energy(mse_map, cov)

    if tier_name == "tier1_unified":
        return _compute_meantopk_from_norm_map(e_map, cov, k=12)

    if tier_name == "tier2_sensor_only":
        raw = _compute_meantopk_from_norm_map(e_map, cov, k=12)
        return raw / max(calib["sensor_frame_p95"][mod], 1e-4)

    d_idx, v_idx = _assign_bin_indices(bundle["dist_cav"], bundle["sigma_alea_0"])
    if tier_name == "tier3_sensor_dist":
        scales = torch.tensor(calib["dist_p95"][mod], device=mse_map.device, dtype=mse_map.dtype)
        denom = scales[d_idx]
        raw = _compute_meantopk_from_norm_map(e_map / denom, cov, k=12)
        if apply_frame_norm:
            return raw / max(calib["tier_frame_p95"]["tier3_sensor_dist"][mod], 1e-4)
        return raw

    if tier_name == "tier4_sensor_dist_vis":
        mat = torch.tensor(calib["dist_vis_p95"][mod], device=mse_map.device, dtype=mse_map.dtype)  # (4, 3)
        denom = mat[d_idx, v_idx]
        raw = _compute_meantopk_from_norm_map(e_map / denom, cov, k=12)
        if apply_frame_norm:
            return raw / max(calib["tier_frame_p95"]["tier4_sensor_dist_vis"][mod], 1e-4)
        return raw

    raise ValueError(f"未知 tier_name: {tier_name}")


class ModuleCEvaluator:
    """流式在 Eval-Set 上生成 4 类小预算/局部攻击并比较 4 组门限在相等 5% 总误报率下的检出率。"""

    def __init__(self, wrapper: HEALBaseWrapper, calib: dict):
        self.wrapper = wrapper
        self.calib = calib
        self.tiers = ["tier1_unified", "tier2_sensor_only", "tier3_sensor_dist", "tier4_sensor_dist_vis"]
        self.attacks = ["scaled_pgd_0.5sigma", "scaled_pgd_1.0sigma", "local_removal_1.0sigma", "local_ghost_2.0sigma"]
        self.scores = {t: {m: {"benign": [], **{a: [] for a in self.attacks}} for m in ["m1", "m2"]} for t in self.tiers}
        self.atk_gen_stats = {m: {a: {"linf": [], "prob_drop": [], "ghost_prob": []} for a in self.attacks} for m in ["m1", "m2"]}

    def eval_frame(self, b: dict):
        cov_local_for_m1 = b["p2"]["cam_valid_mask"][1:2]
        for mod, p_dict in [("m1", b["p1"]), ("m2", b["p2"])]:
            feat = p_dict["heter_feature_2d"]
            rlen = p_dict["record_len"]
            amat = p_dict["affine_matrix"]
            mods = p_dict["agent_modality_list"]
            vmask = p_dict["cam_valid_mask"]
            preg = cov_local_for_m1 if mod == "m1" else None

            mse_benign = b["mse_m1"] if mod == "m1" else b["mse_m2"]
            for t in self.tiers:
                self.scores[t][mod]["benign"].append(score_frame_with_tier(b, mod, mse_benign, self.calib, t))

            f_05, st_05 = generate_scaled_pgd_removal_heal(
                self.wrapper, feat, rlen, amat, mods, vmask, attacker_idx=1, sigma_mult=0.5, steps=10, perturb_region_mask=preg
            )
            f_10, st_10 = generate_scaled_pgd_removal_heal(
                self.wrapper, feat, rlen, amat, mods, vmask, attacker_idx=1, sigma_mult=1.0, steps=10, perturb_region_mask=preg
            )
            f_lr, st_lr = generate_local_removal_heal(
                self.wrapper, feat, rlen, amat, mods, vmask, attacker_idx=1, num_targets=2, patch_radius=5, sigma_mult=1.0, steps=12, perturb_region_mask=preg
            )
            f_lg, st_lg = generate_local_ghost_spoofing_heal(
                self.wrapper, feat, rlen, amat, mods, vmask, attacker_idx=1, num_spoofs=2, patch_radius=5, sigma_mult=2.0, steps=12, frame_seed=b["sample_idx"], perturb_region_mask=preg
            )

            with torch.no_grad():
                o_clean = self.wrapper.forward_from_protocol(feat, rlen, amat, mods)
                if st_lr.get("target_centers_hw"):
                    o_lr = self.wrapper.forward_from_protocol(f_lr, rlen, amat, mods)
                    drops = []
                    for hc, wc in st_lr["target_centers_hw"]:
                        pc = float(torch.sigmoid(o_clean["cls_preds"])[0, :, hc, wc].max().item())
                        pa = float(torch.sigmoid(o_lr["cls_preds"])[0, :, hc, wc].max().item())
                        drops.append(pc - pa)
                    self.atk_gen_stats[mod]["local_removal_1.0sigma"]["prob_drop"].append(float(np.mean(drops)))
                if st_lg.get("anchors_hwr"):
                    o_lg = self.wrapper.forward_from_protocol(f_lg, rlen, amat, mods)
                    gp = [float(torch.sigmoid(o_lg["cls_preds"])[0, rot, hc, wc].item()) for hc, wc, rot in st_lg["anchors_hwr"]]
                    self.atk_gen_stats[mod]["local_ghost_2.0sigma"]["ghost_prob"].append(float(np.max(gp)))

            for aname, f_adv, st in [
                ("scaled_pgd_0.5sigma", f_05, st_05),
                ("scaled_pgd_1.0sigma", f_10, st_10),
                ("local_removal_1.0sigma", f_lr, st_lr),
                ("local_ghost_2.0sigma", f_lg, st_lg),
            ]:
                self.atk_gen_stats[mod][aname]["linf"].append(st["linf_max"])
                mse_adv = compute_adv_l0_mse(self.wrapper, b["z0"], f_adv, amat)
                for t in self.tiers:
                    self.scores[t][mod][aname].append(score_frame_with_tier(b, mod, mse_adv, self.calib, t))

    def finalize(self):
        comparison = OrderedDict()
        for t in self.tiers:
            ben_m1 = np.asarray(self.scores[t]["m1"]["benign"], dtype=np.float64)
            ben_m2 = np.asarray(self.scores[t]["m2"]["benign"], dtype=np.float64)
            ben_all = np.concatenate([ben_m1, ben_m2], axis=0)

            sorted_b = np.sort(ben_all)
            idx_cut = int(round(len(sorted_b) * 0.95)) - 1
            th_5pct = 0.5 * (float(sorted_b[idx_cut]) + float(sorted_b[idx_cut + 1]))
            th_fit = self.calib.get("tier1_fit_p95", float(np.percentile(ben_all, 95))) if t == "tier1_unified" else 1.0

            res_t = {
                "th_at_exact_5pct_total_fpr": round(th_5pct, 5),
                "eval_fpr_at_5pct_thresh": {
                    "total": round(float(np.mean(ben_all > th_5pct)), 5),
                    "m1": round(float(np.mean(ben_m1 > th_5pct)), 5),
                    "m2": round(float(np.mean(ben_m2 > th_5pct)), 5),
                },
                "fit_frozen_thresh_generalization": {
                    "thresh": round(th_fit, 5),
                    "eval_fpr_total": round(float(np.mean(ben_all > th_fit)), 5),
                    "eval_fpr_m1": round(float(np.mean(ben_m1 > th_fit)), 5),
                    "eval_fpr_m2": round(float(np.mean(ben_m2 > th_fit)), 5),
                },
                "attacks": OrderedDict(),
            }

            for a in self.attacks:
                pos_m1 = np.asarray(self.scores[t]["m1"][a], dtype=np.float64)
                pos_m2 = np.asarray(self.scores[t]["m2"][a], dtype=np.float64)
                pos_all = np.concatenate([pos_m1, pos_m2], axis=0)
                res_t["attacks"][a] = {
                    "tpr_at_5pct_total_fpr": {
                        "m1": round(float(np.mean(pos_m1 > th_5pct)), 5),
                        "m2": round(float(np.mean(pos_m2 > th_5pct)), 5),
                        "overall": round(float(np.mean(pos_all > th_5pct)), 5),
                    },
                    "auroc": {
                        "m1": round(compute_auroc(ben_m1, pos_m1), 5),
                        "m2": round(compute_auroc(ben_m2, pos_m2), 5),
                        "overall_vs_all_benign": round(compute_auroc(ben_all, pos_all), 5),
                    },
                    "tpr_at_fit_thresh": {
                        "m1": round(float(np.mean(pos_m1 > th_fit)), 5),
                        "m2": round(float(np.mean(pos_m2 > th_fit)), 5),
                        "overall": round(float(np.mean(pos_all > th_fit)), 5),
                    },
                }
            comparison[t] = res_t

        atk_meta = {
            m: {
                a: {
                    "linf_mean": round(float(np.mean(v["linf"])), 4),
                    **({"mean_target_prob_drop": round(float(np.mean(v["prob_drop"])), 4)} if v["prob_drop"] else {}),
                    **({"mean_max_ghost_prob": round(float(np.mean(v["ghost_prob"])), 4)} if v["ghost_prob"] else {}),
                }
                for a, v in self.atk_gen_stats[m].items()
            }
            for m in ["m1", "m2"]
        }
        return comparison, atk_meta


def plot_vis_dist_and_spikes(mod_a: dict, mod_b: dict, out_png: str):
    """绘制图 1：m1/m2 可见性×距离热力图及摄像头 0-15m 尖峰细分归因图。"""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.0, "axes.titlesize": 11.2, "axes.titleweight": "bold"})

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9), dpi=300)
    fig.subplots_adjust(wspace=0.30, top=0.84, bottom=0.14, left=0.06, right=0.97)
    fig.suptitle(
        "Visibility x Distance Benign Deviation Breakdown & Camera 0-15m Spike Attribution (120 Q95-444 Clean Frames)",
        fontsize=12.8, fontweight="bold", y=0.96,
    )

    for ax_idx, (mod, title) in enumerate([("m1", "(a) m1 Coll. (64L LiDAR): e_pool Mean / [P95]"), ("m2", "(b) m2 Coll. (Camera-LSS): e_pool Mean / [P95]")]):
        ax = axes[ax_idx]
        mean_mat = np.zeros((3, 4), dtype=np.float64)
        p95_mat = np.zeros((3, 4), dtype=np.float64)
        for vi, vlbl in enumerate(VIS_LABELS):
            for di, dlbl in enumerate(DIST_LABELS):
                st = mod_b[mod][vlbl][dlbl]["e_pool_3x3"]
                mean_mat[vi, di] = st["mean"] if st["mean"] is not None else 0.0
                p95_mat[vi, di] = st["p95"] if st["p95"] is not None else 0.0

        vmax = max(12.0, float(np.max(mean_mat)) * 1.15)
        im = ax.imshow(mean_mat, cmap="YlOrRd" if mod == "m2" else "Blues", vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(DIST_LABELS)
        ax.set_yticks(np.arange(3))
        ax.set_yticklabels(VIS_LABELS)
        ax.set_xlabel("Distance from Collaborator Center")
        ax.set_title(title)
        for vi in range(3):
            for di in range(4):
                txt_col = "white" if mean_mat[vi, di] > vmax * 0.55 else "#111111"
                ax.text(di, vi, f"{mean_mat[vi, di]:.2f}\n[{p95_mat[vi, di]:.1f}]", ha="center", va="center", color=txt_col, fontsize=8.8, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (c) 0-15m 细距均值与 MSE>25 尖峰占比
    ax = axes[2]
    sub_means = [mod_a["sub_distance_0_15m"][lbl]["mean"] or 0.0 for lbl in SPIKE_SUB_LABELS]
    sub_spk = [(mod_a["sub_distance_0_15m"][lbl]["spike_frac_gt25"] or 0.0) * 100.0 for lbl in SPIKE_SUB_LABELS]
    x = np.arange(len(SPIKE_SUB_LABELS))
    bars = ax.bar(x - 0.18, sub_means, width=0.35, color="#d95f02", alpha=0.85, label="Mean MSE_per_ch")
    ax.set_xticks(x)
    ax.set_xticklabels(SPIKE_SUB_LABELS)
    ax.set_ylabel("Mean MSE_per_ch", color="#d95f02")
    ax.set_ylim(0.0, max(sub_means) * 1.32)
    ax.set_title("(c) m2 0-15m Sub-Bins: Mean MSE & Extreme Spikes")
    for b_bar, val in zip(bars, sub_means):
        ax.text(b_bar.get_x() + b_bar.get_width() / 2, val + 0.4, f"{val:.1f}", ha="center", va="bottom", fontsize=8.5, color="#b04500", fontweight="bold")

    ax2 = ax.twinx()
    bars2 = ax2.bar(x + 0.18, sub_spk, width=0.35, color="#7570b3", alpha=0.82, label="Spike Ratio (MSE>25, %)")
    ax2.set_ylabel("Extreme Spike Grids (%)", color="#554d99")
    ax2.set_ylim(0.0, max(max(sub_spk), 1.0) * 1.35)
    for b_bar, val in zip(bars2, sub_spk):
        ax2.text(b_bar.get_x() + b_bar.get_width() / 2, val + 0.3, f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5, color="#554d99", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.45)

    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_4tier_comparison(comp: dict, out_png: str):
    """绘制图 2：4 组门限在相同 5% 总误报率下的小预算/局部攻击检出率及 AUROC 对比。"""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.8, "axes.titlesize": 11.2, "axes.titleweight": "bold"})
    tiers = ["tier1_unified", "tier2_sensor_only", "tier3_sensor_dist", "tier4_sensor_dist_vis"]
    t_labels = ["(1) Unified", "(2) Sensor-Only", "(3) Sensor+Dist", "(4) Sensor+Dist+Vis"]
    colors = ["#8c8c8c", "#377eb8", "#ff7f00", "#2ca02c"]

    groups = [
        ("m1\n0.5σ PGD", "m1", "scaled_pgd_0.5sigma"),
        ("m1\n1.0σ PGD", "m1", "scaled_pgd_1.0sigma"),
        ("m1\nLocal Rem.", "m1", "local_removal_1.0sigma"),
        ("m1\nLocal Ghost", "m1", "local_ghost_2.0sigma"),
        ("m2\n0.5σ PGD", "m2", "scaled_pgd_0.5sigma"),
        ("m2\n1.0σ PGD", "m2", "scaled_pgd_1.0sigma"),
        ("m2\nLocal Rem.", "m2", "local_removal_1.0sigma"),
        ("m2\nLocal Ghost", "m2", "local_ghost_2.0sigma"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(14.5, 8.8), dpi=300)
    fig.subplots_adjust(hspace=0.35, top=0.90, bottom=0.09, left=0.06, right=0.98)
    fig.suptitle(
        "4-Tier Threshold Ablation on Identical Eval-Set (Exact 5.0% Total Benign FPR Across m1+m2)",
        fontsize=13.0, fontweight="bold", y=0.96,
    )

    x = np.arange(len(groups))
    w = 0.20

    for ax_i, (metric_key, ylabel, title) in enumerate([
        ("tpr_at_5pct_total_fpr", "Detection Rate TPR (%) @ Total FPR=5%", "(a) Attack Detection Rate TPR (%) at Fixed 5.0% Total Benign FPR"),
        ("auroc", "AUROC", "(b) Threshold-Free Detection AUROC by Sensor & Attack Type"),
    ]):
        ax = axes[ax_i]
        for ti, (tk, tl, col) in enumerate(zip(tiers, t_labels, colors)):
            vals = [comp[tk]["attacks"][atk][metric_key][mod] * (100.0 if ax_i == 0 else 1.0) for _, mod, atk in groups]
            bars = ax.bar(x + (ti - 1.5) * w, vals, width=w, color=col, alpha=0.9, label=tl)
            for b_bar, val in zip(bars, vals):
                fmt = f"{val:.0f}" if ax_i == 0 else f"{val:.2f}"
                ax.text(b_bar.get_x() + b_bar.get_width() / 2, val + (1.5 if ax_i == 0 else 0.015), fmt, ha="center", va="bottom", fontsize=7.2, fontweight="bold")
        ax.axvline(3.5, color="#333333", linestyle="--", linewidth=1.1, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([g[0] for g in groups])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 114 if ax_i == 0 else 1.13)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle=":", alpha=0.55)
        ax.legend(loc="upper right", ncol=4, frameon=True, framealpha=0.92)

    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wrapper = HEALBaseWrapper(data_dir=TRAIN_READY_DIR, device=str(device))
    ds_m2 = wrapper.build_dataset(data_dir=TRAIN_READY_DIR, use_cav=2, visualize=True, collab_modality="m2")
    ds_m1 = wrapper.build_dataset(data_dir=TRAIN_READY_DIR, use_cav=2, visualize=True, collab_modality="m1")

    fit_list, fit_plan, eval_list, eval_plan = split_fit_eval_sequences(wrapper, ds_m2, TRAIN_READY_DIR)
    print(f"[INFO] 严格互斥划分完成: Fit {len(fit_plan)}序列/{len(fit_list)}帧, Eval {len(eval_plan)}序列/{len(eval_list)}帧", flush=True)

    grid_x, grid_y = build_bev_grid_xy_256(device=str(device))
    ab_acc = ModuleABAccumulator()

    # 1. 流式处理 Fit-Set (60 帧)：实时更新模块 A/B，仅缓存极轻量 CPU 标量图用于门限拟合
    fit_lite = []
    for k, (idx, seq, _) in enumerate(fit_list, 1):
        b = extract_clean_frame_bundle(wrapper, ds_m1, ds_m2, idx, seq, grid_x, grid_y, device)
        ab_acc.update(b)
        fit_lite.append({
            "cov_m2": b["cov_m2"].cpu(),
            "dist_cav": b["dist_cav"].cpu(),
            "sigma_alea_0": b["sigma_alea_0"].cpu(),
            "mse_m1": b["mse_m1"].cpu(),
            "mse_m2": b["mse_m2"].cpu(),
        })
        del b
        if k % 15 == 0:
            torch.cuda.empty_cache()
            print(f"  [Fit-Set 阶段 {k}/{len(fit_list)}] 耗时 {time.time() - t0:.1f}s", flush=True)

    # 2. 严格仅在 Fit-Set 拟合 4 组门限归一化参数 (用完立即释放 fit_lite)
    calib = fit_4tier_calibrators(fit_lite)
    fit_t1_scores = [
        score_frame_with_tier(fb, m, fb["mse_m1"] if m == "m1" else fb["mse_m2"], calib, "tier1_unified")
        for fb in fit_lite for m in ["m1", "m2"]
    ]
    calib["tier1_fit_p95"] = round(float(np.percentile(fit_t1_scores, 95)), 6)
    del fit_lite
    print("[STAGE 1] Fit-Set 4 组门限尺度拟合完成，开始 Eval-Set 流式评测...", flush=True)

    # 3. 流式处理 Eval-Set (60 帧)：实时完成模块 A/B 与模块 C (4 类攻击 + 4 组门限对比)
    c_eval = ModuleCEvaluator(wrapper, calib)
    for k, (idx, seq, _) in enumerate(eval_list, 1):
        b = extract_clean_frame_bundle(wrapper, ds_m1, ds_m2, idx, seq, grid_x, grid_y, device)
        ab_acc.update(b)
        c_eval.eval_frame(b)
        del b
        if k % 15 == 0 or k == len(eval_list):
            torch.cuda.empty_cache()
            print(f"  [Eval-Set 阶段 {k}/{len(eval_list)}] 耗时 {time.time() - t0:.1f}s", flush=True)

    mod_a, mod_b = ab_acc.finalize()
    comp_4tier, atk_meta = c_eval.finalize()

    report = {
        "meta": {
            "task": "Step4-1 Visibility x Distance Breakdown, 0-15m Spike Trace & 4-Tier Threshold Ablation",
            "source_dir": TRAIN_READY_DIR,
            "fit_sequences": len(fit_plan),
            "fit_frames": len(fit_list),
            "eval_sequences": len(eval_plan),
            "eval_frames": len(eval_list),
            "total_elapsed_sec": round(time.time() - t0, 2),
        },
        "split_plan": {"fit": fit_plan, "eval": eval_plan},
        "module_a_0_15m_spike_trace": mod_a,
        "module_b_visibility_x_distance": mod_b,
        "module_c_4tier_calibrators": calib,
        "module_c_attack_gen_stats": atk_meta,
        "module_c_4tier_comparison": comp_4tier,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    plot_vis_dist_and_spikes(mod_a, mod_b, OUT_FIG_HEAT)
    plot_4tier_comparison(comp_4tier, OUT_FIG_4TIER)

    print(f"\n[DONE] 报告保存至 {OUT_JSON}，总耗时 {report['meta']['total_elapsed_sec']}s", flush=True)
    print("  [0-15m 尖峰归因]", json.dumps(mod_a["spike_attribution_counts"], ensure_ascii=False), flush=True)
    for tk, tv in comp_4tier.items():
        line = f"  {tk:22s} | FPR(m1={tv['eval_fpr_at_5pct_thresh']['m1']*100:4.1f}%, m2={tv['eval_fpr_at_5pct_thresh']['m2']*100:4.1f}%, tot={tv['eval_fpr_at_5pct_thresh']['total']*100:4.1f}%)"
        for ak in ["scaled_pgd_0.5sigma", "scaled_pgd_1.0sigma", "local_removal_1.0sigma", "local_ghost_2.0sigma"]:
            r = tv["attacks"][ak]["tpr_at_5pct_total_fpr"]
            line += f" | {ak[:9]}: m1={r['m1']*100:4.1f}/m2={r['m2']*100:4.1f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
