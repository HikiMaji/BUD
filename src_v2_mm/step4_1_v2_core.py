# -*- coding: utf-8 -*-
"""
步骤 ④-1 修正版核心公共模块（严格隔离 Fit/Eval、多尺度余弦 + 清晰语义偏差提取与 4 组门限打分内核）。

核心规范：
1. 排除原生 3 相机序列 2021_09_09_13_20_58，全部 19 条完整 4 相机序列（3,536 帧）按父序列严格互斥划分 Fit(11条) 与 Eval(8条)；
2. 距离分 5 档（[0,6), [6,15), [15,30), [30,45), [45,∞) m），将摄像头 0~6m LSS 奇异高幅带独立成档，不豁免；
3. 本车物理可见性分 3 档（Clear / Sparse / Occluded），Tier ④ 采用「清晰区收紧、遮挡区放宽」非对称加权调制；
4. 提供干净与攻击态共用的 4 通道偏差图 (d_cos_0, d_cos_1, d_cos_2, d_sem) 提取及稳健上尾标准化计算接口。
"""

import json
import math
import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJ_ROOT = "/root/autodl-tmp/proj_dir2_hetero_trust"
TRAIN_FULL19_DIR = os.path.join(PROJ_ROOT, "tmp_png_ab_test", "train_full19_444")

# 严格互斥序列划分：11 条 Fit 序列（含全部 7 条历史探索序列，2/3/4/5 车齐全，共 2,312 帧）
FIT_SEQ_NAMES = [
    "2021_08_21_15_41_04",  # 2车, 162帧
    "2021_08_16_22_26_54",  # 3车, 419帧
    "2021_08_18_22_16_12",  # 3车, 267帧
    "2021_08_20_20_39_00",  # 3车, 110帧
    "2021_08_21_22_21_37",  # 3车, 134帧
    "2021_08_22_10_46_58",  # 3车, 127帧
    "2021_08_23_11_22_46",  # 3车, 156帧
    "2021_08_18_19_11_02",  # 4车, 122帧
    "2021_08_21_09_09_41",  # 4车, 174帧
    "2021_08_22_07_24_12",  # 4车, 190帧
    "2021_09_09_23_21_21",  # 5车, 451帧
]

# 8 条全新未探索完整 4 相机序列作为 Eval 盲测集（2/3/4/5/6 车全覆盖，含 3 条 part4 新序列，共 1,224 帧）
EVAL_SEQ_NAMES = [
    "2021_08_18_09_02_56",  # 2车, 74帧
    "2021_08_22_10_10_40",  # 3车, 102帧
    "2021_08_22_22_30_58",  # 3车, 96帧
    "2021_08_23_23_08_17",  # 3车, 103帧
    "2021_08_24_21_29_28",  # 3车, 131帧 (part4)
    "2021_08_22_06_43_37",  # 4车, 181帧
    "2021_09_10_12_07_11",  # 5车, 383帧 (part4)
    "2021_09_09_22_21_11",  # 6车, 154帧 (part4)
]

PROTO_H = 256
PROTO_W = 256
GRID_RES = 0.8
MIN_XY = -102.4
VERTICAL_FOV_RAD = math.radians(27.0)
TARGET_HEIGHT_M = 1.55
REF_BEAM_COUNT = 16.0

DIST_BINS_5 = [(0.0, 6.0), (6.0, 15.0), (15.0, 30.0), (30.0, 45.0), (45.0, 999.0)]
DIST_LABELS_5 = ["0-6m", "6-15m", "15-30m", "30-45m", "45m+"]
VIS_LABELS_3 = ["Clear", "Sparse", "Occluded"]
CHANNEL_NAMES = ["cos0", "cos1", "cos2", "sem"]
TIERS_4 = ["tier1_unified", "tier2_sensor_only", "tier3_sensor_dist", "tier4_sensor_dist_vis"]


def build_bev_grid_xy_256(device="cuda", dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成 HEAL [-102.4, 102.4]^2 (256x256, 0.8m) 网格中心物理坐标。"""
    xs = (torch.arange(PROTO_W, device=device, dtype=dtype) + 0.5) * GRID_RES + MIN_XY
    ys = (torch.arange(PROTO_H, device=device, dtype=dtype) + 0.5) * GRID_RES + MIN_XY
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return gx, gy


def _alloc_balanced_indices(seq_names: List[str], seq_to_idxs: dict, data_dir: str, target_total: int, min_per_seq: int = 6):
    """按序列帧数比例均衡抽取样本索引，并确保每条序列至少采样 min_per_seq 帧。"""
    total_f = sum(len(seq_to_idxs[s]) for s in seq_names)
    raw = {s: max(float(min_per_seq), float(target_total) * len(seq_to_idxs[s]) / float(total_f)) for s in seq_names}
    cnt = {s: int(math.floor(raw[s])) for s in seq_names}
    diff = int(target_total) - sum(cnt.values())
    if diff > 0:
        for s in sorted(cnt.keys(), key=lambda k: raw[k] - cnt[k], reverse=True)[:diff]:
            cnt[s] += 1
    elif diff < 0:
        for s in sorted([k for k in cnt if cnt[k] > min_per_seq], key=lambda k: cnt[k], reverse=True)[:(-diff)]:
            cnt[s] -= 1

    flat_items = []
    plan = OrderedDict()
    for s in seq_names:
        idxs = seq_to_idxs[s]
        n_p = cnt[s]
        pos = np.linspace(0, len(idxs) - 1, n_p, dtype=int)
        picked = [int(idxs[p]) for p in pos]
        cavs = len([d for d in os.listdir(os.path.join(data_dir, s)) if os.path.isdir(os.path.join(data_dir, s, d))])
        for idx_v in picked:
            flat_items.append((idx_v, s, cavs))
        plan[s] = {"raw_cavs": cavs, "total_frames": len(idxs), "sampled": n_p, "indices": picked}
    return flat_items, plan


def build_and_save_split_plan(
    wrapper,
    ds_m2,
    out_json_path: str,
    train_dir: str = TRAIN_FULL19_DIR,
    n_fit: int = 100,
    n_fit_dev: int = 36,
    n_eval: int = 80,
) -> dict:
    """构造并持久化 19 序列零泄露 Fit / Fit-Dev / Fit-Holdout / Eval 划分方案。"""
    sample_seqs = wrapper.get_sample_sequences(ds_m2)
    seq_to_idxs = OrderedDict()
    for i, s in enumerate(sample_seqs):
        seq_to_idxs.setdefault(s, []).append(i)

    assert set(seq_to_idxs.keys()) == set(FIT_SEQ_NAMES + EVAL_SEQ_NAMES), (
        f"目录序列集合不符：额外={set(seq_to_idxs.keys()) - set(FIT_SEQ_NAMES + EVAL_SEQ_NAMES)}, "
        f"缺失={set(FIT_SEQ_NAMES + EVAL_SEQ_NAMES) - set(seq_to_idxs.keys())}"
    )
    assert set(FIT_SEQ_NAMES).isdisjoint(set(EVAL_SEQ_NAMES)), "Fit 与 Eval 序列存在交集！"

    fit_items, fit_plan = _alloc_balanced_indices(FIT_SEQ_NAMES, seq_to_idxs, train_dir, n_fit, min_per_seq=6)
    eval_items, eval_plan = _alloc_balanced_indices(EVAL_SEQ_NAMES, seq_to_idxs, train_dir, n_eval, min_per_seq=6)

    # 在 Fit-100 内部再均匀选出 Fit-Dev (36 帧) 与 Fit-Holdout (64 帧)
    dev_raw = {s: max(2.0, float(n_fit_dev) * fit_plan[s]["sampled"] / float(n_fit)) for s in FIT_SEQ_NAMES}
    dev_cnt = {s: int(math.floor(dev_raw[s])) for s in FIT_SEQ_NAMES}
    d_diff = int(n_fit_dev) - sum(dev_cnt.values())
    if d_diff > 0:
        for s in sorted(dev_cnt.keys(), key=lambda k: dev_raw[k] - dev_cnt[k], reverse=True)[:d_diff]:
            dev_cnt[s] += 1
    elif d_diff < 0:
        for s in sorted([k for k in dev_cnt if dev_cnt[k] > 2], key=lambda k: dev_cnt[k], reverse=True)[:(-d_diff)]:
            dev_cnt[s] -= 1

    fit_dev_items = []
    fit_holdout_items = []
    for s in FIT_SEQ_NAMES:
        picked = fit_plan[s]["indices"]
        cavs = fit_plan[s]["raw_cavs"]
        k_dev = dev_cnt[s]
        dev_pos = set(np.linspace(0, len(picked) - 1, k_dev, dtype=int).tolist())
        fit_plan[s]["fit_dev_indices"] = [picked[p] for p in sorted(dev_pos)]
        fit_plan[s]["fit_holdout_indices"] = [picked[p] for p in range(len(picked)) if p not in dev_pos]
        for idx_v in fit_plan[s]["fit_dev_indices"]:
            fit_dev_items.append((idx_v, s, cavs))
        for idx_v in fit_plan[s]["fit_holdout_indices"]:
            fit_holdout_items.append((idx_v, s, cavs))

    split_plan = OrderedDict({
        "dataset_dir": train_dir,
        "excluded_3cam_seq": "2021_09_09_13_20_58",
        "total_sequences": 19,
        "total_frames": len(sample_seqs),
        "fit_sequences": fit_plan,
        "eval_sequences": eval_plan,
        "fit_indices": [x[0] for x in fit_items],
        "fit_dev_indices": [x[0] for x in fit_dev_items],
        "fit_holdout_indices": [x[0] for x in fit_holdout_items],
        "eval_indices": [x[0] for x in eval_items],
        "counts": {
            "fit_total": len(fit_items),
            "fit_dev": len(fit_dev_items),
            "fit_holdout": len(fit_holdout_items),
            "eval_total": len(eval_items),
        },
    })
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(split_plan, f, indent=2, ensure_ascii=False)
    return split_plan


def load_ego_lidar_np(dataset, idx: int) -> np.ndarray:
    """仅读取本车 ego 唯一原始 .pcd 点云并过滤自车车体与范围外点。"""
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
    raise RuntimeError(f"样本 {idx} 未找到 ego 点云！")


def compute_ego_expected_pts_and_los_256(
    ego_pts_np: np.ndarray,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    num_beams: int = 64,
    num_angle_bins: int = 360,
    occ_margin: float = 2.0,
    target_width_m: float = 2.0,
    horiz_res_deg: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    基于本车 64 线激光雷达物理射线追踪，同时返回：
    - sigma_alea_sq (1,1,256,256) ∈ [0.02, 1.0]: 兼容旧接口的连续偶然不确定性；
    - los_factor    (1,1,256,256) ∈ [0, 1]     : 视线无遮挡因子（<0.5 表示前方存在近距障碍物遮挡）；
    - expected_pts  (1,1,256,256) >= 0         : 穿过标准车辆(高1.5m×宽2.0m)的本车物理预期激光点数 (∝ los / r^2)。
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
            bin_idx = torch.clamp(((obs_theta + math.pi) / (2.0 * math.pi) * num_angle_bins).long(), 0, num_angle_bins - 1)
            r_occ_bins.scatter_reduce_(0, bin_idx, obs_r, reduce="amin", include_self=True)
            r_pad = torch.cat([r_occ_bins[-1:], r_occ_bins, r_occ_bins[:1]], dim=0)
            r_occ_bins = F.max_pool1d(r_pad.view(1, 1, -1), kernel_size=3, stride=1).view(-1)

    grid_bin_idx = torch.clamp(((theta_grid + math.pi) / (2.0 * math.pi) * num_angle_bins).long(), 0, num_angle_bins - 1)
    r_occ_map = r_occ_bins[grid_bin_idx]
    excess_dist = torch.clamp(r_grid - (r_occ_map + occ_margin), min=0.0)
    los_factor = torch.exp(-excess_dist / 3.0)

    safe_dist = torch.clamp(r_grid, min=2.0)
    n_beam_vis = float(num_beams) * (torch.atan(TARGET_HEIGHT_M / safe_dist) / VERTICAL_FOV_RAD) * los_factor
    horiz_res_rad = math.radians(float(horiz_res_deg))
    n_horiz = (float(target_width_m) / safe_dist) / max(horiz_res_rad, 1e-6)
    expected_pts = n_beam_vis * n_horiz

    vis_score = torch.clamp(torch.log1p(n_beam_vis) / math.log1p(REF_BEAM_COUNT), 0.0, 1.0)
    sigma_alea_sq = torch.clamp(1.0 - vis_score, min=0.02, max=1.0)
    return (
        sigma_alea_sq.view(1, 1, PROTO_H, PROTO_W),
        los_factor.view(1, 1, PROTO_H, PROTO_W),
        expected_pts.view(1, 1, PROTO_H, PROTO_W),
    )


def compute_ego_beam_visibility_256(
    ego_pts_np: np.ndarray,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    num_beams: int = 64,
    num_angle_bins: int = 360,
    occ_margin: float = 2.0,
) -> torch.Tensor:
    """兼容旧接口：仅返回基于本车物理遮挡射线追踪的 sigma_{alea,0}^2 ∈ [0.02, 1.0]。"""
    sigma_alea_sq, _, _ = compute_ego_expected_pts_and_los_256(
        ego_pts_np, grid_x, grid_y, num_beams=num_beams, num_angle_bins=num_angle_bins, occ_margin=occ_margin
    )
    return sigma_alea_sq


def assign_dist5_vis3_indices_v2(
    dist_cav: torch.Tensor,
    los_factor: torch.Tensor,
    expected_pts: torch.Tensor,
    clear_pts_tau: float,
    los_occ_tau: float = 0.50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    改进版物理可见性分档（原因 B 根治）：
    - Occluded (2): 前方存在遮挡物（los_factor < los_occ_tau）；
    - Clear    (0): 前方无遮挡（los_factor >= los_occ_tau）且 本车预期激光点数 expected_pts >= clear_pts_tau；
    - Sparse   (1): 前方无遮挡但本车预期激光点数不足（expected_pts < clear_pts_tau）。
    """
    d_idx = torch.full_like(dist_cav, 4, dtype=torch.long)
    d_idx[dist_cav < 45.0] = 3
    d_idx[dist_cav < 30.0] = 2
    d_idx[dist_cav < 15.0] = 1
    d_idx[dist_cav < 6.0] = 0

    v_idx = torch.ones_like(expected_pts, dtype=torch.long)  # 默认 Sparse(1)
    unoccluded = los_factor >= float(los_occ_tau)
    v_idx[unoccluded & (expected_pts >= float(clear_pts_tau))] = 0  # Clear
    v_idx[~unoccluded] = 2                                          # Occluded
    return d_idx, v_idx


def assign_dist5_vis3_indices(dist_cav: torch.Tensor, sigma_alea: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 5 档距离索引 d_idx ∈ {0..4} 与 3 档可见性索引 v_idx ∈ {0:Clear, 1:Sparse, 2:Occluded}。"""
    d_idx = torch.full_like(dist_cav, 4, dtype=torch.long)
    d_idx[dist_cav < 45.0] = 3
    d_idx[dist_cav < 30.0] = 2
    d_idx[dist_cav < 15.0] = 1
    d_idx[dist_cav < 6.0] = 0

    v_idx = torch.zeros_like(sigma_alea, dtype=torch.long)
    v_idx[(sigma_alea >= 0.33) & (sigma_alea < 0.67)] = 1
    v_idx[sigma_alea >= 0.67] = 2
    return d_idx, v_idx


def _masked_avg_pool3x3(val_map: torch.Tensor, mask_map: torch.Tensor) -> torch.Tensor:
    """对 (1, 1, H, W) 偏差图仅在有效掩膜内执行 3x3 局部平均池化，杜绝边界外零值稀释。"""
    m_f = mask_map.to(dtype=val_map.dtype)
    num = F.avg_pool2d(val_map * m_f, kernel_size=3, stride=1, padding=1) * 9.0
    den = torch.clamp(F.avg_pool2d(m_f, kernel_size=3, stride=1, padding=1) * 9.0, min=1e-4)
    return (num / den) * m_f


@torch.no_grad()
def compute_dev_maps_from_protocol(
    wrapper,
    p_dict: dict,
    feat_in: torch.Tensor,
    p_ego_cls: torch.Tensor,
    cov_ego: torch.Tensor,
) -> torch.Tensor:
    """
    给定传输层特征 feat_in (N, 64, 256, 256)，在统一 ego 坐标系下提取 4 通道偏差图 (4, 256, 256)：
    [0] cos0: pyramid_L0 3x3局部池化余弦偏差
    [1] cos1: pyramid_L1 余弦偏差上采样至 256x256
    [2] cos2: pyramid_L2 余弦偏差上采样至 256x256
    [3] sem : 本车独享检测概率与协同融合检测概率之差 |P_ego - P_fused|
    """
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

    rlen = p_dict["record_len"]
    amat = p_dict["affine_matrix"]
    mods = p_dict["agent_modality_list"]
    tmat01 = amat[0, 0, 1:2]
    ac = wrapper.model.pyramid_backbone.align_corners

    out = wrapper.forward_from_protocol(feat_in, rlen, amat, mods)
    cov_f = cov_ego.float()

    # L0/L1/L2 多尺度特征余弦偏差
    cos_maps = []
    for lvl in range(3):
        f_lvl = out["pyramid_feats"][lvl]
        z_ego = f_lvl[0:1]
        hl, wl = z_ego.shape[-2:]
        z_cav = warp_affine_simple(f_lvl[1:2], tmat01, (hl, wl), align_corners=ac)
        c_raw = (1.0 - F.cosine_similarity(z_ego, z_cav, dim=1, eps=1e-6)).unsqueeze(1)
        if lvl == 0:
            c_256 = _masked_avg_pool3x3(c_raw, cov_ego)
        else:
            c_256 = F.interpolate(c_raw, size=(PROTO_H, PROTO_W), mode="bilinear", align_corners=False) * cov_f
        cos_maps.append(c_256)

    p_fused_cls = torch.sigmoid(out["cls_preds"]).max(dim=1, keepdim=True)[0]
    d_sem = torch.abs(p_ego_cls - p_fused_cls) * cov_f

    dev_4ch = torch.cat(cos_maps + [d_sem], dim=1)[0]  # (4, 256, 256)
    return dev_4ch


@torch.no_grad()
def extract_frame_context_and_clean(wrapper, ds_m1, ds_m2, idx: int, gx: torch.Tensor, gy: torch.Tensor) -> dict:
    """加载单帧 m1/m2 成对样本并提取几何/可见性上下文与两模态干净 4 通道偏差图。"""
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
    from opencood.tools import train_utils

    b1 = train_utils.to_device(ds_m1.collate_batch_test([ds_m1[idx]]), wrapper.device_obj)
    b2 = train_utils.to_device(ds_m2.collate_batch_test([ds_m2[idx]]), wrapper.device_obj)

    ego_pts = load_ego_lidar_np(ds_m1, idx)
    sigma_alea = compute_ego_beam_visibility_256(ego_pts, gx, gy, num_beams=64)  # (1,1,256,256)
    occ_mask_ego = sigma_alea >= 0.67

    ptm = b1["ego"]["pairwise_t_matrix"][0, 1, 0].detach().cpu().numpy()
    cx, cy = float(ptm[0, 3]), float(ptm[1, 3])
    dist_cav = torch.sqrt((gx - cx) ** 2 + (gy - cy) ** 2).view(1, 1, PROTO_H, PROTO_W)

    p1 = wrapper.extract_protocol_features(b1["ego"])
    p2 = wrapper.extract_protocol_features(b2["ego"])
    ac = wrapper.model.pyramid_backbone.align_corners

    # 使用统一的 m2 局部有效区裁剪边界 [66:190, 66:190]，确保 m1 与 m2 评估相同空间域
    cov_local_m1 = p2["cam_valid_mask"][1:2]
    cov_ego = warp_affine_simple(cov_local_m1, p2["affine_matrix"][0, 0, 1:2], (PROTO_H, PROTO_W), align_corners=ac) > 0.5

    # 单车 ego 检测概率（同一帧下本车均为 m1，m1/m2 共享，只在需要时保证与各自 feat[0:1] 一致）
    o_ego_single = wrapper.forward_single_agent(p1["heter_feature_2d"][0:1])
    p_ego_cls = torch.sigmoid(o_ego_single["cls_preds"]).max(dim=1, keepdim=True)[0]

    dev_clean_m1 = compute_dev_maps_from_protocol(wrapper, p1, p1["heter_feature_2d"], p_ego_cls, cov_ego)
    dev_clean_m2 = compute_dev_maps_from_protocol(wrapper, p2, p2["heter_feature_2d"], p_ego_cls, cov_ego)

    d_idx, v_idx = assign_dist5_vis3_indices(dist_cav, sigma_alea)

    return {
        "sample_idx": int(idx),
        "p1": p1,
        "p2": p2,
        "p_ego_cls": p_ego_cls,
        "cov_ego": cov_ego,
        "cov_local_m1": cov_local_m1,
        "occ_mask_ego": occ_mask_ego,
        "dist_cav": dist_cav,
        "sigma_alea": sigma_alea,
        "d_idx": d_idx[0, 0],  # (256, 256)
        "v_idx": v_idx[0, 0],  # (256, 256)
        "dev_clean": {"m1": dev_clean_m1, "m2": dev_clean_m2},
    }


def _get_tier_channel_z(
    dev_4ch: torch.Tensor,
    mod: str,
    d_idx: torch.Tensor,
    v_idx: torch.Tensor,
    stats: dict,
    tier_name: str,
    q_key: str = "p95",
    s_floor: float = 0.05,
) -> torch.Tensor:
    """按指定门限层级将 4 通道偏差图标准化为非负上尾 z 分数 (4, H, W)。"""
    device = dev_4ch.device
    dtype = dev_4ch.dtype
    z_list = []

    for c in CHANNEL_NAMES:
        st_c = stats[c]
        if tier_name == "tier1_unified":
            mu = float(st_c["unified"]["median"])
            sc = max(float(st_c["unified"][q_key]) - mu, s_floor)
            mu_map, sc_map = mu, sc
        elif tier_name == "tier2_sensor_only":
            mu = float(st_c["sensor"][mod]["median"])
            sc = max(float(st_c["sensor"][mod][q_key]) - mu, s_floor)
            mu_map, sc_map = mu, sc
        elif tier_name == "tier3_sensor_dist":
            med_vec = torch.tensor([float(st_c["sensor_dist"][mod][dl]["median"]) for dl in DIST_LABELS_5], device=device, dtype=dtype)
            pq_vec = torch.tensor([float(st_c["sensor_dist"][mod][dl][q_key]) for dl in DIST_LABELS_5], device=device, dtype=dtype)
            sc_vec = torch.clamp(pq_vec - med_vec, min=s_floor)
            mu_map = med_vec[d_idx]
            sc_map = sc_vec[d_idx]
        elif tier_name == "tier4_sensor_dist_vis":
            med_mat = torch.tensor(
                [[float(st_c["sensor_dist_vis"][mod][dl][vl]["median"]) for vl in VIS_LABELS_3] for dl in DIST_LABELS_5],
                device=device,
                dtype=dtype,
            )
            pq_mat = torch.tensor(
                [[float(st_c["sensor_dist_vis"][mod][dl][vl][q_key]) for vl in VIS_LABELS_3] for dl in DIST_LABELS_5],
                device=device,
                dtype=dtype,
            )
            sc_mat = torch.clamp(pq_mat - med_mat, min=s_floor)
            mu_map = med_mat[d_idx, v_idx]
            sc_map = sc_mat[d_idx, v_idx]
        else:
            raise ValueError(f"未知 tier_name: {tier_name}")

        ch_i = CHANNEL_NAMES.index(c)
        z_c = torch.clamp((dev_4ch[ch_i] - mu_map) / sc_map, min=0.0)
        z_list.append(z_c)

    return torch.stack(z_list, dim=0)


def compute_raw_frame_score_from_dev(
    dev_4ch: torch.Tensor,
    cov_ego_2d: torch.Tensor,
    d_idx: torch.Tensor,
    v_idx: torch.Tensor,
    mod: str,
    stats: dict,
    tier_name: str,
    cfg: dict,
    method_kind: str,
) -> float:
    """
    从 4 通道偏差图 (4, H, W) 按指定方法 (main / ablation) 与层级 (tier1~tier4) 计算单帧原始异常分数 F_raw。
    """
    q_key = f"p{int(cfg['q_scale'])}"
    z = _get_tier_channel_z(dev_4ch, mod, d_idx, v_idx, stats, tier_name, q_key=q_key, s_floor=0.05)

    w0, w1, w2 = cfg["cos_weights"]
    cos_comb = w0 * z[0] + w1 * z[1] + w2 * z[2]

    if tier_name == "tier4_sensor_dist_vis":
        g_cos = torch.tensor(cfg["gamma_cos"], device=dev_4ch.device, dtype=dev_4ch.dtype)[v_idx]
        cos_term = g_cos * cos_comb
    else:
        cos_term = cos_comb

    if method_kind == "ablation_ms_cos":
        score_map = cos_term
    elif method_kind == "main_ms_cos_clear_sem":
        w_sem = float(cfg["w_sem"])
        if tier_name == "tier4_sensor_dist_vis":
            g_sem = torch.tensor(cfg["gamma_sem"], device=dev_4ch.device, dtype=dev_4ch.dtype)[v_idx]
            sem_term = g_sem * z[3]
        else:
            sem_term = z[3]
        score_map = (1.0 - w_sem) * cos_term + w_sem * sem_term
    else:
        raise ValueError(f"未知 method_kind: {method_kind}")

    vals = score_map[cov_ego_2d]
    if vals.numel() == 0:
        return 0.0
    k = min(int(cfg["top_k"]), vals.numel())
    return float(torch.topk(vals, k).values.mean().item())


def apply_frame_calibration(raw_score: float, mod: str, tier_name: str, frame_norm_mode: str, tier_calib: dict) -> float:
    """按 Fit 阶段冻结的跨模态帧分参数将单帧原始得分归一化（tier1 保持不除分传感器尺度）。"""
    if tier_name == "tier1_unified":
        return float(raw_score)
    st = tier_calib[tier_name][mod]
    med = float(st["median"])
    p95 = float(st["p95"])
    if frame_norm_mode == "div_p95":
        return float(raw_score) / max(p95, 1e-4)
    if frame_norm_mode == "med_p95":
        return (float(raw_score) - med) / max(p95 - med, 1e-4)
    raise ValueError(f"未知 frame_norm_mode: {frame_norm_mode}")


def compute_auroc(neg_scores: np.ndarray, pos_scores: np.ndarray) -> float:
    """精确 Mann-Whitney U 秩和法计算 AUROC（正确处理并列值）。"""
    neg = np.asarray(neg_scores, dtype=np.float64)
    pos = np.asarray(pos_scores, dtype=np.float64)
    n0, n1 = neg.size, pos.size
    if n0 == 0 or n1 == 0:
        return 0.5
    all_s = np.concatenate([neg, pos], axis=0)
    order = np.argsort(all_s)
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    while i < all_s.size:
        j = i
        while j + 1 < all_s.size and all_s[order[j + 1]] == all_s[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    u1 = ranks[n0:].sum() - n1 * (n1 + 1) / 2.0
    return float(u1 / (n0 * n1))


def find_exact_fpr_threshold(ben_scores: np.ndarray, target_fpr: float = 0.05) -> float:
    """返回使良性帧中大于阈值比例精确等于（或最接近）target_fpr 的门限。"""
    s = np.sort(np.asarray(ben_scores, dtype=np.float64))
    n = s.size
    k_fp = int(round(n * float(target_fpr)))
    if k_fp <= 0:
        return float(s[-1]) + 1e-6
    if k_fp >= n:
        return float(s[0]) - 1e-6
    return 0.5 * (float(s[n - k_fp - 1]) + float(s[n - k_fp]))
