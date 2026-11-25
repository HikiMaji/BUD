# -*- coding: utf-8 -*-
"""
开放异构车联网传感器物理退化模拟与目标层有效光束可见度先验模块
所有逻辑严格遵循 v2 优化定稿方案：
1. 支持 64线 / 32线 / 16线 激光雷达在车辆自身局部传感器坐标系下的物理俯仰环降采样；
2. 实现极坐标目标高度层有效光束可见度先验（区分近中距清晰空旷车道与真实遮挡盲区，无三维体素裁剪）；
3. 实现与外部待检特征彻底物理隔离的协作者纯几何距离-线束退化包络 u_geom(d_i, m_i)；
4. 提供测试集真值 3D 框按本车原始点云落点数的客观物理分层工具。
"""

import math
import numpy as np
import torch
import torch.nn.functional as F


# OPV2V 标准协议空间与激光雷达物理常量定义
LIDAR_RANGE = [-140.8, -40.0, -3.0, 140.8, 40.0, 1.0]
PROTO_H = 100   # 对应 y 轴 [-40.0, 40.0]，网格分辨率 0.8m
PROTO_W = 352   # 对应 x 轴 [-140.8, 140.8]，网格分辨率 0.8m
GRID_RES = 0.8
VERTICAL_FOV_RAD = math.radians(27.0)  # CARLA/OPV2V 64线雷达垂直视场角 [-25°, +2°] 共 27°
TARGET_HEIGHT_M = 1.55                 # 标准轿车立面有效高度 (m)
REF_BEAM_COUNT = 16.0                  # 对数可见度饱和参考有效光束数


def build_bev_grid_xy(device="cuda", dtype=torch.float32):
    """
    生成协议 BEV 空间 (H_p=100, W_p=352) 每个网格中心在本车坐标系下的物理坐标 (x, y)
    注意 OpenCOOD 散射约定：行维度 H 对应 y 轴，列维度 W 对应 x 轴。
    返回:
        grid_x: (100, 352)
        grid_y: (100, 352)
    """
    xs = (torch.arange(PROTO_W, device=device, dtype=dtype) + 0.5) * GRID_RES + LIDAR_RANGE[0]
    ys = (torch.arange(PROTO_H, device=device, dtype=dtype) + 0.5) * GRID_RES + LIDAR_RANGE[1]
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return grid_x, grid_y


def downsample_lidar_by_beams(lidar_np: np.ndarray, num_beams: int = 64, total_beams: int = 64) -> np.ndarray:
    """
    在协作者自身局部激光雷达坐标系下，按垂直俯仰角环（Elevation Ring）执行物理线束降采样。
    参数:
        lidar_np: (N, 4) numpy 数组 [x_local, y_local, z_local, intensity]
        num_beams: 目标物理线束数（支持 64, 32, 16, 8）
        total_beams: 原始激光雷达线束数（OPV2V 标准为 64）
    返回:
        采样后的局部坐标系点云 (M, 4)
    """
    if num_beams >= total_beams or lidar_np.shape[0] == 0:
        return lidar_np

    stride = max(1, total_beams // num_beams)
    xy_dist = np.sqrt(lidar_np[:, 0] ** 2 + lidar_np[:, 1] ** 2) + 1e-6
    elev_deg = np.degrees(np.arctan2(lidar_np[:, 2], xy_dist))

    # CARLA 64线雷达典型俯仰范围 [-25.0°, +2.0°]，自适应兼容边界微偏
    min_deg = min(-25.0, float(np.percentile(elev_deg, 0.5)))
    max_deg = max(2.0, float(np.percentile(elev_deg, 99.5)))
    ring_idx = np.floor((elev_deg - min_deg) / max(max_deg - min_deg, 1e-3) * (total_beams - 1) + 0.5).astype(np.int32)
    ring_idx = np.clip(ring_idx, 0, total_beams - 1)

    keep_mask = (ring_idx % stride) == 0
    sampled = lidar_np[keep_mask]
    # 保底防止极端空帧报错
    if sampled.shape[0] < 16:
        return lidar_np[: min(32, lidar_np.shape[0])]
    return sampled


def compute_analytical_beam_interception(dist_map: torch.Tensor, num_beams: float) -> torch.Tensor:
    """
    根据距离 d(x,y) 与传感器线束数 m_i，解析计算无遮挡条件下穿过 1.55m 标准车辆立面的有效激光环数。
    参数:
        dist_map: 任意形状的距离张量 (m)
        num_beams: 传感器物理线束数（如 64, 32, 16）
    返回:
        n_beam: 同形状的非负有效光束数张量
    """
    safe_dist = torch.clamp(dist_map, min=2.0)
    angular_span = torch.atan(TARGET_HEIGHT_M / safe_dist)
    n_beam = float(num_beams) * (angular_span / VERTICAL_FOV_RAD)
    return n_beam


def compute_collaborator_geom_prior(
    cav_xy_in_ego: torch.Tensor,
    num_beams: int,
    grid_x: torch.Tensor = None,
    grid_y: torch.Tensor = None,
) -> torch.Tensor:
    """
    计算与外部待检协议特征完全物理隔离的协作者纯几何距离-线束退化先验 u_geom(d_i(x,y), m_i) in [0.02, 1.0]。
    恶意节点无论如何篡改发送特征 Z_i，均无法改变该几何先验分母。
    参数:
        cav_xy_in_ego: (2,) 协作者传感器中心在本车坐标系下的 (x, y) 坐标
        num_beams: 协作者注册的物理线束数 (64 / 32 / 16)
    返回:
        u_geom: (1, H_p, W_p) 纯几何偶然退化先验张量
    """
    device = cav_xy_in_ego.device
    if grid_x is None or grid_y is None:
        grid_x, grid_y = build_bev_grid_xy(device=device)

    dx = grid_x - cav_xy_in_ego[0]
    dy = grid_y - cav_xy_in_ego[1]
    dist_map = torch.sqrt(dx * dx + dy * dy)

    n_beam = compute_analytical_beam_interception(dist_map, float(num_beams))
    vis_score = torch.clamp(torch.log1p(n_beam) / math.log1p(REF_BEAM_COUNT), 0.0, 1.0)
    # 超过传感器有效感知距离（如 >110m）附加平滑几何衰减
    range_decay = torch.sigmoid((dist_map - 105.0) / 12.0) * 0.25
    u_geom = torch.clamp(1.0 - vis_score + range_decay, min=0.03, max=1.0)
    return u_geom.unsqueeze(0)


def compute_target_layer_visibility_uncertainty(
    pts_in_ref: torch.Tensor,
    sensor_xy_in_ref: torch.Tensor,
    num_beams: int = 64,
    num_angle_bins: int = 360,
    occ_margin: float = 3.2,
    grid_x: torch.Tensor = None,
    grid_y: torch.Tensor = None,
) -> torch.Tensor:
    """
    极坐标目标高度层有效光束可见度驱动的偶然不确定性计算（严格修复 v1 空旷路面真值塌陷漏洞）：
    - 仅提取高于平坦路面的立体障碍物层 z in [-1.30m, +0.85m] 计算各极角扇区的首达遮挡距离包络 r_occ(theta)；
    - 近中距清晰无遮挡空旷车道（如正前方 25m 空地）因 r <= r_occ + margin，享有完整的理论穿过光束数，输出低不确定性；
    - 位于前车/建筑物背后的真实阴影区因 r > r_occ + margin，有效光束平滑衰减至 0，输出高不确定性 (~1.0)。
    参数:
        pts_in_ref: (N, 4) 参考系（本车系）下的点云张量
        sensor_xy_in_ref: (2,) 该传感器在参考系下的中心坐标 (x, y)
        num_beams: 该传感器的线束数
        num_angle_bins: 水平 360° 极角扇区分桶数
        occ_margin: 障碍物自身厚度容差 (m)，保证前车车体自身所在网格不被判为被自己遮挡
    返回:
        sigma_alea_sq: (1, H_p, W_p) 范围在 [0.02, 1.0] 的物理偶然不确定性图
    """
    device = pts_in_ref.device
    if grid_x is None or grid_y is None:
        grid_x, grid_y = build_bev_grid_xy(device=device)

    # 计算协议网格相对于该传感器中心的极坐标 (r_grid, theta_grid)
    gx = grid_x - sensor_xy_in_ref[0]
    gy = grid_y - sensor_xy_in_ref[1]
    r_grid = torch.sqrt(gx * gx + gy * gy)
    theta_grid = torch.atan2(gy, gx)  # [-pi, pi]

    # 初始化各方位扇区的首达遮挡距离为最大视距 160m（无遮挡畅通）
    r_occ_bins = torch.full((num_angle_bins,), 160.0, device=device, dtype=torch.float32)

    if pts_in_ref.shape[0] > 0:
        px = pts_in_ref[:, 0] - sensor_xy_in_ref[0]
        py = pts_in_ref[:, 1] - sensor_xy_in_ref[1]
        pz = pts_in_ref[:, 2]
        pr = torch.sqrt(px * px + py * py)

        # 筛选立体障碍物高度层（排除 z < -1.30m 的平坦地面回波，且排除 r < 2.4m 的自身车体反射）
        obs_mask = (pz > -1.30) & (pz < 0.85) & (pr > 2.4)
        if obs_mask.any():
            obs_r = pr[obs_mask]
            obs_theta = torch.atan2(py[obs_mask], px[obs_mask])
            bin_idx = torch.clamp(
                ((obs_theta + math.pi) / (2.0 * math.pi) * num_angle_bins).long(),
                0,
                num_angle_bins - 1,
            )
            r_occ_bins.scatter_reduce_(0, bin_idx, obs_r, reduce="amin", include_self=True)

            # 对相邻扇区做 3-bin 最大包络平滑，防止稀疏采样点缝隙误穿或单噪点误挡
            r_pad = torch.cat([r_occ_bins[-1:], r_occ_bins, r_occ_bins[:1]], dim=0)
            r_occ_bins = F.max_pool1d(r_pad.view(1, 1, -1), kernel_size=3, stride=1).view(-1)

    grid_bin_idx = torch.clamp(
        ((theta_grid + math.pi) / (2.0 * math.pi) * num_angle_bins).long(),
        0,
        num_angle_bins - 1,
    )
    r_occ_map = r_occ_bins[grid_bin_idx]

    # 计算视线通畅系数：在障碍物前及障碍物自身车体厚度内为 1.0，进入阴影后按指数快速衰减至 0
    excess_dist = torch.clamp(r_grid - (r_occ_map + occ_margin), min=0.0)
    los_factor = torch.exp(-excess_dist / 3.0)

    n_beam_vis = compute_analytical_beam_interception(r_grid, float(num_beams)) * los_factor
    vis_score = torch.clamp(torch.log1p(n_beam_vis) / math.log1p(REF_BEAM_COUNT), 0.0, 1.0)
    sigma_alea_sq = torch.clamp(1.0 - vis_score, min=0.02, max=1.0)
    return sigma_alea_sq.unsqueeze(0)


def count_points_in_3d_boxes(points_xyz: np.ndarray, boxes_corners: np.ndarray) -> np.ndarray:
    """
    客观统计真值 3D 框（8 角点格式 (M, 8, 3)）内落入的原始激光雷达点数，
    用于与被测模型完全无关的测试集客观物理分层评估（Clear / Sparse / Blind-Spot Occluded）。
    """
    if boxes_corners is None or len(boxes_corners) == 0:
        return np.zeros((0,), dtype=np.int32)
    if points_xyz is None or len(points_xyz) == 0:
        return np.zeros((len(boxes_corners),), dtype=np.int32)

    counts = np.zeros((len(boxes_corners),), dtype=np.int32)
    pts_xy = points_xyz[:, :2]
    pts_z = points_xyz[:, 2]

    for i, corners in enumerate(boxes_corners):
        min_z = float(np.min(corners[:, 2])) - 0.15
        max_z = float(np.max(corners[:, 2])) + 0.15
        # 先用快速 AABB 包围盒粗筛
        min_xy = np.min(corners[:, :2], axis=0)
        max_xy = np.max(corners[:, :2], axis=0)
        cand_mask = (
            (pts_xy[:, 0] >= min_xy[0]) & (pts_xy[:, 0] <= max_xy[0]) &
            (pts_xy[:, 1] >= min_xy[1]) & (pts_xy[:, 1] <= max_xy[1]) &
            (pts_z >= min_z) & (pts_z <= max_z)
        )
        if not np.any(cand_mask):
            continue
        cand_xy = pts_xy[cand_mask]
        # 精确旋转矩形内点投影判定（底面 4 个顶点 0,1,2,3）
        p0 = corners[0, :2]
        u = corners[1, :2] - p0
        v = corners[3, :2] - p0
        uu = float(np.dot(u, u)) + 1e-6
        vv = float(np.dot(v, v)) + 1e-6
        rel = cand_xy - p0[None, :]
        proj_u = rel @ u
        proj_v = rel @ v
        in_rect = (proj_u >= 0.0) & (proj_u <= uu) & (proj_v >= 0.0) & (proj_v <= vv)
        counts[i] = int(np.sum(in_rect))
    return counts
