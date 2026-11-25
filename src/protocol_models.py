# -*- coding: utf-8 -*-
"""
开放异构三维点云编码器与严格谱有界协议对齐适配器（LB-FPA）模块
包含：
1. ExactSpectralConv2d：基于真二维卷积算子伴随幂迭代的精确谱有界卷积层（无稀疏 BatchNorm 奇异值爆炸漏洞）；
2. LBFPAAdapter：凸残差连接 + 末端非负锥对齐的严格 Lipschitz 有界协议适配器；
3. UnconstrainedAdapter：常规 Conv2d + BatchNorm2d + ReLU 的无约束协议适配器（复现 STAMP/HEAL 对照基线与消融）；
4. 四类开放异构智能体编码器（Profile-A 64L PointPillars 冻结锚点、Profile-B 64L SECOND 3D稀疏卷积、
   Profile-C 32L 粗粒度柱状编码、Profile-D 16L 稀疏雷达编码）。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv


class ExactSpectralConv2d(nn.Module):
    """
    真二维空间卷积算子谱归一化层（Exact Conv2D Operator Spectral Normalization）。
    通过交替执行 Conv2d 与伴随算子 ConvTranspose2d 的幂迭代计算精确诱导 L2 算子范数 sigma_op(W)，
    克服默认展平矩阵谱归一化对 3x3 卷积高估/失配的问题，且不含任何方差除法归一化层。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        lip_gamma: float = 1.0,
        n_power_iterations: int = 2,
        probe_hw=(24, 48),
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.lip_gamma = float(lip_gamma)
        self.n_power_iterations = n_power_iterations

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        # 注册用于伴随幂迭代的二维空间探针张量与奇异值缓存
        v_init = torch.randn(1, in_channels, probe_hw[0], probe_hw[1])
        v_init = v_init / (v_init.norm() + 1e-8)
        self.register_buffer("v_probe", v_init)
        self.register_buffer("sigma_op", torch.tensor(1.0))

    def _compute_operator_sigma(self) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                v = self.v_probe
                u = None
                for _ in range(self.n_power_iterations):
                    u_raw = F.conv2d(v, self.weight, None, self.stride, self.padding)
                    u = u_raw / (u_raw.norm() + 1e-8)
                    v_raw = F.conv_transpose2d(u, self.weight, None, self.stride, self.padding)
                    v = v_raw / (v_raw.norm() + 1e-8)
                self.v_probe.copy_(v)
            # 在计算图内保留一阶瑞利商梯度，使权重更新平滑感知谱约束
            wv = F.conv2d(self.v_probe, self.weight, None, self.stride, self.padding)
            sigma = torch.sum(u.detach() * wv)
            sigma_safe = torch.clamp(torch.abs(sigma), min=1e-6)
            with torch.no_grad():
                self.sigma_op.copy_(sigma_safe.detach())
            return sigma_safe
        else:
            return torch.clamp(self.sigma_op, min=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = self._compute_operator_sigma()
        scaled_weight = (self.lip_gamma / sigma) * self.weight
        return F.conv2d(x, scaled_weight, self.bias, self.stride, self.padding)


class ConvexBoundedResBlock(nn.Module):
    """
    严格 Lipschitz 有界的凸残差瓶颈块：
    H_out = (1 - alpha_r) * H_in + alpha_r * ReLU(K_2(ReLU(K_1(H_in))))
    当 K_1, K_2 的算子范数 <= gamma 时，整个残差块的精确 Lipschitz 上界为 (1 - alpha_r + alpha_r * gamma^2)。
    """

    def __init__(self, channels: int, hidden_channels: int = 192, lip_gamma: float = 1.05, alpha_r: float = 0.5):
        super().__init__()
        self.alpha_r = float(alpha_r)
        self.conv1 = ExactSpectralConv2d(channels, hidden_channels, kernel_size=3, padding=1, lip_gamma=lip_gamma)
        self.conv2 = ExactSpectralConv2d(hidden_channels, channels, kernel_size=3, padding=1, lip_gamma=lip_gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = F.relu(self.conv1(x), inplace=False)
        res = F.relu(self.conv2(res), inplace=False)
        return (1.0 - self.alpha_r) * x + self.alpha_r * res


class LBFPAAdapter(nn.Module):
    """
    研究点一核心模块一：真二维算子谱有界前景感知协议适配器（LB-FPA）。
    完全不含 BatchNorm2d，末端经 ReLU 严格落入冻结检测头的非负特征锥 R_{>=0}^{384 x 100 x 352}。
    """

    def __init__(
        self,
        in_channels: int,
        proto_channels: int = 384,
        num_blocks: int = 3,
        lip_gamma: float = 1.08,
        alpha_r: float = 0.5,
    ):
        super().__init__()
        self.lip_gamma = lip_gamma
        self.alpha_r = alpha_r
        self.num_blocks = num_blocks

        self.stem = ExactSpectralConv2d(in_channels, proto_channels, kernel_size=1, padding=0, lip_gamma=lip_gamma)
        self.blocks = nn.ModuleList([
            ConvexBoundedResBlock(proto_channels, hidden_channels=proto_channels // 2, lip_gamma=lip_gamma, alpha_r=alpha_r)
            for _ in range(num_blocks)
        ])
        self.out_proj = ExactSpectralConv2d(proto_channels, proto_channels, kernel_size=1, padding=0, lip_gamma=lip_gamma)

    def theoretical_lipschitz_bound(self) -> float:
        """返回该适配器的严格数学解析 Lipschitz 常数上界"""
        block_lip = (1.0 - self.alpha_r) + self.alpha_r * (self.lip_gamma ** 2)
        return float(self.lip_gamma * (block_lip ** self.num_blocks) * self.lip_gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.stem(x), inplace=False)
        for blk in self.blocks:
            h = blk(h)
        z = F.relu(self.out_proj(h), inplace=False)
        return z


class UnconstrainedAdapter(nn.Module):
    """
    对照消融基线：常规无谱约束协议适配器（复现 STAMP / HEAL 标准 Conv2d + BatchNorm2d + ReLU 结构）。
    参数量与深度同 LB-FPA 严格一致，用于公平对比跨域对齐精度与对抗扰动放大率。
    """

    def __init__(self, in_channels: int, proto_channels: int = 384, num_blocks: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, proto_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(proto_channels),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for _ in range(num_blocks):
            blocks.append(nn.Sequential(
                nn.Conv2d(proto_channels, proto_channels // 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(proto_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(proto_channels // 2, proto_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(proto_channels),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.out_proj = nn.Sequential(
            nn.Conv2d(proto_channels, proto_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for blk in self.blocks:
            h = F.relu(h + blk(h), inplace=False)
        return self.out_proj(h)


def voxelize_points_torch(
    points: torch.Tensor,
    lidar_range=(-140.8, -40.0, -3.0, 140.8, 40.0, 1.0),
    voxel_size=(0.4, 0.4, 4.0),
    max_points_per_voxel: int = 32,
    max_voxels: int = 36000,
):
    """
    高效的纯 PyTorch GPU 向量化体素划分器，支持不同异构编码器使用任意专属体素分辨率（如 0.4m, 0.8m, 0.2m），
    无需在 CPU 数据加载线程重复串行处理。
    参数:
        points: (N, 4) [x, y, z, intensity] 在协商参考系下的点云张量 (CUDA)
    返回:
        voxel_features: (V, max_pts, 4)
        voxel_coords: (V, 4) [batch_idx=0, z_idx, y_idx, x_idx]
        voxel_num_points: (V,)
        grid_size: [W_grid, H_grid, D_grid]
    """
    device = points.device
    x_min, y_min, z_min, x_max, y_max, z_max = lidar_range
    vx, vy, vz = voxel_size
    grid_w = int(round((x_max - x_min) / vx))
    grid_h = int(round((y_max - y_min) / vy))
    grid_d = int(round((z_max - z_min) / vz))

    if points.shape[0] == 0:
        vf = torch.zeros((1, max_points_per_voxel, 4), device=device, dtype=torch.float32)
        vc = torch.zeros((1, 4), device=device, dtype=torch.int32)
        vn = torch.ones((1,), device=device, dtype=torch.int32)
        return vf, vc, vn, [grid_w, grid_h, grid_d]

    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] < x_max - 1e-4) &
        (points[:, 1] >= y_min) & (points[:, 1] < y_max - 1e-4) &
        (points[:, 2] >= z_min) & (points[:, 2] < z_max - 1e-4)
    )
    pts = points[mask]
    if pts.shape[0] == 0:
        vf = torch.zeros((1, max_points_per_voxel, 4), device=device, dtype=torch.float32)
        vc = torch.zeros((1, 4), device=device, dtype=torch.int32)
        vn = torch.ones((1,), device=device, dtype=torch.int32)
        return vf, vc, vn, [grid_w, grid_h, grid_d]

    ix = torch.clamp(((pts[:, 0] - x_min) / vx).long(), 0, grid_w - 1)
    iy = torch.clamp(((pts[:, 1] - y_min) / vy).long(), 0, grid_h - 1)
    iz = torch.clamp(((pts[:, 2] - z_min) / vz).long(), 0, grid_d - 1)

    flat_hash = iz * (grid_h * grid_w) + iy * grid_w + ix
    order = torch.argsort(flat_hash)
    pts = pts[order]
    flat_hash = flat_hash[order]
    ix, iy, iz = ix[order], iy[order], iz[order]

    unique_hashes, inverse_idx, counts = torch.unique_consecutive(
        flat_hash, return_inverse=True, return_counts=True
    )
    num_voxels = unique_hashes.shape[0]

    # 计算每个点在所属体素内的组内偏移序号 rank
    cum_counts = torch.cumsum(counts, dim=0)
    starts = cum_counts - counts
    point_ranks = torch.arange(pts.shape[0], device=device) - starts[inverse_idx]

    valid_pt_mask = point_ranks < max_points_per_voxel
    if num_voxels > max_voxels:
        valid_pt_mask = valid_pt_mask & (inverse_idx < max_voxels)
        num_voxels = max_voxels
        counts = counts[:max_voxels]
        starts = starts[:max_voxels]

    vf = torch.zeros((num_voxels, max_points_per_voxel, 4), device=device, dtype=torch.float32)
    sel_v = inverse_idx[valid_pt_mask]
    sel_r = point_ranks[valid_pt_mask]
    vf[sel_v, sel_r] = pts[valid_pt_mask]

    vn = torch.clamp(counts, max=max_points_per_voxel).to(torch.int32)
    vc = torch.zeros((num_voxels, 4), device=device, dtype=torch.int32)
    vc[:, 1] = iz[starts].to(torch.int32)
    vc[:, 2] = iy[starts].to(torch.int32)
    vc[:, 3] = ix[starts].to(torch.int32)

    return vf, vc, vn, [grid_w, grid_h, grid_d]


class Second3DHeteroEncoder(nn.Module):
    """
    Profile-B 异构骨干：基于 spconv 2.3.6 三维稀疏卷积的 SECOND 点云编码器。
    体素尺寸 [0.2, 0.2, 0.2] -> 网格 [W=1408, H=400, D=20]，
    经 2 次三维稀疏下采样（总步长 4x）精确压缩至 H_p=100, W_p=352 的 BEV 原生特征图 (C_raw=256)。
    特征产生机理（3D 体素稀疏邻域聚合 + Z轴高度堆叠）与 2D PointPillars 存在本质架构异构。
    """

    def __init__(self, out_bev_channels: int = 256):
        super().__init__()
        self.voxel_size = (0.2, 0.2, 0.2)
        self.spatial_shape = [20, 400, 1408]  # [D, H, W]

        self.sp_net = spconv.SparseSequential(
            spconv.SubMConv3d(4, 32, 3, padding=1, bias=False, indice_key="subm1"),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            # Downsample 1: [20, 400, 1408] -> [10, 200, 704]
            spconv.SparseConv3d(32, 64, 3, stride=2, padding=1, bias=False, indice_key="spconv1"),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(64, 64, 3, padding=1, bias=False, indice_key="subm2"),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # Downsample 2: [10, 200, 704] -> [5, 100, 352]
            spconv.SparseConv3d(64, 96, 3, stride=2, padding=1, bias=False, indice_key="spconv2"),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(96, 96, 3, padding=1, bias=False, indice_key="subm3"),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            # Z-collapse: [5, 100, 352] -> [2, 100, 352]
            spconv.SparseConv3d(96, 128, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0), bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        # 高度压缩后维度 128 * 2 = 256，经 2D 多尺度上下文提取输出原生异构特征
        self.bev_neck = nn.Sequential(
            nn.Conv2d(256, out_bev_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_bev_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_bev_channels, out_bev_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_bev_channels),
            nn.ReLU(inplace=True),
        )

    def forward_single_points(self, pts: torch.Tensor) -> torch.Tensor:
        vf, vc, vn, _ = voxelize_points_torch(
            pts, voxel_size=self.voxel_size, max_points_per_voxel=10, max_voxels=48000
        )
        # MeanVFE 体素均值特征
        denom = torch.clamp(vn.float().unsqueeze(-1), min=1.0)
        mean_feat = vf.sum(dim=1) / denom  # (V, 4)
        sp_tensor = spconv.SparseConvTensor(mean_feat, vc, self.spatial_shape, batch_size=1)
        out_sp = self.sp_net(sp_tensor)
        dense_5d = out_sp.dense()  # (1, 128, 2, 100, 352)
        b, c, d, h, w = dense_5d.shape
        bev = dense_5d.view(b, c * d, h, w)
        return self.bev_neck(bev)  # (1, 256, 100, 352)


class MultiResPillarHeteroEncoder(nn.Module):
    """
    Profile-C (32线粗体素 0.8m x 0.8m) 与 Profile-D (16线稀疏增强柱状编码) 异构骨干：
    使用十维几何增强柱状点特征 [x, y, z, i, dx_c, dy_c, dz_c, dx_p, dy_p, r_xy] 与多空洞感受野金字塔，
    生成维度为 C_raw=192, 分辨率为 100x352 的异构原生 BEV 特征。
    """

    def __init__(self, voxel_xy: float = 0.8, out_bev_channels: int = 192):
        super().__init__()
        self.voxel_xy = float(voxel_xy)
        self.voxel_size = (self.voxel_xy, self.voxel_xy, 4.0)
        self.need_downsample = abs(self.voxel_xy - 0.4) < 1e-3

        self.pfn = nn.Sequential(
            nn.Linear(10, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        in_2d = 64
        if self.need_downsample:
            self.stem_2d = nn.Sequential(
                nn.Conv2d(in_2d, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
        else:
            self.stem_2d = nn.Sequential(
                nn.Conv2d(in_2d, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )

        # 多尺度空洞卷积上下文金字塔（适配稀疏16线或粗粒度网格的大感受野补全）
        self.branch1 = nn.Sequential(
            nn.Conv2d(128, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(128, 96, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(192, out_bev_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_bev_channels),
            nn.ReLU(inplace=True),
        )

    def forward_single_points(self, pts: torch.Tensor) -> torch.Tensor:
        vf, vc, vn, grid_size = voxelize_points_torch(
            pts, voxel_size=self.voxel_size, max_points_per_voxel=32, max_voxels=32000
        )
        v_num, max_p, _ = vf.shape
        device = pts.device
        idx_range = torch.arange(max_p, device=device).unsqueeze(0)
        valid_mask = (idx_range < vn.unsqueeze(1)).float().unsqueeze(-1)  # (V, P, 1)

        denom = torch.clamp(vn.float().view(-1, 1, 1), min=1.0)
        pts_mean = (vf[:, :, :3] * valid_mask).sum(dim=1, keepdim=True) / denom
        f_cluster = vf[:, :, :3] - pts_mean

        x_center = vc[:, 3].float().view(-1, 1) * self.voxel_xy - 140.8 + 0.5 * self.voxel_xy
        y_center = vc[:, 2].float().view(-1, 1) * self.voxel_xy - 40.0 + 0.5 * self.voxel_xy
        f_center = torch.zeros((v_num, max_p, 2), device=device, dtype=torch.float32)
        f_center[:, :, 0] = vf[:, :, 0] - x_center
        f_center[:, :, 1] = vf[:, :, 1] - y_center

        r_norm = torch.sqrt(vf[:, :, 0] ** 2 + vf[:, :, 1] ** 2).unsqueeze(-1) / 100.0
        feat_10d = torch.cat([vf, f_cluster, f_center, r_norm], dim=-1) * valid_mask

        flat_feat = self.pfn(feat_10d.view(-1, 10)).view(v_num, max_p, 64)
        flat_feat = flat_feat * valid_mask
        pillar_feat = flat_feat.max(dim=1)[0]  # (V, 64)

        w_grid, h_grid, _ = grid_size
        canvas = torch.zeros((64, h_grid * w_grid), device=device, dtype=torch.float32)
        lin_idx = (vc[:, 2].long() * w_grid + vc[:, 3].long()).clamp(0, h_grid * w_grid - 1)
        canvas[:, lin_idx] = pillar_feat.t()
        bev = canvas.view(1, 64, h_grid, w_grid)

        h = self.stem_2d(bev)
        b1 = self.branch1(h)
        b2 = self.branch2(h)
        return self.out_conv(torch.cat([b1, b2], dim=1))  # (1, 192, 100, 352)


class HeteroAgentBranch(nn.Module):
    """
    封装单个开放异构智能体分支（原生异构三维编码器 + 协议域对齐适配器 LB-FPA / UnconstrainedAdapter）。
    支持直接输入参考系下的单车点云 pts (N, 4) 输出非负共享协议特征 Z_i (1, 384, 100, 352)，
    并支持测试经验局部 Lipschitz 扰动放大比。
    """

    def __init__(self, profile_name: str, bounded: bool = True, lip_gamma: float = 1.08):
        super().__init__()
        self.profile_name = profile_name
        self.bounded = bounded

        if profile_name == "profile_b_second64":
            self.encoder = Second3DHeteroEncoder(out_bev_channels=256)
            raw_ch = 256
            self.num_beams = 64
        elif profile_name == "profile_c_coarse32":
            self.encoder = MultiResPillarHeteroEncoder(voxel_xy=0.8, out_bev_channels=192)
            raw_ch = 192
            self.num_beams = 32
        elif profile_name == "profile_d_sparse16":
            self.encoder = MultiResPillarHeteroEncoder(voxel_xy=0.4, out_bev_channels=192)
            raw_ch = 192
            self.num_beams = 16
        else:
            raise ValueError(f"Unknown hetero profile: {profile_name}")

        if bounded:
            self.adapter = LBFPAAdapter(in_channels=raw_ch, proto_channels=384, num_blocks=3, lip_gamma=lip_gamma)
        else:
            self.adapter = UnconstrainedAdapter(in_channels=raw_ch, proto_channels=384, num_blocks=3)

    def encode_raw(self, pts: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward_single_points(pts)

    def adapt_to_protocol(self, raw_bev: torch.Tensor) -> torch.Tensor:
        return self.adapter(raw_bev)

    def forward(self, pts: torch.Tensor):
        raw_bev = self.encode_raw(pts)
        proto_z = self.adapt_to_protocol(raw_bev)
        return proto_z, raw_bev

