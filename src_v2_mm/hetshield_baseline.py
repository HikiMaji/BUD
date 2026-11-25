# -*- coding: utf-8 -*-
"""
HetShield 官方论文（arXiv:2609.17856 Sec. 3.4 & Appendix B.1/B.2）原生复现模块：
包含：
1. Lightweight ConvGRU：时序自回归特征预测网络（~0.9M 参数，1x1 降维 384->128 + 3x3 ConvGRU + 1x1 恢复 128->384，总参数量 0.984M）；
2. Temporal Consistency Score：逐网格归一化预测偏差，有效区域聚合；历史不足 2 帧时严格置 tau_k = 0；
3. Spatial Consistency Score：严格限定在本车与协作车双方均非静默（||f|| > 2.0）的重叠网格集合 O_k 上计算平均余弦相似度；
   当重叠区域占比 |O_k|/(H*W) < 10% 时，空间异常置 0；
4. Trust Gating & Feature Mitigation：整车信任门 s_k = alpha * tau_k + (1 - alpha) * max(0, 1 - sigma_k)，
   以及软门控特征向本车拉回缓解 tilde{f}_k = g_k * f_k + (1 - g_k) * f_ego。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRUCell(nn.Module):
    """标准二维卷积门控循环单元 (ConvGRU Cell)"""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.conv_gates = nn.Conv2d(
            in_channels + hidden_channels,
            2 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.conv_candidate = nn.Conv2d(
            in_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor = None) -> torch.Tensor:
        b, _, height, width = x.shape
        if h_prev is None:
            h_prev = torch.zeros(b, self.hidden_channels, height, width, device=x.device, dtype=x.dtype)

        combined = torch.cat([x, h_prev], dim=1)
        gates = torch.sigmoid(self.conv_gates(combined))
        r, z = gates.chunk(2, dim=1)

        candidate_input = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.conv_candidate(candidate_input))

        h_next = (1.0 - z) * h_prev + z * h_tilde
        return h_next


class TemporalConvGRUPredictor(nn.Module):
    """
    HetShield 论文轻量时序预测器（~0.9M 参数）：
    输入 384 维协议特征 -> 1x1 卷积降维至 128 维 -> 3x3 ConvGRU -> 1x1 卷积升维至 384 维输出预测。
    精确参数量：983,936 (0.984M)。
    """

    def __init__(self, in_channels: int = 384, hidden_channels: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.in_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True)
        self.cell = ConvGRUCell(hidden_channels, hidden_channels, kernel_size=3)
        self.out_proj = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, feat_cur: torch.Tensor, h_prev: torch.Tensor = None):
        x_proj = F.relu(self.in_proj(feat_cur), inplace=False)
        h_next = self.cell(x_proj, h_prev)
        feat_pred_next = F.relu(self.out_proj(h_next), inplace=False)
        return feat_pred_next, h_next


class HetShieldBaseline(nn.Module):
    """
    原生 HetShield 全局防御框架（arXiv:2609.17856）：
    维护单个共享的 TemporalConvGRUPredictor，在验证集上标定全局信任门限 gamma_th。
    """

    def __init__(
        self,
        in_channels: int = 384,
        hidden_channels: int = 128,
        alpha: float = 0.5,
        energy_thresh: float = 2.0,
        min_overlap_ratio: float = 0.10,
        kappa: float = 10.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.energy_thresh = float(energy_thresh)
        self.min_overlap_ratio = float(min_overlap_ratio)
        self.kappa = float(kappa)

        self.predictor = TemporalConvGRUPredictor(in_channels=in_channels, hidden_channels=hidden_channels)
        self.gamma_th = 0.45

    def compute_temporal_consistency(
        self,
        feat_cur: torch.Tensor,
        feat_pred: torch.Tensor = None,
        history_len: int = 2,
    ) -> torch.Tensor:
        if history_len < 2 or feat_pred is None:
            return torch.tensor(0.0, device=feat_cur.device)

        net_norm = feat_cur.norm(dim=1, keepdim=True)
        act_mask = (net_norm > self.energy_thresh).float()

        if act_mask.sum() < 5.0:
            return torch.tensor(0.0, device=feat_cur.device)

        diff = (feat_cur - feat_pred).norm(dim=1, keepdim=True)
        denom = feat_pred.norm(dim=1, keepdim=True) + 1e-4
        norm_diff = diff / denom

        tau_k = (norm_diff * act_mask).sum() / (act_mask.sum() + 1e-6)
        return tau_k

    def compute_spatial_consistency(
        self,
        feat_ego: torch.Tensor,
        feat_cav: torch.Tensor,
    ) -> torch.Tensor:
        _, _, height, width = feat_ego.shape
        total_grids = height * width

        norm_ego = feat_ego.norm(dim=1, keepdim=True)
        norm_cav = feat_cav.norm(dim=1, keepdim=True)

        overlap_mask = ((norm_ego > self.energy_thresh) & (norm_cav > self.energy_thresh)).float()
        overlap_cnt = overlap_mask.sum()

        if (overlap_cnt / float(total_grids)) < self.min_overlap_ratio:
            return torch.tensor(1.0, device=feat_ego.device)

        cos_sim = (feat_ego * feat_cav).sum(dim=1, keepdim=True) / (norm_ego * norm_cav + 1e-6)
        sigma_k = (cos_sim * overlap_mask).sum() / (overlap_cnt + 1e-6)
        return torch.clamp(sigma_k, -1.0, 1.0)

    def evaluate_trust(
        self,
        feat_ego: torch.Tensor,
        feat_cav: torch.Tensor,
        feat_pred_cav: torch.Tensor = None,
        history_len: int = 2,
        gamma_th: float = None,
    ) -> dict:
        th = self.gamma_th if gamma_th is None else gamma_th
        tau_k = self.compute_temporal_consistency(feat_cav, feat_pred_cav, history_len=history_len)
        sigma_k = self.compute_spatial_consistency(feat_ego, feat_cav)

        spat_anomaly = F.relu(1.0 - sigma_k)
        s_k = self.alpha * tau_k + (1.0 - self.alpha) * spat_anomaly

        g_k = torch.sigmoid(-self.kappa * (s_k - th))
        is_quarantined = (s_k > th).item()

        feat_mitigated = g_k * feat_cav + (1.0 - g_k) * feat_ego

        return {
            "tau_k": float(tau_k.item()),
            "sigma_k": float(sigma_k.item()),
            "s_k": float(s_k.item()),
            "g_k": float(g_k.item()),
            "is_quarantined": is_quarantined,
            "feat_mitigated": feat_mitigated,
        }
