# -*- coding: utf-8 -*-
"""
强对照基线：HetShield-PerSensor
解决原生 HetShield 对全部异构车队使用单一时序模型与单一切割门限造成的系统性弱配车误杀缺陷：
1. 按传感器模态与架构分别训练专有 ConvGRU（SECOND-64, Coarse-32, Sparse-16 各自维护适配其点云密度的自回归模型，各 ~0.98M 参数）；
2. 在独立验证集上，按传感器模态分别独立标定 95% 分位数门限 gamma_th^(m)；
3. 推理时分流评估，消除跨传感器动态差异带来的系统性误降权。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src_v2_mm.hetshield_baseline import TemporalConvGRUPredictor


class HetShieldPerSensor(nn.Module):
    """
    分传感器专有 HetShield-PerSensor 强对照模型：
    维护专有 ConvGRU 字典与分模态门限。
    """

    def __init__(
        self,
        profile_names: list = None,
        in_channels: int = 384,
        hidden_channels: int = 128,
        alpha: float = 0.5,
        energy_thresh: float = 2.0,
        min_overlap_ratio: float = 0.10,
        kappa: float = 10.0,
    ):
        super().__init__()
        if profile_names is None:
            profile_names = ["profile_b_second64", "profile_c_coarse32", "profile_d_sparse16"]

        self.profile_names = profile_names
        self.alpha = float(alpha)
        self.energy_thresh = float(energy_thresh)
        self.min_overlap_ratio = float(min_overlap_ratio)
        self.kappa = float(kappa)

        self.predictors = nn.ModuleDict({
            p: TemporalConvGRUPredictor(in_channels=in_channels, hidden_channels=hidden_channels)
            for p in profile_names
        })

        # 分模态门限字典（在验证集上独立标定）
        self.per_sensor_gamma_th = {p: 0.45 for p in profile_names}

    def get_predictor(self, profile_name: str) -> TemporalConvGRUPredictor:
        if profile_name in self.predictors:
            return self.predictors[profile_name]
        return list(self.predictors.values())[0]

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

    def evaluate_trust_persensor(
        self,
        feat_ego: torch.Tensor,
        feat_cav: torch.Tensor,
        profile_name: str,
        feat_pred_cav: torch.Tensor = None,
        history_len: int = 2,
    ) -> dict:
        th = self.per_sensor_gamma_th.get(profile_name, 0.45)
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
            "sensor_th": th,
        }
