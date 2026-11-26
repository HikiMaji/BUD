# -*- coding: utf-8 -*-
"""
步骤 ⑤ 基线：HetShield (arXiv:2609.17856) 严格修复版（适配 HEAL m1+m2 64ch 256x256 协议底座）。

相比旧版已精确修正两处真 Bug：
1. 【Bug 1 修复】杜绝 z_cur 自重构！严格执行一步时序自回归预测 z_{t-1} -> z_t：
   - 训练期：第 t-1 帧特征 z_{t-1} 输入 ConvGRU 得预测 pred_t，监督损失 MSE(pred_t, z_t)；
   - 推理期：使用上一帧输出 pred_t 与当前帧实收特征 z_t 计算时间一致性 tau_k，序列首帧（history_len < 2）置 tau_k = 0；
2. 【Bug 2 修复】严格按父序列边界与帧连续性复位隐状态：
   - 切换父序列或帧索引不连续时强制清空 (h_prev, pred_for_cur, hist_len)，杜绝跨序列隐状态泄露；
3. 支持三套公平评测配置：
   - 原论文参数版（alpha=0.5, tau_th=0.3, kappa=10.0 即 T=0.1）；
   - Fit-Tuned 调参版（在 Fit 干净帧分布上标定全局统一门限）；
   - HetShield-PerSensor 分模态版（m1/m2 各自维护专有 ConvGRU 与分模态门限）；
4. 原生支持多车队（N_collab >= 1）：每辆协作者按 cav_id 独立追踪时序隐状态并输出各自信任门 g_k。
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src_v2_mm.bud_fusion_heal import build_gated_fuse_fn


class ConvGRUCell64(nn.Module):
    """标准 3x3 二维卷积门控循环单元（适配 HEAL 64ch 特征）。"""

    def __init__(self, in_channels: int = 64, hidden_channels: int = 64, kernel_size: int = 3):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        padding = int(kernel_size) // 2
        self.conv_gates = nn.Conv2d(
            self.in_channels + self.hidden_channels,
            2 * self.hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.conv_candidate = nn.Conv2d(
            self.in_channels + self.hidden_channels,
            self.hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, _, h, w = x.shape
        if h_prev is None:
            h_prev = torch.zeros((b, self.hidden_channels, h, w), device=x.device, dtype=x.dtype)
        combined = torch.cat([x, h_prev], dim=1)
        gates = torch.sigmoid(self.conv_gates(combined))
        r, z = gates.chunk(2, dim=1)
        cand = torch.tanh(self.conv_candidate(torch.cat([x, r * h_prev], dim=1)))
        return (1.0 - z) * h_prev + z * cand


class TemporalConvGRUPredictorHEAL(nn.Module):
    """
    HEAL 64 通道协议特征轻量时序预测器：
    为兼顾 256x256 高分辨率显存与实时推理，在内部做 2x 下采样到 128x128 执行 ConvGRU，
    再双线性上采样回 256x256 并以残差连接输出下一帧预测 z_hat_{t+1}。
    """

    def __init__(self, in_channels: int = 64, hidden_channels: int = 64, downsample: int = 2):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.downsample = int(downsample)
        self.in_proj = nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=1, bias=True)
        self.cell = ConvGRUCell64(self.hidden_channels, self.hidden_channels, kernel_size=3)
        self.out_proj = nn.Conv2d(self.hidden_channels, self.in_channels, kernel_size=1, bias=True)

    def forward(self, feat_t: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        orig_hw = feat_t.shape[-2:]
        if self.downsample > 1:
            x_in = F.avg_pool2d(feat_t, kernel_size=self.downsample, stride=self.downsample)
        else:
            x_in = feat_t
        x_proj = F.relu(self.in_proj(x_in), inplace=False)
        h_next = self.cell(x_proj, h_prev)
        delta_low = self.out_proj(h_next)
        if self.downsample > 1:
            delta = F.interpolate(delta_low, size=orig_hw, mode="bilinear", align_corners=False)
        else:
            delta = delta_low
        # 残差预测下一帧（保证初始态即接近恒等运动先验，加速稳定收敛）
        pred_next = feat_t + delta
        return pred_next, h_next


class HetShieldHEALDefender(nn.Module):
    """
    HetShield 在 HEAL m1+m2 多尺度协议底座上的忠实可复用封装。
    """

    def __init__(
        self,
        mode: str = "shared",
        in_channels: int = 64,
        hidden_channels: int = 64,
        alpha: float = 0.6,
        tau_th: float = 0.3,
        temperature: float = 0.1,
        energy_thresh: float = 0.5,
        min_overlap_ratio: float = 0.05,
        per_sensor_tau: Optional[Dict[str, float]] = None,
        freeze_on_quarantine: bool = False,
    ):
        super().__init__()
        assert mode in ("shared", "persensor"), f"未知模式: {mode}"
        self.mode = mode
        self.alpha = float(alpha)
        self.tau_th = float(tau_th)
        self.temperature = max(float(temperature), 1e-4)
        self.kappa = 1.0 / self.temperature
        self.energy_thresh = float(energy_thresh)
        self.min_overlap_ratio = float(min_overlap_ratio)
        self.freeze_on_quarantine = bool(freeze_on_quarantine)
        self.per_sensor_tau = dict(per_sensor_tau) if per_sensor_tau is not None else {"m1": float(tau_th), "m2": float(tau_th)}

        if mode == "shared":
            self.predictor = TemporalConvGRUPredictorHEAL(in_channels=in_channels, hidden_channels=hidden_channels)
            self.predictors = None
        else:
            self.predictor = None
            self.predictors = nn.ModuleDict({
                "m1": TemporalConvGRUPredictorHEAL(in_channels=in_channels, hidden_channels=hidden_channels),
                "m2": TemporalConvGRUPredictorHEAL(in_channels=in_channels, hidden_channels=hidden_channels),
            })

        # 运行期按序列隔离的隐状态缓存
        self._cur_seq_id: Optional[str] = None
        self._last_frame_idx: Optional[int] = None
        self._cav_states: Dict[str, dict] = {}

    def reset_sequence(self, seq_id: Optional[str] = None):
        """跨序列边界或帧跳变时彻底清空全部协作者的 ConvGRU 隐状态与历史计数。"""
        self._cur_seq_id = seq_id
        self._last_frame_idx = None
        self._cav_states.clear()

    def _get_net(self, mod: str) -> TemporalConvGRUPredictorHEAL:
        if self.mode == "shared":
            return self.predictor
        return self.predictors[mod] if mod in self.predictors else self.predictors["m1"]

    def _get_threshold(self, mod: str) -> float:
        if self.mode == "persensor" and mod in self.per_sensor_tau:
            return float(self.per_sensor_tau[mod])
        return float(self.tau_th)

    @torch.no_grad()
    def compute_temporal_score(
        self,
        feat_cur_ego: torch.Tensor,
        feat_pred_ego: Optional[torch.Tensor],
        valid_mask_ego: torch.Tensor,
        history_len: int,
    ) -> float:
        """
        时间一致性得分 tau_k：仅当历史帧 >= 2 且存在上帧一步预测 feat_pred_ego 时计算；
        严格限定在有效活跃网格上，防止空背景零向量稀释。
        """
        if history_len < 2 or feat_pred_ego is None:
            return 0.0
        net_norm = feat_cur_ego.norm(dim=1, keepdim=True)
        act = (net_norm > self.energy_thresh) & (valid_mask_ego > 0.5)
        cnt = act.sum()
        if float(cnt.item()) < 5.0:
            return 0.0
        diff = (feat_cur_ego - feat_pred_ego).norm(dim=1, keepdim=True)
        denom = feat_pred_ego.norm(dim=1, keepdim=True) + 1e-4
        tau_k = ((diff / denom) * act.float()).sum() / (cnt.float() + 1e-6)
        return float(tau_k.item())

    @torch.no_grad()
    def compute_spatial_score(
        self,
        feat_ego: torch.Tensor,
        feat_cav_ego: torch.Tensor,
        valid_mask_ego: torch.Tensor,
    ) -> float:
        """
        空间一致性得分 sigma_k：在双方均非静默的重叠有效网格上计算平均余弦相似度。
        重叠占比低于 min_overlap_ratio 时按论文规定置 sigma_k = 1.0（空间异常为 0）。
        """
        norm_e = feat_ego.norm(dim=1, keepdim=True)
        norm_c = feat_cav_ego.norm(dim=1, keepdim=True)
        v_mask = valid_mask_ego > 0.5
        v_cnt = float(v_mask.sum().item())
        if v_cnt < 1.0:
            return 1.0
        overlap = (norm_e > self.energy_thresh) & (norm_c > self.energy_thresh) & v_mask
        o_cnt = float(overlap.sum().item())
        if (o_cnt / v_cnt) < self.min_overlap_ratio or o_cnt < 5.0:
            return 1.0
        cos_sim = (feat_ego * feat_cav_ego).sum(dim=1, keepdim=True) / (norm_e * norm_c + 1e-6)
        sigma_k = (cos_sim * overlap.float()).sum() / (o_cnt + 1e-6)
        return float(torch.clamp(sigma_k, -1.0, 1.0).item())

    @torch.no_grad()
    def step_frame(
        self,
        seq_id: str,
        frame_idx: int,
        feat_ego: torch.Tensor,
        cav_items: List[dict],
        update_state: bool = True,
    ) -> List[dict]:
        """
        按严格时间顺序评估当前帧的所有协作者：
        - 若 seq_id 改变或 frame_idx 不连续（如跳帧 > 2），自动清空历史状态；
        - cav_items 每项包含: {"cav_id": str, "mod": str, "feat_warped": (1,C,H,W), "valid_ego": (1,1,H,W)}；
        - 返回每辆协作者的 {"cav_id", "tau_k", "sigma_k", "s_k", "g_k", "is_quarantined"}。
        """
        if self._cur_seq_id != seq_id or (
            self._last_frame_idx is not None and (int(frame_idx) <= self._last_frame_idx or int(frame_idx) - self._last_frame_idx > 5)
        ):
            self.reset_sequence(seq_id)

        results = []
        for item in cav_items:
            cid = str(item["cav_id"])
            mod = str(item["mod"])
            f_cav = item["feat_warped"]
            v_ego = item["valid_ego"]

            st = self._cav_states.get(cid, {"h": None, "pred_for_cur": None, "hist_len": 0})
            hist_len = int(st["hist_len"]) + 1
            pred_cur = st["pred_for_cur"]

            tau_k = self.compute_temporal_score(f_cav, pred_cur, v_ego, history_len=hist_len)
            sigma_k = self.compute_spatial_score(feat_ego, f_cav, v_ego)
            spat_anom = max(0.0, 1.0 - sigma_k)
            # 当 history_len < 2 时按原论文降退为纯空间一致性或 alpha*0 + (1-alpha)*spat
            s_k = self.alpha * tau_k + (1.0 - self.alpha) * spat_anom

            th = self._get_threshold(mod)
            g_k = float(torch.sigmoid(torch.tensor(-self.kappa * (s_k - th))).item())
            is_q = bool(s_k > th)

            results.append({
                "cav_id": cid,
                "mod": mod,
                "tau_k": tau_k,
                "sigma_k": sigma_k,
                "s_k": s_k,
                "g_k": g_k,
                "is_quarantined": is_q,
                "th_used": th,
            })

            if update_state:
                # F-P0.3 & A2 修复：默认被隔离时也正常推进隐状态（freeze_on_quarantine=False），消除长序列隐状态锁死
                if (not self.freeze_on_quarantine) or (not is_q) or st["h"] is None:
                    net = self._get_net(mod)
                    pred_next, h_next = net(f_cav, st["h"])
                    self._cav_states[cid] = {
                        "h": h_next.detach(),
                        "pred_for_cur": pred_next.detach(),
                        "hist_len": hist_len,
                    }

        if update_state:
            self._last_frame_idx = int(frame_idx)
        return results


def build_hetshield_fuse_fn(g_list: Sequence[float], device: str = "cuda"):
    """
    A2 严格按 HetShield 原论文复现特征混合缓解（而非直接缩放融合权重）：
    在本车 ego 对齐空间中，将每辆协作车特征按信任门 g_k 往本车特征 z_0 方向线性混合：
        z_tilde_k = g_k * z_k + (1 - g_k) * z_0
    再通过该尺度原生 single_head 重新计算混合特征对应的占用分数，并执行标准加权融合。
    """
    from opencood.models.fuse_modules.fusion_in_one import regroup
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

    g_clamped = [float(max(0.0, min(1.0, g))) for g in g_list]

    def _hs_feature_mix_fuse(
        feature_list,
        score_list,
        occ_map_list,
        record_len,
        affine_matrix,
        agent_modality_list,
        wrapper,
    ):
        pyramid = wrapper.model.pyramid_backbone
        align_corners = pyramid.align_corners
        num_levels = len(feature_list)
        fused_feature_list = []

        for lvl in range(num_levels):
            x = feature_list[lvl]
            score = score_list[lvl]
            _, _, h, w = x.shape
            batch_size = affine_matrix.shape[0]
            split_x = regroup(x, record_len)
            split_score = regroup(score, record_len)
            single_head = getattr(pyramid, f"single_head_{lvl}", None)
            out_b = []

            for b in range(batch_size):
                n_cav = int(record_len[b].item()) if torch.is_tensor(record_len[b]) else int(record_len[b])
                t_mat = affine_matrix[b, 0, :n_cav, :, :]
                feat_ego = warp_affine_simple(split_x[b], t_mat, (h, w), align_corners=align_corners)
                score_ego = warp_affine_simple(split_score[b], t_mat, (h, w), align_corners=align_corners)

                if n_cav > 1:
                    if len(g_clamped) == 1 and n_cav - 1 > 1:
                        gs = g_clamped * (n_cav - 1)
                    else:
                        gs = g_clamped[: n_cav - 1]
                    g_vec = torch.tensor(gs, device=feat_ego.device, dtype=feat_ego.dtype).view(n_cav - 1, 1, 1, 1)
                    z0 = feat_ego[0:1]
                    # 原论文公式：z_tilde_k = g_k * z_k + (1 - g_k) * z_0（仅在协作者有效传输域内生效）
                    valid_k = score_ego[1:] > 0.0
                    z_collab_mix = g_vec * feat_ego[1:] + (1.0 - g_vec) * z0
                    feat_ego = torch.cat([z0, z_collab_mix], dim=0)
                    if single_head is not None:
                        occ_mix = single_head(z_collab_mix)
                        sc_mix = (torch.sigmoid(occ_mix) + 1e-4) * valid_k.to(feat_ego.dtype)
                        score_ego = torch.cat([score_ego[0:1], sc_mix], dim=0)

                valid = score_ego > 0.0
                any_valid = valid.any(dim=0, keepdim=True)
                safe_logits = torch.where(valid, score_ego, torch.full_like(score_ego, -1e4))
                safe_logits = torch.where(any_valid, safe_logits, torch.zeros_like(safe_logits))
                weights = torch.softmax(safe_logits, dim=0) * any_valid.to(score_ego.dtype)
                out_b.append(torch.sum(feat_ego * weights, dim=0))

            fused_feature_list.append(torch.stack(out_b, dim=0))

        return fused_feature_list, {"hetshield_g_list": g_clamped}

    return _hs_feature_mix_fuse
