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


def mix_hetshield_protocol_features_ego(
    heter_feature_2d: torch.Tensor,
    record_len: torch.Tensor,
    affine_matrix: torch.Tensor,
    cam_valid_mask: torch.Tensor,
    g_list: Sequence[float],
    align_corners: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """
    严格对照 HetShield 原论文 Sec 3.2 - Sec 3.4：
    仅在接收端进入多尺度金字塔（pyramid_backbone）之前的**传输协议特征层**执行唯一一次线性混合！
      1. 将各车协议特征 f_k 与有效掩膜 cam_valid_mask 对齐到本车 ego 坐标系，得 z_k 与 valid_ego_k；
      2. 对协作车 k >= 1，按信任门 g_k 往本车特征 z_0 线性混合一次：
           z_tilde_k = g_k * z_k + (1 - g_k) * z_0
      3. 返回混合后的协议特征 (N_total, C, H, W) 与对齐后的有效掩膜 (N_total, 1, H, W)，
         后续进入金字塔 L0/L1/L2 时走原生多尺度骨干与加权融合，绝不在金字塔每一层重复混合。
    """
    from opencood.models.fuse_modules.fusion_in_one import regroup
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

    g_clamped = [float(max(0.0, min(1.0, g))) for g in g_list]
    _, _, h, w = heter_feature_2d.shape
    batch_size = affine_matrix.shape[0]
    split_feat = regroup(heter_feature_2d, record_len)
    split_vmask = regroup(cam_valid_mask.to(dtype=heter_feature_2d.dtype), record_len)

    mixed_batches = []
    valid_batches = []
    cav_cursor = 0

    for b in range(batch_size):
        n_cav = int(record_len[b].item()) if torch.is_tensor(record_len[b]) else int(record_len[b])
        t_mat = affine_matrix[b, 0, :n_cav, :, :]
        feat_ego = warp_affine_simple(split_feat[b], t_mat, (h, w), align_corners=align_corners)
        vmask_ego = warp_affine_simple(split_vmask[b], t_mat, (h, w), align_corners=align_corners) > 0.5
        vmask_ego_f = vmask_ego.to(dtype=heter_feature_2d.dtype)

        if n_cav > 1:
            n_col = n_cav - 1
            if len(g_clamped) == 1 and n_col > 1 and batch_size == 1:
                gs = g_clamped * n_col
            else:
                gs = g_clamped[cav_cursor : cav_cursor + n_col]
                if len(gs) < n_col:
                    gs = list(gs) + [1.0] * (n_col - len(gs))
            cav_cursor += n_col

            g_vec = torch.tensor(gs, device=feat_ego.device, dtype=feat_ego.dtype).view(n_col, 1, 1, 1)
            z0 = feat_ego[0:1]
            # 仅在传输协议层混合一次：z_tilde_k = g_k * z_k + (1 - g_k) * z_0
            z_collab_mix = g_vec * feat_ego[1:] + (1.0 - g_vec) * z0
            feat_ego = torch.cat([z0, z_collab_mix], dim=0)

        mixed_batches.append(feat_ego)
        valid_batches.append(vmask_ego_f)

    return torch.cat(mixed_batches, dim=0), torch.cat(valid_batches, dim=0), g_clamped


def forward_hetshield_protocol_mix(
    wrapper,
    heter_feature_2d: torch.Tensor,
    record_len: torch.Tensor,
    affine_matrix: torch.Tensor,
    agent_modality_list: Sequence[str],
    cam_valid_mask: torch.Tensor,
    g_list: Sequence[float],
) -> dict:
    """
    HetShield 标准前向推理：
    1. 在进入 pyramid_backbone 之前的传输协议特征层执行单次混合 z_tilde_k = g_k * z_k + (1 - g_k) * z_0；
    2. 将混合后的 ego 系协议特征送入原生 pyramid.get_multiscale_feature 获取 L0/L1/L2 多尺度特征；
    3. 各尺度用原生 single_head_i 计算占用分数并执行标准 softmax 加权融合（金字塔各层内部零重复混合！）；
    4. 经 decode_multiscale_feature 与检测头输出预测结果。
    """
    from opencood.models.fuse_modules.fusion_in_one import regroup

    pyramid = wrapper.model.pyramid_backbone
    align_corners = pyramid.align_corners

    z_mixed_ego, valid_ego_l0, g_clamped = mix_hetshield_protocol_features_ego(
        heter_feature_2d=heter_feature_2d,
        record_len=record_len,
        affine_matrix=affine_matrix,
        cam_valid_mask=cam_valid_mask,
        g_list=g_list,
        align_corners=align_corners,
    )

    # 协议层已完成唯一一次混合，下面走完全原生的多尺度金字塔与检测头（不再逐层混合）
    feature_list = list(pyramid.get_multiscale_feature(z_mixed_ego))
    occ_map_list = []
    fused_feature_list = []

    for lvl in range(pyramid.num_levels):
        feat_lvl = feature_list[lvl]
        _, _, hl, wl = feat_lvl.shape
        occ_map = getattr(pyramid, f"single_head_{lvl}")(feat_lvl)
        occ_map_list.append(occ_map)

        if valid_ego_l0.shape[-2:] == (hl, wl):
            v_lvl = valid_ego_l0 > 0.5
        else:
            v_lvl = F.interpolate(valid_ego_l0, size=(hl, wl), mode="nearest") > 0.5
        score_lvl = (torch.sigmoid(occ_map) + 1e-4) * v_lvl.to(dtype=occ_map.dtype)

        split_x = regroup(feat_lvl, record_len)
        split_s = regroup(score_lvl, record_len)
        out_b = []
        for b in range(len(split_x)):
            xb = split_x[b]
            sb = split_s[b]
            valid = sb > 0.0
            any_valid = valid.any(dim=0, keepdim=True)
            safe_logits = torch.where(valid, sb, torch.full_like(sb, -1e4))
            safe_logits = torch.where(any_valid, safe_logits, torch.zeros_like(safe_logits))
            weights = torch.softmax(safe_logits, dim=0) * any_valid.to(sb.dtype)
            out_b.append(torch.sum(xb * weights, dim=0))
        fused_feature_list.append(torch.stack(out_b, dim=0))

    decoded = pyramid.decode_multiscale_feature(fused_feature_list)
    if getattr(wrapper.model, "shrink_flag", False):
        decoded = wrapper.model.shrink_conv(decoded)

    return {
        "pyramid": "collab",
        "cls_preds": wrapper.model.cls_head(decoded),
        "reg_preds": wrapper.model.reg_head(decoded),
        "dir_preds": wrapper.model.dir_head(decoded),
        "occ_single_list": occ_map_list,
        "pyramid_feats": feature_list,
        "decoded_feat": decoded,
        "fusion_diag": {
            "hetshield_g_list": g_clamped,
            "mixing_layer": "protocol_feature_only_once",
            "pyramid_per_level_mixing": False,
        },
    }


def build_hetshield_fuse_fn(*args, **kwargs):
    """已禁用旧版金字塔逐层混合闭包，强制使用 forward_hetshield_protocol_mix 在传输特征层单次混合。"""
    raise RuntimeError(
        "按原论文 Sec 3.4 与代码审查规范，HetShield 仅在传输特征层混合一次，严禁在金字塔逐层混合！"
        "请调用 forward_hetshield_protocol_mix(...)。"
    )
