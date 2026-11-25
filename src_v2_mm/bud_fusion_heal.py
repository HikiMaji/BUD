# -*- coding: utf-8 -*-
"""
BUD-Fusion 核心融合与局部审计模块（研究点一 · HEAL m1+m2 多尺度底座 · 冻结升级版）。

核心特性：
1. 零代码耦合设计：全部分档基准、通道权重、可见性分档余弦门限与整车回退高门限均从外部 JSON 加载，
   验证集就绪后仅替换 JSON 即可完成重标定；
2. 创新点 1（多尺度余弦 L0/L1/L2 + 清晰语义一致性 × 5距离[含0~6m] × 3可见性非对称调制）计算空间异常图，
   帧级校准严格采用稳健 div_p95（坚决不用窄分母 med_p95）；
3. 解决干净盲区召回掉点：在位置容差有向语义 + 多尺度余弦协同软门控中引入按本车可见性 v_idx 分档的
   余弦死区 cos_tau_by_vis 与可见性衰减系数 gamma_cos_gate，保护本车遮挡区诚实补盲真车，同时守住遮挡区伪造防线；
4. 两级混合防御：单辆协作者整帧异常分数超过 Fit 标定高门限 frame_high_tau 时触发整车级回退（gate 全 0），
   否则执行精细空间软门控；
5. 原生支持多协作车队（N_collab >= 1）：每辆协作者独立生成 gate_L0[c] 并统计各自的基准/门控后融合权重，
   削减份额按比例无损回流至可信本车。
"""

import json
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from src_v2_mm.step4_1_v2_core import (
    DIST_LABELS_5,
    PROTO_H,
    PROTO_W,
    VIS_LABELS_3,
    _get_tier_channel_z,
    _masked_avg_pool3x3,
)


def compute_shift_tolerant_semantic_maps(
    p_ego_cls: torch.Tensor,
    p_fused_cls: torch.Tensor,
    cov_ego: torch.Tensor,
    k_tol: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 k_tol x k_tol 位置容差有向语义冲突图（容忍多车融合带来的 ±1.6m 中心微移）：
    - d_drop_tol: 全域逆向擦除冲突 ReLU(P_ego - max_{N_k}(P_fused))，诚实盲区补全处恒为 0；
    - d_add_tol : 正向新增目标冲突 ReLU(P_fused - max_{N_k}(P_ego))，后续按本车可见性调制。
    """
    pad = int(k_tol) // 2
    pf_max = F.max_pool2d(p_fused_cls, kernel_size=int(k_tol), stride=1, padding=pad)
    pe_max = F.max_pool2d(p_ego_cls, kernel_size=int(k_tol), stride=1, padding=pad)
    cov_f = cov_ego.to(dtype=p_ego_cls.dtype)
    d_drop_tol = (torch.clamp(p_ego_cls - pf_max, min=0.0) * cov_f)[0, 0]
    d_add_tol = (torch.clamp(p_fused_cls - pe_max, min=0.0) * cov_f)[0, 0]
    return d_drop_tol, d_add_tol


def _compute_spatial_anomaly_and_cos(
    dev_4ch: torch.Tensor,
    cov_2d: torch.Tensor,
    d_idx: torch.Tensor,
    v_idx: torch.Tensor,
    mod: str,
    stats: dict,
    tier_name: str,
    cfg: dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """按指定门限层级计算本车坐标系下 (256, 256) 空间异常得分图与多尺度余弦组合图。"""
    q_key = f"p{int(cfg['q_scale'])}"
    z = _get_tier_channel_z(dev_4ch, mod, d_idx, v_idx, stats, tier_name, q_key=q_key, s_floor=0.05)

    w0, w1, w2 = cfg["cos_weights"]
    cos_comb = w0 * z[0] + w1 * z[1] + w2 * z[2]
    w_sem = float(cfg["w_sem"])

    if tier_name == "tier4_sensor_dist_vis":
        g_cos = torch.tensor(cfg["gamma_cos"], device=dev_4ch.device, dtype=dev_4ch.dtype)[v_idx]
        g_sem = torch.tensor(cfg["gamma_sem"], device=dev_4ch.device, dtype=dev_4ch.dtype)[v_idx]
        s_map = (1.0 - w_sem) * (g_cos * cos_comb) + w_sem * (g_sem * z[3])
    else:
        s_map = (1.0 - w_sem) * cos_comb + w_sem * z[3]

    cov_f = cov_2d.to(dtype=dev_4ch.dtype)
    return s_map * cov_f, cos_comb * cov_f


def _compute_spatial_anomaly_map(
    dev_4ch: torch.Tensor,
    cov_2d: torch.Tensor,
    d_idx: torch.Tensor,
    v_idx: torch.Tensor,
    mod: str,
    stats: dict,
    tier_name: str,
    cfg: dict,
) -> torch.Tensor:
    """兼容旧接口：按指定门限层级返回空间异常得分图 (256, 256)。"""
    s_map, _ = _compute_spatial_anomaly_and_cos(dev_4ch, cov_2d, d_idx, v_idx, mod, stats, tier_name, cfg)
    return s_map


def build_gate_map_from_smap(
    s_map: torch.Tensor,
    cov_2d: torch.Tensor,
    tau_loc: float,
    r_dil: int,
    beta: float,
) -> torch.Tensor:
    """
    由空间异常图 s_map (256, 256) 构造连续局部门控信任图 gate_L0 (1, 1, 256, 256) ∈ (0, 1]：
    - 小于像素 deadzone tau_loc 的区域保持严格 1.0（保护干净帧融合零损失）；
    - 超门限异常先经 (2*r_dil+1) 最大池化覆及检测头感受野，再做 5x5 平滑与指数衰减。
    """
    cov_f = cov_2d.to(dtype=s_map.dtype)
    exc = torch.clamp(s_map - float(tau_loc), min=0.0) * cov_f
    exc_4d = exc.view(1, 1, PROTO_H, PROTO_W)

    if int(r_dil) > 0:
        k_dil = 2 * int(r_dil) + 1
        exc_dil = F.max_pool2d(exc_4d, kernel_size=k_dil, stride=1, padding=int(r_dil))
    else:
        exc_dil = exc_4d

    exc_smooth = F.avg_pool2d(exc_dil, kernel_size=5, stride=1, padding=2)
    gate_l0 = torch.exp(-float(beta) * exc_smooth)
    gate_l0 = torch.where(cov_2d.view(1, 1, PROTO_H, PROTO_W), gate_l0, torch.ones_like(gate_l0))
    return torch.clamp(gate_l0, min=0.0, max=1.0)


def build_directional_gate_map(
    cos_comb: torch.Tensor,
    d_drop_tol: torch.Tensor,
    d_add_tol: torch.Tensor,
    cov_2d: torch.Tensor,
    v_idx: torch.Tensor,
    cos_tau: Union[float, Sequence[float]],
    drop_tau: float = 0.45,
    add_tau: float = 0.30,
    w_drop: float = 7.0,
    w_add: float = 6.0,
    cos_co_ratio: float = 0.82,
    gamma_add: Sequence[float] = (1.0, 0.3, 0.0),
    gamma_cos_gate: Sequence[float] = (1.0, 1.0, 1.0),
    r_dil: int = 5,
    beta: float = 3.5,
    collab_fg_mask: Optional[torch.Tensor] = None,
    cos_tau_fg: Optional[Sequence[float]] = None,
    return_components: bool = False,
):
    """
    位置容差有向语义 + 前景/空地条件分档多尺度余弦协同软门控生成器：
    1. 原因 A 根治：按 (本车可见性 v_idx ∈ {Clear, Sparse, Occluded} × 协作车前景状态 {bg空地, fg前景})
       分别查表对应的良性标定余弦分位门限（cos_tau 为空地格门限，cos_tau_fg 为前景格门限）；
       - 空地格保持紧凑门限，不放宽；
       - 前景格采用「诚实协作车补盲前景格」真实良性分位数定门限，既不误伤遮挡区诚实真车，
         又不将遮挡区权重置 0，从而保留对「只在遮挡区伪造」高幅对抗异常的检出能力；
    2. 全域 5x5 容差逆向擦除冲突 d_drop_tol 超 drop_tau 独立触发（保护本车强自信真车）；
    3. 5x5 容差正向新增冲突 d_add_tol 经本车可见性 gamma_add 与局部特征余弦协同门控（Clear 区严格防鬼影）。
    """
    cm_s = F.avg_pool2d(cos_comb.view(1, 1, PROTO_H, PROTO_W), kernel_size=3, stride=1, padding=1)[0, 0]

    if isinstance(cos_tau, (list, tuple)):
        tau_bg_vec = torch.tensor([float(x) for x in cos_tau], device=cos_comb.device, dtype=cos_comb.dtype)
        cos_tau_map = tau_bg_vec[v_idx]
    else:
        cos_tau_map = torch.full_like(cm_s, float(cos_tau))

    # 若启用前景/空地双分档，在协作车存在前景响应的格子切换至前景良性分位门限
    # 注意：Clear(0) 区仍有 exc_add 严防清晰空地鬼影，若传入 cos_tau_fg 则按 [Clear, Sparse, Occluded] 精确查表
    if collab_fg_mask is not None and cos_tau_fg is not None:
        tau_fg_vec = torch.tensor([float(x) for x in cos_tau_fg], device=cos_comb.device, dtype=cos_comb.dtype)
        cos_tau_fg_map = tau_fg_vec[v_idx]
        fg_bool = collab_fg_mask.to(device=cos_comb.device, dtype=torch.bool)
        cos_tau_map = torch.where(fg_bool, cos_tau_fg_map, cos_tau_map)

    g_cos_map = torch.tensor([float(x) for x in gamma_cos_gate], device=cos_comb.device, dtype=cos_comb.dtype)[v_idx]
    exc_cos = torch.clamp(cm_s - cos_tau_map, min=0.0) * g_cos_map
    exc_drop = torch.clamp(d_drop_tol - float(drop_tau), min=0.0) * float(w_drop)

    # Clear 区正向新增鬼影的余弦协同门控始终以紧凑空地基线为参考，防止攻击者借 fg 桶逃逸清晰空地鬼影检查
    if isinstance(cos_tau, (list, tuple)):
        tau_bg_ref = torch.tensor([float(x) for x in cos_tau], device=cos_comb.device, dtype=cos_comb.dtype)[v_idx]
    else:
        tau_bg_ref = torch.full_like(cm_s, float(cos_tau))
    co_th_map = float(cos_co_ratio) * tau_bg_ref
    cos_gate = torch.clamp((cm_s - co_th_map) / 0.35, min=0.0, max=1.0)
    g_add_map = torch.tensor([float(x) for x in gamma_add], device=cos_comb.device, dtype=cos_comb.dtype)[v_idx]
    exc_add = torch.clamp(d_add_tol - float(add_tau), min=0.0) * g_add_map * cos_gate * float(w_add)

    cov_f = cov_2d.to(dtype=cos_comb.dtype)
    exc_tot = (exc_cos + exc_drop + exc_add) * cov_f
    gate_l0 = build_gate_map_from_smap(exc_tot, cov_2d, tau_loc=0.0, r_dil=int(r_dil), beta=float(beta))
    if return_components:
        return gate_l0, {
            "cm_s": cm_s * cov_f,
            "exc_cos": exc_cos * cov_f,
            "exc_drop": exc_drop * cov_f,
            "exc_add": exc_add * cov_f,
            "exc_tot": exc_tot,
        }
    return gate_l0


def build_gated_fuse_fn(gate_L0: torch.Tensor) -> Callable:
    """
    构造可嵌入 HEALBaseWrapper.forward_from_protocol(..., custom_fuse_fn=...) 的多尺度局部门控融合闭包。

    支持单协作者 (1, 1, 256, 256) 及多车队 (N_collab, 1, 256, 256) 独立门控：
    先计算与原生 weighted_fuse_safe 完全一致的多尺度基准融合权重 w_base，
    再以 g_l[c] 缩放第 c 辆协作者权重，并将所有协作者削减的总份额无损回流给可信本车 ego，
    保证任意网格权重总和恒定为 1（在有效区域内）。
    """
    from opencood.models.fuse_modules.fusion_in_one import regroup
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

    def _custom_fuse(
        feature_list,
        score_list,
        occ_map_list,
        record_len,
        affine_matrix,
        agent_modality_list,
        wrapper,
    ):
        align_corners = wrapper.model.pyramid_backbone.align_corners
        num_levels = len(feature_list)
        fused_feature_list = []
        level_diags = {}
        w_cav_l0_map = None
        per_cav_w_base_l0 = []
        per_cav_w_gated_l0 = []

        for lvl in range(num_levels):
            x = feature_list[lvl]
            score = score_list[lvl]
            _, _, h, w = x.shape
            batch_size = affine_matrix.shape[0]
            split_x = regroup(x, record_len)
            split_score = regroup(score, record_len)
            out_b = []

            for b in range(batch_size):
                n_cav = int(record_len[b].item()) if torch.is_tensor(record_len[b]) else int(record_len[b])
                t_mat = affine_matrix[b, 0, :n_cav, :, :]
                feat_ego = warp_affine_simple(split_x[b], t_mat, (h, w), align_corners=align_corners)
                score_ego = warp_affine_simple(split_score[b], t_mat, (h, w), align_corners=align_corners)

                valid = score_ego > 0.0
                any_valid = valid.any(dim=0, keepdim=True)
                safe_logits = torch.where(valid, score_ego, torch.full_like(score_ego, -1e4))
                safe_logits = torch.where(any_valid, safe_logits, torch.zeros_like(safe_logits))
                w_base = torch.softmax(safe_logits, dim=0) * any_valid.to(score_ego.dtype)  # (N_cav, 1, H, W)

                if n_cav > 1:
                    g_src = gate_L0.to(device=w_base.device, dtype=w_base.dtype)
                    if g_src.shape[0] == 1 and n_cav - 1 > 1:
                        g_src = g_src.expand(n_cav - 1, -1, -1, -1)
                    if g_src.shape[0] != n_cav - 1:
                        raise ValueError(f"gate_L0 协作者维 {g_src.shape[0]} 与实际协作者数 {n_cav - 1} 不匹配！")
                    if g_src.shape[-2:] == (h, w):
                        g_l = g_src
                    else:
                        g_l = F.interpolate(g_src, size=(h, w), mode="area")
                    g_l = torch.clamp(g_l, min=0.0, max=1.0)

                    w_collab_new = w_base[1:] * g_l
                    reclaim_to_ego = torch.sum(w_base[1:] * (1.0 - g_l), dim=0, keepdim=True)
                    w_ego_new = torch.where(
                        (w_collab_new == 0.0).all(dim=0, keepdim=True),
                        any_valid.to(w_base.dtype),
                        w_base[0:1] + reclaim_to_ego,
                    )
                    w_new = torch.cat([w_ego_new, w_collab_new], dim=0)
                else:
                    w_new = w_base

                out_b.append(torch.sum(feat_ego * w_new, dim=0))

                if b == 0 and n_cav > 1:
                    level_diags[f"L{lvl}_mean_w_cav_base"] = float(w_base[1:].mean().item())
                    level_diags[f"L{lvl}_mean_w_cav_gated"] = float(w_new[1:].mean().item())
                    level_diags[f"L{lvl}_max_w_sum_err"] = float(
                        torch.abs(w_new.sum(dim=0, keepdim=True) - any_valid.to(w_new.dtype)).max().item()
                    )
                    if lvl == 0:
                        w_cav_l0_map = w_new[1:].detach()
                        for ci in range(1, n_cav):
                            v_ci = valid[ci : ci + 1]
                            if v_ci.any():
                                per_cav_w_base_l0.append(float(w_base[ci : ci + 1][v_ci].mean().item()))
                                per_cav_w_gated_l0.append(float(w_new[ci : ci + 1][v_ci].mean().item()))
                            else:
                                per_cav_w_base_l0.append(0.0)
                                per_cav_w_gated_l0.append(0.0)

            fused_feature_list.append(torch.stack(out_b, dim=0))

        fusion_diag = {
            "levels": level_diags,
            "w_cav_L0": w_cav_l0_map,
            "per_cav_w_base_L0": per_cav_w_base_l0,
            "per_cav_w_gated_L0": per_cav_w_gated_l0,
        }
        return fused_feature_list, fusion_diag

    return _custom_fuse


def build_whole_cav_reject_fuse_fn(reject_flags: Union[bool, Sequence[bool]], device: str = "cuda") -> Callable:
    """
    「整车降权」基线融合闭包（支持单协作者 bool 或多车队 bool 列表）：
    若第 c 辆协作者超过帧门限则该车整车权重清零回退本车，否则全一保留协同。
    """
    if isinstance(reject_flags, bool):
        flags = [reject_flags]
    else:
        flags = [bool(x) for x in reject_flags]
    gates = [
        torch.full((1, 1, PROTO_H, PROTO_W), 0.0 if rf else 1.0, device=device, dtype=torch.float32)
        for rf in flags
    ]
    return build_gated_fuse_fn(torch.cat(gates, dim=0))


@torch.no_grad()
def extract_single_cav_dev_and_signed(
    wrapper,
    feat_all: torch.Tensor,
    affine_matrix: torch.Tensor,
    agent_modality_list: Sequence[str],
    cam_valid_mask: torch.Tensor,
    p_ego_cls: torch.Tensor,
    cav_idx: int,
    k_tol: int = 5,
    return_pfused: bool = False,
):
    """
    面向多车队（N_cav >= 2）：仅抽取本车(0)与指定第 cav_idx 辆协作者组成 2 车对，
    计算在统一 ego 坐标系下的 (dev_4ch, d_drop_tol, d_add_tol, cov_ego_2d)。
    保证每辆协作者的异常审计相互解耦，不会被同帧其他恶意车污染！
    """
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

    dev = feat_all.device
    feat_pair = torch.cat([feat_all[0:1], feat_all[cav_idx : cav_idx + 1]], dim=0)
    rlen_pair = torch.tensor([2], device=dev, dtype=torch.long)
    idx_pair = torch.tensor([0, cav_idx], device=dev, dtype=torch.long)
    amat_pair = affine_matrix[:, idx_pair][:, :, idx_pair]
    mods_pair = [agent_modality_list[0], agent_modality_list[cav_idx]]

    tmat01 = amat_pair[0, 0, 1:2]
    ac = wrapper.model.pyramid_backbone.align_corners
    cov_local = cam_valid_mask[cav_idx : cav_idx + 1].float()
    cov_ego = warp_affine_simple(cov_local, tmat01, (PROTO_H, PROTO_W), align_corners=ac) > 0.5
    cov_f = cov_ego.float()

    out_pair = wrapper.forward_from_protocol(feat_pair, rlen_pair, amat_pair, mods_pair)
    cos_maps = []
    for lvl in range(3):
        f_lvl = out_pair["pyramid_feats"][lvl]
        z_ego = f_lvl[0:1]
        hl, wl = z_ego.shape[-2:]
        z_cav = warp_affine_simple(f_lvl[1:2], tmat01, (hl, wl), align_corners=ac)
        c_raw = (1.0 - F.cosine_similarity(z_ego, z_cav, dim=1, eps=1e-6)).unsqueeze(1)
        if lvl == 0:
            c_256 = _masked_avg_pool3x3(c_raw, cov_ego)
        else:
            c_256 = F.interpolate(c_raw, size=(PROTO_H, PROTO_W), mode="bilinear", align_corners=False) * cov_f
        cos_maps.append(c_256)

    p_pair_cls = torch.sigmoid(out_pair["cls_preds"]).max(dim=1, keepdim=True)[0]
    d_sem = torch.abs(p_ego_cls - p_pair_cls) * cov_f
    dev_4ch = torch.cat(cos_maps + [d_sem], dim=1)[0]
    d_drop_tol, d_add_tol = compute_shift_tolerant_semantic_maps(p_ego_cls, p_pair_cls, cov_ego, k_tol=k_tol)
    if return_pfused:
        return dev_4ch, d_drop_tol, d_add_tol, cov_ego[0, 0], p_pair_cls
    return dev_4ch, d_drop_tol, d_add_tol, cov_ego[0, 0]


class BUDFusionHEALAuditor:
    """
    BUD-Fusion HEAL 审计器与两级软门控生成器（冻结升级版）。

    同时支持：
    - 加载外部冻结基础标定 JSON 与局部门控 JSON，或注入内存级配置字典供 Fit 网格调参；
    - 可见性分档余弦门限 cos_tau_by_vis 与 gamma_cos_gate；
    - 整车级高门限回退 frame_high_tau（score_t4 > frame_high_tau[mod] 时整车回退 gate=0）。
    """

    def __init__(
        self,
        base_calib_path: str,
        gating_calib_path: Optional[str] = None,
        gating_override: Optional[dict] = None,
    ):
        with open(base_calib_path, "r", encoding="utf-8") as f:
            base_calib = json.load(f)
        self.stats = base_calib["channel_tier_stats"]
        self.main_cfg = dict(base_calib["frozen_methods"]["main_ms_cos_clear_sem"]["cfg"])
        # 强制使用稳健 div_p95 帧归一化（不使用窄分母 med_p95）
        self.main_cfg["frame_norm_mode"] = "div_p95"

        if gating_override is not None:
            self.gating_calib = gating_override
        elif gating_calib_path is not None and os.path.exists(gating_calib_path):
            with open(gating_calib_path, "r", encoding="utf-8") as f:
                self.gating_calib = json.load(f)
        else:
            self.gating_calib = None

    @torch.no_grad()
    def audit_dev_maps(
        self,
        dev_4ch: torch.Tensor,
        cov_2d: torch.Tensor,
        d_idx: torch.Tensor,
        v_idx: torch.Tensor,
        mod: str,
        build_gate: bool = True,
        d_drop_tol: Optional[torch.Tensor] = None,
        d_add_tol: Optional[torch.Tensor] = None,
        enable_whole_fallback: bool = True,
        collab_fg_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        对单辆协作者的 4 通道偏差图同步输出：
        - Tier 2 (仅传感器) 与 Tier 4 (传感器+5距离+可见性) 的空间异常图、多尺度余弦图与 div_p95 帧分数；
        - 两级门控判定：若启用整车高门限回退且 score_t4 > frame_high_tau[mod]，整车 gate_L0 置 0；
          否则按标定模式（可见性分档有向语义软门控或标量死区）生成局部 gate_L0。
        """
        s_map_t2, _ = _compute_spatial_anomaly_and_cos(
            dev_4ch, cov_2d, d_idx, v_idx, mod, self.stats, "tier2_sensor_only", self.main_cfg
        )
        s_map_t4, cos_comb_t4 = _compute_spatial_anomaly_and_cos(
            dev_4ch, cov_2d, d_idx, v_idx, mod, self.stats, "tier4_sensor_dist_vis", self.main_cfg
        )

        k = int(self.main_cfg["top_k"])
        vals_t2 = s_map_t2[cov_2d]
        vals_t4 = s_map_t4[cov_2d]
        raw_t2 = float(torch.topk(vals_t2, min(k, vals_t2.numel())).values.mean().item()) if vals_t2.numel() else 0.0
        raw_t4 = float(torch.topk(vals_t4, min(k, vals_t4.numel())).values.mean().item()) if vals_t4.numel() else 0.0

        score_t2 = raw_t2
        score_t4 = raw_t4
        gate_l0 = None
        whole_fallback_triggered = False

        if self.gating_calib is not None:
            p95_t2 = max(float(self.gating_calib["frame_div_p95"]["tier2_sensor_only"][mod]), 1e-4)
            p95_t4 = max(float(self.gating_calib["frame_div_p95"]["tier4_sensor_dist_vis"][mod]), 1e-4)
            score_t2 = raw_t2 / p95_t2
            score_t4 = raw_t4 / p95_t4

            if build_gate:
                fh_cfg = self.gating_calib.get("frame_high_tau", None)
                if enable_whole_fallback and fh_cfg is not None and mod in fh_cfg:
                    tau_high = float(fh_cfg[mod])
                    if score_t4 > tau_high:
                        whole_fallback_triggered = True
                        gate_l0 = torch.zeros(
                            (1, 1, PROTO_H, PROTO_W), device=dev_4ch.device, dtype=dev_4ch.dtype
                        )

                if gate_l0 is None:
                    gcfg = self.gating_calib["gating_params"]
                    if gcfg.get("mode") in ("directional_shift_tol", "directional_vis_adaptive", "directional_vis_fg_bg"):
                        if d_drop_tol is None or d_add_tol is None:
                            raise ValueError("有向语义门控模式必须传入 d_drop_tol 与 d_add_tol！")
                        if "cos_tau_by_vis_bg" in gcfg and mod in gcfg["cos_tau_by_vis_bg"]:
                            cos_tau_bg_arg = gcfg["cos_tau_by_vis_bg"][mod]
                            cos_tau_fg_arg = gcfg.get("cos_tau_by_vis_fg", {}).get(mod)
                        elif "cos_tau_by_vis" in gcfg and mod in gcfg["cos_tau_by_vis"]:
                            cos_tau_bg_arg = gcfg["cos_tau_by_vis"][mod]
                            cos_tau_fg_arg = None
                        else:
                            cos_tau_bg_arg = float(gcfg["cos_tau"][mod])
                            cos_tau_fg_arg = None
                        gamma_cos_gate = gcfg.get("gamma_cos_gate", (1.0, 1.0, 1.0))
                        gate_l0 = build_directional_gate_map(
                            cos_comb=cos_comb_t4,
                            d_drop_tol=d_drop_tol,
                            d_add_tol=d_add_tol,
                            cov_2d=cov_2d,
                            v_idx=v_idx,
                            cos_tau=cos_tau_bg_arg,
                            drop_tau=float(gcfg["drop_tau"]),
                            add_tau=float(gcfg["add_tau"]),
                            w_drop=float(gcfg["w_drop"]),
                            w_add=float(gcfg["w_add"]),
                            cos_co_ratio=float(gcfg["cos_co_ratio"]),
                            gamma_add=gcfg["gamma_add"],
                            gamma_cos_gate=gamma_cos_gate,
                            r_dil=int(gcfg["r_dil"]),
                            beta=float(gcfg["beta"]),
                            collab_fg_mask=collab_fg_mask,
                            cos_tau_fg=cos_tau_fg_arg,
                        )
                    else:
                        tau_m = float(gcfg["tau_loc"][mod])
                        gate_l0 = build_gate_map_from_smap(
                            s_map_t4, cov_2d, tau_loc=tau_m, r_dil=int(gcfg["r_dil"]), beta=float(gcfg["beta"])
                        )

        return {
            "s_map_t2": s_map_t2,
            "s_map_t4": s_map_t4,
            "cos_comb_t4": cos_comb_t4,
            "raw_t2": raw_t2,
            "raw_t4": raw_t4,
            "score_t2": score_t2,
            "score_t4": score_t4,
            "gate_L0": gate_l0,
            "whole_fallback": whole_fallback_triggered,
        }
