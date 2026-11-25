# -*- coding: utf-8 -*-
"""
步骤 ⑤ 公开基线封装：LUCIA (USENIX Security 2025) 与 ROBOSAC (ICCV 2023) 接入 HEAL m1+m2 多尺度底座。

严格忠实官方源码实现：
1. LUCIA（对照官方 WiSeR-Lab__SOMBRA_LUCIA/defense/defender.py）：
   - 对齐至本车坐标系后的协议特征经 avg_pool2d(kernel_size=32) 空间压缩；
   - 按通道+空间做全局 L2 归一化，计算两两 L1 距离之和（N=2 时 l1[0]=0, l1[1]=||u1-u0||_1；
     N>=3 时累加全部无序对并做本车保护 l1[0] = min(l1)）；
   - trust = 1 - softmax(l1_scores)，在 HEAL 多尺度融合权重上乘以 trust 并重归一化。
2. ROBOSAC（对照官方 coperception__ROBOSAC/coperception/tools/det/robosac.py）：
   - 严格仅在总车数 N >= 3（即协作车 >= 2）场景启用；单协作车（N == 2）时显式跳过（退化为整车取舍）；
   - 以本车 Ego-Only 3D 预测框集合为参考，在 step_budget 内按固定种子抽样协作子集，
     用 2D BEV IoU (box_matching_thresh=0.3) 计算 Jaccard 系数验证共识，达成共识即采用该子集，
     耗尽预算未达成共识则安全回退 Ego-Only。
"""

import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def cp_defense_lucia_heal(
    aligned_features: torch.Tensor,
    compression_ratio: int = 32,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    官方 LUCIA 核心公式（与 WiSeR-Lab__SOMBRA_LUCIA/defense/defender.py 逐行一致）：
    输入 aligned_features: (N, C, H, W)，第 0 项为可信本车 ego，1..N-1 为已对齐到本车系的协作者。
    返回 (trust_scores: (N,), l1_scores_raw: (N,))。
    """
    feats = aligned_features.detach()
    n = int(feats.shape[0])
    if n < 2:
        return torch.ones((n,), device=feats.device, dtype=feats.dtype), torch.zeros((n,), device=feats.device, dtype=feats.dtype)

    pooled = F.avg_pool2d(feats, kernel_size=int(compression_ratio))
    norms = torch.norm(pooled, p=2, dim=(1, 2, 3), keepdim=True)
    normalized = pooled / torch.clamp(norms, min=eps)

    l1_scores = torch.zeros((n,), device=feats.device, dtype=feats.dtype)
    if n == 2:
        l1_scores[1] = torch.sum(torch.abs(normalized[1] - normalized[0]))
        l1_scores[0] = 0.0
    else:
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.sum(torch.abs(normalized[i] - normalized[j]))
                l1_scores[i] = l1_scores[i] + dist
                l1_scores[j] = l1_scores[j] + dist

    l1_scores[0] = torch.min(l1_scores)
    trust = 1.0 - F.softmax(l1_scores, dim=0)
    return trust, l1_scores


def build_lucia_fuse_fn(
    compression_ratio: int = 32,
    trust_override: Optional[torch.Tensor] = None,
) -> Callable:
    """
    将 LUCIA 信任得分嵌入 HEAL 金字塔多尺度融合：
    在 L0 将对齐后的各车特征送入 cp_defense_lucia_heal 计算 trust (N,)，
    并将所有尺度融合权重按 trust 乘权并重归一化（保持权重和为 1 且可记录各车实际融合权重）。
    """
    from opencood.models.fuse_modules.fusion_in_one import regroup
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

    def _lucia_fuse(
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
        cached_trust = {}
        diag_trust = []
        per_cav_w_base_l0 = []
        per_cav_w_lucia_l0 = []

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

                if lvl == 0:
                    if trust_override is not None:
                        trust = trust_override.to(device=feat_ego.device, dtype=feat_ego.dtype)
                    else:
                        trust, _ = cp_defense_lucia_heal(feat_ego, compression_ratio=compression_ratio)
                    cached_trust[b] = trust
                    if b == 0:
                        diag_trust = [float(v) for v in trust.detach().cpu().tolist()]
                else:
                    trust = cached_trust[b]

                valid = score_ego > 0.0
                any_valid = valid.any(dim=0, keepdim=True)
                safe_logits = torch.where(valid, score_ego, torch.full_like(score_ego, -1e4))
                safe_logits = torch.where(any_valid, safe_logits, torch.zeros_like(safe_logits))
                w_base = torch.softmax(safe_logits, dim=0) * any_valid.to(score_ego.dtype)

                if n_cav > 1:
                    t_view = trust.view(n_cav, 1, 1, 1).to(dtype=w_base.dtype)
                    w_mod = w_base * t_view
                    w_sum = torch.clamp(w_mod.sum(dim=0, keepdim=True), min=1e-8)
                    w_new = (w_mod / w_sum) * any_valid.to(w_base.dtype)
                else:
                    w_new = w_base

                out_b.append(torch.sum(feat_ego * w_new, dim=0))

                if b == 0 and lvl == 0 and n_cav > 1:
                    for ci in range(1, n_cav):
                        v_ci = valid[ci : ci + 1]
                        if v_ci.any():
                            per_cav_w_base_l0.append(float(w_base[ci : ci + 1][v_ci].mean().item()))
                            per_cav_w_lucia_l0.append(float(w_new[ci : ci + 1][v_ci].mean().item()))
                        else:
                            per_cav_w_base_l0.append(0.0)
                            per_cav_w_lucia_l0.append(0.0)

            fused_feature_list.append(torch.stack(out_b, dim=0))

        fusion_diag = {
            "lucia_trust": diag_trust,
            "per_cav_w_base_L0": per_cav_w_base_l0,
            "per_cav_w_gated_L0": per_cav_w_lucia_l0,
        }
        return fused_feature_list, fusion_diag

    return _lucia_fuse


def compute_box_set_jaccard_2d(
    boxes_a: Optional[np.ndarray],
    boxes_b: Optional[np.ndarray],
    iou_thresh: float = 0.3,
) -> float:
    """
    计算两个 3D 检测框集合在 BEV 平面上的 Jaccard 共识系数（对齐 ROBOSAC associate_2_detections）：
    若两组均为空返回 1.0；仅一方为空返回 0.0；
    否则使用轴对齐/多边形 BEV IoU 贪心匹配，返回 TP / (|A| + |B| - TP)。
    boxes_x 形状支持 (M, 8, 3) 角点格式（OpenCOOD/HEAL 标准输出）或 (M, 4) [x1,y1,x2,y2]。
    """
    na = 0 if boxes_a is None else int(len(boxes_a))
    nb = 0 if boxes_b is None else int(len(boxes_b))
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0

    def _to_aabb(b_arr: np.ndarray) -> np.ndarray:
        if b_arr.ndim == 3 and b_arr.shape[1] >= 4:
            xs = b_arr[:, :4, 0]
            ys = b_arr[:, :4, 1]
            return np.stack([xs.min(axis=1), ys.min(axis=1), xs.max(axis=1), ys.max(axis=1)], axis=1)
        return b_arr[:, :4].astype(np.float64)

    # 优先使用精确多边形 IoU（若 OpenCOOD common_utils 可用），否则纯 CPU 下退回快速 AABB IoU
    try:
        from opencood.utils import common_utils
        if boxes_a.ndim == 3 and boxes_b.ndim == 3:
            pa = list(common_utils.convert_format(boxes_a))
            pb = list(common_utils.convert_format(boxes_b))
            rem = list(pb)
            tp = 0
            for poly_a in pa:
                if not rem:
                    break
                ious = common_utils.compute_iou(poly_a, rem)
                best_j = int(np.argmax(ious))
                if float(ious[best_j]) >= float(iou_thresh):
                    tp += 1
                    rem.pop(best_j)
            denom = na + nb - tp
            return float(tp) / float(denom) if denom > 0 else 0.0
    except Exception:
        pass

    aa = _to_aabb(boxes_a)
    bb = _to_aabb(boxes_b)
    x1 = np.maximum(aa[:, None, 0], bb[None, :, 0])
    y1 = np.maximum(aa[:, None, 1], bb[None, :, 1])
    x2 = np.minimum(aa[:, None, 2], bb[None, :, 2])
    y2 = np.minimum(aa[:, None, 3], bb[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, aa[:, 2] - aa[:, 0]) * np.maximum(0.0, aa[:, 3] - aa[:, 1])
    area_b = np.maximum(0.0, bb[:, 2] - bb[:, 0]) * np.maximum(0.0, bb[:, 3] - bb[:, 1])
    iou_mat = inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-8)

    used_b = set()
    tp = 0
    for i in range(na):
        best_j = -1
        best_v = float(iou_thresh)
        for j in range(nb):
            if j in used_b:
                continue
            if float(iou_mat[i, j]) >= best_v:
                best_v = float(iou_mat[i, j])
                best_j = j
        if best_j >= 0:
            used_b.add(best_j)
            tp += 1
    denom = na + nb - tp
    return float(tp) / float(denom) if denom > 0 else 0.0


class ROBOSACHEALDefender:
    """
    ROBOSAC (ICCV 2023) 在 HEAL 多车队上的忠实实现：
    - 严格只在总车数 N >= 3（协作者数 >= 2）时运行；
    - 给定帧内协作车集合 [1..N-1]，采用自适应/固定采样集合大小 k，在 step_budget 步内
      通过比较子集融合框与 Ego-Only 框的 Jaccard 系数寻找无攻击共识子集。
    """

    def __init__(
        self,
        box_matching_thresh: float = 0.3,
        jaccard_thresh: float = 0.3,
        step_budget: int = 5,
        seed: int = 20261120,
    ):
        self.box_matching_thresh = float(box_matching_thresh)
        self.jaccard_thresh = float(jaccard_thresh)
        self.step_budget = int(step_budget)
        self.seed = int(seed)

    @torch.no_grad()
    def run_frame(
        self,
        wrapper,
        feat_all: torch.Tensor,
        record_len: torch.Tensor,
        affine_matrix: torch.Tensor,
        agent_modality_list: Sequence[str],
        batch_post: dict,
        dataset,
        ego_boxes_np: Optional[np.ndarray],
        ego_pb,
        ego_ps,
        ego_gb,
        frame_seed: int = 0,
        assumed_attackers: int = 1,
    ) -> dict:
        """
        执行单帧 ROBOSAC 共识采样防御：
        - 若总车数 N < 3，返回 skipped=True（按实验设计单协作车不对比 ROBOSAC）；
        - 返回 {"skipped", "consensus_found", "selected_collabs", "steps_used", "jaccard", "pb", "ps", "gb"}。
        """
        n_total = int(feat_all.shape[0])
        n_collab = n_total - 1
        if n_total < 3:
            return {
                "skipped": True,
                "reason": "ROBOSAC 仅适用于 >=3 车（>=2 协作车）场景",
                "consensus_found": False,
                "selected_collabs": [],
                "steps_used": 0,
                "jaccard": 0.0,
                "pb": ego_pb,
                "ps": ego_ps,
                "gb": ego_gb,
            }

        rng = random.Random(self.seed + int(frame_seed) * 1009 + n_total * 17)
        all_collabs = list(range(1, n_total))
        # 先探测全协作集合（若全队干净一步即可通过），未通过则按目标无毒子集大小 k 采样
        k_sub = max(1, n_collab - int(assumed_attackers))
        candidates: List[List[int]] = [list(all_collabs)]

        # 构造最多 step_budget 个不重复子集抽样（先尝试 k_sub，再按需降维到 1）
        seen = {tuple(all_collabs)}
        tries = 0
        while len(candidates) < self.step_budget and tries < self.step_budget * 8:
            tries += 1
            sub = sorted(rng.sample(all_collabs, k=k_sub))
            if tuple(sub) not in seen:
                seen.add(tuple(sub))
                candidates.append(sub)
        if len(candidates) < self.step_budget and k_sub > 1:
            for c in all_collabs:
                if len(candidates) >= self.step_budget:
                    break
                if (c,) not in seen:
                    seen.add((c,))
                    candidates.append([c])

        def _eval_subset(sub_collabs: List[int]):
            idx_list = [0] + list(sub_collabs)
            idx_t = torch.tensor(idx_list, device=feat_all.device, dtype=torch.long)
            f_sub = feat_all[idx_t]
            r_sub = torch.tensor([len(idx_list)], device=feat_all.device, dtype=record_len.dtype)
            a_sub = affine_matrix[:, idx_t][:, :, idx_t]
            m_sub = [agent_modality_list[i] for i in idx_list]
            out_sub = wrapper.forward_from_protocol(f_sub, r_sub, a_sub, m_sub)
            pb, ps, gb = wrapper.decode_boxes(batch_post, out_sub, dataset)
            pb_np = pb.detach().cpu().numpy() if (pb is not None and len(pb) > 0) else None
            jac = compute_box_set_jaccard_2d(ego_boxes_np, pb_np, iou_thresh=self.box_matching_thresh)
            return pb, ps, gb, jac

        steps_used = 0
        for sub in candidates[: self.step_budget]:
            steps_used += 1
            pb, ps, gb, jac = _eval_subset(sub)
            if jac >= self.jaccard_thresh:
                return {
                    "skipped": False,
                    "consensus_found": True,
                    "selected_collabs": sub,
                    "steps_used": steps_used,
                    "jaccard": float(jac),
                    "pb": pb,
                    "ps": ps,
                    "gb": gb,
                }

        # 耗尽预算未达成共识，安全回退至 Ego-Only
        return {
            "skipped": False,
            "consensus_found": False,
            "selected_collabs": [],
            "steps_used": steps_used,
            "jaccard": 0.0,
            "pb": ego_pb,
            "ps": ego_ps,
            "gb": ego_gb,
        }
