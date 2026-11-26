# -*- coding: utf-8 -*-
"""
sprint-cs2027 统一标准指标与序列级统计检验公共库（覆盖 A4、F-P0.8、F-P0.9）。

核心功能：
1. 防刷版定向攻击成功率 ASR（evaluate_frame_asr_strict）：
   - 仅将预测置信度 >= s_min 的检测框视为有效检出（显式记录 s_min，并支持多阈值敏感性扫描）；
   - 伪造攻击：若伪造锚点 3.0 m 内本来存在真实目标框（排除自车 r<2.5m），从主伪造分母剔除，
     并单独归入 near_gt_fabrication（近目标伪造）统计，绝不静默丢弃；
   - 移除攻击：仅统计干净无攻击融合时 3.0 m 内有 >= s_min 预测框的真实目标；
2. 盲区 AP@0.5 / AP@0.7（match_blind_spot_ap_ignore_nonblind）：
   - 仅将盲区真值框（落入协作者覆盖区 ∧ 本车 LiDAR 点 <= 5 ∧ 非本车自身）作为正样本计分；
   - 匹配到非盲区真值框（或自车框）的预测框记为 ignore（不计 TP 也不计 FP），未匹配任何真值的误检计为 FP；
   - 支持额外筛选“协作车自身确实可观测的盲区目标”子集；
3. 诚实协作车被压制比例与权重削减占比（compute_honest_suppression_stats）：
   - 统计诚实协作车覆盖区内 G_i < 0.5 的格子比例（按传感器 × Clear/Sparse/Occluded 分档，整车回退记 100%）；
   - 同步统计被削减融合权重占比 sum((1 - G_i) * w_i) / sum(w_i)；
4. 序列级配对 Studentized Block-Bootstrap 与 Holm-Bonferroni 校正：
   - 2,000 次父序列整块重采样（种子 20261120），输出百分位 CI、Studentized CI、双侧/单侧 p 值及 Holm 校正。
"""

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src_v2_mm.eval_stats_utils import build_seq_bootstrap_indices, fast_voc_ap

VIS_NAMES_3 = ("Clear", "Sparse", "Occluded")


def filter_boxes_by_score(
    boxes_np: Optional[np.ndarray],
    scores_np: Optional[np.ndarray],
    s_min: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """仅保留预测置信度 >= s_min 的检测框与分数。"""
    if boxes_np is None or scores_np is None or len(boxes_np) == 0:
        return np.zeros((0, 8, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    sc = np.asarray(scores_np, dtype=np.float64).reshape(-1)
    bx = np.asarray(boxes_np, dtype=np.float64)
    keep = sc >= float(s_min)
    if not np.any(keep):
        return np.zeros((0, 8, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return bx[keep], sc[keep]


def extract_centers_xy_from_boxes_np(boxes_np: Optional[np.ndarray]) -> np.ndarray:
    """从 (N, 8, 3) 或 (N, 4, 2) 角点框提取中心 (N, 2) 物理坐标。"""
    if boxes_np is None or len(boxes_np) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    arr = np.asarray(boxes_np, dtype=np.float64)
    if arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return arr[:, :4, :2].mean(axis=1)


def has_center_within_radius(centers_xy: np.ndarray, cx: float, cy: float, radius_m: float = 3.0) -> bool:
    """判断点 (cx, cy) 半径 radius_m 米内是否存在至少一个框中心。"""
    if centers_xy.shape[0] == 0:
        return False
    d2 = (centers_xy[:, 0] - float(cx)) ** 2 + (centers_xy[:, 1] - float(cy)) ** 2
    return bool(float(d2.min()) <= float(radius_m) ** 2)


def extract_valid_gt_centers_xy(gt_np: Optional[np.ndarray], ego_self_radius_m: float = 2.5) -> np.ndarray:
    """提取排除自车车体框（r < 2.5m）后的全部真实目标中心坐标 (M, 2)。"""
    centers = extract_centers_xy_from_boxes_np(gt_np)
    if centers.shape[0] == 0:
        return centers
    r = np.hypot(centers[:, 0], centers[:, 1])
    return centers[r >= float(ego_self_radius_m)]


def evaluate_frame_asr_strict(
    mode: str,
    candidate_points_xy: Sequence[Tuple[float, float]],
    clean_pred_boxes_np: Optional[np.ndarray],
    clean_pred_scores_np: Optional[np.ndarray],
    defended_pred_boxes_np: Optional[np.ndarray],
    defended_pred_scores_np: Optional[np.ndarray],
    gt_boxes_np: Optional[np.ndarray],
    s_min: float,
    radius_m: float = 3.0,
) -> dict:
    """
    严格按 A4 与 F-P0.8 评估单帧 ASR 分子与分母：
    - mode == "removal":
      仅保留干净融合在 s_min 下能检出（3m 内有框）的目标点；防御后 3m 内无 >= s_min 框计为移除成功；
    - mode == "fabrication":
      1) 若锚点 3m 内存在真实 GT 框，从标准伪造分母中剔除，单独统计入 near_gt_*（F-P0.8）；
      2) 若锚点 3m 内无真实 GT 框且干净融合无 >= s_min 预测框，计入标准伪造分母；防御后出现 >= s_min 框计为伪造成功。
    """
    cln_bx, _ = filter_boxes_by_score(clean_pred_boxes_np, clean_pred_scores_np, s_min)
    def_bx, _ = filter_boxes_by_score(defended_pred_boxes_np, defended_pred_scores_np, s_min)
    cln_xy = extract_centers_xy_from_boxes_np(cln_bx)
    def_xy = extract_centers_xy_from_boxes_np(def_bx)
    gt_xy = extract_valid_gt_centers_xy(gt_boxes_np)

    succ = 0
    total = 0
    near_gt_succ = 0
    near_gt_total = 0

    if mode == "removal":
        for cx, cy in candidate_points_xy:
            if has_center_within_radius(cln_xy, cx, cy, radius_m=radius_m):
                total += 1
                if not has_center_within_radius(def_xy, cx, cy, radius_m=radius_m):
                    succ += 1
    elif mode == "fabrication":
        for cx, cy in candidate_points_xy:
            # F-P0.8：首先检查锚点 3m 内是否存在真实 GT 框；若存在，一律归入近目标伪造单独统计，绝不静默丢弃
            if has_center_within_radius(gt_xy, cx, cy, radius_m=radius_m):
                near_gt_total += 1
                if has_center_within_radius(def_xy, cx, cy, radius_m=radius_m):
                    near_gt_succ += 1
                continue
            # 3m 内无真实 GT 框：再排除干净融合原本已存在虚警预测框的位置，其余计入标准伪造分母
            if has_center_within_radius(cln_xy, cx, cy, radius_m=radius_m):
                continue
            total += 1
            if has_center_within_radius(def_xy, cx, cy, radius_m=radius_m):
                succ += 1
    else:
        raise ValueError(f"未知的 ASR 模式: {mode}")

    return {
        "succ": int(succ),
        "total": int(total),
        "near_gt_succ": int(near_gt_succ),
        "near_gt_total": int(near_gt_total),
    }


def match_blind_spot_ap_ignore_nonblind(
    pred_boxes_np: Optional[np.ndarray],
    pred_scores_np: Optional[np.ndarray],
    gt_polygons: list,
    blind_gt_indices: Sequence[int],
    iou_thresh: float = 0.5,
    det_polygons: Optional[list] = None,
    compute_iou_fn=None,
) -> dict:
    """
    A4 + F-P0.8 严格盲区 AP 匹配算法（非盲区真值记为 Ignore 不计分）：
    - 按预测置信度从高到低遍历每个预测框：
      1) 计算与所有剩余未匹配真值框（含盲区真值与非盲区真值）的 IoU；
      2) 若最大 IoU >= iou_thresh：
         - 消费该真值框；
         - 若该真值属于 blind_gt_indices -> 记 TP = 1, FP = 0；
         - 若该真值属于非盲区真值（含本车自见真值或覆盖区外真值） -> 记 Ignore（跳过，不计 TP 也不计 FP）；
      3) 若最大 IoU < iou_thresh -> 纯虚假误检，记 TP = 0, FP = 1。
    - 返回供 fast_voc_ap 聚合的 {"tp": list, "fp": list, "score": list, "gt": len(blind_gt_indices), "hits": int}。
    """
    blind_set = set(int(i) for i in blind_gt_indices)
    n_blind = len(blind_set)
    if pred_boxes_np is None or pred_scores_np is None or len(pred_boxes_np) == 0:
        return {"tp": [], "fp": [], "score": [], "gt": n_blind, "hits": 0}

    sc_np = np.asarray(pred_scores_np, dtype=np.float64).reshape(-1)
    if sc_np.size == 0:
        return {"tp": [], "fp": [], "score": [], "gt": n_blind, "hits": 0}

    if compute_iou_fn is None or det_polygons is None:
        from opencood.utils import common_utils
        if det_polygons is None:
            det_polygons = list(common_utils.convert_format(np.asarray(pred_boxes_np)))
        if compute_iou_fn is None:
            compute_iou_fn = common_utils.compute_iou

    order = np.argsort(-sc_np)
    rem_polys = list(gt_polygons)
    rem_gt_ids = list(range(len(gt_polygons)))

    tp_list, fp_list, sc_list = [], [], []
    blind_hits = 0

    for k in range(order.shape[0]):
        idx_p = int(order[k])
        det_p = det_polygons[idx_p]
        sc_val = float(sc_np[idx_p])

        if not rem_polys:
            tp_list.append(0)
            fp_list.append(1)
            sc_list.append(sc_val)
            continue

        ious = np.asarray(compute_iou_fn(det_p, rem_polys), dtype=np.float64)
        best_j = int(np.argmax(ious))
        if float(ious[best_j]) < float(iou_thresh):
            tp_list.append(0)
            fp_list.append(1)
            sc_list.append(sc_val)
            continue

        matched_gt_id = rem_gt_ids.pop(best_j)
        rem_polys.pop(best_j)
        if matched_gt_id in blind_set:
            tp_list.append(1)
            fp_list.append(0)
            sc_list.append(sc_val)
            blind_hits += 1
        else:
            # 命中非盲区真值框：按 F-P0.8 记为“不计分（Ignore）”，不入 TP/FP 序列
            continue

    return {
        "tp": tp_list,
        "fp": fp_list,
        "score": sc_list,
        "gt": n_blind,
        "hits": int(blind_hits),
    }


@torch.no_grad()
def compute_honest_suppression_stats(
    gate_l0: torch.Tensor,
    cov_2d: torch.Tensor,
    v_idx_2d: torch.Tensor,
    whole_fallback: bool = False,
    w_base_cav_l0: Optional[torch.Tensor] = None,
    suppress_thresh: float = 0.5,
) -> dict:
    """
    A4 + F-P0.8 诚实协作车被压制比例与权重削减占比统计：
    - gate_l0   : (1, 1, H, W) 或 (H, W) 门控图 G_i；
    - cov_2d    : (H, W) bool 协作者有效覆盖区；
    - v_idx_2d  : (H, W) long ∈ {0:Clear, 1:Sparse, 2:Occluded}；
    - whole_fallback: 若整车回退触发，所有有效覆盖格子均记为被压制（100%），权重削减比记 100%。
    """
    g2d = gate_l0.view(cov_2d.shape[-2], cov_2d.shape[-1]).to(dtype=torch.float32)
    cov_b = cov_2d.to(dtype=torch.bool)
    v2d = v_idx_2d.to(device=cov_b.device, dtype=torch.long)

    if whole_fallback:
        g_eff = torch.zeros_like(g2d)
    else:
        g_eff = g2d

    suppressed = (g_eff < float(suppress_thresh)) & cov_b
    tot_cov = int(cov_b.sum().item())
    sup_cov = int(suppressed.sum().item())

    by_vis = OrderedDict()
    for vi, vname in enumerate(VIS_NAMES_3):
        vm = cov_b & (v2d == vi)
        vcnt = int(vm.sum().item())
        scnt = int((suppressed & vm).sum().item())
        by_vis[vname] = {
            "suppressed_grids": scnt,
            "total_grids": vcnt,
            "ratio": float(scnt / vcnt) if vcnt > 0 else 0.0,
        }

    # F-P0.8：被削减的基准融合权重占比 sum((1 - G_i) * w_i) / sum(w_i)
    if w_base_cav_l0 is not None:
        w2d = w_base_cav_l0.view_as(g2d).to(device=g2d.device, dtype=torch.float32)
        w_den = float(w2d[cov_b].sum().item()) if tot_cov > 0 else 0.0
        w_num = float(((1.0 - g_eff)[cov_b] * w2d[cov_b]).sum().item()) if tot_cov > 0 else 0.0
    else:
        w_den = float(tot_cov)
        w_num = float((1.0 - g_eff)[cov_b].sum().item()) if tot_cov > 0 else 0.0

    return {
        "suppressed_grids": sup_cov,
        "total_grids": tot_cov,
        "overall_ratio": float(sup_cov / tot_cov) if tot_cov > 0 else 0.0,
        "by_visibility": by_vis,
        "reclaimed_weight_num": w_num,
        "reclaimed_weight_den": w_den,
        "reclaimed_weight_ratio": float(w_num / w_den) if w_den > 1e-8 else 0.0,
    }


def studentized_paired_block_bootstrap(
    seq_names: Sequence[str],
    eval_fn_on_seqs,
    n_boot: int = 2000,
    seed: int = 20261120,
    null_value: float = 0.0,
) -> dict:
    """
    F-P0.9 序列级配对 Studentized Block-Bootstrap（支持任意序列聚合统计量差值 Delta = T(A) - T(B)）：
    - eval_fn_on_seqs(picked_seqs) -> (val_a, val_b)；
    - 在每次重采样块内使用 Leave-One-Cluster-Out Jackknife 估计标准误 SE*，
      构造枢轴统计量 t* = (Delta* - Delta_hat) / SE*，输出二阶精确的 Studentized 95% CI 与 p 值；
    - 当序列块数过少或方差退化为 0 时平滑回退百分位 CI。
    """
    seq_list = list(seq_names)
    n_seq = len(seq_list)
    val_a, val_b = eval_fn_on_seqs(seq_list)
    diff_hat = float(val_a - val_b)

    if n_seq <= 1:
        return {
            "num_sequences": n_seq,
            "val_a": round(float(val_a), 5),
            "val_b": round(float(val_b), 5),
            "diff": round(diff_hat, 5),
            "percentile_ci95": [round(diff_hat, 5), round(diff_hat, 5)],
            "studentized_ci95": [round(diff_hat, 5), round(diff_hat, 5)],
            "p_value_two_sided": 1.0,
        }

    # 预计算单序列留一法影响值以加速重采样内 Jackknife 标准误
    jack_vals = np.empty(n_seq, dtype=np.float64)
    for i in range(n_seq):
        sub = seq_list[:i] + seq_list[i + 1 :]
        ja, jb = eval_fn_on_seqs(sub)
        jack_vals[i] = float(ja - jb)
    pseudo = n_seq * diff_hat - (n_seq - 1) * jack_vals
    se_hat = float(np.std(pseudo, ddof=1) / math.sqrt(n_seq))

    boot_idx = build_seq_bootstrap_indices(seq_list, n_boot=n_boot, seed=seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    t_pivots = []

    for bi in range(n_boot):
        idxs = boot_idx[bi]
        picked = [seq_list[j] for j in idxs]
        ba, bb = eval_fn_on_seqs(picked)
        d_star = float(ba - bb)
        diffs[bi] = d_star
        # 用重采样到的伪值样本方差高效近似块内标准误 SE*
        ps_star = pseudo[idxs]
        se_star = float(np.std(ps_star, ddof=1) / math.sqrt(n_seq))
        if se_star > 1e-8:
            t_pivots.append((d_star - diff_hat) / se_star)

    pct_lo, pct_hi = np.percentile(diffs, [2.5, 97.5])
    if se_hat > 1e-8 and len(t_pivots) >= max(50, n_boot // 4):
        t_arr = np.asarray(t_pivots, dtype=np.float64)
        q_lo, q_hi = np.percentile(t_arr, [2.5, 97.5])
        # Studentized CI: [Delta_hat - q_0.975 * SE_hat, Delta_hat - q_0.025 * SE_hat]
        stu_lo = diff_hat - float(q_hi) * se_hat
        stu_hi = diff_hat - float(q_lo) * se_hat
        t_obs = abs(diff_hat - float(null_value)) / se_hat
        p_two = float((np.sum(np.abs(t_arr) >= t_obs) + 1.0) / (len(t_arr) + 1.0))
    else:
        stu_lo, stu_hi = float(pct_lo), float(pct_hi)
        centered = diffs - diff_hat
        p_two = float((np.sum(np.abs(centered) >= abs(diff_hat - float(null_value))) + 1.0) / (n_boot + 1.0))

    return {
        "num_sequences": int(n_seq),
        "val_a": round(float(val_a), 5),
        "val_b": round(float(val_b), 5),
        "diff": round(diff_hat, 5),
        "se_hat": round(se_hat, 6),
        "percentile_ci95": [round(float(pct_lo), 5), round(float(pct_hi), 5)],
        "studentized_ci95": [round(float(stu_lo), 5), round(float(stu_hi), 5)],
        "p_value_two_sided": round(min(1.0, max(0.0, p_two)), 6),
    }


def holm_bonferroni_adjust(p_values: Dict[str, float], alpha: float = 0.05) -> OrderedDict:
    """
    F-P0.9 Holm-Bonferroni 逐步多重假设检验校正（严格控制族错误率 FWER <= alpha）。
    输入 {endpoint_name: raw_p_value}，返回按原始顺序排列的校正后 p 值与显著性结论。
    """
    items = sorted(p_values.items(), key=lambda kv: float(kv[1]))
    m = len(items)
    adj_map = {}
    rej_map = {}
    prev_adj = 0.0
    stopped = False

    for rank, (k, p_raw) in enumerate(items):
        mult = m - rank
        p_adj = min(1.0, max(prev_adj, float(p_raw) * mult))
        prev_adj = p_adj
        if not stopped and p_adj <= float(alpha):
            rej = True
        else:
            stopped = True
            rej = False
        adj_map[k] = round(p_adj, 6)
        rej_map[k] = rej

    out = OrderedDict()
    for k in p_values.keys():
        out[k] = {
            "raw_p": round(float(p_values[k]), 6),
            "holm_adjusted_p": adj_map[k],
            "reject_null_at_alpha": rej_map[k],
            "alpha": float(alpha),
        }
    return out
