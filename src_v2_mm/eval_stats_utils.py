# -*- coding: utf-8 -*-
"""
科研统计与几何误差分解公共模块：
1. 成对差值 Block-Bootstrap 95% 置信区间（按父序列整块重采样，消除跨序列固有方差假重叠）：
   - AP@0.5 / AP@0.7 成对差值 Delta = AP(A) - AP(B) 及其 95% CI 与统计显著性标记；
   - 比率指标（盲区召回率、分组误报率等）成对差值 Delta = Rate(A) - Rate(B) 及其 95% CI；
2. 检测框中心径向/横向误差分解工具（供研究内容二各向异性误差分析与曲线拟合使用）。
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def fast_voc_ap(tp_arr, fp_arr, score_arr, gt_total: int) -> float:
    """向量化 VOC-2010 AP（与 HEAL eval_utils.calculate_ap 严格等价，兼容 Python list 与 ndarray）。"""
    tp_np = np.asarray(tp_arr, dtype=np.float64).reshape(-1)
    fp_np = np.asarray(fp_arr, dtype=np.float64).reshape(-1)
    sc_np = np.asarray(score_arr, dtype=np.float64).reshape(-1)
    if int(gt_total) <= 0 or sc_np.size == 0:
        return 0.0
    order = np.argsort(-sc_np)
    tp_c = np.cumsum(tp_np[order], dtype=np.float64)
    fp_c = np.cumsum(fp_np[order], dtype=np.float64)
    rec = tp_c / float(gt_total)
    prec = tp_c / np.maximum(fp_c + tp_c, 1e-12)

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))


def _concat_seq_ap(seq_map: Dict[str, dict], picked_seqs: Sequence[str]) -> float:
    """按给定序列重采样列表合并 TP/FP/Score 并计算聚合 VOC-2010 AP。"""
    tps, fps, scs = [], [], []
    gt_sum = 0
    for s in picked_seqs:
        rec = seq_map[s]
        gt_sum += int(rec["gt"])
        if len(rec["score"]) > 0:
            tps.append(rec["tp"])
            fps.append(rec["fp"])
            scs.append(rec["score"])
    if gt_sum <= 0 or not scs:
        return 0.0
    return fast_voc_ap(
        np.concatenate(tps, axis=0),
        np.concatenate(fps, axis=0),
        np.concatenate(scs, axis=0),
        gt_sum,
    )


def build_seq_bootstrap_indices(seq_names: Sequence[str], n_boot: int = 2000, seed: int = 20261120) -> np.ndarray:
    """固定随机种子生成 (n_boot, n_seq) 的父序列整块重采样索引表，确保成对比较同批抽取。"""
    rng = np.random.default_rng(int(seed))
    n_seq = len(seq_names)
    return rng.integers(0, n_seq, size=(int(n_boot), n_seq))


def paired_block_bootstrap_ap_diff(
    seq_names: Sequence[str],
    seq_map_a: Dict[str, dict],
    seq_map_b: Dict[str, dict],
    boot_indices: Optional[np.ndarray] = None,
    n_boot: int = 2000,
    seed: int = 20261120,
) -> dict:
    """
    计算方法 A 与方法 B 在同一按序列整块重采样下的成对 AP 差值 Delta = AP(A) - AP(B) 的 95% CI。
    seq_map_x[seq_name] 需包含 np.ndarray 格式的 "tp", "fp", "score" 与整数 "gt"。
    """
    if boot_indices is None:
        boot_indices = build_seq_bootstrap_indices(seq_names, n_boot=n_boot, seed=seed)

    val_a = _concat_seq_ap(seq_map_a, seq_names)
    val_b = _concat_seq_ap(seq_map_b, seq_names)
    diff_point = val_a - val_b

    n_b = boot_indices.shape[0]
    diffs = np.empty(n_b, dtype=np.float64)
    vals_a = np.empty(n_b, dtype=np.float64)
    vals_b = np.empty(n_b, dtype=np.float64)
    seq_list = list(seq_names)

    for bi in range(n_b):
        picked = [seq_list[j] for j in boot_indices[bi]]
        aa = _concat_seq_ap(seq_map_a, picked)
        bb = _concat_seq_ap(seq_map_b, picked)
        vals_a[bi] = aa
        vals_b[bi] = bb
        diffs[bi] = aa - bb

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    a_lo, a_hi = np.percentile(vals_a, [2.5, 97.5])
    b_lo, b_hi = np.percentile(vals_b, [2.5, 97.5])
    return {
        "val_a": round(float(val_a), 5),
        "val_b": round(float(val_b), 5),
        "a_ci95": [round(float(a_lo), 5), round(float(a_hi), 5)],
        "b_ci95": [round(float(b_lo), 5), round(float(b_hi), 5)],
        "diff": round(float(diff_point), 5),
        "diff_ci95": [round(float(lo), 5), round(float(hi), 5)],
        "significant_pos": bool(lo > 0.0),
        "significant_neg": bool(hi < 0.0),
    }


def paired_block_bootstrap_rate_diff(
    seq_names: Sequence[str],
    seq_hits_a: Dict[str, int],
    seq_hits_b: Dict[str, int],
    seq_totals: Dict[str, int],
    boot_indices: Optional[np.ndarray] = None,
    n_boot: int = 2000,
    seed: int = 20261120,
) -> dict:
    """
    计算比率类指标（如盲区召回率、分组误报率）的成对差值 Delta = Rate(A) - Rate(B) 的 95% CI。
    """
    if boot_indices is None:
        boot_indices = build_seq_bootstrap_indices(seq_names, n_boot=n_boot, seed=seed)

    ha_vec = np.array([int(seq_hits_a.get(s, 0)) for s in seq_names], dtype=np.float64)
    hb_vec = np.array([int(seq_hits_b.get(s, 0)) for s in seq_names], dtype=np.float64)
    tot_vec = np.array([int(seq_totals.get(s, 0)) for s in seq_names], dtype=np.float64)

    tot_sum = float(tot_vec.sum())
    val_a = float(ha_vec.sum() / tot_sum) if tot_sum > 0 else 0.0
    val_b = float(hb_vec.sum() / tot_sum) if tot_sum > 0 else 0.0
    diff_point = val_a - val_b

    # 向量化计算全部 2000 次整块重采样
    ha_b = ha_vec[boot_indices].sum(axis=1)
    hb_b = hb_vec[boot_indices].sum(axis=1)
    tot_b = tot_vec[boot_indices].sum(axis=1)
    valid = tot_b > 0
    ra_b = np.where(valid, ha_b / np.maximum(tot_b, 1.0), 0.0)
    rb_b = np.where(valid, hb_b / np.maximum(tot_b, 1.0), 0.0)
    diffs = ra_b - rb_b

    if valid.any():
        lo, hi = np.percentile(diffs[valid], [2.5, 97.5])
        a_lo, a_hi = np.percentile(ra_b[valid], [2.5, 97.5])
        b_lo, b_hi = np.percentile(rb_b[valid], [2.5, 97.5])
    else:
        lo = hi = a_lo = a_hi = b_lo = b_hi = 0.0

    return {
        "val_a": round(val_a, 5),
        "val_b": round(val_b, 5),
        "a_ci95": [round(float(a_lo), 5), round(float(a_hi), 5)],
        "b_ci95": [round(float(b_lo), 5), round(float(b_hi), 5)],
        "diff": round(diff_point, 5),
        "diff_ci95": [round(float(lo), 5), round(float(hi), 5)],
        "total": int(tot_sum),
        "significant_pos": bool(lo > 0.0),
        "significant_neg": bool(hi < 0.0),
    }


def decompose_box_center_radial_tangential_error(
    pred_centers_xy: np.ndarray,
    gt_centers_xy: np.ndarray,
    sensor_origin_xy: Tuple[float, float] = (0.0, 0.0),
    max_match_dist: float = 4.0,
) -> List[dict]:
    """
    将 2D 检测框中心误差分解为沿传感器光心射线方向的径向误差 e_r 与垂直射线方向的横向误差 e_t。

    返回每个贪心最近邻匹配对的字典：
    {"dist": r_gt, "e_radial": e_r, "e_tangential": e_t, "e_total": norm(err)}。
    """
    if pred_centers_xy.size == 0 or gt_centers_xy.size == 0:
        return []
    ox, oy = float(sensor_origin_xy[0]), float(sensor_origin_xy[1])
    used_gt = set()
    records = []

    # 按预测与 GT 距离升序贪心一对一匹配
    diff_mat = pred_centers_xy[:, None, :] - gt_centers_xy[None, :, :]
    dist_mat = np.linalg.norm(diff_mat, axis=-1)
    pairs = []
    for pi in range(pred_centers_xy.shape[0]):
        gi = int(np.argmin(dist_mat[pi]))
        d = float(dist_mat[pi, gi])
        if d <= float(max_match_dist):
            pairs.append((d, pi, gi))
    pairs.sort(key=lambda x: x[0])

    used_pred = set()
    for d, pi, gi in pairs:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)

        gx, gy = float(gt_centers_xy[gi, 0]), float(gt_centers_xy[gi, 1])
        px, py = float(pred_centers_xy[pi, 0]), float(pred_centers_xy[pi, 1])
        rx, ry = gx - ox, gy - oy
        r_gt = math.hypot(rx, ry)
        if r_gt < 1e-3:
            continue
        ux, uy = rx / r_gt, ry / r_gt          # 径向单位向量
        tx, ty = -uy, ux                        # 横向/切向单位向量
        ex, ey = px - gx, py - gy
        e_r = ex * ux + ey * uy
        e_t = ex * tx + ey * ty
        records.append({
            "dist": r_gt,
            "e_radial": float(e_r),
            "e_tangential": float(e_t),
            "e_total": float(d),
        })
    return records
