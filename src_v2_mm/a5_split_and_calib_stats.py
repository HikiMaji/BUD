# -*- coding: utf-8 -*-
"""
A5：标定数据（7 条序列）与调参数据（4 条序列）严格按序列解耦、分档样本统计、
序列级 Bootstrap 95% CI 与稀疏分档自动向上一级合并模块。

严格满足审稿意见 A5 全部要求：
1. Fit 的 11 条父序列拆分为互斥两组：
   - FIT_CALIB_SEQ_NAMES_7（7 条序列）：专门统计 mu_l, s_l, Q, Q_frame, tau_N 等全部标定统计量；
   - FIT_TUNE_SEQ_NAMES_4 （4 条序列）：专门用于网格搜索挑选超参；
   - 强制断言两组零共享序列，且与 Eval/Test 零重叠。
2. 对每个标定分档输出精确的 grid_count（格子数）、frame_count（帧数）、seq_count（序列数）；
3. 对每个阈值 Q 做 2,000 次按父序列整块 Bootstrap，输出 95% 置信区间 q_ci95；
4. 样本过少的稀疏分档（默认 frame_count < 20 或 grid_count < 100）自动向上一级合并
   （例如去掉距离维回退至“传感器×可见性”，或去掉可见性维回退至“传感器×前景/背景”），并在 JSON 的 merged_bins 中显式记录。
"""

from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src_v2_mm.eval_stats_utils import build_seq_bootstrap_indices
from src_v2_mm.step4_1_v2_core import (
    CHANNEL_NAMES,
    DIST_LABELS_5,
    EVAL_SEQ_NAMES,
    FIT_SEQ_NAMES,
    VIS_LABELS_3,
)

# 7 条纯标定序列（覆盖 2/3/4/5 车全部规模，共 1,769 原始帧）
FIT_CALIB_SEQ_NAMES_7 = [
    "2021_08_21_15_41_04",  # 2车, 162帧
    "2021_08_16_22_26_54",  # 3车, 419帧
    "2021_08_18_22_16_12",  # 3车, 267帧
    "2021_08_23_11_22_46",  # 3车, 156帧
    "2021_08_21_09_09_41",  # 4车, 174帧
    "2021_08_22_07_24_12",  # 4车, 190帧
    "2021_09_09_23_21_21",  # 5车, 451帧
]

# 4 条纯调参序列（与上述 7 条完全互斥，共 493 原始帧）
FIT_TUNE_SEQ_NAMES_4 = [
    "2021_08_20_20_39_00",  # 3车, 110帧
    "2021_08_21_22_21_37",  # 3车, 134帧
    "2021_08_22_10_46_58",  # 3车, 127帧
    "2021_08_18_19_11_02",  # 4车, 122帧
]


def assert_disjoint_a5_splits():
    """强制校验 7 标定序列、4 调参序列、8 验证序列三者两两严格互斥且并集完整。"""
    s_cal = set(FIT_CALIB_SEQ_NAMES_7)
    s_tun = set(FIT_TUNE_SEQ_NAMES_4)
    s_evl = set(EVAL_SEQ_NAMES)
    assert len(s_cal) == 7 and len(s_tun) == 4, "A5 标定/调参序列数量必须为 7 + 4！"
    assert s_cal.isdisjoint(s_tun), f"标定与调参序列重叠：{s_cal & s_tun}"
    assert (s_cal | s_tun) == set(FIT_SEQ_NAMES), "7+4 并集与原 FIT-11 序列不一致！"
    assert (s_cal | s_tun).isdisjoint(s_evl), "Fit 与 Eval 序列重叠！"


def seq_block_bootstrap_quantiles(
    per_seq_arrays: Dict[str, np.ndarray],
    seq_order: Sequence[str],
    quantiles: Sequence[float],
    n_boot: int = 2000,
    seed: int = 20261120,
    max_per_seq_sample: int = 8192,
) -> OrderedDict:
    """
    对给定分档内按序列组织的像素值字典 {seq_name: 1D array} 计算点估计分位数及其序列块 Bootstrap 95% CI。
    为保证 2000 次重采样高速完成，当单序列某桶像素数超大时做固定种子无偏子采样。
    """
    non_empty_seqs = [s for s in seq_order if s in per_seq_arrays and per_seq_arrays[s].size > 0]
    if not non_empty_seqs:
        out = OrderedDict()
        for q in quantiles:
            ql = f"p{q}" if q != int(q) else f"p{int(q)}.0"
            out[ql] = {"val": 0.0, "ci95": [0.0, 0.0]}
        return out

    all_cat = np.concatenate([per_seq_arrays[s] for s in non_empty_seqs], axis=0).astype(np.float64, copy=False)
    pt_qs = np.percentile(all_cat, quantiles)

    if len(non_empty_seqs) == 1:
        out = OrderedDict()
        for q, v in zip(quantiles, pt_qs):
            ql = f"p{q}" if q != int(q) else f"p{int(q)}.0"
            rv = round(float(v), 6)
            out[ql] = {"val": rv, "ci95": [rv, rv]}
        return out

    rng = np.random.default_rng(int(seed) + 17)
    slim_seqs = {}
    for s in non_empty_seqs:
        arr = per_seq_arrays[s]
        if arr.size > int(max_per_seq_sample):
            idx = rng.choice(arr.size, size=int(max_per_seq_sample), replace=False)
            slim_seqs[s] = arr[idx].astype(np.float32, copy=False)
        else:
            slim_seqs[s] = arr.astype(np.float32, copy=False)

    boot_idx = build_seq_bootstrap_indices(non_empty_seqs, n_boot=n_boot, seed=seed)
    boot_q_mat = np.empty((n_boot, len(quantiles)), dtype=np.float64)
    for bi in range(n_boot):
        picked = [slim_seqs[non_empty_seqs[j]] for j in boot_idx[bi]]
        cat_b = np.concatenate(picked, axis=0)
        boot_q_mat[bi] = np.percentile(cat_b, quantiles)

    lo_vec = np.percentile(boot_q_mat, 2.5, axis=0)
    hi_vec = np.percentile(boot_q_mat, 97.5, axis=0)

    out = OrderedDict()
    for qi, q in enumerate(quantiles):
        ql = f"p{q}" if q != int(q) else f"p{int(q)}.0"
        out[ql] = {
            "val": round(float(pt_qs[qi]), 6),
            "ci95": [round(float(lo_vec[qi]), 6), round(float(hi_vec[qi]), 6)],
        }
    return out


def build_hierarchical_channel_stats_with_merging(
    per_frame_records: List[dict],
    calib_seq_order: Sequence[str],
    min_frames: int = 20,
    min_grids: int = 100,
    n_boot: int = 2000,
    seed: int = 20261120,
) -> Tuple[OrderedDict, dict]:
    """
    从 7 条标定序列的逐帧记录构造 4 通道统计表（兼容 _get_tier_channel_z 全部键名），
    并实现 A5 要求的：
    - 每个 (传感器 × 距离 × 可见性) 分档的 grid_count / frame_count / seq_count；
    - 样本不足（frame_count < min_frames 或 grid_count < min_grids）时自动去掉距离维，
      向上一级合并到“传感器 × 可见性”（若仍不足则回退到“传感器”），并写入 merged_bins_log；
    - 对 p95 / p98 标尺分位提供序列级 Bootstrap 95% CI。
    """
    # 组织结构: buf[cname][level_key][seq] -> list of 1D arrays; 以及帧/序列计数器
    def _new_bin_tracker():
        return {"by_seq": OrderedDict(), "frames": 0, "seqs": set(), "grids": 0}

    def _push(tracker: dict, seq_name: str, vals_1d: np.ndarray):
        if vals_1d.size == 0:
            return
        tracker["by_seq"].setdefault(seq_name, []).append(vals_1d.astype(np.float32, copy=False))
        tracker["frames"] += 1
        tracker["seqs"].add(seq_name)
        tracker["grids"] += int(vals_1d.size)

    tr_unified = {c: _new_bin_tracker() for c in CHANNEL_NAMES}
    tr_sensor = {c: {m: _new_bin_tracker() for m in ("m1", "m2")} for c in CHANNEL_NAMES}
    tr_sd = {c: {m: {dl: _new_bin_tracker() for dl in DIST_LABELS_5} for m in ("m1", "m2")} for c in CHANNEL_NAMES}
    tr_sv = {c: {m: {vl: _new_bin_tracker() for vl in VIS_LABELS_3} for m in ("m1", "m2")} for c in CHANNEL_NAMES}
    tr_sdv = {
        c: {m: {dl: {vl: _new_bin_tracker() for vl in VIS_LABELS_3} for dl in DIST_LABELS_5} for m in ("m1", "m2")}
        for c in CHANNEL_NAMES
    }

    for rec in per_frame_records:
        seq_n = rec["seq_name"]
        d_np = rec["d_idx"].numpy()
        v_np = rec["v_idx"].numpy()
        for mod in ("m1", "m2"):
            cov_np = rec["cov_by_mod"][mod].numpy().astype(bool)
            dev_np = rec[mod].numpy()
            for ci, cname in enumerate(CHANNEL_NAMES):
                v_all = dev_np[ci][cov_np]
                _push(tr_unified[cname], seq_n, v_all)
                _push(tr_sensor[cname][mod], seq_n, v_all)
                for vi, vl in enumerate(VIS_LABELS_3):
                    mv = cov_np & (v_np == vi)
                    if mv.any():
                        _push(tr_sv[cname][mod][vl], seq_n, dev_np[ci][mv])
                for di, dl in enumerate(DIST_LABELS_5):
                    md = cov_np & (d_np == di)
                    if md.any():
                        _push(tr_sd[cname][mod][dl], seq_n, dev_np[ci][md])
                        for vi, vl in enumerate(VIS_LABELS_3):
                            mdv = md & (v_np == vi)
                            if mdv.any():
                                _push(tr_sdv[cname][mod][dl][vl], seq_n, dev_np[ci][mdv])

    def _finalize_tracker(tr: dict, with_ci: bool = False) -> dict:
        if tr["grids"] == 0:
            return {
                "count": 0, "grid_count": 0, "frame_count": 0, "seq_count": 0,
                "mean": 0.0, "std": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p98": 0.0, "p99": 0.0,
            }
        per_seq = {s: np.concatenate(arrs, axis=0) for s, arrs in tr["by_seq"].items()}
        cat = np.concatenate(list(per_seq.values()), axis=0).astype(np.float64, copy=False)
        qs = np.percentile(cat, [50.0, 90.0, 95.0, 98.0, 99.0])
        res = OrderedDict({
            "count": int(tr["grids"]),
            "grid_count": int(tr["grids"]),
            "frame_count": int(tr["frames"]),
            "seq_count": len(tr["seqs"]),
            "mean": round(float(cat.mean()), 6),
            "std": round(float(cat.std()), 6),
            "median": round(float(qs[0]), 6),
            "p90": round(float(qs[1]), 6),
            "p95": round(float(qs[2]), 6),
            "p98": round(float(qs[3]), 6),
            "p99": round(float(qs[4]), 6),
        })
        if with_ci:
            bci = seq_block_bootstrap_quantiles(per_seq, calib_seq_order, [50.0, 95.0, 98.0], n_boot=n_boot, seed=seed)
            res["bootstrap_ci95"] = {
                "median": bci["p50.0"]["ci95"],
                "p95": bci["p95.0"]["ci95"],
                "p98": bci["p98.0"]["ci95"],
            }
        return res

    out_stats = OrderedDict()
    merged_bins_log = []

    for cname in CHANNEL_NAMES:
        st_c = OrderedDict()
        st_c["unified"] = _finalize_tracker(tr_unified[cname], with_ci=False)
        st_c["sensor"] = OrderedDict({m: _finalize_tracker(tr_sensor[cname][m], with_ci=True) for m in ("m1", "m2")})
        st_c["sensor_vis"] = OrderedDict({
            m: OrderedDict({vl: _finalize_tracker(tr_sv[cname][m][vl], with_ci=True) for vl in VIS_LABELS_3})
            for m in ("m1", "m2")
        })
        st_c["sensor_dist"] = OrderedDict()
        st_c["sensor_dist_vis"] = OrderedDict()

        for m in ("m1", "m2"):
            st_c["sensor_dist"][m] = OrderedDict()
            st_c["sensor_dist_vis"][m] = OrderedDict()
            for dl in DIST_LABELS_5:
                tr_d = tr_sd[cname][m][dl]
                if tr_d["frames"] < int(min_frames) or tr_d["grids"] < int(min_grids):
                    fb = OrderedDict(st_c["sensor"][m])
                    fb["merged_from"] = f"sensor[{m}]"
                    fb["raw_frame_count"] = int(tr_d["frames"])
                    fb["raw_grid_count"] = int(tr_d["grids"])
                    st_c["sensor_dist"][m][dl] = fb
                    if cname == CHANNEL_NAMES[0]:
                        merged_bins_log.append({
                            "table": "sensor_dist",
                            "bin": f"{m}|{dl}",
                            "raw_frame_count": int(tr_d["frames"]),
                            "raw_grid_count": int(tr_d["grids"]),
                            "fallback_to": f"{m} (sensor)",
                        })
                else:
                    st_c["sensor_dist"][m][dl] = _finalize_tracker(tr_d, with_ci=False)

                st_c["sensor_dist_vis"][m][dl] = OrderedDict()
                for vl in VIS_LABELS_3:
                    tr_dv = tr_sdv[cname][m][dl][vl]
                    if tr_dv["frames"] < int(min_frames) or tr_dv["grids"] < int(min_grids):
                        # A5 规范：样本太少的分档向上一级合并（去掉距离维 -> 传感器×可见性；若仍不足 -> 传感器）
                        sv_parent = st_c["sensor_vis"][m][vl]
                        if sv_parent["frame_count"] >= int(min_frames) and sv_parent["grid_count"] >= int(min_grids):
                            fb = OrderedDict(sv_parent)
                            fb_target = f"{m}|{vl} (drop_distance)"
                        else:
                            fb = OrderedDict(st_c["sensor"][m])
                            fb_target = f"{m} (sensor)"
                        fb["merged_from"] = fb_target
                        fb["raw_frame_count"] = int(tr_dv["frames"])
                        fb["raw_grid_count"] = int(tr_dv["grids"])
                        fb["raw_seq_count"] = len(tr_dv["seqs"])
                        st_c["sensor_dist_vis"][m][dl][vl] = fb
                        if cname == CHANNEL_NAMES[0]:
                            merged_bins_log.append({
                                "table": "sensor_dist_vis",
                                "bin": f"{m}|{dl}|{vl}",
                                "raw_frame_count": int(tr_dv["frames"]),
                                "raw_grid_count": int(tr_dv["grids"]),
                                "raw_seq_count": len(tr_dv["seqs"]),
                                "fallback_to": fb_target,
                            })
                    else:
                        st_c["sensor_dist_vis"][m][dl][vl] = _finalize_tracker(tr_dv, with_ci=True)

        out_stats[cname] = st_c

    meta_diag = {
        "min_frames_threshold": int(min_frames),
        "min_grids_threshold": int(min_grids),
        "num_merged_bins": len(merged_bins_log),
        "merged_bins_log": merged_bins_log,
    }
    return out_stats, meta_diag


def build_fg_bg_vis_quantiles_with_ci_and_merging(
    per_seq_bg_fg: dict,
    frame_counts: dict,
    calib_seq_order: Sequence[str],
    q_levels: Sequence[float] = (90.0, 95.0, 98.0, 99.0, 99.5, 99.8, 99.9, 99.99),
    min_frames: int = 20,
    min_grids: int = 100,
    n_boot: int = 2000,
    seed: int = 20261120,
) -> Tuple[OrderedDict, OrderedDict, list]:
    """
    A5：计算 (传感器 × 可见性 × 前景/背景) 阈值 Q 的点估计、序列级 Bootstrap 95% CI、
    (grid_count, frame_count, seq_count) 以及样本不足时的向上一级合并（去掉可见性维 -> 传感器×前景/背景）。

    参数：
    - per_seq_bg_fg[role][mod][vis_name][seq_name] -> list of 1D float32 arrays (role in {"bg", "fg"})
    - frame_counts[role][mod][vis_name] -> int (非空出现帧数)
    """
    out_bg = OrderedDict()
    out_fg = OrderedDict()
    merged_q_log = []

    for role, out_dict in (("bg", out_bg), ("fg", out_fg)):
        for m in ("m1", "m2"):
            out_dict[m] = OrderedDict()
            # 预构建向上一级父桶（去掉可见性维：sensor × role）
            parent_by_seq = OrderedDict()
            parent_frames = 0
            for vn in VIS_LABELS_3:
                parent_frames += int(frame_counts[role][m][vn])
                for s, arr_list in per_seq_bg_fg[role][m][vn].items():
                    if arr_list:
                        parent_by_seq.setdefault(s, []).extend(arr_list)
            parent_cat_by_seq = {s: np.concatenate(v, axis=0) for s, v in parent_by_seq.items() if v}
            parent_q = seq_block_bootstrap_quantiles(parent_cat_by_seq, calib_seq_order, q_levels, n_boot=n_boot, seed=seed)
            parent_grids = int(sum(a.size for a in parent_cat_by_seq.values()))

            for vn in VIS_LABELS_3:
                seq_lists = per_seq_bg_fg[role][m][vn]
                cat_by_seq = {s: np.concatenate(v, axis=0) for s, v in seq_lists.items() if v}
                g_cnt = int(sum(a.size for a in cat_by_seq.values()))
                f_cnt = int(frame_counts[role][m][vn])
                s_cnt = len(cat_by_seq)

                if f_cnt < int(min_frames) or g_cnt < int(min_grids):
                    q_use = parent_q
                    merged_from = f"{m}|{role}_all_vis (drop_visibility)"
                    merged_q_log.append({
                        "table": f"pixel_cos_by_vis_{role}",
                        "bin": f"{m}|{vn}|{role}",
                        "raw_frame_count": f_cnt,
                        "raw_grid_count": g_cnt,
                        "raw_seq_count": s_cnt,
                        "fallback_to": merged_from,
                        "parent_grid_count": parent_grids,
                    })
                else:
                    q_use = seq_block_bootstrap_quantiles(cat_by_seq, calib_seq_order, q_levels, n_boot=n_boot, seed=seed)
                    merged_from = None

                bin_rec = OrderedDict()
                bin_ci = OrderedDict()
                for ql, qinfo in q_use.items():
                    bin_rec[ql] = qinfo["val"]
                    bin_ci[ql] = qinfo["ci95"]
                bin_rec["count"] = g_cnt
                bin_rec["grid_count"] = g_cnt
                bin_rec["frame_count"] = f_cnt
                bin_rec["seq_count"] = s_cnt
                bin_rec["bootstrap_ci95"] = bin_ci
                if merged_from is not None:
                    bin_rec["merged_from"] = merged_from
                out_dict[m][vn] = bin_rec

    return out_bg, out_fg, merged_q_log
