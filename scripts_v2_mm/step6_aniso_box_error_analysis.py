# -*- coding: utf-8 -*-
"""
低成本补充实验（研究内容二各向异性误差证据）：
在干净数据上分别提取 m1（64 线激光雷达）与 m2（环视摄像头 Camera-LSS）的独立检测框中心，
相对于对应传感器原点将 2D 框中心误差分解为「径向误差 e_r（沿视线）」与「横向/切向误差 e_t」，
按距离分档统计标准差与均值，拟合二次径向模型 sigma_r(d) = a + b*d^2 与线性横向模型 sigma_t(d) = a + b*d，
输出冻结统计 JSON 与高清中文/英文对比图。
"""

import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJ_ROOT = "/root/autodl-tmp/proj_dir2_hetero_trust"
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from src_v2_mm.eval_stats_utils import decompose_box_center_radial_tangential_error
from src_v2_mm.heal_base_wrapper import HEALBaseWrapper
from src_v2_mm.step4_1_v2_core import OPV2V_TRAIN_DIR

DEFAULT_EVAL_SPLIT = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_3_eval320_split.json")
DEFAULT_OUT_JSON = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step6_aniso_box_error_stats.json")
DEFAULT_OUT_FIG = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "fig_step6_aniso_radial_tangential_error.png")

DIST_BINS = [(0.0, 15.0), (15.0, 25.0), (25.0, 35.0), (35.0, 45.0), (45.0, 60.0)]


def _fit_quad_and_lin(dists: np.ndarray, sigmas: np.ndarray):
    """拟合 sigma(d) = a + b*d^2（二次）与 sigma(d) = a + b*d（线性）并返回拟合系数与 R^2。"""
    if len(dists) < 2:
        return {"a": 0.0, "b": 0.0, "r2": 0.0}, {"a": 0.0, "b": 0.0, "r2": 0.0}

    def _lsq(x_feat: np.ndarray):
        A = np.stack([np.ones_like(x_feat), x_feat], axis=1)
        coef, _, _, _ = np.linalg.lstsq(A, sigmas, rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((sigmas - pred) ** 2))
        ss_tot = float(np.sum((sigmas - np.mean(sigmas)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
        return {"a": round(float(coef[0]), 6), "b": round(float(coef[1]), 8), "r2": round(float(r2), 4)}

    return _lsq(dists ** 2), _lsq(dists)


def main():
    parser = argparse.ArgumentParser(description="m1/m2 干净数据检测框中心径向与横向误差分档统计与拟合")
    parser.add_argument("--data_dir", type=str, default=OPV2V_TRAIN_DIR)
    parser.add_argument("--eval_split_json", type=str, default=DEFAULT_EVAL_SPLIT)
    parser.add_argument("--max_match_dist", type=float, default=4.0)
    parser.add_argument("--out_json", type=str, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_fig", type=str, default=DEFAULT_OUT_FIG)
    args = parser.parse_args()

    t0 = time.time()
    with open(args.eval_split_json, "r", encoding="utf-8") as f:
        eval_split = json.load(f)
    indices = list(eval_split["all_indices"])

    wrapper = HEALBaseWrapper(device="cuda")
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
    from opencood.tools import train_utils
    from opencood.utils import common_utils

    ds_m1 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m1")
    ds_m2 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m2")

    raw_records = {"m1": [], "m2": []}
    for rank, idx in enumerate(indices):
        for mod, ds in (("m1", ds_m1), ("m2", ds_m2)):
            raw_b = train_utils.to_device(ds.collate_batch_test([ds[idx]]), wrapper.device_obj)
            p = wrapper.extract_protocol_features(raw_b["ego"])
            feat_all = p["heter_feature_2d"]
            amat = p["affine_matrix"]
            tmat01 = amat[0, 0, 1:2]
            ac = wrapper.model.pyramid_backbone.align_corners

            # 将协作者(索引 1)的局部协议特征对齐到本车 ego 坐标系，单独走检测头解码其独立感知框
            # （仅评估该协作者自己看到的框，不受本车融合干扰）
            feat_cav_ego = warp_affine_simple(feat_all[1:2], tmat01, feat_all.shape[-2:], align_corners=ac)
            with torch.no_grad():
                out_cav = wrapper.forward_single_agent(feat_cav_ego)
                pb, ps, gb = wrapper.decode_boxes(raw_b, out_cav, ds)

            if pb is None or len(pb) == 0 or gb is None or len(gb) == 0:
                continue

            pb_np = common_utils.torch_tensor_to_numpy(pb)
            gb_np = common_utils.torch_tensor_to_numpy(gb)
            pred_xy = pb_np[:, :4, :2].mean(axis=1)
            gt_xy = gb_np[:, :4, :2].mean(axis=1)

            # 读取协作者(1)在 ego(0) 坐标系下的真实原点位置：
            # OpenCOOD 定义 pairwise_t_matrix[b, i, j] = T_{j <- i} (i_to_j)，故 1_to_0 为 [0, 1, 0]
            ptm = raw_b["ego"]["pairwise_t_matrix"]
            cav_origin_xy = (float(ptm[0, 1, 0, 0, 3].item()), float(ptm[0, 1, 0, 1, 3].item()))

            recs = decompose_box_center_radial_tangential_error(
                pred_xy, gt_xy, sensor_origin_xy=cav_origin_xy, max_match_dist=args.max_match_dist
            )
            raw_records[mod].extend(recs)

        if (rank + 1) % 40 == 0 or (rank + 1) == len(indices):
            print(f"  [Step 6 误差统计] 进度 {rank + 1}/{len(indices)} ({time.time() - t0:.1f}s)", flush=True)

    summary = OrderedDict()
    for mod in ("m1", "m2"):
        recs = raw_records[mod]
        bin_rows = []
        d_mids, sr_vals, st_vals = [], [], []
        for lo, hi in DIST_BINS:
            sub = [r for r in recs if lo <= r["dist"] < hi]
            if len(sub) < 5:
                continue
            er = np.array([r["e_radial"] for r in sub], dtype=np.float64)
            et = np.array([r["e_tangential"] for r in sub], dtype=np.float64)
            dm = float(np.mean([r["dist"] for r in sub]))
            sr = float(np.std(er, ddof=1))
            st = float(np.std(et, ddof=1))
            d_mids.append(dm)
            sr_vals.append(sr)
            st_vals.append(st)
            bin_rows.append({
                "range_m": [lo, hi],
                "mean_dist_m": round(dm, 3),
                "count": len(sub),
                "radial_std_m": round(sr, 4),
                "radial_abs_mean_m": round(float(np.mean(np.abs(er))), 4),
                "radial_bias_m": round(float(np.mean(er)), 4),
                "tangential_std_m": round(st, 4),
                "tangential_abs_mean_m": round(float(np.mean(np.abs(et))), 4),
                "anisotropy_ratio_sr_over_st": round(sr / max(st, 1e-6), 3),
            })

        dm_arr = np.array(d_mids, dtype=np.float64)
        sr_arr = np.array(sr_vals, dtype=np.float64)
        st_arr = np.array(st_vals, dtype=np.float64)
        rad_quad, rad_lin = _fit_quad_and_lin(dm_arr, sr_arr)
        tan_quad, tan_lin = _fit_quad_and_lin(dm_arr, st_arr)
        summary[mod] = OrderedDict({
            "total_matched_boxes": len(recs),
            "bins": bin_rows,
            "radial_fit_quad_a_plus_bd2": rad_quad,
            "radial_fit_lin_a_plus_bd": rad_lin,
            "tangential_fit_lin_a_plus_bd": tan_lin,
            "tangential_fit_quad_a_plus_bd2": tan_quad,
        })

    out_obj = OrderedDict({
        "meta": {
            "description": "m1(LiDAR) vs m2(Camera-LSS) 检测框中心误差径向/横向分档统计与曲线拟合（研究内容二各向异性实测证据）",
            "num_frames": len(indices),
            "max_match_dist_m": float(args.max_match_dist),
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "results": summary,
    })
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=300)
    d_grid = np.linspace(5.0, 58.0, 200)
    for ax, mod, title in zip(axes, ("m1", "m2"), ("m1 (64-beam LiDAR)", "m2 (Camera-LSS)")):
        rows = summary[mod]["bins"]
        if not rows:
            continue
        xs = np.array([r["mean_dist_m"] for r in rows])
        sr = np.array([r["radial_std_m"] for r in rows])
        st = np.array([r["tangential_std_m"] for r in rows])
        fq = summary[mod]["radial_fit_quad_a_plus_bd2"]
        fl = summary[mod]["tangential_fit_lin_a_plus_bd"]

        ax.plot(xs, sr, "o", color="#D9534F", markersize=7, label=r"Radial $\sigma_r(d)$ (measured)")
        ax.plot(d_grid, fq["a"] + fq["b"] * (d_grid ** 2), "-", color="#D9534F", linewidth=2.0,
                label=rf"Quad fit: {fq['a']:.2f}+{fq['b']:.1e}$d^2$ ($R^2$={fq['r2']:.2f})")
        ax.plot(xs, st, "s", color="#2B6CB0", markersize=7, label=r"Tangential $\sigma_t(d)$ (measured)")
        ax.plot(d_grid, fl["a"] + fl["b"] * d_grid, "--", color="#2B6CB0", linewidth=2.0,
                label=rf"Lin fit: {fl['a']:.2f}+{fl['b']:.3f}$d$ ($R^2$={fl['r2']:.2f})")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Distance to Collaborator Sensor (m)")
        ax.set_ylabel("Box Center Error Std (m)")
        ax.set_ylim(0, max(1.6, float(np.max(sr)) * 1.25))
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8.5, loc="upper left")

    fig.suptitle("Radial vs. Tangential Box Center Localization Error (Clean Dev-320)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[完成] 统计结果：{args.out_json}，曲线图：{args.out_fig}")


if __name__ == "__main__":
    main()
