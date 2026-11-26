# -*- coding: utf-8 -*-
"""
步骤 ④-1 修正版 · Phase 1：纯 Fit 集统计、超参选择与 4 组门限冻结脚本。

严格纪律：
1. 仅读取 11 条 Fit 序列（共 100 帧 Fit，含 36 帧 Fit-Dev 与 64 帧 Fit-Holdout），严禁访问任何 Eval 帧；
2. 基于 Fit-100 统计多尺度余弦 (L0/L1/L2) 与语义冲突 (sem) 在「统一 / 传感器 / 传感器×5距离(含0-6m) / 传感器×5距离×3可见性」各层级分档的良性分布，并计算像素级越界误报对比；
3. 在 Fit-Dev 36 帧上枚举候选离散配置并自动冻结主方法 (MS-Cos + Clear-Sem) 与消融方法 (MS-Cos Only)；
4. 在 Fit-Holdout 64 帧上检核跨模态误报稳定性与 Tier ④ >= Tier ② 有效性，验证无误后持久化冻结文件。
"""

import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch

PROJ_ROOT = "/root/autodl-tmp/proj_dir2_hetero_trust"
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from src_v2_mm.heal_attacks import (
    generate_local_ghost_spoofing_heal,
    generate_local_near0_6m_attack_heal,
    generate_local_removal_heal,
    generate_scaled_pgd_removal_heal,
)
from src_v2_mm.heal_base_wrapper import HEALBaseWrapper
from src_v2_mm.step4_1_v2_core import (
    CHANNEL_NAMES,
    DIST_LABELS_5,
    PROJ_ROOT as CORE_ROOT,
    TIERS_4,
    TRAIN_FULL19_DIR,
    VIS_LABELS_3,
    _get_tier_channel_z,
    build_and_save_split_plan,
    build_bev_grid_xy_256,
    compute_auroc,
    compute_dev_maps_from_protocol,
    extract_frame_context_and_clean,
    find_exact_fpr_threshold,
)

OUT_DIR = os.path.join(CORE_ROOT, "results_v3")
OUT_SPLIT = os.path.join(CORE_ROOT, "results_rq1_v2_mm", "step4_1_v2_split_plan.json")
OUT_CALIB = os.path.join(OUT_DIR, "a1_fit_base_calibrator.json")

FIT_ATTACKS = [
    "scaled_pgd_0.5sigma",
    "scaled_pgd_1.0sigma",
    "local_removal_1.0sigma",
    "local_ghost_2.0sigma",
    "local_near0_6m_removal_2.0sigma",
]


def _summarize_1d(arr: np.ndarray) -> dict:
    """精确计算 1D 浮点数组的中心与上尾分位统计量。"""
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p98": 0.0, "p99": 0.0}
    a = arr.astype(np.float64, copy=False)
    qs = np.percentile(a, [50.0, 90.0, 95.0, 98.0, 99.0])
    return {
        "count": int(a.size),
        "mean": round(float(a.mean()), 6),
        "std": round(float(a.std()), 6),
        "median": round(float(qs[0]), 6),
        "p90": round(float(qs[1]), 6),
        "p95": round(float(qs[2]), 6),
        "p98": round(float(qs[3]), 6),
        "p99": round(float(qs[4]), 6),
    }


def compute_pixel_tier_stats(fit_clean_records: list) -> dict:
    """在 Fit 有效像素上统计 4 通道在 4 个门限层级下的中位与高分位基准（m1 用完整覆盖区，m2 用相机覆盖区，统一新版 v_idx）。"""
    pix = {c: {"unified": [], "m1": [], "m2": []} for c in CHANNEL_NAMES}
    d_list = {"m1": [], "m2": []}
    v_list = {"m1": [], "m2": []}

    for rec in fit_clean_records:
        for mod in ("m1", "m2"):
            cov = rec["cov_by_mod"][mod]
            d_np = rec["d_idx"][cov].numpy()
            v_np = rec["v_idx"][cov].numpy()
            dev = rec[mod]  # CPU float32 (4, 256, 256)
            d_list[mod].append(d_np)
            v_list[mod].append(v_np)
            for ci, cname in enumerate(CHANNEL_NAMES):
                vals = dev[ci][cov].numpy()
                pix[cname][mod].append(vals)

    d_cat = {m: np.concatenate(d_list[m], axis=0) for m in ("m1", "m2")}
    v_cat = {m: np.concatenate(v_list[m], axis=0) for m in ("m1", "m2")}

    out_stats = OrderedDict()
    for cname in CHANNEL_NAMES:
        arr_m1 = np.concatenate(pix[cname]["m1"], axis=0)
        arr_m2 = np.concatenate(pix[cname]["m2"], axis=0)
        arr_all = np.concatenate([arr_m1, arr_m2], axis=0)
        arr_map = {"m1": arr_m1, "m2": arr_m2}

        st_c = OrderedDict()
        st_c["unified"] = _summarize_1d(arr_all)
        st_c["sensor"] = OrderedDict({m: _summarize_1d(arr_map[m]) for m in ("m1", "m2")})
        st_c["sensor_dist"] = OrderedDict()
        st_c["sensor_dist_vis"] = OrderedDict()

        for m in ("m1", "m2"):
            st_c["sensor_dist"][m] = OrderedDict()
            st_c["sensor_dist_vis"][m] = OrderedDict()
            for di, dlbl in enumerate(DIST_LABELS_5):
                dm = d_cat[m] == di
                st_c["sensor_dist"][m][dlbl] = _summarize_1d(arr_map[m][dm])
                st_c["sensor_dist_vis"][m][dlbl] = OrderedDict()
                for vi, vlbl in enumerate(VIS_LABELS_3):
                    dvm = dm & (v_cat[m] == vi)
                    st_c["sensor_dist_vis"][m][dlbl][vlbl] = _summarize_1d(arr_map[m][dvm])
        out_stats[cname] = st_c

    # 计算代表性组合表（等权多尺度余弦，用于分析良性分布趋势与 ② vs ③ vs ④ 像素越界率）
    ms_raw = {}
    for m in ("m1", "m2"):
        c0 = np.concatenate(pix["cos0"][m], axis=0)
        c1 = np.concatenate(pix["cos1"][m], axis=0)
        c2 = np.concatenate(pix["cos2"][m], axis=0)
        ms_raw[m] = (c0 + c1 + c2) / 3.0

    benign_summary = OrderedDict()
    subcell_fpr = OrderedDict()
    for m in ("m1", "m2"):
        benign_summary[m] = OrderedDict()
        subcell_fpr[m] = OrderedDict()
        p95_sensor = float(np.percentile(ms_raw[m], 95.0))
        p95_dist = {di: float(np.percentile(ms_raw[m][d_cat[m] == di], 95.0)) for di in range(5)}
        for vi, vlbl in enumerate(VIS_LABELS_3):
            benign_summary[m][vlbl] = OrderedDict()
            subcell_fpr[m][vlbl] = OrderedDict()
            for di, dlbl in enumerate(DIST_LABELS_5):
                dvm = (d_cat[m] == di) & (v_cat[m] == vi)
                vals = ms_raw[m][dvm]
                benign_summary[m][vlbl][dlbl] = _summarize_1d(vals)
                p95_dv = float(np.percentile(vals, 95.0)) if vals.size > 0 else p95_sensor
                subcell_fpr[m][vlbl][dlbl] = {
                    "tier2_sensor_only_exceed": round(float(np.mean(vals > p95_sensor)), 5) if vals.size else 0.0,
                    "tier3_sensor_dist_exceed": round(float(np.mean(vals > p95_dist[di])), 5) if vals.size else 0.0,
                    "tier4_sensor_dist_vis_exceed": round(float(np.mean(vals > p95_dv)), 5) if vals.size else 0.0,
                }

    return out_stats, {"ms_cos_equal_dist_vis_summary": benign_summary, "subcell_pixel_fpr_compare": subcell_fpr}


def _precompute_z_bank(records_meta: list, dev_list: list, stats: dict, device="cuda") -> dict:
    """按 (tier, q_scale) 在 GPU 上预计算标准化四通道残差图（有效区外置 0），加速超参网格枚举。"""
    bank = {}
    cov_stack = torch.stack([r["cov_2d"] for r in records_meta], dim=0).to(device)  # (N, H, W)
    d_stack = torch.stack([r["d_idx"] for r in records_meta], dim=0).to(device)
    v_stack = torch.stack([r["v_idx"] for r in records_meta], dim=0).to(device)
    dev_stack = torch.stack(dev_list, dim=0).to(device)  # (N, 4, H, W)
    mod_list = [r["mod"] for r in records_meta]
    is_m1 = torch.tensor([m == "m1" for m in mod_list], device=device, dtype=torch.bool)
    is_m2 = ~is_m1

    for q_scale in (95, 98):
        q_key = f"p{q_scale}"
        bank[q_scale] = {}
        for tier in TIERS_4:
            z_out = torch.zeros_like(dev_stack)
            # 按 m1/m2 分批映射（tier1 两者同参数，其余分传感器或查表）
            for mod, mmask in (("m1", is_m1), ("m2", is_m2)):
                if not mmask.any():
                    continue
                idxs = torch.nonzero(mmask, as_tuple=False).squeeze(1)
                for ii in idxs.tolist():
                    z_single = _get_tier_channel_z(
                        dev_stack[ii], mod, d_stack[ii], v_stack[ii], stats, tier, q_key=q_key, s_floor=0.05
                    )
                    z_out[ii] = z_single * cov_stack[ii].unsqueeze(0).float()
            bank[q_scale][tier] = z_out  # GPU (N, 4, 256, 256)
    return {"bank": bank, "v_stack": v_stack, "is_m1": is_m1, "is_m2": is_m2}


def _score_topk_batch(
    z_tensor: torch.Tensor,
    v_stack: torch.Tensor,
    tier: str,
    cfg: dict,
    method_kind: str,
) -> np.ndarray:
    """给定预计算标准化图批量 (N, 4, H, W)，并行向量化计算所有样本的 MeanTopK(K) 分数。"""
    w0, w1, w2 = cfg["cos_weights"]
    cos_comb = w0 * z_tensor[:, 0] + w1 * z_tensor[:, 1] + w2 * z_tensor[:, 2]
    if tier == "tier4_sensor_dist_vis":
        gc = torch.tensor(cfg["gamma_cos"], device=z_tensor.device, dtype=z_tensor.dtype)[v_stack]
        cos_term = gc * cos_comb
    else:
        cos_term = cos_comb

    if method_kind == "ablation_ms_cos":
        smap = cos_term
    else:
        w_sem = float(cfg["w_sem"])
        if tier == "tier4_sensor_dist_vis":
            gs = torch.tensor(cfg["gamma_sem"], device=z_tensor.device, dtype=z_tensor.dtype)[v_stack]
            sem_term = gs * z_tensor[:, 3]
        else:
            sem_term = z_tensor[:, 3]
        smap = (1.0 - w_sem) * cos_term + w_sem * sem_term

    flat = smap.flatten(1)
    top_vals = torch.topk(flat, k=int(cfg["top_k"]), dim=1).values.mean(dim=1)
    return top_vals.detach().cpu().numpy().astype(np.float64)


def _calibrate_and_eval_fit(
    fit_clean_bank: dict,
    fit_clean_meta: list,
    fit_dev_atk_bank: dict,
    fit_dev_atk_meta: list,
    stats: dict,
    method_kind: str,
    cfg: dict,
) -> dict:
    """纯 Fit 内计算 4 组门限校准参数，并在 Fit-Dev 与 Fit-Holdout 上评估有效性（零 Eval 触碰）。"""
    q = int(cfg["q_scale"])
    fnorm = cfg["frame_norm_mode"]
    is_ho = np.array([r["is_holdout"] for r in fit_clean_meta], dtype=bool)
    is_dev = ~is_ho
    is_m1_c = np.array([r["mod"] == "m1" for r in fit_clean_meta], dtype=bool)
    is_m2_c = ~is_m1_c

    atk_mods = np.array([r["mod"] for r in fit_dev_atk_meta])
    atk_names = np.array([r["attack"] for r in fit_dev_atk_meta])
    atk_valid = np.array([r["valid"] for r in fit_dev_atk_meta], dtype=bool)

    tier_res = OrderedDict()
    tier_calib = OrderedDict()

    for tier in TIERS_4:
        clean_raw = _score_topk_batch(fit_clean_bank["bank"][q][tier], fit_clean_bank["v_stack"], tier, cfg, method_kind)
        atk_raw = _score_topk_batch(fit_dev_atk_bank["bank"][q][tier], fit_dev_atk_bank["v_stack"], tier, cfg, method_kind)

        if tier == "tier1_unified":
            th_fit = float(np.percentile(clean_raw, 95.0))
            clean_norm = clean_raw
            atk_norm = atk_raw
            tier_calib[tier] = {"th_fit": round(th_fit, 6)}
        else:
            st_tier = {}
            clean_norm = np.empty_like(clean_raw)
            atk_norm = np.empty_like(atk_raw)
            for mod, cmask in (("m1", is_m1_c), ("m2", is_m2_c)):
                vals_m = clean_raw[cmask]
                med_m = float(np.median(vals_m))
                p95_m = float(np.percentile(vals_m, 95.0))
                st_tier[mod] = {"median": round(med_m, 6), "p95": round(p95_m, 6)}
                if fnorm == "div_p95":
                    clean_norm[cmask] = vals_m / max(p95_m, 1e-4)
                    am = atk_mods == mod
                    atk_norm[am] = atk_raw[am] / max(p95_m, 1e-4)
                else:
                    clean_norm[cmask] = (vals_m - med_m) / max(p95_m - med_m, 1e-4)
                    am = atk_mods == mod
                    atk_norm[am] = (atk_raw[am] - med_m) / max(p95_m - med_m, 1e-4)
            th_fit = 1.0
            tier_calib[tier] = st_tier

        # 1. Fit-Holdout 64 帧在冻结门限 th_fit 下的干净误报检查
        ho_m1 = clean_norm[is_ho & is_m1_c]
        ho_m2 = clean_norm[is_ho & is_m2_c]
        ho_fpr_m1 = float(np.mean(ho_m1 > th_fit))
        ho_fpr_m2 = float(np.mean(ho_m2 > th_fit))
        ho_fpr_tot = float(np.mean(clean_norm[is_ho] > th_fit))

        # 2. Fit-Dev 36 帧上按精确 5% 总误报率评估 5 类攻击 TPR 与 AUROC
        dev_ben_all = clean_norm[is_dev]
        dev_ben_m1 = clean_norm[is_dev & is_m1_c]
        dev_ben_m2 = clean_norm[is_dev & is_m2_c]
        th_dev_5 = find_exact_fpr_threshold(dev_ben_all, 0.05)

        tprs_all = []
        aurocs_all = []
        per_atk = OrderedDict()
        for aname in FIT_ATTACKS:
            tprs_m = {}
            aurocs_m = {}
            pos_list = []
            for mod, ben_m in (("m1", dev_ben_m1), ("m2", dev_ben_m2)):
                pmask = (atk_mods == mod) & (atk_names == aname) & atk_valid
                pos_m = atk_norm[pmask]
                pos_list.append(pos_m)
                tprs_m[mod] = float(np.mean(pos_m > th_dev_5)) if pos_m.size else 0.0
                aurocs_m[mod] = compute_auroc(ben_m, pos_m) if pos_m.size else 0.5
            pos_all = np.concatenate(pos_list, axis=0)
            ov_tpr = float(np.mean(pos_all > th_dev_5)) if pos_all.size else 0.0
            ov_auc = compute_auroc(dev_ben_all, pos_all) if pos_all.size else 0.5
            tprs_all.append(ov_tpr)
            aurocs_all.append(ov_auc)
            per_atk[aname] = {
                "tpr@5": round(ov_tpr, 5),
                "m1_tpr": round(tprs_m["m1"], 5),
                "m2_tpr": round(tprs_m["m2"], 5),
                "m1_auroc": round(aurocs_m["m1"], 5),
                "m2_auroc": round(aurocs_m["m2"], 5),
                "overall_auroc": round(ov_auc, 5),
            }

        tier_res[tier] = {
            "ho_fpr_total": round(ho_fpr_tot, 5),
            "ho_fpr_m1": round(ho_fpr_m1, 5),
            "ho_fpr_m2": round(ho_fpr_m2, 5),
            "dev_fpr_m1_at_5": round(float(np.mean(dev_ben_m1 > th_dev_5)), 5),
            "dev_fpr_m2_at_5": round(float(np.mean(dev_ben_m2 > th_dev_5)), 5),
            "mean_tpr_5pct": round(float(np.mean(tprs_all)), 5),
            "mean_auroc": round(float(np.mean(aurocs_all)), 5),
            "per_attack": per_atk,
        }

    return {"tier_res": tier_res, "tier_calib": tier_calib}


def search_and_freeze_method(
    fit_clean_bank: dict,
    fit_clean_meta: list,
    fit_dev_atk_bank: dict,
    fit_dev_atk_meta: list,
    stats: dict,
    method_kind: str,
) -> dict:
    """在纯 Fit 空间枚举候选离散配置，锁定并冻结最优方案（独立搜索主方法与消融方法）。"""
    q_list = [95, 98]
    k_list = [8, 12, 16]
    w_list = [
        [0.34, 0.33, 0.33],
        [0.20, 0.40, 0.40],
        [0.25, 0.35, 0.40],
        [0.40, 0.30, 0.30],
    ]
    fnorm_list = ["div_p95", "med_p95"]
    gamma_cos_list = [
        [1.0, 1.0, 1.0],
        [1.08, 1.0, 0.92],
        [1.15, 1.0, 0.85],
    ]
    if method_kind == "ablation_ms_cos":
        sem_pairs = [(0.0, [1.0, 1.0, 1.0])]
    else:
        w_sem_list = [0.25, 0.35, 0.45, 0.60]
        g_sem_list = [
            [1.50, 0.80, 0.15],
            [1.80, 0.70, 0.10],
            [1.40, 0.90, 0.25],
            [1.60, 0.85, 0.20],
            [1.50, 0.90, 0.35],
            [1.40, 0.90, 0.40],
        ]
        sem_pairs = [(ws, gs) for ws in w_sem_list for gs in g_sem_list]

    best_obj = -1e9
    best_entry = None
    best_ho_obj = -1e9
    best_ho_entry = None
    best_fallback_obj = -1e9
    best_fallback = None
    total_checked = 0

    for q_scale in q_list:
        for top_k in k_list:
            for cos_w in w_list:
                for fnm in fnorm_list:
                    for gc in gamma_cos_list:
                        for w_sem, gs in sem_pairs:
                            cfg = {
                                "q_scale": q_scale,
                                "top_k": top_k,
                                "cos_weights": cos_w,
                                "frame_norm_mode": fnm,
                                "gamma_cos": gc,
                                "w_sem": w_sem,
                                "gamma_sem": gs,
                            }
                            ev = _calibrate_and_eval_fit(
                                fit_clean_bank, fit_clean_meta, fit_dev_atk_bank, fit_dev_atk_meta,
                                stats, method_kind, cfg
                            )
                            total_checked += 1
                            t4 = ev["tier_res"]["tier4_sensor_dist_vis"]
                            t2 = ev["tier_res"]["tier2_sensor_only"]
                            ho_imb = abs(t4["ho_fpr_m1"] - t4["ho_fpr_m2"])
                            ho_over = max(0.0, t4["ho_fpr_m1"] - 0.12) + max(0.0, t4["ho_fpr_m2"] - 0.12)
                            obj = (
                                0.55 * t4["mean_tpr_5pct"]
                                + 0.35 * t4["mean_auroc"]
                                - 0.25 * ho_imb
                                - 0.50 * ho_over
                            )

                            if method_kind == "main_ms_cos_clear_sem":
                                obs_atks = ("scaled_pgd_0.5sigma", "scaled_pgd_1.0sigma", "local_removal_1.0sigma")
                                t4_obs_auc = float(np.mean([t4["per_attack"][a]["overall_auroc"] for a in obs_atks]))
                                t2_obs_auc = float(np.mean([t2["per_attack"][a]["overall_auroc"] for a in obs_atks]))
                                feasible = (
                                    t4["mean_tpr_5pct"] > t2["mean_tpr_5pct"]
                                    and t4_obs_auc >= t2_obs_auc - 1e-4
                                    and t4["ho_fpr_m1"] <= 0.15
                                    and t4["ho_fpr_m2"] <= 0.15
                                )
                                ho_ok = (t4["ho_fpr_m1"] <= 0.15 and t4["ho_fpr_m2"] <= 0.15)
                            else:
                                feasible = (
                                    t4["mean_tpr_5pct"] >= t2["mean_tpr_5pct"] - 1e-5
                                    and t4["ho_fpr_m1"] <= 0.20
                                    and t4["ho_fpr_m2"] <= 0.20
                                )
                                ho_ok = (t4["ho_fpr_m1"] <= 0.20 and t4["ho_fpr_m2"] <= 0.20)

                            if feasible and obj > best_obj:
                                best_obj = obj
                                best_entry = {"cfg": cfg, "fit_eval": ev["tier_res"], "tier_calib": ev["tier_calib"], "obj": round(obj, 6), "feasible": True, "ho_ok": True}
                            if ho_ok and obj > best_ho_obj:
                                best_ho_obj = obj
                                best_ho_entry = {"cfg": cfg, "fit_eval": ev["tier_res"], "tier_calib": ev["tier_calib"], "obj": round(obj, 6), "feasible": feasible, "ho_ok": True}
                            if obj > best_fallback_obj:
                                best_fallback_obj = obj
                                best_fallback = {"cfg": cfg, "fit_eval": ev["tier_res"], "tier_calib": ev["tier_calib"], "obj": round(obj, 6), "feasible": feasible, "ho_ok": ho_ok}

    if best_entry is not None:
        chosen = best_entry
    elif method_kind == "ablation_ms_cos" and best_ho_entry is not None:
        chosen = best_ho_entry
    else:
        chosen = best_fallback
    chosen["total_grid_configs_checked"] = total_checked
    return chosen


def main():
    t0 = time.time()
    wrapper = HEALBaseWrapper(device="cuda")
    ds_m1 = wrapper.build_dataset(data_dir=TRAIN_FULL19_DIR, use_cav=2, visualize=True, collab_modality="m1")
    ds_m2 = wrapper.build_dataset(data_dir=TRAIN_FULL19_DIR, use_cav=2, visualize=True, collab_modality="m2")

    split_plan = build_and_save_split_plan(wrapper, ds_m2, OUT_SPLIT, TRAIN_FULL19_DIR, n_fit=100, n_fit_dev=36, n_eval=80)
    fit_indices = list(split_plan["fit_indices"])
    fit_dev_set = set(split_plan["fit_dev_indices"])
    fit_idx_set = set(fit_indices)
    eval_idx_set = set(split_plan["eval_indices"])
    assert fit_idx_set.isdisjoint(eval_idx_set), "Fit 与 Eval 索引存在交集！"
    print(f"[Phase 1] Fit={len(fit_indices)} 帧 (Dev={len(fit_dev_set)}, Holdout={len(fit_indices)-len(fit_dev_set)}), Eval={len(eval_idx_set)} 帧（严禁访问）")

    gx, gy = build_bev_grid_xy_256(device="cuda", dtype=torch.float32)

    fit_clean_records = []
    fit_clean_meta = []
    fit_clean_devs = []

    fit_dev_atk_meta = []
    fit_dev_atk_devs = []
    fit_atk_eff = {m: {a: {"valid": 0, "total": 0, "linf": []} for a in FIT_ATTACKS} for m in ("m1", "m2")}

    for rank, idx in enumerate(fit_indices):
        # 代码级零泄露断言：绝不允许读取任何 Eval 索引
        assert idx in fit_idx_set and idx not in eval_idx_set, f"越界读取非 Fit 样本: {idx}"
        b = extract_frame_context_and_clean(wrapper, ds_m1, ds_m2, idx, gx, gy)
        is_dev = idx in fit_dev_set
        is_ho = not is_dev

        cov_m1_cpu = b["cov_ego_m1_full"][0, 0].detach().cpu()
        cov_m2_cpu = b["cov_ego"][0, 0].detach().cpu()
        cov_by_mod_cpu = {"m1": cov_m1_cpu, "m2": cov_m2_cpu}
        d_cpu = b["d_idx"].detach().cpu()
        v_cpu = b["v_idx"].detach().cpu()

        fit_clean_records.append({
            "sample_idx": idx,
            "cov_by_mod": cov_by_mod_cpu,
            "d_idx": d_cpu,
            "v_idx": v_cpu,
            "m1": b["dev_clean"]["m1"].detach().cpu(),
            "m2": b["dev_clean"]["m2"].detach().cpu(),
        })
        for mod in ("m1", "m2"):
            fit_clean_meta.append({"sample_idx": idx, "mod": mod, "is_holdout": is_ho, "cov_2d": cov_by_mod_cpu[mod], "d_idx": d_cpu, "v_idx": v_cpu})
            fit_clean_devs.append(b["dev_clean"][mod].detach().cpu())

        if is_dev:
            for mod, p_dict in (("m1", b["p1"]), ("m2", b["p2"])):
                feat = p_dict["heter_feature_2d"]
                rlen = p_dict["record_len"]
                amat = p_dict["affine_matrix"]
                mods = p_dict["agent_modality_list"]
                vmask = p_dict["cam_valid_mask"]
                # 主实验规范：m1 使用自身完整覆盖范围且不限制攻击区 (preg=None)
                cov_4d_mod = b["cov_ego_m1_full"] if mod == "m1" else b["cov_ego"]
                preg = None

                f_05, st_05 = generate_scaled_pgd_removal_heal(wrapper, feat, rlen, amat, mods, vmask, 1, 0.5, 10, preg)
                f_10, st_10 = generate_scaled_pgd_removal_heal(wrapper, feat, rlen, amat, mods, vmask, 1, 1.0, 10, preg)
                f_lr, st_lr = generate_local_removal_heal(wrapper, feat, rlen, amat, mods, vmask, 1, 2, 5, 1.0, 12, preg)
                f_lg, st_lg = generate_local_ghost_spoofing_heal(wrapper, feat, rlen, amat, mods, vmask, 1, 2, 5, 2.0, 12, idx, preg)
                f_n6, st_n6 = generate_local_near0_6m_attack_heal(wrapper, feat, rlen, amat, mods, vmask, 1, 2.0, 15, idx, preg, False)

                for aname, f_adv, st in (
                    ("scaled_pgd_0.5sigma", f_05, st_05),
                    ("scaled_pgd_1.0sigma", f_10, st_10),
                    ("local_removal_1.0sigma", f_lr, st_lr),
                    ("local_ghost_2.0sigma", f_lg, st_lg),
                    ("local_near0_6m_removal_2.0sigma", f_n6, st_n6),
                ):
                    if aname == "local_removal_1.0sigma":
                        is_val = bool(st.get("perturb_grids", 0) > 0)
                    elif aname == "local_ghost_2.0sigma":
                        is_val = bool(st.get("num_spoof_anchors", 0) > 0 and st.get("perturb_grids", 0) > 0)
                    elif aname == "local_near0_6m_removal_2.0sigma":
                        is_val = bool(st.get("valid", False))
                    else:
                        is_val = True

                    fit_atk_eff[mod][aname]["total"] += 1
                    if is_val:
                        fit_atk_eff[mod][aname]["valid"] += 1
                        fit_atk_eff[mod][aname]["linf"].append(float(st.get("linf_max", 0.0)))

                    dev_a = compute_dev_maps_from_protocol(wrapper, p_dict, f_adv, b["p_ego_cls"], cov_4d_mod).detach().cpu()
                    fit_dev_atk_meta.append({
                        "sample_idx": idx,
                        "mod": mod,
                        "attack": aname,
                        "valid": is_val,
                        "cov_2d": cov_by_mod_cpu[mod],
                        "d_idx": d_cpu,
                        "v_idx": v_cpu,
                    })
                    fit_dev_atk_devs.append(dev_a)

        if (rank + 1) % 20 == 0 or (rank + 1) == len(fit_indices):
            print(f"  已处理 Fit 样本 {rank + 1}/{len(fit_indices)} ({time.time() - t0:.1f}s)")

    print("[Phase 1] 正在计算 Fit-100 全像素 5距离×3可见性 精确分档分布...")
    channel_stats, benign_summary = compute_pixel_tier_stats(fit_clean_records)
    del fit_clean_records

    print("[Phase 1] 正在 GPU 预计算标准残差银行并枚举主方法与消融超参...")
    torch.cuda.empty_cache()
    fit_clean_bank = _precompute_z_bank(fit_clean_meta, fit_clean_devs, channel_stats, device="cuda")
    fit_dev_atk_bank = _precompute_z_bank(fit_dev_atk_meta, fit_dev_atk_devs, channel_stats, device="cuda")

    chosen_main = search_and_freeze_method(fit_clean_bank, fit_clean_meta, fit_dev_atk_bank, fit_dev_atk_meta, channel_stats, "main_ms_cos_clear_sem")
    chosen_ablation = search_and_freeze_method(fit_clean_bank, fit_clean_meta, fit_dev_atk_bank, fit_dev_atk_meta, channel_stats, "ablation_ms_cos")

    # 检验 Re-plan if #3 条件并打印 Fit-Dev 4 组门限明细
    for label, ch in (("main_ms_cos_clear_sem", chosen_main), ("ablation_ms_cos", chosen_ablation)):
        print(f"\n=== 冻结结果 [{label}] (grid_checked={ch['total_grid_configs_checked']}, feasible={ch['feasible']}, ho_ok={ch['ho_ok']}) ===")
        print(f"  配置: {ch['cfg']}")
        for tname in TIERS_4:
            tr = ch["fit_eval"][tname]
            atk_s = " | ".join([f"{a[:12]}:{tr['per_attack'][a]['tpr@5']:.3f}/{tr['per_attack'][a]['overall_auroc']:.3f}" for a in FIT_ATTACKS])
            print(f"  [{tname:21s}] Dev MeanTPR={tr['mean_tpr_5pct']:.4f} MeanAUC={tr['mean_auroc']:.4f} | HO FPR(m1/m2)={tr['ho_fpr_m1']:.4f}/{tr['ho_fpr_m2']:.4f} | {atk_s}")
        if label == "main_ms_cos_clear_sem":
            assert ch["feasible"], f"[{label}] 触发 Re-plan if: 主方法未满足 Tier4>Tier2 且 Holdout<=0.15 条件！"
        else:
            assert ch["ho_ok"], f"[{label}] 触发 Re-plan if: 消融方法 Holdout 误报越界！"

    out_obj = OrderedDict({
        "meta": {
            "description": "步骤 ④-1 修正版纯 Fit 冻结参数文件（零 Eval 泄露）",
            "fit_total_frames": len(fit_indices),
            "fit_dev_frames": len(fit_dev_set),
            "fit_holdout_frames": len(fit_indices) - len(fit_dev_set),
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "channel_tier_stats": channel_stats,
        "benign_dist_vis_summary": benign_summary,
        "fit_dev_attack_stats": {
            m: {a: {"valid": v["valid"], "total": v["total"], "linf_mean": round(float(np.mean(v["linf"])) if v["linf"] else 0.0, 4)} for a, v in fit_atk_eff[m].items()}
            for m in ("m1", "m2")
        },
        "frozen_methods": {
            "main_ms_cos_clear_sem": chosen_main,
            "ablation_ms_cos": chosen_ablation,
        },
    })
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_CALIB, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)
    print(f"\n[Phase 1 完成] 冻结参数已写入: {OUT_CALIB} (耗时 {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
