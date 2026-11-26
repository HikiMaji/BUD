# -*- coding: utf-8 -*-
"""
A5（含 A1、B2 标定规则同步）：Fit 11 条序列严格拆分为 7 条纯标定 + 4 条纯调参（零序列重叠）统一脚本。

严格执行流程：
1. 生成并保存 results_v3/a5_split_plan.json（7 条标定序列 vs 4 条调参序列 vs 8 条 Dev 序列，两两零交集）；
2. 仅在 7 条标定序列（FIT_CALIB_7）干净数据上统计：
   - 4 通道 (传感器 × 5距离 × 3可见性) mu_l, s_l（含 grid/frame/seq 计数、Bootstrap 95% CI、稀疏桶自动向上合并）；
   - 帧级稳健分母 Q_frame (div_p95) 与高门限基准；
   - (传感器 × 3可见性 × 前景/背景) 阈值 Q（同步使用 B2 新前景规则，输出计数、Bootstrap 95% CI 与稀疏桶合并记录）；
3. 仅在 4 条独立调参序列（FIT_TUNE_4）带攻击数据上做网格搜索，完整记录搜索范围、目标函数公式与尝试配置数；
4. 输出 results_v3/a5_base_calibrator.json 与 results_v3/a5_bud_calibrator.json（旧 JSON 绝不覆盖）。
"""

import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

PROJ_ROOT = "/root/autodl-tmp/proj_dir2_hetero_trust"
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from scripts_v2_mm.step4_4_fit_blind_diag_and_recalib import (
    FIT_DEV_STATES,
    _compute_dev_signed_and_pfused,
    _eval_fit_dev_candidate,
    _find_blind_gts_for_frame,
    _slim_batch_for_postprocess,
)
from src_v2_mm.a5_split_and_calib_stats import (
    FIT_CALIB_SEQ_NAMES_7,
    FIT_TUNE_SEQ_NAMES_4,
    assert_disjoint_a5_splits,
    build_fg_bg_vis_quantiles_with_ci_and_merging,
    build_hierarchical_channel_stats_with_merging,
)
from src_v2_mm.bud_fusion_heal import BUDFusionHEALAuditor, build_fg_mask
from src_v2_mm.heal_attacks import (
    generate_local_ghost_spoofing_heal,
    generate_local_near0_6m_attack_heal,
    generate_local_occluded_ghost_heal,
    generate_local_removal_heal,
    generate_scaled_pgd_removal_heal,
)
from src_v2_mm.heal_base_wrapper import HEALBaseWrapper
from src_v2_mm.step4_1_v2_core import (
    DEFAULT_CLEAR_PTS_TAU,
    DEFAULT_LOS_OCC_TAU,
    EVAL_SEQ_NAMES,
    PROTO_H,
    PROTO_W,
    TRAIN_FULL19_DIR,
    VIS_LABELS_3,
    _alloc_balanced_indices,
    build_bev_grid_xy_256,
    extract_frame_context_and_clean,
)

RES_V3 = os.path.join(PROJ_ROOT, "results_v3")
DEFAULT_OLD_BASE = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_1_v2_frozen_calibrator.json")


def main():
    parser = argparse.ArgumentParser(description="A5：7 条标定序列 + 4 条调参序列严格解耦标定与网格搜索")
    parser.add_argument("--data_dir", type=str, default=TRAIN_FULL19_DIR)
    parser.add_argument("--n_calib_frames", type=int, default=105, help="从 7 条标定序列均衡抽取的干净标定总帧数")
    parser.add_argument("--n_tune_frames", type=int, default=40, help="从 4 条调参序列均衡抽取的带攻击调参总帧数")
    parser.add_argument("--min_bin_frames", type=int, default=20, help="分档最低有效帧数，低于此值向上一级合并")
    parser.add_argument("--fg_mode", type=str, default="v1", choices=["v0", "v1"], help="B2 前景判定规则（默认 v1 以本车为准）")
    parser.add_argument("--fg_prob_tau", type=float, default=0.20)
    parser.add_argument("--ref_base_json", type=str, default=DEFAULT_OLD_BASE, help="读取已冻结的通道权重结构模板")
    parser.add_argument("--out_dir", type=str, default=RES_V3)
    args = parser.parse_args()

    t0 = time.time()
    assert_disjoint_a5_splits()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.ref_base_json, "r", encoding="utf-8") as f:
        ref_base = json.load(f)

    wrapper = HEALBaseWrapper(device="cuda")
    ds_m1 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m1")
    ds_m2 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m2")
    gx, gy = build_bev_grid_xy_256(device="cuda", dtype=torch.float32)

    sample_seqs = wrapper.get_sample_sequences(ds_m2)
    seq_to_idxs = OrderedDict()
    for i, s in enumerate(sample_seqs):
        seq_to_idxs.setdefault(s, []).append(i)

    calib_items, calib_plan = _alloc_balanced_indices(
        FIT_CALIB_SEQ_NAMES_7, seq_to_idxs, args.data_dir, int(args.n_calib_frames), min_per_seq=10
    )
    tune_items, tune_plan = _alloc_balanced_indices(
        FIT_TUNE_SEQ_NAMES_4, seq_to_idxs, args.data_dir, int(args.n_tune_frames), min_per_seq=8
    )

    split_a5 = OrderedDict({
        "description": "A5 严格按序列解耦划分：7 条标定序列 vs 4 条调参序列（零重叠）",
        "dataset_dir": args.data_dir,
        "calib_7_sequences": calib_plan,
        "tune_4_sequences": tune_plan,
        "eval_8_sequences_untouched": EVAL_SEQ_NAMES,
        "calib_indices": [x[0] for x in calib_items],
        "tune_indices": [x[0] for x in tune_items],
        "counts": {"calib_seqs": 7, "calib_frames": len(calib_items), "tune_seqs": 4, "tune_frames": len(tune_items)},
    })
    out_split = os.path.join(args.out_dir, "a5_split_plan.json")
    with open(out_split, "w", encoding="utf-8") as f:
        json.dump(split_a5, f, indent=2, ensure_ascii=False)
    print(f"[A5 Step 1] 已保存 7/4 序列解耦划分至 {out_split}", flush=True)

    # === Step 2: 在 7 条标定序列干净帧上统计 4 通道分档表与可见性阈值 ===
    from opencood.tools import train_utils

    calib_records = []
    calib_fg_cache = []
    unocc_blind_exp, unocc_hit_exp = [], []

    for rank, (idx, seq_name, _) in enumerate(calib_items):
        b = extract_frame_context_and_clean(wrapper, ds_m1, ds_m2, idx, gx, gy)
        cov_m1 = b["cov_ego_m1_full"][0, 0].detach().cpu()
        cov_m2 = b["cov_ego"][0, 0].detach().cpu()
        d_cpu = b["d_idx"].detach().cpu()
        v_cpu = b["v_idx"].detach().cpu()

        raw_b1 = train_utils.to_device(ds_m1.collate_batch_test([ds_m1[idx]]), "cpu")
        b_post = train_utils.to_device(_slim_batch_for_postprocess(raw_b1), wrapper.device_obj)
        with torch.no_grad():
            ego_out = wrapper.forward_single_agent(b["p1"]["heter_feature_2d"][0:1])
            ego_pb, ego_ps, ego_gb = wrapper.decode_boxes(b_post, ego_out, ds_m1)
            o1 = wrapper.forward_from_protocol(b["p1"]["heter_feature_2d"], b["p1"]["record_len"], b["p1"]["affine_matrix"], b["p1"]["agent_modality_list"])
            o2 = wrapper.forward_from_protocol(b["p2"]["heter_feature_2d"], b["p2"]["record_len"], b["p2"]["affine_matrix"], b["p2"]["agent_modality_list"])
            pf1 = torch.sigmoid(o1["cls_preds"]).max(dim=1, keepdim=True)[0]
            pf2 = torch.sigmoid(o2["cls_preds"]).max(dim=1, keepdim=True)[0]
            fg1 = build_fg_mask(b["p_ego_cls"], pf1, b["v_idx"], fg_prob_tau=args.fg_prob_tau, mode=args.fg_mode).detach().cpu()
            fg2 = build_fg_mask(b["p_ego_cls"], pf2, b["v_idx"], fg_prob_tau=args.fg_prob_tau, mode=args.fg_mode).detach().cpu()

        binfo = _find_blind_gts_for_frame(ds_m1, idx, ego_pb, ego_ps, ego_gb, b["cov_ego"][0, 0])
        los_2d = b["los_factor"][0, 0].detach().cpu()
        exp_2d = b["expected_pts"][0, 0].detach().cpu()
        for iy, ix in binfo["blind_yx"]:
            if float(los_2d[iy, ix]) >= DEFAULT_LOS_OCC_TAU:
                unocc_blind_exp.append(float(exp_2d[iy, ix]))
        for iy, ix in binfo["ego_hit_yx"]:
            if float(los_2d[iy, ix]) >= DEFAULT_LOS_OCC_TAU:
                unocc_hit_exp.append(float(exp_2d[iy, ix]))

        rec_c = {
            "sample_idx": int(idx),
            "seq_name": seq_name,
            "cov_by_mod": {"m1": cov_m1, "m2": cov_m2},
            "d_idx": d_cpu,
            "v_idx": v_cpu,
            "m1": b["dev_clean"]["m1"].detach().cpu(),
            "m2": b["dev_clean"]["m2"].detach().cpu(),
            "fg_m1": fg1,
            "fg_m2": fg2,
        }
        calib_records.append(rec_c)
        if (rank + 1) % 25 == 0 or (rank + 1) == len(calib_items):
            print(f"  [Calib-7] 抽取进度 {rank + 1}/{len(calib_items)} ({time.time() - t0:.1f}s)", flush=True)

    ch_stats, merge_diag = build_hierarchical_channel_stats_with_merging(
        calib_records, FIT_CALIB_SEQ_NAMES_7, min_frames=int(args.min_bin_frames), n_boot=2000, seed=20261120
    )
    out_base = os.path.join(args.out_dir, "a5_base_calibrator.json")
    base_obj = OrderedDict({
        "meta": {
            "description": "A5 纯 7 条标定序列统计的 4 通道基准（含计数、Bootstrap 95% CI 与稀疏桶合并）",
            "calib_sequences": FIT_CALIB_SEQ_NAMES_7,
            "calib_frames": len(calib_records),
            "visibility_fn": "assign_dist5_vis3_indices_v2",
            "bin_merging_summary": merge_diag,
        },
        "channel_tier_stats": ch_stats,
        "frozen_methods": ref_base["frozen_methods"],
    })
    with open(out_base, "w", encoding="utf-8") as f:
        json.dump(base_obj, f, indent=2, ensure_ascii=False)
    print(f"[A5 Step 2] 已保存基础标定表 {out_base} (合并稀疏桶数={merge_diag['num_merged_bins']})", flush=True)

    # 基于 7 条标定序列计算 Q_frame (div_p95) 与 (传感器×可见性×前景/背景) 阈值 Q
    auditor = BUDFusionHEALAuditor(out_base)
    raw_scores = {"t2": {"m1": [], "m2": []}, "t4": {"m1": [], "m2": []}}
    per_seq_bg_fg = {
        role: {m: {vn: OrderedDict() for vn in VIS_LABELS_3} for m in ("m1", "m2")}
        for role in ("bg", "fg")
    }
    fg_bg_frame_cnts = {
        role: {m: {vn: 0 for vn in VIS_LABELS_3} for m in ("m1", "m2")}
        for role in ("bg", "fg")
    }

    for rec in calib_records:
        s_name = rec["seq_name"]
        d_g = rec["d_idx"].cuda()
        v_g = rec["v_idx"].cuda()
        for mod, fg_k in (("m1", "fg_m1"), ("m2", "fg_m2")):
            cov_g = rec["cov_by_mod"][mod].cuda()
            fg_g = rec[fg_k].cuda()
            dev_g = rec[mod].cuda().float()
            ar = auditor.audit_dev_maps(dev_g, cov_g, d_g, v_g, mod, build_gate=False)
            raw_scores["t2"][mod].append(ar["raw_t2"])
            raw_scores["t4"][mod].append(ar["raw_t4"])
            cm_s = F.avg_pool2d(ar["cos_comb_t4"].view(1, 1, PROTO_H, PROTO_W), kernel_size=3, stride=1, padding=1)[0, 0]
            for vi, vn in enumerate(VIS_LABELS_3):
                m_bg = cov_g & (v_g == vi) & (~fg_g)
                m_fg = cov_g & (v_g == vi) & fg_g
                if m_bg.any():
                    per_seq_bg_fg["bg"][mod][vn].setdefault(s_name, []).append(cm_s[m_bg].detach().cpu().numpy().astype(np.float32))
                    fg_bg_frame_cnts["bg"][mod][vn] += 1
                if m_fg.any():
                    per_seq_bg_fg["fg"][mod][vn].setdefault(s_name, []).append(cm_s[m_fg].detach().cpu().numpy().astype(np.float32))
                    fg_bg_frame_cnts["fg"][mod][vn] += 1

    frame_div_p95 = OrderedDict({
        "tier2_sensor_only": {m: round(max(float(np.percentile(raw_scores["t2"][m], 95.0)), 1e-4), 6) for m in ("m1", "m2")},
        "tier4_sensor_dist_vis": {m: round(max(float(np.percentile(raw_scores["t4"][m], 95.0)), 1e-4), 6) for m in ("m1", "m2")},
    })
    cos_q_bg, cos_q_fg, merged_q_log = build_fg_bg_vis_quantiles_with_ci_and_merging(
        per_seq_bg_fg, fg_bg_frame_cnts, FIT_CALIB_SEQ_NAMES_7,
        min_frames=int(args.min_bin_frames), n_boot=2000, seed=20261120,
    )

    # 整车回退高门限仅由 7 条标定序列干净帧上界安全裕度确定（零触碰 4 条调参序列）
    frame_high_tau = OrderedDict()
    for m in ("m1", "m2"):
        cln_sc = np.asarray(raw_scores["t4"][m], dtype=np.float64) / frame_div_p95["tier4_sensor_dist_vis"][m]
        frame_high_tau[m] = round(max(float(cln_sc.max()) * 1.15, 1.45), 6)

    # === Step 3: 在完全独立的 4 条调参序列（FIT_TUNE_4）上构造攻击并网格搜索门控超参 ===
    tune_eval_items = []
    blind_meta_by_idx = {}
    ego_only_cache = {}
    no_def_cache = {}

    for rank, (idx, seq_name, _) in enumerate(tune_items):
        b = extract_frame_context_and_clean(wrapper, ds_m1, ds_m2, idx, gx, gy)
        raw_b1 = train_utils.to_device(ds_m1.collate_batch_test([ds_m1[idx]]), "cpu")
        b_slim = _slim_batch_for_postprocess(raw_b1)
        b_post = train_utils.to_device(b_slim, wrapper.device_obj)
        with torch.no_grad():
            ego_out = wrapper.forward_single_agent(b["p1"]["heter_feature_2d"][0:1])
            ego_pb, ego_ps, ego_gb = wrapper.decode_boxes(b_post, ego_out, ds_m1)
        ego_only_cache[idx] = (ego_pb, ego_ps, ego_gb)
        blind_meta_by_idx[idx] = _find_blind_gts_for_frame(ds_m1, idx, ego_pb, ego_ps, ego_gb, b["cov_ego"][0, 0])

        occ_mask_4d = b["occ_mask_ego"].float()
        for mod, p_dict in (("m1", b["p1"]), ("m2", b["p2"])):
            feat, rlen, amat = p_dict["heter_feature_2d"], p_dict["record_len"], p_dict["affine_matrix"]
            mods_l, vmask = list(p_dict["agent_modality_list"]), p_dict["cam_valid_mask"]
            cov_4d = b["cov_ego_m1_full"] if mod == "m1" else b["cov_ego"]
            cov_2d = cov_4d[0, 0]
            preg = None

            f_lr, st_lr = generate_local_removal_heal(wrapper, feat, rlen, amat, mods_l, vmask, 1, 2, 5, 1.0, 10, preg)
            f_lg, st_lg = generate_local_ghost_spoofing_heal(wrapper, feat, rlen, amat, mods_l, vmask, 1, 2, 5, 2.0, 10, idx, preg)
            f_lo, st_lo = generate_local_occluded_ghost_heal(wrapper, feat, rlen, amat, mods_l, vmask, occ_mask_4d, 1, 2, 5, 2.0, 12, idx, preg)
            f_ln, st_ln = generate_local_near0_6m_attack_heal(wrapper, feat, rlen, amat, mods_l, vmask, 1, 2.0, 12, idx, preg, False)
            f_p1, st_p1 = generate_scaled_pgd_removal_heal(wrapper, feat, rlen, amat, mods_l, vmask, 1, 1.0, 8, preg)

            states_map = [
                ("clean", feat, True, torch.zeros((PROTO_H, PROTO_W), dtype=torch.bool)),
                ("local_removal_1.0sigma", f_lr, st_lr.get("perturb_grids", 0) > 0, st_lr["attacked_mask_in_ego"][0, 0].detach().cpu() > 0.5),
                ("local_ghost_2.0sigma", f_lg, st_lg.get("num_spoof_anchors", 0) > 0, st_lg["attacked_mask_in_ego"][0, 0].detach().cpu() > 0.5),
                ("local_occluded_ghost_2.0sigma", f_lo, st_lo.get("valid", False), st_lo["attacked_mask_in_ego"][0, 0].detach().cpu() > 0.5),
                ("local_near0_6m_2.0sigma", f_ln, st_ln.get("valid", False), st_ln["attacked_mask_in_ego"][0, 0].detach().cpu() > 0.5),
                ("scaled_pgd_1.0sigma", f_p1, True, cov_2d.detach().cpu()),
            ]
            for st_name, f_cur, is_v, atk_m in states_map:
                dev_a, dd_a, da_a, pf_a = _compute_dev_signed_and_pfused(wrapper, p_dict, f_cur, b["p_ego_cls"], cov_4d, k_tol=5)
                fg_m_a = build_fg_mask(b["p_ego_cls"], pf_a, b["v_idx"], fg_prob_tau=args.fg_prob_tau, mode=args.fg_mode)
                ar_a = auditor.audit_dev_maps(dev_a, cov_2d, b["d_idx"], b["v_idx"], mod, build_gate=False)
                tune_eval_items.append({
                    "sample_idx": int(idx),
                    "seq_name": seq_name,
                    "mod": mod,
                    "state": st_name,
                    "feat": f_cur.half().cpu(),
                    "rlen": rlen.detach().cpu(),
                    "amat": amat.detach().cpu(),
                    "mods_l": mods_l,
                    "batch_slim": b_slim,
                    "cov_2d": cov_2d.detach().cpu(),
                    "d_idx": b["d_idx"].detach().cpu(),
                    "v_idx_v2": b["v_idx"].detach().cpu(),
                    "cos_comb_t4": ar_a["cos_comb_t4"].detach().cpu(),
                    "raw_t4": float(ar_a["raw_t4"]),
                    "score_t4": float(ar_a["raw_t4"]) / frame_div_p95["tier4_sensor_dist_vis"][mod],
                    "d_drop_tol": dd_a.detach().cpu(),
                    "d_add_tol": da_a.detach().cpu(),
                    "fg_mask": fg_m_a.detach().cpu(),
                    "atk_mask_2d": atk_m,
                    "atk_valid": bool(is_v),
                })
        if (rank + 1) % 10 == 0 or (rank + 1) == len(tune_items):
            print(f"  [Tune-4] 构造进度 {rank + 1}/{len(tune_items)} ({time.time() - t0:.1f}s)", flush=True)

    base_ap50, base_ap70, base_brec, _, _ = _eval_fit_dev_candidate(
        wrapper, ds_m1, tune_eval_items, blind_meta_by_idx,
        g_params=None, no_def_cache=no_def_cache, ego_only_cache=ego_only_cache,
    )

    search_space = OrderedDict({
        "bg_q": ["p99.8", "p99.9"],
        "fg_q": ["p98.0", "p99.0", "p99.5"],
        "gamma_cos_gate": [[1.0, 1.0, 1.0], [1.0, 0.85, 0.70]],
        "r_dil_and_beta": [[3, 3.5], [4, 4.0]],
    })
    objective_formula = (
        "J = mean_atk_ap50 + 0.12 * mean_occ_auroc "
        "- 0.80 * max(0, max_clean_blind_recall_loss - 0.03) "
        "- 1.50 * max(0, max_clean_ap50_loss - 0.01)"
    )

    all_records = []
    best_strict, best_strict_obj = None, -1e9
    best_overall, best_overall_obj = None, -1e9
    atk_keys = [k for k in FIT_DEV_STATES if k != "clean"]

    for bq in search_space["bg_q"]:
        for fq in search_space["fg_q"]:
            cos_tau_bg = {m: [cos_q_bg[m][vn][bq] for vn in VIS_LABELS_3] for m in ("m1", "m2")}
            cos_tau_fg = {m: [max(cos_q_bg[m][vn][bq], cos_q_fg[m][vn][fq]) for vn in VIS_LABELS_3] for m in ("m1", "m2")}
            for gc in search_space["gamma_cos_gate"]:
                for rd, bt in search_space["r_dil_and_beta"]:
                    g_params = {
                        "mode": "directional_vis_fg_bg",
                        "fg_rule": args.fg_mode,
                        "k_tol": 5,
                        "fg_prob_tau": float(args.fg_prob_tau),
                        "bg_q": bq,
                        "fg_q": fq,
                        "cos_tau_by_vis_bg": cos_tau_bg,
                        "cos_tau_by_vis_fg": cos_tau_fg,
                        "gamma_cos_gate": gc,
                        "cos_co_ratio": 0.82,
                        "drop_tau": 0.40,
                        "add_tau": 0.30,
                        "w_drop": 7.0,
                        "w_add": 6.0,
                        "gamma_add": [1.0, 0.3, 0.0],
                        "r_dil": int(rd),
                        "beta": float(bt),
                    }
                    ap50, ap70, brec, gst, occ_det = _eval_fit_dev_candidate(
                        wrapper, ds_m1, tune_eval_items, blind_meta_by_idx,
                        g_params=g_params, no_def_cache=no_def_cache, ego_only_cache=ego_only_cache,
                        frame_high_tau=frame_high_tau, use_v_idx_key="v_idx_v2",
                    )
                    cln_ap_loss = {m: base_ap50[m]["clean"] - ap50[m]["clean"] for m in ("m1", "m2")}
                    cln_br_loss = {m: base_brec[m]["clean"] - brec[m]["clean"] for m in ("m1", "m2")}
                    max_ap_loss = max(cln_ap_loss["m1"], cln_ap_loss["m2"])
                    max_br_loss = max(cln_br_loss["m1"], cln_br_loss["m2"])
                    mean_atk_ap50 = float(np.mean([ap50[m][a] for m in ("m1", "m2") for a in atk_keys]))
                    mean_occ_auroc = 0.5 * (occ_det["m1"]["auroc"] + occ_det["m2"]["auroc"])
                    obj = mean_atk_ap50 + 0.12 * mean_occ_auroc - 0.80 * max(0.0, max_br_loss - 0.03) - 1.50 * max(0.0, max_ap_loss - 0.01)

                    rec = {
                        "gating_params": g_params,
                        "ap50": {m: {k: round(v, 5) for k, v in ap50[m].items()} for m in ("m1", "m2")},
                        "ap70": {m: {k: round(v, 5) for k, v in ap70[m].items()} for m in ("m1", "m2")},
                        "blind_recall": {m: {k: round(v, 5) for k, v in brec[m].items()} for m in ("m1", "m2")},
                        "clean_ap50_loss": {m: round(cln_ap_loss[m], 5) for m in ("m1", "m2")},
                        "clean_blind_recall_loss": {m: round(cln_br_loss[m], 5) for m in ("m1", "m2")},
                        "mean_atk_ap50": round(mean_atk_ap50, 5),
                        "occluded_ghost_detection": occ_det,
                        "obj": round(obj, 5),
                    }
                    all_records.append(rec)
                    if max_ap_loss <= 0.010 and max_br_loss <= 0.030 and obj > best_strict_obj:
                        best_strict_obj = obj
                        best_strict = rec
                    if obj > best_overall_obj:
                        best_overall_obj = obj
                        best_overall = rec

    chosen = best_strict if best_strict is not None else best_overall
    out_bud = os.path.join(args.out_dir, "a5_bud_calibrator.json")
    bud_obj = OrderedDict({
        "meta": {
            "description": "A5 严格解耦标定（7 标定序列统计 Q + 4 独立调参序列网格搜索）",
            "calib_7_sequences": FIT_CALIB_SEQ_NAMES_7,
            "tune_4_sequences": FIT_TUNE_SEQ_NAMES_4,
            "calib_frames": len(calib_records),
            "tune_frames": len(tune_items),
            "fg_rule_used_in_calibration": args.fg_mode,
            "grid_search_documentation": {
                "search_space": search_space,
                "num_configs_evaluated": len(all_records),
                "objective_formula": objective_formula,
                "strict_constraint_met": bool(best_strict is not None),
            },
            "fg_bg_merged_bins_log": merged_q_log,
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "visibility_v2": {"clear_pts_tau": DEFAULT_CLEAR_PTS_TAU, "los_occ_tau": DEFAULT_LOS_OCC_TAU},
        "frame_div_p95": frame_div_p95,
        "frame_high_tau": frame_high_tau,
        "pixel_cos_by_vis_bg_quantiles": cos_q_bg,
        "pixel_cos_by_vis_fg_quantiles": cos_q_fg,
        "gating_params": chosen["gating_params"],
        "tune4_chosen_eval": chosen,
        "tune4_grid_records": all_records,
    })
    with open(out_bud, "w", encoding="utf-8") as f:
        json.dump(bud_obj, f, indent=2, ensure_ascii=False)
    print(f"[A5 完成] 已保存解耦门控标定文件 {out_bud} (评估配置数={len(all_records)}, 耗时={time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
