# -*- coding: utf-8 -*-
"""
A5（修复内存堆积、文件名统一与显式 clear_pts_tau 升级版）：
标定序列与调参序列严格按父序列互斥解耦，流式逐帧评估网格搜索（零内存驻留全量特征张量）。

统一文件名输出：
- 基础与初始标定文件：results_v3/bud_calib_v1.json
- 调参后门控标定文件：results_v3/bud_tuned_calibrator.json
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
if os.path.exists(PROJ_ROOT) and PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)
LOCAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if LOCAL_ROOT not in sys.path:
    sys.path.insert(0, LOCAL_ROOT)

from scripts_v2_mm.step4_4_fit_blind_diag_and_recalib import (
    FIT_DEV_STATES,
    _compute_dev_signed_and_pfused,
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
from src_v2_mm.bud_fusion_heal import (
    BUDFusionHEALAuditor,
    build_directional_gate_map,
    build_fg_mask,
    build_gated_fuse_fn,
    compute_frame_high_tau_with_rule,
)
from src_v2_mm.heal_attacks import (
    generate_local_ghost_spoofing_heal,
    generate_local_near0_6m_attack_heal,
    generate_local_occluded_ghost_heal,
    generate_local_removal_heal,
    generate_scaled_pgd_removal_heal,
)
from src_v2_mm.heal_base_wrapper import HEALBaseWrapper
from src_v2_mm.step4_1_v2_core import (
    DEFAULT_LOS_OCC_TAU,
    EVAL_SEQ_NAMES,
    OPV2V_TRAIN_DIR,
    PROTO_H,
    PROTO_W,
    VIS_LABELS_3,
    _alloc_balanced_indices,
    assign_dist5_vis3_indices_v2,
    build_bev_grid_xy_256,
    calibrate_clear_pts_tau_from_samples,
    compute_ego_expected_pts_and_los_256,
    extract_frame_context_and_clean,
    load_ego_lidar_np,
)

RES_V3 = os.path.join(PROJ_ROOT if os.path.exists(PROJ_ROOT) else LOCAL_ROOT, "results_v3")
DEFAULT_OLD_BASE = os.path.join(PROJ_ROOT if os.path.exists(PROJ_ROOT) else LOCAL_ROOT, "results_rq1_v2_mm", "step4_1_v2_frozen_calibrator.json")


def main():
    parser = argparse.ArgumentParser(description="A5 升级版：流式低内存标定与网格搜索（统一文件名、无硬编码 73.853）")
    parser.add_argument("--data_dir", type=str, default=OPV2V_TRAIN_DIR)
    parser.add_argument("--n_calib_frames", type=int, default=105)
    parser.add_argument("--n_tune_frames", type=int, default=40)
    parser.add_argument("--min_bin_frames", type=int, default=20)
    parser.add_argument("--fg_mode", type=str, default="v1", choices=["v0", "v1"])
    parser.add_argument("--fg_prob_tau", type=float, default=0.20)
    parser.add_argument("--ref_base_json", type=str, default=DEFAULT_OLD_BASE)
    parser.add_argument("--out_dir", type=str, default=RES_V3)
    args = parser.parse_args()

    t0 = time.time()
    assert_disjoint_a5_splits()
    os.makedirs(args.out_dir, exist_ok=True)

    ref_frozen_methods = {
        "main_ms_cos_clear_sem": {
            "cfg": {
                "q_scale": 95,
                "top_k": 16,
                "cos_weights": [0.25, 0.35, 0.40],
                "frame_norm_mode": "div_p95",
                "gamma_cos": [1.0, 1.0, 1.0],
                "w_sem": 0.25,
                "gamma_sem": [1.4, 0.9, 0.4],
            }
        }
    }

    from opencood.tools import train_utils
    from opencood.utils import eval_utils

    wrapper = HEALBaseWrapper(device="cuda")
    ds_m1 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m1")
    ds_m2 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m2")
    gx, gy = build_bev_grid_xy_256(device="cuda", dtype=torch.float32)

    sample_seqs = wrapper.get_sample_sequences(ds_m2)
    seq_to_idxs = OrderedDict()
    for i, s in enumerate(sample_seqs):
        seq_to_idxs.setdefault(s, []).append(i)

    calib_items, _ = _alloc_balanced_indices(FIT_CALIB_SEQ_NAMES_7, seq_to_idxs, args.data_dir, int(args.n_calib_frames), min_per_seq=10)
    tune_items, _ = _alloc_balanced_indices(FIT_TUNE_SEQ_NAMES_4, seq_to_idxs, args.data_dir, int(args.n_tune_frames), min_per_seq=8)

    # 第一步：在标定序列干净帧上实测标定 clear_pts_tau（绝不使用旧 JPG 默认值 73.853）
    unocc_blind_exp, unocc_hit_exp = [], []
    excl_self = 0
    for idx, _, _ in calib_items:
        raw_b1 = train_utils.to_device(ds_m1.collate_batch_test([ds_m1[idx]]), "cpu")
        b_post = train_utils.to_device(_slim_batch_for_postprocess(raw_b1), wrapper.device_obj)
        p1 = wrapper.extract_protocol_features(train_utils.to_device(raw_b1, wrapper.device_obj)["ego"])
        with torch.no_grad():
            ego_out = wrapper.forward_single_agent(p1["heter_feature_2d"][0:1])
            ego_pb, ego_ps, ego_gb = wrapper.decode_boxes(b_post, ego_out, ds_m1)
        ego_pts = load_ego_lidar_np(ds_m1, idx)
        _, los_f, exp_pts = compute_ego_expected_pts_and_los_256(ego_pts, gx, gy, num_beams=64)
        binfo = _find_blind_gts_for_frame(ds_m1, idx, ego_pb, ego_ps, ego_gb, torch.ones((PROTO_H, PROTO_W), device=gx.device, dtype=torch.bool))
        excl_self += int(binfo.get("excluded_ego_self", 0))
        los_2d, exp_2d = los_f[0, 0].detach().cpu(), exp_pts[0, 0].detach().cpu()
        for iy, ix in binfo["blind_yx"]:
            if float(los_2d[iy, ix]) >= DEFAULT_LOS_OCC_TAU:
                unocc_blind_exp.append(float(exp_2d[iy, ix]))
        for iy, ix in binfo["ego_hit_yx"]:
            if float(los_2d[iy, ix]) >= DEFAULT_LOS_OCC_TAU:
                unocc_hit_exp.append(float(exp_2d[iy, ix]))

    vis_v2_cfg = calibrate_clear_pts_tau_from_samples(
        unocc_blind_exp, unocc_hit_exp, los_occ_tau=DEFAULT_LOS_OCC_TAU, excluded_ego_self_cnt=excl_self
    )
    c_tau = float(vis_v2_cfg["clear_pts_tau"])

    # 第二步：使用新标定的 c_tau 收集轻量 4 通道偏差统计量
    calib_records = []
    for idx, seq_name, _ in calib_items:
        b = extract_frame_context_and_clean(wrapper, ds_m1, ds_m2, idx, gx, gy, clear_pts_tau=c_tau, los_occ_tau=DEFAULT_LOS_OCC_TAU)
        with torch.no_grad():
            o1 = wrapper.forward_from_protocol(b["p1"]["heter_feature_2d"], b["p1"]["record_len"], b["p1"]["affine_matrix"], b["p1"]["agent_modality_list"])
            o2 = wrapper.forward_from_protocol(b["p2"]["heter_feature_2d"], b["p2"]["record_len"], b["p2"]["affine_matrix"], b["p2"]["agent_modality_list"])
            pf1 = torch.sigmoid(o1["cls_preds"]).max(dim=1, keepdim=True)[0]
            pf2 = torch.sigmoid(o2["cls_preds"]).max(dim=1, keepdim=True)[0]
            fg1 = build_fg_mask(b["p_ego_cls"], pf1, b["v_idx"], fg_prob_tau=args.fg_prob_tau, mode=args.fg_mode).detach().cpu()
            fg2 = build_fg_mask(b["p_ego_cls"], pf2, b["v_idx"], fg_prob_tau=args.fg_prob_tau, mode=args.fg_mode).detach().cpu()
        calib_records.append({
            "sample_idx": int(idx),
            "seq_name": seq_name,
            "cov_by_mod": {"m1": b["cov_ego_m1_full"][0, 0].detach().cpu(), "m2": b["cov_ego"][0, 0].detach().cpu()},
            "d_idx": b["d_idx"].detach().cpu(),
            "v_idx": b["v_idx"].detach().cpu(),
            "m1": b["dev_clean"]["m1"].detach().cpu(),
            "m2": b["dev_clean"]["m2"].detach().cpu(),
            "fg_m1": fg1,
            "fg_m2": fg2,
        })

    ch_stats, merge_diag = build_hierarchical_channel_stats_with_merging(
        calib_records, FIT_CALIB_SEQ_NAMES_7, min_frames=int(args.min_bin_frames), n_boot=2000, seed=20261120
    )
    out_base = os.path.join(args.out_dir, f"bud_calib_{args.fg_mode}.json")
    tmp_base_obj = OrderedDict({
        "channel_tier_stats": ch_stats,
        "frozen_methods": ref_frozen_methods,
    })
    with open(out_base, "w", encoding="utf-8") as f:
        json.dump(tmp_base_obj, f, indent=2, ensure_ascii=False)

    auditor = BUDFusionHEALAuditor(out_base)
    raw_scores = {"t2": {"m1": [], "m2": []}, "t4": {"m1": [], "m2": []}}
    per_seq_bg_fg = {role: {m: {vn: OrderedDict() for vn in VIS_LABELS_3} for m in ("m1", "m2")} for role in ("bg", "fg")}
    fg_bg_frame_cnts = {role: {m: {vn: 0 for vn in VIS_LABELS_3} for m in ("m1", "m2")} for role in ("bg", "fg")}

    for rec in calib_records:
        s_name = rec["seq_name"]
        d_g, v_g = rec["d_idx"].cuda(), rec["v_idx"].cuda()
        for mod, fg_k in (("m1", "fg_m1"), ("m2", "fg_m2")):
            cov_g, fg_g, dev_g = rec["cov_by_mod"][mod].cuda(), rec[fg_k].cuda(), rec[mod].cuda().float()
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
        per_seq_bg_fg, fg_bg_frame_cnts, FIT_CALIB_SEQ_NAMES_7, min_frames=int(args.min_bin_frames), n_boot=2000, seed=20261120
    )
    frame_high_tau, frame_high_tau_rule = compute_frame_high_tau_with_rule(
        raw_scores["t4"], frame_div_p95["tier4_sensor_dist_vis"], safety_mult=1.15, lower_bound=1.45
    )

    init_ct_bg = {m: [cos_q_bg[m][vn]["p99.8"] for vn in VIS_LABELS_3] for m in ("m1", "m2")}
    init_ct_fg = {m: [max(cos_q_bg[m][vn]["p99.8"], cos_q_fg[m][vn]["p99.0"]) for vn in VIS_LABELS_3] for m in ("m1", "m2")}
    base_obj = OrderedDict({
        "meta": {
            "claim": "[临时结论 · 待人工代码复核]",
            "calib_sequences": FIT_CALIB_SEQ_NAMES_7,
            "calib_frames": len(calib_records),
            "fg_mode": args.fg_mode,
            "visibility_fn": "assign_dist5_vis3_indices_v2",
            "bin_merging_summary": merge_diag,
            "fg_bg_bin_merging_log": merged_q_log,
        },
        "visibility_v2": vis_v2_cfg,
        "channel_tier_stats": ch_stats,
        "frozen_methods": ref_frozen_methods,
        "frame_div_p95": frame_div_p95,
        "frame_high_tau": frame_high_tau,
        "frame_high_tau_rule": frame_high_tau_rule,
        "pixel_cos_by_vis_bg_quantiles": cos_q_bg,
        "pixel_cos_by_vis_fg_quantiles": cos_q_fg,
        "gating_params": {
            "mode": "directional_vis_fg_bg",
            "fg_mode": args.fg_mode,
            "fg_rule": args.fg_mode,
            "k_tol": 5,
            "fg_prob_tau": float(args.fg_prob_tau),
            "bg_q": "p99.8",
            "fg_q": "p99.0",
            "cos_tau_by_vis_bg": init_ct_bg,
            "cos_tau_by_vis_fg": init_ct_fg,
            "gamma_cos_gate": [1.0, 1.0, 1.0],
            "cos_co_ratio": 0.82,
            "drop_tau": 0.40,
            "add_tau": 0.30,
            "w_drop": 7.0,
            "w_add": 6.0,
            "gamma_add": [1.0, 0.3, 0.0],
            "r_dil": 3,
            "beta": 3.5,
        },
    })
    with open(out_base, "w", encoding="utf-8") as f:
        json.dump(base_obj, f, indent=2, ensure_ascii=False)

    # 第三步：流式逐帧评测全部候选超参（绝不在内存中堆积多帧攻击特征！）
    search_space = OrderedDict({
        "bg_q": ["p99.8", "p99.9"],
        "fg_q": ["p98.0", "p99.0", "p99.5"],
        "gamma_cos_gate": [[1.0, 1.0, 1.0], [1.0, 0.85, 0.70]],
        "r_dil_and_beta": [[3, 3.5], [4, 4.0]],
    })
    candidates = []
    for bq in search_space["bg_q"]:
        for fq in search_space["fg_q"]:
            ct_bg = {m: [cos_q_bg[m][vn][bq] for vn in VIS_LABELS_3] for m in ("m1", "m2")}
            ct_fg = {m: [max(cos_q_bg[m][vn][bq], cos_q_fg[m][vn][fq]) for vn in VIS_LABELS_3] for m in ("m1", "m2")}
            for gc in search_space["gamma_cos_gate"]:
                for rd, bt in search_space["r_dil_and_beta"]:
                    candidates.append({
                        "mode": "directional_vis_fg_bg",
                        "fg_mode": args.fg_mode,
                        "fg_rule": args.fg_mode,
                        "k_tol": 5,
                        "fg_prob_tau": float(args.fg_prob_tau),
                        "bg_q": bq,
                        "fg_q": fq,
                        "cos_tau_by_vis_bg": ct_bg,
                        "cos_tau_by_vis_fg": ct_fg,
                        "gamma_cos_gate": gc,
                        "cos_co_ratio": 0.82,
                        "drop_tau": 0.40,
                        "add_tau": 0.30,
                        "w_drop": 7.0,
                        "w_add": 6.0,
                        "gamma_add": [1.0, 0.3, 0.0],
                        "r_dil": int(rd),
                        "beta": float(bt),
                    })

    def _new_ap():
        return {0.5: {"fp": [], "tp": [], "score": [], "gt": 0}}

    nd_stats = {m: {st: _new_ap() for st in FIT_DEV_STATES} for m in ("m1", "m2")}
    cand_stats = [{m: {st: _new_ap() for st in FIT_DEV_STATES} for m in ("m1", "m2")} for _ in candidates]

    for idx, _, _ in tune_items:
        b = extract_frame_context_and_clean(wrapper, ds_m1, ds_m2, idx, gx, gy, clear_pts_tau=c_tau, los_occ_tau=DEFAULT_LOS_OCC_TAU)
        raw_b1 = train_utils.to_device(ds_m1.collate_batch_test([ds_m1[idx]]), "cpu")
        b_post = train_utils.to_device(_slim_batch_for_postprocess(raw_b1), wrapper.device_obj)
        occ_mask_4d = b["occ_mask_ego"].float()

        for mod, p_dict, ds_ref in (("m1", b["p1"], ds_m1), ("m2", b["p2"], ds_m2)):
            feat, rlen, amat = p_dict["heter_feature_2d"], p_dict["record_len"], p_dict["affine_matrix"]
            mods_l, vmask = list(p_dict["agent_modality_list"]), p_dict["cam_valid_mask"]
            cov_4d = b["cov_ego_m1_full"] if mod == "m1" else b["cov_ego"]
            cov_2d = cov_4d[0, 0]

            f_lr, _ = generate_local_removal_heal(wrapper, feat, rlen, amat, mods_l, vmask, 1, 2, 5, 1.0, 10)
            f_lg, _ = generate_local_ghost_spoofing_heal(wrapper, feat, rlen, amat, mods_l, vmask, 1, 2, 5, 2.0, 10, idx)
            f_lo, _ = generate_local_occluded_ghost_heal(wrapper, feat, rlen, amat, mods_l, vmask, occ_mask_4d, 1, 2, 5, 2.0, 12, idx)
            f_ln, _ = generate_local_near0_6m_attack_heal(wrapper, feat, rlen, amat, mods_l, vmask, 1, 2.0, 12, idx, None, False)
            f_p1, _ = generate_scaled_pgd_removal_heal(wrapper, feat, rlen, amat, mods_l, vmask, 1, 1.0, 8)

            st_feats = [
                ("clean", feat),
                ("local_removal_1.0sigma", f_lr),
                ("local_ghost_2.0sigma", f_lg),
                ("local_occluded_ghost_2.0sigma", f_lo),
                ("local_near0_6m_2.0sigma", f_ln),
                ("scaled_pgd_1.0sigma", f_p1),
            ]
            for st_name, f_cur in st_feats:
                with torch.no_grad():
                    out_nd = wrapper.forward_from_protocol(f_cur, rlen, amat, mods_l)
                    pb_nd, ps_nd, gb_nd = wrapper.decode_boxes(b_post, out_nd, ds_ref)
                eval_utils.caluclate_tp_fp(pb_nd, ps_nd, gb_nd, nd_stats[mod][st_name], 0.5)

                dev_a, dd_a, da_a, pf_a = _compute_dev_signed_and_pfused(wrapper, p_dict, f_cur, b["p_ego_cls"], cov_4d, k_tol=5)
                fg_m = build_fg_mask(b["p_ego_cls"], pf_a, b["v_idx"], fg_prob_tau=args.fg_prob_tau, mode=args.fg_mode)
                ar_a = auditor.audit_dev_maps(dev_a, cov_2d, b["d_idx"], b["v_idx"], mod, build_gate=False)
                sc_t4 = float(ar_a["raw_t4"]) / frame_div_p95["tier4_sensor_dist_vis"][mod]

                for ci, gp in enumerate(candidates):
                    if sc_t4 > frame_high_tau[mod]:
                        gate = torch.zeros((1, 1, PROTO_H, PROTO_W), device=f_cur.device, dtype=f_cur.dtype)
                    else:
                        gate = build_directional_gate_map(
                            cos_comb=ar_a["cos_comb_t4"], d_drop_tol=dd_a, d_add_tol=da_a, cov_2d=cov_2d, v_idx=b["v_idx"],
                            cos_tau=gp["cos_tau_by_vis_bg"][mod], drop_tau=gp["drop_tau"], add_tau=gp["add_tau"],
                            w_drop=gp["w_drop"], w_add=gp["w_add"], cos_co_ratio=gp["cos_co_ratio"],
                            gamma_add=gp["gamma_add"], gamma_cos_gate=gp["gamma_cos_gate"], r_dil=gp["r_dil"], beta=gp["beta"],
                            collab_fg_mask=fg_m, cos_tau_fg=gp["cos_tau_by_vis_fg"][mod],
                        )
                    with torch.no_grad():
                        out_c = wrapper.forward_from_protocol(f_cur, rlen, amat, mods_l, custom_fuse_fn=build_gated_fuse_fn(gate))
                        pb_c, ps_c, gb_c = wrapper.decode_boxes(b_post, out_c, ds_ref)
                    eval_utils.caluclate_tp_fp(pb_c, ps_c, gb_c, cand_stats[ci][mod][st_name], 0.5)

    atk_keys = [k for k in FIT_DEV_STATES if k != "clean"]
    base_ap = {m: {st: float(eval_utils.calculate_ap(nd_stats[m][st], 0.5)[0]) for st in FIT_DEV_STATES} for m in ("m1", "m2")}
    all_records = []
    best_rec, best_obj = None, -1e9
    for ci, gp in enumerate(candidates):
        ap50 = {m: {st: float(eval_utils.calculate_ap(cand_stats[ci][m][st], 0.5)[0]) for st in FIT_DEV_STATES} for m in ("m1", "m2")}
        max_ap_loss = max(base_ap["m1"]["clean"] - ap50["m1"]["clean"], base_ap["m2"]["clean"] - ap50["m2"]["clean"])
        mean_atk = float(np.mean([ap50[m][a] for m in ("m1", "m2") for a in atk_keys]))
        obj = mean_atk - 1.50 * max(0.0, max_ap_loss - 0.01)
        rec = {"gating_params": gp, "ap50": ap50, "mean_atk_ap50": round(mean_atk, 5), "obj": round(obj, 5)}
        all_records.append(rec)
        if obj > best_obj:
            best_obj = obj
            best_rec = rec

    out_bud = os.path.join(args.out_dir, "bud_tuned_calibrator.json")
    bud_obj = OrderedDict({
        "meta": {
            "claim": "[临时结论 · 待人工代码复核]",
            "fg_rule_used_in_calibration": args.fg_mode,
            "fg_bg_merged_bins_log": merged_q_log,
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "visibility_v2": vis_v2_cfg,
        "channel_tier_stats": ch_stats,
        "frozen_methods": ref_frozen_methods,
        "frame_div_p95": frame_div_p95,
        "frame_high_tau": frame_high_tau,
        "frame_high_tau_rule": frame_high_tau_rule,
        "pixel_cos_by_vis_bg_quantiles": cos_q_bg,
        "pixel_cos_by_vis_fg_quantiles": cos_q_fg,
        "gating_params": best_rec["gating_params"],
        "chosen_eval": best_rec,
        "grid_records": all_records,
    })
    with open(out_bud, "w", encoding="utf-8") as f:
        json.dump(bud_obj, f, indent=2, ensure_ascii=False)
    print(f"[完成] 统一调参标定文件已保存至 {out_bud} (耗时={time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
