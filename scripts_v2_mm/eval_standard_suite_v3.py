# -*- coding: utf-8 -*-
"""
sprint-cs2027 统一标准指标评测与 B4 权衡扫描脚本（覆盖 A0.4、B1-B4 Dev 验证、F-P0.1 可见性掩码基线）。

严格实现 A0.4 规定的全部标准指标集（全部附 2,000 次按序列块 Bootstrap 95% CI，种子 20261120）：
1. m1 / m2 的干净与受攻 AP@0.5、AP@0.7（主实验：m1 用完整激光雷达覆盖区且不限制攻击区，m2 用相机覆盖区）；
2. 盲区 AP@0.5（非盲区真值记为 Ignore 不计 FP，同步报告协作车自身可观测盲区子集）；
3. 诚实协作车被压制比例（G_i < 0.5 比例，按传感器 × Clear/Sparse/Occluded 分档，含权重削减占比）；
4. 5 种攻击的防刷 ASR（全局 PGD 1.0σ、局部移除 1.0σ、局部伪造 2.0σ、遮挡区伪造 2.0σ、近场 0~6m 移除 2.0σ；
   同帧各方法共用同一份攻击扰动；剔除 3m 内真值并单独报告近目标伪造）；
5. 多车场景（>=3 辆协作车，即总车数 >=4）干净 AP@0.5。

支持三大模式（--mode）：
- standard_eval     : 评测当前配置（支持切换 downsample_mode / fg_mode / enable_b3），可与上一版本 JSON 自动生成前后对比表；
- b4_gamma_scan     : B4 任务专用，在同帧共用扰动下扫描 γ(Occluded) ∈ {0.0, 0.1, 0.2, 0.3}，输出遮挡区伪造 ASR vs 干净盲区 AP 权衡曲线；
- vis_mask_baseline : F-P0.1 可见性掩码基线（Clear 格子只用本车 G=0，其余格子正常融合 G=1，不做其他检测）。
"""

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PROJ_ROOT = "/root/autodl-tmp/proj_dir2_hetero_trust"
if os.path.exists(PROJ_ROOT) and PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)
LOCAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if LOCAL_ROOT not in sys.path:
    sys.path.insert(0, LOCAL_ROOT)

from src_v2_mm.bud_fusion_heal import (
    BUDFusionHEALAuditor,
    build_fg_mask,
    build_gated_fuse_fn,
    compute_shift_tolerant_semantic_maps,
    extract_single_cav_dev_and_signed,
)
from src_v2_mm.eval_stats_utils import (
    build_seq_bootstrap_indices,
    fast_voc_ap,
    paired_block_bootstrap_ap_diff,
    paired_block_bootstrap_rate_diff,
)
from src_v2_mm.heal_attacks import (
    generate_local_ghost_spoofing_heal,
    generate_local_near0_6m_attack_heal,
    generate_local_occluded_ghost_heal,
    generate_local_removal_heal,
    generate_scaled_pgd_removal_heal,
)
from src_v2_mm.heal_base_wrapper import HEALBaseWrapper
from src_v2_mm.sprint_metrics_v3 import (
    VIS_NAMES_3,
    compute_honest_suppression_stats,
    evaluate_frame_asr_strict,
    extract_centers_xy_from_boxes_np,
    filter_boxes_by_score,
    has_center_within_radius,
    match_blind_spot_ap_ignore_nonblind,
    studentized_paired_block_bootstrap,
)
from src_v2_mm.step4_1_v2_core import (
    DEFAULT_LOS_OCC_TAU,
    GRID_RES,
    MIN_XY,
    OPV2V_TRAIN_DIR,
    PROTO_H,
    PROTO_W,
    V3_BUD_CALIB_V1_JSON,
    V3_BUD_TUNED_JSON,
    V3_SPLITS_JSON,
    _masked_avg_pool3x3,
    assign_dist5_vis3_indices_v2,
    build_bev_grid_xy_256,
    extract_frame_context_and_clean,
    load_ego_lidar_np,
    require_visibility_v2_taus,
)

RES_V3_DIR = os.path.join(PROJ_ROOT if os.path.exists(PROJ_ROOT) else LOCAL_ROOT, "results_v3")
DEFAULT_EVAL_SPLIT = V3_SPLITS_JSON
DEFAULT_BASE_CALIB = V3_BUD_CALIB_V1_JSON
DEFAULT_FROZEN_GATING = None  # 默认使用 --base_calib_json 自包含标定表，仅当显式传入时覆盖

ATTACKS_5 = (
    "pgd_1.0sigma",
    "local_removal_1.0sigma",
    "local_ghost_2.0sigma",
    "local_occluded_ghost_2.0sigma",
    "local_near0_6m_2.0sigma",
)
B4_GAMMA_CANDIDATES = (0.0, 0.1, 0.2, 0.3)
BOOT_N = 2000
BOOT_SEED = 20261120


def build_visibility_mask_gate(v_idx_2d: torch.Tensor) -> torch.Tensor:
    """
    F-P0.1 决定性对照基线（Visibility-Mask Baseline）：
    - Clear (v_idx == 0) 格子只用本车特征（协作车门控 G = 0.0）；
    - Sparse / Occluded (v_idx != 0) 格子完全信任并正常融合协作车（门控 G = 1.0），不做任何异常检测。
    返回形状 (1, 1, H, W) 的 float32 门控张量。
    """
    v2d = v_idx_2d.view(PROTO_H, PROTO_W) if v_idx_2d.dim() != 2 else v_idx_2d
    g2d = (v2d != 0).to(dtype=torch.float32)
    return g2d.view(1, 1, PROTO_H, PROTO_W)


def _atomic_write_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _slim_batch_for_postprocess(batch_dict: dict) -> dict:
    keep_keys = [
        "object_bbx_center",
        "object_bbx_mask",
        "object_ids",
        "anchor_box",
        "transformation_matrix",
        "transformation_matrix_clean",
        "record_len",
    ]
    slim_ego = {}
    for k in keep_keys:
        if k in batch_dict["ego"]:
            v = batch_dict["ego"][k]
            slim_ego[k] = v.detach().cpu() if torch.is_tensor(v) else v
    return {"ego": slim_ego}


def _grid_hw_to_xy(hc: int, wc: int) -> Tuple[float, float]:
    return (float(wc) + 0.5) * GRID_RES + MIN_XY, (float(hc) + 0.5) * GRID_RES + MIN_XY


def _match_standard_ap(pred_boxes_np, pred_scores_np, gt_polys: list, iou_thresh: float) -> dict:
    from opencood.utils import common_utils

    gt_cnt = len(gt_polys)
    matched = [False] * gt_cnt
    if pred_boxes_np is None or pred_scores_np is None or len(pred_boxes_np) == 0:
        return {"tp": [], "fp": [], "score": [], "gt": gt_cnt, "matched": matched}
    sc_np = np.asarray(pred_scores_np, dtype=np.float64).reshape(-1)
    if sc_np.size == 0:
        return {"tp": [], "fp": [], "score": [], "gt": gt_cnt, "matched": matched}
    order = np.argsort(-sc_np)
    sc_sorted = sc_np[order].tolist()
    det_polys = list(common_utils.convert_format(pred_boxes_np))
    rem_polys = list(gt_polys)
    rem_ids = list(range(gt_cnt))
    tp_l, fp_l = [], []
    for k in range(order.shape[0]):
        dp = det_polys[int(order[k])]
        if not rem_polys:
            tp_l.append(0)
            fp_l.append(1)
            continue
        ious = common_utils.compute_iou(dp, rem_polys)
        bj = int(np.argmax(ious))
        if float(ious[bj]) >= float(iou_thresh):
            tp_l.append(1)
            fp_l.append(0)
            rem_polys.pop(bj)
            matched[rem_ids.pop(bj)] = True
        else:
            tp_l.append(0)
            fp_l.append(1)
    return {"tp": tp_l, "fp": fp_l, "score": sc_sorted, "gt": gt_cnt, "matched": matched}


def _find_blind_gts(ds_m1, idx: int, ego_pb_np, ego_ps_np, gt_np: np.ndarray, gt_polys: list, cov_2d: torch.Tensor) -> List[int]:
    """按客观物理盲区标准筛选：落入协作者覆盖区 ∧ 非本车自身(r>=2.5m) ∧ 本车点云<=5 ∧ Ego漏检。"""
    from matplotlib.path import Path as MplPath

    ego_m = _match_standard_ap(ego_pb_np, ego_ps_np, gt_polys, 0.5)["matched"]
    ego_pts = load_ego_lidar_np(ds_m1, idx)
    cov_np = cov_2d.detach().cpu().numpy()
    blind_ids = []
    for i in range(gt_np.shape[0]):
        cx = float(gt_np[i, :4, 0].mean())
        cy = float(gt_np[i, :4, 1].mean())
        if math.hypot(cx, cy) < 2.5 or ego_m[i]:
            continue
        ix = int(math.floor((cx - MIN_XY) / GRID_RES))
        iy = int(math.floor((cy - MIN_XY) / GRID_RES))
        if not (0 <= ix < PROTO_W and 0 <= iy < PROTO_H and bool(cov_np[iy, ix])):
            continue
        corners = gt_np[i, :4, :2]
        c_pts = ego_pts[
            (ego_pts[:, 0] >= float(corners[:, 0].min()))
            & (ego_pts[:, 0] <= float(corners[:, 0].max()))
            & (ego_pts[:, 1] >= float(corners[:, 1].min()))
            & (ego_pts[:, 1] <= float(corners[:, 1].max()))
            & (ego_pts[:, 2] >= float(gt_np[i, :, 2].min()) - 0.3)
            & (ego_pts[:, 2] <= float(gt_np[i, :, 2].max()) + 0.3)
        ]
        n_in = int(MplPath(corners).contains_points(c_pts[:, :2]).sum()) if c_pts.shape[0] > 0 else 0
        if n_in <= 5:
            blind_ids.append(i)
    return blind_ids


@torch.no_grad()
def _extract_pair_dev(wrapper, p_dict: dict, feat_in: torch.Tensor, p_ego_cls: torch.Tensor, cov_ego: torch.Tensor, k_tol: int = 5):
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

    rlen, amat, mods = p_dict["record_len"], p_dict["affine_matrix"], p_dict["agent_modality_list"]
    tmat01 = amat[0, 0, 1:2]
    ac = wrapper.model.pyramid_backbone.align_corners
    out = wrapper.forward_from_protocol(feat_in, rlen, amat, mods)
    cov_f = cov_ego.float()
    cos_maps = []
    for lvl in range(3):
        f_lvl = out["pyramid_feats"][lvl]
        z_ego = f_lvl[0:1]
        hl, wl = z_ego.shape[-2:]
        z_cav = warp_affine_simple(f_lvl[1:2], tmat01, (hl, wl), align_corners=ac)
        c_raw = (1.0 - torch.nn.functional.cosine_similarity(z_ego, z_cav, dim=1, eps=1e-6)).unsqueeze(1)
        if lvl == 0:
            c_256 = _masked_avg_pool3x3(c_raw, cov_ego)
        else:
            c_256 = torch.nn.functional.interpolate(c_raw, size=(PROTO_H, PROTO_W), mode="bilinear", align_corners=False) * cov_f
        cos_maps.append(c_256)
    pf_cls = torch.sigmoid(out["cls_preds"]).max(dim=1, keepdim=True)[0]
    d_sem = torch.abs(p_ego_cls - pf_cls) * cov_f
    dev_4ch = torch.cat(cos_maps + [d_sem], dim=1)[0]
    d_drop, d_add = compute_shift_tolerant_semantic_maps(p_ego_cls, pf_cls, cov_ego, k_tol=k_tol)
    return out, dev_4ch, d_drop, d_add, pf_cls


def _decode_np(wrapper, b_post: dict, out_dict: dict, ds_ref):
    from opencood.utils import common_utils

    pb, ps, gb = wrapper.decode_boxes(b_post, out_dict, ds_ref)
    pb_np = common_utils.torch_tensor_to_numpy(pb) if (pb is not None and len(pb) > 0) else None
    ps_np = common_utils.torch_tensor_to_numpy(ps).reshape(-1) if (ps is not None and len(ps) > 0) else None
    gb_np = common_utils.torch_tensor_to_numpy(gb) if (gb is not None and len(gb) > 0) else np.zeros((0, 8, 3))
    return pb_np, ps_np, gb_np


def _run_b4_gamma_scan(args, wrapper, ds_m1, ds_m2, eval_split: dict, auditor: BUDFusionHEALAuditor, gx, gy):
    """B4 专用：扫描 γ(Occluded) ∈ {0.0, 0.1, 0.2, 0.3}，报告遮挡区伪造防刷 ASR 与干净盲区 AP@0.5 的权衡曲线。"""
    from opencood.tools import train_utils
    from opencood.utils import common_utils

    t0 = time.time()
    seq_order = list(eval_split["seq_order"])
    gcfg = (auditor.gating_calib or {}).get("gating_params", {})
    fg_prob_tau = float(gcfg.get("fg_prob_tau", 0.20))
    fg_mode = str(gcfg.get("fg_mode", gcfg.get("fg_rule", "v1")))
    c_tau, l_tau = require_visibility_v2_taus(auditor.gating_calib)

    gammas = [float(g) for g in B4_GAMMA_CANDIDATES]
    # 按序列累积盲区 AP 与遮挡伪造 ASR
    blind_seq = {g: {m: {s: {"tp": [], "fp": [], "score": [], "gt": 0} for s in seq_order} for m in ("m1", "m2")} for g in gammas}
    occ_asr_seq = {g: {m: {s: [0, 0] for s in seq_order} for m in ("m1", "m2")} for g in gammas}

    for seq_name in seq_order:
        idxs = eval_split["sequences"][seq_name]["indices"]
        for idx in idxs:
            b = extract_frame_context_and_clean(wrapper, ds_m1, ds_m2, idx, gx, gy, clear_pts_tau=c_tau, los_occ_tau=l_tau)
            raw_b1 = train_utils.to_device(ds_m1.collate_batch_test([ds_m1[idx]]), "cpu")
            b_post = train_utils.to_device(_slim_batch_for_postprocess(raw_b1), wrapper.device_obj)
            with torch.no_grad():
                ego_out = wrapper.forward_single_agent(b["p1"]["heter_feature_2d"][0:1])
            ego_pb_np, ego_ps_np, gt_np = _decode_np(wrapper, b_post, ego_out, ds_m1)
            gt_polys = list(common_utils.convert_format(gt_np))
            occ_mask_4d = b["occ_mask_ego"].float()

            for mod, p_dict, ds_ref in (("m1", b["p1"], ds_m1), ("m2", b["p2"], ds_m2)):
                cov_ego_mod = b["cov_ego_m1_full"] if mod == "m1" else b["cov_ego"]
                cov_2d = cov_ego_mod[0, 0]
                blind_ids = _find_blind_gts(ds_m1, idx, ego_pb_np, ego_ps_np, gt_np, gt_polys, cov_2d)

                feat, rlen, amat = p_dict["heter_feature_2d"], p_dict["record_len"], p_dict["affine_matrix"]
                mods_l, vmask = list(p_dict["agent_modality_list"]), p_dict["cam_valid_mask"]

                # 1) Clean 偏差提取与干净预测
                out_cln, dev_cln, dd_cln, da_cln, pf_cln = _extract_pair_dev(wrapper, p_dict, feat, b["p_ego_cls"], cov_ego_mod)
                cln_pb_np, cln_ps_np, _ = _decode_np(wrapper, b_post, out_cln, ds_ref)
                fg_cln = build_fg_mask(b["p_ego_cls"], pf_cln, b["v_idx"], fg_prob_tau=fg_prob_tau, mode=fg_mode)

                # 2) 同帧固定生成一次遮挡区伪造攻击，四个 γ 候选共用完全相同的扰动！
                f_occ, st_occ = generate_local_occluded_ghost_heal(
                    wrapper, feat, rlen, amat, mods_l, vmask, occluded_mask_ego=occ_mask_4d,
                    attacker_idx=1, num_spoofs=2, patch_radius=5, sigma_mult=2.0, steps=12, frame_seed=idx, perturb_region_mask=None,
                )
                _, dev_occ, dd_occ, da_occ, pf_occ = _extract_pair_dev(wrapper, p_dict, f_occ, b["p_ego_cls"], cov_ego_mod)
                fg_occ = build_fg_mask(b["p_ego_cls"], pf_occ, b["v_idx"], fg_prob_tau=fg_prob_tau, mode=fg_mode)
                occ_cands = [_grid_hw_to_xy(anc[0], anc[1]) for anc in st_occ.get("anchors_hwr", [])]

                for g_occ in gammas:
                    ga_vec = [float(gcfg["gamma_add"][0]), float(gcfg["gamma_add"][1]), float(g_occ)]
                    # Clean 盲区 AP
                    ar_c = auditor.audit_dev_maps(
                        dev_cln, cov_2d, b["d_idx"], b["v_idx"], mod, build_gate=True,
                        d_drop_tol=dd_cln, d_add_tol=da_cln, enable_whole_fallback=True,
                        collab_fg_mask=fg_cln, gamma_add_override=ga_vec,
                    )
                    with torch.no_grad():
                        o_c = wrapper.forward_from_protocol(
                            feat, rlen, amat, mods_l,
                            custom_fuse_fn=build_gated_fuse_fn(ar_c["gate_L0"], downsample_mode=args.downsample_mode, enable_b3=not args.disable_b3),
                        )
                    pb_c, ps_c, _ = _decode_np(wrapper, b_post, o_c, ds_ref)
                    bm = match_blind_spot_ap_ignore_nonblind(pb_c, ps_c, gt_polys, blind_ids, iou_thresh=0.5)
                    bs = blind_seq[g_occ][mod][seq_name]
                    bs["tp"].extend(bm["tp"])
                    bs["fp"].extend(bm["fp"])
                    bs["score"].extend(bm["score"])
                    bs["gt"] += int(bm["gt"])

                    # 遮挡区伪造防刷 ASR
                    if occ_cands:
                        ar_o = auditor.audit_dev_maps(
                            dev_occ, cov_2d, b["d_idx"], b["v_idx"], mod, build_gate=True,
                            d_drop_tol=dd_occ, d_add_tol=da_occ, enable_whole_fallback=True,
                            collab_fg_mask=fg_occ, gamma_add_override=ga_vec,
                        )
                        with torch.no_grad():
                            o_o = wrapper.forward_from_protocol(
                                f_occ, rlen, amat, mods_l,
                                custom_fuse_fn=build_gated_fuse_fn(ar_o["gate_L0"], downsample_mode=args.downsample_mode, enable_b3=not args.disable_b3),
                            )
                        pb_o, ps_o, _ = _decode_np(wrapper, b_post, o_o, ds_ref)
                        ast = evaluate_frame_asr_strict(
                            "fabrication", occ_cands, cln_pb_np, cln_ps_np, pb_o, ps_o, gt_np, s_min=args.s_min
                        )
                        occ_asr_seq[g_occ][mod][seq_name][0] += int(ast["succ"])
                        occ_asr_seq[g_occ][mod][seq_name][1] += int(ast["total"])

        print(f"  [B4 Scan] 完成序列 {seq_name} ({time.time() - t0:.1f}s)", flush=True)

    boot_idx = build_seq_bootstrap_indices(seq_order, n_boot=BOOT_N, seed=BOOT_SEED)
    curve = OrderedDict()
    for g_occ in gammas:
        key = f"gamma_occ_{g_occ:.1f}"
        curve[key] = OrderedDict()
        for mod in ("m1", "m2"):
            ap_d = paired_block_bootstrap_ap_diff(
                seq_order, blind_seq[g_occ][mod], blind_seq[0.0][mod], boot_indices=boot_idx
            )
            h_g = {s: occ_asr_seq[g_occ][mod][s][0] for s in seq_order}
            h_0 = {s: occ_asr_seq[0.0][mod][s][0] for s in seq_order}
            tot = {s: occ_asr_seq[0.0][mod][s][1] for s in seq_order}
            asr_d = paired_block_bootstrap_rate_diff(seq_order, h_g, h_0, tot, boot_indices=boot_idx)
            curve[key][mod] = {
                "clean_blind_ap50": ap_d["val_a"],
                "clean_blind_ap50_ci95": ap_d["a_ci95"],
                "blind_ap50_diff_vs_0": ap_d["diff"],
                "occluded_fab_asr": asr_d["val_a"],
                "occluded_fab_asr_ci95": asr_d["a_ci95"],
                "occluded_fab_asr_diff_vs_0": asr_d["diff"],
            }

    # 按 B4 规则客观判定：默认保持 0.0；仅当 0.1 使平均遮挡伪造 ASR 明显下降(>=2个百分点)且双模态盲区 AP 降幅均 <= 0.005 时推荐 0.1
    c01 = curve["gamma_occ_0.1"]
    max_blind_drop_01 = max(-c01["m1"]["blind_ap50_diff_vs_0"], -c01["m2"]["blind_ap50_diff_vs_0"])
    mean_asr_drop_01 = -0.5 * (c01["m1"]["occluded_fab_asr_diff_vs_0"] + c01["m2"]["occluded_fab_asr_diff_vs_0"])
    recommend_01 = bool(mean_asr_drop_01 >= 0.02 and max_blind_drop_01 <= 0.005)

    out_report = OrderedDict({
        "meta": {
            "claim": "[临时结论 · 待人工代码复核]",
            "task": "B4_gamma_occluded_tradeoff_scan",
            "candidates": gammas,
            "s_min": float(args.s_min),
            "num_sequences": len(seq_order),
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "tradeoff_curve": curve,
        "decision": {
            "chosen_gamma_occluded": 0.1 if recommend_01 else 0.0,
            "mean_occluded_asr_reduction_at_0.1": round(float(mean_asr_drop_01), 5),
            "max_clean_blind_ap50_drop_at_0.1": round(float(max_blind_drop_01), 5),
            "rationale": (
                "γ(Occluded)=0.1 使遮挡区伪造 ASR 下降 >=2% 且盲区 AP 降幅 <=0.5 个点，按 B4 准则采纳 0.1。"
                if recommend_01
                else "保持默认 γ(Occluded)=0.0（0.1 未同时满足 ASR 显著下降与盲区 AP 降幅 <=0.5 个点）。"
            ),
        },
    })
    _atomic_write_json(args.out_json, out_report)
    print(f"\n[B4 完成] 权衡曲线已写入 {args.out_json} -> 推荐 γ(Occluded)={out_report['decision']['chosen_gamma_occluded']}")


def _run_standard_or_vismask_eval(args, wrapper, ds_m1, ds_m2, ds_mix, eval_split: dict, auditor: BUDFusionHEALAuditor, gx, gy):
    """运行 A0.4 全套标准指标评测（standard_eval 或 vis_mask_baseline），并可选与上一版本 JSON 输出 Markdown 前后对比表。"""
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
    from opencood.tools import train_utils
    from opencood.utils import common_utils

    t0 = time.time()
    is_vismask = args.mode == "vis_mask_baseline"
    meth_tag = "vis_mask_baseline" if is_vismask else "bud_current"
    methods = ("no_defense", meth_tag)

    seq_order = list(eval_split["seq_order"])
    gcfg = (auditor.gating_calib or {}).get("gating_params", {})
    fg_prob_tau = float(gcfg.get("fg_prob_tau", 0.20))
    fg_mode = str(args.fg_mode if args.fg_mode else gcfg.get("fg_mode", gcfg.get("fg_rule", "v1")))
    c_tau, l_tau = require_visibility_v2_taus(auditor.gating_calib)

    ap50_seq = {mt: {m: {s: {"tp": [], "fp": [], "score": [], "gt": 0} for s in seq_order} for m in ("m1", "m2")} for mt in methods}
    ap70_seq = {mt: {m: {s: {"tp": [], "fp": [], "score": [], "gt": 0} for s in seq_order} for m in ("m1", "m2")} for mt in methods}
    blind_seq = {mt: {m: {s: {"tp": [], "fp": [], "score": [], "gt": 0} for s in seq_order} for m in ("m1", "m2")} for mt in methods}
    multicav_ge3_seq = {mt: {s: {"tp": [], "fp": [], "score": [], "gt": 0} for s in seq_order} for mt in methods}

    # 5 种攻击 ASR：pgd_1.0sigma 用攻击后 AP 降幅补充报告 + 移除率（干净检出真车被抹除率）统一量化
    asr_seq = {
        mt: {atk: {m: {s: [0, 0] for s in seq_order} for m in ("m1", "m2")} for atk in ATTACKS_5}
        for mt in methods
    }
    honest_sup_acc = {
        m: {
            "sup_grids": 0, "tot_grids": 0, "w_num": 0.0, "w_den": 0.0,
            "by_vis": {vn: [0, 0] for vn in VIS_NAMES_3},
        }
        for m in ("m1", "m2")
    }

    for seq_name in seq_order:
        idxs = eval_split["sequences"][seq_name]["indices"]
        for idx in idxs:
            b = extract_frame_context_and_clean(wrapper, ds_m1, ds_m2, idx, gx, gy, clear_pts_tau=c_tau, los_occ_tau=l_tau)
            raw_b1 = train_utils.to_device(ds_m1.collate_batch_test([ds_m1[idx]]), "cpu")
            b_post = train_utils.to_device(_slim_batch_for_postprocess(raw_b1), wrapper.device_obj)
            with torch.no_grad():
                ego_out = wrapper.forward_single_agent(b["p1"]["heter_feature_2d"][0:1])
            ego_pb_np, ego_ps_np, gt_np = _decode_np(wrapper, b_post, ego_out, ds_m1)
            gt_polys = list(common_utils.convert_format(gt_np))
            occ_mask_4d = b["occ_mask_ego"].float()

            # --- 1) 单协作车 m1 / m2 标准指标（Clean + 5 攻击） ---
            for mod, p_dict, ds_ref in (("m1", b["p1"], ds_m1), ("m2", b["p2"], ds_m2)):
                cov_ego_mod = b["cov_ego_m1_full"] if mod == "m1" else b["cov_ego"]
                cov_2d = cov_ego_mod[0, 0]
                blind_ids = _find_blind_gts(ds_m1, idx, ego_pb_np, ego_ps_np, gt_np, gt_polys, cov_2d)
                feat, rlen, amat = p_dict["heter_feature_2d"], p_dict["record_len"], p_dict["affine_matrix"]
                mods_l, vmask = list(p_dict["agent_modality_list"]), p_dict["cam_valid_mask"]

                out_cln, dev_cln, dd_cln, da_cln, pf_cln = _extract_pair_dev(wrapper, p_dict, feat, b["p_ego_cls"], cov_ego_mod)
                cln_pb_np, cln_ps_np, _ = _decode_np(wrapper, b_post, out_cln, ds_ref)

                if is_vismask:
                    gate_cln = build_visibility_mask_gate(b["v_idx"]).to(device=feat.device)
                    wf_cln = False
                else:
                    fg_cln = build_fg_mask(b["p_ego_cls"], pf_cln, b["v_idx"], fg_prob_tau=fg_prob_tau, mode=fg_mode)
                    ar_cln = auditor.audit_dev_maps(
                        dev_cln, cov_2d, b["d_idx"], b["v_idx"], mod, build_gate=True,
                        d_drop_tol=dd_cln, d_add_tol=da_cln, enable_whole_fallback=True, collab_fg_mask=fg_cln,
                    )
                    gate_cln = ar_cln["gate_L0"]
                    wf_cln = bool(ar_cln["whole_fallback"])

                with torch.no_grad():
                    out_def_c = wrapper.forward_from_protocol(
                        feat, rlen, amat, mods_l,
                        custom_fuse_fn=build_gated_fuse_fn(gate_cln, downsample_mode=args.downsample_mode, enable_b3=not args.disable_b3),
                    )
                def_pb_np, def_ps_np, _ = _decode_np(wrapper, b_post, out_def_c, ds_ref)

                # 统计诚实协作车被压制比例与权重削减比
                hst = compute_honest_suppression_stats(gate_cln, cov_2d, b["v_idx"], whole_fallback=wf_cln)
                honest_sup_acc[mod]["sup_grids"] += hst["suppressed_grids"]
                honest_sup_acc[mod]["tot_grids"] += hst["total_grids"]
                honest_sup_acc[mod]["w_num"] += hst["reclaimed_weight_num"]
                honest_sup_acc[mod]["w_den"] += hst["reclaimed_weight_den"]
                for vn in VIS_NAMES_3:
                    honest_sup_acc[mod]["by_vis"][vn][0] += hst["by_visibility"][vn]["suppressed_grids"]
                    honest_sup_acc[mod]["by_vis"][vn][1] += hst["by_visibility"][vn]["total_grids"]

                for mt, (pb_cur, ps_cur) in (("no_defense", (cln_pb_np, cln_ps_np)), (meth_tag, (def_pb_np, def_ps_np))):
                    r5 = _match_standard_ap(pb_cur, ps_cur, gt_polys, 0.5)
                    r7 = _match_standard_ap(pb_cur, ps_cur, gt_polys, 0.7)
                    rb = match_blind_spot_ap_ignore_nonblind(pb_cur, ps_cur, gt_polys, blind_ids, iou_thresh=0.5)
                    for bucket, rm in ((ap50_seq[mt][mod][seq_name], r5), (ap70_seq[mt][mod][seq_name], r7), (blind_seq[mt][mod][seq_name], rb)):
                        bucket["tp"].extend(rm["tp"])
                        bucket["fp"].extend(rm["fp"])
                        bucket["score"].extend(rm["score"])
                        bucket["gt"] += int(rm["gt"])

                # 5 种攻击生成一次并由各方法共享同一份扰动
                f_pgd, _ = generate_scaled_pgd_removal_heal(wrapper, feat, rlen, amat, mods_l, vmask, attacker_idx=1, sigma_mult=1.0, steps=8)
                f_rem, st_rem = generate_local_removal_heal(wrapper, feat, rlen, amat, mods_l, vmask, attacker_idx=1, num_targets=2, patch_radius=5, sigma_mult=1.0, steps=10)
                f_fab, st_fab = generate_local_ghost_spoofing_heal(wrapper, feat, rlen, amat, mods_l, vmask, attacker_idx=1, num_spoofs=2, patch_radius=5, sigma_mult=2.0, steps=10, frame_seed=idx)
                f_occ, st_occ = generate_local_occluded_ghost_heal(wrapper, feat, rlen, amat, mods_l, vmask, occluded_mask_ego=occ_mask_4d, attacker_idx=1, num_spoofs=2, patch_radius=5, sigma_mult=2.0, steps=12, frame_seed=idx)
                f_near, st_near = generate_local_near0_6m_attack_heal(wrapper, feat, rlen, amat, mods_l, vmask, attacker_idx=1, sigma_mult=2.0, steps=12, frame_seed=idx, enable_ghost_fallback=False)

                # 全局 PGD 1.0σ 的移除目标：本帧所有落入协作者覆盖区且非自车的真实车辆中心
                gt_centers_cov = []
                cov_np_m = cov_2d.detach().cpu().numpy()
                for gi in range(gt_np.shape[0]):
                    gcx, gcy = float(gt_np[gi, :4, 0].mean()), float(gt_np[gi, :4, 1].mean())
                    if math.hypot(gcx, gcy) < 2.5:
                        continue
                    gix, giy = int(math.floor((gcx - MIN_XY) / GRID_RES)), int(math.floor((gcy - MIN_XY) / GRID_RES))
                    if 0 <= gix < PROTO_W and 0 <= giy < PROTO_H and bool(cov_np_m[giy, gix]):
                        gt_centers_cov.append((gcx, gcy))

                atk_specs = [
                    ("pgd_1.0sigma", f_pgd, gt_centers_cov, "removal"),
                    ("local_removal_1.0sigma", f_rem, [_grid_hw_to_xy(h, w) for h, w in st_rem.get("target_centers_hw", [])], "removal"),
                    ("local_ghost_2.0sigma", f_fab, [_grid_hw_to_xy(a[0], a[1]) for a in st_fab.get("anchors_hwr", [])], "fabrication"),
                    ("local_occluded_ghost_2.0sigma", f_occ, [_grid_hw_to_xy(a[0], a[1]) for a in st_occ.get("anchors_hwr", [])], "fabrication"),
                    ("local_near0_6m_2.0sigma", f_near, [_grid_hw_to_xy(h, w) for h, w in st_near.get("target_centers_hw", [])], "removal"),
                ]

                for atk_name, f_adv, cands_xy, mode_s in atk_specs:
                    if not cands_xy:
                        continue
                    out_nd_a, dev_a, dd_a, da_a, pf_a = _extract_pair_dev(wrapper, p_dict, f_adv, b["p_ego_cls"], cov_ego_mod)
                    nd_pb_a, nd_ps_a, _ = _decode_np(wrapper, b_post, out_nd_a, ds_ref)
                    if is_vismask:
                        gate_a = gate_cln
                    else:
                        fg_a = build_fg_mask(b["p_ego_cls"], pf_a, b["v_idx"], fg_prob_tau=fg_prob_tau, mode=fg_mode)
                        ar_a = auditor.audit_dev_maps(
                            dev_a, cov_2d, b["d_idx"], b["v_idx"], mod, build_gate=True,
                            d_drop_tol=dd_a, d_add_tol=da_a, enable_whole_fallback=True, collab_fg_mask=fg_a,
                        )
                        gate_a = ar_a["gate_L0"]
                    with torch.no_grad():
                        out_df_a = wrapper.forward_from_protocol(
                            f_adv, rlen, amat, mods_l,
                            custom_fuse_fn=build_gated_fuse_fn(gate_a, downsample_mode=args.downsample_mode, enable_b3=not args.disable_b3),
                        )
                    df_pb_a, df_ps_a, _ = _decode_np(wrapper, b_post, out_df_a, ds_ref)

                    for mt, (pb_x, ps_x) in (("no_defense", (nd_pb_a, nd_ps_a)), (meth_tag, (df_pb_a, df_ps_a))):
                        st_x = evaluate_frame_asr_strict(mode_s, cands_xy, cln_pb_np, cln_ps_np, pb_x, ps_x, gt_np, s_min=args.s_min)
                        asr_seq[mt][atk_name][mod][seq_name][0] += int(st_x["succ"])
                        asr_seq[mt][atk_name][mod][seq_name][1] += int(st_x["total"])

            # --- 2) 多车场景（>=3 辆协作车，即总车数 >=4）干净 AP@0.5 ---
            raw_m = train_utils.to_device(ds_mix.collate_batch_test([ds_mix[idx]]), wrapper.device_obj)
            p_m = wrapper.extract_protocol_features(raw_m["ego"])
            feat_m = p_m["heter_feature_2d"]
            n_tot = int(feat_m.shape[0])
            if n_tot >= 4:
                b_post_m = train_utils.to_device(_slim_batch_for_postprocess(raw_m), wrapper.device_obj)
                rlen_m, amat_m, mods_m, vmask_m = p_m["record_len"], p_m["affine_matrix"], list(p_m["agent_modality_list"]), p_m["cam_valid_mask"]
                with torch.no_grad():
                    o_ego_m = wrapper.forward_single_agent(feat_m[0:1])
                    p_ego_m = torch.sigmoid(o_ego_m["cls_preds"]).max(dim=1, keepdim=True)[0]
                    o_nd_m = wrapper.forward_from_protocol(feat_m, rlen_m, amat_m, mods_m)
                nd_pb_m, nd_ps_m, gt_m_np = _decode_np(wrapper, b_post_m, o_nd_m, ds_mix)
                gt_m_polys = list(common_utils.convert_format(gt_m_np))

                gates_m = []
                for ci in range(1, n_tot):
                    ptm_ci = raw_m["ego"]["pairwise_t_matrix"][0, ci, 0].detach().cpu().numpy()
                    dist_ci = torch.sqrt((gx - float(ptm_ci[0, 3])) ** 2 + (gy - float(ptm_ci[1, 3])) ** 2).view(1, 1, PROTO_H, PROTO_W)
                    di, vi = assign_dist5_vis3_indices_v2(dist_ci, b["los_factor"], b["expected_pts"], clear_pts_tau=c_tau, los_occ_tau=l_tau)
                    if is_vismask:
                        gates_m.append(build_visibility_mask_gate(vi[0, 0]).to(device=feat_m.device))
                    else:
                        dev_ci, dd_ci, da_ci, cov_ci, pf_ci = extract_single_cav_dev_and_signed(
                            wrapper, feat_m, amat_m, mods_m, vmask_m, p_ego_m, cav_idx=ci, k_tol=5, return_pfused=True
                        )
                        fg_ci = build_fg_mask(p_ego_m, pf_ci, vi[0, 0], fg_prob_tau=fg_prob_tau, mode=fg_mode)
                        ar_ci = auditor.audit_dev_maps(
                            dev_ci, cov_ci, di[0, 0], vi[0, 0], mods_m[ci], build_gate=True,
                            d_drop_tol=dd_ci, d_add_tol=da_ci, enable_whole_fallback=True, collab_fg_mask=fg_ci,
                        )
                        gates_m.append(ar_ci["gate_L0"])

                with torch.no_grad():
                    o_df_m = wrapper.forward_from_protocol(
                        feat_m, rlen_m, amat_m, mods_m,
                        custom_fuse_fn=build_gated_fuse_fn(torch.cat(gates_m, dim=0), downsample_mode=args.downsample_mode, enable_b3=not args.disable_b3),
                    )
                df_pb_m, df_ps_m, _ = _decode_np(wrapper, b_post_m, o_df_m, ds_mix)
                for mt, (pb_k, ps_k) in (("no_defense", (nd_pb_m, nd_ps_m)), (meth_tag, (df_pb_m, df_ps_m))):
                    rm = _match_standard_ap(pb_k, ps_k, gt_m_polys, 0.5)
                    bk = multicav_ge3_seq[mt][seq_name]
                    bk["tp"].extend(rm["tp"])
                    bk["fp"].extend(rm["fp"])
                    bk["score"].extend(rm["score"])
                    bk["gt"] += int(rm["gt"])

        print(f"  [StandardEval | {meth_tag}] 完成序列 {seq_name} ({time.time() - t0:.1f}s)", flush=True)

    # === 聚合全套 A0.4 标准指标与 2,000 次序列块 Bootstrap 95% CI ===
    boot_idx = build_seq_bootstrap_indices(seq_order, n_boot=BOOT_N, seed=BOOT_SEED)
    summary_metrics = OrderedDict()

    for mod in ("m1", "m2"):
        ap5 = paired_block_bootstrap_ap_diff(seq_order, ap50_seq[meth_tag][mod], ap50_seq["no_defense"][mod], boot_indices=boot_idx)
        ap7 = paired_block_bootstrap_ap_diff(seq_order, ap70_seq[meth_tag][mod], ap70_seq["no_defense"][mod], boot_indices=boot_idx)
        bap = paired_block_bootstrap_ap_diff(seq_order, blind_seq[meth_tag][mod], blind_seq["no_defense"][mod], boot_indices=boot_idx)
        hs_m = honest_sup_acc[mod]
        asr_m = OrderedDict()
        for atk in ATTACKS_5:
            h_df = {s: asr_seq[meth_tag][atk][mod][s][0] for s in seq_order}
            h_nd = {s: asr_seq["no_defense"][atk][mod][s][0] for s in seq_order}
            tots = {s: asr_seq["no_defense"][atk][mod][s][1] for s in seq_order}
            rd = paired_block_bootstrap_rate_diff(seq_order, h_df, h_nd, tots, boot_indices=boot_idx)
            asr_m[atk] = {
                meth_tag: {"asr": rd["val_a"], "ci95": rd["a_ci95"]},
                "no_defense": {"asr": rd["val_b"], "ci95": rd["b_ci95"]},
                "diff_vs_no_defense": rd["diff"],
                "diff_ci95": rd["diff_ci95"],
                "total_targets": rd["total"],
            }

        summary_metrics[mod] = OrderedDict({
            "clean_ap50": {meth_tag: {"val": ap5["val_a"], "ci95": ap5["a_ci95"]}, "no_defense": {"val": ap5["val_b"], "ci95": ap5["b_ci95"]}, "diff": ap5["diff"], "diff_ci95": ap5["diff_ci95"]},
            "clean_ap70": {meth_tag: {"val": ap7["val_a"], "ci95": ap7["a_ci95"]}, "no_defense": {"val": ap7["val_b"], "ci95": ap7["b_ci95"]}, "diff": ap7["diff"], "diff_ci95": ap7["diff_ci95"]},
            "clean_blind_ap50": {meth_tag: {"val": bap["val_a"], "ci95": bap["a_ci95"]}, "no_defense": {"val": bap["val_b"], "ci95": bap["b_ci95"]}, "diff": bap["diff"], "diff_ci95": bap["diff_ci95"]},
            "honest_suppression": {
                "overall_ratio_g_lt_0.5": round(hs_m["sup_grids"] / max(1, hs_m["tot_grids"]), 5),
                "reclaimed_weight_ratio": round(hs_m["w_num"] / max(1e-8, hs_m["w_den"]), 5),
                "by_visibility": {
                    vn: round(hs_m["by_vis"][vn][0] / max(1, hs_m["by_vis"][vn][1]), 5)
                    for vn in VIS_NAMES_3
                },
            },
            "attacks_asr": asr_m,
        })

    ge3_seqs = [s for s in seq_order if multicav_ge3_seq["no_defense"][s]["gt"] > 0]
    if ge3_seqs:
        ge3_boot = build_seq_bootstrap_indices(ge3_seqs, n_boot=BOOT_N, seed=BOOT_SEED)
        ge3_ap = paired_block_bootstrap_ap_diff(ge3_seqs, multicav_ge3_seq[meth_tag], multicav_ge3_seq["no_defense"], boot_indices=ge3_boot)
    else:
        ge3_ap = {"val_a": 0.0, "a_ci95": [0.0, 0.0], "val_b": 0.0, "b_ci95": [0.0, 0.0], "diff": 0.0, "diff_ci95": [0.0, 0.0]}

    out_report = OrderedDict({
        "meta": {
            "claim": "[临时结论 · 待人工代码复核]",
            "mode": args.mode,
            "method_tag": meth_tag,
            "downsample_mode": args.downsample_mode,
            "fg_mode": fg_mode,
            "enable_b3": not bool(args.disable_b3),
            "s_min": float(args.s_min),
            "num_sequences": len(seq_order),
            "num_ge3_collab_sequences": len(ge3_seqs),
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "pair_metrics_by_modality": summary_metrics,
        "multicav_ge3_clean_ap50": {
            meth_tag: {"val": ge3_ap["val_a"], "ci95": ge3_ap["a_ci95"]},
            "no_defense": {"val": ge3_ap["val_b"], "ci95": ge3_ap["b_ci95"]},
            "diff": ge3_ap["diff"],
            "diff_ci95": ge3_ap["diff_ci95"],
        },
    })
    _atomic_write_json(args.out_json, out_report)
    print(f"\n[完成] A0.4 标准指标报告已保存至 {args.out_json}")


def _resolve_eval_subset(raw_split: dict, subset_name: str, max_frames: Optional[int] = None) -> dict:
    """兼容 splits_v3.json (subsets[subset_name]) 与单子集 JSON，支持 'indices' 与 'sampled_indices' 及 --max_frames 截断。"""
    if "subsets" in raw_split and subset_name in raw_split["subsets"]:
        sub = raw_split["subsets"][subset_name]
        seq_map = sub["sequences"]
        seq_order = list(sub.get("seq_order", list(seq_map.keys())))
        seqs = OrderedDict(
            (s, {"indices": list(seq_map[s].get("indices", seq_map[s].get("sampled_indices", [])))})
            for s in seq_order
        )
    else:
        seq_order = list(raw_split["seq_order"])
        seqs = OrderedDict(
            (s, {"indices": list(raw_split["sequences"][s].get("indices", raw_split["sequences"][s].get("sampled_indices", [])))})
            for s in seq_order
        )

    if max_frames is not None and int(max_frames) > 0:
        rem = int(max_frames)
        slim_seqs = OrderedDict()
        for s in seq_order:
            if rem <= 0:
                break
            take = seqs[s]["indices"][:rem]
            if take:
                slim_seqs[s] = {"indices": take}
                rem -= len(take)
        return {"seq_order": list(slim_seqs.keys()), "sequences": slim_seqs}
    return {"seq_order": seq_order, "sequences": seqs}


def main():
    parser = argparse.ArgumentParser(description="sprint-cs2027 统一标准指标评测与 B4 扫描")
    parser.add_argument("--mode", type=str, default="standard_eval", choices=["standard_eval", "b4_gamma_scan", "vis_mask_baseline"])
    parser.add_argument("--data_dir", type=str, default=OPV2V_TRAIN_DIR)
    parser.add_argument("--eval_split_json", type=str, default=DEFAULT_EVAL_SPLIT)
    parser.add_argument("--subset", type=str, default="dev_fast", help="从 splits_v3.json 读取的子集名 (dev_fast / dev_full)")
    parser.add_argument("--max_frames", type=int, default=0, help=">0 时仅跑前 N 帧供冒烟测试使用")
    parser.add_argument("--base_calib_json", type=str, default=DEFAULT_BASE_CALIB)
    parser.add_argument("--frozen_gating_json", type=str, default=DEFAULT_FROZEN_GATING)
    parser.add_argument("--downsample_mode", type=str, default="min_pool", choices=["min_pool", "area"])
    parser.add_argument("--fg_mode", type=str, default=None, choices=[None, "v1", "v0"])
    parser.add_argument("--disable_b3", action="store_true")
    parser.add_argument("--s_min", type=float, default=0.20)
    parser.add_argument("--out_json", type=str, required=True)
    args = parser.parse_args()

    with open(args.eval_split_json, "r", encoding="utf-8") as f:
        raw_split = json.load(f)
    eval_split = _resolve_eval_subset(raw_split, args.subset, max_frames=args.max_frames if args.max_frames > 0 else None)

    wrapper = HEALBaseWrapper(device="cuda", use_cav=7)
    ds_m1 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m1")
    ds_m2 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m2")
    gx, gy = build_bev_grid_xy_256(device="cuda", dtype=torch.float32)
    auditor = BUDFusionHEALAuditor(args.base_calib_json, gating_calib_path=args.frozen_gating_json)

    if args.mode == "b4_gamma_scan":
        _run_b4_gamma_scan(args, wrapper, ds_m1, ds_m2, eval_split, auditor, gx, gy)
    else:
        ds_mix = wrapper.build_dataset(data_dir=args.data_dir, use_cav=7, visualize=True, collab_modality="mixed", max_cav=7)
        _run_standard_or_vismask_eval(args, wrapper, ds_m1, ds_m2, ds_mix, eval_split, auditor, gx, gy)


if __name__ == "__main__":
    main()
