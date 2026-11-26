# -*- coding: utf-8 -*-
"""
步骤 ④-5：冻结版 BUD-Fusion Dev-320 全维度验收评测脚本。

涵盖用户全部 4 项收尾要求：
1. 加载纯 Fit 冻结的 step4_4_bud_frozen_calibrator.json（可见性分档余弦门控 + 整车级回退高门限）；
2. 评测 9 大工况（Clean、0.5σ/1.0σ/2.0σ/ε=4.0 全局 PGD、局部擦除 1.0σ、局部鬼影 2.0σ、
   遮挡区伪造 2.0σ、0~6m 近场移除 2.0σ）× 5 方法（ego_only、no_defense、whole_cav_drop、
   bud_local_only 消融、bud_frozen 两级混合）；
3. 全部方法对比统一报告「成对差值 2,000 次按序列 Block-Bootstrap 95% 置信区间」；
4. 按序列断点缓存，严禁触碰任何 Fit 或封存测试集序列。
"""

import argparse
import json
import math
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

from src_v2_mm.bud_fusion_heal import (
    BUDFusionHEALAuditor,
    build_gated_fuse_fn,
    build_whole_cav_reject_fuse_fn,
    compute_shift_tolerant_semantic_maps,
)
from src_v2_mm.eval_stats_utils import (
    build_seq_bootstrap_indices,
    paired_block_bootstrap_ap_diff,
    paired_block_bootstrap_rate_diff,
)
from src_v2_mm.heal_attacks import (
    generate_local_ghost_spoofing_heal,
    generate_local_near0_6m_attack_heal,
    generate_local_occluded_ghost_heal,
    generate_local_removal_heal,
    generate_pgd_removal_heal,
    generate_scaled_pgd_removal_heal,
)
from src_v2_mm.heal_base_wrapper import HEALBaseWrapper
from src_v2_mm.step4_1_v2_core import (
    GRID_RES,
    MIN_XY,
    PROTO_H,
    PROTO_W,
    TRAIN_FULL19_DIR,
    _masked_avg_pool3x3,
    assign_dist5_vis3_indices_v2,
    build_bev_grid_xy_256,
    compute_auroc,
    compute_ego_expected_pts_and_los_256,
    extract_frame_context_and_clean,
    find_exact_fpr_threshold,
    load_ego_lidar_np,
)

DEFAULT_EVAL_SPLIT = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_3_eval320_split.json")
DEFAULT_BASE_CALIB = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_1_v2_frozen_calibrator.json")
DEFAULT_OLD_GATING = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_2_bud_gating_calibrator.json")
DEFAULT_FROZEN_GATING = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_4_bud_frozen_calibrator.json")
DEFAULT_CACHE_DIR = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_5_seq_cache")
DEFAULT_OUT_JSON = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_5_bud_frozen_320_report.json")

CONDITIONS_9 = [
    "clean",
    "pgd_0.5sigma",
    "pgd_1.0sigma",
    "pgd_2.0sigma",
    "pgd_eps4.0",
    "local_removal_1.0sigma",
    "local_ghost_2.0sigma",
    "local_occluded_ghost_2.0sigma",
    "local_near0_6m_2.0sigma",
]
METHODS_5 = ["ego_only", "no_defense", "whole_cav_drop", "bud_local_only", "bud_frozen"]
MODALITIES_2 = ["m1", "m2"]
BOOT_N = 2000
BOOT_SEED = 20261120


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


def _match_tp_fp_and_hits(pred_boxes, pred_score, gt_np: np.ndarray, gt_polygons: list, iou_thresh: float):
    from opencood.utils import common_utils

    gt_cnt = int(gt_np.shape[0])
    matched_gt = [False] * gt_cnt
    if pred_boxes is None or len(pred_boxes) == 0:
        return [], [], [], gt_cnt, matched_gt

    det_np = common_utils.torch_tensor_to_numpy(pred_boxes)
    sc_np = common_utils.torch_tensor_to_numpy(pred_score).reshape(-1)
    if det_np.shape[0] == 0:
        return [], [], [], gt_cnt, matched_gt

    order = np.argsort(-sc_np)
    sc_sorted = sc_np[order]
    det_polys = list(common_utils.convert_format(det_np))
    rem_polys = list(gt_polygons)
    rem_idxs = list(range(gt_cnt))
    fp_list, tp_list = [], []

    for k in range(order.shape[0]):
        det_p = det_polys[int(order[k])]
        if not rem_polys:
            fp_list.append(1)
            tp_list.append(0)
            continue
        ious = common_utils.compute_iou(det_p, rem_polys)
        if np.max(ious) < iou_thresh:
            fp_list.append(1)
            tp_list.append(0)
            continue
        fp_list.append(0)
        tp_list.append(1)
        best_j = int(np.argmax(ious))
        rem_polys.pop(best_j)
        matched_gt[rem_idxs.pop(best_j)] = True
    return tp_list, fp_list, sc_sorted.tolist(), gt_cnt, matched_gt


def _analyze_frame_blind_gts(ds_m1, idx: int, ego_pb, ego_ps, ego_gb, cov_2d: torch.Tensor) -> dict:
    from matplotlib.path import Path as MplPath
    from opencood.utils import common_utils

    gt_np = common_utils.torch_tensor_to_numpy(ego_gb)
    gt_polys = list(common_utils.convert_format(gt_np))
    ego_res_05 = _match_tp_fp_and_hits(ego_pb, ego_ps, gt_np, gt_polys, 0.5)
    ego_res_07 = _match_tp_fp_and_hits(ego_pb, ego_ps, gt_np, gt_polys, 0.7)
    ego_matched = ego_res_05[4]
    ego_pts = load_ego_lidar_np(ds_m1, idx)
    cov_np = cov_2d.detach().cpu().numpy()

    blind_indices = []
    for i in range(gt_np.shape[0]):
        cx = float(gt_np[i, :4, 0].mean())
        cy = float(gt_np[i, :4, 1].mean())
        # 排除本车 Ego 自身车体 GT 框（位于原点附近 r≈0.5m 且自身点云已清空）
        if math.hypot(cx, cy) < 2.5:
            continue
        ix = int(math.floor((cx - MIN_XY) / GRID_RES))
        iy = int(math.floor((cy - MIN_XY) / GRID_RES))
        if not (0 <= ix < PROTO_W and 0 <= iy < PROTO_H and bool(cov_np[iy, ix])):
            continue
        if ego_matched[i]:
            continue
        corners = gt_np[i, :4, :2]
        min_x, max_x = float(corners[:, 0].min()), float(corners[:, 0].max())
        min_y, max_y = float(corners[:, 1].min()), float(corners[:, 1].max())
        z_min = float(gt_np[i, :, 2].min()) - 0.3
        z_max = float(gt_np[i, :, 2].max()) + 0.3
        c_pts = ego_pts[
            (ego_pts[:, 0] >= min_x) & (ego_pts[:, 0] <= max_x)
            & (ego_pts[:, 1] >= min_y) & (ego_pts[:, 1] <= max_y)
            & (ego_pts[:, 2] >= z_min) & (ego_pts[:, 2] <= z_max)
        ]
        n_in = int(MplPath(corners).contains_points(c_pts[:, :2]).sum()) if c_pts.shape[0] > 0 else 0
        if n_in <= 5:
            blind_indices.append(i)

    return {
        "gt_np": gt_np,
        "gt_polys": gt_polys,
        "ego_res_05": ego_res_05,
        "ego_res_07": ego_res_07,
        "blind_indices": blind_indices,
    }


@torch.no_grad()
def _extract_out_and_dev_maps(wrapper, p_dict, feat_in, p_ego_cls, cov_ego, k_tol: int = 5):
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

    rlen = p_dict["record_len"]
    amat = p_dict["affine_matrix"]
    mods = p_dict["agent_modality_list"]
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
        c_raw = (1.0 - F.cosine_similarity(z_ego, z_cav, dim=1, eps=1e-6)).unsqueeze(1)
        if lvl == 0:
            c_256 = _masked_avg_pool3x3(c_raw, cov_ego)
        else:
            c_256 = F.interpolate(c_raw, size=(PROTO_H, PROTO_W), mode="bilinear", align_corners=False) * cov_f
        cos_maps.append(c_256)

    p_fused_cls = torch.sigmoid(out["cls_preds"]).max(dim=1, keepdim=True)[0]
    d_sem = torch.abs(p_ego_cls - p_fused_cls) * cov_f
    dev_4ch = torch.cat(cos_maps + [d_sem], dim=1)[0]
    d_drop_tol, d_add_tol = compute_shift_tolerant_semantic_maps(p_ego_cls, p_fused_cls, cov_ego, k_tol=k_tol)
    return out, dev_4ch, d_drop_tol, d_add_tol, p_fused_cls


def _eval_one_sequence(
    wrapper, ds_m1, ds_m2, auditor, old_auditor, seq_name: str, indices: list, gx, gy, k_tol: int = 5
) -> dict:
    """运行单条 Dev 序列的全部 9 工况 × 5 方法评测（含遮挡区伪造修复前后检出率对比）并打包成缓存 JSON。"""
    from opencood.tools import train_utils

    t0 = time.time()
    vis_v2 = (auditor.gating_calib or {}).get("visibility_v2", None)
    fg_prob_tau = float((auditor.gating_calib or {}).get("gating_params", {}).get("fg_prob_tau", 0.20))

    seq_rec = {
        "seq_name": seq_name,
        "num_frames": len(indices),
        "ap": {
            m: {
                c: {
                    meth: {"tp50": [], "fp50": [], "tp70": [], "fp70": [], "score": [], "gt": 0}
                    for meth in METHODS_5
                }
                for c in CONDITIONS_9
            }
            for m in MODALITIES_2
        },
        "ap_valid_only": {
            m: {
                c: {
                    meth: {"tp50": [], "fp50": [], "tp70": [], "fp70": [], "score": [], "gt": 0}
                    for meth in METHODS_5
                }
                for c in ("local_occluded_ghost_2.0sigma", "local_near0_6m_2.0sigma")
            }
            for m in MODALITIES_2
        },
        "blind": {
            m: {c: {meth: 0 for meth in METHODS_5} for c in CONDITIONS_9}
            for m in MODALITIES_2
        },
        "blind_valid_only": {
            m: {
                c: {meth: 0 for meth in METHODS_5}
                for c in ("local_occluded_ghost_2.0sigma", "local_near0_6m_2.0sigma")
            }
            for m in MODALITIES_2
        },
        "blind_total": 0,
        "blind_total_valid_only": {
            m: {c: 0 for c in ("local_occluded_ghost_2.0sigma", "local_near0_6m_2.0sigma")}
            for m in MODALITIES_2
        },
        "fallback_cnt": {m: {c: 0 for c in CONDITIONS_9} for m in MODALITIES_2},
        "attack_valid_cnt": {m: {c: 0 for c in CONDITIONS_9} for m in MODALITIES_2},
        "clean_scores_t2": {"m1": [], "m2": []},
        "clean_scores_t4": {"m1": [], "m2": []},
        "clean_blind_comp_cnt": {"m1": [], "m2": []},
        # 遮挡区伪造修复前(OldBUD) vs 修复后(NewBUD)检出得分记录
        "occ_det_scores": {
            m: {"clean_old": [], "occ_old": [], "clean_new": [], "occ_new": []}
            for m in MODALITIES_2
        },
    }

    def _record_det(mod: str, cond: str, meth: str, r05, r07, blind_idxs: list, is_valid_atk: bool = True):
        d = seq_rec["ap"][mod][cond][meth]
        d["tp50"].extend(r05[0])
        d["fp50"].extend(r05[1])
        d["tp70"].extend(r07[0])
        d["fp70"].extend(r07[1])
        d["score"].extend(r05[2])
        d["gt"] += int(r05[3])
        if blind_idxs:
            seq_rec["blind"][mod][cond][meth] += sum(1 for bi in blind_idxs if r05[4][bi])
        if is_valid_atk and cond in seq_rec["ap_valid_only"][mod]:
            dv = seq_rec["ap_valid_only"][mod][cond][meth]
            dv["tp50"].extend(r05[0])
            dv["fp50"].extend(r05[1])
            dv["tp70"].extend(r07[0])
            dv["fp70"].extend(r07[1])
            dv["score"].extend(r05[2])
            dv["gt"] += int(r05[3])
            if blind_idxs:
                seq_rec["blind_valid_only"][mod][cond][meth] += sum(1 for bi in blind_idxs if r05[4][bi])

    for fi, idx in enumerate(indices):
        c_tau = float(vis_v2["clear_pts_tau"]) if vis_v2 is not None else 73.853
        l_tau = float(vis_v2.get("los_occ_tau", 0.50)) if vis_v2 is not None else 0.50
        b = extract_frame_context_and_clean(wrapper, ds_m1, ds_m2, idx, gx, gy, clear_pts_tau=c_tau, los_occ_tau=l_tau)
        d_idx = b["d_idx"]
        v_idx = b["v_idx"]
        occ_mask_4d = b["occ_mask_ego"].float()

        raw_b1 = train_utils.to_device(ds_m1.collate_batch_test([ds_m1[idx]]), "cpu")
        b_post = train_utils.to_device(_slim_batch_for_postprocess(raw_b1), wrapper.device_obj)

        with torch.no_grad():
            ego_out = wrapper.forward_single_agent(b["p1"]["heter_feature_2d"][0:1])
            ego_pb, ego_ps, ego_gb = wrapper.decode_boxes(b_post, ego_out, ds_m1)

        # 客观盲区按对应协作者覆盖区分别判定；主实验中 m1 用自身完整覆盖范围，m2 用相机覆盖范围
        binfo_m1 = _analyze_frame_blind_gts(ds_m1, idx, ego_pb, ego_ps, ego_gb, b["cov_ego_m1_full"][0, 0])
        binfo_m2 = _analyze_frame_blind_gts(ds_m1, idx, ego_pb, ego_ps, ego_gb, b["cov_ego"][0, 0])
        gt_np, gt_polys = binfo_m2["gt_np"], binfo_m2["gt_polys"]
        ego_r05, ego_r07 = binfo_m2["ego_res_05"], binfo_m2["ego_res_07"]
        seq_rec["blind_total"] += len(binfo_m2["blind_indices"])
        seq_rec.setdefault("blind_total_by_mod", {"m1": 0, "m2": 0})
        seq_rec["blind_total_by_mod"]["m1"] += len(binfo_m1["blind_indices"])
        seq_rec["blind_total_by_mod"]["m2"] += len(binfo_m2["blind_indices"])

        for mod, p_dict in (("m1", b["p1"]), ("m2", b["p2"])):
            feat = p_dict["heter_feature_2d"]
            rlen = p_dict["record_len"]
            amat = p_dict["affine_matrix"]
            mods_l = list(p_dict["agent_modality_list"])
            vmask = p_dict["cam_valid_mask"]
            # 主实验：m1 使用自身完整激光雷达覆盖范围且不限制攻击区；m2 使用相机覆盖范围
            cov_ego_mod = b["cov_ego_m1_full"] if mod == "m1" else b["cov_ego"]
            cov_2d = cov_ego_mod[0, 0]
            blind_idxs = binfo_m1["blind_indices"] if mod == "m1" else binfo_m2["blind_indices"]
            preg = None

            # 生成本帧全部 8 种攻击（Clean 直接用原特征，统一使用关键字参数防错位）
            f_p05, _ = generate_scaled_pgd_removal_heal(wrapper, feat, rlen, amat, mods_l, vmask, attacker_idx=1, sigma_mult=0.5, steps=8, perturb_region_mask=preg)
            f_p10, _ = generate_scaled_pgd_removal_heal(wrapper, feat, rlen, amat, mods_l, vmask, attacker_idx=1, sigma_mult=1.0, steps=8, perturb_region_mask=preg)
            f_p20, _ = generate_scaled_pgd_removal_heal(wrapper, feat, rlen, amat, mods_l, vmask, attacker_idx=1, sigma_mult=2.0, steps=8, perturb_region_mask=preg)
            f_pe4, _ = generate_pgd_removal_heal(wrapper, feat, rlen, amat, mods_l, vmask, attacker_idx=1, eps=4.0, steps=10, perturb_region_mask=preg)
            f_lr, st_lr = generate_local_removal_heal(wrapper, feat, rlen, amat, mods_l, vmask, attacker_idx=1, num_targets=2, patch_radius=5, sigma_mult=1.0, steps=10, perturb_region_mask=preg)
            f_lg, st_lg = generate_local_ghost_spoofing_heal(wrapper, feat, rlen, amat, mods_l, vmask, attacker_idx=1, num_spoofs=2, patch_radius=5, sigma_mult=2.0, steps=10, frame_seed=idx, perturb_region_mask=preg)
            f_lo, st_lo = generate_local_occluded_ghost_heal(wrapper, feat, rlen, amat, mods_l, vmask, occluded_mask_ego=occ_mask_4d, attacker_idx=1, num_spoofs=2, patch_radius=5, sigma_mult=2.0, steps=12, frame_seed=idx, perturb_region_mask=preg)
            f_ln, st_ln = generate_local_near0_6m_attack_heal(wrapper, feat, rlen, amat, mods_l, vmask, attacker_idx=1, sigma_mult=2.0, steps=12, frame_seed=idx, perturb_region_mask=preg, enable_ghost_fallback=False)

            cond_feats = {
                "clean": (feat, {"valid": True}),
                "pgd_0.5sigma": (f_p05, {"valid": True}),
                "pgd_1.0sigma": (f_p10, {"valid": True}),
                "pgd_2.0sigma": (f_p20, {"valid": True}),
                "pgd_eps4.0": (f_pe4, {"valid": True}),
                "local_removal_1.0sigma": (f_lr, st_lr),
                "local_ghost_2.0sigma": (f_lg, st_lg),
                "local_occluded_ghost_2.0sigma": (f_lo, st_lo),
                "local_near0_6m_2.0sigma": (f_ln, st_ln),
            }

            for cond in CONDITIONS_9:
                f_cur, st_meta = cond_feats[cond]
                is_valid_atk = bool(
                    st_meta is None
                    or (
                        st_meta.get("valid", True)
                        and st_meta.get("num_targets", 1) > 0
                        and st_meta.get("num_spoof_anchors", 1) > 0
                    )
                )
                if is_valid_atk:
                    seq_rec["attack_valid_cnt"][mod][cond] += 1
                    if cond in seq_rec["blind_total_valid_only"][mod]:
                        seq_rec["blind_total_valid_only"][mod][cond] += len(blind_idxs)

                # 1) Ego-Only
                _record_det(mod, cond, "ego_only", ego_r05, ego_r07, blind_idxs, is_valid_atk)

                # 2) No-Defense + 偏差提取
                out_nd, dev_4ch, d_drop, d_add, p_fused_cls = _extract_out_and_dev_maps(
                    wrapper, p_dict, f_cur, b["p_ego_cls"], cov_ego_mod, k_tol=k_tol
                )
                fg_mask = F.max_pool2d(p_fused_cls, kernel_size=3, stride=1, padding=1)[0, 0] > fg_prob_tau
                nd_pb, nd_ps, _ = wrapper.decode_boxes(b_post, out_nd, ds_m1)
                nd_r05 = _match_tp_fp_and_hits(nd_pb, nd_ps, gt_np, gt_polys, 0.5)
                nd_r07 = _match_tp_fp_and_hits(nd_pb, nd_ps, gt_np, gt_polys, 0.7)
                _record_det(mod, cond, "no_defense", nd_r05, nd_r07, blind_idxs, is_valid_atk)

                # 3) 审计并生成局部与两级门控
                ar_loc = auditor.audit_dev_maps(
                    dev_4ch, cov_2d, d_idx, v_idx, mod, build_gate=True,
                    d_drop_tol=d_drop, d_add_tol=d_add, enable_whole_fallback=False, collab_fg_mask=fg_mask
                )
                ar_frz = auditor.audit_dev_maps(
                    dev_4ch, cov_2d, d_idx, v_idx, mod, build_gate=True,
                    d_drop_tol=d_drop, d_add_tol=d_add, enable_whole_fallback=True, collab_fg_mask=fg_mask
                )

                # 记录遮挡区伪造修复前(OldBUD) vs 修复后(NewBUD)检出得分
                if cond in ("clean", "local_occluded_ghost_2.0sigma") and (cond == "clean" or is_valid_atk):
                    inv_new = 1.0 - ar_frz["gate_L0"][0, 0][cov_2d]
                    kn = min(25, int(inv_new.numel()))
                    sc_new = float(torch.topk(inv_new, kn).values.mean().item()) if kn > 0 else 0.0
                    key_tag = "clean" if cond == "clean" else "occ"
                    seq_rec["occ_det_scores"][mod][f"{key_tag}_new"].append(sc_new)
                    if old_auditor is not None:
                        ar_old = old_auditor.audit_dev_maps(
                            dev_4ch, cov_2d, d_idx, v_idx, mod, build_gate=True,
                            d_drop_tol=d_drop, d_add_tol=d_add, enable_whole_fallback=False
                        )
                        inv_old = 1.0 - ar_old["gate_L0"][0, 0][cov_2d]
                        sc_old = float(torch.topk(inv_old, kn).values.mean().item()) if kn > 0 else 0.0
                        seq_rec["occ_det_scores"][mod][f"{key_tag}_old"].append(sc_old)

                if cond == "clean":
                    seq_rec["clean_scores_t2"][mod].append(float(ar_frz["score_t2"]))
                    seq_rec["clean_scores_t4"][mod].append(float(ar_frz["score_t4"]))
                    comp_cnt = sum(1 for bi in blind_idxs if nd_r05[4][bi])
                    seq_rec["clean_blind_comp_cnt"][mod].append(int(comp_cnt))

                # 4) Whole-CAV-Drop（>1.0 整车降权回退 Ego-Only）
                if ar_frz["score_t4"] > 1.0:
                    _record_det(mod, cond, "whole_cav_drop", ego_r05, ego_r07, blind_idxs, is_valid_atk)
                else:
                    _record_det(mod, cond, "whole_cav_drop", nd_r05, nd_r07, blind_idxs, is_valid_atk)

                # 5) BUD-Local-Only（仅局部门控）
                g_loc = ar_loc["gate_L0"]
                if float(g_loc[0, 0][cov_2d].min().item()) >= 1.0 - 1e-6:
                    loc_r05, loc_r07 = nd_r05, nd_r07
                else:
                    with torch.no_grad():
                        out_loc = wrapper.forward_from_protocol(
                            f_cur, rlen, amat, mods_l, custom_fuse_fn=build_gated_fuse_fn(g_loc)
                        )
                        lpb, lps, _ = wrapper.decode_boxes(b_post, out_loc, ds_m1)
                    loc_r05 = _match_tp_fp_and_hits(lpb, lps, gt_np, gt_polys, 0.5)
                    loc_r07 = _match_tp_fp_and_hits(lpb, lps, gt_np, gt_polys, 0.7)
                _record_det(mod, cond, "bud_local_only", loc_r05, loc_r07, blind_idxs, is_valid_atk)

                # 6) BUD-Frozen（局部门控 + 高门限整车回退）
                if ar_frz["whole_fallback"]:
                    seq_rec["fallback_cnt"][mod][cond] += 1
                    _record_det(mod, cond, "bud_frozen", ego_r05, ego_r07, blind_idxs, is_valid_atk)
                else:
                    _record_det(mod, cond, "bud_frozen", loc_r05, loc_r07, blind_idxs, is_valid_atk)

        if (fi + 1) % 10 == 0 or (fi + 1) == len(indices):
            print(f"  [{seq_name}] 进度 {fi + 1}/{len(indices)} ({time.time() - t0:.1f}s)", flush=True)

    seq_rec["elapsed_sec"] = round(time.time() - t0, 2)
    return seq_rec


def _aggregate_with_paired_bootstrap(seq_names: list, seq_caches: dict) -> dict:
    """聚合全部序列缓存并统一计算成对差值 Block-Bootstrap 95% 置信区间。"""
    boot_idx = build_seq_bootstrap_indices(seq_names, n_boot=BOOT_N, seed=BOOT_SEED)

    ap_summary = OrderedDict()
    paired_ap_diffs = OrderedDict()
    blind_summary = OrderedDict()
    paired_blind_diffs = OrderedDict()
    fallback_summary = OrderedDict()
    valid_summary = OrderedDict()

    compare_pairs = [
        ("bud_frozen_vs_no_defense", "bud_frozen", "no_defense"),
        ("bud_frozen_vs_whole_cav_drop", "bud_frozen", "whole_cav_drop"),
        ("bud_frozen_vs_ego_only", "bud_frozen", "ego_only"),
        ("bud_frozen_vs_bud_local_only", "bud_frozen", "bud_local_only"),
    ]

    for mod in MODALITIES_2:
        seq_blind_totals = {
            s: int(seq_caches[s].get("blind_total_by_mod", {}).get(mod, seq_caches[s]["blind_total"]))
            for s in seq_names
        }
        ap_summary[mod] = OrderedDict()
        paired_ap_diffs[mod] = OrderedDict()
        blind_summary[mod] = OrderedDict()
        paired_blind_diffs[mod] = OrderedDict()
        fallback_summary[mod] = OrderedDict()
        valid_summary[mod] = OrderedDict()

        for cond in CONDITIONS_9:
            ap_summary[mod][cond] = OrderedDict()
            paired_ap_diffs[mod][cond] = OrderedDict()
            blind_summary[mod][cond] = OrderedDict()
            paired_blind_diffs[mod][cond] = OrderedDict()

            tot_frames = sum(int(seq_caches[s]["num_frames"]) for s in seq_names)
            fb_cnt = sum(int(seq_caches[s]["fallback_cnt"][mod][cond]) for s in seq_names)
            val_cnt = sum(int(seq_caches[s]["attack_valid_cnt"][mod][cond]) for s in seq_names)
            fallback_summary[mod][cond] = {"count": fb_cnt, "rate": round(fb_cnt / max(tot_frames, 1), 4)}
            valid_summary[mod][cond] = {"valid_frames": val_cnt, "total_frames": tot_frames}

            # 组装各方法按序列 AP 映射
            seq_maps_50 = {}
            seq_maps_70 = {}
            seq_bhits = {}
            for meth in METHODS_5:
                seq_maps_50[meth] = {
                    s: {
                        "tp": np.asarray(seq_caches[s]["ap"][mod][cond][meth]["tp50"], dtype=np.float64),
                        "fp": np.asarray(seq_caches[s]["ap"][mod][cond][meth]["fp50"], dtype=np.float64),
                        "score": np.asarray(seq_caches[s]["ap"][mod][cond][meth]["score"], dtype=np.float64),
                        "gt": int(seq_caches[s]["ap"][mod][cond][meth]["gt"]),
                    }
                    for s in seq_names
                }
                seq_maps_70[meth] = {
                    s: {
                        "tp": np.asarray(seq_caches[s]["ap"][mod][cond][meth]["tp70"], dtype=np.float64),
                        "fp": np.asarray(seq_caches[s]["ap"][mod][cond][meth]["fp70"], dtype=np.float64),
                        "score": np.asarray(seq_caches[s]["ap"][mod][cond][meth]["score"], dtype=np.float64),
                        "gt": int(seq_caches[s]["ap"][mod][cond][meth]["gt"]),
                    }
                    for s in seq_names
                }
                seq_bhits[meth] = {s: int(seq_caches[s]["blind"][mod][cond][meth]) for s in seq_names}

            # 统一通过成对 bootstrap 生成各方法均值与成对差值 95% CI
            for pname, ma, mb in compare_pairs:
                r50 = paired_block_bootstrap_ap_diff(seq_names, seq_maps_50[ma], seq_maps_50[mb], boot_indices=boot_idx)
                r70 = paired_block_bootstrap_ap_diff(seq_names, seq_maps_70[ma], seq_maps_70[mb], boot_indices=boot_idx)
                rbl = paired_block_bootstrap_rate_diff(seq_names, seq_bhits[ma], seq_bhits[mb], seq_blind_totals, boot_indices=boot_idx)
                paired_ap_diffs[mod][cond][pname] = {"ap50": r50, "ap70": r70}
                paired_blind_diffs[mod][cond][pname] = rbl

                if ma not in ap_summary[mod][cond]:
                    ap_summary[mod][cond][ma] = {
                        "ap50": {"val": r50["val_a"], "ci95": r50["a_ci95"]},
                        "ap70": {"val": r70["val_a"], "ci95": r70["a_ci95"]},
                    }
                    blind_summary[mod][cond][ma] = {
                        "val": rbl["val_a"], "ci95": rbl["a_ci95"],
                        "hits": sum(seq_bhits[ma].values()), "total_blind_gt": rbl["total"],
                    }
                if mb not in ap_summary[mod][cond]:
                    ap_summary[mod][cond][mb] = {
                        "ap50": {"val": r50["val_b"], "ci95": r50["b_ci95"]},
                        "ap70": {"val": r70["val_b"], "ci95": r70["b_ci95"]},
                    }
                    blind_summary[mod][cond][mb] = {
                        "val": rbl["val_b"], "ci95": rbl["b_ci95"],
                        "hits": sum(seq_bhits[mb].values()), "total_blind_gt": rbl["total"],
                    }

    # 分组盲区补全误报率（Tier 4 vs Tier 2 成对差值 CI）
    fpr_paired = OrderedDict()
    for mod in MODALITIES_2:
        fpr_paired[mod] = OrderedDict()
        for grp_name, pred_fn in (
            ("overall", lambda c: True),
            ("0", lambda c: c == 0),
            ("1", lambda c: c == 1),
            ("ge2", lambda c: c >= 2),
        ):
            h_t4, h_t2, tots = {}, {}, {}
            for s in seq_names:
                s2 = seq_caches[s]["clean_scores_t2"][mod]
                s4 = seq_caches[s]["clean_scores_t4"][mod]
                cc = seq_caches[s]["clean_blind_comp_cnt"][mod]
                idxs = [i for i, v in enumerate(cc) if pred_fn(v)]
                tots[s] = len(idxs)
                h_t4[s] = sum(1 for i in idxs if s4[i] > 1.0)
                h_t2[s] = sum(1 for i in idxs if s2[i] > 1.0)
            fpr_paired[mod][grp_name] = paired_block_bootstrap_rate_diff(
                seq_names, h_t4, h_t2, tots, boot_indices=boot_idx
            )

    # 仅在攻击实际触发的有效帧子集（valid==True）上单独汇总遮挡伪造与 0~6m 近场移除（防止未触发帧稀释）
    valid_only_subset = OrderedDict()
    for mod in MODALITIES_2:
        valid_only_subset[mod] = OrderedDict()
        for cond in ("local_occluded_ghost_2.0sigma", "local_near0_6m_2.0sigma"):
            valid_only_subset[mod][cond] = OrderedDict()
            seq_btot_v = {s: int(seq_caches[s]["blind_total_valid_only"][mod][cond]) for s in seq_names}
            s50, s70, sbh = {}, {}, {}
            for meth in METHODS_5:
                s50[meth] = {
                    s: {
                        "tp": np.asarray(seq_caches[s]["ap_valid_only"][mod][cond][meth]["tp50"], dtype=np.float64),
                        "fp": np.asarray(seq_caches[s]["ap_valid_only"][mod][cond][meth]["fp50"], dtype=np.float64),
                        "score": np.asarray(seq_caches[s]["ap_valid_only"][mod][cond][meth]["score"], dtype=np.float64),
                        "gt": int(seq_caches[s]["ap_valid_only"][mod][cond][meth]["gt"]),
                    }
                    for s in seq_names
                }
                s70[meth] = {
                    s: {
                        "tp": np.asarray(seq_caches[s]["ap_valid_only"][mod][cond][meth]["tp70"], dtype=np.float64),
                        "fp": np.asarray(seq_caches[s]["ap_valid_only"][mod][cond][meth]["fp70"], dtype=np.float64),
                        "score": np.asarray(seq_caches[s]["ap_valid_only"][mod][cond][meth]["score"], dtype=np.float64),
                        "gt": int(seq_caches[s]["ap_valid_only"][mod][cond][meth]["gt"]),
                    }
                    for s in seq_names
                }
                sbh[meth] = {s: int(seq_caches[s]["blind_valid_only"][mod][cond][meth]) for s in seq_names}
            for pname, ma, mb in compare_pairs:
                valid_only_subset[mod][cond][pname] = {
                    "ap50": paired_block_bootstrap_ap_diff(seq_names, s50[ma], s50[mb], boot_indices=boot_idx),
                    "ap70": paired_block_bootstrap_ap_diff(seq_names, s70[ma], s70[mb], boot_indices=boot_idx),
                    "blind_recall": paired_block_bootstrap_rate_diff(seq_names, sbh[ma], sbh[mb], seq_btot_v, boot_indices=boot_idx),
                }

    # 汇总「只在遮挡区伪造」修复前(OldBUD) vs 修复后(NewBUD) 检出率对比
    occ_det_cmp = OrderedDict()
    for mod in MODALITIES_2:
        occ_det_cmp[mod] = OrderedDict()
        for tag in ("old", "new"):
            neg = np.asarray([v for s in seq_names for v in seq_caches[s].get("occ_det_scores", {}).get(mod, {}).get(f"clean_{tag}", [])], dtype=np.float64)
            pos = np.asarray([v for s in seq_names for v in seq_caches[s].get("occ_det_scores", {}).get(mod, {}).get(f"occ_{tag}", [])], dtype=np.float64)
            if neg.size > 0 and pos.size > 0:
                th5 = find_exact_fpr_threshold(neg, 0.05)
                occ_det_cmp[mod][f"{tag}_bud"] = {
                    "auroc": round(compute_auroc(neg, pos), 5),
                    "tpr_at_5pct_fpr": round(float((pos > th5).mean()), 5),
                    "n_clean": int(neg.size),
                    "n_valid_occ": int(pos.size),
                }

    return OrderedDict({
        "ap_results": ap_summary,
        "paired_ap_diffs": paired_ap_diffs,
        "blind_spot_recall": blind_summary,
        "paired_blind_recall_diffs": paired_blind_diffs,
        "valid_only_new_attacks": valid_only_subset,
        "occluded_ghost_det_before_vs_after": occ_det_cmp,
        "whole_fallback_stats": fallback_summary,
        "attack_valid_stats": valid_summary,
        "paired_blind_comp_fpr_t4_vs_t2": fpr_paired,
    })


def main():
    parser = argparse.ArgumentParser(description="冻结版 BUD-Fusion Dev-320 全维度验收（含成对 Bootstrap 95% CI 与 2 种新攻击）")
    parser.add_argument("--data_dir", type=str, default=TRAIN_FULL19_DIR)
    parser.add_argument("--eval_split_json", type=str, default=DEFAULT_EVAL_SPLIT)
    parser.add_argument("--base_calib_json", type=str, default=DEFAULT_BASE_CALIB)
    parser.add_argument("--old_gating_json", type=str, default=DEFAULT_OLD_GATING)
    parser.add_argument("--frozen_gating_json", type=str, default=DEFAULT_FROZEN_GATING)
    parser.add_argument("--seqs", nargs="+", default=None, help="可选：仅运行指定父序列子集（多进程并行分片加速用）")
    parser.add_argument("--worker_only", action="store_true", help="仅写序列缓存 JSON，不聚合最终总报告")
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out_json", type=str, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    t0 = time.time()
    with open(args.eval_split_json, "r", encoding="utf-8") as f:
        eval_split = json.load(f)
    seq_order = list(eval_split["seq_order"])
    target_seqs = [s for s in seq_order if (args.seqs is None or s in set(args.seqs))]
    os.makedirs(args.cache_dir, exist_ok=True)

    # 若全部序列缓存齐全且非 worker_only，免加载 GPU 模型极速聚合
    all_cached = (not args.worker_only) and all(
        os.path.exists(os.path.join(args.cache_dir, f"{s}.json")) for s in seq_order
    )
    wrapper = ds_m1 = ds_m2 = gx = gy = auditor = old_auditor = None
    if not all_cached:
        wrapper = HEALBaseWrapper(device="cuda")
        ds_m1 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m1")
        ds_m2 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m2")
        gx, gy = build_bev_grid_xy_256(device="cuda", dtype=torch.float32)
        auditor = BUDFusionHEALAuditor(args.base_calib_json, gating_calib_path=args.frozen_gating_json)
        old_auditor = (
            BUDFusionHEALAuditor(args.base_calib_json, gating_calib_path=args.old_gating_json)
            if os.path.exists(args.old_gating_json) else None
        )

    seq_caches = {}
    for seq_name in (target_seqs if args.worker_only else seq_order):
        cpath = os.path.join(args.cache_dir, f"{seq_name}.json")
        if os.path.exists(cpath):
            print(f"[断点复用] {seq_name}")
            with open(cpath, "r", encoding="utf-8") as f:
                seq_caches[seq_name] = json.load(f)
        else:
            idxs = eval_split["sequences"][seq_name]["indices"]
            print(f"[开始评测] {seq_name} ({len(idxs)} 帧)", flush=True)
            rec = _eval_one_sequence(wrapper, ds_m1, ds_m2, auditor, old_auditor, seq_name, idxs, gx, gy)
            _atomic_write_json(cpath, rec)
            seq_caches[seq_name] = rec

    if args.worker_only:
        print(f"\n[Worker完成] seqs={target_seqs} (耗时 {time.time() - t0:.1f}s)")
        return

    agg = _aggregate_with_paired_bootstrap(seq_order, seq_caches)
    with open(args.frozen_gating_json, "r", encoding="utf-8") as f:
        fg_cfg = json.load(f)

    # 以 Dev-320 实测干净盲区召回损失判定最终是否达标 (<= 3.0 个百分点)
    br_loss_m1 = agg["blind_spot_recall"]["m1"]["clean"]["no_defense"]["val"] - agg["blind_spot_recall"]["m1"]["clean"]["bud_frozen"]["val"]
    br_loss_m2 = agg["blind_spot_recall"]["m2"]["clean"]["no_defense"]["val"] - agg["blind_spot_recall"]["m2"]["clean"]["bud_frozen"]["val"]
    dev320_br_met = bool(br_loss_m1 <= 0.030 and br_loss_m2 <= 0.030)

    report = OrderedDict({
        "meta": {
            "description": "步骤 ④-5 冻结版 BUD-Fusion Dev-320 全维度验收（9 工况 × 5 方法，成对差值 2000 次 Block-Bootstrap 95% CI）",
            "seq_order": seq_order,
            "total_frames": int(eval_split["total_frames"]),
            "bootstrap_iters": BOOT_N,
            "bootstrap_seed": BOOT_SEED,
            "frozen_gating_json": args.frozen_gating_json,
            "fit_target_met": fg_cfg.get("meta", {}).get("fit_target_met"),
            "fit_status": fg_cfg.get("meta", {}).get("fit_status"),
            "dev320_clean_blind_recall_loss": {"m1": round(br_loss_m1, 5), "m2": round(br_loss_m2, 5)},
            "dev320_blind_recall_target_met": dev320_br_met,
            "dev320_status": "达标（Dev-320 干净盲区召回损失 <=3%）" if dev320_br_met else "未达标（Dev-320 干净盲区召回损失 >3%）",
            "visibility_v2": fg_cfg.get("visibility_v2"),
            "frame_high_tau": fg_cfg.get("frame_high_tau"),
            "gating_params": fg_cfg.get("gating_params"),
            "elapsed_sec": round(time.time() - t0, 2),
        },
        **agg,
    })
    _atomic_write_json(args.out_json, report)
    print(f"\n[完成] 冻结版 Dev-320 报告已保存至 {args.out_json} (总耗时 {time.time() - t0:.1f}s)")

    # 打印核心验收摘要
    print("\n=== 核心验收摘要（AP@0.5 / 盲区召回 & 成对差值 95% CI） ===")
    for mod in MODALITIES_2:
        cln_nd = report["ap_results"][mod]["clean"]["no_defense"]["ap50"]["val"]
        cln_bf = report["ap_results"][mod]["clean"]["bud_frozen"]["ap50"]["val"]
        br_nd = report["blind_spot_recall"][mod]["clean"]["no_defense"]["val"]
        br_bf = report["blind_spot_recall"][mod]["clean"]["bud_frozen"]["val"]
        br_diff = report["paired_blind_recall_diffs"][mod]["clean"]["bud_frozen_vs_no_defense"]
        print(
            f"  [{mod}] Clean AP50: {cln_nd*100:.2f}% -> {cln_bf*100:.2f}% | "
            f"BlindRec: {br_nd*100:.2f}% -> {br_bf*100:.2f}% "
            f"(Delta={br_diff['diff']*100:+.2f}%, CI=[{br_diff['diff_ci95'][0]*100:+.2f}%, {br_diff['diff_ci95'][1]*100:+.2f}%])"
        )
        for c in ("pgd_1.0sigma", "local_removal_1.0sigma", "local_ghost_2.0sigma", "local_occluded_ghost_2.0sigma", "local_near0_6m_2.0sigma"):
            d_w = report["paired_ap_diffs"][mod][c]["bud_frozen_vs_whole_cav_drop"]["ap50"]
            d_n = report["paired_ap_diffs"][mod][c]["bud_frozen_vs_no_defense"]["ap50"]
            print(
                f"    {c:30s}: BUD={d_w['val_a']*100:.2f}% | vs WholeDrop={d_w['diff']*100:+.2f}% "
                f"[{d_w['diff_ci95'][0]*100:+.2f}%, {d_w['diff_ci95'][1]*100:+.2f}%] | "
                f"vs NoDef={d_n['diff']*100:+.2f}%"
            )


if __name__ == "__main__":
    main()
