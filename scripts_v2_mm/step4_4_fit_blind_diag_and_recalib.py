# -*- coding: utf-8 -*-
"""
步骤 ④-4：纯 Fit 集诊断与冻结重标定（新物理可见性定义 + 前景/空地条件双分档）。

严格落实 5 项核心规范：
1. 【原因 B 根治 · 改可见性定义】：前方无遮挡且本车物理预期激光点数 expected_pts >= clear_pts_tau
   （阈值在 Fit 客观盲区与清晰目标分布上确定）才算 Clear，无遮挡但点数不足的归为 Sparse；
2. 【原因 A 根治 · 前景/空地分开标定】：不靠硬乘 2.2x 放宽遮挡区门限或把遮挡区权重置 0；
   在分档中加入「协作车该位置是否有前景响应（fg vs bg）」，前景格与空地格分开统计良性分位数；
3. 【遮挡伪造前后对比】：显式报告「只在遮挡区伪造」修复前（OldBUD）与修复后（NewBUD）的
   检出率（AUROC / TPR@5%FPR）、攻击区门控值与融合后 AP@0.5；
4. 【Fit 不阻断流水线】：Fit 仅 70 辆盲区车波动大，若 Fit 未严格满足 <=3% 绝不报错中断，
   在 JSON 中显式标明 fit_target_met=False 与 status="未达标（以Dev-320为准）"，以最优参数继续推进；
5. 单次多边形匹配 + 无防御/Ego 回退缓存复用 + 逐组进度打印，全程约 15 分钟跑完。
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
    build_directional_gate_map,
    build_fg_mask,
    build_gated_fuse_fn,
    compute_shift_tolerant_semantic_maps,
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
    GRID_RES,
    MIN_XY,
    OPV2V_TRAIN_DIR,
    PROTO_H,
    PROTO_W,
    VIS_LABELS_3,
    _masked_avg_pool3x3,
    assign_dist5_vis3_indices_v2,
    build_bev_grid_xy_256,
    compute_auroc,
    compute_ego_expected_pts_and_los_256,
    extract_frame_context_and_clean,
    find_exact_fpr_threshold,
    load_ego_lidar_np,
)

DEFAULT_SPLIT = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_1_v2_split_plan.json")
DEFAULT_BASE_CALIB = os.path.join(PROJ_ROOT, "results_v3", "a1_fit_base_calibrator.json")
DEFAULT_OLD_GATING = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_2_bud_gating_calibrator.json")
DEFAULT_OUT = os.path.join(PROJ_ROOT, "results_v3", "step4_4_bud_frozen_calibrator.json")

FIT_DEV_STATES = [
    "clean",
    "local_removal_1.0sigma",
    "local_ghost_2.0sigma",
    "local_occluded_ghost_2.0sigma",
    "local_near0_6m_2.0sigma",
    "scaled_pgd_1.0sigma",
]


def _empty_ap_stat():
    return {0.5: {"fp": [], "tp": [], "score": [], "gt": 0}, 0.7: {"fp": [], "tp": [], "score": [], "gt": 0}}


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


def _match_hits_at_iou(pred_boxes, pred_score, gt_np: np.ndarray, gt_polys: list, iou_thresh: float = 0.5) -> list:
    from opencood.utils import common_utils

    gt_cnt = int(gt_np.shape[0])
    matched = [False] * gt_cnt
    if pred_boxes is None or len(pred_boxes) == 0 or gt_cnt == 0:
        return matched
    det_np = common_utils.torch_tensor_to_numpy(pred_boxes)
    sc_np = common_utils.torch_tensor_to_numpy(pred_score).reshape(-1)
    if det_np.shape[0] == 0:
        return matched
    order = np.argsort(-sc_np)
    det_polys = list(common_utils.convert_format(det_np))
    rem_polys = list(gt_polys)
    rem_idxs = list(range(gt_cnt))
    for k in range(order.shape[0]):
        if not rem_polys:
            break
        ious = common_utils.compute_iou(det_polys[int(order[k])], rem_polys)
        best_j = int(np.argmax(ious))
        if float(ious[best_j]) >= float(iou_thresh):
            rem_polys.pop(best_j)
            matched[rem_idxs.pop(best_j)] = True
    return matched


def _find_blind_gts_for_frame(ds_m1, idx: int, ego_pb, ego_ps, ego_gb, cov_2d: torch.Tensor) -> dict:
    """判定单帧客观盲区 GT（落入 cov_ego ∧ 本车 64 线三维框内点数 <= 5 ∧ Ego-Only 漏检）与清晰检出 GT。"""
    from matplotlib.path import Path as MplPath
    from opencood.utils import common_utils

    gt_np = common_utils.torch_tensor_to_numpy(ego_gb)
    gt_polys = list(common_utils.convert_format(gt_np))
    ego_matched = _match_hits_at_iou(ego_pb, ego_ps, gt_np, gt_polys, 0.5)
    ego_pts = load_ego_lidar_np(ds_m1, idx)
    cov_np = cov_2d.detach().cpu().numpy()

    blind_indices, blind_yx, ego_hit_yx = [], [], []
    excluded_ego_self = 0
    for i in range(gt_np.shape[0]):
        cx = float(gt_np[i, :4, 0].mean())
        cy = float(gt_np[i, :4, 1].mean())
        # 排除本车 Ego 自身车体 GT 框（OPV2V 中位于原点附近 r≈0.5m，且本车点云预处理已清空自身点云）
        if math.hypot(cx, cy) < 2.5:
            excluded_ego_self += 1
            continue
        ix = int(math.floor((cx - MIN_XY) / GRID_RES))
        iy = int(math.floor((cy - MIN_XY) / GRID_RES))
        if not (0 <= ix < PROTO_W and 0 <= iy < PROTO_H and bool(cov_np[iy, ix])):
            continue
        if ego_matched[i]:
            ego_hit_yx.append((iy, ix))
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
            blind_yx.append((iy, ix))

    return {
        "gt_np": gt_np,
        "gt_polys": gt_polys,
        "blind_indices": blind_indices,
        "blind_yx": blind_yx,
        "ego_hit_yx": ego_hit_yx,
        "excluded_ego_self": excluded_ego_self,
    }


@torch.no_grad()
def _compute_dev_signed_and_pfused(wrapper, p_dict, feat_in, p_ego_cls, cov_ego, k_tol: int = 5):
    """单次前向返回 (dev_4ch, d_drop_tol, d_add_tol, p_fused_cls)。"""
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
    return dev_4ch, d_drop_tol, d_add_tol, p_fused_cls


def _build_collab_fg_mask(p_fused_cls: torch.Tensor, fg_prob_tau: float = 0.20) -> torch.Tensor:
    """用 3x3 max-pool 扩展车辆前景响应掩膜（覆及车体周边网格，防止车身边缘滑入空地桶）。"""
    pf_dil = F.max_pool2d(p_fused_cls, kernel_size=3, stride=1, padding=1)[0, 0]
    return pf_dil > float(fg_prob_tau)


def _eval_fit_dev_candidate(
    wrapper,
    ds_ref,
    eval_items: list,
    blind_meta_by_idx: dict,
    g_params: dict = None,
    no_def_cache: dict = None,
    ego_only_cache: dict = None,
    frame_high_tau: dict = None,
    use_v_idx_key: str = "v_idx_v2",
):
    """评测候选门控参数，返回 AP、盲区召回、门控统计及「遮挡区伪造」检出率统计（AUROC / TPR@5%FPR）。"""
    from opencood.tools import train_utils
    from opencood.utils import common_utils, eval_utils

    ap_stats = {m: {st: _empty_ap_stat() for st in FIT_DEV_STATES} for m in ("m1", "m2")}
    blind_hits = {m: {st: 0 for st in FIT_DEV_STATES} for m in ("m1", "m2")}
    blind_tot = {m: {st: 0 for st in FIT_DEV_STATES} for m in ("m1", "m2")}
    # 用于计算「只在遮挡区伪造」的检出率：每帧门控异常峰值 score_det = max(1 - gate_L0)
    det_scores_clean = {"m1": [], "m2": []}
    det_scores_occ = {"m1": [], "m2": []}

    gate_stats = {
        "clean_valid": [],
        "clean_blind_gt": [],
        "lrem_atk": [], "lrem_cln": [],
        "lgho_atk": [], "lgho_cln": [],
        "locc_atk": [], "locc_cln": [],
        "lnear_atk": [], "lnear_cln": [],
    }

    for item in eval_items:
        mod = item["mod"]
        st_name = item["state"]
        sidx = item["sample_idx"]
        b_post = train_utils.to_device(item["batch_slim"], wrapper.device_obj)
        cov_2d = item["cov_2d"].to(wrapper.device_obj)
        binfo = blind_meta_by_idx.get((sidx, mod), blind_meta_by_idx.get(sidx))

        fuse_fn = None
        used_shortcut = False
        if g_params is not None:
            triggered_fb = False
            if frame_high_tau is not None and mod in frame_high_tau:
                if float(item["score_t4"]) > float(frame_high_tau[mod]):
                    triggered_fb = True

            if triggered_fb:
                gate_l0 = torch.zeros((1, 1, PROTO_H, PROTO_W), device=wrapper.device_obj, dtype=torch.float32)
            else:
                cos_comb = item["cos_comb_t4"].to(wrapper.device_obj)
                d_drop_tol = item["d_drop_tol"].to(wrapper.device_obj)
                d_add_tol = item["d_add_tol"].to(wrapper.device_obj)
                v_idx = item[use_v_idx_key].to(wrapper.device_obj)
                fg_key = "fg_mask_old" if use_v_idx_key == "v_idx_old" else "fg_mask"
                fg_mask = item[fg_key].to(wrapper.device_obj) if "cos_tau_by_vis_fg" in g_params else None
                if "cos_tau_by_vis_bg" in g_params:
                    cos_tau_bg = g_params["cos_tau_by_vis_bg"][mod]
                    cos_tau_fg = g_params["cos_tau_by_vis_fg"][mod]
                else:
                    cos_tau_bg = float(g_params["cos_tau"][mod])
                    cos_tau_fg = None

                gate_l0 = build_directional_gate_map(
                    cos_comb=cos_comb,
                    d_drop_tol=d_drop_tol,
                    d_add_tol=d_add_tol,
                    cov_2d=cov_2d,
                    v_idx=v_idx,
                    cos_tau=cos_tau_bg,
                    drop_tau=float(g_params["drop_tau"]),
                    add_tau=float(g_params["add_tau"]),
                    w_drop=float(g_params["w_drop"]),
                    w_add=float(g_params["w_add"]),
                    cos_co_ratio=float(g_params["cos_co_ratio"]),
                    gamma_add=g_params["gamma_add"],
                    gamma_cos_gate=g_params.get("gamma_cos_gate", (1.0, 1.0, 1.0)),
                    r_dil=int(g_params["r_dil"]),
                    beta=float(g_params["beta"]),
                    collab_fg_mask=fg_mask,
                    cos_tau_fg=cos_tau_fg,
                )

            g_valid = gate_l0[0, 0][cov_2d]
            # 帧内局部异常触发强度（用于评估遮挡区伪造检出率）：1 - min(gate) 与 top-25 平均 (1 - gate)
            inv_g = 1.0 - g_valid
            k_d = min(25, int(inv_g.numel()))
            det_s = float(torch.topk(inv_g, k_d).values.mean().item()) if k_d > 0 else 0.0
            if st_name == "clean":
                det_scores_clean[mod].append(det_s)
                gate_stats["clean_valid"].append(float(g_valid.mean().item()))
                for iy, ix in binfo["blind_yx"]:
                    gate_stats["clean_blind_gt"].append(float(gate_l0[0, 0, iy, ix].item()))
            elif st_name == "local_occluded_ghost_2.0sigma" and item.get("atk_valid", True):
                det_scores_occ[mod].append(det_s)

            if item["atk_mask_2d"] is not None:
                atk_m = item["atk_mask_2d"].to(wrapper.device_obj) & cov_2d
                cln_m = cov_2d & (~atk_m)
                tag = {
                    "local_removal_1.0sigma": "lrem",
                    "local_ghost_2.0sigma": "lgho",
                    "local_occluded_ghost_2.0sigma": "locc",
                    "local_near0_6m_2.0sigma": "lnear",
                }.get(st_name)
                if tag is not None:
                    if atk_m.any():
                        gate_stats[f"{tag}_atk"].append(float(gate_l0[0, 0][atk_m].mean().item()))
                    if cln_m.any():
                        gate_stats[f"{tag}_cln"].append(float(gate_l0[0, 0][cln_m].mean().item()))

            if triggered_fb and ego_only_cache is not None and sidx in ego_only_cache:
                pb, ps, gb = ego_only_cache[sidx]
                used_shortcut = True
            elif no_def_cache is not None and float(g_valid.min().item()) >= 1.0 - 1e-6:
                pb, ps, gb = no_def_cache[(mod, st_name, sidx)]
                used_shortcut = True
            else:
                fuse_fn = build_gated_fuse_fn(gate_l0)

        if not used_shortcut:
            with torch.no_grad():
                feat = item["feat_dev"].to(wrapper.device_obj)
                rlen = item["rlen"].to(wrapper.device_obj)
                amat = item["amat"].to(wrapper.device_obj)
                out = wrapper.forward_from_protocol(feat, rlen, amat, item["mods"], custom_fuse_fn=fuse_fn)
                pb, ps, gb = wrapper.decode_boxes(b_post, out, ds_ref)
            if no_def_cache is not None and g_params is None:
                no_def_cache[(mod, st_name, sidx)] = (
                    pb.detach().cpu() if pb is not None else None,
                    ps.detach().cpu() if ps is not None else None,
                    gb.detach().cpu(),
                )

        # 单次构造旋转多边形，同时算出 IoU@0.5 / IoU@0.7 与盲区命中
        gt_np, gt_polys = binfo["gt_np"], binfo["gt_polys"]
        gt_cnt = int(gt_np.shape[0])
        ap_stats[mod][st_name][0.5]["gt"] += gt_cnt
        ap_stats[mod][st_name][0.7]["gt"] += gt_cnt
        blind_tot[mod][st_name] += len(binfo["blind_indices"])

        if pb is not None and len(pb) > 0:
            det_np = common_utils.torch_tensor_to_numpy(pb)
            sc_np = common_utils.torch_tensor_to_numpy(ps).reshape(-1)
            order = np.argsort(-sc_np)
            sc_sorted = sc_np[order].tolist()
            det_polys = list(common_utils.convert_format(det_np))

            for iou_th in (0.5, 0.7):
                rem_p = list(gt_polys)
                rem_i = list(range(gt_cnt))
                matched = [False] * gt_cnt
                fp_l, tp_l = [], []
                for k in range(order.shape[0]):
                    if not rem_p:
                        fp_l.append(1)
                        tp_l.append(0)
                        continue
                    ious = common_utils.compute_iou(det_polys[int(order[k])], rem_p)
                    best_j = int(np.argmax(ious))
                    if float(ious[best_j]) >= iou_th:
                        fp_l.append(0)
                        tp_l.append(1)
                        rem_p.pop(best_j)
                        matched[rem_i.pop(best_j)] = True
                    else:
                        fp_l.append(1)
                        tp_l.append(0)
                ap_stats[mod][st_name][iou_th]["tp"].extend(tp_l)
                ap_stats[mod][st_name][iou_th]["fp"].extend(fp_l)
                ap_stats[mod][st_name][iou_th]["score"].extend(sc_sorted)
                if iou_th == 0.5 and binfo["blind_indices"]:
                    blind_hits[mod][st_name] += sum(1 for bi in binfo["blind_indices"] if matched[bi])

    ap50 = {m: {st: float(eval_utils.calculate_ap(ap_stats[m][st], 0.5)[0]) for st in FIT_DEV_STATES} for m in ("m1", "m2")}
    ap70 = {m: {st: float(eval_utils.calculate_ap(ap_stats[m][st], 0.7)[0]) for st in FIT_DEV_STATES} for m in ("m1", "m2")}
    brec = {
        m: {
            st: (float(blind_hits[m][st]) / float(blind_tot[m][st]) if blind_tot[m][st] > 0 else 0.0)
            for st in FIT_DEV_STATES
        }
        for m in ("m1", "m2")
    }
    occ_det = {}
    for m in ("m1", "m2"):
        neg = np.asarray(det_scores_clean[m], dtype=np.float64)
        pos = np.asarray(det_scores_occ[m], dtype=np.float64)
        if neg.size > 0 and pos.size > 0:
            th5 = find_exact_fpr_threshold(neg, 0.05)
            occ_det[m] = {
                "auroc": round(compute_auroc(neg, pos), 4),
                "tpr_at_5pct_fpr": round(float((pos > th5).mean()), 4),
                "gate_trigger_rate_gt_0.15": round(float((pos > 0.15).mean()), 4),
                "n_valid_occ": int(pos.size),
            }
        else:
            occ_det[m] = {"auroc": 0.5, "tpr_at_5pct_fpr": 0.0, "gate_trigger_rate_gt_0.15": 0.0, "n_valid_occ": 0}
    return ap50, ap70, brec, gate_stats, occ_det


def main():
    parser = argparse.ArgumentParser(description="纯 Fit 标定（新可见性定义 + 前景/空地双分档 + 遮挡伪造前后对比）")
    parser.add_argument("--data_dir", type=str, default=OPV2V_TRAIN_DIR)
    parser.add_argument("--split_json", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--base_calib_json", type=str, default=DEFAULT_BASE_CALIB)
    parser.add_argument("--old_gating_json", type=str, default=DEFAULT_OLD_GATING)
    parser.add_argument("--fg_prob_tau", type=float, default=0.20)
    parser.add_argument("--fg_mode", type=str, default="v1", choices=["v1", "v0"])
    parser.add_argument("--out_json", type=str, default=DEFAULT_OUT)
    args = parser.parse_args()

    t0 = time.time()
    with open(args.split_json, "r", encoding="utf-8") as f:
        split_plan = json.load(f)
    with open(args.old_gating_json, "r", encoding="utf-8") as f:
        old_gating = json.load(f)

    fit_indices = list(split_plan["fit_indices"])
    fit_dev_set = set(split_plan["fit_dev_indices"])
    eval_set = set(split_plan.get("eval_indices", []))
    assert set(fit_indices).isdisjoint(eval_set), "Fit 与 Eval 存在交集，拒绝运行！"
    print(f"[Step 4-4 · 新可见性 + 前景/空地双分档标定] Fit={len(fit_indices)} 帧 (Dev={len(fit_dev_set)})", flush=True)

    wrapper = HEALBaseWrapper(device="cuda")
    from opencood.tools import train_utils
    ds_m1 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m1")
    ds_m2 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m2")
    gx, gy = build_bev_grid_xy_256(device="cuda", dtype=torch.float32)

    auditor = BUDFusionHEALAuditor(args.base_calib_json, gating_calib_path=args.old_gating_json)

    # 第一遍：收集 Fit-100 的 los_factor、expected_pts、偏差图、前景掩膜，以及 Fit-Dev 的客观盲区 GT 分布
    fit100_cache = []
    blind_meta_by_idx = {}
    ego_only_cache = {}
    unocc_blind_exp_pts = []
    unocc_hit_exp_pts = []
    fit_dev_items = []
    k_tol = 5

    for rank, idx in enumerate(fit_indices):
        b = extract_frame_context_and_clean(wrapper, ds_m1, ds_m2, idx, gx, gy, clear_pts_tau=75.0)
        d_idx = b["d_idx"]
        v_idx_old = b["v_idx"]
        ego_pts = load_ego_lidar_np(ds_m1, idx)
        _, los_f, exp_pts = compute_ego_expected_pts_and_los_256(ego_pts, gx, gy, num_beams=64)
        los_2d = los_f[0, 0]
        exp_2d = exp_pts[0, 0]
        is_dev = idx in fit_dev_set

        batch_slim = None
        if is_dev:
            raw_b1 = train_utils.to_device(ds_m1.collate_batch_test([ds_m1[idx]]), "cpu")
            batch_slim = _slim_batch_for_postprocess(raw_b1)
            with torch.no_grad():
                ego_out = wrapper.forward_single_agent(b["p1"]["heter_feature_2d"][0:1])
                b_post_gpu = train_utils.to_device(batch_slim, wrapper.device_obj)
                ego_pb, ego_ps, ego_gb = wrapper.decode_boxes(b_post_gpu, ego_out, ds_m1)
            ego_only_cache[idx] = (
                ego_pb.detach().cpu() if ego_pb is not None else None,
                ego_ps.detach().cpu() if ego_ps is not None else None,
                ego_gb.detach().cpu(),
            )
            binfo_m1 = _find_blind_gts_for_frame(ds_m1, idx, ego_pb, ego_ps, ego_gb, b["cov_ego_m1_full"][0, 0])
            binfo_m2 = _find_blind_gts_for_frame(ds_m1, idx, ego_pb, ego_ps, ego_gb, b["cov_ego"][0, 0])
            blind_meta_by_idx[(idx, "m1")] = binfo_m1
            blind_meta_by_idx[(idx, "m2")] = binfo_m2
            blind_meta_by_idx[idx] = binfo_m2
            # 统计无遮挡盲区车 vs 无遮挡本车清晰命中车的物理预期激光点数，用于 Fit 上客观确定 clear_pts_tau
            for iy, ix in binfo_m1["blind_yx"]:
                if float(los_2d[iy, ix].item()) >= 0.50:
                    unocc_blind_exp_pts.append(float(exp_2d[iy, ix].item()))
            for iy, ix in binfo_m1["ego_hit_yx"]:
                if float(los_2d[iy, ix].item()) >= 0.50:
                    unocc_hit_exp_pts.append(float(exp_2d[iy, ix].item()))

        frame_entry = {
            "idx": idx,
            "p_ego_cls": b["p_ego_cls"].half().cpu(),
            "d_idx": d_idx.detach().cpu(),
            "los_2d": los_2d.detach().cpu(),
            "exp_2d": exp_2d.detach().cpu(),
            "dist_cav": b["dist_cav"].detach().cpu(),
            "mods": {},
        }

        for mod, p_dict in (("m1", b["p1"]), ("m2", b["p2"])):
            # 主实验口径：m1 在自身完整激光雷达覆盖范围标定与受测，m2 在摄像头有效覆盖区标定与受测
            cov_ego_mod = b["cov_ego_m1_full"] if mod == "m1" else b["cov_ego"]
            cov_mod_2d = cov_ego_mod[0, 0]
            dev_c, dd_c, da_c, pf_c = _compute_dev_signed_and_pfused(
                wrapper, p_dict, p_dict["heter_feature_2d"], b["p_ego_cls"], cov_ego_mod, k_tol=k_tol
            )
            fg_m_old_c = _build_collab_fg_mask(pf_c, fg_prob_tau=args.fg_prob_tau)
            frame_entry["mods"][mod] = {
                "cov_2d": cov_mod_2d.detach().cpu(),
                "dev_4ch": dev_c.half().cpu(),
                "pf_cls": pf_c.half().cpu(),
                "fg_mask_old": fg_m_old_c.detach().cpu(),
            }

            if is_dev:
                cov_cpu = cov_mod_2d.detach().cpu()
                v_old_cpu = v_idx_old.detach().cpu()
                rlen_cpu = p_dict["record_len"].detach().cpu()
                amat_cpu = p_dict["affine_matrix"].detach().cpu()
                mods_l = list(p_dict["agent_modality_list"])
                occ_mask_4d = (los_f < 0.50).float()

                fit_dev_items.append({
                    "sample_idx": idx, "mod": mod, "state": "clean",
                    "feat_dev": p_dict["heter_feature_2d"].detach().cpu(),
                    "rlen": rlen_cpu, "amat": amat_cpu, "mods": mods_l,
                    "batch_slim": batch_slim, "cov_2d": cov_cpu,
                    "p_ego_cls": b["p_ego_cls"].half().cpu(),
                    "pf_cls": pf_c.half().cpu(),
                    "d_idx": d_idx.detach().cpu(), "v_idx_old": v_old_cpu,
                    "los_2d": los_2d.detach().cpu(), "exp_2d": exp_2d.detach().cpu(),
                    "dist_cav": b["dist_cav"].detach().cpu(),
                    "dev_4ch": dev_c.half().cpu(),
                    "d_drop_tol": dd_c.detach().cpu(),
                    "d_add_tol": da_c.detach().cpu(),
                    "fg_mask_old": fg_m_old_c.detach().cpu(),
                    "atk_mask_2d": None,
                    "atk_valid": True,
                })

                feat = p_dict["heter_feature_2d"]
                rlen = p_dict["record_len"]
                amat = p_dict["affine_matrix"]
                vmask = p_dict["cam_valid_mask"]
                preg = None

                f_lr, st_lr = generate_local_removal_heal(
                    wrapper, feat, rlen, amat, mods_l, vmask,
                    attacker_idx=1, num_targets=2, patch_radius=5, sigma_mult=1.0, steps=10, perturb_region_mask=preg,
                )
                f_lg, st_lg = generate_local_ghost_spoofing_heal(
                    wrapper, feat, rlen, amat, mods_l, vmask,
                    attacker_idx=1, num_spoofs=2, patch_radius=5, sigma_mult=2.0, steps=10, frame_seed=idx, perturb_region_mask=preg,
                )
                f_lo, st_lo = generate_local_occluded_ghost_heal(
                    wrapper, feat, rlen, amat, mods_l, vmask, occluded_mask_ego=occ_mask_4d,
                    attacker_idx=1, num_spoofs=2, patch_radius=5, sigma_mult=2.0, steps=12, frame_seed=idx, perturb_region_mask=preg,
                )
                f_ln, st_ln = generate_local_near0_6m_attack_heal(
                    wrapper, feat, rlen, amat, mods_l, vmask,
                    attacker_idx=1, sigma_mult=2.0, steps=12, frame_seed=idx, perturb_region_mask=preg, enable_ghost_fallback=False,
                )
                f_p1, _ = generate_scaled_pgd_removal_heal(
                    wrapper, feat, rlen, amat, mods_l, vmask,
                    attacker_idx=1, sigma_mult=1.0, steps=8, perturb_region_mask=preg,
                )

                for st_name, f_adv, st_meta in (
                    ("local_removal_1.0sigma", f_lr, st_lr),
                    ("local_ghost_2.0sigma", f_lg, st_lg),
                    ("local_occluded_ghost_2.0sigma", f_lo, st_lo),
                    ("local_near0_6m_2.0sigma", f_ln, st_ln),
                    ("scaled_pgd_1.0sigma", f_p1, None),
                ):
                    dev_a, dd_a, da_a, pf_a = _compute_dev_signed_and_pfused(
                        wrapper, p_dict, f_adv, b["p_ego_cls"], cov_ego_mod, k_tol=k_tol
                    )
                    fg_m_old_a = _build_collab_fg_mask(pf_a, fg_prob_tau=args.fg_prob_tau)
                    atk_m = st_meta["attacked_mask_in_ego"][0, 0].detach().cpu() > 0.5 if st_meta and "attacked_mask_in_ego" in st_meta else None
                    is_v = bool(st_meta is None or (st_meta.get("valid", True) and st_meta.get("num_targets", 1) > 0 and st_meta.get("num_spoof_anchors", 1) > 0))
                    fit_dev_items.append({
                        "sample_idx": idx, "mod": mod, "state": st_name,
                        "feat_dev": f_adv.detach().cpu(),
                        "rlen": rlen_cpu, "amat": amat_cpu, "mods": mods_l,
                        "batch_slim": batch_slim, "cov_2d": cov_cpu,
                        "p_ego_cls": b["p_ego_cls"].half().cpu(),
                        "pf_cls": pf_a.half().cpu(),
                        "d_idx": d_idx.detach().cpu(), "v_idx_old": v_old_cpu,
                        "los_2d": los_2d.detach().cpu(), "exp_2d": exp_2d.detach().cpu(),
                        "dist_cav": b["dist_cav"].detach().cpu(),
                        "dev_4ch": dev_a.half().cpu(),
                        "d_drop_tol": dd_a.detach().cpu(),
                        "d_add_tol": da_a.detach().cpu(),
                        "fg_mask_old": fg_m_old_a.detach().cpu(),
                        "atk_mask_2d": atk_m,
                        "atk_valid": is_v,
                    })

        fit100_cache.append(frame_entry)
        if (rank + 1) % 20 == 0 or (rank + 1) == len(fit_indices):
            print(f"  已处理 Fit 样本 {rank + 1}/{len(fit_indices)} ({time.time() - t0:.1f}s)", flush=True)

    # === 1. 在 Fit 上标定新可见性定义阈值 clear_pts_tau（原因 B 根治，已排除本车自身框） ===
    # 将无遮挡但本车漏检的远距稀疏盲区车准确归入 Sparse，同时保持近中距高置信车辆在 Clear
    if unocc_blind_exp_pts:
        blind_p85 = float(np.percentile(unocc_blind_exp_pts, 85.0))
        blind_p90 = float(np.percentile(unocc_blind_exp_pts, 90.0))
        raw_tau = max(blind_p90 * 1.35, 60.0)
    else:
        blind_p85, blind_p90 = 60.0, 75.0
        raw_tau = 75.0
    hit_p25 = float(np.percentile(unocc_hit_exp_pts, 25.0)) if unocc_hit_exp_pts else 75.0
    # 综合无遮挡稀疏盲区上界与清晰命中车下分位，限幅在物理合理区间 [60, 120] 点（对应约 31~45m）
    clear_pts_tau = round(float(np.clip(max(raw_tau, hit_p25), 60.0, 120.0)), 3)
    vis_v2_cfg = {
        "clear_pts_tau": clear_pts_tau,
        "los_occ_tau": 0.50,
        "fit_unocc_blind_cnt": len(unocc_blind_exp_pts),
        "fit_unocc_blind_exp_pts_p50": round(float(np.median(unocc_blind_exp_pts)), 2) if unocc_blind_exp_pts else 0.0,
        "fit_unocc_blind_exp_pts_p85": round(blind_p85, 2),
        "fit_unocc_blind_exp_pts_p90": round(blind_p90, 2),
        "fit_unocc_hit_exp_pts_p25": round(hit_p25, 2),
        "excluded_ego_self_gt_cnt": int(sum(v.get("excluded_ego_self", 0) for v in blind_meta_by_idx.values())),
    }
    print(f"\n=== [原因 B · 新可见性定义 Fit 标定] {json.dumps(vis_v2_cfg, ensure_ascii=False)} ===", flush=True)

    # === 2. 基于新可见性定义重算 Fit-100 帧级 div_p95 与 (vis × bg/fg) 条件余弦分位数（原因 A 根治） ===
    raw_scores = {"t2": {"m1": [], "m2": []}, "t4": {"m1": [], "m2": []}}
    pix_cos_bg = {m: {v: [] for v in range(3)} for m in ("m1", "m2")}
    pix_cos_fg = {m: {v: [] for v in range(3)} for m in ("m1", "m2")}

    for fe in fit100_cache:
        _, v_new_4d = assign_dist5_vis3_indices_v2(
            fe["dist_cav"].cuda(), fe["los_2d"].view(1, 1, PROTO_H, PROTO_W).cuda(),
            fe["exp_2d"].view(1, 1, PROTO_H, PROTO_W).cuda(), clear_pts_tau=clear_pts_tau, los_occ_tau=0.50
        )
        v_new = v_new_4d[0, 0]
        d_idx = fe["d_idx"].cuda()
        p_ego_g = fe["p_ego_cls"].cuda().float()
        for mod in ("m1", "m2"):
            cov_2d = fe["mods"][mod]["cov_2d"].cuda()
            dev_4 = fe["mods"][mod]["dev_4ch"].cuda().float()
            pf_g = fe["mods"][mod]["pf_cls"].cuda().float()
            # B2 同步：标定阶段同样使用以本车为准的新前景判定规则
            fg_m = build_fg_mask(p_ego_g, pf_g, v_new, fg_prob_tau=args.fg_prob_tau, mode=args.fg_mode)
            ar = auditor.audit_dev_maps(dev_4, cov_2d, d_idx, v_new, mod, build_gate=False)
            raw_scores["t2"][mod].append(ar["raw_t2"])
            raw_scores["t4"][mod].append(ar["raw_t4"])
            cm_s = F.avg_pool2d(ar["cos_comb_t4"].view(1, 1, PROTO_H, PROTO_W), kernel_size=3, stride=1, padding=1)[0, 0]
            for v_id in range(3):
                m_bg = cov_2d & (v_new == v_id) & (~fg_m)
                m_fg = cov_2d & (v_new == v_id) & fg_m
                if m_bg.any():
                    pix_cos_bg[mod][v_id].append(cm_s[m_bg].detach().cpu().numpy().astype(np.float32))
                if m_fg.any():
                    pix_cos_fg[mod][v_id].append(cm_s[m_fg].detach().cpu().numpy().astype(np.float32))

    frame_div_p95 = OrderedDict({
        "tier2_sensor_only": {m: round(max(float(np.percentile(raw_scores["t2"][m], 95.0)), 1e-4), 6) for m in ("m1", "m2")},
        "tier4_sensor_dist_vis": {m: round(max(float(np.percentile(raw_scores["t4"][m], 95.0)), 1e-4), 6) for m in ("m1", "m2")},
    })

    # 为 Fit-Dev 各项补全 v_idx_v2、B2 新前景掩膜 fg_mask、cos_comb_t4 与新 div_p95 归一化的 score_t4
    for it in fit_dev_items:
        cov_2d = it["cov_2d"].cuda()
        d_idx = it["d_idx"].cuda()
        _, v_new_4d = assign_dist5_vis3_indices_v2(
            it["dist_cav"].cuda(), it["los_2d"].view(1, 1, PROTO_H, PROTO_W).cuda(),
            it["exp_2d"].view(1, 1, PROTO_H, PROTO_W).cuda(), clear_pts_tau=clear_pts_tau, los_occ_tau=0.50
        )
        v_new = v_new_4d[0, 0]
        fg_new = build_fg_mask(
            it["p_ego_cls"].cuda().float(), it["pf_cls"].cuda().float(), v_new,
            fg_prob_tau=args.fg_prob_tau, mode=args.fg_mode,
        )
        ar = auditor.audit_dev_maps(it["dev_4ch"].cuda().float(), cov_2d, d_idx, v_new, it["mod"], build_gate=False)
        it["v_idx_v2"] = v_new.detach().cpu()
        it["fg_mask"] = fg_new.detach().cpu()
        it["cos_comb_t4"] = ar["cos_comb_t4"].detach().cpu()
        it["raw_t4"] = float(ar["raw_t4"])
        it["score_t4"] = float(ar["raw_t4"]) / frame_div_p95["tier4_sensor_dist_vis"][it["mod"]]

    # 标定整车级回退高门限（仅 Fit）
    frame_high_tau = OrderedDict()
    frame_score_summary = OrderedDict()
    for m in ("m1", "m2"):
        p95_m = frame_div_p95["tier4_sensor_dist_vis"][m]
        cln_sc = np.asarray(raw_scores["t4"][m], dtype=np.float64) / p95_m
        pgd_sc = np.asarray([it["score_t4"] for it in fit_dev_items if it["mod"] == m and it["state"] == "scaled_pgd_1.0sigma"], dtype=np.float64)
        loc_sc = np.asarray([it["score_t4"] for it in fit_dev_items if it["mod"] == m and it["state"] in ("local_removal_1.0sigma", "local_ghost_2.0sigma")], dtype=np.float64)
        cln_max = float(cln_sc.max())
        loc_p90 = float(np.percentile(loc_sc, 90.0))
        tau_hi = max(cln_max * 1.08, loc_p90 * 1.02, 1.25)
        frame_high_tau[m] = round(float(tau_hi), 6)
        frame_score_summary[m] = {
            "fit100_clean_max": round(cln_max, 4),
            "fitdev_local_p90": round(loc_p90, 4),
            "fitdev_pgd1s_med": round(float(np.median(pgd_sc)), 4),
            "chosen_frame_high_tau": frame_high_tau[m],
            "fit100_clean_trigger_rate": round(float((cln_sc > tau_hi).mean()), 4),
            "fitdev_pgd1s_trigger_rate": round(float((pgd_sc > tau_hi).mean()), 4),
            "fitdev_local_trigger_rate": round(float((loc_sc > tau_hi).mean()), 4),
        }

    # 计算并打印前景格 vs 空地格的真实良性余弦分位数对比
    q_levels = [90.0, 95.0, 98.0, 99.0, 99.5, 99.8, 99.9]
    q_labels = ["p90.0", "p95.0", "p98.0", "p99.0", "p99.5", "p99.8", "p99.9"]
    cos_q_bg = OrderedDict()
    cos_q_fg = OrderedDict()
    for m in ("m1", "m2"):
        cos_q_bg[m], cos_q_fg[m] = OrderedDict(), OrderedDict()
        for vi, vn in enumerate(VIS_LABELS_3):
            a_bg = np.concatenate(pix_cos_bg[m][vi]).astype(np.float64) if pix_cos_bg[m][vi] else np.array([1.0])
            a_fg = np.concatenate(pix_cos_fg[m][vi]).astype(np.float64) if pix_cos_fg[m][vi] else a_bg
            cos_q_bg[m][vn] = OrderedDict({ql: round(float(qv), 6) for ql, qv in zip(q_labels, np.percentile(a_bg, q_levels))})
            cos_q_fg[m][vn] = OrderedDict({ql: round(float(qv), 6) for ql, qv in zip(q_labels, np.percentile(a_fg, q_levels))})
            cos_q_fg[m][vn]["count"] = int(a_fg.size)
            cos_q_bg[m][vn]["count"] = int(a_bg.size)
    del fit100_cache, pix_cos_bg, pix_cos_fg
    torch.cuda.empty_cache()

    print("\n=== [原因 A · 前景格(fg) vs 空地格(bg) 良性平滑余弦分位数对比] ===", flush=True)
    for m in ("m1", "m2"):
        for vn in VIS_LABELS_3:
            print(
                f"  [{m} | {vn:8s}] bg(p99.8={cos_q_bg[m][vn]['p99.8']:.3f}, n={cos_q_bg[m][vn]['count']}) | "
                f"fg(p95={cos_q_fg[m][vn]['p95.0']:.3f}, p98={cos_q_fg[m][vn]['p98.0']:.3f}, "
                f"p99={cos_q_fg[m][vn]['p99.0']:.3f}, p99.5={cos_q_fg[m][vn]['p99.5']:.3f}, n={cos_q_fg[m][vn]['count']})",
                flush=True,
            )

    # === 3. 基准对照：无防御 vs 修复前旧版 BUD ===
    no_def_cache = {}
    base_ap50, base_ap70, base_brec, _, _ = _eval_fit_dev_candidate(
        wrapper, ds_m1, fit_dev_items, blind_meta_by_idx,
        g_params=None, no_def_cache=no_def_cache, ego_only_cache=ego_only_cache
    )
    old_ap50, old_ap70, old_brec, old_gst, old_occ_det = _eval_fit_dev_candidate(
        wrapper, ds_m1, fit_dev_items, blind_meta_by_idx,
        g_params=old_gating["gating_params"], no_def_cache=no_def_cache,
        ego_only_cache=ego_only_cache, use_v_idx_key="v_idx_old"
    )
    print("\n=== Fit-Dev 修复前基线对比（含遮挡伪造检出率） ===", flush=True)
    for m in ("m1", "m2"):
        print(
            f"  [{m}] OldBUD Clean AP50={old_ap50[m]['clean']:.4f} (loss={base_ap50[m]['clean']-old_ap50[m]['clean']:+.4f}) | "
            f"BlindRec={old_brec[m]['clean']:.4f} (loss={base_brec[m]['clean']-old_brec[m]['clean']:+.4f}) | "
            f"OccGhost Det={old_occ_det[m]} AP50={old_ap50[m]['local_occluded_ghost_2.0sigma']:.4f}",
            flush=True,
        )

    # === 4. 搜索 24 组「前景/空地真实良性分档」配置（不放宽倍数、不置 0 遮挡权重） ===
    bg_q_cands = ["p99.8", "p99.9"]
    fg_q_cands = ["p98.0", "p99.0", "p99.5"]
    gamma_cos_cands = [
        [1.0, 1.0, 1.0],
        [1.0, 0.85, 0.70],
    ]
    rb_cands = [(3, 3.5), (4, 4.0)]

    all_records = []
    best_strict = None
    best_strict_obj = -1e9
    best_overall = None
    best_overall_obj = -1e9
    total_grid = len(bg_q_cands) * len(fg_q_cands) * len(gamma_cos_cands) * len(rb_cands)
    gi = 0

    for bq in bg_q_cands:
        for fq in fg_q_cands:
            cos_tau_bg = {m: [cos_q_bg[m][vn][bq] for vn in VIS_LABELS_3] for m in ("m1", "m2")}
            # Clear 区前景格仍取 max(bg, fg) 且清晰空地鬼影有 exc_add 严把关；Sparse/Occluded 用真实前景良性分位
            cos_tau_fg = {m: [max(cos_q_bg[m][vn][bq], cos_q_fg[m][vn][fq]) for vn in VIS_LABELS_3] for m in ("m1", "m2")}
            for gc in gamma_cos_cands:
                for rd, bt in rb_cands:
                    gi += 1
                    g_params = {
                        "mode": "directional_vis_fg_bg",
                        "k_tol": k_tol,
                        "fg_prob_tau": float(args.fg_prob_tau),
                        "fg_mode": str(args.fg_mode),
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
                        "r_dil": rd,
                        "beta": bt,
                    }
                    ap50, ap70, brec, gst, occ_det = _eval_fit_dev_candidate(
                        wrapper, ds_m1, fit_dev_items, blind_meta_by_idx,
                        g_params=g_params, no_def_cache=no_def_cache, ego_only_cache=ego_only_cache,
                        frame_high_tau=frame_high_tau, use_v_idx_key="v_idx_v2"
                    )
                    cln_ap_loss = {m: base_ap50[m]["clean"] - ap50[m]["clean"] for m in ("m1", "m2")}
                    cln_br_loss = {m: base_brec[m]["clean"] - brec[m]["clean"] for m in ("m1", "m2")}
                    max_ap_loss = max(cln_ap_loss["m1"], cln_ap_loss["m2"])
                    max_br_loss = max(cln_br_loss["m1"], cln_br_loss["m2"])

                    atk_keys = [
                        "local_removal_1.0sigma",
                        "local_ghost_2.0sigma",
                        "local_occluded_ghost_2.0sigma",
                        "local_near0_6m_2.0sigma",
                        "scaled_pgd_1.0sigma",
                    ]
                    mean_atk_ap50 = float(np.mean([ap50[m][a] for m in ("m1", "m2") for a in atk_keys]))
                    mean_occ_auroc = 0.5 * (occ_det["m1"]["auroc"] + occ_det["m2"]["auroc"])
                    # 综合打分兼顾：攻击后 AP、遮挡区伪造检出率、及干净盲区召回保真
                    obj = mean_atk_ap50 + 0.12 * mean_occ_auroc - 0.80 * max(0.0, max_br_loss - 0.03) - 1.50 * max(0.0, max_ap_loss - 0.01)

                    print(
                        f"  [Grid {gi:2d}/{total_grid} bg={bq} fg={fq} gc={gc[2]:.2f} r={rd}] "
                        f"AP_loss=({cln_ap_loss['m1']:+.3f},{cln_ap_loss['m2']:+.3f}) | "
                        f"BR_loss=({cln_br_loss['m1']:+.3f},{cln_br_loss['m2']:+.3f}) | "
                        f"atkAP={mean_atk_ap50:.4f} occAUROC=({occ_det['m1']['auroc']:.3f},{occ_det['m2']['auroc']:.3f}) "
                        f"({time.time()-t0:.0f}s)",
                        flush=True,
                    )

                    rec = {
                        "gating_params": g_params,
                        "ap50": {m: {k: round(v, 5) for k, v in ap50[m].items()} for m in ("m1", "m2")},
                        "ap70": {m: {k: round(v, 5) for k, v in ap70[m].items()} for m in ("m1", "m2")},
                        "blind_recall": {m: {k: round(v, 5) for k, v in brec[m].items()} for m in ("m1", "m2")},
                        "clean_ap50_loss": {m: round(cln_ap_loss[m], 5) for m in ("m1", "m2")},
                        "clean_blind_recall_loss": {m: round(cln_br_loss[m], 5) for m in ("m1", "m2")},
                        "max_clean_ap50_loss": round(max_ap_loss, 5),
                        "max_clean_blind_recall_loss": round(max_br_loss, 5),
                        "mean_atk_ap50": round(mean_atk_ap50, 5),
                        "occluded_ghost_detection": occ_det,
                        "gate_stats": {
                            "clean_valid": round(float(np.mean(gst["clean_valid"])), 5),
                            "clean_blind_gt": round(float(np.mean(gst["clean_blind_gt"])), 5) if gst["clean_blind_gt"] else 1.0,
                            "lrem_atk": round(float(np.mean(gst["lrem_atk"])), 5) if gst["lrem_atk"] else 1.0,
                            "lgho_atk": round(float(np.mean(gst["lgho_atk"])), 5) if gst["lgho_atk"] else 1.0,
                            "locc_atk": round(float(np.mean(gst["locc_atk"])), 5) if gst["locc_atk"] else 1.0,
                        },
                        "obj": round(obj, 5),
                    }
                    all_records.append(rec)

                    if max_ap_loss <= 0.010 and max_br_loss <= 0.030 and obj > best_strict_obj:
                        best_strict_obj = obj
                        best_strict = rec
                    if obj > best_overall_obj:
                        best_overall_obj = obj
                        best_overall = rec

    fit_strict_met = bool(best_strict is not None)
    chosen = best_strict if fit_strict_met else best_overall
    status_str = "达标（Fit-Dev <=3%）" if fit_strict_met else "未达标（Fit仅70辆盲区车波动大，沿用最优参数并在Dev-320验收）"

    print(f"\n=== [最终选型] {status_str} ===", flush=True)
    print("  clean_ap50_loss:", chosen["clean_ap50_loss"])
    print("  clean_blind_recall_loss:", chosen["clean_blind_recall_loss"])
    print("  occluded_ghost_det (OLD vs NEW):")
    print("    OLD:", json.dumps(old_occ_det, ensure_ascii=False))
    print("    NEW:", json.dumps(chosen["occluded_ghost_detection"], ensure_ascii=False))

    out_obj = OrderedDict({
        "meta": {
            "description": "步骤 ④-4 BUD-Fusion 冻结版标定（新物理可见性定义 + 前景/空地条件双分档 + 整车回退高门限）",
            "source_split": args.split_json,
            "source_base_calib": args.base_calib_json,
            "fit_total_frames": len(fit_indices),
            "fit_dev_frames": len(fit_dev_set),
            "frame_norm_mode": "div_p95",
            "fit_target_met": fit_strict_met,
            "fit_status": status_str,
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "visibility_v2": vis_v2_cfg,
        "frame_div_p95": frame_div_p95,
        "frame_high_tau": frame_high_tau,
        "frame_score_summary": frame_score_summary,
        "pixel_cos_by_vis_bg_quantiles": cos_q_bg,
        "pixel_cos_by_vis_fg_quantiles": cos_q_fg,
        "gating_params": chosen["gating_params"],
        "occluded_ghost_before_vs_after_fit_dev": {
            "before_old_bud": {
                "detection": old_occ_det,
                "ap50": {m: round(old_ap50[m]["local_occluded_ghost_2.0sigma"], 5) for m in ("m1", "m2")},
                "locc_atk_gate": round(float(np.mean(old_gst["locc_atk"])), 5) if old_gst["locc_atk"] else 1.0,
            },
            "after_new_bud": {
                "detection": chosen["occluded_ghost_detection"],
                "ap50": {m: chosen["ap50"][m]["local_occluded_ghost_2.0sigma"] for m in ("m1", "m2")},
                "locc_atk_gate": chosen["gate_stats"]["locc_atk"],
            },
        },
        "fit_dev_baseline": {
            "no_defense_ap50": {m: {k: round(v, 5) for k, v in base_ap50[m].items()} for m in ("m1", "m2")},
            "no_defense_blind_recall": {m: {k: round(v, 5) for k, v in base_brec[m].items()} for m in ("m1", "m2")},
            "old_bud_ap50": {m: {k: round(v, 5) for k, v in old_ap50[m].items()} for m in ("m1", "m2")},
            "old_bud_blind_recall": {m: {k: round(v, 5) for k, v in old_brec[m].items()} for m in ("m1", "m2")},
        },
        "fit_dev_chosen_eval": chosen,
        "fit_dev_grid_records": all_records,
    })
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)
    print(f"\n已冻结保存至：{args.out_json} (总耗时 {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
