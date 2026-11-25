# -*- coding: utf-8 -*-
"""
步骤 ⑤-2 & 多车队分层主实验统一评测脚本（等 BUD-Fusion 冻结与 HetShield 训练完成后执行）。

完整实现全部对比基线与多车队设计：
1. 9 种防御/方法同台公平对比：
   - ego_only, no_defense, whole_cav_drop, bud_frozen,
   - hetshield_paper (τ=0.3, T=0.1), hetshield_tuned (Fit p95), hetshield_persensor (分模态 ConvGRU+p95),
   - lucia (官方 USENIX Security'25), robosac (官方 ICCV'23，严格仅在 >=3 车即 collab>=2 场景启用)；
2. 3 种车队组成：
   - mixed（主实验：本车 m1，协作者 m2/m1 交替分配）
   - all_m1（协作者全 m1）
   - all_m2（协作者全 m2）；
3. 按实际协作车数量严格分层报告：collab_1 (1协作车) / collab_2 (2协作车) / collab_ge3 (>=3协作车)；
4. 攻击者设置（固定随机种子，完全可复现）：
   - 主实验：随机选 1 辆协作车为攻击者，在混合车队中分别报告「攻击者为 m1」与「攻击者为 m2」的结果；
   - 压力测试：在 >=4 车（collab >= 3）场景随机选 2 辆协作车同时攻击（two_attackers）；
5. 混合车队专属细粒度指标：
   - 同一帧内 m1、m2 诚实车各自的误报率；
   - 一辆车被攻击时，同帧其他诚实车的融合权重变化（clean vs attacked 下 base / bud / lucia）；
6. HetShield 公平时序设计：对每个评测帧 idx，从原始连续序列自动读取紧邻前一帧 idx-1 推一步预测
   （序列首帧置 history_len=1, tau_k=0），杜绝跨帧抽样导致的时间一致性失真；
7. 全部方法对比统一输出成对差值 2,000 次按序列 Block-Bootstrap 95% 置信区间，并支持按序列断点缓存。
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

PROJ_ROOT = "/root/autodl-tmp/proj_dir2_hetero_trust"
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from src_v2_mm.baselines_robosac_lucia_heal import (
    ROBOSACHEALDefender,
    build_lucia_fuse_fn,
)
from src_v2_mm.bud_fusion_heal import (
    BUDFusionHEALAuditor,
    build_gated_fuse_fn,
    build_whole_cav_reject_fuse_fn,
    extract_single_cav_dev_and_signed,
)
from src_v2_mm.eval_stats_utils import (
    build_seq_bootstrap_indices,
    paired_block_bootstrap_ap_diff,
    paired_block_bootstrap_rate_diff,
)
from src_v2_mm.heal_attacks import (
    generate_local_ghost_spoofing_heal,
    generate_local_removal_heal,
    generate_scaled_pgd_removal_heal,
)
from src_v2_mm.heal_base_wrapper import HEALBaseWrapper
from src_v2_mm.hetshield_heal import (
    HetShieldHEALDefender,
    build_hetshield_fuse_fn,
)
from src_v2_mm.step4_1_v2_core import (
    GRID_RES,
    MIN_XY,
    PROTO_H,
    PROTO_W,
    TRAIN_FULL19_DIR,
    assign_dist5_vis3_indices,
    assign_dist5_vis3_indices_v2,
    build_bev_grid_xy_256,
    compute_ego_beam_visibility_256,
    compute_ego_expected_pts_and_los_256,
    load_ego_lidar_np,
)

DEFAULT_EVAL_SPLIT = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_3_eval320_split.json")
DEFAULT_BASE_CALIB = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_1_v2_frozen_calibrator.json")
DEFAULT_FROZEN_GATING = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_4_bud_frozen_calibrator.json")
DEFAULT_HETSHIELD_CALIB = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step5_1_hetshield_heal_calib.json")
DEFAULT_CACHE_DIR = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step5_2_multicav_cache")
DEFAULT_OUT_JSON = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step5_2_multicav_baselines_report.json")

METHODS_ALL = [
    "ego_only",
    "no_defense",
    "whole_cav_drop",
    "bud_frozen",
    "hetshield_paper",
    "hetshield_tuned",
    "hetshield_persensor",
    "lucia",
    "robosac",
]
ATTACK_TYPES = ["clean", "pgd_1.0sigma", "local_removal_1.0sigma", "local_ghost_2.0sigma"]
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
        if not rem_polys:
            fp_list.append(1)
            tp_list.append(0)
            continue
        ious = common_utils.compute_iou(det_polys[int(order[k])], rem_polys)
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


def _find_multicav_blind_gts(ds, idx: int, ego_pb, ego_ps, ego_gb, union_cov_2d: torch.Tensor) -> dict:
    """多车队客观盲区 GT：落入任意协作车有效覆盖并集 ∧ 本车 64 线三维框内点数 <= 5 ∧ Ego-Only 漏检。"""
    from matplotlib.path import Path as MplPath
    from opencood.utils import common_utils

    gt_np = common_utils.torch_tensor_to_numpy(ego_gb)
    gt_polys = list(common_utils.convert_format(gt_np))
    ego_r05 = _match_tp_fp_and_hits(ego_pb, ego_ps, gt_np, gt_polys, 0.5)
    ego_r07 = _match_tp_fp_and_hits(ego_pb, ego_ps, gt_np, gt_polys, 0.7)
    ego_matched = ego_r05[4]
    ego_pts = load_ego_lidar_np(ds, idx)
    cov_np = union_cov_2d.detach().cpu().numpy()

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
        "ego_r05": ego_r05,
        "ego_r07": ego_r07,
        "blind_indices": blind_indices,
    }


def _collab_tier(n_collab: int) -> str:
    if n_collab <= 1:
        return "collab_1"
    if n_collab == 2:
        return "collab_2"
    return "collab_ge3"


def _apply_attack_to_indices(
    wrapper,
    feat_clean: torch.Tensor,
    rlen: torch.Tensor,
    amat: torch.Tensor,
    mods_l: list,
    vmask: torch.Tensor,
    attack_type: str,
    attacker_indices: list,
    frame_seed: int,
) -> torch.Tensor:
    """对指定的 1 个或多个协作者索引施加选定类型的传输层攻击，其他诚实协作者特征严格不变。"""
    if attack_type == "clean" or not attacker_indices:
        return feat_clean.clone()
    feat_cur = feat_clean.clone()
    for step_k, a_idx in enumerate(attacker_indices):
        if attack_type == "pgd_1.0sigma":
            feat_cur, _ = generate_scaled_pgd_removal_heal(
                wrapper, feat_cur, rlen, amat, mods_l, vmask,
                attacker_idx=a_idx, sigma_mult=1.0, steps=8,
            )
        elif attack_type == "local_removal_1.0sigma":
            feat_cur, _ = generate_local_removal_heal(
                wrapper, feat_cur, rlen, amat, mods_l, vmask,
                attacker_idx=a_idx, num_targets=2, patch_radius=5, sigma_mult=1.0, steps=10,
            )
        elif attack_type == "local_ghost_2.0sigma":
            feat_cur, _ = generate_local_ghost_spoofing_heal(
                wrapper, feat_cur, rlen, amat, mods_l, vmask,
                attacker_idx=a_idx, num_spoofs=2, patch_radius=5, sigma_mult=2.0, steps=10,
                frame_seed=int(frame_seed) + step_k * 97,
            )
        else:
            raise ValueError(f"未知攻击类型: {attack_type}")
    return feat_cur


def _select_attack_scenarios(mods_l: list, frame_seed: int) -> list:
    """
    依据固定随机种子为当前帧构造待测攻击者配置列表：
    - 返回项格式: (role_tag, attacker_indices_list)
    - 单攻击者：分别挑 1 辆 m1 协作车（若存在）与 1 辆 m2 协作车（若存在）；
    - 压力测试：总车数 >= 4（即协作车 >= 3）时随机挑 2 辆协作车作为双攻击者。
    """
    n_total = len(mods_l)
    m1_collabs = [i for i in range(1, n_total) if mods_l[i] == "m1"]
    m2_collabs = [i for i in range(1, n_total) if mods_l[i] == "m2"]
    all_collabs = list(range(1, n_total))
    rng = random.Random(20261120 + int(frame_seed) * 31)

    scenarios = []
    if m1_collabs:
        scenarios.append(("atk_m1", [rng.choice(m1_collabs)]))
    if m2_collabs:
        scenarios.append(("atk_m2", [rng.choice(m2_collabs)]))
    if len(all_collabs) >= 3:
        two_atk = sorted(rng.sample(all_collabs, k=2))
        scenarios.append(("two_attackers", two_atk))
    return scenarios


def _load_hetshield_defenders(hs_calib_path: str, device: str = "cuda") -> dict:
    """加载三套修复版 HetShield 防御器：paper_default / fit_tuned / persensor。"""
    with open(hs_calib_path, "r", encoding="utf-8") as f:
        hc = json.load(f)
    meta = hc["meta"]
    e_th = float(meta["energy_thresh"])
    shared_sd = torch.load(meta["shared_ckpt"], map_location="cpu")
    per_sd = torch.load(meta["persensor_ckpt"], map_location="cpu")

    hs_paper = HetShieldHEALDefender(
        mode="shared", alpha=0.5, tau_th=0.3, temperature=0.1, energy_thresh=e_th
    ).to(device).eval()
    hs_paper.predictor.load_state_dict(shared_sd)

    hs_tuned = HetShieldHEALDefender(
        mode="shared", alpha=0.5, tau_th=float(hc["fit_tuned_shared_tau_p95"]), temperature=0.1, energy_thresh=e_th
    ).to(device).eval()
    hs_tuned.predictor.load_state_dict(shared_sd)

    hs_per = HetShieldHEALDefender(
        mode="persensor", alpha=0.5, tau_th=0.3, temperature=0.1, energy_thresh=e_th,
        per_sensor_tau=hc["fit_persensor_tau_p95"],
    ).to(device).eval()
    for m in ("m1", "m2"):
        hs_per.predictors[m].load_state_dict(per_sd[m])

    return {
        "hetshield_paper": hs_paper,
        "hetshield_tuned": hs_tuned,
        "hetshield_persensor": hs_per,
    }


def _eval_hetshield_step_fair(
    hs_def: HetShieldHEALDefender,
    wrapper,
    seq_name: str,
    idx: int,
    prev_p: dict,
    cur_feat: torch.Tensor,
    cur_p: dict,
) -> list:
    """
    对 HetShield 执行严格公平的真实相邻帧一步预测：
    - 若存在同序列紧邻前一帧 prev_p（idx - 1），先以干净 prev_p 滚动一步生成 pred_t；
    - 再在当前帧 cur_feat 上评估 s_k 与 g_k（不泄露攻击帧到其他工况）。
    """
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

    hs_def.reset_sequence(seq_name)
    ac = wrapper.model.pyramid_backbone.align_corners

    def _build_cav_items(p_dict, feat_tensor):
        n_c = int(feat_tensor.shape[0])
        items = []
        for ci in range(1, n_c):
            tmat = p_dict["affine_matrix"][0, 0, ci : ci + 1]
            fc = warp_affine_simple(feat_tensor[ci : ci + 1], tmat, (PROTO_H, PROTO_W), align_corners=ac)
            vc = warp_affine_simple(p_dict["cam_valid_mask"][ci : ci + 1].float(), tmat, (PROTO_H, PROTO_W), align_corners=ac)
            items.append({
                "cav_id": f"{seq_name}_c{ci}",
                "mod": p_dict["agent_modality_list"][ci],
                "feat_warped": fc,
                "valid_ego": (vc > 0.5).float(),
            })
        return items

    if prev_p is not None and int(prev_p["heter_feature_2d"].shape[0]) == int(cur_feat.shape[0]):
        prev_items = _build_cav_items(prev_p, prev_p["heter_feature_2d"])
        hs_def.step_frame(seq_name, idx - 1, prev_p["heter_feature_2d"][0:1], prev_items, update_state=True)

    cur_items = _build_cav_items(cur_p, cur_feat)
    res = hs_def.step_frame(seq_name, idx, cur_feat[0:1], cur_items, update_state=False)
    return res


def _run_fleet_sequence(
    wrapper,
    ds,
    sample_seqs: list,
    auditor: BUDFusionHEALAuditor,
    hs_defs: dict,
    robosac: ROBOSACHEALDefender,
    fleet_mode: str,
    seq_name: str,
    indices: list,
    gx: torch.Tensor,
    gy: torch.Tensor,
) -> dict:
    """运行单个 (fleet_mode, seq_name) 的全部评测并打包返回可缓存 JSON。"""
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
    from opencood.tools import train_utils
    from opencood.utils import common_utils

    t0 = time.time()
    # 动态收集所有出现过的 (collab_tier, role_tag, attack_type) 桶
    ap_buckets = {}
    blind_buckets = {}
    blind_totals = {}
    # 混合车队专属细粒度记录：诚实车按模态分档误报率（含 BUD 与 HetShield 三版对照）+ 攻击时诚实车权重变化
    honest_fpr_records = {
        "m1": {"fp_t4": 0, "fp_fb": 0, "fp_hs_paper": 0, "fp_hs_tuned": 0, "fp_hs_per": 0, "total": 0},
        "m2": {"fp_t4": 0, "fp_fb": 0, "fp_hs_paper": 0, "fp_hs_tuned": 0, "fp_hs_per": 0, "total": 0},
    }
    honest_weight_records = []

    def _ensure_bucket(tier: str, role: str, atk: str):
        key = f"{tier}|{role}|{atk}"
        if key not in ap_buckets:
            ap_buckets[key] = {
                m: {"tp50": [], "fp50": [], "tp70": [], "fp70": [], "score": [], "gt": 0}
                for m in METHODS_ALL
            }
            blind_buckets[key] = {m: 0 for m in METHODS_ALL}
            blind_totals[key] = 0
        return key

    def _rec(key: str, meth: str, r05, r07, blind_idxs: list):
        d = ap_buckets[key][meth]
        d["tp50"].extend(r05[0])
        d["fp50"].extend(r05[1])
        d["tp70"].extend(r07[0])
        d["fp70"].extend(r07[1])
        d["score"].extend(r05[2])
        d["gt"] += int(r05[3])
        if blind_idxs:
            blind_buckets[key][meth] += sum(1 for bi in blind_idxs if r05[4][bi])

    for fi, idx in enumerate(indices):
        raw_b = train_utils.to_device(ds.collate_batch_test([ds[idx]]), wrapper.device_obj)
        b_post = train_utils.to_device(_slim_batch_for_postprocess(raw_b), wrapper.device_obj)
        p_cur = wrapper.extract_protocol_features(raw_b["ego"])
        feat_clean = p_cur["heter_feature_2d"]
        rlen = p_cur["record_len"]
        amat = p_cur["affine_matrix"]
        mods_l = list(p_cur["agent_modality_list"])
        vmask = p_cur["cam_valid_mask"]
        n_total = int(feat_clean.shape[0])
        n_collab = n_total - 1
        if n_collab < 1:
            continue
        tier = _collab_tier(n_collab)

        # 读取真实紧邻前一帧（若属于同一父序列）供 HetShield 公平一步预测
        prev_p = None
        if idx > 0 and sample_seqs[idx - 1] == seq_name:
            raw_prev = train_utils.to_device(ds.collate_batch_test([ds[idx - 1]]), wrapper.device_obj)
            prev_p = wrapper.extract_protocol_features(raw_prev["ego"])
            del raw_prev

        # 本车物理可见性（优先采用 Fit 标定的新可见性定义 visibility_v2）、单车预测与多车覆盖并集
        ego_pts = load_ego_lidar_np(ds, idx)
        vis_v2 = (auditor.gating_calib or {}).get("visibility_v2", None)
        fg_prob_tau = float((auditor.gating_calib or {}).get("gating_params", {}).get("fg_prob_tau", 0.20))
        sigma_alea, los_f, exp_pts = compute_ego_expected_pts_and_los_256(ego_pts, gx, gy, num_beams=64)
        ac = wrapper.model.pyramid_backbone.align_corners

        per_cav_didx = {}
        per_cav_vidx = {}
        union_cov = torch.zeros((PROTO_H, PROTO_W), device=wrapper.device_obj, dtype=torch.bool)
        for ci in range(1, n_total):
            ptm_ci = raw_b["ego"]["pairwise_t_matrix"][0, ci, 0].detach().cpu().numpy()
            cx, cy = float(ptm_ci[0, 3]), float(ptm_ci[1, 3])
            dist_ci = torch.sqrt((gx - cx) ** 2 + (gy - cy) ** 2).view(1, 1, PROTO_H, PROTO_W)
            if vis_v2 is not None:
                d_i, v_i = assign_dist5_vis3_indices_v2(
                    dist_ci, los_f, exp_pts,
                    clear_pts_tau=float(vis_v2["clear_pts_tau"]),
                    los_occ_tau=float(vis_v2.get("los_occ_tau", 0.50)),
                )
            else:
                d_i, v_i = assign_dist5_vis3_indices(dist_ci, sigma_alea)
            per_cav_didx[ci] = d_i[0, 0]
            per_cav_vidx[ci] = v_i[0, 0]
            cov_ci = warp_affine_simple(vmask[ci : ci + 1].float(), amat[0, 0, ci : ci + 1], (PROTO_H, PROTO_W), align_corners=ac) > 0.5
            union_cov = union_cov | cov_ci[0, 0]

        with torch.no_grad():
            ego_out = wrapper.forward_single_agent(feat_clean[0:1])
            p_ego_cls = torch.sigmoid(ego_out["cls_preds"]).max(dim=1, keepdim=True)[0]
            ego_pb, ego_ps, ego_gb = wrapper.decode_boxes(b_post, ego_out, ds)
            ego_pb_np = common_utils.torch_tensor_to_numpy(ego_pb) if (ego_pb is not None and len(ego_pb) > 0) else None

        binfo = _find_multicav_blind_gts(ds, idx, ego_pb, ego_ps, ego_gb, union_cov)
        gt_np, gt_polys = binfo["gt_np"], binfo["gt_polys"]
        ego_r05, ego_r07 = binfo["ego_r05"], binfo["ego_r07"]
        blind_idxs = binfo["blind_indices"]

        # 构造待测场景列表：Clean 固定跑一次 + 各攻击者角色 × 攻击类型
        run_jobs = [("clean_all", "clean", [])]
        for role_tag, atk_idxs in _select_attack_scenarios(mods_l, idx):
            for atk_t in ATTACK_TYPES:
                if atk_t == "clean":
                    continue
                run_jobs.append((role_tag, atk_t, atk_idxs))

        clean_honest_w_bud = {}
        clean_honest_w_base = {}

        for role_tag, atk_t, atk_idxs in run_jobs:
            bkey = _ensure_bucket(tier, role_tag, atk_t)
            blind_totals[bkey] += len(blind_idxs)

            f_cur = _apply_attack_to_indices(wrapper, feat_clean, rlen, amat, mods_l, vmask, atk_t, atk_idxs, idx)

            # 1) Ego-Only
            _rec(bkey, "ego_only", ego_r05, ego_r07, blind_idxs)

            # 2) No-Defense
            with torch.no_grad():
                out_nd = wrapper.forward_from_protocol(f_cur, rlen, amat, mods_l)
                nd_pb, nd_ps, _ = wrapper.decode_boxes(b_post, out_nd, ds)
            nd_r05 = _match_tp_fp_and_hits(nd_pb, nd_ps, gt_np, gt_polys, 0.5)
            nd_r07 = _match_tp_fp_and_hits(nd_pb, nd_ps, gt_np, gt_polys, 0.7)
            _rec(bkey, "no_defense", nd_r05, nd_r07, blind_idxs)

            # 3) 逐车独立审计 -> Whole-CAV-Drop 与 BUD-Fusion (Frozen)
            reject_flags = []
            bud_gates = []
            for ci in range(1, n_total):
                mod_ci = mods_l[ci]
                dev_ci, d_drop_ci, d_add_ci, cov_ci, p_pair_ci = extract_single_cav_dev_and_signed(
                    wrapper, f_cur, amat, mods_l, vmask, p_ego_cls, cav_idx=ci, k_tol=5, return_pfused=True
                )
                fg_mask_ci = F.max_pool2d(p_pair_ci, kernel_size=3, stride=1, padding=1)[0, 0] > fg_prob_tau
                ar_ci = auditor.audit_dev_maps(
                    dev_ci, cov_ci, per_cav_didx[ci], per_cav_vidx[ci], mod_ci,
                    build_gate=True, d_drop_tol=d_drop_ci, d_add_tol=d_add_ci,
                    enable_whole_fallback=True, collab_fg_mask=fg_mask_ci
                )
                reject_flags.append(bool(ar_ci["score_t4"] > 1.0))
                bud_gates.append(ar_ci["gate_L0"])

                # 记录干净帧下混合车队各模态诚实车误报率
                if atk_t == "clean" and mod_ci in honest_fpr_records:
                    honest_fpr_records[mod_ci]["total"] += 1
                    if ar_ci["score_t4"] > 1.0:
                        honest_fpr_records[mod_ci]["fp_t4"] += 1
                    if ar_ci["whole_fallback"]:
                        honest_fpr_records[mod_ci]["fp_fb"] += 1

            # Whole-CAV-Drop
            if all(reject_flags):
                _rec(bkey, "whole_cav_drop", ego_r05, ego_r07, blind_idxs)
            elif not any(reject_flags):
                _rec(bkey, "whole_cav_drop", nd_r05, nd_r07, blind_idxs)
            else:
                with torch.no_grad():
                    out_wd = wrapper.forward_from_protocol(
                        f_cur, rlen, amat, mods_l,
                        custom_fuse_fn=build_whole_cav_reject_fuse_fn(reject_flags, device=str(wrapper.device_obj))
                    )
                    wpb, wps, _ = wrapper.decode_boxes(b_post, out_wd, ds)
                _rec(bkey, "whole_cav_drop", _match_tp_fp_and_hits(wpb, wps, gt_np, gt_polys, 0.5),
                     _match_tp_fp_and_hits(wpb, wps, gt_np, gt_polys, 0.7), blind_idxs)

            # BUD-Fusion (Frozen)
            gate_stack = torch.cat(bud_gates, dim=0)
            with torch.no_grad():
                out_bud = wrapper.forward_from_protocol(
                    f_cur, rlen, amat, mods_l, custom_fuse_fn=build_gated_fuse_fn(gate_stack)
                )
                bpb, bps, _ = wrapper.decode_boxes(b_post, out_bud, ds)
            _rec(bkey, "bud_frozen", _match_tp_fp_and_hits(bpb, bps, gt_np, gt_polys, 0.5),
                 _match_tp_fp_and_hits(bpb, bps, gt_np, gt_polys, 0.7), blind_idxs)

            # 记录混合车队中一辆车被攻击时，同帧其他诚实车融合权重变化
            fdiag_bud = out_bud.get("fusion_diag", {})
            w_base_l0 = fdiag_bud.get("per_cav_w_base_L0", [])
            w_bud_l0 = fdiag_bud.get("per_cav_w_gated_L0", [])
            if atk_t == "clean" and len(w_bud_l0) == n_collab:
                for ci in range(1, n_total):
                    clean_honest_w_base[ci] = float(w_base_l0[ci - 1])
                    clean_honest_w_bud[ci] = float(w_bud_l0[ci - 1])

            # 4) LUCIA (官方原味)
            with torch.no_grad():
                out_luc = wrapper.forward_from_protocol(
                    f_cur, rlen, amat, mods_l, custom_fuse_fn=build_lucia_fuse_fn(compression_ratio=32)
                )
                lpb, lps, _ = wrapper.decode_boxes(b_post, out_luc, ds)
            _rec(bkey, "lucia", _match_tp_fp_and_hits(lpb, lps, gt_np, gt_polys, 0.5),
                 _match_tp_fp_and_hits(lpb, lps, gt_np, gt_polys, 0.7), blind_idxs)
            w_luc_l0 = out_luc.get("fusion_diag", {}).get("per_cav_w_gated_L0", [])

            if atk_t != "clean" and len(atk_idxs) == 1 and n_collab >= 2 and len(w_bud_l0) == n_collab:
                atk_set = set(atk_idxs)
                for ci in range(1, n_total):
                    if ci in atk_set:
                        continue
                    honest_weight_records.append({
                        "sample_idx": int(idx),
                        "role_tag": role_tag,
                        "attack_type": atk_t,
                        "honest_mod": mods_l[ci],
                        "w_clean_base": clean_honest_w_base.get(ci, 0.0),
                        "w_clean_bud": clean_honest_w_bud.get(ci, 0.0),
                        "w_atk_base": float(w_base_l0[ci - 1]),
                        "w_atk_bud": float(w_bud_l0[ci - 1]),
                        "w_atk_lucia": float(w_luc_l0[ci - 1]) if len(w_luc_l0) == n_collab else 0.0,
                    })

            # 5) 三套修复版 HetShield (真实紧邻前帧公平一步预测)
            hs_key_map = {
                "hetshield_paper": "fp_hs_paper",
                "hetshield_tuned": "fp_hs_tuned",
                "hetshield_persensor": "fp_hs_per",
            }
            for hs_name, hs_def in hs_defs.items():
                hs_res = _eval_hetshield_step_fair(hs_def, wrapper, seq_name, idx, prev_p, f_cur, p_cur)
                if atk_t == "clean":
                    fp_k = hs_key_map[hs_name]
                    for r_item in hs_res:
                        rm = r_item["mod"]
                        if rm in honest_fpr_records and r_item["is_quarantined"]:
                            honest_fpr_records[rm][fp_k] += 1
                g_list = [r["g_k"] for r in hs_res]
                with torch.no_grad():
                    out_hs = wrapper.forward_from_protocol(
                        f_cur, rlen, amat, mods_l,
                        custom_fuse_fn=build_hetshield_fuse_fn(g_list, device=str(wrapper.device_obj))
                    )
                    hpb, hps, _ = wrapper.decode_boxes(b_post, out_hs, ds)
                _rec(bkey, hs_name, _match_tp_fp_and_hits(hpb, hps, gt_np, gt_polys, 0.5),
                     _match_tp_fp_and_hits(hpb, hps, gt_np, gt_polys, 0.7), blind_idxs)

            # 6) ROBOSAC (严格仅在 >=3 车即 collab >= 2 场景执行)
            if n_total >= 3:
                rob_res = robosac.run_frame(
                    wrapper, f_cur, rlen, amat, mods_l, b_post, ds,
                    ego_boxes_np=ego_pb_np, ego_pb=ego_pb, ego_ps=ego_ps, ego_gb=ego_gb,
                    frame_seed=idx, assumed_attackers=max(1, len(atk_idxs)),
                )
                rpb, rps = rob_res["pb"], rob_res["ps"]
                _rec(bkey, "robosac", _match_tp_fp_and_hits(rpb, rps, gt_np, gt_polys, 0.5),
                     _match_tp_fp_and_hits(rpb, rps, gt_np, gt_polys, 0.7), blind_idxs)

        if (fi + 1) % 10 == 0 or (fi + 1) == len(indices):
            print(f"  [{fleet_mode} | {seq_name}] 进度 {fi + 1}/{len(indices)} ({time.time() - t0:.1f}s)", flush=True)

    return {
        "fleet_mode": fleet_mode,
        "seq_name": seq_name,
        "num_frames": len(indices),
        "ap_buckets": ap_buckets,
        "blind_buckets": blind_buckets,
        "blind_totals": blind_totals,
        "honest_fpr_records": honest_fpr_records,
        "honest_weight_records": honest_weight_records,
        "elapsed_sec": round(time.time() - t0, 2),
    }


def _aggregate_fleet_caches(seq_order: list, seq_caches: dict) -> dict:
    """按 (collab_tier, role_tag, attack_type) 汇总全方法 AP/盲区召回及成对差值 Bootstrap 95% CI。"""
    all_keys = sorted({k for sc in seq_caches.values() for k in sc["ap_buckets"].keys()})
    results = OrderedDict()

    for key in all_keys:
        # 仅在包含该桶的父序列集合上执行按序列 Block-Bootstrap
        rel_seqs = [s for s in seq_order if key in seq_caches[s]["ap_buckets"]]
        if not rel_seqs:
            continue
        boot_idx = build_seq_bootstrap_indices(rel_seqs, n_boot=BOOT_N, seed=BOOT_SEED)
        tier = key.split("|")[0]
        active_methods = [m for m in METHODS_ALL if not (m == "robosac" and tier == "collab_1")]

        s50, s70, sbh = {}, {}, {}
        sbtot = {s: int(seq_caches[s]["blind_totals"][key]) for s in rel_seqs}
        for meth in active_methods:
            s50[meth] = {
                s: {
                    "tp": np.asarray(seq_caches[s]["ap_buckets"][key][meth]["tp50"], dtype=np.float64),
                    "fp": np.asarray(seq_caches[s]["ap_buckets"][key][meth]["fp50"], dtype=np.float64),
                    "score": np.asarray(seq_caches[s]["ap_buckets"][key][meth]["score"], dtype=np.float64),
                    "gt": int(seq_caches[s]["ap_buckets"][key][meth]["gt"]),
                }
                for s in rel_seqs
            }
            s70[meth] = {
                s: {
                    "tp": np.asarray(seq_caches[s]["ap_buckets"][key][meth]["tp70"], dtype=np.float64),
                    "fp": np.asarray(seq_caches[s]["ap_buckets"][key][meth]["fp70"], dtype=np.float64),
                    "score": np.asarray(seq_caches[s]["ap_buckets"][key][meth]["score"], dtype=np.float64),
                    "gt": int(seq_caches[s]["ap_buckets"][key][meth]["gt"]),
                }
                for s in rel_seqs
            }
            sbh[meth] = {s: int(seq_caches[s]["blind_buckets"][key][meth]) for s in rel_seqs}

        # 全部基线与 bud_frozen 形成成对差值 95% CI
        paired_vs_bud = OrderedDict()
        meth_summary = OrderedDict()
        for mb in active_methods:
            if mb == "bud_frozen":
                continue
            r50 = paired_block_bootstrap_ap_diff(rel_seqs, s50["bud_frozen"], s50[mb], boot_indices=boot_idx)
            r70 = paired_block_bootstrap_ap_diff(rel_seqs, s70["bud_frozen"], s70[mb], boot_indices=boot_idx)
            rbl = paired_block_bootstrap_rate_diff(rel_seqs, sbh["bud_frozen"], sbh[mb], sbtot, boot_indices=boot_idx)
            paired_vs_bud[f"bud_frozen_vs_{mb}"] = {"ap50": r50, "ap70": r70, "blind_recall": rbl}
            if "bud_frozen" not in meth_summary:
                meth_summary["bud_frozen"] = {
                    "ap50": {"val": r50["val_a"], "ci95": r50["a_ci95"]},
                    "ap70": {"val": r70["val_a"], "ci95": r70["a_ci95"]},
                    "blind_recall": {"val": rbl["val_a"], "ci95": rbl["a_ci95"]},
                }
            meth_summary[mb] = {
                "ap50": {"val": r50["val_b"], "ci95": r50["b_ci95"]},
                "ap70": {"val": r70["val_b"], "ci95": r70["b_ci95"]},
                "blind_recall": {"val": rbl["val_b"], "ci95": rbl["b_ci95"]},
            }

        results[key] = {
            "num_sequences": len(rel_seqs),
            "sequences": rel_seqs,
            "ci_note": "单条父序列无法跨序列整块重采样（CI 退化为点估计）" if len(rel_seqs) == 1 else "2000次按父序列Block-Bootstrap 95% CI",
            "methods": meth_summary,
            "paired_diffs_vs_bud": paired_vs_bud,
        }

    # 汇总混合车队诚实车分模态 FPR（含 BUD 与 HetShield 三版对照）与被攻击时诚实车权重变化
    fpr_agg = OrderedDict()
    for m in ("m1", "m2"):
        fp_t4 = sum(int(seq_caches[s]["honest_fpr_records"][m]["fp_t4"]) for s in seq_order)
        fp_fb = sum(int(seq_caches[s]["honest_fpr_records"][m]["fp_fb"]) for s in seq_order)
        fp_hp = sum(int(seq_caches[s]["honest_fpr_records"][m].get("fp_hs_paper", 0)) for s in seq_order)
        fp_ht = sum(int(seq_caches[s]["honest_fpr_records"][m].get("fp_hs_tuned", 0)) for s in seq_order)
        fp_hr = sum(int(seq_caches[s]["honest_fpr_records"][m].get("fp_hs_per", 0)) for s in seq_order)
        tot = sum(int(seq_caches[s]["honest_fpr_records"][m]["total"]) for s in seq_order)
        fpr_agg[m] = {
            "total_honest_cav_frames": tot,
            "bud_fpr_score_gt_1": round(fp_t4 / max(tot, 1), 5),
            "bud_fpr_whole_fallback": round(fp_fb / max(tot, 1), 5),
            "hetshield_paper_fpr": round(fp_hp / max(tot, 1), 5),
            "hetshield_tuned_fpr": round(fp_ht / max(tot, 1), 5),
            "hetshield_persensor_fpr": round(fp_hr / max(tot, 1), 5),
        }

    all_w_recs = [r for s in seq_order for r in seq_caches[s]["honest_weight_records"]]
    weight_shift = OrderedDict()
    for hm in ("m1", "m2"):
        sub = [r for r in all_w_recs if r["honest_mod"] == hm]
        if sub:
            weight_shift[hm] = {
                "count": len(sub),
                "mean_w_clean_base": round(float(np.mean([r["w_clean_base"] for r in sub])), 5),
                "mean_w_clean_bud": round(float(np.mean([r["w_clean_bud"] for r in sub])), 5),
                "mean_w_atk_base": round(float(np.mean([r["w_atk_base"] for r in sub])), 5),
                "mean_w_atk_bud": round(float(np.mean([r["w_atk_bud"] for r in sub])), 5),
                "mean_w_atk_lucia": round(float(np.mean([r["w_atk_lucia"] for r in sub])), 5),
                "delta_bud_atk_minus_clean": round(float(np.mean([r["w_atk_bud"] - r["w_clean_bud"] for r in sub])), 5),
            }

    return OrderedDict({
        "stratified_results": results,
        "honest_cav_fpr_by_modality": fpr_agg,
        "honest_cav_weight_shift_under_attack": weight_shift,
    })


def main():
    parser = argparse.ArgumentParser(description="步骤 ⑤-2 & 多车队分层主实验（全基线 + 成对 Bootstrap 95% CI）")
    parser.add_argument("--data_dir", type=str, default=TRAIN_FULL19_DIR)
    parser.add_argument("--eval_split_json", type=str, default=DEFAULT_EVAL_SPLIT)
    parser.add_argument("--base_calib_json", type=str, default=DEFAULT_BASE_CALIB)
    parser.add_argument("--frozen_gating_json", type=str, default=DEFAULT_FROZEN_GATING)
    parser.add_argument("--hetshield_calib_json", type=str, default=DEFAULT_HETSHIELD_CALIB)
    parser.add_argument("--fleets", nargs="+", default=["mixed", "all_m1", "all_m2"], choices=["mixed", "all_m1", "all_m2"])
    parser.add_argument("--seqs", nargs="+", default=None, help="可选：仅运行指定父序列子集（多进程并行分片加速用）")
    parser.add_argument("--worker_only", action="store_true", help="仅写序列缓存 JSON，不聚合最终总报告")
    parser.add_argument("--max_cav", type=int, default=7)
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out_json", type=str, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    t0 = time.time()
    with open(args.eval_split_json, "r", encoding="utf-8") as f:
        eval_split = json.load(f)
    seq_order = list(eval_split["seq_order"])
    target_seqs = [s for s in seq_order if (args.seqs is None or s in set(args.seqs))]
    os.makedirs(args.cache_dir, exist_ok=True)

    # 若全部目标序列缓存已齐全且非 worker_only，可免加载 GPU 模型直接极速聚合
    all_cached = (not args.worker_only) and all(
        os.path.exists(os.path.join(args.cache_dir, f"{fm}__{s}.json"))
        for fm in args.fleets for s in seq_order
    )
    wrapper = gx = gy = auditor = hs_defs = robosac = None
    if not all_cached:
        wrapper = HEALBaseWrapper(device="cuda", use_cav=args.max_cav)
        gx, gy = build_bev_grid_xy_256(device="cuda", dtype=torch.float32)
        auditor = BUDFusionHEALAuditor(args.base_calib_json, gating_calib_path=args.frozen_gating_json)
        hs_defs = _load_hetshield_defenders(args.hetshield_calib_json, device="cuda")
        robosac = ROBOSACHEALDefender(box_matching_thresh=0.3, jaccard_thresh=0.3, step_budget=5, seed=BOOT_SEED)

    fleet_map = {"mixed": "mixed", "all_m1": "m1", "all_m2": "m2"}
    fleet_reports = OrderedDict()

    for fleet_mode in args.fleets:
        ds = sample_seqs = None
        seq_caches = {}

        for seq_name in (target_seqs if args.worker_only else seq_order):
            cpath = os.path.join(args.cache_dir, f"{fleet_mode}__{seq_name}.json")
            if os.path.exists(cpath):
                print(f"[断点复用] {fleet_mode} | {seq_name}")
                with open(cpath, "r", encoding="utf-8") as f:
                    seq_caches[seq_name] = json.load(f)
            else:
                if ds is None:
                    ds = wrapper.build_dataset(
                        data_dir=args.data_dir, use_cav=args.max_cav, visualize=True,
                        collab_modality=fleet_map[fleet_mode], max_cav=args.max_cav,
                    )
                    sample_seqs = wrapper.get_sample_sequences(ds)
                idxs = eval_split["sequences"][seq_name]["indices"]
                print(f"[开始评测] {fleet_mode} | {seq_name} ({len(idxs)} 帧)", flush=True)
                rec = _run_fleet_sequence(
                    wrapper, ds, sample_seqs, auditor, hs_defs, robosac,
                    fleet_mode, seq_name, idxs, gx, gy
                )
                _atomic_write_json(cpath, rec)
                seq_caches[seq_name] = rec

        if not args.worker_only:
            fleet_reports[fleet_mode] = _aggregate_fleet_caches(seq_order, seq_caches)

    if args.worker_only:
        print(f"\n[Worker完成] fleets={args.fleets} seqs={target_seqs} (耗时 {time.time() - t0:.1f}s)")
        return

    out_obj = OrderedDict({
        "meta": {
            "description": "步骤 ⑤-2 & 多车队分层主实验统一报告（全基线 + 混合/全m1/全m2 + 成对 Bootstrap 95% CI）",
            "fleets": args.fleets,
            "seq_order": seq_order,
            "bootstrap_iters": BOOT_N,
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "fleets": fleet_reports,
    })
    _atomic_write_json(args.out_json, out_obj)
    print(f"\n[完成] 多车队与基线主实验报告已保存至 {args.out_json} (总耗时 {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
