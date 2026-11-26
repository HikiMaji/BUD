# -*- coding: utf-8 -*-
"""
步骤 ⑤-4：定向局部攻击安全核心指标 ASR（Attack Success Rate，攻击成功率）全基线同台评测。

严格基于 Dev-320 全 8 条未见序列（绝不触碰 opv2v_test），显式区分车队规模与协作车数量分类：
1. 【2 车跨模态异构对】（1 本车 m1 + 1 协作者 m1/m2，全 320 帧）：
   - 攻击：Targeted Object Removal (1.0σ)、Targeted Object Fabrication (2.0σ)、Occluded-Region Fabrication (2.0σ)
   - 方法：No-Defense、Whole-CAV Drop、HetShield-Paper、HetShield-Tuned、LUCIA、BUD-Fusion (Frozen)
2. 【混合多车队】（3~6 车，即 2~5 辆协作车，显式分层统计：2 协作车 [3车] / >=3 协作车 [4~6车] / >=2 协作车总体）：
   - 攻击：Targeted Object Removal (1.0σ)、Targeted Object Fabrication (2.0σ)（随机 1 攻击者分报 m1/m2 + >=4 车双攻击者）
   - 方法：No-Defense、Whole-CAV Drop、HetShield-Paper、HetShield-Tuned、LUCIA、ROBOSAC、BUD-Fusion (Frozen)
3. 【标准 ASR 定义（越低越安全，2,000 次按父序列 Block-Bootstrap 95% CI）】：
   - Removal ASR：原本干净融合可检出（半径 3.0 m 内有预测框）的被攻击真实目标，在攻击+防御后发生漏检（半径 3.0 m 内无预测框）的比例；
   - Fabrication ASR：原本干净融合无框的空白伪造锚点，在攻击+防御后成功诱导出虚假预测框（半径 3.0 m 内出现预测框）的比例。
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, OrderedDict

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
    paired_block_bootstrap_rate_diff,
)
from src_v2_mm.heal_attacks import (
    generate_local_ghost_spoofing_heal,
    generate_local_occluded_ghost_heal,
    generate_local_removal_heal,
)
from src_v2_mm.heal_base_wrapper import HEALBaseWrapper
from src_v2_mm.hetshield_heal import HetShieldHEALDefender, build_hetshield_fuse_fn
from src_v2_mm.step4_1_v2_core import (
    DEFAULT_CLEAR_PTS_TAU,
    DEFAULT_LOS_OCC_TAU,
    GRID_RES,
    MIN_XY,
    PROTO_H,
    PROTO_W,
    TRAIN_FULL19_DIR,
    assign_dist5_vis3_indices_v2,
    build_bev_grid_xy_256,
    compute_ego_expected_pts_and_los_256,
    extract_frame_context_and_clean,
    load_ego_lidar_np,
)

DEFAULT_EVAL_SPLIT = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_3_eval320_split.json")
DEFAULT_BASE_CALIB = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_1_v2_frozen_calibrator.json")
DEFAULT_FROZEN_GATING = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_4_bud_frozen_calibrator.json")
DEFAULT_HS_CALIB = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step5_1_hetshield_heal_calib.json")
DEFAULT_CACHE_DIR = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step5_4_asr_seq_cache")
DEFAULT_OUT_JSON = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step5_4_targeted_asr_report.json")

MATCH_RADIUS_M = 3.0
BOOT_N = 2000
BOOT_SEED = 20261120

PAIR_METHODS = ["no_defense", "whole_cav_drop", "bud_frozen", "hetshield_paper", "hetshield_tuned", "lucia"]
MULTI_METHODS = ["no_defense", "whole_cav_drop", "bud_frozen", "hetshield_paper", "hetshield_tuned", "lucia", "robosac"]


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


def _grid_hw_to_xy(hc: int, wc: int) -> tuple:
    """将 256x256 BEV 网格索引 (iy, ix) 转为本车物理坐标 (cx, cy)（米）。"""
    cx = (float(wc) + 0.5) * GRID_RES + MIN_XY
    cy = (float(hc) + 0.5) * GRID_RES + MIN_XY
    return cx, cy


def _extract_pred_centers_xy(pred_boxes) -> np.ndarray:
    """从 HEAL 后处理检测框张量提取 2D 框中心坐标数组 (M, 2)。"""
    from opencood.utils import common_utils

    if pred_boxes is None or len(pred_boxes) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    det_np = common_utils.torch_tensor_to_numpy(pred_boxes)
    if det_np.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return det_np[:, :4, :2].mean(axis=1).astype(np.float64)


def _has_box_near(pred_xy: np.ndarray, cx: float, cy: float, radius_m: float = MATCH_RADIUS_M) -> bool:
    if pred_xy.shape[0] == 0:
        return False
    d2 = (pred_xy[:, 0] - cx) ** 2 + (pred_xy[:, 1] - cy) ** 2
    return bool(float(d2.min()) <= float(radius_m) ** 2)


def _load_hs_defenders(hs_calib_json: str, device: str = "cuda") -> dict:
    with open(hs_calib_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    e_th = float(cfg["meta"]["energy_thresh"])
    sd = torch.load(cfg["meta"]["shared_ckpt"], map_location="cpu")

    alpha_hs = float(cfg.get("paper_default_params", {}).get("alpha", 0.6))
    hs_paper = HetShieldHEALDefender(mode="shared", alpha=alpha_hs, tau_th=0.3, temperature=0.1, energy_thresh=e_th).to(device).eval()
    hs_paper.predictor.load_state_dict(sd)
    hs_tuned = HetShieldHEALDefender(
        mode="shared", alpha=alpha_hs, tau_th=float(cfg["fit_tuned_shared_tau_p95"]), temperature=0.1, energy_thresh=e_th
    ).to(device).eval()
    hs_tuned.predictor.load_state_dict(sd)
    return {"hetshield_paper": hs_paper, "hetshield_tuned": hs_tuned}


def _prime_hs_for_frame(hs_defs: dict, wrapper, prev_p: dict, seq_name: str, prev_idx: int):
    """每帧仅预热一次紧邻前一帧隐状态，供同帧全部攻击与方法复用。"""
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

    ac = wrapper.model.pyramid_backbone.align_corners
    for hd in hs_defs.values():
        hd.reset_sequence(seq_name)
    if prev_p is None:
        return
    pf = prev_p["heter_feature_2d"]
    p_amat = prev_p["affine_matrix"]
    p_vm = prev_p["cam_valid_mask"]
    p_mods = list(prev_p["agent_modality_list"])
    n_p = int(pf.shape[0])
    items = []
    for ci in range(1, n_p):
        t_c = p_amat[0, 0, ci : ci + 1]
        fc = warp_affine_simple(pf[ci : ci + 1], t_c, (PROTO_H, PROTO_W), align_corners=ac)
        vc = warp_affine_simple(p_vm[ci : ci + 1].float(), t_c, (PROTO_H, PROTO_W), align_corners=ac)
        items.append({"cav_id": f"{seq_name}_c{ci}_{p_mods[ci]}", "mod": p_mods[ci], "feat_warped": fc, "valid_ego": (vc > 0.5).float()})
    for hd in hs_defs.values():
        hd.step_frame(seq_name, prev_idx, pf[0:1], items, update_state=True)


def _score_hs_frame(hd, wrapper, cur_p: dict, cur_feat: torch.Tensor, seq_name: str, idx: int) -> list:
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

    ac = wrapper.model.pyramid_backbone.align_corners
    amat = cur_p["affine_matrix"]
    vm = cur_p["cam_valid_mask"]
    mods = list(cur_p["agent_modality_list"])
    n_tot = int(cur_feat.shape[0])
    items = []
    for ci in range(1, n_tot):
        t_c = amat[0, 0, ci : ci + 1]
        fc = warp_affine_simple(cur_feat[ci : ci + 1], t_c, (PROTO_H, PROTO_W), align_corners=ac)
        vc = warp_affine_simple(vm[ci : ci + 1].float(), t_c, (PROTO_H, PROTO_W), align_corners=ac)
        items.append({"cav_id": f"{seq_name}_c{ci}_{mods[ci]}", "mod": mods[ci], "feat_warped": fc, "valid_ego": (vc > 0.5).float()})
    return hd.step_frame(seq_name, idx, cur_feat[0:1], items, update_state=False)


def _select_mixed_attack_roles(mods_l: list, frame_idx: int) -> list:
    """与 step5_2 保持完全一致的固定随机种子攻击者角色分配。"""
    n_tot = len(mods_l)
    m1_c = [i for i in range(1, n_tot) if mods_l[i] == "m1"]
    m2_c = [i for i in range(1, n_tot) if mods_l[i] == "m2"]
    rng = random.Random(20261120 + int(frame_idx) * 31 + n_tot * 7)
    out = []
    if m1_c:
        out.append(("atk_m1", [rng.choice(m1_c)]))
    if m2_c:
        out.append(("atk_m2", [rng.choice(m2_c)]))
    if n_tot >= 4:
        all_c = list(range(1, n_tot))
        out.append(("two_attackers", sorted(rng.sample(all_c, 2))))
    return out


def _eval_methods_and_asr(
    wrapper,
    ds,
    b_post: dict,
    cur_p: dict,
    f_cur: torch.Tensor,
    p_ego_cls: torch.Tensor,
    ego_pb_np,
    ego_tuple,
    auditor: BUDFusionHEALAuditor,
    hs_defs: dict,
    robosac,
    per_cav_didx: dict,
    per_cav_vidx: dict,
    fg_prob_tau: float,
    seq_name: str,
    idx: int,
    valid_points_xy: list,
    mode: str,
    enable_robosac: bool,
    assumed_attackers: int = 1,
) -> dict:
    """在指定攻击特征 f_cur 上运行所有方法并统计 ASR 命中次数 (succ, total)。"""
    rlen = cur_p["record_len"]
    amat = cur_p["affine_matrix"]
    mods_l = list(cur_p["agent_modality_list"])
    vmask = cur_p["cam_valid_mask"]
    n_total = int(f_cur.shape[0])
    n_tot_pts = len(valid_points_xy)

    def _count_succ(pred_boxes) -> int:
        pxy = _extract_pred_centers_xy(pred_boxes)
        if mode == "removal":
            # 目标移除成功：被攻击真车中心 3.0 m 内不再有预测框
            return sum(1 for (cx, cy) in valid_points_xy if not _has_box_near(pxy, cx, cy))
        # 目标伪造成功：空白伪造锚点中心 3.0 m 内成功诱导出虚假预测框
        return sum(1 for (cx, cy) in valid_points_xy if _has_box_near(pxy, cx, cy))

    res = {}
    # 1) No-Defense
    with torch.no_grad():
        out_nd = wrapper.forward_from_protocol(f_cur, rlen, amat, mods_l)
        nd_pb, _, _ = wrapper.decode_boxes(b_post, out_nd, ds)
    res["no_defense"] = (_count_succ(nd_pb), n_tot_pts)

    # 2) BUD-Fusion + Whole-CAV Drop
    reject_flags, bud_gates = [], []
    for ci in range(1, n_total):
        mod_ci = mods_l[ci]
        dev_ci, d_drop_ci, d_add_ci, cov_ci, p_pair_ci = extract_single_cav_dev_and_signed(
            wrapper, f_cur, amat, mods_l, vmask, p_ego_cls, cav_idx=ci, k_tol=5, return_pfused=True
        )
        fg_m = F.max_pool2d(p_pair_ci, kernel_size=3, stride=1, padding=1)[0, 0] > fg_prob_tau
        ar_ci = auditor.audit_dev_maps(
            dev_ci, cov_ci, per_cav_didx[ci], per_cav_vidx[ci], mod_ci,
            build_gate=True, d_drop_tol=d_drop_ci, d_add_tol=d_add_ci,
            enable_whole_fallback=True, collab_fg_mask=fg_m,
        )
        reject_flags.append(bool(ar_ci["score_t4"] > 1.0))
        bud_gates.append(ar_ci["gate_L0"])

    with torch.no_grad():
        if all(reject_flags):
            out_wd = wrapper.forward_single_agent(f_cur[0:1])
        elif not any(reject_flags):
            out_wd = out_nd
        else:
            out_wd = wrapper.forward_from_protocol(
                f_cur, rlen, amat, mods_l, custom_fuse_fn=build_whole_cav_reject_fuse_fn(reject_flags, device=str(wrapper.device_obj))
            )
        wd_pb, _, _ = wrapper.decode_boxes(b_post, out_wd, ds)
    res["whole_cav_drop"] = (_count_succ(wd_pb), n_tot_pts)

    gate_stack = torch.cat(bud_gates, dim=0)
    with torch.no_grad():
        out_bud = wrapper.forward_from_protocol(f_cur, rlen, amat, mods_l, custom_fuse_fn=build_gated_fuse_fn(gate_stack))
        bud_pb, _, _ = wrapper.decode_boxes(b_post, out_bud, ds)
    res["bud_frozen"] = (_count_succ(bud_pb), n_tot_pts)

    # 3) LUCIA
    with torch.no_grad():
        out_luc = wrapper.forward_from_protocol(f_cur, rlen, amat, mods_l, custom_fuse_fn=build_lucia_fuse_fn(compression_ratio=32))
        luc_pb, _, _ = wrapper.decode_boxes(b_post, out_luc, ds)
    res["lucia"] = (_count_succ(luc_pb), n_tot_pts)

    # 4) HetShield (Paper & Tuned)
    for hs_name in ("hetshield_paper", "hetshield_tuned"):
        hd = hs_defs[hs_name]
        hs_res = _score_hs_frame(hd, wrapper, cur_p, f_cur, seq_name, idx)
        g_list = [float(x["g_k"]) for x in hs_res]
        with torch.no_grad():
            out_hs = wrapper.forward_from_protocol(
                f_cur, rlen, amat, mods_l, custom_fuse_fn=build_hetshield_fuse_fn(g_list, device=str(wrapper.device_obj))
            )
            hs_pb, _, _ = wrapper.decode_boxes(b_post, out_hs, ds)
        res[hs_name] = (_count_succ(hs_pb), n_tot_pts)

    # 5) ROBOSAC (仅 >=3 车混合车队启用)
    if enable_robosac and n_total >= 3:
        ego_pb, ego_ps, ego_gb = ego_tuple
        rob_res = robosac.run_frame(
            wrapper, f_cur, rlen, amat, mods_l, b_post, ds,
            ego_boxes_np=ego_pb_np, ego_pb=ego_pb, ego_ps=ego_ps, ego_gb=ego_gb,
            frame_seed=idx, assumed_attackers=max(1, int(assumed_attackers)),
        )
        res["robosac"] = (_count_succ(rob_res["pb"]), n_tot_pts)

    return res


def _run_one_sequence(wrapper, ds_m1, ds_m2, ds_mix, sample_seqs, auditor, hs_defs, robosac, seq_name: str, idxs: list, gx, gy) -> dict:
    from opencood.tools import train_utils
    from opencood.utils import common_utils

    t0 = time.time()
    vis_v2 = (auditor.gating_calib or {}).get("visibility_v2", None)
    fg_prob_tau = float((auditor.gating_calib or {}).get("gating_params", {}).get("fg_prob_tau", 0.20))

    pair_acc = {}   # key: f"{mod}|{atk}" -> {meth: [succ, tot]}
    multi_acc = {}  # key: f"{tier}|{role}|{atk}" -> {meth: [succ, tot]}
    cav_count_hist = Counter()

    def _add(acc_dict: dict, key: str, meth_res: dict):
        if key not in acc_dict:
            acc_dict[key] = {}
        for m, (s, t) in meth_res.items():
            if m not in acc_dict[key]:
                acc_dict[key][m] = [0, 0]
            acc_dict[key][m][0] += int(s)
            acc_dict[key][m][1] += int(t)

    # === Part 1: 2 车跨模态异构对（1 本车 m1 + 1 协作者 m1/m2） ===
    for fi, idx in enumerate(idxs):
        c_tau = float(vis_v2["clear_pts_tau"]) if vis_v2 is not None else DEFAULT_CLEAR_PTS_TAU
        l_tau = float(vis_v2.get("los_occ_tau", DEFAULT_LOS_OCC_TAU)) if vis_v2 is not None else DEFAULT_LOS_OCC_TAU
        b = extract_frame_context_and_clean(wrapper, ds_m1, ds_m2, idx, gx, gy, clear_pts_tau=c_tau, los_occ_tau=l_tau)
        raw_b1 = train_utils.to_device(ds_m1.collate_batch_test([ds_m1[idx]]), "cpu")
        b_post = train_utils.to_device(_slim_batch_for_postprocess(raw_b1), wrapper.device_obj)
        d_2d, v_2d = b["d_idx"], b["v_idx"]
        los_f, exp_pts = b["los_factor"], b["expected_pts"]
        occ_mask_4d = (los_f < l_tau).float().view(1, 1, PROTO_H, PROTO_W)

        for mod, p_dict, ds_ref in (("m1", b["p1"], ds_m1), ("m2", b["p2"], ds_m2)):
            feat = p_dict["heter_feature_2d"]
            rlen = p_dict["record_len"]
            amat = p_dict["affine_matrix"]
            mods_l = list(p_dict["agent_modality_list"])
            vmask = p_dict["cam_valid_mask"]
            # 主实验威胁模型：单协作车 m1 使用自身完整激光雷达范围，不限制攻击区
            preg = None

            # 干净融合基准框（用于验证被攻击真车原本可检出、伪造空地原本无框）
            with torch.no_grad():
                out_cln = wrapper.forward_from_protocol(feat, rlen, amat, mods_l)
                cln_pb, _, _ = wrapper.decode_boxes(b_post, out_cln, ds_ref)
            cln_xy = _extract_pred_centers_xy(cln_pb)

            # 预热 HetShield 前一帧
            prev_p = None
            if idx > 0 and sample_seqs[idx - 1] == seq_name:
                raw_prev = train_utils.to_device(ds_ref.collate_batch_test([ds_ref[idx - 1]]), wrapper.device_obj)
                prev_p = wrapper.extract_protocol_features(raw_prev["ego"])
                del raw_prev
            _prime_hs_for_frame(hs_defs, wrapper, prev_p, seq_name, idx - 1)

            # 3 种定向局部攻击
            f_rem, st_rem = generate_local_removal_heal(
                wrapper, feat, rlen, amat, mods_l, vmask, attacker_idx=1, num_targets=2, patch_radius=5, sigma_mult=1.0, steps=10, perturb_region_mask=preg
            )
            f_fab, st_fab = generate_local_ghost_spoofing_heal(
                wrapper, feat, rlen, amat, mods_l, vmask, attacker_idx=1, num_spoofs=2, patch_radius=5, sigma_mult=2.0, steps=10, frame_seed=idx, perturb_region_mask=preg
            )
            f_occ, st_occ = generate_local_occluded_ghost_heal(
                wrapper, feat, rlen, amat, mods_l, vmask, occluded_mask_ego=occ_mask_4d,
                attacker_idx=1, num_spoofs=2, patch_radius=5, sigma_mult=2.0, steps=12, frame_seed=idx, perturb_region_mask=preg
            )

            # 校验有效目标中心坐标
            rem_pts = []
            for hc, wc in st_rem.get("target_centers_hw", []):
                cx, cy = _grid_hw_to_xy(hc, wc)
                if _has_box_near(cln_xy, cx, cy):
                    rem_pts.append((cx, cy))
            fab_pts = []
            for anc in st_fab.get("anchors_hwr", []):
                cx, cy = _grid_hw_to_xy(anc[0], anc[1])
                if not _has_box_near(cln_xy, cx, cy):
                    fab_pts.append((cx, cy))
            occ_pts = []
            for anc in st_occ.get("anchors_hwr", []):
                cx, cy = _grid_hw_to_xy(anc[0], anc[1])
                if not _has_box_near(cln_xy, cx, cy):
                    occ_pts.append((cx, cy))

            for atk_tag, f_adv, pts_xy, mode_s in (
                ("object_removal", f_rem, rem_pts, "removal"),
                ("object_fabrication", f_fab, fab_pts, "fabrication"),
                ("occluded_fabrication", f_occ, occ_pts, "fabrication"),
            ):
                if not pts_xy:
                    continue
                mres = _eval_methods_and_asr(
                    wrapper, ds_ref, b_post, p_dict, f_adv, b["p_ego_cls"], None, None,
                    auditor, hs_defs, robosac, {1: d_2d}, {1: v_2d}, fg_prob_tau,
                    seq_name, idx, pts_xy, mode_s, enable_robosac=False, assumed_attackers=1,
                )
                _add(pair_acc, f"{mod}|{atk_tag}", mres)

        # === Part 2: 混合多车队（3~6 车，即 2~5 辆协作车） ===
        raw_m = train_utils.to_device(ds_mix.collate_batch_test([ds_mix[idx]]), wrapper.device_obj)
        b_post_m = train_utils.to_device(_slim_batch_for_postprocess(raw_m), wrapper.device_obj)
        p_m = wrapper.extract_protocol_features(raw_m["ego"])
        feat_m = p_m["heter_feature_2d"]
        n_tot = int(feat_m.shape[0])
        cav_count_hist[n_tot] += 1
        if n_tot >= 3:
            n_collab = n_tot - 1
            tier = "collab_2" if n_collab == 2 else "collab_ge3"
            rlen_m = p_m["record_len"]
            amat_m = p_m["affine_matrix"]
            mods_m = list(p_m["agent_modality_list"])
            vmask_m = p_m["cam_valid_mask"]

            per_d, per_v = {}, {}
            for ci in range(1, n_tot):
                ptm_ci = raw_m["ego"]["pairwise_t_matrix"][0, ci, 0].detach().cpu().numpy()
                dist_ci = torch.sqrt((gx - float(ptm_ci[0, 3])) ** 2 + (gy - float(ptm_ci[1, 3])) ** 2).view(1, 1, PROTO_H, PROTO_W)
                di, vi = assign_dist5_vis3_indices_v2(dist_ci, los_f, exp_pts, clear_pts_tau=c_tau, los_occ_tau=l_tau)
                per_d[ci], per_v[ci] = di[0, 0], vi[0, 0]

            with torch.no_grad():
                o_ego_m = wrapper.forward_single_agent(feat_m[0:1])
                p_ego_m = torch.sigmoid(o_ego_m["cls_preds"]).max(dim=1, keepdim=True)[0]
                epb_m, eps_m, egb_m = wrapper.decode_boxes(b_post_m, o_ego_m, ds_mix)
                epb_m_np = common_utils.torch_tensor_to_numpy(epb_m) if (epb_m is not None and len(epb_m) > 0) else None
                o_cln_m = wrapper.forward_from_protocol(feat_m, rlen_m, amat_m, mods_m)
                cln_pb_m, _, _ = wrapper.decode_boxes(b_post_m, o_cln_m, ds_mix)
            cln_xy_m = _extract_pred_centers_xy(cln_pb_m)

            prev_pm = None
            if idx > 0 and sample_seqs[idx - 1] == seq_name:
                raw_pm = train_utils.to_device(ds_mix.collate_batch_test([ds_mix[idx - 1]]), wrapper.device_obj)
                prev_pm = wrapper.extract_protocol_features(raw_pm["ego"])
                del raw_pm
            _prime_hs_for_frame(hs_defs, wrapper, prev_pm, seq_name, idx - 1)

            for role_tag, atk_idxs in _select_mixed_attack_roles(mods_m, idx):
                # 逐攻击者注入定向移除与虚假伪造并收集全部有效目标中心
                f_rem_m = feat_m.clone()
                rem_pts_m = []
                for step_k, a_idx in enumerate(atk_idxs):
                    f_rem_m, st_rm = generate_local_removal_heal(
                        wrapper, f_rem_m, rlen_m, amat_m, mods_m, vmask_m,
                        attacker_idx=a_idx, num_targets=2, patch_radius=5, sigma_mult=1.0, steps=10,
                    )
                    for hc, wc in st_rm.get("target_centers_hw", []):
                        cx, cy = _grid_hw_to_xy(hc, wc)
                        if _has_box_near(cln_xy_m, cx, cy):
                            rem_pts_m.append((cx, cy))

                f_fab_m = feat_m.clone()
                fab_pts_m = []
                for step_k, a_idx in enumerate(atk_idxs):
                    f_fab_m, st_fm = generate_local_ghost_spoofing_heal(
                        wrapper, f_fab_m, rlen_m, amat_m, mods_m, vmask_m,
                        attacker_idx=a_idx, num_spoofs=2, patch_radius=5, sigma_mult=2.0, steps=10,
                        frame_seed=idx + step_k * 1009,
                    )
                    for anc in st_fm.get("anchors_hwr", []):
                        cx, cy = _grid_hw_to_xy(anc[0], anc[1])
                        if not _has_box_near(cln_xy_m, cx, cy):
                            fab_pts_m.append((cx, cy))

                for atk_tag, f_adv_m, pts_xy_m, mode_s in (
                    ("object_removal", f_rem_m, rem_pts_m, "removal"),
                    ("object_fabrication", f_fab_m, fab_pts_m, "fabrication"),
                ):
                    if not pts_xy_m:
                        continue
                    mres_m = _eval_methods_and_asr(
                        wrapper, ds_mix, b_post_m, p_m, f_adv_m, p_ego_m, epb_m_np, (epb_m, eps_m, egb_m),
                        auditor, hs_defs, robosac, per_d, per_v, fg_prob_tau,
                        seq_name, idx, pts_xy_m, mode_s, enable_robosac=True, assumed_attackers=len(atk_idxs),
                    )
                    _add(multi_acc, f"{tier}|{role_tag}|{atk_tag}", mres_m)
                    if role_tag in ("atk_m1", "atk_m2"):
                        _add(multi_acc, f"ge2_all|single_attacker|{atk_tag}", mres_m)
                        _add(multi_acc, f"ge2_all|{role_tag}|{atk_tag}", mres_m)

        if (fi + 1) % 10 == 0 or (fi + 1) == len(idxs):
            print(f"  [ASR | {seq_name}] 进度 {fi + 1}/{len(idxs)} ({time.time() - t0:.1f}s)", flush=True)

    return {
        "seq_name": seq_name,
        "num_frames": len(idxs),
        "cav_count_hist": {str(k): int(v) for k, v in sorted(cav_count_hist.items())},
        "pair_acc": pair_acc,
        "multi_acc": multi_acc,
    }


def _aggregate_bucket(seq_order: list, seq_caches: dict, section_key: str, methods_list: list) -> OrderedDict:
    all_keys = sorted({k for s in seq_order for k in seq_caches[s][section_key].keys()})
    out = OrderedDict()
    for key in all_keys:
        rel_seqs = [s for s in seq_order if key in seq_caches[s][section_key]]
        boot_idx = build_seq_bootstrap_indices(rel_seqs, n_boot=BOOT_N, seed=BOOT_SEED)
        meth_out = OrderedDict()
        paired_vs_bud = OrderedDict()

        # 取每个序列的总有效目标数
        tots = {s: int(seq_caches[s][section_key][key]["bud_frozen"][1]) for s in rel_seqs}
        bud_h = {s: int(seq_caches[s][section_key][key]["bud_frozen"][0]) for s in rel_seqs}

        for m in methods_list:
            if not all(m in seq_caches[s][section_key][key] for s in rel_seqs):
                continue
            m_h = {s: int(seq_caches[s][section_key][key][m][0]) for s in rel_seqs}
            diff_r = paired_block_bootstrap_rate_diff(rel_seqs, bud_h, m_h, tots, boot_indices=boot_idx)
            if "bud_frozen" not in meth_out:
                meth_out["bud_frozen"] = {
                    "asr": diff_r["val_a"],
                    "ci95": diff_r["a_ci95"],
                    "succ": sum(bud_h.values()),
                    "total_targets": diff_r["total"],
                }
            if m != "bud_frozen":
                meth_out[m] = {
                    "asr": diff_r["val_b"],
                    "ci95": diff_r["b_ci95"],
                    "succ": sum(m_h.values()),
                    "total_targets": diff_r["total"],
                }
                paired_vs_bud[f"bud_minus_{m}"] = {
                    "diff": diff_r["diff"],
                    "diff_ci95": diff_r["diff_ci95"],
                    "significant_lower_asr": bool(diff_r["diff_ci95"][1] < 0.0),
                }

        out[key] = {
            "num_sequences": len(rel_seqs),
            "total_valid_targets": sum(tots.values()),
            "methods": meth_out,
            "paired_asr_diff_vs_bud": paired_vs_bud,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="步骤 ⑤-4 定向攻击成功率 ASR 全基线同台评测")
    parser.add_argument("--data_dir", type=str, default=TRAIN_FULL19_DIR)
    parser.add_argument("--eval_split_json", type=str, default=DEFAULT_EVAL_SPLIT)
    parser.add_argument("--base_calib_json", type=str, default=DEFAULT_BASE_CALIB)
    parser.add_argument("--frozen_gating_json", type=str, default=DEFAULT_FROZEN_GATING)
    parser.add_argument("--hetshield_calib_json", type=str, default=DEFAULT_HS_CALIB)
    parser.add_argument("--seqs", nargs="+", default=None)
    parser.add_argument("--worker_only", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out_json", type=str, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    t0 = time.time()
    with open(args.eval_split_json, "r", encoding="utf-8") as f:
        eval_split = json.load(f)
    seq_order = list(eval_split["seq_order"])
    target_seqs = [s for s in seq_order if (args.seqs is None or s in set(args.seqs))]
    os.makedirs(args.cache_dir, exist_ok=True)

    all_cached = (not args.worker_only) and all(
        os.path.exists(os.path.join(args.cache_dir, f"{s}.json")) for s in seq_order
    )
    wrapper = ds_m1 = ds_m2 = ds_mix = sample_seqs = gx = gy = auditor = hs_defs = robosac = None
    if not all_cached:
        wrapper = HEALBaseWrapper(device="cuda", use_cav=7)
        ds_m1 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m1")
        ds_m2 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m2")
        ds_mix = wrapper.build_dataset(data_dir=args.data_dir, use_cav=7, visualize=True, collab_modality="mixed", max_cav=7)
        sample_seqs = wrapper.get_sample_sequences(ds_m2)
        gx, gy = build_bev_grid_xy_256(device="cuda", dtype=torch.float32)
        auditor = BUDFusionHEALAuditor(args.base_calib_json, gating_calib_path=args.frozen_gating_json)
        hs_defs = _load_hs_defenders(args.hetshield_calib_json, device="cuda")
        robosac = ROBOSACHEALDefender(box_matching_thresh=0.3, jaccard_thresh=0.3, step_budget=5, seed=BOOT_SEED)

    seq_caches = {}
    for seq_name in (target_seqs if args.worker_only else seq_order):
        cpath = os.path.join(args.cache_dir, f"{seq_name}.json")
        if os.path.exists(cpath):
            print(f"[断点复用] Step 5-4 ASR | {seq_name}")
            with open(cpath, "r", encoding="utf-8") as f:
                seq_caches[seq_name] = json.load(f)
        else:
            idxs = eval_split["sequences"][seq_name]["indices"]
            print(f"[开始评测] Step 5-4 ASR | {seq_name} ({len(idxs)} 帧)", flush=True)
            rec = _run_one_sequence(wrapper, ds_m1, ds_m2, ds_mix, sample_seqs, auditor, hs_defs, robosac, seq_name, idxs, gx, gy)
            _atomic_write_json(cpath, rec)
            seq_caches[seq_name] = rec

    if args.worker_only:
        print(f"\n[Worker完成] Step 5-4 ASR seqs={target_seqs} (耗时 {time.time() - t0:.1f}s)")
        return

    total_cav_hist = Counter()
    for s in seq_order:
        for k, v in seq_caches[s]["cav_count_hist"].items():
            total_cav_hist[int(k)] += int(v)

    fleet_size_summary = OrderedDict({
        "total_frames": int(eval_split["total_frames"]),
        "by_total_vehicles": {f"{k}_vehicles ({k-1}_collab)": int(v) for k, v in sorted(total_cav_hist.items())},
        "by_collab_tier": {
            "1_collab (2_vehicles)": int(total_cav_hist.get(2, 0)),
            "2_collab (3_vehicles)": int(total_cav_hist.get(3, 0)),
            "ge3_collab (4_to_6_vehicles)": int(sum(v for k, v in total_cav_hist.items() if k >= 4)),
        },
    })

    report = OrderedDict({
        "meta": {
            "description": "步骤 ⑤-4 定向局部攻击成功率 ASR 全基线同台评测报告（Dev-320，匹配半径 3.0m，2000 次 Block-Bootstrap 95% CI）",
            "match_radius_m": MATCH_RADIUS_M,
            "bootstrap_iters": BOOT_N,
            "fleet_size_summary": fleet_size_summary,
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "two_vehicle_hetero_pair_asr": _aggregate_bucket(seq_order, seq_caches, "pair_acc", PAIR_METHODS),
        "mixed_multicav_fleet_asr": _aggregate_bucket(seq_order, seq_caches, "multi_acc", MULTI_METHODS),
    })
    _atomic_write_json(args.out_json, report)
    print(f"\n[完成] 定向攻击 ASR 报告已保存至 {args.out_json} (总耗时 {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
