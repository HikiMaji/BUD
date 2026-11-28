# -*- coding: utf-8 -*-
"""
M6（科学执行顺序 ②）& M8（科学执行顺序 ④）统一评测、定方法消融与基线诊断脚本。

核心特性：
1. 按序列断点落盘缓存（JSON），支持中断秒续跑，CPU/GPU 内存恒定 < 4 GB；
2. 同帧仅生成 1 次 5 种标准攻击扰动，所有对比变体 100% 共享逐比特一致的攻击特征；
3. --stage m6（开发组快速子集 dev_fast = 400 帧，先定方法）：
   - 单变量严格前后对比（附 2,000 次序列块 Bootstrap 95% CI）：
     * A1 前（旧不一致可见性标定表）vs 后（新统一物理可见性标定表）
     * B1 前（area 下采样）vs 后（min_pool 下采样）
     * B2 前（v0 推理 + bud_calib_v0.json）vs 后（v1 推理 + bud_calib_v1.json）
     * B3 前（enable_b3=False 全还本车）vs 后（enable_b3=True 其他可信车重分配，含多车 >=4 干净与受攻 AP）
   - B4 扫描 γ(Occluded) ∈ {0.0, 0.1, 0.2, 0.3} 权衡曲线；
   - F-P0-1 可见性掩码基线摸底结果；
   - HetShield 诊断结果（诚实 g < 0.5 比例、时间项 τ_k 与空间项 σ_k 分布、m1/m2 干净盲区 AP）；
   - 按客观验收标准输出每项去留决定与透明数值依据至 results_v3/m6_method_decisions.json。
4. --stage m8（开发组全量帧 dev_full = 1,113 帧，冻结前最终确认）：
   - 同台对比定稿调参后的 BUD-Fusion、可见性掩码基线、HetShield（论文默认 0.3 / Tuned / PerSensor）、LUCIA（Tuned），
     输出 results_v3/dev_full_final_confirmation.json（绝不触碰 test 测试集）。
"""

import argparse
import json
import math
import os
import sys
import time
import traceback
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PROJ_ROOT = "/root/autodl-tmp/proj_dir2_hetero_trust"
if os.path.exists(PROJ_ROOT) and PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)
LOCAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if LOCAL_ROOT not in sys.path:
    sys.path.insert(0, LOCAL_ROOT)

from scripts_v2_mm.eval_standard_suite_v3 import (
    ATTACKS_5,
    B4_GAMMA_CANDIDATES,
    BOOT_N,
    BOOT_SEED,
    _atomic_write_json,
    _decode_np,
    _extract_pair_dev,
    _find_blind_gts,
    _grid_hw_to_xy,
    _match_standard_ap,
    _resolve_eval_subset,
    _slim_batch_for_postprocess,
    build_visibility_mask_gate,
)
from src_v2_mm.baselines_robosac_lucia_heal import build_lucia_fuse_fn
from src_v2_mm.bud_fusion_heal import (
    BUDFusionHEALAuditor,
    build_fg_mask,
    build_gated_fuse_fn,
    extract_single_cav_dev_and_signed,
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
    generate_scaled_pgd_removal_heal,
)
from src_v2_mm.heal_base_wrapper import HEALBaseWrapper
from src_v2_mm.hetshield_heal import HetShieldHEALDefender, forward_hetshield_protocol_mix
from src_v2_mm.sprint_metrics_v3 import (
    VIS_NAMES_3,
    compute_honest_suppression_stats,
    evaluate_frame_asr_strict,
    match_blind_spot_ap_ignore_nonblind,
)
from src_v2_mm.step4_1_v2_core import (
    GRID_RES,
    MIN_XY,
    OPV2V_TRAIN_DIR,
    PROTO_H,
    PROTO_W,
    V3_BASELINES_TUNED_JSON,
    V3_BUD_CALIB_V0_JSON,
    V3_BUD_CALIB_V1_JSON,
    V3_BUD_TUNED_JSON,
    V3_HETSHIELD_CALIB_JSON,
    V3_RESULTS_DIR,
    V3_SPLITS_JSON,
    assign_dist5_vis3_indices_v2,
    build_bev_grid_xy_256,
    extract_frame_context_and_clean,
    require_visibility_v2_taus,
)


def _dist_summary_1d(vals: list) -> dict:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 5),
        "std": round(float(arr.std()), 5),
        "p50": round(float(np.percentile(arr, 50.0)), 5),
        "p90": round(float(np.percentile(arr, 90.0)), 5),
        "p95": round(float(np.percentile(arr, 95.0)), 5),
        "p99": round(float(np.percentile(arr, 99.0)), 5),
    }


def _warped_pair_from_proto(wrapper: HEALBaseWrapper, p_prev: dict) -> tuple:
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

    ac = wrapper.model.pyramid_backbone.align_corners
    fp = p_prev["heter_feature_2d"]
    tmat_p = p_prev["affine_matrix"][0, 0, 1:2]
    fc_p = warp_affine_simple(fp[1:2], tmat_p, (PROTO_H, PROTO_W), align_corners=ac)
    ve_p = (warp_affine_simple(p_prev["cam_valid_mask"][1:2].float(), tmat_p, (PROTO_H, PROTO_W), align_corners=ac) > 0.5).float()
    return (fp[0:1].detach(), fc_p.detach(), ve_p.detach())


def _extract_prev_pair_warped(
    wrapper: HEALBaseWrapper,
    ds_ref,
    sample_seqs: List[str],
    seq_name: str,
    idx: int,
    pre_prev_cpu: Optional[dict] = None,
    pre_prev_proto: Optional[dict] = None,
):
    """单帧同模态仅提取一次紧邻前一帧 idx-1 的对齐协议特征，优先复用已提取协议特征或预取 CPU batch。"""
    from opencood.tools import train_utils

    if not (idx > 0 and sample_seqs[idx - 1] == seq_name):
        return None
    if pre_prev_proto is not None:
        return _warped_pair_from_proto(wrapper, pre_prev_proto)
    raw_prev_cpu = pre_prev_cpu if pre_prev_cpu is not None else ds_ref.collate_batch_test([ds_ref[idx - 1]])
    raw_prev = train_utils.to_device(raw_prev_cpu, wrapper.device_obj)
    p_prev = wrapper.extract_protocol_features(raw_prev["ego"])
    return _warped_pair_from_proto(wrapper, p_prev)


def _iter_seq_prefetched_batches(
    indices: List[int],
    sample_seqs: List[str],
    seq_name: str,
    ds_m1,
    ds_m2,
    ds_mix=None,
    need_prev: bool = True,
):
    """
    通过多进程 DataLoader 并行后台解码当前序列所需的目标帧与紧邻前一帧 idx-1（滑动窗口最多驻留 2 帧 CPU 张量，< 120 MB），
    彻底消除主线程每帧单核串行解码数十张 1920x1080 PNG 导致的 GPU 等待，输出逐比特数学恒等。
    """
    from torch.utils.data import DataLoader, Subset

    prev_needed = {idx - 1 for idx in indices if need_prev and idx > 0 and sample_seqs[idx - 1] == seq_name}
    union_idxs = sorted(set(indices) | prev_needed)
    nw = 2 if len(union_idxs) >= 4 else 0
    dl1 = iter(DataLoader(Subset(ds_m1, union_idxs), batch_size=1, num_workers=nw, shuffle=False, collate_fn=ds_m1.collate_batch_test))
    dl2 = iter(DataLoader(Subset(ds_m2, union_idxs), batch_size=1, num_workers=nw, shuffle=False, collate_fn=ds_m2.collate_batch_test))
    dlm = iter(DataLoader(Subset(ds_mix, union_idxs), batch_size=1, num_workers=nw, shuffle=False, collate_fn=ds_mix.collate_batch_test)) if ds_mix is not None else None
    u_iter = iter(union_idxs)
    win = {}

    for idx in indices:
        want_prev = idx - 1 if (need_prev and idx > 0 and sample_seqs[idx - 1] == seq_name) else None
        while idx not in win:
            u_i = next(u_iter)
            win[u_i] = (next(dl1), next(dl2), next(dlm) if dlm is not None else None)
        b1_c, b2_c, bm_c = win[idx]
        if want_prev is not None and want_prev in win:
            pb1, pb2, pbm = win[want_prev]
        else:
            pb1, pb2, pbm = None, None, None
        for k in [k for k in list(win.keys()) if k < idx]:
            win.pop(k, None)
        yield idx, b1_c, b2_c, bm_c, pb1, pb2, pbm
    win.clear()


def _step_hetshield_one_frame_with_prev(
    defender: HetShieldHEALDefender,
    wrapper: HEALBaseWrapper,
    ds_ref,
    sample_seqs: List[str],
    seq_name: str,
    idx: int,
    mod: str,
    feat_cur: torch.Tensor,
    p_cur: dict,
    prev_warped_tuple: Optional[tuple] = None,
    preload_attempted: bool = False,
) -> dict:
    """严格单步预热同协议评估 HetShield：若 idx-1 属于同一父序列则预热一步，否则首帧历史长=1。"""
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

    ac = wrapper.model.pyramid_backbone.align_corners
    cid = f"{seq_name}_{mod}_c1"
    defender.reset_sequence(seq_name)

    if not preload_attempted:
        prev_warped_tuple = _extract_prev_pair_warped(wrapper, ds_ref, sample_seqs, seq_name, idx)

    if prev_warped_tuple is not None:
        fp0, fc_p, ve_p = prev_warped_tuple
        defender.step_frame(seq_name, idx - 1, fp0, [{"cav_id": cid, "mod": mod, "feat_warped": fc_p, "valid_ego": ve_p}], update_state=True)

    tmat_c = p_cur["affine_matrix"][0, 0, 1:2]
    fc_c = warp_affine_simple(feat_cur[1:2], tmat_c, (PROTO_H, PROTO_W), align_corners=ac)
    ve_c = (warp_affine_simple(p_cur["cam_valid_mask"][1:2].float(), tmat_c, (PROTO_H, PROTO_W), align_corners=ac) > 0.5).float()
    return defender.step_frame(seq_name, idx, feat_cur[0:1], [{"cav_id": cid, "mod": mod, "feat_warped": fc_c, "valid_ego": ve_c}], update_state=False)[0]


def _step_hetshield_multicav(
    defender: HetShieldHEALDefender,
    wrapper: HEALBaseWrapper,
    ds_mix,
    sample_seqs: List[str],
    seq_name: str,
    idx: int,
    feat_cur: torch.Tensor,
    p_cur: dict,
    p_prev_cached: Optional[dict] = None,
    preload_attempted: bool = False,
) -> List[float]:
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
    from opencood.tools import train_utils

    ac = wrapper.model.pyramid_backbone.align_corners
    n_tot = int(feat_cur.shape[0])
    defender.reset_sequence(seq_name)

    if not preload_attempted and idx > 0 and sample_seqs[idx - 1] == seq_name:
        raw_prev = train_utils.to_device(ds_mix.collate_batch_test([ds_mix[idx - 1]]), wrapper.device_obj)
        p_prev_cached = wrapper.extract_protocol_features(raw_prev["ego"])

    if p_prev_cached is not None:
        fp = p_prev_cached["heter_feature_2d"]
        n_prev = int(fp.shape[0])
        mods_p = list(p_prev_cached["agent_modality_list"])
        prev_items = []
        for ci in range(1, min(n_tot, n_prev)):
            tm = p_prev_cached["affine_matrix"][0, 0, ci : ci + 1]
            fci = warp_affine_simple(fp[ci : ci + 1], tm, (PROTO_H, PROTO_W), align_corners=ac)
            vci = (warp_affine_simple(p_prev_cached["cam_valid_mask"][ci : ci + 1].float(), tm, (PROTO_H, PROTO_W), align_corners=ac) > 0.5).float()
            prev_items.append({"cav_id": f"{seq_name}_mc{ci}", "mod": mods_p[ci], "feat_warped": fci, "valid_ego": vci})
        if prev_items:
            defender.step_frame(seq_name, idx - 1, fp[0:1], prev_items, update_state=True)

    mods_c = list(p_cur["agent_modality_list"])
    cur_items = []
    for ci in range(1, n_tot):
        tm = p_cur["affine_matrix"][0, 0, ci : ci + 1]
        fci = warp_affine_simple(feat_cur[ci : ci + 1], tm, (PROTO_H, PROTO_W), align_corners=ac)
        vci = (warp_affine_simple(p_cur["cam_valid_mask"][ci : ci + 1].float(), tm, (PROTO_H, PROTO_W), align_corners=ac) > 0.5).float()
        cur_items.append({"cav_id": f"{seq_name}_mc{ci}", "mod": mods_c[ci], "feat_warped": fci, "valid_ego": vci})
    res_l = defender.step_frame(seq_name, idx, feat_cur[0:1], cur_items, update_state=False)
    return [float(r["g_k"]) for r in res_l]


def _new_method_seq_bucket(include_atk_ap: bool = True) -> dict:
    d = {
        "ap50": {m: {"tp": [], "fp": [], "score": [], "gt": 0} for m in ("m1", "m2")},
        "ap70": {m: {"tp": [], "fp": [], "score": [], "gt": 0} for m in ("m1", "m2")},
        "blind_ap50": {m: {"tp": [], "fp": [], "score": [], "gt": 0} for m in ("m1", "m2")},
        "multicav_ge3_clean_ap50": {"tp": [], "fp": [], "score": [], "gt": 0},
        "multicav_ge3_atk_ap50": {"tp": [], "fp": [], "score": [], "gt": 0},
        "asr": {atk: {m: [0, 0] for m in ("m1", "m2")} for atk in ATTACKS_5},
        "near_gt_fab": {atk: {m: [0, 0] for m in ("m1", "m2")} for atk in ("local_ghost_2.0sigma", "local_occluded_ghost_2.0sigma")},
        "honest_sup": {
            m: {"sup_grids": 0, "tot_grids": 0, "w_num": 0.0, "w_den": 0.0, "by_vis": {vn: [0, 0] for vn in VIS_NAMES_3}}
            for m in ("m1", "m2")
        },
    }
    if include_atk_ap:
        d["atk_ap50"] = {atk: {m: {"tp": [], "fp": [], "score": [], "gt": 0} for m in ("m1", "m2")} for atk in ATTACKS_5}
    return d


def _push_ap(bucket: dict, match_res: dict):
    bucket["tp"].extend(match_res["tp"])
    bucket["fp"].extend(match_res["fp"])
    bucket["score"].extend(match_res["score"])
    bucket["gt"] += int(match_res["gt"])


def _run_sequence_m6_or_m8(
    stage: str,
    seq_name: str,
    indices: List[int],
    wrapper: HEALBaseWrapper,
    ds_m1,
    ds_m2,
    ds_mix,
    sample_seqs: List[str],
    gx: torch.Tensor,
    gy: torch.Tensor,
    cfgs: dict,
    s_min: float,
    cache_dir: str,
    use_cache: bool,
) -> dict:
    os.makedirs(cache_dir, exist_ok=True)
    c_path = os.path.join(cache_dir, f"{stage}_{seq_name}.json")
    if use_cache and os.path.exists(c_path):
        try:
            with open(c_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("indices") == list(indices):
                print(f"  [{stage.upper()} 缓存命中] 序列 {seq_name} ({len(indices)} 帧)", flush=True)
                return cached
        except Exception:
            pass

    from opencood.tools import train_utils
    from opencood.utils import common_utils

    t_s = time.time()
    c_tau, l_tau = cfgs["c_tau"], cfgs["l_tau"]
    method_specs = cfgs["method_specs"]  # OrderedDict: name -> dict
    m_names = ["no_defense"] + list(method_specs.keys())
    seq_buckets = {mn: _new_method_seq_bucket(include_atk_ap=True) for mn in m_names}

    # B4 专用累积（仅 m6）
    b4_gammas = [float(g) for g in B4_GAMMA_CANDIDATES]
    b4_blind = {str(g): {m: {"tp": [], "fp": [], "score": [], "gt": 0} for m in ("m1", "m2")} for g in b4_gammas}
    b4_occ_asr = {str(g): {m: [0, 0] for m in ("m1", "m2")} for g in b4_gammas}

    # HetShield 诊断累积（m6 必报）
    hs_diag = {
        m: {"tau_k": [], "sigma_k": [], "s_k": [], "g_k_paper": [], "g_k_init_p95": [], "blind_ap_paper": {"tp": [], "fp": [], "score": [], "gt": 0}}
        for m in ("m1", "m2")
    }
    skipped_errors = []

    has_hs_methods = any(ms["kind"] == "hetshield" for ms in method_specs.values()) or (stage == "m6" and cfgs.get("hs_diag_defenders") is not None)
    last_proto_cache = {"idx": -999, "m1": None, "m2": None, "mix": None}

    for idx, b1_cpu, b2_cpu, bmix_cpu, pb1_cpu, pb2_cpu, pbmix_cpu in _iter_seq_prefetched_batches(
        indices, sample_seqs, seq_name, ds_m1, ds_m2, ds_mix=ds_mix, need_prev=has_hs_methods
    ):
        b = extract_frame_context_and_clean(
            wrapper, ds_m1, ds_m2, idx, gx, gy, clear_pts_tau=c_tau, los_occ_tau=l_tau, pre_b1_cpu=b1_cpu, pre_b2_cpu=b2_cpu
        )
        b_post = train_utils.to_device(_slim_batch_for_postprocess(b1_cpu), wrapper.device_obj)
        with torch.no_grad():
            ego_out = wrapper.forward_single_agent(b["p1"]["heter_feature_2d"][0:1])
        ego_pb_np, ego_ps_np, gt_np = _decode_np(wrapper, b_post, ego_out, ds_m1)
        gt_polys = list(common_utils.convert_format(gt_np))
        occ_mask_4d = b["occ_mask_ego"].float()

        # ================= 1. 单协作车成对评测 (m1 / m2) =================
        for mod, p_dict, ds_ref, p_prev_cpu_mod in (("m1", b["p1"], ds_m1, pb1_cpu), ("m2", b["p2"], ds_m2, pb2_cpu)):
            cached_prev_p = last_proto_cache[mod] if last_proto_cache["idx"] == idx - 1 else None
            prev_pair_cache = (
                _extract_prev_pair_warped(wrapper, ds_ref, sample_seqs, seq_name, idx, pre_prev_cpu=p_prev_cpu_mod, pre_prev_proto=cached_prev_p)
                if has_hs_methods else None
            )
            cov_ego_mod = b["cov_ego_m1_full"] if mod == "m1" else b["cov_ego"]
            cov_2d = cov_ego_mod[0, 0]
            blind_ids = _find_blind_gts(ds_m1, idx, ego_pb_np, ego_ps_np, gt_np, gt_polys, cov_2d)
            feat, rlen, amat = p_dict["heter_feature_2d"], p_dict["record_len"], p_dict["affine_matrix"]
            mods_l, vmask = list(p_dict["agent_modality_list"]), p_dict["cam_valid_mask"]

            # Clean 无防御与基础偏差图
            out_cln, dev_cln, dd_cln, da_cln, pf_cln = _extract_pair_dev(wrapper, p_dict, feat, b["p_ego_cls"], cov_ego_mod)
            cln_pb_np, cln_ps_np, _ = _decode_np(wrapper, b_post, out_cln, ds_ref)
            cln_m50 = _match_standard_ap(cln_pb_np, cln_ps_np, gt_polys, 0.5)
            cln_m70 = _match_standard_ap(cln_pb_np, cln_ps_np, gt_polys, 0.7)
            cln_mb = match_blind_spot_ap_ignore_nonblind(cln_pb_np, cln_ps_np, gt_polys, blind_ids, 0.5)
            _push_ap(seq_buckets["no_defense"]["ap50"][mod], cln_m50)
            _push_ap(seq_buckets["no_defense"]["ap70"][mod], cln_m70)
            _push_ap(seq_buckets["no_defense"]["blind_ap50"][mod], cln_mb)

            fg_cln_v0 = build_fg_mask(b["p_ego_cls"], pf_cln, b["v_idx"], fg_prob_tau=0.20, mode="v0")
            fg_cln_v1 = build_fg_mask(b["p_ego_cls"], pf_cln, b["v_idx"], fg_prob_tau=0.20, mode="v1")

            # 单协作车 (n_cav=2) 恒等门控精确去重缓存（根据定理 B3-Case4，n_cav=2 时 enable_b3=True/False 逐元素恒等）
            cln_gate_eval_memo = []

            def _eval_clean_2cav_gated(g_t: torch.Tensor, ds_m_str: str):
                if bool((g_t >= 1.0).all().item()):
                    return cln_pb_np, cln_ps_np, cln_m50, cln_m70, cln_mb
                for prev_dsm, prev_gt, prev_res in cln_gate_eval_memo:
                    if prev_dsm == ds_m_str and torch.equal(g_t, prev_gt):
                        return prev_res
                with torch.no_grad():
                    o_tmp = wrapper.forward_from_protocol(
                        feat, rlen, amat, mods_l, custom_fuse_fn=build_gated_fuse_fn(g_t, downsample_mode=ds_m_str, enable_b3=True)
                    )
                pb_t, ps_t, _ = _decode_np(wrapper, b_post, o_tmp, ds_ref)
                res_t = (
                    pb_t, ps_t,
                    _match_standard_ap(pb_t, ps_t, gt_polys, 0.5),
                    _match_standard_ap(pb_t, ps_t, gt_polys, 0.7),
                    match_blind_spot_ap_ignore_nonblind(pb_t, ps_t, gt_polys, blind_ids, 0.5),
                )
                cln_gate_eval_memo.append((ds_m_str, g_t, res_t))
                return res_t

            # 各防御方法在 Clean 上的表现
            for mn, mspec in method_specs.items():
                kind = mspec["kind"]
                if kind == "bud":
                    aud: BUDFusionHEALAuditor = mspec["auditor"]
                    fg_m = fg_cln_v0 if mspec["fg_mode"] == "v0" else fg_cln_v1
                    ar_c = aud.audit_dev_maps(
                        dev_cln, cov_2d, b["d_idx"], b["v_idx"], mod, build_gate=True,
                        d_drop_tol=dd_cln, d_add_tol=da_cln, enable_whole_fallback=True,
                        collab_fg_mask=fg_m, gamma_add_override=mspec.get("gamma_add_override"),
                    )
                    gate_c = ar_c["gate_L0"]
                    wf_c = bool(ar_c["whole_fallback"])
                    _, _, m50_c, m70_c, mb_c = _eval_clean_2cav_gated(gate_c, mspec["downsample_mode"])
                    hst = compute_honest_suppression_stats(gate_c, cov_2d, b["v_idx"], whole_fallback=wf_c)
                    hs_acc = seq_buckets[mn]["honest_sup"][mod]
                    hs_acc["sup_grids"] += hst["suppressed_grids"]
                    hs_acc["tot_grids"] += hst["total_grids"]
                    hs_acc["w_num"] += hst["reclaimed_weight_num"]
                    hs_acc["w_den"] += hst["reclaimed_weight_den"]
                    for vn in VIS_NAMES_3:
                        hs_acc["by_vis"][vn][0] += hst["by_visibility"][vn]["suppressed_grids"]
                        hs_acc["by_vis"][vn][1] += hst["by_visibility"][vn]["total_grids"]
                    _push_ap(seq_buckets[mn]["ap50"][mod], m50_c)
                    _push_ap(seq_buckets[mn]["ap70"][mod], m70_c)
                    _push_ap(seq_buckets[mn]["blind_ap50"][mod], mb_c)
                    continue
                elif kind == "vis_mask":
                    gate_c = build_visibility_mask_gate(b["v_idx"]).to(device=feat.device)
                    _, _, m50_c, m70_c, mb_c = _eval_clean_2cav_gated(gate_c, "min_pool")
                    hst = compute_honest_suppression_stats(gate_c, cov_2d, b["v_idx"], whole_fallback=False)
                    hs_acc = seq_buckets[mn]["honest_sup"][mod]
                    hs_acc["sup_grids"] += hst["suppressed_grids"]
                    hs_acc["tot_grids"] += hst["total_grids"]
                    for vn in VIS_NAMES_3:
                        hs_acc["by_vis"][vn][0] += hst["by_visibility"][vn]["suppressed_grids"]
                        hs_acc["by_vis"][vn][1] += hst["by_visibility"][vn]["total_grids"]
                    _push_ap(seq_buckets[mn]["ap50"][mod], m50_c)
                    _push_ap(seq_buckets[mn]["ap70"][mod], m70_c)
                    _push_ap(seq_buckets[mn]["blind_ap50"][mod], mb_c)
                    continue
                elif kind == "hetshield":
                    hs_r = _step_hetshield_one_frame_with_prev(
                        mspec["defender"], wrapper, ds_ref, sample_seqs, seq_name, idx, mod, feat, p_dict,
                        prev_warped_tuple=prev_pair_cache, preload_attempted=True,
                    )
                    gk = float(hs_r["g_k"])
                    with torch.no_grad():
                        o_c = forward_hetshield_protocol_mix(wrapper, feat, rlen, amat, mods_l, vmask, [gk])
                    hs_acc = seq_buckets[mn]["honest_sup"][mod]
                    n_cov = int(cov_2d.sum().item())
                    if gk < 0.5:
                        hs_acc["sup_grids"] += n_cov
                    hs_acc["tot_grids"] += n_cov
                elif kind == "lucia":
                    with torch.no_grad():
                        o_c = wrapper.forward_from_protocol(
                            feat, rlen, amat, mods_l, custom_fuse_fn=build_lucia_fuse_fn(32, l1_threshold=mspec.get("l1_threshold"))
                        )
                    tr_l = o_c["fusion_diag"].get("lucia_trust", [1.0, 1.0])
                    hs_acc = seq_buckets[mn]["honest_sup"][mod]
                    n_cov = int(cov_2d.sum().item())
                    if len(tr_l) > 1 and float(tr_l[1]) < 0.5:
                        hs_acc["sup_grids"] += n_cov
                    hs_acc["tot_grids"] += n_cov
                else:
                    raise ValueError(f"未知方法类型: {kind}")

                pb_c, ps_c, _ = _decode_np(wrapper, b_post, o_c, ds_ref)
                _push_ap(seq_buckets[mn]["ap50"][mod], _match_standard_ap(pb_c, ps_c, gt_polys, 0.5))
                _push_ap(seq_buckets[mn]["ap70"][mod], _match_standard_ap(pb_c, ps_c, gt_polys, 0.7))
                _push_ap(seq_buckets[mn]["blind_ap50"][mod], match_blind_spot_ap_ignore_nonblind(pb_c, ps_c, gt_polys, blind_ids, 0.5))

            # M6 附加：HetShield 诊断（容错保护，不阻断主线）
            if stage == "m6" and cfgs.get("hs_diag_defenders") is not None:
                try:
                    hs_d_pap = cfgs["hs_diag_defenders"]["paper"]
                    hs_d_p95 = cfgs["hs_diag_defenders"]["init_p95"]
                    r_pap = _step_hetshield_one_frame_with_prev(
                        hs_d_pap, wrapper, ds_ref, sample_seqs, seq_name, idx, mod, feat, p_dict,
                        prev_warped_tuple=prev_pair_cache, preload_attempted=True,
                    )
                    r_p95 = _step_hetshield_one_frame_with_prev(
                        hs_d_p95, wrapper, ds_ref, sample_seqs, seq_name, idx, mod, feat, p_dict,
                        prev_warped_tuple=prev_pair_cache, preload_attempted=True,
                    )
                    hs_diag[mod]["tau_k"].append(float(r_pap["tau_k"]))
                    hs_diag[mod]["sigma_k"].append(float(r_pap["sigma_k"]))
                    hs_diag[mod]["s_k"].append(float(r_pap["s_k"]))
                    hs_diag[mod]["g_k_paper"].append(float(r_pap["g_k"]))
                    hs_diag[mod]["g_k_init_p95"].append(float(r_p95["g_k"]))
                    with torch.no_grad():
                        o_hsp = forward_hetshield_protocol_mix(wrapper, feat, rlen, amat, mods_l, vmask, [float(r_pap["g_k"])])
                    pb_hp, ps_hp, _ = _decode_np(wrapper, b_post, o_hsp, ds_ref)
                    _push_ap(hs_diag[mod]["blind_ap_paper"], match_blind_spot_ap_ignore_nonblind(pb_hp, ps_hp, gt_polys, blind_ids, 0.5))
                except Exception as exc:
                    skipped_errors.append(f"hs_diag({seq_name},{idx},{mod}): {repr(exc)}")

            # 同帧只生成 1 次 5 种攻击扰动，供所有方法与 B4 共享！
            f_pgd, _ = generate_scaled_pgd_removal_heal(wrapper, feat, rlen, amat, mods_l, vmask, attacker_idx=1, sigma_mult=1.0, steps=8)
            f_rem, st_rem = generate_local_removal_heal(wrapper, feat, rlen, amat, mods_l, vmask, attacker_idx=1, num_targets=2, patch_radius=5, sigma_mult=1.0, steps=10)
            f_fab, st_fab = generate_local_ghost_spoofing_heal(wrapper, feat, rlen, amat, mods_l, vmask, attacker_idx=1, num_spoofs=2, patch_radius=5, sigma_mult=2.0, steps=10, frame_seed=idx)
            f_occ, st_occ = generate_local_occluded_ghost_heal(wrapper, feat, rlen, amat, mods_l, vmask, occluded_mask_ego=occ_mask_4d, attacker_idx=1, num_spoofs=2, patch_radius=5, sigma_mult=2.0, steps=12, frame_seed=idx)
            f_near, st_near = generate_local_near0_6m_attack_heal(wrapper, feat, rlen, amat, mods_l, vmask, attacker_idx=1, sigma_mult=2.0, steps=12, frame_seed=idx, enable_ghost_fallback=False)

            # B4 γ(Occluded) 扫描（仅 m6）
            if stage == "m6":
                b4_aud: BUDFusionHEALAuditor = cfgs["b4_auditor"]
                _, dev_occ_b4, dd_occ_b4, da_occ_b4, pf_occ_b4 = _extract_pair_dev(wrapper, p_dict, f_occ, b["p_ego_cls"], cov_ego_mod)
                fg_occ_v1 = build_fg_mask(b["p_ego_cls"], pf_occ_b4, b["v_idx"], fg_prob_tau=0.20, mode="v1")
                occ_cands_b4 = [_grid_hw_to_xy(a[0], a[1]) for a in st_occ.get("anchors_hwr", [])]
                for g_occ in b4_gammas:
                    ga_vec = [1.0, 0.3, float(g_occ)]
                    ar_bc = b4_aud.audit_dev_maps(
                        dev_cln, cov_2d, b["d_idx"], b["v_idx"], mod, build_gate=True,
                        d_drop_tol=dd_cln, d_add_tol=da_cln, enable_whole_fallback=True,
                        collab_fg_mask=fg_cln_v1, gamma_add_override=ga_vec,
                    )
                    with torch.no_grad():
                        o_bc = wrapper.forward_from_protocol(
                            feat, rlen, amat, mods_l, custom_fuse_fn=build_gated_fuse_fn(ar_bc["gate_L0"], downsample_mode="min_pool", enable_b3=True)
                        )
                    pb_bc, ps_bc, _ = _decode_np(wrapper, b_post, o_bc, ds_ref)
                    _push_ap(b4_blind[str(g_occ)][mod], match_blind_spot_ap_ignore_nonblind(pb_bc, ps_bc, gt_polys, blind_ids, 0.5))
                    if occ_cands_b4:
                        ar_bo = b4_aud.audit_dev_maps(
                            dev_occ_b4, cov_2d, b["d_idx"], b["v_idx"], mod, build_gate=True,
                            d_drop_tol=dd_occ_b4, d_add_tol=da_occ_b4, enable_whole_fallback=True,
                            collab_fg_mask=fg_occ_v1, gamma_add_override=ga_vec,
                        )
                        with torch.no_grad():
                            o_bo = wrapper.forward_from_protocol(
                                f_occ, rlen, amat, mods_l, custom_fuse_fn=build_gated_fuse_fn(ar_bo["gate_L0"], downsample_mode="min_pool", enable_b3=True)
                            )
                        pb_bo, ps_bo, _ = _decode_np(wrapper, b_post, o_bo, ds_ref)
                        ast_b4 = evaluate_frame_asr_strict("fabrication", occ_cands_b4, cln_pb_np, cln_ps_np, pb_bo, ps_bo, gt_np, s_min=s_min)
                        b4_occ_asr[str(g_occ)][mod][0] += int(ast_b4["succ"])
                        b4_occ_asr[str(g_occ)][mod][1] += int(ast_b4["total"])

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
                out_nd_a, dev_a, dd_a, da_a, pf_a = _extract_pair_dev(wrapper, p_dict, f_adv, b["p_ego_cls"], cov_ego_mod)
                nd_pb_a, nd_ps_a, _ = _decode_np(wrapper, b_post, out_nd_a, ds_ref)
                _push_ap(seq_buckets["no_defense"]["atk_ap50"][atk_name][mod], _match_standard_ap(nd_pb_a, nd_ps_a, gt_polys, 0.5))
                if cands_xy:
                    st_nd = evaluate_frame_asr_strict(mode_s, cands_xy, cln_pb_np, cln_ps_np, nd_pb_a, nd_ps_a, gt_np, s_min=s_min)
                    seq_buckets["no_defense"]["asr"][atk_name][mod][0] += int(st_nd["succ"])
                    seq_buckets["no_defense"]["asr"][atk_name][mod][1] += int(st_nd["total"])
                    if atk_name in seq_buckets["no_defense"]["near_gt_fab"]:
                        seq_buckets["no_defense"]["near_gt_fab"][atk_name][mod][0] += int(st_nd["near_gt_succ"])
                        seq_buckets["no_defense"]["near_gt_fab"][atk_name][mod][1] += int(st_nd["near_gt_total"])

                fg_a_v0 = build_fg_mask(b["p_ego_cls"], pf_a, b["v_idx"], fg_prob_tau=0.20, mode="v0")
                fg_a_v1 = build_fg_mask(b["p_ego_cls"], pf_a, b["v_idx"], fg_prob_tau=0.20, mode="v1")
                atk_gate_eval_memo = []

                def _eval_atk_2cav_gated(g_t: torch.Tensor, ds_m_str: str):
                    if bool((g_t >= 1.0).all().item()):
                        return nd_pb_a, nd_ps_a, _match_standard_ap(nd_pb_a, nd_ps_a, gt_polys, 0.5)
                    for prev_dsm, prev_gt, prev_res in atk_gate_eval_memo:
                        if prev_dsm == ds_m_str and torch.equal(g_t, prev_gt):
                            return prev_res
                    with torch.no_grad():
                        o_tmp = wrapper.forward_from_protocol(
                            f_adv, rlen, amat, mods_l, custom_fuse_fn=build_gated_fuse_fn(g_t, downsample_mode=ds_m_str, enable_b3=True)
                        )
                    pb_t, ps_t, _ = _decode_np(wrapper, b_post, o_tmp, ds_ref)
                    res_t = (pb_t, ps_t, _match_standard_ap(pb_t, ps_t, gt_polys, 0.5))
                    atk_gate_eval_memo.append((ds_m_str, g_t, res_t))
                    return res_t

                for mn, mspec in method_specs.items():
                    kind = mspec["kind"]
                    if kind == "bud":
                        aud = mspec["auditor"]
                        fg_m = fg_a_v0 if mspec["fg_mode"] == "v0" else fg_a_v1
                        ar_a = aud.audit_dev_maps(
                            dev_a, cov_2d, b["d_idx"], b["v_idx"], mod, build_gate=True,
                            d_drop_tol=dd_a, d_add_tol=da_a, enable_whole_fallback=True,
                            collab_fg_mask=fg_m, gamma_add_override=mspec.get("gamma_add_override"),
                        )
                        df_pb_a, df_ps_a, m50_a = _eval_atk_2cav_gated(ar_a["gate_L0"], mspec["downsample_mode"])
                    elif kind == "vis_mask":
                        gate_vm = build_visibility_mask_gate(b["v_idx"]).to(device=f_adv.device)
                        df_pb_a, df_ps_a, m50_a = _eval_atk_2cav_gated(gate_vm, "min_pool")
                    elif kind == "hetshield":
                        hs_r = _step_hetshield_one_frame_with_prev(
                            mspec["defender"], wrapper, ds_ref, sample_seqs, seq_name, idx, mod, f_adv, p_dict,
                            prev_warped_tuple=prev_pair_cache, preload_attempted=True,
                        )
                        with torch.no_grad():
                            o_a = forward_hetshield_protocol_mix(wrapper, f_adv, rlen, amat, mods_l, vmask, [float(hs_r["g_k"])])
                        df_pb_a, df_ps_a, _ = _decode_np(wrapper, b_post, o_a, ds_ref)
                        m50_a = _match_standard_ap(df_pb_a, df_ps_a, gt_polys, 0.5)
                    elif kind == "lucia":
                        with torch.no_grad():
                            o_a = wrapper.forward_from_protocol(
                                f_adv, rlen, amat, mods_l, custom_fuse_fn=build_lucia_fuse_fn(32, l1_threshold=mspec.get("l1_threshold"))
                            )
                        df_pb_a, df_ps_a, _ = _decode_np(wrapper, b_post, o_a, ds_ref)
                        m50_a = _match_standard_ap(df_pb_a, df_ps_a, gt_polys, 0.5)
                    else:
                        raise ValueError(f"未知方法类型: {kind}")
                    _push_ap(seq_buckets[mn]["atk_ap50"][atk_name][mod], m50_a)
                    if cands_xy:
                        st_df = evaluate_frame_asr_strict(mode_s, cands_xy, cln_pb_np, cln_ps_np, df_pb_a, df_ps_a, gt_np, s_min=s_min)
                        seq_buckets[mn]["asr"][atk_name][mod][0] += int(st_df["succ"])
                        seq_buckets[mn]["asr"][atk_name][mod][1] += int(st_df["total"])
                        if atk_name in seq_buckets[mn]["near_gt_fab"]:
                            seq_buckets[mn]["near_gt_fab"][atk_name][mod][0] += int(st_df["near_gt_succ"])
                            seq_buckets[mn]["near_gt_fab"][atk_name][mod][1] += int(st_df["near_gt_total"])

        # ================= 2. 多车场景 (>=3 协作车，即总车数 >=4) 干净与单攻击者 AP@0.5 =================
        raw_m = train_utils.to_device(bmix_cpu, wrapper.device_obj)
        p_m = wrapper.extract_protocol_features(raw_m["ego"])
        feat_m = p_m["heter_feature_2d"]
        n_tot = int(feat_m.shape[0])
        if n_tot >= 4:
            b_post_m = train_utils.to_device(_slim_batch_for_postprocess(bmix_cpu), wrapper.device_obj)
            rlen_m, amat_m, mods_m, vmask_m = p_m["record_len"], p_m["affine_matrix"], list(p_m["agent_modality_list"]), p_m["cam_valid_mask"]
            with torch.no_grad():
                o_ego_m = wrapper.forward_single_agent(feat_m[0:1])
                p_ego_m = torch.sigmoid(o_ego_m["cls_preds"]).max(dim=1, keepdim=True)[0]
                o_nd_m = wrapper.forward_from_protocol(feat_m, rlen_m, amat_m, mods_m)
            nd_pb_m, nd_ps_m, gt_m_np = _decode_np(wrapper, b_post_m, o_nd_m, ds_mix)
            gt_m_polys = list(common_utils.convert_format(gt_m_np))
            _push_ap(seq_buckets["no_defense"]["multicav_ge3_clean_ap50"], _match_standard_ap(nd_pb_m, nd_ps_m, gt_m_polys, 0.5))

            # 生成多车场景下第 1 辆协作车被 PGD 1.0σ 攻击的特征（专门检验 B3 多车权重重分配给其他可信车的效果）
            f_m_atk, _ = generate_scaled_pgd_removal_heal(wrapper, feat_m, rlen_m, amat_m, mods_m, vmask_m, attacker_idx=1, sigma_mult=1.0, steps=8)
            with torch.no_grad():
                o_nd_ma = wrapper.forward_from_protocol(f_m_atk, rlen_m, amat_m, mods_m)
            nd_pba, nd_psa, _ = _decode_np(wrapper, b_post_m, o_nd_ma, ds_mix)
            _push_ap(seq_buckets["no_defense"]["multicav_ge3_atk_ap50"], _match_standard_ap(nd_pba, nd_psa, gt_m_polys, 0.5))

            per_di, per_vi = {}, {}
            for ci in range(1, n_tot):
                ptm_ci = raw_m["ego"]["pairwise_t_matrix"][0, ci, 0].detach().cpu().numpy()
                dist_ci = torch.sqrt((gx - float(ptm_ci[0, 3])) ** 2 + (gy - float(ptm_ci[1, 3])) ** 2).view(1, 1, PROTO_H, PROTO_W)
                di, vi = assign_dist5_vis3_indices_v2(dist_ci, b["los_factor"], b["expected_pts"], clear_pts_tau=c_tau, los_occ_tau=l_tau)
                per_di[ci], per_vi[ci] = di[0, 0], vi[0, 0]

            p_prev_mc = None
            if any(ms["kind"] == "hetshield" for ms in method_specs.values()) and idx > 0 and sample_seqs[idx - 1] == seq_name:
                if last_proto_cache["idx"] == idx - 1 and last_proto_cache["mix"] is not None:
                    p_prev_mc = last_proto_cache["mix"]
                elif pbmix_cpu is not None:
                    raw_pm = train_utils.to_device(pbmix_cpu, wrapper.device_obj)
                    p_prev_mc = wrapper.extract_protocol_features(raw_pm["ego"])
                else:
                    raw_pm = train_utils.to_device(ds_mix.collate_batch_test([ds_mix[idx - 1]]), wrapper.device_obj)
                    p_prev_mc = wrapper.extract_protocol_features(raw_pm["ego"])

            for tag_state, f_mc in (("multicav_ge3_clean_ap50", feat_m), ("multicav_ge3_atk_ap50", f_m_atk)):
                # 预提取每辆协作车在当前多车状态下的解耦偏差图，供多个 BUD 变体复用
                cav_devs = {}
                for ci in range(1, n_tot):
                    cav_devs[ci] = extract_single_cav_dev_and_signed(
                        wrapper, f_mc, amat_m, mods_m, vmask_m, p_ego_m, cav_idx=ci, k_tol=5, return_pfused=True
                    )
                for mn, mspec in method_specs.items():
                    kind = mspec["kind"]
                    if kind == "bud":
                        aud = mspec["auditor"]
                        gates_m = []
                        for ci in range(1, n_tot):
                            dev_ci, dd_ci, da_ci, cov_ci, pf_ci = cav_devs[ci]
                            fg_ci = build_fg_mask(p_ego_m, pf_ci, per_vi[ci], fg_prob_tau=0.20, mode=mspec["fg_mode"])
                            ar_ci = aud.audit_dev_maps(
                                dev_ci, cov_ci, per_di[ci], per_vi[ci], mods_m[ci], build_gate=True,
                                d_drop_tol=dd_ci, d_add_tol=da_ci, enable_whole_fallback=True,
                                collab_fg_mask=fg_ci, gamma_add_override=mspec.get("gamma_add_override"),
                            )
                            gates_m.append(ar_ci["gate_L0"])
                        with torch.no_grad():
                            o_mc = wrapper.forward_from_protocol(
                                f_mc, rlen_m, amat_m, mods_m,
                                custom_fuse_fn=build_gated_fuse_fn(torch.cat(gates_m, dim=0), downsample_mode=mspec["downsample_mode"], enable_b3=mspec["enable_b3"]),
                            )
                    elif kind == "vis_mask":
                        gates_m = [build_visibility_mask_gate(per_vi[ci]).to(device=f_mc.device) for ci in range(1, n_tot)]
                        with torch.no_grad():
                            o_mc = wrapper.forward_from_protocol(
                                f_mc, rlen_m, amat_m, mods_m,
                                custom_fuse_fn=build_gated_fuse_fn(torch.cat(gates_m, dim=0), downsample_mode="min_pool", enable_b3=True),
                            )
                    elif kind == "hetshield":
                        g_mc = _step_hetshield_multicav(
                            mspec["defender"], wrapper, ds_mix, sample_seqs, seq_name, idx, f_mc, p_m,
                            p_prev_cached=p_prev_mc, preload_attempted=True,
                        )
                        with torch.no_grad():
                            o_mc = forward_hetshield_protocol_mix(wrapper, f_mc, rlen_m, amat_m, mods_m, vmask_m, g_mc)
                    elif kind == "lucia":
                        with torch.no_grad():
                            o_mc = wrapper.forward_from_protocol(
                                f_mc, rlen_m, amat_m, mods_m, custom_fuse_fn=build_lucia_fuse_fn(32, l1_threshold=mspec.get("l1_threshold"))
                            )
                    pb_mc, ps_mc, _ = _decode_np(wrapper, b_post_m, o_mc, ds_mix)
                    _push_ap(seq_buckets[mn][tag_state], _match_standard_ap(pb_mc, ps_mc, gt_m_polys, 0.5))

        last_proto_cache = {"idx": idx, "m1": b["p1"], "m2": b["p2"], "mix": p_m}

    last_proto_cache.clear()
    seq_out = {
        "seq_name": seq_name,
        "indices": list(indices),
        "buckets": seq_buckets,
        "b4_blind": b4_blind,
        "b4_occ_asr": b4_occ_asr,
        "hs_diag": hs_diag,
        "skipped_errors": skipped_errors,
    }
    if use_cache:
        _atomic_write_json(c_path, seq_out)
    print(f"  [{stage.upper()} 完成] 序列 {seq_name}: {len(indices)} 帧 ({time.time() - t_s:.1f}s)", flush=True)
    return seq_out


def _summarize_method_vs_ref(
    seq_order: List[str],
    seq_results: Dict[str, dict],
    meth_a: str,
    meth_b: str,
    boot_idx: np.ndarray,
) -> OrderedDict:
    """计算方法 A 相对方法 B 的全套 A0.4 标准指标及成对序列块 Bootstrap 95% CI。"""
    out = OrderedDict()
    for mod in ("m1", "m2"):
        a5 = {s: seq_results[s]["buckets"][meth_a]["ap50"][mod] for s in seq_order}
        b5 = {s: seq_results[s]["buckets"][meth_b]["ap50"][mod] for s in seq_order}
        d5 = paired_block_bootstrap_ap_diff(seq_order, a5, b5, boot_indices=boot_idx)

        a7 = {s: seq_results[s]["buckets"][meth_a]["ap70"][mod] for s in seq_order}
        b7 = {s: seq_results[s]["buckets"][meth_b]["ap70"][mod] for s in seq_order}
        d7 = paired_block_bootstrap_ap_diff(seq_order, a7, b7, boot_indices=boot_idx)

        ab = {s: seq_results[s]["buckets"][meth_a]["blind_ap50"][mod] for s in seq_order}
        bb = {s: seq_results[s]["buckets"][meth_b]["blind_ap50"][mod] for s in seq_order}
        db = paired_block_bootstrap_ap_diff(seq_order, ab, bb, boot_indices=boot_idx)

        sup_g = sum(seq_results[s]["buckets"][meth_a]["honest_sup"][mod]["sup_grids"] for s in seq_order)
        tot_g = sum(seq_results[s]["buckets"][meth_a]["honest_sup"][mod]["tot_grids"] for s in seq_order)
        wn = sum(seq_results[s]["buckets"][meth_a]["honest_sup"][mod]["w_num"] for s in seq_order)
        wd = sum(seq_results[s]["buckets"][meth_a]["honest_sup"][mod]["w_den"] for s in seq_order)
        by_v = {}
        for vn in VIS_NAMES_3:
            v_s = sum(seq_results[s]["buckets"][meth_a]["honest_sup"][mod]["by_vis"][vn][0] for s in seq_order)
            v_t = sum(seq_results[s]["buckets"][meth_a]["honest_sup"][mod]["by_vis"][vn][1] for s in seq_order)
            by_v[vn] = round(v_s / max(1, v_t), 5)

        asr_dict = OrderedDict()
        atk_ap_dict = OrderedDict()
        for atk in ATTACKS_5:
            ha = {s: seq_results[s]["buckets"][meth_a]["asr"][atk][mod][0] for s in seq_order}
            hb = {s: seq_results[s]["buckets"][meth_b]["asr"][atk][mod][0] for s in seq_order}
            tt = {s: seq_results[s]["buckets"][meth_b]["asr"][atk][mod][1] for s in seq_order}
            rd = paired_block_bootstrap_rate_diff(seq_order, ha, hb, tt, boot_indices=boot_idx)
            asr_dict[atk] = {
                "val_a": rd["val_a"], "a_ci95": rd["a_ci95"],
                "val_b": rd["val_b"], "b_ci95": rd["b_ci95"],
                "diff_a_minus_b": rd["diff"], "diff_ci95": rd["diff_ci95"],
                "total_targets": rd["total"],
            }
            aa_atk = {s: seq_results[s]["buckets"][meth_a]["atk_ap50"][atk][mod] for s in seq_order}
            bb_atk = {s: seq_results[s]["buckets"][meth_b]["atk_ap50"][atk][mod] for s in seq_order}
            da_atk = paired_block_bootstrap_ap_diff(seq_order, aa_atk, bb_atk, boot_indices=boot_idx)
            atk_ap_dict[atk] = {
                "val_a": da_atk["val_a"], "a_ci95": da_atk["a_ci95"],
                "val_b": da_atk["val_b"], "b_ci95": da_atk["b_ci95"],
                "diff_a_minus_b": da_atk["diff"], "diff_ci95": da_atk["diff_ci95"],
            }

        out[mod] = OrderedDict({
            "clean_ap50": {"val_a": d5["val_a"], "a_ci95": d5["a_ci95"], "val_b": d5["val_b"], "b_ci95": d5["b_ci95"], "diff": d5["diff"], "diff_ci95": d5["diff_ci95"]},
            "clean_ap70": {"val_a": d7["val_a"], "a_ci95": d7["a_ci95"], "val_b": d7["val_b"], "b_ci95": d7["b_ci95"], "diff": d7["diff"], "diff_ci95": d7["diff_ci95"]},
            "clean_blind_ap50": {"val_a": db["val_a"], "a_ci95": db["a_ci95"], "val_b": db["val_b"], "b_ci95": db["b_ci95"], "diff": db["diff"], "diff_ci95": db["diff_ci95"]},
            "honest_suppression": {
                "overall_ratio_g_lt_0.5": round(sup_g / max(1, tot_g), 5),
                "reclaimed_weight_ratio": round(wn / max(1e-8, wd), 5),
                "by_visibility": by_v,
            },
            "attacked_ap50": atk_ap_dict,
            "attacks_asr": asr_dict,
        })

    ge3_seqs = [s for s in seq_order if seq_results[s]["buckets"]["no_defense"]["multicav_ge3_clean_ap50"]["gt"] > 0]
    if ge3_seqs:
        ge3_boot = build_seq_bootstrap_indices(ge3_seqs, n_boot=len(boot_idx), seed=BOOT_SEED)
        mc_c_a = {s: seq_results[s]["buckets"][meth_a]["multicav_ge3_clean_ap50"] for s in ge3_seqs}
        mc_c_b = {s: seq_results[s]["buckets"][meth_b]["multicav_ge3_clean_ap50"] for s in ge3_seqs}
        mc_clean = paired_block_bootstrap_ap_diff(ge3_seqs, mc_c_a, mc_c_b, boot_indices=ge3_boot)

        mc_a_a = {s: seq_results[s]["buckets"][meth_a]["multicav_ge3_atk_ap50"] for s in ge3_seqs}
        mc_a_b = {s: seq_results[s]["buckets"][meth_b]["multicav_ge3_atk_ap50"] for s in ge3_seqs}
        mc_atk = paired_block_bootstrap_ap_diff(ge3_seqs, mc_a_a, mc_a_b, boot_indices=ge3_boot)
    else:
        mc_clean = {"val_a": 0.0, "a_ci95": [0.0, 0.0], "val_b": 0.0, "b_ci95": [0.0, 0.0], "diff": 0.0, "diff_ci95": [0.0, 0.0]}
        mc_atk = dict(mc_clean)

    out["multicav_ge3"] = {
        "clean_ap50": mc_clean,
        "pgd_1attacker_ap50": mc_atk,
    }
    return out


def main():
    parser = argparse.ArgumentParser(description="M6 (dev_fast 400帧定方法) & M8 (dev_full 全量确认) 统一评测运行器")
    parser.add_argument("--stage", type=str, required=True, choices=["m6", "m8"])
    parser.add_argument("--data_dir", type=str, default=OPV2V_TRAIN_DIR)
    parser.add_argument("--splits_json", type=str, default=V3_SPLITS_JSON)
    parser.add_argument("--subset", type=str, default=None, help="默认 m6->dev_fast, m8->dev_full")
    parser.add_argument("--calib_v0_json", type=str, default=V3_BUD_CALIB_V0_JSON)
    parser.add_argument("--calib_v1_json", type=str, default=V3_BUD_CALIB_V1_JSON)
    parser.add_argument("--calib_a1_before_json", type=str, default=os.path.join(V3_RESULTS_DIR, "bud_calib_a1_before.json"))
    parser.add_argument("--bud_tuned_json", type=str, default=V3_BUD_TUNED_JSON)
    parser.add_argument("--hs_calib_json", type=str, default=V3_HETSHIELD_CALIB_JSON)
    parser.add_argument("--baselines_tuned_json", type=str, default=V3_BASELINES_TUNED_JSON)
    parser.add_argument("--m6_decisions_json", type=str, default=os.path.join(V3_RESULTS_DIR, "m6_method_decisions.json"))
    parser.add_argument("--cache_dir", type=str, default=os.path.join(V3_RESULTS_DIR, "_m6_m8_seq_cache"))
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--s_min", type=float, default=0.20)
    parser.add_argument("--n_boot", type=int, default=BOOT_N)
    parser.add_argument("--max_frames", type=int, default=0, help=">0 时仅跑前 N 帧用于 M4 2帧冒烟测试")
    parser.add_argument("--only_seqs", type=str, default="", help="逗号分隔的序列子集，仅运行并写序列缓存供多 worker 并行加速")
    args = parser.parse_args()

    t0 = time.time()
    subset_name = args.subset or ("dev_fast" if args.stage == "m6" else "dev_full")
    is_smoke = int(args.max_frames) > 0

    with open(args.splits_json, "r", encoding="utf-8") as f:
        raw_sp = json.load(f)
    eval_split = _resolve_eval_subset(raw_sp, subset_name, max_frames=args.max_frames if is_smoke else None)
    seq_order = list(eval_split["seq_order"])
    # 断言绝对零触碰 validate / test / test_culvercity
    for s in seq_order:
        assert s not in set(raw_sp["groups"].get("validate", [])), f"越界访问 validate 序列: {s}"
        assert s not in set(raw_sp["groups"].get("test", [])), f"硬停机保护：严禁访问 test 序列: {s}"
        assert s not in set(raw_sp["groups"].get("test_culvercity", [])), f"硬停机保护：严禁访问 test_culvercity 序列: {s}"

    wrapper = HEALBaseWrapper(device="cuda", use_cav=7)
    ds_m1 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m1")
    ds_m2 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m2")
    ds_mix = wrapper.build_dataset(data_dir=args.data_dir, use_cav=7, visualize=True, collab_modality="mixed", max_cav=7)
    sample_seqs = wrapper.get_sample_sequences(ds_m2)
    gx, gy = build_bev_grid_xy_256(device="cuda", dtype=torch.float32)

    method_specs = OrderedDict()
    hs_diag_defenders = None
    b4_auditor = None

    if args.stage == "m6":
        aud_v1 = BUDFusionHEALAuditor(args.calib_v1_json)
        aud_v0 = BUDFusionHEALAuditor(args.calib_v0_json)
        aud_a1_bef = BUDFusionHEALAuditor(args.calib_a1_before_json)
        c_tau, l_tau = require_visibility_v2_taus(aud_v1.gating_calib)
        b4_auditor = aud_v1

        method_specs["bud_full_init"] = {"kind": "bud", "auditor": aud_v1, "downsample_mode": "min_pool", "fg_mode": "v1", "enable_b3": True}
        method_specs["a1_before"] = {"kind": "bud", "auditor": aud_a1_bef, "downsample_mode": "min_pool", "fg_mode": "v1", "enable_b3": True}
        method_specs["b1_area"] = {"kind": "bud", "auditor": aud_v1, "downsample_mode": "area", "fg_mode": "v1", "enable_b3": True}
        method_specs["b2_v0"] = {"kind": "bud", "auditor": aud_v0, "downsample_mode": "min_pool", "fg_mode": "v0", "enable_b3": True}
        method_specs["b3_ego_only"] = {"kind": "bud", "auditor": aud_v1, "downsample_mode": "min_pool", "fg_mode": "v1", "enable_b3": False}
        method_specs["vis_mask_baseline"] = {"kind": "vis_mask"}

        if os.path.exists(args.hs_calib_json):
            with open(args.hs_calib_json, "r", encoding="utf-8") as f:
                hs_c = json.load(f)
            hs_pap = HetShieldHEALDefender(mode="shared", alpha=float(hs_c["meta"]["alpha"]), tau_th=0.3, temperature=0.1).cuda().eval()
            hs_pap.predictor.load_state_dict(torch.load(hs_c["meta"]["shared_ckpt"], map_location="cpu"))
            hs_p95 = HetShieldHEALDefender(mode="shared", alpha=float(hs_c["meta"]["alpha"]), tau_th=float(hs_c["fit_tuned_shared_tau_p95"]), temperature=0.1).cuda().eval()
            hs_p95.predictor.load_state_dict(torch.load(hs_c["meta"]["shared_ckpt"], map_location="cpu"))
            hs_diag_defenders = {"paper": hs_pap, "init_p95": hs_p95}
    else:
        aud_tuned = BUDFusionHEALAuditor(args.bud_tuned_json)
        c_tau, l_tau = require_visibility_v2_taus(aud_tuned.gating_calib)
        m6_dec = {"downsample_mode": "min_pool", "fg_mode": "v1", "enable_b3": True, "gamma_occluded": 0.0}
        if os.path.exists(args.m6_decisions_json):
            with open(args.m6_decisions_json, "r", encoding="utf-8") as f:
                m6_dec.update(json.load(f).get("locked_method_switches", {}))

        ga_override = [1.0, 0.3, float(m6_dec["gamma_occluded"])]
        method_specs["bud_tuned"] = {
            "kind": "bud",
            "auditor": aud_tuned,
            "downsample_mode": str(m6_dec["downsample_mode"]),
            "fg_mode": str(m6_dec["fg_mode"]),
            "enable_b3": bool(m6_dec["enable_b3"]),
            "gamma_add_override": ga_override,
        }
        method_specs["vis_mask_baseline"] = {"kind": "vis_mask"}

        with open(args.hs_calib_json, "r", encoding="utf-8") as f:
            hs_c = json.load(f)
        with open(args.baselines_tuned_json, "r", encoding="utf-8") as f:
            bl_t = json.load(f)

        alpha_v = float(hs_c["meta"]["alpha"])
        sh_sd = torch.load(hs_c["meta"]["shared_ckpt"], map_location="cpu")
        ps_sd = torch.load(hs_c["meta"]["persensor_ckpt"], map_location="cpu")

        hs_paper = HetShieldHEALDefender(mode="shared", alpha=alpha_v, tau_th=0.3, temperature=0.1).cuda().eval()
        hs_paper.predictor.load_state_dict(sh_sd)
        method_specs["hetshield_paper"] = {"kind": "hetshield", "defender": hs_paper}

        hs_tuned = HetShieldHEALDefender(mode="shared", alpha=alpha_v, tau_th=float(bl_t["hetshield_tuned"]["chosen_tau_th"]), temperature=0.1).cuda().eval()
        hs_tuned.predictor.load_state_dict(sh_sd)
        method_specs["hetshield_tuned"] = {"kind": "hetshield", "defender": hs_tuned}

        hs_per = HetShieldHEALDefender(mode="persensor", alpha=alpha_v, tau_th=0.3, temperature=0.1, per_sensor_tau=bl_t["hetshield_persensor"]["chosen_per_sensor_tau"]).cuda().eval()
        for m in ("m1", "m2"):
            hs_per.predictors[m].load_state_dict(ps_sd[m])
        method_specs["hetshield_persensor"] = {"kind": "hetshield", "defender": hs_per}

        method_specs["lucia_tuned"] = {"kind": "lucia", "l1_threshold": bl_t["lucia"]["chosen_l1_threshold"]}

    cfgs = {
        "c_tau": c_tau,
        "l_tau": l_tau,
        "method_specs": method_specs,
        "hs_diag_defenders": hs_diag_defenders,
        "b4_auditor": b4_auditor,
    }

    seq_results = OrderedDict()
    all_skipped = []
    only_set = {x.strip() for x in args.only_seqs.split(",") if x.strip()} if args.only_seqs else None
    run_seqs = [s for s in seq_order if s in only_set] if only_set else seq_order
    for s in run_seqs:
        sr = _run_sequence_m6_or_m8(
            args.stage, s, eval_split["sequences"][s]["indices"], wrapper, ds_m1, ds_m2, ds_mix,
            sample_seqs, gx, gy, cfgs, s_min=args.s_min, cache_dir=args.cache_dir, use_cache=not is_smoke,
        )
        seq_results[s] = sr
        all_skipped.extend(sr.get("skipped_errors", []))

    if only_set is not None:
        print(f"[{args.stage.upper()} Worker 完成] 已完成序列缓存分片: {run_seqs} (耗时 {time.time() - t0:.1f}s)", flush=True)
        return

    n_boot_eff = 50 if is_smoke else int(args.n_boot)
    boot_idx = build_seq_bootstrap_indices(seq_order, n_boot=n_boot_eff, seed=BOOT_SEED)

    if args.stage == "m6":
        # 1. 各方法对 no_defense 的绝对表现
        abs_vs_nd = OrderedDict()
        for mn in method_specs:
            abs_vs_nd[mn] = _summarize_method_vs_ref(seq_order, seq_results, mn, "no_defense", boot_idx)

        # 2. 四项单变量修改前后成对比较（After vs Before）
        pairwise_ablations = OrderedDict({
            "A1_unified_visibility": _summarize_method_vs_ref(seq_order, seq_results, "bud_full_init", "a1_before", boot_idx),
            "B1_min_pool_vs_area": _summarize_method_vs_ref(seq_order, seq_results, "bud_full_init", "b1_area", boot_idx),
            "B2_fg_v1_vs_v0_paired_calib": _summarize_method_vs_ref(seq_order, seq_results, "bud_full_init", "b2_v0", boot_idx),
            "B3_trusted_other_vs_ego_only": _summarize_method_vs_ref(seq_order, seq_results, "bud_full_init", "b3_ego_only", boot_idx),
        })

        # 3. B4 权衡曲线
        b4_curve = OrderedDict()
        for g_occ in [float(x) for x in B4_GAMMA_CANDIDATES]:
            gk = str(g_occ)
            ck = f"gamma_occ_{g_occ:.1f}"
            b4_curve[ck] = OrderedDict()
            for mod in ("m1", "m2"):
                ap_d = paired_block_bootstrap_ap_diff(
                    seq_order,
                    {s: seq_results[s]["b4_blind"][gk][mod] for s in seq_order},
                    {s: seq_results[s]["b4_blind"]["0.0"][mod] for s in seq_order},
                    boot_indices=boot_idx,
                )
                hg = {s: seq_results[s]["b4_occ_asr"][gk][mod][0] for s in seq_order}
                h0 = {s: seq_results[s]["b4_occ_asr"]["0.0"][mod][0] for s in seq_order}
                tt = {s: seq_results[s]["b4_occ_asr"]["0.0"][mod][1] for s in seq_order}
                rd = paired_block_bootstrap_rate_diff(seq_order, hg, h0, tt, boot_indices=boot_idx)
                b4_curve[ck][mod] = {
                    "clean_blind_ap50": ap_d["val_a"],
                    "clean_blind_ap50_ci95": ap_d["a_ci95"],
                    "blind_ap50_diff_vs_0": ap_d["diff"],
                    "occluded_fab_asr": rd["val_a"],
                    "occluded_fab_asr_ci95": rd["a_ci95"],
                    "occluded_fab_asr_diff_vs_0": rd["diff"],
                }

        # 4. HetShield 诊断汇总
        hs_diag_summary = OrderedDict()
        for mod in ("m1", "m2"):
            tau_all = [v for s in seq_order for v in seq_results[s]["hs_diag"][mod]["tau_k"]]
            sig_all = [v for s in seq_order for v in seq_results[s]["hs_diag"][mod]["sigma_k"]]
            sk_all = [v for s in seq_order for v in seq_results[s]["hs_diag"][mod]["s_k"]]
            g_pap = np.asarray([v for s in seq_order for v in seq_results[s]["hs_diag"][mod]["g_k_paper"]], dtype=np.float64)
            g_p95 = np.asarray([v for s in seq_order for v in seq_results[s]["hs_diag"][mod]["g_k_init_p95"]], dtype=np.float64)
            b_hs = {s: seq_results[s]["hs_diag"][mod]["blind_ap_paper"] for s in seq_order}
            b_nd = {s: seq_results[s]["buckets"]["no_defense"]["blind_ap50"][mod] for s in seq_order}
            b_diff = paired_block_bootstrap_ap_diff(seq_order, b_hs, b_nd, boot_indices=boot_idx)
            hs_diag_summary[mod] = OrderedDict({
                "honest_gk_lt_0.5_ratio_paper_0.3": round(float((g_pap < 0.5).mean()), 5) if g_pap.size else 0.0,
                "honest_gk_lt_0.5_ratio_calib_p95": round(float((g_p95 < 0.5).mean()), 5) if g_p95.size else 0.0,
                "tau_k_temporal_distribution": _dist_summary_1d(tau_all),
                "sigma_k_spatial_distribution": _dist_summary_1d(sig_all),
                "s_k_combined_distribution": _dist_summary_1d(sk_all),
                "clean_blind_ap50_paper_0.3": {"val": b_diff["val_a"], "ci95": b_diff["a_ci95"], "no_defense": b_diff["val_b"], "diff": b_diff["diff"]},
            })

        # 5. 按客观验收标准生成每项临时去留决定（供 M7 自动锁定，明早人工复核）
        def _mean_asr_diff(pw: dict, atks=ATTACKS_5) -> float:
            return float(np.mean([pw[m]["attacks_asr"][a]["diff_a_minus_b"] for m in ("m1", "m2") for a in atks]))

        def _max_clean_drop(pw: dict) -> float:
            return max(-pw["m1"]["clean_ap50"]["diff"], -pw["m2"]["clean_ap50"]["diff"])

        pw_a1 = pairwise_ablations["A1_unified_visibility"]
        keep_a1 = bool(_max_clean_drop(pw_a1) <= 0.015)

        pw_b1 = pairwise_ablations["B1_min_pool_vs_area"]
        b1_local_asr_diff = _mean_asr_diff(pw_b1, ("local_removal_1.0sigma", "local_ghost_2.0sigma", "local_near0_6m_2.0sigma"))
        keep_b1 = bool(b1_local_asr_diff <= 0.005 and _max_clean_drop(pw_b1) <= 0.010)

        pw_b2 = pairwise_ablations["B2_fg_v1_vs_v0_paired_calib"]
        b2_fab_asr_diff = 0.5 * (
            pw_b2["m1"]["attacks_asr"]["local_ghost_2.0sigma"]["diff_a_minus_b"]
            + pw_b2["m2"]["attacks_asr"]["local_ghost_2.0sigma"]["diff_a_minus_b"]
        )
        b2_max_blind_drop = max(-pw_b2["m1"]["clean_blind_ap50"]["diff"], -pw_b2["m2"]["clean_blind_ap50"]["diff"])
        keep_b2 = bool(b2_fab_asr_diff <= 0.005 and b2_max_blind_drop <= 0.015)

        pw_b3 = pairwise_ablations["B3_trusted_other_vs_ego_only"]
        mc_clean_diff = float(pw_b3["multicav_ge3"]["clean_ap50"]["diff"])
        mc_atk_diff = float(pw_b3["multicav_ge3"]["pgd_1attacker_ap50"]["diff"])
        keep_b3 = bool(mc_clean_diff >= -0.005 and mc_atk_diff >= -0.005)

        c01 = b4_curve["gamma_occ_0.1"]
        b4_max_blind_drop = max(-c01["m1"]["blind_ap50_diff_vs_0"], -c01["m2"]["blind_ap50_diff_vs_0"])
        b4_mean_asr_red = -0.5 * (c01["m1"]["occluded_fab_asr_diff_vs_0"] + c01["m2"]["occluded_fab_asr_diff_vs_0"])
        choose_gamma_01 = bool(b4_mean_asr_red >= 0.02 and b4_max_blind_drop <= 0.005)

        decisions = OrderedDict({
            "meta": {
                "claim": "[临时结论 · 待人工代码复核]",
                "subset": subset_name,
                "num_frames": sum(len(eval_split["sequences"][s]["indices"]) for s in seq_order),
            },
            "per_modification_verdicts": OrderedDict({
                "A1": {
                    "keep": keep_a1,
                    "max_clean_ap50_drop_vs_before": round(_max_clean_drop(pw_a1), 5),
                    "mean_5atk_asr_diff_vs_before": round(_mean_asr_diff(pw_a1), 5),
                    "rationale": "A1 消除标定与推理可见性函数不一致（方法正确性修复），干净 AP 波动在容许范围内，予以保留。" if keep_a1 else "A1 引起超预期干净掉点，暂回退待人工复核。",
                },
                "B1": {
                    "keep": keep_b1,
                    "chosen_downsample_mode": "min_pool" if keep_b1 else "area",
                    "mean_local_asr_diff_minpool_minus_area": round(b1_local_asr_diff, 5),
                    "max_clean_ap50_drop_vs_area": round(_max_clean_drop(pw_b1), 5),
                    "rationale": f"min_pool 相对 area 平均局部攻击 ASR 变化 {b1_local_asr_diff:+.4f}，最大干净 AP 降幅 {_max_clean_drop(pw_b1):.4f} <= 0.010。",
                },
                "B2": {
                    "keep": keep_b2,
                    "chosen_fg_mode": "v1" if keep_b2 else "v0",
                    "mean_ghost_asr_diff_v1_minus_v0": round(b2_fab_asr_diff, 5),
                    "max_clean_blind_ap50_drop_vs_v0": round(b2_max_blind_drop, 5),
                    "rationale": f"配对各自标定表下，v1 相对 v0 局部伪造 ASR 变化 {b2_fab_asr_diff:+.4f}，盲区 AP 最大降幅 {b2_max_blind_drop:.4f}。",
                },
                "B3": {
                    "keep": keep_b3,
                    "chosen_enable_b3": keep_b3,
                    "multicav_ge3_clean_ap50_diff": round(mc_clean_diff, 5),
                    "multicav_ge3_1attacker_ap50_diff": round(mc_atk_diff, 5),
                    "rationale": f"多车 (>=4车) 干净 AP 差值 {mc_clean_diff:+.4f}，单车受攻多车 AP 差值 {mc_atk_diff:+.4f}。",
                },
                "B4": {
                    "chosen_gamma_occluded": 0.1 if choose_gamma_01 else 0.0,
                    "mean_occluded_asr_reduction_at_0.1": round(b4_mean_asr_red, 5),
                    "max_clean_blind_ap50_drop_at_0.1": round(b4_max_blind_drop, 5),
                    "rationale": "γ=0.1 同时满足遮挡伪造 ASR 降幅>=2% 且盲区 AP 降幅<=0.5pt，选 0.1。" if choose_gamma_01 else "保持 γ(Occluded)=0.0（保护盲区召回优先）。",
                },
            }),
            "locked_method_switches": OrderedDict({
                "use_a1_unified_vis": keep_a1,
                "downsample_mode": "min_pool" if keep_b1 else "area",
                "fg_mode": "v1" if keep_b2 else "v0",
                "enable_b3": keep_b3,
                "gamma_occluded": 0.1 if choose_gamma_01 else 0.0,
            }),
        })
        if not is_smoke:
            _atomic_write_json(args.m6_decisions_json, decisions)
        else:
            _atomic_write_json(os.path.join(os.path.dirname(args.out_json), "smoke_m6_decisions.json"), decisions)

        report = OrderedDict({
            "meta": {
                "claim": "[临时结论 · 待人工代码复核]",
                "stage": "M6_dev_fast_400_method_ablation_and_diagnostics",
                "subset": subset_name,
                "num_sequences": len(seq_order),
                "num_frames": sum(len(eval_split["sequences"][s]["indices"]) for s in seq_order),
                "clear_pts_tau_used": c_tau,
                "s_min": float(args.s_min),
                "elapsed_sec": round(time.time() - t0, 2),
                "skipped_noncritical_errors": all_skipped,
            },
            "decisions": decisions,
            "pairwise_ablations_after_minus_before": pairwise_ablations,
            "methods_vs_no_defense": abs_vs_nd,
            "b4_gamma_occluded_tradeoff": b4_curve,
            "visibility_mask_baseline": abs_vs_nd["vis_mask_baseline"],
            "hetshield_diagnostics": hs_diag_summary,
        })
        _atomic_write_json(args.out_json, report)
        print(f"\n[M6 完成] 报告已保存至 {args.out_json} (耗时 {time.time() - t0:.1f}s)", flush=True)
    else:
        methods_vs_nd = OrderedDict()
        bud_vs_baselines = OrderedDict()
        for mn in method_specs:
            methods_vs_nd[mn] = _summarize_method_vs_ref(seq_order, seq_results, mn, "no_defense", boot_idx)
            if mn != "bud_tuned":
                bud_vs_baselines[f"bud_tuned_minus_{mn}"] = _summarize_method_vs_ref(seq_order, seq_results, "bud_tuned", mn, boot_idx)

        report = OrderedDict({
            "meta": {
                "claim": "[临时结论 · 待人工代码复核]",
                "stage": "M8_dev_full_final_confirmation_before_freeze",
                "subset": subset_name,
                "num_sequences": len(seq_order),
                "num_frames": sum(len(eval_split["sequences"][s]["indices"]) for s in seq_order),
                "clear_pts_tau_used": c_tau,
                "s_min": float(args.s_min),
                "c_stage_final_test_touched": False,
                "elapsed_sec": round(time.time() - t0, 2),
            },
            "methods_vs_no_defense": methods_vs_nd,
            "paired_bud_vs_baselines": bud_vs_baselines,
        })
        _atomic_write_json(args.out_json, report)
        print(f"\n[M8 完成] 开发组全量帧最终确认报告已保存至 {args.out_json} (耗时 {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
