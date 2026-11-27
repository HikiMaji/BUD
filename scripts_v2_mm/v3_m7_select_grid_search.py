# -*- coding: utf-8 -*-
"""
M7（科学执行顺序 ③）：在调参组 select（600 帧，覆盖全部 12 条 select 序列）上，
使用完全相同的统一目标函数搜索：
  1. 锁定 M6 去留决定后的最终 BUD-Fusion 门控超参网格 -> results_v3/bud_tuned_calibrator.json & grid_search_records.json
  2. HetShield（调参版全局阈值、分传感器版双模态阈值；论文默认 0.3 版固定不调参）
  3. LUCIA（L1 异常隔离门限，含原版无门限候选） -> results_v3/baselines_tuned_thresholds.json

严格内存与隔离保证：
- 采用逐帧流式评估全部候选配置 + 按序列 JSON 落盘缓存（_m7_select_seq_cache/{seq}.json），
  绝不在内存或磁盘中堆积 600 帧多状态特征张量（杜绝 115 GB 爆内存，内存峰值 < 4 GB）；
- 硬断言 select 序列与 calib / dev / validate / test 零重叠；
- 今晚不接入 CP-uniGuard（按规范第零节第 2 条留待明天白天单独接入）。
"""

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict
from typing import Dict, List

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
    _atomic_write_json,
    _decode_np,
    _extract_pair_dev,
    _find_blind_gts,
    _match_standard_ap,
    _resolve_eval_subset,
    _slim_batch_for_postprocess,
)
from scripts_v2_mm.v3_m6_m8_eval_runner import _extract_prev_pair_warped, _step_hetshield_one_frame_with_prev
from src_v2_mm.baselines_robosac_lucia_heal import build_lucia_fuse_fn, cp_defense_lucia_heal
from src_v2_mm.bud_fusion_heal import (
    BUDFusionHEALAuditor,
    build_directional_gate_map,
    build_fg_mask,
    build_gated_fuse_fn,
)
from src_v2_mm.eval_stats_utils import fast_voc_ap
from src_v2_mm.heal_attacks import (
    generate_local_ghost_spoofing_heal,
    generate_local_near0_6m_attack_heal,
    generate_local_occluded_ghost_heal,
    generate_local_removal_heal,
    generate_scaled_pgd_removal_heal,
)
from src_v2_mm.heal_base_wrapper import HEALBaseWrapper
from src_v2_mm.hetshield_heal import HetShieldHEALDefender, forward_hetshield_protocol_mix
from src_v2_mm.sprint_metrics_v3 import match_blind_spot_ap_ignore_nonblind
from src_v2_mm.step4_1_v2_core import (
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
    VIS_LABELS_3,
    build_bev_grid_xy_256,
    extract_frame_context_and_clean,
    require_visibility_v2_taus,
)

UNIFIED_OBJECTIVE_FORMULA = (
    "J = mean_5atk_ap50 - 1.50 * max(0.0, max_clean_ap50_loss - 0.010) "
    "- 0.80 * max(0.0, max_clean_blind_ap50_loss - 0.020)"
)


def _empty_cand_acc() -> dict:
    return {
        "clean_ap50": {m: {"tp": [], "fp": [], "score": [], "gt": 0} for m in ("m1", "m2")},
        "clean_blind_ap50": {m: {"tp": [], "fp": [], "score": [], "gt": 0} for m in ("m1", "m2")},
        "atk_ap50": {a: {m: {"tp": [], "fp": [], "score": [], "gt": 0} for m in ("m1", "m2")} for a in ATTACKS_5},
    }


def _push(b: dict, r: dict):
    b["tp"].extend(r["tp"])
    b["fp"].extend(r["fp"])
    b["score"].extend(r["score"])
    b["gt"] += int(r["gt"])


def _merge_cand_acc(dst: dict, src: dict):
    for m in ("m1", "m2"):
        _push(dst["clean_ap50"][m], src["clean_ap50"][m])
        _push(dst["clean_blind_ap50"][m], src["clean_blind_ap50"][m])
        for a in ATTACKS_5:
            _push(dst["atk_ap50"][a][m], src["atk_ap50"][a][m])


def _score_unified_objective(acc: dict, nd_ref: dict) -> OrderedDict:
    """对任意防御候选（BUD / HetShield / LUCIA）使用完全相同的目标函数计算得分。"""
    cln_ap = {
        m: fast_voc_ap(acc["clean_ap50"][m]["tp"], acc["clean_ap50"][m]["fp"], acc["clean_ap50"][m]["score"], acc["clean_ap50"][m]["gt"])
        for m in ("m1", "m2")
    }
    bln_ap = {
        m: fast_voc_ap(acc["clean_blind_ap50"][m]["tp"], acc["clean_blind_ap50"][m]["fp"], acc["clean_blind_ap50"][m]["score"], acc["clean_blind_ap50"][m]["gt"])
        for m in ("m1", "m2")
    }
    atk_ap = {
        m: {
            a: fast_voc_ap(acc["atk_ap50"][a][m]["tp"], acc["atk_ap50"][a][m]["fp"], acc["atk_ap50"][a][m]["score"], acc["atk_ap50"][a][m]["gt"])
            for a in ATTACKS_5
        }
        for m in ("m1", "m2")
    }
    cln_loss = {m: nd_ref["clean_ap50"][m] - cln_ap[m] for m in ("m1", "m2")}
    bln_loss = {m: nd_ref["clean_blind_ap50"][m] - bln_ap[m] for m in ("m1", "m2")}
    max_cln_loss = max(cln_loss["m1"], cln_loss["m2"])
    max_bln_loss = max(bln_loss["m1"], bln_loss["m2"])
    mean_atk_ap50 = float(np.mean([atk_ap[m][a] for m in ("m1", "m2") for a in ATTACKS_5]))
    obj = (
        mean_atk_ap50
        - 1.50 * max(0.0, max_cln_loss - 0.010)
        - 0.80 * max(0.0, max_bln_loss - 0.020)
    )
    return OrderedDict({
        "clean_ap50": {m: round(cln_ap[m], 5) for m in ("m1", "m2")},
        "clean_blind_ap50": {m: round(bln_ap[m], 5) for m in ("m1", "m2")},
        "atk_ap50": {m: {a: round(atk_ap[m][a], 5) for a in ATTACKS_5} for m in ("m1", "m2")},
        "clean_ap50_loss_vs_nd": {m: round(cln_loss[m], 5) for m in ("m1", "m2")},
        "clean_blind_ap50_loss_vs_nd": {m: round(bln_loss[m], 5) for m in ("m1", "m2")},
        "mean_5atk_ap50": round(mean_atk_ap50, 5),
        "strict_constraint_met": bool(max_cln_loss <= 0.010 and max_bln_loss <= 0.020),
        "objective_J": round(float(obj), 5),
    })


def _run_one_select_sequence(
    seq_name: str,
    indices: List[int],
    wrapper: HEALBaseWrapper,
    ds_m1,
    ds_m2,
    sample_seqs: List[str],
    gx: torch.Tensor,
    gy: torch.Tensor,
    auditor: BUDFusionHEALAuditor,
    c_tau: float,
    l_tau: float,
    m6_switches: dict,
    bud_candidates: List[dict],
    hs_shared_def: HetShieldHEALDefender,
    hs_per_def: HetShieldHEALDefender,
    hs_shared_taus: List[float],
    hs_per_taus_m1: List[float],
    hs_per_taus_m2: List[float],
    lucia_taus: List,
    cache_dir: str,
    use_cache: bool,
) -> dict:
    os.makedirs(cache_dir, exist_ok=True)
    c_path = os.path.join(cache_dir, f"{seq_name}.json")
    if use_cache and os.path.exists(c_path):
        try:
            with open(c_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("indices") == list(indices) and len(cached.get("bud_accs", [])) == len(bud_candidates):
                print(f"  [M7 缓存命中] select 序列 {seq_name} ({len(indices)} 帧)", flush=True)
                return cached
        except Exception:
            pass

    from opencood.tools import train_utils
    from opencood.utils import common_utils

    t_s = time.time()
    fg_mode = str(m6_switches["fg_mode"])
    ds_mode = str(m6_switches["downsample_mode"])
    en_b3 = bool(m6_switches["enable_b3"])
    fg_prob_tau = float(auditor.gating_calib["gating_params"].get("fg_prob_tau", 0.20))
    frame_div_p95 = auditor.gating_calib["frame_div_p95"]["tier4_sensor_dist_vis"]
    frame_high_tau = auditor.gating_calib["frame_high_tau"]

    nd_acc = _empty_cand_acc()
    bud_accs = [_empty_cand_acc() for _ in bud_candidates]
    hs_sh_accs = [_empty_cand_acc() for _ in hs_shared_taus]
    # PerSensor 解耦加速：单车成对评估时 m1 与 m2 互不影响，可分别独立搜索 m1 候选与 m2 候选（复杂度从 5×5=25 降为 5+5=10 次前向，数学等价！）
    hs_pm1_accs = [_empty_cand_acc() for _ in hs_per_taus_m1]
    hs_pm2_accs = [_empty_cand_acc() for _ in hs_per_taus_m2]
    luc_accs = [_empty_cand_acc() for _ in lucia_taus]

    kappa = 10.0  # T = 0.1

    for idx in indices:
        b = extract_frame_context_and_clean(wrapper, ds_m1, ds_m2, idx, gx, gy, clear_pts_tau=c_tau, los_occ_tau=l_tau)
        raw_b1 = train_utils.to_device(ds_m1.collate_batch_test([ds_m1[idx]]), "cpu")
        b_post = train_utils.to_device(_slim_batch_for_postprocess(raw_b1), wrapper.device_obj)
        with torch.no_grad():
            ego_out = wrapper.forward_single_agent(b["p1"]["heter_feature_2d"][0:1])
        ego_pb_np, ego_ps_np, gt_np = _decode_np(wrapper, b_post, ego_out, ds_m1)
        gt_polys = list(common_utils.convert_format(gt_np))
        occ_mask_4d = b["occ_mask_ego"].float()

        for mod, p_dict, ds_ref in (("m1", b["p1"], ds_m1), ("m2", b["p2"], ds_m2)):
            from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

            prev_pair_cache = _extract_prev_pair_warped(wrapper, ds_ref, sample_seqs, seq_name, idx)
            cov_ego_mod = b["cov_ego_m1_full"] if mod == "m1" else b["cov_ego"]
            cov_2d = cov_ego_mod[0, 0]
            blind_ids = _find_blind_gts(ds_m1, idx, ego_pb_np, ego_ps_np, gt_np, gt_polys, cov_2d)
            feat, rlen, amat = p_dict["heter_feature_2d"], p_dict["record_len"], p_dict["affine_matrix"]
            mods_l, vmask = list(p_dict["agent_modality_list"]), p_dict["cam_valid_mask"]
            ac = wrapper.model.pyramid_backbone.align_corners

            # 生成一次 5 种攻击，所有方法与所有候选参数共享同一份特征张量，流式评完本帧即释放！
            f_pgd, _ = generate_scaled_pgd_removal_heal(wrapper, feat, rlen, amat, mods_l, vmask, attacker_idx=1, sigma_mult=1.0, steps=8)
            f_rem, _ = generate_local_removal_heal(wrapper, feat, rlen, amat, mods_l, vmask, attacker_idx=1, num_targets=2, patch_radius=5, sigma_mult=1.0, steps=10)
            f_fab, _ = generate_local_ghost_spoofing_heal(wrapper, feat, rlen, amat, mods_l, vmask, attacker_idx=1, num_spoofs=2, patch_radius=5, sigma_mult=2.0, steps=10, frame_seed=idx)
            f_occ, _ = generate_local_occluded_ghost_heal(wrapper, feat, rlen, amat, mods_l, vmask, occluded_mask_ego=occ_mask_4d, attacker_idx=1, num_spoofs=2, patch_radius=5, sigma_mult=2.0, steps=12, frame_seed=idx)
            f_near, _ = generate_local_near0_6m_attack_heal(wrapper, feat, rlen, amat, mods_l, vmask, attacker_idx=1, sigma_mult=2.0, steps=12, frame_seed=idx, enable_ghost_fallback=False)

            state_list = [
                ("clean", feat),
                ("pgd_1.0sigma", f_pgd),
                ("local_removal_1.0sigma", f_rem),
                ("local_ghost_2.0sigma", f_fab),
                ("local_occluded_ghost_2.0sigma", f_occ),
                ("local_near0_6m_2.0sigma", f_near),
            ]

            for st_name, f_cur in state_list:
                out_nd, dev_cur, dd_cur, da_cur, pf_cur = _extract_pair_dev(wrapper, p_dict, f_cur, b["p_ego_cls"], cov_ego_mod)
                pb_nd, ps_nd, _ = _decode_np(wrapper, b_post, out_nd, ds_ref)
                r5_nd = _match_standard_ap(pb_nd, ps_nd, gt_polys, 0.5)
                rb_nd = match_blind_spot_ap_ignore_nonblind(pb_nd, ps_nd, gt_polys, blind_ids, 0.5) if st_name == "clean" else None
                if st_name == "clean":
                    _push(nd_acc["clean_ap50"][mod], r5_nd)
                    _push(nd_acc["clean_blind_ap50"][mod], rb_nd)
                else:
                    _push(nd_acc["atk_ap50"][st_name][mod], r5_nd)

                # 预备全零门控结果按需惰性计算缓存（供 BUD 整车回退与 LUCIA 阈值隔离零冗余精确共享）
                zero_gate_cache = {}

                def _get_zero_gate_matches():
                    if "res" not in zero_gate_cache:
                        z_gate = torch.zeros((1, 1, PROTO_H, PROTO_W), device=f_cur.device, dtype=f_cur.dtype)
                        with torch.no_grad():
                            o_z = wrapper.forward_from_protocol(
                                f_cur, rlen, amat, mods_l,
                                custom_fuse_fn=build_gated_fuse_fn(z_gate, downsample_mode=ds_mode, enable_b3=en_b3),
                            )
                        pb_z, ps_z, _ = _decode_np(wrapper, b_post, o_z, ds_ref)
                        r5_z = _match_standard_ap(pb_z, ps_z, gt_polys, 0.5)
                        rb_z = match_blind_spot_ap_ignore_nonblind(pb_z, ps_z, gt_polys, blind_ids, 0.5) if st_name == "clean" else None
                        zero_gate_cache["res"] = (r5_z, rb_z)
                    return zero_gate_cache["res"]

                # 1) 流式评估全部 BUD 候选网格（全1/全0/相邻恒等门控数学精确复用）
                fg_cur = build_fg_mask(b["p_ego_cls"], pf_cur, b["v_idx"], fg_prob_tau=fg_prob_tau, mode=fg_mode)
                ar_base = auditor.audit_dev_maps(dev_cur, cov_2d, b["d_idx"], b["v_idx"], mod, build_gate=False)
                sc_t4 = float(ar_base["raw_t4"]) / float(frame_div_p95[mod])
                is_wf = sc_t4 > float(frame_high_tau[mod])
                last_gate_t = None
                last_bud_matches = None

                for ci, gp in enumerate(bud_candidates):
                    if is_wf:
                        r5_b, rb_b = _get_zero_gate_matches()
                    else:
                        gate = build_directional_gate_map(
                            cos_comb=ar_base["cos_comb_t4"], d_drop_tol=dd_cur, d_add_tol=da_cur,
                            cov_2d=cov_2d, v_idx=b["v_idx"],
                            cos_tau=gp["cos_tau_by_vis_bg"][mod], drop_tau=gp["drop_tau"], add_tau=gp["add_tau"],
                            w_drop=gp["w_drop"], w_add=gp["w_add"], cos_co_ratio=gp["cos_co_ratio"],
                            gamma_add=gp["gamma_add"], gamma_cos_gate=gp["gamma_cos_gate"],
                            r_dil=gp["r_dil"], beta=gp["beta"],
                            collab_fg_mask=fg_cur, cos_tau_fg=gp["cos_tau_by_vis_fg"][mod],
                        )
                        if bool((gate >= 1.0).all().item()):
                            r5_b, rb_b = r5_nd, rb_nd
                        elif last_gate_t is not None and torch.equal(gate, last_gate_t):
                            r5_b, rb_b = last_bud_matches
                        else:
                            with torch.no_grad():
                                o_b = wrapper.forward_from_protocol(
                                    f_cur, rlen, amat, mods_l,
                                    custom_fuse_fn=build_gated_fuse_fn(gate, downsample_mode=ds_mode, enable_b3=en_b3),
                                )
                            pb_b, ps_b, _ = _decode_np(wrapper, b_post, o_b, ds_ref)
                            r5_b = _match_standard_ap(pb_b, ps_b, gt_polys, 0.5)
                            rb_b = match_blind_spot_ap_ignore_nonblind(pb_b, ps_b, gt_polys, blind_ids, 0.5) if st_name == "clean" else None
                            last_gate_t = gate
                            last_bud_matches = (r5_b, rb_b)
                    if st_name == "clean":
                        _push(bud_accs[ci]["clean_ap50"][mod], r5_b)
                        _push(bud_accs[ci]["clean_blind_ap50"][mod], rb_b)
                    else:
                        _push(bud_accs[ci]["atk_ap50"][st_name][mod], r5_b)

                # 2) 流式评估 HetShield (Tuned Shared & PerSensor) 候选阈值（复用本帧预提的 prev_pair_cache！）
                rs_sh = _step_hetshield_one_frame_with_prev(
                    hs_shared_def, wrapper, ds_ref, sample_seqs, seq_name, idx, mod, f_cur, p_dict,
                    prev_warped_tuple=prev_pair_cache, preload_attempted=True,
                )
                sk_sh = float(rs_sh["s_k"])
                for ti, th_val in enumerate(hs_shared_taus):
                    gk = 1.0 / (1.0 + math.exp(max(-40.0, min(40.0, kappa * (sk_sh - float(th_val))))))
                    with torch.no_grad():
                        o_h = forward_hetshield_protocol_mix(wrapper, f_cur, rlen, amat, mods_l, vmask, [gk])
                    pb_h, ps_h, _ = _decode_np(wrapper, b_post, o_h, ds_ref)
                    r5_h = _match_standard_ap(pb_h, ps_h, gt_polys, 0.5)
                    if st_name == "clean":
                        _push(hs_sh_accs[ti]["clean_ap50"][mod], r5_h)
                        _push(hs_sh_accs[ti]["clean_blind_ap50"][mod], match_blind_spot_ap_ignore_nonblind(pb_h, ps_h, gt_polys, blind_ids, 0.5))
                    else:
                        _push(hs_sh_accs[ti]["atk_ap50"][st_name][mod], r5_h)

                rs_ps = _step_hetshield_one_frame_with_prev(
                    hs_per_def, wrapper, ds_ref, sample_seqs, seq_name, idx, mod, f_cur, p_dict,
                    prev_warped_tuple=prev_pair_cache, preload_attempted=True,
                )
                sk_ps = float(rs_ps["s_k"])
                p_taus = hs_per_taus_m1 if mod == "m1" else hs_per_taus_m2
                p_accs = hs_pm1_accs if mod == "m1" else hs_pm2_accs
                for ti, th_val in enumerate(p_taus):
                    gk = 1.0 / (1.0 + math.exp(max(-40.0, min(40.0, kappa * (sk_ps - float(th_val))))))
                    with torch.no_grad():
                        o_hp = forward_hetshield_protocol_mix(wrapper, f_cur, rlen, amat, mods_l, vmask, [gk])
                    pb_hp, ps_hp, _ = _decode_np(wrapper, b_post, o_hp, ds_ref)
                    r5_hp = _match_standard_ap(pb_hp, ps_hp, gt_polys, 0.5)
                    if st_name == "clean":
                        _push(p_accs[ti]["clean_ap50"][mod], r5_hp)
                        _push(p_accs[ti]["clean_blind_ap50"][mod], match_blind_spot_ap_ignore_nonblind(pb_hp, ps_hp, gt_polys, blind_ids, 0.5))
                    else:
                        _push(p_accs[ti]["atk_ap50"][st_name][mod], r5_hp)

                # 3) 流式评估 LUCIA 候选门限（硬门限仅存在 trust=[1,1] 与 trust=[1,0] 两种精确输出，避免重复前向！）
                with torch.no_grad():
                    feat_ego_l0 = warp_affine_simple(f_cur, amat[0, 0, :2], (PROTO_H, PROTO_W), align_corners=ac)
                    _, l1_raw = cp_defense_lucia_heal(feat_ego_l0, compression_ratio=32, l1_threshold=10.0)
                d1_val = float(l1_raw[1].item())
                luc_quarantine_cache = {}

                for li, l_th in enumerate(lucia_taus):
                    if l_th is None:
                        with torch.no_grad():
                            o_l = wrapper.forward_from_protocol(f_cur, rlen, amat, mods_l, custom_fuse_fn=build_lucia_fuse_fn(32, l1_threshold=None))
                        pb_l, ps_l, _ = _decode_np(wrapper, b_post, o_l, ds_ref)
                        r5_l = _match_standard_ap(pb_l, ps_l, gt_polys, 0.5)
                        rb_l = match_blind_spot_ap_ignore_nonblind(pb_l, ps_l, gt_polys, blind_ids, 0.5) if st_name == "clean" else None
                    elif d1_val <= float(l_th):
                        r5_l, rb_l = r5_nd, rb_nd
                    else:
                        if "q" not in luc_quarantine_cache:
                            with torch.no_grad():
                                o_lq = wrapper.forward_from_protocol(f_cur, rlen, amat, mods_l, custom_fuse_fn=build_lucia_fuse_fn(32, l1_threshold=float(l_th)))
                            pb_lq, ps_lq, _ = _decode_np(wrapper, b_post, o_lq, ds_ref)
                            r5_lq = _match_standard_ap(pb_lq, ps_lq, gt_polys, 0.5)
                            rb_lq = match_blind_spot_ap_ignore_nonblind(pb_lq, ps_lq, gt_polys, blind_ids, 0.5) if st_name == "clean" else None
                            luc_quarantine_cache["q"] = (r5_lq, rb_lq)
                        r5_l, rb_l = luc_quarantine_cache["q"]

                    if st_name == "clean":
                        _push(luc_accs[li]["clean_ap50"][mod], r5_l)
                        _push(luc_accs[li]["clean_blind_ap50"][mod], rb_l)
                    else:
                        _push(luc_accs[li]["atk_ap50"][st_name][mod], r5_l)

    out_seq = {
        "seq_name": seq_name,
        "indices": list(indices),
        "nd_acc": nd_acc,
        "bud_accs": bud_accs,
        "hs_sh_accs": hs_sh_accs,
        "hs_pm1_accs": hs_pm1_accs,
        "hs_pm2_accs": hs_pm2_accs,
        "luc_accs": luc_accs,
    }
    if use_cache:
        _atomic_write_json(c_path, out_seq)
    print(f"  [M7 完成] select 序列 {seq_name}: {len(indices)} 帧 ({time.time() - t_s:.1f}s)", flush=True)
    return out_seq


def main():
    parser = argparse.ArgumentParser(description="M7：select 调参组 600 帧统一目标函数流式网格搜索（BUD + HetShield + LUCIA）")
    parser.add_argument("--data_dir", type=str, default=OPV2V_TRAIN_DIR)
    parser.add_argument("--splits_json", type=str, default=V3_SPLITS_JSON)
    parser.add_argument("--calib_v0_json", type=str, default=V3_BUD_CALIB_V0_JSON)
    parser.add_argument("--calib_v1_json", type=str, default=V3_BUD_CALIB_V1_JSON)
    parser.add_argument("--m6_decisions_json", type=str, default=os.path.join(V3_RESULTS_DIR, "m6_method_decisions.json"))
    parser.add_argument("--hs_calib_json", type=str, default=V3_HETSHIELD_CALIB_JSON)
    parser.add_argument("--cache_dir", type=str, default=os.path.join(V3_RESULTS_DIR, "_m7_select_seq_cache"))
    parser.add_argument("--out_bud_tuned_json", type=str, default=V3_BUD_TUNED_JSON)
    parser.add_argument("--out_grid_records_json", type=str, default=os.path.join(V3_RESULTS_DIR, "grid_search_records.json"))
    parser.add_argument("--out_baselines_tuned_json", type=str, default=V3_BASELINES_TUNED_JSON)
    parser.add_argument("--max_frames", type=int, default=0, help=">0 时仅跑前 N 帧用于 M4 2帧冒烟测试")
    args = parser.parse_args()

    t0 = time.time()
    is_smoke = int(args.max_frames) > 0

    with open(args.splits_json, "r", encoding="utf-8") as f:
        raw_sp = json.load(f)
    select_split = _resolve_eval_subset(raw_sp, "select", max_frames=args.max_frames if is_smoke else None)
    seq_order = list(select_split["seq_order"])

    # 硬隔离校验：零触碰 calib / dev / validate / test / test_culvercity
    forbidden = (
        set(raw_sp["groups"].get("calib", []))
        | set(raw_sp["groups"].get("dev", []))
        | set(raw_sp["groups"].get("validate", []))
        | set(raw_sp["groups"].get("test", []))
        | set(raw_sp["groups"].get("test_culvercity", []))
    )
    assert set(seq_order).isdisjoint(forbidden), "硬停机保护：select 网格搜索序列与其他划分重叠！"
    if not is_smoke:
        assert set(seq_order) == set(raw_sp["groups"]["select"]), "select 600 帧必须完整覆盖全部 select 序列！"

    m6_switches = {"use_a1_unified_vis": True, "downsample_mode": "min_pool", "fg_mode": "v1", "enable_b3": True, "gamma_occluded": 0.0}
    if os.path.exists(args.m6_decisions_json):
        with open(args.m6_decisions_json, "r", encoding="utf-8") as f:
            m6_switches.update(json.load(f).get("locked_method_switches", {}))

    chosen_calib_path = args.calib_v1_json if m6_switches["fg_mode"] == "v1" else args.calib_v0_json
    with open(chosen_calib_path, "r", encoding="utf-8") as f:
        base_calib_obj = json.load(f)
    auditor = BUDFusionHEALAuditor(chosen_calib_path)
    c_tau, l_tau = require_visibility_v2_taus(base_calib_obj)

    cos_q_bg = base_calib_obj["pixel_cos_by_vis_bg_quantiles"]
    cos_q_fg = base_calib_obj["pixel_cos_by_vis_fg_quantiles"]
    fg_prob_tau = float(base_calib_obj["gating_params"].get("fg_prob_tau", 0.20))
    gamma_occ = float(m6_switches["gamma_occluded"])

    bud_search_space = OrderedDict({
        "bg_q": ["p99.8", "p99.9"],
        "fg_q": ["p98.0", "p99.0", "p99.5"],
        "gamma_cos_gate": [[1.0, 1.0, 1.0], [1.0, 0.85, 0.70]],
        "r_dil_and_beta": [[3, 3.5], [4, 4.0]],
    })
    bud_candidates = []
    for bq in bud_search_space["bg_q"]:
        for fq in bud_search_space["fg_q"]:
            ct_bg = {m: [cos_q_bg[m][vn][bq] for vn in VIS_LABELS_3] for m in ("m1", "m2")}
            ct_fg = {m: [max(cos_q_bg[m][vn][bq], cos_q_fg[m][vn][fq]) for vn in VIS_LABELS_3] for m in ("m1", "m2")}
            for gc in bud_search_space["gamma_cos_gate"]:
                for rd, bt in bud_search_space["r_dil_and_beta"]:
                    bud_candidates.append(OrderedDict({
                        "mode": "directional_vis_fg_bg",
                        "fg_mode": m6_switches["fg_mode"],
                        "fg_rule": m6_switches["fg_mode"],
                        "downsample_mode": m6_switches["downsample_mode"],
                        "enable_b3": bool(m6_switches["enable_b3"]),
                        "k_tol": 5,
                        "fg_prob_tau": fg_prob_tau,
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
                        "gamma_add": [1.0, 0.3, gamma_occ],
                        "r_dil": int(rd),
                        "beta": float(bt),
                    }))

    with open(args.hs_calib_json, "r", encoding="utf-8") as f:
        hs_c = json.load(f)
    alpha_v = float(hs_c["meta"]["alpha"])
    hs_shared_def = HetShieldHEALDefender(mode="shared", alpha=alpha_v, tau_th=0.3, temperature=0.1).cuda().eval()
    hs_shared_def.predictor.load_state_dict(torch.load(hs_c["meta"]["shared_ckpt"], map_location="cpu"))
    hs_per_def = HetShieldHEALDefender(mode="persensor", alpha=alpha_v, tau_th=0.3, temperature=0.1).cuda().eval()
    ps_sd = torch.load(hs_c["meta"]["persensor_ckpt"], map_location="cpu")
    for m in ("m1", "m2"):
        hs_per_def.predictors[m].load_state_dict(ps_sd[m])

    # 构造 HetShield (Tuned / PerSensor) 与 LUCIA 阈值搜索空间（论文默认 0.3 版不调参！）
    p95_sh = float(hs_c.get("fit_tuned_shared_tau_p95", 0.5))
    p98_sh = float(hs_c.get("fit_tuned_shared_tau_p98", 0.65))
    hs_shared_taus = sorted({round(x, 4) for x in [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, p95_sh, p98_sh, p95_sh * 1.10]})

    pm1_95 = float(hs_c.get("fit_persensor_tau_p95", {}).get("m1", 0.45))
    pm2_95 = float(hs_c.get("fit_persensor_tau_p95", {}).get("m2", 0.55))
    hs_per_taus_m1 = sorted({round(x, 4) for x in [0.25, 0.40, pm1_95, pm1_95 * 1.15, 0.75]})
    hs_per_taus_m2 = sorted({round(x, 4) for x in [0.30, 0.45, pm2_95, pm2_95 * 1.15, 0.85]})

    # LUCIA 归一化池化特征 L1 距离候选（含原版无阈值 None）
    lucia_taus = [None, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 36.0, 45.0]

    wrapper = HEALBaseWrapper(device="cuda")
    ds_m1 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m1")
    ds_m2 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m2")
    sample_seqs = wrapper.get_sample_sequences(ds_m2)
    gx, gy = build_bev_grid_xy_256(device="cuda", dtype=torch.float32)

    nd_tot = _empty_cand_acc()
    bud_tot = [_empty_cand_acc() for _ in bud_candidates]
    hs_sh_tot = [_empty_cand_acc() for _ in hs_shared_taus]
    hs_pm1_tot = [_empty_cand_acc() for _ in hs_per_taus_m1]
    hs_pm2_tot = [_empty_cand_acc() for _ in hs_per_taus_m2]
    luc_tot = [_empty_cand_acc() for _ in lucia_taus]

    total_frames = 0
    for s in seq_order:
        idxs = select_split["sequences"][s]["indices"]
        total_frames += len(idxs)
        sq = _run_one_select_sequence(
            s, idxs, wrapper, ds_m1, ds_m2, sample_seqs, gx, gy, auditor, c_tau, l_tau,
            m6_switches, bud_candidates, hs_shared_def, hs_per_def,
            hs_shared_taus, hs_per_taus_m1, hs_per_taus_m2, lucia_taus,
            cache_dir=args.cache_dir, use_cache=not is_smoke,
        )
        _merge_cand_acc(nd_tot, sq["nd_acc"])
        for i in range(len(bud_candidates)):
            _merge_cand_acc(bud_tot[i], sq["bud_accs"][i])
        for i in range(len(hs_shared_taus)):
            _merge_cand_acc(hs_sh_tot[i], sq["hs_sh_accs"][i])
        for i in range(len(hs_per_taus_m1)):
            _merge_cand_acc(hs_pm1_tot[i], sq["hs_pm1_accs"][i])
        for i in range(len(hs_per_taus_m2)):
            _merge_cand_acc(hs_pm2_tot[i], sq["hs_pm2_accs"][i])
        for i in range(len(lucia_taus)):
            _merge_cand_acc(luc_tot[i], sq["luc_accs"][i])

    nd_ref = {
        "clean_ap50": {
            m: fast_voc_ap(nd_tot["clean_ap50"][m]["tp"], nd_tot["clean_ap50"][m]["fp"], nd_tot["clean_ap50"][m]["score"], nd_tot["clean_ap50"][m]["gt"])
            for m in ("m1", "m2")
        },
        "clean_blind_ap50": {
            m: fast_voc_ap(nd_tot["clean_blind_ap50"][m]["tp"], nd_tot["clean_blind_ap50"][m]["fp"], nd_tot["clean_blind_ap50"][m]["score"], nd_tot["clean_blind_ap50"][m]["gt"])
            for m in ("m1", "m2")
        },
    }

    # === 1. 选出最优 BUD 配置 ===
    bud_records = []
    best_bud_strict, best_bud_overall = None, None
    for i, gp in enumerate(bud_candidates):
        sc = _score_unified_objective(bud_tot[i], nd_ref)
        rec = OrderedDict({"config_id": i, "gating_params": gp, **sc})
        bud_records.append(rec)
        if sc["strict_constraint_met"] and (best_bud_strict is None or sc["objective_J"] > best_bud_strict["objective_J"]):
            best_bud_strict = rec
        if best_bud_overall is None or sc["objective_J"] > best_bud_overall["objective_J"]:
            best_bud_overall = rec
    chosen_bud = best_bud_strict if best_bud_strict is not None else best_bud_overall

    bud_tuned_obj = OrderedDict(base_calib_obj)
    bud_tuned_obj["meta"] = OrderedDict({
        **base_calib_obj["meta"],
        "stage": "M7_select_600_grid_search_tuned",
        "select_sequences": seq_order,
        "select_num_frames": total_frames,
        "locked_m6_switches": m6_switches,
        "unified_objective_formula": UNIFIED_OBJECTIVE_FORMULA,
        "num_bud_configs_evaluated": len(bud_records),
    })
    bud_tuned_obj["gating_params"] = chosen_bud["gating_params"]
    bud_tuned_obj["select_chosen_eval"] = chosen_bud
    _atomic_write_json(args.out_bud_tuned_json, bud_tuned_obj)

    # === 2. 用完全相同的目标函数选出 HetShield (Tuned / PerSensor) 与 LUCIA 最优阈值 ===
    hs_sh_records = []
    best_hs_sh = None
    for i, th in enumerate(hs_shared_taus):
        sc = _score_unified_objective(hs_sh_tot[i], nd_ref)
        rec = OrderedDict({"tau_th": th, **sc})
        hs_sh_records.append(rec)
        if best_hs_sh is None or sc["objective_J"] > best_hs_sh["objective_J"]:
            best_hs_sh = rec

    # PerSensor 5 × 5 = 25 组合精确打分
    hs_ps_records = []
    best_hs_ps = None
    for i1, tm1 in enumerate(hs_per_taus_m1):
        for i2, tm2 in enumerate(hs_per_taus_m2):
            comb_acc = _empty_cand_acc()
            comb_acc["clean_ap50"]["m1"] = hs_pm1_tot[i1]["clean_ap50"]["m1"]
            comb_acc["clean_blind_ap50"]["m1"] = hs_pm1_tot[i1]["clean_blind_ap50"]["m1"]
            comb_acc["clean_ap50"]["m2"] = hs_pm2_tot[i2]["clean_ap50"]["m2"]
            comb_acc["clean_blind_ap50"]["m2"] = hs_pm2_tot[i2]["clean_blind_ap50"]["m2"]
            for a in ATTACKS_5:
                comb_acc["atk_ap50"][a]["m1"] = hs_pm1_tot[i1]["atk_ap50"][a]["m1"]
                comb_acc["atk_ap50"][a]["m2"] = hs_pm2_tot[i2]["atk_ap50"][a]["m2"]
            sc = _score_unified_objective(comb_acc, nd_ref)
            rec = OrderedDict({"per_sensor_tau": {"m1": tm1, "m2": tm2}, **sc})
            hs_ps_records.append(rec)
            if best_hs_ps is None or sc["objective_J"] > best_hs_ps["objective_J"]:
                best_hs_ps = rec

    luc_records = []
    best_luc = None
    for i, lth in enumerate(lucia_taus):
        sc = _score_unified_objective(luc_tot[i], nd_ref)
        rec = OrderedDict({"l1_threshold": lth, **sc})
        luc_records.append(rec)
        if best_luc is None or sc["objective_J"] > best_luc["objective_J"]:
            best_luc = rec

    grid_records_doc = OrderedDict({
        "meta": {
            "claim": "[临时结论 · 待人工代码复核]",
            "stage": "M7_select_600_unified_grid_search",
            "select_sequences": seq_order,
            "select_num_frames": total_frames,
            "unified_objective_formula": UNIFIED_OBJECTIVE_FORMULA,
            "cp_uniguard_note": "按规范第零节第2条，CP-uniGuard 今晚不接入，留待明早人工复核后单独接入。",
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "no_defense_select_reference": {k: {m: round(v, 5) for m, v in d.items()} for k, d in nd_ref.items()},
        "bud_fusion_search": {
            "search_space": bud_search_space,
            "locked_m6_switches": m6_switches,
            "num_configs_evaluated": len(bud_records),
            "chosen_config": chosen_bud,
            "all_records": bud_records,
        },
    })
    _atomic_write_json(args.out_grid_records_json, grid_records_doc)

    baselines_doc = OrderedDict({
        "meta": {
            "claim": "[临时结论 · 待人工代码复核]",
            "select_sequences": seq_order,
            "select_num_frames": total_frames,
            "unified_objective_formula": UNIFIED_OBJECTIVE_FORMULA,
            "hetshield_paper_default_note": "HetShield 论文默认 tau_th=0.3 版严格固定不调参。",
        },
        "hetshield_paper_fixed": {"tau_th": 0.3, "tuned": False},
        "hetshield_tuned": {
            "search_candidates": hs_shared_taus,
            "num_configs_evaluated": len(hs_sh_records),
            "chosen_tau_th": best_hs_sh["tau_th"],
            "chosen_eval": best_hs_sh,
            "all_records": hs_sh_records,
        },
        "hetshield_persensor": {
            "search_candidates": {"m1": hs_per_taus_m1, "m2": hs_per_taus_m2},
            "num_configs_evaluated": len(hs_ps_records),
            "chosen_per_sensor_tau": best_hs_ps["per_sensor_tau"],
            "chosen_eval": best_hs_ps,
            "all_records": hs_ps_records,
        },
        "lucia": {
            "search_candidates": lucia_taus,
            "num_configs_evaluated": len(luc_records),
            "chosen_l1_threshold": best_luc["l1_threshold"],
            "chosen_eval": best_luc,
            "all_records": luc_records,
        },
    })
    _atomic_write_json(args.out_baselines_tuned_json, baselines_doc)

    print(
        f"\n[M7 完成] select={total_frames}帧 | BUD最优J={chosen_bud['objective_J']} | "
        f"HS_Tuned={best_hs_sh['tau_th']} (J={best_hs_sh['objective_J']}) | "
        f"HS_Per={best_hs_ps['per_sensor_tau']} (J={best_hs_ps['objective_J']}) | "
        f"LUCIA={best_luc['l1_threshold']} (J={best_luc['objective_J']}) | 耗时={time.time() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
