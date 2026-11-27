# -*- coding: utf-8 -*-
"""
M5（科学执行顺序 ①）：在 PNG 标定组 calib（≥3,000 帧）上计算标定统计量，
分别产出旧前景规则 v0 与新前景规则 v1 的完整自包含标定表（及供 M6 A1 对照的旧可见性对照表）。

严格满足全部规范与低内存保证：
1. 仅读取 results_v3/splits_v3.json 的 calib 组（26 条序列，3,679 帧），
   强制断言与 select / dev / validate / test / test_culvercity 零重叠；
2. 严格流式低内存设计（内存峰值 < 4 GB）：
   - 按序列落盘缓存 _m5_calib_seq_cache/{seq}.pt，不在内存中常驻全部 26 条序列的 seq_payloads 或全量 calib_records；
   - 通过可重入逐序列流式生成器配合逐通道统计，单序列用完即释放；
3. 在 PNG calib 组上实测重新标定 τ_N（clear_pts_tau），绝不使用旧 JPG 时代的硬编码 73.853；
4. 旧前景规则 v0 -> results_v3/bud_calib_v0.json，新前景规则 v1 -> results_v3/bud_calib_v1.json；
5. 每个分档完整输出 grid_count / frame_count / seq_count、2,000 次序列块 Bootstrap 95% CI、
   <20 帧稀疏桶自动向上合并日志，以及整车回退新规则 frame_high_tau_rule = max(calib_clean_max * 1.15, 1.45)。
"""

import argparse
import gc
import json
import os
import sys
import time
from collections import OrderedDict
from typing import Dict, List, Tuple

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
    _find_blind_gts_for_frame,
    _slim_batch_for_postprocess,
)
from src_v2_mm.a5_split_and_calib_stats import (
    build_fg_bg_vis_quantiles_with_ci_and_merging,
    build_hierarchical_channel_stats_with_merging,
)
from src_v2_mm.bud_fusion_heal import (
    BUDFusionHEALAuditor,
    build_fg_mask,
    compute_frame_high_tau_with_rule,
)
from src_v2_mm.heal_base_wrapper import HEALBaseWrapper
from src_v2_mm.step4_1_v2_core import (
    DEFAULT_LOS_OCC_TAU,
    OPV2V_TRAIN_DIR,
    PROTO_H,
    PROTO_W,
    V3_BUD_CALIB_V0_JSON,
    V3_BUD_CALIB_V1_JSON,
    V3_RESULTS_DIR,
    V3_SPLITS_JSON,
    VIS_LABELS_3,
    assign_dist5_vis3_indices_v2,
    build_bev_grid_xy_256,
    calibrate_clear_pts_tau_from_samples,
    compute_dev_maps_from_protocol,
    compute_ego_expected_pts_and_los_256,
    load_ego_lidar_np,
)

DEFAULT_REF_FROZEN = os.path.join(
    PROJ_ROOT if os.path.exists(PROJ_ROOT) else LOCAL_ROOT,
    "results_rq1_v2_mm",
    "step4_1_v2_frozen_calibrator.json",
)
DEFAULT_SEQ_CACHE_DIR = os.path.join(V3_RESULTS_DIR, "_m5_calib_seq_cache")


def _atomic_write_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _default_frozen_methods_template(ref_path: str) -> dict:
    if os.path.exists(ref_path):
        with open(ref_path, "r", encoding="utf-8") as f:
            return json.load(f)["frozen_methods"]
    return {
        "main_ms_cos_clear_sem": {
            "cfg": {
                "q_scale": 95,
                "cos_weights": [0.45, 0.35, 0.20],
                "gamma_cos": [1.20, 0.90, 0.55],
                "gamma_sem": [1.30, 0.40, 0.00],
                "w_sem": 0.35,
                "top_k": 48,
                "frame_norm_mode": "div_p95",
            }
        }
    }


def _extract_or_load_calib_sequence_to_disk(
    wrapper: HEALBaseWrapper,
    ds_m1,
    ds_m2,
    seq_name: str,
    indices: List[int],
    gx: torch.Tensor,
    gy: torch.Tensor,
    cache_dir: str,
) -> Tuple[str, List[float], List[float], int]:
    """
    按单条序列提取干净帧的几何量、4通道偏差图与预测热力图并保存为磁盘 .pt 分片，
    立即释放张量内存，仅向主流程返回 (pt_path, unocc_blind_pts, unocc_hit_pts, excl_self_tot)！
    """
    os.makedirs(cache_dir, exist_ok=True)
    c_path = os.path.join(cache_dir, f"{seq_name}.pt")
    if os.path.exists(c_path):
        cached = torch.load(c_path, map_location="cpu")
        if cached.get("indices") == list(indices):
            ub = list(cached["unocc_blind_pts"])
            uh = list(cached["unocc_hit_pts"])
            ex = int(cached["excluded_ego_self_cnt"])
            del cached
            print(f"  [M5 缓存命中] 序列 {seq_name} ({len(indices)} 帧)", flush=True)
            return c_path, ub, uh, ex
        del cached

    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
    from opencood.tools import train_utils
    from torch.utils.data import DataLoader, Subset

    t_s = time.time()
    frames_out = []
    unocc_blind_pts = []
    unocc_hit_pts = []
    excl_self_tot = 0
    ac = wrapper.model.pyramid_backbone.align_corners

    nw = 2 if len(indices) >= 8 else 0
    dl_m1 = DataLoader(Subset(ds_m1, indices), batch_size=1, num_workers=nw, shuffle=False, collate_fn=ds_m1.collate_batch_test)
    dl_m2 = DataLoader(Subset(ds_m2, indices), batch_size=1, num_workers=nw, shuffle=False, collate_fn=ds_m2.collate_batch_test)

    for idx, b1_cpu, b2_cpu in zip(indices, dl_m1, dl_m2):
        b1 = train_utils.to_device(b1_cpu, wrapper.device_obj)
        b2 = train_utils.to_device(b2_cpu, wrapper.device_obj)
        b_post = train_utils.to_device(_slim_batch_for_postprocess(b1_cpu), wrapper.device_obj)

        ego_pts = load_ego_lidar_np(ds_m1, idx)
        sigma_alea, los_factor, expected_pts = compute_ego_expected_pts_and_los_256(ego_pts, gx, gy, num_beams=64)

        ptm = b1["ego"]["pairwise_t_matrix"][0, 1, 0].detach().cpu().numpy()
        cx, cy = float(ptm[0, 3]), float(ptm[1, 3])
        dist_cav = torch.sqrt((gx - cx) ** 2 + (gy - cy) ** 2).view(1, 1, PROTO_H, PROTO_W)

        p1 = wrapper.extract_protocol_features(b1["ego"])
        p2 = wrapper.extract_protocol_features(b2["ego"])

        cov_ego = warp_affine_simple(p2["cam_valid_mask"][1:2].float(), p2["affine_matrix"][0, 0, 1:2], (PROTO_H, PROTO_W), align_corners=ac) > 0.5
        cov_m1_full = warp_affine_simple(p1["cam_valid_mask"][1:2].float(), p1["affine_matrix"][0, 0, 1:2], (PROTO_H, PROTO_W), align_corners=ac) > 0.5

        with torch.no_grad():
            o_ego = wrapper.forward_single_agent(p1["heter_feature_2d"][0:1])
            p_ego_cls = torch.sigmoid(o_ego["cls_preds"]).max(dim=1, keepdim=True)[0]
            ego_pb, ego_ps, ego_gb = wrapper.decode_boxes(b_post, o_ego, ds_m1)

            o1 = wrapper.forward_from_protocol(p1["heter_feature_2d"], p1["record_len"], p1["affine_matrix"], p1["agent_modality_list"])
            o2 = wrapper.forward_from_protocol(p2["heter_feature_2d"], p2["record_len"], p2["affine_matrix"], p2["agent_modality_list"])
            pf1 = torch.sigmoid(o1["cls_preds"]).max(dim=1, keepdim=True)[0]
            pf2 = torch.sigmoid(o2["cls_preds"]).max(dim=1, keepdim=True)[0]

            dev_m1 = compute_dev_maps_from_protocol(wrapper, p1, p1["heter_feature_2d"], p_ego_cls, cov_m1_full)
            dev_m2 = compute_dev_maps_from_protocol(wrapper, p2, p2["heter_feature_2d"], p_ego_cls, cov_ego)

        binfo = _find_blind_gts_for_frame(ds_m1, idx, ego_pb, ego_ps, ego_gb, cov_m1_full[0, 0])
        excl_self_tot += int(binfo.get("excluded_ego_self", 0))
        los_2d = los_factor[0, 0].detach().cpu()
        exp_2d = expected_pts[0, 0].detach().cpu()
        for iy, ix in binfo["blind_yx"]:
            if float(los_2d[iy, ix]) >= DEFAULT_LOS_OCC_TAU:
                unocc_blind_pts.append(float(exp_2d[iy, ix]))
        for iy, ix in binfo["ego_hit_yx"]:
            if float(los_2d[iy, ix]) >= DEFAULT_LOS_OCC_TAU:
                unocc_hit_pts.append(float(exp_2d[iy, ix]))

        frames_out.append({
            "sample_idx": int(idx),
            "dist_cav": dist_cav[0, 0].half().cpu(),
            "los_2d": los_2d.half(),
            "exp_2d": exp_2d.half(),
            "sigma_alea_2d": sigma_alea[0, 0].half().cpu(),
            "cov_m1": cov_m1_full[0, 0].bool().cpu(),
            "cov_m2": cov_ego[0, 0].bool().cpu(),
            "p_ego_2d": p_ego_cls[0, 0].half().cpu(),
            "pf1_2d": pf1[0, 0].half().cpu(),
            "pf2_2d": pf2[0, 0].half().cpu(),
            "dev_m1": dev_m1.half().cpu(),
            "dev_m2": dev_m2.half().cpu(),
        })

    payload = {
        "seq_name": seq_name,
        "indices": list(indices),
        "unocc_blind_pts": unocc_blind_pts,
        "unocc_hit_pts": unocc_hit_pts,
        "excluded_ego_self_cnt": excl_self_tot,
        "frames": frames_out,
    }
    tmp_p = c_path + ".tmp"
    torch.save(payload, tmp_p)
    os.replace(tmp_p, c_path)
    del payload, frames_out
    gc.collect()
    print(f"  [M5 抽取落盘完成] 序列 {seq_name}: {len(indices)} 帧 ({time.time() - t_s:.1f}s)", flush=True)
    return c_path, unocc_blind_pts, unocc_hit_pts, excl_self_tot


def _build_one_calibrator_streaming(
    seq_pt_paths: Dict[str, str],
    calib_seq_order: List[str],
    total_frames: int,
    vis_v2_cfg: dict,
    frozen_methods_tmpl: dict,
    fg_mode: str,
    fg_prob_tau: float,
    min_bin_frames: int,
    n_boot: int,
    use_legacy_vis_for_a1_before: bool = False,
) -> OrderedDict:
    """
    严格流式构建完整自包含标定表：逐序列从磁盘加载、处理完单序列即释放，全程无全量帧常驻内存。
    """
    c_tau = float(vis_v2_cfg["clear_pts_tau"])
    l_tau = float(vis_v2_cfg["los_occ_tau"])

    def _stream_frames():
        for s_name in calib_seq_order:
            pl = torch.load(seq_pt_paths[s_name], map_location="cpu")
            for fr in pl["frames"]:
                dist_4d = fr["dist_cav"].float().view(1, 1, PROTO_H, PROTO_W)
                los_4d = fr["los_2d"].float().view(1, 1, PROTO_H, PROTO_W)
                exp_4d = fr["exp_2d"].float().view(1, 1, PROTO_H, PROTO_W)
                d_4d, v_new_4d = assign_dist5_vis3_indices_v2(dist_4d, los_4d, exp_4d, clear_pts_tau=c_tau, los_occ_tau=l_tau)

                if use_legacy_vis_for_a1_before:
                    sa = fr["sigma_alea_2d"].float()
                    v_stat = torch.ones_like(sa, dtype=torch.long)
                    v_stat[sa < 0.33] = 0
                    v_stat[sa >= 0.67] = 2
                else:
                    v_stat = v_new_4d[0, 0]

                pe = fr["p_ego_2d"].float().view(1, 1, PROTO_H, PROTO_W)
                pf1 = fr["pf1_2d"].float().view(1, 1, PROTO_H, PROTO_W)
                pf2 = fr["pf2_2d"].float().view(1, 1, PROTO_H, PROTO_W)
                fg1 = build_fg_mask(pe, pf1, v_stat, fg_prob_tau=fg_prob_tau, mode=fg_mode)
                fg2 = build_fg_mask(pe, pf2, v_stat, fg_prob_tau=fg_prob_tau, mode=fg_mode)

                yield {
                    "sample_idx": fr["sample_idx"],
                    "seq_name": s_name,
                    "cov_by_mod": {"m1": fr["cov_m1"], "m2": fr["cov_m2"]},
                    "d_idx": d_4d[0, 0],
                    "v_idx": v_stat,
                    "m1": fr["dev_m1"],
                    "m2": fr["dev_m2"],
                    "fg_m1": fg1,
                    "fg_m2": fg2,
                }
            del pl

    ch_stats, merge_diag = build_hierarchical_channel_stats_with_merging(
        _stream_frames, calib_seq_order, min_frames=int(min_bin_frames), n_boot=int(n_boot), seed=20261120
    )

    tmp_base = {"channel_tier_stats": ch_stats, "frozen_methods": frozen_methods_tmpl}
    tmp_path = os.path.join(V3_RESULTS_DIR, f"_tmp_base_{fg_mode}_{int(use_legacy_vis_for_a1_before)}.json")
    _atomic_write_json(tmp_path, tmp_base)
    auditor = BUDFusionHEALAuditor(tmp_path)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    raw_scores = {"t2": {"m1": [], "m2": []}, "t4": {"m1": [], "m2": []}}
    per_seq_bg_fg = {role: {m: {vn: OrderedDict() for vn in VIS_LABELS_3} for m in ("m1", "m2")} for role in ("bg", "fg")}
    fg_bg_frame_cnts = {role: {m: {vn: 0 for vn in VIS_LABELS_3} for m in ("m1", "m2")} for role in ("bg", "fg")}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(20261120)
    max_bg_pix_per_frame = 2048  # 单帧空地像素上限无偏抽样（3679帧合计数百万样本，既精确又防内存膨胀）

    for rec in _stream_frames():
        s_name = rec["seq_name"]
        d_g = rec["d_idx"].to(device)
        v_g = rec["v_idx"].to(device)
        for mod, fg_k in (("m1", "fg_m1"), ("m2", "fg_m2")):
            cov_g = rec["cov_by_mod"][mod].to(device)
            fg_g = rec[fg_k].to(device)
            dev_g = rec[mod].to(device=device, dtype=torch.float32)
            ar = auditor.audit_dev_maps(dev_g, cov_g, d_g, v_g, mod, build_gate=False)
            raw_scores["t2"][mod].append(ar["raw_t2"])
            raw_scores["t4"][mod].append(ar["raw_t4"])
            cm_s = F.avg_pool2d(ar["cos_comb_t4"].view(1, 1, PROTO_H, PROTO_W), kernel_size=3, stride=1, padding=1)[0, 0]
            for vi, vn in enumerate(VIS_LABELS_3):
                m_bg = cov_g & (v_g == vi) & (~fg_g)
                m_fg = cov_g & (v_g == vi) & fg_g
                if m_bg.any():
                    arr_bg = cm_s[m_bg].detach().cpu().numpy().astype(np.float32)
                    if arr_bg.size > max_bg_pix_per_frame:
                        arr_bg = arr_bg[rng.choice(arr_bg.size, size=max_bg_pix_per_frame, replace=False)]
                    per_seq_bg_fg["bg"][mod][vn].setdefault(s_name, []).append(arr_bg)
                    fg_bg_frame_cnts["bg"][mod][vn] += 1
                if m_fg.any():
                    per_seq_bg_fg["fg"][mod][vn].setdefault(s_name, []).append(cm_s[m_fg].detach().cpu().numpy().astype(np.float32))
                    fg_bg_frame_cnts["fg"][mod][vn] += 1

    frame_div_p95 = OrderedDict({
        "tier2_sensor_only": {m: round(max(float(np.percentile(raw_scores["t2"][m], 95.0)), 1e-4), 6) for m in ("m1", "m2")},
        "tier4_sensor_dist_vis": {m: round(max(float(np.percentile(raw_scores["t4"][m], 95.0)), 1e-4), 6) for m in ("m1", "m2")},
    })
    cos_q_bg, cos_q_fg, merged_q_log = build_fg_bg_vis_quantiles_with_ci_and_merging(
        per_seq_bg_fg, fg_bg_frame_cnts, calib_seq_order, min_frames=int(min_bin_frames), n_boot=int(n_boot), seed=20261120
    )
    frame_high_tau, frame_high_tau_rule = compute_frame_high_tau_with_rule(
        raw_scores["t4"], frame_div_p95["tier4_sensor_dist_vis"], safety_mult=1.15, lower_bound=1.45
    )

    bg_q_init, fg_q_init = "p99.8", "p99.0"
    cos_tau_bg_init = {m: [cos_q_bg[m][vn][bg_q_init] for vn in VIS_LABELS_3] for m in ("m1", "m2")}
    cos_tau_fg_init = {m: [max(cos_q_bg[m][vn][bg_q_init], cos_q_fg[m][vn][fg_q_init]) for vn in VIS_LABELS_3] for m in ("m1", "m2")}
    init_gating_params = OrderedDict({
        "mode": "directional_vis_fg_bg",
        "fg_mode": fg_mode,
        "fg_rule": fg_mode,
        "k_tol": 5,
        "fg_prob_tau": float(fg_prob_tau),
        "bg_q": bg_q_init,
        "fg_q": fg_q_init,
        "cos_tau_by_vis_bg": cos_tau_bg_init,
        "cos_tau_by_vis_fg": cos_tau_fg_init,
        "gamma_cos_gate": [1.0, 1.0, 1.0],
        "cos_co_ratio": 0.82,
        "drop_tau": 0.40,
        "add_tau": 0.30,
        "w_drop": 7.0,
        "w_add": 6.0,
        "gamma_add": [1.0, 0.3, 0.0],
        "r_dil": 3,
        "beta": 3.5,
    })

    return OrderedDict({
        "meta": {
            "claim": "[临时结论 · 待人工代码复核]",
            "stage": "M5_calib_statistics",
            "fg_mode": fg_mode,
            "visibility_fn": "legacy_sigma_alea_033_067_for_a1_ablation_only" if use_legacy_vis_for_a1_before else "assign_dist5_vis3_indices_v2",
            "calib_num_sequences": len(calib_seq_order),
            "calib_num_frames": int(total_frames),
            "bootstrap_resamples": int(n_boot),
            "channel_bin_merging_summary": merge_diag,
            "fg_bg_bin_merging_log": merged_q_log,
        },
        "visibility_v2": vis_v2_cfg,
        "channel_tier_stats": ch_stats,
        "frozen_methods": frozen_methods_tmpl,
        "frame_div_p95": frame_div_p95,
        "frame_high_tau": frame_high_tau,
        "frame_high_tau_rule": frame_high_tau_rule,
        "pixel_cos_by_vis_bg_quantiles": cos_q_bg,
        "pixel_cos_by_vis_fg_quantiles": cos_q_fg,
        "gating_params": init_gating_params,
    })


def main():
    parser = argparse.ArgumentParser(description="M5：PNG 标定组 (calib >= 3,000 帧) 流式低内存生成 v0/v1 双标定表与新 τ_N")
    parser.add_argument("--data_dir", type=str, default=OPV2V_TRAIN_DIR)
    parser.add_argument("--splits_json", type=str, default=V3_SPLITS_JSON)
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_SEQ_CACHE_DIR)
    parser.add_argument("--out_v0_json", type=str, default=V3_BUD_CALIB_V0_JSON)
    parser.add_argument("--out_v1_json", type=str, default=V3_BUD_CALIB_V1_JSON)
    parser.add_argument("--out_a1_before_json", type=str, default=os.path.join(V3_RESULTS_DIR, "bud_calib_a1_before.json"))
    parser.add_argument("--fg_prob_tau", type=float, default=0.20)
    parser.add_argument("--min_bin_frames", type=int, default=20)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--max_frames", type=int, default=0, help=">0 时仅跑前 N 帧用于 M4 2帧冒烟测试")
    parser.add_argument("--ref_frozen_json", type=str, default=DEFAULT_REF_FROZEN)
    args = parser.parse_args()

    t0 = time.time()
    with open(args.splits_json, "r", encoding="utf-8") as f:
        sp = json.load(f)

    calib_seqs = list(sp["groups"]["calib"])
    other_seqs = (
        set(sp["groups"].get("select", []))
        | set(sp["groups"].get("dev", []))
        | set(sp["groups"].get("validate", []))
        | set(sp["groups"].get("test", []))
        | set(sp["groups"].get("test_culvercity", []))
    )
    assert set(calib_seqs).isdisjoint(other_seqs), "硬停机拦截：calib 序列与其他划分存在重叠！"

    calib_sub = sp["subsets"]["calib"]["sequences"]
    is_smoke = int(args.max_frames) > 0
    rem = int(args.max_frames) if is_smoke else None
    seq_plan = OrderedDict()
    for s in calib_seqs:
        if rem is not None and rem <= 0:
            break
        s_entry = calib_sub[s]
        idxs = list(s_entry.get("indices", s_entry.get("sampled_indices", [])))
        if rem is not None:
            idxs = idxs[:rem]
            rem -= len(idxs)
        if idxs:
            seq_plan[s] = idxs

    total_frames = sum(len(v) for v in seq_plan.values())
    if not is_smoke:
        assert total_frames >= 3000, f"标定组帧数 {total_frames} < 3,000，不满足规范要求！"

    eff_cache_dir = os.path.join(os.path.dirname(args.out_v1_json), "_smoke_m5_cache") if is_smoke else args.cache_dir

    wrapper = HEALBaseWrapper(device="cuda")
    ds_m1 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m1")
    ds_m2 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m2")
    gx, gy = build_bev_grid_xy_256(device="cuda", dtype=torch.float32)

    seq_pt_paths = OrderedDict()
    all_unocc_blind, all_unocc_hit = [], []
    tot_excl_self = 0
    for s, idxs in seq_plan.items():
        pt_p, ub, uh, ex = _extract_or_load_calib_sequence_to_disk(
            wrapper, ds_m1, ds_m2, s, idxs, gx, gy, cache_dir=eff_cache_dir
        )
        seq_pt_paths[s] = pt_p
        all_unocc_blind.extend(ub)
        all_unocc_hit.extend(uh)
        tot_excl_self += int(ex)

    vis_v2_cfg = calibrate_clear_pts_tau_from_samples(
        all_unocc_blind, all_unocc_hit, los_occ_tau=DEFAULT_LOS_OCC_TAU, excluded_ego_self_cnt=tot_excl_self
    )
    print(f"\n[M5 新 τ_N 标定结果] {json.dumps(vis_v2_cfg, ensure_ascii=False)}", flush=True)

    frozen_tmpl = _default_frozen_methods_template(args.ref_frozen_json)
    active_seqs = list(seq_plan.keys())
    n_boot_eff = 50 if is_smoke else int(args.n_boot)

    print("[M5] 正在流式统计旧前景规则 v0 标定表...", flush=True)
    calib_v0 = _build_one_calibrator_streaming(
        seq_pt_paths, active_seqs, total_frames, vis_v2_cfg, frozen_tmpl, fg_mode="v0",
        fg_prob_tau=args.fg_prob_tau, min_bin_frames=args.min_bin_frames, n_boot=n_boot_eff,
    )
    _atomic_write_json(args.out_v0_json, calib_v0)

    print("[M5] 正在流式统计新前景规则 v1 标定表...", flush=True)
    calib_v1 = _build_one_calibrator_streaming(
        seq_pt_paths, active_seqs, total_frames, vis_v2_cfg, frozen_tmpl, fg_mode="v1",
        fg_prob_tau=args.fg_prob_tau, min_bin_frames=args.min_bin_frames, n_boot=n_boot_eff,
    )
    _atomic_write_json(args.out_v1_json, calib_v1)

    print("[M5] 正在流式统计供 M6 A1 对照的旧可见性统计表...", flush=True)
    calib_a1_before = _build_one_calibrator_streaming(
        seq_pt_paths, active_seqs, total_frames, vis_v2_cfg, frozen_tmpl, fg_mode="v1",
        fg_prob_tau=args.fg_prob_tau, min_bin_frames=args.min_bin_frames, n_boot=n_boot_eff,
        use_legacy_vis_for_a1_before=True,
    )
    _atomic_write_json(args.out_a1_before_json, calib_a1_before)

    if is_smoke and os.path.isdir(eff_cache_dir):
        import shutil
        shutil.rmtree(eff_cache_dir, ignore_errors=True)

    print(
        f"\n[M5 完成] 标定帧数={total_frames}, clear_pts_tau={vis_v2_cfg['clear_pts_tau']}, "
        f"frame_high_tau(v1)={calib_v1['frame_high_tau']}, 耗时={time.time() - t0:.1f}s\n"
        f"  - v0: {args.out_v0_json}\n  - v1: {args.out_v1_json}\n  - a1_before: {args.out_a1_before_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
