# -*- coding: utf-8 -*-
"""
步骤 ⑦（最终收口）：独立验证集 OPV2V_VAL 重标定外部门限 + 严密锁控的一次性封测测试集入口。

严格零泄露规范：
1. --mode val_recalib：
   - 仅访问 data_links/opv2v_val（6 条独立验证序列，全含 Q95 JPG），绝不触碰 opv2v_test；
   - 不修改任何模型权重与超参结构，严格沿用 Fit 阶段冻结的 BUD-Fusion 门控结构与 HetShield 权重，
     仅在 OPV2V_VAL 干净数据上重标定外部 JSON 基准：
     (a) 4 通道 (传感器 × 5距离 × 3可见性) 分档基准 -> step8_val_base_calibrator.json
     (b) 帧级稳健 div_p95、整车回退高门限 frame_high_tau 与分可见性余弦死区 cos_tau_by_vis -> step8_val_bud_calibrator.json
     (c) HetShield 在 OPV2V_VAL 连续帧上的 Fit-Tuned 与 PerSensor 95% 门限 -> step8_val_hetshield_calib.json；
2. --mode final_test_once：
   - 带双重安全防误触锁：必须显式传入 --i_confirm_final_frozen_test YES，且校验所有 val 重标定 JSON 存在、
     最终测试输出目录尚未生成结果（防止重复测试刷榜！）；
   - 调用已验证的 step4_5_eval_bud_frozen_320.py 与 step5_2_eval_multicav_and_baselines.py 在 opv2v_test 上一次性出表。
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

PROJ_ROOT = "/root/autodl-tmp/proj_dir2_hetero_trust"
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from src_v2_mm.bud_fusion_heal import BUDFusionHEALAuditor, build_fg_mask
from src_v2_mm.heal_base_wrapper import HEALBaseWrapper
from src_v2_mm.hetshield_heal import HetShieldHEALDefender
from src_v2_mm.step4_1_v2_core import (
    CHANNEL_NAMES,
    DIST_LABELS_5,
    PROTO_H,
    PROTO_W,
    VIS_LABELS_3,
    assign_dist5_vis3_indices_v2,
    build_bev_grid_xy_256,
    compute_dev_maps_from_protocol,
    compute_ego_expected_pts_and_los_256,
    extract_frame_context_and_clean,
    load_ego_lidar_np,
)

VAL_DIR = os.path.join(PROJ_ROOT, "data_links", "opv2v_val")
TEST_DIR = os.path.join(PROJ_ROOT, "data_links", "opv2v_test")
RES_DIR = os.path.join(PROJ_ROOT, "results_v3")

DEFAULT_FIT_BASE = os.path.join(RES_DIR, "a1_fit_base_calibrator.json")
DEFAULT_FIT_GATING = os.path.join(RES_DIR, "b_frozen_bud_calibrator.json")
DEFAULT_FIT_HS = os.path.join(RES_DIR, "a2_hetshield_calib.json")

OUT_VAL_BASE = os.path.join(RES_DIR, "c_val_base_calibrator.json")
OUT_VAL_BUD = os.path.join(RES_DIR, "c_val_bud_calibrator.json")
OUT_VAL_HS = os.path.join(RES_DIR, "c_val_hetshield_calib.json")
OUT_TEST_DIR = os.path.join(RES_DIR, "final_opv2v_test_once")


def _sample_balanced_from_dir(wrapper, ds, target_frames: int = 240) -> tuple:
    """从全部序列按比例均衡抽样用于标定（或全量），返回 (seq_order, seq_plan, all_indices)。"""
    sample_seqs = wrapper.get_sample_sequences(ds)
    seq_to_idxs = OrderedDict()
    for i, s in enumerate(sample_seqs):
        seq_to_idxs.setdefault(s, []).append(i)
    seq_order = list(seq_to_idxs.keys())
    total = len(sample_seqs)

    all_picked = []
    seq_plan = OrderedDict()
    for s in seq_order:
        idxs = seq_to_idxs[s]
        if target_frames <= 0 or target_frames >= total:
            picked = list(idxs)
        else:
            n_p = max(12, int(round(float(target_frames) * len(idxs) / float(total))))
            n_p = min(n_p, len(idxs))
            pos = np.linspace(0, len(idxs) - 1, n_p, dtype=int)
            picked = [int(idxs[p]) for p in pos]
        all_picked.extend(picked)
        seq_plan[s] = {"total_frames": len(idxs), "sampled": len(picked), "indices": picked}
    return seq_order, seq_plan, all_picked


def _quantiles_dict(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"median": 0.0, "p75": 0.05, "p90": 0.1, "p95": 0.15, "p99": 0.2, "count": 0}
    qs = np.percentile(arr, [50.0, 75.0, 90.0, 95.0, 99.0])
    return {
        "median": round(float(qs[0]), 6),
        "p75": round(float(qs[1]), 6),
        "p90": round(float(qs[2]), 6),
        "p95": round(float(qs[3]), 6),
        "p99": round(float(qs[4]), 6),
        "count": int(arr.size),
    }


def run_val_recalibration(args):
    """仅使用 OPV2V_VAL 干净帧重标定全部外部 JSON 门限（严禁访问 opv2v_test）。"""
    t0 = time.time()
    assert os.path.isdir(VAL_DIR), f"未找到验证集目录：{VAL_DIR}"
    with open(args.fit_base_json, "r", encoding="utf-8") as f:
        fit_base = json.load(f)
    with open(args.fit_gating_json, "r", encoding="utf-8") as f:
        fit_gating = json.load(f)
    with open(args.fit_hs_json, "r", encoding="utf-8") as f:
        fit_hs = json.load(f)

    wrapper = HEALBaseWrapper(device="cuda")
    ds_m1 = wrapper.build_dataset(data_dir=VAL_DIR, use_cav=2, visualize=True, collab_modality="m1")
    ds_m2 = wrapper.build_dataset(data_dir=VAL_DIR, use_cav=2, visualize=True, collab_modality="m2")
    gx, gy = build_bev_grid_xy_256(device="cuda", dtype=torch.float32)

    seq_order, seq_plan, val_idxs = _sample_balanced_from_dir(wrapper, ds_m2, target_frames=args.val_target_frames)
    print(f"[Step 8 · VAL 重标定] 序列数={len(seq_order)}, 抽样帧数={len(val_idxs)} (来源: {VAL_DIR})", flush=True)

    # 1. 收集 4 通道在 OPV2V_VAL 上的分档分布（子采样每帧最多 4096 有效像素，省内存且无偏）
    rng = np.random.default_rng(20261121)
    ch_buf_unified = {c: [] for c in CHANNEL_NAMES}
    ch_buf_sensor = {c: {"m1": [], "m2": []} for c in CHANNEL_NAMES}
    ch_buf_sd = {c: {m: {dl: [] for dl in DIST_LABELS_5} for m in ("m1", "m2")} for c in CHANNEL_NAMES}
    ch_buf_sdv = {
        c: {m: {dl: {vl: [] for vl in VIS_LABELS_3} for dl in DIST_LABELS_5} for m in ("m1", "m2")}
        for c in CHANNEL_NAMES
    }

    vis_v2 = fit_gating.get("visibility_v2", {"clear_pts_tau": 73.853, "los_occ_tau": 0.50})
    fg_prob_tau = float(fit_gating.get("gating_params", {}).get("fg_prob_tau", 0.20))

    cached_val_maps = []
    for rank, idx in enumerate(val_idxs):
        b = extract_frame_context_and_clean(wrapper, ds_m1, ds_m2, idx, gx, gy)
        ego_pts = load_ego_lidar_np(ds_m1, idx)
        _, los_f, exp_pts = compute_ego_expected_pts_and_los_256(ego_pts, gx, gy, num_beams=64)
        d_4d, v_4d = assign_dist5_vis3_indices_v2(
            b["dist_cav"], los_f, exp_pts,
            clear_pts_tau=float(vis_v2["clear_pts_tau"]),
            los_occ_tau=float(vis_v2.get("los_occ_tau", 0.50)),
        )
        d_idx = d_4d[0, 0]
        v_idx = v_4d[0, 0]
        d_np = d_idx.detach().cpu().numpy()
        v_np = v_idx.detach().cpu().numpy()

        # 主实验规范：m1 在自身完整激光雷达覆盖区标定，m2 在摄像头有效覆盖区标定
        cov_m1_4d = b["cov_ego_m1_full"] if "cov_ego_m1_full" in b else b["cov_ego"]
        cov_m2_4d = b["cov_ego"]
        dev_m1 = compute_dev_maps_from_protocol(wrapper, b["p1"], b["p1"]["heter_feature_2d"], b["p_ego_cls"], cov_m1_4d)
        dev_m2 = b["dev_clean"]["m2"]

        with torch.no_grad():
            o1 = wrapper.forward_from_protocol(b["p1"]["heter_feature_2d"], b["p1"]["record_len"], b["p1"]["affine_matrix"], b["p1"]["agent_modality_list"])
            o2 = wrapper.forward_from_protocol(b["p2"]["heter_feature_2d"], b["p2"]["record_len"], b["p2"]["affine_matrix"], b["p2"]["agent_modality_list"])
            pf1 = torch.sigmoid(o1["cls_preds"]).max(dim=1, keepdim=True)[0]
            pf2 = torch.sigmoid(o2["cls_preds"]).max(dim=1, keepdim=True)[0]
            fg1 = build_fg_mask(b["p_ego_cls"], pf1, v_idx, fg_prob_tau=fg_prob_tau, mode="v1")
            fg2 = build_fg_mask(b["p_ego_cls"], pf2, v_idx, fg_prob_tau=fg_prob_tau, mode="v1")

        for mod, dev_t, cov_4d in (("m1", dev_m1, cov_m1_4d), ("m2", dev_m2, cov_m2_4d)):
            cov_np = cov_4d[0, 0].detach().cpu().numpy()
            dev_np = dev_t.detach().cpu().numpy()
            ys, xs = np.nonzero(cov_np)
            if ys.size > 4096:
                sel = rng.choice(ys.size, size=4096, replace=False)
                ys, xs = ys[sel], xs[sel]
            dv_sel = d_np[ys, xs]
            vv_sel = v_np[ys, xs]

            for ci, cname in enumerate(CHANNEL_NAMES):
                vals = dev_np[ci, ys, xs].astype(np.float32)
                ch_buf_unified[cname].append(vals)
                ch_buf_sensor[cname][mod].append(vals)
                for di, dl in enumerate(DIST_LABELS_5):
                    md = dv_sel == di
                    if md.any():
                        ch_buf_sd[cname][mod][dl].append(vals[md])
                        for vi, vl in enumerate(VIS_LABELS_3):
                            mv = md & (vv_sel == vi)
                            if mv.any():
                                ch_buf_sdv[cname][mod][dl][vl].append(vals[mv])

        cached_val_maps.append({
            "idx": idx,
            "cov_m1": cov_m1_4d[0, 0].detach().cpu(),
            "cov_m2": cov_m2_4d[0, 0].detach().cpu(),
            "d_idx": d_idx.detach().cpu(),
            "v_idx": v_idx.detach().cpu(),
            "fg_m1": fg1.detach().cpu(),
            "fg_m2": fg2.detach().cpu(),
            "dev_m1": dev_m1.half().cpu(),
            "dev_m2": dev_m2.half().cpu(),
        })
        if (rank + 1) % 40 == 0 or (rank + 1) == len(val_idxs):
            print(f"  [VAL Pass 1] 进度 {rank + 1}/{len(val_idxs)} ({time.time() - t0:.1f}s)", flush=True)

    # 汇总新 4 通道基准（严格对齐 step4_1_v2_core 查表键名 "sensor"）
    new_tier_stats = OrderedDict()
    for cname in CHANNEL_NAMES:
        st_c = OrderedDict()
        st_c["unified"] = _quantiles_dict(np.concatenate(ch_buf_unified[cname]))
        st_c["sensor"] = {m: _quantiles_dict(np.concatenate(ch_buf_sensor[cname][m])) for m in ("m1", "m2")}
        st_c["sensor_dist"] = {
            m: {dl: _quantiles_dict(np.concatenate(ch_buf_sd[cname][m][dl])) if ch_buf_sd[cname][m][dl] else st_c["sensor"][m]
                for dl in DIST_LABELS_5}
            for m in ("m1", "m2")
        }
        st_c["sensor_dist_vis"] = {
            m: {
                dl: {
                    vl: _quantiles_dict(np.concatenate(ch_buf_sdv[cname][m][dl][vl])) if ch_buf_sdv[cname][m][dl][vl] else st_c["sensor_dist"][m][dl]
                    for vl in VIS_LABELS_3
                }
                for dl in DIST_LABELS_5
            }
            for m in ("m1", "m2")
        }
        new_tier_stats[cname] = st_c

    val_base_obj = copy.deepcopy(fit_base)
    val_base_obj.setdefault("meta", {})["recalibrated_on"] = VAL_DIR
    val_base_obj["meta"]["val_frames"] = len(val_idxs)
    val_base_obj["channel_tier_stats"] = new_tier_stats
    os.makedirs(os.path.dirname(OUT_VAL_BASE), exist_ok=True)
    with open(OUT_VAL_BASE, "w", encoding="utf-8") as f:
        json.dump(val_base_obj, f, indent=2, ensure_ascii=False)
    print(f"  已保存 VAL 4通道基准: {OUT_VAL_BASE}")

    # 2. 基于新 VAL 4通道基准，按原冻结分位档重算 div_p95、frame_high_tau 与前景/背景分档 cos_tau
    auditor = BUDFusionHEALAuditor(OUT_VAL_BASE)
    raw_t2 = {"m1": [], "m2": []}
    raw_t4 = {"m1": [], "m2": []}
    pix_cos_bg = {m: {v: [] for v in range(3)} for m in ("m1", "m2")}
    pix_cos_fg = {m: {v: [] for v in range(3)} for m in ("m1", "m2")}

    for item in cached_val_maps:
        d_idx = item["d_idx"].cuda()
        v_idx = item["v_idx"].cuda()
        for mod, dev_k, cov_k, fg_k in (
            ("m1", "dev_m1", "cov_m1", "fg_m1"),
            ("m2", "dev_m2", "cov_m2", "fg_m2"),
        ):
            cov_2d = item[cov_k].cuda()
            fg_2d = item[fg_k].cuda()
            dev_4ch = item[dev_k].cuda().float()
            ar = auditor.audit_dev_maps(dev_4ch, cov_2d, d_idx, v_idx, mod, build_gate=False)
            raw_t2[mod].append(ar["raw_t2"])
            raw_t4[mod].append(ar["raw_t4"])
            cm_s = F.avg_pool2d(ar["cos_comb_t4"].view(1, 1, PROTO_H, PROTO_W), kernel_size=3, stride=1, padding=1)[0, 0]
            for v_id in range(3):
                vm_bg = cov_2d & (v_idx == v_id) & (~fg_2d)
                vm_fg = cov_2d & (v_idx == v_id) & fg_2d
                if vm_bg.any():
                    pix_cos_bg[mod][v_id].append(cm_s[vm_bg].detach().cpu().numpy().astype(np.float32))
                if vm_fg.any():
                    pix_cos_fg[mod][v_id].append(cm_s[vm_fg].detach().cpu().numpy().astype(np.float32))

    frame_div_p95 = OrderedDict({
        "tier2_sensor_only": {m: round(max(float(np.percentile(raw_t2[m], 95.0)), 1e-4), 6) for m in ("m1", "m2")},
        "tier4_sensor_dist_vis": {m: round(max(float(np.percentile(raw_t4[m], 95.0)), 1e-4), 6) for m in ("m1", "m2")},
    })

    fg = copy.deepcopy(fit_gating["gating_params"])
    q_map = {"p90.0": 90.0, "p95.0": 95.0, "p98.0": 98.0, "p99.0": 99.0, "p99.5": 99.5, "p99.8": 99.8, "p99.9": 99.9, "p99.99": 99.99}
    bg_q_val = q_map.get(fg.get("bg_q", "p99.9"), 99.9)
    fg_q_val = q_map.get(fg.get("fg_q", "p98.0"), 98.0)
    fg["cos_tau_by_vis_bg"] = {
        m: [
            round(float(np.percentile(np.concatenate(pix_cos_bg[m][vi]), bg_q_val)), 6)
            if pix_cos_bg[m][vi] else fg["cos_tau_by_vis_bg"][m][vi]
            for vi in range(3)
        ]
        for m in ("m1", "m2")
    }
    fg["cos_tau_by_vis_fg"] = {
        m: [
            round(float(np.percentile(np.concatenate(pix_cos_fg[m][vi]), fg_q_val)), 6)
            if pix_cos_fg[m][vi] else fg["cos_tau_by_vis_fg"][m][vi]
            for vi in range(3)
        ]
        for m in ("m1", "m2")
    }

    # 整车高门限：取 VAL 干净 score_t4 最大值 * 1.08 与 Fit 冻结高门限的较大值（确保 VAL 干净 0% 误触发）
    frame_high_tau = OrderedDict()
    for m in ("m1", "m2"):
        val_sc = np.asarray(raw_t4[m], dtype=np.float64) / frame_div_p95["tier4_sensor_dist_vis"][m]
        fit_hi = float(fit_gating.get("frame_high_tau", {}).get(m, 1.35))
        frame_high_tau[m] = round(max(float(val_sc.max()) * 1.08, fit_hi), 6)

    val_bud_obj = OrderedDict({
        "meta": {
            "description": "步骤 ⑧ 仅在独立验证集 OPV2V_VAL 重标定的 BUD-Fusion 最终门限（零结构改动）",
            "val_dir": VAL_DIR,
            "val_sequences": seq_order,
            "val_frames": len(val_idxs),
            "source_fit_gating": args.fit_gating_json,
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "visibility_v2": fit_gating.get("visibility_v2"),
        "frame_div_p95": frame_div_p95,
        "frame_high_tau": frame_high_tau,
        "gating_params": fg,
    })
    with open(OUT_VAL_BUD, "w", encoding="utf-8") as f:
        json.dump(val_bud_obj, f, indent=2, ensure_ascii=False)
    print(f"  已保存 VAL BUD-Fusion 门限: {OUT_VAL_BUD}")

    # 3. 同步在 OPV2V_VAL 上按单步预热同协议重标定 HetShield 的 95% 分位门限（保持 ConvGRU 权重完全冻结）
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
    from opencood.tools import train_utils

    e_th = float(fit_hs["meta"]["energy_thresh"])
    alpha_hs = float(fit_hs.get("paper_default_params", {}).get("alpha", 0.6))
    hs_s = HetShieldHEALDefender(mode="shared", alpha=alpha_hs, energy_thresh=e_th).cuda().eval()
    hs_s.predictor.load_state_dict(torch.load(fit_hs["meta"]["shared_ckpt"], map_location="cpu"))
    hs_p = HetShieldHEALDefender(mode="persensor", alpha=alpha_hs, energy_thresh=e_th).cuda().eval()
    per_sd = torch.load(fit_hs["meta"]["persensor_ckpt"], map_location="cpu")
    for m in ("m1", "m2"):
        hs_p.predictors[m].load_state_dict(per_sd[m])

    ac = wrapper.model.pyramid_backbone.align_corners
    sample_seqs_val = wrapper.get_sample_sequences(ds_m2)
    sk_s = {"m1": [], "m2": []}
    sk_p = {"m1": [], "m2": []}

    with torch.no_grad():
        for s in seq_order:
            s_idxs = [i for i, sn in enumerate(sample_seqs_val) if sn == s][:40]
            for mod, ds in (("m1", ds_m1), ("m2", ds_m2)):
                cid = f"{s}_{mod}_c1"
                for pos in range(1, len(s_idxs)):
                    idx_prev, idx_cur = s_idxs[pos - 1], s_idxs[pos]
                    hs_s.reset_sequence(s)
                    hs_p.reset_sequence(s)
                    if int(idx_cur) - int(idx_prev) == 1:
                        rp_b = train_utils.to_device(ds.collate_batch_test([ds[idx_prev]]), wrapper.device_obj)
                        pp = wrapper.extract_protocol_features(rp_b["ego"])
                        tp = pp["affine_matrix"][0, 0, 1:2]
                        fcp = warp_affine_simple(pp["heter_feature_2d"][1:2], tp, (PROTO_H, PROTO_W), align_corners=ac)
                        vcp = warp_affine_simple(pp["cam_valid_mask"][1:2].float(), tp, (PROTO_H, PROTO_W), align_corners=ac)
                        p_items = [{"cav_id": cid, "mod": mod, "feat_warped": fcp, "valid_ego": (vcp > 0.5).float()}]
                        hs_s.step_frame(s, int(idx_prev), pp["heter_feature_2d"][0:1], p_items, update_state=True)
                        hs_p.step_frame(s, int(idx_prev), pp["heter_feature_2d"][0:1], p_items, update_state=True)

                    rc_b = train_utils.to_device(ds.collate_batch_test([ds[idx_cur]]), wrapper.device_obj)
                    pc = wrapper.extract_protocol_features(rc_b["ego"])
                    tc = pc["affine_matrix"][0, 0, 1:2]
                    fcc = warp_affine_simple(pc["heter_feature_2d"][1:2], tc, (PROTO_H, PROTO_W), align_corners=ac)
                    vcc = warp_affine_simple(pc["cam_valid_mask"][1:2].float(), tc, (PROTO_H, PROTO_W), align_corners=ac)
                    c_items = [{"cav_id": cid, "mod": mod, "feat_warped": fcc, "valid_ego": (vcc > 0.5).float()}]
                    rs = hs_s.step_frame(s, int(idx_cur), pc["heter_feature_2d"][0:1], c_items, update_state=False)[0]
                    rp = hs_p.step_frame(s, int(idx_cur), pc["heter_feature_2d"][0:1], c_items, update_state=False)[0]
                    sk_s[mod].append(float(rs["s_k"]))
                    sk_p[mod].append(float(rp["s_k"]))

    val_hs_obj = copy.deepcopy(fit_hs)
    val_hs_obj["meta"]["recalibrated_on"] = VAL_DIR
    val_hs_obj["fit_tuned_shared_tau_p95"] = round(float(np.percentile(sk_s["m1"] + sk_s["m2"], 95.0)), 6)
    val_hs_obj["fit_persensor_tau_p95"] = {
        "m1": round(float(np.percentile(sk_p["m1"], 95.0)), 6),
        "m2": round(float(np.percentile(sk_p["m2"], 95.0)), 6),
    }
    with open(OUT_VAL_HS, "w", encoding="utf-8") as f:
        json.dump(val_hs_obj, f, indent=2, ensure_ascii=False)
    print(f"  已保存 VAL HetShield 门限: {OUT_VAL_HS}")
    print(f"\n[完成] 独立验证集重标定全部完成（总耗时 {time.time() - t0:.1f}s），测试集 opv2v_test 未触碰！")


def export_frozen_params_snapshot(out_path: str, base_path: str, bud_path: str, hs_path: str) -> dict:
    """A3：导出所有被冻结参数的完整 JSON 快照（供论文参数表直接读取）。"""
    with open(base_path, "r", encoding="utf-8") as f:
        base_j = json.load(f)
    with open(bud_path, "r", encoding="utf-8") as f:
        bud_j = json.load(f)
    with open(hs_path, "r", encoding="utf-8") as f:
        hs_j = json.load(f)

    snap = OrderedDict({
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_files": {"base_calib": base_path, "bud_calib": bud_path, "hetshield_calib": hs_path},
        "visibility_v2": bud_j.get("visibility_v2"),
        "main_score_cfg": base_j.get("frozen_methods", {}).get("main_ms_cos_clear_sem", {}).get("cfg"),
        "frame_div_p95": bud_j.get("frame_div_p95"),
        "frame_high_tau": bud_j.get("frame_high_tau"),
        "gating_params": bud_j.get("gating_params"),
        "hetshield_params": {
            "paper_default": hs_j.get("paper_default_params"),
            "tuned_shared_tau_p95": hs_j.get("fit_tuned_shared_tau_p95"),
            "persensor_tau_p95": hs_j.get("fit_persensor_tau_p95"),
        },
    })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2, ensure_ascii=False)
    return snap


def run_final_test_once(args):
    """A3：启动即落锁的一次性封测测试集总入口（包含 step4_5、step5_2、step5_4 全部三项）。"""
    if args.i_confirm_final_frozen_test != "YES":
        raise RuntimeError("安全锁拦截：必须显式传入 --i_confirm_final_frozen_test YES 才允许访问封存测试集 opv2v_test！")
    for req in (OUT_VAL_BASE, OUT_VAL_BUD, OUT_VAL_HS):
        if not os.path.exists(req):
            raise FileNotFoundError(f"缺少验证集重标定文件，请先运行 --mode val_recalib：{req}")

    os.makedirs(OUT_TEST_DIR, exist_ok=True)
    lock_started = os.path.join(OUT_TEST_DIR, "FINAL_TEST_STARTED.json")
    lock_executed = os.path.join(OUT_TEST_DIR, "FINAL_TEST_EXECUTED.json")
    if os.path.exists(lock_started) or os.path.exists(lock_executed):
        raise RuntimeError(
            f"A3 硬锁拦截：检测到 {lock_started} 已存在，测试集只允许运行一次，严禁重复运行！"
        )

    # A3 强制要求：在任何测试集数据读取之前，第一时间原子写入 FINAL_TEST_STARTED.json
    with open(lock_started, "w", encoding="utf-8") as f:
        json.dump(
            {
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pid": os.getpid(),
                "val_calibrators": [OUT_VAL_BASE, OUT_VAL_BUD, OUT_VAL_HS],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    snap_path = os.path.join(OUT_TEST_DIR, "FROZEN_PARAMS_SNAPSHOT.json")
    export_frozen_params_snapshot(snap_path, OUT_VAL_BASE, OUT_VAL_BUD, OUT_VAL_HS)
    print(f"[A3] 已写入启动硬锁 {lock_started} 与冻结参数全量快照 {snap_path}", flush=True)

    wrapper = HEALBaseWrapper(device="cuda")
    ds_test = wrapper.build_dataset(data_dir=TEST_DIR, use_cav=2, visualize=True, collab_modality="m2")
    seq_order, seq_plan, all_idxs = _sample_balanced_from_dir(wrapper, ds_test, target_frames=args.test_target_frames)
    test_split_path = os.path.join(OUT_TEST_DIR, "opv2v_test_split.json")
    with open(test_split_path, "w", encoding="utf-8") as f:
        json.dump(OrderedDict({
            "dataset_dir": TEST_DIR,
            "total_frames": len(all_idxs),
            "num_sequences": len(seq_order),
            "bootstrap_blocks": len(seq_order),
            "seq_order": seq_order,
            "sequences": seq_plan,
            "all_indices": all_idxs,
        }), f, indent=2, ensure_ascii=False)

    # 同步固化自适应攻击均衡子集（约 320 帧，覆盖全部测试集序列，供 D5/F-P0-6 共用）
    _, sub_plan, sub_idxs = _sample_balanced_from_dir(wrapper, ds_test, target_frames=320)
    adaptive_subset_path = os.path.join(RES_DIR, "adaptive_subset.json")
    if not os.path.exists(adaptive_subset_path):
        with open(adaptive_subset_path, "w", encoding="utf-8") as f:
            json.dump(OrderedDict({
                "dataset_dir": TEST_DIR,
                "total_frames": len(sub_idxs),
                "num_sequences": len(seq_order),
                "seq_order": seq_order,
                "sequences": sub_plan,
                "all_indices": sub_idxs,
            }), f, indent=2, ensure_ascii=False)
    print(f"[C阶段] 测试集共包含 {len(seq_order)} 条序列（Bootstrap 块数={len(seq_order)}），总帧数={len(all_idxs)}", flush=True)

    py_bin = sys.executable
    cmd1 = [
        py_bin, os.path.join(PROJ_ROOT, "scripts_v2_mm", "step4_5_eval_bud_frozen_320.py"),
        "--data_dir", TEST_DIR,
        "--eval_split_json", test_split_path,
        "--base_calib_json", OUT_VAL_BASE,
        "--frozen_gating_json", OUT_VAL_BUD,
        "--cache_dir", os.path.join(OUT_TEST_DIR, "cache_step4_5"),
        "--out_json", os.path.join(OUT_TEST_DIR, "final_test_bud_9cond_report.json"),
    ]
    cmd2 = [
        py_bin, os.path.join(PROJ_ROOT, "scripts_v2_mm", "step5_2_eval_multicav_and_baselines.py"),
        "--data_dir", TEST_DIR,
        "--eval_split_json", test_split_path,
        "--base_calib_json", OUT_VAL_BASE,
        "--frozen_gating_json", OUT_VAL_BUD,
        "--hetshield_calib_json", OUT_VAL_HS,
        "--cache_dir", os.path.join(OUT_TEST_DIR, "cache_step5_2"),
        "--out_json", os.path.join(OUT_TEST_DIR, "final_test_multicav_baselines_report.json"),
    ]
    cmd3 = [
        py_bin, os.path.join(PROJ_ROOT, "scripts_v2_mm", "step5_4_eval_targeted_asr.py"),
        "--data_dir", TEST_DIR,
        "--eval_split_json", test_split_path,
        "--base_calib_json", OUT_VAL_BASE,
        "--frozen_gating_json", OUT_VAL_BUD,
        "--hetshield_calib_json", OUT_VAL_HS,
        "--cache_dir", os.path.join(OUT_TEST_DIR, "cache_step5_4"),
        "--out_json", os.path.join(OUT_TEST_DIR, "final_test_targeted_asr_report.json"),
    ]

    print("[Final Test 1/3] 启动单协作者 9 工况一次性封测...", flush=True)
    subprocess.run(cmd1, check=True)
    print("[Final Test 2/3] 启动多车队分层全基线一次性封测...", flush=True)
    subprocess.run(cmd2, check=True)
    print("[Final Test 3/3] 启动定向攻击 ASR 一次性封测...", flush=True)
    subprocess.run(cmd3, check=True)

    with open(lock_executed, "w", encoding="utf-8") as f:
        json.dump(
            {
                "executed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_sequences": len(seq_order),
                "test_frames": len(all_idxs),
                "frozen_snapshot": snap_path,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n[封测完成] 全部论文终表已生成于：{OUT_TEST_DIR}")


def main():
    parser = argparse.ArgumentParser(description="步骤 ⑧ 独立验证集重标定与一次性测试集封测入口")
    parser.add_argument("--mode", type=str, required=True, choices=["val_recalib", "final_test_once"])
    parser.add_argument("--fit_base_json", type=str, default=DEFAULT_FIT_BASE)
    parser.add_argument("--fit_gating_json", type=str, default=DEFAULT_FIT_GATING)
    parser.add_argument("--fit_hs_json", type=str, default=DEFAULT_FIT_HS)
    parser.add_argument("--val_target_frames", type=int, default=240)
    parser.add_argument("--test_target_frames", type=int, default=0, help="0 表示测试集全量评估")
    parser.add_argument("--i_confirm_final_frozen_test", type=str, default="NO")
    args = parser.parse_args()

    if args.mode == "val_recalib":
        run_val_recalibration(args)
    else:
        run_final_test_once(args)


if __name__ == "__main__":
    main()
