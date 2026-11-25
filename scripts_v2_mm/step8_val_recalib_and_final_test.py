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

from src_v2_mm.bud_fusion_heal import BUDFusionHEALAuditor
from src_v2_mm.heal_base_wrapper import HEALBaseWrapper
from src_v2_mm.hetshield_heal import HetShieldHEALDefender
from src_v2_mm.step4_1_v2_core import (
    CHANNEL_NAMES,
    DIST_LABELS_5,
    PROTO_H,
    PROTO_W,
    VIS_LABELS_3,
    build_bev_grid_xy_256,
    extract_frame_context_and_clean,
)

VAL_DIR = os.path.join(PROJ_ROOT, "data_links", "opv2v_val")
TEST_DIR = os.path.join(PROJ_ROOT, "data_links", "opv2v_test")
RES_DIR = os.path.join(PROJ_ROOT, "results_rq1_v2_mm")

DEFAULT_FIT_BASE = os.path.join(RES_DIR, "step4_1_v2_frozen_calibrator.json")
DEFAULT_FIT_GATING = os.path.join(RES_DIR, "step4_4_bud_frozen_calibrator.json")
DEFAULT_FIT_HS = os.path.join(RES_DIR, "step5_1_hetshield_heal_calib.json")

OUT_VAL_BASE = os.path.join(RES_DIR, "step8_val_base_calibrator.json")
OUT_VAL_BUD = os.path.join(RES_DIR, "step8_val_bud_calibrator.json")
OUT_VAL_HS = os.path.join(RES_DIR, "step8_val_hetshield_calib.json")
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

    cached_val_maps = []
    for rank, idx in enumerate(val_idxs):
        b = extract_frame_context_and_clean(wrapper, ds_m1, ds_m2, idx, gx, gy)
        cov_2d = b["cov_ego"][0, 0]
        d_idx = b["d_idx"]
        v_idx = b["v_idx"]
        cov_np = cov_2d.detach().cpu().numpy()
        d_np = d_idx.detach().cpu().numpy()
        v_np = v_idx.detach().cpu().numpy()

        for mod in ("m1", "m2"):
            dev_np = b["dev_clean"][mod].detach().cpu().numpy()  # (4, 256, 256)
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
            "cov_2d": cov_2d.detach().cpu(),
            "d_idx": d_idx.detach().cpu(),
            "v_idx": v_idx.detach().cpu(),
            "dev_m1": b["dev_clean"]["m1"].half().cpu(),
            "dev_m2": b["dev_clean"]["m2"].half().cpu(),
        })
        if (rank + 1) % 40 == 0 or (rank + 1) == len(val_idxs):
            print(f"  [VAL Pass 1] 进度 {rank + 1}/{len(val_idxs)} ({time.time() - t0:.1f}s)", flush=True)

    # 汇总新 4 通道基准（保留 Fit 的冻结方法配置结构，仅替换统计基准）
    new_tier_stats = OrderedDict()
    for cname in CHANNEL_NAMES:
        st_c = OrderedDict()
        st_c["unified"] = _quantiles_dict(np.concatenate(ch_buf_unified[cname]))
        st_c["sensor_only"] = {m: _quantiles_dict(np.concatenate(ch_buf_sensor[cname][m])) for m in ("m1", "m2")}
        st_c["sensor_dist"] = {
            m: {dl: _quantiles_dict(np.concatenate(ch_buf_sd[cname][m][dl])) if ch_buf_sd[cname][m][dl] else st_c["sensor_only"][m]
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
    val_base_obj["meta"]["recalibrated_on"] = VAL_DIR
    val_base_obj["meta"]["val_frames"] = len(val_idxs)
    val_base_obj["channel_tier_stats"] = new_tier_stats
    os.makedirs(os.path.dirname(OUT_VAL_BASE), exist_ok=True)
    with open(OUT_VAL_BASE, "w", encoding="utf-8") as f:
        json.dump(val_base_obj, f, indent=2, ensure_ascii=False)
    print(f"  已保存 VAL 4通道基准: {OUT_VAL_BASE}")

    # 2. 基于新 VAL 4通道基准，按原冻结分位档重算 div_p95、frame_high_tau 与 cos_tau_by_vis
    auditor = BUDFusionHEALAuditor(OUT_VAL_BASE)
    raw_t2 = {"m1": [], "m2": []}
    raw_t4 = {"m1": [], "m2": []}
    pix_cos_vis = {m: {v: [] for v in range(3)} for m in ("m1", "m2")}

    for item in cached_val_maps:
        cov_2d = item["cov_2d"].cuda()
        d_idx = item["d_idx"].cuda()
        v_idx = item["v_idx"].cuda()
        for mod, key in (("m1", "dev_m1"), ("m2", "dev_m2")):
            dev_4ch = item[key].cuda().float()
            ar = auditor.audit_dev_maps(dev_4ch, cov_2d, d_idx, v_idx, mod, build_gate=False)
            raw_t2[mod].append(ar["raw_t2"])
            raw_t4[mod].append(ar["raw_t4"])
            cm_s = F.avg_pool2d(ar["cos_comb_t4"].view(1, 1, PROTO_H, PROTO_W), kernel_size=3, stride=1, padding=1)[0, 0]
            for v_id in range(3):
                vm = cov_2d & (v_idx == v_id)
                if vm.any():
                    pix_cos_vis[mod][v_id].append(cm_s[vm].detach().cpu().numpy().astype(np.float32))

    frame_div_p95 = OrderedDict({
        "tier2_sensor_only": {m: round(max(float(np.percentile(raw_t2[m], 95.0)), 1e-4), 6) for m in ("m1", "m2")},
        "tier4_sensor_dist_vis": {m: round(max(float(np.percentile(raw_t4[m], 95.0)), 1e-4), 6) for m in ("m1", "m2")},
    })

    # 保持 Fit 阶段验证好的分位层级与高门限安全裕度比例，仅用 VAL 分布刷新数值
    fg = copy.deepcopy(fit_gating["gating_params"])
    q_map = {"p99.0": 99.0, "p99.5": 99.5, "p99.8": 99.8, "p99.9": 99.9, "p99.95": 99.95, "p99.98": 99.98, "p99.99": 99.99}
    if "cos_tau_vis_q" in fg:
        q_names = fg["cos_tau_vis_q"]
        occ_mult = float(fg.get("cos_tau_occ_mult", 1.0))
        fg["cos_tau_by_vis"] = {
            m: [
                round(
                    float(np.percentile(np.concatenate(pix_cos_vis[m][vi]), q_map[q_names[vi]]))
                    * (occ_mult if vi == 2 else 1.0),
                    6,
                )
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

    # 3. 同步在 OPV2V_VAL 连续帧上重标定 HetShield 的 95% 分位门限（保持 ConvGRU 权重完全冻结）
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
    from opencood.tools import train_utils

    e_th = float(fit_hs["meta"]["energy_thresh"])
    hs_s = HetShieldHEALDefender(mode="shared", energy_thresh=e_th).cuda().eval()
    hs_s.predictor.load_state_dict(torch.load(fit_hs["meta"]["shared_ckpt"], map_location="cpu"))
    hs_p = HetShieldHEALDefender(mode="persensor", energy_thresh=e_th).cuda().eval()
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
                hs_s.reset_sequence(s)
                hs_p.reset_sequence(s)
                for step_i, idx in enumerate(s_idxs):
                    raw_b = train_utils.to_device(ds.collate_batch_test([ds[idx]]), wrapper.device_obj)
                    p = wrapper.extract_protocol_features(raw_b["ego"])
                    tmat = p["affine_matrix"][0, 0, 1:2]
                    fc = warp_affine_simple(p["heter_feature_2d"][1:2], tmat, (PROTO_H, PROTO_W), align_corners=ac)
                    vc = warp_affine_simple(p["cam_valid_mask"][1:2].float(), tmat, (PROTO_H, PROTO_W), align_corners=ac)
                    items = [{"cav_id": f"{s}_{mod}_c1", "mod": mod, "feat_warped": fc, "valid_ego": (vc > 0.5).float()}]
                    rs = hs_s.step_frame(s, step_i, p["heter_feature_2d"][0:1], items, update_state=True)[0]
                    rp = hs_p.step_frame(s, step_i, p["heter_feature_2d"][0:1], items, update_state=True)[0]
                    if step_i >= 1:
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


def run_final_test_once(args):
    """双重锁控的一次性封测测试集总入口（严禁在未经确认前运行）。"""
    if args.i_confirm_final_frozen_test != "YES":
        raise RuntimeError("安全锁拦截：必须显式传入 --i_confirm_final_frozen_test YES 才允许访问封存测试集 opv2v_test！")
    for req in (OUT_VAL_BASE, OUT_VAL_BUD, OUT_VAL_HS):
        if not os.path.exists(req):
            raise FileNotFoundError(f"缺少验证集重标定文件，请先运行 --mode val_recalib：{req}")

    lock_receipt = os.path.join(OUT_TEST_DIR, "FINAL_TEST_EXECUTED.json")
    if os.path.exists(lock_receipt):
        raise RuntimeError(f"防重复测试锁触发：检测到封测回执 {lock_receipt} 已存在，严禁重复评估测试集！")

    os.makedirs(OUT_TEST_DIR, exist_ok=True)
    wrapper = HEALBaseWrapper(device="cuda")
    ds_test = wrapper.build_dataset(data_dir=TEST_DIR, use_cav=2, visualize=True, collab_modality="m2")
    seq_order, seq_plan, all_idxs = _sample_balanced_from_dir(wrapper, ds_test, target_frames=args.test_target_frames)
    test_split_path = os.path.join(OUT_TEST_DIR, "opv2v_test_split.json")
    with open(test_split_path, "w", encoding="utf-8") as f:
        json.dump(OrderedDict({
            "dataset_dir": TEST_DIR,
            "total_frames": len(all_idxs),
            "num_sequences": len(seq_order),
            "seq_order": seq_order,
            "sequences": seq_plan,
            "all_indices": all_idxs,
        }), f, indent=2, ensure_ascii=False)

    py_bin = sys.executable
    # 1) 单协作者 9 工况终测
    cmd1 = [
        py_bin, os.path.join(PROJ_ROOT, "scripts_v2_mm", "step4_5_eval_bud_frozen_320.py"),
        "--data_dir", TEST_DIR,
        "--eval_split_json", test_split_path,
        "--base_calib_json", OUT_VAL_BASE,
        "--frozen_gating_json", OUT_VAL_BUD,
        "--cache_dir", os.path.join(OUT_TEST_DIR, "cache_step4_5"),
        "--out_json", os.path.join(OUT_TEST_DIR, "final_test_bud_9cond_report.json"),
    ]
    # 2) 多车队分层全基线终测
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

    print("[Final Test 1/2] 启动单协作者 9 工况一次性封测...", flush=True)
    subprocess.run(cmd1, check=True)
    print("[Final Test 2/2] 启动多车队分层全基线一次性封测...", flush=True)
    subprocess.run(cmd2, check=True)

    with open(lock_receipt, "w", encoding="utf-8") as f:
        json.dump({"executed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "test_frames": len(all_idxs)}, f, indent=2)
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
