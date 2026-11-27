# -*- coding: utf-8 -*-
"""
步骤 ⑦（精简版）：单帧推理耗时拆解脚本（GPU CUDA Event 精确计时）。

测试内容：
1. 在 1 / 2 / >=3 协作车场景（默认 mixed 车队）上，先做 10 帧 Warmup，再测 100 帧的 Mean / Median / P95 耗时 (ms)；
2. 精细拆解 BUD-Fusion 接收端各子阶段毫秒开销：
   - T_vis_bin: 本车点云物理遮挡射线追踪 + 距离/可见性分档；
   - T_audit_dev: 逐车多尺度余弦与 5x5 位置容差有向语义偏差图提取；
   - T_gate_build: 4 通道归一化 + 可见性余弦死区软门控生成与整车回退判定；
   - T_gated_fuse_head: 门控多尺度融合 + 检测头推理；
   - T_bud_receiver_total: 接收端完整防御+融合+检测端到端总延迟；
3. 同口径对比 No-Defense、BUD-Fusion、HetShield、LUCIA、ROBOSAC (>=3 车) 的单帧端到端推理延迟。
"""

import argparse
import json
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

from src_v2_mm.baselines_robosac_lucia_heal import ROBOSACHEALDefender, build_lucia_fuse_fn
from src_v2_mm.bud_fusion_heal import (
    BUDFusionHEALAuditor,
    build_gated_fuse_fn,
    extract_single_cav_dev_and_signed,
)
from src_v2_mm.heal_base_wrapper import HEALBaseWrapper
from src_v2_mm.hetshield_heal import HetShieldHEALDefender, forward_hetshield_protocol_mix
from src_v2_mm.step4_1_v2_core import (
    DEFAULT_LOS_OCC_TAU,
    OPV2V_TRAIN_DIR,
    PROTO_H,
    PROTO_W,
    V3_BUD_CALIB_V1_JSON,
    V3_BUD_TUNED_JSON,
    V3_HETSHIELD_CALIB_JSON,
    V3_SPLITS_JSON,
    assign_dist5_vis3_indices_v2,
    build_bev_grid_xy_256,
    compute_ego_expected_pts_and_los_256,
    load_ego_lidar_np,
    require_visibility_v2_taus,
)

DEFAULT_EVAL_SPLIT = V3_SPLITS_JSON
DEFAULT_BASE_CALIB = V3_BUD_CALIB_V1_JSON
DEFAULT_FROZEN_GATING = V3_BUD_TUNED_JSON
DEFAULT_HETSHIELD_CALIB = V3_HETSHIELD_CALIB_JSON
DEFAULT_OUT_JSON = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step7_latency_breakdown.json")


def _cuda_time_ms(fn):
    """使用 CUDA Event 精确测量单次 GPU 前向毫秒耗时。"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)), out


def _stat_ms(vals: list) -> dict:
    if not vals:
        return {"n": 0, "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean_ms": round(float(arr.mean()), 3),
        "median_ms": round(float(np.median(arr)), 3),
        "p95_ms": round(float(np.percentile(arr, 95.0)), 3),
    }


def main():
    parser = argparse.ArgumentParser(description="步骤 ⑦ 单帧推理耗时拆解（CUDA Event 精确计时）")
    parser.add_argument("--data_dir", type=str, default=OPV2V_TRAIN_DIR)
    parser.add_argument("--eval_split_json", type=str, default=DEFAULT_EVAL_SPLIT)
    parser.add_argument("--base_calib_json", type=str, default=DEFAULT_BASE_CALIB)
    parser.add_argument("--frozen_gating_json", type=str, default=DEFAULT_FROZEN_GATING)
    parser.add_argument("--hetshield_calib_json", type=str, default=DEFAULT_HETSHIELD_CALIB)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--max_cav", type=int, default=7)
    parser.add_argument("--out_json", type=str, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "单帧耗时拆解必须在 CUDA GPU 环境下运行！"
    t0 = time.time()
    with open(args.eval_split_json, "r", encoding="utf-8") as f:
        eval_split = json.load(f)
    all_idxs = list(eval_split["all_indices"])
    picked_idxs = all_idxs[: min(len(all_idxs), int(args.warmup) + int(args.num_frames))]

    wrapper = HEALBaseWrapper(device="cuda", use_cav=args.max_cav)
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
    from opencood.tools import train_utils
    from opencood.utils import common_utils

    ds = wrapper.build_dataset(
        data_dir=args.data_dir, use_cav=args.max_cav, visualize=True,
        collab_modality="mixed", max_cav=args.max_cav,
    )
    gx, gy = build_bev_grid_xy_256(device="cuda", dtype=torch.float32)
    auditor = BUDFusionHEALAuditor(args.base_calib_json, gating_calib_path=args.frozen_gating_json)

    # 加载 HetShield（若已训练则加载权重，否则用随机权重测纯计算耗时并标注）
    hs_loaded = False
    hs_def = HetShieldHEALDefender(mode="shared", alpha=0.6, tau_th=0.3, temperature=0.1).cuda().eval()
    if os.path.exists(args.hetshield_calib_json):
        with open(args.hetshield_calib_json, "r", encoding="utf-8") as f:
            hc = json.load(f)
        if os.path.exists(hc["meta"]["shared_ckpt"]):
            hs_def.predictor.load_state_dict(torch.load(hc["meta"]["shared_ckpt"], map_location="cpu"))
            hs_loaded = True
    robosac = ROBOSACHEALDefender(box_matching_thresh=0.3, jaccard_thresh=0.3, step_budget=5)

    buckets = {}

    def _get_bucket(key: str):
        if key not in buckets:
            buckets[key] = {
                "bud_vis_bin": [],
                "bud_audit_dev": [],
                "bud_gate_build": [],
                "bud_gated_fuse_head": [],
                "bud_receiver_total": [],
                "no_defense_total": [],
                "hetshield_total": [],
                "lucia_total": [],
                "robosac_total": [],
            }
        return buckets[key]

    ac = wrapper.model.pyramid_backbone.align_corners

    for step_i, idx in enumerate(picked_idxs):
        raw_b = train_utils.to_device(ds.collate_batch_test([ds[idx]]), wrapper.device_obj)
        p = wrapper.extract_protocol_features(raw_b["ego"])
        feat = p["heter_feature_2d"]
        rlen = p["record_len"]
        amat = p["affine_matrix"]
        mods = list(p["agent_modality_list"])
        vmask = p["cam_valid_mask"]
        n_total = int(feat.shape[0])
        n_collab = n_total - 1
        if n_collab < 1:
            continue
        ego_pts = load_ego_lidar_np(ds, idx)

        # 1) No-Defense
        t_nd, _ = _cuda_time_ms(lambda: wrapper.forward_from_protocol(feat, rlen, amat, mods))

        # 2) BUD-Fusion 分阶段精确计时
        vis_v2 = (auditor.gating_calib or {}).get("visibility_v2", None)
        fg_prob_tau = float((auditor.gating_calib or {}).get("gating_params", {}).get("fg_prob_tau", 0.20))

        def _step_vis_bin():
            sa, lf, ep = compute_ego_expected_pts_and_los_256(ego_pts, gx, gy, num_beams=64)
            d_list, v_list = [], []
            c_tau, l_tau = require_visibility_v2_taus(auditor.gating_calib)
            for ci in range(1, n_total):
                ptm = raw_b["ego"]["pairwise_t_matrix"][0, ci, 0]
                cx, cy = float(ptm[0, 3].item()), float(ptm[1, 3].item())
                dist_ci = torch.sqrt((gx - cx) ** 2 + (gy - cy) ** 2).view(1, 1, PROTO_H, PROTO_W)
                di, vi = assign_dist5_vis3_indices_v2(dist_ci, lf, ep, clear_pts_tau=c_tau, los_occ_tau=l_tau)
                d_list.append(di[0, 0])
                v_list.append(vi[0, 0])
            return d_list, v_list

        t_vis, (d_list, v_list) = _cuda_time_ms(_step_vis_bin)

        def _step_audit_dev():
            o_ego = wrapper.forward_single_agent(feat[0:1])
            p_ego_cls = torch.sigmoid(o_ego["cls_preds"]).max(dim=1, keepdim=True)[0]
            dev_items = []
            for ci in range(1, n_total):
                dev_items.append(
                    extract_single_cav_dev_and_signed(
                        wrapper, feat, amat, mods, vmask, p_ego_cls, cav_idx=ci, k_tol=5, return_pfused=True
                    )
                )
            return dev_items

        t_dev, dev_items = _cuda_time_ms(_step_audit_dev)

        def _step_gate_build():
            gates = []
            for ci in range(1, n_total):
                dev_ci, dd_ci, da_ci, cov_ci, pf_ci = dev_items[ci - 1]
                fg_ci = F.max_pool2d(pf_ci, kernel_size=3, stride=1, padding=1)[0, 0] > fg_prob_tau
                ar = auditor.audit_dev_maps(
                    dev_ci, cov_ci, d_list[ci - 1], v_list[ci - 1], mods[ci],
                    build_gate=True, d_drop_tol=dd_ci, d_add_tol=da_ci,
                    enable_whole_fallback=True, collab_fg_mask=fg_ci
                )
                gates.append(ar["gate_L0"])
            return torch.cat(gates, dim=0)

        t_gate, gate_stack = _cuda_time_ms(_step_gate_build)
        t_fuse, _ = _cuda_time_ms(lambda: wrapper.forward_from_protocol(feat, rlen, amat, mods, custom_fuse_fn=build_gated_fuse_fn(gate_stack)))
        t_bud_tot = t_vis + t_dev + t_gate + t_fuse

        # 3) LUCIA
        t_luc, _ = _cuda_time_ms(lambda: wrapper.forward_from_protocol(feat, rlen, amat, mods, custom_fuse_fn=build_lucia_fuse_fn(32)))

        # 4) HetShield（单帧审计 + 门控融合）
        def _step_hs():
            c_items = []
            for ci in range(1, n_total):
                tmat = amat[0, 0, ci : ci + 1]
                fc = warp_affine_simple(feat[ci : ci + 1], tmat, (PROTO_H, PROTO_W), align_corners=ac)
                vc = warp_affine_simple(vmask[ci : ci + 1].float(), tmat, (PROTO_H, PROTO_W), align_corners=ac)
                c_items.append({"cav_id": f"lat_c{ci}", "mod": mods[ci], "feat_warped": fc, "valid_ego": (vc > 0.5).float()})
            rs = hs_def.step_frame("lat_seq", step_i, feat[0:1], c_items, update_state=True)
            gl = [r["g_k"] for r in rs]
            return forward_hetshield_protocol_mix(wrapper, feat, rlen, amat, mods, vmask, gl)

        t_hs, _ = _cuda_time_ms(_step_hs)

        # 5) ROBOSAC (仅 >=3 车)
        t_rob = None
        if n_total >= 3:
            def _step_rob():
                o_e = wrapper.forward_single_agent(feat[0:1])
                epb, eps, egb = wrapper.decode_boxes(raw_b, o_e, ds)
                epb_np = common_utils.torch_tensor_to_numpy(epb) if (epb is not None and len(epb) > 0) else None
                return robosac.run_frame(wrapper, feat, rlen, amat, mods, raw_b, ds, epb_np, epb, eps, egb, frame_seed=idx)
            t_rob, _ = _cuda_time_ms(_step_rob)

        if step_i < int(args.warmup):
            continue

        tier = "collab_1" if n_collab == 1 else ("collab_2" if n_collab == 2 else "collab_ge3")
        for bk in ("overall", tier):
            b = _get_bucket(bk)
            b["bud_vis_bin"].append(t_vis)
            b["bud_audit_dev"].append(t_dev)
            b["bud_gate_build"].append(t_gate)
            b["bud_gated_fuse_head"].append(t_fuse)
            b["bud_receiver_total"].append(t_bud_tot)
            b["no_defense_total"].append(t_nd)
            b["hetshield_total"].append(t_hs)
            b["lucia_total"].append(t_luc)
            if t_rob is not None:
                b["robosac_total"].append(t_rob)

    summary = OrderedDict({
        bk: OrderedDict({k: _stat_ms(v) for k, v in b.items()})
        for bk, b in buckets.items()
    })
    out_obj = OrderedDict({
        "meta": {
            "description": "步骤 ⑦ 单帧推理耗时拆解（CUDA Event 精确计时，10 帧预热后统计）",
            "gpu_name": torch.cuda.get_device_name(0),
            "warmup_frames": int(args.warmup),
            "measured_frames": max(0, len(picked_idxs) - int(args.warmup)),
            "hetshield_weights_loaded": hs_loaded,
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "latency_ms": summary,
    })
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[完成] 耗时拆解报告已保存至：{args.out_json}")


if __name__ == "__main__":
    main()
