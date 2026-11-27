# -*- coding: utf-8 -*-
"""
步骤 ④-2：纯 Fit 集标定 BUD-Fusion 局部门控超参与 div_p95 帧门限（零 Eval/Dev/Test 泄露）。

设计特性：
1. 支持命令行指定 --data_dir / --split_json / --base_calib_json / --out_json，
   未来验证集 (validate) 就绪后只需替换路径参数重跑即可重标定最终门限，无需改任何核心代码；
2. 帧级校准统一使用稳健 div_p95（坚决不用窄分母 med_p95）；
3. 严禁读取任何 Dev/Eval/Test 帧，仅在 Fit-100 计算分档基准并在 Fit-Dev 36 帧上搜索最优
   (tau_loc_q, r_dil, beta)，校验干净 AP 损失与局部攻击门控选择性达标后冻结输出 JSON。
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

from src_v2_mm.bud_fusion_heal import (
    BUDFusionHEALAuditor,
    build_directional_gate_map,
    build_gated_fuse_fn,
    compute_shift_tolerant_semantic_maps,
)
from src_v2_mm.heal_attacks import (
    generate_local_ghost_spoofing_heal,
    generate_local_removal_heal,
    generate_scaled_pgd_removal_heal,
)
from src_v2_mm.heal_base_wrapper import HEALBaseWrapper
from src_v2_mm.step4_1_v2_core import (
    OPV2V_TRAIN_DIR,
    PROTO_H,
    PROTO_W,
    _masked_avg_pool3x3,
    build_bev_grid_xy_256,
    extract_frame_context_and_clean,
    require_visibility_v2_taus,
)

DEFAULT_SPLIT = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_1_v2_split_plan.json")
DEFAULT_BASE_CALIB = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_1_v2_frozen_calibrator.json")
DEFAULT_OUT = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_2_bud_gating_calibrator.json")

FIT_DEV_STATES = [
    "clean",
    "local_removal_1.0sigma",
    "local_ghost_2.0sigma",
    "scaled_pgd_1.0sigma",
]


def _empty_ap_stat():
    return {0.5: {"fp": [], "tp": [], "score": [], "gt": 0}, 0.7: {"fp": [], "tp": [], "score": [], "gt": 0}}


def _slim_batch_for_postprocess(batch_dict: dict) -> dict:
    """只保留 dataset.post_process 与 generate_gt_bbx 所需字段在 CPU 内存，释放占显存/内存的图像与点云体素。"""
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


@torch.no_grad()
def _compute_dev_and_signed(
    wrapper,
    p_dict: dict,
    feat_in: torch.Tensor,
    p_ego_cls: torch.Tensor,
    cov_ego: torch.Tensor,
    k_tol: int = 5,
):
    """单次前向同时返回 4 通道偏差图与 5x5 位置容差有向语义冲突图 (d_drop_tol, d_add_tol)。"""
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
    return dev_4ch, d_drop_tol, d_add_tol


def _eval_ap_from_predictions(wrapper, ds_ref, eval_items: list, g_params: dict, no_def_cache: dict = None):
    """在 Fit-Dev 缓存特征上计算指定有向语义门控参数（或无防御）的 AP@0.5 / AP@0.7 与局部权重统计。"""
    from opencood.tools import train_utils
    from opencood.utils import eval_utils

    ap_stats = {m: {st: _empty_ap_stat() for st in FIT_DEV_STATES} for m in ("m1", "m2")}
    gate_stats = {
        "clean_valid": [],
        "lrem_atk": [], "lrem_cln": [],
        "lgho_atk": [], "lgho_cln": [],
    }

    for item in eval_items:
        mod = item["mod"]
        st_name = item["state"]
        sidx = item["sample_idx"]
        b_post = train_utils.to_device(item["batch_slim"], wrapper.device_obj)
        cov_2d = item["cov_2d"].to(wrapper.device_obj)

        if g_params is None:
            fuse_fn = None
        else:
            cos_comb = item["cos_comb_t4"].to(wrapper.device_obj)
            d_drop_tol = item["d_drop_tol"].to(wrapper.device_obj)
            d_add_tol = item["d_add_tol"].to(wrapper.device_obj)
            v_idx = item["v_idx"].to(wrapper.device_obj)

            gate_l0 = build_directional_gate_map(
                cos_comb=cos_comb,
                d_drop_tol=d_drop_tol,
                d_add_tol=d_add_tol,
                cov_2d=cov_2d,
                v_idx=v_idx,
                cos_tau=float(g_params["cos_tau"][mod]),
                drop_tau=float(g_params["drop_tau"]),
                add_tau=float(g_params["add_tau"]),
                w_drop=float(g_params["w_drop"]),
                w_add=float(g_params["w_add"]),
                cos_co_ratio=float(g_params["cos_co_ratio"]),
                gamma_add=g_params["gamma_add"],
                r_dil=int(g_params["r_dil"]),
                beta=float(g_params["beta"]),
            )
            g_valid = gate_l0[0, 0][cov_2d]

            if st_name == "clean":
                gate_stats["clean_valid"].append(float(g_valid.mean().item()))
            elif st_name in ("local_removal_1.0sigma", "local_ghost_2.0sigma") and item["atk_mask_2d"] is not None:
                atk_m = item["atk_mask_2d"].to(wrapper.device_obj) & cov_2d
                cln_m = cov_2d & (~atk_m)
                tag = "lrem" if st_name == "local_removal_1.0sigma" else "lgho"
                if atk_m.any():
                    gate_stats[f"{tag}_atk"].append(float(gate_l0[0, 0][atk_m].mean().item()))
                if cln_m.any():
                    gate_stats[f"{tag}_cln"].append(float(gate_l0[0, 0][cln_m].mean().item()))

            # 若整帧门控严格全为 1 且已有无防御缓存，直接复用，显著加速搜索
            if no_def_cache is not None and float(g_valid.min().item()) >= 1.0 - 1e-6:
                pb, ps, gb = no_def_cache[(mod, st_name, sidx)]
                eval_utils.caluclate_tp_fp(pb, ps, gb, ap_stats[mod][st_name], 0.5)
                eval_utils.caluclate_tp_fp(pb, ps, gb, ap_stats[mod][st_name], 0.7)
                continue

            fuse_fn = build_gated_fuse_fn(gate_l0)

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
        if pb is not None:
            pb_c, ps_c, gb_c = pb.to(wrapper.device_obj), ps.to(wrapper.device_obj), gb.to(wrapper.device_obj)
        else:
            pb_c, ps_c, gb_c = None, None, gb.to(wrapper.device_obj)
        eval_utils.caluclate_tp_fp(pb_c, ps_c, gb_c, ap_stats[mod][st_name], 0.5)
        eval_utils.caluclate_tp_fp(pb_c, ps_c, gb_c, ap_stats[mod][st_name], 0.7)

    ap50 = {m: {st: float(eval_utils.calculate_ap(ap_stats[m][st], 0.5)[0]) for st in FIT_DEV_STATES} for m in ("m1", "m2")}
    ap70 = {m: {st: float(eval_utils.calculate_ap(ap_stats[m][st], 0.7)[0]) for st in FIT_DEV_STATES} for m in ("m1", "m2")}
    return ap50, ap70, gate_stats


def main():
    parser = argparse.ArgumentParser(description="纯 Fit 标定 BUD-Fusion 局部门控超参与 div_p95 门限")
    parser.add_argument("--data_dir", type=str, default=OPV2V_TRAIN_DIR)
    parser.add_argument("--split_json", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--base_calib_json", type=str, default=DEFAULT_BASE_CALIB)
    parser.add_argument("--out_json", type=str, default=DEFAULT_OUT)
    args = parser.parse_args()

    t0 = time.time()
    with open(args.split_json, "r", encoding="utf-8") as f:
        split_plan = json.load(f)

    fit_indices = list(split_plan["fit_indices"])
    fit_dev_set = set(split_plan["fit_dev_indices"])
    eval_set = set(split_plan.get("eval_indices", []))
    assert set(fit_indices).isdisjoint(eval_set), "Fit 与 Eval 存在交集，拒绝运行！"
    print(f"[Step 2 · Fit 门控标定] Fit={len(fit_indices)} 帧 (Dev={len(fit_dev_set)}), Eval={len(eval_set)} 帧（严禁访问）")

    wrapper = HEALBaseWrapper(device="cuda")
    from opencood.tools import train_utils
    ds_m1 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m1")
    ds_m2 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m2")
    gx, gy = build_bev_grid_xy_256(device="cuda", dtype=torch.float32)

    auditor = BUDFusionHEALAuditor(args.base_calib_json)

    raw_scores = {"t2": {"m1": [], "m2": []}, "t4": {"m1": [], "m2": []}}
    pix_smap_t4 = {"m1": [], "m2": []}
    pix_cos_s_t4 = {"m1": [], "m2": []}
    fit_dev_items = []
    k_tol = 5

    c_tau, l_tau = require_visibility_v2_taus(auditor.base_calib)
    for rank, idx in enumerate(fit_indices):
        assert idx in set(fit_indices) and idx not in eval_set
        b = extract_frame_context_and_clean(wrapper, ds_m1, ds_m2, idx, gx, gy, clear_pts_tau=c_tau, los_occ_tau=l_tau)
        cov_2d = b["cov_ego"][0, 0]
        d_idx = b["d_idx"]
        v_idx = b["v_idx"]
        is_dev = idx in fit_dev_set

        batch_slim = None
        if is_dev:
            raw_b1 = train_utils.to_device(ds_m1.collate_batch_test([ds_m1[idx]]), "cpu")
            batch_slim = _slim_batch_for_postprocess(raw_b1)

        for mod, p_dict in (("m1", b["p1"]), ("m2", b["p2"])):
            ar_cln = auditor.audit_dev_maps(b["dev_clean"][mod], cov_2d, d_idx, v_idx, mod, build_gate=False)
            raw_scores["t2"][mod].append(ar_cln["raw_t2"])
            raw_scores["t4"][mod].append(ar_cln["raw_t4"])
            pix_smap_t4[mod].append(ar_cln["s_map_t4"][cov_2d].detach().cpu().numpy().astype(np.float32))
            cm_s = F.avg_pool2d(ar_cln["cos_comb_t4"].view(1, 1, PROTO_H, PROTO_W), kernel_size=3, stride=1, padding=1)[0, 0]
            pix_cos_s_t4[mod].append(cm_s[cov_2d].detach().cpu().numpy().astype(np.float32))

            if is_dev:
                cov_cpu = cov_2d.detach().cpu()
                vidx_cpu = v_idx.detach().cpu()
                rlen_cpu = p_dict["record_len"].detach().cpu()
                amat_cpu = p_dict["affine_matrix"].detach().cpu()
                mods_l = list(p_dict["agent_modality_list"])

                _, dd_cln, da_cln = _compute_dev_and_signed(
                    wrapper, p_dict, p_dict["heter_feature_2d"], b["p_ego_cls"], b["cov_ego"], k_tol=k_tol
                )
                fit_dev_items.append({
                    "sample_idx": idx, "mod": mod, "state": "clean",
                    "feat_dev": p_dict["heter_feature_2d"].detach().cpu(),
                    "rlen": rlen_cpu, "amat": amat_cpu, "mods": mods_l,
                    "batch_slim": batch_slim, "cov_2d": cov_cpu, "v_idx": vidx_cpu,
                    "cos_comb_t4": ar_cln["cos_comb_t4"].detach().cpu(),
                    "d_drop_tol": dd_cln.detach().cpu(),
                    "d_add_tol": da_cln.detach().cpu(),
                    "atk_mask_2d": None,
                })

                feat = p_dict["heter_feature_2d"]
                rlen = p_dict["record_len"]
                amat = p_dict["affine_matrix"]
                vmask = p_dict["cam_valid_mask"]
                preg = b["cov_local_m1"] if mod == "m1" else None

                f_lr, st_lr = generate_local_removal_heal(wrapper, feat, rlen, amat, mods_l, vmask, 1, 2, 5, 1.0, 10, preg)
                f_lg, st_lg = generate_local_ghost_spoofing_heal(wrapper, feat, rlen, amat, mods_l, vmask, 1, 2, 5, 2.0, 10, idx, preg)
                f_p1, _ = generate_scaled_pgd_removal_heal(wrapper, feat, rlen, amat, mods_l, vmask, 1, 1.0, 8, preg)

                for st_name, f_adv, st_meta in (
                    ("local_removal_1.0sigma", f_lr, st_lr),
                    ("local_ghost_2.0sigma", f_lg, st_lg),
                    ("scaled_pgd_1.0sigma", f_p1, None),
                ):
                    dev_a, dd_a, da_a = _compute_dev_and_signed(
                        wrapper, p_dict, f_adv, b["p_ego_cls"], b["cov_ego"], k_tol=k_tol
                    )
                    ar_a = auditor.audit_dev_maps(dev_a, cov_2d, d_idx, v_idx, mod, build_gate=False)
                    atk_m = st_meta["attacked_mask_in_ego"][0, 0].detach().cpu() > 0.5 if st_meta and "attacked_mask_in_ego" in st_meta else None
                    fit_dev_items.append({
                        "sample_idx": idx, "mod": mod, "state": st_name,
                        "feat_dev": f_adv.detach().cpu(),
                        "rlen": rlen_cpu, "amat": amat_cpu, "mods": mods_l,
                        "batch_slim": batch_slim, "cov_2d": cov_cpu, "v_idx": vidx_cpu,
                        "cos_comb_t4": ar_a["cos_comb_t4"].detach().cpu(),
                        "d_drop_tol": dd_a.detach().cpu(),
                        "d_add_tol": da_a.detach().cpu(),
                        "atk_mask_2d": atk_m,
                    })

        if (rank + 1) % 20 == 0 or (rank + 1) == len(fit_indices):
            print(f"  已处理 Fit 样本 {rank + 1}/{len(fit_indices)} ({time.time() - t0:.1f}s)")

    # 1. 精确计算 div_p95 帧门限、S_map_t4 及平滑后 3x3 cos_comb_t4 像素高分位基准
    frame_div_p95 = OrderedDict({
        "tier2_sensor_only": {
            m: round(max(float(np.percentile(raw_scores["t2"][m], 95.0)), 1e-4), 6) for m in ("m1", "m2")
        },
        "tier4_sensor_dist_vis": {
            m: round(max(float(np.percentile(raw_scores["t4"][m], 95.0)), 1e-4), 6) for m in ("m1", "m2")
        },
    })

    q_levels = [98.0, 99.0, 99.5, 99.8, 99.9]
    q_labels = ["p98.0", "p99.0", "p99.5", "p99.8", "p99.9"]
    pix_quantiles = OrderedDict()
    for m in ("m1", "m2"):
        all_pix = np.concatenate(pix_smap_t4[m], axis=0).astype(np.float64, copy=False)
        qs = np.percentile(all_pix, q_levels)
        pix_quantiles[m] = OrderedDict({ql: round(float(qv), 6) for ql, qv in zip(q_labels, qs)})
    del pix_smap_t4

    cq_levels = [99.0, 99.5, 99.8, 99.9, 99.95]
    cq_labels = ["p99.0", "p99.5", "p99.8", "p99.9", "p99.95"]
    pix_cos_quantiles = OrderedDict()
    for m in ("m1", "m2"):
        all_cos = np.concatenate(pix_cos_s_t4[m], axis=0).astype(np.float64, copy=False)
        cqs = np.percentile(all_cos, cq_levels)
        pix_cos_quantiles[m] = OrderedDict({ql: round(float(qv), 6) for ql, qv in zip(cq_labels, cqs)})
    del pix_cos_s_t4
    torch.cuda.empty_cache()

    print("\n=== Fit-100 帧级 div_p95 与像素高分位 ===")
    print("  frame_div_p95:", json.dumps(frame_div_p95))
    print("  pixel_smap_t4_quantiles:", json.dumps(pix_quantiles))
    print("  pixel_cos_s_t4_quantiles:", json.dumps(pix_cos_quantiles))

    # 2. 算一次 Fit-Dev 无防御基线并缓存
    no_def_cache = {}
    base_ap50, base_ap70, _ = _eval_ap_from_predictions(wrapper, ds_m1, fit_dev_items, None, no_def_cache=no_def_cache)
    print("\n=== Fit-Dev 36 帧无防御基准 AP@0.5 ===")
    print("  m1:", {k: round(v, 4) for k, v in base_ap50["m1"].items()})
    print("  m2:", {k: round(v, 4) for k, v in base_ap50["m2"].items()})

    # 3. 枚举 32 组「5x5位置容差有向语义 + 多尺度余弦协同门控」候选配置寻找最优解
    cos_q_cands = ["p99.5", "p99.8", "p99.9", "p99.95"]
    drop_tau_cands = [0.40, 0.48]
    r_dil_cands = [4, 5]
    beta_cands = [3.0, 4.0]
    fixed_add_tau = 0.30
    fixed_w_drop = 7.0
    fixed_w_add = 6.0
    fixed_co_ratio = 0.82
    fixed_gamma_add = [1.0, 0.3, 0.0]

    best_feasible = None
    best_obj = -1e9
    best_fallback = None
    best_fb_obj = -1e9
    all_records = []

    for cq in cos_q_cands:
        cos_tau_map = {"m1": pix_cos_quantiles["m1"][cq], "m2": pix_cos_quantiles["m2"][cq]}
        for dt in drop_tau_cands:
            for rd in r_dil_cands:
                for bt in beta_cands:
                    g_params = {
                        "mode": "directional_shift_tol",
                        "k_tol": k_tol,
                        "cos_tau_q": cq,
                        "cos_tau": cos_tau_map,
                        "cos_co_ratio": fixed_co_ratio,
                        "drop_tau": dt,
                        "add_tau": fixed_add_tau,
                        "w_drop": fixed_w_drop,
                        "w_add": fixed_w_add,
                        "gamma_add": fixed_gamma_add,
                        "r_dil": rd,
                        "beta": bt,
                    }
                    ap50, ap70, gst = _eval_ap_from_predictions(
                        wrapper, ds_m1, fit_dev_items, g_params, no_def_cache=no_def_cache
                    )
                    cln_loss_m1 = base_ap50["m1"]["clean"] - ap50["m1"]["clean"]
                    cln_loss_m2 = base_ap50["m2"]["clean"] - ap50["m2"]["clean"]
                    max_cln_loss = max(cln_loss_m1, cln_loss_m2)

                    mean_atk_ap50 = float(np.mean([
                        ap50[m][a] for m in ("m1", "m2") for a in ("local_removal_1.0sigma", "local_ghost_2.0sigma", "scaled_pgd_1.0sigma")
                    ]))
                    g_lrem_atk = float(np.mean(gst["lrem_atk"])) if gst["lrem_atk"] else 1.0
                    g_lrem_cln = float(np.mean(gst["lrem_cln"])) if gst["lrem_cln"] else 1.0
                    g_lgho_atk = float(np.mean(gst["lgho_atk"])) if gst["lgho_atk"] else 1.0
                    g_lgho_cln = float(np.mean(gst["lgho_cln"])) if gst["lgho_cln"] else 1.0
                    mean_atk_g = 0.5 * (g_lrem_atk + g_lgho_atk)
                    mean_cln_g = 0.5 * (g_lrem_cln + g_lgho_cln)
                    gate_gap = mean_cln_g - mean_atk_g

                    obj = mean_atk_ap50 + 0.15 * gate_gap
                    rec = {
                        "gating_params": g_params,
                        "ap50": {m: {k: round(v, 5) for k, v in ap50[m].items()} for m in ("m1", "m2")},
                        "ap70": {m: {k: round(v, 5) for k, v in ap70[m].items()} for m in ("m1", "m2")},
                        "clean_ap50_loss": {"m1": round(cln_loss_m1, 5), "m2": round(cln_loss_m2, 5)},
                        "max_clean_ap50_loss": round(max_cln_loss, 5),
                        "mean_atk_ap50": round(mean_atk_ap50, 5),
                        "gate_stats": {
                            "clean_valid_mean": round(float(np.mean(gst["clean_valid"])), 5),
                            "lrem_atk": round(g_lrem_atk, 5),
                            "lrem_cln": round(g_lrem_cln, 5),
                            "lgho_atk": round(g_lgho_atk, 5),
                            "lgho_cln": round(g_lgho_cln, 5),
                        },
                        "obj": round(obj, 5),
                    }
                    all_records.append(rec)
                    print(
                        f"  [Grid {cq:6s} dt={dt:.2f} r={rd} b={bt:.1f}] loss=({cln_loss_m1:+.4f},{cln_loss_m2:+.4f}) | "
                        f"atk_ap50={mean_atk_ap50:.4f} | lrem_g=({g_lrem_atk:.3f}/{g_lrem_cln:.3f}) | "
                        f"lgho_g=({g_lgho_atk:.3f}/{g_lgho_cln:.3f}) | obj={obj:.4f}"
                    )

                    ok_1pct = (cln_loss_m1 <= 0.01 and cln_loss_m2 <= 0.01)
                    if ok_1pct and obj > best_obj:
                        best_obj = obj
                        best_feasible = rec
                    if max_cln_loss <= 0.02 and obj > best_fb_obj:
                        best_fb_obj = obj
                        best_fallback = rec

    chosen = best_feasible if best_feasible is not None else best_fallback
    assert chosen is not None, "触发 Re-plan if #2: 全部候选组干净 AP@0.5 损失均超过 2.0%！"
    gs = chosen["gate_stats"]
    print("\n=== 冻结局部门控超参 ===")
    print(json.dumps(chosen, indent=2, ensure_ascii=False))

    # Re-plan if #2 严格检验
    assert chosen["max_clean_ap50_loss"] <= 0.02, f"触发 Re-plan if #2: 干净 AP@0.5 损失超标: {chosen['max_clean_ap50_loss']}"
    assert gs["lrem_atk"] <= 0.65 and gs["lgho_atk"] <= 0.65, f"触发 Re-plan if #2: 受攻击区门控偏高: {gs}"
    assert gs["lrem_cln"] >= 0.85 and gs["lgho_cln"] >= 0.85, f"触发 Re-plan if #2: 未受攻击区门控误降偏低: {gs}"

    out_obj = OrderedDict({
        "meta": {
            "description": "步骤 ④-2 BUD-Fusion 位置容差有向语义局部门控与稳健 div_p95 冻结标定文件",
            "source_split": args.split_json,
            "source_base_calib": args.base_calib_json,
            "fit_total_frames": len(fit_indices),
            "fit_dev_frames": len(fit_dev_set),
            "frame_norm_mode": "div_p95",
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "frame_div_p95": frame_div_p95,
        "pixel_smap_t4_quantiles": pix_quantiles,
        "pixel_cos_s_t4_quantiles": pix_cos_quantiles,
        "gating_params": chosen["gating_params"],
        "fit_dev_baseline_ap50": {m: {k: round(v, 5) for k, v in base_ap50[m].items()} for m in ("m1", "m2")},
        "fit_dev_chosen_eval": chosen,
        "fit_dev_grid_records": all_records,
    })

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)
    print(f"\n[Step 2 完成] 已写出冻结门控标定：{args.out_json} (耗时 {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
