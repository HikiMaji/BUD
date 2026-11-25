# -*- coding: utf-8 -*-
"""
步骤 ④-3：BUD-Fusion HEAL m1+m2 320 帧全新 Dev 序列全维度端到端评测脚本。

严格遵循零泄露与学术统计规范：
1. 仅从与 Fit 11 条序列严格互斥的 8 条 Dev 序列中按比例均衡抽取 320 帧；
2. 7 工况（Clean、0.5σ/1.0σ/2.0σ/ε=4.0 全局参考 PGD、局部擦除 1.0σ、局部鬼影 2.0σ）
   × 4 方法（Ego-Only、No-Defense、Whole-CAV-Drop 整车降权基线、BUD-Fusion 局部门控）
   × m1/m2 双模态全覆盖，并支持按序列断点缓存续跑；
3. 帧级校准严格使用稳健 div_p95（坚决不用 med_p95），同时报告冻结门限 (>1.0) 与等 5% 总误报率双口径；
4. 客观统计本车盲区 GT 车召回率，及按「诚实协作车在本车盲区补全真车数 (0 / 1 / >=2)」分组的
   Tier 2 vs Tier 4 误报率对比；
5. 全指标给出 2,000 次按序列 Block-Bootstrap 95% 置信区间，并自动输出 3 张高清学术图。
"""

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
import torch
import torch.nn.functional as F

PROJ_ROOT = "/root/autodl-tmp/proj_dir2_hetero_trust"
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from src_v2_mm.bud_fusion_heal import (
    BUDFusionHEALAuditor,
    build_gated_fuse_fn,
    build_whole_cav_reject_fuse_fn,
    compute_shift_tolerant_semantic_maps,
)
from src_v2_mm.heal_attacks import (
    generate_local_ghost_spoofing_heal,
    generate_local_removal_heal,
    generate_pgd_removal_heal,
    generate_scaled_pgd_removal_heal,
)
from src_v2_mm.heal_base_wrapper import HEALBaseWrapper
from src_v2_mm.step4_1_v2_core import (
    EVAL_SEQ_NAMES,
    FIT_SEQ_NAMES,
    GRID_RES,
    MIN_XY,
    PROTO_H,
    PROTO_W,
    TRAIN_FULL19_DIR,
    _alloc_balanced_indices,
    _masked_avg_pool3x3,
    build_bev_grid_xy_256,
    extract_frame_context_and_clean,
    find_exact_fpr_threshold,
    load_ego_lidar_np,
)

DEFAULT_SPLIT = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_1_v2_split_plan.json")
DEFAULT_BASE_CALIB = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_1_v2_frozen_calibrator.json")
DEFAULT_GATING_CALIB = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_2_bud_gating_calibrator.json")
DEFAULT_CACHE_DIR = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_3_seq_cache")
DEFAULT_EVAL_SPLIT_OUT = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_3_eval320_split.json")
DEFAULT_OUT_JSON = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_3_bud_fusion_320_report.json")
DEFAULT_FIG_DIR = os.path.join(PROJ_ROOT, "results_rq1_v2_mm")

CONDITIONS_7 = [
    "clean",
    "pgd_0.5sigma",
    "pgd_1.0sigma",
    "pgd_2.0sigma",
    "pgd_eps4.0",
    "local_removal_1.0sigma",
    "local_ghost_2.0sigma",
]
METHODS_4 = ["ego_only", "no_defense", "whole_cav_drop", "bud_fusion"]
MODALITIES_2 = ["m1", "m2"]
BLIND_GROUPS_3 = ["0", "1", "ge2"]
BOOT_N = 2000
BOOT_SEED = 20261120


def _atomic_write_json(path: str, obj: dict):
    """原子写入 JSON 文件，防止中途中断损坏断点缓存。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _slim_batch_for_postprocess(batch_dict: dict) -> dict:
    """只保留 dataset.post_process 与 generate_gt_bbx 所需字段在 CPU 内存，降低内存占用。"""
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


def _fast_voc_ap(tp_arr: np.ndarray, fp_arr: np.ndarray, score_arr: np.ndarray, gt_total: int) -> float:
    """NumPy 向量化 VOC-2010 AP（与 HEAL eval_utils.calculate_ap 严格等价）。"""
    if gt_total <= 0 or score_arr.size == 0:
        return 0.0
    order = np.argsort(-score_arr)
    tp_c = np.cumsum(tp_arr[order], dtype=np.float64)
    fp_c = np.cumsum(fp_arr[order], dtype=np.float64)
    rec = tp_c / float(gt_total)
    prec = tp_c / (fp_c + tp_c)

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))


def _fast_voc_ap_pair(
    tp50: np.ndarray, fp50: np.ndarray, tp70: np.ndarray, fp70: np.ndarray, score: np.ndarray, gt_total: int
) -> tuple:
    """一次排序同时计算 IoU@0.5 与 IoU@0.7 的 VOC-2010 AP，提速 2 倍。"""
    if gt_total <= 0 or score.size == 0:
        return 0.0, 0.0
    order = np.argsort(-score)
    aps = []
    for tp_a, fp_a in ((tp50, fp50), (tp70, fp70)):
        tp_c = np.cumsum(tp_a[order], dtype=np.float64)
        fp_c = np.cumsum(fp_a[order], dtype=np.float64)
        rec = tp_c / float(gt_total)
        prec = tp_c / (fp_c + tp_c)
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]
        idx = np.nonzero(mrec[1:] != mrec[:-1])[0] + 1
        aps.append(float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx])))
    return aps[0], aps[1]


def _match_tp_fp_and_hits(pred_boxes, pred_score, gt_np: np.ndarray, gt_polygons: list, iou_thresh: float):
    """复刻官方 eval_utils.caluclate_tp_fp 贪心匹配逻辑，额外返回各 GT 是否命中布尔列表。"""
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
        det_p = det_polys[int(order[k])]
        if len(rem_polys) == 0:
            fp_list.append(1)
            tp_list.append(0)
            continue
        ious = common_utils.compute_iou(det_p, rem_polys)
        if np.max(ious) < iou_thresh:
            fp_list.append(1)
            tp_list.append(0)
            continue
        fp_list.append(0)
        tp_list.append(1)
        best_j = int(np.argmax(ious))
        rem_polys.pop(best_j)
        orig_i = rem_idxs.pop(best_j)
        matched_gt[orig_i] = True

    return tp_list, fp_list, sc_sorted.tolist(), gt_cnt, matched_gt


@torch.no_grad()
def _extract_out_and_dev_maps(
    wrapper,
    p_dict: dict,
    feat_in: torch.Tensor,
    p_ego_cls: torch.Tensor,
    cov_ego: torch.Tensor,
    k_tol: int = 5,
) -> tuple:
    """单次原生协作前向同时返回预测字典 out、4 通道偏差图及位置容差有向语义冲突图 (d_drop_tol, d_add_tol)。"""
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
    return out, dev_4ch, d_drop_tol, d_add_tol


def _build_eval320_split(wrapper, ds_m2, data_dir: str, split_plan: dict, out_path: str) -> dict:
    """从与 Fit 严格互斥的 Dev 序列均衡抽取 320 帧，持久化划分方案。"""
    sample_seqs = wrapper.get_sample_sequences(ds_m2)
    seq_to_idxs = OrderedDict()
    for i, s in enumerate(sample_seqs):
        seq_to_idxs.setdefault(s, []).append(i)

    eval_seq_names = list(split_plan["eval_sequences"].keys())
    fit_seq_names = list(split_plan["fit_sequences"].keys())
    fit_indices_set = set(split_plan.get("fit_indices", []))
    assert set(eval_seq_names).isdisjoint(set(fit_seq_names)), "Dev 与 Fit 序列存在重叠！"

    if os.path.abspath(data_dir) == os.path.abspath(TRAIN_FULL19_DIR):
        assert set(eval_seq_names) == set(EVAL_SEQ_NAMES), "Dev 序列名集合与官方锁定清单不符！"
        assert set(seq_to_idxs.keys()) == set(FIT_SEQ_NAMES + EVAL_SEQ_NAMES), "目录序列集合发生变化！"

    flat_items, seq_plan = _alloc_balanced_indices(
        eval_seq_names, seq_to_idxs, data_dir, target_total=320, min_per_seq=24
    )
    sampled_set = {x[0] for x in flat_items}
    assert len(flat_items) == 320, f"期望抽取 320 帧，实际 {len(flat_items)}"
    assert len(sampled_set) == 320, "抽样索引存在重复！"
    assert sampled_set.isdisjoint(fit_indices_set), "Dev-320 与 Fit 索引存在交集！"

    eval320 = OrderedDict({
        "dataset_dir": data_dir,
        "total_frames": 320,
        "num_sequences": len(eval_seq_names),
        "seq_order": eval_seq_names,
        "sequences": seq_plan,
        "all_indices": [x[0] for x in flat_items],
    })
    _atomic_write_json(out_path, eval320)
    return eval320


def _analyze_frame_blind_gts(ds_m1, idx: int, ego_pb, ego_ps, ego_gb, cov_2d: torch.Tensor) -> dict:
    """
    判定单帧客观盲区 GT 车：
    落入协作者有效区 cov_ego ∧ 本车 64 线原始点云命中数 <= 5 ∧ Ego-Only(IoU@0.5) 漏检。
    """
    from opencood.utils import common_utils

    gt_np = common_utils.torch_tensor_to_numpy(ego_gb)
    gt_polys = list(common_utils.convert_format(gt_np))

    ego_res_05 = _match_tp_fp_and_hits(ego_pb, ego_ps, gt_np, gt_polys, 0.5)
    ego_res_07 = _match_tp_fp_and_hits(ego_pb, ego_ps, gt_np, gt_polys, 0.7)
    ego_matched_05 = ego_res_05[4]

    ego_pts = load_ego_lidar_np(ds_m1, idx)
    cov_np = cov_2d.detach().cpu().numpy()

    blind_indices = []
    blind_iy_ix = {}
    pts_all_gt = []
    gt_in_cov_cnt = 0

    for i in range(gt_np.shape[0]):
        cx = float(gt_np[i, :4, 0].mean())
        cy = float(gt_np[i, :4, 1].mean())
        ix = int(math.floor((cx - MIN_XY) / GRID_RES))
        iy = int(math.floor((cy - MIN_XY) / GRID_RES))
        in_cov = (0 <= ix < PROTO_W) and (0 <= iy < PROTO_H) and bool(cov_np[iy, ix])

        poly_xy = np.asarray(gt_polys[i].exterior.coords)[:, :2]
        x_min, x_max = float(poly_xy[:, 0].min()), float(poly_xy[:, 0].max())
        y_min, y_max = float(poly_xy[:, 1].min()), float(poly_xy[:, 1].max())
        z_min = float(gt_np[i, :, 2].min()) - 0.3
        z_max = float(gt_np[i, :, 2].max()) + 0.3

        if ego_pts.shape[0] > 0:
            m_box = (
                (ego_pts[:, 0] >= x_min) & (ego_pts[:, 0] <= x_max)
                & (ego_pts[:, 1] >= y_min) & (ego_pts[:, 1] <= y_max)
                & (ego_pts[:, 2] >= z_min) & (ego_pts[:, 2] <= z_max)
            )
            if m_box.any():
                inside = MplPath(poly_xy).contains_points(ego_pts[m_box, :2])
                n_pts = int(inside.sum())
            else:
                n_pts = 0
        else:
            n_pts = 0

        pts_all_gt.append(n_pts)
        if in_cov:
            gt_in_cov_cnt += 1
        if in_cov and n_pts <= 5 and (not ego_matched_05[i]):
            blind_indices.append(i)
            blind_iy_ix[i] = (iy, ix)

    return {
        "gt_np": gt_np,
        "gt_polys": gt_polys,
        "ego_res_05": ego_res_05,
        "ego_res_07": ego_res_07,
        "blind_indices": blind_indices,
        "blind_iy_ix": blind_iy_ix,
        "gt_in_cov_cnt": gt_in_cov_cnt,
        "mean_pts_gt": float(np.mean(pts_all_gt)) if pts_all_gt else 0.0,
    }


def _generate_condition_feature(wrapper, b: dict, p_dict: dict, mod: str, st_name: str, idx: int):
    """按工况生成传输层输入特征（必要时运行 PGD），m1 保持与 m2 相同的有效裁剪空间域。"""
    feat = p_dict["heter_feature_2d"]
    if st_name == "clean":
        return feat, None

    rlen = p_dict["record_len"]
    amat = p_dict["affine_matrix"]
    mods_l = list(p_dict["agent_modality_list"])
    vmask = p_dict["cam_valid_mask"]
    preg = b["cov_local_m1"] if mod == "m1" else None

    if st_name == "pgd_0.5sigma":
        return generate_scaled_pgd_removal_heal(wrapper, feat, rlen, amat, mods_l, vmask, 1, 0.5, 8, preg)
    if st_name == "pgd_1.0sigma":
        return generate_scaled_pgd_removal_heal(wrapper, feat, rlen, amat, mods_l, vmask, 1, 1.0, 8, preg)
    if st_name == "pgd_2.0sigma":
        return generate_scaled_pgd_removal_heal(wrapper, feat, rlen, amat, mods_l, vmask, 1, 2.0, 8, preg)
    if st_name == "pgd_eps4.0":
        return generate_pgd_removal_heal(wrapper, feat, rlen, amat, mods_l, vmask, 1, 4.0, 10, preg)
    if st_name == "local_removal_1.0sigma":
        return generate_local_removal_heal(wrapper, feat, rlen, amat, mods_l, vmask, 1, 2, 5, 1.0, 10, preg)
    if st_name == "local_ghost_2.0sigma":
        return generate_local_ghost_spoofing_heal(wrapper, feat, rlen, amat, mods_l, vmask, 1, 2, 5, 2.0, 10, idx, preg)
    raise ValueError(f"未知工况: {st_name}")


def _new_seq_ap_buckets():
    return {
        m: {
            st: {
                md: {
                    "0.5": {"tp": [], "fp": [], "score": [], "gt": 0},
                    "0.7": {"tp": [], "fp": [], "score": [], "gt": 0},
                }
                for md in METHODS_4
            }
            for st in CONDITIONS_7
        }
        for m in MODALITIES_2
    }


def _append_ap_res(bucket: dict, res_05: tuple, res_07: tuple):
    bucket["0.5"]["tp"].extend(res_05[0])
    bucket["0.5"]["fp"].extend(res_05[1])
    bucket["0.5"]["score"].extend(res_05[2])
    bucket["0.5"]["gt"] += int(res_05[3])
    bucket["0.7"]["tp"].extend(res_07[0])
    bucket["0.7"]["fp"].extend(res_07[1])
    bucket["0.7"]["score"].extend(res_07[2])
    bucket["0.7"]["gt"] += int(res_07[3])


def _run_single_sequence(
    seq_name: str,
    seq_indices: list,
    cache_path: str,
    wrapper,
    ds_m1,
    ds_m2,
    gx: torch.Tensor,
    gy: torch.Tensor,
    auditor: BUDFusionHEALAuditor,
    k_tol: int = 5,
) -> dict:
    """运行单条 Dev 序列全部帧评估并原子写入缓存，若完整缓存存在则直接加载。"""
    expected_records = len(seq_indices) * len(MODALITIES_2) * len(CONDITIONS_7)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("completed") and len(cached.get("frame_records", [])) == expected_records:
                print(f"  [缓存复用] {seq_name} ({len(seq_indices)} 帧)")
                return cached
        except Exception as e:
            print(f"  [缓存损坏重跑] {seq_name}: {e}")

    from opencood.tools import train_utils

    t_seq = time.time()
    ap_buckets = _new_seq_ap_buckets()
    frame_records = []
    seq_gt_pts_means = []

    for f_pos, idx in enumerate(seq_indices):
        b = extract_frame_context_and_clean(wrapper, ds_m1, ds_m2, idx, gx, gy)
        cov_2d = b["cov_ego"][0, 0]
        d_idx = b["d_idx"]
        v_idx = b["v_idx"]

        raw_b1 = train_utils.to_device(ds_m1.collate_batch_test([ds_m1[idx]]), "cpu")
        b_post = train_utils.to_device(_slim_batch_for_postprocess(raw_b1), wrapper.device_obj)

        with torch.no_grad():
            o_ego = wrapper.forward_single_agent(b["p1"]["heter_feature_2d"][0:1])
            ego_pb, ego_ps, ego_gb = wrapper.decode_boxes(b_post, o_ego, ds_m1)

        blind_info = _analyze_frame_blind_gts(ds_m1, idx, ego_pb, ego_ps, ego_gb, cov_2d)
        gt_np = blind_info["gt_np"]
        gt_polys = blind_info["gt_polys"]
        blind_idxs = blind_info["blind_indices"]
        blind_iy_ix = blind_info["blind_iy_ix"]
        n_blind_gt = len(blind_idxs)
        ego_hits_blind = sum(1 for gi in blind_idxs if blind_info["ego_res_05"][4][gi])
        seq_gt_pts_means.append(blind_info["mean_pts_gt"])

        for mod, p_dict in (("m1", b["p1"]), ("m2", b["p2"])):
            rlen = p_dict["record_len"]
            amat = p_dict["affine_matrix"]
            mods_l = list(p_dict["agent_modality_list"])

            for st_name in CONDITIONS_7:
                feat_in, atk_meta = _generate_condition_feature(wrapper, b, p_dict, mod, st_name, idx)

                with torch.no_grad():
                    out_nodef, dev_4ch, d_drop_tol, d_add_tol = _extract_out_and_dev_maps(
                        wrapper, p_dict, feat_in, b["p_ego_cls"], b["cov_ego"], k_tol=k_tol
                    )
                    nodef_pb, nodef_ps, _ = wrapper.decode_boxes(b_post, out_nodef, ds_m1)
                    nodef_05 = _match_tp_fp_and_hits(nodef_pb, nodef_ps, gt_np, gt_polys, 0.5)
                    nodef_07 = _match_tp_fp_and_hits(nodef_pb, nodef_ps, gt_np, gt_polys, 0.7)

                    ar = auditor.audit_dev_maps(
                        dev_4ch, cov_2d, d_idx, v_idx, mod, build_gate=True,
                        d_drop_tol=d_drop_tol, d_add_tol=d_add_tol,
                    )
                    gate_l0 = ar["gate_L0"]

                    # Whole-CAV-Drop 基线：仅依据 Tier 4 div_p95 帧分数 > 1.0 整车丢弃
                    if ar["score_t4"] > 1.0:
                        out_rej = wrapper.forward_from_protocol(
                            feat_in, rlen, amat, mods_l, custom_fuse_fn=build_whole_cav_reject_fuse_fn(True)
                        )
                        rej_pb, rej_ps, _ = wrapper.decode_boxes(b_post, out_rej, ds_m1)
                        rej_05 = _match_tp_fp_and_hits(rej_pb, rej_ps, gt_np, gt_polys, 0.5)
                        rej_07 = _match_tp_fp_and_hits(rej_pb, rej_ps, gt_np, gt_polys, 0.7)
                    else:
                        rej_05, rej_07 = nodef_05, nodef_07

                    # BUD-Fusion 局部门控融合：在 clean 与局部攻击下始终前向采集 w_cav_L0
                    g_min = float(gate_l0[0, 0][cov_2d].min().item()) if cov_2d.any() else 1.0
                    must_run_bud = st_name in ("clean", "local_removal_1.0sigma", "local_ghost_2.0sigma")
                    if must_run_bud or g_min < 1.0 - 1e-6:
                        out_bud = wrapper.forward_from_protocol(
                            feat_in, rlen, amat, mods_l, custom_fuse_fn=build_gated_fuse_fn(gate_l0)
                        )
                        bud_pb, bud_ps, _ = wrapper.decode_boxes(b_post, out_bud, ds_m1)
                        bud_05 = _match_tp_fp_and_hits(bud_pb, bud_ps, gt_np, gt_polys, 0.5)
                        bud_07 = _match_tp_fp_and_hits(bud_pb, bud_ps, gt_np, gt_polys, 0.7)
                        w_cav_l0 = out_bud["fusion_diag"].get("w_cav_L0")
                    else:
                        bud_05, bud_07 = nodef_05, nodef_07
                        w_cav_l0 = None

                    # 记录 AP buckets
                    _append_ap_res(ap_buckets[mod][st_name]["ego_only"], blind_info["ego_res_05"], blind_info["ego_res_07"])
                    _append_ap_res(ap_buckets[mod][st_name]["no_defense"], nodef_05, nodef_07)
                    _append_ap_res(ap_buckets[mod][st_name]["whole_cav_drop"], rej_05, rej_07)
                    _append_ap_res(ap_buckets[mod][st_name]["bud_fusion"], bud_05, bud_07)

                    # 盲区命中计数
                    hits_nodef = sum(1 for gi in blind_idxs if nodef_05[4][gi])
                    hits_rej = sum(1 for gi in blind_idxs if rej_05[4][gi])
                    hits_bud = sum(1 for gi in blind_idxs if bud_05[4][gi])

                    rec = {
                        "sample_idx": int(idx),
                        "mod": mod,
                        "state": st_name,
                        "score_t2": round(float(ar["score_t2"]), 6),
                        "score_t4": round(float(ar["score_t4"]), 6),
                        "raw_t2": round(float(ar["raw_t2"]), 6),
                        "raw_t4": round(float(ar["raw_t4"]), 6),
                        "n_blind_gt": int(n_blind_gt),
                        "blind_hits": {
                            "ego_only": int(ego_hits_blind),
                            "no_defense": int(hits_nodef),
                            "whole_cav_drop": int(hits_rej),
                            "bud_fusion": int(hits_bud),
                        },
                        "n_blind_comp": None,
                        "gate_clean_valid": None,
                        "w_cav_l0_valid": None,
                        "gate_atk_region": None,
                        "gate_cln_region": None,
                        "w_cav_l0_atk": None,
                        "w_cav_l0_cln": None,
                        "blind_comp_gates": [],
                        "blind_comp_wcavs": [],
                    }

                    if st_name == "clean":
                        comp_idxs = [gi for gi in blind_idxs if nodef_05[4][gi]]
                        rec["n_blind_comp"] = int(len(comp_idxs))
                        if cov_2d.any():
                            rec["gate_clean_valid"] = round(float(gate_l0[0, 0][cov_2d].mean().item()), 6)
                            if w_cav_l0 is not None:
                                rec["w_cav_l0_valid"] = round(float(w_cav_l0[0, 0][cov_2d].mean().item()), 6)
                        for gi in comp_idxs:
                            iy, ix = blind_iy_ix[gi]
                            patch_m = torch.zeros_like(cov_2d)
                            patch_m[max(0, iy - 1) : min(PROTO_H, iy + 2), max(0, ix - 1) : min(PROTO_W, ix + 2)] = True
                            pm = patch_m & cov_2d
                            if pm.any():
                                rec["blind_comp_gates"].append(round(float(gate_l0[0, 0][pm].mean().item()), 6))
                                if w_cav_l0 is not None:
                                    rec["blind_comp_wcavs"].append(round(float(w_cav_l0[0, 0][pm].mean().item()), 6))

                    elif st_name in ("local_removal_1.0sigma", "local_ghost_2.0sigma") and atk_meta is not None:
                        am_raw = atk_meta.get("attacked_mask_in_ego")
                        if am_raw is not None:
                            atk_m = (am_raw[0, 0] > 0.5) & cov_2d
                            cln_m = cov_2d & (~atk_m)
                            if atk_m.any():
                                rec["gate_atk_region"] = round(float(gate_l0[0, 0][atk_m].mean().item()), 6)
                                if w_cav_l0 is not None:
                                    rec["w_cav_l0_atk"] = round(float(w_cav_l0[0, 0][atk_m].mean().item()), 6)
                            if cln_m.any():
                                rec["gate_cln_region"] = round(float(gate_l0[0, 0][cln_m].mean().item()), 6)
                                if w_cav_l0 is not None:
                                    rec["w_cav_l0_cln"] = round(float(w_cav_l0[0, 0][cln_m].mean().item()), 6)

                    frame_records.append(rec)

                del feat_in, atk_meta

        del b, raw_b1, b_post
        if (f_pos + 1) % 20 == 0 or (f_pos + 1) == len(seq_indices):
            torch.cuda.empty_cache()
            print(f"    [{seq_name}] 进度 {f_pos + 1}/{len(seq_indices)} ({time.time() - t_seq:.1f}s)", flush=True)

    seq_out = OrderedDict({
        "seq_name": seq_name,
        "completed": True,
        "num_frames": len(seq_indices),
        "mean_pts_per_gt": round(float(np.mean(seq_gt_pts_means)) if seq_gt_pts_means else 0.0, 2),
        "elapsed_sec": round(time.time() - t_seq, 2),
        "ap_buckets": ap_buckets,
        "frame_records": frame_records,
    })
    _atomic_write_json(cache_path, seq_out)
    return seq_out


def _summarize_with_bootstrap(seq_caches: list, seq_order: list, gating_calib: dict) -> dict:
    """基于 8 条 Dev 序列按序列 2,000 次 Block-Bootstrap 汇总全维度指标与 95% CI。"""
    n_seq = len(seq_order)
    rng = np.random.default_rng(BOOT_SEED)
    boot_mat = rng.integers(0, n_seq, size=(BOOT_N, n_seq))

    def _ci_stats(point_val: float, boot_vals: np.ndarray) -> dict:
        lo, hi = np.percentile(boot_vals, [2.5, 97.5])
        return {
            "val": round(float(point_val), 5),
            "ci95": [round(float(lo), 5), round(float(hi), 5)],
        }

    # 预处理各序列 AP numpy 数组与逐帧索引列表，提速 Bootstrap
    seq_ap_np = []
    seq_clean_recs = []
    seq_all_recs = []
    for sc in seq_caches:
        ap_d = {}
        for m in MODALITIES_2:
            ap_d[m] = {}
            for st in CONDITIONS_7:
                ap_d[m][st] = {}
                for md in METHODS_4:
                    b50 = sc["ap_buckets"][m][st][md]["0.5"]
                    b70 = sc["ap_buckets"][m][st][md]["0.7"]
                    ap_d[m][st][md] = (
                        np.asarray(b50["tp"], dtype=np.float64),
                        np.asarray(b50["fp"], dtype=np.float64),
                        np.asarray(b70["tp"], dtype=np.float64),
                        np.asarray(b70["fp"], dtype=np.float64),
                        np.asarray(b50["score"], dtype=np.float64),
                        int(b50["gt"]),
                    )
        seq_ap_np.append(ap_d)
        seq_all_recs.append(sc["frame_records"])
        seq_clean_recs.append([r for r in sc["frame_records"] if r["state"] == "clean"])

    # 1. AP@0.5, AP@0.7 与 Blind-Spot Recall@0.5（带 95% CI）
    ap_summary = OrderedDict()
    blind_recall_summary = OrderedDict()

    # 预处理每条序列各 (mod, state) 下的盲区 gt 总数与 4 方法命中总数
    seq_blind_counts = []
    for recs in seq_all_recs:
        bd = {(m, st): {md: 0 for md in METHODS_4} for m in MODALITIES_2 for st in CONDITIONS_7}
        bd_den = {(m, st): 0 for m in MODALITIES_2 for st in CONDITIONS_7}
        for r in recs:
            k = (r["mod"], r["state"])
            bd_den[k] += int(r["n_blind_gt"])
            for md in METHODS_4:
                bd[k][md] += int(r["blind_hits"][md])
        seq_blind_counts.append((bd, bd_den))

    for m in MODALITIES_2:
        ap_summary[m] = OrderedDict()
        blind_recall_summary[m] = OrderedDict()
        for st in CONDITIONS_7:
            ap_summary[m][st] = OrderedDict()
            blind_recall_summary[m][st] = OrderedDict()
            for md in METHODS_4:
                # 点估计 AP
                tp50_all = np.concatenate([seq_ap_np[i][m][st][md][0] for i in range(n_seq)])
                fp50_all = np.concatenate([seq_ap_np[i][m][st][md][1] for i in range(n_seq)])
                tp70_all = np.concatenate([seq_ap_np[i][m][st][md][2] for i in range(n_seq)])
                fp70_all = np.concatenate([seq_ap_np[i][m][st][md][3] for i in range(n_seq)])
                sc_all = np.concatenate([seq_ap_np[i][m][st][md][4] for i in range(n_seq)])
                gt_all = sum(seq_ap_np[i][m][st][md][5] for i in range(n_seq))
                pt50, pt70 = _fast_voc_ap_pair(tp50_all, fp50_all, tp70_all, fp70_all, sc_all, gt_all)

                # Bootstrap AP
                b50 = np.empty(BOOT_N, dtype=np.float64)
                b70 = np.empty(BOOT_N, dtype=np.float64)
                for bi in range(BOOT_N):
                    idxs = boot_mat[bi]
                    t5 = np.concatenate([seq_ap_np[i][m][st][md][0] for i in idxs])
                    f5 = np.concatenate([seq_ap_np[i][m][st][md][1] for i in idxs])
                    t7 = np.concatenate([seq_ap_np[i][m][st][md][2] for i in idxs])
                    f7 = np.concatenate([seq_ap_np[i][m][st][md][3] for i in idxs])
                    ss = np.concatenate([seq_ap_np[i][m][st][md][4] for i in idxs])
                    gg = sum(seq_ap_np[i][m][st][md][5] for i in idxs)
                    b50[bi], b70[bi] = _fast_voc_ap_pair(t5, f5, t7, f7, ss, gg)

                ap_summary[m][st][md] = {
                    "ap50": _ci_stats(pt50, b50),
                    "ap70": _ci_stats(pt70, b70),
                }

                # Blind Recall 点估计 + Bootstrap
                num_vec = np.array([seq_blind_counts[i][0][(m, st)][md] for i in range(n_seq)], dtype=np.float64)
                den_vec = np.array([seq_blind_counts[i][1][(m, st)] for i in range(n_seq)], dtype=np.float64)
                pt_rec = float(num_vec.sum() / max(den_vec.sum(), 1.0))
                b_num = num_vec[boot_mat].sum(axis=1)
                b_den = np.maximum(den_vec[boot_mat].sum(axis=1), 1.0)
                b_rec = b_num / b_den
                rec_st = _ci_stats(pt_rec, b_rec)
                rec_st["hits"] = int(num_vec.sum())
                rec_st["total_blind_gt"] = int(den_vec.sum())
                blind_recall_summary[m][st][md] = rec_st

    # 2. 干净帧误报率与按盲区补全分组对比（Tier 2 vs Tier 4，严格 div_p95 双口径）
    all_clean_flat = [r for recs in seq_clean_recs for r in recs]
    th_eq5_t2 = find_exact_fpr_threshold(np.array([r["score_t2"] for r in all_clean_flat], dtype=np.float64), 0.05)
    th_eq5_t4 = find_exact_fpr_threshold(np.array([r["score_t4"] for r in all_clean_flat], dtype=np.float64), 0.05)

    def _group_key(n_comp: int) -> str:
        if n_comp <= 0:
            return "0"
        if n_comp == 1:
            return "1"
        return "ge2"

    regimes = {
        "frozen_div_p95": {"t2_th": 1.0, "t4_th": 1.0},
        "equal_5pct_overall": {"t2_th": float(th_eq5_t2), "t4_th": float(th_eq5_t4)},
    }

    fpr_summary = OrderedDict({"equal_5pct_thresholds": {"tier2": round(th_eq5_t2, 6), "tier4": round(th_eq5_t4, 6)}})
    for reg_name, th_map in regimes.items():
        reg_out = OrderedDict()
        for tier_key, sc_field, th_val in (
            ("tier2_sensor_only", "score_t2", th_map["t2_th"]),
            ("tier4_sensor_dist_vis", "score_t4", th_map["t4_th"]),
        ):
            t_out = OrderedDict()
            for mod_scope in ("m1", "m2", "overall"):
                mod_filter = {"m1", "m2"} if mod_scope == "overall" else {mod_scope}
                m_out = OrderedDict()
                # 全量 320 帧（或 640 帧）汇总 + 三组盲区补全分组
                for g_scope in ["all", "0", "1", "ge2"]:
                    hit_s = np.zeros(n_seq, dtype=np.float64)
                    cnt_s = np.zeros(n_seq, dtype=np.float64)
                    for si, recs in enumerate(seq_clean_recs):
                        for r in recs:
                            if r["mod"] not in mod_filter:
                                continue
                            if g_scope != "all" and _group_key(int(r["n_blind_comp"])) != g_scope:
                                continue
                            cnt_s[si] += 1.0
                            if float(r[sc_field]) > th_val:
                                hit_s[si] += 1.0
                    pt_fpr = float(hit_s.sum() / max(cnt_s.sum(), 1.0))
                    b_den = np.maximum(cnt_s[boot_mat].sum(axis=1), 1.0)
                    b_fpr = hit_s[boot_mat].sum(axis=1) / b_den
                    st_dict = _ci_stats(pt_fpr, b_fpr)
                    st_dict["fp"] = int(hit_s.sum())
                    st_dict["n"] = int(cnt_s.sum())
                    m_out[g_scope] = st_dict
                t_out[mod_scope] = m_out
            reg_out[tier_key] = t_out
        fpr_summary[reg_name] = reg_out

    # 3. 局部门控权重时空选择性 + 盲区补全真车处权重保留（带 95% CI）
    def _boot_seq_mean(seq_lists: list) -> dict:
        flat = [v for lst in seq_lists for v in lst]
        pt = float(np.mean(flat)) if flat else 0.0
        sum_s = np.array([sum(lst) for lst in seq_lists], dtype=np.float64)
        cnt_s = np.array([len(lst) for lst in seq_lists], dtype=np.float64)
        b_den = np.maximum(cnt_s[boot_mat].sum(axis=1), 1.0)
        b_vals = sum_s[boot_mat].sum(axis=1) / b_den
        res = _ci_stats(pt, b_vals)
        res["count"] = int(cnt_s.sum())
        if flat:
            res["p5"] = round(float(np.percentile(flat, 5)), 5)
        return res

    spatial_summary = OrderedDict()
    for m in MODALITIES_2:
        clean_g, clean_w = [[] for _ in range(n_seq)], [[] for _ in range(n_seq)]
        bcomp_g, bcomp_w = [[] for _ in range(n_seq)], [[] for _ in range(n_seq)]
        lrem_g_atk, lrem_g_cln = [[] for _ in range(n_seq)], [[] for _ in range(n_seq)]
        lrem_w_atk, lrem_w_cln = [[] for _ in range(n_seq)], [[] for _ in range(n_seq)]
        lgho_g_atk, lgho_g_cln = [[] for _ in range(n_seq)], [[] for _ in range(n_seq)]
        lgho_w_atk, lgho_w_cln = [[] for _ in range(n_seq)], [[] for _ in range(n_seq)]

        for si, recs in enumerate(seq_all_recs):
            for r in recs:
                if r["mod"] != m:
                    continue
                if r["state"] == "clean":
                    if r["gate_clean_valid"] is not None:
                        clean_g[si].append(r["gate_clean_valid"])
                    if r["w_cav_l0_valid"] is not None:
                        clean_w[si].append(r["w_cav_l0_valid"])
                    bcomp_g[si].extend(r["blind_comp_gates"])
                    bcomp_w[si].extend(r["blind_comp_wcavs"])
                elif r["state"] == "local_removal_1.0sigma":
                    if r["gate_atk_region"] is not None:
                        lrem_g_atk[si].append(r["gate_atk_region"])
                    if r["gate_cln_region"] is not None:
                        lrem_g_cln[si].append(r["gate_cln_region"])
                    if r["w_cav_l0_atk"] is not None:
                        lrem_w_atk[si].append(r["w_cav_l0_atk"])
                    if r["w_cav_l0_cln"] is not None:
                        lrem_w_cln[si].append(r["w_cav_l0_cln"])
                elif r["state"] == "local_ghost_2.0sigma":
                    if r["gate_atk_region"] is not None:
                        lgho_g_atk[si].append(r["gate_atk_region"])
                    if r["gate_cln_region"] is not None:
                        lgho_g_cln[si].append(r["gate_cln_region"])
                    if r["w_cav_l0_atk"] is not None:
                        lgho_w_atk[si].append(r["w_cav_l0_atk"])
                    if r["w_cav_l0_cln"] is not None:
                        lgho_w_cln[si].append(r["w_cav_l0_cln"])

        spatial_summary[m] = OrderedDict({
            "clean_valid_gate": _boot_seq_mean(clean_g),
            "clean_valid_wcav_l0": _boot_seq_mean(clean_w),
            "blind_comp_gt_gate": _boot_seq_mean(bcomp_g),
            "blind_comp_gt_wcav_l0": _boot_seq_mean(bcomp_w),
            "local_removal_gate_attacked": _boot_seq_mean(lrem_g_atk),
            "local_removal_gate_clean_region": _boot_seq_mean(lrem_g_cln),
            "local_removal_wcav_l0_attacked": _boot_seq_mean(lrem_w_atk),
            "local_removal_wcav_l0_clean_region": _boot_seq_mean(lrem_w_cln),
            "local_ghost_gate_attacked": _boot_seq_mean(lgho_g_atk),
            "local_ghost_gate_clean_region": _boot_seq_mean(lgho_g_cln),
            "local_ghost_wcav_l0_attacked": _boot_seq_mean(lgho_w_atk),
            "local_ghost_wcav_l0_clean_region": _boot_seq_mean(lgho_w_cln),
        })

    # 4. 盲区补全分组计数分布汇总
    group_counts = {
        m: {gk: int(fpr_summary["frozen_div_p95"]["tier4_sensor_dist_vis"][m][gk]["n"]) for gk in BLIND_GROUPS_3}
        for m in MODALITIES_2
    }

    return OrderedDict({
        "meta": {
            "description": "步骤 ④-3 BUD-Fusion HEAL m1+m2 Dev-320 全维度评测与 Bootstrap 95% CI 报告",
            "num_sequences": n_seq,
            "seq_names": seq_order,
            "total_frames": sum(sc["num_frames"] for sc in seq_caches),
            "bootstrap_iters": BOOT_N,
            "bootstrap_seed": BOOT_SEED,
            "frame_norm_mode": "div_p95",
            "frozen_gating_params": gating_calib["gating_params"],
            "frozen_frame_div_p95": gating_calib["frame_div_p95"],
            "blind_comp_group_counts": group_counts,
        },
        "ap_results": ap_summary,
        "blind_spot_recall": blind_recall_summary,
        "clean_fpr_and_blind_groups": fpr_summary,
        "spatial_weight_selectivity": spatial_summary,
    })


def _plot_all_figures(report: dict, fig_dir: str) -> list:
    """绘制 3 张高清防裁边论文级图（dpi=300, bbox_inches='tight', pad_inches=0.15）。"""
    os.makedirs(fig_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    out_paths = []
    ap_res = report["ap_results"]
    br_res = report["blind_spot_recall"]
    fpr_res = report["clean_fpr_and_blind_groups"]
    sp_res = report["spatial_weight_selectivity"]

    m_styles = {
        "ego_only": {"label": "Ego-Only (Single Agent)", "color": "#5C6B73", "ls": "--", "marker": "^"},
        "no_defense": {"label": "No-Defense (Native HEAL)", "color": "#D64541", "ls": "-", "marker": "o"},
        "whole_cav_drop": {"label": "Whole-CAV-Drop (Baseline)", "color": "#E67E22", "ls": "-.", "marker": "s"},
        "bud_fusion": {"label": "BUD-Fusion (Ours)", "color": "#1B8A5A", "ls": "-", "marker": "D"},
    }

    # --- 图 1：小预算曲线 AP@0.5 / AP@0.7（Clean -> 0.5σ -> 1.0σ -> 2.0σ -> ε=4.0）---
    budget_states = ["clean", "pgd_0.5sigma", "pgd_1.0sigma", "pgd_2.0sigma", "pgd_eps4.0"]
    budget_labels = ["Clean", "0.5σ\nGlobal PGD", "1.0σ\nGlobal PGD", "2.0σ\nGlobal PGD", "ε=4.0\nRef PGD"]
    xs = np.arange(len(budget_states))

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.4), sharex=True)
    for r_i, (iou_k, iou_title) in enumerate((("ap50", "AP@0.5 (%)"), ("ap70", "AP@0.7 (%)"))):
        for c_i, (mod, mod_title) in enumerate((("m1", "m1 Coll. (64L LiDAR)"), ("m2", "m2 Coll. (Camera-LSS)"))):
            ax = axes[r_i, c_i]
            for md in METHODS_4:
                st = m_styles[md]
                vals = np.array([ap_res[mod][s][md][iou_k]["val"] * 100.0 for s in budget_states])
                los = np.array([ap_res[mod][s][md][iou_k]["ci95"][0] * 100.0 for s in budget_states])
                his = np.array([ap_res[mod][s][md][iou_k]["ci95"][1] * 100.0 for s in budget_states])
                lw = 2.6 if md == "bud_fusion" else 2.0
                ms = 6.5 if md == "bud_fusion" else 5.5
                ax.plot(xs, vals, label=st["label"], color=st["color"], linestyle=st["ls"], marker=st["marker"], lw=lw, ms=ms, zorder=4 if md == "bud_fusion" else 3)
                ax.fill_between(xs, los, his, color=st["color"], alpha=0.12, zorder=1)
                if md == "bud_fusion":
                    ego_vals = [ap_res[mod][s]["ego_only"][iou_k]["val"] * 100.0 for s in budget_states]
                    wcd_vals = [ap_res[mod][s]["whole_cav_drop"][iou_k]["val"] * 100.0 for s in budget_states]
                    for xk, yk, ev, wv in zip(xs, vals, ego_vals, wcd_vals):
                        above = (yk >= max(ev, wv) - 0.5)
                        y_pos = yk + 2.4 if above else yk - 2.6
                        va_mode = "bottom" if above else "top"
                        ax.text(
                            xk, y_pos, f"{yk:.1f}", ha="center", va=va_mode, fontsize=8.3, fontweight="bold",
                            color=st["color"], bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.80),
                            zorder=6,
                        )
                elif md == "no_defense":
                    for xk, yk in zip(xs[1:], vals[1:]):
                        ax.text(
                            xk, yk - 3.6, f"{yk:.1f}", ha="center", va="top", fontsize=8.0, color=st["color"],
                            bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.75),
                            zorder=5,
                        )

            ax.set_title(f"{'abcd'[r_i * 2 + c_i].upper()}) {mod_title} — {iou_title}", fontweight="bold", pad=8)
            ax.set_ylabel(iou_title)
            ax.set_ylim(0, 106)
            ax.set_xticks(xs)
            ax.set_xticklabels(budget_labels)
            ax.grid(True, linestyle=":", alpha=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=4, frameon=True)
    fig.suptitle("Dev-320 (8 Unseen Seqs) Budget Curve AP with 95% Block-Bootstrap CI", fontsize=13.5, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.905])
    p1 = os.path.join(fig_dir, "fig_step4_budget_curve_ap.png")
    fig.savefig(p1, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    out_paths.append(p1)

    # --- 图 2：局部攻击 AP@0.5 + 盲区召回 + BUD-Fusion 时空权重选择性 ---
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.2))
    loc_states = ["clean", "local_removal_1.0sigma", "local_ghost_2.0sigma"]
    loc_labels = ["Clean", "Local Rem.\n(1.0σ, 1-2 Obj)", "Local Ghost\n(2.0σ, 1-2 Obj)"]
    bw = 0.19
    x3 = np.arange(len(loc_states))

    # (a) m1 AP@0.5, (b) m2 AP@0.5
    for c_i, (mod, mtitle) in enumerate((("m1", "m1 Coll. (64L LiDAR)"), ("m2", "m2 Coll. (Camera-LSS)"))):
        ax = axes[0, c_i]
        for mi, md in enumerate(METHODS_4):
            off = (mi - 1.5) * bw
            vals = np.array([ap_res[mod][s][md]["ap50"]["val"] * 100.0 for s in loc_states])
            los = np.array([ap_res[mod][s][md]["ap50"]["ci95"][0] * 100.0 for s in loc_states])
            his = np.array([ap_res[mod][s][md]["ap50"]["ci95"][1] * 100.0 for s in loc_states])
            yerr = np.vstack([np.maximum(vals - los, 0.0), np.maximum(his - vals, 0.0)])
            bars = ax.bar(
                x3 + off, vals, width=bw * 0.92, yerr=yerr, capsize=3,
                color=m_styles[md]["color"], edgecolor="black", linewidth=0.6, alpha=0.90,
                label=m_styles[md]["label"],
            )
            for bx, by, hi in zip(bars, vals, his):
                ax.text(bx.get_x() + bx.get_width() * 0.5, hi + 1.2, f"{by:.1f}", ha="center", va="bottom", fontsize=7.8)
        ax.set_title(f"({'AB'[c_i]}) {mtitle}: Local Attack AP@0.5 (%)", fontweight="bold", pad=8)
        ax.set_ylabel("AP@0.5 (%)")
        ax.set_ylim(0, 112)
        ax.set_xticks(x3)
        ax.set_xticklabels(loc_labels)
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    # (c) Blind-Spot Recall@0.5 (%) across Clean & Local Attacks (m1 & m2)
    ax_c = axes[1, 0]
    br_pairs = [
        ("m1", "clean", "m1\nClean"),
        ("m1", "local_removal_1.0sigma", "m1\nRem."),
        ("m1", "local_ghost_2.0sigma", "m1\nGhost"),
        ("m2", "clean", "m2\nClean"),
        ("m2", "local_removal_1.0sigma", "m2\nRem."),
        ("m2", "local_ghost_2.0sigma", "m2\nGhost"),
    ]
    x6 = np.arange(len(br_pairs))
    for mi, md in enumerate(METHODS_4):
        off = (mi - 1.5) * bw
        vals = np.array([br_res[m][s][md]["val"] * 100.0 for m, s, _ in br_pairs])
        los = np.array([br_res[m][s][md]["ci95"][0] * 100.0 for m, s, _ in br_pairs])
        his = np.array([br_res[m][s][md]["ci95"][1] * 100.0 for m, s, _ in br_pairs])
        yerr = np.vstack([np.maximum(vals - los, 0.0), np.maximum(his - vals, 0.0)])
        bars = ax_c.bar(
            x6 + off, vals, width=bw * 0.92, yerr=yerr, capsize=2.5,
            color=m_styles[md]["color"], edgecolor="black", linewidth=0.6, alpha=0.90,
        )
        for bx, by, hi in zip(bars, vals, his):
            ax_c.text(bx.get_x() + bx.get_width() * 0.5, hi + 1.0, f"{by:.0f}", ha="center", va="bottom", fontsize=7.5)
    ax_c.axvline(2.5, color="#888888", linestyle="--", linewidth=1.0)
    ax_c.set_title("(C) Blind-Spot Recall@0.5 (%): Zero-Point Ego Blind GTs in Cov.", fontweight="bold", pad=8)
    ax_c.set_ylabel("Blind-Spot Recall@0.5 (%)")
    ax_c.set_ylim(0, 112)
    ax_c.set_xticks(x6)
    ax_c.set_xticklabels([x[2] for x in br_pairs])
    ax_c.grid(True, axis="y", linestyle=":", alpha=0.5)

    # (d) Spatial Weight Selectivity of BUD-Fusion: Gate_L0 & w_cav_L0
    ax_d = axes[1, 1]
    sp_keys = [
        ("clean_valid_gate", "Clean\nValid"),
        ("blind_comp_gt_gate", "Blind GT\nComp."),
        ("local_removal_gate_clean_region", "Rem.\nClean Reg."),
        ("local_removal_gate_attacked", "Rem.\nAttacked"),
        ("local_ghost_gate_clean_region", "Ghost\nClean Reg."),
        ("local_ghost_gate_attacked", "Ghost\nAttacked"),
    ]
    w_keys = [
        "clean_valid_wcav_l0",
        "blind_comp_gt_wcav_l0",
        "local_removal_wcav_l0_clean_region",
        "local_removal_wcav_l0_attacked",
        "local_ghost_wcav_l0_clean_region",
        "local_ghost_wcav_l0_attacked",
    ]
    x_sp = np.arange(len(sp_keys))
    w_sp = 0.34
    vals_m1 = np.array([sp_res["m1"][k]["val"] for k, _ in sp_keys])
    vals_m2 = np.array([sp_res["m2"][k]["val"] for k, _ in sp_keys])
    wc_m1 = np.array([sp_res["m1"][wk]["val"] for wk in w_keys])
    wc_m2 = np.array([sp_res["m2"][wk]["val"] for wk in w_keys])

    b1 = ax_d.bar(x_sp - w_sp * 0.56, vals_m1, width=w_sp * 0.92, color="#2E86AB", edgecolor="black", linewidth=0.6, label="Gate_L0 (m1 LiDAR)")
    b2 = ax_d.bar(x_sp + w_sp * 0.56, vals_m2, width=w_sp * 0.92, color="#A23B72", edgecolor="black", linewidth=0.6, label="Gate_L0 (m2 Camera)")
    for bx, gv, wv in zip(b1, vals_m1, wc_m1):
        ax_d.text(bx.get_x() + bx.get_width() * 0.5, gv + 0.015, f"{gv:.2f}\nw={wv:.2f}", ha="center", va="bottom", fontsize=6.6, color="#1D5973")
    for bx, gv, wv in zip(b2, vals_m2, wc_m2):
        ax_d.text(bx.get_x() + bx.get_width() * 0.5, gv + 0.015, f"{gv:.2f}\nw={wv:.2f}", ha="center", va="bottom", fontsize=6.6, color="#70234D")

    ax_d.set_title("(D) BUD-Fusion Local Gate_L0 & Collab Weight w_cav_L0 Selectivity", fontweight="bold", pad=8)
    ax_d.set_ylabel("Mean Local Gate_L0 (0~1)")
    ax_d.set_ylim(0, 1.24)
    ax_d.set_xticks(x_sp)
    ax_d.set_xticklabels([x[1] for x in sp_keys])
    ax_d.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax_d.legend(loc="lower left", fontsize=8.5, frameon=True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=4, frameon=True)
    fig.suptitle("Local Attacks & Blind-Spot Preservation: BUD-Fusion vs Whole-CAV-Drop (Dev-320)", fontsize=13.5, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.905])
    p2 = os.path.join(fig_dir, "fig_step4_local_attack_and_blind_recall.png")
    fig.savefig(p2, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    out_paths.append(p2)

    # --- 图 3：按「本车盲区诚实协作车补全真车数 (0 / 1 / >=2)」分组的 Tier 2 vs Tier 4 误报率 ---
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.0), sharey=True)
    g_titles = {"m1": "m1 Coll. (64L LiDAR)", "m2": "m2 Coll. (Camera-LSS)", "overall": "Overall (m1 + m2 Pooled)"}
    reg_rows = [
        ("equal_5pct_overall", "Equal 5.0% Overall FPR Regime"),
        ("frozen_div_p95", "Frozen Fit div_p95 Threshold Regime (Score > 1.0)"),
    ]
    bw3 = 0.34
    xg = np.arange(len(BLIND_GROUPS_3))

    for r_i, (reg_k, reg_title) in enumerate(reg_rows):
        for c_i, scope in enumerate(("m1", "m2", "overall")):
            ax = axes[r_i, c_i]
            d_t2 = fpr_res[reg_k]["tier2_sensor_only"][scope]
            d_t4 = fpr_res[reg_k]["tier4_sensor_dist_vis"][scope]

            x_labels = [
                f"0 Blinds\n(n={d_t4['0']['n']})",
                f"1 Blind\n(n={d_t4['1']['n']})",
                f">=2 Blinds\n(n={d_t4['ge2']['n']})",
            ]
            for ti, (t_label, t_dict, t_col) in enumerate((
                ("Tier 2: Sensor Only (div_p95)", d_t2, "#4B7BEC"),
                ("Tier 4: Sensor+Dist+Vis (div_p95)", d_t4, "#1B8A5A"),
            )):
                off = (ti - 0.5) * bw3
                vals = np.array([t_dict[g]["val"] * 100.0 for g in BLIND_GROUPS_3])
                los = np.array([t_dict[g]["ci95"][0] * 100.0 for g in BLIND_GROUPS_3])
                his = np.array([t_dict[g]["ci95"][1] * 100.0 for g in BLIND_GROUPS_3])
                yerr = np.vstack([np.maximum(vals - los, 0.0), np.maximum(his - vals, 0.0)])
                bars = ax.bar(
                    xg + off, vals, width=bw3 * 0.92, yerr=yerr, capsize=3,
                    color=t_col, edgecolor="black", linewidth=0.6, alpha=0.90, label=t_label,
                )
                for bx, by, hi in zip(bars, vals, his):
                    ax.text(bx.get_x() + bx.get_width() * 0.5, hi + 0.6, f"{by:.1f}%", ha="center", va="bottom", fontsize=8.2)

            ax.axhline(5.0, color="#D64541", linestyle="--", linewidth=1.2, label="5.0% Target FPR")
            all_t2 = d_t2["all"]["val"] * 100.0
            all_t4 = d_t4["all"]["val"] * 100.0
            ax.set_title(
                f"{g_titles[scope]} ({'Eq-5%' if r_i == 0 else 'Frozen p95'})\nAll-Frame FPR: T2={all_t2:.1f}% | T4={all_t4:.1f}%",
                fontsize=9.8, fontweight="bold", pad=6,
            )
            if c_i == 0:
                ax.set_ylabel("Benign False Positive Rate (%)")
            ax.set_xticks(xg)
            ax.set_xticklabels(x_labels)
            ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    max_y = max(
        fpr_res[rk][tk][sc][gk]["ci95"][1] * 100.0
        for rk, _ in reg_rows
        for tk in ("tier2_sensor_only", "tier4_sensor_dist_vis")
        for sc in ("m1", "m2", "overall")
        for gk in BLIND_GROUPS_3
    )
    axes[0, 0].set_ylim(0, max(22.0, math.ceil(max_y + 4.0)))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=3, frameon=True)
    fig.suptitle("Clean-Frame Benign FPR Grouped by Honest Blind-Spot Completions (Dev-320)", fontsize=13.5, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.905])
    p3 = os.path.join(fig_dir, "fig_step4_blind_completion_fpr_t2_vs_t4.png")
    fig.savefig(p3, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    out_paths.append(p3)
    return out_paths


def main():
    parser = argparse.ArgumentParser(description="步骤 ④-3：BUD-Fusion HEAL m1+m2 Dev-320 全维度评测")
    parser.add_argument("--data_dir", type=str, default=TRAIN_FULL19_DIR)
    parser.add_argument("--split_json", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--base_calib_json", type=str, default=DEFAULT_BASE_CALIB)
    parser.add_argument("--gating_calib_json", type=str, default=DEFAULT_GATING_CALIB)
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--eval_split_out", type=str, default=DEFAULT_EVAL_SPLIT_OUT)
    parser.add_argument("--out_json", type=str, default=DEFAULT_OUT_JSON)
    parser.add_argument("--fig_dir", type=str, default=DEFAULT_FIG_DIR)
    args = parser.parse_args()

    t0 = time.time()
    with open(args.split_json, "r", encoding="utf-8") as f:
        split_plan = json.load(f)
    with open(args.gating_calib_json, "r", encoding="utf-8") as f:
        gating_calib = json.load(f)

    assert gating_calib.get("meta", {}).get("frame_norm_mode") == "div_p95", "严禁使用非 div_p95 门控标定文件！"

    wrapper = HEALBaseWrapper(device="cuda")
    ds_m1 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m1")
    ds_m2 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m2")
    gx, gy = build_bev_grid_xy_256(device="cuda", dtype=torch.float32)

    eval320 = _build_eval320_split(wrapper, ds_m2, args.data_dir, split_plan, args.eval_split_out)
    auditor = BUDFusionHEALAuditor(args.base_calib_json, gating_calib_path=args.gating_calib_json)
    k_tol = int(gating_calib.get("gating_params", {}).get("k_tol", 5))
    os.makedirs(args.cache_dir, exist_ok=True)

    print(f"[Step 3 · Dev-320 评测] 序列数={eval320['num_sequences']}, 总帧数={eval320['total_frames']}, k_tol={k_tol}")
    seq_caches = []
    for s_i, seq_name in enumerate(eval320["seq_order"]):
        seq_indices = eval320["sequences"][seq_name]["indices"]
        cache_path = os.path.join(args.cache_dir, f"{seq_name}.json")
        sc = _run_single_sequence(seq_name, seq_indices, cache_path, wrapper, ds_m1, ds_m2, gx, gy, auditor, k_tol=k_tol)
        seq_caches.append(sc)

        if s_i == 0:
            # 首条序列诊断输出（核查 Re-plan if #3）
            diag_lines = [f"\n=== 首条序列诊断 ({seq_name}, {sc['num_frames']} 帧, GT均值点数={sc['mean_pts_per_gt']}) ==="]
            for mod in MODALITIES_2:
                def _seq_ap50(st_k, md_k):
                    b5 = sc["ap_buckets"][mod][st_k][md_k]["0.5"]
                    return _fast_voc_ap(
                        np.asarray(b5["tp"], dtype=np.float64),
                        np.asarray(b5["fp"], dtype=np.float64),
                        np.asarray(b5["score"], dtype=np.float64),
                        int(b5["gt"]),
                    )

                ap_cln = _seq_ap50("clean", "no_defense")
                ap_bud = _seq_ap50("clean", "bud_fusion")
                ap_lr_nd = _seq_ap50("local_removal_1.0sigma", "no_defense")
                ap_lg_nd = _seq_ap50("local_ghost_2.0sigma", "no_defense")
                recs_m = [r for r in sc["frame_records"] if r["mod"] == mod]
                b_gt_tot = sum(r["n_blind_gt"] for r in recs_m if r["state"] == "clean")
                b_comp_tot = sum(r["n_blind_comp"] or 0 for r in recs_m if r["state"] == "clean")
                lrem_a = [r["gate_atk_region"] for r in recs_m if r["state"] == "local_removal_1.0sigma" and r["gate_atk_region"] is not None]
                lrem_c = [r["gate_cln_region"] for r in recs_m if r["state"] == "local_removal_1.0sigma" and r["gate_cln_region"] is not None]
                la_m = float(np.mean(lrem_a)) if lrem_a else float("nan")
                lc_m = float(np.mean(lrem_c)) if lrem_c else float("nan")
                diag_lines.append(
                    f"  {mod}: Clean AP@0.5(ND={ap_cln:.4f}, BUD={ap_bud:.4f}) | LocalRem ND={ap_lr_nd:.4f}, LocalGhost ND={ap_lg_nd:.4f} | "
                    f"BlindGT={b_gt_tot}, Comp={b_comp_tot} | LRem gate(atk={la_m:.3f}, cln={lc_m:.3f})"
                )
                assert ap_cln >= 0.55, f"触发 Re-plan if (首条序列): {mod} 干净 AP@0.5={ap_cln:.4f} < 0.55"
                assert b_gt_tot > 0, f"触发 Re-plan if (首条序列): {mod} 盲区 GT 数为 0"
                assert (ap_lr_nd < ap_cln - 1e-4) or (ap_lg_nd < ap_cln - 1e-4), (
                    f"触发 Re-plan if (首条序列): {mod} 局部攻击未降低 No-Defense AP ({ap_lr_nd:.4f}/{ap_lg_nd:.4f} vs {ap_cln:.4f})"
                )
            max_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            diag_lines.append(f"  GPU Peak Mem: {max_mem_gb:.2f} GB | 首序列耗时: {sc['elapsed_sec']:.1f}s\n")
            print("\n".join(diag_lines), flush=True)

    print(f"\n[汇总统计与 2,000 次 Block-Bootstrap] 开始聚合 8 条序列...", flush=True)
    report = _summarize_with_bootstrap(seq_caches, eval320["seq_order"], gating_calib)
    report["meta"]["total_elapsed_sec"] = round(time.time() - t0, 2)
    _atomic_write_json(args.out_json, report)

    fig_paths = _plot_all_figures(report, args.fig_dir)

    # 输出关键验收摘要（核对 Re-plan if 条件）
    print("\n=== Dev-320 关键验收核查摘要 ===")
    for m in MODALITIES_2:
        nd_c = report["ap_results"][m]["clean"]["no_defense"]["ap50"]["val"]
        bud_c = report["ap_results"][m]["clean"]["bud_fusion"]["ap50"]["val"]
        lr_bud = report["ap_results"][m]["local_removal_1.0sigma"]["bud_fusion"]["ap50"]["val"]
        lr_wcd = report["ap_results"][m]["local_removal_1.0sigma"]["whole_cav_drop"]["ap50"]["val"]
        lg_bud = report["ap_results"][m]["local_ghost_2.0sigma"]["bud_fusion"]["ap50"]["val"]
        lg_wcd = report["ap_results"][m]["local_ghost_2.0sigma"]["whole_cav_drop"]["ap50"]["val"]
        br_lr_bud = report["blind_spot_recall"][m]["local_removal_1.0sigma"]["bud_fusion"]["val"]
        br_lr_wcd = report["blind_spot_recall"][m]["local_removal_1.0sigma"]["whole_cav_drop"]["val"]
        print(
            f"  {m}: Clean loss={(nd_c - bud_c)*100:+.2f}% | "
            f"LRem AP(BUD={lr_bud:.4f} vs WCD={lr_wcd:.4f}), BlindRec(BUD={br_lr_bud:.3f} vs WCD={br_lr_wcd:.3f}) | "
            f"LGho AP(BUD={lg_bud:.4f} vs WCD={lg_wcd:.4f})"
        )
    print(f"[Step 3 完成] 报告写出：{args.out_json}")
    print("  生成图表：", fig_paths)
    print(f"  总耗时: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
