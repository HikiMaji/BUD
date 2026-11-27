# -*- coding: utf-8 -*-
"""
步骤 ③ 验证脚本：在 HEAL 官方 m1+m2 底座传输层实现三类攻击并验证无防御 AP 明显下降。

评测内容：
1. 第 0 帧单元测试：
   - 验证 weighted_fuse_safe 前向与官方 HeterPyramidCollab.forward 逐元素误差 < 1e-5；
   - 验证三类攻击零 NaN/Inf 梯度，且 m2 有效区以外扰动严格为 0.0。
2. opv2v_val 60 帧跨车数均衡抽样（5 条 2-CAV 序列各 8 帧 + 1 条 7-CAV 序列 20 帧），不标定任何防御门限：
   - 对比 Ego-Only (m1)、Clean Collab (m1+m2)、PGD Removal、Ghost Spoofing、Surrogate Transfer Removal；
   - 分别汇报全范围 [-102.4, 102.4]^2 与核心共视区 [-51.2, 51.2]^2，以及 2-CAV/7-CAV 车数分层指标。
"""

import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch

PROJ_ROOT = "/root/autodl-tmp/proj_dir2_hetero_trust"
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from src_v2_mm.heal_attacks import (
    generate_ghost_spoofing_heal,
    generate_pgd_removal_heal,
    generate_surrogate_transfer_removal_heal,
)
from src_v2_mm.heal_base_wrapper import HEALBaseWrapper

OUT_JSON = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step3_heal_m1m2_attacks_verify.json")


def init_stat():
    """初始化 OpenCOOD AP 评测统计桶。"""
    return {t: {"tp": [], "fp": [], "gt": 0, "score": []} for t in [0.3, 0.5, 0.7]}


def finalize_stat(stat: dict) -> dict:
    """从 TP/FP 累计桶计算 AP@0.3/0.5/0.7 与 IoU=0.5 匹配情况。"""
    from opencood.utils import eval_utils

    res = {}
    for t in [0.3, 0.5, 0.7]:
        if len(stat[t]["tp"]) > 0 and stat[t]["gt"] > 0:
            ap = float(eval_utils.calculate_ap(stat, t)[0])
        else:
            ap = 0.0
        res[f"ap{int(round(t * 100))}"] = round(ap, 6)

    res["gt_total"] = int(stat[0.5]["gt"])
    res["pred_total"] = int(len(stat[0.5]["tp"]))
    res["tp_at_05"] = int(sum(stat[0.5]["tp"]))
    res["fp_at_05"] = int(sum(stat[0.5]["fp"]))
    return res


def filter_boxes_by_xy_bound(pb, ps, gt, bound: float):
    """按中心点 |x|<=bound 且 |y|<=bound 裁剪预测框与真值框（兼容空张量/None）。"""
    if gt is not None and len(gt) > 0:
        gt_c = gt.mean(dim=1)
        gt_mask = (gt_c[:, 0].abs() <= bound) & (gt_c[:, 1].abs() <= bound)
        gt_sub = gt[gt_mask]
    else:
        gt_sub = gt

    if pb is not None and len(pb) > 0:
        pb_c = pb.mean(dim=1)
        pb_mask = (pb_c[:, 0].abs() <= bound) & (pb_c[:, 1].abs() <= bound)
        if pb_mask.any():
            pb_sub, ps_sub = pb[pb_mask], ps[pb_mask]
        else:
            pb_sub, ps_sub = None, None
    else:
        pb_sub, ps_sub = None, None
    return pb_sub, ps_sub, gt_sub


def accumulate_stats(bucket: dict, pb, ps, gt):
    """同时向全范围 [-102.4, 102.4]^2 与核心区 [-51.2, 51.2]^2 累加 TP/FP。"""
    from opencood.utils import eval_utils

    for t in [0.3, 0.5, 0.7]:
        eval_utils.caluclate_tp_fp(pb, ps, gt, bucket["full_102.4"], t)

    pb_51, ps_51, gt_51 = filter_boxes_by_xy_bound(pb, ps, gt, 51.2)
    for t in [0.3, 0.5, 0.7]:
        eval_utils.caluclate_tp_fp(pb_51, ps_51, gt_51, bucket["core_51.2"], t)


def make_condition_buckets():
    """构建各车数层及总体的评测统计桶。"""
    return {
        "overall": {"full_102.4": init_stat(), "core_51.2": init_stat()},
        "2cav": {"full_102.4": init_stat(), "core_51.2": init_stat()},
        "7cav": {"full_102.4": init_stat(), "core_51.2": init_stat()},
    }


def select_balanced_val_indices(wrapper: HEALBaseWrapper, dataset, val_dir: str):
    """
    跨车数均衡抽样 60 帧：
    每条 2-CAV 序列均匀取 8 帧（5*8=40 帧），7-CAV 序列 2021_09_09_19_27_35 均匀取 20 帧。
    """
    sample_seqs = wrapper.get_sample_sequences(dataset)
    seq_to_indices = OrderedDict()
    for idx, seq in enumerate(sample_seqs):
        seq_to_indices.setdefault(seq, []).append(idx)

    selected = []
    seq_summary = OrderedDict()
    for seq, idxs in seq_to_indices.items():
        seq_path = os.path.join(val_dir, seq)
        cav_dirs = [d for d in os.listdir(seq_path) if os.path.isdir(os.path.join(seq_path, d))]
        n_cav_raw = len(cav_dirs)
        pick_n = 20 if n_cav_raw >= 4 else 8
        pick_pos = np.linspace(0, len(idxs) - 1, pick_n, dtype=int)
        picked = [int(idxs[p]) for p in pick_pos]
        cav_group = "7cav" if n_cav_raw >= 4 else "2cav"
        for s_idx in picked:
            selected.append((s_idx, seq, cav_group, n_cav_raw))
        seq_summary[seq] = {
            "raw_cavs": n_cav_raw,
            "cav_group": cav_group,
            "total_frames": len(idxs),
            "sampled_frames": len(picked),
            "sampled_indices": picked,
        }
    return selected, seq_summary


def run_unit_test_frame0(wrapper: HEALBaseWrapper, dataset, device: torch.device) -> dict:
    """在第 0 帧执行前向等价性断言、梯度有限性与有效掩膜边界单元测试。"""
    from opencood.tools import train_utils

    batch = dataset.collate_batch_test([dataset[0]])
    batch_gpu = train_utils.to_device(batch, device)
    d_ego = batch_gpu["ego"]

    with torch.no_grad():
        official_out = wrapper.model(d_ego)
        proto = wrapper.extract_protocol_features(d_ego)
        custom_out = wrapper.forward_from_protocol(
            proto["heter_feature_2d"],
            proto["record_len"],
            proto["affine_matrix"],
            proto["agent_modality_list"],
        )

        err_cls = float((official_out["cls_preds"] - custom_out["cls_preds"]).abs().max().item())
        err_reg = float((official_out["reg_preds"] - custom_out["reg_preds"]).abs().max().item())
        err_dir = float((official_out["dir_preds"] - custom_out["dir_preds"]).abs().max().item())

    assert err_cls < 1e-5, f"cls_preds 前向差异超限: {err_cls}"
    assert err_reg < 1e-5, f"reg_preds 前向差异超限: {err_reg}"
    assert err_dir < 1e-5, f"dir_preds 前向差异超限: {err_dir}"

    feat = proto["heter_feature_2d"]
    valid_mask = proto["cam_valid_mask"]
    invalid_m2 = valid_mask[1:2] < 0.5

    # 逐一检查 3 类攻击与掩膜外零扰动约束
    atk_checks = {}
    f_pgd, st_pgd = generate_pgd_removal_heal(
        wrapper, feat, proto["record_len"], proto["affine_matrix"], proto["agent_modality_list"], valid_mask, attacker_idx=1
    )
    out_pgd = wrapper.forward_from_protocol(f_pgd, proto["record_len"], proto["affine_matrix"], proto["agent_modality_list"])
    assert torch.isfinite(out_pgd["cls_preds"]).all()
    assert float((f_pgd[1:2] - feat[1:2]).abs()[:, :, invalid_m2[0, 0]].max().item()) == 0.0
    atk_checks["pgd_removal"] = st_pgd

    f_ghost, st_ghost = generate_ghost_spoofing_heal(
        wrapper, feat, proto["record_len"], proto["affine_matrix"], proto["agent_modality_list"], valid_mask, attacker_idx=1, frame_seed=0
    )
    out_ghost = wrapper.forward_from_protocol(f_ghost, proto["record_len"], proto["affine_matrix"], proto["agent_modality_list"])
    assert torch.isfinite(out_ghost["cls_preds"]).all()
    assert float((f_ghost[1:2] - feat[1:2]).abs()[:, :, invalid_m2[0, 0]].max().item()) == 0.0
    atk_checks["ghost_spoofing"] = st_ghost

    f_sur, st_sur = generate_surrogate_transfer_removal_heal(wrapper, feat, valid_mask, attacker_idx=1)
    out_sur = wrapper.forward_from_protocol(f_sur, proto["record_len"], proto["affine_matrix"], proto["agent_modality_list"])
    assert torch.isfinite(out_sur["cls_preds"]).all()
    assert float((f_sur[1:2] - feat[1:2]).abs()[:, :, invalid_m2[0, 0]].max().item()) == 0.0
    atk_checks["surrogate_transfer_removal"] = st_sur

    return {
        "official_vs_wrapper_max_abs_diff": {
            "cls_preds": err_cls,
            "reg_preds": err_reg,
            "dir_preds": err_dir,
        },
        "all_zero_outside_valid_mask": True,
        "all_outputs_finite": True,
        "frame0_attack_stats": atk_checks,
    }


def main():
    t_start = time.time()
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wrapper = HEALBaseWrapper(device=str(device))
    dataset = wrapper.build_dataset(use_cav=2, visualize=True)
    print(f"[INFO] HEALBaseWrapper 就绪: env={wrapper.env_info}, pad_layers={len(wrapper.checked_pad_layers)}, frames={len(dataset)}", flush=True)

    # 1. 第 0 帧严格单元测试
    unit_res = run_unit_test_frame0(wrapper, dataset, device)
    print("[UNIT TEST PASS]", json.dumps(unit_res, ensure_ascii=False), flush=True)

    # 2. 60 帧跨车数均衡抽样评测（不标定门限）
    val_dir = wrapper.hypes["validate_dir"]
    sampled_list, seq_summary = select_balanced_val_indices(wrapper, dataset, val_dir)
    print(f"[INFO] 抽样完成: 共 {len(sampled_list)} 帧 (2-CAV 40帧 + 7-CAV 20帧)", flush=True)

    conditions = [
        "ego_only_m1",
        "clean_collab",
        "pgd_removal",
        "ghost_spoofing",
        "surrogate_transfer_removal",
    ]
    buckets = {c: make_condition_buckets() for c in conditions}
    atk_diag_acc = {
        "pgd_removal": {"linf": [], "l2": [], "elapsed": []},
        "ghost_spoofing": {"linf": [], "l2": [], "anchors": [], "elapsed": []},
        "surrogate_transfer_removal": {"linf": [], "l2": [], "elapsed": []},
    }

    from opencood.tools import train_utils

    for step_i, (sample_idx, seq, cav_group, _) in enumerate(sampled_list, 1):
        batch = dataset.collate_batch_test([dataset[sample_idx]])
        batch_gpu = train_utils.to_device(batch, device)
        d_ego = batch_gpu["ego"]

        proto = wrapper.extract_protocol_features(d_ego)
        feat = proto["heter_feature_2d"]
        rlen = proto["record_len"]
        amat = proto["affine_matrix"]
        mods = proto["agent_modality_list"]
        vmask = proto["cam_valid_mask"]
        assert mods == ["m1", "m2"], f"sample {sample_idx} 预期模态为 ['m1', 'm2']，实际={mods}"

        # (1) Ego-Only (m1) 与 Clean Collab (m1+m2)
        with torch.no_grad():
            out_ego = wrapper.forward_single_agent(feat[0:1])
            pb, ps, gt = wrapper.decode_boxes(batch_gpu, out_ego, dataset)
            accumulate_stats(buckets["ego_only_m1"]["overall"], pb, ps, gt)
            accumulate_stats(buckets["ego_only_m1"][cav_group], pb, ps, gt)

            out_clean = wrapper.forward_from_protocol(feat, rlen, amat, mods)
            pb, ps, gt = wrapper.decode_boxes(batch_gpu, out_clean, dataset)
            accumulate_stats(buckets["clean_collab"]["overall"], pb, ps, gt)
            accumulate_stats(buckets["clean_collab"][cav_group], pb, ps, gt)

        # (2) 攻击 1：白盒 PGD 目标擦除
        t0 = time.time()
        f_pgd, st_pgd = generate_pgd_removal_heal(wrapper, feat, rlen, amat, mods, vmask, attacker_idx=1)
        atk_diag_acc["pgd_removal"]["elapsed"].append(time.time() - t0)
        atk_diag_acc["pgd_removal"]["linf"].append(st_pgd["linf_max"])
        atk_diag_acc["pgd_removal"]["l2"].append(st_pgd["l2_mean_valid"])
        with torch.no_grad():
            out_pgd = wrapper.forward_from_protocol(f_pgd, rlen, amat, mods)
            pb, ps, gt = wrapper.decode_boxes(batch_gpu, out_pgd, dataset)
            accumulate_stats(buckets["pgd_removal"]["overall"], pb, ps, gt)
            accumulate_stats(buckets["pgd_removal"][cav_group], pb, ps, gt)

        # (3) 攻击 2：随机道路空白区鬼影伪造
        t0 = time.time()
        f_ghost, st_ghost = generate_ghost_spoofing_heal(
            wrapper, feat, rlen, amat, mods, vmask, attacker_idx=1, frame_seed=sample_idx
        )
        atk_diag_acc["ghost_spoofing"]["elapsed"].append(time.time() - t0)
        atk_diag_acc["ghost_spoofing"]["linf"].append(st_ghost["linf_max"])
        atk_diag_acc["ghost_spoofing"]["l2"].append(st_ghost["l2_mean_valid"])
        atk_diag_acc["ghost_spoofing"]["anchors"].append(st_ghost["num_spoof_anchors"])
        with torch.no_grad():
            out_ghost = wrapper.forward_from_protocol(f_ghost, rlen, amat, mods)
            pb, ps, gt = wrapper.decode_boxes(batch_gpu, out_ghost, dataset)
            accumulate_stats(buckets["ghost_spoofing"]["overall"], pb, ps, gt)
            accumulate_stats(buckets["ghost_spoofing"][cav_group], pb, ps, gt)

        # (4) 攻击 3：无本车信息黑盒/代理迁移擦除
        t0 = time.time()
        f_sur, st_sur = generate_surrogate_transfer_removal_heal(wrapper, feat, vmask, attacker_idx=1)
        atk_diag_acc["surrogate_transfer_removal"]["elapsed"].append(time.time() - t0)
        atk_diag_acc["surrogate_transfer_removal"]["linf"].append(st_sur["linf_max"])
        atk_diag_acc["surrogate_transfer_removal"]["l2"].append(st_sur["l2_mean_valid"])
        with torch.no_grad():
            out_sur = wrapper.forward_from_protocol(f_sur, rlen, amat, mods)
            pb, ps, gt = wrapper.decode_boxes(batch_gpu, out_sur, dataset)
            accumulate_stats(buckets["surrogate_transfer_removal"]["overall"], pb, ps, gt)
            accumulate_stats(buckets["surrogate_transfer_removal"][cav_group], pb, ps, gt)

        if step_i % 15 == 0 or step_i == len(sampled_list):
            print(f"  [进度 {step_i}/{len(sampled_list)}] 已耗时 {time.time() - t_start:.1f}s (当前序列 {seq})", flush=True)

    summary_results = OrderedDict()
    for c in conditions:
        summary_results[c] = {
            "overall": {
                "full_102.4": finalize_stat(buckets[c]["overall"]["full_102.4"]),
                "core_51.2": finalize_stat(buckets[c]["overall"]["core_51.2"]),
            },
            "sub_2cav": {
                "full_102.4": finalize_stat(buckets[c]["2cav"]["full_102.4"]),
                "core_51.2": finalize_stat(buckets[c]["2cav"]["core_51.2"]),
            },
            "sub_7cav": {
                "full_102.4": finalize_stat(buckets[c]["7cav"]["full_102.4"]),
                "core_51.2": finalize_stat(buckets[c]["7cav"]["core_51.2"]),
            },
        }

    attack_summary = {}
    for k, v in atk_diag_acc.items():
        attack_summary[k] = {
            "linf_max_mean": round(float(np.mean(v["linf"])), 4),
            "l2_mean_over_all_grids": round(float(np.mean(v["l2"])), 4),
            "avg_gen_sec_per_frame": round(float(np.mean(v["elapsed"])), 4),
        }
        if "anchors" in v:
            attack_summary[k]["avg_spoof_anchors"] = round(float(np.mean(v["anchors"])), 2)

    report = {
        "meta": {
            "task": "Step3 Verify HEAL m1+m2 Transport-Layer Attacks (No Defense)",
            "env_info": wrapper.env_info,
            "num_checked_sym_pad_layers": len(wrapper.checked_pad_layers),
            "image_format_on_disk": "lossless PNG (/root/autodl-tmp/datasets/OPV2V/validate)",
            "num_sampled_frames": len(sampled_list),
            "elapsed_total_sec": round(time.time() - t_start, 2),
        },
        "unit_test_frame0": unit_res,
        "sampling_plan": seq_summary,
        "attack_generation_stats": attack_summary,
        "results": summary_results,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] 结果已写入 {OUT_JSON}，总耗时 {report['meta']['elapsed_total_sec']}s", flush=True)
    for c, r in summary_results.items():
        print(
            f"  {c:28s} | core51.2 AP50={r['overall']['core_51.2']['ap50']*100:5.2f}% "
            f"AP70={r['overall']['core_51.2']['ap70']*100:5.2f}% TP/FP={r['overall']['core_51.2']['tp_at_05']}/{r['overall']['core_51.2']['fp_at_05']} | "
            f"full102.4 AP50={r['overall']['full_102.4']['ap50']*100:5.2f}%",
            flush=True,
        )


if __name__ == "__main__":
    main()
