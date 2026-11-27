# -*- coding: utf-8 -*-
"""
M3：在官方无损原版 PNG 测试集 (/root/autodl-tmp/datasets/OPV2V/test, 16 序列 2,170 帧) 上
复现 HEAL 官方干净精度，并与官方真实报告/发布的基准数字逐项核对（硬停机门槛：绝对偏差 <= 1.0 个百分点）。

严格遵循规范：
1. 仅评测 HEAL 原生冻结模型干净推理精度（零防御模块、零攻击、零超参调整，绝不触碰 C 阶段防御终测锁）；
2. 直接使用 HEAL 官方内置模态分配文件：
   - opencood/modality_assign/opv2v_4modality.json
   - opencood/modality_assign/opv2v_4modality_in_order.json
3. 完整复现并核对官方权重包真实存在的全部 m1 / m2 基准 YAML：
   - [Official-1] m2 单车 (stage2/m2_alignto_m1/eval_no_102.4_102.4_epoch25.yaml, comm_range=70, max_cav=5 全车队GT):
       AP@0.3=45.07%, AP@0.5=35.07%, AP@0.7=19.29%
   - [Official-2] m1 单车 (final_infer/eval_intermediate0919_204.8_102.4_epoch1_use_cav1.yaml, comm_range=180, max_cav=5 全车队GT):
       AP@0.3=78.54%, AP@0.5=76.45%, AP@0.7=66.29%
   - [Official-3] m1+m2 异构协同 (final_infer/eval_intermediate0919_204.8_102.4_epoch1_use_cav2.yaml, comm_range=180, max_cav=5 全车队GT):
       AP@0.3=84.94%, AP@0.5=82.63%, AP@0.7=72.69%
4. 同时运行官方非截断全协作车配置（all_cavs, max_cav=5 全部可用协作车参与融合，不用两车截断）：
   - Stage-1 m1 Base 全车队同构融合 (comm_range=70, range=102.4x102.4, max_cav=5 全部协作车)
   - Final-Infer m1+m2 全车队异构融合 (comm_range=70 & 180, range=204.8x102.4, use_cav=5 全部协作车，使用官方 opv2v_4modality.json)
   （注：m3 SECOND 模型因官方 README 声明 HuggingFace 权重通道数缺陷不纳入对比；无对应官方单列数字的全车队 m1+m2 仅客观记录实测值，不与不存在的数字硬比。）
5. 硬停机检查：所有存在官方真实数字的对比项（AP@0.5 与 AP@0.7），若任一项 |repro - official| > 1.0 个百分点，立即返回非零退出码触发硬停机！
"""

import argparse
import copy
import json
import os
import sys
import time
from collections import OrderedDict

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "2"

import torch
from torch.utils.data import DataLoader

torch.set_num_threads(2)

PROJ_ROOT = "/root/autodl-tmp/proj_dir2_hetero_trust"
if os.path.exists(PROJ_ROOT) and PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)
LOCAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if LOCAL_ROOT not in sys.path:
    sys.path.insert(0, LOCAL_ROOT)

from src_v2_mm.heal_base_wrapper import HEAL_ROOT, init_heal_runtime
from src_v2_mm.heal_env_check import assert_effnet_symmetric_padding
from src_v2_mm.step4_1_v2_core import OPV2V_TEST_DIR, V3_RESULTS_DIR

CKPT_ROOT = os.path.join(PROJ_ROOT, "checkpoints_v2_mm", "heal_official")
STAGE1_M1_DIR = os.path.join(CKPT_ROOT, "stage1", "Pyramid_m1_base_2023_08_14_04_28_12")
STAGE2_M2_DIR = os.path.join(CKPT_ROOT, "stage2", "m2_alignto_m1")
FINAL_INFER_DIR = os.path.join(CKPT_ROOT, "final_infer")

OFFICIAL_ASSIGN_4MOD = os.path.join(HEAL_ROOT, "opencood", "modality_assign", "opv2v_4modality.json")
OFFICIAL_ASSIGN_IN_ORDER = os.path.join(HEAL_ROOT, "opencood", "modality_assign", "opv2v_4modality_in_order.json")
DEFAULT_OUT_JSON = os.path.join(V3_RESULTS_DIR, "heal_reproduction_check.json")

# 官方权重仓库真实存在的评测结果 YAML 精确数字（单位：小数，1 个百分点 = 0.01）
OFFICIAL_GROUND_TRUTH = OrderedDict({
    "m2_single_stage2_no_102.4": {
        "source_yaml": "checkpoints_v2_mm/heal_official/stage2/m2_alignto_m1/eval_no_102.4_102.4_epoch25.yaml",
        "ap30": 0.45069334679888295,
        "ap50": 0.35069454726985844,
        "ap70": 0.19291394517978025,
    },
    "m1_single_official_in_order": {
        "source_yaml": "checkpoints_v2_mm/heal_official/final_infer/eval_intermediate0919_204.8_102.4_epoch1_use_cav1.yaml",
        "ap30": 0.7854326635647838,
        "ap50": 0.7644759959146037,
        "ap70": 0.6629306458747647,
    },
    "m1_m2_collab_official_in_order": {
        "source_yaml": "checkpoints_v2_mm/heal_official/final_infer/eval_intermediate0919_204.8_102.4_epoch1_use_cav2.yaml",
        "ap30": 0.8494047896518128,
        "ap50": 0.8263351820028744,
        "ap70": 0.7269480899704631,
    },
})


def _disable_depth_supervision(model):
    for m_attr in ("depth_supervision_m2", "depth_supervision_m4"):
        if hasattr(model, m_attr):
            setattr(model, m_attr, False)
    for enc_attr in ("encoder_m2", "encoder_m4"):
        if hasattr(model, enc_attr):
            enc = getattr(model, enc_attr)
            enc.depth_supervision = False
            if hasattr(enc, "camencode"):
                enc.camencode.depth_supervision = False


def _init_stat():
    return {t: {"tp": [], "fp": [], "gt": 0, "score": []} for t in (0.3, 0.5, 0.7)}


def _finalize_stat(stat, eval_utils) -> OrderedDict:
    out = OrderedDict()
    for t, k in ((0.3, "ap30"), (0.5, "ap50"), (0.7, "ap70")):
        ap = float(eval_utils.calculate_ap(stat, t)[0]) if (len(stat[t]["tp"]) > 0 and stat[t]["gt"] > 0) else 0.0
        out[k] = round(ap, 6)
        out[f"{k}_pct"] = round(ap * 100.0, 2)
    out["gt_total"] = int(stat[0.5]["gt"])
    out["pred_total"] = int(len(stat[0.5]["tp"]))
    return out


def _eval_stage2_m2_no_fusion(test_dir: str, device: torch.device) -> OrderedDict:
    """复现官方 stage2/m2_alignto_m1/eval_no_102.4_102.4_epoch25.yaml（comm_range=70, max_cav=5 全车队联合GT）。"""
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import inference_utils, train_utils
    from opencood.utils import eval_utils
    from opencood.utils.common_utils import update_dict

    t0 = time.time()
    hypes = yaml_utils.load_yaml(os.path.join(STAGE2_M2_DIR, "config.yaml"), opt=None)
    r_102 = [-102.4, -102.4, -3, 102.4, 102.4, 1]
    hypes = update_dict(hypes, {"cav_lidar_range": r_102, "lidar_range": r_102, "gt_range": r_102})
    hypes = yaml_utils.load_general_params(hypes)
    hypes["validate_dir"] = test_dir
    hypes["test_dir"] = test_dir
    hypes["input_source"] = ["camera"]
    # 严格对齐官方 stage2/m2_alignto_m1/config.yaml 原生配置：使用 additional/test 下的 bev_visibility.png 过滤相机可见真值框
    hypes["add_data_extension"] = ["bev_visibility.png"]
    hypes["label_type"] = "camera"
    hypes["heter"]["assignment_path"] = OFFICIAL_ASSIGN_4MOD

    model = train_utils.create_model(hypes)
    ck = torch.load(os.path.join(STAGE2_M2_DIR, "net_epoch25.pth"), map_location="cpu")
    matched = {k: v for k, v in ck.items() if k in model.state_dict()}
    model.load_state_dict(matched, strict=True)
    assert_effnet_symmetric_padding(model, min_layers=5)
    _disable_depth_supervision(model)
    model.to(device).eval()

    ds = build_dataset(hypes, visualize=True, train=False)
    ds.use_hdf5 = False
    dl = DataLoader(ds, batch_size=1, num_workers=4, collate_fn=ds.collate_batch_test, shuffle=False, drop_last=False)

    stat = _init_stat()
    for i, batch_data in enumerate(dl):
        if batch_data is None:
            continue
        with torch.no_grad():
            bg = train_utils.to_device(batch_data, device)
            res = inference_utils.inference_no_fusion(bg, model, ds, single_gt=False)
        for t in (0.3, 0.5, 0.7):
            eval_utils.caluclate_tp_fp(res["pred_box_tensor"], res["pred_score"], res["gt_box_tensor"], stat, t)
        if (i + 1) % 500 == 0:
            print(f"  [m2_single] {i + 1}/{len(ds)} ({time.time() - t0:.1f}s)", flush=True)

    metrics = _finalize_stat(stat, eval_utils)
    print(f"[M3] m2_single (no_102.4, all_cavs_gt): {metrics} ({time.time() - t0:.1f}s)", flush=True)
    return OrderedDict({
        "config": {
            "ckpt": "stage2/m2_alignto_m1/net_epoch25.pth",
            "fusion_method": "no (ego m2 detection vs all-CAVs fused GT)",
            "comm_range": float(hypes["comm_range"]),
            "max_cav": int(hypes["train_params"]["max_cav"]),
            "detection_range": r_102,
            "assignment_json": "opencood/modality_assign/opv2v_4modality.json",
            "num_frames": len(ds),
        },
        "metrics": metrics,
        "elapsed_sec": round(time.time() - t0, 2),
    })


def _eval_stage1_m1_all_cavs(test_dir: str, device: torch.device) -> OrderedDict:
    """运行官方 stage1/Pyramid_m1_base 全车队同构协同 (inference.py 默认：comm_range=70, max_cav=5 全部协作车参与)。"""
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import inference_utils, train_utils
    from opencood.utils import eval_utils
    from opencood.utils.common_utils import update_dict

    t0 = time.time()
    hypes = yaml_utils.load_yaml(os.path.join(STAGE1_M1_DIR, "config.yaml"), opt=None)
    r_102 = [-102.4, -102.4, -3, 102.4, 102.4, 1]
    hypes = update_dict(hypes, {"cav_lidar_range": r_102, "lidar_range": r_102, "gt_range": r_102})
    hypes = yaml_utils.load_general_params(hypes)
    hypes["validate_dir"] = test_dir
    hypes["test_dir"] = test_dir
    hypes["input_source"] = ["lidar"]
    hypes["add_data_extension"] = []
    hypes["label_type"] = "lidar"
    hypes["heter"]["assignment_path"] = OFFICIAL_ASSIGN_4MOD

    model = train_utils.create_model(hypes)
    ck = torch.load(os.path.join(STAGE1_M1_DIR, "net_epoch_bestval_at23.pth"), map_location="cpu")
    model.load_state_dict({k: v for k, v in ck.items() if k in model.state_dict()}, strict=True)
    model.to(device).eval()

    ds = build_dataset(hypes, visualize=True, train=False)
    ds.use_hdf5 = False
    dl = DataLoader(ds, batch_size=1, num_workers=4, collate_fn=ds.collate_batch_test, shuffle=False, drop_last=False)

    stat = _init_stat()
    cav_counts = {}
    for batch_data in dl:
        if batch_data is None:
            continue
        nc = len(batch_data["ego"]["agent_modality_list"])
        cav_counts[str(nc)] = cav_counts.get(str(nc), 0) + 1
        with torch.no_grad():
            bg = train_utils.to_device(batch_data, device)
            res = inference_utils.inference_intermediate_fusion(bg, model, ds)
        for t in (0.3, 0.5, 0.7):
            eval_utils.caluclate_tp_fp(res["pred_box_tensor"], res["pred_score"], res["gt_box_tensor"], stat, t)

    metrics = _finalize_stat(stat, eval_utils)
    print(f"[M3] stage1_m1_all_cavs (comm=70, max_cav=5): {metrics} ({time.time() - t0:.1f}s)", flush=True)
    return OrderedDict({
        "config": {
            "ckpt": "stage1/Pyramid_m1_base_2023_08_14_04_28_12/net_epoch_bestval_at23.pth",
            "fusion_method": "intermediate (all available CAVs, no 2-car truncation)",
            "comm_range": float(hypes["comm_range"]),
            "max_cav": int(hypes["train_params"]["max_cav"]),
            "detection_range": r_102,
            "participating_cav_count_hist": cav_counts,
            "num_frames": len(ds),
        },
        "metrics": metrics,
        "elapsed_sec": round(time.time() - t0, 2),
    })


def _eval_final_infer_official_and_all_cavs(test_dir: str, device: torch.device) -> OrderedDict:
    """
    加载官方 final_infer/net_epoch1.pth (m1+m2)，同时完成：
    1) 官方 inference_heter_in_order.py 精确同配置评测 (range=204.8x102.4, comm_range=180, max_cav=5 全车队GT):
       - m1_single_official_in_order (对应官方 eval_..._use_cav1.yaml)
       - m1_m2_collab_official_in_order (对应官方 eval_..._use_cav2.yaml)
    2) 无两车截断的全协作车评测 (use_cav=5 == max_cav，全部可用协作车参与融合):
       - m1_m2_all_cavs_comm70 (comm_range=70, max_cav=5, 官方 opv2v_4modality.json)
       - m1_m2_all_cavs_comm180 (comm_range=180, max_cav=5, 官方 opv2v_4modality.json)
    """
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import inference_utils, train_utils
    from opencood.utils import eval_utils
    from opencood.utils.common_utils import update_dict

    hypes_base = yaml_utils.load_yaml(os.path.join(FINAL_INFER_DIR, "config.yaml"), opt=None)
    for drop_m in ("m3", "m4"):
        hypes_base["heter"]["modality_setting"].pop(drop_m, None)
        hypes_base["model"]["args"].pop(drop_m, None)
    hypes_base["heter"]["lidar_channels_dict"] = {}

    r_204 = [-204.8, -102.4, -3, 204.8, 102.4, 1]
    hypes_base = update_dict(hypes_base, {"cav_lidar_range": r_204, "lidar_range": r_204, "gt_range": r_204})
    hypes_base = update_dict(
        hypes_base,
        {"mapping_dict": {"m1": "m1", "m2": "m2", "m3": "m2", "m4": "m2"}, "ego_modality": "m1"},
    )
    hypes_base = yaml_utils.load_general_params(hypes_base)
    hypes_base["validate_dir"] = test_dir
    hypes_base["test_dir"] = test_dir
    hypes_base["fusion"]["core_method"] = "intermediateheterinfer"
    hypes_base["input_source"] = ["lidar", "camera"]
    hypes_base["add_data_extension"] = []
    hypes_base["label_type"] = "lidar"

    model = train_utils.create_model(hypes_base)
    ck = torch.load(os.path.join(FINAL_INFER_DIR, "net_epoch1.pth"), map_location="cpu")
    model.load_state_dict({k: v for k, v in ck.items() if k in model.state_dict()}, strict=True)
    assert_effnet_symmetric_padding(model, min_layers=5)
    _disable_depth_supervision(model)
    model.to(device).eval()

    runs = [
        ("m1_single_official_in_order", 1, 180.0, OFFICIAL_ASSIGN_IN_ORDER, {"m1": "m1", "m2": "m2", "m3": "m2", "m4": "m2"}),
        ("m1_m2_collab_official_in_order", 2, 180.0, OFFICIAL_ASSIGN_IN_ORDER, {"m1": "m1", "m2": "m2", "m3": "m2", "m4": "m2"}),
        ("m1_m2_all_cavs_comm70", 5, 70.0, OFFICIAL_ASSIGN_4MOD, {"m1": "m1", "m2": "m2", "m3": "m1", "m4": "m2"}),
        ("m1_m2_all_cavs_comm180", 5, 180.0, OFFICIAL_ASSIGN_4MOD, {"m1": "m1", "m2": "m2", "m3": "m1", "m4": "m2"}),
    ]

    out = OrderedDict()
    for tag, use_cav, comm_r, assign_p, map_d in runs:
        t_r = time.time()
        h = copy.deepcopy(hypes_base)
        h["use_cav"] = int(use_cav)
        h["comm_range"] = float(comm_r)
        h["heter"]["assignment_path"] = assign_p
        h["heter"]["mapping_dict"] = dict(map_d)
        h = update_dict(h, {"mapping_dict": dict(map_d)})

        ds = build_dataset(h, visualize=True, train=False)
        ds.use_hdf5 = False
        dl = DataLoader(ds, batch_size=1, num_workers=4, collate_fn=ds.collate_batch_test, shuffle=False, drop_last=False)

        stat = _init_stat()
        mod_combos = {}
        for i, batch_data in enumerate(dl):
            if batch_data is None:
                continue
            mk = "+".join(batch_data["ego"]["agent_modality_list"])
            mod_combos[mk] = mod_combos.get(mk, 0) + 1
            with torch.no_grad():
                bg = train_utils.to_device(batch_data, device)
                res = inference_utils.inference_intermediate_fusion(bg, model, ds)
            for t in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(res["pred_box_tensor"], res["pred_score"], res["gt_box_tensor"], stat, t)
            if (i + 1) % 500 == 0:
                print(f"  [{tag}] {i + 1}/{len(ds)} ({time.time() - t_r:.1f}s)", flush=True)

        metrics = _finalize_stat(stat, eval_utils)
        print(f"[M3] {tag} (comm={comm_r}, use_cav={use_cav}): {metrics} ({time.time() - t_r:.1f}s)", flush=True)
        out[tag] = OrderedDict({
            "config": {
                "ckpt": "final_infer/net_epoch1.pth",
                "comm_range": float(comm_r),
                "max_cav_gt": int(h["train_params"]["max_cav"]),
                "use_cav_collab": int(use_cav),
                "detection_range": r_204,
                "assignment_json": os.path.relpath(assign_p, HEAL_ROOT),
                "modality_combinations": mod_combos,
                "num_frames": len(ds),
            },
            "metrics": metrics,
            "elapsed_sec": round(time.time() - t_r, 2),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="M3：PNG 版 OPV2V test 集复现 HEAL 官方干净精度与硬停机检查")
    parser.add_argument("--test_dir", type=str, default=OPV2V_TEST_DIR)
    parser.add_argument("--out_json", type=str, default=DEFAULT_OUT_JSON)
    parser.add_argument("--max_diff_pct_points", type=float, default=1.0, help="硬停机容差（百分点，默认 1.0）")
    parser.add_argument("--reuse_non_stage2_from", type=str, default="", help="可选：复用同次已跑完的 stage1/final_infer 结果，仅重跑补齐 bev_visibility 后的 stage2 m2")
    args = parser.parse_args()

    t_start = time.time()
    env_info = init_heal_runtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m2_res = _eval_stage2_m2_no_fusion(args.test_dir, device)
    if args.reuse_non_stage2_from and os.path.exists(args.reuse_non_stage2_from):
        with open(args.reuse_non_stage2_from, "r", encoding="utf-8") as f:
            prev_all = json.load(f)["all_evaluated_configurations"]
        s1_res = prev_all["stage1_m1_all_cavs_comm70"]
        fi_res = {
            k: prev_all[k]
            for k in (
                "m1_single_official_in_order",
                "m1_m2_collab_official_in_order",
                "m1_m2_all_cavs_comm70",
                "m1_m2_all_cavs_comm180",
            )
        }
    else:
        s1_res = _eval_stage1_m1_all_cavs(args.test_dir, device)
        fi_res = _eval_final_infer_official_and_all_cavs(args.test_dir, device)

    all_runs = OrderedDict({
        "m1_single_official_in_order": fi_res["m1_single_official_in_order"],
        "m2_single_stage2_no_102.4": m2_res,
        "m1_m2_collab_official_in_order": fi_res["m1_m2_collab_official_in_order"],
        "stage1_m1_all_cavs_comm70": s1_res,
        "m1_m2_all_cavs_comm70": fi_res["m1_m2_all_cavs_comm70"],
        "m1_m2_all_cavs_comm180": fi_res["m1_m2_all_cavs_comm180"],
    })

    comparisons = OrderedDict()
    max_abs_diff_pp = 0.0
    hard_stop_triggered = False
    failed_items = []

    for key, gt_ref in OFFICIAL_GROUND_TRUTH.items():
        rep = all_runs[key]["metrics"]
        item_cmp = OrderedDict({"source_yaml": gt_ref["source_yaml"]})
        for mk in ("ap30", "ap50", "ap70"):
            off_pp = round(float(gt_ref[mk]) * 100.0, 2)
            rep_pp = round(float(rep[mk]) * 100.0, 2)
            diff_pp = round(float(rep[mk] - gt_ref[mk]) * 100.0, 4)
            abs_pp = abs(diff_pp)
            if mk in ("ap50", "ap70"):
                max_abs_diff_pp = max(max_abs_diff_pp, abs_pp)
                if abs_pp > float(args.max_diff_pct_points):
                    hard_stop_triggered = True
                    failed_items.append(f"{key}.{mk}: repro={rep_pp:.2f}% vs official={off_pp:.2f}% (|diff|={abs_pp:.2f}pp > {args.max_diff_pct_points}pp)")
            item_cmp[mk] = {
                "official_pct": off_pp,
                "reproduced_pct": rep_pp,
                "diff_pct_points": round(diff_pp, 2),
                "abs_diff_pct_points": round(abs_pp, 2),
                "within_1pp": bool(abs_pp <= float(args.max_diff_pct_points)),
            }
        comparisons[key] = item_cmp

    report = OrderedDict({
        "meta": {
            "claim": "[临时结论 · 待人工代码复核]",
            "stage": "M3_HEAL_official_clean_AP_reproduction",
            "test_dir": args.test_dir,
            "image_format": "lossless_png",
            "env_info": env_info,
            "c_stage_final_defense_test_touched": False,
            "m3_second_note": "HEAL 官方 README 明确说明公开权重包中 m3 (SECOND) 存在输入通道数缺陷，故仅与真实存在且有效的 m1 / m2 官方基准比较。",
            "all_cavs_note": "同步报告无两车截断的全部可用协作车 (max_cav=5, comm_range=70/180) 实测结果；因原论文未单列 m3/m4 映射为 m1/m2 的全车队数字，按规范仅对真实存在的官方 YAML 数字做偏差校验。",
            "tolerance_pct_points": float(args.max_diff_pct_points),
            "max_abs_diff_pct_points_ap50_ap70": round(max_abs_diff_pp, 4),
            "hard_stop_triggered": hard_stop_triggered,
            "failed_checks": failed_items,
            "elapsed_sec": round(time.time() - t_start, 2),
        },
        "official_comparisons": comparisons,
        "all_evaluated_configurations": all_runs,
    })

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    tmp_p = args.out_json + ".tmp"
    with open(tmp_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.replace(tmp_p, args.out_json)

    print(f"\n[M3 完成] 报告已写入 {args.out_json} | max_abs_diff={max_abs_diff_pp:.4f}pp | hard_stop={hard_stop_triggered}", flush=True)
    if hard_stop_triggered:
        print(f"[硬停机红线触发] HEAL 官方复现偏差超过 {args.max_diff_pct_points} 个百分点：{failed_items}", file=sys.stderr, flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
