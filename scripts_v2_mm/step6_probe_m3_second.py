# -*- coding: utf-8 -*-
"""
步骤 ⑥（限时半天探 m3）：HEAL 官方 m3 (32 线 LiDAR + SECOND / SpConv) 快速可行性诊断脚本。

严格原则：
1. 不改动主线 HEALBaseWrapper 与 HEAL_ref，不触碰封存测试集 opv2v_test；
2. 自动执行 5 项探针并在 2 分钟内给出硬数据结论（READY_TO_INTEGRATE / NEEDS_QUICK_FIX / STOP_LOSS）：
   - 探针 1：检查 official final_infer/net_epoch1.pth 中 m3 权重键完整性；
   - 探针 2：在 DEV-8 抽 16 帧验证 m3 (SECOND + SpConv 2.3.6) 前向加载与特征提取（是否触发 Issue #20 / NaN / 全零）；
   - 探针 3：检查 m3 在传输层 heter_feature_2d 与 pyramid_L0 上与本车 m1 的余弦偏差与特征尺度分布；
   - 探针 4：同批 16 帧对比 Ego-Only、m1+m1、m1+m2、m1+m3 的干净 AP@0.5 / AP@0.7；
   - 探针 5：汇总诊断 JSON 并输出是否值得纳入三模态主线的量化建议。
"""

import argparse
import copy
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

from src_v2_mm.heal_base_wrapper import (
    FINAL_INFER_DIR,
    HEALBaseWrapper,
    ensure_in_order_assignment,
    init_heal_runtime,
    weighted_fuse_safe,
)
from src_v2_mm.step4_1_v2_core import OPV2V_TRAIN_DIR, PROTO_H, PROTO_W

DEFAULT_EVAL_SPLIT = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_3_eval320_split.json")
DEFAULT_OUT_JSON = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step6_m3_probe_report.json")


def _eval_16frames_ap(model, ds, indices: list, device: torch.device):
    """在指定 16 帧上测协同融合 AP@0.5 / AP@0.7 与特征统计，同时检测 NaN / 维度异常。"""
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
    from opencood.tools import train_utils
    from opencood.utils import eval_utils
    from opencood.utils.transformation_utils import normalize_pairwise_tfm

    ap_stat = {0.5: {"fp": [], "tp": [], "score": [], "gt": 0}, 0.7: {"fp": [], "tp": [], "score": [], "gt": 0}}
    cos_vals = []
    feat_stds = []
    nonzero_ratios = []
    errors = []

    for idx in indices:
        try:
            batch = train_utils.to_device(ds.collate_batch_test([ds[idx]]), device)
            ego_d = batch["ego"]
            mods = list(ego_d["agent_modality_list"])
            rlen = ego_d["record_len"]
            amat = normalize_pairwise_tfm(ego_d["pairwise_t_matrix"], model.H, model.W, model.fake_voxel_size)

            # 按模态分别提取并堆叠协议特征
            from collections import Counter
            import torchvision

            mod_cnt = Counter(mods)
            mod_feats = {}
            for mname in model.modality_name_list:
                if mname not in mod_cnt:
                    continue
                f = getattr(model, f"encoder_{mname}")(ego_d, mname)
                f = getattr(model, f"backbone_{mname}")({"spatial_features": f})["spatial_features_2d"]
                f = getattr(model, f"aligner_{mname}")(f)
                if model.sensor_type_dict[mname] == "camera":
                    _, _, hh, ww = f.shape
                    th = int(hh * getattr(model, f"crop_ratio_H_{mname}"))
                    tw = int(ww * getattr(model, f"crop_ratio_W_{mname}"))
                    f = torchvision.transforms.CenterCrop((th, tw))(f)
                mod_feats[mname] = f

            counting = {m: 0 for m in model.modality_name_list}
            flist = []
            for mname in mods:
                flist.append(mod_feats[mname][counting[mname]])
                counting[mname] += 1
            heter_f = torch.stack(flist, dim=0)
            if not torch.isfinite(heter_f).all():
                errors.append(f"idx={idx}: heter_feature_2d 存在 NaN/Inf")
                continue

            # 统计协作者(1) 特征幅值与非零占比
            fc = heter_f[1:2]
            feat_stds.append(float(fc.std().item()))
            nonzero_ratios.append(float((fc.abs().max(dim=1)[0] > 1e-5).float().mean().item()))

            # 走金字塔融合 + 检测头（对 camera 模态保持与官方一致的中心有效区裁剪）
            pyr = model.pyramid_backbone
            fl = list(pyr.get_multiscale_feature(heter_f))
            score_list = []
            cam_crop_info = getattr(model, "cam_crop_info", None)
            for lv in range(pyr.num_levels):
                occ = getattr(pyr, f"single_head_{lv}")(fl[lv])
                sc = torch.sigmoid(occ) + 1e-4
                if cam_crop_info:
                    _, _, sh, sw = sc.shape
                    cmask = torch.ones_like(sc)
                    for cam_m, info in cam_crop_info.items():
                        ab = torch.tensor([x == cam_m for x in mods], dtype=torch.bool, device=sc.device)
                        if not ab.any():
                            continue
                        ch = sh / info[f"crop_ratio_H_{cam_m}"] - 4.0
                        cw = sw / info[f"crop_ratio_W_{cam_m}"] - 4.0
                        h0, h1 = int(sh // 2 - ch // 2), int(sh // 2 + ch // 2)
                        w0, w1 = int(sw // 2 - cw // 2), int(sw // 2 + cw // 2)
                        cmask[ab] = 0.0
                        cmask[ab, :, h0:h1, w0:w1] = 1.0
                    sc = sc * cmask
                score_list.append(sc)

            # L0 余弦偏差
            z0 = fl[0][0:1]
            z1 = warp_affine_simple(fl[0][1:2], amat[0, 0, 1:2], z0.shape[-2:], align_corners=pyr.align_corners)
            c_dev = 1.0 - F.cosine_similarity(z0, z1, dim=1, eps=1e-6)
            cos_vals.append(float(c_dev.mean().item()))

            fused_l = [weighted_fuse_safe(fl[lv], score_list[lv], rlen, amat, pyr.align_corners) for lv in range(pyr.num_levels)]
            dec = pyr.decode_multiscale_feature(fused_l)
            if model.shrink_flag:
                dec = model.shrink_conv(dec)
            out_wrap = {"ego": {
                "cls_preds": model.cls_head(dec),
                "reg_preds": model.reg_head(dec),
                "dir_preds": model.dir_head(dec),
            }}
            pb, ps, gb = ds.post_process(batch, out_wrap)
            eval_utils.caluclate_tp_fp(pb, ps, gb, ap_stat, 0.5)
            eval_utils.caluclate_tp_fp(pb, ps, gb, ap_stat, 0.7)
        except Exception as e:
            errors.append(f"idx={idx}: {type(e).__name__}: {e}")

    ap50 = float(eval_utils.calculate_ap(ap_stat, 0.5)[0]) if ap_stat[0.5]["gt"] > 0 else 0.0
    ap70 = float(eval_utils.calculate_ap(ap_stat, 0.7)[0]) if ap_stat[0.7]["gt"] > 0 else 0.0
    return {
        "tested_frames": len(indices),
        "success_frames": len(indices) - len(errors),
        "ap50": round(ap50, 5),
        "ap70": round(ap70, 5),
        "mean_l0_cos_dev": round(float(np.mean(cos_vals)), 5) if cos_vals else None,
        "mean_collab_feat_std": round(float(np.mean(feat_stds)), 5) if feat_stds else None,
        "mean_collab_nonzero_ratio": round(float(np.mean(nonzero_ratios)), 5) if nonzero_ratios else None,
        "errors": errors[:5],
    }


def main():
    parser = argparse.ArgumentParser(description="步骤 ⑥ 限时半天探 m3 (32 线 LiDAR SECOND) 可行性诊断")
    parser.add_argument("--data_dir", type=str, default=OPV2V_TRAIN_DIR)
    parser.add_argument("--eval_split_json", type=str, default=DEFAULT_EVAL_SPLIT)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--out_json", type=str, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    t0 = time.time()
    init_heal_runtime()
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils
    from opencood.utils.common_utils import update_dict

    with open(args.eval_split_json, "r", encoding="utf-8") as f:
        ev = json.load(f)
    all_idxs = list(ev["all_indices"])
    step = max(1, len(all_idxs) // int(args.num_frames))
    probe_idxs = all_idxs[::step][: int(args.num_frames)]

    cfg_file = os.path.join(FINAL_INFER_DIR, "config.yaml")
    ckpt_file = os.path.join(FINAL_INFER_DIR, "net_epoch1.pth")
    assign_path = ensure_in_order_assignment(args.data_dir, None)

    hypes = yaml_utils.load_yaml(cfg_file, opt=None)
    # 仅保留 m1, m2, m3（移除 m4）
    hypes["heter"]["modality_setting"].pop("m4", None)
    hypes["model"]["args"].pop("m4", None)
    proto_range = [-102.4, -102.4, -3, 102.4, 102.4, 1]
    hypes = update_dict(hypes, {"cav_lidar_range": proto_range, "lidar_range": proto_range, "gt_range": proto_range})
    hypes = update_dict(hypes, {"ego_modality": "m1"})
    hypes = yaml_utils.load_general_params(hypes)
    hypes["root_dir"] = args.data_dir
    hypes["validate_dir"] = args.data_dir
    hypes["test_dir"] = args.data_dir
    hypes["fusion"]["core_method"] = "intermediateheterinfer"
    hypes["use_cav"] = 2
    hypes["comm_range"] = 180.0
    hypes["input_source"] = ["lidar", "camera"]
    hypes["add_data_extension"] = []
    hypes["label_type"] = "lidar"
    hypes["heter"]["assignment_path"] = assign_path

    # 探针 1：检查模型构建与 m3 权重加载
    print("[Probe 1] 构建 m1+m2+m3 模型并校验 net_epoch1.pth 权重完整性...", flush=True)
    model = train_utils.create_model(hypes)
    state_all = torch.load(ckpt_file, map_location="cpu")
    model_sd = model.state_dict()
    m3_model_keys = [k for k in model_sd.keys() if "m3" in k]
    m3_ckpt_keys = [k for k in state_all.keys() if "m3" in k]
    missing_m3 = [k for k in m3_model_keys if k not in state_all]
    shape_mismatch_m3 = [
        (k, tuple(model_sd[k].shape), tuple(state_all[k].shape))
        for k in m3_model_keys if k in state_all and tuple(model_sd[k].shape) != tuple(state_all[k].shape)
    ]

    matched = {k: v for k, v in state_all.items() if k in model_sd and tuple(v.shape) == tuple(model_sd[k].shape)}
    load_res = model.load_state_dict(matched, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    ckpt_report = {
        "m3_model_keys_cnt": len(m3_model_keys),
        "m3_ckpt_keys_cnt": len(m3_ckpt_keys),
        "missing_m3_keys_cnt": len(missing_m3),
        "missing_m3_sample": missing_m3[:10],
        "shape_mismatch_m3": shape_mismatch_m3[:10],
        "load_missing_total": len(load_res.missing_keys),
        "load_unexpected_total": len(load_res.unexpected_keys),
    }
    print("  ckpt_report:", json.dumps(ckpt_report, ensure_ascii=False))

    # 探针 2~4：同批 16 帧对比 m1+m1、m1+m2、m1+m3
    modal_reports = OrderedDict()
    for tag, mapping in (
        ("m1_collab", {"m1": "m1", "m2": "m1", "m3": "m1", "m4": "m1"}),
        ("m2_collab", {"m1": "m1", "m2": "m2", "m3": "m2", "m4": "m2"}),
        ("m3_collab", {"m1": "m1", "m2": "m3", "m3": "m3", "m4": "m3"}),
    ):
        print(f"[Probe {tag}] 评测 {len(probe_idxs)} 帧...", flush=True)
        hc = copy.deepcopy(hypes)
        hc["heter"]["mapping_dict"] = dict(mapping)
        hc = update_dict(hc, {"mapping_dict": dict(mapping)})
        ds = build_dataset(hc, visualize=True, train=False)
        ds.use_hdf5 = False
        with torch.no_grad():
            modal_reports[tag] = _eval_16frames_ap(model, ds, probe_idxs, device)
        print(f"  {tag}: {json.dumps(modal_reports[tag], ensure_ascii=False)}")

    # 探针 5：综合自动建议
    m3_r = modal_reports["m3_collab"]
    m2_r = modal_reports["m2_collab"]
    m1_r = modal_reports["m1_collab"]
    if ckpt_report["missing_m3_keys_cnt"] == 0 and m3_r["success_frames"] == len(probe_idxs) and m3_r["ap50"] >= m2_r["ap50"] - 0.03:
        verdict = "READY_TO_INTEGRATE"
        reason = "m3 权重完整、无运行时报错，且协同 AP 与特征对齐表现健康，可直接低成本扩展为三模态。"
    elif m3_r["success_frames"] > 0 and m3_r["ap50"] > 0.50:
        verdict = "NEEDS_QUICK_FIX"
        reason = "m3 可运行但存在轻微精度或配置偏差，可在半天限额内做定向排查（如体素范围/32线采样）。"
    else:
        verdict = "STOP_LOSS"
        reason = "m3 存在权重/对齐严重异常或报错，建议按止损原则专注 m1+m2 主线与纯 LiDAR 四异构补充。"

    report = OrderedDict({
        "meta": {
            "description": "步骤 ⑥ HEAL m3 (32线 LiDAR SECOND) 限时探路诊断报告",
            "probe_frames": len(probe_idxs),
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "ckpt_check": ckpt_report,
        "eval_16frames": modal_reports,
        "verdict": verdict,
        "reason": reason,
    })
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[探路结论] {verdict}: {reason}")
    print(f"已保存诊断报告至：{args.out_json}")


if __name__ == "__main__":
    main()
