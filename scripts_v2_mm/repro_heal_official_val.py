# -*- coding: utf-8 -*-
"""
使用 HEAL_ref 官方推理管线在全量 opv2v_val（6 序列，965 帧）上复现官方指标，
并在加载权重后统一修复 efficientnet_pytorch 0.7.1 相对官方 0.7.0 的 5 层非对称填充差异。
评测项目：
  1. stage2/m2_alignto_m1/net_epoch25.pth:
     - 官方 no 口径 (range=102.4,102.4, 全 CAV 合并 GT，无 bev_visibility 过滤)
     - 官方 single 口径 (range=51.2,51.2, 单 ego GT，含无过滤及基于 ego .pcd 点云 pts>=1/5/10 近似可见性过滤)
     - 分序列 AP 与高置信预测框径向缩放比 r_pred / r_gt 统计
  2. final_infer/net_epoch1.pth (m2 单车分支):
     - no (102.4) 与 single (51.2, 含无过滤及 pts>=1/5/10 近似)
  3. final_infer/net_epoch1.pth (HeterPyramidCollab 异构协同官方管线, range=204.8,102.4):
     - use_cav=1 (m1 单车 64 线 LiDAR) vs. use_cav=2 (m1 + m2 异构协同)
"""

import os
import sys
import json
import time
import copy
import importlib.util
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader

# -------------------------
# 1. 预加载 box_overlaps 扩展与 HEAL_ref（不修改 HEAL_ref 任何文件）
# -------------------------
HEAL_ROOT = "/root/autodl-tmp/proj_dir2_hetero_trust/third_party_v2_mm/HEAL_ref"
BOX_OVERLAPS_SO = (
    "/root/autodl-tmp/proj_dir2_hetero_trust/third_party/OpenCOOD/"
    "opencood/utils/box_overlaps.cpython-38-x86_64-linux-gnu.so"
)

spec = importlib.util.spec_from_file_location("opencood.utils.box_overlaps", BOX_OVERLAPS_SO)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.modules["opencood.utils.box_overlaps"] = mod
sys.path.insert(0, HEAL_ROOT)

from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.data_utils.datasets.basedataset.opv2v_basedataset import OPV2VBaseDataset
from opencood.utils import eval_utils
from opencood.utils.common_utils import update_dict

# 内存级兼容：opv2v_val 上相机文件为 .jpg，若 .png 不存在则自动回退同名 .jpg
_orig_find_camera_files = OPV2VBaseDataset.find_camera_files


@staticmethod
def _patched_find_camera_files(cav_path, timestamp, sensor="camera"):
    files = _orig_find_camera_files(cav_path, timestamp, sensor=sensor)
    out = []
    for p in files:
        if os.path.exists(p):
            out.append(p)
        else:
            jpg_p = p[:-4] + ".jpg" if p.endswith(".png") else p
            out.append(jpg_p if os.path.exists(jpg_p) else p)
    return out


OPV2VBaseDataset.find_camera_files = _patched_find_camera_files

VAL_DIR = "/root/autodl-tmp/proj_dir2_hetero_trust/data_links/opv2v_val"
STAGE2_M2_DIR = "/root/autodl-tmp/proj_dir2_hetero_trust/checkpoints_v2_mm/heal_official/stage2/m2_alignto_m1"
FINAL_INFER_DIR = "/root/autodl-tmp/proj_dir2_hetero_trust/checkpoints_v2_mm/heal_official/final_infer"
ASSIGN_4MOD = os.path.join(HEAL_ROOT, "opencood/modality_assign/opv2v_4modality.json")
TMP_IN_ORDER_JSON = "/root/autodl-tmp/proj_dir2_hetero_trust/tmp_png_ab_test/opv2v_val_4modality_in_order.json"
OUT_JSON = "/root/autodl-tmp/proj_dir2_hetero_trust/results_rq1_v2_mm/heal_m2_official_repro_report.json"


def fix_effnet_symmetric_padding(model):
    """
    将 efficientnet_pytorch 0.7.1 中 5 层 stride=2 的非对称 Conv2dStaticSamePadding
    恢复为官方 requirements.txt 指定的 0.7.0 对称填充 (1,1,1,1)/(2,2,2,2)。
    """
    fixed = []
    for name, m in model.named_modules():
        if hasattr(m, "static_padding") and isinstance(m.static_padding, torch.nn.ZeroPad2d):
            pad = m.static_padding.padding
            if isinstance(pad, tuple) and len(pad) == 4 and (pad[0] != pad[1] or pad[2] != pad[3]):
                sym = (max(pad[0], pad[1]), max(pad[0], pad[1]), max(pad[2], pad[3]), max(pad[2], pad[3]))
                m.static_padding.padding = sym
                fixed.append({"layer": name, "from": list(pad), "to": list(sym)})
    return fixed


def disable_depth_supervision_m2(model):
    """关闭推理阶段读取第4通道 depth GT 的分支。"""
    if hasattr(model, "depth_supervision_m2"):
        model.depth_supervision_m2 = False
    if hasattr(model, "encoder_m2"):
        model.encoder_m2.depth_supervision = False
        if hasattr(model.encoder_m2, "camencode"):
            model.encoder_m2.camencode.depth_supervision = False


def init_stat():
    return {t: {"tp": [], "fp": [], "gt": 0, "score": []} for t in [0.3, 0.5, 0.7]}


def finalize_stat(stat):
    res = {}
    for t in [0.3, 0.5, 0.7]:
        ap = float(eval_utils.calculate_ap(stat, t)[0]) if (len(stat[t]["tp"]) > 0 and stat[t]["gt"] > 0) else 0.0
        res[f"ap{int(round(t * 100))}"] = ap
    res["gt_total"] = int(stat[0.5]["gt"])
    res["pred_total"] = int(len(stat[0.5]["tp"]))
    return res


def count_lidar_points_in_corners3d(points_xyz: np.ndarray, corners_3d: np.ndarray):
    """统计 ego LiDAR 点云落入每个 3D GT 边界框 [N, 8, 3] 内部的点数（采用纯几何投影）。"""
    n_boxes = corners_3d.shape[0]
    counts = np.zeros(n_boxes, dtype=np.int32)
    if n_boxes == 0 or points_xyz.shape[0] == 0:
        return counts

    pts = points_xyz[:, :3]
    for i in range(n_boxes):
        c = corners_3d[i]
        x_min, x_max = float(c[:, 0].min()), float(c[:, 0].max())
        y_min, y_max = float(c[:, 1].min()), float(c[:, 1].max())
        z_min, z_max = float(c[:, 2].min()), float(c[:, 2].max())

        aabb_mask = (
            (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max)
            & (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max)
            & (pts[:, 2] >= z_min - 0.1) & (pts[:, 2] <= z_max + 0.1)
        )
        if not np.any(aabb_mask):
            continue
        p_sub = pts[aabb_mask, :2]

        z_mid = 0.5 * (z_min + z_max)
        bot = c[c[:, 2] <= z_mid][:, :2]
        if bot.shape[0] != 4:
            bot = c[np.argsort(c[:, 2])[:4]][:, :2]
        center = bot.mean(axis=0)
        angles = np.arctan2(bot[:, 1] - center[1], bot[:, 0] - center[0])
        bot = bot[np.argsort(angles)]
        p0, p1, p3 = bot[0], bot[1], bot[3]
        u = p1 - p0
        v = p3 - p0
        uu = float(np.dot(u, u)) + 1e-8
        vv = float(np.dot(v, v)) + 1e-8

        rel = p_sub - p0
        proj_u = rel @ u
        proj_v = rel @ v
        inside = (proj_u >= 0.0) & (proj_u <= uu) & (proj_v >= 0.0) & (proj_v <= vv)
        counts[i] = int(np.sum(inside))
    return counts


def get_scenario_index_per_sample(dataset):
    """返回每个样本索引对应的序列名。"""
    seq_names = [os.path.basename(p) for p in dataset.scenario_folders]
    sample_seqs = []
    prev = 0
    for seq_idx, end_idx in enumerate(dataset.len_record):
        sample_seqs.extend([seq_names[seq_idx]] * (end_idx - prev))
        prev = end_idx
    return sample_seqs


def build_val_in_order_assignment():
    """按 HEAL 官方 assign_modality_4_in_order 规则生成 opv2v_val 顺序模态分配 JSON。"""
    assign = OrderedDict()
    for seq in sorted(os.listdir(VAL_DIR)):
        sp = os.path.join(VAL_DIR, seq)
        if not os.path.isdir(sp):
            continue
        cavs = sorted([c for c in os.listdir(sp) if os.path.isdir(os.path.join(sp, c))])
        if cavs and cavs[0] == "-1":
            cavs = cavs[1:] + cavs[:1]
        assign[seq] = OrderedDict()
        for j, cav_id in enumerate(cavs):
            assign[seq][cav_id] = f"m{j % 4 + 1}"
    os.makedirs(os.path.dirname(TMP_IN_ORDER_JSON), exist_ok=True)
    with open(TMP_IN_ORDER_JSON, "w", encoding="utf-8") as f:
        json.dump(assign, f, indent=2)
    return assign


def run_m2_single_or_no(ckpt_path, range_xy, mode_name, device, calc_vis_and_radial=False):
    """使用官方 LateheterFusionDataset + inference_no_fusion 在全量 opv2v_val 上评测 m2。"""
    t0 = time.time()
    hypes = yaml_utils.load_yaml(os.path.join(STAGE2_M2_DIR, "config.yaml"), opt=None)
    new_r = [-range_xy, -range_xy, -3, range_xy, range_xy, 1]
    hypes = update_dict(hypes, {"cav_lidar_range": new_r, "lidar_range": new_r, "gt_range": new_r})
    hypes = yaml_utils.load_general_params(hypes)
    hypes["validate_dir"] = VAL_DIR
    hypes["test_dir"] = VAL_DIR
    hypes["input_source"] = ["camera"]
    hypes["add_data_extension"] = []
    hypes["label_type"] = "lidar"
    hypes["heter"]["assignment_path"] = ASSIGN_4MOD

    model = train_utils.create_model(hypes)
    ck = torch.load(ckpt_path, map_location="cpu")
    sub = {k: v for k, v in ck.items() if k in model.state_dict()}
    load_res = model.load_state_dict(sub, strict=True)
    fixed = fix_effnet_symmetric_padding(model)
    assert len(fixed) == 5, f"期望修复 5 层 EfficientNet padding，实际={len(fixed)}"
    disable_depth_supervision_m2(model)
    model.to(device).eval()

    ds = build_dataset(hypes, visualize=True, train=False)
    ds.use_hdf5 = False
    sample_seqs = get_scenario_index_per_sample(ds)
    dl = DataLoader(
        ds,
        batch_size=1,
        num_workers=4,
        collate_fn=ds.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    stat_all = init_stat()
    by_seq_stat = {}
    vis_stats = {k: init_stat() for k in [1, 5, 10]} if calc_vis_and_radial else None
    radial_ratios = []
    radial_diffs = []
    single_gt = (mode_name == "single")

    print(
        f"\n[START] {os.path.basename(os.path.dirname(ckpt_path))}/{os.path.basename(ckpt_path)} "
        f"mode={mode_name} range={range_xy} frames={len(ds)} strict={load_res}",
        flush=True,
    )

    for idx, batch_data in enumerate(dl):
        if batch_data is None:
            continue
        seq = sample_seqs[idx]
        if seq not in by_seq_stat:
            by_seq_stat[seq] = init_stat()

        origin_lidar_np = batch_data["ego"]["origin_lidar"][0].numpy() if calc_vis_and_radial else None

        with torch.no_grad():
            batch_gpu = train_utils.to_device(batch_data, device)
            infer_res = inference_utils.inference_no_fusion(batch_gpu, model, ds, single_gt=single_gt)

        pb = infer_res["pred_box_tensor"]
        ps = infer_res["pred_score"]
        gt = infer_res["gt_box_tensor"]

        for t in [0.3, 0.5, 0.7]:
            eval_utils.caluclate_tp_fp(pb, ps, gt, stat_all, t)
            eval_utils.caluclate_tp_fp(pb, ps, gt, by_seq_stat[seq], t)

        if calc_vis_and_radial and gt is not None and len(gt) > 0:
            gt_np = gt.cpu().numpy()
            pt_counts = count_lidar_points_in_corners3d(origin_lidar_np, gt_np)
            for min_pts in [1, 5, 10]:
                keep_g = pt_counts >= min_pts
                gt_sub = gt[torch.from_numpy(keep_g).to(gt.device)]
                for t in [0.3, 0.5, 0.7]:
                    eval_utils.caluclate_tp_fp(pb, ps, gt_sub, vis_stats[min_pts], t)

            if pb is not None and len(pb) > 0:
                gc = gt_np.mean(axis=1)[:, :2]
                pc = pb.cpu().numpy().mean(axis=1)[:, :2]
                ps_np = ps.cpu().numpy()
                for pi in range(len(pc)):
                    if ps_np[pi] < 0.4:
                        continue
                    dists = np.linalg.norm(gc - pc[pi], axis=1)
                    gi = int(np.argmin(dists))
                    if dists[gi] < 3.5:
                        rg = float(np.linalg.norm(gc[gi]))
                        rp = float(np.linalg.norm(pc[pi]))
                        if rg > 5.0:
                            radial_ratios.append(rp / rg)
                            radial_diffs.append(rp - rg)

        if (idx + 1) % 200 == 0 or (idx + 1) == len(ds):
            print(f"  [{mode_name} {range_xy}] 进度 {idx + 1}/{len(ds)}, 耗时 {time.time() - t0:.1f}s", flush=True)

    out = {
        "ckpt": ckpt_path,
        "mode": mode_name,
        "range": [-range_xy, -range_xy, -3, range_xy, range_xy, 1],
        "num_frames": len(ds),
        "fixed_padding_layers": len(fixed),
        "elapsed_sec": round(time.time() - t0, 2),
        "overall": finalize_stat(stat_all),
        "per_sequence": {s: finalize_stat(st) for s, st in sorted(by_seq_stat.items())},
    }
    if calc_vis_and_radial:
        out["lidar_visibility_approx"] = {
            f"pts_ge_{k}": finalize_stat(vis_stats[k]) for k in [1, 5, 10]
        }
        rr = np.asarray(radial_ratios, dtype=np.float64)
        rd = np.asarray(radial_diffs, dtype=np.float64)
        out["radial_bias_score_ge_04"] = {
            "count": int(rr.size),
            "ratio_mean": float(rr.mean()) if rr.size else None,
            "ratio_median": float(np.median(rr)) if rr.size else None,
            "diff_m_mean": float(rd.mean()) if rd.size else None,
            "diff_m_median": float(np.median(rd)) if rd.size else None,
        }
    print(f"[DONE] {mode_name} range={range_xy}: overall={out['overall']}", flush=True)
    return out


def run_collab_m1_m2(device):
    """
    使用官方 final_infer/net_epoch1.pth + HeterPyramidCollab + IntermediateHeterinferFusionDataset
    在全量 opv2v_val 上对比 use_cav=1 (m1 单车 64 线 LiDAR) 与 use_cav=2 (m1 + m2 异构协同)。
    """
    t0 = time.time()
    hypes = yaml_utils.load_yaml(os.path.join(FINAL_INFER_DIR, "config.yaml"), opt=None)

    for drop_m in ["m3", "m4"]:
        hypes["heter"]["modality_setting"].pop(drop_m, None)
        hypes["model"]["args"].pop(drop_m, None)
    if "lidar_channels_dict" in hypes["heter"]:
        hypes["heter"]["lidar_channels_dict"] = {}

    new_r = [-204.8, -102.4, -3, 204.8, 102.4, 1]
    hypes = update_dict(hypes, {"cav_lidar_range": new_r, "lidar_range": new_r, "gt_range": new_r})
    hypes = update_dict(
        hypes,
        {
            "mapping_dict": {"m1": "m1", "m2": "m2", "m3": "m2", "m4": "m2"},
            "ego_modality": "m1",
        },
    )
    hypes = yaml_utils.load_general_params(hypes)
    hypes["validate_dir"] = VAL_DIR
    hypes["test_dir"] = VAL_DIR
    hypes["fusion"]["core_method"] = "intermediateheterinfer"
    hypes["comm_range"] = 180
    hypes["input_source"] = ["lidar", "camera"]
    hypes["add_data_extension"] = []
    hypes["label_type"] = "lidar"
    hypes["heter"]["assignment_path"] = TMP_IN_ORDER_JSON

    model = train_utils.create_model(hypes)
    ck = torch.load(os.path.join(FINAL_INFER_DIR, "net_epoch1.pth"), map_location="cpu")
    model_sd = model.state_dict()
    matched = {k: v for k, v in ck.items() if k in model_sd}
    load_res = model.load_state_dict(matched, strict=True)
    fixed = fix_effnet_symmetric_padding(model)
    assert len(fixed) == 5, f"HeterPyramidCollab 期望修复 5 层 EfficientNet padding，实际={len(fixed)}"
    disable_depth_supervision_m2(model)
    model.to(device).eval()

    collab_results = {}
    for use_cav in [1, 2]:
        t_c = time.time()
        h_run = copy.deepcopy(hypes)
        h_run["use_cav"] = use_cav
        ds = build_dataset(h_run, visualize=True, train=False)
        ds.use_hdf5 = False
        sample_seqs = get_scenario_index_per_sample(ds)
        dl = DataLoader(
            ds,
            batch_size=1,
            num_workers=4,
            collate_fn=ds.collate_batch_test,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )

        stat_204 = init_stat()
        stat_51 = init_stat()
        by_seq = {}
        modality_counter = {}

        print(f"\n[START] HeterPyramidCollab use_cav={use_cav} frames={len(ds)} strict={load_res}", flush=True)
        for idx, batch_data in enumerate(dl):
            if batch_data is None:
                continue
            seq = sample_seqs[idx]
            if seq not in by_seq:
                by_seq[seq] = init_stat()

            mods_key = "+".join(batch_data["ego"]["agent_modality_list"])
            modality_counter[mods_key] = modality_counter.get(mods_key, 0) + 1

            with torch.no_grad():
                batch_gpu = train_utils.to_device(batch_data, device)
                infer_res = inference_utils.inference_intermediate_fusion(batch_gpu, model, ds)

            pb = infer_res["pred_box_tensor"]
            ps = infer_res["pred_score"]
            gt = infer_res["gt_box_tensor"]

            for t in [0.3, 0.5, 0.7]:
                eval_utils.caluclate_tp_fp(pb, ps, gt, stat_204, t)
                eval_utils.caluclate_tp_fp(pb, ps, gt, by_seq[seq], t)

            gt_c = gt.mean(dim=1)
            gt_51 = gt[(gt_c[:, 0].abs() <= 51.2) & (gt_c[:, 1].abs() <= 51.2)]
            if pb is not None and len(pb) > 0:
                pb_c = pb.mean(dim=1)
                mk = (pb_c[:, 0].abs() <= 51.2) & (pb_c[:, 1].abs() <= 51.2)
                pb_51, ps_51 = (pb[mk], ps[mk]) if mk.any() else (None, None)
            else:
                pb_51, ps_51 = None, None
            for t in [0.3, 0.5, 0.7]:
                eval_utils.caluclate_tp_fp(pb_51, ps_51, gt_51, stat_51, t)

            if (idx + 1) % 200 == 0 or (idx + 1) == len(ds):
                print(f"  [collab use_cav={use_cav}] 进度 {idx + 1}/{len(ds)}, 耗时 {time.time() - t_c:.1f}s", flush=True)

        collab_results[f"use_cav{use_cav}"] = {
            "use_cav": use_cav,
            "modality_combinations": modality_counter,
            "elapsed_sec": round(time.time() - t_c, 2),
            "range_204.8_102.4": finalize_stat(stat_204),
            "range_51.2_51.2_subset": finalize_stat(stat_51),
            "per_sequence_204.8_102.4": {s: finalize_stat(st) for s, st in sorted(by_seq.items())},
        }
        print(f"[DONE] collab use_cav={use_cav}: {collab_results[f'use_cav{use_cav}']['range_204.8_102.4']}", flush=True)

    collab_results["total_elapsed_sec"] = round(time.time() - t0, 2)
    return collab_results


def main():
    t_all = time.time()
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 运行设备: {device}", flush=True)

    assign_val = build_val_in_order_assignment()
    print(f"[INFO] 已生成 opv2v_val 顺序模态分配: {TMP_IN_ORDER_JSON} ({len(assign_val)} 序列)", flush=True)

    ckpt_s2 = os.path.join(STAGE2_M2_DIR, "net_epoch25.pth")
    ckpt_fi = os.path.join(FINAL_INFER_DIR, "net_epoch1.pth")

    report = {
        "meta": {
            "val_dir": VAL_DIR,
            "efficientnet_pytorch_installed": __import__("efficientnet_pytorch").__version__,
            "padding_fix_applied": "symmetric (0.7.0 behavior: (1,1,1,1)/(2,2,2,2) on 5 stride=2 layers)",
            "image_format_on_disk": "Q95-420 JPG (opv2v_val)",
            "note_visibility": "opv2v_val 当前未含 additional-001.zip 中的 bev_visibility.png，单车评估同步报告全量 LiDAR GT 与基于 ego .pcd 点云 pts>=1/5/10 近似可见性过滤结果",
        },
        "m2_stage2_no_102.4": run_m2_single_or_no(ckpt_s2, 102.4, "no", device, calc_vis_and_radial=False),
        "m2_stage2_single_51.2": run_m2_single_or_no(ckpt_s2, 51.2, "single", device, calc_vis_and_radial=True),
        "m2_final_infer_no_102.4": run_m2_single_or_no(ckpt_fi, 102.4, "no", device, calc_vis_and_radial=False),
        "m2_final_infer_single_51.2": run_m2_single_or_no(ckpt_fi, 51.2, "single", device, calc_vis_and_radial=True),
        "collab_final_infer_m1_m2": run_collab_m1_m2(device),
    }
    report["meta"]["total_elapsed_sec"] = round(time.time() - t_all, 2)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[ALL DONE] 完整复现报告已保存至 {OUT_JSON}，总耗时 {report['meta']['total_elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
