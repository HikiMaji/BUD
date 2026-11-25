# -*- coding: utf-8 -*-
"""
Raw PNG vs. Q95-420 JPG vs. Q95-444 JPG (subsampling=0) 三组配对定量 A/B 测试脚本
基于官方 HEAL_OPV2V 预训练 Camera-LSS (stage2/m2_alignto_m1/net_epoch25.pth)
评估维度：
  1. 像素级保真度 (PSNR / SSIM / RGB RMSE / 文件体积)
  2. BEV 特征与深度/分类头一致性 (feat_lss / feat_aligned / depth KL / cls sigmoid)
  3. 3D 目标检测一致性 (AP@0.3/0.5/0.7, ΔAP, score>0.3 框匹配率与中心偏移)
  4. 协议空间安全统计量敏感度 (HetShield 逐栅格余弦异常度 1-cos 与范数漂移)
"""

import os
import sys
import io
import json
import time
import math
import importlib.util
from collections import OrderedDict

import yaml
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from skimage.metrics import structural_similarity as compute_ssim

# -------------------------
# 路径与扩展模块预加载（只读使用 HEAL_ref，不修改其源码）
# -------------------------
HEAL_ROOT = "/root/autodl-tmp/proj_dir2_hetero_trust/third_party_v2_mm/HEAL_ref"
BOX_OVERLAPS_SO = (
    "/root/autodl-tmp/proj_dir2_hetero_trust/third_party/OpenCOOD/"
    "opencood/utils/box_overlaps.cpython-38-x86_64-linux-gnu.so"
)

if os.path.exists(BOX_OVERLAPS_SO):
    spec = importlib.util.spec_from_file_location("opencood.utils.box_overlaps", BOX_OVERLAPS_SO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["opencood.utils.box_overlaps"] = mod
else:
    raise FileNotFoundError(f"未找到编译好的 box_overlaps 扩展: {BOX_OVERLAPS_SO}")

sys.path.insert(0, HEAL_ROOT)

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.heter_pyramid_single import HeterPyramidSingle
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.utils import eval_utils, common_utils
from opencood.utils.camera_utils import sample_augmentation, img_transform, normalize_img
from opencood.utils.transformation_utils import x1_to_x2

# 试验数据与模型路径配置
CONFIG_PATH = "/root/autodl-tmp/proj_dir2_hetero_trust/checkpoints_v2_mm/heal_official/stage2/m2_alignto_m1/config.yaml"
CKPT_PATH = "/root/autodl-tmp/proj_dir2_hetero_trust/checkpoints_v2_mm/heal_official/stage2/m2_alignto_m1/net_epoch25.pth"
PNG_ROOT = "/root/autodl-tmp/proj_dir2_hetero_trust/tmp_png_ab_test/val_30frames_raw_png"
SAMPLE_JSON = os.path.join(PNG_ROOT, "sample_list.json")
JPG_VAL_ROOT = "/root/autodl-tmp/proj_dir2_hetero_trust/data_links/opv2v_val"
OUT_DIR = "/root/autodl-tmp/proj_dir2_hetero_trust/results_rq1_v2_mm"
OUT_JSON = os.path.join(OUT_DIR, "png_vs_jpg_ab_report_fixed_pad.json")

# CARLA/UE4 到 OpenCV 相机坐标轴变换矩阵（与 HEAL opv2v_basedataset.py 严格一致）
UE4_TO_OPENCV = np.array(
    [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)
CAMS = ["camera0", "camera1", "camera2", "camera3"]


def encode_decode_q95_444(pil_rgb: Image.Image):
    """在内存中以 Q95、4:4:4 无色度子采样 (subsampling=0) 编码并重新解码为 RGB 图像。"""
    buf = io.BytesIO()
    pil_rgb.save(buf, format="JPEG", quality=95, subsampling=0, optimize=True)
    nbytes = buf.tell()
    buf.seek(0)
    decoded = Image.open(buf).convert("RGB")
    decoded.load()
    return decoded, nbytes


def detect_jpeg_subsampling(jpg_path: str):
    """读取 JPEG 文件色度采样标识，用于确认磁盘 JPG 的实际编码格式。"""
    try:
        from PIL import JpegImagePlugin
        with Image.open(jpg_path) as im:
            sub = JpegImagePlugin.get_sampling(im)
            return int(sub)
    except Exception:
        return -1


def calc_pixel_metrics(arr_ref: np.ndarray, arr_cmp: np.ndarray):
    """计算单张 RGB uint8 图像的 PSNR (dB)、SSIM 以及归一化到 [0,1] 的 RMSE。"""
    diff_f = (arr_cmp.astype(np.float64) - arr_ref.astype(np.float64)) / 255.0
    rmse_01 = float(np.sqrt(np.mean(diff_f ** 2)))
    mse_255 = float(np.mean((arr_cmp.astype(np.float64) - arr_ref.astype(np.float64)) ** 2))
    psnr = 100.0 if mse_255 < 1e-12 else float(10.0 * math.log10((255.0 ** 2) / mse_255))
    ssim = float(compute_ssim(arr_ref, arr_cmp, channel_axis=2, data_range=255))
    return psnr, ssim, rmse_01


def build_camera_input_dict(pil_imgs, params, data_aug_conf, device):
    """严格对齐 HEAL 中 Camera 分支推理阶段预处理与相机外参矩阵构造。"""
    imgs_list, intrins_list, extrins_list = [], [], []
    rots_list, trans_list, post_rots_list, post_trans_list = [], [], [], []
    lidar_pose = np.array(params["lidar_pose"], dtype=np.float32)

    resize, resize_dims, crop, flip, rotate = sample_augmentation(data_aug_conf, is_train=False)

    for idx, pil_img in enumerate(pil_imgs):
        cam_key = f"camera{idx}"
        cam_coords = np.array(params[cam_key]["cords"], dtype=np.float32)
        camera_to_lidar = x1_to_x2(cam_coords, lidar_pose).astype(np.float32) @ UE4_TO_OPENCV
        cam_intrin = np.array(params[cam_key]["intrinsic"], dtype=np.float32)

        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)
        img_out, post_rot2, post_tran2 = img_transform(
            [pil_img],
            post_rot,
            post_tran,
            resize=resize,
            resize_dims=resize_dims,
            crop=crop,
            flip=flip,
            rotate=rotate,
        )
        post_tran_3 = torch.zeros(3)
        post_rot_3 = torch.eye(3)
        post_tran_3[:2] = post_tran2
        post_rot_3[:2, :2] = post_rot2

        img_tensor = normalize_img(img_out[0])
        imgs_list.append(img_tensor)
        intrins_list.append(torch.from_numpy(cam_intrin))
        extrins_list.append(torch.from_numpy(camera_to_lidar))
        rots_list.append(torch.from_numpy(camera_to_lidar[:3, :3].copy()))
        trans_list.append(torch.from_numpy(camera_to_lidar[:3, 3].copy()))
        post_rots_list.append(post_rot_3)
        post_trans_list.append(post_tran_3)

    return {
        "inputs_m2": {
            "imgs": torch.stack(imgs_list).unsqueeze(0).to(device),
            "intrins": torch.stack(intrins_list).unsqueeze(0).to(device),
            "extrinsics": torch.stack(extrins_list).unsqueeze(0).to(device),
            "rots": torch.stack(rots_list).unsqueeze(0).to(device),
            "trans": torch.stack(trans_list).unsqueeze(0).to(device),
            "post_rots": torch.stack(post_rots_list).unsqueeze(0).to(device),
            "post_trans": torch.stack(post_trans_list).unsqueeze(0).to(device),
        }
    }


def build_gt_and_post_data_dict(params, post_processor, anchor_box_cpu):
    """构造单车相机后处理及评估所需的 CPU data_dict（与 HEAL_ref 推理阶段一致使用 lidar 全范围真值框）。"""
    ego_pose = np.array(params["lidar_pose"], dtype=np.float32)
    cav_contents = [{"params": params}]
    object_np, object_mask, object_ids = post_processor.generate_object_center(
        cav_contents, ego_pose, enlarge_z=False
    )
    eye4 = torch.eye(4, dtype=torch.float32)
    return {
        "ego": {
            "transformation_matrix": eye4,
            "transformation_matrix_clean": eye4,
            "anchor_box": anchor_box_cpu,
            "object_bbx_center": torch.from_numpy(object_np).unsqueeze(0).float(),
            "object_bbx_mask": torch.from_numpy(object_mask).unsqueeze(0).float(),
            "object_ids": object_ids,
        }
    }


def init_result_stat():
    return {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }


def safe_calculate_ap(result_stat, iou_th):
    if len(result_stat[iou_th]["tp"]) == 0 or result_stat[iou_th]["gt"] == 0:
        return 0.0
    ap, _, _ = eval_utils.calculate_ap(result_stat, iou_th)
    return float(ap)


def compute_bev_feature_diff(feat_ref: torch.Tensor, feat_cmp: torch.Tensor, active_mask: torch.Tensor = None):
    """
    计算两张 BEV 特征图 [1, C, H, W] 的全图及活跃区域余弦相似度、相对 L2 误差与逐栅格异常统计。
    返回仅基于全图非零特征或指定前景掩码的计算结果。
    """
    f_ref = feat_ref[0]  # [C, H, W]
    f_cmp = feat_cmp[0]  # [C, H, W]

    # 全局展平相似度与相对 L2 误差
    v_ref = f_ref.reshape(-1)
    v_cmp = f_cmp.reshape(-1)
    cos_global = float(F.cosine_similarity(v_ref.unsqueeze(0), v_cmp.unsqueeze(0), eps=1e-8).item())
    rel_l2_global = float((torch.norm(v_cmp - v_ref, p=2) / (torch.norm(v_ref, p=2) + 1e-8)).item())

    # 逐栅格向量余弦相似度与范数相对漂移
    norm_ref = torch.norm(f_ref, p=2, dim=0)  # [H, W]
    norm_cmp = torch.norm(f_cmp, p=2, dim=0)  # [H, W]
    valid_mask = norm_ref > 1e-6

    dot = torch.sum(f_ref * f_cmp, dim=0)
    cos_map = dot / (norm_ref * norm_cmp + 1e-8)
    cos_map = torch.clamp(cos_map, -1.0, 1.0)
    anom_map = 1.0 - cos_map
    norm_rel_map = torch.abs(norm_cmp - norm_ref) / (norm_ref + 1e-8)

    out = {
        "cos_global": cos_global,
        "rel_l2_global": rel_l2_global,
        "grid_cos_valid": cos_map[valid_mask].cpu().numpy(),
        "grid_anom_valid": anom_map[valid_mask].cpu().numpy(),
        "grid_norm_rel_valid": norm_rel_map[valid_mask].cpu().numpy(),
    }

    if active_mask is not None and bool(active_mask.any()):
        a_ref = f_ref[:, active_mask]
        a_cmp = f_cmp[:, active_mask]
        cos_act_vec = float(
            F.cosine_similarity(a_ref.reshape(1, -1), a_cmp.reshape(1, -1), eps=1e-8).item()
        )
        rel_l2_act = float((torch.norm(a_cmp - a_ref, p=2) / (torch.norm(a_ref, p=2) + 1e-8)).item())
        out.update({
            "cos_active_vec": cos_act_vec,
            "rel_l2_active": rel_l2_act,
            "grid_anom_active": anom_map[active_mask].cpu().numpy(),
            "grid_norm_rel_active": norm_rel_map[active_mask].cpu().numpy(),
        })
    return out


def match_predictions_against_png(pred_png, score_png, pred_cmp, score_cmp, score_thresh=0.3):
    """以 score > score_thresh 筛选高置信预测框，按置信度从高到低匹配 BEV 多边形，统计一致率与中心偏移。"""
    if pred_png is None or len(pred_png) == 0:
        n_cmp = 0 if pred_cmp is None else int((score_cmp > score_thresh).sum().item())
        return {"n_png": 0, "n_cmp": n_cmp, "n_matched": 0, "matched_ious": [], "center_err_cm": []}

    png_np = common_utils.torch_tensor_to_numpy(pred_png)
    s_png_np = common_utils.torch_tensor_to_numpy(score_png)
    keep_png = s_png_np > score_thresh
    png_np = png_np[keep_png]
    s_png_np = s_png_np[keep_png]

    if pred_cmp is None or len(pred_cmp) == 0:
        return {"n_png": int(len(png_np)), "n_cmp": 0, "n_matched": 0, "matched_ious": [], "center_err_cm": []}

    cmp_np = common_utils.torch_tensor_to_numpy(pred_cmp)
    s_cmp_np = common_utils.torch_tensor_to_numpy(score_cmp)
    keep_cmp = s_cmp_np > score_thresh
    cmp_np = cmp_np[keep_cmp]
    s_cmp_np = s_cmp_np[keep_cmp]

    if len(png_np) == 0 or len(cmp_np) == 0:
        return {
            "n_png": int(len(png_np)),
            "n_cmp": int(len(cmp_np)),
            "n_matched": 0,
            "matched_ious": [],
            "center_err_cm": [],
        }

    # 从高置信度到低置信度匹配
    order_cmp = np.argsort(-s_cmp_np)
    cmp_np = cmp_np[order_cmp]
    polys_png = list(common_utils.convert_format(png_np))
    polys_cmp = list(common_utils.convert_format(cmp_np))
    centers_png = png_np.mean(axis=1)[:, :2]  # (N, 2) 单位：米
    centers_cmp = cmp_np.mean(axis=1)[:, :2]

    matched_ious = []
    center_err_cm = []
    avail_idx = list(range(len(polys_png)))
    for i in range(len(polys_cmp)):
        if not avail_idx:
            break
        cand_polys = [polys_png[j] for j in avail_idx]
        ious = common_utils.compute_iou(polys_cmp[i], cand_polys)
        best_pos = int(np.argmax(ious))
        best_iou = float(ious[best_pos])
        if best_iou >= 0.5:
            matched_j = avail_idx.pop(best_pos)
            matched_ious.append(best_iou)
            dist_m = float(np.linalg.norm(centers_cmp[i] - centers_png[matched_j]))
            center_err_cm.append(dist_m * 100.0)

    return {
        "n_png": int(len(png_np)),
        "n_cmp": int(len(cmp_np)),
        "n_matched": int(len(matched_ious)),
        "matched_ious": matched_ious,
        "center_err_cm": center_err_cm,
    }


def summarize_array(arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return {"mean": None, "p95": None, "p99": None, "max": None, "min": None}
    return {
        "mean": float(np.mean(a)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(np.max(a)),
        "min": float(np.min(a)),
    }


def fix_effnet_symmetric_padding(model):
    """将 efficientnet_pytorch 0.7.1 的 5 层非对称静态填充恢复为官方 0.7.0 对称填充。"""
    fixed = []
    for name, m in model.named_modules():
        if hasattr(m, "static_padding") and isinstance(m.static_padding, torch.nn.ZeroPad2d):
            pad = m.static_padding.padding
            if isinstance(pad, tuple) and len(pad) == 4 and (pad[0] != pad[1] or pad[2] != pad[3]):
                sym = (max(pad[0], pad[1]), max(pad[0], pad[1]), max(pad[2], pad[3]), max(pad[2], pad[3]))
                m.static_padding.padding = sym
                fixed.append(name)
    return fixed


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 运行设备: {device}", flush=True)

    # 1. 加载官方 HEAL Camera-LSS 配置与权重
    cfg = load_yaml(CONFIG_PATH, opt=None)
    model = HeterPyramidSingle(cfg["model"]["args"])
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    load_res = model.load_state_dict(ckpt, strict=True)
    print(f"[INFO] 权重严格加载完成: {load_res}", flush=True)
    fixed_layers = fix_effnet_symmetric_padding(model)
    assert len(fixed_layers) == 5, f"Expected 5 fixed layers, got {len(fixed_layers)}"
    print(f"[INFO] 已修复 EfficientNet 对称 padding 层数: {len(fixed_layers)} -> {fixed_layers}", flush=True)

    # 关键：由于验证数据无 depth 真值图，加载权重后关闭推理期 depth_supervision 索引第4通道分支
    model.depth_supervision_m2 = False
    model.encoder_m2.depth_supervision = False
    model.encoder_m2.camencode.depth_supervision = False

    model.to(device)
    model.eval()

    post_processor = VoxelPostprocessor(cfg["postprocess"], train=False)
    anchor_box_cpu = torch.from_numpy(post_processor.generate_anchor_box()).float()
    data_aug_conf = cfg["heter"]["modality_setting"]["m2"]["data_aug_conf"]
    crop_h_ratio = float(model.crop_ratio_H_m2)
    crop_w_ratio = float(model.crop_ratio_W_m2)

    # 注册前向 Hook 提取 LSS 特征、深度分布预测、对齐后协议特征
    hook_cache = {}

    def hook_depth_head(_mod, _inp, out):
        # depth_head 输出为 logits: [B*N, 48, fH, fW]
        hook_cache["depth_probs"] = F.softmax(out.detach(), dim=1)

    def hook_encoder(_mod, _inp, out):
        hook_cache["feat_lss"] = out.detach()

    def hook_aligner(_mod, _inp, out):
        _, _, h, w = out.shape
        cropped = torchvision.transforms.CenterCrop(
            (int(h * crop_h_ratio), int(w * crop_w_ratio))
        )(out.detach())
        hook_cache["feat_aligned"] = cropped

    model.encoder_m2.camencode.depth_head.register_forward_hook(hook_depth_head)
    model.encoder_m2.register_forward_hook(hook_encoder)
    model.aligner_m2.register_forward_hook(hook_aligner)

    # 2. 读取并校验 30 帧样本列表
    with open(SAMPLE_JSON, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"[INFO] 样本总数: {len(samples)} 帧 ({len(samples) * 4} 张图像)", flush=True)

    arms = ["raw_png", "q95_420", "q95_444"]
    result_stats = {arm: init_result_stat() for arm in arms}

    # 各维度统计累积器
    size_kb = {arm: [] for arm in arms}
    disk_subsampling_codes = []
    pix_metrics = {
        "q95_420": {"psnr": [], "ssim": [], "rmse_01": []},
        "q95_444": {"psnr": [], "ssim": [], "rmse_01": []},
    }
    feat_stats = {
        arm: {
            "lss_cos_global": [],
            "lss_rel_l2": [],
            "aligned_cos_global": [],
            "aligned_rel_l2": [],
            "aligned_cos_active": [],
            "aligned_rel_l2_active": [],
            "depth_kl_mean": [],
            "depth_kl_max": [],
            "cls_mae_all": [],
            "cls_maxae_all": [],
            "cls_mae_active": [],
            "cls_maxae_active": [],
            "anom_valid_arrays": [],
            "anom_active_arrays": [],
            "norm_rel_valid_arrays": [],
            "norm_rel_active_arrays": [],
        }
        for arm in ["q95_420", "q95_444"]
    }
    box_match_acc = {
        arm: {
            "n_png": 0,
            "n_cmp": 0,
            "n_matched": 0,
            "matched_ious": [],
            "center_err_cm": [],
        }
        for arm in ["q95_420", "q95_444"]
    }

    # 3. 逐帧执行配对推理
    for idx_sample, item in enumerate(samples):
        seq, cav_id, frame_id = item["seq"], item["cav_id"], item["frame_id"]
        yaml_path = os.path.join(JPG_VAL_ROOT, seq, cav_id, f"{frame_id}.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"缺少标注文件: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)

        pil_by_arm = {arm: [] for arm in arms}
        arr_png_list = []

        for cam_name in CAMS:
            png_path = os.path.join(PNG_ROOT, seq, cav_id, f"{frame_id}_{cam_name}.png")
            jpg_path = os.path.join(JPG_VAL_ROOT, seq, cav_id, f"{frame_id}_{cam_name}.jpg")
            if not os.path.exists(png_path):
                raise FileNotFoundError(f"缺少原始 PNG: {png_path}")
            if not os.path.exists(jpg_path):
                raise FileNotFoundError(f"缺少服务端 JPG: {jpg_path}")

            size_kb["raw_png"].append(os.path.getsize(png_path) / 1024.0)
            size_kb["q95_420"].append(os.path.getsize(jpg_path) / 1024.0)
            if idx_sample == 0:
                disk_subsampling_codes.append(detect_jpeg_subsampling(jpg_path))

            im_png = Image.open(png_path).convert("RGB")
            im_png.load()
            im_jpg420 = Image.open(jpg_path).convert("RGB")
            im_jpg420.load()
            im_jpg444, nbytes_444 = encode_decode_q95_444(im_png)
            size_kb["q95_444"].append(nbytes_444 / 1024.0)

            arr_png = np.asarray(im_png, dtype=np.uint8)
            arr_420 = np.asarray(im_jpg420, dtype=np.uint8)
            arr_444 = np.asarray(im_jpg444, dtype=np.uint8)
            arr_png_list.append(arr_png)

            for arm_name, arr_cmp in [("q95_420", arr_420), ("q95_444", arr_444)]:
                psnr, ssim, rmse_01 = calc_pixel_metrics(arr_png, arr_cmp)
                pix_metrics[arm_name]["psnr"].append(psnr)
                pix_metrics[arm_name]["ssim"].append(ssim)
                pix_metrics[arm_name]["rmse_01"].append(rmse_01)

            pil_by_arm["raw_png"].append(im_png)
            pil_by_arm["q95_420"].append(im_jpg420)
            pil_by_arm["q95_444"].append(im_jpg444)

        post_data_dict = build_gt_and_post_data_dict(params, post_processor, anchor_box_cpu)

        frame_outputs = {}
        with torch.no_grad():
            for arm in arms:
                hook_cache.clear()
                inp_dict = build_camera_input_dict(pil_by_arm[arm], params, data_aug_conf, device)
                model_out = model(inp_dict)
                cls_preds = model_out["cls_preds"].detach().cpu()
                reg_preds = model_out["reg_preds"].detach().cpu()
                dir_preds = model_out["dir_preds"].detach().cpu()

                pred_box, pred_score = post_processor.post_process(
                    post_data_dict,
                    {"ego": {"cls_preds": cls_preds, "reg_preds": reg_preds, "dir_preds": dir_preds}},
                )
                gt_box = post_processor.generate_gt_bbx(post_data_dict)

                for iou_th in [0.3, 0.5, 0.7]:
                    eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box, result_stats[arm], iou_th)

                frame_outputs[arm] = {
                    "depth_probs": hook_cache["depth_probs"],
                    "feat_lss": hook_cache["feat_lss"],
                    "feat_aligned": hook_cache["feat_aligned"],
                    "cls_probs": torch.sigmoid(cls_preds),
                    "pred_box": pred_box,
                    "pred_score": pred_score,
                }

        # 以 Raw PNG 为基准计算维度 2、3、4 差异
        ref = frame_outputs["raw_png"]
        ref_norm = torch.norm(ref["feat_aligned"][0], p=2, dim=0)
        q90_thresh = torch.quantile(ref_norm, 0.90)
        active_grid_mask = ref_norm >= q90_thresh
        active_cls_mask = ref["cls_probs"] > 0.2

        for arm in ["q95_420", "q95_444"]:
            cmp_out = frame_outputs[arm]
            lss_diff = compute_bev_feature_diff(ref["feat_lss"], cmp_out["feat_lss"])
            aligned_diff = compute_bev_feature_diff(
                ref["feat_aligned"], cmp_out["feat_aligned"], active_mask=active_grid_mask
            )

            p_ref = ref["depth_probs"]
            p_cmp = cmp_out["depth_probs"]
            kl_map = torch.sum(p_ref * (torch.log(p_ref + 1e-8) - torch.log(p_cmp + 1e-8)), dim=1)

            cls_abs_diff = torch.abs(cmp_out["cls_probs"] - ref["cls_probs"])

            st = feat_stats[arm]
            st["lss_cos_global"].append(lss_diff["cos_global"])
            st["lss_rel_l2"].append(lss_diff["rel_l2_global"])
            st["aligned_cos_global"].append(aligned_diff["cos_global"])
            st["aligned_rel_l2"].append(aligned_diff["rel_l2_global"])
            st["aligned_cos_active"].append(aligned_diff["cos_active_vec"])
            st["aligned_rel_l2_active"].append(aligned_diff["rel_l2_active"])
            st["depth_kl_mean"].append(float(kl_map.mean().item()))
            st["depth_kl_max"].append(float(kl_map.max().item()))
            st["cls_mae_all"].append(float(cls_abs_diff.mean().item()))
            st["cls_maxae_all"].append(float(cls_abs_diff.max().item()))
            if bool(active_cls_mask.any()):
                st["cls_mae_active"].append(float(cls_abs_diff[active_cls_mask].mean().item()))
                st["cls_maxae_active"].append(float(cls_abs_diff[active_cls_mask].max().item()))

            st["anom_valid_arrays"].append(aligned_diff["grid_anom_valid"])
            st["anom_active_arrays"].append(aligned_diff["grid_anom_active"])
            st["norm_rel_valid_arrays"].append(aligned_diff["grid_norm_rel_valid"])
            st["norm_rel_active_arrays"].append(aligned_diff["grid_norm_rel_active"])

            m = match_predictions_against_png(
                ref["pred_box"], ref["pred_score"], cmp_out["pred_box"], cmp_out["pred_score"], score_thresh=0.3
            )
            bm = box_match_acc[arm]
            bm["n_png"] += m["n_png"]
            bm["n_cmp"] += m["n_cmp"]
            bm["n_matched"] += m["n_matched"]
            bm["matched_ious"].extend(m["matched_ious"])
            bm["center_err_cm"].extend(m["center_err_cm"])

        if (idx_sample + 1) % 5 == 0 or idx_sample == 0 or (idx_sample + 1) == len(samples):
            print(
                f"[PROGRESS] {idx_sample + 1}/{len(samples)} 帧完成 "
                f"({seq}/{cav_id}/{frame_id}), 已耗时 {time.time() - t0:.1f}s",
                flush=True,
            )

    # 4. 汇总全部四个维度结果
    ap_summary = {}
    for arm in arms:
        ap_summary[arm] = {
            "ap30": safe_calculate_ap(result_stats[arm], 0.3),
            "ap50": safe_calculate_ap(result_stats[arm], 0.5),
            "ap70": safe_calculate_ap(result_stats[arm], 0.7),
            "gt_total_boxes": int(result_stats[arm][0.5]["gt"]),
            "pred_total_boxes": int(len(result_stats[arm][0.5]["tp"])),
        }

    avg_png_kb = float(np.mean(size_kb["raw_png"]))
    report = {
        "meta": {
            "num_frames": len(samples),
            "num_images": len(samples) * 4,
            "checkpoint": CKPT_PATH,
            "strict_load": str(load_res),
            "disk_jpg_subsampling_codes_frame0": disk_subsampling_codes,
            "elapsed_sec": round(time.time() - t0, 2),
            "fixed_padding_layers": fixed_layers,
        },
        "dimension_1_pixel_fidelity": {
            "raw_png": {
                "avg_size_kb": avg_png_kb,
                "compression_ratio_vs_png": 1.0,
            },
        },
        "dimension_2_bev_feature_consistency": {},
        "dimension_3_detection_consistency": {
            "ap_metrics": ap_summary,
            "relative_to_raw_png": {},
        },
        "dimension_4_security_sensitivity": {},
    }

    for arm in ["q95_420", "q95_444"]:
        avg_kb = float(np.mean(size_kb[arm]))
        report["dimension_1_pixel_fidelity"][arm] = {
            "avg_size_kb": avg_kb,
            "compression_ratio_vs_png": float(avg_png_kb / avg_kb),
            "psnr_db_mean": float(np.mean(pix_metrics[arm]["psnr"])),
            "psnr_db_min": float(np.min(pix_metrics[arm]["psnr"])),
            "ssim_mean": float(np.mean(pix_metrics[arm]["ssim"])),
            "ssim_min": float(np.min(pix_metrics[arm]["ssim"])),
            "rgb_rmse_01_mean": float(np.mean(pix_metrics[arm]["rmse_01"])),
            "rgb_rmse_01_max": float(np.max(pix_metrics[arm]["rmse_01"])),
        }

        st = feat_stats[arm]
        report["dimension_2_bev_feature_consistency"][arm] = {
            "feat_lss_cos_global_mean": float(np.mean(st["lss_cos_global"])),
            "feat_lss_cos_global_min": float(np.min(st["lss_cos_global"])),
            "feat_lss_rel_l2_mean": float(np.mean(st["lss_rel_l2"])),
            "feat_lss_rel_l2_max": float(np.max(st["lss_rel_l2"])),
            "feat_aligned_cos_global_mean": float(np.mean(st["aligned_cos_global"])),
            "feat_aligned_cos_global_min": float(np.min(st["aligned_cos_global"])),
            "feat_aligned_rel_l2_mean": float(np.mean(st["aligned_rel_l2"])),
            "feat_aligned_rel_l2_max": float(np.max(st["aligned_rel_l2"])),
            "feat_aligned_cos_active_mean": float(np.mean(st["aligned_cos_active"])),
            "feat_aligned_rel_l2_active_mean": float(np.mean(st["aligned_rel_l2_active"])),
            "depth_kl_mean": float(np.mean(st["depth_kl_mean"])),
            "depth_kl_max": float(np.max(st["depth_kl_max"])),
            "cls_sigmoid_mae_all": float(np.mean(st["cls_mae_all"])),
            "cls_sigmoid_maxae_all": float(np.max(st["cls_maxae_all"])),
            "cls_sigmoid_active_frames": len(st["cls_mae_active"]),
            "cls_sigmoid_mae_active": float(np.mean(st["cls_mae_active"])) if st["cls_mae_active"] else None,
            "cls_sigmoid_maxae_active": float(np.max(st["cls_maxae_active"])) if st["cls_maxae_active"] else None,
        }

        bm = box_match_acc[arm]
        report["dimension_3_detection_consistency"]["relative_to_raw_png"][arm] = {
            "delta_ap30": ap_summary[arm]["ap30"] - ap_summary["raw_png"]["ap30"],
            "delta_ap50": ap_summary[arm]["ap50"] - ap_summary["raw_png"]["ap50"],
            "delta_ap70": ap_summary[arm]["ap70"] - ap_summary["raw_png"]["ap70"],
            "score_gt_03_png_boxes": bm["n_png"],
            "score_gt_03_cmp_boxes": bm["n_cmp"],
            "score_gt_03_matched_iou05": bm["n_matched"],
            "match_rate_vs_png": float(bm["n_matched"] / bm["n_png"]) if bm["n_png"] > 0 else None,
            "matched_mean_iou": float(np.mean(bm["matched_ious"])) if bm["matched_ious"] else None,
            "center_offset_cm_mean": float(np.mean(bm["center_err_cm"])) if bm["center_err_cm"] else None,
            "center_offset_cm_p95": float(np.percentile(bm["center_err_cm"], 95)) if bm["center_err_cm"] else None,
        }

        anom_valid = np.concatenate(st["anom_valid_arrays"])
        anom_active = np.concatenate(st["anom_active_arrays"])
        norm_valid = np.concatenate(st["norm_rel_valid_arrays"])
        norm_active = np.concatenate(st["norm_rel_active_arrays"])

        report["dimension_4_security_sensitivity"][arm] = {
            "grid_cos_anomaly_valid": {
                **summarize_array(anom_valid),
                "total_valid_grids": int(anom_valid.size),
                "frac_gt_0.30_hetshield_default": float(np.mean(anom_valid > 0.30)),
                "frac_gt_0.10": float(np.mean(anom_valid > 0.10)),
                "frac_gt_0.05": float(np.mean(anom_valid > 0.05)),
            },
            "grid_cos_anomaly_active_top10pct": {
                **summarize_array(anom_active),
                "total_active_grids": int(anom_active.size),
                "frac_gt_0.30_hetshield_default": float(np.mean(anom_active > 0.30)),
                "frac_gt_0.10": float(np.mean(anom_active > 0.10)),
                "frac_gt_0.05": float(np.mean(anom_active > 0.05)),
            },
            "grid_norm_rel_delta_valid": summarize_array(norm_valid),
            "grid_norm_rel_delta_active_top10pct": summarize_array(norm_active),
        }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] 完整报告已保存: {OUT_JSON}", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
