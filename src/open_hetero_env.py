# -*- coding: utf-8 -*-
"""
开放异构协同感知场景构建与数据流封装环境（OpenHeteroEnv）：
1. 直接复用 OpenCOOD 标准数据集读取、真值 3D 框生成、Anchor 标签生成与旋转框 NMS 后处理；
2. 对本车（Ego, index=0）固定采用 Profile-A（64线原始点云 + 冻结预训练 PointPillars 锚点模型）；
3. 对外部协作者（index>=1）在协作者自身局部传感器坐标系下执行物理激光束环降采样（64/32/16线），
   再经刚体位姿矩阵 T_{i -> ego} 变换至协商参考系，支持分配任意异构配置（Profile-A / B / C / D）；
4. 提供冻结锚点特征提取、前景高斯权重掩膜生成及客观真值框点数分层统计功能。
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.point_pillar_intermediate import PointPillarIntermediate
from opencood.utils import box_utils
from opencood.utils.pcd_utils import mask_points_by_range, mask_ego_points

from src.hetero_sensors import (
    LIDAR_RANGE,
    PROTO_H,
    PROTO_W,
    GRID_RES,
    build_bev_grid_xy,
    downsample_lidar_by_beams,
    compute_collaborator_geom_prior,
    compute_target_layer_visibility_uncertainty,
    count_points_in_3d_boxes,
)


# 开放车联网协作者异构角色池（覆盖架构异构、体素分辨率异构与物理线束异构）
COLLAB_PROFILES = [
    "profile_b_second64",   # 64线 + SECOND 三维稀疏体素卷积骨干
    "profile_d_sparse16",   # 16线稀疏雷达 + 稀疏多感受野柱状骨干
    "profile_c_coarse32",   # 32线中配雷达 + 0.8m 粗粒度柱状骨干
]

PROFILE_BEAMS = {
    "profile_a_pp64": 64,
    "profile_b_second64": 64,
    "profile_c_coarse32": 32,
    "profile_d_sparse16": 16,
}


def assign_deterministic_profile(sample_idx: int, cav_rank: int, cav_id: str) -> str:
    """
    为外部协作者确定性分配异构画像（可复现、均衡覆盖所有异构类型，确保不同对比方法面对完全一致的异构车队）。
    """
    try:
        cid_num = abs(int(cav_id))
    except ValueError:
        cid_num = sum(ord(ch) for ch in str(cav_id))
    slot = (sample_idx + cav_rank + cid_num) % len(COLLAB_PROFILES)
    return COLLAB_PROFILES[slot]


def to_gpu_nested(data):
    if isinstance(data, dict):
        return {k: to_gpu_nested(v) for k, v in data.items()}
    if isinstance(data, list):
        return [to_gpu_nested(v) for v in data]
    if isinstance(data, torch.Tensor):
        return data.cuda(non_blocking=True)
    return data


def build_foreground_gaussian_mask(gt_box_center: np.ndarray, gt_mask: np.ndarray, device="cuda") -> torch.Tensor:
    """
    根据本车系下的真值 3D 框中心 (max_num, 7) [x, y, z, h, w, l, yaw] 或 [x, y, z, l, w, h, yaw]
    生成协议空间 (1, 1, 100, 352) 的前景高斯增强权重图 M_fg in [0, 1]（仅用于训练阶段！）。
    """
    fg_map = torch.zeros((1, 1, PROTO_H, PROTO_W), device=device, dtype=torch.float32)
    valid_boxes = gt_box_center[gt_mask > 0.5]
    if len(valid_boxes) == 0:
        return fg_map

    grid_x, grid_y = build_bev_grid_xy(device=device)
    for box in valid_boxes:
        bx, by = float(box[0]), float(box[1])
        dist_sq = (grid_x - bx) ** 2 + (grid_y - by) ** 2
        gauss = torch.exp(-dist_sq / (2.0 * (2.2 ** 2)))
        fg_map = torch.maximum(fg_map, gauss.view(1, 1, PROTO_H, PROTO_W))
    return fg_map


class FrozenAnchorWrapper(nn.Module):
    """
    包装并严格冻结预训练 64线 PointPillarIntermediate 基准模型：
    提供单车锚点协议特征提取 extract_anchor_z 与冻结检测头推理 predict_boxes_logits。
    """

    def __init__(self, cfg_path: str, ckpt_path: str, pre_processor):
        super().__init__()
        hypes = yaml_utils.load_yaml(cfg_path, None)
        self.hypes = hypes
        self.pre_processor = pre_processor
        self.model = PointPillarIntermediate(hypes["model"]["args"])
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state, strict=True)

        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        # 强制永远保持 eval 模式，确保冻结 BN 统计量与基准模型完全恒定
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def extract_anchor_z_from_np(self, pts_in_ego_np: np.ndarray, device="cuda") -> torch.Tensor:
        """从参考系下的 64线 numpy 点云提取 (1, 384, 100, 352) 冻结锚点协议特征"""
        if pts_in_ego_np.shape[0] < 8:
            pts_in_ego_np = np.array(
                [[0.0, 0.0, -1.0, 0.1], [5.0, 5.0, -1.0, 0.1]], dtype=np.float32
            )
        proc = self.pre_processor.preprocess(pts_in_ego_np)
        merged = self.pre_processor.collate_batch([proc])
        vf = torch.as_tensor(merged["voxel_features"], device=device, dtype=torch.float32)
        vc = torch.as_tensor(merged["voxel_coords"], device=device, dtype=torch.int32)
        vn = torch.as_tensor(merged["voxel_num_points"], device=device, dtype=torch.int32)
        # 防御 PillarVFE 内部 .squeeze() 在单一体素 V==1 时将 (1, 64) 压扁为一维张量的边界 Bug
        if vf.shape[0] < 2:
            vf = torch.cat([vf, torch.zeros((2, vf.shape[1], vf.shape[2]), device=device, dtype=vf.dtype)], dim=0)
            pad_c = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 1]], device=device, dtype=vc.dtype)
            vc = torch.cat([vc, pad_c], dim=0)
            vn = torch.cat([vn, torch.ones((2,), device=device, dtype=vn.dtype)], dim=0)
        bd = {
            "voxel_features": vf,
            "voxel_coords": vc,
            "voxel_num_points": vn,
            "record_len": torch.ones(1, dtype=torch.long, device=device),
        }
        bd = self.model.pillar_vfe(bd)
        bd = self.model.scatter(bd)
        bd = self.model.backbone(bd)
        return bd["spatial_features_2d"]  # (1, 384, 100, 352)

    def head_forward(self, z_fused: torch.Tensor) -> dict:
        return {
            "psm": self.model.cls_head(z_fused),
            "rm": self.model.reg_head(z_fused),
        }


def load_opv2v_split_dataset(cfg_path: str, split_dir: str, train_dir: str, is_train: bool = False):
    """加载指定划分的 OpenCOOD IntermediateFusionDataset（禁用随机几何翻转以确保多分支坐标对齐可控）"""
    hypes = yaml_utils.load_yaml(cfg_path, None)
    hypes["root_dir"] = train_dir
    hypes["validate_dir"] = split_dir
    if is_train:
        hypes["data_augment"] = []
    ds = build_dataset(hypes, visualize=False, train=False)
    # 若需读取指定目录而非默认 validate_dir，当 split_dir 为 train 时直接替换其检索路径
    if split_dir != hypes["validate_dir"]:
        hypes["validate_dir"] = split_dir
        ds = build_dataset(hypes, visualize=False, train=False)
    return ds, hypes


def prepare_hetero_frame_bundle(
    dataset,
    sample_idx: int,
    anchor_wrapper: FrozenAnchorWrapper,
    device: str = "cuda",
    include_teacher: bool = False,
):
    """
    解析单帧多智能体原始传感器数据，生成标准化的开放异构帧数据包：
    - ego_pts_ref_np / ego_pts_t: 本车 64线参考系点云
    - z_ego: (1, 384, 100, 352) 本车锚点协议特征
    - sigma_alea_ego: (1, 1, 100, 352) 本车极坐标目标层有效光束可见度偶然不确定性
    - cav_list: 每个外部协作者的字典（含 cav_id, profile, num_beams, pts_hetero_t, pts_full64_np,
                cav_xy, u_geom, dist_map, 以及可选的 z_teacher）
    - batch_for_post: 供 OpenCOOD 标准 post_process 与损失函数使用的完整字典
    """
    base_data_dict = dataset.retrieve_base_data(sample_idx, cur_ego_pose_flag=True)
    grid_x, grid_y = build_bev_grid_xy(device=device)

    ego_id = None
    ego_pose = None
    for cid, cdict in base_data_dict.items():
        if cdict["ego"]:
            ego_id = cid
            ego_pose = cdict["params"]["lidar_pose"]
            break

    object_stack = []
    object_id_stack = []
    cav_entries = []
    ego_pts_ref_np = None

    cav_rank = 0
    for cid, cdict in base_data_dict.items():
        pose_i = cdict["params"]["lidar_pose"]
        dist_ego = math.hypot(pose_i[0] - ego_pose[0], pose_i[1] - ego_pose[1])
        if dist_ego > 70.0:  # OpenCOOD 标准 COM_RANGE = 70m
            continue

        t_mat = cdict["params"]["transformation_matrix"]
        obj_center, obj_mask, obj_ids = dataset.post_processor.generate_object_center([cdict], ego_pose)
        valid_oc = obj_center[obj_mask == 1]
        if len(valid_oc) > 0:
            object_stack.append(valid_oc)
            object_id_stack += obj_ids

        # 在车辆自身局部激光雷达坐标系过滤自车外壳反射点
        raw_local = mask_ego_points(cdict["lidar_np"].copy())

        if cdict["ego"]:
            pts_ego = raw_local.copy()
            pts_ego[:, :3] = box_utils.project_points_by_matrix_torch(pts_ego[:, :3], t_mat)
            pts_ego = mask_points_by_range(pts_ego, LIDAR_RANGE)
            ego_pts_ref_np = pts_ego
        else:
            cav_rank += 1
            prof = assign_deterministic_profile(sample_idx, cav_rank, cid)
            n_beams = PROFILE_BEAMS[prof]

            # 1) 满 64线点云投影（仅用于训练期提取教师锚点或测试期客观统计协作者原始点数，不泄露给测试推理特征！）
            pts_64 = raw_local.copy()
            pts_64[:, :3] = box_utils.project_points_by_matrix_torch(pts_64[:, :3], t_mat)
            pts_64 = mask_points_by_range(pts_64, LIDAR_RANGE)
            if pts_64.shape[0] < 32:
                continue

            # 2) 在协作者自身局部传感器系下按物理激光环精确降采样，再投影至参考系
            local_sub = downsample_lidar_by_beams(raw_local, num_beams=n_beams, total_beams=64)
            pts_hetero = local_sub.copy()
            pts_hetero[:, :3] = box_utils.project_points_by_matrix_torch(pts_hetero[:, :3], t_mat)
            pts_hetero = mask_points_by_range(pts_hetero, LIDAR_RANGE)

            cav_xy = torch.tensor([float(t_mat[0, 3]), float(t_mat[1, 3])], device=device, dtype=torch.float32)
            u_geom = compute_collaborator_geom_prior(cav_xy, num_beams=n_beams, grid_x=grid_x, grid_y=grid_y)
            dist_map = torch.sqrt((grid_x - cav_xy[0]) ** 2 + (grid_y - cav_xy[1]) ** 2).unsqueeze(0)

            entry = {
                "cav_id": str(cid),
                "profile": prof,
                "num_beams": n_beams,
                "pts_hetero_t": torch.from_numpy(pts_hetero).to(device=device, dtype=torch.float32),
                "pts_full64_np": pts_64,
                "cav_xy": cav_xy,
                "u_geom": u_geom,          # (1, 100, 352)
                "dist_map": dist_map,      # (1, 100, 352)
            }
            if include_teacher:
                entry["z_teacher"] = anchor_wrapper.extract_anchor_z_from_np(pts_64, device=device)
                pts_hetero_t = entry["pts_hetero_t"]
                entry["u_alea_gt"] = compute_target_layer_visibility_uncertainty(
                    pts_hetero_t, cav_xy, num_beams=n_beams, grid_x=grid_x, grid_y=grid_y
                )
            cav_entries.append(entry)

    # 组装全局去重真值框与标签（与 IntermediateFusionDataset 完全一致）
    if len(object_id_stack) > 0 and len(object_stack) > 0:
        unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack_arr = np.vstack(object_stack)[unique_indices]
        unique_obj_ids = [object_id_stack[i] for i in unique_indices]
    else:
        object_stack_arr = np.zeros((0, 7), dtype=np.float32)
        unique_obj_ids = []

    max_num = dataset.params["postprocess"]["max_num"]
    object_bbx_center = np.zeros((max_num, 7), dtype=np.float32)
    mask = np.zeros((max_num,), dtype=np.float32)
    n_valid = min(object_stack_arr.shape[0], max_num)
    if n_valid > 0:
        object_bbx_center[:n_valid, :] = object_stack_arr[:n_valid, :]
        mask[:n_valid] = 1.0
        unique_obj_ids = unique_obj_ids[:n_valid]

    anchor_box = dataset.post_processor.generate_anchor_box()
    label_dict = dataset.post_processor.generate_label(
        gt_box_center=object_bbx_center, anchors=anchor_box, mask=mask
    )

    batch_for_post = {
        "ego": {
            "object_bbx_center": torch.from_numpy(object_bbx_center).unsqueeze(0).cuda(),
            "object_bbx_mask": torch.from_numpy(mask).unsqueeze(0).cuda(),
            "object_ids": unique_obj_ids,
            "anchor_box": torch.from_numpy(anchor_box).cuda(),
            "label_dict": dataset.post_processor.collate_batch([label_dict]),
            "transformation_matrix": torch.eye(4, dtype=torch.float32).cuda(),
        }
    }
    batch_for_post["ego"]["label_dict"] = to_gpu_nested(batch_for_post["ego"]["label_dict"])

    ego_pts_t = torch.from_numpy(ego_pts_ref_np).to(device=device, dtype=torch.float32)
    ego_xy = torch.zeros(2, device=device, dtype=torch.float32)
    sigma_alea_ego = compute_target_layer_visibility_uncertainty(
        ego_pts_t, ego_xy, num_beams=64, grid_x=grid_x, grid_y=grid_y
    ).unsqueeze(0)  # (1, 1, 100, 352)
    z_ego = anchor_wrapper.extract_anchor_z_from_np(ego_pts_ref_np, device=device)

    return {
        "sample_idx": sample_idx,
        "ego_id": str(ego_id),
        "ego_pts_np": ego_pts_ref_np,
        "z_ego": z_ego,
        "sigma_alea_ego": sigma_alea_ego,
        "cav_entries": cav_entries,
        "gt_center_np": object_bbx_center,
        "gt_mask_np": mask,
        "batch_for_post": batch_for_post,
    }
