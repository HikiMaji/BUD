# -*- coding: utf-8 -*-
"""
HEAL 官方底座可微封装与数据适配模块（研究点一 · HEAL m1+m2 主线）。

核心职责：
1. 强制校验运行环境（efficientnet_pytorch==0.7.0）及原生 17 层对称静态填充；
2. 动态加载 box_overlaps C++ 扩展并在内存中兼容 .jpg 图像路径（不改动 HEAL_ref 与数据盘）；
3. 封装官方 final_infer/net_epoch1.pth（本车 m1 64线 PointPillars，协作者 m2 Camera-LSS），
   保留官方 m2 ConvNeXt 对齐器与 [-102.4, 102.4]^2 协议网格 (64 x 256 x 256)；
4. 提供无 NaN 梯度的可微多尺度融合函数 weighted_fuse_safe 与协议层前向接口，
   既支持白盒梯度攻击，也为后续 BUD-Fusion / HetShield 预留 custom_fuse_fn 扩展接口。
"""

import copy
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter, OrderedDict
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torchvision

from src_v2_mm.heal_env_check import assert_effnet_symmetric_padding, assert_heal_env

PROJ_ROOT = "/root/autodl-tmp/proj_dir2_hetero_trust"
HEAL_ROOT = os.path.join(PROJ_ROOT, "third_party_v2_mm", "HEAL_ref")
BOX_OVERLAPS_SO = os.path.join(
    PROJ_ROOT,
    "third_party",
    "OpenCOOD",
    "opencood",
    "utils",
    "box_overlaps.cpython-38-x86_64-linux-gnu.so",
)
FINAL_INFER_DIR = os.path.join(PROJ_ROOT, "checkpoints_v2_mm", "heal_official", "final_infer")
DEFAULT_VAL_DIR = os.path.join(PROJ_ROOT, "data_links", "opv2v_val")
DEFAULT_IN_ORDER_JSON = os.path.join(PROJ_ROOT, "tmp_png_ab_test", "opv2v_val_4modality_in_order.json")

_ENV_INITIALIZED = False


def init_heal_runtime() -> dict:
    """初始化 HEAL 运行时：校验依赖版本、注入 box_overlaps 扩展及 .jpg 内存兼容补丁。"""
    global _ENV_INITIALIZED
    env_info = assert_heal_env("0.7.0")
    if _ENV_INITIALIZED:
        return env_info

    if "opencood.utils.box_overlaps" not in sys.modules:
        if not os.path.exists(BOX_OVERLAPS_SO):
            raise FileNotFoundError(f"未找到编译扩展：{BOX_OVERLAPS_SO}")
        spec = importlib.util.spec_from_file_location("opencood.utils.box_overlaps", BOX_OVERLAPS_SO)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules["opencood.utils.box_overlaps"] = mod

    if HEAL_ROOT not in sys.path:
        sys.path.insert(0, HEAL_ROOT)

    # 内存级图像后缀兼容：若磁盘已转码为 .jpg 而无 .png，则自动回退读取同名 .jpg
    from opencood.data_utils.datasets.basedataset.opv2v_basedataset import OPV2VBaseDataset

    if not getattr(OPV2VBaseDataset, "_jpg_fallback_patched", False):
        orig_find_camera_files = OPV2VBaseDataset.find_camera_files

        @staticmethod
        def _patched_find_camera_files(cav_path, timestamp, sensor="camera"):
            files = orig_find_camera_files(cav_path, timestamp, sensor=sensor)
            resolved = []
            for p in files:
                if os.path.exists(p):
                    resolved.append(p)
                else:
                    jpg_p = p[:-4] + ".jpg" if p.endswith(".png") else p
                    resolved.append(jpg_p if os.path.exists(jpg_p) else p)
            return resolved

        OPV2VBaseDataset.find_camera_files = _patched_find_camera_files
        OPV2VBaseDataset._jpg_fallback_patched = True

    _ENV_INITIALIZED = True
    return env_info


def _default_assignment_path_for_dir(data_dir: str) -> str:
    """根据数据目录名与路径哈希生成专属顺序模态分配 JSON 路径，避免 train 与 val 共用冲突。"""
    abs_dir = os.path.abspath(data_dir)
    if abs_dir == os.path.abspath(DEFAULT_VAL_DIR):
        return DEFAULT_IN_ORDER_JSON
    base_name = os.path.basename(abs_dir.rstrip(os.sep)) or "split"
    short_hash = hashlib.md5(abs_dir.encode("utf-8")).hexdigest()[:8]
    return os.path.join(PROJ_ROOT, "tmp_png_ab_test", f"assign_{base_name}_{short_hash}.json")


def ensure_in_order_assignment(data_dir: str, out_json_path: Optional[str] = None) -> str:
    """按 HEAL 官方 assign_modality_4_in_order 规则生成或复用序列模态分配 JSON。"""
    if out_json_path is None:
        out_json_path = _default_assignment_path_for_dir(data_dir)
    if os.path.exists(out_json_path) and os.path.getsize(out_json_path) > 10:
        return out_json_path

    assign = OrderedDict()
    for seq in sorted(os.listdir(data_dir)):
        seq_path = os.path.join(data_dir, seq)
        if not os.path.isdir(seq_path):
            continue
        cavs = sorted([c for c in os.listdir(seq_path) if os.path.isdir(os.path.join(seq_path, c))])
        if cavs and cavs[0] == "-1":
            cavs = cavs[1:] + cavs[:1]
        assign[seq] = OrderedDict()
        for j, cav_id in enumerate(cavs):
            assign[seq][cav_id] = f"m{j % 4 + 1}"

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(assign, f, indent=2)
    return out_json_path


def weighted_fuse_safe(
    x: torch.Tensor,
    score: torch.Tensor,
    record_len: torch.Tensor,
    affine_matrix: torch.Tensor,
    align_corners: bool = False,
) -> torch.Tensor:
    """
    安全可微版 PyramidFusion 多尺度加权融合函数。

    相比 HEAL_ref 原版 weighted_fuse，避免了对全零区域填充 -inf 后再用 where(isnan, 0)
    在反向传播时产生的 NaN 梯度；在前向数值上与原版保持严格一致。
    """
    from opencood.models.fuse_modules.fusion_in_one import regroup
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

    _, _, h, w = x.shape
    batch_size = affine_matrix.shape[0]
    split_x = regroup(x, record_len)
    split_score = regroup(score, record_len)
    out = []

    for b in range(batch_size):
        n_cav = int(record_len[b].item()) if torch.is_tensor(record_len[b]) else int(record_len[b])
        t_mat = affine_matrix[b, 0, :n_cav, :, :]
        feat_ego = warp_affine_simple(split_x[b], t_mat, (h, w), align_corners=align_corners)
        score_ego = warp_affine_simple(split_score[b], t_mat, (h, w), align_corners=align_corners)

        valid = score_ego > 0.0
        any_valid = valid.any(dim=0, keepdim=True)
        # 对无效位置以大负数掩蔽（而非 -inf），防止反向传播 exp/softmax 产生 0 * inf = NaN
        safe_logits = torch.where(valid, score_ego, torch.full_like(score_ego, -1e4))
        safe_logits = torch.where(any_valid, safe_logits, torch.zeros_like(safe_logits))
        weights = torch.softmax(safe_logits, dim=0) * any_valid.to(score_ego.dtype)
        out.append(torch.sum(feat_ego * weights, dim=0))

    return torch.stack(out, dim=0)


class HEALBaseWrapper(nn.Module):
    """HEAL 官方 m1(LiDAR)+m2(Camera) 异构底座封装器，支持传输层提取与可微融合前向。"""

    def __init__(
        self,
        cfg_path: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        data_dir: str = DEFAULT_VAL_DIR,
        assignment_path: Optional[str] = None,
        comm_range: float = 180.0,
        use_cav: int = 2,
        device: str = "cuda",
    ):
        super().__init__()
        self.env_info = init_heal_runtime()

        from opencood.hypes_yaml import yaml_utils
        from opencood.tools import train_utils
        from opencood.utils.common_utils import update_dict

        cfg_file = cfg_path or os.path.join(FINAL_INFER_DIR, "config.yaml")
        ckpt_file = ckpt_path or os.path.join(FINAL_INFER_DIR, "net_epoch1.pth")
        if assignment_path is None:
            assignment_path = ensure_in_order_assignment(data_dir, DEFAULT_IN_ORDER_JSON)

        hypes = yaml_utils.load_yaml(cfg_file, opt=None)
        # 主线仅启用 m1(本车 PointPillars) 与 m2(协作车 EfficientNet-LSS)
        for drop_m in ["m3", "m4"]:
            hypes["heter"]["modality_setting"].pop(drop_m, None)
            hypes["model"]["args"].pop(drop_m, None)
        hypes["heter"]["lidar_channels_dict"] = {}

        # 保持官方默认协议范围 [-102.4, 102.4]^2，使 heter_feature_2d 为 (N, 64, 256, 256)
        proto_range = [-102.4, -102.4, -3, 102.4, 102.4, 1]
        hypes = update_dict(hypes, {"cav_lidar_range": proto_range, "lidar_range": proto_range, "gt_range": proto_range})
        hypes = update_dict(
            hypes,
            {
                "mapping_dict": {"m1": "m1", "m2": "m2", "m3": "m2", "m4": "m2"},
                "ego_modality": "m1",
            },
        )
        hypes = yaml_utils.load_general_params(hypes)
        hypes["root_dir"] = data_dir
        hypes["validate_dir"] = data_dir
        hypes["test_dir"] = data_dir
        hypes["fusion"]["core_method"] = "intermediateheterinfer"
        hypes["use_cav"] = int(use_cav)
        hypes["comm_range"] = float(comm_range)
        hypes["input_source"] = ["lidar", "camera"]
        hypes["add_data_extension"] = []
        hypes["label_type"] = "lidar"
        hypes["heter"]["assignment_path"] = assignment_path
        self.hypes = hypes

        model = train_utils.create_model(hypes)
        state_all = torch.load(ckpt_file, map_location="cpu")
        model_sd = model.state_dict()
        matched = {k: v for k, v in state_all.items() if k in model_sd}
        load_res = model.load_state_dict(matched, strict=True)
        self.checked_pad_layers = assert_effnet_symmetric_padding(model, min_layers=5)

        # 关闭推理期无需深度的监督分支
        if hasattr(model, "depth_supervision_m2"):
            model.depth_supervision_m2 = False
        if hasattr(model, "encoder_m2"):
            model.encoder_m2.depth_supervision = False
            if hasattr(model.encoder_m2, "camencode"):
                model.encoder_m2.camencode.depth_supervision = False

        # 冻结底座全部参数与 BN 运行统计量，并避免 ReLU inplace 影响白盒求导
        for param in model.parameters():
            param.requires_grad_(False)
        for mod in model.modules():
            if isinstance(mod, nn.ReLU) and mod.inplace:
                mod.inplace = False

        self.device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device_obj).eval()
        self.load_result = str(load_res)

    def train(self, mode: bool = True):
        """强制始终处于 eval 模式，确保冻结底座 BN 统计量不被污染。"""
        super().train(False)
        self.model.eval()
        return self

    def build_dataset(
        self,
        data_dir: Optional[str] = None,
        use_cav: Optional[int] = None,
        visualize: bool = True,
        collab_modality: str = "m2",
        assignment_path: Optional[str] = None,
        max_cav: Optional[int] = None,
    ):
        """
        构建官方 IntermediateHeterinferFusionDataset。

        支持通过 collab_modality ('m1' / 'm2' / 'mixed') 精确切换车队构成：
        - 'm1'   : 全部协作者均为 64 线激光雷达 (m1)；
        - 'm2'   : 全部协作者均为环视摄像头 (m2)；
        - 'mixed': 本车为 m1，协作者按 m2, m1, m2, m1... 严格交替分配（适用于多车队混合主实验）。
        """
        from opencood.data_utils.datasets import build_dataset
        from opencood.utils.common_utils import update_dict

        if collab_modality not in ("m1", "m2", "mixed"):
            raise ValueError(f"当前主线仅支持 collab_modality='m1'/'m2'/'mixed'，收到：{collab_modality}")

        h_cfg = copy.deepcopy(self.hypes)
        target_dir = data_dir if data_dir is not None else h_cfg["validate_dir"]
        h_cfg["validate_dir"] = target_dir
        h_cfg["test_dir"] = target_dir

        if assignment_path is not None:
            h_cfg["heter"]["assignment_path"] = assignment_path
        elif data_dir is not None:
            h_cfg["heter"]["assignment_path"] = ensure_in_order_assignment(target_dir, None)

        if max_cav is not None:
            if "wild_setting" in h_cfg and isinstance(h_cfg["wild_setting"], dict):
                h_cfg["wild_setting"]["max_cav"] = int(max_cav)
            h_cfg["max_cav"] = int(max_cav)
            h_cfg = update_dict(h_cfg, {"max_cav": int(max_cav)})
        if use_cav is not None:
            h_cfg["use_cav"] = int(use_cav)

        if collab_modality == "m1":
            mapping = {"m1": "m1", "m2": "m1", "m3": "m1", "m4": "m1"}
        elif collab_modality == "m2":
            mapping = {"m1": "m1", "m2": "m2", "m3": "m2", "m4": "m2"}
        else:
            # 本车(j=0 -> m1) 固定 m1，协作者 j=1..5 按 m2, m1, m2, m1... 交替分配
            mapping = {"m1": "m1", "m2": "m2", "m3": "m1", "m4": "m2"}
        h_cfg["heter"]["mapping_dict"] = dict(mapping)
        h_cfg = update_dict(h_cfg, {"mapping_dict": dict(mapping)})

        ds = build_dataset(h_cfg, visualize=visualize, train=False)
        ds.use_hdf5 = False
        return ds

    @staticmethod
    def get_sample_sequences(dataset) -> List[str]:
        """返回数据集中每帧索引对应的父序列名列表。"""
        seq_names = [os.path.basename(p) for p in dataset.scenario_folders]
        sample_seqs: List[str] = []
        prev = 0
        for seq_idx, end_idx in enumerate(dataset.len_record):
            sample_seqs.extend([seq_names[seq_idx]] * (end_idx - prev))
            prev = end_idx
        return sample_seqs

    def build_cam_crop_mask(
        self,
        ref_map: torch.Tensor,
        agent_modality_list: List[str],
        edge_trim: int = 4,
    ) -> torch.Tensor:
        """
        构造与 HEAL_ref PyramidFusion.forward_collab 完全一致的相机有效区域掩膜。
        在 L0 (256x256) 下，crop_ratio=2.0 与 edge_trim=4 对应中心 [66:190, 66:190]。
        """
        cam_crop_info = self.model.cam_crop_info
        mask = torch.ones_like(ref_map, device=ref_map.device)
        if not cam_crop_info:
            return mask

        _, _, h, w = mask.shape
        for cam_modality, info in cam_crop_info.items():
            agent_bool = torch.tensor([x == cam_modality for x in agent_modality_list], dtype=torch.bool, device=ref_map.device)
            if not agent_bool.any():
                continue
            crop_h = h / info[f"crop_ratio_H_{cam_modality}"] - float(edge_trim)
            crop_w = w / info[f"crop_ratio_W_{cam_modality}"] - float(edge_trim)
            start_h = int(h // 2 - crop_h // 2)
            end_h = int(h // 2 + crop_h // 2)
            start_w = int(w // 2 - crop_w // 2)
            end_w = int(w // 2 + crop_w // 2)

            mask[agent_bool] = 0.0
            mask[agent_bool, :, start_h:end_h, start_w:end_w] = 1.0
        return mask

    @torch.no_grad()
    def extract_protocol_features(self, batch_ego: dict) -> dict:
        """
        冻结提取传输层协议特征 heter_feature_2d (N, 64, 256, 256) 及相对变换矩阵与有效区掩膜。
        """
        from opencood.utils.transformation_utils import normalize_pairwise_tfm

        agent_modality_list = list(batch_ego["agent_modality_list"])
        record_len = batch_ego["record_len"]
        affine_matrix = normalize_pairwise_tfm(
            batch_ego["pairwise_t_matrix"], self.model.H, self.model.W, self.model.fake_voxel_size
        )

        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        for modality_name in self.model.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feat = getattr(self.model, f"encoder_{modality_name}")(batch_ego, modality_name)
            feat = getattr(self.model, f"backbone_{modality_name}")({"spatial_features": feat})["spatial_features_2d"]
            feat = getattr(self.model, f"aligner_{modality_name}")(feat)
            if self.model.sensor_type_dict[modality_name] == "camera":
                _, _, h, w = feat.shape
                target_h = int(h * getattr(self.model, f"crop_ratio_H_{modality_name}"))
                target_w = int(w * getattr(self.model, f"crop_ratio_W_{modality_name}"))
                feat = torchvision.transforms.CenterCrop((target_h, target_w))(feat)
            modality_feature_dict[modality_name] = feat

        counting = {m: 0 for m in self.model.modality_name_list}
        feat_list = []
        for modality_name in agent_modality_list:
            feat_idx = counting[modality_name]
            feat_list.append(modality_feature_dict[modality_name][feat_idx])
            counting[modality_name] += 1

        heter_feature_2d = torch.stack(feat_list, dim=0)
        # 以 L0 边缘裁剪区域 [66:190, 66:190] 作为 m2 传输有效区，避开边界 2 格的官方屏蔽带
        ref_scalar = torch.empty((heter_feature_2d.shape[0], 1, heter_feature_2d.shape[2], heter_feature_2d.shape[3]),
                                 device=heter_feature_2d.device, dtype=heter_feature_2d.dtype)
        cam_valid_mask = self.build_cam_crop_mask(ref_scalar, agent_modality_list, edge_trim=4)

        return {
            "heter_feature_2d": heter_feature_2d,
            "affine_matrix": affine_matrix,
            "record_len": record_len,
            "agent_modality_list": agent_modality_list,
            "cam_valid_mask": cam_valid_mask,
        }

    def forward_from_protocol(
        self,
        heter_feature_2d: torch.Tensor,
        record_len: torch.Tensor,
        affine_matrix: torch.Tensor,
        agent_modality_list: List[str],
        custom_fuse_fn: Optional[Callable] = None,
    ) -> dict:
        """
        从传输层 heter_feature_2d 开始执行可微金字塔多尺度融合与检测头前向传播。
        支持外部传入 custom_fuse_fn(feature_list, score_list, record_len, affine_matrix, agent_modality_list, wrapper)
        以接入 BUD-Fusion / HetShield 等后置防御模块。
        """
        pyramid = self.model.pyramid_backbone
        feature_list = list(pyramid.get_multiscale_feature(heter_feature_2d))

        occ_map_list = []
        score_list = []
        for i in range(pyramid.num_levels):
            occ_map = getattr(pyramid, f"single_head_{i}")(feature_list[i])
            occ_map_list.append(occ_map)
            score = torch.sigmoid(occ_map) + 1e-4
            cam_crop_mask = self.build_cam_crop_mask(occ_map, agent_modality_list, edge_trim=4)
            score_list.append(score * cam_crop_mask)

        if custom_fuse_fn is not None:
            fused_feature_list, fusion_diag = custom_fuse_fn(
                feature_list=feature_list,
                score_list=score_list,
                occ_map_list=occ_map_list,
                record_len=record_len,
                affine_matrix=affine_matrix,
                agent_modality_list=agent_modality_list,
                wrapper=self,
            )
        else:
            fused_feature_list = [
                weighted_fuse_safe(
                    feature_list[i], score_list[i], record_len, affine_matrix, pyramid.align_corners
                )
                for i in range(pyramid.num_levels)
            ]
            fusion_diag = {}

        decoded = pyramid.decode_multiscale_feature(fused_feature_list)
        if self.model.shrink_flag:
            decoded = self.model.shrink_conv(decoded)

        output_dict = {
            "pyramid": "collab",
            "cls_preds": self.model.cls_head(decoded),
            "reg_preds": self.model.reg_head(decoded),
            "dir_preds": self.model.dir_head(decoded),
            "occ_single_list": occ_map_list,
            "pyramid_feats": feature_list,
            "decoded_feat": decoded,
            "fusion_diag": fusion_diag,
        }
        return output_dict

    def forward_single_agent(self, feat_single: torch.Tensor) -> dict:
        """单车协议特征独立走金字塔与检测头，用于 Ego-Only 基线与黑盒代理攻击。"""
        pyramid = self.model.pyramid_backbone
        feature_list = list(pyramid.get_multiscale_feature(feat_single))
        occ_map_list = [getattr(pyramid, f"single_head_{i}")(feature_list[i]) for i in range(pyramid.num_levels)]
        decoded = pyramid.decode_multiscale_feature(feature_list)
        if self.model.shrink_flag:
            decoded = self.model.shrink_conv(decoded)
        return {
            "cls_preds": self.model.cls_head(decoded),
            "reg_preds": self.model.reg_head(decoded),
            "dir_preds": self.model.dir_head(decoded),
            "occ_single_list": occ_map_list,
            "pyramid_feats": feature_list,
        }

    def decode_boxes(self, batch_data: dict, output_dict: dict, dataset):
        """使用官方 dataset.post_process 解码 NMS 后预测框、置信度与真值框。"""
        out_wrap = {"ego": {
            "cls_preds": output_dict["cls_preds"],
            "reg_preds": output_dict["reg_preds"],
            "dir_preds": output_dict["dir_preds"],
        }}
        pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, out_wrap)
        return pred_box_tensor, pred_score, gt_box_tensor
