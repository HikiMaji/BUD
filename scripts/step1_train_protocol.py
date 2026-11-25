# -*- coding: utf-8 -*-
"""
阶段一训练脚本（step1_train_protocol.py）：
在 OPV2V 官方训练集（20 条独立父序列）上训练开放异构编码分支与协议适配器：
1. 训练三类异构智能体分支的真二维算子谱有界适配器 LB-FPA（profile_b_second64, profile_c_coarse32, profile_d_sparse16）；
2. 训练同架构/同参数量的无约束对照适配器 UnconstrainedAdapter（Conv2d + BatchNorm2d + ReLU，复现 STAMP/HEAL 基线与消融）；
3. 训练独立随机种子的代理适配器（Surrogate Adapters），用于 HetPoison-style 无标签跨异构原生层迁移攻击测试；
4. 在验证集上实测并导出 LB-FPA vs UnconstrainedAdapter 的对齐余弦相似度、单分支检测 AP 及经验局部 Lipschitz 扰动放大率。
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OPENCOOD_ROOT = os.path.join(PROJECT_ROOT, "third_party", "OpenCOOD")
for p in [PROJECT_ROOT, OPENCOOD_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from opencood.loss.point_pillar_loss import PointPillarLoss
from opencood.utils import eval_utils

from src.open_hetero_env import (
    COLLAB_PROFILES,
    PROFILE_BEAMS,
    FrozenAnchorWrapper,
    load_opv2v_split_dataset,
    prepare_hetero_frame_bundle,
    build_foreground_gaussian_mask,
    to_gpu_nested,
)
from src.protocol_models import HeteroAgentBranch
from src.hetero_sensors import downsample_lidar_by_beams
from opencood.utils import box_utils
from opencood.utils.pcd_utils import mask_points_by_range, mask_ego_points


CFG_PATH = "/root/autodl-tmp/ToolV2X_resources/cad_native_2026_10_05/AdvCollaborativePerception/models/OpenCOOD/pointpillar_attentive_fusion/config.yaml"
CKPT_PATH = "/root/autodl-tmp/ToolV2X_resources/cad_native_2026_10_05/AdvCollaborativePerception/models/OpenCOOD/pointpillar_attentive_fusion/latest.pth"


def set_seed(seed: int = 2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_training_instances(dataset, anchor_wrapper: FrozenAnchorWrapper, max_frames: int = 480, device="cuda"):
    """
    从 OPV2V 训练集的全部 20 条序列中均匀采样代表帧，预提取本车/协作者原始局部点云及 64线冻结锚点特征标签缓存到内存，
    使三组异构分支（LB-FPA、Unconstrained、Surrogate）共享完全一致的训练样本且加速 GPU 迭代。
    """
    total_len = len(dataset)
    stride = max(1, total_len // max_frames)
    frame_indices = list(range(0, total_len, stride))[:max_frames]
    print(f"[Data] Sampling {len(frame_indices)} frames evenly across {total_len} OPV2V train frames...")

    samples = []
    t0 = time.time()
    for count, idx in enumerate(frame_indices):
        base_data = dataset.retrieve_base_data(idx, cur_ego_pose_flag=True)
        ego_pose = None
        for _, cdict in base_data.items():
            if cdict["ego"]:
                ego_pose = cdict["params"]["lidar_pose"]
                break

        # 选取本车自身点云（在自身坐标系即 t_mat=I）及最近的一辆有效协作车构造多样视角训练对
        selected_cars = []
        for cid, cdict in base_data.items():
            pose_i = cdict["params"]["lidar_pose"]
            dist = ((pose_i[0] - ego_pose[0]) ** 2 + (pose_i[1] - ego_pose[1]) ** 2) ** 0.5
            if dist <= 65.0:
                selected_cars.append((cid, cdict))
            if len(selected_cars) >= 2:
                break

        # 生成当前帧在 ego 系下的全局检测标签用于冻结检测头任务监督
        obj_stack, obj_ids = [], []
        for _, cdict in base_data.items():
            oc, om, oids = dataset.post_processor.generate_object_center([cdict], ego_pose)
            voc = oc[om == 1]
            if len(voc) > 0:
                obj_stack.append(voc)
                obj_ids += oids
        if len(obj_ids) > 0 and len(obj_stack) > 0:
            uniq = [obj_ids.index(x) for x in set(obj_ids)]
            obj_arr = np.vstack(obj_stack)[uniq]
        else:
            obj_arr = np.zeros((0, 7), dtype=np.float32)

        max_num = dataset.params["postprocess"]["max_num"]
        gt_center = np.zeros((max_num, 7), dtype=np.float32)
        gt_mask = np.zeros((max_num,), dtype=np.float32)
        nv = min(len(obj_arr), max_num)
        if nv > 0:
            gt_center[:nv] = obj_arr[:nv]
            gt_mask[:nv] = 1.0

        anchor_box = dataset.post_processor.generate_anchor_box()
        label_dict = dataset.post_processor.generate_label(gt_box_center=gt_center, anchors=anchor_box, mask=gt_mask)
        collated_label = dataset.post_processor.collate_batch([label_dict])

        for _, cdict in selected_cars[:1]:  # 每帧取 ego 视角（完整单车标签完美对齐）避免跨车单车盲区标签噪声
            raw_local = mask_ego_points(cdict["lidar_np"].copy())
            t_mat = cdict["params"]["transformation_matrix"]
            pts_64 = raw_local.copy()
            pts_64[:, :3] = box_utils.project_points_by_matrix_torch(pts_64[:, :3], t_mat)
            pts_64 = mask_points_by_range(pts_64, [-140.8, -40.0, -3.0, 140.8, 40.0, 1.0])
            if pts_64.shape[0] < 64:
                continue

            z_teacher = anchor_wrapper.extract_anchor_z_from_np(pts_64, device=device).cpu().half()
            fg_mask = build_foreground_gaussian_mask(gt_center, gt_mask, device=device).cpu().half()

            # 预先按 64 / 32 / 16 线生成局部降采样并投影的点云
            pts_by_beam = {}
            for nb in [64, 32, 16]:
                sub_l = downsample_lidar_by_beams(raw_local, num_beams=nb, total_beams=64)
                p_ref = sub_l.copy()
                p_ref[:, :3] = box_utils.project_points_by_matrix_torch(p_ref[:, :3], t_mat)
                p_ref = mask_points_by_range(p_ref, [-140.8, -40.0, -3.0, 140.8, 40.0, 1.0])
                pts_by_beam[nb] = p_ref.astype(np.float32)

            samples.append({
                "pts_by_beam": pts_by_beam,
                "z_teacher": z_teacher,
                "fg_mask": fg_mask,
                "label_dict": collated_label,
            })

        if (count + 1) % 100 == 0 or (count + 1) == len(frame_indices):
            print(f"  Cached {count + 1}/{len(frame_indices)} frames ({len(samples)} instances) in {time.time() - t0:.1f}s")

    return samples


def compute_alignment_and_task_loss(
    z_pred: torch.Tensor,
    z_teacher: torch.Tensor,
    fg_mask: torch.Tensor,
    label_dict_gpu: dict,
    anchor_wrapper: FrozenAnchorWrapper,
    det_criterion: PointPillarLoss,
):
    """
    前景独立归一化协议蒸馏 + 冻结检测头函数级响应对齐 + 真实标签检测损失：
    解决 99% 空背景对前景车辆特征学习的稀释问题，使异构编码器快速具备强盲区目标表征能力。
    """
    fg_region = (fg_mask > 0.08).float()
    bg_region = 1.0 - fg_region

    l1_map = torch.abs(z_pred - z_teacher).mean(dim=1, keepdim=True)
    loss_fg_l1 = (l1_map * fg_region).sum() / (fg_region.sum() + 1.0)
    loss_bg_l1 = (l1_map * bg_region).sum() / (bg_region.sum() + 1.0)

    cos_map = 1.0 - F.cosine_similarity(z_pred, z_teacher, dim=1).unsqueeze(1)
    loss_fg_cos = (cos_map * fg_region).sum() / (fg_region.sum() + 1.0)
    loss_bg_cos = (cos_map * bg_region).sum() / (bg_region.sum() + 1.0)

    # 冻结检测头函数级蒸馏（直接对齐分类置信度与前景框回归输出）
    head_pred = anchor_wrapper.head_forward(z_pred)
    with torch.no_grad():
        head_teach = anchor_wrapper.head_forward(z_teacher)
        teach_prob = torch.sigmoid(head_teach["psm"])
        teach_fg_anchor = (teach_prob.max(dim=1, keepdim=True)[0] > 0.15).float()

    cls_distill = F.binary_cross_entropy_with_logits(head_pred["psm"], teach_prob, reduction="none")
    loss_cls_kd = (cls_distill * (1.0 + 12.0 * teach_fg_anchor)).mean()
    reg_distill = F.smooth_l1_loss(head_pred["rm"], head_teach["rm"], reduction="none").mean(dim=1, keepdim=True)
    loss_reg_kd = (reg_distill * (fg_region + teach_fg_anchor).clamp(max=1.0)).sum() / (
        (fg_region + teach_fg_anchor).clamp(max=1.0).sum() + 1.0
    )

    loss_det = det_criterion(head_pred, label_dict_gpu)

    total = (
        4.0 * loss_fg_l1
        + 1.5 * loss_bg_l1
        + 2.0 * loss_fg_cos
        + 0.5 * loss_bg_cos
        + 2.0 * loss_cls_kd
        + 1.5 * loss_reg_kd
        + 0.35 * loss_det
    )
    cos_fg_val = 1.0 - float(loss_fg_cos.item())
    return total, float(loss_fg_l1.item()), cos_fg_val, float(loss_det.item())


def train_single_branch(
    profile_name: str,
    bounded: bool,
    train_samples: list,
    anchor_wrapper: FrozenAnchorWrapper,
    det_criterion: PointPillarLoss,
    epochs: int = 8,
    lr: float = 1.5e-3,
    lip_gamma: float = 1.08,
    seed: int = 2026,
    tag: str = "main",
) -> HeteroAgentBranch:
    set_seed(seed)
    branch = HeteroAgentBranch(profile_name=profile_name, bounded=bounded, lip_gamma=lip_gamma).cuda()
    branch.train()

    optimizer = torch.optim.AdamW(branch.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = max(1, (epochs * len(train_samples) + 1) // 2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4)
    nb = PROFILE_BEAMS[profile_name]

    mode_str = f"Bounded(LB-FPA, gamma={lip_gamma})" if bounded else "Unconstrained(Conv+BN)"
    print(f"\n>>> Training [{tag}] {profile_name} | {mode_str} | samples={len(train_samples)}, epochs={epochs}")

    for ep in range(1, epochs + 1):
        t_ep = time.time()
        order = list(range(len(train_samples)))
        random.shuffle(order)
        ep_l1, ep_cos, ep_det = [], [], []

        optimizer.zero_grad()
        for step_i, s_idx in enumerate(order):
            item = train_samples[s_idx]
            pts_t = torch.from_numpy(item["pts_by_beam"][nb]).cuda(non_blocking=True)
            z_teach = item["z_teacher"].cuda(non_blocking=True).float()
            fg_m = item["fg_mask"].cuda(non_blocking=True).float()
            lbl_gpu = to_gpu_nested(item["label_dict"])

            z_pred, _ = branch(pts_t)
            loss, v_l1, v_cos, v_det = compute_alignment_and_task_loss(
                z_pred, z_teach, fg_m, lbl_gpu, anchor_wrapper, det_criterion
            )
            (loss / 2.0).backward()

            if (step_i + 1) % 2 == 0 or (step_i + 1) == len(order):
                torch.nn.utils.clip_grad_norm_(branch.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            ep_l1.append(v_l1)
            ep_cos.append(v_cos)
            ep_det.append(v_det)

        print(
            f"  Epoch {ep:02d}/{epochs:02d} ({time.time() - t_ep:.1f}s) | "
            f"FgL1={np.mean(ep_l1):.4f} | FgCosSim={np.mean(ep_cos):.4f} | DetLoss={np.mean(ep_det):.4f}"
        )

    # 在纯训练样本子集上精确校准 BatchNorm 运行统计量（解决 batch_size=1 下 BN 均值方差漂移，保障所有基线满血公平）
    with torch.no_grad():
        branch.train()
        for m in branch.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.reset_running_stats()
                m.momentum = None  # 累积精确均值
        calib_n = min(len(train_samples), 64)
        for s_idx in range(calib_n):
            pts_t = torch.from_numpy(train_samples[s_idx]["pts_by_beam"][nb]).cuda(non_blocking=True)
            _ = branch(pts_t)
    branch.eval()
    return branch


def evaluate_branch_and_lipschitz(branch: HeteroAgentBranch, val_ds, anchor_wrapper: FrozenAnchorWrapper, num_eval: int = 80):
    """在验证集子集上实测单分支独立检测 AP@0.5、对齐余弦相似度及经验局部 Lipschitz 放大率"""
    branch.eval()
    nb = PROFILE_BEAMS[branch.profile_name]
    stat = {0.5: {"tp": [], "fp": [], "gt": 0, "score": []}, 0.7: {"tp": [], "fp": [], "gt": 0, "score": []}}
    cos_list = []
    emp_lip_ratios = []

    stride = max(1, len(val_ds) // num_eval)
    indices = list(range(0, len(val_ds), stride))[:num_eval]

    for idx in indices:
        bundle = prepare_hetero_frame_bundle(val_ds, idx, anchor_wrapper, include_teacher=False)
        raw_ego = bundle["ego_pts_np"]
        # 用本车视角做单车独立检测精度与对抗扰动放大率评估
        sub_pts = downsample_lidar_by_beams(raw_ego, num_beams=nb, total_beams=64)
        pts_t = torch.from_numpy(sub_pts.astype(np.float32)).cuda()

        with torch.no_grad():
            z_pred, raw_bev = branch(pts_t)
            cos_v = F.cosine_similarity(z_pred, bundle["z_ego"], dim=1).mean().item()
            cos_list.append(cos_v)
            out = anchor_wrapper.head_forward(z_pred)
            pb, ps, gb = val_ds.post_process(bundle["batch_for_post"], {"ego": out})
            eval_utils.caluclate_tp_fp(pb, ps, gb, stat, 0.5)
            eval_utils.caluclate_tp_fp(pb, ps, gb, stat, 0.7)

        # 5步幂法寻找最坏对抗扰动方向，实测适配器经验局部 Lipschitz 常数 ||A(F+d) - A(F)||_2 / ||d||_2
        delta = (0.25 * torch.randn_like(raw_bev)).detach().requires_grad_(True)
        for _ in range(3):
            z_pert = branch.adapt_to_protocol(raw_bev.detach() + delta)
            diff_norm = (z_pert - z_pred.detach()).norm()
            g = torch.autograd.grad(diff_norm, delta)[0]
            delta = (0.25 * g / (g.norm() + 1e-8)).detach().requires_grad_(True)
        with torch.no_grad():
            z_pert = branch.adapt_to_protocol(raw_bev + delta)
            lip_ratio = float(((z_pert - z_pred).norm() / (delta.norm() + 1e-8)).item())
            emp_lip_ratios.append(lip_ratio)

    ap50 = float(eval_utils.calculate_ap(stat, 0.5, True)[0])
    ap70 = float(eval_utils.calculate_ap(stat, 0.7, True)[0])
    theo_lip = branch.adapter.theoretical_lipschitz_bound() if branch.bounded else None
    return {
        "standalone_ap50": round(ap50, 4),
        "standalone_ap70": round(ap70, 4),
        "mean_cos_sim": round(float(np.mean(cos_list)), 4),
        "empirical_lip_mean": round(float(np.mean(emp_lip_ratios)), 4),
        "empirical_lip_max": round(float(np.max(emp_lip_ratios)), 4),
        "theoretical_lip_bound": round(theo_lip, 4) if theo_lip is not None else "Unbounded",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_frames", type=int, default=420)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1.6e-3)
    parser.add_argument("--lip_gamma", type=float, default=1.12)
    parser.add_argument("--val_eval_frames", type=int, default=80)
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    train_dir = os.path.join(PROJECT_ROOT, "data_links", "opv2v_train")
    val_dir = os.path.join(PROJECT_ROOT, "data_links", "opv2v_val")

    train_ds, hypes = load_opv2v_split_dataset(CFG_PATH, train_dir, train_dir, is_train=True)
    val_ds, _ = load_opv2v_split_dataset(CFG_PATH, val_dir, train_dir, is_train=False)

    anchor_wrapper = FrozenAnchorWrapper(CFG_PATH, CKPT_PATH, train_ds.pre_processor).cuda()
    det_criterion = PointPillarLoss(hypes["loss"]["args"]).cuda()

    train_samples = collect_training_instances(train_ds, anchor_wrapper, max_frames=args.max_frames)
    surr_samples = train_samples[::2]  # 代理模型仅使用半数子集与独立种子训练，模拟攻击者本地代理

    summary_report = {}

    for prof in COLLAB_PROFILES:
        # 1. 训练本文谱有界适配器分支 (LB-FPA)
        b_bounded = train_single_branch(
            prof, True, train_samples, anchor_wrapper, det_criterion,
            epochs=args.epochs, lr=args.lr, lip_gamma=args.lip_gamma, seed=2026, tag="LB-FPA"
        )
        torch.save(b_bounded.state_dict(), os.path.join(out_dir, f"{prof}_bounded.pth"))
        metrics_b = evaluate_branch_and_lipschitz(b_bounded, val_ds, anchor_wrapper, num_eval=args.val_eval_frames)
        print(f"  [Eval LB-FPA {prof}] {metrics_b}")

        # 2. 训练同架构无约束对照适配器分支 (Unconstrained Conv+BN)
        b_unconst = train_single_branch(
            prof, False, train_samples, anchor_wrapper, det_criterion,
            epochs=args.epochs, lr=args.lr, seed=2026, tag="Unconstrained"
        )
        torch.save(b_unconst.state_dict(), os.path.join(out_dir, f"{prof}_unconstrained.pth"))
        metrics_u = evaluate_branch_and_lipschitz(b_unconst, val_ds, anchor_wrapper, num_eval=args.val_eval_frames)
        print(f"  [Eval Unconstrained {prof}] {metrics_u}")

        # 3. 训练用于 HetPoison 跨异构迁移攻击的独立代理适配器 (Surrogate)
        b_surr = train_single_branch(
            prof, False, surr_samples, anchor_wrapper, det_criterion,
            epochs=max(4, args.epochs - 2), lr=args.lr, seed=9999, tag="Surrogate"
        )
        torch.save(b_surr.state_dict(), os.path.join(out_dir, f"{prof}_surrogate.pth"))

        summary_report[prof] = {
            "LB_FPA": metrics_b,
            "Unconstrained": metrics_u,
        }

    report_path = os.path.join(PROJECT_ROOT, "outputs", "step1_protocol_alignment_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    print(f"\n[Step 1 Complete] Saved checkpoints & alignment/Lipschitz report to {report_path}")


if __name__ == "__main__":
    main()
