# -*- coding: utf-8 -*-
"""
开放异构协同感知五维对抗威胁生成模块（严格落实防位置/尺寸捷径与白盒自适应评估准则）：
1. PGDRemovalAttack：针对目标前景区域的迭代梯度特征擦除攻击；
2. GhostSpoofingAttack：在道路可通行区随机采样无车锚点（覆盖 15m~70m 不同距离与随机朝向）的优化伪造与特征块移植伪造；
3. HetPoisonTransferAttack：无标签单步前向跨异构迁移扰动（支持在原生特征层注入，实测适配器 Lipschitz 放大效应）；
4. WhiteBoxAdaptiveAttack：完全知晓本防御全部模块（LB-FPA + 球面码本 + 非对称白化冲突 + MeanTopK 门控）的端到端白盒自适应攻击。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.uncertainty_fusion import fuse_protocol_standard


def sample_random_spoof_centers(
    z_ego: torch.Tensor,
    cls_head_ref: nn.Module,
    num_spoofs: int = 4,
    seed_offset: int = 0,
):
    """
    在道路通行区域（x in [-68m, +68m], y in [-16m, +16m]，即网格 h in [30, 70], w in [88, 264]）
    随机采样远离现有前景目标的无车空白网格坐标，彻底杜绝固定坐标或固定区域捷径。
    返回:
        centers: list of (h_idx, w_idx, anchor_rot_idx)
    """
    with torch.no_grad():
        fg_prob = torch.sigmoid(cls_head_ref(z_ego))[0].max(dim=0)[0]  # (H=100, W=352)
        # 膨胀现有前景区域 7x7（约 5.6m 安全缓冲），确保伪造点不与真车重叠
        occupied = F.max_pool2d(fg_prob.view(1, 1, 100, 352), kernel_size=9, stride=1, padding=4)[0, 0] > 0.18

    g = torch.Generator(device="cpu")
    g.manual_seed(20261118 + int(seed_offset))

    centers = []
    attempts = 0
    while len(centers) < num_spoofs and attempts < 80:
        attempts += 1
        h_c = int(torch.randint(34, 66, (1,), generator=g).item())
        w_c = int(torch.randint(120, 232, (1,), generator=g).item())
        # 排除距离本车过近 (<8m) 的原点区
        if abs(h_c - 50) < 6 and abs(w_c - 176) < 10:
            continue
        if bool(occupied[h_c, w_c].item()):
            continue
        if any(abs(h_c - hh) < 7 and abs(w_c - ww) < 10 for hh, ww, _ in centers):
            continue
        rot_idx = int(torch.randint(0, 2, (1,), generator=g).item())
        centers.append((h_c, w_c, rot_idx))
    return centers


def generate_pgd_removal_attack(
    z_all_clean: torch.Tensor,
    attacker_idx: int,
    cls_head_ref: nn.Module,
    eps: float = 3.5,
    steps: int = 10,
) -> torch.Tensor:
    """
    攻击 1：PGD 目标擦除攻击（针对标准协议自注意力融合，企图抹除场景内所有真实前景目标响应）。
    参数:
        z_all_clean: (1 + K, C, H, W)
        attacker_idx: 被攻陷协作者在 z_all 中的索引 (>=1)
        eps: L_inf 协议特征扰动上限
    返回:
        z_all_adv: 注入恶意擦除特征后的协议张量
    """
    z_adv = z_all_clean.clone().detach()
    z_orig_i = z_all_clean[attacker_idx : attacker_idx + 1].clone().detach()

    with torch.no_grad():
        # 以干净全协同高置信前景网格作为攻击者企图抹除的目标掩膜（无真值标签泄漏，仅用模型自身预测！）
        z_clean_fused = fuse_protocol_standard(z_all_clean, cls_head_ref=cls_head_ref)
        fg_score = torch.sigmoid(cls_head_ref(z_clean_fused)).max(dim=1, keepdim=True)[0]
        target_mask = (fg_score > 0.22).float()
        if target_mask.sum() < 1.0:
            target_mask = (fg_score >= fg_score.flatten().topk(32)[0][-1]).float()

    delta = torch.zeros_like(z_orig_i, requires_grad=True)
    step_lr = eps / max(steps * 0.6, 1.0)

    for _ in range(steps):
        z_cur_i = F.relu(z_orig_i + delta)
        z_stack = z_adv.clone()
        z_stack[attacker_idx : attacker_idx + 1] = z_cur_i

        fused = fuse_protocol_standard(z_stack, cls_head_ref=cls_head_ref)

        logits = cls_head_ref(fused)  # (1, 2, H, W)
        # 最小化前景目标处的分类概率 -> 等价于最小化 logits（梯度下降压灭前景框）
        loss = (torch.sigmoid(logits) * target_mask).sum() / (target_mask.sum() + 1e-4)
        grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
        with torch.no_grad():
            delta.add_(-step_lr * grad.sign())
            delta.clamp_(-eps, eps)
        del z_cur_i, z_stack, fused, logits, loss, grad

    z_adv[attacker_idx : attacker_idx + 1] = F.relu(z_orig_i + delta.detach())
    return z_adv


def generate_ghost_spoofing_attack(
    z_all_clean: torch.Tensor,
    attacker_idx: int,
    cls_head_ref: nn.Module,
    eps: float = 4.5,
    steps: int = 10,
    num_spoofs: int = 4,
    frame_seed: int = 0,
) -> torch.Tensor:
    """
    攻击 2：随机道路无车锚点伪造鬼影攻击（结合真实车辆特征块移植初始化 + 局部高置信伪造优化）。
    每帧动态随机采样不同距离与朝向的道路空白区，无固定位置/尺寸捷径。
    """
    z_adv = z_all_clean.clone().detach()
    z_orig_i = z_all_clean[attacker_idx : attacker_idx + 1].clone().detach()
    centers = sample_random_spoof_centers(z_all_clean[0:1], cls_head_ref, num_spoofs=num_spoofs, seed_offset=frame_seed)
    if not centers:
        return z_adv

    n_tot, c_dim, h_dim, w_dim = z_all_clean.shape

    # 步骤 A：若当前帧存在高置信真车特征块，先提取 5x5 真实前景块移植到伪造锚点作为流形内种子（Patch-Replay 初始化）
    z_seeded = z_orig_i.clone()
    with torch.no_grad():
        own_prob = torch.sigmoid(cls_head_ref(z_orig_i))[0].max(dim=0)[0]
        max_val, max_flat = own_prob.view(-1).max(dim=0)
        if float(max_val.item()) > 0.35:
            src_h = int(max_flat.item()) // w_dim
            src_w = int(max_flat.item()) % w_dim
            if 3 <= src_h < h_dim - 3 and 3 <= src_w < w_dim - 3:
                patch = z_orig_i[:, :, src_h - 2 : src_h + 3, src_w - 2 : src_w + 3].clone()
                for hc, wc, _ in centers:
                    z_seeded[:, :, hc - 2 : hc + 3, wc - 2 : wc + 3] = patch

    # 步骤 B：局部掩膜约束下的多步前景伪造增强
    spoof_mask = torch.zeros((1, 1, h_dim, w_dim), device=z_all_clean.device)
    target_cls = torch.zeros((1, 2, h_dim, w_dim), device=z_all_clean.device)
    for hc, wc, rot_idx in centers:
        spoof_mask[:, :, max(0, hc - 2) : min(h_dim, hc + 3), max(0, wc - 2) : min(w_dim, wc + 3)] = 1.0
        target_cls[:, rot_idx, max(0, hc - 1) : min(h_dim, hc + 2), max(0, wc - 1) : min(w_dim, wc + 2)] = 1.0

    delta = torch.zeros_like(z_seeded, requires_grad=True)
    step_lr = eps / max(steps * 0.55, 1.0)

    for _ in range(steps):
        z_cur_i = F.relu(z_seeded + delta * spoof_mask)
        z_stack = z_adv.clone()
        z_stack[attacker_idx : attacker_idx + 1] = z_cur_i

        fused = fuse_protocol_standard(z_stack, cls_head_ref=cls_head_ref)

        logits = cls_head_ref(fused)
        loss = F.binary_cross_entropy_with_logits(logits, target_cls, reduction="none")
        masked_loss = (loss * spoof_mask).sum() / (spoof_mask.sum() + 1e-4)
        grad = torch.autograd.grad(masked_loss, delta, retain_graph=False)[0]
        with torch.no_grad():
            delta.add_(-step_lr * grad.sign())
            delta.clamp_(-eps, eps)
        del z_cur_i, z_stack, fused, logits, loss, masked_loss, grad

    z_adv[attacker_idx : attacker_idx + 1] = F.relu(z_seeded + (delta.detach() * spoof_mask))
    return z_adv


def generate_hetpoison_raw_transfer_attack(
    raw_bev: torch.Tensor,
    surrogate_adapter: nn.Module,
    surrogate_cls_head: nn.Module,
    target_adapter: nn.Module,
    raw_eps: float = 1.8,
) -> torch.Tensor:
    """
    攻击 3：HetPoison-style 单步无标签跨异构迁移特征污染攻击（在发送端原生特征 raw_bev 层单次生成，
    再通过实际部署的 target_adapter（UnconstrainedAdapter 或 LB-FPA）进入协议空间）。
    无需真实标签、无需访问接收端检测头或防御门控，直接实证无约束适配器的扰动放大效应 vs LB-FPA 的有界抑制作用！
    """
    raw_in = raw_bev.clone().detach().requires_grad_(True)
    z_surr = surrogate_adapter(raw_in)
    logits_surr = surrogate_cls_head(z_surr)
    probs = torch.sigmoid(logits_surr)
    # 无标签单步自擦除 + 背景伪响应联合损失（无需 GT，直接反转高响应区并搅乱高能量通道）
    pseudo_fg = (probs.detach() > 0.20).float()
    loss_erase = (probs * pseudo_fg).sum() / (pseudo_fg.sum() + 1.0)
    loss_jam = -0.35 * probs[:, :, 30:70, 95:255].mean()
    total_obj = loss_erase + loss_jam
    grad = torch.autograd.grad(total_obj, raw_in)[0]

    with torch.no_grad():
        raw_adv = raw_bev - raw_eps * grad.sign()
        z_target_adv = target_adapter(raw_adv)
    return z_target_adv


def generate_whitebox_adaptive_attack(
    z_all_clean: torch.Tensor,
    attacker_idx: int,
    bud_module: nn.Module,
    sigma_alea_ego: torch.Tensor,
    u_geom_cavs: torch.Tensor,
    dist_maps_cavs: torch.Tensor,
    profile_names: list,
    cls_head_ref: nn.Module,
    eps: float = 3.5,
    steps: int = 12,
    stealth_lambda: float = 1.5,
    frame_seed: int = 0,
) -> torch.Tensor:
    """
    攻击 4：完全知晓本防御全套机制与门限的端到端白盒自适应潜伏攻击（White-Box Adaptive Attack）。
    攻击者同时对完整的 BUDFusionModule（含球面码本、非对称白化冲突与 MeanTopK 门控）反向传播，
    企图在压低 s_frame 异常度以逃避门控的同时擦除真目标并注入假目标。
    """
    z_adv = z_all_clean.clone().detach()
    z_orig_i = z_all_clean[attacker_idx : attacker_idx + 1].clone().detach()
    cav_sub_idx = attacker_idx - 1

    with torch.no_grad():
        fg_ego = torch.sigmoid(cls_head_ref(z_all_clean[0:1])).max(dim=1, keepdim=True)[0]
        erase_mask = (fg_ego > 0.25).float()

    centers = sample_random_spoof_centers(z_all_clean[0:1], cls_head_ref, num_spoofs=3, seed_offset=frame_seed + 77)
    spoof_mask = torch.zeros_like(erase_mask)
    for hc, wc, _ in centers:
        spoof_mask[:, :, max(0, hc - 2) : min(100, hc + 3), max(0, wc - 2) : min(352, wc + 3)] = 1.0

    delta = torch.zeros_like(z_orig_i, requires_grad=True)
    step_lr = eps / max(steps * 0.6, 1.0)

    ug_sub = u_geom_cavs[cav_sub_idx : cav_sub_idx + 1]
    dm_sub = dist_maps_cavs[cav_sub_idx : cav_sub_idx + 1]
    prof_sub = [profile_names[cav_sub_idx]]
    z_ego_det = z_all_clean[0:1].detach()

    for _ in range(steps):
        z_cur_i = F.relu(z_orig_i + delta)
        z_pair = torch.cat([z_ego_det, z_cur_i], dim=0)

        z_fused, _ = bud_module.forward_fuse(
            z_pair,
            sigma_alea_ego=sigma_alea_ego,
            u_geom_cavs=ug_sub,
            dist_maps_cavs=dm_sub,
            profile_names=prof_sub,
            cls_head_ref=cls_head_ref,
        )
        probs_fused = torch.sigmoid(cls_head_ref(z_fused)).max(dim=1, keepdim=True)[0]

        # 破坏目标：压低真目标概率 + 抬高随机伪造点概率
        dmg_erase = -(probs_fused * erase_mask).sum() / (erase_mask.sum() + 1.0)
        dmg_spoof = (probs_fused * spoof_mask).sum() / (spoof_mask.sum() + 1.0)
        atk_gain = dmg_erase + dmg_spoof

        # 潜伏逃逸正则：直接对完整的 DUID+UCR-Fusion 帧级异常度 s_frame 与局部白化冲突 d_cal 求精确白盒梯度！
        d_cal_i, epis_i, _, _, s_frame_i, _, _ = bud_module.compute_verification_maps(
            z_pair, sigma_alea_ego, ug_sub, dm_sub, prof_sub, cls_head_ref=cls_head_ref
        )
        stealth_pen = stealth_lambda * (s_frame_i.mean() + 0.5 * d_cal_i.mean() + 2.0 * epis_i.mean())

        obj = atk_gain - stealth_pen
        grad = torch.autograd.grad(obj, delta, retain_graph=False)[0]
        with torch.no_grad():
            delta.add_(step_lr * grad.sign())
            delta.clamp_(-eps, eps)
        del z_cur_i, z_pair, z_fused, probs_fused, d_cal_i, epis_i, s_frame_i, obj, grad

    z_adv[attacker_idx : attacker_idx + 1] = F.relu(z_orig_i + delta.detach())
    return z_adv
