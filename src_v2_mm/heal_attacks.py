# -*- coding: utf-8 -*-
"""
HEAL 传输层（heter_feature_2d, 64x256x256）对抗攻击实现模块。

严格约束：
1. 所有攻击扰动仅施加于发送端协作车（m1 或 m2）在传输层 heter_feature_2d 上的通道；
2. 扰动通过有效区域掩膜严格限制在合法传输视野或指定局部区域内，
   绝不对实数域特征施加人工 ReLU 截断；
3. 支持两类参考全局攻击、小预算攻击、局部精准攻击与专项边界压测：
   - 全局参考攻击: generate_pgd_removal_heal / generate_ghost_spoofing_heal / generate_surrogate_transfer_removal_heal
   - 小预算按尺度攻击 (0.5σ / 1σ / 2σ): generate_scaled_pgd_removal_heal
   - 局部精准攻击 (仅改 1~2 个目标邻域，其余不动): generate_local_removal_heal / generate_local_ghost_spoofing_heal
   - 专项压力与边界攻击: generate_local_near0_6m_attack_heal / generate_local_occluded_ghost_heal
"""

import math
import torch
import torch.nn.functional as F


def _warp_mask(mask: torch.Tensor, affine_matrix: torch.Tensor, dst_idx: int, src_idx: int) -> torch.Tensor:
    """将 src_idx 坐标系下的掩膜 (1, 1, H, W) 精确变换至 dst_idx 坐标系。"""
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

    h, w = mask.shape[-2:]
    t_mat = affine_matrix[0, dst_idx, src_idx : src_idx + 1, :, :]
    warped = warp_affine_simple(mask, t_mat, (h, w), align_corners=False)
    return (warped > 0.5).float()


def compute_feature_scale_sigma(feat_slice: torch.Tensor, valid_mask: torch.Tensor) -> float:
    """计算协作者传输层特征在有效空间区域内的逐元素标准差 sigma。"""
    mask_bool = valid_mask[0, 0] > 0.5
    if mask_bool.any():
        vals = feat_slice[0, :, mask_bool]
        return max(float(vals.std().item()), 1e-4)
    return max(float(feat_slice.std().item()), 1e-4)


def _sample_ghost_anchors_in_cov(
    cls_prob_ego: torch.Tensor,
    cov_in_ego: torch.Tensor,
    num_spoofs: int = 6,
    frame_seed: int = 0,
):
    """
    在协作者投影覆盖区与无真车空白区的交集中，确定性随机采样互不重叠的伪造锚点 (h, w, anchor_idx)。
    优先选取本车中心视野 [-51.2, 51.2]^2 ([68:188, 68:188]) 区域，避免边界裁切与固定位置捷径。
    """
    _, _, h, w = cls_prob_ego.shape
    occupied = F.max_pool2d(cls_prob_ego, kernel_size=9, stride=1, padding=4)[0, 0] > 0.16

    center_zone = torch.zeros_like(occupied)
    center_zone[68:188, 68:188] = True
    cov_bool = cov_in_ego[0, 0] > 0.5

    cand_mask = cov_bool & center_zone & (~occupied)
    cand_coords = torch.nonzero(cand_mask, as_tuple=False)
    if cand_coords.shape[0] < 32:
        cand_mask = cov_bool & (~occupied)
        cand_coords = torch.nonzero(cand_mask, as_tuple=False)
    if cand_coords.shape[0] == 0:
        cand_coords = torch.nonzero(~occupied, as_tuple=False)

    dist_center = torch.sqrt((cand_coords[:, 0].float() - h * 0.5) ** 2 + (cand_coords[:, 1].float() - w * 0.5) ** 2)
    cand_far = cand_coords[dist_center >= 8.0]
    if cand_far.shape[0] >= 16:
        cand_coords = cand_far

    g = torch.Generator(device="cpu")
    g.manual_seed(20261120 + int(frame_seed))
    perm = torch.randperm(cand_coords.shape[0], generator=g)

    selected = []
    for idx in perm.tolist():
        hc = int(cand_coords[idx, 0].item())
        wc = int(cand_coords[idx, 1].item())
        if any(abs(hc - hh) < 7 and abs(wc - ww) < 7 for hh, ww, _ in selected):
            continue
        rot_idx = int(torch.randint(0, 2, (1,), generator=g).item())
        selected.append((hc, wc, rot_idx))
        if len(selected) >= num_spoofs:
            break
    return selected


def _sample_local_real_targets_in_cov(
    cls_prob_ego: torch.Tensor,
    cov_in_ego: torch.Tensor,
    num_targets: int = 2,
):
    """从本车与协作者共视区选取置信度最高的 1~2 个离散车辆中心网格 (h, w)。"""
    score_in_cov = cls_prob_ego * cov_in_ego
    # 先做 9x9 非极大抑制，避免在同一辆车上重复选中相邻像素
    local_max = F.max_pool2d(score_in_cov, kernel_size=9, stride=1, padding=4)
    peaks = ((score_in_cov == local_max) & (score_in_cov > 0.25))[0, 0]
    coords = torch.nonzero(peaks, as_tuple=False)

    if coords.shape[0] == 0:
        # 兜底取共视区最高概率点
        flat_idx = int(torch.argmax(score_in_cov).item())
        w_dim = score_in_cov.shape[3]
        return [(flat_idx // w_dim, flat_idx % w_dim)]

    peak_vals = score_in_cov[0, 0, coords[:, 0], coords[:, 1]]
    order = torch.argsort(peak_vals, descending=True)
    selected = []
    for idx in order.tolist():
        hc = int(coords[idx, 0].item())
        wc = int(coords[idx, 1].item())
        if any(abs(hc - hh) < 7 and abs(wc - ww) < 7 for hh, ww in selected):
            continue
        selected.append((hc, wc))
        if len(selected) >= num_targets:
            break
    return selected


def _build_patch_mask(shape_hw: tuple, centers: list, radius: int, device, dtype) -> torch.Tensor:
    """围绕若干 (h, w) 中心生成 (1, 1, H, W) 局部矩形掩膜。"""
    h, w = shape_hw
    mask = torch.zeros((1, 1, h, w), device=device, dtype=dtype)
    for item in centers:
        hc, wc = item[0], item[1]
        h0, h1 = max(0, hc - radius), min(h, hc + radius + 1)
        w0, w1 = max(0, wc - radius), min(w, wc + radius + 1)
        mask[:, :, h0:h1, w0:w1] = 1.0
    return mask


def _run_collab_pgd_optimization(
    wrapper,
    feat_clean: torch.Tensor,
    record_len: torch.Tensor,
    affine_matrix: torch.Tensor,
    agent_modality_list: list,
    perturb_mask: torch.Tensor,
    attacker_idx: int,
    loss_fn,
    eps: float,
    steps: int,
    err_tag: str,
) -> tuple:
    """协同融合空间下的通用 PGD L_inf 有界扰动优化内核。"""
    delta = torch.zeros_like(feat_clean[attacker_idx : attacker_idx + 1], requires_grad=True)
    step_lr = float(eps) / max(float(steps) * 0.4, 1.0)
    grad_finite = True

    for _ in range(steps):
        adv_slice = feat_clean[attacker_idx : attacker_idx + 1] + delta * perturb_mask
        feat_stack = torch.cat([feat_clean[:attacker_idx], adv_slice, feat_clean[attacker_idx + 1 :]], dim=0)

        out = wrapper.forward_from_protocol(feat_stack, record_len, affine_matrix, agent_modality_list)
        loss = loss_fn(out)

        grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
        if not torch.isfinite(grad).all():
            grad_finite = False
            raise RuntimeError(f"{err_tag} 检测到非有限梯度 (NaN/Inf)")

        with torch.no_grad():
            delta.add_(-step_lr * grad.sign())
            delta.clamp_(-eps, eps)
            delta.mul_(perturb_mask)

    final_delta = (delta.detach() * perturb_mask).clamp(-eps, eps)
    feat_adv = feat_clean.clone()
    feat_adv[attacker_idx : attacker_idx + 1] = feat_clean[attacker_idx : attacker_idx + 1] + final_delta
    return feat_adv, final_delta, grad_finite


def generate_pgd_removal_heal(
    wrapper,
    heter_feature_2d: torch.Tensor,
    record_len: torch.Tensor,
    affine_matrix: torch.Tensor,
    agent_modality_list: list,
    cam_valid_mask: torch.Tensor,
    attacker_idx: int = 1,
    eps: float = 4.0,
    steps: int = 15,
    perturb_region_mask: torch.Tensor = None,
) -> tuple:
    """
    全局白盒 PGD 协同目标擦除攻击。
    直接在传输层 heter_feature_2d[attacker_idx] 上优化 L_inf 有界扰动 delta，
    最小化协同融合后前景目标处的分类激活。
    """
    feat_clean = heter_feature_2d.detach()
    base_mask = cam_valid_mask[attacker_idx : attacker_idx + 1].detach()
    valid_mask = base_mask * perturb_region_mask.detach() if perturb_region_mask is not None else base_mask

    with torch.no_grad():
        clean_out = wrapper.forward_from_protocol(feat_clean, record_len, affine_matrix, agent_modality_list)
        cls_prob = torch.sigmoid(clean_out["cls_preds"]).max(dim=1, keepdim=True)[0]
        cov_in_ego = _warp_mask(valid_mask, affine_matrix, dst_idx=0, src_idx=attacker_idx)

        target_mask = ((cls_prob > 0.20) & (cov_in_ego > 0.5)).float()
        if target_mask.sum() < 8.0:
            masked_prob = cls_prob * cov_in_ego
            if masked_prob.max() <= 1e-4:
                masked_prob = cls_prob
            k_val = min(32, masked_prob.numel())
            thresh = masked_prob.flatten().topk(k_val)[0][-1]
            target_mask = (masked_prob >= thresh).float()

    def _loss(out):
        return (F.softplus(out["cls_preds"]) * target_mask).sum() / (target_mask.sum() + 1e-4)

    feat_adv, final_delta, grad_finite = _run_collab_pgd_optimization(
        wrapper, feat_clean, record_len, affine_matrix, agent_modality_list,
        valid_mask, attacker_idx, _loss, eps, steps, "generate_pgd_removal_heal"
    )
    stats = {
        "grad_finite": grad_finite,
        "target_grids": int(target_mask.sum().item()),
        "eps_used": float(eps),
        "linf_max": float(final_delta.abs().max().item()),
        "l2_mean_valid": float(final_delta.norm(dim=1).mean().item()),
        "perturb_grid_ratio": float((valid_mask > 0.5).float().mean().item()),
    }
    return feat_adv, stats


def generate_scaled_pgd_removal_heal(
    wrapper,
    heter_feature_2d: torch.Tensor,
    record_len: torch.Tensor,
    affine_matrix: torch.Tensor,
    agent_modality_list: list,
    cam_valid_mask: torch.Tensor,
    attacker_idx: int = 1,
    sigma_mult: float = 1.0,
    steps: int = 12,
    perturb_region_mask: torch.Tensor = None,
) -> tuple:
    """
    小预算（按特征自身标准差 sigma 的倍数 0.5σ / 1σ / 2σ）全局 PGD 目标擦除攻击。
    """
    base_mask = cam_valid_mask[attacker_idx : attacker_idx + 1].detach()
    eff_mask = base_mask * perturb_region_mask.detach() if perturb_region_mask is not None else base_mask
    sigma = compute_feature_scale_sigma(heter_feature_2d[attacker_idx : attacker_idx + 1], eff_mask)
    eps = float(sigma_mult) * sigma
    feat_adv, stats = generate_pgd_removal_heal(
        wrapper=wrapper,
        heter_feature_2d=heter_feature_2d,
        record_len=record_len,
        affine_matrix=affine_matrix,
        agent_modality_list=agent_modality_list,
        cam_valid_mask=cam_valid_mask,
        attacker_idx=attacker_idx,
        eps=eps,
        steps=steps,
        perturb_region_mask=perturb_region_mask,
    )
    stats["feature_sigma"] = float(sigma)
    stats["sigma_mult"] = float(sigma_mult)
    return feat_adv, stats


def generate_local_removal_heal(
    wrapper,
    heter_feature_2d: torch.Tensor,
    record_len: torch.Tensor,
    affine_matrix: torch.Tensor,
    agent_modality_list: list,
    cam_valid_mask: torch.Tensor,
    attacker_idx: int = 1,
    num_targets: int = 2,
    patch_radius: int = 5,
    sigma_mult: float = 1.0,
    steps: int = 12,
    perturb_region_mask: torch.Tensor = None,
) -> tuple:
    """
    局部目标精准擦除攻击：仅选取 1~2 个真实共视车辆目标的局部邻域施加扰动，其余全部区域保持不变。
    """
    feat_clean = heter_feature_2d.detach()
    base_mask = cam_valid_mask[attacker_idx : attacker_idx + 1].detach()
    valid_mask = base_mask * perturb_region_mask.detach() if perturb_region_mask is not None else base_mask
    sigma = compute_feature_scale_sigma(feat_clean[attacker_idx : attacker_idx + 1], valid_mask)
    eps = float(sigma_mult) * sigma

    with torch.no_grad():
        clean_out = wrapper.forward_from_protocol(feat_clean, record_len, affine_matrix, agent_modality_list)
        cls_prob = torch.sigmoid(clean_out["cls_preds"]).max(dim=1, keepdim=True)[0]
        cov_in_ego = _warp_mask(valid_mask, affine_matrix, dst_idx=0, src_idx=attacker_idx)
        centers = _sample_local_real_targets_in_cov(cls_prob, cov_in_ego, num_targets=num_targets)

        shape_hw = cls_prob.shape[-2:]
        target_mask_ego = _build_patch_mask(shape_hw, centers, radius=1, device=feat_clean.device, dtype=feat_clean.dtype)
        ego_patch = _build_patch_mask(shape_hw, centers, radius=patch_radius, device=feat_clean.device, dtype=feat_clean.dtype)
        # 逆向映射回协作者局部坐标系，并严格限制在有效传输区内
        local_patch = _warp_mask(ego_patch, affine_matrix, dst_idx=attacker_idx, src_idx=0) * valid_mask
        attacked_mask_in_ego = _warp_mask(local_patch, affine_matrix, dst_idx=0, src_idx=attacker_idx)

    if local_patch.sum() < 1.0:
        return feat_clean.clone(), {
            "grad_finite": True,
            "num_targets": 0,
            "eps_used": float(eps),
            "feature_sigma": float(sigma),
            "linf_max": 0.0,
            "l2_mean_valid": 0.0,
            "perturb_grids": 0,
            "attacked_mask_in_ego": attacked_mask_in_ego,
        }

    def _loss(out):
        return (F.softplus(out["cls_preds"]) * target_mask_ego).sum() / (target_mask_ego.sum() + 1e-4)

    feat_adv, final_delta, grad_finite = _run_collab_pgd_optimization(
        wrapper, feat_clean, record_len, affine_matrix, agent_modality_list,
        local_patch, attacker_idx, _loss, eps, steps, "generate_local_removal_heal"
    )
    stats = {
        "grad_finite": grad_finite,
        "num_targets": len(centers),
        "target_centers_hw": centers,
        "eps_used": float(eps),
        "feature_sigma": float(sigma),
        "sigma_mult": float(sigma_mult),
        "linf_max": float(final_delta.abs().max().item()),
        "l2_mean_valid": float(final_delta.norm(dim=1).mean().item()),
        "perturb_grids": int((local_patch > 0.5).sum().item()),
        "attacked_mask_in_ego": attacked_mask_in_ego,
    }
    return feat_adv, stats


def _local_grid_radius_256(shape_hw: tuple, device, dtype, xy_range: float = 204.8) -> torch.Tensor:
    """生成传感器自身局部坐标系 [-102.4, 102.4]^2 网格中心到原点距离 (1, 1, H, W)。"""
    h, w = shape_hw
    res_x = float(xy_range) / float(w)
    res_y = float(xy_range) / float(h)
    half = 0.5 * float(xy_range)
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) * res_x - half
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) * res_y - half
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.sqrt(gx * gx + gy * gy).view(1, 1, h, w)


def _build_spoof_targets_and_loss(cls_shape: tuple, anchors: list, device, dtype):
    """根据伪造锚点列表 (h, w, rot_idx) 构建伪造掩膜与可微 BCE+Reg 损失闭包。"""
    _, _, h, w = cls_shape
    spoof_mask_2ch = torch.zeros(cls_shape, device=device, dtype=dtype)
    for hc, wc, rot_idx in anchors:
        h0, h1 = max(0, hc - 1), min(h, hc + 2)
        w0, w1 = max(0, wc - 1), min(w, wc + 2)
        spoof_mask_2ch[:, rot_idx, h0:h1, w0:w1] = 1.0
    spoof_mask_1ch = spoof_mask_2ch.max(dim=1, keepdim=True)[0]

    def _loss(out):
        cls_logits = out["cls_preds"]
        reg_preds = out["reg_preds"]
        bce_pos = F.binary_cross_entropy_with_logits(cls_logits, torch.ones_like(cls_logits), reduction="none")
        loss_cls = (bce_pos * spoof_mask_2ch).sum() / (spoof_mask_2ch.sum() + 1e-4)
        loss_reg = 0.05 * (reg_preds * spoof_mask_1ch).pow(2).sum() / (spoof_mask_1ch.sum() + 1e-4)
        return loss_cls + loss_reg

    return spoof_mask_2ch, spoof_mask_1ch, _loss


def _sample_region_ghost_anchors(
    cls_prob_ego: torch.Tensor,
    region_mask_ego: torch.Tensor,
    num_spoofs: int,
    min_sep_grids: int,
    seed: int,
    edge_margin: int = 6,
) -> list:
    """在指定本车空间掩膜内，筛选置信度小于 0.10 且远离边界与真车的空白锚点。"""
    _, _, h, w = cls_prob_ego.shape
    edge_ok = torch.zeros((h, w), device=cls_prob_ego.device, dtype=torch.bool)
    edge_ok[edge_margin : h - edge_margin, edge_margin : w - edge_margin] = True
    base_ok = (region_mask_ego[0, 0] > 0.5) & edge_ok

    # 优先选择大邻域无真车网格，不足时逐步退化至单格 cls < 0.10
    occ_9 = F.max_pool2d(cls_prob_ego, kernel_size=9, stride=1, padding=4)[0, 0] >= 0.15
    occ_3 = F.max_pool2d(cls_prob_ego, kernel_size=3, stride=1, padding=1)[0, 0] >= 0.12
    occ_1 = cls_prob_ego[0, 0] >= 0.10

    cand_coords = torch.nonzero(base_ok & (~occ_9), as_tuple=False)
    if cand_coords.shape[0] < num_spoofs:
        cand_coords = torch.nonzero(base_ok & (~occ_3), as_tuple=False)
    if cand_coords.shape[0] < num_spoofs:
        cand_coords = torch.nonzero(base_ok & (~occ_1), as_tuple=False)
    if cand_coords.shape[0] == 0:
        return []

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    perm = torch.randperm(cand_coords.shape[0], generator=g)

    selected = []
    for idx in perm.tolist():
        hc = int(cand_coords[idx, 0].item())
        wc = int(cand_coords[idx, 1].item())
        if any(abs(hc - hh) < min_sep_grids and abs(wc - ww) < min_sep_grids for hh, ww, _ in selected):
            continue
        rot_idx = int(torch.randint(0, 2, (1,), generator=g).item())
        selected.append((hc, wc, rot_idx))
        if len(selected) >= num_spoofs:
            break
    return selected


def generate_ghost_spoofing_heal(
    wrapper,
    heter_feature_2d: torch.Tensor,
    record_len: torch.Tensor,
    affine_matrix: torch.Tensor,
    agent_modality_list: list,
    cam_valid_mask: torch.Tensor,
    attacker_idx: int = 1,
    eps: float = 4.5,
    steps: int = 15,
    num_spoofs: int = 6,
    frame_seed: int = 0,
    perturb_region_mask: torch.Tensor = None,
) -> tuple:
    """
    全局随机道路空白区鬼影伪造攻击。
    在无真车的协作者覆盖区随机采点，联合优化 cls_preds 伪造置信度与回归稳定正则。
    """
    feat_clean = heter_feature_2d.detach()
    base_mask = cam_valid_mask[attacker_idx : attacker_idx + 1].detach()
    valid_mask = base_mask * perturb_region_mask.detach() if perturb_region_mask is not None else base_mask

    with torch.no_grad():
        clean_out = wrapper.forward_from_protocol(feat_clean, record_len, affine_matrix, agent_modality_list)
        cls_prob = torch.sigmoid(clean_out["cls_preds"]).max(dim=1, keepdim=True)[0]
        cov_in_ego = _warp_mask(valid_mask, affine_matrix, dst_idx=0, src_idx=attacker_idx)
        anchors = _sample_ghost_anchors_in_cov(cls_prob, cov_in_ego, num_spoofs=num_spoofs, frame_seed=frame_seed)
        spoof_mask_2ch, _, _loss = _build_spoof_targets_and_loss(
            clean_out["cls_preds"].shape, anchors, feat_clean.device, feat_clean.dtype
        )

    if spoof_mask_2ch.sum() < 1.0:
        return feat_clean.clone(), {"grad_finite": True, "num_spoof_anchors": 0, "linf_max": 0.0, "l2_mean_valid": 0.0}

    feat_adv, final_delta, grad_finite = _run_collab_pgd_optimization(
        wrapper, feat_clean, record_len, affine_matrix, agent_modality_list,
        valid_mask, attacker_idx, _loss, eps, steps, "generate_ghost_spoofing_heal"
    )
    stats = {
        "grad_finite": grad_finite,
        "num_spoof_anchors": len(anchors),
        "eps_used": float(eps),
        "linf_max": float(final_delta.abs().max().item()),
        "l2_mean_valid": float(final_delta.norm(dim=1).mean().item()),
    }
    return feat_adv, stats


def generate_local_ghost_spoofing_heal(
    wrapper,
    heter_feature_2d: torch.Tensor,
    record_len: torch.Tensor,
    affine_matrix: torch.Tensor,
    agent_modality_list: list,
    cam_valid_mask: torch.Tensor,
    attacker_idx: int = 1,
    num_spoofs: int = 2,
    patch_radius: int = 5,
    sigma_mult: float = 2.0,
    steps: int = 12,
    frame_seed: int = 0,
    perturb_region_mask: torch.Tensor = None,
) -> tuple:
    """
    局部空白区鬼影伪造攻击：仅在 1~2 个随机空白目标锚点局部块内施加扰动，其余区域严格不变。
    """
    feat_clean = heter_feature_2d.detach()
    base_mask = cam_valid_mask[attacker_idx : attacker_idx + 1].detach()
    valid_mask = base_mask * perturb_region_mask.detach() if perturb_region_mask is not None else base_mask
    sigma = compute_feature_scale_sigma(feat_clean[attacker_idx : attacker_idx + 1], valid_mask)
    eps = float(sigma_mult) * sigma

    with torch.no_grad():
        clean_out = wrapper.forward_from_protocol(feat_clean, record_len, affine_matrix, agent_modality_list)
        cls_prob = torch.sigmoid(clean_out["cls_preds"]).max(dim=1, keepdim=True)[0]
        cov_in_ego = _warp_mask(valid_mask, affine_matrix, dst_idx=0, src_idx=attacker_idx)
        anchors = _sample_ghost_anchors_in_cov(cls_prob, cov_in_ego, num_spoofs=num_spoofs, frame_seed=frame_seed)

        spoof_mask_2ch, _, _loss = _build_spoof_targets_and_loss(
            clean_out["cls_preds"].shape, anchors, feat_clean.device, feat_clean.dtype
        )
        ego_patch = _build_patch_mask(cls_prob.shape[-2:], anchors, radius=patch_radius, device=feat_clean.device, dtype=feat_clean.dtype)
        local_patch = _warp_mask(ego_patch, affine_matrix, dst_idx=attacker_idx, src_idx=0) * valid_mask
        attacked_mask_in_ego = _warp_mask(local_patch, affine_matrix, dst_idx=0, src_idx=attacker_idx)

    if spoof_mask_2ch.sum() < 1.0 or local_patch.sum() < 1.0:
        return feat_clean.clone(), {
            "grad_finite": True,
            "num_spoof_anchors": 0,
            "eps_used": float(eps),
            "feature_sigma": float(sigma),
            "linf_max": 0.0,
            "l2_mean_valid": 0.0,
            "perturb_grids": 0,
            "attacked_mask_in_ego": attacked_mask_in_ego,
        }

    feat_adv, final_delta, grad_finite = _run_collab_pgd_optimization(
        wrapper, feat_clean, record_len, affine_matrix, agent_modality_list,
        local_patch, attacker_idx, _loss, eps, steps, "generate_local_ghost_spoofing_heal"
    )
    stats = {
        "grad_finite": grad_finite,
        "num_spoof_anchors": len(anchors),
        "anchors_hwr": anchors,
        "eps_used": float(eps),
        "feature_sigma": float(sigma),
        "sigma_mult": float(sigma_mult),
        "linf_max": float(final_delta.abs().max().item()),
        "l2_mean_valid": float(final_delta.norm(dim=1).mean().item()),
        "perturb_grids": int((local_patch > 0.5).sum().item()),
        "attacked_mask_in_ego": attacked_mask_in_ego,
    }
    return feat_adv, stats


def generate_local_near0_6m_attack_heal(
    wrapper,
    heter_feature_2d: torch.Tensor,
    record_len: torch.Tensor,
    affine_matrix: torch.Tensor,
    agent_modality_list: list,
    cam_valid_mask: torch.Tensor,
    attacker_idx: int = 1,
    sigma_mult: float = 2.0,
    steps: int = 15,
    frame_seed: int = 0,
    perturb_region_mask: torch.Tensor = None,
    enable_ghost_fallback: bool = False,
) -> tuple:
    """
    0~6m 近场高幅带内局部目标擦除攻击（近场隐身压测）：
    100% 扰动像素被严格约束在协作者局部距离 d_cav < 6.0m 内，使用全局温和预算 2.0*sigma_global
    擦除协作者自身或紧邻 6m 内的真实车辆前景激活。
    """
    feat_clean = heter_feature_2d.detach()
    base_mask = cam_valid_mask[attacker_idx : attacker_idx + 1].detach()
    valid_mask = base_mask * perturb_region_mask.detach() if perturb_region_mask is not None else base_mask
    sigma = compute_feature_scale_sigma(feat_clean[attacker_idx : attacker_idx + 1], valid_mask)
    eps = float(sigma_mult) * sigma

    r_local = _local_grid_radius_256(feat_clean.shape[-2:], feat_clean.device, feat_clean.dtype)
    near_mask_local = (r_local < 6.0).to(feat_clean.dtype) * valid_mask

    with torch.no_grad():
        clean_out = wrapper.forward_from_protocol(feat_clean, record_len, affine_matrix, agent_modality_list)
        cls_prob = torch.sigmoid(clean_out["cls_preds"]).max(dim=1, keepdim=True)[0]
        near_ego = _warp_mask(near_mask_local, affine_matrix, dst_idx=0, src_idx=attacker_idx)
        near_ego_bool = near_ego > 0.5
        max_clean_fg = float(cls_prob[near_ego_bool].max().item()) if near_ego_bool.any() else 0.0

        target_mask_ego = ((cls_prob > 0.20) & near_ego_bool).to(feat_clean.dtype)
        if target_mask_ego.sum() < 1.0 and max_clean_fg > 0.15:
            target_mask_ego = ((cls_prob > 0.15) & near_ego_bool).to(feat_clean.dtype)

    if target_mask_ego.sum() >= 1.0 and near_mask_local.sum() >= 1.0:
        with torch.no_grad():
            near_centers = _sample_local_real_targets_in_cov(cls_prob, near_ego, num_targets=2)
            if not near_centers:
                masked_p = (cls_prob * target_mask_ego)[0, 0]
                flat_i = int(torch.argmax(masked_p).item())
                w_dim = int(masked_p.shape[-1])
                near_centers = [(flat_i // w_dim, flat_i % w_dim)]

        def _rem_loss(out):
            return (F.softplus(out["cls_preds"]) * target_mask_ego).sum() / (target_mask_ego.sum() + 1e-4)

        feat_adv, final_delta, grad_finite = _run_collab_pgd_optimization(
            wrapper, feat_clean, record_len, affine_matrix, agent_modality_list,
            near_mask_local, attacker_idx, _rem_loss, eps, steps, "generate_local_near0_6m_attack_heal"
        )
        with torch.no_grad():
            adv_out = wrapper.forward_from_protocol(feat_adv, record_len, affine_matrix, agent_modality_list)
            adv_cls = torch.sigmoid(adv_out["cls_preds"]).max(dim=1, keepdim=True)[0]
            after_max_fg = float(adv_cls[target_mask_ego > 0.5].max().item())
            prob_drop = max_clean_fg - after_max_fg
            attacked_mask_in_ego = near_ego

        nz_mask = final_delta.abs().max(dim=1, keepdim=True)[0] > 0.0
        if nz_mask.any():
            in_ratio = float((nz_mask & (r_local < 6.0)).sum().item()) / float(nz_mask.sum().item())
        else:
            lp_bool = near_mask_local > 0.5
            in_ratio = float((lp_bool & (r_local < 6.0)).sum().item()) / max(float(lp_bool.sum().item()), 1.0)

        return feat_adv, {
            "valid": True,
            "submode": "removal",
            "grad_finite": grad_finite,
            "num_targets": len(near_centers),
            "target_centers_hw": near_centers,
            "target_grids": int(target_mask_ego.sum().item()),
            "target_mask_max_clean": float(max_clean_fg),
            "prob_drop": float(prob_drop),
            "max_ghost_prob": 0.0,
            "eps_used": float(eps),
            "feature_sigma": float(sigma),
            "sigma_mult": float(sigma_mult),
            "linf_max": float(final_delta.abs().max().item()),
            "l2_mean_valid": float(final_delta.norm(dim=1).mean().item()),
            "perturb_grids": int((near_mask_local > 0.5).sum().item()),
            "perturb_in_0_6m_ratio": float(in_ratio),
            "attacked_mask_in_ego": attacked_mask_in_ego,
        }

    # 无近场前景目标时，按开关决定返回无效或回退近场伪造
    if enable_ghost_fallback and near_mask_local.sum() >= 1.0:
        ring_local = ((r_local >= 1.6) & (r_local <= 5.0)).to(feat_clean.dtype) * near_mask_local
        with torch.no_grad():
            ring_in_ego = _warp_mask(ring_local, affine_matrix, dst_idx=0, src_idx=attacker_idx)
            anchors = _sample_region_ghost_anchors(
                cls_prob, ring_in_ego, num_spoofs=1, min_sep_grids=4,
                seed=20261120 + int(frame_seed) + 600, edge_margin=6,
            )
            spoof_mask_2ch, _, _ghost_loss = _build_spoof_targets_and_loss(
                clean_out["cls_preds"].shape, anchors, feat_clean.device, feat_clean.dtype
            )
        if len(anchors) > 0 and spoof_mask_2ch.sum() >= 1.0:
            eps_fb = 3.0 * sigma
            feat_adv, final_delta, grad_finite = _run_collab_pgd_optimization(
                wrapper, feat_clean, record_len, affine_matrix, agent_modality_list,
                near_mask_local, attacker_idx, _ghost_loss, eps_fb, steps, "generate_local_near0_6m_fallback"
            )
            with torch.no_grad():
                adv_out = wrapper.forward_from_protocol(feat_adv, record_len, affine_matrix, agent_modality_list)
                gp = float(max(torch.sigmoid(adv_out["cls_preds"])[0, r, h, w].item() for h, w, r in anchors))
            nz_mask = final_delta.abs().max(dim=1, keepdim=True)[0] > 0.0
            in_ratio = (
                float((nz_mask & (r_local < 6.0)).sum().item()) / float(nz_mask.sum().item())
                if nz_mask.any() else 1.0
            )
            return feat_adv, {
                "valid": True,
                "submode": "ghost_fallback",
                "grad_finite": grad_finite,
                "target_grids": 0,
                "target_mask_max_clean": float(max_clean_fg),
                "prob_drop": 0.0,
                "max_ghost_prob": float(gp),
                "anchors_hwr": anchors,
                "eps_used": float(eps_fb),
                "feature_sigma": float(sigma),
                "sigma_mult": 3.0,
                "linf_max": float(final_delta.abs().max().item()),
                "l2_mean_valid": float(final_delta.norm(dim=1).mean().item()),
                "perturb_grids": int((near_mask_local > 0.5).sum().item()),
                "perturb_in_0_6m_ratio": float(in_ratio),
                "attacked_mask_in_ego": near_ego,
            }

    return feat_clean.clone(), {
        "valid": False,
        "submode": "none",
        "grad_finite": True,
        "num_targets": 0,
        "target_centers_hw": [],
        "target_grids": 0,
        "target_mask_max_clean": float(max_clean_fg),
        "prob_drop": 0.0,
        "max_ghost_prob": 0.0,
        "eps_used": float(eps),
        "feature_sigma": float(sigma),
        "sigma_mult": float(sigma_mult),
        "linf_max": 0.0,
        "perturb_grids": 0,
        "perturb_in_0_6m_ratio": 0.0,
        "attacked_mask_in_ego": near_ego,
    }


def generate_local_occluded_ghost_heal(
    wrapper,
    heter_feature_2d: torch.Tensor,
    record_len: torch.Tensor,
    affine_matrix: torch.Tensor,
    agent_modality_list: list,
    cam_valid_mask: torch.Tensor,
    occluded_mask_ego: torch.Tensor,
    attacker_idx: int = 1,
    num_spoofs: int = 2,
    patch_radius: int = 5,
    sigma_mult: float = 2.0,
    steps: int = 12,
    frame_seed: int = 0,
    perturb_region_mask: torch.Tensor = None,
) -> tuple:
    """
    只在本车遮挡区伪造攻击（能力边界压测）：
    在 ego 遮挡区内选空白锚点，并将局部扰动严格限制在映射到协作者的遮挡区域内部。
    """
    feat_clean = heter_feature_2d.detach()
    base_mask = cam_valid_mask[attacker_idx : attacker_idx + 1].detach()
    valid_mask = base_mask * perturb_region_mask.detach() if perturb_region_mask is not None else base_mask
    sigma = compute_feature_scale_sigma(feat_clean[attacker_idx : attacker_idx + 1], valid_mask)
    eps = float(sigma_mult) * sigma

    occ_ego_f = (occluded_mask_ego > 0.5).to(device=feat_clean.device, dtype=feat_clean.dtype)
    occ_mask_local = _warp_mask(occ_ego_f, affine_matrix, dst_idx=attacker_idx, src_idx=0)

    with torch.no_grad():
        clean_out = wrapper.forward_from_protocol(feat_clean, record_len, affine_matrix, agent_modality_list)
        cls_prob = torch.sigmoid(clean_out["cls_preds"]).max(dim=1, keepdim=True)[0]
        cov_in_ego = _warp_mask(valid_mask, affine_matrix, dst_idx=0, src_idx=attacker_idx)

        # 优先在 5x5 全遮挡区深处采锚点，保证局部块落在遮挡区内；不足时退化到 3x3 全遮挡区
        occ_5 = (F.avg_pool2d(occ_ego_f, kernel_size=5, stride=1, padding=2) > 0.99).to(feat_clean.dtype) * cov_in_ego
        occ_3 = (F.avg_pool2d(occ_ego_f, kernel_size=3, stride=1, padding=1) > 0.99).to(feat_clean.dtype) * cov_in_ego
        anchors = _sample_region_ghost_anchors(
            cls_prob, occ_5, num_spoofs=num_spoofs, min_sep_grids=12, seed=20261120 + int(frame_seed) + 900, edge_margin=6
        )
        if len(anchors) == 0:
            anchors = _sample_region_ghost_anchors(
                cls_prob, occ_3, num_spoofs=num_spoofs, min_sep_grids=12, seed=20261120 + int(frame_seed) + 900, edge_margin=6
            )

        spoof_mask_2ch, _, _loss = _build_spoof_targets_and_loss(
            clean_out["cls_preds"].shape, anchors, feat_clean.device, feat_clean.dtype
        )
        ego_patch = _build_patch_mask(cls_prob.shape[-2:], anchors, radius=patch_radius, device=feat_clean.device, dtype=feat_clean.dtype)
        local_patch = _warp_mask(ego_patch, affine_matrix, dst_idx=attacker_idx, src_idx=0) * occ_mask_local * valid_mask
        attacked_mask_in_ego = _warp_mask(local_patch, affine_matrix, dst_idx=0, src_idx=attacker_idx)

    if len(anchors) == 0 or spoof_mask_2ch.sum() < 1.0 or local_patch.sum() < 9.0:
        return feat_clean.clone(), {
            "valid": False,
            "grad_finite": True,
            "num_spoof_anchors": 0,
            "anchors_hwr": [],
            "eps_used": float(eps),
            "feature_sigma": float(sigma),
            "sigma_mult": float(sigma_mult),
            "linf_max": 0.0,
            "perturb_grids": int((local_patch > 0.5).sum().item()),
            "perturb_in_occ_ratio": 0.0,
            "attacked_mask_in_ego": attacked_mask_in_ego,
        }

    feat_adv, final_delta, grad_finite = _run_collab_pgd_optimization(
        wrapper, feat_clean, record_len, affine_matrix, agent_modality_list,
        local_patch, attacker_idx, _loss, eps, steps, "generate_local_occluded_ghost_heal"
    )
    atk_bool = attacked_mask_in_ego > 0.5
    occ_bool = occ_ego_f > 0.5
    in_occ_ratio = float((atk_bool & occ_bool).sum().item()) / max(float(atk_bool.sum().item()), 1.0)

    stats = {
        "valid": True,
        "grad_finite": grad_finite,
        "num_spoof_anchors": len(anchors),
        "anchors_hwr": anchors,
        "eps_used": float(eps),
        "feature_sigma": float(sigma),
        "sigma_mult": float(sigma_mult),
        "linf_max": float(final_delta.abs().max().item()),
        "l2_mean_valid": float(final_delta.norm(dim=1).mean().item()),
        "perturb_grids": int((local_patch > 0.5).sum().item()),
        "perturb_in_occ_ratio": float(in_occ_ratio),
        "attacked_mask_in_ego": attacked_mask_in_ego,
    }
    return feat_adv, stats


def generate_surrogate_transfer_removal_heal(
    wrapper,
    heter_feature_2d: torch.Tensor,
    cam_valid_mask: torch.Tensor,
    attacker_idx: int = 1,
    eps: float = 4.0,
    steps: int = 15,
) -> tuple:
    """
    无本车信息的代理迁移擦除攻击（HetPoison 风格黑盒迁移）。
    绝不读取本车特征 heter_feature_2d[0] 或协同变换矩阵 affine_matrix。
    """
    feat_clean = heter_feature_2d.detach()
    feat_cav_clean = feat_clean[attacker_idx : attacker_idx + 1]
    valid_mask = cam_valid_mask[attacker_idx : attacker_idx + 1].detach()

    with torch.no_grad():
        single_clean = wrapper.forward_single_agent(feat_cav_clean)
        cls_prob_cav = torch.sigmoid(single_clean["cls_preds"]).max(dim=1, keepdim=True)[0]
        target_mask = ((cls_prob_cav > 0.15) & (valid_mask > 0.5)).float()
        if target_mask.sum() < 8.0:
            masked_prob = cls_prob_cav * valid_mask
            k_val = min(48, masked_prob.numel())
            thresh = masked_prob.flatten().topk(k_val)[0][-1]
            target_mask = ((masked_prob >= thresh) & (valid_mask > 0.5)).float()

        target_mask_dil = F.max_pool2d(target_mask, kernel_size=3, stride=1, padding=1) * valid_mask

    delta = torch.zeros_like(feat_cav_clean, requires_grad=True)
    step_lr = float(eps) / max(float(steps) * 0.4, 1.0)
    grad_finite = True

    for _ in range(steps):
        adv_slice = feat_cav_clean + delta * valid_mask
        out_single = wrapper.forward_single_agent(adv_slice)

        loss_erase = (F.softplus(out_single["cls_preds"]) * target_mask_dil).sum() / (target_mask_dil.sum() + 1e-4)

        loss_hijack = 0.0
        for occ_map in out_single["occ_single_list"]:
            m_scaled = F.interpolate(target_mask_dil, size=occ_map.shape[-2:], mode="nearest")
            loss_hijack = loss_hijack - (torch.sigmoid(occ_map) * m_scaled).sum() / (m_scaled.sum() + 1e-4)

        loss = loss_erase + 0.5 * loss_hijack
        grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
        if not torch.isfinite(grad).all():
            grad_finite = False
            raise RuntimeError("generate_surrogate_transfer_removal_heal 检测到非有限梯度 (NaN/Inf)")

        with torch.no_grad():
            delta.add_(-step_lr * grad.sign())
            delta.clamp_(-eps, eps)
            delta.mul_(valid_mask)

    final_delta = (delta.detach() * valid_mask).clamp(-eps, eps)
    feat_adv = feat_clean.clone()
    feat_adv[attacker_idx : attacker_idx + 1] = feat_cav_clean + final_delta

    stats = {
        "grad_finite": grad_finite,
        "target_grids": int(target_mask_dil.sum().item()),
        "linf_max": float(final_delta.abs().max().item()),
        "l2_mean_valid": float(final_delta.norm(dim=1).mean().item()),
    }
    return feat_adv, stats
