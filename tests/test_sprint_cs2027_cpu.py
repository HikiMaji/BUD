# -*- coding: utf-8 -*-
"""
sprint-cs2027 全量 CPU 单元测试套件（零外部数据集依赖，本机纯 PyTorch 即可毫秒级运行）。

核心覆盖范围：
1. 【B3 强制 5 项单元测试（必须 100% 通过方可合并）】：
   - Case 1: 所有 G = 1 时，ω̃ = ω；
   - Case 2: 只有攻击者覆盖（ω_0 = 0, ω_a = 1）、G_a = 0.1 的格子，ω̃_a = 0.1·ω_a，其余权重全部归本车 ω̃_0；
   - Case 3: 随机多车/多门控下，每个格子上 Σ ω̃ = Σ ω（权重总量严格守恒）；
   - Case 4: 只有 1 辆协作车时，不论 G_1 取值多少（特别包含 G_1 = 0.95），结果与“全部还给本车”的原实现逐元素完全一致；
   - Case 5: 两辆协作车，特别测试 G_1 = 0.95、G_2 = 0.2 时，协作车 2 被削减的权重按 ω_0 : ω_1 分给本车和协作车 1，
             且协作车 1 自身被削减的 0.05·ω_1 100% 只分给本车、绝不回到它自己；
2. 【B1 粗尺度门控 Min-Pool 单元测试】：验证小面积低门控在 L1/L2 不被均值池化稀释；
3. 【B2 新前景规则单元测试】：验证 Clear 区以本车为准防伪造骗宽松阈值，非 Clear 区保留诚实补盲前景；
4. 【A1 单一可见性函数校验】：确认旧函数已移除且 assign_dist5_vis3_indices_v2 物理分档准确；
5. 【A2 HetShield 核对修复单测】：验证默认 α=0.6、隔离不锁死隐状态、以及 z̃_k = g_k z_k + (1-g_k) z_0 线性混合重算占用分数；
6. 【A4 标准指标与统计检验单测】：验证防刷 ASR（s_min 过滤 + 3m 内真值剔除并单列近目标伪造）、盲区 AP Ignore 非盲区真值、
   诚实压制比例与权重削减比、序列级 Studentized Bootstrap + Holm 校正；
7. 【A5 标定/调参序列硬隔离与稀疏桶合并单测】：验证 7/4 序列零交集及帧数 < 20 自动向上一级合并。
"""

import os
import sys
import types
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _install_lightweight_cpu_shims():
    """在本机未安装 torchvision / opencood C++ 扩展时注入数学等价的纯 PyTorch 垫片。"""
    if "torchvision" not in sys.modules:
        tv = types.ModuleType("torchvision")
        tv.ops = types.ModuleType("torchvision.ops")
        sys.modules["torchvision"] = tv
        sys.modules["torchvision.ops"] = tv.ops

    try:
        import opencood.models.fuse_modules.fusion_in_one  # noqa: F401
        import opencood.models.sub_modules.torch_transformation_utils  # noqa: F401
    except ImportError:
        oc = types.ModuleType("opencood")
        oc_models = types.ModuleType("opencood.models")
        oc_fuse = types.ModuleType("opencood.models.fuse_modules")
        oc_fio = types.ModuleType("opencood.models.fuse_modules.fusion_in_one")
        oc_sub = types.ModuleType("opencood.models.sub_modules")
        oc_trans = types.ModuleType("opencood.models.sub_modules.torch_transformation_utils")
        oc_utils = types.ModuleType("opencood.utils")
        oc_common = types.ModuleType("opencood.utils.common_utils")

        def regroup(x: torch.Tensor, record_len: torch.Tensor):
            sizes = [int(v.item()) if torch.is_tensor(v) else int(v) for v in record_len]
            return list(torch.split(x, sizes, dim=0))

        def warp_affine_simple(src: torch.Tensor, M: torch.Tensor, dsize: tuple, align_corners: bool = False):
            # 当传入单位阵时直接恒等返回，避免插值边界数值舍入干扰精确代数检验
            n, c, h, w = src.shape
            th, tw = int(dsize[0]), int(dsize[1])
            eye = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=M.device, dtype=M.dtype).unsqueeze(0)
            if h == th and w == tw and torch.allclose(M, eye.expand_as(M), atol=1e-6):
                return src.clone()
            grid = F.affine_grid(M, [n, c, th, tw], align_corners=align_corners)
            return F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=align_corners)

        def torch_tensor_to_numpy(t):
            return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)

        def convert_format(boxes_np: np.ndarray):
            # 返回轴对齐矩形包装对象用于轻量单测 IoU 计算
            return [boxes_np[i, :4, :2].astype(np.float64) for i in range(boxes_np.shape[0])]

        def compute_iou(poly_a: np.ndarray, polys_b: list) -> np.ndarray:
            ax0, ay0 = float(poly_a[:, 0].min()), float(poly_a[:, 1].min())
            ax1, ay1 = float(poly_a[:, 0].max()), float(poly_a[:, 1].max())
            area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
            ious = []
            for pb in polys_b:
                bx0, by0 = float(pb[:, 0].min()), float(pb[:, 1].min())
                bx1, by1 = float(pb[:, 0].max()), float(pb[:, 1].max())
                area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
                ix0, iy0 = max(ax0, bx0), max(ay0, by0)
                ix1, iy1 = min(ax1, bx1), min(ay1, by1)
                inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                union = area_a + area_b - inter
                ious.append(inter / union if union > 1e-12 else 0.0)
            return np.asarray(ious, dtype=np.float64)

        oc_fio.regroup = regroup
        oc_trans.warp_affine_simple = warp_affine_simple
        oc_common.torch_tensor_to_numpy = torch_tensor_to_numpy
        oc_common.convert_format = convert_format
        oc_common.compute_iou = compute_iou

        sys.modules["opencood"] = oc
        sys.modules["opencood.models"] = oc_models
        sys.modules["opencood.models.fuse_modules"] = oc_fuse
        sys.modules["opencood.models.fuse_modules.fusion_in_one"] = oc_fio
        sys.modules["opencood.models.sub_modules"] = oc_sub
        sys.modules["opencood.models.sub_modules.torch_transformation_utils"] = oc_trans
        sys.modules["opencood.utils"] = oc_utils
        sys.modules["opencood.utils.common_utils"] = oc_common


_install_lightweight_cpu_shims()

from src_v2_mm import step4_1_v2_core  # noqa: E402
from src_v2_mm.heal_base_wrapper import ensure_in_order_assignment  # noqa: E402
from src_v2_mm.a5_split_and_calib_stats import (  # noqa: E402
    FIT_CALIB_SEQ_NAMES_7,
    FIT_TUNE_SEQ_NAMES_4,
    build_fg_bg_vis_quantiles_with_ci_and_merging,
)
from src_v2_mm.bud_fusion_heal import (  # noqa: E402
    build_directional_gate_map,
    build_fg_mask,
    build_gated_fuse_fn,
    compute_frame_high_tau_with_rule,
    redistribute_weights_ego_only,
    redistribute_weights_trusted_other,
)
from scripts_v2_mm.eval_standard_suite_v3 import build_visibility_mask_gate  # noqa: E402
from src_v2_mm.hetshield_heal import (  # noqa: E402
    HetShieldHEALDefender,
    build_hetshield_fuse_fn,
    forward_hetshield_protocol_mix,
    mix_hetshield_protocol_features_ego,
)
from src_v2_mm.sprint_metrics_v3 import (  # noqa: E402
    compute_honest_suppression_stats,
    evaluate_frame_asr_strict,
    holm_bonferroni_adjust,
    match_blind_spot_ap_ignore_nonblind,
    studentized_paired_block_bootstrap,
)


class _DummyPyramidBackbone(nn.Module):
    def __init__(self, c_list=(8, 16, 32)):
        super().__init__()
        self.align_corners = False
        self.num_levels = 3
        self.get_multiscale_calls = 0
        self.last_input_to_pyramid = None
        self.l0 = nn.Conv2d(c_list[0], c_list[0], kernel_size=1, bias=False)
        self.l1 = nn.Conv2d(c_list[0], c_list[1], kernel_size=2, stride=2, bias=False)
        self.l2 = nn.Conv2d(c_list[1], c_list[2], kernel_size=2, stride=2, bias=False)
        self.single_head_0 = nn.Conv2d(c_list[0], 1, kernel_size=1)
        self.single_head_1 = nn.Conv2d(c_list[1], 1, kernel_size=1)
        self.single_head_2 = nn.Conv2d(c_list[2], 1, kernel_size=1)
        for m in (self.single_head_0, self.single_head_1, self.single_head_2):
            nn.init.constant_(m.weight, 0.0)
            nn.init.constant_(m.bias, 1.0)

    def get_multiscale_feature(self, x: torch.Tensor):
        self.get_multiscale_calls += 1
        self.last_input_to_pyramid = x.detach().clone()
        f0 = self.l0(x)
        f1 = self.l1(f0)
        f2 = self.l2(f1)
        return [f0, f1, f2]

    def decode_multiscale_feature(self, fused_list):
        return fused_list[0]


class _DummyFusionNet(nn.Module):
    def __init__(self, c_list=(64, 128, 256)):
        super().__init__()
        self.single_head_0 = nn.Conv2d(c_list[0], 1, kernel_size=1)
        self.single_head_1 = nn.Conv2d(c_list[1], 1, kernel_size=1)
        self.single_head_2 = nn.Conv2d(c_list[2], 1, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
                nn.init.constant_(m.bias, -4.0)


class _DummyModel:
    def __init__(self):
        self.pyramid_backbone = _DummyPyramidBackbone()
        self.shrink_flag = False
        self.fusion_net = _DummyFusionNet()
        self.cls_head = nn.Conv2d(8, 2, kernel_size=1)
        self.reg_head = nn.Conv2d(8, 7, kernel_size=1)
        self.dir_head = nn.Conv2d(8, 2, kernel_size=1)


class _DummyWrapper:
    def __init__(self):
        self.model = _DummyModel()


class TestSprintCS2027CPU(unittest.TestCase):
    """覆盖 B3 全部 5 条强制单元测试及 A1/A2/A4/A5/B1/B2 全链路验证。"""

    # =========================================================================
    # B3 强制单元测试 1~5
    # =========================================================================
    def test_b3_case1_all_gates_one(self):
        """[B3 单测 1] 所有 G = 1 时，不论车数与有效掩膜分布，严格满足 ω̃ = ω。"""
        torch.manual_seed(20261120)
        for n_cav in (2, 3, 5):
            raw_logits = torch.randn(n_cav, 1, 16, 16)
            valid = torch.rand(n_cav, 1, 16, 16) > 0.25
            valid[:, :, 0, 0] = False  # 含全无效格
            any_valid = valid.any(dim=0, keepdim=True)
            safe_logits = torch.where(valid, raw_logits, torch.full_like(raw_logits, -1e4))
            safe_logits = torch.where(any_valid, safe_logits, torch.zeros_like(safe_logits))
            w_base = torch.softmax(safe_logits, dim=0) * any_valid.float()

            gates_one = torch.ones((n_cav - 1, 1, 16, 16), dtype=torch.float32)
            w_tilde = redistribute_weights_trusted_other(w_base, gates_one, any_valid=any_valid, tau_trust=0.9)
            self.assertTrue(torch.allclose(w_tilde, w_base, atol=1e-7), f"n_cav={n_cav} 时 G=1 未保持 ω̃=ω")

    def test_b3_case2_only_attacker_covered_grid(self):
        """[B3 单测 2] 只有攻击者覆盖（ω_0=0, ω_a=1）、G_a=0.1 的格子，ω̃_a=0.1·ω_a，其余 0.9 权重全归本车。"""
        # 2 车场景：本车 ω_0=0，攻击者 ω_1=1，G_1=0.1
        w_base_2 = torch.tensor([[[[0.0]]], [[[1.0]]]], dtype=torch.float32)
        any_valid = torch.tensor([[[[True]]]])
        g_2 = torch.tensor([[[[0.1]]]], dtype=torch.float32)
        w_out_2 = redistribute_weights_trusted_other(w_base_2, g_2, any_valid=any_valid, tau_trust=0.9)
        self.assertAlmostEqual(float(w_out_2[1].item()), 0.1, places=6)
        self.assertAlmostEqual(float(w_out_2[0].item()), 0.9, places=6)

        # 3 车场景：本车 ω_0=0，诚实车1在该格无覆盖 ω_1=0 (G_1=0.95)，攻击者2独占覆盖 ω_2=1 (G_2=0.1)
        # 此时 sum_{k in T_{-2}} ω_k = ω_0 + ω_1 = 0，按第三版公式 R_2=0.9 必须全部归还本车 ω̃_0！
        w_base_3 = torch.tensor([[[[0.0]]], [[[0.0]]], [[[1.0]]]], dtype=torch.float32)
        g_3 = torch.tensor([[[[0.95]]], [[[0.10]]]], dtype=torch.float32)
        w_out_3 = redistribute_weights_trusted_other(w_base_3, g_3, any_valid=any_valid, tau_trust=0.9)
        self.assertAlmostEqual(float(w_out_3[2].item()), 0.1, places=6)
        self.assertAlmostEqual(float(w_out_3[1].item()), 0.0, places=6)
        self.assertAlmostEqual(float(w_out_3[0].item()), 0.9, places=6)

    def test_b3_case3_weight_conservation_every_grid(self):
        """[B3 单测 3] 随机多车队与极端门控组合下，每个格子上 Σ ω̃ = Σ ω（权重总量守恒，误差 < 1e-6）。"""
        torch.manual_seed(20270110)
        for n_cav in (2, 3, 4, 6):
            raw_logits = torch.randn(n_cav, 1, 32, 32) * 2.0
            valid = torch.rand(n_cav, 1, 32, 32) > 0.35
            valid[:, :, :2, :2] = False  # 强制包含无效区
            valid[0, :, 2:4, 2:4] = False  # 强制包含本车无覆盖而协作者有覆盖区
            any_valid = valid.any(dim=0, keepdim=True)
            safe_logits = torch.where(valid, raw_logits, torch.full_like(raw_logits, -1e4))
            safe_logits = torch.where(any_valid, safe_logits, torch.zeros_like(safe_logits))
            w_base = torch.softmax(safe_logits, dim=0) * any_valid.float()

            # 构造覆盖 [0.0, 0.1, 0.5, 0.89, 0.90, 0.95, 1.0] 的混合门控张量
            choices = torch.tensor([0.0, 0.1, 0.5, 0.89, 0.90, 0.95, 1.0])
            idx_g = torch.randint(0, len(choices), (n_cav - 1, 1, 32, 32))
            gates = choices[idx_g]

            w_tilde = redistribute_weights_trusted_other(w_base, gates, any_valid=any_valid, tau_trust=0.9)
            max_sum_err = float((w_tilde.sum(dim=0) - w_base.sum(dim=0)).abs().max().item())
            self.assertLess(max_sum_err, 1e-6, f"n_cav={n_cav} 权重守恒误差超限: {max_sum_err}")
            self.assertGreaterEqual(float(w_tilde.min().item()), -1e-7)

    def test_b3_case4_single_collab_matches_ego_reclaim_including_095(self):
        """[B3 单测 4] 只有 1 辆协作车时，不论 G_1 取值多少（特别测 G_1 = 0.95），结果与原实现完全一致。"""
        torch.manual_seed(42)
        raw_logits = torch.randn(2, 1, 16, 16)
        valid = torch.rand(2, 1, 16, 16) > 0.2
        valid[:, :, 0, 0] = False
        valid[0, :, 0, 1] = False
        valid[1, :, 0, 1] = True  # 仅协作车覆盖格
        any_valid = valid.any(dim=0, keepdim=True)
        safe_logits = torch.where(valid, raw_logits, torch.full_like(raw_logits, -1e4))
        safe_logits = torch.where(any_valid, safe_logits, torch.zeros_like(safe_logits))
        w_base = torch.softmax(safe_logits, dim=0) * any_valid.float()

        # 逐个测试含 G_1 = 0.95 在内的全部关键门控值，以及随机空间门控图
        for g_val in (0.0, 0.1, 0.5, 0.89, 0.90, 0.95, 0.99, 1.0):
            g_map = torch.full((1, 1, 16, 16), g_val, dtype=torch.float32)
            w_b3 = redistribute_weights_trusted_other(w_base, g_map, any_valid=any_valid, tau_trust=0.9)
            w_old = redistribute_weights_ego_only(w_base, g_map, any_valid=any_valid)
            self.assertTrue(
                torch.allclose(w_b3, w_old, atol=1e-7),
                f"单协作车 G_1={g_val} 时 B3 与原实现不一致！max_diff={(w_b3 - w_old).abs().max().item()}",
            )

        # 额外通过 build_gated_fuse_fn 端到端验证 G_1 = 0.95 在 L0/L1/L2 多尺度融合输出完全一致
        wrapper = _DummyWrapper()
        feat_l0 = torch.randn(2, 16, 256, 256)
        score_l0 = torch.randn(2, 1, 256, 256) + 0.5
        rlen = torch.tensor([2])
        amat = torch.tensor([[[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]]])
        gate_095 = torch.full((1, 1, 256, 256), 0.95, dtype=torch.float32)
        f_on = build_gated_fuse_fn(gate_095, enable_b3=True, tau_trust=0.9)
        f_off = build_gated_fuse_fn(gate_095, enable_b3=False, tau_trust=0.9)
        out_on, _ = f_on([feat_l0], [score_l0], [None], rlen, amat, ["m1", "m2"], wrapper)
        out_off, _ = f_off([feat_l0], [score_l0], [None], rlen, amat, ["m1", "m2"], wrapper)
        self.assertTrue(torch.allclose(out_on[0], out_off[0], atol=1e-6))

    def test_b3_case5_two_collabs_g1_095_g2_02_exact_shares(self):
        """[B3 单测 5] 两辆协作车 G_1=0.95, G_2=0.2：R_2 按 ω_0:ω_1 分给本车和车1，车1被扣 0.05·ω_1 只分给本车！"""
        any_valid = torch.tensor([[[[True]]]])
        gates = torch.tensor([[[[0.95]]], [[[0.20]]]], dtype=torch.float32)

        # 组 A：ω = [0.3, 0.3, 0.4] -> R_1 = 0.015 全归本车；R_2 = 0.32 按 1:1 分给本车(0.16)与车1(0.16)
        w_a = torch.tensor([[[[0.30]]], [[[0.30]]], [[[0.40]]]], dtype=torch.float32)
        out_a = redistribute_weights_trusted_other(w_a, gates, any_valid=any_valid, tau_trust=0.9)
        self.assertAlmostEqual(float(out_a[0].item()), 0.30 + 0.015 + 0.16, places=6)  # 0.475
        self.assertAlmostEqual(float(out_a[1].item()), 0.95 * 0.30 + 0.16, places=6)   # 0.445（绝不含 0.015 任何部分！）
        self.assertAlmostEqual(float(out_a[2].item()), 0.20 * 0.40, places=6)          # 0.080
        self.assertAlmostEqual(float(out_a.sum().item()), 1.0, places=6)

        # 组 B：不等比基准权重 ω = [0.2, 0.6, 0.2] -> ω_0 : ω_1 = 1 : 3
        # R_1 = 0.05 * 0.6 = 0.03（只归本车！若错误按总池分给 T，车1会多拿 0.0225）
        # R_2 = 0.80 * 0.2 = 0.16 -> 本车得 0.16 * (0.2/0.8) = 0.04，车1得 0.16 * (0.6/0.8) = 0.12
        w_b = torch.tensor([[[[0.20]]], [[[0.60]]], [[[0.20]]]], dtype=torch.float32)
        out_b = redistribute_weights_trusted_other(w_b, gates, any_valid=any_valid, tau_trust=0.9)
        self.assertAlmostEqual(float(out_b[0].item()), 0.20 + 0.03 + 0.04, places=6)   # 0.27
        self.assertAlmostEqual(float(out_b[1].item()), 0.95 * 0.60 + 0.12, places=6)   # 0.69
        self.assertAlmostEqual(float(out_b[2].item()), 0.20 * 0.20, places=6)          # 0.04
        self.assertAlmostEqual(float(out_b.sum().item()), 1.0, places=6)

        # 验证自适应攻击直通估计模式前向结果逐位一致且梯度可导
        gates_req = gates.clone().requires_grad_(True)
        out_ste = redistribute_weights_trusted_other(
            w_b, gates_req, any_valid=any_valid, tau_trust=0.9, differentiable_ste=True
        )
        self.assertTrue(torch.allclose(out_ste.detach(), out_b, atol=1e-6))
        out_ste[1].sum().backward()
        self.assertIsNotNone(gates_req.grad)

    # =========================================================================
    # B1 / B2 / A1 / A2 / A4 / A5 模块级单元测试
    # =========================================================================
    def test_b1_min_pool_vs_area_downsample(self):
        """[B1 单测] 粗尺度 L1/L2 门控采用 min_pool，单格/局部强压制不会被 2x2 / 4x4 均值稀释。"""
        wrapper = _DummyWrapper()
        gate_l0 = torch.ones((1, 1, 256, 256), dtype=torch.float32)
        gate_l0[0, 0, 10, 10] = 0.02  # 4x4 块内仅 1 个像素被门控压到 0.02

        # 构造本车特征全 0、协作者特征全 10、双方 score 相等（w_base = [0.5, 0.5]）
        feats, scores = [], []
        for h in (256, 128, 64):
            f = torch.zeros((2, 4, h, h), dtype=torch.float32)
            f[1] = 10.0
            s = torch.ones((2, 1, h, h), dtype=torch.float32)
            feats.append(f)
            scores.append(s)
        rlen = torch.tensor([2])
        amat = torch.tensor([[[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]]])

        fn_min = build_gated_fuse_fn(gate_l0, downsample_mode="min_pool")
        fn_area = build_gated_fuse_fn(gate_l0, downsample_mode="area")
        out_min, _ = fn_min(feats, scores, [None, None, None], rlen, amat, ["m1", "m2"], wrapper)
        out_area, _ = fn_area(feats, scores, [None, None, None], rlen, amat, ["m1", "m2"], wrapper)

        # 在 L2 (64x64) 对应格子 (10//4=2, 10//4=2) 处：
        # min_pool 门控 = 0.02 -> 协作者权重 = 0.5 * 0.02 = 0.01 -> 输出 = 0.10
        # area 门控 = (15*1.0 + 0.02)/16 = 0.93875 -> 输出 = 4.69375（严重漏过局部攻击）
        val_min_l2 = float(out_min[2][0, 0, 2, 2].item())
        val_area_l2 = float(out_area[2][0, 0, 2, 2].item())
        self.assertAlmostEqual(val_min_l2, 0.10, places=4)
        self.assertGreater(val_area_l2, 4.5)

    def test_b2_fg_mask_ego_anchored_rule(self):
        """[B2 单测] Clear 区仅认本车前景，阻断伪造骗取宽松阈值；Sparse/Occluded 保留协作车补盲前景。"""
        p_ego = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
        p_fused = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
        v_idx = torch.zeros((256, 256), dtype=torch.long)  # 默认全 Clear(0)

        # 位置 A (20, 20)：Clear 空地，本车无车(0.01)，攻击者在融合图中伪造出高置信假车(0.92)
        p_fused[0, 0, 20, 20] = 0.92
        # 位置 B (60, 60)：Clear 本车真车(0.85)
        p_ego[0, 0, 60, 60] = 0.85
        # 位置 C (100, 100)：Occluded(2) 盲区，本车看不见(0.01)，诚实协作车补盲检出(0.88)
        v_idx[98:103, 98:103] = 2
        p_fused[0, 0, 100, 100] = 0.88

        fg_v1 = build_fg_mask(p_ego, p_fused, v_idx, fg_prob_tau=0.20, mode="v1")
        fg_v0 = build_fg_mask(p_ego, p_fused, v_idx, fg_prob_tau=0.20, mode="v0")

        self.assertTrue(bool(fg_v0[20, 20].item()))   # 旧规则被伪造骗入前景桶
        self.assertFalse(bool(fg_v1[20, 20].item()))  # B2 新规则成功拦截 Clear 空地伪造！
        self.assertTrue(bool(fg_v1[60, 60].item()))   # Clear 本车真车正常保留
        self.assertTrue(bool(fg_v1[100, 100].item())) # Occluded 诚实补盲前景正常保留

    def test_a1_single_visibility_function(self):
        """[A1 & 审查 3.4 单测] 旧版可见性函数与硬编码 73.853 已彻底移除，强制显式传入正数 clear_pts_tau。"""
        self.assertFalse(hasattr(step4_1_v2_core, "assign_dist5_vis3_indices"))
        self.assertFalse(hasattr(step4_1_v2_core, "DEFAULT_CLEAR_PTS_TAU"))
        dist = torch.tensor([[[[4.0, 15.0, 55.0]]]])
        los = torch.tensor([[[[0.9, 0.9, 0.2]]]])
        exp_pts = torch.tensor([[[[150.0, 20.0, 100.0]]]])
        # 不传 clear_pts_tau 必须立即报 TypeError
        with self.assertRaises(TypeError):
            step4_1_v2_core.assign_dist5_vis3_indices_v2(dist, los, exp_pts)
        # 传非法 clear_pts_tau <= 0 必须报 ValueError
        with self.assertRaises(ValueError):
            step4_1_v2_core.assign_dist5_vis3_indices_v2(dist, los, exp_pts, clear_pts_tau=0.0)
        # require_visibility_v2_taus 缺失字段时必须报 ValueError
        with self.assertRaises(ValueError):
            step4_1_v2_core.require_visibility_v2_taus({})

        d_i, v_i = step4_1_v2_core.assign_dist5_vis3_indices_v2(dist, los, exp_pts, clear_pts_tau=80.0, los_occ_tau=0.50)
        self.assertEqual(d_i.flatten().tolist(), [0, 2, 4])
        self.assertEqual(v_i.flatten().tolist(), [0, 1, 2])  # Clear, Sparse, Occluded

    def test_a2_hetshield_alpha06_no_freeze_and_linear_blend(self):
        """[A2 & 审查 3.5/3.6 单测] 验证 HetShield 默认 α=0.6、隔离不锁死隐状态、且仅在进入金字塔前的传输协议特征层混合唯一一次。"""
        hs = HetShieldHEALDefender(mode="shared", tau_th=0.3, temperature=0.1)
        self.assertAlmostEqual(hs.alpha, 0.6, places=6)
        self.assertFalse(hs.freeze_on_quarantine)

        z0 = torch.randn(1, 64, 256, 256)
        zk_1 = torch.randn(1, 64, 256, 256)
        zk_atk = torch.randn(1, 64, 256, 256) * 8.0  # 强扰动触发 g_k < 0.5
        valid = torch.ones((1, 1, 256, 256))
        hs.step_frame("seq0", 0, z0, [{"cav_id": "c1", "mod": "m1", "feat_warped": zk_1, "valid_ego": valid}], update_state=True)
        h_before = hs._cav_states["c1"]["h"].clone()
        res2 = hs.step_frame("seq0", 1, z0, [{"cav_id": "c1", "mod": "m1", "feat_warped": zk_atk, "valid_ego": valid}], update_state=True)
        h_after = hs._cav_states["c1"]["h"].clone()
        self.assertLess(res2[0]["g_k"], 0.5)
        self.assertFalse(torch.allclose(h_before, h_after), "隔离后隐状态不应锁死在旧帧！")

        # 验证旧版逐金字塔层混合 build_hetshield_fuse_fn 已硬禁用
        with self.assertRaises(RuntimeError):
            build_hetshield_fuse_fn([0.5])

        # 验证 forward_hetshield_protocol_mix 仅在协议特征层（进入 pyramid_backbone 前）执行单次混合：
        # 当 g_1 = 0 时，无论攻击特征多大 (999.0)，送入金字塔的协议输入及最终输出与纯本车特征严格相等！
        wrapper = _DummyWrapper()
        f_ego = torch.ones((1, 8, 32, 32)) * 0.5
        f_mal = torch.ones((1, 8, 32, 32)) * 999.0
        feat_proto_mal = torch.cat([f_ego, f_mal], dim=0)
        feat_proto_cln = torch.cat([f_ego, f_ego], dim=0)
        vmask = torch.ones((2, 1, 32, 32))
        rlen = torch.tensor([2])
        amat = torch.tensor([[[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]]])

        out_zero = forward_hetshield_protocol_mix(wrapper, feat_proto_mal, rlen, amat, ["m1", "m2"], vmask, [0.0])
        in_to_pyr = wrapper.model.pyramid_backbone.last_input_to_pyramid
        self.assertEqual(wrapper.model.pyramid_backbone.get_multiscale_calls, 1)
        self.assertEqual(out_zero["fusion_diag"]["mixing_layer"], "protocol_feature_only_once")
        self.assertFalse(out_zero["fusion_diag"]["pyramid_per_level_mixing"])
        # 送入 pyramid_backbone 的第 1 辆车协议特征已经被混合成 f_ego！
        self.assertTrue(torch.allclose(in_to_pyr[1], f_ego[0], atol=1e-6))

        out_ref = forward_hetshield_protocol_mix(wrapper, feat_proto_cln, rlen, amat, ["m1", "m2"], vmask, [1.0])
        self.assertTrue(torch.allclose(out_zero["cls_preds"], out_ref["cls_preds"], atol=1e-6))

        # 验证审查 3.6：整车回退阈值规则 max(calib_clean_max * 1.15, 1.45) 及下限保护
        fh_tau, fh_rule = compute_frame_high_tau_with_rule(
            {"m1": [0.5, 1.0], "m2": [0.8, 2.0]},
            {"m1": 1.0, "m2": 1.0},
            safety_mult=1.15,
            lower_bound=1.45,
        )
        self.assertAlmostEqual(fh_tau["m1"], 1.45, places=6)   # 1.0 * 1.15 = 1.15 < 1.45 -> 下限 1.45
        self.assertAlmostEqual(fh_tau["m2"], 2.30, places=6)   # 2.0 * 1.15 = 2.30 > 1.45 -> 2.30
        self.assertEqual(fh_rule["formula"], "max(calib_clean_max * 1.15, 1.45)")

    def test_a4_strict_metrics_and_bootstrap(self):
        """[A4 单测] 验证防刷 ASR 近真值分离、盲区 AP Ignore 非盲区真值、诚实压制比例及 Studentized Bootstrap。"""
        def _make_box(cx, cy, w=2.0, l=4.0):
            corners = np.array([
                [cx - l / 2, cy - w / 2, 0.0],
                [cx + l / 2, cy - w / 2, 0.0],
                [cx + l / 2, cy + w / 2, 0.0],
                [cx - l / 2, cy + w / 2, 0.0],
            ], dtype=np.float64)
            return np.vstack([corners, corners])

        gt_boxes = np.stack([_make_box(10.0, 0.0), _make_box(30.0, 0.0)], axis=0)  # 10m 非盲区车，30m 盲区车
        cln_boxes = np.stack([_make_box(10.0, 0.0)], axis=0)
        cln_scores = np.array([0.90])
        # 防御后预测：在 (10, 0) 检出真车(0.90)，在 (11, 0) 距真值 1m 处有框(0.85)，在 (50, 0) 空地处有低分噪声框(0.10 < s_min=0.25)
        def_boxes = np.stack([_make_box(10.0, 0.0), _make_box(11.0, 0.0), _make_box(50.0, 0.0)], axis=0)
        def_scores = np.array([0.90, 0.85, 0.10])

        asr_st = evaluate_frame_asr_strict(
            mode="fabrication",
            candidate_points_xy=[(11.0, 0.0), (50.0, 0.0)],
            clean_pred_boxes_np=cln_boxes,
            clean_pred_scores_np=cln_scores,
            defended_pred_boxes_np=def_boxes,
            defended_pred_scores_np=def_scores,
            gt_boxes_np=gt_boxes,
            s_min=0.25,
            radius_m=3.0,
        )
        # (11, 0) 距真值 (10, 0) 仅 1m <= 3m -> 必须从主伪造分母剔除并计入 near_gt_total=1；
        # (50, 0) 计入主伪造分母 total=1，但预测框分数 0.10 < s_min=0.25 -> succ=0！
        self.assertEqual(asr_st["total"], 1)
        self.assertEqual(asr_st["succ"], 0)
        self.assertEqual(asr_st["near_gt_total"], 1)

        # 盲区 AP：gt[0](10,0) 为非盲区真值，gt[1](30,0) 为盲区真值(blind_gt_indices=[1])
        # 预测框命中 gt[0](10,0) 和 gt[1](30,0)：命中 gt[0] 必须 Ignore（不计 FP），盲区 TP=[1], FP=[0]！
        from opencood.utils import common_utils
        gt_polys = common_utils.convert_format(gt_boxes)
        pred_b = np.stack([_make_box(10.0, 0.0), _make_box(30.0, 0.0)], axis=0)
        pred_s = np.array([0.95, 0.85])
        b_ap = match_blind_spot_ap_ignore_nonblind(pred_b, pred_s, gt_polys, blind_gt_indices=[1], iou_thresh=0.5)
        self.assertEqual(b_ap["tp"], [1])
        self.assertEqual(b_ap["fp"], [0])
        self.assertEqual(b_ap["gt"], 1)

        # 诚实压制统计
        gate = torch.tensor([[0.2, 0.8], [0.4, 0.95]])
        cov = torch.ones((2, 2), dtype=torch.bool)
        vis = torch.tensor([[0, 0], [2, 2]])
        w_b = torch.full((2, 2), 0.5)
        h_st = compute_honest_suppression_stats(gate, cov, vis, whole_fallback=False, w_base_cav_l0=w_b)
        self.assertEqual(h_st["suppressed_grids"], 2)
        self.assertEqual(h_st["total_grids"], 4)

        # Studentized Bootstrap + Holm 校正
        seqs = [f"s{i}" for i in range(8)]
        va = {s: 0.80 + 0.012 * i for i, s in enumerate(seqs)}
        vb = {s: 0.70 + 0.010 * i for i, s in enumerate(seqs)}
        boot_res = studentized_paired_block_bootstrap(
            seqs,
            eval_fn_on_seqs=lambda picked: (float(np.mean([va[s] for s in picked])), float(np.mean([vb[s] for s in picked]))),
            n_boot=400,
            seed=20261120,
        )
        self.assertAlmostEqual(boot_res["diff"], float(np.mean(list(va.values())) - np.mean(list(vb.values()))), places=4)
        holm = holm_bonferroni_adjust({"ep1": 0.001, "ep2": 0.02, "ep3": 0.04, "ep4": 0.20})
        self.assertTrue(holm["ep1"]["reject_null_at_alpha"])
        self.assertFalse(holm["ep4"]["reject_null_at_alpha"])

    def test_a5_disjoint_sequences_and_sparse_bin_merging(self):
        """[A5 单测] 验证 7 条标定序列与 4 条调参序列严格零交集，且帧数 < 20 的稀疏分档自动向上合并。"""
        self.assertEqual(len(FIT_CALIB_SEQ_NAMES_7), 7)
        self.assertEqual(len(FIT_TUNE_SEQ_NAMES_4), 4)
        self.assertTrue(set(FIT_CALIB_SEQ_NAMES_7).isdisjoint(set(FIT_TUNE_SEQ_NAMES_4)))

        # 构造 Occluded 前景帧数仅 5 帧 (< 20) 的模拟分档，验证自动合并至传感器全可见性池
        per_seq = {
            role: {m: {vn: {} for vn in ("Clear", "Sparse", "Occluded")} for m in ("m1", "m2")}
            for role in ("bg", "fg")
        }
        frame_cnts = {
            role: {m: {vn: 25 for vn in ("Clear", "Sparse", "Occluded")} for m in ("m1", "m2")}
            for role in ("bg", "fg")
        }
        rng = np.random.default_rng(123)
        for role in ("bg", "fg"):
            for m in ("m1", "m2"):
                for vn in ("Clear", "Sparse", "Occluded"):
                    for s in FIT_CALIB_SEQ_NAMES_7:
                        per_seq[role][m][vn][s] = [rng.uniform(0.1, 1.0, size=20).astype(np.float32)]
        frame_cnts["fg"]["m2"]["Occluded"] = 5  # 故意制造帧数 < 20 的稀疏桶
        _, q_fg, merged_log = build_fg_bg_vis_quantiles_with_ci_and_merging(
            per_seq, frame_cnts, FIT_CALIB_SEQ_NAMES_7, min_frames=20, n_boot=200, seed=20261120
        )
        self.assertIn("merged_from", q_fg["m2"]["Occluded"])
        self.assertGreaterEqual(len(merged_log), 1)

    def test_b4_gamma_scan_and_vis_mask_gate(self):
        """[B4 & F-P0.1 单测] 验证 γ(Occluded) 扫描单调增强遮挡区新增抑制，且可见性掩码基线门控准确。"""
        v_idx = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)
        # pad 到 256x256 验证 build_visibility_mask_gate
        v_256 = torch.zeros((256, 256), dtype=torch.long)
        v_256[10:20, 10:20] = 1
        v_256[30:40, 30:40] = 2
        g_vm = build_visibility_mask_gate(v_256)
        self.assertEqual(float(g_vm[0, 0, 0, 0].item()), 0.0)     # Clear -> 只用本车 G=0
        self.assertEqual(float(g_vm[0, 0, 15, 15].item()), 1.0)   # Sparse -> 正常融合 G=1
        self.assertEqual(float(g_vm[0, 0, 35, 35].item()), 1.0)   # Occluded -> 正常融合 G=1

        # 验证 γ(Occluded) 从 0.0 增至 0.2 时，在 Occluded 区正向新增冲突处的门控值严格降低（加强伪造压制）
        cos_comb = torch.full((256, 256), 0.95)
        d_drop = torch.zeros((256, 256))
        d_add = torch.full((256, 256), 0.75)
        cov = torch.ones((256, 256), dtype=torch.bool)
        v_occ = torch.full((256, 256), 2, dtype=torch.long)
        g_00 = build_directional_gate_map(
            cos_comb, d_drop, d_add, cov, v_occ, cos_tau=[1.0, 1.0, 1.0], gamma_add=[1.0, 0.3, 0.0]
        )
        g_02 = build_directional_gate_map(
            cos_comb, d_drop, d_add, cov, v_occ, cos_tau=[1.0, 1.0, 1.0], gamma_add=[1.0, 0.3, 0.2]
        )
        self.assertAlmostEqual(float(g_00.mean().item()), 1.0, places=5)
        self.assertLess(float(g_02.mean().item()), 0.65)

    def test_ensure_in_order_assignment_auto_refresh_on_new_sequences(self):
        """[数据适配单测] 当 validate 从旧子集补齐新序列时，自动检测缺失序列并更新模态分配 JSON。"""
        import json
        import shutil
        import tempfile

        tmp_dir = tempfile.mkdtemp(prefix="bud_assign_test_")
        try:
            data_root = os.path.join(tmp_dir, "validate")
            out_json = os.path.join(tmp_dir, "opv2v_val_4modality_in_order.json")
            for s in ("seqA", "seqB"):
                os.makedirs(os.path.join(data_root, s, "101"))
                os.makedirs(os.path.join(data_root, s, "202"))
            ensure_in_order_assignment(data_root, out_json)
            with open(out_json, "r", encoding="utf-8") as f:
                first_map = json.load(f)
            self.assertEqual(sorted(first_map.keys()), ["seqA", "seqB"])

            # 新增第 3 条序列 seqC，再次调用应自动刷新而非直接返回旧 2 序列缓存
            os.makedirs(os.path.join(data_root, "seqC", "303"))
            ensure_in_order_assignment(data_root, out_json)
            with open(out_json, "r", encoding="utf-8") as f:
                refreshed_map = json.load(f)
            self.assertEqual(sorted(refreshed_map.keys()), ["seqA", "seqB", "seqC"])
            self.assertEqual(refreshed_map["seqC"]["303"], "m1")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_gate2_splits_v3_and_m7_objective_and_near06m_and_hs2frame(self):
        """[Gate 2 契约单测] 验证 splits_v3 解析、M7 Python list AP 打分、近场攻击返回 target_centers_hw 与 HetShield 2帧冒烟标定。"""
        import json
        from scripts_v2_mm.eval_standard_suite_v3 import ATTACKS_5, _resolve_eval_subset
        from scripts_v2_mm.step5_1_train_hetshield_heal import calibrate_hetshield_single_step_protocol
        from scripts_v2_mm.v3_m7_select_grid_search import _empty_cand_acc, _score_unified_objective
        from src_v2_mm.heal_attacks import generate_local_near0_6m_attack_heal

        # 1) 验证真实 results_v3/splits_v3.json 子集解析帧数与键名 100% 匹配
        sp_path = os.path.join(REPO_ROOT, "results_v3", "splits_v3.json")
        if os.path.exists(sp_path):
            with open(sp_path, "r", encoding="utf-8") as f:
                sp = json.load(f)
            for sub_name, exp_cnt in (("calib", 3679), ("select", 600), ("dev_fast", 400), ("dev_full", 1113)):
                resolved = _resolve_eval_subset(sp, sub_name)
                tot = sum(len(v["indices"]) for v in resolved["sequences"].values())
                self.assertEqual(tot, exp_cnt, f"{sub_name} 解析帧数 {tot} != {exp_cnt}")
            smoke2 = _resolve_eval_subset(sp, "dev_fast", max_frames=2)
            self.assertEqual(sum(len(v["indices"]) for v in smoke2["sequences"].values()), 2)

        # 2) 验证 v3_m7 _score_unified_objective 接收 Python list 的 tp/fp/score/gt 正常打分
        acc = _empty_cand_acc()
        for m in ("m1", "m2"):
            acc["clean_ap50"][m] = {"tp": [1, 1, 0], "fp": [0, 0, 1], "score": [0.95, 0.85, 0.40], "gt": 2}
            acc["clean_blind_ap50"][m] = {"tp": [1], "fp": [0], "score": [0.80], "gt": 1}
            for a in ATTACKS_5:
                acc["atk_ap50"][a][m] = {"tp": [1, 0], "fp": [0, 1], "score": [0.90, 0.50], "gt": 2}
        nd_ref = {"clean_ap50": {"m1": 1.0, "m2": 1.0}, "clean_blind_ap50": {"m1": 1.0, "m2": 1.0}}
        sc = _score_unified_objective(acc, nd_ref)
        self.assertIn("objective_J", sc)
        self.assertGreater(sc["mean_5atk_ap50"], 0.4)

        # 3) 验证 generate_local_near0_6m_attack_heal 返回非空 target_centers_hw
        class _MiniNearWrapper:
            def __init__(self):
                self.model = types.SimpleNamespace(
                    pyramid_backbone=types.SimpleNamespace(align_corners=False)
                )
                self.w = nn.Conv2d(4, 2, kernel_size=1)
                nn.init.constant_(self.w.weight, 0.1)
                nn.init.constant_(self.w.bias, -3.0)

            def forward_from_protocol(self, feat, rlen, amat, mods):
                # 融合两车特征，并在中心 (128, 128) 即 0~6m 近场制造高置信目标
                fused = 0.5 * (feat[0:1] + feat[1:2])
                cls_p = self.w(fused)
                hot = torch.zeros_like(cls_p)
                hot[:, 0, 127:130, 127:130] = 6.0
                return {"cls_preds": cls_p + hot, "reg_preds": torch.zeros((1, 14, 256, 256), device=feat.device)}

        mw = _MiniNearWrapper()
        feat_in = torch.ones((2, 4, 256, 256))
        vmask = torch.ones((2, 1, 256, 256))
        rlen = torch.tensor([2])
        amat = torch.tensor([[[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]]])
        _, st_near = generate_local_near0_6m_attack_heal(mw, feat_in, rlen, amat, ["m1", "m2"], vmask, steps=2)
        self.assertTrue(st_near["valid"])
        self.assertIn("target_centers_hw", st_near)
        self.assertGreaterEqual(len(st_near["target_centers_hw"]), 1)

        # 4) 验证 HetShield 单步标定在 2 帧极短冒烟序列下非空不崩
        hs_sh = HetShieldHEALDefender(mode="shared", tau_th=0.3).eval()
        hs_ps = HetShieldHEALDefender(mode="persensor", tau_th=0.3).eval()
        e_t = torch.randn(1, 64, 256, 256).half()
        c_t = torch.randn(1, 64, 256, 256).half()
        v_t = torch.ones((1, 1, 256, 256), dtype=torch.bool)
        tiny_store = {
            "seq_smoke": {
                "indices": [10, 11],
                "ego": [e_t, e_t],
                "m1_cav": [c_t, c_t],
                "m1_val": [v_t, v_t],
                "m2_cav": [c_t, c_t],
                "m2_val": [v_t, v_t],
            }
        }
        cal_2f = calibrate_hetshield_single_step_protocol(hs_sh, hs_ps, tiny_store, ["seq_smoke"], stride=1, tail_ratio=0.85)
        self.assertGreaterEqual(len(cal_2f["shared"]["m1"]["s_k"]), 1)
        self.assertGreaterEqual(len(cal_2f["persensor"]["m2"]["s_k"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
