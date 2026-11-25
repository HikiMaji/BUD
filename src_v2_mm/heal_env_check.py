# -*- coding: utf-8 -*-
"""
HEAL 跨模态运行环境与 EfficientNet 对称填充运行时校验模块。

说明：
HEAL_ref 依赖 efficientnet_pytorch==0.7.0，其中 Conv2dStaticSamePadding 在 stride=2
下采用对称静态零填充 (1, 1, 1, 1) / (2, 2, 2, 2)。若误用 0.7.1 版本，其非对称填充
(0, 1, 0, 1) / (1, 2, 1, 2) 会造成 LSS 3D 视锥投影特征整体偏移与尺度收缩。
本模块用于在训练、推理与评测启动前强制校验环境版本及模型层填充对称性。
"""

import torch
import torch.nn as nn


def assert_heal_env(expected_effnet_ver: str = "0.7.0") -> dict:
    """校验 HEAL 依赖的 efficientnet_pytorch 版本，并返回核心库版本信息。"""
    try:
        import efficientnet_pytorch
    except ImportError as exc:
        raise RuntimeError(
            "未检测到 efficientnet_pytorch，请先在当前环境中安装指定版本："
            f"/root/miniconda3/bin/pip install --no-deps efficientnet_pytorch=={expected_effnet_ver}"
        ) from exc

    effnet_ver = getattr(efficientnet_pytorch, "__version__", "unknown")
    if effnet_ver != expected_effnet_ver:
        raise RuntimeError(
            f"检测到 efficientnet_pytorch 版本为 {effnet_ver}，与 HEAL 官方要求 {expected_effnet_ver} 不一致！"
            "这会导致 EfficientNet 静态填充非对称从而严重破坏 Camera-LSS 空间对齐精度。\n"
            "请执行以下命令降级锁定版本：\n"
            f"  /root/miniconda3/bin/pip install --no-deps --force-reinstall efficientnet_pytorch=={expected_effnet_ver}"
        )

    try:
        import spconv
        spconv_ver = getattr(spconv, "__version__", "unknown")
    except ImportError:
        spconv_ver = "not_installed"

    return {
        "torch": getattr(torch, "__version__", "unknown"),
        "efficientnet_pytorch": effnet_ver,
        "spconv": spconv_ver,
    }


def assert_effnet_symmetric_padding(model: nn.Module, min_layers: int = 5) -> list:
    """检查模型中所有 EfficientNet 静态零填充层是否保持上下/左右对称填充。"""
    checked_layers = []
    for name, module in model.named_modules():
        static_pad = getattr(module, "static_padding", None)
        if not isinstance(static_pad, nn.ZeroPad2d):
            continue

        pad = tuple(static_pad.padding)
        if len(pad) != 4 or pad[0] != pad[1] or pad[2] != pad[3]:
            raise AssertionError(
                f"检测到非对称静态填充层 {name}: padding={pad}，"
                "请检查 efficientnet_pytorch 是否已正确降级至 0.7.0！"
            )
        checked_layers.append((name, pad))

    if len(checked_layers) < min_layers:
        raise AssertionError(
            f"仅检测到 {len(checked_layers)} 个 ZeroPad2d 静态填充层（预期至少 {min_layers} 个），"
            "请确认传入模型中包含 EfficientNet Camera-LSS 编码分支。"
        )

    return checked_layers
