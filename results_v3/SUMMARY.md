# BUD-Fusion 投稿冲刺阶段汇总（Computers & Security 2027）

> 分支：`sprint-cs2027`
> 硬性数据隔离规范：A、B 阶段所有方法决策与超参选择仅基于 `train_full19_444` 内的 Fit（7 条标定序列 + 4 条调参序列，零交集）与 Dev-320（8 条独立序列）；`opv2v_val` 仅用于 C 阶段外部统计重标定；`opv2v_test` 仅在 C 阶段带硬锁（`FINAL_TEST_STARTED.json`）一次性执行。
> 单协作车 `m1` 覆盖区双轨规范：
> - **主实验**（主表、防刷 ASR、自适应攻击、`m1` 标定统计）：`m1` 使用自身**完整激光雷达覆盖范围**（`cov_ego_m1_full`），攻击区域不限制在相机视野内；
> - **动机实验**（表 2，`step5_3` / D1，`m1` 与 `m2` 跨模态校准对比）：保留**共同视野**（`cov_ego` + `dev_clean_common`），确保差异仅来自传感器本身。

---

## 1. A + B 阶段代码开发与本机 CPU 单元测试清单

| 任务编号 | 提交记录 | 核心改动摘要 | CPU 单测状态 | GPU Dev 评测状态 | 结论（保留/放弃） |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **A0** | `81d0a06` | 新建 `sprint-cs2027` 分支；`.gitignore` 屏蔽根目录 4 个代理文件；归档冲刺方案与 `results_v3/` 目录 | 已验证（零敏感文件泄露） | — | 基础环境就绪 |
| **A4** | `264f6c6` | 新增 `src_v2_mm/sprint_metrics_v3.py`：防刷 ASR（$s_{\min}$ 过滤 + 3m 真值剔除并单列 `near_gt_fabrication`）、盲区 AP（非盲区真值记 Ignore）、诚实压制率与权重削减比、Studentized Block-Bootstrap + Holm 校正 | 通过（`test_a4`） | 待开 GPU 运行 | 标准指标底座 |
| **A3** | `29dbff2` | `step8_val_recalib_and_final_test.py`：测试启动即写 `FINAL_TEST_STARTED.json` 硬锁，`step5_4` 纳入终测，导出 `FROZEN_PARAMS_SNAPSHOT.json` 与固定均衡子集 `adaptive_subset.json`，修复键名潜伏问题 | 编译通过 | C 阶段执行 | 终测防重跑硬锁 |
| **A1** | `c83a4de` | 删除旧 0.33/0.67 版 `assign_dist5_vis3_indices`，全仓统一唯一可见性函数 `assign_dist5_vis3_indices_v2`；输出 `cov_ego_m1_full` 与共同视野双轨上下文 | 通过（`test_a1`） | 待开 GPU 运行 | 待跑 Dev 前后表 |
| **A5** | `fbaf261` | 新增 `a5_split_and_calib_stats.py` 与 `a5_fit7_calib_tune4_search.py`：Fit 11 序列硬拆分为 7 标定 + 4 调参（零交集），输出分档计数、Bootstrap 95% CI 及 `<20` 帧稀疏桶自动向上合并 | 通过（`test_a5`） | 待开 GPU 运行 | 待跑 Dev 前后表 |
| **A2** | `0f1f822` | `hetshield_heal.py`：默认 $\alpha=0.6$，特征向本车线性混合 $\tilde{z}_k = g_k z_k + (1-g_k)z_0$ 并经 `single_head` 重算占用分数，解除隔离隐状态锁死，标定协议与测试统一 | 通过（`test_a2`） | 待开 GPU 运行 | 待跑 Dev 前后表 |
| **B1** | `26beb43` | `bud_fusion_heal.py`：粗尺度 L1/L2 门控下采样改为 `-F.max_pool2d(-g_src, kernel_size=s, stride=s)`（保留 `area` 开关供 D4-(h) 消融） | 通过（`test_b1`） | 待开 GPU 运行 | 待按 Dev 判据验收 |
| **B2** | `62acaf4` | 新增 `build_fg_mask(..., mode="v1")`：前景判定以本车为准，在 `step4_4` 标定、`a5` 标定、`step8` 重标定及 `step4_5 / 5_2 / 5_4` 推理全链路同步 | 通过（`test_b2`） | 待开 GPU 运行 | 待按 Dev 判据验收 |
| **B3** | `71e6127` | 实现第三版权重重分配 `redistribute_weights_trusted_other`（$T_{-i} = T \setminus \{i\}$，无其他可信车覆盖时归本车），支持 D3/D4 开关与自适应攻击 STE 梯度 | **5/5 强制单测全通过**（含 $G_1=0.95$ 与 $G_1=0.95, G_2=0.2$） | 待开 GPU 运行 | 待按 Dev 判据验收 |
| **B4** | 本次提交 | 新增 `scripts_v2_mm/eval_standard_suite_v3.py`：集成 A0.4 标准指标评测、`b4_gamma_scan`（扫描 $\gamma(\text{Occluded}) \in \{0, 0.1, 0.2, 0.3\}$）与 F-P0.1 `vis_mask_baseline` | 通过（`test_b4`） | 待开 GPU 运行 | 默认 0.0，待实测取舍 |

---

## 2. 远端服务器开 GPU 后按序执行命令清单

将本地 `sprint-cs2027` 分支同步至远端服务器 `/root/autodl-tmp/proj_dir2_hetero_trust` 并确认原版 PNG 数据软链就绪后，严格按以下顺序逐项跑出 `results_v3/*.json` 与 Dev 前后对比表：

```bash
cd /root/autodl-tmp/proj_dir2_hetero_trust

# 0) 先跑一遍 12 项 CPU 单测确认远端环境完全一致
/root/miniconda3/bin/python tests/test_sprint_cs2027_cpu.py

# 1) A1 + A5：在 7 条纯标定序列统计 4 通道表/可见性阈值，在 4 条独立调参序列搜索冻结门控参数
/root/miniconda3/bin/python scripts_v2_mm/a5_fit7_calib_tune4_search.py \
  --out_dir results_v3

# 2) A2：按修正后的 HetShield 协议（α=0.6 + 单步预热 + 真实可见性掩膜）重标定并导出诊断统计
/root/miniconda3/bin/python scripts_v2_mm/step5_1_train_and_calib_hetshield.py \
  --out_json results_v3/a2_hetshield_calib.json

# 3) B1 / B2 / B3 逐步在 Dev-320 上评测 A0.4 全套标准指标（生成消融前后对比表以决定每项去留）
# (a) B1 对照（area vs min_pool）
/root/miniconda3/bin/python scripts_v2_mm/eval_standard_suite_v3.py \
  --mode standard_eval --downsample_mode area --fg_mode v0 --disable_b3 \
  --out_json results_v3/dev_step0_before_b123.json

/root/miniconda3/bin/python scripts_v2_mm/eval_standard_suite_v3.py \
  --mode standard_eval --downsample_mode min_pool --fg_mode v0 --disable_b3 \
  --out_json results_v3/dev_step1_after_b1.json

# (b) B2 对照（同步新前景标定 + 推理 mode=v1）
/root/miniconda3/bin/python scripts_v2_mm/eval_standard_suite_v3.py \
  --mode standard_eval --downsample_mode min_pool --fg_mode v1 --disable_b3 \
  --out_json results_v3/dev_step2_after_b1_b2.json

# (c) B3 对照（启用第三版可信集合 T_{-i} 权重重分配）
/root/miniconda3/bin/python scripts_v2_mm/eval_standard_suite_v3.py \
  --mode standard_eval --downsample_mode min_pool --fg_mode v1 \
  --out_json results_v3/dev_step3_after_b1_b2_b3.json

# 4) B4：扫描 γ(Occluded) ∈ {0.0, 0.1, 0.2, 0.3} 输出权衡曲线
/root/miniconda3/bin/python scripts_v2_mm/eval_standard_suite_v3.py \
  --mode b4_gamma_scan --out_json results_v3/b4_gamma_occluded_scan.json

# 5) F-P0.1 决定性先导对照（可在 Dev 上先行摸底或 C 阶段后在测试集报告）：可见性掩码基线
/root/miniconda3/bin/python scripts_v2_mm/eval_standard_suite_v3.py \
  --mode vis_mask_baseline --out_json results_v3/dev_vis_mask_baseline.json
```
