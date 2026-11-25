# BUD: 面向车联网异构协同感知的安全防御研究

本仓库包含跨模态异构协同感知（`64` 线激光雷达 `PointPillars` + `4` 路环视摄像头 `Camera-LSS`，基于冻结官方 `HEAL` 协议特征底座）安全防御研究的**核心源码、冻结实验结果（OPV2V Dev-320）、统计评测脚本、论文图表与总体研究方案**，便于独立代码审计与实验复查。

> **数据严格隔离声明**：
> - `FIT-11`（11 条父序列，2,312 帧）：仅使用无攻击数据完成离线条件分位数标定与基线训练（Attack-Agnostic，零攻击样本训练）；
> - `DEV-8` / `Dev-320`（8 条未见父序列，320 帧，含 583 辆纯他车客观盲区 GT）：本仓库所有已冻结实测主表与对比实验均在此严格互斥分片上完成，采用 **2,000 次按父序列 Block-Bootstrap 95% CI**；
> - `opv2v_val`（6 序列，965 帧）：独立验证集，留作终测前外部门限重标定；
> - `opv2v_test`（16 序列，2,170 帧）：最终测试集，当前绝对封存未触碰。

---

## 一、快速复查入口（推荐阅读顺序）

如需快速核实本项目的算法实现、攻击构造与实验结论，建议按以下顺序查阅：

1. **总体研究方案与实验汇总**：
   - `docs/master_thesis_proposal_v5_integrated.md`：总体研究方案（v6，含三大研究内容：前级位姿对齐 `AnisoSafe-Align` → 后级特征融合 `BUD-Fusion` → 跨层系统集成）
   - `docs/experiments_summary_all.md`：9 项跨模态冻结实测结果总表与核心结论索引
2. **研究内容二核心算法实现（`BUD-Fusion`）**：
   - `src_v2_mm/step4_1_v2_core.py`：本车 64 线激光雷达射线遮挡包络、预期激光点数与三档物理可见性（`Clear / Sparse / Occluded`）解析
   - `src_v2_mm/bud_fusion_heal.py`：四维条件偏差校准、5×5 容差有向语义偏差、细粒度空间软门控、整车级高门限回退与权重差额归还本车融合
3. **攻击构造与对比基线实现**：
   - `src_v2_mm/heal_attacks.py`：全局 PGD、定向目标移除、虚假目标伪造、遮挡区目标伪造、近场目标移除等白盒协议特征攻击
   - `src_v2_mm/baselines_robosac_lucia_heal.py`：`ROBOSAC` (ICCV'23) 与 `LUCIA` (USENIX Security'25) 在 `HEAL` 底座上的官方忠实复现
   - `src_v2_mm/hetshield_heal.py`：`HetShield` (arXiv:2609.17856) 64 通道 ConvGRU 时空一致性防御复现（注：`step5_3` 为无锁死单步预热精确评测；`step5_1` 连续标定阶段存在待修复的 `tau_th=0.3` 隐状态锁死偏高问题，详见代码注释）
4. **核心冻结实测结果 JSON**：
   - `results_rq1_v2_mm/step4_5_bud_frozen_320_report.json`（2 车异构对全 9 工况 AP@0.5 与 583 辆盲区召回）
   - `results_rq1_v2_mm/step5_2_mixed_report.json`（3–6 车混合车队 9 方法全基线分层 AP@0.5）
   - `results_rq1_v2_mm/step5_4_targeted_asr_report.json`（定向目标移除与虚假伪造攻击成功率 ASR 全基线对比）
   - `results_rq1_v2_mm/step5_3_hetero_unified_threshold_evidence.json`（跨模态统一阈值分布错位与等误报率对比实证）
   - `results_rq1_v2_mm/step6_aniso_box_error_stats.json`（研究内容一跨模态径向/横向定位误差各向异性实测）
   - `results_rq1_v2_mm/step7_latency_breakdown.json`（RTX 4090 CUDA Event 单帧推理耗时拆解）

---

## 二、仓库目录与文件作用总览

### 1. `src_v2_mm/` —— 跨模态主线核心源码（`HEAL` `m1 LiDAR + m2 Camera`）
| 文件名 | 核心作用 |
| :--- | :--- |
| `heal_base_wrapper.py` | 冻结官方 `HEAL` 跨模态底座封装：数据加载、协议特征提取（`64×256×256`）、多车混合传感器交替分配、自定义门控融合钩子与 3D 框解码 |
| `step4_1_v2_core.py` | 物理可见性解析核心（射线遮挡包络 $\operatorname{los}(x)$ 与预期激光点数 $N_{\text{exp}}(x)$）、距离×可见性分档及多尺度偏差图提取 |
| `bud_fusion_heal.py` | `BUD-Fusion` 完整审计器与两级门控融合实现：四维条件校准、5×5 容差有向语义偏差、前景/空地双分档软门控、整车高门限回退与 `WholeDrop` 消融基线 |
| `heal_attacks.py` | 传输层协议特征白盒攻击生成器：全局 PGD（`0.5σ/1.0σ/2.0σ/ε=4.0`）、定向目标移除、虚假目标伪造、遮挡区目标伪造、近场移除 |
| `baselines_robosac_lucia_heal.py` | `ROBOSAC`（2D 框 Jaccard 子集共识采样）与 `LUCIA`（`avg_pool2d(32)` + L2 归一化 + L1 距离 + `1-softmax` 信任加权）在 `HEAL` 上的忠实复现 |
| `hetshield_heal.py` | `HetShield` 64 通道 ConvGRU 时序预测器与时空一致性信任门控封装（支持 Shared / PerSensor 模式与跨序列状态复位） |
| `hetshield_baseline.py` / `hetshield_persensor.py` | 早期 LiDAR 阶段 `HetShield` 基线与分模态变体实现 |
| `eval_stats_utils.py` | 2,000 次按父序列整块抽样（Block-Bootstrap）成对差值与 95% 置信区间统计工具 |
| `git_provenance.py` | 实验结果 JSON 自动注入 Git Commit / 分支 / 脏状态溯源元数据工具 |
| `heal_env_check.py` / `test_params.py` | 远端环境依赖检查与参数自检辅助脚本 |

### 2. `scripts_v2_mm/` —— 跨模态实验、标定、评测与出图脚本
| 脚本名 | 核心作用 |
| :--- | :--- |
| `repro_heal_official_val.py` | 复现验证官方 `HEAL` 预训练权重在 `m1/m2` 上的基准 3D 检测精度 |
| `step3_verify_heal_attacks.py` | 验证各类白盒特征攻击在 `HEAL` `m1/m2` 上的破坏有效性 |
| `step4_pre_analyze_benign_deviation.py` | 分析无攻击跨模态特征偏差随距离、模态与遮挡的统计分布 |
| `step4_1_v2_fit_calibrate.py` / `step4_1_v2_eval_once.py` | 在 `FIT-11` 上离线标定 `BUD-Fusion` 条件偏差分位数并执行初测 |
| `step4_1_visibility_spike_and_4tier_ablation.py` | 四档条件校准（`Tier ①` 统一至 `Tier ④` 模态×距离×可见性）消融实验 |
| `step4_2_fit_gating_calibrator.py` / `step4_3_eval_bud_fusion_320.py` | 初版软门控标定与 Dev-320 开发集评估 |
| `step4_4_fit_blind_diag_and_recalib.py` | 诊断盲区误伤与遮挡区伪造原因，完成“预期点数可见性 + 前景/空地双分档 + 整车高门限回退”冻结标定 |
| `step4_5_eval_bud_frozen_320.py` | **主实验 ①**：2 车异构对全 9 工况 × 5 方法在 Dev-320 上的冻结端到端验收（AP@0.5 / AP@0.7 / 583 辆盲区召回） |
| `step5_1_train_hetshield_heal.py` | 在 `FIT-11` 上训练 `HetShield` 64ch ConvGRU 并标定门限 |
| `step5_2_eval_multicav_and_baselines.py` | **主实验 ②**：3–6 车混合车队 9 方法全基线分层评测（含单/双攻击者、诚实车 FPR 与权重零连坐统计） |
| `step5_3_hetero_unified_threshold_evidence.py` | **立足点实证**：与阈值无关的跨模态分布错位、`LUCIA` 同帧信任失衡及等误报率下统一判据 vs 条件判据对比 |
| `step5_4_eval_targeted_asr.py` | **主实验 ③**：全基线定向目标移除与虚假目标伪造攻击成功率（ASR）分层统计 |
| `step6_aniso_box_error_analysis.py` | **研究内容一实测**：统计真实 `m1/m2` 框中心沿径向/横向定位误差并拟合各向异性曲线 |
| `step6_probe_m3_second.py` | `m3`（SECOND）权重兼容性限时探路与止损记录 |
| `step7_latency_breakdown.py` | RTX 4090 CUDA Event 精确单帧推理延迟拆解（对比 `BUD-Fusion` 与 `ROBOSAC / LUCIA / HetShield`） |
| `step8_val_recalib_and_final_test.py` | 封测专用脚本：`opv2v_val` 外部门限重标定与双重锁控 `opv2v_test` 一次性终测（尚未触发终测） |
| `plot_ppt_frozen_figures.py` / `plot_ppt_three_story_separate.py` / `plot_ppt_motivation_dilemma.py` / `plot_ppt_rq1_cn.py` | 自动读取冻结 JSON 绘制 300 DPI 中文学术图表（零手填数据） |
| `test_png_vs_jpg_heal*.py` / `stream_download_and_extract_q95.py` / `run_overnight_pipeline.sh` | 图像压缩格式 A/B 精度验证、数据集流式解压与远端流水线脚本 |

### 3. `results_rq1_v2_mm/` —— 冻结实测报告与标定产物
* 存放 `step3` 至 `step7` 全部实验的冻结 JSON 报告、标定参数表（`step4_4_bud_frozen_calibrator.json` 等）、数据集严格互斥划分清单（`step4_1_v2_split_plan.json`、`step4_3_eval320_split.json`）与实验日志。

### 4. `ppt_figures/` —— 300 DPI 高清结果图
* 存放由上述绘图脚本自动生成的全套中文高清对比图（`fig0~fig4`、`fig_story0~fig_story4`）。

### 5. `docs/` —— 研究方案与实验汇总文档
* `master_thesis_proposal_v5_integrated.md`：硕士学位论文总体研究方案（v6 定稿）
* `experiments_summary_all.md`：全部 9 项跨模态实测结果汇总
* `rq1_proposal_v1.md` / `rq1_proposal_v2_final.md`：早期研究点方案演进归档

### 6. `src/` & `scripts/` & `logs/` —— 前期纯 LiDAR 四异构探索归档
* 前期在纯激光雷达四异构底座（64/32/16 线）上开展的发送端谱约束适配器 `LB-FPA` 探索代码、运行脚本与日志，与跨模态主线独立归档。
