# BUD-Fusion v3 实验运行阶段汇总（`sprint-cs2027`）

> **[临时结论 · 待人工代码复核]**：本文件记录通宵自动流程 `M1 ～ M8` 各阶段的实时产出、数值对比与判断依据。在人工审查代码与日志前，本文件中的所有实验结论均视为临时结论。
> **停机保护状态**：正常（未触发任何硬停机红线；零触碰 C 阶段测试集正式防御评测 `final_test_once`）。

---

## 阶段进度总览

| 节点 | 内容 | 状态 | 独立子 Agent 审核 |
| :--- | :--- | :--- | :--- |
| **M1** | 全量 43 序列全新分层划分（`splits_v3.json`）与 PNG 默认路径切换 | 已完成 | `PASS`（Gate 1 复审通过） |
| **M2** | 6 项代码审查问题修复与 CPU 单元测试全检 | 已完成 | `PASS`（Gate 2 复审通过） |
| **M3** | PNG 测试集复现 HEAL 官方干净精度（硬停机检查） | 已完成 | `PASS`（Gate 3 审核通过） |
| **M4** | 全阶段各 2 帧冒烟测试与代码推送 `origin/sprint-cs2027` | 已完成 | 待 Gate 4 审核 |
| **M5** | 标定组（`≥3,000` 帧）生成 `v0 / v1` 双标定表 & 训练 HetShield | 待开始 | - |
| **M6** | 开发组 `400` 帧逐项定方法对比 + B4 扫描 + 可见性掩码与 HetShield 诊断 | 待开始 | - |
| **M7** | 调参组 `600` 帧最终方法与基线统一目标函数网格搜索 | 待开始 | - |
| **M8** | 开发组全量帧最终确认与 7 项明早交付物汇总推送 | 待开始 | - |

---

## 一、M1 交付：OPV2V 全量原版 PNG 数据集序列清单与全新分层划分

- **清单与划分文件**：`results_v3/splits_v3.json`
- **划分方案**：彻底摒弃旧 `Fit-11 / Dev-8` 捆绑与旧 JPG `train_full19_444`，扫描 `/root/autodl-tmp/datasets/OPV2V/train` 全部 43 条完整 4 相机无损 PNG 序列（排除原生 3 相机序列 `2021_09_09_13_20_58`），按 **`总车数档位 (2 / 3 / 4 / ≥5) × CARLA 城镇`** 以固定种子 `20261120` 进行分层随机划分，并通过 OpenCOOD 数据集加载对齐帧数：

| 分组 | 序列数 | 总帧数 | 帧数占比 | 总车数 `≥4` 序列数 | 车数分布 (`2/3/4/5/6`) | 固定抽帧规模（种子 `20261120`） |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`calib`（标定组）** | 26 | 3,679 | 57.72% | 3 | `16 / 7 / 1 / 1 / 1` | **3,679 帧**（全量 `≥ 3,000`） |
| **`select`（调参组）** | 12 | 1,582 | 24.82% | 3 | `5 / 4 / 3 / 0 / 0` | **600 帧**（覆盖全部 12 条序列） |
| **`dev`（开发组）** | 5 | 1,113 | 17.46% | 2 | `1 / 2 / 1 / 1 / 0` | 快速对比 **400 帧** / 终审全量 **1,113 帧** |
| **训练集合计** | **43** | **6,374** | **100.0%** | **8** | `22 / 13 / 5 / 2 / 1` | - |
| **`validate`（官方验证集）** | 9 | 1,980 | - | - | 零重叠断言通过 | - |
| **`test`（官方测试集）** | 16 | 2,170 | - | - | 零重叠断言通过 | 仅 M3 干净复现（严禁跑防御终测） |

- **跨集零重叠断言**：`calib / select / dev / validate / test / test_culvercity` 两两序列交集严格为 `0`（`disjoint_assertions_passed = true`）。
- **默认数据目录切换**：`src_v2_mm` 与 `scripts_v2_mm` 所有脚本默认数据目录均已切换至无损 PNG 版 `/root/autodl-tmp/datasets/OPV2V/{train, validate, test}`。

---

## 二、M2 交付：6 项代码审查问题修复与全量单元测试

1. **(3.1) 标定与调参文件名统一**：在 `src_v2_mm/step4_1_v2_core.py` 中统一定义自包含标定表常量 `bud_calib_v0.json`、`bud_calib_v1.json`、`bud_tuned_calibrator.json`、`hetshield_calib_v3.json`、`baselines_tuned_thresholds.json`，全仓消除 `a5_bud_calibrator.json` vs `a5_bud_frozen_calibrator.json` 错位，`BUDFusionHEALAuditor` 原生支持单文件自包含加载。
2. **(3.2) HetShield 训练脚本修正与新划分接入**：修正并升级 `scripts_v2_mm/step5_1_train_hetshield_heal.py`，强制读取 `splits_v3.json` 的 `calib` 组 26 条序列（断言与 `select / dev / validate / test` 零交集），论文默认 `0.3` 版固定不调参。
3. **(3.3) 内存控制（杜绝 115 GB 特征堆积）**：`a5_fit7_calib_tune4_search.py` 与新建的 `v3_m5_calibrate.py`、`v3_m6_m8_eval_runner.py`、`v3_m7_select_grid_search.py` 全部采用**逐帧流式多候选评估 + 按序列断点落盘缓存**，彻底移除 `tune_eval_items` 全量特征驻留内存设计（内存峰值 `< 4 GB`）。
4. **(3.4) 移除硬编码 `73.853`，强制新标定 `clear_pts_tau` 显式传参**：移除 `DEFAULT_CLEAR_PTS_TAU`，`assign_dist5_vis3_indices_v2` 与 `extract_frame_context_and_clean` 将 `clear_pts_tau` 设为必填无默认参数，新增 `require_visibility_v2_taus` 与 `calibrate_clear_pts_tau_from_samples` 强制在 PNG `calib` 组重新标定并显式传递。
5. **(3.5) HetShield 仅在传输协议特征层单次混合**：在 `src_v2_mm/hetshield_heal.py` 实现 `mix_hetshield_protocol_features_ego` 与 `forward_hetshield_protocol_mix`，严格按原论文 Sec 3.4 在进入 `pyramid_backbone` 前的协议特征层执行唯一一次 $\tilde{z}_k = g_k z_k + (1 - g_k) z_0$，禁用金字塔 L0/L1/L2 逐层重复混合。
6. **(3.6) 整车回退阈值规则与临时结论元数据入档**：实现 `compute_frame_high_tau_with_rule`，将 `frame_high_tau = max(calib_clean_max * 1.15, 1.45)` 公式、标定干净最大值与论文参数表同步说明写入 JSON，并注入 `meta.claim = "[临时结论 · 待人工代码复核]"`。
7. **单元测试全检**：`tests/test_sprint_cs2027_cpu.py` 共 **14 项**单元测试（含 B3 全部 5 项、无硬编码 `73.853`、HetShield 协议层单次混合、`frame_high_tau` 规则、`splits_v3` 解析契约、M7 目标函数 Python list AP、近场攻击 `target_centers_hw` 及 HetShield 2 帧冒烟标定）本机与远端 100% 通过。

---

## 三、M3 交付：HEAL 官方干净精度复现对照表（硬停机门槛检查通过）

- **结果文件**：`results_v3/heal_reproduction_check.json`（运行日志：`logs/v3_m3_heal_repro.log`）
- **测试数据集**：官方原版无损 PNG `/root/autodl-tmp/datasets/OPV2V/test`（16 条序列，2,170 帧，配套官方 `additional/test/*_bev_visibility.png` 与官方原版 `opencood/modality_assign/opv2v_4modality*.json`；纯冻结底座干净推理，**零触碰 C 阶段防御终测**）。
- **硬停机检查结论**：全部官方真实存在指标的最大绝对偏差仅为 **`0.03` 个百分点**（远低于 `1.0` 个百分点的硬停机红线），`hard_stop_triggered = false`。

### 3.1 与 HEAL 官方真实存在基准 YAML 的逐项对照表

| 评测配置（官方配置，全车队 `max_cav=5` 联合生成场景 GT） | 官方来源文件 | 指标 | 官方基准 (%) | 本次 PNG 复现 (%) | 偏差 (百分点) | 是否 `≤ 1.0` pp |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`m1` 单车**（`comm_range=180`, `range=204.8×102.4`） | `final_infer/eval_..._use_cav1.yaml` | AP@0.3 / **AP@0.5** / **AP@0.7** | 78.54 / **76.45** / **66.29** | 78.53 / **76.44** / **66.32** | -0.02 / **-0.00** / **+0.03** | **通过** |
| **`m2` 单车**（`fusion=no`, `comm_range=70`, `range=102.4×102.4`, `label_type=camera`） | `stage2/m2_alignto_m1/eval_no_102.4_102.4_epoch25.yaml` | AP@0.3 / **AP@0.5** / **AP@0.7** | 45.07 / **35.07** / **19.29** | 45.08 / **35.04** / **19.30** | +0.01 / **-0.03** / **+0.01** | **通过** |
| **`m1 + m2` 异构协同**（`comm_range=180`, `range=204.8×102.4`） | `final_infer/eval_..._use_cav2.yaml` | AP@0.3 / **AP@0.5** / **AP@0.7** | 84.94 / **82.63** / **72.69** | 84.94 / **82.64** / **72.71** | -0.00 / **+0.00** / **+0.02** | **通过** |

### 3.2 官方非截断全协作车配置（`all_cavs`, 全部可用协作车参与融合）补充实测记录

> 注：HEAL 官方 README 声明公开权重包中 `m3`（SECOND）存在输入通道缺陷，且原论文未单列将 `m3/m4` 映射为 `m1/m2` 的全车队数字；按规范“只与真实存在的官方数字比较”，以下无两车截断的全协作车实测结果客观记录备查：

| 全协作车配置（无两车截断，`max_cav=5` 全部可用车参与融合） | 通信距离 | 检测范围 | 实测 AP@0.3 (%) | 实测 AP@0.5 (%) | 实测 AP@0.7 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage-1 `m1` Base 全车队同构融合** | 70 m | `[-102.4, 102.4]²` | 96.90 | **96.35** | **92.71** |
| **Final-Infer `m1 + m2` 全车队异构融合**（官方 `opv2v_4modality.json`） | 70 m | `[-204.8, 204.8] × [-102.4, 102.4]` | 88.64 | **87.05** | **79.40** |
| **Final-Infer `m1 + m2` 全车队异构融合**（官方 `opv2v_4modality.json`） | 180 m | `[-204.8, 204.8] × [-102.4, 102.4]` | 90.30 | **88.98** | **82.40** |

---

## 四、M4 交付：全阶段各 2 帧冒烟测试与代码推送核验

- **核验记录文件**：`results_v3/m4_smoke_verification.json`（远端运行日志：`logs/v3_m4_smoke.log`）
- **独立临时目录**：`results_v3/_smoke_2frames/`（跑通并校验全部 10 份 JSON 后已自动彻底删除，`smoke_temp_dir_cleaned = true`，未污染正式目录）。
- **全阶段 2 帧冒烟执行结果**（全部退出码为 `0`，CPU/GPU 内存平稳 `< 4 GB`，所有输出 JSON 均包含完整字段及 `meta.claim = "[临时结论 · 待人工代码复核]"`）：
  1. **阶段 ① `calib` 标定（2 帧，`v0` + `v1` + `a1_before`）**：`scripts_v2_mm/v3_m5_calibrate.py --max_frames 2` 退出码 `0`，成功生成 `bud_calib_v0.json`、`bud_calib_v1.json`、`bud_calib_a1_before.json`（含新标定 `visibility_v2.clear_pts_tau`、`frame_high_tau_rule`、分档统计与 CI）。
  2. **阶段 ② HetShield 训练与标定（2 帧）**：`scripts_v2_mm/step5_1_train_hetshield_heal.py --max_total_frames 2 --epochs 1` 退出码 `0`，成功训练 Shared/PerSensor ConvGRU 并生成 `hetshield_calib_v3.json`。
  3. **阶段 ③ `dev` 定方法对比 + B4 $\gamma$ 扫描 + 可见性掩码基线 + HetShield 诊断（2 帧）**：`scripts_v2_mm/v3_m6_m8_eval_runner.py --stage m6 --max_frames 2` 退出码 `0`，成功生成 `m6_smoke_report.json` 与 `smoke_m6_decisions.json`。
  4. **阶段 ④ `select` 统一目标函数网格搜索（2 帧，BUD + HetShield Tuned/PerSensor + LUCIA）**：`scripts_v2_mm/v3_m7_select_grid_search.py --max_frames 2` 退出码 `0`，成功生成 `bud_tuned_calibrator.json`、`grid_search_records.json`、`baselines_tuned_thresholds.json`。
  5. **阶段 ⑤（附加）`dev_full` 最终确认模式（2 帧）**：`scripts_v2_mm/v3_m6_m8_eval_runner.py --stage m8 --max_frames 2` 退出码 `0`，成功生成 `m8_smoke_report.json`。
- **代码同步与推送**：已清理全部临时冒烟缓存，排除临时缓存与本地代理杂项文件后提交并推送至 `origin/sprint-cs2027`。

---

## 异常与跳过记录

- 当前无异常或跳过项。
