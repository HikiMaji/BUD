# BUD-Fusion 通宵执行与子 Agent 逐关审核规范（M1 ～ M8）

> **权威优先级声明**：本文件整合了《开跑前的完整修改意见》与《开跑前 3 点关键调整》，作为今晚通宵自动流程与各独立审核子 Agent 的**唯一最高执行判据**。
> **分支**：`sprint-cs2027` | **输出目录**：`results_v3/`（严禁覆盖 `results_rq1_v2_mm/`） | **严禁触碰**：C 阶段测试集正式防御评测（`final_test_once`）。

---

## 零、全局硬规则（停机红线、容错策略与人工复核约定）

1. **三项硬停机红线（触发任一必须立即停止后续流程并写入报告）**：
   - ① **HEAL 复现偏差超限**：在 PNG 版 `test` 上用官方配置复现 HEAL 时，任一官方真实报告存在的指标偏差 **> 1.0 个百分点**；
   - ② **单元测试未全过**：`tests/test_sprint_cs2027_cpu.py` 中有任何 1 条单元测试失败；
   - ③ **数据划分存在重叠**：`calib / select / dev` 三组之间，或它们与 `validate / test / test_culvercity` 之间存在任何重叠序列。
2. **非关键环节容错跳过原则（严禁因小失大卡死整晚）**：
   - 除上述 3 项硬停机红线及核心主线外，若其他**非关键环节**出现报错（例如某个对比基线局部报错、某项附加诊断统计异常且快速修复 1 次未果），**必须将完整异常原因与定位记录进日志文件及 `results_v3/SUMMARY.md` 的「异常与跳过记录」小节，跳过该子项并继续往下执行后续任务**，绝不可因此中断整晚主流程。
   - **今晚不接入 `CP-uniGuard`**：`CP-uniGuard` 作为全新融合基线风险与工作量较大，留待明天白天单独接入；今晚 M7 仅搜索 **BUD-Fusion、HetShield（调参版、分传感器版）与 LUCIA**。
3. **子 Agent 逐关审核与结论临时性原则**：
   - 每完成一个节点 `Mk`（`k = 1..8`），必须调用独立审核子 Agent 对照本文件对应 Gate 清单核查代码 diff、日志与输出 JSON；审核 `PASS` 方可进入下一节点，`REJECT` 则修改后重审（非关键旁支遵循第 2 条容错记录）。
   - **子 Agent 审核不能替代人工审核**：明早人工审查代码前，今晚产出的全部实验结论均视为**临时结论**（在 `SUMMARY.md` 中显式标注 `[临时结论 · 待人工代码复核]`）。
   - M6 按验收标准自动决定每项修改的去留以供 M7 继续搜索，但**每一项去留的具体数值对比、95% CI、验收阈值对照及判断依据必须完整写进 `results_v3/SUMMARY.md`**，并保留全部中间 JSON 与标定表供明早复核。
4. **运行稳健性**：
   - 所有长耗时评测必须支持**按序列写盘缓存与断点续跑**；
   - 远端 GPU 任务一律通过 `nohup` 后台运行并将标准输出/错误重定向至 `logs/v3_*.log`；
   - 每完成一个阶段立即更新一次 `results_v3/SUMMARY.md`。

---

## 一、开跑前核心要求基准（完整核对项）

1. **数据与划分（全新 43 序列方案）**：
   - 摒弃旧 `Fit-11 / Dev-8` 分组，不再使用 `train_full19_444`（JPG）和 `Dev-320`；
   - 扫描 `/root/autodl-tmp/datasets/OPV2V/train` 全部可用完整 4 相机 PNG 序列（排除只有 3 个相机的序列如 `2021_09_09_13_20_58`）；
   - **全部 43 条序列重新划分**：按 **“总车数（`2 / 3 / 4 / ≥5`）× 城镇”** 分层随机（固定种子 `20261120`），使 `calib / select / dev`（标定 / 调参 / 开发）三组帧数比例约 **`60% / 25% / 15%`**，且**每组都必须包含总车数 `≥4` 的序列**；
   - 断言三组之间、以及与 `validate / test / test_culvercity`（若目录已存在）之间零重叠；
   - 序列清单（`inventory`：序列名、帧数、总车数、城镇、相机数）与划分、固定抽帧索引统一写入 `results_v3/splits_v3.json`；
   - 所有脚本默认数据目录改为 PNG 版 `/root/autodl-tmp/datasets/OPV2V/{train, validate, test}`；
   - 抽帧规模（固定种子 `20261120`，所有方法共用）：
     - `calib`（标定组）：**`≥ 3,000` 帧**；
     - `select`（调参组）：约 **`600` 帧**（每条序列等间隔抽帧，覆盖全部 `select` 序列）；
     - `dev`（开发组）：快速对比子集 **`400` 帧**（覆盖全部 `dev` 序列），`dev` **全量帧**用于 M8 最终确认。
2. **科学执行顺序（先定方法，再挑参数）**：
   - ① 在 `calib` 标定组上计算标定统计量，**旧前景规则 `v0`** 和**新前景规则 `v1`** 各出一份完整标定表；
   - ② 在 `dev` 开发组 `400` 帧上，用现有冻结参数逐项对比 **A1、B1、B2（修改前后各自配对 `v0 / v1` 标定表！）、B3** 修改前后的标准指标，加上 **B4** 的 $\gamma \in \{0, 0.1, 0.2, 0.3\}$ 扫描和**可见性掩码基线**摸底，按验收标准决定每项去留；
   - ③ 在 `select` 调参组 `600` 帧上，对确定后的最终方法做网格搜索；**HetShield（调参版、分传感器版）**与 **LUCIA** 的阈值也在这一步、用**完全相同的目标函数**搜索（HetShield 论文默认 `0.3` 版不调参；`CP-uniGuard` 今晚不做）；
   - ④ 在 `dev` 开发组**全量帧**上做冻结前的最终确认。
3. **代码审查指出的 6 项必改问题**：
   - **(3.1) 文件名统一**：消除 `a5_bud_calibrator.json` vs `a5_bud_frozen_calibrator.json` 错位，全链路统一标定与调参输出文件名；
   - **(3.2) 脚本名修正**：修正命令与调用中的 `step5_1_train_hetshield_heal.py` 真实脚本名；
   - **(3.3) 内存控制**：彻底杜绝类似 `tune_eval_items` 在内存中缓存 600 帧多状态特征（~115 GB 必爆内存），改为**落盘按序列/按帧磁盘缓存或流式评测**；
   - **(3.4) `τ_N`（`clear_pts_tau`）重标定与显式传参**：严禁依赖旧 JPG 上标定的硬编码默认值 `73.853`，必须在 PNG `calib` 组重新标定并显式传给所有调用处；
   - **(3.5) HetShield 细节**：ConvGRU 在 `calib` 标定组上训练；论文默认 `0.3` 版不调参；经核对原论文 Sec 3.4，特征混合 $\tilde{z}_k = g_k z_k + (1 - g_k) z_0$ **仅在传输协议特征层（进入金字塔前）执行一次**，不在金字塔每一层重复混合；
   - **(3.6) 整车回退阈值规则入档**：`frame_high_tau = max(calib_clean_max * 1.15, 1.45)` 作为方法改动显式记录进标定 JSON 与参数快照，供论文参数表同步更新。

---

## 二、M1 ～ M8 分步执行与子 Agent 审核细则

### M1：全量 43 序列全新分层划分与 PNG 默认路径切换
- **动作**：
  - 编写并运行 `scripts_v2_mm/build_splits_v3.py`，扫描 `/root/autodl-tmp/datasets/OPV2V/train` 全部 43 条 4 相机序列（断言排除 3 相机序列），按 `(总车数档位 2/3/4/≥5 × 城镇区域)` 以种子 `20261120` 全新分层随机生成 `calib / select / dev`（帧比例约 `60% / 25% / 15%`，每组含 `≥4` 车序列），生成共用抽帧索引（`calib ≥ 3000`、`select ≈ 600`、`dev_fast = 400`、`dev_full = 全量`），断言与 `validate / test / test_culvercity` 零重叠，输出 `results_v3/splits_v3.json`；
  - 将 `src_v2_mm` 与 `scripts_v2_mm` 中相关脚本默认数据目录统一指向 `/root/autodl-tmp/datasets/OPV2V/{train, validate, test}`。
- **Gate 1 子 Agent 审核项**：
  - [ ] 是否完全摒弃了固定旧 Fit-11/Dev-8 的捆绑、实现全 43 序列按车数×城镇分层随机？
  - [ ] `calib / select / dev` 帧比例是否接近 `60% / 25% / 15%`、抽帧数满足 `≥3000 / ~600 / 400` 且每组均含 `≥4` 车序列？
  - [ ] 跨集零重叠断言是否执行并通过？默认路径是否已切换至 PNG 目录？

### M2：6 项代码审查问题修复与 CPU 单元测试全检
- **动作**：
  - 修复 (3.1) 文件名统一、(3.2) `step5_1_train_hetshield_heal.py` 脚本名与新划分接入、(3.3) 调参/评测改为磁盘缓存或流式评测防内存溢出、(3.4) 移除 `73.853` 默认值改为强制显式读取传入 PNG 新标定 `clear_pts_tau`、(3.5) HetShield 改为仅在传输协议特征层混合一次（不逐金字塔层混合）、(3.6) `frame_high_tau` 公式元数据入档及各输出 JSON 注入 `meta.claim`；
  - 在 `tests/test_sprint_cs2027_cpu.py` 增补 HetShield 传输层单次混合与无硬编码 `73.853` 检查，本机与远端运行全部单元测试。
- **Gate 2 子 Agent 审核项**：
  - [ ] 标定输出与评测读取文件名是否 100% 一致？
  - [ ] 网格搜索与评测循环中是否已彻底消除内存堆积全量特征张量的问题？
  - [ ] 是否已消除 `73.853` 默认回退？HetShield 是否确系仅在传输特征层混合一次？
  - [ ] 单元测试是否 100% 全部 `OK`？（任一失败触发硬停机）

### M3：PNG 测试集复现 HEAL 官方干净精度（硬停机门槛）
- **动作**：
  - 编写并运行 `scripts_v2_mm/reproduce_heal_official_ap.py`，严格采用 HEAL 官方评测配置（官方检测范围、通信距离与全部可用协作车，绝不用 `use_cav=2` 两车截断），在 PNG 版 `test` 上复现 HEAL 官方真实存在的数字：
    - `m1` 单车（官方 `eval_..._use_cav1.yaml`: `AP@0.5 = 76.45%, AP@0.7 = 66.29%`）
    - `m2` 单车（官方 `stage2/m2_alignto_m1/eval_no_102.4_102.4_epoch25.yaml`: `AP@0.5 = 35.07%, AP@0.7 = 19.29%`）
    - `m1 + m2` 协同（官方 `eval_..._use_cav2.yaml`: `AP@0.5 = 82.63%, AP@0.7 = 72.69%`）
  - 生成 `results_v3/heal_reproduction_check.json` 与对照表。**若任一指标与官方数字绝对偏差 > 1.0 个百分点，立即停机排查！**
- **Gate 3 子 Agent 审核项**：
  - [ ] 评测设置是否严格对齐 HEAL 官方配置（非两车截断）且仅与真实存在的官方数字比较？
  - [ ] 所有对比项绝对偏差是否均 `≤ 1.0` 个百分点？

### M4：全阶段各 2 帧冒烟测试与代码推送 `sprint-cs2027`
- **动作**：
  - 使用独立临时目录 `results_v3/_smoke_2frames/`，对：① `calib` 标定（`v0` & `v1`）、② HetShield 训练与诊断、③ `dev` 标准指标对比 + B4 $\gamma$ 扫描 + 可见性掩码基线、④ `select` 网格搜索（BUD + HetShield + LUCIA）各跑 **2 帧冒烟测试**；
  - 确认全部脚本 0 报错跑通、内存/显存正常、JSON 字段齐全后，清理 `_smoke_2frames/`，在本地与远端对齐提交代码并 `git push origin sprint-cs2027`。
- **Gate 4 子 Agent 审核项**：
  - [ ] 四个阶段 2 帧冒烟是否全部退出码为 `0` 且输出 JSON 字段完整？
  - [ ] Git 提交是否干净（无临时缓存、无敏感代理文件）且已成功推送至 `origin/sprint-cs2027`？

### M5：执行顺序 ① —— 标定组（`≥3,000` 帧）生成 `v0 / v1` 双标定表 & 训练 HetShield
- **动作**：
  - `nohup` 运行标定脚本（日志写入 `logs/v3_m5_calib.log`，按序列缓存）：在 `calib` 组重新标定 PNG 版 `clear_pts_tau`，并分别产出旧前景规则标定表 `results_v3/bud_calib_v0.json` 与新前景规则标定表 `results_v3/bud_calib_v1.json`（均含分档格子数/帧数/序列数、2,000 次序列块 Bootstrap 95% CI、`<20` 帧稀疏桶向上合并日志、`frame_high_tau` 规则说明）；
  - `nohup` 运行 `step5_1_train_hetshield_heal.py`（日志写入 `logs/v3_m5_hetshield.log`）：在 `calib` 组训练 Shared 与 PerSensor ConvGRU，保存权重并输出初始分位数与收敛记录；
  - 更新 `results_v3/SUMMARY.md`。
- **Gate 5 子 Agent 审核项**：
  - [ ] `bud_calib_v0.json` 与 `bud_calib_v1.json` 是否均基于 `≥3,000` 帧生成，且新 `clear_pts_tau`、分档样本量、CI、合并记录齐全？
  - [ ] HetShield ConvGRU 是否仅在 `calib` 组训练并正常收敛？

### M6：执行顺序 ② —— 开发组 `400` 帧逐项方法对比 + B4 扫描 + 可见性掩码与 HetShield 诊断
- **动作**：
  - 保持现有冻结超参档位不变，在 `dev` 快速子集 `400` 帧（按序列断点缓存，同帧共享攻击扰动）上依次评测标准指标全套（附 2,000 次序列块 Bootstrap 95% CI）：
    1. **A1** 前 vs 后（旧不一致统计表 vs 新一致统计表）；
    2. **B1** 前（`area`）vs 后（`min_pool`）；
    3. **B2** 前（`v0` 推理 + `bud_calib_v0.json`）vs 后（`v1` 推理 + `bud_calib_v1.json`）；
    4. **B3** 前（全还本车）vs 后（第三版其他可信车重分配）；
    5. **B4** 扫描 $\gamma(\text{Occluded}) \in \{0.0, 0.1, 0.2, 0.3\}$ 权衡曲线；
    6. **可见性掩码基线**摸底结果与 **HetShield 诊断结果**（诚实 $g < 0.5$ 比例、$\tau_k$ 与 $\sigma_k$ 分布、`m1+m2` 干净盲区 AP）；
  - 按方案各条验收标准得出每项临时去留决定，将前后对比 Markdown 表与详细判断依据写入 `results_v3/SUMMARY.md`（标注 `[临时结论 · 待人工代码复核]`）。
- **Gate 6 子 Agent 审核项**：
  - [ ] 每项前后对比是否严格只变一个因素（尤其 B2 前后标定表配对是否正确、同帧攻击扰动是否一致）？
  - [ ] 每项去留决定的数值判据与推理是否已完整透明地写入 `SUMMARY.md`？

### M7：执行顺序 ③ —— 调参组 `600` 帧最终方法与基线同等目标函数搜索
- **动作**：
  - 锁定 M6 确定的方法组合，在 `select` 调参组 `600` 帧（覆盖全部 `select` 序列，磁盘缓存防爆内存）上做网格搜索，保存最优配置至 `results_v3/bud_tuned_calibrator.json` 并将搜索范围、目标函数表达式、尝试配置总数与完整得分表写入 `results_v3/grid_search_records.json`；
  - 在同一 `select` `600` 帧上、使用**完全相同的目标函数**搜索 **HetShield（调参版全局阈值、分传感器版阈值）** 与 **LUCIA** 的最优阈值（HetShield 论文默认 `0.3` 版固定不调参），写入 `results_v3/baselines_tuned_thresholds.json`；
  - 更新 `results_v3/SUMMARY.md`。
- **Gate 7 子 Agent 审核项**：
  - [ ] BUD、HetShield（Tuned / PerSensor）、LUCIA 是否严格在同一 `select` 600 帧上使用同一目标函数搜索？
  - [ ] 搜索过程是否零触碰 `dev`、`validate` 与 `test`？网格搜索元数据是否完整？

### M8：执行顺序 ④ —— 开发组全量帧最终确认与 7 项交付物汇总推送
- **动作**：
  - 在 `dev` 开发组**全量帧**上，对定稿调参后的 BUD-Fusion、可见性掩码基线、HetShield（默认 `0.3` / Tuned / PerSensor）、LUCIA 运行完整标准指标终审确认，写入 `results_v3/dev_full_final_confirmation.json`；
  - 全面核对并整理齐备以下 **7 项明早交付物** 汇总至 `results_v3/SUMMARY.md`（所有结论标明 `[临时结论 · 待人工代码复核]`），同步回本地 `E:\BUD` 并推送至 `origin/sprint-cs2027`：
    1. `results_v3/splits_v3.json`（43 序列清单与划分）；
    2. HEAL 官方复现对照表（`results_v3/heal_reproduction_check.json`）；
    3. 各标定分档样本量与阈值 95% 置信区间（`results_v3/bud_calib_v1.json`）；
    4. A1、B1、B2、B3、B4 每项修改的前后对比表与临时去留判断依据；
    5. 网格搜索记录（搜索范围、目标函数、尝试配置数，`results_v3/grid_search_records.json`）；
    6. HetShield 诊断结果（$g<0.5$ 比例、$\tau_k/\sigma_k$ 分布、干净盲区 AP）；
    7. 可见性掩码基线摸底结果；
  - **严格停在 C 阶段之前，绝不启动 `final_test_once`。**
- **Gate 8 子 Agent 终审项**：
  - [ ] 上述 7 项交付物文件与 `SUMMARY.md` 表格是否全部真实存在、数值自洽？
  - [ ] `FINAL_TEST_STARTED.json` 是否确未生成（零触碰 C 阶段终测）？
  - [ ] 最新 `results_v3/*.json` 与 `SUMMARY.md` 是否已成功推送到远端 GitHub `sprint-cs2027` 分支？
