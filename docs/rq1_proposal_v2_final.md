# 硕士学位论文研究点一完整深化方案（v2 专家审查优化定稿版）

## 0. 课题整体定位与三点递进体系（全面脱敏与边界加固版）

- **学位论文大题目**：《面向开放异构车联网的不确定性解耦与可信协同感知研究》
- **学科属性**：网络空间安全（车联网开放多智能体协同感知安全与可信人工智能）
- **核心立论边界**：
  现有车联网协同感知安全研究普遍建立在**封闭同构假设**之上，即默认所有协作车辆拥有相同的传感器感知密度、统一的骨干编码架构以及同分布的稠密鸟瞰图（BEV）特征，进而依赖跨车特征残差、抽样共识或受污特征生成修补实现防御。然而，真实开放车联网（Open V2X）中跨品牌、跨代际车辆搭载着截然不同的传感器物理规格（如 64 线高配激光雷达、32 线中配雷达、16 线低成本稀疏雷达等）与异构私有三维编码器（如柱状体素网络 `PointPillars`、三维稀疏卷积网络 `SECOND` 等），需经**共享协议空间（Shared Protocol BEV Space）**完成中层特征交互。在开放异构环境下，“传感器物理稀疏退化”、“跨架构表征域偏移”与“恶意对抗特征篡改”三者在协议空间深度纠缠，导致传统同构安全防御陷入高误报误杀良性弱配车与放宽门限漏放攻击的结构性死局。
- **全篇三点递进架构（由表征融合层 $\to$ 多视角空间博弈层 $\to$ 跨帧动态信誉层严密递进）**：
  1. **【研究点一（本方案深化与预开题核心实证点）】**：**面向开放异构协同的谱有界协议对齐与双源不确定性解耦融合方法**——聚焦协议域表征对齐与单帧非协调威胁防御，解决无约束适配器放大对抗扰动、单维置信度语义错位、以及良性异构不确定性与恶意特征擦除/伪造攻击纠缠引发的高误报与盲区补盲损失难题。
  2. **【研究点二（后续递进点）】**：**面向盲区流形潜伏与异构拜占庭合谋的跨视角遮挡相容性子群甄别方法**——针对研究点一在单帧信息论上的明确能力边界（即攻击者在共视区伪装诚实、仅在本车纯遮挡盲区重放真实车辆流形特征片段，或多恶意节点协同合谋），利用中层协议特征的前后景射线遮挡因果传递相容性与异构图谱团簇分离实现深层甄别。
  3. **【研究点三（后续递进点）】**：**能力-信誉双轨空时解耦演化的在线协同感知机制与系统验证**——将静态传感器物理能力画像与动态诚实行为信誉在时间轴上彻底解耦，抵御跨帧慢速渐进擦除与间歇投毒攻击，并完成开放异构多场景原型系统集成验证。

---

## 1. 现有方案不足深度分析

### 1.1 开放异构协同感知方法：无约束协议适配器放大对抗扰动，共训置信度易遭劫持
以 `MPDA` (AAAI'23)、`HEAL` (ICLR'24)、`PnPDA` (ECCV'24) 与 `STAMP` (ICLR'25) 为代表的开放异构方法，通过后向对齐或轻量级协议适配器将异构特征映射至统一协议空间，再由自注意力（`AttFusion`）或分类头置信度（`Where2comm`）完成融合。在对抗威胁下存在三大缺陷：
1. **无约束适配器与稀疏批归一化引发跨域扰动非线性放大**：现有适配器堆叠无谱约束的卷积层与 `BatchNorm2d`（BN）。由于激光雷达 BEV 特征高度稀疏，大量通道统计方差极小（$\widehat{\text{Var}}_c \ll 10^{-3}$），BN 的方差倒数缩放因子 $|\alpha_c|/\sqrt{\widehat{\text{Var}}_c + \epsilon}$ 与大奇异值卷积核叠加，使得局部 Lipschitz 常数急剧膨胀。正如 `HetPoison` (2026) 所证实，异构性本身并不能天然防御攻击，无约束适配器反而会将单步迁移扰动或传感器离群噪点放大为大面积协议空间伪影。
2. **背景主导对齐稀释前景语义，遗留跨架构表征微偏移**：BEV 网格中超过 $98\%$ 为空背景，全图均等回归损失被空背景主导，导致 `SECOND`（3D 稀疏体素卷积）与 `PointPillars`（2D 柱状散射）在目标边缘及远距稀疏区仍残留显著的跨架构表征偏差。
3. **共训分类置信度混淆“前景存在性”与“观测可靠性”，且无界点积注意力易被吞噬**：`Where2comm` 直接取分类头输出 $\max \sigma(\text{cls\_head}(Z))$ 充当置信度，既把“近距清晰无车空地”误判为低置信，又使伪造攻击可通过拉高分类响应直接窃取最高权重；同时标准 `AttFusion` 采用无归一化指数点积 $\exp(q_0^\top k_i / \sqrt{C_p})$，恶意节点只需沿本车特征方向推高协议特征范数即可产生指数级权重吞噬（`LUCIA/SOMBRA` USENIX'25）。

### 1.2 现有协同感知安全防御：强同构同分布假设导致高误报自毁与盲区补盲丧失
以 `ROBOSAC`、`LUCIA` (USENIX'25)、`CP-uniGuard` (2026, 基于 PASAC 抽样与检测框一致性损失 `CCLoss`)、`GLST` (2026) 为代表的防御方法，在开放异构场景下存在三大失灵机制：
1. **良性异构退化与恶意篡改混同引发高误报（False Quarantine）**：16 线稀疏雷达在中远距离存在严重的物理偶然退化，异构骨干存在固有跨域微偏移。同构一致性判据将此类良性异构分歧一概判为恶意冲突：门限严则大比例误杀良性弱配车，丧失协同收益；门限宽则被隐蔽攻击穿透。
2. **忽视非对称可见性与局部遮挡，对称残差扼杀“遮挡补盲”核心价值**：在本车被前车遮挡的盲区，本车为静默背景而近距良性协作者传来强前景特征；反之在共视距离内，良性协作者也可能被其他障碍物局部挡住视线而发来静默背景。缺乏双边可见度与静默非对称建模的残差过滤，要么抹杀盲区真车补盲，要么误杀局部受遮挡的良性协作者。
3. **全局高分位数对单目标局部攻击失明，且抽样重推理延迟高昂**：单辆汽车仅占约 $18$ 个 BEV 网格（不足全图 $35,200$ 网格的 $0.06\%$），全图均值或 95% 分位数统计对精准单车擦除/伪造完全失明；而 `ROBOSAC/PASAC` 需反复抽样子集重跑检测头计算 `CCLoss`，难以满足车载毫秒级实时性需求。

---

## 2. 想要解决的核心科学问题与威胁模型

### 2.1 核心科学问题
在**冻结基准检测模型、零攻击样本训练（Attack-Agnostic Training）且单次检测头前向推理**的约束下，如何构建具有严格数学 Lipschitz 上界的异构协议对齐映射，并在协议空间内生解耦**“几何可见度驱动的物理偶然不确定性”**与**“球面流形偏离驱动的跨域认知不确定性”**，从而在不误伤良性异构弱配车、无损保留遮挡盲区补盲增益的同时，精准阻断局部及全局恶意特征擦除、伪造与跨架构迁移污染（含知晓防御结构的白盒自适应攻击）？

### 2.2 威胁模型与能力边界界定
- **系统设定**：本车（Ego，$i=0$，搭载 64 线 LiDAR 与预训练冻结 `PointPillars` 基准编码器及检测头，自身软硬件可信但受物理视距与视线遮挡限制）与 $K$ 辆开放异构协作者（$i=1,\dots,K$，涵盖不同物理线束如 64/32/16 线及异构三维编码器如 `SECOND`、粗粒度 `Pillar` 等）在统一协商参考网格系下交换共享协议特征 $Z_i \in \mathbb{R}_{\ge 0}^{C_p \times H_p \times W_p}$（并附带标准 V2X 位姿 $T_i$ 与公开传感器线束规格标识 $m_i$）。
- **攻击者模型**：协作车辆中存在被攻陷的恶意节点，可在发送端或协议空间构造有界或局部恶意特征扰动 $\Delta Z_i$，涵盖五类典型攻击：① **梯度目标擦除攻击（PGD Object Removal）**；② **随机道路锚点伪造鬼影攻击（Ghost Spoofing，含优化伪造与跨帧特征块移植）**；③ **跨异构代理模型单步迁移攻击（`HetPoison`-style Transfer Attack）**；④ **操纵特征幅值试图欺骗不确定性权重的伪装攻击**；⑤ **完全知晓本防御全部模块与门限的白盒自适应潜伏攻击（White-box Adaptive Attack）**。
- **诚实能力边界声明**：研究点一基于单帧双边物理可见度校准与良性球面流形评分，能够完备防御上述五类波及共视核验域、违背物理距离能力下界或偏离良性协议球面流形的攻击；对于“攻击者在共视区 100% 诚实、仅在本车纯遮挡盲区重放真实干净车辆协议特征片段（In-Manifold Pure Blind-Spot Replay）”以及“多恶意节点协同合谋”，单帧局部信息论上不可完全可辨，该高阶威胁作为研究点二的核心攻关对象无缝衔接。

---

## 3. 两大核心技术难点

1. **难点一：稀疏非负协议空间中“严格 Lipschitz 有界映射”与“防坍缩双源不确定性解耦”的协同建模难题**
   如何在消除稀疏 `BatchNorm` 奇异值爆炸并兼容冻结检测头非负特征锥的前提下保证异构对齐精度；同时解决“平坦空旷路面无激光落点被误判为盲区”的物理真值塌陷问题，以及“99% 空背景网格导致流形原型坍缩误伤真车前景”的表征退化问题。
2. **难点二：抗双向自适应欺骗的“单目标微簇异常检出”与“非对称遮挡补盲保护”融合难题**
   恶意节点可通过操纵特征统计量放大冲突分母或推高点积注意力实施自适应绕过，且单目标篡改仅占全图不足 $0.1\%$ 网格；如何在物理隔离外部不可信特征对检测分母干扰的前提下，同时实现 18 网格级局部篡改捕获、良性中远距遮挡静默豁免以及本车纯盲区高可靠补盲的无损放行。

---

## 4. 两大核心创新点

1. **创新点 1.1：提出真二维算子谱有界异构协议适配器与“目标层光束可见度—平衡球面流形”双源不确定性解耦机制**
   摒弃传统无约束卷积与稀疏批归一化，设计基于精确二维卷积伴随幂迭代与凸残差连接的严格 Lipschitz 有界非负协议适配器（LB-FPA），从理论上锁定跨域扰动放大上界；首创极坐标目标高度层有效光束可见度先验（准确区分近中距清晰空旷路面与真实遮挡盲区）与前景平衡采样的单位超球面原型评分器（消除零背景坍缩与幅值干扰），实现无需攻击样本的物理偶然与流形认知双源不确定性解耦。
2. **创新点 1.2：提出几何分母隔离的非对称白化冲突审计与有界余弦防稀释软门控融合算法**
   构建协作者侧由纯几何距离-线束先验物理锁定的非对称双边白化冲突度量，结合 $3\times 3$ 连通簇池化 `Top-K` 单车尺度帧级审计、非静默前景防稀释门与有界温度余弦注意力机制，彻底封死“自证高方差逃逸”与“指数点积权重吞噬”两类自适应漏洞，在单次检测头前向传播内实现良性异构零误杀增益与强对抗鲁棒性的统一。

---

## 5. 提出的完整算法设计（`BUD-Fusion`: Bounded Uncertainty-Disentangled Fusion）

设统一共享协议 BEV 空间维度为 $Z_i \in \mathbb{R}_{\ge 0}^{C_p \times H_p \times W_p}$（OPV2V 标准网格 $H_p = 100, W_p = 352$，物理分辨率 $0.8\text{m}\times 0.8\text{m}$，$C_p = 384$）。本车 64 线 `PointPillars` 骨干与检测头 $\text{Head}_{\text{ref}}$ 预训练权重全程冻结（注：利用单车 `record_len=[1]` 时 `AttBEVBackbone` 内自注意力严格等价于恒等映射且 `deblocks` 末端为 `ReLU` 的结构特性，锚点协议特征天然位于非负锥 $\mathbb{R}_{\ge 0}^{384\times 100\times 352}$）。

### 算法模块一：真二维算子谱有界前景感知协议适配器（LB-FPA）
1. **严格 Lipschitz 有界凸残差架构（无 `BatchNorm` 爆炸漏洞）**：
   对第 $m$ 类异构编码器输出的原生 BEV 特征 $F_i^{\text{raw}} \in \mathbb{R}^{C_m \times H_p \times W_p}$，构建由输入投影层、$B$ 个凸残差块（Convex Residual Blocks）与输出非负映射组成的适配器 $\mathcal{A}_m$。对任意二维卷积核 $W_l$，通过真二维卷积算子幂迭代（交替执行 $u \leftarrow \text{Conv2d}(v, W_l)/\|\cdot\|_2$ 与 $v \leftarrow \text{ConvTranspose2d}(u, W_l)/\|\cdot\|_2$）计算精确空间卷积算子谱范数 $\sigma_{\text{op}}(W_l)$，定义有界仿射卷积 $\tilde{\mathcal{K}}_l(X) = \gamma_l \frac{\text{Conv2d}(X, W_l)}{\sigma_{\text{op}}(W_l)} + b_l$。第 $b$ 个凸残差块（取 $\alpha_r = 0.5$）与末端非负锥对齐输出定义为：
   $$H^{(b)} = (1 - \alpha_r) H^{(b-1)} + \alpha_r \cdot \text{ReLU}\Big( \tilde{\mathcal{K}}_{b,2}\big( \text{ReLU}(\tilde{\mathcal{K}}_{b,1}(H^{(b-1)})) \big) \Big), \quad Z_i = \text{ReLU}\big(\tilde{\mathcal{K}}_{\text{out}}(H^{(B)})\big)$$
   **定理保证**：由于彻底移除了方差除法归一化且 `ReLU` 为 1-Lipschitz 算子，对任意输入扰动 $\Delta F$，协议域输出扰动严格满足紧上界：
   $$\| \mathcal{A}_m(F + \Delta F) - \mathcal{A}_m(F) \|_2 \le \gamma_{\text{stem}} \big(1 - \alpha_r + \alpha_r \gamma^2\big)^B \gamma_{\text{out}} \|\Delta F\|_2$$
2. **前景增强后向协议对齐训练**：
   结合真值框高斯前景掩膜 $M_{\text{fg}}(x,y) \in [0, 1]$，联合优化前景加权协议蒸馏损失与冻结检测头任务损失：
   $$\mathcal{L}_{\text{align}} = \frac{1}{H_p W_p}\sum_{x,y}\big(1 + \alpha_{\text{fg}} M_{\text{fg}}(x,y)\big)\Big( \|Z_i(x,y) - Z_i^{\text{teacher}}(x,y)\|_1 + \big(1 - \cos(Z_i(x,y), Z_i^{\text{teacher}}(x,y))\big) \Big) + \mathcal{L}_{\text{det}}\big(\text{Head}_{\text{ref}}(Z_i), Y\big)$$
- **模块作用**：在不重训本车基准模型的前提下实现跨架构/跨线束特征高精度协议对齐，并从结构上锁死对抗扰动的跨域放大率。

### 算法模块二：目标层光束可见度与平衡球面流形双源不确定性解耦器（DUID）
本模块纯作为接收端不确定性评分器（**绝不将重构特征混入融合 Value，彻底杜绝特征恢复伪影与假车美化风险**），解耦输出两类物理语义正交的不确定性图：

1. **极坐标目标层有效光束可见度驱动的偶然不确定性（解决空旷路面真值塌陷，零三维体素裁剪）**：
   - **本车可信可见度先验 $\sigma_{\text{alea},0}^2(x,y)$**：将水平 $360^\circ$ 离散化为方位微扇区 $\theta_{\text{bin}}$，提取本车点云在立体障碍物高度层 $z \in [-1.3\text{m}, 0.8\text{m}]$（滤除平坦路面回波）的首达遮挡距离包络 $r_{\text{occ}}(\theta_{\text{bin}})$。对任一协议网格 $(x,y) \leftrightarrow (r, \theta)$：若 $r \le r_{\text{occ}}(\theta) + \Delta_{\text{margin}}$（视线无遮挡，含近中距清晰空旷车道），按传感器垂直俯仰角分布解析计算该距离处穿过 $1.5\text{m}$ 标准目标柱体的有效激光束环数 $n_{\text{beam}}(r, m_0)$；若 $r > r_{\text{occ}}(\theta) + \Delta_{\text{margin}}$（位于障碍物遮挡阴影后），令有效光束数为 $0$。据此定义：
     $$\sigma_{\text{alea},0}^2(x,y) = 1 - \min\left(1,\; \frac{\ln\big(1 + n_{\text{beam}}^{\text{vis}}(x,y)\big)}{\ln(1 + N_{\text{ref}})}\right) \in [0, 1]$$
   - **协作者几何锁定先验 $u_{\text{geom}}(d_i, m_i)$ 与偶然度估计**：在训练集上按距离分桶与传感器线束类型 $m_i$ 离线标定不可篡改的纯几何退化包络 $u_{\text{geom}}(d_i(x,y), m_i) \in (0, 1]$，并训练轻量预测头输出协作者局部偶然不确定性 $\hat{u}_{\text{alea},i}(x,y)$（以对应目标层光束退化真值为监督）。

2. **前景平衡紧致单位超球面原型锥驱动的认知不确定性 $\sigma_{\text{epis},i}^2(x,y)$（防零背景坍缩、防子空间过完备与幅值爆炸）**：
   - 先扣除协议特征空间背景均值提取净激活响应 $\tilde{z}_i(x,y) = Z_i(x,y) - \bar{b}_i$（其中 $\bar{b}_i = \frac{1}{H_p W_p}\sum_{u,v} Z_i(u,v)$），定义非静默激活指示器 $\mathcal{A}_{\text{act}, i}(x,y) = \mathbb{I}(\|\tilde{z}_i(x,y)\|_2 > \eta_{\text{act}})$（$\eta_{\text{act}} = 2.0$，静默背景直接输出 $\sigma_{\text{epis},i}^2 = 0$）。
   - 对非静默活跃网格投影至单位超球面 $\bar{z}_i(x,y) = \frac{\tilde{z}_i(x,y)}{\|\tilde{z}_i(x,y)\|_2 + \epsilon} \in \mathbb{S}^{C_p-1}$。在良性训练集上以 **前景目标活跃网格与背景高激活网格分层平衡采样**，结合正交多样性正则 $\mathcal{L}_{\text{div}} = \|\text{ReLU}(P P^\top - I - 0.35)\|_F^2$ 训练包含 $M_c = 32$ 个单位基的紧致球面原型库 $P = [p_1,\dots,p_{M_c}]^\top \in \mathbb{R}^{M_c \times C_p}$。为避免线性软组合张成过完备子空间或反向原型凸组合坍缩至球心内部，采用**光滑最近邻余弦锥偏离度**：
     $$d_{\text{cone}}\big(\bar{z}_i(x,y)\big) = \max\left(0,\; 1 - \frac{1}{\tau_p}\ln\sum_{m=1}^{M_c}\exp\big(\tau_p \bar{z}_i(x,y)^\top p_m\big)\right)$$
   - 综合球面锥偏离度与良性 99% 净响应范数上界 $\overline{\mu}_{99}$ 越界惩罚定义认知不确定性：
     $$\sigma_{\text{epis}, i}^2(x,y) = \mathcal{A}_{\text{act}, i}(x,y) \cdot \left[ d_{\text{cone}}\big(\bar{z}_i(x,y)\big) + \lambda_{\text{norm}} \max\Big(0, \frac{\|\tilde{z}_i(x,y)\|_2}{\overline{\mu}_{99}} - 1\Big)^2 \right]$$
- **模块作用**：① 使本车准确识别“25m 清晰空地（低 $\sigma_{\text{alea},0}^2$）”与“前车遮挡阴影（高 $\sigma_{\text{alea},0}^2$）”；② 仅对非静默活跃网格执行轻量矩阵内积，在不误伤高范数真实车辆的前提下，敏锐捕获违背良性协议通道方向结构或幅值包络的对抗扰动。

### 算法模块三：几何分母隔离的非对称白化审计与有界余弦融合（UCR-Fusion）
1. **分母物理隔离的非对称双边白化冲突与极坐标可见域语义冲突**：
   设静默指示 $\mathcal{S}_i(x,y) = 1 - \mathcal{A}_{\text{act}, i}(x,y)$。为豁免协作者在中远距（$d_i > r_{\text{near}} = 14\text{m}$）因局部视线受挡产生的正常静默无响应，同时严惩清晰空地伪造、近距无遮挡反常抹除及非零反向擦除特征，构造**分母与待检特征 $Z_i$ 彻底物理隔离**的特征白化冲突度量：
   $$D_i^{\text{feat}}(x,y) = \frac{\big(1 - \mathcal{S}_i(x,y)\cdot \mathbb{I}(d_i(x,y) > r_{\text{near}})\big) \cdot \frac{1}{C_p}\|Z_0(x,y) - Z_i(x,y)\|_2^2}{\tau_a \big(\sigma_{\text{alea},0}^2(x,y) + u_{\text{geom}}(d_i(x,y), m_i)\big) + \tau_e \sigma_{\text{dom}}^2(m_i) + \epsilon}$$
   同时，零额外卷积开销地复用冻结分类头 $1\times 1$ 显著度响应 $p_i(x,y) = \max\sigma(\text{cls\_head}_{\text{ref}}(Z_i(x,y)))$，在极坐标光束确认的清晰可见域构造非对称前景语义冲突项：
   $$C_i^{\text{sem}}(x,y) = \mathbb{I}(\sigma_{\text{alea},0}^2 < \theta_{\text{clear}})\Big[\text{ReLU}(p_i - p_0 - \delta_s) + 0.85\,\mathbb{I}(u_{\text{geom},i} < \theta_{\text{close}})(1 - \mathcal{S}_i\mathbb{I}(d_i > r_{\text{near}}))\text{ReLU}(p_0 - p_i - \delta_e)\Big]$$
   综合局部冲突图为 $D_i^{\text{cal}}(x,y) = D_i^{\text{feat}}(x,y) + \gamma_{\text{sem}} C_i^{\text{sem}}(x,y)$（$\gamma_{\text{sem}} = 4.5$）。
   > **防自适应逃逸性质**：白化分母中协作者侧仅含几何锁定的 $u_{\text{geom}}(d_i, m_i)$ 与离线标定的模型域方差下界保护常数 $\sigma_{\text{dom}}^2(m_i)$，攻击者无论如何篡改 $Z_i$ 均无法放大分母逃避冲突检验！

2. **连通簇池化 `MeanTopK` 与验证集分位数尺度对齐的无量纲最大值帧级审计**：
   在双边共视核验域 $\Omega_{\text{covis}, i} = \{(x,y) \mid \sigma_{\text{alea},0}^2(x,y) < \theta_{\text{clear}} \;\land\; u_{\text{geom}}(d_i(x,y), m_i) < \theta_{\text{reach}}\}$ 上，先经 $3\times 3$ 均值滤波聚合连通目标簇能量，分别提取单车微簇尺度统计量：$r_i^{\text{covis}} = \text{MeanTopK}_{12}(D_i^{\text{feat}}\cdot \mathbb{I}_{\Omega_{\text{covis},i}})$、$r_i^{\text{sem}} = \text{MeanTopK}_{10}(C_i^{\text{sem}})$ 以及兼顾局部单车与全图弥散篡改的 $r_i^{\text{epis}} = \text{MeanTopK}_{12}(\sigma_{\text{epis},i}^2) + 0.6\,\text{MeanTopK}_{120}(\sigma_{\text{epis},i}^2)$。利用独立 `validate` 集良性异构分布离线标定三路指标各自的 95% 分位数尺度 $(q_{\text{covis}}, q_{\text{sem}}, q_{\text{epis}})$，通过无量纲最大值融合彻底消除多源量纲失配：
   $$s_i^{\text{frame}} = \max\left(\frac{r_i^{\text{covis}}}{q_{\text{covis}}},\; \frac{r_i^{\text{sem}}}{q_{\text{sem}}},\; \frac{r_i^{\text{epis}}}{q_{\text{epis}}}\right)$$
   再按验证集良性联合分布 95% 分位数门限 $\tau_{\text{frame}}$ 生成帧级软信誉 $\eta_i = \sigma\big(-\kappa (s_i^{\text{frame}} - \tau_{\text{frame}})\big)$。无论攻击者实施仅占十余网格的局部单车伪造/擦除，还是全图低幅弥散白盒攻击，任一维度越界均能零稀释触发抑制！

3. **非静默前景防稀释 + 零保底精度权重 + 有界温度余弦软门控融合**：
   - **零保底精度权重**（封死刷高方差漏洞）：令 $u_{\text{alea},i}^{\text{eff}} = \max\big(\hat{u}_{\text{alea},i}(x,y), u_{\text{geom}}(d_i(x,y), m_i)\big)$，协作者精度权重 $\pi_i(x,y) = \text{clip}(1 - 0.75 u_{\text{alea},i}^{\text{eff}}, 0, 1)$（下界严格为 $0$，本车 $\pi_0(x,y) = \text{clip}(1 - 0.85\sigma_{\text{alea},0}^2(x,y), 0.15, 1)$ 保底锚定）；
   - **非静默前景防稀释显著度门**：令有界显著度 $\tilde{p}_i = \text{clip}(p_i, 0, 0.85)$，本车 $\nu_0 = (\tilde{p}_0 + 0.025)^2$，协作者 $\nu_i = (\tilde{p}_i + 0.025)^2 \cdot \mathbb{I}(p_i > 0.012)\cdot (1 - \tilde{p}_0 \mathcal{S}_i)$。当本车已清晰见车而协作者该处静默时 $\nu_i \to 0$，既防止良性遮挡车的零特征冲淡本车前景，又使“置零式目标擦除”彻底失效；
   - **局部网格软门控**：$g_i(x,y) = \exp\big(-\beta_d \max(0, D_i^{\text{cal}}(x,y) - \tau_{\text{grid}}) - \beta_e \max(0, \sigma_{\text{epis},i}^2(x,y) - \tau_{\text{epis}})\big)$；
   - **有界余弦融合**（封死指数点积劫持，锁死在 $[e^{-\beta_{\text{att}}}, e^{\beta_{\text{att}}}]$）：基于净响应单位方向 $\bar{z}_i$ 定义有界相似度 $\text{sim}(Z_0, Z_i) = \beta_{\text{att}} \big[\mathbb{I}(p_0 > 0.10)\,\bar{z}_0^\top \bar{z}_i + \mathbb{I}(p_0 \le 0.10)\cdot 0.85\big]$（$\beta_{\text{att}} = 1.2$），对 $i=0$ 固定 $\eta_0 = g_0 = 1$：
     $$w_i(x,y) = \frac{\eta_i \cdot g_i(x,y) \cdot \pi_i(x,y) \cdot \nu_i(x,y) \cdot \exp\big(\text{sim}(Z_0(x,y), Z_i(x,y))\big)}{\sum_{j=0}^K \eta_j \cdot g_j(x,y) \cdot \pi_j(x,y) \cdot \nu_j(x,y) \cdot \exp\big(\text{sim}(Z_0(x,y), Z_j(x,y))\big)}, \quad Z_{\text{fused}}(x,y) = \sum_{i=0}^K w_i(x,y) Z_i(x,y)$$

---

## 6. 实验底座、严格数据隔离与公平验证方案

1. **实验底座与数据集**：
   - 基于 `OpenCOOD` 框架与 PyTorch 2.0 + CUDA 11.8 + `spconv 2.3.6`（RTX 4090），全部代码与产物严格隔离于 `/root/autodl-tmp/proj_dir2_hetero_trust/`。
   - **严格父序列隔离**：OPV2V 官方 `train`（20 序列，3,927 帧）用于训练异构适配器、独立代理模型与无监督球面原型库；`validate`（6 序列，采样 240 帧）用于模型选择及**所有防御方法统一的 95% 良性分位数门限一次性标定**；`test`（16 序列，覆盖全部序列均衡采样 150 帧共 2,154 个真值框）全程冻结终测，零测试集调参。
2. **完整四类开放异构智能体画像**：
   - 已实现并评测完整四类异构智能体：**Profile-A**（64 线 `PointPillars` 冻结基准）、**Profile-B**（64 线 `SECOND` 三维稀疏体素卷积异构骨干）、**Profile-C**（32 线物理俯仰环降采样 + `0.8m` 粗粒度柱状体素编码器）、**Profile-D**（16 线物理俯仰环降采样低成本稀疏雷达编码器）。
3. **五维场景与防捷径攻击评测**：
   - `Clean`（良性开放异构）、`PGD-Removal`（梯度目标擦除）、`Ghost-Spoofing`（在 $15\text{m}\sim 70\text{m}$ 道路可行驶区随机无车位置、随机朝向注入的优化伪造与跨帧特征块移植攻击，严防位置/尺寸捷径）、`Transfer-Attack`（跨异构代理模型单步迁移攻击）、以及**`White-box Adaptive Attack`（联合最小化 $s_i^{\text{frame}}$ 与 $D_i^{\text{cal}}$ 的白盒自适应攻击）**。
4. **同底座公平对比基线與模型无关客观真值分层指标**：
   - 所有融合与防御基线（`Ego-Only`、`Unconstrained-AttFusion`、`Hetero-Where2comm`、`ROBOSAC-Full`、**补齐本车可见域条件的公平增强基线 `ROBOSAC-VisCalib`**、`Global-Feat-Residual`）共享完全一致的异构编码底座（并单列 `Unconstrained Adapter` vs `LB-FPA` 消融）。
   - 除总体 `AP@0.5 / AP@0.7`、良性异构误杀率 `BFQR` 与延迟（ms）外，严格按真值框内**本车原始激光点数 $N_{\text{pts}}^{\text{ego}}$** 客观划分为**清晰层（$\ge 15$ 点）**、**稀疏层（$1\sim 14$ 点）**与**纯盲区补盲层（$0$ 点且协作者 $\ge 5$ 点）**分别统计召回率与精确率。
