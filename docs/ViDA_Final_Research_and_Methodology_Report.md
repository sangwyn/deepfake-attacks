# ViDA Final Research & Methodology Report

**Visibility-guided Dual-domain Adaptive Attack for AADD-2026**

- 作者 / Author: Mingyao Duan（branch `mingyao-dev`）
- 日期 / Date: 2026-09-03
- 正式版本 / Final version: **ViDA v3.4-final-no-stage0**（`team_repo/attacks/vida.py`，commit `66096c9`）
- 评估器 / Evaluator: 官方 `evaluate.py`，**未做任何修改**
- 数据集 / Database: `/home/aiattacks/dataset/celebA/TEST`（200 张 = 100 fake PNG + 100 real JPG，均为 1024×1024，无重复文件；manifest 见 `experiments/yang_comparison/manifest.json`）
- 检测器 / Detectors: `vit_b_16`（RGB 空间域）与 `densenet121_dct`（灰度 DCT 频域），官方 checkpoint
- 攻击方向 / Direction: fake → real（target = Real = class 0）；ε = 8/255（L∞, RGB）

> 本报告数字全部来自实际存在的 logs / JSON / Yang PDF。凡属推测或尚未验证的内容，文中明确标注 *Not yet verified*。

---

## 0. 摘要 / Abstract

本报告系统阐述 ViDA（Visibility-guided Dual-domain Adaptive attack）的研究动机、方法学设计与实验证据。核心论点是：

> **ViDA 不是"又一个更强的 PGD"，而是一个以检测器预处理可见性几何（visibility geometry）为出发点、以官方分数（score）为直接优化目标的对抗优化框架。**

ViDA 把攻击流程组织为五个职责明确的阶段：MI-FGSM 获取（Acquisition）→ 二分线搜索最小化（Minimality）→ Adam 质量恢复（Quality Recovery）→ 官方 uint8 验证（Verification）→ 难例救援（Hard-case Rescue）。最终版本 v3.4-final-no-stage0 在 Yang 使用的数据库（`celebA/TEST`）200 张图上，经**未修改的官方评估器**评测，达到 **397.97 / 400**，两个检测器欺骗率均为 **100%**，平均 SSIM = 0.9945，LPIPS = 0.0046。

报告同时记录了 hard-case diagnosis 的科学发现（ViT 是难例的瓶颈检测器；最难的样本并非干净置信度最强的样本，而是决策边界在感知约束下几何代价最高的样本），以及 best-of-k 多起点新发现（同一图像同一方法的独立运行在质量上存在显著 run-to-run 方差，高质量解存在但并非每次都被恢复轨迹找到）。

---

## 1. 研究背景与问题定义 / Background and Problem

### 1.1 评分规则决定优化目标

AADD-2026 官方分数（aggregate = `sum`）为：

$$
\text{Score}=\sum_{\text{images}}\sum_{d\in\{\text{ViT},\text{DCT}\}}
\underbrace{\big(0.5\cdot \mathrm{SSIM}+0.5\cdot(1-\mathrm{LPIPS}_{\text{alex}})\big)}_{Q,\ \text{per-image quality}\in[0,1]}
\cdot \mathbb{1}\big[\,\hat y_d = \text{Real}\,\big]
$$

该目标函数有两个直接推论：

1. **欺骗率是入场券，质量是区分度。** 检测器没被骗到，该 (image, detector) 对的贡献直接归零；被骗到之后，每一分差距都来自 Q。因此"用最小感知代价完成欺骗"是核心优化问题，而不是"用最大扰动保证欺骗"。
2. **Q 的两个分量特性不同。** SSIM 是局部统计相似度（窗口化均值/方差/协方差），LPIPS-Alex 是深度特征距离。后者对高频、结构性扰动更敏感。两个检测器的预处理又恰好把不同的图像投影暴露给攻击。

### 1.2 两个检测器观察的是同一图像的不同投影

- **ViT-B/16**：`RGB → resize 256×256 → center crop 224×224 → ImageNet normalization → ViT`
- **DenseNet-121-DCT**：`RGB → grayscale (0.299R+0.587G+0.114B, 0–255 scale) → LANCZOS resize 256×256 → center crop 128×128 → orthonormal DCT-II → log(|c|+1e-6) → DenseNet`

由此产生非对称的"可见性几何"（256 规范网格上）：

| 区域/通道 | ViT | DCT | 含义 |
|---|---|---|---|
| 外圈 border（256 网格外 16px 环） | 不可见 | 不可见 | 扰动纯属质量浪费 |
| 环带 annulus（224 方减 128 方） | 可见 | 不可见（不裁、且转灰度前在 1024 全图） | ViT-only 空间区域 |
| 中心 128×128 | 可见 | 可见 | 共享优化区域 |
| 色度 chroma（luma 为零的 RGB 方向） | 可见 | 被灰度化抹除 | ViT-only 通道方向 |
| DCT 高频系数 | 间接 | 直接 | 频域敏感方向 |

（注：DCT 的 128 crop 来自 256 网格；在 1024 输入图上对应中心约 512px 区域。我们的可微复刻在全分辨率上构建掩码与 Lanczos 重采样矩阵，与官方路径逐项对齐。）

---

## 2. Yang 五方向工作回顾 / Review of Yang's Five Directions

Yang 的完整报告（*AADD-2026 Five Research Directions — Complete Integrated Stage Report*, 2026-09-03）在统一实验设置下评测了五个研究方向。统一设置为：

- 数据：`/home/aiattacks/dataset/celebA/TEST/TEST_FAKE` 的**全部 100 张 fake 图**（不含 real）；
- 检测器：ViT-B/16 与 DenseNet-121-DCT，class 0 = Real；
- 名义预算：ε = 8/255，step 0.5/255，seed 0；方向 1/2/4/5 为 40 iterations；方向 3 为冻结 universal component + 残差（residual ε = 2/255，10 次残差更新），**预算/流程并非 compute-matched**；
- 分数：报告明确声明是 **local similarity-weighted cumulative score（内部诊断分数），不是 official leaderboard score**；内部筛选门限 mean SSIM ≥ 0.94 且 mean LPIPS ≤ 0.15。

Table 7（Yang 报告，100 fake）：

| Direction | ViT Real | DCT Real | SSIM | LPIPS | Score（内部） |
|---|---:|---:|---:|---:|---:|
| D1 Joint PGD（联合梯度） | 97% | 46% | 0.9429 | 0.1592 | 127.9333 |
| D2 Full frequency（频域分解） | 97% | 48% | 0.9430 | 0.1590 | 129.5576 |
| D3 Universal + residual | 16% | 82% | 0.9375 | 0.1876 | 85.8264* |
| D4 ISP / camera prior | 91% | 51% | 0.9464 | 0.1355 | 128.7722 |
| D5 Adaptive scheduler | 44% | 41% | 0.7385 | 0.3669 | 59.4220 |

（*D3 分数由 per-image 行按统一公式重构；Yang 报告已注明。）

各方向解决的问题：

- **D1 Joint PGD** 证明了**单一 RGB 扰动可以同时降低两个检测器的 target loss**：`L = 0.5·L_ViT + 0.5·L_DCT`，配合可微 Torch DCT、SciPy parity check、分支梯度归一化、小步长多迭代、L∞ 投影。它是最强 ViT baseline（97%），但 DCT 只有 46%，且 LPIPS 0.159 略高于内部门限 0.15。
- **D2 Frequency decomposition** 研究 DCT 频带敏感性与源→目标迁移：对更新梯度在 DCT 空间做滤波（注意：它滤波的是 update gradient，不约束最终扰动谱）。全频结果 97%/48%；迁移实验显示 ViT-source 几乎不迁移到 DCT（1%），DCT-source 到 ViT 为 0%，说明漏洞是 representation-dependent 而非 universal。
- **D3 Universal components** 研究可复用通用扰动 `δ(x)=δ_universal + r(x)`：对 DCT 有效（82%）但 ViT 只有 16%；held-out 上共享 depthwise filter 对 ViT <0.7%。定位为 detector-specific prior。
- **D4 ISP/camera-statistical prior** 利用生成图像缺少相机采集统计（shot/read noise、ISP 相关性）：`Y=0.299R+0.587G+0.114B`，`σ(Y)=√(σ_p²Y+σ_g²)`。质量最好（SSIM 0.9464, LPIPS 0.1355），但 standalone noise 为 0% Real，且无 RAW/标定数据，属于 quality-constrained follow-up 而非已验证的物理指纹。
- **D5 Adaptive scheduling** 借鉴 ARMOR++ 反馈分配思想（无 VLM/LLM agent），在像素 PGD / DI-FGSM / DCT 低频 / 全局噪声四个候选间调度，以 MSE 为质量代理。结果 44%/41%、SSIM 0.7385，被 Yang 自己冻结为 negative result（候选原语互补性不足，且 MSE 代理与真实 SSIM/LPIPS 脱节）。

**Yang 工作确立的基线认知**：(i) 联合梯度优化可以让一个 RGB 扰动同时作用于两个检测器；(ii) 空间域与频域漏洞不互相迁移；(iii) 频域检测器 DCT 在联合优化下明显更难（D1/D2 仅 46–48%）；(iv) 质量与欺骗率存在张力。

---

## 3. Yang 方法的不足与 protocol 差异 / Limitations of the Yang baseline

本节严格区分"报告中明确没有的组件"与"未作为中心组件的内容"。

**方法层面（基于 Yang 报告文本）：**

1. **DCT 分支欺骗率低（46–51%）是主基线的主要短板。** 可能原因中，有两条与我们的消融发现直接相关：
   - 我们的 G 系列实验表明，**动量（MI-FGSM, μ=1）对频域检测器是必需的**：纯 sign I-FGSM 在 100 步时对 DCT 只能达到约 1–2/8 的欺骗率，动量后才能稳定翻转；Yang D1 的描述（small steps, 40 iterations）未将动量作为中心组件（动量出现在 D5 scheduler 中而非 D1/D2）。
   - 我们的预处理对齐实验表明，DCT 分支的可微预处理必须与评估器**逐项一致**（0–255 灰度尺度、PIL LANCZOS 缩放核）；Yang D1 报告使用 **bicubic preprocessing** 与 SciPy DCT parity（DCT 本身做了 parity，但缩放核是 bicubic）。*bicubic 与 LANCZOS 的差异是否足以解释 DCT gap，Not yet verified*；我们仅能确认在我们的管线中核/尺度不匹配会使分支梯度指向错误方向。
2. **质量主要是 post-evaluated，而非 in-loop optimized。** Yang 的 SSIM/LPIPS 是攻击后的评估指标（D5 甚至用 MSE 代理），攻击目标是检测器 loss；没有把 `0.5·SSIM+0.5·(1−LPIPS)` 作为循环内的优化目标。
3. **没有显式的最小化搜索（minimality search）。** 攻击结束后不回答"完成欺骗所需的最小扰动是多少"。
4. **没有连续域质量恢复阶段。** 未见在 fooling region 内重分配扰动、最大化感知质量的阶段。
5. **uint8 闭环验证不是中心组件。** 我们的经验是 float 优化域与 uint8 评分域之间存在舍入间隙，必须用官方 transform 在存盘图上做硬门控；Yang 报告强调 post-save metrics，但未将其作为 accept/reject 门控。
6. **难例没有专门机制。** D3 的 universal+residual 是 detector-specific 的另一路线，不是对同一图的细粒度重攻。
7. **D5 的教训本身很有价值**：用 MSE 代理质量、用互补性不足的候选原语做调度，会得到 negative result——这反向支持了"质量目标必须是真实 SSIM/LPIPS 且必须 in-loop"的设计。

**Protocol 差异（禁止据此做绝对排名）：**

- Yang Table 7 是 **100 张 fake-only** 的内部相似度加权累计分（满分量级 100×2=200，且 Q<1）；ViDA v3.4 的 397.97 是 **200 张（100 fake+100 real）** 的官方 `sum` 分（满分 400）。两者分母、样本构成、分数定义均不同。
- 迭代预算不同：Yang 名义 40 iterations、step 0.5/255；ViDA 为 80+60 获取 + 多阶段质量恢复 + 难例救援，计算量更大。
- Yang 明确声明其分数不是 leaderboard 分数、未用 eligibility manifest（raw target-hit rate 而非 clean-corrected ASR）；ViDA 的数字由未修改官方 `evaluate.py` 直接产出。
- 因此本报告**不做"397.97 vs 129.6"式的直接高低排名**；可比的方法层结论是：在相同数据库、相同检测器、相同 ε 下，ViDA 把两个检测器都推到 100% 且 Q≈0.99，而 Yang D1/D2 的 DCT 停在 46–48%、LPIPS ≈ 0.16。

---

## 4. ViDA 的核心观察与假设 / Core observations and hypotheses

ViDA 的出发点不是"换一个梯度融合公式"，而是三条观察：

**观察 1：检测器消费的是预处理后的投影，不是图像本身。** 攻击的有效自由度由预处理决定——外圈像素无人可见，色度对 DCT 不存在，环带 luma 对 DCT 不可见。这把攻击空间天然划分为 ViT-only、DCT-useful、shared、waste 四类。

**观察 2：欺骗之后的问题性质改变。** 获取阶段（acquisition）是"如何跨过决策边界"；跨过之后，问题变为"在保持边界在正确一侧的前提下，如何让 δ 的感知代价最小"。后者是一个带离散（检测器判定）约束的连续优化问题，与 PGD 的固定步长 sign 步进不是同一个优化几何。

**观察 3：评分发生在 uint8 存盘图上。** 优化在 float 域，评分在 round 后的 uint8 图上；任何不经过官方 transform 验证的"成功"都可能在存盘后失效。

据此提出的假设（沿用 PLAN 中的编号）：

- **H1（可见性非对称存在）**：不同预处理管线产生不同的扰动敏感性。——实验支持（G1/G3、单检测器成本实验）。
- **H2（检测器独占方向存在）**：存在 ViT 可见而 DCT 近似不可见的扰动方向。——部分支持：色度方向确实 DCT-blind 且可独立翻转 ViT，但**色度扰动并不感知廉价**（LPIPS 代价与全 RGB 相当），因此 ViDA 最终选择 joint gradient + 质量恢复，而非严格的盲子空间路由（见 §9 证据台账）。
- **H3（自适应预算提升分数效率）**：固定迭代/固定预算不是最优，易图早停、难图加码。——实验支持（v3 系列版本对比）。

---

## 5. ViDA 框架总览 / Framework overview

ViDA 的数据流与职责链：

```text
Original image (H×W×3 uint8)
        │
        ▼
[Differentiable evaluator replicas]  ← 两个检测器预处理的可微精确复刻
        │           （LANCZOS 重采样矩阵 / 正交 DCT-II / 稳定 log 反传）
        ▼
Stage A — MI-FGSM Acquisition        进入 fooling region（不追求质量）
        ▼
Stage B — Binary Line Search         找到仍 fooled 的最小扰动尺度 t
        ▼
Stage C — Adam Quality Recovery      在 fooled 约束内最大化 Q（多 phase）
        │      每个候选都经 official-transform uint8 门控接受/拒绝
        ▼
Stage D — Official uint8 Verification  存盘图复核；漏网则续攻
        ▼
Stage E — Hard-case Rescue           细步长/随机起点多轮重攻，best-of-k 保留
        ▼
Best uint8 image  argmax(n_detectors_fooled, Q)
```

核心逻辑链：**Detector preprocessing → Visibility geometry → Attack acquisition → Minimality search → Quality recovery → Official uint8 verification → Hard-case rescue**。

它与 naive gradient fusion（如 Yang D1 的 `0.5L_ViT+0.5L_DCT` 一步到位）的本质区别在于：**fooling 与 quality 被解耦为条件触发的不同优化阶段，且每个阶段的 accept/reject 都以评分域（uint8 + official transform）而非优化域（float + differentiable proxy）为最终裁判。**

三条贡献层次（详见 §13）：

1. **Visibility-aware formulation**：从预处理导出攻击可见性几何；
2. **Score-aware conditional optimization**：acquisition → minimality → recovery 的条件阶段化；
3. **Hard-case optimization stabilization**：诊断 + 救援 + 探索性 multi-start，处理决策边界附近的不稳定优化。

---

## 6. Stage-by-Stage 方法学（上）：Stage A 与 Stage B

### 6.1 可微评估器复刻（所有阶段的基础）

两个检测器预处理都被复刻为可微模块，前向与官方 `evaluate.py` 逐项一致：

- **ViT branch**：`TF.resize(256, antialias) → center_crop(224) → ImageNet normalize → vit_b_16`；
- **DCT branch**：灰度化（0–255 scale）→ **可微 Lanczos**：用固定可分离重采样矩阵 `M_h, M_w` 复刻 PIL `Image.resize(LANCZOS)`（对 1024→256 验证与 PIL 差异 < 1/255 量级；bicubic 差异可达约 10/255）→ 中心 128 裁剪 → 正交 DCT-II（`D g Dᵀ`）→ `log(|c|+1e-6)`。
- **稳定 log**：前向与官方完全一致；反传用 magnitude-capped 自定义 autograd（`sign(x)/(|x|+0.01)`），避免近零高频系数上 ~10⁶ 的梯度爆炸支配 sign 梯度。

这一层是工程前提：**梯度必须沿与评分相同的函数流回像素**。

### 6.2 Stage A — MI-FGSM Acquisition（获取）

- **输入**：原图 x（float, [0,1]），零初始化 δ（Stage E 的重攻轮可以用随机 PGD 起点）。
- **输出**：一个使两个检测器 float 预测均为 Real 的 δ₀（或耗尽预算后尽力而为的 δ）。
- **优化目标**：两个检测器 targeted margin loss 之和：

$$
L_A(\delta)=\sum_{d}\mathrm{softplus}\big(\max_{c\neq t} z_d(\tilde x)_c - z_d(\tilde x)_t + \kappa\big),\quad t=\text{Real}=0,\ \kappa=0.5
$$

- **更新规则**（MI-FGSM, Dong et al. 2018）：

$$
g_t=\frac{\nabla_\delta L_A}{\frac{1}{N}\sum|\nabla_\delta L_A|+10^{-12}},\qquad
m_t=\mu\, m_{t-1}+g_t,\qquad
\delta_{t+1}=\Pi_{\|\delta\|_\infty\le\varepsilon}\big((\delta_t-\alpha\,\mathrm{sign}(m_t))\odot M\big)
$$

  其中 μ=1.0，α=2/255，ε=8/255，M 为 border mask（256 网格外圈 16px 恒零，最近邻上采样到全分辨率）。
- **早停（early stop）**：两个检测器 float margin 均 < 0 即停。
- **Tail**：若早停/预算耗尽时仍有检测器未翻转，追加最多 60 步联合步进（难图加码）。
- **为什么必须是 MI-FGSM 而非 I-FGSM**：实验上纯 sign I-FGSM 对 DCT 分支在 100 步时仅约 1–2/8 翻转率——log-DCT 损失面震荡剧烈，动量（μ=1 + 均值绝对值归一化）稳定了跨迭代方向。
- **为什么 border mask**：外圈像素不进入任何检测器的 crop，扰动它们不可能帮助欺骗，只会降低 SSIM/LPIPS（G1 验证：mask 对 ASR 无害、质量改善）。
- **Stage A 的职责边界**：**只负责快速进入 successful/fooling region，不追求最终质量。** sign 步进的性质决定了它必然 overshoot——每一步在整个可见区域同向推动，跨过边界时扰动幅度远超必要。这正是 Stage B 存在的原因。

### 6.3 Stage B — Binary Line Search（最小化搜索）

- **输入**：Stage A 输出的 δ₀（已 fooled）。
- **输出**：δ_B = t*·δ₀，t* 为仍使两个检测器在**评分域**被骗的最小尺度。
- **判据（关键）**：不使用 float 分支预测，而是把 `(x+tδ)` 经 **clip → round → uint8 → PIL → 官方 `transform`（含 LANCZOS/scipy DCT）→ 检测器 argmax** 走一遍，判定是否仍为 Real。
- **算法**：二分 12 次，不变量为"hi 通过门控、lo 不通过"：

$$
t^*=\min_{t\in[0,1]}\ t\quad\text{s.t.}\quad \forall d:\ \hat y_d(\mathrm{uint8}(x+t\delta_0))=\text{Real}
$$

- **为什么有效**：sign 攻击沿单一方向累积，跨越边界后方向不变、幅度冗余；整体缩放 δ 不改变其方向，只去除冗余幅度，且门控保证不会丢欺骗。
- **为什么不直接减小步长**：减小步长降低 overshoot 但不能回收已经过度的累积，且会拖慢获取；线搜索是对"已完成轨迹"的**直接后验最小化**，一次 12 次二分的代价仅约 0.1 秒/次（小分辨率检测器前向）。
- **重复使用**：Stage C 的每个 Adam phase 之间再次执行线搜索——Adam 重排了 δ 的空间结构后，新的冗余幅度会再次出现。

### 6.4 Stage C — Adam Quality Recovery（质量恢复）

- **输入**：δ_B（已 fooled 且经幅度最小化）。
- **输出**：在保持已 fooled 检测器的前提下，使 Q 最大化的 δ_C。
- **问题重述**：进入 fooling region 后，优化从"如何跨过边界"变为"**在边界保持在正确一侧的约束下，让图像尽可能接近原图**"。这是一个连续目标、离散约束的优化。
- **目标函数**（对连续 δ）：

$$
L_C(\delta)=\underbrace{\sum_{d\in\mathcal L}\mathrm{softplus}\big(m_d(\tilde x)+\kappa_t\big)}_{\text{margin buffer（仅锁定的检测器）}}+\lambda\,(1-Q(\tilde x)),\quad
Q=0.5\,\mathrm{SSIM}_{\mathrm{diff}}+0.5\,(1-\mathrm{LPIPS}_{\mathrm{alex}})
$$

  其中 $m_d=z_{d,\text{Fake}}-z_{d,\text{Real}}$ 为 margin（<0 表示 fooled），$\mathcal L$ 为当前已 fooled 的检测器集合，λ=3，SSIM 为可微 box-filter 实现，LPIPS 为 Alex 网络（与评估器同网络、全局只加载一次）。
- **优化器**：Adam（β₁=0.9, β₂=0.999），初始 lr 0.4/255，后半程 ×0.6。
- **为什么 Adam 优于 sign 步进**：质量目标在像素间高度不均匀——某些像素的扰动对 LPIPS 几乎无贡献但对检测器 margin 重要，另一些反之。Adam 的逐坐标自适应步长天然实现这种差异化分配；sign 步进对所有坐标等幅移动，无法做"重分配"。
- **κ annealing（1.0 → 0.1）**：早期要求 margin 显著为负（安全裕量），随恢复进行允许 margin 贴向 0（决策边界）。**直觉：margin 只需为负，任何"骗得更狠"的强度都是白付的感知代价**；真正的安全网是 uint8 门控而非 margin 数值。
- **硬约束投影**：每个候选 δ 投影回 L∞ 球（ε=8/255）与 border mask。
- **Accept / reject 机制（核心）**：候选只有在**官方 transform 作用于 round 后 uint8 图**、且所有已锁定检测器仍判 Real 时才被接受；否则丢弃。优化过程中始终维护全局最优 `best = argmax(n_fooled, Q)`。
- **多 phase 组织**：每个 phase 约 17 步 Adam，phase 间重置 Adam 动量并执行一次 Stage B 线搜索；总预算自适应——当某 phase 后线搜索已无缩放空间（hi>0.999）且 Q 提升 <0.002，连续两次即早停；难图最多用约 2× 基础恢复预算。
- **为什么不能"刚骗到边界就停"**：(i) 刚越过边界的解往往是空间密集、感知显著的扰动（sign 步的等幅推动），margin 虽为负但 LPIPS 差；(ii) 质量目标允许把扰动从感知敏感坐标**重分配**到廉价坐标（如色度/纹理方向），在边界几何允许的范围内显著降失真；(iii) 边界是非凸的，"最近越过点"不等于"最低感知代价点"。实验上 Stage C 把假图平均 LPIPS 从获取后的量级降到 0.009（见 §10 版本演进）。

### 6.5 Stage D — Official uint8 Verification / Re-attack（存盘验证）

- **输入**：Stage C 的 δ_C。
- **输出**：经评分域确认的最佳 uint8 图；必要时续攻后的 δ。
- **存在原因**：`float optimization ≠ uint8 scored artifact`。优化在 [0,1] 浮点域，评分对象是 `round(255·clip(x+δ))` 后再经 PIL/scipy 预处理的图。舍入、clip、resize kernel 的任何微小差异都可能让边界附近的检测器翻转。
- **机制**：(1) round 为 uint8；(2) 跑官方两条预处理 + 检测器；(3) 若有检测器"漏网"，从当前 δ 继续 MI-FGSM（最多 3 轮 ×20 步）；(4) 每一步都更新全局 best。该阶段保证最终返回图的 ASR 不以可微代理的近似为前提。

### 6.6 Stage E — Hard-case Rescue（难例救援 / multi-start best-of-k）

- **触发条件**：A–D 结束后 `best < (2 detectors fooled, Q≥0.92)`。
- **机制**：用一组多样化配置**从零重跑完整 A–D**：(step, iterations) ∈ {(1/255, 160+80), (2/255, 120+60), (1/255, 200+100)}，随机 PGD 起点半径 r ∈ {2/255, 5/255, 6/255, 0}，部分轮次在获取期开启 DI（Diverse-Inputs, p=0.3/0.5）；一旦 Q≥0.92 即提前停止。
- **为什么细步长救援有效**：2/255 的粗步长对易图快速，但在难图上要走到 ε 上限才能翻转 ViT，留下**空间上大范围饱和的结构性失真**，Stage C 难以回收；1/255 细步长沿更短路径到达决策边界，轨迹本身破坏性更小。
- **为什么需要随机性**：见 §9——决策边界附近的离散 uint8 门控使恢复轨迹对初始/数值条件敏感，不同起点进入不同解盆地；best-of-k 把"偶尔出现的高质量解"变成"稳定产出"。
- **best 保留**：所有轮次共享同一个 `best = argmax(n_fooled, Q)`，任何轮次都不会使结果变差。

---

## 7. 为什么 Stage 按此顺序组织 / Why this ordering

顺序本身是方法论：

1. **先获取后最小化**：没有 fooling 就没有可保留的东西；而 sign 获取必然 overshoot，所以线搜索紧随其后。
2. **先最小化后恢复**：线搜索先把"幅度冗余"剥掉，让 Adam 从一个更小、更干净的起点做"空间重分配"；若先恢复，Adam 会在含冗余幅度的解上浪费自由度。
3. **恢复在 float 域、裁判在 uint8 域**：可微代理提供梯度方向，官方门控决定接受与否——梯度效率与评分可靠性分离。
4. **验证先于救援**：先确认是否真的需要救援（避免对已解决的图浪费算力），救援只针对 Q 不达标图。
5. **粗获取兜底、细救援攻坚**：默认配置对 ~95% 的图快速且足够；只有难例支付多轮重攻成本（易图在 Pass 1 即达 Q≥0.92，实测真实图平均秒级返回）。

---

## 8. Hard-case Diagnosis（难例诊断，case study）

脚本 `my_attack/diag_hard.py`，对象 001539、001556（最难），对照 001553（救援显著有效）、001549（易）。

| 测量 | 001539 | 001556 | 001553（对照） | 001549（对照） |
|---|---|---|---|---|
| 干净 margin ViT / DCT（logit 差） | +2.24 / +2.17 | +2.19 / +2.15 | +2.19 / +2.17 | +2.21 / +2.18 |
| 联合攻击后最小翻转尺度 t：ViT / DCT（细/粗步长） | **0.94 / 0.36–0.56** | **0.94 / 0.38–0.63** | 0.19 / 0.69–0.94 | 0.31 / 0.56–0.75 |
| 只攻 ViT 的最小 Q（SSIM/LPIPS） | 0.767（0.834/0.300） | 0.799（0.858/0.260） | 0.978 | 0.953 |
| 只攻 DCT 的最小 Q（SSIM/LPIPS） | **0.991**（0.995/0.014） | **0.997**（0.998/0.004） | 0.953 | 0.943 |
| 中心共享区梯度符号一致率 | 0.50 | 0.50 | 0.50 | 0.50 |
| ViT 攻击梯度能量：luma / chroma | 0.03 / 0.89 | 0.05 / 0.81 | 0.06 / 0.80 | 0.05 / 0.83 |
| ViT-only 攻击中 ε 饱和像素比例 | 0.73 | 0.56 | ~0 | ~0 |

诊断结论：

1. **最难的图并不是检测器最"确信"的图。** 四张图干净置信度几乎相同（均为 ~90% Fake，margin ≈ +2.2）。难例的本质是：**目标决策边界在感知约束下几何代价高昂**——对 001539/001556，翻转 ViT 需要接近 ε 上限的空间密集扰动（单检测器下界实验显示 ViT-only 最优仍只有 Q≈0.77–0.80，56–73% 可见像素饱和），而翻转 DCT 几乎免费（DCT-only Q≥0.99）。
2. **在两张难例上 ViT 是瓶颈检测器。** 对 001539/001556，联合攻击后 ViT 需要 t≈0.94 的扰动尺度，DCT 在 t≈0.36–0.63 即翻转；单检测器成本下界更悬殊（ViT-only Q=0.77–0.80 vs DCT-only Q≥0.99）。值得注意的是瓶颈在对照图上会易位（001553/001549 反而是 ViT 在更小尺度翻转、DCT 需要更大尺度），但两张对照图的单检测器成本都很低（Q≥0.94），不构成困难。这解释了 naive joint optimization 在难例上的低效：梯度预算按检测器平均分配，而难例中 ViT 一侧需要的推动远大于 DCT。
3. **两个检测器的获取梯度近正交。** 中心共享区符号一致率 ≈ 0.50（抛硬币水平），cosine ≈ 0；不是强对抗（negative transfer 不明显），但联合梯度和互相稀释。ViT 的攻击梯度能量 81–89% 在色度方向，与其 RGB 视图一致。
4. 科学表述：*hardness 来自边界几何（到达边界所需扰动的感知代价）而非置信度数值*——这是一个 case-study 级别的证据（n=4 张图），全数据集分布形态 *Not yet verified*。

---

## 9. Best-of-k 新发现 / Multi-start variability

**现象链（presentation finding，老师已认可）：**

```text
Same image + same method
  → different optimization trajectories
  → large Q variance
  → high-quality valid solutions exist
  → multi-start best-of-k stabilizes recovery
```

在开发期不带 Stage E 的版本上，同一图独立运行：001539 的 Q ∈ {0.819, 0.967, 0.977}（LPIPS 0.019–0.184），001556 ∈ {0.767, 0.767, 0.860}。高质量合法解确实存在（LPIPS 0.01–0.05、扰动 6–7/255、margin 贴边界），但离散 uint8 接受门控叠加优化/数值层面的可变性，会把 Adam 恢复轨迹送入不同解盆地。**注意：我们表述为"边界门控 + 优化可变性"，不把 numerical noise 断言为已证明的唯一因果解释。**

**正式 best-of-k 实验**（`my_attack/best_of_k.py`：k 次独立完整管线、独立随机种子，选择规则 argmax(n_fooled, Q)）：

| 图 | 类型 | k | best Q | mean Q | std Q | worst Q | 全部 2/2 fooled |
|---|---|---|---|---|---|---|---|
| 001539 | hard | 3 | 0.988 | 0.973 | 0.012 | 0.959 | ✓（3/3） |
| 001539 | hard | 5 | 0.971 | 0.953 | 0.016 | 0.925 | ✓（5/5） |
| 001556 | hard | 3 | 0.991 | 0.960 | 0.022 | 0.943 | ✓（3/3） |
| 001556 | hard | 5 | 0.988 | 0.962 | 0.019 | 0.941 | ✓（5/5） |
| 001553 | rescue-effective | 3 | 0.998 | 0.996 | 0.003 | 0.992 | ✓ |
| 001553 | rescue-effective | 5 | 0.998 | 0.997 | 0.001 | 0.996 | ✓ |
| 001549 | easy | 3 | 0.998 | 0.998 | 0.001 | 0.997 | ✓ |
| 001549 | easy | 5 | 0.998 | 0.998 | 0.0002 | 0.998 | ✓ |

（k=3 与 k=5 合计 32 次独立运行，**全部骗过两个检测器**；runtime：难图 60–180 s/次，易图 30–70 s/次，单张 L40。原始数据：`my_attack/best_of_k_results_k3.json`、`best_of_k_results_k5.json`。）

解读：

- **方差集中在难例**（std 0.012–0.022）；易图 std ≤ 0.003，best-of-k 几乎无增益。
- best-of-k 的首要作用是**解的可靠性（solution quality 的 tail risk）**而非 ASR——ASR 在所有运行中已是 100%。
- 内置 Stage E 已经把单次调用的下限从历史的 0.70–0.86 抬到 ≥0.925；外部再做 k 次完整重跑，仅在最难 2 张图上多 +0.02–0.03 Q。

**决策：best-of-k 不作为默认配置。** 理由：(i) 对 ≥95% 的图无增益；(ii) 计算量近似线性 ×k；(iii) 增益集中在已经 0.95+ 的图像。它是**已验证的探索性增强（validated exploratory enhancement）**：可作为 Stage E 后 Q 仍 <0.92–0.95 的个别图像的离线精修，以及 presentation 的新发现，而非主方法的必选模块。

---

## 10. 实验结果：严格区分三个层次 / Three levels of evidence

**Level 1 — 历史开发参考（不作为最终结果）。** ViDA v3.3 *含 Stage 0 clean-skip*：397.20/400，SSIM 0.9930，LPIPS 0.0070（100 假均值 1.972 / 100 真 2.000）。该版本利用了"干净图已判 Real 则原样返回"的捷径，按老师要求**从正式版中删除**，数字仅作开发轨迹参考，不用于任何对比排名。

**Level 2 — 最终合规版本 ViDA v3.4-final-no-stage0。** 所有图片（含 real）统一走完整管线，无 ground-truth shortcut，官方 `evaluate.py` 未修改：

| 指标 | 值 |
|---|---|
| Official score（sum） | **397.97 / 400**（99.5%） |
| ViT ASR / DCT ASR | **100% / 100%**（200/200） |
| Mean SSIM / LPIPS | 0.9945 / 0.0046 |
| Fake 组（n=100）均值 / 中位 / 最差 | 1.9798 / 1.9858 / 1.8955（仅 1 张 <1.90，6 张 <1.95） |
| Real 组（n=100）均值 / 最差 | 1.9999 / 1.9956 |
| Runtime | 单张 L40，1 h 30 min（均值 27 s/张） |

值得记录的现象：**删除 clean-skip 后，仍有 88/100 张 real 被逐字节原样返回**（SSIM=1.0000, LPIPS=0.0000）。机制上：已判 Real 的图 targeted-margin 梯度≈0，Stage A 几乎不产生扰动，Stage B 线搜索把近零 δ 继续缩小，Stage C 恢复将其清零——这是优化的自然结果而非 shortcut。其余 12 张 real 有亚感知级改动（最差 1.9956）。

**Level 3 — 探索性增强 ViDA + external best-of-k。** 仅在 4 张代表图上以 k=3/5 验证（§9）；**全数据集收益尚未验证（Not yet verified）**，不与 Level 2 合并报告。

**开发期版本演进（fake 组均值，100 张，官方评估器）**：naive joint 1.508 → v2 ViDA（mask+早停+tail）1.592 → v2+符号步恢复 1.747 → v3.0（线搜索+Adam+uint8 门控）1.844 → v3.1（κ 退火）1.916 → v3.2（自适应恢复）1.945 → v3.3（含 Stage 0，1.972）→ **v3.4-no-stage0 1.980**。

**与 Yang 数字的关系**：Yang Table 7 为 100-fake-only 内部诊断分（满分量级 ~200，且 DCT ASR 46–51%）；ViDA Level 2 为 200 图官方 sum 分。**两者 sample count、分母、迭代预算、分数定义不同，不做绝对值排名**；方法层可对照的信息是：同一数据库、同一检测器、同一 ε 下，ViDA 把 DCT 从 ~46–48% 推到 100%、LPIPS 从 ~0.16 降到 0.005 量级，且这些差异来自可定位的机制（动量对频域的必要性、LANCZOS/0–255 预处理对齐、in-loop 质量优化、uint8 门控）。

---

## 11. Yang vs ViDA 方法层对比 / Methodological comparison

| 维度 | Yang（五方向报告） | ViDA（本报告） |
|---|---|---|
| Core idea | 联合梯度优化（D1）/ 频带滤波（D2）/ 通用扰动（D3）/ ISP 先验（D4）/ 调度（D5） | 预处理可见性几何驱动的 score 优化框架 |
| Representation awareness | detector level（两个 loss 相加） | preprocessing / visibility level（空间、通道、频域可见性） |
| 空间分解 | 不是中心组件 | 显式 visibility reasoning（border mask、中心/环带） |
| 质量目标 | 主要 post-evaluated（D5 用 MSE 代理） | in-loop 直接优化 0.5·SSIM+0.5·(1−LPIPS) |
| 最小扰动搜索 | 不是中心组件 | 二分线搜索（获取后 + 每个恢复 phase 后） |
| 连续质量恢复 | 不是中心组件 | Adam 多 phase 恢复 + κ 退火 + 自适应预算 |
| uint8 闭环验证 | post-save metrics，非 accept 门控 | 官方 transform uint8 硬门控贯穿 B/C/D/E |
| 难例机制 | D3 为另一路线（detector-specific） | 专用细步长多起点救援（Stage E） |
| Multi-start | 不是中心组件 | 内置 Stage E；外部 best-of-k 为已验证探索项 |
| 自适应算力 | limited（统一 40 iter） | per-image 自适应（易图早停、难图加码/救援） |
| 频域分支梯度稳定性 | 可微 DCT + SciPy parity；bicubic 预处理 | LANCZOS 精确复刻 + 0–255 尺度 + log 反传 cap |
| 评测协议 | 100 fake 内部诊断分（作者声明非 leaderboard） | 200 图官方评估器原样输出 |

（"不是中心组件"均依据 Yang 报告文本；其代码库可能包含其他工程处理，未逐行审计的部分不做断言。）

---

## 12. 证据台账：哪些设计有实验支持 / Evidence ledger

**有直接实验支持（本项目 logs/JSON）：**

- DCT 分支预处理必须与评估器逐项一致（0–255 灰度、LANCZOS 核）：不一致时分支"成功"与官方判定矛盾（早期 G 实验）。
- `log(|c|+1e-6)` 反传在近零系数上爆炸（~10⁶），cap 后稳定。
- 动量 MI-FGSM 对 DCT 必需（纯 I-FGSM 100 步仅 ~1–2/8）。
- Border mask 无害且有益（G1）；低频/平滑扰动无法骗过 DCT（G3）。
- 色度方向 DCT-blind 且可独立翻转 ViT，但**色度不感知廉价**（LPIPS 代价与全 RGB 相当）→ 放弃严格盲路由，改 joint + recovery。
- 线搜索/Adam 恢复/κ 退火/自适应预算/救援的每一步收益：版本演进表（fake 均值 1.747 → 1.980）。
- uint8 门控的必要性：float 域与存盘域存在翻转间隙（Stage D 存在的理由）。
- ViT 是难例瓶颈、梯度近正交、ViT 梯度以色度为主：§8 case study。
- best-of-k 降低难例 tail risk、对易图无增益：§9。

**合理假设但未完全验证（Not yet verified）：**

- visibility 分解相对纯恢复的**独立因果贡献**未做严格隔离消融（当前增益与 recovery 耦合出现）。
- "梯度近正交源于预处理表示差异"是描述性结论，因果未隔离。
- 可微 SSIM/LPIPS 代理与评分器（skimage SSIM、LPIPS-Alex）的残差：LPIPS 同网络，SSIM 为 box-filter 近似，极小残差未量化。
- 难例硬度的全数据集分布（case study n=4）。
- best-of-k 全量集收益；多数实验单 seed；对隐藏检测器/官方测试集的泛化。

---

## 13. 贡献抽象 / Contributions

1. **Visibility-aware formulation.** 把"两个检测器"重新表述为"同一图像的两个预处理投影"，从预处理导出攻击可见性几何（空间可见域、通道可见域、频域敏感性），并以此决定扰动可以存在于何处。
2. **Score-aware conditional optimization.** 攻击被组织为 acquisition（进入 fooling region）→ minimality（剥除幅度冗余）→ recovery（在离散 fooled 约束内最大化连续质量目标）的条件阶段链，裁判统一为评分域（uint8 + official transform），把官方分数从 post-hoc 评估变成 in-loop 优化目标。
3. **Hard-case optimization stabilization.** 通过难例诊断（瓶颈定位到 ViT 的密集扰动需求与正交梯度）、细步长救援与多起点 best-of-k，把决策边界附近的不稳定优化从"偶尔成功"变为"稳定成功"，并给出何时不需要多起点的定量判据。

---

## 14. 局限性 / Limitations

1. **Protocol 差异**：Yang 数字为 100-fake 内部诊断分，ViDA 为 200 图官方分，预算/迭代数不匹配；未做 compute-matched 统一重跑（按要求不重跑 Yang）。
2. **200 图 dev 结果不是 leaderboard**：官方 AADD_2026_Test 未发布；分布偏移未知。
3. **检测器范围**：主结论仅针对两个计分检测器；NPR/AIDE 黑盒迁移另文报告（AIDE 0%、NPR DI 后 82.5%），隐藏检测器泛化未证明。
4. **诊断为 case study**（n=4）；全数据集硬度分布与瓶颈归因未完成。
5. **best-of-k 全量收益未验证**；多数对比为单 seed，未做 bootstrap/配对检验。
6. **visibility 分解的独立消融不完整**；质量代理 SSIM 为近似实现。
7. **Runtime**：难例救援使单图成本从秒级到 3–4 分钟（全 200 图 90 分钟/L40）；官方时间限制未知，若受限需按图预算截断。
8. **float/uint8 间隙**由硬门控兜底，但理论上若存在任何 float 域 fool 而 uint8 域永远无法 fool 的图，Stage D 会返回 best-effort 解（当前 200 图中未出现）。

---

## 15. Future Work

- 官方测试集发布后直接运行（模块自包含、分辨率自适应）；记录 Wilson 区间与配对 bootstrap。
- ViT-focused 攻坚：针对瓶颈检测器的 chroma/纹理感知代价建模、CW 风格 margin 约束下的 L2/LPIPS 最小化内层优化。
- compute-matched 的 Yang/ViDA 统一消融（同迭代、同预算、同子集）。
- best-of-k 的预算感知版本：按 Stage E 后实测 Q 决定是否触发外部重启。
- 隐藏检测器鲁棒性：EOT/DiffJPEG、resize/blur 期望梯度（Yang future work 中亦列出）。

---

## 16. 结论 / Conclusion

> **ViDA 应被理解为一个 visibility-aware、score-driven 的对抗优化框架，而不是一条单一的梯度融合规则。**

在 Yang 使用的数据库上，删除 Stage 0 捷径后的合规版本 ViDA v3.4-final-no-stage0 经未修改的官方评估器评测达到 **397.97/400**，两个检测器 **100% 欺骗**，SSIM 0.9945 / LPIPS 0.0046；高质量恢复（而非单纯攻击强度）已成为主要区分度来源。Hard-case 分析表明最难样本的瓶颈是 ViT 一侧的密集扰动几何代价而非置信度；best-of-k 实验表明恢复阶段在离散决策边界附近存在系统性 run-to-run 方差，多起点能将其稳定化——该机制以内置 Stage E 形式保留在主方法中，外部 k 次重跑作为可选离线精修。

---

## 附录 A. 复现与产物索引 / Artifacts

| 产物 | 路径 |
|---|---|
| 攻击代码（正式版） | `team_repo/attacks/vida.py`（commit `66096c9`，本地，未推送） |
| 正式配置 | `team_repo/configs/v34_final_no_stage0.yaml` |
| 数据 manifest | `team_repo/experiments/yang_comparison/manifest.json` |
| 最终结果 JSON | `team_repo/experiments/yang_comparison/vida_v34_final_no_stage0.json` |
| 最终运行日志 | `my_attack/results_vida_v34final_nostage0_dev200.log` |
| 历史 v3.3 日志（Level 1 参考） | `my_attack/results_vida_v3.3_dev200.log`（397.20/400，含 Stage 0） |
| 难例诊断脚本/日志 | `my_attack/diag_hard.py`、`diag_hard.log` |
| best-of-k 脚本/数据/日志 | `my_attack/best_of_k.py`、`best_of_k_results_k{3,5}.json`、`best_of_k_k{3,5}.log` |
| Yang 报告（引用来源） | `AADD-2026-Five-Directions-Complete-Research-Report-EN.pdf`（团队提供） |

复现命令：

```bash
cd team_repo
python evaluate.py --config configs/v34_final_no_stage0.yaml
```

---

## 附录 B. 为什么 100% fooling rate 是可信的 / Why the 100% fooling rate is credible

本附录给出可独立审计的证据链，回应"100% 是否可信"的问题。

### B.1 100% 的准确含义（scope）

- **设置是白盒（white-box）**：我们持有两个计分检测器的权重与梯度，这正是 AADD-2026 的白盒协议；不是黑盒或隐藏模型声明。
- **数据**：指定 dev split `/home/aiattacks/dataset/celebA/TEST`（100 fake + 100 real，共 200 张）。100% 指这 200 张图经攻击后两个检测器均判 Real。
- **预算**：L∞ 扰动 ≤ 8/255（官方预算）。
- **不声明**：官方隐藏测试集 100%、或对未见检测器 100%。黑盒迁移实验明确显示下降：NPR 57.5%（DI 后 82.5%）、AIDE 0%。白盒/黑盒的巨大落差是文献中的标准现象（Dong et al. 2018; Xie et al. 2019）。

### B.2 为什么白盒 100% 是预期结果而非异常

1. **检测器是无防御的标准分类器。** ViT-B/16 与 DenseNet-121 为检测准确率训练，无对抗训练、无证书防御、无输入随机化。对抗 ML 文献中，迭代 L∞ 攻击（PGD/MI-FGSM）在 ε=8/255 下对这类模型达到接近 100% ASR 是常态——100% 是强白盒攻击的正常水平。
2. **团队基线交叉印证。** Yang 的 Direction 1（朴素 40 步 joint PGD）已达 **ViT 97%**。ViT 100% 只是更多迭代、动量与自适应算力带来的边际改进；真正难的检测器是 DCT。
3. **DCT 从 Yang 的 46–48% 到 100% 的三个具体原因（均有消融，不是"调参更努力"）：**
   - **动量对频域模型必需**：纯 sign I-FGSM 对 DCT 在 100 步时仅约 1–2/8 翻转；MI-FGSM（μ=1、均值绝对值归一化）稳定翻转——log-DCT 特征梯度高频噪声大，动量稳定跨迭代方向。
   - **预处理逐项对齐**：DCT 评估器消费 0–255 灰度 + PIL LANCZOS 缩放；尺度/核不匹配会把梯度引离真实决策边界。我们用可微重采样矩阵复刻 LANCZOS（与 PIL 误差 <1/255），并在早期记录过"分支内成功但官方评分失败"的不匹配事故并修正。
   - **稳定 log 反传**：`log(|DCT|+1e-6)` 在近零系数上产生 ~10⁶ 量级梯度；magnitude-capped backward 使 sign 更新可用，前向与评估器逐位一致。
4. **ASR 是构造性保证（guaranteed by construction）。** Stage D 对 round 后的 uint8 图重跑官方 transform，任一检测器漏网即续攻；返回图按 argmax(骗过检测器数, Q) 择优。fooling 不是"从梯度推测"，而是用评估器自己的代码路径在最终像素上验证通过才保留。

### B.3 独立复核（攻击代码完全不参与评分）

200 张攻击后图全部保存为**无损 PNG**（`my_attack/verify_outputs/vida_v34_images_png/`），由独立脚本 `my_attack/rescore_saved.py` 从磁盘读图重打分——该脚本**只 import 官方 `evaluate.py` 的模型工厂、官方 transform、skimage SSIM、LPIPS-Alex，不 import 攻击模块**：

```text
images rescored : 200
ViT  Real rate  : 200/200
DCT  Real rate  : 200/200
mean SSIM       : 0.9941
mean LPIPS      : 0.0053
TOTAL SCORE     : 397.7590 / 400
```

该独立分数与官方评估器同次运行的 JSON（397.759003）逐位一致。任何第三方用自己的评估代码读这些 PNG 都可复现。完整日志：`my_attack/verify_outputs/rescore_independent.log`；逐图 200 行：`per_image_results.csv`。

### B.4 可信度支撑细节

- **Checkpoint sha256**：`vit_b_16.pth = 5e9677d8…58d81`，与 Yang 五方向报告中记录的哈希**完全一致**；`densenet121_dct.pth = 5bbaf5c5…5e643`。
- **评估器未修改**：`team_repo/evaluate.py` 原样使用；攻击代码仅 `attacks/vida.py`（commit `66096c9`）。
- **诚实记录的边界效应**：若攻击图按 `.jpg` 后缀保存（PIL 触发 JPEG 有损压缩），200 张中有 2 张边界真实图（ViT 本就误判 Fake 的真实图）在压缩后翻回 Fake；官方评估器对**内存中的 uint8 数组**打分（不经压缩），故官方分数不受影响。所有复核产物均为无损 PNG，该 JPEG caveat 如实记录于 `verify_outputs/REPRODUCIBILITY.md`。
- **两次运行的 0.2 分差异**（397.97 vs 397.76）：多起点救援在边界图上的 run-to-run 方差（即 §9 的 hard-case instability 发现），两次运行两个检测器均为 200/200。
- **自我证伪的开放报告**：黑盒 AIDE 迁移率 0% 与白盒 100% 并列报告——若要编造"通用攻击"，不会主动报 0%。

### B.5 复现命令

```bash
# 1) 官方评估器原样运行（保存全部攻击图）
cd team_repo && python evaluate.py --config configs/verify_saved.yaml
# 2) 独立重打分（不 import 攻击代码）
python my_attack/rescore_saved.py \
    --orig my_attack/verify_outputs/db_png \
    --adv  my_attack/verify_outputs/vida_v34_images_png \
    --weights team_repo/weights
```
