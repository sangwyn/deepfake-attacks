# ViDA v3 完整汇报 / Full Report — AADD-2026 对抗攻击

**作者 / Author:** Mingyao Duan (branch `mingyao-dev`)
**日期 / Date:** 2026-09-03
**代码 / Code:** `team_repo/attacks/vida.py`（团队模板 `attack(image, classifiers, device)`，官方 `evaluate.py` 不改动 / unchanged official evaluator）
**开发集 / Dev set:** `~/dataset/celebA/TEST`（100 假 + 100 真，1024×1024，Oleg 指定的评估/开发划分）

---

## 0. 一句话结论 / One-line result

**在 200 张开发集图片上，官方评分器给出 397.20 / 400（99.3% 满分），两个计分检测器欺骗率均为 100%，平均 SSIM = 0.993、LPIPS = 0.007。**

**On the 200-image dev set, the unchanged official evaluator scores ViDA v3 at 397.20 / 400 (99.3% of the maximum), with 100% fooling rate on both graded detectors and mean SSIM = 0.993, LPIPS = 0.007.**

---

## 1. 挑战与评分规则 / Challenge and scoring

### 1.1 任务 / Task
对每张输入图片生成一张对抗图，让两个 AI 生成图像检测器把**假图判为真**（官方计分方向 fake→real，target class = Real = 0），同时对抗图要在人眼感知上与原图尽可能接近。扰动量上限 ε = 8/255（L∞，RGB 通道）。

For every input image we produce an adversarial image that makes **both AI-generated-image detectors label fakes as Real** (the graded direction; target class Real = 0), while staying perceptually close to the original. The perturbation budget is ε = 8/255 in the L∞ sense.

两个计分检测器 / Two graded detectors:
- `vit_b_16`: ViT-B/16，看 RGB，先缩放到 256×256 再中心裁剪 224×224，ImageNet 归一化；
- `densenet121_dct`: DenseNet-121 单通道输入，路径为 灰度化（0–255）→ LANCZOS 缩放 256 → 中心裁剪 128×128 → 二维正交 DCT-II → `log(|x|+1e-6)`。

### 1.2 分数公式 / Score formula

$$\text{Score}=\sum_{images}\sum_{detectors}\underbrace{\big(0.5\cdot SSIM+0.5\cdot(1-LPIPS_{alex})\big)}_{Q,\ \text{质量分}\in[0,1]}\cdot\mathbb{1}(\text{检测器把该图判为 Real})$$

- 每个检测器对每张图**最多贡献 1 分**：被骗过（indicator=1）且图片与原图完全一致（SSIM=1, LPIPS=0 → Q=1）时拿到 1 分；
- 每张图有 2 个检测器 → **每张图最多 2 分**；
- 开发集 200 张图 → **满分 200 × 2 = 400**。官方测试集若有 N 张图，满分即为 2N（聚合方式 aggregate = `sum`）。
- 任何一个检测器没被骗到，该图对该检测器的贡献直接是 **0**（不是减分，是归零）；被骗到但图片质量差，则拿 0 到 1 之间的小数。

Each detector contributes **at most 1 point per image**: it pays 1 only when it is fooled (indicator=1), scaled by quality Q ∈ [0,1]. With 2 detectors the per-image maximum is 2, so a 200-image dev set has a maximum of 400 (aggregate = sum). A detector that is not fooled contributes exactly 0; a fooled but visibly distorted image contributes a fraction.

### 1.3 这意味着什么 / Strategic implication

100% 欺骗率只是"入场券"——真正拉开差距的是**在骗过检测器的前提下让 Q 尽量接近 1**，即"用最小的感知代价骗过检测器"。

100% fooling is only the ticket. The competition is decided by **how close Q stays to 1 while fooling both detectors** — i.e. the cheapest possible perceptual perturbation.

---

## 2. 核心思想 / Core idea

检测器并不是"看"原图本身，而是看原图经过各自预处理后的**投影**：ViT 看 RGB、中心 224；DCT 检测器看灰度、中心 128、且是 DCT 频域对数特征。两个检测器的"视野几何"不同：

Detectors never see the raw image — each sees a different **projection** through its own preprocessing (ViT: RGB / central 224; DCT: grayscale / central 128 / log-DCT). ViDA exploits this visibility geometry:

1. **不扰动任何检测器都看不见的像素**（256 规范网格外圈 16 像素，即 1024 图上的外圈 64 像素）——纯质量浪费；/ Never perturb pixels outside every detector's field of view (the outer border ring) — pure quality waste.
2. **在完整 1024 分辨率上优化扰动**（不在低分辨率上做），避免缩放带来的低通滤波伪影；/ Optimise the perturbation at full 1024 resolution to avoid resize low-pass artefacts.
3. **把两个检测器的预处理管线做成可微的精确副本**（PIL LANCZOS 用固定可微重采样矩阵复刻、DCT/对数完全一致），梯度端到端流回像素；/ Replicate both detector preprocessing pipelines differentiably (PIL LANCZOS reproduced with fixed separable resampling matrices; exact DCT and log), so white-box gradients flow end to end.
4. **先用 MI-FGSM 动量攻击骗到检测器，再把全部剩余预算用来"找回"感知质量**，且每一步都以官方评分器在 uint8 存盘图上的真实判定为安全门控。/ First acquire the fool with MI-FGSM momentum attack, then spend the remaining budget on recovering perceptual quality — with the official evaluator's own uint8 prediction as the hard safety gate at every step.

---

## 3. 方法详解 / Method in detail

最终攻击流程（`attacks/vida.py` 中 `attack()`）为 6 个阶段。每张输入图片，**都从零创建一个与图片同形状的扰动张量 δ（1×3×1024×1024），单独优化**——图片之间互不共享扰动；而检测器、LPIPS、SSIM 三个网络全程只加载一次、所有图片共享。

The pipeline (`attack()` in `attacks/vida.py`) has 6 stages. **Per image, a fresh perturbation tensor δ of the image's shape (1×3×1024×1024) is created from zero and optimised independently** — no perturbation is shared across images. The detector, LPIPS and SSIM networks, in contrast, are loaded once globally and shared by every image.

### Stage 0 — 干净图跳过 / Clean skip
用**评估器自己的官方 transform**在原图上跑两个检测器：如果本来就都判 Real（典型情况：真实图；以及检测器"看走眼"的假图），直接**原样返回**——SSIM=1、LPIPS=0，白拿满分 2 分。对检测器误判为 Fake 的真实图，则照常攻击拉回（实测 13/100 张真实图被检测器误判，极小扰动即可修正，真实图平均得分 1.9999）。

Run both detectors on the untouched image through the evaluator's own transforms; if every detector already predicts Real (true for reals, and for easy fakes), return the image unchanged for a free 2.0 (SSIM=1, LPIPS=0). Misclassified reals are lightly corrected.

### Stage A — MI-FGSM 动量攻击获取 / Acquisition
两个检测器的 target margin loss（softplus，把目标类 logit 推高过错误类）求和；动量 μ=1.0、梯度按绝对值均值归一化、sign 更新、步进 2/255、上限 ε=8/255；扰动乘边界掩码。两个检测器一被骗到就早停；没骗到则进入 Stage A.2 的 "tail" 最多 60 步续攻（难图多吃迭代）。MI-FGSM 对频域检测器是必需的——纯符号 I-FGSM 在 100 步时只能骗过 1–2/8。

MI-FGSM (momentum µ=1.0, mean-abs normalised gradients, sign steps of 2/255, L∞ clamp 8/255, border mask) on the sum of both detectors' targeted margin losses. Early stop as soon as both detectors are fooled; a 60-step tail handles hard images. Momentum is essential for the DCT detector (plain I-FGSM saturates at ~1–2/8).

### Stage A.5 — 二分线搜索剥除冗余扰动 / Binary line search
Stage A 的符号步进会"过度攻击"（每步整个可见区域一起动，骗到的时候扰动已远超必要）。拿到骗到的 δ 后，在尺度 t∈[0,1] 上做二分搜索：用**评估器官方 transform 在 uint8 存盘图上判定**，找到仍能骗过两个检测器的最小 t，令 δ ← t·δ。这一步直接、且保证安全地把多余幅度砍掉（12 次二分，每次只是两个检测器的小分辨率前向，约 0.1 秒）。

Sign-step acquisition overshoots badly. Once fooled, we binary-search the smallest scale t of the whole perturbation that still fools both detectors **as measured by the official transforms on the uint8 image**, and set δ ← t·δ. This provably strips redundant magnitude at zero risk.

### Stage B — Adam 质量恢复（自适应多阶段）/ Adam quality recovery
固定 δ 的方向不如把扰动"重新分配"到感知更便宜的像素上。用 Adam 在连续值 δ 上直接优化：
- 目标：最大化 Q = 0.5·SSIM + 0.5·(1−LPIPS)（与官方同指标，可微）；
- 约束：对已骗过的检测器加 softplus 裕量项，裕量 κ **从 1.0 退火到 0.1**——让检测器的 logit 差贴住判定边界即可，不必"用力过猛"（真正的安全网是 uint8 真值门控）；
- 每个候选 δ 都投影回 ε 球和边界掩码，并且**只有当官方 transform 在 uint8 图上仍判定两个检测器全被骗过时才接受**；
- 每阶段 17 步 Adam、阶段之间重置动量并再插一次线搜索；lr 后期 ×0.6 做精细抛光；
- **自适应预算**：若某阶段后线搜索已无缩放空间（hi>0.999）且 Q 提升 <0.002 就早停（简单图少花时间），难图最多吃约 2 倍预算。

Adam optimises the continuous perturbation to maximise the differentiable official metric Q, with a softplus margin buffer (annealed 1.0→0.1 so fooled margins hug the decision boundary), ε-ball projection, and a **ground-truth accept gate**: a candidate is kept only if the official transforms on the uint8 image still fool the detectors. Phases reset momentum, run a line search between them, and the phase loop is adaptive (early stop when line search has no slack and Q is flat; hard images get up to ~2× budget).

### Stage C — uint8 存盘复核与续攻 / Verify and re-attack
最终交付的是 round 后的 uint8 图（评估器对它评分），而优化在 float 域进行，存在舍入间隙。因此最终再用官方 transform 复核：哪个检测器在存盘图上"漏网"，就从当前 δ 继续 MI-FGSM 续攻（最多 3 轮 ×20 步），并在整个过程中始终保留**（骗到检测器数, Q）最优**的那张 uint8 图。这保证 ASR 不会因任何 float/uint8 或预处理微差而掉。

Because the scored artefact is the rounded uint8 image, we re-verify with the official transforms and, if any detector slipped, continue MI-FGSM (up to 3×20 steps). The best uint8 image by (detectors fooled, then Q) seen anywhere in the pipeline is always retained — guaranteeing the fooling rate.

### Stage D — 难图救援通道 / Rescue pass
若第一轮（默认快配置：2/255 步长、80+60 步）结束后 Q < 0.90 或仍有检测器没骗过，则**从零重跑一轮更细步长（1/255、160+80 步）**的完整流程。原理：小 L∞ 步长沿更短的路径到达决策边界，在难图上扰动结构显著更小；两轮结果由 `best` 机制自动择优。实测救援通道只对约 16 张难图触发，把多张图从 1.4–1.6 拉到 1.99。

If the first (fast, 2/255 step) pass ends with Q < 0.90 or a detector not fooled, run a second, independent pass from zero with a finer acquisition step (1/255, 160+80 steps): smaller steps trace a shorter path to the decision boundary. The best-of-both image is kept automatically. The rescue triggers on ~16 hard images and lifts several from 1.4–1.6 to ~1.99.

---

## 4. 演进过程与每步"为什么调" / Evolution and rationale

全部数字均由**官方 `evaluate.py` 不改动**在 200 张 dev 图上跑出（日志：`my_attack/results_vida_*.log`）。

All numbers below come from the **unchanged official `evaluate.py`** on the 200 dev images.

| 版本 / Version | 总分 / Score (/400) | 假图均贡献 / Fake mean | 真实图 / Real mean | SSIM | LPIPS | 改动动机与内容 / What changed and why |
|---|---|---|---|---|---|---|
| v2（09-02 收尾） | ≈349 | 1.747 | 盲目扰动 ~1.75 | 0.890 | 0.143 | MI-FGSM 获取 + 符号步质量恢复；**问题：LPIPS 太高，且对真实图也白白加扰动** / sign-step recovery; high LPIPS; reals perturbed needlessly |
| v3.0 | 384.31 | 1.844 | 1.999 | 0.959 | 0.0376 | 四个修正：(1) LPIPS/SSIM 全局缓存；(2) **干净图跳过**（真实图白拿满分）；(3) **二分线搜索剥冗余**；(4) **Adam 恢复 + 官方 uint8 真值门控**，替代粗符号步 / global model cache; clean skip; binary line search; Adam recovery gated by official uint8 prediction; re-attack loop |
| v3.1 | 391.60 | 1.916 | 2.000 | 0.983 | 0.0246 | 发现难图被**固定裕量 κ=1.0** 卡住（不敢把 margin 贴边界）→ **κ 从 1.0 退火到 0.2**，且 Adam 改 30+30+20 三段、段间插线搜索、动量重置 / fixed margin buffer kept perturbation larger than needed; anneal κ 1.0→0.2; three Adam phases with inter-phase line search and momentum reset |
| v3.2 | 394.50 | 1.945 | 2.000 | 0.989 | 0.0164 | 固定 3 阶段对简单图浪费、对难图不足 → **自适应阶段**：无松弛且 Q 走平即早停，难图预算翻倍；κ 下限降到 0.1；后期 lr×0.6 / adaptive phase budget (early stop on convergence, ~2× budget for hard images), κ floor 0.1, late lr decay |
| **v3.3** | **397.20** | **1.972** | **2.000** | **0.993** | **0.0070** | 仍有 ~8 张难图 LPIPS 0.27–0.39（大步长过度攻击的结构性失真，恢复阶段难以弥补）→ **Stage D 救援**：细步长（1/255）+ 更多迭代从零重跑，最优择优 / 8 hard fakes still needed large structured distortion; fine-step rescue pass from zero, best-of-two kept |

关键洞察 / Key insights behind the tweaks:

1. **评估在 uint8 存盘图上做，优化在 float 上做**——所有"是否安全/是否更好"的判定都必须回到官方 transform + uint8 舍入上，否则可能白优化甚至掉 ASR。这就是为什么"真值门控"贯穿 A.5/B/C/D。/ The scorer uses uint8 images while optimisation is in float — every safety/quality gate must run the official transforms on the rounded image.
2. **"骗到"之后的博弈从攻击问题变成质量问题**：margin 只要为负即可，任何多余的欺骗强度都是白付的感知代价。线搜索（砍幅度）+ Adam（重分配到便宜像素）+ κ 退火（允许贴边界）都是这一原则的执行。/ Once fooled, margin only needs to stay negative; all extra fooling strength is wasted perceptual cost.
3. **粗步长是难图失真的根源**：2/255 符号步在难图上走到 ε 上限才能骗过，结构性失真大；细步长（1/255）沿更短路径到边界，救援通道在难图上单图提升最高 +0.56 分。/ Coarse 2/255 steps force hard images against the ε cap with large structured distortion; finer 1/255 steps reach the boundary shorter, gaining up to +0.56 per hard image.
4. **全局缓存与早停让"对每张图下重注"变得便宜**：单图平均约 40–90 秒（L40），简单图几秒（真实图瞬时返回），难图约 2–4 分钟。/ Caching and adaptive stopping make heavy per-image optimisation cheap in aggregate.

### 救援通道逐图增益（v3.2 → v3.3，前 9）/ Rescue per-image gains

| 图 / Image | v3.2 | v3.3 | 增益 / Gain |
|---|---|---|---|
| 001560 | 1.430 | 1.993 | +0.563 |
| 001553 | 1.524 | 1.994 | +0.470 |
| 001538 | 1.457 | 1.904 | +0.447 |
| 001589 | 1.548 | 1.965 | +0.417 |
| 001596 | 1.589 | 1.986 | +0.398 |
| 001521 | 1.732 | 1.992 | +0.261 |
| 001539 | 1.385 | 1.640 | +0.254 |
| 001569 | 1.819 | 1.987 | +0.168 |
| 001556 | 1.530 | 1.664 | +0.133 |

救援合计 +2.70 分。v3.3 假图中位数 1.986，仅 2 张 <1.8（最难的 001539/001556 仍有 0.18–0.21 LPIPS，是仅剩的约 2.8 分空间，需要额外策略，见 §7）。

The rescue pass adds +2.70 points in total. Median fake contribution is 1.986; only 2 images remain below 1.8.

---

## 5. 最终结果细目 / Final numbers (v3.3, official evaluator)

| 指标 / Metric | 全部 200 张 / All | 100 假图 / Fakes | 100 真实图 / Reals |
|---|---|---|---|
| 平均单图贡献 / Mean contribution | 1.986 | 1.972 | 2.000 |
| 最差单图 / Worst image | 1.640 | 1.640（001539） | 1.996 |
| 平均 SSIM | 0.9930 | 0.9861 | 1.0000 |
| 平均 LPIPS | 0.0070 | 0.0140 | 0.0001 |
| 两检测器 ASR | **100% / 100%** | 100% | 100%（87 张原样返回、13 张微扰修正） |
| **总分 / Total** | **397.20 / 400** | 197.20 | 199.99 |

参考基线（同 dev 集，09-02 的 v2 实验）/ Baselines (same dev set, v2 experiments):
naive joint MI-FGSM 固定 80 步：假图 1.508；v2 ViDA（mask+早停+tail）：1.592；v2 + 符号步恢复：1.747。v3.3 假图 1.972，相对 v2 恢复版 **+0.225/张（+12.9%）**，相对 naive **+0.464/张（+30.8%）**。

vs. naive joint: fake mean 1.508 → 1.972 (+30.8%); vs. yesterday's v2 recovery: 1.747 → 1.972 (+12.9%).

---

## 6. 黑盒迁移（非计分，补充）/ Black-box transfer (not graded)

Oleg 加了两个纯迁移实验用检测器（不计官方分数）：`npr`（ResNet-50）与 `aide`（ConvNeXt-XXL）。v3 的优化目标只含两个计分检测器；DI（Diverse-Inputs，`DI_PROB=0.5` 开关）可把对 NPR 的黑盒迁移从 57.5% 提到 82.5%（白盒 ViT 略降为 90%），AIDE 始终为 0%（SRM/频域预处理 + 超大骨干，对迁移攻击鲁棒）。官方分数目标用 `DI_PROB=0`（默认）。迁移展示与官方分数是两个操作点，见 `SUMMARY.md` §6。

For transfer experiments only: DI (switch `DI_PROB=0.5`) raises NPR black-box transfer 57.5%→82.5% at a small white-box ViT cost; AIDE stays 0%. Default (`DI_PROB=0`) maximises the official score.

---

## 7. 剩余空间与后续 / Remaining headroom and next steps

- 仅剩约 2.8 分空间，集中在 2–3 张极难图（001539: 1.640、001556: 1.664），其 LPIPS 0.18–0.21，说明在 1/255 步长下仍需接近 ε 上限的结构性扰动。可选策略：多种子/多随机起点攻击取最优、显式 L2 最小化的内层优化、或对这几张图单独诊断哪个检测器是瓶颈（margin 分析）。/ Only ~2.8 points remain, concentrated in 2–3 extremely hard images; options: multi-seed/best-of-k acquisition, explicit inner L2 minimisation, per-detector bottleneck diagnosis.
- 官方 `AADD_2026_Test` 集发布后：`attacks/vida.py` 无需改动即可直接跑（攻击模块完全自包含，自动适配不同分辨率）；若官方图片含未误判真实图，Stage 0 自动原样返回。/ No code change needed for the official test set; the module is self-contained and resolution-adaptive.
- 运行耗时（L40）：真实图瞬时；假图约 40–90 秒，难图救援约 2–4 分钟。/ Runtime on L40: reals instant; fakes ~40–90 s; rescue images ~2–4 min.

## 8. 难图瓶颈诊断与多样性重启（v3.4）/ Hard-image bottleneck diagnosis

v3.3 之后剩余 ~2.8 分集中在两张图（001539: 1.64、001556: 1.66）。诊断脚本 `my_attack/diag_hard.py`，对照图 001553（救援成功）、001549（简单）：

After v3.3, ~2.8 points remained concentrated in two images. Diagnosis (`my_attack/diag_hard.py`), with 001553 (rescue-fixed) and 001549 (easy) as controls:

**发现 1：ViT 是瓶颈，DCT 几乎免费 / ViT is the binding detector; DCT is nearly free.**

| 指标 / Metric | 001539 | 001556 | 001553（对照） | 001549（对照） |
|---|---|---|---|---|
| 干净图 ViT / DCT margin（logit 差） | +2.24 / +2.17 | +2.19 / +2.15 | +2.19 / +2.17 | +2.21 / +2.18 |
| 联合攻击后最小翻转尺度 t（ViT / DCT） | **0.94 / 0.19** | **0.94 / 0.63** | 0.19 / 0.69 | 0.31 / 0.75 |
| 只攻 ViT 的最小扰动 Q（SSIM/LPIPS） | 0.767（0.83/0.30） | 0.799（0.86/0.26） | 0.978 | 0.953 |
| 只攻 DCT 的最小扰动 Q | **0.991**（0.995/0.014） | **0.997**（0.998/0.004） | 0.953 | 0.943 |

两检测器干净置信度几乎相同（90% Fake，margin ≈ +2.2），但骗 ViT 需要接近 ε 上限的**空间密集**扰动（单攻 56–73% 可见像素撞上限），而 DCT 只需中心区很小扰动——这与 ViT 全局注意力、DCT 仅看中心 128 灰度的结构一致。

All four images have near-identical detector confidence (~90% Fake, margin ≈ +2.2); the difference is that moving ViT requires a spatially dense perturbation near the ε cap (56–73% of visible pixels saturate in a ViT-only attack), while DCT flips with tiny centre-region perturbations — consistent with ViT's global attention vs. DCT's 128-px grayscale crop.

**发现 2：两检测器攻击梯度近乎正交 / The acquisition gradients are near-orthogonal.**

中心共享区（两检测器都看得见）的梯度符号一致率只有 **0.50（随机水平）**，cos ≈ 0；环区一致率 0.02（DCT 无梯度）。不是强对抗，但联合梯度和互相稀释——好在 DCT 在很小尺度就翻转，瓶颈完全在 ViT 一侧。

Sign-agreement in the shared centre region is 0.50 (chance level), cosine ≈ 0 — not antagonistic, but the joint sum dilutes each detector's direction. DCT flips at small scale regardless, so the bottleneck is entirely ViT.

**发现 3：高质量解存在但轨迹混沌（关键）/ Cheap solutions exist; the trajectory is chaotic (key finding).**

同一代码、同一张图，独立运行 v3.3 攻击 6 次：001539 的 Q 在 **0.70–0.98** 之间大幅波动，001556 在 0.77–0.86 之间；好解（LPIPS 0.01–0.05、linf 6–7/255、margin 贴边界）确实存在，但 uint8 接受门控在决策边界附近是离散的，数值噪声导致 Adam 恢复轨迹在"好/坏盆地"间分叉。

Repeated v3.3 runs on the same image varied wildly (001539 Q 0.70–0.98, 001556 0.77–0.86): the cheap solution (LPIPS 0.01–0.05, perturbation 6–7/255, margin hugging the boundary) exists, but the discrete uint8 accept gate at the decision boundary makes the recovery trajectory bifurcate between cheap and expensive basins from ~1e-6 numeric noise.

**对策：多样性 best-of-k 重启 / Fix: diverse best-of-k rescue passes.**

Stage D 改为最多 6 个救援配置循环，相邻配置在**采集步长（1/255 vs 2/255）、随机 PGD 起点半径（0–6/255）、采集期 DI 开关**三个维度上变化，一旦 Q ≥ 0.92 即早停，全程由 `remember()` 按（骗到的检测器数, Q）择优保留。验证结果（两图各 2 次调用）：

Stage D now cycles up to 6 rescue configs varying acquisition step, random PGD-start radius and per-pass Diverse-Inputs, stopping at Q ≥ 0.92, keeping the best image across passes. Repeated validation:

| 图 | 旧 v3.3（单次） | v3.4（4 次调用） |
|---|---|---|
| 001539 | 0.82（LPIPS 0.18） | **0.974 / 0.985**（LPIPS 0.020 / 0.010） |
| 001556 | 0.77–0.86 | **0.949 / 0.989**（LPIPS 0.042 / 0.008） |

每次调用都在 1–2 个救援配置内命中并早停（50–104 秒/张）。简单图不受影响（Pass 1 即达 Q≥0.92，不触发救援）。

Every call now hits a cheap solution within 1–2 rescue passes (50–104 s/image); easy images are unaffected (pass 1 already above the Q threshold).

## 9. 复现 / Reproduce

```bash
cd team_repo
# 开发集 200 张（configs/dev_celebA.yaml: original_root 指向 celebA/TEST，models_dir: weights）
python evaluate.py --config configs/dev_celebA.yaml
```
日志 / Logs: `my_attack/results_vida_v3{,.1,.2,.3}_dev200.log`；JSON: `team_repo/experiments/results_vida_v3_dev200.json`。
主交付 / Main deliverable: `team_repo/attacks/vida.py`（DI 开关默认关闭供官方分数使用）。
