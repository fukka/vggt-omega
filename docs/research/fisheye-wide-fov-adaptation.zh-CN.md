# 将冻结的深度 / 3D 基础模型适配到鱼眼与大 FoV 输入

> 本文是 [`fisheye-wide-fov-adaptation.md`](fisheye-wide-fov-adaptation.md) 的中文版，并补充了
> 2026-07-31 调查新增的 §6。
>
> 原始综述日期：2026-07-29。每一篇论文都对照过其 arXiv 摘要页、官方项目页或官方仓库；§7 列出无法确认的内容。
> **"已核实"** = 我抓取了一手来源并读过。**"声称"** = 论文自己的主张，我没有独立复现。

---

## 1. 范围与约束

目标：让 **Depth Anything V2**（[arXiv:2406.09414](https://arxiv.org/abs/2406.09414)，NeurIPS 2024）
和 **VGGT / VGGT-Ω**（[arXiv:2503.11651](https://arxiv.org/abs/2503.11651)；
[arXiv:2605.15195](https://arxiv.org/abs/2605.15195)，CVPR 2026 Oral）在**鱼眼 / 大 FoV**
图像上运行——这里是 Aria 214-1 KB4 鱼眼——并且在**不重训练 backbone** 的前提下处理畸变。
允许的预算：什么都不做、一个重投影包装层、少量可学习 embedding、一个 LoRA/adapter，
或者位置编码上的一个小残差。完整重训练不在范围内。

本综述针对的具体失败：**VGGT 把它的深度输出耦合到了自己估计的 FoV 上**。
这是结构性的，不是偶然的——`pose_enc` 是 `[T(3), quat(4), fov_h, fov_w]`，
而 depth→points 的转换要除以由这两个通道导出的 `f = (H/2)/tan(fov/2)`
（本仓库 `vggt_omega/utils/pose_enc.py:16,24-26,32-43`）。
在从鱼眼帧切出的窄 tangent crop 上，`pose_enc[7:9]` 是错的，于是几何相对输入发生弯曲。
DAv2 没有这样的通道，在同样的 crop 上保持对齐。
因此最重要的文献是：(i) 相机估计能否被**覆写**而不是被推断；
(ii) 更深层的原因是不是**位置编码里的针孔先验**，而非 FoV head。

有两篇论文几乎正面回答了这两个问题：**OmniVGGT**（覆写，§3.2）与 **RayTun3R**（PE 先验，§3.3）。

---

## 2. 三大技术路线

| | **(a) 重投影输入 → 冻结的针孔模型** | **(b) 让模型具备相机感知 / 相机条件化** | **(c) 冻结 backbone 上的小型畸变/相机 embedding 或 adapter** |
|---|---|---|---|
| **代表工作** | VGGT-360、360MonoDepth、OmniFusion、PatchFusion、Depth Pro（多尺度 patch）、Depth Anywhere（cubemap 伪标签）、PaGeR（cubemap + DA3） | DAC、UniK3D、UniDepth/V2、Metric3D/v2、Pow3R、MapAnything、X-Lens、Prompt Depth Anything | **Fisheye3R**、**RayTun3R**、**OmniVGGT**（GeoAdapter）、RePer-360、FishRoPE、LoRA3D |
| **怎么处理畸变** | 在模型看到之前就去掉：gnomonic/tangent 或 cubemap patch 局部是直线的。畸变从不进入网络，只在 fuse-back 步骤重新进入。 | 编码成网络显式消费的相机表示：ERP 规范化（DAC）、球谐射线束（UniK3D）、稠密 ray map（Pow3R、MapAnything）、规范焦距缩放（Metric3D）、通用反投影图（X-Lens）。 | 畸变留在图像里；修正的是网络的**接口**。学习到的 token 重新校准特征（Fisheye3R），或给位置编码加径向/角向残差使 token 几何匹配镜头（RayTun3R、FishRoPE），或用零初始化的旁路注入已知内参（OmniVGGT）。 |
| **训练成本** | backbone 为零。VGGT-360 完全免训练。部分成员只训练一个 fusion 网络（OmniFusion、PatchFusion）。 | 最高。DAC / UniK3D / UniDepth / Metric3D / Pow3R / MapAnything / X-Lens 全都是从头训练或完整微调（MapAnything：64×H200 训 10 天；Pow3R：8×A100 训 5 天）。 | 非零里最低。RayTun3R：**10,752** 参数，约 30 个三帧窗口，每场景 2–3 小时。Fisheye3R：**约 344 k** 参数，40 k iters / 4 GPU 约 20 小时。OmniVGGT 的 GeoAdapter 是 26.8 M 参数，但训练用了 32×A100 十天。 |
| **backbone 冻结？** | 是，完全冻结。 | 否。 | Fisheye3R 是（声称）、RayTun3R 是（已核实："all backbone weights remain frozen"）、RePer-360 是、FishRoPE 是（冻结 DINOv2 + LoRA）。**OmniVGGT 否**——它从 VGGT 权重初始化并微调。 |
| **已知失效模式** | fuse-back 步骤的接缝与逐视图尺度不一致；每个 crop 内部丢失了真正的大 FoV 上下文；以及本仓库测到的情况——**窄 crop 本身就会触发 backbone 的相机自估计出错**，所以"局部直线化"对 VGGT 并不充分。成本随视图数增长。 | 推理时需要标定（DAC：是，已核实）。需要本项目排除的重训练。UniK3D 指出没有角度损失时大 FoV 输出会**收缩**。 | Fisheye3R：无公开代码（已核实为占位仓库，2026-07）；需要相机类型分类器与掩码注意力以避免退化透视输入。RayTun3R：假设**已知标定**，且**逐序列**适配——不是单一全局 adapter。FishRoPE：无深度结果。 |

该领域自己的综述——*Panoramic Scene Understanding: A Survey from Distortion-Aware Engineering to
Sphere-Native Modeling*，Zhu & Fan（[arXiv:2606.27745](https://arxiv.org/abs/2606.27745)）——
的横切观察是：sphere-native 算子"无法复用透视预训练的 backbone，因此没有 scale 起来"，
整个领域已经收敛到**几何感知的 tokenization**：改输入接口，保留预训练权重。
这正是路线 (c)，也正是本项目的约束所选中的那条。

---

## 3. 论文笔记

### 3.1 免训练 / 测试时投影策略

**VGGT-360: Geometry-Consistent Zero-Shot Panoramic Depth Estimation** — Jiayi Yuan, Haobo Jiang,
De Wen Soh, Na Zhao。CVPR 2026。[arXiv:2603.18943](https://arxiv.org/abs/2603.18943) ·
代码 [github.com/Yuanjiayii/VGGT-360](https://github.com/Yuanjiayii/VGGT-360)
（**已核实：有真实代码** — `main.py`、`utils/`、`vggt_visfeat/`、`assets/`；无 LICENSE，README 中无 checkpoint）。
三个免训练模块，全部从方法部分核实。
*不确定性引导的自适应投影*：从 `N_B ≥ 6` 个 base view 出发实现全覆盖与可控重叠，
用 Sobel 梯度导出的逐像素不确定性 `U(p) = σ(−Z(p))`（其中 `Z(p) = (G(p) − median)/τ`）打分，
再给 top-K 最不确定的 base view 各加两个预定义 yaw/pitch 偏移的邻居视图。
*结构显著性增强注意力*：在 VGGT 帧内注意力的 logits 上做**加性 log-confidence bias**，
`softmax(QKᵀ/√d + log(M_s))`——是 softmax 之前的 bias，不是对 value 的重新加权，权重不变。
*相关性加权 3D 修正*：从最后一层注意力图计算 sharpness（归一化香农熵）、locality（高斯空间紧致度）
和 symmetry（Bhattacharyya 系数），相加归一化成逐点权重 `C_vk`，用 `D_erp(r) = Σ C_vk D_vk / Σ C_vk` 融合。
在 Stanford2D3D 与 Replica360-2K 上**声称** Abs Rel 相对此前 SOTA 提升 27–36%。

*为什么重要：* 这是被移植的锚点论文。注意这篇论文**没有**包含的东西（已核实）：
没有讨论用的是 VGGT **预测的**内参还是每个视图**已知的**渲染内参，也没有 FoV/焦距误差分析。
因此本仓库测到的 FoV 耦合效应是上游方法真正的空白，不是它处理过而你漏看了。

**360MonoDepth** — Rey-Area, Yuan, Richardt。CVPR 2022。
[arXiv:2111.15669](https://arxiv.org/abs/2111.15669)。
把 ERP 投到一组切平面，每个 tangent view 跑现成的**透视**深度估计器，
再用**可变形多尺度对齐 + 梯度域融合**重组。透视模型完全不动。
*为什么重要：* 路线 (a) 中 fuse-back 那一半最干净的先验工作，它的融合阶段比置信度加权平均更能处理接缝。
如果 ADT 鱼眼融合出现接缝，这就是要借鉴的配方。

**OmniFusion** — Li, Guo et al。CVPR 2022 Oral。
[arXiv:2203.00838](https://arxiv.org/abs/2203.00838)。
Tangent patch → 逐 patch CNN → 把 **3D 几何特征与 2D 图像特征拼接**的 fusion 来吸收 patch 间的不一致，
外加跨 patch 的自注意力 transformer 与迭代深度精化。
*为什么重要：* 它明确指认 patch 间的尺度/偏移不一致为 tangent-patch 方法的核心问题，
并用显式几何特征解决——和 VGGT-360 用 cross-view attention 治的是同一个病。
第一作者与 DAC 相同，这条谱系值得知道。

**PatchFusion** — Zhenyu Li, Shariq Farooq Bhat, Peter Wonka。
[arXiv:2312.02284](https://arxiv.org/abs/2312.02284)。
粗糙的全局预测 + 更精细但不一致的 tile，由一个 patch-wise 网络在高层特征引导下融合；
Global-to-Local 模块提供上下文，因此不需要 patch 选择启发式；
Consistency-Aware Training/Inference 显式惩罚 tile **重叠**区域的不一致。基于 ZoeDepth。
*为什么重要：* CAT/CAI 是可以直接移植的想法——在你的 tangent view 之间加一个重叠一致性目标
或测试时一致性检查，与 VGGT-360 基于注意力的置信度正交。

**Depth Pro** — Bochkovskii et al。ICLR 2025。
[arXiv:2410.02073](https://arxiv.org/abs/2410.02073)。
多尺度 ViT 稠密预测；**无需相机元数据**即可输出 metric，并带一个专门的、SOTA 的**焦距估计 head**。
*为什么重要：* 反例，证明独立的 FoV head 可以是准的。它的焦距 head 是给 VGGT 喂外部估计
（而不是信任 `pose_enc[7:9]`）的候选——但见 §7，我没有核实它在鱼眼衍生 crop 上的行为。

**Depth Anywhere** — Ning-Hsu Wang, Yu-Lun Liu。NeurIPS 2024。
[arXiv:2406.12849](https://arxiv.org/abs/2406.12849)。
用透视深度模型作**教师**，通过**六面 cubemap 投影**给 360 图像打伪标签，
加上离线无效区域掩码与在线半监督联合训练。
*为什么重要：* 如果哪天你想要一个原生吃鱼眼的学生模型，这是最便宜的监督路线——
教师跑在它本来就正确的矩形 cube face 上。这是蒸馏，会训练一个学生；但*教师*保持冻结。

**PaGeR** — Bozic et al。[arXiv:2605.26368](https://arxiv.org/abs/2605.26368)。
通过**固定的 6×504×504 cubemap** 把 **Depth Anything 3**（da3-giant）提升到全景，
因此显存和运行时间与输入分辨率无关；一次前向返回 scale-invariant 深度、metric 深度、
世界坐标系法向和天空掩码。附带两个新数据集。
*为什么重要：* VGGT-360 最接近的已发表亲戚，但基于 DA3 且**有训练**——
它说明 cubemap-as-multi-view 这个技巧在现代前馈 FM 上有效，
且它固定预算的 cubemap 比 VGGT-360 的自适应方案更便宜。可作为 baseline；不是免训练。

### 3.2 相机条件化 / 相机提示的几何——各自如何*注入*相机

这一节回答"相机估计能否被覆写而非推断"。
跨四篇论文核实的简短答案：**一直胜出的注入格式是稠密的逐像素 ray map，patchify 后加到图像 token 上。**

**OmniVGGT** — CVPR 2026 Highlight。[arXiv:2511.10560](https://arxiv.org/abs/2511.10560) ·
代码 [github.com/Livioni/OmniVGGT-official](https://github.com/Livioni/OmniVGGT-official)（MIT）。
**对 VGGT backbone 而言最直接相关的相机条件化论文。**
已核实机制：一个 **GeoAdapter** 带两个分支（相机、深度），注入 VGGT 的**交替注意力（AA）块**。
相机分支是 `L+1 = 25` 个独立编码器，每个是**单层 linear**，每个 AA 块一个。
内参不是参数化成 `K`，而是 **`f ∈ ℝ²`，即视场角**——*和 VGGT `pose_enc[7:9]` 完全相同的两个数*——
与四元数 `q ∈ ℝ⁴` 和平移 `t ∈ ℝ³` 打包成 `g = {q, t, f}`。
位姿先相对相机 1 归一化（`Gⱼ′ = Gⱼ G₁⁻¹`）并按相机到原点的平均距离缩放。
注入是**零初始化卷积**，由可用性掩码门控：
`e′_{c,i,l} = e_{c,i,l} + 𝒵𝒞_l( m_i · e^aux + (1 − m_i) · placeholder )`。
深度用**单个**编码器：一层卷积，kernel 14（匹配 patch stride）。
训练用**随机多模态融合**——采样 `Q ∈ [0,S]` 张图接收相机参数、独立采样 `O ∈ [0,S]` 张接收深度，
再加 10% 的纯 RGB batch——这是推理时任意子集能工作的原因。
成本：GeoAdapter **26.8 M** 参数；**32×A100 训 10 天**，从 VGGT 权重初始化（所以 backbone 是微调的，不是冻结的）。
Table 4 **声称** `w/ K+RT` 在 AUC@30° 上优于纯 RGB。

*为什么重要：* 它证明了你想要的那种覆写在架构上是成立的——
VGGT 的相机信息可以在每个 AA 块通过一个零初始化的门控旁路被*提供*，
而且用的正是你的 `pose_enc` 已经在用的 `(q, t, fov_h, fov_w)` 参数化，不会破坏表示。
代价是：如已发表，这是一次 10 天重训。可移植的部分是**接口**，
以及"零初始化门控意味着未训练的 adapter 恰好是恒等映射"这个事实——
所以一个*微型*版本可以只在 ADT 上训练。见 I2。

**Pow3R** — Jang et al。CVPR 2025。[arXiv:2503.17316](https://arxiv.org/abs/2503.17316)。
已核实机制：内参变成**稠密 ray map**——像素 `(i,j)` 的射线是 `K⁻¹[i, j, 1]`——
然后**像 RGB 一样 patchify 并 embed**，通过**逐块 MLP** 注入编码器。
深度先验是 `[D/‖D‖, M] ∈ ℝ^{W×H×2}`，走同样的路径。
相对位姿 `P₁₂` 因为不是像素对齐的，改为在自注意力与交叉注意力之后**加到两个 decoder 的全局 CLS token 上**。
用**随机模态 dropout** 训练，覆盖 8.5 M 图像对。新增参数很小（head **+0.1%**，inject-1 变体的辅助注入 **+4%**），
但 DUSt3R backbone 是**微调**的，8×A100 上 224 px 训 3 天 + 512 px 训 2 天。
*为什么重要：* `K⁻¹[i,j,1]` 这个 ray map 是文献中最干净、最可移植的相机注入格式——
而且关键在于它**没有绑定针孔 `K`**。把 `K⁻¹` 换成 `VGGT-360-fisheye/utils/fisheye_cam.py`
里的 KB4 反投影 `kb4_unproject(u,v)`，*同样形状*的张量现在描述的就是 Aria 鱼眼。
已核实的注意事项：**Pow3R 从未测试非针孔内参**——机制可推广，证据不可。

**MapAnything** — [arXiv:2509.13414](https://arxiv.org/abs/2509.13414)。
同样的 ray map 思路，工业化版本。内参以**射线方向** `R^i ∈ ℝ^{3×H×W}` 进入，
经一个浅层卷积编码器（单次 **pixel-unshuffle 14**），投影到 DINOv2 隐维度（1024 × H/14 × W/14），
**与图像 patch 特征相加**。输出被刻意**因子化**：射线方向、沿射线的 up-to-scale 深度、
位姿（四元数 + 相对第 1 帧的 up-to-scale 平移），以及一个**全局 metric scale** `m`，
满足 `X_i^metric = m · X̃_i`。64×H200，6 + 4 天。
*为什么重要：* 两点。(1) 它是 Fisheye3R 适配的三个 backbone 之一，所以它的接口与 Fisheye3R 的 token 已知兼容。
(2) 论文**声称**通用*中心*相机标定都可以表示为射线方向，且方法"在适当训练下"可扩展到鱼眼——
即表示本身与 FoV 无关，尽管发布的模型没有这样训练。
**因子化**输出也值得抄：把"哪条射线"和"沿射线多远"分开，正是能阻止错误 FoV 弯曲深度的解耦。

**Depth Any Camera (DAC)** — Yuliang Guo et al。CVPR 2025。
[arXiv:2501.02464](https://arxiv.org/abs/2501.02464)。
已经是这里的 baseline，所以只记机制（已核实）：**ERP 是规范空间**——每个像素是一个 `(纬度 λ, 经度 φ)`，
patch 固定为 500×700。透视训练图像用**给定相机参数的畸变与投影函数**经 gnomonic 几何转成 ERP patch，
其中 **pitch-aware** 的部分是在 `λ_c` 上加噪声，让 patch 形状变化并落到高畸变纬度。
推理时做 **FoV 对齐**，缩放输入使其 FoV 匹配 crop 区域的 FoV。
Backbone 是 **iDisc**；SILog 损失；从图像尺寸导出虚拟焦距做 metric 缩放。
**推理时需要内参**（已核实）。
*为什么重要：* 正确定位了本仓库已有的 baseline——DAC 属于路线 (b) 且需要标定，而 Aria 你有。
它的 ITA 转换也可作为任何最小微调方案的*数据增强*：这是从透视图像合成高畸变训练样本的已核实配方。

**UniK3D** — Piccinelli et al。CVPR 2025。[arXiv:2503.16591](https://arxiv.org/abs/2503.16591)。
把**射线束**表示为学习到的**球谐**叠加——**最高 3 阶、排除常数分量、15 个谐波张量**——
与模型无关，因此没有针孔/矫正假设。输出活在**球面** 3D 空间，把相机与场景解耦，
**角度损失**与相机模块共同"防止大视场相机 3D 输出的收缩"。
*为什么重要：* 它命名的大 FoV **收缩**失效与你的弯曲伪影是近亲——
两者都是"模型把射线场搞错了，所以几何径向变形"。
15 系数的球谐射线场也是文献里最紧凑的相机参数化：可作为*学习式*少参数相机提示的候选（§4，I5）。

**UniDepth / UniDepthV2** — Piccinelli et al。CVPR 2024；
V2 [arXiv:2502.20110](https://arxiv.org/abs/2502.20110)。
一个**自可提示（self-promptable）相机模块**预测**稠密**相机表示，用它**条件化深度特征**，
输出在**伪球面**空间中把相机与深度解耦；**几何不变性损失**强制相机提示后的深度特征保持不变。
*为什么重要：* "self-**promptable**"这个词是重点——相机表示是*一个可以被 prompt 替换掉的稠密条件信号*。
这是覆写 VGGT FoV 的概念模板：不要跟估计器对抗，喂给它。
几何不变性损失也是最小 adapter 的正确目标形状——见 I3。

**Metric3D / Metric3Dv2** — Yin et al。ICCV 2023 / TPAMI。
[arXiv:2307.10984](https://arxiv.org/abs/2307.10984) · [arXiv:2404.15506](https://arxiv.org/abs/2404.15506)。
**规范相机空间变换**，明确设计为"可以毫不费力地插入现有单目模型"，有两种可互换形式：
**(1)** 训练时按 `f_canonical / f_original` 重缩放 GT 深度；
**(2)** **缩放输入图像**以模拟规范相机，同样按焦距比。推理时用去规范化变换把预测的 metric 深度映射回去。
*为什么重要：* 最便宜的相机条件化——它就是一次 **resize**，零参数、零训练。
形式 (2) 是真正只在测试时的操作。对于一个你*渲染出来因而知道真实 FoV* 的 tangent crop，
把 crop 缩放到冻结模型表现最好的那个等效焦距，是一行代码的实验。见 I1。

**Prompting Depth Anything** — Lin et al。CVPR 2025。
[arXiv:2412.14015](https://arxiv.org/abs/2412.14015)。
首次把 **prompting** 带到深度基础模型：一张低成本 **LiDAR** 深度图作为 prompt，
在**深度 decoder 内部多尺度融合**（不是 encoder）。
*为什么重要：* 证明 **decoder 侧**的 prompt 就足以重新锚定 DAv2 系模型的 metric 行为。
这里的类比：用稀疏的*已知几何*信号 prompt decoder——
例如 `fisheye_cam.py` 给出的解析逐像素入射角 `θ(u,v)`，或少量 ADT GT 深度——而不是改 encoder。

**X-Lens** — Heng Zhou et al。2026-07-14。[arXiv:2607.12993](https://arxiv.org/abs/2607.12993) ·
代码 [github.com/zhouhengamerica/XLens](https://github.com/zhouhengamerica/XLens)。
路线 (b) 与 (c) 最新的综合。用**通用反投影图**取代内参矩阵，无需改架构即可覆盖针孔、鱼眼和 360；
再加 (i) **逐层且逐相机类型的标定 token**（Fisheye3R 风格）、
(ii) 注入**交叉注意力**的 **Jacobian 畸变 bias**，使畸变鱼眼区域匹配无畸变针孔观测、
(iii) 用于射线空间位置编码的 **FishRoPE**。
输出因子化为 `(归一化深度, 置信度, 单一全局 metric scale)`，且**刻意不预测位姿或内参**，
因为部署时假设是标定过的装置。DINOv2 + DPT，0.04 B 参数，最高 41 FPS。
三阶段训练：针孔预训练 → 鱼眼 token 适配 → 异构联合。**声称** AbsRel 比最强 baseline 降低 25.4%。
*为什么重要：* 两个可直接偷的设计决策。
(1) 当标定已知时**彻底丢掉相机 head**——这是对 FoV 耦合最干净的回答。
(2) **交叉注意力中的 Jacobian 畸变 bias**：既然 KB4 的局部 Jacobian 是解析的，
这就是一个*零参数*的注意力修改。也独立确认了标定 token + 球面 RoPE 可以组合。

**AnyCalib** — Javier Tirado-Garín, Javier Civera。ICCV 2025。
[arXiv:2503.12701](https://arxiv.org/abs/2503.12701) ·
代码 [github.com/javrtg/AnyCalib](https://github.com/javrtg/AnyCalib)（Apache-2.0）。
回归**逐像素射线场**，由此可对针孔、Brown-Conrady **和 Kannala-Brandt** 以**闭式**导出内参；
能处理裁剪/拉伸过的图像。**声称**在标定上优于 3D 基础模型，尽管训练数据少得多。
*为什么重要：* 它是 FoV 耦合诊断的 sanity-check 仪器。
在同样的 tangent crop 上跑 AnyCalib，把它的 KB/针孔估计与 VGGT 的 `pose_enc[7:9]`
和已知渲染 FoV 比较——三个数，其中一个是 ground truth。
它也提供 RayTun3R 在缺少真实标定时所需的标定（RayTun3R 的附录 B 正是这么做的）。

### 3.3 几何 backbone 的参数高效适配

**RayTun3R: Online Camera Adaptation in 3D Foundation Models** — Daniil Sinitsyn, Nikita Araslanov,
Daniel Cremers（TUM / MCML）。2026-07-02。[arXiv:2607.02711](https://arxiv.org/abs/2607.02711)。
**无代码 URL**——论文说"our code will be made publicly available"。

**先读这一篇。** 它是与本项目约束最接近的已发表工作，*而且*它诊断了观察到的弯曲背后的机制。

*诊断（已核实）：* 失败被归因于预训练 3D FM **位置编码里的针孔偏置**，由 **Jacobian** 论证确立：
对针孔相机，反投影 Jacobian `J_κ⁻¹ = ∂κ⁻¹/∂(u,v)` 在整幅图上是常数，
因为一个像素的步长在任何地方都把视线方向改变同样的 `1/f_{x,y}`。对鱼眼它强烈依赖位置。
他们测量*位置嵌入* Jacobian 的最大奇异值 `σ₁` 和局部面元 `det(J_PE^⊤ J_PE)` 随归一化半径的变化，
发现预训练 embedding **近乎平坦**（针孔式），而适配后的变成**半径相关**。

*方法（已核实）：* backbone 完全冻结（**Depth Anything 3、VGGT、π³**），所有残差 adapter 参数零初始化。
绝对 PE 得到一个围绕标定主点的极坐标分箱残差：
`P′(u,v) = P_A(u,v) + t_r(ρ_{u,v}) + ρ_{u,v} · δ_θ(θ_{u,v})`，
其中 `t_r`、`δ_θ` 是 **20 个径向**和 **8 个角向**分箱上的可学习查找表。
RoPE 得到 `ω′(u,v) = ω(u,v) + Δ_r(ρ_{u,v})`，一个**跨 RoPE 频率共享的径向查找表**，每箱一个参数，**20 箱**。
对 DA3-Small（`C = 384`）：`(20 + 8) × 384` 个 PE 参数 + 20 个 RoPE + 开销 ≈ **10,752 个可训练参数**。
两个无参数的部分：有效镜头圆之外的 patch 用**有效 token 均值**替换；
每个 patch 按其中心处鱼眼→针孔映射的**局部线性化**重采样；
对 DPT 式的 head，规则的**预测网格坐标**被替换为经标定映射去畸变后的相机感知坐标。

*训练（已核实）：* 三帧窗口，`L_reproj`、`L_pose`（与 MAGSAC++ 位姿的角度差）、边缘感知平滑，
加上对位置修正的 L2 与全变分正则。适配集：每序列 **30 个三帧窗口**；
每个 ETH3D 场景端到端约 2–3 小时；**无额外推理成本**（约 100 ms/帧，与原版 DA3 相同）。

*结果（声称，按报告核实）：* 数据集 KITTI-360（185°）、TUM-VI（195°）、ScanNet++（115°）、
ETH3D（110°）、FIORD（200°）。DA3-Small 旋转误差 ETH3D 8.59° → 0.70°，KITTI-360 1.69° → 0.84°，
TUM-VI 10.41° → 2.41°；平移 ETH3D 15.16° → 4.48°，KITTI-360 12.81° → 2.92°。
深度 AbsRel ETH3D 0.178 → 0.107，ScanNet++ 0.282 → 0.108。**对 π³ 和 VGGT 也报告了改进**（Table 2）。
消融：**学习到的绝对 PE 残差贡献最大**；仅径向分箱就已完成大部分，角向分箱做精修；
**仅 RoPE 表现很差**；patch 去畸变单独作用很小。
附录 B 显示用 **AnyCalib 预测的**标定代替真实标定仍然有效，且优于 LoRA baseline 和 "CalTok" baseline。
Fisheye3R 被当作**同期工作**引用，**没有数值对比**。

*为什么重要：* 这是一个 1 万参数、冻结 backbone、兼容 VGGT 的 adapter，能修好鱼眼几何；
它的 Jacobian 诊断给了你一个**可测量的靶子**：
如果你 vendored 的 PE 的 `det(J_PE^⊤J_PE)` 对半径是平的，针孔先验就存在，FoV head 可能是症状而非病因。
两个诚实的限制：它**假设已知标定**（没问题——你有 Aria KB4），
且它从一小段时间片段**逐序列**适配，所以是在线适配，而非一个通用鱼眼 adapter。

**Fisheye3R** — Ruxiao Duan et al（Yale + Google）。ECCV 2026。
[arXiv:2603.28896](https://arxiv.org/abs/2603.28896) ·
[github.com/android-xr/fisheye3r](https://github.com/android-xr/fisheye3r) —
**已核实至今仍是占位仓库**：只有 `.github/`、`docs/`、`LICENSE`（Apache-2.0）、`README.md`，
写着 "Implementation — Coming soon."，无权重。
已核实机制：**每层 K = 8 个标定 token**，插入每个图像编码器层与每个交替注意力（帧内与全局）块，
**除了前 `L₀ = 12` 层**，初始化为 `N(0, 1e-6)`；
**插入后丢弃**——token 在第 `ℓ` 层与图像 token 拼接、参与注意力、然后丢弃，
这"把潜在的标定效应局部化到每一层"。
混合的透视/鱼眼 batch 由一个作用在 `L₀` 层 class token 上的**线性（逻辑回归）相机类型分类器**处理，
`M_s = 𝕀(ψ(x_{s,0}^{(L₀)}) > 0.5)`，加上二值注意力掩码，让标定 token 只影响鱼眼 token。
三种方案：**SSL** `L = ℒ(f(I^p); T⁻¹ ∘ f(T(I^p), φ))`——只用无标签透视图像，
畸变 `T` 合成鱼眼，*未适配*的模型作为教师；**SL**——透视 GT 经同样合成；**SL+**——直接用真实鱼眼 GT。
成本：**约 344 k** 可训练参数 vs **1.23 B** 冻结（Table 4），AdamW 1e-5 → 1e-7，
**40 k iters，4 GPU 约 20 小时**，峰值约 35 GiB。
适配 **VGGT、π³ 和 MapAnything**；**声称**在位姿、深度、点图和 **FoV** 估计上都有一致提升。
*为什么重要：* 另一个锚点。它的机制与 RayTun3R 正交（token vs PE 残差），
这正是 RayTun3R 的消融打败 "CalTok" baseline 之所以有信息量的原因，也是两者可以**叠加**的原因。
它的 **SSL** 方案是最突出的实用性质：**完全不需要鱼眼数据、不需要任何 GT**，
只需要无标签透视图像加上你的 KB4 前向模型——而本仓库已经有 `fisheye3r/distortion.py`。

**LoRA3D** — ICLR 2025。[arXiv:2412.07746](https://arxiv.org/abs/2412.07746)。
用模型**自己的多视图预测**把预训练的 DUSt3R 系模型特化到目标场景：
鲁棒全局优化对齐稀疏视图预测，预测置信度被**重新校准**以更好反映实际点精度，
校准后的置信度门控**伪标签**用于 **LoRA** 微调。无外部先验、无人工标注。
**声称**在 160+ 场景上提升最高 88%；**单 GPU 5 分钟**，每个 adapter **18 MB**。
*为什么重要：* 与其他一切配套的自监督引擎。
VGGT-360 重叠的 tangent view 已经给了 LoRA3D 消费的多视图一致性信号——
所以一个逐序列的 ADT adapter 可以**完全不用 GT 深度**训练出来。

**RePer-360** — Cheng Guan et al。2026-03-06。[arXiv:2603.05999](https://arxiv.org/abs/2603.05999)。
在保留预训练透视先验的同时把深度**基础模型**适配到全景：
一个轻量**几何对齐引导模块**从两种互补投影（**ERP** 与 **CP**/cubemap）导出调制信号，
一个 **Self-Conditioned AdaLN-Zero** 机制发出**逐像素缩放因子**，弥合透视→全景的特征分布差距。
**声称**用 **1% 的训练数据**就能超过标准微调，同域设置下 RMSE 提升约 20%。
*为什么重要：* AdaLN-**Zero** 与 OmniVGGT 的零初始化卷积、LLaMA-Adapter 的零门控是同一个技巧——
初始化即恒等，因此不会损坏冻结的先验。
"1% 数据"这个结果是本综述中最强的证据，说明**以投影为条件的特征调制**是高回报、低数据量的干预。
它作用在 DAv2 式模型上，因此可直接迁移到本项目的 DAv2 那一半。

**LLaMA-Adapter** — Zhang et al。ICLR 2024。[arXiv:2303.16199](https://arxiv.org/abs/2303.16199)。
可学习的适配 prompt 前置到较高的 transformer 层，通过**零初始化注意力与零门控**注入，
让新线索逐渐进入而预训练知识得以保留。冻结 7 B 模型上 **1.2 M** 可训练参数，8×A100 不到 1 小时。
*为什么重要：* OmniVGGT、RePer-360 与 RayTun3R 都依赖的零初始化门控的源头。
当你论证"初始化为恒等的 adapter 可以安全地装到 VGGT-Ω 上"时值得引用。

**Visual Prompt Tuning** — Jia, Tang et al。ECCV 2022。
[arXiv:2203.12119](https://arxiv.org/abs/2203.12119)。
参数量 <1%，只在**输入空间**可训练，backbone 冻结；常常打败完整微调。
*为什么重要：* Fisheye3R 标定 token 的正式祖先；"输入空间的学习 token 就够了"的参考文献。

**PPEA-Depth**（[arXiv:2312.13066](https://arxiv.org/abs/2312.13066)）与
**ER-LoRA**（[arXiv:2509.00665](https://arxiv.org/abs/2509.00665)）——专门针对**稠密深度回归**的 PEFT，
前者在 encoder *与* decoder 上渐进，后者用有效秩（effective rank）来挑选适配对象。
*为什么重要：* PEFT 可迁移到稠密回归而不只是分类的证据；
ER-LoRA 的秩准则是一种有原则的方式来选择*哪些* VGGT 块值得加 adapter，而不是靠猜。

**FiT / FiTv2** 与 **RoPE-ViT**。
FiT 把图像当作变长 token 序列处理，用掩码 MHSA 与 **2D RoPE**，
并指出朴素的 LLM 长度外推迁移得很差——因此有 **VisionNTK / VisionYaRN**。
*为什么重要：* 如果你以非原生分辨率或长宽比喂 VGGT tangent crop，
PE 重采样方案就是叠在鱼眼问题之上的一个混杂因素。
FiT 是刻意处理它的参考，而 VisionNTK 式的频率重缩放是一个**零参数**旋钮，
位置正好在 RayTun3R 放学习残差的地方。

### 3.4 针对畸变的位置编码与注意力手术

**FishRoPE** — Qualcomm。[arXiv:2604.10391](https://arxiv.org/abs/2604.10391)。未找到代码 URL。
已核实机制。像素通过**逆 Kannala–Brandt** 映射到角度：
`r = √((u−c_x)² + (v−c_y)²)`，`θ = r⁻¹_KB(r)`（多项式求逆），`φ = atan2(v−c_y, u−c_x)`。
embedding 维度对半分——`d/2` 给 `θ`，`d/2` 给 `φ`——各自做标准 RoPE 旋转，
于是注意力 logits 依赖**角度间隔而非像素距离**。
冻结 **DINOv2 ViT-B/14**，**LoRA r=16, α=32 作用在 query 与 value 投影上**，
86 M 冻结参数上约 **3 M** 可训练。**不是免训练。**
在 WoodScape 2D 检测（**声称** 54.3 mAP）与 SynWoodScapes BEV 分割（65.1 mIoU）上评估——**没有评估深度**。
*为什么重要：* 你需要的那个 KB4 → RoPE 构造被完整写出来了，用的是与 Aria 相同的镜头模型。
它是 RayTun3R *学习式*径向 RoPE 残差的*解析*对应物——
而 RayTun3R 的消融（仅 RoPE 表现很差）是一个警告：单靠 RoPE 手术可能不足以解决几何任务。
X-Lens 把 FishRoPE 用于深度是反面证据。

**SpheRoPE** — Or Hirschorn et al。2026-06-30。[arXiv:2606.32033](https://arxiv.org/abs/2606.32033)。
把球面先验注入**冻结的预训练**扩散 transformer，**免训练且免优化**，
方法是把 RoPE 的**低频通道重参数化为球面上的 3D 笛卡尔坐标**，同时**谐波量化高频通道**以强制精确周期性。
*为什么重要：* 存在性证明——**球面 RoPE 手术可以在冻结 transformer 上完全免训练**。
低/高频拆分是可迁移的洞见：低频承载值得重参数化的几何，高频主要只需保持周期性。
领域是生成而非深度——因此当作机制，不是证据。

**PanoFormer** — Shen et al。ECCV 2022。[arXiv:2203.09283](https://arxiv.org/abs/2203.09283)。
在**球面切空间上**划分 patch，使 token 畸变最小，并在自注意力模块中加入**可学习的 token flow**。
*为什么重要：* "切域 token"在架构上与 tangent view 渲染是同一个动作，
只是发生在 *tokenizer 内部*而非作为预处理——是路线 (a) 与 (c) 之间的中间路径。
与 RayTun3R 无参数的**局部线性化 patch 重采样**密切相关。

**EGformer** — Yun et al。ICCV 2023。[arXiv:2304.07803](https://arxiv.org/abs/2304.07803)。
与其试图去掉畸变，不如**把 equirectangular 几何用作局部注意力的显式 bias**。
*为什么重要：* "用已知几何给注意力加 bias"这个模式，也正是 VGGT-360 的 log-confidence bias
和 X-Lens 的 Jacobian bias 在做的事。对冻结模型而言，
在注意力 logits 上加一个几何 bias 是存在的最便宜的干预——零参数。

**SGFormer** — Junsong Zhang et al。[arXiv:2404.14979](https://arxiv.org/abs/2404.14979)。
把球面先验整合进 ViT，并把 decoder 改造成**球面先验 decoder（SPDecoder）**。
*为什么重要：* decoder 侧的几何先验——相关，因为 VGGT 的 DPT head 也带一个 2D 预测网格，
而那正是 RayTun3R 修正的对象。

**Sector Patch Embedding (SPE)** — Dianyi Yang et al（BIT）。2023-03-26。
[arXiv:2303.14645](https://arxiv.org/abs/2303.14645)。
采样**与鱼眼畸变模式对齐的圆形扇形 patch**，并用**可学习的极坐标**编码位置。
*为什么重要：* 用极坐标假设替换方形网格 patchify 假设。
注意它与 RayTun3R 极坐标分箱 PE 残差的强烈家族相似性——
围绕主点的极坐标是鱼眼的自然坐标图，无论用于采样还是用于索引 PE。

**SphereNet** — Coors, Condurache, Geiger。ECCV 2018。
调整卷积滤波器的**采样位置**以反转畸变，并把滤波器绕球面包裹；
因为它建立在常规卷积上，所以**能把现有的透视 CNN 模型迁移**到全向情形。
*为什么重要：* 本项目论点最早的干净表述——改变*你在哪里采样*，保留预训练权重。
另见 **Gauge Equivariant Convolutional Networks and the Icosahedral CNN**
（[arXiv:1902.04615](https://arxiv.org/abs/1902.04615)）。
两者都是 CNN 时代的，无法复用 ViT 权重——这正是综述所述 sphere-native 方法没能 scale 的原因。

### 3.5 鱼眼专用深度估计

**FisheyeDistanceNet** — Ravi Kumar et al。ICRA 2020。
[arXiv:1910.04076](https://arxiv.org/abs/1910.04076)。
自监督，从**原始**单目鱼眼视频估计**尺度感知的欧氏距离**与自运动，**无需矫正**。
*为什么重要：* "在鱼眼上预测欧氏 **range** 而非平面 z"这个约定的源头，
`CONTEXT.md` 已经把它当作承重结构。对本仓库评分域纪律的独立支持。

**SynDistNet** — WACV 2021。[arXiv:2008.04017](https://arxiv.org/abs/2008.04017)。
多任务：联合学习语义分割，并用其预测**引导**自监督距离估计。

**SVDistNet** — Ravi Kumar et al。IEEE T-ITS。
[arXiv:2104.04420](https://arxiv.org/abs/2104.04420)。
**要注意的机制：** "camera-geometry adaptive multi-scale convolutions which utilize the
**camera parameters as a conditional input**"，让单个模型跨不同鱼眼相机泛化而**无需逐型号重训**。
*为什么重要：* 本综述中最早的*以内参为条件来吸收镜头差异*的实例——
Pow3R/MapAnything ray map 与 OmniVGGT GeoAdapter 在基础模型之前的祖先。

**OmniDet** — Ravi Kumar, Yogamani et al。IEEE RA-L 2021。
[arXiv:2102.07448](https://arxiv.org/abs/2102.07448)。
在**未矫正**鱼眼上做六个任务，共享 encoder，并有"一种新的基于相机几何的适配机制，
**在训练与推理时都**编码鱼眼畸变模型"。
*为什么重要：* WoodScape/OmniDet 这条线是鱼眼原生稠密预测被做出来的地方；
它也是 FishRoPE 与 X-Lens 使用的评测场，因此是旧鱼眼文献与 2026 年 adapter 论文之间的桥梁。

### 3.6 前馈 3D 基础模型，以及哪些是相机参数无关的

这里重要的性质是：模型内部是否有一个相机估计，其深度可能*因它而错*。

| 模型 | arXiv | 代码 | 设计上相机参数无关？ |
|---|---|---|---|
| **DUSt3R** | [2312.14132](https://arxiv.org/abs/2312.14132) | [naver/dust3r](https://github.com/naver/dust3r) | **是。** 回归点图，"无需相机标定或视点位姿的先验信息"。 |
| **MASt3R** | [2406.09756](https://arxiv.org/abs/2406.09756) | [naver/mast3r](https://github.com/naver/mast3r) | 是 — DUSt3R + 稠密局部特征 head 与匹配损失。 |
| **Spann3R** | [2408.16061](https://arxiv.org/abs/2408.16061) | [HengyiWang/spann3r](https://github.com/HengyiWang/spann3r) | 是 — 外部空间记忆把点图放进全局坐标系。3DV 2025。 |
| **Fast3R** | [2501.13928](https://arxiv.org/abs/2501.13928) | [opencv.org/fast3r](https://opencv.org/fast3r/) | 是 — N 张图一次前向，无全局对齐。CVPR 2025。 |
| **CUT3R** | [2501.12387](https://arxiv.org/abs/2501.12387) | [CUT3R/CUT3R](https://github.com/CUT3R/CUT3R) | 是 — 循环持久状态，在线 metric-scale 点图。 |
| **MoGe** | [2410.19115](https://arxiv.org/abs/2410.19115) | CVPR 2025 | **是，而且很刻意。** 预测**仿射不变**点图；相机偏移、**焦距**与深度是事后*从*点图导出的——与 VGGT 的依赖方向相反。 |
| **MoGe-2** | [2507.02546](https://arxiv.org/abs/2507.02546) | NeurIPS 2025 | 把 MoGe 扩展到 metric 尺度而不丢失仿射不变的相对几何。 |
| **π³** | [2507.13347](https://arxiv.org/abs/2507.13347) | [yyfz/Pi3](https://github.com/yyfz/Pi3) | **是。** 完全**置换等变**，无参考视图。ICLR 2026。 |
| **VGGT** | [2503.11651](https://arxiv.org/abs/2503.11651) | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | **否。** 追加**相机 token**；**相机 head** 预测外参*与内参*。这就是耦合。 |
| **VGGT-Ω** | [2605.15195](https://arxiv.org/abs/2605.15195) | 本仓库 | **否** — 相同的 9 维位姿编码。CVPR 2026 Oral；显存少约 70%，**单一稠密预测 head + 多任务监督**，去掉了高分辨率层。 |
| **MapAnything** | [2509.13414](https://arxiv.org/abs/2509.13414) | [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) | **可选条件化** — 预测射线，但也*接受*射线。见 §3.2。 |
| **Depth Anything 3** | [2511.10647](https://arxiv.org/abs/2511.10647) | [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | "有或没有已知相机位姿"都能工作；一个**纯 DINO transformer** 就够，**单一 depth-ray 目标**取代多任务学习。**声称**位姿 +44.3% / 几何 +25.1% 优于 VGGT。 |

*这张表为什么重要：* MoGe 与 π³ 是 FoV 耦合实验的天然**对照组**。
如果同一个在 VGGT 下弯曲的 tangent crop 在 π³ 或 MoGe 下保持笔直——两者都是从几何导出相机而非反过来——
那么诊断就从架构侧得到了确认，与你已有的 DAv2 对照互补。
注意 RayTun3R 发现 π³ 与 VGGT 在鱼眼上*仍然*退化，尽管它们是相机参数无关的——
这正是它的论点：**更深层的原因是位置编码，而不是相机 head**。

---

## 4. 值得在这里尝试的想法

按 工作量 × 成功可能性 排序。每个想法都注明它借用了哪篇论文的机制。

### Tier 1 — 几小时的工作，信息价值高

**I1. 在 tangent crop 上做 Metric3D 规范化 resize（零参数、零训练）。**
机制：Metric3D 规范相机变换的**方法 2**——缩放输入以模拟规范焦距，再对输出做去规范化。
因为 tangent view 是你*渲染*的，它们的真实焦距已知。
扫描缩放因子，同时观察 `pose_enc[7:9]` 与 `align%`。
如果某个把 crop 等效焦距带到 VGGT 训练典型焦距的缩放让推断 FoV 收敛到真值、弯曲停止，
你就有了一个免费的修复和一个干净的解释。这是本文档中最便宜的实验，且直击测到的耦合。

**I2. 字面意义上覆写 `pose_enc[7:9]`。**
机制：OmniVGGT 把 `f ∈ ℝ²` = **视场角**当作辅助相机 token，通过零初始化卷积在每个 AA 块注入，
也就是说它把同样这两个数当作*输入*。在造任何 adapter 之前，先免费测试*下游*那一半：
正常跑 VGGT，然后用**已知渲染 FoV** 而不是预测值把 depth 转成 points
（你的 `pose_enc.py:42-43` 已经隔离了这一步——换掉 `fov_h`、`fov_w`）。
如果弯曲消失，耦合就纯粹在 depth→points 转换里，完全不需要训练。
如果仍然存在，FoV 估计已经污染了**特征**，你需要 I3/I4。
这一个实验就干净地划分了问题，值得在 Tier 2 之前做。

**I3. 在 vendored 的 PE 上复现 RayTun3R 的 Jacobian 诊断。**
机制：测量位置嵌入 Jacobian 的 `σ₁` 与 `det(J_PE^⊤ J_PE)` 随归一化半径的变化；平坦 ⇒ 针孔先验存在。
这是对 `vggt_visfeat/layers/vision_transformer.py` 与 `layers/attention.py` 的只读测量。
它告诉你该把力气花在 PE 残差（RayTun3R）还是标定 token（Fisheye3R）上——
RayTun3R 自己的消融说**绝对 PE 残差贡献最大**而仅 RoPE 表现不佳，
所以这个测量决定那 10 kB 参数该放哪。

**I4. 用相机参数无关的 backbone 做对照实验。**
机制：MoGe 从点图导出焦距而非反过来；π³ 完全置换等变、无参考视图。
把它们作为 `depth_probe.py` 的 **backend** 加到 `vggt1b` / `vggt_omega` / `official` 旁边。
按 RayTun3R 的预期结果：它们也会退化，但*方式不同*——有针孔 PE 偏置但无 FoV 耦合。
这在实验上分离了两个机制，而任何单模型实验都做不到。

### Tier 2 — 几天的工作，预期收益强

**I5. 用 KB4 ray map 作为相机提示（Pow3R 格式，VGGT 注入点）。**
机制：Pow3R 把稠密 ray map `K⁻¹[i,j,1]` 完全像 RGB 一样 patchify 并经逐块 MLP 注入；
MapAnything 用 pixel-unshuffle-14 卷积编码器做同样的事并加到 DINOv2 patch 特征上。
把你的 **KB4 反投影**（`VGGT-360-fisheye/utils/fisheye_cam.py`）替换 `K⁻¹`，
ray map 现在描述的就是 Aria 鱼眼——张量形状不变，
所以一个浅层卷积（stride 14 匹配 VGGT patch 网格）加一个**零初始化**输出卷积，
就是一个初始化即恒等的即插即用 adapter。
这是这里天花板最高的想法：它让模型看到*真实的镜头*而非矫正近似，并且完全不再需要 tangent crop。
已核实的注意事项：Pow3R 从未测试非针孔内参，MapAnything 只**声称**鱼眼"在适当训练下"可表示——
所以这是有机制依据但无证据依据的，需要自己的训练。

**I6. 在 VGGT-Ω 上跑 Fisheye3R SSL，不需要鱼眼数据。**
机制：`L_SSL = ℒ(f(I^p); T⁻¹ ∘ f(T(I^p), φ))`——用 `T` 合成鱼眼，
以*未适配*模型的透视预测经 `T⁻¹` 拉回作为监督。约 344 k 参数，40 k iters，4 GPU 约 20 小时。
本仓库已经有 `fisheye3r/train.py --scheme ssl` 与 `fisheye3r/distortion.py`，
剩下的工作是一批无标签透视图像加 GPU 时间。
它**不需要 ADT GT，也不需要鱼眼数据**，这使它成为风险最低的*训练*选项。
注意官方代码仍是占位（已核实），所以你的复现目前是唯一的实现——这在任何写作中都值得明说。

**I7. RayTun3R 式极坐标 PE 残差，并与 I6 叠加。**
机制：`P′(u,v) = P_A(u,v) + t_r(ρ) + ρ·δ_θ(θ)`，20 径向 + 8 角向分箱，
加 `ω′ = ω + Δ_r(ρ)`，20 个共享径向 RoPE 分箱，对 `C = 384` 约 **10,752** 参数；零初始化。
对 ADT 特别有吸引力的两个理由：ADT 序列正是它需要的"短时间片段"，
且该机制与 Fisheye3R 的 token **正交**——RayTun3R 打败了标定 token baseline，
所以叠加 PE 残差*和* token 是未被探索的，而且便宜。
另外立刻采用它的两个免费部分：**镜头圆外的有效 token 均值替换**（你已经有解析有效性掩码）
与**按 KB4 映射在每个 patch 中心局部线性化的 patch 重采样**。

**I8. 零参数几何注意力 bias。**
机制：三篇论文都是往注意力 logits 上加 bias 而不学习任何东西——
VGGT-360 的 `softmax(QKᵀ/√d + log M_s)`、EGformer 的 equirectangular 几何 bias、
X-Lens 的交叉注意力 **Jacobian 畸变 bias**。
KB4 的局部 Jacobian 是解析的，所以形如 `−λ · d_angular(i,j)` 的 bias 或 Jacobian 导出项
**零参数**、零训练。既然你已经为 VGGT-360 的显著性 bias 改了 `layers/attention.py`，
这只是对你自己代码的小扩展。Tier-2 里最便宜的；深度上的天花板未知。

**I9. LoRA3D 式自校准，用你自己重叠的 tangent view。**
机制：重新校准预测置信度使其反映真实点精度，用它门控伪标签，微调一个 **LoRA**——
单 GPU 5 分钟，每个 adapter 18 MB，无外部先验或标注。
VGGT-360 的重叠视图加上你的相关性加权融合已经产出了 LoRA3D 消费的多视图一致性信号与置信度。
这能得到一个**不需要 GT 深度的逐序列 ADT adapter**，且与 I7 可组合。

### Tier 3 — 值得知道，工作量更大或确定性更低

**I10. 给 DAv2 那一半用 AdaLN-Zero 自调制。** RePer-360 从两种互补投影导出调制信号，
经 **Self-Conditioned AdaLN-Zero** 发出逐像素缩放因子，**声称**用 **1% 数据**打败完整微调。
对鱼眼，把你的 `rectifier`/`tangent`/`raw_roi` view source 当作"互补投影"——探针机制已经就位。

**I11. 给 vendored RoPE 做解析 FishRoPE。** 用 `θ = r⁻¹_KB(r)`、`φ = atan2(...)` 替换像素网格 RoPE，
`d/2` + `d/2` 拆分；并考虑 SpheRoPE 的拆分——低频几何重参数化、高频谐波量化——在冻结 transformer 上完全免训练。
被 RayTun3R"仅 RoPE 修正表现不佳"的发现所抑制；当作 I7 的一个组件，而非独立解法。

**I12. 已知标定时丢掉相机 head。** X-Lens **刻意不预测位姿或内参**，
输出因子化的 `(归一化深度, 置信度, 全局尺度)`；MapAnything 的因子化 `(射线, 沿射线深度, 位姿, 尺度)` 是同一想法。
如果 I2 显示 FoV 估计已污染特征，有原则的修复就是彻底不再向 VGGT 索要相机，
改用固定 KB4 射线的因子化形式。架构侵入性强，但对标定过的 Aria 装置这是正确的终态。

**I13. 现有移植的 fuse-back 升级。** 360MonoDepth 的**可变形多尺度对齐 + 梯度域融合**
与 PatchFusion 的重叠区域 **Consistency-Aware Training/Inference** 都强于置信度加权平均。
与 FoV 问题无关——只有当接缝而非弯曲主导误差时才做。

**I14. 测量相机估计，不要假设。** 在同样的 crop 上跑 **AnyCalib**：
它回归逐像素射线场并以闭式恢复 **Kannala–Brandt** 内参。
这在 VGGT 的 `pose_enc[7:9]` 和已知渲染 FoV 之外提供独立的第三个数字。低成本；属于 `checks/`。

**I15. 用 ER-LoRA 的有效秩准则决定在哪里适配。** 如果走 LoRA/adapter 路线，
ER-LoRA 的 Selecting–Tuning–Maintaining 策略用有效秩决定哪些层与任务相关——
比 Fisheye3R 手工设定的 `L₀ = 12` 截断更有原则，而这值得重新审视，因为 VGGT-Ω 的块结构与 VGGT 不同。

### 关于 FoV 耦合问题的横切说明

文献中存在三种不同的答案，而且它们并不互斥：

1. **修正那个数字** — 用已知 FoV 覆写 `pose_enc[7:9]`（OmniVGGT 的注入格式；I2），
   或规范化输入使估计变得正确（Metric3D；I1）。
2. **移除依赖** — 因子化输出并提供射线，不要相机 head（X-Lens、MapAnything；I12）；
   或者用从几何导出相机的 backbone（MoGe、π³；I4）。
3. **修正真正的病因** — 如果 RayTun3R 是对的，FoV 误差是**位置编码中针孔先验**的下游产物，
   修正 PE 会同时修好两者（I3、I7）。

I3 与 I2 合起来决定你处在这三者中的哪一个，而且都很便宜。**先做这两个。**

---

## 5. VGGT-Omega 输入契约与尺度/FoV 问题

### 5.1 训练数据的 FoV 边界（已核实）

**分布**没有公布。但**其中一半数据的边界**公布了，而且很锐利。

训练数据分两半。公开的那一半（§3.5.1）列出约 3M 序列（Aria series、Bedlam、BEHAVIOR-1K、Co3Dv2、
uCo3D、DL3DV、Dynamic Replica、EDEN、EFM3D、HOT3D、Habitat、Hypersim、Mapfree、Mapillary Metropolis、
MPSD、Megadepth、Megasynth、Mid-Air、Mvssynth、ParallelDomain-4D、Replica、SAIL-VOS、ScanNet series、
TartanAirV2、TartanGround、Taskonomy、UnrealStereo4K、Virtual KITTI、Waymo、WildRGBD，
外加未指明的内部数据集）。**对这一半没有任何 FoV 或相机模型的声明。**
注意 Aria 数据*确实*在里面（Aria series、HOT3D、EFM3D）——但用的是那些发布版本的投影，
而标准发布通常是矫正过的针孔，不是原始鱼眼。

标注的那一半（§3.5.2，约 40M 网络视频 → 约 800K 保留序列）有显式过滤器，这是承重证据。
在 "Reconstruction and filtering" 下：

> "For successful reconstructions, we discard sequences that fail heuristic checks, e.g., an image
> registration ratio < 99.5%, **a field of view outside [30°, 120°]**, or a **distortion ratio >
> 0.1**. These criteria aggressively remove cases with degenerate motion or extreme zoom."

再往上游，VLM 预过滤 prompt（附录 A.3）把它做成 **Step 1 硬拒绝**：

> "5. Non-Pinhole Projections: Is the footage 360° equirectangular or heavily distorted fisheye
> without calibration?"

最后，附录 C Limitations 用作者自己的话陈述了失效模式：

> "reconstruction quality often degrades if the **field of view changes abruptly** (e.g., shifting
> from 10° to 160° in a few seconds) **or the camera is highly distorted**... These limitations are
> **primarily attributable to the distribution of our training data**."

**结论。** 120° 输入正好在过滤器上界——最后一个能通过标注的 FoV，因此是先验最薄的地方。
40° 输入在 [30°,120°] 内但接近下界，且额外落在同一句话说"被激进移除"的"极端变焦"区域。
两者名义上都不算分布外；但都在边缘。
**原始未去畸变的鱼眼则明确是分布外**——被拒绝了两次（VLM 硬拒绝；畸变比 > 0.1）——
而且独立于任何训练统计，它**无法被** `pose_enc.py` 里 R⁹ 的
"针孔 + 主点居中"参数化**表示**。这完整且充分地解释了为什么 `raw_roi` 在每个 FoV 都比 `tangent` 差，
根本不需要诉诸尺度。

### 5.2 角分辨率假说：证据是反着的

假说是：518 px crop 在 FoV 40（约 13 px/deg，高于传感器的约 11.35 px/deg）提供了插值出来的、
无信息的细节，从而损害稠密预测。两条独立的一手证据说这个效应的**符号是反的**。

**(a) 零新增信息的上采样可测量地*帮助*稠密深度。** BoostingMonocularDepth
（[arXiv:2105.14021](https://arxiv.org/abs/2105.14021)）做了正是这个对照：
"我们用 192 × 192 的原始输入图像并简单上采样……这样**输入中的高频信息量保持不变，
但我们仍然看到高分辨率结果中细节的增加**，表明网络容量存在上限。"
瓶颈是*每次前向能吐出多少细节*，而不是输入含有多少细节。

**(b) 真正的机制是上下文密度，而且是非单调的。**
"当这些线索在图像中相距超过感受野时，网络无法在信息不足的像素周围生成连贯的深度估计"，
反方向则是："**低于感受野尺寸的分辨率不会改善结构，实际上会降低性能，因为网络容量没有被充分利用。**"
上下文密度太低会用不满模型；太高会压垮模型。
他们用边缘图作为上下文线索的代理，定义 **R₀** 为"每个像素都能在一次前向中获得上下文信息的最大分辨率"。
质量曲线在 R₀-R₂₀ 附近达到峰值，之后退化。**这与我们的 FoV 扫描形状相同**。

他们的 Table 1 还带一个我们应该重视的警告：
在原图只有 640×480 的 **Ibims-1** 上，边界指标改善而全局指标*变差*
（D³R 0.3698→**0.3269** 更好；ORD 0.4002→0.4504、RMSE 0.1596→0.1687 都更差）。
**把分辨率/细节推高可以在改善边界指标的同时损害全局几何。**
`align%` 是一个边界指标。我们不应仅凭 `align%` 就断定某个配置在几何上更好。

**(c) 经典的训练/测试尺度非单调结果。** "Fixing the train-test resolution discrepancy"
（[arXiv:1906.06423](https://arxiv.org/abs/1906.06423)）显示 224 训练的 ResNet-50 在测试分辨率 **288** 达到峰值：

| K_test | 64 | 128 | 224 | 256 | **288** | 320 | 384 | 448 |
|---|---|---|---|---|---|---|---|---|
| top-1 | 29.4 | 65.4 | 77.0 | 78.0 | **78.4** | 78.3 | 77.7 | 76.6 |

机制是**表观物体尺寸**。对固定 1024 token 预算的 VGGT-Ω 而言，表观尺度旋钮**就是** FoV。
峰值出现在"匹配传感器的那个 FoV"之外，是预期结果，不是异常。

**(d) 我们自己的数字不符合角分辨率的说法。** 源是 1408 px 覆盖约 124° ≈ 11.35 px/deg。
512 px 输入在 **FoV ≈ 45°** 时匹配。如果匹配原生传感器细节驱动 `align%`，峰值应在 45-60°。
但峰值在 **100°**，那里 crop 是 5.12 px/deg——比传感器细节**降采样 2.2×**——
而 40°（唯一上采样的那行）是表中最差的 tangent 分数。
观察到的排序与角分辨率解释不一致，与训练先验/上下文密度解释一致。

**一个必须控制的混杂因素。** `align%` 由输入 Sobel 边缘计数归一化，
而分子分母都随 FoV 变化。在相信曲线形状之前，
重跑扫描并把边缘阈值设成让**边缘数量**（而非百分位）在各 FoV 大致恒定。

### 5.3 基于 patch 的推理：领域实际做法，以及"少而大 vs 多而小"的数字

**共同规则：绝不改变 backbone 看到的像素尺寸。** PatchFusion §3.1：
"我们使用**等于或接近基础深度模型原生分辨率的固定 patch 尺寸**。"
Depth Pro §3.1：画布固定 1536×1536，"输入图像在每个尺度被切成 **384 × 384** 的 patch"。
Boosting 把 tile 尺寸固定为网络感受野。
**三者都改变哪一部分世界落入窗口；没有一个改变窗口的像素尺寸。**
我们的扫描做的恰恰相反——把像素固定在 512 而改变世界内容，即改变角分辨率，
而这正是这些系统刻意冻结的那个旋钮。

| 系统 | patch 尺寸 | 如何选择 | 重叠 | 融合 |
|---|---|---|---|---|
| Depth Pro | 384²（= ViT 原生） | 固定；画布 1536² 降采样到 3 个尺度 → 25 + 9 + 1 = **35 个 patch** | **25%**（仅最细的两个尺度，"以避免接缝"） | 用目标区域的 **Voronoi 划分**合并到特征图，再上采样并由 DPT decoder 融合；另一个作用在整图 384² 的图像编码器"把 patch 预测锚定在全局上下文中" |
| PatchFusion | 固定为基础模型原生分辨率 | P=16 非重叠网格；+33 平移 → P=49；+N 随机 → R=N | 平移/随机放置 | 端到端：粗糙全局分支 + 精细 patch 分支，带**一致性感知训练/推理** |
| Boosting | = 感受野尺寸，**逐 tile 生长** | 在基础分辨率 tile，**1/3 重叠**；丢弃边缘密度*低于*整图的 tile；**生长**边缘密度*高于*整图的 tile 直到匹配 | 1/3 | 训练好的 Pix2Pix/10 层 U-Net 合并网络 |

Boosting 的选择规则是最值得偷的：**选择让边缘密度与整图相等的 crop 尺寸**。
它是边缘密度准则，而我们的对齐指标是基于边缘的，所以两者直接兼容。

**少而大 vs 多而小的数字。** PatchFusion 的 Table 3（补充材料）是我找到的唯一干净的 patch 数量扫描：

| 配置 | RMS ↓ | SEE ↓ |
|---|---|---|
| ZoeDepth COARSE（无 tiling） | 1.0777 | 0.8326 |
| + PatchFusion P=16 | 1.0743 | 0.8284 |
| + PatchFusion P=40 | 1.0678 | 0.8219 |
| + PatchFusion R=128 | 1.0620 | 0.8195 |
| + PatchFusion R=256 | 1.0580 | 0.8194 |
| + PatchFusion R=1024 | 1.0536 | 0.8178 |

对 patch 数量单调，且**收益急剧递减**：16 → 1024 是 64 倍算力换 **1.9%** 相对 RMS。
大部分价值在最初几十个 patch；没有悬崖，也没有反转。

他们的同域 Table 1 补了一个对我们重要的警告：
P=16 → P=49 → R=128 单调改善每个*全局*指标，而**边界指标却往反方向走**
（SEE 0.8382→0.8462→**0.8488**）。
所以**更多、更重叠的 patch 可靠地帮助全局几何，但不可靠地帮助边界指标**——两者会脱钩，且双向都会。
既然 `align%` 是边界指标，"加更多 crop"并不保证能推动它。

Depth Pro 的 Table 9 是互补结果，在边界上指向另一边：
**35 个原生 384 patch 比一个插值到 1536 的 ViT 高 +9.9% F1 和 +23% DIS R**。
两个结果的区别值得记住：Depth Pro 的增益来自**在原生分辨率切 patch 而不是拉伸 backbone**；
PatchFusion 的 SEE 抖动来自**在已经正确的尺度上不断增加冗余 patch**。
前者是大而可靠的效应；后者小且符号不稳定。

**把这些读回扫描。** 文献一致的建议会是：不要再在固定像素下扫 FoV。
改为把窗口固定在 VGGT-Ω 的原生预算（1024 token、262,144 px 面积、长宽比在训练带内），
然后改变**有多少这样的窗口铺满鱼眼锥体、以及重叠多少**——
即沿 raw_roi/tangent/rectifier 轴与 tile 数量轴移动，而不是角分辨率轴。

---

## 6. 2026-07-31 新增调查（本节为中文版新增）

### 6.1 VGGT 的深度错位是已知问题吗？——是，有三个层级的证据

**(1) 作者自己说了。** 见 §5.1 附录 C 原文。

**(2) 有人已经量过你正在量的那条曲线。**
[UAVFF3D: A Geometry-Aware Benchmark for Feed-Forward UAV 3D Reconstruction](https://arxiv.org/html/2605.17942)
是一个 benchmark，在 **HFOV = 25/35/45/55/65/75/85/95°** 上系统扫描，
覆盖 VGGT、π³、MapAnything、Pi3X 和 DA3。已核实的发现：

- 最优在 **65–75°**；原文："Ray Error and Pose ATE of pretrained models are usually lower around
  65°–75°, but increase markedly under narrower or wider HFOVs."
- 曲线是**峰形/U 形，不是单调的**。最差在极端值（25° 和 95°）。
- 机制："RGB-only feed-forward models are strongly influenced by implicit camera-geometry priors
  in the training distribution."
- **对本项目论点最关键的一条**：*微调过的*模型在极端 HFOV 仍然误差很高——
  "recovering camera rays, focal-length-related projection geometry, and the camera–scene scale
  relationship from RGB images alone remains difficult."
- 他们**没有**报告预测 FoV 与真实 FoV 的误差。**这个空白是你的。**

**(3) `pose_enc` 与 depth head 的分歧在官方仓库有记录且无人回答。**
[facebookresearch/vggt#180](https://github.com/facebookresearch/vggt/issues/180)：
在纵向（portrait）图像上，用 `pose_encoding_to_extri_intri()` 的内参把 depth head 反投影得到的点云
**对不上** point head 的点云；转置成横向后两者都正常。Issue 至今开放，无 maintainer 回复。

**#180 的机制拆解（本次新增分析）：**

- **预处理确实会切内容。** `vggt_visfeat/utils/load_fn.py` 默认的 `crop` 模式：
  `new_width = 518` 恒定，`new_height` 按比例，若 `> 518` 则中心裁剪。
  横向图不裁剪；纵向图被裁——例如 512×1024 → 518×1036 → 裁成 518×518，
  **上下各丢 259 行，垂直视场砍掉一半**；正方形（H = W）→ 518×518，不裁剪。
- **但裁剪解释不了"两个 head 互相矛盾"**，因为两个 head 看的是同一张预处理后的图。
  真正的原因是：**point head 绕过 `pose_enc`，depth head 依赖它**。
  只要 camera head 对该长宽比给出错误的 `fov_h / fov_w`，路径 A 就歪，路径 B 不歪。
- **推论（可直接用）：两个 head 的分歧量本身就是一个不需要 GT 的 `pose_enc` 误差探针。**
  本仓库 vendored 的模型三个 head 齐全（`vggt_visfeat/models/vggt.py:28-30`）。
- **一个反讽：** VGGT 官方 README 明确推荐走 `pose_enc` 那条路——
  原话是深度 + 相机反投影 "usually leads to more accurate 3D points than point map branch"。
  而本仓库 `main_adt.py` 融合的是 `‖world_points‖`，走的是 point head。
- 社区在 #180 里猜测"训练 dataloader 强制 landscape"并贴了 `training/data/datasets/co3d.py`。
  **我读了那个文件，没有找到任何强制横向、转置或断言 W > H 的代码**；
  若存在应在父类 `BaseDataset.process_one_image()`。**这条目前只是未经证实的猜测。**

### 6.2 针对该问题的工作

| 工作 | 途径 | 成本 | 状态 |
|---|---|---|---|
| RayTun3R | 径向/角向 PE 残差，**冻结** VGGT/DA3/π³ | **10,752 参数**，2–3 h/场景 | 无代码；需已知标定；**逐序列** |
| Fisheye3R（ECCV 2026） | 学习 token，冻结 backbone | 约 344 k 参数，40 k iters | **无公开代码**（已核实占位） |
| OmniVGGT | GeoAdapter，零初始化 | 26.8 M 参数，32×A100 × 10 天 | 微调——违反本项目约束 |
| [CAM3R](https://arxiv.org/html/2603.22631) | Ray Module + Cross-view Module，球谐 | 4×H200，300–500 epochs，**不冻结** | **在 ADT 鱼眼上报告 99.0 / 95.0 RRA@15 / RTA@15** |

**CAM3R 是本次新增中最重要的**：它在**你的数据集上**有结果。
它也量化了 baseline 的崩溃：DUSt3R 在 360Loc 上 RRA@15 / RTA@15 = **0.0% / 0.0%**（声称，按报告）；
2D3DS 上 10.6% / 6.0%。根因归于"models are predominantly trained on perspective datasets,
which implicitly constrains them to a standard pinhole camera geometry"。
四篇工作共同的挑战：所有解决这个问题的人，要么需要已知标定，要么需要逐场景适配，要么需要真实训练算力。
**没有人有一个单一的、全局的、冻结 backbone 的鱼眼 adapter。**

**一个必须记录的纠正：** 搜索中高排名出现的 **Geo-ID**（arXiv:2603.13859）
是关于 intrinsic *image* decomposition（albedo / roughness / metallicity），
**不是相机内参**。与本课题无关。在引用前核对了一手来源。

### 6.3 为什么 ERP 论文得到相反的结论——原因在本仓库的代码里

**上游 VGGT-360 用 110° 的视图。本移植默认 60°。**

```
main_erp_upstream.py:91    parser.add_argument('--FOV', type=int, default=110)
utils/gen_views.py:166     neighbor_fov=90.0
utils/gen_views.py:109     neighbor_fov=85.0
utils/fisheye_views.py:58  DEFAULT_FOV_DEG = 60.0        ← 本移植
main_adt.py:511            --fov default=60.0            ← 本移植
```

而且 `ERP2Persp`（`utils/ERP_utils.py:13-14`）构造的是**正方形**切平面——
两个轴都用 `linspace(-tan(FOV/2), tan(FOV/2))`——
所以上游的视图边到边 110°，**角到角约 127°**。
布局：8 个 base view（6 个赤道方向按 60° yaw 间隔 + 2 个极点）+ 最多 4 个不确定性增强视图 = ≤12 个，
全部在**一次调用**中通过模型（`model(images=persp_imgs_tensor, ...)`）——
这与本仓库"必须联合前向"的发现完全一致。

在 110° FoV 配 60° yaw 间隔下，上游视图**大量重叠**。
他们不是在紧密铺砌球面，而是用少数几个非常宽、高度冗余的视图覆盖它。
本移植的 60° 来自另一个理由——docstring 说它被选来使 `tilt + fov/2 ≈ 62.3°` *铺满 Aria 锥体*，
是几何论证，不是继承的超参数。

**因此："视图越窄越差"这个发现并不与 VGGT-360 矛盾，它与本移植自己引入的一个参数矛盾。
而表现最好的配置（单个约 110–120° 视图）本质上就是上游的设定。**

**尚未解决的张力：** UAVFF3D 把最优放在 65–75° 并说 >75° 会退化，
而 VGGT-360 成功使用 110°。领域不同（航拍 vs 室内全景）、指标不同、
且 UAVFF3D 从未测试超过 95°。这是一个真实的开放问题。

### 6.4 UAVFF3D 的域可迁移性评估

**不可迁移：**
- **最优值本身（65–75°）。** 航拍场景是远距离、近平面、弱视差、纹理稀疏，常含天空。
  Aria 是 0.5–5 m 的室内杂乱环境，深度范围大、视差强。
  最匹配某个*场景*上下文密度的 FoV 是场景属性。
- **指标。** Ray Error / Pose ATE / Chamfer-L1 是以位姿为中心的多视图指标；
  你的是单帧深度对齐。模型可以位姿稳定而深度弯曲。
- **输入。** 他们的是在选定 HFOV 下渲染的干净针孔图；
  你的是真实 KB4 传感器的 gnomonic 重采样——带插值模糊、径向 MTF 衰减和暗角，这些渲染针孔图都没有。

**可迁移（作为假设而非结论）：**
- **机制。** "训练分布中的隐式相机几何先验"是*模型*属性而非场景属性——
  且 VGGT-Ω 附录 C 用作者自己的话独立说了同样的事。两个独立来源收敛于同一机制，值得据此行动。
- **对本项目论点最重要的一条：** *微调过的*模型在极端 HFOV 仍然失败。
  如果这在智能眼镜鱼眼域也复现，它就是"不要微调、去修接口"这一立场最强的论据。
- **实验协议。** 跨多个 backbone 的受控、内容匹配的 FoV 扫描，正是需要的实验形状。

---

## 7. 未经证实 / 无法确认

以下陈述我**无法**追溯到一手来源。当作开放问题，不是事实。

- **RayTun3R 代码。** 论文说"our code will be made publicly available"但没给 URL，我也没找到仓库。
- **FishRoPE 代码。** 论文正文中没有仓库 URL；Qualcomm 归属来自论文 HTML，作者全名未确认。
- **UniK3D 推理时用 ground-truth 相机。** 我无法确认球谐相机模块能否被已知射线场*提示*。这影响 I5。
- **Depth Anything 3 的内参输入。** 已核实它在"有或没有已知相机**位姿**"下都能工作；
  我**没有**确认它接受已知**内参**。
- **Depth Pro 的焦距 head 在鱼眼衍生 crop 上的表现。** 其焦距估计在普通图像上**声称** SOTA；
  关于从鱼眼帧切出的窄 crop 没有任何已核实的信息——而那正是 I1/I14 关心的情况。
- **RayTun3R 或 Fisheye3R 的数字能否复现。** 两者都按发表报告；我没有复现任何东西，
  且两篇论文**互不比较**（已核实）。
- **VGGT-360 的 CVPR 归属 vs arXiv。** arXiv 摘要页没有 venue 字段；CVPR 2026 的判定来自搜索结果标题。
  Fisheye3R 的 "ECCV 2026" 与 OmniVGGT 的 "CVPR 2026 Highlight" 同理。
- **VGGT-Ω 实际的训练 FoV *分布*。** 确实没有文档。只有硬边界（`[30°, 120°]` + 畸变比 ≤ 0.1）
  作用在约 800K 标注网络视频序列上；约 3M 公开数据集序列**没有**任何 FoV 或相机模型约束。
  把"峰值接近 100° 反映训练先验"当作合理推断，不是有文档的事实。
- **`[30°, 120°]` 的约定**——水平、垂直还是对角 FoV——论文没有说明。
  这很重要：4:3 画幅上 120° 对角只有约 98° 水平。
- **训练长宽比带 `[0.33, 1.33]` 的约定**同样未说明，且它与 loader 的 `[0.5, 2.0]` crop 带**不一致**
  （`load_fn.py` 第 68 行）——loader 接受论文没有声称训练过的形状。
  对当前正方形 crop 扫描无关（长宽比 1.0 在两个带内），但一旦喂非正方形 crop 就是活的隐患。
- **VGGT-Ω 论文中没有分辨率消融，也没有 OOD 相机消融。**
  §4.3 只消融了模型规模、数据规模、register attention、多任务损失、自监督和标注质量。
  所以"VGGT-Ω 偏离 512 会退化多少"或"在大 FoV 相机上退化多少"没有第一方数字。
- **VGGT #180 中"训练 dataloader 强制 landscape"的猜测**（见 §6.1）。
- **`pose_enc` 布局**是这里唯一从*本地源码*而非论文核实的主张：
  `vggt_omega/utils/pose_enc.py` 第 16、24-26、32-43 行给出
  `[translation(3), quaternion(4), fov_h, fov_w]`，`fy = (H/2)/tan(fov_h/2)`，`fx = (W/2)/tan(fov_w/2)`。
  官方 VGGT README **没有**记录这个布局（已核实——它只暴露 `pose_encoding_to_extri_intri`）。
