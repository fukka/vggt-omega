# RayTun3R 训练回路 vs VGGT 官方训练代码：差异与问题排查

日期：2026-08-17。对照的上游是 `facebookresearch/vggt@main`（`training/` + `vggt/`，
共 69 个文件，已下载到本次会话的 scratchpad）。我们这边是 `raytun3r/`（train.py /
losses.py / adapter.py / backbones.py）以及它冻结调用的骨干
`VGGT-360-fisheye/vggt_visfeat/`（上游 `vggt/` 的一个 fork）。

## 0. 先说口径：这两套东西不是同一类训练

| | VGGT `training/` | RayTun3R `raytun3r/` |
|---|---|---|
| 目标 | 有监督多数据集微调整个模型 | 测试时逐场景适配，只学 PE 残差 |
| 可训参数 | 全模型（默认冻 `*aggregator*`，只训 head） | 10,772 个（20·C + 8·C + 20） |
| 监督信号 | GT 外参/内参/深度/点图 | 自监督：UFM 对应 + MAGSAC++ 位姿伪标签 |
| 损失 | conf 加权 L2 + 多尺度梯度 + 相机 L1（多阶段 γ=0.6 衰减） | Eq.8 重投影 + Eq.9 位姿 + Eq.10 平滑 + Eq.11 L2 + Eq.12 TV |
| 优化器 | AdamW 5e-5, wd 0.05, warmup 5% + cosine, 20 epoch | Adam 1e-3, 无调度, 300 iter |
| batch | `max_img_per_gpu=48`, accum_steps 分块反传, DDP | 每步 1 个 3 帧窗口 |
| AMP | bf16 autocast + GradScaler，head 段 `autocast(enabled=False)` | bf16 autocast（CUDA 上），head 段同样 fp32 |
| 梯度裁剪 | 按模块分组 `max_norm=1.0` | 全体一起 `clip_grad_norm_(..., 1.0)` |
| 数值守卫 | `check_and_fix_inf_nan` 遍布每个损失分量 | 只有 `reprojection_loss` 有 `nan_to_num` |
| 尺度归一 | `normalize_camera_extrinsics_and_points_batch`：以第一台相机为原点、点云平均距离归一 | 不需要（reproj 里 depth 与 t 同源，尺度自动抵消） |

**关键结构性反差**：上游默认配方 `frozen_module_names: ["*aggregator*"]` 把
aggregator 整个冻住，因此聚合器输出根本不 `requires_grad`，反传不需要保存它的
激活。RayTun3R 恰好相反——可训参数插在**最前面一层**（patch 的绝对 PE 与 RoPE
角），所以梯度必须穿过全部 48 个 block 再到两个 DPT head。下面 §3 的显存问题都由
这一点派生。

## 1. 骨干代码差异：`vggt_visfeat` vs 上游 `vggt/`

| 文件 | 变更行数 |
|---|---|
| `models/aggregator.py` | 153 |
| `utils/geometry.py` | 128 |
| `models/vggt.py` | 70 |
| `layers/attention.py` | 61 |
| `heads/dpt_head.py` | 35 |
| `layers/vision_transformer.py` | 11 |
| `heads/camera_head.py` | 10（只有 import 路径与注释） |
| `layers/rope.py` / `heads/head_act.py` / `utils/pose_enc.py` / `utils/load_fn.py` | 0（逐字节相同） |

逐字节相同的那几个很重要：**RoPE、head 激活、pose 编解码、图像加载归一化都没被动
过**，所以 raytun3r 的 `data.py:208` 用 `/255.0` 给 [0,1]、由 aggregator 内部做
ImageNet mean/std（`aggregator.py:283`）这条链路和上游完全一致，没有双重归一化。

实质差异四条：

1. **上游新增了 `cached_layer_indices=(4, 11, 17, 23)`，我们的 fork 没有。** 上游只把
   4 层的 `concat_inter` 放进 `output_list`，其余填 `None`；我们的 fork 把全部 24 层
   都 `torch.cat` 后存下来（`aggregator.py:342-344`）。
2. **`VGGT.forward` 恒定跑 `point_head`。** 上游用 `enable_point` 开关（默认 False），
   我们的 fork 把三个 head 全部硬编码建好并每次都算（`models/vggt.py:78-89`）。
   RayTun3R 只用 `depth` 和 `pose_enc`，`world_points` 纯属浪费——而且因为它在
   autograd 图里，它的激活要一直留到 backward 之后才释放。
3. **`pose_enc_list` 被丢弃**，只留最后一次迭代。上游训练损失
   （`compute_camera_loss`）要用完整的 list 做 γ=0.6 多阶段监督。对 RayTun3R 影响不
   大（`camera_head.py:116` 本来就在迭代之间 `detach()`，只监督最后一次在梯度上是
   自洽的），但这个 fork 无法直接喂给上游的 loss。
4. **`attention.py` 增加了 `save_attn` / `att_mask` 分支**。`save_attn=False` 且
   `att_mask=None` 时走的是和上游一样的 SDPA 路径，等价。RayTun3R 显式传
   `save_attn=False`（`backbones.py:498`），没问题。同理 `dpt_head` 的
   `is_feats` 默认 True，但 `vggt.py` 调用时显式传了 `is_feats=False`，也没问题。

## 2. 训练回路的问题清单

### P0-1　`pe_table` 在训练中恒为 `None`，Eq.12 的 TV 项算的是残差而不是 `P'`

证据链：

- `backbones.py:145` `self._pe_table = None`；`backbones.py:226` `remove()` 里再置 None，
  而 `install()` 第一件事就是调 `remove()`。
- 唯一的写入点在 abs-PE hook 里（`backbones.py:450-451`、`722-723`、`891-892`），
  **只有 forward 才会触发**。
- `train.py:58` 的 `pe_table = backbone.pe_table()` 在循环外、第一次 forward 之前取一次，
  之后再不更新。
- `train.py:main` 里 `install()` 之后到 `fit_adapter()` 之间没有任何 forward。

所以对 `vggt` / `da3` / `pi3` 这三个有绝对 PE 的骨干，`tv_penalty` 一路走
`pe_table is None` 的回退分支（`losses.py:214`），退化成"只惩罚残差自身的 TV"。
`raytun3r/README.md` 第 6 条写的是"`P_A` 通过交叉项进入，因此在第一次 forward 时
从 PE hook 捕获"——实现和文档在这里对不上。

影响：`w_TV = 20` 是 Eq.13 里最大的权重。`TV(P_A + r)` 与 `TV(r)` 差一个交叉项
`2⟨∇P_A, ∇r⟩`；前者允许残差去抵消预训练表自身的粗糙度，后者只会把残差往零压。也就是
说训练里权重最大的那一项，方向和论文写的不一样。

为什么单测没抓到：`tests/test_raytun3r.py:361 test_tv_uses_the_absolute_table_when_available`
测的是 `tv_penalty` 函数本身（直接传 table），没有覆盖 `train.py` 的接线；
`smoke_test.py:236` 恰好是在一次 forward 之后才调 `bb.pe_table()`，所以那边是对的。

修复（最小改动）：把取值挪进循环，或在建 optimizer 之后先跑一次 warm-up forward。

```python
# raytun3r/train.py, fit_adapter 内
with torch.no_grad():
    backbone.forward(windows[0].images[None])   # 触发 PE hook，填 _pe_table
pe_table = backbone.pe_table()
```

### P0-2　VGGT 路径下 Eq.6 的 RoPE 修正只作用到 global-attention 的最后一帧

`backbones.py:318-334` 的 `hook_tokens` 用
`n_prefix = out.shape[-2] - gh*gw` 推断前缀 token 数。这个启发式只对**单帧**布局成立。

而 `aggregator.py:133` 建的是**同一个** `RotaryPositionEmbedding2D` 实例，同时传给
`frame_blocks` 和 `global_blocks`（`aggregator.py:147` 与 `164`），所以 hook 在两种
block 里都会触发：

- frame block：token 形状 `(B·S, heads, P, d)`，`P = 5 + gh·gw = 5 + 1296 = 1301`，
  `n_prefix = 5` ✔，每一帧都被修正。
- global block：token 形状 `(B, heads, S·P, d)`，`N = 3·1301 = 3903`，
  `n_prefix = 3903 − 1296 = 2607`。第 2 帧（末帧）的 patch token 恰好从
  `2·1301 + 5 = 2607` 开始——**偏移没错位**（对任意 S 都成立：
  `S·k + (S−1)·G` 同时是 `n_prefix` 和末帧 patch 起点），
  **但第 0、1 帧的 patch token 在全部 24 个 global block 里完全没有被修正**。

也就是 S=3 时，global 段有 2/3 的 patch token 拿不到 Eq.6。第 0 帧还是定义坐标系、
主导位姿输出的参考帧。

严重度定级说明：RoPE 表只有 20 个参数（占 10,772 的 0.2%），论文 Tab.7(b) 也说
纯 RoPE 适配很弱（19.52° vs 0.48°），所以对最终指标的影响多半有限——但这是确凿的
实现错误，并且**它让 RoPE 相关的消融结果不可用**。

修复方向：hook 里按帧 reshape，而不是靠 `N − gh·gw` 猜前缀：

```python
G = gh * gw
n = out.shape[-2]
per_frame = G + n_special          # n_special 从 aggregator.patch_start_idx 拿
if n % per_frame == 0:             # global：S 帧拼接
    S = n // per_frame
    v = out.reshape(*out.shape[:-2], S, per_frame, out.shape[-1])
    v[..., n_special:, :] = adapter.rope_tokens(v[..., n_special:, :], n_blocks=2)
    return v.reshape_as(out)
```

### P1-1　骨干停在 `eval()`，梯度检查点被整个关掉

`backbones.py:131-133` 在 `__init__` 里 `model.eval()`，全仓库没有任何
`.train()` 调用。而 aggregator 的检查点是 `if self.training:` 门控的
（`aggregator.py:367` 与 `401`）。于是 RayTun3R 反传时，48 个 block 的激活**全量保存**。

这在上游是不会发生的：上游默认冻 aggregator，压根不反传穿过它；真要训 aggregator 时
它会在 `train()` 模式下自动开检查点（上游注释原话就是提醒要 `model.train()`）。

好消息是这里可以直接开：VGGT 的 aggregator 与内嵌的 DINOv2 ViT
`drop_path_rate = 0.0`（`vision_transformer.py:56`，Aggregator 未覆写）、
`drop = attn_drop = 0.0`（`block.py:36-37, 39`）、**没有任何 BatchNorm**。所以
`model.train()` 对数值是恒等的，唯一的效果就是打开检查点。

粗略量级（S=3、504×504、bf16、1301 token/帧）：48 个 block 的激活约 5–6 GB，
`output_list` + 两个 intermediates 列表另加约 0.77 GB，再加两个全分辨率 DPT head。
开检查点后 block 部分可以降到大约十分之一。这是估算，不是实测。

建议：给 `Backbone` 加一个开关，训练时把 aggregator 置 `train()`（保持
`requires_grad=False`），评测时置回 `eval()`。

### P1-2　`batch_size > 1` 时整批共用一张计算图

`train.py:67-78`：循环里 `acc = loss if acc is None else acc + loss`，循环结束后才
`(acc/len(idx)).backward()`。这意味着 batch 里每个窗口的完整激活都要同时存活，显存
随 `--batch-size` 线性增长。上游对应的
`trainer.py:_run_steps_on_batch_chunks` 是**每个 chunk 各自 backward 累积梯度**。

修复是一行：

```python
(loss / len(idx)).backward()      # 放进 for k in idx 循环内部
```

### P1-3　稀疏匹配器会把 `L_reproj` 静默缩小两个数量级

`reprojection_loss`（`losses.py:110`）除以 `omega = valid.sum()`（Ω 的像素数）而不是
`sum(w)`——这是对的，Eq.8 就是这么写的，注释也解释得很清楚。但后果是
**`L_reproj` 的有效尺度正比于"有置信度的像素占 Ω 的比例"**。

- UFM（论文的选择）：稠密，`w ≈ 1` 覆盖大部分 Ω，比例 ≈ 1。
- `SIFTMatcher`（`matching.py:216-224`）：只在通过 ratio test 的整数关键点上写
  `weight = 1.0`，量级 10³；504×504 的 Ω 约 2×10⁵ 像素 → 比例约 **0.005–0.02**。

也就是说用 SIFT 兜底时，`L_reproj` 相对 `w_smooth=10 · L_smooth`、
`w_TV=20 · L_TV` 被压低约 50–200 倍，目标函数实际上被正则项支配，训练近似退化成
"把残差压回零"。`build_matcher` 只在回退时发一条 `RuntimeWarning`，没有量化这件事。

建议：`fit_adapter` 每次打印 `sum(w)/|Ω|`；非 UFM 时要么按该比例重标定 `w_smooth/
w_L2/w_TV`，要么在报告里显式标注这个系数。

### P2-1　MAGSAC 的阈值在鱼眼边缘不是角度阈值

`relative_pose_magsac`（`matching.py:265-290`）把 bearing 除以 z 投到归一化平面，然后
用 `thr = tan(0.5°)` 作阈值，注释写的是"an angular threshold expressed on the
normalised plane"。这个等价只在近光轴成立：角度→归一化平面的局部放大率是 `1/cos²θ`，
θ=85° 时约 **131 倍**。结果是边缘的正确匹配被系统性判成 outlier——而边缘正是鱼眼
适配最关心的区域，位姿伪标签因此偏向中心视场。

另外 `front = (bi[:,2] > 1e-3)` 直接丢掉 θ>90° 的匹配，对 ScanNet++（θ_max≈85°）无
影响，但对论文 benchmark 里 185–200° 的镜头会丢掉一整圈。

建议：改用球面残差（Sampson on bearings），或至少按每点的 `1/cos²θ` 缩放阈值。

### P2-2　缺少上游有而我们没有的几个稳态措施

- **无 LR 调度 / warmup**：上游是 5% 线性 warmup + cosine 到 1e-8；我们是恒定 1e-3。
  论文只说了 lr 1e-3，所以这不算 bug，但零初始化的表在第一步就吃满学习率。
- **无验证 / 早停 / 中途评测**：300 iter 跑完直接存盘，发散了看不出来。
- **NaN 守卫不全**：上游每个损失分量都过 `check_and_fix_inf_nan`；我们只有
  `reprojection_loss` 里有 `nan_to_num`，`pose_loss` / `smoothness_loss` /
  `tv_penalty` 没有。`fit_adapter` 也不检查 `loss.item()` 是否有限（上游
  `trainer.py:682` 一旦非有限就停）。
- **`torch.randint` 有放回采样**：300 步、30 个窗口，有放回采样下有窗口整轮没被抽到
  的概率不低（≈ (29/30)^300 ≈ 4×10⁻⁵ 单个窗口，但 batch=1 时覆盖分布仍然不均）。
  上游是 epoch 式遍历。

### P2-3　两处失效的防御代码

- `losses.py:100` 的 `ok = torch.isfinite(err) & (Xj[...,2] > -1e6)`：注释说"落在成像
  锥背后的点没有意义，丢掉"，但 `> -1e6` 实际上什么都不筛（只挡 `-inf`）。真要按
  成像锥筛，判据应该是 `camera` 的 `theta_max`，不是 z 的符号。
- `backbones.py:326` 的注释说 VGGT 的前缀 token "carry position −1"；实际是
  `pos = pos + 1` 之后前缀取 0（`aggregator.py:315-317`）。不影响行为，但会误导下一
  个读这段的人。

## 3. 检查过、确认没问题的部分

- 图像归一化链路：`/255 → [0,1] → aggregator 内部 ImageNet mean/std`，与上游一致。
- 冻结：`smoke_test` 与 `test_gradient_reaches_adapter_and_backbone_stays_frozen`
  都断言了"梯度到达每张适配器表 且 骨干无梯度"。
- 全链路无 `torch.no_grad()` 阻断：`vggt_visfeat` 里唯一的 `@torch.no_grad()` 是
  `mask_persp_to_tokens`（只在传 `persp_masks` 时走）；`dpt_head.py:261` 的
  `.detach()` 只作用在辅助的 `out_feat` 上，`preds` 没被切断。
- bf16 而非 fp16：与上游 config 一致，且 fp16 是 VGGT 已知的失效模式。
- RoPE 轴向布局：`RadialRoPE.rotate_tokens(n_blocks=2)` 的
  `cat(-t2, t1)` 与 `RotaryPositionEmbedding2D._rotate_features` 在
  `chunk(2)` 后的每一半上逐字对应，组合旋转是精确的。
- 尺度：`reprojection_loss` 里 depth 与 t 出自同一次预测，VGGT 的"以首帧为原点、点云
  平均距离归一"约定自动抵消，不需要上游 `_process_batch` 那套归一化。
- 相机分辨率一致性：`data.py:191-192` 与 `backbones.install` 的 `camera.resized`
  两处对齐，`working_size` 用的是 `backbone.patch_size`。

## 4. 建议的修复顺序

1. P0-1 `pe_table`（2 行，直接改变权重最大那一项的语义）
2. P0-2 global-attention 的 RoPE 帧对齐（约 10 行）
3. P1-1 训练时 `aggregator.train()` 开检查点（约 5 行，纯显存）
4. P1-2 把 `backward()` 挪进 batch 循环（1 行）
5. P1-3 打印 `sum(w)/|Ω|`，并在非 UFM 时给出量化警告
6. P2-* 数值守卫、非有限损失中止、MAGSAC 角度阈值

1–2 会改变数值结果，跑过的 RayTun3R 训练结果需要在修完之后重跑。

---

# 附：修复记录（2026-08-18）

全部问题已修复，`raytun3r/` 下六个文件、+1033/−70 行，新增 18 个测试（48 → 66）。
**这台 Mac 上没有可用的 torch，所以没有任何一条是跑出来验证的**——下面标注了每条
的验证方式。

## 修复清单

| # | 问题 | 改动 | 独立复核 |
|---|---|---|---|
| P0-1 | `pe_table` 恒为 None | `train.py`：改为每次 forward **之后**取 | ✅ 通过 |
| P0-2 | RoPE 只修正 global 段最后一帧 | `backbones.py`：从 `pos` 反推帧布局 | ✅ 发现两处缺陷，已再修 |
| P1-1 | eval() 关掉梯度检查点 | `Backbone.grad_checkpointing()` + `--no-grad-checkpointing` | ✅ 发现三处，已再修 |
| P1-2 | 整批共用一张计算图 | `backward()` 移进 batch 循环 | ✅ 通过，无缺陷 |
| P1-3 | 稀疏匹配器静默改变权重 | 量化 + 记录 + 硬下限 | ✅ 发现四处，已再修 |
| P2-1 | MAGSAC 阈值非角度 | 中位数缩放 + 角度残差重打分 | ⚠️ 仅自查 |
| P2-2 | 缺 NaN 守卫 / 非有限中止 | `losses._finite` + `fit_adapter` 抛错 | ⚠️ 仅自查 |
| P2-3 | 失效的越界过滤 | 改为按图像对角线截断，**不再丢弃** | ⚠️ 仅自查 |
| P2-4 | 有放回采样 | 改为洗牌 epoch 遍历 | ⚠️ 仅自查 |
| — | fork 恒算 point_head | `VGGTBackbone.load(drop_point_head=True)` | ⚠️ 仅自查 |
| — | SIFT 丢弃 `device`（既有崩溃） | `matching.py` 设备处理 | ⚠️ 仅自查 |

复核过程中额外发现并修掉的（不在原报告里）：

1. **`_pe_table` 存的是 `pos_embed` 的视图不是副本**——DINOv2 快路径下
   `pos` *就是* `self.pos_embed`，改为 `.detach().clone()`。
2. **`--matcher sift --device cuda` 直接崩**：`SIFTMatcher` 接了 `device` 却丢掉，
   `Matcher._mask` 拿 CPU 权重乘 CUDA mask，死在 `build_windows`，训练根本没开始。
3. **coverage 只进 stdout 不进日志**：现在 `match_coverage` 和 `matcher` 都写进
   `train_log.json`，并对 <5% 直接抛错（`--allow-sparse-matcher` 才放行）。
4. **`has_rope=True` 但没有任何 RoPE 模块匹配 → 静默无操作**：`Pi3Backbone` 的
   RoPE 类名可能是 `RoPE2D`，两个名字都不匹配，20 个参数进了优化器却拿不到梯度。
   现在 install 时直接报错并列出模型里真实的 RoPE 类名。
5. **`_rope_frame_layout` 会被退化的 `pos` 骗到**：全零 `pos` 在每个除数上都"周期"，
   于是随便哪个周期都会被确认。现在要求一帧内的 patch 坐标**互不相同**，并按
   `(n, g)` 缓存（这个 hook 每步要触发约 200 次）。
6. **自己写的两个测试是空的**：`sparse_w[valid].reshape(-1)[:8] = 1.0` 写进的是
   布尔索引返回的**副本**；RoPE 拒绝测试传的全零 `pos` 反而被成功解析。两个都已重写。
7. `grad_checkpointing(True)` 原本在 `try:` 外面；检查点测试只断言"跑过"而没断言
   **重算**发生（现在断言 forward + recompute 恰好两次）；`finally` 路径没有覆盖
   （现在有一个故意抛错的测试）。

## 第二轮修复（复核之后自查出来的）

8. **SIFT 像素碰撞：任意选中的对应，还给它满置信度。** 两个关键点 `int(round())`
   到同一像素时，`target` 是后写者胜出而 `weight` 保持 1.0 —— 对应关系由
   BFMatcher 的迭代顺序决定，然后被当成完全可信。SIFT 本身就会在同一位置为每个
   主方向各发一个关键点，所以这不是罕见情况。改为保留描述子距离更近的那个。
9. **install 中途失败会留下半装配的模型。** `_install_hooks` 依次装四组 hook，而
   `_hook_rope` 现在会故意抛错；抛错时前面的 hook 已经注册、方法已经替换，但
   `install()` 没走完，没有任何后续 `remove()` 认领这些状态。改为 try/except →
   `remove()` → 重新抛出。
10. **给 vggt_omega 加检查点是个陷阱**（复核提出）。它的 aggregator 根本没有
    `self.training` 检查点分支可开，而且 train 模式在那边**不是恒等的**：
    `vggt_omega.py` 会丢掉 `predictions["images"]`，`vision_transformer.py` 走另一条
    cls-norm 分支。基类返回 False 是正确行为不是遗漏，已写进 docstring 防止被"好心"
    推广。

## 第三轮：三个复核 agent 找到的问题（含我自己引入的一个回归）

11. **⚠️ 我引入的回归：`losses._finite` 让静默损坏取代了响亮的崩溃。** 我把每个
    损失项包了 `nan_to_num`，声称"backward 是掩码所以不会有 `0 * NaN`"。**这是错的**，
    已在本机 torch 2.2.2 上实测：

    ```
    p = Parameter([1., 2.]);  term = (p * [1., inf]).sum();  _finite(term).backward()
    -> total = 0.0        p.grad = tensor([0., nan])
    ```

    掩码注入的那个零，在上一层遇到**非有限的局部导数**，变成 `0 * inf = NaN`。图在
    上游就已经被污染，护住标量救不回来。更糟的是 `clip_grad_norm_` 拿到一个 NaN 梯度
    会返回 nan 并把**每个**参数的梯度都乘上它（实测确认），Adam 的动量再把它固化 ——
    于是训练会跑满 300 步、打印一路看似正常的有限数字、存下一个全 NaN 的 adapter。
    **修法：删掉 `_finite`，改为检查梯度范数**（`clip_grad_norm_` 的返回值当且仅当
    存在非有限梯度时非有限），在 `opt.step()` 之前中止。

12. **⚠️ 我引入的回归：`found_rope` 报错会打挂所有 eval baseline。** 那个守卫的
    理由是"RadialRoPE 参数进了优化器却没有前向路径"，但 `eval.py` 对
    `vanilla` / `param_free` / `center_ph` / `multi_ph` / `lora` / `caltok`
    全都是 `install(None, ...)` —— 根本没有 adapter，也就没有 RadialRoPE。
    改为只在 `adapter is not None and adapter.rope is not None` 时报错。
13. **重打分把 MAGSAC 拒掉的点又放了回来。** 角度残差是一维的，随机错误 bearing
    落在 0.5° 内的概率约 `sin(0.5°)≈0.9%`，4096 个点里约 37 个粗差被投票进
    `recoverPose`。改为 `angular & inliers`：只收窄，不替换。
14. **截断上限用错了范数。** `err` 是 `.abs().sum(-1)`（曼哈顿），两个画面内点能产生
    的最大值是 `(w-1)+(h-1)`，不是欧氏对角线 —— 对角线偏小约 30%，在测试场景里
    **裁掉了 2.87% 的加权像素**，而且集中在边缘。改为 `width + height`。
15. **install 回滚没有清 adapter**；`_match_coverage` 没有自己套 `valid`（只是
    "按惯例"与 Eq. 8 一致）；SIFT 碰撞按原始描述子距离而不是 Lowe ratio 排序；
    `Matches.sample` 的 CPU generator 配 CUDA 索引会抛错。都已修。
16. **OpenCV < 4.5 没有 `USAC_MAGSAC`，也不接受 `maxIters`** —— 这是既有问题，
    本机 cv2 4.4.0 上 `test_magsac_recovers_pose_from_exact_matches` 在我改动之前
    就是失败的。现在会降级到 RANSAC 并**大声警告**（论文指定 MAGSAC++，降级必须
    跟着数字一起被记录）。

## 测试实际跑过了（更正）

前两轮我说"这台 Mac 没有可用的 torch"，**那是错的** —— 我只查了 PATH 上的
`python3` 和 `python3.9`–`3.12`，漏了 `~/opt/anaconda3/bin/python`（3.8.3 +
torch 2.2.2 + cv2 4.4.0）。现在：

```
73 passed, 1 skipped   (skip = da3，可选依赖未安装)
smoke_test: all checks passed
```

为了跑起来我在那个 anaconda 环境里装了 `huggingface_hub==0.25.2`（`tiny_vggt`
经由 `VGGT.from_pretrained` 的 mixin 间接需要它）。

## 还需要在 GPU 机器上确认的两件事

1. **DA3 的 RoPE 布局**。`DA3Backbone` 声明 `has_rope=True`，注释说是
   `RotaryPositionEmbedding2D`。新代码走 `pos` 反推，对单帧和多帧都对；但
   `depth_anything_3` 在这台 Mac 上装不了，`test_da3_hooks_fire_on_the_real_package`
   没跑过。**这是论文的主骨干，先跑这条测试再信任 DA3 的任何结果。**
2. **Pi3 的 RoPE 类名**。`"RoPE2D"` 是按 CroCo/DUSt3R 的命名加进去的，没有验证。
   如果名字不对，install 会直接报错并把真实类名打出来——照着加一行即可。

## 关于没改的两项

- **学习率调度**：论文写的就是恒定 Adam 1e-3，上游的 warmup+cosine 是给 20 epoch
  全模型微调用的。加调度会偏离论文，没加。
- **vendored fork 缓存全部 24 层**（上游只缓存 4 层）：这要改
  `VGGT-360-fisheye/vggt_visfeat/models/aggregator.py`，是 VGGT-360-fisheye 项目
  共用的推理代码。改它风险大于收益（省下的量级远小于检查点已经省掉的），所以留作
  一条 fork 同步建议，没动。`point_head` 那一半从 RayTun3R 侧解决了，不碰共用代码。

## 重跑

P0-1、P0-2、P2-1、P2-3 都会改变数值结果。**已有的 RayTun3R 训练结果和
`adapter.pt` 需要在修复后重跑**，旧的 `train_log.json` 与新的不可比
（TV 项现在多了 `TV(P_A)` 这个大常数）。
