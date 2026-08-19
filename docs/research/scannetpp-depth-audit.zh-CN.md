# ScanNet++ 深度：与主流训练仓库的对照，以及我们这条链路的问题排查

> **英文总览见 [scannetpp-camera-reference.md](scannetpp-camera-reference.md)** —
> 相机硬件、发布了哪几套图、视场的完整推导、八个坑，以及与 Aria/ADT 的域差量化。
> 本文只管深度这条链路，是它的补充。

日期：2026-08-18。检查对象是本仓库里**唯一**碰 ScanNet++ 的代码路径 `raytun3r/`
（`data.py` / `cameras.py` / `eval.py` / `metrics.py`）。`vggt_omega/`、`finetune/`、
`slambench/` 里没有任何 ScanNet++ 入口，`grep -ril scannet` 可以复核。

---

## 0. 先把口径说清楚：我们没有在 ScanNet++ 上做深度监督训练

这一点必须放在最前面，否则下面所有对照都会被误读。

| | 主流 ScanNet++ 深度训练 | 我们 `raytun3r/` |
|---|---|---|
| 训练目标 | 用 GT 深度监督整个（或大部分）网络 | 逐场景拟合 10,772 个 PE 残差参数 |
| GT 深度进不进损失 | 进，是主监督 | **完全不进**。`train.py:177` 调 `build_windows` 时没传 `with_gt`，默认 `False`，`gt_depth` 从头到尾是 `None` |
| 监督信号 | `render_depth` | UFM 对应 + MAGSAC++ 位姿伪标签（自监督） |
| GT 深度用在哪 | 训练 + 评测 | **只有评测**：`eval.py:117-121` 的 `AbsRel` / `δ₁.₂₅` |
| 数据量 | 全数据集，scene-level split | 单场景、30 个窗口 |

所以「我们在 ScanNet++ 上的训练有没有问题」这个问题，正确的拆法是两问：

1. **我们读 ScanNet++ 的方式对不对**（几何约定、深度语义、掩码、坏帧）——这部分和监督训练仓库完全可比，也是风险集中的地方；
2. **我们的协议对不对**（拟合集/评测集怎么切、评了多少）——这部分和论文比，不和别人比。

下面 §4 是我们做对的，§5 是问题清单。

---

## 1. 对照了哪些「正经」仓库，以及一手证据

我没有只读二手描述，几个关键结论是从源码里读出来的：

| 来源 | 角色 | 我实际读了什么 |
|---|---|---|
| [`scannetpp/scannetpp`](https://github.com/scannetpp/scannetpp) | 官方工具箱 | `common/render.py`、`dslr/undistort.py`、`dslr/downscale.py` |
| [`liu115/renderpy`](https://github.com/liu115/renderpy) | 官方 `render_depth` 的**实际渲染器** | `src/opengl/opengl.h`（fragment / fisheye vertex shader、`convertIntrinsicToProjection`）、`src/render.h`、`src/opt/camera.h` |
| [`yuliangguo/depth_any_camera`](https://github.com/yuliangguo/depth_any_camera)（CVPR 2025，已在 `third_party/`） | 在 ScanNet++ 鱼眼上做**度量深度训练/测试** | `dac/dataloders/scannetpp.py`、`splits/scannetpp/*.py` |
| [`facebookresearch/map-anything`](https://github.com/facebookresearch/map-anything) | 在 ScanNet++ 上做**度量 3D / 深度训练**（Meta） | `data_processing/wai_processing/scripts/conversion/scannetppv2.py`、`configs/undistortion/scannetppv2.yaml`、`configs/rendering.yaml`、`mapanything/datasets/wai/scannetpp.py` |
| [ScanNet++ 官方文档](https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation) | 字段语义 | `is_bad` / `mask_path` / `has_mask` / `frames` vs `test_frames` |

本地还有一份真场景样本（`<staged>/3f15a9266d`，由
`raytun3r.experiments.make_local_sample` 生成：`transforms.json` + 24 帧图 +
两套掩码，**没有 `render_depth`**），下面的实测数字都来自它。样本受 ScanNet++
许可约束，**不入库**，路径按各自机器为准。

---

## 2. ScanNet++ DSLR 的事实（在真标定上实测，不是转述）

`3f15a9266d/dslr/nerfstudio/transforms.json`：

```
camera_model = OPENCV_FISHEYE      w = 1752   h = 1168
fl_x = 616.721   fl_y = 617.354    cx = 878.593   cy = 589.767
k = (0.0611, 0.00335, 0.00299, -0.00100)
frames = 896     test_frames = 10   has_mask = true
is_bad = 143 / 896  (15.9%)
```

由此解出的真实视场（KB4 前向多项式数值求逆）：

| 方向 | 归一化半径 | 入射角 θ | 全视场 |
|---|---|---|---|
| 对角（画幅角点） | 1.7153 | 84.84° | **169.7°** |
| 水平 | 1.4246 | 73.23° | 146.5° |
| 垂直 | 0.9553 | 51.95° | 103.9° |

在 504×336 工作分辨率上：

* `valid_mask` 覆盖率 = **1.0000**（整个矩形都在成像锥内，角点有真实内容，实测角点 40×40 灰度 22–119，不是黑边）
* 入射角 min 0.16° / mean **49.11°** / max 84.73°
* **36.1%** 的像素落在论文所称的 115° 锥之外；5.2% 在 75° 之外
* z→range 的换算因子 `1/cos θ`：mean **1.926**、median 1.570、p99 5.92、max **10.89**

最后一行是本文最重要的一个数：**如果不做 z-buffer→欧氏距离的换算，ScanNet++ DSLR 上的
深度平均就错 93%，边缘错 10 倍。** 任何全局尺度对齐都吸收不掉它（它是径向变化的）。

---

## 3. 逐项对照表

| 项 | 官方工具箱 | DAC (CVPR'25) | MapAnything (Meta) | 我们 `raytun3r/` | 判定 |
|---|---|---|---|---|---|
| 用哪套图 | 提供 `resized_images`（鱼眼）与 `resized_undistorted_images`（针孔） | 鱼眼 → 转 ERP | **先 undistort 成针孔**（`center_principal_point: True`）再训 | 鱼眼原图 | ✅ 方法本身要求鱼眼 |
| 深度来源 | `render_depth`，uint16 mm，0=无效 | 用官方 `render_depth` | **自己用 nvdiffrast 重渲**，float32 EXR，near 0.01 / far 1000 | 用官方 `render_depth` | ✅ |
| 深度语义 | **planar z**（见 §4.1 证据） | 显式除以单位光线 z（每场景 LUT） | 针孔下 z 即可 | 显式除以 `ray_grid()[...,2]` | ✅ 一致 |
| 换算发生的分辨率 | — | 把 LUT 用 `INTER_CUBIC` 缩到深度图尺寸 | — | **在渲染分辨率上换算，再降采样** | ✅ 更严格 |
| 像素中心约定 | renderpy: `X_n=(i−cx)/fx` | `+0.5`（`u+0.5`） | 依赖 undistort | `(i−cx)/fx`，`arange` 无 +0.5 | ✅ 与渲染器一致（DAC 差半像素） |
| `theta_max` / 有效锥 | renderpy 用 ETH3D 的 `radius_cutoff`（多项式单调性拐点） | LUT 里 `|xy|>1` 记 `isnan` | — | `_default_theta_max`：拐点 ∧ 画幅角点 | ✅ 与渲染器同源 |
| `is_bad` 坏帧 | 文档：「模糊或重阴影」 | **不过滤** | 保留为帧属性供下游过滤 | **过滤掉**（143/896） | ✅ 我们更干净 |
| 匿名掩码 `resized_anon_masks` | 文档：0=无效，被涂 (255,0,255) | **不用** | **作为一等 frame modality 传下去** | 有 `anon_mask()`，**但全流程从不调用** | ❌ 见 §5.2 |
| `test_frames` | 官方每场景 10 帧留出 | 自建 scene-level 80/10/10（seed 555） | 非测试场景 `frames + test_frames` 合并；测试场景只用 `frames`；按 `nvs_test.txt` 排除 | **完全忽略** | ❌ 见 §5.1 |
| 深度上下限 | 渲染 near 0.05 / far **20.0**，再 clip 到 65535 mm | `min_depth 0.01 / max_depth 40` | `nan_to_num` | 只有 `d > 1e-6`，`--max-depth` 默认 None | ⚠️ 见 §5.3 |
| 视图配对 | — | 单图 | **pairwise covisibility，阈值 0.25** | 连续帧 + 2px 光流阈值 | ⚠️ 见 §5.4 |
| 位姿约定 | nerfstudio/OpenGL c2w | 不用位姿 | `gl2cv()` | 翻 y/z 后求逆，已用 MAGSAC++ 独立校验到 0.17° | ✅ |

---

## 4. 我们做对了的四件事（其中三件是别人踩坑的地方）

这些不是自我表扬，是因为它们现在有了一手证据，值得钉死，免得以后被重新翻案。

### 4.1 `render_depth` 是 planar z —— 从渲染器源码确证，不再是推断

`raytun3r/data.py:139-148` 的 docstring 说 ScanNet++ 是 z-buffer，之前的依据只是 DAC 的
`DATA.md` 一句话。现在有直接证据。renderpy 的 fragment shader
（`src/opengl/opengl.h:148-157`）：

```glsl
in vec3 fragmentCoord;
layout (location = 1) out highp float depth;
void main() { color = fragmentColor; depth = fragmentCoord.z; }
```

而鱼眼 vertex shader（同文件 299-337）只改写 `cameraCoord.x/.y` 做畸变，**`z` 原封不动**，
末尾 `fragmentCoord = cameraCoord.xyz`。所以输出的就是相机系下沿光轴的 planar z，
与像素的光线方向无关。`common/render.py` 再 `*1000` 存 uint16 mm。

→ 我们 `data.py:265-278` 的 `d / cos.clamp_min(1e-3)` 是对的，而且做在**渲染分辨率**上
（先换算后重采样），这一点比 DAC 严格：DAC 是把 LUT 三次插值到深度图尺寸再除，
在 `cos` 最小的边缘会把邻居的 θ 配到本像素的 z 上。

### 4.2 像素中心约定与渲染器一致（DAC 差半像素）

renderpy 的 `convertIntrinsicToProjection`（`opengl.h:564-590`）：

```cpp
mat[2][0] = 2 * (0.5f + cx) / width - 1.0f;
```

OpenGL 视口把 NDC 映到窗口坐标、像素列 `i` 的中心在 `i+0.5`，代入解得
**`i = fx·X_n + cx`**，即 `X_n = (i − cx)/fx`。这正是 `cameras.py:58-65` 的 `pixel_grid`
（`arange`，不加 0.5）+ `unproject` 的做法。DAC 的 `create_fisheye_grid_scannetpp.py`
里写的是 `u + 0.5`，相对生成 `render_depth` 的渲染器差半个像素。

量级（本场景实测，把 cx 平移半个原始像素后看 `1/cos` 的相对变化）：
mean 0.08%、p99 0.28%、max 0.49%。所以这是个**亚 1% 的效应**，不用为它重跑任何东西，
但值得记下来：我们是对齐渲染器的那一侧。

> 顺带一个上游隐患：官方 `dslr/downscale.py` 的 `compute_resize_intrinsic` 用的是
> `cx * scale_factor`，没有 `(cx+0.5)·s − 0.5` 修正。如果 `dslr/nerfstudio/transforms.json`
> 是这么生成的，发布的 cx/cy 本身就带约 0.375 px 偏移。**没有验证，别当结论用**——
> 但如果哪天出现说不清的亚像素残差，这是第一个该查的地方。

### 4.3 `theta_max` 与渲染器同源

`cameras.py::_default_theta_max` 取「KB4 前向多项式的单调性拐点」∧「画幅角点半径」。
renderpy 的 `radius_cutoff_squared()`（`src/opt/camera.h:103`）转发给 ETH3D 的
`FisheyePolynomial4Camera`，注释直接指向 ETH3D `camera_base_impl.h:413` 的裁剪逻辑——
同一个判据。也就是说，**GT 深度存在的区域和我们 Ω 的定义是同一套几何**，
本场景下二者都覆盖整个矩形（实测 valid frac = 1.0000）。

### 4.4 过滤 `is_bad`

官方文档：`is_bad` 表示「图像模糊或有重阴影」。DAC 的 split 生成脚本
（`prepare_scannetpp_split_files_in_nyu_format.py`）直接 `glob('*.JPG')`，**一帧不滤**；
MapAnything 把 `is_bad` 原样带进 `scene_meta.json` 交给下游。我们默认丢掉
（`data.py:172-178`，本场景 143/896 = 16%，集中在 8 段连续区间，最长 132 帧）。
这是三者里最保守的做法，保留。

---

## 5. 问题清单

### P1-1　`test_frames` 被完全忽略，拟合集和评测集来自同一个 seeded shuffle

**现象。** `data.py:166` 只读 `meta["frames"]`，`meta["test_frames"]`（本场景 10 帧）
从来没被读过。同时：

* `train.py:177`：`build_windows(..., seq_len=3, n_windows=30, seed=0)`
* `eval.py:294`：`build_windows(..., seq_len=2, n_windows=100, seed=0)`

两边 `starts` 都用 `np.random.RandomState(seed).shuffle`，**同一个默认 seed=0**。

**实测（n=753 可用帧，stride 10）：**

| | 值 |
|---|---|
| 评测覆盖的不同帧 | 189 帧 = **序列的 25%** |
| 其中被拟合过的 | 39 帧 = **评测帧的 21%** |
| `AbsRel`/`δ₁.₂₅` 实际算在几帧上 | 100 帧（每个窗口只用 `pred.depth[0]`）= 序列的 **13.3%** |
| 这 100 帧里被拟合过的 | 20 帧 |

**为什么是问题。** 论文的协议原文是「在 30 个窗口上拟合，**在整段序列上评测**」。
我们评的是 100 个随机窗口，不是整段；而且这 100 个窗口和拟合窗口出自同一个 RNG 流，
有五分之一重叠。这不会让方法「作弊」到看不出问题的地步（自监督测试时适配本来就
在同一段序列上跑），但它让**每一个报出来的数字都不能直接对 Tab. 1/2/3**——
而对上论文的表正是这份代码存在的理由。

对照：MapAnything 明确按 `nvs_test.txt` 做场景级排除，并且对非测试场景把
`frames + test_frames` 合并进训练；DAC 自建 scene-level 80/10/10。两者都有明确的
「哪些帧不许被看到」，我们没有。

**建议。**
1. `ScanNetPPFisheye` 增加 `split={"train","test","all"}`，读 `meta["test_frames"]`；
2. `eval.py` 增加 `--full-sequence`，按 `--stride` 遍历全序列而不是抽 `--windows` 个；
3. 至少让 `eval.py` 的 seed 默认与 `train.py` 不同，并在结果 JSON 里记下重叠帧数。

### P1-2　匿名掩码加载了、测试了，但流水线里从来没调用过

**现象。** `data.py:214` 的 `anon_mask()` 写得很仔细（连 `mask_path` 是裸文件名这个坑都
处理了），`tests/test_raytun3r.py:703` 也测了路径解析。但全仓库 `grep anon_mask` 的结果
只有 `data.py` 自己和那个测试——`build_windows`、`load_sequence`、`eval.py` 一个都没调。

**实测（本地 24 帧）：**

* 24/24 帧都有被匿名的像素；占比 mean **0.22%**、median 0.21%、max 0.60%
* 被匿名区域在 RGB 里的均值是 **(238, 17, 238)**，即官方文档说的品红 (255,0,255)

**为什么是问题（不是因为面积，是因为性质）。** 0.22% 的面积对 `AbsRel` 影响微乎其微，
但这些像素：

* 是**饱和品红**，对任何在自然图像上预训练的骨干都是极端 OOD 输入，而它们直接进
  patch embedding；
* 边界处图像梯度极大 → Eq. 10 的 edge-aware 平滑权重在那里被彻底关掉；
* 覆盖的是**人脸和屏幕**，即场景里唯一会动的东西。UFM 在这些块上产生的对应会进
  Eq. 8 的重投影损失和 MAGSAC++ 的位姿伪标签，而它们违反静态场景假设。

MapAnything 把 `anon_mask_distorted` 当作一等 frame modality 一路带下去；DAC 不用。
我们既然已经把掩码读进来了，不用它是纯粹的浪费。

**另有一个潜伏 bug。** `data.py:197-199` 的搜索顺序是
`("resized_anon_masks", "resized_undistorted_masks", "masks")`。第二项是**给去畸变图用的**
掩码（官方文档：同样语义，但对应 undistorted 版本）。一旦某个场景缺 `resized_anon_masks`，
`anon_mask()` 会静默返回一张几何上对不上的掩码贴到鱼眼原图上。现在因为没人调用所以不炸，
修 P1-2 的时候必须一起改：找不到 `resized_anon_masks` 就返回 `None` 并 warn，不要回退。

**建议。** 在 `Window` 上加 `image_valid`，`build_windows` 里
`valid = camera.valid_mask(...) & anon_mask(i)`，让它同时作用于 Eq. 8/10 的 Ω 和
`depth_metrics` 的 `valid`。

### P1-3　ScanNet++ 上的 `AbsRel` / `δ₁.₂₅` 从来没有真正测出来过

`render_depth` 在 `results` 分支那次运行时就不存在，本地样本的 `MASKS_ADDED.json` 也确认
`dslr/render_depth` 在 netapp 上**任何检查过的场景里都不存在**。也就是说 Tab. 3 右半张表
到今天为止一个数都没有。

这不是代码 bug，但它是这条链路上最大的实际缺口，而且**补它需要做几个会影响数字的决定**，
现在应该先记下来：

* `common/render.py` 的默认是 `near = 0.05`、`far = 20.0`，之后 `*1000` 再 `clip(0, 65535)`。
  → 超过 20 m 的真实深度**不会被渲出来**（记为 0 = 无效），而不是被记成 20 m。渲的时候
  用了什么 near/far 必须写进产物旁边，否则以后没人说得清 0 到底是「网格有洞」还是「太远」。
* renderpy 的鱼眼畸变是**在 vertex shader 里逐顶点算的**，三角形内部按线性光栅化插值。
  ScanNet++ 的网格是 `mesh_aligned_0.05.ply`（5 cm），在 θ→85° 的边缘，真实鱼眼下弯曲的
  三角形边被渲成直线。这是**参考数据本身的精度上限**，而且正好落在 RayTun3R 最关心的区域。
  我没有量化它——但在解释边缘 `AbsRel` 时不能假装它不存在。
* 渲染读的是 `dslr/colmap/`（单相机），输出分辨率来自 COLMAP 的 `cameras.txt`。我们的
  `depth()` 用 `d.shape` 反推相机（`data.py:268`），对分辨率不匹配是稳健的——这点比 DAC 好
  （DAC 直接把 LUT `INTER_CUBIC` 缩过去）——但**没有 assert**。建议加一句：深度图尺寸与
  `transforms.json` 的 `w/h` 宽高比不一致就报错，不要静默。

### P2-1　深度没有任何上下限保护

`metrics.py:164-171` 的 `depth_metrics` 只有 `gt > 1e-6`，`eval.py` 的 `--max-depth`
默认 `None`。DAC 用 `min_depth 0.01 / max_depth 40`。

实际风险有限（渲染器 far=20 已经把远处切成 0 了），但两个具体隐患：

* 换算是 `d / cos`，边缘 `1/cos` 最大 10.89。一个 20 m 的 planar z 在边缘会变成 218 m 的
  range。这在几何上是**自洽的**（说明那条光线穿出了窗户），但只要网格在窗外有任何伪几何，
  它就会以 200 m 量级的值进 `align_scale` 和 `AbsRel`。
* `align_scale` 默认 `robust=True`（比值的中位数，`metrics.py:72-73`），对上述离群值是稳健的；
  但 `AbsRel` 的均值不是。

**建议。** `depth_metrics` 加一个默认的 `min_depth`（0.05，即渲染 near）和一个
**在换算之后**生效的 range 上限（比如 30 m），并把被裁掉的像素比例打出来。

### P2-2　视图配对没有共视度约束

MapAnything 用 pairwise covisibility（阈值 0.25）来采样多视图组；我们用「连续帧 + 平均光流
≥ 2 px」（`data.py:404-441`）。在 stride 10 上这大致够用（README Finding 1 已经量化过
stride 的影响），但它只保证「动了」，不保证「看的是同一块地方」。ScanNet++ DSLR 是手持环拍，
连续 3 帧转过 ~20° 时重叠会掉得很快。

**建议。** 在 `build_windows` 里顺手记录每个窗口的匹配覆盖率 `n_matches / |Ω|` 并写进
结果 JSON。不需要马上改采样策略，先把这个数测出来。

### P3-1　`--stride` 默认还是 1

`train.py:129` 和 `eval.py:243` 都是 `default=1`，而 README 的 Finding 1 明确说
stride 1 「测的是协议不是方法」、stride 10 是两个实验驱动脚本的默认。命令行默认值和
文档结论相反，是个纯粹的脚枪。建议把两处默认改成 10，并在 `stride < 5` 时 warn。

---

## 6. 如果哪天真要在 ScanNet++ 上做监督深度训练，现在缺什么

诚实地列一下，免得以后以为「改几行就能训」：

1. `render_depth`——**要自己渲**，需要 scannetpp toolkit + renderpy + 每个场景的
   `scans/mesh_aligned_0.05.ply`（35 MB/场景）；
2. 场景级 split——官方 `splits/nvs_sem_train.txt`（856 场景）/ `nvs_sem_val.txt`（50 场景）；
3. 一个真的 dataloader——现在的 `build_windows` 是单进程、每次跑都重跑 matcher、
   `O(场景)` 的，不能喂给 DDP；
4. 深度损失——`losses.py` 里没有任何和 GT 深度有关的项；
5. 一个决定：**训鱼眼还是训去畸变**。MapAnything 选 undistort 成针孔，DAC 选转 ERP。
   对我们而言，如果目的是延续 fisheye/wide-FOV 那条线，就应该保持鱼眼——但那意味着
   要么像 DAC 那样进 ERP，要么模型本身得吃相机参数。

---

## 7. 建议的修复顺序

| # | 项 | 改动量 | 会不会改数字 |
|---|---|---|---|
| 1 | P1-2 掩码接进 Ω（并去掉 `resized_undistorted_masks` 回退） | ~15 行 | 会（小） |
| 2 | P1-1 读 `test_frames` + `--full-sequence` + eval/train seed 分离 | ~40 行 | **会（大，影响与论文的可比性）** |
| 3 | P3-1 `--stride` 默认改 10 | 2 行 | 会（已知） |
| 4 | P2-1 深度上下限 + 裁剪比例统计 | ~10 行 | 会（小） |
| 5 | P2-2 记录匹配覆盖率 | ~5 行 | 不会（纯观测） |
| 6 | P1-3 渲 `render_depth`，并把 near/far 写进产物 | 一个真活儿 | 从「没有数」到「有数」 |

1、2、4 会改变数值，跑过的 ScanNet++ 结果需要在修完之后重跑；README「Measured results」
那一节的免责声明要再加一条。

---

## 附：本文所有实测数字的复现方式

FOV / `1/cos` / 半像素敏感度 —— 纯 numpy，只需要 `transforms.json`：

```bash
python3 - <<'PY'
import json, math, numpy as np
m = json.load(open('<scene>/dslr/nerfstudio/transforms.json'))
fx, fy, cx, cy, W, H = m['fl_x'], m['fl_y'], m['cx'], m['cy'], m['w'], m['h']
k = np.array([m['k1'], m['k2'], m['k3'], m['k4']])
td = lambda t: t * (1 + k[0]*t**2 + k[1]*t**4 + k[2]*t**6 + k[3]*t**8)
ts = np.linspace(0, math.pi/2, 20000); rs = td(ts)
r = math.hypot(max(cx, W-cx)/fx, max(cy, H-cy)/fy)
print('diagonal FOV %.1f deg' % (2*math.degrees(np.interp(r, rs, ts))))
PY
```

拟合/评测重叠 —— 复刻 `build_windows` 的 seeded shuffle，不需要数据：

```bash
python3 - <<'PY'
import numpy as np
n = 753
def picks(seq_len, stride, n_windows, seed=0):
    starts = list(range(0, max(n-(seq_len-1)*stride, 0)))
    np.random.RandomState(seed).shuffle(starts)
    out = []
    for s in starts:
        idx = [s + k*stride for k in range(seq_len)]
        if idx[-1] < n: out.append(idx)
        if len(out) >= n_windows: break
    return out
tr = picks(3, 10, 30); ev = picks(2, 10, 100)
trf = {i for w in tr for i in w}; evf = {i for w in ev for i in w}
print(len(evf), len(trf & evf))
PY
```
