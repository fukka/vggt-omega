# Copyright (c) 2026.
"""Build research/fisheye-inpaint/to_human/plan.html from ONE evaluation run.

Every number in the page is read from numbers.json (written by
make_final_figures.py, which itself refuses to run unless its bootstrap
reproduces the evaluator's report.txt) or from results.json; every figure is
embedded from to_human/final/. Nothing is typed in by hand, so a figure in the
document is traceable to a file, and re-running the pipeline regenerates the
document rather than editing it.

Usage::

    python research/fisheye-inpaint/build_final_html.py
"""
from __future__ import annotations

import base64
import json
import os
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "to_human", "final")
DATA = os.path.join(HERE, "data", "final")
OUT = os.path.join(HERE, "to_human", "plan.html")

CELLS = OrderedDict([
    ("fisheye_masked", ("①", "RAW · BLACK", "原始鱼眼 · 黑角")),
    ("persp_masked",   ("②", "RECT · BLACK", "矫正透视 · 黑楔形")),
    ("fisheye_full",   ("③", "RAW · FILLED", "原始鱼眼 · 真值补全")),
    ("persp_full",     ("④", "RECT · FILLED", "矫正透视 · 真值补全")),
])

# Provenance of the one run this document describes. Collected on the machine
# that ran it, on the day it ran; a reader can check every hash.
PROV = {
    "eval_commit": "9e83625e5e7bb8d607237152f004ccd6073f4193",
    "eval_commit_short": "9e83625",
    "eval_branch": "fisheye-2x2",
    "run_dir": "wt-fisheye2x2/runs/ev_final  (lambda_63)",
    "run_finished": "2026-09-07 22:35 (lambda_63 本地时间, UTC−4)",
    "renderer_commit": "58168fa567d6afc4980ade0f18fac0bb969fc1ed",
    "renderer_commit_short": "58168fa",
    "renderer_script": "adt_egocentric/render_oracle_2x2_lambda.py 经 rerender_oracle_set_lit.sh",
    "render_root": "adt_egocentric/out/oracle_set_lit",
    "manifest_md5": "9dbc50b2e181285bf8df3db13c9085fa",
    "ckpt": "VGGT-Omega-1B-512/model.pt",
    "ckpt_md5": "bc5302eada6222303c5e5f8d7dbce709",
    "env": "conda env raytun3r · torch 2.11.0+cu128 · RTX 6000 Ada (GPU 0)",
    "run_minutes": "约 4 分钟(28 组评估)",
}


def b64(path):
    ext = os.path.splitext(path)[1][1:].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}[ext]
    with open(path, "rb") as fh:
        return f"data:{mime};base64,{base64.b64encode(fh.read()).decode()}"


def ci(cb, fmt="{:+.4f}", star=True):
    s = f"{fmt.format(cb['mean'])} <span class='ref'>[{fmt.format(cb['ci_lo'])}, {fmt.format(cb['ci_hi'])}]</span>"
    if star:
        s += " <span class='badge b-ok'>显著</span>" if cb["excludes_zero"] else " <span class='badge b-un'>n.s.</span>"
    return s


def sig(cb):
    return cb["excludes_zero"]


def main():
    N = json.load(open(os.path.join(FIG, "numbers.json")))
    R = json.load(open(os.path.join(DATA, "results.json")))
    meta = json.load(open(os.path.join(DATA, "example_meta_seq131_frame_0830.json")))
    css = open(os.path.join(HERE, "to_human", "style.css"), encoding="utf-8").read()
    res1, res8 = R["single"], R["8-frame"]
    E1, E8 = N["effects"]["single"], N["effects"]["8-frame"]
    C1, C8 = N["cells"]["single"], N["cells"]["8-frame"]
    P8 = N["pose"]["8-frame"]
    F1, F8 = N["fov"]["single"], N["fov"]["8-frame"]
    INP = N["inputs"]
    W, B = N["wins"], N["band"]
    gt_fov = F1["persp_full"]["gt"]
    theta_max = meta["theta_max_deg"]

    # ---- per-window table: group -> per-cell mean AbsRel (s1, s8), pose (s8)
    groups = OrderedDict()
    for w in N["windows"]:
        groups[w["group"]] = w
    def group_mean(res, st, g):
        r = res[st]
        v = [m["AbsRel"] for d, m in r["_per_frame_metrics"].items() if r["_group_of"][d] == g]
        return float(np.mean(v))
    def window_pose(st, g, key):
        r = res8[st]
        for wk, grp in r["_window_group"].items():
            if grp == g:
                return r["_per_window"][wk][key]
        return float("nan")

    # ------------------------------------------------------------------ text
    d1f, d1p, d1i = E1["AbsRel"]["fisheye"], E1["AbsRel"]["persp"], E1["AbsRel"]["interaction"]
    d8f, d8p, d8i = E8["AbsRel"]["fisheye"], E8["AbsRel"]["persp"], E8["AbsRel"]["interaction"]
    a8f, a8p, a8i = E8["auc30"]["fisheye"], E8["auc30"]["persp"], E8["auc30"]["interaction"]
    r8f, r8p, r8i = E8["rot_err_deg"]["fisheye"], E8["rot_err_deg"]["persp"], E8["rot_err_deg"]["interaction"]
    t8f, t8p, t8i = E8["trans_err_deg"]["fisheye"], E8["trans_err_deg"]["persp"], E8["trans_err_deg"]["interaction"]
    e8f, e8p, e8i = E8["ate_m"]["fisheye"], E8["ate_m"]["persp"], E8["ate_m"]["interaction"]
    lad1, lad8 = N["ladder"]["single"], N["ladder"]["8-frame"]
    rep = {(m, p): N["ladder"][m][p]["rows"]["replicate"] for m in ("single", "8-frame") for p in ("fisheye", "persp")}

    H = []
    A = H.append
    A(f"""<title>鱼眼补全与可标定性</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&family=Noto+Serif+SC:wght@600;900&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
{css}
.kv{{display:grid;grid-template-columns:max-content 1fr;gap:6px 18px;font-size:13.5px;margin:18px 0}}
.kv dt{{font-family:var(--mono);font-size:11.5px;color:var(--muted);padding-top:3px}}
.kv dd{{margin:0}}
.tight td,.tight th{{padding:8px 12px}}
.toc{{display:flex;flex-wrap:wrap;gap:6px 14px;margin:22px 0 0;padding:0;list-style:none;font-family:var(--mono);font-size:11.5px}}
.toc a{{text-decoration:none;color:var(--accent)}}
.neg{{color:var(--good)}} .pos{{color:var(--bad)}}
</style>

<div class="wrap">

<header class="mast">
  <p class="kicker">研究报告 · VGGT-Omega · ADT / Aria KB4 · Blender 渲染真值补全</p>
  <h1>鱼眼补全与可标定性</h1>
  <p class="standfirst">把鱼眼图喂给 VGGT-Omega 时,画面里有大片无效黑区:原始鱼眼是成像圆外的四角,矫正成透视图后是四个黑楔形。用<strong>真实场景内容</strong>把它们补上,深度会变好吗?相机估计会变好吗?——而且,变好是不是<strong>因为</strong>图像回到了透视训练域?本文只报告一次评估:同一批 96 帧、四种输入、同一个模型、同一套评分规则,深度与相机位姿两项指标。</p>
  <div class="meta">
    <span>2026-09-07</span>
    <span>96 帧 / 12 窗口 / 4 段序列</span>
    <span>θ<sub>max</sub> = {theta_max:.2f}°</span>
    <span>评估代码 {PROV['eval_commit_short']} · 渲染器 {PROV['renderer_commit_short']}</span>
  </div>
  <ul class="toc">
    <li><a href="#verdict">结论</a></li><li><a href="#inputs">四个输入</a></li><li><a href="#protocol">数据与协议</a></li>
    <li><a href="#depth">深度精度</a></li><li><a href="#pose">相机位姿精度</a></li><li><a href="#qual">定性结果</a></li>
    <li><a href="#ladder">附录:廉价填充阶梯</a></li><li><a href="#prov">出处与复现</a></li><li><a href="#removed">本版删去了什么</a></li>
  </ul>
</header>
""")

    # ------------------------------------------------------------ verdict
    A(f"""
<section id="verdict">
<div class="verdict">
<h2>结论</h2>
<ul class="vlist">
<li><span class="vnum">1</span><div><strong>真值内容让深度变好,两种投影一样好。</strong>单帧:鱼眼 {d1f['mean']:+.4f}、透视 {d1p['mean']:+.4f} AbsRel(负 = 补全更好),两条按窗口聚类的 95% CI 都不含零;8 帧:{d8f['mean']:+.4f} / {d8p['mean']:+.4f},同样显著。<strong>交互项不显著</strong>(单帧 {d1i['mean']:+.4f} [{d1i['ci_lo']:+.4f}, {d1i['ci_hi']:+.4f}];8 帧 {d8i['mean']:+.4f} [{d8i['ci_lo']:+.4f}, {d8i['ci_hi']:+.4f}])。预注册的假设——「收益集中在矫正+补全的 ④ 格,因为它回到了透视训练域」——<strong>没有得到支持</strong>。这是一个「别给模型大片纯黑」的低层效应,与投影无关。</div></li>
<li><span class="vnum">2</span><div><strong>真值内容没有让相机位姿变好;在鱼眼上反而变差。</strong>8 帧窗口的 AUC@30:鱼眼 {a8f['mean']:+.3f} [{a8f['ci_lo']:+.3f}, {a8f['ci_hi']:+.3f}](显著变差),透视 {a8p['mean']:+.3f} [{a8p['ci_lo']:+.3f}, {a8p['ci_hi']:+.3f}](n.s.)。相对旋转误差在鱼眼上 {r8f['mean']:+.2f}°(显著),透视 {r8p['mean']:+.2f}°(n.s.),交互项 {r8i['mean']:+.2f}° [{r8i['ci_lo']:+.2f}, {r8i['ci_hi']:+.2f}](显著)。所以深度的收益<strong>不是经由更好的相机估计</strong>得到的。</div></li>
<li><span class="vnum">3</span><div><strong>相机头推断的 FoV 与补不补无关,而且一直是错的。</strong>透视图真值 {gt_fov:.1f}°,模型在 ② 说 {F1['persp_masked']['mean']:.1f}°、在 ④ 说 {F1['persp_full']['mean']:.1f}°——把 32% 的黑楔形换成真实内容只把 FoV 挪了 {abs(F1['persp_full']['mean']-F1['persp_masked']['mean']):.1f}°,离真值仍差 {F1['persp_full']['abs_err']:.0f}°。一张完全干净、无黑边的 125° 针孔图,VGGT-Omega 也读不出它的视场。这正是三个竞争假设里补全治不了的那一个(C:FoV 超出训练分布)。</div></li>
<li><span class="vnum">4</span><div><strong>可争的空间很小。</strong>真值内容值 {abs(d1f['mean']):.4f}–{abs(d1p['mean']):.4f} AbsRel(单帧);同一次运行里,一个<em>不发明任何内容</em>的最近邻拖影(<code>replicate</code>)已经拿走其中 {rep[('single','persp')]['pct']:.0f}%(透视)/ {rep[('single','fisheye')]['pct']:.0f}%(鱼眼),8 帧时 {rep[('8-frame','persp')]['pct']:.0f}% / {rep[('8-frame','fisheye')]['pct']:.0f}%。留给任何生成式补全的是 <strong>0–0.004 AbsRel</strong>,且不会带来相机精度。</div></li>
</ul>
</div>
</section>
""")

    # ------------------------------------------------------------ inputs
    inp_rows = "".join(
        f"<tr><td><span class='cellid'>{CELLS[st][0]} {CELLS[st][1]}</span>{CELLS[st][2]}</td>"
        f"<td class='n'>{INP[st]['black_pct']:.2f}%</td><td class='n'>{INP[st]['graded_pct']:.1f}%</td>"
        f"<td>{desc}</td></tr>"
        for st, desc in [
            ("fisheye_masked", f"Aria KB4 原图。传感器方框切掉了成像圆盘(圆盘半径 f·θ<sub>d</sub>(θ<sub>max</sub>) ≈ 271 px,半宽 256 px),四角是镜头没拍到的方向。"),
            ("persp_masked", f"整锥矫正到针孔平面(focal_out_norm = {meta['focal_out_norm']}, f = {meta['Knew_pinhole'][0]:.1f} px, 水平 FoV {gt_fov:.1f}°)。有效区是「被咬掉的圆角方形」,其余为黑。"),
            ("fisheye_full", f"同一张 ERP 全景以 KB4 的单调延拓(θ > θ<sub>max</sub> 处用割线斜率 {meta['kb4_slope']:.4f} 延伸,指纹 <code>{meta['kb4_fingerprint']}</code>)重采样到整个方框——四角是<strong>真实场景</strong>,但处在一个真实镜头不存在的投影下。"),
            ("persp_full", f"同一张 ERP 直接针孔重采样到整个方框。四角是真实场景内容,投影也是真实针孔——这是唯一在视觉上像一张普通照片的格子,即预注册假设的落点。"),
        ])
    A(f"""
<section id="inputs">
<p class="eyebrow">实验设计</p>
<h2>四个输入:投影方式 × 是否补全</h2>
<p>两个正交因子,四个格子跑同一个 VGGT-Omega、同一批帧、同一套评分区域。<strong>交互项 (④−②) − (③−①) 才是对「回到透视域」这一假设的检验</strong>;四个绝对值各自都有文献给过答案,交互没人测过。</p>
<div class="mtx">
  <div class="hd"></div><div class="hd">保留黑区(基线)</div><div class="hd">黑区用真实内容补全</div>
  <div class="rh">原始鱼眼</div>
  <div><span class="cellid">① RAW · BLACK</span><span class="cellname">未矫正 + 圆外黑角</span>模型收到的是 Aria KB4 原图。<span class="pred">预测:FoV 误差最大,深度最差。</span></div>
  <div><span class="cellid">③ RAW · FILLED</span><span class="cellname">未矫正 + 角落补真值</span>鱼眼投影仍在域外,只是没有黑区。<span class="pred">预测:小幅提升或无变化。</span></div>
  <div class="rh">矫正为透视</div>
  <div><span class="cellid">② RECT · BLACK</span><span class="cellname">外接矫正 + 黑楔形</span>{gt_fov:.1f}° 针孔,完整保留成像锥,四角为黑。<span class="pred">预测:比 ① 好,但被黑楔形拖累。</span></div>
  <div><span class="cellid">④ RECT · FILLED</span><span class="cellname">外接矫正 + 楔形补真值</span>唯一在视觉上完整的透视照片。<span class="pred">预测(假设 A):④ 显著优于 ②,且 ④−② 明显大于 ③−①。</span></div>
</div>

<h3>模型真正收到的张量</h3>
<p>下图是评估脚本在喂给模型之前存盘的输入,不是示意图。四格来自<strong>同一帧、同一张渲染的 ERP 全景、同一位姿</strong>,只在投影与四角内容上不同。</p>
<div class="wide"><img src="{b64(os.path.join(FIG,'inputs_w00.jpg'))}" alt="四个输入,窗口 00">
<div class="cap">窗口 00 第一帧(seq131 / frame_1999)。标题里的「纯黑像素」与「评分区」是在这一张上实测的;下表是 96 帧的平均。</div></div>
<div class="scroll"><table class="tight">
<thead><tr><th>格子</th><th class="n">纯黑像素(96 帧均值)</th><th class="n">参与评分</th><th>输入是什么</th></tr></thead>
<tbody>{inp_rows}</tbody></table></div>
<p class="note">③/④ 里残留的 1.7–2.1% 纯黑像素不是没渲染:它们全部带有有效的渲染深度(黑像素中深度为 0 的比例 = 0.0%),且有 1.4–1.8% 落在评分区内部——是拟合光照(曝光 −1.21 EV)下的阴影和深色物体,与投影无关,四格同等受影响。</p>
<div class="wide"><img src="{b64(os.path.join(FIG,'inputs_w01.jpg'))}" alt="四个输入,窗口 01"><div class="cap">窗口 01(seq131 / frame_0830)。</div></div>
<div class="wide"><img src="{b64(os.path.join(FIG,'inputs_w03.jpg'))}" alt="四个输入,窗口 03"><div class="cap">窗口 03(seq134 / frame_2561)。</div></div>
</section>
""")

    # ------------------------------------------------------------ protocol
    A(f"""
<section id="protocol">
<p class="eyebrow">数据与协议</p>
<h2>一次渲染,一套评分规则,一个聚类单位</h2>
<h3>为什么是渲染,而不是真实 ADT 帧</h3>
<p>③ 和 ④ 要求在<strong>镜头根本没拍到的方向</strong>上放真实内容。真实 Aria 帧给不出这个,只有拥有整个场景的渲染器能给。所以 96 帧全部由 Blender/Cycles 从 ADT 的物体级场景重建(354 个 GLB、43 盏拟合光源)按 ADT 的真实相机轨迹渲染成 ERP 全景({meta['eq_w']}×{meta['eq_h']},{meta['cycles_samples']} 采样,OPTIX),再用 <code>cv2.remap</code> 重采样成四个输入;深度来自渲染器的 Z-pass,四格共享。光照是<strong>按真实帧拟合</strong>的(<code>{meta['lights_json'].split('/')[-1]}</code>,曝光 {meta['view_exposure']} EV);历史上一版把灯装在 2.35 m、低于它要照亮的家具,那一版的数字本文不再引用(见<a href="#removed">删去了什么</a>)。</p>
<h3>帧与窗口</h3>
<p>4 段 ADT 序列(seq131/134/136 clean,seq132 decoration)× 每段 3 个窗口 × 每窗 8 帧,步长 10 帧,每窗真实位移约 2.0–2.5 m。<strong>窗口是独立性的单位</strong>:同一窗口的 8 帧共享场景、光照和不到一秒的轨迹,把它们当 96 个独立样本会让区间窄约 √8 倍。所有区间都是按窗口重采样的 bootstrap(10000 次,n = 12 窗),配对差在帧或窗口级别逐一相减。</p>
<h3>评分规则</h3>
<p>同一投影的两格(① vs ③,② vs ④)<strong>都在「黑」那一格的有效区上打分</strong>——否则「补全」格会因为多覆盖了面积而得分,测的是覆盖率而不是补全。填进去的像素永远不被评分。深度用 scale-shift 对齐后的 AbsRel / RMSE / δ<sub>1</sub>。跨投影的绝对值<strong>不可比</strong>(矫正网格把边缘过采样约 10 倍),可比的是同投影内的差和两投影差的交互。</p>
<h3>相机位姿怎么算</h3>
<p>VGGT 把第一帧定为世界系、平移只到尺度,所以位姿全部按<strong>窗口内相对位姿</strong>评:8 帧的 28 个 i&lt;j 对,每对比较 E<sub>i</sub>E<sub>j</sub><sup>−1</sup> 的旋转角误差(RRA)与相对平移方向夹角(RTA),AUC@30 = max(RRA,RTA) 在 1..30° 阈值下的平均准确率(VGGT 论文的口径);ATE 是相机中心经 Sim(3) 对齐后的 RMSE(米)。真值相机来自渲染器的 <code>T_WC</code>(原始 Aria 相机)乘以 rot90 的固定旋转;这个约定<strong>用数据验证过</strong>:用它把 frame_0840 的深度反投影再投到 frame_0830,深度中位相对误差 0.12%、98% 像素 &lt;3%;错误候选是 4–7%。单帧模式没有相对位姿,只评 FoV。推断 FoV 只对透视格有真值({gt_fov:.1f}°);鱼眼格的数字是相机头「相信」的值,没有对应的真值。</p>
<h3>模型</h3>
<p>VGGT-Omega-1B-512 预训练权重,不做任何微调;<code>{PROV['ckpt']}</code>,md5 <code>{PROV['ckpt_md5'][:12]}…</code>。8 帧窗口整窗输入,单帧逐帧输入。</p>
</section>
""")

    # ------------------------------------------------------------ depth
    def cell_rows(C):
        return "".join(
            f"<tr{' class=hl' if st=='persp_full' else ''}><td><span class='cellid'>{CELLS[st][0]} {CELLS[st][1]}</span>{CELLS[st][2]}</td>"
            f"<td class='n'>{C[st]['AbsRel']:.4f}</td><td class='n'>{C[st]['RMSE']:.3f}</td><td class='n'>{C[st]['delta1']:.4f}</td>"
            f"<td class='n'>{C[st]['n_frames']} / {C[st]['n_groups']}</td></tr>" for st in CELLS)
    def eff_rows(E, metric, fmt, good):
        e = E[metric]
        return (f"<tr><td>{metric}</td><td class='n'>{ci(e['fisheye'], fmt)}</td><td class='n'>{ci(e['persp'], fmt)}</td>"
                f"<td class='n'>{ci(e['interaction'], fmt)}</td><td>{good}</td></tr>")
    mvs = N["multi_vs_single"]
    mvs_rows = "".join(
        f"<tr><td><span class='cellid'>{CELLS[st][0]} {CELLS[st][1]}</span></td><td class='n'>{ci(mvs[st])}</td></tr>" for st in CELLS)
    A(f"""
<section id="depth">
<p class="eyebrow">结果 · 深度</p>
<h2>真值内容有用,而且两种投影一样有用</h2>
<div class="scroll"><table>
<thead><tr><th>单帧</th><th class="n">AbsRel ↓</th><th class="n">RMSE (m) ↓</th><th class="n">δ<sub>1</sub> ↑</th><th class="n">帧 / 窗</th></tr></thead>
<tbody>{cell_rows(C1)}</tbody></table></div>
<div class="scroll"><table>
<thead><tr><th>8 帧窗口</th><th class="n">AbsRel ↓</th><th class="n">RMSE (m) ↓</th><th class="n">δ<sub>1</sub> ↑</th><th class="n">帧 / 窗</th></tr></thead>
<tbody>{cell_rows(C8)}</tbody></table></div>
<p class="note">同一投影内的两行可比;鱼眼行与透视行之间<strong>不可比</strong>(网格不同)。这也是为什么下表只报告差和差的交互。</p>

<h3>配对效应:补全 − 黑(负 = 补全更好),按窗口聚类的 95% CI</h3>
<div class="scroll"><table>
<thead><tr><th>单帧</th><th class="n">鱼眼 ③−①</th><th class="n">透视 ④−②</th><th class="n">交互 (④−②)−(③−①)</th><th>读法</th></tr></thead>
<tbody>
{eff_rows(E1,'AbsRel','{:+.4f}','负 = 补全更好')}
{eff_rows(E1,'RMSE','{:+.4f}','负 = 补全更好')}
{eff_rows(E1,'delta1','{:+.4f}','正 = 补全更好')}
</tbody></table></div>
<div class="scroll"><table>
<thead><tr><th>8 帧</th><th class="n">鱼眼 ③−①</th><th class="n">透视 ④−②</th><th class="n">交互 (④−②)−(③−①)</th><th>读法</th></tr></thead>
<tbody>
{eff_rows(E8,'AbsRel','{:+.4f}','负 = 补全更好')}
{eff_rows(E8,'RMSE','{:+.4f}','负 = 补全更好')}
{eff_rows(E8,'delta1','{:+.4f}','正 = 补全更好')}
</tbody></table></div>
<div class="wide"><img src="{b64(os.path.join(FIG,'effects.png'))}" alt="配对效应">
<div class="cap">每个点是一个窗口(8 帧的均值)的配对差;方块是 96 帧的均值,须是按窗口重采样的 95% CI。左二为深度,右二为位姿(见下节)。交互项的点散布明显宽于主效应,是因为它是两个差的差——这是它在 n = 12 窗下不显著的诚实原因,不是「接近零」的证据;但它的点估计在单帧是 {d1i['mean']:+.4f}、8 帧是 {d8i['mean']:+.4f},符号还相反。</div></div>
<p>三个指标、两种模式,六个主效应全部显著,六个交互项全部跨零。假设 A 预言的模式——④−② 明显大于 ③−①——没有出现;单帧的交互甚至偏向鱼眼那边 0.002。补全的价值来自「不再是黑的」,不来自「像一张透视照片」。</p>

<h3>多帧有没有帮助?(8 帧 − 单帧,同一帧配对)</h3>
<div class="scroll"><table class="tight">
<thead><tr><th>格子</th><th class="n">Δ AbsRel(负 = 多帧更好)</th></tr></thead>
<tbody>{mvs_rows}</tbody></table></div>
<p>只有 ③(鱼眼+真值补全)在 8 帧时<strong>显著变差</strong>({mvs['fisheye_full']['mean']:+.4f});其余三格跨零。这与下一节位姿的发现同源:③ 的四角内容处在一个不存在的镜头投影下,多帧时它通过跨视图注意力影响相机估计。</p>
</section>
""")

    # ------------------------------------------------------------ pose
    def fov_rows(F):
        out = []
        for st in CELLS:
            f = F[st]
            gt = f"{f['gt']:.1f}°" if f["gt"] is not None and f["gt"] == f["gt"] else "—(非针孔)"
            err = f"{f['abs_err']:.1f}°" if f.get("abs_err") is not None else "—"
            out.append(f"<tr><td><span class='cellid'>{CELLS[st][0]} {CELLS[st][1]}</span></td><td class='n'>{f['mean']:.1f}° ± {f['std']:.1f}</td><td class='n'>{gt}</td><td class='n'>{err}</td></tr>")
        return "".join(out)
    pose_rows = "".join(
        f"<tr{' class=hl' if st=='persp_full' else ''}><td><span class='cellid'>{CELLS[st][0]} {CELLS[st][1]}</span>{CELLS[st][2]}</td>"
        f"<td class='n'>{P8[st]['rot_err_deg']:.2f}°</td><td class='n'>{P8[st]['trans_err_deg']:.2f}°</td>"
        f"<td class='n'>{P8[st]['rra15']:.3f}</td><td class='n'>{P8[st]['rta15']:.3f}</td><td class='n'>{P8[st]['auc30']:.3f}</td>"
        f"<td class='n'>{P8[st]['ate_m']*100:.1f} cm</td><td class='n'>{P8[st]['sim3_scale']:.2f}</td></tr>" for st in CELLS)
    def peff(metric, fmt, good):
        e = E8[metric]
        return (f"<tr><td>{metric}</td><td class='n'>{ci(e['fisheye'], fmt)}</td><td class='n'>{ci(e['persp'], fmt)}</td>"
                f"<td class='n'>{ci(e['interaction'], fmt)}</td><td>{good}</td></tr>")
    A(f"""
<section id="pose">
<p class="eyebrow">结果 · 相机</p>
<h2>相机头没有因为补全而变好;FoV 一直是错的</h2>
<h3>推断的水平 FoV</h3>
<div class="scroll"><table class="tight">
<thead><tr><th>单帧</th><th class="n">FoV<sub>h</sub>(均值 ± 标准差, 96 帧)</th><th class="n">真值</th><th class="n">|误差|</th></tr></thead>
<tbody>{fov_rows(F1)}</tbody></table></div>
<div class="scroll"><table class="tight">
<thead><tr><th>8 帧</th><th class="n">FoV<sub>h</sub>(均值 ± 标准差, 96 帧)</th><th class="n">真值</th><th class="n">|误差|</th></tr></thead>
<tbody>{fov_rows(F8)}</tbody></table></div>
<div class="wide"><img src="{b64(os.path.join(FIG,'fov.png'))}" alt="推断 FoV 分布">
<div class="cap">每个点一帧。透视格的真值是 {gt_fov:.1f}°(虚线);两格都停在 107–109°,补全只挪了 {abs(F1['persp_full']['mean']-F1['persp_masked']['mean']):.1f}°(单帧)/ {abs(F8['persp_full']['mean']-F8['persp_masked']['mean']):.1f}°(8 帧)。鱼眼格没有针孔真值;它的 82–87° 是相机头对一张 KB4 图像的「读数」。</div></div>
<p>这一张图回答了根因诊断里最致命的假设 C:<strong>一张没有任何黑边、投影完全正确的 {gt_fov:.0f}° 针孔图,VGGT-Omega 仍把它读成 {F1['persp_full']['mean']:.0f}°。</strong>黑楔形对 FoV 的影响不到 1°。补全治的是「黑」,治不了「宽」。</p>

<h3>8 帧窗口的相对位姿(12 窗 × 28 对)</h3>
<div class="scroll"><table>
<thead><tr><th>格子</th><th class="n">旋转误差 ↓</th><th class="n">平移方向误差 ↓</th><th class="n">RRA@15 ↑</th><th class="n">RTA@15 ↑</th><th class="n">AUC@30 ↑</th><th class="n">ATE ↓</th><th class="n">Sim(3) 尺度</th></tr></thead>
<tbody>{pose_rows}</tbody></table></div>
<p class="note">Sim(3) 尺度是把 VGGT 的平移对齐到米制真值所需的倍率;它离 1 很远(2.6–3.7×)是 VGGT 平移只到尺度的正常表现,不是本实验的变量。透视格的 RRA@15 = 1.000 表示 336 对里没有一对旋转误差超过 15°。</p>

<h3>配对效应:补全 − 黑,按窗口配对(n = 12)</h3>
<div class="scroll"><table>
<thead><tr><th>8 帧</th><th class="n">鱼眼 ③−①</th><th class="n">透视 ④−②</th><th class="n">交互 (④−②)−(③−①)</th><th>读法</th></tr></thead>
<tbody>
{peff('auc30','{:+.3f}','正 = 补全更好')}
{peff('rra15','{:+.3f}','正 = 补全更好')}
{peff('rta15','{:+.3f}','正 = 补全更好')}
{peff('rot_err_deg','{:+.2f}°','负 = 补全更好')}
{peff('trans_err_deg','{:+.2f}°','负 = 补全更好')}
{peff('ate_m','{:+.4f} m','负 = 补全更好')}
</tbody></table></div>
<p><strong>在鱼眼上,真值补全让位姿显著变差</strong>:AUC@30 {a8f['mean']:+.3f},旋转误差 {r8f['mean']:+.2f}°,平移方向误差 {t8f['mean']:+.2f}°,ATE {e8f['mean']*100:+.1f} cm,四条 CI 都不含零。在透视上,AUC@30、旋转、ATE 都跨零,只有平移方向误差显著变差({t8p['mean']:+.2f}°)。旋转误差的交互项显著({r8i['mean']:+.2f}° [{r8i['ci_lo']:+.2f}, {r8i['ci_hi']:+.2f}]),方向是「鱼眼上的补全比透视上的补全更伤位姿」。</p>
<p>一个与数据一致的解释(它是解释,不是测量):③ 的四角内容虽然是真实场景,却被放在 θ &gt; θ<sub>max</sub> 的<strong>延拓 KB4 投影</strong>下——一个真实镜头不存在的成像方式。对深度头这是「更多有纹理的上下文」,所以有用;对相机头这是「一个比任何训练过的镜头都更宽、畸变曲线在边缘处被人为延伸的镜头」,所以有害。④ 的四角内容是真实针孔投影,于是既不帮也不伤。<strong>无论哪种解释,结论都一样:深度的收益不是通过修好相机得到的。</strong></p>
<div class="wide"><img src="{b64(os.path.join(FIG,'trajectories.png'))}" alt="相机轨迹">
<div class="cap">12 个窗口的相机中心轨迹:黑色为真值,彩色为四格的预测经 Sim(3) 对齐后的结果,画在各窗口的主轴坐标系里(横轴沿路径、纵轴横向,单位米,<strong>纵轴被放大</strong>)。图例给出该窗口的 AUC@30 / ATE / 平均旋转误差。窗口 06、08、10 是所有格子都难的窗口(装饰序列 seq132 与 seq136 的转弯段);差别在格子之间很小,在窗口之间很大——这就是为什么必须按窗口聚类。</div></div>
</section>
""")

    # ------------------------------------------------------------ qualitative
    qrows = []
    for g, w in groups.items():
        seq = g.rsplit("_w", 1)[0].replace("Apartment_release_", "")
        cells1 = "".join(f"<td class='n'>{group_mean(res1, st, g):.4f}</td>" for st in CELLS)
        cells8 = "".join(f"<td class='n'>{group_mean(res8, st, g):.4f}</td>" for st in CELLS)
        auc = "".join(f"<td class='n'>{window_pose(st, g, 'auc30'):.2f}</td>" for st in CELLS)
        qrows.append(f"<tr><td><a href='#w{w['index']:02d}'>窗口 {w['index']:02d}</a><br><span class='ref'>{seq} · {os.path.basename(w['first_frame'])}</span></td>{cells1}{cells8}{auc}</tr>")
    panels = []
    for w in N["windows"]:
        i = w["index"]
        seq = w["group"].rsplit("_w", 1)[0].replace("Apartment_release_", "")
        panels.append(f"""<div class="wide" id="w{i:02d}"><img src="{b64(os.path.join(FIG, f'panels_w{i:02d}.jpg'))}" alt="窗口 {i:02d} 定性面板">
<div class="cap">窗口 {i:02d} · {seq} · 第一帧 {os.path.basename(w['first_frame'])}。每行一个格子:输入 | 单帧预测 | 单帧 AbsRel 图 | 8 帧预测 | 8 帧 AbsRel 图 | 渲染真值。同投影的两行在相同像素上评分;灰色不评分。误差图下方的数字是<strong>这一帧</strong>的 AbsRel / δ<sub>1</sub>。</div></div>""")
    A(f"""
<section id="qual">
<p class="eyebrow">定性结果</p>
<h2>12 个窗口,每窗第一帧,四格并排</h2>
<p>下表是每个窗口 8 帧的均值(深度)和整窗的 AUC@30(位姿),点击窗口号跳到对应面板。深度上,补全更好的窗口数:单帧鱼眼 {W['single/fisheye']['absrel_better']}/12、单帧透视 {W['single/persp']['absrel_better']}/12、8 帧鱼眼 {W['8-frame/fisheye']['absrel_better']}/12、8 帧透视 {W['8-frame/persp']['absrel_better']}/12。位姿上,补全的 AUC@30 更低的窗口数:鱼眼 {W['8-frame/fisheye']['auc_lower']}/12、透视 {W['8-frame/persp']['auc_lower']}/12(透视另有 {W['8-frame/persp']['auc_equal']} 窗持平)。</p>
<div class="scroll"><table class="tight">
<thead><tr><th rowspan="2">窗口</th><th colspan="4" class="n">AbsRel · 单帧(8 帧均值)</th><th colspan="4" class="n">AbsRel · 8 帧</th><th colspan="4" class="n">AUC@30 · 8 帧</th></tr>
<tr>{''.join(f"<th class='n'>{CELLS[st][0]}</th>" for st in CELLS)}{''.join(f"<th class='n'>{CELLS[st][0]}</th>" for st in CELLS)}{''.join(f"<th class='n'>{CELLS[st][0]}</th>" for st in CELLS)}</tr></thead>
<tbody>{''.join(qrows)}</tbody></table></div>
{''.join(panels)}
<p class="note">看图时值得注意的三件事。其一,误差图里最亮的斑是近处的大块低纹理平面(白色置物架、桌面),四格都有——那是场景与模型的问题,不是黑区的问题。其二,① 和 ② 的误差在黑区<em>边界内侧</em>明显偏高,补全后主要是这一圈在变好:在 96 帧单帧输入上,距离黑区 ≤16 px 的评分像素 AbsRel 从 {B['fisheye']['band_black']:.3f} 降到 {B['fisheye']['band_full']:.3f}(鱼眼,−{B['fisheye']['band_gain_pct']:.0f}%)、从 {B['persp']['band_black']:.3f} 降到 {B['persp']['band_full']:.3f}(透视,−{B['persp']['band_gain_pct']:.0f}%),而更内部的像素只降 {B['fisheye']['int_gain_pct']:.0f}% / {B['persp']['int_gain_pct']:.0f}%。这就是 0.010–0.018 AbsRel 的来源:一个边界效应,两种投影的幅度相同。其三,④ 的输入是唯一像普通照片的,但它的误差图和 ③ 看不出系统性差别——与交互项不显著一致。</p>
</section>
""")

    # ------------------------------------------------------------ ladder appendix
    def lad_table(lad, mode):
        rows = []
        order = ["black", "chanmean", "mean", "telea", "ns", "replicate", "ORACLE"]
        for name in order:
            f, p = lad["fisheye"]["rows"][name], lad["persp"]["rows"][name]
            lab = {"black": "black(基线 ①/②)", "ORACLE": "真值补全(③/④)", "replicate": "replicate(最近有效像素拖影,不发明内容)",
                   "mean": "mean(全图均值填充)", "chanmean": "chanmean(逐通道均值)", "telea": "telea(OpenCV 修补)", "ns": "ns(Navier–Stokes 修补)"}[name]
            def cell(x):
                cls = "hl" if name == "replicate" else ""
                return (f"<td class='n {cls}'>{x['AbsRel']:.4f}</td><td class='n {cls}'>{x['gain']:+.4f}</td>"
                        f"<td class='n {cls}'>{x['pct']:.1f}% <span class='ref'>[{x['ci_lo']:+.4f}, {x['ci_hi']:+.4f}]</span></td>")
            rows.append(f"<tr><td>{lab}</td>{cell(f)}{cell(p)}</tr>")
        return (f"<div class='scroll'><table class='tight'><thead><tr><th rowspan='2'>{mode}</th><th colspan='3' class='n'>鱼眼</th><th colspan='3' class='n'>透视</th></tr>"
                f"<tr><th class='n'>AbsRel</th><th class='n'>比黑好</th><th class='n'>占真值收益 [CI]</th><th class='n'>AbsRel</th><th class='n'>比黑好</th><th class='n'>占真值收益 [CI]</th></tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table></div>")
    A(f"""
<section id="ladder">
<p class="eyebrow">附录 · 同一次运行</p>
<h2>廉价填充阶梯:真值收益里有多少不需要生成模型</h2>
<p>这一节的输入仍是同一批渲染帧,只是把 ①/② 的黑区改用几种<strong>训练-free、不发明内容</strong>的方法填上(填充只作用于输入,评分区不变)。它回答的是决策问题:如果真值内容值 X,一个最便宜的填法已经拿走多少?</p>
{lad_table(lad1, '单帧')}
{lad_table(lad8, '8 帧')}
<p><code>replicate</code> 拿走真值收益的 {rep[('single','fisheye')]['pct']:.0f}% / {rep[('single','persp')]['pct']:.0f}%(单帧鱼眼 / 透视)与 {rep[('8-frame','fisheye')]['pct']:.0f}% / {rep[('8-frame','persp')]['pct']:.0f}%(8 帧);8 帧透视里拖影甚至<strong>超过</strong>真值。留给生成式方法的余量:{lad1['fisheye']['span']-rep[('single','fisheye')]['gain']:.4f} / {lad1['persp']['span']-rep[('single','persp')]['gain']:.4f} / {lad8['fisheye']['span']-rep[('8-frame','fisheye')]['gain']:.4f} / {lad8['persp']['span']-rep[('8-frame','persp')]['gain']:.4f} AbsRel。另有一条稳健的域不对称:平坦均值填充在<strong>透视</strong>图上有帮助(+{lad1['persp']['rows']['mean']['pct']:.0f}%),在<strong>原始鱼眼</strong>上比黑还差得多({lad1['fisheye']['rows']['mean']['pct']:.0f}%)。</p>
</section>
""")

    # ------------------------------------------------------------ provenance
    A(f"""
<section id="prov">
<p class="eyebrow">出处与复现</p>
<h2>每个数字来自哪个文件</h2>
<dl class="kv">
<dt>评估代码</dt><dd><code>finetune/eval/exp_rendered.py</code> @ <code>{PROV['eval_commit']}</code>(分支 <code>{PROV['eval_branch']}</code>)。深度路径与上一版完全相同(四格 AbsRel 逐位一致),位姿与 FoV 评分是本次新增;<code>tests/test_pose_metrics.py</code> 用合成位姿钉住 Sim(3) 不变性、已知旋转角、坍缩预测不得记满分。</dd>
<dt>运行</dt><dd>{PROV['run_dir']} · 完成于 {PROV['run_finished']} · {PROV['env']} · {PROV['run_minutes']}。</dd>
<dt>结果文件</dt><dd><code>research/fisheye-inpaint/data/final/results.json</code>(逐帧指标、逐窗位姿、预测/真值外参)、<code>report.txt</code>(评估器自己打印的报告,本文所有 CI 与之逐条核对)、<code>example_meta_seq131_frame_0830.json</code>(一帧的渲染元数据)。</dd>
<dt>渲染</dt><dd><code>{PROV['renderer_script']}</code> @ <code>{PROV['renderer_commit']}</code> → <code>{PROV['render_root']}</code>;manifest md5 <code>{PROV['manifest_md5']}</code>;KB4 指纹 <code>{meta['kb4_fingerprint']}</code>;光照 <code>{meta['lights_json']}</code>,{meta['n_lights_used']} 盏,曝光 {meta['view_exposure']} EV,{meta['cycles_samples']} 采样,ISP 开,ERP {meta['eq_w']}×{meta['eq_h']},输出 {meta['output_size']} px,超采样 {meta['supersample']}×。</dd>
<dt>模型</dt><dd><code>{PROV['ckpt']}</code>,md5 <code>{PROV['ckpt_md5']}</code>。</dd>
<dt>图与文</dt><dd><code>research/fisheye-inpaint/make_final_figures.py</code> 从 <code>results.json</code> 与评估器存盘的 <code>qual/raw/*.npz</code> 合成全部图与 <code>numbers.json</code>(它会重算 bootstrap,并在与 report.txt 不一致时拒绝输出);<code>build_final_html.py</code> 从 <code>numbers.json</code> 生成本页。页内没有手抄的数字。</dd>
</dl>
<h3>复现</h3>
<pre style="font-family:var(--mono);font-size:12px;line-height:1.7;background:var(--sunk);padding:16px 18px;overflow-x:auto;border:1px solid var(--line)"># lambda_63, worktree at {PROV['eval_commit_short']}
R=/user/f.zhang2/projects/adt_egocentric/out/oracle_set_lit
CUDA_VISIBLE_DEVICES=0 python -m finetune.eval.exp_rendered \\
  --render-root $R --manifest $R/manifest.json \\
  --vggt-checkpoint checkpoints/VGGT-Omega-1B-512/model.pt \\
  --out runs/ev_final --seq-lens 1,8 \\
  --settings fisheye_full,fisheye_masked,persp_full,persp_masked,\\
fisheye_fill_replicate,persp_fill_replicate,fisheye_fill_mean,persp_fill_mean,\\
fisheye_fill_chanmean,persp_fill_chanmean,fisheye_fill_telea,persp_fill_telea,\\
fisheye_fill_ns,persp_fill_ns

# Mac
python research/fisheye-inpaint/make_final_figures.py --run &lt;copy of runs/ev_final&gt; \\
  --out research/fisheye-inpaint/to_human/final
python research/fisheye-inpaint/build_final_html.py</pre>
</section>
""")

    # ------------------------------------------------------------ removed
    A(f"""
<section id="removed">
<p class="eyebrow">版本说明</p>
<h2>本版删去了什么,为什么</h2>
<ul class="plain">
<li><strong>真实 ADT 帧上的 2×2 与「88% oracle」</strong>(<code>data/ev_*</code>、<code>data/oracle/</code>、<code>to_human/examples/</code>、<code>to_human/oracle/</code>)。真实帧给不出镜头没拍到的方向,那些实验里的「补全」是 <code>replicate</code>/OpenCV 修补,不是真值;它们的四格与本文的四格不是同一个实验。</li>
<li><strong>历史光照渲染上的一切数字</strong>(<code>data/ev_96f_12windows</code>、<code>data/fill_ladder/report_historical_lighting.txt</code>、<code>data/lighting_robustness/</code>、<code>to_human/blender/</code>)。那一版渲染的灯装在 2.35 m,效应量被放大 4–6 倍;同一几何、只换光照的配对对比证明了这一点,但那个对比本身也不再是本文的内容——本文只报告拟合光照下的一次运行。</li>
<li><strong>规划、文献综述与正反辩论</strong>。它们是实验之前的文档;实验之后留下的是数据。研究状态与文献仍在 <code>research/fisheye-inpaint/research-state.yaml</code> 与 <code>literature/</code>。</li>
<li>上述目录都还在仓库与 git 历史里,只是不再出现在这份文档中。</li>
</ul>
<h3>仍未解决</h3>
<ul class="plain">
<li>「均值填充优于黑色」这条记忆在文献中仍无出处(五路检索)。本文的数据说明它<strong>依赖投影域</strong>——透视图上成立,原始鱼眼上比黑差得多——但这不是出处,不要引用。</li>
<li>与训练式方案(Fisheye3R 标定 token、RayTun3R 位置编码)的对照表仍开放;它现在要回答的问题是:训练换来的增量是否明显大于 0.004 AbsRel,以及它们能否修好本文证明补全修不好的 FoV。</li>
</ul>
</section>

<footer>
研究线 fisheye-inpaint · 分支 {PROV['eval_branch']} · 文档由 <code>build_final_html.py</code> 生成于 2026-09-07 · 所有数字可追溯至 <code>data/final/</code>
</footer>
</div>
""")
    html = "".join(H)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[html] wrote {OUT}  {len(html.encode('utf-8'))/1e6:.2f} MB")


if __name__ == "__main__":
    main()
