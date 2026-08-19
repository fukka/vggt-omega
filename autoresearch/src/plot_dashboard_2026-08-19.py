"""Dashboard figure set, 2026-08-19: every number re-read from its JSON
(H5 pilot r=8 zones come from the H7 eval logs' shared before-arm + the
pilot analysis, provenance noted inline)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Hiragino Sans GB", "PingFang SC",
                               "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "to_human" / "assets"
Z = "near_rim(<=2m,>=38deg)"
GREEN, RED, GREY, BLUE = "#27ae60", "#c0392b", "#95a5a6", "#2980b9"

# ---------------- fig 1: where to intervene (success & failure panorama)
r10 = json.load(open(ROOT / "experiments/h2-center-safe-adapter/results/run_010_even_odd.json"))
r11 = json.load(open(ROOT / "experiments/h2-center-safe-adapter/results/run_011_even_odd.json"))
before = r11["zones"][Z]["before"]
d_table = (r10["zones"][Z]["after"] - before) / before * 100
d_head = (r11["zones"][Z]["after"] - before) / before * 100
h8 = json.load(open(ROOT / "experiments/h8-equal-area/results/probe_a_seq131.json"))["zones"][Z]
d_h8 = (h8["equalarea"] - h8["plain"]) / h8["plain"] * 100
cp = json.load(open(ROOT / "experiments/bench/results/centerph_seq131_odd.json"))["zones"]["near_center(<=2m,<=11deg)"]["absrel_covered"]
cpv = json.load(open(ROOT / "experiments/bench/results/centerph_seq131_odd_vanillacovered.json"))["zones"]["near_center(<=2m,<=11deg)"]["absrel_covered"]
d_cp = (cp - cpv) / cpv * 100
h7r8 = json.load(open(ROOT / "experiments/h7-theta-gated-lora/results/h7_eval_r8.json"))["zones"][Z]["after"]
d_h7 = (h7r8 - 0.567) / 0.567 * 100   # vs uniform-LoRA pilot anchor (h5 analysis.md)
d_h5 = (0.567 - 1.408) / 1.408 * 100  # H5 pilot analysis.md (252px even/odd)
items = [
    ("裁剪矫正 Center-PH\n(近场中心)", d_cp, RED, "动输入"),
    ("等立体角重采样 H8\n(近场边缘)", d_h8, RED, "动输入"),
    ("块内去畸变 H3\n(全图)", 0.0, RED, "动输入"),
    ("视角门控 H7\n(相对普通微调)", d_h7, RED, "喂几何"),
    ("48数查表 第0级\n(近场边缘)", d_table, GREEN, "编码器后面"),
    ("看图小头 第1级\n(近场边缘)", d_head, GREEN, "编码器后面"),
    ("加权微调 第2级试点\n(近场边缘)", d_h5, GREEN, "编码器后面"),
    ("边缘跨帧注意力 第3级\n(待GPU评测)", None, GREY, "编码器后面"),
]
fig, ax = plt.subplots(figsize=(10, 5.2))
ys = np.arange(len(items))[::-1]
for y, (lab, v, c, grp) in zip(ys, items):
    if v is None:
        ax.barh(y, 1, color=GREY, alpha=.35); ax.text(2, y, "等待 GPU 评测", va="center", fontsize=9, color="#666")
    else:
        ax.barh(y, v, color=c)
        ax.text(v + (1.5 if v >= 0 else -1.5), y, f"{v:+.0f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=10, fontweight="bold")
ax.set_yticks(ys, [i[0] for i in items], fontsize=9)
ax.axvline(0, color="#333", lw=1)
ax.set_xlim(-85, 82)
ax.set_xlabel("误差变化(%,负 = 变好)")
ax.set_title("在哪儿动手才有效:动输入都变差(红),编码器后面都变好(绿)")
ax.grid(alpha=.3, axis="x")
sec = ax.secondary_yaxis("right")
sec.set_yticks(ys, [i[3] for i in items])
sec.tick_params(length=0)
fig.tight_layout(); fig.savefig(A / "fig_intervene.png", dpi=140); plt.close(fig)

# ---------------- fig 2: frozen baselines (#37)
meta = json.load(open(ROOT / "data/bench/meta.json"))["results"]
models = list(meta.keys())
labels = {"unik3d_vitl": "UniK3D-L", "da3_small": "DA3-S", "da3_large": "DA3-L",
          "vggt_omega": "VGGT-Ω", "dav2_large": "DAv2-L"}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
x = np.arange(len(models))
for ax, key, ttl in [(axes[0], "whole_absrel", "整图误差 AbsRel"),
                     (axes[1], "near_rim_absrel", "近场边缘 AbsRel")]:
    ax.bar(x - .18, [meta[m]["seq136"][key] for m in models], .34, label="seq136", color=BLUE)
    ax.bar(x + .18, [meta[m]["decoration_seq132"][key] for m in models], .34, label="decoration_132", color="#e67e22")
    ax.set_xticks(x, [labels[m] for m in models], fontsize=9)
    ax.set_title(ttl); ax.grid(alpha=.3, axis="y")
axes[0].legend(fontsize=9)
fig.suptitle("冻结模型基准(#37,两个留出场景;注意:这两个场景近场边缘内容很少,数字天然偏小)", fontsize=11)
fig.tight_layout(); fig.savefig(A / "fig_baselines.png", dpi=140); plt.close(fig)

# ---------------- fig 3: H7 + H8 pilots
h7 = {"普通 r=8\n(基准)": 0.567,
      "门控 r=8": h7r8,
      "普通 r=4": json.load(open(ROOT / "experiments/h7-theta-gated-lora/results/h5_eval_r4.json"))["zones"][Z]["after"],
      "门控 r=4": json.load(open(ROOT / "experiments/h7-theta-gated-lora/results/h7_eval_r4.json"))["zones"][Z]["after"]}
h8z = json.load(open(ROOT / "experiments/h8-equal-area/results/probe_a_seq131.json"))["zones"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ks = list(h7)
ax.bar(range(4), [h7[k] for k in ks], color=[GREEN, RED, GREEN, RED])
ax.set_xticks(range(4), ks, fontsize=9)
for i, k in enumerate(ks):
    ax.text(i, h7[k] + .01, f"{h7[k]:.3f}", ha="center", fontsize=10, fontweight="bold")
ax.set_title("H7:视角门控(红)对普通微调(绿)毫无增益")
ax.set_ylabel("近场边缘 AbsRel(低=好)"); ax.grid(alpha=.3, axis="y")
ax = axes[1]
zn = ["near_rim(<=2m,>=38deg)", "near_center(<=2m,<=11deg)", "center(<=11deg)", "far(>=3m)"]
zl = ["近场边缘", "近场中心", "中心", "远处"]
x = np.arange(4)
ax.bar(x - .18, [h8z[z]["plain"] for z in zn], .34, label="原始鱼眼", color=GREEN)
ax.bar(x + .18, [h8z[z]["equalarea"] for z in zn], .34, label="等立体角重采样", color=RED)
ax.set_xticks(x, zl, fontsize=10); ax.legend(fontsize=9)
ax.set_title("H8:重采样让每个区域都变差"); ax.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(A / "fig_h7h8.png", dpi=140); plt.close(fig)

# ---------------- fig 4: Center-PH
cpj = json.load(open(ROOT / "experiments/bench/results/centerph_seq131_odd.json"))["zones"]
cvj = json.load(open(ROOT / "experiments/bench/results/centerph_seq131_odd_vanillacovered.json"))["zones"]
zl = ["近场边缘\n(只剩50%覆盖)", "近场中心", "中心", "远处"]
x = np.arange(4)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.bar(x - .18, [cvj[z]["absrel_covered"] for z in zn], .34, label="原始鱼眼(同像素)", color=GREEN)
ax.bar(x + .18, [cpj[z]["absrel_covered"] for z in zn], .34, label="裁剪矫正 Center-PH", color=RED)
ax.set_xticks(x, zl, fontsize=10); ax.legend(fontsize=9)
ax.set_ylabel("AbsRel(低=好)")
ax.set_title("对手最强的深度基线在自我中心近场失灵(近场中心 +62%)")
ax.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(A / "fig_centerph.png", dpi=140); plt.close(fig)
print("wrote 4 figures")
