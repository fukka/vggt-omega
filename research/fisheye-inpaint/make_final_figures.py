# Copyright (c) 2026.
"""Compose the figures and the numbers for the fisheye-inpaint document from ONE
run of finetune.eval.exp_rendered (results.json + qual/raw/*.npz).

Everything the HTML shows comes out of here, so a number in the document can be
traced to a file in the run directory and nothing is typed in by hand. The
bootstrap is re-implemented byte-for-byte (same seed, same draw order) and the
script refuses to write anything if its AbsRel intervals do not match the ones
the evaluator printed in report.txt -- a mismatch would mean the two are not
describing the same data.

Usage::

    python research/fisheye-inpaint/make_final_figures.py \\
        --run <copied runs/ev_final> --out research/fisheye-inpaint/to_human/final
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# CJK labels: fall through to whatever the machine has.
plt.rcParams["font.family"] = ["PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC",
                               "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# All display strings live here so the same run can produce a Chinese research
# log and an English report from one set of numbers -- a figure translated by
# hand is a figure that can drift from its data.
LANGS = ("zh", "en")
L = {}          # set by set_lang()

STRINGS = {
 "cells": {
   "zh": {"fisheye_masked": ("①", "RAW · BLACK", "原始鱼眼 · 黑角"),
          "persp_masked":   ("②", "RECT · BLACK", "矫正透视 · 黑楔形"),
          "fisheye_full":   ("③", "RAW · FILLED", "原始鱼眼 · 真值补全"),
          "persp_full":     ("④", "RECT · FILLED", "矫正透视 · 真值补全"),
          "persp_crop":     ("⑤", "RECT · CROP", "内接矫正 · 天然无黑区"),
          "persp_crop_lores": ("⑤ᵇ", "CROP · BLURRED", "内接矫正,降到 ④ 的采样率")},
   "en": {"fisheye_masked": ("1", "RAW · BLACK", "raw fisheye, black corners"),
          "persp_masked":   ("2", "RECT · BLACK", "rectified, black wedges"),
          "fisheye_full":   ("3", "RAW · FILLED", "raw fisheye, true corners"),
          "persp_full":     ("4", "RECT · FILLED", "rectified, true wedges"),
          "persp_crop":     ("5", "RECT · CROP", "inscribed crop, no black"),
          "persp_crop_lores": ("5b", "CROP · BLURRED", "crop, bandlimited to 4")},
 },
 "zh": {
   "input": "模型输入", "pred": "预测深度", "errmap": "AbsRel 图", "gt": "真值深度(渲染 Z-pass)",
   "single": "单帧", "multi": "8 帧", "cbar": "AbsRel(0–0.3 截断)",
   "panel_title": "窗口 {wi:02d} · {seq}/{fr} · 五格同一帧;评分区以外为灰;①③ 同区、②④ 同区、⑤ 全幅(各自区域)",
   "black_frac": "纯黑像素 {blk:.2f}% · 评分区 {gr:.1f}%",
   "traj_gt": "真值", "along": "沿路径 (m)", "across": "横向 (m)",
   "traj_win": "窗口 {i:02d} · {seq} · 起始 {fr}",
   "traj_title": "8 帧窗口相机轨迹 · 预测经 Sim(3) 对齐到真值后画在该窗口的主轴坐标系里(米)· ★ = 第一帧 · 注意横纵尺度不同",
   "fov_title": "相机头推断的视场角 · {lab}", "fov_y": "视场角 (deg)",
   "fov_true": "真值", "fov_pred": "模型读数", "fov_err": "差 {e:.1f}°",
   "eff_x": ["鱼眼\n③−①", "透视\n④−②", "交互项\n(④−②)−(③−①)"],
   "eff_titles": ["Δ AbsRel · 单帧(真值 − 黑;负 = 真值更好)", "Δ AbsRel · 8 帧",
                  "Δ AUC@30 · 8 帧(正 = 真值更好)", "Δ ATE (m) · 8 帧(负 = 真值更好)"],
   "eff_sup": "每个点 = 一个窗口的配对差;方块 = 均值,须 = 按窗口聚类的 95% bootstrap CI;* = CI 不含零",
   "crop_titles": ["Δ AbsRel · 单帧 · 各自区域", "Δ AbsRel · 单帧 · 公共区域(⑤ 的)",
                   "Δ AbsRel · 8 帧 · 各自区域", "Δ AbsRel · 8 帧 · 公共区域",
                   "Δ AUC@30 · 8 帧(正 = ⑤ 更好)"],
   "crop_sup": "⑤ 内接裁剪 减 其它格子:点 = 一个窗口,方块 = 均值,须 = 按窗口聚类 95% CI,* = 不含零;深度负 = ⑤ 更好",
   "f4_pairs": ["④ 补真值 124.7°", "⑤ᵇ 降采样 106.9°", "⑤ 裁剪 106.9°"],
   "f4_t0": "AbsRel · 最小公共区域\n(空心 = 单帧,实心 = 8 帧)",
   "f4_t1": "Δ AbsRel · 公共区域(负 = ⑤ 更好)",
   "f4_x1": ["⑤−④\n单帧", "⑤−⑤ᵇ\n单帧", "⑤−④\n8 帧", "⑤−⑤ᵇ\n8 帧"],
   "f4_t2": "相机位姿 AUC@30 · 8 帧(越高越好)",
   "f4_t3": "多帧收益(单帧 − 8 帧 AbsRel;正 = 多帧有用)",
   "f4_sup": "⑤ 内接裁剪 vs ④ 补满真值的宽视场 · ⑤ᵇ = ⑤ 降到 ④ 的采样率(把「视场」与「清晰度」拆开)· * = 95% CI 不含零",
   "f4_sup_nc": "⑤ 内接裁剪 vs ④ 补满真值的宽视场 · 深度在两者共有的场景方向上打分 · * = 95% CI 不含零",
 },
 "en": {
   "input": "input to the model", "pred": "predicted depth", "errmap": "relative error",
   "gt": "ground truth (rendered)",
   "single": "1 frame", "multi": "8 frames", "cbar": "AbsRel (clipped at 0.3)",
   "panel_title": "Window {wi:02d} · {seq}/{fr} — one frame, all five inputs. Grey = not scored.",
   "black_frac": "pure black {blk:.2f}% · scored {gr:.1f}%",
   "traj_gt": "ground truth", "along": "along path (m)", "across": "across path (m)",
   "traj_win": "Window {i:02d} · {seq} · from {fr}",
   "traj_title": "Camera path of each 8-frame window, predictions Sim(3)-aligned to truth and drawn in that window's own axes (metres). Star = first frame. Note the axes are not equally scaled.",
   "fov_title": "Field of view the model infers · {lab}", "fov_y": "field of view (deg)",
   "fov_true": "true", "fov_pred": "model", "fov_err": "off by {e:.1f}°",
   "eff_x": ["fisheye\n3 − 1", "rectified\n4 − 2", "interaction\n(4−2) − (3−1)"],
   "eff_titles": ["AbsRel change, 1 frame\n(true content minus black; lower is better)",
                  "AbsRel change, 8 frames",
                  "AUC@30 change, 8 frames\n(higher is better)",
                  "ATE change (m), 8 frames\n(lower is better)"],
   "eff_sup": "Each dot is one window. Square = mean over 96 frames, whisker = 95% bootstrap CI resampling windows. * = interval excludes zero.",
   "crop_titles": ["AbsRel, 1 frame, own regions", "AbsRel, 1 frame, shared region",
                   "AbsRel, 8 frames, own regions", "AbsRel, 8 frames, shared region",
                   "AUC@30, 8 frames (higher = 5 better)"],
   "crop_sup": "Input 5 (inscribed crop) minus each other input. Dot = one window, square = mean, whisker = 95% CI over windows. Negative depth = 5 is better.",
   "f4_pairs": ["4  wide + filled\n124.7°", "5b  crop, blurred\n106.9°", "5  crop\n106.9°"],
   "f4_t0": "AbsRel on the shared region\n(hollow = 1 frame, solid = 8 frames)",
   "f4_t1": "AbsRel difference on the shared region\n(negative = 5 is better)",
   "f4_x1": ["5 − 4\n1 frame", "5 − 5b\n1 frame", "5 − 4\n8 frames", "5 − 5b\n8 frames"],
   "f4_t2": "Camera pose AUC@30, 8 frames\n(higher is better)",
   "f4_t3": "Gain from using 8 frames instead of 1\n(positive = multi-frame helps)",
   "f4_sup": "Input 5 (crop) against input 4 (wide, wedges filled with true content). 5b is 5 blurred to 4's sampling rate, which separates 'field of view' from 'sharpness'. * = 95% CI excludes zero.",
   "f4_sup_nc": "Input 5 (crop) against input 4 (wide, wedges filled from the Blender render). Depth is scored on the directions both can see. * = 95% CI excludes zero.",
 },
}


def set_lang(lang, drop_control=False):
    """Point the module's labels at one language, once, before any figure.

    ``drop_control`` leaves the sampling-matched arm out of the figures. It is
    a control, not a cell of the design: once it has shown that sharpness does
    not explain the crop's win, a document aimed at a reader rather than at the
    record is clearer without it.
    """
    global L, CELLS, CTRL, LABEL, CROP_VS, SHOW_CTRL
    assert lang in LANGS
    L = STRINGS[lang]
    c = STRINGS["cells"][lang]
    CELLS = OrderedDict((k, c[k]) for k in
                        ("fisheye_masked", "persp_masked", "fisheye_full", "persp_full", "persp_crop"))
    CTRL = ("persp_crop_lores", c["persp_crop_lores"])
    LABEL = dict(CELLS, **{CTRL[0]: CTRL[1]})
    SHOW_CTRL = not drop_control
    CROP_VS = tuple(k for k in ("persp_masked", "persp_full", "fisheye_masked", "fisheye_full",
                                "persp_crop_lores")
                    if SHOW_CTRL or k != CTRL[0])
    if lang == "en":
        plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]


CELLS = OrderedDict([
    ("fisheye_masked", ("①", "RAW · BLACK", "原始鱼眼 · 黑角")),
    ("persp_masked",   ("②", "RECT · BLACK", "矫正透视 · 黑楔形")),
    ("fisheye_full",   ("③", "RAW · FILLED", "原始鱼眼 · 真值补全")),
    ("persp_full",     ("④", "RECT · FILLED", "矫正透视 · 真值补全")),
    ("persp_crop",     ("⑤", "RECT · CROP", "内接矫正 · 天然无黑区")),
])
# The sampling-matched control is not a cell of the design: it exists only to
# split the crop-vs-filled contrast into "field of view" and "sharpness".
CTRL = ("persp_crop_lores", ("⑤ᵇ", "CROP · BLURRED", "内接矫正,降到 ④ 的采样率"))
COLORS = {"fisheye_masked": "#98362f", "persp_masked": "#a86a15",
          "fisheye_full": "#33704a", "persp_full": "#0b6b6e", "persp_crop": "#5b4a8a",
          "persp_crop_lores": "#9a86c4"}
CROP_VS = ("persp_masked", "persp_full", "fisheye_masked", "fisheye_full", "persp_crop_lores")
LABEL = dict(CELLS, **{CTRL[0]: CTRL[1]})
POSE_KEYS = ("rot_err_deg", "trans_err_deg", "rra15", "rta15", "auc30", "ate_m", "sim3_scale")


# --------------------------------------------------------------------------- #
# statistics -- identical to finetune.eval.exp_rendered.cluster_bootstrap
# --------------------------------------------------------------------------- #
def cluster_bootstrap(a, b, group_of, n_boot=10000, seed=0):
    keys = sorted(set(a) & set(b) & set(group_of))
    if len(keys) < 3:
        return None
    by_group = OrderedDict()
    for k in keys:
        by_group.setdefault(group_of[k], []).append(a[k] - b[k])
    gkeys = list(by_group)
    if len(gkeys) < 2:
        return None
    vals = [np.asarray(by_group[g], dtype=float) for g in gkeys]
    d_all = np.concatenate(vals)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(gkeys), size=(n_boot, len(gkeys)))
    boots = np.array([np.concatenate([vals[j] for j in row]).mean() for row in draws])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n_pairs": int(len(d_all)), "n_groups": len(gkeys), "mean": float(d_all.mean()),
            "ci_lo": float(lo), "ci_hi": float(hi), "excludes_zero": bool(lo > 0 or hi < 0)}


def paired_effects(res, metric, unit="frame"):
    """full - masked per projection, plus the interaction, window-clustered.

    unit='frame': per-frame metric from _per_frame_metrics, clustered by window.
    unit='window': per-window metric from _per_window (pose), one value per window.
    """
    def vals(st):
        r = res.get(st) or {}
        if unit == "frame":
            return {k: v[metric] for k, v in (r.get("_per_frame_metrics") or {}).items()}, r.get("_group_of") or {}
        return {k: v[metric] for k, v in (r.get("_per_window") or {}).items()}, r.get("_window_group") or {}
    out = {}
    diffs = {}
    for proj in ("fisheye", "persp"):
        a, ga = vals(f"{proj}_full")
        b, _ = vals(f"{proj}_masked")
        cb = cluster_bootstrap(a, b, ga)
        if cb is None:
            return out
        out[proj] = cb
        diffs[proj] = {k: a[k] - b[k] for k in set(a) & set(b)}
        out[proj]["per_group"] = _per_group_means(diffs[proj], ga)
    common = sorted(set(diffs["persp"]) & set(diffs["fisheye"]))
    dd = {k: diffs["persp"][k] - diffs["fisheye"][k] for k in common}
    _, gp = vals("persp_full")
    out["interaction"] = cluster_bootstrap(dd, {k: 0.0 for k in dd}, gp)
    out["interaction"]["per_group"] = _per_group_means(dd, gp)
    return out


def crop_effects(res, metric, unit="frame"):
    """persp_crop - other, per frame (clustered by window) or per window (pose)."""
    def vals(st):
        r = res.get(st) or {}
        if unit == "frame":
            return {k: v[metric] for k, v in (r.get("_per_frame_metrics") or {}).items()}, r.get("_group_of") or {}
        return {k: v[metric] for k, v in (r.get("_per_window") or {}).items()}, r.get("_window_group") or {}
    a, ga = vals("persp_crop")
    out = {}
    for st in CROP_VS:
        b, _ = vals(st)
        cb = cluster_bootstrap(a, b, ga)
        if cb is None:
            continue
        cb["per_group"] = _per_group_means({k: a[k] - b[k] for k in set(a) & set(b)}, ga)
        out[st] = cb
    return out


def _per_group_means(d, group_of):
    g = OrderedDict()
    for k in sorted(d):
        g.setdefault(group_of.get(k, k), []).append(d[k])
    return {k: float(np.mean(v)) for k, v in g.items()}


def check_against_report(results, report_txt):
    """The re-implemented bootstrap must reproduce the evaluator's AbsRel lines."""
    pat = re.compile(r"^\[(\S+)\] true content vs black.*?(?=^\[|\Z)", re.S | re.M)
    line = re.compile(r"^\s+(fisheye|persp|INTERACTION)\s+([+-]\d\.\d{4})\s+CI\(win\) \[([+-]\d\.\d{4}), ([+-]\d\.\d{4})\]", re.M)
    n = 0
    for blk in pat.finditer(report_txt):
        mode = blk.group(1)
        eff = paired_effects(results[mode], "AbsRel")
        for m in line.finditer(blk.group(0)):
            key = {"fisheye": "fisheye", "persp": "persp", "INTERACTION": "interaction"}[m.group(1)]
            got = (eff[key]["mean"], eff[key]["ci_lo"], eff[key]["ci_hi"])
            want = tuple(float(x) for x in m.group(2, 3, 4))
            if any(abs(g - w) > 5e-5 for g, w in zip(got, want)):
                raise SystemExit(f"bootstrap mismatch vs report.txt [{mode}] {key}: {got} vs {want}")
            n += 1
    # The crop-vs-others lines, if present.
    blk = re.compile(r"^\[(\S+)\] inscribed crop.*?(?=^\[|\Z)", re.S | re.M)
    line2 = re.compile(r"^\s+vs (\S+)\s+([+-]\d\.\d{4})\s+CI\(win\) \[([+-]\d\.\d{4}), ([+-]\d\.\d{4})\]", re.M)
    for b in blk.finditer(report_txt):
        mode = b.group(1)
        ce = crop_effects(results[mode], "AbsRel")
        for m in line2.finditer(b.group(0)):
            got = (ce[m.group(1)]["mean"], ce[m.group(1)]["ci_lo"], ce[m.group(1)]["ci_hi"])
            want = tuple(float(x) for x in m.group(2, 3, 4))
            if any(abs(g - w) > 5e-5 for g, w in zip(got, want)):
                raise SystemExit(f"crop bootstrap mismatch vs report.txt [{mode}] {m.group(1)}: {got} vs {want}")
            n += 1
    if n < 6:
        raise SystemExit(f"only {n} report lines checked -- report.txt format changed?")
    return n


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _load_raw(raw_dir, setting, sl, wi):
    f = os.path.join(raw_dir, f"{setting}_s{sl}_w{wi:02d}.npz")
    if not os.path.isfile(f):
        return None
    z = np.load(f)
    return {k: z[k] for k in z.files}


def fig_panels(raw_dir, res_by_mode, wi, out_img):
    """One window's first frame: for each cell, input | pred s1 | err s1 | pred s8 | err s8 | GT."""
    rows = list(CELLS)
    fig, axes = plt.subplots(len(rows), 6, figsize=(18, 3.15 * len(rows)), constrained_layout=True)
    col_titles = [L["input"], f'{L["pred"]} · {L["single"]}', f'{L["errmap"]} · {L["single"]}',
                  f'{L["pred"]} · {L["multi"]}', f'{L["errmap"]} · {L["multi"]}', L["gt"]]
    fdir = None
    for r, st in enumerate(rows):
        # At seq_len=1 every "window" is one frame, so the single-frame dump of
        # 8-frame window wi is frame 8*wi; assert on the frame dir, not the index.
        z8 = _load_raw(raw_dir, st, 8, wi)
        z1 = _load_raw(raw_dir, st, 1, 8 * wi)
        if z1 is None or z8 is None:
            for a in axes[r]:
                a.axis("off")
            continue
        assert str(z1["frame_dir"]) == str(z8["frame_dir"])
        fdir = str(z1["frame_dir"])
        gt, m = z1["gt"].astype(np.float32), z1["mask"].astype(bool)
        vmin, vmax = np.percentile(gt[m], [2, 98])
        cid, cname, zh = LABEL[st]
        ax = axes[r]
        ax[0].imshow(z1["rgb"])
        ax[0].set_ylabel(f"{cid} {cname}\n{zh}", fontsize=11, fontweight="bold")
        for c, (z, sl, lab) in enumerate([(z1, 1, L["single"]), (z8, 8, L["multi"])]):
            pred = z["pred"].astype(np.float32)
            pm = res_by_mode["single" if sl == 1 else "8-frame"][st]["_per_frame_metrics"][fdir]
            ax[1 + 2 * c].imshow(np.where(m, pred, np.nan), vmin=vmin, vmax=vmax, cmap="turbo")
            err = np.where(m, np.abs(pred - gt) / np.maximum(gt, 1e-6), np.nan)
            im = ax[2 + 2 * c].imshow(err, vmin=0, vmax=0.3, cmap="magma")
            ax[2 + 2 * c].set_xlabel(f"AbsRel {pm['AbsRel']:.4f} · δ₁ {pm['delta1']:.3f}", fontsize=10.5)
        ax[5].imshow(np.where(m, gt, np.nan), vmin=vmin, vmax=vmax, cmap="turbo")
        for a in ax:
            a.set_xticks([]); a.set_yticks([]); a.set_facecolor("#c9d2d1")
        if r == 0:
            for a, t in zip(ax, col_titles):
                a.set_title(t, fontsize=11.5, loc="left")
    fig.colorbar(im, ax=axes[:, 4].tolist(), fraction=0.03, pad=0.01, shrink=.6, label=L["cbar"])
    seq = os.path.basename(os.path.dirname(fdir)).replace("Apartment_release_", "")
    fig.suptitle(L["panel_title"].format(wi=wi, seq=seq, fr=os.path.basename(fdir)),
                 fontsize=13, x=0.005, ha="left")
    fig.savefig(out_img, dpi=62, facecolor="white", pil_kwargs={"quality": 86, "optimize": True})
    plt.close(fig)
    return fdir


def fig_inputs(raw_dir, wi, out_png):
    show = list(CELLS) + ([CTRL[0]] if SHOW_CTRL else [])
    fig, axes = plt.subplots(1, len(show), figsize=(3.6 * len(show), 4.4))
    for a, st in zip(axes, show):
        z = _load_raw(raw_dir, st, 1, wi)
        if z is None:
            a.axis("off"); continue
        a.imshow(z["rgb"]); cid, cname, zh = LABEL[st]
        blk = float((z["rgb"].max(-1) == 0).mean()) * 100
        a.set_title(f"{cid} {cname}\n{zh}\n" + L["black_frac"].format(blk=blk, gr=z["mask"].mean() * 100),
                    fontsize=9.5)
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=72, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 88, "optimize": True})
    plt.close(fig)


def _centres(E):
    E = np.asarray(E, float)
    return -np.einsum("nji,nj->ni", E[:, :, :3], E[:, :, 3])


def _umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    H = (sc.T @ dc) / len(src)
    U, D, Vt = np.linalg.svd(H)
    d = np.ones(3); d[-1] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag(d) @ U.T
    s = float(np.dot(D, d)) / max(float(np.mean(np.sum(sc ** 2, 1))), 1e-12)
    return s, R, (s * (R @ sc.T).T) + mu_d


def fig_trajectories(res8, out_png, max_windows=12):
    """Each window in its own frame: GT centres rotated so x runs along the path.

    A 2 m walk with centimetre lateral wander is a flat line in world axes; the
    PCA frame spreads it out. Predictions are Sim(3)-aligned to GT first, then
    put through the same rotation, so a deviation from the black line IS the
    error the ATE measures.
    """
    wkeys = list((res8["persp_full"]["_windows"]).keys())[:max_windows]
    n = len(wkeys)
    ncol, nrow = 3, int(np.ceil(n / 3))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 4.3 * nrow), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for i, wk in enumerate(wkeys):
        ax = axes[i // ncol, i % ncol]
        gt = _centres(res8["persp_full"]["_windows"][wk]["gt_E"])
        mu = gt.mean(0)
        _u, _s, vt = np.linalg.svd(gt - mu)
        P = vt[:2]                      # along-path, cross-path
        g2 = (gt - mu) @ P.T
        ax.plot(g2[:, 0], g2[:, 1], "k-o", lw=2.4, ms=5.5, label=L["traj_gt"], zorder=5)
        ax.plot(g2[0, 0], g2[0, 1], "k*", ms=14, zorder=6)
        for st in CELLS:
            w = res8[st]["_windows"].get(wk)
            if not w or "pred_E" not in w:
                continue
            p = _centres(w["pred_E"])
            if np.var(p, 0).sum() < 1e-12:
                continue
            _, _, al = _umeyama(p, gt)
            a2 = (al - mu) @ P.T
            cid = LABEL[st][0]
            ax.plot(a2[:, 0], a2[:, 1], "-o", color=COLORS[st], lw=1.4, ms=3.8, alpha=.9,
                    label=f"{cid}  AUC@30 {w['auc30']:.2f} · ATE {w['ate_m']*100:.0f} cm · rot {w['rot_err_deg']:.1f}°")
        seq = wk.split("/")[-2].replace("Apartment_release_", "")
        ax.set_title(L["traj_win"].format(i=i, seq=seq, fr=wk.split("/")[-1]), fontsize=10)
        ax.set_xlabel(L["along"], fontsize=9); ax.set_ylabel(L["across"], fontsize=9)
        ax.grid(alpha=.3); ax.tick_params(labelsize=8)
        ax.legend(fontsize=7.5, loc="best", framealpha=.9)
    for j in range(n, nrow * ncol):
        axes[j // ncol, j % ncol].axis("off")
    fig.suptitle(L["traj_title"], fontsize=12, x=0.005, ha="left")
    fig.savefig(out_png, dpi=80, facecolor="white")
    plt.close(fig)


def fig_fov(res1, res8, out_png):
    """Each pinhole input against ITS OWN true field of view.

    Only the three pinhole arms appear. A fisheye frame has no pinhole field of
    view to be right about, so putting it on the same axis invites the reader to
    compare it against a truth line that is not its own. Inputs 2 and 4 happen
    to share a truth (124.7 deg); input 5's is different (106.8 deg), and that
    is the whole point -- so each input carries its own truth marker beside it
    rather than the figure carrying one line for everybody.

    The model emits two numbers (VGGT calls them fov_h and fov_w); the frames
    are square with fx == fy, so both truths coincide and the two predictions
    agree within a degree. The vertical one is plotted.
    """
    arms = [a for a in ("persp_masked", "persp_full", "persp_crop") if a in res1]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (res, lab) in zip(axes, [(res1, L["single"]), (res8, L["multi"])]):
        for k_, st in enumerate(arms):
            i = k_ * 1.5
            r = res[st]
            f = np.array([v[0] for v in r["_per_frame_fov"].values()])
            gt = r["gt_fov_h_deg"]
            ax.scatter(i + 0.12 + rng.uniform(-.1, .1, len(f)), f, s=13, color=COLORS[st], alpha=.55)
            ax.plot([i + .02, i + .22], [f.mean()] * 2, color=COLORS[st], lw=2.6, zorder=5)
            ax.text(i + .26, f.mean(), f'{L["fov_pred"]} {f.mean():.1f}°', fontsize=9,
                    color=COLORS[st], va="center", fontweight="bold")
            ax.plot([i - .30, i - .04], [gt] * 2, color="#1d1a24", lw=3, zorder=6)
            ax.text(i - .33, gt, f'{L["fov_true"]} {gt:.1f}°', fontsize=9, color="#1d1a24",
                    ha="right", va="center")
            ax.annotate("", xy=(i + .12, gt), xytext=(i + .12, f.mean()),
                        arrowprops=dict(arrowstyle="<->", color=COLORS[st], lw=1.1, alpha=.75))
            ax.text(i + .17, (gt + f.mean()) / 2, L["fov_err"].format(e=abs(gt - f.mean())),
                    fontsize=8.5, color=COLORS[st], va="center")
        ax.set_xticks([k_ * 1.5 for k_ in range(len(arms))])
        ax.set_xticklabels([f"{LABEL[s_][0]} {LABEL[s_][1]}" for s_ in arms], fontsize=9.5)
        ax.set_xlim(-.9, (len(arms) - 1) * 1.5 + .95)
        ax.set_title(L["fov_title"].format(lab=lab), fontsize=11, loc="left")
        ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel(L["fov_y"])
    fig.tight_layout()
    fig.savefig(out_png, dpi=80, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_effects(effects, out_png):
    """Per-window paired deltas (dots) with the clustered mean ± CI (bar), depth and pose."""
    panels = list(zip(("AbsRel", "AbsRel", "auc30", "ate_m"),
                      ("single", "8-frame", "8-frame", "8-frame"), L["eff_titles"]))
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (metric, mode, title) in zip(axes, panels):
        e = effects[mode][metric]
        for i, key in enumerate(("fisheye", "persp", "interaction")):
            if key not in e:
                continue
            cb = e[key]
            pg = np.array(list(cb["per_group"].values()))
            col = {"fisheye": "#33704a", "persp": "#0b6b6e", "interaction": "#5f7376"}[key]
            ax.scatter(np.full(len(pg), i) + np.linspace(-.15, .15, len(pg)), pg, s=16, color=col, alpha=.55)
            ax.errorbar([i], [cb["mean"]], yerr=[[cb["mean"] - cb["ci_lo"]], [cb["ci_hi"] - cb["mean"]]],
                        fmt="s", color=col, ms=7, capsize=5, lw=2, zorder=5)
            fmt = "{:+.3f}" if metric in ("auc30", "ate_m") else "{:+.4f}"
            ax.text(i + 0.2, cb["mean"], fmt.format(cb["mean"]) + ("*" if cb["excludes_zero"] else ""),
                    ha="left", va="center", fontsize=9, color=col, fontweight="bold")
        ax.axhline(0, color="k", lw=.8)
        ax.set_xlim(-0.5, 2.85)
        ax.set_xticks(range(3)); ax.set_xticklabels(L["eff_x"], fontsize=9)
        ax.set_title(title, fontsize=10.5, loc="left"); ax.grid(axis="y", alpha=.3)
    fig.suptitle(L["eff_sup"], fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=80, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_crop_effects(numbers, out_png):
    """persp_crop minus each other arm, per window, own vs common region, depth and pose."""
    panels = list(zip(("crop_effects_own", "crop_effects_common", "crop_effects_own",
                       "crop_effects_common", "crop_effects_own"),
                      ("single", "single", "8-frame", "8-frame", "8-frame"),
                      ("AbsRel", "AbsRel", "AbsRel", "AbsRel", "auc30"), L["crop_titles"]))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.3))
    for ax, (key, mode, metric, title) in zip(axes, panels):
        e = numbers[key][mode][metric]
        for i, st in enumerate(CROP_VS):
            if st not in e:
                continue
            cb = e[st]
            col = COLORS[st]
            pg = np.array(list(cb["per_group"].values()))
            ax.scatter(np.full(len(pg), i) + np.linspace(-.15, .15, len(pg)), pg, s=14, color=col, alpha=.5)
            ax.errorbar([i], [cb["mean"]], yerr=[[cb["mean"] - cb["ci_lo"]], [cb["ci_hi"] - cb["mean"]]],
                        fmt="s", color=col, ms=6.5, capsize=4, lw=2, zorder=5)
            fmt = "{:+.3f}" if metric == "auc30" else "{:+.4f}"
            ax.text(i + 0.18, cb["mean"], fmt.format(cb["mean"]) + ("*" if cb["excludes_zero"] else ""),
                    ha="left", va="center", fontsize=8.5, color=col, fontweight="bold")
        ax.axhline(0, color="k", lw=.8); ax.set_xlim(-0.5, len(CROP_VS) - 0.2)
        ax.set_xticks(range(len(CROP_VS))); ax.set_xticklabels([f"{LABEL['persp_crop'][0]}−{LABEL[s][0]}" for s in CROP_VS], fontsize=9)
        ax.set_title(title, fontsize=10, loc="left"); ax.grid(axis="y", alpha=.3)
    fig.suptitle(L["crop_sup"], fontsize=11, x=0.005, ha="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=80, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_five_vs_four(numbers, out_png):
    """The question the whole cell-5 arm exists for, in one figure."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
    keep = [0, 1, 2] if SHOW_CTRL else [0, 2]
    pairs = [list(zip(("persp_full", "persp_crop_lores", "persp_crop"), L["f4_pairs"]))[i] for i in keep]

    ax = axes[0]                                   # depth on the common region
    for j, (mode, mk) in enumerate((("single", "o"), ("8-frame", "s"))):
        cc = numbers["cells_common"][mode]
        for i, (st, _lab) in enumerate(pairs):
            ax.plot([i], [cc[st]["AbsRel"]], mk, color=COLORS[st], ms=11,
                    mfc=COLORS[st] if j else "white", mew=2)
    ax.set_xticks(range(len(pairs))); ax.set_xticklabels([p[1] for p in pairs], fontsize=9)
    ax.set_xlim(-.5, len(pairs) - .5); ax.margins(y=.16)
    ax.set_title(L["f4_t0"], fontsize=10.5, loc="left")
    ax.set_ylabel("AbsRel ↓"); ax.grid(axis="y", alpha=.3)

    ax = axes[1]                                   # the two crop contrasts
    for i, (key, lab) in enumerate((("AbsRel", "单帧"), ("AbsRel", "8 帧"))):
        pass
    for i, (X, lab) in enumerate(((numbers["crop_effects_common"]["single"], "单帧"),
                                  (numbers["crop_effects_common"]["8-frame"], "8 帧"))):
        for j, st in enumerate(("persp_full", "persp_crop_lores") if SHOW_CTRL else ("persp_full",)):
            if st not in X["AbsRel"]:
                continue
            cb = X["AbsRel"][st]
            x = i * 2 + j * 0.7
            pg = np.array(list(cb["per_group"].values()))
            ax.scatter(np.full(len(pg), x) + np.linspace(-.1, .1, len(pg)), pg, s=13,
                       color=COLORS[st], alpha=.5)
            ax.errorbar([x], [cb["mean"]], yerr=[[cb["mean"] - cb["ci_lo"]], [cb["ci_hi"] - cb["mean"]]],
                        fmt="s", color=COLORS[st], ms=7, capsize=4, lw=2, zorder=5)
            ax.text(x + .12, cb["mean"], f"{cb['mean']:+.4f}" + ("*" if cb["excludes_zero"] else ""),
                    fontsize=8.5, color=COLORS[st], fontweight="bold", va="center")
    ax.axhline(0, color="k", lw=.8); ax.set_xlim(-.4, 3.6)
    if SHOW_CTRL:
        ax.set_xticks([0, .7, 2, 2.7]); ax.set_xticklabels(L["f4_x1"], fontsize=8.5)
    else:
        ax.set_xticks([0, 2]); ax.set_xticklabels([L["f4_x1"][0], L["f4_x1"][2]], fontsize=9)
    ax.set_title(L["f4_t1"], fontsize=10.5, loc="left"); ax.grid(axis="y", alpha=.3)

    ax = axes[2]                                   # pose
    P = numbers["pose"]["8-frame"]
    for i, (st, lab) in enumerate(pairs):
        if st not in P:
            continue
        ax.bar(i, P[st]["auc30"], color=COLORS[st], width=.6)
        ax.text(i, P[st]["auc30"] + .012, f"{P[st]['auc30']:.3f}\nATE {P[st]['ate_m']*100:.1f}cm",
                ha="center", fontsize=9)
    ax.set_xticks(range(len(pairs))); ax.set_xticklabels([p[1] for p in pairs], fontsize=9)
    ax.set_ylim(0, 1.18); ax.set_title(L["f4_t2"], fontsize=10.5, loc="left")
    ax.grid(axis="y", alpha=.3)

    ax = axes[3]                                   # multi-frame gain per setting
    mv = numbers["multi_vs_single"]
    for i, (st, lab) in enumerate(pairs):
        if st not in mv:
            continue
        cb = mv[st]
        g, lo, hi = -cb["mean"], -cb["ci_hi"], -cb["ci_lo"]     # sign: + = multi helps
        ax.errorbar([i], [g], yerr=[[g - lo], [hi - g]], fmt="s", color=COLORS[st], ms=9, capsize=5, lw=2)
        ax.text(i + .12, g, f"{g:+.4f}" + ("*" if cb["excludes_zero"] else ""), fontsize=9,
                color=COLORS[st], fontweight="bold", va="center")
    ax.axhline(0, color="k", lw=.8); ax.set_xlim(-.4, len(pairs) - .1 + .4)
    ax.set_xticks(range(len(pairs))); ax.set_xticklabels([p[1] for p in pairs], fontsize=9)
    ax.set_title(L["f4_t3"], fontsize=10.5, loc="left")
    ax.grid(axis="y", alpha=.3)
    fig.suptitle(L["f4_sup"] if SHOW_CTRL else L["f4_sup_nc"], fontsize=11, x=0.005, ha="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=80, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="the --region own run (all settings)")
    ap.add_argument("--run-crop", required=True, help="the --region crop run (five arms)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="zh", choices=LANGS)
    ap.add_argument("--drop-control", action="store_true",
                    help="leave persp_crop_lores out of the figures (it is a control, not a cell)")
    args = ap.parse_args()
    set_lang(args.lang, args.drop_control)
    results = json.load(open(os.path.join(args.run, "results.json")))
    report = open(os.path.join(args.run, "report.txt")).read()
    n = check_against_report(results, report)
    results_c = json.load(open(os.path.join(args.run_crop, "results.json")))
    report_c = open(os.path.join(args.run_crop, "report.txt")).read()
    n += check_against_report(results_c, report_c)
    print(f"[figures] bootstrap reproduces {n} report lines across both regimes")
    # The two regimes must be the same frames, the same model, the same
    # predictions: pose (region-independent) must agree to the bit.
    for mode in results_c:
        for st in results_c[mode]:
            if "_per_window" in results_c[mode][st]:
                for wk, v in results_c[mode][st]["_per_window"].items():
                    assert abs(v["auc30"] - results[mode][st]["_per_window"][wk]["auc30"]) < 1e-12, (mode, st, wk)
    print("[figures] pose identical across regimes -- same predictions, only the grading differs")
    os.makedirs(args.out, exist_ok=True)
    raw = os.path.join(args.run, "qual", "raw")
    res1, res8 = results["single"], results["8-frame"]

    # ---- numbers
    numbers = {"cells": {}, "effects": {}, "pose": {}, "fov": {}, "ladder": {}, "windows": []}
    for mode, res in results.items():
        numbers["cells"][mode] = {st: {k: res[st][k] for k in ("AbsRel", "RMSE", "delta1", "n_frames", "n_groups", "fov_h_deg", "fov_h_std")}
                                  for st in res}
        numbers["effects"][mode] = {m: paired_effects(res, m) for m in ("AbsRel", "delta1", "RMSE")}
        if mode != "single":
            for m in ("auc30", "rot_err_deg", "trans_err_deg", "ate_m", "rra15", "rta15"):
                numbers["effects"][mode][m] = paired_effects(res, m, unit="window")
            numbers["pose"][mode] = {st: {k: res[st][k] for k in POSE_KEYS} | {"n_windows": res[st]["n_windows"]}
                                     for st in res if "auc30" in res[st]}
        numbers["fov"][mode] = {st: {"mean": res[st]["fov_h_deg"], "std": res[st]["fov_h_std"],
                                     "gt": res[st].get("gt_fov_h_deg"), "abs_err": res[st].get("fov_h_abs_err_deg")}
                                for st in res}
        # fill ladder: gain over black as % of oracle, clustered CI
        lad = {}
        for proj in ("fisheye", "persp"):
            blk = {k: v["AbsRel"] for k, v in res[f"{proj}_masked"]["_per_frame_metrics"].items()}
            orc = {k: v["AbsRel"] for k, v in res[f"{proj}_full"]["_per_frame_metrics"].items()}
            go = res[f"{proj}_masked"]["_group_of"]
            span = float(np.mean([blk[k] - orc[k] for k in sorted(set(blk) & set(orc))]))
            rows = {}
            for st in sorted(s for s in res if s.startswith(f"{proj}_fill_")) + [f"{proj}_full"]:
                v = {k: vv["AbsRel"] for k, vv in res[st]["_per_frame_metrics"].items()}
                gain = {k: blk[k] - v[k] for k in sorted(set(v) & set(blk))}
                cb = cluster_bootstrap(gain, {k: 0.0 for k in gain}, go)
                name = "ORACLE" if st.endswith("_full") else st.split("_fill_")[1]
                rows[name] = {"AbsRel": float(np.mean(list(v.values()))), "gain": float(np.mean(list(gain.values()))),
                              "pct": 100 * float(np.mean(list(gain.values()))) / span, "ci_lo": cb["ci_lo"], "ci_hi": cb["ci_hi"]}
            rows["black"] = {"AbsRel": float(np.mean(list(blk.values()))), "gain": 0.0, "pct": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
            lad[proj] = {"span": span, "rows": rows}
        numbers["ladder"][mode] = lad
    # ---- does multi-frame pay off differently in one arm than another? The
    # per-setting gain answers "does it help here"; the paired CONTRAST answers
    # the question actually asked -- whether more content in the input makes the
    # multi-frame mode worth more -- and it is paired on the same windows, so
    # the between-window variance that dominates the per-setting numbers drops
    # out.
    def mf_gain(res1_, res8_, st):
        a = {k: v["AbsRel"] for k, v in res1_[st]["_per_frame_metrics"].items()}
        b = {k: v["AbsRel"] for k, v in res8_[st]["_per_frame_metrics"].items()}
        return {k: a[k] - b[k] for k in set(a) & set(b)}          # + = multi helps
    numbers["multi_contrast"] = {}
    for tag, (ra, rb) in (("own", (res1, res8)), ("common", (results_c["single"], results_c["8-frame"]))):
        go = rb["persp_crop"]["_group_of"]
        base = mf_gain(ra, rb, "persp_crop")
        numbers["multi_contrast"][tag] = {}
        for st in ("persp_full", "persp_masked", "fisheye_full", "persp_crop_lores"):
            if st not in rb:
                continue
            other = mf_gain(ra, rb, st)
            d = {k: base[k] - other[k] for k in set(base) & set(other)}
            cb = cluster_bootstrap(d, {k: 0.0 for k in d}, go)
            cb["per_group"] = _per_group_means(d, go)
            numbers["multi_contrast"][tag][st] = cb

    # ---- is the multi-frame gain explained by the pose being right? Tested at
    # the level where it could be causal -- the same window, different arms --
    # rather than at the level where "fisheye vs perspective" would fake it.
    rows = []
    for st in res8:
        if "_per_window" not in res8[st]:
            continue
        g = _per_group_means(mf_gain(res1, res8, st), res8[st]["_group_of"])
        auc = {res8[st]["_window_group"][w]: v["auc30"] for w, v in res8[st]["_per_window"].items()}
        rows += [(st, k, g[k], auc[k]) for k in g]
    G = np.array([r[2] for r in rows]); A = np.array([r[3] for r in rows])
    def centred(by):
        Gc, Ac = G.copy(), A.copy()
        for key in set(r[by] for r in rows):
            m = np.array([r[by] == key for r in rows])
            Gc[m] -= G[m].mean(); Ac[m] -= A[m].mean()
        return float(np.corrcoef(Gc, Ac)[0, 1])
    numbers["mediation"] = {"n": len(rows), "r_raw": float(np.corrcoef(G, A)[0, 1]),
                            "r_within_window": centred(1), "r_within_setting": centred(0),
                            "per_setting": {st: {"gain": float(np.mean([r[2] for r in rows if r[0] == st])),
                                                 "auc30": float(np.mean([r[3] for r in rows if r[0] == st]))}
                                            for st in dict.fromkeys(r[0] for r in rows)}}

    # ---- the inscribed crop: vs the others, own and common region; the 2x2
    # effects and all five cells under the common region.
    numbers["crop_effects_own"], numbers["crop_effects_common"] = {}, {}
    numbers["cells_common"], numbers["effects_common"] = {}, {}
    for mode in results:
        numbers["crop_effects_own"][mode] = {m: crop_effects(results[mode], m) for m in ("AbsRel", "delta1", "RMSE")}
        numbers["crop_effects_common"][mode] = {m: crop_effects(results_c[mode], m) for m in ("AbsRel", "delta1", "RMSE")}
        if mode != "single":
            for m in ("auc30", "rot_err_deg", "trans_err_deg", "ate_m", "rra15", "rta15"):
                numbers["crop_effects_own"][mode][m] = crop_effects(results[mode], m, unit="window")
        numbers["cells_common"][mode] = {st: {k: results_c[mode][st][k] for k in ("AbsRel", "RMSE", "delta1", "n_frames", "n_groups", "n_valid")}
                                         for st in results_c[mode]}
        numbers["effects_common"][mode] = {m: paired_effects(results_c[mode], m) for m in ("AbsRel", "delta1", "RMSE")}
    # multi-frame minus single, per setting
    numbers["multi_vs_single"] = {}
    for st in res1:
        a = {k: v["AbsRel"] for k, v in res1[st]["_per_frame_metrics"].items()}
        b = {k: v["AbsRel"] for k, v in res8[st]["_per_frame_metrics"].items()}
        d = {k: b[k] - a[k] for k in set(a) & set(b)}
        numbers["multi_vs_single"][st] = cluster_bootstrap(d, {k: 0.0 for k in d}, res8[st]["_group_of"])

    # ---- the control arm must actually be blurrier than the arm it controls,
    # measured on the real inputs rather than trusted from the resize call.
    hf = lambda x: float(np.abs(np.diff(x.astype(np.float32), axis=1)).mean())
    sharp, blur = [], []
    for f in sorted(glob.glob(os.path.join(raw, "persp_crop_s1_w*.npz"))):
        b = f.replace("persp_crop_s1", "persp_crop_lores_s1")
        if not os.path.isfile(b):
            continue
        zs, zb = np.load(f), np.load(b)
        assert str(zs["frame_dir"]) == str(zb["frame_dir"])
        sharp.append(hf(zs["rgb"])); blur.append(hf(zb["rgb"]))
    if sharp:
        numbers["control"] = {"n": len(sharp), "hf_sharp": float(np.mean(sharp)),
                              "hf_blur": float(np.mean(blur)),
                              "hf_drop_pct": 100 * (1 - float(np.mean(blur)) / float(np.mean(sharp)))}

    # ---- what the model actually received: pure-black and graded fractions,
    # measured on the dumped tensors of all 96 single-frame inputs, not on the
    # analytic mask alone.
    numbers["inputs"] = {}
    for st in CELLS:
        blk, grd = [], []
        for f in sorted(glob.glob(os.path.join(raw, f"{st}_s1_w*.npz"))):
            z = np.load(f)
            blk.append(float((z["rgb"].max(-1) == 0).mean()))
            grd.append(float(z["mask"].mean()))
        numbers["inputs"][st] = {"black_pct": 100 * float(np.mean(blk)), "graded_pct": 100 * float(np.mean(grd)),
                                 "n": len(blk)}

    # ---- where in the image the fill effect lives: AbsRel in a band within
    # 16 px of the hole boundary vs the interior, black vs true content, on all
    # 96 single-frame inputs. Prose about "a bright ring at the border" is
    # otherwise an eyeballed claim.
    from scipy import ndimage
    numbers["band"] = {}
    for proj in ("fisheye", "persp"):
        acc = {"band_black": [], "band_full": [], "int_black": [], "int_full": []}
        for f in sorted(glob.glob(os.path.join(raw, f"{proj}_masked_s1_w*.npz"))):
            zm, zf = np.load(f), np.load(f.replace("_masked_", "_full_"))
            assert str(zm["frame_dir"]) == str(zf["frame_dir"])
            m, gt = zm["mask"].astype(bool), zm["gt"].astype(np.float32)
            dist = ndimage.distance_transform_edt(m)
            band, inner = m & (dist <= 16), m & (dist > 16)
            for tag, z in (("black", zm), ("full", zf)):
                e = np.abs(z["pred"].astype(np.float32) - gt) / np.maximum(gt, 1e-6)
                acc[f"band_{tag}"].append(float(e[band].mean()))
                acc[f"int_{tag}"].append(float(e[inner].mean()))
        r = {k: float(np.mean(v)) for k, v in acc.items()}
        r["band_gain_pct"] = 100 * (r["band_black"] - r["band_full"]) / r["band_black"]
        r["int_gain_pct"] = 100 * (r["int_black"] - r["int_full"]) / r["int_black"]
        numbers["band"][proj] = r
    # ---- per-window wins: in how many of the 12 windows is true content better?
    numbers["wins"] = {}
    for mode, res in results.items():
        for proj in ("fisheye", "persp"):
            f, b = res[f"{proj}_full"], res[f"{proj}_masked"]
            g = OrderedDict()
            for d, mm in f["_per_frame_metrics"].items():
                g.setdefault(f["_group_of"][d], []).append(mm["AbsRel"] - b["_per_frame_metrics"][d]["AbsRel"])
            w = {"absrel_better": int(sum(np.mean(v) < 0 for v in g.values())), "n": len(g)}
            if "_per_window" in f:
                a = {k: v["auc30"] for k, v in f["_per_window"].items()}
                bb = {k: v["auc30"] for k, v in b["_per_window"].items()}
                d = [a[k] - bb[k] for k in a]
                w.update({"auc_higher": int(sum(x > 1e-9 for x in d)), "auc_equal": int(sum(abs(x) <= 1e-9 for x in d)),
                          "auc_lower": int(sum(x < -1e-9 for x in d))})
            numbers["wins"][f"{mode}/{proj}"] = w

    # ---- figures
    wkeys = list(res8["persp_full"]["_windows"].keys())
    for wi, wk in enumerate(wkeys):
        fdir = fig_panels(raw, results, wi, os.path.join(args.out, f"panels_w{wi:02d}.jpg"))
        numbers["windows"].append({"index": wi, "key": wk, "first_frame": fdir,
                                   "group": res8["persp_full"]["_windows"][wk]["group"]})
        if wi < 4:
            fig_inputs(raw, wi, os.path.join(args.out, f"inputs_w{wi:02d}.jpg"))
    fig_trajectories(res8, os.path.join(args.out, "trajectories.png"))
    fig_fov(res1, res8, os.path.join(args.out, "fov.png"))
    fig_effects(numbers["effects"], os.path.join(args.out, "effects.png"))
    fig_crop_effects(numbers, os.path.join(args.out, "crop_effects.png"))
    fig_five_vs_four(numbers, os.path.join(args.out, "five_vs_four.png"))
    with open(os.path.join(args.out, "numbers.json"), "w") as fh:
        json.dump(numbers, fh, indent=1, ensure_ascii=False)
    print(f"[figures] wrote {len(wkeys)} panel figures + inputs/trajectories/fov/effects + numbers.json -> {args.out}")


if __name__ == "__main__":
    main()
