# Copyright (c) 2026.
"""VGGT-Omega on Blender-rendered ADT frames: 4 settings x {single, multi}-frame.

The four settings are the 2x2 of projection x validity, all rendered from the same
equirectangular panorama at the same pose, so they differ in nothing else:

    fisheye_full     Aria KB4, KB4 extended past its turnover -> 0% invalid
    fisheye_masked   same pixels, real imaged-disc mask       -> 14.75% invalid
    persp_full       pinhole aligned to FisheyeRectifier      -> 0% invalid
    persp_masked     same pixels, analytic rectification mask -> 32.29% invalid

Why rendered rather than real footage: the `_full` arms hold **true content** in
regions a real Aria frame cannot supply, because the lens never imaged them. That
is the ground-truth ("oracle") fill the earlier real-footage 2x2 could only
approximate with `replicate`, and it is obtainable only from a renderer that owns
the scene.

Scoring rule — the one that makes full-vs-masked a clean contrast
----------------------------------------------------------------
Both arms of a projection are scored on the **masked** arm's valid region (the
smaller set), never on the full arm's larger one. Otherwise the full arm would be
credited for area the masked arm does not even have, and the comparison would
measure coverage instead of the thing under test. With this rule the two arms see
different inputs and are graded on identical pixels, so any difference is the
encoder reacting to what was in the invalid region.

Across projections the pixel grids differ (the rectified grid oversamples the
periphery ~10x), so absolute numbers are not comparable between the fisheye and
perspective rows. The within-row contrast, and the interaction between rows, are.

Single vs multi-frame
---------------------
`--seq-len 1` feeds one frame; `--seq-len 8` feeds a window. VGGT resolves
monocular scale ambiguity through cross-view attention, so the multi-frame arm is
where a corrupted input can do damage beyond its own frame. Windows must carry
real camera translation — pure rotation is triangulation-degenerate and would
handicap the multi-frame arm for reasons unrelated to the fill.

Usage
-----
::

    python -m finetune.eval.exp_rendered --render-root <ROOT> \\
        --vggt-checkpoint <CKPT> --out runs/exp_rendered --seq-len 1
    python -m finetune.eval.exp_rendered --render-root <ROOT> ... --seq-len 8
"""
from __future__ import annotations

import sys as _sys, os as _os
if not __package__:
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
    __package__ = "finetune.eval"

import argparse
import glob
import json
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .metrics import align_depth, depth_metrics

SETTINGS = ("fisheye_full", "fisheye_masked", "persp_full", "persp_masked")
# Which analytic mask grades each setting. Both arms of a projection are graded on
# the MASKED arm's region -- see the scoring rule in the module docstring.
GRADING_MASK = {
    "fisheye_full": "mask_fisheye_valid", "fisheye_masked": "mask_fisheye_valid",
    "persp_full": "mask_persp_valid",     "persp_masked": "mask_persp_valid",
}


def find_frames(root: str, sequences: Optional[List[str]] = None) -> List[str]:
    """Every frame dir under the render root, sorted by (sequence, frame index)."""
    dirs = sorted(glob.glob(os.path.join(root, "*", "frame_*")))
    if sequences:
        keep = set(sequences)
        dirs = [d for d in dirs if os.path.basename(os.path.dirname(d)) in keep]
    return [d for d in dirs if os.path.isfile(os.path.join(d, "meta.json"))]


class RenderedWindowDataset(Dataset):
    """Windows of rendered frames for one setting.

    Windows never span sequences. When a manifest with explicit window ids is
    present it is honoured, because the renderer chose those groupings for their
    inter-frame baseline; otherwise frames are chunked in order.
    """

    def __init__(self, root: str, setting: str, seq_len: int = 1,
                 sequences: Optional[List[str]] = None,
                 manifest: Optional[str] = None) -> None:
        if setting not in SETTINGS:
            raise ValueError(f"unknown setting {setting!r}; expected one of {SETTINGS}")
        self.setting, self.seq_len, self.root = setting, seq_len, root
        frames = find_frames(root, sequences)
        if not frames:
            raise SystemExit(f"[exp_rendered] no rendered frames under {root!r}")

        groups: "OrderedDict[str, List[str]]" = OrderedDict()
        if manifest and os.path.isfile(manifest):
            man = json.load(open(manifest))
            rows = man["frames"] if isinstance(man, dict) else man
            for r in rows:
                # The manifest stores `dir` RELATIVE to the render root, so it must
                # be joined before testing -- an unjoined relative path silently
                # fails os.path.isdir and yields zero windows.
                d = r.get("dir") or os.path.join(r["sequence"], f"frame_{int(r['frame_idx']):04d}")
                if not os.path.isabs(d):
                    d = os.path.join(root, d)
                if os.path.isdir(d):
                    key = r.get("window_id") or f"{r['sequence']}/w{r.get('window', 0)}"
                    groups.setdefault(key, []).append(d)
        else:
            for d in frames:
                groups.setdefault(os.path.basename(os.path.dirname(d)), []).append(d)

        self.windows: List[List[str]] = []
        # Frame -> group key.  The GROUP (the renderer's own window: consecutive
        # frames of one trajectory segment in one sequence) is the unit of
        # independence, and it does NOT depend on seq_len: at seq_len=1 every eval
        # window is a single frame, but those frames still share a scene, a
        # lighting and a few tenths of a second of trajectory. Resampling them as
        # if independent is what makes a frame-level CI far too narrow.
        self.group_of: Dict[str, str] = {}
        for gkey, ds in groups.items():
            ds = sorted(ds)
            for d in ds:
                self.group_of[d] = gkey
            # Non-overlapping chunks so every frame is scored exactly once.
            for i in range(0, len(ds) - seq_len + 1, seq_len):
                self.windows.append(ds[i:i + seq_len])
        if not self.windows:
            raise SystemExit(
                f"[exp_rendered] no windows of length {seq_len} from "
                f"{len(frames)} frames / {len(groups)} groups. If a manifest was "
                f"given, check its `dir` fields resolve under --render-root.")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> dict:
        imgs, deps, masks = [], [], []
        for d in self.windows[i]:
            rgb = np.load(os.path.join(d, f"{self.setting}_rgb.npy"))
            if rgb.dtype == np.uint8:
                rgb = rgb.astype(np.float32) / 255.0
            dep = np.load(os.path.join(d, f"{self.setting}_depth.npy")).astype(np.float32)
            gm = np.load(os.path.join(d, f"{GRADING_MASK[self.setting]}.npy")).astype(bool)
            imgs.append(torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1))
            deps.append(torch.from_numpy(dep))
            # Grade only where the analytic mask AND the GT agree -- the full arm's
            # depth exists outside the mask, and must not be scored there.
            masks.append(torch.from_numpy(gm & (dep > 0)))
        return {"images": torch.stack(imgs), "depths": torch.stack(deps),
                "valid_masks": torch.stack(masks), "dirs": self.windows[i]}


@torch.no_grad()
def evaluate(model, root: str, setting: str, seq_len: int, device: torch.device,
             sequences: Optional[List[str]] = None, manifest: Optional[str] = None,
             align: str = "scale_shift", qual_dir: Optional[str] = None) -> dict:
    ds = RenderedWindowDataset(root, setting, seq_len, sequences, manifest)
    per_frame, fovs = [], []
    for wi in range(len(ds)):
        s = ds[wi]
        preds = model(s["images"].unsqueeze(0).to(device))
        dp = preds["depth"]
        if dp.ndim == 5:
            dp = dp.squeeze(-1)
        dp = dp[0].float().cpu().numpy()
        pe = preds.get("pose_enc")
        if pe is not None:
            fovs.append(np.degrees(pe[0, :, 7:9].float().cpu().numpy()))
        for fi in range(dp.shape[0]):
            gt = s["depths"][fi].numpy()
            m = s["valid_masks"][fi].numpy()
            if m.sum() < 100:
                continue
            pa = align_depth(dp[fi], gt, m, mode=align)
            met = depth_metrics(pa, gt, m)
            # Key on the frame dir so the two arms of a projection can be PAIRED:
            # they see the same scene, so an unpaired test throws away the variance
            # that the pairing removes and badly understates significance.
            met["_dir"] = s["dirs"][fi]
            per_frame.append(met)
            if qual_dir and wi == 0 and fi == 0:
                _save_qual(qual_dir, setting, seq_len, s["images"][fi].numpy(),
                           pa, gt, m)
    if not per_frame:
        return {}
    out = {k: float(np.mean([f[k] for f in per_frame]))
           for k in per_frame[0]
           if not k.startswith("_") and isinstance(per_frame[0][k], (int, float))}
    out["n_frames"] = len(per_frame)
    out["n_windows"] = len(ds)
    out["_per_frame"] = {f["_dir"]: float(f["AbsRel"]) for f in per_frame}
    out["_group_of"] = {f["_dir"]: ds.group_of.get(f["_dir"], f["_dir"]) for f in per_frame}
    out["n_groups"] = len(set(out["_group_of"].values()))
    if fovs:
        f = np.concatenate(fovs, 0)
        out["fov_h_deg"] = float(f[:, 0].mean())
        out["fov_h_std"] = float(f[:, 0].std())
    return out


def _save_qual(qual_dir: str, setting: str, seq_len: int, img_chw: np.ndarray,
               pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> None:
    """RGB | pred | GT | error, on one shared depth scale so panels are comparable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(qual_dir, exist_ok=True)
    vmin, vmax = np.percentile(gt[mask], [2, 98])
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    ax[0].imshow(np.clip(img_chw.transpose(1, 2, 0), 0, 1)); ax[0].set_title(setting)
    for a, (d, t) in zip(ax[1:3], [(pred, "pred (scale-shift)"), (gt, "GT (rendered)")]):
        a.imshow(np.where(mask, d, np.nan), vmin=vmin, vmax=vmax, cmap="turbo"); a.set_title(t)
    err = np.where(mask, np.abs(pred - gt) / np.maximum(gt, 1e-6), np.nan)
    im = ax[3].imshow(err, vmin=0, vmax=0.3, cmap="magma"); ax[3].set_title("AbsRel")
    fig.colorbar(im, ax=ax[3], fraction=0.046)
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{setting}  ·  {'single-frame' if seq_len == 1 else f'{seq_len}-frame'}")
    fig.tight_layout()
    fig.savefig(os.path.join(qual_dir, f"{setting}_s{seq_len}.png"), dpi=110)
    plt.close(fig)


def paired_bootstrap(a: Dict[str, float], b: Dict[str, float],
                     n_boot: int = 10000, seed: int = 0) -> Optional[dict]:
    """Bootstrap CI for mean(a - b) over frames present in BOTH arms.

    Paired, because the two arms are the same scenes: resampling frames
    independently would reintroduce the between-scene variance that pairing
    removes, and understate significance. Returns None if the arms share no
    frames (which would itself mean the comparison is not what it claims).
    """
    keys = sorted(set(a) & set(b))
    if len(keys) < 3:
        return None
    d = np.array([a[k] - b[k] for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n_pairs": len(d), "mean": float(d.mean()),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def cluster_bootstrap(a: Dict[str, float], b: Dict[str, float],
                      group_of: Dict[str, str], n_boot: int = 10000,
                      seed: int = 0) -> Optional[dict]:
    """Paired bootstrap that resamples GROUPS, not frames.

    The frames of one rendered window share a scene, a lighting and a fraction of
    a second of trajectory, so they are not independent draws. A frame-level
    bootstrap treats n frames as n samples and returns an interval that is too
    narrow by roughly sqrt(frames per group). Resampling whole groups -- taking
    every paired frame of each drawn group -- propagates the between-group
    variance instead, which is the variance a new sequence would actually show.

    The estimate is the mean over frames (identical to the frame-level point
    estimate); only the interval differs. Reported alongside the number of
    groups, because with a handful of groups the interval is wide for a real
    reason and quoting it without n_groups invites the same overconfidence the
    frame-level version produced.
    """
    keys = sorted(set(a) & set(b) & set(group_of))
    if len(keys) < 3:
        return None
    by_group: "OrderedDict[str, List[float]]" = OrderedDict()
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
    return {"n_pairs": len(d_all), "n_groups": len(gkeys), "mean": float(d_all.mean()),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "frames_per_group": [len(v) for v in vals]}


def report(results: Dict[str, Dict[str, dict]], out_dir: str) -> str:
    """Per-setting table plus the contrasts that actually answer the question."""
    lines = ["", "=" * 86,
             "VGGT-Omega on rendered ADT — 4 settings x single/multi-frame",
             "(AbsRel lower better; graded on the MASKED arm's region in both arms)",
             "=" * 86,
             f"{'setting':<18}{'mode':<8}{'n':>5}{'win':>5}{'AbsRel':>10}{'RMSE':>9}"
             f"{'delta1':>10}{'fov_h':>9}"]
    for mode, res in results.items():
        for s in SETTINGS:
            r = res.get(s) or {}
            if not r:
                continue
            lines.append(f"{s:<18}{mode:<8}{r.get('n_frames',0):>5}"
                         f"{r.get('n_groups',0):>5}"
                         f"{r.get('AbsRel',float('nan')):>10.4f}{r.get('RMSE',float('nan')):>9.3f}"
                         f"{r.get('delta1',float('nan')):>10.4f}{r.get('fov_h_deg',float('nan')):>9.1f}")
    lines.append("")
    for mode, res in results.items():
        def pf(s):
            return (res.get(s) or {}).get("_per_frame", {})
        def go(s):
            return (res.get(s) or {}).get("_group_of", {})
        lines.append(f"[{mode}] true content vs black  (AbsRel full - masked; "
                     f"negative = true content helps):")
        lines.append("    CI(win) resamples WINDOWS and is the one to quote; CI(frm) "
                     "resamples frames and is too narrow (frames in a window share a "
                     "scene) -- it is shown only to make the gap visible.")
        eff = {}
        for proj in ("fisheye", "persp"):
            a_, b_ = pf(f"{proj}_full"), pf(f"{proj}_masked")
            bs = paired_bootstrap(a_, b_)
            if bs is None:
                lines.append(f"    {proj:<8} (no paired frames)")
                continue
            cb = cluster_bootstrap(a_, b_, go(f"{proj}_full") or go(f"{proj}_masked"))
            eff[proj] = cb or bs
            if cb:
                star = "  SIGNIFICANT" if cb["excludes_zero"] else "  n.s. (CI spans 0)"
                lines.append(
                    f"    {proj:<8} {cb['mean']:+.4f}  CI(win) "
                    f"[{cb['ci_lo']:+.4f}, {cb['ci_hi']:+.4f}]  "
                    f"CI(frm) [{bs['ci_lo']:+.4f}, {bs['ci_hi']:+.4f}]  "
                    f"n={cb['n_pairs']}f/{cb['n_groups']}w{star}")
            else:
                star = "  SIGNIFICANT" if bs["excludes_zero"] else "  n.s. (CI spans 0)"
                lines.append(f"    {proj:<8} {bs['mean']:+.4f}  CI(frm) "
                             f"[{bs['ci_lo']:+.4f}, {bs['ci_hi']:+.4f}]  "
                             f"n={bs['n_pairs']} (one window only){star}")
        if len(eff) == 2:
            # Interaction, paired at frame level: (full-masked)_persp - (full-masked)_fisheye
            fk = sorted(set(pf("fisheye_full")) & set(pf("fisheye_masked")))
            pk = sorted(set(pf("persp_full")) & set(pf("persp_masked")))
            common = sorted(set(fk) & set(pk))
            if len(common) >= 3:
                dd = {k: (pf("persp_full")[k] - pf("persp_masked")[k])
                         - (pf("fisheye_full")[k] - pf("fisheye_masked")[k])
                      for k in common}
                d = np.array(list(dd.values()))
                rng = np.random.default_rng(1)
                b = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
                lo, hi = np.percentile(b, [2.5, 97.5])
                gmap = go("persp_full") or go("fisheye_full")
                cb = cluster_bootstrap(dd, {k: 0.0 for k in dd}, gmap)
                if cb:
                    sig = "SIGNIFICANT" if cb["excludes_zero"] else "n.s. (CI spans 0)"
                    lines += [f"    INTERACTION {cb['mean']:+.4f}  CI(win) "
                              f"[{cb['ci_lo']:+.4f}, {cb['ci_hi']:+.4f}]  "
                              f"CI(frm) [{lo:+.4f}, {hi:+.4f}]  "
                              f"n={cb['n_pairs']}f/{cb['n_groups']}w  {sig}",
                              "      negative => true content helps MORE in the "
                              "perspective domain"]
                else:
                    sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s. (CI spans 0)"
                    lines += [f"    INTERACTION {d.mean():+.4f}  CI(frm) "
                              f"[{lo:+.4f}, {hi:+.4f}]  n={len(d)}  {sig} "
                              f"(one window only)",
                              "      negative => true content helps MORE in the "
                              "perspective domain"]
    txt = "\n".join(lines)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.txt"), "w") as fh:
        fh.write(txt + "\n")
    return txt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render-root", required=True)
    ap.add_argument("--vggt-checkpoint", required=True)
    ap.add_argument("--out", default="runs/exp_rendered")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--sequences", default="", help="comma-separated; default all")
    ap.add_argument("--seq-lens", default="1,8", help="single- and multi-frame modes")
    ap.add_argument("--align", default="scale_shift")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seqs = args.sequences.split(",") if args.sequences else None
    os.makedirs(args.out, exist_ok=True)

    from .run_eval import _load_vggt_base
    model = _load_vggt_base(args.vggt_checkpoint, device)
    model.eval()

    results: Dict[str, Dict[str, dict]] = OrderedDict()
    for sl in [int(x) for x in args.seq_lens.split(",")]:
        mode = "single" if sl == 1 else f"{sl}-frame"
        results[mode] = {}
        for s in SETTINGS:
            print(f"\n[exp_rendered] {s}  ·  {mode}")
            results[mode][s] = evaluate(
                model, args.render_root, s, sl, device, seqs,
                args.manifest or None, args.align,
                qual_dir=os.path.join(args.out, "qual"))
            print("   ", {k: round(v, 4) for k, v in results[mode][s].items()
                          if isinstance(v, float)})
            with open(os.path.join(args.out, "results.json"), "w") as fh:
                json.dump(results, fh, indent=2, default=str)
    print(report(results, args.out))


if __name__ == "__main__":
    main()
