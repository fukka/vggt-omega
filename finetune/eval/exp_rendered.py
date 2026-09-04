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
                d = r.get("dir") or os.path.join(root, r["sequence"], f"frame_{int(r['frame_idx']):04d}")
                if os.path.isdir(d):
                    groups.setdefault(f"{r['sequence']}/w{r.get('window', 0)}", []).append(d)
        else:
            for d in frames:
                groups.setdefault(os.path.basename(os.path.dirname(d)), []).append(d)

        self.windows: List[List[str]] = []
        for _, ds in groups.items():
            ds = sorted(ds)
            # Non-overlapping chunks so every frame is scored exactly once.
            for i in range(0, len(ds) - seq_len + 1, seq_len):
                self.windows.append(ds[i:i + seq_len])
        if not self.windows:
            raise SystemExit(f"[exp_rendered] no windows of length {seq_len} "
                             f"(found {len(frames)} frames)")

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
            per_frame.append(depth_metrics(pa, gt, m))
            if qual_dir and wi == 0 and fi == 0:
                _save_qual(qual_dir, setting, seq_len, s["images"][fi].numpy(),
                           pa, gt, m)
    if not per_frame:
        return {}
    out = {k: float(np.mean([f[k] for f in per_frame]))
           for k in per_frame[0] if isinstance(per_frame[0][k], (int, float))}
    out["n_frames"] = len(per_frame)
    out["n_windows"] = len(ds)
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


def report(results: Dict[str, Dict[str, dict]], out_dir: str) -> str:
    """Per-setting table plus the contrasts that actually answer the question."""
    lines = ["", "=" * 86,
             "VGGT-Omega on rendered ADT — 4 settings x single/multi-frame",
             "(AbsRel lower better; graded on the MASKED arm's region in both arms)",
             "=" * 86,
             f"{'setting':<18}{'mode':<8}{'n':>5}{'AbsRel':>10}{'RMSE':>9}{'delta1':>10}{'fov_h':>9}"]
    for mode, res in results.items():
        for s in SETTINGS:
            r = res.get(s) or {}
            if not r:
                continue
            lines.append(f"{s:<18}{mode:<8}{r.get('n_frames',0):>5}"
                         f"{r.get('AbsRel',float('nan')):>10.4f}{r.get('RMSE',float('nan')):>9.3f}"
                         f"{r.get('delta1',float('nan')):>10.4f}{r.get('fov_h_deg',float('nan')):>9.1f}")
    lines.append("")
    for mode, res in results.items():
        def ar(s):
            return (res.get(s) or {}).get("AbsRel", float("nan"))
        fe = ar("fisheye_full") - ar("fisheye_masked")   # oracle content effect, fisheye
        pe = ar("persp_full") - ar("persp_masked")       # oracle content effect, perspective
        lines += [f"[{mode}] true content vs black:",
                  f"    fisheye  (full - masked): {fe:+.4f}",
                  f"    persp    (full - masked): {pe:+.4f}",
                  f"    interaction (persp - fisheye): {pe - fe:+.4f}",
                  "      negative => the true content helps MORE once the projection is perspective"]
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
