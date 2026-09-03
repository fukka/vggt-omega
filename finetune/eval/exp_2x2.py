# Copyright (c) 2026.
"""2x2 experiment: projection (raw fisheye | rectified) x fill (black | filled).

Question
--------
Rectifying an Aria KB4 fisheye into a single perspective frame leaves black
wedges. Does filling them recover VGGT-Omega's accuracy by restoring the model's
ability to estimate its own camera?

The four cells separate the two factors, so the readout is the **interaction**::

    (4 - 2) - (3 - 1)

i.e. does filling pay off *more* in the perspective domain than in the fisheye
domain? Both main effects already have answers in the literature (Fisheye3R
arXiv:2603.28896 argues rectification is lossy; FisheyeEX arXiv:2206.05844
argues fisheye outpainting helps). The interaction is what nobody has measured:
the claim that filling only matters once the projection is already in-domain.

Cells
-----
::

    id  projection  fill        focal_out_norm  hFoV     black
    1   raw         black       -               fisheye  14.7% px (sensor corners)
    2   rect        black       0.262           124.7    21.6% px / 6.7% sr
    3   raw         filled      -               fisheye  filled
    4   rect        filled      0.262           124.7    filled      <- the hypothesis
    5   rect        black       0.371           106.9     0.0%       <- inscribed, free alternative
    0   rect        black       0.550            84.6     0.0%       <- the repo's historical default

Cell 5 is the alternative cell 4 must beat: black-free by construction, costing
16.7% of the imaged cone. Cell 0 shows where the repo has been standing all
along -- it keeps only ~56% of the cone.

Comparability
-------------
Within a row (black vs filled) the two cells differ in exactly one thing, so the
difference is clean. *Across* rows the pixel grids differ -- the rectified grid
oversamples the periphery by ~10x -- so absolute numbers are not directly
comparable between the raw and rectified rows. The interaction term is a
difference of within-row differences, so that grid weighting cancels to first
order. This is the main reason to report the interaction rather than four
absolute values.

Every cell scores only pixels inside the analytic imaged cone with valid GT, so
filled pixels are never compared against ground truth: the fill changes what the
encoder sees, never what the metric measures.

Usage
-----
::

    # write example inputs for all six cells (no GPU, no model)
    python -m finetune.eval.exp_2x2 --adt-root <ROOT> --dump-examples out/examples

    # the experiment
    python -m finetune.eval.exp_2x2 --adt-root <ROOT> \\
        --vggt-checkpoint <CKPT> --out runs/exp2x2 --max-frames 100
"""
from __future__ import annotations

import sys as _sys, os as _os
if not __package__:
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
    __package__ = "finetune.eval"

import argparse
import json
import os
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import torch

from ..data.rectify import FOCAL_OUT_CIRCUMSCRIBED, FOCAL_OUT_INSCRIBED

# ─────────────────────────────────────────────────────────────────────────────
# Cell definitions. `fill` is the strategy for the out-of-cone region; the
# default "replicate" is the strongest fill that invents no content at all
# (every filled pixel is a copy of a real one), which makes it the right first
# probe: if it already closes the gap, no generative filler is needed.
# ─────────────────────────────────────────────────────────────────────────────
def make_cells(fill: str = "replicate") -> "OrderedDict[str, dict]":
    return OrderedDict([
        ("1_raw_black", dict(
            label="① RAW·BLACK", rectify=False, fill="black", focal_out_norm=None,
            blurb="raw fisheye, sensor corners left black")),
        ("2_rect_black", dict(
            label="② RECT·BLACK", rectify=True, fill="black",
            focal_out_norm=FOCAL_OUT_CIRCUMSCRIBED,
            blurb="circumscribed rectification, 21.6% black wedges")),
        ("3_raw_filled", dict(
            label="③ RAW·FILLED", rectify=False, fill=fill, focal_out_norm=None,
            blurb=f"raw fisheye, corners filled ({fill})")),
        ("4_rect_filled", dict(
            label="④ RECT·FILLED", rectify=True, fill=fill,
            focal_out_norm=FOCAL_OUT_CIRCUMSCRIBED,
            blurb=f"circumscribed rectification, wedges filled ({fill})")),
        ("5_rect_inscribed", dict(
            label="⑤ RECT·INSCRIBED", rectify=True, fill="black",
            focal_out_norm=FOCAL_OUT_INSCRIBED,
            blurb="inscribed crop, black-free by construction")),
        ("0_rect_default", dict(
            label="⓪ RECT·DEFAULT", rectify=True, fill="black", focal_out_norm=None,
            blurb="the repo's historical default (focal_out_norm=0.55)")),
    ])


def _dataset_for(cell: dict, seq_dirs: List[str], res: int, seq_len: int,
                 max_frames: Optional[int], rgb_subdir: str):
    from .adt_depth import ADTWindowDataset
    return ADTWindowDataset(
        seq_dirs, seq_len=seq_len, image_resolution=res, max_frames=max_frames,
        rgb_subdir=rgb_subdir, rectify=cell["rectify"], camera_preset="aria-214-1",
        focal_out_norm=cell["focal_out_norm"], fill=cell["fill"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Example dumper -- writes exactly the tensors the model is fed
# ─────────────────────────────────────────────────────────────────────────────
def dump_examples(out_dir: str, seq_dirs: List[str], res: int, rgb_subdir: str,
                  fill: str, frame: int = 0) -> Dict[str, str]:
    """Render one ADT frame through every cell and save PNGs + a labelled montage.

    Pulls the image out of ``ADTWindowDataset`` rather than re-deriving it, so
    what is shown is byte-for-byte what the model receives.
    """
    from PIL import Image, ImageDraw
    os.makedirs(out_dir, exist_ok=True)
    cells = make_cells(fill)
    tiles, paths = [], {}

    for cid, cell in cells.items():
        ds = _dataset_for(cell, seq_dirs, res, seq_len=1, max_frames=frame + 1,
                          rgb_subdir=rgb_subdir)
        sample = ds[0]
        img = sample["images"][0].permute(1, 2, 0).numpy()          # [H,W,3] float
        arr = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        p = os.path.join(out_dir, f"{cid}.png")
        Image.fromarray(arr).save(p)
        paths[cid] = p
        gt_valid = float((sample["valid_masks"][0].numpy()).mean())
        tiles.append((cid, cell, arr, gt_valid))
        print(f"  [{cell['label']}] {cid}.png  scored-pixel fraction={gt_valid:.1%}")

    # Montage: 2 rows x 3 cols, captioned.
    pad, cap, cols = 10, 46, 3
    rows = (len(tiles) + cols - 1) // cols
    W = cols * res + (cols + 1) * pad
    H = rows * (res + cap) + (rows + 1) * pad
    sheet = Image.new("RGB", (W, H), (245, 246, 246))
    dr = ImageDraw.Draw(sheet)
    for i, (cid, cell, arr, gt_valid) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (res + pad)
        y = pad + r * (res + cap + pad)
        sheet.paste(Image.fromarray(arr), (x, y))
        dr.text((x + 2, y + res + 6), cell["label"], fill=(16, 26, 28))
        dr.text((x + 2, y + res + 20), cell["blurb"][:58], fill=(95, 115, 118))
        dr.text((x + 2, y + res + 32), f"scored pixels: {gt_valid:.1%}",
                fill=(95, 115, 118))
    mp = os.path.join(out_dir, "montage.png")
    sheet.save(mp)
    paths["montage"] = mp
    print(f"  montage -> {mp}")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# FoV probe -- the metric closest to the hypothesis
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def probe_fov(model, cell: dict, seq_dirs: List[str], res: int, seq_len: int,
              device: torch.device, rgb_subdir: str, n_windows: int = 8) -> dict:
    """VGGT's inferred FoV (pose_enc[7]=fov_h, [8]=fov_w, radians) per cell.

    This is the readout closest to the hypothesis: the claim is that black
    regions corrupt the camera estimate, and that a bent camera estimate is what
    bends the depth (commit 04b6d4f: DAv2 is unaffected on the same crop).
    """
    ds = _dataset_for(cell, seq_dirs, res, seq_len, max_frames=n_windows * seq_len,
                      rgb_subdir=rgb_subdir)
    fovs = []
    for i in range(min(n_windows, len(ds))):
        imgs = ds[i]["images"].unsqueeze(0).to(device)
        preds = model(imgs)
        pe = preds["pose_enc"]
        if pe is None:
            return {}
        fovs.append(pe[..., 7:9].float().cpu().numpy().reshape(-1, 2))
    if not fovs:
        return {}
    f = np.degrees(np.concatenate(fovs, 0))
    return {"fov_h_deg_mean": float(f[:, 0].mean()), "fov_h_deg_std": float(f[:, 0].std()),
            "fov_w_deg_mean": float(f[:, 1].mean()), "fov_w_deg_std": float(f[:, 1].std()),
            "n": int(len(f))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adt-root", required=True)
    ap.add_argument("--rgb-subdir", default="videos_synthetic")
    ap.add_argument("--vggt-checkpoint", default="")
    ap.add_argument("--out", default="runs/exp2x2")
    ap.add_argument("--dump-examples", default="")
    ap.add_argument("--fill", default="replicate")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--max-seqs", type=int, default=4)
    ap.add_argument("--cells", default="", help="comma-separated cell ids (default: all)")
    ap.add_argument("--skip-dav2", action="store_true")
    args = ap.parse_args()

    from .run_eval import _find_adt_seq_dirs
    seq_dirs = _find_adt_seq_dirs(args.adt_root, args.rgb_subdir)
    if args.max_seqs:
        seq_dirs = seq_dirs[: args.max_seqs]
    print(f"[exp2x2] {len(seq_dirs)} ADT sequence(s), rgb_subdir={args.rgb_subdir!r}")

    if args.dump_examples:
        dump_examples(args.dump_examples, seq_dirs, args.resolution,
                      args.rgb_subdir, args.fill)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    cells = make_cells(args.fill)
    if args.cells:
        want = set(args.cells.split(","))
        cells = OrderedDict((k, v) for k, v in cells.items() if k in want)

    from .run_eval import _load_vggt_base, make_vggt_predict
    from .adt_depth import run_adt_eval
    model = _load_vggt_base(args.vggt_checkpoint, device)
    predict = make_vggt_predict(model, device)

    results: Dict[str, dict] = {}
    for cid, cell in cells.items():
        print(f"\n{'=' * 74}\n[exp2x2] {cell['label']}  --  {cell['blurb']}\n{'=' * 74}")
        m = run_adt_eval(
            predict, cell["label"], seq_dirs, device,
            seq_len=args.seq_len, image_resolution=args.resolution,
            align_modes=("none", "scale_shift"), max_frames=args.max_frames,
            rgb_subdir=args.rgb_subdir, rectify=cell["rectify"],
            focal_out_norm=cell["focal_out_norm"], fill=cell["fill"],
        )
        fov = probe_fov(model, cell, seq_dirs, args.resolution, args.seq_len,
                        device, args.rgb_subdir)
        results[cid] = {"cell": {k: v for k, v in cell.items()},
                        "metrics": m, "fov_probe": fov}
        with open(os.path.join(args.out, "results.json"), "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"[exp2x2] {cell['label']} inferred FoV: {fov}")

    _report(results, args.out)


def _report(results: Dict[str, dict], out: str) -> None:
    """Print the cell table and, when all four cells ran, the interaction term."""
    def absrel(cid):
        try:
            return float(results[cid]["metrics"]["scale_shift"]["AbsRel"])
        except Exception:
            return float("nan")

    lines = ["", "=" * 78, "2x2 RESULTS  (AbsRel, scale-shift aligned, lower is better)", "=" * 78,
             f"{'cell':<18}{'AbsRel':>10}{'delta1':>10}{'fov_h°':>10}{'fov_w°':>10}"]
    for cid, r in results.items():
        ss = r["metrics"].get("scale_shift", {}) if isinstance(r["metrics"], dict) else {}
        fv = r.get("fov_probe") or {}
        lines.append(f"{cid:<18}{ss.get('AbsRel', float('nan')):>10.4f}"
                     f"{ss.get('delta1', float('nan')):>10.4f}"
                     f"{fv.get('fov_h_deg_mean', float('nan')):>10.1f}"
                     f"{fv.get('fov_w_deg_mean', float('nan')):>10.1f}")
    need = ["1_raw_black", "2_rect_black", "3_raw_filled", "4_rect_filled"]
    if all(c in results for c in need):
        d_rect = absrel("4_rect_filled") - absrel("2_rect_black")   # fill effect, rectified
        d_raw = absrel("3_raw_filled") - absrel("1_raw_black")      # fill effect, raw
        lines += ["", f"fill effect | rectified  (4-2): {d_rect:+.4f}",
                  f"fill effect | raw fisheye (3-1): {d_raw:+.4f}",
                  f"INTERACTION      (4-2)-(3-1): {d_rect - d_raw:+.4f}",
                  "  negative => filling helps MORE in the perspective domain "
                  "(the hypothesis)"]
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(out, "report.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
