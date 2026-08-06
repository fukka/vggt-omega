"""Is the evaluation data itself right? Run this before trusting any number.

**The question.** ``vanilla`` is the easiest thing in the paper to reproduce: no
adapter, no training, no randomness -- a frozen backbone on a posed pair. If our
``vanilla`` does not match the paper's, nothing downstream means anything. On
ScanNet++ ``3f15`` with VGGT the paper reports ``R = 7.21``; we measure 0.554 at
stride 1 and 2.379 at stride 10. That gap has to be explained before the adapter
is worth discussing.

**The key point about ``R_err``.** It is an *absolute* angular error, so its scale
is set by how much rotation there is to estimate. A model that predicts identity
scores exactly the mean GT rotation magnitude. Two runs' ``R_err`` are therefore
only comparable when their GT rotation distributions match -- and ``--stride``
changes that distribution directly. This script reports the identity-predictor
score alongside everything else, because ``vanilla`` only carries information to
the extent it beats that number.

**What it checks**, all from ``transforms.json`` alone -- no images, no weights,
no GPU, a few seconds:

1. Keys the loader ignores (``test_frames``, ``applied_transform``,
   ``applied_scale``, per-frame ``mask_path``). Any of these being present and
   ignored is a candidate for a silent mismatch with the paper.
2. The field of view implied by the intrinsics, at the frame edges and corner.
   This settles the paper's stated 115 deg against the ~170 deg we derive.
3. Whether the poses are metric. Camera-centre extent and path length say whether
   "1.09 cm between consecutive frames" is centimetres or normalised units.
4. Per stride: baseline, **GT rotation magnitude (= the identity-predictor
   ``R_err``)**, and which stride would put us in the paper's regime.

Usage::

    python -m raytun3r.experiments.data_audit --path /netapp/datasets/f.zhang2/scannetpp/data/3f15a9266d

Add ``--json out.json`` to write a machine-readable copy for the results branch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import List

import torch

__all__ = ["main"]

#: What the paper reports for this scene, to compare the regime against.
#: VGGT on ScanNet++ 3f15, Tab. 2: R, t. DA3-Small Tab. 5 raytun3r: 0.40 / 2.2.
PAPER_VGGT_VANILLA = (7.21, 16.6)


def _rot_angle_deg(R: torch.Tensor) -> float:
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    return math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0))))


def _fmt(v: List[float]) -> str:
    if not v:
        return "n/a"
    s = sorted(v)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    return (f"median {q(0.5):8.4f}   mean {sum(s)/len(s):8.4f}   "
            f"p10 {q(0.1):8.4f}   p90 {q(0.9):8.4f}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser("raytun3r.experiments.data_audit", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", required=True, help="ScanNet++ scene directory")
    p.add_argument("--strides", default="1,5,10,20,40,60,80")
    p.add_argument("--json", default=None, help="also write the report here")
    args = p.parse_args(argv)

    from ..data import ScanNetPPFisheye

    out = {"scene": os.path.basename(args.path.rstrip("/"))}

    # -- 1. what transforms.json holds that the loader does not read ----------
    tpath = os.path.join(args.path, "dslr", "nerfstudio", "transforms.json")
    with open(tpath) as f:
        meta = json.load(f)
    frames = meta.get("frames", [])
    ignored = {k: (len(meta[k]) if isinstance(meta.get(k), list) else meta.get(k))
               for k in ("test_frames", "applied_transform", "applied_scale")
               if k in meta}
    per_frame_keys = sorted({k for fr in frames[:50] for k in fr})
    out["transforms"] = {
        "camera_model": meta.get("camera_model"), "w": meta.get("w"), "h": meta.get("h"),
        "n_frames": len(frames), "ignored_top_level_keys": ignored,
        "per_frame_keys": per_frame_keys,
    }
    print(f"=== 1. transforms.json  ({tpath})")
    print(f"  camera_model : {meta.get('camera_model')}")
    print(f"  resolution   : {meta.get('w')} x {meta.get('h')}")
    print(f"  frames       : {len(frames)}")
    for k, v in ignored.items():
        print(f"  !! IGNORED by the loader: {k} = {v}")
    if not ignored:
        print("  (no ignored top-level keys)")
    print(f"  per-frame keys: {per_frame_keys}")
    if "mask_path" in per_frame_keys:
        print("  !! frames carry mask_path -- ScanNet++ ships per-frame masks and the")
        print("     loader ignores them. If those masks define the valid fisheye region,")
        print("     the paper's Omega is the mask and ours is the whole rectangle.")

    # -- 2. field of view actually implied by the intrinsics ------------------
    src = ScanNetPPFisheye(args.path)
    cam = src.camera                       # already at the working resolution
    W, H = cam.width, cam.height
    rx = max(cam.cx, W - 1 - cam.cx) / cam.fx
    ry = max(cam.cy, H - 1 - cam.cy) / cam.fy
    def th(r):
        return math.degrees(float(cam.theta_of_r(torch.tensor([float(r)], dtype=torch.float64))[0]))
    horiz, vert, diag = th(rx), th(ry), th(math.hypot(rx, ry))
    out["fov_deg"] = {"horizontal_total": 2 * horiz, "vertical_total": 2 * vert,
                      "diagonal_total": 2 * diag,
                      "theta_max_deg": math.degrees(cam.theta_max),
                      "working_resolution": [W, H]}
    print(f"\n=== 2. field of view from the intrinsics (working res {W}x{H})")
    print(f"  fx,fy = {cam.fx:.2f},{cam.fy:.2f}   cx,cy = {cam.cx:.2f},{cam.cy:.2f}   k = {cam.k}")
    print(f"  horizontal total FOV : {2*horiz:7.2f} deg   (half {horiz:.2f})")
    print(f"  vertical   total FOV : {2*vert:7.2f} deg   (half {vert:.2f})")
    print(f"  diagonal   total FOV : {2*diag:7.2f} deg   (half {diag:.2f})")
    print(f"  theta_max (our Omega): {math.degrees(cam.theta_max):7.2f} deg half-angle")
    print(f"  paper states 115 deg for ScanNet++. Which of the above, if any, is it?")

    # -- 3. are the poses metric? --------------------------------------------
    P = [src.pose(i) for i in range(len(src))]
    ok = [i for i, x in enumerate(P) if x is not None]
    C = {i: (-P[i][0].double().T @ P[i][1].double()) for i in ok}
    stack = torch.stack([C[i] for i in ok])
    extent = (stack.max(0).values - stack.min(0).values).tolist()
    path_len = float(sum((C[b] - C[a]).norm() for a, b in zip(ok, ok[1:])))
    out["poses"] = {"n_posed": len(ok), "bbox_extent": extent, "path_length": path_len}
    print(f"\n=== 3. pose sanity")
    print(f"  posed frames    : {len(ok)} / {len(src)}")
    print(f"  camera bbox     : {extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f}")
    print(f"  total path      : {path_len:.3f}")
    print(f"  -> if the bbox is metres this is a room; if it is ~1 the poses are")
    print(f"     normalised and every distance quoted so far is in the wrong unit.")

    # -- 4. the regime, per stride -------------------------------------------
    print(f"\n=== 4. per-stride regime  (identity-predictor R_err == median GT rotation)")
    print(f"  {'stride':>7} {'pairs':>7} | {'GT rotation (deg) = identity-predictor R_err':^58}")
    print(f"  {'':>7} {'':>7} | {'baseline':^58}")
    rows = {}
    for s in [int(x) for x in args.strides.split(",") if x.strip()]:
        pairs = [(a, b) for a, b in zip(ok, ok[s:])]
        if not pairs:
            continue
        rot = [_rot_angle_deg(P[b][0].double() @ P[a][0].double().T) for a, b in pairs]
        base = [float((C[b] - C[a]).norm()) for a, b in pairs]
        rows[s] = {"n_pairs": len(pairs),
                   "gt_rotation_deg_median": sorted(rot)[len(rot) // 2],
                   "gt_rotation_deg_mean": sum(rot) / len(rot),
                   "baseline_median": sorted(base)[len(base) // 2]}
        print(f"  {s:7d} {len(pairs):7d} | rot  {_fmt(rot)}")
        print(f"  {'':>7} {'':>7} | base {_fmt(base)}")
    out["per_stride"] = rows

    # -- the conclusion the script exists to reach ---------------------------
    print(f"\n=== 5. which stride puts us in the paper's regime?")
    print(f"  paper: VGGT vanilla on this scene = {PAPER_VGGT_VANILLA[0]} deg R_err.")
    print(f"  vanilla can only score that on pairs whose GT rotation is at least")
    print(f"  comparable. Pick the stride whose median GT rotation is nearest it:")
    if rows:
        best = min(rows, key=lambda s: abs(rows[s]["gt_rotation_deg_median"] - PAPER_VGGT_VANILLA[0]))
        print(f"    -> stride {best}  (median GT rotation "
              f"{rows[best]['gt_rotation_deg_median']:.3f} deg)")
        print(f"  If even the largest stride here stays far below {PAPER_VGGT_VANILLA[0]} deg,")
        print(f"  then no stride reproduces the paper on this frame set, and the frame")
        print(f"  set itself differs from theirs -- that is the finding, and it is worth")
        print(f"  more than any further sweep.")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[data-audit] wrote {args.json}")


if __name__ == "__main__":
    main()
