"""Cache VGGT's predictions once, so a LADDER of students can be trained against
the same teacher without re-paying 70 ms/frame per architecture per epoch.

Measured on lambda_63: VGGT-1B is 1157.9 M params, 5.50 GB peak, 70 ms/frame at
504 px. Running it online would add that to every step of every student; caching
makes student training teacher-free, which is what makes comparing several
architectures affordable at all.

Stored per frame, float16:
    depth  (H, W)   planar z, the convention the whole repo uses (ticket 016)
    conf   (H, W)   the teacher's own uncertainty

`conf` is cached deliberately. This project's finding is that models degrade
toward the rim; whether the teacher KNOWS it does is a separate question, and a
student can be taught the teacher's uncertainty as well as its answer. Throwing
it away here would make that experiment impossible later without a re-run.

Depth is stored in the teacher's NATIVE convention and the convention is written
into the manifest, not assumed. The repo has already paid for this once: two
producers disagreed on z-vs-range and the resulting radial warp was an order of
magnitude larger than the effect being measured.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[3]))
sys.path.insert(0, str(_HERE.parents[1] / "h1-rim-pose-value" / "code"))

from adt_pose_value import AriaLocalPairs  # noqa: E402


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--backbone", default="vggt")
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--convention", default="z",
                   help="stored as-is and recorded; 'z' is the repo's planar-z")
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    src = AriaLocalPairs(os.path.expanduser(a.seq), size=a.size)
    idx = list(range(0, len(src.paths), a.stride))
    out = Path(a.out); (out / "npz").mkdir(parents=True, exist_ok=True)

    from raytun3r.backbones import build_backbone
    bb = build_backbone(a.backbone, weights="pretrained", device=a.device)
    bb.install(None, src.camera, (a.size, a.size), patch_undistort=False,
               border_token=False, dpt_grid=False, depth_convention=a.convention)
    net = bb.model if hasattr(bb, "model") else bb
    n_par = sum(q.numel() for q in net.parameters())
    print(f"[cache] {a.backbone} {n_par/1e6:.1f}M params -> {out}", flush=True)

    t0 = time.time()
    done = 0
    for k in idx:
        stem = os.path.splitext(os.path.basename(src.paths[k]))[0]
        f = out / "npz" / f"{stem}.npz"
        if f.exists():
            done += 1; continue
        with torch.no_grad():
            pr = bb.forward(src.image(k)[None, None].to(a.device))
        pr.require_convention(a.convention)
        np.savez_compressed(
            f,
            depth=pr.depth[0].cpu().numpy().astype(np.float16),
            conf=pr.conf[0].cpu().numpy().astype(np.float16))
        done += 1
        if done % 200 == 0:
            el = time.time() - t0
            print(f"[cache] {done}/{len(idx)}  {el:.0f}s  "
                  f"{1000*el/max(done,1):.0f} ms/frame", flush=True)

    (out / "manifest.json").write_text(json.dumps({
        "seq": a.seq, "backbone": a.backbone, "params_m": n_par / 1e6,
        "size": a.size, "stride": a.stride,
        "depth_convention": a.convention,
        "dtype": "float16", "fields": ["depth", "conf"],
        "n_frames": len(idx),
        "note": "depth is the teacher's NATIVE convention, recorded not assumed",
    }, indent=1))
    print(f"[cache] done {done} frames in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
