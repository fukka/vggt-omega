# Copyright (c) 2026.
"""Where the SLAM grid's wall clock actually goes.

Splits the run into the three things it does per scored frame -- decode the
video, rectify for the rect_derect arm, and run the model -- and times each
separately, so "vggt_1b is 5x vggt_omega under context" can be attributed
rather than guessed at.

Times the model at S = 1, 3, 5, 10 because the grid's cost is dominated by how
each backbone scales with the window, not by its single-frame cost.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slambench import baselines as B   # noqa: E402
from slambench import camera as C      # noqa: E402
from slambench import data as D        # noqa: E402
from slambench import models as M      # noqa: E402
from slambench import split as S       # noqa: E402

SIZES = (1, 3, 5, 10)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timeit(fn, reps=3, warmup=1):
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--egosynth-root", default=os.environ.get("EGOSYNTH", ""))
    ap.add_argument("--calib-root", default=os.environ.get("EGOSYNTH_CALIB", ""))
    ap.add_argument("--models", default="vggt_1b,vggt_omega,da3_large,da3_small")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()

    dev = torch.device("cuda")
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    print(f"tf32 matmul={torch.backends.cuda.matmul.allow_tf32}  "
          f"cudnn={torch.backends.cudnn.allow_tf32}\n")

    # One real clip, so decode and rectify are timed on real data.
    sp = S.build(a.egosynth_root, ["aea"], 25, 1)
    ds, take, clip, npz, video, frs = sp.by_clip()[0]
    cam = C.load(C.calibration_path(a.calib_root, ds, take), dataset=ds,
                 take=take, out_size=D.RES)
    want = sorted(range(min(10, frs[0].clip_frames)))

    print("== HARNESS, per clip / per frame ==")
    t = timeit(lambda: D.decode_frames(video, want), reps=a.reps)
    print(f"  decode_frames  {len(want)} frames from mp4   {t * 1000:8.1f} ms"
          f"   ({t / len(want) * 1000:.1f} ms/frame)")
    frames = D.decode_frames(video, want)
    f0 = frames[want[0]]
    pts = D.read_points(npz, frs[0].index)
    t = timeit(lambda: D.read_points(npz, frs[0].index), reps=a.reps)
    print(f"  read_points    {len(pts)} points from npz  {t * 1000:8.1f} ms")

    rd = B.build(B.RECT_DERECT, M.load_model("analytic", None), cam, 110.0)
    g = rd.rectify(D.resize_frame(f0, 518))
    t = timeit(lambda: rd.rectify(D.resize_frame(f0, 518)), reps=a.reps)
    print(f"  rectify        one 518px frame            {t * 1000:8.1f} ms"
          f"   (grey {g[0].shape})")

    print("\n== MODEL FORWARD, ms per call, by window size ==")
    print(f"  {'model':12s}{'px':>5s}{'tok/fr':>8s}"
          + "".join(f"{f'S={s}':>10s}" for s in SIZES)
          + f"{'S10/S1':>9s}{'per-frame':>11s}")
    print("  " + "-" * 84)
    rows = {}
    for key in [k.strip() for k in a.models.split(",") if k.strip()]:
        model = M.load_model(key, dev, checkpoint=os.environ.get("VGGT_OMEGA_CKPT"))
        px = model.input_size
        try:
            ps = model_patch(model)
        except Exception:
            ps = 14
        tok = (px // ps) ** 2
        stack = [D.resize_frame(frames[i], px) for i in want]
        ts = []
        for s in SIZES:
            sub = stack[:s]
            ts.append(timeit(lambda sub=sub: model.predict_stack(sub, target=len(sub) - 1),
                             reps=a.reps))
        rows[key] = ts
        print(f"  {key:12s}{px:5d}{tok:8d}"
              + "".join(f"{v * 1000:10.1f}" for v in ts)
              + f"{ts[-1] / ts[0]:9.2f}{ts[-1] / 10 * 1000:10.1f}ms")
        del model
        torch.cuda.empty_cache()

    print("\n== WHAT THE GRID PAYS ==")
    print("  Per scored frame the run does 2 arms x {1,3,5,10} = 8 forward")
    print("  passes, 19 frame-inputs per arm. Predicted per-frame model cost:")
    for key, ts in rows.items():
        tot = 2 * sum(ts)
        print(f"  {key:12s}{tot * 1000:9.1f} ms/frame  ->  "
              f"{tot * 400 / 60:7.1f} min over the 400-frame grid")


def model_patch(model):
    return 16 if "omega" in model.key else 14


if __name__ == "__main__":
    main()
