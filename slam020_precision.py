# Copyright (c) 2026.
"""VGGT-1B is the only one of the four backbones running in fp32. What does it cost?

`raytun3r.backbones`:

  * `DA3Backbone.forward`      -- explicit `torch.autocast(cuda, bfloat16)`
  * `VGGTOmegaBackbone.forward`-- `self.model(images)`, and vggt_omega's own
                                  forward opens `torch.autocast(device_type="cuda")`
  * `VGGTBackbone.forward`     -- `self.model(images)`, and the vendored
                                  `vggt_visfeat` opens **no** autocast, then
                                  runs its heads under `autocast(enabled=False)`

So three of the four run their aggregator in bf16 and VGGT-1B runs it in fp32.
On top of that, torch defaults `allow_tf32=False` for matmul, so VGGT-1B's fp32
is *true* fp32 rather than the tensor-core path.

This measures both levers on speed AND on the depth itself, because a speedup
that moves the published numbers is not free -- it is a re-run.

    A  baseline        as published
    B  +tf32           TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, no code change
    C  +bf16 autocast  what the other three already do, via monkeypatch here
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slambench import data as D       # noqa: E402
from slambench import models as M     # noqa: E402
from slambench import split as S      # noqa: E402

SIZES = (1, 3, 5, 10)


def sync():
    torch.cuda.synchronize()


def timeit(fn, reps=3):
    fn()
    sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def bf16_patch(model):
    """Wrap the backbone call in bf16 autocast, as DA3Backbone already does."""
    bb = model_backbone(model)
    orig = bb.__class__.forward

    def fwd(self, images):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return orig(self, images)
    bb.__class__.forward = fwd
    return lambda: setattr(bb.__class__, "forward", orig)


def model_backbone(model):
    """slambench.models wraps zoo -> adapter -> backbone; reach it via the closure.

    The free variables are looked up BY NAME rather than by position, because
    the order of `co_freevars` is an implementation detail and picking [0] gave
    `_as01` on this build.
    """
    fn = model._predict_stack
    names = fn.__code__.co_freevars
    env = dict(zip(names, (c.cell_contents for c in fn.__closure__)))
    adapter = env.get("adapter")
    if adapter is None or not hasattr(adapter, "backbone"):
        raise SystemExit(f"[precision] no adapter with a .backbone in {names}")
    return adapter.backbone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--egosynth-root", default=os.environ.get("EGOSYNTH", ""))
    ap.add_argument("--model", default="vggt_1b")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()

    dev = torch.device("cuda")
    tf32 = torch.backends.cuda.matmul.allow_tf32
    label = "B +tf32" if tf32 else "A baseline"
    print(f"# {label}   matmul.allow_tf32={tf32}   torch {torch.__version__}")

    sp = S.build(a.egosynth_root, ["aea"], 25, 1)
    _ds, _take, _clip, _npz, video, frs = sp.by_clip()[0]
    want = sorted(range(min(10, frs[0].clip_frames)))
    frames = D.decode_frames(video, want)

    model = M.load_model(a.model, dev, checkpoint=os.environ.get("VGGT_OMEGA_CKPT"))
    stack = [D.resize_frame(frames[i], model.input_size) for i in want]

    def measure(tag):
        ts, ds = [], []
        for s in SIZES:
            sub = stack[:s]
            ts.append(timeit(lambda sub=sub: model.predict_stack(sub, target=len(sub) - 1),
                             reps=a.reps))
            ds.append(model.predict_stack(sub, target=len(sub) - 1))
        print(f"  {tag:16s}" + "".join(f"{v * 1000:10.1f}" for v in ts)
              + f"{sum(ts) * 2 * 400 / 60:11.1f} min")
        return ts, ds

    print(f"\n  {'setting':16s}" + "".join(f"{f'S={s}':>10s}" for s in SIZES)
          + f"{'grid cost':>11s}")
    print("  " + "-" * 68)
    t_base, d_base = measure(label)

    undo = bf16_patch(model)
    t_bf, d_bf = measure("C +bf16" + ("+tf32" if tf32 else ""))
    undo()

    print(f"\n  speedup from bf16: "
          + ", ".join(f"S={s} {tb / tf_:.2f}x" for s, tb, tf_ in zip(SIZES, t_base, t_bf)))

    print("\n  DOES THE DEPTH MOVE?  relative difference of the returned map")
    print(f"  {'window':>8s}{'median |rel|':>15s}{'p99 |rel|':>12s}{'max |rel|':>12s}")
    for s, b, f in zip(SIZES, d_base, d_bf):
        r = np.abs(f - b) / np.maximum(np.abs(b), 1e-6)
        print(f"  {s:8d}{np.median(r):15.2e}{np.percentile(r, 99):12.2e}{r.max():12.2e}")
    print("\n  A per-frame affine is fitted before scoring, so a uniform scale")
    print("  change would cancel; what matters is the spatial pattern, which is")
    print("  why the spread and not just the median is printed.")


if __name__ == "__main__":
    main()
