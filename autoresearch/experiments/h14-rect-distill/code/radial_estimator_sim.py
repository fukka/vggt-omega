"""H18.5, in advance: what does the 16-number estimator actually need?

lambda_63 has been unreachable for hours and H18.5 needs a GPU pass. But its
central question — "is the binding constraint more pixels or more viewpoints?"
— is a question about the ESTIMATOR, and that can be characterised here.

The mechanism this tests. What identifies the per-bin slope `a` is the spread of
`log(pred)` WITHIN that bin. A frame is one scene view, so its pixels occupy a
narrow band of depth; adding pixels from the same frame packs more points into
that band and barely widens it, while adding a frame from a different viewpoint
moves the band. If that is right, frames-from-different-viewpoints buy slope
precision and pixels-from-one-viewpoint do not, and the gap between the two
grows as the viewpoints spread.

Generative model, chosen to have the structure that matters and nothing else:

    frame f:  band centre c_f, pixels' log(pred) ~ U(c_f - w/2, c_f + w/2)
              per-frame bias e_f ~ N(0, sigma_frame)      <- scene-level, not pixel-level
    pixel:    log(target) = a(theta)*log(pred) + b(theta) + e_f + N(0, sigma_px)

    `spread`  c_f ~ U over the full depth range      (varied viewpoints)
    `single`  c_f ~ U over a narrow slice of it      (one sequence, similar views)

`sigma_frame` is what makes the FRAME rather than the pixel the unit of
information: without it, 20k pixels from one frame would count as 20k samples
and the answer would be trivially "pixels".

**This is a characterisation of the estimator, not a result about ADT.** What it
produces is a falsifiable ordering for H18.5 to check against real data, plus
the pixel-count sweep that says whether pixels-per-frame matter at all once a
few thousand are in hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_BINS = 8
# The Apartment fit, which H18.6 showed is well described by a quadratic.
A_TRUE = np.polyval(np.polyfit((np.arange(8) + .5) / 8,
                               [1.3373, 1.4295, 1.4681, 1.4605,
                                1.4534, 1.4336, 1.3999, 1.3412], 2),
                    (np.arange(8) + .5) / 8)
B_TRUE = np.linspace(0.012, -0.252, N_BINS)
LO, HI = np.log(0.4), np.log(10.0)          # the global log-depth range


def draw(rng, n_frames, px, mode, w, sigma_frame, sigma_px):
    """One fitting set: n_frames frames of `px` pixels each."""
    if mode == "spread":
        centres = rng.uniform(LO + w / 2, HI - w / 2, n_frames)
    else:                                    # one sequence: similar viewpoints
        mid = (LO + HI) / 2
        centres = rng.uniform(mid - 0.35 * (HI - LO) / 2 + w / 2,
                              mid + 0.35 * (HI - LO) / 2 - w / 2, n_frames)
    lp, lt, bn = [], [], []
    for c in centres:
        p = rng.uniform(c - w / 2, c + w / 2, px)
        b = rng.integers(0, N_BINS, px)
        e = rng.normal(0, sigma_frame)       # one bias for the whole frame
        lp.append(p)
        lt.append(A_TRUE[b] * p + B_TRUE[b] + e + rng.normal(0, sigma_px, px))
        bn.append(b)
    return np.concatenate(lp), np.concatenate(lt), np.concatenate(bn)


def fit(lp, lt, bn):
    C = np.zeros(N_BINS)
    for i in range(N_BINS):
        m = bn == i
        if m.sum() < 20:
            C[i] = 1.0
            continue
        X = np.stack([lp[m], np.ones(m.sum())], 1)
        C[i] = np.linalg.lstsq(X, lt[m], rcond=None)[0][0]
    return C


def stability(rng, n_frames, px, mode, draws, **kw):
    """Mean pairwise |da| between independent fitting sets — the same readout
    H18.5 uses on real data, so the two are directly comparable."""
    D = np.stack([fit(*draw(rng, n_frames, px, mode, **kw)) for _ in range(draws)])
    return float(np.mean([np.abs(D[i] - D[j]).mean()
                          for i in range(len(D)) for j in range(i + 1, len(D))]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", default="2,4,8,15,30,60,120,240,480")
    p.add_argument("--px", type=int, default=20000)
    p.add_argument("--band-width", type=float, default=0.9,
                   help="log-depth span a single frame covers (0.9 ~ a 2.5x "
                        "near-to-far ratio within one view)")
    p.add_argument("--sigma-frame", type=float, default=0.06)
    p.add_argument("--sigma-px", type=float, default=0.30)
    p.add_argument("--draws", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    kw = dict(w=a.band_width, sigma_frame=a.sigma_frame, sigma_px=a.sigma_px)
    counts = [int(x) for x in a.frames.split(",")]
    res = {"spread": {}, "single": {}}

    print(f"[sim] {a.px} px/frame, band {a.band_width:.2f} in log-depth, "
          f"sigma_frame {a.sigma_frame}, sigma_px {a.sigma_px}, {a.draws} draws")
    print(f"\n{'frames':>8s}{'spread |da|':>14s}{'single |da|':>14s}{'ratio':>8s}")
    for n in counts:
        rng = np.random.default_rng(a.seed + n)
        s = stability(rng, n, a.px, "spread", a.draws, **kw)
        rng = np.random.default_rng(a.seed + n)
        g = stability(rng, n, a.px, "single", a.draws, **kw)
        res["spread"][n], res["single"][n] = s, g
        print(f"{n:>8d}{s:>14.4f}{g:>14.4f}{g / max(s, 1e-9):>8.2f}")

    def crossing(d):
        ks = sorted(d)
        for k in ks:
            if d[k] < 0.10:
                return k
        return None
    cs, cg = crossing(res["spread"]), crossing(res["single"])
    print(f"\n[sim] |da| < 0.10 reached at: spread {cs} frames, single {cg} frames")

    # does adding pixels to a fixed frame budget help at all?
    print(f"\n{'px/frame':>9s}{'|da| @ 60 spread frames':>26s}")
    px_res = {}
    for px in (500, 2000, 8000, 20000, 60000):
        rng = np.random.default_rng(a.seed + 99)
        v = stability(rng, 60, px, "spread", a.draws, **kw)
        px_res[px] = v
        print(f"{px:>9d}{v:>26.4f}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"stability": res, "crossing": {"spread": cs, "single": cg},
             "px_sweep": px_res, "config": vars(a)}, indent=1))


if __name__ == "__main__":
    main()
