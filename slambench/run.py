# Copyright (c) 2026.
"""Score depth models against ego-synth 5B's SLAM ground truth.

    python -m slambench.run --egosynth-root $EGOSYNTH --models vggt_1b --baselines raw

**Not the ADT-FOV experiment.** There is no eccentricity axis here and no
binning; this asks how accurate each model is on real egocentric footage, full
stop. See ``slambench/__init__.py``.

The grid is

    model x baseline x dataset

    models     vggt_1b | vggt_omega | dav2_large | da3_small | da3_large
    baselines  raw          the fisheye frame as it is
               rect_derect  rectify, predict, map the depth back -- inside the
                            baseline, so the harness never learns it happened
    datasets   aea | nymeria | egoexo4d | oxford, reported separately because
               their scene scales differ by an order of magnitude and every
               metric here is relative

Both baselines are scored on the **same points**. A pinhole cannot cover the
whole fisheye cone, so ``rect_derect`` has no answer at the rim; scoring the two
over different point sets would compare the sets as much as the baselines. The
intersection is what both are scored on, and what each gave up to reach it is
reported beside the numbers rather than folded into them.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from slambench import _REPO  # noqa: F401  (import registers sys.path)

from slambench import baselines as B      # noqa: E402
from slambench import camera as C         # noqa: E402
from slambench import data as D           # noqa: E402
from slambench import metrics as MT       # noqa: E402
from slambench import models as M         # noqa: E402
from slambench import report as R         # noqa: E402
from slambench import split as S          # noqa: E402


def _load_camera(calib_root: str, dataset: str, take: str, verify: bool
                 ) -> Optional[C.Fisheye624]:
    path = C.calibration_path(calib_root, dataset, take)
    if not os.path.isfile(path):
        raise C.CalibrationUnavailable(
            f"no camera model at {path}. Fetch it with "
            f"tools/fetch_egosynth_calibration.py (ticket 012), or run "
            f"--baselines raw, which needs none.")
    cam = C.load(path, dataset=dataset, take=take, out_size=D.RES)
    return C.require_verified(cam) if verify else cam


def run(a: argparse.Namespace) -> dict:
    os.makedirs(a.out, exist_ok=True)
    datasets = [d.strip() for d in a.datasets.split(",") if d.strip()]
    arms = [b.strip() for b in a.baselines.split(",") if b.strip()]
    bad = [b for b in arms if b not in B.BASELINES]
    if bad:
        raise SystemExit(f"[slambench] unknown baseline(s) {bad}; "
                         f"choose from {list(B.BASELINES)}")
    if B.RECT_DERECT in arms and not a.calib_root:
        raise SystemExit(
            f"[slambench] --baselines {B.RECT_DERECT} needs --calib-root: it maps "
            f"predictions through each take's own camera model, and there is no "
            f"nominal Aria calibration to fall back on (a nominal one is wrong by "
            f"more than the effect being measured). See ticket 012.")
    keys = [k.strip() for k in a.models.split(",") if k.strip()]

    ready, skipped = M.available(keys)
    for key, state, detail in skipped:
        print(f"[slambench] {key}: {state} — {detail}")
    if skipped and not a.skip_unavailable:
        raise SystemExit(
            f"[slambench] {len(skipped)} of {len(keys)} requested models cannot "
            f"run ({', '.join(k for k, _, _ in skipped)}). Fix them with the "
            f"instructions above, or pass --skip-unavailable to run the rest "
            f"(the report and results.json then record what was left out).")
    if not ready:
        raise SystemExit("[slambench] no model is runnable; see above. Use "
                         "--models analytic for a weight-free harness run.")

    sp = (S.Split.load(a.manifest) if a.manifest
          else S.build(a.egosynth_root, datasets, a.n_frames, a.takes))
    if sp.protocol != S.PROTOCOL:
        raise SystemExit(f"[slambench] --manifest {a.manifest} was written by "
                         f"{sp.protocol!r}, not {S.PROTOCOL!r}")
    sp.save(os.path.join(a.out, "manifest.json"))

    device = None
    if [k for k in ready if k not in M.STANDINS]:
        import torch
        device = torch.device(a.device)
        if device.type != "cuda":
            print("[slambench] WARNING: real weights on CPU — minutes per frame. "
                  "Use --device cuda on the GPU box.")

    clips = sp.by_clip()
    runs: List[dict] = []
    for key in ready:
        print(f"\n[slambench] ══ {key} ══")
        t0 = time.time()
        model = M.load_model(key, device, checkpoint=a.omega_checkpoint,
                             bias=a.oracle_bias)
        MT.check_protocol(key, model.align_mode)
        print(f"[slambench]   {model.family} {model.size} | {model.params_m:.0f}M "
              f"| align={model.align_mode} | frames at {model.input_size}px")

        # scores[(dataset, arm)] -> [per-frame metrics]; kept[(dataset, arm)]
        # -> the share of points that arm could answer for, before intersecting.
        scores: Dict[tuple, list] = {}
        kept: Dict[tuple, list] = {}
        thin = 0
        for ci, (ds, take, clip, npz, video, idxs) in enumerate(clips):
            cam = None
            if B.RECT_DERECT in arms:
                cam = _load_camera(a.calib_root, ds, take, not a.allow_unverified)
            arm_objs = {n: B.build(n, model, cam, a.rect_fov) for n in arms}
            frames = D.decode_frames(video, idxs)
            for i in idxs:
                pts = D.read_points(npz, i, sigma_max=a.sigma_max)
                if len(pts) < D.MIN_FRAME_POINTS:
                    thin += 1
                    continue
                preds = {n: arm_objs[n].predict(frames[i], pts) for n in arms}
                # Both arms on the points both could answer for.
                support = np.ones(len(pts), bool)
                for p in preds.values():
                    support &= np.isfinite(p) & (p > 0)
                for n, p in preds.items():
                    own = np.isfinite(p) & (p > 0)
                    kept.setdefault((ds, n), []).append(float(own.mean()))
                    met = MT.score_frame(np.where(support, p, np.nan), pts.d,
                                         model.align_mode,
                                         max_depth=a.max_depth,
                                         min_points=a.min_points)
                    if met is None:
                        thin += 1
                        continue
                    scores.setdefault((ds, n), []).append(met)
            if (ci + 1) % max(1, a.log_every) == 0:
                print(f"[slambench]   clip {ci + 1}/{len(clips)} "
                      f"({time.time() - t0:.0f}s)")

        for (ds, arm), rows in sorted(scores.items()):
            agg = MT.aggregate(rows)
            agg["coverage"] = float(np.mean(kept.get((ds, arm), [np.nan])))
            runs.append(dict(model=key, family=model.family, size=model.size,
                             params_m=model.params_m, align=model.align_mode,
                             input_size=model.input_size, dataset=ds,
                             baseline=arm, **agg))
        if thin:
            print(f"[slambench]   {thin} frame(s) under {a.min_points} usable "
                  f"points were not scored")
        del model
        if device is not None and device.type == "cuda":
            import torch
            torch.cuda.empty_cache()
        print(f"[slambench]   done in {time.time() - t0:.0f}s")

    payload = dict(
        protocol=S.PROTOCOL, digest=sp.digest, egosynth_root=sp.root,
        n_frames=len(sp.frames), datasets=sp.datasets, takes=sp.takes,
        requested_models=keys,
        skipped_models=[dict(model=k, state=s, detail=d) for k, s, d in skipped],
        config=dict(baselines=arms, datasets=datasets,
                    takes_per_dataset=sp.takes_per_dataset,
                    n_frames_per_take=sp.n_frames_per_take,
                    # The release ships unfiltered on purpose, so the cut is part
                    # of the result and travels with it.
                    sigma_max=a.sigma_max,
                    sigma_column="inv_dist_std (1/m, scale-invariant)",
                    gt_variant=D.VARIANT, max_depth=a.max_depth,
                    min_points=a.min_points, rect_fov=a.rect_fov,
                    calib_root=a.calib_root or None,
                    orientation_verified=list(C.VERIFIED_ROTATION)),
        runs=runs)
    with open(os.path.join(a.out, "results.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    R.write_all(payload, a.out)
    print(f"\n[slambench] wrote {a.out}/results.json (+ csv, report.txt)")
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--egosynth-root", default=os.environ.get("EGOSYNTH", ""),
                   help="ego-synth 5B root (aea/ nymeria/ egoexo4d/ oxford/)")
    p.add_argument("--calib-root", default=os.environ.get("EGOSYNTH_CALIB", ""),
                   help="per-take camera models from ticket 012; required by "
                        "the rect_derect baseline, unused by raw")
    p.add_argument("--manifest", default=None,
                   help="reuse a frozen split instead of rebuilding it")
    p.add_argument("--datasets", default=",".join(D.DATASETS))
    p.add_argument("--models", default=",".join(M.DEFAULT_MODELS),
                   help=f"comma keys, or {M.ANALYTIC!r}/{M.ORACLE!r} for a "
                        f"weight-free run")
    p.add_argument("--baselines", default=",".join(B.BASELINES))
    p.add_argument("--n-frames", type=int, default=25,
                   help="frames PER TAKE, spread over its clips (not a prefix)")
    p.add_argument("--takes", type=int, default=8,
                   help="takes PER DATASET (0 = all). The release is 1 611 "
                        "takes; the cap enters the split digest")
    p.add_argument("--sigma-max", type=float, default=D.DEFAULT_SIGMA_MAX,
                   help="drop points whose MPS inv_dist_std (1/m) is at or above "
                        "this. The GT ships UNFILTERED by design; this cut is "
                        "written into results.json with every number it produced")
    p.add_argument("--max-depth", type=float, default=MT.MAX_DEPTH_M,
                   help="GT validity ceiling (m); generous because Oxford is "
                        "outdoors with a 23 m p99")
    p.add_argument("--min-points", type=int, default=D.MIN_FRAME_POINTS,
                   help="a frame with fewer usable points is not scored: the "
                        "alignment affine is fitted over the whole frame")
    p.add_argument("--rect-fov", type=float, default=B.DEFAULT_RECT_FOV_DEG,
                   help="field of view rect_derect rectifies to (deg)")
    p.add_argument("--allow-unverified", action="store_true",
                   help="run rect_derect with a sensor-to-upright rotation that "
                        "has not passed verify_camera. A quarter-turn error does "
                        "not degrade the score, it scores a different part of "
                        "the image — so this is for debugging, not for numbers")
    p.add_argument("--oracle-bias", type=float, default=0.0,
                   help="multiplicative error injected by --models oracle, for "
                        "checking the harness reports what it was given")
    p.add_argument("--omega-checkpoint", default=None)
    p.add_argument("--skip-unavailable", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--out", default="eval_out/slambench")
    return p


def main() -> None:
    a = build_parser().parse_args()
    if not a.egosynth_root and not a.manifest:
        raise SystemExit("[slambench] pass --egosynth-root (or $EGOSYNTH), "
                         "or --manifest")
    run(a)


if __name__ == "__main__":
    main()
