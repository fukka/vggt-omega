# Copyright (c) 2026.
"""Score depth models against ego-synth 5B's SLAM ground truth.

    python -m slambench.run --egosynth-root $EGOSYNTH --models vggt_1b --baselines raw

**Not the ADT-FOV experiment.** There is no eccentricity axis here and no
binning; this asks how accurate each model is on real egocentric footage, full
stop. See ``slambench/__init__.py``.

The grid is

    model x baseline x context x dataset

    models     vggt_1b | vggt_omega | dav2_large | da3_small | da3_large
    baselines  raw          the fisheye frame as it is
               rect_derect  rectify, predict, map the depth back -- inside the
                            baseline, so the harness never learns it happened
    context    how many frames the model sees in ONE forward pass, of which
               exactly one is scored: --context-frames 1,3,5,10 sweeps it on a
               single model load. Monocular models are refused, not quietly
               handed one frame
    datasets   aea | nymeria | egoexo4d | oxford, reported separately because
               their scene scales differ by an order of magnitude and every
               metric here is relative

Every arm is scored on the **same points**. A pinhole cannot cover the whole
fisheye cone, so ``rect_derect`` has no answer at the rim; scoring the arms over
different point sets would compare the sets as much as the arms. The intersection
is what all of them are scored on, and what each gave up to reach it is reported
beside the numbers rather than folded into them. The context arms answer at the
same points as each other by construction — only the split's own frame is ever
scored — which is why the context is deliberately **not** in the split digest.
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


def _context_sizes(spec: str) -> List[int]:
    """``"1,3,5,10"`` -> ``[1, 3, 5, 10]``, sorted, deduplicated, all >= 1.

    A list rather than one value because the sweep is the experiment: 1 against
    3 against 5 against 10 has to be measured on **one** frozen frame list and
    one model load, or the difference between the arms is partly the difference
    between the runs.
    """
    out = sorted({int(t) for t in str(spec).split(",") if t.strip()})
    if not out or out[0] < 1:
        raise SystemExit(f"[slambench] --context-frames {spec!r}: expected one "
                         f"or more integers >= 1, e.g. '1,3,5,10'")
    return out


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
    ctxs = _context_sizes(a.context_frames)

    # The vggt360 arm brings its own backbone (VGGT-1B, for the attention its
    # fusion reads), so it is a lens strategy for that model and no other. Two
    # refusals, both up front, both against tables that would look complete:
    #   * paired with another model, every row would be labelled with a network
    #     that never ran;
    #   * given a temporal context, it would put 9xN views in one pass while the
    #     column header said N frames.
    if B.VGGT360 in arms:
        if not a.calib_root:
            raise SystemExit(
                f"[slambench] --baselines {B.VGGT360} needs --calib-root: it "
                f"warps through each take's own camera model in both directions. "
                f"See ticket 012.")
        wrong = [k for k in keys if k != "vggt_1b"]
        if wrong:
            raise SystemExit(
                f"[slambench] --baselines {B.VGGT360} runs the vendored VGGT-1B "
                f"itself — its fusion reads frame attention off a 37x37 patch "
                f"grid that no other backbone exposes. Asking for {wrong} would "
                f"report a {B.VGGT360} row per model, all of them the same "
                f"numbers, each labelled with a network that never ran. Use "
                f"--models vggt_1b, and run the others in a separate pass.")
        if max(ctxs) > 1:
            raise SystemExit(
                f"[slambench] --baselines {B.VGGT360} with --context-frames "
                f"{a.context_frames}: this arm already hands VGGT a nine-view "
                f"reconstruction of one frame, so an N-frame context is 9N views "
                f"in a single pass and not the temporal sweep the column claims. "
                f"Run it at --context-frames 1.")

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

    if max(ctxs) > 1:
        mono = [k for k in ready if not M.takes_context(k)]
        if mono:
            raise SystemExit(
                f"[slambench] --context-frames {a.context_frames} asks for a "
                f"multi-frame context, and {mono} cannot take one — they are "
                f"monocular. Scoring their target frame alone would fill the "
                f"table with rows that read 'context does not help' when nothing "
                f"was tried. Run them in a separate --context-frames 1 pass, or "
                f"drop them from --models.")

    sp = (S.Split.load(a.manifest) if a.manifest
          else S.build(a.egosynth_root, datasets, a.n_frames, a.takes))
    if sp.protocol != S.PROTOCOL:
        raise SystemExit(f"[slambench] --manifest {a.manifest} was written by "
                         f"{sp.protocol!r}, not {S.PROTOCOL!r}")
    if max(ctxs) > 1 and any(f.clip_frames < 1 for f in sp.frames):
        raise SystemExit(
            f"[slambench] --manifest {a.manifest} predates context support: its "
            f"frames carry no clip length, and a context window has to know "
            f"where the clip ends before it can be placed inside it. Guessing "
            f"from the frames present would shrink every window silently — the "
            f"split holds {sp.n_frames_per_take} of a clip's ~121 frames. "
            f"Rebuild the split from --egosynth-root; the digest is unchanged "
            f"by context, so the results stay comparable.")
    sp.save(os.path.join(a.out, "manifest.json"))

    device = None
    if [k for k in ready if k not in M.STANDINS]:
        import torch
        device = torch.device(a.device)
        if device.type != "cuda":
            print("[slambench] WARNING: real weights on CPU — minutes per frame. "
                  "Use --device cuda on the GPU box.")

    # The VGGT-360 pipeline is loaded once for the whole run: it is a 1.2 GB
    # backbone, and the only thing that varies per take is the lens it is
    # handed. Its per-take view maps are cached inside the lens adapter.
    v360_pipe = v360_cfg = None
    if B.VGGT360 in arms:
        from utils.pipeline import VGGT360Config, VGGT360Pipeline
        v360_cfg = VGGT360Config(
            fov=a.vggt360_fov, ring_tilt=a.vggt360_ring_tilt,
            n_ring=a.vggt360_n_ring, persp_size=a.vggt360_persp_size,
            adaptive=not a.vggt360_no_adaptive,
            sa_mask=not a.vggt360_no_sa_mask, fuse=a.vggt360_fuse,
            head=a.vggt360_head, dtype=a.vggt360_dtype).check()
        v360_pipe = VGGT360Pipeline(v360_cfg, device=str(device)).load()

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

        # scores[(dataset, arm, context)] -> [per-frame metrics]; kept[...] ->
        # the share of points that arm could answer for, before intersecting.
        scores: Dict[tuple, list] = {}
        kept: Dict[tuple, list] = {}
        thin = 0
        for ci, (ds, take, clip, npz, video, frs) in enumerate(clips):
            cam = None
            if B.RECT_DERECT in arms or B.VGGT360 in arms:
                cam = _load_camera(a.calib_root, ds, take, not a.allow_unverified)
            arm_objs = {n: B.build(n, model, cam, a.rect_fov,
                                   vggt360_pipe=v360_pipe, vggt360_cfg=v360_cfg)
                        for n in arms}
            if B.VGGT360 in arms:
                # How the ADT-designed layout sits on THIS take's lens, in the
                # log beside the numbers rather than reconstructed afterwards.
                from slambench.vggt360 import layout_report
                print(f"[slambench]   {ds}/{take}: "
                      f"{layout_report(v360_cfg, arm_objs[B.VGGT360].lens)}")
            # Every window of every context size, so the mp4 is decoded once for
            # all arms rather than once per arm.
            wins, want = {}, set()
            for f in frs:
                for n in ctxs:
                    w = S.context_window(f.clip_frames, f.index, n,
                                         a.context_stride)
                    wins[(f.index, n)] = w
                    want.update(w[0])
            frames = D.decode_frames(video, sorted(want))
            for f in frs:
                pts = D.read_points(npz, f.index, sigma_max=a.sigma_max)
                if len(pts) < D.MIN_FRAME_POINTS:
                    thin += 1
                    continue
                preds = {}
                for n in ctxs:
                    idx, tgt = wins[(f.index, n)]
                    stack = [frames[j] for j in idx]
                    for arm in arms:
                        preds[(arm, n)] = arm_objs[arm].predict(stack, pts, tgt)
                # Every arm on the points every arm could answer for: the same
                # rule the two baselines already shared, over the larger grid.
                # A context arm that answered on fewer points would otherwise be
                # compared against a different set, not a different context.
                support = np.ones(len(pts), bool)
                for p in preds.values():
                    support &= np.isfinite(p) & (p > 0)
                for (arm, n), p in preds.items():
                    own = np.isfinite(p) & (p > 0)
                    kept.setdefault((ds, arm, n), []).append(float(own.mean()))
                    met = MT.score_frame(np.where(support, p, np.nan), pts.d,
                                         model.align_mode,
                                         max_depth=a.max_depth,
                                         min_points=a.min_points)
                    if met is None:
                        thin += 1
                        continue
                    scores.setdefault((ds, arm, n), []).append(met)
            if (ci + 1) % max(1, a.log_every) == 0:
                print(f"[slambench]   clip {ci + 1}/{len(clips)} "
                      f"({time.time() - t0:.0f}s)")

        for (ds, arm, n), rows in sorted(scores.items()):
            agg = MT.aggregate(rows)
            agg["coverage"] = float(np.mean(kept.get((ds, arm, n), [np.nan])))
            runs.append(dict(model=key, family=model.family, size=model.size,
                             params_m=model.params_m, align=model.align_mode,
                             input_size=model.input_size, dataset=ds,
                             baseline=arm, context=n,
                             context_stride=int(a.context_stride), **agg))
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
                    # The context sweep. It is NOT in the split digest, on
                    # purpose: every arm scores the same points, so folding it
                    # in would give them different digests and make the harness
                    # refuse the one comparison the sweep exists to make.
                    context_frames=ctxs, context_stride=int(a.context_stride),
                    # The release ships unfiltered on purpose, so the cut is part
                    # of the result and travels with it.
                    sigma_max=a.sigma_max,
                    sigma_column="inv_dist_std (1/m, scale-invariant)",
                    gt_variant=D.VARIANT, max_depth=a.max_depth,
                    min_points=a.min_points, rect_fov=a.rect_fov,
                    calib_root=a.calib_root or None,
                    orientation_verified=list(C.VERIFIED_ROTATION),
                    # The lens-aware arm's configuration, recorded whether or
                    # not it ran: a vggt360 row whose layout is not written down
                    # beside it is not readable, and recording it only when
                    # present would make the two cases differ for the wrong
                    # reason.
                    vggt360=(v360_cfg.__dict__ if v360_cfg is not None
                             else None)),
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
    p.add_argument("--baselines", default=",".join(B.DEFAULT_BASELINES),
                   help=f"lens strategies to score, from {list(B.BASELINES)}. "
                        f"The default is the published two; {B.VGGT360!r} is "
                        f"this repo's VGGT-360-fisheye port and is asked for by "
                        f"name, because adding it to the default would change "
                        f"what every existing command measures")
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
    p.add_argument("--context-frames", default="1",
                   help="frames handed to a multi-view model in ONE forward "
                        "pass, of which only the split's own frame is scored. "
                        "A comma list sweeps it in one run on one model load, "
                        "e.g. '1,3,5,10'. Monocular models are refused, not "
                        "silently given one frame")
    p.add_argument("--context-stride", type=int, default=1,
                   help="spacing of the context window in frames. ego-synth "
                        "clips are 20 fps, so 10 consecutive frames span 0.45 s "
                        "and a head-worn camera has barely moved; without both "
                        "a consecutive and a strided arm a null result cannot "
                        "tell the model from the missing baseline")
    p.add_argument("--allow-unverified", action="store_true",
                   help="run rect_derect with a sensor-to-upright rotation that "
                        "has not passed verify_camera. A quarter-turn error does "
                        "not degrade the score, it scores a different part of "
                        "the image — so this is for debugging, not for numbers")
    p.add_argument("--oracle-bias", type=float, default=0.0,
                   help="multiplicative error injected by --models oracle, for "
                        "checking the harness reports what it was given")
    # -- the vggt360 arm's own layout --------------------------------------- #
    # Defaults are VGGT-360-fisheye/main_adt.py's, so `--baselines vggt360` with
    # no further flags is the same model that driver reports on ADT, here on
    # ego-synth's lens and ground truth.
    g = p.add_argument_group(
        "vggt360", "VGGT-360-fisheye layout; ignored unless --baselines "
                   "includes vggt360. Defaults match main_adt.py.")
    g.add_argument("--vggt360-fov", type=float, default=60.0,
                   help="per-view FOV (deg) of the tangent views")
    g.add_argument("--vggt360-ring-tilt", type=float, default=26.0,
                   help="ring tilt off the optical axis (deg); the layout rule "
                        "is tilt + fov/2 >~ the lens' usable cone, which is "
                        "derived per take and printed, not assumed")
    g.add_argument("--vggt360-n-ring", type=int, default=8,
                   help="ring view count (the centre view is extra)")
    g.add_argument("--vggt360-persp-size", type=int, default=518,
                   help="side length each tangent view is rendered at. 518 is "
                        "the backbone's own token grid, so nothing is resampled "
                        "between the view and the network. main_adt.py uses "
                        "512, which VGGT then bicubic-resizes up to 518 anyway")
    g.add_argument("--vggt360-fuse", choices=["attn", "mean"], default="attn",
                   help="correlation-weighted fusion (the paper) or uniform")
    g.add_argument("--vggt360-head", choices=["depth", "point"], default="depth",
                   help="range source: depth head z * secant, or the point "
                        "head's ||world_points||")
    g.add_argument("--vggt360-dtype", choices=["bf16", "fp16", "fp32"],
                   default="bf16", help="autocast dtype for the VGGT pass")
    g.add_argument("--vggt360-no-adaptive", action="store_true",
                   help="module-1 ablation: base views only")
    g.add_argument("--vggt360-no-sa-mask", action="store_true",
                   help="module-2 ablation: vanilla attention")

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
