"""Online adaptation: fit the adapter on a short temporal segment.

Paper Sec. 4.3 + "Implementation details": Adam, lr 1e-3, gradient clipping at
norm 1.0, batches of three-frame windows, all residual parameters zero-init,
backbone frozen throughout. After adaptation, inference is a single forward pass
per fisheye frame with no extra runtime cost.

    python -m raytun3r.train \
        --backbone vggt --weights pretrained \
        --dataset scannetpp --path /data/scannetpp/data/<scene> \
        --out runs/rt3r/<scene>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from dataclasses import asdict
from typing import List, Optional

import torch

from .adapter import RayTun3RAdapter
from .backbones import build_backbone
from .baselines import attach_caltok, attach_lora
from .data import build_windows, load_sequence
from .losses import LossWeights, total_loss
from .matching import build_matcher

__all__ = ["fit_adapter", "main"]


def _match_coverage(windows: List, valid: torch.Tensor) -> float:
    """Mean ``sum(w) / |Omega|`` over every ordered pair of every window.

    This is the factor by which Eq. 8's ``1/|Omega|`` normalisation scales
    ``L_reproj`` relative to a matcher that is confident everywhere. It is a
    property of the matcher, not of the model, so it is measured once.
    """
    omega = max(float(valid.sum()), 1.0)
    # Masked here too, not just trusted to have been masked at construction:
    # reprojection_loss applies ``w * valid``, so this is the factor Eq. 8 really
    # uses rather than a number that merely agrees with it by convention.
    sums = [float((m.weight * valid.to(m.weight.dtype)).sum()) / omega
            for win in windows for m in win.matches.values()]
    if not sums:
        return 0.0
    out = float(sum(sums) / len(sums))
    if out > 1.0 + 1e-6:
        # Only reachable if the windows were matched against a wider camera than
        # the one fit_adapter was handed, in which case the reported factor is
        # not the one Eq. 8 applies.
        raise ValueError(
            f"matcher coverage {out:.3f} exceeds 1: the correspondences were built "
            f"against a different valid mask than this fit is using.")
    return out


def fit_adapter(backbone, windows: List, camera, *, iters: int = 300,
                lr: float = 1e-3, clip: float = 1.0, weights: LossWeights = None,
                convention: str = "range", batch_size: int = 1,
                params: Optional[List[torch.nn.Parameter]] = None,
                log_every: int = 25, seed: int = 0, verbose: bool = True,
                grad_checkpointing: bool = True, matcher_name: Optional[str] = None,
                min_coverage: float = 0.05) -> dict:
    """Fit whatever trainable parameters are attached, on the given windows.

    ``params=None`` trains the backbone's installed :class:`RayTun3RAdapter`;
    passing an explicit list is how the LoRA and CalTok baselines reuse the exact
    same objective, so the comparison isolates *what* is adapted.

    ``grad_checkpointing`` asks the frozen backbone to recompute trunk
    activations in the backward pass instead of storing them. The adapter sits at
    the first layer, so the whole trunk is on the gradient path and its
    activations dominate the step's memory; see
    :meth:`Backbone.grad_checkpointing` for why this is numerically inert on the
    backbones that support it. Turn it off to trade the memory back for speed.

    ``matcher_name`` and ``min_coverage`` are about Eq. 8's ``1/|Omega|``: the
    term's magnitude scales with how much of the disc the matcher is confident
    about, so the matcher silently sets ``L_reproj``'s weight against the
    regularisers. Both the measured coverage and the matcher that produced it are
    returned and written to ``train_log.json``; below ``min_coverage`` the fit
    refuses to run at all.
    """
    weights = weights or LossWeights()
    torch.manual_seed(seed)

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    trainable = params if params is not None else list(backbone.adapter.parameters())
    trainable = [p for p in trainable if p.requires_grad]
    if not trainable:
        raise ValueError("nothing to train: no parameter requires grad")

    opt = torch.optim.Adam(trainable, lr=lr)
    dev = windows[0].images.device
    valid = camera.valid_mask(*windows[0].images.shape[-2:], device=dev)

    # Eq. 8 divides by |Omega|, not by sum(w), so L_reproj scales with how much
    # of the disc the matcher is confident about. That is the equation as written
    # -- but it means the term's weight against w_smooth=10 / w_L2=2 / w_TV=20 is
    # set by the matcher, not by Eq. 13. UFM is dense (coverage near 1); the SIFT
    # fallback writes w=1 on a few thousand integer keypoints out of ~2e5 pixels,
    # i.e. coverage ~0.01, which quietly divides L_reproj by ~100 and leaves the
    # regularisers to dominate. Report it rather than let it hide in a ratio.
    coverage = _match_coverage(windows, valid)
    if verbose:
        print(f"[fit] matcher={matcher_name or 'unknown'}  coverage: mean "
              f"sum(w)/|Omega| = {coverage:.3f} over {len(windows)} windows")
    detail = (f"Matcher {matcher_name or 'unknown'!r} covers {coverage:.4f} of Omega, "
              f"so Eq. 8's |Omega| normalisation scales L_reproj by about that "
              f"factor while w_smooth={weights.smooth}, w_L2={weights.l2} and "
              f"w_TV={weights.tv} are unchanged. The paper's weights assume UFM's "
              f"dense correspondences.")
    if coverage < min_coverage:
        raise RuntimeError(
            detail + f" Below {min_coverage:g} the objective is not merely tilted "
            f"toward its regularisers, it is them to three significant figures, and "
            f"a number from such a run is not worth recording with a caveat. Pass "
            f"min_coverage=0 (CLI: --allow-sparse-matcher) to fit anyway, and say so "
            f"wherever the result is reported.")
    if coverage < 0.2:
        warnings.warn(detail + " Record this coverage next to any number from this "
                      "run.", RuntimeWarning)

    history = []
    ckpt_on = False
    t0 = time.time()
    g = torch.Generator().manual_seed(seed)
    # Shuffled epochs, not sampling with replacement. randint draws independently,
    # so with 30 windows and 300 steps the number of times each window is seen is
    # Binomial(300, 1/30) -- a spread of roughly 4x between the least- and
    # most-visited window on a typical seed, which is an uncontrolled reweighting
    # of the adaptation set. Upstream VGGT traverses epochs; this does the same,
    # reshuffling whenever the current permutation runs out.
    order: List[int] = []

    def _next_index() -> int:
        if not order:
            order.extend(torch.randperm(len(windows), generator=g).tolist())
        return order.pop(0)

    try:
        # Inside the try, so the finally below owns it from the moment it is on.
        ckpt_on = backbone.grad_checkpointing(True) if grad_checkpointing else False
        if verbose:
            if ckpt_on:
                print("[fit] gradient checkpointing on: the frozen backbone runs in "
                      "train mode, which for these models toggles nothing but the "
                      "checkpoint branch (no dropout, no droppath, no batchnorm)")
            elif grad_checkpointing:
                print(f"[fit] {type(backbone).__name__} does not support gradient "
                      f"checkpointing; the whole trunk's activations are retained "
                      f"for the backward pass")

        for it in range(iters):
            idx = [_next_index() for _ in range(batch_size)]
            opt.zero_grad(set_to_none=True)

            parts_sum = {}
            for k in idx:
                win = windows[k]
                pred = backbone.forward(win.images[None])
                # ``pe_table()`` is P_A on the bound token grid, and the backbone
                # can only capture it while the absolute-PE hook is running --
                # i.e. from inside a forward. Reading it once *before* the loop
                # (as this did) returned None on every backbone that has an
                # absolute table, silently reducing Eq. 12 from
                # TV(P_A + residual) to TV(residual): the pretrained table's own
                # roughness dropped out of the term carrying the objective's
                # largest weight (w_TV = 20).
                loss, parts = total_loss(pred, win.as_batch(), camera, backbone.adapter,
                                         weights=weights, convention=convention,
                                         valid=valid, pe_table=backbone.pe_table())
                if not math.isfinite(parts["total"]):
                    raise RuntimeError(
                        f"loss went non-finite at iteration {it} on window {k}: "
                        f"{ {kk: vv for kk, vv in parts.items()} }")
                # Backward here, not after the loop. Summing the batch into one
                # tensor and calling backward() once keeps every window's graph
                # -- and with the adapter at the first layer that means every
                # window's full trunk activations -- alive simultaneously, so
                # peak memory grows linearly with --batch-size. Gradients
                # accumulate into .grad either way, so this is the same step;
                # it is what VGGT's own trainer does per accumulation chunk
                # (training/trainer.py::_run_steps_on_batch_chunks).
                (loss / len(idx)).backward()
                for kk, vv in parts.items():
                    parts_sum[kk] = parts_sum.get(kk, 0.0) + vv / len(idx)

            gnorm = torch.nn.utils.clip_grad_norm_(trainable, clip)
            # The gradient, not the loss, is where a broken step is detectable.
            # A NaN anywhere in the forward makes the derivative through
            # camera.project NaN, and nothing downstream can undo that: a
            # substitution's zero meets a non-finite local derivative one node up
            # and becomes 0 * inf. So the *loss* can be perfectly finite while
            # every gradient is NaN -- and clip_grad_norm_ then multiplies every
            # parameter's gradient by that NaN coefficient, which Adam's moments
            # make permanent. The fit would run out its 300 iterations logging
            # plausible numbers and save an all-NaN adapter.
            #
            # clip_grad_norm_ returns the total norm, which is non-finite exactly
            # when some gradient is, so this one check covers every parameter.
            if not math.isfinite(float(gnorm)):
                raise RuntimeError(
                    f"gradient went non-finite at iteration {it} (total norm "
                    f"{float(gnorm)}) while the loss was finite "
                    f"({parts_sum.get('total')}). Something in the forward pass "
                    f"produced NaN/inf -- check the prediction, the depth "
                    f"convention and the camera model. Stopping here: every "
                    f"parameter's gradient has already been scaled by this norm, "
                    f"so continuing would only save a corrupt adapter.")
            opt.step()

            parts_sum["grad_norm"] = float(gnorm)
            parts_sum["iter"] = it
            history.append(parts_sum)
            if verbose and (it % log_every == 0 or it == iters - 1):
                print(f"[fit] {it:4d}/{iters}  total={parts_sum['total']:.4f}  "
                      f"reproj={parts_sum['reproj']:.3f}  pose={parts_sum['pose']:.4f}  "
                      f"smooth={parts_sum['smooth']:.4f}  |g|={float(gnorm):.3f}")
    finally:
        # Back to eval() whatever happened, so a later forward -- evaluation,
        # another fit, the direct path -- is not silently left in train mode.
        if ckpt_on:
            backbone.grad_checkpointing(False)

    if verbose:
        print(f"[fit] done in {time.time() - t0:.1f}s over {len(windows)} windows")
    # match_coverage and matcher go in the returned dict, not just on stdout:
    # they are the two numbers that decide whether L_reproj had any weight at all
    # in this fit, so any result read back from train_log.json needs them beside
    # it. Every programmatic caller runs verbose=False.
    return {"history": history, "seconds": time.time() - t0,
            "match_coverage": coverage, "matcher": matcher_name}


from .backbones import BACKBONE_NAMES  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("raytun3r.train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="vggt", choices=BACKBONE_NAMES)
    p.add_argument("--weights", default="pretrained")
    p.add_argument("--variant", default="small",
                   choices=["small", "base", "large", "giant"],
                   help="DA3 size; 'small' is the paper's primary backbone")
    p.add_argument("--dataset", default="scannetpp", choices=["scannetpp", "adt"])
    p.add_argument("--path", required=True, help="scene / sequence directory")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--method", default="raytun3r",
                   choices=["raytun3r", "lora", "caltok"],
                   help="what to adapt; all three use the same objective")
    p.add_argument("--n-radial", type=int, default=20)
    p.add_argument("--n-angular", type=int, default=8)
    p.add_argument("--n-rope-radial", type=int, default=20)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--caltok-t", type=int, default=4)

    p.add_argument("--no-patch-undistort", action="store_true")
    p.add_argument("--no-border-token", action="store_true")
    p.add_argument("--no-dpt-grid", action="store_true")
    p.add_argument("--grid-mode", default="auto", choices=["auto", "tan", "angular"])

    p.add_argument("--matcher", default="auto", choices=["auto", "ufm", "raft", "sift"])
    p.add_argument("--windows", type=int, default=30)
    p.add_argument("--seq-len", type=int, default=3)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min-flow-px", type=float, default=2.0)
    p.add_argument("--max-size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--extrinsics-json", default=None, help="ADT T_device_camera")
    p.add_argument("--max-fov", type=float, default=None,
                   help="restrict Omega to this total FOV in degrees (never widens); "
                        "the knob for the paper's stated 115 deg vs ScanNet++'s real ~170")

    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--w-pose", type=float, default=1.0)
    p.add_argument("--w-smooth", type=float, default=10.0)
    p.add_argument("--w-l2", type=float, default=2.0)
    p.add_argument("--w-tv", type=float, default=20.0)
    p.add_argument("--keep-bad", action="store_true",
                   help="keep ScanNet++ frames the dataset flags is_bad; they are "
                        "dropped by default (143 of 896 on 3f15a9266d)")
    p.add_argument("--allow-sparse-matcher", action="store_true",
                   help="fit even when the matcher covers less than 5%% of Omega. "
                        "Eq. 8 divides by |Omega|, so at that coverage L_reproj is "
                        "numerically absent and the fit is its regularisers")
    p.add_argument("--no-grad-checkpointing", action="store_true",
                   help="keep the frozen backbone's trunk activations instead of "
                        "recomputing them in the backward pass: faster per step, "
                        "and several times the memory. The adapter sits at the "
                        "first layer, so the whole trunk is on the gradient path")
    p.add_argument("--convention", default="range", choices=["range", "z"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--windows-cache", type=str, default=None, metavar="PATH",
                   help="torch.save/load the built windows (images, matches, "
                        "MAGSAC pose targets). Loads PATH if it exists, else "
                        "builds normally and saves to it. This pins the fit's "
                        "inputs across runs: the matcher's GPU forward is not "
                        "run-to-run reproducible, and MAGSAC turns that into a "
                        "discretely different Eq. 9 target -- which is what an "
                        "A/B over anything fit-side (e.g. checkpointing) must "
                        "not be confounded by (issue #26)")
    return p


def main(argv=None) -> None:
    args = build_argparser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    backbone = build_backbone(args.backbone, weights=args.weights, device=args.device,
                              **({"variant": args.variant} if args.backbone == "da3" else {}))
    source = load_sequence(args.dataset, args.path, max_size=args.max_size,
                           patch=backbone.patch_size, max_frames=args.max_frames,
                           depth_convention=args.convention,
                           **({"extrinsics_json": args.extrinsics_json}
                              if args.dataset == "adt" else
                              {"keep_bad": args.keep_bad}))
    if args.max_fov is not None:
        before = 2 * math.degrees(source.camera.theta_max)
        source.camera = source.camera.with_max_fov(args.max_fov)
        after = 2 * math.degrees(source.camera.theta_max)
        print(f"[data] Omega restricted: {before:.0f} deg -> {after:.0f} deg FOV "
              f"(images unchanged; this narrows where the method is scored)")

    print(f"[data] {source.name}: {len(source)} frames at {source.h}x{source.w}, "
          f"FOV {2 * torch.rad2deg(torch.tensor(source.camera.theta_max)):.0f} deg")

    if args.windows_cache is not None and os.path.exists(args.windows_cache):
        # weights_only=False: Window is a dataclass, not a bare state dict. The
        # file is our own output, written two lines below. map_location is
        # --device, not "cpu": fit_adapter takes its working device from
        # windows[0].images.device, so a cpu-loaded cache silently overrides
        # --device and then dies inside the backbone forward (issue #26).
        windows = torch.load(args.windows_cache, map_location=args.device,
                             weights_only=False)
        matcher_label = f"cache:{args.windows_cache}"
        print(f"[data] windows loaded from {args.windows_cache} "
              f"({len(windows)} windows; matcher and MAGSAC not re-run)")
    else:
        matcher = build_matcher(args.matcher, device=args.device)
        windows = build_windows(source, matcher, n_windows=args.windows,
                                seq_len=args.seq_len, stride=args.stride,
                                min_flow_px=args.min_flow_px, seed=args.seed,
                                device=args.device)
        matcher_label = matcher.name
        if args.windows_cache is not None:
            torch.save(windows, args.windows_cache)
            print(f"[data] windows saved to {args.windows_cache}")

    adapter, params, handles = None, None, []
    if args.method == "raytun3r":
        adapter = backbone.make_adapter(n_radial=args.n_radial, n_angular=args.n_angular,
                                        n_rope_radial=args.n_rope_radial).to(args.device)
        print(f"[adapter] {adapter.param_breakdown()}")
    else:
        print(f"[adapter] method={args.method}: parameter-free corrections disabled so the "
              f"comparison isolates the adapted component")

    backbone.install(
        adapter, source.camera, (source.h, source.w),
        patch_undistort=(args.method == "raytun3r" and not args.no_patch_undistort),
        border_token=(args.method == "raytun3r" and not args.no_border_token),
        dpt_grid=(args.method == "raytun3r" and not args.no_dpt_grid),
        grid_mode=args.grid_mode,
        depth_convention=args.convention,
    )

    if args.method == "lora":
        mods, handles = attach_lora(backbone, r=args.lora_r, alpha=args.lora_alpha)
        params = list(mods.parameters())
    elif args.method == "caltok":
        mods, handles = attach_caltok(backbone, n_tokens=args.caltok_t)
        params = list(mods.parameters())
    if params is not None:
        print(f"[adapter] {args.method}: {sum(p.numel() for p in params)} trainable parameters")

    weights = LossWeights(pose=args.w_pose, smooth=args.w_smooth,
                          l2=args.w_l2, tv=args.w_tv)
    stats = fit_adapter(backbone, windows, source.camera, iters=args.iters, lr=args.lr,
                        clip=args.clip, weights=weights, convention=args.convention,
                        batch_size=args.batch_size, params=params, seed=args.seed,
                        grad_checkpointing=not args.no_grad_checkpointing,
                        matcher_name=matcher_label,
                        min_coverage=0.0 if args.allow_sparse_matcher else 0.05)

    ckpt = {"method": args.method, "backbone": args.backbone, "args": vars(args)}
    if adapter is not None:
        ckpt["adapter"] = adapter.state_dict()
        ckpt["param_breakdown"] = adapter.param_breakdown()
    else:
        ckpt["baseline_state"] = {f"m{i}": m.state_dict() for i, m in enumerate(mods)}
    torch.save(ckpt, os.path.join(args.out, "adapter.pt"))
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump({"history": stats["history"], "seconds": stats["seconds"],
                   "match_coverage": stats["match_coverage"],
                   "matcher": stats["matcher"]}, f, indent=2)
    print(f"[out] wrote {args.out}/adapter.pt")

    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
