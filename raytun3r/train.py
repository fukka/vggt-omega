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


def fit_adapter(backbone, windows: List, camera, *, iters: int = 300,
                lr: float = 1e-3, clip: float = 1.0, weights: LossWeights = None,
                convention: str = "range", batch_size: int = 1,
                params: Optional[List[torch.nn.Parameter]] = None,
                log_every: int = 25, seed: int = 0, verbose: bool = True) -> dict:
    """Fit whatever trainable parameters are attached, on the given windows.

    ``params=None`` trains the backbone's installed :class:`RayTun3RAdapter`;
    passing an explicit list is how the LoRA and CalTok baselines reuse the exact
    same objective, so the comparison isolates *what* is adapted.
    """
    weights = weights or LossWeights()
    torch.manual_seed(seed)

    trainable = params if params is not None else list(backbone.adapter.parameters())
    trainable = [p for p in trainable if p.requires_grad]
    if not trainable:
        raise ValueError("nothing to train: no parameter requires grad")

    opt = torch.optim.Adam(trainable, lr=lr)
    dev = windows[0].images.device
    valid = camera.valid_mask(*windows[0].images.shape[-2:], device=dev)
    pe_table = backbone.pe_table()

    history = []
    t0 = time.time()
    g = torch.Generator().manual_seed(seed)
    for it in range(iters):
        idx = torch.randint(0, len(windows), (batch_size,), generator=g).tolist()
        opt.zero_grad(set_to_none=True)

        acc, parts_sum = None, {}
        for k in idx:
            win = windows[k]
            pred = backbone.forward(win.images[None])
            loss, parts = total_loss(pred, win.as_batch(), camera, backbone.adapter,
                                     weights=weights, convention=convention,
                                     valid=valid, pe_table=pe_table)
            acc = loss if acc is None else acc + loss
            for kk, vv in parts.items():
                parts_sum[kk] = parts_sum.get(kk, 0.0) + vv / len(idx)

        (acc / len(idx)).backward()
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, clip)
        opt.step()

        parts_sum["grad_norm"] = float(gnorm)
        parts_sum["iter"] = it
        history.append(parts_sum)
        if verbose and (it % log_every == 0 or it == iters - 1):
            print(f"[fit] {it:4d}/{iters}  total={parts_sum['total']:.4f}  "
                  f"reproj={parts_sum['reproj']:.3f}  pose={parts_sum['pose']:.4f}  "
                  f"smooth={parts_sum['smooth']:.4f}  |g|={float(gnorm):.3f}")

    if verbose:
        print(f"[fit] done in {time.time() - t0:.1f}s over {len(windows)} windows")
    return {"history": history, "seconds": time.time() - t0}


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
    p.add_argument("--convention", default="range", choices=["range", "z"])
    p.add_argument("--seed", type=int, default=0)
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

    matcher = build_matcher(args.matcher, device=args.device)
    windows = build_windows(source, matcher, n_windows=args.windows,
                            seq_len=args.seq_len, stride=args.stride,
                            min_flow_px=args.min_flow_px, seed=args.seed,
                            device=args.device)

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
                        batch_size=args.batch_size, params=params, seed=args.seed)

    ckpt = {"method": args.method, "backbone": args.backbone, "args": vars(args)}
    if adapter is not None:
        ckpt["adapter"] = adapter.state_dict()
        ckpt["param_breakdown"] = adapter.param_breakdown()
    else:
        ckpt["baseline_state"] = {f"m{i}": m.state_dict() for i, m in enumerate(mods)}
    torch.save(ckpt, os.path.join(args.out, "adapter.pt"))
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump({"history": stats["history"], "seconds": stats["seconds"]}, f, indent=2)
    print(f"[out] wrote {args.out}/adapter.pt")

    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
