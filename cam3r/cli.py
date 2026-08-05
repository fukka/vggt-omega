# Copyright (c) 2026.
"""Argument groups and model construction shared by ``train`` and ``eval_adt``.

Both entry points need the same model geometry and the same ADT/pair-window
options.  Keeping one copy means the two cannot drift into building differently
shaped models from the same flags.

Defaults are the paper's (Table S3 / Sec. 4.2): ViT-L width 1024, 24 layers,
768-wide decoder, 512 px long edge.  A CPU run needs explicit small values --
see ``cam3r/README.md`` for the exact invocations used locally.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

from cam3r.model import CAM3R, CAM3RConfig

ADT_ROOT_ENV = "CAM3R_ADT_ROOT"


def default_adt_root() -> Optional[str]:
    """ADT dataset root from ``$CAM3R_ADT_ROOT``; no hard-coded machine paths."""
    return os.environ.get(ADT_ROOT_ENV)


def add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model (defaults = paper Table S3)")
    g.add_argument("--resolution", type=int, default=512, help="long edge, paper: 512")
    g.add_argument("--patch-size", type=int, default=16)
    g.add_argument("--width", type=int, default=1024, help="ViT-L encoder width")
    g.add_argument("--dec-width", type=int, default=768, help="decoder width (DUSt3R: 768)")
    g.add_argument("--depth", type=int, default=24, help="encoder blocks")
    g.add_argument("--heads", type=int, default=16)
    g.add_argument("--dec-heads", type=int, default=12, help="decoder heads (DUSt3R: 12)")
    g.add_argument("--angular-width", type=int, default=512,
                   help="angular module width (Table S3 projects 1024 -> 512)")
    g.add_argument("--angular-heads", type=int, default=8, help="angular module heads (UniK3D: 8)")
    g.add_argument("--dpt-features", type=int, default=256, help="DPT D_feat (Table S3: 256)")
    g.add_argument("--dust3r", default=None, help="DUSt3R checkpoint for Cross-view Module init")
    g.add_argument("--unik3d", default=None, help="UniK3D checkpoint for Ray Module init")


def add_adt_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("ADT data and pair window (defaults = paper Sec. D.3)")
    g.add_argument("--adt-root", default=default_adt_root(),
                   help=f"ADT dataset root (or set ${ADT_ROOT_ENV})")
    g.add_argument("--rgb-subdir", default="videos_rgb")
    g.add_argument("--extrinsics-json", default=None, help="JSON holding T_device_camera")
    g.add_argument("--max-frames", type=int, default=100)
    g.add_argument("--max-pairs", type=int, default=None)
    g.add_argument("--min-baseline", type=float, default=0.35)
    g.add_argument("--max-baseline", type=float, default=1.75)
    g.add_argument("--min-angle", type=float, default=25.0)
    g.add_argument("--max-angle", type=float, default=65.0)


def require_adt_root(args: argparse.Namespace) -> str:
    if not args.adt_root:
        raise SystemExit(f"--adt-root is required (or set ${ADT_ROOT_ENV})")
    return args.adt_root


def config_from_args(args: argparse.Namespace) -> CAM3RConfig:
    return CAM3RConfig(
        img_size=args.resolution,
        patch_size=args.patch_size,
        ray_embed_dim=args.width,
        ray_depth=args.depth,
        ray_heads=args.heads,
        ray_angular_dim=args.angular_width,
        ray_angular_heads=args.angular_heads,
        cv_embed_dim=args.width,
        cv_enc_depth=args.depth,
        cv_dec_embed_dim=args.dec_width,
        cv_dec_depth=max(2, args.depth // 2),
        cv_heads=args.heads,
        # DUSt3R's BaseDecoder is 768 wide with 12 heads, and its weights are
        # what initializes this decoder -- reusing the *encoder's* head count
        # would reinterpret that pretrained attention layout.
        cv_dec_heads=args.dec_heads,
        dpt_features=args.dpt_features,
    )


def build_model(args: argparse.Namespace) -> CAM3R:
    return CAM3R(config_from_args(args))
