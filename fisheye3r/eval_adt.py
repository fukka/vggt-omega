"""Evaluate Fisheye3R on ADT raw fisheye — the paper's ADT test protocol.

This is a thin bridge onto the repo's existing, validated ADT loader
(`finetune/eval/adt_depth.ADTWindowDataset`), which already handles the Aria
KB4 fisheye model, the 270 deg sensor rotation, and mm->m depth. The only
Fisheye3R-specific choices here are:

  * rectify=False  -> feed the RAW fisheye frame (Fisheye3R handles distortion
                     natively via calibration tokens; do NOT undistort first).
  * fisheye_flags all-True -> every ADT frame is fisheye, so tokens are active.
  * scale+shift depth alignment per sequence -> the paper's depth protocol.

Usage:
  python -m fisheye3r.eval_adt \
      --weights checkpoints/vggt_omega.pt \
      --tokens runs/f3r/tokens_ssl_final.pt \
      --adt-root /Users/fengjiazhang/Desktop/ADT \
      --seq Apartment_release_clean_seq131_M1292 --seq-len 8

Pass --baseline instead of --tokens for the unadapted VGGT-Omega numbers
(the "before adaptation" row you compare against).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import torch

from finetune.eval.adt_depth import ADTWindowDataset

from fisheye3r.eval import depth_metrics
from fisheye3r.model import Fisheye3R
from fisheye3r.train import load_base


def resolve_seq_dirs(adt_root: str, seqs: list[str] | None) -> list[str]:
    if seqs:
        return [os.path.join(adt_root, s) for s in seqs]
    # Auto-discover: any subdir that has a depth_npy/ folder.
    found = []
    for d in sorted(glob.glob(os.path.join(adt_root, "*"))):
        if os.path.isdir(os.path.join(d, "depth_npy")) or os.path.isdir(os.path.join(d, "videos_synthetic")):
            found.append(d)
    if not found:
        raise SystemExit(f"no ADT sequences with depth under {adt_root}")
    return found


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    base = load_base(args.weights, device)
    model = Fisheye3R(base, num_tokens=args.tokens_k, encoder_skip_layers=args.l0,
                      modules=args.modules).to(device).eval()
    if args.tokens:
        model.load_tokens(args.tokens, map_location=device)
    elif not args.baseline:
        raise SystemExit("pass --tokens <ckpt> or --baseline")

    seq_dirs = resolve_seq_dirs(args.adt_root, args.seq)
    accum: dict[str, list[float]] = defaultdict(list)

    for seq_len in args.seq_lens:
        dataset = ADTWindowDataset(
            seq_dirs,
            seq_len=seq_len,
            image_resolution=args.resolution,
            patch_size=16,
            rectify=False,                 # <-- raw fisheye: the Fisheye3R setting
            rgb_subdir=args.rgb_subdir,
            max_frames=args.max_frames,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.workers)

        for item in loader:
            images = item["images"].to(device)            # (1, S, 3, H, W)
            gt = item["depths"].to(device)                # (1, S, H, W)
            valid = item["valid_masks"].to(device)        # (1, S, H, W)
            B, S = images.shape[:2]
            flags = (None if args.use_classifier
                     else torch.zeros(B, S, dtype=torch.bool, device=device) if args.baseline
                     else torch.ones(B, S, dtype=torch.bool, device=device))

            preds = model(images, fisheye_flags=flags, use_classifier=args.use_classifier)
            for k, v in depth_metrics(preds["depth"], gt, valid).items():
                accum[k].append(v)

    results = {k: sum(v) / len(v) for k, v in sorted(accum.items()) if v}
    tag = "baseline" if args.baseline else (os.path.basename(args.tokens) if args.tokens else "tokens")
    print(f"\n=== ADT raw-fisheye | {tag} | seq_lens={args.seq_lens} | "
          f"{len(seq_dirs)} seq ===")
    for k, v in results.items():
        print(f"  {k:>16s}: {v:.4f}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"tag": tag, "seq_lens": args.seq_lens, **results}, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--tokens", default=None)
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--adt-root", required=True)
    p.add_argument("--seq", nargs="*", default=None, help="sequence dir names (default: auto-discover)")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[8])
    p.add_argument("--rgb-subdir", default="videos_synthetic",
                   help="videos_synthetic (aligned w/ GT depth) or videos_rgb (real sensor)")
    p.add_argument("--max-frames", type=int, default=100)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--tokens-k", type=int, default=8)
    p.add_argument("--l0", type=int, default=12)
    p.add_argument("--modules", default="EFG")
    p.add_argument("--use-classifier", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
