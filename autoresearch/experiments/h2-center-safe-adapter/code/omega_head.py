"""H2.4: the feature head on VGGT-Omega (frozen), per-scene.

Protocol: ../protocol-h2.4-vggt-omega.md (committed before any weighted run).

CPU-side: `--random-init` exercises the full path with a small random config
(structure validation ONLY — numbers from it are meaningless by design).
Box-side: pass --weights <checkpoint.pt> for the real thing.

Usage:
    python autoresearch/experiments/h2-center-safe-adapter/code/omega_head.py \
        --seq <seq_dir> --split halves --weights <ckpt> --out results/run_013_<seq>.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "h1-rim-pose-value" / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adt_pose_value import AriaLocalPairs, DEFAULT_SEQ    # noqa: E402
from finetune.eval.metrics import align_depth             # noqa: E402
from feature_head import Head                             # noqa: E402

THETA_BINS = 8
GT_DEPTH_EDGES = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)
PATCH = 16                     # VGGT-Omega patch size (not DA3's 14)
SEED = 0
CACHE_ROOT = Path(os.environ.get(
    "H2_CACHE",
    "/private/tmp/claude-501/-Users-fengjiazhang-Desktop-ADT-vggt-omega/"
    "c1591315-71f0-4ca2-9183-9f58fff82acc/scratchpad/h24_cache"))


def build_model(weights: str, random_init: bool, device: str):
    from vggt_omega.models.vggt_omega import VGGTOmega
    if random_init:
        # small-width random config (structure check only; block count is the
        # Aggregator default — the API exposes no depth knob).
        model = VGGTOmega(embed_dim=384, patch_size=PATCH)
    else:
        model = VGGTOmega(patch_size=PATCH)
        sd = torch.load(weights, map_location="cpu")
        model.load_state_dict(sd.get("model", sd), strict=True)
    return model.to(device).eval()


def cache_seq(src, frames, model, device, cache: Path,
              sanitize: bool = False) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    todo = [n for n in frames if not (cache / f"pred_{n}.npy").exists()
            or not (cache / f"feat_{n}.npy").exists()]
    if not todo:
        return
    cos_t = torch.cos(src.camera.incidence_grid(src.h, src.w))
    for n in todo:
        t0 = time.time()
        img = src.image(n)[None, None].to(device)         # (1, 1, 3, H, W)
        with torch.no_grad():
            tokens_list, patch_start = model.aggregator(img)
            final = tokens_list[-1]
            depth, _conf = model.dense_head(
                tokens_list, images=img, patch_token_start=patch_start)
        # dense head emits planar z; convert ONCE to euclidean range through
        # the calibrated KB4 camera (CONTEXT.md depth-convention discipline).
        z = depth.reshape(src.h, src.w).float().cpu()
        tok = final.reshape(-1, final.shape[-1])[patch_start:]
        if sanitize:
            # RANDOM-INIT ONLY: untrained weights emit NaN; substitute finite
            # dummies so the downstream path is exercised. Numbers meaningless.
            g = torch.Generator().manual_seed(SEED + n)
            z = 2.0 + 0.5 * torch.rand(z.shape, generator=g)
            tok = torch.nan_to_num(tok, nan=0.0) \
                + 0.01 * torch.randn(tok.shape, generator=g)
        np.save(cache / f"pred_{n}.npy", (z / cos_t.clamp_min(1e-6)).numpy())
        np.save(cache / f"feat_{n}.npy", tok.float().cpu().numpy())
        print(f"  cached {n} ({time.time() - t0:.1f}s, tok {tuple(tok.shape)})",
              flush=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", default=DEFAULT_SEQ)
    p.add_argument("--size", type=int, default=512)       # divisible by 16
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--split", default="halves", choices=["even_odd", "halves"])
    p.add_argument("--weights", default=None)
    p.add_argument("--random-init", action="store_true")
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    torch.manual_seed(SEED)
    assert args.random_init or args.weights, "--weights or --random-init"

    src = AriaLocalPairs(os.path.expanduser(args.seq), size=args.size)
    theta = src.camera.incidence_grid(src.h, src.w)
    cone = (theta <= src.camera.theta_max).numpy()
    cos_t = torch.cos(theta).numpy()
    t_edges = np.linspace(0.0, float(src.camera.theta_max), THETA_BINS + 1)
    t_idx = np.clip(np.digitize(theta.numpy(), t_edges) - 1, 0, THETA_BINS - 1)
    gh, gw = src.h // PATCH, src.w // PATCH
    theta_p = theta.numpy().reshape(gh, PATCH, gw, PATCH).mean((1, 3)).ravel()

    depth_paths = {os.path.basename(q).replace(".npy", ""): q for q in
                   glob.glob(os.path.join(os.path.expanduser(args.seq),
                                          "depth_npy", "*.npy"))}
    stem = lambda n: os.path.splitext(os.path.basename(src.paths[n]))[0]
    frames = [n for n in range(len(src.paths)) if stem(n) in depth_paths]
    if len(frames) > args.max_frames:
        frames = frames[::max(1, len(frames) // args.max_frames)][:args.max_frames]
    if args.split == "even_odd":
        train = [n for k, n in enumerate(frames) if k % 2 == 0]
        test = [n for k, n in enumerate(frames) if k % 2 == 1]
    else:
        half = len(frames) // 2
        train, test = frames[:half], frames[half:]
    print(f"[h2.4] {len(train)} train / {len(test)} test "
          f"({'RANDOM-INIT — structure check only' if args.random_init else 'weighted'})")

    model = build_model(args.weights, args.random_init, args.device)
    cache = CACHE_ROOT / (os.path.basename(args.seq.rstrip('/'))
                          + (".rnd" if args.random_init else ""))
    cache_seq(src, frames, model, args.device, cache,
              sanitize=args.random_init)
    preds = {n: np.load(cache / f"pred_{n}.npy") for n in frames}
    feats = {n: np.load(cache / f"feat_{n}.npy") for n in frames}

    def gt_range(n):
        gz = np.load(depth_paths[stem(n)]).astype(np.float32)
        gz = torch.nn.functional.interpolate(
            torch.from_numpy(gz)[None, None], size=(src.h, src.w),
            mode="nearest")[0, 0].numpy() / 1000.0
        return gz / np.clip(cos_t, 1e-6, None), gz

    def patch_pool(m, vmask=None):
        r = m.reshape(gh, PATCH, gw, PATCH).transpose(0, 2, 1, 3).reshape(gh * gw, -1)
        if vmask is None:
            return np.median(r, axis=1)
        v = vmask.reshape(gh, PATCH, gw, PATCH).transpose(0, 2, 1, 3).reshape(gh * gw, -1)
        out = np.full(gh * gw, np.nan)
        for k in range(gh * gw):
            if v[k].sum() >= 30:
                out[k] = np.median(r[k][v[k]])
        return out

    X, A, Y = [], [], []
    for n in train:
        gr, gz = gt_range(n)
        d = preds[n]
        valid = cone & (gz > 0) & (gr <= args.depth_max_m) & (d > 1e-6)
        r = np.log(gr) - np.log(np.clip(d, 1e-6, None))
        tgt = patch_pool(r - float(np.median(r[valid])), vmask=valid)
        dp = patch_pool(np.log(np.clip(d, 1e-6, None)))
        ok = np.isfinite(tgt)
        X.append(feats[n][ok])
        A.append(np.stack([np.sin(theta_p[ok]), np.cos(theta_p[ok]), dp[ok]], -1))
        Y.append(tgt[ok])
    X = torch.from_numpy(np.concatenate(X)).float()
    A = torch.from_numpy(np.concatenate(A)).float()
    Y = torch.from_numpy(np.concatenate(Y)).float()
    print(f"[h2.4] {len(Y)} training patches, feat dim {X.shape[-1]}")

    head = Head(X.shape[-1])
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for ep in range(args.epochs):
        opt.zero_grad()
        loss = (head(X, A) - Y).abs().mean()
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d} L1 {loss.item():.4f}", flush=True)

    nb_d = len(GT_DEPTH_EDGES) - 1
    t_mid = [math.degrees(0.5 * (t_edges[i] + t_edges[i + 1]))
             for i in range(THETA_BINS)]

    def joint(correct: bool):
        s_ = np.zeros((THETA_BINS, nb_d))
        n_ = np.zeros((THETA_BINS, nb_d), dtype=np.int64)
        for n in test:
            gr, gz = gt_range(n)
            d = preds[n].copy()
            if correct:
                dp = patch_pool(np.log(np.clip(d, 1e-6, None)))
                with torch.no_grad():
                    corr = head(torch.from_numpy(feats[n]).float(),
                                torch.from_numpy(np.stack(
                                    [np.sin(theta_p), np.cos(theta_p), dp],
                                    -1)).float()).numpy().reshape(gh, gw)
                corr_px = torch.nn.functional.interpolate(
                    torch.from_numpy(corr)[None, None].float(),
                    size=(src.h, src.w), mode="bilinear",
                    align_corners=False)[0, 0].numpy()
                d = d * np.exp(corr_px)
            valid = cone & (gz > 0) & (gr <= args.depth_max_m) & (d > 1e-6)
            aligned = align_depth(d, gr, valid, mode="scale_shift")
            absrel = (np.abs(aligned - gr) / np.clip(gr, 1e-6, None))[valid]
            ti = t_idx[valid]
            di = np.clip(np.digitize(gr[valid], GT_DEPTH_EDGES) - 1, 0, nb_d - 1)
            flat = ti * nb_d + di
            s_ += np.bincount(flat, weights=absrel,
                              minlength=THETA_BINS * nb_d).reshape(THETA_BINS, nb_d)
            n_ += np.bincount(flat,
                              minlength=THETA_BINS * nb_d).reshape(THETA_BINS, nb_d)
        return s_ / np.maximum(n_, 1), n_

    before, counts = joint(False)
    after, _ = joint(True)
    print("\nheld-out joint AbsRel BEFORE -> AFTER:")
    for j in range(nb_d):
        row = " ".join(f"{before[i, j]:5.3f}>{after[i, j]:5.3f}"
                       for i in range(THETA_BINS))
        print(f"{GT_DEPTH_EDGES[j]:4.0f}-{GT_DEPTH_EDGES[j + 1]:2.0f} m  {row}")
    nr = [(i, j) for i in range(THETA_BINS) for j in range(nb_d)
          if t_mid[i] >= 38 and GT_DEPTH_EDGES[j + 1] <= 2.0]
    w = np.array([counts[i, j] for i, j in nr], float)
    b = float((np.array([before[i, j] for i, j in nr]) * w).sum() / w.sum())
    a = float((np.array([after[i, j] for i, j in nr]) * w).sum() / w.sum())
    print(f"near_rim: {b:.3f} -> {a:.3f} ({(a - b) / b * 100:+.1f}%)")

    if args.out:
        dst = Path(__file__).resolve().parents[1] / args.out \
            if not os.path.isabs(args.out) else Path(args.out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            json.dump({"split": args.split, "random_init": args.random_init,
                       "before": before.tolist(), "after": after.tolist(),
                       "counts": counts.tolist(), "theta_bin_mid_deg": t_mid,
                       "near_rim": {"before": b, "after": a},
                       "config": {k: v for k, v in vars(args).items()}},
                      f, indent=2)
        print(f"\n[h2.4] wrote {dst}")


if __name__ == "__main__":
    main()
