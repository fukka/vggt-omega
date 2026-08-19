"""H2.3: one feature head across scenes (leave-one-scene-out).

Protocol: ../protocol-h2.3-cross-scene.md (committed before any run). Written
CPU-side; intended to run on the box where the six-sequence split lives.

Per-sequence caches land under $H2_CACHE/<seq_basename>/ automatically, so no
per-sequence env juggling is needed.

Usage:
    python autoresearch/experiments/h2-center-safe-adapter/code/cross_scene.py \
        --train-seqs /path/seqA,/path/seqB,... --eval-seqs /path/seqF \
        --out results/run_012_fold_seqF.json
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

from adt_pose_value import AriaLocalPairs                 # noqa: E402
from finetune.eval.metrics import align_depth             # noqa: E402
from feature_head import Head                             # noqa: E402

THETA_BINS = 8
GT_DEPTH_EDGES = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)
PATCH = 14
SEED = 0
CACHE_ROOT = Path(os.environ.get(
    "H2_CACHE",
    "/private/tmp/claude-501/-Users-fengjiazhang-Desktop-ADT-vggt-omega/"
    "c1591315-71f0-4ca2-9183-9f58fff82acc/scratchpad/h2_cross_cache"))


class SeqData:
    """One sequence: frames with GT depth, cached preds + final-block tokens."""

    def __init__(self, seq_dir: str, size: int, max_frames: int) -> None:
        self.src = AriaLocalPairs(seq_dir, size=size)
        self.name = os.path.basename(seq_dir.rstrip("/"))
        self.cache = CACHE_ROOT / self.name
        dp = {os.path.basename(q).replace(".npy", ""): q for q in
              glob.glob(os.path.join(seq_dir, "depth_npy", "*.npy"))}
        fr = [n for n in range(len(self.src.paths))
              if os.path.basename(self.src.paths[n]).replace(".jpg", "") in dp]
        if len(fr) > max_frames:
            fr = fr[::max(1, len(fr) // max_frames)][:max_frames]
        self.frames, self.depth_paths = fr, dp

    def ensure_cached(self) -> None:
        self.cache.mkdir(parents=True, exist_ok=True)
        todo = [n for n in self.frames
                if not (self.cache / f"pred_{n}.npy").exists()
                or not (self.cache / f"feat_{n}.npy").exists()]
        if not todo:
            return
        from raytun3r.backbones import build_backbone
        bb = build_backbone("da3", weights="pretrained",
                            device="cuda" if torch.cuda.is_available() else "cpu",
                            variant="small")
        bb.install(None, self.src.camera, (self.src.h, self.src.w),
                   patch_undistort=False, border_token=False, dpt_grid=False,
                   depth_convention="range")
        vit = bb._vit()
        blocks = getattr(vit, "blocks", None) or getattr(vit, "layers")
        grabbed: Dict[str, torch.Tensor] = {}
        h = blocks[-1].register_forward_hook(
            lambda _m, _i, out: grabbed.__setitem__(
                "tok", out[0] if isinstance(out, tuple) else out))
        n_patch = (self.src.h // PATCH) * (self.src.w // PATCH)
        for n in todo:
            t0 = time.time()
            with torch.no_grad():
                pr = bb.forward(self.src.image(n)[None, None].to(
                    next(bb.parameters()).device
                    if hasattr(bb, "parameters") else "cpu"))
            pr.require_convention("range")
            np.save(self.cache / f"pred_{n}.npy",
                    pr.depth[0].cpu().numpy())
            tok = grabbed["tok"].reshape(-1, grabbed["tok"].shape[-1])
            extra = tok.shape[0] - n_patch
            assert 0 <= extra < 16, f"token layout: {extra} extra"
            np.save(self.cache / f"feat_{n}.npy",
                    tok[extra:].to(torch.float32).cpu().numpy())
            print(f"  [{self.name}] cached {n} ({time.time() - t0:.1f}s)",
                  flush=True)
        h.remove()

    def pred(self, n): return np.load(self.cache / f"pred_{n}.npy")
    def feat(self, n): return np.load(self.cache / f"feat_{n}.npy")

    def gt_range(self, n) -> Tuple[np.ndarray, np.ndarray]:
        stem = os.path.basename(self.src.paths[n]).replace(".jpg", "")
        gz = np.load(self.depth_paths[stem]).astype(np.float32)
        gz = torch.nn.functional.interpolate(
            torch.from_numpy(gz)[None, None],
            size=(self.src.h, self.src.w), mode="nearest")[0, 0].numpy() / 1000.0
        cos_t = torch.cos(self.src.camera.incidence_grid(
            self.src.h, self.src.w)).numpy()
        return gz / np.clip(cos_t, 1e-6, None), gz


def geometry(src):
    theta = src.camera.incidence_grid(src.h, src.w)
    cone = (theta <= src.camera.theta_max).numpy()
    t_edges = np.linspace(0.0, float(src.camera.theta_max), THETA_BINS + 1)
    t_idx = np.clip(np.digitize(theta.numpy(), t_edges) - 1, 0, THETA_BINS - 1)
    gh, gw = src.h // PATCH, src.w // PATCH
    theta_p = theta.numpy().reshape(gh, PATCH, gw, PATCH).mean((1, 3)).ravel()
    return cone, t_idx, t_edges, gh, gw, theta_p


def patch_pool(m, gh, gw, vmask=None):
    r = m.reshape(gh, PATCH, gw, PATCH).transpose(0, 2, 1, 3).reshape(gh * gw, -1)
    if vmask is None:
        return np.median(r, axis=1)
    v = vmask.reshape(gh, PATCH, gw, PATCH).transpose(0, 2, 1, 3).reshape(gh * gw, -1)
    out = np.full(gh * gw, np.nan)
    for k in range(gh * gw):
        if v[k].sum() >= 30:
            out[k] = np.median(r[k][v[k]])
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-seqs", required=True)
    p.add_argument("--eval-seqs", required=True)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    torch.manual_seed(SEED)

    train_sd = [SeqData(s.strip(), args.size, args.max_frames)
                for s in args.train_seqs.split(",") if s.strip()]
    eval_sd = [SeqData(s.strip(), args.size, args.max_frames)
               for s in args.eval_seqs.split(",") if s.strip()]
    for sd in train_sd + eval_sd:
        sd.ensure_cached()

    cone, t_idx, t_edges, gh, gw, theta_p = geometry(train_sd[0].src)

    X, A, Y = [], [], []
    for sd in train_sd:
        for n in sd.frames:
            gr, gz = sd.gt_range(n)
            d = sd.pred(n)
            valid = cone & (gz > 0) & (gr <= args.depth_max_m) & (d > 1e-6)
            r = np.log(gr) - np.log(np.clip(d, 1e-6, None))
            r_med = float(np.median(r[valid]))
            tgt = patch_pool(r - r_med, gh, gw, vmask=valid)
            dp = patch_pool(np.log(np.clip(d, 1e-6, None)), gh, gw)
            ok = np.isfinite(tgt)
            X.append(sd.feat(n)[ok])
            A.append(np.stack([np.sin(theta_p[ok]), np.cos(theta_p[ok]),
                               dp[ok]], -1))
            Y.append(tgt[ok])
    X = torch.from_numpy(np.concatenate(X)).float()
    A = torch.from_numpy(np.concatenate(A)).float()
    Y = torch.from_numpy(np.concatenate(Y)).float()
    print(f"[h2.3] {len(Y)} training patches from {len(train_sd)} sequences")

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
    summary: Dict = {"train_seqs": [sd.name for sd in train_sd],
                     "eval": {}, "config": vars(args),
                     "theta_bin_mid_deg": t_mid}
    for sd in eval_sd:
        s_b = np.zeros((THETA_BINS, nb_d)); s_a = np.zeros((THETA_BINS, nb_d))
        n_ = np.zeros((THETA_BINS, nb_d), dtype=np.int64)
        for n in sd.frames:
            gr, gz = sd.gt_range(n)
            d0 = sd.pred(n)
            dp = patch_pool(np.log(np.clip(d0, 1e-6, None)), gh, gw)
            with torch.no_grad():
                corr = head(torch.from_numpy(sd.feat(n)).float(),
                            torch.from_numpy(np.stack(
                                [np.sin(theta_p), np.cos(theta_p), dp],
                                -1)).float()).numpy().reshape(gh, gw)
            corr_px = torch.nn.functional.interpolate(
                torch.from_numpy(corr)[None, None].float(),
                size=(sd.src.h, sd.src.w), mode="bilinear",
                align_corners=False)[0, 0].numpy()
            d1 = d0 * np.exp(corr_px)
            for tag, d in (("before", d0), ("after", d1)):
                valid = cone & (gz > 0) & (gr <= args.depth_max_m) & (d > 1e-6)
                aligned = align_depth(d, gr, valid, mode="scale_shift")
                absrel = (np.abs(aligned - gr) / np.clip(gr, 1e-6, None))[valid]
                ti = t_idx[valid]
                di = np.clip(np.digitize(gr[valid], GT_DEPTH_EDGES) - 1,
                             0, nb_d - 1)
                flat = ti * nb_d + di
                acc = s_b if tag == "before" else s_a
                acc += np.bincount(flat, weights=absrel,
                                   minlength=THETA_BINS * nb_d
                                   ).reshape(THETA_BINS, nb_d)
                if tag == "before":
                    n_ += np.bincount(flat, minlength=THETA_BINS * nb_d
                                      ).reshape(THETA_BINS, nb_d)
        before = s_b / np.maximum(n_, 1)
        after = s_a / np.maximum(n_, 1)
        print(f"\n=== eval {sd.name}: BEFORE -> AFTER ===")
        for j in range(nb_d):
            row = " ".join(f"{before[i, j]:5.3f}>{after[i, j]:5.3f}"
                           for i in range(THETA_BINS))
            print(f"{GT_DEPTH_EDGES[j]:4.0f}-{GT_DEPTH_EDGES[j + 1]:2.0f} m  {row}")
        nr = [(i, j) for i in range(THETA_BINS) for j in range(nb_d)
              if t_mid[i] >= 38 and GT_DEPTH_EDGES[j + 1] <= 2.0]
        w = np.array([n_[i, j] for i, j in nr], float)
        b = float((np.array([before[i, j] for i, j in nr]) * w).sum() / w.sum())
        a = float((np.array([after[i, j] for i, j in nr]) * w).sum() / w.sum())
        print(f"near_rim: {b:.3f} -> {a:.3f} ({(a - b) / b * 100:+.1f}%)")
        summary["eval"][sd.name] = {"before": before.tolist(),
                                    "after": after.tolist(),
                                    "counts": n_.tolist(),
                                    "near_rim": {"before": b, "after": a}}

    if args.out:
        dst = Path(__file__).resolve().parents[1] / args.out \
            if not os.path.isabs(args.out) else Path(args.out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[h2.3] wrote {dst}")


if __name__ == "__main__":
    main()
