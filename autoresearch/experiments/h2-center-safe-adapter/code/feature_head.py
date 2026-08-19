"""H2.2: frozen-feature correction head vs the H2.1 table.

Protocol: ../protocol-h2.2-feature-head.md (committed before this ran). CPU.

Stage 1 caches DA3-Small final-block patch tokens per frame (frozen model,
deterministic). Stage 2 trains a ~25k-param zero-init head on train frames'
patch-level residuals and evaluates the corrected depth on held-out frames
under the protocol of record, next to the H2.1 table's result.

Usage (repo root):
    <venv>/bin/python autoresearch/experiments/h2-center-safe-adapter/code/feature_head.py \
        --split even_odd --out results/run_011_even_odd.json
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
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "h1-rim-pose-value" / "code"))

from adt_pose_value import AriaLocalPairs, DEFAULT_SEQ   # noqa: E402
from finetune.eval.metrics import align_depth            # noqa: E402

THETA_BINS = 8
GT_DEPTH_EDGES = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)
PATCH = 14
SEED = 0
SCRATCH = Path("/private/tmp/claude-501/-Users-fengjiazhang-Desktop-ADT-vggt-omega/"
               "c1591315-71f0-4ca2-9183-9f58fff82acc/scratchpad")
PRED_CACHE = SCRATCH / "h2_pred_cache"
FEAT_CACHE = SCRATCH / "h2_feat_cache"


def cache_all(src) -> None:
    """One frozen forward per frame, caching pred range and final-block tokens."""
    FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    PRED_CACHE.mkdir(parents=True, exist_ok=True)
    todo = [n for n in range(len(src.paths))
            if not (FEAT_CACHE / f"{n}.npy").exists()
            or not (PRED_CACHE / f"{n}.npy").exists()]
    if not todo:
        return
    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device="cpu",
                        variant="small")
    bb.install(None, src.camera, (src.h, src.w), patch_undistort=False,
               border_token=False, dpt_grid=False, depth_convention="range")
    vit = bb._vit()
    blocks = None
    for name in ("blocks", "layers"):
        blocks = getattr(vit, name, None)
        if blocks is not None:
            break
    assert blocks is not None, "cannot find transformer blocks on the ViT"
    grabbed: Dict[str, torch.Tensor] = {}

    def hook(_m, _i, out):
        grabbed["tok"] = out[0] if isinstance(out, tuple) else out

    h = blocks[-1].register_forward_hook(hook)
    n_patch = (src.h // PATCH) * (src.w // PATCH)
    for n in todo:
        t0 = time.time()
        with torch.no_grad():
            pr = bb.forward(src.image(n)[None, None])
        pr.require_convention("range")
        np.save(PRED_CACHE / f"{n}.npy", pr.depth[0].numpy())
        tok = grabbed["tok"]
        tok = tok.reshape(-1, tok.shape[-1])
        assert tok.shape[0] >= n_patch, f"tokens {tok.shape} < {n_patch} patches"
        # patch tokens are the LAST n_patch (cls/registers lead) — recorded
        # assumption; sanity: leftover count must be small (< 16)
        extra = tok.shape[0] - n_patch
        assert extra < 16, f"unexpected token layout: {extra} extra tokens"
        np.save(FEAT_CACHE / f"{n}.npy",
                tok[extra:].to(torch.float32).numpy())
        print(f"  cached feats {n} ({time.time() - t0:4.1f}s, extra={extra})",
              flush=True)
    h.remove()


class Head(nn.Module):
    def __init__(self, c_in: int):
        super().__init__()
        self.ln = nn.LayerNorm(c_in)
        self.mlp = nn.Sequential(nn.Linear(c_in + 3, 64), nn.GELU(),
                                 nn.Linear(64, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feats, aux):
        return self.mlp(torch.cat([self.ln(feats), aux], dim=-1)).squeeze(-1)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", default=DEFAULT_SEQ)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--split", default="even_odd", choices=["even_odd", "halves"])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    torch.manual_seed(SEED)

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
    frames = [n for n in range(len(src.paths))
              if os.path.basename(src.paths[n]).replace(".jpg", "") in depth_paths]
    if args.split == "even_odd":
        train = [n for k, n in enumerate(frames) if k % 2 == 0]
        test = [n for k, n in enumerate(frames) if k % 2 == 1]
    else:
        half = len(frames) // 2
        train, test = frames[:half], frames[half:]
    print(f"[h2.2] split {args.split}: {len(train)} train / {len(test)} test")

    cache_all(src)
    preds = {n: np.load(PRED_CACHE / f"{n}.npy") for n in frames}
    feats = {n: np.load(FEAT_CACHE / f"{n}.npy") for n in frames}
    c_in = feats[frames[0]].shape[-1]

    def gt_range(n: int):
        stem = os.path.basename(src.paths[n]).replace(".jpg", "")
        gz = np.load(depth_paths[stem]).astype(np.float32)
        gz = torch.nn.functional.interpolate(
            torch.from_numpy(gz)[None, None], size=(src.h, src.w),
            mode="nearest")[0, 0].numpy() / 1000.0
        return gz / np.clip(cos_t, 1e-6, None), gz

    def patch_pool(m: np.ndarray, red="median", vmask=None) -> np.ndarray:
        r = m.reshape(gh, PATCH, gw, PATCH).transpose(0, 2, 1, 3).reshape(gh * gw, -1)
        if vmask is None:
            return np.median(r, axis=1) if red == "median" else r.mean(1)
        v = vmask.reshape(gh, PATCH, gw, PATCH).transpose(0, 2, 1, 3).reshape(gh * gw, -1)
        out = np.full(gh * gw, np.nan)
        for k in range(gh * gw):
            sel = v[k]
            if sel.sum() >= 30:
                out[k] = np.median(r[k][sel])
        return out

    # ---- training set: per-patch residual targets ----
    X, A, Y = [], [], []
    for n in train:
        gr, gz = gt_range(n)
        d = preds[n]
        valid = cone & (gz > 0) & (gr <= args.depth_max_m) & (d > 1e-6)
        r = np.log(gr) - np.log(np.clip(d, 1e-6, None))
        r_med = float(np.median(r[valid]))
        tgt = patch_pool(r - r_med, vmask=valid)
        dp = patch_pool(np.log(np.clip(d, 1e-6, None)))
        ok = np.isfinite(tgt)
        X.append(feats[n][ok])
        A.append(np.stack([np.sin(theta_p[ok]), np.cos(theta_p[ok]), dp[ok]], -1))
        Y.append(tgt[ok])
    X = torch.from_numpy(np.concatenate(X)).float()
    A = torch.from_numpy(np.concatenate(A)).float()
    Y = torch.from_numpy(np.concatenate(Y)).float()
    print(f"[h2.2] {len(Y)} training patches, feat dim {c_in}")

    head = Head(c_in)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    prev = float("inf")
    for ep in range(args.epochs):
        opt.zero_grad()
        loss = (head(X, A) - Y).abs().mean()
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d} L1 {loss.item():.4f}", flush=True)
        if abs(prev - loss.item()) < 1e-6:
            print(f"  plateau at epoch {ep}")
            break
        prev = loss.item()

    # ---- evaluate ----
    nb_d = len(GT_DEPTH_EDGES) - 1

    def joint(correct: bool):
        s_ = np.zeros((THETA_BINS, nb_d))
        n_ = np.zeros((THETA_BINS, nb_d), dtype=np.int64)
        for n in test:
            gr, gz = gt_range(n)
            d = preds[n].copy()
            if correct:
                dp = patch_pool(np.log(np.clip(d, 1e-6, None)))
                with torch.no_grad():
                    corr = head(
                        torch.from_numpy(feats[n]).float(),
                        torch.from_numpy(np.stack(
                            [np.sin(theta_p), np.cos(theta_p), dp], -1)).float(),
                    ).numpy().reshape(gh, gw)
                corr_px = torch.nn.functional.interpolate(
                    torch.from_numpy(corr)[None, None].float(),
                    size=(src.h, src.w), mode="bilinear",
                    align_corners=False)[0, 0].numpy()
                d = d * np.exp(corr_px)
            valid = cone & (gz > 0) & (gr <= args.depth_max_m) & (d > 1e-6)
            aligned = align_depth(d, gr, valid, mode="scale_shift")
            absrel = (np.abs(aligned - gr) / gr)[valid]
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
    t_mid = [math.degrees(0.5 * (t_edges[i] + t_edges[i + 1]))
             for i in range(THETA_BINS)]
    print("\nheld-out joint AbsRel BEFORE -> AFTER (rows=GT depth, cols=theta):")
    print("depth\\theta  " + " ".join(f"{t:12.1f}" for t in t_mid))
    for j in range(nb_d):
        row = " ".join(f"{before[i, j]:5.3f}>{after[i, j]:5.3f}"
                       for i in range(THETA_BINS))
        print(f"{GT_DEPTH_EDGES[j]:4.0f}-{GT_DEPTH_EDGES[j + 1]:2.0f} m  {row}")

    zones = {
        "near_rim(<=2m,>=38deg)": [(i, j) for i in range(THETA_BINS)
                                   for j in range(nb_d)
                                   if t_mid[i] >= 38 and GT_DEPTH_EDGES[j + 1] <= 2.0],
        "near_center(<=2m,<=11deg)": [(i, j) for i in range(THETA_BINS)
                                      for j in range(nb_d)
                                      if t_mid[i] <= 11 and GT_DEPTH_EDGES[j + 1] <= 2.0],
        "center(<=11deg)": [(i, j) for i in range(THETA_BINS)
                            for j in range(nb_d) if t_mid[i] <= 11],
        "far(>=3m)": [(i, j) for i in range(THETA_BINS) for j in range(nb_d)
                      if GT_DEPTH_EDGES[j] >= 3.0],
    }
    summary = {"split": args.split, "before": before.tolist(),
               "after": after.tolist(), "counts": counts.tolist(),
               "theta_bin_mid_deg": t_mid, "zones": {},
               "n_train_patches": int(len(Y)), "config": vars(args)}
    for name, cells in zones.items():
        w = np.array([counts[i, j] for i, j in cells], float)
        b = float((np.array([before[i, j] for i, j in cells]) * w).sum() / w.sum())
        a = float((np.array([after[i, j] for i, j in cells]) * w).sum() / w.sum())
        summary["zones"][name] = {"before": b, "after": a}
        print(f"{name}: {b:.3f} -> {a:.3f}  ({(a - b) / b * 100:+.1f}%)")

    if args.out:
        dst = Path(__file__).resolve().parents[1] / args.out \
            if not os.path.isabs(args.out) else Path(args.out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[h2.2] wrote {dst}")


if __name__ == "__main__":
    main()
