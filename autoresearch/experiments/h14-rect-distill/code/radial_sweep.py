"""H18.5 — how many frames, and how spread out, does the 16-number fit need?

H18.3 left an asymmetry it could not explain away. The Apartment-fitted radial
curve (240 frames, four sequences, real motion) is bit-identical under a
range-matched refit; the LiteOffice one (120 frames, two near-static sequences)
moves a lot and gets worse. One fitting set estimates the object and the other
does not, and nothing measured so far says how much is enough — which is exactly
the number a calibration procedure has to quote.

Two axes, crossed:

    frames drawn   2, 4, 8, 15, 30, 60, 120, 240
    how drawn      `spread` (evenly over the four sequences) | `single` (one)

`single` vs `spread` at a matched frame count is what separates "needs more
pixels" from "needs more viewpoints". LiteOffice has both confounded.

Five independent draws per cell, and two readouts:

    stability   mean absolute PAIRWISE difference in a(theta) between draws.
                Needs no held-out data — it measures the estimator's spread.
    transfer    near_rim gain on the two held-out Apartment sequences and the
                two LiteOffice ones, mean +- sd over the draws.

Cost is one GPU pass. Fitting samples are cached subsampled (20k pixels per
frame); evaluation frames are cached whole, because scoring needs the per-frame
scale+shift alignment. Every fit and every evaluation after that is numpy, so
the whole 2 x 8 x 5 sweep is essentially free.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[3]))
sys.path.insert(0, str(_HERE.parents[1] / "h1-rim-pose-value" / "code"))
sys.path.append(str(_HERE.parents[1] / "h5-rim-finetune" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "common"))

import importlib.util as _ilu  # noqa: E402
import upright as U  # noqa: E402


def _load(name, path):
    sp = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(sp); sp.loader.exec_module(m); return m


_H5 = _HERE.parents[1] / "h5-rim-finetune" / "code"
Seq = _load("h5_train", _H5 / "train.py").Seq
_ev = _load("h5_eval", _H5 / "eval_lora.py")
THETA_BINS, EDG = _ev.THETA_BINS, _ev.GT_DEPTH_EDGES
from finetune.eval.metrics import align_depth  # noqa: E402

A = "/user/f.zhang2/Documents/projectaria_tools_adt_data_clean"
L = "/user/f.zhang2/Documents/adt_liteoffice_extracted"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--arm", default="omega110")
    p.add_argument("--fit-seqs", default=",".join(
        f"{A}/Apartment_release_clean_{s}_M1292" for s in ("seq131", "seq133", "seq134", "seq135")))
    p.add_argument("--eval-seqs", default=",".join(
        [f"{A}/Apartment_release_clean_seq136_M1292",
         f"{A}/Apartment_release_decoration_seq132_M1292",
         f"{L}/DinoToy_seq030", f"{L}/BlackCeramicBowl_seq030"]))
    p.add_argument("--counts", default="2,4,8,15,30,60,120,240")
    p.add_argument("--draws", type=int, default=5)
    p.add_argument("--px-per-frame", type=int, default=20000)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--variant", default="small")
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    rng = np.random.default_rng(a.seed)

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device=a.device, variant=a.variant)

    def install(cam):
        bb.install(None, cam, (a.size, a.size), patch_undistort=False,
                   border_token=False, dpt_grid=False, depth_convention="z")

    def bins_of(cam):
        th = cam.incidence_grid(a.size, a.size)
        e = np.linspace(0.0, float(cam.theta_max), THETA_BINS + 1)
        return (th, torch.cos(th),
                np.clip(np.digitize(th.numpy(), e) - 1, 0, THETA_BINS - 1),
                0.5 * (e[:-1] + e[1:]) * 180 / np.pi,
                (th <= cam.theta_max).numpy())

    # ---- pass 1: fitting samples, one entry per FRAME so draws are by frame --
    root = Path(a.cache_root) / a.arm
    frames = []           # (seq_index, logp[20k], logt[20k], bin[20k])
    for si, sd in enumerate(a.fit_seqs.split(",")):
        s = Seq(sd, a.size, a.max_frames)
        th, cos_t, t_idx, _, _ = bins_of(s.src.camera)
        d = root / s.name
        man = json.loads((d / "manifest.json").read_text())
        cov = np.load(d / "covered.npy")
        off = man.get("log_offset_vs_raw", {})
        install(s.src.camera)
        for n in s.frames:
            stem = s.stem(n)
            tgt = np.load(d / "npz" / f"{stem}.npz")["depth"].astype(np.float32)
            tgt = tgt * float(np.exp(-off.get(stem, 0.0)))
            with torch.no_grad():
                pr = U.forward_range(bb, s.src.image(n).to(a.device),
                                     cos_t.to(a.device)).float().cpu().numpy()
            m = np.flatnonzero((cov & (tgt > 1e-6) & (pr > 1e-6)).ravel())
            if m.size > a.px_per_frame:
                m = rng.choice(m, a.px_per_frame, replace=False)
            frames.append((si, np.log(pr.ravel()[m]), np.log(tgt.ravel()[m]),
                           t_idx.ravel()[m]))
        print(f"[sweep] fit pass {s.name}: {len(s.frames)} frames")
    n_seq = len(a.fit_seqs.split(","))
    print(f"[sweep] {len(frames)} fitting frames over {n_seq} sequences")

    # ---- pass 2: evaluation frames, kept whole -----------------------------
    ev = {}
    for sd in a.eval_seqs.split(","):
        s = Seq(sd, a.size, a.max_frames)
        th, cos_t, t_idx, t_mid, cone = bins_of(s.src.camera)
        install(s.src.camera)
        P, G = [], []
        for n in s.frames:
            with torch.no_grad():
                pr = U.forward_range(bb, s.src.image(n).to(a.device),
                                     cos_t.to(a.device)).float().cpu().numpy()
            P.append(pr.astype(np.float16)); G.append(s.gt_range(n, cos_t).numpy().astype(np.float16))
        ev[s.name] = dict(P=P, G=G, t_idx=t_idx, t_mid=t_mid, cone=cone)
        print(f"[sweep] eval pass {s.name}: {len(P)} frames")
    del bb
    torch.cuda.empty_cache()

    # ---- everything below is numpy ----------------------------------------
    NB = len(EDG) - 1

    def fit(idx):
        LP = np.concatenate([frames[i][1] for i in idx])
        LT = np.concatenate([frames[i][2] for i in idx])
        BN = np.concatenate([frames[i][3] for i in idx])
        C = np.zeros((THETA_BINS, 2))
        for b in range(THETA_BINS):
            m = BN == b
            if m.sum() < 200:
                C[b] = (1.0, 0.0); continue
            X = np.stack([LP[m], np.ones(m.sum())], 1)
            C[b] = np.linalg.lstsq(X, LT[m], rcond=None)[0]
        return C

    def rim_gain(C, name):
        d = ev[name]
        t_idx, t_mid, cone = d["t_idx"], d["t_mid"], d["cone"]
        acc = {k: [np.zeros((THETA_BINS, NB)), np.zeros((THETA_BINS, NB))]
               for k in ("frozen", "corr")}
        for pr16, gt16 in zip(d["P"], d["G"]):
            pr = pr16.astype(np.float32); gt = gt16.astype(np.float32)
            lp = np.log(np.clip(pr, 1e-6, None))
            for k, dd in (("frozen", pr),
                          ("corr", np.exp(C[t_idx, 0] * lp + C[t_idx, 1]))):
                v = cone & (gt > 0) & (gt <= a.depth_max_m) & (dd > 1e-6)
                if v.sum() < 1000:
                    continue
                al = align_depth(dd, gt, v, mode="scale_shift")
                ar = (np.abs(al - gt) / np.clip(gt, 1e-6, None))[v]
                di = np.clip(np.digitize(gt[v], EDG) - 1, 0, NB - 1)
                flat = t_idx[v] * NB + di
                acc[k][0] += np.bincount(flat, weights=ar,
                                         minlength=THETA_BINS * NB).reshape(THETA_BINS, NB)
                acc[k][1] += np.bincount(flat, minlength=THETA_BINS * NB).reshape(THETA_BINS, NB)
        out = {}
        for k, (s_, n_) in acc.items():
            tab = np.divide(s_, n_, out=np.zeros_like(s_), where=n_ > 0)
            cells = [(i, j) for i in range(THETA_BINS) for j in range(NB)
                     if t_mid[i] >= 38 and EDG[j + 1] <= 2.0]
            w = np.array([n_[i, j] for i, j in cells], float)
            out[k] = float((np.array([tab[i, j] for i, j in cells]) * w).sum()
                           / max(w.sum(), 1.0))
        return 100 * (out["corr"] / out["frozen"] - 1)

    by_seq = {}
    for i, (si, *_rest) in enumerate(frames):
        by_seq.setdefault(si, []).append(i)
    counts = [int(x) for x in a.counts.split(",")]
    names = list(ev)
    res = {}
    print(f"\n{'mode':>7s}{'N':>5s}{'stab |da|':>11s}" + "".join(f"{n[:16]:>18s}" for n in names))
    for mode in ("spread", "single"):
        for N in counts:
            if mode == "single" and N > len(by_seq[0]):
                continue
            draws, gains = [], []
            for k in range(a.draws):
                r = np.random.default_rng(a.seed + 1000 * k + N)
                if mode == "spread":
                    per = max(1, N // n_seq)
                    idx = []
                    for si in by_seq:
                        take = min(per, len(by_seq[si]))
                        idx += list(r.choice(by_seq[si], take, replace=False))
                    idx = idx[:N] if len(idx) >= N else idx
                else:
                    si = int(r.integers(n_seq))
                    idx = list(r.choice(by_seq[si], min(N, len(by_seq[si])), replace=False))
                C = fit(idx)
                draws.append(C[:, 0])
                gains.append([rim_gain(C, nm) for nm in names])
            D = np.array(draws)
            stab = float(np.mean([np.abs(D[i] - D[j]).mean()
                                  for i in range(len(D)) for j in range(i + 1, len(D))]))
            Gm = np.array(gains)
            res[f"{mode}_{N}"] = {"stability": stab, "coef_a_mean": D.mean(0).tolist(),
                                  "gain_mean": Gm.mean(0).tolist(), "gain_sd": Gm.std(0).tolist(),
                                  "n_frames": N, "mode": mode}
            print(f"{mode:>7s}{N:>5d}{stab:>11.3f}"
                  + "".join(f"{Gm.mean(0)[j]:>12.1f}±{Gm.std(0)[j]:<5.1f}" for j in range(len(names))))

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"eval_order": names, "cells": res,
                                           "config": vars(a)}, indent=1))


if __name__ == "__main__":
    main()
