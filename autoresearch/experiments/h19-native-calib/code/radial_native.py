"""H19 - calibrate the radial curve on the device it will run on, with no labels.

H18.2 found that a 16-number radial curve is what crosses a room; H18.5 found
that ~30 frames fit it. Both fitted on four Apartment sequences and applied the
result to LiteOffice, a different room AND a different pair of glasses.

The fit was already label-free - it regresses the cached omega110 teacher on the
frozen prediction, and ground truth enters only at scoring. What is untested is
fitting on the *target device*. The curve is a property of the lens, and in
deployment you have footage from the camera in your hand, not from M1292.

So: fit on LiteOffice's own frames, apply to LiteOffice's other sequence, and
compare against the Apartment-fitted curve already on record (-26.0% on DinoToy,
-8.9% on BlackCeramicBowl). Zero ground truth in any fitting loop.

`global` (2 params, no theta dependence) is carried through every cell as the
control that decides the reading: if it matches `radial`, the native fit is a
rescale and not a radial relation.

Also measures the stability of a 30-frame DinoToy fit, to test H18.5's own
follow-up prediction - that near-static footage carries less per frame than the
walking-around Apartment footage, where 30 single-sequence frames gave 0.060.

Cost is one GPU pass over six sequences; every fit and score after that is numpy.
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

APT = [f"{A}/Apartment_release_clean_{s}_M1292" for s in ("seq131", "seq133", "seq134", "seq135")]
LITE = [f"{L}/DinoToy_seq030", f"{L}/BlackCeramicBowl_seq030"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--arm", default="omega110")
    p.add_argument("--fit-seqs", default=",".join(APT + LITE))
    p.add_argument("--eval-seqs", default=",".join(
        [f"{A}/Apartment_release_clean_seq136_M1292",
         f"{A}/Apartment_release_decoration_seq132_M1292"] + LITE))
    p.add_argument("--px-per-frame", type=int, default=20000)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--variant", default="small")
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--fit-pred-range", default="",
                   help="lo,hi metres on the FROZEN PREDICTION, fitting pixels "
                        "only. Uses the prediction, not GT, so it stays "
                        "label-free. Matches the fitting range across devices.")
    p.add_argument("--stability-draws", type=int, default=5)
    p.add_argument("--stability-frames", type=int, default=30)
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
        return (torch.cos(th),
                np.clip(np.digitize(th.numpy(), e) - 1, 0, THETA_BINS - 1),
                0.5 * (e[:-1] + e[1:]) * 180 / np.pi,
                (th <= cam.theta_max).numpy())

    def predict(s, n, cos_t):
        with torch.no_grad():
            return U.forward_range(bb, s.src.image(n).to(a.device),
                                   cos_t.to(a.device)).float().cpu().numpy()

    # ---- pass 1: fitting samples, one entry per frame, keyed by sequence ----
    root = Path(a.cache_root) / a.arm
    fit_frames = {}
    lo = hi = None
    if a.fit_pred_range:
        lo, hi = (float(x) for x in a.fit_pred_range.split(","))
    for sd in a.fit_seqs.split(","):
        s = Seq(sd, a.size, a.max_frames)
        cos_t, t_idx, _, _ = bins_of(s.src.camera)
        d = root / s.name
        man = json.loads((d / "manifest.json").read_text())
        cov = np.load(d / "covered.npy")
        off = man.get("log_offset_vs_raw", {})
        install(s.src.camera)
        got = []
        for n in s.frames:
            stem = s.stem(n)
            tgt = np.load(d / "npz" / f"{stem}.npz")["depth"].astype(np.float32)
            tgt = tgt * float(np.exp(-off.get(stem, 0.0)))
            pr = predict(s, n, cos_t)
            ok = cov & (tgt > 1e-6) & (pr > 1e-6)
            if lo is not None:
                ok = ok & (pr >= lo) & (pr <= hi)
            m = np.flatnonzero(ok.ravel())
            if m.size < 200:
                continue
            if m.size > a.px_per_frame:
                m = rng.choice(m, a.px_per_frame, replace=False)
            got.append((np.log(pr.ravel()[m]), np.log(tgt.ravel()[m]), t_idx.ravel()[m]))
        fit_frames[s.name] = got
        print(f"[h19] fit pass {s.name}: {len(got)} usable frames", flush=True)

    # ---- pass 2: evaluation frames, kept whole -----------------------------
    ev = {}
    for sd in a.eval_seqs.split(","):
        s = Seq(sd, a.size, a.max_frames)
        cos_t, t_idx, t_mid, cone = bins_of(s.src.camera)
        install(s.src.camera)
        P, G = [], []
        for n in s.frames:
            P.append(predict(s, n, cos_t).astype(np.float16))
            G.append(s.gt_range(n, cos_t).numpy().astype(np.float16))
        ev[s.name] = dict(P=P, G=G, t_idx=t_idx, t_mid=t_mid, cone=cone)
        print(f"[h19] eval pass {s.name}: {len(P)} frames", flush=True)
    del bb
    torch.cuda.empty_cache()

    # ---- everything below is numpy -----------------------------------------
    NB = len(EDG) - 1

    def fit(frs, radial=True):
        LP = np.concatenate([f[0] for f in frs])
        LT = np.concatenate([f[1] for f in frs])
        BN = np.concatenate([f[2] for f in frs])
        C = np.zeros((THETA_BINS, 2))
        if not radial:
            X = np.stack([LP, np.ones_like(LP)], 1)
            C[:] = np.linalg.lstsq(X, LT, rcond=None)[0]
            return C
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
        return 100.0 * (out["corr"] / out["frozen"] - 1.0)

    def nm(path):
        return Path(path).name

    FITSETS = {
        "apartment": [nm(x) for x in APT],
        "dino":      [nm(LITE[0])],
        "bowl":      [nm(LITE[1])],
        "lite_both": [nm(x) for x in LITE],
    }
    ev_names = [nm(x) for x in a.eval_seqs.split(",")]

    res = {"cells": {}, "config": vars(a), "eval_order": ev_names}
    print(f"\n{'fitset':>10s} {'form':>7s} " + " ".join(f"{n[:22]:>23s}" for n in ev_names))
    for fs, members in FITSETS.items():
        frs = [f for mname in members for f in fit_frames.get(mname, [])]
        if not frs:
            print(f"[h19] SKIP {fs}: no fitting frames"); continue
        for form in ("radial", "global"):
            C = fit(frs, radial=(form == "radial"))
            gains = [rim_gain(C, n) for n in ev_names]
            key = f"{fs}_{form}"
            res["cells"][key] = {"fitset": fs, "form": form,
                                 "n_frames": len(frs),
                                 "coef_a": C[:, 0].tolist(),
                                 "gain": gains,
                                 "members": members}
            print(f"{fs:>10s} {form:>7s} " + " ".join(f"{g:>22.1f}%" for g in gains), flush=True)

    # ---- B3: is near-static footage worse per frame? -----------------------
    print()
    stab = {}
    for fs in ("dino", "bowl", "apartment"):
        pool = [f for mname in FITSETS[fs] for f in fit_frames.get(mname, [])]
        k = min(a.stability_frames, len(pool))
        if len(pool) < k + 1:
            continue
        r2 = np.random.default_rng(a.seed + 7)
        D = np.stack([fit([pool[i] for i in r2.choice(len(pool), k, replace=False)])[:, 0]
                      for _ in range(a.stability_draws)])
        v = float(np.mean([np.abs(D[i] - D[j]).mean()
                           for i in range(len(D)) for j in range(i + 1, len(D))]))
        stab[fs] = {"frames": k, "pool": len(pool), "mean_pairwise_da": v}
        print(f"[h19] stability {fs:>10s}: {k} of {len(pool)} frames, |da| = {v:.4f}")
    res["stability"] = stab
    print("[h19] reference: 30 frames from ONE Apartment sequence gave 0.060 (H18.5)")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(f"[h19] wrote {a.out}")


if __name__ == "__main__":
    main()
