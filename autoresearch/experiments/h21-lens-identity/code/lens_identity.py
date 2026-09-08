"""H21 - is the correction curve a property of the LENS, or largely lens-independent?

Both reports say the transferable part is "a recalibration of depth against
viewing angle - a property of the lens". Nobody has changed the lens with
everything else held fixed; real devices confound lens with room, scene, wearer
and motion. H15's lens family fixes that: many lenses over ONE cone filling the
same disc, so a warp between any two is a pure radial re-distribution of the
same rays - no void, no extrapolation, same content.

Costs no teacher inference. The cache stores range, and grid_between resamples
by ray, so range is invariant when the lens changes at a fixed cone: the teacher
target under lens L is an exact `nearest` resample of the cached target. Only
the frozen prediction is recomputed, because the network genuinely sees a
differently warped image - which is the point.

Control detail that decides the reading: EVERY arm, including aria_kb4, is
warped once from the real Aria camera. Leaving the aria arm un-warped would
confound resampling blur with lens shape.
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
sys.path.insert(0, str(_HERE.parents[1] / "h15-lens-holdout" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "common"))

import importlib.util as _ilu  # noqa: E402
import upright as U  # noqa: E402
import lens_family as LF  # noqa: E402


def _load(name, path):
    sp = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(sp); sp.loader.exec_module(m); return m


_H5 = _HERE.parents[1] / "h5-rim-finetune" / "code"
Seq = _load("h5_train", _H5 / "train.py").Seq
_ev = _load("h5_eval", _H5 / "eval_lora.py")
THETA_BINS, EDG = _ev.THETA_BINS, _ev.GT_DEPTH_EDGES
from finetune.eval.metrics import align_depth  # noqa: E402

A = "/user/f.zhang2/Documents/projectaria_tools_adt_data_clean"
APT = [f"{A}/Apartment_release_clean_{s}_M1292" for s in ("seq131", "seq133", "seq134", "seq135")]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--arm", default="omega110")
    p.add_argument("--lenses", default="aria_kb4,equidistant,stereographic,equisolid")
    p.add_argument("--fit-seqs", default=",".join(APT))
    p.add_argument("--eval-seqs", default=",".join(
        [f"{A}/Apartment_release_clean_seq136_M1292",
         f"{A}/Apartment_release_decoration_seq132_M1292"]))
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
    lenses = a.lenses.split(",")

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device=a.device, variant=a.variant)

    def bins_of(cam):
        th = cam.incidence_grid(a.size, a.size)
        e = np.linspace(0.0, float(cam.theta_max), THETA_BINS + 1)
        return (torch.cos(th),
                np.clip(np.digitize(th.numpy(), e) - 1, 0, THETA_BINS - 1),
                0.5 * (e[:-1] + e[1:]) * 180 / np.pi,
                (th <= cam.theta_max).numpy())

    # ---- pass 1: fitting samples, per lens -------------------------------
    root = Path(a.cache_root) / a.arm
    fit = {L: [] for L in lenses}
    for sd in a.fit_seqs.split(","):
        s = Seq(sd, a.size, a.max_frames)
        src = s.src.camera
        fam = LF.lens_family(lenses, a.size, float(src.theta_max))
        # one warp per lens, reused for every frame of this sequence
        W = {L: LF.grid_between(src, fam[L]) for L in lenses}
        B = {L: bins_of(fam[L]) for L in lenses}
        d = root / s.name
        man = json.loads((d / "manifest.json").read_text())
        cov = torch.from_numpy(np.load(d / "covered.npy").astype(np.float32))
        off = man.get("log_offset_vs_raw", {})
        for L in lenses:
            g, valid = W[L]
            cov_L = (LF.warp(cov, g, mode="nearest") > 0.5).numpy() & valid.numpy()
            bb.install(None, fam[L], (a.size, a.size), patch_undistort=False,
                       border_token=False, dpt_grid=False, depth_convention="z")
            cos_t, t_idx, _, _ = B[L]
            for n in s.frames:
                stem = s.stem(n)
                tgt = np.load(d / "npz" / f"{stem}.npz")["depth"].astype(np.float32)
                tgt = tgt * float(np.exp(-off.get(stem, 0.0)))
                # range is invariant under a fixed-cone lens change: resample it
                tgt_L = LF.warp(torch.from_numpy(tgt), g, mode="nearest").numpy()
                img_L = LF.warp(s.src.image(n), g, mode="bilinear")
                with torch.no_grad():
                    pr = U.forward_range(bb, img_L.to(a.device),
                                         cos_t.to(a.device)).float().cpu().numpy()
                m = np.flatnonzero((cov_L & (tgt_L > 1e-6) & (pr > 1e-6)).ravel())
                if m.size < 200:
                    continue
                if m.size > a.px_per_frame:
                    m = rng.choice(m, a.px_per_frame, replace=False)
                fit[L].append((np.log(pr.ravel()[m]), np.log(tgt_L.ravel()[m]),
                               t_idx.ravel()[m]))
        print(f"[h21] fit {s.name}: " +
              " ".join(f"{L}={len(fit[L])}" for L in lenses), flush=True)

    # ---- pass 2: evaluation frames, per lens ------------------------------
    ev = {}
    for sd in a.eval_seqs.split(","):
        s = Seq(sd, a.size, a.max_frames)
        src = s.src.camera
        fam = LF.lens_family(lenses, a.size, float(src.theta_max))
        W = {L: LF.grid_between(src, fam[L]) for L in lenses}
        for L in lenses:
            g, valid = W[L]
            cos_t, t_idx, t_mid, cone = bins_of(fam[L])
            bb.install(None, fam[L], (a.size, a.size), patch_undistort=False,
                       border_token=False, dpt_grid=False, depth_convention="z")
            P, G = [], []
            for n in s.frames:
                img_L = LF.warp(s.src.image(n), g, mode="bilinear")
                with torch.no_grad():
                    pr = U.forward_range(bb, img_L.to(a.device),
                                         cos_t.to(a.device)).float().cpu().numpy()
                gt = s.gt_range(n, torch.cos(src.incidence_grid(a.size, a.size)))
                gt_L = LF.warp(gt, g, mode="nearest").numpy()
                P.append(pr.astype(np.float16)); G.append(gt_L.astype(np.float16))
            ev[(s.name, L)] = dict(P=P, G=G, t_idx=t_idx, t_mid=t_mid,
                                   cone=cone & valid.numpy())
        print(f"[h21] eval {s.name}: {len(s.frames)} frames x {len(lenses)} lenses", flush=True)
    del bb
    torch.cuda.empty_cache()

    # ---- numpy ------------------------------------------------------------
    NB = len(EDG) - 1

    def fit_curve(frs, radial=True):
        LP = np.concatenate([f[0] for f in frs]); LT = np.concatenate([f[1] for f in frs])
        BN = np.concatenate([f[2] for f in frs])
        C = np.zeros((THETA_BINS, 2))
        if not radial:
            C[:] = np.linalg.lstsq(np.stack([LP, np.ones_like(LP)], 1), LT, rcond=None)[0]
            return C
        for b in range(THETA_BINS):
            m = BN == b
            if m.sum() < 200:
                C[b] = (1.0, 0.0); continue
            C[b] = np.linalg.lstsq(np.stack([LP[m], np.ones(m.sum())], 1), LT[m], rcond=None)[0]
        return C

    def rim_gain(C, key):
        d = ev[key]
        t_idx, t_mid, cone = d["t_idx"], d["t_mid"], d["cone"]
        acc = {k: [np.zeros((THETA_BINS, NB)), np.zeros((THETA_BINS, NB))]
               for k in ("frozen", "corr")}
        for pr16, gt16 in zip(d["P"], d["G"]):
            pr = pr16.astype(np.float32); gt = gt16.astype(np.float32)
            lp = np.log(np.clip(pr, 1e-6, None))
            for k, dd in (("frozen", pr), ("corr", np.exp(C[t_idx, 0] * lp + C[t_idx, 1]))):
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
        o = {}
        for k, (s_, n_) in acc.items():
            tab = np.divide(s_, n_, out=np.zeros_like(s_), where=n_ > 0)
            cells = [(i, j) for i in range(THETA_BINS) for j in range(NB)
                     if t_mid[i] >= 38 and EDG[j + 1] <= 2.0]
            w = np.array([n_[i, j] for i, j in cells], float)
            o[k] = float((np.array([tab[i, j] for i, j in cells]) * w).sum() / max(w.sum(), 1.0))
        return 100.0 * (o["corr"] / o["frozen"] - 1.0)

    curves = {L: fit_curve(fit[L]) for L in lenses}
    glob = {L: fit_curve(fit[L], radial=False) for L in lenses}
    ev_seqs = [Path(x).name for x in a.eval_seqs.split(",")]

    print("\n[h21] fitted a(theta) per lens")
    for L in lenses:
        print(f"  {L:>14s}  " + " ".join(f"{v:.3f}" for v in curves[L][:, 0]))
    pw = [(La, Lb, float(np.abs(curves[La][:, 0] - curves[Lb][:, 0]).mean()))
          for i, La in enumerate(lenses) for Lb in lenses[i + 1:]]
    mean_pw = float(np.mean([d for _, _, d in pw]))
    print(f"\n[h21] pairwise |da| between lens curves: mean {mean_pw:.4f}"
          f"  (H20 noise floor 0.069)")
    for La, Lb, d_ in pw:
        print(f"    {La:>14s} vs {Lb:<14s} {d_:.4f}")

    res = {"lenses": lenses, "eval_seqs": ev_seqs, "config": vars(a),
           "coef_a": {L: curves[L][:, 0].tolist() for L in lenses},
           "pairwise_da": [{"a": x, "b": y, "d": z} for x, y, z in pw],
           "mean_pairwise_da": mean_pw, "matrix": {}, "global_diag": {}}

    for seq in ev_seqs:
        print(f"\n[h21] {seq}: rows = curve fitted on, cols = applied to")
        hdr = "fit / apply"
        print(hdr.rjust(14) + " " + " ".join(f"{L[:12]:>13s}" for L in lenses))
        for La in lenses:
            row = [rim_gain(curves[La], (seq, Lb)) for Lb in lenses]
            res["matrix"][f"{seq}|{La}"] = row
            print(f"{La:>14s} " + " ".join(f"{g:>12.1f}%" for g in row))
        gl = [rim_gain(glob[L], (seq, L)) for L in lenses]
        res["global_diag"][seq] = gl
        print(f"{'global (diag)':>14s} " + " ".join(f"{g:>12.1f}%" for g in gl))

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(f"\n[h21] wrote {a.out}")


if __name__ == "__main__":
    main()
