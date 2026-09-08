"""H20 - is it the staticness, or is it the device?

H19 found the Apartment-fitted curve beats LiteOffice's own on LiteOffice by
~10x, and I published a mechanism for it: what the fit needs is motion, not a
lens match. That rests on calling LiteOffice "near-static", which had never been
measured - it was inherited from H9's remark about a static wearer starving the
parallax anchors, and from H18.6 finding LiteOffice's curves are scatter. Both
are indirect.

Two parts.

  a) Measure the motion. Camera centre per frame from GT pose (C = -R^T t),
     reported as spread about the centroid, path length and bbox diagonal, for
     all six fitting sequences on one scale.

  b) The within-device control, which is what breaks H19's confound. From the
     SAME Apartment cached frames pick two 30-frame subsets - `high` by
     farthest-point sampling, `low` as the 30 nearest neighbours of a random
     anchor frame - and fit the same curve on each. Device, lens, room, scene,
     frame count and teacher are all held fixed; the only variable is how far
     the camera travelled.

If `low` ~= `high`, staticness is not the mechanism, the difference in H19 was
the device after all, and what I published in both reports is wrong. That
outcome is the reason to run this.

Limitation recorded up front: the cache holds every ~10th source frame, so `low`
is "30 frames from a short stretch of the walk", not 30 consecutive video
frames. The contrast is therefore weaker than a real static capture - passing is
conservative, failing is not conclusive alone.
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


def motion_stats(C):
    """C: (n,3) camera centres in world metres."""
    if len(C) < 2:
        return dict(spread_m=0.0, path_m=0.0, bbox_diag_m=0.0, n=len(C))
    c = C.mean(0)
    return dict(
        spread_m=float(np.linalg.norm(C - c, axis=1).mean()),
        path_m=float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum()),
        bbox_diag_m=float(np.linalg.norm(C.max(0) - C.min(0))),
        n=int(len(C)))


def pick_high(C, k, rng):
    """Farthest-point sampling from a random start: maximum camera spread."""
    idx = [int(rng.integers(len(C)))]
    d = np.linalg.norm(C - C[idx[0]], axis=1)
    while len(idx) < k:
        j = int(np.argmax(d)); idx.append(j)
        d = np.minimum(d, np.linalg.norm(C - C[j], axis=1))
    return np.array(idx)


def pick_low(C, k, rng):
    """The k frames nearest a random anchor: minimum camera spread."""
    a = int(rng.integers(len(C)))
    return np.argsort(np.linalg.norm(C - C[a], axis=1))[:k]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--arm", default="omega110")
    p.add_argument("--fit-seqs", default=",".join(APT + LITE))
    p.add_argument("--eval-seqs", default=",".join(
        [f"{A}/Apartment_release_clean_seq136_M1292",
         f"{A}/Apartment_release_decoration_seq132_M1292"] + LITE))
    p.add_argument("--subset", type=int, default=30)
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
        return (torch.cos(th),
                np.clip(np.digitize(th.numpy(), e) - 1, 0, THETA_BINS - 1),
                0.5 * (e[:-1] + e[1:]) * 180 / np.pi,
                (th <= cam.theta_max).numpy())

    def predict(s, n, cos_t):
        with torch.no_grad():
            return U.forward_range(bb, s.src.image(n).to(a.device),
                                   cos_t.to(a.device)).float().cpu().numpy()

    # ---- pass 1: fitting samples + camera centre, per frame ----------------
    root = Path(a.cache_root) / a.arm
    pools, motion = {}, {}
    for sd in a.fit_seqs.split(","):
        s = Seq(sd, a.size, a.max_frames)
        cos_t, t_idx, _, _ = bins_of(s.src.camera)
        d = root / s.name
        man = json.loads((d / "manifest.json").read_text())
        cov = np.load(d / "covered.npy")
        off = man.get("log_offset_vs_raw", {})
        install(s.src.camera)
        got, cen = [], []
        for n in s.frames:
            po = s.src.pose(n)
            if po is None:
                continue
            R, t = po
            C = (-(R.transpose(-1, -2) @ t)).numpy().astype(np.float64)
            stem = s.stem(n)
            tgt = np.load(d / "npz" / f"{stem}.npz")["depth"].astype(np.float32)
            tgt = tgt * float(np.exp(-off.get(stem, 0.0)))
            pr = predict(s, n, cos_t)
            m = np.flatnonzero((cov & (tgt > 1e-6) & (pr > 1e-6)).ravel())
            if m.size < 200:
                continue
            if m.size > a.px_per_frame:
                m = rng.choice(m, a.px_per_frame, replace=False)
            got.append((np.log(pr.ravel()[m]), np.log(tgt.ravel()[m]), t_idx.ravel()[m]))
            cen.append(C)
        pools[s.name] = dict(frames=got, C=np.stack(cen) if cen else np.zeros((0, 3)))
        motion[s.name] = motion_stats(pools[s.name]["C"])
        st = motion[s.name]
        print(f"[h20] {s.name}: {len(got)} fr  spread {st['spread_m']:.3f} m  "
              f"path {st['path_m']:.2f} m  bbox {st['bbox_diag_m']:.2f} m", flush=True)

    # ---- pass 2: evaluation frames ----------------------------------------
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
        print(f"[h20] eval {s.name}: {len(P)} frames", flush=True)
    del bb
    torch.cuda.empty_cache()

    # ---- numpy from here ---------------------------------------------------
    NB = len(EDG) - 1

    def fit(frs):
        LP = np.concatenate([f[0] for f in frs])
        LT = np.concatenate([f[1] for f in frs])
        BN = np.concatenate([f[2] for f in frs])
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

    # pool the four Apartment sequences into one frame list with one C array
    apt_names = [Path(x).name for x in APT]
    frames = [f for n in apt_names for f in pools[n]["frames"]]
    C_all = np.concatenate([pools[n]["C"] for n in apt_names])
    ev_names = [Path(x).name for x in a.eval_seqs.split(",")]

    res = {"motion": motion, "eval_order": ev_names, "config": vars(a), "arms": {}}
    print(f"\n[h20] Apartment pool: {len(frames)} frames, "
          f"spread {motion_stats(C_all)['spread_m']:.3f} m")
    print(f"\n{'arm':>6s} {'|da|':>7s} {'spread_m':>9s} " +
          " ".join(f"{n[:20]:>21s}" for n in ev_names))

    for arm, pick in (("high", pick_high), ("low", pick_low)):
        Cs, gains, spreads = [], [], []
        for dr in range(a.draws):
            r = np.random.default_rng(a.seed + 100 * dr + (0 if arm == "high" else 1))
            idx = pick(C_all, a.subset, r)
            Cf = fit([frames[i] for i in idx])
            Cs.append(Cf[:, 0])
            spreads.append(motion_stats(C_all[idx])["spread_m"])
            gains.append([rim_gain(Cf, n) for n in ev_names])
        D = np.stack(Cs); G = np.array(gains)
        da = float(np.mean([np.abs(D[i] - D[j]).mean()
                            for i in range(len(D)) for j in range(i + 1, len(D))]))
        res["arms"][arm] = {"mean_pairwise_da": da,
                            "spread_m_mean": float(np.mean(spreads)),
                            "gain_mean": G.mean(0).tolist(),
                            "gain_sd": G.std(0).tolist(),
                            "coef_a_mean": D.mean(0).tolist()}
        print(f"{arm:>6s} {da:>7.4f} {np.mean(spreads):>9.3f} " +
              " ".join(f"{m:>13.1f}+-{s:4.1f}" for m, s in zip(G.mean(0), G.std(0))))

    print("\n[h20] H19 reference: LiteOffice 30-frame |da| was 0.100 (Bowl) / 0.127 (Dino)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(f"[h20] wrote {a.out}")


if __name__ == "__main__":
    main()
