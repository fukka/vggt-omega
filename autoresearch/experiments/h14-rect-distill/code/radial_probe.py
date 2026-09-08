"""H18.2 — how much of the rectified teacher is a purely radial relation?

H18's cross-room result inverted its in-room one: the un-rectified control
`omega_rt` carries most of the gain on a held-out sequence and **none** across
rooms, while every rectified arm transfers. Both targets are deterministic
functions of the image, so determinism is not the distinction. The recorded
hypothesis is that the rectified path teaches a *systematic radial relation* —
a property of the lens, not of the room — and that this is why it carries.

This replaces the 122.9k-parameter LoRA student with something that **can only**
express such a relation, fits it on the four training sequences and nothing
else, and applies it unchanged to the held-out sequences and to LiteOffice.

    radial   one log-log curve per theta bin (8 bins)        16 parameters
    global   one log-log curve for the whole image            2 parameters

`global` is the control that decides the reading: it has no radial structure at
all, so if it matches `radial` then the effect is a rescale and not a radial
relation. Same pair of arms as H9, different supervision — there the anchors
came from parallax, here the targets come from the rectified teacher.

The fit is `log(target) = a_bin * log(pred) + b_bin`, least squares, in the
direction the correction is applied (H9 measured that fitting the other way and
inverting is materially worse). Targets are the cached teacher depths with the
same per-frame log offset the trainer removes, so the student and this probe
see exactly the same supervision.

This is an instrument, not a method: 16 parameters will not beat a LoRA
in-room. What it recovers cross-room is a lower bound on how much of the
transferable signal is purely radial.
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    A = "/user/f.zhang2/Documents/projectaria_tools_adt_data_clean"
    L = "/user/f.zhang2/Documents/adt_liteoffice_extracted"
    p.add_argument("--cache-root", required=True, help="the omega110 cache root")
    p.add_argument("--arm", default="omega110")
    p.add_argument("--train-seqs", default=",".join(
        f"{A}/Apartment_release_clean_{s}_M1292" for s in ("seq131", "seq133", "seq134", "seq135")))
    p.add_argument("--eval-seqs", default=",".join(
        [f"{A}/Apartment_release_clean_seq136_M1292",
         f"{A}/Apartment_release_decoration_seq132_M1292",
         f"{L}/DinoToy_seq030", f"{L}/BlackCeramicBowl_seq030"]))
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--variant", default="small")
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device=a.device, variant=a.variant)

    def frozen_range(s, n, cos_t):
        with torch.no_grad():
            return U.forward_range(bb, s.src.image(n).to(a.device),
                                   cos_t.to(a.device)).float().cpu().numpy()

    # ---- fit on the training sequences only --------------------------------
    root = Path(a.cache_root) / a.arm
    logs_p, logs_t, bins = [], [], []
    for sd in a.train_seqs.split(","):
        s = Seq(sd, a.size, a.max_frames)
        cam = s.src.camera
        theta = cam.incidence_grid(a.size, a.size)
        cos_t = torch.cos(theta)
        t_edges = np.linspace(0.0, float(cam.theta_max), THETA_BINS + 1)
        t_idx = np.clip(np.digitize(theta.numpy(), t_edges) - 1, 0, THETA_BINS - 1)
        d = root / s.name
        man = json.loads((d / "manifest.json").read_text())
        cov = np.load(d / "covered.npy")
        off = man.get("log_offset_vs_raw", {})
        bb.install(None, cam, (a.size, a.size), patch_undistort=False,
                   border_token=False, dpt_grid=False, depth_convention="z")
        for n in s.frames:
            stem = s.stem(n)
            tgt = np.load(d / "npz" / f"{stem}.npz")["depth"].astype(np.float32)
            tgt = tgt * float(np.exp(-off.get(stem, 0.0)))   # what the trainer sees
            pr = frozen_range(s, n, cos_t)
            m = cov & (tgt > 1e-6) & (pr > 1e-6)
            logs_p.append(np.log(pr[m])); logs_t.append(np.log(tgt[m])); bins.append(t_idx[m])
        print(f"[radial] fitted on {s.name}: {len(s.frames)} frames")
    LP, LT, BN = np.concatenate(logs_p), np.concatenate(logs_t), np.concatenate(bins)

    def fit(x, y):
        A_ = np.stack([x, np.ones_like(x)], 1)
        return np.linalg.lstsq(A_, y, rcond=None)[0]

    coef_global = fit(LP, LT)
    coef_radial = np.zeros((THETA_BINS, 2))
    for b in range(THETA_BINS):
        m = BN == b
        coef_radial[b] = fit(LP[m], LT[m]) if m.sum() > 1000 else (1.0, 0.0)
    print(f"[radial] global  a={coef_global[0]:.4f} b={coef_global[1]:+.4f}")
    for b in range(THETA_BINS):
        print(f"[radial]  bin{b}  a={coef_radial[b,0]:.4f} b={coef_radial[b,1]:+.4f}")

    # ---- apply, unchanged, everywhere --------------------------------------
    out = {}
    for sd in a.eval_seqs.split(","):
        s = Seq(sd, a.size, a.max_frames)
        cam = s.src.camera
        theta = cam.incidence_grid(a.size, a.size)
        cos_t = torch.cos(theta)
        theta_np = theta.numpy()
        cone = (theta <= cam.theta_max).numpy()
        t_edges = np.linspace(0.0, float(cam.theta_max), THETA_BINS + 1)
        t_idx = np.clip(np.digitize(theta_np, t_edges) - 1, 0, THETA_BINS - 1)
        t_mid = 0.5 * (t_edges[:-1] + t_edges[1:]) * 180 / np.pi
        NB = len(EDG) - 1
        bb.install(None, cam, (a.size, a.size), patch_undistort=False,
                   border_token=False, dpt_grid=False, depth_convention="z")
        acc = {k: [np.zeros((THETA_BINS, NB)), np.zeros((THETA_BINS, NB))]
               for k in ("frozen", "global", "radial")}
        for n in s.frames:
            pr = frozen_range(s, n, cos_t)
            gt = s.gt_range(n, cos_t).numpy()
            lp = np.log(np.clip(pr, 1e-6, None))
            preds = {"frozen": pr,
                     "global": np.exp(coef_global[0] * lp + coef_global[1]),
                     "radial": np.exp(coef_radial[t_idx, 0] * lp + coef_radial[t_idx, 1])}
            for k, d_ in preds.items():
                v = cone & (gt > 0) & (gt <= a.depth_max_m) & (d_ > 1e-6)
                if v.sum() < 1000:
                    continue
                al = align_depth(d_, gt, v, mode="scale_shift")
                ar = (np.abs(al - gt) / np.clip(gt, 1e-6, None))[v]
                di = np.clip(np.digitize(gt[v], EDG) - 1, 0, NB - 1)
                flat = t_idx[v] * NB + di
                acc[k][0] += np.bincount(flat, weights=ar,
                                         minlength=THETA_BINS * NB).reshape(THETA_BINS, NB)
                acc[k][1] += np.bincount(flat, minlength=THETA_BINS * NB).reshape(THETA_BINS, NB)
        z = {}
        for k, (s_, n_) in acc.items():
            tab = np.divide(s_, n_, out=np.zeros_like(s_), where=n_ > 0)
            zz = {}
            for nm, keep in (("near_rim", lambda i, j: t_mid[i] >= 38 and EDG[j + 1] <= 2.0),
                             ("near_center", lambda i, j: t_mid[i] <= 11 and EDG[j + 1] <= 2.0),
                             ("all", lambda i, j: True)):
                cells = [(i, j) for i in range(THETA_BINS) for j in range(NB) if keep(i, j)]
                w = np.array([n_[i, j] for i, j in cells], float)
                zz[nm] = float((np.array([tab[i, j] for i, j in cells]) * w).sum()
                               / max(w.sum(), 1.0))
            z[k] = zz
        out[s.name] = z
        f = z["frozen"]["near_rim"]
        print(f"[radial] {s.name}: rim frozen {f:.4f} | "
              f"global {z['global']['near_rim']:.4f} "
              f"({100 * (z['global']['near_rim'] / f - 1):+.1f}%) | "
              f"radial {z['radial']['near_rim']:.4f} "
              f"({100 * (z['radial']['near_rim'] / f - 1):+.1f}%)")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"coef_global": coef_global.tolist(), "coef_radial": coef_radial.tolist(),
             "zones": out, "config": vars(a)}, indent=1))


if __name__ == "__main__":
    main()
