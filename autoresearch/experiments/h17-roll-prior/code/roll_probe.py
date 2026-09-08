"""Is the camera's roll already legible in the frozen features?

H15 found that conditioning this backbone on geometry it can already derive
from the image is inert -- the real per-token Jacobian field lost to a
position-shuffled fake 10/16 and to no field at all 10/16, while a *wrong*
field hurt 16/16. If roll is likewise decodable from the frozen features, then
handing the network an IMU-measured roll is predicted inert for the same
reason, and the fix has to change the function (augment, or de-roll the input)
rather than add an input.

So: roll the frame by a known angle, read the frozen features, and fit a linear
map from features to (sin roll, cos roll). Ridge, fit on four training
sequences, reported on two held-out ones.

The whole experiment turns on ONE control. Rotating the *picture* injects hard
black wedges whose orientation IS the roll, so a probe on picture-rolled frames
measures nothing about the scene. The treatment arm therefore rolls the virtual
CAMERA and re-images (an 89 deg co-axial pinhole view, corner ray 54.3 deg,
inside Aria's 54.83 deg cone at every roll, frame fill 1.000 -- no black
anywhere). The picture-rolled arm is kept as the control, and the gap between
the two is how much of roll-legibility is the wedge.

Features are the last encoder block's tokens, pooled to a 3x3 spatial grid --
NOT globally mean-pooled, which would destroy the layout that orientation lives
in -- then PCA'd on the training split before the ridge. Chance level for a
uniform draw over [-45, 45] is 22.5 deg mean absolute error.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[3]))
sys.path.insert(0, str(_HERE.parents[1] / "h1-rim-pose-value" / "code"))
sys.path.append(str(_HERE.parents[1] / "h5-rim-finetune" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "common"))
sys.path.insert(0, str(_HERE.parents[1] / "h14-rect-distill" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "h16-orientation" / "code"))

import importlib.util as _ilu  # noqa: E402
import upright as U  # noqa: E402
import rect_teacher as RT  # noqa: E402
import roll_controls as RC  # noqa: E402


def _load(name, path):
    sp = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(sp); sp.loader.exec_module(m); return m


Seq = _load("h5_train", _HERE.parents[1] / "h5-rim-finetune" / "code" / "train.py").Seq


#: Where the image encoder's blocks live, per backbone family. Named, not
#: discovered: DA3 also carries `head.scratch.*Block` (the DPT fusion stack,
#: which is post-encoder) and `cam_enc.trunk.*` (a camera-pose encoder that does
#: not even run on a plain depth forward). Taking "the last module whose class
#: name ends in Block" picks `cam_enc.trunk.3`, whose hook never fires.
ENCODER_PREFIX = ("backbone.pretrained.blocks", "aggregator.frame_blocks",
                  "aggregator.global_blocks")


def last_block(model):
    """The last block of the image encoder, by declaration order."""
    cands = [(n, m) for n, m in model.named_modules()
             if n.startswith(ENCODER_PREFIX) and type(m).__name__.lower().endswith("block")]
    if not cands:
        have = sorted({".".join(n.split(".")[:-1]) for n, m in model.named_modules()
                       if type(m).__name__.lower().endswith("block")})
        raise SystemExit(f"[probe] no encoder block under {ENCODER_PREFIX}; block groups: {have}")
    return cands[-1]


def pool_grid(tok: torch.Tensor, gh: int, gw: int, k: int) -> np.ndarray:
    """(N, C) tokens on a gh x gw grid -> (k*k*C,) features."""
    x = tok.reshape(1, gh, gw, -1).permute(0, 3, 1, 2)
    return F.adaptive_avg_pool2d(x, (k, k)).reshape(-1).float().cpu().numpy()


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    A = "/user/f.zhang2/Documents/projectaria_tools_adt_data_clean"
    p.add_argument("--root", default=A)
    p.add_argument("--train-seqs", default="clean_seq131,clean_seq133,clean_seq134,clean_seq135")
    p.add_argument("--test-seqs", default="clean_seq136,decoration_seq132")
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--angles", default="-45,-35,-25,-15,-5,5,15,25,35,45")
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--view-fov", type=float, default=89.0)
    p.add_argument("--view-size", type=int, default=630)
    p.add_argument("--variant", default="small")
    p.add_argument("--n-pca", type=int, default=192)
    p.add_argument("--pool", type=int, default=3, help="spatial pooling grid side")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    p.add_argument("--cache", default=None, help="npz of extracted features; reused if present")
    a = p.parse_args(argv)

    angles = [float(x) for x in a.angles.split(",")]
    rng = np.random.default_rng(a.seed)

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device=a.device, variant=a.variant)
    name, mod = last_block(bb.model)
    print(f"[probe] hooking {name} ({type(mod).__name__})")
    grab = {}

    def hook(_m, _i, out):
        grab["t"] = out[0] if isinstance(out, (tuple, list)) else out
    mod.register_forward_hook(hook)

    def install(camera, hw):
        bb.install(None, camera, hw, patch_undistort=False, border_token=False,
                   dpt_grid=False, depth_convention="z")

    def feats_of(img_bchw, hw):
        with torch.no_grad():
            bb.forward(img_bchw[None, None])
        t = grab["t"]
        t = t[0] if t.dim() == 3 else t
        n_patch = (hw[0] // bb.patch_size) * (hw[1] // bb.patch_size)
        if t.shape[0] > n_patch:                    # drop cls/register prefix
            t = t[t.shape[0] - n_patch:]
        return pool_grid(t, hw[0] // bb.patch_size, hw[1] // bb.patch_size, a.pool)

    # One rig per angle, reused across every frame and sequence.
    seq_names = [s.strip() for s in (a.train_seqs + "," + a.test_seqs).split(",")]
    probe_rows = {"camera": {}, "picture": {}}
    for split, names in (("train", a.train_seqs.split(",")), ("test", a.test_seqs.split(","))):
        for arm in probe_rows:
            probe_rows[arm].setdefault(split, {"X": [], "y": [], "seq": []})

    cache = Path(a.cache) if a.cache else None
    if cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        for arm in probe_rows:
            for split in ("train", "test"):
                probe_rows[arm][split] = {"X": list(z[f"{arm}_{split}_X"]),
                                          "y": list(z[f"{arm}_{split}_y"]),
                                          "seq": list(z[f"{arm}_{split}_seq"])}
        print(f"[probe] reusing features from {cache}")
    else:
        for sn in seq_names:
            sn = sn.strip()
            split = "train" if sn in a.train_seqs else "test"
            s = Seq(f"{a.root}/Apartment_release_{sn}_M1292", a.size, a.frames)
            cam = s.src.camera
            ucx, ucy = RC.upright_principal_point(cam, a.size)
            rigs = {d: RT.Rig(cam, [RC.RolledView(fov_x_deg=a.view_fov, width=a.view_size,
                                                  height=a.view_size, roll_deg=d)])
                    for d in angles}
            for n in s.frames:
                img = s.src.image(n).to(a.device)
                deg = float(rng.choice(angles))
                # -- treatment: roll the camera, re-image. No black anywhere.
                v = rigs[deg].views[0]
                install(v.pin, (a.view_size, a.view_size))
                warped = RT.warp(img, v.grid_in.to(img.device))
                probe_rows["camera"][split]["X"].append(
                    feats_of(U.to_model(warped), (a.view_size, a.view_size)))
                probe_rows["camera"][split]["y"].append(deg)
                probe_rows["camera"][split]["seq"].append(sn)
                # -- control: roll the picture. Black wedges give the angle away.
                install(cam, (a.size, a.size))
                g = RC.roll_grid(a.size, ucx, ucy, deg).to(a.device)
                rolled = RC.warp(U.to_model(img), g)
                probe_rows["picture"][split]["X"].append(
                    feats_of(rolled, (a.size, a.size)))
                probe_rows["picture"][split]["y"].append(deg)
                probe_rows["picture"][split]["seq"].append(sn)
            print(f"[probe] {sn}: {len(s.frames)} frames done ({split})")
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **{
                f"{arm}_{sp}_{k}": np.array(probe_rows[arm][sp][k])
                for arm in probe_rows for sp in ("train", "test") for k in ("X", "y", "seq")})
            print(f"[probe] features cached to {cache}")

    def fit_eval(arm, permute=False):
        """`permute` shuffles the TRAINING labels only: the identical pipeline on
        a signal-free problem. That is the honest chance level for this readout.
        The naive "predict the median" baseline is NOT the right null here --
        regressing (sin, cos) and taking atan2 turns a degenerate fit into
        uniformly distributed angles, whose MAE is ~90 deg, not ~22.5."""
        tr, te = probe_rows[arm]["train"], probe_rows[arm]["test"]
        Xtr = np.stack(tr["X"]); Xte = np.stack(te["X"])
        ytr = np.array(tr["y"]); yte = np.array(te["y"])
        if permute:
            ytr = np.random.default_rng(a.seed + 1).permutation(ytr)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        # PCA on the training split only
        k = min(a.n_pca, Xtr.shape[0] - 1, Xtr.shape[1])
        _, _, Vt = np.linalg.svd(Xtr, full_matrices=False)
        P = Vt[:k].T
        Ztr, Zte = Xtr @ P, Xte @ P
        # Regress the ANGLE, in degrees. The first version regressed
        # (sin, cos) and took atan2; within +-45 deg cos only ranges over
        # [0.707, 1], so its variance is tiny, R2_cos came out at -72, and the
        # good sin fit (R2 0.61) was destroyed by the readout. There is no
        # wraparound in this range, so direct regression is both simpler and
        # better conditioned. (sin/cos is kept below as a secondary read.)
        Ttr = np.stack([ytr.astype(float), np.sin(np.radians(ytr))], 1)
        # inner split by sequence-free random holdout, only to pick alpha
        idx = np.arange(len(ytr)); rng.shuffle(idx)
        cut = int(0.8 * len(idx)); i_f, i_v = idx[:cut], idx[cut:]
        best = None
        for alpha in (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
            W = np.linalg.solve(Ztr[i_f].T @ Ztr[i_f] + alpha * np.eye(k),
                                Ztr[i_f].T @ Ttr[i_f])
            pv = (Ztr[i_v] @ W)[:, 0]
            e = np.abs(wrap180(pv - ytr[i_v])).mean()
            if best is None or e < best[1]:
                best = (alpha, e, None)
        alpha = best[0]
        W = np.linalg.solve(Ztr.T @ Ztr + alpha * np.eye(k), Ztr.T @ Ttr)
        Pte = Zte @ W
        pred = Pte[:, 0]
        err = np.abs(wrap180(pred - yte))
        per_seq = {sn: float(err[np.array(te["seq"]) == sn].mean())
                   for sn in set(te["seq"])}
        Tte = np.stack([yte.astype(float), np.sin(np.radians(yte))], 1)
        r2 = [float(1.0 - ((Pte[:, j] - Tte[:, j]) ** 2).sum()
                    / max(((Tte[:, j] - Tte[:, j].mean()) ** 2).sum(), 1e-9))
              for j in (0, 1)]
        return {"alpha": alpha, "n_train": int(len(ytr)), "n_test": int(len(yte)),
                "n_pca": int(k), "mae_deg": float(err.mean()),
                "r2_angle": r2[0], "r2_sin": r2[1],
                "median_ae_deg": float(np.median(err)), "per_seq_mae": per_seq,
                "chance_mae_deg": float(np.abs(yte - np.median(ytr)).mean()),
                # A linear probe cannot legitimately do WORSE than predicting the
                # training median; if it does, the readout is wrong, not the
                # features. (It did once: arctan2 was fed (cos, sin).)
                "worse_than_chance": bool(err.mean() > np.abs(yte - np.median(ytr)).mean())}

    res = {}
    for arm in ("camera", "picture"):
        res[arm] = fit_eval(arm)
        res[arm + "_null"] = fit_eval(arm, permute=True)
    print(f"\n{'arm':<16s}{'MAE':>8s}{'median':>8s}{'R2ang':>8s}{'R2sin':>8s}   per-sequence")
    for arm, r in res.items():
        print(f"{arm:<16s}{r['mae_deg']:>8.2f}{r['median_ae_deg']:>8.2f}"
              f"{r['r2_angle']:>8.3f}{r['r2_sin']:>8.3f}   "
              + ", ".join(f"{k}={v:.1f}" for k, v in sorted(r['per_seq_mae'].items())))
    for arm in ("camera", "picture"):
        d = res[arm + "_null"]["mae_deg"] - res[arm]["mae_deg"]
        print(f"[probe] {arm}: {d:+.1f} deg better than its own label-permuted null "
              f"({res[arm]['mae_deg']:.1f} vs {res[arm + '_null']['mae_deg']:.1f})")
    print("\n[probe] locked prediction: camera-rolled MAE < 10 deg => roll is legible "
          "from the scene => conditioning on measured roll is predicted inert.\n"
          "        camera-rolled MAE > 20 deg => the features do not encode gravity "
          "=> an IMU roll input carries information the model lacks.")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"arms": res, "config": vars(a)}, indent=1))


if __name__ == "__main__":
    main()
