"""H41 - the residual-roll curve, with an exact per-frame zero.

Each frame's real head roll psi is read from MPS (H17.1's method, copied from
roll_distribution: the device->camera rotation is TRANSPOSED and the quarter-turn
offset is SUBTRACTED). The view is then rendered at

    roll_deg = psi + d

so the residual roll the backbone sees is exactly -d, and d = 0 is that frame's
own gravity-aligned reference. Every render is normalised by that frame's d = 0
error, so the curve needs no binning and carries no drift between recordings.

Scored, like H39, on the per-frame intersection of every arm's coverage - the
arms differ in orientation, so they cover different parts of the cone.
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
sys.path.insert(0, str(_HERE.parents[1] / "h14-rect-distill" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "h16-orientation" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "h39-gravity-render" / "code"))

import importlib.util as _ilu  # noqa: E402
import upright as U  # noqa: E402
import rect_teacher as RT  # noqa: E402
import roll_controls as RC  # noqa: E402
from gravity_render import frame_rolls  # noqa: E402


def _load(name, path):
    sp = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(sp); sp.loader.exec_module(m); return m


_H5 = _HERE.parents[1] / "h5-rim-finetune" / "code"
Seq = _load("h5_train", _H5 / "train.py").Seq
from finetune.eval.metrics import align_depth  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--offset-deg", type=float, default=270.0)
    p.add_argument("--deltas", default="0,2,4,6,8,11,15,22",
                   help="residual roll is MINUS these; pass with = if negative")
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--models", default="da3:small")
    p.add_argument("--omega-ckpt",
                   default="checkpoints/VGGT-Omega-1B-512/model.pt",
                   help="vggt_omega does not load from 'pretrained'; it needs "
                        "this checkpoint. H42 hit that the first time this "
                        "script was run with a model list it had never seen.")
    p.add_argument("--view-fov", type=float, default=89.0)
    p.add_argument("--view-size", type=int, default=630)
    p.add_argument("--common-theta-deg", type=float, default=44.0)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--max-join-ms", type=float, default=33.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    deltas = [float(x) for x in a.deltas.split(",")]
    if 0.0 not in deltas:
        sys.exit("[h41] d = 0 must be in --deltas; it is the per-frame reference")

    s = Seq(a.seq, a.size, a.max_frames)
    cam = s.src.camera
    theta = cam.incidence_grid(a.size, a.size)
    cos_t = torch.cos(theta)
    cone = (theta <= cam.theta_max).numpy()
    common = cone & (np.rad2deg(theta.numpy()) <= a.common_theta_deg)
    gts = {f: s.gt_range(f, cos_t).numpy() for f in s.frames}

    psi_all, dt_all = frame_rolls(Path(a.seq), Path(a.calib), a.offset_deg,
                                  [s.stem(f) for f in s.frames])
    keep = dt_all <= a.max_join_ms
    s.frames = [f for f, k in zip(s.frames, keep) if k]
    psi = psi_all[keep]
    gts = {f: gts[f] for f in s.frames}
    if len(s.frames) < 10:
        sys.exit(f"[h41] only {len(s.frames)} frames survive the timestamp join")
    print(f"[h41] {s.name}: {len(s.frames)} frames, |roll| median "
          f"{np.median(np.abs(psi)):.2f} deg; deltas {deltas}", flush=True)

    from raytun3r.backbones import build_backbone

    out = {}
    for spec in [x.strip() for x in a.models.split(",") if x.strip()]:
        name, _, variant = spec.partition(":")
        kw = {"variant": variant} if variant else {}
        w8 = a.omega_ckpt if name == "vggt_omega" else "pretrained"
        bb = build_backbone(name, weights=w8, device=a.device, **kw)
        ps = bb.patch_size
        vs = int(round(a.view_size / ps)) * ps
        rig0 = RT.Rig(cam, [RC.RolledView(fov_x_deg=a.view_fov, width=vs,
                                          height=vs, roll_deg=0.0)], patch=ps)
        bb.install(None, rig0.views[0].pin, (vs, vs), patch_undistort=False,
                   border_token=False, dpt_grid=False, depth_convention="z")

        preds = {d: {} for d in deltas}
        masks = {}
        for j, f in enumerate(s.frames):
            src = s.src.image(f).to(a.device)
            cov_all = None
            for d in deltas:
                rig = RT.Rig(cam, [RC.RolledView(fov_x_deg=a.view_fov, width=vs,
                                                 height=vs,
                                                 roll_deg=float(psi[j] + d))],
                             patch=ps)
                with torch.no_grad():
                    z, _ = rig.teach(lambda w, _v: U.forward_z(bb, w), src,
                                     align=False)
                cv = rig.covered.numpy()
                preds[d][f] = np.where(cv, z.float().cpu().numpy(), 0.0)
                cov_all = cv if cov_all is None else (cov_all & cv)
            masks[f] = cone & cov_all & common

        # per-frame AbsRel at every delta, then normalise by that frame's d = 0
        per_frame = {str(d): [] for d in deltas}
        for f in s.frames:
            gt = gts[f]
            for d in deltas:
                pr = preds[d][f]
                v = masks[f] & (gt > 0) & (gt <= a.depth_max_m) & (pr > 1e-6)
                if v.sum() < 1000:
                    per_frame[str(d)].append(None)
                    continue
                al = align_depth(pr, gt, v, mode="scale_shift")
                per_frame[str(d)].append(
                    float(np.mean(np.abs(al - gt)[v] / np.clip(gt, 1e-6, None)[v])))
        ref = per_frame["0.0"] if "0.0" in per_frame else per_frame["0"]
        g = {}
        for d in deltas:
            k = str(d)
            r = [x / y for x, y in zip(per_frame[k], ref)
                 if x is not None and y not in (None, 0)]
            g[k] = (float(np.mean(r)), float(np.std(r)), len(r))
            print(f"  residual {-d:+6.1f} deg   g = {g[k][0]:.4f} "
                  f"± {g[k][1]:.4f}  (n={g[k][2]})", flush=True)
        out[spec] = {"g": g, "per_frame": per_frame, "deltas": deltas}
        del bb
        torch.cuda.empty_cache()

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"seq": s.name, "frames": len(s.frames),
             "roll_deg": [float(x) for x in psi], "models": out,
             "config": vars(a)}, indent=1))
        print(f"[h41] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
