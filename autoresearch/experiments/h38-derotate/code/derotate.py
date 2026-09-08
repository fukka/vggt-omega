"""H38 - is de-rotating with a known roll a net win, once you pay for the border?

Three arms per angle, same frames and same scored mask:

    raw    the view rendered at roll a                (no roll removed, no border)
    derot  that view rotated by -a before inference,  (roll removed, border paid)
           the planar-z prediction rotated back by +a
    null   that view rotated by -a and then by +a     (no roll removed, border paid)

`null` is the untransformed-baseline bar. It pays exactly the same two
resamplings and loses exactly the same corner content as `derot` while removing
no roll at all, so anything `derot` gains over `null` is the roll and nothing
else. H34 was void for want of this kind of control.

Rotating the PLANAR Z back is exact - `upright.forward_z`'s docstring
establishes that planar z is invariant under a roll about the optical axis - so
the arms differ only in what the backbone was shown.

The scored region is the inscribed disc (theta <= common_theta_deg) of the
square view, and a rotation about the centre maps that disc to itself, so the
black corners lie entirely outside the scored pixels. Same geometry as H17.5,
H29 and H37: a hard border adjacent to, but not inside, what is measured.
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


_H5 = _HERE.parents[1] / "h5-rim-finetune" / "code"
Seq = _load("h5_train", _H5 / "train.py").Seq
_ev = _load("h5_eval", _H5 / "eval_lora.py")
THETA_BINS, EDG = _ev.THETA_BINS, _ev.GT_DEPTH_EDGES
from finetune.eval.metrics import align_depth  # noqa: E402


def rot(x: torch.Tensor, deg: float, mode: str = "bilinear") -> torch.Tensor:
    """Rotate [C,H,W] about the frame centre, zeros outside. `deg` is applied to
    the sampling grid, so the sign is resolved against geometry, not taste."""
    if deg == 0.0:
        return x
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    A = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], dtype=torch.float32,
                     device=x.device)[None]
    g = F.affine_grid(A, (1, x.shape[0], x.shape[1], x.shape[2]),
                      align_corners=False)
    return F.grid_sample(x[None], g, mode=mode, padding_mode="zeros",
                         align_corners=False)[0]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", required=True)
    p.add_argument("--angles", default="0,30,-30")
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--models", default="da3:small")
    p.add_argument("--omega-ckpt", default="checkpoints/VGGT-Omega-1B-512/model.pt")
    p.add_argument("--view-fov", type=float, default=89.0)
    p.add_argument("--view-size", type=int, default=630)
    p.add_argument("--common-theta-deg", type=float, default=44.0)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    s = Seq(a.seq, a.size, a.max_frames)
    cam = s.src.camera
    H = W = a.size
    theta = cam.incidence_grid(H, W)
    cos_t = torch.cos(theta)
    theta_np = theta.numpy()
    cone = (theta <= cam.theta_max).numpy()
    t_edges = np.linspace(0.0, float(cam.theta_max), THETA_BINS + 1)
    t_idx = np.clip(np.digitize(theta_np, t_edges) - 1, 0, THETA_BINS - 1)
    t_mid = 0.5 * (t_edges[:-1] + t_edges[1:]) * 180 / np.pi
    NB = len(EDG) - 1
    common = cone & (np.rad2deg(theta_np) <= a.common_theta_deg)
    gts = {f: s.gt_range(f, cos_t).numpy() for f in s.frames}
    angles = [float(x) for x in a.angles.split(",")]

    from raytun3r.backbones import build_backbone

    def zones(preds, mask):
        s_ = np.zeros((THETA_BINS, NB)); n_ = np.zeros((THETA_BINS, NB))
        for f, d in preds.items():
            gt = gts[f]
            v = mask & (gt > 0) & (gt <= a.depth_max_m) & (d > 1e-6)
            if v.sum() < 1000:
                continue
            al = align_depth(d, gt, v, mode="scale_shift")
            ar = (np.abs(al - gt) / np.clip(gt, 1e-6, None))[v]
            di = np.clip(np.digitize(gt[v], EDG) - 1, 0, NB - 1)
            flat = t_idx[v] * NB + di
            s_ += np.bincount(flat, weights=ar, minlength=THETA_BINS * NB).reshape(THETA_BINS, NB)
            n_ += np.bincount(flat, minlength=THETA_BINS * NB).reshape(THETA_BINS, NB)
        tab = np.divide(s_, n_, out=np.zeros_like(s_), where=n_ > 0)
        out = {}
        for nm, keep in (("near_rim", lambda i, j: t_mid[i] >= 38 and EDG[j + 1] <= 2.0),
                         ("center", lambda i, j: t_mid[i] <= 11),
                         ("all", lambda i, j: True)):
            cells = [(i, j) for i in range(THETA_BINS) for j in range(NB) if keep(i, j)]
            w = np.array([n_[i, j] for i, j in cells], float)
            out[nm] = float((np.array([tab[i, j] for i, j in cells]) * w).sum() / max(w.sum(), 1.0))
        out["rim_over_center"] = out["near_rim"] / max(out["center"], 1e-9)
        return out

    all_models, sign_report = {}, {}
    for spec in [x.strip() for x in a.models.split(",") if x.strip()]:
        name, _, variant = spec.partition(":")
        kw = {"variant": variant} if variant else {}
        w8 = a.omega_ckpt if name == "vggt_omega" else "pretrained"
        try:
            bb = build_backbone(name, weights=w8, device=a.device, **kw)
        except Exception as exc:
            print(f"[h38] {spec:14s} unavailable: {exc.__class__.__name__}: {exc}")
            continue
        ps = bb.patch_size
        vs = int(round(a.view_size / ps)) * ps
        c = (vs - 1) / 2.0
        yy, xx = torch.meshgrid(torch.arange(vs, dtype=torch.float32),
                                torch.arange(vs, dtype=torch.float32), indexing="ij")
        disc = ((((xx - c) ** 2 + (yy - c) ** 2) <= (vs / 2.0 - 1.0) ** 2)
                .to(a.device))
        print(f"[h38] {spec}: patch {ps}, view {vs}x{vs} @ {a.view_fov} deg", flush=True)

        # ---- resolve the rotation sign by geometry, before any model runs ----
        # Render the same frame at roll 0 and at roll +A; rotating the rolled
        # render by the correct sign must reproduce the level one on the disc.
        SIGN_A = 30.0
        f0 = s.frames[0]
        src0 = s.src.image(f0).to(a.device)
        def render(deg):
            rig = RT.Rig(cam, [RC.RolledView(fov_x_deg=a.view_fov, width=vs,
                                             height=vs, roll_deg=deg)], patch=ps)
            v = rig.views[0]
            out = {}
            def grab(warped, _view):
                out["img"] = warped.detach().clone()
                return torch.ones(warped.shape[-2:], device=warped.device)
            bb.install(None, v.pin, (vs, vs), patch_undistort=False,
                       border_token=False, dpt_grid=False, depth_convention="z")
            with torch.no_grad():
                rig.teach(grab, src0, align=False)
            return out["img"]
        lvl, rolled = render(0.0), render(SIGN_A)
        d3 = disc[None].expand_as(lvl)
        res = {}
        for sgn in (+1, -1):
            back = rot(rolled, sgn * SIGN_A)
            res[sgn] = float(((back - lvl).abs() * d3).sum() / d3.sum())
        SIGN = +1 if res[+1] <= res[-1] else -1
        sign_report[spec] = {"plus": res[+1], "minus": res[-1], "chosen": SIGN,
                             "probe_angle_deg": SIGN_A}
        print(f"[h38] {spec}: rotation sign probe  +1 -> L1 {res[+1]:.4f}   "
              f"-1 -> L1 {res[-1]:.4f}   chosen {SIGN:+d}", flush=True)

        results = {"raw": {}, "derot": {}, "null": {}}
        for deg in angles:
            rig = RT.Rig(cam, [RC.RolledView(fov_x_deg=a.view_fov, width=vs,
                                             height=vs, roll_deg=deg)], patch=ps)
            v = rig.views[0]
            bb.install(None, v.pin, (vs, vs), patch_undistort=False,
                       border_token=False, dpt_grid=False, depth_convention="z")
            cov = rig.covered.numpy()
            back = SIGN * deg          # takes the rolled view to level
            fwd = -SIGN * deg          # takes a level-frame quantity back
            for arm in ("raw", "derot", "null"):
                preds = {}
                for f in s.frames:
                    def fz(warped, _view, _arm=arm):
                        if _arm == "raw":
                            return U.forward_z(bb, warped)
                        if _arm == "derot":
                            z = U.forward_z(bb, rot(warped, back))
                            return rot(z[None], fwd)[0]
                        return U.forward_z(bb, rot(rot(warped, back), fwd))
                    with torch.no_grad():
                        d, _ = rig.teach(fz, s.src.image(f).to(a.device), align=False)
                    preds[f] = np.where(cov, d.float().cpu().numpy(), 0.0)
                results[arm][str(deg)] = zones(preds, cone & cov & common)
            r = {k: results[k][str(deg)]["all"] for k in results}
            print(f"  {deg:>6.1f}deg   raw {r['raw']:.4f}   derot {r['derot']:.4f}"
                  f"   null {r['null']:.4f}", flush=True)
        all_models[spec] = results
        del bb
        torch.cuda.empty_cache()

    print(f"\n{'backbone':<14s}{'angle':>8s}{'roll cost':>11s}{'derot':>10s}"
          f"{'null':>10s}{'derot vs raw0':>15s}")
    verdicts = {}
    for spec, R in all_models.items():
        base = R["raw"]["0.0"]["all"]
        vd = {"raw0_all": base, "angles": {}}
        for k in R["raw"]:
            if float(k) == 0.0:
                continue
            raw, der, nul = (R[x][k]["all"] for x in ("raw", "derot", "null"))
            vd["angles"][k] = {
                "roll_cost_pct": 100 * (raw / base - 1),
                "derot_vs_raw_pct": 100 * (der / raw - 1),
                "null_vs_raw_pct": 100 * (nul / raw - 1),
                "derot_vs_raw0_pct": 100 * (der / base - 1),
                "raw_all": raw, "derot_all": der, "null_all": nul}
            e = vd["angles"][k]
            print(f"{spec:<14s}{k:>8s}{e['roll_cost_pct']:>10.1f}%"
                  f"{e['derot_vs_raw_pct']:>9.1f}%{e['null_vs_raw_pct']:>9.1f}%"
                  f"{e['derot_vs_raw0_pct']:>14.1f}%")
        verdicts[spec] = vd

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"seq": s.name, "frames": len(s.frames), "models": all_models,
             "summary": verdicts, "sign_probe": sign_report,
             "config": vars(a)}, indent=1))
        print(f"\n[h38] wrote {a.out}")


if __name__ == "__main__":
    main()
