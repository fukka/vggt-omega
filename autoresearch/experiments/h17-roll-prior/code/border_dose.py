"""How much hard black border does it take to hurt? A dose-response curve.

Three measurements in this project have said "a hard black boundary is a large
insult", each at one dose and each in a different setting:

  * H14's 110 deg teacher: 22.5% of the frame black -> the teacher inverts in
    every zone (it had been better than the raw model at 98.7% fill);
  * h16's `fisheye_disc`: 1.7% of the CONE masked at 0 deg -> +25% all-image,
    +38% near_rim;
  * H17.5's `pinhole_masked`: the inscribed disc of a rectified view, i.e.
    21.9% of the frame black -> +106% all-image, +141% near_rim.

Those are three doses in three settings, so the standing rule ("never feed the
model a hard black border") rests on an unmeasured curve. This measures it: one
setting, one projection, one dose axis.

The dose is a black border of width w around the rectified 89 deg view, so the
blacked fraction is `1 - ((vs - 2w) / vs)^2`. A border is used rather than a
disc so that the removed content is spread evenly over the frame edge, and so
that small doses are reachable — the inscribed disc of a square cannot remove
less than 21.5%, which is why H17.5 could not be severity-matched to
`fisheye_disc`.

Run at 0 deg (the cost of the border alone) and at 30 deg (whether the border
and the roll interact, or merely add).

Scored on theta <= 44 deg, as everywhere else in h16/h17. Every dose keeps that
region intact: the widest border here (63 px of 630) removes content outside
theta 40.7 deg, and `--max-width` refuses anything that would eat into the
scored cap.
"""
from __future__ import annotations

import argparse
import json
import math
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", required=True)
    p.add_argument("--widths", default="0,5,12,25,45,70,100")
    p.add_argument("--angles", default="0,30")
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--variant", default="small")
    p.add_argument("--view-fov", type=float, default=89.0)
    p.add_argument("--view-size", type=int, default=630)
    p.add_argument("--common-theta-deg", type=float, default=30.0,
                   help="scored cap. 30 rather than h16's 44 because an 89 deg "
                        "view's half-edge ray is 44.5 deg: at a 44 deg cap only "
                        "5 px of frame are outside the scored region, so no "
                        "meaningful border fits. Zones are redefined against "
                        "the cap, and near_rim (theta >= 38) does not exist here.")
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    s = Seq(a.seq, a.size, a.max_frames)
    cam = s.src.camera
    theta = cam.incidence_grid(a.size, a.size)
    cos_t = torch.cos(theta)
    theta_np = theta.numpy()
    cone = (theta <= cam.theta_max).numpy()
    t_edges = np.linspace(0.0, float(cam.theta_max), THETA_BINS + 1)
    t_idx = np.clip(np.digitize(theta_np, t_edges) - 1, 0, THETA_BINS - 1)
    t_mid = 0.5 * (t_edges[:-1] + t_edges[1:]) * 180 / np.pi
    NB = len(EDG) - 1
    common = cone & (np.rad2deg(theta_np) <= a.common_theta_deg)
    gts = {f: s.gt_range(f, cos_t).numpy() for f in s.frames}

    vs = a.view_size
    f_px = (vs / 2.0) / math.tan(math.radians(a.view_fov) / 2.0)
    widths = [int(x) for x in a.widths.split(",")]
    for w in widths:
        # theta of the ray at the middle of the surviving frame edge
        th = math.degrees(math.atan((vs / 2.0 - w) / f_px))
        if th < a.common_theta_deg:
            raise SystemExit(
                f"[dose] border {w}px cuts to theta {th:.1f} deg, inside the "
                f"scored theta <= {a.common_theta_deg} cap; the dose would be "
                f"removing scored content, not framing it")
    print(f"[dose] view {vs}x{vs} @ {a.view_fov} deg, f={f_px:.1f}px")
    for w in widths:
        frac = 1.0 - ((vs - 2 * w) / vs) ** 2
        th = math.degrees(math.atan((vs / 2.0 - w) / f_px))
        print(f"       border {w:>3d}px -> {100 * frac:5.1f}% of the frame black, "
              f"edge ray {th:.1f} deg")

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device=a.device, variant=a.variant)

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
        # Zones against the cap, not h16's fixed ones: near_rim (theta >= 38)
        # is empty at a 30 deg cap. `outer` is the decile of the cone nearest
        # the border, and is what says whether the damage is local to the
        # border or global.
        lo = a.common_theta_deg - 10.0
        for nm, keep in (("outer", lambda i, j: t_mid[i] >= lo),
                         ("center", lambda i, j: t_mid[i] <= 11),
                         ("all", lambda i, j: True)):
            cells = [(i, j) for i in range(THETA_BINS) for j in range(NB) if keep(i, j)]
            wgt = np.array([n_[i, j] for i, j in cells], float)
            out[nm] = float((np.array([tab[i, j] for i, j in cells]) * wgt).sum()
                            / max(wgt.sum(), 1.0))
        return out

    res = {}
    for deg in [float(x) for x in a.angles.split(",")]:
        rig = RT.Rig(cam, [RC.RolledView(fov_x_deg=a.view_fov, width=vs, height=vs,
                                         roll_deg=deg)])
        v = rig.views[0]
        bb.install(None, v.pin, (vs, vs), patch_undistort=False, border_token=False,
                   dpt_grid=False, depth_convention="z")
        cov = rig.covered.numpy()
        res[str(deg)] = {}
        for w in widths:
            m = torch.zeros(vs, vs, device=a.device)
            m[w:vs - w, w:vs - w] = 1.0
            preds = {}
            for f in s.frames:
                with torch.no_grad():
                    d, _ = rig.teach(lambda warped, _v, _m=m: U.forward_z(bb, warped * _m),
                                     s.src.image(f).to(a.device), align=False)
                preds[f] = np.where(cov, d.float().cpu().numpy(), 0.0)
            z = zones(preds, cone & cov & common)
            z["black_frac"] = 1.0 - ((vs - 2 * w) / vs) ** 2
            res[str(deg)][str(w)] = z
            print(f"  {deg:>3.0f}deg  border {w:>3d}px ({100 * z['black_frac']:5.1f}% black)"
                  f"  all {z['all']:.4f}  outer {z['outer']:.4f}  ctr {z['center']:.4f}")

    print(f"\n{'border':>8s}{'% black':>9s}{'0deg all':>11s}{'30deg all':>11s}"
          f"{'0deg outer':>12s}{'0deg ctr':>11s}   cost at 0 deg (all / outer / ctr)")
    b0 = res["0.0"]["0"]
    for w in widths:
        z0 = res["0.0"][str(w)]
        z3 = res.get("30.0", {}).get(str(w))
        c = tuple(100 * (z0[k] / b0[k] - 1) for k in ("all", "outer", "center"))
        print(f"{w:>8d}{100 * z0['black_frac']:>8.1f}%{z0['all']:>11.4f}"
              f"{(z3['all'] if z3 else float('nan')):>11.4f}"
              f"{z0['outer']:>12.4f}{z0['center']:>11.4f}"
              f"   {c[0]:+7.1f}% {c[1]:+7.1f}% {c[2]:+7.1f}%")
    if "30.0" in res:
        print("\n[dose] does the border interact with roll, or just add?")
        for w in widths:
            r = 100 * (res["30.0"][str(w)]["all"] / res["0.0"][str(w)]["all"] - 1)
            print(f"        border {w:>3d}px: 30 deg costs {r:+6.1f}% on top of its own 0 deg")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"seq": s.name, "frames": len(s.frames), "doses": res,
             "config": vars(a)}, indent=1))


if __name__ == "__main__":
    main()
