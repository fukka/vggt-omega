"""H17.5 — is the fisheye roll curve steep because of the boundary or the projection?

h16 measured three arms whose 30 deg rise in all-image AbsRel was +48%
(`pinhole`), +82% (`fisheye_frame`) and +146% (`fisheye_disc`). `pinhole`
differs from the others in two ways at once — no hard black boundary, and a
rectified projection — so nothing there attributes the gap.

The fourth arm people reach for (a virtual fisheye whose cone sits strictly
inside its frame) does not separate them either: it still shows the model a hard
black annulus. Aria's cone does not fit inside its square frame, so a roll
performed in a fisheye frame either clips content or introduces an annulus. The
confound cannot be removed from that side.

So put the boundary INTO the clean arm instead. Mask the 89 deg pinhole view to
its inscribed disc:

    half-edge ray of an 89 deg square view = 44.5 deg
    corner ray                             = 54.3 deg
    inscribed disc                        == the theta <= 44.5 deg cap

Everything is already scored on theta <= 44 deg, so the mask removes only
content *outside the scored region* — exactly the surgery `fisheye_disc`
performs when it drops the 1.7% of the cone outside its inscribed circle.
`pinhole_masked` is therefore `pinhole` plus one hard black boundary and nothing
else: same projection, same resampling, same scored pixels, same rolled-camera
construction.

Locked before running: >= +100% at 30 deg means the boundary explains the gap;
<= +70% means the projection does; in between means both.
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
    p.add_argument("--angles", default="-40,-30,-20,-10,0,10,20,30,40")
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--variant", default="small")
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

    vs = a.view_size
    c = (vs - 1) / 2.0
    yy, xx = torch.meshgrid(torch.arange(vs, dtype=torch.float32),
                            torch.arange(vs, dtype=torch.float32), indexing="ij")
    disc = (((xx - c) ** 2 + (yy - c) ** 2) <= (vs / 2.0 - 1.0) ** 2).to(a.device)
    print(f"[bnd] view {vs}x{vs} @ {a.view_fov} deg; inscribed disc keeps "
          f"{float(disc.float().mean()):.3f} of the frame "
          f"(the theta <= {a.view_fov / 2:.1f} deg cap)")

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
        for nm, keep in (("near_rim", lambda i, j: t_mid[i] >= 38 and EDG[j + 1] <= 2.0),
                         ("center", lambda i, j: t_mid[i] <= 11),
                         ("all", lambda i, j: True)):
            cells = [(i, j) for i in range(THETA_BINS) for j in range(NB) if keep(i, j)]
            w = np.array([n_[i, j] for i, j in cells], float)
            out[nm] = float((np.array([tab[i, j] for i, j in cells]) * w).sum() / max(w.sum(), 1.0))
        out["rim_over_center"] = out["near_rim"] / max(out["center"], 1e-9)
        return out

    results = {"pinhole": {}, "pinhole_masked": {}}
    for deg in angles:
        rig = RT.Rig(cam, [RC.RolledView(fov_x_deg=a.view_fov, width=vs, height=vs,
                                         roll_deg=deg)])
        v = rig.views[0]
        bb.install(None, v.pin, (vs, vs), patch_undistort=False, border_token=False,
                   dpt_grid=False, depth_convention="z")
        cov = rig.covered.numpy()
        for arm, masked in (("pinhole", False), ("pinhole_masked", True)):
            preds = {}
            for f in s.frames:
                def fz(warped, _view, _m=masked):
                    return U.forward_z(bb, warped * disc if _m else warped)
                with torch.no_grad():
                    d, _ = rig.teach(fz, s.src.image(f).to(a.device), align=False)
                preds[f] = np.where(cov, d.float().cpu().numpy(), 0.0)
            results[arm][str(deg)] = zones(preds, cone & cov & common)
        r0, r1 = results["pinhole"][str(deg)], results["pinhole_masked"][str(deg)]
        print(f"  {deg:>5.0f}deg   pinhole all {r0['all']:.4f} rim {r0['near_rim']:.4f}"
              f"   masked all {r1['all']:.4f} rim {r1['near_rim']:.4f}")

    print(f"\n{'arm':<16s}{'0deg all':>10s}" + "".join(f"{'+/-' + str(int(d)):>9s}"
          for d in (10, 20, 30, 40)))
    summary = {}
    for arm, A in results.items():
        base = A["0.0"]["all"]
        row, rises = [], {}
        for d in (10, 20, 30, 40):
            vals = [A[k]["all"] for k in (f"{d}.0", f"-{d}.0") if k in A]
            r = 100 * (float(np.mean(vals)) / base - 1)
            rises[d] = r
            row.append(f"{r:>8.0f}%")
        summary[arm] = {"base_all": base, "rise_pct": rises,
                        "base_rim": A["0.0"]["near_rim"]}
        print(f"{arm:<16s}{base:>10.4f}" + "".join(row))

    m0 = summary["pinhole"]["rise_pct"][30]
    m1 = summary["pinhole_masked"]["rise_pct"][30]
    c_all = 100 * (summary["pinhole_masked"]["base_all"] / summary["pinhole"]["base_all"] - 1)
    c_rim = 100 * (summary["pinhole_masked"]["base_rim"] / summary["pinhole"]["base_rim"] - 1)
    print(f"\n[bnd] 0 deg cost of the mask alone: all {c_all:+.1f}%, near_rim {c_rim:+.1f}% "
          f"(fisheye_disc paid +25% / +38%)")
    verdict = ("the BOUNDARY explains the gap" if m1 >= 100 else
               "the PROJECTION explains the gap" if m1 <= 70 else
               "BOTH contribute, neither dominates")
    print(f"[bnd] 30 deg rise: pinhole {m0:.0f}%, pinhole_masked {m1:.0f}%, "
          f"fisheye_disc +146% (h16) -> {verdict}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"seq": s.name, "frames": len(s.frames), "arms": results,
             "summary": summary, "mask_cost_0deg": {"all_pct": c_all, "rim_pct": c_rim},
             "verdict": verdict, "config": vars(a)}, indent=1))


if __name__ == "__main__":
    main()
