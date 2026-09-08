"""Two controls for the roll curve: is it the roll, or the black region?

`fine_rotation.py` rolls the stored fisheye frame about the principal point
with zero padding. That changes what is black: the sensor's dark corners leave
the frame and hard-black wedges enter it, and the frame border clips the imaged
disc at different azimuths. A model that reacts to the padding would trace the
same falling curve as a model that reacts to being rolled. Two arms separate
them; both score the SAME pixels at every angle.

  fisheye_disc  The upright frame is masked to the largest disc about the
                principal point that fits inside the frame, at every angle
                including 0. A disc rolled about its centre is itself, so the
                black region is identical at every angle: only the content
                rolls.
  pinhole       The frame is re-imaged as an 89 deg co-axial pinhole view
                (H14's rig) whose virtual camera is rolled about its optical
                axis. Nothing is padded: at 89 deg the corner ray is 54.3 deg,
                inside Aria's 54.83 deg cone at every roll, so every pixel of
                every input is real content. The prediction is mapped back to
                the fisheye grid and scored on theta <= 44 deg, the disc every
                rolled square contains.

`fisheye_frame` is `fine_rotation.py`'s arm re-run here for a like-for-like
comparison, and `resample0` runs 0 deg through the interpolator so the cost of
one bilinear pass is on the table too.

Rotation centre: the image the model sees is the UPRIGHT one, so the centre is
the principal point AFTER the quarter turn, (H-1-cy, cx) -- not (cx, cy).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
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

import importlib.util as _ilu  # noqa: E402
import upright as U  # noqa: E402
import rect_teacher as RT  # noqa: E402


def _load(name, path):
    sp = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(sp); sp.loader.exec_module(m); return m


_H5 = _HERE.parents[1] / "h5-rim-finetune" / "code"
Seq = _load("h5_train", _H5 / "train.py").Seq
_ev = _load("h5_eval", _H5 / "eval_lora.py")
THETA_BINS, EDG = _ev.THETA_BINS, _ev.GT_DEPTH_EDGES
from finetune.eval.metrics import align_depth  # noqa: E402


def roll_grid(size: int, cx: float, cy: float, deg: float) -> torch.Tensor:
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    ys, xs = torch.meshgrid(torch.arange(size, dtype=torch.float32),
                            torch.arange(size, dtype=torch.float32), indexing="ij")
    x, y = xs - cx, ys - cy
    sx, sy = ca * x + sa * y + cx, -sa * x + ca * y + cy
    return torch.stack((2 * (sx + 0.5) / size - 1, 2 * (sy + 0.5) / size - 1), dim=-1)


def warp(t: torch.Tensor, g: torch.Tensor, mode="bilinear") -> torch.Tensor:
    x = t[None] if t.dim() == 3 else t[None, None]
    out = F.grid_sample(x, g[None].to(x.device, x.dtype), mode=mode,
                        padding_mode="zeros", align_corners=False)
    return out[0] if t.dim() == 3 else out[0, 0]


@dataclass
class RolledView(RT.ViewSpec):
    """A co-axial view rolled about its own optical axis."""
    roll_deg: float = 0.0

    def rotation(self) -> torch.Tensor:
        a = math.radians(self.roll_deg)
        c, s = math.cos(a), math.sin(a)
        return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def upright_principal_point(cam, size: int):
    """(cx, cy) of the stored frame -> (cx, cy) after U.to_model's quarter turn.

    Checked against the turn itself rather than trusted: a marker placed at the
    stored principal point must land where this says.
    """
    m = torch.zeros(size, size)
    ix, iy = int(round(float(cam.cx))), int(round(float(cam.cy)))
    m[iy, ix] = 1.0
    r = torch.nonzero(U.to_model(m))[0]
    uy, ux = int(r[0]), int(r[1])
    ex, ey = float(cam.cx) + (ux - ix), float(cam.cy) + (uy - iy)
    return ex, ey


def main(argv=None) -> None:
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

    # rotation centre in the upright frame, and the disc that fits around it
    ucx, ucy = upright_principal_point(cam, a.size)
    r_disc = min(ucx, ucy, W - 1 - ucx, H - 1 - ucy) - 1.0
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32), indexing="ij")
    disc_up = ((xs - ucx) ** 2 + (ys - ucy) ** 2 <= r_disc ** 2)
    disc_stored = U.from_model(disc_up.float()).numpy() > 0.5
    print(f"[roll] stored pp ({float(cam.cx):.1f},{float(cam.cy):.1f}) -> upright "
          f"({ucx:.1f},{ucy:.1f}); disc r={r_disc:.1f}px keeps "
          f"{(disc_stored & cone).sum() / cone.sum():.3f} of the cone; "
          f"common mask theta<={a.common_theta_deg:g} keeps {common.sum() / cone.sum():.3f}")

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device=a.device, variant=a.variant)

    def install(camera, hw):
        bb.install(None, camera, hw, patch_undistort=False, border_token=False,
                   dpt_grid=False, depth_convention="z")

    gts = {f: s.gt_range(f, cos_t).numpy() for f in s.frames}

    def score(preds, mask):
        """preds: {frame: range (H,W) stored frame}. Zones on `mask`, plus on `common`."""
        out = {}
        for tag, m_ in (("own", mask), ("common", common)):
            s_ = np.zeros((THETA_BINS, NB)); n_ = np.zeros((THETA_BINS, NB))
            for f, d in preds.items():
                gt = gts[f]
                v = m_ & (gt > 0) & (gt <= a.depth_max_m) & (d > 1e-6)
                if v.sum() < 1000:
                    continue
                al = align_depth(d, gt, v, mode="scale_shift")
                ar = (np.abs(al - gt) / np.clip(gt, 1e-6, None))[v]
                di = np.clip(np.digitize(gt[v], EDG) - 1, 0, NB - 1)
                flat = t_idx[v] * NB + di
                s_ += np.bincount(flat, weights=ar, minlength=THETA_BINS * NB).reshape(THETA_BINS, NB)
                n_ += np.bincount(flat, minlength=THETA_BINS * NB).reshape(THETA_BINS, NB)
            tab = np.divide(s_, n_, out=np.zeros_like(s_), where=n_ > 0)
            z_ = {}
            for nm, keep in (("near_rim", lambda i, j: t_mid[i] >= 38 and EDG[j + 1] <= 2.0),
                             ("center", lambda i, j: t_mid[i] <= 11),
                             ("all", lambda i, j: True)):
                cells = [(i, j) for i in range(THETA_BINS) for j in range(NB) if keep(i, j)]
                w = np.array([n_[i, j] for i, j in cells], float)
                z_[nm] = float((np.array([tab[i, j] for i, j in cells]) * w).sum() / max(w.sum(), 1.0))
            z_["rim_over_center"] = z_["near_rim"] / max(z_["center"], 1e-9)
            z_["n_px"] = float(n_.sum())
            out[tag] = z_
        return out

    angles = [float(x) for x in a.angles.split(",")]
    results = {"fisheye_frame": {}, "fisheye_disc": {}, "pinhole": {}}

    # ---- fisheye arms ------------------------------------------------------
    install(cam, (H, W))
    for arm in ("fisheye_frame", "fisheye_disc"):
        for deg in angles + ["resample0"]:
            d_ = 0.0 if deg == "resample0" else deg
            force = deg == "resample0"
            g_in = roll_grid(a.size, ucx, ucy, d_).to(a.device)
            g_out = roll_grid(a.size, ucx, ucy, -d_).to(a.device)
            preds = {}
            for f in s.frames:
                img = U.to_model(s.src.image(f).to(a.device))
                if arm == "fisheye_disc":
                    img = img * disc_up.to(a.device)
                if d_ != 0.0 or force:
                    img = warp(img, g_in)
                    if arm == "fisheye_disc":
                        img = img * disc_up.to(a.device)
                with torch.no_grad():
                    z = bb.forward(img[None, None]).depth[0]
                if d_ != 0.0 or force:
                    z = warp(z, g_out)
                z = U.from_model(z).cpu()
                preds[f] = (z / cos_t.clamp_min(1e-6)).numpy()
            mask = cone if arm == "fisheye_frame" else (cone & disc_stored)
            results[arm][str(deg)] = score(preds, mask)

    # ---- pinhole arm -------------------------------------------------------
    for deg in angles:
        rig = RT.Rig(cam, [RolledView(fov_x_deg=a.view_fov, width=a.view_size,
                                      height=a.view_size, roll_deg=deg)])
        v = rig.views[0]
        install(v.pin, (v.spec.height, v.spec.width))
        cov = rig.covered.numpy()
        preds = {}
        for f in s.frames:
            with torch.no_grad():
                d, _ = rig.teach(lambda warped, view: U.forward_z(bb, warped),
                                 s.src.image(f).to(a.device), align=False)
            preds[f] = np.where(cov, d.float().cpu().numpy(), 0.0)
        r = score(preds, cone & cov)
        r["fill"] = rig.fill_fraction
        r["cone_coverage"] = rig.coverage
        r["common_covered"] = float((cov & common).sum() / common.sum())
        results["pinhole"][str(deg)] = r

    # ---- print -------------------------------------------------------------
    for arm, rows in results.items():
        print(f"\n[{arm}]  (scored on the arm's own mask | on the common theta<={a.common_theta_deg:g} mask)")
        print(f"{'roll':>10s} {'all':>8s} {'near_rim':>9s} {'center':>8s} {'rim/ctr':>8s} | "
              f"{'all':>8s} {'rim38-44':>9s} {'center':>8s} {'rim/ctr':>8s}")
        for k, r in rows.items():
            o, c = r["own"], r["common"]
            extra = f"  fill={r['fill']:.3f}" if "fill" in r else ""
            print(f"{k:>10s} {o['all']:8.4f} {o['near_rim']:9.4f} {o['center']:8.4f} {o['rim_over_center']:8.2f} | "
                  f"{c['all']:8.4f} {c['near_rim']:9.4f} {c['center']:8.4f} {c['rim_over_center']:8.2f}{extra}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({
            "seq": s.name, "frames": len(s.frames), "upright_pp": [ucx, ucy],
            "disc_r_px": r_disc, "arms": results, "config": vars(a)}, indent=1))


if __name__ == "__main__":
    main()
