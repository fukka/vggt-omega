"""H48 - ScanNet++ content through Aria's lens, then H47's pipeline unchanged.

Everything downstream of the resampling is the same as H47: a 60 deg co-axial
view (corner ray 42.4 deg, fully inside the cone, no black anywhere), three arms
(normal / mirror / twice), scored at theta <= 28 deg. Only whose photons are in
the frame changes.

The uncovered crescent ScanNet++ leaves in Aria's disc sits at 51.57-54.83 deg,
outside a 60 deg view entirely, so nothing in the scored region is void. It is
still masked, never filled.
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
sys.path.insert(0, str(_HERE.parents[1] / "common"))
sys.path.insert(0, str(_HERE.parents[1] / "h14-rect-distill" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "h16-orientation" / "code"))

import upright as U  # noqa: E402
import rect_teacher as RT  # noqa: E402
import roll_controls as RC  # noqa: E402
from raytun3r.cameras import from_aria  # noqa: E402
from raytun3r.data import ScanNetPPFisheye  # noqa: E402
from autoresearch.data.scannetpp_aria import (AriaRemap,  # noqa: E402
                                             depth_spread)
from finetune.eval.metrics import align_depth  # noqa: E402

ARMS = ("normal", "mirror", "twice")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--side", type=int, default=504, help="Aria frame side")
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--models", default="da3:small,da3:large,vggt,vggt_omega")
    p.add_argument("--omega-ckpt", default="checkpoints/VGGT-Omega-1B-512/model.pt")
    p.add_argument("--view-fov", type=float, default=60.0)
    p.add_argument("--view-size", type=int, default=630)
    p.add_argument("--common-theta-deg", type=float, default=28.0)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    side = a.side
    dst = from_aria(side, side)
    src = ScanNetPPFisheye(a.scene, max_size=side, patch=14,
                           max_frames=a.max_frames, depth_convention="z")
    remap = AriaRemap.build(src.camera, dst, (side, side))
    st = remap.stats()
    print(f"[h48] [{src.name}] disc {st['disc_frac_of_square']:.3f} of square, "
          f"covered {st['covered_frac_of_disc']:.4f} of disc, "
          f"void {st['void_frac_of_disc']:.4f}", flush=True)

    theta = dst.incidence_grid(side, side)
    cos_t = torch.cos(theta)
    cone = (theta <= dst.theta_max).numpy()
    common = cone & (np.rad2deg(theta.numpy()) <= a.common_theta_deg)

    # Resample once; the arms all see the same frames.
    # Same preservation check as the reciprocal arm (see its comment): the GT
    # depth spread over the scored cone must survive a pure change of lens.
    th_src = src.camera.incidence_grid(src.h, src.w)
    src_common = ((th_src <= src.camera.theta_max).numpy()
                  & (np.rad2deg(th_src.numpy()) <= a.common_theta_deg))
    spread_native, spread_warped = [], []

    frames = []
    for i in range(len(src)):
        d = src.depth(i)
        if d is None:
            continue
        dm, ok = (d if isinstance(d, tuple) else (d, None))
        dm = (dm.numpy() if torch.is_tensor(dm) else np.asarray(dm))
        img = src.image(i).permute(1, 2, 0).numpy()
        wi, vi = remap.image(img)
        wd, vd = remap.depth((dm * 1000.0).astype(np.float32))
        gz = wd / 1000.0
        valid = common & vi & vd & (gz > 0) & (gz <= a.depth_max_m)
        if valid.sum() < 1000:
            continue
        src_valid = src_common & (dm > 0) & (dm <= a.depth_max_m)
        spread_native.append(depth_spread(dm, src_valid))
        spread_warped.append(depth_spread(gz, valid))
        frames.append((torch.from_numpy(wi).permute(2, 0, 1).float(), gz, valid))
    print(f"[h48] [{src.name}] {len(frames)} usable frames", flush=True)
    sn = float(np.mean(spread_native)) if spread_native else 0.0
    sw = float(np.mean(spread_warped)) if spread_warped else 0.0
    rel = abs(sw - sn) / max(sn, 1e-6)
    verdict = ("OK" if rel <= 0.15 else
               "FAIL - the resample did not preserve the content; arm is void")
    print(f"[h48] [{src.name}] depth spread native {sn:.3f} -> warped {sw:.3f} "
          f"({rel:+.1%}) {verdict}", flush=True)
    if not frames:
        sys.exit(f"[h48] {src.name}: nothing usable")

    from raytun3r.backbones import build_backbone

    out = {}
    for spec in [x.strip() for x in a.models.split(",") if x.strip()]:
        name, _, variant = spec.partition(":")
        kw = {"variant": variant} if variant else {}
        w8 = a.omega_ckpt if name == "vggt_omega" else "pretrained"
        try:
            bb = build_backbone(name, weights=w8, device=a.device, **kw)
        except Exception as exc:
            print(f"[h48] {spec:14s} unavailable: {exc.__class__.__name__}: {exc}")
            continue
        ps = bb.patch_size
        vs = int(round(a.view_size / ps)) * ps
        rig = RT.Rig(dst, [RC.RolledView(fov_x_deg=a.view_fov, width=vs,
                                         height=vs, roll_deg=0.0)], patch=ps)
        bb.install(None, rig.views[0].pin, (vs, vs), patch_undistort=False,
                   border_token=False, dpt_grid=False, depth_convention="z")
        cov = rig.covered.numpy()

        per = {arm: [] for arm in ARMS}
        for img, gz, valid in frames:
            src_t = img.to(a.device)
            for arm in ARMS:
                def fz(w, _v=None, _a=arm):
                    if _a == "normal":
                        return U.forward_z(bb, w)
                    if _a == "twice":
                        x = torch.flip(torch.flip(w, dims=[-1]), dims=[-1])
                        return torch.flip(torch.flip(U.forward_z(bb, x), dims=[-1]),
                                          dims=[-1])
                    return torch.flip(U.forward_z(bb, torch.flip(w, dims=[-1])),
                                      dims=[-1])
                with torch.no_grad():
                    z, _ = rig.teach(fz, src_t, align=False)
                pr = np.where(cov, z.float().cpu().numpy(), 0.0)
                v = valid & cov & (pr > 1e-6)
                if v.sum() < 1000:
                    per[arm].append(None); continue
                al = align_depth(pr, gz, v, mode="scale_shift")
                per[arm].append(float(np.mean(
                    np.abs(al - gz)[v] / np.clip(gz, 1e-6, None)[v])))

        n = [x for x in per["normal"] if x is not None]
        mirror = [y / x for x, y in zip(per["normal"], per["mirror"])
                  if x not in (None, 0) and y is not None]
        tw = [abs(y / x - 1) for x, y in zip(per["normal"], per["twice"])
              if x not in (None, 0) and y is not None]
        if not n or not mirror:
            print(f"[h48] [{src.name}] {spec}: nothing scorable"); del bb; continue
        cost = 100 * (float(np.mean(mirror)) - 1)
        tw_max = 100 * max(tw)
        out[spec] = {"n_frames": len(n), "normal_absrel": float(np.mean(n)),
                     "mirror_cost_pct": cost, "plumbing_max_rel_pct": tw_max,
                     "per_frame": per}
        print(f"[h48] [{src.name}] {spec}: normal AbsRel {np.mean(n):.4f} | "
              f"mirror {cost:+.1f}% | plumbing {tw_max:.6f}% "
              f"{'OK' if tw_max < 0.01 else '<-- BROKEN'}", flush=True)
        del bb
        torch.cuda.empty_cache()

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"scene": Path(a.scene).name, "remap_stats": st, "models": out,
             "preservation": {"spread_native": sn, "spread_warped": sw,
                              "rel_diff": rel, "gate_pass": rel <= 0.15},
             "config": vars(a)}, indent=1))
        print(f"[h48] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
