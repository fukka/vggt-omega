"""H48 reciprocal arm — Aria content through ScanNet++'s lens.

The other direction of the 2x2. Everything downstream of the resampling is the
same 60 deg co-axial view and the same three arms; only whose optics the content
is pushed through changes.

    Aria content, Aria lens        = H47   (measured)
    ScanNet++ content, Aria lens   = H48   (mirror_sca2aria.py)
    Aria content, ScanNet++ lens   = HERE
    ScanNet++ content, SN++ lens   = H46   (measured)

Planar z is invariant under this resample -- same optical centre, same optical
axis, no rotation -- which is why AriaRemap.depth may move the VALUES unchanged.
Under any rotation that stops being true; see scannetpp_aria.py's header.

Geometry checked with the corrected corner-ray formula (h47/CORRECTION.md): the
60 deg view's corner ray is 39.2 deg and Aria's content exists to 54.83 deg, so
the region with no Aria source never enters the scored view.
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
sys.path.append(str(_HERE.parents[1] / "h5-rim-finetune" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "h14-rect-distill" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "h16-orientation" / "code"))

import importlib.util as _ilu  # noqa: E402
import upright as U  # noqa: E402
import rect_teacher as RT  # noqa: E402
import roll_controls as RC  # noqa: E402
from raytun3r.data import ScanNetPPFisheye  # noqa: E402
from autoresearch.data.scannetpp_aria import AriaRemap, to_grid  # noqa: E402
from finetune.eval.metrics import align_depth  # noqa: E402


def _load(name, path):
    sp = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(sp); sp.loader.exec_module(m); return m


Seq = _load("h5_train", _HERE.parents[1] / "h5-rim-finetune" / "code" / "train.py").Seq

ARMS = ("normal", "mirror", "twice")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", required=True, help="an ADT sequence directory")
    p.add_argument("--lens-scene", required=True,
                   help="a ScanNet++ scene, used ONLY for its camera")
    p.add_argument("--size", type=int, default=504)
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

    src = Seq(a.seq, a.size, a.max_frames)
    aria = src.src.camera
    lens = ScanNetPPFisheye(a.lens_scene, max_size=a.size, patch=14,
                            max_frames=1, depth_convention="z")
    dst = lens.camera
    H, W = lens.h, lens.w
    remap = AriaRemap.build(aria, dst, (H, W))
    st = remap.stats()
    print(f"[h48r] {src.name} -> {lens.name}'s lens: {W}x{H}, "
          f"covered {st['covered_frac_of_disc']:.4f} of disc, "
          f"void {st['void_frac_of_disc']:.4f}", flush=True)

    theta = dst.incidence_grid(H, W)
    cone = (theta <= dst.theta_max).numpy()
    common = cone & (np.rad2deg(theta.numpy()) <= a.common_theta_deg)

    frames = []
    for f in src.frames:
        # depth_npy is planar z in millimetres; Seq.gt_range divides by cos to
        # get range, which is exactly what must NOT happen before a pure lens
        # re-parameterisation.
        # depth_npy is on the sensor's 1408 grid; the Aria camera of record is
        # built at --size, and AriaRemap's maps index THAT grid. The first run
        # of this arm skipped this line and warped the sensor's top-left corner
        # as if it were the whole frame -- see analysis.md, arm VOID.
        gz_mm = to_grid(np.load(src.dp[src.stem(f)]).astype(np.float32),
                        (int(aria.height), int(aria.width)))
        img = src.src.image(f).permute(1, 2, 0).numpy()
        wi, vi = remap.image(img)
        wd, vd = remap.depth(gz_mm)
        gz = wd / 1000.0
        valid = common & vi & vd & (gz > 0) & (gz <= a.depth_max_m)
        if valid.sum() < 1000:
            continue
        frames.append((torch.from_numpy(wi).permute(2, 0, 1).float(), gz, valid))
    print(f"[h48r] {len(frames)} usable frames", flush=True)
    if not frames:
        sys.exit("[h48r] nothing usable")

    from raytun3r.backbones import build_backbone

    out = {}
    for spec in [x.strip() for x in a.models.split(",") if x.strip()]:
        name, _, variant = spec.partition(":")
        kw = {"variant": variant} if variant else {}
        w8 = a.omega_ckpt if name == "vggt_omega" else "pretrained"
        try:
            bb = build_backbone(name, weights=w8, device=a.device, **kw)
        except Exception as exc:
            print(f"[h48r] {spec:14s} unavailable: {exc.__class__.__name__}: {exc}")
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
            print(f"[h48r] {spec}: nothing scorable"); del bb; continue
        # ratio of means is the citable statistic (H33, H37); the mean of
        # per-frame ratios is kept beside it so the two cannot drift apart.
        rom = 100 * (float(np.mean([y for x, y in zip(per["normal"], per["mirror"])
                                    if x not in (None, 0) and y is not None]))
                     / float(np.mean(n)) - 1)
        out[spec] = {"n_frames": len(n), "normal_absrel": float(np.mean(n)),
                     "mirror_cost_pct_ratio_of_means": rom,
                     "mirror_cost_pct_mean_of_ratios": 100 * (float(np.mean(mirror)) - 1),
                     "plumbing_max_rel_pct": 100 * max(tw), "per_frame": per}
        o = out[spec]
        print(f"[h48r] [{src.name}] {spec}: normal {o['normal_absrel']:.4f} | "
              f"mirror {rom:+.1f}% (ratio of means) | plumbing "
              f"{o['plumbing_max_rel_pct']:.6f}% "
              f"{'OK' if o['plumbing_max_rel_pct'] < 0.01 else '<-- BROKEN'}",
              flush=True)
        del bb
        torch.cuda.empty_cache()

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"seq": src.name, "lens_scene": lens.name, "remap_stats": st,
             "models": out, "config": vars(a)}, indent=1))
        print(f"[h48r] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
