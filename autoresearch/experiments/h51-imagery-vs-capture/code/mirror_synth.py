"""H51 - the mirror test on real sensor frames and on ADT's Blender renders.

Same construction as H47 (60 deg co-axial view, scored at theta <= 28 deg, the
three arms normal/mirror/twice), run twice over THE SAME FRAMES: once on
`videos_rgb` and once on `videos_synthetic`. Scene, pose, lens and ground truth
are identical between the two runs; only the photometry differs. See
protocol.md for what that does and does not separate.

The frame list is built here from the three-way stem overlap of
videos_synthetic / videos_rgb / depth_npy, rather than through H5's `Seq`, for
two reasons: `Seq` pairs images to depth from `videos_rgb` alone and would
silently pick a different frame set per arm, and issue #35 (`cpu`) names H5's
training runs.

Conventions are COPIED VERBATIM from h44/mirror_curve.py -- the roll read
(device->camera rotation TRANSPOSED, quarter turn SUBTRACTED), gt_range's
division by cos theta, and the rig/backbone install. H39 lost a run by
re-deriving one of these instead of copying it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[3]))
sys.path.insert(0, str(_HERE.parents[1] / "h1-rim-pose-value" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "common"))
sys.path.insert(0, str(_HERE.parents[1] / "h14-rect-distill" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "h16-orientation" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "h39-gravity-render" / "code"))

import upright as U  # noqa: E402
import rect_teacher as RT  # noqa: E402
import roll_controls as RC  # noqa: E402
from gravity_render import frame_rolls  # noqa: E402
from adt_pose_value import AriaLocalPairs  # noqa: E402
from autoresearch.data.scannetpp_aria import to_grid  # noqa: E402
from finetune.eval.metrics import align_depth  # noqa: E402

ARMS = ("normal", "mirror", "twice")


def three_way_stems(seq_dir: str):
    """Stems present in videos_synthetic AND videos_rgb AND depth_npy.

    Sorted, so both arms of the same sequence walk the same frames in the same
    order without having to pass a list between them.
    """
    def stems(sub, ext):
        return {os.path.splitext(os.path.basename(p))[0]
                for p in glob.glob(os.path.join(seq_dir, sub, ext))}
    syn = stems("videos_synthetic", "*.png") | stems("videos_synthetic", "*.jpg")
    rgb = stems("videos_rgb", "*.png") | stems("videos_rgb", "*.jpg")
    dep = stems("depth_npy", "*.npy")
    return sorted(syn & rgb & dep)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--rgb-subdir", default="videos_rgb",
                   choices=("videos_rgb", "videos_synthetic"))
    p.add_argument("--offset-deg", type=float, default=270.0)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--models", default="da3:small,da3:large,vggt,vggt_omega")
    p.add_argument("--omega-ckpt", default="checkpoints/VGGT-Omega-1B-512/model.pt")
    p.add_argument("--view-fov", type=float, default=60.0)
    p.add_argument("--view-size", type=int, default=630)
    p.add_argument("--common-theta-deg", type=float, default=28.0)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--max-join-ms", type=float, default=33.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    name = os.path.basename(a.seq.rstrip("/"))
    stems = three_way_stems(a.seq)
    if len(stems) < a.max_frames:
        sys.exit(f"[h51] {name}: only {len(stems)} paired frames")
    # Evenly spaced across the recording, deterministic, identical per arm.
    idx = np.linspace(0, len(stems) - 1, a.max_frames).round().astype(int)
    stems = [stems[i] for i in dict.fromkeys(idx)]

    src = AriaLocalPairs(a.seq, size=a.size, rgb_subdir=a.rgb_subdir)
    pos = {os.path.splitext(os.path.basename(q))[0]: i
           for i, q in enumerate(src.paths)}
    frames = [pos[s] for s in stems if s in pos]
    if len(frames) != len(stems):
        sys.exit(f"[h51] {name}: {len(stems) - len(frames)} stems missing from "
                 f"{a.rgb_subdir} -- the arms would not be paired")

    cam = src.camera
    theta = cam.incidence_grid(a.size, a.size)
    cos_t = torch.cos(theta)
    cone = (theta <= cam.theta_max).numpy()
    common = cone & (np.rad2deg(theta.numpy()) <= a.common_theta_deg)

    # gt_range, copied: planar-z millimetres on the sensor grid -> range metres
    # on the camera's grid. NEAREST, never interpolation, across a depth edge.
    gts = {}
    for s, f in zip(stems, frames):
        z = np.load(os.path.join(a.seq, "depth_npy", s + ".npy")).astype(np.float32)
        z = to_grid(z, (a.size, a.size)) / 1000.0
        gts[f] = (torch.from_numpy(z) / cos_t.clamp_min(1e-6)).numpy()

    psi_all, dt_all = frame_rolls(Path(a.seq), Path(a.calib), a.offset_deg, stems)
    keep = dt_all <= a.max_join_ms
    frames = [f for f, k in zip(frames, keep) if k]
    stems = [s for s, k in zip(stems, keep) if k]
    psi = psi_all[keep]
    if len(frames) < 10:
        sys.exit(f"[h51] {name}: only {len(frames)} frames survive the join")
    print(f"[h51] {name} [{a.rgb_subdir}]: {len(frames)} paired frames, "
          f"|roll| median {np.median(np.abs(psi)):.2f} deg", flush=True)

    from raytun3r.backbones import build_backbone

    out = {}
    for spec in [x.strip() for x in a.models.split(",") if x.strip()]:
        mname, _, variant = spec.partition(":")
        kw = {"variant": variant} if variant else {}
        w8 = a.omega_ckpt if mname == "vggt_omega" else "pretrained"
        try:
            bb = build_backbone(mname, weights=w8, device=a.device, **kw)
        except Exception as exc:
            print(f"[h51] {spec:14s} unavailable: {exc.__class__.__name__}: {exc}")
            continue
        ps = bb.patch_size
        vs = int(round(a.view_size / ps)) * ps
        rig0 = RT.Rig(cam, [RC.RolledView(fov_x_deg=a.view_fov, width=vs,
                                          height=vs, roll_deg=0.0)], patch=ps)
        bb.install(None, rig0.views[0].pin, (vs, vs), patch_undistort=False,
                   border_token=False, dpt_grid=False, depth_convention="z")

        per = {arm: [] for arm in ARMS}
        for j, f in enumerate(frames):
            img = src.image(f).to(a.device)
            rig = RT.Rig(cam, [RC.RolledView(fov_x_deg=a.view_fov, width=vs,
                                             height=vs, roll_deg=float(psi[j]))],
                         patch=ps)
            gt = gts[f]
            for arm in ARMS:
                def fz(w, _v=None, _a=arm):
                    if _a == "normal":
                        return U.forward_z(bb, w)
                    if _a == "twice":
                        x = torch.flip(torch.flip(w, dims=[-1]), dims=[-1])
                        z = U.forward_z(bb, x)
                        return torch.flip(torch.flip(z, dims=[-1]), dims=[-1])
                    z = U.forward_z(bb, torch.flip(w, dims=[-1]))
                    return torch.flip(z, dims=[-1])
                with torch.no_grad():
                    z, _ = rig.teach(fz, img, align=False)
                cv = rig.covered.numpy()
                pr = np.where(cv, z.float().cpu().numpy(), 0.0)
                v = (common & cv & (gt > 0) & (gt <= a.depth_max_m) & (pr > 1e-6))
                if v.sum() < 1000:
                    per[arm].append(None); continue
                al = align_depth(pr, gt, v, mode="scale_shift")
                per[arm].append(float(np.mean(
                    np.abs(al - gt)[v] / np.clip(gt, 1e-6, None)[v])))

        pairs = [(x, y) for x, y in zip(per["normal"], per["mirror"])
                 if x not in (None, 0) and y is not None]
        if not pairs:
            print(f"[h51] {spec}: no scorable frames"); del bb; continue
        nmean = float(np.mean([x for x, _ in pairs]))
        mmean = float(np.mean([y for _, y in pairs]))
        cost = 100.0 * (mmean / nmean - 1.0)      # ratio of means, H37's rule
        tw = [abs(y / x - 1) for x, y in zip(per["normal"], per["twice"])
              if x not in (None, 0) and y is not None]
        tw_max = 100 * max(tw) if tw else float("nan")
        out[spec] = {"n_frames": len(pairs), "normal_absrel": nmean,
                     "mirror_cost_pct_ratio_of_means": cost,
                     "plumbing_max_rel_pct": tw_max, "per_frame": per}
        print(f"[h51] [{name}/{a.rgb_subdir}] {spec}: normal AbsRel {nmean:.4f} | "
              f"mirror {cost:+.1f}% | plumbing {tw_max:.6f}% "
              f"{'OK' if tw_max < 0.01 else '<-- BROKEN'}", flush=True)
        del bb
        torch.cuda.empty_cache()

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"seq": name, "arm": a.rgb_subdir, "stems": stems,
             "models": out, "config": vars(a)}, indent=1))
        print(f"[h51] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
