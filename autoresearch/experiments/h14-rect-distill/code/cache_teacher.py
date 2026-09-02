"""Build the H14 teacher cache: one dense pseudo-label map per frame, no GT.

Two arms, and the ONLY difference between them is the space the frozen backbone
was run in:

    --arm rect        warp the frame into a co-axial virtual pinhole, run the
                      model there, transport its depth back to the fisheye grid
    --arm roundtrip   run the model on the RAW fisheye, then send its depth
                      through the SAME pinhole and back

`roundtrip` is the control that decides the experiment. It carries the same
resampling budget, the same coverage mask and the same loss geometry as `rect`;
what it does not carry is the change of image formation. A student that
improves from it has gained from self-distillation, and we would have learned
nothing about the lens prior. The one asymmetry, stated rather than hidden:
`rect` resamples the image once and the depth once, `roundtrip` resamples the
depth twice and the image never.

There is a PRE-CHECK before either arm is worth training, and `--score-teacher`
runs it: score the teacher against GT on the standard (theta x depth) zones.
The premise of this whole experiment is ticket 024A -- the controlled rim/centre
ratio is 1.25-1.81x on raw fisheye and collapses to ~1.0 on rectified input --
but that was measured with a different harness, at ~85 deg, with the fovbench
model set. If the rect teacher is NOT better at the near rim than the raw model
on THIS backbone at THIS configuration, the premise does not transfer and no
amount of distillation can fix it. Stop there and spend nothing on training.

GT is read ONLY by `--score-teacher`, which produces a diagnostic and never
writes a target. The cache itself is label-free, which is the claim.

Usage (box):
    python .../h14-rect-distill/code/cache_teacher.py --arm rect \
        --seq $ADT/Apartment_release_clean_seq131_M1292 \
        --out results/h14-teacher/rect/seq131 --score-teacher
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[3]))
sys.path.insert(0, str(_HERE.parents[1] / "h1-rim-pose-value" / "code"))
sys.path.append(str(_HERE.parents[1] / "h5-rim-finetune" / "code"))
sys.path.insert(0, str(_HERE))

import importlib.util as _ilu  # noqa: E402
import rect_teacher as RT  # noqa: E402


def _load(name, path):
    sp = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


_H5 = _HERE.parents[1] / "h5-rim-finetune" / "code"
_h5_train = _load("h5_train", _H5 / "train.py")
_h5_eval = _load("h5_eval", _H5 / "eval_lora.py")
Seq = _h5_train.Seq
THETA_BINS = _h5_eval.THETA_BINS
GT_DEPTH_EDGES = _h5_eval.GT_DEPTH_EDGES


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_HERE,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def zone_table(absrel_sum, counts, t_mid) -> dict:
    """The zones of record -- same cells as #35/#36 and H12."""
    nb_d = len(GT_DEPTH_EDGES) - 1
    mean = np.divide(absrel_sum, counts, out=np.zeros_like(absrel_sum),
                     where=counts > 0)
    zones = {}
    for zname, keep in {
        "near_rim(<=2m,>=38deg)": lambda i, j: t_mid[i] >= 38 and GT_DEPTH_EDGES[j + 1] <= 2.0,
        "near_center(<=2m,<=11deg)": lambda i, j: t_mid[i] <= 11 and GT_DEPTH_EDGES[j + 1] <= 2.0,
        "center(<=11deg)": lambda i, j: t_mid[i] <= 11,
        "far(>=3m)": lambda i, j: GT_DEPTH_EDGES[j] >= 3.0,
    }.items():
        cells = [(i, j) for i in range(THETA_BINS) for j in range(nb_d) if keep(i, j)]
        w = np.array([counts[i, j] for i, j in cells], float)
        if w.sum() == 0:
            continue
        zones[zname] = float((np.array([mean[i, j] for i, j in cells]) * w).sum() / w.sum())
    return zones


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=("rect", "roundtrip"))
    p.add_argument("--seq", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=504,
                   help="the STUDENT's grid; the cache is written on it")
    p.add_argument("--teacher-fov", type=float, default=95.0,
                   help="95 is what the sweep picked: 98.7%% real content and "
                        "84%% cone coverage. 110 covers the cone but goes "
                        "22.5%% black and inverts the teacher in every zone.")
    p.add_argument("--layout", default="single", choices=("single", "ring"),
                   help="single = one co-axial square view (H14). ring = one "
                        "89 deg centre view plus six tangentially elongated "
                        "views tiling the annulus (H14.2): the only layout that "
                        "covers the cone AND keeps every frame filled.")
    p.add_argument("--ring-tilt", type=float, default=40.0)
    p.add_argument("--ring-fov-x", type=float, default=38.0)
    p.add_argument("--ring-width", type=int, default=280)
    p.add_argument("--ring-height", type=int, default=210)
    p.add_argument("--n-ring", type=int, default=8)
    p.add_argument("--teacher-size", type=int, default=630,
                   help="630 keeps centre sampling at parity with the fisheye "
                        "at 504 (ratio 1.009); see rect_teacher.virtual_pinhole")
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--variant", default="small")
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--score-teacher", action="store_true",
                   help="also score the teacher against GT (diagnostic only)")
    p.add_argument("--precheck-only", action="store_true",
                   help="run the pre-check and write no cache -- for sweeps")
    p.add_argument("--allow-partial-coverage", action="store_true",
                   help="permit a pinhole narrower than the cone. ONLY for the "
                        "FOV sweep: a teacher that cannot see the rim cannot "
                        "teach it, so a cache built this way is a diagnostic, "
                        "never a training target. Teacher and raw are still "
                        "scored on identical pixels within one FOV, but the "
                        "pixel SET differs between FOVs, so zone levels are "
                        "not comparable across the sweep -- only the "
                        "teacher-minus-raw delta within a row is.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)

    s = Seq(os.path.expanduser(a.seq), a.size, a.max_frames)
    cam = s.src.camera
    if a.layout == "ring":
        rig = RT.Rig.ring(cam, centre_fov_deg=89.0, centre_size=a.teacher_size,
                          n_ring=a.n_ring, tilt_deg=a.ring_tilt,
                          ring_fov_x_deg=a.ring_fov_x, ring_width=a.ring_width,
                          ring_height=a.ring_height)
    else:
        rig = RT.Rig.single(cam, fov_deg=a.teacher_fov, size=a.teacher_size)
    covered = rig.covered
    cone = cam.valid_mask(a.size, a.size)
    cov = rig.coverage
    theta_t = cam.incidence_grid(a.size, a.size)
    rim_cov = rig.zone_coverage(theta_t, 38.0, 54.9)
    print(f"[h14/{a.arm}] {s.name}: {len(s.frames)} frames, layout={a.layout} "
          f"({len(rig.views)} views, sizes {rig.sizes}), cone coverage "
          f"{cov:.4f}, rim-band coverage {rim_cov:.4f}, mean frame fill "
          f"{rig.fill_fraction:.4f}")
    if rig.fill_fraction < 0.95:
        print(f"[h14] WARNING: {(1 - rig.fill_fraction) * 100:.1f}% of the "
              f"teacher's frame is black. The sweep measured the teacher "
              f"INVERTING (near_rim -35% -> +15%) between 1.3% and 22.5% black.")
    if cov < 0.80 and not a.allow_partial_coverage:
        raise SystemExit(
            f"[h14] the rig covers only {cov:.4f} of the imaged cone. Below "
            f"0.80 this is Center-PH's measured failure (49.6% near-rim "
            f"coverage), not a teacher. Pass --allow-partial-coverage only "
            f"for a diagnostic sweep.")

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device=a.device,
                        variant=a.variant)

    theta = cam.incidence_grid(a.size, a.size)
    cos_fish = torch.cos(theta).clamp_min(1e-6)
    cos_dev = cos_fish.to(a.device)

    # Everything is carried in planar z inside the backbone (DA3's native
    # convention, so `_finalize` converts nothing) and turned into range
    # exactly once, against the axis the value is defined by: `Rig.teach`
    # divides by the VIEW's cos, the fisheye path by the fisheye's. Installing
    # a camera and letting _finalize divide would use the wrong axis for any
    # tilted view.
    def install(camera, hw):
        bb.install(None, camera, hw, patch_undistort=False, border_token=False,
                   dpt_grid=False, depth_convention="z")

    out = Path(a.out)
    (out / "npz").mkdir(parents=True, exist_ok=True)

    # ---- pass 1: the teacher --------------------------------------------
    t0 = time.time()
    teacher = {}
    scales = {}
    if a.arm == "rect":
        # The rig's frames differ in size, so the backbone is re-installed when
        # the size changes. `install` only re-registers hooks here (all the
        # geometry flags are off), so paying it per view is cheap; feeding a
        # 280x154 frame to a backbone installed for 630x630 would silently use
        # the wrong patch grid.
        installed = {"hw": None}
        def forward_z(warped, view):
            hw = (view.spec.height, view.spec.width)
            if installed["hw"] != hw:
                install(view.pin, hw)
                installed["hw"] = hw
            with torch.no_grad():
                return bb.forward(warped[None, None]).depth[0]
        for n in s.frames:
            d, info = rig.teach(forward_z, s.src.image(n).to(a.device))
            teacher[s.stem(n)] = d.float().cpu().numpy()
            scales[s.stem(n)] = info["log_scale"]
    else:
        install(cam, (a.size, a.size))
        for n in s.frames:
            with torch.no_grad():
                z = bb.forward(s.src.image(n)[None, None].to(a.device)).depth[0]
            d, info = rig.roundtrip(z / cos_dev)
            teacher[s.stem(n)] = d.float().cpu().numpy()
            scales[s.stem(n)] = info["log_scale"]

    # ---- pass 2: the raw-fisheye reference -------------------------------
    # Always run, not only under --score-teacher: the per-frame log offset
    # below is part of the TARGET, not a diagnostic.
    install(cam, (a.size, a.size))
    cov_np = covered.numpy()
    offsets = {}
    if a.score_teacher:
        theta_np = theta.numpy()
        t_edges = np.linspace(0.0, float(cam.theta_max), THETA_BINS + 1)
        t_idx = np.clip(np.digitize(theta_np, t_edges) - 1, 0, THETA_BINS - 1)
        t_mid = 0.5 * (t_edges[:-1] + t_edges[1:]) * 180 / np.pi
        nb_d = len(GT_DEPTH_EDGES) - 1
        acc = {k: [np.zeros((THETA_BINS, nb_d)), np.zeros((THETA_BINS, nb_d))]
               for k in ("teacher", "raw")}
        # How much of each zone the teacher can actually answer for. A zone
        # gain on 40% of its pixels is not the same claim as one on 95%, and
        # pooled numbers do not say which.
        zone_cov = {}
        from finetune.eval.metrics import align_depth

    for n in s.frames:
        stem = s.stem(n)
        with torch.no_grad():
            zr = bb.forward(s.src.image(n)[None, None].to(a.device)).depth[0]
        raw = (zr / cos_dev).float().cpu().numpy()
        tea = teacher[stem]
        both = cov_np & (raw > 1e-6) & (tea > 1e-6)
        # Median, not mean: a few saturated pixels at the cone edge would drag
        # a mean and silently rescale every target in the frame.
        offsets[stem] = (float(np.median(np.log(tea[both]) - np.log(raw[both])))
                         if both.sum() > 1000 else 0.0)
        if a.score_teacher:
            gr = s.gt_range(n, cos_fish).numpy()
            valid = both & (gr > 0) & (gr <= a.depth_max_m)
            if valid.sum() < 1000:
                continue
            for key, d in (("teacher", tea), ("raw", raw)):
                al = align_depth(d, gr, valid, mode="scale_shift")
                ar = (np.abs(al - gr) / np.clip(gr, 1e-6, None))[valid]
                di = np.clip(np.digitize(gr[valid], GT_DEPTH_EDGES) - 1, 0, nb_d - 1)
                flat = t_idx[valid] * nb_d + di
                acc[key][0] += np.bincount(flat, weights=ar,
                                           minlength=THETA_BINS * nb_d
                                           ).reshape(THETA_BINS, nb_d)
                acc[key][1] += np.bincount(flat, minlength=THETA_BINS * nb_d
                                           ).reshape(THETA_BINS, nb_d)
            in_zone = (gr > 0) & (gr <= a.depth_max_m) & cone.numpy()
            for zn, keep in (("near_rim(<=2m,>=38deg)",
                              (np.rad2deg(theta_np) >= 38) & (gr <= 2.0)),
                             ("center(<=11deg)", np.rad2deg(theta_np) <= 11)):
                m = in_zone & keep
                if m.sum():
                    p_, q_ = zone_cov.get(zn, (0.0, 0.0))
                    zone_cov[zn] = (p_ + float((m & cov_np).sum()), q_ + float(m.sum()))

    if not a.precheck_only:
        for stem, d in teacher.items():
            np.savez_compressed(out / "npz" / f"{stem}.npz",
                                depth=d.astype(np.float16))
        np.save(out / "covered.npy", cov_np)

    manifest = {
        "arm": a.arm, "seq": s.name, "seq_dir": a.seq, "frames": len(s.frames),
        "stems": [s.stem(n) for n in s.frames],
        "size": a.size, "teacher_fov_deg": a.teacher_fov,
        "teacher_size": a.teacher_size, "layout": a.layout,
        "n_views": len(rig.views), "view_sizes": rig.sizes,
        "backbone": f"da3-{a.variant}",
        "depth_convention": "range", "cone_coverage": cov,
        "rim_band_coverage": rim_cov,
        "mean_frame_fill": rig.fill_fraction, "precheck_only": a.precheck_only,
        "view_log_scales": scales, "git": git_rev(),
        "used_gt_for_targets": False,
        "log_offset_vs_raw": offsets,
        "log_offset_median": float(np.median(list(offsets.values()))),
        "seconds": round(time.time() - t0, 1), "config": vars(a),
    }
    print(f"[h14/{a.arm}] median log-offset vs raw fisheye: "
          f"{manifest['log_offset_median']:+.4f} "
          f"(roundtrip should be ~0 by construction)")

    if a.score_teacher:
        tz = zone_table(acc["teacher"][0], acc["teacher"][1], t_mid)
        rz = zone_table(acc["raw"][0], acc["raw"][1], t_mid)
        zc = {k: (v[0] / v[1] if v[1] else 0.0) for k, v in zone_cov.items()}
        manifest["precheck"] = {"teacher": tz, "raw_fisheye": rz,
                                "zone_coverage": zc}
        print(f"[h14/{a.arm}] PRE-CHECK (teacher vs raw, same pixels):")
        for k in tz:
            extra = f"  [teacher sees {zc[k] * 100:.1f}% of this zone]" if k in zc else ""
            print(f"    {k}: raw {rz[k]:.4f} -> teacher {tz[k]:.4f} "
                  f"({(tz[k] - rz[k]) / rz[k] * 100:+.2f}%){extra}")
        key = "near_rim(<=2m,>=38deg)"
        if key in tz:
            manifest["precheck"]["near_rim_teacher_beats_raw"] = bool(tz[key] < rz[key])
            if a.arm == "rect" and tz[key] >= rz[key]:
                print("[h14] PREMISE NOT CONFIRMED at this configuration: the "
                      "teacher is not better at the near rim than the raw "
                      "model. Do not train a student on this cache.")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[h14/{a.arm}] {'pre-check only, no cache written' if a.precheck_only else f'wrote {len(teacher)} maps'}"
          f" -> {out} ({manifest['seconds']}s)")


if __name__ == "__main__":
    main()
