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
    p.add_argument("--teacher-fov", type=float, default=110.0)
    p.add_argument("--teacher-size", type=int, default=630,
                   help="630 keeps centre sampling at parity with the fisheye "
                        "at 504 (ratio 1.009); see rect_teacher.virtual_pinhole")
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--variant", default="small")
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--score-teacher", action="store_true",
                   help="also score the teacher against GT (diagnostic only)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)

    s = Seq(os.path.expanduser(a.seq), a.size, a.max_frames)
    cam = s.src.camera
    pin = RT.virtual_pinhole(cam, a.teacher_fov, a.teacher_size)
    RT.assert_shared_axis(cam, pin)
    g_in, _ = RT.grid_fisheye_to_pinhole(cam, pin)
    g_out, covered = RT.grid_pinhole_to_fisheye(cam, pin)
    cone = cam.valid_mask(a.size, a.size)
    cov = float((covered & cone).sum()) / float(cone.sum())
    print(f"[h14/{a.arm}] {s.name}: {len(s.frames)} frames, pinhole "
          f"{a.teacher_size}px @ {a.teacher_fov} deg, cone coverage {cov:.4f}")
    if cov < 0.999:
        raise SystemExit(
            f"[h14] the pinhole covers only {cov:.4f} of the imaged cone. A "
            f"teacher that cannot see the rim cannot teach it -- that is "
            f"Center-PH's measured failure (49.6% near-rim coverage), not a "
            f"design this experiment may repeat.")

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device=a.device,
                        variant=a.variant)
    # Installed for the camera the model is actually FED. `_finalize` converts
    # native depth to `range` against `self.camera`, so installing the fisheye
    # here and feeding a pinhole render would apply the wrong theta -- a smooth
    # radial error, i.e. exactly the thing being measured.
    if a.arm == "rect":
        bb.install(None, pin, (a.teacher_size, a.teacher_size),
                   patch_undistort=False, border_token=False, dpt_grid=False,
                   depth_convention="range")
    else:
        bb.install(None, cam, (a.size, a.size),
                   patch_undistort=False, border_token=False, dpt_grid=False,
                   depth_convention="range")

    out = Path(a.out)
    (out / "npz").mkdir(parents=True, exist_ok=True)
    g_in_d = g_in.to(a.device)
    g_out_d = g_out.to(a.device)

    # ---- pass 1: the teacher --------------------------------------------
    t0 = time.time()
    teacher = {}
    for n in s.frames:
        img = s.src.image(n).to(a.device)
        with torch.no_grad():
            if a.arm == "rect":
                d_pin = bb.forward(RT.warp(img, g_in_d)[None, None]).depth[0]
                d_fish = RT.warp(d_pin, g_out_d)
            else:
                d_raw = bb.forward(img[None, None]).depth[0]
                d_fish = RT.warp(RT.warp(d_raw, g_in_d), g_out_d)
        teacher[s.stem(n)] = d_fish.float().cpu().numpy()

    # ---- pass 2: the raw-fisheye reference -------------------------------
    # Always run, not only under --score-teacher, because the per-frame log
    # offset below is part of the TARGET and not a diagnostic. Re-installing
    # per frame instead of once per pass would re-hook the model 60 times.
    bb.install(None, cam, (a.size, a.size), patch_undistort=False,
               border_token=False, dpt_grid=False, depth_convention="range")
    cov_np = covered.numpy()
    cos_t = torch.cos(cam.incidence_grid(a.size, a.size))
    offsets = {}
    if a.score_teacher:
        theta_np = cam.incidence_grid(a.size, a.size).numpy()
        t_edges = np.linspace(0.0, float(cam.theta_max), THETA_BINS + 1)
        t_idx = np.clip(np.digitize(theta_np, t_edges) - 1, 0, THETA_BINS - 1)
        t_mid = 0.5 * (t_edges[:-1] + t_edges[1:]) * 180 / np.pi
        nb_d = len(GT_DEPTH_EDGES) - 1
        acc = {k: [np.zeros((THETA_BINS, nb_d)), np.zeros((THETA_BINS, nb_d))]
               for k in ("teacher", "raw")}
        from finetune.eval.metrics import align_depth

    for n in s.frames:
        stem = s.stem(n)
        with torch.no_grad():
            raw = bb.forward(s.src.image(n)[None, None].to(a.device)
                             ).depth[0].float().cpu().numpy()
        tea = teacher[stem]
        both = cov_np & (raw > 1e-6) & (tea > 1e-6)
        # Median, not mean: a handful of saturated pixels at the cone edge
        # would drag a mean and silently rescale every target in the frame.
        offsets[stem] = float(np.median(np.log(tea[both]) - np.log(raw[both]))
                              ) if both.sum() > 1000 else 0.0
        if a.score_teacher:
            gr = s.gt_range(n, cos_t).numpy()
            for key, d in (("teacher", tea), ("raw", raw)):
                valid = both & (gr > 0) & (gr <= a.depth_max_m)
                if valid.sum() < 1000:
                    continue
                al = align_depth(d, gr, valid, mode="scale_shift")
                ar = (np.abs(al - gr) / np.clip(gr, 1e-6, None))[valid]
                di = np.clip(np.digitize(gr[valid], GT_DEPTH_EDGES) - 1, 0, nb_d - 1)
                flat = t_idx[valid] * nb_d + di
                acc[key][0] += np.bincount(flat, weights=ar,
                                           minlength=THETA_BINS * nb_d
                                           ).reshape(THETA_BINS, nb_d)
                acc[key][1] += np.bincount(flat, minlength=THETA_BINS * nb_d
                                           ).reshape(THETA_BINS, nb_d)

    for stem, d in teacher.items():
        np.savez_compressed(out / "npz" / f"{stem}.npz",
                            depth=d.astype(np.float16))
    np.save(out / "covered.npy", cov_np)

    manifest = {
        "arm": a.arm, "seq": s.name, "seq_dir": a.seq, "frames": len(s.frames),
        "stems": [s.stem(n) for n in s.frames],
        "size": a.size, "teacher_fov_deg": a.teacher_fov,
        "teacher_size": a.teacher_size, "backbone": f"da3-{a.variant}",
        "depth_convention": "range", "cone_coverage": cov,
        "git": git_rev(), "used_gt_for_targets": False,
        "log_offset_vs_raw": offsets,
        "log_offset_median": float(np.median(list(offsets.values()))),
        "seconds": round(time.time() - t0, 1),
        "config": vars(a),
    }
    print(f"[h14/{a.arm}] median log-offset vs raw fisheye: "
          f"{manifest['log_offset_median']:+.4f} "
          f"(roundtrip should be ~0 by construction)")

    if a.score_teacher:
        # Both scored on the SAME pixels (`both` & the same validity rule), so
        # the comparison is not a coverage artefact -- the failure mode that
        # made Center-PH look reasonable until its 49.6% was measured.
        tz = zone_table(acc["teacher"][0], acc["teacher"][1], t_mid)
        rz = zone_table(acc["raw"][0], acc["raw"][1], t_mid)
        manifest["precheck"] = {"teacher": tz, "raw_fisheye": rz}
        print(f"[h14/{a.arm}] PRE-CHECK (teacher vs raw, same pixels):")
        for k in tz:
            print(f"    {k}: raw {rz[k]:.4f} -> teacher {tz[k]:.4f} "
                  f"({(tz[k] - rz[k]) / rz[k] * 100:+.2f}%)")
        key = "near_rim(<=2m,>=38deg)"
        if key in tz:
            manifest["precheck"]["near_rim_teacher_beats_raw"] = bool(tz[key] < rz[key])
            if a.arm == "rect" and tz[key] >= rz[key]:
                print("[h14] PREMISE NOT CONFIRMED on this backbone/config: the "
                      "rect teacher is not better at the near rim than the raw "
                      "model. 024A does not transfer here; training a student "
                      "on it cannot help. Record this and stop.")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[h14/{a.arm}] wrote {len(teacher)} maps -> {out} "
          f"({manifest['seconds']}s)")


if __name__ == "__main__":
    main()
