"""How much does a real head-mounted camera actually roll?

The roll-sensitivity curve (h16) says the model degrades once the image is more
than ~20 deg off upright. That only matters if real egocentric footage GETS
that far off upright, and nothing in this project had measured it -- the "roll
is small in ADT" claim was an assumption.

Aria's MPS closed-loop trajectory carries both the device pose in a
gravity-aligned world frame and the gravity vector itself, per frame, at 30 Hz.
So the roll is not estimated here, it is READ:

    R_world_camera = R_world_device @ R_camera_device^T
    up_camera      = R_world_camera^T @ (-g/|g|)
    roll           = atan2(up_x, -up_y)          # 0 = world-up is image-up

measured in the frame the MODEL sees, i.e. after `upright.to_model`'s quarter
turn (UPRIGHT_K=3). The quarter turn is applied here as a constant offset on
the angle rather than to an image, and which offset is right is not asserted:
all four are computed and the one whose distribution is centred on zero is
reported, with the other three printed. That is an independent check on
UPRIGHT_K, from geometry instead of from a loss.

Also reported per sequence: the fraction of frames beyond each threshold in the
h16 curve, and the same for the ANGULAR RATE, because a per-frame gravity
correction has to track a moving head, not a static offset.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

THRESHOLDS = (10.0, 20.0, 30.0, 40.0)


def quat_to_R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def read_trajectory(path: Path):
    """(t_us, R_world_device, gravity_world) arrays."""
    ts, Rs, gs = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ts.append(float(row["tracking_timestamp_us"]))
            Rs.append(quat_to_R(float(row["qx_world_device"]), float(row["qy_world_device"]),
                                float(row["qz_world_device"]), float(row["qw_world_device"])))
            gs.append([float(row["gravity_x_world"]), float(row["gravity_y_world"]),
                       float(row["gravity_z_world"])])
    return np.array(ts), np.array(Rs), np.array(gs)


def rolls_deg(R_wd, g_w, R_cd):
    """Roll of the camera about its optical axis, degrees, in the STORED frame."""
    up_w = -g_w / np.linalg.norm(g_w, axis=-1, keepdims=True)
    R_wc = R_wd @ R_cd.T                       # world <- camera
    up_c = np.einsum("nji,nj->ni", R_wc, up_w)  # R_wc^T @ up_w
    return np.degrees(np.arctan2(up_c[:, 0], -up_c[:, 1]))


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def summarise(r, name, dt_s=None):
    a = np.abs(r)
    out = {"n": int(r.size), "median_abs": float(np.median(a)),
           "mean_abs": float(a.mean()), "p90_abs": float(np.percentile(a, 90)),
           "p99_abs": float(np.percentile(a, 99)), "max_abs": float(a.max()),
           "signed_median": float(np.median(r))}
    for t in THRESHOLDS:
        out[f"frac_gt_{t:g}"] = float((a > t).mean())
    if dt_s is not None and r.size > 1:
        rate = np.abs(wrap180(np.diff(r))) / dt_s
        out["rate_deg_per_s_median"] = float(np.median(rate))
        out["rate_deg_per_s_p99"] = float(np.percentile(rate, 99))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/user/f.zhang2/Documents/projectaria_tools_adt_data_clean")
    p.add_argument("--calib", required=True, help="the repo's T_device_camera json")
    p.add_argument("--glob", default="Apartment_release_*")
    p.add_argument("--extra", nargs="*", default=[],
                   help="extra sequence dirs (e.g. the LiteOffice extractions)")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    cal = json.loads(Path(a.calib).read_text())
    qx, qy, qz, qw = cal["T_device_camera"]["quaternion_xyzw"]
    R_cd = quat_to_R(qx, qy, qz, qw).T          # camera <- device

    seqs = sorted(Path(a.root).glob(a.glob)) + [Path(x) for x in a.extra]
    per_seq, offsets_check = {}, {k: [] for k in (0, 90, 180, 270)}
    for d in seqs:
        t = d / "groundtruth" / "aria_trajectory.csv"
        if not t.exists():
            continue
        ts, R_wd, g_w = read_trajectory(t)
        r_stored = rolls_deg(R_wd, g_w, R_cd)
        for k in offsets_check:
            offsets_check[k].append(np.abs(wrap180(r_stored - k)))
        dt = float(np.median(np.diff(ts))) * 1e-6
        per_seq[d.name] = {"stored": r_stored, "dt_s": dt}

    # Which quarter turn puts the wearer's head near upright? Geometry, not a loss.
    med = {k: float(np.median(np.concatenate(v))) for k, v in offsets_check.items()}
    best = min(med, key=med.get)
    print("[roll] median |roll| for each quarter-turn offset (deg): "
          + ", ".join(f"{k}deg->{v:.1f}" for k, v in sorted(med.items())))
    print(f"[roll] the model's frame is the {best} deg offset "
          f"(median |roll| {med[best]:.1f} deg); UPRIGHT_K=3 is a 90 deg clockwise turn")

    rows, allr = {}, []
    for name, d in per_seq.items():
        r = wrap180(d["stored"] - best)
        allr.append(r)
        rows[name] = summarise(r, name, d["dt_s"])
    pooled = summarise(np.concatenate(allr), "POOLED",
                       float(np.median([d["dt_s"] for d in per_seq.values()])))

    hdr = f"{'sequence':<44s}{'n':>7s}{'med':>7s}{'p90':>7s}{'p99':>7s}{'max':>7s}" \
          + "".join(f"{'>'+str(int(t)):>8s}" for t in THRESHOLDS)
    print("\n" + hdr)
    for name, s in sorted(rows.items()):
        print(f"{name:<44s}{s['n']:>7d}{s['median_abs']:>7.1f}{s['p90_abs']:>7.1f}"
              f"{s['p99_abs']:>7.1f}{s['max_abs']:>7.1f}"
              + "".join(f"{100*s[f'frac_gt_{t:g}']:>7.1f}%" for t in THRESHOLDS))
    print(f"{'POOLED':<44s}{pooled['n']:>7d}{pooled['median_abs']:>7.1f}"
          f"{pooled['p90_abs']:>7.1f}{pooled['p99_abs']:>7.1f}{pooled['max_abs']:>7.1f}"
          + "".join(f"{100*pooled[f'frac_gt_{t:g}']:>7.1f}%" for t in THRESHOLDS))
    print(f"\n[roll] roll rate: median {pooled['rate_deg_per_s_median']:.1f} deg/s, "
          f"p99 {pooled['rate_deg_per_s_p99']:.1f} deg/s")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        hist, edges = np.histogram(np.concatenate(allr), bins=np.arange(-90, 91, 2.5))
        Path(a.out).write_text(json.dumps(
            {"quarter_turn_offset_deg": best, "offset_medians": med,
             "per_sequence": rows, "pooled": pooled,
             "hist_counts": hist.tolist(), "hist_edges": edges.tolist(),
             "config": vars(a)}, indent=1))


if __name__ == "__main__":
    main()
