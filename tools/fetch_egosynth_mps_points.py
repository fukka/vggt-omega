# Copyright (c) 2026.
"""Fetch the MPS semi-dense **world** points ego-synth re-projected into ``d``.

Why this exists
---------------
Every ``slambench`` number is scored against ``pts.d`` as if it were planar z
about the camera axis. Nothing had ever measured that — it is the data card's
word (and the npz's own ``meta``), and this repo has already been burned once by
exactly this assumption. See ticket 016.

Deciding it needs the thing ego-synth re-projected: MPS's semi-dense point
cloud, in the recording's world frame. Both readings — planar z and euclidean
range — are then computable for every point, and they differ by ``sec(theta)``:
1.00 on axis, 1.36 at 43 deg, 1.74 at 55 deg.

What it fetches, and what it deliberately does not
--------------------------------------------------
Only ``semidense_points.csv.gz``, by the same ranged-zip read
``fetch_egosynth_calibration`` uses (:func:`fetch_zip_member` is imported from
it rather than re-implemented):

    aea       mps_slam_points.zip   ->  semidense_points.csv.gz        44.8 MB
                                        (of a 349.1 MB archive)
    nymeria   recording_head.zip    ->  recording_head/mps/slam/
                                          semidense_points.csv.gz     161.4 MB
                                        (of a 593.0 MB archive)

**The trajectory is not fetched, and does not need to be.** Ticket 016 specifies
``closed_loop_trajectory.csv`` plus ``T_device_camera``, composed here. That
composition is already done, by the producer, and shipped in the release:
``camera_poses.json`` carries ``T_world_camera`` per frame for the whole take,
and states its own provenance —

    "extrinsics_source": "MPS closed_loop_trajectory (T_world_device)
                          @ T_device_camera(camera-rgb)"
    "stored_pose":       "T_world_camera (camera->world), 4x4 row-major"
    "timing":            "each frame's pose looked up by its exact VRS
                          capture_timestamp_ns (interpolated in the ~1kHz
                          trajectory)"

Re-deriving it here would fetch 101 MB per Nymeria take to reproduce a
composition the producer already did, and would introduce two ways to get it
wrong (the extrinsic, and the timestamp interpolation) in the middle of a check
whose whole point is to *remove* an unverified assumption. So the poses are read
from the release and only the points are fetched.

The world frame is per-take ("this recording's MPS gravity-aligned metric
frame"), which is fine: the check never leaves one take.

Licence
-------
The output is derived from AEA and Nymeria, both licensed per recipient. It is
written outside the repository by default and **must not be committed** —
``vggt-omega`` is public and its history is permanent and mirrored.

Usage
-----
    python tools/fetch_egosynth_mps_points.py aea \\
        --urls ~/Desktop/ADT/AriaEverydayActivities_download_urls.json \\
        --takes loc1_script1_seq1_rec1 \\
        --out ~/Desktop/ADT/ego-synth-5b-mps

``--takes`` is a comma-separated list of take names, or a directory of take
directories (e.g. the staged sample's ``<ds>/``). The URL JSONs are signed and
expiring; they are credentials, they are not in this repo, and they must not be
committed to it.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch_egosynth_calibration import fetch_zip_member  # noqa: E402

#: The member holding the world points, per dataset. Read off the real archives'
#: central directories, not from a data-format page.
MEMBER = {
    "aea": "semidense_points.csv.gz",
    "nymeria": "recording_head/mps/slam/semidense_points.csv.gz",
}

#: The one download group per dataset that contains it.
GROUP = {"aea": "mps_slam_points", "nymeria": "recording_head"}

#: The columns MPS writes. ``uid`` identifies the point across the recording;
#: the two sigmas are the same quantities ego-synth carries per observation.
COLUMNS = ("uid", "graph_uid", "px_world", "py_world", "pz_world",
           "inv_dist_std", "dist_std")


def wanted_takes(spec: Optional[str]) -> Optional[List[str]]:
    """Take names from a comma-separated list or a directory of take dirs."""
    if not spec:
        return None
    if os.path.isdir(spec):
        return sorted(d for d in os.listdir(spec)
                      if os.path.isdir(os.path.join(spec, d)))
    return [s.strip() for s in spec.split(",") if s.strip()]


def summarise(blob: bytes) -> dict:
    """Row count and the world-frame extent, without keeping the decompressed csv.

    Cheap, and it is the difference between "the fetch returned bytes" and "the
    fetch returned a point cloud" — a truncated ranged read decompresses to a
    valid prefix and would otherwise pass silently.
    """
    import numpy as np
    head, n, lo, hi = None, 0, None, None
    with gzip.open(io.BytesIO(blob), "rt") as fh:
        head = fh.readline().strip().split(",")
        rows = []
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 5:
                continue
            rows.append(parts[2:5])
            n += 1
        if rows:
            a = np.asarray(rows, float)
            lo, hi = a.min(axis=0).tolist(), a.max(axis=0).tolist()
    return {"columns": head, "n_points": n, "world_min": lo, "world_max": hi}


def run(ds: str, urls_path: str, out_root: str, takes: Optional[str],
        force: bool = False) -> int:
    with open(urls_path) as fh:
        doc = json.load(fh)
    seqs: Dict[str, dict] = doc["sequences"]
    names = wanted_takes(takes) or sorted(seqs)
    member, group = MEMBER[ds], GROUP[ds]
    out_ds = os.path.join(os.path.expanduser(out_root), ds)
    os.makedirs(out_ds, exist_ok=True)
    print(f"[mps] {ds}: {len(names)} take(s) -> {out_ds}")

    done = fail = 0
    for i, name in enumerate(names, 1):
        if name not in seqs:
            print(f"  [{i}/{len(names)}] {name}: not in the URL JSON — re-export "
                  f"it with that sequence selected")
            fail += 1
            continue
        entry = seqs[name].get(group)
        if entry is None:
            print(f"  [{i}/{len(names)}] {name}: no {group!r} group in the JSON")
            fail += 1
            continue
        dst_dir = os.path.join(out_ds, name)
        dst = os.path.join(dst_dir, "semidense_points.csv.gz")
        if os.path.exists(dst) and not force:
            print(f"  [{i}/{len(names)}] {name}: exists, skipping")
            done += 1
            continue
        try:
            blob = fetch_zip_member(entry["download_url"],
                                    int(entry["file_size_bytes"]), member)
            info = summarise(blob)
        except Exception as e:                # noqa: BLE001 — report, continue
            print(f"  [{i}/{len(names)}] {name}: {type(e).__name__}: {e}")
            fail += 1
            continue
        os.makedirs(dst_dir, exist_ok=True)
        with open(dst, "wb") as fh:
            fh.write(blob)
        info.update(take=name, dataset=ds, member=member,
                    bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
        with open(os.path.join(dst_dir, "points_meta.json"), "w") as fh:
            json.dump(info, fh, indent=1)
        print(f"  [{i}/{len(names)}] {name}: {info['n_points']:,} points, "
              f"{len(blob) / 1e6:.1f} MB read, sha256 {info['sha256'][:12]}")
        done += 1
    print(f"[mps] {ds}: {done} written, {fail} failed")
    return 1 if fail else 0


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=sorted(MEMBER))
    p.add_argument("--urls", required=True,
                   help="the dataset's download-URL JSON (signed, expiring; "
                        "never commit it)")
    p.add_argument("--out", required=True,
                   help="destination root — keep it OUTSIDE this repository")
    p.add_argument("--takes", default=None,
                   help="comma-separated take names, or a directory of them")
    p.add_argument("--force", action="store_true", help="re-fetch existing files")
    a = p.parse_args()
    sys.exit(run(a.dataset, a.urls, a.out, a.takes, a.force))


if __name__ == "__main__":
    main()
