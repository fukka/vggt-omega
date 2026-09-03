"""Extract an ADT sequence into the layout this repo's loaders expect.

The six Apartment sequences were extracted before this script existed, so its
first job is to REPRODUCE that extraction: `--verify-against <extracted seq>`
re-derives frames from the raw sources and compares them byte-for-byte with what
is already on disk. An extractor for a second scene that has not been checked
against the first is a silent way to make two scenes incomparable.

Layout written (matching `AriaLocalPairs` / h5's `Seq`):

    <out>/videos_rgb/frame_%06d_<capture_ns>.png     RGB, native size
    <out>/depth_npy/frame_%06d_<capture_ns>.npy      uint16 MILLIMETRES, planar z
    <out>/groundtruth/aria_trajectory.csv            copied from the download
    <out>/camera.json                                THIS sequence's calibration

`camera.json` is the part that is new. The repo's `from_aria()` hard-codes one
device's KB4, and ADT's second scene is recorded on a DIFFERENT device whose
calibration differs materially -- 2.7% in focal length and a sign flip on k4.
Scoring LiteOffice through the Apartment's lens would produce a smooth radial
error, which is indistinguishable from "the model is bad at the rim" and is the
exact bug class that invalidated #38 v1. So every extracted sequence carries the
calibration it was actually recorded with, and `tools/adt_camera.py` builds the
camera from that file.

Depth is written in ADT's own convention -- uint16 millimetres of PLANAR Z,
measured by `VGGT-360-fisheye/checks/check_gt_depth_domain.py` -- because that
is what `Seq.gt_range` expects to divide by cos(theta). Nothing is converted
here; the conversion happens once, downstream, where it is declared.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def _provider(seq_dir: Path):
    from projectaria_tools.projects.adt import (
        AriaDigitalTwinDataPathsProvider, AriaDigitalTwinDataProvider)
    paths = AriaDigitalTwinDataPathsProvider(str(seq_dir))
    return AriaDigitalTwinDataProvider(paths.get_datapaths())


def _camera_json(gt, stream_id) -> dict:
    """Everything needed to rebuild this sequence's lens, from the provider.

    Read from the calibration the DEPTH WAS RENDERED WITH, not from
    `mps/slam/online_calibration.jsonl`: the online one drifts per timestamp and
    is expressed at raw sensor resolution, while this one describes the frames
    actually delivered.
    """
    cal = gt.get_aria_camera_calibration(stream_id)
    if cal is None:
        raise SystemExit("[extract] no camera calibration for the RGB stream")
    w, h = (int(x) for x in cal.get_image_size())
    out = {"label": cal.get_label(), "width": w, "height": h,
           "model": "FISHEYE624", "params": [float(x) for x in cal.projection_params()]}
    for attr, key in (("get_serial_number", "serial"),
                      ("get_max_solid_angle", "max_solid_angle")):
        fn = getattr(cal, attr, None)
        if fn is not None:
            try:
                out[key] = fn() if not isinstance(fn(), (bytes,)) else fn().decode()
            except Exception:
                pass
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", required=True, help="a downloaded ADT sequence dir")
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0, help="0 = all")
    p.add_argument("--verify-against", default=None,
                   help="an already-extracted sequence; compare instead of writing")
    a = p.parse_args(argv)

    seq = Path(a.seq).expanduser()
    from projectaria_tools.core.stream_id import StreamId
    gt = _provider(seq)
    sid = StreamId("214-1")

    cam = _camera_json(gt, sid)
    ts_all = list(gt.get_aria_device_capture_timestamps_ns(sid))
    idx = list(range(0, len(ts_all), max(1, a.stride)))
    if a.max_frames:
        idx = idx[: a.max_frames]
    print(f"[extract] {seq.name}: {len(ts_all)} RGB timestamps, taking {len(idx)}")
    print(f"[extract] camera {cam['label']} {cam['width']}x{cam['height']} "
          f"f={cam['params'][0]:.3f} cx={cam['params'][1]:.3f} cy={cam['params'][2]:.3f}")

    if a.verify_against:
        ref = Path(a.verify_against).expanduser()
        have = sorted((ref / "videos_rgb").glob("*.png"))
        if not have:
            raise SystemExit(f"[extract] nothing to verify against in {ref}")
        stems = {p.stem: p for p in have}
        checked = mismatched = 0
        for i in idx:
            t = ts_all[i]
            stem = f"frame_{i:06d}_{t}"
            if stem not in stems:
                continue
            from PIL import Image
            img = gt.get_aria_image_by_timestamp_ns(t, sid)
            if not img.is_valid():
                continue
            mine = np.asarray(img.data().to_numpy_array())
            theirs = np.asarray(Image.open(stems[stem]).convert("RGB"))
            same_rgb = mine.shape == theirs.shape and np.array_equal(mine, theirs)
            dpath = ref / "depth_npy" / f"{stem}.npy"
            same_d = None
            if dpath.exists():
                dep = gt.get_depth_image_by_timestamp_ns(t, sid)
                if dep.is_valid():
                    md = np.asarray(dep.data().to_numpy_array())
                    td = np.load(dpath)
                    same_d = md.shape == td.shape and np.array_equal(md, td)
            checked += 1
            if not same_rgb or same_d is False:
                mismatched += 1
                print(f"  MISMATCH {stem}: rgb_equal={same_rgb} depth_equal={same_d}")
            if checked >= 8:
                break
        print(f"[extract] verified {checked} frames, {mismatched} mismatched")
        raise SystemExit(0 if (checked and not mismatched) else 1)

    out = Path(a.out).expanduser()
    (out / "videos_rgb").mkdir(parents=True, exist_ok=True)
    (out / "depth_npy").mkdir(parents=True, exist_ok=True)
    (out / "groundtruth").mkdir(parents=True, exist_ok=True)
    (out / "camera.json").write_text(json.dumps(cam, indent=1))
    for name in ("aria_trajectory.csv", "instances.json", "metadata.json",
                 "scene_objects.csv", "3d_bounding_box.csv", "2d_bounding_box.csv"):
        src = seq / name
        if src.exists():
            shutil.copy2(src, out / "groundtruth" / name)

    from PIL import Image
    written = skipped = 0
    for i in idx:
        t = ts_all[i]
        img = gt.get_aria_image_by_timestamp_ns(t, sid)
        dep = gt.get_depth_image_by_timestamp_ns(t, sid)
        if not (img.is_valid() and dep.is_valid()):
            skipped += 1
            continue
        stem = f"frame_{i:06d}_{t}"
        Image.fromarray(np.asarray(img.data().to_numpy_array())).save(
            out / "videos_rgb" / f"{stem}.png")
        d = np.asarray(dep.data().to_numpy_array())
        if d.dtype != np.uint16:
            d = d.astype(np.uint16)
        np.save(out / "depth_npy" / f"{stem}.npy", d)
        written += 1
    print(f"[extract] wrote {written} frames ({skipped} skipped) -> {out}")


if __name__ == "__main__":
    main()
