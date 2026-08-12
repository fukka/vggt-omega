# Copyright (c) 2026.
"""Fetch the Aria camera calibration for the four ego-synth 5B source datasets.

ego-synth 5B ships **no camera model** — not in ``meta.json``, not in the npz.
Scoring a model that was fed a rectified image needs one: mapping its depth back
onto the raw fisheye points is fisheye pixel -> ray -> pinhole pixel, and the
first step is exactly the calibration this release omits. This script gets it.

What it fetches, and why that file
----------------------------------
Aria's RGB camera is a **FISHEYE624** model (``projectaria_tools.core.calibration``
lists ``FISHEYE624``, ``KANNALA_BRANDT_K3``, ``LINEAR``, ``SPHERICAL``). Two
places carry it: the factory calibration inside the recording's VRS, and MPS's
``online_calibration`` — the same intrinsics re-estimated over the recording, so
it tracks thermal drift. Only the second is separable from the multi-GB VRS, and
it is what this script pulls.

    aea       mps_slam_calibration.zip  ->  online_calibration.csv     1.6 MB/seq
    nymeria   recording_head.zip        ->  online_calibration.jsonl   (see below)
    egoexo4d  take_trajectory           ->  online_calibration.jsonl
    oxford    aria/<loc>/mps/           ->  (confirm on arrival)

Nymeria is the awkward one, and the fix is the point of this script
------------------------------------------------------------------
Nymeria has no calibration-sized download group: the smallest group containing
``online_calibration.jsonl`` is ``recording_head``, a **576 MB zip** per
sequence. Over the 254 takes ego-synth used that is ~146 GB to obtain a few MB
of calibration.

But it is a *zip*, and a zip keeps its central directory at the **end**. The CDN
honours HTTP range requests (measured: ``HTTP 206`` on a 64 KB tail request, and
all 13 central-directory entries were readable from that tail alone). So one
range request finds the member and a second fetches only its bytes. That is what
:func:`fetch_zip_member` does, and it turns 146 GB into a few hundred MB.

Nothing here is Nymeria-specific: AEA's group is small enough to take whole, but
the same path works if that ever changes.

Usage
-----
The URL JSONs are obtained per dataset (AEA from projectaria.com, Nymeria from
the Aria Dataset Explorer) and carry **signed, expiring** links — they are
credentials, they are not in this repo, and they must not be committed to it.

    python tools/fetch_egosynth_calibration.py aea \\
        --urls AriaEverydayActivities_download_urls.json \\
        --out /data/f.zhang2/ego-synth-5b-calib

    python tools/fetch_egosynth_calibration.py nymeria \\
        --urls nymeria_download_urls.json \\
        --out /data/f.zhang2/ego-synth-5b-calib \\
        --takes /data/f.zhang2/ego-synth-5b/nymeria

``--takes`` restricts the download to the takes ego-synth actually used; point it
at that dataset's directory under the release (or at a file of names, one per
line). Without it every sequence in the JSON is fetched.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import struct
import subprocess
import sys
import zlib
from typing import Dict, List, Optional, Tuple

#: The member each dataset's archive hides the calibration in. AEA's is at the
#: zip root and is **jsonl**, not the ``mps/slam/online_calibration.csv`` the
#: published AEA data-format page lists — read off the real archive, not the doc.
MEMBER = {
    "aea": "online_calibration.jsonl",
    "nymeria": "recording_head/mps/slam/online_calibration.jsonl",
}

#: The one group per dataset that contains it (see the module docstring).
GROUP = {"aea": "mps_slam_calibration", "nymeria": "recording_head"}

#: Aria's RGB model. 15 params: f, cx, cy, k1..k6, p1, p2, s1..s4 — the
#: ``FISHEYE624`` of ``projectaria_tools.core.calibration.CameraModelType``.
RGB_LABEL = "camera-rgb"
FISHEYE624_PARAMS = 15


# --------------------------------------------------------------------------- #
# Ranged zip reading
# --------------------------------------------------------------------------- #

def _curl_range(url: str, start: int, end: int, timeout: int = 120) -> bytes:
    """``bytes=start-end`` of ``url``, via curl.

    curl rather than urllib because it uses the system certificate store; the
    Python that ships with some of these boxes cannot verify the CDN's chain and
    fails with CERTIFICATE_VERIFY_FAILED before any range is even requested.
    """
    p = subprocess.run(
        ["curl", "-sSf", "--retry", "3", "--max-time", str(timeout),
         "-r", f"{start}-{end}", url],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"curl failed ({p.returncode}): "
                           f"{p.stderr.decode()[:200]}")
    return p.stdout


def _central_directory(url: str, size: int, tail: int = 65536) -> bytes:
    """The zip's central directory, fetched from the end of the file.

    Grows the tail request until the End Of Central Directory record is found
    and the directory it points at is fully covered — a zip with many members,
    or a long comment, does not fit in the first guess.
    """
    for want in (tail, 1 << 20, 1 << 24):
        blob = _curl_range(url, max(0, size - want), size - 1)
        i = blob.rfind(b"PK\x05\x06")
        if i < 0:
            continue
        cd_size, cd_off = struct.unpack("<II", blob[i + 12:i + 20])
        start = size - len(blob)
        if cd_off >= start:                      # directory is inside the tail
            return blob[cd_off - start:cd_off - start + cd_size]
        return _curl_range(url, cd_off, cd_off + cd_size - 1)
    raise RuntimeError("no end-of-central-directory record found")


def _find_member(cd: bytes, name: str) -> Tuple[int, int, int]:
    """``(local_header_offset, compressed_size, method)`` for ``name``."""
    off = 0
    while off + 46 <= len(cd):
        if cd[off:off + 4] != b"PK\x01\x02":
            break
        method, = struct.unpack("<H", cd[off + 10:off + 12])
        csize, = struct.unpack("<I", cd[off + 20:off + 24])
        n, m, k = struct.unpack("<HHH", cd[off + 28:off + 34])
        lho, = struct.unpack("<I", cd[off + 42:off + 46])
        fname = cd[off + 46:off + 46 + n].decode("utf-8", "replace")
        if fname == name:
            return lho, csize, method
        off += 46 + n + m + k
    raise KeyError(name)


def fetch_zip_member(url: str, size: int, name: str) -> bytes:
    """One member of a remote zip, without downloading the archive.

    Two ranged reads: the central directory (from the tail), then the member's
    own bytes. See the module docstring for why this exists.
    """
    lho, csize, method = _find_member(_central_directory(url, size), name)
    # The local header repeats the name and may carry a different extra field,
    # so its true length is only known after reading it.
    head = _curl_range(url, lho, lho + 29)
    if head[:4] != b"PK\x03\x04":
        raise RuntimeError(f"no local header at {lho}")
    n, m = struct.unpack("<HH", head[26:30])
    start = lho + 30 + n + m
    raw = _curl_range(url, start, start + csize - 1)
    if method == 0:
        return raw
    if method == 8:
        return zlib.decompress(raw, -zlib.MAX_WBITS)
    raise RuntimeError(f"unsupported zip compression method {method}")


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #

def reduce_rgb(blob: bytes) -> dict:
    """The RGB camera model, out of a whole online-calibration time series.

    ``online_calibration.jsonl`` is MPS re-estimating the calibration over the
    recording — 32 577 records and 145 MB on the Nymeria take measured, for
    three cameras, of which this evaluation needs one. Keeping the series per
    take would be ~37 GB across the 254 Nymeria takes ego-synth used, to carry
    fifteen numbers that barely move.

    So it is reduced here, at fetch time, to the **median** of each parameter
    over every calibrated record, with the interquartile range kept beside it.
    The median rather than the first record because online calibration tracks
    thermal drift and the first seconds of a recording are the least settled;
    the IQR because a parameter that actually moves during a recording must be
    visible as such rather than silently averaged away.

    Resolution is **not** normalised here. These params are at the RGB sensor's
    own native resolution (f~1218.6, c~(1462.1, 1444.4) => a 2880 sensor), while
    ego-synth's frames are 896 and its ``meta.source_width`` is 1408. Rescaling
    is the consumer's job and is done once, explicitly, where the convention can
    be tested — see the ticket.
    """
    import numpy as np
    rows, serials, n_seen, n_cal = [], set(), 0, 0
    for line in io.BytesIO(blob):
        if not line.strip():
            continue
        n_seen += 1
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        for c in rec.get("CameraCalibrations", ()):
            if c.get("Label") != RGB_LABEL:
                continue
            if not c.get("Calibrated", True):
                continue
            p = c.get("Projection", {})
            params = p.get("Params")
            if not params:
                continue
            n_cal += 1
            rows.append(params)
            if c.get("SerialNumber"):
                serials.add(c["SerialNumber"])
            model = p.get("Name")
    if not rows:
        raise RuntimeError(f"no calibrated {RGB_LABEL!r} record found")
    a = np.asarray(rows, float)
    med = np.median(a, axis=0)
    iqr = np.percentile(a, 75, axis=0) - np.percentile(a, 25, axis=0)
    return {
        "label": RGB_LABEL,
        "model": model,
        "params": med.tolist(),
        "params_iqr": iqr.tolist(),
        "n_records": n_seen,
        "n_calibrated": n_cal,
        "serial_numbers": sorted(serials),
        "note": ("median over the recording's online calibration; params are at "
                 "the RGB sensor's native resolution and must be rescaled to the "
                 "frame they will be used on"),
    }


def wanted_takes(spec: Optional[str]) -> Optional[set]:
    """The take names to keep: a directory of them, or a file of names."""
    if not spec:
        return None
    if os.path.isdir(spec):
        return {d for d in os.listdir(spec)
                if os.path.isdir(os.path.join(spec, d))}
    with open(spec) as fh:
        return {ln.strip() for ln in fh if ln.strip()}


def run(ds: str, urls_path: str, out_root: str, takes: Optional[str],
        limit: int = 0, force: bool = False, keep_raw: bool = False) -> int:
    with open(urls_path) as fh:
        doc = json.load(fh)
    seqs: Dict[str, dict] = doc["sequences"]
    keep = wanted_takes(takes)
    names = sorted(seqs)
    if keep is not None:
        missing = keep - set(names)
        names = [n for n in names if n in keep]
        if missing:
            print(f"[calib] {len(missing)} requested take(s) are not in the "
                  f"URL JSON, e.g. {sorted(missing)[:3]} — re-export it from the "
                  f"dataset explorer with those sequences selected")
    if limit:
        names = names[:limit]
    member = MEMBER[ds]
    group = GROUP[ds]
    out_ds = os.path.join(out_root, ds)
    os.makedirs(out_ds, exist_ok=True)
    print(f"[calib] {ds}: {len(names)} take(s) -> {out_ds}")

    done = fail = 0
    for i, name in enumerate(names, 1):
        entry = seqs[name].get(group)
        if entry is None:
            print(f"  [{i}/{len(names)}] {name}: no {group!r} group in the JSON "
                  f"— re-export selecting that group"); fail += 1; continue
        dst_dir = os.path.join(out_ds, name)
        dst = os.path.join(dst_dir, "camera_rgb.json")
        if os.path.exists(dst) and not force:
            done += 1; continue
        try:
            blob = fetch_zip_member(entry["download_url"],
                                    int(entry["file_size_bytes"]), member)
            rgb = reduce_rgb(blob)
        except Exception as e:                    # noqa: BLE001 — report, continue
            print(f"  [{i}/{len(names)}] {name}: {type(e).__name__}: {e}")
            fail += 1
            continue
        os.makedirs(dst_dir, exist_ok=True)
        rgb["take"] = name
        rgb["dataset"] = ds
        with open(dst, "w") as fh:
            json.dump(rgb, fh, indent=1)
        if keep_raw:
            with open(os.path.join(dst_dir, os.path.basename(member)), "wb") as fh:
                fh.write(blob)
        done += 1
        if i % 10 == 0 or i == len(names) or i == 1:
            f, cx, cy = rgb["params"][:3]
            print(f"  [{i}/{len(names)}] {name}: {rgb['model']} "
                  f"f={f:.2f} c=({cx:.1f},{cy:.1f}) from "
                  f"{rgb['n_calibrated']}/{rgb['n_records']} records "
                  f"({len(blob) / 1e6:.1f} MB read)")
    print(f"[calib] {ds}: {done} written, {fail} failed")
    return 1 if fail else 0


def describe(path: str) -> None:
    """Print what a fetched calibration file actually contains."""
    n = os.path.getsize(path)
    with open(path, "rb") as fh:
        head = fh.readline().decode("utf-8", "replace").strip()
    print(f"  {path}  ({n / 1e6:.2f} MB)")
    if path.endswith(".jsonl"):
        rec = json.loads(head)
        cams = [c.get("Label") or c.get("label") for c in
                (rec.get("CameraCalibrations") or rec.get("camera") or [])]
        print(f"    first record keys: {sorted(rec)[:8]}")
        print(f"    cameras: {cams}")
    else:
        print(f"    header: {head[:200]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=sorted(MEMBER))
    p.add_argument("--urls", required=True, help="the dataset's download-URL JSON "
                                                 "(signed, expiring; never commit it)")
    p.add_argument("--out", required=True, help="destination root")
    p.add_argument("--takes", default=None,
                   help="restrict to these takes: a directory of take dirs "
                        "(e.g. the ego-synth release's own <ds>/ folder) or a "
                        "file of names, one per line")
    p.add_argument("--limit", type=int, default=0, help="first N takes only")
    p.add_argument("--force", action="store_true", help="re-fetch existing files")
    p.add_argument("--keep-raw", action="store_true",
                   help="also write the full online_calibration.jsonl beside the "
                        "reduction (145 MB per Nymeria take — see reduce_rgb)")
    p.add_argument("--describe", default=None,
                   help="print what one fetched file contains, and exit")
    a = p.parse_args()
    if a.describe:
        describe(a.describe)
        return
    sys.exit(run(a.dataset, a.urls, a.out, a.takes, a.limit, a.force, a.keep_raw))


if __name__ == "__main__":
    main()
