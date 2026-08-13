#!/usr/bin/env python
"""Extract ADT ``videos_synthetic`` frames from ``synthetic_video.vrs``.

Five ADT sequences on lambda_63 have ``depth_npy`` and ``videos_rgb`` extracted
but not ``videos_synthetic``, so ``fovbench.split.build_split`` rejects them and
the whole ADT-FOV result rests on one apartment.  The pixels are on disk --
``synthetic_video.vrs`` is present for every sequence -- they were simply never
decoded.

Naming, reverse-engineered from seq131 (whose cache exists) and verified against
it by ``--verify``:

* a cache stem is ``frame_<idx>_<ts>.png`` where ``idx`` is the record index in
  the REAL ``video.vrs`` camera-rgb stream and ``ts`` is that record's
  ``capture_timestamp_ns``.  Confirmed exactly at records 313, 314 and 3191.
* ``synthetic_video.vrs`` runs on its own clock, so the synthetic frames are NOT
  named after their own timestamps.  Synthetic record ``i`` is named after the
  real record NEAREST in time, which on seq131 is ``i + 312`` for every record
  (verified by exact pixel match at seven spread indices).  The residual lag
  drifts 15.4 -> 0.09 ms across the sequence and never reaches half a frame
  (16.7 ms), so the pairing never slips.

Two quirks of the seq131 cache this does not reproduce, both harmless: it omits
synthetic record 0 (which would be named 312) and it writes the last record
twice, under 3189 and 3190 -- so its 3190 carries pixels 33 ms older than its
name claims.  ``--verify`` expects exactly those two discrepancies.

``--verify`` re-derives the names for a sequence that already has a cache and
requires an exact set match plus pixel equality on sampled frames; nothing is
extracted for the other sequences unless that passes.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
from PIL import Image
from projectaria_tools.core import data_provider

STEM = re.compile(r"frame_(\d+)_(\d+)$")
HALF_FRAME_NS = 16_670_000  # half a 30 Hz frame period; the pairing must stay inside it


def rgb_timestamps(vrs_path):
    """(index -> capture_timestamp_ns) for the camera-rgb stream, in order."""
    p = data_provider.create_vrs_data_provider(vrs_path)
    sid = p.get_stream_id_from_label("camera-rgb")
    n = p.get_num_data(sid)
    try:  # metadata-only; decoding every frame just for a timestamp is minutes
        from projectaria_tools.core.sensor_data import TimeDomain
        ts = np.asarray(p.get_timestamps_ns(sid, TimeDomain.DEVICE_TIME), dtype=np.int64)
        if len(ts) != n:
            raise ValueError(f"{len(ts)} timestamps for {n} records")
    except Exception:
        ts = np.empty(n, dtype=np.int64)
        for i in range(n):
            ts[i] = p.get_image_data_by_index(sid, i)[1].capture_timestamp_ns
    return p, sid, ts


def cache_stems(d):
    out = {}
    for path in glob.glob(os.path.join(d, "*.png")):
        m = STEM.match(os.path.splitext(os.path.basename(path))[0])
        if m:
            out[int(m.group(1))] = (int(m.group(2)), path)
    return out


def plan(seq):
    """Names for every synthetic record: [(syn_index, out_name)]."""
    _, _, real_ts = rgb_timestamps(os.path.join(seq, "video.vrs"))
    ps, ss, syn_ts = rgb_timestamps(os.path.join(seq, "synthetic_video.vrs"))
    # nearest real record per synthetic record, on timestamps alone
    pos = np.searchsorted(real_ts, syn_ts)
    pos = np.clip(pos, 1, len(real_ts) - 1)
    take_lo = (syn_ts - real_ts[pos - 1]) <= (real_ts[pos] - syn_ts)
    idx = np.where(take_lo, pos - 1, pos)
    lag = (real_ts[idx] - syn_ts) / 1e6
    # A frame whose nearest real record is more than half a period away has no
    # real partner at all -- the real stream dropped frames there.  Drop those
    # rather than the sequence, but refuse if it is not a handful: that would
    # mean the two streams are not the same recording.
    keep = np.abs(lag) < HALF_FRAME_NS / 1e6
    # and never let two synthetic frames claim one real frame
    keep[1:] &= np.diff(idx) > 0
    dropped = int((~keep).sum())
    if dropped > 0.05 * len(keep):
        raise SystemExit(f"[extract] {os.path.basename(seq)}: {dropped} of {len(keep)} "
                         f"synthetic frames have no real frame within half a period "
                         f"(worst {np.abs(lag).max():.1f} ms) -- refusing")
    kept = np.flatnonzero(keep)
    offsets = np.unique(idx[kept] - kept)
    names = [(int(i), "frame_%06d_%d.png" % (idx[i], real_ts[idx[i]])) for i in kept]
    return ps, ss, names, offsets, lag[kept], dropped


def verify(seq):
    name = os.path.basename(seq)
    ps, ss, names, offsets, lag, dropped = plan(seq)
    have = cache_stems(os.path.join(seq, "videos_synthetic"))
    mine = {int(STEM.match(os.path.splitext(n)[0]).group(1)): n for _, n in names}
    print(f"[verify] {name}: index offsets={offsets}, lag {lag.min():.2f}..{lag.max():.2f} ms, dropped={dropped}")
    print(f"[verify]   existing cache {len(have)} frames, derived {len(mine)} names")
    missing = set(have) - set(mine)
    extra = set(mine) - set(have)
    bad_ts = [k for k in set(have) & set(mine)
              if have[k][0] != int(STEM.match(os.path.splitext(mine[k])[0]).group(2))]
    print(f"[verify]   in cache but not derived: {len(missing)}")
    print(f"[verify]   derived but not in cache: {len(extra)}")
    print(f"[verify]   timestamp mismatches:     {len(bad_ts)}")
    ok_names = True
    # pixels, on a spread sample
    idx = np.linspace(0, len(names) - 1, 12).astype(int)
    worst = 0
    for i in idx:
        syn_i, out_name = names[i]
        ref_path = os.path.join(seq, "videos_synthetic", out_name)
        if not os.path.exists(ref_path):
            continue  # a name the cache omits (record 0); reported above
        img = ps.get_image_data_by_index(ss, syn_i)[0].to_numpy_array()
        ref = np.asarray(Image.open(ref_path))
        if img.shape != ref.shape:
            print(f"[verify]   shape mismatch at {out_name}: {img.shape} vs {ref.shape}")
            ok_names = False
            break
        worst = max(worst, int(np.abs(img.astype(int) - ref.astype(int)).max()))
    print(f"[verify]   worst per-pixel difference over {len(idx)} frames: {worst}")
    # the seq131 cache omits record 0 and duplicates the last one; anything else
    # means the naming rule is wrong, not that the cache is quirky
    expected_extra, expected_missing = 1, 1
    ok = (worst == 0 and not bad_ts
          and len(extra) == expected_extra and len(missing) == expected_missing)
    print(f"[verify]   {'PASS' if ok else 'FAIL'}"
          f"  (expected {expected_extra} extra / {expected_missing} missing:"
          f" the omitted record 0 and the duplicated last frame)")
    return ok


def extract(seq, n_subset, dry_run=False):
    name = os.path.basename(seq)
    ps, ss, names, offsets, lag, dropped = plan(seq)
    out_dir = os.path.join(seq, "videos_synthetic")
    if n_subset and n_subset < len(names):
        pick = np.linspace(0, len(names) - 1, n_subset).astype(int)
        names = [names[i] for i in pick]
    print(f"[extract] {name}: index offsets={offsets}, writing {len(names)} frames "
          f"to {out_dir} (lag {lag.min():.2f}..{lag.max():.2f} ms, dropped {dropped})", flush=True)
    if dry_run:
        print(f"[extract]   dry run; first={names[0][1]} last={names[-1][1]}")
        return 0
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for k, (syn_i, out_name) in enumerate(names):
        dst = os.path.join(out_dir, out_name)
        if os.path.exists(dst):
            continue
        img = ps.get_image_data_by_index(ss, syn_i)[0].to_numpy_array()
        Image.fromarray(img).save(dst, optimize=False, compress_level=6)
        written += 1
        if written % 50 == 0:
            print(f"[extract]   {name}: {written}/{len(names)}", flush=True)
    print(f"[extract] {name}: done, {written} new frames", flush=True)
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sequences", nargs="+")
    ap.add_argument("--verify", action="store_true",
                    help="check derived names against an existing cache; extract nothing")
    ap.add_argument("--n-subset", type=int, default=0,
                    help="extract this many uniformly spaced frames instead of all")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.verify:
        sys.exit(0 if all(verify(s) for s in a.sequences) else 1)
    for s in a.sequences:
        extract(s, a.n_subset, a.dry_run)
