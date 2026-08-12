# Download the Aria camera calibration for the four ego-synth source datasets

**Owner:** gpu
**Files I may touch:** nothing in the repo — this ticket only downloads data.
The fetcher it runs, `tools/fetch_egosynth_calibration.py`, is already landed.
**Blocked by:** none.

> **Ego-Exo4D is PAUSED** by the owner's decision. Do not open the licence
> request. Three datasets — `aea`, `nymeria`, `oxford` — are in scope; the
> evaluation runs on those and `egoexo4d` is simply absent from it until told
> otherwise. Everything below is written for the three.

## Goal

Every `aea`, `nymeria` and `oxford` take ego-synth 5B was built from has its Aria
RGB camera model on disk at
`/data/f.zhang2/ego-synth-5b-calib/<ds>/<take>/camera_rgb.json`.

## Read this before starting

The calibration this ticket downloads has **not yet been shown to describe
ego-synth's frames**. `slambench/verify_camera.py` measures that and currently
fails at ~4 px median reprojection against a sub-pixel bar — see ticket 013 for
what has been ruled out. Downloading is still the right next step, because the
open question needs many takes to settle rather than the two staged locally, but
**do not treat these files as usable until `verify_camera` passes**. The `raw`
baseline needs none of them and is unaffected.

## Why this is needed

ego-synth 5B ships **no camera model** — checked in all four datasets' `meta.json`:
`aea` and `oxford` carry nothing, `nymeria`'s only "distortion" string is
`fps_note`'s "motion distortion", and `egoexo4d` records a *path* to the source
release's calibration, not its values.

Without it the SLAM evaluation can only run the raw-fisheye baseline. The
rectify → model → de-rectify baseline maps a predicted depth back onto the raw
fisheye points, which is *fisheye pixel → ray → pinhole pixel*, and the first
step is exactly the model this release omits.

The model is Aria's **FisheyeRadTanThinPrism** (`FISHEYE624` in
`projectaria_tools.core.calibration`), 15 params: `f, cx, cy, k1..k6, p1, p2,
s1..s4`. It is **per device**, not per dataset — two takes measured here differ
by `f` 1214.03 vs 1218.65 and `cx` 1469.5 vs 1462.1, which is 2.2 px at the 896
frame. A nominal calibration will not do.

## What to download, and the one trick that makes it cheap

MPS's `online_calibration` carries it, and unlike the factory calibration it is
separable from the multi-GB VRS.

| dataset | group | member | cost |
|---|---|---|---|
| `aea` | `mps_slam_calibration` | `online_calibration.jsonl` | 1.6 MB/seq, 0.18 GB for all 143 |
| `nymeria` | `recording_head` | `recording_head/mps/slam/online_calibration.jsonl` | see below |
| `egoexo4d` | `take_trajectory` | `trajectory/online_calibration.jsonl` | small |
| `oxford` | `aria/<loc>/mps/` | confirm on arrival | unknown |

**AEA's member is `online_calibration.jsonl` at the zip root** — *not* the
`mps/slam/online_calibration.csv` the published AEA data-format page lists. That
was read off the real archive; the doc is wrong.

**Nymeria has no calibration-sized group.** The smallest group containing the
file is `recording_head`, a 576 MB zip per sequence — 146 GB over the 254 takes
ego-synth used, to obtain a few numbers. But it is a *zip*, whose central
directory sits at the end, and the CDN honours range requests (measured:
`HTTP 206` on a 64 KB tail, all 13 directory entries readable from that tail
alone). `tools/fetch_egosynth_calibration.py` therefore takes two ranged reads
per sequence instead of the archive.

The fetcher also **reduces on the way in**: `online_calibration.jsonl` is a time
series (32 577 records / 145 MB on the Nymeria take measured), of which this
evaluation needs one camera. It writes the median of each parameter with the
IQR beside it, ~900 bytes per take. On AEA the IQR is exactly 0 — that dataset's
online calibration does not move over a recording.

## Steps

### 1. Get the two URL JSONs onto the box

They are on the Mac at `~/Downloads/`:

    AriaEverydayActivities_download_urls.json     143 sequences
    nymeria_download_urls.json                   1100 sequences

**They are signed, expiring credentials and must never be committed** — this
repo is public and its history is permanent and mirrored. Move them out of band
(`scp`), the way `raytun3r/experiments/make_local_sample.py` describes for
licensed data. Both sets of links **expire 2026-09-11** (decoded from the URLs'
own `oe` parameter), so this ticket has until then.

### 2. AEA — all 143 takes

ego-synth used all 143 AEA sequences and the JSON carries exactly those 143, so
no filtering is needed.

```bash
python tools/fetch_egosynth_calibration.py aea \
  --urls /data/f.zhang2/urls/AriaEverydayActivities_download_urls.json \
  --out  /data/f.zhang2/ego-synth-5b-calib
```

### 3. Nymeria — the 254 takes ego-synth used, of 1100 in the JSON

`--takes` accepts the release's own dataset directory and keeps only those names:

```bash
python tools/fetch_egosynth_calibration.py nymeria \
  --urls  /data/f.zhang2/urls/nymeria_download_urls.json \
  --out   /data/f.zhang2/ego-synth-5b-calib \
  --takes /data/f.zhang2/ego-synth-5b/nymeria
```

If it reports takes missing from the JSON, re-export it from the
[Nymeria explorer](https://explorer.projectaria.com/nymeria) with those
sequences selected and the `recording_head` group ticked.

### 4. Oxford Day-and-Night — no gating, but no cheap path either

This one needs a decision before any bytes move, because **the ranged-zip trick
does not apply**. Measured against the HuggingFace API:

    aria/<loc>/mps/multi/day_23.tar.gz          18.7 GB
    aria/<loc>/mps/multi/day_night_44.tar.gz    24.3 GB
    aria/<loc>/mps/multi/night_21.tar.gz         5.6 GB
    aria/<loc>/vrs/blur/<uuid>.vrs         1.7-2.5 GB each, 44 files = 108.7 GB

...and that is **one** of five locations (`bodleian-library`, `hb-allen-centre`,
`keble-college`, `observatory-quarter`, `oxford-robotics-institute`).

A `.tar.gz` is a gzip *stream*: unlike a zip it has no central directory to seek
to, so a member cannot be range-read. Two options, cheapest experiment first:

1. **Stream the smallest archive and abort early.** `night_21.tar.gz` is 5.6 GB;
   tar stores members sequentially, so `curl … | tar -xO --wildcards
   '*online_calibration*'` gets the file without ever writing the archive, and
   can be killed once it appears. Whether that is cheap depends entirely on
   where in the tar the member sits — try it on one location and measure.
2. **Per-recording VRS.** `vrs/blur/<uuid>.vrs` carries the factory device
   calibration, read with `projectaria_tools`' `get_device_calibration()`. That
   is a *different* calibration from the MPS online one used for aea/nymeria —
   factory rather than re-estimated — which is a difference worth recording in
   the results if this route is taken.

`mps/multi/<group>.txt` lists the VRS uuids in each multi-SLAM group, and is
under 2 KB, so it is free to fetch first and tells you which recordings a group
covers.

**Unresolved:** whether Oxford's multi-session MPS even contains a per-recording
`online_calibration`. The README says "per image camera parameters"; nobody has
looked inside. Option 1's experiment answers that too.

## Done when

- [ ] `/data/f.zhang2/ego-synth-5b-calib/{aea,nymeria,oxford}/<take>/camera_rgb.json`
      exists for every take that dataset contributes to ego-synth
- [ ] each file's `model` reads `FisheyeRadTanThinPrism` and `params` has 15 entries
- [ ] the take counts match the release: aea 143, nymeria 254, oxford 124
      (`egoexo4d` is paused and out of scope)
- [ ] issue commented with the counts, total bytes, and which Oxford route worked

## What is verified, and what is not

**Verified on this Mac against the live CDN and the staged sample**, one take
each of aea and nymeria: the fetcher runs end to end, the ranged zip read works,
and both takes reduce to a 15-param `FisheyeRadTanThinPrism` with a serial
number and a `T_Device_Camera`.

**The calibration has NOT been shown to describe ego-synth's frames.**
`slambench/verify_camera.py` exists now and measures exactly this, by projecting
the release's own rectified points through the fetched model and comparing the
resulting pixel cloud against the actual fisheye cloud. It reports

    best case ~4 px median, ~5 % of points within 1 px

against a 0.5-2 % chance rate and a sub-pixel bar — i.e. better than random and
nowhere near right. Ruled out, each measured rather than argued:

* all four 90° rotations, and a continuous roll swept at 2° (no peak anywhere);
* the resolution — implied sensor size swept 1000-4200 px; the best (~2820-2840)
  still leaves 1.4-1.9 px median and only 10-21 % within 1 px;
* the device-to-camera extrinsic, a real ~38.7° tilt that is not the answer;
* the projection implementation, now the reference `projectaria_tools` one.

Still open: a **crop** before the resize (only scale was swept, not the principal
point — a joint search over both is the obvious next move), a rectification axis
tilted rather than merely rolled, or an online calibration describing a different
stream than the one ego-synth read.

This is why the download is still worth doing — settling it wants many takes,
not the two staged here — and equally why **nothing downstream may use these
files until `verify_camera` passes**. `camera.require_verified` enforces that in
code.

## Needs a GPU run afterwards?

No GPU needed — this is a download. It unblocks the rect → de-rect baseline of
the SLAM evaluation, which does need the box.
