# Download the Aria camera calibration for the four ego-synth source datasets

**Owner:** gpu
**Files I may touch:** nothing in the repo — this ticket only downloads data.
The fetcher it runs, `tools/fetch_egosynth_calibration.py`, is already landed.
**Blocked by:** none for aea/nymeria/oxford. Ego-Exo4D needs a licence request
that takes ~48 h to approve — **start that first, on day one**, then do the rest
while it clears.

## Goal

Every take ego-synth 5B was built from has its Aria RGB camera model on disk at
`/data/f.zhang2/ego-synth-5b-calib/<ds>/<take>/camera_rgb.json`.

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

### 0. Start the Ego-Exo4D licence request FIRST — it gates everything else

Request at <https://ego4d.dev/request/ego-exo4d> using **f.zhang2@samsung.com**.
Approval takes ~48 h and then emails AWS credentials that **expire in 14 days**,
so do not request it early and let it lapse — request it the day this ticket
starts, and download within the fortnight.

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

### 4. Ego-Exo4D — once the credentials arrive

```bash
egoexo -o /data/f.zhang2/egoexo4d-traj --parts take_trajectory --views ego
```

Then fold `takes/<take>/trajectory/online_calibration.jsonl` into the same
layout. The fetcher's `reduce_rgb` is the function to reuse — it takes the file
contents as bytes and returns the `camera_rgb.json` body — but the egoexo CLI
delivers plain files rather than a remote zip, so this step is a short script,
not a rerun of the fetcher. 1 090 of the 2 380 take dirs have depth; only those
matter.

### 5. Oxford Day-and-Night — no gating, straight from HuggingFace

```bash
pip install -U "huggingface_hub[cli]"
hf download active-vision-lab/oxford-day-and-night --repo-type dataset \
  --include "aria/*/mps/*" --local-dir /data/f.zhang2/oxford-day-and-night
```

Five locations: `bodleian-library`, `hb-allen-centre`, `keble-college`,
`observatory-quarter`, `oxford-robotics-institute`. The README says `mps/` holds
multi-session MPS with "per image camera parameters"; **the exact filenames are
unconfirmed** — the HuggingFace tree view does not expand to file level. Look
first, then map onto the same layout. If `mps/` turns out not to carry the
intrinsics, the `vrs/` folder does (anonymised VRS, released 2026-02-23), read
with `projectaria_tools`' `get_device_calibration()`.

## Done when

- [ ] `/data/f.zhang2/ego-synth-5b-calib/{aea,nymeria,egoexo4d,oxford}/<take>/camera_rgb.json`
      exists for every take that dataset contributes to ego-synth
- [ ] each file's `model` reads `FisheyeRadTanThinPrism` and `params` has 15 entries
- [ ] the take counts match the release: aea 143, nymeria 254, egoexo4d 1 090,
      oxford 124
- [ ] issue commented with the counts and total bytes

## What is already verified, and what is not

**Verified on this Mac against the staged sample**, one take of each of aea and
nymeria: the fetcher runs end to end, the ranged zip read works against the live
CDN, and both takes reduce to a 15-param `FisheyeRadTanThinPrism` with a serial
number.

**Not settled: the resolution and orientation convention.** The params are at
the sensor's native resolution — `f~1218`, `c~(1462, 1444)` can only be a 2880
sensor — while ego-synth's frames are 896 and its `meta.source_width` is 1408.
Scaling 2880 → 896 and trying all four 90° rotations against the sample's own
rect↔fisheye correspondences picks **a different rotation per dataset**:
`aea` 90° CCW, `nymeria` 0°, each the only one with any correspondence inside
2 px. That is consistent with ego-synth's per-dataset scripts carrying
`input_orientation` as a separate argument.

But the best rotation still leaves p05 ≈ 1.4 px rather than sub-pixel, and the
correspondence set used to measure it is **contaminated**: pairs were matched on
exact float16 depth, and two different points sharing a depth get paired. The
~90 % of pairs that are far are most likely false pairs, not model error — some
pairs land at 0.20-0.50 px, which a wrong model could not produce. Settling this
needs an uncontaminated matcher and is the **first step of the ticket that
consumes this data**, not of this one. Do not treat the convention as decided.

## Needs a GPU run afterwards?

No GPU needed — this is a download. It unblocks the rect → de-rect baseline of
the SLAM evaluation, which does need the box.
