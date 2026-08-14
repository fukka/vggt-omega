# Download the Aria camera calibration for the four ego-synth source datasets

**Owner:** cpu (the fetch) → gpu (the takes list, and consuming the result)
**Status:** **done** — calibrations fetched for aea and nymeria; oxford and egoexo4d outstanding, which is what gates rect_derect on those two (see #020).
**Files I may touch:** nothing in the repo — this ticket only downloads data.
The fetcher it runs, `tools/fetch_egosynth_calibration.py`, is already landed.
**Blocked by:** nothing any more. Nymeria's take list is off the box, and Oxford
has a route.

> **LOCK, 2026-08-14 — `gpu` is running the Nymeria fetch. Do not start it.**
> The take list went back over the phone (254 names, 9641 B, sha256
> `5304711f…`), and rather than round-trip it, the `gpu` session pulled
> `nymeria_download_urls.json` off the phone and started the fetch on its own
> Mac. It measures **~53 s/take there, not the ~7.5 s/take this ticket records**
> — that figure was measured on the `cpu` machine's link — so it is a ~3 hour
> job finishing around 17:10 EDT, not 32 minutes. Two sessions fetching the same
> 254 takes would be 160 GB of duplicated reads against a signed URL set that
> expires 2026-09-11.

## The route changed — fetch on the CPU machine, not the box

This ticket used to say: move the signed URL JSONs to the box, then fetch there.
That had the credentials crossing machines to reach the data, and #18 stalled on
exactly that — the JSONs are on the CPU machine and neither this Mac's siblings
nor lambda_63 had them.

**It is the wrong direction.** The fetcher reduces 5-80 MB of `online_calibration`
per take to about **1 KB** on the way in. So run it where the URL JSONs already
are, and move the *reduction* — a few hundred KB for a whole dataset — to the box
instead. The credentials never leave the machine they were downloaded onto, and
the transfer is small enough to go over any channel at hand.

Measured on the CPU machine, over Wi-Fi:

| dataset | per take | 
|---|---|
| `aea` | ~0.35 s, 5.3 MB read, 1 KB written |
| `nymeria` | ~7.5 s, 66-82 MB read, 1 KB written |

so AEA's 143 takes cost about 4 minutes and Nymeria's 254 about 32.

> **Ego-Exo4D is PAUSED** by the owner's decision. Do not open the licence
> request. Three datasets — `aea`, `nymeria`, `oxford` — are in scope; the
> evaluation runs on those and `egoexo4d` is simply absent from it until told
> otherwise. Everything below is written for the three.

## Goal

Every `aea`, `nymeria` and `oxford` take ego-synth 5B was built from has its Aria
RGB camera model on disk at
`/data/f.zhang2/ego-synth-5b-calib/<ds>/<take>/camera_rgb.json`.

## Read this before starting

**The calibration is now known to describe ego-synth's frames, and this is no
longer a blocker — but the resolution convention is not the obvious one.**
`verify_camera` passes on both staged takes at 0.29 px median reprojection with
96.9 % of points inside 1 px, against a 0.5 px bar. Ticket 013 has the chain and
its provenance. The short version, because it will bite anyone who reads these
files with a naive rescale:

> The MPS `online_calibration` RGB intrinsics are at the **2880 full sensor**.
> The stream every one of these datasets was recorded at is **1408**, which is a
> **2816 centre crop binned 2x** — not a plain downsample. Scaling 2880 → 896
> directly is wrong by 2.3 % in focal, which is 6.7 px of reprojection.
> This is projectaria_tools issue #322, and `slambench/camera.py` implements it.

So these files are usable, and `slambench.camera.load` is the only thing that
should ever read them. What this ticket is still needed for is **scale**: the
verdict rests on one take per dataset, and `oxford` has never been measured at
all. The `raw` baseline needs none of them and is unaffected.

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

### 1. The URL JSONs stay put

They are on the CPU machine at `~/Desktop/ADT/`:

    AriaEverydayActivities_download_urls.json     143 sequences
    nymeria_download_urls.json                   1100 sequences

> They lived in `~/Downloads/` until 2026-08-14 and were **moved out**: macOS
> TCC does not grant this agent the Downloads folder, so `cp`, `head` and
> `adb push` all returned `Operation not permitted` there. AEA was fetched
> before that took effect, which is why it succeeded and nothing after it did.
> Anything the agent must read belongs outside `~/Downloads`.

**They are signed, expiring credentials and must never be committed** — this
repo is public and its history is permanent and mirrored. They also no longer
need to be copied anywhere: the fetch happens on the machine that already holds
them (see the route note above). The links **expire 2026-09-11** (decoded from
the URLs' own `oe` parameter), so this ticket has until then.

### 2. AEA — all 143 takes — **DONE**

ego-synth used all 143 AEA sequences and the JSON carries exactly those 143, so
no filtering is needed.

```bash
python tools/fetch_egosynth_calibration.py aea \
  --urls ~/Desktop/ADT/AriaEverydayActivities_download_urls.json \
  --out  ~/Desktop/ADT/ego-synth-5b-calib
```

Ran on the CPU machine, **143/143 written, 0 failed**, ~4 minutes, 158 KB on
disk (34 KB tarred). Every file is `FisheyeRadTanThinPrism` with 15 params.

Two things the full set shows that one take could not:

* **7 distinct Aria devices** across the 143 takes, `f` spanning 1213.09–1220.47
  at the 2880 sensor — 7.4 px there, 2.3 px at the 896 frame. This ticket's claim
  that a nominal calibration will not do is now measured over a dataset rather
  than argued from two takes.
* The fetched set is **consumable end to end**: `verify_camera` run against it
  on the staged take reproduces the verdict exactly — rot 90°, NN median
  **0.29 px, 96.9 % within 1 px**, twin 0.31 px, and 4.0–4.1 px for the other
  three quarter turns.

What is left for this dataset is moving those 42 KB to the box — **in flight**:

    egosynth-calib.tar.gz   42 024 bytes
    sha256  47e9918f8d96c57c157c58f4ab9abb467dfd461be78142d38108ef3aa395d0e7
    holds   aea/ 143 takes, nymeria/ 1 (the staged take), SHA256SUMS.txt

on the phone at `/sdcard/Documents/`, beside `egosynth-calib-README.txt`, which
repeats the resolution warning above. On the box:

```bash
mkdir -p /data/f.zhang2/ego-synth-5b-calib
tar xzf egosynth-calib.tar.gz -C /data/f.zhang2/ego-synth-5b-calib
cd /data/f.zhang2/ego-synth-5b-calib && shasum -c SHA256SUMS.txt | grep -v OK
```

The last line should print nothing. The manifest is **per file**, not just the
tarball, because a truncated calibration still parses as JSON and still produces
numbers — a corrupt transfer here fails as a plausible result, not as an error.

**Re-fetching on the box instead would not be equivalent.** `verify_camera`'s
0.29 px verdict was measured against *these bytes*; a re-fetch produces a set
nobody has checked, at 0.76 GB of CDN reads, and makes the expiring credentials
a standing dependency of the box. Ship the bytes and check the hash.

### 3. Nymeria — the 254 takes ego-synth used, of 1100 in the JSON

**This one needs a file off the box first, and there is no way around it.**
`--takes` filters by the release's own directory names, and those names exist in
exactly one place: `/data/f.zhang2/ego-synth-5b/` on lambda_63. The release is
not published anywhere the names could be listed remotely, and the CPU machine
holds one Nymeria take, so nothing local can name the other 253. Fetching all
1100 blind would be ~2.3 hours and ~80 GB read to throw three quarters away.
So: on the box,

```bash
ls /data/f.zhang2/ego-synth-5b/nymeria > nymeria_takes.txt
```

(~10 KB, no licence concern — it is a list of names), then on the CPU machine:

```bash
python tools/fetch_egosynth_calibration.py nymeria \
  --urls  ~/Desktop/ADT/nymeria_download_urls.json \
  --out   ~/Desktop/ADT/ego-synth-5b-calib \
  --takes nymeria_takes.txt
```

`--takes` accepts a directory of take dirs *or* a file of names, one per line.
Budget ~32 minutes. If it reports takes missing from the JSON, re-export it from
the [Nymeria explorer](https://explorer.projectaria.com/nymeria) with those
sequences selected and the `recording_head` group ticked.

### 4. Oxford Day-and-Night — **route 2, and it is cheap after all**

> **SETTLED 2026-08-14.** Route 2 was taken and measured. Two things this
> section did not know:
>
> **The take names map 1:1 onto the release.** ego-synth's
> `<loc>__<uuid>` is exactly `aria/<loc>/vrs/blur/<uuid>.vrs` in
> `active-vision-lab/oxford-day-and-night`, and the per-location counts match
> the release take-for-take: bodleian 44, keble 24, observatory 25, robotics 21,
> hb-allen 10 = **124**. So there is no group/member puzzle here at all.
>
> **The 1.7–2.5 GB is not the price.** The device calibration lives in the VRS
> header, so a **head-only ranged read reproduces it exactly**. Verified against
> a full 1.67 GB download of `1b688365`: 32 MB, 8 MB and even **2 MB** heads all
> give `f, cx, cy = 610.0692, 716.6490, 711.0387`, identical to the whole file.
> The fetch therefore costs ~1 GB for all 124 at an 8 MB head, not ~250 GB.
>
> **But the resolution convention differs from aea/nymeria, and this is the
> thing that will bite.** The factory calibration is at the **1408 stream**,
> while the MPS `online_calibration` for aea/nymeria is at the **2880 full
> sensor**: `aea f / oxford f = 1.99`. `slambench.camera.load` implements the
> 2880 → 2816 crop → bin 2× chain, so handing it a 1408-native file would apply
> that reduction a *second* time and halve the focal again — silently, and with
> plausible-looking numbers out the other end. **These files are therefore
> staged at `/data/f.zhang2/oxford-calib-factory/`, deliberately NOT inside
> `ego-synth-5b-calib/`.** Teaching `camera.load` to carry a per-file source
> resolution is a code change and so a separate ticket, per this ticket's own
> "nothing in the repo" rule and #020's.
>
> Oxford still cannot join `rect_derect` even once that lands: its rotation has
> never been measured and `camera.VERIFIED_ROTATION` excludes it by design.
> Calibration was necessary, not sufficient.

The original survey, kept for the argument. **The ranged-zip trick does not
apply**; measured against the HuggingFace API:

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

**Still unresolved, and now optional:** whether Oxford's multi-session MPS
contains a per-recording `online_calibration`. Nobody has looked inside. It only
matters if the factory/online difference above turns out to bite — route 2
already produces a 15-param `FISHEYE624` per take, so this is a question about
*which* calibration is better, not about whether one is available.

## Done when

- [x] **aea** — 143/143 fetched on the CPU machine, 0 failed, 148 KB
- [x] **aea** — handed off and **landed**: the 42 KB tarball came off the phone
      on 2026-08-14, sha256 matched, and its per-file manifest verified
      **144/144 with 0 mismatches on both `lambda_63` and `space-container`**
      (`/data/f.zhang2/ego-synth-5b-calib` and
      `/group-volume/Fengjia/data/ego-synth-5b-calib`). The nymeria take in it
      is the staged one, so aea is 143 and nymeria is 1 so far.
- [x] **nymeria** — `nymeria_takes.txt` is off the box and back on the phone
      (254 names, sha256 verified both sides). The fetch itself is **running on
      the `gpu` session's Mac** — see the lock at the top.
- [ ] **oxford** — route 2 chosen (per-recording VRS factory calibration). The
      take names map **1:1** onto the release: ego-synth's `<loc>__<uuid>` is
      exactly `aria/<loc>/vrs/blur/<uuid>.vrs` in
      `active-vision-lab/oxford-day-and-night`, and the per-location counts
      match the release exactly (bodleian 44, keble 24, observatory 25,
      robotics 21, hb-allen 10 = 124). Cost is the problem, not identification:
      ~2 GB per take, ~250 GB for all 124.
- [x] each fetched file's `model` reads `FisheyeRadTanThinPrism` and `params` has
      15 entries — checked across all 143
- [ ] `/data/f.zhang2/ego-synth-5b-calib/{aea,nymeria,oxford}/<take>/camera_rgb.json`
      on the **box**, for every take that dataset contributes to ego-synth
- [ ] the take counts match the release: aea 143, nymeria 254, oxford 124
      (`egoexo4d` is paused and out of scope)
- [ ] issue commented with the counts, total bytes, and which Oxford route worked

## What is verified, and what is not

**The fetch is verified.** Against the live CDN, on the CPU machine: the ranged
zip read works, and all 143 AEA takes reduce to a 15-param
`FisheyeRadTanThinPrism` with a serial number and a `T_Device_Camera`.

**And the calibration is now verified to describe ego-synth's frames** — which
this section, before ticket 013's `a0150f9`, said it was not. It reported ~4 px
median and ~5 % of points within 1 px, and listed a crop-before-the-resize as one
of three open guesses. **That guess was right**, and it was not the only
correction: see ticket 013 for the chain and its provenance. `verify_camera` now
reads **0.29 px median, 96.9 % within 1 px** on the staged takes of `aea` and
`nymeria`, and both are in `camera.VERIFIED_ROTATION`.

What has *not* been settled is scale: that verdict rests on **one take per
dataset**, and `oxford` has never been measured at all. So the rule still holds,
and `camera.require_verified` still enforces it in code — but it is now a rule
about which datasets have been checked, not about whether the files are usable.

## Needs a GPU run afterwards?

No GPU needed — this is a download. It unblocks the rect → de-rect baseline of
the SLAM evaluation, which does need the box.
