# ADT skeleton sequences: how many pixels are hands/body, and where

**Owner:** gpu
**Status:** open — not started.
**Files I may touch:** new script `autoresearch/experiments/h4-dynamics/code/hand_pixel_stats.py`
(create), results to the `results` branch as `results/autoresearch-h4-stats/*.json`.
Nothing else.
**Blocked by:** none.
**What is waiting on it:** autoresearch hypothesis H4 ("dynamic hand regions
contribute a disproportionate share of depth error") needs to know, before any
modeling, whether hands are even a measurable fraction of ADT egocentric frames.
The Mac-local sequence (`Apartment_release_clean_seq131`) has zero human/hand
instances (all 357 instances are static objects — verified 2026-08-18), so this
is box-only.

## The task

Pick 2–3 ADT sequences WITH skeleton (names contain `skeleton`, or check
`instances.json` for human/body categories) from
`/group-volume/Fengjia/data/projectaria_tools_adt_data_clean`. For every frame
that has both `seg_npy` and `depth_npy` (subsample to ~100 frames/seq is fine):

1. Identify dynamic-instance pixels: any instance whose category is human /
   body / hand (list what categories you actually find in `instances.json` —
   ADT labels the wearer's body parts; record the exact names used).
2. Report per frame, inside the RGB imaged cone (theta <= 54.83 deg):
   - fraction of pixels that are dynamic;
   - their distribution over incidence angle theta (8 bins to 54.83 deg) —
     egocentric hands are expected to concentrate low-center; we want the
     actual radial profile;
   - GT depth range of dynamic pixels (median, p10, p90) vs static pixels.
3. One JSON per sequence: `{seq, frames: [{frame_id, dyn_frac, dyn_theta_hist,
   dyn_depth_med, static_depth_med, ...}], categories_found: [...]}`.

No model inference in this ticket — statistics of the GT only. The eval protocol
that consumes these lands CPU-side under
`autoresearch/experiments/h4-dynamics/` with its own locked protocol.

## Acceptance

- 2–3 JSONs on `results`, a comment here naming the sequences and the dynamic
  categories found, and one sentence on whether hands are >1% of cone pixels
  (if they are ~0%, H4 dies cheaply and that is a fine outcome).
