# Separate the two experiments: delete fovbench's ego-synth arm, and make the seam two-sided

**Owner:** cpu — **done**, landed on `organized`.
**Files touched:** `tests/` (new), `fovbench/{datasets_egosynth,split,run,report,geometry}.py`,
`fovbench/tests/{test_egosynth,test_report}.py`, `fovbench/README.md`,
`slambench/{__init__.py,tests/test_separation.py}`
**Blocked by:** none.

## Goal

`fovbench/` reads ADT and nothing else; `slambench/` reads ego-synth 5B and
nothing else; a test at the repository root enforces both directions and fails on
a *duplicate* as well as on an import.

## What was wrong

`slambench/tests/test_separation.py` forbade `slambench` from importing
`fovbench`, and its own docstring said why: *“the SLAM data was implemented
inside the FOV experiment.”* That was fixed by half. `slambench/data.py` was
written as a clean ego-synth reader; `fovbench/datasets_egosynth.py` — 700 lines
— was never removed.

The guard could not see it, because it asked the wrong question:

    it asked     does slambench IMPORT fovbench?
    it needed    does either package READ the other's dataset?

`fovbench` never imported `slambench`. It reimplemented it. Twelve of the same
definitions — `RES`, `DATASETS`, `DEFAULT_SIGMA_MAX`, `MIN_FRAME_POINTS`,
`find_takes`, `read_card`, `read_points`, `FramePoints`, `decode_frames`,
`sample_prediction`, `_clip_sort_key`, the `np.rint` overflow guard — each one a
gotcha from the same data card, implemented twice.

Three things made it worth deleting rather than reconciling:

* **It has produced no result.** Nothing on the `results` branch, nothing in
  `GPU_EXPERIMENTS.md`.
* **It has already cost a collision.** Commit `33d3c55` is *“back out another
  session’s in-flight egosynth work.”* Two sessions share this working tree.
* **A correction to one copy does not reach the other.** Ticket 016 documented
  that ego-synth’s planar-z convention rests on the data card alone; that caveat
  landed in `slambench/data.py` and nowhere near the second reader, which makes
  the same assumption silently.

## What was done

**1. The invariant first, and watched go red.** `tests/test_experiment_separation.py`
states dataset ownership rather than import direction, and scans *code* — it
strips docstrings and comments via AST, so the two packages can go on discussing
each other in prose while neither may open the other’s files. It named the four
offenders on its first run:

    fovbench/datasets_egosynth.py   ego-synth, egoexo4d, egosynth, inv_dist_std, nymeria, sparse_depth
    fovbench/run.py                 EGOSYNTH, ego-synth, egoexo4d, egosynth, inv_dist_std, nymeria, sparse_depth
    fovbench/split.py               EGOSYNTH, egosynth
    fovbench/report.py              ego-synth, egosynth

It lives at the repository root, not inside either package, so it is nobody’s to
quietly relax. It absorbed the three old one-sided tests, which are now
parametrised over both directions.

**2. The arm deleted.** `datasets_egosynth.py` and `tests/test_egosynth.py`;
`run_egosynth`, `_egosynth_clips`, `_egosynth_cells` and five CLI flags from
`run.py`; `EGOSYNTH_PROTOCOL` and `build_egosynth_split` from `split.py`; the
dataset branch and the sparse-GT header lines from `report.py`.

**3. The apparatus that existed only to serve it.** Each had exactly one consumer
and their docstrings said so — *“The ADT-FOV experiment does not use this;
datasets_egosynth does.”* Gone: `standardise_by_depth`, `anchored_ratios`,
`DEPTH_STRATA`, `MIN_STRATUM_PX`, `MIN_ANCHOR_SPREAD`, the `anchored_ratio`
entry in `METRIC_KEYS` and its computation in `bin_by`, and the `drift` column
in `report.summarise`. `_relative_spread` stays — `_gt_stats` uses it for
`gt_spread`, which the depth-confound table reports.

Four sites in `report.py` handled “a radial run with no theta axis”, which was
only ever ego-synth’s raw fisheye. The fallback is unreachable on ADT and is
gone; the surrounding docstrings now give the ADT reason for the code that
remains, rather than a reason that no longer exists.

## The number that matters

**Nothing the ADT experiment measures moved.** A fingerprint of the whole ADT
scoring path — both views, both protocols, the cross-frame reduction, the pooled
profiles, `summarise`, and the pure-geometry ring-coverage tables — was captured
before the first deletion and diffed after the last:

    keys removed   anchored_ratio, drift     (both NaN by construction on ADT)
    values moved   0

That is the whole claim of this ticket: 1 817 lines removed, and every published
ADT number is bit-identical. The fingerprint script is throwaway and is not in
the repo; it drove `full_frame_view -> _score_radial -> _reduce_radial ->
summarise` and `render_window -> _score_window -> _reduce_windows` on the
synthetic box scene `fovbench/tests/test_end_to_end.py` already uses, so it can
be rebuilt from that fixture in a few minutes if a future refactor wants the
same gate.

## Done when

- [x] `python -m pytest tests fovbench/tests slambench/tests -q` — 188 passed, 4 skipped
- [x] the ownership seam goes red on the pre-deletion tree and green after
- [x] the ADT fingerprint diff shows no moved value
- [x] `fovbench/README.md` and `slambench/__init__.py` state the new arrangement

## Not in scope

Candidates C and D of the architecture review, both of which want a design
conversation before any code moves:

* **C — one Aria lens, four descriptions.** `VGGT-360-fisheye/utils/fisheye_cam.py`,
  `finetune/eval/baselines/aria_fisheye.py` and `finetune/data/rectify.py` carry
  the same four calibration constants, with a comment instructing the reader to
  change them in two places; `rectify.py` rotates the principal point as
  `W - cy` where `fisheye_cam.py` uses `(H-1) - cy`, a one-pixel disagreement
  between the two arms of the FOV experiment (measured at 0.002° of incidence
  bias, which is why it survived). Contradicts `fisheye_cam.py`’s stated
  “self-contained subproject” boundary, so it needs that argued first.
* **D — the model-registry wrapper.** `fovbench/models.py` and
  `slambench/models.py` do the same registry job. **The two `split.py` files must
  stay duplicated**: each carries its own `PROTOCOL` token precisely so a digest
  from one experiment can never be mistaken for the other’s, and merging them
  would recreate the coupling this ticket removes.
