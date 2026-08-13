# Audit Eq. 7's depth convention against what the VGGT head actually emits

**Owner:** cpu
**Status:** **not audited** — raytun3r, outside the FOV/SLAM pass of 2026-08-13. Check before picking up.
**Files I may touch:** `raytun3r/losses.py`, `raytun3r/tests/test_raytun3r.py`
**Blocked by:** none

## Goal

Decide, and encode in a test, whether `--convention range` (the default) is
self-consistent with the depth the backbone emits. Right now there is a plausible
mismatch that would distort every reconstructed point radially.

## Context

Eq. 7 is `X_i(u,v) = D_i(u,v)·κ⁻¹(u,v)` and does not say how `κ⁻¹` is normalised.
`raytun3r/README.md` interpretation decision 1 documents the choice and defaults
to `range` (unit ray), because the `z=1` ray diverges at 90° and is unusable on
the 185–200° datasets.

The tension: `backbones.py::Prediction` states — and CONTEXT.md agrees — that
"the VGGT-family depth heads emit planar z, not euclidean range". If `D` is planar
z and `κ⁻¹` is the unit ray, then `X` is wrong by a factor of `1/cos θ`. That is
1.74× at the Aria rim and **~10.8× at the ScanNet++ frame corner** (θ = 84.7°),
so it is not a rounding concern — it is a radial warp of the whole point cloud,
applied inside the reprojection loss (Eq. 8) and inside `d_reproj` (Eq. 16).

Note this cuts both ways and is genuinely undecided: on fisheye input VGGT is out
of distribution, so "what the head emits" may not be cleanly either quantity.

## Steps

1. Reason it through and write it up in the `losses.py::backproject` docstring:
   for `X = D·κ⁻¹` to be geometrically consistent, the normalisation of `κ⁻¹` has
   to match the definition of `D`. State which pairing the code uses and why.
2. Add a CPU test that pins the relationship rather than the choice: for a toy
   camera, `backproject(d, cam, convention="z")` and
   `backproject(d/cos θ, cam, convention="range")` must agree to float tolerance
   inside the cone. That makes any future silent switch visible.
3. Do **not** change the default on reasoning alone. If the analysis says `z` is
   the consistent reading, write the argument in the ticket and relabel `gpu` —
   the empirical check needs ADT ground-truth depth.

## Empirical check for GPU-Claude, if step 3 says so

`scratchpad/diag_convention.py` (attach it to the issue) scores VGGT's raw depth
against ADT ground truth read both ways, each with its own scale alignment.
Whichever fits better is what the head emits:

```bash
PYTHONPATH=$PWD python -u diag_convention.py \
  /user/f.zhang2/Documents/projectaria_tools_adt_data_clean/Apartment_release_clean_seq131_M1292 6
```

## Done when

- [ ] `python -m pytest raytun3r/tests -q` passes, including the new test
- [ ] `python raytun3r/smoke_test.py` passes
- [ ] `backproject` docstring states the consistency argument explicitly
- [ ] pushed to `organized`, issue commented with the sha

## Needs a GPU run afterwards?

maybe — only if the analysis concludes the default should change
