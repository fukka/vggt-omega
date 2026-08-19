# H8 — equal-solid-angle tokenization (probe protocol, locked before any run)

Hypothesis (state H8): uniform-pixel patches on KB4 give the center 1.73x
the solid angle of a rim patch (measured, src/solid_angle_probe.py). An
equal-solid-angle input remap (a) may flatten the radial error field at
source, and (b) enables ~35-40% token reduction at matched center density.

## Probe A (this protocol): zero-training remap test
- Remap camera: equisolid-angle projection r = 2 f sin(theta/2), which has
  exactly uniform solid angle per pixel area. Implemented as KannalaBrandt
  with the Taylor coefficients k = (-1/24, 1/1920, -1/322560, 0) — max
  deviation from 2 sin(theta/2) is <1e-6 normalized units over
  theta <= 54.7 deg.
- Same 504x504 canvas, f chosen so theta_max lands at the half-width
  (f = 251.5 / (2 sin(theta_max/2))). Bilinear resample from the fisheye
  frame via exact KB4 projection (same machinery as the Center-PH row).
- Run frozen DA3-S on the remapped image; map predictions back to the
  FISHEYE grid; identical joint-table eval; control arm = plain fisheye
  through the same script (same frames, same alignment, same masks).
- Data: local seq131, all 28 frames, 504px.

## Predictions (locked)
- P1: if center oversampling contributes to the radial gradient, the
  near-rim rows' rise with theta flattens (near-rim AbsRel drops >=15%
  vs the plain-fisheye arm) with center change within +-10%.
- P2 (null outcome): if the compression is a scale-prior/content effect
  (like Center-PH's near-field failure suggests), the field barely moves
  (<10% everywhere) — H8's efficiency claim (Probe B, smaller canvas)
  can still proceed, but the "fixes bias at source" claim dies.

## Decision rule
P1 -> Probe B (392px equisolid canvas = 0.6x tokens at matched center
density) + consider H8 as a method component. P2 -> record, keep only the
efficiency angle if Probe B shows error parity at 0.6x tokens.
