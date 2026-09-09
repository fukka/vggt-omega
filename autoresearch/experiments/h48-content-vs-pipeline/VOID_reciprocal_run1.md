# The reciprocal arm's first run is VOID — and how the diagnostic caught it

The 2×2 came back "neither cell large", one of the three outcomes
[`protocol.md`](protocol.md) named before the run. It was not read that way,
because two things about it did not fit:

- the four backbones agreed to three decimals in both arms
  (R: 0.0844 / 0.0830 / 0.0837 / 0.0835), where the native cells spread
  0.051 – 0.119;
- per-frame AbsRel barely moved from frame to frame.

Four independent models do not land on the same number. The scheduled
diagnostic ([`code/diag_h48.py`](code/diag_h48.py), committed before it ran)
asked whether either arm still had **resolving power**: align a *constant*
prediction — a fronto-parallel plane that knows nothing about the scene — with
the same `scale_shift` fit, and see what AbsRel it scores. That is the score a
model gets for knowing nothing. A cell whose models sit at that floor cannot
show an effect of any size, and its zero says nothing about the mirror effect.

## What the diagnostic found instead

| cell | content | lens | n scored px | GT depth p5/p50/p95 | spread | know-nothing floor |
|---|---|---|---|---|---|---|
| `aa` | Aria | Aria | 41 369 | 1.14 / 2.57 / 4.42 m | 1.270 | 0.4888 |
| `r`  | Aria | ScanNet++ | 23 254 | **2.18 / 2.67 / 3.23 m** | **0.398** | 0.0950 |
| `a`  | ScanNet++ | Aria | 34 084 | 1.13 / 1.44 / 1.66 m | 0.371 | 0.1010 |
| `ss` | ScanNet++ | ScanNet++ | 20 654 | 1.13 / 1.43 / 1.66 m | 0.373 | 0.1005 |

Rows `aa` and `r` are the *same ADT sequence* scored over the *same* θ ≤ 28°
cone. A change of lens re-parameterises which pixel holds which ray; it cannot
flatten a depth distribution threefold. The R arm was not measuring Aria
content.

## Cause

`AriaRemap`'s `map_x` / `map_y` are **pixel coordinates in the source camera's
own grid**. The Aria camera of record is built at `--size` (504). ADT's
`depth_npy` is on the sensor's native 1408 grid. `cv2.remap` does not object to
the mismatch — it samples the sub-rectangle those coordinates happen to land
in. So the arm warped the sensor's **top-left 504×504 corner** and scored every
model against it, while showing the model the whole frame. Ground truth and
image came from different parts of the scene.

The images were never affected: they arrive at 504 already. Only the depth
path, and only in this arm — the forward arm's camera is 504×336 and its depth
arrives at 336×504.

That also explains the symptom that raised the alarm. With ground truth
unrelated to the image, no model can do better than the affine fit through the
mask's mean, so all four land together just above the floor: models
0.0830 – 0.0844 against a floor of 0.0950 — 12% of headroom, i.e. none.

## Fixes

1. `AriaRemap` now records the source grid and **refuses** an array of any
   other shape, in `image()` as well as `depth()`. A silent wrong number
   becomes a crash. (`autoresearch/data/scannetpp_aria.py`)
2. `to_grid()` in the same module: nearest-neighbour, never interpolation —
   averaging across a depth discontinuity invents a surface that is in neither
   frame.
3. The reciprocal arm resamples `depth_npy` onto the camera's grid first.

## Verification the fix is a fix, not a patch

Re-running the diagnostic on cell `r` after the fix:

    r  seq138: 20 frames  1.15 / 2.58 / 4.40 m  spread 1.252  floor 0.4836

against native `aa`'s 1.14 / 2.57 / 4.42, spread 1.270, floor 0.4888. This is
the check the arm should always have carried: **a pure lens re-parameterisation
of the same content must preserve the depth distribution over the same cone**,
because it moves rays between pixels and does not touch what the rays hit. It
now does, to two decimals.

## The floor table is a result in its own right

Read the last column by rows and columns:

- by **content row**: Aria 0.4888 / 0.4836, ScanNet++ 0.1010 / 0.1005;
- by **lens column**: Aria 0.4888 / 0.1010, ScanNet++ 0.4836 / 0.1005.

The know-nothing floor is a property of the scene's depth distribution, and it
travels through *either* lens intact. That is independent evidence that the
resampling carries content rather than destroying it — the thing the protocol's
third outcome would have required to be false. It was measured without running
a single model.

It also sets a limitation on the forward arm that must be stated with its
result: ScanNet++'s central 28° cone spans only 1.13 – 1.66 m. It is nearly a
fronto-parallel patch, and a floor of 0.101 against models at 0.035 – 0.039
leaves 2.6–2.9× of headroom, against 4–9× in the Aria cells. The forward arm
can resolve an effect, but it is the weaker half of the 2×2.
