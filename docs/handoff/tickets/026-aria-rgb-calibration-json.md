# Export the Aria camera-rgb calibration to JSON (extrinsic + intrinsics)

**Owner:** gpu
**Status:** **DONE** — verified 2026-09-09 (CPU side, during the lambda_63
outage). The deliverable has existed since commit `4c38261`, whose message
credits **ticket 27** instead of this one, which is why this ticket still read
"not started". Ticket 27 (ADT hand-pixel stats) is a different task and remains
genuinely open.
**Files I may touch:** create `cam3r/data/adt_camera_rgb_calibration.json` only.
Results/artifacts to the `results` branch as usual; the JSON itself is small
enough to commit to `organized` via a comment on this issue if preferred.
**Blocked by:** none. Needs only `projectaria_tools` and any ADT sequence VRS.
**What is waiting on it:** the autoresearch H1.3 experiments
(`autoresearch/experiments/h1-rim-pose-value/`) currently recover the
device→camera rotation by a hand-eye bootstrap from classical poses (verified to
0.77–0.96°, `results/run_006.json`), which is good enough for rotation but has
no lever arm — so every translation-direction number on ADT stays flagged
"approximate" until this JSON exists. `cam3r/adt.py::resolve_extrinsics` already
consumes exactly this file format.

## The task

On the box, for one ADT sequence (seq131 preferred, any works — the factory
calibration is per-device and the box root has the VRS files):

```python
from projectaria_tools.core import data_provider
p = data_provider.create_vrs_data_provider(".../Apartment_release_clean_seq131_M1292/main_recording.vrs")
c = p.get_device_calibration().get_camera_calib("camera-rgb")
T = c.get_transform_device_camera()   # -> translation + quaternion
# also dump: projection params (KB4+ coefficients), image size they refer to
```

Write JSON with keys `T_device_camera: {translation: [x,y,z], quaternion_xyzw:
[x,y,z,w]}` plus a `projection` block with the raw parameter vector and the
calibration resolution (record it — the 2880 vs 1408 resolution trap is
documented in `docs/`, memory `aria-calibration-resolution-trap`).

## Acceptance

- Rotation part of `T_device_camera` should be ~38–43° from identity (the
  hand-eye bootstrap measured 40.55°; CPU side will cross-check and report).
- Post the JSON (or its path on `results`) as a comment on this issue.

---

## Verification, 2026-09-09

`cam3r/data/adt_camera_rgb_calibration.json` exists and meets every acceptance
criterion above:

| asked for | present |
|---|---|
| `T_device_camera.translation` | `[-0.004281, -0.011842, -0.005114]` |
| `T_device_camera.quaternion_xyzw` | `[0.32414, 0.040212, 0.041077, 0.944261]` |
| rotation ~38–43° from identity | **38.44°** — recomputed from the quaternion here, and matching the file's own `rotation_angle_from_identity_deg`. The hand-eye bootstrap measured 40.55°, so the two agree to 2.1°. |
| projection block with the raw parameter vector | `model_name: FISHEYE624`, `params`: 15 values |
| **the calibration resolution recorded** (the 2880 vs 1408 trap) | `image_size` present |
| source sequence | `Apartment_release_clean_seq131_M1292` — the ticket's preferred one |

**It has been in use all along**: the autoresearch roll line passes it as
`--calib` in H39, H41, H42 and H43 to read per-frame head roll from MPS, which
is what `roll_distribution.py` was already doing.

**What this unblocks, and what is still carrying the caveat.**
`resolve_extrinsics` returns `exact=True` only when an `extrinsics_json` is
passed; several benchmark call sites already pass this file
(`autoresearch/experiments/bench/code/raytun3r_row.py`, `raytun3r/train.py`,
`raytun3r/eval.py`, `cam3r/eval_adt.py`).

**But the experiment this ticket was written for does not.**
`autoresearch/experiments/h1-rim-pose-value/code/adt_pose_value.py` still
recovers the device→camera rotation with `hand_eye_rotation(...)`, and its own
failure path prints *"file the GPU ticket for the calibration JSON"* — the
ticket that has been satisfied since `4c38261`. So H1.3's ADT
translation-direction numbers are still in exactly the state described above,
for the reason that this ticket was never closed and nobody went back to wire
the file in.

**The rotation cross-check the acceptance asked CPU side to report:** the
hand-eye bootstrap measures **40.55°**, the factory calibration **38.44°** —
agreeing to **2.1°**, comfortably inside the 38–43° window. So the bootstrap was
sound for *rotation*. What it cannot supply, and what the JSON adds, is the
**lever arm**: `translation = [-0.004281, -0.011842, -0.005114]`, about **13 mm**
between the device origin and the RGB sensor. That is the part every
translation-direction number was missing.

**Deliberately not done here:** wiring the JSON into `adt_pose_value.py`.
Changing how a published experiment obtains its extrinsics changes its numbers,
and that needs a protocol and a re-run, not a quiet edit during an outage.
