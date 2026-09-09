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

**What this unblocks, and what it does not.** The ticket says ADT
translation-direction numbers stay flagged approximate "until this JSON exists".
It exists — but `cam3r.adt.resolve_extrinsics` reports `exact=True/False` per
call, and whether any published number is still carrying the caveat has **not**
been re-checked here. That check is CPU-side, cheap, and is the natural
follow-up; it is deliberately not claimed as done.
