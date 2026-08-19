# DRAFT GPU ticket (not filed — waiting for human to see day-1/day-2 reports)

Two small GPU-box items that unblock autoresearch hypotheses. Would become one
`gpu`-labelled GitHub issue per the handoff policy, with the usual "files I may
touch" list.

## 1. Export the Aria RGB extrinsic + intrinsics JSON (unblocks exact H1.3 GT)

On the box (projectaria_tools available), for seq131 (and ideally once per
device generation):

```python
from projectaria_tools.core import data_provider
p = data_provider.create_vrs_data_provider(".../main_recording.vrs")
c = p.get_device_calibration().get_camera_calib("camera-rgb")
# dump T_device_camera (translation + quaternion_xyzw) and the full KB4+
# projection params at native resolution to cam3r/data/adt_seq131_camera_rgb.json
```

`cam3r.adt.resolve_extrinsics` already consumes this via its `extrinsics_json`
argument. CPU-side H1.3 currently bootstraps the rotation part by hand-eye from
classical poses (gate-verified) — the JSON replaces a bootstrap with a
calibration, and adds the lever arm needed for translation-direction claims.

## 2. H4 measurement: hand/dynamic pixel share on a skeleton sequence

The local "clean" seq131 has zero hand/human instances (verified: all 357
instances static categories). H4 (dynamic-region error share) needs any ADT
sequence WITH skeleton from the box root
(`/group-volume/Fengjia/data/projectaria_tools_adt_data_clean`):

- Per frame: fraction of RGB-cone pixels that are hand/body instances
  (seg_npy + instances.json), their depth range, and their share of DA3/VGGT
  depth error (protocol to be locked CPU-side first — this item is only the
  data pull + a small stats script, no modeling).
- Deliverable: one JSON of per-frame stats on the `results` branch.
