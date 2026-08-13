# Render ScanNet++ DSLR depth from the mesh, to unlock the one span-free target

**Owner:** gpu
**Status:** **open** — raytun3r. No run under `results/`.
**Files I may touch:** nothing under `raytun3r/` — a rendering job plus a written
answer. Output to the dataset tree on netapp; a note to `results`.
**Blocked by:** none, and independent of #10 — run it whenever there is idle GPU.

## Goal

`dslr/render_depth/` exists for `3f15a9266d`, holding dense metric depth for the
fisheye frames the loader already reads — or a written finding that it cannot be
produced and why.

## Why this is worth real work rather than a copy

You established that `render_depth` is genuinely absent (no `depth_file_path` on
any frame, no such directory on any scene you checked), and that the scene ships
`scans/mesh_aligned_0.05.ply` (35 MB) plus a 3-frame panoramic `panocam/depth`.
So this has to be rendered.

It is worth it because of what depth metrics are. Everything we have argued about
for three tickets — `R°` — is an **absolute angle** whose scale is set by the frame
span, which is exactly why a one-parameter stride fit could hit the paper's number
without meaning anything (`R = 0.42 + 0.170·I`, R² = 0.9984).

**`AbsRel` and `δ₁.₂₅` have no such dependence.** They are per-pixel comparisons
of predicted to ground-truth depth; how much the camera rotated between two frames
does not enter. So Tab. 3 left is the only target we have that the span degeneracy
cannot reach, and the only one that tests the backbone's health directly rather
than through the protocol question.

Tab. 3 left, DA3-Small, ScanNet++ (AbsRel ↓ / δ₁.₂₅ ↑):

| method | AbsRel / δ₁.₂₅ |
|---|---|
| Vanilla | 0.282 / 0.601 |
| Center-PH | **0.066 / 0.961** |
| RayTun3R | 0.108 / 0.886 |

Note this is a mean over unnamed scenes, so the absolute level is not matchable
from one scene — but the **structure** is a strong, paper-conceded claim:
Center-PH *wins depth outright on ScanNet++*, beating RayTun3R by a wide margin,
because it "produces perspective images close to the backbone's pretraining
distribution". If our Center-PH does not win depth here, that is a real finding
about our pipeline and it is visible without settling the protocol first.

## The job

Render dense depth for the DSLR frames of `3f15a9266d` from
`scans/mesh_aligned_0.05.ply`, using the poses and intrinsics already in
`dslr/nerfstudio/transforms.json`.

The official toolkit (`github.com/scannetpp/scannetpp`) has the renderer that
produced `render_depth` for the scenes that ship it — start there rather than
writing one.

**Three things that will make or break it:**

1. **It must render the fisheye model, not a pinhole.** The frames we evaluate on
   are `dslr/resized_images`, which are raw `OPENCV_FISHEYE` with `k1..k4` and
   146.3° horizontal FOV. Depth rendered through a pinhole would be silently
   misaligned everywhere off-axis — worst exactly where the fisheye argument
   lives.
2. **Record the depth convention explicitly** — planar z or euclidean range. This
   repo converts once at the boundary and every metric depends on the tag being
   right; a wrong tag is a per-pixel radial warp that no global scale absorbs.
   Write it in the note, do not leave it to be inferred.
3. **Match the naming the loader expects**: `dslr/render_depth/<image stem>.png`,
   uint16 millimetres (`depth_scale=1e-3`), 0 meaning invalid. If the toolkit
   emits something else, say so rather than converting silently.

## Sanity check before anyone trusts it

Cheap and catches convention and alignment errors at once: for a handful of
frames, back-project the rendered depth through the fisheye camera into 3D, then
into a *second* frame with the known relative pose, and compare against the second
frame's rendered depth. Agreement to a few centimetres means the intrinsics, the
poses and the convention are mutually consistent. Disagreement that grows with
radius means the render was pinhole, or the convention is planar-z where it was
read as range.

Report the number.

## Scope

**Only `3f15a9266d`.** It is the named Tab. 2/5 sequence and the one we have
staged locally, so it is the one that pays off. If the pipeline turns out to be
cheap, say so and we will decide about the rest — do not batch the dataset.

If the toolkit needs a GPU rasteriser built, or turns out to need assets the
download does not include, **stop and report that**. A written "this needs X,
which we do not have" is a good outcome; a hand-rolled renderer is not.

## Recording

A note in `results/scannetpp-render-depth/` with: what produced it, the depth
convention, the units, the sanity-check residual, frame count, and any frames
that failed. Do not commit depth maps — ScanNet++ is licensed per-recipient and
this repo is public.

## Done when

- [ ] `dslr/render_depth/` populated on netapp for `3f15a9266d`, or a written
      finding that it cannot be
- [ ] the sanity-check residual reported
- [ ] the depth convention and units stated explicitly
- [ ] note pushed to `results`; hand back to `cpu`

## Needs CPU-Claude afterwards?

yes — wiring `AbsRel`/`δ₁.₂₅` into the comparison against Tab. 3 left, and
deciding whether Center-PH wins depth here the way the paper says it does.
