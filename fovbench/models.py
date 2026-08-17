# Copyright (c) 2026.
"""The models under test, behind one call.

Four vanilla, off-the-shelf networks — no fine-tuning, no adapters, no fisheye
awareness of any kind:

    ``vggt_1b``      VGGT-1B (DINOv2 + RoPE), the vendored ``vggt_visfeat``
    ``vggt_omega``   VGGT-Omega 1B/512 (DINOv3, RoPE only), local checkpoint
    ``dav2_large``   Depth-Anything V2 Large (relative disparity)
    ``da3_large``    Depth-Anything 3 Large

They are already registered, loaded and profiled by
:mod:`finetune.eval.baselines.model_zoo`; this module adds only what the FOV
experiment needs on top:

* **native render size** — each view is rendered at the model's own token grid
  so nothing is resampled between the view construction and the network. A
  resize would change the view's effective field of view, and field of view is
  the variable under study. VGGT and DA3 take 518 (patch 14, 37x37 tokens),
  VGGT-Omega 512 (patch 16). The 1.2% difference in sampling density is the
  honest cost of not resampling either model's input; it is not a difference in
  what the view *covers*.
* **an analytic stand-in** — ``--models analytic`` runs the entire harness with
  no weights, no downloads and no GPU. It is how the pipeline is exercised on a
  laptop, and, given GT, it can inject a *known* radial bias so the reported
  eccentricity curve can be checked against the answer that was put in.

Alignment mode travels with the model, from the registry: disparity-space affine
for Depth-Anything V2 (the MiDaS protocol), depth-space affine for the
up-to-scale depth heads. Mixing those up is worth more than any effect here.

The one model that is not vanilla
---------------------------------
``vggt360`` is this repository's own VGGT-360-fisheye port, and it breaks the
sentence at the top of this file on purpose: it is *given* the lens. It takes the
raw fisheye frame, splits it into a centre view plus an 8-direction ring of 60 deg
tangent views, reconstructs all nine in one VGGT pass, and fuses the result back
onto the fisheye grid (``utils/pipeline.py``, shared with ``slambench``). It is
here so the FOV curve of a lens-aware method can be read on the same axes as the
four that are not — which is the only comparison that says whether the port buys
anything at the rim.

Three things about it are unlike every other entry, and each is enforced rather
than documented and hoped for:

* **It answers on the raw fisheye only.** The pipeline consumes a fisheye frame
  and a camera model. A rectified pinhole is not one, and a 40 deg ``window`` is
  not one either — the nine-view layout tiles a 55 deg cone, so aimed at a crop
  it would be eight views of nothing. ``run.py`` refuses those combinations
  before any weights load rather than producing a degenerate column.
* **It needs the whole view, not just its pixels** — the lens describing the
  frame, and the native frame that frame was resampled from. See
  ``Model.predict``'s ``view`` and ``geometry.FrameView.source_rgb``.
* **It reads at 1408 and answers at 518.** The nine tangent views are cut from
  ADT's own frame, which is the resolution the port is designed for and what
  ``main_adt.py`` uses; the fused answer is then delivered on the harness's
  518 scoring grid, so every metric, mask and eccentricity bin is exactly the
  one the other four models are scored under. Only the sharpness of what the
  network saw differs — and it differs by a lot: a 60 deg view at 518 px is a
  0.62x downsample of the 1408 frame and a 1.69x *up*sample of the 518 one.

  This makes the row **not** resolution-matched to the vanilla four, and that is
  the deliberate choice: matching it (``--vggt360-source view``) measures a
  starved version of the method — nine crops carrying 9x the pixels of a vanilla
  input and none of the extra detail — and reports it as the method. Neither
  arm is neutral. Quote either with its sentence attached; ``GPU_EXPERIMENTS.md``
  0b carries both.
"""
from __future__ import annotations

import os
import sys
import zlib
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

from fovbench import _REPO  # noqa: F401  (import registers sys.path)

from finetune.eval.baselines import model_zoo as zoo  # noqa: E402

#: The benchmark's default line-up, in report order. ``vggt360`` is deliberately
#: **not** here: the default grid is the four vanilla models, and adding a
#: lens-aware fifth to it would silently change what every published command in
#: ``GPU_EXPERIMENTS.md`` measures. Ask for it by name.
DEFAULT_MODELS: Tuple[str, ...] = ("vggt_1b", "vggt_omega", "dav2_large", "da3_large")

#: The stand-in that needs nothing installed.
ANALYTIC = "analytic"

#: This repository's VGGT-360-fisheye port. See the module docstring.
VGGT360 = "vggt360"

#: The backbone ``vggt360`` runs on. Its availability *is* the port's
#: availability, so the registry is asked about that key rather than a second
#: description of the same weights being written here.
VGGT360_BACKBONE = "vggt_1b"

#: Models that can be handed several frames in one forward pass. The three
#: multi-view backbones plus the stand-in; Depth-Anything V2 is monocular and
#: is deliberately absent, so asking for a context with it is refused up front
#: rather than silently scoring the target alone.
CONTEXT_CAPABLE: Tuple[str, ...] = ("vggt_1b", "vggt_omega", "da3_large",
                                    "da3_small", ANALYTIC)

#: Input side length, by model key. Everything in the DINOv2/DINOv3 depth family
#: here is patch-14 and ships at 518; VGGT-Omega is patch-16 and ships at 512.
#: Keys only — an earlier version also fell back to ``spec.kind``, which happened
#: to work solely because "vggt_omega" is both a key and a kind.
_NATIVE_SIZE = {"vggt_omega": 512}
_DEFAULT_NATIVE_SIZE = 518


def native_size(key: str) -> int:
    """Render size for ``key``'s views — its own token grid (see module docs)."""
    return _NATIVE_SIZE.get(key, _DEFAULT_NATIVE_SIZE)


@dataclass
class Model:
    """A loaded model plus what the report needs to describe it."""

    key: str
    family: str
    size: str
    align_mode: str
    input_size: int
    params_m: float = float("nan")
    supports_context: bool = False
    #: True for a model that builds its own views out of the frame and so needs
    #: the whole :class:`~fovbench.geometry.FrameView` — its lens, and the
    #: native frame it was resampled from — rather than just the pixels. Only
    #: ``vggt360``; every off-the-shelf network here is handed pixels and given
    #: no camera at all, which is the premise of the whole experiment.
    needs_view: bool = False
    #: Views this model can answer on at all. ``None`` means "any"; ``vggt360``
    #: names ``("fisheye",)`` and ``run.py`` refuses the rest up front.
    views: Optional[Tuple[str, ...]] = None
    _predict: Optional[Callable] = field(default=None, repr=False)
    _predict_stack: Optional[Callable] = field(default=None, repr=False)

    def predict(self, rgb_u8: np.ndarray,
                gt_z: Optional[np.ndarray] = None,
                theta_deg: Optional[np.ndarray] = None,
                view=None) -> np.ndarray:
        """RGB uint8 ``(H, W, 3)`` -> planar-z depth ``(H, W)``, positive.

        ``gt_z``/``theta_deg`` are ignored by every real model; only the
        analytic stand-in reads them, so that a laptop run can verify the
        harness reports the bias it was given. Passing them costs nothing and
        keeps the driver free of a per-model special case.

        ``view`` is the whole :class:`~fovbench.geometry.FrameView` — its own
        camera, and the native frame it was resampled from. Every off-the-shelf
        model ignores it, which is the point of this benchmark; ``vggt360``
        cannot work without it, because it re-renders the frame into nine
        tangent views and a camera half a pixel out puts each of them a third of
        a degree off the axis they are binned by.
        """
        if self.needs_view:
            if view is None or view.cam is None:
                raise SystemExit(
                    f"[fovbench] {self.key} was called without the view's "
                    f"camera. It renders its own views out of the frame, so a "
                    f"missing lens is a wrong geometry, not a missing option.")
            return self._predict(view)
        return self._predict(rgb_u8, gt_z, theta_deg)

    def predict_stack(self, rgb_u8_list, target: int,
                      gt_z: Optional[np.ndarray] = None,
                      theta_deg: Optional[np.ndarray] = None) -> np.ndarray:
        """Depth for ONE frame of a stack the model saw in a single pass.

        ``target`` indexes the frame that is scored; the rest are context. A
        model without a multi-frame path raises rather than silently scoring the
        target alone — a run that asked for 10 frames and quietly got 1 would
        read as "context does not help", which is the opposite of nothing.
        """
        if len(rgb_u8_list) == 1:
            return self.predict(rgb_u8_list[0], gt_z, theta_deg)
        if self._predict_stack is None:
            raise SystemExit(
                f"[fovbench] {self.key} has no multi-frame path: it is monocular. "
                f"Run it with --context-frames 1, or drop it from a context run.")
        return self._predict_stack(rgb_u8_list, target)


# --------------------------------------------------------------------------- #
# Analytic stand-in
# --------------------------------------------------------------------------- #

def _analytic_predict(radial_bias: float, seed: int) -> Callable:
    """Depth that is GT bent by a known function of eccentricity.

    ``pred = gt * (1 + radial_bias * theta_rad^2)`` — the same shape as the
    error a pinhole-trained network makes on a wide lens (it reads the periphery
    as if the rays were less inclined than they are), which makes it a fair
    exercise of the whole scoring path. Without GT it falls back to a smooth
    function of image intensity, so the harness can still be driven end to end
    on views whose GT is absent.

    The jitter is seeded **from the frame's own content**, not from a generator
    that advances call by call. A shared stream would make this model's output a
    function of the order frames happened to be scored in — so the same frame
    would score differently under ``--workers 8`` than under ``--workers 1``,
    and differently again if a run covered a different number of frames. That is
    not a property any model should have, least of all the one the tests use as
    a fixed point.
    """
    def _jitter(d: np.ndarray) -> np.ndarray:
        # A cheap stride-sampled checksum: distinct per frame, identical for the
        # same frame every time, independent of when it is asked for.
        tag = zlib.crc32(np.ascontiguousarray(d[::16, ::16]).tobytes())
        return np.random.default_rng([seed, tag]).normal(0, 1e-3, d.shape)

    def predict(rgb_u8, gt_z=None, theta_deg=None):
        if gt_z is None:
            g = rgb_u8.astype(np.float32).mean(-1) / 255.0
            return (1.0 + 4.0 * g).astype(np.float32)
        d = np.asarray(gt_z, np.float32).copy()
        if theta_deg is not None and radial_bias:
            d = d * (1.0 + radial_bias * np.radians(theta_deg, dtype=np.float32) ** 2)
        # A GT-shaped hole is not a prediction; fill it so the model "predicts"
        # everywhere, as a real one does, and let the mask do the excluding.
        fill = float(np.median(d[d > 0])) if (d > 0).any() else 1.0
        d[d <= 0] = fill
        return np.clip(d + _jitter(d), 1e-3, None).astype(np.float32)

    return predict


# --------------------------------------------------------------------------- #
# VGGT-360-fisheye — the lens-aware entry
# --------------------------------------------------------------------------- #

def _vggt360_predict(pipe, key: str, source: str = "native") -> Callable:
    """The port, wrapped to look like any other ``(rgb) -> planar z`` model.

    ``source`` decides which pixels the nine tangent views are cut from, and it
    is the single most consequential setting on this row:

    ``native``  ADT's own 1408 frame — the resolution the port is designed for
                and the one ``main_adt.py`` uses. A 60 deg view at 518 px is
                then a **0.62x downsample** of the source, which is what
                ``crop_supersample`` exists to filter. The answer is still fused
                onto the harness's 518 scoring grid, so every metric, mask and
                eccentricity bin is unchanged; only the sharpness of what the
                network saw differs.
    ``view``    the 518 px frame the other four models are handed. Strictly
                resolution-matched, and it starves the method: the same 60 deg
                view is then a **1.69x upsample**, so the nine crops carry 9x
                the pixels of a vanilla input and none of the extra detail.

    ``native`` is the default because the alternative measures a handicapped
    version of the method and reports it as the method. Neither is neutral, and
    a claim from either needs its sentence — see ``GPU_EXPERIMENTS.md`` 0b.

    Three conversions happen here and nowhere else, because each of them is
    about the *harness* rather than about the method:

    1. **range -> planar z.** Fusion produces euclidean range along each ray;
       ``FrameView.gt_z`` is planar z about the camera axis, and the difference
       is ``cos(theta)`` — 1.00 on axis, about 1.74x at 55 deg. It is radial, so
       no scale-and-shift can absorb it, and scoring without it is the exact
       error CONTEXT.md records this port having made once. This is
       ``main_adt.py``'s default ``--eval-domains z``, the dataset's own
       convention, so the number is comparable with that driver's z rows.
    2. **the cone's corners.** The pipeline answers for the imaged cone; the
       frame is square. ``fovbench`` masks its own corners out anyway, but it
       fits the alignment affine over its mask before it does, and one NaN in
       there takes the whole frame to NaN. The holes get a constant.
    3. **nothing else.** No rescaling, no clipping, no per-bin anything.
    """
    from utils.pipeline import fill_uncovered, range_to_planar_z

    from fovbench.geometry import fisheye_rays

    def predict(view):
        # Cut the views from here; deliver the answer on ``view.cam``'s grid.
        if source == "native" and view.source_rgb is not None:
            src_rgb, src_cam = view.source_rgb, view.source_cam
        else:
            src_rgb, src_cam = view.rgb, view.cam

        # ``fisheye_rays`` is this package's memoised LUT and is the *same*
        # object the harness binned this frame's theta from, so the conversion
        # below and the eccentricity axis cannot describe different lenses.
        rays, cone = fisheye_rays(view.cam)
        ff = pipe.range_map(np.ascontiguousarray(src_rgb), src_cam,
                            out_lens=view.cam, out_ray_lut=(rays, cone))
        z = range_to_planar_z(ff.range_map, rays[..., 2])
        filled, n_hole = fill_uncovered(z, ff.covered, where=ff.cone)
        if n_hole:
            _report_holes(key, n_hole, int(ff.cone.sum()))
        return filled

    return predict


#: Worst hole fraction seen, per model key — printed once at the end rather than
#: per frame. A pipeline that stopped covering its cone would otherwise be
#: visible only as a slightly worse score.
_HOLE_WORST: dict = {}


def _report_holes(key: str, n_hole: int, n_cone: int) -> None:
    frac = n_hole / max(n_cone, 1)
    if frac > _HOLE_WORST.get(key, 0.0):
        _HOLE_WORST[key] = frac
        if frac > 0.01:
            print(f"[fovbench]   {key}: {frac:.2%} of the imaged cone had no "
                  f"view covering it and was filled with a constant "
                  f"(expect <1%; a large value means the ring no longer tiles "
                  f"its cone)")


def load_vggt360(device, **kw) -> Model:
    """Build the VGGT-360-fisheye pipeline as a ``Model``.

    Its ``align_mode`` is ``scale_shift``, the same depth-space affine the three
    up-to-scale depth heads are scored under, so its level is directly readable
    against VGGT-1B, VGGT-Omega and DA3 — and, as everywhere in this package,
    not against DAv2.
    """
    from utils.pipeline import VGGT360Config, VGGT360Pipeline

    state, detail = zoo.status(zoo.get_specs([VGGT360_BACKBONE])[0])
    if state != "ready":
        raise SystemExit(f"[fovbench] {VGGT360} needs the {VGGT360_BACKBONE} "
                         f"weights: {state} — {detail}")

    cfg = VGGT360Config(
        fov=kw.get("v360_fov", 60.0),
        ring_tilt=kw.get("v360_ring_tilt", 26.0),
        n_ring=kw.get("v360_n_ring", 8),
        persp_size=kw.get("v360_persp_size", 518),
        adaptive=kw.get("v360_adaptive", True),
        sa_mask=kw.get("v360_sa_mask", True),
        fuse=kw.get("v360_fuse", "attn"),
        head=kw.get("v360_head", "depth"),
        dtype=kw.get("v360_dtype", "bf16"),
        model_path=kw.get("v360_model_path", "facebook/VGGT-1B"),
    ).check()
    source = kw.get("v360_source", "native")
    if source not in ("native", "view"):
        raise SystemExit(f"[fovbench] vggt360 source must be 'native' or "
                         f"'view', got {source!r}")
    pipe = VGGT360Pipeline(cfg, device=str(getattr(device, "type", device))).load()

    return Model(key=VGGT360, family="vggt360", size=cfg.describe(),
                 align_mode="scale_shift", input_size=native_size(VGGT360),
                 params_m=pipe.num_params / 1e6,
                 # One frame in, nine views out. A *temporal* context would be
                 # nine views per frame in one pass, which is a different
                 # experiment and not one this wrapper silently performs.
                 supports_context=False,
                 needs_view=True, views=("fisheye",),
                 _predict=_vggt360_predict(pipe, VGGT360, source))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_model(key: str, device, **kw) -> Model:
    """Load one model. ``device`` is a torch device (ignored by ``analytic``).

    Raises ``SystemExit`` with the registry's own instruction when weights or a
    dependency are missing — the same message ``--list`` prints, so a failure
    here is actionable without reading the code.
    """
    if key == VGGT360:
        return load_vggt360(device, **kw)
    if key == ANALYTIC:
        fn = _analytic_predict(kw.get("radial_bias", 0.0), kw.get("seed", 0))
        return Model(key=key, family="analytic", size="—",
                     align_mode=kw.get("align_mode", "scale_shift"),
                     input_size=native_size(key), params_m=0.0,
                     supports_context=True, _predict=fn,
                     # The stand-in reads GT, so context cannot change its
                     # answer; it accepts a stack only so the driver's
                     # multi-frame path can be exercised without weights.
                     _predict_stack=lambda imgs, tgt: fn(imgs[tgt], None, None))

    spec = zoo.get_specs([key])[0]
    state, detail = zoo.status(spec)
    if state != "ready":
        raise SystemExit(f"[fovbench] {key}: {state} — {detail}")

    adapter = zoo.build_adapter(spec, **{k: v for k, v in kw.items()
                                         if k in ("checkpoint", "use_camera")})
    adapter.load(device)
    cam = None                       # every adapter used here ignores it

    def predict(rgb_u8, gt_z=None, theta_deg=None):
        rgb01 = np.clip(rgb_u8.astype(np.float32) / 255.0, 0.0, 1.0)
        return np.asarray(adapter.predict_frame(rgb01, cam, "view"), np.float32)

    def predict_stack(rgb_u8_list, target):
        imgs = [np.clip(r.astype(np.float32) / 255.0, 0.0, 1.0)
                for r in rgb_u8_list]
        return np.asarray(adapter.predict_frames(imgs, target), np.float32)

    ctx = bool(getattr(adapter, "supports_context", False))
    return Model(key=key, family=spec.family, size=spec.size,
                 align_mode=spec.align_native[0], input_size=native_size(key),
                 params_m=adapter.num_params() / 1e6, supports_context=ctx,
                 _predict=predict,
                 _predict_stack=predict_stack if ctx else None)


def available(keys) -> Tuple[list, list]:
    """Split ``keys`` into (runnable, [(key, state, detail), ...] skipped)."""
    ready, skipped = [], []
    for k in keys:
        if k == ANALYTIC:
            ready.append(k)
            continue
        # ``vggt360`` is a pipeline around VGGT-1B, so it is available exactly
        # when that backbone is — asked of the registry, not restated here.
        probe = VGGT360_BACKBONE if k == VGGT360 else k
        state, detail = zoo.status(zoo.get_specs([probe])[0])
        if state == "ready":
            ready.append(k)
        else:
            skipped.append((k, state, detail))
    return ready, skipped


def restricted_views(key: str) -> Optional[Tuple[str, ...]]:
    """Views ``key`` can answer on, **without loading anything**.

    ``run.py`` calls this before the split is built and before any weight is
    touched, so a run that asked ``vggt360`` for the rectified arm fails in a
    second rather than after a model load and a decode.
    """
    return ("fisheye",) if key == VGGT360 else None
