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
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

from fovbench import _REPO  # noqa: F401  (import registers sys.path)

from finetune.eval.baselines import model_zoo as zoo  # noqa: E402

#: The benchmark's default line-up, in report order.
DEFAULT_MODELS: Tuple[str, ...] = ("vggt_1b", "vggt_omega", "dav2_large", "da3_large")

#: The stand-in that needs nothing installed.
ANALYTIC = "analytic"

#: Models that can be handed several frames in one forward pass. The three
#: multi-view backbones plus the stand-in; Depth-Anything V2 is monocular and
#: is deliberately absent, so asking for a context with it is refused up front
#: rather than silently scoring the target alone.
CONTEXT_CAPABLE: Tuple[str, ...] = ("vggt_1b", "vggt_omega", "da3_large",
                                    ANALYTIC)

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
    _predict: Optional[Callable] = field(default=None, repr=False)
    _predict_stack: Optional[Callable] = field(default=None, repr=False)

    def predict(self, rgb_u8: np.ndarray,
                gt_z: Optional[np.ndarray] = None,
                theta_deg: Optional[np.ndarray] = None) -> np.ndarray:
        """RGB uint8 ``(H, W, 3)`` -> planar-z depth ``(H, W)``, positive.

        ``gt_z``/``theta_deg`` are ignored by every real model; only the
        analytic stand-in reads them, so that a laptop run can verify the
        harness reports the bias it was given. Passing them costs nothing and
        keeps the driver free of a per-model special case.
        """
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
    """
    rng = np.random.default_rng(seed)

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
        return np.clip(d + rng.normal(0, 1e-3, d.shape), 1e-3, None).astype(np.float32)

    return predict


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_model(key: str, device, **kw) -> Model:
    """Load one model. ``device`` is a torch device (ignored by ``analytic``).

    Raises ``SystemExit`` with the registry's own instruction when weights or a
    dependency are missing — the same message ``--list`` prints, so a failure
    here is actionable without reading the code.
    """
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
        state, detail = zoo.status(zoo.get_specs([k])[0])
        (ready.append(k) if state == "ready" else skipped.append((k, state, detail)))
    return ready, skipped
