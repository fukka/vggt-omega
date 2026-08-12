# Copyright (c) 2026.
"""The models under test, behind one call.

Five vanilla, off-the-shelf networks — no fine-tuning, no adapters, no fisheye
awareness of any kind:

    ``vggt_1b``      VGGT-1B                     up-to-scale depth
    ``vggt_omega``   VGGT-Omega 1B/512           up-to-scale depth
    ``dav2_large``   Depth-Anything V2 Large     relative **disparity**
    ``da3_small``    Depth-Anything 3 Small      up-to-scale depth
    ``da3_large``    Depth-Anything 3 Large      up-to-scale depth

They are registered, loaded and profiled by
``finetune/eval/baselines/model_zoo.py``, which belongs to neither experiment and
is shared. This module adds only what the SLAM evaluation needs on top: the
model's native input size, and its alignment mode carried from the registry
rather than restated (see ``metrics.check_protocol``).

Two stand-ins, and what each is honest about
--------------------------------------------
Neither needs weights, a GPU or a download, and they exercise different halves of
the harness on purpose:

``analytic``  returns a **dense depth map** built from image intensity. Its score
              is meaningless, and that is fine — what it exercises is the whole
              path *including the gather*: decode, feed, predict, sample at the
              point list, align, score, report. It is the end-to-end smoke test.

``oracle``    returns **per-point** depth derived from the ground truth with a
              known affine and a known error. Its score is the thing being
              checked: an oracle with no injected error must read AbsRel 0, and
              one given a known bias must read it back. It exercises the metric
              path and skips the gather entirely.

Keeping them apart is deliberate. A single stand-in that reads ground truth *and*
answers per-point would leave the gather — the step the data card warns hardest
about — untested by every run that used it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

from slambench import _REPO  # noqa: F401  (import registers sys.path)

from finetune.eval.baselines import model_zoo as zoo  # noqa: E402

#: The line-up, in report order.
DEFAULT_MODELS: Tuple[str, ...] = ("vggt_1b", "vggt_omega", "dav2_large",
                                   "da3_small", "da3_large")

#: Weight-free stand-ins. See the module docstring for what each proves.
ANALYTIC = "analytic"
ORACLE = "oracle"
STANDINS = (ANALYTIC, ORACLE)

#: Input side length by model key. The DINOv2/DINOv3 depth family here is
#: patch-14 and ships at 518; VGGT-Omega is patch-16 and ships at 512.
_NATIVE_SIZE = {"vggt_omega": 512}
_DEFAULT_NATIVE_SIZE = 518


def native_size(key: str) -> int:
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
    _predict: Optional[Callable] = field(default=None, repr=False)

    def predict(self, rgb_u8: np.ndarray,
                gt: Optional[np.ndarray] = None) -> np.ndarray:
        """RGB uint8 ``(H, W, 3)`` -> depth.

        A real model returns a dense ``(H, W)`` map and never looks at ``gt``.
        The ``oracle`` stand-in returns a flat ``(N,)`` array instead, one entry
        per ground-truth point; the caller dispatches on the shape. Passing
        ``gt`` costs nothing and keeps the baselines free of a per-model case.
        """
        return self._predict(rgb_u8, gt)


# --------------------------------------------------------------------------- #
# Stand-ins
# --------------------------------------------------------------------------- #

def _analytic_predict(seed: int) -> Callable:
    """A dense map from image intensity — plausible in shape, wrong in value."""
    rng = np.random.default_rng(seed)

    def predict(rgb_u8, gt=None):
        g = np.asarray(rgb_u8, np.float32).mean(-1) / 255.0
        d = 1.0 + 4.0 * g
        return np.clip(d + rng.normal(0, 1e-3, d.shape), 1e-3, None).astype(np.float32)

    return predict


def _oracle_predict(bias: float, noise: float, scale: float, shift: float,
                    seed: int) -> Callable:
    """Ground truth, put through a known distortion, per point.

    ``pred = gt * (1 + bias) / scale + shift``. The ``scale``/``shift`` exist so
    that the oracle is *not* already metric: an evaluation which only ever sees a
    metric prediction never exercises the alignment it is supposed to apply, and
    would pass just as well with the alignment deleted.

    ``shift`` is **added**, not subtracted, and that is not cosmetic. A depth
    head's output is positive everywhere; a shift large enough to drive it
    negative would be clamped here, and the clamp — not the model — would then be
    what the metrics saw. Subtracting 0.4 from ``gt / 3`` on a scene starting at
    0.8 m does exactly that to the near third of the points.
    """
    rng = np.random.default_rng(seed)

    def predict(rgb_u8, gt=None):
        if gt is None:
            raise SystemExit("[slambench] the 'oracle' stand-in scores against "
                             "ground truth and cannot run without it")
        g = np.asarray(gt, np.float64)
        d = g * (1.0 + bias)
        if noise:
            d = d + rng.normal(0.0, noise, d.shape)
        out = (d / scale) + shift
        if np.any(out <= 0):
            raise SystemExit(
                "[slambench] the oracle's scale/shift drive some predictions "
                "non-positive; the clamp, not the model, would be what the "
                "metrics saw. Use a positive shift.")
        return out.astype(np.float32)

    return predict


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_model(key: str, device, **kw) -> Model:
    """Load one model. ``device`` is a torch device, ignored by the stand-ins.

    Raises ``SystemExit`` carrying the registry's own instruction when weights or
    a dependency are missing — the same message ``--list`` prints, so a failure
    is actionable without reading code.
    """
    if key == ANALYTIC:
        return Model(key=key, family="analytic", size="—",
                     align_mode=kw.get("align_mode", "scale_shift"),
                     input_size=kw.get("input_size") or _DEFAULT_NATIVE_SIZE,
                     params_m=0.0, _predict=_analytic_predict(kw.get("seed", 0)))
    if key == ORACLE:
        return Model(key=key, family="oracle", size="—",
                     align_mode=kw.get("align_mode", "scale_shift"),
                     input_size=kw.get("input_size") or _DEFAULT_NATIVE_SIZE,
                     params_m=0.0,
                     _predict=_oracle_predict(kw.get("bias", 0.0),
                                              kw.get("noise", 0.0),
                                              kw.get("scale", 3.0),
                                              kw.get("shift", 0.4),
                                              kw.get("seed", 0)))

    spec = zoo.get_specs([key])[0]
    state, detail = zoo.status(spec)
    if state != "ready":
        raise SystemExit(f"[slambench] {key}: {state} — {detail}")

    adapter = zoo.build_adapter(spec, **{k: v for k, v in kw.items()
                                         if k in ("checkpoint", "use_camera")})
    adapter.load(device)

    def predict(rgb_u8, gt=None):
        rgb01 = np.clip(np.asarray(rgb_u8, np.float32) / 255.0, 0.0, 1.0)
        return np.asarray(adapter.predict_frame(rgb01, None, "view"), np.float32)

    return Model(key=key, family=spec.family, size=spec.size,
                 align_mode=spec.align_native[0], input_size=native_size(key),
                 params_m=adapter.num_params() / 1e6, _predict=predict)


def available(keys) -> Tuple[list, list]:
    """Split ``keys`` into (runnable, [(key, state, detail), ...] skipped)."""
    ready, skipped = [], []
    for k in keys:
        if k in STANDINS:
            ready.append(k)
            continue
        state, detail = zoo.status(zoo.get_specs([k])[0])
        (ready.append(k) if state == "ready" else skipped.append((k, state, detail)))
    return ready, skipped
