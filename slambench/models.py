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

One frame or several
--------------------
Three of the five — ``vggt_1b``, ``vggt_omega``, ``da3_small``/``da3_large`` —
are **multi-view** models that this evaluation has always run one frame at a
time. :meth:`Model.predict_stack` is the other path: a window of frames in one
forward pass, of which exactly one is scored. ``dav2_large`` is monocular and
says so by raising, rather than scoring the target alone and reporting a table
that reads "context does not help" when nothing was tried.

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
    #: Can this model be handed several frames in **one forward pass** and use
    #: them jointly? True for the multi-view backbones (VGGT, VGGT-Omega, DA3),
    #: false for the monocular ones (Depth-Anything V2) and for the stand-ins.
    supports_context: bool = False
    _predict: Optional[Callable] = field(default=None, repr=False)
    _predict_stack: Optional[Callable] = field(default=None, repr=False)

    def predict(self, rgb_u8: np.ndarray,
                gt: Optional[np.ndarray] = None) -> np.ndarray:
        """RGB uint8 ``(H, W, 3)`` -> depth.

        A real model returns a dense ``(H, W)`` map and never looks at ``gt``.
        The ``oracle`` stand-in returns a flat ``(N,)`` array instead, one entry
        per ground-truth point; the caller dispatches on the shape. Passing
        ``gt`` costs nothing and keeps the baselines free of a per-model case.
        """
        return self._predict(rgb_u8, gt)

    def predict_stack(self, frames, target: int = -1,
                      gt: Optional[np.ndarray] = None) -> np.ndarray:
        """Depth for **one** frame of a stack the model saw jointly.

        ``frames`` is the whole window in temporal order and ``target`` indexes
        the one whose depth comes back. The other frames exist so that cross-view
        attention has something to attend to; nothing about them is scored.

        A stack of one is the ordinary single-frame call, so the two paths do not
        diverge. A stack of more than one on a model that cannot take it raises
        rather than quietly scoring the target alone — that failure mode returns
        a full, plausible table that reads "context does not help", which is the
        one wrong answer that looks like a result.
        """
        frames = list(frames)
        if len(frames) == 1:
            return self.predict(frames[0], gt)
        if self._predict_stack is None:
            raise SystemExit(
                f"[slambench] {self.key} cannot take a {len(frames)}-frame "
                f"context: it is monocular, and scoring the target frame alone "
                f"would report 'context does not help' when nothing was tried. "
                f"Run it with --context-frames 1, or drop it from --models.")
        return self._predict_stack(frames, target, gt)


# --------------------------------------------------------------------------- #
# Stand-ins
# --------------------------------------------------------------------------- #

def _analytic_predict(seed: int) -> Callable:
    """A dense map from image intensity — plausible in shape, wrong in value.

    The generator is built **per call**, not once, so the answer depends on the
    frame and not on how many times this stand-in has been asked before. A
    stand-in whose job is to prove the harness adds nothing cannot itself change
    with call order: the context sweep asks for the same frame once per arm, and
    with a running generator the arms would differ by the stand-in's own noise
    rather than by anything under test.
    """
    def predict(rgb_u8, gt=None):
        rng = np.random.default_rng(seed)
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

def _standin_stack(predict: Callable) -> Callable:
    """The stand-ins' multi-frame path: answer for the target, ignore the rest.

    They are harness probes, not models — ``analytic``'s score is meaningless by
    construction and ``oracle`` never looks at the image at all — so ignoring the
    context is the honest thing for them to do, and it is what makes the
    plumbing runnable without weights. It is emphatically **not** what a real
    monocular model may do here: there, silently scoring the target alone would
    produce a table that reads "context does not help", which is why
    :meth:`Model.predict_stack` refuses instead.
    """
    return lambda frames, target, gt=None: predict(frames[target], gt)


def load_model(key: str, device, **kw) -> Model:
    """Load one model. ``device`` is a torch device, ignored by the stand-ins.

    Raises ``SystemExit`` carrying the registry's own instruction when weights or
    a dependency are missing — the same message ``--list`` prints, so a failure
    is actionable without reading code.
    """
    if key == ANALYTIC:
        p = _analytic_predict(kw.get("seed", 0))
        return Model(key=key, family="analytic", size="—",
                     align_mode=kw.get("align_mode", "scale_shift"),
                     input_size=kw.get("input_size") or _DEFAULT_NATIVE_SIZE,
                     params_m=0.0, _predict=p, _predict_stack=_standin_stack(p))
    if key == ORACLE:
        p = _oracle_predict(kw.get("bias", 0.0), kw.get("noise", 0.0),
                            kw.get("scale", 3.0), kw.get("shift", 0.4),
                            kw.get("seed", 0))
        return Model(key=key, family="oracle", size="—",
                     align_mode=kw.get("align_mode", "scale_shift"),
                     input_size=kw.get("input_size") or _DEFAULT_NATIVE_SIZE,
                     params_m=0.0, _predict=p, _predict_stack=_standin_stack(p))

    spec = zoo.get_specs([key])[0]
    state, detail = zoo.status(spec)
    if state != "ready":
        raise SystemExit(f"[slambench] {key}: {state} — {detail}")

    adapter = zoo.build_adapter(spec, **{k: v for k, v in kw.items()
                                         if k in ("checkpoint", "use_camera")})
    adapter.load(device)

    def _as01(rgb_u8):
        return np.clip(np.asarray(rgb_u8, np.float32) / 255.0, 0.0, 1.0)

    def predict(rgb_u8, gt=None):
        return np.asarray(adapter.predict_frame(_as01(rgb_u8), None, "view"),
                          np.float32)

    #: ``Adapter.supports_context`` is the registry's own statement about the
    #: backbone; it is not restated here. The multi-view backbones take
    #: ``(S, 3, H, W)`` natively and return ``(S, H, W)``, so a window is one
    #: forward pass rather than a loop — looping would give each frame its own
    #: pass and measure nothing the single-frame arm has not already measured.
    ctx = bool(getattr(adapter, "supports_context", False))

    def predict_stack(frames, target=-1, gt=None):
        out = adapter.predict_frames([_as01(f) for f in frames], target=target)
        return np.asarray(out, np.float32)

    return Model(key=key, family=spec.family, size=spec.size,
                 align_mode=spec.align_native[0], input_size=native_size(key),
                 params_m=adapter.num_params() / 1e6,
                 supports_context=ctx, _predict=predict,
                 _predict_stack=predict_stack if ctx else None)


def takes_context(key: str) -> bool:
    """Whether ``key`` may be *asked* for a multi-frame window — **no weights**.

    ``supports_context`` is a class attribute of the adapter, so this answers
    before anything is downloaded or moved to a GPU. That is the point: a run
    that asks five models for a 10-frame context and finds out on the third that
    one of them is monocular has already spent two model-loads to learn it.

    The stand-ins pass, and they do not *use* the context (see
    :func:`_standin_stack`). They are how the sweep's plumbing is exercised
    without weights, and their numbers are meaningless by construction, so there
    is no reading of them under which ignoring the context is a claim.
    """
    if key in STANDINS:
        return True
    return bool(getattr(zoo.build_adapter(zoo.get_specs([key])[0]),
                        "supports_context", False))


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
