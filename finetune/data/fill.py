# Copyright (c) 2026.
"""Fill strategies for the invalid region of a fisheye / rectified-fisheye frame.

Both input domains carry pixels that carry no image:

* **raw fisheye** — the sensor is square but the lens images a circle, so the
  four corners (incidence ``theta > theta_max``, past the KB4 polynomial
  turnover) are black.
* **rectified** — undistorting the whole imaged cone to a pinhole plane needs a
  frame that circumscribes the cone's disc, leaving four black wedges (21.5% of
  the pixels for the Aria 214-1 cone, but only 6.7% of the solid angle: the rim
  is magnified ~10x, so those wedges are pixel-space giants holding almost no
  scene).

Leaving them at zero is not a neutral "no information" signal — blacking out
pixels is a distinct, biasing input pattern (Jain et al., *Missingness Bias in
Model Debugging*, ICLR 2022), and networks are known to read absolute position
off exactly this kind of boundary cue (Kayhan & van Gemert, CVPR 2020). This
module supplies the alternatives so the choice becomes a measured variable
rather than an unexamined default.

The fill is only ever cosmetic with respect to scoring: every caller keeps the
analytic validity mask and evaluates inside it, so filled pixels are never
compared against ground truth. What the fill changes is what the *encoder* sees.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

FILL_MODES: Tuple[str, ...] = ("oracle", "black", "mean", "chanmean", "replicate",
                               "mirror", "noise", "telea", "ns")


def _as_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def apply_fill(img_hwc: np.ndarray, valid_hw: np.ndarray, mode: str = "black",
               seed: int = 0) -> np.ndarray:
    """Fill ``~valid`` in a float HWC image in [0,1].

    Parameters
    ----------
    img_hwc  : ``(H, W, C)`` float32 in [0,1]. Invalid pixels may hold anything.
    valid_hw : ``(H, W)`` bool — True where the pixel carries real image content.
    mode     : one of :data:`FILL_MODES`.

        ``black``      zeros (the current default everywhere in this repo)
        ``mean``       scalar mean over valid pixels (all channels alike)
        ``chanmean``   per-channel mean over valid pixels
        ``replicate``  nearest valid pixel (exact, via distance transform labels)
        ``mirror``     reflect the image about its border into the hole
        ``noise``      Gaussian matched to the valid region's per-channel moments
        ``telea``      cv2.inpaint, Telea's fast marching
        ``ns``         cv2.inpaint, Navier-Stokes

    Returns
    -------
    ``(H, W, C)`` float32 in [0,1]; pixels inside ``valid`` are untouched.

    Notes
    -----
    ``replicate`` is the cheap fill that matters most: it is the strongest
    strategy that invents no content at all (every filled pixel is a copy of a
    real one). If the accuracy curve saturates at ``replicate``, then the damage
    from black regions is low-level input statistics and a generative filler
    buys nothing — the negative result the 2x2 is designed to be able to reach.
    """
    if mode not in FILL_MODES:
        raise ValueError(f"unknown fill mode {mode!r}; expected one of {FILL_MODES}")

    img = np.ascontiguousarray(img_hwc, dtype=np.float32)
    if img.ndim == 2:
        img = img[..., None]
    valid = np.ascontiguousarray(valid_hw).astype(bool)
    hole = ~valid

    # Nothing to do: no hole, or (degenerate) no valid pixel to fill from.
    if not hole.any() or not valid.any():
        out = img.copy()
        if mode != "oracle":
            out[hole] = 0.0
        return out

    if mode == "oracle":
        # Leave the region exactly as it is. Only meaningful when the "hole" is a
        # SYNTHETIC mask imposed on a frame that really does hold image content
        # there (see FisheyeRectifier(synth_hole_inscribed=True)) -- then this arm
        # is the ground-truth fill, the upper bound every other mode is measured
        # against. On a genuinely unimaged region it would leave garbage, so the
        # caller is responsible for using it only in the synthetic-hole setting.
        return img.copy()

    if mode == "black":
        out = img.copy()
        out[hole] = 0.0
        return out

    if mode in ("mean", "chanmean"):
        vals = img[valid]                                   # (Nvalid, C)
        fillv = vals.mean() if mode == "mean" else vals.mean(axis=0)
        out = img.copy()
        out[hole] = fillv
        return out

    if mode == "noise":
        vals = img[valid]
        mu, sd = vals.mean(axis=0), vals.std(axis=0)
        rng = np.random.default_rng(seed)
        out = img.copy()
        out[hole] = np.clip(rng.normal(mu, np.maximum(sd, 1e-6),
                                       size=(int(hole.sum()), img.shape[2])), 0.0, 1.0)
        return out

    import cv2

    if mode == "replicate":
        # distanceTransformWithLabels gives, for every zero pixel, the label of the
        # nearest non-zero pixel. Seeding labels with a raster index of the valid
        # pixels turns that into an exact nearest-valid-pixel gather -- no blending,
        # no invented colour, which is precisely the property we want to test.
        src = valid.astype(np.uint8)
        _, labels = cv2.distanceTransformWithLabels(
            1 - src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(valid)
        # OpenCV labels the nearest-zero-of-(1-src) pixels 1..N in raster order.
        order = np.argsort(ys * valid.shape[1] + xs)
        ys, xs = ys[order], xs[order]
        idx = np.clip(labels - 1, 0, len(ys) - 1)
        out = img.copy()
        out[hole] = img[ys[idx[hole]], xs[idx[hole]]]
        return out

    if mode == "mirror":
        # Reflect about the valid region's bounding box, then keep only what lands
        # in the hole. Cheap, invents nothing, but (unlike replicate) carries real
        # texture rather than a smeared edge colour.
        #
        # This only means anything when the hole is a BORDER BAND. Both holes in
        # this repo are not: the fisheye's is four corners and the rectified one is
        # four wedges, each surrounded by valid pixels, so the valid region's
        # bounding box is essentially the whole frame, the padding is empty, and
        # the reflection cannot reach the hole at all. Measured on a rendered ADT
        # frame it changed 0.0% of the fisheye hole and 3.4% of the rectified one
        # -- while returning successfully. An arm that silently fills nothing
        # scores exactly like black and reads as a clean "mirroring does not help",
        # which is a conclusion about this function rather than about mirroring.
        # So it refuses instead.
        ys, xs = np.nonzero(valid)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        H, W = img.shape[:2]
        rr, cc = np.ogrid[:H, :W]
        reachable = (rr < y0) | (rr >= y1) | (cc < x0) | (cc >= x1)
        unreached = float((hole & ~reachable).sum()) / max(1, int(hole.sum()))
        if unreached > 0.05:
            raise ValueError(
                f"'mirror' cannot fill this hole: {unreached:.1%} of it lies inside "
                f"the valid region's bounding box, so reflecting about that box "
                f"leaves it untouched. Reflection only reaches a hole that is a "
                f"border band; this one is enclosed by valid pixels. Use "
                f"'replicate' for a fill that invents nothing.")
        core = img[y0:y1, x0:x1]
        pad = cv2.copyMakeBorder(core, y0, H - y1, x0, W - x1, cv2.BORDER_REFLECT_101)
        out = img.copy()
        out[hole] = pad[hole]
        return out

    # telea / ns -- OpenCV inpainting, 8-bit only.
    flag = cv2.INPAINT_TELEA if mode == "telea" else cv2.INPAINT_NS
    src8 = _as_u8(img)
    if src8.shape[2] == 1:
        src8 = src8[..., 0]
    filled = cv2.inpaint(src8, hole.astype(np.uint8), 5, flag)
    if filled.ndim == 2:
        filled = filled[..., None]
    out = img.copy()
    out[hole] = filled.astype(np.float32)[hole] / 255.0
    return out
