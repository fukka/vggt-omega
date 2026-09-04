"""Fit and remove the radial range-compression field from sparse anchors.

THE FIELD, AS MEASURED
----------------------
run_009 is alignment-free and says the failure is a *precise* miscalibration,
not noise: dispersion 2-10% everywhere, while 0-1 m content is placed 1.7-3.3x
too far (worst at the rim) and 5-10 m content 1.4-1.8x too near. A monotone
map that squeezes the range axis and squeezes it more at large incidence.

The model for that, per incidence bin, is one line in log-log:

    log(pred) = c(theta) + g(theta) * log(true)

``g < 1`` IS the compression -- the same signature as the RayTun3R repro's
depth-gain 0.406 -- and the correction is the inverse of the same line:

    log(true_hat) = (log(pred) - c(theta)) / g(theta)

WHY THIS IS NOT run_010 AGAIN
-----------------------------
H2.1's 48-parameter table was indexed by ``(theta, predicted depth)`` and it
FAILED with a mechanism: the compression makes predicted depth many-to-one in
true depth, so an output-indexed correction pushes the majority's fix onto the
minority, and the near centre paid for it. Two things are different here.
First, this is fitted against *triangulated truth at the anchors*, so it is not
inferred from the prediction's own statistics. Second, it is a monotone
invertible map per bin -- ``g > 0`` is enforced -- so it cannot be many-to-one
by construction. Whether that is enough is what the run decides.

THE ARMS
--------
``raycal``   per-theta-bin (g, c) -- the hypothesis
``global``   ONE (g, c) for the whole image. The control that matters: if a
             single log-linear recalibration does as well, the field's RADIAL
             structure is not what is being exploited and the claim collapses
             to "the model's range axis is miscalibrated", which is already
             known.
``shuffled`` per-bin, but the anchors' theta labels are permuted before
             binning. Same anchors, same count, same per-bin sample sizes, no
             radial correspondence.
``raycal_bal`` per-bin, but each bin's line is fitted on a DEPTH-BALANCED
             sample. Added 2026-09-04 against a diagnosed failure, not a hunch:
             the first run beat both controls on 6/6 sequences and still failed
             its locked bar, damaging near_centre by +72%/+57%. The mechanism is
             sampling, not modelling -- at small theta the anchors are almost
             all far content (walls), so the centre bins' lines are fitted over
             a narrow far range and mis-extrapolate onto near content. Balancing
             the depth histogram inside each bin removes exactly that, and it
             does NOT reintroduce H2.1's failure: the fit stays ONE monotone
             line per bin, so it cannot be many-to-one. Indexing the correction
             by predicted depth would be H2.1 again and is deliberately not
             done.
``none``     identity.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

__all__ = ["fit_field", "apply_field", "ARMS"]

ARMS = ("raycal", "raycal_bal", "global", "shuffled", "none")

#: A bin whose fitted gain leaves this range is not describing a compression;
#: it is describing noise, and it would invert or explode the correction.
G_MIN, G_MAX = 0.15, 3.0


def _fit_line(log_pred: np.ndarray, log_true: np.ndarray) -> Tuple[float, float]:
    """Least squares ``log_pred = c + g * log_true``; returns ``(g, c)``."""
    if log_true.size < 2 or np.allclose(log_true, log_true[0]):
        return 1.0, float(np.mean(log_pred - log_true)) if log_true.size else 0.0
    g, c = np.polyfit(log_true, log_pred, 1)
    return float(g), float(c)


def _fit_line_w(log_pred: np.ndarray, log_true: np.ndarray,
                w: np.ndarray) -> Tuple[float, float]:
    """Weighted least squares ``log_pred = c + g * log_true``."""
    if log_true.size < 2 or w.sum() <= 0:
        return 1.0, 0.0
    W = w / w.sum()
    mx, my = float((W * log_true).sum()), float((W * log_pred).sum())
    vxx = float((W * (log_true - mx) ** 2).sum())
    vxy = float((W * (log_true - mx) * (log_pred - my)).sum())
    if vxx <= 1e-12:
        return 1.0, my - mx
    g = vxy / vxx
    return g, my - g * mx


def fit_field(arm: str, theta: np.ndarray, pred: np.ndarray, true: np.ndarray,
              theta_max: float, n_bins: int = 8, min_per_bin: int = 40,
              seed: int = 0) -> Dict:
    """Fit ``(g, c)`` per incidence bin from the anchors."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")
    ok = (pred > 1e-6) & (true > 1e-6) & np.isfinite(pred) & np.isfinite(true)
    theta, pred, true = theta[ok], pred[ok], true[ok]
    lp, lt = np.log(pred), np.log(true)
    g_glob, c_glob = _fit_line(lp, lt)
    g_glob = float(np.clip(g_glob, G_MIN, G_MAX))

    edges = np.linspace(0.0, float(theta_max), n_bins + 1)
    if arm == "none":
        return {"arm": arm, "edges": edges.tolist(),
                "g": [1.0] * n_bins, "c": [0.0] * n_bins,
                "n": [0] * n_bins, "g_global": 1.0, "c_global": 0.0,
                "n_anchors": int(theta.size)}
    if arm == "global":
        return {"arm": arm, "edges": edges.tolist(),
                "g": [g_glob] * n_bins, "c": [c_glob] * n_bins,
                "n": [int(theta.size)] * n_bins,
                "g_global": g_glob, "c_global": c_glob,
                "n_anchors": int(theta.size)}

    th = theta
    if arm == "shuffled":
        # Permute the RADIAL LABEL only. Same anchors, same values, same
        # per-bin counts in distribution -- no correspondence between an
        # anchor's incidence angle and its own miscalibration.
        th = np.random.default_rng(seed).permutation(theta)

    idx = np.clip(np.digitize(th, edges) - 1, 0, n_bins - 1)
    # Depth-balancing weights: inside each theta bin, weight anchors so the
    # depth histogram is flat in log-depth. Computed against a COMMON set of
    # depth edges so every bin's line is fitted over the same range.
    d_edges = np.geomspace(max(true.min(), 0.2), min(true.max(), 12.0), 9)
    d_idx = np.clip(np.digitize(true, d_edges) - 1, 0, len(d_edges) - 2)

    gs, cs, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if arm == "raycal_bal" and int(m.sum()) >= min_per_bin:
            w = np.zeros(int(m.sum()))
            dsub = d_idx[m]
            for j in range(len(d_edges) - 1):
                k = dsub == j
                if k.sum():
                    w[k] = 1.0 / k.sum()
            g, c = _fit_line_w(lp[m], lt[m], w)
            gs.append(float(np.clip(g, G_MIN, G_MAX))); cs.append(float(c))
            ns.append(int(m.sum()))
            continue
        if int(m.sum()) < min_per_bin:
            # Too few anchors to fit a line. Falling back to the global fit is
            # the honest move: inventing a per-bin slope from 5 points is how a
            # correction becomes a ring artefact.
            gs.append(g_glob); cs.append(c_glob); ns.append(int(m.sum()))
            continue
        g, c = _fit_line(lp[m], lt[m])
        gs.append(float(np.clip(g, G_MIN, G_MAX))); cs.append(float(c))
        ns.append(int(m.sum()))
    return {"arm": arm, "edges": edges.tolist(), "g": gs, "c": cs, "n": ns,
            "g_global": g_glob, "c_global": c_glob, "n_anchors": int(theta.size)}


def apply_field(pred: np.ndarray, theta: np.ndarray, field: Dict) -> np.ndarray:
    """Invert the fitted line at every pixel: ``log(true) = (log(pred)-c)/g``.

    ``(g, c)`` are interpolated LINEARLY in theta between bin centres rather
    than applied piecewise-constant: a piecewise-constant correction writes
    visible rings into the depth map at the bin edges, and a ring is a radial
    artefact in an experiment whose whole subject is radial artefacts.
    """
    edges = np.asarray(field["edges"], float)
    mid = 0.5 * (edges[:-1] + edges[1:])
    g = np.interp(theta, mid, np.asarray(field["g"], float))
    c = np.interp(theta, mid, np.asarray(field["c"], float))
    out = np.exp((np.log(np.clip(pred, 1e-6, None)) - c) / np.clip(g, G_MIN, G_MAX))
    return np.where(pred > 1e-6, out, pred)
