# Copyright (c) 2026.
"""Multi-view -> fisheye fusion (VGGT-360 module 3, fisheye port).

Upstream ``depth_set_to_equirect_attention`` fuses the per-view depth maps
back onto the ERP panorama: it builds the ERP longitude/latitude *ray field*,
gnomonically projects every ray into each perspective view, samples depth and
a per-pixel attention-derived confidence weight, and takes the weighted mean.

Everything after the ray field is generic to ANY output ray field — the
gnomonic projection, visibility test, cv2.remap sampling and weighted average
never mention ERP.  So the port is: replace the lon/lat grid with the KB4
per-pixel ray LUT of the fisheye frame (``fisheye_cam.fisheye_ray_lut``) and
keep the rest, with two fisheye-specific additions:

  1. **Analytic per-view valid masks** multiply into the fusion weights.
     Ring views have cone-clipped black corners where VGGT still hallucinates
     depth values; a fisheye ray near that boundary would bilinearly mix them
     in.  The masks (eroded a few px for safety) zero those samples.
  2. The output is masked to the fisheye's physically imaged cone.

``build_selfview_confidence`` — the paper's sharpness/locality(recv)/symmetry
attention-metric combination (module 3's correlation weights) — is vendored
**unchanged** from upstream ``utils/ERP_utils.py``: it consumes only the saved
frame-attention matrices and is projection-agnostic.

Conventions (must match ``fisheye_views.fisheye_to_persp`` exactly; verified
end-to-end by ``checks/check_fisheye2persp.py`` test C):
  camera frame x right / y down / z forward; a view's tangent grid spans
  ``[-tan(fov/2), +tan(fov/2)]`` in both axes with y down, so a ray with
  view coords ``(x, y, z)`` lands at ``u = ((x/z)/t + 1)/2 * (W-1)`` and
  ``v = ((y/z)/t + 1)/2 * (H-1)`` (no vertical flip anywhere).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .fisheye_cam import FisheyeCam, fisheye_ray_lut
from .fisheye_views import ViewParam, view_rotation


# --------------------------------------------------------------------------- #
# Generic ray-field fusion (numpy-only; also used by the geometry checker)
# --------------------------------------------------------------------------- #

def fuse_views_to_fisheye(
    values: Sequence[np.ndarray],
    view_params: Sequence[ViewParam],
    cam: FisheyeCam,
    weights: Optional[Sequence[np.ndarray]] = None,
    view_valids: Optional[Sequence[np.ndarray]] = None,
    interp: str = "linear",
    min_weight: float = 0.0,
    erode_valid_px: int = 3,
    rescue_rim: bool = True,
    ray_lut: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted-average fusion of per-view maps onto the fisheye pixel grid.

    Parameters
    ----------
    values      : per-view maps, each ``(Hc, Wc)`` or ``(Hc, Wc, C)`` float.
                  For the depth pipeline these are radial distances; the
                  checker also feeds RGB through to validate geometry.
    view_params : matching ``(azimuth, tilt, fov)`` per view.
    cam         : fisheye intrinsics of the OUTPUT grid (defines both its size
                  and its rays).
    weights     : optional per-view ``(Hc, Wc)`` confidence maps (e.g. from
                  ``build_selfview_confidence``); None -> uniform.
    view_valids : optional per-view ``(Hv, Wv)`` analytic valid masks from
                  ``fisheye_to_persp`` (any resolution — remapped like the
                  values).  Strongly recommended: excludes cone-clipped
                  corners from the average.
    interp      : 'linear' or 'nearest' sampling of the value maps.
    min_weight  : drop samples whose weight falls below this (upstream knob).
    erode_valid_px : erosion radius applied to view_valids before use, so
                  bilinear value samples never straddle the invalid boundary.
    ray_lut     : optional ``(rays[H,W,3], inside_cone[H,W])`` for the output
                  grid, replacing ``fisheye_ray_lut(cam)``.  Same reason as
                  ``fisheye_to_persp(project=...)``: a FISHEYE624 lens has no
                  KB4 inverse, and this is where the inverse is needed.  It
                  also lets a caller that already built the LUT for its own
                  masking hand it over instead of paying for it twice.
                  ``cam`` still supplies the output grid size, and the two must
                  agree — checked below, because a mismatch would fuse onto a
                  correctly-shaped grid with every ray wrong.
    rescue_rim  : two-tier fallback (default True).  Eroding the valid masks
                  retires a thin band at the cone rim in EVERY view at once
                  (all views share the same theta_max), which would leave
                  ~0.1-0.4%% of the imaged cone with zero coverage — enlarging
                  or re-tilting the views cannot fix this (measured: the miss
                  is erosion-, not layout-, driven).  Instead, pixels whose
                  *eroded*-weight sum is empty fall back to the un-eroded
                  weights: full coverage, with the slightly riskier
                  boundary-adjacent samples confined to the rim band only.

    Returns
    -------
    fused    : ``(H, W)`` or ``(H, W, C)`` — weighted mean, 0 where no data.
    coverage : ``(H, W)`` int32 — number of views contributing per pixel
               (counted on whichever tier the pixel used).
    """
    assert len(values) == len(view_params)
    H, W = cam.H, cam.W
    if ray_lut is None:
        rays, cone = fisheye_ray_lut(cam)          # (H, W, 3), (H, W)
    else:
        rays, cone = ray_lut
        if rays.shape[:2] != (H, W) or cone.shape[:2] != (H, W):
            raise ValueError(
                f"ray_lut is {rays.shape[:2]} but cam describes a {(H, W)} "
                f"grid; fusing would land every ray on the wrong pixel")

    first = np.asarray(values[0])
    C = 1 if first.ndim == 2 else first.shape[2]
    interp_flag = cv2.INTER_NEAREST if interp == "nearest" else cv2.INTER_LINEAR

    erp_num = np.zeros((H, W, C), dtype=np.float64)
    erp_den = np.zeros((H, W), dtype=np.float64)
    coverage = np.zeros((H, W), dtype=np.int32)
    # rim-rescue tier: same sums with UN-eroded validity (see docstring)
    num_loose = np.zeros((H, W, C), dtype=np.float64)
    den_loose = np.zeros((H, W), dtype=np.float64)
    cov_loose = np.zeros((H, W), dtype=np.int32)

    for i, (val, (psi, tilt, fov)) in enumerate(zip(values, view_params)):
        val = np.asarray(val, dtype=np.float32)
        if val.ndim == 2:
            val = val[..., None]
        Hc, Wc = val.shape[:2]

        # Fisheye ray -> this view's frame.  v_cam = R @ v_view  =>  row-vector
        # form v_view = v_cam @ R  (right-multiplying by R == R^T @ column).
        R = view_rotation(psi, tilt).astype(np.float32)
        d_v = rays @ R
        zc = d_v[..., 2]
        t = np.float32(np.tan(np.radians(fov) * 0.5))
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = d_v[..., 0] / zc
            yc = d_v[..., 1] / zc
        vis = (zc > 1e-6) & (np.abs(xc) <= t) & (np.abs(yc) <= t) & cone
        if not np.any(vis):
            continue

        # Gnomonic coords -> view pixel coords (y down, matching the render).
        mapx = ((xc / t + 1.0) * 0.5 * (Wc - 1)).astype(np.float32)
        mapy = ((yc / t + 1.0) * 0.5 * (Hc - 1)).astype(np.float32)
        mapx[~vis] = -1.0  # BORDER_CONSTANT(0) -> sample dies
        mapy[~vis] = -1.0

        sampled = cv2.remap(val, mapx, mapy, interpolation=interp_flag,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        if sampled.ndim == 2:
            sampled = sampled[..., None]

        # Per-view weight: attention confidence x analytic validity.
        w = np.ones((H, W), dtype=np.float32)
        if weights is not None and weights[i] is not None:
            w_i = np.asarray(weights[i], dtype=np.float32)
            if w_i.ndim == 3:
                w_i = w_i[0] if w_i.shape[0] == 1 else w_i[..., 0]
            if w_i.shape != (Hc, Wc):   # same value-grid guard as the masks
                w_i = cv2.resize(w_i, (Wc, Hc), interpolation=cv2.INTER_LINEAR)
            w = cv2.remap(w_i, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        w_loose = w
        if view_valids is not None and view_valids[i] is not None:
            vv = np.asarray(view_valids[i], dtype=np.float32)
            # mapx/mapy are in the VALUE map's pixel coords (Hc, Wc).  The
            # valid masks typically come from fisheye_to_persp at persp_size
            # (512) while VGGT outputs 518 — resample the mask onto the value
            # grid first, or the remap below would read it misaligned by
            # ~Hc/Hv (a ~6 px skew at the rim; real bug found in GPU runs).
            if vv.shape != (Hc, Wc):
                vv = cv2.resize(vv, (Wc, Hc), interpolation=cv2.INTER_NEAREST)
            vv_raw = cv2.remap(vv, mapx, mapy, interpolation=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            if erode_valid_px > 0:
                kernel = np.ones((2 * erode_valid_px + 1,) * 2, np.uint8)
                vv_er = cv2.erode(vv, kernel)
                vv_s = cv2.remap(vv_er, mapx, mapy,
                                 interpolation=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            else:
                vv_s = vv_raw
            w_loose = w * vv_raw
            w = w * vv_s
        w = np.where(vis, w, 0.0)
        w_loose = np.where(vis, w_loose, 0.0)
        if min_weight > 0:
            w = np.where(w >= min_weight, w, 0.0)
            w_loose = np.where(w_loose >= min_weight, w_loose, 0.0)

        coverage += (w > 0).astype(np.int32)
        erp_num += (w[..., None] * sampled).astype(np.float64)
        erp_den += w.astype(np.float64)
        if rescue_rim:
            cov_loose += (w_loose > 0).astype(np.int32)
            num_loose += (w_loose[..., None] * sampled).astype(np.float64)
            den_loose += w_loose.astype(np.float64)

    if rescue_rim:
        # rim band: pixels the eroded tier left empty but the raw tier covers
        rim = (erp_den <= 0) & (den_loose > 0)
        erp_num[rim] = num_loose[rim]
        erp_den[rim] = den_loose[rim]
        coverage[rim] = cov_loose[rim]

    fused = np.zeros((H, W, C), dtype=np.float32)
    nz = erp_den > 0
    fused[nz] = (erp_num[nz] / erp_den[nz, None]).astype(np.float32)
    if C == 1:
        fused = fused[..., 0]
    return fused, coverage


# --------------------------------------------------------------------------- #
# Cross-view consistency diagnostics + scale harmonisation
#
# Why: seams at view boundaries in the fused depth mean the per-view radial
# distances DISAGREE where views overlap.  Since all views share one optical
# center, overlapping rays must have identical range — any systematic ratio
# between two views is VGGT scale drift (pure-rotation multi-view is a
# degenerate case for triangulation, so per-view monocular scale can wander).
# These helpers (a) measure that disagreement directly and (b) optionally
# correct it with a least-squares per-view scale before fusion.
# --------------------------------------------------------------------------- #

def per_view_fisheye_ranges(
    values: Sequence[np.ndarray],
    view_params: Sequence[ViewParam],
    cam: FisheyeCam,
    view_valids: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Each view's value map resampled alone onto the fisheye grid.

    Returns ``maps (S, H, W) float32`` (NaN where the view does not see the
    ray) and ``ok (S, H, W) bool``.  This is the single most useful debugging
    artifact: a montage of these shows exactly which view disagrees where,
    before any weighting can blur the story.
    """
    rays, cone = fisheye_ray_lut(cam)
    S = len(values)
    maps = np.full((S, cam.H, cam.W), np.nan, dtype=np.float32)
    ok = np.zeros((S, cam.H, cam.W), dtype=bool)
    for i, (val, (psi, tilt, fov)) in enumerate(zip(values, view_params)):
        val = np.asarray(val, dtype=np.float32)
        Hc, Wc = val.shape[:2]
        R = view_rotation(psi, tilt).astype(np.float32)
        d_v = rays @ R
        zc = d_v[..., 2]
        t = np.float32(np.tan(np.radians(fov) * 0.5))
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = d_v[..., 0] / zc
            yc = d_v[..., 1] / zc
        vis = (zc > 1e-6) & (np.abs(xc) <= t) & (np.abs(yc) <= t) & cone
        mapx = ((xc / t + 1.0) * 0.5 * (Wc - 1)).astype(np.float32)
        mapy = ((yc / t + 1.0) * 0.5 * (Hc - 1)).astype(np.float32)
        mapx[~vis] = -1.0
        mapy[~vis] = -1.0
        sampled = cv2.remap(val, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        good = vis & (sampled > 0)
        if view_valids is not None and view_valids[i] is not None:
            vv = np.asarray(view_valids[i], dtype=np.float32)
            if vv.shape != (Hc, Wc):
                vv = cv2.resize(vv, (Wc, Hc), interpolation=cv2.INTER_NEAREST)
            vv_s = cv2.remap(vv, mapx, mapy, interpolation=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            good &= vv_s > 0.5
        maps[i][good] = sampled[good]
        ok[i] = good
    return maps, ok


def pairwise_scale_stats(maps: np.ndarray, ok: np.ndarray,
                         min_overlap: int = 500):
    """Median range ratio between every overlapping view pair.

    Returns ``(ratio (S, S), n_overlap (S, S))`` with ``ratio[i, j] =
    median(maps_i / maps_j)`` over their shared pixels (NaN if the overlap is
    below ``min_overlap``).  On a healthy reconstruction every ratio is ~1;
    ratios off by >5-10%% mean VGGT gave the views inconsistent scales and the
    fused depth WILL show seams — fix upstream (masks/views) or harmonise.
    """
    S = maps.shape[0]
    ratio = np.full((S, S), np.nan, dtype=np.float64)
    n_ov = np.zeros((S, S), dtype=np.int64)
    for i in range(S):
        ratio[i, i] = 1.0
        for j in range(i + 1, S):
            both = ok[i] & ok[j]
            n = int(both.sum())
            n_ov[i, j] = n_ov[j, i] = n
            if n < min_overlap:
                continue
            r = float(np.median(maps[i][both] / np.clip(maps[j][both], 1e-9, None)))
            ratio[i, j] = r
            ratio[j, i] = 1.0 / r if r > 0 else np.nan
    return ratio, n_ov


def harmonize_view_scales(maps: np.ndarray, ok: np.ndarray,
                          min_overlap: int = 500) -> np.ndarray:
    """Per-view scales ``s (S,)`` making overlaps agree: ``s_i*m_i ~= s_j*m_j``.

    Weighted least squares on the overlap graph in log space:
    ``log s_i - log s_j = -log median(m_i/m_j)`` for every pair with enough
    overlap, weight ``sqrt(n_overlap)``; gauge fixed by ``mean(log s) = 0`` so
    the global (free) scale is untouched.  Views with no usable overlap keep
    ``s = 1``.  Apply by scaling each view's radial map before fusion.
    """
    S = maps.shape[0]
    ratio, n_ov = pairwise_scale_stats(maps, ok, min_overlap)
    rows, rhs, wts = [], [], []
    for i in range(S):
        for j in range(i + 1, S):
            if not np.isfinite(ratio[i, j]) or ratio[i, j] <= 0:
                continue
            row = np.zeros(S)
            row[i], row[j] = 1.0, -1.0
            rows.append(row)
            rhs.append(-np.log(ratio[i, j]))
            wts.append(np.sqrt(float(n_ov[i, j])))
    if not rows:
        return np.ones(S, dtype=np.float32)
    # gauge row: sum(log s) = 0, weighted strongly
    rows.append(np.ones(S))
    rhs.append(0.0)
    wts.append(float(np.sum(wts)))
    A = np.asarray(rows) * np.asarray(wts)[:, None]
    b = np.asarray(rhs) * np.asarray(wts)
    logs, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.exp(logs).astype(np.float32)


# --------------------------------------------------------------------------- #
# Attention-derived per-view confidence — vendored UNCHANGED from upstream
# utils/ERP_utils.py (paper module 3: sharpness / recv / symmetry metrics).
# Projection-agnostic: consumes only the saved frame-attention matrices.
# torch is imported lazily so the numpy-only geometry checker never needs it.
# --------------------------------------------------------------------------- #

def _global_pct_norm(x, low, high, eps):
    import torch
    flat = x.reshape(-1)
    lo_q = torch.quantile(flat, q=low)
    hi_q = torch.quantile(flat, q=high)
    return (x.clamp(min=lo_q, max=hi_q) - lo_q) / (hi_q - lo_q + eps)


def _build_midband_gate(x, low_q, high_q, floor, eps):
    import torch
    flat = x.reshape(-1)
    low_v = torch.quantile(flat, q=low_q)
    high_v = torch.quantile(flat, q=high_q)
    center = 0.5 * (low_v + high_v)
    radius = 0.5 * (high_v - low_v)
    if radius <= eps:
        return torch.ones_like(x)
    gate = 1.0 - torch.abs(x - center) / (radius + eps)
    gate = gate.clamp(min=0.0, max=1.0)
    return floor + (1.0 - floor) * gate


def build_selfview_confidence(
        self_attn_per_layer,
        toks_per_img_total=1374,
        num_special_tokens=5,
        patch_grid_hw=(37, 37),
        img_hw=(518, 518),
        use_logits: bool = False,
        head_reduce: str = "median",
        layer_weights=None,
        percentile_clip=(0.05, 0.95),
        eps: float = 1e-8,
):
    """Per-view, per-pixel confidence from the saved frame attention.

    Upstream VGGT-360 verbatim.  Combines, per patch token:
      * sharpness — Bhattacharyya overlap between the masked and unmasked
        attention rows (attention concentration / distinctiveness),
      * recv      — column sums, "how much other tokens attend to me",
        mid-band gated so both extremes are damped,
      * symmetry  — Bhattacharyya between A and its column-normalised
        transpose (bidirectional consistency),
    then percentile-normalises, mixes, floors at 0.05 and bilinearly upsamples
    the 37x37 patch grid to ``img_hw``.  Output ``(I, 1, H, W)`` weights.
    """
    import torch
    import torch.nn.functional as F

    if isinstance(self_attn_per_layer, dict):
        masked_data = self_attn_per_layer["masked"]
        unmasked_data = self_attn_per_layer.get("unmasked", masked_data)
        layers = masked_data if isinstance(masked_data, list) else [masked_data]
        sharp_layers = unmasked_data if isinstance(unmasked_data, list) else [unmasked_data]
    elif isinstance(self_attn_per_layer, torch.Tensor):
        layers = [self_attn_per_layer]
        sharp_layers = layers
    else:
        layers = self_attn_per_layer
        sharp_layers = layers

    I, Hh, N, _ = layers[0].shape
    Ttot = toks_per_img_total
    S = num_special_tokens
    Hp, Wp = patch_grid_hw
    Tpatch = Hp * Wp
    assert N == Ttot
    L = len(layers)

    device = layers[0].device

    if layer_weights is None:
        layer_weights = torch.ones(L, dtype=torch.float32, device=device) / max(L, 1)
    else:
        layer_weights = torch.tensor(layer_weights, dtype=torch.float32, device=device)
        layer_weights = layer_weights / (layer_weights.sum() + eps)

    patch_slice = slice(S, Ttot)
    sharp_all = torch.zeros(I, Tpatch, device=device)
    recv_all = torch.zeros(I, Tpatch, device=device)
    sym_all = torch.zeros(I, Tpatch, device=device)

    def _to_prob(blk):
        if use_logits:
            blk = blk - blk.max(dim=-1, keepdim=True).values
            return blk.softmax(dim=-1)
        blk = torch.nan_to_num(blk, nan=0.0, posinf=0.0, neginf=0.0)
        return blk / (blk.sum(dim=-1, keepdim=True) + eps)

    for k in range(L):
        A_blk = _to_prob(layers[k][:, :, patch_slice, patch_slice].float())
        A_unmasked = sharp_layers[k][:, :, patch_slice, patch_slice].float()
        A_unmasked = torch.nan_to_num(A_unmasked, nan=0.0, posinf=0.0, neginf=0.0)
        A_unmasked = A_unmasked / (A_unmasked.sum(dim=-1, keepdim=True) + eps)
        sharp_h = (A_unmasked * A_blk).clamp(min=0.0).sqrt().sum(dim=-1)

        recv_h = A_blk.sum(dim=-2)
        col_sum = A_blk.sum(dim=-2, keepdim=False).unsqueeze(-1)
        A_prime = A_blk.transpose(-2, -1) / (col_sum + eps)
        sym_h = (A_blk * A_prime).clamp(min=0.0).sqrt().sum(dim=-1)

        if head_reduce == "median":
            sharp_k = sharp_h.median(dim=1).values
            recv_k = recv_h.median(dim=1).values
            sym_k = sym_h.median(dim=1).values
        else:
            sharp_k = sharp_h.mean(dim=1)
            recv_k = recv_h.mean(dim=1)
            sym_k = sym_h.mean(dim=1)

        w = layer_weights[k]
        sharp_all += w * sharp_k
        recv_all += w * recv_k
        sym_all += w * sym_k

    low, high = percentile_clip

    sharp_norm = _global_pct_norm(sharp_all, low=low, high=high, eps=eps)
    recv_norm = _global_pct_norm(recv_all, low=low, high=high, eps=eps)
    sym_norm = _global_pct_norm(sym_all, low=low, high=high, eps=eps)

    recv_gate_map = _build_midband_gate(recv_norm, 0, 0.6, 0.5, eps)
    recv_for_combine = recv_gate_map * recv_norm

    score_patch = sym_norm.clamp(min=eps) * (1.0 + recv_for_combine) + 0.5 * sharp_norm
    score_patch = _global_pct_norm(score_patch, low=low, high=high, eps=eps)
    score_floor = 0.05
    score_patch = score_floor + (1.0 - score_floor) * score_patch

    S_pix_list = []
    for i in range(I):
        S_grid = score_patch[i].reshape(1, 1, Hp, Wp)
        S_pix = F.interpolate(S_grid, size=img_hw, mode="bilinear", align_corners=False)
        S_pix_list.append(S_pix)
    S_pix = torch.cat(S_pix_list, dim=0)
    return S_pix


# --------------------------------------------------------------------------- #
# The depth entry point used by main_adt.py
# --------------------------------------------------------------------------- #

def depth_set_to_fisheye_attention(
    depths: Sequence[np.ndarray],
    view_params: Sequence[ViewParam],
    cam: FisheyeCam,
    attention_maps=None,
    view_valids: Optional[Sequence[np.ndarray]] = None,
    interp: str = "linear",
    min_weight: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fisheye analogue of upstream ``depth_set_to_equirect_attention``.

    ``depths`` are the per-view radial distances ``||world_points||`` from the
    VGGT point head (euclidean range along the ray — NOT planar z).  If
    ``attention_maps`` (the dict returned by the modified VGGT forward) is
    given, per-pixel correlation weights are derived exactly as in the paper;
    otherwise the fusion falls back to a uniform average (the ``--fuse mean``
    ablation).

    Returns ``(fused_range (H, W) float32, coverage (H, W) int32)`` on the
    fisheye grid defined by ``cam``.
    """
    weights = None
    if attention_maps is not None:
        w = build_selfview_confidence(attention_maps)[:, 0, :, :].cpu().numpy()
        weights = [w[i] for i in range(w.shape[0])]
    return fuse_views_to_fisheye(
        depths, view_params, cam,
        weights=weights, view_valids=view_valids,
        interp=interp, min_weight=min_weight,
    )
