import cv2, numpy as np


def SA_confidence(rgb_persp, tau=0.5, margin=0.05, valid_mask=None):
    """Structure-aware confidence map (VGGT-360 module 2), fisheye-ported.

    Numerics are upstream-identical: Sobel gradient magnitude -> keep the top
    20% -> percentile normalise -> sigmoid around the median -> relax toward 1
    in an edge band (``edge_relaxation_window``) so cross-view attention is
    encouraged where neighbouring views overlap.  The map is injected into the
    frame attention as an additive log-bias (see ``layers/attention.py``).

    Fisheye change (the ONLY change vs upstream)
    --------------------------------------------
    ``valid_mask``: upstream detects invalid pixels as ``rgb.sum() > 0`` —
    fine on ERP where every pixel is imaged, but wrong on fisheye views whose
    cone-clipped corners are *geometrically* empty: a black-pixel test also
    fires on genuinely dark image content and misses noisy vignette pixels.
    Pass the analytic mask from ``fisheye_to_persp`` instead; it propagates as
    VGGT's ``rgb_mask`` so the attention bias actively suppresses keys from
    never-imaged regions.  Falls back to the upstream test when None.

    Returns
    -------
    M        : (H, W) float32 confidence in [0.10, 1].
    mask_rgb : (H, W) float32 validity in {0, 1}.
    """
    g = cv2.cvtColor(rgb_persp.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    mag = cv2.GaussianBlur(mag, (0,0), 1.0)
    p95 = np.percentile(mag, 80)
    mag_filtered = np.where(mag >= p95, mag, 0)

    p1, p99 = np.percentile(mag_filtered, 1), np.percentile(mag_filtered, 99)
    T = np.clip((mag_filtered - p1) / (p99 - p1 + 1e-6), 0, 1)

    t0 = float(np.median(T))
    M = 1.0 / (1.0 + np.exp(-(T - t0) / max(tau, 1e-6)))

    if valid_mask is not None:
        mask_rgb = (np.asarray(valid_mask) > 0.5).astype(np.float32)
    else:
        mask_rgb = (rgb_persp.sum(axis=-1) > 0).astype(np.float32)

    Wc = edge_relaxation_window(H=rgb_persp.shape[0], W=rgb_persp.shape[1], margin=margin)
    M = np.clip(M ** Wc, 0.10, 1.0)
    return M, mask_rgb


def edge_relaxation_window(H, W, margin=0.04):
    """Cosine window: ~1 at the frame border, 0 inside (upstream, verbatim).

    Used as an exponent on the confidence map — near the border ``M**~0 -> 1``
    regardless of texture, i.e. full confidence in the overlap band between
    neighbouring views, which is exactly where cross-view attention must be
    free to interact for seamless stitching.
    """
    yy = np.linspace(-1, 1, H)[:, None]
    xx = np.linspace(-1, 1, W)[None, :]
    d = np.maximum(np.abs(xx), np.abs(yy))
    t = (d - (1 - margin)) / max(margin, 1e-6)
    t = np.clip(t, 0.0, 1.0)

    W_center = 0.5 * (1 + np.cos(t * np.pi))
    return W_center.astype(np.float32)
