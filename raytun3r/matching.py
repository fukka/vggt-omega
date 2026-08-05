"""Dense correspondences and the MAGSAC++ pose target.

The adaptation objective (Sec. 4.3) needs two things from an external matcher:

* ``m_ij(u,v)`` and ``w_ij(u,v)`` -- the matched pixel and its confidence, which
  drive the reprojection loss (Eq. 8). The paper uses UFM [44].
* a relative pose ``(R~_ij, t~_ij)`` estimated *once* with MAGSAC++ [45], used as
  a fixed pose target (Eq. 9). Because it is computed before adaptation from an
  external matcher, the adapter cannot influence its own pseudo-label.

Matchers are pluggable. ``UFMMatcher`` is the paper's choice and is used when
importable; ``RAFTMatcher`` and ``SIFTMatcher`` are fallbacks so that the loss,
the training loop and the tests run on a machine without UFM. Fallbacks change
the *quality* of supervision, not the method -- record which one produced a
number before comparing against the paper.

Pose estimation is done on **bearing vectors**, not pixels: on a 185-200 deg
fisheye there is no essential-matrix formulation in pixel space, so matches are
unprojected through the calibrated camera and the essential matrix is fitted to
the resulting rays.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .cameras import Camera, pixel_grid

__all__ = [
    "Matches",
    "Matcher",
    "UFMMatcher",
    "RAFTMatcher",
    "SIFTMatcher",
    "build_matcher",
    "relative_pose_magsac",
    "mean_flow_magnitude",
]


@dataclass
class Matches:
    """Dense correspondence field from image i to image j.

    ``target`` holds, for every pixel of image i, the matched pixel in image j;
    ``weight`` is the confidence, already zeroed outside the imaged disc.
    """

    target: Tensor        # (H, W, 2) pixel coordinates in image j
    weight: Tensor        # (H, W) confidence in [0, 1]

    def sample(self, max_points: int = 4096, *, generator=None) -> Tuple[Tensor, Tensor, Tensor]:
        """Confidence-weighted subsample -> ``(src_uv, dst_uv, weight)``."""
        h, w = self.weight.shape
        src = pixel_grid(h, w, device=self.weight.device, dtype=self.target.dtype)
        flat_w = self.weight.reshape(-1)
        keep = torch.nonzero(flat_w > 0, as_tuple=False).squeeze(-1)
        if keep.numel() > max_points:
            idx = torch.randperm(keep.numel(), generator=generator, device=keep.device)[:max_points]
            keep = keep[idx]
        return (src.reshape(-1, 2)[keep], self.target.reshape(-1, 2)[keep], flat_w[keep])


class Matcher:
    """Interface: given two ``(3, H, W)`` images in [0, 1], return :class:`Matches`."""

    name = "base"

    def __call__(self, image_i: Tensor, image_j: Tensor, *,
                 valid: Optional[Tensor] = None) -> Matches:
        raise NotImplementedError

    @staticmethod
    def _mask(weight: Tensor, valid: Optional[Tensor]) -> Tensor:
        return weight if valid is None else weight * valid.to(weight.dtype)


class UFMMatcher(Matcher):
    """The paper's matcher: UFM [44] (unified flow and matching)."""

    name = "ufm"

    def __init__(self, device="cpu", weights: Optional[str] = None):
        try:
            from uniflowmatch.models.ufm import UniFlowMatchConfidence  # type: ignore
        except ImportError as exc:               # pragma: no cover - optional dep
            raise ImportError(
                "UFM is not installed. Install it with "
                "`git clone --recursive https://github.com/UniFlowMatch/UFM` then "
                "`pip install -e UFM -e UFM/UniCeption`, or pass "
                "--matcher raft|sift to use a fallback."
            ) from exc
        self.model = UniFlowMatchConfidence.from_pretrained(
            weights or "infinity1096/UFM-Base").to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, image_i: Tensor, image_j: Tensor, *,
                 valid: Optional[Tensor] = None) -> Matches:
        # UFM normalises internally and needs to be told what it is being handed.
        # Images here are float32 in [0, 1], which is UniCeption's "identity"
        # normalisation; UFM then rescales to its encoder's own statistics.
        # Omitting this trips an assert, and passing uint8 instead would cost a
        # lossy round-trip through 8-bit.
        out = self.model.predict_correspondences_batched(
            source_image=image_i[None].to(self.device),
            target_image=image_j[None].to(self.device),
            data_norm_type="identity",
        )
        flow = out.flow.flow_output[0].permute(1, 2, 0)          # (H, W, 2)
        conf = out.covisibility.mask[0].to(flow.dtype)           # (H, W)
        h, w = conf.shape
        target = pixel_grid(h, w, device=flow.device, dtype=flow.dtype) + flow
        return Matches(target=target, weight=self._mask(conf, valid))


class RAFTMatcher(Matcher):
    """Optical-flow fallback using torchvision's RAFT.

    Flow is a dense but *non-robust* substitute for UFM: it has no covisibility
    estimate, so confidence is derived from a forward-backward consistency check
    instead, which is what the reprojection loss actually needs to downweight
    occlusions and out-of-frame matches.
    """

    name = "raft"

    def __init__(self, device="cpu", fb_threshold: float = 2.0):
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        self.weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=self.weights, progress=False).to(device).eval()
        self.device = device
        self.fb_threshold = fb_threshold

    @torch.no_grad()
    def _flow(self, a: Tensor, b: Tensor) -> Tensor:
        # RAFT wants [-1, 1] and sides divisible by 8.
        pad_h = (-a.shape[-2]) % 8
        pad_w = (-a.shape[-1]) % 8
        a2 = F.pad(a[None], (0, pad_w, 0, pad_h)) * 2 - 1
        b2 = F.pad(b[None], (0, pad_w, 0, pad_h)) * 2 - 1
        flow = self.model(a2.to(self.device), b2.to(self.device))[-1][0]
        return flow[:, : a.shape[-2], : a.shape[-1]].permute(1, 2, 0)

    @torch.no_grad()
    def __call__(self, image_i: Tensor, image_j: Tensor, *,
                 valid: Optional[Tensor] = None) -> Matches:
        fwd = self._flow(image_i, image_j)
        bwd = self._flow(image_j, image_i)
        h, w = fwd.shape[:2]
        grid = pixel_grid(h, w, device=fwd.device, dtype=fwd.dtype)
        target = grid + fwd

        # Forward-backward consistency: warp bwd to i's frame and re-add.
        norm = torch.stack((2 * target[..., 0] / max(w - 1, 1) - 1,
                            2 * target[..., 1] / max(h - 1, 1) - 1), dim=-1)
        bwd_at = F.grid_sample(bwd.permute(2, 0, 1)[None], norm[None],
                               mode="bilinear", padding_mode="border",
                               align_corners=True)[0].permute(1, 2, 0)
        err = (fwd + bwd_at).norm(dim=-1)
        conf = (1.0 - err / self.fb_threshold).clamp(0.0, 1.0)

        inside = ((target[..., 0] >= 0) & (target[..., 0] <= w - 1)
                  & (target[..., 1] >= 0) & (target[..., 1] <= h - 1))
        conf = conf * inside.to(conf.dtype)
        return Matches(target=target, weight=self._mask(conf, valid))


class SIFTMatcher(Matcher):
    """Sparse OpenCV SIFT + ratio test, densified by nearest-neighbour scatter.

    CPU-only and dependency-free, for smoke tests and for a first end-to-end run
    on a machine without a GPU. Sparse coverage makes it a weak signal for the
    reprojection loss; it is adequate for the MAGSAC++ pose target.
    """

    name = "sift"

    def __init__(self, device="cpu", ratio: float = 0.8, n_features: int = 4000):
        import cv2

        self.cv2 = cv2
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        self.ratio = ratio

    def _gray(self, img: Tensor) -> np.ndarray:
        a = (img.detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        return self.cv2.cvtColor(a, self.cv2.COLOR_RGB2GRAY)

    def __call__(self, image_i: Tensor, image_j: Tensor, *,
                 valid: Optional[Tensor] = None) -> Matches:
        cv2 = self.cv2
        gi, gj = self._gray(image_i), self._gray(image_j)
        ki, di = self.sift.detectAndCompute(gi, None)
        kj, dj = self.sift.detectAndCompute(gj, None)

        h, w = image_i.shape[-2:]
        target = pixel_grid(h, w, dtype=torch.float32)
        weight = torch.zeros(h, w)

        if di is None or dj is None or len(ki) < 2 or len(kj) < 2:
            return Matches(target=target, weight=self._mask(weight, valid))

        matches = cv2.BFMatcher().knnMatch(di, dj, k=2)
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance >= self.ratio * n.distance:
                continue
            u, v = ki[m.queryIdx].pt
            uu, vv = int(round(u)), int(round(v))
            if 0 <= uu < w and 0 <= vv < h:
                target[vv, uu, 0], target[vv, uu, 1] = kj[m.trainIdx].pt
                weight[vv, uu] = 1.0
        return Matches(target=target, weight=self._mask(weight, valid))


_MATCHERS = {"ufm": UFMMatcher, "raft": RAFTMatcher, "sift": SIFTMatcher}


def build_matcher(name: str = "auto", device="cpu", **kw) -> Matcher:
    """Build a matcher. ``auto`` prefers UFM, then RAFT, then SIFT."""
    if name != "auto":
        if name not in _MATCHERS:
            raise ValueError(f"unknown matcher {name!r}; choose from {sorted(_MATCHERS)} or 'auto'")
        return _MATCHERS[name](device=device, **kw)

    for candidate in ("ufm", "raft", "sift"):
        try:
            m = _MATCHERS[candidate](device=device, **kw)
        except Exception as exc:                  # pragma: no cover - env dependent
            warnings.warn(f"matcher {candidate!r} unavailable ({exc.__class__.__name__}); "
                          f"falling back", RuntimeWarning)
            continue
        if candidate != "ufm":
            warnings.warn(
                f"Using the {candidate!r} matcher, not UFM. The paper's numbers assume UFM; "
                f"record this when reporting results.", RuntimeWarning)
        return m
    raise RuntimeError("no matcher could be constructed")


# ---------------------------------------------------------------------------
# Pose target (Eq. 9)
# ---------------------------------------------------------------------------


def relative_pose_magsac(matches: Matches, camera: Camera, *,
                         threshold_deg: float = 0.5, confidence: float = 0.9999,
                         max_iters: int = 10000, min_matches: int = 8,
                         max_points: int = 4096) -> Optional[Tuple[Tensor, Tensor]]:
    """Fixed pose target ``(R~_ij, t~_ij)`` via MAGSAC++ on bearing vectors.

    Returns ``None`` when the match set is too small or degenerate, in which case
    the caller should drop the pose term for that pair rather than train on a
    meaningless target.

    ``t~`` is a unit translation *direction*: the essential matrix determines
    translation only up to scale, which is all Eq. 9 uses.
    """
    import cv2

    src, dst, w = matches.sample(max_points=max_points)
    if src.shape[0] < min_matches:
        return None

    # Pixels -> unit bearings, then to a normalised plane so OpenCV's essential
    # solver (which assumes a pinhole with K = I) applies to any lens.
    bi = camera.unproject(src)
    bj = camera.unproject(dst)
    front = (bi[:, 2] > 1e-3) & (bj[:, 2] > 1e-3)
    if int(front.sum()) < min_matches:
        return None
    pi = (bi[front, :2] / bi[front, 2:3]).detach().cpu().numpy().astype(np.float64)
    pj = (bj[front, :2] / bj[front, 2:3]).detach().cpu().numpy().astype(np.float64)

    # An angular threshold expressed on the normalised plane.
    thr = math.tan(math.radians(threshold_deg))
    E, inliers = cv2.findEssentialMat(
        pi, pj, cameraMatrix=np.eye(3), method=cv2.USAC_MAGSAC,
        prob=confidence, threshold=thr, maxIters=max_iters,
    )
    if E is None or E.shape[0] < 3:
        return None
    E = E[:3, :3]

    n_in, R, t, _ = cv2.recoverPose(E, pi, pj, cameraMatrix=np.eye(3), mask=inliers)
    if n_in < min_matches:
        return None

    R_t = torch.as_tensor(R, dtype=torch.float32)
    t_t = torch.as_tensor(t.reshape(3), dtype=torch.float32)
    n = t_t.norm()
    if not torch.isfinite(R_t).all() or n < 1e-8:
        return None
    return R_t, t_t / n


def mean_flow_magnitude(matches: Matches) -> float:
    """Mean confident optical-flow displacement, in pixels.

    Used to drop nearly-static windows from the adaptation set (Sec. 5,
    "Evaluation protocol"). Limitation (v) in the paper: with degenerate motion
    the self-supervised constraints go weak, because large depth or
    translation-direction errors induce only small reprojection errors.
    """
    h, w = matches.weight.shape
    grid = pixel_grid(h, w, device=matches.target.device, dtype=matches.target.dtype)
    disp = (matches.target - grid).norm(dim=-1)
    wsum = matches.weight.sum()
    if float(wsum) <= 0:
        return 0.0
    return float((disp * matches.weight).sum() / wsum)
