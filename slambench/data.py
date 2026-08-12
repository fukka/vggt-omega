# Copyright (c) 2026.
"""Reading ego-synth 5B, in the form the producer captured it.

Semi-dense MPS SLAM points projected into every frame of four egocentric Aria
datasets (``aea``, ``nymeria``, ``egoexo4d``, ``oxford``): 1 611 takes, 24 931
clips, 380 GiB. Format, provenance and the ten gotchas are in
`docs/data/ego-synth-5b-sparse-depth.md`, and every guard below cites the one it
implements.

**Only the fisheye point set is read.** The release also carries the same points
projected into a 110 deg rectified pinhole, and this module deliberately ignores
it: rectifying is a thing a *baseline* may choose to do, so consuming a
pre-rectified ground truth would bake one baseline's choice into the measurement
and leave the raw baseline scored on someone else's resampling. The rectified
stream has exactly one legitimate use here — as an independent check on the
camera model, in ``verify_camera.py`` — and that is not scoring.

Gather, never scatter
---------------------
The ground truth is a point list, not a map, and it must stay one. ``u``/``v``
are float16 and quantise to half a pixel above 512, so scattering the points into
an 896 array collides them — 5 292 points onto 4 150 distinct pixels on the frame
the data card measured, last write wins, ~20 % lost. The metrics are per-point
anyway, so gathering the *prediction* at the point list is both lossless and less
code. Nothing here ever builds a dense ground truth.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from slambench import _REPO  # noqa: F401  (import registers sys.path)

#: Everything in this release was resized to 896x896 for Wan-5B video training
#: (gotcha 1). The source Aria frames are 1408x1408, and the calibration is at
#: the sensor's own resolution again -- see ``camera.load``.
RES: int = 896

#: The four datasets that have sparse depth. ``data_hot3d_5b`` exists but its
#: sparse depth is 0 bytes, and ``data_combined_5b`` only symlinks these four.
DATASETS: Tuple[str, ...] = ("aea", "nymeria", "egoexo4d", "oxford")

#: The stream this evaluation scores against. Named once so that the choice is
#: visible rather than spelled into a dozen f-strings.
VARIANT: str = "fisheye"

#: Default cut on ``inv_dist_std`` -- MPS inverse-distance 1-sigma in 1/metres,
#: which is scale-invariant triangulation quality. The release ships
#: **unfiltered on purpose** so the consumer chooses; this is the choice, and the
#: runner writes it into ``results.json`` so no number can be read without it.
DEFAULT_SIGMA_MAX: float = 0.01

#: A frame with fewer surviving points than this is not scored: the alignment
#: affine is fitted over the whole frame, so this is a floor on the fit.
MIN_FRAME_POINTS: int = 256


@dataclass
class Take:
    """One recording's clips, and the metadata they share."""

    dataset: str
    name: str
    path: str
    clips: List[str] = field(default_factory=list)
    #: clip id -> frame count, from ``meta.clips[].num_frames`` rather than
    #: assumed. Every clip in the release measured 121, but a short clip must
    #: shorten its own frame list rather than be discovered as a decode failure.
    clip_frames: Dict[str, int] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.name}"

    def frames_in(self, clip: str) -> int:
        return int(self.clip_frames.get(clip, 0))

    def npz(self, clip: str) -> str:
        return os.path.join(self.path, "sparse_depth", f"{clip}.npz")

    def video(self, clip: str) -> str:
        return os.path.join(self.path, VARIANT, f"{clip}.mp4")


def find_takes(root: str, datasets: Sequence[str] = DATASETS,
               limit: int = 0, verbose: bool = True) -> List[Take]:
    """Takes under ``root`` that carry sparse depth, in a stable order.

    Filtered on the presence of ``sparse_depth/*.npz``, never on the directory
    listing: ``egoexo4d`` has 1 090 populated take dirs out of 2 380 (gotcha 9),
    so a listing-based walk hands back 1 290 takes with no ground truth.

    ``limit`` caps the takes taken *per dataset*, 0 for all. The cap is a real
    subsample, so it is recorded in the manifest and enters the split digest.
    """
    out: List[Take] = []
    for ds in datasets:
        ds_dir = os.path.join(root, ds)
        if not os.path.isdir(ds_dir):
            raise SystemExit(
                f"[slambench] --egosynth-root {root!r} has no {ds!r} directory. "
                f"Expected one of {DATASETS} (note that 'data_combined_5b' only "
                f"symlinks these and 'data_hot3d_5b' has no sparse depth).")
        found: List[Take] = []
        for name in sorted(os.listdir(ds_dir)):
            path = os.path.join(ds_dir, name)
            meta_path = os.path.join(path, "meta.json")
            if not os.path.isfile(meta_path):
                continue
            clips = sorted(
                (os.path.basename(p)[:-4]
                 for p in glob.glob(os.path.join(path, "sparse_depth", "*.npz"))),
                key=_clip_sort_key)
            if not clips:
                continue
            with open(meta_path) as fh:
                meta = json.load(fh)
            n_by_clip = {str(c["clip_name"]): int(c["num_frames"])
                         for c in meta.get("clips", ())}
            clips = [c for c in clips if c in n_by_clip]
            if not clips:
                continue
            found.append(Take(dataset=ds, name=name, path=path, clips=clips,
                              clip_frames={c: n_by_clip[c] for c in clips}))
        kept = found[:limit] if limit else found
        if verbose:
            note = f"{len(kept)} of {len(found)} takes" if limit else f"{len(found)} takes"
            print(f"  [slambench] {ds}: {note}, "
                  f"{sum(len(t.clips) for t in kept)} clips")
        out.extend(kept)
    if not out:
        raise SystemExit(f"[slambench] no take under {root!r} has sparse_depth/*.npz")
    return out


def _clip_sort_key(clip: str) -> Tuple[int, str]:
    """Clip ids are the start frame index in the source recording, so sort them
    numerically -- ``'10000'`` before ``'2096'`` is the wrong order for a stride."""
    return (int(clip), clip) if clip.isdigit() else (1 << 62, clip)


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #

@dataclass
class FramePoints:
    """The ground truth of one (clip, frame): a point list, never a map."""

    u: np.ndarray             # (N,) float32, pixel x in the 896 fisheye frame
    v: np.ndarray             # (N,) float32, pixel y
    d: np.ndarray             # (N,) float32 metres, planar z about the camera axis
    inv_dist_std: np.ndarray  # (N,) float32 1/m, scale-invariant triangulation quality
    dist_std: np.ndarray      # (N,) float32 m, the depth's own 1-sigma

    def __len__(self) -> int:
        return int(self.u.size)

    @property
    def index(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(vi, ui)`` int32, clipped -- the gather index into an 896 map.

        ``np.rint`` alone is not enough: ``u`` reaches exactly 895.5 and rint
        rounds half to even, so ``rint(895.5) == 896``, one past the end of an
        896 array (gotcha 2). The clip is a fix, not tidiness.
        """
        ui = np.clip(np.rint(self.u), 0, RES - 1).astype(np.int32)
        vi = np.clip(np.rint(self.v), 0, RES - 1).astype(np.int32)
        return vi, ui


def read_card(npz_path: str) -> dict:
    """The npz's own ``meta`` key -- the authority on that file's conventions.

    The data card describes the release; this is the file's statement about
    itself, and the card says to read it before trusting the card.
    """
    z = np.load(npz_path, allow_pickle=True)
    return json.loads(str(z["meta"][0]))


def read_points(npz_path: str, frame: int,
                sigma_max: Optional[float] = DEFAULT_SIGMA_MAX) -> FramePoints:
    """One frame's fisheye points, as float32 and ready to score.

    Casting out of float16 first is not cosmetic: the stored depth is ~0.05 %
    relative (~5 mm at 10 m, ~6 cm at the 120 m cap, gotcha 3) and float16
    arithmetic on top of that is needlessly lossy. Every metric downstream is
    computed in float64.

    ``sigma_max`` cuts on ``inv_dist_std``, the scale-invariant column. The npz
    meta is explicit that it is *not* ``1/dist_std`` and not derivable from
    ``dist_std`` and depth, so both columns are carried and only this one is cut
    on. ``None`` keeps every point.
    """
    z = np.load(npz_path, allow_pickle=True)
    sel = z[f"{VARIANT}_frame"].astype(np.int32) == int(frame)
    uvd = z[f"{VARIANT}_uvd"][sel].astype(np.float32)
    inv = z[f"{VARIANT}_inv_dist_std"][sel].astype(np.float32)
    dst = z[f"{VARIANT}_dist_std"][sel].astype(np.float32)

    if sigma_max is not None and uvd.size:
        keep = np.isfinite(inv) & (inv < float(sigma_max))
        uvd, inv, dst = uvd[keep], inv[keep], dst[keep]
    if not uvd.size:
        z0 = np.zeros(0, np.float32)
        return FramePoints(z0, z0.copy(), z0.copy(), z0.copy(), z0.copy())
    return FramePoints(u=uvd[:, 0], v=uvd[:, 1], d=uvd[:, 2],
                       inv_dist_std=inv, dist_std=dst)


# --------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------- #

def decode_frames(path: str, indices: Iterable[int]) -> Dict[int, np.ndarray]:
    """Decode the given frame indices of one clip mp4 to RGB uint8.

    Read sequentially rather than seeking. A clip is 121 inter-coded frames, so
    ``CAP_PROP_POS_FRAMES`` seeking is both slower in aggregate and -- on some
    builds -- off by a frame or two, which would silently score a prediction
    against the *next* frame's points. Sequential decode cannot do that.

    Frames are already upright (``meta.args.pose_orientation = "upright"``), so
    unlike this repo's ADT loaders there is no 270 deg rotation here (gotcha 5).
    """
    import cv2
    want = sorted({int(i) for i in indices})
    if not want:
        return {}
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"[slambench] cannot open {path!r} — is OpenCV built "
                         f"with video support?")
    out: Dict[int, np.ndarray] = {}
    try:
        last, i = want[-1], 0
        while i <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if i in want:
                out[i] = np.ascontiguousarray(frame[..., ::-1])   # BGR -> RGB
            i += 1
    finally:
        cap.release()
    missing = [i for i in want if i not in out]
    if missing:
        raise SystemExit(f"[slambench] {path!r} ended before frame(s) {missing}; "
                         f"the clip is short or truncated")
    return out


def sample_prediction(pred: np.ndarray, pts: FramePoints) -> np.ndarray:
    """A predicted depth **map**, gathered at the point list.

    At the ground truth's own 896 grid this is literally ``pred[vi, ui]``, the
    protocol the data card states. When the model ran at its own token grid
    instead -- so that nothing is resampled between the frame and the network --
    the points are mapped into that grid on the pixel-centre convention
    ``x' = (x + 0.5) * s - 0.5`` and sampled bilinearly.

    The direction matters and is the whole reason this function exists rather
    than its inverse: scattering the ground truth into a map would drop ~20 % of
    the points to pixel collisions and pick an arbitrary winner among those that
    collide.
    """
    if pred.ndim != 2:
        raise ValueError(f"expected an (H, W) depth map, got shape {pred.shape}")
    H, W = pred.shape
    if (H, W) == (RES, RES):
        vi, ui = pts.index
        return pred[vi, ui].astype(np.float32)

    sx, sy = W / float(RES), H / float(RES)
    x = np.clip((pts.u + 0.5) * sx - 0.5, 0, W - 1)
    y = np.clip((pts.v + 0.5) * sy - 0.5, 0, H - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    fx, fy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
    p = pred.astype(np.float32)
    top = p[y0, x0] * (1 - fx) + p[y0, x1] * fx
    bot = p[y1, x0] * (1 - fx) + p[y1, x1] * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)


def resize_frame(rgb: np.ndarray, out_size: int) -> np.ndarray:
    """The 896 frame at a model's own token grid, or untouched if it already is."""
    if out_size == rgb.shape[0] == rgb.shape[1]:
        return rgb
    import cv2
    return np.clip(cv2.resize(rgb.astype(np.float32), (out_size, out_size),
                              interpolation=cv2.INTER_AREA), 0, 255).astype(np.uint8)
