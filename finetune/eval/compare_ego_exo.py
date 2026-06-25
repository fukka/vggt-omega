# Copyright (c) 2026.
"""Pretrained VGGT-Omega: ego-only vs ego+exo, on Ego-Exo4D (depth + point cloud).

A *qualitative probe of the released (pretrained) checkpoint* — NOT the finetuned
LoRA. For a few "examples" it runs the model twice on the SAME egocentric frames:

    run A (ego-only)   : N ego (Aria 214-1 RGB) frames
    run B (ego + exo)  : the same N ego frames  +  K exocentric (GoPro) frames

Ego frames are kept FIRST so frame 0 (the privileged reference / world-scale
anchor) is the same ego frame in both runs and per-frame ego depth is directly
comparable at indices ``[0:N]``.

Data layout (same root finetuning uses; see configs/base.yaml ``data_root``)::

    <root>/<take>/aria01_214-1/000000.jpg ...   <- ego  (clip_pattern "*214-1")
                  cam01/000000.jpg ...          <- exo  (GoPro; frame-aligned)
                  cam02/ ...  cam03/ ...  cam04/ ...
                  aria01_1201-1/ , _1201-2/     <- Aria SLAM grey (ignored)

These are the *same* per-camera frame folders finetuning reads — training just
keeps the ego stream via ``clip_pattern: "*214-1"`` and drops the exo ``cam*``
folders. So ``--take-dir`` defaults to the finetuning ``data_root``; if that root
holds many takes, each "example" is a different take (scene diversity); if it
points at a single take, each example is a temporal window. Raw ``frame_aligned_
videos/*.mp4`` takes and explicit ``--ego-video/--exo-videos`` (e.g. ADT) also work.

Defaults feed RECTIFIED ego frames (Aria KB4 fisheye->pinhole) and RECTIFIED+CROPPED exo frames
(GoPro KB3 undistort, then center-cropped to the ego aspect so the stack needs no padding).

Outputs, under ``<out>/example_xx/``, with ego-only and ego+exo saved SEPARATELY::

    ego_only/inputs/frame_*.png       exact frames fed to VGGT (rectified ego)
    ego_only/depth/frame_*.png        VGGT depth (colorised)
    ego_only/predictions.npz          depth, depth_conf, pose_enc, extrinsic, intrinsic, valid
    ego_only/points.ply (+ .glb)      point cloud
    ego_plus_exo/inputs/frame_*.png   frames fed to VGGT (rectified ego + rectified+cropped exo)
    ego_plus_exo/depth/frame_*.png    VGGT depth for all frames
    ego_plus_exo/predictions.npz      raw predictions
    ego_plus_exo/points.ply (+.glb)   cloud (all frames); points_ego.ply = ego frames only
    compare/frame_*.png               ego depth: ego-only | ego+exo | scale-aligned rel-diff

Ego-Exo4D has no dense depth GT, so the printed summary reports *change* (global
ego-depth scale ratio B/A and median scale-aligned structural change), not error.
For real numbers use the Aria MPS semi-dense cloud as sparse GT (see mps_depth.py).

Usage
-----
    python -m finetune.eval.compare_ego_exo --checkpoint /path/vggt_omega_1b_512.pt
    # ^ --take-dir defaults to configs/base.yaml data_root; defaults: 8 ego frames,
    #   all 4 exo cams, 1 frame/cam. Override any of them, or point elsewhere:
    python -m finetune.eval.compare_ego_exo --checkpoint ... \
        --take-dir /data/egoexo4d/takes/<take> --num-ego 8 --num-exo-cams 4
"""
from __future__ import annotations

import sys as _sys, os as _os
if not __package__:
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
    __package__ = "finetune.eval"

import argparse
import glob
import os
import re
from fnmatch import fnmatch
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------- #
# Camera unprojection (copied from demo_gradio so we don't import gradio)
# --------------------------------------------------------------------------- #
def unproject_depth_map_to_point_map(depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """depth [S,H,W,1] + (cam-from-world) extrinsic [S,3,4] + intrinsic [S,3,3] -> world points [S,H,W,3]."""
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))
    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]
    camera_points = np.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1)
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum("sij,shwj->shwi", np.transpose(rotation, (0, 2, 1)),
                     camera_points - translation[:, None, None, :])


# --------------------------------------------------------------------------- #
# Camera frame sources: a directory of pre-extracted frames OR an mp4.
# (The finetuning root is per-camera frame folders; raw Ego-Exo4D/ADT are mp4.)
# --------------------------------------------------------------------------- #
def _list_image_files(folder: str) -> List[str]:
    def nat(s):
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]
    files = [f for f in os.listdir(folder) if f.lower().endswith(_IMG_EXTS)]
    files.sort(key=nat)
    return [os.path.join(folder, f) for f in files]


def _has_images(folder: str) -> bool:
    return os.path.isdir(folder) and any(f.lower().endswith(_IMG_EXTS) for f in os.listdir(folder))


class FrameSource:
    """One camera. ``paths_at(indices)`` -> ordered frame paths on disk."""

    def __init__(self, path: str, name: str):
        self.path, self.name = path, name
        self.is_dir = os.path.isdir(path)
        if self.is_dir:
            self.frames = _list_image_files(path)
            self.n = len(self.frames)
        else:
            cap = cv2.VideoCapture(path)
            self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.frames = None

    def paths_at(self, indices: List[int], out_dir: str, prefix: str) -> List[str]:
        idxs = [int(np.clip(i, 0, max(self.n - 1, 0))) for i in indices]
        if self.is_dir:                                  # frames already on disk — reuse them
            return [self.frames[i] for i in idxs] if self.n else []
        os.makedirs(out_dir, exist_ok=True)              # mp4 — decode the requested frames
        cap, paths = cv2.VideoCapture(self.path), []
        for k, i in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                print(f"[warn] could not read frame {i} from {os.path.basename(self.path)}")
                continue
            p = os.path.join(out_dir, f"{prefix}_{k:03d}_f{i:06d}.png")
            cv2.imwrite(p, frame)
            paths.append(p)
        cap.release()
        return paths


def _is_exo(name: str) -> bool:
    n = name.lower()
    if "aria" in n:
        return False
    if any(t in n for t in ("1201", "211-", "_et", "eyetrack", "slam", "imu", "audio")):
        return False
    return True


def find_camera_sources(take_dir: str, ego_pattern: str) -> Optional[Tuple[FrameSource, List[FrameSource]]]:
    """If ``take_dir`` is one take, return (ego, [exo]); else None. Frame-folder first, mp4 second."""
    if not os.path.isdir(take_dir):
        return None
    subs = [os.path.join(take_dir, x) for x in sorted(os.listdir(take_dir))
            if os.path.isdir(os.path.join(take_dir, x))]

    # (1) per-camera frame folders (the finetuning layout)
    ego_dirs = [s for s in subs if fnmatch(os.path.basename(s), ego_pattern)] or \
               [s for s in subs if "aria" in os.path.basename(s).lower() and "214-1" in os.path.basename(s)]
    if ego_dirs and _has_images(ego_dirs[0]):
        exo = [FrameSource(s, os.path.basename(s)) for s in subs
               if _is_exo(os.path.basename(s)) and _has_images(s)]
        return FrameSource(ego_dirs[0], os.path.basename(ego_dirs[0])), exo

    # (2) frame_aligned_videos/*.mp4 (raw Ego-Exo4D / ADT)
    fav = os.path.join(take_dir, "frame_aligned_videos")
    vids = sorted(glob.glob(os.path.join(fav if os.path.isdir(fav) else take_dir, "*.mp4")))
    ego_v = [v for v in vids if "214-1" in os.path.basename(v)] or \
            [v for v in vids if "aria" in os.path.basename(v).lower()]
    if ego_v:
        exo = [FrameSource(v, os.path.basename(v)) for v in vids if not _is_aria(v)]
        return FrameSource(ego_v[0], os.path.basename(ego_v[0])), exo
    return None


def _is_aria(p: str) -> bool:
    return "aria" in os.path.basename(p).lower()


def find_takes(root: str, ego_pattern: str) -> List[Tuple[str, str, FrameSource, List[FrameSource]]]:
    """``root`` may be a single take or a parent of many takes -> list of (name, take_dir, ego, [exo])."""
    one = find_camera_sources(root, ego_pattern)
    if one is not None:
        return [(os.path.basename(os.path.normpath(root)), root, one[0], one[1])]
    takes = []
    for child in sorted(os.listdir(root)):
        cd = os.path.join(root, child)
        src = find_camera_sources(cd, ego_pattern) if os.path.isdir(cd) else None
        if src is not None:
            takes.append((child, cd, src[0], src[1]))
    return takes


def ego_indices(center: int, num: int, span: int, total: int) -> List[int]:
    lo, hi = center - span // 2, center + span // 2
    return [int(np.clip(round(v), 0, max(total - 1, 0))) for v in np.linspace(lo, hi, num)]


def exo_indices(ego_idxs: List[int], frames_per_exo: int) -> List[int]:
    """Static exo cams: 1 frame at the ego-window CENTRE; >1 spreads across the window."""
    if frames_per_exo >= len(ego_idxs):
        return list(ego_idxs)
    if frames_per_exo <= 1:
        return [ego_idxs[len(ego_idxs) // 2]]
    sel = np.unique(np.linspace(0, len(ego_idxs) - 1, frames_per_exo).round().astype(int))
    return [ego_idxs[j] for j in sel]


def default_data_root() -> str:
    """Read configs/base.yaml ``data_root`` so the default matches finetuning."""
    try:
        import yaml
        with open(os.path.join(_REPO_ROOT, "finetune", "configs", "base.yaml")) as f:
            cfg = yaml.safe_load(f) or {}

        def find(d, key):
            if isinstance(d, dict):
                if key in d:
                    return d[key]
                for v in d.values():
                    r = find(v, key)
                    if r is not None:
                        return r
            return None

        return find(cfg, "data_root") or ""
    except Exception:
        return ""


def default_clip_pattern() -> str:
    try:
        import yaml
        with open(os.path.join(_REPO_ROOT, "finetune", "configs", "base.yaml")) as f:
            cfg = yaml.safe_load(f) or {}
        return (cfg.get("data", {}) or {}).get("clip_pattern") or "*214-1"
    except Exception:
        return "*214-1"


# --------------------------------------------------------------------------- #
# Exocentric GoPro rectification (Ego-Exo4D).
# Per the Ego-Exo4D "Undistort frames" tutorial, the exo GoPros use a Kannala-
# Brandt K3 *fisheye* model (NOT the Aria KB4, and NOT pinhole), with per-camera
# intrinsics in gopro_calibs.csv (intrinsics_0..7 = fx, fy, cx, cy, k1..k4). The
# frame_aligned exo videos are DISTORTED, so undistort with cv2.fisheye, scaling
# the calibration K from its reference resolution to the actual frame size.
#   https://docs.ego-exo4d-data.org/tutorials/undistort/
# NOTE: column names / file location / camera-id matching below are best-effort
# from the docs — verify against your gopro_calibs.csv (logged at runtime).
# --------------------------------------------------------------------------- #
class GoProRectifier:
    """cv2.fisheye KB3 undistorter for one GoPro; rescales K to the frame size; per-size map cache."""

    def __init__(self, fx, fy, cx, cy, dist, calib_wh, balance: float = 0.0):
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.D = np.asarray(dist, np.float64).reshape(4, 1)
        self.cw, self.ch = calib_wh
        self.balance = float(balance)
        self._maps: dict = {}

    def _maps_for(self, W: int, H: int):
        if (W, H) not in self._maps:
            import cv2
            sx = W / self.cw if self.cw else 1.0
            sy = H / self.ch if self.ch else 1.0
            K = np.array([[self.fx * sx, 0, self.cx * sx], [0, self.fy * sy, self.cy * sy], [0, 0, 1.0]], np.float64)
            newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, self.D, (W, H), np.eye(3), balance=self.balance)
            self._maps[(W, H)] = cv2.fisheye.initUndistortRectifyMap(K, self.D, np.eye(3), newK, (W, H), cv2.CV_16SC2)
        return self._maps[(W, H)]

    def __call__(self, img_hwc: np.ndarray) -> np.ndarray:
        import cv2
        H, W = img_hwc.shape[:2]
        m1, m2 = self._maps_for(W, H)
        return np.ascontiguousarray(cv2.remap(img_hwc, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), np.float32)


def _first(row: dict, keys, cast):
    for k in keys:
        if k in row and str(row[k]).strip() != "":
            try:
                return cast(row[k])
            except (TypeError, ValueError):
                pass
    return None


def load_gopro_calibs(csv_path: str, balance: float = 0.0) -> List[Tuple[str, GoProRectifier]]:
    """Parse gopro_calibs.csv -> ordered [(cam_name, GoProRectifier)] (one row per static GoPro)."""
    import csv as _csv
    out: List[Tuple[str, GoProRectifier]] = []
    with open(csv_path, newline="") as f:
        for row in _csv.DictReader(f):
            try:
                k = [float(row[f"intrinsics_{j}"]) for j in range(8)]
            except (KeyError, ValueError):
                continue
            w = _first(row, ["image_width", "width", "w"], float)
            h = _first(row, ["image_height", "height", "h"], float)
            name = _first(row, ["cam_name", "camera_name", "gopro_uid", "cam_uid", "uid", "serial", "name"], str) \
                or (next(iter(row.values()), "") if row else "")
            out.append((str(name), GoProRectifier(k[0], k[1], k[2], k[3], k[4:8], (w, h), balance)))
    return out


def find_gopro_calibs(take_dir: Optional[str], override: str) -> str:
    """Locate gopro_calibs.csv: explicit override (file or dir), else search take_dir and parents."""
    if override:
        if os.path.isfile(override):
            return override
        cand = os.path.join(override, "gopro_calibs.csv")
        if os.path.isfile(cand):
            return cand
    d = take_dir
    for _ in range(4):                       # walk up a few levels (frames may sit beside trajectory/)
        if not d or not os.path.isdir(d):
            break
        for sub in ("", "trajectory", "trajectory/gopro", "gopro", "mps"):
            cand = os.path.join(d, sub, "gopro_calibs.csv")
            if os.path.isfile(cand):
                return cand
        d = os.path.dirname(os.path.normpath(d))
    return ""


def make_exo_resolver(calibs: List[Tuple[str, GoProRectifier]]):
    """cam_name/index -> GoProRectifier (exact, then substring, then positional)."""
    by_name = {n.lower(): r for n, r in calibs}

    def resolve(cam_name: str, idx: int) -> Optional[GoProRectifier]:
        key = os.path.splitext(str(cam_name))[0].lower()
        if key in by_name:
            return by_name[key]
        for n, r in calibs:
            if n and (key in n.lower() or n.lower() in key):
                return r
        return calibs[idx][1] if idx < len(calibs) else None    # positional fallback (both sorted)

    return resolve


# --------------------------------------------------------------------------- #
# Model + inference (mirrors demo_gradio.run_model on the pretrained net)
# --------------------------------------------------------------------------- #
def load_pretrained(checkpoint: str, device: torch.device) -> torch.nn.Module:
    from vggt_omega.models import VGGTOmega
    print(f"[ego-exo] loading pretrained VGGT-Omega: {checkpoint}")
    model = VGGTOmega()
    sd = torch.load(checkpoint, map_location="cpu")
    if isinstance(sd, dict):
        sd = sd.get("model", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ego-exo]   loaded (missing={len(missing)}, unexpected={len(unexpected)})")
    return model.to(device).eval()


# ---- Size-mismatch handling: rectify ego, letterbox to a shared canvas, mask ---- #
# One VGGT forward stacks all frames into [S,3,H,W] with a SINGLE RoPE grid, so every
# frame must share H,W. Ego (704x704, 1:1) and exo (960x540, 16:9) differ, so we:
#   (1) rectify the ego fisheye -> pinhole (raw fisheye into a pinhole model is wrong);
#   (2) resize each frame to ~1024 tokens keeping its native aspect ('balanced'), so
#       NO anisotropic stretch (which would be out-of-distribution for the model);
#   (3) letterbox-pad every frame to a common canvas (shared across ego-only and
#       ego+exo runs so the ego depth maps line up pixel-for-pixel);
#   (4) carry a per-frame validity mask (true content rect, minus the ego's black
#       out-of-FoV corners) so padded / invalid pixels never enter the point cloud,
#       the depth diff, or the scale stats. The model still SEES the white border
#       (unavoidable without retraining) — we just never SCORE or PLOT it.
def _resample():
    from PIL import Image
    return getattr(Image, "Resampling", Image).BICUBIC


def _target_hw(aspect_ratio: float, image_resolution: int, mode: str, patch: int = 16):
    from vggt_omega.utils.load_fn import _balanced_target_shape, _max_size_target_shape
    fn = _balanced_target_shape if mode == "balanced" else _max_size_target_shape
    return fn(aspect_ratio, image_resolution, patch)


def _center_crop_to_aspect(arr: np.ndarray, target_ar: float) -> np.ndarray:
    """Center-crop a (rectified, pinhole) image to height/width aspect ``target_ar``.

    Valid ONLY after undistortion: cropping a pinhole image just shifts the principal point
    (cx,cy) and narrows the FoV — no distortion is introduced. (Cropping a *fisheye* image
    before undistort would be wrong.)
    """
    h, w = arr.shape[:2]
    ar = h / max(w, 1)
    if abs(ar - target_ar) < 1e-3:
        return arr
    if ar > target_ar:                                  # too tall -> crop height
        nh = max(1, int(round(w * target_ar))); y0 = (h - nh) // 2
        return arr[y0:y0 + nh]
    nw = max(1, int(round(h / target_ar))); x0 = (w - nw) // 2   # too wide -> crop width
    return arr[:, x0:x0 + nw]


def compute_canvas(paths: List[str], image_resolution: int, mode: str,
                   crop_aspects: Optional[List[Optional[float]]] = None) -> Tuple[int, int]:
    """Common (H, W) all frames are padded to = max target H and W over the set.

    A non-None ``crop_aspects[i]`` means frame i is center-cropped to that aspect first, so its
    effective aspect (and target size) is that value — used to size the shared canvas correctly.
    """
    from PIL import Image
    from vggt_omega.utils.load_fn import _crop_to_supported_aspect_ratio
    H = W = 0
    for i, p in enumerate(paths):
        ca = crop_aspects[i] if crop_aspects and i < len(crop_aspects) else None
        if ca is not None:
            ar = ca
        else:
            im = _crop_to_supported_aspect_ratio(Image.open(p).convert("RGB"))
            w, h = im.size
            ar = h / max(w, 1)
        th, tw = _target_hw(ar, image_resolution, mode)
        H, W = max(H, th), max(W, tw)
    return H, W


def preprocess(paths: List[str], rectifiers: List, image_resolution: int, mode: str,
               canvas_hw: Tuple[int, int], pad_value: float = 1.0,
               crop_aspects: Optional[List[Optional[float]]] = None):
    """Paths -> (images [S,3,H,W] float[0,1], valid [S,H,W] bool).

    ``rectifiers`` is per-frame and aligned with ``paths``: an ego FisheyeRectifier, a per-camera
    GoProRectifier, or None (treat as pinhole). Each maps HWC float[0,1] -> rectified HWC; calling
    it on a ones image yields the FoV mask (out-of-FoV -> 0 via BORDER_CONSTANT).
    ``crop_aspects[i]`` (post-rectify center-crop to that h/w aspect) lets exo be cropped to ego's
    aspect so the stack needs no letterbox padding.
    """
    from PIL import Image
    from vggt_omega.utils.load_fn import _crop_to_supported_aspect_ratio
    Hc, Wc = canvas_hw
    resample = _resample()
    imgs, masks = [], []
    for i, p in enumerate(paths):
        arr = np.asarray(_crop_to_supported_aspect_ratio(Image.open(p).convert("RGB")), np.float32) / 255.0
        fov = np.ones(arr.shape[:2], np.float32)
        rect = rectifiers[i] if i < len(rectifiers) else None
        if rect is not None:                                    # fisheye -> pinhole (+ FoV mask)
            fov = (rect(np.ones_like(arr))[..., 0] > 0.5).astype(np.float32)
            arr = rect(arr)
        ca = crop_aspects[i] if crop_aspects and i < len(crop_aspects) else None
        if ca is not None:                                      # post-rectify center crop (pinhole-valid)
            arr = _center_crop_to_aspect(arr, ca)
            fov = _center_crop_to_aspect(fov, ca)
        h, w = arr.shape[:2]
        th, tw = _target_hw(h / max(w, 1), image_resolution, mode)
        rim = np.asarray(Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((tw, th), resample),
                         np.float32) / 255.0
        rfov = np.asarray(Image.fromarray((fov * 255).astype(np.uint8)).resize((tw, th), Image.NEAREST),
                          np.float32) / 255.0
        canvas = np.full((Hc, Wc, 3), pad_value, np.float32)
        vmask = np.zeros((Hc, Wc), np.float32)
        y0, x0 = (Hc - th) // 2, (Wc - tw) // 2                  # centred letterbox
        canvas[y0:y0 + th, x0:x0 + tw] = rim
        vmask[y0:y0 + th, x0:x0 + tw] = rfov
        imgs.append(torch.from_numpy(canvas).permute(2, 0, 1))
        masks.append(torch.from_numpy(vmask > 0.5))
    return torch.stack(imgs), torch.stack(masks)


def run_vggt(model, images: torch.Tensor, device: torch.device) -> dict:
    """Run the model on a preprocessed [S,3,H,W] tensor -> numpy predictions dict (batch squeezed)."""
    from vggt_omega.utils.pose_enc import encoding_to_camera

    images = images.to(device)
    with torch.inference_mode():
        preds = model(images)
    extr, intr = encoding_to_camera(preds["pose_enc"], preds["images"].shape[-2:])
    preds["extrinsic"], preds["intrinsic"] = extr, intr

    out = {}
    for k, v in preds.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().float().cpu().numpy()
            if v.shape[0] == 1:
                v = v[0]
            out[k] = v
    out["world_points_from_depth"] = unproject_depth_map_to_point_map(
        out["depth"], out["extrinsic"], out["intrinsic"]
    )
    return out


# --------------------------------------------------------------------------- #
# Visualisation: depth panels + point clouds
# --------------------------------------------------------------------------- #
def _get_cmap(name: str):
    try:                                          # matplotlib >= 3.5
        from matplotlib import colormaps
        return colormaps.get_cmap(name)
    except Exception:                             # older matplotlib
        from matplotlib import cm
        try:
            return cm.get_cmap(name)
        except ValueError:
            return cm.get_cmap("viridis")


def colorize_depth(depth: np.ndarray, vmin: float, vmax: float, cmap: str = "turbo") -> np.ndarray:
    d = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    return (_get_cmap(cmap)(d)[..., :3] * 255).astype(np.uint8)


def save_compare_panels(rgb_chw: np.ndarray, depth_a: np.ndarray, depth_b: np.ndarray,
                        valid: np.ndarray, out_dir: str) -> None:
    """Per ego frame: [RGB | ego-only depth | ego+exo depth (scale-aligned) | signed rel-diff].

    ``rgb_chw`` is the preprocessed ego image [n,3,H,W] (aligned with the depth canvas);
    ``valid`` [n,H,W] bool masks out letterbox padding and the ego's black FoV corners.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    for i in range(depth_a.shape[0]):
        da, db, m = depth_a[i, ..., 0], depth_b[i, ..., 0], valid[i]
        if not m.any():
            continue
        vmin, vmax = np.percentile(da[m], 2), np.percentile(da[m], 98)
        s = np.median(da[m]) / max(np.median(db[m]), 1e-6)     # median-align B->A: isolate structure
        rel = (db * s - da) / np.maximum(0.5 * (db * s + da), 1e-6)
        rel = np.where(m, rel, np.nan)                         # don't draw padded/invalid pixels

        rgb = np.transpose(rgb_chw[i], (1, 2, 0))
        fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
        ax[0].imshow(np.clip(rgb, 0, 1));             ax[0].set_title("ego RGB (rectified)")
        ax[1].imshow(colorize_depth(da, vmin, vmax)); ax[1].set_title("depth: ego-only")
        ax[2].imshow(colorize_depth(db, vmin, vmax)); ax[2].set_title("depth: ego+exo")
        im = ax[3].imshow(rel, cmap="coolwarm", vmin=-0.3, vmax=0.3)
        ax[3].set_title(f"rel-diff (scale-aligned)\nglobal scale B/A = {1.0/max(s,1e-6):.3f}")
        for a in ax:
            a.axis("off")
        fig.colorbar(im, ax=ax[3], fraction=0.046)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"frame_{i:03d}.png"), dpi=110)
        plt.close(fig)


def _images_to_rgb(images: np.ndarray) -> np.ndarray:
    return np.transpose(images, (0, 2, 3, 1)) if images.ndim == 4 and images.shape[1] == 3 else images


def masked_points(pred: dict, conf_thres: float, frame_slice: Optional[slice], max_points: int,
                  valid: Optional[np.ndarray] = None):
    """Confidence/edge-filtered world points + RGB colours, optionally restricted to some frames.

    ``valid`` [S,H,W] bool (letterbox + ego-FoV mask) zeroes confidence on padded/invalid pixels
    so they never enter the cloud.
    """
    pts, conf, imgs = pred["world_points_from_depth"], pred["depth_conf"].astype(np.float32).copy(), _images_to_rgb(pred["images"])
    try:
        from visual_util import depth_edge
        conf[depth_edge(pred["depth"][..., 0], rtol=0.03)] = 0.0
    except Exception:
        pass
    if valid is not None:
        conf = conf * valid.astype(np.float32)
    if frame_slice is not None:
        pts, conf, imgs = pts[frame_slice], conf[frame_slice], imgs[frame_slice]

    v = pts.reshape(-1, 3)
    c = (imgs.reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)
    f = conf.reshape(-1)
    mask = np.isfinite(v).all(axis=1) & np.isfinite(f)
    if conf_thres > 0 and mask.any():
        mask &= f >= np.percentile(f[mask], conf_thres)
    mask &= f > 1e-5
    v, c = v[mask], c[mask]
    if 0 < max_points < len(v):
        idx = np.linspace(0, len(v) - 1, max_points).astype(np.int64)
        v, c = v[idx], c[idx]
    return v, c


def save_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Minimal binary PLY writer (no deps) — opens in MeshLab/CloudCompare/Open3D."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = len(points)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n").encode("ascii")
    verts = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    verts["x"], verts["y"], verts["z"] = points[:, 0], points[:, 1], points[:, 2]
    verts["r"], verts["g"], verts["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(verts.tobytes())
    print(f"[ego-exo]   point cloud -> {path}  ({n} pts)")


def maybe_save_glb(pred: dict, conf_thres: float, path: str) -> None:
    try:
        from visual_util import predictions_to_glb
        predictions_to_glb(pred, conf_thres=conf_thres, show_cam=True).export(path)
        print(f"[ego-exo]   GLB -> {path}")
    except Exception as e:
        print(f"[ego-exo]   (skipped GLB: {e})")


def save_run_outputs(run_dir: str, imgs: torch.Tensor, pred: dict, valid: np.ndarray,
                     conf_thres: float, max_points: int, ego_n: Optional[int] = None) -> None:
    """Save one run (ego-only OR ego+exo) self-contained: the exact model inputs, the VGGT depth,
    the raw predictions, and the point cloud.

        <run_dir>/inputs/frame_*.png   rectified (ego) / rectified+cropped (exo) frames fed to VGGT
        <run_dir>/depth/frame_*.png    colorised predicted depth (all frames, invalid masked black)
        <run_dir>/predictions.npz      depth, depth_conf, pose_enc, extrinsic, intrinsic, valid
        <run_dir>/points.ply (+.glb)   confidence/edge/validity-filtered cloud (all frames)
        <run_dir>/points_ego.ply       ego-frames-only cloud (ego+exo run; ego cloud *with* exo context)
    """
    os.makedirs(run_dir, exist_ok=True)
    arr = imgs.numpy() if isinstance(imgs, torch.Tensor) else imgs
    S = arr.shape[0]

    in_dir = os.path.join(run_dir, "inputs")
    dep_dir = os.path.join(run_dir, "depth")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(dep_dir, exist_ok=True)
    depth = pred["depth"][..., 0]
    for i in range(S):
        rgb = (np.clip(np.transpose(arr[i], (1, 2, 0)), 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(in_dir, f"frame_{i:03d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        dd, m = depth[i], valid[i]
        lo, hi = (np.percentile(dd[m], 2), np.percentile(dd[m], 98)) if m.any() else (0.0, 1.0)
        col = colorize_depth(dd, lo, hi) * m[..., None]
        cv2.imwrite(os.path.join(dep_dir, f"frame_{i:03d}.png"), cv2.cvtColor(col, cv2.COLOR_RGB2BGR))

    np.savez_compressed(
        os.path.join(run_dir, "predictions.npz"),
        depth=pred["depth"], depth_conf=pred["depth_conf"], pose_enc=pred["pose_enc"],
        extrinsic=pred["extrinsic"], intrinsic=pred["intrinsic"], valid=valid,
    )

    pred_masked = dict(pred)                                  # also exclude invalid from the GLB
    pred_masked["depth_conf"] = pred["depth_conf"] * valid.astype(np.float32)
    v, c = masked_points(pred, conf_thres, None, max_points, valid)
    save_ply(os.path.join(run_dir, "points.ply"), v, c)
    maybe_save_glb(pred_masked, conf_thres, os.path.join(run_dir, "points.glb"))
    if ego_n is not None and ego_n < S:
        ve, ce = masked_points(pred, conf_thres, slice(0, ego_n), max_points, valid)
        save_ply(os.path.join(run_dir, "points_ego.ply"), ve, ce)
    print(f"[ego-exo]   run saved -> {run_dir}/ (inputs/ depth/ predictions.npz points.ply)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(
        description="Pretrained VGGT-Omega ego-only vs ego+exo on Ego-Exo4D (depth + point cloud)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, help="pretrained VGGT-Omega .pt (e.g. vggt_omega_1b_512.pt)")
    p.add_argument("--take-dir", default=default_data_root(),
                   help="finetuning data_root by default (a take, or a parent of takes); "
                        "ego=clip_pattern subdir, exo=cam*/gp* subdirs, or frame_aligned_videos/*.mp4")
    p.add_argument("--ego-video", help="explicit ego video/dir (use with --exo-videos; for ADT/custom rigs)")
    p.add_argument("--exo-videos", nargs="*", default=[], help="explicit exo videos/dirs (with --ego-video)")
    p.add_argument("--out", default="eval_out/egoexo_ego_vs_exo")
    # Defaults justified in the README/answer: 8 ego = training seq_len; window 16 = seq_len*stride
    # (stride 2); 4 exo cams = the full Ego-Exo4D GoPro surround; 1 frame/cam because the exo cams
    # are STATIC (extra frames add no new baseline, only dynamic-content/redundancy).
    p.add_argument("--num-ego", type=int, default=8, help="ego frames per example (= training seq_len)")
    p.add_argument("--window-frames", type=int, default=16, help="span ego frames cover (= seq_len*stride)")
    p.add_argument("--num-exo-cams", type=int, default=4, help="exo cameras added in run B (full surround)")
    p.add_argument("--frames-per-exo", type=int, default=1, help="frames per (static) exo cam, at the ego-window centre")
    p.add_argument("--num-examples", type=int, default=3, help="takes (or windows, for a single take) to probe")
    p.add_argument("--image-resolution", type=int, default=512)
    # Size-mismatch handling (ego 704x704 fisheye vs exo 960x540): keep native aspect and
    # letterbox to a shared canvas. 'balanced' (default) spreads padding evenly (~25% each);
    # 'max_size' leaves ego unpadded but wastes ~44% of the exo frame.
    p.add_argument("--preprocess", choices=["balanced", "max_size"], default="balanced",
                   help="aspect-preserving target sizing before letterbox padding")
    p.add_argument("--pad-value", type=float, default=1.0, help="letterbox pad value (1.0=white, the library default)")
    p.add_argument("--exo-crop", choices=["none", "ego"], default="ego",
                   help="'ego' (default)=center-crop exo to ego's aspect after undistort (no padding, "
                        "drops peripheral FoV); 'none'=letterbox exo (keep full FoV, ~25%% padding)")
    p.add_argument("--no-rectify-ego", action="store_true",
                   help="do NOT rectify the ego fisheye (default: Aria KB4 -> pinhole, the model's assumption)")
    p.add_argument("--ego-fisheye-preset", default="aria-214-1",
                   help="FisheyeRectifier preset for the ego stream (data/rectify.py)")
    p.add_argument("--no-rectify-exo", action="store_true",
                   help="do NOT rectify exo GoPro (default: KB3 fisheye undistort from gopro_calibs.csv)")
    p.add_argument("--gopro-calibs", default="",
                   help="gopro_calibs.csv path (or its dir); default: auto-search the take dir")
    p.add_argument("--exo-balance", type=float, default=0.0,
                   help="cv2.fisheye undistort balance for exo (0=no black border/cropped, 1=keep all FoV)")
    p.add_argument("--conf-thres", type=float, default=20.0, help="percentile conf threshold for point clouds")
    p.add_argument("--max-points", type=int, default=200_000)
    p.add_argument("--clip-pattern", default=default_clip_pattern(), help="ego camera glob (default from base.yaml)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    device = torch.device(a.device)
    if device.type != "cuda":
        print("[ego-exo] WARNING: VGGT-Omega forward uses CUDA autocast; CPU/MPS is slow (smoke only).")

    # Ego (Aria KB4) fisheye -> pinhole. Exo GoPro uses a DIFFERENT model (KB3) and per-camera
    # calibration (gopro_calibs.csv), built later per take. The model assumes pinhole; raw fisheye
    # is wrong at the periphery for both.
    ego_rect = None
    if not a.no_rectify_ego:
        from finetune.data.rectify import FisheyeRectifier
        ego_rect = FisheyeRectifier(preset=a.ego_fisheye_preset)
        print(f"[ego-exo] ego rectification: {a.ego_fisheye_preset} (Aria KB4 fisheye->pinhole)")

    # Resolve takes: explicit ego/exo override, else discover under --take-dir.
    if a.ego_video:
        ego_src = FrameSource(a.ego_video, os.path.basename(a.ego_video))
        exo_src = [FrameSource(v, os.path.basename(v)) for v in a.exo_videos]
        takes = [(os.path.basename(os.path.normpath(a.ego_video)), None, ego_src, exo_src)]
    else:
        if not a.take_dir:
            p.error("no --take-dir and configs/base.yaml has no data_root; pass --take-dir or --ego-video/--exo-videos.")
        if not os.path.isdir(a.take_dir):
            p.error(f"--take-dir {a.take_dir!r} is not a directory (finetuning data_root may be a remote path; "
                    f"pass a local --take-dir or --ego-video/--exo-videos).")
        takes = find_takes(a.take_dir, a.clip_pattern)
        if not takes:
            p.error(f"no Ego-Exo4D takes found under {a.take_dir!r} "
                    f"(need <take>/{a.clip_pattern}/ frames or frame_aligned_videos/*.mp4).")

    # Examples: many takes -> one centred window each (scene diversity);
    # a single take -> several temporal windows.
    if len(takes) == 1:
        name, take_dir0, ego_src, exo_all = takes[0]
        fracs = np.linspace(0.15, 0.85, a.num_examples)
        examples = [(name, take_dir0, ego_src, exo_all, int(fr * max(ego_src.n - 1, 1))) for fr in fracs]
    else:
        examples = [(nm, td, es, xa, es.n // 2) for (nm, td, es, xa) in takes[:a.num_examples]]

    print(f"[ego-exo] take-dir: {a.take_dir or '(explicit videos)'}")
    print(f"[ego-exo] {len(examples)} example(s); per run B: {a.num_ego} ego + "
          f"<= {a.num_exo_cams} exo cam x {a.frames_per_exo} frame")

    model = load_pretrained(a.checkpoint, device)
    os.makedirs(a.out, exist_ok=True)
    summary = []

    for ex, (name, take_dir0, ego_src, exo_all, center) in enumerate(examples):
        exo_srcs = exo_all[: a.num_exo_cams]
        ex_dir = os.path.join(a.out, f"example_{ex:02d}_{re.sub(r'[^A-Za-z0-9_-]', '_', name)[:40]}")
        frames_dir = os.path.join(ex_dir, "frames")
        ego_idxs = ego_indices(center, a.num_ego, a.window_frames, ego_src.n)
        exo_idxs = exo_indices(ego_idxs, a.frames_per_exo)
        print(f"\n[ego-exo] ===== example {ex}: take={name!r}  ego idxs={ego_idxs}  "
              f"exo={[s.name for s in exo_srcs] or 'NONE'} @ idxs={exo_idxs} =====")

        # Per-camera GoPro (KB3) rectifiers for this take, from gopro_calibs.csv.
        exo_resolve = None
        if not a.no_rectify_exo and exo_srcs:
            calib_csv = find_gopro_calibs(take_dir0, a.gopro_calibs)
            if calib_csv:
                calibs = load_gopro_calibs(calib_csv, a.exo_balance)
                exo_resolve = make_exo_resolver(calibs)
                print(f"[ego-exo]   exo rectification: {len(calibs)} GoPro(s) from {calib_csv} (KB3 fisheye->pinhole)")
            else:
                print("[ego-exo]   WARN: no gopro_calibs.csv found — exo left UNRECTIFIED (distorted GoPro fed as "
                      "pinhole). Pass --gopro-calibs <file|dir>, or --no-rectify-exo to silence.")

        ego_paths = ego_src.paths_at(ego_idxs, frames_dir, "ego")
        if not ego_paths:
            print("[ego-exo]   no ego frames; skipping.")
            continue
        exo_paths, exo_rects = [], []
        for ci, es in enumerate(exo_srcs):
            ps = es.paths_at(exo_idxs, frames_dir, f"exo{ci}")
            exo_paths += ps
            exo_rects += [exo_resolve(es.name, ci) if exo_resolve else None] * len(ps)

        n = len(ego_paths)
        # ONE shared canvas (from the ego+exo set) for BOTH runs, so the ego depth maps
        # line up pixel-for-pixel and only the added exo tokens differ between A and B.
        all_paths = ego_paths + exo_paths
        rects_a = [ego_rect] * n                          # run A: ego only
        rects_b = [ego_rect] * n + exo_rects              # run B: ego (KB4) + per-cam exo (KB3)
        # Optional: crop exo to ego's aspect (post-undistort) so no letterbox padding is needed.
        crop_a = crop_b = None
        if a.exo_crop == "ego" and exo_paths:
            from PIL import Image as _Img
            ew, eh = _Img.open(ego_paths[0]).size         # rectify keeps size; raw aspect is fine
            ego_ar = eh / max(ew, 1)
            crop_a = [None] * n
            crop_b = [None] * n + [ego_ar] * len(exo_paths)
        canvas = compute_canvas(all_paths, a.image_resolution, a.preprocess, crop_b)
        imgs_a, valid_a = preprocess(ego_paths, rects_a, a.image_resolution, a.preprocess, canvas, a.pad_value, crop_a)
        pred_a = run_vggt(model, imgs_a, device)
        if exo_paths:
            imgs_b, valid_b = preprocess(all_paths, rects_b, a.image_resolution, a.preprocess, canvas, a.pad_value, crop_b)
            pred_b = run_vggt(model, imgs_b, device)
        else:
            pred_b, valid_b = pred_a, valid_a
        valid_a_np, valid_b_np = valid_a.numpy(), valid_b.numpy()
        valid_ego = valid_a_np[:n]                                    # ego masks identical across runs

        depth_a, depth_b = pred_a["depth"][:n], pred_b["depth"][:n]   # ego frames are first in run B

        # Save each run self-contained and separate: inputs + depth + raw preds + cloud.
        save_run_outputs(os.path.join(ex_dir, "ego_only"), imgs_a, pred_a, valid_a_np,
                         a.conf_thres, a.max_points)
        if exo_paths:
            save_run_outputs(os.path.join(ex_dir, "ego_plus_exo"), imgs_b, pred_b, valid_b_np,
                             a.conf_thres, a.max_points, ego_n=n)
        # Side-by-side ego depth: ego-only vs ego+exo (+ scale-aligned rel-diff).
        save_compare_panels(imgs_a[:n].numpy(), depth_a, depth_b, valid_ego, os.path.join(ex_dir, "compare"))

        # Scale / structural-change stats over the VALID ego region only.
        da, db = depth_a[..., 0][valid_ego], depth_b[..., 0][valid_ego]
        med_a, med_b = float(np.median(da)), float(np.median(db))
        s = med_a / max(med_b, 1e-6)
        struct = float(np.median(np.abs((db * s - da) / np.maximum(0.5 * (db * s + da), 1e-6))))
        summary.append((name[:24], med_a, med_b, med_b / max(med_a, 1e-6), struct))
        print(f"[ego-exo]   ego median depth A={med_a:.3f}m B={med_b:.3f}m "
              f"(scale B/A={med_b/max(med_a,1e-6):.3f}); scale-aligned structural change={struct:.3f}")

    print(f"\n{'='*84}\n  pretrained VGGT-Omega — ego-only vs ego+exo (Ego-Exo4D)\n{'='*84}")
    print(f"  {'take':24s}{'medDepth A':>12s}{'medDepth B':>12s}{'scale B/A':>12s}{'struct chg':>12s}")
    for nm, ma, mb, sc, st in summary:
        print(f"  {nm:24s}{ma:>12.3f}{mb:>12.3f}{sc:>12.3f}{st:>12.3f}")
    print(f"\n[ego-exo] done. Per example under {a.out}/example_*/ : ego_only/ and ego_plus_exo/ "
          f"(each: inputs/ depth/ predictions.npz points.ply), plus compare/frame_*.png")


if __name__ == "__main__":
    main()
