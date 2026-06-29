# Copyright (c) 2026.
"""UniK3D baseline predictor (non-rectified Aria fisheye).

UniK3D (Piccinelli et al., CVPR 2025, https://github.com/lpiccinelli-eth/UniK3D)
is a universal *metric* monocular model: given an RGB frame and, optionally, a
camera, it predicts a dense 3D point map (hence metric depth + rays). It natively
supports a ``Fisheye624`` camera, so we feed the **raw Aria fisheye** frame plus
the Aria KB4 intrinsics — no rectification.

Loading mirrors the repo's ``hubconf.py`` (config json + HF ``pytorch_model.bin``)
with a ``from_pretrained`` fast-path. Weights download from
``lpiccinelli/unik3d-{vits,vitb,vitl}`` on first use.

``UniK3DPredictor.predict(rgb_uint8_hwc, cam)`` returns metric depth ``[H,W]``
(planar z, ``out["depth"] = points[:, 2]``) in the input fisheye frame — the same
"depth" convention the VGGT/DAv2 ADT eval uses, so the numbers are comparable.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

import numpy as np
import torch

from .aria_fisheye import AriaFisheye


def _add_repo_to_path(repo_root: Optional[str]) -> str:
    """Resolve + sys.path-insert the UniK3D repo root."""
    root = repo_root or os.environ.get("UNIK3D_ROOT")
    if root is None:
        # default: <vggt-omega>/third_party/UniK3D (in-repo; created by setup_baselines.sh)
        here = os.path.dirname(os.path.abspath(__file__))
        repo = os.path.dirname(os.path.dirname(os.path.dirname(here)))  # -> vggt-omega
        root = os.path.join(repo, "third_party", "UniK3D")
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"UniK3D repo not found at {root!r}. Run "
            f"`bash finetune/eval/baselines/setup_baselines.sh unik3d` to clone+install "
            f"it under the repo, or pass --unik3d-root / set $UNIK3D_ROOT."
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


class UniK3DPredictor:
    """Wrap a pretrained UniK3D model behind a simple per-frame predict()."""

    def __init__(
        self,
        backbone: str = "vitl",
        device: Optional[torch.device] = None,
        repo_root: Optional[str] = None,
        resolution_level: int = 9,
        interpolation_mode: str = "bilinear",
        use_camera: bool = True,
    ) -> None:
        self.root = _add_repo_to_path(repo_root)
        from unik3d.models import UniK3D as _UniK3D

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.use_camera = use_camera
        self.backbone = backbone

        model = None
        if hasattr(_UniK3D, "from_pretrained"):
            try:
                model = _UniK3D.from_pretrained(f"lpiccinelli/unik3d-{backbone}")
            except Exception as e:  # noqa: BLE001 - fall back to manual hub load
                print(f"[unik3d] from_pretrained failed ({e}); using hubconf path")
        if model is None:
            import huggingface_hub
            with open(os.path.join(self.root, "configs", f"config_{backbone}.json")) as f:
                config = json.load(f)
            model = _UniK3D(config)
            path = huggingface_hub.hf_hub_download(
                repo_id=f"lpiccinelli/unik3d-{backbone}",
                filename="pytorch_model.bin", repo_type="model")
            info = model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
            print(f"[unik3d] loaded unik3d-{backbone} "
                  f"(missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)})")

        model.resolution_level = resolution_level
        model.interpolation_mode = interpolation_mode
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def _infer(self, rgb_uint8_hwc: np.ndarray, camera) -> dict:
        """Core: rgb ``[H,W,3]`` uint8 + a UniK3D camera (or None) → outputs dict.

        Returns ``{"depth": [H,W] planar z, "points": [H,W,3], "rgb": [H,W,3] uint8}``,
        all at the input resolution / frame.
        """
        rgb = torch.from_numpy(np.ascontiguousarray(rgb_uint8_hwc)).permute(2, 0, 1)
        out = self.model.infer(rgb=rgb, camera=camera, normalize=True, rays=None)
        depth = out["depth"][0, 0].float().cpu().numpy()           # [H,W] planar z
        points = out["points"][0].permute(1, 2, 0).float().cpu().numpy()  # [H,W,3]
        return {"depth": depth, "points": points, "rgb": rgb_uint8_hwc}

    def predict(self, rgb_uint8_hwc: np.ndarray, cam: AriaFisheye) -> dict:
        """rgb ``[H,W,3]`` uint8 + Aria intrinsics → outputs dict (see ``_infer``).

        ``cam`` must describe the *exact* frame passed in (size + rotation already
        applied). When ``use_camera`` is False the camera is withheld and UniK3D
        predicts the rays itself (zero-shot, no intrinsics) — useful to isolate the
        benefit of supplying the fisheye model.
        """
        from unik3d.utils.camera import Fisheye624

        camera = None
        if self.use_camera:
            params = torch.tensor(cam.fisheye624_params(), dtype=torch.float32)
            camera = Fisheye624(params=params)
        return self._infer(rgb_uint8_hwc, camera)

    def predict_official(self, image_path: str, camera_json: Optional[str] = None) -> dict:
        """Run the repo's own example: an image + its ``{name, params}`` camera json
        (the camera models live in ``unik3d.utils.camera``). Sanity-checks the model
        + weights + environment independently of the Aria pipeline."""
        from PIL import Image as _Image
        from unik3d.utils.camera import (MEI, OPENCV, Fisheye624,  # noqa: F401
                                         Pinhole, Spherical)

        rgb = np.array(_Image.open(image_path).convert("RGB"))
        camera = None
        if camera_json and os.path.isfile(camera_json):
            with open(camera_json) as f:
                cd = json.load(f)
            camera = eval(cd["name"])(params=torch.tensor(cd["params"], dtype=torch.float32))
            print(f"[unik3d] official camera: {cd['name']}")
        return self._infer(rgb, camera)
