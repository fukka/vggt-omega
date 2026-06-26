# Copyright (c) 2026.
"""Depth-Any-Camera (DAC) baseline predictor (non-rectified Aria fisheye).

DAC (Guo et al., CVPR 2025, https://github.com/yuliangguo/depth_any_camera) is a
metric model trained only on perspective images that generalises zero-shot to
fisheye/360 by operating in an **equirectangular (ERP)** representation: it warps
the input (fisheye) frame into an ERP patch using the camera intrinsics, predicts
metric depth there, and warps back. We feed the **raw Aria fisheye** frame plus
the Aria KB4 intrinsics as an ``OPENCV_FISHEYE`` camera — no rectification.

``DACPredictor.predict(img_hwc01, cam)`` returns the metric depth prediction on
the ERP ``fwd_sz`` grid (euclidean range) together with the ERP active mask and
the lat/long ranges — i.e. everything needed to score it against a GT warped to
the same grid via :func:`erp_utils.fisheye_to_erp_fwd`. The ``cano_sz`` /
``fwd_sz`` / ``crop_wFoV`` used are exposed so the caller warps GT identically.

Model + weights come from the repo's HF release
(``yuliangguo/depth-any-camera``): a ``*.json`` config (top-level ``model_name``,
``cano_sz``) and a ``*.pt`` checkpoint. See ``download_weights.py``.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import numpy as np
import torch

from .aria_fisheye import AriaFisheye
from .erp_utils import add_dac_to_path, fisheye_to_erp_fwd

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class DACPredictor:
    def __init__(
        self,
        config_file: str,
        model_file: str,
        device: Optional[torch.device] = None,
        repo_root: Optional[str] = None,
        fwd_sz: Tuple[int, int] = (500, 750),
        crop_wFoV: float = 180.0,
    ) -> None:
        self.root = add_dac_to_path(repo_root)
        from dac.models.idisc_erp import IDiscERP  # noqa: F401
        from dac.models.idisc import IDisc  # noqa: F401
        from dac.models.idisc_equi import IDiscEqui  # noqa: F401
        from dac.models.cnn_depth import CNNDepth  # noqa: F401

        with open(config_file) as f:
            config = json.load(f)
        self.model_name = config["model_name"]
        cano_sz = config["cano_sz"]
        self.cano = int(cano_sz[0])
        self.fwd_sz = tuple(fwd_sz)
        self.crop_wFoV = float(crop_wFoV)

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        model = eval(self.model_name).build(config)
        model.load_pretrained(model_file)
        self.model = model.to(self.device).eval()
        print(f"[dac] loaded {self.model_name} (cano={self.cano}, fwd_sz={self.fwd_sz}, "
              f"crop_wFoV={self.crop_wFoV}) from {os.path.basename(model_file)}")

    @torch.no_grad()
    def _predict(self, img_hwc01: np.ndarray, cam_params: dict) -> dict:
        """Core: RGB ``[H,W,3]`` in [0,1] + a DAC ``cam_params`` dict → ERP prediction.

        ``cam_params`` is any model DAC's ``cam_to_erp_patch_fast`` understands
        (OPENCV_FISHEYE for Aria; the official scannetpp sample uses the same).
        Returns the prediction on the ``fwd_sz`` ERP grid::

            depth   [fh,fw] float32  metric euclidean range (already × scale factor)
            active  [fh,fw] float32  in-FOV ERP region
            points  [fh,fw,3]        reconstructed 3D point map (ERP unprojection)
            image_u8[fh,fw,3] uint8  ERP RGB the network saw (for visualisation)
            lat_range / long_range   the IDiscERP geometry inputs
            cano / fwd_sz / crop_wFoV  warp params (warp GT with the same values)
        """
        from dac.utils.unproj_pcd import reconstruct_pcd_erp

        warp = fisheye_to_erp_fwd(
            img_hwc01, np.zeros(img_hwc01.shape[:2], np.float32),
            np.zeros(img_hwc01.shape[:2], np.float32),
            cam_params, self.cano, self.fwd_sz, self.crop_wFoV,
        )
        img = torch.from_numpy(warp["image_u8"]).permute(2, 0, 1).float() / 255.0
        img = ((img - torch.tensor(_IMAGENET_MEAN).view(3, 1, 1))
               / torch.tensor(_IMAGENET_STD).view(3, 1, 1)).unsqueeze(0).to(self.device)
        lat = warp["lat_range"].unsqueeze(0).to(self.device)
        lon = warp["long_range"].unsqueeze(0).to(self.device)

        if self.model_name == "IDiscERP":
            preds, _, _ = self.model(img, lat, lon)
        else:
            preds, _, _ = self.model(img)
        preds = preds * warp["pred_scale_factor"]
        depth = np.ascontiguousarray(preds[0, 0].float().cpu().numpy())  # [fh,fw] range
        active = warp["active"]
        # 3D points by unprojecting the ERP depth (euclidean range) over the patch.
        points = reconstruct_pcd_erp(
            depth.copy(), mask=(active > 0.5),
            lat_range=[float(warp["lat_range"][0]), float(warp["lat_range"][1])],
            long_range=[float(warp["long_range"][0]), float(warp["long_range"][1])])
        return {
            "depth": depth, "active": active, "points": points,
            "image_u8": warp["image_u8"],
            "lat_range": warp["lat_range"], "long_range": warp["long_range"],
            "cano": self.cano, "fwd_sz": self.fwd_sz, "crop_wFoV": self.crop_wFoV,
        }

    def predict(self, img_hwc01: np.ndarray, cam: AriaFisheye) -> dict:
        """fisheye RGB ``[H,W,3]`` in [0,1] + Aria cam → ERP depth prediction (see ``_predict``)."""
        return self._predict(img_hwc01, cam.opencv_fisheye_params())

    def predict_official(self, image_path: str, sample_json: str) -> dict:
        """Run the repo's own example: load the sample image + its ``cam_params``
        (e.g. ``demo/input/scannetpp_sample.json``, an OPENCV_FISHEYE camera) and
        push them through the same ERP path. Sanity-checks model + weights + env."""
        from PIL import Image as _Image

        with open(sample_json) as f:
            sample = json.load(f)
        img_path = image_path or os.path.join(os.path.dirname(sample_json),
                                               os.path.basename(sample["image_filename"]))
        img01 = np.asarray(_Image.open(img_path).convert("RGB"), np.float32) / 255.0
        print(f"[dac] official sample: {sample.get('dataset_name')} "
              f"({sample['cam_params'].get('camera_model')})")
        return self._predict(img01, dict(sample["cam_params"]))
