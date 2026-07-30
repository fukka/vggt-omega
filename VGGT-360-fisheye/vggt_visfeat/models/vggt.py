# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt_visfeat.models.aggregator import Aggregator
from vggt_visfeat.heads.camera_head import CameraHead
from vggt_visfeat.heads.dpt_head import DPTHead

# NOTE(fisheye port): the point-track head (``track_head`` + ``heads/track_modules``
# + the ``vggt_visfeat/dependency`` vggsfm tree) has been removed — this port only
# consumes the camera / depth / point-map heads for depth fusion, and the tracker
# pulled in a large SfM dependency that was never exercised.  The pretrained
# checkpoint still carries ``track_head.*`` weights; ``from_pretrained`` below
# loads non-strictly so those extra keys are simply ignored.


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, is_all_frames=True):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Load non-strictly so the removed ``track_head.*`` checkpoint keys are
        ignored instead of raising.  Falls back gracefully if the installed
        ``huggingface_hub`` does not accept a ``strict`` kwarg."""
        kwargs.setdefault("strict", False)
        try:
            return super().from_pretrained(*args, **kwargs)
        except TypeError:
            kwargs.pop("strict", None)
            return super().from_pretrained(*args, **kwargs)

    def forward(self, images, persp_masks=None, rgb_masks=None, save_attn=True):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization
        """

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx, att = self.aggregator(
            images, persp_masks, rgb_masks, save_attn
        )

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, is_feats=False
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, is_feats=False
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf


        predictions["images"] = images

        return predictions, att
