"""Fisheye3R model adaptation: calibration tokens + masked attention (paper Sec. 3.2, 3.4).

Mapping of the paper's modules onto VGGT-Omega:
  E (image encoder)        -> aggregator.patch_embed (DinoVisionTransformer, L1 = 24 blocks);
                              tokens are inserted into the last L1 - L0 blocks (L0 = 12).
  F (frame-wise attention) -> aggregator.frame_blocks (24 blocks).
  G (global attention)     -> aggregator.inter_frame_blocks (24 blocks; VGGT-Omega runs a
                              few of them on camera/register tokens only - "register" type -
                              we treat those as G layers too).

Per layer, K learnable d-dim calibration tokens are *prepended* to the token
sequence, participate in that layer's attention, and are dropped immediately
after (Eq. 6-8: each layer has its own token set, localizing the effect).
Prepending (instead of the paper's appending) is semantically identical -
attention is permutation-invariant over the non-RoPE prefix - and keeps
VGGT-Omega's RoPE convention valid, which requires patch tokens to be the
trailing H*W tokens of the sequence.

Masked attention (Eq. 14-15) controls the tokens per frame so one network
serves mixed perspective/fisheye sequences:
  - frame-level attention batches are split by camera type (exactly equivalent
    to the paper's block mask, since frames are independent sequences there);
  - global attention uses a two-pass trick: one pass with calibration keys and
    one without, selecting per query row by camera type. This is exactly
    Eq. 15 - keys are identical across passes, so row-wise selection
    reproduces the masked result - without materializing the (SN+K)^2 mask.

Only the calibration tokens (and optionally the tiny camera-type classifier)
are trainable; the backbone stays frozen (Sec. 3.2, "the original backbone is
entirely frozen").
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

from vggt_omega.models.aggregator import slice_expand_and_flatten
from vggt_omega.models.vggt_omega import VGGTOmega


class Fisheye3R(nn.Module):
    def __init__(
        self,
        base: VGGTOmega,
        num_tokens: int = 8,          # K (paper Fig. 7: K=8 default, K=1 already competitive)
        encoder_skip_layers: int = 12,  # L0 (paper Fig. 7: classification saturates at 12)
        modules: str = "EFG",         # which modules receive tokens (paper Tab. 4)
        token_init_std: float = 1e-6,  # supp: N(0, 1e-6)
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if base.dense_head is None or base.camera_head is None:
            raise ValueError("Fisheye3R needs the depth and camera heads of VGGT-Omega")

        self.base = base
        self.base.requires_grad_(False)
        self.num_tokens = num_tokens
        self.encoder_skip_layers = encoder_skip_layers
        self.modules_adapted = modules.upper()
        self.grad_checkpoint = grad_checkpoint

        agg = base.aggregator
        enc = agg.patch_embed
        embed_dim = agg.camera_token.shape[-1]
        n_enc_blocks = len(enc.blocks)
        if encoder_skip_layers >= n_enc_blocks:
            raise ValueError(f"L0={encoder_skip_layers} must be < encoder depth {n_enc_blocks}")

        def make_tokens(n_layers: int) -> nn.Parameter:
            return nn.Parameter(torch.randn(n_layers, num_tokens, embed_dim) * token_init_std)

        self.enc_tokens = make_tokens(n_enc_blocks - encoder_skip_layers) if "E" in self.modules_adapted else None
        self.frame_tokens = make_tokens(agg.depth) if "F" in self.modules_adapted else None
        self.global_tokens = make_tokens(agg.depth) if "G" in self.modules_adapted else None

        # Camera-type classifier psi (Eq. 14): logistic regression on the
        # encoder's class token taken after L0 blocks. Trained separately.
        self.camera_classifier = nn.Linear(embed_dim, 1)

    # ------------------------------------------------------------------ utils

    def trainable_parameters(self):
        for name in ("enc_tokens", "frame_tokens", "global_tokens"):
            p = getattr(self, name)
            if p is not None:
                yield p

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def _autocast(self, device: torch.device):
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.autocast(device_type="cuda", dtype=dtype)
        return contextlib.nullcontext()

    def _run_block(self, blk, x, rope):
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(lambda t: blk(t, rope), x, use_reentrant=False)
        return blk(x, rope)

    # ------------------------------------------------ token-adapted attention

    def _frame_level(self, blk, x, rope, tokens, flags_flat):
        """Frame-level block (encoder or F). x: (N_seq, T, d), flags: (N_seq,) bool.

        Masked attention via batch split: each sequence belongs entirely to one
        camera type here, so blocking the calibration keys for a perspective
        frame (Eq. 15, M_F) is the same as not appending tokens to it at all.
        """
        if tokens is None or not flags_flat.any():
            return self._run_block(blk, x, rope)
        K = tokens.shape[0]
        if flags_flat.all():
            xa = torch.cat([tokens.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype), x], dim=1)
            return self._run_block(blk, xa, rope)[:, K:]
        out = torch.empty_like(x)
        fi = flags_flat.nonzero(as_tuple=True)[0]
        pi = (~flags_flat).nonzero(as_tuple=True)[0]
        xa = torch.cat([tokens.unsqueeze(0).expand(fi.numel(), -1, -1).to(x.dtype), x[fi]], dim=1)
        out[fi] = self._run_block(blk, xa, rope)[:, K:]
        out[pi] = self._run_block(blk, x[pi], rope)
        return out

    def _global_level(self, blk, x_seq, tokens, flags, tokens_per_frame):
        """Global block (G). x_seq: (B, S*T, d), flags: (B, S) bool.

        Two-pass equivalent of the paper's global mask M_G (Eq. 15): image
        tokens of fisheye frames read from the pass that includes calibration
        keys, perspective frames from the pass without them.
        """
        if tokens is None or not flags.any():
            return self._run_block(blk, x_seq, None)
        B = x_seq.shape[0]
        K = tokens.shape[0]
        xa = torch.cat([tokens.unsqueeze(0).expand(B, -1, -1).to(x_seq.dtype), x_seq], dim=1)
        out_full = self._run_block(blk, xa, None)[:, K:]
        if flags.all():
            return out_full
        out_plain = self._run_block(blk, x_seq, None)
        row_flags = (
            flags.unsqueeze(-1).expand(B, flags.shape[1], tokens_per_frame).reshape(B, -1, 1)
        )
        return torch.where(row_flags, out_full, out_plain)

    # ------------------------------------------------------------- encoder (E)

    def _encode(self, images_flat, flags_flat):
        """DINOv2/v3 encoder with per-layer calibration tokens on blocks > L0.

        Returns (patch_tokens (N_seq, HW, d), cls_l0 (N_seq, d)) where cls_l0
        is the class token after L0 blocks used by the camera classifier
        (paper Sec. 3.4: features before any calibration influence).
        """
        enc = self.base.aggregator.patch_embed
        if getattr(enc, "untie_cls_and_patch_norms", False) or getattr(enc, "untie_global_and_local_cls_norm", False):
            raise NotImplementedError("untied encoder norms are not supported")

        x, (H, W) = enc.prepare_tokens_with_masks(images_flat)
        cls_l0 = None
        for i, blk in enumerate(enc.blocks):
            rope = enc.rope_embed(H=H, W=W) if enc.rope_embed is not None else None
            if i == self.encoder_skip_layers:
                cls_l0 = x[:, 0]
            if self.enc_tokens is not None and i >= self.encoder_skip_layers:
                x = self._frame_level(blk, x, rope, self.enc_tokens[i - self.encoder_skip_layers], flags_flat)
            else:
                x = self._run_block(blk, x, rope)
        if cls_l0 is None:  # L0 == depth would skip the capture point
            cls_l0 = x[:, 0]
        x = enc.norm(x)
        return x[:, enc.n_storage_tokens + 1 :], cls_l0

    # ---------------------------------------------------------- aggregator (F/G)

    def _aggregate(self, images, flags):
        """Re-implementation of Aggregator.forward with calibration tokens."""
        agg = self.base.aggregator
        B, S, C, H, W = images.shape
        flags_flat = flags.reshape(B * S)

        images_n = (images - agg._resnet_mean) / agg._resnet_std
        images_n = images_n.view(B * S, C, H, W)

        camera_token = slice_expand_and_flatten(agg.camera_token, B, S)
        register_token = slice_expand_and_flatten(agg.register_token, B, S)

        patch_tokens, cls_l0 = self._encode(images_n, flags_flat)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, N, D = tokens.shape
        pts = agg.patch_token_start

        grid = (H // agg.patch_size, W // agg.patch_size)
        with torch.no_grad():
            sin, cos = agg.rope_embed(H=grid[0], W=grid[1])
            frame_rope = (
                sin.to(device=tokens.device, dtype=torch.float32),
                cos.to(device=tokens.device, dtype=torch.float32),
            )

        outputs = []
        for idx in range(agg.depth):
            # F: frame-wise attention.
            tokens = tokens.view(B * S, N, D)
            tokens = self._frame_level(
                agg.frame_blocks[idx],
                tokens,
                frame_rope,
                self.frame_tokens[idx] if self.frame_tokens is not None else None,
                flags_flat,
            )
            frame_tokens_out = tokens.view(B, S, N, D)

            # G: inter-frame attention (global over all tokens, or over the
            # camera/register subsequence for "register"-type blocks).
            g_tok = self.global_tokens[idx] if self.global_tokens is not None else None
            attn_type = agg.inter_frame_attention_types[idx]
            t = tokens.view(B, S, N, D)
            if attn_type == "global":
                seq = t.reshape(B, S * N, D)
                seq = self._global_level(agg.inter_frame_blocks[idx], seq, g_tok, flags, N)
                tokens = seq.view(B, S, N, D)
            else:  # register
                head = t[:, :, :pts].reshape(B, S * pts, D)
                tail = t[:, :, pts:]
                head = self._global_level(agg.inter_frame_blocks[idx], head, g_tok, flags, pts)
                tokens = torch.cat([head.view(B, S, pts, D), tail], dim=2)

            if idx in agg.cached_layer_indices:
                outputs.append(torch.cat([frame_tokens_out, tokens], dim=-1))
            else:
                outputs.append(None)

        return outputs, pts, cls_l0

    # ---------------------------------------------------------------- forward

    @torch.no_grad()
    def l0_class_features(self, images: torch.Tensor) -> torch.Tensor:
        """Class token after the first L0 encoder blocks, (B, S, d).

        These features precede any calibration influence, so the camera-type
        decision does not depend on the tokens it gates (paper Sec. 3.4).
        """
        if images.dim() == 4:
            images = images.unsqueeze(0)
        B, S = images.shape[:2]
        agg = self.base.aggregator
        images_n = ((images - agg._resnet_mean) / agg._resnet_std).view(B * S, *images.shape[2:])
        enc = agg.patch_embed
        with self._autocast(images.device):
            x, (H, W) = enc.prepare_tokens_with_masks(images_n)
            for blk in enc.blocks[: self.encoder_skip_layers]:
                rope = enc.rope_embed(H=H, W=W) if enc.rope_embed is not None else None
                x = blk(x, rope)
        return x[:, 0].float().view(B, S, -1)

    @torch.no_grad()
    def predict_camera_type(self, images: torch.Tensor) -> torch.Tensor:
        """Eq. 14: per-frame fisheye logit from the frozen L0 features."""
        feats = self.l0_class_features(images)
        return self.camera_classifier(feats).squeeze(-1)

    def forward(
        self,
        images: torch.Tensor,
        fisheye_flags: torch.Tensor | None = None,
        use_classifier: bool = False,
    ) -> dict[str, torch.Tensor]:
        """images: (B, S, 3, H, W) in [0, 1].

        fisheye_flags: (B, S) bool - which frames get calibration tokens. When
        None: all frames are treated as fisheye unless use_classifier is set,
        in which case Eq. 14 decides per frame.
        """
        if images.dim() == 4:
            images = images.unsqueeze(0)
        B, S = images.shape[:2]
        if fisheye_flags is None:
            if use_classifier:
                fisheye_flags = self.predict_camera_type(images) > 0.0
            else:
                fisheye_flags = torch.ones(B, S, dtype=torch.bool, device=images.device)
        fisheye_flags = fisheye_flags.to(images.device)

        with self._autocast(images.device):
            aggregated, pts, cls_l0 = self._aggregate(images, fisheye_flags)

        preds: dict[str, torch.Tensor] = {"cls_l0": cls_l0, "fisheye_flags": fisheye_flags}
        with torch.autocast(device_type=images.device.type, enabled=False):
            preds["pose_enc"] = self.base.camera_head(aggregated, patch_token_start=pts)
            depth, conf = self.base.dense_head(aggregated, images=images, patch_token_start=pts)
            preds["depth"] = depth.squeeze(-1) if depth.shape[-1] == 1 else depth
            preds["depth_conf"] = conf.squeeze(-1) if conf.shape[-1] == 1 else conf
        return preds

    # ------------------------------------------------------------ persistence

    def save_tokens(self, path: str) -> None:
        state = {
            "num_tokens": self.num_tokens,
            "encoder_skip_layers": self.encoder_skip_layers,
            "modules": self.modules_adapted,
            "enc_tokens": self.enc_tokens,
            "frame_tokens": self.frame_tokens,
            "global_tokens": self.global_tokens,
            "camera_classifier": self.camera_classifier.state_dict(),
        }
        torch.save(state, path)

    def load_tokens(self, path: str, map_location="cpu") -> None:
        state = torch.load(path, map_location=map_location, weights_only=False)
        for name in ("enc_tokens", "frame_tokens", "global_tokens"):
            saved = state.get(name)
            if (saved is None) != (getattr(self, name) is None):
                raise ValueError(f"token set {name} mismatch between checkpoint and model config")
            if saved is not None:
                getattr(self, name).data.copy_(saved.data)
        self.camera_classifier.load_state_dict(state["camera_classifier"])
