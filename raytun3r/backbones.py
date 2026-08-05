"""Frozen 3D foundation model backbones, and where RayTun3R hooks into them.

The paper evaluates DA3, pi^3 and VGGT. This module exposes three switchable
backbones -- ``vggt``, ``vggt_omega`` and ``da3`` -- behind one interface, so the
adapter, losses and evaluation are written once.

Four attachment points, matching Sec. 4.2:

===================  ==========================================================
Attachment           Where it lands
===================  ==========================================================
absolute PE (Eq. 5)  the ViT's positional-embedding table, before it is added
RoPE (Eq. 6)         every ``RopePositionEmbedding``'s ``(sin, cos)`` output
patch tokenization   a resample of the image just before ``patch_embed``, plus
                     mean-token fill of patches outside the lens circle
DPT grid             the dense head's ``_apply_pos_embed`` coordinate grid
===================  ==========================================================

**A structural caveat that decides what each backbone can show.** RayTun3R's
dominant term is the absolute-PE residual: the paper's Tab. 7(b) measures
0.48 deg rotation for the full adapter, 0.68 deg for absolute PE alone, and
19.52 deg for RoPE alone -- i.e. RoPE-only adaptation is close to no adaptation.

* ``vggt`` (vendored ``vggt_visfeat``) is DINOv2-based and *has* a learned
  ``pos_embed`` table. Both branches apply. This is the faithful target.
* ``vggt_omega`` (this repo's model) is DINOv3-based: RoPE only, no absolute PE
  table anywhere in the ViT. Only Eq. 6 applies, giving 20 trainable parameters.
  By the paper's own ablation this should land near the unadapted model -- it
  reproduces the "RoPE only" row, not the headline result. It is wired up so the
  claim can be *tested* here, not because it is expected to match Tab. 1.
* ``da3`` is the paper's primary backbone and needs the external
  ``depth_anything_3`` package; the import is guarded.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import corrections as C
from .adapter import RayTun3RAdapter
from .cameras import Camera

__all__ = ["Prediction", "Backbone", "build_backbone", "BACKBONES"]


@dataclass
class Prediction:
    """Per-frame geometry from a backbone, in a backbone-independent form.

    ``depth`` is planar z in the camera frame (see the repo's CONTEXT.md -- the
    VGGT-family depth heads emit planar z, not euclidean range). ``R``/``t`` are
    camera-from-world, so the relative pose of frame j w.r.t. i is
    ``R_j R_i^T`` and ``t_j - R_j R_i^T t_i``.

    ``covered`` marks where the method actually produced a depth. It is ``None``
    for a backbone that predicts everywhere; the virtual-pinhole baselines set it
    because they only cover part of the fisheye frame (Center-PH sees ~31% of a
    170 deg frame) and leave the rest at zero. Scoring those zeros as if they
    were predictions turns d_reproj into a coverage statistic -- a zero depth
    backprojects to the optical centre and reprojects onto the principal point,
    which on this data costs ~117 px against a real error of ~0.2 px.
    """

    depth: Tensor          # (S, H, W)
    conf: Tensor           # (S, H, W)
    R: Tensor              # (S, 3, 3)
    t: Tensor              # (S, 3)
    extra: dict = None
    covered: Optional[Tensor] = None    # (S, H, W) bool, or None for "everywhere"

    def relative(self, i: int, j: int) -> Tuple[Tensor, Tensor]:
        R_rel = self.R[j] @ self.R[i].transpose(-1, -2)
        t_rel = self.t[j] - R_rel @ self.t[i]
        return R_rel, t_rel


class Backbone(nn.Module):
    """Wraps a frozen model and owns the four attachment points."""

    #: filled in by subclasses
    patch_size: int = 14
    embed_dim: int = 1024
    has_abs_pe: bool = False
    has_rope: bool = False

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        self.adapter: Optional[RayTun3RAdapter] = None
        self.camera: Optional[Camera] = None
        self._handles: List[Any] = []
        self._patched: List[Tuple[nn.Module, str, Any]] = []
        self._grid: Optional[Tuple[int, int]] = None
        self._undistort_grid: Optional[Tensor] = None
        self._valid_patches: Optional[Tensor] = None
        # P_A on the current token grid, captured the first time the PE hook
        # fires. The TV regulariser (Eq. 12) is defined on P' = P_A + residual,
        # not on the residual alone, so the pretrained table is needed.
        self._pe_table: Optional[Tensor] = None
        self.use_patch_undistort = False
        self.use_border_token = False
        self.use_dpt_grid = False

    # -- construction -----------------------------------------------------

    def make_adapter(self, *, n_radial: int = 20, n_angular: int = 8,
                     n_rope_radial: int = 20) -> RayTun3RAdapter:
        return RayTun3RAdapter(
            self.embed_dim,
            n_radial=n_radial, n_angular=n_angular, n_rope_radial=n_rope_radial,
            use_abs_pe=self.has_abs_pe, use_rope=self.has_rope,
        )

    def install(self, adapter: Optional[RayTun3RAdapter], camera: Camera,
                image_hw: Tuple[int, int], *, patch_undistort: bool = True,
                border_token: bool = True, dpt_grid: bool = True,
                preserve_scale: bool = True, grid_mode: str = "auto") -> "Backbone":
        """Attach the adapter and the parameter-free corrections.

        ``adapter=None`` installs only the parameter-free parts, which is how the
        "Param.-free only" row of Tab. 8 is produced.
        """
        self.remove()
        h, w = image_hw
        if h % self.patch_size or w % self.patch_size:
            raise ValueError(
                f"image size {image_hw} is not a multiple of patch size {self.patch_size}"
            )
        gh, gw = h // self.patch_size, w // self.patch_size

        cam = camera.resized(w, h) if (camera.height, camera.width) != (h, w) else camera
        self.camera = cam
        self._grid = (gh, gw)
        self.adapter = adapter
        if adapter is not None:
            adapter.bind(cam, gh, gw, self.patch_size)

        self.use_patch_undistort = patch_undistort
        self.use_border_token = border_token
        self.use_dpt_grid = dpt_grid

        dev = next(self.model.parameters()).device
        if patch_undistort:
            self._undistort_grid = C.patch_undistort_grid(
                cam, h, w, self.patch_size, preserve_scale=preserve_scale, device=dev)
        if border_token:
            self._valid_patches = C.patch_valid_mask(cam, h, w, self.patch_size, device=dev)

        self._install_hooks(grid_mode=grid_mode)
        return self

    def remove(self) -> None:
        """Restore the frozen model exactly as it was."""
        for hd in self._handles:
            hd.remove()
        self._handles.clear()
        for module, name, original in reversed(self._patched):
            if original is None:
                try:
                    delattr(module, name)
                except AttributeError:
                    pass
            else:
                setattr(module, name, original)
        self._patched.clear()
        self._undistort_grid = None
        self._valid_patches = None
        self._pe_table = None

    def pe_table(self) -> Optional[Tensor]:
        """``(H*W, C)`` pretrained absolute PE on the bound grid, or ``None``.

        ``None`` for RoPE-only backbones such as ``vggt_omega``; the TV
        regulariser then falls back to the residual (see ``losses.tv_penalty``).
        """
        return self._pe_table

    # -- hook plumbing ----------------------------------------------------

    def _patch_method(self, module: nn.Module, name: str, fn: Callable) -> None:
        original = module.__dict__.get(name, None)
        self._patched.append((module, name, original))
        setattr(module, name, types.MethodType(fn, module))

    def _install_hooks(self, *, grid_mode: str) -> None:
        self._hook_tokenizer()
        self._hook_rope()
        self._hook_abs_pe()
        if self.use_dpt_grid:
            self._hook_dpt_grid(grid_mode)

    # -- shared implementations -------------------------------------------

    def _tokenizer_modules(self) -> List[nn.Module]:
        """The ``patch_embed`` conv/linear whose input is the raw image."""
        raise NotImplementedError

    def _hook_tokenizer(self) -> None:
        if not (self.use_patch_undistort or self.use_border_token):
            return
        for mod in self._tokenizer_modules():
            if self.use_patch_undistort:
                def pre(_m, args, _self=self):
                    if not args or not torch.is_tensor(args[0]):
                        return None
                    x = args[0]
                    if x.dim() != 4 or _self._undistort_grid is None:
                        return None
                    if x.shape[-2:] != _self._undistort_grid.shape[:2]:
                        return None
                    return (C.undistort_patches(x, _self._undistort_grid),) + tuple(args[1:])

                self._handles.append(mod.register_forward_pre_hook(pre))

            if self.use_border_token:
                def post(_m, _args, out, _self=self):
                    if not torch.is_tensor(out) or out.dim() != 3:
                        return None
                    if _self._valid_patches is None:
                        return None
                    valid = _self._valid_patches.reshape(-1)
                    if out.shape[1] != valid.numel():
                        return None
                    return C.fill_border_tokens(out, valid.to(out.device))

                self._handles.append(mod.register_forward_hook(post))

    def _hook_rope(self) -> None:
        """Apply Eq. 6 to every RoPE module's ``(sin, cos)``.

        The correction is applied through the angle-addition identities on the
        returned pair, so the backbone's own frequency schedule never has to be
        re-derived. ``enable_grad`` is required because at least one call site
        (``vggt_omega``'s aggregator) evaluates RoPE under ``torch.no_grad``,
        which would otherwise silently detach the adapter from the graph.
        """
        if not self.has_rope:
            return

        def hook(_m, _args, out, _self=self):
            if _self.adapter is None or _self.adapter.rope is None:
                return None
            if not (isinstance(out, tuple) and len(out) == 2):
                return None
            sin, cos = out
            if not torch.is_tensor(sin) or sin.dim() != 2:
                return None
            gh, gw = _self._grid
            if sin.shape[0] != gh * gw:
                return None
            with torch.enable_grad():
                return _self.adapter.rope_sincos(sin, cos)

        def hook_tokens(_m, _args, out, _self=self):
            """VGGT-style RoPE: the module returns rotated tokens, not (sin, cos).

            VGGT rotates the *whole* sequence, prefix camera/register tokens
            included (they carry position -1). Those have no image location, so
            the radial correction applies only to the trailing patch tokens.
            """
            if _self.adapter is None or _self.adapter.rope is None:
                return None
            if not torch.is_tensor(out) or out.dim() < 3:
                return None
            gh, gw = _self._grid
            n_prefix = out.shape[-2] - gh * gw
            if n_prefix < 0:
                return None
            with torch.enable_grad():
                patch = _self.adapter.rope_tokens(out[..., n_prefix:, :], n_blocks=2)
                if n_prefix == 0:
                    return patch
                return torch.cat((out[..., :n_prefix, :], patch), dim=-2)

        for mod in self.model.modules():
            name = type(mod).__name__
            if name == "RopePositionEmbedding":
                self._handles.append(mod.register_forward_hook(hook))
            elif name == "RotaryPositionEmbedding2D":
                self._handles.append(mod.register_forward_hook(hook_tokens))

    def _hook_abs_pe(self) -> None:
        raise NotImplementedError

    def _hook_dpt_grid(self, grid_mode: str) -> None:
        raise NotImplementedError

    def _make_apply_pos_embed(self, grid_mode: str, create_uv_grid_ref, position_grid_to_embed_ref):
        """Build a replacement ``_apply_pos_embed`` that uses camera-aware coords.

        Mirrors the original method exactly apart from the grid it starts from.
        """
        outer = self

        def _apply_pos_embed(self, x: Tensor, W: int, H: int, ratio: float = 0.1) -> Tensor:
            patch_w, patch_h = x.shape[-1], x.shape[-2]
            if outer.camera is None:
                pos = create_uv_grid_ref(patch_w, patch_h, aspect_ratio=W / H,
                                         dtype=x.dtype, device=x.device)
            else:
                pos = C.camera_aware_uv_grid(
                    outer.camera, patch_w, patch_h, aspect_ratio=W / H,
                    mode=grid_mode, device=x.device, dtype=torch.float32).to(x.dtype)
            pos = position_grid_to_embed_ref(pos, x.shape[1])
            pos = pos * ratio
            pos = pos.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
            return x + pos

        return _apply_pos_embed

    # -- forward ----------------------------------------------------------

    def forward(self, images: Tensor) -> Prediction:
        raise NotImplementedError

    def valid_mask(self, h: int, w: int, device=None) -> Tensor:
        return self.camera.valid_mask(h, w, device=device)


# ---------------------------------------------------------------------------
# VGGT (vendored vggt_visfeat, DINOv2 -- absolute PE, no RoPE)
# ---------------------------------------------------------------------------


class VGGTBackbone(Backbone):
    # DINOv2 ViT (learned ``pos_embed``) inside an aggregator that additionally
    # applies 2D axial RoPE (``rope_freq=100``), so both Eq. 5 and Eq. 6 apply.
    patch_size = 14
    embed_dim = 1024
    has_abs_pe = True
    has_rope = True

    @classmethod
    def load(cls, weights: Optional[str] = None, device="cpu", **kw) -> "VGGTBackbone":
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "VGGT-360-fisheye"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from vggt_visfeat.models.vggt import VGGT

        if weights in (None, "", "pretrained"):
            model = VGGT.from_pretrained("facebook/VGGT-1B")
        else:
            model = VGGT(**kw)
            sd = torch.load(weights, map_location="cpu")
            sd = sd.get("model", sd.get("state_dict", sd))
            model.load_state_dict(sd, strict=False)
        return cls(model.to(device))

    def _vit(self) -> nn.Module:
        return self.model.aggregator.patch_embed

    def _tokenizer_modules(self) -> List[nn.Module]:
        return [self._vit().patch_embed]

    def _hook_abs_pe(self) -> None:
        vit = self._vit()
        outer = self

        def interpolate_pos_encoding(self, x, w, h, _orig=vit.interpolate_pos_encoding):
            pos = _orig(x, w, h)                       # (1, 1 + N, C), cls first
            gh, gw = outer._grid
            if pos.shape[1] - 1 == gh * gw and outer._pe_table is None:
                outer._pe_table = pos[0, 1:].detach()
            if outer.adapter is None or outer.adapter.pe is None:
                return pos
            if pos.shape[1] - 1 != gh * gw:
                return pos
            res = outer.adapter.pe_residual().to(pos.dtype).to(pos.device)
            # Eq. 5 touches patch positions only; the class token has no position.
            return torch.cat((pos[:, :1], pos[:, 1:] + res.unsqueeze(0)), dim=1)

        self._patch_method(vit, "interpolate_pos_encoding", interpolate_pos_encoding)

    def _hook_dpt_grid(self, grid_mode: str) -> None:
        from vggt_visfeat.heads.utils import create_uv_grid, position_grid_to_embed

        fn = self._make_apply_pos_embed(grid_mode, create_uv_grid, position_grid_to_embed)
        for head_name in ("depth_head", "point_head"):
            head = getattr(self.model, head_name, None)
            if head is not None and hasattr(head, "_apply_pos_embed"):
                self._patch_method(head, "_apply_pos_embed", fn)

    def forward(self, images: Tensor) -> Prediction:
        from vggt_visfeat.utils.pose_enc import pose_encoding_to_extri_intri

        if images.dim() == 4:
            images = images.unsqueeze(0)
        # save_attn=True makes the aggregator collect and assert on frame
        # attention maps, which RayTun3R never reads.
        preds, _ = self.model(images, save_attn=False)
        h, w = images.shape[-2:]
        extri, _ = pose_encoding_to_extri_intri(preds["pose_enc"], (h, w))
        extri = extri[0]                                   # (S, 3, 4) cam-from-world
        depth = preds["depth"][0, ..., 0]                  # (S, H, W)
        conf = preds.get("depth_conf", torch.ones_like(depth[:, None]))[0]
        return Prediction(depth=depth, conf=conf, R=extri[:, :3, :3], t=extri[:, :3, 3],
                          extra={"pose_enc": preds["pose_enc"]})


# ---------------------------------------------------------------------------
# VGGT-Omega (this repo, DINOv3 -- RoPE only, no absolute PE)
# ---------------------------------------------------------------------------


class VGGTOmegaBackbone(Backbone):
    patch_size = 16
    embed_dim = 1024
    has_abs_pe = False      # DINOv3: no learned absolute positional table
    has_rope = True

    @classmethod
    def load(cls, weights: Optional[str] = None, device="cpu", **kw) -> "VGGTOmegaBackbone":
        from vggt_omega.models.vggt_omega import VGGTOmega

        model = VGGTOmega(**kw)
        if weights:
            sd = torch.load(weights, map_location="cpu")
            sd = sd.get("model", sd.get("state_dict", sd))
            model.load_state_dict(sd, strict=False)
        return cls(model.to(device))

    def _tokenizer_modules(self) -> List[nn.Module]:
        vit = self.model.aggregator.patch_embed
        inner = getattr(vit, "patch_embed", None)
        return [inner if inner is not None else vit]

    def _hook_abs_pe(self) -> None:
        # Nothing to hook: DINOv3 carries position purely through RoPE. See the
        # module docstring -- this is the structural reason this backbone can
        # only reproduce the paper's "RoPE only" ablation row.
        return

    def _hook_dpt_grid(self, grid_mode: str) -> None:
        from vggt_omega.models.heads.utils import create_uv_grid, position_grid_to_embed

        fn = self._make_apply_pos_embed(grid_mode, create_uv_grid, position_grid_to_embed)
        head = getattr(self.model, "dense_head", None)
        if head is not None and hasattr(head, "_apply_pos_embed"):
            self._patch_method(head, "_apply_pos_embed", fn)

    def _forward_cpu(self, images: Tensor) -> dict:
        """``VGGTOmega.forward`` without its hardcoded CUDA autocast.

        ``vggt_omega/models/vggt_omega.py`` opens ``torch.autocast(device_type=
        "cuda")`` and calls ``torch.cuda.is_bf16_supported()`` unconditionally,
        which raises on a CPU-only host. Rather than edit that file, this mirrors
        its forward in full precision so smoke tests and unit tests can run
        anywhere. The CUDA path below still goes through the model itself.
        """
        model = self.model
        tokens_list, patch_start = model.aggregator(images)
        preds = {}
        if model.camera_head is not None:
            preds["pose_enc"] = model.camera_head(tokens_list, patch_token_start=patch_start)
        if model.dense_head is not None:
            depth, conf = model.dense_head(tokens_list, images=images,
                                           patch_token_start=patch_start)
            preds["depth"], preds["depth_conf"] = depth, conf
        return preds

    def forward(self, images: Tensor) -> Prediction:
        from vggt_omega.utils.pose_enc import encoding_to_camera

        if images.dim() == 4:
            images = images.unsqueeze(0)
        preds = self.model(images) if images.is_cuda else self._forward_cpu(images)
        h, w = images.shape[-2:]
        extri, _ = encoding_to_camera(preds["pose_enc"], (h, w))
        extri = extri[0]
        depth = preds["depth"][0, ..., 0]
        conf = preds.get("depth_conf", torch.ones_like(depth[:, None]))[0]
        return Prediction(depth=depth, conf=conf, R=extri[:, :3, :3], t=extri[:, :3, 3],
                          extra={"pose_enc": preds["pose_enc"]})


# ---------------------------------------------------------------------------
# Depth Anything 3 (paper's primary backbone; external dependency)
# ---------------------------------------------------------------------------


class DA3Backbone(Backbone):
    patch_size = 14
    embed_dim = 384         # DA3-Small; overridden per variant in load()
    has_abs_pe = True
    has_rope = True

    _VARIANTS = {"small": 384, "base": 768, "large": 1024}

    @classmethod
    def load(cls, weights: Optional[str] = None, device="cpu", variant: str = "small",
             **kw) -> "DA3Backbone":
        try:
            from depth_anything_3.api import DepthAnything3  # type: ignore
        except ImportError as exc:               # pragma: no cover - optional dep
            raise ImportError(
                "The 'da3' backbone needs the depth_anything_3 package "
                "(https://github.com/ByteDance-Seed/depth-anything-3). Install it, or "
                "use --backbone vggt, which is vendored in this repo."
            ) from exc

        model = DepthAnything3.from_pretrained(weights or f"depth-anything/DA3NESTED-GIANT")
        obj = cls(model.to(device))
        obj.embed_dim = cls._VARIANTS.get(variant, 384)
        return obj

    def _vit(self) -> nn.Module:
        for attr in ("backbone", "encoder", "vit", "patch_embed"):
            mod = getattr(self.model, attr, None)
            if mod is not None and hasattr(mod, "pos_embed"):
                return mod
        for mod in self.model.modules():
            if hasattr(mod, "pos_embed") and isinstance(getattr(mod, "pos_embed"), nn.Parameter):
                return mod
        raise RuntimeError("could not locate the DA3 ViT carrying 'pos_embed'")

    def _tokenizer_modules(self) -> List[nn.Module]:
        vit = self._vit()
        inner = getattr(vit, "patch_embed", None)
        return [inner if inner is not None else vit]

    def _hook_abs_pe(self) -> None:
        vit = self._vit()
        outer = self
        orig = getattr(vit, "interpolate_pos_encoding", None)
        if orig is None:
            raise RuntimeError("DA3 ViT has no interpolate_pos_encoding to hook")

        def interpolate_pos_encoding(self, x, w, h, _orig=orig):
            pos = _orig(x, w, h)
            gh, gw = outer._grid
            n_prefix = pos.shape[1] - gh * gw
            if n_prefix >= 0 and outer._pe_table is None:
                outer._pe_table = pos[0, n_prefix:].detach()
            if outer.adapter is None or outer.adapter.pe is None:
                return pos
            if n_prefix < 0:
                return pos
            res = outer.adapter.pe_residual().to(pos.dtype).to(pos.device)
            return torch.cat((pos[:, :n_prefix], pos[:, n_prefix:] + res.unsqueeze(0)), dim=1)

        self._patch_method(vit, "interpolate_pos_encoding", interpolate_pos_encoding)

    def _hook_dpt_grid(self, grid_mode: str) -> None:  # pragma: no cover - optional dep
        for mod in self.model.modules():
            if hasattr(mod, "_apply_pos_embed"):
                try:
                    from depth_anything_3.model.heads.utils import (  # type: ignore
                        create_uv_grid, position_grid_to_embed,
                    )
                except ImportError:
                    return
                self._patch_method(mod, "_apply_pos_embed",
                                   self._make_apply_pos_embed(grid_mode, create_uv_grid,
                                                              position_grid_to_embed))

    def forward(self, images: Tensor) -> Prediction:  # pragma: no cover - optional dep
        if images.dim() == 4:
            images = images.unsqueeze(0)
        out = self.model(images)
        depth = out["depth"] if isinstance(out, dict) else out.depth
        pose = out["extrinsics"] if isinstance(out, dict) else out.extrinsics
        depth = depth.squeeze(0) if depth.dim() == 4 else depth
        pose = pose.squeeze(0) if pose.dim() == 4 else pose
        conf = (out.get("conf") if isinstance(out, dict) else None)
        conf = torch.ones_like(depth) if conf is None else conf.squeeze(0)
        return Prediction(depth=depth, conf=conf, R=pose[:, :3, :3], t=pose[:, :3, 3])


BACKBONES = {
    "vggt": VGGTBackbone,
    "vggt_omega": VGGTOmegaBackbone,
    "da3": DA3Backbone,
}


def build_backbone(name: str, weights: Optional[str] = None, device="cpu", **kw) -> Backbone:
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
    return BACKBONES[name].load(weights=weights, device=device, **kw)
