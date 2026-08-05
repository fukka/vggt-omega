"""Comparison methods from Sec. 5 "Baselines".

Two families:

*Input-space projection* -- reuse the pinhole backbone by feeding it pinhole
imagery, at the cost of field of view (Center-PH) or compute (Multi-PH):

* :class:`CenterPH` undistorts each fisheye frame into one forward-looking
  110 deg virtual pinhole crop. The paper picks 110 deg to avoid the severe centre
  compression and peripheral stretching of wider rectilinear projections.
* :class:`MultiPH` adds four tilted views and fuses all five back into the
  fisheye frame, "improving coverage at roughly proportional extra cost"
  (Tab. 4(b): ~4x the per-frame latency).

*Lightweight adaptation* -- train a small parameter set with the same objective
RayTun3R uses, so the comparison isolates *what* is adapted rather than how:

* :class:`LoRAAdapter` on the attention QKV projections, ``r=8, alpha=16``
  (the paper's default from Tab. 3, right), ~147.5K trainable parameters.
* :class:`CalibrationTokens` (CalTok [11]), ``t=4`` learned tokens, ~18.4K.

Two implementation notes worth carrying into any comparison:

* Multi-PH pose fusion is under-specified in the paper. All views of a frame
  share an optical centre and differ by a *known* rotation, so each view's
  predicted pose is mapped back through that rotation and the results are
  averaged (chordal rotation mean). Center-PH is the single-view special case.
* CalTok here is a compact re-implementation for this comparison. This repo also
  carries a fuller calibration-token model in ``fisheye3r/`` (the Fisheye3R
  reproduction, which builds on the same idea); that one is not reused because
  it inserts tokens under a different masking scheme.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbones import Backbone, Prediction
from .cameras import Camera, Pinhole, pixel_grid

__all__ = ["VirtualView", "CenterPH", "MultiPH", "LoRAAdapter", "CalibrationTokens",
           "attach_lora", "attach_caltok"]


# ---------------------------------------------------------------------------
# Projection baselines
# ---------------------------------------------------------------------------


def _rot_from_axis_angle(axis: Tensor, angle: float) -> Tensor:
    a = F.normalize(axis.to(torch.float64), dim=0)
    K = torch.tensor([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]],
                     dtype=torch.float64)
    return (torch.eye(3, dtype=torch.float64) + math.sin(angle) * K
            + (1 - math.cos(angle)) * K @ K).float()


@dataclass
class VirtualView:
    """A virtual pinhole camera sharing the fisheye's optical centre.

    ``R_vc`` rotates a ray from the *virtual* view's frame into the fisheye
    camera frame, so a pose predicted in the virtual frame maps back as
    ``R_cam = R_vc R_pred`` (and the translation likewise).
    """

    pinhole: Pinhole
    R_vc: Tensor          # (3, 3)

    def sampling_grid(self, fisheye: Camera) -> Tensor:
        """``(H, W, 2)`` normalised grid pulling the fisheye image into this view."""
        h, w = self.pinhole.height, self.pinhole.width
        uv = pixel_grid(h, w, dtype=torch.float32)
        rays = self.pinhole.unproject(uv) @ self.R_vc.transpose(0, 1)
        src = fisheye.project(rays)
        g = torch.empty_like(src)
        g[..., 0] = 2.0 * (src[..., 0] + 0.5) / fisheye.width - 1.0
        g[..., 1] = 2.0 * (src[..., 1] + 0.5) / fisheye.height - 1.0
        # ``rays`` is already in the fisheye camera frame, so the incidence angle
        # is just the angle to +z. Rays outside the imaged cone are pushed out of
        # range so grid_sample's zero padding blanks them.
        theta = torch.acos(rays[..., 2].clamp(-1.0, 1.0))
        g[theta > fisheye.theta_max] = -2.0
        return g

    def coverage(self, fisheye: Camera) -> Tensor:
        """``(Hf, Wf)`` bool: which fisheye pixels this view sees."""
        rays = fisheye.ray_grid()
        local = rays @ self.R_vc            # fisheye frame -> virtual frame
        front = local[..., 2] > 1e-6
        uv = self.pinhole.project(local)
        return (front & (uv[..., 0] >= 0) & (uv[..., 0] <= self.pinhole.width - 1)
                & (uv[..., 1] >= 0) & (uv[..., 1] <= self.pinhole.height - 1))


def make_views(fisheye: Camera, *, fov_deg: float = 110.0, n: int = 1,
               tilt_deg: float = 45.0, size: Optional[int] = None) -> List[VirtualView]:
    """Center-PH (``n=1``) or Multi-PH (``n=5``) virtual pinhole rig."""
    side = size or max(fisheye.height, fisheye.width)
    ph = fisheye.to_pinhole(fov_deg=fov_deg, width=side, height=side)

    views = [VirtualView(ph, torch.eye(3))]
    if n > 1:
        ang = math.radians(tilt_deg)
        axes = [torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, -1.0, 0.0]),
                torch.tensor([1.0, 0.0, 0.0]), torch.tensor([-1.0, 0.0, 0.0])]
        for a in axes[: n - 1]:
            views.append(VirtualView(ph, _rot_from_axis_angle(a, ang)))
    return views


def _chordal_rotation_mean(rots: Tensor) -> Tensor:
    """Projection of the arithmetic mean onto SO(3) (chordal L2 mean)."""
    U, _, Vh = torch.linalg.svd(rots.mean(dim=0).double())
    R = U @ Vh
    if torch.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vh
    return R.float()


class ProjectionBaseline:
    """Runs a frozen backbone on virtual pinhole views and fuses back to fisheye."""

    def __init__(self, backbone: Backbone, fisheye: Camera, views: Sequence[VirtualView],
                 *, depth_convention: str = "range"):
        self.backbone = backbone
        self.fisheye = fisheye
        self.views = list(views)
        self.depth_convention = depth_convention
        self._grids = [v.sampling_grid(fisheye) for v in self.views]
        self._cover = [v.coverage(fisheye) for v in self.views]

    def warp_in(self, images: Tensor, k: int) -> Tensor:
        g = self._grids[k].to(images.device, images.dtype)[None].expand(images.shape[0], -1, -1, -1)
        return F.grid_sample(images, g, mode="bilinear", padding_mode="zeros",
                             align_corners=False)

    @torch.no_grad()
    def __call__(self, images: Tensor) -> Prediction:
        s = images.shape[0]
        hf, wf = self.fisheye.height, self.fisheye.width
        dev = images.device

        depth_acc = torch.zeros(s, hf, wf, device=dev)
        best = torch.full((s, hf, wf), -1.0, device=dev)
        Rs, ts = [], []

        for k, view in enumerate(self.views):
            warped = self.warp_in(images, k)
            pred = self.backbone.forward(warped[None])

            # Depth: virtual-frame planar z -> range along the fisheye ray.
            rays = self.fisheye.ray_grid(hf, wf, device=dev)
            local = rays @ view.R_vc.to(dev)
            uv = view.pinhole.project(local)
            g = torch.stack((2 * (uv[..., 0] + 0.5) / view.pinhole.width - 1.0,
                             2 * (uv[..., 1] + 0.5) / view.pinhole.height - 1.0), dim=-1)
            samp = F.grid_sample(pred.depth[:, None], g[None].expand(s, -1, -1, -1),
                                 mode="bilinear", padding_mode="border",
                                 align_corners=False)[:, 0]
            if self.depth_convention == "range":
                samp = samp / local[..., 2].clamp_min(1e-3)

            cover = self._cover[k].to(dev)
            # Prefer the view whose axis is closest to each ray.
            score = torch.where(cover, local[..., 2], torch.full_like(local[..., 2], -1.0))
            take = cover[None] & (score[None] > best)
            depth_acc = torch.where(take, samp, depth_acc)
            best = torch.where(take, score[None].expand_as(best), best)

            Rs.append(view.R_vc.to(dev) @ pred.R)
            ts.append(pred.t @ view.R_vc.to(dev).transpose(0, 1))

        R = torch.stack([_chordal_rotation_mean(torch.stack([Rs[k][i] for k in range(len(self.views))]))
                         for i in range(s)])
        t = torch.stack(ts).mean(dim=0)
        covered = best > -1.0
        return Prediction(depth=depth_acc, conf=covered.float(), R=R, t=t,
                          covered=covered)


def CenterPH(backbone: Backbone, fisheye: Camera, *, fov_deg: float = 110.0,
             **kw) -> ProjectionBaseline:
    return ProjectionBaseline(backbone, fisheye, make_views(fisheye, fov_deg=fov_deg, n=1), **kw)


def MultiPH(backbone: Backbone, fisheye: Camera, *, fov_deg: float = 110.0,
            tilt_deg: float = 45.0, **kw) -> ProjectionBaseline:
    return ProjectionBaseline(backbone, fisheye,
                              make_views(fisheye, fov_deg=fov_deg, n=5, tilt_deg=tilt_deg), **kw)


# ---------------------------------------------------------------------------
# PEFT baselines
# ---------------------------------------------------------------------------


class LoRAAdapter(nn.Module):
    """Low-rank update on one ``nn.Linear``: ``W x + (alpha/r) B A x``, ``B`` zero-init."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.A.T @ self.B.T) * self.scale


_HEAD_PREFIXES = ("camera_head", "depth_head", "point_head", "dense_head",
                  "text_alignment_head")


def _is_head(name: str) -> bool:
    return any(name.startswith(h) or f".{h}." in name for h in _HEAD_PREFIXES)


def attach_lora(backbone: Backbone, *, r: int = 8, alpha: float = 16.0,
                target: str = "qkv", n_layers: int = 12) -> Tuple[nn.ModuleList, List]:
    """Attach LoRA to the last ``n_layers`` attention QKV projections.

    Paper default ``r=8, alpha=16`` (Tab. 3, right). ``n_layers=12`` reproduces
    the quoted 147.5K trainable parameters on DA3-Small: each ``384 -> 1152`` QKV
    projection costs ``8*384 + 1152*8 = 12,288``, and ``12 * 12,288 = 147,456``.

    Implemented as forward hooks, so the frozen weights are never modified and
    ``handles`` fully undoes it. Prediction heads are excluded -- the paper keeps
    them frozen.
    """
    mods, handles = nn.ModuleList(), []
    device = next(backbone.model.parameters()).device
    targets = [(n, m) for n, m in backbone.model.named_modules()
               if isinstance(m, nn.Linear) and n.endswith(target) and not _is_head(n)]
    if not targets:
        raise RuntimeError(f"no encoder nn.Linear ending in {target!r} in this backbone")
    if n_layers:
        targets = targets[-n_layers:]

    for _, mod in targets:
        lora = LoRAAdapter(mod, r=r, alpha=alpha).to(device)
        mods.append(lora)

        def hook(_m, args, out, _l=lora):
            with torch.enable_grad():
                return out + _l(args[0])

        handles.append(mod.register_forward_hook(hook))
    return mods, handles


class CalibrationTokens(nn.Module):
    """CalTok [11]: ``t`` learned tokens spliced into a block's token sequence.

    Inserted before the block and dropped after, so the sequence length the rest
    of the model sees is unchanged. Tokens are near-zero initialised so the
    adapted model starts at the frozen one.

    They go at the **front**, alongside the camera/register tokens both backbones
    already carry there, rather than at the end. Appending would desynchronise
    positional bookkeeping that is indexed from the tail: VGGT passes a separate
    ``pos`` array sized to the sequence, and DINOv3 applies RoPE to the trailing
    ``H*W`` tokens. Prefixing leaves the patch-token block contiguous and last.
    """

    def __init__(self, dim: int, n_tokens: int = 4):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(n_tokens, dim) * 1e-6)
        self.n = n_tokens


def attach_caltok(backbone: Backbone, *, n_tokens: int = 4, n_layers: int = 12
                  ) -> Tuple[nn.ModuleList, List]:
    """Attach CalTok to the last ``n_layers`` attention blocks."""
    # Only encoder blocks: the prediction heads contain modules of the same class
    # at twice the width (``dim_in = 2 * embed_dim``), and splicing tokens into
    # those would both mismatch the width and corrupt the pose/depth readout.
    blocks = []
    for name, m in backbone.model.named_modules():
        if type(m).__name__ not in ("Block", "SelfAttentionBlock"):
            continue
        if _is_head(name):
            continue
        norm = getattr(m, "norm1", None)
        dim = getattr(norm, "normalized_shape", [None])[0] if norm is not None else None
        if dim is not None and dim != backbone.embed_dim:
            continue
        blocks.append(m)
    if not blocks:
        raise RuntimeError("no encoder transformer blocks found for CalTok")
    blocks = blocks[-n_layers:]

    device = next(backbone.model.parameters()).device
    mods, handles = nn.ModuleList(), []
    for blk in blocks:
        ct = CalibrationTokens(backbone.embed_dim, n_tokens).to(device)
        mods.append(ct)

        def pre(_m, args, kwargs, _c=ct):
            if not args or not torch.is_tensor(args[0]) or args[0].dim() != 3:
                return None
            x = args[0]
            with torch.enable_grad():
                tok = _c.tokens[None].expand(x.shape[0], -1, -1).to(x.dtype).to(x.device)
                new_args = (torch.cat((tok, x), dim=1),) + tuple(args[1:])

            # VGGT hands RoPE a per-token position array; extend it to match.
            # Its convention for non-spatial tokens is position *0* (the grid is
            # shifted by +1 to free that slot), and the RoPE lookup is an
            # embedding, so a negative marker would index out of range.
            def extend_pos(p):
                if not torch.is_tensor(p) or p.dim() != 3 or p.shape[1] != x.shape[1]:
                    return p
                pad = p.new_zeros((p.shape[0], _c.n, p.shape[2]))
                return torch.cat((pad, p), dim=1)

            kwargs = dict(kwargs)
            if "pos" in kwargs:
                kwargs["pos"] = extend_pos(kwargs["pos"])
            new_args = tuple(extend_pos(a) if i else a for i, a in enumerate(new_args))
            return new_args, kwargs

        def post(_m, _args, out, _c=ct):
            if torch.is_tensor(out) and out.dim() == 3:
                # ``.contiguous()`` matters: VGGT's aggregator calls ``.view()``
                # on the block output, which rejects a sliced tensor.
                return out[:, _c.n:].contiguous()
            return None

        handles.append(blk.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(blk.register_forward_hook(post))
    return mods, handles
