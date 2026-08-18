"""Frozen 3D foundation model backbones, and where RayTun3R hooks into them.

The paper evaluates DA3, pi^3 and VGGT. This module exposes three switchable
backbones -- ``vggt``, ``vggt_omega`` and ``da3`` -- behind one interface, so the
adapter, losses and evaluation are written once.

Four attachment points, matching Sec. 4.2:

===================  ==========================================================
Attachment           Where it lands
===================  ==========================================================
absolute PE (Eq. 5)  the ViT's positional-embedding table, before it is added
RoPE (Eq. 6)         every RoPE module: the ``(sin, cos)`` pair for DINOv3-style
                     ones, the rotated tokens for VGGT-style ones
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
import warnings
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import corrections as C
from .adapter import RayTun3RAdapter
from .cameras import DEPTH_CONVENTIONS, Camera, convert_depth

__all__ = ["Prediction", "Backbone", "build_backbone", "BACKBONES"]


@dataclass
class Prediction:
    """Per-frame geometry from a backbone, in a backbone-independent form.

    ``R``/``t`` are camera-from-world, so the relative pose of frame j w.r.t. i
    is ``R_j R_i^T`` and ``t_j - R_j R_i^T t_i``.

    ``depth_convention`` says what ``depth`` *means*, and it is not decoration.
    Every depth head in this repo natively emits **planar z** (CONTEXT.md
    measured it for the VGGT family by RANSAC plane fit), while Eq. 7's
    ``X = D kappa^-1`` is only correct when the normalisation of ``kappa^-1``
    matches. Producers therefore convert to the convention the run asked for and
    tag it here, and consumers assert on the tag rather than assume.

    This used to be implicit, and the two producers disagreed: the virtual-pinhole
    baselines divided by ``cos`` to get range while the direct fisheye path handed
    the head's raw planar z straight to ``backproject(convention="range")``. Both
    were then compared against ADT ground truth that *had* been converted. The
    resulting radial warp is worth ~0.66 px of ``d_reproj`` on Aria geometry and
    ~0.99 px on a 170 deg frame -- against a measured method-to-method spread of
    0.10 px, i.e. the artefact was an order of magnitude larger than the effect
    being measured.

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
    depth_convention: str = "z"         # what `depth` means; see the class docstring

    def require_convention(self, convention: str) -> None:
        """Fail loudly when a consumer's convention differs from this map's.

        The mismatch it guards is silent by construction -- both readings are
        plain positive depth maps of the same shape, so nothing downstream can
        notice -- and it costs more than the effect under study.
        """
        if self.depth_convention != convention:
            raise ValueError(
                f"depth is {self.depth_convention!r} but the caller asked for "
                f"{convention!r}. These differ by a per-pixel 1/cos(theta), up to "
                f"~11x at a 170 deg frame corner, and no global scale alignment can "
                f"absorb it. Install the backbone with depth_convention="
                f"{convention!r} rather than converting here."
            )

    def relative(self, i: int, j: int) -> Tuple[Tensor, Tensor]:
        R_rel = self.R[j] @ self.R[i].transpose(-1, -2)
        t_rel = self.t[j] - R_rel @ self.t[i]
        return R_rel, t_rel


#: RoPE modules that return a ``(sin, cos)`` pair for the caller to apply.
_ROPE_SINCOS_MODULES = {"RopePositionEmbedding"}
#: RoPE modules that return already-rotated tokens.
_ROPE_TOKEN_MODULES = {"RotaryPositionEmbedding2D", "RoPE2D"}


class Backbone(nn.Module):
    """Wraps a frozen model and owns the four attachment points."""

    #: filled in by subclasses
    patch_size: int = 14
    embed_dim: int = 1024
    has_abs_pe: bool = False
    has_rope: bool = False
    #: How many non-patch tokens a token-rotating RoPE module sees ahead of each
    #: frame's patch block (VGGT: 1 camera + 4 register tokens, per frame).
    #: Declared, never inferred: the token count alone cannot distinguish one
    #: frame from S frames concatenated, and guessing it silently leaves S-1
    #: frames uncorrected in every global-attention block -- see ``_hook_rope``.
    #: ``None`` means "this backbone has no such module"; finding one anyway is
    #: an error at install time rather than a silent no-op.
    n_prefix_tokens: Optional[int] = None
    #: what this model's depth head natively emits, before any conversion.
    #: Every head wrapped here is pinhole-trained and emits planar z.
    native_depth: str = "z"

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
        self._rope_layout_cache: dict = {}
        self.use_patch_undistort = False
        self.use_border_token = False
        self.use_dpt_grid = False
        #: convention every Prediction from this backbone is emitted in.
        self.depth_convention = "range"

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
                preserve_scale: bool = True, grid_mode: str = "auto",
                depth_convention: str = "range") -> "Backbone":
        """Attach the adapter and the parameter-free corrections.

        ``adapter=None`` installs only the parameter-free parts, which is how the
        "Param.-free only" row of Tab. 8 is produced.

        ``depth_convention`` is what :attr:`Prediction.depth` will be emitted in,
        converted from :attr:`native_depth`. It must match what the losses,
        metrics and ground-truth loader use, or the geometry is warped radially;
        :meth:`Prediction.require_convention` enforces that downstream.
        """
        self.remove()
        h, w = image_hw
        if h % self.patch_size or w % self.patch_size:
            raise ValueError(
                f"image size {image_hw} is not a multiple of patch size {self.patch_size}"
            )
        gh, gw = h // self.patch_size, w // self.patch_size

        if depth_convention not in DEPTH_CONVENTIONS:
            raise ValueError(f"depth_convention must be one of {DEPTH_CONVENTIONS}")
        self.depth_convention = depth_convention

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
        self._rope_layout_cache = {}
        # Drop the camera too, so a post-remove forward returns the head's native
        # depth exactly as a pre-install one did. "Restores the model exactly as it
        # was" has to cover the output convention, not just the module tree.
        self.camera = None
        self._grid = None
        self.depth_convention = "range"

    def grad_checkpointing(self, enabled: bool = True) -> bool:
        """Turn the frozen model's gradient checkpointing on for the backward pass.

        Returns whether anything changed, so callers can report it.

        RayTun3R's trainable parameters sit at the *first* layer -- the patch
        positional encoding -- so unlike VGGT's own fine-tuning recipe (which
        freezes ``*aggregator*`` and trains only the heads, and therefore never
        backpropagates through the trunk at all) every activation in the trunk
        has to be kept for the backward pass. That is the whole memory cost of
        this method, and checkpointing is what upstream uses to pay it.

        Backbones that gate checkpointing on ``self.training`` can only enable it
        by leaving eval mode, which is safe *only* if the model has no
        train/eval-dependent layers. Each subclass decides; the default is to do
        nothing, so a backbone that has not been checked keeps running in
        ``eval()`` exactly as before.

        **Do not "helpfully" generalise this to vggt_omega.** Its aggregator has
        no ``self.training`` checkpoint gate to switch on in the first place, and
        train mode there is *not* inert: ``vggt_omega/models/vggt_omega.py``
        drops ``predictions["images"]`` and
        ``models/layers/vision_transformer.py`` takes a different cls-norm branch.
        The inherited no-op is the correct behaviour for that backbone, not an
        oversight.
        """
        return False

    @staticmethod
    def _rope_frame_layout(pos, n: int, g: int):
        """Recover ``(n_prefix, n_frames)`` from a RoPE module's position argument.

        ``pos`` is ``(B, N, 2)``, one grid coordinate per token, laid out as
        ``n_frames`` repeats of ``[n_prefix specials, g patches]``. The frame
        period is therefore the smallest ``p >= g`` that divides ``N`` and under
        which ``pos`` is periodic -- read off the data, so it needs no assumption
        about how the model encodes its special tokens.

        **Why the smallest period is the right one, and when.** If two divisors
        of ``N`` are both periods then so is their gcd, so a spurious period ``d``
        below the true ``p = K + g`` satisfies ``d <= p/2``. When ``K < g`` that
        puts ``d < g``, which means indices ``i`` and ``i + d`` both fall inside
        one frame's patch block and force two patch coordinates to be equal --
        excluded by the ``g``-distinct requirement below. So the scan cannot
        under-segment while ``K < g`` (VGGT: 5 vs 1296), and it can never
        over-segment, because the true period always divides ``N``, is always
        periodic, and always has a distinct patch block.

        Returns ``None`` when ``pos`` is missing or unusable, leaving the caller
        to fall back on a declaration. Note this catches a *too wide* installed
        grid (the slice straddles a frame boundary, distinctness fails) but not a
        too narrow one, which yields a consistent-looking shifted window.
        """
        if pos is None or not torch.is_tensor(pos) or pos.dim() != 3:
            return None
        if pos.shape[-1] != 2 or pos.shape[-2] != n or g <= 0 or n < g:
            return None
        p0 = pos[0]
        for per in range(g, n + 1):
            if n % per:
                continue
            s = n // per
            if s > 1 and not bool((p0.reshape(s, per, 2) == p0[:per]).all()):
                continue
            # Periodicity alone is not enough: an all-constant ``pos`` is periodic
            # at every divisor, so a degenerate tensor would "confirm" whatever
            # period was tried first. A real frame's patch block holds g *distinct*
            # grid coordinates, so require that -- it is what makes ``pos``
            # informative rather than merely consistent.
            patches = p0[per - g:per]
            if torch.unique(patches, dim=0).shape[0] != g:
                return None
            return per - g, s
        return None

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
        try:
            self._hook_tokenizer()
            self._hook_rope()
            self._hook_abs_pe()
            if self.use_dpt_grid:
                self._hook_dpt_grid(grid_mode)
        except Exception:
            # Any of these can raise -- _hook_rope now does so deliberately when
            # a backbone declares has_rope but carries no matching module. The
            # earlier hooks are already registered and methods already patched by
            # then, so without this the caller gets an exception *and* a frozen
            # model left half-instrumented, which no later remove() owns because
            # install() never finished.
            self.adapter = None
            self.remove()
            raise

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

            The prefix camera/register tokens have no image location, so the
            radial correction applies only to the patch tokens.

            **One RoPE module serves two different token layouts.** VGGT builds a
            single ``RotaryPositionEmbedding2D`` and hands the same instance to
            every frame block *and* every global block
            (``vggt_visfeat/models/aggregator.py``), so this hook fires on both:

                frame attention   (B*S, heads, K + G,     D)   -- one frame
                global attention  (B,   heads, S*(K + G), D)   -- S frames concatenated

            with ``K`` the prefix size and ``G = gh*gw``. Inferring the prefix as
            ``N - G`` is only right for the first layout. On the second it yields
            ``S*K + (S-1)*G``, which is *exactly* the start of the last frame's
            patch block -- so it lands on a correctly aligned slice and looks
            fine, while leaving the other ``S-1`` frames (including frame 0, the
            reference frame that defines the coordinate system) with no Eq. 6
            correction at all, in every global block.

            So the layout is not guessed from the token count. The module is
            called as ``rope(q, pos)``, and ``pos`` is ``(B, N, 2)`` carrying one
            grid coordinate per token -- it is the authority on where the frame
            boundaries are, for any backbone, with nothing declared.
            :meth:`_rope_frame_layout` reads it. ``n_prefix_tokens`` is only the
            fallback for a module called without positions.

            Cost: correcting all ``S`` frames instead of one makes the rotated
            temporaries ``S`` times larger, and they are retained for backward
            because the adapter's gradient runs through them. That is the price
            of the correction being applied at all, not a regression.
            """
            if _self.adapter is None or _self.adapter.rope is None:
                return None
            if not torch.is_tensor(out) or out.dim() < 3:
                return None
            gh, gw = _self._grid
            g = gh * gw
            n = out.shape[-2]

            # Cached per (n, g): the search is a divisor scan plus a unique(),
            # and this hook fires twice per attention block (q and k) in all 48
            # blocks, every iteration.
            pos = _args[1] if len(_args) > 1 else None
            key = (n, g)
            if key in _self._rope_layout_cache:
                layout = _self._rope_layout_cache[key]
            else:
                layout = _self._rope_frame_layout(pos, n, g)
                _self._rope_layout_cache[key] = layout
            if layout is None:
                k = _self.n_prefix_tokens
                if k is None:
                    # No positions and no declaration. The single-frame reading is
                    # only unambiguous when the sequence cannot hold two frames'
                    # worth of patches; the other case is the one that used to be
                    # silently wrong, so refuse it.
                    if n < g:
                        return None
                    if n - g >= g:
                        raise RuntimeError(
                            f"{type(_self).__name__}: {n} RoPE tokens on a {gh}x{gw} "
                            f"grid is more than one frame's worth, the module was "
                            f"called without positions, and the backbone declares no "
                            f"n_prefix_tokens. Refusing to guess: 'prefix = n - {g}' "
                            f"aligns on the *last* frame and silently leaves every "
                            f"earlier frame uncorrected.")
                    k = n - g
                if (g + k) <= 0 or n % (g + k):
                    raise RuntimeError(
                        f"{type(_self).__name__} declares n_prefix_tokens={k} but "
                        f"{n} tokens is not a whole number of {g + k}-token frames "
                        f"on a {gh}x{gw} grid. The declaration or the installed "
                        f"image size is wrong.")
                layout = (k, n // (g + k))
            k, s = layout

            lead = out.shape[:-2]
            framed = out.reshape(*lead, s, g + k, out.shape[-1])
            with torch.enable_grad():
                patch = _self.adapter.rope_tokens(framed[..., k:, :], n_blocks=2)
                if k == 0:
                    return patch.reshape(*lead, n, out.shape[-1])
                joined = torch.cat((framed[..., :k, :], patch), dim=-2)
                return joined.reshape(*lead, n, out.shape[-1])

        found_rope = False
        for mod in self.model.modules():
            name = type(mod).__name__
            if name in _ROPE_SINCOS_MODULES:
                self._handles.append(mod.register_forward_hook(hook))
                found_rope = True
            elif name in _ROPE_TOKEN_MODULES:
                self._handles.append(mod.register_forward_hook(hook_tokens))
                found_rope = True

        if not found_rope and self.adapter is not None and self.adapter.rope is not None:
            # ``has_rope`` is what makes make_adapter() allocate RadialRoPE. If no
            # module matched, those parameters are in the optimiser but on no
            # forward path: they get no gradient, Adam skips them, and the run is
            # reported as a full adapter while Eq. 6 was never applied. Fail here
            # rather than ship a partial adapter under the wrong name.
            names = sorted({type(m).__name__ for m in self.model.modules()
                            if "rope" in type(m).__name__.lower()
                            or "rotary" in type(m).__name__.lower()})
            raise RuntimeError(
                f"{type(self).__name__} sets has_rope=True but no RoPE module "
                f"matched {sorted(_ROPE_SINCOS_MODULES | _ROPE_TOKEN_MODULES)}. "
                f"RoPE-ish classes actually present: {names or 'none'}. Add the "
                f"class to _ROPE_SINCOS_MODULES (returns (sin, cos)) or to "
                f"_ROPE_TOKEN_MODULES (returns rotated tokens), or set "
                f"has_rope=False so no RadialRoPE parameters are allocated.")

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

    def _finalize(self, pred: Prediction) -> Prediction:
        """Put a raw head prediction into the installed depth convention.

        Every concrete ``forward`` ends here, so there is exactly one place where
        a head's native planar z becomes whatever the run asked for. Doing it per
        backbone instead is how the direct path and the pinhole baselines drifted
        apart in the first place.

        Before :meth:`install` there is no camera, so no conversion is possible --
        ``z -> range`` needs the ray grid. Such a prediction is tagged with the
        head's native convention and left alone, which is also what keeps
        ``forward`` outside an install bit-identical across ``install``/``remove``.
        """
        if self.camera is None:
            pred.depth_convention = self.native_depth
            return pred
        pred.depth = convert_depth(pred.depth, self.camera,
                                   src=self.native_depth, dst=self.depth_convention)
        pred.depth_convention = self.depth_convention
        return pred

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
    def load(cls, weights: Optional[str] = None, device="cpu",
             drop_point_head: bool = True, **kw) -> "VGGTBackbone":
        """``drop_point_head`` removes the unused world-point head; see below."""
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

        # Drop the point head. The vendored fork builds all three heads
        # unconditionally and runs them on every forward, where upstream gates
        # them (``enable_point`` defaults to False in
        # ``vggt/models/vggt.py``). RayTun3R reads ``depth`` and ``pose_enc``
        # only -- ``world_points`` is never consumed by any loss, metric or
        # baseline -- but it is a full-resolution DPT head whose activations sit
        # in the autograd graph until backward, because they descend from the
        # adapter like everything else. ``VGGT.forward`` already guards on
        # ``self.point_head is not None``, and ``_hook_dpt_grid`` skips a missing
        # head, so this is a supported state rather than a hack.
        if drop_point_head:
            model.point_head = None
        return cls(model.to(device))

    @property
    def n_prefix_tokens(self) -> int:
        """``patch_start_idx``: 1 camera token + ``num_register_tokens``, per frame."""
        return int(self.model.aggregator.patch_start_idx)

    def grad_checkpointing(self, enabled: bool = True) -> bool:
        """VGGT gates ``torch.utils.checkpoint`` on ``self.training``.

        Both places that do so -- ``Aggregator._process_{frame,global}_attention``
        and the DINOv2 ViT's ``forward_features`` -- are plain
        ``if self.training: checkpoint(...) else: block(...)``, so leaving eval
        mode is the only way to switch them on. That is safe here because VGGT
        has nothing else that reads ``training``:

        * ``drop_path_rate`` defaults to 0.0 and neither ``Aggregator`` nor
          ``vit_large`` overrides it, so ``Block.forward``'s two
          ``self.training and sample_drop_ratio > ...`` branches are both false
          and it takes the same ``else`` path in either mode;
        * ``drop`` and ``attn_drop`` default to 0.0, so every ``nn.Dropout`` is
          identity in train mode too;
        * there is no ``BatchNorm`` anywhere in ``vggt_visfeat``;
        * the heads never read ``self.training``.

        So this trades compute for memory and changes no numbers. Parameters stay
        ``requires_grad=False`` -- ``train()`` is a mode flag, not an unfreeze.

        **One thing it is not inert for, which is why this is a method and not a
        blanket ``train()``:** the checkpointed call site passes positional args
        only, so ``save_attn`` / ``att_mask`` / ``rgb_mask`` never reach the block
        in train mode. Both ``VGGT.forward`` and ``Aggregator.forward`` default
        ``save_attn=True``, so a caller that leaves the default and enables this
        would get a stale or missing attention map. RayTun3R pins
        ``save_attn=False`` (see :meth:`VGGTBackbone.forward`), which is what
        makes the switch safe here.
        """
        self.model.train(enabled)
        return True

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
                # clone, not a view: in DINOv2's fast path (npatch == N and w == h)
                # ``pos`` IS ``self.pos_embed``, so a view would make P_A track the
                # live table instead of pinning the pretrained one.
                outer._pe_table = pos[0, 1:].detach().clone()
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
        #
        # The autocast is not an optimisation, it is how VGGT is specified to
        # run, and this was the only caller in the repo omitting it. The
        # evidence is inside the model: ``vggt_visfeat/models/vggt.py:72``
        # wraps the heads in ``autocast(enabled=False)``, which is a no-op
        # unless a caller has opened one -- the model is written expecting a
        # bf16 aggregator and fp32 heads. Every other VGGT call site here
        # supplies it (``main_adt.py`` defaults ``--dtype bf16``,
        # ``main_erp_upstream.py`` hardcodes the autocast), and the sibling
        # backbones do too: ``vggt_omega/models/vggt_omega.py:41`` opens bf16
        # for its aggregator and disables it at :51 for its heads, and
        # ``depth_anything_3/api.py:126`` uses this exact dtype expression.
        #
        # fp16 rather than bf16 is a KNOWN VGGT failure mode -- the official
        # repo recommends bf16 and ``main_adt.py`` warns loudly when it falls
        # back -- so the fallback here is pre-Ampere only, where bf16 does not
        # exist.
        if images.is_cuda:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type="cuda", dtype=dtype):
                preds, _ = self.model(images, save_attn=False)
        else:
            preds, _ = self.model(images, save_attn=False)
        h, w = images.shape[-2:]
        extri, _ = pose_encoding_to_extri_intri(preds["pose_enc"], (h, w))
        extri = extri[0]                                   # (S, 3, 4) cam-from-world
        depth = preds["depth"][0, ..., 0]                  # (S, H, W)
        conf = preds.get("depth_conf", torch.ones_like(depth[:, None]))[0]
        return self._finalize(
            Prediction(depth=depth, conf=conf, R=extri[:, :3, :3], t=extri[:, :3, 3],
                       extra={"pose_enc": preds["pose_enc"]}))


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
        return self._finalize(
            Prediction(depth=depth, conf=conf, R=extri[:, :3, :3], t=extri[:, :3, 3],
                       extra={"pose_enc": preds["pose_enc"]}))


# ---------------------------------------------------------------------------
# Depth Anything 3 (paper's primary backbone; external dependency)
# ---------------------------------------------------------------------------


class DA3Backbone(Backbone):
    """Depth Anything 3 -- the paper's primary backbone (Tab. 1, 4, 7b are DA3-Small).

    Verified against ``depth_anything_3`` 0.1.1. Three things about that package
    shape the code below, all confirmed by building ``da3-small`` and running it:

    * ``DepthAnything3.forward`` wraps the whole call in ``torch.no_grad()`` and
      queries ``torch.cuda.is_bf16_supported()``. Adaptation needs gradients
      through the frozen model to reach the adapter, so this wraps the **inner**
      ``DepthAnything3Net`` and applies its own autocast. Going through the public
      wrapper would train nothing and crash on a CPU-only host.
    * The DPT positional grid is added by ``DualDPT._add_pos_embed``, not the
      ``_apply_pos_embed`` VGGT uses.
    * ``create_uv_grid``'s docstring claims ``(width, height, 2)`` here too, while
      ``_add_pos_embed`` consumes it as ``(height, width, 2)`` -- the same trap
      documented in :func:`corrections.camera_aware_uv_grid`.
    """

    patch_size = 14
    embed_dim = 384         # DA3-Small; re-read from the ViT in load()
    has_abs_pe = True       # DinoVisionTransformer.pos_embed
    has_rope = True         # RotaryPositionEmbedding2D, rope_start=4 on da3-small

    #: variant -> (config name, HuggingFace id)
    _VARIANTS = {
        "small": ("da3-small", "depth-anything/DA3-SMALL"),
        "base": ("da3-base", "depth-anything/DA3-BASE"),
        "large": ("da3-large", "depth-anything/DA3-LARGE"),
        "giant": ("da3-giant", "depth-anything/DA3-GIANT"),
    }

    @classmethod
    def load(cls, weights: Optional[str] = None, device="cpu", variant: str = "small",
             **kw) -> "DA3Backbone":
        """``weights=None`` builds the architecture with random init (for tests);
        ``"pretrained"`` fetches the variant's released checkpoint; anything else
        is passed to ``from_pretrained`` verbatim."""
        if variant not in cls._VARIANTS:
            raise ValueError(f"unknown DA3 variant {variant!r}; "
                             f"choose from {sorted(cls._VARIANTS)}")
        cfg_name, hub_id = cls._VARIANTS[variant]

        try:
            from depth_anything_3.cfg import create_object, load_config  # type: ignore
            from depth_anything_3.registry import MODEL_REGISTRY  # type: ignore
        except ImportError as exc:               # pragma: no cover - optional dep
            raise ImportError(
                "The 'da3' backbone needs the depth_anything_3 package "
                "(https://github.com/ByteDance-Seed/depth-anything-3). Install it, or "
                "use --backbone vggt, which is vendored in this repo."
            ) from exc

        if weights in (None, "random"):
            net = create_object(load_config(MODEL_REGISTRY[cfg_name]))
        else:
            repo = hub_id if weights == "pretrained" else weights
            try:
                from depth_anything_3.api import DepthAnything3  # type: ignore
                wrapper = DepthAnything3.from_pretrained(repo)
                # Unwrap: the public forward is no-grad, see the class docstring.
                net = wrapper.model
            except ImportError:
                # `api` drags in the video-export stack (moviepy et al.), which is
                # irrelevant here and absent on a plain install. The released repos
                # are just config.json + model.safetensors, so build the same
                # architecture from the registry and load the weights directly.
                net = cls._load_released_weights(cfg_name, repo, create_object,
                                                 load_config, MODEL_REGISTRY)

        obj = cls(net.to(device))
        vit = obj._vit()
        obj.embed_dim = int(vit.pos_embed.shape[-1])
        return obj

    @staticmethod
    def _load_released_weights(cfg_name, repo, create_object, load_config, registry):
        """Build from the registry config and load ``model.safetensors`` directly.

        Used only when ``depth_anything_3.api`` cannot be imported. Keeps the same
        architecture the wrapper would have built, so the hooks still find
        ``pos_embed`` and ``_add_pos_embed`` in the usual places.
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        net = create_object(load_config(registry[cfg_name]))
        sd = load_file(hf_hub_download(repo, "model.safetensors"))
        # The wrapper stores the net under `model.`; strip it if present.
        if any(k.startswith("model.") for k in sd):
            sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        missing, _ = net.load_state_dict(sd, strict=False)
        # The released checkpoints omit the auxiliary DPT output head ("_aux"),
        # which the depth/pose path never reads. Anything else missing would mean
        # a partly-initialised backbone, which must not run silently.
        core = [k for k in missing if "_aux" not in k]
        if core:
            raise RuntimeError(
                f"DA3 {cfg_name}: {len(core)} non-auxiliary tensors missing from "
                f"{repo}/model.safetensors, e.g. {sorted(core)[:3]}. Refusing to run a "
                f"partly-initialised backbone -- install depth_anything_3's full "
                f"dependencies so the official loader can be used.")
        return net

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
                outer._pe_table = pos[0, n_prefix:].detach().clone()
            if outer.adapter is None or outer.adapter.pe is None:
                return pos
            if n_prefix < 0:
                return pos
            res = outer.adapter.pe_residual().to(pos.dtype).to(pos.device)
            return torch.cat((pos[:, :n_prefix], pos[:, n_prefix:] + res.unsqueeze(0)), dim=1)

        self._patch_method(vit, "interpolate_pos_encoding", interpolate_pos_encoding)

    def _hook_dpt_grid(self, grid_mode: str) -> None:
        """Patch ``DualDPT._add_pos_embed``.

        The method is ``_add_pos_embed`` in ``depth_anything_3.model.dualdpt`` /
        ``dpt``, and the helpers live in ``model.utils.head_utils``. Both differ
        from VGGT's naming, and both were wrong here -- guarded by ``hasattr`` and
        a bare ``except ImportError``, so the correction silently did nothing
        rather than failing. Anything unexpected now raises.
        """
        from depth_anything_3.model.utils.head_utils import (  # type: ignore
            create_uv_grid, position_grid_to_embed,
        )

        targets = [m for m in self.model.modules() if hasattr(m, "_add_pos_embed")]
        if not targets:
            raise RuntimeError(
                "no DA3 head exposing _add_pos_embed; the package layout has changed. "
                "Re-check against depth_anything_3.model.dualdpt before trusting "
                "--no-dpt-grid ablations."
            )
        for mod in targets:
            self._patch_method(mod, "_add_pos_embed",
                               self._make_apply_pos_embed(grid_mode, create_uv_grid,
                                                          position_grid_to_embed))

    def forward(self, images: Tensor) -> Prediction:
        if images.dim() == 4:
            images = images.unsqueeze(0)
        # Autocast on CUDA only, and *not* under no_grad -- see the class docstring.
        if images.is_cuda:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type="cuda", dtype=dtype):
                out = self.model(images)
        else:
            out = self.model(images)

        depth = out["depth"].squeeze(0)                    # (B, S, H, W) -> (S, H, W)
        pose = out["extrinsics"].squeeze(0)                # (B, S, 3, 4) -> (S, 3, 4)
        # The key is 'depth_conf', matching the VGGT family; 'conf' does not exist
        # and silently fell back to ones.
        conf = out["depth_conf"].squeeze(0) if "depth_conf" in out else torch.ones_like(depth)
        return self._finalize(
            Prediction(depth=depth.float(), conf=conf.float(),
                       R=pose[:, :3, :3].float(), t=pose[:, :3, 3].float()))


# ---------------------------------------------------------------------------
# pi^3 -- the paper's second named-sequence backbone (Tab. 2)
# ---------------------------------------------------------------------------


class Pi3Backbone(Backbone):
    """``pi^3``: Scalable Permutation-Equivariant Visual Geometry Learning.

    The paper's Tab. 2 reports this backbone on the same named sequences as VGGT,
    so it doubles the number of *named-scene* targets available on ScanNet++
    ``3f15``: vanilla 6.17 / 19.7 / 38.6 and Center-PH 2.28 / 25.7 / 5.2, both
    training-free and both measured under the same protocol as the VGGT rows.

    Three facts about the upstream package shape the code below, all read off
    ``yyfz/Pi3@main``:

    * **It is not on PyPI**, but its ``pyproject.toml`` does declare a ``pi3``
      package, so ``pip install git+https://github.com/yyfz/Pi3.git`` works on
      Python >= 3.10. :meth:`load` tries the plain import first and falls back to
      finding a clone on ``sys.path``, the way :class:`VGGTBackbone` does.
    * **Only ``decoder_size='large'`` is coherent upstream.** The ``small``/``base``
      branches build a ``dec_embed_dim``-wide register token and concatenate it
      with the 1024-wide encoder output without a projection, so ``decode()``
      raises. ``large`` is what the released config uses, and the only value here.
    * ``pi3/models/layers/attention.py`` imports ``torch.nn.attention``, which
      landed in **torch 2.3** -- older torch fails at import, not at run time.
    * **``camera_poses`` is camera-to-world**, unlike VGGT's ``extrinsics`` and
      unlike everything else in this repo, which is cam-from-world. The forward
      below inverts it. Getting this backwards costs nothing at small rotations
      and grows with the baseline, which is the worst way for it to be wrong.
    * **Depth is ``local_points[..., 2]``**, i.e. planar z, so ``native_depth``
      stays ``"z"`` and :meth:`_finalize` does the conversion as usual.

    The RoPE fallback in ``pi3/models/layers/pos_embed.py`` is pure PyTorch and
    ``FlashAttentionRope`` calls ``torch.nn.functional.scaled_dot_product_attention``,
    so nothing here needs a CUDA extension compiled.
    """

    patch_size = 14
    embed_dim = 1024        # DINOv2 ViT-L/14 with registers
    has_abs_pe = True       # encoder is dinov2_vitl14_reg -> carries pos_embed
    has_rope = True         # decoder RoPE2D, pos_type='rope100'
    native_depth = "z"

    #: released weights; the repo's config is {"decoder_size": "large",
    #: "pos_type": "rope100"} and the class is a PyTorchModelHubMixin.
    HUB_ID = "yyfz233/Pi3"

    @classmethod
    def _import_pi3(cls):
        """``pi3.models.pi3.Pi3``, from a pip install or from a clone on disk."""
        import os
        import sys
        from pathlib import Path

        try:
            from pi3.models.pi3 import Pi3  # type: ignore
            return Pi3
        except ImportError:
            pass

        env = os.environ.get("PI3_ROOT")
        candidates = ([Path(env)] if env else []) + [
            Path(__file__).resolve().parents[1] / "Pi3",
            Path(__file__).resolve().parents[2] / "Pi3",
        ]
        for c in candidates:
            if (c / "pi3" / "models" / "pi3.py").exists():
                if str(c) not in sys.path:
                    sys.path.insert(0, str(c))
                from pi3.models.pi3 import Pi3  # type: ignore
                return Pi3
        raise ImportError(
            "The 'pi3' backbone needs https://github.com/yyfz/Pi3, which is not on "
            "PyPI. Either\n"
            "    pip install git+https://github.com/yyfz/Pi3.git      (needs Python >= 3.10)\n"
            "or clone it beside this repo / point $PI3_ROOT at it:\n"
            "    git clone https://github.com/yyfz/Pi3.git\n"
            "It also needs torch >= 2.3 for torch.nn.attention. Searched: "
            + ", ".join(str(c) for c in candidates))

    @classmethod
    def load(cls, weights: Optional[str] = None, device="cpu", **kw) -> "Pi3Backbone":
        Pi3 = cls._import_pi3()
        kw.setdefault("decoder_size", "large")   # the only coherent branch upstream

        if weights in (None, "random"):
            model = Pi3(**kw)                    # architecture only, for tests
        else:
            model = Pi3.from_pretrained(cls.HUB_ID if weights == "pretrained" else weights)
        obj = cls(model.to(device))
        obj.embed_dim = int(obj._vit().pos_embed.shape[-1])
        return obj

    def _vit(self) -> nn.Module:
        return self.model.encoder

    def _tokenizer_modules(self) -> List[nn.Module]:
        return [self._vit().patch_embed]

    def _hook_abs_pe(self) -> None:
        # DINOv2's `interpolate_pos_encoding` returns (1, 1+N, C) with the class
        # token first -- the same contract VGGT's does. Registers are concatenated
        # *after* this call in `prepare_tokens_with_masks` and carry their own
        # embedding, so they are untouched by Eq. 5, which is correct: they have
        # no image position.
        vit = self._vit()
        outer = self

        def interpolate_pos_encoding(self, x, w, h, _orig=vit.interpolate_pos_encoding):
            pos = _orig(x, w, h)
            gh, gw = outer._grid
            if pos.shape[1] - 1 == gh * gw and outer._pe_table is None:
                # clone, not a view: in DINOv2's fast path (npatch == N and w == h)
                # ``pos`` IS ``self.pos_embed``, so a view would make P_A track the
                # live table instead of pinning the pretrained one.
                outer._pe_table = pos[0, 1:].detach().clone()
            if outer.adapter is None or outer.adapter.pe is None:
                return pos
            if pos.shape[1] - 1 != gh * gw:
                return pos
            res = outer.adapter.pe_residual().to(pos.dtype).to(pos.device)
            return torch.cat((pos[:, :1], pos[:, 1:] + res.unsqueeze(0)), dim=1)

        self._patch_method(vit, "interpolate_pos_encoding", interpolate_pos_encoding)

    def _hook_dpt_grid(self, grid_mode: str) -> None:
        # pi^3's heads are `LinearPts3d`, not a DPT with `_apply_pos_embed`: there
        # is no positional grid inside the head to make camera-aware. So this
        # parameter-free correction has no attachment point here, and silently
        # doing nothing is the honest behaviour -- but it must be visible, because
        # a `--backbone pi3` run is then not comparable to a VGGT/DA3 run that had
        # `dpt_grid` on.
        warnings.warn(
            "pi3 has no DPT positional grid to correct (its heads are LinearPts3d); "
            "the dpt_grid correction is a no-op for this backbone. Pass "
            "--no-dpt-grid to make that explicit in the run config.", RuntimeWarning)

    def forward(self, images: Tensor) -> Prediction:
        if images.dim() == 4:
            images = images.unsqueeze(0)
        out = self.model(images)

        # camera-to-world -> cam-from-world, the convention used everywhere here.
        c2w = out["camera_poses"][0].float()               # (S, 4, 4)
        R_c2w, t_c2w = c2w[:, :3, :3], c2w[:, :3, 3]
        R = R_c2w.transpose(-1, -2)
        t = -torch.einsum("sij,sj->si", R, t_c2w)

        depth = out["local_points"][0, ..., 2].float()     # (S, H, W) planar z
        conf = out["conf"][0, ..., 0].float()              # (S, H, W)
        return self._finalize(Prediction(depth=depth, conf=conf, R=R, t=t))


BACKBONES = {
    "vggt": VGGTBackbone,
    "vggt_omega": VGGTOmegaBackbone,
    "da3": DA3Backbone,
    "pi3": Pi3Backbone,
}

#: Single source of truth for every ``--backbone`` choice list. Adding a backbone
#: to BACKBONES without touching seven argparsers is the point.
BACKBONE_NAMES = sorted(BACKBONES)


def build_backbone(name: str, weights: Optional[str] = None, device="cpu", **kw) -> Backbone:
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
    return BACKBONES[name].load(weights=weights, device=device, **kw)
