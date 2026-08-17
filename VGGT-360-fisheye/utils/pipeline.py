# Copyright (c) 2026.
"""The VGGT-360-fisheye forward pass, as one object three drivers can call.

    fisheye frame  ->  9 tangent views  ->  one VGGT pass  ->  fused range map

This is the port's per-frame core, lifted out of ``main_adt.py`` when the second
and third callers appeared. The three are:

    ``main_adt.py``            ADT, dense GT, scored on the fisheye grid
    ``fovbench.models``        the ADT-FOV test, as a model beside the vanilla four
    ``slambench.baselines``    the SLAM evaluation, as a lens strategy beside
                               ``raw`` and ``rect_derect``

They differ in what they do with the fused map, not in how it is produced, and a
number from one of them is a number about the same pipeline as the others. That
is the whole reason this file exists rather than a copy in each: the alternative
was three implementations of a method whose result is a comparison *between*
those three tables.

What travels and what does not
------------------------------
The three modules of the paper all live here — adaptive view generation,
structure-saliency attention, correlation-weighted fusion — and so does the
depth-head ``z -> range`` secant conversion, because it is a property of how the
views were rendered rather than of any dataset.

**Depth convention does not.** This returns euclidean *range* along each ray,
which is what fusion produces and what neither ADT nor ego-synth stores: both
store planar z, and both have been *measured* to —
``checks/check_gt_depth_domain.py`` RANSAC-fits ADT's scene planes and peaks at
the z hypothesis, and ``slambench/verify_depth_convention.py`` (ticket 016,
closed 2026-08-14) puts ego-synth's residual at 0.0002 flat across incidence
angle, which is the float16 noise floor of the stored value.

So every caller converts, and :func:`range_to_planar_z` is what they call. It is
not applied here because the *mask* differs: ADT scores dense pixels and
ego-synth a point list, and one of them also scores a ``range`` domain for
comparison with Depth-Any-Camera's published protocol.

Lenses other than Aria KB4
--------------------------
:meth:`VGGT360Pipeline.range_map` takes a ``lens``, not a hardcoded camera. The
ADT path passes :class:`utils.fisheye_cam.FisheyeCam` and gets the KB4 geometry
this port was written on. ego-synth passes a FISHEYE624 adapter, whose
tangential and thin-prism terms are not radially symmetric and so have no KB4
equivalent — fitting one would put a lens error inside the warp the method is
made of. See ``slambench/vggt360.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from .att_utils import SA_confidence
from .fisheye_cam import fisheye_ray_lut
from .fisheye_fusion import build_selfview_confidence, fuse_views_to_fisheye
from .fisheye_views import fisheye_to_persp, view_generation_fisheye

#: ``(azimuth, tilt, fov)`` degrees.
ViewParam = Tuple[float, float, float]

#: The port's own layout, and what "the 60 deg model" means throughout this
#: repository: one centre view plus an 8-direction ring, each 60 deg wide,
#: tilted 26 deg off the optical axis. The rule is ``tilt + fov/2 >~ 54.8``, the
#: Aria *usable* cone — not the 62.33 deg KB4 fold-back turnover, which is a
#: property of the fitted polynomial and about 7.5 deg past where the lens
#: actually images. These are ``main_adt.py``'s defaults, restated here rather
#: than in an argparse block so that a driver which never sees that CLI still
#: gets the same model.
DEFAULT_FOV_DEG = 60.0
DEFAULT_RING_TILT_DEG = 26.0
DEFAULT_N_RING = 8


@dataclass
class VGGT360Config:
    """Everything that makes one configuration of the pipeline a *model*.

    The defaults are ``main_adt.py``'s, so a driver that constructs this with no
    arguments is running the configuration every VGGT-360-fisheye number in this
    repository was produced under.
    """

    #: View layout (module 1).
    fov: float = DEFAULT_FOV_DEG
    ring_tilt: float = DEFAULT_RING_TILT_DEG
    n_ring: int = DEFAULT_N_RING
    adaptive: bool = True
    max_views: int = 13
    #: Side length each tangent view is rendered at. **518, the backbone's own
    #: token grid** (patch 14 x 37), so that nothing is resampled between the
    #: view construction and the network — the same rule ``fovbench.models``
    #: applies to the four vanilla models.
    #:
    #: ``main_adt.py`` passes **512** instead and keeps doing so, because every
    #: VGGT-360-fisheye number in this repository was produced at that value.
    #: 512 is not a multiple of 14, so ``load_and_preprocess_images`` (mode
    #: "crop", target 518) bicubic-resizes each view up by 1.0117x on the way
    #: in: a small blur, applied to all nine views, that buys nothing. Rendering
    #: at 518 removes that step rather than compensating for it. Whether to move
    #: that driver too is a decision about re-running its published table, not a
    #: decision about which value is right.
    persp_size: int = 518
    crop_supersample: int = 3

    #: Structure-saliency attention bias (module 2).
    sa_mask: bool = True

    #: Fusion (module 3). ``attn`` is the paper's correlation weighting;
    #: ``mean`` is the uniform-weight ablation.
    fuse: str = "attn"
    erode_valid_px: int = 3

    #: Range source. The depth head's planar z times the view secant is
    #: measurably less bumpy on ADT than the point head's ``||world_points||``.
    head: str = "depth"

    #: Backbone. Only VGGT-1B through the vendored ``vggt_visfeat`` can run
    #: this: ``fuse="attn"`` reads frame attention off a 37x37 patch grid and
    #: ``sa_mask`` injects a per-view log-bias into it, neither of which any
    #: model-zoo adapter exposes. :meth:`check` says so up front.
    model_path: str = "facebook/VGGT-1B"
    dtype: str = "bf16"

    def check(self) -> "VGGT360Config":
        if self.fuse not in ("attn", "mean"):
            raise ValueError(f"fuse must be 'attn' or 'mean', got {self.fuse!r}")
        if self.head not in ("depth", "point"):
            raise ValueError(f"head must be 'depth' or 'point', got {self.head!r}")
        if self.dtype not in ("bf16", "fp16", "fp32"):
            raise ValueError(f"dtype must be bf16/fp16/fp32, got {self.dtype!r}")
        if self.n_ring < 1:
            raise ValueError(f"n_ring must be >= 1, got {self.n_ring}")
        return self

    def describe(self) -> str:
        """One line for a report, naming everything that could change a number."""
        mods = [f"fov{self.fov:g}", f"tilt{self.ring_tilt:g}", f"ring{self.n_ring}",
                f"fuse={self.fuse}", f"head={self.head}",
                "adaptive" if self.adaptive else "no-adaptive",
                "sa-mask" if self.sa_mask else "no-sa-mask",
                f"{self.persp_size}px", self.dtype]
        return " ".join(mods)

    def covers_cone(self, theta_max_deg: float) -> float:
        """Share of a ``theta_max_deg`` cone the ring reaches, radially.

        The layout rule in one number, so a driver on a lens this port was not
        designed around can print how far the ring actually gets instead of
        assuming it tiles. ``1.0`` means the ring's outer edge reaches the rim.
        """
        return float((self.ring_tilt + self.fov / 2.0) / max(theta_max_deg, 1e-6))


@dataclass
class ViewPass:
    """What one VGGT pass produced, before anything is fused.

    The pipeline is exposed in two halves rather than one call because
    ``main_adt.py`` works *between* them: it measures cross-view scale spread
    and can least-squares harmonise the per-view scales before fusion. A single
    ``range_map`` would have forced that driver to keep its own copy of the
    forward pass, which is the duplication this module exists to end.
    """

    params: List[ViewParam]
    #: ``(S, h, w)`` euclidean range per view, in each view's own frame.
    radial: np.ndarray
    #: Per-view RGB as handed to the network, and the analytic valid masks.
    images: List[np.ndarray] = field(default_factory=list)
    valids: List[np.ndarray] = field(default_factory=list)
    #: ``(S, H, W)`` attention-derived fusion weights, or ``None`` for ``mean``.
    weights: Optional[List[np.ndarray]] = None
    #: VGGT's own per-view pose encoding, when the head emitted one.
    pose_enc: Optional[np.ndarray] = None


@dataclass
class FusedFrame:
    """One frame's fused prediction, plus what a report needs to qualify it."""

    #: ``(H, W)`` euclidean range along each fisheye ray. **Not planar z** —
    #: see the module docstring, and :func:`range_to_planar_z`.
    range_map: np.ndarray
    #: ``(H, W)`` int, how many views contributed to each pixel.
    coverage: np.ndarray
    #: ``(H, W)`` bool, the lens' imaged cone.
    cone: np.ndarray
    view_params: List[ViewParam] = field(default_factory=list)

    @property
    def n_views(self) -> int:
        return len(self.view_params)

    @property
    def covered(self) -> np.ndarray:
        """Pixels that are inside the cone *and* that some view answered for."""
        return self.cone & (self.coverage > 0) & np.isfinite(self.range_map) \
            & (self.range_map > 0)

    @property
    def cone_coverage(self) -> float:
        """Share of the imaged cone the fusion answered for. Expect ~0.996-1.0.

        Worth printing rather than assuming: the erosion at fusion retires a
        thin band at the rim of every view at once, and although ``rescue_rim``
        puts most of it back, a layout that does not tile its cone would show up
        here first and nowhere else.
        """
        n = int(self.cone.sum())
        return float(self.covered.sum()) / n if n else float("nan")


def range_to_planar_z(range_map: np.ndarray, cos_theta: np.ndarray) -> np.ndarray:
    """``range * cos(theta)`` — euclidean range along a ray to planar z.

    The conversion is radial, so it is **not** absorbable by any scale-and-shift
    alignment: on this lens it is 1.00 on axis and about 1.74x at 55 deg. Scoring
    the fused range against a planar-z ground truth without it is the error
    CONTEXT.md records as having cost this port a result once.

    Why range is the quantity fusion works in
    -----------------------------------------
    Not because the answer has to be range — it does not, and this function is
    how it stops being. Because range is what the nine views can be *added up*
    in.

    The views share one optical centre, so a given fisheye ray is seen by
    several of them and fusion averages their answers for it. Euclidean range is
    the same number in every view that sees that ray. Planar z about a **view's
    own axis** is not: it is ``range * cos(angle to that view's axis)``, and that
    angle differs per view — up to ``sec(30 deg) = 1.155`` within a single 60 deg
    view. Averaging those would blend nine different quantities. It is also what
    makes ``head="depth"`` and ``head="point"`` interchangeable at all: the point
    head's ``||world_points||`` is range natively, and the depth head reaches the
    same quantity through the view secant.

    Planar z about the **camera** axis would have served equally — it too is a
    property of the ray rather than of the view. Converting after fusion rather
    than before is not merely equivalent but better: ``cos theta`` is then read
    exactly, once, per output ray, instead of being bilinearly interpolated on
    each view's grid. Measured on a synthetic field the two orders agree to
    2.5e-5 through the interior and part company only in the rim band, where a
    single-view pixel sampled across the validity boundary reaches 15%. So: fuse
    in range, convert on the output grid.
    """
    return (np.asarray(range_map, np.float32)
            * np.asarray(cos_theta, np.float32)).astype(np.float32)


def fill_uncovered(pred: np.ndarray, covered: np.ndarray,
                   where: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
    """Give the uncovered pixels of ``where`` a finite value. Returns (map, n).

    Fusion answers for the cone and not for the square frame's corners, but a
    harness that owns its own validity mask cannot be told that: it will fit an
    affine over its mask, and one NaN inside that mask breaks the fit. How it
    breaks depends on the mode and neither way is survivable — ``scale_shift``
    raises ``LinAlgError`` out of ``lstsq`` and ends the run, ``scale_only``
    returns an all-NaN frame that is then silently dropped
    (``fovbench/tests/test_vggt360.py`` pins both).

    A caller that *does* own the mask should pass the NaN through instead;
    ``slambench`` does exactly that, because there an unanswered point is
    information the report already knows how to carry.

    The fill is the **median of what was covered**, a constant. A constant adds
    no structure, so it cannot manufacture the radial trend these benchmarks
    look for; it only dilutes. The count comes back so the caller can report it
    rather than discover it, and on this pipeline it is a fraction of a percent.
    """
    out = np.asarray(pred, np.float32).copy()
    good = np.asarray(covered, bool) & np.isfinite(out)
    hole = (~good if where is None else (np.asarray(where, bool) & ~good))
    n = int(hole.sum())
    if n:
        out[hole] = float(np.median(out[good])) if good.any() else 1.0
    return out, n


class VGGT360Pipeline:
    """The loaded network plus the three modules around it.

    Load once, call :meth:`range_map` per frame. The object holds a torch model,
    so it is not thread-safe; ``fovbench`` serialises the forward pass behind its
    own lock for exactly this reason.
    """

    def __init__(self, cfg: Optional[VGGT360Config] = None, device: str = "cuda"):
        self.cfg = (cfg or VGGT360Config()).check()
        self.device = str(device)
        self.model = None
        self._secant: dict = {}

    # -- loading ------------------------------------------------------------ #
    def load(self, verbose: bool = True) -> "VGGT360Pipeline":
        """Bring up VGGT-1B and check the weights actually landed.

        ``PyTorchModelHubMixin`` loads non-strictly, so a key mismatch leaves
        layers at random init and the pipeline returns confident nonsense rather
        than failing. The check is cheap and the failure it catches is not.
        """
        from vggt_visfeat.models.vggt import VGGT

        if verbose:
            print(f"[vggt360] loading {self.cfg.model_path} "
                  f"({self.cfg.describe()})")
        self.model = VGGT.from_pretrained(self.cfg.model_path).to(self.device).eval()
        if verbose:
            self._weight_check()
        return self

    def _weight_check(self) -> None:
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            ckpt = load_file(hf_hub_download(self.cfg.model_path,
                                             "model.safetensors"))
            keys = set(self.model.state_dict().keys())
            missing = sorted(keys - set(ckpt.keys()))
            # ``track_head.*`` is deliberately absent (the unused tracker was
            # removed); those checkpoint keys are not an error.
            unexpected = sorted(k for k in set(ckpt.keys()) - keys
                                if not k.startswith("track_head."))
            if missing or unexpected:
                print(f"[vggt360]   WEIGHT CHECK FAILED: {len(missing)} model "
                      f"keys not in checkpoint (random init!), "
                      f"{len(unexpected)} checkpoint keys unused. "
                      f"First few missing: {missing[:5]}")
            else:
                print(f"[vggt360]   weight check OK: {len(keys)} keys matched")
        except Exception as e:                    # local path / no safetensors
            print(f"[vggt360]   weight check skipped ({type(e).__name__}: {e})")

    @property
    def num_params(self) -> float:
        if self.model is None:
            return float("nan")
        return sum(p.numel() for p in self.model.parameters())

    # -- the forward pass --------------------------------------------------- #
    def _torch_dtype(self):
        import torch
        d = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[self.cfg.dtype]
        if d is torch.bfloat16 and self.device == "cuda" \
                and torch.cuda.get_device_capability()[0] < 8:
            # fp16 quality is a known VGGT failure mode; never fall back silently.
            print("[vggt360] pre-Ampere GPU: bf16 -> fp16. Expect noisy depth; "
                  "--dtype fp32 rules mixed precision out.")
            d = torch.float16
        return d

    def view_secant(self, fov_deg: float, H: int, W: int) -> np.ndarray:
        """Per-pixel ``sqrt(1 + x^2 + y^2)`` of a view's tangent grid.

        Turns the depth head's planar z about the *view* axis into euclidean
        range along each pixel ray, which is what fusion adds up. The point head
        needs none of this (``||world_points||`` is already range) and is
        empirically noisier, which is what ``head`` exists to test.
        """
        key = (round(float(fov_deg), 3), int(H), int(W))
        if key not in self._secant:
            t = math.tan(math.radians(fov_deg) / 2.0)
            xs = np.linspace(-t, t, W, dtype=np.float32)
            ys = np.linspace(-t, t, H, dtype=np.float32)
            xv, yv = np.meshgrid(xs, ys)
            self._secant[key] = np.sqrt(1.0 + xv * xv + yv * yv).astype(np.float32)
        return self._secant[key]

    def render_views(self, rgb: np.ndarray, lens,
                     project: Optional[Callable] = None,
                     maps_for: Optional[Callable] = None):
        """Module 1: the view layout, rendered, with its analytic valid masks.

        ``maps_for(azimuth, tilt, fov, size) -> (mapx, mapy, valid)`` is an
        optional cache of the warp geometry, which depends on the lens and the
        aim but never on the pixels. It changes no number — see
        ``fisheye_views.fisheye_to_persp``'s ``maps`` — and on a lens without a
        closed-form projection it is the difference between a run and a week.
        """
        cfg = self.cfg
        ss = max(1, int(cfg.crop_supersample))
        params = view_generation_fisheye(
            rgb, lens, fov_deg=cfg.fov, ring_tilt_deg=cfg.ring_tilt,
            n_ring=cfg.n_ring, adaptive=cfg.adaptive,
            max_total=cfg.max_views,
            view_hw=(cfg.persp_size, cfg.persp_size),
            supersample=cfg.crop_supersample, project=project,
            maps_for=maps_for)
        imgs, sa, valids = [], [], []
        for (psi, tilt, fov) in params:
            persp, valid = fisheye_to_persp(
                rgb, lens, psi, tilt, fov, height=cfg.persp_size,
                width=cfg.persp_size, supersample=cfg.crop_supersample,
                project=project,
                maps=None if maps_for is None
                else maps_for(psi, tilt, fov, cfg.persp_size * ss))
            s, vm = SA_confidence(persp, valid_mask=valid > 0.5)
            imgs.append(persp)
            sa.append(s)
            valids.append(vm)
        return params, imgs, sa, valids

    def _check_hooks(self, project, ray_lut) -> None:
        """The two lens hooks are asymmetric, and only one pairing is an error.

        ``project`` without ``ray_lut`` renders the views through the caller's
        lens and then fuses them through KB4 — two different cameras in one
        frame, which is a silent geometry error rather than a partial
        configuration, and is refused.

        ``ray_lut`` without ``project`` is the opposite case and is allowed: the
        KB4 render is correct, and the LUT is simply the one the caller already
        built. ``fovbench`` passes exactly that, so the ``cos(theta)`` it
        converts with and the ``theta`` it bins by are one array rather than two
        computations that could drift. A LUT that did *not* come from this lens
        would be wrong here — the shape is checked at fusion, the values are the
        caller's responsibility, and ``fisheye_ray_lut`` is memoised so there is
        no reason to build a second one.
        """
        if self.model is None:
            raise RuntimeError("VGGT360Pipeline.load() was never called")
        if project is not None and ray_lut is None:
            raise ValueError(
                "project= was given without ray_lut=: the views would be "
                "rendered through your lens and fused through KB4. Pass the "
                "matching ray_lut, or neither.")

    def predict_views(self, rgb: np.ndarray, lens,
                      project: Optional[Callable] = None,
                      maps_for: Optional[Callable] = None) -> ViewPass:
        """Modules 1 and 2: render the layout, run VGGT once over all of it.

        Returns per-view euclidean range, still in each view's own frame. Fusing
        it is :meth:`fuse`, and the gap between the two is where a caller may
        harmonise scales.
        """
        import torch
        from PIL import Image
        from vggt_visfeat.utils.load_fn2 import load_and_preprocess_images

        if self.model is None:
            raise RuntimeError("VGGT360Pipeline.load() was never called")

        cfg = self.cfg
        params, imgs, sa, valids = self.render_views(rgb, lens, project,
                                                    maps_for)

        # -- module 2: one multi-view pass with mask-biased attention -------- #
        pil = [Image.fromarray(np.clip(p, 0, 255).astype(np.uint8)) for p in imgs]
        images = load_and_preprocess_images(pil).to(self.device)
        persp_masks = None if not cfg.sa_mask else torch.from_numpy(np.array(sa))
        rgb_masks = None if not cfg.sa_mask else torch.from_numpy(np.array(valids))
        want_attn = cfg.fuse == "attn"
        dtype = self._torch_dtype()
        autocast = self.device == "cuda" and dtype is not torch.float32

        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=dtype,
                                enabled=autocast):
                predictions, attention = self.model(
                    images=images, persp_masks=persp_masks,
                    rgb_masks=rgb_masks, save_attn=want_attn)

            if cfg.head == "depth":
                z = predictions["depth"][0, ..., 0].float().cpu().numpy()
                radial = np.stack([z[i] * self.view_secant(params[i][2],
                                                           *z[i].shape)
                                   for i in range(z.shape[0])]).astype(np.float32)
            else:
                wp = predictions["world_points"][0].float().cpu().numpy()
                radial = np.linalg.norm(wp, axis=-1).astype(np.float32)

            weights = None
            if want_attn:
                w = build_selfview_confidence(attention)[:, 0, :, :].cpu().numpy()
                weights = [w[i] for i in range(w.shape[0])]
            pose_enc = (predictions["pose_enc"][0].float().cpu().numpy()
                        if "pose_enc" in predictions else None)

        del predictions, attention
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return ViewPass(params=list(params), radial=radial, images=imgs,
                        valids=valids, weights=weights, pose_enc=pose_enc)

    def fuse(self, vp: ViewPass, lens,
             ray_lut: Optional[Tuple[np.ndarray, np.ndarray]] = None
             ) -> FusedFrame:
        """Module 3: the per-view ranges, back onto the fisheye grid."""
        fused, coverage = fuse_views_to_fisheye(
            [vp.radial[i] for i in range(vp.radial.shape[0])], vp.params, lens,
            weights=vp.weights, view_valids=vp.valids, interp="linear",
            erode_valid_px=self.cfg.erode_valid_px, ray_lut=ray_lut)
        cone = (ray_lut[1] if ray_lut is not None
                else fisheye_ray_lut(lens)[1]).astype(bool)
        return FusedFrame(range_map=fused.astype(np.float32), coverage=coverage,
                          cone=np.asarray(cone, bool), view_params=list(vp.params))

    def range_map(self, rgb: np.ndarray, lens,
                  project: Optional[Callable] = None,
                  ray_lut: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                  maps_for: Optional[Callable] = None,
                  out_lens=None,
                  out_ray_lut: Optional[Tuple[np.ndarray, np.ndarray]] = None
                  ) -> FusedFrame:
        """One fisheye frame -> one fused euclidean-range map.

        Parameters
        ----------
        rgb    : ``(H, W, 3)`` uint8 fisheye frame, upright.
        lens   : the camera describing **this frame** — the one the tangent
                 views are cut out of. ``FisheyeCam``, or any object with ``H``,
                 ``W``, ``theta_max()`` and, when ``project``/``ray_lut`` are
                 given, nothing else.
        project, ray_lut : the non-KB4 hooks (see the module docstring). Supply
                 both or neither: one alone would render the views through one
                 lens and fuse them through another, which is a geometry error
                 that looks exactly like a mediocre model.
        out_lens, out_ray_lut : the grid the answer is *delivered* on, when that
                 is not the grid the frame arrived on. This is the one place the
                 pipeline reads at one resolution and writes at another, and it
                 exists because the two are genuinely different questions: the
                 views should be cut from the sharpest pixels available, while
                 the answer has to land on whatever grid the caller scores. The
                 ADT-FOV harness renders its views from the native 1408 frame and
                 fuses onto its 518 scoring grid; ``main_adt.py`` and
                 ``slambench`` read and write the same grid and leave these
                 ``None``.
        """
        self._check_hooks(project, ray_lut)
        vp = self.predict_views(rgb, lens, project, maps_for)
        if out_lens is None:
            return self.fuse(vp, lens, ray_lut)
        return self.fuse(vp, out_lens, out_ray_lut)
