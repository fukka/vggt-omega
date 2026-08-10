# Copyright (c) 2026.
"""A registry of monocular-depth models to benchmark on ADT, with a uniform
adapter interface, availability checks, and (best-effort) downloads.

Each :class:`ModelSpec` describes one model *variant* (a family + size). An
:class:`Adapter` wraps it behind two prediction calls the benchmark runner uses:

    adapter.predict_frame(rgb01, cam, frame) -> depth_z [H,W]   (planar z, frame px)
    adapter.predict_erp(rgb01, cam)          -> {"depth","active"}  (DAC only)

``kind`` routes to the wrapper implementation:

    "dac"          -- Depth-Any-Camera (ERP-native metric); reuse ``DACPredictor``.
    "unik3d"       -- UniK3D (any-camera metric); reuse ``UniK3DPredictor``.
    "hf_depth"     -- anything exposed via ``transformers.AutoModelForDepthEstimation``
                      (Depth-Anything V2/V3, MiDaS/DPT, ZoeDepth, Depth Pro, ...).
    "metric3d_hub" -- Metric3D v2 via ``torch.hub`` (best-effort).
    "vggt" /
    "vggt_omega" /
    "da3"          -- 3D foundation models, wrapped by ``raytun3r.backbones``,
                      run **single-view** so they sit in the same monocular
                      comparison as the rows above. See ``BackboneAdapter``.

Everything imports lazily, so listing the registry and checking availability work
without transformers / the third-party repos installed, and one missing dependency
never breaks the others.

Output convention
-----------------
``predict_frame`` always returns a positive *planar-z* depth map in the input
frame. For ``output_type == "relative"`` models (disparity/inverse-depth nets) the
returned value is ``1/disparity`` (an up-to-scale depth) so the disparity-space
alignment in :mod:`finetune.eval.metrics` recovers scale+shift correctly. For
``"metric"`` models it is metres.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .aria_fisheye import AriaFisheye

# Repo root (…/vggt-omega) and the conventional checkpoints dir.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_CKPT = os.path.join(_REPO, "checkpoints")


# --------------------------------------------------------------------------- #
# Persistent HuggingFace cache (repo-local)
# --------------------------------------------------------------------------- #
# The GPU box runs from a container image: ``~/.cache`` is wiped whenever the
# image reloads, but the repo (working dir) persists. DAC weights already live
# under ``<repo>/checkpoints/``; mirror that for every HF model (Depth-Anything,
# MiDaS, ZoeDepth, UniK3D, …) by pointing HuggingFace at ``<repo>/checkpoints/hf``.
# We set HF_HOME/HF_HUB_CACHE *before* huggingface_hub is imported anywhere
# (model_zoo is benchmark_adt's first import), so transformers AND UniK3D's
# internal ``from_pretrained`` land in the repo automatically. ``download()`` still
# *searches* the old default cache and copies from it instead of re-downloading.
def _default_hf_hub_cache() -> str:
    """Where HF would cache by default — mirror huggingface_hub's resolution
    WITHOUT importing it (importing freezes the constant before we can redirect)."""
    if os.environ.get("HF_HUB_CACHE"):
        return os.path.expanduser(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return os.path.join(os.path.expanduser(os.environ["HF_HOME"]), "hub")
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(os.path.expanduser(base), "huggingface", "hub")


_DEFAULT_HF_HUB = _default_hf_hub_cache()   # often ~/.cache/huggingface/hub (ephemeral)
if os.environ.get("VGGT_HF_HOME"):                       # explicit override → redirect
    _HF_HOME, _redirect = os.path.expanduser(os.environ["VGGT_HF_HOME"]), True
elif os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE"):
    # user already picked a cache (assumed persistent) — respect it, don't redirect
    _HF_HOME, _redirect = (os.environ.get("HF_HOME") or os.path.dirname(_DEFAULT_HF_HUB)), False
else:
    _HF_HOME, _redirect = os.path.join(_REPO, "checkpoints", "hf"), True
if _redirect:
    os.environ["HF_HOME"] = _HF_HOME
    _HF_HUB = os.path.join(_HF_HOME, "hub")
    os.environ["HF_HUB_CACHE"] = _HF_HUB
else:
    _HF_HUB = _DEFAULT_HF_HUB
try:
    os.makedirs(_HF_HUB, exist_ok=True)
except Exception:
    pass

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    key: str                       # unique cli key, e.g. "dav2_large"
    family: str                    # "Depth-Anything-V2"
    size: str                      # "Large", "ViT-L", "Swin-L", ...
    kind: str                      # dac | unik3d | hf_depth | metric3d_hub
    ref: str                       # hf id / variant / hub entrypoint
    output_type: str               # "metric" | "relative"
    align_native: Tuple[str, ...]  # native alignment(s); "none" is always added
    on_device: bool = False        # small / mobile-friendly
    experimental: bool = False     # uncertain availability (id may not exist yet)
    note: str = ""

    @property
    def align_modes(self) -> Tuple[str, ...]:
        """('none',) + native, de-duplicated, order preserved."""
        out: List[str] = ["none"]
        for m in self.align_native:
            if m not in out:
                out.append(m)
        return tuple(out)


# disparity-space affine for relative (MiDaS/Depth-Anything); median scale for
# metric; depth-space affine for up-to-scale DEPTH heads (the VGGT family and
# DA3 emit depth, not disparity, so a disparity-space fit would be the wrong
# protocol and a median scale cannot absorb their offset).
_REL = ("disparity_scale_shift",)
_MET = ("scale_only",)
_AFF = ("scale_shift",)

#: Local VGGT-Omega checkpoint (gated; no automatic download). Override with
#: ``$VGGT_OMEGA_CKPT`` or ``build_adapter(spec, checkpoint=...)``.
OMEGA_CKPT = os.environ.get(
    "VGGT_OMEGA_CKPT", os.path.join(_CKPT, "VGGT-Omega-1B-512", "model.pt"))

REGISTRY: List[ModelSpec] = [
    # ── 3D foundation models, run single-view ───────────────────────────────
    ModelSpec("vggt_1b", "VGGT", "1B", "vggt", "facebook/VGGT-1B",
              "up_to_scale", _AFF, note="vendored vggt_visfeat; DINOv2 + RoPE"),
    ModelSpec("vggt_omega", "VGGT-Omega", "1B/512", "vggt_omega", OMEGA_CKPT,
              "up_to_scale", _AFF, note="local .pt (gated); DINOv3, RoPE only"),
    ModelSpec("da3_large", "Depth-Anything-3", "Large", "da3", "large",
              "up_to_scale", _AFF, note="needs the depth_anything_3 package"),
    ModelSpec("da3_small", "Depth-Anything-3", "Small", "da3", "small",
              "up_to_scale", _AFF, on_device=True),
    # ── Depth-Any-Camera (ERP-native, metric, fisheye/360) ──────────────────
    ModelSpec("dac_swinl_indoor", "Depth-Any-Camera", "Swin-L", "dac",
              "dac_swinl_indoor", "metric", _MET, note="indoor; best on Aria fisheye"),
    ModelSpec("dac_rn101_indoor", "Depth-Any-Camera", "RN101", "dac",
              "dac_resnet101_indoor", "metric", _MET, on_device=True, note="indoor; lighter"),
    # ── UniK3D (any-camera metric) ──────────────────────────────────────────
    ModelSpec("unik3d_vits", "UniK3D", "ViT-S", "unik3d", "vits", "metric", _MET, on_device=True),
    ModelSpec("unik3d_vitb", "UniK3D", "ViT-B", "unik3d", "vitb", "metric", _MET),
    ModelSpec("unik3d_vitl", "UniK3D", "ViT-L", "unik3d", "vitl", "metric", _MET),
    # ── Depth-Anything V2 (relative) ────────────────────────────────────────
    ModelSpec("dav2_small", "Depth-Anything-V2", "Small", "hf_depth",
              "depth-anything/Depth-Anything-V2-Small-hf", "relative", _REL, on_device=True),
    ModelSpec("dav2_base", "Depth-Anything-V2", "Base", "hf_depth",
              "depth-anything/Depth-Anything-V2-Base-hf", "relative", _REL),
    ModelSpec("dav2_large", "Depth-Anything-V2", "Large", "hf_depth",
              "depth-anything/Depth-Anything-V2-Large-hf", "relative", _REL),
    # ── Depth-Anything V2 (metric, indoor/outdoor) ──────────────────────────
    ModelSpec("dav2_metric_indoor", "Depth-Anything-V2-Metric", "Large/Hypersim", "hf_depth",
              "depth-anything/Depth-Anything-V2-Metric-Hypersim-Large-hf", "metric", _MET,
              note="indoor metric"),
    ModelSpec("dav2_metric_outdoor", "Depth-Anything-V2-Metric", "Large/VKITTI", "hf_depth",
              "depth-anything/Depth-Anything-V2-Metric-VKITTI-Large-hf", "metric", _MET,
              note="outdoor metric"),
    # ── Depth-Anything V3 (best-effort; id may change) ──────────────────────
    ModelSpec("dav3_large", "Depth-Anything-V3", "Large", "hf_depth",
              "depth-anything/Depth-Anything-V3-Large-hf", "relative", _REL,
              experimental=True, note="experimental id; override with --dav3-id"),
    # ── MiDaS / DPT (relative inverse-depth) ────────────────────────────────
    ModelSpec("dpt_large", "MiDaS/DPT", "Large", "hf_depth",
              "Intel/dpt-large", "relative", _REL),
    ModelSpec("dpt_hybrid", "MiDaS/DPT", "Hybrid", "hf_depth",
              "Intel/dpt-hybrid-midas", "relative", _REL),
    ModelSpec("dpt_swin2_tiny", "MiDaS/DPT", "SwinV2-Tiny", "hf_depth",
              "Intel/dpt-swinv2-tiny-256", "relative", _REL, on_device=True,
              note="mobile-friendly"),
    # ── ZoeDepth (metric) ───────────────────────────────────────────────────
    ModelSpec("zoedepth_nyu", "ZoeDepth", "NYU", "hf_depth",
              "Intel/zoedepth-nyu", "metric", _MET, note="indoor metric"),
    ModelSpec("zoedepth_nk", "ZoeDepth", "NYU+KITTI", "hf_depth",
              "Intel/zoedepth-nyu-kitti", "metric", _MET),
    # ── Apple Depth Pro (metric; needs transformers>=4.43) ──────────────────
    ModelSpec("depth_pro", "DepthPro", "ViT", "hf_depth",
              "apple/DepthPro-hf", "metric", _MET, experimental=True,
              note="needs recent transformers"),
    # ── Metric3D v2 (metric; torch.hub) ─────────────────────────────────────
    ModelSpec("metric3d_vit_small", "Metric3Dv2", "ViT-S", "metric3d_hub",
              "metric3d_vit_small", "metric", _MET, on_device=True, experimental=True),
    ModelSpec("metric3d_vit_large", "Metric3Dv2", "ViT-L", "metric3d_hub",
              "metric3d_vit_large", "metric", _MET, experimental=True),
]

_BY_KEY = {s.key: s for s in REGISTRY}


def get_specs(keys: Optional[List[str]] = None) -> List[ModelSpec]:
    if not keys:
        return list(REGISTRY)
    out: List[ModelSpec] = []
    for k in keys:
        if k in _BY_KEY:
            out.append(_BY_KEY[k])
        else:  # also accept a family substring (e.g. "dav2")
            hit = [s for s in REGISTRY if k.lower() in s.key.lower()
                   or k.lower() in s.family.lower()]
            if not hit:
                raise SystemExit(f"[zoo] unknown model key {k!r}. Known: {list(_BY_KEY)}")
            out.extend(hit)
    # de-dup preserving order
    seen, uniq = set(), []
    for s in out:
        if s.key not in seen:
            seen.add(s.key); uniq.append(s)
    return uniq


# --------------------------------------------------------------------------- #
# Availability + download
# --------------------------------------------------------------------------- #
def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def _repo_dir_name(repo_id: str) -> str:
    """HF hub-cache folder for a repo id (``org/name`` → ``models--org--name``)."""
    return "models--" + repo_id.replace("/", "--")


def _disp(path: str) -> str:
    """Repo-relative path for display when under the repo, else the abs path."""
    try:
        rel = os.path.relpath(path, _REPO)
        return rel if not rel.startswith("..") else path
    except Exception:
        return path


def _hf_cached_in(repo_id: str, cache_dir: str) -> bool:
    """True if ``repo_id`` is materialised in the given hub cache dir."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False
    try:
        return isinstance(
            try_to_load_from_cache(repo_id, "config.json", cache_dir=cache_dir), str)
    except Exception:
        return False


def _hf_locate(repo_id: str) -> Optional[str]:
    """Hub cache dir holding ``repo_id`` — prefer the repo-local cache, then the
    (possibly ephemeral) default. ``None`` if not downloaded anywhere."""
    if _hf_cached_in(repo_id, _HF_HUB):
        return _HF_HUB
    if _DEFAULT_HF_HUB != _HF_HUB and _hf_cached_in(repo_id, _DEFAULT_HF_HUB):
        return _DEFAULT_HF_HUB
    return None


def _hf_status(repo_id: str) -> Tuple[str, str]:
    """Shared (state, detail) for HF-cache models (hf_depth + unik3d)."""
    loc = _hf_locate(repo_id)
    if loc == _HF_HUB:
        return "ready", _disp(os.path.join(_HF_HUB, _repo_dir_name(repo_id)))
    if loc is not None:  # in the ephemeral default cache only
        return "download", (f"in {loc} only — --download copies it into "
                            f"{_disp(_HF_HUB)} (persistent)")
    return "download", repo_id


def _copy_repo_cache(repo_id: str, src_hub: str, dst_hub: str) -> str:
    """Copy a model's ``models--org--name`` tree between hub caches. HF stores each
    blob once and points ``snapshots/<rev>/<file>`` at it via a *relative* symlink
    (``../../blobs/<sha>``); copying with ``symlinks=True`` preserves that layout so
    the destination is self-contained (real blobs + relative links within the tree)."""
    import shutil
    name = _repo_dir_name(repo_id)
    src, dst = os.path.join(src_hub, name), os.path.join(dst_hub, name)
    os.makedirs(dst_hub, exist_ok=True)
    shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True)
    return dst


def _dac_files(spec: ModelSpec) -> Tuple[str, str]:
    return (os.path.join(_CKPT, f"{spec.ref}.json"),
            os.path.join(_CKPT, f"{spec.ref}.pt"))


#: DA3 variant -> released HuggingFace repo (mirrors ``raytun3r.backbones``).
DA3_REPOS = {"small": "depth-anything/DA3-SMALL", "base": "depth-anything/DA3-BASE",
             "large": "depth-anything/DA3-LARGE", "giant": "depth-anything/DA3-GIANT"}


def status(spec: ModelSpec) -> Tuple[str, str]:
    """Return (state, detail). state ∈ {ready, download, unavailable}.

    ``ready``       -- can run now (deps present + weights local/cached).
    ``download``    -- deps present but weights not yet fetched (``--download`` will get them).
    ``unavailable`` -- a dependency or repo is missing (``detail`` says how to fix).
    """
    if spec.kind == "dac":
        cfg, wts = _dac_files(spec)
        if not os.path.isdir(os.path.join(_REPO, "third_party", "depth_any_camera")):
            return "unavailable", "clone DAC: bash finetune/eval/baselines/setup_baselines.sh dac"
        if os.path.isfile(cfg) and os.path.isfile(wts):
            return "ready", os.path.relpath(wts, _REPO)
        return "download", f"HF yuliangguo/depth-any-camera → {os.path.relpath(wts, _REPO)}"

    if spec.kind == "unik3d":
        if not os.path.isdir(os.path.join(_REPO, "third_party", "UniK3D")):
            return "unavailable", "clone UniK3D: bash finetune/eval/baselines/setup_baselines.sh unik3d"
        return _hf_status(f"lpiccinelli/unik3d-{spec.ref}")

    if spec.kind == "hf_depth":
        if not _have("transformers"):
            return "unavailable", "pip install -U transformers"
        return _hf_status(spec.ref)

    if spec.kind == "metric3d_hub":
        if not _have("torch"):
            return "unavailable", "pip install torch"
        return "download", f"torch.hub yvanyin/metric3d:{spec.ref} (best-effort)"

    if spec.kind == "vggt":
        if not os.path.isdir(os.path.join(_REPO, "VGGT-360-fisheye", "vggt_visfeat")):
            return "unavailable", "the vendored vggt_visfeat package is missing"
        return _hf_status(spec.ref)

    if spec.kind == "vggt_omega":
        # Gated weights: there is no download path, only a pointer.
        if os.path.isfile(spec.ref):
            return "ready", _disp(spec.ref)
        return "unavailable", (
            f"checkpoint not found at {_disp(spec.ref)} — request access at "
            f"https://huggingface.co/facebook/VGGT-Omega, then set $VGGT_OMEGA_CKPT")

    if spec.kind == "da3":
        if not _have("depth_anything_3"):
            return "unavailable", ("pip install --no-deps depth-anything-3 && "
                                   "pip install omegaconf addict einops")
        return _hf_status(DA3_REPOS[spec.ref])

    return "unavailable", f"unknown kind {spec.kind!r}"


def download(spec: ModelSpec) -> Tuple[bool, str]:
    """Fetch weights for a ``download``-state spec. Returns (ok, message).

    HuggingFace models land in ``~/.cache/huggingface/hub/`` (the default HF
    cache); they are NOT re-downloaded on subsequent runs because
    :func:`_hf_cached` finds them there via ``try_to_load_from_cache``.

    DAC weights go to ``<repo>/checkpoints/``.
    """
    st, _ = status(spec)
    if st == "ready":
        return True, "already present"
    if st == "unavailable":
        return False, "dependency/repo missing — see `status`"
    try:
        if spec.kind == "dac":
            from huggingface_hub import hf_hub_download
            os.makedirs(_CKPT, exist_ok=True)
            for fn in (f"{spec.ref}.json", f"{spec.ref}.pt"):
                hf_hub_download(repo_id="yuliangguo/depth-any-camera", filename=fn,
                                repo_type="model", local_dir=_CKPT)
            return True, f"downloaded DAC config+weights → {_CKPT}"
        if spec.kind in ("unik3d", "hf_depth", "vggt", "da3"):
            repo = {"unik3d": f"lpiccinelli/unik3d-{spec.ref}",
                    "da3": DA3_REPOS.get(spec.ref, spec.ref)}.get(spec.kind, spec.ref)
            # 1) already repo-local — nothing to do (survives image reloads)
            if _hf_cached_in(repo, _HF_HUB):
                return True, f"already in {_disp(_HF_HUB)}"
            # 2) present only in the ephemeral default cache — copy, don't re-download
            if _DEFAULT_HF_HUB != _HF_HUB and _hf_cached_in(repo, _DEFAULT_HF_HUB):
                dst = _copy_repo_cache(repo, _DEFAULT_HF_HUB, _HF_HUB)
                return True, f"copied {repo}: {_DEFAULT_HF_HUB} → {_disp(dst)}"
            # 3) nowhere yet — download straight into the repo-local cache
            from huggingface_hub import snapshot_download
            path = snapshot_download(repo_id=repo, repo_type="model", cache_dir=_HF_HUB)
            return True, f"downloaded {repo} → {_disp(path)}"
        if spec.kind == "metric3d_hub":
            import torch
            torch.hub.load("yvanyin/metric3d", spec.ref, pretrain=True)
            return True, "cached via torch.hub"
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        auth_hint = ""
        if "401" in msg or "Invalid username" in msg or "GatedRepo" in type(exc).__name__:
            repo_url = (f"https://huggingface.co/{spec.ref}"
                        if spec.kind == "hf_depth" else "https://huggingface.co")
            auth_hint = (f"\n    → gated repo — fix:\n"
                         f"      1. huggingface-cli login   (or set HF_TOKEN env var)\n"
                         f"      2. accept model card at {repo_url}")
        return False, f"{type(exc).__name__}: {exc}{auth_hint}"
    return False, "nothing to do"


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #
class Adapter:
    """Base adapter. Subclasses set ``self.torch_modules`` in ``load`` and
    implement ``predict_frame`` (and ``predict_erp`` for DAC)."""

    def __init__(self, spec: ModelSpec, **kw):
        self.spec = spec
        self.kw = kw
        self.device = None
        self.torch_modules: List = []

    # -- lifecycle ---------------------------------------------------------- #
    def load(self, device) -> "Adapter":
        raise NotImplementedError

    # -- prediction --------------------------------------------------------- #
    def predict_frame(self, rgb01: np.ndarray, cam: AriaFisheye, frame: str) -> np.ndarray:
        raise NotImplementedError

    def predict_erp(self, rgb01: np.ndarray, cam: AriaFisheye) -> dict:
        raise NotImplementedError(f"{self.spec.key} has no ERP-native path")

    # -- profiling ---------------------------------------------------------- #
    def num_params(self) -> int:
        return int(sum(p.numel() for m in self.torch_modules for p in m.parameters()))

    def weight_mb(self) -> float:
        return self.num_params() * 4 / 1e6  # fp32 estimate


def build_adapter(spec: ModelSpec, **kw) -> Adapter:
    return {
        "dac": DACAdapter,
        "unik3d": UniK3DAdapter,
        "hf_depth": HFDepthAdapter,
        "metric3d_hub": Metric3DAdapter,
        "vggt": BackboneAdapter,
        "vggt_omega": BackboneAdapter,
        "da3": BackboneAdapter,
    }[spec.kind](spec, **kw)


def patch_align(H: int, W: int, patch: int) -> Tuple[int, int]:
    """Nearest ``patch``-multiple size, at least one patch on each side."""
    return (max(patch, int(round(H / patch)) * patch),
            max(patch, int(round(W / patch)) * patch))


class BackboneAdapter(Adapter):
    """VGGT / VGGT-Omega / Depth-Anything-3 behind ``predict_frame``.

    All three are *multi-view* 3D foundation models, and all three are run here
    with **one image per forward pass**. That is deliberate, not a limitation:
    the benchmark compares them against monocular depth nets, and a multi-view
    run would let cross-view attention fuse evidence the monocular rows never
    see — the comparison would then be about view count, not about the models.
    ``vggt_visfeat``'s own preprocessing is bypassed for the same reason the
    ``hf_depth`` rows bypass theirs: the caller renders each view at the model's
    native token grid, so nothing is resampled between the view construction and
    the network.

    The wrapping lives in :mod:`raytun3r.backbones`, which already owns the
    loaders, the CPU fallbacks and — the reason to reuse it rather than reload
    the models here — the depth-convention discipline: an uninstalled backbone
    returns its head's native **planar z**, tagged, which is exactly the
    quantity ``predict_frame`` is contracted to return.
    """

    #: spec.kind -> raytun3r.backbones.BACKBONES key (they coincide today, but
    #: the zoo's kinds are a public CLI surface and that module's are not).
    _BACKBONE = {"vggt": "vggt", "vggt_omega": "vggt_omega", "da3": "da3"}

    def load(self, device):
        import torch
        if _REPO not in sys.path:
            sys.path.insert(0, _REPO)
        from raytun3r.backbones import build_backbone

        kind = self.spec.kind
        kw = {}
        if kind == "vggt":
            weights = None                       # -> VGGT.from_pretrained(spec.ref)
        elif kind == "vggt_omega":
            weights = self.kw.get("checkpoint") or self.spec.ref
            if not os.path.isfile(weights):
                raise FileNotFoundError(
                    f"VGGT-Omega checkpoint not found: {weights} (set "
                    f"$VGGT_OMEGA_CKPT or pass checkpoint=...)")
        else:                                    # da3
            weights, kw["variant"] = "pretrained", self.spec.ref

        self.backbone = build_backbone(self._BACKBONE[kind], weights=weights,
                                       device=device, **kw)
        self.backbone.eval()
        self.device = device
        self.torch_modules = [self.backbone]
        return self

    def predict_frame(self, rgb01: np.ndarray, cam, frame: str) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        H, W = rgb01.shape[:2]
        x = torch.from_numpy(
            np.ascontiguousarray(np.clip(rgb01, 0, 1).transpose(2, 0, 1))
        )[None].float().to(self.device)
        h, w = patch_align(H, W, self.backbone.patch_size)
        if (h, w) != (H, W):
            x = F.interpolate(x, (h, w), mode="bilinear", align_corners=False)
        with torch.no_grad():
            depth = self.backbone(x).depth[0]            # (h, w) planar z
        if depth.shape != (H, W):
            depth = F.interpolate(depth[None, None].float(), (H, W),
                                  mode="bilinear", align_corners=False)[0, 0]
        return depth.float().clamp_min(1e-3).cpu().numpy()


class HFDepthAdapter(Adapter):
    """transformers AutoModelForDepthEstimation (Depth-Anything, MiDaS/DPT,
    ZoeDepth, Depth Pro, …). Runs on whatever RGB frame the runner feeds it."""

    def load(self, device):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        ref = self.spec.ref
        self.model = AutoModelForDepthEstimation.from_pretrained(
            ref, cache_dir=_HF_HUB).to(device).eval()
        try:
            self.proc = AutoImageProcessor.from_pretrained(ref, cache_dir=_HF_HUB)
        except Exception:
            self.proc = None
        self.device = device
        self.torch_modules = [self.model]
        return self

    def _pixel_values(self, rgb01: np.ndarray):
        import torch
        import torch.nn.functional as F
        if self.proc is not None:
            u8 = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
            return self.proc(images=u8, return_tensors="pt")["pixel_values"].to(self.device)
        # fallback: imagenet-normalised, sides rounded to /14
        x = torch.from_numpy(rgb01.transpose(2, 0, 1)).float()[None].to(self.device)
        mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        H, W = rgb01.shape[:2]
        h14, w14 = (max(14, round(H / 14) * 14), max(14, round(W / 14) * 14))
        if (h14, w14) != (H, W):
            x = F.interpolate(x, (h14, w14), mode="bilinear", align_corners=False)
        return x

    def predict_frame(self, rgb01, cam, frame):
        import torch
        import torch.nn.functional as F
        H, W = rgb01.shape[:2]
        px = self._pixel_values(rgb01)
        with torch.no_grad():
            out = self.model(pixel_values=px).predicted_depth
        if out.dim() == 2:
            out = out[None, None]
        elif out.dim() == 3:
            out = out[:, None]
        out = F.interpolate(out, (H, W), mode="bilinear", align_corners=False)[0, 0]
        if self.spec.output_type == "relative":
            disp = out.clamp_min(0.0)
            scale = disp.mean().clamp_min(1e-6)
            depth = 1.0 / (disp / scale).clamp_min(1e-2)
        else:
            depth = out.clamp_min(1e-3)
        return depth.float().cpu().numpy()


class UniK3DAdapter(Adapter):
    def load(self, device):
        from .predict_unik3d import UniK3DPredictor
        self.pred = UniK3DPredictor(backbone=self.spec.ref, device=device,
                                    use_camera=self.kw.get("use_camera", True))
        self.device = device
        self.torch_modules = [self.pred.model]
        return self

    def predict_frame(self, rgb01, cam, frame):
        u8 = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
        out = self.pred.predict(u8, cam)       # planar z in the fisheye frame
        return out["depth"].astype(np.float32)


class DACAdapter(Adapter):
    def load(self, device):
        from .predict_dac import DACPredictor
        cfg, wts = _dac_files(self.spec)
        self.pred = DACPredictor(cfg, wts, device=device)
        self.device = device
        self.torch_modules = [self.pred.model]
        return self

    def predict_erp(self, rgb01, cam):
        """DAC's native output: ERP euclidean-range depth + in-FOV mask."""
        out = self.pred.predict(rgb01, cam)
        return {"depth": out["depth"], "active": out["active"]}

    def predict_frame(self, rgb01, cam, frame):
        # DAC is ERP-native; the runner scores it via predict_erp. A fisheye-frame
        # planar-z map would need the ERP→fisheye inverse remap; not on the hot path.
        raise NotImplementedError("DAC is ERP-native; use predict_erp")


class Metric3DAdapter(Adapter):
    """Metric3D v2 via torch.hub (best-effort; metric, needs intrinsics)."""

    def load(self, device):
        import torch
        self.model = torch.hub.load("yvanyin/metric3d", self.spec.ref, pretrain=True)
        self.model = self.model.to(device).eval()
        self.device = device
        self.torch_modules = [self.model]
        return self

    def predict_frame(self, rgb01, cam, frame):
        import torch
        u8 = (np.clip(rgb01, 0, 1) * 255).astype(np.float32)
        x = torch.from_numpy(u8.transpose(2, 0, 1))[None].to(self.device)
        mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        with torch.no_grad():
            pred, *_ = self.model.inference({"input": x})
        d = pred.squeeze().float().cpu().numpy()
        if d.shape != rgb01.shape[:2]:
            import cv2
            d = cv2.resize(d, (rgb01.shape[1], rgb01.shape[0]), interpolation=cv2.INTER_NEAREST)
        return np.clip(d, 1e-3, None).astype(np.float32)
