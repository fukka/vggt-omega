# Copyright (c) 2026.
"""Evaluate UniK3D and Depth-Any-Camera on raw (non-rectified) Aria fisheye.

Two any-camera metric depth baselines, tested the way they are meant to be used —
straight on the **non-rectified fisheye** Aria frame (no fisheye→pinhole rectify):

  * ADT (dense GT)  -> **quantitative** depth metrics.
  * EgoExo4D        -> **qualitative** depth visualisations (no metric GT).

Data loading mirrors ``finetune/eval/run_eval.py`` / ``adt_depth.py``: the ADT
path reuses ``ADTWindowDataset`` with ``seq_len=1`` and ``rectify=False`` (real
``videos_rgb`` fisheye + ``depth_npy`` GT, 270° CCW rotation, uint16 mm → metres).

Because the two models predict in different native domains (UniK3D in the image
plane, DAC in equirectangular space), ADT metrics are reported in two domains:

  1. **Fisheye-frame (planar depth)** — UniK3D's native output vs GT directly
     (same convention as the VGGT/DAv2 ADT eval, so numbers are comparable).
  2. **ERP (euclidean range), head-to-head** — both models scored on the identical
     ERP grid DAC predicts on (UniK3D's depth is warped there too). This is the
     fair common-domain comparison and depends only on DAC's validated warp.

Both domains report alignment modes ``none`` (metric) and ``scale_shift``.

Examples
--------
ADT quantitative (both models) on the GPU box::

    python -m finetune.eval.baselines.run_baselines --mode adt \\
        --adt-root /group-volume/Fengjia/data/projectaria_tools_adt_data_clean \\
        --dac-config checkpoints/dac_swinl_indoor.json \\
        --dac-weights checkpoints/dac_swinl_indoor.pt

EgoExo4D qualitative::

    python -m finetune.eval.baselines.run_baselines --mode egoexo \\
        --egoexo-root EgoX/example/egoexo4D \\
        --dac-config checkpoints/dac_swinl_indoor.json \\
        --dac-weights checkpoints/dac_swinl_indoor.pt
"""
from __future__ import annotations

import sys as _sys, os as _os
if not __package__:
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__))))))
    __package__ = "finetune.eval.baselines"

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch

from ..adt_depth import ADTWindowDataset
from ..metrics import aggregate_metrics, align_depth, depth_metrics
from ..run_eval import _find_seq_dirs
from .aria_fisheye import (aria_centered, aria_intrinsics, aria_ray_z,
                           planar_to_range, range_to_planar)
from .erp_utils import fisheye_to_erp_fwd
from .io_utils import save_sample

_TABLE_METRICS = ["AbsRel", "SqRel", "RMSE", "delta1", "delta2", "scale_ratio", "n_frames"]
_PCT = {"delta1", "delta2", "delta3"}
_ALIGN_MODES = ("none", "scale_shift")
# ERP head-to-head grid defaults (scannetpp fisheye preset); overridden by DAC cfg.
_ERP_CANO, _ERP_FWD, _ERP_WFOV = 1400, (500, 750), 180.0


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _to_uint8_hwc(img_chw01: torch.Tensor) -> np.ndarray:
    return (img_chw01.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def _build_out_tag(a) -> str:
    """Descriptive output path derived from active models and their key settings.

    Examples::

        eval_out/unik3d-vitl_dac-swinl_indoor   # both
        eval_out/unik3d-vitl                     # unik3d only
        eval_out/dac-swinl_indoor                # dac only
    """
    want = {m.strip() for m in a.models.split(",") if m.strip()}
    parts: list[str] = []
    if "unik3d" in want:
        tag = f"unik3d-{a.unik3d_backbone}"
        if a.no_unik3d_camera:
            tag += "-nocam"
        parts.append(tag)
    if "dac" in want and a.dac_config:
        stem = os.path.splitext(os.path.basename(a.dac_config))[0]  # dac_swinl_indoor
        parts.append(stem)
    return os.path.join("eval_out", "_".join(parts) if parts else "baselines")


def _fmt(v, key) -> str:
    if not isinstance(v, (int, float)) or v != v:
        return "  N/A  "
    if key in _PCT:
        return f"{v*100:5.1f}%"
    if key == "n_frames":
        return f"{int(v):6d}"
    return f"{v:7.4f}"


def _print_table(title: str, results: Dict[str, dict], order: List[str]) -> str:
    lines = [f"\n{'='*84}", f"  {title}", f"{'='*84}"]
    for mode in _ALIGN_MODES:
        lines.append(f"\n  alignment = {mode}")
        header = f"  {'Model':24s}" + "".join(f"  {k:>10s}" for k in _TABLE_METRICS)
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for key in order:
            if key not in results or mode not in results[key]:
                continue
            md = results[key][mode]
            row = f"  {key:24s}" + "".join(f"  {_fmt(md.get(k, float('nan')), k):>10s}"
                                           for k in _TABLE_METRICS)
            lines.append(row)
    text = "\n".join(lines)
    print(text)
    return text


# --------------------------------------------------------------------------- #
# ADT quantitative
# --------------------------------------------------------------------------- #

def run_adt(args, unik, dac) -> None:
    seq_dirs = _find_seq_dirs(args.adt_root, args.adt_rgb_subdir, "depth_npy")
    if not seq_dirs:
        raise SystemExit(
            f"[baselines] no ADT sequences with {args.adt_rgb_subdir}/ + depth_npy/ under "
            f"{args.adt_root!r}")
    print(f"[baselines] ADT: {len(seq_dirs)} sequence(s); non-rectified fisheye "
          f"(rgb={args.adt_rgb_subdir}), res={args.image_resolution}, "
          f"<= {args.adt_max_frames} frames/seq")
    # The ERP head-to-head domain needs DAC's geometry even if the DAC *model*
    # isn't selected — ensure the repo is importable up front (honours --dac-root).
    from .erp_utils import add_dac_to_path
    add_dac_to_path(args.dac_root)

    # Single-frame, non-rectified fisheye windows (real sensor RGB vs dense GT).
    ds = ADTWindowDataset(
        seq_dirs, seq_len=1, window_stride=1, image_resolution=args.image_resolution,
        depth_scale=0.001, depth_max_m=args.adt_depth_max,
        max_frames=(None if args.adt_max_frames < 0 else args.adt_max_frames),
        rgb_subdir=args.adt_rgb_subdir, depth_subdir="depth_npy", rectify=False,
    )

    erp_cano = dac.cano if dac is not None else _ERP_CANO
    erp_fwd = dac.fwd_sz if dac is not None else _ERP_FWD
    erp_wfov = dac.crop_wFoV if dac is not None else _ERP_WFOV
    # ERP depth is euclidean range = planar_z / cos(theta), so it exceeds the
    # planar cap toward the fisheye edge (cos→~0.46). Use a looser cap there; the
    # warped valid mask (from the planar z<=adt_depth_max GT) still governs which
    # pixels are scored.
    erp_depth_max = args.adt_depth_max / 0.4

    # accumulators: domain -> variant -> mode -> [per-frame metric dicts]
    fish: Dict[str, Dict[str, List[dict]]] = {}
    erp: Dict[str, Dict[str, List[dict]]] = {}
    def _slot(store, var):
        store.setdefault(var, {m: [] for m in _ALIGN_MODES})

    qdir = os.path.join(args.out, "qual")
    n_saved = 0
    for idx in range(len(ds)):
        item = ds[idx]
        img_chw = item["images"][0]
        gt = item["depths"][0].numpy().astype(np.float32)
        mask = item["valid_masks"][0].numpy().astype(bool)
        if mask.sum() < 10:
            continue
        H, W = gt.shape
        rgb_u8 = _to_uint8_hwc(img_chw)
        img01 = rgb_u8.astype(np.float32) / 255.0
        cam = aria_intrinsics(H, W, rotated=True)
        cos_map = aria_ray_z(cam)
        # GT in both conventions.
        if args.gt_euclidean:
            gt_range = gt; gt_planar = range_to_planar(gt, cos_map)
        else:
            gt_planar = gt; gt_range = planar_to_range(gt, cos_map)

        # GT warped once to the ERP head-to-head grid (euclidean range).
        gt_warp = fisheye_to_erp_fwd(img01, gt_range, mask.astype(np.float32),
                                     cam.opencv_fisheye_params(), erp_cano, erp_fwd, erp_wfov)
        gt_erp = gt_warp["depth"]
        gt_erp_mask = (gt_warp["valid"] > 0.5) & (gt_warp["active"] > 0.5)

        # ---- UniK3D ----
        ou = None
        if unik is not None:
            _slot(fish, "unik3d"); _slot(erp, "unik3d")
            ou = unik.predict(rgb_u8, cam)                  # dict: depth/points/rgb
            uz = ou["depth"]                                # planar z, [H,W]
            for m in _ALIGN_MODES:
                fish["unik3d"][m].append(
                    depth_metrics(align_depth(uz, gt_planar, mask, m), gt_planar, mask,
                                  max_depth=args.adt_depth_max))
            # warp UniK3D (as range) onto the same ERP grid as GT.
            ur = planar_to_range(uz, cos_map)
            uw = fisheye_to_erp_fwd(img01, ur, mask.astype(np.float32),
                                    cam.opencv_fisheye_params(), erp_cano, erp_fwd, erp_wfov)
            um = gt_erp_mask & (uw["active"] > 0.5)
            for m in _ALIGN_MODES:
                erp["unik3d"][m].append(
                    depth_metrics(align_depth(uw["depth"], gt_erp, um, m), gt_erp, um,
                                  max_depth=erp_depth_max))

        # ---- DAC ----
        if dac is not None:
            _slot(erp, "dac")
            out = dac.predict(img01, cam)                   # ERP range, fwd grid
            dm = gt_erp_mask & (out["active"] > 0.5)
            for m in _ALIGN_MODES:
                erp["dac"][m].append(
                    depth_metrics(align_depth(out["depth"], gt_erp, dm, m), gt_erp, dm,
                                  max_depth=erp_depth_max))

        # ---- qualitative: save input + depth(.npy/.png) + point cloud(.ply) ----
        if n_saved < args.n_qual:
            if ou is not None:
                save_sample(os.path.join(qdir, "unik3d"), f"{n_saved:03d}",
                            ou["rgb"], ou["depth"], ou["points"], ou["rgb"], mask)
            if dac is not None:
                save_sample(os.path.join(qdir, "dac"), f"{n_saved:03d}",
                            out["image_u8"], out["depth"], out["points"],
                            out["image_u8"], out["active"] > 0.5)
            n_saved += 1

        if (idx + 1) % 25 == 0:
            print(f"[baselines]   ADT {idx+1}/{len(ds)} frames")

    # aggregate + report
    fish_res = {v: {m: aggregate_metrics(fish[v][m]) for m in _ALIGN_MODES} for v in fish}
    erp_res = {v: {m: aggregate_metrics(erp[v][m]) for m in _ALIGN_MODES} for v in erp}
    t1 = _print_table("ADT (dense GT) — fisheye-frame, planar depth  [UniK3D native]",
                      fish_res, ["unik3d"])
    t2 = _print_table("ADT (dense GT) — ERP, euclidean range  [head-to-head]",
                      erp_res, ["unik3d", "dac"])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "adt_results.json"), "w") as f:
        json.dump({"fisheye_planar": fish_res, "erp_range": erp_res}, f, indent=2)
    with open(os.path.join(args.out, "adt_summary.txt"), "w") as f:
        f.write(t1 + "\n" + t2 + "\n")
    print(f"\n[baselines] ADT results → {args.out}/  (qual panels in {qdir}/)")


# --------------------------------------------------------------------------- #
# EgoExo4D qualitative
# --------------------------------------------------------------------------- #

def run_egoexo(args, unik, dac) -> None:
    from .egoexo_data import find_egoexo_clips, sample_frames

    clips = find_egoexo_clips(args.egoexo_root, stream=args.egoexo_stream)
    print(f"[baselines] EgoExo4D: {len(clips)} clip(s), {args.n_frames} frame(s) each "
          f"({args.egoexo_stream}, raw fisheye, qualitative)")
    out_dir = os.path.join(args.out, "egoexo")
    for clip, vp in clips:
        for fi, rgb_u8 in sample_frames(vp, n=args.n_frames):
            H, W = rgb_u8.shape[:2]
            cam = aria_centered(H, W)                       # upright, centered fisheye
            img01 = rgb_u8.astype(np.float32) / 255.0
            tag = f"{clip}_f{fi:04d}"
            if unik is not None:
                ou = unik.predict(rgb_u8, cam)              # input + depth + point cloud
                save_sample(os.path.join(out_dir, "unik3d"), tag,
                            ou["rgb"], ou["depth"], ou["points"], ou["rgb"])
            if dac is not None:
                out = dac.predict(img01, cam)
                save_sample(os.path.join(out_dir, "dac"), tag,
                            out["image_u8"], out["depth"], out["points"],
                            out["image_u8"], out["active"] > 0.5)
        print(f"[baselines]   {clip}: done")
    print(f"\n[baselines] EgoExo4D outputs (input/depth/pcd) → {out_dir}/")


# --------------------------------------------------------------------------- #
# Official-example sanity check — run each repo's own demo asset end-to-end.
# --------------------------------------------------------------------------- #

def run_official(args, unik, dac) -> None:
    """Run each repo's fisheye demo asset and save our prediction alongside the
    repo's pre-computed reference for direct comparison.

    Camera-type audit of all shipped demo assets:

    UniK3D (assets/demo/):
      kitti360.png   + kitti360.json  → MEI       **fisheye** ← default
      equirectangular.jpg + .json     → Spherical  360°
      dl3dv.png      + dl3dv.json     → OPENCV     perspective
      scannet.jpg    + scannet.json   → Fisheye624 k≈0 (ScanNet is perspective;
                                        near-zero k means no real distortion)
      bears/berzirk/naruto/venice/poorthings/luke → no camera json

    DAC (demo/input/):
      scannetpp_sample.json  → OPENCV_FISHEYE  **fisheye** ← default
      kitti360_sample.json   → MEI              fisheye (GT + official output available)
      matterport3d_sample.json → ERP            360°
      nyu_sample.json / kitti_sample.json → PINHOLE  perspective

    UniK3D reference: only scannet.npy exists (ships with the repo); it is a
    [H,W,3] point map where z = planar depth. For kitti360 there is no pre-computed
    reference — the model output stands on its own.
    DAC reference: demo/input/<name>_depth.png (GT, int32 mm) and
                   demo/output/<name>_output_subplot.jpg (official vis).
    """
    import shutil
    from .io_utils import save_depth_png

    out_dir = os.path.join(args.out, "official")
    if unik is not None:
        # kitti360 is the genuine fisheye example UniK3D ships (MEI camera, 933×933).
        # scannet.json is labelled Fisheye624 but k≈0, i.e. effectively perspective.
        img = args.unik3d_official_image or os.path.join(unik.root, "assets/demo/kitti360.png")
        camj = args.unik3d_official_camera or os.path.join(unik.root, "assets/demo/kitti360.json")
        stem = os.path.splitext(os.path.basename(img))[0]
        print(f"[baselines] UniK3D official: {os.path.relpath(img, unik.root)} (fisheye MEI)")
        o = unik.predict_official(img, camj)
        udir = os.path.join(out_dir, "unik3d")
        save_sample(udir, stem, o["rgb"], o["depth"], o["points"], o["rgb"])
        # Reference: repo ships scannet.npy (z = depth); no npy for kitti360 — skip if absent.
        ref_npy = os.path.join(os.path.dirname(img), f"{stem}.npy")
        if os.path.isfile(ref_npy):
            ref_pts = np.load(ref_npy)                      # [H,W,3] float32
            ref_depth = ref_pts[:, :, 2]                    # z = planar depth (metres)
            ref_mask = ref_depth > 0
            np.save(os.path.join(udir, f"{stem}_ref_depth.npy"), ref_depth.astype(np.float32))
            save_depth_png(os.path.join(udir, f"{stem}_ref_depth.png"), ref_depth, ref_mask)
            print(f"[baselines]   UniK3D ref depth saved (from {stem}.npy, z channel)")
        else:
            print(f"[baselines]   no reference npy for {stem} — prediction only")

    if dac is not None:
        # scannetpp uses OPENCV_FISHEYE (same model family as Aria KB4): genuine fisheye.
        # kitti360 uses MEI fisheye and also ships GT depth + official output if preferred.
        sample = args.dac_official_sample or os.path.join(dac.root, "demo/input/scannetpp_sample.json")
        import json as _json
        with open(sample) as _f:
            _s = _json.load(_f)
        stem = _s.get("dataset_name", os.path.splitext(os.path.basename(sample))[0].replace("_sample", ""))
        cam_model = _s.get("cam_params", {}).get("camera_model", "?")
        print(f"[baselines] DAC official: {os.path.relpath(sample, dac.root)} ({cam_model})")
        o = dac.predict_official(None, sample)
        ddir = os.path.join(out_dir, "dac")
        save_sample(ddir, stem,
                    o["image_u8"], o["depth"], o["points"], o["image_u8"], o["active"] > 0.5)
        # GT depth: repo ships <stem>_depth.png (int32, mm) → metres
        gt_png = os.path.join(dac.root, f"demo/input/{stem}_depth.png")
        if os.path.isfile(gt_png):
            from PIL import Image as _Image
            gt_m = np.array(_Image.open(gt_png)).astype(np.float32) / 1000.0
            os.makedirs(ddir, exist_ok=True)
            np.save(os.path.join(ddir, f"{stem}_gt_depth.npy"), gt_m)
            save_depth_png(os.path.join(ddir, f"{stem}_gt_depth.png"), gt_m, gt_m > 0)
            print(f"[baselines]   DAC GT depth saved (from {stem}_depth.png)")
        # Official output visualization
        off_jpg = os.path.join(dac.root, f"demo/output/{stem}_output_subplot.jpg")
        if os.path.isfile(off_jpg):
            os.makedirs(ddir, exist_ok=True)
            shutil.copy(off_jpg, os.path.join(ddir, f"{stem}_official_output.jpg"))
            print(f"[baselines]   DAC official output viz copied")

    print(f"\n[baselines] official sanity-check outputs → {out_dir}/")
    print(f"[baselines]   unik3d/: {{stem}}_{{input,depth,pcd}}.* (+ ref_depth if npy available)")
    print(f"[baselines]   dac/:    {{stem}}_{{input,depth,pcd,gt_depth,official_output}}.*")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(
        description="UniK3D + DAC on non-rectified Aria fisheye (ADT quant, EgoExo4D qual)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=["adt", "egoexo", "official", "both", "all"],
                   default="both",
                   help="adt/egoexo eval; 'official' = each repo's own demo as a "
                        "sanity check; 'both' = adt+egoexo; 'all' = +official")
    p.add_argument("--out", default=None,
                   help="output root (default: auto from model+variant, "
                        "e.g. eval_out/unik3d-vitl_dac-swinl_indoor)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # which models
    p.add_argument("--models", default="unik3d,dac",
                   help="comma list: any of unik3d,dac")
    # ADT
    p.add_argument("--adt-root", default="/group-volume/Fengjia/data/projectaria_tools_adt_data_clean")
    p.add_argument("--adt-rgb-subdir", default="videos_synthetic",
                   choices=["videos_synthetic", "videos_rgb"],
                   help="fisheye RGB source: videos_synthetic (rendered, pixel-aligned "
                        "with depth_npy GT — cleanest quant) or videos_rgb (real sensor, "
                        "slight GT misregistration)")
    p.add_argument("--adt-max-frames", type=int, default=100, help="-1 = all")
    p.add_argument("--adt-depth-max", type=float, default=10.0)
    p.add_argument("--image-resolution", type=int, default=512)
    p.add_argument("--gt-euclidean", action="store_true",
                   help="treat ADT GT as euclidean range (default: planar z, matching VGGT eval)")
    p.add_argument("--n-qual", type=int, default=6, help="ADT qual panels per model")
    # EgoExo
    p.add_argument("--egoexo-root", default="../EgoX/example/egoexo4D",
                   help="dir containing videos/<clip>/<stream>.mp4 (run from vggt-omega)")
    p.add_argument("--egoexo-stream", default="ego_GT", choices=["ego_GT", "ego_Prior"])
    p.add_argument("--n-frames", type=int, default=4, help="EgoExo frames per clip")
    # UniK3D
    p.add_argument("--unik3d-root", default=None, help="repo root (else $UNIK3D_ROOT / third_party)")
    p.add_argument("--unik3d-backbone", default="vitl", choices=["vits", "vitb", "vitl"])
    p.add_argument("--unik3d-resolution-level", type=int, default=9)
    p.add_argument("--no-unik3d-camera", action="store_true",
                   help="withhold the fisheye camera (UniK3D predicts rays itself)")
    p.add_argument("--unik3d-official-image", default=None,
                   help="official-mode image (default <repo>/assets/demo/kitti360.png, MEI fisheye)")
    p.add_argument("--unik3d-official-camera", default=None,
                   help="official-mode camera json (default <repo>/assets/demo/kitti360.json)")
    # DAC
    p.add_argument("--dac-root", default=None, help="repo root (else $DAC_ROOT / third_party)")
    p.add_argument("--dac-config", default="checkpoints/dac_swinl_indoor.json")
    p.add_argument("--dac-weights", default="checkpoints/dac_swinl_indoor.pt")
    p.add_argument("--dac-fwd-h", type=int, default=500)
    p.add_argument("--dac-fwd-w", type=int, default=750)
    p.add_argument("--dac-crop-wfov", type=float, default=180.0)
    p.add_argument("--dac-official-sample", default=None,
                   help="official-mode sample json (default <repo>/demo/input/scannetpp_sample.json)")
    a = p.parse_args()

    if a.out is None:
        a.out = _build_out_tag(a)
    print(f"[baselines] output dir: {a.out}")

    device = torch.device(a.device)
    want = {m.strip() for m in a.models.split(",") if m.strip()}

    unik = None
    if "unik3d" in want:
        from .predict_unik3d import UniK3DPredictor
        unik = UniK3DPredictor(
            backbone=a.unik3d_backbone, device=device, repo_root=a.unik3d_root,
            resolution_level=a.unik3d_resolution_level,
            use_camera=not a.no_unik3d_camera)
    dac = None
    if "dac" in want:
        from .predict_dac import DACPredictor
        dac = DACPredictor(
            config_file=a.dac_config, model_file=a.dac_weights, device=device,
            repo_root=a.dac_root, fwd_sz=(a.dac_fwd_h, a.dac_fwd_w),
            crop_wFoV=a.dac_crop_wfov)
    if unik is None and dac is None:
        raise SystemExit("[baselines] no models selected (--models unik3d,dac)")

    if a.mode in ("official", "all"):
        run_official(a, unik, dac)
    if a.mode in ("adt", "both", "all"):
        run_adt(a, unik, dac)
    if a.mode in ("egoexo", "both", "all"):
        run_egoexo(a, unik, dac)


if __name__ == "__main__":
    main()
