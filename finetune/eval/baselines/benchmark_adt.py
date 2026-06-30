# Copyright (c) 2026.
"""Benchmark every available monocular-depth model on ADT (fisheye), with size,
runtime, FLOPs and depth metrics under each model's native alignment.

Two leaderboards (``--domains both``):

  • ``common`` — a single, apples-to-apples space: **ERP euclidean-range** (DAC's
    native domain). Every other model is run on the raw Aria fisheye frame; its
    planar-z prediction is converted to range (``z/cosθ`` via the fisheye ray LUT)
    and forward-warped into the same ERP grid the GT uses (the *tested*
    ``fisheye_to_erp_fwd``), so all models are scored pixel-for-pixel against one
    GT. Pinhole-trained nets (DAv2/MiDaS/…) are expected to degrade here — that's
    the fisheye gap the project targets.

  • ``native`` — each model in the domain it was built for, each at its best:
    pinhole nets on the **rectified pinhole** frame vs pinhole GT; DAC in ERP;
    UniK3D in fisheye planar-z. Rows are NOT cross-comparable (different spaces).

Alignment: every model is reported under ``none`` (metric) **and** its native mode
— ``disparity_scale_shift`` for relative nets (MiDaS protocol), ``scale_only``
(median) for metric nets. See :mod:`finetune.eval.metrics`.

Models are discovered from :mod:`model_zoo`; missing weights are downloaded
(``--download``) or skipped with instructions. ``--list`` just prints the registry
+ availability. Profiling (latency, FLOPs) runs on the eval device — use a GPU.

    python -m finetune.eval.baselines.benchmark_adt --list
    python -m finetune.eval.baselines.benchmark_adt --download --models dav2,dac,unik3d
    python -m finetune.eval.baselines.benchmark_adt --adt-root /path/to/adt \
        --models all --domains both --max-frames 50 --out eval_out/bench
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from . import model_zoo as zoo
from .aria_fisheye import aria_intrinsics
from .erp_utils import aria_fisheye_ray_lut, fisheye_to_erp_fwd
from ..metrics import aggregate_metrics, align_depth, depth_metrics

# Fixed ERP grid for the common table (matches DAC swinl/rn101 indoor: cano=1400).
ERP_CANO, ERP_FWD, ERP_WFOV = 1400, (500, 750), 180.0
_METRIC_KEYS = ("AbsRel", "delta1", "RMSE")


# --------------------------------------------------------------------------- #
# ERP common-space helpers (z-depth -> euclidean range -> ERP grid)
# --------------------------------------------------------------------------- #
def _z_to_range(depth_z: np.ndarray, ray_z: np.ndarray) -> np.ndarray:
    """Planar z-depth -> euclidean range along the ray (range = z / cosθ)."""
    return depth_z / np.clip(ray_z, 1e-3, None)


def _warp_to_erp(range_hw: np.ndarray, valid_hw: np.ndarray, cam) -> dict:
    z = np.zeros(range_hw.shape, np.float32)
    return fisheye_to_erp_fwd(z[..., None].repeat(3, -1), range_hw.astype(np.float32),
                              valid_hw.astype(np.float32), cam.opencv_fisheye_params(),
                              ERP_CANO, ERP_FWD, ERP_WFOV)


def _erp_gt(cam, gt_z: np.ndarray, valid: np.ndarray, ray_z: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray]:
    """GT fisheye planar-z -> ERP range GT + valid/active mask on the ERP grid."""
    gt_range = _z_to_range(gt_z, ray_z) * (valid > 0)
    w = _warp_to_erp(gt_range, valid, cam)
    mask = (w["active"] > 0.5) & (w["valid"] > 0.5) & (w["depth"] > 1e-3)
    return w["depth"].astype(np.float32), mask


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #
def _flops_g(adapter: zoo.Adapter, rgb01: np.ndarray) -> Optional[float]:
    """Best-effort GFLOPs for hf_depth adapters.

    Primary: ``torch.profiler(with_flops=True)`` — built-in, handles matmul /
    linear / conv / scaled_dot_product_attention in PyTorch ≥ 2.0.
    Fallback: fvcore if profiler total is 0 (e.g. all ops unsupported).
    """
    if adapter.spec.kind != "hf_depth":
        return None
    try:
        import torch
        px = adapter._pixel_values(rgb01)
        dev = getattr(adapter, "device", None)
        acts = [torch.profiler.ProfilerActivity.CPU]
        if dev is not None and dev.type == "cuda":
            acts.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.no_grad():
            with torch.profiler.profile(activities=acts, with_flops=True,
                                        record_shapes=True) as prof:
                adapter.model(pixel_values=px)
        total = sum(getattr(e, "flops", 0) or 0 for e in prof.key_averages())
        if total > 0:
            return float(total) / 1e9
        # fvcore fallback — catches conv/linear for models not traced by profiler
        if zoo._have("fvcore"):
            from fvcore.nn import FlopCountAnalysis
            fa = FlopCountAnalysis(adapter.model, (px,))
            fa.unsupported_ops_warnings(False)
            fa.uncalled_modules_warnings(False)
            t = float(fa.total())
            if t > 0:
                return t / 1e9
    except Exception:
        pass
    return None


def _latency_ms(adapter: zoo.Adapter, rgb01: np.ndarray, cam, device, iters: int) -> float:
    is_dac = adapter.spec.kind == "dac"
    call = ((lambda: adapter.predict_erp(rgb01, cam)) if is_dac
            else (lambda: adapter.predict_frame(rgb01, cam, "fisheye")))
    for _ in range(2):  # warmup
        call()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        call()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


# --------------------------------------------------------------------------- #
# Per-model evaluation
# --------------------------------------------------------------------------- #
def _eval_common(adapter, frames, cam, ray_z) -> Dict[str, dict]:
    """ERP-range common space. ``frames`` = list of (rgb01[H,W,3], gt_z[H,W], valid[H,W])."""
    modes = adapter.spec.align_modes
    acc = {m: [] for m in modes}
    is_dac = adapter.spec.kind == "dac"
    for rgb01, gt_z, valid in frames:
        gt_erp, mask = _erp_gt(cam, gt_z, valid, ray_z)
        if mask.sum() < 64:
            continue
        if is_dac:
            out = adapter.predict_erp(rgb01, cam)
            pred_erp = out["depth"].astype(np.float32)
            mask = mask & (out["active"] > 0.5)
        else:
            z = adapter.predict_frame(rgb01, cam, "fisheye")
            pred_erp = _warp_to_erp(_z_to_range(z, ray_z), np.ones_like(z), cam)["depth"]
        for m in modes:
            acc[m].append(depth_metrics(align_depth(pred_erp, gt_erp, mask, m), gt_erp, mask))
    return {m: aggregate_metrics(acc[m]) for m in modes}


def _eval_native_planar(adapter, frames, cam) -> Dict[str, dict]:
    """Native planar-z space: score the model's depth vs frame GT directly."""
    modes = adapter.spec.align_modes
    acc = {m: [] for m in modes}
    for rgb01, gt_z, valid in frames:
        mask = valid > 0
        if mask.sum() < 64:
            continue
        z = adapter.predict_frame(rgb01, cam, "native")
        for m in modes:
            acc[m].append(depth_metrics(align_depth(z, gt_z, mask, m), gt_z, mask))
    return {m: aggregate_metrics(acc[m]) for m in modes}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _load_frames(adt_root, rgb_subdir, rectify, res, max_frames) -> List[tuple]:
    from ..adt_depth import ADTWindowDataset
    from ..run_eval import _find_seq_dirs
    seq_dirs = _find_seq_dirs(adt_root, rgb_subdir, "depth_npy")
    if not seq_dirs:
        raise SystemExit(f"[bench] no ADT seqs under {adt_root!r} (rgb_subdir={rgb_subdir!r})")
    ds = ADTWindowDataset(seq_dirs, seq_len=1, window_stride=1, image_resolution=res,
                          depth_scale=0.001, depth_max_m=10.0, max_frames=max_frames,
                          rgb_subdir=rgb_subdir, depth_subdir="depth_npy", rectify=rectify)
    frames = []
    for i in range(len(ds)):
        it = ds[i]
        rgb = it["images"][0].permute(1, 2, 0).clamp(0, 1).numpy().astype(np.float32)
        gt = it["depths"][0].numpy().astype(np.float32)
        valid = it["valid_masks"][0].numpy().astype(np.float32)
        frames.append((rgb, gt, valid))
    return frames


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(x, nd=3):
    return "  n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _cell(met) -> str:
    """Fixed-width AbsRel/δ1/RMSE cell (lower AbsRel & RMSE better, higher δ1 better)."""
    if not met:
        return f"{'—':^21s}"
    return f"{_fmt(met['AbsRel']):>6s}/{_fmt(met['delta1']):>5s}/{_fmt(met['RMSE'],2):>6s}"


def _print_table(title: str, rows: List[dict], modes: List[str]):
    print(f"\n{'='*94}\n{title}\n  cells = AbsRel / δ1 / RMSE(m)   ·   ↓AbsRel ↓RMSE better, ↑δ1 better\n{'='*94}")
    hdr = f"{'model':21s}{'size':13s}{'par(M)':>7s}{'GFLOP':>7s}{'ms':>7s}"
    for m in modes:
        hdr += "  | " + f"{m:^21s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = (f"{r['key']:21s}{r['size']:13s}{_fmt(r['params_m'],1):>7s}"
                f"{_fmt(r['flops_g'],1):>7s}{_fmt(r['latency_ms'],1):>7s}")
        for m in modes:
            line += "  | " + _cell(r["metrics"].get(m))
        print(line)


def _save(out_dir, payload):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "benchmark.json"), "w") as f:
        json.dump(payload, f, indent=2)
    # flat CSV: one row per model × domain × align mode
    import csv
    with open(os.path.join(out_dir, "benchmark.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "key", "family", "size", "output_type", "params_m",
                    "weight_mb", "flops_g", "latency_ms", "align", "AbsRel", "delta1",
                    "RMSE", "n_frames", "n_valid_total"])
        for dom, rows in payload["tables"].items():
            for r in rows:
                for m, met in r["metrics"].items():
                    if not met:
                        continue
                    w.writerow([dom, r["key"], r["family"], r["size"], r["output_type"],
                                _fmt(r["params_m"], 2), _fmt(r["weight_mb"], 1),
                                _fmt(r["flops_g"], 2), _fmt(r["latency_ms"], 2), m,
                                _fmt(met["AbsRel"]), _fmt(met["delta1"]),
                                _fmt(met["RMSE"], 3), met.get("n_frames", 0),
                                met.get("n_valid_total", 0)])
    print(f"\n[bench] wrote {out_dir}/benchmark.{{json,csv}}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cmd_list(specs):
    print(f"{'key':22s}{'family':24s}{'size':14s}{'type':9s}{'dev':4s}{'state':12s}detail")
    print("-" * 110)
    for s in specs:
        st, detail = zoo.status(s)
        dev = "📱" if s.on_device else ""
        exp = " (exp)" if s.experimental else ""
        print(f"{s.key:22s}{s.family:24s}{s.size:14s}{s.output_type:9s}{dev:4s}"
              f"{st:12s}{detail}{exp}")
    print(f"\n{len(specs)} variants. align modes are per-model: relative→disparity_scale_shift, "
          f"metric→scale_only (+ none always).")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="all",
                   help="comma keys/families (e.g. dav2,dac,unik3d) or 'all'")
    p.add_argument("--list", action="store_true", help="print registry + availability and exit")
    p.add_argument("--download", action="store_true", help="fetch missing weights for selected")
    p.add_argument("--domains", default="both", choices=["common", "native", "both"])
    p.add_argument("--adt-root", default="/group-volume/Fengjia/data/projectaria_tools_adt_data_clean")
    p.add_argument("--adt-rgb-subdir", default="videos_synthetic",
                   help="GT-aligned rendered RGB (videos_synthetic) or real (videos_rgb)")
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--latency-iters", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--allow-cpu", action="store_true",
                   help="permit CPU (default: error — runtime/FLOPs need a GPU)")
    p.add_argument("--dav3-id", default=None, help="override the experimental DAv3 HF id")
    p.add_argument("--out", default="eval_out/benchmark_adt")
    a = p.parse_args()

    specs = zoo.get_specs(None if a.models == "all" else
                          [m.strip() for m in a.models.split(",") if m.strip()])
    if a.dav3_id:
        import dataclasses
        specs = [dataclasses.replace(s, ref=a.dav3_id, experimental=False)
                 if s.family == "Depth-Anything-V3" else s for s in specs]

    if a.list:
        _cmd_list(specs)
        return

    if a.download:
        print("[bench] resolving weights for selected models …")
        for s in specs:
            st, _ = zoo.status(s)
            if st == "download":
                ok, msg = zoo.download(s)
                print(f"  {'✓' if ok else '✗'} {s.key}: {msg}")
            else:
                print(f"  · {s.key}: {st}")

    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[bench] --device cuda but torch.cuda.is_available() is False "
                         "(CPU-only torch build?). Install CUDA torch or use --allow-cpu.")
    if device.type != "cuda" and not a.allow_cpu:
        raise SystemExit("[bench] refusing to run on CPU: latency/FLOPs need a GPU. "
                         "Use --device cuda on a GPU box, or --allow-cpu for a scaffolding run.")
    print(f"[bench] device = {device} | ADT = {a.adt_root} ({a.adt_rgb_subdir})")

    ready = [s for s in specs if zoo.status(s)[0] == "ready"]
    skipped = [s for s in specs if zoo.status(s)[0] != "ready"]
    for s in skipped:
        print(f"[bench] skip {s.key}: {zoo.status(s)[1]} (run with --download to fetch)")
    if not ready:
        raise SystemExit("[bench] no ready models. Use --list to see availability, "
                         "--download to fetch weights.")

    cam = aria_intrinsics(a.res, a.res, rotated=True)
    ray_z = aria_fisheye_ray_lut(cam)[..., 2].astype(np.float32)

    want_common = a.domains in ("common", "both")
    want_native = a.domains in ("native", "both")
    fisheye_frames = _load_frames(a.adt_root, a.adt_rgb_subdir, False, a.res, a.max_frames)
    pinhole_frames = None  # lazy: only if a pinhole-native model is run

    tables: Dict[str, List[dict]] = {"common": [], "native": []}
    for s in ready:
        print(f"\n[bench] ── {s.key} ({s.family} {s.size}) ──")
        try:
            ad = zoo.build_adapter(s)
            ad.load(device)
        except Exception as exc:  # noqa: BLE001
            print(f"[bench]   load FAILED: {type(exc).__name__}: {exc}")
            continue

        # The common (ERP) table warps GT at cano=ERP_CANO; DAC must predict on the
        # same ERP grid for that row to be valid (indoor swinl/rn101 use cano=1400).
        if s.kind == "dac" and getattr(ad.pred, "cano", ERP_CANO) != ERP_CANO:
            print(f"[bench]   WARN DAC cano={ad.pred.cano} != ERP grid {ERP_CANO}; the "
                  f"common-table DAC row may be misaligned (GT is warped at cano={ERP_CANO}).")

        sample = fisheye_frames[0][0]
        prof = dict(params_m=ad.num_params() / 1e6, weight_mb=ad.weight_mb(),
                    flops_g=_flops_g(ad, sample),
                    latency_ms=_latency_ms(ad, sample, cam, device, a.latency_iters))
        print(f"[bench]   params={prof['params_m']:.1f}M  weight≈{prof['weight_mb']:.0f}MB  "
              f"FLOPs={_fmt(prof['flops_g'],1)}G  latency={prof['latency_ms']:.1f}ms/frame")

        base = dict(key=s.key, family=s.family, size=s.size, output_type=s.output_type, **prof)
        if want_common:
            try:
                m = _eval_common(ad, fisheye_frames, cam, ray_z)
            except Exception as exc:  # noqa: BLE001
                print(f"[bench]   common eval FAILED: {type(exc).__name__}: {exc}")
                m = {}
            tables["common"].append({**base, "metrics": m})

        if want_native:
            try:
                if s.kind in ("hf_depth", "metric3d_hub"):
                    if pinhole_frames is None:  # lazy: rectified-pinhole pass only if needed
                        pinhole_frames = _load_frames(a.adt_root, a.adt_rgb_subdir, True,
                                                      a.res, a.max_frames)
                    m = _eval_native_planar(ad, pinhole_frames, cam)
                elif s.kind == "unik3d":
                    m = _eval_native_planar(ad, fisheye_frames, cam)
                else:  # dac: native == ERP == common
                    m = _eval_common(ad, fisheye_frames, cam, ray_z) if not want_common \
                        else tables["common"][-1]["metrics"]
                tables["native"].append({**base, "metrics": m})
            except Exception as exc:  # noqa: BLE001
                print(f"[bench]   native eval FAILED: {type(exc).__name__}: {exc}")

        del ad
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if want_common and tables["common"]:
        modes = sorted({m for r in tables["common"] for m in r["metrics"]},
                       key=lambda x: (x != "none", x))
        _print_table("COMMON — ERP euclidean-range (apples-to-apples; fisheye input)",
                     tables["common"], modes)
    if want_native and tables["native"]:
        modes = sorted({m for r in tables["native"] for m in r["metrics"]},
                       key=lambda x: (x != "none", x))
        _print_table("NATIVE — each model in its own domain (NOT cross-comparable)",
                     tables["native"], modes)

    _save(a.out, dict(device=str(device), adt_root=a.adt_root, rgb_subdir=a.adt_rgb_subdir,
                      max_frames=a.max_frames, erp=dict(cano=ERP_CANO, fwd=ERP_FWD, wfov=ERP_WFOV),
                      tables=tables))


if __name__ == "__main__":
    main()
