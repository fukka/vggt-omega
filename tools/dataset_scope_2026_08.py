# Copyright (c) 2026.
"""Reproduce every number in ``docs/research/dataset-scope-2026-08.md``.

Three independent checks, each runnable on its own machine:

``--adt ROOT``      **lambda_63 only.** Per-sequence frame inventory of the ADT
                    tree: how many frames each of ``videos_rgb``,
                    ``videos_synthetic`` and ``depth_npy`` holds, the frame-id
                    stride of the synthetic stream, and — for
                    ``decoration_seq132`` — how many of the 502 ``bedroom``
                    frames survive into each stream. The synthetic stride is the
                    whole question: a 400-frame ``videos_synthetic`` beside a
                    2730-frame ``videos_rgb`` is either a truncated extraction
                    (the first 400 frames) or a strided one (400 frames spread
                    over the whole take), and only the second is usable.

``--scannetpp CSV`` Summarises the two CSVs produced on the box by ``--emit``
                    (see below): scene completeness, DSLR frame counts, and the
                    per-scene fisheye field of view recovered from the KB4
                    coefficients in ``transforms.json``.

``--budget ...``    The render-cost table: how many scenes and frames a given
                    subsample costs in wall clock and disk, at the rate the one
                    already-rendered scene measured. Needs the same CSV plus
                    ``metadata/scene_types.json`` and the DAC scene list, so the
                    subsample can be stratified by room type instead of turning
                    into 221 apartments.

``--coverage``      **CPU-only, no data.** The lens question. Takes ScanNet++'s
                    measured KB4 calibration and this repo's Aria calibration of
                    record and asks: if a ScanNet++ DSLR frame is resampled onto
                    Aria's lens, what fraction of Aria's imaged cone receives a
                    real source pixel? A remap whose target has directions the
                    source never saw must leave them masked, not filled — see
                    ``fovbench``'s padding notes.

The Aria side imports ``finetune.eval.baselines.aria_fisheye``, so the
calibration stays defined in exactly one place.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import re
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

#: ADT frame files are ``frame_<6-digit id>_<capture timestamp ns>.<ext>``.
_FRAME_RE = re.compile(r"frame_(\d+)_")


# --------------------------------------------------------------------------- #
# ADT
# --------------------------------------------------------------------------- #
def _frame_ids(d: str):
    """Frame ids present in an ADT stream directory (``{}`` if there is none)."""
    if not os.path.isdir(d):
        return {}
    out = {}
    for f in os.listdir(d):
        m = _FRAME_RE.match(f)
        if m:
            out[int(m.group(1))] = f
    return out


def adt_report(root: str) -> None:
    seqs = sorted(s for s in os.listdir(root) if s.startswith("Apartment"))
    print(f"{'sequence':44s} {'rgb':>6s} {'syn':>6s} {'depth':>6s} "
          f"{'syn_span':>14s} {'stride':>14s} {'syn&depth':>9s}")
    for s in seqs:
        p = os.path.join(root, s)
        rgb = _frame_ids(os.path.join(p, "videos_rgb"))
        syn = _frame_ids(os.path.join(p, "videos_synthetic"))
        dep = _frame_ids(os.path.join(p, "depth_npy"))
        if syn:
            k = sorted(syn)
            span = f"{k[0]}-{k[-1]}"
            diffs = collections.Counter(k[i + 1] - k[i] for i in range(len(k) - 1))
            stride = ",".join(f"{d}x{n}" for d, n in sorted(diffs.items()))
        else:
            span, stride = "-", "-"
        both = len(set(syn) & set(dep))
        print(f"{s:44s} {len(rgb):6d} {len(syn):6d} {len(dep):6d} "
              f"{span:>14s} {stride:>14s} {both:9d}")

    # The bedroom segment, which is the only room-labelled data in the tree.
    for s in seqs:
        for name in ("room_annotations.csv", "room_annotations.json"):
            p = os.path.join(root, s, "videos_rgb", name)
            if not os.path.isfile(p):
                continue
            if name.endswith(".csv"):
                rows = list(csv.DictReader(open(p)))
                lab = [(int(r["frame_number"]), r["room"]) for r in rows]
            else:
                # The JSON form is interval-coded: {"annotations": [{"start",
                # "end", "room"}, ...]}. Intervals are inclusive and only cover
                # the labelled part of the take; everything else is unlabelled,
                # which is not the same thing as "not that room".
                j = json.load(open(p))
                lab = [(i, iv["room"])
                       for iv in j.get("annotations", [])
                       for i in range(int(iv["start"]), int(iv["end"]) + 1)]
            counts = collections.Counter(r for _, r in lab)
            print(f"\n[{s}] {name}: {len(lab)} rows, {dict(counts)}")
            base = os.path.join(root, s)
            rgb = _frame_ids(os.path.join(base, "videos_rgb"))
            syn = _frame_ids(os.path.join(base, "videos_synthetic"))
            dep = _frame_ids(os.path.join(base, "depth_npy"))
            for room in sorted(counts):
                if room == "unknown":
                    continue
                ids = [i for i, r in lab if r == room]
                print(f"  room={room!r} n={len(ids)} "
                      f"ids {min(ids)}-{max(ids)} | "
                      f"rgb {sum(i in rgb for i in ids)} "
                      f"depth {sum(i in dep for i in ids)} "
                      f"synthetic {sum(i in syn for i in ids)} "
                      f"syn&depth {sum(i in syn and i in dep for i in ids)}")


# --------------------------------------------------------------------------- #
# ScanNet++ inventory  (run --emit on the box, then feed the CSVs back here)
# --------------------------------------------------------------------------- #
_EMIT = r'''
import os, json, csv
S = {root!r}
scan, intr = [], []
for i, sc in enumerate(sorted(os.listdir(S))):
    d = os.path.join(S, sc, "dslr")
    r = {{"scene": sc}}
    for k in ("resized_images", "resized_undistorted_images", "render_depth",
              "resized_anon_masks"):
        p = os.path.join(d, k)
        r[k] = len(os.listdir(p)) if os.path.isdir(p) else -1
    r["transforms"] = int(os.path.isfile(os.path.join(d, "nerfstudio",
                                                      "transforms.json")))
    r["mesh"] = int(os.path.isfile(os.path.join(S, sc, "scans",
                                                "mesh_aligned_0.05.ply")))
    r["iphone"] = int(os.path.isdir(os.path.join(S, sc, "iphone")))
    scan.append(r)
    p = os.path.join(d, "nerfstudio", "transforms.json")
    if os.path.isfile(p):
        j = json.load(open(p))
        intr.append({{"scene": sc, "model": j.get("camera_model"),
                     "w": j.get("w"), "h": j.get("h"),
                     "fl_x": j.get("fl_x"), "fl_y": j.get("fl_y"),
                     "cx": j.get("cx"), "cy": j.get("cy"),
                     "k1": j.get("k1"), "k2": j.get("k2"),
                     "k3": j.get("k3"), "k4": j.get("k4"),
                     "n_frames": len(j.get("frames", [])),
                     "n_test": len(j.get("test_frames", []) or [])}})
for rows, path in ((scan, "/tmp/spp_scan.csv"), (intr, "/tmp/spp_intr.csv")):
    w = csv.DictWriter(open(path, "w"), fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
    print("wrote", path, len(rows))
'''


def emit_script(root: str) -> None:
    """Print the inventory script to run on the box (it needs the netapp mount)."""
    sys.stdout.write(_EMIT.format(root=root))


def _kb4_r(theta, k):
    t2 = theta * theta
    return theta * (1 + k[0] * t2 + k[1] * t2 ** 2 + k[2] * t2 ** 3 + k[3] * t2 ** 4)


def _theta_at_radius(r_norm, k, n=200001):
    """Largest monotonic theta whose KB4 radius reaches ``r_norm`` (radians)."""
    th = np.linspace(0.0, np.pi / 2, n)
    r = _kb4_r(th, k)
    turn = np.nonzero(np.diff(r) <= 0)[0]
    stop = int(turn[0]) + 1 if turn.size else n
    th, r = th[:stop], r[:stop]
    if r_norm >= r[-1]:
        return float(th[-1]), True          # clipped by the KB4 turnover
    return float(np.interp(r_norm, r, th)), False


def scannetpp_report(scan_csv: str, intr_csv: str, dac_scenes=()) -> None:
    scan = {r["scene"]: r for r in csv.DictReader(open(scan_csv))}
    intr = {r["scene"]: r for r in csv.DictReader(open(intr_csv))}
    for r in scan.values():
        for k in r:
            if k != "scene":
                r[k] = int(r[k])
    print(f"scenes on disk: {len(scan)}")
    print(f"  with transforms.json : {sum(r['transforms'] for r in scan.values())}")
    print(f"  with mesh_aligned    : {sum(r['mesh'] for r in scan.values())}")
    print(f"  with render_depth    : "
          f"{sum(r['render_depth'] >= 0 for r in scan.values())}")
    complete = {s: r for s, r in scan.items()
                if r["resized_images"] > 0 and r["transforms"] and r["mesh"]}
    n = [r["resized_images"] for r in complete.values()]
    print(f"COMPLETE (images + transforms + mesh): {len(complete)} scenes, "
          f"{sum(n)} DSLR frames "
          f"(min {min(n)}, median {int(np.median(n))}, max {max(n)})")
    already = {s: r["render_depth"] for s, r in scan.items()
               if r["render_depth"] > 0}
    print(f"  render_depth already present: {already}")
    need = sum(r["resized_images"] for s, r in complete.items()
               if scan[s]["render_depth"] < 0)
    print(f"  frames still needing a render pass: {need}")

    fovs = []
    for s, r in complete.items():
        i = intr.get(s)
        if not i or i["model"] != "OPENCV_FISHEYE":
            continue
        fx, fy = float(i["fl_x"]), float(i["fl_y"])
        cx, cy = float(i["cx"]), float(i["cy"])
        W, H = int(i["w"]), int(i["h"])
        k = tuple(float(i[f"k{j}"]) for j in (1, 2, 3, 4))
        corner = math.hypot(max(cx, W - 1 - cx) / fx, max(cy, H - 1 - cy) / fy)
        horiz = max(cx, W - 1 - cx) / fx
        vert = max(cy, H - 1 - cy) / fy
        td, clipd = _theta_at_radius(corner, k)
        th, _ = _theta_at_radius(horiz, k)
        tv, _ = _theta_at_radius(vert, k)
        fovs.append((s, 2 * math.degrees(td), 2 * math.degrees(th),
                     2 * math.degrees(tv), clipd, W, H))
    d = np.array([f[1] for f in fovs])
    h = np.array([f[2] for f in fovs])
    v = np.array([f[3] for f in fovs])
    print(f"\nfisheye FOV over {len(fovs)} complete OPENCV_FISHEYE scenes (deg):")
    for name, a in (("diagonal", d), ("horizontal", h), ("vertical", v)):
        print(f"  {name:10s} min {a.min():7.2f}  p5 {np.percentile(a,5):7.2f}  "
              f"median {np.median(a):7.2f}  p95 {np.percentile(a,95):7.2f}  "
              f"max {a.max():7.2f}")
    print(f"  scenes whose corner is past the KB4 turnover: "
          f"{sum(f[4] for f in fovs)}")
    sizes = collections.Counter((f[5], f[6]) for f in fovs)
    print(f"  frame sizes: {dict(sizes)}")
    for s in dac_scenes:
        if s in complete:
            print(f"  [DAC] {s}: {complete[s]['resized_images']} frames")


# --------------------------------------------------------------------------- #
# The lens question: does ScanNet++ cover Aria's cone?
# --------------------------------------------------------------------------- #
def coverage_report(intr_csv: str = "", side: int = 352) -> None:
    from finetune.eval.baselines.aria_fisheye import (aria_intrinsics,
                                                      kb4_max_incidence)

    # Aria's own 1408 px geometry, evaluated on a `side`-px grid. The cone is
    # scale-free, so a coarse grid changes the fractions below only in the last
    # reported digit and makes a 1000-scene sweep take seconds.
    a = aria_intrinsics(1408, 1408, rotated=True)
    th_max = a.usable_theta_max()
    H = W = int(side)
    k = 1408.0 / H
    fx, fy, cx, cy = a.fx / k, a.fy / k, a.cx / k, a.cy / k
    print("Aria RGB (repo calibration of record, KB4 fit, 1408x1408)")
    print(f"  fx={a.fx} fy={a.fy} cx={a.cx} cy={a.cy}")
    print(f"  k={a.k}")
    print(f"  KB4 turnover      {math.degrees(kb4_max_incidence(a.k)):.2f} deg "
          f"half-angle")
    print(f"  usable theta_max  {math.degrees(th_max):.2f} deg half-angle "
          f"({2*math.degrees(th_max):.2f} deg cone)")

    # Aria's target grid -> unit rays (only pixels inside the usable cone).
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    x = (u - cx) / fx
    y = (v - cy) / fy
    rd = np.hypot(x, y)
    th = _kb4_invert(rd, a.k)
    inside = np.isfinite(th) & (th <= th_max)
    sx = np.where(rd > 0, x / np.maximum(rd, 1e-12), 0.0)
    sy = np.where(rd > 0, y / np.maximum(rd, 1e-12), 0.0)
    thc = np.nan_to_num(th)
    rays = np.stack([np.sin(thc) * sx, np.sin(thc) * sy, np.cos(thc)], -1)
    print(f"  pixels inside the cone: {int(inside.sum())} / {H*W} "
          f"({100.0*inside.mean():.1f}%)")

    # The rim band: the outer third of the cone by incidence angle. This is the
    # region the whole FOV experiment reads, so it is where a void matters.
    lo = th_max * 2.0 / 3.0
    rim = inside & (th >= lo)
    print(f"  rim band theta in [{math.degrees(lo):.2f}, "
          f"{math.degrees(th_max):.2f}] deg: {int(rim.sum())} px "
          f"({100.0*rim.sum()/inside.sum():.1f}% of the cone)")

    scenes = _coverage_scenes(intr_csv)
    print("\nfraction of Aria's imaged cone that lands inside a ScanNet++ frame")
    if len(scenes) > 8:
        cone, band = [], []
        for _, q in scenes:
            cov = _project_kb4(rays, q)
            cone.append((cov & inside).sum() / inside.sum())
            band.append((rim & ~cov).sum() / rim.sum())
        cone = 100 * np.asarray(cone)
        band = 100 * np.asarray(band)
        print(f"  over {len(scenes)} scenes:")
        for label, a in (("covered, whole cone (%)", cone),
                         ("VOID, rim band only (%)", band)):
            print(f"    {label:24s} min {a.min():6.2f}  p5 {np.percentile(a,5):6.2f}"
                  f"  median {np.median(a):6.2f}  p95 {np.percentile(a,95):6.2f}"
                  f"  max {a.max():6.2f}")
        print(f"    scenes with zero void: {int((cone >= 99.999).sum())} "
              f"/ {len(scenes)}")
        # The largest target cone a given source can fill: its vertical half-FOV.
        vh = np.asarray([q["vert"] / 2.0 for _, q in scenes])
        print(f"    ScanNet++ vertical HALF-fov (deg): min {vh.min():.2f}  "
              f"p5 {np.percentile(vh,5):.2f}  median {np.median(vh):.2f}  "
              f"max {vh.max():.2f}   (Aria asks for "
              f"{math.degrees(th_max):.2f})")
        for cap in (np.percentile(vh, 5), np.median(vh)):
            sub = inside & (th <= math.radians(cap))
            print(f"    capping the target at {cap:5.2f} deg is void-free and "
                  f"reaches {100.0*sub.sum()/inside.sum():.1f}% of the cone")
        return
    print(f"{'scene':14s} {'src diag':>9s} {'src horiz':>10s} {'src vert':>9s} "
          f"{'covered':>8s} {'void':>7s} {'rim void':>9s} {'src kept':>9s}")
    for name, q in scenes:
        cov = _project_kb4(rays, q)
        frac = (cov & inside).sum() / max(1, inside.sum())
        rv = (rim & ~cov).sum() / max(1, rim.sum())
        print(f"{name:14s} {q['diag']:8.2f}  {q['horiz']:9.2f}  {q['vert']:8.2f}  "
              f"{100*frac:7.2f}%  {100*(1-frac):6.2f}%  {100*rv:8.2f}%  "
              f"{100*_source_kept(q, th_max):8.2f}%")


def _source_kept(q, th_max: float, step: int = 4) -> float:
    """Share of the source frame's pixels that fall inside the target's cone.

    The other half of the coverage question: the void says what the target asks
    for and cannot get, this says what the source has and the target throws
    away. Both are real and neither implies the other.
    """
    u, v = np.meshgrid(np.arange(0, q["W"], step) + 0.5,
                       np.arange(0, q["H"], step) + 0.5)
    r = np.hypot((u - q["cx"]) / q["fx"], (v - q["cy"]) / q["fy"])
    th = _kb4_invert(r, q["k"])
    return float((np.isfinite(th) & (th <= th_max)).mean())


def _kb4_invert(r_norm, k, n=100001):
    th = np.linspace(0.0, np.pi / 2, n)
    r = _kb4_r(th, k)
    turn = np.nonzero(np.diff(r) <= 0)[0]
    stop = int(turn[0]) + 1 if turn.size else n
    th, r = th[:stop], r[:stop]
    out = np.interp(r_norm, r, th, left=0.0, right=np.nan)
    return np.where(r_norm > r[-1], np.nan, out)


def _project_kb4(rays, p):
    """True where a unit ray lands inside this KB4 camera's rectangle."""
    xyz = rays.reshape(-1, 3)
    th = np.arccos(np.clip(xyz[:, 2], -1, 1))
    rho = np.hypot(xyz[:, 0], xyz[:, 1])
    rd = _kb4_r(th, p["k"])
    sx = np.where(rho > 0, xyz[:, 0] / np.maximum(rho, 1e-12), 0.0)
    sy = np.where(rho > 0, xyz[:, 1] / np.maximum(rho, 1e-12), 0.0)
    uu = p["fx"] * rd * sx + p["cx"]
    vv = p["fy"] * rd * sy + p["cy"]
    ok = ((th <= p["theta_turn"]) & (uu >= 0) & (uu <= p["W"] - 1)
          & (vv >= 0) & (vv <= p["H"] - 1))
    return ok.reshape(rays.shape[:2])


def _coverage_scenes(intr_csv: str):
    """(name, params) for the scenes to test coverage against."""
    rows = []
    if intr_csv and os.path.isfile(intr_csv):
        rows = [r for r in csv.DictReader(open(intr_csv))
                if r["model"] == "OPENCV_FISHEYE"]
    else:
        # The two calibrations docs/research/scannetpp-camera-reference.md holds.
        rows = [{"scene": "3f15a9266d", "w": 1752, "h": 1168,
                 "fl_x": 616.721, "fl_y": 617.354, "cx": 878.593, "cy": 589.767,
                 "k1": 0.06109, "k2": 0.003350, "k3": 0.002988, "k4": -0.001002}]
    out = []
    for r in rows:
        W, H = int(r["w"]), int(r["h"])
        k = tuple(float(r[f"k{j}"]) for j in (1, 2, 3, 4))
        p = {"fx": float(r["fl_x"]), "fy": float(r["fl_y"]),
             "cx": float(r["cx"]), "cy": float(r["cy"]), "W": W, "H": H, "k": k}
        th = np.linspace(0, np.pi / 2, 100001)
        rr = _kb4_r(th, k)
        turn = np.nonzero(np.diff(rr) <= 0)[0]
        p["theta_turn"] = float(th[int(turn[0])]) if turn.size else np.pi / 2
        for tag, rad in (("diag", math.hypot(max(p["cx"], W - 1 - p["cx"]) / p["fx"],
                                             max(p["cy"], H - 1 - p["cy"]) / p["fy"])),
                         ("horiz", max(p["cx"], W - 1 - p["cx"]) / p["fx"]),
                         ("vert", max(p["cy"], H - 1 - p["cy"]) / p["fy"])):
            t, _ = _theta_at_radius(rad, k)
            p[tag] = 2 * math.degrees(t)
        out.append((r["scene"], p))
    return out


def budget_report(scan_csv: str, types_json: str, dac_file: str) -> None:
    """The render-cost table: how many scenes x frames, at what wall clock.

    ``SEC_PER_FRAME`` and ``MB_PER_FRAME`` are measured, not guessed — from the
    one scene already rendered on the box (``3f15a9266d``, 906 frames): the
    mtime span of ``render_depth`` divided by the frame count, and the summed
    size of ``render_depth`` *and* ``render_rgb``, because ``common/render.py``
    writes both whether or not the RGB is wanted.
    """
    SEC_PER_FRAME = 0.204
    MB_PER_FRAME = 0.632

    scan = {r["scene"]: {k: (v if k == "scene" else int(v))
                         for k, v in r.items()}
            for r in csv.DictReader(open(scan_csv))}
    types = json.load(open(types_json)) if types_json else {}
    dac = set(open(dac_file).read().split()) if dac_file else set()
    complete = {s: r for s, r in scan.items()
                if r["resized_images"] > 0 and r["transforms"] and r["mesh"]}
    pool = {s: r for s, r in complete.items() if s not in dac}
    print(f"complete {len(complete)} scenes / "
          f"{sum(r['resized_images'] for r in complete.values())} frames")
    print(f"held out (DAC) {len(dac & set(complete))} scenes; "
          f"pool {len(pool)} scenes / "
          f"{sum(r['resized_images'] for r in pool.values())} frames")
    if types:
        c = collections.Counter(types.get(s, "?") for s in pool)
        print(f"pool spans {len(c)} scene types; top: {c.most_common(8)}")

    # Round-robin over scene type, so a subsample stays diverse rather than
    # becoming 221 apartments.
    by_type = collections.defaultdict(list)
    for s in sorted(pool):
        by_type[types.get(s, "?")].append(s)
    keys = sorted(by_type, key=lambda k: -len(by_type[k]))

    def pick(n):
        out, i = [], 0
        while len(out) < n:
            grew = False
            for k in keys:
                if i < len(by_type[k]) and len(out) < n:
                    out.append(by_type[k][i])
                    grew = True
            i += 1
            if not grew:
                break
        return out

    print(f"\n{'option':34s} {'scenes':>7s} {'frames':>9s} {'hours':>7s} "
          f"{'GB':>7s} {'types':>6s}")
    for n, stride, label in ((len(pool), 1, "whole pool, every frame"),
                             (300, 1, "300 scenes, every frame"),
                             (300, 4, "300 scenes, stride 4"),
                             (150, 4, "150 scenes, stride 4"),
                             (120, 6, "120 scenes, stride 6"),
                             (60, 4, "60 scenes, stride 4")):
        sel = pick(n)
        fr = sum(-(-pool[s]["resized_images"] // stride) for s in sel)
        print(f"{label:34s} {len(sel):7d} {fr:9d} {fr*SEC_PER_FRAME/3600:7.1f} "
              f"{fr*MB_PER_FRAME/1000:7.1f} "
              f"{len(set(types.get(s, '?') for s in sel)):6d}")
    if dac:
        # The DAC test scenes have to be rendered too, or there is no protocol.
        test = ("1d003b07bd", "1f7cbbdde1", "2a1a3afad9", "3e928dc2f6",
                "4ef75031e3")
        fr = sum(complete[s]["resized_images"] for s in test if s in complete)
        print(f"{'DAC test (5 scenes, required)':34s} {len(test):7d} {fr:9d} "
              f"{fr*SEC_PER_FRAME/3600:7.2f} {fr*MB_PER_FRAME/1000:7.1f}")
        nomesh = sorted(s for s in dac
                        if s in scan and not scan[s]["mesh"])
        print(f"\nDAC scenes with no mesh (cannot be rendered): {nomesh}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adt", metavar="ROOT",
                    help="ADT root (lambda_63: "
                         "~/Documents/projectaria_tools_adt_data_clean)")
    ap.add_argument("--emit", metavar="SCANNETPP_DATA",
                    help="print the ScanNet++ inventory script to run on the box")
    ap.add_argument("--scannetpp", nargs=2, metavar=("SCAN_CSV", "INTR_CSV"),
                    help="summarise the two CSVs --emit produced")
    ap.add_argument("--budget", nargs=3,
                    metavar=("SCAN_CSV", "SCENE_TYPES_JSON", "DAC_SCENES_TXT"),
                    help="render-cost table over the non-DAC pool")
    ap.add_argument("--coverage", action="store_true",
                    help="ScanNet++ -> Aria cone coverage (needs no data)")
    ap.add_argument("--intr", default="", help="intrinsics CSV for --coverage; "
                                               "without it, one reference scene")
    ap.add_argument("--grid", type=int, default=352,
                    help="side of the Aria grid --coverage evaluates on")
    a = ap.parse_args()
    if not any((a.adt, a.emit, a.scannetpp, a.budget, a.coverage)):
        ap.print_help()
        return 2
    if a.adt:
        adt_report(a.adt)
    if a.emit:
        emit_script(a.emit)
    if a.scannetpp:
        scannetpp_report(*a.scannetpp)
    if a.budget:
        budget_report(*a.budget)
    if a.coverage:
        coverage_report(a.intr, a.grid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
