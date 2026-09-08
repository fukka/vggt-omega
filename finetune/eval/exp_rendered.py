# Copyright (c) 2026.
"""VGGT-Omega on Blender-rendered ADT frames: 4 settings x {single, multi}-frame.

The four settings are the 2x2 of projection x validity, all rendered from the same
equirectangular panorama at the same pose, so they differ in nothing else:

    fisheye_full     Aria KB4, KB4 extended past its turnover -> 0% invalid
    fisheye_masked   same pixels, real imaged-disc mask       -> 14.75% invalid
    persp_full       pinhole aligned to FisheyeRectifier      -> 0% invalid
    persp_masked     same pixels, analytic rectification mask -> 32.29% invalid
    persp_crop       inscribed pinhole (focal_out_norm 0.371, 106.9 deg): no
                     invalid region by construction -- the free alternative that
                     drops the rim instead of filling the corners (cell 5)

Grading region (``--region``): ``own`` grades a full/masked pair on the masked
arm's mask and the crop on its own frame; ``crop`` restricts EVERY arm to the
crop's footprint on its grid (``mask_<proj>_in_crop.npy``), the smallest region
common to all five, so cross-arm comparisons are on identical scene directions.

Why rendered rather than real footage: the `_full` arms hold **true content** in
regions a real Aria frame cannot supply, because the lens never imaged them. That
is the ground-truth ("oracle") fill the earlier real-footage 2x2 could only
approximate with `replicate`, and it is obtainable only from a renderer that owns
the scene.

Scoring rule — the one that makes full-vs-masked a clean contrast
----------------------------------------------------------------
Both arms of a projection are scored on the **masked** arm's valid region (the
smaller set), never on the full arm's larger one. Otherwise the full arm would be
credited for area the masked arm does not even have, and the comparison would
measure coverage instead of the thing under test. With this rule the two arms see
different inputs and are graded on identical pixels, so any difference is the
encoder reacting to what was in the invalid region.

Across projections the pixel grids differ (the rectified grid oversamples the
periphery ~10x), so absolute numbers are not comparable between the fisheye and
perspective rows. The within-row contrast, and the interaction between rows, are.

Camera pose and FoV
-------------------
Depth is the downstream readout; the camera head is the thing the hypothesis is
about. In the multi-frame mode every window is scored on its relative poses
(RRA/RTA/AUC@30 as in the VGGT paper, plus a Sim(3) ATE in metres) against the
renderer's own camera trajectory, so a black region that corrupts the camera
estimate shows up here before it shows up in depth. The inferred FoV is graded
against the true pinhole FoV for the perspective arms only -- a fisheye image
has no pinhole FoV to be right about.

Single vs multi-frame
---------------------
`--seq-len 1` feeds one frame; `--seq-len 8` feeds a window. VGGT resolves
monocular scale ambiguity through cross-view attention, so the multi-frame arm is
where a corrupted input can do damage beyond its own frame. Windows must carry
real camera translation — pure rotation is triangulation-degenerate and would
handicap the multi-frame arm for reasons unrelated to the fill.

Usage
-----
::

    python -m finetune.eval.exp_rendered --render-root <ROOT> \\
        --vggt-checkpoint <CKPT> --out runs/exp_rendered --seq-len 1
    python -m finetune.eval.exp_rendered --render-root <ROOT> ... --seq-len 8
"""
from __future__ import annotations

import sys as _sys, os as _os
if not __package__:
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
    __package__ = "finetune.eval"

import argparse
import glob
import json
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.fill import apply_fill
from .metrics import align_depth, depth_metrics

SETTINGS = ("fisheye_full", "fisheye_masked", "persp_full", "persp_masked", "persp_crop")
PROJECTIONS = ("fisheye", "persp")
REGIONS = ("own", "crop")


def parse_setting(setting: str):
    """'fisheye_full' -> ('fisheye', 'full');  'persp_fill_mean' -> ('persp', 'mean').

    The fill arms are the point of the ladder: black, then progressively better
    cheap fills, then the rendered oracle. They share the MASKED arm's pixels and
    differ only in what goes in the hole, so the ladder measures how much of the
    oracle's gain a fill that invents nothing can already capture -- which is the
    number that decides whether a generative filler is worth building.
    """
    for proj in PROJECTIONS:
        if setting == f"{proj}_full":
            return proj, "full"
        if setting == f"{proj}_masked":
            return proj, "black"
        if setting == f"{proj}_crop":
            return proj, "crop"
        if setting.startswith(f"{proj}_fill_"):
            return proj, setting[len(proj) + 6:]
    raise ValueError(f"unknown setting {setting!r}")


def grading_mask_of(setting: str) -> str:
    """Every arm of a projection is graded on that projection's analytic mask --
    never on the larger region the full arm happens to have. Otherwise the full
    arm would score better for covering more, and the experiment would measure
    coverage instead of what is in the hole."""
    proj, arm = parse_setting(setting)
    return f"mask_{proj}_crop_valid" if arm == "crop" else f"mask_{proj}_valid"


# The images in a frame dir are in the rot90(k=3) upright frame; meta.json's
# ``T_WC`` is the RAW Aria RGB camera (OpenCV axes, sensor orientation). A unit
# ray (dx, dy, dz) in the rotated camera's axes is (dy, -dx, dz) in the raw
# camera's, i.e. raw = A_ROT @ rotated.
A_ROT = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

POSE_KEYS = ("rot_err_deg", "trans_err_deg", "rra5", "rra15", "rra30",
             "rta5", "rta15", "rta30", "auc30", "ate_m", "sim3_scale")


def gt_camera(frame_dir: str) -> Tuple[np.ndarray, float]:
    """Ground-truth camera of a rendered frame: (3x4 cam-from-world, pinhole hFoV).

    The camera that made the image is the raw one conjugated by A_ROT, so its
    pose is ``T_WC @ diag(A_ROT, 1)`` and its cam-from-world the inverse of that.
    This was checked on data rather than read off a docstring: unprojecting
    frame_0840's persp depth with this pose and reprojecting into frame_0830
    lands with 0.12% median relative depth error and 98% of pixels within 3%;
    the identity gives 4.5%, A_ROT^T 7.5%, an axis flip 3.9-5.0%.

    The FoV is the pinhole arm's true horizontal field of view from
    ``Knew_pinhole``; it is meaningless for the fisheye arms, whose image is not
    a pinhole projection, and callers must not grade those against it.
    """
    m = json.load(open(os.path.join(frame_dir, "meta.json")))
    T = np.asarray(m["T_WC"], dtype=np.float64)
    A4 = np.eye(4)
    A4[:3, :3] = A_ROT
    E = np.linalg.inv(T @ A4)[:3]
    _fx, fy, _cx, _cy = m["Knew_pinhole"]
    fov_h = float(np.degrees(2.0 * np.arctan((m["output_size"] / 2.0) / fy)))
    return E, fov_h


def _rot_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def _vec_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        # A zero relative translation has no direction. Both zero: agree; one
        # zero: the worst a direction error can be.
        return 0.0 if (na < 1e-9 and nb < 1e-9) else 90.0
    return float(np.degrees(np.arccos(np.clip(float(a @ b) / (na * nb), -1.0, 1.0))))


def window_pose_metrics(pred_E: np.ndarray, gt_E: np.ndarray) -> dict:
    """Relative-pose accuracy of one window, VGGT-style, plus a Sim(3) ATE.

    ``pred_E`` and ``gt_E`` are (S, 3, 4) cam-from-world. Every pair i<j is
    compared through its relative pose E_i E_j^-1, so neither the global frame
    (VGGT anchors camera 0 at the identity) nor the global scale (VGGT's
    translation is up to scale) can enter: RRA is the angle of
    R_ij^pred (R_ij^gt)^T, RTA the angle between the two relative translation
    directions. AUC@30 is the mean accuracy of max(RRA, RTA) over the thresholds
    1..30 deg, the quantity the VGGT paper reports. ATE is the RMSE of the camera
    centres after a Sim(3) alignment to the GT centres, in metres, so it does
    carry the metric scale of the GT trajectory.
    """
    from .metrics import _umeyama_sim3
    S = pred_E.shape[0]

    def to4(E):
        out = np.tile(np.eye(4), (E.shape[0], 1, 1))
        out[:, :3, :] = E
        return out
    P4, G4 = to4(np.asarray(pred_E, np.float64)), to4(np.asarray(gt_E, np.float64))
    rra, rta = [], []
    for i in range(S):
        for j in range(i + 1, S):
            rp = P4[i] @ np.linalg.inv(P4[j])
            rg = G4[i] @ np.linalg.inv(G4[j])
            rra.append(_rot_angle_deg(rp[:3, :3] @ rg[:3, :3].T))
            rta.append(_vec_angle_deg(rp[:3, 3], rg[:3, 3]))
    rra, rta = np.asarray(rra), np.asarray(rta)
    worst = np.maximum(rra, rta)
    out = {"rot_err_deg": float(rra.mean()), "trans_err_deg": float(rta.mean())}
    for th in (5, 15, 30):
        out[f"rra{th}"] = float((rra < th).mean())
        out[f"rta{th}"] = float((rta < th).mean())
    out["auc30"] = float(np.mean([(worst < th).mean() for th in range(1, 31)]))
    cp = -np.einsum("nji,nj->ni", P4[:, :3, :3], P4[:, :3, 3])
    cg = -np.einsum("nji,nj->ni", G4[:, :3, :3], G4[:, :3, 3])
    if float(np.var(cp, axis=0).sum()) < 1e-12:
        # Every predicted centre coincides: no Sim(3) can spread them out. Score
        # it as the GT trajectory's own spread, which is what "no motion
        # recovered" costs, rather than letting a degenerate fit hide it.
        out["ate_m"] = float(np.sqrt(np.mean(np.sum((cg - cg.mean(0)) ** 2, axis=1))))
        out["sim3_scale"] = float("nan")
    else:
        sc, _R, _t, aligned = _umeyama_sim3(cp, cg)
        out["ate_m"] = float(np.sqrt(np.mean(np.sum((aligned - cg) ** 2, axis=1))))
        out["sim3_scale"] = float(sc)
    out["n_pairs"] = int(len(rra))
    return out


def find_frames(root: str, sequences: Optional[List[str]] = None) -> List[str]:
    """Every frame dir under the render root, sorted by (sequence, frame index)."""
    dirs = sorted(glob.glob(os.path.join(root, "*", "frame_*")))
    if sequences:
        keep = set(sequences)
        dirs = [d for d in dirs if os.path.basename(os.path.dirname(d)) in keep]
    return [d for d in dirs if os.path.isfile(os.path.join(d, "meta.json"))]


class RenderedWindowDataset(Dataset):
    """Windows of rendered frames for one setting.

    Windows never span sequences. When a manifest with explicit window ids is
    present it is honoured, because the renderer chose those groupings for their
    inter-frame baseline; otherwise frames are chunked in order.
    """

    def __init__(self, root: str, setting: str, seq_len: int = 1,
                 sequences: Optional[List[str]] = None,
                 manifest: Optional[str] = None, region: str = "own") -> None:
        parse_setting(setting)          # raises on an unknown setting
        if region not in REGIONS:
            raise ValueError(f"region must be one of {REGIONS}, got {region!r}")
        self.setting, self.seq_len, self.root, self.region = setting, seq_len, root, region
        frames = find_frames(root, sequences)
        if not frames:
            raise SystemExit(f"[exp_rendered] no rendered frames under {root!r}")

        groups: "OrderedDict[str, List[str]]" = OrderedDict()
        if manifest and os.path.isfile(manifest):
            man = json.load(open(manifest))
            rows = man["frames"] if isinstance(man, dict) else man
            for r in rows:
                # The manifest stores `dir` RELATIVE to the render root, so it must
                # be joined before testing -- an unjoined relative path silently
                # fails os.path.isdir and yields zero windows.
                d = r.get("dir") or os.path.join(r["sequence"], f"frame_{int(r['frame_idx']):04d}")
                if not os.path.isabs(d):
                    d = os.path.join(root, d)
                if os.path.isdir(d):
                    key = r.get("window_id") or f"{r['sequence']}/w{r.get('window', 0)}"
                    groups.setdefault(key, []).append(d)
        else:
            for d in frames:
                groups.setdefault(os.path.basename(os.path.dirname(d)), []).append(d)

        self.windows: List[List[str]] = []
        # Frame -> group key.  The GROUP (the renderer's own window: consecutive
        # frames of one trajectory segment in one sequence) is the unit of
        # independence, and it does NOT depend on seq_len: at seq_len=1 every eval
        # window is a single frame, but those frames still share a scene, a
        # lighting and a few tenths of a second of trajectory. Resampling them as
        # if independent is what makes a frame-level CI far too narrow.
        self.group_of: Dict[str, str] = {}
        for gkey, ds in groups.items():
            ds = sorted(ds)
            for d in ds:
                self.group_of[d] = gkey
            # Non-overlapping chunks so every frame is scored exactly once.
            for i in range(0, len(ds) - seq_len + 1, seq_len):
                self.windows.append(ds[i:i + seq_len])
        if not self.windows:
            raise SystemExit(
                f"[exp_rendered] no windows of length {seq_len} from "
                f"{len(frames)} frames / {len(groups)} groups. If a manifest was "
                f"given, check its `dir` fields resolve under --render-root.")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> dict:
        imgs, deps, masks = [], [], []
        proj, arm = parse_setting(self.setting)
        for d in self.windows[i]:
            # The arm's OWN valid mask: what the fill treats as the hole.
            vm = np.load(os.path.join(d, f"{grading_mask_of(self.setting)}.npy")).astype(bool)
            src = {"full": "full", "crop": "crop"}.get(arm, "masked")
            rgb = np.load(os.path.join(d, f"{proj}_{src}_rgb.npy"))
            if rgb.dtype == np.uint8:
                rgb = rgb.astype(np.float32) / 255.0
            dep = np.load(os.path.join(d, f"{proj}_{src}_depth.npy")).astype(np.float32)
            if arm not in ("full", "black", "crop"):
                # The masked arm already holds zeros in the hole; fill it. The
                # grading mask is untouched, so filled pixels are never scored.
                rgb = apply_fill(rgb, vm, arm)
            # The GRADING mask: the own mask, or -- region="crop" -- the own mask
            # restricted to the inscribed crop's footprint on this grid, so every
            # arm is scored on the same scene directions (the smallest region,
            # the crop's). The crop's footprint of itself is the whole frame.
            gm = vm
            if self.region == "crop" and arm != "crop":
                gm = gm & np.load(os.path.join(d, f"mask_{proj}_in_crop.npy")).astype(bool)
            imgs.append(torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1))
            deps.append(torch.from_numpy(dep))
            # Grade only where the analytic mask AND the GT agree -- the full arm's
            # depth exists outside the mask, and must not be scored there.
            masks.append(torch.from_numpy(gm & (dep > 0)))
        return {"images": torch.stack(imgs), "depths": torch.stack(deps),
                "valid_masks": torch.stack(masks), "dirs": self.windows[i]}


@torch.no_grad()
def evaluate(model, root: str, setting: str, seq_len: int, device: torch.device,
             sequences: Optional[List[str]] = None, manifest: Optional[str] = None,
             align: str = "scale_shift", qual_dir: Optional[str] = None,
             region: str = "own") -> dict:
    from vggt_omega.utils.rotation import quat_to_mat
    ds = RenderedWindowDataset(root, setting, seq_len, sequences, manifest, region)
    proj, _arm = parse_setting(setting)
    per_frame, fovs = [], []
    windows: "OrderedDict[str, dict]" = OrderedDict()
    for wi in range(len(ds)):
        s = ds[wi]
        preds = model(s["images"].unsqueeze(0).to(device))
        dp = preds["depth"]
        if dp.ndim == 5:
            dp = dp.squeeze(-1)
        dp = dp[0].float().cpu().numpy()
        pe = preds.get("pose_enc")
        pred_E, fov_deg = None, None
        if pe is not None:
            pe = pe[0].float().cpu()
            fov_deg = np.degrees(pe[:, 7:9].numpy())
            fovs.append(fov_deg)
            R = quat_to_mat(pe[:, 3:7]).numpy()
            pred_E = np.concatenate([R, pe[:, :3].numpy()[:, :, None]], -1)
        gt_cams = [gt_camera(d) for d in s["dirs"]]
        gt_E = np.stack([c[0] for c in gt_cams])
        # The pinhole FoV is only a ground truth for the pinhole arms.
        gt_fov = gt_cams[0][1] if proj == "persp" else float("nan")
        wkey = s["dirs"][0]
        wrec = {"dirs": list(s["dirs"]), "gt_E": gt_E.tolist(),
                "group": ds.group_of.get(wkey, wkey), "gt_fov_h_deg": gt_fov}
        if pred_E is not None:
            wrec["pred_E"] = pred_E.tolist()
            wrec["fov_deg"] = fov_deg.tolist()
            if seq_len > 1:
                wrec.update(window_pose_metrics(pred_E, gt_E))
        windows[wkey] = wrec
        for fi in range(dp.shape[0]):
            gt = s["depths"][fi].numpy()
            m = s["valid_masks"][fi].numpy()
            if m.sum() < 100:
                continue
            pa = align_depth(dp[fi], gt, m, mode=align)
            met = depth_metrics(pa, gt, m)
            # Key on the frame dir so the two arms of a projection can be PAIRED:
            # they see the same scene, so an unpaired test throws away the variance
            # that the pairing removes and badly understates significance.
            met["_dir"] = s["dirs"][fi]
            if fov_deg is not None:
                met["_fov"] = [float(fov_deg[fi, 0]), float(fov_deg[fi, 1])]
            per_frame.append(met)
            if qual_dir and fi == 0:
                if wi == 0:
                    _save_qual(qual_dir, setting, seq_len, s["images"][fi].numpy(),
                               pa, gt, m)
                # Raw panels for every window of the four main arms (the fill
                # arms only for window 0): the figures are composed offline, and
                # one window is an anecdote.
                if setting in SETTINGS or wi == 0:
                    _save_qual_raw(qual_dir, setting, seq_len, wi,
                                   s["images"][fi].numpy(), pa, gt, m, s["dirs"][fi])
    if not per_frame:
        return {}
    out = {k: float(np.mean([f[k] for f in per_frame]))
           for k in per_frame[0]
           if not k.startswith("_") and isinstance(per_frame[0][k], (int, float))}
    out["n_frames"] = len(per_frame)
    out["n_windows"] = len(ds)
    out["_per_frame"] = {f["_dir"]: float(f["AbsRel"]) for f in per_frame}
    out["_per_frame_metrics"] = {
        f["_dir"]: {k: float(v) for k, v in f.items()
                    if not k.startswith("_") and isinstance(v, (int, float))}
        for f in per_frame}
    out["_group_of"] = {f["_dir"]: ds.group_of.get(f["_dir"], f["_dir"]) for f in per_frame}
    out["n_groups"] = len(set(out["_group_of"].values()))
    out["_windows"] = windows
    if fovs:
        f = np.concatenate(fovs, 0)
        out["fov_h_deg"] = float(f[:, 0].mean())
        out["fov_h_std"] = float(f[:, 0].std())
        out["fov_w_deg"] = float(f[:, 1].mean())
        out["_per_frame_fov"] = {f["_dir"]: f["_fov"] for f in per_frame if "_fov" in f}
        out["gt_fov_h_deg"] = float(windows[next(iter(windows))]["gt_fov_h_deg"])
        if np.isfinite(out["gt_fov_h_deg"]):
            out["fov_h_abs_err_deg"] = float(np.mean(
                [abs(v[0] - out["gt_fov_h_deg"]) for v in out["_per_frame_fov"].values()]))
    if seq_len > 1 and all("auc30" in w for w in windows.values()):
        for k in POSE_KEYS:
            out[k] = float(np.nanmean([w[k] for w in windows.values()]))
        out["_per_window"] = {wk: {k: w[k] for k in POSE_KEYS} for wk, w in windows.items()}
        out["_window_group"] = {wk: w["group"] for wk, w in windows.items()}
    return out


def _save_qual_raw(qual_dir: str, setting: str, seq_len: int, wi: int,
                   img_chw: np.ndarray, pred: np.ndarray, gt: np.ndarray,
                   mask: np.ndarray, frame_dir: str) -> None:
    """Everything a panel needs, uncomposed, so figures can be laid out offline."""
    d = os.path.join(qual_dir, "raw")
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(
        os.path.join(d, f"{setting}_s{seq_len}_w{wi:02d}.npz"),
        rgb=(np.clip(img_chw.transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8),
        pred=pred.astype(np.float16), gt=gt.astype(np.float16), mask=mask,
        frame_dir=np.array(frame_dir), setting=np.array(setting), seq_len=seq_len)


def _save_qual(qual_dir: str, setting: str, seq_len: int, img_chw: np.ndarray,
               pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> None:
    """RGB | pred | GT | error, on one shared depth scale so panels are comparable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(qual_dir, exist_ok=True)
    vmin, vmax = np.percentile(gt[mask], [2, 98])
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    ax[0].imshow(np.clip(img_chw.transpose(1, 2, 0), 0, 1)); ax[0].set_title(setting)
    for a, (d, t) in zip(ax[1:3], [(pred, "pred (scale-shift)"), (gt, "GT (rendered)")]):
        a.imshow(np.where(mask, d, np.nan), vmin=vmin, vmax=vmax, cmap="turbo"); a.set_title(t)
    err = np.where(mask, np.abs(pred - gt) / np.maximum(gt, 1e-6), np.nan)
    im = ax[3].imshow(err, vmin=0, vmax=0.3, cmap="magma"); ax[3].set_title("AbsRel")
    fig.colorbar(im, ax=ax[3], fraction=0.046)
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{setting}  ·  {'single-frame' if seq_len == 1 else f'{seq_len}-frame'}")
    fig.tight_layout()
    fig.savefig(os.path.join(qual_dir, f"{setting}_s{seq_len}.png"), dpi=110)
    plt.close(fig)


def paired_bootstrap(a: Dict[str, float], b: Dict[str, float],
                     n_boot: int = 10000, seed: int = 0) -> Optional[dict]:
    """Bootstrap CI for mean(a - b) over frames present in BOTH arms.

    Paired, because the two arms are the same scenes: resampling frames
    independently would reintroduce the between-scene variance that pairing
    removes, and understate significance. Returns None if the arms share no
    frames (which would itself mean the comparison is not what it claims).
    """
    keys = sorted(set(a) & set(b))
    if len(keys) < 3:
        return None
    d = np.array([a[k] - b[k] for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n_pairs": len(d), "mean": float(d.mean()),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def cluster_bootstrap(a: Dict[str, float], b: Dict[str, float],
                      group_of: Dict[str, str], n_boot: int = 10000,
                      seed: int = 0) -> Optional[dict]:
    """Paired bootstrap that resamples GROUPS, not frames.

    The frames of one rendered window share a scene, a lighting and a fraction of
    a second of trajectory, so they are not independent draws. A frame-level
    bootstrap treats n frames as n samples and returns an interval that is too
    narrow by roughly sqrt(frames per group). Resampling whole groups -- taking
    every paired frame of each drawn group -- propagates the between-group
    variance instead, which is the variance a new sequence would actually show.

    The estimate is the mean over frames (identical to the frame-level point
    estimate); only the interval differs. Reported alongside the number of
    groups, because with a handful of groups the interval is wide for a real
    reason and quoting it without n_groups invites the same overconfidence the
    frame-level version produced.
    """
    keys = sorted(set(a) & set(b) & set(group_of))
    if len(keys) < 3:
        return None
    by_group: "OrderedDict[str, List[float]]" = OrderedDict()
    for k in keys:
        by_group.setdefault(group_of[k], []).append(a[k] - b[k])
    gkeys = list(by_group)
    if len(gkeys) < 2:
        return None
    vals = [np.asarray(by_group[g], dtype=float) for g in gkeys]
    d_all = np.concatenate(vals)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(gkeys), size=(n_boot, len(gkeys)))
    boots = np.array([np.concatenate([vals[j] for j in row]).mean() for row in draws])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n_pairs": len(d_all), "n_groups": len(gkeys), "mean": float(d_all.mean()),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "frames_per_group": [len(v) for v in vals]}


def report(results: Dict[str, Dict[str, dict]], out_dir: str, region: str = "own") -> str:
    """Per-setting table plus the contrasts that actually answer the question."""
    lines = ["", "=" * 86,
             "VGGT-Omega on rendered ADT — settings x single/multi-frame",
             "(AbsRel lower better; graded on the MASKED arm's region in both arms)"
             if region == "own" else
             "(AbsRel lower better; EVERY arm graded on the inscribed crop's footprint -- "
             "the smallest common region -- intersected with its own valid mask)",
             "=" * 86,
             f"{'setting':<18}{'mode':<8}{'n':>5}{'win':>5}{'AbsRel':>10}{'RMSE':>9}"
             f"{'delta1':>10}{'fov_h':>9}"]
    all_settings = [s for m in results for s in results[m]]
    seen_s = list(dict.fromkeys(all_settings))
    for mode, res in results.items():
        for s in seen_s:
            r = res.get(s) or {}
            if not r:
                continue
            lines.append(f"{s:<18}{mode:<8}{r.get('n_frames',0):>5}"
                         f"{r.get('n_groups',0):>5}"
                         f"{r.get('AbsRel',float('nan')):>10.4f}{r.get('RMSE',float('nan')):>9.3f}"
                         f"{r.get('delta1',float('nan')):>10.4f}{r.get('fov_h_deg',float('nan')):>9.1f}")
    lines.append("")
    for mode, res in results.items():
        def pf(s):
            return (res.get(s) or {}).get("_per_frame", {})
        def go(s):
            return (res.get(s) or {}).get("_group_of", {})
        lines.append(f"[{mode}] true content vs black  (AbsRel full - masked; "
                     f"negative = true content helps):")
        lines.append("    CI(win) resamples WINDOWS and is the one to quote; CI(frm) "
                     "resamples frames and is too narrow (frames in a window share a "
                     "scene) -- it is shown only to make the gap visible.")
        eff = {}
        for proj in ("fisheye", "persp"):
            a_, b_ = pf(f"{proj}_full"), pf(f"{proj}_masked")
            bs = paired_bootstrap(a_, b_)
            if bs is None:
                lines.append(f"    {proj:<8} (no paired frames)")
                continue
            cb = cluster_bootstrap(a_, b_, go(f"{proj}_full") or go(f"{proj}_masked"))
            eff[proj] = cb or bs
            if cb:
                star = "  SIGNIFICANT" if cb["excludes_zero"] else "  n.s. (CI spans 0)"
                lines.append(
                    f"    {proj:<8} {cb['mean']:+.4f}  CI(win) "
                    f"[{cb['ci_lo']:+.4f}, {cb['ci_hi']:+.4f}]  "
                    f"CI(frm) [{bs['ci_lo']:+.4f}, {bs['ci_hi']:+.4f}]  "
                    f"n={cb['n_pairs']}f/{cb['n_groups']}w{star}")
            else:
                star = "  SIGNIFICANT" if bs["excludes_zero"] else "  n.s. (CI spans 0)"
                lines.append(f"    {proj:<8} {bs['mean']:+.4f}  CI(frm) "
                             f"[{bs['ci_lo']:+.4f}, {bs['ci_hi']:+.4f}]  "
                             f"n={bs['n_pairs']} (one window only){star}")
        if len(eff) == 2:
            # Interaction, paired at frame level: (full-masked)_persp - (full-masked)_fisheye
            fk = sorted(set(pf("fisheye_full")) & set(pf("fisheye_masked")))
            pk = sorted(set(pf("persp_full")) & set(pf("persp_masked")))
            common = sorted(set(fk) & set(pk))
            if len(common) >= 3:
                dd = {k: (pf("persp_full")[k] - pf("persp_masked")[k])
                         - (pf("fisheye_full")[k] - pf("fisheye_masked")[k])
                      for k in common}
                d = np.array(list(dd.values()))
                rng = np.random.default_rng(1)
                b = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
                lo, hi = np.percentile(b, [2.5, 97.5])
                gmap = go("persp_full") or go("fisheye_full")
                cb = cluster_bootstrap(dd, {k: 0.0 for k in dd}, gmap)
                if cb:
                    sig = "SIGNIFICANT" if cb["excludes_zero"] else "n.s. (CI spans 0)"
                    lines += [f"    INTERACTION {cb['mean']:+.4f}  CI(win) "
                              f"[{cb['ci_lo']:+.4f}, {cb['ci_hi']:+.4f}]  "
                              f"CI(frm) [{lo:+.4f}, {hi:+.4f}]  "
                              f"n={cb['n_pairs']}f/{cb['n_groups']}w  {sig}",
                              "      negative => true content helps MORE in the "
                              "perspective domain"]
                else:
                    sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s. (CI spans 0)"
                    lines += [f"    INTERACTION {d.mean():+.4f}  CI(frm) "
                              f"[{lo:+.4f}, {hi:+.4f}]  n={len(d)}  {sig} "
                              f"(one window only)",
                              "      negative => true content helps MORE in the "
                              "perspective domain"]
    # Camera pose. The depth question has a scoring rule that keeps the arms
    # honest (same graded pixels); pose needs none, because every quantity is a
    # RELATIVE pose within a window and VGGT's frame and scale conventions cancel.
    for mode, res in results.items():
        pose_settings = [st for st in seen_s if "auc30" in (res.get(st) or {})]
        if not pose_settings:
            continue
        lines += ["", f"[{mode}] camera pose, per window (relative poses over all "
                      f"i<j pairs; RRA/RTA = fraction of pairs under the threshold; "
                      f"ATE after Sim(3) on the centres, metres):",
                  f"    {'setting':<22}{'win':>4}{'rotErr':>8}{'trErr':>8}{'RRA@15':>8}"
                  f"{'RTA@15':>8}{'AUC@30':>8}{'ATE(m)':>8}{'scale':>7}{'fov_h':>7}"]
        for st in pose_settings:
            r = res[st]
            lines.append(f"    {st:<22}{r.get('n_windows',0):>4}"
                         f"{r['rot_err_deg']:>8.2f}{r['trans_err_deg']:>8.2f}"
                         f"{r['rra15']:>8.3f}{r['rta15']:>8.3f}{r['auc30']:>8.3f}"
                         f"{r['ate_m']:>8.3f}{r['sim3_scale']:>7.2f}"
                         f"{r.get('fov_h_deg', float('nan')):>7.1f}")
        lines.append(f"    true content vs black, paired per WINDOW (full - masked; "
                     f"for AUC/RRA/RTA positive = true content helps, for errors "
                     f"negative = it helps):")
        for key in ("auc30", "rot_err_deg", "trans_err_deg", "ate_m"):
            row = f"    {key:<14}"
            eff = {}
            for proj in PROJECTIONS:
                a_ = ((res.get(f"{proj}_full") or {}).get("_per_window") or {})
                b_ = ((res.get(f"{proj}_masked") or {}).get("_per_window") or {})
                wg = ((res.get(f"{proj}_full") or {}).get("_window_group") or {})
                av = {k: v[key] for k, v in a_.items()}
                bv = {k: v[key] for k, v in b_.items()}
                cb = cluster_bootstrap(av, bv, wg)
                if cb is None:
                    row += f"  {proj}: (no paired windows)"
                    continue
                eff[proj] = {k: av[k] - bv[k] for k in set(av) & set(bv)}
                star = "*" if cb["excludes_zero"] else " "
                row += (f"  {proj}: {cb['mean']:+.3f} [{cb['ci_lo']:+.3f},"
                        f"{cb['ci_hi']:+.3f}]{star}")
            if len(eff) == 2:
                common = sorted(set(eff["persp"]) & set(eff["fisheye"]))
                dd = {k: eff["persp"][k] - eff["fisheye"][k] for k in common}
                wg = ((res.get("persp_full") or {}).get("_window_group") or {})
                cb = cluster_bootstrap(dd, {k: 0.0 for k in dd}, wg)
                if cb:
                    star = "*" if cb["excludes_zero"] else " "
                    row += (f"  interaction: {cb['mean']:+.3f} [{cb['ci_lo']:+.3f},"
                            f"{cb['ci_hi']:+.3f}]{star}")
            lines.append(row)
        lines.append("    (* = window-clustered 95% CI excludes zero)")
    # Inferred FoV. Only the pinhole arms have a ground truth to grade against;
    # the fisheye arms' number is reported because it is what the camera head
    # believes, not because anything true corresponds to it.
    for mode, res in results.items():
        fov_settings = [st for st in seen_s if "fov_h_deg" in (res.get(st) or {})]
        if not fov_settings:
            continue
        lines += ["", f"[{mode}] inferred horizontal FoV (deg):",
                  f"    {'setting':<22}{'mean':>8}{'std':>7}{'GT':>8}{'|err|':>8}"]
        for st in fov_settings:
            r = res[st]
            gt = r.get("gt_fov_h_deg", float("nan"))
            err = r.get("fov_h_abs_err_deg", float("nan"))
            lines.append(f"    {st:<22}{r['fov_h_deg']:>8.1f}{r['fov_h_std']:>7.1f}"
                         f"{gt:>8.1f}{err:>8.1f}")

    # The inscribed crop against everything else. It is the free alternative
    # (no black, no fill, 83% of the cone), so the question is whether cropping
    # the black away beats filling it in -- and, in the common-region run,
    # whether either beats the raw fisheye on identical scene directions.
    for mode, res in results.items():
        if "persp_crop" not in res:
            continue
        def pfc(st):
            return (res.get(st) or {}).get("_per_frame") or {}
        gc = (res["persp_crop"].get("_group_of") or {})
        lines += ["", f"[{mode}] inscribed crop (5) vs the others  (AbsRel crop - other; "
                      f"negative = crop better; region = {region}):"]
        for st in ("persp_masked", "persp_full", "fisheye_masked", "fisheye_full"):
            cb = cluster_bootstrap(pfc("persp_crop"), pfc(st), gc)
            if cb is None:
                continue
            star = "  SIGNIFICANT" if cb["excludes_zero"] else "  n.s. (CI spans 0)"
            note = "" if st.startswith("persp") else "  (cross-projection: same directions, different pixel grid)"
            lines.append(f"    vs {st:<16}{cb['mean']:+.4f}  CI(win) [{cb['ci_lo']:+.4f}, "
                         f"{cb['ci_hi']:+.4f}]  n={cb['n_pairs']}f/{cb['n_groups']}w{star}{note}")
        if "_per_window" in res["persp_crop"]:
            wg = res["persp_crop"]["_window_group"]
            for key in ("auc30", "rot_err_deg", "trans_err_deg", "ate_m"):
                row = f"    pose {key:<14}"
                for st in ("persp_masked", "persp_full", "fisheye_masked", "fisheye_full"):
                    a_ = {k: v[key] for k, v in res["persp_crop"]["_per_window"].items()}
                    b_ = {k: v[key] for k, v in ((res.get(st) or {}).get("_per_window") or {}).items()}
                    cb = cluster_bootstrap(a_, b_, wg)
                    if cb is None:
                        continue
                    star = "*" if cb["excludes_zero"] else " "
                    row += f"  vs {st}: {cb['mean']:+.3f} [{cb['ci_lo']:+.3f},{cb['ci_hi']:+.3f}]{star}"
                lines.append(row)
            lines.append("    (pose is relative within a window and does not depend on the grading region)")

    # The fill ladder, if any fill arms were run. The decision-relevant number is
    # not "does true content help" -- that is already answered -- but how much of
    # that gain a fill which INVENTS NOTHING already captures. If a flat mean or a
    # nearest-valid smear recovers most of it, a generative filler is buying the
    # remainder, and the remainder is the budget for the whole idea.
    for mode, res in results.items():
        for proj in PROJECTIONS:
            def pfm(s):
                return (res.get(s) or {}).get("_per_frame") or {}
            gom = ((res.get(f"{proj}_masked") or {}).get("_group_of") or {})
            blk, orc = pfm(f"{proj}_masked"), pfm(f"{proj}_full")
            fills = [s for s in res if s.startswith(f"{proj}_fill_")]
            if not (blk and orc and fills):
                continue
            common = sorted(set(blk) & set(orc))
            span = float(np.mean([blk[k] - orc[k] for k in common]))
            lines += ["", f"[{mode}] {proj} fill ladder — the oracle recovers "
                          f"{span:+.4f} AbsRel over black; how much does each "
                          f"cheap fill capture?",
                      f'    {"fill":<12}{"AbsRel":>9}{"vs black":>10}{"% of oracle":>13}'
                      f'{"CI(win) on the gain":>26}']
            rows = [("black", blk), *[(s[len(proj) + 6:], pfm(s)) for s in sorted(fills)],
                    ("ORACLE", orc)]
            for name, vals in rows:
                ks = sorted(set(vals) & set(blk))
                if not ks:
                    continue
                gain = {k: blk[k] - vals[k] for k in ks}
                g = float(np.mean(list(gain.values())))
                pct = 100.0 * g / span if abs(span) > 1e-9 else float("nan")
                cb = cluster_bootstrap(gain, {k: 0.0 for k in gain}, gom)
                ci = (f'[{cb["ci_lo"]:+.4f}, {cb["ci_hi"]:+.4f}]' if cb else "--")
                lines.append(f'    {name:<12}{np.mean([vals[k] for k in ks]):>9.4f}'
                             f'{g:>+10.4f}{pct:>12.1f}%{ci:>26}')

    # Does multi-frame help, per setting? The modes score the SAME frames, so the
    # comparison is paired; and it is a question the table invites but cannot
    # answer -- fisheye_masked reads WORSE at 8 frames than at 1 in the means,
    # which looks like a mechanism (bad input propagating across views) until the
    # paired interval is computed and straddles zero.
    modes = list(results)
    if len(modes) >= 2:
        base_mode, other_modes = modes[0], modes[1:]
        for mode in other_modes:
            lines += ["", f"does multi-frame help? ({mode} minus {base_mode}, "
                          f"paired per frame, clustered by window; "
                          f"negative = it helps):"]
            for st in seen_s:
                ra = (results[base_mode].get(st) or {})
                rb = (results[mode].get(st) or {})
                a_, b_ = ra.get("_per_frame") or {}, rb.get("_per_frame") or {}
                go = rb.get("_group_of") or ra.get("_group_of") or {}
                common = sorted(set(a_) & set(b_))
                if len(common) < 3:
                    continue
                d = {k: b_[k] - a_[k] for k in common}
                cb = cluster_bootstrap(d, {k: 0.0 for k in d}, go)
                if cb is None:
                    continue
                mark = ("  SIGNIFICANT" if cb["excludes_zero"]
                        else "  n.s. (CI spans 0)")
                lines.append(f'    {st:<18}{cb["mean"]:+.4f}  CI(win) '
                             f'[{cb["ci_lo"]:+.4f}, {cb["ci_hi"]:+.4f}]  '
                             f'n={cb["n_pairs"]}f/{cb["n_groups"]}w{mark}')

    txt = "\n".join(lines)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.txt"), "w") as fh:
        fh.write(txt + "\n")
    return txt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render-root", required=True)
    ap.add_argument("--vggt-checkpoint", required=True)
    ap.add_argument("--out", default="runs/exp_rendered")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--sequences", default="", help="comma-separated; default all")
    ap.add_argument("--seq-lens", default="1,8", help="single- and multi-frame modes")
    ap.add_argument("--settings", default=",".join(SETTINGS),
                    help="comma-separated; '<proj>_full', '<proj>_masked', or "
                         "'<proj>_fill_<mode>' for the fill ladder")
    ap.add_argument("--align", default="scale_shift")
    ap.add_argument("--region", default="own", choices=REGIONS,
                    help="own: each arm on its own valid mask (the masked arm's for a "
                         "full/masked pair); crop: every arm restricted to the "
                         "inscribed crop's footprint, the smallest common region")
    args = ap.parse_args()

    settings = [x.strip() for x in args.settings.split(",") if x.strip()]
    for s in settings:
        parse_setting(s)                # fail before loading a 1B model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seqs = args.sequences.split(",") if args.sequences else None
    os.makedirs(args.out, exist_ok=True)

    from .run_eval import _load_vggt_base
    model = _load_vggt_base(args.vggt_checkpoint, device)
    model.eval()

    results: Dict[str, Dict[str, dict]] = OrderedDict()
    for sl in [int(x) for x in args.seq_lens.split(",")]:
        mode = "single" if sl == 1 else f"{sl}-frame"
        results[mode] = {}
        for s in settings:
            print(f"\n[exp_rendered] {s}  ·  {mode}")
            results[mode][s] = evaluate(
                model, args.render_root, s, sl, device, seqs,
                args.manifest or None, args.align,
                qual_dir=os.path.join(args.out, "qual"), region=args.region)
            print("   ", {k: round(v, 4) for k, v in results[mode][s].items()
                          if isinstance(v, float)})
            with open(os.path.join(args.out, "results.json"), "w") as fh:
                json.dump(results, fh, indent=2, default=str)
    print(report(results, args.out, args.region))


if __name__ == "__main__":
    main()
