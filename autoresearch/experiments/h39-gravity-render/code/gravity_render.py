"""H39 - fold the REAL gravity vector into the rendering warp. Does it pay?

H38 showed that rotating the picture costs more than the roll it removes,
and pointed at the alternative: render the view gravity-aligned in the first
place, so no border is ever created. That was an inference from the structure
of H38, not a measurement - `raw@0` there is a view aligned to the DEVICE, and
device and gravity coincide only when the head is level.

This measures it on real frames with their real head roll, read from the MPS
closed-loop trajectory exactly as H17.1 read it (roll_distribution.rolls_deg).

Three arms per frame:

    device   the view rendered at roll_deg 0        (what the pipeline does today)
    grav_p   the view rendered at roll_deg +psi
    grav_m   the view rendered at roll_deg -psi

BOTH signs are rendered on purpose. The convention chain from
`rolls_deg`'s atan2, through `RolledView.rotation`, to what `RT.Rig` does with
it, is ambiguous enough that asserting a sign would be guessing. One of the two
leaves a residual roll of zero and the other leaves 2*psi, so the identification
is made by the SHAPE of the result - one arm better than `device`, the other
worse by roughly what the h16 curve charges for a 2*psi roll - and not by
picking whichever number is smaller.

The two arms cover different parts of the cone, because rolling a square view
moves its corners. Every frame is therefore scored on the INTERSECTION of the
three arms' coverage, computed per frame.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[3]))
sys.path.insert(0, str(_HERE.parents[1] / "h1-rim-pose-value" / "code"))
sys.path.append(str(_HERE.parents[1] / "h5-rim-finetune" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "common"))
sys.path.insert(0, str(_HERE.parents[1] / "h14-rect-distill" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "h16-orientation" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "h17-roll-prior" / "code"))

import importlib.util as _ilu  # noqa: E402
import upright as U  # noqa: E402
import rect_teacher as RT  # noqa: E402
import roll_controls as RC  # noqa: E402
import roll_distribution as RD  # noqa: E402


def _load(name, path):
    sp = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(sp); sp.loader.exec_module(m); return m


_H5 = _HERE.parents[1] / "h5-rim-finetune" / "code"
Seq = _load("h5_train", _H5 / "train.py").Seq
_ev = _load("h5_eval", _H5 / "eval_lora.py")
THETA_BINS, EDG = _ev.THETA_BINS, _ev.GT_DEPTH_EDGES
from finetune.eval.metrics import align_depth  # noqa: E402

ARMS = ("device", "grav_p", "grav_m")


def frame_rolls(seq_dir: Path, calib: Path, offset_deg: float, stems):
    """Per-frame roll in degrees, joined to the stored frames by timestamp.

    Both conventions here are copied from `roll_distribution.main`, not
    re-derived: the device->camera rotation is TRANSPOSED, and the quarter-turn
    offset is SUBTRACTED. Getting either wrong moves the whole distribution by
    tens or hundreds of degrees -- the first run of this experiment reported a
    median |roll| of 168 deg against H17.1's 2.7 and was discarded.

    Returns (roll_deg, join_error_ms). Frames whose nearest trajectory sample is
    further than one video frame away are the caller's problem to drop; the
    depth frames start about a second before the trajectory does.
    """
    ts, R_wd, g_w = RD.read_trajectory(seq_dir / "groundtruth" / "aria_trajectory.csv")
    cal = json.loads(Path(calib).read_text())
    qx, qy, qz, qw = cal["T_device_camera"]["quaternion_xyzw"]
    R_cd = RD.quat_to_R(qx, qy, qz, qw).T            # camera <- device
    r = RD.wrap180(RD.rolls_deg(R_wd, g_w, R_cd) - offset_deg)
    out, dt = [], []
    for st in stems:
        m = re.search(r"_(\d+)$", st)
        t_us = float(m.group(1)) / 1000.0     # stems carry nanoseconds
        i = int(np.argmin(np.abs(ts - t_us)))
        out.append(float(r[i])); dt.append(abs(float(ts[i] - t_us)) / 1000.0)
    return np.array(out), np.array(dt)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--offset-deg", type=float, default=270.0,
                   help="quarter-turn offset chosen by H17.1 (UPRIGHT_K=3)")
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--models", default="da3:small,vggt_omega")
    p.add_argument("--omega-ckpt", default="checkpoints/VGGT-Omega-1B-512/model.pt")
    p.add_argument("--view-fov", type=float, default=89.0)
    p.add_argument("--view-size", type=int, default=630)
    p.add_argument("--common-theta-deg", type=float, default=44.0)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-join-ms", type=float, default=33.0,
                   help="drop frames whose nearest trajectory sample is further "
                        "away than this; one video frame at 30 Hz is 33 ms")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    s = Seq(a.seq, a.size, a.max_frames)
    cam = s.src.camera
    theta = cam.incidence_grid(a.size, a.size)
    cos_t = torch.cos(theta)
    theta_np = theta.numpy()
    cone = (theta <= cam.theta_max).numpy()
    t_edges = np.linspace(0.0, float(cam.theta_max), THETA_BINS + 1)
    t_idx = np.clip(np.digitize(theta_np, t_edges) - 1, 0, THETA_BINS - 1)
    t_mid = 0.5 * (t_edges[:-1] + t_edges[1:]) * 180 / np.pi
    NB = len(EDG) - 1
    common = cone & (np.rad2deg(theta_np) <= a.common_theta_deg)
    gts = {f: s.gt_range(f, cos_t).numpy() for f in s.frames}

    psi_all, dt_all = frame_rolls(Path(a.seq), Path(a.calib), a.offset_deg,
                                  [s.stem(f) for f in s.frames])
    # One video frame at 30 Hz is 33 ms. Anything further from a trajectory
    # sample than that is not a roll measurement for THIS frame, so it is
    # dropped rather than used.
    keep = dt_all <= a.max_join_ms
    if not keep.all():
        print(f"[h39] dropping {int((~keep).sum())} of {len(keep)} frames whose "
              f"nearest trajectory sample is > {a.max_join_ms:g} ms away "
              f"(worst {dt_all.max():.0f} ms)", flush=True)
    s.frames = [f for f, k in zip(s.frames, keep) if k]
    psi = psi_all[keep]
    gts = {f: gts[f] for f in s.frames}
    if len(s.frames) < 10:
        sys.exit(f"[h39] only {len(s.frames)} frames survive the timestamp join")
    print(f"[h39] {s.name}: {len(s.frames)} frames, |roll| median "
          f"{np.median(np.abs(psi)):.2f} deg, p90 {np.percentile(np.abs(psi), 90):.2f}, "
          f"max {np.abs(psi).max():.2f}; worst join {dt_all[keep].max():.1f} ms", flush=True)

    from raytun3r.backbones import build_backbone

    def score_frames(pred_by_frame, mask_by_frame):
        s_ = np.zeros((THETA_BINS, NB)); n_ = np.zeros((THETA_BINS, NB))
        for f, d in pred_by_frame.items():
            gt = gts[f]
            v = mask_by_frame[f] & (gt > 0) & (gt <= a.depth_max_m) & (d > 1e-6)
            if v.sum() < 1000:
                continue
            al = align_depth(d, gt, v, mode="scale_shift")
            ar = (np.abs(al - gt) / np.clip(gt, 1e-6, None))[v]
            di = np.clip(np.digitize(gt[v], EDG) - 1, 0, NB - 1)
            flat = t_idx[v] * NB + di
            s_ += np.bincount(flat, weights=ar, minlength=THETA_BINS * NB).reshape(THETA_BINS, NB)
            n_ += np.bincount(flat, minlength=THETA_BINS * NB).reshape(THETA_BINS, NB)
        tab = np.divide(s_, n_, out=np.zeros_like(s_), where=n_ > 0)
        out = {}
        for nm, keep in (("near_rim", lambda i, j: t_mid[i] >= 38 and EDG[j + 1] <= 2.0),
                         ("center", lambda i, j: t_mid[i] <= 11),
                         ("all", lambda i, j: True)):
            cells = [(i, j) for i in range(THETA_BINS) for j in range(NB) if keep(i, j)]
            w = np.array([n_[i, j] for i, j in cells], float)
            out[nm] = float((np.array([tab[i, j] for i, j in cells]) * w).sum() / max(w.sum(), 1.0))
        return out

    def per_frame_absrel(d, gt, v):
        if v.sum() < 1000:
            return None
        al = align_depth(d, gt, v, mode="scale_shift")
        return float(np.mean(np.abs(al - gt)[v] / np.clip(gt, 1e-6, None)[v]))

    all_models = {}
    for spec in [x.strip() for x in a.models.split(",") if x.strip()]:
        name, _, variant = spec.partition(":")
        kw = {"variant": variant} if variant else {}
        w8 = a.omega_ckpt if name == "vggt_omega" else "pretrained"
        try:
            bb = build_backbone(name, weights=w8, device=a.device, **kw)
        except Exception as exc:
            print(f"[h39] {spec:14s} unavailable: {exc.__class__.__name__}: {exc}")
            continue
        ps = bb.patch_size
        vs = int(round(a.view_size / ps)) * ps
        rig0 = RT.Rig(cam, [RC.RolledView(fov_x_deg=a.view_fov, width=vs,
                                          height=vs, roll_deg=0.0)], patch=ps)
        bb.install(None, rig0.views[0].pin, (vs, vs), patch_undistort=False,
                   border_token=False, dpt_grid=False, depth_convention="z")
        print(f"[h39] {spec}: patch {ps}, view {vs}x{vs}", flush=True)

        preds = {k: {} for k in ARMS}
        masks = {}
        for j, f in enumerate(s.frames):
            src = s.src.image(f).to(a.device)
            cov_all = None
            for arm in ARMS:
                deg = 0.0 if arm == "device" else (psi[j] if arm == "grav_p" else -psi[j])
                rig = RT.Rig(cam, [RC.RolledView(fov_x_deg=a.view_fov, width=vs,
                                                 height=vs, roll_deg=float(deg))],
                             patch=ps)
                with torch.no_grad():
                    d, _ = rig.teach(lambda w, _v: U.forward_z(bb, w), src, align=False)
                cv = rig.covered.numpy()
                preds[arm][f] = np.where(cv, d.float().cpu().numpy(), 0.0)
                cov_all = cv if cov_all is None else (cov_all & cv)
            masks[f] = cone & cov_all & common

        R = {arm: score_frames(preds[arm], masks) for arm in ARMS}
        pf = {arm: [] for arm in ARMS}
        for f in s.frames:
            gt = gts[f]
            for arm in ARMS:
                v = masks[f] & (gt > 0) & (gt <= a.depth_max_m) & (preds[arm][f] > 1e-6)
                pf[arm].append(per_frame_absrel(preds[arm][f], gt, v))
        all_models[spec] = {"pooled": R, "per_frame": pf}
        for arm in ARMS:
            print(f"  {arm:<8s} all {R[arm]['all']:.4f}   rim {R[arm]['near_rim']:.4f}",
                  flush=True)
        print(f"  grav_p vs device {100*(R['grav_p']['all']/R['device']['all']-1):+.2f}%   "
              f"grav_m vs device {100*(R['grav_m']['all']/R['device']['all']-1):+.2f}%",
              flush=True)
        del bb
        torch.cuda.empty_cache()

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"seq": s.name, "frames": [int(f) for f in s.frames],
             "roll_deg": [float(x) for x in psi],
             "join_worst_ms": float(dt_all[keep].max()),
             "frames_dropped": int((~keep).sum()),
             "models": all_models, "config": vars(a)}, indent=1))
        print(f"[h39] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
