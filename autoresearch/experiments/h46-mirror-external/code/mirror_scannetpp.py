"""H46 - the mirror test on ScanNet++, with no rig and no warp.

The model is shown the captured frame itself. Three arms per frame:

    normal   the frame as captured
    mirror   flip the frame, predict, flip the planar z back
    twice    flip and unflip before predicting -- THE IDENTITY

`twice` must equal `normal` exactly. It is the bar that turned H44's void into
H45's result and it travels with the method: if it does not hold, nothing here
is a measurement.

Ground truth and prediction are both PLANAR Z (`depth_convention="z"` on both
the loader and the backbone), so nothing is converted on either side.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[3]))
sys.path.insert(0, str(_HERE.parents[1] / "common"))
sys.path.insert(0, str(_HERE.parents[1] / "h14-rect-distill" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "h16-orientation" / "code"))

import upright as U  # noqa: E402
import rect_teacher as RT  # noqa: E402
import roll_controls as RC  # noqa: E402
from raytun3r.data import ScanNetPPFisheye  # noqa: E402
from finetune.eval.metrics import align_depth  # noqa: E402

ARMS = ("normal", "mirror", "twice")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--models", default="da3:small,da3:large,vggt,vggt_omega")
    p.add_argument("--omega-ckpt", default="checkpoints/VGGT-Omega-1B-512/model.pt")
    p.add_argument("--view-fov", type=float, default=0.0,
                   help="0 = feed the captured frame. >0 = rectify a co-axial "
                        "pinhole view of this FOV first, which is what H45 did "
                        "on Aria. A square 89 deg view has a 54.2 deg corner "
                        "ray, inside this lens's 84.8 deg cone, so nothing is "
                        "black and the two experiments become like-for-like.")
    p.add_argument("--view-size", type=int, default=630)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    from raytun3r.backbones import build_backbone

    out = {}
    for spec in [x.strip() for x in a.models.split(",") if x.strip()]:
        name, _, variant = spec.partition(":")
        kw = {"variant": variant} if variant else {}
        w8 = a.omega_ckpt if name == "vggt_omega" else "pretrained"
        try:
            bb = build_backbone(name, weights=w8, device=a.device, **kw)
        except Exception as exc:
            print(f"[h46] {spec:14s} unavailable: {exc.__class__.__name__}: {exc}")
            continue
        ps = bb.patch_size
        src = ScanNetPPFisheye(a.scene, max_size=a.size, patch=ps,
                               max_frames=a.max_frames, depth_convention="z")
        cam = src.camera
        H, W = src.h, src.w
        cone = cam.valid_mask(H, W).numpy() if hasattr(cam, "valid_mask") else \
            np.ones((H, W), bool)
        rig = None
        if a.view_fov > 0:
            vs = int(round(a.view_size / ps)) * ps
            rig = RT.Rig(cam, [RC.RolledView(fov_x_deg=a.view_fov, width=vs,
                                             height=vs, roll_deg=0.0)], patch=ps)
            bb.install(None, rig.views[0].pin, (vs, vs), patch_undistort=False,
                       border_token=False, dpt_grid=False, depth_convention="z")
            cone = cone & rig.covered.numpy()
        else:
            bb.install(None, cam, (H, W), patch_undistort=False,
                       border_token=False, dpt_grid=False, depth_convention="z")
        print(f"[h46] [{src.name}] {spec}: {len(src)} frames, {W}x{H}, "
              f"{src.n_bad} bad dropped, "
              f"{'rectified %.0f deg view' % a.view_fov if rig else 'raw frame'}",
              flush=True)

        per = {arm: [] for arm in ARMS}
        for i in range(len(src)):
            d = src.depth(i)
            if d is None:
                continue
            gt, ok = (d if isinstance(d, tuple) else (d, None))
            gt = gt.numpy() if torch.is_tensor(gt) else np.asarray(gt)
            valid = cone & (gt > 0) & (gt <= a.depth_max_m)
            if ok is not None:
                valid = valid & (ok.numpy() if torch.is_tensor(ok) else np.asarray(ok))
            if valid.sum() < 1000:
                continue
            img = src.image(i).to(a.device)
            for arm in ARMS:
                def fz(w, _v=None, _a=arm):
                    if _a == "normal":
                        return U.forward_z(bb, w)
                    if _a == "twice":
                        x = torch.flip(torch.flip(w, dims=[-1]), dims=[-1])
                        return torch.flip(torch.flip(U.forward_z(bb, x), dims=[-1]),
                                          dims=[-1])
                    return torch.flip(U.forward_z(bb, torch.flip(w, dims=[-1])),
                                      dims=[-1])
                with torch.no_grad():
                    if rig is None:
                        z = fz(img)
                    else:
                        z, _ = rig.teach(fz, img, align=False)
                pr = z.float().cpu().numpy()
                v = valid & (pr > 1e-6)
                if v.sum() < 1000:
                    per[arm].append(None)
                    continue
                al = align_depth(pr, gt, v, mode="scale_shift")
                per[arm].append(float(np.mean(
                    np.abs(al - gt)[v] / np.clip(gt, 1e-6, None)[v])))

        n = [x for x in per["normal"] if x is not None]
        if not n:
            print(f"[h46] [{src.name}] {spec}: no scorable frames"); del bb; continue
        mirror = [y / x for x, y in zip(per["normal"], per["mirror"])
                  if x not in (None, 0) and y is not None]
        tw = [abs(y / x - 1) for x, y in zip(per["normal"], per["twice"])
              if x not in (None, 0) and y is not None]
        cost = 100 * (float(np.mean(mirror)) - 1)
        tw_max = 100 * max(tw) if tw else float("nan")
        out[spec] = {"n_frames": len(n), "normal_absrel": float(np.mean(n)),
                     "mirror_cost_pct": cost, "plumbing_max_rel_pct": tw_max,
                     "per_frame": per}
        print(f"[h46] [{src.name}] {spec}: normal AbsRel {np.mean(n):.4f} | "
              f"mirror {cost:+.1f}% | plumbing {tw_max:.6f}% "
              f"{'OK' if tw_max < 0.01 else '<-- BROKEN'}", flush=True)
        del bb
        torch.cuda.empty_cache()

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"scene": Path(a.scene).name, "models": out, "config": vars(a)},
            indent=1))
        print(f"[h46] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
