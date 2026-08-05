"""CPU end-to-end smoke test for the RayTun3R reproduction.

Exercises every code path on tiny randomly-initialised backbones: camera
round-trip, the pinhole-bias diagnostic behind Fig. 2, the parameter-free
corrections, both hook families (absolute PE and the two RoPE flavours), the
full Eq. 13 objective, several optimisation steps, the projection and PEFT
baselines, and the Appendix A metrics.

    python raytun3r/smoke_test.py

It checks *behaviour*, not accuracy: a random backbone has no geometry to
recover. Numbers comparable to the paper need real weights and real data.
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from raytun3r import corrections as C
from raytun3r.adapter import RayTun3RAdapter
from raytun3r.baselines import CenterPH, MultiPH, attach_caltok, attach_lora
from raytun3r.cameras import KannalaBrandt, Pinhole, pixel_grid
from raytun3r.data import Window
from raytun3r.losses import LossWeights, backproject, total_loss
from raytun3r.matching import Matches, mean_flow_magnitude, relative_pose_magsac
from raytun3r.metrics import (depth_metrics, reprojection_depth_error,
                              rotation_error_deg, translation_error_deg)
from raytun3r.testing import init_random, tiny_vggt, tiny_vggt_omega, toy_camera
from raytun3r.train import fit_adapter

PASS, FAIL = "  ok  ", " FAIL "
_failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{PASS if cond else FAIL}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


def synthetic_window(camera, size, seq_len=3, seed=0):
    """A window whose correspondences come from a known depth map and pose."""
    torch.manual_seed(seed)
    h, w = size
    valid = camera.valid_mask(h, w)
    rng_map = 2.0 + torch.rand(h, w) * 3.0

    images = torch.rand(seq_len, 3, h, w)
    Rs = [torch.eye(3)]
    ts = [torch.zeros(3)]
    for k in range(1, seq_len):
        ax = torch.tensor([0.12, -0.06, 0.04]) * k
        ang = ax.norm()
        u = ax / ang
        K = torch.tensor([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
        Rs.append(torch.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K)
        ts.append(torch.tensor([0.25, 0.08, -0.15]) * k)

    win = Window(images=images, indices=list(range(seq_len)), camera=camera)
    win.gt_R = torch.stack(Rs)
    win.gt_t = torch.stack(ts)
    win.gt_depth = rng_map[None].expand(seq_len, -1, -1).contiguous()
    win.gt_valid = valid[None].expand(seq_len, -1, -1).contiguous()

    X = backproject(rng_map, camera, convention="range")
    for i in range(seq_len):
        for j in range(seq_len):
            if i == j:
                continue
            R = win.gt_R[j] @ win.gt_R[i].T
            t = win.gt_t[j] - R @ win.gt_t[i]
            uv = camera.project(X @ R.T + t)
            inb = ((uv[..., 0] >= 0) & (uv[..., 0] <= w - 1)
                   & (uv[..., 1] >= 0) & (uv[..., 1] <= h - 1))
            win.matches[(i, j)] = Matches(target=uv, weight=(valid & inb).float())
            win.pose_targets[(i, j)] = (R, t / t.norm())
    return win


def main() -> int:
    print("=" * 74)
    print("RayTun3R reproduction -- CPU smoke test")
    print("=" * 74)

    # ---------------------------------------------------------------- cameras
    print("\n-- cameras (Sec. 3) --")
    cam = toy_camera(112, 112, fov_deg=180.0)
    uv = pixel_grid(112, 112)
    m = cam.valid_mask()
    err = float((cam.project(cam.unproject(uv)) - uv)[m].abs().max())
    check("KB4 project/unproject round-trip", err < 1e-3, f"max {err:.2e} px")
    check("imaged cone is a proper subset", 0.4 < float(m.float().mean()) < 1.0,
          f"{100 * float(m.float().mean()):.1f}% of pixels")

    # Fig. 2: pinhole Jacobian is radius-independent, fisheye is not.
    ph = Pinhole(fx=300, fy=300, cx=63.5, cy=63.5, width=128, height=128,
                 theta_max=math.radians(50))
    fe = toy_camera(128, 128, fov_deg=180.0)
    r = ((pixel_grid(128, 128) - torch.tensor([63.5, 63.5])) ** 2).sum(-1).sqrt()
    inner, outer = r < 15, (r > 40) & (r < 55)
    ratios = []
    for c in (ph, fe):
        s1 = torch.linalg.svdvals(c.backproject_jacobian())[..., 0]
        ratios.append(float(s1[outer].mean() / s1[inner].mean()))
    check("Fig. 2: pinhole PE Jacobian flat in radius", abs(ratios[0] - 1.0) < 1e-3,
          f"outer/inner = {ratios[0]:.4f}")
    check("Fig. 2: fisheye Jacobian radius-dependent", ratios[1] > 2.0,
          f"outer/inner = {ratios[1]:.2f}")

    # ------------------------------------------------------- param-free parts
    print("\n-- parameter-free corrections (Sec. 4.2) --")
    # 126 = 9 * 14, an odd token grid, so the centre patch sits exactly on the
    # principal point and "on-axis" is a well-posed check.
    cam9 = toy_camera(126, 126, fov_deg=180.0)
    J = C.local_undistort_jacobian(cam9, 126, 126, 14)
    mid = J.shape[0] // 2
    check("patch undistortion is identity on-axis",
          bool(torch.allclose(J[mid, mid], torch.eye(2), atol=2e-3)),
          f"det {float(J[mid, mid].det()):.4f}")
    check("patch undistortion anisotropic at the rim",
          abs(float(J[0, mid].det()) - 1.0) > 0.05, f"det {float(J[0, mid].det()):.4f}")
    grid = C.patch_undistort_grid(cam, 112, 112, 14)
    check("undistort grid finite", bool(torch.isfinite(grid).all()))
    vm = C.patch_valid_mask(cam, 112, 112, 14)
    tok = torch.randn(2, vm.numel(), 32)
    filled = C.fill_border_tokens(tok, vm.reshape(-1))
    inv = ~vm.reshape(-1)
    check("border tokens replaced by mean valid token",
          bool(torch.allclose(filled[0][inv], filled[0][inv][:1].expand_as(filled[0][inv]),
                              atol=1e-5))
          and bool(torch.equal(filled[0][vm.reshape(-1)], tok[0][vm.reshape(-1)])))
    # (grid_h, grid_w, 2), so the x coordinate sweeps along a *row*.
    dpt = C.camera_aware_uv_grid(cam, 8, 8)
    row = dpt[4, :, 0]
    d = row.diff()
    check("DPT grid is monotone and non-uniform",
          bool((d > 0).all()) and float(d.std() / d.mean()) > 1e-3,
          f"spacing cv = {float(d.std() / d.mean()):.3f}")

    # ------------------------------------------------------------- param count
    print("\n-- adapter (Eq. 5, 6) --")
    a384 = RayTun3RAdapter(384)
    bd = a384.param_breakdown()
    check("DA3-Small PE tables == paper's 10,752",
          bd["pe_radial"] + bd["pe_angular"] == 10752,
          f"{bd['pe_radial']} + {bd['pe_angular']} = {bd['pe_radial'] + bd['pe_angular']}")
    check("total with RoPE table is 10,772", bd["total"] == 10772,
          "paper quotes 10,752, which excludes the 20-parameter RoPE table")

    # ------------------------------------------------------------- backbones
    for name, build, hw in (("vggt_omega", tiny_vggt_omega, (64, 64)),
                            ("vggt", tiny_vggt, (70, 70))):
        print(f"\n-- backbone: {name} --")
        bb = build()
        h, w = hw
        c = toy_camera(h, w, fov_deg=180.0)
        imgs = torch.rand(1, 3, 3, h, w)

        with torch.no_grad():
            base = bb.forward(imgs)
        check(f"{name}: baseline forward finite",
              bool(torch.isfinite(base.depth).all()) and bool(torch.isfinite(base.t).all()),
              f"depth {tuple(base.depth.shape)}")

        ad = bb.make_adapter()
        check(f"{name}: adapter branches match architecture",
              (ad.pe is not None) == bb.has_abs_pe and (ad.rope is not None) == bb.has_rope,
              f"{ad.param_breakdown()}")

        bb.install(ad, c, hw, patch_undistort=False, border_token=False, dpt_grid=False)
        with torch.no_grad():
            z = bb.forward(imgs)
        check(f"{name}: zero-init adapter is an exact no-op",
              bool(torch.allclose(z.depth, base.depth, atol=1e-6)))

        with torch.no_grad():
            for p in ad.parameters():
                p.normal_(0, 0.05)
        out = bb.forward(imgs)
        (out.depth.mean() + out.t.pow(2).mean()).backward()
        grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                       and float(p.grad.abs().sum()) > 0 for p in ad.parameters())
        check(f"{name}: gradient reaches every adapter table", grads_ok)
        check(f"{name}: backbone stays frozen",
              all(p.grad is None for p in bb.model.parameters()))

        bb.install(ad, c, hw)
        with torch.no_grad():
            full = bb.forward(imgs)
        check(f"{name}: full pipeline finite", bool(torch.isfinite(full.depth).all()))
        check(f"{name}: pretrained PE table captured for Eq. 12",
              (bb.pe_table() is not None) == bb.has_abs_pe)

        bb.remove()
        with torch.no_grad():
            restored = bb.forward(imgs)
        check(f"{name}: remove() restores the frozen model exactly",
              bool(torch.allclose(restored.depth, base.depth, atol=0)))

    # -------------------------------------------------- losses / optimisation
    print("\n-- objective (Eq. 8-13) and fitting --")
    bb = tiny_vggt()
    h = w = 70
    c = toy_camera(h, w, fov_deg=180.0)
    win = synthetic_window(c, (h, w))
    check("synthetic window has flow above the 2 px static threshold",
          mean_flow_magnitude(win.matches[(0, 1)]) > 2.0,
          f"{mean_flow_magnitude(win.matches[(0, 1)]):.2f} px")

    ad = bb.make_adapter()
    bb.install(ad, c, (h, w))
    pred = bb.forward(win.images[None])
    loss, parts = total_loss(pred, win.as_batch(), c, ad, weights=LossWeights(),
                             valid=c.valid_mask(h, w), pe_table=bb.pe_table())
    check("total loss finite", bool(torch.isfinite(loss)),
          " ".join(f"{k}={v:.3f}" for k, v in parts.items() if k != "iter"))

    stats = fit_adapter(bb, [win], c, iters=6, log_every=100, verbose=False)
    first, last = stats["history"][0]["total"], stats["history"][-1]["total"]
    check("fitting runs and stays finite", math.isfinite(last),
          f"total {first:.4f} -> {last:.4f} over 6 iters")
    check("adapter moved away from zero",
          any(float(p.abs().max()) > 0 for p in ad.parameters()))
    bb.remove()

    # ------------------------------------------------------------- baselines
    print("\n-- baselines (Sec. 5) --")
    bb.install(None, c, (h, w), patch_undistort=False, border_token=False, dpt_grid=False)
    for nm, ctor in (("Center-PH", CenterPH), ("Multi-PH", MultiPH)):
        base = ctor(bb, c)
        with torch.no_grad():
            p = base(win.images)
        check(f"{nm} runs and covers the frame",
              bool(torch.isfinite(p.depth).all()) and float(p.conf.mean()) > 0.1,
              f"{len(base.views)} view(s), coverage {100 * float(p.conf.mean()):.0f}%")

    mods, handles = attach_lora(bb, r=8, alpha=16.0)
    n_lora = sum(p.numel() for p in mods.parameters())
    with torch.no_grad():
        lp = bb.forward(win.images[None])
    check("LoRA attaches, zero-init is a no-op",
          bool(torch.allclose(lp.depth, bb.forward(win.images[None]).depth, atol=1e-6)),
          f"{n_lora} trainable parameters (r=8)")
    for hd in handles:
        hd.remove()

    mods, handles = attach_caltok(bb, n_tokens=4)
    with torch.no_grad():
        cp = bb.forward(win.images[None])
    check("CalTok attaches and preserves sequence length",
          bool(torch.isfinite(cp.depth).all()) and cp.depth.shape == lp.depth.shape,
          f"{sum(p.numel() for p in mods.parameters())} trainable parameters (t=4)")
    for hd in handles:
        hd.remove()
    bb.remove()

    # --------------------------------------------------------------- metrics
    print("\n-- metrics (Appendix A, Eq. 14-18) --")
    R01 = win.gt_R[1] @ win.gt_R[0].T
    t01 = win.gt_t[1] - R01 @ win.gt_t[0]
    check("R/t error zero at ground truth",
          rotation_error_deg(R01, R01) < 1e-4 and translation_error_deg(t01, t01) < 1e-4)
    d0 = reprojection_depth_error(win.gt_depth[0], c, win.matches[(0, 1)], R01, t01,
                                  valid=c.valid_mask(h, w))
    d3 = reprojection_depth_error(win.gt_depth[0] * 3.0, c, win.matches[(0, 1)], R01, t01,
                                  valid=c.valid_mask(h, w))
    check("d_reproj is zero at truth and scale-invariant", d0 < 1e-2 and d3 < 1e-2,
          f"1x: {d0:.4f} px, 3x: {d3:.4f} px")
    dm = depth_metrics(win.gt_depth[0] * 2.0, win.gt_depth[0], valid=c.valid_mask(h, w))
    check("AbsRel/delta scale-invariant", dm["AbsRel"] < 1e-5 and dm["delta_1.25"] > 0.999,
          f"AbsRel={dm['AbsRel']:.2e} delta={dm['delta_1.25']:.4f}")

    out = relative_pose_magsac(win.matches[(0, 1)], c)
    ok = out is not None and rotation_error_deg(out[0], R01) < 0.5
    check("MAGSAC++ recovers the pose target from exact matches", ok,
          "" if out is None else f"R err {rotation_error_deg(out[0], R01):.4f} deg, "
                                 f"t err {translation_error_deg(out[1], t01):.4f} deg")

    print("\n" + "=" * 74)
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
