"""CPU smoke test for the Fisheye3R reproduction.

Runs a tiny (embed_dim=128) randomly-initialized VGGT-Omega through every
code path: KB distortion round-trip, token-adapted forward (pure fisheye,
pure perspective, hybrid), init-time equivalence with the frozen baseline,
one SSL optimization step, metric functions, and token save/load.

    python fisheye3r/smoke_test.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggt_omega.models.vggt_omega import VGGTOmega

from fisheye3r.distortion import distort_images, sample_kb_cameras, undistort_dense
from fisheye3r.losses import scheme_loss
from fisheye3r.model import Fisheye3R
from fisheye3r.eval import depth_metrics, pose_metrics

torch.manual_seed(0)


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "ok " if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        raise SystemExit(1)


# --------------------------------------------------- 1. distortion round-trip
B, H, W = 2, 64, 64
cam = sample_kb_cameras(B, H, W, k123_range=(-0.15, 0.15), k4_range=(-0.01, 0.01))
ramp = torch.linspace(0, 1, W).view(1, 1, 1, W).expand(B, 3, H, W).clone()
ramp[:, 1] = torch.linspace(0, 1, H).view(1, H, 1)
fish, fish_valid = distort_images(ramp, cam)
undone, undo_valid = undistort_dense(fish, cam)
err = (undone - ramp).abs()[undo_valid.unsqueeze(1).expand_as(ramp)]
check("distortion round-trip T^-1(T(x)) ~ x", float(err.mean()) < 0.02, f"MAE={err.mean():.4f}")
check("round-trip valid region non-trivial", 0.2 < undo_valid.float().mean() < 1.0,
      f"valid={undo_valid.float().mean():.2f}")

# ------------------------------------------------------------- 2. tiny model
base = VGGTOmega(embed_dim=128)
# A from-scratch VGGT-Omega carries torch.empty()-initialized tensors that a
# real checkpoint would fill (LayerScale.gamma, bias_mask NaN sentinels, ...);
# sanitize them so the random tiny model is numerically valid.
from vggt_omega.models.layers.layer_scale import LayerScale

with torch.no_grad():
    for mod in base.modules():
        if isinstance(mod, LayerScale):
            mod.reset_parameters()
    for t in list(base.parameters()) + list(base.buffers()):
        if t.dtype.is_floating_point and not torch.isfinite(t).all():
            t.normal_(0, 0.02)
    for mod in base.modules():
        if hasattr(mod, "bias_mask"):
            mod.bias_mask.fill_(1.0)
base.eval().requires_grad_(False)

model = Fisheye3R(base, num_tokens=2, encoder_skip_layers=12)
n_params = model.num_trainable_parameters()
expected = (24 - 12 + 24 + 24) * 2 * 128
check("trainable params = tokens only", n_params == expected, f"{n_params:,}")

images = torch.rand(1, 2, 3, H, W)
with torch.no_grad():
    on = model(images, fisheye_flags=torch.ones(1, 2, dtype=torch.bool))
    off = model(images, fisheye_flags=torch.zeros(1, 2, dtype=torch.bool))
    hybrid = model(images, fisheye_flags=torch.tensor([[True, False]]))

check("depth shape", tuple(on["depth"].shape) == (1, 2, H, W), str(tuple(on["depth"].shape)))
check("pose_enc shape", tuple(on["pose_enc"].shape) == (1, 2, 9))
diff = (on["depth"] - off["depth"]).abs().max()
check("init tokens ~= frozen baseline", float(diff) < 1e-2, f"max depth diff={diff:.2e}")
check("hybrid (two-pass masked attention) runs", torch.isfinite(hybrid["depth"]).all().item())

# ------------------------------------------------------------- 3. SSL step
flags = torch.ones(1, 2, dtype=torch.bool)
cam2 = sample_kb_cameras(1, H, W, k123_range=(-0.15, 0.15), k4_range=(-0.01, 0.01))
from fisheye3r.train import repeat_cam

cam2 = repeat_cam(cam2, 2)
fish2, _ = distort_images(images.view(2, 3, H, W), cam2)
student = model(fish2.view(1, 2, 3, H, W), fisheye_flags=flags)
with torch.no_grad():
    teacher = model(images, fisheye_flags=torch.zeros_like(flags))
loss, logs = scheme_loss(
    student, teacher["depth"], teacher["pose_enc"], cam=cam2, flags=flags,
    teacher_conf=teacher["depth_conf"],
)
loss.backward()
grads = [p.grad for p in model.trainable_parameters()]
check("SSL loss finite", torch.isfinite(loss).item(), f"loss={float(loss):.4f} {logs}")
check("token grads flow", all(g is not None and g.abs().sum() > 0 for g in grads))
check("backbone stays frozen", all(p.grad is None for p in model.base.parameters()))

# ------------------------------------------------------------- 4. metrics
extr = torch.eye(4)[:3].unsqueeze(0).repeat(4, 1, 1)
extr[:, :3, 3] = torch.randn(4, 3)
pm = pose_metrics(extr, extr.clone())
check("pose metrics on identical poses", pm["pose/RRA@30"] == 1.0 and pm["pose/ATE"] < 1e-4, str(pm))
gt = torch.rand(1, 2, H, W) * 5 + 0.5
dm = depth_metrics(gt * 2.0 + 1.0, gt, torch.ones_like(gt, dtype=torch.bool))
check("depth metrics undo scale+shift", dm["depth/AbsRel"] < 1e-5, str(dm))

# ------------------------------------------------------- 5. save/load tokens
with tempfile.TemporaryDirectory() as td:
    p = str(Path(td) / "tok.pt")
    model.save_tokens(p)
    model2 = Fisheye3R(base, num_tokens=2, encoder_skip_layers=12)
    model2.load_tokens(p)
    same = all(
        torch.equal(a, b) for a, b in zip(model.trainable_parameters(), model2.trainable_parameters())
    )
    check("token save/load round-trip", same)

print("\nall smoke tests passed")
