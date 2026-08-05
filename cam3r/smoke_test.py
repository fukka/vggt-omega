"""CPU end-to-end check for the CAM3R reproduction.

Walks every code path once at tiny width: SH ray fitting on the real Aria lens,
a two-view forward, all three losses, an optimizer step that must reduce the
loss, scene-graph pruning, Ray-Aware Global Alignment against known poses,
checkpoint round-trip, and -- when the local ADT sample is present -- a real
two-view item end to end.

    python cam3r/smoke_test.py

Complements ``cam3r/tests/`` (run with pytest); this is the single script that
answers "is the whole thing wired together".
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cam3r.adt import ADTPairDataset, find_adt_sequences
from cam3r.alignment import PairwiseEdge, prune_scene_graph, ray_aware_global_alignment
from cam3r.cameras import aria_214_1_kb4
from cam3r.cli import default_adt_root
from cam3r.data import CurriculumSampler, TwoViewSource, synthesize_view, random_camera_for
from cam3r.geometry import make_se3, random_rotations, relative_pose, se3_inverse, transform_points
from cam3r.losses import cam3r_loss
from cam3r.metrics import mean_average_accuracy, pose_accuracy, relative_pose_error
from cam3r.model import CAM3R, CAM3RConfig
from cam3r.rays import fit_ray_field

torch.manual_seed(0)


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "ok " if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        raise SystemExit(1)


# --------------------------------------------------- 1. ray parameterization
H = W = 96
rays, valid = aria_214_1_kb4(H, W).ray_field(H, W)
fit = fit_ray_field(rays, valid, H, W)
check("SH(deg<=3) represents the Aria KB4 lens",
      fit.mean_deg < 0.25, f"mean {fit.mean_deg:.4f} deg, max {fit.max_deg:.4f} deg")
check("fisheye cone masks the corners",
      0.3 < float(valid.float().mean()) < 0.95, f"valid {float(valid.float().mean()):.2f}")

# --------------------------------------------------------------- 2. forward
cfg = CAM3RConfig(img_size=64, patch_size=16, ray_embed_dim=32, ray_depth=2, ray_heads=2,
                  cv_embed_dim=32, cv_enc_depth=2, cv_dec_embed_dim=24, cv_dec_depth=2,
                  cv_heads=2, cv_dec_heads=2)
model = CAM3R(cfg)
img1, img2 = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
out = model(img1, img2)
check("two-view forward shapes",
      out["points"][0].shape == (1, 3, 64, 64) and out["R"].shape == (1, 3, 3))
check("rays are unit vectors", torch.allclose(out["rays"][0].norm(dim=1), torch.ones(1, 64, 64), atol=1e-4))
check("X = d * r (Eq. 1)",
      torch.allclose(out["points"][0], out["rays"][0] * out["radial"][0].unsqueeze(1), atol=1e-6))
check("R is a rotation",
      torch.allclose(out["R"] @ out["R"].transpose(-1, -2), torch.eye(3).expand(1, 3, 3), atol=1e-4))
check("model size reported", model.num_parameters() > 0, f"{model.num_parameters():,} params")

# ------------------------------------------------------ 3. losses + one step
targets = {
    "rays": [torch.nn.functional.normalize(torch.randn(1, 3, 64, 64), dim=1) for _ in range(2)],
    "points": [torch.randn(1, 3, 64, 64).abs() + 1.0 for _ in range(2)],
    "valid": [torch.ones(1, 64, 64, dtype=torch.bool) for _ in range(2)],
    "R": random_rotations(1, seed=3),
    "t": torch.randn(1, 3),
}
loss, logs = cam3r_loss(out, targets, conf_mode="dust3r")
check("total loss finite", bool(torch.isfinite(loss)), f"{float(loss):.4f}")
check("all three terms present", all(k in logs for k in ("loss/angular", "loss/regr", "loss/pose")))

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
first = float(loss)
for _ in range(12):
    opt.zero_grad()
    out_i = model(img1, img2)
    loss_i, _ = cam3r_loss(out_i, targets, conf_mode="dust3r")
    loss_i.backward()
    opt.step()
check("optimizer reduces the loss", float(loss_i) < first, f"{first:.4f} -> {float(loss_i):.4f}")

# ---------------------------------------------------------- 4. metrics
R_gt, t_gt = random_rotations(8, seed=4), torch.randn(8, 3)
rot, tra = relative_pose_error(R_gt, t_gt, R_gt.clone(), t_gt.clone())
acc = pose_accuracy(rot, tra, thresholds=(15,))
check("metrics are perfect on identical poses",
      acc["RRA@15"] == 1.0 and mean_average_accuracy(rot, tra) == 1.0, str(acc))

# ------------------------------------------- 5. pruning + global alignment
n_views, n_pts = 4, 250
world = torch.randn(n_pts, 3, dtype=torch.float64) * 2 + torch.tensor([0.0, 0.0, 8.0], dtype=torch.float64)
Rw = random_rotations(n_views, seed=5, dtype=torch.float64)
Rw[0] = torch.eye(3, dtype=torch.float64)
tw = torch.randn(n_views, 3, dtype=torch.float64) * 0.4
tw[0] = 0.0
T_wc = make_se3(Rw, tw)
scales = torch.tensor([1.0, 2.0, 0.5, 3.0], dtype=torch.float64)

edges = []
for i in range(n_views):
    for j in range(i + 1, n_views):
        edges.append(PairwiseEdge(
            i=i, j=j,
            points_i=transform_points(se3_inverse(T_wc[i]), world) / scales[i],
            points_j=transform_points(se3_inverse(T_wc[j]), world) / scales[j],
            conf=torch.ones(n_pts, dtype=torch.float64),
            overlap=0.9,
        ))
junk = PairwiseEdge(i=0, j=3, points_i=torch.randn(40, 3, dtype=torch.float64),
                    points_j=torch.randn(40, 3, dtype=torch.float64),
                    conf=torch.ones(40, dtype=torch.float64), overlap=0.01)
kept = prune_scene_graph(edges + [junk], min_overlap=0.2)
check("pruning drops the non-overlapping edge", len(kept) == len(edges), f"{len(kept)}/{len(edges) + 1}")

res = ray_aware_global_alignment(n_views, kept, iters=300)
worst_rot = 0.0
for i in range(n_views):
    for j in range(i + 1, n_views):
        a = relative_pose(se3_inverse(T_wc[i]).unsqueeze(0), se3_inverse(T_wc[j]).unsqueeze(0))
        b = relative_pose(se3_inverse(res.poses[i]).unsqueeze(0), se3_inverse(res.poses[j]).unsqueeze(0))
        worst_rot = max(worst_rot, float(relative_pose_error(b[0], b[1], a[0], a[1])[0]))
check("RAGA recovers known poses", worst_rot < 2.0, f"worst rotation error {worst_rot:.3f} deg")
ratio = res.scales / res.scales[0]
check("RAGA recovers relative scales",
      torch.allclose(ratio, (scales / scales[0]).to(ratio.dtype), rtol=0.05), str(ratio.tolist()))

# --------------------------------------------------- 6. panorama synthesis
pano = torch.rand(3, 64, 128)
depth = torch.full((64, 128), 4.0)
synth = synthesize_view(pano, random_camera_for("kb4", 48, 48), (48, 48), depth=depth)
check("panorama -> fisheye synthesis",
      synth["image"].shape == (3, 48, 48) and torch.allclose(
          synth["points"].norm(dim=0)[synth["valid"]],
          torch.full_like(synth["points"].norm(dim=0)[synth["valid"]], 4.0), atol=1e-3))

sampler = CurriculumSampler(
    [TwoViewSource(range(5), "kb4", supports_heterogeneous=True),
     TwoViewSource(range(5), "erp")], phase=2, seed=0
)
check("phase-2 curriculum emits cross-lens draws",
      any(sampler.sample().heterogeneous for _ in range(100)))

# ------------------------------------------------------- 7. save / load
with tempfile.TemporaryDirectory() as td:
    path = str(Path(td) / "cam3r.pt")
    torch.save({"model": model.state_dict()}, path)
    clone = CAM3R(cfg)
    clone.load_state_dict(torch.load(path, map_location="cpu")["model"])
    clone.eval()
    model.eval()
    with torch.no_grad():
        same = torch.allclose(model(img1, img2)["points"][0], clone(img1, img2)["points"][0], atol=1e-6)
    check("checkpoint round-trip", same)

# ------------------------------------------------------- 8. real ADT sample
adt_root = default_adt_root() or os.path.expanduser("~/Documents/projectaria_tools_adt_data")
seqs = find_adt_sequences(adt_root)
if seqs:
    ds = ADTPairDataset(seqs, resolution=64, max_frames=28,
                        baseline_m=(0.0, 99.0), angle_deg=(0.0, 180.0))
    item = ds[0]
    check("ADT pair loads", item["images"][0].shape == (3, 64, 64), f"{len(ds)} pairs")
    rays0, valid0 = item["rays"][0], item["valid"][0]
    radial = item["points"][0].norm(dim=0)
    rim = valid0 & (rays0[2] < 0.6)
    ratio = float((radial[rim] / item["points"][0][2][rim].clamp(min=1e-6)).median())
    check("ADT GT is radial range, not planar z", ratio > 1.5, f"rim range/z = {ratio:.2f}")
    check("extrinsics provenance recorded",
          ds.extrinsics_source in {"json", "mps", "projectaria_tools", "device-frame-fallback"},
          f"{ds.extrinsics_source} (exact={ds.extrinsics_exact})")

    model.eval()
    with torch.no_grad():
        real = model(item["images"][0].unsqueeze(0), item["images"][1].unsqueeze(0))
    check("forward on a real ADT pair", bool(torch.isfinite(real["points"][0]).all()))
else:
    print(f"[skip] real ADT sample not found under {adt_root}")

print("\nall smoke tests passed")
