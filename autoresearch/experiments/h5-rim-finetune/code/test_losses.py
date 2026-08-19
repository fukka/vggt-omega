"""CPU tests for the H5 losses — geometry-verified, no backbone needed.

Run: <venv>/bin/python autoresearch/experiments/h5-rim-finetune/code/test_losses.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent
                       / "experiments" / "h1-rim-pose-value" / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adt_pose_value import AriaLocalPairs, DEFAULT_SEQ  # noqa: E402
import losses  # noqa: E402


def main() -> None:
    src = AriaLocalPairs(DEFAULT_SEQ, size=252)   # small grid, fast
    cam = src.camera
    theta = cam.incidence_grid(src.h, src.w)

    # 1. weight map: near-rim must outweigh center at the same depth
    gt = torch.full((src.h, src.w), 0.7)
    w = losses.compression_weight_map(theta, gt)
    c = w[src.h // 2, src.w // 2]
    rim_mask = theta > math.radians(48)
    assert w[rim_mask].mean() > c + 0.3, (w[rim_mask].mean(), c)
    print(f"ok  weight map: center {float(c):.2f} < rim "
          f"{float(w[rim_mask].mean()):.2f} at 0.7m")

    # 2. warp round-trip: identity pose maps every pixel to itself
    rng = torch.full((src.h, src.w), 2.0)
    uv_j, r_j, front = losses.warp_i_to_j(rng, cam, torch.eye(3),
                                          torch.zeros(3))
    ys, xs = torch.meshgrid(torch.arange(src.h, dtype=torch.float32),
                            torch.arange(src.w, dtype=torch.float32),
                            indexing="ij")
    cone = theta <= cam.theta_max
    err = (uv_j - torch.stack([xs, ys], -1)).norm(dim=-1)[cone & front]
    assert float(err.max()) < 0.05, float(err.max())
    assert torch.allclose(r_j[cone], rng[cone], atol=1e-4)
    print(f"ok  identity warp: max uv err {float(err.max()):.4f} px")

    # 3. warp consistency: a consistent pair under a real translation gives
    #    (near-)zero loss; an inconsistent one does not
    t = torch.tensor([0.05, 0.0, 0.0])
    # frame j depth constructed by warping i's geometry exactly:
    uv_j, r_in_j, front = losses.warp_i_to_j(rng, cam, torch.eye(3), t)
    # build j's range map by splatting nearest (coarse, so tolerance loose)
    rj = torch.full_like(rng, float("nan"))
    ui = uv_j[..., 0].round().long().clamp(0, src.w - 1)
    vi = uv_j[..., 1].round().long().clamp(0, src.h - 1)
    rj[vi[front], ui[front]] = r_in_j[front]
    rj = torch.nan_to_num(rj, nan=2.0)
    l_cons = losses.multiframe_rim_loss(rng, rj, cam, torch.eye(3), t, theta)
    l_bad = losses.multiframe_rim_loss(rng, rj * 1.5, cam, torch.eye(3), t,
                                       theta)
    assert float(l_cons) < 0.05 < float(l_bad), (float(l_cons), float(l_bad))
    print(f"ok  mv loss: consistent {float(l_cons):.4f} << inconsistent "
          f"{float(l_bad):.4f}")

    # 4. rim feature loss: zero when tokens match; positive otherwise;
    #    gradient flows to student only
    tokens = torch.randn(100, 16, requires_grad=True)
    tp = torch.linspace(0, float(cam.theta_max), 100)
    assert float(losses.rim_feature_loss(tokens, tokens.detach(), tp)) == 0.0
    l = losses.rim_feature_loss(tokens * 2, tokens.detach(), tp)
    l.backward()
    assert tokens.grad is not None
    print("ok  rim feature loss: identity-zero, grad flows")

    # 5. depth loss decreases when prediction moves toward GT
    pred = gt * 2.0
    valid = cone
    l1 = losses.depth_loss(pred, gt, valid, theta)
    l2 = losses.depth_loss(gt * 1.1, gt, valid, theta)
    assert float(l2) < float(l1)
    print("ok  depth loss: ordered")

    print("\nall H5 loss tests passed")


if __name__ == "__main__":
    main()
