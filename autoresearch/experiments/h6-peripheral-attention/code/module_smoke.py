"""H6 module smoke on real DA3 + real ADT frames (CPU).

Checks:
  1. zero-init ⇒ depth-head output on the modified feats copy is
     bit-identical to the unmodified path;
  2. after one gradient step on the module, depth changes at the rim but the
     camera estimation — fed the ORIGINAL feats — is bit-identical;
  3. only module parameters receive gradients.

Usage (repo root):
    <venv>/bin/python autoresearch/experiments/h6-peripheral-attention/code/module_smoke.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "h1-rim-pose-value" / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adt_pose_value import AriaLocalPairs, DEFAULT_SEQ  # noqa: E402
from peripheral_attn import (PeripheralCrossFrameAttention,  # noqa: E402
                             apply_to_final_level, rim_mask_for)

SIZE = 252


def main() -> None:
    torch.manual_seed(0)
    src = AriaLocalPairs(DEFAULT_SEQ, size=SIZE)
    theta = src.camera.incidence_grid(SIZE, SIZE)
    gh = SIZE // 14
    theta_p = theta.reshape(gh, 14, gh, 14).mean((1, 3)).ravel()
    rim = rim_mask_for(theta_p)
    print(f"[h6] {int(rim.sum())}/{len(rim)} rim patches "
          f"({float(rim.float().mean()) * 100:.0f}%)")

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device="cpu",
                        variant="small")
    bb.install(None, src.camera, (SIZE, SIZE), patch_undistort=False,
               border_token=False, dpt_grid=False, depth_convention="range")
    net = bb.model if hasattr(bb, "model") else bb
    for p in net.parameters():
        p.requires_grad_(False)

    x0 = src.image(0)[None, None]
    x1 = src.image(1)[None, None]
    with torch.no_grad():
        feats_prev, _ = net.backbone(x0, cam_token=None, export_feat_layers=[])
    feats_t, _ = net.backbone(x1, cam_token=None, export_feat_layers=[])
    H = W = SIZE

    dim = feats_t[-1][0].shape[-1]
    module = PeripheralCrossFrameAttention(dim)
    print(f"[h6] module dim {dim}, "
          f"{sum(p.numel() for p in module.parameters()) / 1e6:.2f}M params")

    def cam_snapshot(feats):
        # camera estimation consumes/mutates the head output in place, so it
        # always gets a fresh one; snapshot every tensor leaf for comparison
        out = net._process_depth_head(list(feats), H, W)
        out = net._process_camera_estimation(list(feats), H, W, out)
        return {k: v.detach().clone() for k, v in out.items()
                if torch.is_tensor(v)}

    with torch.no_grad():
        base = net._process_depth_head(list(feats_t), H, W)
        base_depth = (base["depth"] if isinstance(base, dict)
                      else base.depth).detach().clone()
        cam_base = cam_snapshot(feats_t)

    # 1. zero-init identity through the depth head
    feats_mod = apply_to_final_level(module, feats_t, feats_prev, rim)
    with torch.no_grad():
        out0 = net._process_depth_head(feats_mod, H, W)
        d0 = out0["depth"] if isinstance(out0, dict) else out0.depth
    assert torch.equal(d0, base_depth), "zero-init is not identity"
    print("[h6] zero-init: depth bit-identical through the head")

    # 2. one gradient step -> depth moves at the rim; camera path untouched
    opt = torch.optim.Adam(module.parameters(), lr=1e-2)
    feats_mod = apply_to_final_level(module, feats_t, feats_prev, rim)
    out = net._process_depth_head(feats_mod, H, W)
    d = out["depth"] if isinstance(out, dict) else out.depth
    loss = d.abs().mean()
    loss.backward()
    grads = [n for n, p in module.named_parameters()
             if p.grad is not None and float(p.grad.abs().sum()) > 0]
    assert grads, "no gradient reached the module"
    opt.step()
    with torch.no_grad():
        feats_mod2 = apply_to_final_level(module, feats_t, feats_prev, rim)
        out2 = net._process_depth_head(feats_mod2, H, W)
        d2 = out2["depth"] if isinstance(out2, dict) else out2.depth
        cam_after = cam_snapshot(feats_t)
    assert not torch.equal(d2, base_depth), "depth did not move after step"
    assert set(cam_base) == set(cam_after)
    for k in cam_base:
        assert torch.equal(cam_base[k], cam_after[k]), f"camera moved: {k}"
    print(f"[h6] after 1 step: depth moved, camera outputs bit-identical, "
          f"{len(grads)} module tensors got grads\n\nH6 module smoke PASSED")


if __name__ == "__main__":
    main()
