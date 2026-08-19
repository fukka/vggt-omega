"""H5 training smoke: 2 frames, 1 GT-pose pair, 2 optimizer steps, CPU.

Checks, in order:
  1. LoRA injection hits the intended layers (dense head + last-4 block MLPs)
     and nothing else trains;
  2. with LoRA disabled the model is bit-identical to pristine (the teacher);
  3. all three protocol losses evaluate on real ADT frames and backprop;
  4. two Adam steps reduce the total loss;
  5. after training, the LoRA-disabled path is STILL bit-identical
     (pose-safety mechanics: the base weights never moved).

Usage (repo root):
    <venv>/bin/python autoresearch/experiments/h5-rim-finetune/code/train_smoke.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "h1-rim-pose-value" / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adt_pose_value import AriaLocalPairs, DEFAULT_SEQ  # noqa: E402
import losses  # noqa: E402
import lora  # noqa: E402

SIZE = 252
# DA3-Small: ViT linears live at backbone.pretrained.blocks.N.mlp.fc{1,2};
# the DPT head is conv-based (no Linears), so the protocol's "head" share of
# LoRA starts EMPTY — documented deviation; conv-LoRA is the fallback if the
# MLP-only variant underfits. cam_enc/cam_dec (the pose path) are never
# matched by these patterns, by construction.
LORA_PATTERNS = [r"backbone\.pretrained\.blocks\.(8|9|10|11)\.mlp\.fc[12]$"]


def main() -> None:
    torch.manual_seed(0)
    src = AriaLocalPairs(DEFAULT_SEQ, size=SIZE)
    cam = src.camera
    theta = cam.incidence_grid(SIZE, SIZE)
    cone = theta <= cam.theta_max

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device="cpu",
                        variant="small")
    bb.install(None, cam, (SIZE, SIZE), patch_undistort=False,
               border_token=False, dpt_grid=False, depth_convention="range")
    net = bb.model if hasattr(bb, "model") else bb

    # pristine reference BEFORE injection
    img0 = src.image(0)[None, None]
    with torch.no_grad():
        ref = bb.forward(img0).depth[0].clone()

    hits = lora.inject(net, LORA_PATTERNS, r=8, alpha=16)
    n_lora = sum(p.numel() for p in lora.lora_parameters(net))
    n_frozen = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    print(f"[smoke] LoRA on {len(hits)} linears, {n_lora/1e3:.1f}k trainable "
          f"vs {n_frozen/1e6:.1f}M frozen")
    assert len(hits) >= 4, [h[0] for h in hits]
    trainable = [n for n, p in net.named_parameters() if p.requires_grad]
    assert all(("A" in n.split(".")[-1] or "B" in n.split(".")[-1])
               for n in trainable), trainable[:5]

    # teacher identity at init AND with LoRA disabled
    with torch.no_grad():
        out_init = bb.forward(img0).depth[0]
        with lora.lora_disabled(net):
            out_dis = bb.forward(img0).depth[0]
    assert torch.equal(out_dis, ref), "disabled path is not bit-identical"
    assert torch.allclose(out_init, ref, atol=1e-5), "zero-init not identity"
    print("[smoke] teacher path bit-identical; zero-init is identity")

    # data: frames 0 and 1 with GT depth + GT relative pose (official calib)
    def gt_range(n):
        stem = src.paths[n].split("/")[-1].replace(".jpg", "")
        import glob, os
        dp = glob.glob(str(Path(DEFAULT_SEQ) / "depth_npy" / f"{stem}.npy"))[0]
        gz = torch.from_numpy(np.load(dp).astype(np.float32))
        gz = torch.nn.functional.interpolate(gz[None, None], size=(SIZE, SIZE),
                                             mode="nearest")[0, 0] / 1000.0
        return gz / torch.cos(theta).clamp_min(1e-6)

    cal = json.loads(Path("cam3r/data/adt_camera_rgb_calibration.json").read_text())
    q = cal["T_device_camera"]["quaternion_xyzw"]
    x, y, z, w = q
    R_dc = torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]],
        dtype=torch.float64)
    C = R_dc.T
    gi, gj = src.pose(0), src.pose(1)
    R_dev = gj[0] @ gi[0].transpose(-1, -2)
    t_dev = gj[1] - R_dev @ gi[1]
    R_rel = (C @ R_dev @ C.T).float()
    t_rel = (C @ t_dev).float()

    gt0, gt1 = gt_range(0), gt_range(1)
    imgs = [src.image(0)[None, None], src.image(1)[None, None]]

    grabbed = {}
    vit = bb._vit()
    blocks = getattr(vit, "blocks", None) or getattr(vit, "layers")
    hook = blocks[-1].register_forward_hook(
        lambda _m, _i, out: grabbed.__setitem__(
            "tok", out[0] if isinstance(out, tuple) else out))
    n_patch = (SIZE // 14) ** 2
    gh = SIZE // 14
    theta_p = theta.reshape(gh, 14, gh, 14).mean((1, 3)).ravel()

    opt = torch.optim.Adam(lora.lora_parameters(net), lr=3e-4)
    hist = []
    hist_d = []
    for step in range(5):
        opt.zero_grad()
        total = 0.0
        preds, toks = [], []
        for k in (0, 1):
            pred = bb.forward(imgs[k])
            d = pred.depth[0]
            tok = grabbed["tok"].reshape(-1, grabbed["tok"].shape[-1])
            toks.append(tok[tok.shape[0] - n_patch:])
            preds.append(d)
        with torch.no_grad(), lora.lora_disabled(net):
            bb.forward(imgs[0])
            ttok = grabbed["tok"].reshape(-1, grabbed["tok"].shape[-1])
            ttok = ttok[ttok.shape[0] - n_patch:]
        l_d = (losses.depth_loss(preds[0], gt0, cone & (gt0 > 0), theta)
               + losses.depth_loss(preds[1], gt1, cone & (gt1 > 0), theta))
        l_f = losses.rim_feature_loss(toks[0], ttok, theta_p)
        l_m = losses.multiframe_rim_loss(preds[0], preds[1], cam,
                                         R_rel, t_rel, theta)
        loss = l_d + 1.0 * l_f + 0.5 * l_m
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            list(lora.lora_parameters(net)), 1.0)
        opt.step()
        hist.append(float(loss))
        hist_d.append(float(l_d))
        print(f"[smoke] step {step}: total {float(loss):.4f} "
              f"(depth {float(l_d):.4f}, feat {float(l_f):.6f}, "
              f"mv {float(l_m):.4f}), grad {float(gnorm):.3f}", flush=True)
    hook.remove()
    # Composite-objective dynamics: the feat term starts at exactly 0 and must
    # rise as LoRA moves; the smoke gate is that TOTAL ends below start over 5
    # steps and the depth term decreases monotonically-ish.
    assert hist[-1] < hist[0], hist
    assert hist_d[-1] < hist_d[0], hist_d
    print(f"[smoke] total {hist[0]:.4f}->{hist[-1]:.4f}, "
          f"depth {hist_d[0]:.4f}->{hist_d[-1]:.4f} over 5 steps")

    with torch.no_grad(), lora.lora_disabled(net):
        out_after = bb.forward(img0).depth[0]
    assert torch.equal(out_after, ref), "base weights moved!"
    print("[smoke] base path STILL bit-identical after training — "
          "pose-safety mechanics verified\n\nH5 smoke PASSED")


if __name__ == "__main__":
    main()
