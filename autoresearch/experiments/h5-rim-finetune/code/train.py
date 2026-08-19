"""H5 trainer: rim-targeted, pose-preserving LoRA finetune of DA3-Small.

Protocol: ../protocol.md. Mechanics verified by train_smoke.py (teacher =
LoRA-disabled path, bit-identical; base weights never move).

Saves ONLY the LoRA tensors + config to --out-dir (checkpoint is a few MB).

Usage (box):
    python autoresearch/experiments/h5-rim-finetune/code/train.py \
        --train-seqs <4 clean seq dirs, comma-separated> \
        --epochs 20 --out-dir results/autoresearch-h5-train/run_015
CPU smoke (Mac):
    ... --train-seqs <seq131> --epochs 1 --steps-per-epoch 3 --size 252
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "h1-rim-pose-value" / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adt_pose_value import AriaLocalPairs  # noqa: E402
import losses  # noqa: E402
import lora  # noqa: E402

LORA_PATTERNS = [r"backbone\.pretrained\.blocks\.(8|9|10|11)\.mlp\.fc[12]$"]
CALIB = "cam3r/data/adt_camera_rgb_calibration.json"


def camera_conjugation() -> torch.Tensor:
    cal = json.loads(Path(CALIB).read_text())
    x, y, z, w = cal["T_device_camera"]["quaternion_xyzw"]
    R_dc = torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]],
        dtype=torch.float64)
    return R_dc.T                                     # camera-from-device


class Seq:
    def __init__(self, seq_dir: str, size: int, max_frames: int):
        self.src = AriaLocalPairs(seq_dir, size=size)
        self.name = os.path.basename(seq_dir.rstrip("/"))
        dp = {os.path.splitext(os.path.basename(q))[0]: q for q in
              glob.glob(os.path.join(seq_dir, "depth_npy", "*.npy"))}
        stem = lambda n: os.path.splitext(os.path.basename(self.src.paths[n]))[0]
        fr = [n for n in range(len(self.src.paths)) if stem(n) in dp]
        if len(fr) > max_frames:
            fr = fr[::max(1, len(fr) // max_frames)][:max_frames]
        self.frames, self.dp, self.stem = fr, dp, stem
        # adjacent-frame pairs with GT pose
        self.pairs: List[Tuple[int, int]] = [
            (a, b) for a, b in zip(fr[:-1], fr[1:])
            if self.src.pose(a) is not None and self.src.pose(b) is not None]

    def gt_range(self, n: int, cos_t: torch.Tensor) -> torch.Tensor:
        gz = torch.from_numpy(
            np.load(self.dp[self.stem(n)]).astype(np.float32))
        gz = torch.nn.functional.interpolate(
            gz[None, None], size=cos_t.shape, mode="nearest")[0, 0] / 1000.0
        return gz / cos_t.clamp_min(1e-6)

    def rel_pose(self, a: int, b: int, C: torch.Tensor):
        gi, gj = self.src.pose(a), self.src.pose(b)
        R_dev = gj[0] @ gi[0].transpose(-1, -2)
        t_dev = gj[1] - R_dev @ gi[1]
        return (C @ R_dev @ C.T).float(), (C @ t_dev).float()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-seqs", required=True)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=0,
                   help="0 = one pass over all pairs")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lambda-f", type=float, default=1.0)
    p.add_argument("--depth-alpha", type=float, default=2.0,
                   help="compression-weight strength; 0 = unweighted L1 "
                        "(the plain-LoRA control together with "
                        "--lambda-f 0 --lambda-m 0)")
    p.add_argument("--lambda-m", type=float, default=0.5)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    seqs = [Seq(s.strip(), args.size, args.max_frames)
            for s in args.train_seqs.split(",") if s.strip()]
    C = camera_conjugation()
    print(f"[h5] {len(seqs)} sequences, "
          f"{sum(len(s.pairs) for s in seqs)} training pairs")

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device=args.device,
                        variant="small")
    cam = seqs[0].src.camera
    h = w = args.size
    bb.install(None, cam, (h, w), patch_undistort=False, border_token=False,
               dpt_grid=False, depth_convention="range")
    net = bb.model if hasattr(bb, "model") else bb
    hits = lora.inject(net, LORA_PATTERNS, r=args.lora_r, alpha=2 * args.lora_r)
    print(f"[h5] LoRA on {len(hits)} linears, "
          f"{sum(x.numel() for x in lora.lora_parameters(net)) / 1e3:.1f}k params")
    assert hits, "LoRA matched nothing"

    theta = cam.incidence_grid(h, w)
    cone = theta <= cam.theta_max
    cos_t = torch.cos(theta)
    grabbed: Dict[str, torch.Tensor] = {}
    vit = bb._vit()
    blocks = getattr(vit, "blocks", None) or getattr(vit, "layers")
    hook = blocks[-1].register_forward_hook(
        lambda _m, _i, out: grabbed.__setitem__(
            "tok", out[0] if isinstance(out, tuple) else out))
    n_patch = (h // 14) * (w // 14)
    gh, gw = h // 14, w // 14
    theta_p = theta.reshape(gh, 14, gw, 14).mean((1, 3)).ravel()

    def tokens() -> torch.Tensor:
        t = grabbed["tok"].reshape(-1, grabbed["tok"].shape[-1])
        return t[t.shape[0] - n_patch:]

    opt = torch.optim.Adam(lora.lora_parameters(net), lr=args.lr)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log: List[Dict] = []
    for ep in range(args.epochs):
        pairs = [(s, a, b) for s in seqs for a, b in s.pairs]
        random.shuffle(pairs)
        if args.steps_per_epoch:
            pairs = pairs[:args.steps_per_epoch]
        agg = {"depth": 0.0, "feat": 0.0, "mv": 0.0, "total": 0.0}
        t0 = time.time()
        for s, a, b in pairs:
            opt.zero_grad()
            ims = [s.src.image(k)[None, None].to(args.device) for k in (a, b)]
            preds, toks = [], []
            for im in ims:
                pr = bb.forward(im)
                preds.append(pr.depth[0])
                toks.append(tokens())
            with torch.no_grad(), lora.lora_disabled(net):
                bb.forward(ims[0])
                ttok0 = tokens()
                bb.forward(ims[1])
                ttok1 = tokens()
            gta = s.gt_range(a, cos_t).to(args.device)
            gtb = s.gt_range(b, cos_t).to(args.device)
            R_rel, t_rel = s.rel_pose(a, b, C)
            l_d = (losses.depth_loss(preds[0], gta,
                                     cone.to(args.device) & (gta > 0)
                                     & (gta <= args.depth_max_m), theta,
                                     alpha=args.depth_alpha)
                   + losses.depth_loss(preds[1], gtb,
                                       cone.to(args.device) & (gtb > 0)
                                       & (gtb <= args.depth_max_m), theta,
                                       alpha=args.depth_alpha))
            l_f = (losses.rim_feature_loss(toks[0], ttok0, theta_p)
                   + losses.rim_feature_loss(toks[1], ttok1, theta_p))
            l_m = losses.multiframe_rim_loss(preds[0], preds[1], cam,
                                             R_rel, t_rel, theta)
            loss = l_d + args.lambda_f * l_f + args.lambda_m * l_m
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(lora.lora_parameters(net)), 1.0)
            opt.step()
            for k, v in (("depth", l_d), ("feat", l_f), ("mv", l_m),
                         ("total", loss)):
                agg[k] += float(v)
        n = max(1, len(pairs))
        row = {k: v / n for k, v in agg.items()}
        row.update(epoch=ep, n_pairs=len(pairs), sec=round(time.time() - t0, 1))
        log.append(row)
        print(f"[h5] epoch {ep:3d}: total {row['total']:.4f} "
              f"(d {row['depth']:.4f} f {row['feat']:.5f} m {row['mv']:.4f}) "
              f"{row['sec']}s", flush=True)
        state = {name: {"A": m.A.detach().cpu(), "B": m.B.detach().cpu()}
                 for name, m in hits}
        torch.save({"lora": state, "config": vars(args),
                    "patterns": LORA_PATTERNS, "epoch": ep},
                   out / "lora_last.pt")
        with open(out / "train_log.json", "w") as f:
            json.dump(log, f, indent=2)
    hook.remove()
    print(f"[h5] done; checkpoint + log in {out}")


if __name__ == "__main__":
    main()
