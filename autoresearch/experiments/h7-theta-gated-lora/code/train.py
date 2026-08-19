"""H7 trainer: the H5 training loop with GatedLoRALinear injection.

Deliberately a thin mirror of h5's main (same losses, defaults, seed
handling) so the ONLY difference between the uniform anchor and this arm is
the gate. Reuses Seq / losses / camera_conjugation from the h5 module.

Usage (repo root):
    <venv>/bin/python autoresearch/experiments/h7-theta-gated-lora/code/train.py \
        --train-seqs <dir> --size 252 --epochs 10 --out-dir <dir>
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "h5-rim-finetune" / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_spec = _ilu.spec_from_file_location(
    "h5_train", Path(__file__).resolve().parents[2]
    / "h5-rim-finetune" / "code" / "train.py")
_h5 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_h5)
import losses  # noqa: E402
import lora as h5_lora  # noqa: E402
import gated_lora  # noqa: E402


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-seqs", required=True)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lambda-f", type=float, default=1.0)
    p.add_argument("--depth-alpha", type=float, default=2.0)
    p.add_argument("--lambda-m", type=float, default=0.5)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    seqs = [_h5.Seq(s.strip(), args.size, args.max_frames)
            for s in args.train_seqs.split(",") if s.strip()]
    C = _h5.camera_conjugation()

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device=args.device,
                        variant="small")
    cam = seqs[0].src.camera
    h = w = args.size
    bb.install(None, cam, (h, w), patch_undistort=False, border_token=False,
               dpt_grid=False, depth_convention="range")
    net = bb.model if hasattr(bb, "model") else bb
    hits = gated_lora.inject(net, _h5.LORA_PATTERNS, r=args.lora_r,
                             alpha=2 * args.lora_r)
    n_par = sum(x.numel() for x in gated_lora.parameters(net))
    print(f"[h7] gated LoRA on {len(hits)} linears, {n_par / 1e3:.1f}k params")
    assert hits, "gated LoRA matched nothing"

    theta = cam.incidence_grid(h, w)
    cone = theta <= cam.theta_max
    cos_t = torch.cos(theta)
    gh, gw = h // 14, w // 14
    n_patch = gh * gw
    theta_p = theta.reshape(gh, 14, gw, 14).mean((1, 3)).ravel()

    # token count: probe one forward to size the gate context
    with torch.no_grad():
        grabbed: Dict[str, torch.Tensor] = {}
        vit = bb._vit()
        blocks = getattr(vit, "blocks", None) or getattr(vit, "layers")
        hook = blocks[-1].register_forward_hook(
            lambda _m, _i, out: grabbed.__setitem__(
                "tok", out[0] if isinstance(out, tuple) else out))
        bb.forward(seqs[0].src.image(seqs[0].frames[0])[None, None]
                   .to(args.device))
        n_total = grabbed["tok"].reshape(-1, grabbed["tok"].shape[-1]).shape[0]
    gated_lora.set_theta(theta_p, n_total, float(cam.theta_max))
    print(f"[h7] tokens: {n_total} total, {n_patch} patches, "
          f"{n_total - n_patch} special")

    def tokens() -> torch.Tensor:
        t = grabbed["tok"].reshape(-1, grabbed["tok"].shape[-1])
        return t[t.shape[0] - n_patch:]

    opt = torch.optim.Adam(gated_lora.parameters(net), lr=args.lr)
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
            with torch.no_grad(), h5_lora.lora_disabled(net):
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
            torch.nn.utils.clip_grad_norm_(
                list(gated_lora.parameters(net)), 1.0)
            opt.step()
            for k, v in (("depth", l_d), ("feat", l_f), ("mv", l_m),
                         ("total", loss)):
                agg[k] += float(v)
        n = max(1, len(pairs))
        row = {k: v / n for k, v in agg.items()}
        row.update(epoch=ep, n_pairs=len(pairs), sec=round(time.time() - t0, 1))
        log.append(row)
        print(f"[h7] epoch {ep:3d}: total {row['total']:.4f} "
              f"(d {row['depth']:.4f} f {row['feat']:.5f} m {row['mv']:.4f}) "
              f"{row['sec']}s", flush=True)
        torch.save({"lora": gated_lora.state_of(hits), "config": vars(args),
                    "patterns": _h5.LORA_PATTERNS, "epoch": ep,
                    "gated": True,
                    "gate_curve": gated_lora.gate_curve(
                        hits, float(cam.theta_max))},
                   out / "gated_lora_last.pt")
        with open(out / "train_log.json", "w") as f:
            json.dump(log, f, indent=2)
    hook.remove()


if __name__ == "__main__":
    main()
