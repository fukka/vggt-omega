"""H12 pilot: does lens-Jacobian conditioning beat a shuffled field?

Three arms, IDENTICAL in architecture, parameter count, data, seed and loss.
The only difference is what the FiLM conditioner is shown:

    --arm jac        the real per-token (log_area, log_aniso, theta/theta_max)
    --arm shuffled   the same values, fixed permutation of token positions
    --arm theta      theta only, zero-padded to the same width

`shuffled` is the arm that decides the experiment. A model that improves on a
scrambled geometry field has gained from capacity, not geometry, and we would
have re-run H5 with more parameters. If jac <= shuffled: STOP, do not proceed to
ScanNet++, and publish the fourth controlled negative.

The loss is the PLAIN depth loss (alpha=0). Rim weighting is the thing that
already failed against its own control in H5 (-80.6% vs plain's -83.5%), so
carrying it here would confound the one question being asked.

Usage (box):
    python .../h12-lens-jacobian/code/train.py --arm jac \
      --train-seqs <4 clean seq dirs> --epochs 20 --out-dir <...>
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_H12 = Path(__file__).resolve().parent
sys.path.insert(0, str(_H12.parents[3]))
sys.path.insert(0, str(_H12.parents[1] / "h1-rim-pose-value" / "code"))
sys.path.append(str(_H12.parents[1] / "h5-rim-finetune" / "code"))
sys.path.insert(0, str(_H12))

import losses  # noqa: E402
import lora  # noqa: E402
import jacobian as J  # noqa: E402
from film import FiLMConditioner, make_arm_field  # noqa: E402

# h5's Seq by FILE, not by name: this module is also called train.py, so a
# plain `from train import Seq` imports itself. Same importlib pattern that
# h6's eval_module.py already uses to borrow it.
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location(
    "h5_train", _H12.parents[1] / "h5-rim-finetune" / "code" / "train.py")
_h5 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_h5)
Seq = _h5.Seq

LORA_PATTERNS = [r"backbone\.pretrained\.blocks\.(8|9|10|11)\.mlp\.fc[12]$"]


def token_jacobian_field(cam, h: int, w: int, patch: int = 14) -> torch.Tensor:
    """Per-token (log_area, log_aniso, theta/theta_max), shape (P, 3).

    theta comes from the CAMERA (authoritative, already used by every other
    experiment here); only the two Jacobian channels are computed from the KB4
    coefficients, so this cannot drift from the repo's lens of record.
    """
    theta = cam.incidence_grid(h, w).cpu().numpy().astype(np.float64)
    k = getattr(cam, "k", None)
    if k is None:
        raise SystemExit("[h12] camera exposes no KB4 coefficients `k`; the "
                         "Jacobian channels cannot be computed without them.")
    k = tuple(float(x) for x in (k.tolist() if hasattr(k, "tolist") else k))[:4]
    la, ln = J.log_area_aniso(theta, lambda t: J.kb4_d(t, k),
                              lambda t: J.kb4_dprime(t, k))
    tmax = float(cam.theta_max)
    gh, gw = h // patch, w // patch
    def pool(a):
        return a.reshape(gh, patch, gw, patch).mean((1, 3)).ravel()
    f = np.stack([pool(la), pool(ln), pool(theta) / tmax], axis=-1)
    return torch.from_numpy(f).float()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=("jac", "shuffled", "theta"))
    p.add_argument("--train-seqs", required=True)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--film-hidden", type=int, default=32)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)

    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    seqs = [Seq(s.strip(), a.size, a.max_frames)
            for s in a.train_seqs.split(",") if s.strip()]
    print(f"[h12/{a.arm}] {len(seqs)} sequences, "
          f"{sum(len(s.frames) for s in seqs)} frames")

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device=a.device,
                        variant="small")
    cam = seqs[0].src.camera
    h = w = a.size
    bb.install(None, cam, (h, w), patch_undistort=False, border_token=False,
               dpt_grid=False, depth_convention="range")
    net = bb.model if hasattr(bb, "model") else bb
    hits = lora.inject(net, LORA_PATTERNS, r=a.lora_r, alpha=2 * a.lora_r)
    assert hits, "LoRA matched nothing"

    theta = cam.incidence_grid(h, w)
    cone = theta <= cam.theta_max
    cos_t = torch.cos(theta)
    field_jac = token_jacobian_field(cam, h, w)
    gen = torch.Generator().manual_seed(a.seed)
    field = make_arm_field(field_jac, a.arm, gen).to(a.device)
    print(f"[h12/{a.arm}] token field {tuple(field.shape)}  "
          f"log_area[{field[:,0].min():.3f},{field[:,0].max():.3f}]  "
          f"log_aniso[{field[:,1].min():.3f},{field[:,1].max():.3f}]")

    vit = bb._vit()
    blocks = getattr(vit, "blocks", None) or getattr(vit, "layers")
    dim = int(field_jac.shape[0] and next(blocks[-1].parameters()).shape[-1])
    film = FiLMConditioner(3, dim, hidden=a.film_hidden).to(a.device)

    def film_hook(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out
        t2 = film(t, field)
        return (t2,) + tuple(out[1:]) if isinstance(out, tuple) else t2
    blocks[-1].register_forward_hook(film_hook)

    n_lora = sum(x.numel() for x in lora.lora_parameters(net))
    n_film = sum(x.numel() for x in film.parameters())
    print(f"[h12/{a.arm}] LoRA {n_lora/1e3:.1f}k + FiLM {n_film/1e3:.1f}k params")

    params = list(lora.lora_parameters(net)) + list(film.parameters())
    opt = torch.optim.Adam(params, lr=a.lr)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    log: List[Dict] = []
    for ep in range(a.epochs):
        items = [(s, n) for s in seqs for n in s.frames]
        random.shuffle(items)
        if a.steps_per_epoch:
            items = items[:a.steps_per_epoch]
        tot, cnt, t0 = 0.0, 0, time.time()
        for s, n in items:
            opt.zero_grad()
            im = s.src.image(n)[None, None].to(a.device)
            pred = bb.forward(im).depth[0]
            gt = s.gt_range(n, cos_t).to(a.device)
            valid = (cone.to(a.device) & (gt > 0) & (gt <= a.depth_max_m))
            if not bool(valid.any()):
                continue
            # alpha=0: PLAIN depth loss. Rim weighting is what failed in H5.
            loss = losses.depth_loss(pred, gt, valid, theta.to(a.device), alpha=0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss); cnt += 1
        rec = {"epoch": ep, "depth": tot / max(cnt, 1), "n": cnt,
               "sec": round(time.time() - t0, 1)}
        log.append(rec)
        print(f"[h12/{a.arm}] ep{ep:02d} depth {rec['depth']:.4f} "
              f"({cnt} steps, {rec['sec']}s)", flush=True)
        torch.save({"lora": {k: v for k, v in net.state_dict().items()
                             if "lora_" in k},
                    "film": film.state_dict(),
                    "arm": a.arm, "config": vars(a)}, out / "cond_last.pt")
        (out / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"[h12/{a.arm}] done -> {out}")


if __name__ == "__main__":
    main()
