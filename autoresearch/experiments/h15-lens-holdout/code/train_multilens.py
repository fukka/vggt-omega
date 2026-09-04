"""H15: lens-Jacobian conditioning, trained on MANY lenses, tested on a new one.

H12 asked whether the real field beats a shuffled one and answered no -- on a
single lens, where the two arms are information-equivalent by construction (see
`lens_family.py`). This puts the same question where it can be answered: the
frame is warped into a lens drawn from a family at every step, and the decider
is a lens the model has never seen.

Four arms, identical in architecture, data, seed, optimiser and loss:

    --arm jac          the real field of the lens the frame is currently in
    --arm mismatched   a real field of a DIFFERENT lens (smooth, wrong)
    --arm shuffled     this lens's field, per-lens position permutation
    --arm none         no conditioner: plain LoRA, the standing baseline

DEPTH CONVENTION, EXPLICITLY
----------------------------
The backbone is installed ONCE with `depth_convention="z"`, which is DA3's
native convention and therefore a no-op inside `_finalize`. This is not a
convenience: `_finalize` would otherwise convert z -> range using the camera
that was installed, and the installed camera cannot follow the lens the image
was warped into without re-hooking the model on every step. Installing the Aria
camera and feeding a stereographically warped frame would apply the wrong
theta -- a smooth radial error that no scale alignment can absorb, which is
exactly the bug class that invalidated #38 v1. So the conversion is done here,
per lens, against the lens the pixels are actually in.

GT is resampled and NOT converted: planar z is invariant under a co-axial lens
re-parameterisation, and `test_planar_z_is_invariant_under_a_lens_warp` checks
it numerically.

Usage (box):
    python .../h15-lens-holdout/code/train_multilens.py --arm jac \\
      --train-seqs <4 clean seq dirs> --epochs 40 \\
      --out-dir results/autoresearch-h15-lensholdout/jac
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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[3]))
sys.path.insert(0, str(_HERE.parents[1] / "h1-rim-pose-value" / "code"))
sys.path.append(str(_HERE.parents[1] / "h5-rim-finetune" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "h12-lens-jacobian" / "code"))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "common"))

import importlib.util as _ilu  # noqa: E402
import losses  # noqa: E402
import lora  # noqa: E402
import lens_family as LF  # noqa: E402
import upright as U  # noqa: E402
import arms as A  # noqa: E402
from film import FiLMConditioner  # noqa: E402


def _load(name, path):
    sp = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


_H5 = _HERE.parents[1] / "h5-rim-finetune" / "code"
Seq = _load("h5_train", _H5 / "train.py").Seq

LORA_PATTERNS = [r"backbone\.pretrained\.blocks\.(8|9|10|11)\.mlp\.fc[12]$"]


def build_lens_geometry(aria, size: int, names) -> Dict[str, dict]:
    """Everything that depends only on the lens, computed once."""
    out = {}
    for name in names:
        cam = LF.make_lens(name, size, float(aria.theta_max))
        grid, valid = LF.grid_between(aria, cam)
        theta = cam.incidence_grid(size, size)
        out[name] = {
            "cam": cam, "grid": grid, "valid": valid, "theta": theta,
            "cos": torch.cos(theta).clamp_min(1e-6),
            "cone": cam.valid_mask(size, size),
            "field": LF.token_field(cam, size),
        }
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=A.ARMS)
    p.add_argument("--train-seqs", required=True)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--steps-per-epoch", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--film-hidden", type=int, default=32)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--variant", default="small")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)

    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    seqs = [Seq(s.strip(), a.size, a.max_frames)
            for s in a.train_seqs.split(",") if s.strip()]
    aria = seqs[0].src.camera
    train_names = list(LF.TRAIN_LENSES)
    all_names = train_names + list(LF.HELDOUT_LENSES)
    geo = build_lens_geometry(aria, a.size, train_names)
    print(f"[h15/{a.arm}] {len(seqs)} sequences, "
          f"{sum(len(s.frames) for s in seqs)} frames, "
          f"{len(train_names)} training lenses: {', '.join(train_names)}")
    print(f"[h15/{a.arm}] held out and never seen: "
          f"{', '.join(LF.HELDOUT_LENSES)}")

    fields = {n: geo[n]["field"] for n in train_names}
    if a.arm != "none":
        arm_fields = {n: A.arm_field(a.arm, n, fields, all_names, train_names,
                                     a.seed).to(a.device) for n in train_names}
        for n in train_names:
            f = arm_fields[n]
            print(f"    {n:14s} log_area[{f[:,0].min():+.3f},{f[:,0].max():+.3f}] "
                  f"log_aniso[{f[:,1].min():+.3f},{f[:,1].max():+.3f}]")

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device=a.device,
                        variant=a.variant)
    # convention="z" == DA3's native == a no-op in _finalize; see the module
    # docstring for why anything else would be wrong here.
    bb.install(None, aria, (a.size, a.size), patch_undistort=False,
               border_token=False, dpt_grid=False, depth_convention="z")
    net = bb.model if hasattr(bb, "model") else bb
    hits = lora.inject(net, LORA_PATTERNS, r=a.lora_r, alpha=2 * a.lora_r)
    assert hits, "LoRA matched nothing"

    film = None
    current = {"field": None}
    if a.arm != "none":
        vit = bb._vit()
        blocks = getattr(vit, "blocks", None) or getattr(vit, "layers")
        dim = int(next(blocks[-1].parameters()).shape[-1])
        film = FiLMConditioner(3, dim, hidden=a.film_hidden).to(a.device)

        def film_hook(_m, _i, out):
            if current["field"] is None:
                return out
            t = out[0] if isinstance(out, tuple) else out
            t2 = film(t, current["field"])
            return (t2,) + tuple(out[1:]) if isinstance(out, tuple) else t2
        blocks[-1].register_forward_hook(film_hook)

    params = list(lora.lora_parameters(net))
    if film is not None:
        params += list(film.parameters())
    n_lora = sum(x.numel() for x in lora.lora_parameters(net))
    n_film = sum(x.numel() for x in film.parameters()) if film else 0
    print(f"[h15/{a.arm}] LoRA {n_lora/1e3:.1f}k + FiLM {n_film/1e3:.1f}k params")

    opt = torch.optim.Adam(params, lr=a.lr)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    ones = torch.ones(a.size, a.size)
    log: List[Dict] = []
    for ep in range(a.epochs):
        items = [(s, n) for s in seqs for n in s.frames]
        random.shuffle(items)
        if a.steps_per_epoch:
            items = items[:a.steps_per_epoch]
        tot, cnt, t0 = 0.0, 0, time.time()
        per_lens = {n: 0 for n in train_names}
        for s, n in items:
            lens = random.choice(train_names)
            g = geo[lens]
            per_lens[lens] += 1
            opt.zero_grad()
            # `s.gt_range(n, ones)` returns the stored PLANAR Z (the loader
            # divides by whatever cos map it is handed); ones makes that
            # explicit rather than reaching into the loader's internals.
            gt_z = LF.warp(s.gt_range(n, ones), g["grid"], mode="nearest")
            img = LF.warp(s.src.image(n), g["grid"], mode="bilinear")
            gt = (gt_z / g["cos"]).to(a.device)
            valid = ((g["cone"] & g["valid"]).to(a.device)
                     & (gt > 0) & (gt <= a.depth_max_m))
            if not bool(valid.any()):
                continue
            current["field"] = None if a.arm == "none" else arm_fields[lens]
            pred_z = U.forward_z(bb, img.to(a.device))
            pred = pred_z / g["cos"].to(a.device)
            # alpha=0: PLAIN log-L1. Rim weighting lost to its own control in H5.
            loss = losses.depth_loss(pred, gt, valid,
                                     g["theta"].to(a.device), alpha=0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss); cnt += 1
        rec = {"epoch": ep, "depth": tot / max(cnt, 1), "n": cnt,
               "sec": round(time.time() - t0, 1), "per_lens": per_lens}
        log.append(rec)
        print(f"[h15/{a.arm}] ep{ep:02d} depth {rec['depth']:.4f} "
              f"({cnt} steps, {rec['sec']}s)", flush=True)
        state = {name: {"A": m.A.detach().cpu(), "B": m.B.detach().cpu()}
                 for name, m in hits}
        assert state, "no LoRA tensors collected -- refusing to save an empty checkpoint"
        ck = {"lora": state, "patterns": LORA_PATTERNS, "arm": a.arm,
              "epoch": ep, "config": vars(a), "train_lenses": train_names,
              "heldout_lenses": list(LF.HELDOUT_LENSES), "all_lenses": all_names}
        if film is not None:
            ck["film"] = film.state_dict()
        torch.save(ck, out / "cond_last.pt")
        (out / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"[h15/{a.arm}] done -> {out}")


if __name__ == "__main__":
    main()
