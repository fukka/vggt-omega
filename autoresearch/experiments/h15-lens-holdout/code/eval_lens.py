"""H15 eval: every arm, on every lens, on a held-out sequence.

The decider is the HELD-OUT lens. The training lenses are scored in the same
run and reported beside it, because the in-domain row is not a throwaway: H12's
result predicts that `jac` and `shuffled` tie there, and reproducing that tie
here is what turns H12 from an unexplained negative into a measured statement
about identifiability. A win on the held-out lens WITHOUT the in-domain tie
would mean something else is different between the arms and would need
explaining before anything is claimed.

`before` is the frozen backbone: LoRA disabled and the conditioner switched
off, so every arm shares one baseline on every lens.

Usage (box):
    python .../h15-lens-holdout/code/eval_lens.py \\
      --seq $ADT/Apartment_release_clean_seq136_M1292 \\
      --ckpt results/autoresearch-h15-lensholdout/jac/cond_last.pt \\
      --out results/autoresearch-h15-lensholdout/eval_jac_seq136.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

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
_h5_train = _load("h5_train", _H5 / "train.py")
_h5_eval = _load("h5_eval", _H5 / "eval_lora.py")
Seq = _h5_train.Seq
load_lora = _h5_eval.load_lora
THETA_BINS = _h5_eval.THETA_BINS
GT_DEPTH_EDGES = _h5_eval.GT_DEPTH_EDGES

from finetune.eval.metrics import align_depth  # noqa: E402

ZONES = {
    "near_rim(<=2m,>=38deg)": lambda tm, j: tm >= 38 and GT_DEPTH_EDGES[j + 1] <= 2.0,
    "near_center(<=2m,<=11deg)": lambda tm, j: tm <= 11 and GT_DEPTH_EDGES[j + 1] <= 2.0,
    "center(<=11deg)": lambda tm, j: tm <= 11,
    "far(>=3m)": lambda tm, j: GT_DEPTH_EDGES[j] >= 3.0,
}


def zones_from(table, counts, t_mid) -> Dict[str, float]:
    nb_d = len(GT_DEPTH_EDGES) - 1
    out = {}
    for name, keep in ZONES.items():
        cells = [(i, j) for i in range(THETA_BINS) for j in range(nb_d)
                 if keep(t_mid[i], j)]
        w = np.array([counts[i, j] for i, j in cells], float)
        if w.sum() == 0:
            continue
        out[name] = float((np.array([table[i, j] for i, j in cells]) * w).sum() / w.sum())
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--lenses", default=None,
                   help="default: every training lens plus every held-out one")
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--variant", default="small")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    arm = ck["arm"]
    train_names = ck["train_lenses"]
    all_names = ck["all_lenses"]
    names = ([x.strip() for x in a.lenses.split(",")] if a.lenses else all_names)
    base_seed = int(ck["config"]["seed"])

    s = Seq(os.path.expanduser(a.seq), a.size, a.max_frames)
    aria = s.src.camera
    geo = {}
    for n in names:
        cam = LF.make_lens(n, a.size, float(aria.theta_max))
        grid, valid = LF.grid_between(aria, cam)
        theta = cam.incidence_grid(a.size, a.size)
        geo[n] = {"cam": cam, "grid": grid, "valid": valid, "theta": theta,
                  "cos": torch.cos(theta).clamp_min(1e-6),
                  "cone": cam.valid_mask(a.size, a.size),
                  "field": LF.token_field(cam, a.size)}

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device=a.device,
                        variant=a.variant)
    bb.install(None, aria, (a.size, a.size), patch_undistort=False,
               border_token=False, dpt_grid=False, depth_convention="z")
    net = bb.model if hasattr(bb, "model") else bb
    load_lora(net, a.ckpt)

    current = {"field": None}
    if arm != "none":
        vit = bb._vit()
        blocks = getattr(vit, "blocks", None) or getattr(vit, "layers")
        dim = int(next(blocks[-1].parameters()).shape[-1])
        film = FiLMConditioner(3, dim, hidden=ck["config"]["film_hidden"]).to(a.device)
        film.load_state_dict(ck["film"])
        film.eval()

        def hook(_m, _i, out):
            if current["field"] is None:
                return out
            t = out[0] if isinstance(out, tuple) else out
            t2 = film(t, current["field"])
            return (t2,) + tuple(out[1:]) if isinstance(out, tuple) else t2
        blocks[-1].register_forward_hook(hook)

    ones = torch.ones(a.size, a.size)
    nb_d = len(GT_DEPTH_EDGES) - 1
    results = {}
    for name in names:
        g = geo[name]
        t_edges = np.linspace(0.0, float(g["cam"].theta_max), THETA_BINS + 1)
        t_idx = np.clip(np.digitize(g["theta"].numpy(), t_edges) - 1, 0, THETA_BINS - 1)
        t_mid = 0.5 * (t_edges[:-1] + t_edges[1:]) * 180 / np.pi
        # The field this arm shows for THIS lens, rebuilt by the shared rule --
        # a held-out lens has no saved tensor to load.
        fld = (None if arm == "none" else
               A.arm_field(arm, name, {k: geo[k]["field"] for k in geo},
                           all_names, train_names, base_seed).to(a.device))
        tables, counts = {}, np.zeros((THETA_BINS, nb_d))
        for phase in ("before", "after"):
            current["field"] = None if phase == "before" else fld
            s_ = np.zeros((THETA_BINS, nb_d)); n_ = np.zeros((THETA_BINS, nb_d))
            for k in s.frames:
                img = LF.warp(s.src.image(k), g["grid"], mode="bilinear")
                gt_z = LF.warp(s.gt_range(k, ones), g["grid"], mode="nearest")
                gt = (gt_z / g["cos"]).numpy()
                with torch.no_grad():
                    if phase == "before":
                        with lora.lora_disabled(net):
                            pz = U.forward_z(bb, img.to(a.device))
                    else:
                        pz = U.forward_z(bb, img.to(a.device))
                d = (pz.cpu() / g["cos"]).numpy()
                valid = ((g["cone"] & g["valid"]).numpy() & (gt > 0)
                         & (gt <= a.depth_max_m) & (d > 1e-6))
                if valid.sum() < 1000:
                    continue
                al = align_depth(d, gt, valid, mode="scale_shift")
                ar = (np.abs(al - gt) / np.clip(gt, 1e-6, None))[valid]
                di = np.clip(np.digitize(gt[valid], GT_DEPTH_EDGES) - 1, 0, nb_d - 1)
                flat = t_idx[valid] * nb_d + di
                s_ += np.bincount(flat, weights=ar, minlength=THETA_BINS * nb_d
                                  ).reshape(THETA_BINS, nb_d)
                n_ += np.bincount(flat, minlength=THETA_BINS * nb_d
                                  ).reshape(THETA_BINS, nb_d)
            tables[phase] = np.divide(s_, n_, out=np.zeros_like(s_), where=n_ > 0)
            counts = n_
        zb = zones_from(tables["before"], counts, t_mid)
        za = zones_from(tables["after"], counts, t_mid)
        held = name not in train_names
        results[name] = {"held_out": held, "before": zb, "after": za,
                         "table_before": tables["before"].tolist(),
                         "table_after": tables["after"].tolist(),
                         "counts": counts.tolist(),
                         "theta_bin_mid_deg": t_mid.tolist()}
        tag = "HELD-OUT" if held else "train   "
        print(f"[h15/{arm}] {tag} {name}")
        for z in za:
            print(f"    {z}: {zb[z]:.4f} -> {za[z]:.4f} "
                  f"({(za[z] - zb[z]) / zb[z] * 100:+.2f}%)")

    res = {"arm": arm, "seq": s.name, "ckpt": a.ckpt,
           "train_lenses": train_names, "heldout_lenses": ck["heldout_lenses"],
           "lenses": results, "config": vars(a)}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(f"[h15/{arm}] wrote {a.out}")


if __name__ == "__main__":
    main()
