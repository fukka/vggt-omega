"""Held-out eval for the H12 arms. Same zones and split as #35/#36, so the
numbers land beside them rather than in a table of their own.

`before` disables BOTH LoRA and FiLM, so it is the pretrained backbone and every
arm shares one baseline. What decides the pilot is the `after` column ACROSS
arms: jac vs shuffled at equal capacity, equal data, equal seed.

The per-token field is read from the CHECKPOINT, never rebuilt here. Rebuilding
it would let an eval-time change of camera or patch size silently score a
different conditioning than the one that was trained.
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

_H12 = Path(__file__).resolve().parent
sys.path.insert(0, str(_H12.parents[3]))
sys.path.insert(0, str(_H12.parents[1] / "h1-rim-pose-value" / "code"))
sys.path.insert(0, str(_H12))
sys.path.append(str(_H12.parents[1] / "h5-rim-finetune" / "code"))

import lora  # noqa: E402
from film import FiLMConditioner  # noqa: E402

import importlib.util as _ilu  # noqa: E402
_H5 = _H12.parents[1] / "h5-rim-finetune" / "code"
def _load(name, path):
    sp = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(sp); sp.loader.exec_module(m); return m
_h5_train = _load("h5_train", _H5 / "train.py")
_h5_eval = _load("h5_eval", _H5 / "eval_lora.py")
Seq = _h5_train.Seq
load_lora = _h5_eval.load_lora
THETA_BINS = _h5_eval.THETA_BINS
GT_DEPTH_EDGES = _h5_eval.GT_DEPTH_EDGES

from finetune.eval.metrics import align_depth  # noqa: E402


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    arm = ck["arm"]
    s = Seq(os.path.expanduser(a.seq), a.size, a.max_frames)
    cam = s.src.camera
    h = w = a.size
    theta = cam.incidence_grid(h, w)
    cone = theta <= cam.theta_max
    cos_t = torch.cos(theta)
    t_edges = np.linspace(0.0, float(cam.theta_max), THETA_BINS + 1)
    t_idx = np.clip(np.digitize(theta.numpy(), t_edges) - 1, 0, THETA_BINS - 1)
    t_mid = 0.5 * (t_edges[:-1] + t_edges[1:]) * 180 / np.pi

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device=a.device, variant="small")
    bb.install(None, cam, (h, w), patch_undistort=False, border_token=False,
               dpt_grid=False, depth_convention="range")
    net = bb.model if hasattr(bb, "model") else bb
    n = load_lora(net, a.ckpt)

    field = ck["field"].to(a.device)
    vit = bb._vit()
    blocks = getattr(vit, "blocks", None) or getattr(vit, "layers")
    dim = int(next(blocks[-1].parameters()).shape[-1])
    film = FiLMConditioner(3, dim, hidden=ck["config"]["film_hidden"]).to(a.device)
    film.load_state_dict(ck["film"])
    film.eval()
    film_on = {"v": True}

    def hook(_m, _i, out):
        if not film_on["v"]:
            return out
        t = out[0] if isinstance(out, tuple) else out
        t2 = film(t, field)
        return (t2,) + tuple(out[1:]) if isinstance(out, tuple) else t2
    blocks[-1].register_forward_hook(hook)
    print(f"[h12/{arm}] LoRA into {n} layers, FiLM field {tuple(field.shape)}")

    nb_d = len(GT_DEPTH_EDGES) - 1
    tables, counts = {}, np.zeros((THETA_BINS, nb_d))
    for a_name in ("before", "after"):
        film_on["v"] = (a_name == "after")
        s_ = np.zeros((THETA_BINS, nb_d)); n_ = np.zeros((THETA_BINS, nb_d))
        for k in s.frames:
            with torch.no_grad():
                if a_name == "before":
                    with lora.lora_disabled(net):
                        pred = bb.forward(s.src.image(k)[None, None].to(a.device))
                else:
                    pred = bb.forward(s.src.image(k)[None, None].to(a.device))
            d = pred.depth[0].cpu().numpy()
            gr = s.gt_range(k, cos_t).numpy()
            valid = (cone.numpy() & (gr > 0) & (gr <= a.depth_max_m) & (d > 1e-6))
            if valid.sum() < 1000:
                continue
            aligned = align_depth(d, gr, valid, mode="scale_shift")
            absrel = (np.abs(aligned - gr) / np.clip(gr, 1e-6, None))[valid]
            ti = t_idx[valid]
            di = np.clip(np.digitize(gr[valid], GT_DEPTH_EDGES) - 1, 0, nb_d - 1)
            flat = ti * nb_d + di
            s_ += np.bincount(flat, weights=absrel,
                              minlength=THETA_BINS * nb_d).reshape(THETA_BINS, nb_d)
            n_ += np.bincount(flat, minlength=THETA_BINS * nb_d
                              ).reshape(THETA_BINS, nb_d)
        tables[a_name] = np.divide(s_, n_, out=np.zeros_like(s_), where=n_ > 0)
        counts = n_

    zones: Dict[str, Dict] = {}
    for zname, cells in {
        "near_rim(<=2m,>=38deg)": [(i, j) for i in range(THETA_BINS) for j in range(nb_d)
                                   if t_mid[i] >= 38 and GT_DEPTH_EDGES[j + 1] <= 2.0],
        "near_center(<=2m,<=11deg)": [(i, j) for i in range(THETA_BINS) for j in range(nb_d)
                                      if t_mid[i] <= 11 and GT_DEPTH_EDGES[j + 1] <= 2.0],
        "center(<=11deg)": [(i, j) for i in range(THETA_BINS) for j in range(nb_d)
                            if t_mid[i] <= 11],
        "far(>=3m)": [(i, j) for i in range(THETA_BINS) for j in range(nb_d)
                      if GT_DEPTH_EDGES[j] >= 3.0],
    }.items():
        wgt = np.array([counts[i, j] for i, j in cells], float)
        if wgt.sum() == 0:
            continue
        b = float((np.array([tables["before"][i, j] for i, j in cells]) * wgt).sum() / wgt.sum())
        af = float((np.array([tables["after"][i, j] for i, j in cells]) * wgt).sum() / wgt.sum())
        zones[zname] = {"before": b, "after": af}
        print(f"  {zname}: {b:.4f} -> {af:.4f} ({(af - b) / b * 100:+.2f}%)")

    res = {"arm": arm, "seq": a.seq, "ckpt": a.ckpt, "zones": zones,
           "before": tables["before"].tolist(), "after": tables["after"].tolist(),
           "counts": counts.tolist(), "theta_bin_mid_deg": t_mid.tolist(),
           "config": vars(a)}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(f"[h12/{arm}] wrote {a.out}")


if __name__ == "__main__":
    main()
