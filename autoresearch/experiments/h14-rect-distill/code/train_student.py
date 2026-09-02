"""H14: distil the frozen backbone's PINHOLE answer back into the fisheye domain.

Three arms, identical in architecture, parameter count, data, seed, optimiser
and loss form. They differ only in what the student is asked to match:

    --arm rect             teacher run in ONE co-axial 95 deg view      (NO GT)
    --arm rect_ring        teacher run in a centre view + a ring of eight
                           tangentially elongated views (H14.2)         (NO GT)
    --arm roundtrip        the SAME teacher on the raw fisheye, put through the
                           single view's resampling                     (NO GT)
    --arm roundtrip_ring   likewise, through the ring's resampling      (NO GT)
    --arm gt               the ground-truth depth map                (reference)

`rect` sees only ~70% of the near-rim zone (a 95 deg co-axial view cannot reach
further without going black at the corners); `rect_ring` sees 99.3% of it, at a
lower average teacher gain because the 29% it adds is the hardest outermost
annulus. Which makes the better STUDENT is the question H14.2 asks.

`roundtrip` is the control that decides the experiment; `gt` is the ceiling and
is exactly the plain-LoRA row that already beat every rim-targeted method in
#35 (near-rim -83.5% on seq136), so the label-free arms can be read as a
fraction of what labels buy.

The adapter is plain LoRA on the same four MLP blocks as H5/H12, at the same
rank -- deliberately nothing new. The novelty under test is the SUPERVISION,
not the parameterisation, and re-using the standing baseline's parameter set is
what makes the comparison against it legitimate.

WHY THIS MIGHT WORK, IN ONE LINE
--------------------------------
Ticket 024A: the controlled rim/centre AbsRel ratio is 1.25-1.81x on raw
fisheye and ~1.0 on rectified input. The information is there; the model cannot
read it through a fisheye. So stop designing a mechanism to make it read better
and hand it its own pinhole answer as a target.

WHAT WOULD MAKE A WIN UNINTERESTING, AND WHY IT CANNOT HAPPEN QUIETLY
--------------------------------------------------------------------
A student that merely learned a global rescale of the teacher would show
nothing: the eval of record aligns scale AND shift per frame before binning, so
an affine gain is invisible by construction. And the target is scale-aligned to
the frozen model's own output per frame before training (`--scale-align`,
default on), so the capacity is not spent on that constant in the first place.
For `roundtrip` that alignment is a no-op by construction -- the manifest's
`log_offset_median` says so -- which is the check that it is not doing the work.

Usage (box):
    python .../h14-rect-distill/code/train_student.py --arm rect \\
      --train-seqs <4 clean seq dirs> --cache-root results/h14-teacher \\
      --epochs 20 --out-dir results/autoresearch-h14-rect/rect
"""
from __future__ import annotations

import argparse
import json
import os
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
sys.path.insert(0, str(_HERE))

import importlib.util as _ilu  # noqa: E402
import losses  # noqa: E402
import lora  # noqa: E402


def _load(name, path):
    sp = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


_H5 = _HERE.parents[1] / "h5-rim-finetune" / "code"
Seq = _load("h5_train", _H5 / "train.py").Seq

#: The same four MLP blocks H5 and H12 adapt. Not a tuning knob here.
LORA_PATTERNS = [r"backbone\.pretrained\.blocks\.(8|9|10|11)\.mlp\.fc[12]$"]


class TeacherCache:
    """Per-sequence cache written by `cache_teacher.py`.

    Frames are addressed by STEM, not by index: the trainer and the cache
    build their frame lists independently, and an index would silently pair a
    frame with another frame's depth if either selection ever changed. Every
    stem the trainer asks for must be present, or the run stops.
    """

    def __init__(self, root: str, arm: str, seq_name: str, scale_align: bool):
        self.dir = Path(root) / arm / seq_name
        man = self.dir / "manifest.json"
        if not man.exists():
            have = sorted(q.name for q in (Path(root) / arm).glob("*")
                          if q.is_dir()) if (Path(root) / arm).exists() else []
            raise SystemExit(
                f"[h14] no teacher cache at {self.dir} -- run cache_teacher.py "
                f"--arm {arm} for {seq_name} first. Present under "
                f"{Path(root) / arm}: {have or 'nothing'}. The key is the FULL "
                f"sequence directory name, which is what Seq.name gives.")
        self.manifest = json.loads(man.read_text())
        if self.manifest["arm"] != arm:
            raise SystemExit(f"[h14] cache at {self.dir} says arm "
                             f"{self.manifest['arm']!r}, asked for {arm!r}")
        if self.manifest.get("used_gt_for_targets", True):
            raise SystemExit("[h14] this cache was built with GT targets; the "
                             "label-free claim would be false.")
        self.covered = torch.from_numpy(np.load(self.dir / "covered.npy"))
        self.offsets = self.manifest.get("log_offset_vs_raw", {})
        self.scale_align = scale_align
        self._cache: Dict[str, torch.Tensor] = {}

    def target(self, stem: str) -> torch.Tensor:
        if stem not in self._cache:
            f = self.dir / "npz" / f"{stem}.npz"
            if not f.exists():
                raise SystemExit(
                    f"[h14] frame {stem} is not in the teacher cache "
                    f"{self.dir}. The trainer's frame selection and the "
                    f"cache's have diverged; re-run cache_teacher.py with the "
                    f"same --size/--max-frames.")
            d = torch.from_numpy(np.load(f)["depth"].astype(np.float32))
            if self.scale_align:
                # Remove the teacher's global scale relative to the frozen
                # model's own output. The eval removes it anyway; spending LoRA
                # capacity on a per-frame constant would only dilute the signal.
                d = d * float(np.exp(-self.offsets.get(stem, 0.0)))
            self._cache[stem] = d
        return self._cache[stem]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True,
                   choices=("rect", "rect_ring", "roundtrip",
                            "roundtrip_ring", "gt"))
    p.add_argument("--train-seqs", required=True)
    p.add_argument("--cache-root", default=None,
                   help="required for the two label-free arms")
    p.add_argument("--size", type=int, default=504)
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--depth-max-m", type=float, default=10.0)
    p.add_argument("--scale-align", dest="scale_align", action="store_true",
                   default=True)
    p.add_argument("--no-scale-align", dest="scale_align", action="store_false")
    p.add_argument("--variant", default="small")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)

    if a.arm != "gt" and not a.cache_root:
        raise SystemExit("[h14] --cache-root is required for a label-free arm")

    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    seqs = [Seq(s.strip(), a.size, a.max_frames)
            for s in a.train_seqs.split(",") if s.strip()]
    print(f"[h14/{a.arm}] {len(seqs)} sequences, "
          f"{sum(len(s.frames) for s in seqs)} frames")

    caches = ({s.name: TeacherCache(a.cache_root, a.arm, s.name, a.scale_align)
               for s in seqs} if a.arm != "gt" else {})

    from raytun3r.backbones import build_backbone
    bb = build_backbone("da3", weights="pretrained", device=a.device,
                        variant=a.variant)
    cam = seqs[0].src.camera
    h = w = a.size
    bb.install(None, cam, (h, w), patch_undistort=False, border_token=False,
               dpt_grid=False, depth_convention="range")
    net = bb.model if hasattr(bb, "model") else bb
    hits = lora.inject(net, LORA_PATTERNS, r=a.lora_r, alpha=2 * a.lora_r)
    assert hits, "LoRA matched nothing"

    theta = cam.incidence_grid(h, w)
    cone = (theta <= cam.theta_max).to(a.device)
    cos_t = torch.cos(theta)
    theta_d = theta.to(a.device)

    n_lora = sum(x.numel() for x in lora.lora_parameters(net))
    print(f"[h14/{a.arm}] LoRA {n_lora / 1e3:.1f}k params, "
          f"scale_align={a.scale_align}")

    params = list(lora.lora_parameters(net))
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
            # The label-free claim lives in this branch and nowhere else:
            # `s.gt_range` is reachable only under --arm gt.
            if a.arm == "gt":
                target = s.gt_range(n, cos_t).to(a.device)
                valid = cone & (target > 0) & (target <= a.depth_max_m)
            else:
                c = caches[s.name]
                target = c.target(s.stem(n)).to(a.device)
                valid = (cone & c.covered.to(a.device) & (target > 1e-6)
                         & (target <= a.depth_max_m))
            if not bool(valid.any()):
                continue
            # alpha=0: the PLAIN log-L1. Rim weighting is what lost to its own
            # control in H5, and importing it here would confound the question.
            loss = losses.depth_loss(pred, target, valid, theta_d, alpha=0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss); cnt += 1
        rec = {"epoch": ep, "depth": tot / max(cnt, 1), "n": cnt,
               "sec": round(time.time() - t0, 1)}
        log.append(rec)
        print(f"[h14/{a.arm}] ep{ep:02d} depth {rec['depth']:.4f} "
              f"({cnt} steps, {rec['sec']}s)", flush=True)
        # Collected from the LoRALinear modules, as H5 and H12 do. A
        # state_dict filter on "lora_" matches NOTHING here (the tensors are
        # named .A/.B) and saved an empty checkpoint in the H12 pilot, caught
        # only by auditing tensor norms.
        state = {name: {"A": m.A.detach().cpu(), "B": m.B.detach().cpu()}
                 for name, m in hits}
        assert state, "no LoRA tensors collected -- refusing to save an empty checkpoint"
        torch.save({"lora": state, "patterns": LORA_PATTERNS,
                    "arm": a.arm, "epoch": ep, "config": vars(a),
                    "used_gt": a.arm == "gt"}, out / "lora_last.pt")
        (out / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"[h14/{a.arm}] done -> {out}")


if __name__ == "__main__":
    main()
