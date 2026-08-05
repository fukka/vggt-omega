# Standing brief: CPU-Claude

Read this first, then your ticket. You are the account **without** access to the
GPU box. Everything you need is in this repo.

## Setup

```bash
git clone https://github.com/fukka/vggt-omega.git && cd vggt-omega
git checkout organized
```

Deps for the test suite only (no GPU, no weights, no datasets):

```bash
pip install "torch>=2.0" torchvision numpy Pillow einops safetensors opencv-python pytest
```

Optional, and worth it — it unlocks the DA3-Small test, i.e. the paper's primary
backbone. The package declares `torch>=2.10`, but only its `xformers`/serving
extras need that; `--no-deps` plus the three real imports works on torch 2.2:

```bash
pip install --no-deps depth-anything-3 && pip install omegaconf addict einops
```

`depth_anything_3.api` additionally pulls in `moviepy`; `DA3Backbone.load(weights=None)`
avoids it by building from the config registry directly.

## Your verification loop

This is the point of the split — you can verify almost everything locally:

```bash
python -m pytest raytun3r/tests -q
```

```bash
python raytun3r/smoke_test.py
```

46 tests + 35 checks, ~25 s total, CPU-only. They cover camera round-trips, the
paper's pinhole-bias premise, zero-init adapters being exact no-ops, gradients
reaching every table while the backbone stays frozen, loss/metric invariants,
MAGSAC++ pose recovery, the depth-convention pairing, and — when
`depth_anything_3` is installed — DA3-Small hooks against the real package.
**A change that passes both is ready for a GPU run.**

They do *not* cover: real weights, real data, or any number in the paper. Never
claim a paper result is reproduced — that is GPU-Claude's evidence to produce.

## What you cannot do

* Run anything on `lambda_63`, or read ADT / ScanNet++. Those paths exist on one
  machine and you are not on it.
* Produce `R°`, `t°`, `d_reproj`, `AbsRel`, `δ₁.₂₅`. You may only *interpret*
  numbers already committed to the `results` branch.

If your ticket turns out to need the box, stop, relabel it `gpu`, and comment
what you got to.

## Handing back

```bash
git add <only the files your ticket named> && git commit && git push origin organized
gh issue comment <N> --body "pushed <sha>; pytest+smoke green; needs GPU run: <yes/no>"
gh issue edit <N> --add-label gpu --remove-label cpu   # only if a run is needed
```

## Repo orientation

`raytun3r/` is a from-scratch reproduction of *RayTun3R: Online Camera Adaptation
in 3D Foundation Models* ([arXiv:2607.02711](https://arxiv.org/abs/2607.02711)).
No official code exists; everything is reconstructed from the paper text.
`raytun3r/README.md` has the paper→code map, the interpretation decisions, and
the discrepancies found in the paper itself. Read that before changing anything —
several surprising choices are deliberate and documented there.

Module map:

| file | holds |
|---|---|
| `cameras.py` | KB4 / EUCM / pinhole, `theta_max`, `Omega` |
| `adapter.py` | Eq. 5, 6 — the only trainable parameters |
| `corrections.py` | the three parameter-free corrections |
| `backbones.py` | hook points on a frozen VGGT / VGGT-Omega / DA3 |
| `losses.py` | Eq. 7–13 |
| `metrics.py` | Eq. 14–18 |
| `data.py` | ScanNet++ and ADT loaders, window construction |
| `matching.py` | UFM, MAGSAC++ pose target |
| `experiments/` | full-dataset and ablation drivers (`scannetpp_all`, `fov_sweep`) |

## Two invariants that are easy to break

* **Depth convention.** Backbones declare `native_depth`, `install()` converts
  once at the boundary, `Prediction` carries the tag, consumers call
  `require_convention`. Never convert depth ad hoc in a new call site — that is
  precisely how the direct path and the pinhole baselines silently diverged.
* **Ω is `camera.valid_mask`,** which is `theta_max`. Anything that changes
  `theta_max` changes every loss and every metric at once. That is deliberate
  (`--max-fov` uses it), but it means a "harmless" camera tweak is never local.

## House style

Match the surrounding code: dense explanatory docstrings that say *why*, with
paper equation numbers cited inline. When you fix a deviation from the paper,
state in the docstring what the paper says, what the code did, and what breaks —
that is the convention here and it is what makes the reproduction auditable.
