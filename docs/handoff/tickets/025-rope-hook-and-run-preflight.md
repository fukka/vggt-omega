# Pre-flight for the reproduction: five things only the box can confirm

**Owner:** gpu
**Files I may touch:** nothing — runs and one issue comment. No code changes.
**Blocked by:** none. **Blocks:** #4 step 2 (the adapter runs).

## Goal

Five checks, all cheap, that stand between the current code and any number from
#4 being worth recording. Four of them are new because of the paper↔code audit on
`organized`; one is an environment question that has been silently wrong here.

None of this needs training. Budget ~20 minutes of GPU time total.

## Context

The audit (`docs/research/raytun3r-vs-vggt-training.zh-CN.md`) found and fixed six
defects in the adaptation loop, two of which change every number a fit produces.
The local suite is green — **73 tests + 33 smoke checks** — but three claims cannot
be tested on a machine without `depth_anything_3`, `pi3`, or a GPU, and they are
exactly the claims that decide whether #4's DA3 and π³ rows mean anything.

The two that change results:

* **Eq. 12 was computing `TV(residual)`, not `TV(P_A + residual)`.** `pe_table()`
  was read once before the loop, i.e. before any forward had run the hook that
  captures `P_A`, so it was `None` for the whole fit — on every backbone that has
  an absolute PE table. `w_TV = 20` is the largest weight in Eq. 13.
* **Eq. 6 reached only the last frame in global attention.** VGGT hands one
  `RotaryPositionEmbedding2D` to both the frame blocks and the global blocks; the
  hook inferred the token prefix as `N − gh·gw`, which on the global layout lands
  exactly on the last frame's patch block. Frames 0..S−2 — including frame 0, the
  reference frame — got no correction in any of the 24 global blocks.

**Any `adapter.pt` or `train_log.json` produced before this commit is void.** The
TV term alone shifts the objective by the constant `TV(P_A)` plus a cross term, so
old and new loss curves are not comparable either.

## Steps

### 1. DA3's RoPE layout — the one with real risk

`DA3Backbone` sets `has_rope = True` and its comment claims the module is
`RotaryPositionEmbedding2D`. Nothing on the Mac can check that:
`test_da3_hooks_fire_on_the_real_package` **skips** without the package, so DA3
has zero coverage of the reworked hook.

```bash
python -m pytest raytun3r/tests -q -k "da3 or rope"
```

Then, on the real package, confirm the hook sees the layout it thinks it does:

```bash
python - <<'EOF'
from raytun3r.backbones import DA3Backbone, Backbone
bb = DA3Backbone.load(weights="pretrained", variant="small", device="cuda")
seen = []
for m in bb.model.modules():
    n = type(m).__name__
    if "rope" in n.lower() or "rotary" in n.lower():
        seen.append(n)
print("RoPE classes present:", sorted(set(seen)))
EOF
```

* Classes are `RotaryPositionEmbedding2D` → good, continue.
* Anything else → **stop and report the name.** `install()` now raises rather than
  silently applying no correction, so this is a hard failure, not a degradation.
  The fix is one line in `_ROPE_TOKEN_MODULES` (`raytun3r/backbones.py`), but it is
  a `cpu` ticket — do not edit code on the box.

Second risk on the same path: if DA3 calls its RoPE as `rope(q, positions=pos)`
(keyword, not positional) the hook cannot see `pos`, DA3 declares no
`n_prefix_tokens`, and a multi-frame call raises at the first forward. The
traceback will say `n_prefix_tokens` — report it verbatim if you see it.

### 2. π³'s RoPE class name

Same question, unverified. `"RoPE2D"` was added to the token-RoPE set on the
CroCo/DUSt3R naming convention alone.

```bash
python -c "
from raytun3r.backbones import Pi3Backbone
bb = Pi3Backbone.load(weights='pretrained', device='cuda')
print(sorted({type(m).__name__ for m in bb.model.modules()
              if 'rope' in type(m).__name__.lower()}))"
```

If it is `cuRoPE2D` (the compiled kernel), **stop** — that variant rotates tokens
in place and a returned-value hook is the wrong shape for it.

Before this commit π³ had *no* RoPE hook at all: neither name matched, so its 20
`RadialRoPE` parameters sat in the optimiser receiving no gradient for the whole
fit. **Every previous `--backbone pi3` number is a partial-adapter number.**

### 3. OpenCV version — this one has been wrong quietly

MAGSAC++ (`cv2.USAC_MAGSAC`) and the `maxIters` kwarg both arrived in OpenCV 4.5.
On anything older the code now falls back to plain RANSAC **and warns**. The paper
specifies MAGSAC++ for the Eq. 9 pose target.

```bash
python -c "import cv2; print(cv2.__version__, hasattr(cv2,'USAC_MAGSAC'))"
```

`False` → `pip install -U opencv-python` in the run env before anything in #4, and
say in the report which version produced the numbers.

### 4. Gradient checkpointing: inert, and worth it?

Checkpointing is now **on by default** during the fit. The argument that it is
numerically inert is structural (no dropout, no droppath, no BatchNorm anywhere in
`vggt_visfeat`, and the heads never read `self.training`) and is verified against
the tiny model, not against VGGT-1B. One A/B settles it:

```bash
python -m raytun3r.train --backbone vggt --weights pretrained \
  --dataset scannetpp --path /netapp/datasets/f.zhang2/scannetpp/data/3f15a9266d \
  --windows 2 --iters 3 --seed 0 --out /tmp/ckpt-on
python -m raytun3r.train --backbone vggt --weights pretrained \
  --dataset scannetpp --path /netapp/datasets/f.zhang2/scannetpp/data/3f15a9266d \
  --windows 2 --iters 3 --seed 0 --no-grad-checkpointing --out /tmp/ckpt-off
diff <(jq -c '.history[].total' /tmp/ckpt-on/train_log.json) \
     <(jq -c '.history[].total' /tmp/ckpt-off/train_log.json) && echo IDENTICAL
```

Report the peak memory of each (`nvidia-smi --query-gpu=memory.used`) — that is the
number that decides what `--seq-len` / resolution is now affordable. Configurations
that OOM'd before may fit now; the Eq. 6 fix pushes the other way, since global
blocks now rotate `S×` as many tokens.

### 5. The matcher coverage on real data — report it, do not act on it

Eq. 8 divides by `|Ω|`, not by `sum(w)`, so `L_reproj`'s weight against
`w_smooth=10 / w_L2=2 / w_TV=20` is set by how much of the disc the matcher is
confident about. `fit_adapter` now measures that, prints it, writes it to
`train_log.json`, and **refuses to run below 5%**.

The step-4 run prints it. Just paste the line:

```
[fit] matcher=ufm  coverage: mean sum(w)/|Omega| = 0.NNN over 2 windows
```

UFM should be well above 0.2. If it is not, that is a finding about UFM on
fisheye, not a reason to pass `--allow-sparse-matcher`.

## Done when

- [ ] DA3's RoPE class names reported; `install()` completes without raising
- [ ] π³'s RoPE class names reported
- [ ] `cv2.__version__` and `USAC_MAGSAC` availability reported
- [ ] checkpointing A/B: loss curves identical, both peak-memory figures given
- [ ] UFM coverage line pasted
- [ ] issue commented; relabel `cpu` if any of 1–3 needs a code change

## Needs CPU-Claude afterwards?

Only if a RoPE class name does not match, or if the checkpointing A/B is not
bit-identical. Otherwise this closes and #4 step 2 is unblocked.
