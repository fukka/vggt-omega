# Two leftovers from #25: DA3's RoPE hook won't install, and checkpointing isn't bit-identical

**Owner:** gpu (was cpu; fixed, needs the GPU re-check below)
**Files I may touch:** `raytun3r/backbones.py`, `raytun3r/train.py`, `raytun3r/tests/test_raytun3r.py`
**Blocked by:** none. **Blocks:** DA3's row in #4 step 2 (item 1 only; item 2 doesn't block a run, it blocks trusting the numbers).

## Goal

Two independent findings from running #25 on `lambda_63` at `organized@222d4a3`,
env `raytun3r` (torch 2.11.0+cu128, cv2 4.11.0). Full data in the
[#25 comment](https://github.com/fukka/vggt-omega/issues/25#issuecomment-5335209060).
Neither needs a GPU to diagnose or fix; both need a GPU to re-verify afterwards.

## Item 1 — DA3's RayTun3R hook install() raises, hard stop

`RoPE classes present: ['RotaryPositionEmbedding2D']` — the class name #25 was
checking for is right. But the real-package test fails anyway:

```
$ python -m pytest raytun3r/tests -q -k "da3 or rope"
FAILED raytun3r/tests/test_raytun3r.py::test_da3_hooks_fire_on_the_real_package

RuntimeError: DA3Backbone: 52 RoPE tokens on a 5x5 grid is more than one frame's
worth, the module was called without positions, and the backbone declares no
n_prefix_tokens. Refusing to guess: 'prefix = n - 25' aligns on the *last* frame
and silently leaves every earlier frame uncorrected.
```

DA3 calls its RoPE module with `positions` as a keyword argument, so the returned-
value hook never sees it, and `DA3Backbone` declares no `n_prefix_tokens` for the
multi-frame case. `install()` now raises rather than silently under-correcting
(that's #25's point 1 working as intended) — but it means DA3 cannot run the
`raytun3r` method at all right now. Per #25's own diagnosis this is a one-line fix
in `_ROPE_TOKEN_MODULES` in `raytun3r/backbones.py`; likely `DA3Backbone` just
needs to declare `n_prefix_tokens` (or the hook needs to read DA3's `positions`
kwarg the way #25's Eq. 6 fix already does for VGGT's positional call).

**Scope note:** DA3's `vanilla` and `center_ph` methods are unaffected — neither
installs this hook, and both ran cleanly in `raytun3r.eval` on `3f15a9266d`
(`runs/rt3r/3f15a9266d-da3/results_vanilla_centerph.json` on lambda_63, not yet
pushed to `results`). Only DA3 + `raytun3r` method is blocked.

## Item 2 — gradient-checkpointing A/B is not bit-identical on VGGT-1B

Same scene, `--backbone vggt --windows 2 --iters 3 --seed 0`, checkpointing on vs
`--no-grad-checkpointing`, otherwise identical:

```
diff <(jq -c '.history[].total' /tmp/ckpt-on/train_log.json) \
     <(jq -c '.history[].total' /tmp/ckpt-off/train_log.json)
1,3c1,3
< 6.319521903991699   6.189236640930176   5.866147518157959
---
> 6.347949981689453   6.216237068176270   5.862402915954590
```

Differences are ~0.01–0.03 on `total` at every one of the 3 iterations, and
`pose`/`|g|` diverge more (iter 0 `pose` 1.2049 vs 1.2333; iter 2 `|g|` 17.338 vs
14.541). That's larger and more consistent than float accumulation noise on truly
identical ops. The structural argument in #25's commit — no dropout, no droppath,
no BatchNorm anywhere in `vggt_visfeat`, heads never read `self.training` — is
still correct as far as it goes; something else is moving. Candidates worth
checking before assuming a repo bug: cuDNN algorithm selection differing between
the original and recomputed forward pass under `torch.utils.checkpoint`, or a
non-deterministic reduction (e.g. `atomicAdd`-based ops) whose result depends on
recomputation. Peak memory did behave as expected (checkpointing on: ~10.4 GB for
the process; off: ~20.4 GB — measured by polling `nvidia-smi` during the run,
delta over the other job's 26841 MiB baseline on that GPU), so the memory side of
the claim is fine; only the "numerically inert" half is in question.

## Steps

1. Read `raytun3r/backbones.py`'s `_ROPE_TOKEN_MODULES` / `DA3Backbone` and
   `install()` to see exactly how the hook infers prefix tokens and how DA3 calls
   its RoPE module (positional vs. `positions=`).
2. Fix item 1 — declare `n_prefix_tokens` for DA3 (matching how VGGT's Eq. 6 fix
   reads the layout off the `pos` tensor) or teach the hook to see DA3's keyword
   call. Add a case to `test_da3_hooks_fire_on_the_real_package` or a CPU-side
   fake-module test that pins the calling convention, so this can't regress
   silently again.
3. For item 2, read `raytun3r/train.py`'s checkpointing wiring and whatever
   `torch.utils.checkpoint` call it makes. Decide whether `use_reentrant=False`
   (or an equivalent determinism knob) closes the gap, or whether the
   non-determinism is upstream in `vggt_visfeat` and worth documenting rather
   than chasing.
4. `python -m pytest raytun3r/tests -q` and `python raytun3r/smoke_test.py`.

## Done when

- [ ] `test_da3_hooks_fire_on_the_real_package`-equivalent passes without a GPU
      (fake module) and the real-package version is described as fixed for GPU
      to re-check
- [ ] item 2 has a stated conclusion — either a fix, or a one-paragraph
      explanation of the remaining source of non-determinism and whether it's
      safe to ignore
- [ ] pushed to `organized`, issue commented with the sha

## Needs a GPU run afterwards?

yes → relabel `gpu` and ask for: (a) `pytest -k "da3 or rope"` green on the real
package on `lambda_63`, confirming `install()` no longer raises; (b) re-run the
checkpointing A/B command from #25 and confirm `diff ... && echo IDENTICAL`, or
report the new (hopefully smaller) discrepancy.


---

## RESOLVED (cpu, 2026-08-18) — both items, neither was what it looked like

### Item 1: not a keyword-argument problem, and not a missing declaration

Read `depth_anything_3`'s actual source
(`model/dinov2/vision_transformer.py::_prepare_rope`, `layers/attention.py`).
DA3 calls its RoPE **positionally** — `self.rope(q, pos)` — so the hook saw the
tensor all along. The real mechanism: DA3's alternating trunk hands the same
module real per-patch coordinates in **local** attention but `pos_nodiff` in
**global** attention — specials at `(0,0)` and *every patch at `(1,1)`*. Global
attention is spatially unencoded **by design** (identical rotations cancel in
q·k; the only information carried is special-vs-patch). The layout scan
correctly found no frame layout in that tensor, and the fallback then blamed
"called without positions" — the message lied about the cause.

The fix follows the mechanism: a call whose positions cannot address a grid
(fewer than three distinct coordinates) is **skipped**, not refused — Eq. 6
corrects RoPE where RoPE encodes patch position, and rotating `pos_nodiff`
would *introduce* spatial structure into a pathway the architecture defines as
position-free. Local-attention calls read their layout off `pos` as before and
get the correction; no `n_prefix_tokens` declaration is needed for DA3. The
hook is also now registered `with_kwargs=True`, so a package that really does
pass `positions=` by keyword works too. The discriminator deliberately runs on
the actual tensor every call, before the layout cache — at S=1 a local call and
a global call share `(n, g)`, and the cache must not alias them.

Tests: `test_rope_hook_skips_spatially_unencoded_positions` (DA3's exact nodiff
layouts, plus the S=1 cache-alias guard), `test_rope_hook_reads_keyword_positions`,
and `test_rope_hook_refuses_an_unreadable_multi_frame_layout` (the refusal now
needs positions that are present-but-unreadable, since keyword calls succeed).

### Item 2: checkpointing is (still) numerically inert — the A/B measured the matcher

The GPU's own numbers decide it: at iter 0, `total − pose` is **5.1146 in both
runs** (6.3195−1.2049 and 6.3479−1.2333, identical to print rounding). The
forward pass was bit-identical; the entire difference sat in the pose term,
i.e. in the **Eq. 9 MAGSAC target**, which is computed by `build_windows`
before the fit and outside any checkpointing. OpenCV's USAC solvers are
deterministic given identical inputs (fixed default `randomGeneratorState`), so
the variance is upstream: UFM's GPU forward is not run-to-run reproducible, and
MAGSAC's discrete minimal-set selection amplifies a tiny match perturbation
into a visibly different pose target. Continuous losses (reproj) absorb the
same perturbation invisibly. `|g|` then diverges because the pose gradients
differ, and everything after iter 0 compounds. **The two processes were never
running the same objective — and that has nothing to do with checkpointing.**

Direct evidence: with the windows held fixed, ckpt on/off is **bit-identical**
over a full fit (loss history and final adapter parameters), on CPU where
kernels are deterministic. Pinned as
`test_grad_checkpointing_is_bit_identical_on_identical_windows`.

New tooling: `train.py --windows-cache PATH` saves/loads the built windows
(images, matches, pose targets) so a GPU A/B can pin the fit's inputs.

### GPU re-check (relabel `gpu`)

1. `python -m pytest raytun3r/tests -q -k "da3 or rope"` on the real package —
   expect the previously-failing test green, `install()` no longer raises.
2. The #25 A/B, this time controlled:
   run 1 `--windows-cache /tmp/win3f15.pt` (builds + saves), run 2 same flag +
   `--no-grad-checkpointing` (loads). Expect `diff ... && echo IDENTICAL`, up to
   CUDA kernel nondeterminism (grid_sample backward is atomicAdd-based, so tiny
   iter≥1 drift is possible on GPU even at fixed inputs — iter 0 must match).
3. Optional, to see the thing #25's A/B actually measured: same config twice,
   *without* the cache — the drift should reappear with checkpointing held fixed.

## CLOSED (gpu re-check 2026-08-19, then one cpu fix)

Both items came back green from the box (issue #26 thread):

1. `pytest -k "da3 or rope"` on the real package: **9 passed** — including
   `test_da3_hooks_fire_on_the_real_package`, which used to raise.
2. Controlled A/B off one cached window set, VGGT, 3 iters: every forward-pass
   loss term at iter 0 (`reproj`, `pose`, `smooth`, `l2`, `tv`, `total`)
   **bit-identical to full float32 precision** between ckpt on and off. Only
   `grad_norm` differs (~0.03%), and only from iter 0 onward — a backward-pass
   quantity, consistent with `grid_sample`'s atomicAdd on CUDA. Checkpointing
   changes nothing numerically, as diagnosed.

The re-check also found a real bug in the new tooling: `--windows-cache`
loaded with `map_location="cpu"`, but `fit_adapter` takes its device from
`windows[0].images.device` (`train.py:102`), so the *loading* run silently
ran on cpu and died in `aggregator.forward` with a cuda/cpu mismatch. The box
worked around it with a standalone driver; fixed properly on cpu as
`map_location=args.device` (`ea55dd3`). Nothing else outstanding here.
