# Reproduce the paper's VGGT vanilla, and thereby validate the whole harness

**Owner:** gpu
**Files I may touch:** nothing under `raytun3r/` — runs only. Results to `results`.
**Blocked by:** none. **Blocks:** #4 (ticket 003) — do that one after this passes.

## Goal

`vanilla` on ScanNet++ `3f15a9266d` with VGGT matches the paper's **7.21° / 16.6°
/ 39.4**, under some stated protocol — or we know it cannot, and by how much.

## Why this comes before any adapter work

`vanilla` is the only number in the paper with **no adapter, no training, no
matcher, no randomness**: a frozen backbone on a posed pair. So it isolates
exactly the things we are unsure of — the data loader, the camera model, the pose
convention, the metric. If it matches, all four are validated at once and every
later comparison is trustworthy. If it does not, no adapter result can be
interpreted, because we would be measuring a different thing from the paper.

We have twice drawn conclusions from adapter numbers and twice had to withdraw
them. This stops that.

## The one command

```bash
python -m raytun3r.experiments.vanilla_repro --backbone vggt --weights pretrained --path /netapp/datasets/f.zhang2/scannetpp/data/3f15a9266d --out runs/vanilla-repro/3f15a9266d-vggt.json
```

Then **paste the final table and the `[repro] best:` line.** That is the whole
deliverable. The script names the verdict itself; no analysis needed on your side.

Expect a few minutes: pose needs no correspondences, so it runs without UFM.

## What it does, so you can sanity-check it rather than read it

The paper under-specifies the protocol in four ways that all move `R°`, and we
have been guessing them one at a time across several round-trips. This enumerates
them in one run — stage 1 sweeps `stride`, stage 2 varies the rest at the two
closest strides:

| axis | why it is open |
|---|---|
| `stride` | not a paper concept; it says "consecutive image pairs", but a 1.09 cm baseline makes stride 1 nearly static, and `R°` is an absolute angle whose scale is set by how much rotation there is to estimate |
| `is_bad` | ScanNet++ flags 143 of 896 frames here as unusable; the paper never says whether it honours the flag |
| resolution | "maximum patch-aligned resolution of 504 × 504" reads as either a 504 cap on the long side (→ 504×336) or a square 504×504 |
| `seq_len` | these are multi-view models, so a pair's prediction depends on what else is in the window |

## Reading the result

* **Some configuration lands within ~1° of 7.21** → that is the paper's protocol.
  Say which, and ticket 003 gets re-run under it. Harness validated.
* **Nothing comes close** → the script says so explicitly. That is a real finding,
  not a failure: it means the gap is *not* the evaluation protocol, and the
  backbone or its preprocessing is the next suspect. Equally valuable, and it
  saves ticket 003's GPU time.

Do not run anything else on this ticket. If the result suggests an obvious
follow-up, put it in a comment rather than running it — the point is one cheap
round-trip.

## Scope note: only VGGT can be checked this way

The paper gives exactly one named-scene vanilla number for a backbone we have:
Tab. 2, VGGT, ScanNet++ `3f15`. `π³` has one (6.17 / 19.7 / 38.6) but is not
implemented here. **DA3-Small has none** — Tab. 1's ScanNet++ row is a mean over
scenes the paper never names, and Tab. 6's DA3 baseline is ETH3D. So DA3-S vanilla
cannot be validated per-scene; it inherits whatever protocol this run establishes.

## Recording

`results/vanilla-repro-3f15a9266d/` with the JSON and a `meta.json` carrying
`git rev-parse HEAD`, torch version, and the VGGT checkpoint id.

## Done when

- [ ] the table and `[repro] best:` line are pasted into the issue
- [ ] a one-line verdict: which protocol reproduces 7.21°, or that none does
- [ ] pushed to `results`; hand back to `cpu`

## Needs CPU-Claude afterwards?

yes — deciding what ticket 003 becomes, which depends entirely on this answer.
