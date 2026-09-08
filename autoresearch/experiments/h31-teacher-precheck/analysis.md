# H31 — the method choice was sound. The verification campaign closes.

`results/*/manifest.json`, 42 pre-checks (3 configs × 14 recordings, 20 frames
each). Rim gain of the teacher against the raw fisheye model, same pixels.

## The two bars that decide it both pass unanimously

**B1 passes 13/13.** `omega_wide` beats `da3_wide` at the rim on **every**
recording. The bar was deliberately strict — a single failure would have
falsified it, because a method choice should not rest on a majority vote — and
it did not fail once.

**B2 passes 13/13.** `da3_wide` is **worse than doing nothing** on every
recording, +6.9% to +31.1%. The finding that forced the switch away from a DA3
teacher on the wide view holds everywhere.

| config | thirteen recordings | seq136 | published |
|---|---|---|---|
| DA3-Small, 95° narrow | −6.9% ± 6.2 | −9.9% | −15% |
| DA3-Small, 110° wide | **+19.8% ± 7.9** | +35.2% | +33% |
| VGGT-Omega, 110° wide | **−48.2% ± 7.2** | −69.1% | −65% |

**So §4.3's teacher was chosen correctly**, and the reason it was chosen — that
the wide view is unusable for DA3 and fine for VGGT-Omega — is now established on
thirteen recordings rather than one.

## A reproduction gap worth flagging

The two decision-critical arms reproduce closely on seq136 (+33 → +35.2,
−65 → −69.1). **The narrow arm does not**: published −15%, reproduced −9.9%.
The likely cause is the frame count — this run used 20 frames against the
original 60, and the narrow arm has the smallest effect and so the worst
signal-to-noise. It is recorded rather than smoothed over; nothing in the line
depends on the narrow number, but a 5-point reproduction gap should not go
unmentioned.

## B3 — the prediction holds, and the pattern gets its proper form

H30's standing instruction predicted seq136's `omega_wide` would read **more than
1 σ stronger** than the thirteen-recording mean. It reads **−2.91 σ** — the
strongest confirmation the instruction has had.

More usefully, H31 supplies two interventions with **opposite signs**, which
forces a better statement of the pattern. `da3_wide` *hurts* and seq136 reads
+1.95 σ **more harmful**; `omega_wide` *helps* and seq136 reads 2.91 σ **more
helpful**. So the rule is not "reads high" — it is:

> **seq136 amplifies the magnitude of whatever you do to it, in whichever
> direction the intervention points.**

All nine interventions measured across this campaign are amplified:

| | |z| | | | |z| |
|---|---|---|---|---|
| H25 radial curve | 5.74 | | H30 lens advantage | 1.29 |
| H26 adapter | 2.60 | | H31 `da3_wide` | 1.95 |
| H27 GT arm | 3.12 | | H31 `omega_wide` | 2.91 |
| H29 border, small | 3.32 | | H31 `da3_narrow` | 0.48 |
| H29 border, large | 5.96 | | | |

**Nine of nine in the amplification direction** — sign test p ≈ 0.002, median
|z| = 2.91 — against **three unperturbed measurements at −0.11, +0.07 and
+0.52** (H28). The split is now as clean as this data can make it.

Still an observation and **not a mechanism**. It says what that recording does,
not why.

## The campaign is closed

Nothing published in this line now rests on a single recording, except:

* the **rearranged-room** column (one recording, but H25 showed it is the
  representative one);
* the **cross-room** numbers, which were never measured on seq136 and are
  blocked on a data decision that has been put to the user and not acted on.
