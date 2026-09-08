# H35 — for the model you would actually ship, roll costs 0.46%.

Dense grid, four backbones, six recordings, integrated against H17's 60,105
frames. All four measured on the *same* recordings, so the columns are directly
comparable.

| backbone | expected cost | ±20° penalty | ±30° penalty | tail share |
|---|---|---|---|---|
| DA3-Small | **1.80% ± 1.31** | +14.0% | +42.0% | **23.0%** |
| DA3-Large | 1.85% ± 1.33 | +9.3% | +27.8% | 14.9% |
| VGGT | **0.34% ± 0.23** | +2.9% | +7.0% | 17.1% |
| VGGT-Omega | **0.46% ± 0.25** | +2.6% | +4.8% | **9.8%** |

## B1 passes — and H17's gap survives integration almost intact

VGGT-Omega costs **0.46%** against DA3-Small's **1.80%** — a ratio of **0.25**,
i.e. **4× cheaper**, against a bar of "under half".

I expected the integral to compress H17's 4–5× gap, because it weights the flat
low-angle region far more heavily than the 30° comparison H17 used. **It barely
compresses at all.** The sensitivity gap between pretraining families is not an
artefact of looking at large angles.

**B2 passes**: DA3-Small reads 1.80% here against H32's 1.46% ± 1.09 on thirteen
recordings — inside the spread, so this six-recording subset and the runner agree
with H32.

## B3 — the tail advice is DA3-specific, and narrows

H32 found that frames beyond ±20°, which are 1.48% of the data, carry **26%** of
the expected cost, and both reports turned that into "if you spend effort on
roll, spend it on the rare large tilt".

That holds for DA3-Small (**23.0%** here) and **not** for VGGT-Omega
(**9.8%**) — a 13-point gap. The multi-view curves are flat enough that the rare
tilt stops being disproportionate.

**So the tail advice narrows to DA3-like models.** For a multi-view-pretrained
model there is no tail worth handling, because there is barely any cost to
begin with.

## Bigger is not the axis

DA3-Large costs **1.85%**, statistically identical to DA3-Small's 1.80%, despite
being far better at large angles (+27.8% against +42.0% at 30°). The integral is
dominated by the region under 15°, where the two are alike.

**Scaling the single-image model does not reduce the roll cost you would
actually pay.** Changing the pretraining does, by 4×. That is the same shape as
§4.1's finding that a bigger model does not fix the rim penalty either.

## The statement this line can now close on

> On ordinary indoor footage, ignoring head roll costs **0.3–0.5%** for a
> multi-view-pretrained model and **~1.8%** for a single-image one. Both are
> cheap. If you are on a single-image model and care, the rare tilt beyond ±20°
> is where a quarter of your cost is; on a multi-view model there is nothing
> there worth chasing.

That is the deployment number the line has been missing, and it is measured
rather than inferred from three separate quantities.
