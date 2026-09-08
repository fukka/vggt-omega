"""H37 — four-backbone border ordering across fourteen recordings.

Reads H29's da3 arms and H37's vggt arms, which used the same runner, the same
recordings and the same single level angle, so the border costs are directly
comparable. Border cost = pinhole_masked / pinhole - 1 on all-image AbsRel.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
H29 = HERE.parents[0] / "h29-border-variance" / "results"
H37 = HERE / "results"
SEQS = ["seq136"] + [f"seq{n}" for n in (137, 138, 140, 141, 142, 143,
                                         144, 145, 146, 147, 148, 149, 150)]
MODELS = ["da3:small", "da3:large", "vggt", "vggt_omega"]


def cost(path: Path, model: str) -> float | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    m = d["models"].get(model)
    if m is None:
        return None
    a = m["pinhole"]["0.0"]["all"]
    b = m["pinhole_masked"]["0.0"]["all"]
    return 100.0 * (b / a - 1.0)


rows: dict[str, dict[str, float]] = {}
for s in SEQS:
    r = {}
    for m in MODELS:
        src = H29 if m.startswith("da3") else H37
        v = cost(src / f"models_{s}.json", m)
        if v is not None:
            r[m] = v
    if r:
        rows[s] = r

held = [s for s in SEQS if s != "seq136" and len(rows.get(s, {})) == 4]
print(f"[h37] {len(rows)} recordings with data; {len(held)} complete held-out\n")

hdr = f"{'seq':<9}" + "".join(f"{m:>13}" for m in MODELS)
print(hdr)
for s in SEQS:
    r = rows.get(s, {})
    line = f"{s:<9}" + "".join(
        f"{r[m]:>+12.1f}%" if m in r else f"{'--':>13}" for m in MODELS)
    print(line + ("   <- published" if s == "seq136" else ""))

print("\n--- thirteen held-out recordings ---")
stats = {}
for m in MODELS:
    v = [rows[s][m] for s in held]
    mean, sd = st.mean(v), st.stdev(v)
    z = (rows["seq136"][m] - mean) / sd
    stats[m] = dict(mean=mean, sd=sd, z=z, min=min(v), max=max(v))
    print(f"{m:<12} {mean:>+8.1f}% +- {sd:5.1f}   seq136 "
          f"{rows['seq136'][m]:>+8.1f}%  z={z:+.2f}   "
          f"range {min(v):+.1f} .. {max(v):+.1f}")

# --- the bars ---
worst = sum(1 for s in held
            if rows[s]["da3:large"] == max(rows[s][m] for m in MODELS))
best = sum(1 for s in held
           if rows[s]["vggt_omega"] == min(rows[s][m] for m in MODELS))
maj = len(held) // 2 + 1
neg = [(s, m) for s in held for m in MODELS if rows[s][m] <= 0]

print(f"\nB1  da3:large worst of four on {worst}/{len(held)} "
      f"(majority {maj}) -> {'PASS' if worst >= maj else 'FAIL'}")
print(f"B2  vggt_omega best of four on {best}/{len(held)} "
      f"-> {'PASS' if best >= maj else 'FAIL'}")
b3 = all(stats[m]["z"] > 0 for m in ("vggt", "vggt_omega"))
print(f"B3  seq136 reads high out of sample: vggt z={stats['vggt']['z']:+.2f}, "
      f"vggt_omega z={stats['vggt_omega']['z']:+.2f} -> "
      f"{'PASS' if b3 else 'FAIL'}")
print(f"sanity  border cost positive everywhere: "
      f"{'OK' if not neg else 'VOID -> ' + str(neg[:5])}")

out = H37 / "summary.json"
out.write_text(json.dumps(
    dict(rows=rows, held_out=held, stats=stats,
         bars=dict(b1_worst=worst, b2_best=best, b3_pass=b3,
                   majority=maj, negative_cells=neg)), indent=2))
print(f"\n[h37] wrote {out}")
