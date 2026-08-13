"""Per-sequence pen for every cell -- item 3, does the shape hold across scenes.

The aggregated 6-sequence run cannot answer this: results.json carries one set of
bins pooled over all frames, so a single scene with a different shape is invisible
in it.  Each sequence therefore gets its own single-sequence run (a one-symlink
ADT root), and this compares them.
"""
import json
import os
import sys

SEQ = [("seq131", "fovA"), ("seq132", "fovS_seq132"), ("seq133", "fovS_seq133"),
       ("seq134", "fovS_seq134"), ("seq135", "fovS_seq135"), ("seq136", "fovS_seq136")]
MODELS = ("vggt_1b", "vggt_omega", "dav2_large", "da3_large")


def pen(run, key="AbsRel"):
    b = [c for c in run["bins"] if c.get("n_px_mean")]
    if len(b) < 2 or not b[0].get(key):
        return None
    return b[-1][key] / b[0][key]


def load():
    out = {}
    for short, d in SEQ:
        f = os.path.join(d, "results.json")
        if not os.path.exists(f):
            print(f"  !! {short}: {f} missing")
            continue
        j = json.load(open(f))
        out[short] = {"digest": j["digest"], "n": j["n_frames"],
                      "runs": {(r["model"], r["stream"], r["view"]): r for r in j["runs"]}}
    return out


def main():
    D = load()
    shorts = [s for s, _ in SEQ if s in D]
    print("digests (each sequence is its own split, so these differ by design):")
    for s in shorts:
        print(f"  {s}: {D[s]['digest']}  n_frames={D[s]['n']}")

    for view in ("fisheye", "rect"):
        for stream in ("synthetic", "real"):
            print(f"\n=== pen, {view} / {stream} ===")
            print(f"{'model':12s}" + "".join(f"{s:>9s}" for s in shorts) + f"{'min':>9s}{'max':>9s}{'spread':>8s}")
            for m in MODELS:
                vals = []
                for s in shorts:
                    r = D[s]["runs"].get((m, stream, view))
                    vals.append(pen(r) if r else None)
                good = [v for v in vals if v]
                line = "".join(f"{v:9.3f}" if v else "        —" for v in vals)
                if good:
                    line += f"{min(good):9.3f}{max(good):9.3f}{max(good) - min(good):8.2f}"
                print(f"{m:12s}{line}")

    # does every sequence agree on the sign?  that is the actual question
    print("\n=== sign agreement: how many sequences have pen > 1 (periphery worse) ===")
    print(f"{'model':12s}{'view':9s}{'stream':11s}{'pen>1':>7s}{'of':>4s}   per-sequence pen")
    for m in MODELS:
        for view in ("fisheye", "rect"):
            for stream in ("synthetic", "real"):
                vals = [pen(D[s]["runs"][(m, stream, view)])
                        for s in shorts if (m, stream, view) in D[s]["runs"]]
                vals = [v for v in vals if v]
                if not vals:
                    continue
                n_up = sum(v > 1 for v in vals)
                print(f"{m:12s}{view:9s}{stream:11s}{n_up:7d}{len(vals):4d}   "
                      + " ".join(f"{v:.2f}" for v in vals))

    # the innermost-bin anomaly that #14 raised: is it one scene or all of them?
    print("\n=== dav2 rect innermost vs the rest, per sequence (the seq131 anomaly) ===")
    print(f"{'seq':8s}{'stream':11s}" + "".join(f"{b:>8s}" for b in
          ("0-10", "10-20", "20-30", "30-40", "40-50", "50-55")) + f"{'pen':>8s}")
    for s in shorts:
        for stream in ("real", "synthetic"):
            r = D[s]["runs"].get(("dav2_large", stream, "rect"))
            if not r:
                continue
            v = [c.get("AbsRel") for c in r["bins"]]
            print(f"{s:8s}{stream:11s}" + "".join(f"{x:8.3f}" if x else "       —" for x in v)
                  + f"{pen(r):8.3f}")


if __name__ == "__main__":
    sys.exit(main())
