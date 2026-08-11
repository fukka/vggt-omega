"""Render the two tables that are in results.json but not in report.txt:
the radius-binned axis, and src_px_per_out_px per window cell."""
import json, sys

d = json.load(open(sys.argv[1]))
runs = d["runs"]
redges = d["config"]["radius_edges"]
tedges = d["config"]["theta_edges"]


def populated(cells):
    """Bins with pixels in them. An empty bin is geometry, not a measurement:
    the fisheye image circle has nothing past radius 1.0, as the rectified
    pinhole has almost nothing past 42.3 deg."""
    return [c for c in cells if c.get("n_px_mean")]


def pen(cells, key="AbsRel"):
    v = [c[key] for c in populated(cells) if c.get(key) is not None]
    if len(v) < 2 or not v[0]:
        return None
    return v[-1] / v[0]


def drift(cells):
    v = [c.get("anchored_ratio") for c in populated(cells)]
    v = [x for x in v if x is not None and x == x]
    if len(v) < 2 or not v[-1]:
        return None
    return v[0] / v[-1]


def fmt(v, w=7, p=3):
    if v is None or v != v:
        return "—".rjust(w)
    return f"{v:{w}.{p}f}"


def cell(c, key, w=7, p=3):
    return "—".rjust(w) if not c.get("n_px_mean") else fmt(c.get(key), w, p)


for view in ("rect", "fisheye"):
    for metric in ("AbsRel", "delta1"):
        hdr = "  ".join(f"{lo:g}-{hi:g}".rjust(7)
                        for lo, hi in zip(redges[:-1], redges[1:]))
        print(f"\n  RADIUS · {view} · {metric}   (r / half-width; 1.0 = edge midpoint, 1.45 ~ corner)")
        print("  " + "-" * 96)
        print(f"  {'model':<13} {'stream':<11} " + hdr + "     pen  drift*")
        print("  " + "-" * 96)
        for r in sorted([x for x in runs if x["protocol"] == "radial" and x["view"] == view],
                        key=lambda x: (x["model"], x["stream"])):
            cells = r["radius_bins"]
            row = "  ".join(cell(c, metric) for c in cells)
            print(f"  {r['model']:<13} {r['stream']:<11} {row} "
                  f"{fmt(pen(cells), 7, 2)} {fmt(drift(cells), 7, 3)}")

print("\n\n  WINDOW · src_px_per_out_px   (source pixels per output pixel, per aim)")
print("  " + "-" * 78)
tilts = d["config"]["tilts"]
print(f"  {'view':<9} {'model':<13} {'stream':<11} " +
      "  ".join(f"t{t:g}".rjust(7) for t in tilts))
print("  " + "-" * 78)
for view in ("rect", "fisheye"):
    for r in sorted([x for x in runs if x["protocol"] == "window" and x["view"] == view],
                    key=lambda x: (x["model"], x["stream"])):
        row = "  ".join(fmt(c.get("src_px_per_out_px"), 7, 2) for c in r["cells"])
        print(f"  {view:<9} {r['model']:<13} {r['stream']:<11} {row}")

print("\n\n  RADIUS vs THETA — bin edges are not the same directions")
print("  " + "-" * 60)
print(f"  theta_edges  : {tedges}")
print(f"  radius_edges : {redges}")
