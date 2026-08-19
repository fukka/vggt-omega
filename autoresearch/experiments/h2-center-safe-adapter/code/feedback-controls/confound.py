"""Is the near-rim 'before' number a property of the rim, or of the frame's
depth histogram interacting with the per-frame global affine?

Within ONE sequence (seq131), split the 28 frames by their share of pixels
within 2 m, and compare near-rim uncorrected AbsRel. If near-heavy frames look
like decoration_seq132 and far-heavy frames look like the clean sequences, the
cross-sequence spread in 'before' is the histogram, not the decor.

Second check: refit the eval affine on near-rim pixels only, and see how much
of the near-rim penalty survives.
"""
import glob, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/fengjiazhang/Desktop/ADT/vggt-omega")
from theta_np import theta_grid, theta_max_native
from finetune.eval.metrics import align_depth

SEQ = "/Users/fengjiazhang/Documents/projectaria_tools_adt_data/Apartment_release_clean_seq131_M1292"
OLD = ("/private/tmp/claude-501/-Users-fengjiazhang-Desktop-ADT-vggt-omega/"
       "c1591315-71f0-4ca2-9183-9f58fff82acc/scratchpad")
SIZE, TB = 504, 8
EDGES = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)
theta = theta_grid(SIZE); tmax = theta_max_native()
cone = theta <= tmax; cos_t = np.cos(theta)
t_edges = np.linspace(0.0, tmax, TB + 1)
t_idx = np.clip(np.digitize(theta, t_edges) - 1, 0, TB - 1)
t_mid = [np.degrees(0.5 * (t_edges[i] + t_edges[i + 1])) for i in range(TB)]
RIM = np.isin(t_idx, [i for i in range(TB) if t_mid[i] >= 38]) & cone

paths = sorted(glob.glob(os.path.join(SEQ, "videos_rgb", "*.jpg")))
dmap = {os.path.basename(q)[:-4]: q for q in glob.glob(os.path.join(SEQ, "depth_npy", "*.npy"))}
frames = [n for n in range(len(paths)) if os.path.basename(paths[n])[:-4] in dmap]


def gt_range(n):
    gz = np.load(dmap[os.path.basename(paths[n])[:-4]]).astype(np.float32)
    if gz.shape != (SIZE, SIZE):                      # nearest, as in run 011
        yi = (np.arange(SIZE) * gz.shape[0] / SIZE).astype(int)
        xi = (np.arange(SIZE) * gz.shape[1] / SIZE).astype(int)
        gz = gz[np.ix_(yi, xi)]
    gz = gz / 1000.0
    return gz / np.clip(cos_t, 1e-6, None), gz


rows = []
for n in frames:
    gr, gz = gt_range(n); d = np.load(f"{OLD}/h2_pred_cache/{n}.npy")
    valid = cone & (gz > 0) & (gr <= 10.0) & (d > 1e-6)
    nearmask = valid & (gr <= 2.0)
    rimnear = valid & RIM & (gr <= 2.0)
    if rimnear.sum() < 500:
        continue
    share = nearmask.sum() / valid.sum()
    out = {}
    for tag, fitmask in (("global", valid), ("near_only", nearmask), ("rim_near_only", rimnear)):
        al = align_depth(d, gr, fitmask, mode="scale_shift")
        out[tag] = float((np.abs(al - gr) / gr)[rimnear].mean())
    rows.append((n, share, out))

rows.sort(key=lambda r: r[1])
half = len(rows) // 2
print(f"{len(rows)} frames of seq131, sorted by share of pixels within 2 m\n")
print(f"{'group':22s} {'<2m share':>10s} {'rim near AbsRel (affine fit on...)':>44s}")
print(f"{'':22s} {'':>10s} {'all valid':>14s} {'near px':>14s} {'rim-near px':>14s}")
for name, grp in (("far-heavy half", rows[:half]), ("near-heavy half", rows[half:])):
    s = np.mean([r[1] for r in grp])
    g = np.mean([r[2]["global"] for r in grp])
    nn = np.mean([r[2]["near_only"] for r in grp])
    rr = np.mean([r[2]["rim_near_only"] for r in grp])
    print(f"{name:22s} {s*100:9.1f}% {g:14.3f} {nn:14.3f} {rr:14.3f}")
print()
allsh = np.array([r[1] for r in rows]); allg = np.array([r[2]["global"] for r in rows])
print("per-frame corr(<2m share, near-rim AbsRel) =", round(float(np.corrcoef(allsh, allg)[0, 1]), 3),
      f"  over {len(rows)} frames")
print("frame-level spread of near-rim AbsRel:", round(float(allg.min()), 3), "..", round(float(allg.max()), 3))
