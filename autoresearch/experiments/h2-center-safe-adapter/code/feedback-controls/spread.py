"""How reliable is a per-sequence near-rim 'before'/'after' number at all?
Bootstrap the 14-frame test-split mean over seq131's frames."""
import glob, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/fengjiazhang/Desktop/ADT/vggt-omega")
from theta_np import theta_grid, theta_max_native
from finetune.eval.metrics import align_depth

SEQ = "/Users/fengjiazhang/Documents/projectaria_tools_adt_data/Apartment_release_clean_seq131_M1292"
OLD = ("/private/tmp/claude-501/-Users-fengjiazhang-Desktop-ADT-vggt-omega/"
       "c1591315-71f0-4ca2-9183-9f58fff82acc/scratchpad")
SIZE, TB = 504, 8
theta = theta_grid(SIZE); tmax = theta_max_native()
cone = theta <= tmax; cos_t = np.cos(theta)
t_edges = np.linspace(0.0, tmax, TB + 1)
t_idx = np.clip(np.digitize(theta, t_edges) - 1, 0, TB - 1)
t_mid = [np.degrees(0.5 * (t_edges[i] + t_edges[i + 1])) for i in range(TB)]
RIM = np.isin(t_idx, [i for i in range(TB) if t_mid[i] >= 38]) & cone

paths = sorted(glob.glob(os.path.join(SEQ, "videos_rgb", "*.jpg")))
dmap = {os.path.basename(q)[:-4]: q for q in glob.glob(os.path.join(SEQ, "depth_npy", "*.npy"))}
frames = [n for n in range(len(paths)) if os.path.basename(paths[n])[:-4] in dmap]

S_, N_ = [], []
for n in frames:
    gz = np.load(dmap[os.path.basename(paths[n])[:-4]]).astype(np.float32)
    yi = (np.arange(SIZE) * gz.shape[0] / SIZE).astype(int)
    gz = gz[np.ix_(yi, yi)] / 1000.0
    gr = gz / np.clip(cos_t, 1e-6, None)
    d = np.load(f"{OLD}/h2_pred_cache/{n}.npy")
    valid = cone & (gz > 0) & (gr <= 10.0) & (d > 1e-6)
    rn = valid & RIM & (gr <= 2.0)
    al = align_depth(d, gr, valid, mode="scale_shift")
    e = (np.abs(al - gr) / np.clip(gr, 1e-6, None))[rn]
    S_.append(float(e.sum())); N_.append(int(rn.sum()))
S_, N_ = np.array(S_), np.array(N_)
print("per-frame near-rim AbsRel, 28 frames:")
print("  ", np.round(np.sort(S_ / np.maximum(N_, 1)), 2))
rng = np.random.default_rng(0)
bs = []
for _ in range(4000):
    k = rng.choice(len(frames), 14, replace=True)     # a 14-frame split
    bs.append(S_[k].sum() / N_[k].sum())
bs = np.array(bs)
print(f"\npixel-pooled near-rim AbsRel over a random 14-frame split:")
print(f"  median {np.median(bs):.3f}   90% range {np.percentile(bs,5):.3f} .. {np.percentile(bs,95):.3f}")
print(f"  published even/odd 'before' = 1.023,  halves 'before' = 0.639")
