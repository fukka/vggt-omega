"""Control for run 011: is the gain coming from the IMAGE FEATURES, or just
from a smooth function of (theta, predicted depth)?

Pure numpy (no usable torch on this Mac). Reuses the frozen DA3-Small caches
written by the original run 011 session, and a numpy KB4 grid validated to 7
decimals against the published theta bin edges.

Arms, identical training in every other respect:
  full      : LN(feat 384) + [sin t, cos t, log d]   <- the published head
  aux_only  :               [sin t, cos t, log d]    <- smooth version of the H2.1 table
  theta_only:               [sin t, cos t]           <- pure radial function
"""
import glob, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/fengjiazhang/Desktop/ADT/vggt-omega")
from theta_np import theta_grid, theta_max_native
from finetune.eval.metrics import align_depth

SEQ = "/Users/fengjiazhang/Documents/projectaria_tools_adt_data/Apartment_release_clean_seq131_M1292"
OLD = ("/private/tmp/claude-501/-Users-fengjiazhang-Desktop-ADT-vggt-omega/"
       "c1591315-71f0-4ca2-9183-9f58fff82acc/scratchpad")
SIZE, PATCH, TB = 504, 14, 8
EDGES = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)
GH = GW = SIZE // PATCH

theta = theta_grid(SIZE); tmax = theta_max_native()
cone = theta <= tmax
cos_t = np.cos(theta)
t_edges = np.linspace(0.0, tmax, TB + 1)
t_idx = np.clip(np.digitize(theta, t_edges) - 1, 0, TB - 1)
theta_p = theta.reshape(GH, PATCH, GW, PATCH).mean((1, 3)).ravel()
t_mid = [np.degrees(0.5 * (t_edges[i] + t_edges[i + 1])) for i in range(TB)]

paths = sorted(glob.glob(os.path.join(SEQ, "videos_rgb", "*.jpg")))
dmap = {os.path.basename(q)[:-4]: q for q in glob.glob(os.path.join(SEQ, "depth_npy", "*.npy"))}
frames = [n for n in range(len(paths)) if os.path.basename(paths[n])[:-4] in dmap]
pred = {n: np.load(f"{OLD}/h2_pred_cache/{n}.npy") for n in frames}
feat = {n: np.load(f"{OLD}/h2_feat_cache/{n}.npy") for n in frames}


def gt_range(n):
    gz = np.load(dmap[os.path.basename(paths[n])[:-4]]).astype(np.float32)
    if gz.shape != (SIZE, SIZE):                       # nearest, as in run 011
        yi = (np.arange(SIZE) * gz.shape[0] / SIZE).astype(int)
        xi = (np.arange(SIZE) * gz.shape[1] / SIZE).astype(int)
        gz = gz[np.ix_(yi, xi)]
    gz = gz / 1000.0
    return gz / np.clip(cos_t, 1e-6, None), gz


def patch_pool(m, vmask=None):
    r = m.reshape(GH, PATCH, GW, PATCH).transpose(0, 2, 1, 3).reshape(GH * GW, -1)
    if vmask is None:
        return np.median(r, axis=1)
    v = vmask.reshape(GH, PATCH, GW, PATCH).transpose(0, 2, 1, 3).reshape(GH * GW, -1)
    out = np.full(GH * GW, np.nan)
    for k in range(GH * GW):
        if v[k].sum() >= 30:
            out[k] = np.median(r[k][v[k]])
    return out


def upsample(c):                                        # bilinear, align_corners=False
    g = c.reshape(GH, GW)
    y = (np.arange(SIZE) + 0.5) * GH / SIZE - 0.5
    x = (np.arange(SIZE) + 0.5) * GW / SIZE - 0.5
    y0 = np.clip(np.floor(y).astype(int), 0, GH - 1); y1 = np.clip(y0 + 1, 0, GH - 1)
    x0 = np.clip(np.floor(x).astype(int), 0, GW - 1); x1 = np.clip(x0 + 1, 0, GW - 1)
    wy = np.clip(y - y0, 0, 1)[:, None]; wx = np.clip(x - x0, 0, 1)[None, :]
    return ((g[np.ix_(y0, x0)] * (1 - wy) * (1 - wx) + g[np.ix_(y1, x0)] * wy * (1 - wx)
             + g[np.ix_(y0, x1)] * (1 - wy) * wx + g[np.ix_(y1, x1)] * wy * wx))


def gelu(x):
    c = np.sqrt(2 / np.pi); u = c * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1 + np.tanh(u))


def dgelu(x):
    c = np.sqrt(2 / np.pi); u = c * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * 0.044715 * x ** 2)


class Head:
    """LayerNorm(features) + 2-layer MLP, zero-init last layer -- run 011's Head."""

    def __init__(self, c_feat, n_aux, rng):
        self.c = c_feat
        d = c_feat + n_aux
        b = 1 / np.sqrt(d)
        self.g = np.ones(c_feat); self.b = np.zeros(c_feat)
        self.W1 = rng.uniform(-b, b, (d, 64)); self.b1 = rng.uniform(-b, b, 64)
        self.W2 = np.zeros((64, 1)); self.b2 = np.zeros(1)
        self.p = ["g", "b", "W1", "b1", "W2", "b2"]
        self.m = {k: np.zeros_like(getattr(self, k)) for k in self.p}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in self.p}
        self.t = 0

    def fwd(self, F, A):
        if self.c:
            mu = F.mean(-1, keepdims=True); var = F.var(-1, keepdims=True)
            self.xh = (F - mu) / np.sqrt(var + 1e-5)
            self.sd = np.sqrt(var + 1e-5)
            z = self.xh * self.g + self.b
            self.hin = np.concatenate([z, A], -1)
        else:
            self.hin = A
        self.h = self.hin @ self.W1 + self.b1
        self.a = gelu(self.h)
        return (self.a @ self.W2 + self.b2)[:, 0]

    def step(self, out, y, lr=1e-3):
        N = len(y)
        do = (np.sign(out - y) / N)[:, None]
        gW2 = self.a.T @ do; gb2 = do.sum(0)
        dh = (do @ self.W2.T) * dgelu(self.h)
        gW1 = self.hin.T @ dh; gb1 = dh.sum(0)
        if self.c:
            dz = (dh @ self.W1.T)[:, :self.c]
            gg = (dz * self.xh).sum(0); gb = dz.sum(0)
        else:
            gg = np.zeros_like(self.g); gb = np.zeros_like(self.b)
        self.t += 1
        for k, gr in zip(self.p, [gg, gb, gW1, gb1, gW2, gb2]):
            self.m[k] = 0.9 * self.m[k] + 0.1 * gr
            self.v[k] = 0.999 * self.v[k] + 0.001 * gr * gr
            mh = self.m[k] / (1 - 0.9 ** self.t); vh = self.v[k] / (1 - 0.999 ** self.t)
            setattr(self, k, getattr(self, k) - lr * mh / (np.sqrt(vh) + 1e-8))


def build(idx, arm):
    X, A, Y = [], [], []
    for n in idx:
        gr, gz = gt_range(n); d = pred[n]
        valid = cone & (gz > 0) & (gr <= 10.0) & (d > 1e-6)
        r = np.log(gr) - np.log(np.clip(d, 1e-6, None))
        tgt = patch_pool(r - np.median(r[valid]), vmask=valid)
        dp = patch_pool(np.log(np.clip(d, 1e-6, None)))
        ok = np.isfinite(tgt)
        X.append(feat[n][ok]); Y.append(tgt[ok])
        A.append(aux(theta_p[ok], dp[ok], arm))
    return np.concatenate(X), np.concatenate(A), np.concatenate(Y)


def aux(tp, dp, arm):
    if arm == "theta_only":
        return np.stack([np.sin(tp), np.cos(tp)], -1)
    return np.stack([np.sin(tp), np.cos(tp), dp], -1)


def zones(test, head, arm, correct):
    nb = len(EDGES) - 1
    s = np.zeros((TB, nb)); c = np.zeros((TB, nb))
    for n in test:
        gr, gz = gt_range(n); d = pred[n].copy()
        if correct:
            dp = patch_pool(np.log(np.clip(d, 1e-6, None)))
            corr = head.fwd(feat[n] if head.c else np.zeros((GH * GW, 0)),
                            aux(theta_p, dp, arm))
            d = d * np.exp(upsample(corr))
        valid = cone & (gz > 0) & (gr <= 10.0) & (d > 1e-6)
        al = align_depth(d, gr, valid, mode="scale_shift")
        ar = (np.abs(al - gr) / gr)[valid]
        flat = t_idx[valid] * nb + np.clip(np.digitize(gr[valid], EDGES) - 1, 0, nb - 1)
        s += np.bincount(flat, weights=ar, minlength=TB * nb).reshape(TB, nb)
        c += np.bincount(flat, minlength=TB * nb).reshape(TB, nb)
    e = s / np.maximum(c, 1)
    Z = {"near_rim": [(i, j) for i in range(TB) for j in range(nb) if t_mid[i] >= 38 and EDGES[j + 1] <= 2.0],
         "near_center": [(i, j) for i in range(TB) for j in range(nb) if t_mid[i] <= 11 and EDGES[j + 1] <= 2.0],
         "center": [(i, j) for i in range(TB) for j in range(nb) if t_mid[i] <= 11],
         "far": [(i, j) for i in range(TB) for j in range(nb) if EDGES[j] >= 3.0]}
    out = {}
    for k, cells in Z.items():
        w = np.array([c[i, j] for i, j in cells], float)
        out[k] = float((np.array([e[i, j] for i, j in cells]) * w).sum() / w.sum())
    return out


for split in ("even_odd", "halves"):
    if split == "even_odd":
        train = [n for k, n in enumerate(frames) if k % 2 == 0]
        test = [n for k, n in enumerate(frames) if k % 2 == 1]
    else:
        h = len(frames) // 2; train, test = frames[:h], frames[h:]
    base = zones(test, None, None, False)
    print(f"\n=== split {split} ({len(train)} train / {len(test)} test frames) ===")
    print(f"{'arm':12s} " + " ".join(f"{k:>12s}" for k in base))
    print(f"{'uncorrected':12s} " + " ".join(f"{base[k]:12.3f}" for k in base))
    for arm in ("full", "aux_only", "theta_only"):
        rng = np.random.default_rng(0)
        X, A, Y = build(train, arm)
        head = Head(X.shape[1] if arm == "full" else 0, A.shape[1], rng)
        Xin = X if arm == "full" else np.zeros((len(Y), 0))
        prev = np.inf
        for ep in range(300):
            out = head.fwd(Xin, A)
            loss = np.abs(out - Y).mean()
            head.step(out, Y)
            if abs(prev - loss) < 1e-6:
                break
            prev = loss
        z = zones(test, head, arm, True)
        print(f"{arm:12s} " + " ".join(f"{z[k]:12.3f}" for k in base)
              + f"   (L1 {loss:.4f}, {ep+1} ep)")
