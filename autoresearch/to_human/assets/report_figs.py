"""Example figures for the report: what each experiment's INPUT and LABEL look like.

Everything is computed on the same seq136 frame the report already uses (found by
matching against the existing input figure), with the frozen DA3-Small, upright.
Outputs are JPEGs in --out, displayed upright (same quarter turn as U.to_model).
"""
import sys, os, math, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPO = "/user/f.zhang2/projects/vggt-omega-organized"
E = f"{REPO}/autoresearch/experiments"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{E}/h1-rim-pose-value/code")
sys.path.append(f"{E}/h5-rim-finetune/code")
sys.path.insert(0, f"{E}/common")
sys.path.insert(0, f"{E}/h14-rect-distill/code")
sys.path.insert(0, f"{E}/h15-lens-holdout/code")
sys.path.insert(0, f"{E}/h9-raycal-tta/code")
import importlib.util as ilu


def _load(name, path):
    sp = ilu.spec_from_file_location(name, path)
    m = ilu.module_from_spec(sp); sp.loader.exec_module(m); return m


_h5 = _load("h5_train", f"{E}/h5-rim-finetune/code/train.py")
Seq, camera_conjugation = _h5.Seq, _h5.camera_conjugation
_ev = _load("h5_eval", f"{E}/h5-rim-finetune/code/eval_lora.py")
THETA_BINS, EDG = _ev.THETA_BINS, _ev.GT_DEPTH_EDGES
import upright as U
import rect_teacher as RT
import lens_family as LF
import anchors as AN
from finetune.eval.metrics import align_depth
from raytun3r.backbones import build_backbone
from raytun3r.matching import build_matcher
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt
def _cmap(n): return _plt.get_cmap(n)

p = argparse.ArgumentParser()
p.add_argument("--seq", default="/user/f.zhang2/Documents/projectaria_tools_adt_data_clean/Apartment_release_clean_seq136_M1292")
p.add_argument("--ref", default="/tmp/fig0_ref.jpeg")
p.add_argument("--out", default="/tmp/report_figs")
p.add_argument("--device", default="cuda")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = a.device
SIZE = 504

# ----------------------------------------------------------------- helpers
def up(x):
    """stored frame -> upright, for display. numpy (H,W) or (H,W,3)."""
    if x.ndim == 3:
        return np.rot90(x, 3, axes=(0, 1))
    return np.rot90(x, 3)

def save_rgb(arr_u8, name, q=88):
    Image.fromarray(np.ascontiguousarray(arr_u8)).save(f"{a.out}/{name}.jpg", quality=q)
    print("  wrote", name)

def img_to_u8(t):  # (3,H,W) float 0..1 stored frame -> upright u8
    return up((t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

def cmap_u8(vals, mask, cmap, vmin, vmax):
    v = np.clip((vals - vmin) / (vmax - vmin), 0, 1)
    rgb = (_cmap(cmap)(v)[..., :3] * 255).astype(np.uint8)
    rgb[~mask] = 0
    return up(rgb)

# stored (x,y) -> upright (x,y) under np.rot90(k=3): checked, not assumed.
_t = np.zeros((7, 5)); _t[2, 4] = 1
_u = np.rot90(_t, 3); _iy, _ix = np.argwhere(_u == 1)[0]
assert (_ix, _iy) == (7 - 1 - 2, 4), (_ix, _iy)
def up_xy(x, y, H=SIZE):
    return H - 1 - y, x

# ----------------------------------------------------------------- data
s = Seq(a.seq, SIZE, 60)
cam = s.src.camera
theta = cam.incidence_grid(SIZE, SIZE)
cos_t = torch.cos(theta)
cone = (theta <= cam.theta_max).numpy()
theta_deg = np.rad2deg(theta.numpy())

# find the frame the report already shows
ref = np.asarray(Image.open(a.ref).convert("RGB").resize((63, 63), Image.BILINEAR)).astype(np.float32) / 255
best = None
for n in s.frames:
    im = img_to_u8(s.src.image(n))
    im = np.asarray(Image.fromarray(im).resize((63, 63), Image.BILINEAR)).astype(np.float32) / 255
    d = float(((im - ref) ** 2).mean())
    if best is None or d < best[0]:
        best = (d, n)
n0 = best[1]
print(f"[frame] matched report frame: index {n0} stem {s.stem(n0)} mse {best[0]:.5f}")
img0 = s.src.image(n0)
gt0 = s.gt_range(n0, cos_t).numpy()
valid_gt = cone & (gt0 > 0) & (gt0 <= 10.0)

bb = build_backbone("da3", weights="pretrained", device=dev, variant="small")
def install(camera, hw):
    bb.install(None, camera, hw, patch_undistort=False, border_token=False,
               dpt_grid=False, depth_convention="z")
install(cam, (SIZE, SIZE))

def zones(absrel, v):
    out = {}
    for nm, keep in (("near_rim", (theta_deg >= 38) & (gt0 <= 2.0)),
                     ("near_center", (theta_deg <= 11) & (gt0 <= 2.0)),
                     ("center", theta_deg <= 11), ("all", np.ones_like(v))):
        m = v & keep
        out[nm] = float(absrel[m].mean()) if m.sum() else float("nan")
    return out

def err_map(rng):
    v = valid_gt & (rng > 1e-6)
    al = align_depth(rng, gt0, v, mode="scale_shift")
    ar = np.abs(al - gt0) / np.clip(gt0, 1e-6, None)
    return ar, v

ERR_MAX = 0.6
stats = {"frame": s.stem(n0), "index": int(n0)}

# ================================================================== 1. roll
print("[roll]")
def roll_grid(size, cx, cy, deg):
    an = math.radians(deg); ca, sa = math.cos(an), math.sin(an)
    ys, xs = torch.meshgrid(torch.arange(size, dtype=torch.float32),
                            torch.arange(size, dtype=torch.float32), indexing="ij")
    x, y = xs - cx, ys - cy
    sx, sy = ca * x + sa * y + cx, -sa * x + ca * y + cy
    return torch.stack((2 * (sx + 0.5) / size - 1, 2 * (sy + 0.5) / size - 1), dim=-1)

def warp(t, g, mode="bilinear"):
    x = t[None] if t.dim() == 3 else t[None, None]
    o = F.grid_sample(x, g[None].to(x.device, x.dtype), mode=mode,
                      padding_mode="zeros", align_corners=False)
    return o[0] if t.dim() == 3 else o[0, 0]

stats["roll"] = {}
for deg in (0.0, 10.0, 20.0, 30.0, 40.0):
    g_in = roll_grid(SIZE, float(cam.cx), float(cam.cy), deg).to(dev)
    g_out = roll_grid(SIZE, float(cam.cx), float(cam.cy), -deg).to(dev)
    img = U.to_model(img0.to(dev))
    if deg:
        img = warp(img, g_in)
    # the model's actual input, already upright -> no display turn
    inp = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    save_rgb(inp, f"roll_{int(deg)}_in")
    with torch.no_grad():
        z = bb.forward(img[None, None]).depth[0]
    if deg:
        z = warp(z, g_out)
    z = U.from_model(z).cpu()
    rng = (z / cos_t.clamp_min(1e-6)).numpy()
    ar, v = err_map(rng)
    save_rgb(cmap_u8(ar, v, "magma", 0.0, ERR_MAX), f"roll_{int(deg)}_err")
    stats["roll"][str(int(deg))] = zones(ar, v)
    print(f"  {deg:>4.0f}°  {stats['roll'][str(int(deg))]}")

# ================================================================== 2. H14
print("[h14]")
with torch.no_grad():
    raw_rng = U.forward_range(bb, img0.to(dev), cos_t.to(dev)).float().cpu()
rig = RT.Rig.single(cam, 95.0, 630)
installed = {"hw": None}
def forward_z(warped, view):
    hw = (view.spec.height, view.spec.width)
    if installed["hw"] != hw:
        install(view.pin, hw); installed["hw"] = hw
    with torch.no_grad():
        return U.forward_z(bb, warped)
tea, info = rig.teach(forward_z, img0.to(dev))
tea = tea.float().cpu().numpy()
install(cam, (SIZE, SIZE)); installed["hw"] = (SIZE, SIZE)
rt, _ = rig.roundtrip(raw_rng.to(dev))
rt = rt.float().cpu().numpy()
cov = rig.covered.numpy()
# teacher view image (what the frozen model sees on the rect path)
view_img = RT.warp(img0, rig.views[0].grid_in)
save_rgb(img_to_u8(view_img), "h14_view95")
save_rgb(img_to_u8(img0), "h14_in")
# labels are ranges; align each to GT scale for display only (labels are scale-free)
def disp_depth(rng, mask, name):
    m = mask & valid_gt & (rng > 1e-6)
    al = align_depth(rng, gt0, m, mode="scale_shift")
    save_rgb(cmap_u8(al, mask & (rng > 1e-6), "jet", 0.3, 6.0), name)
disp_depth(tea, cov, "h14_label_rect")
disp_depth(rt, cov, "h14_label_roundtrip")
disp_depth(raw_rng.numpy(), cone, "h14_raw_pred")
save_rgb(cmap_u8(gt0, valid_gt, "jet", 0.3, 6.0), "h14_gt")
ar_t, v_t = err_map(np.where(cov, tea, 0.0))
ar_r, v_r = err_map(raw_rng.numpy())
stats["h14"] = {"teacher_on_covered": zones(ar_t, v_t),
                "raw_on_covered": zones(ar_r, v_r & cov),
                "raw_all": zones(ar_r, v_r),
                "coverage_cone": float(cov[cone].mean()),
                "coverage_near_rim": float(cov[cone & (theta_deg >= 38) & (gt0 <= 2) & (gt0 > 0)].mean())}
print("  ", json.dumps(stats["h14"]))

# ================================================================== 3. H15
print("[h15]")
names = ["aria_kb4", "equidistant", "rectilinear", "stereographic", "equisolid"]
fam = LF.lens_family(names, SIZE, float(cam.theta_max))
fields = {n: LF.token_field(fam[n], SIZE) for n in names}
la = np.stack([fields[n][:, 0].numpy() for n in names])
vmin, vmax = float(la.min()), float(la.max())
gh = SIZE // 14
def field_img(f, name):
    m = f[:, 0].numpy().reshape(gh, gh)
    m = np.kron(m, np.ones((14, 14)))
    save_rgb(cmap_u8(m, cone, "viridis", vmin, vmax), name)
for n in names:
    g, valid = LF.grid_between(cam, fam[n])
    save_rgb(img_to_u8(LF.warp(img0, g)), f"lens_{n}")
    field_img(fields[n], f"field_{n}")
perm = torch.randperm(fields["aria_kb4"].shape[0], generator=torch.Generator().manual_seed(0))
field_img(fields["aria_kb4"][perm], "field_aria_kb4_shuffled")
stats["h15"] = {"log_area_range": [vmin, vmax]}

# ================================================================== 4. H9
print("[h9]")
matcher = build_matcher("auto", device=dev)
C = camera_conjugation()
packed = []
for off in (-10, +10):
    m = n0 + off
    assert 0 <= m < len(s.src.paths) and s.src.pose(n0) is not None and s.src.pose(m) is not None
    with torch.no_grad():
        mm = matcher(img0.to(dev), s.src.image(m).to(dev), valid=torch.from_numpy(cone).to(dev))
    packed.append((mm.target.detach().cpu().double().reshape(-1, 2),
                   mm.weight.detach().cpu().reshape(-1), s.rel_pose(n0, m, C)))
    save_rgb(img_to_u8(s.src.image(m)), f"h9_partner_{'m' if off < 0 else 'p'}10")
good = (packed[0][1] > 0) & (packed[1][1] > 0) & torch.from_numpy(cone.reshape(-1))
idx = torch.nonzero(good, as_tuple=False).squeeze(-1)
gen = torch.Generator().manual_seed(0)
idx = idx[torch.randperm(idx.numel(), generator=gen)[:3000]]
uv = torch.stack([(idx % SIZE).double(), (idx // SIZE).double()], dim=-1)
plist = [(t[idx], R.double(), tt.double()) for t, _, (R, tt) in packed]
kept = AN.anchors_from_pairs(cam, uv, plist, min_parallax_deg=1.0, agree_tol=0.10, max_range_m=12.0)
loose = AN.anchors_from_pairs(cam, uv, plist, min_parallax_deg=1.0, agree_tol=10.0, max_range_m=12.0)
k_set = {(int(x), int(y)) for x, y in kept.uv.round().tolist()}
l_set = {(int(x), int(y)) for x, y in loose.uv.round().tolist()}
gate_rej = l_set - k_set
base = Image.fromarray(img_to_u8(img0)).convert("RGB")
d = ImageDraw.Draw(base)
jet = _cmap("jet")
for (x, y), r in zip(kept.uv.tolist(), kept.rng.tolist()):
    ux, uy = up_xy(x, y)
    c = tuple(int(255 * q) for q in jet(np.clip((r - 0.3) / 5.7, 0, 1))[:3])
    d.ellipse([ux - 2.5, uy - 2.5, ux + 2.5, uy + 2.5], fill=c)
save_rgb(np.asarray(base), "h9_anchors_kept")
base2 = Image.fromarray(img_to_u8(img0)).convert("RGB")
d2 = ImageDraw.Draw(base2)
for (x, y) in gate_rej:
    ux, uy = up_xy(x, y)
    d2.ellipse([ux - 2.5, uy - 2.5, ux + 2.5, uy + 2.5], fill=(230, 40, 40))
save_rgb(np.asarray(base2), "h9_anchors_rejected")
# how good are the anchors? compare kept ranges to GT at those pixels
ki = kept.uv.round().long()
gk = gt0[ki[:, 1].clamp(0, SIZE - 1), ki[:, 0].clamp(0, SIZE - 1)]
okg = gk > 0
rel = np.abs(kept.rng.numpy()[okg] - gk[okg]) / gk[okg]
stats["h9"] = {"matched_both": int(idx.numel()), "kept": len(kept), "loose": len(loose),
               "gate_rejected": len(gate_rej), "rim_share": float((kept.theta.numpy() >= math.radians(38)).mean()),
               "median_parallax_deg": float(np.degrees(np.median(kept.parallax.numpy()))),
               "anchor_absrel_vs_gt_median": float(np.median(rel)),
               "anchor_absrel_vs_gt_mean": float(rel.mean())}
print("  ", json.dumps(stats["h9"]))

json.dump(stats, open(f"{a.out}/stats.json", "w"), indent=1)
print("DONE")
