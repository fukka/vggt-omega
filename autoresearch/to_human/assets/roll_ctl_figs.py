"""Figures for the roll controls, same seq136 frame as report_figs.py.

pinhole_{deg}_in / _err : 89 deg co-axial view rolled about its axis (no black
anywhere), error mapped back to the fisheye grid on theta<=44 deg.
disc_0_in / disc_0_err  : the hard-disc-masked fisheye at 0 deg, for the
"1.7% of pixels cost +38% at the rim" panel.
"""
import sys, os, math, json
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "/tmp")
REPO = "/user/f.zhang2/projects/vggt-omega-organized"
E = f"{REPO}/autoresearch/experiments"
for p_ in (REPO, f"{E}/h1-rim-pose-value/code", f"{E}/common", f"{E}/h14-rect-distill/code", f"{E}/h16-orientation/code"):
    sys.path.insert(0, p_)
sys.path.append(f"{E}/h5-rim-finetune/code")
import importlib.util as ilu
def _load(name, path):
    sp = ilu.spec_from_file_location(name, path); m = ilu.module_from_spec(sp); sp.loader.exec_module(m); return m
Seq = _load("h5_train", f"{E}/h5-rim-finetune/code/train.py").Seq
import upright as U, rect_teacher as RT
import roll_controls as RC
from finetune.eval.metrics import align_depth
from raytun3r.backbones import build_backbone
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "/tmp/report_figs"; os.makedirs(OUT, exist_ok=True)
dev = "cuda"; SIZE = 504; N0 = 1410
s = Seq("/user/f.zhang2/Documents/projectaria_tools_adt_data_clean/Apartment_release_clean_seq136_M1292", SIZE, 60)
assert s.stem(N0) == "frame_001918_15100343821150", s.stem(N0)
cam = s.src.camera
theta = cam.incidence_grid(SIZE, SIZE); cos_t = torch.cos(theta)
cone = (theta <= cam.theta_max).numpy(); tdeg = np.rad2deg(theta.numpy())
common = cone & (tdeg <= 44.0)
img0 = s.src.image(N0); gt0 = s.gt_range(N0, cos_t).numpy()
valid = cone & (gt0 > 0) & (gt0 <= 10.0)

def up(x): return np.rot90(x, 3, axes=(0, 1)) if x.ndim == 3 else np.rot90(x, 3)
def save(arr, name): Image.fromarray(np.ascontiguousarray(arr)).save(f"{OUT}/{name}.jpg", quality=88); print("  wrote", name)
def img_u8(t): return up((t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
def cm_u8(v, mask, cmap, lo, hi):
    x = np.clip((v - lo) / (hi - lo), 0, 1); rgb = (plt.get_cmap(cmap)(x)[..., :3] * 255).astype(np.uint8); rgb[~mask] = 0; return up(rgb)
def zones(ar, v):
    o = {}
    for nm, k in (("near_rim", (tdeg >= 38) & (gt0 <= 2)), ("center", tdeg <= 11), ("all", np.ones_like(v))):
        m = v & k; o[nm] = float(ar[m].mean()) if m.sum() else float("nan")
    return o
def errmap(rng, mask):
    v = mask & valid & (rng > 1e-6); al = align_depth(rng, gt0, v, mode="scale_shift")
    return np.abs(al - gt0) / np.clip(gt0, 1e-6, None), v

bb = build_backbone("da3", weights="pretrained", device=dev, variant="small")
def install(camera, hw): bb.install(None, camera, hw, patch_undistort=False, border_token=False, dpt_grid=False, depth_convention="z")
stats = {}
for deg in (0, 20, 30, 40):
    rig = RT.Rig(cam, [RC.RolledView(fov_x_deg=89.0, width=630, height=630, roll_deg=float(deg))])
    v = rig.views[0]; install(v.pin, (630, 630))
    save(img_u8(RT.warp(img0, v.grid_in)), f"pinhole_{deg}_in")
    with torch.no_grad():
        d, _ = rig.teach(lambda w, view: U.forward_z(bb, w), img0.to(dev), align=False)
    rng = np.where(rig.covered.numpy(), d.float().cpu().numpy(), 0.0)
    ar, vm = errmap(rng, common)
    save(cm_u8(ar, vm, "magma", 0, 0.6), f"pinhole_{deg}_err")
    stats[f"pinhole_{deg}"] = zones(ar, vm); print(deg, stats[f"pinhole_{deg}"])

# hard disc at 0 deg
install(cam, (SIZE, SIZE))
ucx, ucy = RC.upright_principal_point(cam, SIZE)
r = min(ucx, ucy, SIZE - 1 - ucx, SIZE - 1 - ucy) - 1.0
ys, xs = torch.meshgrid(torch.arange(SIZE, dtype=torch.float32), torch.arange(SIZE, dtype=torch.float32), indexing="ij")
disc_up = ((xs - ucx) ** 2 + (ys - ucy) ** 2 <= r ** 2)
disc_st = U.from_model(disc_up.float()).numpy() > 0.5
img = U.to_model(img0.to(dev)) * disc_up.to(dev)
save((img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), "disc_0_in")
with torch.no_grad():
    z = bb.forward(img[None, None]).depth[0]
rng = (U.from_model(z).cpu() / cos_t.clamp_min(1e-6)).numpy()
ar, vm = errmap(rng, cone & disc_st)
save(cm_u8(ar, vm, "magma", 0, 0.6), "disc_0_err")
stats["disc_0"] = zones(ar, vm); print("disc0", stats["disc_0"])
with torch.no_grad():
    rng2 = U.forward_range(bb, img0.to(dev), cos_t.to(dev)).float().cpu().numpy()
ar2, vm2 = errmap(rng2, cone & disc_st)
stats["frame_0_on_disc"] = zones(ar2, vm2); print("frame0 on disc mask", stats["frame_0_on_disc"])
json.dump(stats, open(f"{OUT}/roll_ctl_stats.json", "w"), indent=1); print("DONE")
