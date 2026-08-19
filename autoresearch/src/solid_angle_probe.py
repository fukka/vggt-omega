import sys, math, torch
sys.path.insert(0, "/Users/fengjiazhang/Desktop/ADT/vggt-omega")
from raytun3r.cameras import from_aria
cam = from_aria(504, 504)
H = W = 504; P = 14
theta = cam.incidence_grid(H, W)          # (H,W) radians
cone = theta <= cam.theta_max
# per-pixel solid angle: dOmega = sin(theta) dtheta dphi = |J| per pixel.
# numerically: unproject pixel corners -> spherical excess approx via cross products
ys, xs = torch.meshgrid(torch.arange(H+1, dtype=torch.float64)-0.5,
                        torch.arange(W+1, dtype=torch.float64)-0.5, indexing="ij")
d = cam.unproject(torch.stack([xs, ys], -1).reshape(-1,2).float()).reshape(H+1, W+1, 3).double()
d = d / d.norm(dim=-1, keepdim=True)
a = d[:-1,:-1]; b = d[:-1,1:]; c = d[1:,1:]; e = d[1:,:-1]
def tri(u,v,w):  # solid angle of spherical triangle (Van Oosterom-Strackee)
    num = (u * torch.cross(v, w, dim=-1)).sum(-1)
    den = 1 + (u*v).sum(-1) + (v*w).sum(-1) + (u*w).sum(-1)
    return 2*torch.atan2(num.abs(), den)
omega_px = tri(a,b,c) + tri(a,c,e)        # (H,W)
omega_patch = omega_px.reshape(H//P, P, W//P, P).sum((1,3))   # (36,36)
theta_patch = theta.reshape(H//P, P, W//P, P).mean((1,3))
in_cone = theta_patch <= cam.theta_max
center = omega_patch[17:19,17:19].mean()
rim = omega_patch[in_cone & (theta_patch > math.radians(45))].mean()
print(f"theta_max = {math.degrees(cam.theta_max):.2f} deg")
print(f"patch solid angle: center {center:.6f} sr, rim(>45deg) {rim:.6f} sr, ratio center/rim = {center/rim:.2f}")
n_tok = int(in_cone.sum())
omega_cone = omega_patch[in_cone].sum()
# equal-area tokenization: same max angular support as the CURRENT center patch
n_equal = omega_cone / center
print(f"tokens in cone now: {n_tok}; equal-area tokens at center-patch density: {n_equal:.0f} ({(1-n_equal/n_tok)*100:.1f}% fewer)")
# and at the mean density (area-neutral):
print(f"cone solid angle {omega_cone:.4f} sr; mean patch {omega_cone/n_tok:.6f} sr")
