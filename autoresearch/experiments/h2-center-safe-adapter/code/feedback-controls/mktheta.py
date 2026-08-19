import sys, os, numpy as np, torch
sys.path.insert(0,'.'); sys.path.insert(0,'autoresearch/experiments/h1-rim-pose-value/code')
if not hasattr(torch, "searchsorted"):
    def _ss(seq, vals, **kw):
        return torch.from_numpy(np.searchsorted(seq.detach().numpy(), vals.detach().numpy())).long()
    torch.searchsorted = _ss

_mg = torch.meshgrid
def _meshgrid(*t, indexing="ij", **kw):
    out = _mg(*t)
    return out if indexing == "ij" else tuple(x.t() for x in out)
torch.meshgrid = _meshgrid
from adt_pose_value import AriaLocalPairs, DEFAULT_SEQ
src = AriaLocalPairs(os.path.expanduser(DEFAULT_SEQ), size=504)
th = src.camera.incidence_grid(src.h, src.w).numpy()
out = os.environ["S"]
np.save(out + "/theta.npy", th)
np.save(out + "/theta_max.npy", np.array([float(src.camera.theta_max)]))
with open(out + "/frames.txt", "w") as f:
    for p in src.paths: f.write(os.path.basename(p) + "\n")
print("ok", src.h, src.w, float(src.camera.theta_max), len(src.paths))
