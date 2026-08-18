"""Trim the pod's videos_rgb inventory to the #019 pool.

The pod's ADT copy is a SUPERSET of lambda's for videos_rgb -- it carries real
frames lambda's export does not. fovbench's split is the intersection of the
three streams, so a superset stream makes a 400-frame pool where #019 had 399,
a different even spread, and a different digest (2f95f0c6b21d, not
601fcb22767e). Nothing is wrong with either copy; they are simply not the same
inventory, and the digest is doing exactly the job it exists for.

So build videos_rgb as a directory of symlinks holding only the ids in lambda's
intersection pool. That is not a thumb on the scale: the intersection is a
subset of every stream on both boxes, so restricting one stream to it leaves the
intersection itself untouched -- and the digest gate, not this argument, decides.
Contents are then compared file by file (sha256) before any model runs.
"""
import os, sys, glob, re, json
pool = json.load(open("/group-volume/Fengjia/temp/lambda_isect.json"))
ROOT = "/group-volume/Fengjia/data/adt-024-6seq"
FILT = "/group-volume/Fengjia/data/adt-024-rgbfilt"
rx = re.compile(r"frame[_-]?(\d+)")
def fid(p):
    m = rx.search(os.path.basename(p)); return str(int(m.group(1))) if m else None
for seq, ids in pool.items():
    want = set(ids)
    src = os.path.realpath(os.path.join(ROOT, seq, "videos_rgb"))
    dst = os.path.join(FILT, seq, "videos_rgb")
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(dst):
        os.unlink(os.path.join(dst, f))
    n = 0
    for ext in ("*.png","*.jpg","*.jpeg"):
        for p in glob.glob(os.path.join(src, ext)):
            if fid(p) in want:
                os.symlink(p, os.path.join(dst, os.path.basename(p))); n += 1
    link = os.path.join(ROOT, seq, "videos_rgb")
    if os.path.islink(link): os.unlink(link)
    os.symlink(dst, link)
    print(f"  {seq}: pool {len(want)} -> linked {n}")
