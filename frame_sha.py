"""The digest says two boxes chose the same 300 frame KEYS. It says nothing
about the pixels behind them. This hashes what the split actually resolves to."""
import os, sys, hashlib
sys.path.insert(0, os.environ["REPO"])
from fovbench.split import build_split
sp = build_split(os.environ["ADT"], n_frames=50, verbose=False)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
    return h.hexdigest()[:16]
rows = []
for f in sp.frames:
    rows.append(f"{f.key} {sha(f.depth)} {sha(f.rgb['real'])} {sha(f.rgb['synthetic'])}")
rows.sort()
print("digest", sp.digest, "frames", len(rows))
print("rollup", hashlib.md5("\n".join(rows).encode()).hexdigest())
open(sys.argv[1], "w").write("\n".join(rows) + "\n")
