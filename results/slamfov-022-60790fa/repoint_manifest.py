# Repoint #020's manifest from lambda_63's data root to the pod's.
# The frame SELECTION is the ticket's contract; the paths are not part of it.
import json, os, sys

src, dst, old, new = sys.argv[1:5]
m = json.load(open(src))
assert m["root"] == old, f"unexpected root {m['root']!r}"
m["root"] = new
missing = []
for f in m["frames"]:
    for k in ("npz", "video"):
        f[k] = new + f[k][len(old):] if f[k].startswith(old) else f[k]
        if not os.path.exists(f[k]):
            missing.append(f[k])
if missing:
    print(f"MISSING {len(missing)} of {2*len(m['frames'])} referenced files, e.g.:")
    for p in missing[:5]:
        print("  ", p)
    sys.exit(1)
json.dump(m, open(dst, "w"))
print(f"repointed ok: {len(m['frames'])} frames, digest {m['digest']}, all files present")
