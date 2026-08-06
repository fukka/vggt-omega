"""Stage a small, drop-in ScanNet++ sample for off-box analysis.

CPU-Claude cannot see `/netapp`, so every question about the data currently costs
a round-trip through GPU-Claude. A few MB of one scene removes that entirely: the
output is a directory with the *same layout* as a real ScanNet++ scene, so
``ScanNetPPFisheye(<out>)`` opens it unchanged and every script in this package --
``data_audit``, ``data_probes``, even ``train``/``eval`` on a couple of frames --
runs against it locally.

**This does not go in the repo.** `vggt-omega` is public and ScanNet++ is licensed
per-recipient; committing it would publish licensed data to people who hold no
licence, and `git rm` would not take it back -- history on a public repo is
permanent and mirrored. Stage it here, move it out of band, keep it out of git.
`.gitignore` should already cover the usual sample paths.

**What it takes, and why that is enough.**

* **The full `transforms.json`.** Poses and intrinsics for *every* frame -- a
  couple of MB of text, and the thing that matters most: it is what
  ``data_audit`` needs to characterise the whole trajectory, so the stride and
  identity-predictor analysis stays exact rather than being computed off a
  subset.
* **A handful of frames**, resized to the working resolution the pipeline uses
  anyway. Sampled as short runs at several offsets from a few anchors, so that
  pairs at stride 1, 2, 5, 10, 20, 40 and 60 all exist locally -- that is what
  makes the stride question answerable off-box.
* **Their masks and rendered depth**, when present, for the Ω and
  planar-z-vs-range probes.

Frames not copied are simply absent; the loader and probes skip unreadable ones,
which is why the full frame list can stay while the imagery does not.

Usage on the box::

    python -m raytun3r.experiments.make_local_sample \\
        --src /netapp/datasets/f.zhang2/scannetpp/data/3f15a9266d \\
        --out /sdcard/data/scannetpp_example

Then copy that directory to the Mac and point anything at it::

    python -m raytun3r.experiments.data_audit --path <copied>/3f15a9266d
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import List, Optional

__all__ = ["main"]

#: Offsets from each anchor. Chosen so consecutive-pair, and stride 1/2/5/10/20/
#: 40/60 pairs, all exist in the sample without copying whole runs.
DEFAULT_OFFSETS = (0, 1, 2, 5, 10, 20, 40, 60)

#: How many places in the sequence to anchor those offsets at.
DEFAULT_ANCHORS = 3


def _pick(n: int, anchors: int, offsets) -> List[int]:
    if n <= 0:
        return []
    span = max(offsets)
    out = set()
    for a in range(anchors):
        base = int((a + 0.5) * max(n - span, 1) / anchors)
        for o in offsets:
            if base + o < n:
                out.add(base + o)
    return sorted(out)


def _resize_keep(path_in: str, path_out: str, max_size: int, depth: bool) -> Optional[int]:
    """Copy one image, resized to ``max_size`` on the long side. Returns bytes."""
    from PIL import Image

    try:
        im = Image.open(path_in)
    except (FileNotFoundError, OSError):
        return None
    w, h = im.size
    s = min(max_size / max(w, h), 1.0)
    if s < 1.0:
        # NEAREST for depth and masks: they must not be interpolated across
        # discontinuities, and uint16 depth must survive as uint16.
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))),
                       Image.NEAREST if depth else Image.BICUBIC)
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    if depth:
        im.save(path_out)
    else:
        im.convert("RGB").save(path_out, quality=92)
    return os.path.getsize(path_out)


def main(argv=None) -> None:
    p = argparse.ArgumentParser("raytun3r.experiments.make_local_sample",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="a real ScanNet++ scene directory")
    p.add_argument("--out", required=True, help="where to stage the sample")
    p.add_argument("--anchors", type=int, default=DEFAULT_ANCHORS)
    p.add_argument("--offsets", default=",".join(str(o) for o in DEFAULT_OFFSETS))
    p.add_argument("--max-size", type=int, default=504,
                   help="long side of the copied imagery; the pipeline's working size")
    p.add_argument("--no-depth", action="store_true")
    p.add_argument("--no-masks", action="store_true")
    args = p.parse_args(argv)

    scene = os.path.basename(args.src.rstrip("/"))
    dst = os.path.join(args.out, scene)
    src_dslr = os.path.join(args.src, "dslr")
    dst_dslr = os.path.join(dst, "dslr")

    tsrc = os.path.join(src_dslr, "nerfstudio", "transforms.json")
    with open(tsrc) as f:
        meta = json.load(f)
    frames = meta.get("frames", [])
    offsets = [int(x) for x in args.offsets.split(",") if x.strip()]
    idx = _pick(len(frames), args.anchors, offsets)

    os.makedirs(os.path.join(dst_dslr, "nerfstudio"), exist_ok=True)
    shutil.copy2(tsrc, os.path.join(dst_dslr, "nerfstudio", "transforms.json"))

    img_root = os.path.join(src_dslr, "resized_images")
    if not os.path.isdir(img_root):
        img_root = os.path.join(src_dslr, "undistorted_images")

    took = {"images": 0, "masks": 0, "depth": 0}
    size = 0
    for i in idx:
        fr = frames[i]
        rel = fr["file_path"]
        stem = os.path.splitext(os.path.basename(rel))[0]

        n = _resize_keep(os.path.join(img_root, rel),
                         os.path.join(dst_dslr, os.path.basename(img_root), rel),
                         args.max_size, depth=False)
        if n:
            took["images"] += 1
            size += n

        if not args.no_masks:
            # ScanNet++ names the mask after the image stem; the per-frame
            # `mask_path`, when present, may be relative to either dslr/ or masks/.
            m = fr.get("mask_path")
            cands = [os.path.join(src_dslr, "masks", stem + ".png")]
            if m:
                cands = [os.path.join(src_dslr, m),
                         os.path.join(src_dslr, "masks", os.path.basename(m))] + cands
            for cand in cands:
                if os.path.exists(cand):
                    n = _resize_keep(cand, os.path.join(dst_dslr, "masks",
                                                        os.path.basename(cand)),
                                     args.max_size, depth=True)
                    if n:
                        took["masks"] += 1
                        size += n
                    break

        if not args.no_depth:
            n = _resize_keep(os.path.join(src_dslr, "render_depth", stem + ".png"),
                             os.path.join(dst_dslr, "render_depth", stem + ".png"),
                             args.max_size, depth=True)
            if n:
                took["depth"] += 1
                size += n

    tsize = os.path.getsize(os.path.join(dst_dslr, "nerfstudio", "transforms.json"))
    manifest = {
        "scene": scene, "source": args.src,
        "frames_in_transforms": len(frames), "frames_copied": idx,
        "counts": took, "max_size": args.max_size,
        "anchors": args.anchors, "offsets": offsets,
        "bytes": {"imagery": size, "transforms": tsize, "total": size + tsize},
        "strides_available": sorted({b - a for a in idx for b in idx if b > a}),
        "NOTICE": ("ScanNet++ is licensed per-recipient and this sample is derived "
                   "from it. Do not commit it to a public repository and do not "
                   "redistribute it. Transfer out of band."),
    }
    with open(os.path.join(dst, "SAMPLE_MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    mb = manifest["bytes"]["total"] / 1e6
    print(f"[sample] {scene}: {len(idx)} frames selected of {len(frames)}")
    print(f"[sample]   images {took['images']}  masks {took['masks']}  depth {took['depth']}")
    print(f"[sample]   transforms.json kept in full ({tsize/1e6:.2f} MB) -- data_audit "
          f"still sees the whole trajectory")
    print(f"[sample]   total {mb:.1f} MB at {dst}")
    print(f"[sample]   pair strides available locally: "
          f"{sorted(set(offsets) & set(manifest['strides_available']))}")
    print(f"[sample] NOTICE: licensed data. Move it out of band; do not commit it.")


if __name__ == "__main__":
    main()
