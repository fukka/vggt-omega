# Copyright (c) 2026.
"""Run `slambench.run` with VGGT's aggregator in bf16, without editing anything.

`raytun3r.backbones.DA3Backbone` and VGGT-Omega's own forward already run in
bf16; the vendored VGGT does not. This wraps `VGGTBackbone.forward` in the same
autocast the other two use and then hands straight over to the ordinary CLI, so
the only difference between this and `python -m slambench.run` is the dtype.

It is a wrapper and not a patch to `raytun3r/` on purpose: that module is shared
with the FOV experiment, and changing it there would move fovbench's published
VGGT-1B numbers as a side effect of speeding up this one.

    python slam020_bf16_run.py --egosynth-root ... --models vggt_1b ...
"""
from __future__ import annotations

import sys

import torch

from raytun3r.backbones import VGGTBackbone


def enable():
    orig = VGGTBackbone.forward

    def forward(self, images):
        if not images.is_cuda:
            return orig(self, images)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=dtype):
            return orig(self, images)

    VGGTBackbone.forward = forward
    print(f"[bf16] VGGTBackbone.forward wrapped in autocast "
          f"({'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'})",
          file=sys.stderr)


if __name__ == "__main__":
    enable()
    from slambench.run import main
    main()
