# Copyright (c) 2026.
"""Download Depth-Any-Camera config + weights from the HF release.

UniK3D weights download automatically on first model build (via huggingface_hub),
so only DAC needs an explicit fetch. Files land in ``--out`` (default
``checkpoints/``), matching the paths the DAC repo's own scripts expect.

    python -m finetune.eval.baselines.download_weights              # swinl indoor
    python -m finetune.eval.baselines.download_weights --variant dac_resnet101_indoor

Repo: https://huggingface.co/yuliangguo/depth-any-camera
"""
from __future__ import annotations

import argparse
import os

_REPO = "yuliangguo/depth-any-camera"
# variant -> (config.json, weights.pt) filenames in the HF repo.
_VARIANTS = {
    "dac_swinl_indoor": ("dac_swinl_indoor.json", "dac_swinl_indoor.pt"),
    "dac_resnet101_indoor": ("dac_resnet101_indoor.json", "dac_resnet101_indoor.pt"),
    "dac_swinl_outdoor": ("dac_swinl_outdoor.json", "dac_swinl_outdoor.pt"),
    "dac_resnet101_outdoor": ("dac_resnet101_outdoor.json", "dac_resnet101_outdoor.pt"),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default="dac_swinl_indoor", choices=list(_VARIANTS))
    p.add_argument("--out", default="checkpoints")
    a = p.parse_args()

    from huggingface_hub import hf_hub_download

    os.makedirs(a.out, exist_ok=True)
    for fn in _VARIANTS[a.variant]:
        path = hf_hub_download(repo_id=_REPO, filename=fn, repo_type="model",
                               local_dir=a.out)
        print(f"[download] {fn} -> {path}")
    cfg = os.path.join(a.out, _VARIANTS[a.variant][0])
    wts = os.path.join(a.out, _VARIANTS[a.variant][1])
    print(f"\nUse:\n  --dac-config {cfg} --dac-weights {wts}")


if __name__ == "__main__":
    main()
