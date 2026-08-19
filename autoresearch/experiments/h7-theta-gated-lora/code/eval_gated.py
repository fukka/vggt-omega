"""H7 eval: the H5 eval_lora protocol with gated-LoRA loading.

Wraps h5 eval_lora.main, replacing its load_lora with the gated loader and
arming the theta context first. Guards against the gate silently not
applying (token-count mismatch would make the arm look uniform).

Usage (repo root):
    <venv>/bin/python autoresearch/experiments/h7-theta-gated-lora/code/eval_gated.py \
        --seq <seq dir> --lora <gated_lora_last.pt> --size 252 --out <json>
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "h5-rim-finetune" / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "h1-rim-pose-value" / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gated_lora  # noqa: E402
from adt_pose_value import AriaLocalPairs  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "h5_eval", Path(__file__).resolve().parents[2]
    / "h5-rim-finetune" / "code" / "eval_lora.py")
_h5e = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_h5e)


def main(argv=None) -> None:
    # peek at seq/size to arm the theta context before h5e.main runs
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--seq")
    peek.add_argument("--size", type=int, default=504)
    known, _ = peek.parse_known_args(argv)
    import os
    cam = AriaLocalPairs(os.path.expanduser(known.seq), size=known.size).camera
    h = w = known.size
    theta = cam.incidence_grid(h, w)
    gh, gw = h // 14, w // 14
    theta_p = theta.reshape(gh, 14, gw, 14).mean((1, 3)).ravel()
    applied = {"n": 0}

    def load_gated(net, ckpt_path: str) -> int:
        ck = torch.load(ckpt_path, map_location="cpu")
        assert ck.get("gated"), f"{ckpt_path} is not a gated checkpoint"
        n = gated_lora.load_into(net, ck)
        # size the context from the actual token stream: one probe forward
        vit = None
        for m in net.modules():
            if hasattr(m, "blocks") or hasattr(m, "layers"):
                vit = m
                break
        blocks = getattr(vit, "blocks", None) or getattr(vit, "layers")
        seen = {}
        hk = blocks[-1].register_forward_hook(
            lambda _m, _i, out: seen.__setitem__(
                "n", (out[0] if isinstance(out, tuple) else out)
                .reshape(-1, (out[0] if isinstance(out, tuple) else out)
                         .shape[-1]).shape[0]))
        with torch.no_grad():
            net.backbone(torch.zeros(1, 1, 3, h, w), cam_token=None,
                         export_feat_layers=[])
        hk.remove()
        gated_lora.set_theta(theta_p, seen["n"], float(cam.theta_max))
        assert gated_lora._CTX["feat"].shape[0] == seen["n"]
        # instrument one module to prove the gate path runs during eval
        mods = [m for m in net.modules()
                if isinstance(m, gated_lora.GatedLoRALinear)]
        orig = mods[0].gate

        def counting_gate(device, dtype):
            applied["n"] += 1
            return orig(device, dtype)
        mods[0].gate = counting_gate
        print(f"[h7-eval] gated LoRA loaded into {n} layers, "
              f"theta context armed for {seen['n']} tokens")
        return n

    _h5e.load_lora = load_gated
    _h5e.main(argv)
    assert applied["n"] > 0, "gate never ran during eval — token mismatch?"
    print(f"[h7-eval] gate applied in {applied['n']} forward calls — OK")


if __name__ == "__main__":
    main()
