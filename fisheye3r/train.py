"""Fisheye3R training: calibration tokens on a frozen VGGT-Omega.

Usage (paper defaults; supp Sec. 6 "Training"):

  # 1) fit the camera-type classifier (logistic regression, ~1 min)
  python -m fisheye3r.train fit-classifier \
      --weights checkpoints/vggt_omega.pt --data-root /data/perspective --out runs/f3r

  # 2) train tokens - scheme is one of ssl | sl | slplus
  python -m fisheye3r.train train --scheme ssl \
      --weights checkpoints/vggt_omega.pt --data-root /data/perspective \
      --out runs/f3r --iters 40000

SSL needs only unlabeled perspective RGB (Eq. 11). SL additionally uses GT
depth + cameras of the perspective data (Eq. 12). SL+ trains directly on a
fisheye dataset with GT (Eq. 13) - point --data-root at it and no distortion
synthesis is applied.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vggt_omega.models.vggt_omega import VGGTOmega
from vggt_omega.utils.pose_enc import extri_intri_to_pose_encoding

from fisheye3r.data import SequenceDataset, sequence_collate
from fisheye3r.distortion import KannalaBrandtCamera, distort_images, sample_kb_cameras
from fisheye3r.losses import scheme_loss
from fisheye3r.model import Fisheye3R


def load_base(weights: str, device: torch.device) -> VGGTOmega:
    model = VGGTOmega()
    state = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load_base] missing={len(missing)} unexpected={len(unexpected)} keys")
    return model.to(device).eval()


def repeat_cam(cam: KannalaBrandtCamera, times: int) -> KannalaBrandtCamera:
    fields = {
        k: (v.repeat_interleave(times, dim=0) if torch.is_tensor(v) else v)
        for k, v in cam.__dict__.items()
    }
    return KannalaBrandtCamera(**fields)


def pick_flags(batch_shape: tuple[int, int], camera_mix: str, device: torch.device) -> torch.Tensor:
    """Sequence-level camera-type assignment (supp: fully perspective /
    fully fisheye / hybrid with p=0.5 per frame)."""
    B, S = batch_shape
    if camera_mix == "fisheye":
        return torch.ones(B, S, dtype=torch.bool, device=device)
    kind = torch.randint(0, 3, (B,), device=device)
    flags = torch.rand(B, S, device=device) < 0.5
    flags[kind == 0] = False
    flags[kind == 1] = True
    return flags


def synthesize(images: torch.Tensor, flags: torch.Tensor, intrinsics: torch.Tensor | None):
    """Distort the flagged frames of a perspective batch. Returns
    (distorted images, per-frame KB cameras)."""
    B, S, C, H, W = images.shape
    seq_cam = sample_kb_cameras(
        B, H, W,
        perspective_intrinsics=intrinsics[:, 0] if intrinsics is not None else None,
        device=images.device,
    )
    cam = repeat_cam(seq_cam, S)
    flat = images.view(B * S, C, H, W)
    fish, _ = distort_images(flat, cam)
    f = flags.view(B * S, 1, 1, 1)
    return torch.where(f, fish, flat).view(B, S, C, H, W), cam


# --------------------------------------------------------------------- train


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    base = load_base(args.weights, device)
    model = Fisheye3R(
        base,
        num_tokens=args.tokens_k,
        encoder_skip_layers=args.l0,
        modules=args.modules,
        grad_checkpoint=args.grad_checkpoint,
    ).to(device)
    if args.resume_tokens:
        model.load_tokens(args.resume_tokens, map_location=device)
    print(f"trainable parameters: {model.num_trainable_parameters():,}")

    dataset = SequenceDataset(
        args.data_root,
        split="train",
        resolution=args.resolution,
        max_frames=args.max_frames,
        require_depth=args.scheme in ("sl", "slplus"),
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=args.workers,
        collate_fn=sequence_collate, drop_last=True,
    )

    params = list(model.trainable_parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    def lr_at(step: int) -> float:  # cosine 1e-5 -> 1e-7 (supp Sec. 6)
        t = min(step / max(args.iters, 1), 1.0)
        return args.lr_end + 0.5 * (args.lr - args.lr_end) * (1 + math.cos(math.pi * t))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"train_{args.scheme}.jsonl"

    step, t0 = 0, time.time()
    data_iter = iter(loader)
    while step < args.iters:
        opt.zero_grad(set_to_none=True)
        logs = {}
        for _ in range(args.accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            images = batch["images"].to(device)
            B, S = images.shape[:2]
            H, W = images.shape[-2:]
            intr = batch.get("intrinsics")
            intr = intr.to(device) if intr is not None else None

            if args.scheme == "slplus":
                # Real fisheye + GT: direct supervision, no synthesis (Eq. 13).
                flags = torch.ones(B, S, dtype=torch.bool, device=device)
                student = model(images, fisheye_flags=flags)
                extr = batch["extrinsics"].to(device)
                target_enc = extri_intri_to_pose_encoding(extr, intr, (H, W))
                target_depth = batch["depth"].to(device)
                loss, logs = scheme_loss(
                    student, target_depth, target_enc, cam=None, flags=flags,
                    target_valid=target_depth > 0, already_undistorted=True,
                    supervise_fov_on_fisheye=False,
                )
            else:
                flags = pick_flags((B, S), args.camera_mix, device)
                distorted, cam = synthesize(images, flags, intr)
                student = model(distorted, fisheye_flags=flags)

                if args.scheme == "ssl":
                    # Pseudo-labels: the frozen model on the original
                    # perspective frames (Eq. 11). Running the wrapper with
                    # all-False flags is exactly the frozen backbone.
                    with torch.no_grad():
                        teacher = model(images, fisheye_flags=torch.zeros_like(flags))
                    target_depth, target_enc = teacher["depth"], teacher["pose_enc"]
                    teacher_conf = teacher["depth_conf"]
                    target_valid = None
                else:  # sl
                    extr = batch["extrinsics"].to(device)
                    target_enc = extri_intri_to_pose_encoding(extr, intr, (H, W))
                    target_depth = batch["depth"].to(device)
                    teacher_conf, target_valid = None, target_depth > 0

                loss, logs = scheme_loss(
                    student, target_depth, target_enc, cam=cam, flags=flags,
                    target_valid=target_valid, teacher_conf=teacher_conf,
                )
            (loss / args.accum).backward()

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.step()
        step += 1

        if step % args.log_every == 0:
            rec = {"step": step, "lr": lr_at(step), "sec": round(time.time() - t0, 1), **logs}
            print(json.dumps(rec))
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if step % args.save_every == 0 or step == args.iters:
            model.save_tokens(out_dir / f"tokens_{args.scheme}_{step:06d}.pt")
    model.save_tokens(out_dir / f"tokens_{args.scheme}_final.pt")


# ----------------------------------------------------------- fit-classifier


def fit_classifier(args: argparse.Namespace) -> None:
    """Pre-train the logistic-regression camera-type classifier (Eq. 14).
    The paper reports convergence in ~1 minute and 99.9%+ accuracy."""
    device = torch.device(args.device)
    base = load_base(args.weights, device)
    model = Fisheye3R(base, num_tokens=args.tokens_k, encoder_skip_layers=args.l0).to(device)

    dataset = SequenceDataset(args.data_root, split="train", resolution=args.resolution, max_frames=4)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.workers,
                        collate_fn=sequence_collate)

    feats, labels = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            B, S = images.shape[:2]
            flags = torch.rand(B, S, device=device) < 0.5
            intr = batch.get("intrinsics")
            distorted, _ = synthesize(images, flags, intr.to(device) if intr is not None else None)
            feats.append(model.l0_class_features(distorted).reshape(-1, model.camera_classifier.in_features).cpu())
            labels.append(flags.reshape(-1).float().cpu())
            if sum(f.shape[0] for f in feats) >= args.classifier_samples:
                break
    X = torch.cat(feats).to(device)
    y = torch.cat(labels).to(device)

    clf = model.camera_classifier.to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
    for it in range(500):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(clf(X).squeeze(-1), y)
        loss.backward()
        opt.step()
    acc = ((clf(X).squeeze(-1) > 0) == (y > 0.5)).float().mean()
    print(f"classifier accuracy on {len(y)} frames: {acc:.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_tokens(out_dir / "tokens_classifier_only.pt")


# ------------------------------------------------------------------ argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--weights", required=True, help="VGGT-Omega checkpoint (.pt)")
        sp.add_argument("--data-root", required=True)
        sp.add_argument("--out", required=True)
        sp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        sp.add_argument("--resolution", type=int, default=512)
        sp.add_argument("--workers", type=int, default=4)
        sp.add_argument("--tokens-k", type=int, default=8)
        sp.add_argument("--l0", type=int, default=12)

    t = sub.add_parser("train")
    common(t)
    t.add_argument("--scheme", choices=["ssl", "sl", "slplus"], required=True)
    t.add_argument("--modules", default="EFG", help="subset of EFG to calibrate (Tab. 4)")
    t.add_argument("--iters", type=int, default=40_000)
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--lr-end", type=float, default=1e-7)
    t.add_argument("--accum", type=int, default=1)
    t.add_argument("--max-frames", type=int, default=24)
    t.add_argument("--camera-mix", choices=["fisheye", "mixed"], default="fisheye",
                   help="'fisheye' trains on fully-distorted sequences (supp: masked "
                        "attention can then be applied only at inference)")
    t.add_argument("--grad-checkpoint", action="store_true")
    t.add_argument("--resume-tokens", default=None)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--save-every", type=int, default=2000)
    t.set_defaults(func=train)

    c = sub.add_parser("fit-classifier")
    common(c)
    c.add_argument("--classifier-samples", type=int, default=4096)
    c.set_defaults(func=fit_classifier)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
