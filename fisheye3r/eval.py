"""Fisheye3R evaluation (paper Sec. 4.1 "Metrics", supp "Evaluation").

Tasks & metrics reproduced:
  camera pose (angular):  RRA@30, RTA@30, AUC@30 over all frame pairs
  camera pose (distance): ATE (RMSE of camera centers after Sim(3) Umeyama)
  depth map:              AbsRel, RMSE, delta_1 after per-sequence
                          scale-and-shift alignment (paper protocol)
  FoV:                    median horizontal/vertical FoV error in degrees
                          (from VGGT-Omega's 9D pose encoding)
Point-map Acc/Comp/CD are omitted: the minimal VGGT-Omega checkpoint exposes
depth + camera heads only (points would be a derived quantity here).

Two evaluation modes:
  --synthesize-fisheye : perspective test data is distorted on the fly with
      seeded KB cameras (ScanNet++-fisheye-style protocol); predictions are
      undistorted with T^-1 before comparison against perspective GT.
  default              : the dataset is real fisheye imagery with fisheye-
      domain GT depth (ADT-style); predictions are compared directly.

Usage:
  python -m fisheye3r.eval --weights checkpoints/vggt_omega.pt \
      --tokens runs/f3r/tokens_ssl_final.pt --data-root /data/adt_test \
      --lengths 4 8 16 32 --out runs/f3r/eval_adt.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vggt_omega.utils.pose_enc import encoding_to_camera, extri_intri_to_pose_encoding

from fisheye3r.data import SequenceDataset, sequence_collate
from fisheye3r.distortion import sample_kb_cameras, distort_images, undistort_dense
from fisheye3r.model import Fisheye3R
from fisheye3r.train import load_base, repeat_cam


# ------------------------------------------------------------------- metrics


def scale_shift_align(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Least-squares scale+shift on depth over the whole sequence (paper:
    'aligned to ground truth by applying a scale and shift per sequence')."""
    p, g = pred[valid], gt[valid]
    A = torch.stack([p, torch.ones_like(p)], dim=-1)
    sol = torch.linalg.lstsq(A, g.unsqueeze(-1)).solution.squeeze(-1)
    scale, shift = sol[0], sol[1]
    if not torch.isfinite(scale) or scale <= 0:
        scale = (g.median() / p.median().clamp(min=1e-9)).clamp(min=1e-9)
        shift = torch.zeros_like(scale)
    return pred * scale + shift


def depth_metrics(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    valid = valid & (gt > 0) & torch.isfinite(gt) & torch.isfinite(pred)
    if valid.sum() < 100:
        return {}
    pred = scale_shift_align(pred, gt, valid).clamp(min=1e-6)
    p, g = pred[valid], gt[valid]
    absrel = ((p - g).abs() / g).mean()
    rmse = torch.sqrt(((p - g) ** 2).mean())
    ratio = torch.maximum(p / g, g / p)
    return {
        "depth/AbsRel": float(absrel),
        "depth/RMSE": float(rmse),
        "depth/delta1": float((ratio < 1.25).float().mean()),
    }


def rotation_angle_deg(R: torch.Tensor) -> torch.Tensor:
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((tr - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cos))


def vector_angle_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    an = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    bn = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return torch.rad2deg(torch.acos((an * bn).sum(-1).clamp(-1, 1)))


def umeyama_sim3(src: torch.Tensor, dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Least-squares similarity transform src -> dst. src/dst: (N, 3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / src.shape[0]
    U, D, Vt = torch.linalg.svd(cov)
    S = torch.eye(3, device=src.device)
    if torch.det(U) * torch.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs**2).sum() / src.shape[0]
    scale = (D * S.diagonal()).sum() / var_s.clamp(min=1e-12)
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def pose_metrics(pred_extr: torch.Tensor, gt_extr: torch.Tensor, tau: float = 30.0) -> dict[str, float]:
    """pred/gt extrinsics: (S, 3or4, 4) camera-from-world."""
    S = pred_extr.shape[0]
    if S < 2:
        return {}
    Rp, tp = pred_extr[:, :3, :3], pred_extr[:, :3, 3]
    Rg, tg = gt_extr[:, :3, :3], gt_extr[:, :3, 3]

    rot_errs, trans_errs = [], []
    for i in range(S):
        for j in range(i + 1, S):
            Rrel_p = Rp[i] @ Rp[j].T
            Rrel_g = Rg[i] @ Rg[j].T
            rot_errs.append(rotation_angle_deg(Rrel_p @ Rrel_g.T))
            trel_p = tp[i] - Rrel_p @ tp[j]
            trel_g = tg[i] - Rrel_g @ tg[j]
            trans_errs.append(vector_angle_deg(trel_p, trel_g))
    rot = torch.stack(rot_errs)
    trans = torch.stack(trans_errs)

    thresholds = torch.arange(1, int(tau) + 1, device=rot.device).float()
    worst = torch.maximum(rot, trans)
    auc = (worst.unsqueeze(0) < thresholds.unsqueeze(1)).float().mean()

    centers_p = torch.einsum("sij,sj->si", Rp.transpose(1, 2), -tp)
    centers_g = torch.einsum("sij,sj->si", Rg.transpose(1, 2), -tg)
    try:
        s, R, t = umeyama_sim3(centers_p, centers_g)
        ate = torch.sqrt((((s * centers_p @ R.T + t) - centers_g) ** 2).sum(-1).mean())
    except Exception:
        ate = torch.tensor(float("nan"))

    return {
        "pose/RRA@30": float((rot < tau).float().mean()),
        "pose/RTA@30": float((trans < tau).float().mean()),
        "pose/AUC@30": float(auc),
        "pose/ATE": float(ate),
    }


def fov_metrics(pose_enc: torch.Tensor, gt_intr: torch.Tensor, hw: tuple[int, int]) -> dict[str, float]:
    H, W = hw
    fov_h = torch.rad2deg(pose_enc[:, 7])
    fov_w = torch.rad2deg(pose_enc[:, 8])
    gt_h = torch.rad2deg(2 * torch.atan((H / 2) / gt_intr[:, 1, 1]))
    gt_w = torch.rad2deg(2 * torch.atan((W / 2) / gt_intr[:, 0, 0]))
    return {
        "fov/vErr": float((fov_h - gt_h).abs().median()),
        "fov/hErr": float((fov_w - gt_w).abs().median()),
    }


# ---------------------------------------------------------------------- eval


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, float]:
    device = torch.device(args.device)
    base = load_base(args.weights, device)
    model = Fisheye3R(base, num_tokens=args.tokens_k, encoder_skip_layers=args.l0,
                      modules=args.modules).to(device).eval()
    if args.tokens:
        model.load_tokens(args.tokens, map_location=device)
    elif not args.baseline:
        raise ValueError("pass --tokens <ckpt> or --baseline for the unadapted model")

    accum: dict[str, list[float]] = defaultdict(list)
    for length in args.lengths:
        dataset = SequenceDataset(args.data_root, split="test", resolution=args.resolution,
                                  eval_length=length, augment=False)
        loader = DataLoader(dataset, batch_size=1, num_workers=args.workers,
                            collate_fn=sequence_collate)
        torch.manual_seed(args.seed)  # deterministic KB cameras across runs

        for batch in loader:
            images = batch["images"].to(device)
            B, S = images.shape[:2]
            H, W = images.shape[-2:]
            intr = batch.get("intrinsics")
            intr = intr.to(device) if intr is not None else None

            if args.synthesize_fisheye:
                seq_cam = sample_kb_cameras(
                    B, H, W,
                    perspective_intrinsics=intr[:, 0] if intr is not None else None,
                    device=device,
                )
                cam = repeat_cam(seq_cam, S)
                fish, _ = distort_images(images.view(B * S, 3, H, W), cam)
                inputs = fish.view(B, S, 3, H, W)
            else:
                cam, inputs = None, images

            if args.baseline:
                flags = torch.zeros(B, S, dtype=torch.bool, device=device)
            elif args.use_classifier:
                flags = None
            else:
                flags = torch.ones(B, S, dtype=torch.bool, device=device)
            preds = model(inputs, fisheye_flags=flags, use_classifier=args.use_classifier)

            # ---- depth ----
            if "depth" in batch:
                gt_depth = batch["depth"].to(device)
                pred_depth = preds["depth"]
                if args.synthesize_fisheye:
                    ud, valid = undistort_dense(pred_depth.view(B * S, 1, H, W), cam)
                    pred_depth = ud.view(B, S, H, W)
                    valid = valid.view(B, S, H, W)
                else:
                    valid = torch.ones_like(pred_depth, dtype=torch.bool)
                for k, v in depth_metrics(pred_depth, gt_depth, valid).items():
                    accum[k].append(v)

            # ---- pose / fov ----
            if "extrinsics" in batch:
                pred_extr, _ = encoding_to_camera(preds["pose_enc"], (H, W))
                for k, v in pose_metrics(pred_extr[0].float(), batch["extrinsics"][0, :, :3].to(device)).items():
                    accum[k].append(v)
                if intr is not None and args.synthesize_fisheye:
                    for k, v in fov_metrics(preds["pose_enc"][0], intr[0], (H, W)).items():
                        accum[k].append(v)

    results = {k: sum(v) / len(v) for k, v in sorted(accum.items()) if v}
    name = "baseline" if args.baseline else (args.tokens or "tokens")
    print(f"\n=== {Path(args.data_root).name} | {name} | lengths={args.lengths} ===")
    for k, v in results.items():
        print(f"  {k:>16s}: {v:.4f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--tokens", default=None)
    p.add_argument("--baseline", action="store_true", help="evaluate the unadapted backbone")
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lengths", type=int, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--tokens-k", type=int, default=8)
    p.add_argument("--l0", type=int, default=12)
    p.add_argument("--modules", default="EFG")
    p.add_argument("--synthesize-fisheye", action="store_true")
    p.add_argument("--use-classifier", action="store_true",
                   help="gate tokens per frame with the camera-type classifier (Sec. 3.4)")
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
