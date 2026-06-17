# Copyright (c) 2026.
"""Lightweight training logger: JSONL + CSV + optional TensorBoard + curves.

One ``TrainLogger`` per run, constructed on rank 0 only (pass ``enabled=False``
on other ranks to get a no-op). It records every logged metric to:

    <out_dir>/metrics.jsonl   one JSON object per log call (robust, streaming)
    <out_dir>/metrics.csv     written at close() — union of all metric columns
    <out_dir>/train_log.txt   human-readable console mirror
    <out_dir>/tb/             TensorBoard event files (if --tensorboard)
    <out_dir>/loss_curves.png rendered at close() (if matplotlib is available)

The in-memory ``history`` (list of row dicts) is what the curve plot reads, so
plotting needs no file round-trip.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List


class TrainLogger:
    def __init__(self, out_dir: str, enabled: bool = True, use_tensorboard: bool = False) -> None:
        self.enabled = enabled
        self.out_dir = out_dir
        self.history: List[dict] = []
        self._jsonl = None
        self._txt = None
        self.tb = None
        if not enabled:
            return
        os.makedirs(out_dir, exist_ok=True)
        # line-buffered so a crash still leaves a complete log up to the last line
        self._jsonl = open(os.path.join(out_dir, "metrics.jsonl"), "a", buffering=1)
        self._txt = open(os.path.join(out_dir, "train_log.txt"), "a", buffering=1)
        if use_tensorboard:
            tb_dir = os.path.join(out_dir, "tb")
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(tb_dir)
                self.text(f"[logger] TensorBoard -> {tb_dir}  (view: tensorboard --logdir {out_dir})")
            except Exception as e:  # pragma: no cover
                # Most commonly the 'tensorboard' pip package is missing; do not
                # silently train for hours without the requested logging.
                self.text(
                    f"[logger] WARNING: --tensorboard requested but unavailable: {e}\n"
                    f"[logger]          install it with `pip install tensorboard` "
                    f"(metrics.jsonl/csv are still written regardless)."
                )

    # ------------------------------------------------------------------ #
    def log_scalars(self, step: int, split: str, phase: str, metrics: Dict[str, float]) -> None:
        """Record a row of scalar metrics. ``split`` in {train, val}; ``phase`` in {A, B}."""
        row = {"step": int(step), "split": split, "phase": phase,
               **{k: float(v) for k, v in metrics.items()}}
        self.history.append(row)
        if not self.enabled:
            return
        self._jsonl.write(json.dumps(row) + "\n")
        if self.tb is not None:
            for k, v in metrics.items():
                self.tb.add_scalar(f"{split}/{phase}/{k}", float(v), int(step))
            self.tb.flush()  # make events visible promptly (don't wait for close/120s)

    def text(self, line: str) -> None:
        """Print to console and mirror into train_log.txt (rank-0 / enabled only)."""
        if not self.enabled:
            return
        print(line, flush=True)
        if self._txt is not None:
            self._txt.write(line + "\n")

    # ------------------------------------------------------------------ #
    def _write_csv(self) -> None:
        import csv

        if not self.history:
            return
        keys: List[str] = []
        for row in self.history:
            for k in row:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(self.out_dir, "metrics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.history)

    def _plot_curves(self) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:  # pragma: no cover
            self.text(f"[logger] matplotlib unavailable ({e}); skipping loss curves.")
            return

        phases = ["A", "B"]
        fig, axes = plt.subplots(1, len(phases), figsize=(6 * len(phases), 4), squeeze=False)
        for ax, phase in zip(axes[0], phases):
            for split, style in (("train", "-"), ("val", "o-")):
                pts = [(r["step"], r["total"]) for r in self.history
                       if r["phase"] == phase and r["split"] == split and "total" in r]
                if pts:
                    xs, ys = zip(*sorted(pts))
                    ax.plot(xs, ys, style, label=f"{split}", markersize=4)
            ax.set_title(f"Phase {phase} total loss")
            ax.set_xlabel("global step")
            ax.set_ylabel("loss")
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.tight_layout()
        path = os.path.join(self.out_dir, "loss_curves.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        self.text(f"[logger] wrote {path}")

    def close(self) -> None:
        if not self.enabled:
            return
        self._write_csv()
        self._plot_curves()
        if self.tb is not None:
            self.tb.close()
        if self._jsonl is not None:
            self._jsonl.close()
        if self._txt is not None:
            self._txt.close()
