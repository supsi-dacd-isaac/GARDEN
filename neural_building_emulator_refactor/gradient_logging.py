"""Persistent, low-overhead gradient diagnostics for LSTM training."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path


class GradientUpdateLogger:
    """Append one flushed row per update without retaining training batches."""

    def __init__(self, output_dir: Path, model_name: str) -> None:
        directory = Path(output_dir) / "artifacts" / model_name
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        self.path = directory / f"gradient_updates_{model_name}_{timestamp}.csv"
        self._file = self.path.open("x", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            (
                "epoch",
                "batch_index",
                "loss",
                "grad_norm_pre_clip",
                "gradient_clip_norm",
                "skip_grad_norm_above",
                "accepted",
                "reason",
            )
        )
        self._file.flush()

    def log(
        self,
        *,
        epoch: int,
        batch_index: int,
        loss: float,
        grad_norm: float,
        gradient_clip_norm: float,
        skip_grad_norm_above: float,
        accepted: bool,
        reason: str,
    ) -> None:
        self._writer.writerow(
            (
                epoch,
                batch_index,
                loss,
                grad_norm,
                gradient_clip_norm,
                skip_grad_norm_above,
                int(accepted),
                reason,
            )
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()
