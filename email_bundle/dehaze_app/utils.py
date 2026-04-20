from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_psnr(prediction: torch.Tensor, target: torch.Tensor, max_pixel: float = 1.0, eps: float = 1e-8) -> float:
    mse = torch.nn.functional.mse_loss(prediction, target, reduction="none")
    mse = mse.flatten(start_dim=1).mean(dim=1)
    psnr = 20 * math.log10(max_pixel) - 10 * torch.log10(mse + eps)
    return psnr.mean().item()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
