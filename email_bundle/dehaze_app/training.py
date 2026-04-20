from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .config import CHECKPOINT_DIR, EPOCHS, GRAD_CLIP_NORM, LEARNING_RATE, MAX_KAGGLE_PAIRS, SEED, WEIGHT_DECAY
from .data import build_dataloaders
from .models import MODEL_FACTORIES
from .utils import compute_psnr, get_device, save_json, seed_everything


def evaluate_model(model: nn.Module, data_loader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    batches = 0

    with torch.no_grad():
        for clean_images, hazy_images in data_loader:
            clean_images = clean_images.to(device)
            hazy_images = hazy_images.to(device)
            outputs = model(hazy_images)
            loss = criterion(outputs, clean_images)
            running_loss += loss.item()
            running_psnr += compute_psnr(outputs, clean_images)
            batches += 1

    return running_loss / max(batches, 1), running_psnr / max(batches, 1)


def checkpoint_path(model_name: str) -> Path:
    return CHECKPOINT_DIR / f"{model_name}_best.pt"


def metrics_path(model_name: str) -> Path:
    return CHECKPOINT_DIR / f"{model_name}_metrics.json"


def train_model(model_name: str, epochs: int = EPOCHS, max_pairs: int = MAX_KAGGLE_PAIRS) -> dict:
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model '{model_name}'. Expected one of {sorted(MODEL_FACTORIES)}")

    seed_everything(SEED)
    device = get_device()
    train_loader, val_loader = build_dataloaders(max_pairs=max_pairs)
    model = MODEL_FACTORIES[model_name]().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = {"train_loss": [], "val_loss": [], "val_psnr": [], "best_epoch": None}
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for clean_images, hazy_images in train_loader:
            clean_images = clean_images.to(device)
            hazy_images = hazy_images.to(device)

            outputs = model(hazy_images)
            loss = criterion(outputs, clean_images)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss, val_psnr = evaluate_model(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)

        print(
            f"[{model_name}] Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val PSNR: {val_psnr:.2f} dB"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_epoch"] = epoch
            torch.save(
                {
                    "model_name": model_name,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_psnr": val_psnr,
                },
                checkpoint_path(model_name),
            )

    history["best_val_loss"] = best_val_loss
    history["best_val_psnr"] = max(history["val_psnr"]) if history["val_psnr"] else None
    save_json(metrics_path(model_name), history)
    return history


def ensure_checkpoint(model_name: str) -> Path:
    path = checkpoint_path(model_name)
    if not path.exists():
        print(f"Checkpoint for '{model_name}' not found. Training now...")
        train_model(model_name)
    return path
