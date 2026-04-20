from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from .config import IMAGE_SIZE
from .models import MODEL_FACTORIES
from .training import checkpoint_path
from .utils import get_device


resize_to_tensor = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ]
)


def available_models() -> list[str]:
    return [name for name in MODEL_FACTORIES if checkpoint_path(name).exists()]


def load_model(model_name: str) -> torch.nn.Module:
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model '{model_name}'")

    ckpt_path = checkpoint_path(model_name)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for '{model_name}': {ckpt_path}")

    device = get_device()
    model = MODEL_FACTORIES[model_name]().to(device)
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def dehaze_pil_image(image: Image.Image, model_name: str) -> tuple[Image.Image, dict]:
    device = get_device()
    model = load_model(model_name)
    input_tensor = resize_to_tensor(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor).cpu().squeeze(0).clamp(0, 1)

    output_array = (output_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    output_image = Image.fromarray(output_array)

    info = {
        "model_name": model_name,
        "device": str(device),
        "input_size": f"{image.size[0]}x{image.size[1]}",
        "processed_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
    }
    return output_image, info
