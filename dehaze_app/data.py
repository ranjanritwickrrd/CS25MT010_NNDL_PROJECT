from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from .config import BATCH_SIZE, DATA_DIR, IMAGE_SIZE, MAX_KAGGLE_PAIRS, SEED


resize_to_tensor = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ]
)


def list_image_files(folder_path: Path) -> list[Path]:
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    return [path for path in folder_path.rglob("*") if path.suffix.lower() in image_extensions]


class PairedFolderDehazeDataset(Dataset):
    """Pairs hazy and clean images by matching file name."""

    def __init__(self, clean_dir: Path, hazy_dir: Path, max_pairs: int | None = None, seed: int = SEED) -> None:
        self.clean_dir = Path(clean_dir)
        self.hazy_dir = Path(hazy_dir)
        self.max_pairs = max_pairs
        self.seed = seed
        self.pairs = self._build_pairs()

    def _build_pairs(self) -> list[tuple[Path, Path]]:
        clean_lookup = {path.name: path for path in list_image_files(self.clean_dir)}
        pairs: list[tuple[Path, Path]] = []

        for hazy_path in list_image_files(self.hazy_dir):
            clean_path = clean_lookup.get(hazy_path.name)
            if clean_path is not None:
                pairs.append((clean_path, hazy_path))

        if self.max_pairs is not None and len(pairs) > self.max_pairs:
            rng = random.Random(self.seed)
            rng.shuffle(pairs)
            pairs = pairs[: self.max_pairs]

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        clean_path, hazy_path = self.pairs[index]
        clean_tensor = resize_to_tensor(Image.open(clean_path).convert("RGB"))
        hazy_tensor = resize_to_tensor(Image.open(hazy_path).convert("RGB"))
        return clean_tensor, hazy_tensor


def build_real_dataset(max_pairs: int = MAX_KAGGLE_PAIRS) -> PairedFolderDehazeDataset:
    clean_dir = DATA_DIR / "clean"
    hazy_dir = DATA_DIR / "hazy"
    dataset = PairedFolderDehazeDataset(clean_dir=clean_dir, hazy_dir=hazy_dir, max_pairs=max_pairs, seed=SEED)
    if len(dataset) == 0:
        raise RuntimeError(
            "No paired real dataset images were found in data/clean and data/hazy. "
            "Run the Kaggle setup first."
        )
    return dataset


def build_dataloaders(max_pairs: int = MAX_KAGGLE_PAIRS) -> tuple[DataLoader, DataLoader]:
    dataset = build_real_dataset(max_pairs=max_pairs)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    split_generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_generator)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader
