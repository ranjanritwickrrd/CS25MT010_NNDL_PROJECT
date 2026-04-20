from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dehaze_app.training import train_model


def main() -> None:
    for model_name in ["baseline", "residual", "skip"]:
        train_model(model_name)


if __name__ == "__main__":
    main()
