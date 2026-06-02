"""Run full project with one command: train then start web app."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    print("==> Step 1/2: Training model...")
    subprocess.run([sys.executable, str(ROOT / "src" / "train.py")], check=True)

    print("\n==> Step 2/2: Starting web app at http://127.0.0.1:5000")
    subprocess.run([sys.executable, str(ROOT / "app.py")], check=True)


if __name__ == "__main__":
    main()
