"""Huấn luyện mô hình dự báo giá nhà."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import MODEL_PATH, format_vnd, train_model


def main() -> None:
    print("Đang huấn luyện mô hình dự báo giá nhà...")
    metrics = train_model()
    print(f"\nĐã lưu mô hình tại: {MODEL_PATH}")
    print(f"Số mẫu huấn luyện: {metrics['samples']:,}")
    print(f"MAE:  {format_vnd(metrics['mae'])}")
    print(f"RMSE: {format_vnd(metrics['rmse'])}")
    print(f"R²:   {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
