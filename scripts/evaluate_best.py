"""重載 best checkpoint 對 test 集評估（M4+ 通用工具）。

用途：訓練後（或拿到他人 checkpoint 後）重新產生評估報表與圖 —
不重訓、只載入 state_dict 評估。

用法：
  # Docker（CPU，含載入 ~4 分鐘）或 Colab（GPU，秒級）
  python scripts/evaluate_best.py --ckpt artifacts/m4_baseline/best_model.pt

產出：console per-class 表 + <ckpt 目錄>/confusion_matrix.png（LogNorm 色階）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wm811k import config  # noqa: E402
from wm811k.data import WM811KDataset, build_splits, labeled_frame  # noqa: E402
from wm811k.evaluate import evaluate_model  # noqa: E402
from wm811k.loader import load_lswmd  # noqa: E402
from wm811k.model import WaferCNN  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/m4_baseline/best_model.pt")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/4] Loading LSWMD.pkl ...", flush=True)
    df = load_lswmd()
    lf = labeled_frame(df)
    _, _, te = build_splits(lf)

    print(f"[2/4] Loading checkpoint {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location=device)
    print(f"      (trained to epoch {ckpt['epoch']}, "
          f"val macro-F1 {ckpt['val_macro_f1']:.4f})")
    model = WaferCNN(num_classes=config.NUM_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    print(f"[3/4] Evaluating on test ({len(te):,} samples, {device}) ...")
    loader = DataLoader(WM811KDataset(lf, te), batch_size=args.batch_size)
    out_png = str(Path(args.ckpt).parent / "confusion_matrix.png")
    evaluate_model(model, loader, device=device, out_png=out_png,
                   title="Test confusion matrix (LogNorm)")
    print(f"[4/4] Confusion matrix saved to {out_png}")


if __name__ == "__main__":
    main()
