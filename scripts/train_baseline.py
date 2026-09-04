"""M4 入口：訓練 WaferCNN 基線 + test 評估。

用法：
  # Docker 冒煙測試（每類 64 張、2 epochs，確認數值正常，<5 分鐘）
  docker compose run --rm app python scripts/train_baseline.py --smoke

  # Docker 短訓練（CPU 也可跑，先看趨勢）
  docker compose run --rm app python scripts/train_baseline.py --epochs 5

  # Colab 正式訓練（GPU，建議 30 epochs + early stop）
  python scripts/train_baseline.py            # 全量 121k train / 30 epochs

產出（artifacts/m4_baseline/）：
  best_model.pt      # best val macro-F1 的 state_dict（含訓練 config）
  history.json       # 每 epoch loss/macro-F1 + best 摘要
  learning_curve.png # 學習曲線
  confusion_matrix.png  # test 集混淆矩陣（best checkpoint 重載後評估）

Colab 起手（正式訓練主場）：
  !git clone https://github.com/Andrewneteye4343/wm811k-defect-classification
  %cd wm811k-defect-classification
  !pip install -q -r requirements.txt
  !mkdir -p data/raw
  import kagglehub
  p = kagglehub.dataset_download("qingyi/wm811k-wafer-map")
  !cp "{p}/LSWMD.pkl" data/raw/
  !python scripts/train_baseline.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# bootstrap：讓腳本可直接執行（python scripts/x.py 找不到 src/wm811k）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wm811k import config  # noqa: E402
from wm811k.data import WM811KDataset, build_splits, labeled_frame  # noqa: E402
from wm811k.evaluate import evaluate_model  # noqa: E402
from wm811k.loader import load_lswmd  # noqa: E402
from wm811k.model import WaferCNN  # noqa: E402
from wm811k.train import run_training  # noqa: E402


def _smoke_sample(lf, n_per_class: int = 64, seed: int = 42):
    """冒煙測試：每類隨機抽 n 張（保持不平衡結構但縮小 1000 倍）。"""
    rng = np.random.default_rng(seed)
    keep = []
    for cls in config.FAILURE_TYPES:
        idxs = np.where(lf["label"].to_numpy() == cls)[0]
        pick = rng.choice(idxs, size=min(n_per_class, len(idxs)),
                          replace=False)
        keep.extend(pick)
    return lf.iloc[sorted(keep)].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train WaferCNN baseline (M4)")
    ap.add_argument("--smoke", action="store_true",
                    help="每類抽 64 張跑 2 epochs 冒煙測試")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--out-dir", default="artifacts/m4_baseline")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    if args.smoke:
        args.epochs = min(args.epochs, 2)
        print("SMOKE MODE: 每類抽 64 張、2 epochs（驗證數值正常）")

    print(f"[1/5] Loading LSWMD.pkl ...", flush=True)
    df = load_lswmd()
    lf = labeled_frame(df)
    if args.smoke:
        lf = _smoke_sample(lf)
        print(f"      smoke subset: {len(lf):,} labeled rows")

    print(f"[2/5] Stratified split 70/15/15 (seed {args.seed}) ...")
    tr, va, te = build_splits(lf, seed=args.seed)
    print(f"      train {len(tr):,} / val {len(va):,} / test {len(te):,}")

    print("[3/5] Building DataLoaders ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator().manual_seed(args.seed)
    ds_tr = WM811KDataset(lf, tr)
    ds_va = WM811KDataset(lf, va)
    ds_te = WM811KDataset(lf, te)
    train_loader = DataLoader(ds_tr, batch_size=args.batch_size,
                              shuffle=True, generator=gen)
    val_loader = DataLoader(ds_va, batch_size=args.batch_size)
    test_loader = DataLoader(ds_te, batch_size=args.batch_size)

    print(f"[4/5] Training on {device} ...")
    model = WaferCNN(num_classes=config.NUM_CLASSES)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      WaferCNN parameters: {n_params:,}")
    history = run_training(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, device=device,
        out_dir=args.out_dir, patience=args.patience, seed=args.seed,
    )

    print(f"[5/5] Evaluating best checkpoint on test set ...")
    best = torch.load(Path(args.out_dir) / "best_model.pt",
                      map_location=device)
    model.load_state_dict(best["state_dict"])
    metrics = evaluate_model(
        model, test_loader, device=device,
        out_png=str(Path(args.out_dir) / "confusion_matrix.png"),
        title="Test confusion matrix",
    )
    print(f"\nDone. Best val macro-F1: {history['best_val_macro_f1']:.4f} "
          f"at epoch {history['best_epoch']} | "
          f"test macro-F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
