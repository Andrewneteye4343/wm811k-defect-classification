"""M5+ 通用實驗入口：一次一變因，自動記錄 runs.csv。

變因旗標（可任意組合）：
  --loss-weight    方案 A：CrossEntropy 加頻率反比權重（改 loss 代價）
  --balanced-sampler 方案 B：WeightedRandomSampler（改資料餵食分佈）
  --augment        方案 C：D4 旋轉+鏡像增強（改資料多樣性）
  （無旗標 = baseline，與 M4 相同的訓練設定）

用法（Docker 冒煙 / Colab 正式）：
  python scripts/train_experiment.py --run-name m5a_weighted --loss-weight
  python scripts/train_experiment.py --run-name m5b_sampler --balanced-sampler
  python scripts/train_experiment.py --run-name m5c_augment --augment
  python scripts/train_experiment.py --run-name m5d_combined \
      --loss-weight --augment          # 合併最有效的（M5 末段決定）

產出：
  artifacts/runs/<run-name>/   # best_model.pt + history.json + 圖
  experiments/runs.csv         # 一行一實驗（append）— 實驗對照表
每輪固定：30 epochs、lr 1e-3、seed 42、同一切分 → 只動宣告的變因。
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wm811k import config  # noqa: E402
from wm811k.augment import random_d4  # noqa: E402
from wm811k.data import (  # noqa: E402
    WM811KDataset, balanced_sampler, build_splits, build_splits_by_lot,
    class_weights, label_counts, labeled_frame,
)
from wm811k.evaluate import evaluate_model  # noqa: E402
from wm811k.loader import load_lswmd  # noqa: E402
from wm811k.model import ResNetWafer, WaferCNN  # noqa: E402
from wm811k.train import run_training  # noqa: E402

RUNS_CSV = Path(__file__).resolve().parents[1] / "experiments" / "runs.csv"


def _smoke_sample(lf, n_per_class: int = 64, seed: int = 42):
    rng = np.random.default_rng(seed)
    keep = []
    for cls in config.FAILURE_TYPES:
        idxs = np.where(lf["label"].to_numpy() == cls)[0]
        pick = rng.choice(idxs, size=min(n_per_class, len(idxs)),
                          replace=False)
        keep.extend(pick)
    return lf.iloc[sorted(keep)].reset_index(drop=True)


def _append_run(row: dict) -> None:
    """runs.csv append 一行（首次執行建檔含 header）。"""
    df = pd.DataFrame([row])
    if RUNS_CSV.exists():
        old = pd.read_csv(RUNS_CSV)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(RUNS_CSV, index=False)
    print(f"\n[runs.csv] appended -> {RUNS_CSV}")


def main() -> None:
    ap = argparse.ArgumentParser(description="M5 experiment runner")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--loss-weight", action="store_true", help="方案 A")
    ap.add_argument("--weight-power", type=float, default=1.0,
                    help="權重指數（1.0=原始；0.5=sqrt 溫和化，m5d 用）")
    ap.add_argument("--balanced-sampler", action="store_true", help="方案 B")
    ap.add_argument("--augment", action="store_true", help="方案 C")
    ap.add_argument("--model", choices=["wafercnn", "resnet"],
                    default="wafercnn", help="M6a：模型架構")
    ap.add_argument("--by-lot", action="store_true",
                    help="M6b：GroupShuffleSplit by lotName（洩漏對照）")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--smoke", action="store_true",
                    help="每類抽 64 張 2 epochs 冒煙")
    args = ap.parse_args()

    flags = [args.loss_weight, args.balanced_sampler, args.augment]
    variant = "+".join(name for name, on in
                       [("weighted", args.loss_weight),
                        ("sampler", args.balanced_sampler),
                        ("augment", args.augment)] if on) or "baseline"
    out_dir = f"artifacts/runs/{args.run_name}"
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        print(f"SMOKE MODE (2 epochs) | run={args.run_name} variant={variant}")

    print(f"[1/6] Loading LSWMD.pkl ...", flush=True)
    df = load_lswmd()
    lf = labeled_frame(df)
    if args.smoke:
        lf = _smoke_sample(lf)
        print(f"      smoke subset: {len(lf):,} labeled rows")

    tr, va, te = build_splits(lf, seed=args.seed)
    if args.by_lot:
        tr, va, te = build_splits_by_lot(lf, seed=args.seed)
        print(f"      [by-lot] GroupShuffleSplit by lotName (lot 不跨 split)")
    y_train = lf["label"].to_numpy()[tr]
    print(f"[2/6] Split: train {len(tr):,} / val {len(va):,} / test {len(te):,}")

    # ── 組裝變因（一次一變因的核心邏輯）──
    criterion = None
    if args.loss_weight:
        # 注意：counts 必須由 label_counts 轉 index（字串比 int 會全 False
        # → weights 全 1 → weighted loss 靜默失效，2026-09-04 m5a 踩過）
        counts = label_counts(y_train)
        w = class_weights(counts, power=args.weight_power)
        criterion = nn.CrossEntropyLoss(weight=w.to("cuda"
                        if torch.cuda.is_available() else "cpu"))
        print(f"      [A] weighted CE (power={args.weight_power})  "
              f"weights={np.round(w.numpy(), 2)}")

    sampler = None
    if args.balanced_sampler:
        sampler = balanced_sampler(y_train, seed=args.seed)
        print("      [B] balanced sampler (1/頻率抽樣)")

    transform = random_d4 if args.augment else None
    if args.augment:
        print("      [C] D4 augmentation (rot90 + flip)")

    print("[3/6] Building DataLoaders ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator().manual_seed(args.seed)
    ds_tr = WM811KDataset(lf, tr, transform=transform)
    train_loader = DataLoader(ds_tr, batch_size=args.batch_size,
                              shuffle=sampler is None, sampler=sampler,
                              generator=None if sampler else gen)
    val_loader = DataLoader(WM811KDataset(lf, va), batch_size=args.batch_size)
    test_loader = DataLoader(WM811KDataset(lf, te), batch_size=args.batch_size)

    print(f"[4/6] Training on {device} ... (variant={variant})")
    model_cls = ResNetWafer if args.model == "resnet" else WaferCNN
    model = model_cls(num_classes=config.NUM_CLASSES)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      {model_cls.__name__} parameters: {n_params:,}")
    history = run_training(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, device=device,
        out_dir=out_dir, patience=args.patience, seed=args.seed,
        criterion=criterion,
    )

    print(f"[5/6] Evaluating best checkpoint on test set ...")
    best = torch.load(Path(out_dir) / "best_model.pt", map_location=device)
    model.load_state_dict(best["state_dict"])
    metrics = evaluate_model(
        model, test_loader, device=device,
        out_png=str(Path(out_dir) / "confusion_matrix.png"),
        title=f"Test confusion matrix - {args.run_name}",
    )

    print(f"[6/6] Logging run to runs.csv ...")
    _append_run({
        "run_name": args.run_name,
        "variant": variant,
        "date": dt.date.today().isoformat(),
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "best_epoch": history["best_epoch"],
        "best_val_macro_f1": history["best_val_macro_f1"],
        "test_macro_f1": round(metrics["macro_f1"], 5),
        "test_accuracy": round(metrics["accuracy"], 5),
        "note": "",
    })
    print(f"\nDone: {args.run_name} | test macro-F1 {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
