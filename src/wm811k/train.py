"""手寫訓練迴圈（M4）：train_one_epoch / validate / run_training。

設計原則（教學重點）：
  1. 不靠 PyTorch Lightning / Trainer — 手寫整個迴圈才能看懂
     「optimizer.zero_grad → loss.backward → optimizer.step」三行
     在幹嘛（清梯度 → 反向傳播算梯度 → 沿梯度更新權重）。
  2. 函式化：train/validate 分離、可單測、可重用（M5/M6 換 loss、
     sampler、增強都不會動到骨架）。
  3. checkpoint 只存「best val macro-F1」的 state_dict（不存整個
     model 物件 → 跨環境載入安全）。
  4. early stopping：val macro-F1 連續 patience 個 epoch 沒破紀錄就
     停 — 省時間也避免過度擬合 val（val 不是拿來「追」的）。

每 epoch 記錄：train_loss（有 dropout → 略高正常）、val_loss、
val_macro_f1。**不報 train/val accuracy**（none 85% 陷阱）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from wm811k import config
from wm811k.metrics import macro_f1


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    """跑一個訓練 epoch，回傳平均 loss。model 會更新權重。"""
    model.train()  # 訓練模式：dropout 啟用、BN 用 batch 統計
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()      # 1. 清掉上一步的梯度（PyTorch 會累積！）
        out = model(x)             #    forward：logits (B, 9)
        loss = criterion(out, y)   #    CrossEntropy 內部已含 softmax
        loss.backward()            # 2. 反向傳播：每參數的 ∂loss/∂w
        optimizer.step()           # 3. w -= lr * grad（Adam 自適應版本）
        total_loss += loss.item() * len(x)
        n += len(x)
    return total_loss / n


@torch.no_grad()  # 驗證不追蹤梯度：省記憶體、省計算
def validate(model, loader, criterion, device, num_classes: int | None = None):
    """驗證一個 epoch，回傳 (平均 loss, macro-F1)。不改權重。"""
    num_classes = config.NUM_CLASSES if num_classes is None else num_classes
    model.eval()  # 評估模式：dropout 關閉（推論用全部 neuron）
    total_loss, n = 0.0, 0
    all_y, all_pred = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += criterion(out, y).item() * len(x)
        n += len(x)
        all_y.append(y.cpu().numpy())
        all_pred.append(out.argmax(dim=1).cpu().numpy())
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    f1 = macro_f1(y_true, y_pred, num_classes=num_classes)
    return total_loss / n, f1


def _save_checkpoint(model, out_dir: Path, epoch: int, val_f1: float) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "best_model.pt"
    torch.save({
        "epoch": epoch,
        "val_macro_f1": val_f1,
        "state_dict": model.state_dict(),
        "config": {
            "model": type(model).__name__,
            "num_classes": config.NUM_CLASSES,
            "input_size": config.TARGET_SIZE,
        },
    }, path)
    return path


def run_training(model, train_loader, val_loader, *, epochs: int = 30,
                 lr: float = 1e-3, device="cpu", out_dir="artifacts/m4_baseline",
                 patience: int = 8, seed: int = 42) -> dict:
    """完整訓練流程，回傳 history（dict of lists）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device)
    model = model.to(device)  # 關鍵：模型搬到 device（input 已 .to(device)，
    # 兩者不一致會 RuntimeError — CPU 環境測不出這個 bug，GPU 才現形）

    criterion = nn.CrossEntropyLoss()  # M4 基線：無 class weight（M5 處理）
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    out_dir = Path(out_dir)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_macro_f1": []}
    best_f1, best_epoch, bad_epochs = -1.0, -1, 0

    print(f"Training {type(model).__name__} | epochs={epochs} lr={lr} "
          f"device={device} | patience={patience}")
    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_f1 = validate(model, val_loader, criterion, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(round(tr_loss, 5))
        history["val_loss"].append(round(va_loss, 5))
        history["val_macro_f1"].append(round(va_f1, 5))
        print(f"  epoch {epoch:>2}/{epochs} | train_loss {tr_loss:.4f} | "
              f"val_loss {va_loss:.4f} | val_macro_F1 {va_f1:.4f}"
              + ("  ★ best" if va_f1 > best_f1 else ""))

        if va_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = va_f1, epoch, 0
            ckpt = _save_checkpoint(model, out_dir, epoch, va_f1)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  Early stop at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    # 歷史存檔 + 學習曲線
    history["best_epoch"] = best_epoch
    history["best_val_macro_f1"] = round(best_f1, 5)
    history["checkpoint"] = str(ckpt)
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    _plot_history(history, out_dir / "learning_curve.png")
    print(f"Best val macro-F1 {best_f1:.4f} at epoch {best_epoch} "
          f"(checkpoint: {ckpt})")
    return history


def _plot_history(history: dict, path: Path) -> None:
    """學習曲線：loss（左）與 val macro-F1（右）。圖內文字用英文。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ep = history["epoch"]
    ax1.plot(ep, history["train_loss"], label="train loss")
    ax1.plot(ep, history["val_loss"], label="val loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("Training curve")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(ep, history["val_macro_f1"], label="val macro-F1", color="green")
    ax2.axhline(max(history["val_macro_f1"]), ls="--", color="gray", alpha=0.6)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("macro-F1")
    ax2.set_title("Validation macro-F1")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
