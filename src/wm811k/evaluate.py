"""測試集評估（M4）：per-class 指標表 + 混淆矩陣熱圖。

M7 會在此基礎上加 Grad-CAM 與錯誤案例並排圖 — 評估模組只做「誠實報數」：
test 集是訓練期間從未碰過的資料，這裡的分數才是可對外宣稱的數字。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from wm811k import config
from wm811k.metrics import accuracy, confusion_matrix, per_class_prf


@torch.no_grad()
def predict_loader(model, loader, device="cpu"):
    """對整個 loader 推論，回傳 (y_true, y_pred) numpy 陣列。"""
    model.eval()
    model.to(device)
    all_y, all_pred = [], []
    for x, y in loader:
        x = x.to(device)
        all_y.append(y.numpy())
        all_pred.append(model(x).argmax(dim=1).cpu().numpy())
    return np.concatenate(all_y), np.concatenate(all_pred)


def evaluate_model(model, loader, device="cpu", out_png: str | None = None,
                   title: str = "Confusion matrix"):
    """完整評估：印 per-class 表，可選存混淆矩陣圖。回傳指標 dict。"""
    y_true, y_pred = predict_loader(model, loader, device)
    C = confusion_matrix(y_true, y_pred, config.NUM_CLASSES)
    prec, rec, f1 = per_class_prf(C)
    macro = float(f1.mean())
    acc = accuracy(y_true, y_pred)

    print(f"=== Evaluation ({len(y_true):,} samples) ===")
    print(f"accuracy    : {acc:.4f}   (僅對照 — none 主導，非主指標)")
    print(f"macro-F1    : {macro:.4f}   (主指標：9 類 F1 平均)")
    print(f"\n{'class':<11}{'n_true':>8}{'precision':>11}{'recall':>9}{'F1':>8}")
    for k, name in config.IDX_TO_LABEL.items():
        n_true = int(C[k].sum())
        print(f"{name:<11}{n_true:>8}{prec[k]:>11.4f}{rec[k]:>9.4f}{f1[k]:>8.4f}")

    if out_png:
        _plot_confusion(C, Path(out_png), title)
    return {
        "accuracy": acc,
        "macro_f1": macro,
        "per_class_f1": {config.IDX_TO_LABEL[k]: float(f1[k])
                         for k in range(config.NUM_CLASSES)},
    }


def _plot_confusion(C: np.ndarray, path: Path, title: str) -> None:
    """混淆矩陣熱圖（實際值 × 預測值，含數字標註）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = config.FAILURE_TYPES
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(C, cmap="Blues")
    ax.set_xticks(range(config.NUM_CLASSES), labels, rotation=45, ha="right")
    ax.set_yticks(range(config.NUM_CLASSES), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    # 數字標註（none 數量級太大 → 用科學記號或縮放色）
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            v = C[i, j]
            if v > 0:
                ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=7,
                        color="white" if v > C.max() * 0.5 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
