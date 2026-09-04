"""分類指標（M4）：從 confusion matrix 手寫 per-class PRF 與 macro-F1。

為什麼手寫而不是 sklearn.metrics：這是學習專案 — 你必須懂 macro-F1 是
「9 個類別 F1 的算術平均」而不是什麼黑箱數字，之後讀論文、面試被問
「macro 和 weighted 差在哪」才能講清楚。實作與 sklearn 對拍驗證（tests）。

定義（以類別 k 為例）：
  - precision_k = TP_k / (TP_k + FP_k) = 對角[k] / 預測為 k 的總數
                 （「說是 k 的裡面，有多少真的是 k」— 寧缺勿濫？）
  - recall_k    = TP_k / (TP_k + FN_k) = 對角[k] / 實際為 k 的總數
                 （「真的是 k 的裡面，抓到多少」— 寧可錯殺？）
  - F1_k        = 2 * P * R / (P + R)  （兩者的調和平均）
  - macro-F1    = mean(F1_0 ... F1_8)  （每類平等 — 不受 none 85% 影響）
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    """C[k, j] = 實際 k 被預測成 j 的數量。輸入為 label index 陣列。"""
    C = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        C[t, p] += 1
    return C


def per_class_prf(C: np.ndarray):
    """從 confusion matrix 算 per-class precision / recall / F1。

    回傳三個 (num_classes,) float 陣列。zero-division 處理：
    - 某類完全沒被預測（pred_sum=0）→ precision 0
    - 某類完全沒出現（true_sum=0）→ recall 0
    - 兩者皆 0 → F1 0（與 sklearn zero_division=0 一致）
    """
    C = np.asarray(C, dtype=np.float64)
    tp = np.diag(C)
    pred_sum = C.sum(axis=0)  # 每類被預測的總數（column）
    true_sum = C.sum(axis=1)  # 每類實際總數（row）

    prec = tp / (pred_sum + EPS)
    rec = tp / (true_sum + EPS)
    # 當 prec 與 rec 都趨近 0（類別不存在於資料），F1 應為 0 而非 NaN
    f1 = 2 * prec * rec / (prec + rec + EPS)
    # EPS 修正：tp=0 且 sum=0 的類 → prec=0、rec=0 → f1 = 0/(0+EPS)=0 ✓
    # 但 tp=0 且 sum>0 → prec=0、rec=0 → 0 ✓；tp>0 → 正常值 ✓
    return prec, rec, f1


def macro_f1(y_true, y_pred, num_classes: int) -> float:
    """9 類 F1 的算術平均（主指標 — 每類平等，不被 none 主導）。"""
    C = confusion_matrix(y_true, y_pred, num_classes)
    _, _, f1 = per_class_prf(C)
    return float(f1.mean())


def accuracy(y_true, y_pred) -> float:
    """僅供對照（刻意不用它當主指標 — none 85% 會讓 accuracy 自欺）。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())
