"""metrics.py 單元測試：手寫 PRF/macro-F1 與 sklearn 對拍 + 教學案例。"""
import numpy as np
import pytest
from sklearn.metrics import (
    confusion_matrix as sk_confusion,
    f1_score,
    precision_recall_fscore_support,
)

from wm811k import config
from wm811k.metrics import accuracy, confusion_matrix, macro_f1, per_class_prf


def test_confusion_matches_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 9, 500)
    y_pred = rng.integers(0, 9, 500)
    assert (confusion_matrix(y_true, y_pred, 9)
            == sk_confusion(y_true, y_pred, labels=range(9))).all()


def test_prf_match_sklearn():
    """手寫 PRF 與 sklearn（zero_division=0）一致 — 含未出現類別。"""
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 9, 1000)
    y_pred = rng.integers(0, 9, 1000)
    C = confusion_matrix(y_true, y_pred, 9)
    prec, rec, f1 = per_class_prf(C)
    sk_p, sk_r, sk_f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(9), zero_division=0)
    np.testing.assert_allclose(prec, sk_p, atol=1e-10)
    np.testing.assert_allclose(rec, sk_r, atol=1e-10)
    np.testing.assert_allclose(f1, sk_f, atol=1e-10)
    assert macro_f1(y_true, y_pred, 9) == pytest.approx(
        f1_score(y_true, y_pred, average="macro", zero_division=0))


def test_hand_computed_small_case():
    """2 類手算案例：C = [[2, 1], [1, 6]]
    class 0: P=2/3, R=2/3, F1=2/3；class 1: P=6/7, R=6/7, F1=6/7
    macro-F1 = (2/3 + 6/7)/2
    """
    C = np.array([[2, 1], [1, 6]])
    prec, rec, f1 = per_class_prf(C)
    assert prec[0] == pytest.approx(2 / 3)
    assert rec[0] == pytest.approx(2 / 3)
    assert f1[1] == pytest.approx(6 / 7)
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1])
    assert macro_f1(y_true, y_pred, 2) == pytest.approx((2 / 3 + 6 / 7) / 2)


def test_naive_all_none_trap():
    """教學案例：全猜 none（真實 85.2% 分佈）→ accuracy 0.85 但 macro-F1 ≈ 0.10。

    證明為什麼 macro-F1 是主指標：none 的 F1≈0.92 只貢獻 1/9，
    其他 8 類全錯（P=0 → F1=0）→ macro ≈ 0.92/9 ≈ 0.10。
    """
    rng = np.random.default_rng(42)
    n = 10000
    # 真實類別比例（M3 驗證數字）
    counts = np.array([4294, 555, 5189, 3593, 149, 866, 1193, 9680, 147431])
    probs = counts / counts.sum()
    y_true = rng.choice(config.NUM_CLASSES, size=n, p=probs)
    y_pred = np.full(n, config.LABEL_TO_IDX["none"])  # 全猜 none
    acc = accuracy(y_true, y_pred)
    mf1 = macro_f1(y_true, y_pred, config.NUM_CLASSES)
    assert acc > 0.80                        # accuracy 自欺：>80%
    assert mf1 < 0.20                        # macro-F1 誠實：<0.2
    # none 的 F1 = 2P/(1+P)，P=0.852 → F1=0.920 → macro = F1/9 ≈ 0.102
    p_none = probs[config.LABEL_TO_IDX["none"]]
    f1_none = 2 * p_none / (1 + p_none)
    assert mf1 == pytest.approx(f1_none / 9, abs=0.005)


def test_empty_class_handling():
    """某類完全沒出現 → 不 NaN；macro-F1 平均「全部 9 類」（不存在類 F1=0）。

    注意 sklearn 預設 labels=None 只平均有出現的類（0.8）— 語意不同！
    我們的 macro-F1 是固定 9 類平均（與 test 集 9 類齊全時一致），
    因此對拍時要明確傳 labels=range(9)。
    """
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1])
    mf1 = macro_f1(y_true, y_pred, num_classes=9)  # 類 2..8 不存在
    assert np.isfinite(mf1)
    assert mf1 == pytest.approx(0.8 * 2 / 9, abs=1e-10)  # 兩類各 F1=0.8
    assert mf1 == pytest.approx(f1_score(
        y_true, y_pred, labels=list(range(9)),
        average="macro", zero_division=0))
