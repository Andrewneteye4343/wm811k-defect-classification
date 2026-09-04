"""data.py 單元測試（M3）：labeled_frame / build_splits / WM811KDataset。"""
import numpy as np
import pandas as pd
import pytest
import torch

from wm811k import config
from wm811k.data import WM811KDataset, build_splits, labeled_frame

from conftest import make_labeled_df, make_wafer


# ── labeled_frame ───────────────────────────────────────────────

def test_labeled_frame_filters_unlabeled_and_normalizes(fake_df):
    lf = labeled_frame(fake_df)
    # 57 列 − 3 無標籤 = 54
    assert len(lf) == 54
    # 9 類都還在，且 label 是乾淨小寫
    assert set(lf["label"]) == set(config.FAILURE_TYPES)
    assert lf["label"].isin(config.FAILURE_TYPES).all()
    # 不污染原始 df
    assert "label" not in fake_df.columns


# ── build_splits ────────────────────────────────────────────────

def _balanced_df(n_per_class=100):
    """每類 100 張 → 900 列，方便比例驗證。"""
    return make_labeled_df(n_per_class=n_per_class, include_unlabeled=False)


def test_splits_proportions():
    lf = labeled_frame(_balanced_df(n_per_class=100))
    n = len(lf)
    tr, va, te = build_splits(lf)
    assert len(tr) / n == pytest.approx(0.70, abs=0.01)
    assert len(va) / n == pytest.approx(0.15, abs=0.01)
    assert len(te) / n == pytest.approx(0.15, abs=0.01)


def test_splits_no_overlap_and_full_cover():
    lf = labeled_frame(_balanced_df(n_per_class=30))
    n = len(lf)
    tr, va, te = build_splits(lf)
    tr_s, va_s, te_s = set(tr), set(va), set(te)
    assert tr_s & va_s == set()
    assert tr_s & te_s == set()
    assert va_s & te_s == set()
    assert len(tr_s | va_s | te_s) == n


def test_splits_stratified_each_class_represented():
    """每個類別在三個 split 都按比例出現（含 minority）。"""
    lf = labeled_frame(_balanced_df(n_per_class=30))
    tr, va, te = build_splits(lf)
    for split in (tr, va, te):
        counts = lf["label"].to_numpy()[split]
        # 每類都至少 1 張
        assert len(set(counts)) == 9


def test_splits_reproducible_same_seed():
    lf = labeled_frame(_balanced_df(n_per_class=20))
    a = build_splits(lf, seed=42)
    b = build_splits(lf, seed=42)
    for x, y in zip(a, b):
        assert (x == y).all()


def test_splits_stratified_imbalanced_minority():
    """極端不平衡（90/10）：minority 在每個 split 都保持 ~10%（stratify 生效）。"""
    rng = np.random.default_rng(0)
    maps = [make_wafer(seed=int(rng.integers(1_000_000))) for _ in range(200)]
    labels = [np.array([["normal"]])] * 180 + [np.array([["rare"]])] * 20
    df = pd.DataFrame({"waferMap": maps, "failureType": labels})
    lf = labeled_frame(df)
    tr, va, te = build_splits(lf)
    y = lf["label"].to_numpy()
    for split, tol in ((tr, 0.03), (va, 0.08), (te, 0.08)):
        share = (y[split] == "rare").mean()
        assert share == pytest.approx(0.10, abs=tol), \
            f"minority share {share:.3f} in split deviates from 10%"


# ── WM811KDataset ───────────────────────────────────────────────

def test_dataset_len_and_getitem_shape(fake_df):
    lf = labeled_frame(fake_df)
    # Dataset 吃任意 indices（不一定要 build_splits 的產出）
    ds = WM811KDataset(lf, np.arange(10))
    assert len(ds) == 10
    x, y = ds[0]
    assert x.shape == (1, config.TARGET_SIZE, config.TARGET_SIZE)
    assert x.dtype == torch.float32
    assert float(x.min()) >= 0.0 and float(x.max()) <= 2.0
    assert 0 <= y < config.NUM_CLASSES


def test_dataset_labels_match_config(fake_df):
    """每個 getitem 的 y 都對應 config.LABEL_TO_IDX（含 none 在第 8 位）。"""
    lf = labeled_frame(fake_df)
    ds = WM811KDataset(lf, np.arange(len(lf)))
    labels = lf["label"].tolist()
    for i in range(len(ds)):
        _, y = ds[i]
        assert y == config.LABEL_TO_IDX[labels[i]]


def test_dataset_custom_size(fake_df):
    lf = labeled_frame(fake_df)
    ds = WM811KDataset(lf, np.arange(5), size=16)
    x, _ = ds[0]
    assert x.shape == (1, 16, 16)
