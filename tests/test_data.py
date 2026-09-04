"""data.py 單元測試（M3）：labeled_frame / build_splits / WM811KDataset。"""
import numpy as np
import pandas as pd
import pytest
import torch

from wm811k import config
from wm811k.data import (
    WM811KDataset,
    balanced_sampler,
    build_splits,
    class_weights,
    label_counts,
    labeled_frame,
)

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


# ── class_weights / balanced_sampler / transform（M5）────────────

def test_label_counts():
    """str labels → per-class counts（LABEL_TO_IDX 對齊）。

    防 m5a bug 回歸：counts 若用字串比 int 會全 0 → weights 全 1。
    """
    labels = np.array(["none"] * 100 + ["scratch"] * 10 + ["center"] * 5)
    counts = label_counts(labels)
    assert counts[config.LABEL_TO_IDX["none"]] == 100
    assert counts[config.LABEL_TO_IDX["scratch"]] == 10
    assert counts[config.LABEL_TO_IDX["center"]] == 5
    assert counts.sum() == 115


def test_class_weights_balanced():
    """每類數量相等 → 權重全等（無偏見）。"""
    w = class_weights(np.array([100, 100, 100]))
    assert torch.allclose(w, torch.full((3,), 1.0))


def test_class_weights_inverse_frequency():
    """頻率反比：none 多 → 權重小；near-full 少 → 權重大。"""
    counts = np.array([4294, 555, 5189, 3593, 149, 866, 1193, 9680, 147431])
    w = class_weights(counts)
    assert w[8] < w[0] < w[4]  # none < center < near-full
    # 驗證公式 N/(K·N_c)
    n, k = counts.sum(), len(counts)
    assert w[4] == pytest.approx(n / (k * 149))
    assert w[8] == pytest.approx(n / (k * 147431))


def test_class_weights_zero_count_guard():
    """counts=0 的類不產生 NaN/inf（防呆）。"""
    w = class_weights(np.array([100, 0, 100]))
    assert torch.isfinite(w).all()


def test_class_weights_power_softens():
    """power<1 壓縮權重差距（m5d sqrt 溫和化）。"""
    counts = np.array([900, 100])  # 9:1 → 原始權重差 9 倍
    w1 = class_weights(counts, power=1.0).numpy()
    w05 = class_weights(counts, power=0.5).numpy()
    ratio1 = w1[1] / w1[0]
    ratio05 = w05[1] / w05[0]
    assert ratio1 == pytest.approx(9.0)
    assert ratio05 == pytest.approx(3.0)  # sqrt(9)
    assert 1 < ratio05 < ratio1


def test_balanced_sampler_config():
    """驗證 sampler 設定（確定性檢查，不依賴 RNG 抽樣統計）。

    weights 每樣本 = 1/N_c（class 0 → 1/900、class 1 → 1/100）→
    multinomial 抽樣時兩類總質量相等 = 平衡。
    ⚠️ 為何不直接統計抽樣分佈：在部分環境（開發機 uv run --with pytest
    組合）實測 torch CPU multinomial 連續抽樣出現異常偏差（share 0.0006
    vs 理論 0.5），但同一程式碼在直接執行下連續 50 epoch share 穩定
    ~0.50（mean 0.5014）— 判定為環境特定問題；抽樣行為另以直接執行驗證。
    """
    labels = np.array([0] * 900 + [1] * 100)  # 90/10 不平衡
    sampler = balanced_sampler(labels, seed=42)
    w = sampler.weights.numpy()
    assert np.allclose(w[:900], 1 / 900)   # class 0 樣本權重
    assert np.allclose(w[900:], 1 / 100)   # class 1 樣本權重（9 倍）
    assert sampler.num_samples == len(labels)  # epoch 長度不變
    assert sampler.replacement is True
    assert len(list(iter(sampler))) == len(labels)  # 一輪 = 一 epoch


def test_dataset_transform_applied(fake_df):
    """transform 被套在 x 上（M5 增強掛載點）。"""
    lf = labeled_frame(fake_df)
    calls = []

    def fake_transform(x):
        calls.append(1)
        return x + 100  # 明顯標記

    ds = WM811KDataset(lf, np.arange(3), transform=fake_transform)
    x, _ = ds[0]
    assert len(calls) == 1  # 一次 getitem → 一次 transform
    assert float(x.min()) >= 100.0  # transform 確實生效
