"""preprocess.py 單元測試（M3）：crop / resize / normalize_label。"""
import numpy as np
import pytest

from wm811k import config
from wm811k.preprocess import (
    crop_to_square,
    normalize_label,
    preprocess_map,
    resize_map,
)

from conftest import make_wafer


# ── crop_to_square ──────────────────────────────────────────────

def test_crop_square_input_unchanged():
    """die 區已是方形：crop 後不變。"""
    wm = np.ones((20, 20), dtype=np.uint8)
    out = crop_to_square(wm)
    assert out.shape == (20, 20)
    assert (out == wm).all()


def test_crop_circle_wafer_preserves_all_dies():
    """圓形 wafer（陣列矩形）：crop 後 die（非零）全保留 — 零資訊損失。"""
    wm = make_wafer(rows=48, cols=40, radius=16, n_fail=5, seed=1)
    out = crop_to_square(wm)
    assert out.shape[0] == out.shape[1]
    assert (out != 0).sum() == (wm != 0).sum()  # die 一顆都沒少


def test_crop_offcenter_die_region():
    """die 區不在陣列中央（off-center）：以 die bbox 為準裁，不裁歪。"""
    wm = np.zeros((40, 50), dtype=np.uint8)
    wm[5:15, 30:45] = 1  # bbox 高 10 × 寬 15，偏離中心
    out = crop_to_square(wm)
    assert out.shape == (10, 10)  # side = min(10, 15)
    assert (out != 0).sum() == 100  # 裁出的 10×10 全在 die 區內


def test_crop_all_background_no_crash():
    """全背景（理論上不發生）：防呆回傳原樣，不崩潰。"""
    wm = np.zeros((15, 20), dtype=np.uint8)
    out = crop_to_square(wm)
    assert out.shape == (15, 20)


def test_crop_realistic_rect_array():
    """真實常見尺寸 32×29（矩形陣列）：裁出 29×29（wafer 圓 bounding 方）。"""
    wm = make_wafer(rows=32, cols=29, radius=13, n_fail=3, seed=3)
    out = crop_to_square(wm)
    assert out.shape[0] == out.shape[1]
    assert (out != 0).sum() == (wm != 0).sum()  # die 全保留


# ── resize_map / preprocess_map ─────────────────────────────────

def test_preprocess_map_default_shape_and_dtype():
    wm = make_wafer(rows=33, cols=29, n_fail=4, seed=2)
    out = preprocess_map(wm)
    assert out.shape == (config.TARGET_SIZE, config.TARGET_SIZE)
    assert out.dtype == np.uint8


def test_preprocess_map_custom_size():
    wm = make_wafer(n_fail=2, seed=4)
    out = preprocess_map(wm, size=16)
    assert out.shape == (16, 16)


def test_resize_values_stay_discrete():
    """NEAREST 縮放：輸出值域仍是 {0,1,2}（不產生 bilinear 的中間值）。"""
    wm = make_wafer(rows=50, cols=50, n_fail=8, seed=5)
    out = preprocess_map(wm)
    assert set(np.unique(out)) <= {0, 1, 2}


def test_resize_map_upscale():
    """小圖放大到 32：shape 正確且值域保留。"""
    wm = make_wafer(rows=12, cols=12, n_fail=1, seed=6)
    out = resize_map(wm, size=32)
    assert out.shape == (32, 32)
    assert set(np.unique(out)) <= {0, 1, 2}


# ── normalize_label ─────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("Edge-Ring", "edge-ring"),          # Title Case → lower
    ("none", "none"),                    # none 本來就小寫
    ("  Donut  ", "donut"),              # 空白清理
    (np.array([["Center"]]), "center"),  # 2D 巢狀容器（真實結構）
    (np.array([[["Loc"]]]), "loc"),      # 更深巢狀
    ([["Scratch"]], "scratch"),          # list 巢狀
    ("", ""),                            # 空字串 → ""
    (np.array([], dtype=float), ""),     # 空容器 = 無標籤 → ""
])
def test_normalize_label(raw, expected):
    assert normalize_label(raw) == expected
