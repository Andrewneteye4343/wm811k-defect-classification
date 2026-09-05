"""pytest 共用 fixtures：合成 wafer map / DataFrame 製造器。

合成資料刻意模擬真實 LSWMD.pkl 的結構：
  - waferMap = 2D uint8 ndarray（方形陣列內切圓形 die 區；1=pass、2=fail、0=背景）
  - failureType = 2D numpy 巢狀容器（array([['Center']]) 等，py2 存法）
真實資料太大（2.1GB pkl）無法在 CI/開發機載入，所有管線測試都用合成資料。
"""
import numpy as np
import pandas as pd
import pytest

from wm811k import config


def make_wafer(rows=None, cols=None, radius=None, n_fail=2, seed=0):
    """模擬真實 wafer map：方形陣列內切圓形 die 區（1=pass、2=fail、0=背景）。"""
    rng = np.random.default_rng(seed)
    rows = int(rows) if rows else int(rng.integers(28, 52))
    cols = int(cols) if cols else int(rng.integers(28, 52))
    cy, cx = rows // 2, cols // 2
    radius = int(radius) if radius else min(rows, cols) // 2 - 1
    yy, xx = np.mgrid[:rows, :cols]
    wm = np.where((yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2, 1, 0)
    wm = wm.astype(np.uint8)
    if n_fail:
        die_y, die_x = np.where(wm == 1)
        pick = rng.choice(len(die_y), size=min(n_fail, len(die_y)), replace=False)
        wm[die_y[pick], die_x[pick]] = 2  # 少量 fail die
    return wm


def make_labeled_df(n_per_class=6, include_unlabeled=True, seed=42):
    """合成有標籤 DataFrame：9 類各 n_per_class 張 +（可選）3 張無標籤。

    lotName：每 3 張同 lot（模擬真實：lot 內 wafer 相似；M6b by-lot 測試用）。
    """
    rng = np.random.default_rng(seed)
    maps, labels, lots = [], [], []
    idx = 0
    for cls in config.FAILURE_TYPES:
        for _ in range(n_per_class):
            maps.append(make_wafer(seed=int(rng.integers(1_000_000))))
            labels.append(np.array([[cls]]))  # 模擬 2D 巢狀容器
            lots.append(np.array([[f"L{idx // 3}"]]))  # 巢狀 lotName
            idx += 1
    if include_unlabeled:
        for _ in range(3):
            maps.append(make_wafer(seed=int(rng.integers(1_000_000))))
            labels.append(np.array([[]], dtype=object))  # 空容器 = 無標籤
            lots.append(np.array([[f"U{idx // 3}"]]))
            idx += 1
    return pd.DataFrame(
        {"waferMap": maps, "failureType": labels, "lotName": lots})


@pytest.fixture
def fake_df():
    """9 類各 6 張 + 3 無標籤 = 57 列。"""
    return make_labeled_df(n_per_class=6)
