"""LSWMD.pkl 讀取相容層（M1 除錯成果，M2+ 共用）。

真實資料 LSWMD.pkl 是 2015 年「Python 2 + pandas 0.16」的雙重古董 pickle，
新版 pandas（3.x）直接讀會依序踩兩個坑：
  1. ModuleNotFoundError: No module named 'pandas.indexes'
     （pandas 0.20+ 把 pandas.indexes.* 改名 pandas.core.indexes.*）
  2. UnicodeDecodeError: 'ascii' codec can't decode byte ...
     （py2 字串是 bytes，py3 預設以 ASCII 解碼 BINSTRING opcode）

本模組提供：
  - register_pandas_legacy_aliases()：讀取前把舊模組名 alias 到新模組
  - read_pickle_robust(path)：pd.read_pickle + 失敗時用 latin1 相容讀取器
  - load_lswmd(path=None)：上述兩者的包裝（預設路徑取自 config）
  - unwrap_scalar(v)：把 py2 時代的巢狀容器（list/ndarray 任意組合）解開成純量
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from wm811k import config


def register_pandas_legacy_aliases() -> bool:
    """把 pandas.indexes.* alias 到 pandas.core.indexes.*（sys.modules 註冊）。

    回傳 True 表示有註冊（代表檔案確實是舊版 pandas 產物）。
    先註冊父套件與子模組，反序列化時 import 直接命中 cache，
    且拿到的是「同一個」新模組物件（class identity 正確）。
    """
    import importlib
    import sys

    if "pandas.indexes" in sys.modules:
        return False  # 環境裡已有（舊版 pandas 或先前已註冊）
    try:
        import pandas.indexes  # noqa: F401
        return False
    except ModuleNotFoundError:
        pass  # 新版 pandas：需要 alias

    parent = importlib.import_module("pandas.core.indexes")
    sys.modules["pandas.indexes"] = parent
    for sub in ("base", "range", "numeric", "multi", "period", "datetimes",
                "timedeltas", "category", "interval", "datetimelike",
                "offsets", "flags"):
        try:
            mod = importlib.import_module(f"pandas.core.indexes.{sub}")
        except ModuleNotFoundError:
            continue
        sys.modules[f"pandas.indexes.{sub}"] = mod
    # 更早的 pandas（0.13–0.15）用 pandas.core.index 路徑，一併 alias
    sys.modules.setdefault("pandas.core.index",
                           importlib.import_module("pandas.core.indexes.base"))
    return True


def read_pickle_robust(path: Path):
    """讀舊 pickle — pd.read_pickle 失敗時改用 latin1 相容讀取器。

    pandas 3.x 已移除 py2 支援，連官方 pickle_compat.Unpickler 也不再強制
    latin1（實測仍用 ascii 解碼）；因此自訂 subclass 繼承它（保留舊 pandas
    內部 module 名稱對應表），但強制 encoding='latin1' — 每個 byte 對應
    一個字元，永不失敗；ASCII 內容解碼結果完全相同。
    """
    from pandas.compat import pickle_compat

    class _CompatLatin1Unpickler(pickle_compat.Unpickler):
        def __init__(self, *args, **kwargs):
            kwargs["encoding"] = "latin1"
            super().__init__(*args, **kwargs)

    try:
        return pd.read_pickle(path)  # 新 pickle 走 C 實作，快（1GB 檔差異大）
    except (UnicodeDecodeError, ModuleNotFoundError, ImportError,
            AttributeError):
        with open(path, "rb") as f:
            return _CompatLatin1Unpickler(f).load()


def load_lswmd(path=None):
    """載入 LSWMD.pkl（自動處理雙重古董 pickle）。path 省略用 config 預設。"""
    p = Path(path).expanduser() if path else config.RAW_DATA_PATH
    register_pandas_legacy_aliases()
    return read_pickle_robust(p)


def unwrap_scalar(v, _depth: int = 0) -> str:
    """把 py2 時代的巢狀容器解到最內層純量，統一成 str（空 → ""）。

    真實資料的 object 欄位每格是巢狀結構（list/ndarray 任意組合），
    例如 failureType = array([['none']]) 或 [['none']] 或 []；
    也可能直接是 bytes / None / NaN。遞迴解開直到純量：
      [['none']] → 'none'；np.array(['none']) → 'none'；
      [] / np.array([]) / None / NaN → ""；b'x' → 'x'
    對無法辨識的型別回傳 <型別名>（寧可顯示也不卡住/報錯）。
    """
    if _depth > 5:
        return f"<{type(v).__name__}>"
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            v = v.item()
        elif v.size == 0:
            return ""
        else:
            v = v.flat[0]
    elif isinstance(v, (list, tuple)):
        if len(v) == 0:
            return ""
        v = v[0]
    if isinstance(v, bytes):
        return v.decode("latin1")
    if isinstance(v, str):
        return str(v)  # 純 str；np.str_ 等子類也一併轉乾淨
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, (int, float, np.integer, np.floating, np.bool_)):
        return str(v)
    return unwrap_scalar(v, _depth + 1)
