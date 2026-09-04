"""資料層（M3）：有標籤清洗、stratified 切分、PyTorch Dataset。

M2 定案：自行 stratified 70/15/15（種子 42）— 內建 trianTestLabel 無
validation 且比例失衡（有標籤內 Test 68.6% > Training 31.4%），不使用。
M6 另做 by-lot GroupShuffleSplit 對照組量化 lot 洩漏（本檔不處理）。

切分做在「索引層」而非資料複製：回傳 position indices，
Dataset 吃 (df, indices) → 同一份 df 可同時供 train/val/test 三個
Dataset 使用，零複製、切分與資料載入解耦。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from wm811k import config
from wm811k.loader import unwrap_scalar
from wm811k.preprocess import preprocess_map


def labeled_frame(df: pd.DataFrame) -> pd.DataFrame:
    """取「有標籤」列，新增標準化 label 欄。

    - 真實資料 failureType 空容器 = 無標籤（78.7%）→ 過濾掉
    - label 欄 = unwrap + strip + lower（Edge-Ring → edge-ring，對齊
      config.FAILURE_TYPES 的小寫順序）
    回傳複製的 DataFrame（不污染原始 df）。
    """
    out = df.copy()
    out["label"] = out["failureType"].map(unwrap_scalar).str.strip().str.lower()
    return out[out["label"] != ""].reset_index(drop=True)


def build_splits(df_labeled: pd.DataFrame, seed: int | None = None):
    """自行 stratified 切分（70/15/15，種子固定）。

    參數：df_labeled = labeled_frame() 的輸出（需有 label 欄）。
    回傳：(train_idx, val_idx, test_idx) — 對齊 df_labeled 的 position
    indices（不是原始 DataFrame index；Dataset 用 position 存取）。

    實作：sklearn train_test_split 兩階段 —
      1) 切 train / (val+test)（30%）
      2) 在暫存集內再切 val / test（各半）
    兩次都 stratify=label，且第二階段換 seed 避免兩次 shuffle 相關。
    """
    seed = config.SEED if seed is None else seed
    y = df_labeled["label"].to_numpy()
    n = len(df_labeled)
    idx = np.arange(n)

    tmp_frac = config.VAL_RATIO + config.TEST_RATIO  # 0.30
    train_idx, tmp_idx = train_test_split(
        idx, test_size=tmp_frac, stratify=y, random_state=seed,
    )
    val_frac_of_tmp = config.VAL_RATIO / tmp_frac  # 0.5 → val:test = 1:1
    val_idx, test_idx = train_test_split(
        tmp_idx,
        test_size=1 - val_frac_of_tmp,
        stratify=y[tmp_idx],
        random_state=seed + 1,  # 避免兩次 shuffle 序列相同
    )
    return train_idx, val_idx, test_idx


class WM811KDataset(Dataset):
    """吃清洗後 df + position indices，getitem 即時前處理。

    效能考量：__init__ 把 waferMap / label 欄抽成 list —
    pandas 的 .loc 是 O(n) 標籤查詢，訓練迴圈每 step 呼叫會很慢；
    抽成 list 後 getitem 是 O(1)。waferMap 每格是 ndarray reference，
    抽 list 不複製資料（172,950 張原始圖只在記憶體一份）。

    getitem 回傳 (x, y)：
      x = float32 (1, TARGET_SIZE, TARGET_SIZE)（CV 慣例含 channel dim；
          conv 需要 (B,C,H,W)，batch 由 DataLoader 疊出）
         值域 {0,1,2}（未歸一化 — normalization 是 M4 訓練迴圈/transform 的職責）
      y = int label index（對齊 config.LABEL_TO_IDX）
    """
    def __init__(self, df_labeled: pd.DataFrame, indices,
                 size: int | None = None):
        self.maps = df_labeled["waferMap"].tolist()
        self.labels = df_labeled["label"].tolist()
        self.indices = np.asarray(indices, dtype=np.int64)
        self.size = config.TARGET_SIZE if size is None else size

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        wm = preprocess_map(self.maps[idx], self.size)
        x = torch.from_numpy(wm).float().unsqueeze(0)  # (1, 32, 32)
        y = config.LABEL_TO_IDX[self.labels[idx]]
        return x, y
