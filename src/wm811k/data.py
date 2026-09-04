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


def label_counts(labels, num_classes: int | None = None) -> np.ndarray:
    """標準化 label 字串陣列 → per-class counts（對齊 config.LABEL_TO_IDX）。

    ⚠️ 防呆來源（2026-09-04 m5a 實測 bug）：counts 必須用 label **index**
    比對 — 若拿字串 label 直接比 int（`y_train == 3`）會全 False →
    counts 全 0 → class_weights 防呆給全 1 → weighted loss 靜默失效
    （log 顯示 weights=[1,1,...] 但無人察覺，等同跑 baseline）。
    """
    idx = np.array([config.LABEL_TO_IDX[s] for s in labels], dtype=np.int64)
    num_classes = config.NUM_CLASSES if num_classes is None else num_classes
    return np.array([(idx == c).sum() for c in range(num_classes)],
                    dtype=np.int64)


def class_weights(counts: np.ndarray, power: float = 1.0) -> torch.Tensor:
    """頻率反比類別權重：w_c = (N / (K * N_c)) ** power。

    - N = 總樣本數、K = 類別數、N_c = 類別 c 的樣本數
    - none（85.2%）權重 ~0.13、near-full（0.1%）權重 ~129 → CrossEntropy
      對 minority 的錯誤懲罰大幅提高（M5 方案 A）
    - power < 1.0 溫和化（m5d 用 0.5 = sqrt）：m5a_fixed 實測全權重
      （差距 1000 倍）讓模型過度補償 → minority precision 崩（scratch
      precision 0.08、macro 0.7306）；sqrt 把差距壓到 ~31 倍是常見緩解
    - 防呆：counts=0 的類給權重 1（stratify 保證 train 每類都有，正常不會發生）
    """
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.sum()
    k = len(counts)
    # np.where 的三參數都會先被求值 → 0 除會噴 RuntimeWarning，
    # 用 errstate 靜音（counts>0 的分支才有效，0 的分支最後選 1.0）
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(counts > 0, (n / (k * counts)) ** power, 1.0)
    return torch.tensor(w, dtype=torch.float32)


def balanced_sampler(labels, seed: int | None = None):
    """WeightedRandomSampler：抽樣權重 = 1/頻率 → 每類被抽中機率相等。

    機制：每樣本權重 = 1/N_c（所屬類計數的反比），replacement=True 抽樣。
    效果：minority 每 epoch 被看到多次、none 被看到變少（epoch 內近似
    平衡）→ 梯度不再被 none 主導（M5 方案 B）。與方案 A 的差別：
    A 改 loss 代價、B 改資料餵食分佈。
    num_samples = len(labels)：epoch 總步數不變。
    """
    from torch.utils.data import WeightedRandomSampler

    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    inv = {c: 1.0 / cnt for c, cnt in zip(classes, counts)}
    weights = np.array([inv[c] for c in labels], dtype=np.float64)
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    return WeightedRandomSampler(
        weights, num_samples=len(labels), replacement=True, generator=gen)


class WM811KDataset(Dataset):
    """吃清洗後 df + position indices，getitem 即時前處理。

    效能考量：__init__ 把 waferMap / label 欄抽成 list —
    pandas 的 .loc 是 O(n) 標籤查詢，訓練迴圈每 step 呼叫會很慢；
    抽成 list 後 getitem 是 O(1)。waferMap 每格是 ndarray reference，
    抽 list 不複製資料（172,950 張原始圖只在記憶體一份）。

    transform：可選的 augmentation callable（M5 用 augment.random_d4），
    套在回傳的 x 上 → 每個 epoch 每張圖抽不同變體（增強只作用於
    訓練資料 — val/test 的 Dataset 不傳 transform）。

    getitem 回傳 (x, y)：
      x = float32 (1, TARGET_SIZE, TARGET_SIZE)（CV 慣例含 channel dim；
          conv 需要 (B,C,H,W)，batch 由 DataLoader 疊出）
         值域 {0,1,2}（未歸一化 — normalization 是 M4 訓練迴圈/transform 的職責）
      y = int label index（對齊 config.LABEL_TO_IDX）
    """
    def __init__(self, df_labeled: pd.DataFrame, indices,
                 size: int | None = None, transform=None):
        self.maps = df_labeled["waferMap"].tolist()
        self.labels = df_labeled["label"].tolist()
        self.indices = np.asarray(indices, dtype=np.int64)
        self.size = config.TARGET_SIZE if size is None else size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        wm = preprocess_map(self.maps[idx], self.size)
        x = torch.from_numpy(wm).float().unsqueeze(0)  # (1, 32, 32)
        if self.transform is not None:
            x = self.transform(x)
        y = config.LABEL_TO_IDX[self.labels[idx]]
        return x, y
