"""前處理管線（M3）：wafer map → 固定尺寸張量。

M2 定案策略（見 experiments/notes.md）：
  1. 先裁「含 die（非零）區域的最大內接方」：wafer 是方形陣列內切的圓形，
     陣列四角與長邊多餘皆為背景 0 → 裁掉零資訊損失。
  2. 再等比縮放到 TARGET_SIZE（32×32）。resize 用 NEAREST：
     waferMap 是離散類別（0=背景 / 1=pass / 2=fail），nearest 不會創造
     bilinear 會產生的「半 pass」中間值。

設計原則：**純函式** — 不碰 I/O、不碰 PyTorch、不依賴 config 以外的狀態，
因此可獨立單測（M3 的 pytest 範圍）、之後換模型/實驗都不會動到這裡。
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from wm811k import config
from wm811k.loader import unwrap_scalar


def crop_to_square(wm: np.ndarray) -> np.ndarray:
    """裁出「含 die（非零像素）區域」的最大內接方。

    做法：
      1. 找出非零（die）列/行的 bbox（trim 掉全背景邊）— 比直接裁陣列中央
         穩健：即使晶圓圓心不在陣列正中央（off-center lot）也不會裁歪。
      2. 以 bbox 中心為準取 min(高, 寬) 見方。對正常資料（die 區是圓形，
         bbox 接近方形）這等同於「wafer 圓的 bounding square」— die 全保留。

    防呆：全背景（理論上不會發生）或退化尺寸時原樣回傳，絕不崩潰。
    """
    h, w = wm.shape
    rows = np.any(wm != 0, axis=1)
    cols = np.any(wm != 0, axis=0)
    if not rows.any():  # 全背景
        return wm

    r0 = int(np.argmax(rows))
    r1 = int(h - np.argmax(rows[::-1]))
    c0 = int(np.argmax(cols))
    c1 = int(w - np.argmax(cols[::-1]))

    side = min(r1 - r0, c1 - c0)
    if side <= 0:
        return wm

    # 以 bbox 中心為準，clamp 到陣列範圍內
    cr, cc = (r0 + r1) // 2, (c0 + c1) // 2
    r_start = max(0, min(cr - side // 2, h - side))
    c_start = max(0, min(cc - side // 2, w - side))
    return wm[r_start:r_start + side, c_start:c_start + side]


def resize_map(wm: np.ndarray, size: int | None = None) -> np.ndarray:
    """等比縮放到 (size, size)。size 省略用 config.TARGET_SIZE。

    用 PIL NEAREST：輸入為 uint8 離散值 {0,1,2}，nearest 保證輸出值域
    仍是 {0,1,2}（不產生中間值）— 這是「類別圖」而非「自然影像」，
    不能用 bilinear/lanczos 平滑。
    """
    size = config.TARGET_SIZE if size is None else size
    img = Image.fromarray(np.asarray(wm, dtype=np.uint8))
    img = img.resize((size, size), Image.NEAREST)
    return np.array(img, dtype=np.uint8)  # np.array 強制 copy → writable（torch.from_numpy 需要）


def preprocess_map(wm: np.ndarray, size: int | None = None) -> np.ndarray:
    """完整前處理：裁內接方 → 等比縮放。回傳 uint8 (size, size) 陣列。"""
    return resize_map(crop_to_square(np.asarray(wm)), size)


def normalize_label(raw) -> str:
    """把原始標籤（巢狀容器 + Title Case）標準化成 config 小寫形式。

    真實資料 failureType 每格是巢狀容器（例 array([['Edge-Ring']])），
    且類別為 Title Case（Edge-Ring/Center…）、none 為小寫 →
    unwrap_scalar 解巢後 lower() 統一（空標籤 → ""）。
    """
    return unwrap_scalar(raw).strip().lower()
