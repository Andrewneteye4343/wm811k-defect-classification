"""D4 對稱增強（M5）：wafer map 的 domain-informed 資料擴增。

D4 = 正方形的對稱群：4 個旋轉（0°/90°/180°/270°）× 2 個鏡像 = 8 個變體。

為什麼只做 D4、不做任意角度旋轉：
  wafer map 是 die 的**離散格點** — 任意角度旋轉需要插值，等於「發明」
  不存在的 die 位置（類別圖 0/1/2 不能平滑插值，見 preprocess 的 NEAREST
  理由）。D4 是格點保真的完整對稱群 — 旋轉/鏡像後每個像素仍落在格點上。

為什麼是 domain-informed（不是亂試增強）：
  製程缺陷（center/donut/edge-ring/scratch…）在晶圓上可出現在任何旋轉
  方位 — 旋轉 90° 的 wafer 仍是同一種缺陷（真實世界的等價樣本），
  不是「假資料」。這對 scratch（M4 最弱，方向多變）特別有幫助。

實作：torch 張量操作（rot90/flip = 純排列、無插值），transform 掛在
Dataset 層 → 每個 epoch 每張圖抽不同變體 = 免費的資料多樣性。
"""
from __future__ import annotations

import torch


def random_d4(x: torch.Tensor) -> torch.Tensor:
    """隨機套用一個 D4 變體。x: (1, H, W) 或 (H, W) tensor。

    每次呼叫獨立隨機 → 同一張圖在不同 epoch 以不同方位出現。
    值域不變（純排列，無插值）→ 仍是 {0,1,2} 離散類別圖。
    """
    k = int(torch.randint(0, 4, (1,)).item())
    x = torch.rot90(x, k, dims=(-2, -1))
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=(-1,))  # 水平鏡像
    return x


def d4_variants(x: torch.Tensor) -> list:
    """回傳全部 8 個 D4 變體（測試/視覺化用；訓練用 random_d4）。"""
    out = []
    for k in range(4):
        rot = torch.rot90(x, k, dims=(-2, -1))
        out.append(rot)
        out.append(torch.flip(rot, dims=(-1,)))
    return out
