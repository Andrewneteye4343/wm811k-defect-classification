"""WaferCNN：小型 CNN（M4 基線，刻意全手寫以便逐層講解）。

架構（輸入 32×32 單通道）：
  Conv1 1→16 ch, 3×3, pad1 → ReLU → MaxPool2  → (16, 16, 16)
  Conv2 16→32 ch, 3×3, pad1 → ReLU → MaxPool2  → (32, 8, 8)
  Conv3 32→64 ch, 3×3, pad1 → ReLU → MaxPool2  → (64, 4, 4)
  Flatten (1024) → Linear 128 → ReLU → Dropout 0.3 → Linear 9

為什麼這樣設計（教學重點）：
  1. 特徵金字塔：淺層 conv 學「線段/邊緣」（8×8 解析度還看得到單顆 die），
     深層 conv 學「形狀零件」（4×4 只剩 macro pattern）→ 分類用形狀不是像素
  2. 每層 MaxPool 把解析度砍半（32→16→8→4）：權重共享 + 降維 =
     平移容忍（pattern 稍微位移仍觸發同一特徵）+ 視野加倍（深層每顆
     neuron 看到原圖更大區域 → 才能看到 edge-ring 的「環」）
  3. 通道數遞增（16→32→64）補償解析度下降：資訊不流失，只換表示法
  4. Dropout(0.3)：隨機關閉 neuron → 防止「背答案」（overfitting），
     基線就有（資料類別極不平衡，先求穩）

參數量約 156k（<1M 目標）：conv 都是 3×3 小濾波器，絕大多數參數在
最後的 Linear（1024→128 = 131k）。這也解釋為什麼 CNN 對小圖有效率：
卷積層參數與圖大小無關，全連接層才是參數大戶。
"""
from __future__ import annotations

import torch
from torch import nn

from wm811k import config


class WaferCNN(nn.Module):
    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 32, 32) → logits (B, num_classes)。"""
        return self.classifier(self.features(x))
