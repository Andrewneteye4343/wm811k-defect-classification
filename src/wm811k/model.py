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


# ─────────────────────────────────────────────────────────────────────
# M6：手寫殘差 CNN（ResNet 風格，學習 skip connection）
# ─────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """殘差塊（ResNet 的 BasicBlock 精神）：conv3x3-BN-ReLU-conv3x3-BN + skip。

    output = F(x) + shortcut(x)，其中 F = 兩個 3×3 conv（學習「殘差」）。

    為什麼有效（梯度高速公路）：若 F 學不到東西，網路可讓 x 原封不動
    通過（F→0 時 output≈x）→ 深層永遠不劣於淺層；反向傳播時梯度經
    shortcut 直接流回淺層，解決深網梯度消失。ResNet 靠它堆到 152 層。

    downsample（in_ch != out_ch 或 stride≠1）：shortcut 需升維才能相加 —
    用 1×1 conv（只改 channel 數不學空間特徵，是「投影」不是「濾波」）。
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)   # skip connection（核心一行）
        return self.relu(out)


class ResNetWafer(nn.Module):
    """手寫殘差 CNN（M6，32×32 單通道輸入）。

    結構：
      stem: conv 1→16, 3×3 → BN → ReLU            (16, 32, 32)
      Stage1: ResBlock(16,16) + ResBlock(16,16) → MaxPool   (16,16,16)
      Stage2: ResBlock(16,32) + ResBlock(32,32) → MaxPool   (32,8,8)
      Stage3: ResBlock(32,64) + ResBlock(64,64) → MaxPool   (64,4,4)
      GlobalAvgPool → FC(64 → 9)

    與 WaferCNN 的差別：① 殘差連接（深層可訓練）② GAP 取代 Flatten+FC
    （GAP 把每個 channel 平均成 1 值 → 參數暴減且天然抗過擬合 —
    Network in Network 的洞見：channel = 特徵種類）③ BN 加速收斂兼正則化。
    """
    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(16, 16), ResidualBlock(16, 16), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            ResidualBlock(16, 32), ResidualBlock(32, 32), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            ResidualBlock(32, 64), ResidualBlock(64, 64), nn.MaxPool2d(2))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x).flatten(1)   # (B, 64)
        return self.fc(x)
