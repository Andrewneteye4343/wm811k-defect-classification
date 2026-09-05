"""Grad-CAM 可解釋性（M7）：手寫實作，不依賴 torchcam 套件。

原理（講解用）：
  1. 取「目標層」（最後一卷積段的輸出）特徵圖 A（C, H, W）
  2. 目標類別 c 的 logit y^c 對 A 做 backward → 梯度 G = ∂y^c/∂A
     （G 的每個 channel 告訴你：這個特徵 map 對「判斷成 c」的貢獻方向）
  3. 通道權重 α_k = global-average-pool(G_k)（每個特徵圖一個權重）
  4. 熱圖 = ReLU(Σ_k α_k · A_k)（只保留正向貢獻 — 負貢獻是「反證」不算）
  5. 上採樣回輸入尺寸（32×32）→ 歸一化 0–1 → 疊加在原 wafer 圖上

DSP 類比：A 是 64 個頻道（每 channel 偵測一種 pattern），G 告訴你哪個
頻道對 y^c 是 matched filter → 熱圖 = 加權重組後「模型決策的空間依據」。

為什麼用最後一卷積層：愈深層特徵圖愈「語意化」（淺層是線段/邊緣，
深層是「整圈環」/「整片 cluster」）— 我們想知道的是高層決策依據。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from wm811k.model import ResNetWafer, WaferCNN


def default_target_layer(model) -> torch.nn.Module:
    """每個模型慣用的 Grad-CAM 目標層（最後一卷積段輸出）。

    WaferCNN：model.features（Sequential 最後是 MaxPool → (64,4,4)）
    ResNetWafer：model.layer3（最後殘差段 → (64,4,4)）
    """
    if isinstance(model, ResNetWafer):
        return model.layer3
    if isinstance(model, WaferCNN):
        return model.features
    raise ValueError(f"未定義 target layer 的模型: {type(model).__name__}")


class GradCAM:
    """手寫 Grad-CAM（單樣本）。用法：

        cam = GradCAM(model, target_layer=default_target_layer(model))
        heat = cam.heatmap(x, class_idx=3)   # x: (1,1,32,32) → (32,32) 0~1
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self._acts: dict = {}
        self._grads: dict = {}
        # forward hook：抓目標層輸出特徵圖 A
        self._fwd_h = target_layer.register_forward_hook(self._capture_act)
        # backward hook：抓 A 的梯度（grad_output[0] = ∂loss/∂A）
        self._bwd_h = target_layer.register_full_backward_hook(
            self._capture_grad)

    def _capture_act(self, module, inp, out):
        self._acts["a"] = out

    def _capture_grad(self, module, grad_in, grad_out):
        self._grads["g"] = grad_out[0]

    def heatmap(self, x: torch.Tensor, class_idx: int) -> torch.Tensor:
        """x: (1, C, H, W) 未歸一化輸入 → 熱圖 (H, W)，值域 0~1。"""
        # Grad-CAM 慣例：輸入標記需梯度（避免 full backward hook 語意警告；
        # clone 避免污染 caller 的 tensor）
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
        self.model.zero_grad()
        logits = self.model(x)
        # 目標類別分數反向到特徵圖
        logits[0, class_idx].backward(retain_graph=True)

        A = self._acts["a"][0]              # (C, h, w)
        G = self._grads["g"][0]             # (C, h, w)
        alpha = G.mean(dim=(1, 2), keepdim=True)   # (C, 1, 1)
        cam = (alpha * A).sum(dim=0)        # (h, w)
        cam = F.relu(cam)                   # 只看正向貢獻
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze()
        # 歸一化 0~1（除最大值；全零時保持零）
        m = cam.max()
        # detach：熱圖是輸出資料（要 .numpy() 畫圖），不需梯度
        return (cam / m if m > 0 else cam).detach()

    def __del__(self):
        self._fwd_h.remove()
        self._bwd_h.remove()
