"""interpret.py 單元測試：Grad-CAM 基本性質（合成輸入，無需真實資料）。"""
import torch

from wm811k.interpret import GradCAM, default_target_layer
from wm811k.model import ResNetWafer, WaferCNN


def _gradcam_for(model_cls):
    model = model_cls()
    cam = GradCAM(model, default_target_layer(model))
    return model, cam


def test_heatmap_shape_and_range_wafercnn():
    """熱圖 shape = 輸入尺寸、值域 [0,1]。"""
    model, cam = _gradcam_for(WaferCNN)
    x = torch.rand(1, 1, 32, 32)
    heat = cam.heatmap(x, class_idx=0)
    assert heat.shape == (32, 32)
    assert heat.min() >= 0.0
    assert heat.max() <= 1.0 + 1e-6


def test_heatmap_shape_and_range_resnet():
    model, cam = _gradcam_for(ResNetWafer)
    x = torch.rand(1, 1, 32, 32)
    heat = cam.heatmap(x, class_idx=3)
    assert heat.shape == (32, 32)
    assert 0.0 <= heat.min() <= heat.max() <= 1.0 + 1e-6


def test_heatmap_distinguishes_patterns():
    """不同輸入圖案 → 熱圖不同（模型看不同位置）。"""
    model, cam = _gradcam_for(WaferCNN)
    x_center = torch.zeros(1, 1, 32, 32)
    x_center[0, 0, 12:20, 12:20] = 2.0   # 中央方塊
    x_edge = torch.zeros(1, 1, 32, 32)
    x_edge[0, 0, :, :2] = 2.0            # 左緣直條
    h1 = cam.heatmap(x_center, class_idx=4)  # near-full 較像中央大片
    h2 = cam.heatmap(x_edge, class_idx=0)
    assert not torch.allclose(h1, h2, atol=1e-3)


def test_heatmap_both_models_run_without_error():
    """兩個模型的 checkpoint 路徑（forward+backward）都能跑。"""
    for cls in (WaferCNN, ResNetWafer):
        model, cam = _gradcam_for(cls)
        x = torch.rand(1, 1, 32, 32)
        for c in (0, 5, 8):
            h = cam.heatmap(x, class_idx=c)
            assert h.shape == (32, 32)
