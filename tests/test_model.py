"""model.py 單元測試：forward shape、參數量、梯度流。"""
import pytest
import torch

from wm811k import config
from wm811k.model import WaferCNN


def test_forward_shape_single():
    model = WaferCNN()
    x = torch.randn(1, 1, config.TARGET_SIZE, config.TARGET_SIZE)
    out = model(x)
    assert out.shape == (1, config.NUM_CLASSES)


def test_forward_shape_batch():
    model = WaferCNN()
    x = torch.randn(17, 1, 32, 32)  # 非 2 的倍數 batch
    out = model(x)
    assert out.shape == (17, 9)


def test_parameter_count_under_1m():
    """基線目標：<1M 參數（實際約 156k）。"""
    model = WaferCNN()
    n = sum(p.numel() for p in model.parameters())
    assert n < 1_000_000
    assert n > 50_000  # 也不該小到沒學習能力


def test_gradient_flows():
    """backward 後每個參數都有 grad — 訓練迴圈能動的最低保證。"""
    model = WaferCNN()
    x = torch.randn(4, 1, 32, 32)
    y = torch.randint(0, 9, (4,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} 沒有梯度"
        assert torch.isfinite(p.grad).all()


def test_feature_map_pyramid():
    """特徵金字塔：32→16→8→4（每個 conv 段後解析度砍半）。"""
    model = WaferCNN()
    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        # 依序經過 features 的 conv 段，觀察中間 shape
        h = x
        shapes = []
        for i, layer in enumerate(model.features):
            h = layer(h)
            if isinstance(layer, torch.nn.MaxPool2d):
                shapes.append(tuple(h.shape))
    assert shapes == [(1, 16, 16, 16), (1, 32, 8, 8), (1, 64, 4, 4)]
