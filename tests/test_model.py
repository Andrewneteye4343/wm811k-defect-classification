"""model.py 單元測試：forward shape、參數量、梯度流。"""
import pytest
import torch

from wm811k import config
from wm811k.model import ResNetWafer, ResidualBlock, WaferCNN


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


# ── M6：ResidualBlock / ResNetWafer ────────────────────────────────

def test_resblock_same_channels_shape():
    """in==out、stride=1：輸入輸出 shape 不變（identity shortcut）。"""
    blk = ResidualBlock(16, 16)
    x = torch.randn(2, 16, 32, 32)
    assert blk(x).shape == x.shape


def test_resblock_downsample_shape():
    """in≠out：shortcut 1×1 升維（ResNet 的 downsample 路徑）。"""
    blk = ResidualBlock(16, 32, stride=2)
    x = torch.randn(2, 16, 32, 32)
    assert blk(x).shape == (2, 32, 16, 16)


def test_resnetwafer_forward_shape():
    model = ResNetWafer()
    x = torch.randn(4, 1, 32, 32)
    assert model(x).shape == (4, config.NUM_CLASSES)


def test_resnetwafer_param_count_reasonable():
    """殘差網參數適中（略多於 WaferCNN 且 <2M — GAP 取代大 FC 讓參數不暴漲）。"""
    n_resnet = sum(p.numel() for p in ResNetWafer().parameters())
    n_wafer = sum(p.numel() for p in WaferCNN().parameters())
    assert n_wafer < n_resnet < 2_000_000


def test_resnetwafer_gradients_flow():
    """殘差網梯度流（skip connection 的價值：深層也可訓練）。"""
    model = ResNetWafer()
    x = torch.randn(2, 1, 32, 32)
    model(x).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} 沒有梯度（skip 失效？）"


def test_resblock_identity_when_conv_zero():
    """殘差核心性質：F→0 時 output≈x（block 可退化成 identity）。"""
    blk = ResidualBlock(16, 16)
    blk.eval()
    for m in blk.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.zeros_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
    x_pos = torch.rand(2, 16, 32, 32)  # 正值輸入（避免 ReLU 非線性干擾）
    with torch.no_grad():
        out = blk(x_pos)
    assert torch.allclose(out, x_pos, atol=1e-4)
