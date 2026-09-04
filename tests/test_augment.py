"""augment.py 單元測試：D4 變體保真性（格點不插值 → 值域不變）。"""
import torch

from wm811k.augment import d4_variants, random_d4


def _sample_map():
    """構造不對稱圖案（左上角有值）→ 8 個 D4 變體應全不同。"""
    x = torch.zeros(1, 8, 8)
    x[0, 1, 2] = 1
    x[0, 3, 3] = 2
    return x


def test_d4_variants_count_and_values():
    x = _sample_map()
    variants = d4_variants(x)
    assert len(variants) == 8
    for v in variants:
        assert v.shape == x.shape
        # 純排列（無插值）→ 值域不變
        assert set(torch.unique(v).tolist()) <= {0.0, 1.0, 2.0}
        # 非零像素數不變
        assert (v != 0).sum() == (x != 0).sum()


def test_d4_variants_all_distinct():
    """8 個變體互不相同（圖案不對稱才測得出）。"""
    x = _sample_map()
    variants = d4_variants(x)
    seen = set()
    for v in variants:
        key = tuple(v.flatten().tolist())
        assert key not in seen, "D4 變體重複"
        seen.add(key)
    assert len(seen) == 8


def test_d4_variants_include_original():
    """0° 旋轉 + 無鏡像 = 原圖。"""
    x = _sample_map()
    assert torch.equal(d4_variants(x)[0], x)


def test_random_d4_output_is_valid_variant():
    """random_d4 的輸出必為 8 變體之一（值域與像素數守恆）。"""
    x = _sample_map()
    torch.manual_seed(0)
    for _ in range(20):
        out = random_d4(x)
        assert out.shape == x.shape
        assert set(torch.unique(out).tolist()) <= {0.0, 1.0, 2.0}
        assert (out != 0).sum() == (x != 0).sum()


def test_random_d4_not_deterministic():
    """兩次呼叫不同 seed → 大概率不同變體（增強要多樣性）。"""
    x = _sample_map()
    torch.manual_seed(1)
    a = random_d4(x)
    torch.manual_seed(2)
    b = random_d4(x)
    assert not torch.equal(a, b)  # 8 變體中撞同一個的機率 1/8


def test_random_d4_rotation_equivariance_sanity():
    """旋轉兩次 90° = 180°（torch.rot90 語意驗證）。"""
    x = _sample_map()
    r2 = torch.rot90(torch.rot90(x, 1, dims=(-2, -1)), 1, dims=(-2, -1))
    assert torch.equal(r2, torch.rot90(x, 2, dims=(-2, -1)))
