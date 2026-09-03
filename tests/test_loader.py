"""loader 相容層單元測試（M1 除錯成果固化，M2/M3 共用基礎）。"""
import numpy as np
import pandas as pd

from wm811k.loader import load_lswmd, unwrap_scalar


def test_unwrap_nested_lists():
    assert unwrap_scalar([["none"]]) == "none"
    assert unwrap_scalar(["Edge-Ring"]) == "Edge-Ring"
    assert unwrap_scalar([["Edge-Loc", "x"]]) == "Edge-Loc"  # 取第一元素


def test_unwrap_ndarray():
    assert unwrap_scalar(np.array([["none"]])) == "none"  # 真實資料結構
    assert unwrap_scalar(np.array(["Center"])) == "Center"
    assert unwrap_scalar(np.array([["Loc"]])) == "Loc"


def test_unwrap_empty_is_blank():
    assert unwrap_scalar([]) == ""
    assert unwrap_scalar(np.array([])) == ""
    assert unwrap_scalar(np.array([[]])) == ""
    assert unwrap_scalar(None) == ""
    assert unwrap_scalar(np.nan) == ""


def test_unwrap_bytes_and_scalars():
    assert unwrap_scalar(b"Random") == "Random"
    assert unwrap_scalar("none") == "none"
    assert unwrap_scalar(np.float64(1683.0)) == "1683.0"
    assert unwrap_scalar(1.0) == "1.0"


def test_load_fake_pkl(tmp_path):
    df = pd.DataFrame({
        "failureType": [np.array([["none"]]), np.array([], dtype=float)],
        "x": [1, 2],
    })
    p = tmp_path / "fake.pkl"
    df.to_pickle(p)
    out = load_lswmd(p)
    assert len(out) == 2
    vals = out["failureType"].map(unwrap_scalar).tolist()
    assert vals == ["none", ""]
