"""冒煙測試：確認套件可匯入、基本常數正確。"""
from wm811k import config


def test_nine_classes():
    assert len(config.FAILURE_TYPES) == 9
    assert config.NUM_CLASSES == 9
    assert "none" in config.FAILURE_TYPES


def test_label_mapping_roundtrip():
    for i, name in enumerate(config.FAILURE_TYPES):
        assert config.LABEL_TO_IDX[name] == i
        assert config.IDX_TO_LABEL[i] == name


def test_paths_resolve():
    assert config.PROJECT_ROOT.exists()
    assert config.RAW_DATA_PATH.name == "LSWMD.pkl"
