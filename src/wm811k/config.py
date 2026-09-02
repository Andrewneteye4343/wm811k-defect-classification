"""集中管理路徑、類別對照與共用超參數。

M0 骨架版：類別對照依資料集文件先建立，M1 探查後若有出入會修正；
TARGET_SIZE 等資料相關決策在 M2 EDA 後定案。
"""
from pathlib import Path

# ── 路徑（以本檔位置推算專案根：src/wm811k/config.py → 上兩層）──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "LSWMD.pkl"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# ── 重現性 ──
SEED = 42

# ── 類別對照（資料集慣例順序；M1 探查後確認）──
FAILURE_TYPES = [
    "center",
    "donut",
    "edge-loc",
    "loc",
    "near-full",
    "random",
    "scratch",
    "edge-ring",
    "none",  # 正常片
]
LABEL_TO_IDX = {name: i for i, name in enumerate(FAILURE_TYPES)}
IDX_TO_LABEL = {i: name for i, name in enumerate(FAILURE_TYPES)}
NUM_CLASSES = len(FAILURE_TYPES)

# ── 影像處理（M2 依 EDA 尺寸分佈定案）──
TARGET_SIZE = 32  # 統一縮放/填補後的方形邊長
