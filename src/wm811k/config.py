"""集中管理路徑、類別對照與共用超參數。

M2 EDA 定案版（2026-09-03）：
- waferMap 0/1/2 語意：1=pass（正常 die）、2=fail（缺陷 die）、0=無 die（背景）
  （自證 cell + MathWorks 官方文件交叉驗證，見 experiments/notes.md M2 章節）
- 標籤 lower() 標準化（真實資料 Title Case：Edge-Ring/Center…，none 小寫）
- 9 類含 none（產線任務需判定正常片；macro-F1 應對 85% 不平衡）
- 尺寸策略：裁最大內接方再等比縮放（wafer 圓內切方形陣列，裁掉皆背景）
- 切分：自行 stratified 70/15/15（內建 split 無 validation 且比例失衡），種子 42
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

# ── 類別對照（資料集慣例順序；M1 探查確認 9 類、M2 定案保留 none）──
FAILURE_TYPES = [
    "center",
    "donut",
    "edge-loc",
    "loc",
    "near-full",
    "random",
    "scratch",
    "edge-ring",
    "none",  # 正常片（無系統性缺陷圖案，允許零星 fail die）
]
LABEL_TO_IDX = {name: i for i, name in enumerate(FAILURE_TYPES)}
IDX_TO_LABEL = {i: name for i, name in enumerate(FAILURE_TYPES)}
NUM_CLASSES = len(FAILURE_TYPES)

# ── 切分（M2 定案：自行 stratified；M6 加 by-lot GroupShuffleSplit 洩漏對照）──
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

# ── 影像前處理（M2 EDA 定案）──
TARGET_SIZE = 32  # 統一輸入方形邊長（資料 die 區中位 ~30–40 見方）
CROP_TO_SQUARE = True  # 先裁最大內接方（含 die 區）再等比縮放，避免幾何變形
