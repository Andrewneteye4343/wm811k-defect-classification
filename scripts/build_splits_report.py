"""真實資料驗證（M3）：清洗 → stratified 切分 → 每類比例報表。

執行（Docker）：
    docker compose run --rm app python scripts/build_splits_report.py

流程：載入 LSWMD.pkl（~150s）→ 有標籤清洗（172,950）→ stratified 70/15/15
→ 印每個 split 的類別分佈（驗證 stratify 正確、minority 在每區都有代表）
→ 從 train Dataset 抽 5 張確認整條前處理管線（crop→resize→tensor→label）。

預期：每類在 train/val/test 的比例 ≈ 全體比例（stratify 生效）；
near-full（全資料 149 張）在 val/test 也各約 20+ 張。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# 讓腳本可以直接執行（不需先 pip install 本專案），路徑對齊 src/wm811k/
# （pytest 靠 pytest.ini 的 pythonpath=src 找套件，但 python scripts/x.py 不會）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm811k import config  # noqa: E402
from wm811k.data import WM811KDataset, build_splits, labeled_frame
from wm811k.loader import load_lswmd


def _print_share(name: str, y: pd.Series, total: int) -> None:
    counts = y.value_counts().reindex(config.FAILURE_TYPES, fill_value=0)
    parts = [f"{c} ({c / total * 100:.1f}%)" for c in counts]
    print(f"  {name:<6} n={total:>7,}  " + "  ".join(parts))


def main() -> None:
    print("[1/4] Loading LSWMD.pkl (about 150 s) ...", flush=True)
    df = load_lswmd()
    print(f"      raw rows: {len(df):,}")

    print("[2/4] Filtering labeled + normalizing labels ...", flush=True)
    lf = labeled_frame(df)
    n_labeled = len(lf)
    n_unlabeled = len(df) - n_labeled
    print(f"      labeled: {n_labeled:,} ({n_labeled / len(df) * 100:.1f}%)  "
          f"unlabeled: {n_unlabeled:,} ({n_unlabeled / len(df) * 100:.1f}%)")

    print("[3/4] Stratified split 70/15/15 (seed 42) ...", flush=True)
    tr, va, te = build_splits(lf)
    y = lf["label"].to_numpy()
    print("\n  Class shares per split (count and % of labeled set):")
    _print_share("all", pd.Series(y), n_labeled)
    _print_share("train", pd.Series(y[tr]), len(tr))
    _print_share("val", pd.Series(y[va]), len(va))
    _print_share("test", pd.Series(y[te]), len(te))

    print("\n[4/4] Dataset smoke test (first 5 train samples) ...", flush=True)
    ds = WM811KDataset(lf, tr[:1000])
    for i in range(5):
        x, lbl = ds[i]
        print(f"      sample {i}: x={tuple(x.shape)} {x.dtype} "
              f"values [{float(x.min()):.0f}..{float(x.max()):.0f}]  "
              f"label={config.IDX_TO_LABEL[lbl]}")
    print("\nAll checks done. If class shares match and samples look sane, M3 is verified.")


if __name__ == "__main__":
    main()
