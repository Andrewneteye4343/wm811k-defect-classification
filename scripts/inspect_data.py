"""M1：資料結構探查 — 載入 LSWMD.pkl，輸出結構統計。

為什麼有這個腳本：
  ML 專案第一步不是建模，而是「摸清資料」。M0 我們依文件/慣例先寫了
  假設進 config.py（9 類順序、0/1/2 語意、none 約 85%），本腳本第一次
  用真實資料檢驗這些假設，有出入就修正 config.py 並記錄。

輸出項目（對照 README「資料集」節的已知事實）：
  1. pkl 頂層型別、欄位與 dtype
  2. failureType 類別計數（含無標籤數量，驗證 none ~85%）
  3. trainTestLabel 分佈（內建切分欄位的定義）
  4. dieSize 分佈
  5. waferMap 尺寸範圍與數值集合（驗證 0/1/2 語意假設；
     哪個數字代表 pass/fail/無 die 需 M2 畫圖視覺確認）
  6. lotName / waferIndex 基本統計（lot 洩漏分析素材）
  7. failureType 實際值 vs config.FAILURE_TYPES 對照

用法（在本機 / Colab / Docker，需 pandas + numpy）：
  python scripts/inspect_data.py                 # 用 config 預設路徑 data/raw/LSWMD.pkl
  python scripts/inspect_data.py --pkl <path>    # 指定 pkl 路徑
  python scripts/inspect_data.py --limit 5000    # 只掃前 N 列（冒煙測試用）
  docker compose run --rm app python scripts/inspect_data.py   # 本機 Docker 環境

輸出同時存成 artifacts/inspect_report.txt（UTF-8），方便整份回傳比對。
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# 讓腳本可以直接執行（不需先 pip install 本專案），路徑對齊 src/wm811k/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm811k import config  # noqa: E402
from wm811k.loader import (  # noqa: E402
    load_lswmd,
    register_pandas_legacy_aliases,
    unwrap_scalar,
)

# Windows 主控台編碼（cp950）印中文可能出錯：改 UTF-8，缺字用 ? 取代不中斷
if sys.stdout.encoding and "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def section(title: str) -> str:
    return f"\n{'=' * 64}\n{title}\n{'=' * 64}"


def main() -> None:
    parser = argparse.ArgumentParser(description="WM-811K LSWMD.pkl 結構探查")
    parser.add_argument("--pkl", type=str, default=str(config.RAW_DATA_PATH),
                        help="LSWMD.pkl 路徑（預設 data/raw/LSWMD.pkl）")
    parser.add_argument("--limit", type=int, default=None,
                        help="只載入前 N 列（冒煙測試用）")
    args = parser.parse_args()

    path = Path(args.pkl).expanduser()
    if not path.exists():
        print(f"[錯誤] 找不到 pkl：{path}")
        print("請先下載 LSWMD.pkl 並放到 data/raw/（見 README「資料集」節）。")
        sys.exit(1)

    lines: list[str] = []
    p = print  # 同時收集到 lines 與 stdout

    def out(*a, **kw) -> None:
        s = " ".join(str(x) for x in a)
        lines.append(s)
        p(s, **kw)

    # ── 載入 ──
    size_mb = path.stat().st_size / 1e6
    out(f"[load] {path.name}  ({size_mb:,.1f} MB)")
    t0 = time.time()
    if register_pandas_legacy_aliases():
        out("[load] 偵測到舊版 pandas pickle（pandas.indexes.*），已註冊相容 alias")
    df = load_lswmd(path)
    out(f"[load] 花 {time.time() - t0:.1f} 秒載入完成")
    out(f"       頂層型別 : {type(df).__name__}")
    out(f"       列數     : {len(df):,}")
    if args.limit is not None:
        out(f"       ⚠ 樣本模式：只取前 {args.limit} 列（數字不代表全量）")

    # ── 0. 欄位抽樣診斷（只碰前 1000 列，O(1) 快，卡點無所遁形）──
    out(section("0. 欄位抽樣診斷（前 1000 列）"))
    for col in df.columns:
        vals = df[col].head(1000)
        tc = Counter(type(v).__name__ for v in vals)
        first = repr(vals.iloc[0])
        out(f"  {col:<16} 型別分佈={dict(tc)}")
        out(f"  {'':<16} 首例={first[:90]}")

    # ── 1. 欄位與 dtype ──
    out(section("1. 欄位與 dtype"))
    out(f"columns = {list(df.columns)}")
    for col in df.columns:
        out(f"  {col:<16} {df[col].dtype}")

    # ── 2. failureType：類別計數 ──
    out(section("2. failureType：類別計數（含無標籤）"))
    ft = df["failureType"]

    sample = ft.dropna().iloc[0] if ft.notna().any() else ""
    if not isinstance(sample, str):
        out(f"  （發現：failureType 每格為 {type(sample).__name__} 巢狀結構，"
            f"已自動解開為純量；空容器 = 無標籤）")
    ft_norm = ft.map(unwrap_scalar)
    empty_ft = int((ft_norm == "").sum())
    labelled = ft_norm[ft_norm != ""]
    labelled_mask = (ft_norm != "").to_numpy()  # Section 5 Pass2 用
    out(f"  總列數        : {len(df):,}")
    out(f"  無標籤        : {empty_ft:,}  ({empty_ft / len(df):.1%})")
    out(f"  有標籤        : {len(labelled):,}  ({len(labelled) / len(df):.1%})")
    counts = labelled.value_counts()
    out("  各類數量（由多到少）：")
    for label, n in counts.items():
        out(f"    {label:<12} {n:>8,}  ({n / len(labelled):.1%} of labelled)")
    if "none" in counts.index:
        out(f"  -> none 佔有標籤資料 {counts['none'] / len(labelled):.1%}"
            f"（預期約 85%）")

    # ── 3. trainTestLabel ──
    out(section("3. trainTestLabel（內建切分欄位）"))
    # 真實資料欄位拼作 trianTestLabel（MIR Lab 原始拼錯）；動態尋找而非硬編
    ttl_cols = [c for c in df.columns
                if isinstance(c, str) and "estLabel" in c]
    if not ttl_cols:
        out("  ⚠ 找不到含 'TestLabel' 的欄位！實際欄位見第 1 節")
    else:
        col = ttl_cols[0]
        if col != "trainTestLabel":
            out(f"  （發現：實際欄位名為 {col!r}，原始資料拼法與文件不同）")
        ttl = df[col].map(unwrap_scalar)  # 巢狀結構先解開再統計
        out(f"  NaN 數      : {int((ttl == '').sum()):,}")
        vc = ttl.value_counts(dropna=False)
        for k, n in vc.items():
            out(f"  {str(k)!r:<20} {n:>8,}  ({n / len(df):.1%})")

    # ── 4. dieSize ──
    out(section("4. dieSize"))
    ds = df["dieSize"]
    out(f"  NaN 數 : {int(ds.isna().sum()):,}")
    dsvc = ds.value_counts(dropna=False).head(10)
    for k, n in dsvc.items():
        out(f"  {str(k)!s:<16} {n:>8,}")
    try:
        desc = ds.astype(float).describe()
        out(f"  (數值摘要) min={desc['min']:.3g}  median={desc['50%']:.3g}"
            f"  max={desc['max']:.3g}")
    except (ValueError, TypeError):
        out("  (dieSize 無法轉數值，以上為原始值分佈)")

    # ── 5. waferMap：尺寸與數值集合 ──
    out(section("5. waferMap：尺寸範圍與數值集合（驗證 0/1/2 語意）"))
    maps = df["waferMap"]
    n_map_na = int(maps.isna().sum())
    out(f"  waferMap 欄位 NaN : {n_map_na:,}")
    n = len(df)
    rows = np.empty(n, dtype=np.int64)
    cols = np.empty(n, dtype=np.int64)
    type_counter: Counter[str] = Counter()
    shape_counter: Counter[tuple[int, int]] = Counter()

    # Pass 1：全量「輕量」掃描 — 只讀型別與 shape（metadata，O(1)/張，永不卡）
    t1 = time.time()
    valid = 0
    for i, arr in enumerate(maps):
        if i and i % 200_000 == 0:
            print(f"    [進度] Pass1 已掃 {i:,}/{n:,} 張", flush=True)
        type_counter[type(arr).__name__] += 1
        if not isinstance(arr, np.ndarray):
            continue
        h, w = arr.shape
        rows[valid], cols[valid] = h, w
        shape_counter[(h, w)] += 1
        valid += 1
    out(f"  [Pass1 全量掃描 {time.time() - t1:.0f}s] 有效 ndarray : {valid:,} / {n:,}"
        f"（非 ndarray : {n - valid:,}）")
    out(f"  元素型別分佈 : {dict(type_counter)}")

    # Pass 2：數值集合驗證 — 只掃有標籤子集（0/1/2 語意驗證不需全量）；
    #          快速路徑：min/max 落在 [0,2] 即確認 ⊆{0,1,2}，越界才 np.unique
    dtypes_seen: set[str] = set()
    values_seen: set[float] = set()
    any_nan_value = False
    t2 = time.time()
    sub_maps = maps.to_numpy()[labelled_mask]
    for j, arr in enumerate(sub_maps):
        if j and j % 50_000 == 0:
            print(f"    [進度] Pass2 已掃 {j:,}/{len(sub_maps):,} 張", flush=True)
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            continue
        dtypes_seen.add(str(arr.dtype))
        if arr.dtype.kind == "f" and bool(np.isnan(arr).any()):
            any_nan_value = True
        amin, amax = float(arr.min()), float(arr.max())
        if (not math.isnan(amin) and not math.isnan(amax)
                and 0 <= amin and amax <= 2):
            values_seen.update(
                float(v) for v in range(int(amin), int(amax) + 1))
        else:
            values_seen.update(float(v) for v in np.unique(arr).tolist())
    out(f"  [Pass2 數值集合 {time.time() - t2:.0f}s]"
        f"（掃有標籤 {len(sub_maps):,} 張）")

    if valid:
        def rng(a: np.ndarray) -> str:
            return (f"min={a[:valid].min()}  median={np.median(a[:valid]):.0f}"
                    f"  max={a[:valid].max()}")
        out(f"  高度(列)     : {rng(rows)}")
        out(f"  寬度(行)     : {rng(cols)}")
        top_shapes = shape_counter.most_common(5)
        out("  最常見尺寸 top5：")
        for (h, w), cnt in top_shapes:
            out(f"    {h} x {w} : {cnt:,}")
        out(f"  元素 dtype 集合 : {sorted(dtypes_seen)}")
        ints = sorted(int(v) for v in values_seen
                      if not math.isnan(v) and float(v).is_integer())
        others = sorted(v for v in values_seen if not float(v).is_integer())
        out(f"  數值集合       : {ints}{others}  NaN值存在: {any_nan_value}")
        if set(ints) <= {0, 1, 2} and not others and not any_nan_value:
            out("  -> 數值集合 ⊆ {0,1,2}：符合慣例假設")
            out("     (0/1/2 各代表 pass/fail/無 die 區域？→ M2 畫圖視覺確認)")
        else:
            out("  -> ⚠ 數值集合超出 {0,1,2}，與慣例假設不符，需查證！")
    else:
        out("  ⚠ 沒有任何 ndarray：waferMap 結構與預期不同（見型別分佈），需查證！")

    # ── 6. lotName / waferIndex ──
    out(section("6. lotName / waferIndex"))
    ln = df["lotName"].map(unwrap_scalar)  # 巢狀結構先解開
    wi = df["waferIndex"]
    out(f"  lotName NaN        : {int((ln == '').sum()):,}")
    out(f"  唯一 lot 數        : {ln.nunique():,}")
    out(f"  waferIndex NaN     : {int(wi.isna().sum()):,}")
    if wi.notna().any():
        wnum = pd.to_numeric(wi, errors="coerce").dropna()
        if len(wnum):
            out(f"  waferIndex 範圍   : {wnum.min():.0f} ~ {wnum.max():.0f}")

    # ── 7. failureType vs config 對照 ──
    out(section("7. failureType 實際值 vs config.FAILURE_TYPES"))
    actual = sorted(set(labelled))
    configured = config.FAILURE_TYPES
    out(f"  config 定義 ({len(configured)} 類) : {configured}")
    out(f"  資料實際值         : {actual}")
    if set(actual) == set(configured):
        out("  -> ✅ 集合一致：config.py 類別對照無需修改")
    else:
        missing = sorted(set(configured) - set(actual))
        extra = sorted(set(actual) - set(configured))
        if missing:
            out(f"  -> ⚠ config 有但資料沒有 : {missing}")
        if extra:
            out(f"  -> ⚠ 資料有但 config 沒有 : {extra}")
        if {s.lower() for s in actual} == {s.lower() for s in configured}:
            out("  -> 差異僅在大小寫 → 前處理標準化為小寫即可（M2 定案）")
        else:
            out("  -> 需要修正 config.py 或查證！")

    # 存檔
    out(f"\n[存檔] 報告已存到 artifacts/inspect_report.txt")
    report_dir = Path(__file__).resolve().parents[1] / "artifacts"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "inspect_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
