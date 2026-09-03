# 實驗筆記（experiments/notes.md）

記錄每個里程碑的「探查結果 vs 假設」比對、決策理由與心得。

---

## M1 資料探查（2026-09-02）

執行：`scripts/inspect_data.py`（Docker 容器 pandas 3.x 讀 2015 年 py2 + pandas 0.16 pickle，
需 alias + latin1 相容層，見 scripts 內註解）。載入 ~157s，掃描 811,457 張 ~3s。

### 驗證通過（與計畫/README 假設一致）

| 項目 | 實際值 | 假設 | 結果 |
|---|---|---|---|
| 總列數 | 811,457 | 811,457 | ✅ |
| 有標籤 | 172,950 (21.3%) | ~21% | ✅ |
| none 佔有標籤 | 147,431 (85.2%) | ~85% | ✅ |
| waferMap 數值 | {0,1,2}、dtype uint8、無 NaN | 0/1/2 | ✅（pass/fail/無 die 語意待 M2 視覺確認）|
| 類別集合 | 9 類（僅大小寫不同）| 9 類 | ✅ |

### 發現與出入（影響後續決策）

1. **欄位名 `trianTestLabel`**：MIR Lab 原始拼錯（少一個 i），非文件寫的
   trainTestLabel → 程式需動態找欄位，不能硬編。
2. **failureType / trianTestLabel 每格是 2D numpy array**（py2 存法，
   `array([['none']])`、空 = 無標籤）→ 前處理必須 unwrap（已實作 `_unwrap_scalar`）。
   *這是 v4 前統計卡死數小時的根因（巢狀物件無法 hash）。*
3. **標籤為 Title Case**（Edge-Ring / Center…，唯 none 小寫）→ config 已用小寫
   FAILURE_TYPES；前處理時 `lower()` 標準化（M2 定案）。
4. **內建切分不平衡且對齊有標籤**：有標籤 172,950 中 Test 118,595 (68.6%) 多於
   Training 54,355 (31.4%)；無標籤 638,507 列的 trianTestLabel 全為空。
   → 傾向**不用內建切分**，M3 自行 stratified（種子 42）。
5. **唯一 lot 46,293**（文件/README 寫 46,393，差 100）→ 以實際資料為準。
6. **waferMap 尺寸**：列 6–300、行 3–205，中位約 35×35；最常見 32×29 (108,687)、
   25×27 (64,083)、49×39 (39,323)、26×26 (30,078)。存在 300 寬的極端大圖
   → M2 尺寸定案需討論「直接 resize（大圖細節流失）vs crop 內接方」。
7. **dieSize 範圍大**：min 3、median 953、max 48,100，多種離散值（不同產品線），
   有極端值 → M2 EDA 檢查是否資料錯誤。
8. waferIndex 為 float64、範圍 1–25（每 lot ≤25 片）。

### 效能教訓（重要）

- 811,457 張逐張 `np.unique` 收集數值集合 → 實測卡死數小時（巢狀/大陣列時退化）。
- 修正：Pass 1 全量只讀型別與 shape（metadata，O(1)/張）+ Pass 2 數值集合只掃
  有標籤子集且用 min/max 快速路徑（落在 [0,2] 即 ⊆{0,1,2}）→ **2 秒完成**。
- 通則：大資料先「型別診斷」（抽樣 1000 筆型別分佈）再做全量統計；巢狀結構
  先 unwrap 再 value_counts / nunique，避免無謂卡死。
