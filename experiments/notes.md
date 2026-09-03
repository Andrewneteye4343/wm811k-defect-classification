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
| waferMap 數值 | {0,1,2}、dtype uint8、無 NaN | 0/1/2 | ✅（語意 M2 定案：1=pass/2=fail/0=背景）|
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

---

## M2 EDA 與定案（2026-09-03）

執行：`notebooks/01_eda.ipynb`（Docker + JupyterLab，6 探索單元 Part A–F）
→ Andrew 截圖回傳觀察 → 自證 cell 定案 0/1/2 語意 → 外部權威交叉驗證 →
**四項 ML 前置決策全部定案**，寫入 `src/wm811k/config.py`。

### 0/1/2 語意定案：1=pass / 2=fail / 0=無 die（背景）

自證 cell（每類前 20 張，統計 waferMap 數值比例 %）：

| 類 | 0 | 1 | 2 |
|---|---|---|---|
| none（正常片） | 22.1 | 74.4 | 3.5 |
| center（圓心缺陷） | 20.5 | 60.8 | 18.6 |

推導：
1. none 的正常 die 佔絕大多數 → **1 = pass**
2. center 缺陷集中在圓心 → 2 由 3.5% 暴增到 18.6%（+15pp）→ **2 = fail**
3. 0 恆定 ~21% ≈ 1−π/4（方形陣列內切圓的圓外面積比 21.5%）→ **0 = 背景**

交叉驗證（Andrew 主動要求找反證 — 正確的科學態度）：
- MathWorks 官方範例（讀 MIR 官方 MATLAB 檔 `WM811K.mat`）明載：0=background、
  1=correctly behaving dies、2=defective dies
- globalsino 引 waferMap 欄位說明：0=no die、1=normal die、2=defective die

⚠️ **「0=pass/1=fail」迷思的來源**：部分研究 pre-processing 把原始碼重映射成
二元缺陷 mask（例：Infineon StreamGen 的 `preprocess`：1→0、2→1、0→NaN，
註解 "switch values for a better zero-center distribution"）→ 論文讀者誤當原始編碼。
教訓：domain 資料語意以官方文件 + 資料自證為準，勿照單全收二手文獻。

### 四項 ML 前置決策（Andrew 表決，全數定案）

1. **尺寸策略：裁最大內接方 → 等比縮放 32×32**。wafer 圓內切方形陣列，裁掉
   皆背景（0）零資訊損失；直接 resize 會把圓壓成橢圓（edge-ring 環變形），
   人為製造混淆。極端長條圖（寬 3–5）M3 管線再處理。
2. **9 類（保留 none）**：none = 無系統性圖案（零星 fail die 3.5% 不構成 pattern，
   仍屬 none）；產線真實任務需判定「正常片」；8 類強迫分類會把正常片硬分進缺陷類。
3. **自行 stratified 70/15/15（seed 42）**：內建 split 無 validation、有標籤內
   Test 68.6% 失衡、train 太少。**M6 加 by-lot GroupShuffleSplit 對照組**量化
   lot 洩漏造成的高估（同 lot wafer 高度相似，隨機切分會高估效能）。
4. **輸入尺寸 32×32**：die 區中位 ~30–40 見方，解析度接近原圖、CPU 訓練可行；
   64×64 留給 M5 當實驗變因。

### Andrew 視覺觀察紀錄（EDA 圖鑑的價值）

- none 圖「不乾淨」：有零星黑點（fail die）→ 印證 none≠零缺陷而是無系統性圖案
- 預測 **edge-ring vs none 最易混淆**（圖鑑觀察；文獻常提 random/scratch 亦難分）
  → M7 用混淆矩陣實證
- lot 抽樣圖顯示同 lot wafer 高度相似 → lot 洩漏風險真實存在
- 尺寸分佈確認多數圖 20–50 見方；dieSize 多離散值（不同產品線，非資料錯誤）
