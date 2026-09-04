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

---

## M3 前處理管線 + Dataset + 切分（2026-09-03）

執行：`python -m pytest`（**34 passed**：3 smoke + 5 loader + 17 preprocess + 9 data）
+ `python scripts/build_splits_report.py`（真實資料，~4 分鐘）。

### 定案落地（M2 決策 → 程式碼）

| 決策 | 實作 |
|---|---|
| 裁內接方 | `preprocess.crop_to_square`：以 die bbox 中心裁 min(高,寬) 見方（trim 全背景邊 → 比裁陣列中央穩健，off-center lot 不裁歪；die 全保留 = 零資訊損失，測試驗證非零像素數不變）|
| 縮放 32×32 | `resize_map`：PIL **NEAREST** — waferMap 是離散類別 {0,1,2}，nearest 不產生 bilinear 的中間值 |
| 標籤 lower() | `normalize_label` = unwrap_scalar + strip + lower（`labeled_frame` 批量套用）|
| stratified 70/15/15 | `build_splits`：sklearn train_test_split 兩階段（第二階段 seed+1 避免兩次 shuffle 序列相關），回傳 position indices（train/val/test 共享同一份 df 零複製）|

### 真實資料驗證數字（172,950 有標籤）

| split | n | 類別比例（9 類序）— 每列與 all 一致 = stratify 生效 |
|---|---|---|
| all | 172,950 | 2.5/0.3/3.0/2.1/0.1/0.5/0.7/5.6/85.2% |
| train | 121,065 (70.0%) | 同左（near-full 104 張）|
| val | 25,942 (15.0%) | 同左（near-full 23 張）|
| test | 25,943 (15.0%) | 同左（near-full 22 張）|

near-full 全資料僅 149 張，val/test 仍各有 22–23 張 → minority 每區都有代表，
這是隨機切分辦不到的（也是 M6 by-lot 對照要量化的另一面）。

### 技術教訓

- **pytest 全綠 ≠ `python scripts/x.py` 能跑**：pytest 靠 pytest.ini `pythonpath=src`
  找套件；直接執行時 sys.path 只有 scripts/ → 每個可直接執行的 script 都要自帶
  `sys.path.insert(0, parents[1]/"src")`（build_splits_report.py 初版漏了，M1 的
  inspect_data.py 有 → 已統一 pattern）。
- **PIL `np.asarray(img)` 回傳 read-only buffer** → `torch.from_numpy` 抱怨 →
  用 `np.array(img)`（強制 copy，32×32 小圖成本可忽略）。
- sklearn `train_test_split(stratify=)` 要求每類 ≥2 筆（兩階段切分時小類會踩）
  → 測試合成資料每類給足樣本。

---

## M4 基線 CNN（2026-09-04）

執行：Colab T4 GPU、`scripts/train_baseline.py --epochs 30`（30 epochs 跑滿、
best @ epoch 25 — 未觸發 early stop）。修正：run_training 漏 `model.to(device)`
（GPU 首跑即 RuntimeError — CPU 冒煙測不出 device bug）。

### 基線結果（WaferCNN 155,657 參數、無權重 CrossEntropy、Adam lr 1e-3）

| 指標 | 值 | 對照 |
|---|---|---|
| test macro-F1 | **0.7952** | naive 全猜 none ≈ 0.10 |
| test accuracy | 0.9630 | none 主導（僅對照）|
| best val macro-F1 | 0.8152 @ epoch 25 | |

per-class test F1：none 0.984 / edge-ring 0.966 / center 0.905 / near-full 0.889 /
donut 0.826 / random 0.825 / edge-loc 0.757 / loc 0.657 / **scratch 0.347**（最弱，
印證文獻 MathWorks 實測 Scratch 最難）。near-full 僅 22 張卻 0.889 — 圖案極端
明確（幾乎全黑）→ 少樣本但與其他類距離遠。

### 學習曲線觀察（overfitting 活教材）

epoch 1 完 val macro-F1 已 0.598（121k 樣本一輪內完成大部分學習，「全猜 none」
的 0.10 起點只存在於 epoch 0）；epoch 12–30 macro-F1 高原 ~0.75–0.81；
**epoch 17 起 train_loss 續降（0.065→0.042）但 val_loss 反向上升（0.148→0.227）**
= overfitting 開始 → early stopping 的 patience 8 保護 checkpoint 停在 epoch 25。

### 混淆矩陣流向（LogNorm 圖觀察，Andrew 實測回報）

- **scratch 誤判流向（多→少）：none > loc > edge-loc** — scratch recall 僅 0.30
- **loc 誤判流向：none > scratch > edge-loc**
- **edge-ring ↔ none 無明顯互混** → 修正 M2 預測（視覺直覺被資料推翻）

### 密度軸統一理論（M2 random/none 發現的延伸）

scratch（細線、die 少）與小 loc cluster 在 32×32 縮放後 fail die 密度低 →
**在密度軸上靠近 none** → 被併入 none。誤判根源不是「形狀相似」而是
「低密度圖案 ≈ none」— 與 M2 random/none 的密度軸發現串成同一條線：
WM811K 分類的核心挑戰之一 = **fail die 密度連續體**（none ← 低密度缺陷 ←
高密度結構圖案）。對比 edge-ring/center/near-full 有「大片高對比結構」
→ 遠離 none → 高分。M5 方向：weighted loss 直接懲罰「偷懶猜 none」。

---

## M5 類別不平衡實驗（2026-09-04）

方法：一次一變因（同 seed、同切分、同 30 epochs），全記錄 experiments/runs.csv。

### 結果對照（test macro-F1；baseline 錨點 0.7952）

| run | 變因 | test macro-F1 | Δ | 判定 |
|---|---|---|---|---|
| m4_baseline | 無 | 0.7952 | — | 錨點 |
| m5a_weighted_fixed | weighted CE（全權重，差距 1000 倍）| 0.7306 | −0.065 | ❌ 過度補償 |
| m5b_sampler | balanced sampler | 0.7655 | −0.030 | ❌ 過度補償 |
| m5c_augment | D4 增強 | 0.8122 | +0.017 | ✅ |
| **m5d_combined** | **D4 增強 + sqrt 溫和權重** | **0.8238** | **+0.029** | ✅ **冠軍** |

### 踩坑：m5a weights 全 1 bug（教訓深刻）

m5a 首跑 log 顯示 `weights=[1,1,...×9]` — 原因：`y_train == c`（label 字串比對 int）
全 False → counts 全 0 → 防呆給全 1 → weighted loss 靜默失效（等同 baseline）。
⚠️ 早在使用者先前自跑 Colab 實驗就出現過 weights 全 1 紅旗，當時標記「需確認」
未追 — 這次同訊號才確認為 bug。教訓：**看到 weights 全 1 / 任何「應該生效卻
沒生效」的 log 訊號，立即停下來查 code，不要繼續跑**。修復：data.py 新增
`label_counts()`（字串→index counts）+ 回歸測試（防再犯）。

### 過度補償統一理論（兩個平衡方案的失敗模式）

全權重 weighted（0.7306）與 balanced sampler（0.7655）都低於 baseline，
失敗模式相同：把 minority 重要性拉高 → 模型過度補償 → 狂猜 minority →
**precision 崩盤**（scratch precision：baseline 0.41 → sampler 0.18 →
weighted 0.08）。權重差距 1000 倍太激進 + 訓練不穩（early stop @ epoch 11）。
文獻常見現象：minority 稀少且與 none 在密度軸重疊時，強迫平衡犧牲 precision。

### 溫和化成功（m5d）

sqrt 權重（差距 1000→31 倍：none 0.36 vs near-full 11.37）不再過度補償：
scratch F1 **0.537**（precision 0.57 保住、recall 0.51 拉高 — 對比 m5a_fixed
precision 0.08 崩盤）。macro-F1 三階段演化：
**0.7952（baseline）→ 0.8122（+augment）→ 0.8238（+sqrt 權重）**。
權重代價：center/edge-loc/edge-ring/none 小降（−0.01~−0.02），但 minority
大升（scratch +0.19、donut/random/near-full +0.04~0.08）→ 淨效應正向。

### M5 最終配方（M6/M7 沿用）

**WaferCNN + D4 augmentation（random_d4）+ CrossEntropy(sqrt 權重,
power=0.5) + Adam lr 1e-3 + 30 epochs** → test macro-F1 **0.8238**。
scratch 三階段：0.347（M4）→ 0.493（M5c）→ 0.537（M5d）。

