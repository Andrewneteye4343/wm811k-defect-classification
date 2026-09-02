# WM-811K 晶圓圖瑕疵分類（Wafer Map Defect Classification）

以 WM-811K 真實晶圓圖資料集，用 PyTorch 建立瑕疵分類模型：將晶圓圖分為 **8 種瑕疵圖形 + 正常（none）共 9 類**。
求職用深度學習作品集專案 — 完整走過 資料探查 → 前處理 → 建模 → 改善 → 評估 → 交付。

## 狀態（里程碑）

- [x] **M0** 專案骨架與環境
- [ ] **M1** 資料取得與結構探查
- [ ] **M2** EDA 與視覺化（9 類 vs 8 類、固定尺寸定案）
- [ ] **M3** 前處理管線 + Dataset + 切分（pytest）
- [ ] **M4** 小型 CNN 基線（手寫訓練迴圈）
- [ ] **M5** 類別不平衡處理（weighted loss / sampler / 旋轉增強）
- [ ] **M6** 模型強化 + lot 洩漏對照
- [ ] **M7** 可解釋性（Grad-CAM）與最終評估
- [ ] **M8** 交付包裝（README 重現、demo）

## 資料集

- 811,457 張晶圓圖、來自 46,393 個 lot（真實晶圓廠量產資料）
- 僅 172,950 張（約 21%）有標籤；none（正常）佔約 85%
- 來源：MIR Lab（論文：Wu, Jang & Chen, IEEE TSM 2015）
- Kaggle：`qingyi/wm811k-wafer-map`（單一 `LSWMD.pkl`）

## 目錄結構

```
wm811k-defect-classification/
├── data/raw/LSWMD.pkl        # 原始資料（不入 git）
├── data/processed/           # 前處理結果（不入 git）
├── notebooks/                # EDA / 實驗 notebook
├── scripts/                  # 一次性/輔助腳本
├── src/wm811k/               # 主程式碼（套件）
│   ├── config.py             # 路徑、類別對照、種子、超參數
│   ├── data.py               # 載入 pkl、清洗、Dataset（M3）
│   ├── preprocess.py         # 縮放/填補 → 固定尺寸（M3）
│   ├── metrics.py            # macro-F1、per-class PRF（M3）
│   ├── model.py              # CNN 模型（M4）
│   ├── train.py              # 訓練迴圈（M4）
│   └── evaluate.py           # 評估、混淆矩陣、Grad-CAM（M4/M7）
├── tests/                    # pytest 單元測試（管線為主）
├── experiments/              # runs.csv 實驗紀錄、notes.md 心得
├── artifacts/                # 圖表、checkpoint（不入 git）
├── requirements.txt
└── pytest.ini
```

## 開發環境

- 訓練主場：**Google Colab**（GPU）；本機（4GB VRAM）為輔
- Python 3.11 + PyTorch 2.x；管線用 pytest 驗證
- Colab 起手（M1 後適用）：clone repo → `!pip install kagglehub` → 下載 `LSWMD.pkl` → 執行 scripts / notebook
