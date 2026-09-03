# WM-811K 本機開發/執行環境
# 用途：資料探查（M1）、EDA（M2）、管線 pytest（M3）、CPU 冒煙訓練（M4+）
# 注意：模型正式訓練主場是 Google Colab（GPU），本 image 只裝 CPU 版 torch
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# OpenMP runtime（torch 執行期可能用到；slim image 預設沒有）
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 1) torch/torchvision 先裝「CPU 版」——
#    坑：pip install torch 預設拉 CUDA 版（~2.5GB 下載、image 暴肥）；
#    指定 CPU index 只約 200MB。requirements.txt 的 torch>=2.1 之後會被視為已滿足而跳過
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2) 其餘依賴（numpy/pandas/matplotlib/seaborn/scikit-learn/pytest/kagglehub）
COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["python"]
