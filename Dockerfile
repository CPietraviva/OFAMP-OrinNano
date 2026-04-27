FROM python:3.11-slim-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch CPU
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# TimesFM 2.5 da GitHub con supporto xreg per covariates
RUN pip install --no-cache-dir "timesfm[torch,xreg] @ git+https://github.com/google-research/timesfm.git"

# App dependencies
RUN pip install --no-cache-dir \
    streamlit \
    pandas \
    numpy \
    plotly \
    requests \
    yfinance \
    huggingface_hub

RUN mkdir -p /app/data

# Pre-scarica il modello 2.5
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('google/timesfm-2.5-200m-pytorch')"