# 📈 OFAMP — Oasis-Finance AI Multi-Predictor
### v1.0 — Running on NVIDIA Jetson Orin Nano

> ⚠️ **DISCLAIMER:** This is an experimental project for educational purposes only.  
> Predictions do not constitute financial advice or investment recommendations.  
> Any financial decision is the sole responsibility of the user.

---

## 🧠 What is OFAMP?

OFAMP is an AI-powered financial forecasting tool built on **Google TimesFM 2.5** (200M parameters), running locally on an **NVIDIA Jetson Orin Nano 8GB**. It combines time series forecasting with macro-economic covariates to generate price predictions for stocks, ETFs, and cryptocurrencies.

**Key features:**
- 📊 Price forecasting with confidence bands (p10/p90 quantiles) — native TimesFM output
- 🔗 Macro covariates: VIX, DXY (US Dollar Index), TNX (10Y Treasury Yield), Volume
- 📉 Technical indicators: SMA 20/50, RSI 14, MACD, Bollinger Bands
- 📊 Rolling backtesting with trader sentiment analysis ("What if?")
- 🎯 Interactive data visualization with correlation analysis
- 💾 Data download/upload via Yahoo Finance or local CSV files

---

## 🏗️ Architecture

```
Streamlit UI (ofamp_app.py)
    ↓
TimesFM 2.5 — 200M params (CPU inference on Jetson)
    ↓
Covariates engine: VIX · DXY · TNX · Volume
    ↓
Correlation-based adjustment (±3% max)
```

**3 core files:**
| File | Description |
|------|-------------|
| `ofamp_app.py` | Main Streamlit app — UI, charts, orchestration |
| `ofamp_functions.py` | Pure functions — indicators, data prep, layout |
| `ofamp_guide.md` | In-app user guide |

---

## 🚀 Hardware & Requirements

| Component | Spec |
|-----------|------|
| **Board** | NVIDIA Jetson Orin Nano 8GB DDR5 |
| **JetPack** | 5.1.3 |
| **Runtime** | Docker + Streamlit |
| **Model** | TimesFM 2.5 (google/timesfm-2.5-200m-pytorch) |
| **Inference** | CPU (~2-4s per prediction) |

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/CPietraviva/OFAMP-OrinNano.git
cd OFAMP-OrinNano
```

### 2. Build and run with Docker
```bash
sudo docker compose up -d
```

### 3. Open the app
```
http://localhost:8501
```

---

## 📊 Optimal Operating Profile

Based on empirical testing:

| Parameter | Optimal Value |
|-----------|--------------|
| Historical data | 1 year |
| Context (memory) | 128 days |
| Forecast horizon | 30–40 days |

---

## ✅ Validated Results

| Asset | Data until | Horizon | Target AI | Real price | Error |
|-------|-----------|---------|-----------|------------|-------|
| NUCL.L | Mar 2026 | 37 days | 60.76 | 60.70 | **6 cents** |
| BTC-USD | Mar 2026 | 36 days | 75,480 | 74,783 | **0.9%** |

> Results on stable/lateral markets. Model is conservative on strong structural trends.

---

## 🎯 When to use OFAMP

**✅ Best for:**
- Markets in consolidation or lateral phase
- Assets in correction after a trend
- Uncertain market conditions where direction is unclear

**⚠️ Use with caution:**
- Strong structural trends (the model will underestimate)
- In obvious trends, trader intuition is already sufficient

---

## 🔮 Roadmap

- [ ] Cloud deployment on Google Cloud Run + Vertex AI (TimesFM API)
- [ ] Additional covariates (sentiment, additional macro indicators)
- [ ] Multi-ticker comparison

---

## 👤 Author

**Claudio Pietraviva** — IT Specialist & Project Manager  
🌐 [claudiopietraviva.ch](https://claudiopietraviva.ch)  
🐙 [github.com/CPietraviva](https://github.com/CPietraviva)

---

*Built with ❤️ on a Jetson Orin Nano — because powerful AI doesn't need a data center.*