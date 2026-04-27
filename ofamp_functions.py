"""
OFAMP — Oasis-Finance AI Multi-Predictor
ofamp_functions.py

Contiene tutte le funzioni pure di calcolo, download e preparazione dati.
Nessuna dipendenza da Streamlit — importabile e testabile indipendentemente.
"""

import numpy as np
import pandas as pd
import yfinance as yf


# ─────────────────────────────────────────────
# INDICATORI TECNICI
# ─────────────────────────────────────────────

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI — Relative Strength Index.
    Misura la velocità e l'ampiezza dei movimenti di prezzo.
    Oscilla tra 0 e 100. Valori > 70 = ipercomprato, < 30 = ipervenduto.
    """
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26,
              signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD — Moving Average Convergence Divergence.
    Misura il momentum confrontando EMA12 e EMA26.
    Restituisce: (macd_line, signal_line, histogram)
    - Istogramma positivo = momentum rialzista
    - Istogramma negativo = momentum ribassista
    """
    ema_fast    = series.ewm(span=fast, adjust=False).mean()
    ema_slow    = series.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def calc_bollinger(series: pd.Series, window: int = 20,
                   std_dev: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bande di Bollinger.
    Misurano la volatilità rispetto alla SMA20.
    Restituisce: (banda_superiore, sma, banda_inferiore)
    - Prezzo vicino alla banda superiore = potenzialmente caro
    - Prezzo vicino alla banda inferiore = potenzialmente a sconto
    - Bande strette = bassa volatilità, spesso precede movimenti forti
    """
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return sma + std_dev * std, sma, sma - std_dev * std




# ─────────────────────────────────────────────
# PREPARAZIONE DATI
# ─────────────────────────────────────────────

def prepara_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizza un DataFrame grezzo (da yfinance o CSV) in formato standard.
    Aggiunge tutti gli indicatori tecnici: SMA20, SMA50, RSI, MACD, Bollinger.
    """
    df = raw_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.rename(columns={df.columns[0]: 'ds'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    c_col = [c for c in df.columns if str(c).lower() == 'close'][0]
    df['Close'] = pd.to_numeric(df[c_col], errors='coerce')
    df = df.dropna(subset=['ds', 'Close']).sort_values('ds').reset_index(drop=True)

    df['SMA20']  = df['Close'].rolling(20).mean()
    df['SMA50']  = df['Close'].rolling(50).mean()
    df['RSI']    = calc_rsi(df['Close'])
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = calc_macd(df['Close'])
    df['BB_upper'], df['BB_mid'], df['BB_lower']   = calc_bollinger(df['Close'])
    return df


# ─────────────────────────────────────────────
# DOWNLOAD DATI
# ─────────────────────────────────────────────

def download_close(ticker: str, start, end,
                   index: pd.Index) -> tuple[pd.Series | None, str]:
    """
    Scarica la serie Close di un ticker da Yahoo Finance
    e la reindicizza sull'indice del ticker principale.
    Restituisce (series, status_message).
    """
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return None, "empty"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close_col = [c for c in df.columns if str(c).lower() == 'close']
        if not close_col:
            return None, "no close col"
        series = pd.to_numeric(df[close_col[0]], errors='coerce')
        series.index = pd.to_datetime(series.index)
        series = series.reindex(pd.to_datetime(index)).ffill().bfill()
        return series, "ok"
    except Exception as ex:
        return None, str(ex)


# ─────────────────────────────────────────────
# COVARIATES — LOGICA PREDITTIVA
# ─────────────────────────────────────────────

def prep_cov_zscore(series: pd.Series | None,
                    n_context: int,
                    n_horizon: int) -> list | None:
    """
    Trasforma una serie in z-score delle variazioni % giornaliere.

    Razionale finanziario:
    - Invece di passare valori assoluti (VIX=20.5, DXY=104.3) che cambiano
      regime nel tempo, il modello riceve deviazioni standardizzate:
      "oggi il VIX è +2.3σ rispetto alla norma del periodo"
    - Il futuro viene proiettato a 0.0 (nessuna anomalia attesa = neutro)
      che è l'assunzione più conservativa e non introduce bias direzionale

    Implementazione:
    1. Calcola variazioni % giornaliere: Δ% = (x[t] - x[t-1]) / |x[t-1]|
    2. Standardizza sulla finestra context: z = (Δ% - μ) / σ
    3. Concatena con horizon di zeri per il periodo futuro

    Parametro ridge (usato in forecast_with_covariates):
    - ridge=5.0 penalizza i coefficienti xreg, evitando che correlazioni
      storiche instabili vengano applicate rigidamente alla previsione
    """
    if series is None:
        return None
    vals = series.values[-n_context:]
    # Variazioni % giornaliere
    pct = np.diff(vals) / (np.abs(vals[:-1]) + 1e-10)
    # Z-score sulla finestra context
    mu, sigma = pct.mean(), pct.std() + 1e-10
    z = (pct - mu) / sigma
    # Pad iniziale a zero (primo giorno senza diff precedente)
    z = np.concatenate([[0.0], z])
    # Punto 3 — Mean reversion: il futuro decade esponenzialmente verso zero
    # z[-1] è l'ultima anomalia nota; decade con τ=10 giorni
    # Questo è più realistico del flat forward (VIX/DXY tendono a tornare alla media)
    last_z = z[-1]
    decay  = np.array([last_z * np.exp(-i / 10.0) for i in range(1, n_horizon + 1)])
    return [np.concatenate([z, decay]).tolist()]


def get_tv_layout(chart_height: int, **kwargs) -> dict:
    """
    Layout standard stile TradingView per i grafici Plotly.
    Sfondo #131722, griglia scura, scala prezzi a destra.
    """
    base = dict(
        template="plotly_dark",
        paper_bgcolor='#131722',
        plot_bgcolor='#131722',
        hovermode="x",
        height=chart_height,
        margin=dict(l=20, r=60, t=40, b=40),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(19,23,34,0.8)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(
            gridcolor='#1e222d', showgrid=True, zeroline=False,
            rangeslider=dict(visible=False),
            showspikes=True, spikecolor='#888', spikethickness=1,
            spikedash='dot', spikemode='across',
        ),
        yaxis=dict(
            gridcolor='#1e222d', showgrid=True, side='right', zeroline=False,
            showspikes=True, spikecolor='#888', spikethickness=1,
            spikedash='dot', spikemode='across',
        ),
    )
    base.update(kwargs)
    return base