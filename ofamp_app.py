"""
OFAMP — Oasis-Finance AI Multi-Predictor
ofamp_app.py

Entry point principale Streamlit.
Contiene layout, widget, grafici e logica di orchestrazione.
Le funzioni di calcolo sono in ofamp_functions.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import timesfm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import zipfile
import io
from datetime import date

from ofamp_functions import (
    calc_rsi, calc_macd, calc_bollinger,
    prepara_df, download_close,
    prep_cov_zscore, get_tv_layout
)

# ─────────────────────────────────────────────
# SETUP PAGINA
# ─────────────────────────────────────────────
st.set_page_config(page_title="OFAMP v1.0 — Oasis-Finance AI Multi-Predictor", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1rem !important; }
    section[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; margin-top: 0 !important; }
    header[data-testid="stHeader"] { height: 2.5rem; background: transparent; }
    .stTabs [data-baseweb="tab-list"] { gap: 16px; }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 12px 28px;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e222d;
        border-bottom: 3px solid #2962FF;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.header("🛠️ Configurazione")


context_days = st.sidebar.select_slider(
    "📅 Giorni storici da analizzare:",
    options=[64, 96, 128, 160, 192, 256],
    value=128,
    help="Quanti giorni di prezzi passati vede il modello AI prima di fare la previsione. "
         "128 giorni = circa 6 mesi. Più giorni = più storia, ma non sempre più precisione."
)
if context_days <= 96:
    st.sidebar.info(f"⚡ {context_days}gg — reattivo al trend recente")
elif context_days <= 160:
    st.sidebar.success(f"✅ {context_days}gg — zona ottimale")
elif context_days <= 256:
    st.sidebar.warning(f"⚠️ {context_days}gg — ciclo lungo, usa con cautela")

forecast_days = st.sidebar.slider(
    "🔮 Giorni da prevedere:",
    min_value=7, max_value=60, value=30,
    help="Quanti giorni futuri vuoi che il modello preveda. "
         "30 giorni = previsione del prossimo mese. Oltre i 45 giorni l'incertezza cresce molto."
)
chart_height = st.sidebar.slider(
    "📐 Altezza grafici:",
    min_value=300, max_value=1200, value=600, step=50,
    help="Altezza in pixel dei grafici. Aumenta se vuoi più dettaglio visivo."
)

sentiment = 0  # sentiment gestito solo nel tab Backtesting


st.title("📈 Oasis-Finance AI Multi-Predictor  v1.0")
st.markdown("<p style='color:#888; font-size:0.85rem; margin-top:-12px;'>by Claudio Pietraviva</p>",
            unsafe_allow_html=True)


# ─── DISCLAIMER — fondo sidebar ───
st.sidebar.divider()
with st.sidebar.expander("⚠️ Disclaimer", expanded=True):
    st.caption(
        "Questo strumento è un **esperimento a scopo educativo**. "
        "Le previsioni non costituiscono consulenza finanziaria "
        "né raccomandazioni di investimento. Qualsiasi decisione "
        "finanziaria è di esclusiva responsabilità dell'utente."
    )

tab1, tab1b, tab2, tab3, tab4 = st.tabs(["📥  Dati", "📊  Dati Disponibili", "📈  Analisi & Previsione", "📊  Backtesting", "📖  Guida & Indicatori"])

# ═══════════════════════════════════════════════
# TAB 1 — DOWNLOAD DATI
# ═══════════════════════════════════════════════
with tab1:
    st.subheader("📥 Scarica dati da Yahoo Finance")

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        ticker_input = st.text_input("Ticker Yahoo Finance:", value="",
                                      placeholder="es. AAPL, NUCL.L, BTC-USD")

    with col_b:
        date_start = st.date_input("Data inizio:", value=date.today().replace(year=date.today().year - 1))
    with col_c:
        date_end = st.date_input("Data fine:", value=date.today())

    scarica = st.button("⬇️ Scarica dati")
    yahoo_feedback = st.empty()  # placeholder per notifica + bottoni Yahoo

    # ─── CARICA DA FILE LOCALI ───
    st.divider()
    st.subheader("📂 Oppure carica da file CSV locali")
    st.caption("Utile per riusare dati già scaricati senza fare chiamate a Yahoo Finance.")

    up_col1, up_col2, up_col3, up_col4 = st.columns(4)
    with up_col1:
        f_ticker = st.file_uploader("📈 Ticker (price CSV)", type=["csv"], key="up_ticker")
    with up_col2:
        f_vix = st.file_uploader("😨 VIX (VIX.csv)", type=["csv"], key="up_vix")
    with up_col3:
        f_dxy = st.file_uploader("💵 DXY (DXY.csv)", type=["csv"], key="up_dxy")
    with up_col4:
        f_tnx = st.file_uploader("📊 TNX (TNX.csv)", type=["csv"], key="up_tnx")

    if f_ticker is not None:
        if st.button("📂 Carica dati da file", use_container_width=False):
            try:
                # Ticker principale
                raw = pd.read_csv(f_ticker, index_col=0, parse_dates=True)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                ticker_name = f_ticker.name.replace("_historical_data.csv","").replace("_price.csv","").upper()

                # VIX
                if f_vix is not None:
                    vix_df = pd.read_csv(f_vix, index_col=0, parse_dates=True)
                    vix_series = pd.to_numeric(vix_df.iloc[:, 0], errors='coerce')
                    vix_series = vix_series.reindex(pd.to_datetime(raw.index)).ffill().bfill()
                    vix_msg = "✅"
                else:
                    vix_series = None
                    vix_msg = "⚠️ non caricato"

                # DXY
                if f_dxy is not None:
                    dxy_df = pd.read_csv(f_dxy, index_col=0, parse_dates=True)
                    dxy_series = pd.to_numeric(dxy_df.iloc[:, 0], errors='coerce')
                    dxy_series = dxy_series.reindex(pd.to_datetime(raw.index)).ffill().bfill()
                    dxy_msg = "✅"
                else:
                    dxy_series = None
                    dxy_msg = "⚠️ non caricato"

                # TNX
                if f_tnx is not None:
                    tnx_df = pd.read_csv(f_tnx, index_col=0, parse_dates=True)
                    tnx_series = pd.to_numeric(tnx_df.iloc[:, 0], errors='coerce')
                    tnx_series = tnx_series.reindex(pd.to_datetime(raw.index)).ffill().bfill()
                    tnx_msg = "✅"
                else:
                    tnx_series = None
                    tnx_msg = "⚠️ non caricato"

                # Volume — sempre dal CSV ticker (colonna Volume standard Yahoo Finance)
                vol_col = [c for c in raw.columns if str(c).lower() == 'volume']
                if vol_col:
                    vol_series = np.log1p(pd.to_numeric(raw[vol_col[0]], errors='coerce').ffill().bfill())
                    vol_msg = "✅"
                else:
                    vol_series = None
                    vol_msg = "⚠️ non trovato"


                st.session_state['raw_download'] = raw
                st.session_state['ticker_live']  = ticker_name
                st.session_state['valuta']        = 'USD'
                st.session_state['res_df']        = None
                st.session_state['vix_series']    = vix_series
                st.session_state['dxy_series']    = dxy_series
                st.session_state['vol_series']    = vol_series
                st.session_state['tnx_series']    = tnx_series
                st.session_state['download_source'] = 'csv'
                st.success(f"✅ Caricato: **{ticker_name}** ({len(raw)} giorni) | "
                           f"VIX {vix_msg} | DXY {dxy_msg} | Volume {vol_msg} | TNX {tnx_msg}")

            except Exception as e:
                st.error(f"Errore caricamento: {e}")
                st.exception(e)



    if scarica:
        with st.spinner(f"Download {ticker_input} da Yahoo Finance..."):
            try:
                raw = yf.download(ticker_input, start=date_start, end=date_end, progress=False)
                if raw.empty:
                    st.error("Nessun dato trovato. Controlla il ticker.")
                else:
                    # Rileva valuta automaticamente
                    try:
                        info   = yf.Ticker(ticker_input).info
                        valuta = info.get('currency', 'USD')
                    except:
                        valuta = 'USD'

                    # Scarica covariates: VIX, DXY, Volume
                    cov_status = []

                    vix_series, vix_msg = download_close("^VIX", date_start, date_end, raw.index)
                    cov_status.append(f"VIX ✅" if vix_series is not None else f"VIX ❌ ({vix_msg})")

                    dxy_series, dxy_msg = download_close("DX-Y.NYB", date_start, date_end, raw.index)
                    cov_status.append(f"DXY ✅" if dxy_series is not None else f"DXY ❌ ({dxy_msg})")

                    tnx_series, tnx_msg = download_close("^TNX", date_start, date_end, raw.index)
                    cov_status.append(f"TNX ✅" if tnx_series is not None else f"TNX ❌ ({tnx_msg})")

                    # Volume dal CSV principale
                    try:
                        raw_flat = raw.copy()
                        if isinstance(raw_flat.columns, pd.MultiIndex):
                            raw_flat.columns = raw_flat.columns.get_level_values(0)
                        vol_col = [c for c in raw_flat.columns if str(c).lower() == 'volume']
                        if vol_col:
                            vol_series = pd.to_numeric(raw_flat[vol_col[0]], errors='coerce').ffill().bfill()
                            vol_series = np.log1p(vol_series)
                            cov_status.append("Volume ✅")
                        else:
                            vol_series = None
                            cov_status.append("Volume ❌ (col not found)")
                    except Exception as ex:
                        vol_series = None
                        cov_status.append(f"Volume ❌ ({ex})")

                    st.session_state['raw_download'] = raw
                    st.session_state['ticker_live']  = ticker_input.upper()
                    st.session_state['valuta']       = valuta
                    st.session_state['res_df']       = None
                    st.session_state['vix_series']   = vix_series
                    st.session_state['dxy_series']   = dxy_series
                    st.session_state['vol_series']   = vol_series
                    st.session_state['tnx_series']   = tnx_series
                    st.session_state['download_source'] = 'yahoo'
                    with yahoo_feedback.container():
                        st.success(f"✅ {len(raw)} giorni scaricati per "
                                   f"{ticker_input.upper()} — Valuta: {valuta} | "
                                   f"Covariates: {' | '.join(cov_status)}")
            except Exception as e:
                st.error(f"Errore download: {e}")
                st.exception(e)

    # Bottoni download Yahoo — persistenti nel placeholder
    if (st.session_state.get('download_source') == 'yahoo' and
            'raw_download' in st.session_state and
            st.session_state['raw_download'] is not None):
        raw_dl   = st.session_state['raw_download']
        ticker_l = st.session_state['ticker_live']
        valuta_l = st.session_state.get('valuta', 'USD')
        display  = raw_dl.copy()
        if isinstance(display.columns, pd.MultiIndex):
            display.columns = display.columns.get_level_values(0)
        vix_s = st.session_state.get('vix_series')
        dxy_s = st.session_state.get('dxy_series')
        tnx_s = st.session_state.get('tnx_series')
        cov_ok = ' | '.join(filter(None, [
            'VIX ✅' if vix_s is not None else None,
            'DXY ✅' if dxy_s is not None else None,
            'TNX ✅' if tnx_s is not None else None,
            'Volume ✅' if st.session_state.get('vol_series') is not None else None,
        ]))
        with yahoo_feedback.container():
            st.success(f"✅ **{len(display)} giorni** scaricati per **{ticker_l}** — "
                       f"Valuta: {valuta_l} | Covariates: {cov_ok}")
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                st.download_button("💾 Salva CSV ticker",
                    data=display.to_csv().encode('utf-8'),
                    file_name=f"{ticker_l}_historical_data.csv",
                    mime="text/csv", use_container_width=True)
            with btn_col2:
                if vix_s is not None or dxy_s is not None or tnx_s is not None:
                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr(f"{ticker_l}_price.csv", display.to_csv())
                        if vix_s is not None:
                            zf.writestr("VIX.csv", vix_s.rename("VIX_Close").to_frame().to_csv())
                        if dxy_s is not None:
                            zf.writestr("DXY.csv", dxy_s.rename("DXY_Close").to_frame().to_csv())
                        if tnx_s is not None:
                            zf.writestr("TNX.csv", tnx_s.rename("TNX_Close").to_frame().to_csv())
                    zip_buf.seek(0)
                    st.download_button("📦 Salva ZIP completo (+ covariates)",
                        data=zip_buf.getvalue(),
                        file_name=f"{ticker_l}_dataset_completo.zip",
                        mime="application/zip", use_container_width=True)



# ═══════════════════════════════════════════════
# TAB 1B — DATI DISPONIBILI
# ═══════════════════════════════════════════════
with tab1b:
    st.subheader("📊 Dati Disponibili")
    st.caption("Visualizza e confronta le serie scaricate. La correlazione mostra la relazione con il ticker principale.")

    if 'raw_download' not in st.session_state or st.session_state['raw_download'] is None:
        st.warning("⬅️ Vai al tab **📥 Dati** e scarica prima i dati di un ticker.")
    else:
        raw_v    = st.session_state['raw_download']
        ticker_v = st.session_state['ticker_live']
        vix_v    = st.session_state.get('vix_series')
        dxy_v    = st.session_state.get('dxy_series')
        tnx_v    = st.session_state.get('tnx_series')
        vol_v    = st.session_state.get('vol_series')

        raw_flat = raw_v.copy()
        if isinstance(raw_flat.columns, pd.MultiIndex):
            raw_flat.columns = raw_flat.columns.get_level_values(0)
        raw_flat = raw_flat.reset_index()
        raw_flat.rename(columns={raw_flat.columns[0]: 'ds'}, inplace=True)
        raw_flat['ds'] = pd.to_datetime(raw_flat['ds'])
        price_series = raw_flat.set_index('ds')['Close']
        price_ret    = price_series.pct_change().dropna()

        # Volume raw (non log) per visualizzazione a barre — riusa raw_flat già preparato
        vol_col_list = [c for c in raw_flat.columns if str(c).lower() == 'volume']
        vol_raw = None
        if vol_col_list:
            vol_raw = pd.to_numeric(raw_flat[vol_col_list[0]], errors='coerce').ffill().bfill()
            vol_raw.index = raw_flat['ds']

        available = {}
        available[ticker_v] = {'series': price_series, 'color': '#2962FF', 'label': f'{ticker_v} — Prezzo chiusura', 'type': 'line'}
        if vol_raw is not None:
            available['Volume'] = {'series': vol_raw, 'color': None, 'label': f'{ticker_v} — Volume scambi', 'type': 'bar'}
        if vix_v is not None:
            available['VIX'] = {'series': vix_v, 'color': '#F7525F', 'label': 'VIX — Indice volatilità', 'type': 'line'}
        if dxy_v is not None:
            available['DXY'] = {'series': dxy_v, 'color': '#F6C85F', 'label': 'DXY — Indice dollaro USA', 'type': 'line'}
        if tnx_v is not None:
            available['TNX'] = {'series': tnx_v, 'color': '#A855F7', 'label': 'TNX — Treasury 10Y (%)', 'type': 'line'}

        selected = st.multiselect(
            "Seleziona serie da visualizzare:",
            options=list(available.keys()),
            default=list(available.keys())
        )

        if selected:
            for name in selected:
                meta   = available[name]
                series = meta['series'].copy()
                series.index = pd.to_datetime(series.index).normalize()
                series = series.dropna()

                # Correlazione con ticker principale (se non è il ticker stesso)
                corr_text = ""
                yahoo_ticker = {'VIX': '^VIX', 'DXY': 'DX-Y.NYB', 'TNX': '^TNX', 'Volume': None}
                if name != ticker_v:
                    s_ret = series.pct_change().dropna()
                    s_ret.index = pd.to_datetime(s_ret.index).normalize()
                    common = price_ret.align(s_ret, join='inner')
                    r = float(common[0].corr(common[1]))
                    if not np.isnan(r):
                        direction = "negativa" if r < -0.1 else ("positiva" if r > 0.1 else "neutrale")
                        strength  = "forte" if abs(r) > 0.4 else ("moderata" if abs(r) > 0.2 else "debole")
                        corr_text = f" | r={r:+.3f} con {ticker_v} ({strength}, {direction})"

                # Grafico individuale — scala reale, asse Y agganciato ai dati
                fig_v = go.Figure()
                chart_type = meta.get('type', 'line')

                if chart_type == 'bar':
                    # Volume — barre verdi/rosse basate su variazione prezzo
                    price_chg = price_series.reindex(series.index).diff()
                    bar_colors = ['#26a69a' if (price_chg.get(d, 0) or 0) >= 0
                                  else '#ef5350' for d in series.index]
                    fig_v.add_trace(go.Bar(
                        x=series.index, y=series.values,
                        name='Volume', marker_color=bar_colors, opacity=0.8,
                    ))
                    fig_v.update_layout(**get_tv_layout(260,
                        title=f"{meta['label']}",
                        yaxis=dict(gridcolor='#1e222d', showgrid=True, side='right',
                                   zeroline=False, tickformat='.3s')
                    ))
                elif name == ticker_v:
                    # Ticker principale — sempre linea nel tab dati disponibili
                    fig_v.add_trace(go.Scatter(
                        x=series.index, y=series.values,
                        name=name, line=dict(color=meta['color'], width=2),
                    ))
                    fig_v.update_layout(**get_tv_layout(260,
                        title=f"{meta['label']}",
                    ))
                else:
                    # Covariates — sempre linea
                    y_min = float(series.min()) * 0.98
                    y_max = float(series.max()) * 1.02
                    fig_v.add_trace(go.Scatter(
                        x=series.index, y=series.values,
                        name=name, line=dict(color=meta['color'], width=2),
                    ))
                    fig_v.update_layout(**get_tv_layout(260,
                        title=f"{meta['label']}{corr_text}",
                        yaxis=dict(gridcolor='#1e222d', showgrid=True, side='right',
                                   zeroline=False, range=[y_min, y_max])
                    ))
                st.plotly_chart(fig_v, use_container_width=True)

                # Link Yahoo Finance sotto il grafico
                yt = yahoo_ticker.get(name) or (ticker_v if name == ticker_v else None)
                if yt:
                    st.caption(f"🔗 [Apri {name} su Yahoo Finance](https://finance.yahoo.com/quote/{yt}/)")
        else:
            st.info("Seleziona almeno una serie dal menu.")

# ═══════════════════════════════════════════════
# TAB 3 — BACKTESTING
# ═══════════════════════════════════════════════
with tab3:
    st.subheader("📊 Backtesting Rolling")
    st.caption("Verifica l'accuratezza del modello sui dati storici già caricati. Il sentiment del trader viene applicato a tutte le finestre — utile per verificare se il tuo giudizio storico avrebbe migliorato la precisione.")

    if 'raw_download' not in st.session_state or st.session_state['raw_download'] is None:
        st.warning("⬅️ Vai al tab **📥 Dati** e scarica prima i dati di un ticker.")
    else:
        df_bt      = prepara_df(st.session_state['raw_download'])
        ticker_bt  = st.session_state['ticker_live']
        valuta_bt  = st.session_state.get('valuta', 'USD')

        st.info(f"📡 Dati disponibili: **{ticker_bt}** — {len(df_bt)} giorni")

        bt_col1, bt_col2, bt_col3 = st.columns(3)
        with bt_col1:
            bt_context = st.selectbox(
                "📅 Giorni storici (Context):",
                [64, 96, 128, 160, 192, 256], index=2,
                help="Quanti giorni di storia usa il modello per ogni previsione di test. "
                     "Deve corrispondere al Context che usi nell'analisi reale."
            )
        with bt_col2:
            bt_horizon = st.slider(
                "🔮 Giorni da prevedere (Horizon):",
                7, 60, 30,
                help="Quanto avanti prevede il modello in ogni finestra di test. "
                     "Es: 30 = il modello prevede i successivi 30 giorni, poi confronta con i reali."
            )
        with bt_col3:
            bt_step = st.slider(
                "⏭️ Passo tra test (Step):",
                5, 30, 14,
                help="Ogni quanti giorni avanza la finestra di test. "
                     "Step=14 = un test ogni 2 settimane. Più basso = più test ma più lento."
            )

        # Calcola quante finestre disponibili
        n_windows = max(0, (len(df_bt) - bt_context - bt_horizon) // bt_step)
        st.caption(
            f"🧮 Con questi parametri: **{n_windows} test** disponibili | "
            f"Tempo stimato: ~{n_windows * 2} secondi"
        )

        # Sentiment slider — solo qui per analisi "e se?"
        st.divider()
        st.markdown("**🧠 Sentiment del Trader — Analisi 'E se?'**")
        st.caption("Simula come cambierebbero i risultati se avessi applicato il tuo giudizio di mercato. "
                   "Ogni punto = ±1% sulla previsione. Confronta i risultati con sentiment=0 (modello puro).")
        bt_sentiment = st.slider(
            "Correzione sentiment:",
            min_value=-10, max_value=10, value=0, step=1,
            key="bt_sentiment",
            help="0 = modello puro | +5 = avresti corretto del +5% (eri rialzista) | -3 = eri ribassista"
        )
        if bt_sentiment == 0:
            st.info("➡️ Sentiment neutro — risultati del modello puro")
        elif 0 < bt_sentiment <= 5:
            st.success(f"📈 Sentiment rialzista +{bt_sentiment}% applicato a ogni finestra")
        elif bt_sentiment > 5:
            st.warning(f"📈 Correzione forte +{bt_sentiment}% — verifica se migliora davvero")
        elif -5 <= bt_sentiment < 0:
            st.error(f"📉 Sentiment ribassista {bt_sentiment}% applicato a ogni finestra")
        else:
            st.warning(f"📉 Correzione forte {bt_sentiment}% — verifica se migliora davvero")

        if n_windows < 3:
            st.error("Dati insufficienti per il backtesting. Scarica almeno 1 anno di dati.")
        elif st.button("▶️ Esegui Backtesting", type="primary"):
            results = []
            progress = st.progress(0, text="Inizializzazione modello...")

            try:
                # Carica modello una volta sola
                bt_model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                    "google/timesfm-2.5-200m-pytorch"
                )
                bt_model.compile(
                    timesfm.ForecastConfig(
                        max_context=bt_context,
                        max_horizon=bt_horizon,
                        normalize_inputs=True,
                        use_continuous_quantile_head=True,
                        fix_quantile_crossing=True,
                        return_backcast=True,
                    )
                )

                prices_all = df_bt['Close'].values

                for i in range(n_windows):
                    start_idx  = i * bt_step
                    end_idx    = start_idx + bt_context
                    future_idx = end_idx + bt_horizon

                    if future_idx > len(prices_all):
                        break

                    ctx_prices = prices_all[start_idx:end_idx].tolist()
                    real_fut   = prices_all[end_idx:future_idx]
                    last_real  = prices_all[end_idx - 1]

                    pt, qt = bt_model.forecast(horizon=bt_horizon, inputs=[ctx_prices])
                    p50_bt = np.array(pt)[0][-bt_horizon:]
                    q_bt   = np.array(qt)[0][-bt_horizon:]
                    p10_bt = q_bt[:, 0]
                    p90_bt = q_bt[:, -1]

                    # Applica sentiment del trader se impostato
                    # Permette di verificare: "con il mio sentiment di allora, migliorava?"
                    if bt_sentiment != 0:
                        sent_adj = bt_sentiment / 100.0
                        p50_bt = p50_bt * (1 + sent_adj)
                        p10_bt = p10_bt * (1 + sent_adj)
                        p90_bt = p90_bt * (1 + sent_adj)

                    # Metriche per questa finestra
                    last_date   = df_bt['ds'].iloc[end_idx - 1]
                    target_pred = p50_bt[-1]
                    target_real = real_fut[-1]
                    err_pct     = (target_pred - target_real) / target_real * 100
                    direction_ok = ((target_pred > last_real) == (target_real > last_real))
                    # % prezzi reali dentro la banda
                    inside = np.mean((real_fut >= p10_bt) & (real_fut <= p90_bt)) * 100

                    results.append({
                        'Data':          last_date.strftime('%d.%m.%Y'),
                        'Ultimo reale':  round(last_real, 2),
                        'Pred. finale':  round(target_pred, 2),
                        'Reale finale':  round(target_real, 2),
                        'Errore %':      round(err_pct, 2),
                        'Dir. corretta': '✅' if direction_ok else '❌',
                        '% in banda':    round(inside, 1),
                    })

                    progress.progress((i + 1) / n_windows,
                                      text=f"Finestra {i+1}/{n_windows}...")

                progress.empty()
                st.session_state['bt_results'] = results
                st.session_state['bt_ticker']  = ticker_bt

            except Exception as e:
                st.error(f"Errore backtesting: {e}")
                st.exception(e)

        # Mostra risultati
        if st.session_state.get('bt_results') and st.session_state.get('bt_ticker') == ticker_bt:
            results = st.session_state['bt_results']
            df_res  = pd.DataFrame(results)

            mae      = df_res['Errore %'].abs().mean()
            dir_ok   = (df_res['Dir. corretta'] == '✅').mean() * 100
            banda_ok = df_res['% in banda'].mean()

            st.markdown("### 📈 Risultati Aggregati")
            m1, m2, m3 = st.columns(3)
            m1.metric("Errore medio assoluto", f"{mae:.2f}%",
                      help="Media degli errori % sulla previsione puntuale finale")
            m2.metric("Direzione corretta", f"{dir_ok:.0f}%",
                      help="% finestre in cui il modello ha previsto correttamente su/giù")
            m3.metric("Prezzi in banda", f"{banda_ok:.0f}%",
                      help="% giorni reali caduti dentro la Volatilità Attesa")

            # Grafico errori — sopra
            fig_bt = go.Figure()
            def errore_color(v):
                av = abs(v)
                if av < 5:    return '#26a69a'   # verde — errore piccolo ✅
                elif av < 15: return '#F6C85F'   # giallo — errore medio ⚠️
                else:         return '#ef5350'   # rosso — errore grande ❌

            fig_bt.add_trace(go.Bar(
                x=df_res['Data'], y=df_res['Errore %'],
                marker_color=[errore_color(v) for v in df_res['Errore %']],
                name='Errore %',
                text=[f"{v:+.1f}%" for v in df_res['Errore %']],
                textposition='outside',
            ))
            fig_bt.add_hline(y=0, line_color='white', line_dash='dash', opacity=0.3)
            fig_bt.update_layout(
                template="plotly_dark", paper_bgcolor='#131722', plot_bgcolor='#131722',
                height=400, title=f"{ticker_bt} — Errore % per finestra (verde=sovrastima, rosso=sottostima)",
                margin=dict(l=20, r=20, t=40, b=80),
                yaxis=dict(gridcolor='#1e222d', zeroline=False),
                xaxis=dict(gridcolor='#1e222d', tickangle=-45),
            )
            st.plotly_chart(fig_bt, use_container_width=True)

            # Tabella dettaglio — sotto
            st.markdown("### 📋 Dettaglio Finestre")
            st.dataframe(df_res, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 4 — GUIDA & INDICATORI
# ═══════════════════════════════════════════════
with tab4:
    st.subheader("📖 Guida all'app e agli Indicatori")
    try:
        with open('/app/ofamp_guide.md', 'r', encoding='utf-8') as f:
            st.markdown(f.read())
    except FileNotFoundError:
        st.error("File guida.md non trovato. Assicurati che sia nella cartella /app.")

# ═══════════════════════════════════════════════
# TAB 2 — ANALISI & PREVISIONE
# ═══════════════════════════════════════════════
with tab2:
    st.subheader("📈 Analisi & Previsione")

    if 'raw_download' not in st.session_state or st.session_state['raw_download'] is None:
        st.warning("⬅️ Vai al tab **📥 Dati** e scarica prima i dati di un ticker.")
    else:
        df_clean    = prepara_df(st.session_state['raw_download'])
        ticker_hint = st.session_state['ticker_live']
        valuta      = st.session_state.get('valuta', 'USD')
    
        st.info(f"📡 Dati: **{ticker_hint}** | {len(df_clean)} giorni | Valuta: **{valuta}**")
    
        last_p      = df_clean['Close'].iloc[-1]
        last_d      = df_clean['ds'].iloc[-1]
        last_rsi    = df_clean['RSI'].iloc[-1]
        last_macd_h = df_clean['MACD_hist'].iloc[-1]
        last_bb_u   = df_clean['BB_upper'].iloc[-1]
        last_bb_l   = df_clean['BB_lower'].iloc[-1]
    
        # ─── METRICS ───
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Ultima Chiusura ({last_d.strftime('%d/%m')})", f"{last_p:.2f} {valuta}")

        # Prezzo live — caricamento automatico silenzioso
        if st.session_state.get('live_price_ticker') != ticker_hint:
            try:
                fi = yf.Ticker(ticker_hint).fast_info
                live_price = float(fi['last_price'])
                st.session_state['live_price'] = live_price
                st.session_state['live_price_ticker'] = ticker_hint
            except:
                live_price = None
                st.session_state['live_price'] = None
        else:
            live_price = st.session_state.get('live_price')

        with c2:
            if live_price:
                diff_live = ((live_price - last_p) / last_p) * 100
                st.metric("Prezzo Attuale", f"{live_price:.2f} {valuta}", delta=f"{diff_live:+.2f}%")

        rsi_icon = "🔴" if last_rsi > 70 else ("🟢" if last_rsi < 30 else "🟡")
        c3.metric("RSI — Relative Strength Index (14gg)", f"{rsi_icon} {last_rsi:.1f}")
        c4.metric("MACD — Moving Avg Convergence Divergence", "▲ Rialzista" if last_macd_h > 0 else "▼ Ribassista")
    
        # ─── GRAFICO STORICO ───
        st.subheader("📊 Analisi Storica")
        fig_h = go.Figure()
        # Filtra NaN — BB e SMA richiedono N giorni prima di avere valori
        df_bb = df_clean[df_clean['BB_upper'].notna()]
        fig_h.add_trace(go.Scatter(x=df_bb['ds'], y=df_bb['BB_upper'],
            line=dict(color='rgba(255,165,0,0.3)', dash='dot'), name='BB Upper', hoverinfo='skip'))
        fig_h.add_trace(go.Scatter(x=df_bb['ds'], y=df_bb['BB_lower'],
            fill='tonexty', fillcolor='rgba(255,165,0,0.05)',
            line=dict(color='rgba(255,165,0,0.3)', dash='dot'), name='BB Lower', hoverinfo='skip'))
        fig_h.add_trace(go.Scatter(x=df_clean['ds'], y=df_clean['Close'],
            name='Prezzo', line=dict(color='#2962FF', width=2)))
        df_sma50  = df_clean[df_clean['SMA50'].notna()]
        fig_h.add_trace(go.Scatter(x=df_sma50['ds'], y=df_sma50['SMA50'],
            name='SMA 50', line=dict(color='#F6C85F', width=1)))
        df_sma20 = df_clean[df_clean['SMA20'].notna()]
        fig_h.add_trace(go.Scatter(x=df_sma20['ds'], y=df_sma20['SMA20'],
            name='SMA 20', line=dict(color='#F7525F', width=1)))
        fig_h.update_layout(**get_tv_layout(chart_height,
            title=f"{ticker_hint} — Storico con Bande di Bollinger"))
        st.plotly_chart(fig_h, use_container_width=True)
    
        # ─── RSI + MACD ───
        with st.expander("📉 Indicatori Tecnici (RSI & MACD — clicca per espandere):"):
            fig_ind = make_subplots(rows=2, cols=1, shared_xaxes=True,
                subplot_titles=("RSI — Relative Strength Index (14gg)", "MACD — Moving Average Convergence Divergence"), vertical_spacing=0.12)
            fig_ind.add_trace(go.Scatter(x=df_clean['ds'], y=df_clean['RSI'],
                name='RSI', line=dict(color='#00CF8D')), row=1, col=1)
            fig_ind.add_hline(y=70, line_dash="dash", line_color="red",   row=1, col=1)
            fig_ind.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
            colors_hist = ['#00CF8D' if v >= 0 else '#FF4B4B' for v in df_clean['MACD_hist']]
            fig_ind.add_trace(go.Bar(x=df_clean['ds'], y=df_clean['MACD_hist'],
                name='Istogramma', marker_color=colors_hist), row=2, col=1)
            fig_ind.add_trace(go.Scatter(x=df_clean['ds'], y=df_clean['MACD'],
                name='MACD', line=dict(color='#00BFFF')), row=2, col=1)
            fig_ind.add_trace(go.Scatter(x=df_clean['ds'], y=df_clean['MACD_signal'],
                name='Signal', line=dict(color='orange')), row=2, col=1)
            fig_ind.update_layout(
                template="plotly_dark", paper_bgcolor='#131722', plot_bgcolor='#131722',
                hovermode="x unified", height=chart_height,
                margin=dict(l=20, r=60, t=40, b=40),
                xaxis=dict(gridcolor='#1e222d'), yaxis=dict(gridcolor='#1e222d'),
                xaxis2=dict(gridcolor='#1e222d'), yaxis2=dict(gridcolor='#1e222d'),
            )
            st.plotly_chart(fig_ind, use_container_width=True)
    
        # ─── ANALISI AI ───
        if st.button("🚀 Esegui Analisi AI"):
            with st.spinner(f"TimesFM: calcolo {forecast_days} giorni..."):
                try:
                    # ═══════════════════════════════════════════════════════
                    # ❤️  CUORE DEL SISTEMA — MOTORE AI TIMESFM 2.5
                    #
                    # Flusso:
                    #   1. Carica modello TimesFM 2.5 (200M parametri, CPU)
                    #   2. Prepara input: ultimi context_days prezzi reali
                    #   3. Prepara covariates: VIX, DXY, TNX, Volume (z-score + mean reversion)
                    #   4. model.forecast() → p50 (mediana previsione)
                    #   5. Quantili nativi TimesFM (p10/p90) — non modificati
                    #   6. Aggiustamento covariates ±3% basato su correlazioni reali
                    #   7. Salva risultati in session_state → visualizzazione
                    # ═══════════════════════════════════════════════════════
                    # ── TimesFM 2.5 API ──
                    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                        "google/timesfm-2.5-200m-pytorch"
                    )
                    model.compile(
                        timesfm.ForecastConfig(
                            max_context=context_days,
                            max_horizon=forecast_days,
                            normalize_inputs=True,
                            use_continuous_quantile_head=True,
                            fix_quantile_crossing=True,
                            return_backcast=True,
                        )
                    )
                    prices_input = df_clean['Close'].values[-context_days:].tolist()

                    # Prepara covariates — context + horizon (flat forward per il futuro)
                    vix_series = st.session_state.get('vix_series')
                    dxy_series = st.session_state.get('dxy_series')
                    vol_series = st.session_state.get('vol_series')
                    tnx_series = st.session_state.get('tnx_series')

                    # Verifica covariates disponibili per aggiustamento
                    cov_available = {n: s for n, s in [
                        ("VIX", vix_series), ("DXY", dxy_series),
                        ("Volume", vol_series), ("TNX", tnx_series)
                    ] if s is not None}

                    # ── PREVISIONE BASE TimesFM — quantili nativi ──
                    # Usiamo direttamente p10/p90 del modello: sono calibrati
                    # internamente su miliardi di serie temporali. Non li sovrascriviamo.
                    point_base, quantile_base = model.forecast(
                        horizon=forecast_days,
                        inputs=[prices_input],
                    )
                    p50  = np.array(point_base)[0][-forecast_days:]
                    q_all = np.array(quantile_base)[0][-forecast_days:]  # (horizon, 10)
                    p10  = q_all[:, 0]   # quantile 0.1 nativo TimesFM
                    p90  = q_all[:, -1]  # quantile 0.9 nativo TimesFM

                    # ctx_high/ctx_low servono solo per le linee Massimo/Minimo
                    ctx_prices = np.array(prices_input)
                    ctx_high   = float(ctx_prices.max())
                    ctx_low    = float(ctx_prices.min())

                    if cov_available:
                        # ── AGGIUSTAMENTO COVARIATES ──
                        # Correlazione storica + z-score recente con mean reversion
                        close_arr = np.array(prices_input)
                        ret_arr   = np.diff(close_arr) / (np.abs(close_arr[:-1]) + 1e-10)
                        price_ret = pd.Series(ret_arr)
                        cov_signal = 0.0
                        cov_info = []

                        for name, series in [("VIX", vix_series), ("DXY", dxy_series), ("Volume", vol_series), ("TNX", tnx_series)]:
                            if series is None:
                                continue
                            cov_vals = series.values[-context_days:]
                            cov_ret  = np.diff(cov_vals) / (np.abs(cov_vals[:-1]) + 1e-10)

                            # Correlazione reale su array numpy
                            min_len = min(len(price_ret), len(cov_ret))
                            corr = float(np.corrcoef(
                                np.array(price_ret)[-min_len:],
                                cov_ret[-min_len:]
                            )[0, 1])
                            if np.isnan(corr): corr = 0.0

                            # Mean reversion: le covariates decadono verso la loro media storica
                            mu_cov   = float(np.mean(cov_ret))
                            std_cov  = float(np.std(cov_ret)) + 1e-10
                            recent_5 = cov_ret[-5:]
                            # Z-score recente rispetto alla media storica
                            z_recent = float(np.mean((recent_5 - mu_cov) / std_cov))
                            # Mean reversion: il segnale decade verso zero nel tempo
                            # (dopo 10 giorni metà dell'anomalia rientra)
                            decay = np.exp(-np.arange(forecast_days) / 10.0)
                            contrib = corr * z_recent * float(np.mean(decay))
                            cov_signal += contrib
                            cov_info.append(f"{name}(r={corr:+.2f},z={z_recent:+.1f})")

                        # Aggiustamento conservativo ±3%
                        # Trasla rigidamente p50, p10, p90 dello stesso fattore
                        # preservando la forma e asimmetria nativa di TimesFM
                        adj = np.clip(cov_signal * 0.01, -0.03, 0.03)
                        p50 = p50 * (1 + adj)
                        p10 = p10 * (1 + adj)
                        p90 = p90 * (1 + adj)
                        st.caption(f"✅ Covariates: {' | '.join(cov_info)} → Aggiustamento: {adj*100:+.2f}%")
                    else:
                        st.caption("ℹ️ Nessuna covariate — quantili nativi TimesFM")

    
                    dates_f = pd.date_range(
                        start=last_d + pd.Timedelta(days=1),
                        periods=len(p50), freq='D'
                    )
                    res_df = pd.DataFrame({
                        'Data': dates_f, 'Target': p50,
                        'Minimo': p10,   'Massimo': p90
                    })
                    target_f     = res_df['Target'].iloc[-1]
                    diff_pct     = ((target_f - last_p) / last_p) * 100
                    st.session_state['res_df']          = res_df
                    st.session_state['target_f']        = target_f
                    st.session_state['diff_pct']        = diff_pct
                    st.session_state['context_used']    = context_days
                    st.session_state['forecast_used']   = forecast_days
                    st.session_state['sentiment_used']  = sentiment
    
                except Exception as e:
                    st.error(f"Errore: {e}")
                    st.exception(e)
    
        # ─── RISULTATI ───
        # Invalida previsione se context o horizon sono cambiati
        if (st.session_state.get('context_used') != context_days or
                st.session_state.get('forecast_used') != forecast_days or
                st.session_state.get('sentiment_used') != sentiment):
            if st.session_state.get('res_df') is not None:
                st.warning("⚠️ Hai modificato i parametri — riesegui l'Analisi AI per aggiornare la previsione.")

        if st.session_state.get('res_df') is not None and            st.session_state.get('context_used') == context_days and            st.session_state.get('forecast_used') == forecast_days:
            res_df       = st.session_state['res_df']
            target_f     = st.session_state['target_f']
            diff_pct     = st.session_state['diff_pct']
            c2.metric(
                f"Target AI ({res_df['Data'].iloc[-1].strftime('%d/%m')})",
                f"{target_f:.2f} {valuta}", delta=f"{diff_pct:+.2f}%"
            )
    
            # ─── GRAFICO PREVISIONE ───
            st.subheader(f"📈 {ticker_hint} — Previsione Trend (+{len(res_df)} Giorni)")
            fig_p = go.Figure()
            tail = df_clean.tail(context_days)
            fig_p.add_trace(go.Scatter(
                x=tail['ds'], y=tail['Close'],
                name='Storico', line=dict(color='#2962FF', width=2),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Storico: %{y:,.2f}<extra></extra>"
            ))
            fig_p.add_trace(go.Scatter(
                x=tail['ds'], y=tail['SMA50'],
                name='SMA 50', line=dict(color='#F6C85F', width=1), hoverinfo='skip'
            ))
            # SMA20 solo dove ha valori validi (non NaN) nel tail
            sma20_tail = tail[tail['SMA20'].notna()]
            if len(sma20_tail) > 0:
                fig_p.add_trace(go.Scatter(
                    x=sma20_tail['ds'], y=sma20_tail['SMA20'],
                    name='SMA 20', line=dict(color='#F7525F', width=1), hoverinfo='skip'
                ))
            # Banda e Target AI — partono dal giorno dopo last_d, semplice
            fig_p.add_trace(go.Scatter(
                x=pd.concat([res_df['Data'], res_df['Data'][::-1]]),
                y=pd.concat([res_df['Massimo'], res_df['Minimo'][::-1]]),
                fill='toself', fillcolor='rgba(168,85,247,0.12)',
                line=dict(color='rgba(255,255,255,0)'), name='Volatilità Attesa', hoverinfo='skip'
            ))
            cd_target = np.stack([res_df['Minimo'], res_df['Massimo']], axis=1)
            fig_p.add_trace(go.Scatter(
                x=res_df['Data'], y=res_df['Target'],
                name='Target AI (mediana)',
                line=dict(color='#A855F7', width=3),
                customdata=cd_target,
                hovertemplate=(
                    "<b>%{x|%d %b %Y}</b><br>"
                    "Target AI: %{y:,.2f}<br>"
                    "Vol. Min: %{customdata[0]:,.2f}<br>"
                    "Vol. Max: %{customdata[1]:,.2f}<extra></extra>"
                )
            ))
            y_min = float(min(res_df['Minimo'].min(), tail['Close'].min()))
            y_max = float(max(res_df['Massimo'].max(), tail['Close'].max()))
            fig_p.add_trace(go.Scatter(
                x=[last_d, last_d], y=[y_min, y_max], mode='lines',
                line=dict(color='white', dash='dash', width=1),
                opacity=0.4, name='Oggi', showlegend=False, hoverinfo='skip'
            ))

            # ─── SUPPORTO E RESISTENZA — calcolati sul context_days ───
            lookback = min(context_days, len(df_clean))
            recent   = df_clean['Close'].iloc[-lookback:]
            resist   = float(recent.max())
            support  = float(recent.min())
            # Estremi del grafico: da inizio tail a fine previsione
            x_start = tail['ds'].iloc[0]
            x_end   = res_df['Data'].iloc[-1]
            fig_p.add_trace(go.Scatter(
                x=[x_start, x_end], y=[resist, resist], mode='lines',
                line=dict(color='#F7525F', dash='dot', width=1.5),
                opacity=0.7, name=f'Massimo {resist:.2f}', hoverinfo='skip'
            ))
            fig_p.add_trace(go.Scatter(
                x=[x_start, x_end], y=[support, support], mode='lines',
                line=dict(color='#26a69a', dash='dot', width=1.5),
                opacity=0.7, name=f'Minimo {support:.2f}', hoverinfo='skip'
            ))

            fig_p.update_layout(**get_tv_layout(chart_height, ))
            st.plotly_chart(fig_p, use_container_width=True)

            # ─── SELETTORE DATA CON CALENDARIO ───
            min_date = tail['ds'].iloc[0].date()
            max_date = res_df['Data'].iloc[-1].date()
            sel_dt = st.date_input(
                "🔍 Seleziona data per vedere i dettagli:",
                value=None,
                min_value=min_date,
                max_value=max_date,
                format="DD.MM.YYYY",
                key="date_selector"
            )

            st.markdown("---")
            cols_pt = st.columns(4)
            if sel_dt:
                hist_match = tail[tail['ds'].dt.date == sel_dt]
                fore_match = res_df[res_df['Data'].dt.date == sel_dt]

                if not hist_match.empty:
                    cols_pt[0].metric("📅 Data", sel_dt.strftime('%d.%m.%Y'))
                    cols_pt[1].metric("💰 Storico", f"{hist_match['Close'].iloc[0]:,.2f} {valuta}")
                    cols_pt[2].metric("📉 Vol. Min", "—")
                    cols_pt[3].metric("📈 Vol. Max", "—")
                elif not fore_match.empty:
                    cols_pt[0].metric("📅 Data", sel_dt.strftime('%d.%m.%Y'))
                    cols_pt[1].metric("🎯 Target AI", f"{fore_match['Target'].iloc[0]:,.2f} {valuta}")
                    cols_pt[2].metric("📉 Vol. Min", f"{fore_match['Minimo'].iloc[0]:,.2f} {valuta}")
                    cols_pt[3].metric("📈 Vol. Max", f"{fore_match['Massimo'].iloc[0]:,.2f} {valuta}")
                else:
                    cols_pt[0].metric("📅 Data", sel_dt.strftime('%d.%m.%Y'))
                    cols_pt[1].metric("⚠️", "Data non disponibile")
                    cols_pt[2].metric("📉 Vol. Min", "—")
                    cols_pt[3].metric("📈 Vol. Max", "—")
            else:
                cols_pt[0].metric("📅 Data", "— seleziona")
                cols_pt[1].metric("💰 Prezzo", "—")
                cols_pt[2].metric("📉 Vol. Min", "—")
                cols_pt[3].metric("📈 Vol. Max", "—")
            st.markdown("---")

            # ─── PANNELLO STATISTICHE ───
            with st.expander("📊 Statistiche del periodo & Correlazioni Covariates"):
                s_col1, s_col2 = st.columns(2)
                with s_col1:
                    st.markdown("**📈 Statistiche Asset**")
                    close_s   = df_clean['Close']
                    vol_daily = close_s.pct_change().std() * 100
                    gain_pct  = ((close_s.iloc[-1] - close_s.min()) / close_s.min()) * 100
                    drop_pct  = ((close_s.max() - close_s.iloc[-1]) / close_s.max()) * 100
                    max_date = df_clean.loc[close_s.idxmax(), 'ds'].strftime('%d.%m.%Y')
                    min_date = df_clean.loc[close_s.idxmin(), 'ds'].strftime('%d.%m.%Y')
                    lines = [
                        f"- **Volatilità giornaliera:** {vol_daily:.2f}%",
                        f"- **Ultimo prezzo:** {last_p:.2f} {valuta} ({last_d.strftime('%d.%m.%Y')})",
                        f"- **Massimo periodo:** {close_s.max():.2f} {valuta} ({max_date})",
                        f"- **Minimo periodo:** {close_s.min():.2f} {valuta} ({min_date})",
                        f"- **Dal minimo:** +{gain_pct:.1f}%",
                        f"- **Dal massimo:** -{drop_pct:.1f}%",
                        f"- **Giorni analizzati:** {len(df_clean)}",
                    ]
                    st.markdown("\n".join(lines))
                with s_col2:
                    st.markdown("**🔗 Correlazioni con Covariates**")
                    # Usa ds come indice e raw_download per le covariates (mantengono le date)
                    price_s_idx = df_clean.set_index('ds')['Close']
                    price_ret_s = price_s_idx.pct_change().dropna()
                    price_ret_s.index = pd.to_datetime(price_ret_s.index).normalize()
                    corr_lines  = []
                    raw_dl = st.session_state.get('raw_download')
                    for cov_name, cov_ser in [
                        ("VIX", st.session_state.get('vix_series')),
                        ("DXY", st.session_state.get('dxy_series')),
                        ("TNX", st.session_state.get('tnx_series')),
                        ("Volume", st.session_state.get('vol_series')),
                    ]:
                        if cov_ser is not None:
                            s_ret = cov_ser.pct_change().dropna()
                            s_ret.index = pd.to_datetime(s_ret.index).normalize()
                            common = price_ret_s.align(s_ret, join='inner')
                            r = float(common[0].corr(common[1]))
                            if np.isnan(r): r = 0.0
                            direction = "negativa" if r < -0.1 else ("positiva" if r > 0.1 else "neutrale")
                            strength  = "forte" if abs(r) > 0.4 else ("moderata" if abs(r) > 0.2 else "debole")
                            corr_lines.append(f"- **{cov_name}:** r={r:+.3f} ({strength}, {direction})")
                        else:
                            corr_lines.append(f"- **{cov_name}:** non disponibile")
                    st.markdown("\n".join(corr_lines))

            st.divider()
    
            with st.expander("📂 Tabella Risultati Previsione"):
                st.dataframe(res_df, use_container_width=True)