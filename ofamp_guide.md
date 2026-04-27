> ⚠️ **DISCLAIMER — Leggere prima di usare**
>
> **OFAMP è un progetto sperimentale a scopo educativo e di ricerca.**
> Le previsioni generate non costituiscono consulenza finanziaria, raccomandazioni di investimento,
> o suggerimenti di acquisto/vendita di strumenti finanziari.
> Qualsiasi decisione finanziaria è di esclusiva responsabilità dell'utente.
>
> **Il buon senso del trader vale sempre più di qualsiasi algoritmo.**

---

## 🧠 TimesFM — Cos'è e come funziona

**TimesFM** (Time Series Foundation Model) è un modello AI sviluppato da Google, addestrato su miliardi di serie temporali reali. Lavora esclusivamente sui **pattern numerici** del prezzo — non conosce notizie, fondamentali o eventi macro.

| Context | Comportamento |
|---------|--------------|
| **64–96 gg** | Molto reattivo al trend recente |
| **128–160 gg** | Bilanciato — zona ottimale ✅ |
| **192–256 gg** | Vede cicli lunghi — usa con cautela |

> Il context massimo è **256 giorni**. Oltre questa soglia i benefici sono marginali su serie finanziarie.

---

## 📅 Profilo Operativo Ottimale

Dall'esperienza diretta su diversi asset:

| Parametro | Valore ottimale |
|-----------|----------------|
| Campione dati | 1 anno |
| Giorni storici (Context) | 128 giorni |
| Giorni da prevedere | 30–40 giorni |

Con 1 anno di dati e context=128, il modello usa attivamente gli ultimi 128 giorni. I dati precedenti servono per SMA 20, SMA 50 e baseline statistica delle covariates.

---

## 🎯 Quando usare OFAMP

### ✅ Utile quando:
- Mercato in **consolidamento** o laterale
- Asset in **correzione** dopo un trend — per capire se rimbalza
- Momenti di **incertezza** dove non è chiara la direzione
- Es: NUCL.L a 59 dopo essere stato a 70 — la previsione aggiunge valore

### ⚠️ Da usare con cautela quando:
- Asset in **trend strutturale forte** (es. oro in bull run da 2 anni)
- In questi casi il buon senso del trader vale già tanto quanto il modello
- TimesFM è conservativo: tende a sottostimare i trend forti

> In un trend rialzista ovvio, anche senza AI è ragionevole aspettarsi continuazione. Lo strumento aggiunge valore nei casi **non ovvi**.

---

## 📥 Tab Dati — Scarica e carica i dati

### Scarica da Yahoo Finance
Inserisci il ticker Yahoo Finance (es. `BTC-USD`, `NUCL.L`, `AAPL`) e seleziona il periodo.

Il sistema scarica automaticamente anche le covariates:
- **VIX** (`^VIX`) — indice di volatilità del mercato
- **DXY** (`DX-Y.NYB`) — indice del dollaro USA
- **TNX** (`^TNX`) — rendimento Treasury USA 10 anni
- **Volume** — dal file del ticker principale

Dopo il download appare la conferma con i bottoni **Salva CSV** e **Salva ZIP** (con tutte le covariates incluse).

### Carica da file CSV locali
Utile per riusare dati già scaricati. Carica i 4 file separatamente:
- Ticker principale (es. `BTC-USD_price.csv`)
- `VIX.csv`, `DXY.csv`, `TNX.csv`

---

## 📊 Tab Dati Disponibili

Visualizza le serie scaricate come grafici separati, ognuno con la sua scala reale.

- **Ticker principale** — prezzo di chiusura
- **Volume** — barre verdi (giorno rialzista) / rosse (giorno ribassista)
- **VIX, DXY, TNX** — linea con scala reale e correlazione con il ticker

**Correlazione** mostrata nel titolo di ogni grafico:
- `r > +0.4` → positiva forte
- `r < -0.4` → negativa forte
- `-0.2 < r < +0.2` → neutrale

Usa il **multiselect** per scegliere quali serie visualizzare.

---

## 📈 Tab Analisi & Previsione

### Configurazione (sidebar)

| Parametro | Descrizione | Ottimale |
|-----------|-------------|---------|
| **📅 Giorni storici** | Quanti giorni vede il modello AI | 128 gg |
| **🔮 Giorni da prevedere** | Orizzonte della previsione | 30-40 gg |
| **📐 Altezza grafici** | Pixel di altezza dei grafici | 600 px |

### Metriche in cima
- **Ultima Chiusura** — ultimo prezzo nel dataset con data
- **Prezzo Attuale** — prezzo live da Yahoo Finance (aggiornato automaticamente)
- **RSI** — con icona 🔴🟡🟢
- **MACD** — momentum rialzista/ribassista

### Grafico Storico
Mostra il prezzo con:
- **SMA 20** (rossa) — media mobile 20 giorni, trend di breve
- **SMA 50** (gialla) — media mobile 50 giorni, trend di medio
- **Bande di Bollinger** — volatilità relativa alla SMA 20

### Indicatori Tecnici (expander)
- **RSI 14** — ipercomprato > 70, ipervenduto < 30
- **MACD** — istogramma verde = momentum rialzista

### Previsione AI
Clicca **🚀 Esegui Analisi AI** per avviare TimesFM.

> ⚠️ Se cambi i parametri (giorni storici o giorni da prevedere) dopo aver calcolato, l'app chiede di rieseguire.

---

## 🔮 Come viene calcolata la Previsione

### La linea Target AI (viola)

Output diretto di **TimesFM 2.5** (Google, 200M parametri). Il modello riceve gli ultimi `N giorni storici` di prezzi di chiusura e produce una previsione giorno per giorno. Questa è la **stima mediana** — lo scenario centrale più probabile.

### La Volatilità Attesa (banda viola)

Intervallo di confidenza calcolato da TimesFM — quantili p10 e p90. Il prezzo ha circa **80% di probabilità** di trovarsi dentro la banda. La banda si allarga nel tempo e può essere asimmetrica (più spazio verso l'alto o verso il basso a seconda del regime di mercato).

> ✅ **Validazioni reali:**
> - NUCL.L con dati a marzo 2026, previsione 37 giorni: Target AI 60.76, reale 60.70 — errore **6 centesimi**
> - BTC-USD con dati a marzo 2026, previsione 36 giorni: Target AI 75,480, reale 74,783 — errore **0.9%**

### Massimo e Minimo del periodo (linee tratteggiate)

Prezzi estremi del periodo analizzato — riferimento per il trader per valutare livelli chiave.

### Le Covariates (VIX, DXY, TNX, Volume)

Affinano la previsione tramite correlazioni reali calcolate sui dati:

| Covariate | Ticker | Relazione tipica |
|-----------|--------|-----------------|
| **VIX** | `^VIX` | Negativa — VIX alto → pressione ribassista |
| **DXY** | `DX-Y.NYB` | Negativa — dollaro forte → pressione ribassista su asset rischiosi |
| **TNX** | `^TNX` | Negativa — tassi alti → costo capitale alto → pressione ribassista |
| **Volume** | dal ticker | Positiva — volume alto → conferma del trend |

Il segnale è calcolato come z-score con mean reversion (decade verso la media in ~10 giorni). L'aggiustamento totale è limitato a **±3%** — le covariates affinano, non sostituiscono il modello.

---

## 📊 Tab Backtesting Rolling

Testa l'accuratezza storica del modello sui dati già caricati.

### Parametri

| Parametro | Descrizione |
|-----------|-------------|
| **📅 Giorni storici (Context)** | Giorni di storia per ogni test — usa lo stesso valore dell'analisi reale |
| **🔮 Giorni da prevedere (Horizon)** | Quanti giorni avanti prevede in ogni test |
| **⏭️ Passo tra test (Step)** | Ogni quanti giorni avanza la finestra — Step=14 = un test ogni 2 settimane |

### Metriche

| Metrica | Cosa misura | Buon valore |
|---------|-------------|-------------|
| **Errore medio assoluto** | Distanza media % tra previsione e reale | < 8% su mercati moderati |
| **Direzione corretta** | % volte il modello prevede correttamente su/giù | > 55% indica valore aggiunto |
| **Prezzi in banda** | % giorni reali dentro la Volatilità Attesa | > 60% indica buona calibrazione |

### Grafico errori
- 🟢 **Verde** — errore < 5% → previsione accurata
- 🟡 **Giallo** — errore 5-15% → accettabile
- 🔴 **Rosso** — errore > 15% → da analizzare

### Sentiment del Trader — Analisi "E se?"

Slider da -10 a +10 applicato a tutte le finestre di test. Permette di rispondere a:
*"Se il mese scorso ero rialzista (+3), il modello avrebbe sbagliato di meno?"*

Confronta i risultati con sentiment=0 (modello puro) e con sentiment=+N (con il tuo giudizio).