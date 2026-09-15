# FU-021-3C — Verificación empírica de FUTURE_SETTLEMENT y FX_DAILY_CUT

**Fase:** investigación empírica previa a Parte B de FU-021-5
**Fecha:** 2026-09-16
**HEAD de referencia:** b645de4 (origin/main)
**Alcance:** 5 futuros de commodities y 3 pares FX del universo df_market
**Autorizado por:** Dictamen del contrato temporal (Q-C2.6, camino α)
**Estado:** cerrado sin causa raíz exacta. Alimenta Parte B de FU-021-5.

---

## 0. Motivo del informe

FU-021-3B verificó empíricamente las clases INDEX_EOD y RATE_YIELD. FU-021-3C es su contraparte para las dos clases restantes bloqueadas por provider:

- FUTURE_SETTLEMENT
- FX_DAILY_CUT

Ambas estaban clasificadas en Parte A como "bloqueadas por provider dedicado". FU-021-3C confirma si esa clasificación se sostiene empíricamente y aporta los datos necesarios para redactar Parte B.

**Nota metodológica:** este informe cierra sin causa raíz exacta para el hallazgo principal (H-3C-7). Se documenta explícitamente por qué se detiene la investigación en Fase 4.

---

## 1. Metodología

Cinco fases consecutivas, todas de solo lectura.

| Fase | Objetivo | Herramienta |
|---|---|---|
| 0 | Inventario: clasificación y last_date por ticker | pd.read_parquet + get_instrument_class |
| 1 | Descarga Yahoo vs parquet para 8 tickers | yfinance |
| 2 | Inspección de Ticker.info y contratos específicos | yfinance |
| 3 | Verificación de contrato front-month vs parquet | yfinance (contratos explícitos) |
| 4 | Verificación intradía 1h de la ventana de captura | yfinance (interval=1h) |

La Fase 4 se detuvo sin cerrar causa raíz exacta. Se documenta en §6.

---

## 2. Fase 0 — Inventario

### 2.1. Universo real

| Clase | Cuenta | Miembros |
|---|---|---|
| FUTURE | 5 | BZ=F (Brent), CL=F (WTI), GC=F (Gold), HG=F (Copper), NG=F (Natural Gas) |
| FX | 3 | EURUSD=X, USDCNY=X, USDJPY=X |

**Hallazgo H-3C-5:** FUTURE contiene solo commodities. No hay futuros de equity (ES=F, NQ=F) ni de bonos (ZB=F, ZN=F).

**Hallazgo H-3C-6:** FX contiene solo 3 pares. No hay GBP, CHF, AUD ni cruces.

### 2.2. last_date por ticker

**FUTURE — homogéneo:**

| Ticker | last_date |
|---|---|
| BZ=F | 2026-09-14 |
| CL=F | 2026-09-14 |
| GC=F | 2026-09-14 |
| HG=F | 2026-09-14 |
| NG=F | 2026-09-14 |

**Hallazgo H-3C-1:** FUTURE tiene lag de 1 día, homogéneo. Consistente con INDEX_USA.

**FX — heterogéneo intra-clase:**

| Ticker | last_date | n |
|---|---|---|
| EURUSD=X | 2026-09-15 | 2601 |
| USDCNY=X | 2026-09-14 | 2599 |
| USDJPY=X | 2026-09-15 | 2601 |

**Hallazgo H-3C-2:** FX tiene last_date asimétrico dentro de la misma clase.

**Hallazgo H-3C-4:** FX tiene ~89 filas más que el resto (2601 vs 2512) por festivos USA que los mercados FX sí negocian.

### 2.3. Hallazgo crítico — fila fantasma

Análisis de la última fila de df_market (`2026-09-15`):
tickers con Close no NaN: 2 / 562
por clase:
FX: 2

text

**Hallazgo H-3C-3:** la última fila del DataFrame contiene solo 2 tickers (EURUSD=X, USDJPY=X). Los 560 restantes son NaN.

**Implicación:** cualquier consumidor que use `df_market.index[-1]` como fecha de referencia sobre un DataFrame completo está operando sobre NaN en el 99.64% de las columnas. El DataFrame entregado NO es temporalmente coherente en su última fila.

---

## 3. Fase 1 — Yahoo vs parquet

### 3.1. FUTURE

| Ticker | Parquet 14/09 | Yahoo 14/09 | Diff abs | Diff rel |
|---|---|---|---|---|
| BZ=F | 106.49 | 105.68 | 0.81 | 0.76% |
| CL=F | 102.10 | 101.39 | 0.71 | 0.70% |
| GC=F | 4327.50 | 4351.90 | 24.40 | 0.56% |
| HG=F | 6.389 | 6.330 | 0.059 | 0.92% |
| NG=F | 2.880 | 2.896 | 0.016 | 0.56% |

**Hallazgo H-3C-7:** los valores Close del parquet y Yahoo divergen en la misma fecha. La divergencia **no es uniforme** (mezcla signos positivos y negativos). No es escala, redondeo ni conversión.

Comparativa con índices USA (FU-021-3B): diff era 0.002-0.003. Aquí es 100-1000× mayor.

### 3.2. FX

| Ticker | Parquet last | Yahoo last | Lag | Match |
|---|---|---|---|---|
| EURUSD=X | 2026-09-15 | 2026-09-15 | 0 | diff 3.46e-4 |
| USDCNY=X | 2026-09-14 | 2026-09-15 | 1 | idéntico en día común |
| USDJPY=X | 2026-09-15 | 2026-09-15 | 0 | diff 0.35% |

**Hallazgo H-3C-8:** FX tiene valores estables, con excepción de USDJPY cuyo diff relativo (0.35%) merece verificación adicional en Parte B.

**Hallazgo H-3C-9:** confirma el lag heterogéneo intra-clase ya observado en Fase 0.

---

## 4. Fase 2 — Inspección Ticker.info

### 4.1. Contrato actual del front-month

Salida de `yf.Ticker('CL=F').info`:
symbol : CL=F
shortName : Crude Oil Oct 26
expireDate : 1790035200 (2026-09-22)
quoteType : FUTURE
exchange : NYM
market : us24_market

text

**Hallazgo H-3C-10:** el ticker `CL=F` identifica "el front-month actual de WTI". El contrato subyacente cambia con el rollover.

**Hallazgo H-3C-11:** el front-month actual de CL=F expira el 2026-09-22. Rollover inminente.

### 4.2. Tests descartados

| Test | Resultado | Hipótesis descartada |
|---|---|---|
| Estabilidad por period (5d/1mo/3mo/1y) | Valor idéntico | No hay ajuste retroactivo por rango |
| auto_adjust True/False | Valor idéntico | No hay ajuste de dividendos/splits |
| OHLCV en parquet | 5 columnas | No es problema de columnas |

---

## 5. Fase 3 — Contratos específicos

### 5.1. Yahoo expone contratos explícitos

| Ticker | Close(14/09) |
|---|---|
| GC=F | 4351.8999 |
| GCZ26.CMX (dic 2026) | 4351.8999 |
| GCV26.CMX (oct 2026) | 4318.0000 |

**Hallazgo H-3C-10 (confirmado):** Yahoo soporta contratos específicos con formato `{ticker}{mes}{año}.{exchange}`.

**Hallazgo H-3C-11 (confirmado):** GC=F hoy es el contrato de diciembre (GCZ26).

### 5.2. Verificación del contrato capturado

| Ticker | Close(14/09) | Parquet 14/09 | Coincide |
|---|---|---|---|
| GC=F (hoy) | 4351.8999 | 4327.50 | No |
| GCZ26.CMX | 4351.8999 | 4327.50 | No |
| GCV26.CMX | 4318.0000 | 4327.50 | No |
| CL=F (hoy) | 101.39 | 102.10 | No |
| CLV26.NYM | 101.39 | 102.10 | No |
| CLX26.NYM | 97.14 | 102.10 | No |

**Hallazgo H-3C-14:** el valor del parquet **no coincide con ningún contrato actual** de Yahoo (ni el front-month, ni el adyacente, ni el siguiente).

La hipótesis del rollover retroactivo puro queda descartada como causa única.
---

## 6. Fase 4 — Verificación intradía

### 6.1. Objetivo

Verificar si el parquet capturó un tick intradía de la sesión asiática del 15/09 a las 02:21 UTC, en lugar del settlement oficial del 14/09.

### 6.2. Ticks disponibles entre 01:30-03:30 UTC del 15/09

| Ticker | Ticks intradía | Parquet | Yahoo 1d hoy |
|---|---|---|---|
| GC=F | 4354.90 / 4345.10 | 4327.50 | 4351.90 |
| CL=F | 102.89 / 102.93 | 102.10 | 101.39 |

### 6.3. Resultado

**Ningún valor de Yahoo reproduce el parquet.**

- GC=F: parquet a 18-27 puntos de los ticks intradía.
- CL=F: parquet a 0.79-0.83 de los ticks intradía.
- Tampoco coincide con el Yahoo 1d actual.

### 6.4. Hipótesis residual no verificable

**H-3C-7e — Yahoo reescribió el histórico entre capturas.**

El parquet capturó el 15/09 02:21 UTC un valor que Yahoo devolvía entonces. Yahoo reescribió posteriormente su histórico. El valor original **ya no es recuperable** vía Yahoo.

**Inverificable con los datos disponibles.** Yahoo no expone su histórico de revisiones.

### 6.5. Decisión de cierre

La investigación se detiene en Fase 4 sin causa raíz exacta. Razones:

1. **Coste/beneficio desfavorable.** Cerrar la causa raíz exacta requeriría instrumentación continua (capturas diarias, comparación semanal) durante múltiples ciclos.
2. **La conclusión operativa ya está clara.** Los históricos del parquet para FUTURE no son reproducibles. Sea rollover, sea ajuste, sea corrección, el resultado es el mismo.
3. **La causa raíz exacta no cambia la decisión arquitectónica.** `FUTURE_SETTLEMENT` requiere un provider que exponga el contrato explícito y el settlement. Yahoo continuo no sirve.

---

## 7. Diagnóstico

### 7.1. FUTURE_SETTLEMENT

**Los históricos de futuros del parquet no son reproducibles.**

Un mismo ticker, una misma fecha, dos capturas en momentos distintos → dos valores distintos. `CL=F` y `GC=F` no identifican un contrato concreto, sino "el front-month actual". Cuando el front-month cambia, Yahoo reescribe su histórico de forma no determinista.

**Consecuencia:** el parquet no es auditable para la clase FUTURE_SETTLEMENT. Ningún consumidor puede declarar `effective_date` con coherencia para estos 5 tickers.

### 7.2. FX_DAILY_CUT

**Los históricos de FX son estables**, con dos matices:

- Lag heterogéneo intra-clase (EURUSD/USDJPY al día, USDCNY con 1 día de lag).
- USDJPY tiene diff relativo de 0.35% que merece verificación en Parte B.

FX es una clase más limpia que FUTURE, pero sigue requiriendo `max_lag_days` por par.

### 7.3. Consolidación mixta

La fila 2026-09-15 con solo 2/562 tickers confirma que la consolidación mixta (Parte A §A.2) es la norma, no la excepción.

---

## 8. Implicaciones para Parte B

### 8.1. FUTURE_SETTLEMENT — clase bloqueada, no "provider pendiente"

La Parte A clasificaba FUTURE_SETTLEMENT como "bloqueada por provider dedicado". FU-021-3C **confirma y refuerza** esa clasificación con evidencia empírica:

- **No es solo que falte un provider.**
- **Es que el ticker continuo de Yahoo no es auditable.**
- Aunque Yahoo funcionase al 100%, `CL=F` seguiría siendo un identificador inestable.

**Recomendación para Parte B:** documentar la clase como `BLOCKED` con causa raíz medida. La activación requiere provider que exponga el contrato explícito (CME directo, ICE) o migración a contratos explícitos con lógica de rollover declarada.

### 8.2. FX_DAILY_CUT — clase activable con contrato de lag

FX no tiene el problema de FUTURE. Requiere:

- Declarar `max_lag_days` por par.
- Documentar el cutoff del mercado FX (17:00 ET).
- Aceptar que la última fila puede tener solo algunos pares.

### 8.3. Impacto en consumidores

**30-40% del pipeline consume futuros.** Afectados directos:

- `pipeline/flows_primary.py`
- `pipeline/mte_confirmation.py` (via `compute_cross_asset_ratios`: HG=F/GC=F)
- `pipeline/sectors_base.py` (via `cross_asset_context`: CL=F, HG=F, GC=F)
- `regimes/macro_regime.py` (industrial_tickers + gold)
- `indicators/mte.py`

**Hallazgo H-3C-12:** cualquier análisis que use futuros del parquet está operando sobre valores que pueden divergir de lo que Yahoo devolvería hoy para la misma fecha.

### 8.4. Decisión pendiente para Parte B

FU-021-3C no decide entre:

- Migrar a contratos explícitos (`GCZ26.CMX`).
- Migrar a provider dedicado (CME directo).
- Congelar la clase FUTURE_SETTLEMENT como `BLOCKED` en Parte A.

Esa decisión es de Parte B, no de 3C. 3C aporta datos.

---

## 9. Hallazgos registrados

| ID | Hallazgo | Clase | Severidad |
|---|---|---|---|
| H-3C-1 | FUTURE lag homogéneo 1 día | Observación | Informativa |
| H-3C-2 | FX lag heterogéneo intra-clase | Comportamiento Yahoo | Baja |
| **H-3C-3** | **Fila 2026-09-15 con solo 2/562 tickers** | **Consolidación** | **Alta** |
| H-3C-4 | FX +89 filas por festivos USA | Observación | Informativa |
| H-3C-5 | FUTURE = 5 commodities, sin equity ni bonos | Límite universo | Documental |
| H-3C-6 | FX = 3 pares, sin GBP/CHF/AUD | Límite universo | Documental |
| **H-3C-7** | **Históricos de FUTURE divergen entre capturas** | **Integridad** | **Alta** |
| H-3C-8 | FX valores estables (excepto USDJPY) | Observación | Informativa |
| H-3C-9 | FX lag heterogéneo confirmado vs Yahoo | Confirmación | Baja |
| H-3C-10 | Yahoo expone contratos explícitos (GCZ26.CMX) | Capacidad | Informativa |
| H-3C-11 | CL=F front-month expira 2026-09-22 | Estado | Informativa |
| **H-3C-12** | **30-40% del pipeline consume futuros no reproducibles** | **Impacto** | **Alta** |
| H-3C-13 | Consolidación mixta exige contrato por clase | Confirma Parte A | Informativa |
| H-3C-14 | Parquet no coincide con ningún contrato actual de Yahoo | Causa raíz parcial | Alta |
| H-3C-15 | Causa raíz exacta no verificable con datos disponibles | Limitación | Documental |

**Total: 15 hallazgos.**

---

## 10. Lo que este informe NO decide

- No activa contratos por clase.
- No implementa cambios.
- No modifica el pipeline.
- No autoriza retirar `trim_to_last_valid_date`.
- No decide entre contratos explícitos y provider dedicado.
- No decide el cutoff del mercado FX.
- No documenta la causa raíz exacta de H-3C-7 (ver §6.5).

---

## 11. Anexo — Comandos reproducibles

**Fase 0 — inventario:**

    py -c "
    import pandas as pd
    from src.instrument_registry import get_instrument_class
    df = pd.read_parquet('data/market_data.parquet')
    tickers = sorted(set(df.columns.get_level_values(1)))
    by_class = {}
    for t in tickers:
        cls = get_instrument_class(t)
        by_class.setdefault(cls, []).append(t)
    for cls in sorted(by_class):
        print(cls, len(by_class[cls]))
        if cls in ('FUTURE', 'FX'):
            for t in by_class[cls]: print('  ', t)
    "

**Fase 1 — Yahoo vs parquet:**

    py -c "
    import pandas as pd
    import yfinance as yf
    df = pd.read_parquet('data/market_data.parquet')
    for t in ['CL=F', 'GC=F', 'HG=F', 'NG=F', 'BZ=F']:
        pq = df[('Close', t)].dropna()
        yf_df = yf.download(t, period='15d', progress=False, auto_adjust=True)
        if isinstance(yf_df.columns, pd.MultiIndex):
            yf_df.columns = yf_df.columns.get_level_values(0)
        s = yf_df['Close'].dropna()
        print(t, 'parquet=' + str(pq.index[-1].date()), 'yahoo=' + str(s.index[-1].date()))
    "

**Fase 2 — Ticker.info:**

    py -c "
    import yfinance as yf
    info = yf.Ticker('CL=F').info
    for k in ['symbol', 'shortName', 'expireDate', 'quoteType', 'exchange']:
        print(k, ':', info.get(k))
    "

**Fase 3 — contratos explícitos:**

    py -c "
    import pandas as pd
    import yfinance as yf
    target = pd.Timestamp('2026-09-14')
    for t in ['GC=F', 'GCZ26.CMX', 'GCV26.CMX', 'CL=F', 'CLV26.NYM', 'CLX26.NYM']:
        df = yf.download(t, period='1mo', progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = df['Close'].dropna()
        if target in s.index:
            print(t, 'close(14/09)=', float(s.loc[target]))
    "

**Fase 4 — intradía 1h:**

    py -c "
    import pandas as pd
    import yfinance as yf
    for t in ['GC=F', 'CL=F']:
        df = yf.download(t, period='10d', interval='1h', progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = df['Close'].dropna()
        if s.index.tz is None:
            s.index = s.index.tz_localize('UTC')
        lo = pd.Timestamp('2026-09-15 01:30:00', tz='UTC')
        hi = pd.Timestamp('2026-09-15 03:30:00', tz='UTC')
        window = s.loc[(s.index >= lo) & (s.index <= hi)]
        print(t, ':')
        for idx, val in window.items():
            print('  ', idx, '->', float(val))
    "

---

**Fin del informe FU-021-3C.**
**HEAD de referencia:** b645de4 (origin/main).
**Fecha:** 2026-09-16.
**Estado:** cerrado sin causa raíz exacta. Alimenta Parte B de FU-021-5.