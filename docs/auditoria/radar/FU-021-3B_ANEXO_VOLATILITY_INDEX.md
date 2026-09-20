# FU-021-3B — Anexo: verificación empírica de VOLATILITY_INDEX

**Fase:** investigación empírica complementaria a FU-021-3B
**Fecha:** 2026-09-16
**HEAD de referencia:** 794f0e9 (origin/main)
**Informe principal:** FU-021-3B_INFORME_INDEX_EOD_RATE_YIELD.md (b645de4)
**Alcance:** 3 índices de volatilidad del universo `df_market` (`^VIX`, `^VIX3M`, `^VXN`)
**Estado:** cerrado. Complementa FU-021-3B.

---

## 0. Motivo del anexo

FU-021-3B midió empíricamente las clases INDEX_EOD y RATE_YIELD. Durante la Fase 0 de 3B se descubrió que el universo no-equity contiene **6 clases**, no 5 como documentó Parte A de FU-021-5. La clase VOLATILITY_INDEX apareció en el inventario pero no fue medida.

Este anexo cierra esa laguna con la misma metodología. Confirma si VOLATILITY_INDEX es clase independiente o hereda de INDEX_EOD_USA.

---

## 1. Inventario

### 1.1. Universo

| Ticker | n (parquet) | first_date | last_date | close |
|---|---|---|---|---|
| `^VIX` | 2515 | 2016-09-14 | 2026-09-14 | 17.10 |
| `^VIX3M` | 2473 | 2016-09-15 | 2026-09-14 | 19.28 |
| `^VXN` | 2512 | 2016-09-15 | 2026-09-14 | 22.05 |

**Hallazgo H-3B-bis-3:** `^VIX3M` tiene ~40 filas menos que `^VIX` y `^VXN` y empieza un día más tarde. Es un ticker con histórico más irregular, no un problema del pipeline.

---

## 2. Yahoo vs parquet

| Ticker | Parquet last | Yahoo last | Lag | Match día común |
|---|---|---|---|---|
| `^VIX` | 2026-09-14 | 2026-09-15 | 1d | idéntico (17.10) |
| `^VIX3M` | 2026-09-14 | 2026-09-15 (n=1) | 1d | no verificable |
| `^VXN` | 2026-09-14 | 2026-09-15 | 1d | idéntico (22.05) |

**Hallazgo H-3B-bis-2:** `^VIX3M` con descarga individual `period='15d'` devuelve **1 sola fila**. Anomalía de `yfinance`, no del pipeline. La descarga en batch (que es la que usa el pipeline) sí trae el histórico completo.

**Confirmación:** batch download de los 3 tickers juntos devuelve los 3 hasta 2026-09-15 sin anomalía.

---

## 3. Veredicto

**VOLATILITY_INDEX hereda de INDEX_EOD_USA.**

| Aspecto | INDEX_EOD_USA | VOLATILITY_INDEX |
|---|---|---|
| Calendario | NYSE | CBOE (mismo calendario bursátil USA) |
| Lag observado | 1 día | 1 día |
| Valores estables | Sí | Sí |
| Descarga | Yahoo batch | Yahoo batch |

**Hallazgo H-3B-bis-1:** VOLATILITY_INDEX no requiere contrato propio. Se puede declarar como **subclase de INDEX_EOD_USA** o como clase análoga con los mismos parámetros temporales.

---

## 4. Implicaciones para Parte A

- §A.1 debe pasar de 5 a 6 clases.
- `VOLATILITY_INDEX` debe añadirse al diagrama de clases.
- El contrato puede delegar en INDEX_EOD_USA (sin parámetros propios).
- La partición de consolidación debe incluir `VOLATILITY_INDEX` como subconjunto homogéneo.

---

## 5. Hallazgos registrados

| ID | Hallazgo | Severidad |
|---|---|---|
| H-3B-bis-1 | VOLATILITY_INDEX ~ INDEX_EOD_USA (lag 1d, valores estables) | Informativa |
| H-3B-bis-2 | `^VIX3M` individual download `yfinance` devuelve n=1; batch OK | Documental |
| H-3B-bis-3 | `^VIX3M` tiene ~40 filas menos y empezó 1 día más tarde | Documental |

---

## 6. Anexo — Comandos reproducibles

**Inventario:**

    py -c "
    import pandas as pd
    df = pd.read_parquet('data/market_data.parquet')
    for t in ['^VIX', '^VIX3M', '^VXN']:
        s = df[('Close', t)].dropna()
        print(t, len(s), s.index[-1].date())
    "

**Yahoo vs parquet:**

    py -c "
    import pandas as pd
    import yfinance as yf
    df = pd.read_parquet('data/market_data.parquet')
    for t in ['^VIX', '^VIX3M', '^VXN']:
        pq = df[('Close', t)].dropna()
        yf_df = yf.download(t, period='15d', progress=False, auto_adjust=True)
        if isinstance(yf_df.columns, pd.MultiIndex):
            yf_df.columns = yf_df.columns.get_level_values(0)
        s = yf_df['Close'].dropna()
        print(t, 'parquet=' + str(pq.index[-1].date()), 'yahoo=' + str(s.index[-1].date()), 'n_yahoo=' + str(len(s)))
    "

**Batch download:**

    py -c "
    import pandas as pd
    import yfinance as yf
    df = yf.download(['^VIX', '^VIX3M', '^VXN'], period='15d', progress=False, auto_adjust=True)
    print('shape:', df.shape)
    for t in ['^VIX', '^VIX3M', '^VXN']:
        s = df.xs(t, level=1, axis=1)['Close'].dropna()
        print(t, 'last=', s.index[-1].date())
    "

---

**Fin del anexo VOLATILITY_INDEX.**
**HEAD de referencia:** 794f0e9 (origin/main).
**Fecha:** 2026-09-16.
**Estado:** cerrado. Complementa FU-021-3B.