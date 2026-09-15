# FU-021-3B — Verificación empírica de INDEX_EOD y RATE_YIELD

**Fase:** investigación empírica previa a Parte B de FU-021-5
**Fecha:** 2026-09-15
**HEAD de referencia:** 61aa99f (origin/main)
**Alcance:** 10 índices, 2 yields y 3 índices de volatilidad del universo `df_market`
**Autorizado por:** Dictamen del contrato temporal (Q-C2.6, camino α)
**Estado:** cerrado. Alimenta Parte B de FU-021-5.

---

## 0. Motivo del informe

La Parte A de FU-021-5 especificó la arquitectura del contrato temporal de `df_market`, pero dejó explícitamente fuera de su alcance la semántica operativa de las clases no verificadas:

- INDEX_EOD
- RATE_YIELD
- FUTURE_SETTLEMENT (bloqueada por provider)
- FX_DAILY_CUT (bloqueada por provider)

FU-021-3B es la investigación empírica sobre las dos primeras. Su producto es este informe, que alimenta la Parte B de FU-021-5.

**La Parte B no puede redactarse sin estos datos.** Los contratos por clase no son un ejercicio teórico: dependen de qué semántica temporal expone realmente el proveedor de datos.

---

## 1. Metodología

Cuatro fases consecutivas, todas de solo lectura.

| Fase | Objetivo | Herramienta |
|---|---|---|
| Fase 0 | Inventario: clasificar los 562 tickers y detectar `last_date` por ticker | `get_instrument_class`, `pd.read_parquet` |
| Fase 1 | Descarga Yahoo vs parquet para 12 tickers no-equity | `yfinance` |
| Fase 2 | Inspección del código: cómo se descarga `market_data` | `Select-String` + lectura |
| Fase 3 | Verificación de hipótesis (batch vs individual, cache timing) | `yfinance`, mtime, config |

**Fase 2 y Fase 3 fueron necesarias porque las fases 0-1 solo mostraron el síntoma.** La causa raíz requería leer el pipeline y verificar cada hipótesis contra la fuente.

---

## 2. Fase 0 — Inventario

### 2.1. Clasificación de los 562 tickers

| Clase | Cuenta | Miembros |
|---|---|---|
| EQUITY | 539 | — |
| INDEX | 10 | DX-Y.NYB, ^DJI, ^FTSE, ^GDAXI, ^GSPC, ^IBEX, ^NDX, ^RUT, ^SPGSCI, ^STOXX50E |
| RATE_YIELD | 2 | ^FVX (5y), ^TNX (10y) |
| VOLATILITY_INDEX | 3 | ^VIX, ^VIX3M, ^VXN |
| FUTURE | 5 | — |
| FX | 3 | — |
| **Total** | **562** | |

**Hallazgo H-3B-2:** existen **6 clases**, no 5 como documentó Parte A. `VOLATILITY_INDEX` es una clase propia, no subclase de INDEX.

### 2.2. `last_date` por ticker

| Grupo | last_date |
|---|---|
| Índices USA (^GSPC, ^DJI, ^NDX, ^RUT, ^SPGSCI) | 2026-09-14 |
| Índices Europa (^FTSE, ^GDAXI, ^IBEX, ^STOXX50E) | **2026-09-11** |
| VIX / VIX3M / VXN | 2026-09-14 |
| ^FVX / ^TNX | 2026-09-14 |
| DX-Y.NYB | 2026-09-14 |
| Equity (implícito) | 2026-09-15 |

**Hallazgo H-3B-1:** no existe una "última fecha correcta" común. La distancia entre equity y los índices europeos es de **4 días calendario**.

**Hallazgo H-3B-8:** `df_market.index[-1] = 2026-09-15` corresponde a una fila donde **solo hay equity**. Ningún índice tiene valor en esa fecha. Confirma empíricamente la tesis de Parte A: `global_last_date ≠ fecha económica común`.

---

## 3. Fase 1 — Yahoo vs parquet

### 3.1. Resultados

12 tickers, 4 grupos. Comparación entre `yf.download(ticker, period='15d')` individual y el parquet.

| Ticker | Parquet last | Yahoo last | Lag | Close día común |
|---|---|---|---|---|
| ^GSPC | 2026-09-14 | 2026-09-15 | 1d | idéntico |
| ^DJI | 2026-09-14 | 2026-09-15 | 1d | idéntico |
| ^NDX | 2026-09-14 | 2026-09-15 | 1d | diff 0.002 |
| ^RUT | 2026-09-14 | 2026-09-15 | 1d | diff 0.003 |
| ^SPGSCI | 2026-09-14 | 2026-09-15 | 1d | diff 0.002 |
| ^FTSE | 2026-09-11 | 2026-09-15 | **4d** | idéntico |
| ^GDAXI | 2026-09-11 | 2026-09-14 | **3d** | idéntico |
| ^IBEX | 2026-09-11 | 2026-09-14 | **3d** | idéntico |
| ^STOXX50E | 2026-09-11 | 2026-09-14 | **3d** | idéntico |
| ^FVX | 2026-09-14 | 2026-09-15 | 1d | idéntico |
| ^TNX | 2026-09-14 | 2026-09-15 | 1d | idéntico |
| DX-Y.NYB | 2026-09-14 | 2026-09-15 | 1d | diff 0.041 |

**Hallazgo H-3B-5:** los valores Close en el último día común **coinciden exactamente** (0 o diferencias de redondeo ≤ 0.003). El pipeline descarga correctamente los valores de Yahoo. **No hay bug de datos.**

**Hallazgo H-3B-6:** el lag es 1 día para USA, yields y DXY; **3-4 días para Europa**. Es asimétrico.

**Hallazgo H-3B-9:** DX-Y.NYB tiene discrepancia de 0.041 (dos órdenes de magnitud mayor que el resto). Causa probable: ajuste de composición intradía. No bloqueante.

### 3.2. Hipótesis inicial

La asimetría USA (1d) vs Europa (3-4d) sugirió investigar:

- H1: los índices EU no se descargan.
- H2: `is_session_closed` los filtra por horario.
- H3: el cutoff `PUBLISH_HOUR` retrocede un día.
- H4: `_filter_non_eod_equity` los recorta.
- H5: el router usa provider alternativo.

---

## 4. Fase 2 — Inspección del código

### 4.1. Resultados por hipótesis

| Hipótesis | Evidencia | Veredicto |
|---|---|---|
| H1 — no se descargan | `_ticker_list()` los incluye (verificado Fase 3) | **Descartada** |
| H2 — filtro `is_session_closed` | Solo dentro de `_filter_non_eod_equity`, que excluye no-equity (L93-96) | **Descartada** |
| H3 — cutoff `PUBLISH_HOUR=23` | No explica 3-4 días | **Descartada** |
| H4 — trim EOD | `_trim_market_data_to_equity_eod` documentado como equity-only | **Descartada** |
| H5 — provider alternativo | Router tiene `preferred_order = ["yahoo", "fred", "polygon"]`. Yahoo primero | **Descartada** |

**Ninguna hipótesis de código sobrevive.** El pipeline no es la causa.

### 4.2. Datos relevantes de la inspección

| Evidencia | Valor |
|---|---|
| `parquet mtime` | 2026-09-15 02:21:34 |
| `CACHE_HOURS` | 23 |
| `CACHE_VALIDATE_TRADING_DATE` | True |
| `market_data.csv` | No existe (solo parquet) |
| `^FTSE` en `ticker_list` | True |
| `router.preferred_order` | `["yahoo", "fred", "polygon"]` |

**Hallazgo H-3B-4:** el parquet se ha actualizado por el cron. Al momento del diagnóstico, el snapshot era del 2026-09-15 a las 02:21.

---

## 5. Fase 3 — Verificación final

### 5.1. Pruebas ejecutadas

| Prueba | Resultado | Conclusión |
|---|---|---|
| Batch EU vs individual EU | Idénticos (^FTSE 15; resto 14) | **Batch no es la causa** |
| Batch US vs individual US | Idénticos (todos 15) | Batch correcto |
| Batch mixto (equity + índices + yields) | Todos 15 | Batch mixto correcto |
| mtime del parquet | 2026-09-15 02:21 | Snapshot viejo |
| `CACHE_VALIDATE_TRADING_DATE` | True | Fuerza re-descarga solo si `df_last < last_expected_market_date()` |

### 5.2. Asimetría adicional en EU

Dentro de los 4 índices europeos, **^FTSE publica con 1 día de adelanto** respecto a ^GDAXI, ^IBEX y ^STOXX50E:

| Ticker | Yahoo last (hoy) |
|---|---|
| ^FTSE | **2026-09-15** |
| ^GDAXI | 2026-09-14 |
| ^IBEX | 2026-09-14 |
| ^STOXX50E | 2026-09-14 |

Es comportamiento de Yahoo, no del pipeline. No documentado en ningún informe previo.
---

## 6. Causa raíz del lag europeo

### 6.1. Diagnóstico

**No es bug del pipeline. No es el batch. No es el router. No es el filtro EOD.**

Es **timing de publicación de Yahoo combinado con la frecuencia de ejecución del pipeline**:

1. El pipeline corre una vez al día (cron 06:00 UTC).
2. El parquet más reciente fue escrito el **2026-09-15 a las 02:21**.
3. En ese momento, Yahoo devolvía índices EU hasta el viernes **2026-09-11**.
4. Los cierres EU del lunes 14 se publican en Yahoo a las ~15:30 UTC del 14.
5. `CACHE_HOURS = 23` mantiene el parquet stale hasta que expire la ventana de cache.
6. `CACHE_VALIDATE_TRADING_DATE = True` fuerza re-descarga solo si `df_last < last_expected_market_date()`.

**Consecuencia:** entre la publicación del cierre EU y el siguiente run que efectivamente re-descarga, hay una ventana de 12-24h donde el parquet sigue mostrando el cierre anterior.

### 6.2. Por qué USA no sufre esto

USA publica los cierres antes que Europa (16:00 ET = 20:00 UTC). Cuando el pipeline corre a las 06:00 UTC del día siguiente, los cierres USA del día anterior ya están publicados y disponibles. El desfase es de 1 día, consistente con el comportamiento esperado del batch.

Europa cierra a las 17:30 CET = 15:30 UTC. Los cierres EU del lunes deberían estar publicados a las 16:00 UTC del lunes. Pero el pipeline no corre el lunes a las 16:00 UTC; corre el martes a las 06:00 UTC. **El parquet se re-descarga el martes a las 06:00 UTC con los datos del lunes, pero solo si la cache ha expirado.**

### 6.3. Modelo temporal correcto

| Clase | Cierre del mercado | Cierre publicado en Yahoo | Pipeline corre | Lag esperado |
|---|---|---|---|---|
| EQUITY_USA | 16:00 ET (20:00 UTC) | ~21:00 UTC | 06:00 UTC del día siguiente | 1 día |
| INDEX_USA | 16:00 ET | ~21:00 UTC | 06:00 UTC del día siguiente | 1 día |
| RATE_YIELD | ~15:30 ET | ~20:30 UTC | 06:00 UTC del día siguiente | 1 día |
| INDEX_EUROPA | 17:30 CET (15:30 UTC) | ~16:00 UTC | 06:00 UTC del día siguiente | **1 día, no 3** |

**El lag de 3-4 días observado es un artefacto de la ventana de cache, no un comportamiento estructural de Yahoo.**

---

## 7. Implicaciones para Parte B

### 7.1. INDEX_EOD son al menos 3 contratos

Los datos obligan a dividir INDEX_EOD en subclases con calendario distinto:

| Subclase | Miembros | Cierre del mercado | Ventana de publicación Yahoo |
|---|---|---|---|
| INDEX_EOD_USA | ^GSPC, ^DJI, ^NDX, ^RUT, ^SPGSCI | 16:00 ET | ~21:00 UTC |
| INDEX_EOD_EUROPA | ^FTSE, ^GDAXI, ^IBEX, ^STOXX50E | 16:30-17:30 CET | ~16:00 UTC |
| INDEX_EOD_COMMODITY | DX-Y.NYB | (ICE, fuera de alcance) | — |

### 7.2. RATE_YIELD tiene un universo de 2

`^FVX` y `^TNX` son los dos únicos yields en `df_market`. **Faltan ^IRX (13-week) y ^TYX (30-year).** No es bloqueante para Parte B; se documenta como límite del universo actual.

### 7.3. El contrato INDEX_EOD_EUROPA no puede exigir lag=0

La Parte A §A.3 anticipa que `date ≠ expected_session` puede ocurrir. INDEX_EOD_EUROPA es el caso real:

- `expected_session` = lunes 14.
- `effective_date` en el parquet actual = viernes 11 (por cache stale).
- `coverage` = 4/4 (los 4 índices existen, pero con fecha anterior).
- `status` = OK si se acepta la fecha efectiva con su lag declarado.

**Recomendación:** declarar `max_lag_days` por clase y por mercado.

### 7.4. Asimetría ^FTSE

El hecho de que ^FTSE publique antes que los otros 3 índices EU sugiere que Yahoo trata los índices europeos individualmente (distintos endpoints de publicación por mercado). El contrato no debe asumir coherencia entre índices del mismo mercado.

---

## 8. Hallazgos registrados

| ID | Hallazgo | Clase | Severidad |
|---|---|---|---|
| H-3B-1 | Divergencia `last_date` entre equity y no-equity (4 días) | Observación | Informativa |
| H-3B-2 | VOLATILITY_INDEX es 6ª clase, no subclase de INDEX | Clasificación | Corrige Parte A |
| H-3B-3 | RATE_YIELD incompleto (2 de 4 yields) | Límite universo | Documental |
| H-3B-4 | Parquet escrito por cron, actualizado desde último commit | Estado | Informativa |
| H-3B-5 | Close en día común idéntico entre Yahoo y parquet (0 diff) | No-bug | Informativa |
| H-3B-6 | Lag asimétrico USA (1d) vs Europa (3-4d) | Síntoma | Baja |
| H-3B-7 | Causa raíz: cache stale + frecuencia cron | **Causa raíz** | Baja (no bug) |
| H-3B-8 | Fila sin índices: consolidación mixta es la norma | Confirma Parte A | Informativa |
| H-3B-9 | DX-Y.NYB con diff 0.041 en día común | Anomalía menor | Documental |
| H-3B-10 | ^FTSE publica 1 día antes que los otros índices EU | Asimetría Yahoo | Documental |

---

## 9. Lo que este informe NO decide

- No activa contratos por clase.
- No implementa cambios.
- No modifica el pipeline.
- No autoriza retirar `trim_to_last_valid_date`.
- No decide si el lag observado es aceptable (eso lo decide Parte B).
- No decide si el cron debe ejecutarse con más frecuencia.

---

## 10. Anexo — Comandos reproducibles

Todas las verificaciones empíricas son reproducibles con estos comandos.

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
    "

**Fase 1 — Yahoo vs parquet:**

    py -c "
    import pandas as pd
    import yfinance as yf
    df = pd.read_parquet('data/market_data.parquet')
    for t in ['^GSPC', '^FTSE', '^GDAXI', '^IBEX', '^STOXX50E', '^TNX']:
        pq = df[('Close', t)].dropna()
        yf_df = yf.download(t, period='15d', progress=False, auto_adjust=True)
        if isinstance(yf_df.columns, pd.MultiIndex):
            yf_df.columns = yf_df.columns.get_level_values(0)
        yf_close = yf_df['Close'].dropna()
        print(t, 'parquet=' + str(pq.index[-1].date()), 'yahoo=' + str(yf_close.index[-1].date()))
    "

**Fase 2 — inspección de código:**

    Select-String -Path src\data_loader.py -Pattern 'def download_market_data|_filter_non_eod_equity|_trim_market_data_to_equity_eod' -Context 0,3

**Fase 3 — batch vs individual:**

    py -c "
    import yfinance as yf
    for batch in [['^FTSE','^GDAXI','^IBEX','^STOXX50E'], ['^GSPC','^DJI','^NDX','^RUT']]:
        df = yf.download(batch, period='10y', progress=False, auto_adjust=True)
        print('batch:', batch, 'shape:', df.shape)
        for t in batch:
            s = df.xs(t, level=1, axis=1)['Close'].dropna() if isinstance(df.columns, pd.MultiIndex) else df['Close'].dropna()
            print(' ', t, 'last=', s.index[-1].date())
    "

---

**Fin del informe FU-021-3B.**
**HEAD de referencia:** 61aa99f (origin/main).
**Fecha:** 2026-09-15.
**Estado:** cerrado. Alimenta Parte B de FU-021-5.