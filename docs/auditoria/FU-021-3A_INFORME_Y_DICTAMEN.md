# FU-021-3A — Informe tecnico y dictamen del auditor externo

**Fecha:** 2026-09-15
**Estado:** RESUELTO
**Commits de referencia:** 78b7583 (fix), 747cfb1 (merge rama), fded14b (merge origin/main)
**HEAD final:** fded14b (origin/main)

---

## 1. Pregunta al auditor

En el run de `daily_run.yml` del 2026-09-15 a las 17:04 UTC (13:04 ET, sesion USA
abierta), el filtro `EQUITY_EOD` de FU-021-3A acepto una fila intradia como fecha
efectiva (`effective=2026-09-15`, `lag=0d`, sin trim). El manifest marco
`status=VALID` con `last_date=2026-09-15` y `expected=2026-09-14`.

Pregunta: es este comportamiento (a) defecto de implementacion de FU-021-3A,
(b) defecto de diseno de `resolve_effective_date` fuera de su contexto original,
(c) vacio de la regla `status` del manifest FU-002, (d) combinacion, o
(e) comportamiento esperado? Y en su caso, cual es la correccion que no rompe
FU-018 ni FU-020 Fase 1?

---

## 2. Contexto: dos pipelines temporales asimetricos

El sistema tiene dos rutas de ingesta de mercado que comparten modulos pero
aplican controles temporales distintos.

**Ruta A — `stock_prices.parquet` (313 tickers equities):**

    YahooProvider (raw)
      -> stock_data_loader.download_stock_prices()
      -> FU-018: filtro de sesion por batch (_filter_non_eod_batch)
      -> pipeline/leaders.compute_leaders()
      -> FU-020 Fase 1: resolve_effective_date (cobertura >= 90%)
      -> write_artifact_with_manifest()

**Ruta B — `market_data.parquet` (562 tickers mixtos):**

    YahooProvider (raw)
      -> data_loader.download_market_data()
      -> concat de lotes
      -> clean_oil_prices()
      -> FU-021-3A: _trim_market_data_to_equity_eod
      -> write_artifact_with_manifest()
    data_load.load_all_data()
      -> trim_to_last_valid_date (min_coverage=0.5, sin calendario)
      -> validate_market_data()
      -> df_market a 13 modulos pipeline

Diferencia estructural: Ruta A tiene doble candado temporal (FU-018 + FU-020).
Ruta B tenia candado unico (FU-021-3A) mas un trim adicional ciego al calendario.

---

## 3. Hechos observados

### 3.1 Run 2026-09-15T17:04Z (baseline, pre-fix)

Hora de inicio: 17:04:23Z. Descarga: 17:13:23Z. Equivalente ET: 13:13 ET.
Sesion USA abierta.

    [FU-021-3A] EQUITY_EOD effective=2026-09-15 requested=2026-09-15 lag=0d coverage=100.00% (539/539)

    [MANIFEST] data/market_data.parquet:
        status=VALID
        rows=2605, tickers=562
        pct_dup=0.002, close_nan=0, n_close_observed=562
        coverage_pct=1.0000
        last_date=2026-09-15, expected=2026-09-14
        last_date_is_expected_session=False

Lectura mecanica:
- `lag=0d` implica `effective_date == requested_date == 2026-09-15`.
- `requested_date` = `prices.index[-1]` = ultima fila del DataFrame concatenado.
- El manifest detecta la anomalia (`last_date_is_expected_session=False`) pero la
  regla FU-002 no dispara INVALID porque solo lo hace cuando el valor es `True`.

### 3.2 Criterio de aceptacion del dictamen previo

Para el mismo escenario (sesion USA abierta):

    ANTES
      requested = 2026-09-15
      effective = 2026-09-15
      lag       = 0d
      last_date = 2026-09-15

    DESPUES (criterio reformulado en dictamen v2)
      requested       = 2026-09-14
      effective       = 2026-09-14
      function_lag    = 0d
      reference_lag   = 1d
      last_date       = 2026-09-14
      last_date_is_expected_session = True

---

## 4. Codigo relevante verificado

### 4.1 `src/effective_date.py::resolve_effective_date`

Lineas decisivas:

    requested_date = prices.index[-1]
    coverage_series = close_df[present].notna().sum(axis=1) / n_eligible
    valid_rows = coverage_series[coverage_series >= min_coverage]
    effective_date = valid_rows.index[-1]
    lag_days = (pd.Timestamp(requested_date) - pd.Timestamp(effective_date)).days

- No importa `src.market_calendar`. No consulta calendario.
- No recibe `reference_date`. Su nocion de "hoy" es `prices.index[-1]`.
- La seleccion es por cobertura de NaN, no por sesion bursatil.

Conclusion: la funcion es correcta en su contrato ("ultima fecha con cobertura
suficiente"). No responde a "ultima sesion cerrada con Close EOD".

### 4.2 `src/market_hours.py` (FU-018)

    is_trading_session(market, session_date)
    get_session_close(market, session_date)
    is_session_closed(market, session_date, reference_date)

`reference_date` debe ser tz-aware. `config/market_close_regular.csv` cubre
US_EQUITY (16:00 NY), LSE (16:30 London), XETRA/BME/EURONEXT (17:30 CET).
`config/market_close_exceptions.csv` vacio.

### 4.3 `src/stock_data_loader.py::_filter_non_eod_batch` (FU-018)

Politica a nivel de batch: si ALGUN ticker del batch esta en mercado abierto
o tiene mercado desconocido, se elimina la ultima fila completa del batch.
Invocado antes de `all_data.append`, y en retry individual.

### 4.4 `src/instrument_registry.py::get_market`

    if ticker.endswith(".L"):    return "LSE"
    if ticker.endswith(".DE"):   return "XETRA"
    if ticker.endswith(".MC"):   return "BME"
    if ticker.endswith(".PA/.AS/.MI"): return "EURONEXT"
    if "." not in ticker:        return "US_EQUITY"
    return "UNKNOWN"

Hallazgo critico: futuros (`CL=F`, `BZ=F`, ...), indices no-USA (`^FTSE`,
`^GDAXI`, ...) e FX (`EURUSD=X`, ...) reciben `US_EQUITY`. Solo `DX-Y.NYB`
cae en UNKNOWN. 22 de 23 tickers no-equity mal clasificados.

Esto invalida el "port directo de FU-018" a `market_data`: aplicar la logica
`US_EQUITY` a futuros/FX/indices no-USA seria semanticamente incorrecto.

### 4.5 `src/utils.py::trim_to_last_valid_date`

Tercer trim en la Ruta B. Aplicado en `data_load.py:27` despues de FU-021-3A.
`min_coverage=0.5`. Ciego al calendario. Puede hacer que `df_market` que
consumen los 13 modulos de pipeline no sea identico al parquet publicado.

### 4.6 `run.py:41`

    reference_date = datetime.now(ZoneInfo("Europe/Madrid"))

Llega propagado hasta `download_market_data`. `is_session_closed` puede
recibirlo sin ValueError.

---

## 5. Causa raiz

Defecto principal: FU-021-3A replico el control de cobertura de FU-020 pero
no el control de sesion de FU-018. El diseno FU-021-2 lo describia como
"unificar la ruta: market_data debe aplicar el mismo filtro EOD que stock_prices".
Ese filtro EOD son dos capas (FU-018 + FU-020), y se porto una sola.

Defecto secundario: la regla `status=INVALID` de FU-002 solo dispara si
`last_date_is_expected_session == True`. Cuando es `False` (fila intradia),
la regla no actua. Deteccion sin bloqueo.

Defecto terciario (descubierto en la revision del codigo): `get_market()`
clasifica 22 de 23 tickers no-equity como `US_EQUITY`. Cualquier port directo
de FU-018 a market_data habria aplicado semantica NYSE a futuros, indices
no-USA y FX.

---

## 6. Dictamen del auditor externo — respuestas a A1-A4

Resultado: `GO CONDICIONADO` para la correccion. El informe original cambio
una conclusion importante del dictamen previo: no autoriza un "port directo
de FU-018", por el hallazgo de `get_market`. Solucion aprobada: separar tres
capas (clasificacion -> sesion/EOD de la clase -> resolucion por cobertura).

### P1 — A1: semantica de `requested_date` / `lag_days`

Dictamen: **A1.2**.

Mantener `resolve_effective_date` intacta (contrato y `requested_date`). No
usar su `lag_days` como indicador de retraso respecto al run. Añadir en el
caller una metrica distinta:

    reference_lag_days = (reference_date.date() - effective_date.date()).days

Log con ambos valores, con nombres que no se confundan:
`function_lag` y `reference_lag`.

### P2 — A2: alcance del filtro FU-018

Dictamen: **A2.2**.

Aplicar el mecanismo FU-018 SOLO al universo EQUITY_EOD (539 tickers).
No-equity (INDEX_EOD, RATE_YIELD, FUTURE_SETTLEMENT, FX_DAILY_CUT) queda
fuera, pendiente de fases 3B/3C. No usar `get_market()` indiscriminadamente
para deducir el universo equity. A2.3 (corregir `get_market`) queda como
deuda independiente.

### P3 — A3: tercer trim `trim_to_last_valid_date`

Dictamen: **A3.1** aprobado como destino arquitectonico, **ejecucion
diferida** hasta auditoria de consumidores. No recomienda A3.2 (no-op) ni
A3.3 (dejar y documentar). La retirada se ejecutara tras el sub-informe
de los 13 modulos consumidores.

### P4 — A4: secuencia canonica

Dictamen: **modificada**. Todo trim antes del manifest. Consumidores no
vuelven a decidir. Secuencia:

    clasificacion por contrato
      -> filtro EOD de la clase aplicable
      -> resolve_effective_date (cobertura >= 90%)
      -> trim final productor
      -> artefacto definitivo
      -> MANIFEST
      -> consumidores

### P5 — Consumidores antes o despues

Dividir en 3 fases:
- Fase 1 — correccion minima (A1.2 + A2.2). Ejecutada en este ciclo.
- Fase 2 — auditoria de los 13 consumidores. Pendiente.
- Fase 3 — retirada de `trim_to_last_valid_date` (A3.1). Diferida.

### P6 — Criterio de aceptacion E2E reformulado

No exigir `resolve_effective_date.lag_days == 1`. Exigir:

    effective       = 2026-09-14
    reference_lag   = 1d
    last_date       = 2026-09-14
    last_date_is_expected_session = True

`function_lag = 0d` es aceptable.

---

## 7. Arquitectura aprobada

    Yahoo
      |
      v
    market_data raw
      |
      +-- 539 EQUITY_EOD
      |     |
      |     +-- filtro sesion/EOD FU-018 (solo equity)
      |     +-- resolve_effective_date >= 90%
      |
      +-- 23 instrumentos NO-EQUITY
            |
            +-- NO tocar todavia (3B/3C pendientes)
      |
      v
    market_data.parquet definitivo
      |
      v
    FU-002 manifest
      |
      v
    13 consumidores

Clasificacion temporal y semantica economica separadas.

---

## 8. Patch autorizado y aplicado

### 8.1 Cambios en `src/data_loader.py` (5 cambios, +73/-2)

1. Import `from zoneinfo import ZoneInfo` y
   `from src.market_hours import is_trading_session, is_session_closed`
   y `from src.instrument_registry import get_market`.

2. Nueva funcion `_filter_non_eod_equity(data, reference_date)`:
   aplica mecanismo FU-018 (sesion/EOD) SOLO al universo EQUITY_EOD.
   Politica conservadora: si algun equity con dato en la ultima fecha
   tiene observacion no EOD, elimina la ultima fila completa. No-equity
   no se inspecciona.

3. Fallback `reference_date = datetime.now(ZoneInfo("Europe/Madrid"))`
   cuando no se pasa argumento. Coherente con FU-018-3c.

4. Llamada al filtro insertada antes de `_trim_market_data_to_equity_eod`:

        data, _eod_info = _filter_non_eod_equity(data, reference_date)

5. Log ampliado con `function_lag` y `reference_lag` (antes solo `lag`):

        [FU-021-3A] EQUITY_EOD effective=... requested=... function_lag=Xd reference_lag=Yd coverage=... (n/m)

### 8.2 Tests nuevos — `tests/test_fu021_3a_equity_eod.py` (+114/-1)

5 tests:
- `test_filter_non_eod_equity_usa_open_trims`
- `test_filter_non_eod_equity_usa_closed_keeps`
- `test_filter_non_eod_equity_ignores_non_equity_only`
- `test_filter_non_eod_equity_100pct_intraday_still_trims` (test clave del auditor: 100% cobertura != EOD)
- `test_reference_lag_differs_from_function_lag`

### 8.3 No toca

- `resolve_effective_date` (intacta, dictamen explicito)
- `market_hours.py`, `instrument_registry.py`, `market_calendar.py`
- `run.py`
- `pipeline/leaders.py`, `pipeline/data_load.py`
- `trim_to_last_valid_date` (A3.1 diferido)
- Manifest FU-002 (va a FU-002-bis)

---

## 9. Verificacion E2E

### 9.1 Verificacion local (post-patch)

- `compileall`: OK
- `pyflakes`: 0 warnings
- `pytest tests/ validation/`: **334 passed + 2 skipped** (317 previos + 12 FU-021-3A + 5 nuevos)

### 9.2 Run 18:04Z sobre `fix/fu021-3a-v2`

    [FU-021-3A] EQUITY_EOD: 539/539 equity con vela no EOD en 2026-09-15 -> eliminar ultima fila
    [FU-021-3A] EQUITY_EOD effective=2026-09-14 requested=2026-09-14 function_lag=0d reference_lag=1d coverage=99.81% (538/539)

    [MANIFEST] data/market_data.parquet:
        status=VALID_WITH_MISSING
        last_date=2026-09-14, expected=2026-09-14
        last_date_is_expected_session=True

    [FU-020] effective=2026-09-14 requested=2026-09-15 lag=1d coverage=100.00% (313/313)

    VALIDATION GATE: Sin errores (10 comprobaciones OK)

### 9.3 Run 18:20Z sobre `main` (post-merge)

Identico al 18:04Z. Tres criterios del dictamen cumplidos:
- Linea `-> eliminar ultima fila` presente.
- Formato `function_lag` / `reference_lag` con valores `0d` y `1d`.
- Manifest con `last_date_is_expected_session=True`.

Cobertura 538/539 = 99.81% > 90%. `status=VALID_WITH_MISSING` coherente
con `close_nan=2` del manifest. No es desviacion del fix.

### 9.4 Nota sobre el alcance de la verificacion

Los runs 17:04Z y 17:51Z fueron lanzados desde `main` (rama por defecto en
el desplegable `Use workflow from`). Mostraron el comportamiento pre-fix.
Los runs 18:04Z (rama aislada) y 18:20Z (main con fix) validan el patch.
Criterio del auditor cumplido sobre el codigo real que ejecutan los crons.

---

## 10. Estado del expediente

    FU-021-3A (implementacion original)   NO-GO (dictamen previo)
    Causa raiz                            Confirmada
    resolve_effective_date                Conservada intacta
    A1                                    A1.2 aplicada
    A2                                    A2.2 aplicada
    A3                                    A3.1 aprobada, diferida
    A4                                    Secuencia corregida
    E2E                                   Verificada
    Estado final                          RESUELTO 2026-09-15

---

## 11. Cola viva derivada

- **FU-002-bis** — Endurecer regla INVALID para `last_date > expected_session`.
- **FU-021-5** — Metadata temporal por clase en manifest (esquema JSON).
- **FU-021-3B** — INDEX_EOD + RATE_YIELD. Requiere verificacion empirica
  del Close de Yahoo para 13 indices y 2 yields.
- **FU-021-3C** — FUTURE_SETTLEMENT + FX_DAILY_CUT. Bloqueado sin provider
  con semantica verificada.
- **A2.3 (deuda registry)** — Corregir `get_market` para `^`, `=F`, `=X`.
- **A3.1** — Retirar `trim_to_last_valid_date` de `data_load.py`. Diferido
  hasta sub-informe de los 13 consumidores.
- **Gap manifest `stock_prices.parquet`** — El parquet publicado tiene
  `last_date=2026-09-15` mientras `pipeline/leaders` recorta a 14/09 via
  FU-020. Anotado para FU-002-bis o posterior.

---

**Fin del informe.** Commit de referencia: `fded14b`. Fecha: 2026-09-15.
