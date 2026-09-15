# FU-021-5 — Especificación del contrato temporal de `df_market`

**Versión:** v2 (unificada)
**Fecha:** 2026-09-16
**HEAD de referencia:** 0eb7e4f (origin/main)
**Ratificado por:** Dictamen externo FU-021-5 Parte B (2026-09-16)
**Documentos base:** FU-021-3A + Adendums 1 y 2, Ciclo 2, FU-021-3B, FU-021-3B-bis, FU-021-3C, Parte A v1 (61aa99f), Parte B (468fc08)
**Estado:** ratificado con correcciones. Pendiente de implementación.

---

## 0. Alcance y delimitación

### 0.1. Alcance

Especificar el contrato temporal de `df_market` conforme al dictamen del contrato temporal y al dictamen de Parte B:

- taxonomía de familias, subclases y contratos;
- semántica operativa de cada contrato (efectiva, esperada, lag, cobertura, estado);
- regla de consolidación de un DataFrame mixto;
- metadata de fecha y cobertura por artefacto;
- contrato de writers históricos;
- propagación hacia consumidores;
- rediseño del estado persistente de MTE;
- migración de `darkpool`;
- máquina de estados (FSM) formal.

### 0.2. Lo que este documento NO es

- No es un patch.
- No activa contratos.
- No retira `trim_to_last_valid_date` (A3.1 sigue bloqueada).
- No modifica el informe original FU-021-3A ni sus adendos.
- No toca código.

### 0.3. Correcciones respecto a la v1

La v1 (61aa99f, corregida en e54bab8) declaraba "5 clases" (→ 6 tras corrección). El dictamen de Parte B exige reformular la taxonomía:

| Corrección | Motivo |
|---|---|
| "6 clases" → "familias + subclases + contratos" | La tabla operativa contiene 9 unidades, no 6 |
| `DX-Y.NYB` → `INDEX_EOD_CURRENCY` | Es índice de divisas, no commodity |
| `^SPGSCI` → `INDEX_EOD_COMMODITY` | Es índice commodity genuino |
| "30-40% del pipeline usa futuros" → "5 de 562 instrumentos (~0.9%)" | Dependencia funcional ≠ peso del instrumento |
| Añadido `STALE` como estado formal | Ratificado por Q-B.1 |
| Añadidos `per_pair_max_lag`, `per_ticker_lag`, `activation_requirement` | Ratificados por Q-B.2, Q-B.3, Q-B.4 |
| FUTURE_SETTLEMENT = `BLOCKED` | Ratificado por Q-B.4 |
| Máquina de estados formal (FSM) | Ratificada por dictamen §"Sobre STALE y BLOCKED" |

### 0.4. Terminología

Nomenclatura oficial (consolidada en Adendum 2 + dictamen Parte B):

| Categoría | Cuenta | Definición |
|---|---|---|
| Productor | 1 | `src/pipeline/data_load.py` |
| Consumidor directo (Capa 1) | 12 | `src/pipeline/*` |
| Consumidor indirecto Capa 2a | 3 | `regimes/*` |
| Consumidor indirecto Capa 2b | 15 | `indicators/*` |
| Consumidor de artefacto | 1 | `indicators/darkpool.py` |
| Orquestador | 1 | `run.py` |

**Consumidores por firma: 30.** Consumidores totales (incluyendo artefacto): 31.

---

## 1. Principios arquitectónicos

### 1.1. Principios ratificados

1. **Contratos por subconjunto homogéneo, no un contrato agregado.** `df_market` no tiene una única "última fecha correcta".
2. **Familias ≠ subclases ≠ contratos.** La taxonomía debe ser explícita (ver §A.1).
3. **Consolidación explícita.** El DataFrame consolidado conoce qué `effective_date` corresponde a cada grupo.
4. **Metadata cuádruple.** Los artefactos distinguen `date`, `effective_date`, `expected_date`, `coverage`.
5. **Fuente canónica única.** `df_market` en memoria es la única fuente intra-run.
6. **Responsabilidad temporal única.** La resolución temporal existe una sola vez, en el productor.
7. **Estado versionado.** El estado persistente de MTE lleva contrato, versión y fecha efectiva.
8. **FSM formal.** Todo contrato expone `status` según máquina de estados explícita (§A.10).

### 1.2. Principios que NO cambian

- Determinismo. Sin ML, sin optimización.
- Sin datos ficticios: `N/D` antes que imputar.
- R1–R4 (FU-020) intactas.
- `resolve_effective_date` intacta.
- Manifest FU-002 intacto.
- B2 (`is_market_day`) sigue siendo el único control de sesión.
- A3.1 bloqueada hasta completar las 6 condiciones del dictamen previo.

### 1.3. Interfaz conceptual
RAW
│
▼
CLASSIFICATION (get_instrument_class)
│
├── Familias (5): EQUITY, INDEX, RATE_YIELD, FUTURE, FX
│
▼
CONTRATOS (9):
├── EQUITY_EOD
├── INDEX_EOD_USA
├── INDEX_EOD_EUROPA
├── INDEX_EOD_COMMODITY
├── INDEX_EOD_CURRENCY
├── VOLATILITY_INDEX
├── RATE_YIELD
├── FUTURE_SETTLEMENT (BLOCKED)
└── FX_DAILY_CUT
│
▼
TEMPORAL RESOLUTION POR CONTRATO
│ (effective_date, expected_date, lag_days, coverage, status)
▼
CONSOLIDATION
│ (df_market conoce qué fecha corresponde a cada contrato)
▼
df_market CONSOLIDADO
│
▼
CONSUMIDORES (30)
│
├── writers históricos (20)
├── MTE state
└── report

```

---

## 2. §A.1 — Taxonomía: familias, subclases y contratos

### 2.1. Definiciones

**Familia.** Categoría económica amplia (equity, index, yield, future, fx). No tiene semántica temporal propia.

**Subclase.** Agrupación dentro de una familia con calendario común (por ejemplo, índices USA vs índices Europa).

**Contrato.** Unidad operativa del contrato temporal. Es lo que se resuelve, lo que se declara, lo que los consumidores consumen.

**Regla:** cada contrato produce un resultado de resolución temporal completo:
effective_date
expected_date
lag_days
coverage
status

```

### 2.2. Familia EQUITY

**Contrato derivado:** `EQUITY_EOD`

**Universo:** 539 tickers (acciones + ETFs USA).

**Estado:** cerrado por FU-021-3A (commit `78b7583` + `747cfb1` + `fded14b`).

**Semántica:**
- Fecha efectiva = última sesión con cierre EOD.
- Calendario: NYSE.
- Filtro de sesión (FU-018) + resolución por cobertura (FU-020).
- `min_coverage = 0.90`.

**Interfaz:**
contract EQUITY_EOD:
family : EQUITY
eligible_universe : list[ticker] # 539
min_coverage : 0.90
session_calendar : "NYSE"
max_lag_days : 0
filter_function : _filter_non_eod_equity # FU-021-3A
resolver : resolve_effective_date # FU-020
outputs:
effective_date : date
expected_date : date
lag_days : int
coverage : float [0,1]
status : OK | STALE | INSUFFICIENT | BLOCKED | PENDING

```

**Invariantes:**
- `effective_date <= expected_date`.
- `coverage >= 0.90` para status `OK` o `STALE`.
- Ningún registro con fecha no bursátil.

### 2.3. Familia INDEX

**Subclases:** 4 (USA, EUROPA, COMMODITY, CURRENCY).

**Contratos derivados:** 4.

#### 2.3.1. INDEX_EOD_USA

**Universo:** `^GSPC`, `^DJI`, `^NDX`, `^RUT`.

**Nota (corrección v2):** `^SPGSCI` se ha movido a `INDEX_EOD_COMMODITY`.

**Semántica:**
- Calendario: NYSE.
- Cierre: 16:00 ET (20:00 UTC).
- Publicación Yahoo: ~21:00 UTC.
- Lag típico observado (FU-021-3B §3.1): 1 día.

**Interfaz:**
contract INDEX_EOD_USA:
family : INDEX
eligible_universe : [^GSPC, ^DJI, ^NDX, ^RUT]
session_calendar : "NYSE"
max_lag_days : 1
min_coverage : 1.0
per_ticker_lag : {
"^GSPC": 1,
"^DJI": 1,
"^NDX": 1,
"^RUT": 1,
}
outputs: (effective_date, expected_date, lag_days, coverage, status)

```

**Evidencia empírica (FU-021-3B §3.1):** todos los miembros con `last_date = 2026-09-14` en parquet; Yahoo `2026-09-15`; valores Close idénticos en día común (diff ≤ 0.003); lag = 1 día uniforme.

#### 2.3.2. INDEX_EOD_EUROPA

**Universo:** `^FTSE`, `^GDAXI`, `^IBEX`, `^STOXX50E`.

**Semántica:**
- Calendario: LSE (^FTSE), XETRA (^GDAXI), BME (^IBEX), Euronext (^STOXX50E).
- Cierre: 16:30-17:30 CET (14:30-15:30 UTC).
- Lag típico observado (FU-021-3B §3.1): 3-4 días por artefacto de cache. Estructural: 1 día.

**Interfaz:**
contract INDEX_EOD_EUROPA:
family : INDEX
eligible_universe : [^FTSE, ^GDAXI, ^IBEX, ^STOXX50E]
session_calendar : "LSE|XETRA|BME|EURONEXT"
max_lag_days : 5
min_coverage : 1.0
per_ticker_lag : {
"^FTSE": 1,
"^GDAXI": 2,
"^IBEX": 2,
"^STOXX50E": 2,
}
outputs: (effective_date, expected_date, lag_days, coverage, status)

```

**Evidencia empírica (FU-021-3B §5.2):**

| Ticker | Parquet last | Yahoo last | Lag |
|---|---|---|---|
| ^FTSE | 2026-09-11 | 2026-09-15 | 4d |
| ^GDAXI | 2026-09-11 | 2026-09-14 | 3d |
| ^IBEX | 2026-09-11 | 2026-09-14 | 3d |
| ^STOXX50E | 2026-09-11 | 2026-09-14 | 3d |

**Hallazgo H-3B-10:** `^FTSE` publica 1 día antes. `per_ticker_lag` registra este comportamiento esperado.

**Causa raíz del lag observado (FU-021-3B §6):** artefacto de la ventana de cache (`CACHE_HOURS=23`) + frecuencia del cron. No es bug.

#### 2.3.3. INDEX_EOD_COMMODITY

**Universo:** `^SPGSCI` (S&P GSCI).

**Nota (corrección v2):** `DX-Y.NYB` se ha movido a `INDEX_EOD_CURRENCY`. `^SPGSCI` sale de `INDEX_EOD_USA`.

**Semántica:**
- Índice de materias primas.
- Calendario: referencia bursátil USA para la publicación del índice.
- Lag esperado: 1 día.

**Interfaz:**
contract INDEX_EOD_COMMODITY:
family : INDEX
eligible_universe : [^SPGSCI]
session_calendar : "NYSE"
max_lag_days : 1
min_coverage : 1.0
per_ticker_lag : {"^SPGSCI": 1}
outputs: (effective_date, expected_date, lag_days, coverage, status)

```

#### 2.3.4. INDEX_EOD_CURRENCY

**Universo:** `DX-Y.NYB` (US Dollar Index, ICE).

**Nota (corrección v2):** renombrado desde `INDEX_EOD_COMMODITY` por dictamen. `DX-Y.NYB` es un índice de divisas, no un commodity.

**Semántica:**
- Calendario: ICE.
- Cierre: 17:00 ET (21:00 UTC).
- Lag típico: 1 día.

**Interfaz:**
contract INDEX_EOD_CURRENCY:
family : INDEX
eligible_universe : [DX-Y.NYB]
session_calendar : "ICE"
max_lag_days : 1
min_coverage : 1.0
per_ticker_lag : {"DX-Y.NYB": 1}
outputs: (effective_date, expected_date, lag_days, coverage, status)

```

**Hallazgo H-3B-9:** diff de 0.041 en día común (frente a 0.002-0.003 de índices puros). Característica conocida del índice de divisas.

### 2.4. Familia RATE_YIELD

**Contrato derivado:** `RATE_YIELD`

**Universo:** `^FVX`, `^TNX`.

**Semántica:**
- Fuente: US Treasury.
- Publicación: tras cierre del mercado de bonos (~15:30 ET).
- Lag típico: 1 día.

**Interfaz:**
contract RATE_YIELD:
family : RATE_YIELD
eligible_universe : [^FVX, ^TNX]
session_calendar : "NYSE"
max_lag_days : 1
min_coverage : 1.0
publication_source : "US_TREASURY"
per_ticker_lag : {
"^FVX": 1,
"^TNX": 1,
}
outputs: (effective_date, expected_date, lag_days, coverage, status)

```

**Hallazgo H-3B-3:** universo limitado a 2 yields. Faltan `^IRX` y `^TYX`. No bloqueante.

### 2.5. Familia VOLATILITY (subclase de INDEX)

**Contrato derivado:** `VOLATILITY_INDEX`

**Universo:** `^VIX`, `^VIX3M`, `^VXN`.

**Semántica:**
- Calendario: CBOE (mismo calendario bursátil USA).
- Cierre: 16:15 ET.
- Hereda parámetros de `INDEX_EOD_USA`.

**Interfaz:**
contract VOLATILITY_INDEX:
family : INDEX
subclase_de : INDEX_EOD_USA (delegación)
eligible_universe : [^VIX, ^VIX3M, ^VXN]
session_calendar : "NYSE"
max_lag_days : 1
min_coverage : 1.0
per_ticker_lag : {
"^VIX": 1,
"^VIX3M": 1,
"^VXN": 1,
}
outputs: (effective_date, expected_date, lag_days, coverage, status)

```

**Evidencia empírica (FU-021-3B-bis §3):** comportamiento idéntico a INDEX_EOD_USA; lag 1 día; valores Close idénticos en día común.

**Nota especial ^VIX3M (H-3B-bis-2):** descarga individual `yfinance` devuelve 1 fila. Descarga en batch funciona. No es problema del pipeline.

### 2.6. Familia FUTURE

**Contrato derivado:** `FUTURE_SETTLEMENT`

**Universo:** `BZ=F`, `CL=F`, `GC=F`, `HG=F`, `NG=F`.

**Estado:** `BLOCKED` por dictamen Q-B.4.

**Semántica:**
- Fuente actual: Yahoo continuo.
- **No auditable:** el ticker continuo no identifica un contrato concreto. Yahoo reescribe los históricos de forma no determinista.

**Interfaz:**
contract FUTURE_SETTLEMENT:
family : FUTURE
eligible_universe : [BZ=F, CL=F, GC=F, HG=F, NG=F]
settlement_source : "CME|ICE"
max_lag_days : N/A
min_coverage : N/A
status : BLOCKED
activation_requirement:

provider que exponga contrato específico

settlement oficial, no close intradía

identificador estable por contrato

lógica de rollover declarada
outputs: (effective_date, expected_date, lag_days, coverage, status)

```

**Evidencia empírica (FU-021-3C §3.1, §5.2):** los valores Close del parquet y Yahoo divergen en la misma fecha (0.56%-0.92% relativo). No es redondeo. La investigación descartó rollover retroactivo puro, ajuste por rango, `auto_adjust`, snapshot intradía. Hipótesis residual no verificable: Yahoo reescribe su histórico entre capturas.

**Impacto en consumidores (corrección v2):** 5 de 562 instrumentos (~0.9%) del universo. Dependencia funcional en 5 consumidores: `flows_primary`, `mte_confirmation`, `sectors_base`, `macro_regime`, `mte`. **Peso del instrumento ≠ dependencia funcional.**

### 2.7. Familia FX

**Contrato derivado:** `FX_DAILY_CUT`

**Universo:** `EURUSD=X`, `USDCNY=X`, `USDJPY=X`.

**Semántica:**
- Mercado: FX spot 24h.
- Cutoff: 17:00 ET.
- Lag heterogéneo por par.

**Interfaz:**
contract FX_DAILY_CUT:
family : FX
eligible_universe : [EURUSD=X, USDCNY=X, USDJPY=X]
session_calendar : "FX_24h"
cutoff_time : "17:00 America/New_York"
max_lag_days : dict por par
min_coverage : 0.5
per_pair_max_lag : {
"EURUSD=X": 0,
"USDJPY=X": 0,
"USDCNY=X": 1,
}
per_pair_expected_lag : {
"EURUSD=X": 0,
"USDJPY=X": 0,
"USDCNY=X": 1,
}
outputs: (effective_date, expected_date, lag_days, coverage, status)

```

**Evidencia empírica (FU-021-3C §2.2, §3.2):**

| Par | Parquet last | Yahoo last | Lag |
|---|---|---|---|
| EURUSD | 2026-09-15 | 2026-09-15 | 0d |
| USDCNY | 2026-09-14 | 2026-09-15 | 1d |
| USDJPY | 2026-09-15 | 2026-09-15 | 0d |

**Nota del dictamen Q-B.2:** `per_pair_max_lag` no es excusa para ocultar caídas. El contrato declara `expected`, `observed`, `max_tolerated` por separado.

### 2.8. Resumen de contratos

| # | Contrato | Familia | Universo | Estado |
|---|---|---|---|---|
| 1 | `EQUITY_EOD` | EQUITY | 539 | OK (activo) |
| 2 | `INDEX_EOD_USA` | INDEX | 4 | OK / STALE |
| 3 | `INDEX_EOD_EUROPA` | INDEX | 4 | OK / STALE |
| 4 | `INDEX_EOD_COMMODITY` | INDEX | 1 | OK / STALE |
| 5 | `INDEX_EOD_CURRENCY` | INDEX | 1 | OK / STALE |
| 6 | `VOLATILITY_INDEX` | INDEX | 3 | OK / STALE |
| 7 | `RATE_YIELD` | RATE_YIELD | 2 | OK / STALE |
| 8 | `FUTURE_SETTLEMENT` | FUTURE | 5 | **BLOCKED** |
| 9 | `FX_DAILY_CUT` | FX | 3 | OK / STALE |

**9 contratos.** Familias: 5. Subclases INDEX: 4.

---

## 3. §A.2 — Regla de consolidación de `df_market`

### 3.1. El problema

Los 562 instrumentos de `df_market` no comparten calendario ni semántica económica. No existe una única "última fecha correcta".

**Error a evitar:**
df_market
↓
UN único effective_date
↓
Tratado como fecha económica común

```

### 3.2. effective_date por contrato
df_market
↓
effective_date POR CONTRATO
↓
Consolidación explícita

```

Ningún contrato hereda `effective_date` de otro.

### 3.3. `global_last_date` ≠ fecha económica común

El DataFrame consolidado puede exponer `global_last_date = max(effective_date de cada contrato OK)`.

**Invariante:** es una propiedad **técnica**, no una fecha económica común.

**Prohibición:** ningún consumidor puede usar `global_last_date` como la fecha representativa del universo completo.

**Obligación:** cada consumidor declara qué contrato(s) consume y sobre qué `effective_date`.

### 3.4. Operación de consolidación
consolidate_df_market(contracts_resolved: dict[contract_name, TemporalContract]) -> df_market:

1. Validación: universos disjuntos
assert all universos disjuntos
assert all status in [OK, STALE, INSUFFICIENT, BLOCKED, PENDING]

2. Concatenación por columna
df = concat([c.data for c in contracts_resolved])

3. Metadata adjunta
df.temporal_meta = {
'by_contract': {
name: {
'effective_date': c.effective_date,
'expected_date': c.expected_date,
'lag_days': c.lag_days,
'coverage': c.coverage,
'status': c.status,
}
for name, c in contracts_resolved.items()
},
'global_last_date': max(c.effective_date for c in contracts_resolved if c.status in (OK, STALE)),
'reference_date': datetime,
'run_id': str,
}
return df

```

**Invariantes:**

1. **Unicidad:** ningún ticker en dos contratos.
2. **Completitud:** todo contrato declarado o marcado `BLOCKED`/`PENDING`.
3. **Trazabilidad:** cada fila conoce su contrato de origen.
4. **No imposición:** el DataFrame no obliga a todos los contratos a compartir fecha.

### 3.5. Qué ve cada consumidor

El consumidor recibe `df_market.temporal_meta` con:
{
'by_contract': {
'EQUITY_EOD': {...},
'INDEX_EOD_USA': {...},
...
},
'global_last_date': date,
'reference_date': datetime,
'run_id': str
}

```

Y declara:
- qué contrato(s) usa;
- qué `effective_date` corresponde;
- qué `coverage` tiene su cálculo.
---

## 4. §A.3 — Metadata de fecha y cobertura por artefacto

### 4.1. Las 4 columnas

Decisión del dictamen Q-C2.3 (opción d) y ratificada en Parte B:

| Columna | Definición | Ejemplo |
|---|---|---|
| `date` | Fecha de observación representada por la fila | 2026-09-14 |
| `effective_date` | Última fecha que cumple el contrato de cobertura | 2026-09-14 |
| `expected_date` | Fecha que el contrato esperaba en `reference_date` | 2026-09-15 |
| `coverage` | Cobertura del universo elegible en `effective_date` | 0.9981 |

**Nota (v2):** Parte A v1 usaba `expected_session`. El dictamen de Parte B aprobó renombrar a `expected_date` para no confundir con `is_market_day` de B2.

### 4.2. Relación entre columnas

**Para métrica agregada:**
date ≈ effective_date

```
cuando el cálculo representa exclusivamente esa fecha.

**Regla:** nunca escribir `expected_date` como si fuera automáticamente la fecha observada.

### 4.3. Obligatoriedad por tipo de artefacto

| Tipo | date | effective_date | expected_date | coverage |
|---|---|---|---|---|
| Métrica agregada (una fila por fecha) | Obligatorio | Obligatorio | Obligatorio | Obligatorio |
| Métrica por sector (una fila por sector-fecha) | Obligatorio | Obligatorio | Opcional | Opcional |
| Estado persistente (JSON) | No aplica | Obligatorio | Opcional | No aplica |
| Serie temporal larga | Obligatorio | Obligatorio | Opcional | Opcional |
| Report markdown | No aplica | Recomendado en cabecera | Opcional | No aplica |

### 4.4. Ejemplos

**`sector_breadth.csv` (contrato EQUITY_EOD):**

Estado objetivo:
date,effective_date,expected_date,coverage,sector,pct_above_ema20,...
2026-09-14,2026-09-14,2026-09-14,1.00,XLK,...

```

**`sector_concentration.csv` (contrato EQUITY_EOD):**

Estado objetivo:
date,effective_date,expected_date,coverage,sector,metric,...
2026-09-14,2026-09-14,2026-09-15,0.9981,XLK,...

```

---

## 5. §A.4 — Contrato de writers

### 5.1. Los 20 writers identificados

**Capa 1 (13 writers):**

| Módulo | Artefacto | Patrón actual | Contrato |
|---|---|---|---|
| `engines.py` | `sector_persistence.csv` | P1 | EQUITY_EOD |
| `sectors_base.py` | `sector_rank_history.csv` | P2 | EQUITY_EOD |
| `sectors_base.py` | `sector_rank_deltas.csv` | P2 | EQUITY_EOD |
| `sectors_base.py` | `sector_dispersion.csv` | P2 | EQUITY_EOD |
| `sectors_base.py` | `sector_correlation_matrix.csv` | P2 | EQUITY_EOD |
| `sectors_base.py` | `sector_correlation_summary.csv` | P2 | EQUITY_EOD |
| `sectors_base.py` | `cross_asset_correlation.csv` | P2 | EQUITY_EOD |
| `sectors_base.py` | `cross_asset_context.csv` | P2 | EQUITY_EOD |
| `flows_primary.py` | `sector_flow_characteristics.csv` | Delegado | — |
| `breadth_metrics.py` | `sector_breadth.csv` | **P6 (correcto)** | EQUITY_EOD |
| `breadth_metrics.py` | `sector_breadth_momentum.csv` | Delegado | EQUITY_EOD |
| `market_data.py` | `volatility_structure.csv` | Delegado | EQUITY_EOD + VOLATILITY_INDEX |
| `market_data.py` | `data_quality.csv` | Delegado | — |

**Capa 2b (5 writers):**

| Módulo | Artefacto | Patrón actual | Contrato |
|---|---|---|---|
| `cross_asset_context.py` | `cross_asset_*.csv` | P3 | EQUITY_EOD + INDEX_EOD_CURRENCY + FX |
| `sector_correlation.py` | `sector_correlation_*.csv` | P3 | EQUITY_EOD |
| `rs_internal.py` | `rs_internal.csv` | P4 | EQUITY_EOD |
| `sector_leader_divergence.py` | `sector_leader_divergence.csv` | P5 | EQUITY_EOD + df_stocks |
| `volatility_structure.py` | `volatility_structure.csv` | P7 | EQUITY_EOD + VOLATILITY_INDEX |

**Casos especiales (2):**

| Módulo | Artefacto | Patrón actual | Nota |
|---|---|---|---|
| `stock_leader.py` | `analisis_lideres.csv` | P0 (sin `date`) | Sin columna temporal |
| `mte.py` | `mte_state.json` + `MTE_STATE_FILE` | P8 | JSON, ver §7 |

### 5.2. Regla general

> **Los writers consumen la fecha temporal resuelta del productor; no la vuelven a resolver.**

**Prohibiciones explícitas:**
PROHIBIDO — walk-back local
last_market_session(df_market.index[-1])

PROHIBIDO — resolver en el consumidor
resolve_effective_date(df_market, ...)

PROHIBIDO — trim local
trim_to_last_valid_date(df_market, ...)

```

**Obligación:** cada writer recibe `temporal_meta` (dict). De él extrae `effective_date`, `expected_date`, `coverage`.

### 5.3. Patrón correcto (P6)

`sector_breadth.py` es el modelo:
def compute_sector_breadth(df_market, df_stocks, holdings_df, as_of_date=None):
if as_of_date is not None:
if not is_market_day(as_of_date.date()):
raise ValueError(...)
df_market = df_market.loc[:as_of_date]
df_stocks = df_stocks.loc[:as_of_date]

Cálculo ...
row['date'] = as_of_date
row['effective_date'] = meta['effective_date']
row['expected_date'] = meta['expected_date']
row['coverage'] = meta['coverage']

```

### 5.4. Reescritura por patrón

| Patrón | Estrategia |
|---|---|
| **P0** | Añadir columnas `date`, `effective_date`, `coverage` |
| **P1** | Sustituir `_observation_date_from_df(df_market)` por `temporal_meta['by_contract'][c]['effective_date']` |
| **P2** | Sustituir `df_market.index[-1]` por `effective_date` del contrato |
| **P3** | Sustituir `_observation_date_from_df(returns_df)` por `effective_date` del intersect de contratos usados |
| **P4** | Sustituir `_observation_date_from_df(intersect)` por `effective_date` del intersect |
| **P5** | Sustituir `_observation_date_from_df(df_stocks)` por `effective_date` de EQUITY_EOD |
| **P6** | Ya correcto. Añadir `effective_date`, `expected_date`, `coverage` |
| **P7** | Sustituir `_observation_date_from_df(vix)` por `effective_date` del contrato combinado EQUITY_EOD + VOLATILITY_INDEX |
| **P8** | Ver §7 |

---

## 6. §A.5 — Propagación hacia los 30 consumidores

### 6.1. Grafo
run.py (orquestador)
│ reference_date, run_id
▼
data_load.load_all_data (productor)
│ df_market + df_market.temporal_meta
▼
Capa 1 (12) ─────► Capa 2a (3) ─────► Capa 2b (15)
│
│ df_stocks + df_stocks_effective_meta (FU-020)
▼
mte_confirmation, sector_metrics (C16)

```

### 6.2. `temporal_meta`
temporal_meta = {
'by_contract': {
'EQUITY_EOD': {'effective_date': ..., 'expected_date': ..., 'lag_days': ..., 'coverage': ..., 'status': 'OK'},
'INDEX_EOD_USA': {...},
'INDEX_EOD_EUROPA': {...},
'INDEX_EOD_COMMODITY': {...},
'INDEX_EOD_CURRENCY': {...},
'VOLATILITY_INDEX': {...},
'RATE_YIELD': {...},
'FUTURE_SETTLEMENT': {'status': 'BLOCKED'},
'FX_DAILY_CUT': {...},
},
'global_last_date': date,
'reference_date': datetime,
'run_id': str,
}

```

### 6.3. API de acceso
def get_effective_meta(df_market, contracts: list[str]) -> dict:
"""
Devuelve el meta temporal para los contratos indicados.
Si el consumidor usa varios, devuelve el intersect efectivo
(min effective_date, min coverage, status combinado).
"""

```

**Ejemplo:**
meta = get_effective_meta(df_market, ['EQUITY_EOD'])
effective = meta['effective_date']
coverage = meta['coverage']

```

### 6.4. Tratamiento de FUTURE_SETTLEMENT

**Ratificado por Q-B.7 (opción β).** Con FUTURE_SETTLEMENT bloqueado:

- El componente futuro se **excluye del cálculo**.
- El consumidor recalcula el resto de evidencia.
- Metadata explícita en el output:
future_status = BLOCKED
future_coverage = 0
future_contribution = excluded

```

**Rechazado γ (UNRELIABLE con inclusión numérica):** produce "dato no auditable + score aparentemente válido", exactamente lo que hay que evitar.

**Rechazado α (inclusión silenciosa):** los outputs no son auditables.

**Comportamiento si un consumidor requiere futuros obligatoriamente:** pasa a `INSUFFICIENT`, no fabrica el resultado.

### 6.5. Compatibilidad con veredicto B

Los 30 consumidores tienen veredicto `B` (Compatibles condicionados). Significa:

- No requieren cambio interno si el productor entrega `df_market` temporalmente resuelto.
- Cambios mecánicos:
  - Añadir `temporal_meta` a la firma (o adjuntarlo al df).
  - Sustituir `df_market.index[-1]` por consultas a `temporal_meta`.
  - Declarar en el output qué contrato y qué `effective_date` usan.
- Estimación: 30 ficheros, 1-3 líneas por fichero.

---

## 7. §A.6 — Rediseño del estado MTE

### 7.1. Estado actual

`mte.py` escribe dos ficheros con estado persistente (C43 + C48):

| Fichero | Path | Contenido |
|---|---|---|
| Estado con histéresis | `outputs/state/mte_state.json` (hardcoded L1290) | `{scenario, pending}` |
| Snapshot del run | `MTE_STATE_FILE` (config) | `{scenario, confidence, msi, ipi, srs, shs, cls, ips}` |

**Problemas:**
- Dos paths sin coherencia garantizada.
- Sin versión de contrato temporal.
- Sin `effective_date`.
- Un cambio en `df_market` puede producir transición no lineal.

### 7.2. Esquema único versionado

**Path único:** `outputs/state/mte_state.json`.

**Esquema:**
{
"schema_version": 1,
"temporal_contract_version": "FU-021-5-v2",
"effective_date": "2026-09-14",
"expected_date": "2026-09-15",
"coverage": 0.9981,
"scenario": "EXPANSION",
"pending": null,
"confidence": 0.72,
"metrics": {
"srs": 0.15, "shs": -0.05, "cls": 0.22, "ips": 0.10,
"msi": 55.0, "ipi": 55.0
},
"futures_status": "BLOCKED",
"written_at": "2026-09-15T20:00:00+02:00"
}

```

### 7.3. Reset por cambio de contrato

**Regla:**
same temporal_contract_version → state reutilizable
temporal_contract_version changed materially → state incompatible → reset

```

**Implementación:**
def load_previous_scenario():
try:
with open(STATE_FILE) as f:
data = json.load(f)
if data.get('temporal_contract_version') != CURRENT_TEMPORAL_CONTRACT_VERSION:
print('MTE state: contrato temporal cambió. Reset.')
return 'MIXED', None
return data.get('scenario', 'MIXED'), data.get('pending')
except:
return 'MIXED', None

```

### 7.4. Migración del estado existente

**No hay migración.** El estado actual se descarta en el primer run tras activación. El estado no es un histórico de mercado; es una máquina de transición. Sin contrato, no es auditable.

### 7.5. Eliminación del path hardcoded

`STATE_FILE` hardcoded (L1290) se elimina. `MTE_STATE_FILE` (config) es el único path. Documentar en `config/settings.py`.
---

## 8. §A.7 — Migración de `darkpool`

### 8.1. Objetivo

Eliminar la única excepción al principio "df_market en memoria es la fuente canónica".

**Estado actual:**

    compute_darkpool_signals()  # sin parámetros
        |
        +-- pd.read_parquet('data/market_data.parquet')  # L189
        +-- pd.read_parquet('data/stock_prices.parquet')  # L194

**Estado objetivo:**

    compute_darkpool_signals(df_market, df_stocks=None)
        |
        +-- usa df_market (en memoria)
        +-- usa df_stocks (en memoria, opcional)

### 8.2. Firma nueva

    def compute_darkpool_signals(df_market, df_stocks=None):
        """
        Calcula señales de dark pools desde FINRA ATS.

        Args:
            df_market: DataFrame consolidado (fuente canónica).
            df_stocks: DataFrame de equities (FU-020), opcional.

        Returns:
            dict con señales o None.
        """

### 8.3. Cambios en `market_data.py`

**Actual:** `compute_market_data(df_market)` no recibe `df_stocks`.

**Objetivo:**

    def compute_market_data(df_market, df_stocks=None):
        # ...
        darkpool_data = _compute_darkpool(df_market, df_stocks)
        # ...

**Cambio en `run.py`:** pasar `df_stocks` a `compute_market_data`.

### 8.4. Propagación de `df_stocks`

`df_stocks` se genera en `leaders.py` (fase 6a). `market_data.py` se ejecuta en fase 9b. En el orden del pipeline, `df_stocks` está disponible.

**Cadena:**

    run.py L113 -> leaders.py -> df_stocks
    run.py L153 -> compute_market_data(df_market)  # actual
    run.py L153 -> compute_market_data(df_market, df_stocks=df_stocks)  # objetivo

### 8.5. Compatibilidad

`compute_darkpool_signals(df_market, df_stocks=None)` es retrocompatible.

**Pendiente Parte B:** decidir si `df_stocks` es obligatorio o opcional. Hoy la extracción de volúmenes se hace de ambos parquets. Si se pasa solo `df_market`, la cobertura de volúmenes cae.

**Recomendación:** `df_stocks` obligatorio. Requiere cambio en la firma de `compute_market_data`.

---

## 9. §A.8 — Plan de verificación

### 9.1. Tests obligatorios

**Taxonomía (nuevo en v2):**
- Test: clasificación de los 562 tickers en familias + subclases + contratos.
- Test: `get_instrument_class` retorna valor esperado por familia.
- Test: ningún ticker en dos contratos.
- Test: 9 contratos, 5 familias, 4 subclases INDEX.

**Consolidación:**
- Test: `global_last_date = max(effective_date)` de contratos OK/STALE.
- Test: `effective_date` por contrato no se propaga a otro.
- Test: `temporal_meta` accesible desde `df_market`.
- Test: `by_contract` contiene las 9 entradas.

**FSM (nuevo en v2):**
- Test: transiciones válidas `PENDING → OK/STALE/INSUFFICIENT/BLOCKED`.
- Test: `STALE` requiere `0 < lag_days <= max_lag_days`.
- Test: `INSUFFICIENT` requiere `coverage < min_coverage`.
- Test: `BLOCKED` no transiciona sin `activation_requirement` resuelto.

**Metadata cuádruple:**
- Test: cada writer histórico tiene `date`, `effective_date`, `expected_date`, `coverage`.
- Test: `date ≈ effective_date` para métricas agregadas.

**MTE:**
- Test: `schema_version` presente.
- Test: `temporal_contract_version` presente.
- Test: `futures_status=BLOCKED` cuando FUTURE_SETTLEMENT está bloqueado.
- Test: reset si `temporal_contract_version` difiere.

**Darkpool:**
- Test: `compute_darkpool_signals(df_market)` funciona sin parquet.
- Test: `compute_darkpool_signals(df_market, df_stocks)` cubre ambos universos.

**FUTURE_SETTLEMENT bloqueado:**
- Test: consumidores afectados producen `future_status=BLOCKED` en output.
- Test: `mte.py` incluye `futures_status` en el estado JSON.
- Test: `macro_regime` recalcula sin componente futuro.

### 9.2. Verificación en producción real

Antes de declarar cerrada la Parte A, ejecutar `daily_run.yml` manual y verificar:

- Log con `[FU-021-5]` en cada contrato resuelto.
- `market_data.parquet` con manifest extendido (contratos por clase).
- Reportes markdown con cabecera temporal.
- Estado MTE con `futures_status=BLOCKED`.

### 9.3. Verificación de no-regresión

Diff contra snapshot pre-activación:
- Secciones `##` idénticas en orden.
- Subsecciones `###` idénticas.
- Líneas idénticas >= 90%.

---

## 10. §A.9 — Criterios de cierre

La Parte A v2 se considerará cerrada cuando:

1. Este documento esté aprobado por dictamen.
2. La Parte B esté ratificada (ya ratificada en 2026-09-16).
3. Exista un plan de tests concreto.
4. Exista un plan de migración por fases.
5. No haya contradicciones con R1-R4 ni con A2.3.

**La Parte A v2 NO autoriza implementación.** Autoriza diseño y planificación.

---

## 11. §A.10 — FSM formal de contratos

**Nuevo en v2.** Ratificado por dictamen de Parte B (§"Sobre STALE y BLOCKED").

### 11.1. Diagrama

    PENDING
       |
       | resolución
       v
    +-- OK
    +-- STALE
    +-- INSUFFICIENT
    +-- BLOCKED

### 11.2. Semántica de cada estado

| Estado | Condición | Significado |
|---|---|---|
| `PENDING` | Contrato definido, no activado | Estado inicial |
| `OK` | `effective_date == expected_date` y `coverage >= min_coverage` | Contrato satisfecho, dentro de frescura esperada |
| `STALE` | `0 < lag_days <= max_lag_days` y `coverage >= min_coverage` | Contrato satisfecho con lag permitido |
| `INSUFFICIENT` | `coverage < min_coverage` o `lag_days > max_lag_days` | Cobertura o frescura insuficientes |
| `BLOCKED` | `activation_requirement` no resuelto | Contrato no auditable / no disponible de forma fiable |

### 11.3. Transiciones

    PENDING -> OK            | si resolución produce frescura perfecta
    PENDING -> STALE         | si resolución produce frescura con lag permitido
    PENDING -> INSUFFICIENT  | si resolución detecta cobertura insuficiente
    PENDING -> BLOCKED       | si el contrato no puede activarse

    OK        -> STALE         | si el siguiente run trae lag
    OK        -> INSUFFICIENT  | si cobertura cae
    STALE     -> OK            | si el siguiente run recupera frescura
    STALE     -> INSUFFICIENT  | si lag excede max_lag_days o cobertura cae
    INSUFFICIENT -> OK         | si cobertura se recupera
    INSUFFICIENT -> STALE      | si cobertura se recupera con lag
    BLOCKED   -> PENDING       | solo si activation_requirement se resuelve

### 11.4. Prohibiciones

- `OK` o `STALE` **no pueden** declararse si `coverage < min_coverage`.
- `BLOCKED` **no puede** transicionar directamente a `OK`/`STALE`/`INSUFFICIENT` sin pasar por `PENDING`.
- `STALE` **no significa** "cualquier dato antiguo es aceptable".

### 11.5. Metadata obligatoria en todo estado

Independientemente del estado, el contrato **siempre** expone:

    effective_date  : date | None
    expected_date   : date | None
    lag_days        : int | None
    coverage        : float | None
    status          : OK | STALE | INSUFFICIENT | BLOCKED | PENDING

En estado `BLOCKED`, `effective_date` y `lag_days` pueden ser `None`. El estado documenta que el contrato no puede resolverse, no que no existe.

---

## 12. Lo que la Parte A v2 NO decide

- No activa contratos por clase.
- No retira `trim_to_last_valid_date` (A3.1 sigue bloqueada).
- No decide el orden de implementación.
- No implementa nada.
- No decide el provider alternativo para FUTURE_SETTLEMENT.
- No decide el cutoff exacto de FX.
- No decide si `df_stocks` en `compute_darkpool_signals` es obligatorio u opcional.

---

**Fin de la Parte A v2.**
**HEAD de referencia:** 0eb7e4f (origin/main).
**Fecha:** 2026-09-16.
**Estado:** ratificado con correcciones. Pendiente de implementación.
