# FU-021-5 — Especificación del contrato temporal de `df_market`

**Parte A — Arquitectura contractual**
**Fecha:** 2026-09-15
**HEAD de referencia:** 42c395c (origin/main)
**Autorizado por:** Dictamen del auditor externo — Contrato temporal de `df_market` (2026-09-15)
**Documentos base:** Adendum 1 (`694b172`), Informe Ciclo 2 (`bd18d22`), Briefing (`0c888c6`), Adendum 2 (`42c395c`)
**Estado:** borrador Parte A. Pendiente de dictamen sobre Parte B.

---

## 0. Alcance y separación Parte A / Parte B

### 0.1. Alcance del documento

Especificar el contrato temporal de `df_market` conforme a la autorización del dictamen del contrato temporal:

- arquitectura contractual por clase;
- regla de consolidación de un DataFrame mixto;
- metadata de fecha y cobertura por artefacto;
- contrato de writers históricos;
- propagación hacia consumidores;
- rediseño del estado persistente de MTE;
- migración de `darkpool`.

### 0.2. Separación Parte A / Parte B

El dictamen obliga a distinguir dos partes:

| Parte | Contenido | Estado |
|---|---|---|
| **A** | Arquitectura, responsabilidades, invariantes, propagación, writers, MTE, darkpool | Cerrable ahora |
| **B** | Semántica operativa de las clases no verificadas: INDEX_EOD, RATE_YIELD, FUTURE_SETTLEMENT, FX_DAILY_CUT | Bloqueada por FU-021-3B/3C |

**Este documento cubre Parte A.** La Parte B se publicará cuando 3B/3C aporten la verificación empírica necesaria.

### 0.3. Lo que este documento NO es

- No es un patch.
- No activa contratos.
- No retira `trim_to_last_valid_date` (A3.1 sigue bloqueada).
- No modifica el informe original FU-021-3A ni sus adendos.
- No toca código.

### 0.4. Terminología

Nomenclatura oficial (adoptada en Adendum 2):

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

### 1.1. Principios derivados del dictamen

1. **Contratos por clase, no un contrato agregado.** `df_market` no tiene una única "última fecha correcta". Tiene una frontera temporal por subconjunto homogéneo.
2. **Consolidación explícita.** La operación de construir `df_market` a partir de las clases resueltas debe estar declarada, no ser implícita.
3. **Metadata cuádruple.** Los artefactos que necesiten trazabilidad temporal distinguen `date`, `effective_date`, `expected_session`, `coverage`.
4. **Fuente canónica única.** `df_market` en memoria es la única fuente de verdad intra-run. Ningún módulo lee parquet directo.
5. **Responsabilidad temporal única.** La resolución temporal existe una sola vez, en el productor. Los consumidores consumen el DataFrame ya resuelto.
6. **Estado versionado.** El estado persistente de MTE lleva contrato, versión y fecha efectiva. Se resetea por cambio material de contrato.

### 1.2. Principios que NO cambian

- Determinismo. Sin ML, sin optimización.
- Sin datos ficticios: `N/D` antes que imputar.
- Respeto a R1–R4 (FU-020).
- `resolve_effective_date` intacta (FU-020).
- Manifest FU-002 intacto.
- B2 (`is_market_day`) sigue siendo el único control de sesión.

### 1.3. Interfaz conceptual
RAW
│
▼
CLASSIFICATION (get_instrument_class)
│
├── EQUITY_EOD
├── INDEX_EOD
├── RATE_YIELD
├── FUTURE_SETTLEMENT
└── FX_DAILY_CUT
│
▼
TEMPORAL RESOLUTION PER CLASS
│ (efectivo por clase: effective_date, expected_session, coverage, status)
▼
CONSOLIDATION
│ (operación declarada, invariantes explícitas)
▼
df_market CONSOLIDADO
│
▼
CONSUMIDORES (30)
│
├── writers históricos (20)
├── MTE state
└── report

text

---

## 2. §A.1 — Contratos por clase

### 2.1. Clasificación del universo

El universo de `df_market` (562 instrumentos) se particiona en 5 clases homogéneas.

La clasificación económica se resuelve vía `get_instrument_class(ticker)` (`src/instrument_registry.py`, introducido en A2.3, commit `fd12ea1`).

| Clase | Universo | Cuenta aprox. |
|---|---|---|
| `EQUITY_EOD` | Acciones y ETFs USA | 539 |
| `INDEX_EOD` | Índices no-USA | ~13 |
| `RATE_YIELD` | Yields (Treasury, curvas) | ~2 |
| `FUTURE_SETTLEMENT` | Futuros commodities y equity | ~5 |
| `FX_DAILY_CUT` | Pares FX | ~3 |

**Nota terminológica:** `get_instrument_class != get_market`. La primera clasifica económicamente; la segunda clasifica bursátilmente. Son responsabilidades disjuntas (A2.3).

### 2.2. Contrato EQUITY_EOD

**Estado:** cerrado (FU-021-3A v2, commits `78b7583` + `747cfb1` + `fded14b`).

**Semántica:**
- Fecha efectiva = última sesión con cierre EOD.
- Sesión definida por calendario NYSE.
- Filtro de sesión (FU-018) + resolución por cobertura (FU-020) aplicados.

**Interfaz:**
class EQUITY_EOD:
eligible_universe : list[ticker] # 539
min_coverage : float = 0.90
session_calendar : "NYSE"
filter_function : _filter_non_eod_equity # FU-021-3A
resolver : resolve_effective_date # FU-020
outputs:
effective_date : date
expected_session : date
coverage : float in [0,1]
status : OK | INSUFFICIENT_COVERAGE

text

**Invariantes:**
- `effective_date <= expected_session`.
- `coverage >= min_coverage` para status `OK`.
- Ningún registro con fecha no bursátil.

### 2.3. Contrato INDEX_EOD

**Estado:** esquema definido. Semántica pendiente de FU-021-3B.

**Semántica (provisional, requiere verificación empírica):**
- Índices no-USA: `^FTSE`, `^GDAXI`, `^FCHI`, `^STOXX50E`, etc.
- Fecha efectiva = última sesión del mercado origen con cierre EOD.
- Sesión definida por calendario del mercado origen (LSE, XETRA, Euronext, etc.).

**Interfaz:**
class INDEX_EOD:
eligible_universe : list[ticker]
market_calendar : per-ticker (LSE | XETRA | EURONEXT | ...)
outputs:
effective_date : date
expected_session : date
coverage : float
status : OK | PENDING_VERIFICATION | INSUFFICIENT

text

**Pendiente Parte B:**
- Verificar empíricamente que Yahoo devuelve Close EOD para índices no-USA.
- Definir calendarios de sesión por mercado origen.

### 2.4. Contrato RATE_YIELD

**Estado:** esquema definido. Semántica pendiente de FU-021-3B.

**Semántica (provisional):**
- Yields: `^TNX`, `^FVX`, `^TYX`, `^IRX`.
- Fecha efectiva = última publicación del yield.
- Puede tener calendario FRED-like (no calendario bursátil).

**Interfaz:**
class RATE_YIELD:
eligible_universe : list[ticker]
publication_source : "US_TREASURY" | "FRED"
outputs:
effective_date : date
expected_session : date
coverage : float
status : OK | PENDING_VERIFICATION | INSUFFICIENT

text

**Pendiente Parte B:**
- Verificar semántica temporal del Close de Yahoo para yields.
- Definir si es calendario bursátil o publicación FRED.

### 2.5. Contrato FUTURE_SETTLEMENT

**Estado:** esquema definido. Semántica bloqueada por FU-021-3C.

**Semántica (provisional):**
- Futuros: `CL=F`, `BZ=F`, `NG=F`, `GC=F`, `HG=F`, `ES=F`, `NQ=F`.
- Fecha efectiva = fecha de settlement del contrato.
- Requiere provider dedicado (no Yahoo).

**Interfaz:**
class FUTURE_SETTLEMENT:
eligible_universe : list[ticker]
settlement_source : "CME" | "ICE" | "NYMEX" | ...
outputs:
effective_date : date
expected_session : date
coverage : float
status : OK | BLOCKED | INSUFFICIENT

text

**Pendiente Parte B:**
- Definir provider.
- Definir semántica de settlement (¿hora, sesión, cutoff?).

### 2.6. Contrato FX_DAILY_CUT

**Estado:** esquema definido. Semántica bloqueada por FU-021-3C.

**Semántica (provisional):**
- FX: `EURUSD=X`, `DX-Y.NYB`, `USDJPY=X`, etc.
- Fecha efectiva = cierre diario FX (típicamente 17:00 ET).
- Requiere provider dedicado.

**Interfaz:**
class FX_DAILY_CUT:
eligible_universe : list[ticker]
cutoff_time : "17:00 America/New_York" # provisional
outputs:
effective_date : date
expected_session : date
coverage : float
status : OK | BLOCKED | INSUFFICIENT

text

**Pendiente Parte B:**
- Definir cutoff exacto.
- Definir provider.

### 2.7. Interfaz común

Los 5 contratos exponen una interfaz común:
TemporalContract:
eligible_universe : list[ticker]
effective_date : date
expected_session : date
coverage : float in [0,1]
status : OK | INSUFFICIENT | BLOCKED | PENDING

text

**Invariantes comunes:**
- `effective_date <= expected_session` si `status == OK`.
- `coverage` calculado sobre `eligible_universe` en `effective_date`.
- Ningún ticker clasificado en dos clases simultáneas.

**Estados terminales:**
- `OK`: contrato resuelto y aplicable.
- `INSUFFICIENT`: cobertura por debajo del umbral.
- `BLOCKED`: contrato no activable (clase no implementada).
- `PENDING`: contrato pendiente de verificación empírica.

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

text

**Regla aprobada:**
df_market
↓
effective_date POR CLASE
↓
Consolidación explícita de un DataFrame mixto

text

### 3.2. effective_date por clase

Tras la resolución temporal, cada clase tiene su propio `effective_date`:
EQUITY_EOD → 2026-09-14
INDEX_EOD → ? (pendiente 3B)
RATE_YIELD → ? (pendiente 3B)
FUTURE_SETTLEMENT → ? (bloqueado 3C)
FX_DAILY_CUT → ? (bloqueado 3C)

text

**Regla:** ninguna clase hereda `effective_date` de otra.

### 3.3. `global_last_date` ≠ fecha económica común

El DataFrame consolidado puede tener una propiedad técnica llamada `global_last_date`:
global_last_date = max(effective_date de cada clase OK)

text

**Invariante:** `global_last_date` es una propiedad **técnica** del DataFrame. No es una fecha económica común.

**Prohibición:** ningún consumidor puede usar `global_last_date` como si fuera la fecha representativa del universo completo.

**Consumidor correcto:** debe declarar qué clase(s) consume y sobre qué `effective_date`.

### 3.4. Operación de consolidación
consolidate_df_market(classes_resolved: dict[class_name, TemporalContract]) -> df_market:

1. Validación
assert all(c.eligible_universe desjunta)
assert all(c.status in [OK, INSUFFICIENT, BLOCKED, PENDING])

2. Concat por columna, respetando cada clase
df = concat([c.data for c in classes_resolved])

3. Metadata adjunta
df.temporal_meta = {
'by_class': {
cls: {
'effective_date': c.effective_date,
'expected_session': c.expected_session,
'coverage': c.coverage,
'status': c.status,
}
for cls, c in classes_resolved.items()
},
'global_last_date': max(c.effective_date for c in classes_resolved if c.status == OK)
}

return df

text

**Invariantes de consolidación:**

1. **Unicidad:** ningún ticker en dos clases.
2. **Completitud:** toda clase declarada o marcada `BLOCKED`/`PENDING`.
3. **Trazabilidad:** cada fila del DataFrame consolidado conoce su clase de origen.
4. **No imposición:** el DataFrame consolidado no obliga a todas las clases a compartir fecha.

### 3.5. Qué ve cada consumidor

Un consumidor de Capa 1 (`src/pipeline/*`) recibe:
df_market.temporal_meta = {
'by_class': {
'EQUITY_EOD': {'effective_date': date, 'coverage': float, ...},
'INDEX_EOD': {'status': 'PENDING', ...},
...
},
'global_last_date': date
}

text

El consumidor declara:
- qué clase(s) usa;
- qué `effective_date` corresponde a esa clase;
- qué cobertura efectiva tiene su cálculo.

**Ejemplo conceptual** (consumidor que usa solo EQUITY_EOD):
def compute_something(df_market):
meta = df_market.temporal_meta['by_class']['EQUITY_EOD']
effective = meta['effective_date']
coverage = meta['coverage']

cálculo ...
return {'result': ..., 'effective_date': effective, 'coverage': coverage}

text

---

**Fin del chunk 1.** Las secciones §4–§9 se añaden en el chunk 2.---

## 4. §A.3 — Metadata de fecha y cobertura por artefacto

### 4.1. Semántica de las 4 columnas

Decisión del dictamen (Q-C2.3, opción d): distinguir al menos 4 conceptos temporales en artefactos que necesiten trazabilidad.

| Columna | Definición | Ejemplo |
|---|---|---|
| `date` | Fecha de observación representada por la fila del dataset procesado | 2026-09-14 |
| `effective_date` | Última fecha que cumple el contrato de cobertura exigido | 2026-09-14 |
| `expected_session` | Fecha que el contrato esperaba para el instrumento/grupo en `reference_date` | 2026-09-15 |
| `coverage` | Cobertura del universo elegible en `effective_date` | 0.9981 |

### 4.2. Relación entre columnas

**Para métrica agregada:**

    date ≈ effective_date

cuando el cálculo representa exclusivamente esa fecha.

**Regla:** nunca escribir `expected_session` como si fuera automáticamente la fecha observada. `expected_session` documenta la expectativa del contrato; `effective_date` documenta lo efectivamente calculado.

### 4.3. Cuándo cada columna es obligatoria

| Tipo de artefacto | date | effective_date | expected_session | coverage |
|---|---|---|---|---|
| Métrica agregada (una fila por fecha) | Obligatorio | Obligatorio | Obligatorio | Obligatorio |
| Métrica por sector (una fila por sector-fecha) | Obligatorio | Obligatorio | Opcional | Opcional |
| Estado persistente (JSON) | No aplica | Obligatorio | Opcional | No aplica |
| Serie temporal larga (histórico) | Obligatorio | Obligatorio | Opcional | Opcional |
| Report markdown | No aplica | Recomendado en cabecera | Opcional | No aplica |

**Nota:** "Opcional" significa que puede omitirse si el artefacto declara su contrato de clase y cobertura en metadatos externos (por ejemplo, en el manifest FU-002).

### 4.4. Ejemplo aplicado al CSV `sector_breadth.csv`

Estado actual (patrón P6, único correcto):

    date,sector,pct_above_ema20,...
    2026-09-14,XLK,...

Estado objetivo tras FU-021-5:

    date,effective_date,expected_session,coverage,sector,pct_above_ema20,...
    2026-09-14,2026-09-14,2026-09-14,1.00,XLK,...

### 4.5. Ejemplo aplicado al CSV `sector_concentration.csv`

Estado actual (patrón P5, resuelto parcialmente por C16):

    date,sector,metric,...
    2026-09-14,XLK,...

Estado objetivo:

    date,effective_date,expected_session,coverage,sector,metric,...
    2026-09-14,2026-09-14,2026-09-15,0.9981,XLK,...

---

## 5. §A.4 — Contrato de writers

### 5.1. Los 20 writers identificados

**Capa 1 (13 writers):**

| Módulo | Artefacto | Patrón actual | Clase |
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
| `market_data.py` | `volatility_structure.csv` | Delegado | EQUITY_EOD + VIX |
| `market_data.py` | `data_quality.csv` | Delegado | — |

**Capa 2b (5 writers):**

| Módulo | Artefacto | Patrón actual | Clase |
|---|---|---|---|
| `cross_asset_context.py` | `cross_asset_*.csv` | P3 | EQUITY_EOD + FX + DXY |
| `sector_correlation.py` | `sector_correlation_*.csv` | P3 | EQUITY_EOD |
| `rs_internal.py` | `rs_internal.csv` | P4 | EQUITY_EOD |
| `sector_leader_divergence.py` | `sector_leader_divergence.csv` | P5 | EQUITY_EOD + `df_stocks` |
| `volatility_structure.py` | `volatility_structure.csv` | P7 | EQUITY_EOD + VIX |

**Casos especiales (2 writers):**

| Módulo | Artefacto | Patrón actual | Nota |
|---|---|---|---|
| `stock_leader.py` | `analisis_lideres.csv` | P0 (sin `date`) | Sin columna temporal |
| `mte.py` | `mte_state.json` + `MTE_STATE_FILE` | P8 | JSON de estado, ver §A.6 |

### 5.2. Regla general

> **Los writers deben consumir la fecha temporal resuelta del productor; no volver a resolverla independientemente.**

**Prohibiciones explícitas (dictamen):**

    # PROHIBIDO — walk-back local
    last_market_session(df_market.index[-1])

    # PROHIBIDO — resolver en el consumidor
    resolve_effective_date(df_market, ...)

    # PROHIBIDO — trim local
    trim_to_last_valid_date(df_market, ...)

**Obligación:** cada writer recibe `temporal_meta` (dict) desde el productor. De él extrae `effective_date`, `expected_session`, `coverage`.

### 5.3. Patrón correcto (P6)

`sector_breadth.py` implementa el contrato deseado. Es el modelo a replicar:

    def compute_sector_breadth(df_market, df_stocks, holdings_df, as_of_date=None):
        # Validación de sesión
        if as_of_date is not None:
            if not is_market_day(as_of_date.date()):
                raise ValueError(...)
            df_market = df_market.loc[:as_of_date]
            df_stocks = df_stocks.loc[:as_of_date]
        # Cálculo ...
        # Escritura
        row['date'] = as_of_date

**Extensión requerida:** añadir `effective_date`, `expected_session`, `coverage` de `temporal_meta`.

### 5.4. Reescritura por patrón

| Patrón | Estrategia |
|---|---|
| **P0** | Añadir columna `date` + `effective_date` + `coverage` |
| **P1** | Sustituir `_observation_date_from_df(df_market)` por `temporal_meta['by_class'][cls]['effective_date']` |
| **P2** | Sustituir `df_market.index[-1]` por `temporal_meta['by_class'][cls]['effective_date']` |
| **P3** | Sustituir `_observation_date_from_df(returns_df)` por `effective_date` del intersect de clases usadas |
| **P4** | Sustituir `_observation_date_from_df(intersect)` por `effective_date` del intersect de clases usadas |
| **P5** | Sustituir `_observation_date_from_df(df_stocks)` por `effective_date` de la clase EQUITY_EOD |
| **P6** | Ya correcto. Añadir `effective_date`, `expected_session`, `coverage` |
| **P7** | Sustituir `_observation_date_from_df(vix)` por `effective_date` de la clase combinada EQUITY_EOD + VIX |
| **P8** | Ver §A.6 |

---

## 6. §A.5 — Propagación hacia los 30 consumidores

### 6.1. Grafo de propagación

    run.py (orquestador)
       │
       │  reference_date, run_id
       │
       ▼
    data_load.load_all_data (productor)
       │
       │  df_market + df_market.temporal_meta
       │
       ▼
    Capa 1 (12) ────────────► Capa 2a (3) ────► Capa 2b (15)
       │
       │  df_stocks + df_stocks_effective_meta (FU-020)
       │
       ▼
    mte_confirmation, sector_metrics (ya resuelto C16)

### 6.2. `temporal_meta` como objeto

`df_market.temporal_meta` es un atributo adjunto al DataFrame (implementación concreta: atributo Python, columna especial, o dict paralelo; decisión de implementación). Contiene:

    temporal_meta = {
        'by_class': {
            'EQUITY_EOD': {
                'effective_date': date,
                'expected_session': date,
                'coverage': float,
                'status': 'OK',
            },
            'INDEX_EOD': {...},
            ...
        },
        'global_last_date': date,
        'reference_date': datetime,
        'run_id': str,
    }

### 6.3. API de acceso

Los consumidores acceden a través de una función helper:

    def get_effective_meta(df_market, classes: list[str]) -> dict:
        """
        Devuelve el meta temporal para las clases indicadas.
        Si el consumidor usa varias clases, devuelve el intersect efectivo.
        """

**Ejemplo** (consumidor que usa solo EQUITY_EOD):

    meta = get_effective_meta(df_market, ['EQUITY_EOD'])
    effective = meta['effective_date']
    coverage = meta['coverage']

**Ejemplo** (consumidor que usa EQUITY_EOD + VIX):

    meta = get_effective_meta(df_market, ['EQUITY_EOD', 'INDEX_EOD_VIX'])
    # Devuelve effective_date = min(effective de cada clase)

### 6.4. Compatibilidad con veredicto B

Los 30 consumidores auditados obtuvieron veredicto **B — COMPATIBLE CONDICIONADO**. Significa:

- **No requieren cambio interno** si el productor entrega `df_market` ya temporalmente resuelto.
- **No hay reescritura estructural** de la lógica de cálculo.
- **Cambios necesarios:**
  - Añadir `temporal_meta` a la firma (o adjuntarlo a `df_market`).
  - Sustituir accesos a `df_market.index[-1]` por consultas a `temporal_meta`.
  - Declarar en el output qué clase y qué `effective_date` usan.

**Estimación de impacto:** 30 ficheros, 1-3 líneas por fichero. Es un cambio mecánico.

---

## 7. §A.6 — Rediseño del estado MTE

### 7.1. Estado actual

`mte.py` escribe dos ficheros con estado persistente (C43 + C48):

| Fichero | Path | Contenido |
|---|---|---|
| Estado con histéresis | `outputs/state/mte_state.json` (hardcoded, L1290) | `{scenario, pending}` |
| Snapshot del run | `MTE_STATE_FILE` (config) | `{scenario, confidence, msi, ipi, srs, shs, cls, ips}` |

**Problemas:**
- Dos paths sin garantía de coherencia.
- Sin versión de contrato temporal.
- Sin fecha efectiva.
- Un cambio en `df_market` puede producir transición no lineal.

### 7.2. Esquema único y versionado

**Dictamen (Q-C2.4, opción c+b):** path único + versionado + fecha + reset por cambio de contrato.

**Path único propuesto:**

    outputs/state/mte_state.json

**Esquema:**

    {
      "schema_version": 1,
      "temporal_contract_version": "FU-021-5-v1",
      "effective_date": "2026-09-14",
      "expected_session": "2026-09-15",
      "coverage": 0.9981,
      "scenario": "EXPANSION",
      "pending": null,
      "confidence": 0.72,
      "metrics": {
        "srs": 0.15,
        "shs": -0.05,
        "cls": 0.22,
        "ips": 0.10,
        "msi": 55.0,
        "ipi": 55.0
      },
      "written_at": "2026-09-15T20:00:00+02:00"
    }

### 7.3. Reset por cambio de contrato

**Regla:**

    same temporal_contract_version → state reutilizable
    temporal_contract_version changed materially → state incompatible → reset

**Implementación conceptual:**

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

### 7.4. Migración del estado existente

**No hay migración de estado histórico.** El estado actual se considera no versionado y se descarta en el primer run tras la activación.

**Justificación:** el estado no es un histórico de mercado, es una máquina de transición. Su valor depende del contrato bajo el que se generó. Sin contrato, no es auditable.

### 7.5. Eliminación del path hardcoded

El `STATE_FILE` hardcoded (L1290) debe eliminarse. El `MTE_STATE_FILE` (config) pasa a ser el único path. Se documenta en `config/settings.py`.

---

## 8. §A.7 — Migración de `darkpool`

### 8.1. Objetivo

Eliminar la única excepción al principio "df_market en memoria es la fuente canónica".

**Estado actual:**

    compute_darkpool_signals()  # sin parámetros
        │
        ├── pd.read_parquet('data/market_data.parquet')  ← L189
        └── pd.read_parquet('data/stock_prices.parquet')  ← L194

**Estado objetivo:**

    compute_darkpool_signals(df_market, df_stocks=None)
        │
        ├── usa df_market (en memoria)
        └── usa df_stocks (en memoria, opcional)

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

**`compute_market_data(df_market)` actual:** no recibe `df_stocks`.

**Estado objetivo:**

    def compute_market_data(df_market, df_stocks=None):
        # ...
        darkpool_data = _compute_darkpool(df_market, df_stocks)
        # ...

**Cambio en `run.py`:** pasar `df_stocks` a `compute_market_data`.

### 8.4. Propagación de `df_stocks`

`df_stocks` se genera en `leaders.py` (fase 6a). `market_data.py` se ejecuta en fase 9b. En el orden del pipeline, `df_stocks` ya está disponible cuando se llama a `compute_market_data`.

**Cadena:**

    run.py L113 → leaders.py → df_stocks
    run.py L153 → compute_market_data(df_market)  ← actual
    run.py L153 → compute_market_data(df_market, df_stocks=df_stocks)  ← objetivo

### 8.5. Compatibilidad

`compute_darkpool_signals(df_market, df_stocks=None)` es retrocompatible. Si un caller no pasa `df_stocks`, funciona con solo `df_market` (aunque pierde la parte de equities).

**Pendiente Parte B / próxima fase:** decidir si `darkpool` requiere `df_stocks` como obligatorio o como opcional. Hoy la extracción de volúmenes se hace de ambos parquets. Si se pasa solo `df_market`, la cobertura de volúmenes cae.

**Recomendación:** `df_stocks` obligatorio (no opcional). Requiere cambio en la firma de `compute_market_data`.

---

## 9. §A.8 — Plan de verificación

### 9.1. Tests obligatorios

**Contratos por clase:**
- Test: clasificación de los 562 tickers en las 5 clases.
- Test: `get_instrument_class` retorna valor esperado para cada clase.
- Test: ningún ticker en dos clases.

**Consolidación:**
- Test: `global_last_date = max(effective_date)` de clases `OK`.
- Test: `effective_date` por clase no se propaga a otra clase.
- Test: `temporal_meta` accesible desde `df_market`.

**Metadata cuádruple:**
- Test: cada writer histórico tiene las 4 columnas (`date`, `effective_date`, `expected_session`, `coverage`).
- Test: `date ≈ effective_date` para métricas agregadas.

**MTE:**
- Test: `schema_version` presente.
- Test: `temporal_contract_version` presente.
- Test: reset si `temporal_contract_version` difiere.

**Darkpool:**
- Test: `compute_darkpool_signals(df_market)` funciona sin parquet.
- Test: `compute_darkpool_signals(df_market, df_stocks)` cubre ambos universos.

### 9.2. Verificación en producción real

Antes de declarar cerrada la Parte A, ejecutar `daily_run.yml` manual y verificar:

- Log con `[FU-021-5]` en cada clase resuelta.
- `market_data.parquet` con manifest extendido (contratos por clase).
- Reportes markdown con cabecera temporal.

### 9.3. Verificación de no-regresión

Diff contra snapshot pre-activación:
- Secciones `##` idénticas en orden.
- Subsecciones `###` idénticas.
- Líneas idénticas ≥ 90%.

---

## 10. §A.9 — Criterios de cierre de Parte A

La Parte A se considerará cerrada cuando:

1. Este documento esté aprobado por dictamen.
2. La Parte B esté esquematizada (aunque no cerrada).
3. Exista un plan de tests concreto.
4. Exista un plan de migración por fases (activación, MTE, darkpool, writers).
5. No haya contradicciones con R1–R4 ni con A2.3.

**La Parte A NO autoriza implementación.** Autoriza diseño y planificación.

---

## 11. Lo que la Parte A NO decide

- No activa contratos por clase (Parte B + 3B/3C).
- No retira `trim_to_last_valid_date` (A3.1 sigue bloqueada).
- No decide el orden de implementación.
- No implementa nada.
- No decide la semántica operativa de INDEX_EOD, RATE_YIELD, FUTURE_SETTLEMENT, FX_DAILY_CUT.
- No decide el esquema JSON exacto del estado MTE (provisional).
- No decide si `df_stocks` en `compute_darkpool_signals` es obligatorio u opcional.

---

**Fin de la Parte A.**
**HEAD de referencia:** 42c395c (origin/main).
**Fecha:** 2026-09-15.
**Estado:** borrador para dictamen. Parte B pendiente.