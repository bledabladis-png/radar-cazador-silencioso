# FU-021-5 — Plan de implementación

**Versión:** v1
**Fecha:** 2026-09-16
**HEAD de referencia:** 5eda93a (origin/main)
**Base documental:** Parte A v2 (`5eda93a`) + Parte B ratificada (`468fc08`) + dictamen FU-021-5 Parte B
**Estado:** propuesta. Pendiente de ratificación.

---

## 0. Alcance y principios

### 0.1. Objetivo

Planificar la implementación de FU-021-5 tras la ratificación de la Parte A v2 y la Parte B. Esta implementación transforma la especificación documental en:

- Módulos nuevos que resuelven los 9 contratos.
- Propagación de metadata a los 30 consumidores.
- Adaptación de los 20 writers históricos.
- Rediseño del estado MTE.
- Migración de `darkpool`.
- Activación secuencial de contratos.
- Retirada de `trim_to_last_valid_date` (A3.1).

### 0.2. Orden de fases (según dictamen)
Parte A v2 ✅ publicado (5eda93a)

Implementación contractual → este plan

Propagación de metadata → este plan

Writers → este plan

MTE state → este plan

darkpool → este plan

Activación secuencial → este plan

A3.1 → este plan

```

**Paralelo:** FU-021-3C-bis (investigación de provider dedicado para futures). No bloquea el resto.

### 0.3. Principios de implementación

1. **Un cambio = una verificación = un commit.** Cada fase produce commits discretos y verificables.
2. **Local-first.** Refactors grandes acumulan commits locales; push único al cierre de fase.
3. **Backup .orig antes de cada patch.**
4. **Tests obligatorios antes de commit.**
5. **Rollback quirúrgico por fase** (ver §5).
6. **No activar nada hasta que todas las condiciones de esa fase estén verdes.**
7. **No mezclar fases.** Si una fase destapa problema de otra, se reordena o se abre follow-up.

### 0.4. Terminología

Nomenclatura consolidada en Parte A v2 §0.4:

- Productor, Consumidores (30), Consumidor de artefacto, Orquestador.
- Familias (5), Contratos (9).
- FSM: `PENDING | OK | STALE | INSUFFICIENT | BLOCKED`.

### 0.5. Decisiones del dictamen del plan (2026-09-16)

Ajustes obligatorios aplicados al plan v1:

| # | Decisión | Sección afectada |
|---|---|---|
| A1 | `temporal_meta` como transporte explícito; `df.attrs` solo auxiliar | §2.5, Fase 3 |
| A2 | Snapshot + `migration boundary` para `darkpool_history.csv` | Fase 7 |

Decisiones adicionales del dictamen:

- **D1:** No existe "fecha global" de `df_market`. Solo `temporal_meta` con `per_contract` + `consolidated`. La fecha de consolidación técnica no sustituye las fechas por contrato.
- **D2:** `BLOCKED` ≠ `STALE`. `STALE` = contrato válido con lag tolerado; `BLOCKED` = contrato no auditable/activable.

---

## 1. Mapa de módulos

### 1.1. Módulos nuevos

Ubicación propuesta: `src/temporal_contracts/` (directorio nuevo).

| Módulo | Rol | Contratos |
|---|---|---|
| `__init__.py` | Punto de entrada, expone API pública | — |
| `base.py` | Interfaz común `TemporalContract` + FSM | — |
| `equity_eod.py` | Contrato EQUITY_EOD | 1 |
| `index_eod.py` | Contratos INDEX_EOD_* (4) | 4 |
| `volatility_index.py` | Contrato VOLATILITY_INDEX | 1 |
| `rate_yield.py` | Contrato RATE_YIELD | 1 |
| `future_settlement.py` | Contrato FUTURE_SETTLEMENT (BLOCKED) | 1 |
| `fx_daily_cut.py` | Contrato FX_DAILY_CUT | 1 |
| `consolidate.py` | Operación de consolidación | — |
| `registry.py` | Registro de contratos activos | — |

**Total:** 10 ficheros, 9 contratos + infraestructura.

### 1.2. Módulos afectados (no nuevos)

| Módulo | Cambio |
|---|---|
| `src/data_loader.py` | Productor: invoca contratos, consolida, adjunta `temporal_meta` |
| `src/utils.py` | Helper `get_effective_meta()` |
| `src/instrument_registry.py` | (ya tiene `get_instrument_class`) — verificar que cubre las 5 familias |
| `run.py` | Propaga `df_stocks` a `compute_market_data` (darkpool) |
| `src/pipeline/*.py` (12) | Consumidores Capa 1: leen `temporal_meta` |
| `regimes/*.py` (3) | Consumidores Capa 2a: leen `temporal_meta` |
| `indicators/*.py` (15) | Consumidores Capa 2b: leen `temporal_meta` |
| `indicators/darkpool.py` | Firma nueva, elimina lectura de parquet |
| `indicators/mte.py` | Estado versionado + `futures_status=BLOCKED` |
| `config/settings.py` | `CURRENT_TEMPORAL_CONTRACT_VERSION` |

### 1.3. Writers afectados (20)

Lista completa en Parte A v2 §5.1. Distribución por fase:

- Fase 4 (Writers): 20 writers, patrón P0–P8.
- Afecta a: `engines.py`, `sectors_base.py`, `flows_primary.py`, `breadth_metrics.py`, `market_data.py`, `cross_asset_context.py`, `sector_correlation.py`, `rs_internal.py`, `sector_leader_divergence.py`, `volatility_structure.py`, `stock_leader.py`, `mte.py`.

---

## 2. API del contrato temporal

### 2.1. Interfaz común
class TemporalContract(ABC):
name : str # "EQUITY_EOD", etc.
family : str # "EQUITY", "INDEX", ...
eligible_universe : list[str] # tickers
session_calendar : str # "NYSE", etc.
max_lag_days : int | dict[str, int] # global o por ticker/par
min_coverage : float
per_ticker_lag : dict[str, int] | None
per_pair_max_lag : dict[str, int] | None
activation_req : dict | None # para FUTURE

@abstractmethod
def resolve(self, df_market, reference_date) -> TemporalResolution:
"""Resuelve effective_date, expected_date, lag_days, coverage, status."""

def is_eligible(self, ticker: str) -> bool:
"""True si el ticker pertenece al universo del contrato."""

```

### 2.2. Resultado de resolución
@dataclass
class TemporalResolution:
contract_name : str
effective_date: date | None
expected_date : date | None
lag_days : int | None
coverage : float | None
status : str # "OK" | "STALE" | "INSUFFICIENT" | "BLOCKED" | "PENDING"
per_member : dict[str, dict] | None # desglose por ticker/par (opcional)

```

### 2.3. API pública (expuesta por `__init__.py`)
def resolve_all_contracts(df_market, reference_date) -> dict[str, TemporalResolution]:
"""Resuelve los 9 contratos y devuelve el dict de resoluciones."""

def consolidate(df_market, resolutions: dict) -> pd.DataFrame:
"""Consolida df_market con temporal_meta adjunto."""

def get_effective_meta(df_market, contracts: list[str]) -> dict:
"""Devuelve metadata efectiva para los contratos indicados."""

def get_contract(name: str) -> TemporalContract:
"""Devuelve el contrato por nombre."""

```

### 2.4. Metadata adjunta a `df_market`
df_market.temporal_meta = {
'by_contract': {
'EQUITY_EOD': {...},
'INDEX_EOD_USA': {...},
'INDEX_EOD_EUROPA': {...},
'INDEX_EOD_COMMODITY': {...},
'INDEX_EOD_CURRENCY': {...},
'VOLATILITY_INDEX': {...},
'RATE_YIELD': {...},
'FUTURE_SETTLEMENT': {'status': 'BLOCKED', ...},
'FX_DAILY_CUT': {...},
},
'global_last_date': date,
'reference_date': datetime,
'run_id': str,
}

```

### 2.5. Implementación de `temporal_meta` sobre DataFrame

**Ratificado por dictamen Q-P.3 (2026-09-16).**

**Autoridad:** `temporal_meta` es un **dict explícito** transportado junto con `df_market`. No es un atributo del DataFrame. Es una estructura separada que se propaga como par de retorno o dataclass.

**Espejo auxiliar (opcional):** `df.attrs['temporal_meta']` puede utilizarse como cache/conveniencia, pero **nunca como fuente de verdad**.

**Mecanismo autorizado:**

`dict` paralelo. El productor retorna `(df_market, temporal_meta)` o un `MarketDataBundle` dataclass.

**Razón del dictamen:**

`df.attrs` depende de cómo pandas propague/copie atributos durante `.loc`, `.copy`, `concat`, `merge`, `groupby` y transformaciones intermedias. Aunque un test demuestre que sobrevive hoy, eso no lo convierte en un contrato fuerte.

**Implementación concreta:**

    @dataclass
    class MarketDataBundle:
        df_market: pd.DataFrame
        temporal_meta: dict

Los consumidores reciben `bundle` y acceden a `bundle.df_market` y `bundle.temporal_meta`.

**Espejo auxiliar (opcional):** `bundle.df_market.attrs['temporal_meta'] = bundle.temporal_meta` puede mantenerse como conveniencia. No es autoridad.

## 3. Fases de implementación

### 3.1. Resumen de fases

| # | Fase | Alcance | Dependencia | Complejidad |
|---|---|---|---|---|
| 1 | Módulos base | `base.py`, `registry.py`, `__init__.py` | — | Baja |
| 2 | Contratos individuales | 9 contratos | Fase 1 | Media |
| 3 | Consolidación | `consolidate.py`, `data_loader.py` | Fase 2 | Media |
| 4 | Propagación metadata | `utils.py` + 30 consumidores | Fase 3 | Alta |
| 5 | Writers | 20 writers | Fase 4 | Alta |
| 6 | MTE state versionado | `mte.py` | Fase 4 | Media |
| 7 | darkpool migración | `darkpool.py` + `market_data.py` + `run.py` | Fase 4 | Media |
| 8 | Activación secuencial | Contratos por orden | Fases 1-7 | Media |
| 9 | A3.1 | Retirar `trim_to_last_valid_date` | Fases 1-8 | Alta |

**Total:** 9 fases. Las fases 1–3 son mecánicas. Las fases 4–7 son refactor de consumidores. Las fases 8–9 son activación y cierre.

### 3.2. Detalle por fase

Cada fase incluye: objetivo, alcance, API, tests, criterio de aceptación, commits estimados, rollback.

#### Fase 1 — Módulos base

**Objetivo:** crear infraestructura del sistema de contratos.

**Alcance:**
- `src/temporal_contracts/__init__.py` (API pública).
- `src/temporal_contracts/base.py` (interfaz `TemporalContract`, FSM).
- `src/temporal_contracts/registry.py` (registro de los 9 contratos).

**API:**
- Clase abstracta `TemporalContract`.
- `Dataclass TemporalResolution`.
- FSM: función `compute_status(effective_date, expected_date, lag_days, coverage, max_lag, min_coverage) -> str`.

**Tests:**
- `test_temporal_contracts_base.py`:
  - FSM produce `OK` para casos correctos.
  - FSM produce `STALE` para lag positivo ≤ max.
  - FSM produce `INSUFFICIENT` para coverage < min.
  - FSM produce `BLOCKED` cuando `activation_req` no resuelto.
  - FSM produce `PENDING` para contratos sin resolver.
- `test_temporal_contracts_registry.py`:
  - 9 contratos registrados.
  - Nombres coinciden con Parte A v2.
  - Familias correctas.

**Criterio de aceptación:**
- compileall OK.
- pyflakes 0 warnings.
- Tests nuevos pasan.
- Suite global sin regresión (369 + N).

**Commits estimados:** 2.
1. `feat(temporal_contracts): base.py + FSM`.
2. `feat(temporal_contracts): registry + __init__`.

**Rollback:** eliminar directorio `src/temporal_contracts/`.

#### Fase 2 — Contratos individuales

**Objetivo:** implementar los 9 contratos.

**Alcance:**
- 7 ficheros de contrato (ver §1.1).

**API:** cada contrato implementa `resolve(df_market, reference_date) -> TemporalResolution`.

**Tests:** un fichero por contrato (~2-4 tests cada uno). Total ~25 tests nuevos.

**Criterio de aceptación:**
- Los 9 contratos resuelven correctamente contra `df_market` actual.
- `EQUITY_EOD` produce resultados equivalentes a `resolve_effective_date` (FU-020).
- `FUTURE_SETTLEMENT` retorna `status=BLOCKED` sin excepción.
- Suite global verde.

**Commits estimados:** 4.
1. `feat(temporal_contracts): EQUITY_EOD + INDEX_EOD_*`.
2. `feat(temporal_contracts): VOLATILITY_INDEX + RATE_YIELD`.
3. `feat(temporal_contracts): FUTURE_SETTLEMENT (BLOCKED)`.
4. `feat(temporal_contracts): FX_DAILY_CUT`.

**Rollback:** revertir commits de la fase.

#### Fase 3 — Consolidación

**Objetivo:** integrar los contratos en el productor.

**Alcance:**
- `src/temporal_contracts/consolidate.py`.
- `src/data_loader.py`: invocar `resolve_all_contracts` + `consolidate`.
- `src/utils.py`: helper `get_effective_meta`.
- `config/settings.py`: `CURRENT_TEMPORAL_CONTRACT_VERSION`.
- **Test empírico de persistencia:** verificar que `df.attrs['temporal_meta']` sobrevive a operaciones reales del pipeline. **Solo como referencia** (no es autoridad tras dictamen Q-P.3).
- **Transporte explícito:** `temporal_meta` viaja como parte de un `MarketDataBundle` (dataclass) o par de retorno.

**API:**
- `resolve_all_contracts(df_market, reference_date)` → dict.
- `consolidate(df_market, resolutions)` → df con `temporal_meta`.
- `get_effective_meta(df_market, contracts)` → dict.

**Tests:**
- `test_temporal_contracts_consolidate.py`:
  - `df_market.temporal_meta` contiene 9 entradas.
  - `global_last_date = max(effective_date)` de contratos OK/STALE.
  - `FUTURE_SETTLEMENT` en `BLOCKED`.
  - Universos disjuntos.
- `test_data_loader_temporal_meta.py`:
  - `download_market_data` adjunta `temporal_meta`.
  - Contratos resueltos loguean `[FU-021-5]`.

**Criterio de aceptación:**
- `download_market_data` devuelve `df_market` con `temporal_meta`.
- Los datos del parquet son idénticos a antes (no cambia el contenido, solo se adjunta metadata).
- Suite global verde.
- Log `[FU-021-5]` visible en ejecución manual de `py run.py`.

**Commits estimados:** 2.
1. `feat(temporal_contracts): consolidate.py`.
2. `feat(data_loader): integrar resolve_all + consolidate`.

**Rollback:** revertir commits; `df_market` vuelve a no tener metadata.

#### Fase 4 — Propagación de metadata

**Objetivo:** que los 30 consumidores lean `temporal_meta`.

**Alcance:**
- Capa 1 (12 consumidores): sustituir `df_market.index[-1]` por consultas a `temporal_meta`.
- Capa 2a (3 consumidores): mismos cambios.
- Capa 2b (15 indicadores): mismos cambios.

**API:** cada consumidor declara qué contratos usa:
def compute_xxx(df_market, ...):
meta = get_effective_meta(df_market, ['EQUITY_EOD'])

...
```

**Tests:**
- Un test por consumidor que verifique:
  - Lee `temporal_meta` correctamente.
  - Declara el contrato en su output.
  - No usa `df_market.index[-1]` directo (test estructural por grep).

**Criterio de aceptación:**
- Ningún consumidor contiene `df_market.index[-1]` ni `iloc[-1]` sobre df completo (solo sobre subconjuntos).
- Todos declaran `effective_date` y `coverage` en sus outputs.
- Suite global verde.
- Report markdown sale idéntico o con cambios documentados.

**Commits estimados:** ~6 (2 por capa).
1. `feat(capa1): propagar temporal_meta en 12 consumidores`.
2. `test(capa1): verificacion estructural`.
3. `feat(capa2a): propagar temporal_meta en 3 consumidores`.
4. `test(capa2a): verificacion estructural`.
5. `feat(capa2b): propagar temporal_meta en 15 indicadores`.
6. `test(capa2b): verificacion estructural`.

**Rollback:** revertir cada sub-bloque por separado.

**Riesgo:** alto. Cada consumidor puede tener comportamiento no trivial. Se recomienda hacer capa por capa y verificar con `py run.py` entre sub-bloques.
#### Fase 5 — Writers

**Objetivo:** adaptar los 20 writers históricos para que escriban las 4 columnas temporales.

**Alcance:**
- 20 writers identificados en Parte A v2 §5.1.
- 9 patrones (P0–P8) con estrategia de reescritura (Parte A v2 §5.4).

**API:** cada writer recibe `temporal_meta` y escribe:
date, effective_date, expected_date, coverage

```

**Tests:**
- Un test por patrón P0–P8.
- Test estructural: `grep` sobre el repo buscando `_observation_date_from_df(` en writers históricos debe devolver 0.
- Test funcional: para cada CSV, verificar que las 4 columnas existen y son coherentes.

**Criterio de aceptación:**
- Los 20 writers escriben las 4 columnas.
- Ningún writer usa walk-back local.
- Ningún writer usa `df_market.index[-1]` directo.
- CSV históricos son compatibles (columnas añadidas, no eliminadas).
- Suite global verde.

**Commits estimados:** ~4 (por grupo de patrón).
1. `fix(writers): P0 + P1`.
2. `fix(writers): P2 + P3 + P4`.
3. `fix(writers): P5 + P6 + P7`.
4. `fix(writers): P8 (mte, ver Fase 6)`.

**Rollback:** revertir commits; CSV vuelven al formato anterior.

**Riesgo:** medio. Añadir columnas a CSV históricos puede afectar consumidores de esos CSV. Requiere verificar a mano.

#### Fase 6 — MTE state versionado

**Objetivo:** rediseñar el estado persistente de `mte.py` según Parte A v2 §7.

**Alcance:**
- `indicators/mte.py`: eliminar `STATE_FILE` hardcoded (L1290).
- `indicators/mte.py`: usar solo `MTE_STATE_FILE` de config.
- `config/settings.py`: `CURRENT_TEMPORAL_CONTRACT_VERSION`.
- Esquema JSON versionado (§7.2 Parte A v2).
- Reset por cambio de contrato (§7.3 Parte A v2).
- Campo `futures_status` en JSON (§6.4 Parte A v2).

**API:**
def load_previous_scenario() -> tuple[str, str|None]:
"""Lee estado. Reset si temporal_contract_version difiere."""

def save_scenario(scenario, pending=None, metadata=None):
"""Escribe estado con versionado y metadata temporal."""

```

**Tests:**
- `test_mte_state_versioning.py`:
  - Schema version presente.
  - `temporal_contract_version` presente.
  - Reset si versión difiere.
  - `futures_status=BLOCKED` cuando FUTURE_SETTLEMENT bloqueado.
  - Path único (`MTE_STATE_FILE`).

**Criterio de aceptación:**
- `mte_state.json` contiene `schema_version` y `temporal_contract_version`.
- Un cambio de versión resetea el estado.
- El estado antiguo se descarta silenciosamente en el primer run tras activación (con log).
- Suite global verde.

**Commits estimados:** 2.
1. `feat(mte): estado versionado + reset por contrato`.
2. `feat(mte): futures_status en estado`.

**Rollback:** revertir commits; el estado antiguo sigue en disco.

**Riesgo:** bajo-medio. El estado es JSON, no histórico. Reset es trivial.

#### Fase 7 — darkpool migración

**Objetivo:** eliminar la lectura directa del parquet (Parte A v2 §8).

**Alcance:**
- `indicators/darkpool.py`: firma `compute_darkpool_signals(df_market, df_stocks=None)`.
- `indicators/darkpool.py`: eliminar `pd.read_parquet` (L189, L194).
- `src/pipeline/market_data.py`: `_compute_darkpool(df_market, df_stocks)`.
- `src/pipeline/market_data.py`: firma `compute_market_data(df_market, df_stocks=None)`.
- `run.py` L153: pasar `df_stocks` a `compute_market_data`.

**API:**
def compute_darkpool_signals(df_market, df_stocks=None) -> dict | None:
"""Consume df_market y df_stocks en memoria. Sin lectura de parquet."""

def compute_market_data(df_market, df_stocks=None) -> dict:
"""Pasa df_stocks a darkpool."""

```

**Tests:**
- `test_darkpool_migracion.py`:
  - `compute_darkpool_signals(df_market)` funciona sin parquet.
  - `compute_darkpool_signals(df_market, df_stocks)` cubre ambos universos.
  - El módulo no contiene `pd.read_parquet('data/market_data.parquet')` (test estructural).
  - El módulo no contiene `pd.read_parquet('data/stock_prices.parquet')` (test estructural).
- `test_market_data_darkpool.py`:
  - `compute_market_data` propaga `df_stocks`.
  - `run.py` pasa `df_stocks` a `compute_market_data`.

**Snapshot previo a la migración (ratificado por dictamen Q-P.7):**

Antes de modificar `darkpool.py`:

- `SHA256` del histórico `outputs/history/darkpool_history.csv`.
- Número de filas.
- Última fecha.
- Backup del CSV a `outputs/history/darkpool_history_pre_fu0215.csv`.
- Documentar `migration boundary` = (fecha, commit hash) para distinguir filas anteriores y posteriores.

**Criterio de aceptación:**
- `darkpool.py` ya no lee parquet directo.
- `darkpool_history.csv` se sigue generando.
- Snapshot del histórico creado y registrado.
- `migration boundary` documentado.
- Los valores de `dark_pool_pct` no cambian respecto a la versión previa.
- Suite global verde.
- Ejecución de `py run.py` produce `darkpool_data` sin warnings.

**Commits estimados:** 3.
1. `refactor(darkpool): firma con df_market + df_stocks`.
2. `refactor(darkpool): eliminar lectura directa de parquet`.
3. `refactor(market_data, run): propagar df_stocks a darkpool`.

**Rollback:** revertir commits; darkpool vuelve a leer parquet.

**Riesgo:** medio. `darkpool.py` genera estado persistente entre runs. Verificar que el histórico `darkpool_history.csv` no se rompe.

#### Fase 8 — Activación secuencial

**Objetivo:** activar los contratos en el orden aprobado por dictamen.

**Alcance (orden ratificado en Q-B.6):**
1. INDEX_EOD_USA
2. INDEX_EOD_EUROPA
3. VOLATILITY_INDEX
4. RATE_YIELD
5. FX_DAILY_CUT
6. FUTURE_SETTLEMENT → permanece `BLOCKED`

**API:** cada activación cambia el estado del contrato:
registry.activate('INDEX_EOD_USA') # PENDING → OK/STALE

```

**Tests:**
- Un test por activación.
- Verifica que los consumidores afectados por cada contrato pasan de `future_status` a estado real.
- Verifica que la desactivación de un contrato no rompe otros.

**Criterio de aceptación:**
- Cada contrato activado aparece como `OK` o `STALE` en `temporal_meta`.
- Los consumidores declaran `effective_date` correcto en outputs.
- El report markdown refleja las fechas reales.
- FUTURE_SETTLEMENT permanece `BLOCKED` con `future_status=BLOCKED` en consumidores afectados.
- Suite global verde.
- Ejecución de `py run.py` completa sin errores.

**Commits estimados:** ~5 (uno por contrato activado, o agrupados).
1. `feat(activation): INDEX_EOD_USA`.
2. `feat(activation): INDEX_EOD_EUROPA`.
3. `feat(activation): VOLATILITY_INDEX + RATE_YIELD`.
4. `feat(activation): FX_DAILY_CUT`.
5. `docs(activation): FUTURE_SETTLEMENT permanece BLOCKED`.

**Rollback:** desactivar cada contrato individualmente (estado revierte a `PENDING`).

**Riesgo:** medio-alto. Primer contacto con datos reales de cada contrato. Verificar cada activación con `py run.py` antes de la siguiente.

#### Fase 9 — A3.1

**Objetivo:** retirar `trim_to_last_valid_date` de `data_load.py`.

**Alcance:**
- `src/pipeline/data_load.py`: eliminar import y llamada al trim.
- Docstring actualizada.
- Tests que asuman el trim se actualizan.

**API:** ninguna nueva. Es una eliminación.

**Tests:**
- `test_a31_sin_trim.py`:
  - `load_all_data` no aplica trim.
  - `df_market` devuelto no tiene el trim.
  - Los consumidores siguen funcionando sin el trim.
  - El report markdown sigue siendo coherente.

**Criterio de aceptación:**
- `data_load.py` no contiene `trim_to_last_valid_date`.
- Ejecución real de `py run.py` produce report idéntico o con cambios documentados.
- Gate 10/10.
- Suite global verde.

**Commits estimados:** 2.
1. `refactor(data_load): retirar trim_to_last_valid_date`.
2. `test(data_load): verificacion post-retirada`.

**Rollback:** restaurar el import y la llamada. Trivial.

**Riesgo:** alto. Es la retirada final. Requiere que todas las fases anteriores estén cerradas y verificadas.

---

## 4. Resumen de commits por fase

| Fase | Commits estimados | Riesgo |
|---|---|---|
| 1. Módulos base | 2 | Bajo |
| 2. Contratos individuales | 4 | Bajo |
| 3. Consolidación | 2 | Bajo |
| 4. Propagación metadata | 6 | Alto |
| 5. Writers | 4 | Medio |
| 6. MTE state | 2 | Bajo-Medio |
| 7. darkpool | 3 | Medio |
| 8. Activación secuencial | 5 | Medio-Alto |
| 9. A3.1 | 2 | Alto |
| **Total** | **~30 commits** | — |

**Estimación de sesiones:** 8–12 sesiones de trabajo (2–4 horas/sesión).

---

## 5. Plan de rollback

### 5.1. Estrategia general

- **Backup .orig** antes de cada patch individual.
- **Commit por bloque lógico.** Cada commit es revertible por separado.
- **`git revert`** como mecanismo estándar.
- **Estado de emergencia:** `git reset --hard <commit_anterior>` si se detecta regresión grave.

### 5.2. Rollback por fase

| Fase | Rollback |
|---|---|
| 1 | `git revert` o `rm -rf src/temporal_contracts/` |
| 2 | `git revert` de los 4 commits |
| 3 | `git revert` de los 2 commits; `df_market` deja de tener `temporal_meta` |
| 4 | `git revert` por capa (Capa 1, 2a, 2b independientes) |
| 5 | `git revert` por grupo de patrón |
| 6 | `git revert`; el estado antiguo sigue en disco |
| 7 | `git revert`; darkpool vuelve a leer parquet |
| 8 | Desactivar contratos uno a uno (sin revertir código) |
| 9 | `git revert` de los 2 commits |

### 5.3. Estado de emergencia

Si una fase produce regresión en producción:

1. **Detener el cron.**
2. **`git revert <commit>`** de la fase.
3. **`git push`** inmediato.
4. **Ejecución manual de `py run.py`** para verificar recuperación.
5. **Documentar en `FOLLOWUPS.md`.**

---

## 6. Riesgos identificados

### 6.1. Riesgos técnicos

| # | Riesgo | Fase | Mitigación |
|---|---|---|---|
| R1 | `df.attrs['temporal_meta']` no sobrevive a operaciones pandas | 3 | Verificar empíricamente antes de implementar. Si falla, usar dict paralelo |
| R2 | Consumidores con lógica no trivial (ej. `sectors_base.py` con 7 writers) | 4 | Verificar uno a uno con `py run.py` |
| R3 | CSV históricos rompen por columnas nuevas | 5 | Verificar manualmente cada CSV post-cambio |
| R4 | Reset de MTE state genera transición de escenario inesperada | 6 | Ejecutar en entorno controlado antes de producción |
| R5 | `darkpool.py` usa estado persistente entre runs | 7 | Verificar `darkpool_history.csv` post-migración |
| R6 | ACTIVACIÓN de un contrato rompe consumidor no verificado | 8 | Activar de uno en uno, verificar cada uno |
| R7 | Retirada del trim produce deriva en report | 9 | Diff contra snapshot previo |

### 6.2. Riesgos de proceso

| # | Riesgo | Mitigación |
|---|---|---|
| P1 | El plan se rompe por hallazgo imprevisto | Fase puede reordenarse o abrir follow-up |
| P2 | Múltiples sesiones producen divergencia | Actualizar este plan entre fases |
| P3 | Cambio de contrato en Parte A v2 durante implementación | Congelar Parte A v2 hasta completar fases 1–9 |

### 6.3. Riesgos de regresión silenciosa

| # | Riesgo | Mitigación |
|---|---|---|
| S1 | Consumidor produce output sin declarar cobertura | Test estructural por grep |
| S2 | Writer sigue usando `_observation_date_from_df` | Test estructural por grep |
| S3 | `darkpool.py` sigue leyendo parquet | Test estructural |
| S4 | MTE state no resetea en cambio de contrato | Test de schema version |

---

## 7. Criterios de cierre

### 7.1. Cierre por fase

Cada fase se cierra cuando:

1. Todos los tests pasan.
2. compileall OK.
3. pyflakes 0 warnings.
4. Suite global sin regresión.
5. Commits pusheados.
6. Documentación actualizada si aplica.

### 7.2. Cierre de la implementación completa

La implementación completa se cierra cuando:

1. Las 9 fases están cerradas.
2. Los 9 contratos resuelven correctamente en producción.
3. Los 30 consumidores leen `temporal_meta`.
4. Los 20 writers escriben las 4 columnas temporales.
5. El estado MTE está versionado.
6. `darkpool` no lee parquet directo.
7. `FUTURE_SETTLEMENT` permanece `BLOCKED`.
8. `trim_to_last_valid_date` retirado.
9. Report markdown coherente.
10. Gate 10/10.
11. Ejecución real de `py run.py` completa.

### 7.3. Cierre de A3.1

A3.1 se cierra tras la fase 9. Requiere las 6 condiciones del dictamen previo:

| Condición | Fase |
|---|---|
| Contrato de cada clase | Fases 1-2 |
| Regla de consolidación | Fase 3 |
| Propagación de metadata | Fase 4 |
| Writers adaptados | Fase 5 |
| MTE compatible | Fase 6 |
| darkpool fuera de fuente paralela | Fase 7 |

---

## 8. Trabajo paralelo

### 8.1. FU-021-3C-bis — Provider dedicado para futures

**Objetivo:** investigar alternativas a Yahoo continuo para `FUTURE_SETTLEMENT`.

**Caminos:**
1. Contratos explícitos (`GCZ26.CMX`) con lógica de rollover declarada.
2. Provider dedicado (CME directo, ICE).
3. Congelar `BLOCKED` permanentemente.

**No bloquea** las fases 1–9. Puede ejecutarse en paralelo.

**Producto:** informe técnico + recomendación para desbloquear `FUTURE_SETTLEMENT`.

### 8.2. Actualización de Follow-ups

Con cada fase cerrada, actualizar `docs/auditoria/FOLLOWUPS.md` con:
- Estado de la fase.
- Hallazgos nuevos.
- Cambios de scope.

---

## 9. Lo que este plan NO decide

- No activa ningún contrato.
- No retira `trim_to_last_valid_date`.
- No decide el provider para FUTURE_SETTLEMENT.
- No decide el cutoff exacto de FX.
- No decide si `df_stocks` en `compute_darkpool_signals` es obligatorio u opcional.
- No decide si `df.attrs` o dict paralelo para `temporal_meta` (se decide en Fase 3).
- No define la implementación exacta de la FSM (Fase 1).
- No define los módulos concretos de cada contrato (Fase 2).

---

## 10. Preguntas al auditor

### Q-P.1 — Orden de las fases

¿Aprueba el orden propuesto (1 base → 2 contratos → 3 consolidación → 4 propagación → 5 writers → 6 MTE → 7 darkpool → 8 activación → 9 A3.1), o prefiere reordenar?

### Q-P.2 — Ubicación de los módulos nuevos

¿Aprueba `src/temporal_contracts/` como directorio para los módulos nuevos, o prefiere otra ubicación (`src/contracts/`, `src/temporal/`)?

### Q-P.3 — `temporal_meta` sobre DataFrame

¿Aprueba el uso de `df.attrs['temporal_meta']` (mecanismo pandas nativo), o prefiere dict paralelo explícito?

### Q-P.4 — Contratos y `resolve_effective_date`

`EQUITY_EOD` puede implementarse como wrapper sobre `resolve_effective_date` (FU-020) o como lógica independiente. ¿Prefiere wrapper (menor duplicación) o lógica independiente (mayor aislamiento)?

### Q-P.5 — Orden de commits por fase

¿Aprueba el número estimado de commits por fase (~30 totales), o prefiere granularidad distinta (menos commits, más compactos)?

### Q-P.6 — Fase 8 con 6 sub-fases

La activación secuencial (Fase 8) se plantea como 5-6 activaciones una a una. ¿Prefiere agrupar alguna (`INDEX_EOD_USA + VOLATILITY_INDEX + RATE_YIELD` como bloque común)?

### Q-P.7 — Rollback de Fase 7

`darkpool.py` escribe estado persistente (`darkpool_history.csv`). ¿Debe incluirse en el rollback o se preserva entre versiones?

### Q-P.8 — Trabajo paralelo

¿Aprueba iniciar FU-021-3C-bis en paralelo, o prefiere secuencia estricta?

---

**Fin del plan de implementación FU-021-5.**
**HEAD de referencia:** 5eda93a (origin/main).
**Fecha:** 2026-09-16.
**Estado:** propuesta. Pendiente de ratificación por dictamen.
