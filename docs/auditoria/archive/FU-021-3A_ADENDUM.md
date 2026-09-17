# FU-021-3A — Adendum de corrección documental

**Fecha:** 2026-09-15
**HEAD de referencia:** 7c44b4a (origin/main)
**Informe original:** FU-021-3A (commits 78b7583 + 747cfb1 + fded14b)
**Dictamen que motiva este adendum:** cierre del Ciclo 1 del subinforme de consumidores de `df_market`.
**Estado:** documento activo. No modifica el informe original, que queda como registro histórico.

---

## 1. Motivo del adendum

El informe FU-021-3A (§4.5, §7, §11) delega el subinforme previo a A3.1 sobre "los 13 consumidores de `df_market`". Ese conteo fue verificado durante la preparación del subinforme y resultó ser una descripción incompleta del universo real.

Este adendum:

1. Corrige el alcance documental del "13".
2. Registra la clasificación real por capas.
3. Fija la corrección semántica: `B ≠ listo para A3.1`.
4. Actualiza el estado de A3.1 tras el cierre del Ciclo 1.
5. Fija la regla sobre `trim_to_last_valid_date`.
6. Registra los hallazgos C1–C20 y H-DP-1 detectados en el Ciclo 1.

No reabre la implementación de FU-021-3A. No modifica el informe original. Es corrección documental.

---

## 2. Corrección del alcance documental

### 2.1. Qué decía el informe original

> *"Puede hacer que `df_market` que consumen los 13 módulos de pipeline no sea idéntico al parquet publicado."* — FU-021-3A §4.5

> *"Diferido hasta sub-informe de los 13 módulos consumidores."* — FU-021-3A §11

### 2.2. Qué es realmente

El "13" corresponde a **ficheros de `src/pipeline/` que mencionan `df_market`**:

| Grupo | Cuenta |
|---|---|
| Consumidores directos en `src/pipeline/` | 12 |
| Productor (`src/pipeline/data_load.py`) | 1 |
| **Total "13"** | **13** |

El informe etiquetó ese conjunto como "consumidores". Terminológicamente impreciso: uno es productor.

**Lo relevante no es el número. Es lo que el 13 excluye.**

### 2.3. Universo real

| Ficheros que mencionan `df_market` | Referencias |
|---|---|
| 33 | 171 |

**33 ficheros, no 13.** El "13" era un subconjunto del subconjunto.

### 2.4. Terminología adoptada (aprobada por auditor)

| Categoría | Definición |
|---|---|
| **Productor** | Crea/carga `df_market` |
| **Consumidor directo** | Recibe `df_market` por firma |
| **Consumidor indirecto** | Recibe `df_market` a través de otro módulo |
| **Consumidor de artefacto** | Lee `market_data.parquet` directamente |

Reservar "consumidor de `df_market`" para quien recibe el objeto por parámetro. No incluye al productor ni al consumidor de artefacto.

---

## 3. Clasificación real de los 33 ficheros

### 3.1. Capa 0 — Productor (1)

| Fichero | Rol |
|---|---|
| `src/pipeline/data_load.py` | Productor. Aplica `trim_to_last_valid_date` (L27). Punto de acción de A3.1. |

### 3.2. Capa 1 — Consumidores directos (12)

| Fichero |
|---|
| `src/pipeline/regimes.py` |
| `src/pipeline/sectors_base.py` |
| `src/pipeline/flows_primary.py` |
| `src/pipeline/leaders.py` |
| `src/pipeline/sector_metrics.py` |
| `src/pipeline/breadth_metrics.py` |
| `src/pipeline/engines.py` |
| `src/pipeline/slpm.py` |
| `src/pipeline/diagnostics.py` |
| `src/pipeline/market_data.py` |
| `src/pipeline/mte_confirmation.py` |
| `src/pipeline/indices_intl.py` |

### 3.3. Capa 2a — Consumidores indirectos de régimen (3)

| Fichero | Importado por | Referencias a `df_market` |
|---|---|---|
| `regimes/macro_regime.py` | `src/pipeline/regimes.py` | **23** |
| `regimes/structural_engine.py` | `src/pipeline/engines.py` | 3 |
| `regimes/tactical_engine.py` | `src/pipeline/engines.py` | 4 |

### 3.4. Capa 2b — Consumidores indirectos de indicadores (15)

| Fichero | Importado por |
|---|---|
| `indicators/commodity_market_correlation.py` | `pipeline/diagnostics.py` |
| `indicators/credit.py` | `regimes/macro_regime.py` |
| `indicators/cross_asset.py` | `pipeline/mte_confirmation.py` |
| `indicators/cross_asset_context.py` | `pipeline/sectors_base.py` |
| `indicators/index_leaders.py` | `pipeline/indices_intl.py` |
| `indicators/index_phase.py` | `pipeline/indices_intl.py` |
| `indicators/mte.py` | `pipeline/mte_confirmation.py` |
| `indicators/rs_internal.py` | `pipeline/sector_metrics.py` |
| `indicators/sector_breadth.py` | `pipeline/breadth_metrics.py` |
| `indicators/sector_correlation.py` | `pipeline/sectors_base.py` |
| `indicators/sector_leader_divergence.py` | `pipeline/sector_metrics.py` |
| `indicators/slpm_v12.py` | `pipeline/slpm.py` |
| `indicators/stock_leader.py` | `pipeline/leaders.py` |
| `indicators/vol_metrics.py` | `pipeline/mte_confirmation.py` |
| `indicators/volatility_structure.py` | `pipeline/market_data.py` |

### 3.5. Consumidor de artefacto (1)

| Fichero | Fuente | Firma |
|---|---|---|
| `indicators/darkpool.py` | `data/market_data.parquet` + `data/stock_prices.parquet` (lectura directa) | `compute_darkpool_signals()` — sin parámetros |

### 3.6. Orquestador (1)

| Fichero | Rol |
|---|---|
| `run.py` | Orquestador. Único llamante de `load_all_data` (L59). No es consumidor. |

### 3.7. Resumen por capa

| Capa | Cuenta |
|---|---|
| Productor | 1 |
| Consumidores directos (Capa 1) | 12 |
| Consumidores indirectos (Capa 2a) | 3 |
| Consumidores indirectos (Capa 2b) | 15 |
| Consumidor de artefacto | 1 |
| Orquestador | 1 |
| **TOTAL ficheros que mencionan `df_market`** | **33** |

---

## 4. Corrección semántica — `B ≠ listo para A3.1`

### 4.1. Error de lectura a evitar

El Ciclo 1 cerró con 14 fichas en veredicto `B`. Eso **no** significa "A3.1 puede ejecutarse".

### 4.2. Significado real de `B`

> **B — COMPATIBLE CONDICIONADO:** este consumidor no necesita modificación propia **si el productor cumple el nuevo contrato**.

`B` es un veredicto condicional, no un visto bueno.

### 4.3. Consecuencia

- Los 14 consumidores `B` dependen de que el productor entregue `df_market` con contrato temporal completo.
- El productor (`data_load.py`) **no tiene hoy** contrato temporal completo para el 100% del universo.
- Retirar el trim sin sustituto elimina una protección defectuosa sin poner la correcta.

---

## 5. Estado de A3.1

### 5.1. Decisión del dictamen de cierre de Ciclo 1

| Punto | Dictamen |
|---|---|
| **A3.1** | NO-GO — esperar FU-021-5 |
| C11/C14 | Integrar en FU-021-5, no parchear individualmente |
| C16/C19 | Ciclo 2 |
| Ciclo 2 | Autorizado a abrirse ya |
| Adendum FU-021-3A | Autorizado (este documento) |

### 5.2. Motivo del bloqueo

El universo de `df_market` es heterogéneo (562 tickers mixtos). El contrato temporal está resuelto solo parcialmente:

| Universo | Cuenta | Contrato temporal |
|---|---|---|
| EQUITY_EOD | 539 | Resuelto por FU-021-3A |
| INDEX_EOD | — | Pendiente FU-021-3B |
| RATE_YIELD | — | Pendiente FU-021-3B |
| FUTURE_SETTLEMENT | — | Pendiente FU-021-3C |
| FX_DAILY_CUT | — | Pendiente FU-021-3C |
| No-equity restante | 23 | Sin contrato |

Retirar el trim con el 96% del universo resuelto deja al 4% restante sin autoridad temporal.

### 5.3. Condición de desbloqueo

```
A3.1 NO-GO
Motivo: ausencia de contrato temporal completo para 562/562
Desbloqueo: FU-021-5 + contratos 3B/3C
```

### 5.4. Secuencia aprobada

```
CICLO 1
cerrado

        ↓

ADENDUM FU-021-3A (este documento)
corregir alcance + hallazgos

        ↓

CICLO 2
Capa 2b
darkpool
C16
C19
C20
auditoría de dependencias temporales

        ↓

FU-021-5
contratos por clase
definición de fecha efectiva por contrato
integración productor/consumidores

        ↓

FU-021-3B
INDEX_EOD / RATE_YIELD

        ↓

FU-021-3C
FUTURE_SETTLEMENT / FX_DAILY_CUT

        ↓

A3.1
retirar trim_to_last_valid_date
```

---

## 6. Regla fija sobre `trim_to_last_valid_date`

**Fijada por dictamen del auditor. Registro permanente.**

```
trim_to_last_valid_date()
→ NO se sustituye por otro trim genérico.
→ Cuando se elimine, la responsabilidad temporal existirá
  UNA SOLA VEZ, en el productor / contrato temporal.
→ Los consumidores consumen df_market ya resuelto.
→ Prohibido añadir walk-back local (last_market_session suelto)
  en writers o consumidores.
```

### 6.1. Sustituciones prohibidas

Prohibido:

```python
# NO hacer esto
last_market_session(df_market.index[-1])
```

Prohibido:

```python
# NO hacer esto
resolve_effective_date(df_market, ...)  # dentro de un consumidor
```

### 6.2. Fuente canónica

**`df_market` en memoria es la fuente canónica del pipeline.**

El parquet `data/market_data.parquet` es **artefacto publicado**, no fuente de verdad intra-run.

### 6.3. Excepción registrada

`indicators/darkpool.py` es la **única excepción conocida**: lee `data/market_data.parquet` directamente (L189). Migración asignada a Ciclo 2 (ver H-DP-1).

---

## 7. Registro de hallazgos

Todos los hallazgos detectados durante el Ciclo 1. Clasificación por ciclo asignado.

### 7.1. Hallazgos con acción en ciclo 2 o posterior

| ID | Descripción | Ficha origen | Sensibilidad a A3.1 | Acción requerida | Ciclo |
|---|---|---|---|---|---|
| **C11** | `sector_persistence.csv` es writer histórico con `_observation_date_from_df(df_market)` | 9 (`engines.py`) | Sí | Integrar en FU-021-5 | FU-021-5 |
| **C14** | `sectors_base.py` escribe 7 históricos con `df_market.index[-1]` directo | 11 | Sí | Integrar en FU-021-5 | FU-021-5 |
| **C16** | `sector_metrics` no recibe `effective_meta` de FU-020. R3 violada por diseño | 13 | Indirecta | Propagar `effective_meta` o documentar excepción | Ciclo 2 |
| **C18** | `market_data.py` es host de H-DP-1 vía `_compute_darkpool()` sin args | 16 | Sí | Corregir al migrar `darkpool` | Ciclo 2 |
| **C19** | `indices_intl` llama `download_stock_prices()` sin `reference_date` | 17 | Sí | Añadir `reference_date` o documentar excepción | Ciclo 2 (alta prioridad) |
| **C20** | `select_index_leaders(None, ...)` — primer arg `df_market=None` | 17 | Ninguna | Verificar si es código muerto o fallback | Ciclo 2 |
| **H-DP-1** | `darkpool.py` es consumidor por disco, no por firma | 5 | Bloquea cierre de A3.1 | Migrar a recibir `df_market` por parámetro | Ciclo 2 |

### 7.2. Hallazgos documentales (sin acción inmediata)

| ID | Descripción | Ficha origen | Ciclo |
|---|---|---|---|
| C1 | `financial_score` pasado como `liquidity_score` en `macro_regime` — naming engañoso, no bug | 6 | Documental |
| C2 | `real_liquidity_score` siempre `None` en producción (4º posicional no pasado) | 1 | Documental |
| C10 | `datetime.now()` como `reference_date` fallback en `breadth_metrics` | 8 | Documental |
| C12 | `datetime.now()` para calcular edad darkpool en `mte_confirmation` | 10 | Documental |
| C15 | 6 calendarios en `flows_primary` (6 providers con contratos propios) | 12 | Documental |

### 7.3. Hallazgos de robustez preexistente

| ID | Descripción | Ficha origen | Ciclo |
|---|---|---|---|
| C3 | `leader_breadth` aceptado y no usado en `structural_engine` | 2 | Registro |
| C4 | Aceleración mal calculada cuando `21 <= len(close_sector) < 26` en `tactical_engine` | 3 | Registro |
| C5 | `benchmark='^GSPC'` hardcoded en ambos engines | 2, 3 | Verificar si aplica a sector europeo |
| C6 | `mom20_prev` en `iloc[-6]` con `len >= 26` produce ventanas solapadas | 3 | Registro |
| C9 | `df_stocks_effective_meta` solo propagada en rama OK de `leaders` | 7 | Robustez |
| C13 | `'all_signals' in dir()` check frágil en `mte_confirmation` | 10 | Robustez |
| C17 | `.iloc[-21]` posicional en `diagnostics` — asume 21 sesiones consecutivas | 15 | Registro |

### 7.4. Conteo total

```
C1 – C20:    20 hallazgos
H-DP-1:       1 hallazgo estructural
TOTAL:       21 hallazgos
```

- 7 con acción asignada a Ciclo 2 o posterior.
- 5 documentales.
- 7 de robustez preexistente.

---

## 8. Cadenas de custodia temporal

Detectadas durante el Ciclo 1. Documentan cómo la información temporal viaja (o no) por el pipeline.

### 8.1. `reference_date` (de `run.py`)

**Alcanza a:** `leaders`, `breadth_metrics`.
**Debería alcanzar a:** `indices_intl` (C19).
**No alcanza a:** `regimes`, `engines`, `mte_confirmation`, `sectors_base`, `flows_primary`, `sector_metrics`, `slpm`, `diagnostics`, `market_data`.

**Consecuencia:** solo 2 de 17 fichas del Ciclo 1 pueden participar en controles temporales basados en la fecha del run.

### 8.2. `effective_meta` (FU-020)

**Generado por:** `leaders.py::compute_leaders`.
**Consumido por:** `mte_confirmation.py::compute_mte_confirmation` (vía `compute_advance_decline`).
**No pasa por:** ningún otro módulo.

**Consecuencia:** cadena de custodia 1→1 sobre 17 fichas. `sector_metrics` (C16) y otros consumidores que escriben histórico no declaran cobertura efectiva.

### 8.3. `expected_session` (B2)

**Generado y consumido en:** `breadth_metrics.py` exclusivamente.
**No alcanza a:** ningún otro módulo.

**Consecuencia:** B2 (session integrity) está aplicado a 1 de 17 fichas, no como política transversal.

---

## 9. Estado de los veredictos del Ciclo 1

| # | Ficha | Veredicto |
|---|---|---|
| 1 | `regimes/macro_regime.py` | B |
| 2 | `regimes/structural_engine.py` | B |
| 3 | `regimes/tactical_engine.py` | B |
| 4 | `src/pipeline/data_load.py` | P — Productor |
| 5 | `indicators/darkpool.py` | C — Ciclo 2 |
| 6 | `src/pipeline/regimes.py` | B |
| 7 | `src/pipeline/leaders.py` | B |
| 8 | `src/pipeline/breadth_metrics.py` | B |
| 9 | `src/pipeline/engines.py` | B |
| 10 | `src/pipeline/mte_confirmation.py` | B |
| 11 | `src/pipeline/sectors_base.py` | B |
| 12 | `src/pipeline/flows_primary.py` | B |
| 13 | `src/pipeline/sector_metrics.py` | B |
| 14 | `src/pipeline/slpm.py` | B |
| 15 | `src/pipeline/diagnostics.py` | B |
| 16 | `src/pipeline/market_data.py` | B |
| 17 | `src/pipeline/indices_intl.py` | B |

**Resumen:** 14 B, 1 C (H-DP-1), 1 P (productor), 0 D.

**Interpretación:** todos los consumidores `B` dependen de que el productor cumpla el nuevo contrato. `B` no autoriza retirar el trim.

---

## 10. Cierre y firmas

### 10.1. Alcance documental corregido

| Documento | Alcance declarado |
|---|---|
| FU-021-3A original | "13 consumidores" (ficheros de `src/pipeline/` que mencionan `df_market`) |
| FU-021-3A + este adendum | 33 ficheros; clasificación por capas; distinción semántica |

### 10.2. Decisiones ejecutadas

- Corrección del conteo (13 → 33, con desglose por capas).
- Corrección semántica (`B ≠ listo para A3.1`).
- Estado de A3.1 actualizado (NO-GO hasta FU-021-5).
- Regla fija sobre `trim_to_last_valid_date`.
- Registro de 21 hallazgos (C1–C20 + H-DP-1).
- Cadenas de custodia temporal documentadas.

### 10.3. Lo que este adendum NO decide

- No autoriza A3.1. Esa decisión depende de FU-021-5.
- No autoriza migración de `darkpool.py`. Asignado a Ciclo 2.
- No autoriza correcciones C11/C14/C16/C19/C20. Asignadas a Ciclo 2 o FU-021-5.
- No modifica el informe original FU-021-3A, que queda como registro histórico.

### 10.4. Referencias

- Informe original: `docs/auditoria/FU-021-3A_INFORME_Y_DICTAMEN.md` (commits 78b7583 + 747cfb1 + fded14b).
- Prompt maestro v6.14: sección 11.8 (manifest FU-002), sección 12 (limitaciones conocidas).

---

**Fin del adendum FU-021-3A.**
**HEAD de referencia:** 7c44b4a (origin/main).
**Fecha:** 2026-09-15.
**Estado:** activo, no reemplaza al informe original.
