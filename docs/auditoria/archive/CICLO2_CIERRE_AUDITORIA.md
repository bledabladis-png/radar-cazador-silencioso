# Ciclo 2 — Informe de cierre de auditoría

**Fecha:** 2026-09-15
**HEAD de referencia:** d3790b3 (origin/main)
**Ciclo auditado:** Ciclo 2 del subinforme derivado de FU-021-3A
**Alcance:** 15 indicadores de Capa 2b + cierre de C16/C19/C20 + reconfirmación de H-DP-1
**Motivo:** aportar al auditor externo evidencia consolidada antes de decidir arquitectura de FU-021-5 y posición de darkpool
**Estado:** pendiente de dictamen

---

## 1. Resumen ejecutivo

El Ciclo 2 cierra con:

- **32/32 fichas** de auditoría (17 de Ciclo 1 + 15 de Ciclo 2).
- **0 hallazgos C o D nuevos** no cubiertos por dictamen previo.
- **2 fixes ejecutados y pusheados:** C19 (33ba2fa) y C16 (d3790b3).
- **1 verificación cerrada:** C20 (parámetro vestigial, sin cambio).
- **1 hallazgo estructural reconfirmado:** H-DP-1 (darkpool como consumidor por disco).
- **28 hallazgos nuevos registrados** (C21–C48) más los 21 del Ciclo 1 = **49 hallazgos** en el registro acumulado.

**Pregunta central del informe:**

> **¿Existe evidencia suficiente para diseñar el contrato temporal global de df_market, aunque todavía no exista evidencia suficiente para activarlo?**

**Respuesta corta:** la evidencia permite especificar la arquitectura del contrato. No permite cerrar la semántica operativa de cada clase ni activarlo.

---

## 2. Contexto y alcance

### 2.1. Origen

El dictamen de cierre del Ciclo 1 (2026-09-15) autorizó:

- Ciclo 1: cerrado.
- A3.1: bloqueada hasta FU-021-5.
- Ciclo 2: autorizado a abrir.
- Adendum FU-021-3A: autorizado (publicado 694b172).

El Ciclo 2 tenía por objeto auditar la Capa 2b (15 indicadores con firma que recibe df_market), más los fixes C16, C19 y C20 ya asignados por dictamen.

### 2.2. Secuencia ejecutada

| Bloque | Contenido | Resultado |
|---|---|---|
| B0 | Verificación post-adendum | HEAD alineado, adendum verificado |
| B1 | C19 — indices_intl sin reference_date | Fix pusheado (33ba2fa) |
| B2 | C20 — select_index_leaders(None, ...) | A — vestigial. Sin cambio |
| B3 | C16 — propagación effective_meta | Fix pusheado (d3790b3) |
| B4 | Capa 2b (15 indicadores) | 15 × B. Sin C/D |
| B5 | darkpool migración | Diferido por decisión de secuencia |
| B6 | Cierre del Ciclo 2 | Este documento |

### 2.3. Universo auditado

| Capa | Ficheros | Fichas |
|---|---|---|
| Capa 0 (productor) | 1 | Ficha 4 |
| Capa 1 (consumidores directos) | 12 | Fichas 6–17 |
| Capa 2a (regímenes) | 3 | Fichas 1–3 |
| Capa 2b (indicadores) | 15 | Fichas 18–32 |
| Consumidor de artefacto | 1 | Ficha 5 |
| **Total** | **32** | **32/32** |

---

## 3. Estado del sistema tras el Ciclo 2

| Métrica | Valor |
|---|---|
| HEAD | d3790b3 = origin/main |
| Tests locales | 369 passed + 2 skipped (362 + 7 nuevos: 3 C19 + 4 C16) |
| Gate | 10/10 |
| pyflakes | 0 warnings |
| compileall | OK |
| Commits del Ciclo 2 | 33ba2fa (C19), d3790b3 (C16) |
| A3.1 | bloqueada (contrato incompleto) |
| FU-021-5 diseño | puede avanzar |
| FU-021-5 activación | bloqueada hasta diseño + 3B/3C |
| FU-021-3B/3C | bloqueadas |
| darkpool | pendiente de decisión de arquitectura |

---

## 4. Clasificación de hallazgos

Los 49 hallazgos del registro acumulado (C1–C48 + H-DP-1) se clasifican en seis categorías excluyentes según la acción que requieren.

### 4.1. Hallazgos que bloquean A3.1

Un hallazgo bloquea A3.1 si su resolución es requisito para retirar trim_to_last_valid_date sin sustituto. Ninguno de los 49 bloquea A3.1 por sí mismo: **A3.1 está bloqueada por diseño (contrato incompleto 562/562)**, no por un hallazgo concreto. Los hallazgos condicionan el cierre documental de A3.1 cuando se ejecute.

| ID | Descripción | Efecto |
|---|---|---|
| C11 | sector_persistence.csv escribe fecha desde _observation_date_from_df(df_market) | Su fecha cambiará con A3.1 |
| C14 | sectors_base.py escribe 7 históricos con df_market.index[-1] | Sus 7 fechas cambiarán con A3.1 |
| H-DP-1 | darkpool.py lee market_data.parquet de disco | Divergencia memoria/parquet hasta migración |

Estos tres ya están asignados por dictamen: C11/C14 → FU-021-5, H-DP-1 → decisión de arquitectura pendiente.

### 4.2. Hallazgos que condicionan FU-021-5

Hallazgos que deben integrarse en el diseño del contrato temporal antes de activarlo.

| ID | Descripción | Sensibilidad | Origen |
|---|---|---|---|
| C11 | sector_persistence.csv — patrón P1 | Alta | Ficha 9 |
| C14 | sectors_base.py — patrón P2 (7 writers) | Alta | Ficha 11 |
| C22 | cross_asset_context.py — patrón P3 | Alta | Ficha 20 |
| C27 | sector_leader_divergence.py — asimetría df_market vs df_stocks sin alinear | Alta | Ficha 25 |
| C28 | sector_leader_divergence.py — patrón P5 | Media | Ficha 25 |
| C26 | rs_internal.py — patrón P4 | Media | Ficha 23 |
| C31 | sector_breadth.py — patrón P6, modelo correcto | Positivo | Ficha 27 |
| C35 | volatility_structure.py — patrón P7 | Media | Ficha 29 |
| C38 | stock_leader.py — writer sin columna date (P0) | Media | Ficha 30 |
| C43 | mte.py — doble writer de estado con histéresis persistente | Alta | Ficha 32 |
| C48 | STATE_FILE y MTE_STATE_FILE divergentes | Media | Ficha 32 |
| C19 | indices_intl sin reference_date | RESUELTO (33ba2fa) | — |
| C16 | sector_metrics sin effective_meta | RESUELTO (d3790b3) | — |

**Observación clave:** hay **ocho patrones distintos (P0–P7) para decidir la fecha de un writer** más el patrón P8 (estado JSON sin fecha). Es la evidencia empírica más contundente de que la fecha de df_market es una propiedad implícita del pipeline, no un contrato explícito.

### 4.3. Hallazgos preexistentes meramente documentales

| ID | Descripción |
|---|---|
| C1 | financial_score pasado como liquidity_score en macro_regime — naming engañoso |
| C2 | real_liquidity_score siempre None en producción |
| C5 | benchmark='^GSPC' hardcoded en ambos engines de régimen |
| C10 | datetime.now() como reference_date fallback en breadth_metrics |
| C12 | datetime.now() para calcular edad darkpool en mte_confirmation |
| C15 | 6 calendarios en flows_primary (6 providers con contratos propios) |
| C21 | credit.py:36 dropna() reindexa implícitamente |
| C24 | index_phase.py:26 period='5y' sin reference_date |
| C29 | index_leaders.py:21 period='1y' sin reference_date |
| C30 | index_leaders.py alineación asimétrica (RS sí, Wyckoff no) |
| C33 | vol_metrics.py min_periods relajados |
| C34 | vol_metrics.py sin declarar fecha de observación (R1) |
| C39 | stock_leader.py flow_proxy_z guardado como EMA |
| C40 | slpm_v12.evaluate_slpm_v12 — df_market vestigial |
| C42 | slpm.py propaga df_market vestigial a slpm_v12 |
| C44 | credit_stress_score no recibe df_market |
| C45 | nfci_series/credit_oas_series siempre None |
| C47 | classify_mte desempate no alineado con matriz de transiciones |

### 4.4. Deudas de robustez

| ID | Descripción |
|---|---|
| C3 | leader_breadth no usado en structural_engine |
| C4 | Aceleración mal calculada cuando 21 <= len < 26 en tactical_engine |
| C6 | mom20_prev en iloc[-6] produce ventanas solapadas |
| C9 | df_stocks_effective_meta solo propagada en rama OK de leaders |
| C13 | 'all_signals' in dir() frágil en mte_confirmation |
| C17 | .iloc[-21] posicional en diagnostics |
| C23 | index_phase.py:19 línea muerta |
| C25 | index_phase.py:18,30 bare except |
| C32 | sector_breadth.py:as_of_date=None sin validación |
| C36 | volatility_structure.py pcr_data con campos opcionales |
| C37 | stock_leader.py:125-133 bloque rho calculado y descartado |
| C41 | slpm_v12._safe_mean definido antes de imports + antes de docstring |
| C46 | bare except en sector_rotation_score y compute_mte |

### 4.5. Hallazgos ya resueltos

| ID | Descripción | Commit |
|---|---|---|
| C16 | sector_metrics propaga effective_meta | d3790b3 |
| C19 | indices_intl propaga reference_date | 33ba2fa |
| C20 | select_index_leaders(None, ...) — vestigial, sin cambio | Verificación |

### 4.6. Hallazgos positivos / patrones reutilizables

| ID | Descripción | Aporte |
|---|---|---|
| C31 | sector_breadth.py — as_of_date explícito + is_market_day + slice de dfs + fecha validada | Modelo único correcto de writer temporal |
| C16 (patrón) | sector_metrics recibe effective_meta y lo usa como reference_date con fallback documentado | Modelo de propagación R3 |
| C19 (patrón) | compute_indices_intl(df_market, reference_date=None, run_id=None) — propagación desde run.py | Modelo de contract-first |
| _observation_date_from_df | Helper src/utils reutilizable | Centraliza extracción de fecha (aunque no normaliza) |

**C31 es el patrón de referencia.** Es el único writer del pipeline que cumple simultáneamente B2 (session integrity), R1 (declaración de cobertura implícita) y coherencia entre dfs.

---

## 5. Patrones de writer de fecha — consolidado P0–P8

Nueve patrones distintos para decidir "qué fecha escribe un writer histórico".

| Patrón | Módulos | Objeto de fecha | B2 | R1 |
|---|---|---|---|---|
| P0 | stock_leader.py (C38) | Sin columna date | No | No |
| P1 | engines.py (C11) | _observation_date_from_df(df_market) | No | No |
| P2 | sectors_base.py (C14, 7 writers) | df_market.index[-1] | No | No |
| P3 | cross_asset_context.py (C22), sector_correlation.py | _observation_date_from_df(returns_df) | No | No |
| P4 | rs_internal.py (C26) | _observation_date_from_df(intersect) | No | No |
| P5 | sector_leader_divergence.py (C28) | _observation_date_from_df(df_stocks) | No | No |
| P6 | sector_breadth.py (C31) | as_of_date validado | Sí | Parcial |
| P7 | volatility_structure.py (C35) | _observation_date_from_df(vix) | No | No |
| P8 | mte.py (C43) | Sin fecha (JSON de estado) | No | No |

**Total writers afectados por A3.1:** 20 writers con fecha heterogénea + 2 writers de estado JSON + 1 writer sin fecha.

**Diagnóstico:** no hay contrato. Cada módulo resuelve la fecha por su cuenta. R3 (FU-020) se cumple en 1 de 20 casos (C16, resuelto). R1 (FU-020) se cumple en 0.

---

## 6. Respuesta a la pregunta fundamental

> **¿Existe ya evidencia suficiente para diseñar el contrato temporal global de df_market, aunque todavía no exista evidencia suficiente para activarlo?**

### 6.1. Respuesta: la evidencia permite especificar la arquitectura

La evidencia acumulada permite especificar la arquitectura, las responsabilidades, las invariantes y los requisitos de validación de FU-021-5. No permite todavía cerrar la semántica operativa de cada contrato para las clases no verificadas ni activarlos en producción.

**Razones:**

1. **Los 32 módulos consumidores están clasificados.** Todos son B o P. Ninguno es C o D.
2. **Los 20 writers históricos están identificados.** Cada uno con su patrón P0–P8 documentado.
3. **El modelo correcto ya existe en producción:** sector_breadth.py (C31) implementa el contrato deseado en miniatura. Es replicable.
4. **Los dos casos de fallo están medidos:** darkpool.py (lectura por disco) y mte.py (estado con histéresis). Ambos tienen diseño claro de migración.
5. **La frontera entre df_market (562 mixtos) y df_stocks (313 equities) está delimitada.** No hay confusión.

### 6.2. Respuesta: no, la evidencia no es suficiente para la activación

**Razones:**

1. **El universo no-equity no tiene contrato:** 23 tickers sin semántica temporal definida (INDEX_EOD, RATE_YIELD, FUTURE_SETTLEMENT, FX_DAILY_CUT).
2. **El orden productor → consumidor no está definido para estados persistentes:** mte.py genera estado que persiste entre runs. Un cambio mal secuenciado produce escenarios incoherentes durante varios runs.
3. **La distinción df_market.index[-1] vs expected_session vs reference_date.date() no está unificada.**
4. **darkpool.py sigue leyendo el parquet directamente.** La fuente canónica "df_market en memoria" no se cumple mientras exista esta excepción.

### 6.3. Separación explícita

| Dimensión | Estado |
|---|---|
| Arquitectura del contrato | Evidencia suficiente |
| Semántica operativa por clase | Pendiente de 3B/3C |
| Activación en producción | Bloqueada |

### 6.4. Consecuencia para el dictamen

El dictamen puede autorizar:
- **FU-021-5 diseño:** sí, con la evidencia actual.
- **FU-021-5 activación:** no, hasta cerrar 3B/3C y migrar darkpool.
---

## 7. Estado actual y siguiente paso

### 7.1. Estado de bloqueos

| Elemento | Estado | Condición de desbloqueo |
|---|---|---|
| A3.1 (retirar trim_to_last_valid_date) | BLOQUEADA | FU-021-5 cerrado + 3B/3C |
| FU-021-5 diseño | AUTORIZADO | — |
| FU-021-5 activación | BLOQUEADA | Diseño cerrado + 3B/3C |
| FU-021-3B (INDEX_EOD + RATE_YIELD) | BLOQUEADO | Verificación empírica Close Yahoo |
| FU-021-3C (FUTURE_SETTLEMENT + FX_DAILY_CUT) | BLOQUEADO | Provider dedicado |
| darkpool migración | pendiente | Decisión de arquitectura |

### 7.2. Opciones de siguiente paso

**Opción α — Diseñar FU-021-5.**
- Aprovechar la evidencia del Ciclo 2.
- Definir: contrato por clase, semántica temporal por universo, API del productor, contrato de writers.
- Sin tocar código. Solo dictamen + especificación.
- Deja A3.1 y darkpool a la espera.

**Opción β — Diseñar migración darkpool antes.**
- Eliminar la única excepción a la fuente canónica.
- Refactor de firma de compute_darkpool_signals + market_data.py.
- Requiere dictamen específico por ser cambio de API intra-pipeline.

**Opción γ — Ambas en paralelo.**
- Diseño FU-021-5 (documental) + diseño darkpool (técnico) simultáneos.
- Riesgo: si FU-021-5 decide que darkpool debe recibir df_market por contrato, el diseño de darkpool cambia.

### 7.3. Siguiente paso propuesto para dictamen

Valorar si procede iniciar el diseño formal de FU-021-5, incluyendo contrato de productor, semántica de writers, propagación temporal y tratamiento del estado persistente de MTE.

La decisión entre las opciones α, β o γ corresponde al dictamen. Este informe no anticipa la elección.

---

## 8. Preguntas al auditor

### Q-C2.1 — ¿Existe evidencia suficiente para diseñar FU-021-5?

Según §6.1, la evidencia permite especificar la arquitectura, las responsabilidades, las invariantes y los requisitos de validación. La pregunta es si el auditor comparte el diagnóstico de que esa base es suficiente para producir una especificación de arquitectura de FU-021-5, entendiendo que la semántica operativa por clase queda pendiente.

### Q-C2.2 — Secuencia darkpool vs FU-021-5

¿Recomienda el auditor:
- (α) Diseñar FU-021-5 primero y darkpool después como caso específico.
- (β) Diseñar darkpool primero por ser la única excepción técnica.
- (γ) Abordarlo dentro de FU-021-5 sin ciclo separado.

### Q-C2.3 — Contrato de writers

Los 9 patrones P0–P8 convergen en una pregunta que ningún documento responde:

> ¿Qué semántica exacta debe tener la fecha escrita en un CSV histórico?

- ¿Fecha de observación del dato (dataset)?
- ¿Fecha efectiva declarada (R1/R2 FU-020)?
- ¿Fecha esperada de sesión (expected_session)?
- ¿Una combinación con columnas separadas (date, effective_date, expected_session)?

El dictamen de cierre debe fijar la semántica antes de que FU-021-5 diseñe los writers.

### Q-C2.4 — Estado persistente de MTE (C43)

mte.py genera estado con histéresis que persiste entre runs. ¿Cómo debe convivir con el contrato temporal?

Opciones:
- (α) El estado se invalida cuando cambia el contrato temporal.
- (β) El estado incluye la fecha efectiva y se versiona.
- (γ) El estado se migra a outputs/state/mte_state.json con esquema y versionado.

Actualmente hay dos paths (STATE_FILE hardcoded, MTE_STATE_FILE desde config) sin coherencia garantizada (C48).

### Q-C2.5 — Cierre documental de A3.1

Con el Ciclo 2 cerrado y A3.1 bloqueada hasta FU-021-5:

- ¿Debe redactarse ahora un documento de especificación de FU-021-5 o esperar a que el auditor dicte primero el contrato temporal?
- ¿Debe el adendum FU-021-3A actualizarse con los hallazgos C21–C48?

### Q-C2.6 — Prioridades para el Ciclo 3

¿Qué debe priorizar el siguiente ciclo?
- (α) Diseño FU-021-5.
- (β) Diseño migración darkpool.
- (γ) Fixes de robustez menores (C23, C25, C37, C41, C46).
- (δ) Cierre documental adicional (adendum 2).

### Q-C2.7 — C7/C8 (huecos de nomenclatura)

Los identificadores C7 y C8 no están asignados en el registro. ¿Debe el auditor solicitar su uso para futuros hallazgos, o se saltan por convención?

### Q-C2.8 — Granularidad del contrato

¿FU-021-5 debe definir un único contrato temporal agregado para df_market, o debe declarar contratos independientes por subconjunto homogéneo de instrumentos y posteriormente definir una regla de consolidación para el DataFrame mixto de 562 tickers?

Justificación: los patrones P0–P8 y el caso USA+UK demuestran que un único expected_session global es insuficiente. El DataFrame mixto actual (562 tickers, 5 semánticas distintas) no puede resolverse con un solo contrato sin ambigüedad.

---

## 9. Anexos

### Anexo A — Veredictos completos (32 fichas)

| Ficha | Módulo | Veredicto |
|---|---|---|
| 1 | regimes/macro_regime.py | B |
| 2 | regimes/structural_engine.py | B |
| 3 | regimes/tactical_engine.py | B |
| 4 | src/pipeline/data_load.py | P (productor) |
| 5 | indicators/darkpool.py | C (ciclo 2) |
| 6 | src/pipeline/regimes.py | B |
| 7 | src/pipeline/leaders.py | B |
| 8 | src/pipeline/breadth_metrics.py | B |
| 9 | src/pipeline/engines.py | B |
| 10 | src/pipeline/mte_confirmation.py | B |
| 11 | src/pipeline/sectors_base.py | B |
| 12 | src/pipeline/flows_primary.py | B |
| 13 | src/pipeline/sector_metrics.py | B |
| 14 | src/pipeline/slpm.py | B |
| 15 | src/pipeline/diagnostics.py | B |
| 16 | src/pipeline/market_data.py | B |
| 17 | src/pipeline/indices_intl.py | B |
| 18 | indicators/credit.py | B |
| 19 | indicators/cross_asset.py | B |
| 20 | indicators/cross_asset_context.py | B |
| 21 | indicators/index_phase.py | B |
| 22 | indicators/commodity_market_correlation.py | B |
| 23 | indicators/rs_internal.py | B |
| 24 | indicators/sector_correlation.py | B |
| 25 | indicators/sector_leader_divergence.py | B |
| 26 | indicators/index_leaders.py | B |
| 27 | indicators/sector_breadth.py | B |
| 28 | indicators/vol_metrics.py | B |
| 29 | indicators/volatility_structure.py | B |
| 30 | indicators/stock_leader.py | B |
| 31 | indicators/slpm_v12.py | B |
| 32 | indicators/mte.py | B |

**Totales:** 30 B, 1 P, 1 C.

### Anexo B — Commits del Ciclo 2

| Commit | Descripción |
|---|---|
| 694b172 | Create FU-021-3A_ADENDUM.md |
| 33ba2fa | fix(C19): indices_intl propaga reference_date y run_id a download_stock_prices |
| d3790b3 | fix(C16): sector_metrics propaga effective_meta (FU-020) a writers de concentracion y representatividad |

### Anexo C — Registro completo de hallazgos C1–C48 + H-DP-1

Ver §4 del presente informe y el Adendum FU-021-3A_ADENDUM.md (694b172).

---

**Fin del informe de cierre del Ciclo 2.**
**HEAD de referencia:** d3790b3 (origin/main).
**Fecha:** 2026-09-15.
**Estado:** pendiente de dictamen externo.