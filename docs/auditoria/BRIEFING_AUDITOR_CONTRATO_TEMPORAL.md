# Briefing para auditoría externa
## Contrato temporal de `df_market` — Caso completo y preguntas

**Fecha:** 2026-09-15
**HEAD de referencia:** bd18d22 (origin/main)
**Autor:** Ingeniero Supervisor
**Solicita:** Dictamen arquitectónico sobre el contrato temporal de `df_market`
**Documentos técnicos relacionados:** Adendum FU-021-3A (`694b172`), Informe de cierre Ciclo 2 (`bd18d22`)
**Estado:** pendiente de dictamen

---

## 0. Cómo leer este documento

Este documento es **autocontenido**. No asume que el lector haya visto el repositorio, el prompt maestro del sistema ni informes previos.

Estructura:

- **§1–§2** — Resumen y contexto: qué es el sistema y qué problema tiene.
- **§3–§5** — Investigación: qué se descubrió, cómo y con qué metodología.
- **§6–§7** — Resultados: las 32 fichas auditadas, los 49 hallazgos, los 9 patrones detectados.
- **§8–§9** — Estado actual y opciones de siguiente paso.
- **§10** — Preguntas al auditor, expandidas con contexto.
- **§11** — Anexos: tablas de referencia.

Los hechos verificables se distinguen de las hipótesis. Cuando algo es hipótesis, se marca explícitamente.

---

## 1. Resumen ejecutivo

### 1.1. Qué se ha auditado

Un sistema determinista de análisis macro-sectorial ("Radar de Rotación Sectorial") construye un DataFrame de mercado llamado `df_market` que alimenta 30 módulos consumidores en cascada. Ese DataFrame tiene 562 instrumentos heterogéneos: 539 acciones y ETFs USA con semántica EOD, más 23 instrumentos no-equity (índices, futuros, FX, yields) con semánticas temporales distintas.

La auditoría se ha centrado en una pregunta concreta: **¿qué fecha cree tener `df_market` cuando llega a cada consumidor, y quién decide esa fecha?**

### 1.2. Qué se ha encontrado

- **La fecha de `df_market` no es un contrato explícito.** Es una propiedad implícita que cada módulo resuelve por su cuenta.
- **Se han identificado 9 patrones distintos (P0–P8) para decidir qué fecha escribe un writer histórico.** Ninguno declara cobertura. Solo uno (P6, `sector_breadth.py`) valida sesión bursátil.
- **20 writers históricos dependen de la fecha de `df_market` de formas heterogéneas.** Un cambio en la última fila del DataFrame produce cambios no lineales en CSV históricos y en el estado persistente del motor MTE.
- **Existe una segunda fuente de verdad:** `darkpool.py` lee `data/market_data.parquet` directamente desde disco, saltándose el objeto en memoria que ven los otros 29 consumidores.
- **El universo no-equity (23 de 562 instrumentos) no tiene contrato temporal definido.**

### 1.3. Qué se ha hecho

- **32 fichas de auditoría** con veredicto individual por módulo.
- **2 fixes ejecutados:** C19 (`33ba2fa`) e C16 (`d3790b3`).
- **1 verificación cerrada:** C20 (parámetro vestigial, sin cambio).
- **49 hallazgos registrados** clasificados por acción requerida.
- **Tests:** 362 → 369 passed + 2 skipped.
- **Gate 10/10, pyflakes 0, compileall OK.**

### 1.4. Qué se pregunta al auditor

Ocho preguntas concretas, enunciadas en §10. La principal:

> **¿Se puede diseñar el contrato temporal global de `df_market` con la evidencia actual, aunque todavía no se pueda activar?**

Respuesta provisional del Ingeniero Supervisor (§6): **diseño sí, activación no**. Falta cerrar las clases no-equity (FU-021-3B/3C) y resolver la excepción de `darkpool`.

---

## 2. Contexto del sistema

### 2.1. Qué es el Radar

Sistema determinista, descriptivo y auditable de análisis macro-sectorial. Sin ML predictivo, sin optimización de parámetros, sin automatización de trading. Todos los outputs son diagnósticos, no recomendaciones.

- **Entorno:** Windows, PowerShell, Python.
- **Ejecución:** cron diario en GitHub Actions.
- **Cobertura:** 313 tickers de equity (262 Yahoo + 13 Euronext + 19 Xetra + 19 BME).
- **Universo `df_market`:** 562 instrumentos mixtos (equity + índices + futuros + FX + yields).
- **Flujo principal:**
  ```
  OHLCV → Regímenes → Motores → Indicadores → Scores → Reporte Markdown
  ```
- **Validación:** Gate de 10 comprobaciones + 369 tests locales.

### 2.2. El DataFrame central

`df_market` es un DataFrame multi-índice (columna × ticker → serie temporal). Se construye una vez al día:

1. `download_market_data(reference_date, run_id)` descarga 562 tickers.
2. Se aplica un filtro EOD (FU-021-3A) sobre los 539 equity.
3. Se escribe `data/market_data.parquet`.
4. Se aplica `trim_to_last_valid_date(df_market)` en memoria (con `min_coverage=0.5`).
5. Se pasa `df_market` a los consumidores.

**El trim del paso 4 es el objeto de la disputa.** Es la última autoridad temporal genérica del pipeline. Su retirada es A3.1.

### 2.3. Arquitectura de consumidores

33 ficheros mencionan `df_market`, clasificados en 4 capas:

| Capa | Descripción | Cuenta |
|---|---|---|
| 0 | Productor (`data_load.py`) | 1 |
| 1 | Consumidores directos (`src/pipeline/*`) | 12 |
| 2a | Régimen (`regimes/*`) | 3 |
| 2b | Indicadores (`indicators/*`) | 15 |
| — | Consumidor de artefacto (`darkpool.py`) | 1 |
| — | Orquestador (`run.py`) | 1 |
| **Total** | | **33** |

### 2.4. Reglas de FU-020 ya establecidas

Antes de este caso, el sistema ya tenía 4 reglas aprobadas por auditor sobre la resolución temporal:

- **R1:** Una métrica agregada nunca se publica sin declarar la fecha efectiva y la cobertura.
- **R2:** La fecha efectiva se resuelve con `resolve_effective_date()`, no por posición física.
- **R3:** La resolución temporal debe compartirse entre métricas derivadas del mismo universo.
- **R4:** Filtro de sesión (FU-018) y resolución por cobertura (FU-020) son controles independientes.

Estas reglas se aplican hoy solo a `df_stocks` (313 equity), no a `df_market` (562 mixtos).

---

## 3. El problema

### 3.1. Formulación inicial

El informe de la fase FU-021-3A (2026-09-15) declaró:

> *"El `trim_to_last_valid_date(..., 0.5)` es incorrecto como autoridad temporal, pero al retirarlo sin sustituirlo por una arquitectura completa se elimina la última barrera. Diferido hasta sub-informe de los **13 consumidores de `df_market`**."*

La tarea asignada al sub-informe era: auditar los 13 consumidores antes de retirar el trim.

### 3.2. El "13" no resiste verificación

Se lanzaron comandos de conteo contra el repositorio. Resultados:

| Conteo | Valor real |
|---|---|
| Ficheros que mencionan `df_market` (todo el repo) | **33** |
| Ficheros de `src/pipeline/` que mencionan `df_market` | **13** (12 consumidores + 1 productor) |
| Consumidores reales por firma (todo el repo) | **30** (12 + 3 + 15) |
| Consumidor por disco (no firma) | **1** (`darkpool.py`) |

**El "13" era un conteo correcto de un universo mal etiquetado.** No eran 13 consumidores, eran 13 ficheros de `src/pipeline/` que mencionan `df_market`, de los cuales uno es productor. Los 20 ficheros restantes de `regimes/` e `indicators/` no estaban contemplados en ninguna versión del sub-informe.

### 3.3. Implicación

Si A3.1 se hubiera ejecutado con alcance "13 consumidores", se habría retirado el trim afectando a 33 ficheros de los cuales **20 nunca fueron auditados**. Es exactamente el escenario "arreglar 1, romper 3".

### 3.4. Ampliación del alcance

Se elevó a auditoría externa. El auditor aprobó ampliar el alcance a:

- Ciclo 1: 12 Capa 1 + 3 Capa 2a + productor + frontera (`darkpool`) = 17 fichas.
- Ciclo 2: 15 Capa 2b = 15 fichas.

Total: **32 fichas**.

---

## 4. Cronología de la investigación

| Fecha | Hito |
|---|---|
| 2026-09-15 | FU-021-3A (filtro EOD equity) cerrado. Surge la pregunta sobre `trim_to_last_valid_date`. |
| 2026-09-15 | Subinforme elevado a auditoría con alcance ampliado. |
| 2026-09-15 | **Ciclo 1** ejecutado: 17 fichas. 14 B, 1 C (H-DP-1), 1 P (productor). |
| 2026-09-15 | **Dictamen Ciclo 1:** A3.1 bloqueada hasta FU-021-5. Ciclo 2 autorizado. Adendum autorizado. |
| 2026-09-15 | **Adendum FU-021-3A** publicado (`694b172`): corrige el "13" a 33 ficheros con desglose por capas. |
| 2026-09-15 | **Ciclo 2** ejecutado: 15 fichas de Capa 2b. Todas B. |
| 2026-09-15 | Fix **C19** pusheado (`33ba2fa`): `indices_intl` propaga `reference_date`. |
| 2026-09-15 | Fix **C16** pusheado (`d3790b3`): `sector_metrics` propaga `effective_meta`. |
| 2026-09-15 | **Informe de cierre del Ciclo 2** publicado (`bd18d22`). |
| 2026-09-15 | **Este briefing** — solicitud de dictamen arquitectónico. |

---

## 5. Metodología

### 5.1. Fichas de auditoría

Por cada uno de los 32 ficheros del universo auditado se produjo una ficha con:

- **4 campos obligatorios:** `dependencia_temporal` (explícita/implícita/ninguna), `fuente` (memoria/parquet), `fecha_usada` (`index[-1]` / explícita / otra), `sensibilidad_a_A3.1` (sí/no).
- **6 ejes de análisis (H1–H6):**
  - H1: ¿usa última fila?
  - H2: ¿calcula ventanas desde `index[-1]`?
  - H3: ¿asume EOD?
  - H4: ¿asume fecha común entre instrumentos?
  - H5: ¿lógica distinta por ticker?
  - H6: ¿temporalidad implícita?
- **Veredicto:** A / B / C / D.
- **Justificación:** cita literal con `archivo:línea`.

### 5.2. Semántica de veredictos

- **A — COMPATIBLE:** no depende de la semántica temporal que A3.1 altera.
- **B — COMPATIBLE CONDICIONADO:** funciona si el productor cumple el nuevo contrato.
- **C — REQUIERE CAMBIO:** depende del trim o de una fecha implícita.
- **D — BLOQUEANTE:** no puede convivir con el nuevo contrato.
- **P — PRODUCTOR:** es el locus de la acción (no consumidor).

**Punto clave:** **`B` no significa "listo para retirar el trim".** Significa "este consumidor no necesita modificación propia si el productor entrega un DataFrame ya temporalmente resuelto".

### 5.3. Regla de parada

- `C/D` nuevo no cubierto por dictamen previo → detener y elevar.
- `C/D` ya contemplado → registrar y continuar.

En los 32 fichas no apareció ningún `C/D` nuevo. La regla no se activó en Ciclo 2.

---

## 6. Resultados

### 6.1. Veredictos

| Veredicto | Cuenta |
|---|---|
| A | 0 |
| B | 30 |
| C | 1 (`darkpool.py`, H-DP-1) |
| D | 0 |
| P (productor) | 1 |
| **Total** | **32** |

Ningún fichero requiere cambio estructural para convivir con A3.1. El único `C` es `darkpool.py`, ya conocido y asignado a decisión de arquitectura.

### 6.2. Hallazgos

**49 hallazgos** clasificados en 6 categorías:

| Categoría | Cuenta | Ejemplos |
|---|---|---|
| Bloquean cierre de A3.1 | 3 | C11, C14, H-DP-1 |
| Condicionan diseño FU-021-5 | 13 | C22, C27, C31, C35, C38, C43, C48 |
| Preexistentes documentales | 18 | C1, C2, C5, C10... |
| Deudas de robustez | 13 | C3, C4, C6, C9, C23... |
| Ya resueltos | 3 | C16, C19, C20 |
| Positivos / reutilizables | 4 | C31 (modelo), patrón C16/C19, helper |
| **Total** | **49 + H-DP-1 = 50** | |

### 6.3. Los 9 patrones de writer (P0–P8)

**El hallazgo central de la auditoría.** Nueve formas distintas de decidir qué fecha se escribe en un CSV histórico:

| Patrón | Módulo | Fecha escrita | Valida sesión |
|---|---|---|---|
| **P0** | `stock_leader.py` | **Ninguna** (sin columna `date`) | — |
| **P1** | `engines.py` | `_observation_date_from_df(df_market)` | No |
| **P2** | `sectors_base.py` (7 writers) | `df_market.index[-1]` | No |
| **P3** | `cross_asset_context.py`, `sector_correlation.py` | `_observation_date_from_df(returns_df)` | No |
| **P4** | `rs_internal.py` | `_observation_date_from_df(serie_intersectada)` | No |
| **P5** | `sector_leader_divergence.py` | `_observation_date_from_df(df_stocks)` | No |
| **P6** | `sector_breadth.py` | `as_of_date` (caller, validado con `is_market_day`) | **Sí** |
| **P7** | `volatility_structure.py` | `_observation_date_from_df(vix)` | No |
| **P8** | `mte.py` | Estado JSON sin fecha (persistente con histéresis) | No |

**Diagnóstico:** no hay contrato. Cada módulo decide por su cuenta. R1 (declaración de cobertura) se cumple en 0 writers. R3 (resolución compartida) se cumple en 1 de 20 casos.

### 6.4. Caso MTE — estado persistente

`mte.py` (1964 LOC, motor de transición macro) escribe **dos ficheros de estado JSON** con histéresis: un cambio en la última fila de `df_market` puede no producir un cambio de escenario de forma lineal, sino desplazarlo uno o dos runs, o bloquearlo. Es el único módulo del pipeline con estado que persiste entre runs.

Además usa dos paths distintos (`STATE_FILE` hardcoded en `outputs/state/mte_state.json`, `MTE_STATE_FILE` desde config) sin garantía de coherencia.

### 6.5. Caso `darkpool.py` — segunda fuente de verdad

`darkpool.py` lee directamente `data/market_data.parquet` de disco (L189). No recibe `df_market` por firma. Consecuencia: existen dos rutas paralelas a la misma información temporal:

- **Ruta A (memoria):** `df_market` en memoria → 30 consumidores.
- **Ruta B (disco):** `data/market_data.parquet` → `darkpool`.

Hoy los datos coinciden. Bajo A3.1 sin sustituto, ambas rutas se alinearían. Pero el contrato "df_market en memoria es la fuente canónica" no se cumple mientras un módulo lea el parquet directamente.

### 6.6. Fixes ejecutados

- **C19** (`33ba2fa`): `indices_intl.compute_indices_intl` ahora propaga `reference_date` y `run_id` a `download_stock_prices`. Antes se ejecutaba una segunda descarga sin fecha.
- **C16** (`d3790b3`): `sector_metrics.compute_sector_metrics` acepta `effective_meta` (de FU-020) y lo usa como `reference_date` para sus writers. Cumple R3 en 2 writers.

---

## 7. El hallazgo central

> **La fecha de `df_market` es actualmente una propiedad implícita del pipeline, no un contrato explícito.**

Evidencia acumulada:

1. **9 patrones distintos** para decidir la fecha que se escribe. Ni siquiera hay un helper unificador (`_observation_date_from_df` extrae la fecha pero no la normaliza ni la valida).
2. **20 writers históricos** afectados por `df_market` de formas heterogéneas.
3. **2 estados persistentes** (JSON de MTE) con histéresis.
4. **1 excepción a la fuente canónica** (`darkpool`).
5. **0 declaraciones de cobertura** en writers (R1 incumplida en el 100%).
6. **1 caso de cumplimiento correcto** (`sector_breadth.py`, patrón P6), replicable pero único.

**Implicación arquitectónica:** cualquier cambio en la última fila de `df_market` (por ejemplo retirar el trim sin sustituto, o aplicar FU-021-5 sin diseño previo) produce efectos heterogéneos en 20 puntos distintos del pipeline, sin señal detectable y sin declaración de cobertura.

**Conclusión provisional:** se puede especificar la arquitectura del contrato temporal de `df_market` (responsabilidades, invariantes, validación). No se puede activar en producción hasta cerrar las clases no-equity y migrar `darkpool`.

---

## 8. Estado actual del sistema

### 8.1. Estado verificado

| Métrica | Valor |
|---|---|
| HEAD | `bd18d22` = origin/main |
| Tests locales | 369 passed + 2 skipped |
| Gate | 10/10 |
| pyflakes | 0 warnings |
| compileall | OK |
| Cobertura de datos | 313/313 equity |
| Universo `df_market` | 562 (539 equity EOD + 23 no-equity sin contrato) |

### 8.2. Estado de bloqueos

| Elemento | Estado | Condición de desbloqueo |
|---|---|---|
| A3.1 (retirar `trim_to_last_valid_date`) | 🔴 BLOQUEADA | FU-021-5 cerrado + contratos 3B/3C |
| FU-021-5 diseño | 🟢 Puede avanzar | — |
| FU-021-5 activación | 🔴 BLOQUEADA | Diseño cerrado + 3B/3C |
| FU-021-3B (INDEX_EOD + RATE_YIELD) | 🔴 BLOQUEADO | Verificación empírica Close Yahoo |
| FU-021-3C (FUTURE_SETTLEMENT + FX_DAILY_CUT) | 🔴 BLOQUEADO | Provider dedicado |
| `darkpool` migración | ⏸ Pendiente | Decisión de arquitectura |

### 8.3. Lo que NO se ha hecho

- No se ha retirado el trim.
- No se ha modificado el informe original FU-021-3A.
- No se ha migrado `darkpool`.
- No se ha activado ningún contrato por clase.
- No se han tocado históricos.

---

## 9. Opciones de siguiente paso

### 9.1. Opción α — Diseñar FU-021-5

Producir un documento de especificación arquitectónica con:
- Contrato del productor (qué garantiza `df_market` al entregarse).
- Semántica del contrato por clase (EQUITY_EOD, INDEX_EOD, RATE_YIELD, FUTURE_SETTLEMENT, FX_DAILY_CUT).
- Contrato de writers (qué fecha escriben, con qué cobertura).
- Propagación temporal (cómo viaja `effective_meta` de productor a writers).
- Tratamiento del estado persistente de MTE.

**Coste:** 1-2 sesiones documentales. **Sin tocar código.**

### 9.2. Opción β — Diseñar migración `darkpool` primero

Refactor de `compute_darkpool_signals` para que reciba `df_market` por parámetro. Requiere cambio en `market_data.py`.

**Coste:** 1 sesión. **Riesgo:** si FU-021-5 decide que `darkpool` debe recibir `df_market` por contrato, el diseño se rehace.

### 9.3. Opción γ — Ambas en paralelo

**Ventaja:** tiempo.
**Riesgo:** duplicación de diseño.

### 9.4. Opción δ — Cerrar primero pendientes documentales

Redactar adendum 2 del FU-021-3A con los 28 hallazgos nuevos (C21–C48). Cerrar huecos de nomenclatura (C7/C8).

**Coste:** 1 sesión. **Valor:** documental, no desbloquea.

---

## 10. Preguntas al auditor

Las 8 preguntas, expandidas con el contexto necesario para responderlas sin consultar otros documentos.

### Q-C2.1 — ¿Existe evidencia suficiente para diseñar FU-021-5?

**Contexto:** La auditoría ha producido 32 fichas con veredicto, 49 hallazgos clasificados, 9 patrones P0–P8, 20 writers históricos identificados, 2 casos de fallo estructural (`darkpool`, `mte`).

**Pregunta:** ¿Basta esa evidencia para producir un documento cerrado de arquitectura de FU-021-5, entendiendo que la semántica operativa por clase queda pendiente (3B/3C)?

**Respuesta provisional:** sí para arquitectura, no para semántica por clase.

### Q-C2.2 — Secuencia `darkpool` vs FU-021-5

**Contexto:** `darkpool.py` es la única excepción a la fuente canónica (`df_market` en memoria). Lee el parquet directamente.

**Pregunta:** ¿Se diseña/migra `darkpool` antes que FU-021-5 (opción β), después como caso específico (opción α), o integrado dentro de FU-021-5 sin ciclo separado (opción γ)?

### Q-C2.3 — Semántica del contrato de writers

**Contexto:** 9 patrones distintos (P0–P8) para escribir la fecha en un CSV histórico. Ninguno declara cobertura. Solo uno valida sesión.

**Pregunta:** ¿Qué semántica exacta debe tener la fecha escrita en un CSV histórico? Opciones:

- (a) Fecha de observación del dato (dataset).
- (b) Fecha efectiva declarada (R1/R2 FU-020).
- (c) Fecha esperada de sesión (`expected_session`).
- (d) Combinación con columnas separadas (`date`, `effective_date`, `expected_session`).

Esta decisión condiciona el diseño de los writers de FU-021-5.

### Q-C2.4 — Estado persistente de MTE

**Contexto:** `mte.py` genera estado JSON con histéresis que persiste entre runs. Un cambio en la última fila de `df_market` puede desplazar transiciones de escenario o bloquearlas. Hay dos paths (`STATE_FILE`, `MTE_STATE_FILE`) sin coherencia garantizada.

**Pregunta:** ¿Cómo debe convivir el estado con el nuevo contrato temporal?

- (a) El estado se invalida cuando cambia el contrato temporal.
- (b) El estado incluye la fecha efectiva y se versiona.
- (c) El estado se migra a un esquema con versionado y path único.

### Q-C2.5 — Cierre documental de A3.1

**Contexto:** A3.1 está bloqueada hasta FU-021-5. El informe de cierre del Ciclo 2 documenta los 28 hallazgos nuevos.

**Pregunta:** ¿Debe redactarse ahora un documento de especificación de FU-021-5, o esperar a que el auditor fije primero el contrato temporal? ¿Debe el adendum FU-021-3A actualizarse con los hallazgos C21–C48?

### Q-C2.6 — Prioridades para el Ciclo 3

**Pregunta:** ¿Qué debe priorizar el siguiente ciclo?

- (α) Diseño FU-021-5.
- (β) Diseño migración `darkpool`.
- (γ) Fixes de robustez menores (C23, C25, C37, C41, C46).
- (δ) Cierre documental adicional (adendum 2).

### Q-C2.7 — Huecos de nomenclatura C7/C8

**Contexto:** Los identificadores C7 y C8 no están asignados en el registro de hallazgos. El resto sigue secuencia C1–C48.

**Pregunta:** ¿Debe el auditor solicitar su uso para futuros hallazgos, o se saltan por convención?

### Q-C2.8 — Granularidad del contrato

**Contexto:** El DataFrame `df_market` tiene 562 instrumentos con 5 semánticas temporales distintas:

| Universo | Cuenta | Contrato |
|---|---|---|
| EQUITY_EOD (acciones + ETFs USA) | 539 | ✅ FU-021-3A |
| INDEX_EOD | ~13 | ❌ Pendiente 3B |
| RATE_YIELD | ~2 | ❌ Pendiente 3B |
| FUTURE_SETTLEMENT | ~5 | ❌ Pendiente 3C |
| FX_DAILY_CUT | ~3 | ❌ Pendiente 3C |

**Pregunta:** ¿FU-021-5 debe definir un único contrato temporal agregado para `df_market`, o debe declarar contratos independientes por subconjunto homogéneo de instrumentos y posteriormente definir una regla de consolidación para el DataFrame mixto?

**Justificación:** los 9 patrones P0–P8 y el caso USA+UK demuestran que un único `expected_session` global es insuficiente para un DataFrame con 5 semánticas distintas.

---

## 11. Anexos

### Anexo A — Veredictos completos (32 fichas)

[Tabla completa con las 32 fichas y sus veredictos]

### Anexo B — Registro de hallazgos C1–C48 + H-DP-1

[Remisión al informe de cierre del Ciclo 2, §4]

### Anexo C — Commits de la investigación

| Commit | Descripción |
|---|---|
| `7c44b4a` | Punto de entrada del ciclo |
| `694b172` | Adendum FU-021-3A |
| `33ba2fa` | Fix C19 |
| `d3790b3` | Fix C16 |
| `bd18d22` | Informe de cierre Ciclo 2 |

### Anexo D — Documentos del expediente

- Prompt maestro v6.14
- Subinforme Ciclo 1 (consumidores)
- Dictamen Ciclo 1
- Adendum FU-021-3A (`694b172`)
- Informe de cierre Ciclo 2 (`bd18d22`)

---

**Fin del briefing para auditoría externa.**
**HEAD de referencia:** `bd18d22` (origin/main).
**Fecha:** 2026-09-15.
**Estado:** pendiente de dictamen.
