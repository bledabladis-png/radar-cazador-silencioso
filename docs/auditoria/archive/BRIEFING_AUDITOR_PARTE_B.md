# Briefing para auditoría externa — Parte B de FU-021-5

## Contratos por clase del `df_market` y preguntas de ratificación

**Fecha:** 2026-09-16
**HEAD de referencia:** 468fc08 (origin/main)
**Documento principal:** FU-021-5_ESPECIFICACION_CONTRATO_TEMPORAL_PARTE_B.md (468fc08)
**Documentos base:** FU-021-5 Parte A (61aa99f / e54bab8), FU-021-3B (b645de4), FU-021-3B-bis (e54bab8), FU-021-3C (794f0e9)
**Estado:** pendiente de dictamen

---

## 0. Cómo leer este documento

Este briefing es **autocontenido**. No asume que el lector haya seguido los ciclos previos.

Estructura:

- **§1–§2** — Resumen ejecutivo y contexto mínimo del sistema.
- **§3–§4** — Cronología de la investigación y base empírica (3B, 3B-bis, 3C).
- **§5** — Qué propone Parte B.
- **§6** — Estado del bloqueo de A3.1.
- **§7–§8** — Las 8 preguntas al auditor, con contexto expandido, opciones concretas y consecuencias de cada una.

Las 8 preguntas no son un cuestionario técnico: son decisiones de arquitectura que condicionan la implementación posterior de FU-021-5 y el desbloqueo de A3.1.

---

## 1. Resumen ejecutivo

### 1.1. Dónde estamos

El sistema tiene un DataFrame central, `df_market`, que alimenta 30 consumidores. Ese DataFrame contiene 562 instrumentos heterogéneos (equity, índices, futuros, FX, yields, volatility), y **no tiene contrato temporal explícito**. Cada consumidor resuelve la fecha por su cuenta.

Tras Ciclo 1, Ciclo 2, Adendums 1 y 2, dictamen del contrato temporal y Parte A de FU-021-5, la arquitectura contractual está especificada pero **no ratificada clase a clase**.

### 1.2. Qué se ha hecho ahora

Se han ejecutado 3 investigaciones empíricas:

- **FU-021-3B**: midió INDEX_EOD y RATE_YIELD.
- **FU-021-3B-bis**: midió VOLATILITY_INDEX (clase descubierta al vuelo).
- **FU-021-3C**: midió FUTURE_SETTLEMENT y FX_DAILY_CUT.

Con esos datos, se ha redactado la Parte B de FU-021-5, que fija la **semántica operativa de las 6 clases**.

### 1.3. Qué se pide al auditor

Ratificar 8 decisiones, la más importante de las cuales es:

> **¿FUTURE_SETTLEMENT debe quedar como clase `BLOCKED` (no auditable con el ticker continuo de Yahoo), o debe migrarse a contratos explícitos / provider dedicado?**

Esa decisión afecta al 30-40% del pipeline.

### 1.4. Qué sigue bloqueado

- **A3.1** (retirar `trim_to_last_valid_date`): bloqueada hasta completar 4 condiciones de implementación.
- **FU-021-5 activación**: bloqueada hasta ratificación de Parte B.
- **FUTURE_SETTLEMENT**: bloqueada por causa empírica, no burocrática.

---

## 2. Contexto mínimo del sistema

### 2.1. El DataFrame central

`df_market` (562 instrumentos) se construye una vez al día:

1. Descarga por lotes desde Yahoo.
2. Se aplica filtro EOD a los 539 equity (FU-021-3A).
3. Se escribe `data/market_data.parquet`.
4. Se aplica un trim genérico (0.5 de cobertura mínima) en memoria.
5. Se pasa a 30 consumidores.

**El trim del paso 4 (A3.1) es el objeto de disputa.**

### 2.2. Las 6 clases identificadas empíricamente

| Clase | Cuenta | Naturaleza |
|---|---|---|
| EQUITY_EOD | 539 | Acciones y ETFs USA |
| INDEX_EOD_USA | 5 | ^GSPC, ^DJI, ^NDX, ^RUT, ^SPGSCI |
| INDEX_EOD_EUROPA | 4 | ^FTSE, ^GDAXI, ^IBEX, ^STOXX50E |
| INDEX_EOD_COMMODITY | 1 | DX-Y.NYB |
| VOLATILITY_INDEX | 3 | ^VIX, ^VIX3M, ^VXN |
| RATE_YIELD | 2 | ^FVX, ^TNX |
| FUTURE_SETTLEMENT | 5 | BZ=F, CL=F, GC=F, HG=F, NG=F |
| FX_DAILY_CUT | 3 | EURUSD=X, USDCNY=X, USDJPY=X |

### 2.3. Por qué importa

Cada clase tiene un calendario de publicación distinto. La fila más reciente de `df_market` puede contener solo unos pocos tickers. Ejemplo real del 2026-09-15: **2 de 562 tickers** (EURUSD y USDJPY). El resto son NaN.

Cualquier consumidor que use `.iloc[-1]` está operando sobre NaN en el 99.64% de las columnas.

---

## 3. Cronología de la investigación

| Fecha | Hito |
|---|---|
| 2026-09-15 | Ciclo 1 cerrado. Adendum 1. |
| 2026-09-15 | Ciclo 2 cerrado. Adendum 2. |
| 2026-09-15 | Dictamen del contrato temporal. 8 preguntas resueltas. |
| 2026-09-15 | Parte A publicada (61aa99f). |
| 2026-09-16 | FU-021-3B cerrado (b645de4): INDEX_EOD + RATE_YIELD. |
| 2026-09-16 | FU-021-3C cerrado (794f0e9): FUTURE + FX. |
| 2026-09-16 | Anexo VOLATILITY_INDEX + Parte A corregida (e54bab8). |
| 2026-09-16 | Parte B publicada (468fc08). |
| 2026-09-16 | Este briefing. |

---

## 4. Base empírica

Tres informes empíricos alimentan Parte B. Resumen de los hallazgos relevantes para el dictamen.

### 4.1. FU-021-3B — INDEX_EOD y RATE_YIELD

**Método:** comparación Yahoo individual vs parquet para 12 tickers.

**Resultado principal:** los valores Close en el día común coinciden exactamente (diff ≤ 0.003). El pipeline no tiene bug de datos.

**Hallazgo crítico:** existe una divergencia de fechas entre parquet y Yahoo, pero es por **timing de publicación + ventana de cache**, no por comportamiento estructural del pipeline.

| Grupo | Lag observado |
|---|---|
| USA (índices) | 1 día |
| Europa (índices) | 3-4 días (artefacto cache) |
| Yields | 1 día |
| DXY | 1 día |

**Causa raíz europea:** el parquet se escribe el 15/09 a las 02:21 UTC. En ese momento, Yahoo tenía los índices EU hasta el 11/09 (viernes). Los cierres EU del lunes 14 se publican a las ~15:30 UTC. El pipeline no se re-ejecuta hasta 23h después por `CACHE_HOURS = 23`.

**Asimetría adicional:** ^FTSE publica 1 día antes que ^GDAXI, ^IBEX, ^STOXX50E.

### 4.2. FU-021-3B-bis — VOLATILITY_INDEX

**Resultado:** ^VIX, ^VIX3M, ^VXN se comportan idénticos a INDEX_EOD_USA.

- Lag 1 día.
- Valores Close idénticos en día común.
- Descarga en batch correcta.

**Anomalía menor:** ^VIX3M con descarga individual `yfinance` devuelve 1 sola fila. La descarga en batch (la que usa el pipeline) funciona. No es problema del pipeline.

**Conclusión:** VOLATILITY_INDEX hereda de INDEX_EOD_USA sin contrato propio.

### 4.3. FU-021-3C — FUTURE y FX

**Hallazgo principal (FUTURE):** los históricos del parquet para futuros **no son reproducibles**.

| Ticker | Parquet 14/09 | Yahoo 14/09 (hoy) | Diff rel |
|---|---|---|---|
| BZ=F | 106.49 | 105.68 | 0.76% |
| CL=F | 102.10 | 101.39 | 0.70% |
| GC=F | 4327.50 | 4351.90 | 0.56% |
| HG=F | 6.389 | 6.330 | 0.92% |
| NG=F | 2.880 | 2.896 | 0.56% |

**Comparación:** en índices USA la diff era 0.002-0.003. Aquí es 100-1000× mayor.

**Hipótesis investigadas y descartadas:**
- Rollover retroactivo puro.
- Ajuste por rango temporal.
- Ajuste por `auto_adjust`.
- Snapshot intradía (sesión asiática).

**Hipótesis residual:** Yahoo reescribió su histórico entre las dos capturas. **No verificable.**

**Conclusión:** `CL=F`, `GC=F`, etc. son tickers continuos que no identifican un contrato específico. Yahoo los ajusta de forma no determinista. **La clase no es auditable con Yahoo continuo.**

**Hallazgo secundario (FX):** los valores FX son estables, con lag heterogéneo intra-clase:

| Par | Parquet last | Yahoo last | Lag |
|---|---|---|---|
| EURUSD | 2026-09-15 | 2026-09-15 | 0d |
| USDCNY | 2026-09-14 | 2026-09-15 | 1d |
| USDJPY | 2026-09-15 | 2026-09-15 | 0d |

**Hallazgo adicional:** la fila 2026-09-15 de `df_market` contiene solo 2 de 562 tickers (0.36%). El resto son NaN.
---

## 5. Qué propone Parte B

### 5.1. Contratos por clase

Para cada clase, Parte B define:

- Miembros concretos.
- Calendario bursátil.
- Lag máximo tolerado (`max_lag_days`).
- Cobertura mínima.
- Estado esperado.
- Evidencia empírica que lo respalda.

### 5.2. Tabla resumen

| Clase | Lag máx | Cobertura mín | Estado | Activable |
|---|---|---|---|---|
| EQUITY_EOD | 0 | 0.90 | OK | ✅ (ya activa) |
| INDEX_EOD_USA | 1 | 1.0 | OK / STALE | ✅ Sí |
| INDEX_EOD_EUROPA | 5 | 1.0 | OK / STALE | ✅ Sí |
| INDEX_EOD_COMMODITY | 1 | 1.0 | OK / STALE | ✅ Sí |
| VOLATILITY_INDEX | 1 | 1.0 | OK / STALE | ✅ Sí |
| RATE_YIELD | 1 | 1.0 | OK / STALE | ✅ Sí |
| FUTURE_SETTLEMENT | N/A | N/A | **BLOCKED** | 🔴 No |
| FX_DAILY_CUT | 0-1 por par | 0.5 | OK / STALE | ✅ Sí |

### 5.3. Cambios propuestos a la Parte A

Parte B propone 4 cambios adicionales a la Parte A publicada:

1. **`status STALE`** como nuevo estado (lag positivo dentro de tolerancia). Parte A solo definía `OK | INSUFFICIENT | BLOCKED | PENDING`.
2. **`per_pair_max_lag`** para FX (cada par publica a su ritmo).
3. **`per_ticker_lag`** para INDEX_EOD_EUROPA (^FTSE publica antes).
4. **`activation_requirement`** para FUTURE_SETTLEMENT (documentar condiciones de desbloqueo).

### 5.4. Lo que Parte B NO hace

- No activa contratos.
- No implementa cambios.
- No retira `trim_to_last_valid_date`.
- No decide el provider alternativo para FUTURE.
- No decide el cutoff exacto de FX.

---

## 6. Estado del bloqueo de A3.1

El dictamen del contrato temporal fijó 6 condiciones para desbloquear A3.1:

| Condición | Estado |
|---|---|
| 1. Contrato de cada clase relevante | ✅ **Cerrado por Parte B** |
| 2. Regla de consolidación de `df_market` | ✅ **Definida en Parte A §A.2** |
| 3. Propagación de effective metadata | 🔴 Pendiente implementación |
| 4. Writers adaptados (20 identificados) | 🔴 Pendiente implementación |
| 5. Estado MTE compatible | 🔴 Pendiente diseño (Parte A §A.6) |
| 6. `darkpool` fuera de fuente paralela | 🔴 Pendiente migración (Parte A §A.7) |

**2 de 6 cerradas documentalmente. Las 4 restantes son trabajo de implementación.**

---

## 7. Las 8 preguntas al auditor

Cada pregunta incluye contexto, opciones concretas y consecuencias de cada una.

### Q-B.1 — Aceptación del `status STALE`

**Contexto:** Parte A definía 4 estados para el contrato de clase: `OK | INSUFFICIENT | BLOCKED | PENDING`. Los informes empíricos muestran que existe un quinto caso real: contratos donde el lag positivo está dentro de tolerancia. Ejemplo: `INDEX_EOD_USA` con lag=1 día es `OK`; con lag=2 días pero cobertura 100% no es `INSUFFICIENT` (la cobertura es correcta), es `STALE`.

**Pregunta:** ¿Aprueba el auditor un nuevo estado `STALE` para representar lag positivo dentro de tolerancia?

**Opciones:**

- (α) Aprobar `STALE` como nuevo estado.
- (β) Reclasificar `STALE` como caso de `OK` con metadato `lag_days > 0`.
- (γ) Reclasificar `STALE` como `INSUFFICIENT` (no distinguir).

**Consecuencia:**
- (α) Contratos más expresivos, lógica de consumidor ligeramente más compleja.
- (β) Menos estados, pero el consumidor debe leer `lag_days` para saber si la fecha es actual.
- (γ) Simplifica el modelo, pero pierde información sobre cobertura vs frescura.

**Recomendación Parte B:** opción (α).

---

### Q-B.2 — `per_pair_max_lag` para FX

**Contexto:** Los 3 pares FX del universo publican con lag heterogéneo. EURUSD y USDJPY están al día; USDCNY va 1 día atrás. Un único `max_lag_days` no captura esta realidad.

**Pregunta:** ¿Aprueba `per_pair_max_lag` como parámetro del contrato FX_DAILY_CUT?

**Opciones:**

- (α) `per_pair_max_lag` como dict explícito por par.
- (β) Un único `max_lag_days = 1` para toda la clase, aceptando que todos los pares se miden igual.
- (γ) Subdividir FX en subclases por par (FX_MAJOR_EURUSD, FX_CNH, ...).

**Consecuencia:**
- (α) Modelo realista, permite declarar tolerancias asimétricas.
- (β) Más simple, pero trata iguales a pares que no lo son.
- (γ) Explosión de subclases, poco práctico con 3 pares.

**Recomendación Parte B:** opción (α).

---

### Q-B.3 — `per_ticker_lag` para INDEX_EOD_EUROPA

**Contexto:** ^FTSE publica 1 día antes que ^GDAXI, ^IBEX, ^STOXX50E. Es una asimetría del mismo mercado (Europa) que no puede explicarse por calendario (todos cierran 16:30-17:30 CET).

**Pregunta:** ¿Aprueba `per_ticker_lag` para declarar esta asimetría?

**Opciones:**

- (α) `per_ticker_lag` como dict explícito por ticker.
- (β) Un único `max_lag_days = 5` que cubra ambos casos.
- (γ) Subdividir INDEX_EOD_EUROPA en dos subclases.

**Consecuencia:**
- (α) Modelo granular, permite auditar por ticker.
- (β) Trata uniformemente, pero pierde la capacidad de detectar cuándo ^FTSE va anormalmente retrasado.
- (γ) Duplica subclases por razones de publicación, no de mercado.

**Recomendación Parte B:** opción (α).

---

### Q-B.4 — FUTURE_SETTLEMENT como `BLOCKED` definitivo

**Contexto:** Los históricos de futuros del parquet no son reproducibles. `CL=F` o `GC=F` identifican "el front-month actual" y Yahoo reescribe sus históricos entre capturas. La investigación no logró causa raíz exacta, pero sí conclusión operativa: **no es auditable**.

**Pregunta:** ¿Confirma el auditor que FUTURE_SETTLEMENT debe quedar como `BLOCKED` hasta decisión explícita sobre alternativa?

**Opciones:**

- (α) `BLOCKED` con las 3 alternativas documentadas (contratos explícitos, provider dedicado, congelar permanentemente).
- (β) Migrar ya a contratos explícitos (`GCZ26.CMX`) con lógica de rollover propia.
- (γ) Congelar `BLOCKED` permanentemente y aceptar impacto en 30-40% del pipeline.

**Consecuencia:**
- (α) Espera una decisión separada. Deja el pipeline en estado conocido.
- (β) Requiere desarrollo significativo (lógica de rollover, calendario de expiraciones). No resuelve sin verificación empírica previa.
- (γ) Acepta degradación silenciosa en 5 consumidores.

**Recomendación Parte B:** opción (α).

---

### Q-B.5 — Modificaciones a Parte A (§B.7.2)

**Contexto:** Parte A ya está publicada (61aa99f, corregida en e54bab8). Parte B propone 4 cambios adicionales: `status STALE`, `per_pair_max_lag`, `per_ticker_lag`, `activation_requirement`.

**Pregunta:** ¿Aprueba esas 4 modificaciones a la Parte A, o prefiere diferirlas?

**Opciones:**

- (α) Aprobar todas y publicar Parte A v2.
- (β) Aprobar solo algunas (indicar cuáles).
- (γ) Diferir a una Parte A v2 tras implementación.

**Consecuencia:**
- (α) Documento unificado, sin divergencias entre Parte A publicada y Parte B propuesta.
- (β) Requiere Parte A v2 parcial.
- (γ) Divergencia documental temporal. Parte B queda como propuesta no ratificada.

**Recomendación Parte B:** opción (α).

---

### Q-B.6 — Orden de activación

**Contexto:** Las clases activables son: INDEX_EOD (3 subclases), VOLATILITY_INDEX, RATE_YIELD, FX_DAILY_CUT. FUTURE_SETTLEMENT queda `BLOCKED`.

**Pregunta:** ¿Aprueba el orden propuesto?

**Opciones:**

- (α) INDEX_EOD + VOLATILITY_INDEX + RATE_YIELD → FX_DAILY_CUT → FUTURE_SETTLEMENT.
- (β) FX_DAILY_CUT primero (es el más simple, solo 3 tickers).
- (γ) Todo a la vez (activación simultánea de todas las clases activables).

**Consecuencia:**
- (α) Activa primero el grupo calendario USA+EU (más cohesionado), luego FX, luego FUTURE.
- (β) Empieza por lo pequeño, valida el mecanismo con menos tickers.
- (γ) Activa todo, difícil aislar problemas si fallan.

**Recomendación Parte B:** opción (α).

---

### Q-B.7 — Comportamiento de consumidores con FUTURE bloqueado

**Contexto:** 5 consumidores usan futuros: `flows_primary`, `mte_confirmation`, `sectors_base`, `macro_regime`, `mte`. Con FUTURE_SETTLEMENT `BLOCKED`, sus históricos no son auditable.

**Pregunta:** ¿Cómo deben comportarse esos consumidores mientras la clase esté bloqueada?

**Opciones:**

- (α) Consumir futuros igualmente, asumiendo el riesgo. Documentar el riesgo en los outputs.
- (β) Excluir futuros del cálculo hasta desbloqueo. Degradar cobertura.
- (γ) Consumir futuros pero declarar `status=UNRELIABLE` en sus outputs.

**Consecuencia:**
- (α) Simplicidad operativa, pero outputs no auditables para esos componentes.
- (β) Outputs auditables, pero cobertura reducida. Afecta a `mte`, `macro_regime`, `sectors_base`.
- (γ) Mantiene outputs, declara explícitamente la falta de garantía temporal.

**Recomendación Parte B:** opción (γ).

---

### Q-B.8 — Desbloqueo de A3.1

**Contexto:** Las 6 condiciones del dictamen previo para desbloquear A3.1 son: contrato por clase, regla de consolidación, propagación, writers, MTE, darkpool. Las 2 primeras están cerradas documentalmente. Las 4 restantes requieren implementación.

**Pregunta:** ¿Debe reconsiderarse el desbloqueo de A3.1 ahora, o mantenerlo bloqueado hasta completar las 4 condiciones restantes?

**Opciones:**

- (α) Mantener bloqueada hasta completar las 4 condiciones. A3.1 se ejecuta al final.
- (β) Reconsiderar tras la Parte B. A3.1 podría ejecutarse parcialmente (por ejemplo, sobre EQUITY_EOD solo).
- (γ) Desbloquear A3.1 ya, aceptando las condiciones pendientes como deuda documentada.

**Consecuencia:**
- (α) Secuencia limpia, A3.1 espera a la implementación completa.
- (β) Activa EQUITY_EOD primero, deja las demás clases para después.
- (γ) Riesgo alto: retirar el trim sin propagación, writers, MTE y darkpool resueltos.

**Recomendación Parte B:** opción (α).

---

## 8. Priorización de las preguntas

Si el auditor quiere acotar, el orden de importancia es:

| Prioridad | Preguntas | Motivo |
|---|---|---|
| **Alta** | Q-B.4 (FUTURE BLOCKED), Q-B.7 (consumidores), Q-B.8 (A3.1) | Bloquean trabajo real |
| **Media** | Q-B.1 (STALE), Q-B.2 (per_pair), Q-B.3 (per_ticker) | Afectan diseño de Parte A v2 |
| **Baja** | Q-B.5 (modificaciones Parte A), Q-B.6 (orden activación) | Ratificación formal |

---

**Fin del briefing.**
**HEAD de referencia:** 468fc08 (origin/main).
**Fecha:** 2026-09-16.
**Estado:** pendiente de dictamen. Parte A publicada, Parte B propuesta.