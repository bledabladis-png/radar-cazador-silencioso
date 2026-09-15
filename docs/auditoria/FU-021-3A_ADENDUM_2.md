# FU-021-3A — Adendum 2: cierre documental del Ciclo 2



**Fecha:** 2026-09-15

**HEAD de referencia:** 0c888c6 (origin/main)

**Adendum 1:** docs/auditoria/FU-021-3A_ADENDUM.md (694b172)

**Informe Ciclo 2:** docs/auditoria/CICLO2_CIERRE_AUDITORIA.md (bd18d22)

**Briefing auditor:** docs/auditoria/BRIEFING_AUDITOR_CONTRATO_TEMPORAL.md (0c888c6)

**Dictamen que motiva este adendum:** Dictamen del auditor externo — Contrato temporal de `df_market` (2026-09-15)

**Estado:** documento activo. No modifica el informe original ni el Adendum 1.



\---



## 1. Motivo del adendum



El Ciclo 2 de auditoría (15 indicadores de Capa 2b) ha producido:



\- 28 hallazgos nuevos (C21–C48).

\- Una corrección terminológica sobre la nomenclatura de consumidores.

\- Una extensión del análisis de patrones de writer (P0–P8).

\- La confirmación de los dos estados persistentes de `mte.py`.

\- La resolución de C16, C19 y C20.



Este adendum cierra documentalmente el Ciclo 2 con:



1\. Corrección terminológica ("30 + 1 + 1 + 1").

2\. Cronología del Ciclo 2.

3\. Registro de los hallazgos C21–C48 clasificados por acción.

4\. Consolidación del análisis de patrones P0–P8.

5\. Referencia al dictamen del contrato temporal de `df_market`.

6\. Estado del sistema tras el dictamen.



No reabre el informe original FU-021-3A ni el Adendum 1. Los complementa.



\---



## 2. Corrección terminológica



### 2.1. Observación del auditor



El dictamen del contrato temporal señala:



> *"Se habla de '30 módulos consumidores en cascada' y también de `darkpool` como excepción. Conviene reservar **30 consumidores por firma + 1 consumidor de artefacto + 1 productor**."*



### 2.2. Terminología definitiva adoptada



A partir de este adendum, la nomenclatura oficial de los 33 ficheros que mencionan `df_market` es:



| Categoría | Cuenta | Definición |

|---|---|---|

| Productor | 1 | Crea/carga `df_market` (`src/pipeline/data_load.py`) |

| Consumidor directo (Capa 1) | 12 | `src/pipeline/*`, recibe `df_market` por firma desde `run.py` |

| Consumidor indirecto Capa 2a | 3 | `regimes/*`, recibe `df_market` a través de Capa 1 |

| Consumidor indirecto Capa 2b | 15 | `indicators/*`, recibe `df_market` a través de Capa 1 o 2a |

| Consumidor de artefacto | 1 | `indicators/darkpool.py`, lee `market_data.parquet` de disco |

| Orquestador | 1 | `run.py`, propaga pero no consume |

| **Total ficheros que mencionan `df_market`** | **33** | |



**Consumidores por firma:** 12 + 3 + 15 = **30**.

**Consumidores totales (incluyendo artefacto):** 30 + 1 = **31**.

**Total ficheros del pipeline (incluyendo productor y orquestador):** 31 + 1 + 1 = **33**.



### 2.3. Uso en documentación futura



\- "consumidor" se reserva a quien recibe `df_market` por firma.

\- "consumidor de artefacto" identifica a `darkpool.py` y a cualquier módulo futuro con el mismo patrón.

\- "productor" identifica a `data_load.py`.

\- "orquestador" identifica a `run.py`.



Esta terminología se aplica en la especificación de FU-021-5.



\---



## 3. Cronología del Ciclo 2



| Fecha | Hito | Commit |

|---|---|---|

| 2026-09-15 | Punto de entrada (A2.3 cerrado) | 7c44b4a |

| 2026-09-15 | Adendum 1 publicado | 694b172 |

| 2026-09-15 | Fix C19 pusheado | 33ba2fa |

| 2026-09-15 | Fix C16 pusheado | d3790b3 |

| 2026-09-15 | Informe de cierre Ciclo 2 publicado | bd18d22 |

| 2026-09-15 | Briefing para auditor publicado | 0c888c6 |

| 2026-09-15 | Dictamen del contrato temporal recibido | — |

| 2026-09-15 | Este Adendum 2 | (pendiente) |



### 3.1. Bloques ejecutados



| Bloque | Contenido | Resultado |

|---|---|---|

| B0 | Verificación post-adendum | HEAD alineado, Adendum 1 verificado |

| B1 | C19 — `indices_intl` sin `reference_date` | Fix pusheado |

| B2 | C20 — `select_index_leaders(None, ...)` | A — vestigial, sin cambio |

| B3 | C16 — propagación `effective_meta` | Fix pusheado |

| B4 | Capa 2b (15 indicadores) | 15 × B. Sin C/D |

| B5 | darkpool migración | Diferido por decisión de secuencia (resuelto por dictamen: integrado en FU-021-5) |

| B6 | Cierre del Ciclo 2 | Informe + Briefing |

| — | Dictamen del contrato temporal | 8 preguntas resueltas |

| — | Este Adendum 2 | Cierre documental |



### 3.2. Estado del sistema al cierre del Ciclo 2



| Métrica | Valor |

|---|---|

| HEAD al recibir dictamen | 0c888c6 |

| Tests locales | 369 passed + 2 skipped |

| Gate | 10/10 |

| pyflakes | 0 warnings |

| compileall | OK |

| Fixes del Ciclo 2 | C19 (33ba2fa), C16 (d3790b3) |

| Verificaciones cerradas | C20 |



\---



## 4. Registro de hallazgos C21–C48



Los 28 hallazgos detectados durante el Ciclo 2, clasificados por acción requerida.



### 4.1. Condicionan diseño de FU-021-5



Hallazgos que deben integrarse en la especificación arquitectónica del contrato temporal.



| ID | Descripción | Origen | Sensibilidad |

|---|---|---|---|

| C22 | `cross_asset_context.py:88,113` — fecha escrita desde `returns_df` (patrón P3) | Ficha 20 | Alta |

| C26 | `rs_internal.py:79` — fecha escrita desde serie intersectada (patrón P4) | Ficha 23 | Media |

| C27 | `sector_leader_divergence.py:32,57` — asimetría `df_market` vs `df_stocks` sin alinear | Ficha 25 | Alta |

| C28 | `sector_leader_divergence.py:84` — fecha escrita sobre `df_stocks` (patrón P5) | Ficha 25 | Media |

| C31 | `sector_breadth.py` — patrón P6, **modelo correcto** | Ficha 27 | Positivo |

| C35 | `volatility_structure.py:84` — patrón P7 (`_observation_date_from_df(vix)`) | Ficha 29 | Media |

| C38 | `stock_leader.py:141-144` — writer `analisis_lideres.csv` sin columna `date` (P0) | Ficha 30 | Media |

| C43 | `mte.py` — doble writer de estado JSON con histéresis persistente | Ficha 32 | **Alta** |

| C48 | `mte.py` — `STATE_FILE` y `MTE_STATE_FILE` divergentes | Ficha 32 | Media |



### 4.2. Hallazgos documentales preexistentes



| ID | Descripción |

|---|---|

| C21 | `credit.py:36` — `dropna()` reindexa implícitamente |

| C24 | `index_phase.py:26` — `period='5y'` sin `reference_date` |

| C29 | `index_leaders.py:21` — `period='1y'` sin `reference_date` |

| C30 | `index_leaders.py` — alineación asimétrica (RS sí, Wyckoff no) |

| C33 | `vol_metrics.py` — `min_periods` relajados |

| C34 | `vol_metrics.py` — sin declarar fecha de observación (R1) |

| C39 | `stock_leader.py:34-80` — `flow_proxy_z` guardado como EMA, no z-score |

| C40 | `slpm_v12.evaluate_slpm_v12` — `df_market` vestigial |

| C42 | `slpm.py` — propaga `df_market` vestigial a `slpm_v12` |

| C44 | `credit_stress_score` — no recibe `df_market` |

| C45 | `nfci_series`/`credit_oas_series` — siempre `None` |

| C47 | `classify_mte` — desempate no alineado con matriz de transiciones |



### 4.3. Deudas de robustez



| ID | Descripción |

|---|---|

| C23 | `index_phase.py:19` — línea muerta |

| C25 | `index_phase.py:18,30` — bare `except:` |

| C32 | `sector_breadth.py:as_of_date=None` — sin validación |

| C36 | `volatility_structure.py` — `pcr_data` con campos opcionales |

| C37 | `stock_leader.py:125-133` — bloque `rho` calculado y descartado |

| C41 | `slpm_v12.py:2-8` — `_safe_mean` definido antes de imports + antes de docstring |

| C46 | `mte.py` — bare `except:` en `sector_rotation_score` y `compute_mte` |



### 4.4. Resumen por categoría



| Categoría | Cuenta |

|---|---|

| Condicionan FU-021-5 | 9 |

| Documentales preexistentes | 12 |

| Deudas de robustez | 7 |

| **Total** | **28** |



\---



## 5. Consolidación del análisis de patrones P0–P8



### 5.1. Tabla definitiva



Nueve patrones identificados para decidir la fecha escrita en artefactos (CSV o JSON).



| Patrón | Módulos | Objeto de fecha | Valida sesión | Declara cobertura |

|---|---|---|---|---|

| **P0** | `stock_leader.py` | Sin columna `date` | — | No |

| **P1** | `engines.py` | `_observation_date_from_df(df_market)` | No | No |

| **P2** | `sectors_base.py` (7 writers) | `df_market.index[-1]` | No | No |

| **P3** | `cross_asset_context.py`, `sector_correlation.py` | `_observation_date_from_df(returns_df)` | No | No |

| **P4** | `rs_internal.py` | `_observation_date_from_df(intersect)` | No | No |

| **P5** | `sector_leader_divergence.py` | `_observation_date_from_df(df_stocks)` | No | No |

| **P6** | `sector_breadth.py` | `as_of_date` (caller, validado) | **Sí** | Parcial |

| **P7** | `volatility_structure.py` | `_observation_date_from_df(vix)` | No | No |

| **P8** | `mte.py` | JSON de estado sin fecha | No | No |



### 5.2. Interpretación



\- **P0–P8** representan **nueve autoridades temporales implícitas** en el pipeline.

\- **Solo P6** aplica un control de sesión bursátil (B2).

\- **Ningún patrón declara cobertura** del universo elegible (R1 incumplida).

\- **P8** introduce un tipo nuevo: el estado persistente entre runs. No es CSV, es JSON. Su semántica temporal afecta a la decisión del run siguiente.



### 5.3. Cobertura de la corrección



El contrato temporal de FU-021-5 debe:



1\. **Unificar P0–P7** bajo una única semántica de fecha (Q-C2.3 resuelto por dictamen: opción d).

2\. **Extender P6** a los 20 writers históricos.

3\. **Redefinir P8** bajo el esquema versionado (Q-C2.4 resuelto por dictamen: opción c+b).



\---



## 6. Referencia al dictamen del contrato temporal



### 6.1. Documento



Dictamen del auditor externo — Contrato temporal de `df_market` (2026-09-15).



### 6.2. Decisiones registradas



| Pregunta | Decisión del auditor |

|---|---|

| Q-C2.1 | 🟢 GO arquitectura / 🔴 NO-GO activación |

| Q-C2.2 | 🟢 γ — `darkpool` integrado en FU-021-5 |

| Q-C2.3 | 🟢 (d) — `date` + `effective_date` + `expected_session` + `coverage` |

| Q-C2.4 | 🟢 c+b — path único + versionado + fecha + reset por contrato |

| Q-C2.5 | 🟢 Diseñar FU-021-5 ahora + actualizar adendum |

| Q-C2.6 | 🟢 α — FU-021-5 prioridad |

| Q-C2.7 | 🟢 Mantener C7/C8 vacantes; siguiente hallazgo C49 |

| Q-C2.8 | 🟢 Contratos por clase + consolidación explícita |



### 6.3. Consecuencias inmediatas



\- **A3.1** sigue bloqueada (mejor definida: requiere los 6 criterios del §8.2).

\- **FU-021-5** pasa a `GO — DISEÑO ARQUITECTÓNICO AUTORIZADO`.

\- **`darkpool`** deja de ser decisión pendiente; queda subordinado a FU-021-5.

\- **Próximo documento:** especificación formal del contrato temporal de `df_market`.



### 6.4. Autorización concreta



El dictamen exige que el próximo documento no sea un patch, sino la **especificación formal** del contrato temporal, con:



1\. Cinco contratos por clase.

2\. Regla de consolidación de `df_market`.

3\. Metadata de fecha/cobertura por artefacto.

4\. Contrato de writers.

5\. Propagación hacia los 30 consumidores.

6\. Rediseño del estado MTE.

7\. Migración de `darkpool`.



\---



## 7. Estado del sistema tras el dictamen



### 7.1. Métricas



| Métrica | Valor |

|---|---|

| HEAD | 0c888c6 (origin/main) |

| Tests locales | 369 passed + 2 skipped |

| Gate | 10/10 |

| pyflakes | 0 warnings |

| compileall | OK |

| A3.1 | 🔴 bloqueada |

| FU-021-5 arquitectura | 🟢 autorizada |

| FU-021-5 activación | 🔴 bloqueada |

| FU-021-3B/3C | 🔴 bloqueadas |

| `darkpool` | 🟢 integrado en FU-021-5 |



### 7.2. Criterios de desbloqueo de A3.1



Según el dictamen, A3.1 se desbloqueará cuando:

✅ contrato de cada clase relevante

✅ regla de consolidación de df_market

✅ propagación de effective metadata

✅ writers adaptados

✅ estado MTE compatible

✅ darkpool fuera de fuente paralela



text



Y después:

A3.1 → retirar trim_to_last_valid_date



text



### 7.3. Secuencia aprobada

Ciclo 1 (auditoría) ✅ cerrado

Adendum 1 ✅ publicado (694b172)

Ciclo 2 (auditoría) ✅ cerrado (bd18d22, 0c888c6)

Adendum 2 ✅ este documento

Dictamen contrato ✅ recibido

─────────────────────────────

FU-021-5 arquitectura 🟢 GO

FU-021-5 Parte A ⏸ pendiente de redacción

FU-021-5 Parte B 🔴 bloqueada por 3B/3C

FU-021-3B 🔴 bloqueada

FU-021-3C 🔴 bloqueada

A3.1 🔴 bloqueada



text



\---



## 8. Nomenclatura de hallazgos



### 8.1. C7/C8



Los identificadores C7 y C8 permanecen **vacantes** por decisión del dictamen (Q-C2.7). No se reutilizan ni se renumeran. El próximo hallazgo nuevo será **C49**.



### 8.2. Convención



\- **C1–C48**: hallazgos del registro acumulado.

\- **C7, C8**: vacantes por convención.

\- **C49+**: nuevos hallazgos.

\- **H-DP-1**: hallazgo estructural independiente (no numerado en la serie C).

\- **P0–P8**: patrones de writer de fecha.

\- **Q-C1.x, Q-C2.x**: preguntas al auditor por ciclo.



\---



## 9. Cierre del expediente documental



### 9.1. Documentos vigentes



| Documento | Commit | Rol |

|---|---|---|

| `FU-021-3A_INFORME_Y_DICTAMEN.md` | `fded14b` | Informe original FU-021-3A |

| `FU-021-3A_ADENDUM.md` | `694b172` | Corrección del "13" → 33. Hallazgos C1–C20 + H-DP-1 |

| `CICLO2_CIERRE_AUDITORIA.md` | `bd18d22` | Informe técnico del Ciclo 2. Hallazgos C21–C48 |

| `BRIEFING_AUDITOR_CONTRATO_TEMPORAL.md` | `0c888c6` | Briefing autocontenido para auditor |

| **`FU-021-3A_ADENDUM_2.md`** | (este) | Cierre documental del Ciclo 2 |



### 9.2. Trazabilidad completa

FU-021-3A (original)

↓

Adendum 1 (694b172)

↓

Ciclo 1 cerrado

↓

Dictamen Ciclo 1

↓

Ciclo 2 ejecutado (33ba2fa, d3790b3)

↓

Ciclo 2 cerrado (bd18d22)

↓

Briefing auditor (0c888c6)

↓

Dictamen contrato temporal

↓

Adendum 2 (este documento)

↓

FU-021-5 (especificación arquitectónica)

↓

FU-021-3B / 3C

↓

A3.1



text



### 9.3. Lo que este adendum NO decide



\- No autoriza A3.1.

\- No autoriza migración de `darkpool` fuera de FU-021-5.

\- No activa contratos por clase.

\- No modifica el informe original ni el Adendum 1.

\- No modifica la especificación de FU-021-5 (que se redactará después).



\---



**Fin del Adendum 2.**

**HEAD de referencia:** 0c888c6 (origin/main).

**Fecha:** 2026-09-15.

**Estado:** cierre documental del Ciclo 2. Complementa al Adendum 1 y al informe de cierre del Ciclo 2.

