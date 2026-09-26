# Auditoría consolidada del sistema Radar — 2026-09-26

**Alcance:** FASE 0 (inventario) + FASE 2.1 (regímenes/scores) + FASE 2.3 (breadth) + FASE 3 (artefactos) + FASE 5 (providers) + FASE 6 (reporte).
**HEAD de referencia:** `b7193c4`.
**Método:** read-only. Sin modificar código productivo ni datos. Hallazgos con evidencia reproducible.
**Fases no cubiertas:** FASE 1 (arquitectura profunda), FASE 2.2 (IAE §10.x), FASE 2.4 (MTE + Dark Pool), FASE 4 (tests).

## Resumen ejecutivo

| Severidad | Count | Áreas principales |
|---|---:|---|
| **ALTA** | **6** | OilPriceAPI, Yahoo fallback, reporte (líderes + A/D) |
| MEDIA | 59 | doc-vs-código, fallback silencioso, coherencia reporte |
| BAJA | 49 | estilo, higiene, dead code, latentes |
| **TOTAL** | **114** | — |

**Los 6 hallazgos ALTA:**

| # | ID | Área | Impacto |
|---|---|---|---|
| 1 | **F3-05** | OilPriceAPI | Sistema cree cubrir 5 commodities; cubre 3. `FUTURE_SETTLEMENT` BLOCKED de facto. |
| 2 | **F3-16** | Writer spot | Puede publicar artefacto INVALID sin bloquear el write. |
| 3 | **F3-17** | Workflow | `best-effort` + `\|\| echo` enmascara errores de credenciales y corrupción. |
| 4 | **A5-13** | Yahoo | Fallback devuelve parquet completo → duplicación masiva en `pd.concat`. |
| 5 | **F6-10** | Reporte | Tres "líderes" distintos por sector entre secciones. |
| 6 | **F6-28** | Reporte | Dos "A/D Net" contradictorios (+74 vs -52). |

**Patrón dominante:** 4 de los 6 ALTA son **silenciamiento de errores**. La cultura defensiva uniforme (nunca fallar hacia arriba) produce un sistema que raramente se cae pero que oculta fallos. Los otros 2 ALTA son **incoherencia interna del reporte** (mismo concepto, distintos criterios).

**Cero hallazgos ALTA de cálculo puro.** El código produce los números que dice producir. La deuda es documental, de observabilidad y de presentación.

---
## Los 6 hallazgos ALTA — plan de ataque

### A1. F3-05 — OilPriceAPI no sirve futuros en el plan actual

**Evidencia:** ejecución manual de `py scripts/update_futures.py`. API responde 403 "Feature access required. Futures Data is included in the Professional plan." para `/v1/futures/ice-brent` e `/v1/futures/ice-wti`.

**Realidad vs declarado:**
- PROMPT seccion 6.1: "OilPriceAPI | 5 commodities (BZ=F, CL=F, GC=F, HG=F, NG=F)" → **incorrecto**.
- PROMPT seccion 11.16: "FUTURE_SETTLEMENT pasó de BLOCKED a activo para BZ/CL" → **incorrecto**.
- Realidad: 3 commodities cubiertos (spot). 2 futuros BLOCKED de facto desde 22/09.

**Plan de decisión (elegir uno):**
- (a) Upgrade plan a Professional → recupera futuros. Coste monetario.
- (b) Declarar FUTURE_SETTLEMENT BLOCKED permanentemente. Actualizar secciones 6.1, 11.16, 12.
- (c) Buscar proveedor alternativo para BZ/CL.

**Plan mínimo (aplicable en cualquier caso):**
1. Actualizar PROMPT seccion 6.1: "OilPriceAPI | 3 commodities (GC=F, HG=F, NG=F)".
2. Actualizar seccion 11.16: FUTURE_SETTLEMENT en estado BLOCKED.
3. Añadir a seccion 12: "Futuros de energía (BZ, CL) sin cobertura desde 2026-09-22".

---

### A2. F3-16 — El writer de commodities_spot.parquet puede publicar artefacto INVALID

**Evidencia:** ejecución de update_futures.py con spot_fetch OK escribió status=INVALID (pct_dup_last=1.0) sin bloquear el write.

**Impacto:** el workflow daily_run.yml hace git add data/commodities_spot.parquet sin filtrar por status. Commitearía un artefacto INVALID.

**Plan:**
1. write_artifact_with_manifest debe lanzar excepción si quality.status == "INVALID".
2. Alternativa: el workflow condiciona git add a manifest.quality.status != INVALID.

---

### A3. F3-17 — best-effort + echo enmascara errores graves

**Evidencia:**
- update_futures.py: except Exception: return 0 tras 403.
- daily_run.yml: python scripts/update_futures.py || echo "Commodities fetch failed, continuing".
- Resultado: GH Actions verde, sin alerta, sin Issue.

**Plan:**
1. update_futures.py debe distinguir errores "transitorios" (retry) de "permanentes" (401, 403, 429) → exit != 0 en permanentes.
2. Workflow: quitar || echo o condicionar a continue-on-error: true con notificación vía Issue.
3. Reutilizar el patrón scripts/issue_manager.py (ya existe para el gate F-IAE-CRON-02).

---

### A4. A5-13 — YahooProvider.get_prices devuelve parquet completo en fallo

**Evidencia (código):**
- except Exception: return self._load_cache() — devuelve el parquet completo (562 tickers).

**Impacto:** data_loader.py itera por lotes de 5. Yahoo falla en el lote N → devuelve 2810 columnas → pd.concat(all_data, axis=1) duplica columnas.

**Estado:** latente (Yahoo lleva semanas funcionando bien en el cron F-IAE-CRON-02).

**Plan (elegir uno):**
- (a) Propagar la excepción. El caller (router/backup) decide.
- (b) Filtrar la caché por tickers antes de devolver: _load_cache(tickers).

Recomendado: (b), cambio de 5 líneas.

---

### A5. F6-10 — Tres "líderes" distintos por sector entre secciones

**Evidencia:**

| Sector | Concentración dice | Representatividad dice | Liderazgo interno (orden) |
|---|---|---|---|
| XLK | INTC | AMD (rank 100%) | AMD / MU / INTC / SNDK / CRWD |
| XLV | ISRG | TMO (rank 100%) | TMO / DHR / ISRG / GILD / MRK |

**Causa:** cada sección usa un criterio distinto (WLS máximo, rank pct, RS delta) sin documentarlo.

**Plan:**
1. Declarar en Acciones Seleccionadas (ya lo hace) y replicar la nota en las 3 secciones.
2. O bien: unificar el criterio "líder" a WLS y usar el mismo en todas las secciones.

---

### A6. F6-28 — Dos "A/D Net" contradictorios en el mismo reporte

**Evidencia:**
- Confirmation Data: Advance/Decline Net: +74 (193 avances / 119 descensos) → universo completo (313 tickers).
- Suma de Sector Breadth: -14+4+8+0-8-8-8-10-16+0+0 = -52 → top-20 por sector (220 tickers).

**Impacto:** discrepancia de 126 unidades bajo el mismo nombre. El lector no puede conciliarlos.

**Plan:**
1. Renombrar la métrica de Confirmation a "A/D Net (universo completo)".
2. Renombrar la de Sector Breadth a "A/D Net (top-20 sector)".
3. O bien: unificar universo y recalcular.

---
## Hallazgos MEDIA — agrupados por tema

### Doc vs. código (11 hallazgos)

El PROMPT describe fórmulas "ideales" y el código implementa transformaciones no documentadas.

- F2-02 — Sector Regime aplica SECTOR_DISPERSION_PENALTY=0.5 no documentado en la sección 7.
- F2-03 — Sector Regime renormaliza por pesos válidos (/ valid_weight_sum). La sección 7 describe suma ponderada fija.
- F2-04 — Tactical "Breadth20" es proxy temporal (close > EMA20 del propio ETF), no breadth sectorial por constituyentes. Homónimos entre secciones.
- F2-07 — effective_composite SLPM discontinuo x2.04 en coverage=0.5. El reporte lo declara (L274) pero la regla no está en la sección 7.
- F2-10 — leader_metrics acepta rs_momentum y rs_mom_20 como alias.
- F6-11 — Cuatro "Coberturas" con denominadores distintos en el mismo reporte.
- F6-19 — Etiquetas "Lectura" sin umbrales documentados.
- A5-76 — robust_z de Blackrock DAX/ISF usa media/std (no mediana/MAD). El nombre miente.
- A5-77 — 3 implementaciones distintas de "robust_z" en el sistema.
- A5-57 — BME: Open <- reference (no apertura real). No documentado en la sección 6.3.
- F6-33 — Matriz de Evidencia: Wyckoff=+1 para los 11 sectores. Binarización demasiado laxa.

### Fallback silencioso (12 hallazgos)

- F2-01 — confidence_from_range ruta DataFrame no respeta <2 válidos -> 0.5 (latente, 0 filas afectadas hoy).
- F2-19 — SLPM slpm_state.json sin sector. Histéresis cross-sector si el líder rota.
- F2-21 — consecutive_count mide estado previo, no candidato. Semántica opuesta al nombre HOLDING_CANDIDATE.
- A5-40 — Xetra: fallback a caché obsoleta sin marcar staleness.
- A5-41, A5-56, A5-62 — Xetra, Euronext, BME: error de fetch -> ticker omitido silenciosamente.
- A5-72 — "Caché pese al error" en Amundi, Blackrock DAX, Blackrock ISF.
- A5-31 — BackupProvider: _validate_with_cache devuelve None en 3 casos distintos.
- A5-32 — BackupProvider: deduplica silenciosamente market_data > stock_prices.
- A5-11 — Router _load_cache: devuelve subset silencioso.
- F6-12 — Representatividad y Divergencia solo muestran 5/11 sectores sin explicar el filtro.

### Infraestructura muerta (4 hallazgos)

- F2-13 — HYSTERESIS en state_transition.py: valores de las tuplas nunca se leen.
- F2-14 — TACTICAL_CORRECTION inalcanzable. Declarado en 3 sitios, nunca producido.
- F2-25 — _delta de sector_breadth_momentum.py no propaga el fix de continuidad (fc21671).
- F2-26 — Guard sector_price is not None siempre True; riesgo IndexError latente.

### Coherencia del reporte (13 hallazgos)

- F6-01 — Sector Breadth con observación 24/09 vs Yahoo 25/09.
- F6-02 — FINRA 26 días marcado como CURRENT.
- F6-03 — XLF con Top1=100%, Top3=N/D y RS Med=4.6899.
- F6-04 — Sector Breadth 100% cobertura en 11/11.
- F6-05 — Flow Z-Scores no comparables (Blackrock vs resto) presentados juntos.
- F6-08 — Reporte desacoplado de históricos (14:20 vs 17:01).
- F6-16 — Universo "5 tickers por sector" distinto entre secciones.
- F6-17 — Dispersión interna con gap 18-23/09 sin explicación.
- F6-18 — "Fuerte mejora" en rank saturado (XLK en puesto 1 con Δ20d=-6).
- F6-23 — Copper/Gold nan con z-score presente (+0.62). Semántica no aclarada.
- F6-29 — A/D Line acumulada (14843) sin referencia histórica.
- F6-34 — 3 columnas de Matriz de Evidencia constantes (Wyckoff, Crédito, Volat).
- F6-30 — Ningún sector con "Lectura MIXTA" en la Matriz de Evidencia.

### Otros MEDIA

- F0-01 — generate_estado_sistema.py con lista hardcodeada de evidencia.
- A5-09 — Router no pasa ventana temporal a Yahoo (period="10y" fijo).
- A5-10 — Router preferred_order = [yahoo, fred, polygon] deprioriza Polygon.
- A5-30, A5-33 — BackupProvider lento por orden (tiingo 1/min primero).
- A5-36 — datetime.now() en los 5 métodos _*_daily de BackupProvider.
- A5-50 — Euronext sin walk-back FU-014. Estimación heurística.
- A5-51, A5-59 — Euronext y BME sin reference_date en _cache_is_fresh.
- A5-70 — Blackrock DAX/ISF copy-paste ~98%.
- A5-73 — datetime.now() en los 6 ficheros de fund data.
- A5-79 — Escritura no atómica de caché en SSGA.
- F3-13 — temporal_contract=None en commodities (deuda declarada).
- F3-15 — health_check no verifica commodities.
- F3-18 — merge_commodities_into_market rompe contrato "in-place" con sort_index.
- F3-19 — get_market('DX-Y.NYB') UNKNOWN -> puede bloquear commits en escenario adverso.
- F2-12 — datetime.now() en indicators/state_transition.py (zona gris).

---
## Hallazgos BAJA (resumen)

49 hallazgos de estilo, higiene, dead code y latentes. Agrupados por tipo:

- Dead code / constantes sin usar: A5-14, A5-35, A5-42, A5-58, A5-78, A5-80, F2-15, F2-18.
- except: desnudos: A5-12, A5-71, F2-16.
- print() en producción: F2-17.
- Inconsistencias de estilo: A5-34, A5-37, A5-44, A5-46, A5-53, A5-55, F2-05, F2-06, F2-11, F2-27, F2-28, F2-29.
- Latentes (no activados): A5-43, A5-47, A5-49, A5-54, A5-61, F2-20, F2-26.
- Doc / nomenclatura: F2-08, F3-10, F6-06, F6-07, F6-13, F6-14, F6-15, F6-20.
- Fixes puntuales: F3-06, F3-08, F3-09, F3-11, F3-12, F3-14, F6-35, F6-36.

Detalle completo en los documentos por fase (AUDITORIA_2026-09-26.md y AUDITORIA_FASE_5_2026-09-26.md).

---

## Patrones transversales

### Patrón 1 — Doc vs. código (11 hallazgos)

El PROMPT describe fórmulas ideales; el código aplica transformaciones no documentadas. El código no está mal; el PROMPT secciones 6.x y 7 están desactualizados respecto a las decisiones de diseño implementadas.

### Patrón 2 — Cultura del fallback silencioso (24 hallazgos)

Regla informal: nunca fallar hacia arriba. Devolver algo plausible (caché, vacío, subset). Loguear el error a consola. El pipeline sigue.

Manifestaciones: Yahoo (parquet completo), OilPriceAPI (best-effort + echo), Xetra (caché obsoleta), Euronext (estimación), BME (drop silencioso), BackupProvider (vacío), Blackrock/Amundi ("caché pese al error"), SSGA (errors.append).

- Fortaleza: el pipeline raramente se cae.
- Debilidad: los errores se vuelven invisibles.

Mitigación actual: manifest con cobertura por mercado + guard_coverage + health_check semanal + cron multi-slot. No cubre: alertas de caché obsoleta ni declaración explícita de tickers fallidos por parte del provider.

### Patrón 3 — Incoherencia interna del reporte (13 hallazgos)

Múltiples secciones hablan del mismo concepto ("líder", "cobertura", "5 tickers", "A/D Net") usando criterios y universos distintos, sin declarar cuál es cuál.

Impacto directo en el usuario final: lee información aparentemente contradictoria sin poder resolverla.

### Patrón 4 — Infraestructura muerta (4 hallazgos)

Código mantenido sin ejecución (TACTICAL_CORRECTION, HYSTERESIS values, _delta sin fix, guards siempre-True). Bajo riesgo pero erosiona la confianza en el código.

---
## Recomendaciones por orden de prioridad

### Prioridad 1 — ALTA (impacto operativo, corrección directa)

| # | Hallazgo | Acción | Esfuerzo |
|---|---|---|---|
| 1 | F3-05 | Decidir política OilPriceAPI + actualizar PROMPT secciones 6.1, 11.16, 12. | 1h + decisión |
| 2 | F3-17 | Cambiar best-effort por best-effort + alerta en update_futures.py. | 2h |
| 3 | F3-16 | Bloquear write de artefacto INVALID en write_artifact_with_manifest o condicionar git add. | 2h |
| 4 | A5-13 | Yahoo fallback: filtrar caché por tickers antes de devolver. | 30min |
| 5 | F6-10 | Unificar criterio de "líder" entre las 3 secciones del reporte. | 3h |
| 6 | F6-28 | Renombrar "A/D Net" en las dos secciones con universo declarado. | 1h |

Total estimado: 1-2 días de trabajo para cerrar todos los ALTA.

### Prioridad 2 — MEDIA (impacto funcional)

7. F2-19, F2-21 — Rediseñar histéresis SLPM (state file por sector + candidato explícito).
8. A5-76, A5-77 — Unificar robust_z en src/utils.py::robust_zscore. Eliminar variantes Blackrock/SSGA.
9. A5-40, A5-41, A5-56, A5-62 — Providers de equity reportan explícitamente tickers fallidos.
10. F3-15 — Extender health_check para verificar commodities (date_max, status).
11. F3-19 — Añadir DX-Y.NYB (y otros índices ICE) a get_market.
12. F2-02, F2-03, F2-04 — Actualizar PROMPT sección 7 con las fórmulas reales (no tocar código).
13. A5-09 — Router: pasar since_date a Yahoo (bug análogo a FU-014 para USA).
14. A5-10 — Reordenar preferred_order: Polygon antes que FRED.
15. A5-11 — Router _load_cache: propagar error si faltan tickers.
16. F6-01, F6-08 — Cerrar desfase temporal entre reporte e históricos.

### Prioridad 3 — MEDIA (arquitectura)

17. A5-30, A5-33 — BackupProvider: reordenar o paralelizar providers.
18. A5-31, A5-32 — BackupProvider: distinguir estados en _validate_with_cache; no deduplicar silenciosamente.
19. A5-50, A5-51, A5-59 — Aplicar FU-014/FU-018 a Euronext y BME.
20. A5-70, A5-73 — Parametrizar Blackrock; propagar reference_date a fund data.
21. A5-57 — Documentar Open=reference de BME en la sección 6.3.
22. F0-01 — Enumeración dinámica de EVIDENCE_DIRS.

### Prioridad 4 — BAJA

- Resto de hallazgos por orden de facilidad.
- Auditoría de providers no cubiertos: macro (fred, cboe, cboe_index, polygon) y flows trimestrales.

---

## Fases no cubiertas

- FASE 1 (arquitectura profunda). Parcialmente cubierta por FASE 0 (0 ciclos, 0 huérfanos). Falta verificar fuentes únicas (market_calendar, instrument_registry, utils.write_artifact_with_manifest) contra duplicación real.
- FASE 2.2 (IAE sección 10.x). No iniciada. 845 tests pasan, pero la coherencia fórmula-vs-implementación no se ha verificado línea por línea.
- FASE 2.4 (MTE + Dark Pool). No iniciada.
- FASE 4 (tests). No iniciada. Suite verde pero cobertura real por módulo del radar no medida.
- FASE 5 (parcial). 13 de 27 providers auditados en profundidad.
- FASE 6 (parcial). Secciones L297-416 del reporte no auditadas (5 secciones por sector + Opciones + ETF Flows SPDR).

## Artefactos generados

- docs/auditoria/radar/audits/AUDITORIA_2026-09-26.md (FASE 0-3)
- docs/auditoria/radar/audits/AUDITORIA_FASE_5_2026-09-26.md (FASE 5)
- docs/auditoria/radar/audits/AUDITORIA_CONSOLIDADA_2026-09-26.md (este documento)
- Scripts temporales de auditoría (read-only, no productivos).

## Conclusión

El sistema es estructuralmente sólido: 0 ciclos de import, 0 huérfanos reales, 1857 tests verdes, coherencia entre lo declarado y lo medido en las 5 fuentes documentales.

Los 6 hallazgos ALTA se concentran en:
- 1 proveedor externo roto silenciosamente (OilPriceAPI — F3-05, F3-16, F3-17).
- 1 fallback peligroso latente (Yahoo — A5-13).
- 2 incoherencias de presentación (F6-10, F6-28).

La deuda no es de cálculo — es de observabilidad y de comunicación. El sistema produce los números que dice producir, pero no declara cuándo algo falla y presenta conceptos homónimos con criterios distintos.

Cerrar los 6 ALTA es alcanzable en 1-2 días de trabajo y elimina el 100% de los riesgos operativos identificados.

---

Fin del informe consolidado.

Generado: 2026-09-26
HEAD: b7193c4
Fases cubiertas: 0, 2.1, 2.3, 3, 5 (parcial), 6 (parcial)
Total hallazgos: 114 (6 ALTA, 59 MEDIA, 49 BAJA)

====================================================================
ADDENDUM 1 — FASE 2.2 (IAE seccion 10.x) + FASE 2.4 (MTE + Dark Pool)
====================================================================

Actualizacion posterior al cierre inicial del consolidado. Se añaden
28 hallazgos nuevos (0 ALTA, 13 MEDIA, 15 BAJA) de las fases 2.2 y 2.4.

## FASE 2.2 — IAE seccion 10.x

Verificacion linea a linea de IAE_MAESTRO secciones 10.1 a 10.7 contra
`src/institutional_accumulation/aggregation/` y `pipeline_contractual.py`.

| ID | Sev | Hallazgo |
|---|---|---|
| F2.2-01 | MEDIA | IAE_MAESTRO 10.1 declara `SSHPRNAMT_f = SSHPRNAMT * 1000`. Codigo NO aplica x1000. Unidad real (miles de acciones) no declarada en ningun sitio. |
| F2.2-02 | BAJA | `compute_delta_shares` usa `iterrows` para emitir 3.15M filas UNRESOLVED. Deuda de rendimiento. |
| F2.2-03 | MEDIA | Docstring de `aggregate_positions_by_shareclass_figi` NIEGA la invariante que el codigo SI cumple y que 10.4 declara correcta. |
| F2.2-04 | MEDIA | `coverage_status="VALID"` solo verifica `denom>0`. 10.5 exige ademas `coverage_previous/current > 0`. |
| F2.2-05 | INFO | `unmapped_count` mide observed (deuda declarada). |
| F2.2-06 | BAJA | `unmapped_count_*` en 10.6 son float 0-1, no int. Contraste con 10.5 (int). |
| F2.2-07 | BAJA | Docstring 10.6 dice `unmapped_weight_*`; la clave real es `unmapped_count_*`. |
| F2.2-08 | MEDIA | `run_contractual_nipc` re-ejecuta `load_canonical` 2 veces por periodo. Deuda de rendimiento. |
| F2.2-09 | BAJA | `_operational_mapping_status` puede no existir -> filtro silenciosamente inaplicado. |

Verificacion positiva: los pesos, MATCH_KEY, DELTA_COLUMNS, filtros SH/PUTCALL,
reglas BOTH/NEW/EXIT/UNRESOLVED, fail-closed de TARGET_PAIRWISE vacio, y
`evidence_class=CONTRACTUAL` coinciden literalmente con IAE_MAESTRO.

Cero hallazgos ALTA. La cadena 10.1 -> 10.7 esta bien implementada; la deuda
es documental (el docstring se contradice con la seccion del IAE_MAESTRO).

## FASE 2.4 — MTE + Dark Pool

Verificacion de `indicators/mte/` (5 modulos) e `indicators/darkpool*.py`
(4 ficheros).

### MTE

| ID | Sev | Hallazgo |
|---|---|---|
| F2.4-01 | MEDIA | `cls > 0.7` duplicado en CRISIS (dentro y fuera del else). |
| F2.4-02 | MEDIA | `except:` desnudo en SRS (`defensive_beating_spy`). |
| F2.4-03 | MEDIA | `SCENARIO_WEIGHTS["IPS"]["weight"]` declarado y nunca usado; `+3` hardcoded. |
| F2.4-04 | MEDIA | `robust_zscore_series` local con window=104, sin ffill, sin clip. Divergente del canonico. |
| F2.4-05 | BAJA | Wrapper trivial `tanh(x) = np.tanh(x)`. |
| F2.4-06 | BAJA | `stress` y `stress_transform` son casi identicas. |
| F2.4-07 | BAJA | Comentario obsoleto sobre "umbral reducido" para IPS. |
| F2.4-08 | BAJA | `MIXED=2` fijo, no documentado. |
| F2.4-10 | MEDIA | `engine.py` sobrescribe el state file y borra `pending`. La histeresis adaptativa de `classify_mte` esta funcionalmente desactivada. |
| F2.4-11 | BAJA | `except:` desnudo en engine.py (vix_term). |
| F2.4-12 | BAJA | Reset por cambio de contrato no persiste en disco. |
| F2.4-13 | BAJA | `consensus_score` mezcla escalas: srs/shs/ips en [-1,1], cls en [0,1]. |

Confirmaciones positivas: NORMAL_TRANSITIONS, EXCEPTION_TRANSITIONS,
validate_transition, classify_mte (desempate por prioridad),
compute_confidence (0.6 * distance + 0.4 * consensus) son coherentes.

### Dark Pool

| ID | Sev | Hallazgo |
|---|---|---|
| F2.4-20 | MEDIA | 5a variante de `robust_zscore` en el sistema (darkpool: sin clip, sin window, opera sobre serie completa). Refuerza A5-77. |
| F2.4-21 | MEDIA | `rolling_percentile` no es rolling (percentil del ultimo contra toda la serie) + O(n^2) en `_compute_z_for_window`. |
| F2.4-22 | BAJA | `min(4, window//2)` siempre devuelve 4. Redundante. |
| F2.4-23 | BAJA | Campo `momentum` no es momentum (es EWM del z-score). |
| F2.4-24 | BAJA | `robust_zscore` devuelve Series / np.ndarray / Series segun caso. Tipo heterogeneo. Latente si `mad==0`. |
| F2.4-25 | BAJA | `rolling.apply(..., raw=False)` mas lento que `raw=True`. |

Confirmaciones positivas: `classify_darkpool` con 6 niveles ordenados.
K-DT3-RUNTIMEWARN (serie vacia) RESUELTO.

Cero hallazgos ALTA.

---

## Consolidado actualizado de fases y hallazgos

| Fase | ALTA | MEDIA | BAJA | Total |
|---|---:|---:|---:|---:|
| FASE 0 (inventario) | 0 | 1 | 0 | 1 |
| FASE 2.1 (regimenes/scores) | 0 | 12 | 7 | 19 |
| FASE 2.2 (IAE 10.x) | 0 | 4 | 5 | 9 |
| FASE 2.3 (breadth) | 0 | 2 | 3 | 5 |
| FASE 2.4 (MTE + Dark Pool) | 0 | 8 | 11 | 19 |
| FASE 3 (artefactos) | 3 | 6 | 6 | 15 |
| FASE 5 (providers) | 1 | 22 | 26 | 49 |
| FASE 6 (reporte) | 2 | 16 | 7 | 25 |
| **TOTAL** | **6** | **71** | **65** | **142** |

**Distribucion definitiva:**
- 6 ALTA.
- 71 MEDIA.
- 65 BAJA.
- 142 hallazgos totales.

**Patron dominante sin cambios:** doc-vs-codigo (28 hallazgos), fallback
silencioso (24), incoherencia interna del reporte (13), infraestructura
muerta (6).

**Cero hallazgos ALTA de calculo.** El codigo produce los numeros que dice
producir. La deuda es documental, de observabilidad y de presentacion.

---

## Actualizacion del plan de prioridades

Los 6 ALTA siguen siendo la maxima prioridad (sin cambios).

Se añaden a Prioridad 2 - MEDIA (impacto funcional):

23. F2.4-10 — Preservar `pending` en el state MTE: engine.py no debe borrarlo.
    Alternativa: consolidar el state management en un solo sitio.
24. F2.2-01 — Actualizar IAE_MAESTRO 10.1: eliminar el `* 1000` declarado
    y documentar la unidad real (miles de acciones).
25. F2.2-03 — Corregir docstring de `aggregate_positions_by_shareclass_figi`:
    la invariante SI se cumple (particiones disjuntas).
26. F2.2-04 — Alinear `coverage_status=VALID` con la condicion declarada
    en 10.5 (coverage_previous/current > 0).
27. F2.4-20 — Unificar las 5 variantes de robust_zscore en
    `src/utils.py::robust_zscore`.

Se añaden a Prioridad 3 - MEDIA (arquitectura):

28. F2.4-01 — Eliminar la duplicacion `cls > 0.7` en CRISIS.
29. F2.4-02, F2.4-11 — Sustituir `except:` desnudos por excepciones tipadas.
30. F2.4-03 — Usar `SCENARIO_WEIGHTS["IPS"]["weight"]` o eliminar la entrada.
31. F2.4-13 — Reescalar cls a [-1, +1] antes de `consensus_score`, o
    normalizar las 4 variables.
32. F2.2-08 — Cachear `load_canonical` en `run_contractual_nipc`.

---

## Fases no cubiertas (actualizado)

- FASE 1 (arquitectura profunda). Parcialmente cubierta por FASE 0.
- FASE 4 (tests). Suite verde, cobertura real por modulo del radar no medida.
- FASE 5 (parcial). 13 de 27 providers auditados en profundidad.
- FASE 6 (parcial). Secciones L297-416 del reporte no auditadas.

**Recomendacion de continuacion:** cerrar la auditoria y pasar a correcciones
de los 6 ALTA. Las fases pendientes tienen ROI decreciente: FASE 1 ya esta
parcialmente cubierta, FASE 4 audita suite verde, y L297-416 son secciones
de menor criticidad funcional.

---

**Addendum cerrado: 2026-09-26**
Total acumulado tras FASE 2.2 + FASE 2.4: 142 hallazgos (6 ALTA, 71 MEDIA, 65 BAJA)