# 68 - Respuesta al dictamen de auditoria externa 2026-10-25

**Fecha:** 2026-10-25 (post-cierre)
**Dictamen referenciado:** commit declarado `e491906c`, 2026-10-25.
**Estado de la respuesta:** parcial. 6 de 9 hallazgos cerrados; 3 abiertos.

## Veredicto de la respuesta

El dictamen se acepta en lo esencial. Los siguientes hallazgos se han
resuelto end-to-end con codigo + tests + run real verificado:

- H-03 (FORMING invalida si rompe rango): corregido, historico 1086
  fechas regenerado post-fix. 52.3% de filas cambian solo en FORMING y
  RANGE; las fases confirmadas no cambian.
- H-06 (WLS stability = tanh(median/MAD) confusion): corregido, ahora
  usa `wyckoff_stability` (MAD-based). 43% signos opuestos -> misma
  metrica.
- PIT holdings (composicion estatica aplicada retroactivamente):
  resuelto con snapshot append-only diario. En 6-12 meses habra
  historia PIT real.

Los siguientes hallazgos tienen accion parcial o futura y quedan como
items abiertos:

- H-01 (periodo 10 anos vs 4.2): el render ya declara el periodo
  correcto (2022-08-11 -> hoy, ~4.2 anos, 1086 sesiones). No hay
  documento `67_structure_positive_investigacion` publicado; el
  analisis se hizo en chat, no commiteado. La correccion aplicada
  esta en el render y en doc 66.
- H-02 (SOW confirma con 3): el pseudocodigo del informe era
  abreviado. El codigo real itera `range(cand_pos, max_pos + 1)`,
  no dos posiciones. Tests `test_h02_sow_*` (3 en total) verifican
  los tres cierres minimos, los dos cierres con recuperacion, y
  cero cierres. No es bug del codigo.
- H-04 (respaldo estadistico de ratio_pos): implementado como metrica
  continua con N est. Faltan intervalos de confianza y criterios de
  inclusion por estado. Item `DICT3-H04-INTERVALS` abierto.
- H-05 (fillna Close/0): contrato v1.8, protegido por test de
  regresion. El dictamen razona correctamente que 0/2512 filas no
  prueba seguridad futura. RESUELTO 2026-10-26: columna `bar_flags`
  anadida en `build_ticker_df` (commit 053f1a7a). Item
  `DICT3-H05-FILLNA-FUTURE` FIXED. Ver seccion H-05 ampliada.
- Top-20 cobertura (XLI 57.2%, SPY 49.9%): sin ampliar. Item
  `DICT3-TOP20-COBERTURA` abierto.
- Re-acumulacion de continuacion: frente nuevo. Item
  `DICT3-REACCUMULATION` abierto.

## Detalle por hallazgo

### H-01. Periodo de analisis

El informe original decia "10 anos" por confusion con el rango del
parquet `market_data` (2021-10-04 -> 2026-10-23). El historico Wyckoff
regenerado (post-fix H-03) es 2022-08-11 -> 2026-10-23: 1086 sesiones,
aproximadamente 4.2 anos. Doc 66 contiene la cifra correcta. El render
del reporte (commit `1589e079`) declara el periodo correcto en nota al
pie. Doc 67 no existe como fichero (el analisis fue en chat); esta
respuesta lo sustituye.

### H-02. Logica de confirmacion SOW

El pseudocodigo del informe de investigacion (§4.6) iteraba sobre
`[candidate_pos, candidate_pos + SOW_TIMEOUT_SESSIONS]`, lo cual no
permite alcanzar `SOW_CONFIRM_CLOSES = 3`. El codigo real
(`wyckoff_distribution.py::detect_sow_in_range`) itera sobre
`range(cand_pos, max_pos + 1)`, cubriendo toda la ventana de 10
sesiones. Los tests `test_h02_sow_confirma_con_tres_cierres_bajo_soporte`,
`test_h02_sow_no_confirma_con_dos_cierres_y_recuperacion` y
`test_h02_sow_no_confirma_sin_ruptura` (en
`tests/test_wyckoff_audit_2026_10_25.py`) verifican el contrato.

### H-03. FORMING invalida si rompe rango

Corregido en `wyckoff_orchestrator.py::_dist_derived_phase` y
`_acc_derived_phase`. Ahora `FORMING` solo se mantiene si el precio
sigue dentro del rango +- tol. Si rompe por cualquier lado, retorna
`RANGE`. Historico regenerado: 6244/11946 filas (52.3%) cambian
solo en `count_accumulation_forming`, `count_distribution_forming`,
`count_range` y `count_conflict`. Fases confirmadas (ACC, MARKUP,
DIST, MARKDOWN): 0 cambios. Tests en
`tests/test_wyckoff_audit_2026_10_25.py::test_h03_*`.

### H-04. respaldo estadistico de ratio_pos

Publicado como `structure_ratio` continuo en
`sector_regime_matrix.csv` (Bloque 2), con `n_estructuras` para
respaldar. Eliminada la version binaria (`structure_positive`). El
dictamen pide ademas: intervalos de confianza por cohorte anual,
tamanos muestrales, criterios de inclusion. Item
`DICT3-H04-INTERVALS` abierto con alcance.

### H-05. fillna Close/0 en build_ticker_df

Politica actual: `build_ticker_df` rellena Open/High/Low con Close y
Volume con 0. Es contrato v1.8, protegido por
`test_regresion_i2_nan_internos_no_degenera` (bug K-INDEX-RANGE-01).
Medicion sobre parquets: 0/2512 filas con H=L=C, 0 filas Volume=0.
El dictamen razona que esto no prueba seguridad futura. Correcto: el
riesgo residual no es 0 si en el futuro aparece una barra con datos
incompletos reales. Item `DICT3-H05-FILLNA-FUTURE` abierto para
distinguir barra-sin-negociacion de barra-incompleta (indicador de
calidad por fila) sin romper el contrato v1.8.

**Resolucion (2026-10-26).** Anadida columna `bar_flags` en el df
devuelto por `build_ticker_df`, con codigos separados por `;` en orden
fijo:

- `MISSING_OHLC`: alguno de Open/High/Low era NaN pre-fill.
- `ZERO_VOLUME`: Volume era NaN o 0 pre-fill.
- `NO_RANGE`: High == Low post-fill (TR=0).
- `OK`: ninguno de los anteriores.

Los detectores (`detect_spring`, `detect_sow`, `detect_sow_v2`) y el
scoring (`wyckoff_score`) NO leen `bar_flags`. Los fillna del contrato
v1.8 se mantienen intactos. Los CSVs persistidos no llevan
`bar_flags`: es una columna de auditoria en memoria, no un cambio de
schema.

Evidencia: commit 053f1a7a, 7 tests en `tests/test_h05_bar_flags.py`,
41 tests de contrato v1.8 sin regresion, run real 2026-10-26 con
EXIT=0 y 6 CSVs wyckoff_holdings byte-a-byte identicos pre/post.

### H-06. WLS stability

Corregido. `stock_metrics.py` ahora usa
`wyckoff_stability(combined)` = `1 - 2*tanh(MAD(combined)/K)`, no
`tanh(score_median/(score_mad+eps))`. La version antigua media
conviccion direccional (arrastraba t_norm, corr=+0.49 vs +0.16),
saturaba 36% de tickers en +-1.0 y divergia 43% en signo de la nueva
metrica. Tests en `tests/test_h06_stability_fix.py`.

### PIT holdings

Snapshot append-only diario (`holdings_snapshots.csv`), escrito en
daily_run + 3 scripts de update. Filtro por hash SHA256 de tickers
ordenados; solo se anade fila si la composicion cambia. En 6-12 meses
habra historia PIT para regeneraciones futuras. Los historicos previos
(2022-2026) fueron regenerados con la composicion 2026Q4 y son
susceptibles de sesgo de supervivencia: no se usan para analisis
estadisticos, solo descriptivos.

### Top-20 cobertura

Sin ampliar. Cobertura actual: XLI 57.2%, SPY 49.9%, IWM excluido
(6.2%). Item `DICT3-TOP20-COBERTURA` abierto. Requiere:
universo dinamico por grupo (80-90% peso objetivo), minimo de
componentes, descarga adicional, regeneracion del historico.

### Re-acumulacion

Frente no implementado. Detector actual solo detecta reversion
(Wyckoff clasico: SC -> AR -> Range -> SOS; BC -> AR -> Range -> SOW).
Re-acumulacion de continuacion (rango en tendencia alcista sana que se
rompe al alza) no se detecta. Item `DICT3-REACCUMULATION` abierto.

## Pruebas que se aportan como gate

- `tests/test_wyckoff_audit_2026_10_25.py`: 9 tests (H-02 SOW, H-03
  FORMING).
- `tests/test_h06_stability_fix.py`: 3 tests (stability = wyckoff
  stability, no saturacion, comparativo vs version antigua).
- `tests/test_holdings_snapshot.py`: 8 tests (hash determinista, skip
  por mismo hash, append por cambio).
- `tests/test_update_scripts_snapshot_call.py`: 6 tests (3 scripts x
  2, AST).
- `tests/test_xetra_provider_regression.py`: 10 tests (bug Xetra
  `date has no attr date`).
- Coherencia snapshot vs agregado: `test_wyckoff_holdings_coherence.py`
  + `test_sector_wyckoff_distribution.py::test_coherence_snapshot_vs_csvs`.

## Pruebas que el dictamen pide y no se aportan aun

- Intervalos de confianza por cohorte anual para ratio_pos.
- Tratamiento explicito de observaciones solapadas y correlacion
  entre sectores.
- Evaluacion out-of-sample con protocolo congelado y holdout
  independiente.
- Valor incremental del WLS vs RS sola.

Estos se agrupan bajo `DICT3-H04-INTERVALS` y bajo la fase 6 del
refactor (descarga unificada) como prerequisito para cualquier prueba
estadistica adicional. Sin PIT completo y sin ampliacion de cobertura,
los resultados historicos no son concluyentes.

## Cambios estructurales ya en produccion

- Schema versioning en outputs (Fase 1).
- Sector_wyckoff_distribution y sector_regime_matrix en v2.
- Snapshot unico de holdings (`compute_wyckoff_snapshot`, Fase 4).
- `structure_ratio` continuo en Matriz de Regimen (Bloque 2).
- `stability` del WLS = wyckoff_stability.
- PIT snapshot append-only.
- Bug Xetra normalizado (`_as_date`).

## Lo que no se acepta del dictamen

Nada. Todas las objeciones son validas tecnicamente. Las mitigaciones
aportadas (contrato v1.8, medicion 0/2512, snapshot PIT) son
razonamientos parciales, no demostraciones futuras. Se aceptan como
WONT_FIX o IN_PROGRESS con alcance documentado.
