# Doc 70-v2b - DICT3-TOP20-COBERTURA. Aceptacion del dictamen 2026-10-26

**Fichero:** docs/auditoria/wyckoff/70_top20_cobertura_design_v2b_2026-10-26.md
**Estado:** para revision externa.
**Predecesor:** doc 70-v2 (commiteado en `ebdeac02`), que respondio los 7 hallazgos P0-H01..P0-H07 del dictamen anterior.
**Motivo de este documento:** respuesta formal al dictamen 2026-10-26, recibido tras el envio de v2. Veredicto recibido: APROBACION CONDICIONADA de A.2c, NO cierre global de `DICT3-TOP20-COBERTURA`, con 6 acciones pendientes.

---

## 0. Resumen ejecutivo

A.2c (selector por cobertura de peso, solo sectores USA) queda aprobada condicionada. El frente `DICT3-TOP20-COBERTURA` sigue OPEN hasta completar las 6 acciones de la seccion 2. El sistema es productivo con `target_weight` desde `e7f27cd0`; el estado OPEN es deuda documental y de verificacion, no deuda productiva.

Correcciones respecto a la documentacion previa, detectadas al preparar esta respuesta:

- XLU/XLV/XLY no cumplian nominalmente el 85% antes de A.2c (83.1% / 78.9% / 84.5% con top-20). No cambian materialmente porque sus tickers extra no estan en parquet, no porque ya cumplieran. Doc 70-v2 seccion P0-H03 ya lo refleja en tabla; se formaliza aqui.
- SHA del fix PCR es `affdc5c8` (no `9338dcd4`, typo en handoff sesion 11).
- Commit `f4528b49` dice "(P0-H06)" en el mensaje; el dictamen lo numera P1-H06. Misma entidad.
- `TOP20-FALTANTES-DATOS` no requiere descarga de los 8 tickers desde cero: 7 de 8 estan en `stock_prices_reserved_cohort.parquet` y `stock_prices_reserved_r3000.parquet`, holdout de validacion SOW (consumido por `scripts/purged_kfold_sow_c1_*.py`). No se promueven. Requiere parquet nuevo de produccion. MRNA no esta en ningun parquet.

---

## 1. Alcance aceptado

A.2c aplica solo a los 11 sectores USA. Confirmado. Los frentes separados:

| Frente | Estado | Ticket |
|---|---|---|
| Sectores USA (11) | A.2c aplicado, aprobacion condicionada | `DICT3-TOP20-COBERTURA` |
| Indices USA (SPY/QQQ/DIA) | nlargest, cobertura 49.9/66.8/85.0 | `TOP20-USA-INDICES` (nuevo) |
| Indices EU (FEZ/DAXEX/ISF.L/LYXI) | nlargest, cobertura 66.3/78.4/60.8/93.2 | `TOP20-EU-INDICES` (nuevo) |
| IWM | nlargest, 6.2% | `IWM-HOLDINGS-EXPANSION` (existente) |
| Tickers seleccionados sin datos | 8 tickers, 7 en holdout SOW | `TOP20-FALTANTES-DATOS` (nuevo) |
| Holdings EU incompletos | peso_total_csv < 100% | `TOP20-EU-HOLDINGS-INCOMPLETOS` (nuevo) |
| H-04 intervals | OPEN, no se mezcla con A.2c | `DICT3-H04-INTERVALS` (existente) |

Los 6 tickets nuevos o promovidos se han registrado en `docs/gobernanza/REGISTRO.yaml` en commit `46a5bfe3`.

---

## 2. Aceptacion de las 6 acciones del dictamen

### P1-H01 - Selector: estados explicitos

Aceptado. Estado actual: `_select_target_holdings` retorna `(df, cobertura_frac, alcanzado: bool)`. Los tres casos "sin universo" (grupo None o vacio, todos los weight NaN, total<=0) devuelven `alcanzado=True`. Eso confunde "no hay universo" con "objetivo cumplido".

Plan:
- Sustituir bool por str con tres estados: REACHED, UNREACHABLE, INVALID_UNIVERSE.
- REACHED: cumsum alcanza target_abs dentro de [min_tickers, max_tickers].
- UNREACHABLE: cumsum nunca alcanza target_abs antes de max_tickers.
- INVALID_UNIVERSE: grupo None o vacio, todos weight NaN, o total<=0.
- Tests nuevos para los tres estados por via independiente.
- Adaptar los dos consumidores actuales (`scripts/regenerate_wyckoff_history.py:106`, `indicators/wyckoff_holdings_report.py:216`), que hoy descartan el tercer elemento. Pasaran a leer el estado explicitamente.

Nota de impacto: ningun consumidor lee el booleano hoy (verificado por AST). El cambio es de contrato, no de comportamiento productivo. No hay bug observable en produccion.

### P1-H03 - Backfill: invariantes por fila

Aceptado. Sustituir "0 regresiones porque n_total no baja" por verificacion explicita de 6 invariantes sobre las 1086 fechas x 11 sectores:

- (a) 1086 fechas x 11 sectores, sin duplicados.
- (b) fase y eventos identicos para ticker comun (date, sector, ticker).
- (c) count_* reconcilia con n_total.
- (d) pct_* reproducible desde count y denominador.
- (e) CSV coincide con snapshot.
- (f) diferencias explicadas por expansion y disponibilidad.

Coste estimado: 1 sesion. Script de verificacion nuevo, no modifica produccion.

### P1-H02 - Cobertura efectiva ponderada

Aceptado. Tabla por grupo con columnas: seleccionado, con datos, evaluable, publicado. Definir formalmente evaluable (barras >= MIN_BARS_THEORETICAL) vs publicado (fila efectivamente emitida en CSV). Mantener abiertos `TOP20-FALTANTES-DATOS` y `TOP20-EU-HOLDINGS-INCOMPLETOS`.

### P1-H04 - Sesgo PIT visible en artefactos

Aceptado. La nota "composicion 2026Q4 aplicada retrospectivamente" debe ir en los artefactos consumidos: cabecera del CSV `sector_wyckoff_distribution.csv`, metadato del manifest de `sector_regime_matrix`, o nota en el reporte. No solo en doc de auditoria.

### P1-H06 - Gobernanza separada

Aceptado. Tickets independientes en `REGISTRO.yaml` (commit `46a5bfe3`):

- `PERSIST-REPLACE-DATE` - FIXED, commit `f4528b49`.
- `PCR-SKIP-NO-DATA` - FIXED, commit `affdc5c8`.
- `TOP20-USA-INDICES` - OPEN.
- `TOP20-EU-INDICES` - OPEN.
- `TOP20-FALTANTES-DATOS` - OPEN.
- `TOP20-EU-HOLDINGS-INCOMPLETOS` - OPEN.
- `IWM-HOLDINGS-EXPANSION` - ya existia, OPEN, sin cambios.

### P1-H05 - H-04 separado

Aceptado. `DICT3-H04-INTERVALS` sigue OPEN. No se cierra con A.2c. Decision FIXED/OPEN pendiente de auditor externo sobre doc 69 completo.

---

## 3. Correccion documental

La tabla original de doc 70-v2 incluye "XLU 83.1%, XLV 78.9%, XLY 84.5%" en el resumen y "resto no cambia porque ya cumplia" en el cuerpo. La segunda afirmacion es falsa. Correccion:

> XLU/XLV/XLY no alcanzaban el 85% nominal con top-20. A.2c los selecciona por encima de 20 (21, 26, 21 respectivamente), pero sus tickers extra (CNP en XLU; CI, MRNA, BSX, HCA, COR, CAH en XLV; NKE en XLY) no estan en `stock_prices.parquet`. 7 de 8 estan en parquet reservado SOW; MRNA no esta en ninguno. La cobertura efectiva no sube hasta que se descarguen a parquet de produccion.

---

## 4. Lo que este documento NO hace

- No cierra `DICT3-TOP20-COBERTURA`.
- No cierra `DICT3-H04-INTERVALS`.
- No ejecuta las 6 acciones: las acepta y planifica.
- No toca codigo de produccion.
- No toca parquets reservados.

---

## 5. Orden de ejecucion

Las 6 acciones se ejecutaran en commits separados, en orden:

1. REGISTRO (`46a5bfe3`, commit 1a) + doc 70-v2b (este, commit 1b).
2. P1-H01 - selector con estados + tests (commit 2).
3. P1-H06 - tickets de gobernanza (ya cubierto en commit 1a).
4. P1-H04 - sesgo PIT en artefactos (commit 3).
5. P1-H02 - cobertura efectiva (commit 4 doc + script).
6. P1-H03 - invariantes backfill (commit 5 codigo + doc).
7. P1-H05 - separado, depende del auditor.

---

**Preparado 2026-10-26 para revision externa. Pendiente de firma del auditor.**