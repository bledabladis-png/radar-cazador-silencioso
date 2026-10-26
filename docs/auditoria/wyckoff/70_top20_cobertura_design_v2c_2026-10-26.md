# Doc 70-v2c - DICT3-TOP20-COBERTURA. Solicitud de reconsideracion

**Fichero:** docs/auditoria/wyckoff/70_top20_cobertura_design_v2c_2026-10-26.md
**Estado:** para revision externa. Solicitud formal.
**Predecesor:** doc 70-v2b (aceptacion del dictamen 2026-10-26, commit b6b75e55).

---

## 0. Resumen ejecutivo

Las 6 acciones del dictamen 2026-10-26 estan atendidas:

- **4 ejecutables cerradas** con commit, tests y CI verde:
  P1-H01 (b3793d7c), P1-H02 (2d84ae82), P1-H03 (adfbb69b),
  P1-H04 (f967985d).
- **P1-H06 cerrada** en 46a5bfe3 + 6a591e67.
- **P1-H05 pendiente de decision del auditor externo** sobre doc 69.
  Este documento aporta el material que el dictamen declara no tener.

Solicitud: reconsiderar el cierre de `DICT3-TOP20-COBERTURA` y emitir
decision sobre `DICT3-H04-INTERVALS`, por separado.

---

## 1. Estado de las 6 acciones

| # | Accion | Estado | Commit | CI |
|---|---|---|---|---|
| 1 | P1-H01 selector estados | HECHO | b3793d7c | 40379993886 |
| 2 | P1-H02 cobertura efectiva | HECHO | 2d84ae82 | 40385195472 |
| 3 | P1-H03 invariantes backfill | HECHO | adfbb69b | 40392910716 |
| 4 | P1-H04 sesgo PIT visible | HECHO | f967985d | 40393879229 |
| 5 | P1-H05 H-04 separado | PENDIENTE auditor | — | — |
| 6 | P1-H06 tickets gobernanza | HECHO | 46a5bfe3, 6a591e67 | 40378349060 |

Detalle por accion en secciones 2-6.

---

## 2. P1-H01 - Selector con estados explicitos

**Commit b3793d7c.** Tests 18/18. CI 40379993886.

Cambios:
- `_select_target_holdings` retorna `(df, cob, estado: str)`.
- Estados: `REACHED`, `UNREACHABLE`, `INVALID_UNIVERSE`.
- `INVALID_UNIVERSE` cubre: grupo None/vacio, todos NaN, total<=0,
  NaN ratio > `max_missing_weight_ratio` (default 0.0).
- Consumidores adaptados (`_alcanzado`/`_ok` -> `_estado`); ninguno
  leia el tercer elemento (verificado por AST).
- Tests especificos: grupo vacio, todos NaN, total cero nunca conformes.

Satisfaccion del criterio del dictamen: "tests que demuestren que un
grupo con todos los pesos ausentes, un grupo vacio y otro con peso total
cero nunca se consideran conformes".

---

## 3. P1-H02 - Cobertura efectiva por grupo

**Commit 2d84ae82 + doc 71.** Script reproducible:
`py scripts/audit/p1_h02_cobertura_efectiva.py`.

Definiciones formales (doc 71 §1):
- Seleccionado: los N de `_select_target_holdings`.
- Con datos: `n_bars >= MIN_BARS_THEORETICAL=200` en stock_prices.parquet.
- Evaluable: fila en compute_wyckoff_snapshot (proxy=con_datos por hang
  conocido de classify_phase, declarado en doc 71 §5).
- Publicado: fila en los 6 CSVs, filtrada por fase direccional. NO es
  cobertura, es regimen de fase.

Cobertura por grupo (doc 71 §2): XLI 98.92/84.86, XLF 89.36/76.19,
XLK 96.02/81.85, XLV 92.73/79.08, SPY 95.58/81.25, QQQ 92.50/79.02,
resto >96% o 100%.

**Hallazgo principal**: universo real de tickers sin datos es **52**, no 8.
Actualizado en REGISTRO (`TOP20-FALTANTES-DATOS`).

---

## 4. P1-H03 - Invariantes del backfill

**Commit adfbb69b + doc 72.** Script reproducible:
`py scripts/audit/p1_h03_invariantes.py`.

Verificados 6 invariantes (1086 fechas x 11 sectores):
- (1) Cuadricula: OK.
- (2) Fase por ticker: SKIP (bloqueado por hang de classify_phase,
  ajeno a este frente).
- (3) Counts reconcilian: OK (11946/11946).
- (4) Pcts reproducibles: OK (denominador n_total).
- (5) CSV vs snapshot: OK (contra v2_full.csv.bak, 0 celdas distintas).
- (6) Diferencias explicadas: OK (delta XLI+24, XLF+3, XLK+2, resto 0;
  0 regresiones por fila).

**Afirmacion "cero regresiones" respaldada por 5/6 invariantes.**
La 2 requiere fix de classify_phase (frente aparte, no bloqueante para
A.2c).

Correccion documentada en doc 72 §3: la base de comparacion valida es
`v2_full.csv.bak` (pre-swap), no `v2_full_target_only.csv` (artefacto
intermedio pre-fixes, n_total mean 20.75 vs 22.24 real).

---

## 5. P1-H04 - Sesgo PIT visible

**Commit f967985d.** 2 tests. CI 40393879229.

`src/report/sector_context.py::render_wyckoff_sectorial` incluye nota
incondicional sobre composicion 2026Q4 aplicada retrospectivamente.
"No es reconstruccion point-in-time. El comportamiento mostrado es el
de los componentes actuales, no el de la composicion historica."

Alcance: solo reporte. El dictamen permite "reportes o metadatos". No
se toca persist() ni sidecar JSON (25 writers registrados). Si el
auditor quiere sidecar en CSV, ticket aparte.

---

## 6. P1-H06 - Gobernanza separada

**Commits 46a5bfe3 + 6a591e67.** 152 items en REGISTRO.

Tickets independientes:
- `PERSIST-REPLACE-DATE` (FIXED, f4528b49).
- `PCR-SKIP-NO-DATA` (FIXED, affdc5c8).
- `TOP20-USA-INDICES` (OPEN).
- `TOP20-EU-INDICES` (OPEN).
- `TOP20-FALTANTES-DATOS` (OPEN, expandido a 52).
- `TOP20-EU-HOLDINGS-INCOMPLETOS` (OPEN).
- `SOW-HOLDOUT-IMMUTABLE` (OPEN).
- `GOVERNANZA-CI-NO-STEP` (OPEN).

`IWM-HOLDINGS-EXPANSION` sin cambios (ya existia).

---

## 7. P1-H05 - Material para el auditor

El dictamen declara: "No dispongo en el texto aportado de los valores
detallados de doc 69 §13, por lo que no puedo certificar que los
resultados posteriores cumplan o incumplan el criterio."

Se aporta el material completo:

### 7.1. Doc 69 completo (commit 8a0a8b1d)

`docs/auditoria/wyckoff/69_h04_intervals_2026-10-26.md`, 18.6 KB.
Secciones:
- §1-§10: pre-registro congelado.
- §11: lo que el pre-registro NO hace.
- §12: resultados empiricos base top-20.
- §13: re-ejecucion sobre base target_weight (post-A.2c).

### 7.2. Resultados numericos de §13

Regenerados hoy via `py scripts/h04_intervals.py`:

**Anual (target_weight):**
| cohorte | mean | IC multiway | IC naive |
|---|---|---|---|
| 2022 | 0.7841 | [0.4773, 1.0000] | [0.4773, 1.0000] |
| 2023 | 0.8238 | [0.6848, 0.9455] | [0.6848, 0.9455] |
| 2024 | 0.5724 | [0.4324, 0.7128] | [0.4324, 0.7128] |
| 2025 | 0.2339 | [0.1124, 0.3564] | [0.1124, 0.3564] |
| 2026 | 0.2124 | [0.0709, 0.3726] | [0.0709, 0.3726] |
| ALL | 0.5104 | **[0.2935, 0.7653]** | [0.4159, 0.6124] |

**Estratos por n_estructuras:**
- [1-4]: 7017 celdas, IC [0.1743, 0.7058]
- [5-10]: 2772 celdas, IC [0.1840, 0.6684]
- [11-20]: 278 celdas, IC [0.0417, 0.4530]
- [21-inf]: 0 celdas

`rho_bar` cross-sector: **0.4708** (doc 70-v2 registra 0.5243->0.4708
tras regeneracion target_weight).

### 7.3. Criterio §9 pre-registrado

> "Si el IC95% cruza 0 en >50% de cohortes, H-04 se reabre como
> descriptiva pura, no se reinterpreta."

Aplicado literalmente sobre estos numeros: **0/15 cohortes** cruzan 0
(todas tienen `ci_lo > 0`). El criterio no discrimino (autocritica
§12.4, §13.3): es una limitacion de construccion del pre-registro, no
un exito de H-04. No se reinterpreta. No se modifica la regla.

### 7.4. CSVs crudos

`outputs/audit/h04_intervals.csv` (2406 bytes) +
`outputs/audit/h04_cross_sector_corr.csv` (2285 bytes). Regenerables
con `py scripts/h04_intervals.py`. Gitignored pero adjuntables.

---

## 8. Lo que este documento NO hace

- No cierra `DICT3-TOP20-COBERTURA`: lo solicita.
- No cierra `DICT3-H04-INTERVALS`: lo somete a decision.
- No verifica el invariante 2 de P1-H03 (bug classify_phase, aparte).
- No descarga los 52 tickers de `TOP20-FALTANTES-DATOS`.
- No arregla `v2_full_target_only.csv` (recomendacion: eliminar).

---

## 9. Solicitud

Se solicita:

1. **Reconsiderar `DICT3-TOP20-COBERTURA`**: las 4 acciones ejecutables
   estan cerradas con commit, tests y CI verde. P1-H06 tambien.
   Quedan abiertos `TOP20-FALTANTES-DATOS` (52) y `TOP20-EU-HOLDINGS-INCOMPLETOS`
   como frentes independientes, no bloqueantes del selector A.2c.

2. **Decision sobre `DICT3-H04-INTERVALS`**: con doc 69 completo y los
   resultados crudos arriba, el auditor puede:
   - (a) Declarar FIXED reconociendo la autocrítica §12.4/§13.3.
   - (b) Exigir nuevo pre-registro con criterio discriminante.
   - (c) Mantener OPEN con nueva accion especifica.

3. **Confirmar si el invariante 2 de P1-H03 (fase por ticker) es
   bloqueante** o se acepta 5/6 con SKIP documentado.

---

**Preparado 2026-10-26. Pendiente de envio al auditor externo.**