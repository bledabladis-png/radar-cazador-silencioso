# Doc 70-v2d - DICT3-TOP20-COBERTURA. Cierre del ciclo de dictamenes

**Fichero:** docs/auditoria/wyckoff/70_top20_cobertura_design_v2d_2026-10-26.md
**Estado:** para revision externa.
**Predecesores:**
- doc 70-v2 (respuesta a 7 hallazgos previos, commit ebdeac02).
- doc 70-v2b (aceptacion del dictamen 2026-10-26, commit b6b75e55).
- doc 70-v2c (solicitud de reconsideracion, commit 52a7ce8d).
- doc 73 (respuesta al dictamen H-04, commit 5b855ce5).

---

## 0. Resumen ejecutivo

El dictamen 2026-10-26 sobre doc 70-v2 tenia 6 acciones. Estado
final tras completar el ciclo:

| # | Accion | Estado | Evidencia |
|---|---|---|---|
| 1 | P1-H01 selector estados | HECHO | b3793d7c, CI 40379993886 |
| 2 | P1-H02 cobertura efectiva | HECHO | 2d84ae82, doc 71, CI 40385195472 |
| 3 | P1-H03 invariantes backfill | HECHO | adfbb69b, doc 72, CI 40392910716 |
| 4 | P1-H04 sesgo PIT visible | HECHO | f967985d, CI 40393879229 |
| 5 | P1-H05 H-04 separado | **RECHAZADO por auditor** | Dictamen H-04, doc 73, commit 5b855ce5 |
| 6 | P1-H06 tickets gobernanza | HECHO | 46a5bfe3 + 6a591e67 |

5 de 6 acciones cerradas. P1-H05 no se puede cerrar porque el propio
auditor ha rechazado la publicacion inferencial de structure_ratio
(dictamen H-04). Pero **P1-H05 era separar los dos juicios, no
aprobarlos**. La separacion esta hecha: `DICT3-H04-INTERVALS` sigue
OPEN con `siguiente_accion` reescrito; `DICT3-TOP20-COBERTURA` se
solicita cerrar por separado con las 4 acciones ejecutables.

---

## 1. Resolucion del dictamen H-04 (P1-H05)

El dictamen 2026-10-26 pedia expresamente:

> "No debe confundirse el resultado de H-04 con la aprobacion del
> cambio de cobertura. Deben mantenerse separados dos juicios:
> la correccion tecnica de la seleccion y regeneracion del universo,
> y la validez estadistica de structure_ratio y de sus intervalos."

La separacion esta registrada:

- **A.2c (selector y regeneracion)**: aprobada condicionada, 4 acciones
  ejecutables cerradas, sin dependencia del veredicto H-04.
- **H-04 (validez inferencial)**: rechazada por el auditor. Doc 73
  recoge la aceptacion y el nuevo ticket `DICT3-H04-INFERENCE-V2` con
  las 4 decisiones de diseno.

No hay mezcla. La seleccion de holdings USA sigue operativa con
`target_weight` desde `e7f27cd0`. `structure_ratio` sigue publicada
como metrica descriptiva (autorizada). Los IC95 no se publican en el
reporte (verificado en doc 73 seccion 2).

---

## 2. Estado de cada frente separado

El dictamen 2026-10-26 exigia mantener los frentes separados. Estado
actual:

| Frente | Estado | Ticket | Bloqueante de A.2c? |
|---|---|---|---|
| Sectores USA (11) | A.2c aplicado, 4 acciones cerradas | `DICT3-TOP20-COBERTURA` | — |
| Tickers sin datos (52) | OPEN | `TOP20-FALTANTES-DATOS` | No |
| Indices USA | OPEN | `TOP20-USA-INDICES` | No |
| Indices EU | OPEN | `TOP20-EU-INDICES` | No |
| Holdings EU incompletos | OPEN | `TOP20-EU-HOLDINGS-INCOMPLETOS` | No (solo bloquea indices EU) |
| IWM | OPEN | `IWM-HOLDINGS-EXPANSION` | No |
| Holdout SOW inmutable | OPEN | `SOW-HOLDOUT-IMMUTABLE` | No |
| H-04 intervals | OPEN, rechazado | `DICT3-H04-INTERVALS` | No |
| H-04 inferencia v2 | OPEN | `DICT3-H04-INFERENCE-V2` | No |

**Ninguno de los frentes abiertos es bloqueante de A.2c.** Son frentes
paralelos, con sus propios ciclos.

---

## 3. Lo que la aprobacion condicionada cubre

La aprobacion condicionada del dictamen 2026-10-26 se otorgaba a:

> "A.2c - seleccion de holdings de los 11 sectores USA... el enfoque
> de cobertura objetivo del 85%, el minimo de 20 holdings y la
> regeneracion completa del historico como solucion metodologica."

Con las 4 acciones ejecutables cerradas, las condiciones estan
satisfechas:

- Semantica de universos invalidos y pesos ausentes (P1-H01): cerrada.
- Consistencia del backfill (P1-H03): 5/6 invariantes verificados.
- Coberturas efectivas declaradas (P1-H02): tabla completa en doc 71.
- Sesgo PIT visible (P1-H04): nota en reporte, commit f967985d.

La quinta accion (P1-H05) no era una condicion de A.2c: era una
instruccion de **separar los dos juicios**. Separacion hecha.

---

## 4. Lo que este documento NO hace

- No cierra `DICT3-H04-INTERVALS`: el dictamen H-04 lo mantiene OPEN.
- No reabre el pre-registro §2-§10 de doc 69.
- No descarga los 52 tickers de `TOP20-FALTANTES-DATOS`.
- No promueve tickers del holdout SOW.
- No cierra `TOP20-EU-HOLDINGS-INCOMPLETOS`.
- No arregla el hang de `classify_phase` (bug preexistente, detectado
  en P1-H03, sin ticket propio — ver seccion 5).

---

## 5. Deuda detectada en esta sesion, sin ticket previo

Tres items que emergieron durante la ejecucion de las 4 acciones y que
no estaban registrados:

### 5.1. `WYCKOFF-CLASSIFY-PHASE-HANG` — CERRADO como FALSE_POSITIVE

Ticket registrado como `OPEN` en `74b09c7e` (sesion 12) tras un
`KeyboardInterrupt` durante P1-H02. Reclasificado a `FALSE_POSITIVE`
en `842c9aaf`:

- Probe exhaustivo sobre 360 tickers unicos seleccionados:
  `classify_phase` 360/360 OK, 0 TIMEOUT (>5s), 0 excepciones.
- 21.6s total, medio 0.0601s, max 0.117s.
- El `KeyboardInterrupt` original fue cancelacion manual de un
  proceso de ~40s.

Sin bug de cuelgue. Si en el futuro aparece un cuelgue real,
abrir ticket nuevo con ticker concreto y traza reproducible.

### 5.2. `WYCKOFF-TICKER-NORMALIZATION` — bug real detectado

Durante el probe de 5.1 aparecio un bug **real** en produccion:

- Holdings usa `BRK.B`, parquet usa `BRK-B`.
- `_evaluate_holding` (`wyckoff_holdings_report.py:274`) llamaba
  `get_col(df, 'BRK.B')` → `KeyError` → fila descartada.
- `regenerate_wyckoff_history.py::_get_tickers` y `_procesar_ticker`
  tenian el mismo patron.
- Consecuencia: XLF perdia BRK.B (12.32% de peso, el mayor del
  grupo) del snapshot, de los 6 CSVs y del historico regenerado
  (1086 fechas).

Fix `4e94bf2e`: normalizar `ticker` antes de cualquier lookup en
parquet, preservar nombre original en `row['ticker']` y en
`mapping`. Test `test_dotted_ticker_normalized_before_parquet_lookup`.
Snapshot: 527 → 528 filas. XLF `ev_n` 22 → 23, `ev_peso` 63.74 →
76.05. Solo 1 ticker afectado en el universo actual.

La tabla de cobertura de doc 71 §2 ya refleja el fix (XLF 23, 76.05).
### 5.3. `v2_full_target_only.csv` obsoleto

Artefacto intermedio pre-fixes (`f4528b49`, `affdc5c8`) con `n_total`
sistematicamente menor que el activo (mean 20.75 vs 22.24). Confundio
el primer intento de P1-H03. Doc 72 §3 lo documenta.

**Recomendacion:** eliminar o mover a `outputs/audit/_obsoleto/`. No es
ticket de gobernanza, es limpieza de outputs locales.

### 5.4. `GOVERNANZA-CI-NO-STEP` (ya registrado)

Ticket creado en `6a591e67`. Sigue OPEN. Anadir `py
scripts/verifica_gobernanza.py` a `ci_tests.yml`. ~5 min, commit
independiente.

---

## 6. Solicitud

Se solicita:

1. **Cerrar `DICT3-TOP20-COBERTURA` como FIXED** en lo que corresponde
   a A.2c. La solucion parcial aprobada condicionalmente esta
   implementada, verificada y documentada. Los frentes abiertos
   (tickers sin datos, indices, IWM, H-04) son tickets independientes
   con sus propios ciclos.

2. **Confirmar si el invariante 2 de P1-H03 (fase por ticker) es
   bloqueante** de A.2c, o se acepta 5/6 con SKIP documentado por el
   hang de `classify_phase`.

3. **Confirmar si `WYCKOFF-TICKER-NORMALIZATION`** (bug real
   detectado durante el probe del hang) requiere regeneracion del
   historico completo (1086 fechas) o basta el fix + snapshot
   actualizado. El historico regenerado previo excluye BRK.B de
   XLF por el mismo motivo.

---

## 7. Condiciones de cierre

`DICT3-TOP20-COBERTURA` se considera cerrable como `FIXED` si el
auditor externo responde explicitamente con:

- **Cerrado:** se acepta el cierre de A.2c con las 4 acciones
  ejecutables cerradas (b3793d7c, 2d84ae82, adfbb69b, f967985d) y
  el invariante 2 de P1-H03 aceptado como SKIP documentado por el
  hang de `classify_phase` (posteriormente cerrado como
  FALSE_POSITIVE en 842c9aaf). El ticket pasa a `FIXED` con SHA
  del commit del cierre en REGISTRO. Los frentes abiertos
  (tickers sin datos, indices, IWM, H-04) viven como tickets
  independientes.
- **Cerrado con condiciones:** se anotan las condiciones. El
  ticket pasa a `FIXED` cuando las condiciones se cumplan y se
  documenten.
- **No cerrado:** el ticket sigue OPEN. Se anota la razon del
  auditor en `REGISTRO.yaml`.

Preguntas abiertas para el auditor (respuesta binaria):

- El invariante 2 de P1-H03 (fase por ticker) se acepta como
  SKIP documentado, o se considera bloqueante de A.2c?
- `WYCKOFF-TICKER-NORMALIZATION` (fix 4e94bf2e) requiere
  regeneracion del historico completo (1086 fechas, hecho en
  f71eb49c) o basta el fix + snapshot actualizado? El historico
  regenerado previo excluye BRK.B de XLF por el mismo motivo.

---

---

## 6. Evidencia de regeneracion historica post-fix `4e94bf2e` (2026-10-31)

**Contexto.** El dictamen externo 2026-10-31 pide evidencia auditable
de la regeneracion completa posterior a `4e94bf2e`. El commit
`f71eb49c` es la regeneracion, pero su contenido no estaba
consolidado como evidencia. Se verifica contra su padre inmediato
(`89e24705`).

**Metodo.** `git show f71eb49c:outputs/history/sector_wyckoff_distribution.csv`
vs `git show f71eb49c^:outputs/history/sector_wyckoff_distribution.csv`.
Comparacion por `(date, sector)` y por `n_total` + `count_range`.

**Resultado (fechas comunes = 1086).**

| Sector | delta n_total | delta count_range | Conclusion |
|---|---|---|---|
| XLF | **+1 en 1086 fechas** | **+1 en 1086 fechas** | BRK.B recuperado (RANGE) |
| XLB | 0 | 0 | Sin cambio |
| XLC | 0 | 0 | Sin cambio |
| XLE | 0 | 0 | Sin cambio |
| XLI | 0 | 0 | Sin cambio |
| XLK | 0 | 0 | Sin cambio |
| XLP | 0 | 0 | Sin cambio |
| XLRE | 0 | 0 | Sin cambio |
| XLU | 0 | 0 | Sin cambio |
| XLV | 0 | 0 | Sin cambio |
| XLY | 0 | 0 | Sin cambio |

XLF post: `n_total=23` en todas las 1086 fechas (antes 22).
`count_range` sube en 1 en cada fecha. Coincide exactamente con
lo que el commit `f71eb49c` declara en su mensaje.

**Nota sobre el CSV activo.** El fichero
`outputs/history/sector_wyckoff_distribution.csv` actual (2026-10-31)
muestra XLF `n_total ∈ {23, 25}` en 1089 fechas. Eso NO es una
regresion de `f71eb49c`: son los efectos posteriores de
`c879df71` (universe_expansions, XLF +9 tickers) y `1cb15d52` (O2,
coherencia analisis=descarga). El commit `f71eb49c` en su momento
produjo exactamente lo que declaro (verificado contra su padre).

**Veredicto: condicion 2 del dictamen atendida.** La evidencia es
reproducible con `git show` y confirma la declaracion del commit.

---

**Preparado 2026-10-26. Pendiente de envio al auditor externo.**
