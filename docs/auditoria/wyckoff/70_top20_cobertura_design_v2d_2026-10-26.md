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

### 5.1. `WYCKOFF-CLASSIFY-PHASE-HANG`

`classify_phase` cuelga con al menos un ticker del universo actual.
Traza: `_trend_component` en `indicators/wyckoff_v1.py:99` alinea Series
con indices incompatibles al sumar `ma_slow + 1e-9`. Bug preexistente,
no introducido por esta sesion.

Detectado al intentar calcular `compute_wyckoff_snapshot` en P1-H02.
Consecuencia: el invariante 2 de P1-H03 queda SKIP; la columna
`evaluable` del CSV de P1-H02 usa `con_datos` como proxy.

**No registrado aun.** Debe abrirse ticket propio con reproduccion
minima y fix candidato (`_trend_component` debe usar `.to_numpy()` o
`.values` en lugar de operar Series alineadas).

### 5.2. `v2_full_target_only.csv` obsoleto

Artefacto intermedio pre-fixes (`f4528b49`, `affdc5c8`) con `n_total`
sistematicamente menor que el activo (mean 20.75 vs 22.24). Confundio
el primer intento de P1-H03. Doc 72 §3 lo documenta.

**Recomendacion:** eliminar o mover a `outputs/audit/_obsoleto/`. No es
ticket de gobernanza, es limpieza de outputs locales.

### 5.3. `GOVERNANZA-CI-NO-STEP` (ya registrado)

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

3. **Confirmar si `WYCKOFF-CLASSIFY-PHASE-HANG` debe abrirse como
   ticket propio** en REGISTRO antes del cierre de sesion, o se
   documenta solo aqui.

---

**Preparado 2026-10-26. Pendiente de envio al auditor externo.**