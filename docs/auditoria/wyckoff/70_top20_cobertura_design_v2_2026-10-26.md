# Doc 70-v2 — DICT3-TOP20-COBERTURA. Respuesta al dictamen 2026-10-26

**Fichero:** docs/auditoria/wyckoff/70_top20_cobertura_design_v2_2026-10-26.md
**Estado:** para revisión externa.
**Predecesor:** doc 70 v1, NO APROBADO. Marcado en el fichero con nota explícita.

---

## 0. Resumen ejecutivo

**Cambio aplicado en producción (commit e7f27cd0):** el selector de holdings pasa de `nlargest(20)` a "acumular por peso hasta cubrir 85% del total del grupo, con mínimo 20". Solo en los 11 sectores USA. Índices USA y EU siguen con nlargest.

**Evidencia:**
- Selector: 15 tests deterministas + 4 tests de `selection_mode`.
- Regeneración: 1086 fechas × 11 sectores regeneradas. 0 regresiones verificadas por fila `(date, sector)`.
- XLI 20→44 tickers, XLF 19→22, XLK 20→22. Resto sin cambio material.
- H-03 re-verificado: 30/30 (idéntico, no depende de la base).
- H-04 re-ejecutado: doc 69 §13, criterios pre-registrados intactos.
- Bug persist detectado y corregido (huérfanas en subconjuntos filtrados).
- Bug PCR detectado y corregido (receipt ausente rompía el gate).

**Pendiente de revisión externa:**
- Doc 70-v2 (este).
- Cierre de `DICT3-TOP20-COBERTURA` como FIXED (o nueva acción).
- Deudas separadas: índices USA (SPY/QQQ/DIA), índices EU, IWM,
  `peso_total<100` en 3 ETFs EU, `TOP20-FALTANTES-DATOS` (tickers
  seleccionados sin datos en parquet).

**Dictamen recibido 2026-10-26 (post-envío v2):**

APROBACIÓN CONDICIONADA de A.2c. NO aprobación del cierre global
de `DICT3-TOP20-COBERTURA`. 6 acciones concretas pendientes (§4
de doc 70-v2b, a redactar en siguiente sesión). Esta versión v2
corresponde al estado antes de recibir ese dictamen.

---

## 1. Respuesta a los 7 hallazgos del dictamen

### P0-H01. Selector no seguro

**Aceptado.**

Contrato implementado en `_select_target_holdings` (`indicators/wyckoff_holdings_report.py`):

- Búsqueda explícita de posición sobre `cumsum` con bucle. `idxmax()` eliminado.
- Denominador: **peso total del grupo completo**, no el de la selección.
- `NaN` en `weight`: descartado antes del cálculo. No se imputa.
- Grupo vacío o `total<=0`: selección vacía, `alcanzado=True` (no hay objetivo que incumplir).
- Objetivo inalcanzable (`cumsum` nunca alcanza `target_abs`): devuelve hasta `max_tickers` con `alcanzado=False`. Nunca finge conformidad.
- Validación explícita de `target_weight ∈ (0,1]`, `min_tickers ≥ 1`, `max_tickers ≥ min_tickers`.

**Constantes:**
- `TARGET_WEIGHT_COVERAGE = 0.85` (objetivo operativo, dentro del 80-90% del dictamen).
- `MIN_TICKERS_PER_GROUP = TOP_N_HOLDINGS = 20`. Evita regresar grupos ya >85% con top-20.
- `MAX_TICKERS_PER_GROUP = 300`. Protección operativa. Si el máximo se alcanza antes del 85%, `alcanzado=False`.

**Tests (15, `test_select_target_holdings.py`):** cubren índice no numérico (bug `idxmax` original), `NaN` en weight, denominador correcto, inalcanzable por max, inalcanzable por peso insuficiente, `min_tickers` forzado, grupo concentrado, grupo plano, validaciones, determinismo.

**Lo que este selector NO resuelve:** la cobertura es sobre holdings **seleccionados**, no sobre holdings **evaluables**. Ver P0-H03.

### P0-H02. Universo de aplicación y criterios se contradicen

**Aceptado. Alcance redefinido.**

A.2c aplica solo a los 11 sectores USA. Índices USA (SPY, QQQ, DIA) y EU (FEZ, DAXEX, ISF.L, LYXI) siguen con nlargest.

**División explícita:**

| Frente | Estado | Cobertura actual | Cobertura objetivo |
|---|---|---|---|
| Sectores USA (11) | **A.2c aplicado** | XLI 57→86, XLF 73→85+, XLK 80→85+, resto sin cambio | 85% |
| Índices USA (SPY/QQQ/DIA) | **Abierto, ticket separado** | 49.9%, 66.8%, 85% | 85% (SPY/QQQ requieren) |
| Índices EU (FEZ/DAXEX/ISF.L/LYXI) | **Abierto, ticket separado** | 66.3%, 78.4%, 60.8%, 93.2% | 85% (pendiente) |
| IWM | **Abierto, ticket separado** | 6.2% (excluido por diseño) | 50-70% |

**REGISTRO.yaml** refleja esta división en el campo `siguiente_accion` de `DICT3-TOP20-COBERTURA`. No se declara resuelto el alcance global.

### P0-H03. Cobertura de holdings ≠ cobertura efectiva

**Aceptado. Reconocido como problema no resuelto.**

| Grupo | Seleccionados | En parquet | Evaluables | Publicados | Gap sel-eval |
|---|---|---|---|---|---|
| XLI | 44 | 44 | 44 | 44 | 0 |
| XLF | 22 | 22 | 22 | 22 | 0 |
| XLK | 22 | 22 | 22 | 22 | 0 |
| XLV | 26 (target) | 20 | 20 | 20 | 6 seleccionados sin datos |
| XLU | 21 (target) | 20 | 20 | 20 | 1 seleccionado sin datos |
| XLY | 21 (target) | 20 | 20 | 20 | 1 seleccionado sin datos |
| FEZ | 16 | 16 | 6 | 16 | 10 con <460 barras |

Los tickers seleccionados pero ausentes del parquet:
- XLV: CI, MRNA, BSX, HCA, COR, CAH
- XLU: CNP
- XLY: NKE

**Consecuencia:** XLV/XLU/XLY quedan por debajo del 85% de **cobertura efectiva** aunque el selector sí los incluya. Para que la cobertura efectiva suba, hay que descargar esos tickers.

**Ticket abierto:** `TOP20-FALTANTES-DATOS`. Próxima entrega: descargar los 8 tickers con protocolo de 6 pasos. Coste estimado: 1 sesión.

**Además, `peso_total_csv` < 100 en 3 ETFs EU:**
- FEZ: 66.26% (el CSV solo tiene 66% del ETF real)
- DAXEX: 78.43%
- ISF.L: 60.80%

Un "85% de cobertura" en FEZ significa 85% del 66% del ETF = **56% del ETF real**. El target no es comparable. **Ticket abierto:** `TOP20-EU-HOLDINGS-INCOMPLETOS`. Requiere obtener holdings completos.

**No se declara FIXED el frente hasta que estos dos tickets se cierren.**

### H-04. Backfill completo no resuelve el sesgo histórico (PIT)

**Aceptado. Documentado explícitamente como limitación.**

El backfill de A.2c se hizo con `data/etf_holdings.csv` actual (composición 2026Q4) aplicado a las 1086 fechas históricas.

- Si el objetivo fuera "composición PIT de cada fecha", sería incorrecto.
- El objetivo de esta regeneración es "comportamiento histórico de los componentes actuales", coherente con la decisión ya tomada en sesión 3 (commit a655fc95).
- **Mismo sesgo de supervivencia ya reconocido**, no uno nuevo.
- La alternativa (no backfill) creaba discontinuidad estructural, peor para H-03/H-04.

**Declaración en H-04 §13.5 de doc 69.** Los números de §12 y §13 son "comportamiento histórico de los componentes actuales", no "composición PIT".

**Limitación futura:** el sistema ya tiene `holdings_snapshots.csv` (PIT append-only desde 2026-10-25). En 12-24 meses habrá suficiente historia PIT para regenerar con composición real. Frente aparte cuando llegue el momento.

### H-05. 30/30 no debe ser único criterio de H-03

**Aceptado.**

- **Antes:** 30/30 era la única verificación empírica.
- **Ahora:** 15 tests deterministas del selector + 4 de selection_mode + 9 preexistentes de H-03 + 30/30 muestreo como complemento.
- Re-ejecutado sobre nueva base: 30/30 (idéntico, no depende de la base).

**Pendiente:** validación histórica exhaustiva (todos los casos elegibles) si el auditor lo requiere. Coste: 1 sesión.

### H-06. Comparación histórica debe distinguir comunes de nuevos

**Aceptado. Verificación ajustada.**

Verificación de 0 regresiones por fila `(date, sector)`:
- Regresión: `n_total(v2) < n_total(activo)`.
- Resultado: **0/11946 filas**.
- Expansión: 3258 filas (XLI 1086, XLF 1086, XLK 1086). Ningún otro sector cambia.

Lo que sí cambia: `pct_*` en XLI/XLF/XLK (denominador nuevo). Lo que no cambia: fase de cada ticker para `(date, sector, ticker)` común. Esto se verifica en los CSVs de holdings con el fix `replace_date` — antes había huérfanas (GE/PWR/VRT), ahora 0 inconsistencias `sum(count)=n_total`.

### H-07. H-04 debe evaluarse con criterios congelados

**Aceptado.**

- Doc 69 §13 es anexo de verificación, **no modifica** §2-§10.
- El pre-registro §9 ("si IC95% cruza 0 en >50% cohortes") **no discriminó** — ya reconocido en §12.4. No se reinterpreta.
- Los números nuevos están en §13 con la comparación §12 vs §13.
- No hay "ajuste a posteriori" para cerrar. La decisión de FIXED/OPEN está en el auditor.

---

## 2. Hallazgos nuevos (detectados en la ejecución, no en el dictamen)

### N-01. Granularidad extrema

69.7% de celdas `(date, sector)` tienen `n_estructuras ≤ 4` (antes 71% con top-20). Con denominador 1-4, `structure_ratio` toma valores discretos. Válido como agregado, inútil como señal por sector individual. Doc 69 §12.3 y §13.3.

### N-02. Bug persist (huérfanas)

`persist()` es append-only por `merge_keys`. Los 6 CSVs de holdings son subconjuntos filtrados por fase. Ticker que sale de la fase deja su fila anterior en el CSV. Detectado con GE/PWR/VRT (14 filas XLI en CSV, 11 en snapshot). **Corregido** con `replace_date=True` en A.2c. Tests: 2.

### N-03. Bug PCR (receipt ausente)

`compute_pcr_signals` retornaba `None` sin `skip()` cuando CBOE no respondía. El gate detectaba "sin receipt pcr_history" y fallaba el run entero. **Corregido** con `skip('pcr_history', 'NO_DATA')` en los dos `return None` tempranos. Tests: 4.

---

## 3. Cambios en producción (commits)

| Commit | Contenido |
|---|---|
| `404dc7c5` | Fase 2.1: yaml + get_stock_list fuente 5 (XLI 28 tickers) |
| `557048f8` | Selector + 15 tests |
| `0c14a8a3` | selection_mode paramétrico + 4 tests |
| `5e4fe2fc` | regenerate_wyckoff_history parametrizado |
| `efd53cfe` | MIN_TICKERS_PER_GROUP = TOP_N_HOLDINGS (evita regresión) |
| `f4528b49` | Fix persist replace_date + 2 tests |
| `9338dcd4` | Fix PCR skip NO_DATA + 4 tests |
| `e7f27cd0` | A.2c completo: run.py + outputs regenerados + pre_top20_target |
| `a14b463b` | REGISTRO TOP20 con evidencia |
| `8a0a8b1d` | Doc 69 §13 + H-03/H-04 evidencia actualizada |

CI verde en todos los pushes. Suite: 3823 passed al cierre.

---

## 4. Estado de gobernanza

`DICT3-TOP20-COBERTURA` **sigue OPEN**. No se cierra sin:

1. Aprobación de este doc 70-v2 (recibida: aprobación condicionada 2026-10-26).
2. Cierre (o nuevo alcance) de `TOP20-FALTANTES-DATOS` (XLV/XLU/XLY y demás).
3. Cierre (o nuevo alcance) de `TOP20-EU-HOLDINGS-INCOMPLETOS` (FEZ/DAXEX/ISF.L).
4. Decisión sobre índice USA (SPY/QQQ/DIA) y EU, IWM.

Pendientes de auditoría que no bloquean uso productivo:
- Validación histórica exhaustiva de H-03 (todos los casos, no muestra).
- Decisión de FIXED/OPEN de `DICT3-H04-INTERVALS` con doc 69 completo.
- Las 6 acciones del dictamen 2026-10-26 (ver §4 siguiente).

---

## 5. Lo que este cambio NO hace

- No expande índices USA ni EU.
- No toca IWM.
- No resuelve el problema de tickers seleccionados sin datos en parquet.
- No cambia el sesgo PIT histórico (ya reconocido).
- No publica `structure_ratio` con IC en el reporte.
- No cierra `DICT3-TOP20-COBERTURA` como FIXED.

---

**Preparado 2026-10-26 para revisión externa. Aprobación condicionada recibida 2026-10-26. Acciones pendientes en doc 70-v2b.**

**Fin del doc 70-v2.**