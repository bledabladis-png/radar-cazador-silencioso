# P66 - Informe de cierre del paso 5 (solicitud al auditor)

**Objeto:** comunicacion al auditor externo de la resolucion de las
2 condiciones establecidas en el dictamen #41 (GO CONDICIONADO) y
solicitud de cierre definitivo del ciclo P66.

**Origen:** dictamen #41 (auditoria de salida P66, 2026-09-21).

**Fecha:** 2026-09-21.

**HEAD al redactar:** 4502fa6.

**Naturaleza:** documento de entrega. NO normativo. En caso de conflicto
con el dictamen #41 o con `NIPC_CONTRATOS_SEMANTICOS_v1.md`, gana el
dictamen o el contrato.

---

## 0. Resumen ejecutivo

El dictamen #41 establecio dos condiciones para declarar el cierre del
paso 5:

1. La decision A/B (lectura A) debe quedar incorporada a la trazabilidad
   formal del ciclo, no solo expresada en el expediente.
2. Los 3 fallos de `tests/test_freshness.py` deben demostrarse
   preexistentes con evidencia directa, no por afirmacion.

**Ambas condiciones estan resueltas.** Commit `4502fa6`.

**Se solicita al auditor:** cierre definitivo del paso 5 del GO #40.

---

## 1. Condicion 1 - Trazabilidad A/B incorporada

**Texto del dictamen #41 (seccion 10):**

> El paso 5 puede continuar con esta base, pero el expediente no debe
> etiquetarse como "ciclo P66 cerrado sin reservas" hasta que la decision
> quede incorporada a la trazabilidad formal del ciclo.

**Resolucion aplicada:**

| Documento | Contenido |
|---|---|
| `iae/DICTAMENES.md` #41 | Lectura A adoptada. Los 4 grupos permanecen R3=N/D. Prohibiciones confirmadas. Estado operativo. |
| `iae/evidence/p66_e2e_probe/README.md` apendice §8 | Lectura A adoptada. Sin cambios en `resolve_r3`. Referencia a #41. |
| Commit `4502fa6` | Registro formal del dictamen y sus consecuencias. |

**Estado: CUMPLIDA.**
---

## 2. Condicion 2 - Reserva freshness resuelta con evidencia

**Texto del dictamen #41 (seccion 5):**

> "preexistentes, no relacionados" debe considerarse demostrada
> solamente si existe evidencia de baseline anterior a los commits P66
> que produzca los mismos tres fallos.

**Resolucion aplicada:** evidencia en `iae/evidence/p66_baseline_pre/`.

**Metodologia:**

    git worktree add --detach D:\_baseline_p66_tmp 64b0637

- `64b0637` es el padre de `bff0945`, primer commit del ciclo P66.
- Copia de los parquets actuales (gitignored) al worktree para
  reproducir exactamente el estado local.
- Ejecucion `tests/test_freshness.py -q --tb=line`.
- Worktree eliminado tras la prueba.

**Resultado:**

| Ejecucion | HEAD | Parquets | Resultado |
|---|---|---|---|
| Pre-P66 | `64b0637` | 2026-09-16 | **3 failed + 34 passed + 2 skipped** |
| Actual | `b0f8850` | 2026-09-16 | **3 failed + 1067 passed + 2 skipped** |

**Fallos identicos en ambos:**

- `test_market_data_fresh` -> market_data stale: 5 dias (max 4)
- `test_stock_prices_fresh` -> stock_prices stale: 5 dias (max 4)
- `test_european_tickers_recent` -> solo 19/51 (37%) europeos recientes

**Prueba adicional (frontera del ciclo):**

    git log --oneline 64b0637..HEAD -- tests/test_freshness.py
        (sin salida)

    git log --oneline 64b0637..HEAD -- data/market_data.parquet data/stock_prices.parquet
        (sin salida)

El ciclo P66 no toca ni el test ni los parquets. Causa raiz: los
parquets en disco (`index.max() = 2026-09-16`, 5 dias < 2026-09-21)
estan desactualizados; se regeneran con el proximo `daily_run.yml`
(cron `0 4 * * *`). No se regeneran a mano para no activar el patron
K-RUN-OUT-OF-WINDOW-01.

**Estado: CUMPLIDA con evidencia directa.**
---

## 3. Estado del ciclo P66

| Paso | Descripcion | Estado | Commit / evidencia |
|---|---|---|---|
| 1 | Traslado §14.3 a contrato | CERRADO | `3a233b4` |
| 2 | Tests contractuales §14.3 | CERRADO | `bff0945` |
| 3 | Implementacion `reporting_dedup.py` | CERRADO TECNICAMENTE | `fe7c1dd` + `3be5465` + `ea50880` |
| 4 | Probe e2e sobre 13F reales | CERRADO | `a00ee48` |
| Divergencia A/B | Lectura A adoptada | RESUELTA | `4502fa6` + `iae/DICTAMENES.md` #41 |
| Reserva freshness | Baseline pre-P66 | RESUELTA | `4502fa6` + `iae/evidence/p66_baseline_pre/` |
| 5 | Auditoria de salida | ABIERTO (solicitud de cierre) | este informe |
| 6 | Activacion `DROP_DUP` | NO AUTORIZADO | (requiere dictamen especifico) |

---

## 4. Lo que NO se ha tocado desde el dictamen #41

- `src/institutional_accumulation/aggregation/reporting_dedup.py`: sin cambios.
- `resolve_r3`: sin cambios (lectura A conservada).
- `NIPC_CONTRATOS_SEMANTICOS_v1.md` §14.3: sin cambios.
- `NIPC_COVERAGE_POLICY.md` v1.0: INTACTA (hash 57f2d01f...).
- `MATCH_KEY`: intacto.
- C2: intacto.
- `delta_shares.py`: intacto.
- `nipc.py`: intacto.
- `coverage.py`: intacto.
- `relationships.py`: intacto.
- `DROP_DUP`: NO activado.
- OpenFIGI masivo: NO autorizado / NO ejecutado.
- Push a `origin/main`: NO.
- Certificacion "acumulacion": NO emitida.
---

## 5. Solicitud al auditor

Se solicita formalmente:

1. **Cierre definitivo del paso 5 del GO #40.**

   Condiciones del dictamen #41 cumplidas:
   - Trazabilidad A/B incorporada (`iae/DICTAMENES.md` #41 + `p66_e2e_probe/README.md` §8).
   - Reserva freshness resuelta con evidencia directa (`iae/evidence/p66_baseline_pre/`).

2. **Confirmacion del estado del ciclo P66 tras el cierre del paso 5.**

   | Elemento | Estado propuesto |
   |---|---|
   | Paso 1 | CERRADO |
   | Paso 2 | CERRADO |
   | Paso 3 | CERRADO TECNICAMENTE (lectura A) |
   | Paso 4 | CERRADO |
   | Paso 5 | CERRADO (solicitado) |
   | Paso 6 (`DROP_DUP`) | NO AUTORIZADO - requiere dictamen especifico |

3. **Aclaracion del estatus del ciclo P66 en su conjunto:**
   - ?Ciclo P66 CERRADO, con paso 6 pendiente de dictamen especifico?
   - ?Ciclo P66 CERRADO en su totalidad?
   - ?Otra etiqueta que el auditor considere apropiada?

---

## 6. Bloqueos vigentes sin cambios

| Bloqueo | Estado |
|---|---|
| THRESHOLD_1 | UNDEFINED |
| THRESHOLD_2 | BLOQUEADO |
| Gate-NIPC.2 | BLOQUEADO |
| Gate-NIPC.3 | NO AUTORIZADO |
| OpenFIGI masivo | NO AUTORIZADO |
| Policy v1.3 aplicacion | NO AUTORIZADA |
| Push a `origin/main` | NO |
| Activacion `DROP_DUP` | NO AUTORIZADA |
| Certificacion "acumulacion" | BLOQUEADA |
---

## 7. Trazabilidad

### 7.1. Commits del ciclo P66 (todos locales, sin push)

    3a233b4  docs(iae): traslado §14.3 reformulada
    bff0945  test(p66): contrato 14.3 - capa A/B/C
    fe7c1dd  feat(p66): 14.3.6 canonicalizacion FormNum
    3be5465  feat(p66): 14.3.5 mapping + 14.3.1 PASO 1-2 + 14.3.2 cadena
    ea50880  feat(p66): 14.3.1 R3 tri-state + 14.3.4 R4 + L3 booleano
    a00ee48  chore(p66): probe e2e
    b0f8850  docs(p66): expediente para auditoria de salida
    4502fa6  docs(p66): dictamen #41 + baseline pre-P66 + resolucion A/B
    (este)   docs(p66): informe de cierre del paso 5

### 7.2. Evidencia directa

    iae/evidence/p66_e2e_probe/       (probe e2e + README + output + HASHES)
    iae/evidence/p66_baseline_pre/    (baseline freshness + README + output + HASHES)
    iae/evidence/p66_gate04..gate10/  (gates previos, sin cambios)

### 7.3. Documentos normativos

    NIPC_CONTRATOS_SEMANTICOS_v1.md §14.3    contrato reformulado
    iae/DICTAMENES.md #41                    dictamen de auditoria de salida
    iae/P66_EXPEDIENTE_AUDITORIA_SALIDA.md   expediente de salida
    iae/P66_L3_REFORMULACION_PROPUESTA.md    v7-ter

---

## 8. Estado final del repo

    HEAD local            (el de este commit)
    origin/main           9d4a81e
    Ahead                 ~194 commits locales
    Behind                3 (bot CI)
    Working tree          limpio
    Push                  NO
    Suite (local)         1067 passed + 2 skipped + 3 failed preexistentes
    pyflakes              LIMPIO
    compileall            OK

Los 3 failed preexistentes son los de `test_freshness.py`, demostrados
preexistentes en seccion 2 de este informe.

---

Fin del informe. Redactado 2026-09-21.