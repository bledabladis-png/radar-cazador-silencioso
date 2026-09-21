# IAE - P66 Dossier consolidado para analisis del auditor

**Objeto:** documento consolidado del ciclo P66, desde su origen
(P65 WONT FIX) hasta su estado actual. Destinado al analisis del
auditor externo.

**Origen:** sintesis de dictamenes #28 a #41, expediente de auditoria
de salida, informe de cierre del paso 5 y evidencia empirica.

**Fecha:** 2026-09-21.

**HEAD al redactar:** 12abb5f.

**Naturaleza:** documento de sintesis. NO normativo. En caso de
conflicto, gana `NIPC_CONTRATOS_SEMANTICOS_v1.md` o el dictamen
correspondiente.

---

## 0. Resumen ejecutivo

P66 reformulo el contrato §14.3 (L3 - Manager Duplication Contract)
para permitir la materializacion del requisito 4 de L3 desde el Data
Set SEC sin XML crudo, usando la tabla `OTHERMANAGER`.

Estado actual:

    Paso 1 (traslado)         CERRADO
    Paso 2 (tests)            CERRADO
    Paso 3 (implementacion)   CERRADO TECNICAMENTE
    Paso 4 (probe e2e)        CERRADO
    Paso 5 (auditoria)        ABIERTO - solicitud de cierre enviada
    Paso 6 (DROP_DUP)         NO AUTORIZADO

    Divergencia A/B           RESUELTA (lectura A, dictamen #41)
    Reserva freshness         RESUELTA (evidencia p66_baseline_pre)

**Se solicita al auditor:** revision del dossier + cierre formal
del paso 5.

---

## 1. Objeto del dossier

Proporcionar al auditor externo una vision consolidada del ciclo P66
para su analisis: origen, contrato, dictamenes, implementacion, tests,
probe e2e, decisiones tomadas y estado actual.

El dossier sustituye a la lectura secuencial de 6 documentos dispersos,
pero NO sustituye a las fuentes originales (contrato, dictamenes,
evidencia).
---

## 2. Origen y motivacion

**Antecedente:** P65 (Manager Duplication Contract, 2026-09-20) cerro
su ciclo con GO CONDICIONADO v3. Su requisito 4 (L3) exigia "evidencia
cruzada entre filings" para autorizar deduplicacion intra-periodo.

P65 v2 intento materializar ese requisito desde `OTHERMANAGER2` y
`ADDITIONALINFORMATION` sin exito. Cerrado como WONT FIX.

**Descubrimiento:** investigacion posterior confirmo que la fuente
estructurada existe en `OTHERMANAGER` del Data Set SEC
("other managers reporting for this manager"), distinta de
`OTHERMANAGER2` ("included in this report").

**Consecuencia:** se abrio P66 para reformular §14.3 y materializar
el requisito 4 sin dependencia de XML crudo ni parser paralelo.

---

## 3. El contrato §14.3 reformulado

**Documento:** `NIPC_CONTRATOS_SEMANTICOS_v1.md` §14.3.

**Estructura actual (7 subsecciones):**

| § | Objeto |
|---|---|
| 14.3.1 | R3 tri-state (TRUE/FALSE/N/D) + PASO 0 completitud |
| 14.3.2 | Validacion de cadena de amendments |
| 14.3.3 | Semantica RESTATEMENT / NEW HOLDINGS |
| 14.3.4 | R4 candidate_A + CONFLICT scoped a A |
| 14.3.5 | Mapping FormNum -> CIK |
| 14.3.6 | Canonicalizacion IAE de Form13FFileNumber |
| 14.3.7 | Clausula de cierre contractual |

**Regla booleana L3:**

    L3(A, B, S, period) = R1 AND R2 AND (R3 == TRUE)
                          AND (R4 == MATCH) AND R5

**Estados preservados:** N/D y CONFLICT nunca se convierten en False.
Impiden L3 == True sin colapsar a un booleano negativo.
---

## 4. Cronologia de dictamenes (#28 a #41)

| # | Tema | Resultado |
|---|---|---|
| 28 | Propuesta v2-bis | NO-GO, 6 bloqueos |
| 29 | Gate 0.8 exigido | Correccion de alcance |
| 30 | CONFLICT > MATCH, base unico | Bloqueos |
| 31 | R3 con REPORTTYPE, CONFLICT scoped | Ajustes normativos |
| 32 | GO CONDICIONADO FINAL | 2 correcciones finales |
| 33 | CONFLICT ampliado, alcance filing A | 2 precisiones |
| 34 | FormNum flexible, candidate_A, L3 booleano | 4 bloqueos + 2 rec. |
| 35 | BASE_R4 global, cadena completa | 5 bloqueos + 1 ajuste |
| 36 | BASE/AMENDMENT por SUBMISSIONTYPE | 3 cierres + 2 ajustes |
| 37 | Flujo unico R3, familia global | 3 correcciones |
| 38 | §3.4 como nota, trazabilidad | 1 bloqueo + 2 editoriales |
| 39 | PASO 0 completitud scope | 1 bloqueo + 1 editorial |
| 40 | GO CONTRACTUAL FINAL | Cierre de la fase contractual |
| 41 | Auditoria de salida | GO CONDICIONADO |

**Fase de contrato:** dictamenes #28 a #40 (13 iteraciones hasta GO).
**Fase de implementacion + auditoria:** dictamen #41.

**Decisiones clave del auditor a lo largo del ciclo:**

- `OTHERMANAGER` (no `OTHERMANAGER2`) es la fuente de R4.
- Universo R4 = NOTICE + COMBINATION (via `REPORTTYPE`, no solo `SUBMISSIONTYPE`).
- CIK primario; `Form13FFileNumber` fallback resoluble.
- CONFLICT > MATCH, scoped a `candidate_A`.
- N/D != NO_MATCH.
- Fail-closed en pluralidad documental (>1 familia, >1 base).
- PASO 0: completitud del scope demostrable antes de R3=FALSE.
---

## 5. Gates empiricos ejecutados

| Gate | Objeto | Resultado |
|---|---|---|
| 0.1 | Cobertura OTHERMANAGER por tipo | 100% en NOTICE + COMBINATION |
| 0.2 | Caso Vanguard | Modelo Q4 vs Q1 |
| 0.3 | Granularidad de identidad | CIK / FormNum / N/D |
| 0.4 | Mapping FormNum -> CIK | 1:1, 0 N:1 |
| 0.4-reissue | Filtro PERIODOFREPORT declarado | Directorios cross-periodo |
| 0.5 | Amendment probe | RESTATEMENT sustituye |
| 0.6 | Combination probe | R4 = 2.334 / 2.319 |
| 0.7 | NEW HOLDINGS probe | 2/2 + 1/1 identicos |
| 0.8 | Cobertura identidad R4 completo | 91.70% / 91.31% |
| 0.9 | Multiples filings base | 1 caso MIXED (CIK 0002016827) |
| 0.10 | FormNum representation | Padding flexible |

Evidencia en `iae/evidence/p66_gateXX_*/`.

---

## 6. Implementacion tecnica

**Fichero:** `src/institutional_accumulation/aggregation/reporting_dedup.py`.

**Funciones puras nuevas (10):**

- `canonicalize_form13f_filenumber` (§14.3.6).
- `classify_filing_family` / `classify_filing_role` (§14.3.1 PASO 1-2).
- `build_formnum_cik_mapping` / `resolve_formnum_to_cik` (§14.3.5).
- `validate_amendment_chain` (§14.3.2).
- `resolve_r3` (§14.3.1).
- `resolve_r4` + `_row_identity_state` (§14.3.4).
- `evaluate_l3` (§14.3).

**Propiedades:** deterministas, sin IO, sin `datetime.now()`, sin
side effects.

**No integradas en el pipeline P65:** `build_effective_reporting_snapshot`
y `classify_reporting_transition` no invocan las nuevas funciones.
Es deliberado: R4 aislado NO autoriza DROP_DUP.

---

## 7. Tests y regresion

| Bloque | Resultado |
|---|---|
| `tests/test_p66_contract.py` (nuevo, 31 tests) | 31 passed |
| `tests/test_sec_13f_reporting_dedup.py` (P65) | 31 passed |
| Suite global (locales) | 1067 passed + 2 skipped + 3 failed preexistentes |
| pyflakes | LIMPIO |
| compileall | OK |

**Capa A** (10 tests): integridad documental de §14.3.
**Capa B** (3 tests): no regresion P65.
**Capa C** (18 tests): semantica ejecutable sobre funciones puras.

Los 12 xfail originales fueron retirados progresivamente conforme
se implementaba cada subseccion.
---

## 8. Probe e2e

**Evidencia:** `iae/evidence/p66_e2e_probe/`.

**Ejecucion sobre datos SEC reales Q4 2025 + Q1 2026.**

| Metrica | 2025Q4 | 2026Q1 |
|---|---:|---:|
| Filings totales | 10,676 | 10,776 |
| Filings R4 no clasificables | 0 | 0 |
| Grupos (CIK, PERIOD) | 10,524 | 10,648 |
| R3 = TRUE | 2,286 | 2,309 |
| R3 = FALSE | 8,235 | 8,337 |
| R3 = N/D | 3 | 2 |
| Mapping FormNum -> CIK | 10,524 | 10,648 |

**Coincidencia verificada:** R3=TRUE coincide exactamente con el
conteo "con 1 filing base" del Gate 0.9.

**Divergencia material detectada:** §14.3.1 PASO 1 filtra a R4 antes
de PASO 2. Un `13F-HR / 13F HOLDINGS REPORT` (no-R4) seguido de un
amendment `13F-HR/A / 13F COMBINATION REPORT` (R4, RESTATEMENT)
produce `BASE=0 + AMENDMENT>=1 -> R3=N/D`.

Casos reales: 2 en Q4 + 2 en Q1 = 4 grupos sobre 21.172 (0.02%).

**Lecturas posibles:**

    (A) Literal actual  -> R3=N/D  (implementada)
    (B) Extendida       -> R3=TRUE (no implementada)

**Resolucion del auditor (#41):** lectura A adoptada. Sin cambios
de codigo. Los 4 grupos permanecen R3=N/D.

**Reserva de auditoria (freshness):** los 3 fallos de
`tests/test_freshness.py` requieren evidencia de baseline pre-P66.
Resuelto: worktree detached en `64b0637` + parquets actuales ->
3 failed identicos al HEAD actual. Evidencia en
`iae/evidence/p66_baseline_pre/`.
---

## 9. Dictamen #41 - Auditoria de salida

**Resultado:** GO CONDICIONADO. NO cierre definitivo del ciclo.

| Elemento | Dictamen |
|---|---|
| Paso 1 | CERRADO |
| Paso 2 | CERRADO |
| Paso 3 | CERRADO TECNICAMENTE |
| Paso 4 | CERRADO |
| Divergencia A/B | LECTURA A |
| Paso 5 | ABIERTO, sin cierre definitivo |
| DROP_DUP | BLOQUEADO |

**Condiciones del #41 para cerrar el paso 5:**

1. Decision A/B incorporada a la trazabilidad formal del ciclo.
2. Fallos freshness demostrados preexistentes con evidencia directa.

**Prohibiciones confirmadas:** no DROP_DUP, no tocar modulos fuera del
alcance, no cambios contractuales por inferencia, no convertir N/D en
NO_MATCH, no push.

---

## 10. Acciones post-#41

    commit 4502fa6  dictamen #41 + baseline pre-P66 + resolucion A/B
    commit 12abb5f  informe de cierre del paso 5

**Condicion 1 cumplida:** `iae/DICTAMENES.md` #41 + apendice §8 en
`iae/evidence/p66_e2e_probe/README.md`.

**Condicion 2 cumplida:** evidencia directa en
`iae/evidence/p66_baseline_pre/` (worktree detached en `64b0637` +
parquets actuales -> 3 failed identicos al HEAD actual).

---

## 11. Estado actual y bloqueos

    HEAD local              12abb5f
    origin/main             9d4a81e
    Ahead                   194 commits
    Behind                  3 (bot CI)
    Working tree            limpio
    Push                    NO

**Bloqueos vigentes:** THRESHOLD_1/2 UNDEFINED, Gate-NIPC.2 BLOQUEADO,
Gate-NIPC.3 NO AUTORIZADO, OpenFIGI masivo NO AUTORIZADO, Policy v1.3
NO APLICADA, DROP_DUP NO AUTORIZADO, certificacion "acumulacion"
BLOQUEADA.

---

## 12. Preguntas al auditor

1. **Cierre formal del paso 5** del GO #40.
2. **Confirmacion del estado del ciclo P66** tras el cierre del paso 5.
3. **Aclaracion de la etiqueta final** del ciclo: "cerrado con paso 6
   pendiente" / "cerrado en su totalidad" / otra.
4. **Confirmacion de que la lectura A** de §14.3.1 no requiere
   iteracion contractual adicional.
---

## Anexo A - Trazabilidad de commits del ciclo P66

    3a233b4  docs(iae): traslado §14.3 reformulada
    bff0945  test(p66): contrato 14.3 - capa A/B/C
    fe7c1dd  feat(p66): 14.3.6 canonicalizacion FormNum
    3be5465  feat(p66): 14.3.5 mapping + 14.3.1 PASO 1-2 + 14.3.2 cadena
    ea50880  feat(p66): 14.3.1 R3 tri-state + 14.3.4 R4 + L3 booleano
    a00ee48  chore(p66): probe e2e
    b0f8850  docs(p66): expediente para auditoria de salida
    4502fa6  docs(p66): dictamen #41 + baseline pre-P66 + resolucion A/B
    12abb5f  docs(p66): informe de cierre del paso 5

## Anexo B - Evidencia en disco

    iae/evidence/p66_e2e_probe/       probe + output + README + HASHES
    iae/evidence/p66_baseline_pre/    baseline + output + README + HASHES
    iae/evidence/p66_gate04..10/      gates empiricos

## Anexo C - Documentos de referencia

    NIPC_CONTRATOS_SEMANTICOS_v1.md §14.3      contrato reformulado
    iae/P66_L3_REFORMULACION_PROPUESTA.md      v7-ter (propuesta congelada)
    iae/P66_EXPEDIENTE_AUDITORIA_SALIDA.md     expediente de salida
    iae/P66_INFORME_CIERRE_PASO5.md            informe de cierre paso 5
    iae/P66_INFORME_HALLAZGO.md                informe de hallazgo
    iae/DICTAMENES.md #28 a #41                dictamenes

---

Fin del dossier. Redactado 2026-09-21. HEAD 12abb5f.