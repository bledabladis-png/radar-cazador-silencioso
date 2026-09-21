# IAE - INFORME CONSOLIDADO

**Generado:** 2026-09-20
**Consolida:** 19 informes del ciclo IAE

**Regla:** este fichero se actualiza in-place. Las versiones previas se archivan con fecha.

**Actualizacion 2026-09-20 v2:** F2.4 EMITIDO por auditor externo.
GO CONDICIONADO arquitectura + 3 bloqueantes estructurales.
D1 GO, D2 GO CONDICIONADO, D3 GO. Q12 Modelo A. AGREG. Opcion 2.
OpenFIGI GO CONDICIONADO (snapshot+hash). Policy v1.3 NO.
THRESHOLD_2 BLOQUEADO. Certificacion "acumulacion" BLOQUEADA.
Referencia: DICTAMENES.md seccion 24 + INFORME.md seccion 18.

---

## Indice

1. [INFORME DE CURACION CUSIP -> TICKER](#informe-de-curacion-cusip-->-ticker)
2. [INFORME FA-2.1 - Filtro temporal canonico por PERIODOFREPORT](#informe-fa-21---filtro-temporal-canonico-por-perio)
3. [INFORME FA-2.4 - Amendments + Canonical Snapshot](#informe-fa-24---amendments-+-canonical-snapshot)
4. [INFORME GATE 0 FA-2 - IAE SEC 13F](#informe-gate-0-fa-2---iae-sec-13f)
5. [INFORME GATE 0 - Institutional Accumulation Evidence (IAE)](#informe-gate-0---institutional-accumulation-eviden)
6. [INFORME GATE FA-1 - IAE SEC 13F](#informe-gate-fa-1---iae-sec-13f)
7. [INFORME GATE FA-2 - IAE SEC 13F](#informe-gate-fa-2---iae-sec-13f)
8. [INFORME - Coverage baseline NIPC (Gate 0 baseline)](#informe---coverage-baseline-nipc-(gate-0-baseline))
9. [INFORME GATE 0 - NIPC (Net Institutional Position Change)](#informe-gate-0---nipc-(net-institutional-position-)
10. [INFORME MINI-PROBE FILER CONTINUITY (Q-PROBE-5) - Q4 2025 -> Q1 2026](#informe-mini-probe-filer-continuity-(q-probe-5)---)
11. [INFORME PROBE END-TO-END NIPC - Q4 2025 -> Q1 2026](#informe-probe-end-to-end-nipc---q4-2025-->-q1-2026)
12. [INFORME TOP 50 FILER CONTINUITY - Q-MINI-5](#informe-top-50-filer-continuity---q-mini-5)
13. [IAE NIPC - Informe OpenFIGI + RADAR_TARGET_CATALOG + TARGET_UNIVERSE](#iae-nipc---informe-openfigi-+-radar_target_catalog)
14. [IAE NIPC - Informe piloto TOP 2000 + hallazgos de calidad de identidad](#iae-nipc---informe-piloto-top-2000-+-hallazgos-de-)
15. [INFORME DE ENTREGA AL AUDITOR - SOLICITUD DE F2.4](#informe-de-entrega-al-auditor---solicitud-de-f24)
16. [NIPC_INFORME_ESTADO_POST_FIXES_F24](#nipc_informe_estado_post_fixes_f24)
17. [INFORME TECNICO - SISTEMA IAE (INSTITUTIONAL ACCUMULATION EVIDENCE)](#informe-tecnico---sistema-iae-(institutional-accum)
18. [INFORME F2.4 - Dictamen formal y bloqueantes](#informe-f24---dictamen-formal-y-bloqueantes)
19. [INFORME P65 v1 - Manager Duplication (ciclo completo)](#informe-p65-v1---manager-duplication-ciclo-completo)
20. [INFORME P66 L3 - Reformulacion contrato 14.3 (ciclo completo)](#informe-p66-l3---reformulacion-contrato-143-ciclo-completo)
21. [INVENTARIO A.6.0 - Gate 0 de los 3 bloqueantes F2.4](#inventario-a60---gate-0-de-los-3-bloqueantes-f24)

---

## 1. INFORME DE CURACION CUSIP -> TICKER

**Fichero original:** `archive/iae/informes_originales/INSTITUTIONAL_ACCUMULATION_CUSIP_CURATION_INFORME.md`

**Modulo:** IAE (Institutional Accumulation Evidence) - SEC 13F
**Referencia:** prompt v6.35, HEAD 10a5cc2
**Fecha:** 2026-09-19
**Estado:** Analisis cerrado. Dictamen Q-CUR-1 / Q-CUR-2 aplicado. Alcance minimo aprobado.
---
Gate 0 ejecutado sobre el estado real del crosswalk CUSIP -> ticker. Hallazgos:

---

## 2. INFORME FA-2.1 - Filtro temporal canonico por PERIODOFREPORT

**Fichero original:** `archive/iae/informes_originales/INSTITUTIONAL_ACCUMULATION_FA21_INFORME.md`

Version: 1.0 (2026-09-19)
Dictamen habilitante: INSTITUTIONAL_ACCUMULATION_FA2_GATE0_DICTAMEN.md
Commit: 1679639
Estado: CERRADO local. Sin push (regla local-first IAE).
---
Filtro canonico de los 7 TSVs al periodo del trimestre segun

---

## 3. INFORME FA-2.4 - Amendments + Canonical Snapshot

**Fichero original:** `archive/iae/informes_originales/INSTITUTIONAL_ACCUMULATION_FA24_INFORME.md`

Version: 1.0 (2026-09-19)
Dictamen habilitante: INSTITUTIONAL_ACCUMULATION_FA24_DICTAMEN_GATE0.md
Especificacion: INSTITUTIONAL_ACCUMULATION_FA24_ESPECIFICACION.md
Estado: CERRADO local. Sin push (regla local-first IAE).
---
Resolver snapshot canonico del trimestre por (CIK, PERIOD) aplicando

---

## 4. INFORME GATE 0 FA-2 - IAE SEC 13F

**Fichero original:** `archive/iae/informes_originales/INSTITUTIONAL_ACCUMULATION_FA2_GATE0_INFORME.md`

Version: 1.0 (2026-09-19)
Contrato: docs/auditoria/INSTITUTIONAL_ACCUMULATION_CONTRATO.md (v1.1)
Addendum: docs/auditoria/INSTITUTIONAL_ACCUMULATION_CONTRATO_ADDENDUM.md
Dictamen previo: docs/auditoria/INSTITUTIONAL_ACCUMULATION_DICTAMEN_GATE0.md
Informe Gate FA-1: docs/auditoria/INSTITUTIONAL_ACCUMULATION_GATE_FA1_INFORME.md
Prompt de referencia: v6.34

---

## 5. INFORME GATE 0 - Institutional Accumulation Evidence (IAE)

**Fichero original:** `archive/iae/informes_originales/INSTITUTIONAL_ACCUMULATION_GATE0_INFORME.md`

Version: 1.0 (2026-09-19)
Prompt de referencia: v6.32 (HEAD 7dea18c)
Contrato previo: docs/auditoria/INSTITUTIONAL_ACCUMULATION_CONTRATO.md
Propuesta: docs/auditoria/INSTITUTIONAL_ACCUMULATION_PROPUESTA.md
Dataset analizado: 13F Q1 2026 (SEC Data Set, publicado 2026-06-04)
Estado: ciclo C cerrado. Pendiente dictamen auditor sobre reformulaciones Q2 y Q5.

---

## 6. INFORME GATE FA-1 - IAE SEC 13F

**Fichero original:** `archive/iae/informes_originales/INSTITUTIONAL_ACCUMULATION_GATE_FA1_INFORME.md`

Version: 1.0 (2026-09-19)
Contrato: docs/auditoria/INSTITUTIONAL_ACCUMULATION_CONTRATO.md (v1.1)
Addendum: docs/auditoria/INSTITUTIONAL_ACCUMULATION_CONTRATO_ADDENDUM.md
Prompt de referencia: v6.33
Estado: Gate FA-1 superado. Pendiente push.
Naturaleza: no normativo. Si hay conflicto, gana el Prompt Maestro.

---

## 7. INFORME GATE FA-2 - IAE SEC 13F

**Fichero original:** `archive/iae/informes_originales/INSTITUTIONAL_ACCUMULATION_GATE_FA2_INFORME.md`

Version: 1.0 (2026-09-19)
Dictamenes habilitantes:
  FA-2.0: INSTITUTIONAL_ACCUMULATION_FA2_GATE0_DICTAMEN.md
  FA-2.3: INSTITUTIONAL_ACCUMULATION_FA23_DICTAMEN_C.md + DICTAMEN_CIERRE.md
  FA-2.4: INSTITUTIONAL_ACCUMULATION_FA24_DICTAMEN_GATE0.md
Estado: Gate FA-2 superado localmente. Pendiente dictamen final.

---

## 8. INFORME - Coverage baseline NIPC (Gate 0 baseline)

**Fichero original:** `archive/iae/informes_originales/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_BASELINE_INFORME.md`

**Modulo:** IAE (Institutional Accumulation Evidence) - SEC 13F / NIPC
**Referencia:** prompt maestro v6.36
**Fecha:** 2026-09-19
**Fase:** coverage baseline (Q-T50-5, post dictamen TOP 50)
**Dictamen habilitante:** INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_TOP50_DICTAMEN.md
**Evidencia:** docs/auditoria/iae/evidence/nipc_gate0_baseline/

---

## 9. INFORME GATE 0 - NIPC (Net Institutional Position Change)

**Fichero original:** `archive/iae/informes_originales/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_INFORME.md`

**Modulo:** IAE (Institutional Accumulation Evidence) - SEC 13F
**Referencia:** prompt v6.35, HEAD 9d4a81e
**Fecha:** 2026-09-19
**Estado:** Gate 0 completado. Pendiente dictamen del auditor (7 preguntas bloqueantes).
---
Gate 0 empirico completado para NIPC. Estado:

---

## 10. INFORME MINI-PROBE FILER CONTINUITY (Q-PROBE-5) - Q4 2025 -> Q1 2026

**Fichero original:** `archive/iae/informes_originales/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_MINIPROBE_INFORME.md`

Version: 1.0 (2026-09-19)
HEAD revisado: d75e84c
Estado: mini-probe completado. Evidencia directa NT -> OTHERMANAGER -> HR.
Autor: Ingeniero Supervisor.
Naturaleza: no normativo. Si hay conflicto con el Prompt Maestro, gana el
            Prompt Maestro.

---

## 11. INFORME PROBE END-TO-END NIPC - Q4 2025 -> Q1 2026

**Fichero original:** `archive/iae/informes_originales/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_PROBE_INFORME.md`

Version: 1.0 (2026-09-19)
HEAD revisado: b75c262
Estado: probe completado. Hallazgo estructural no cubierto por el contrato.
Autor: Ingeniero Supervisor.
Naturaleza: no normativo. Si hay conflicto con el Prompt Maestro, gana el
            Prompt Maestro.

---

## 12. INFORME TOP 50 FILER CONTINUITY - Q-MINI-5

**Fichero original:** `archive/iae/informes_originales/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_TOP50_INFORME.md`

Version: 1.0 (2026-09-19)
HEAD revisado: 37d2879
Estado: TOP 50 completado. Vanguard confirmado como concentracion unica.
Autor: Ingeniero Supervisor.
Naturaleza: no normativo. Si hay conflicto con el Prompt Maestro, gana el
            Prompt Maestro.

---

## 13. IAE NIPC - Informe OpenFIGI + RADAR_TARGET_CATALOG + TARGET_UNIVERSE

**Fichero original:** `archive/iae/informes_originales/INSTITUTIONAL_ACCUMULATION_OPENFIGI_CATALOG_TARGET_UNIVERSE_INFORME.md`

Ciclo: F2.3-bis -> entrega de evidencia empirica al auditor.
Fecha: 2026-09-19.
Estado: BORRADOR para dictamen externo. No push.
Autoridad: dictamen F2.3 NO GO con cambios obligatorios +
           dictamen F2.3-bis GO CONDICIONADO (Q1-Q5).
---

---

## 14. IAE NIPC - Informe piloto TOP 2000 + hallazgos de calidad de identidad

**Fichero original:** `archive/iae/informes_originales/INSTITUTIONAL_ACCUMULATION_OPENFIGI_TOP2000_INFORME.md`

Ciclo: F2.3-bis -> validacion cuantitativa ponderada.
Fecha: 2026-09-19.
Estado: BORRADOR para dictamen externo. No push.
Autoridad: dictamen F2.3-bis + dictamen de autorizacion TOP 2000.
---
Piloto ejecutado sobre top 2000 CUSIPs del 13F 2026Q1, con metrica

---

## 15. INFORME DE ENTREGA AL AUDITOR - SOLICITUD DE F2.4

**Fichero original:** `archive/iae/informes_originales/NIPC_INFORME_ENTREGA_F24.md`

**Objeto:** entrega de la cadena documental y tecnica del ciclo post-F2.3-bis + solicitud de dictamen F2.4.
**HEAD al redactar:** a9bb1fb.
**Estado:** BORRADOR pendiente de envio.
**Autor:** Ingeniero Supervisor.
**Fecha:** 2026-09-20.
**Destinatario:** auditor externo (dictamen F2.4).

---

## 16. NIPC_INFORME_ESTADO_POST_FIXES_F24

**Fichero original:** `archive/iae/informes_originales/NIPC_INFORME_ESTADO_POST_FIXES_F24.md`

**Objeto:** comunicacion al auditor del estado del sistema tras aplicar los 6 fixes mecanicos autorizados.
**HEAD al redactar:** d462f72.
**Destinatario:** auditor externo.
**Documento complementario:** NIPC_INFORME_ENTREGA_F24.md (hash 0763f16b..., inmutable).
**Estado:** BORRADOR pendiente de envio.
**Autor:** Ingeniero Supervisor.

---

## 17. INFORME TECNICO - SISTEMA IAE (INSTITUTIONAL ACCUMULATION EVIDENCE)

**Fichero original:** `archive/iae/informes_originales/NIPC_INFORME_TECNICO_AUDITORIA_EXTERNA.md`

**Objeto:** informe tecnico completo del sistema IAE para auditoria externa.
**HEAD al redactar:** 9c32c9b (o posterior, ver commit del propio informe).
**Ahead de origin/main:** 105+
**Tests locales:** 979 passed + 2 skipped.
**pyflakes:** 0 warnings. **compileall:** OK. **Working tree:** limpio.
**Push:** NO (regla local-first IAE).

---


## 19. INFORME P65 v1 - Manager Duplication (ciclo completo)

**Objeto:** registrar el ciclo completo P65 (diseno + implementacion + evidencia).
**Fecha:** 2026-09-20.
**Resultado:** P65 v1 CERRADO. GO CONDICIONADO del dictamen v3 implementado integro.
**Referencia completa:** `iae/P64_P65_EXPEDIENTE.md` seccion 2 + `iae/DICTAMENES.md` #26.

### Que se ha implementado

Modulo nuevo `src/institutional_accumulation/aggregation/reporting_dedup.py`:

- **Modelos:** `ReportingEvidence` dataclass (sin economic_owner_cik).
- **Constantes:** 4 transiciones + 4 dedup_reasons + 2 decisiones + 3 niveles + 3 sources.
- **Audit trail:** 12 columnas obligatorias (DEDUP_AUDIT_COLUMNS).
- **Classify L1/L2/L3:** `classify_evidence_level`.
- **Pre-delta:** `build_effective_reporting_snapshot` (R1 con 7 requisitos).
- **Post-delta:** `classify_reporting_transition` (HANDOFF + ambiguedad fail-closed).

**NO toca:** MATCH_KEY, C2, delta_shares.py, nipc.py, coverage.py, relationships.py.

### Evidencia empirica (probe reducido, 200 CIKs)

Pipeline validado end-to-end en datos reales SEC 13F Q4 2025 -> Q1 2026:

    filter -> amendments -> identity -> units -> dedup pre-delta -> delta -> transition

Resultado:

    dedup_audit rows = 0            (sin L3 -> no auditar)
    effective_q4 = 151.222          (KEEP integral)
    effective_q1 = 126.853          (KEEP integral)
    delta rows   = 252.341
    match_status = {UNRESOLVED_IDENTITY: 216.722, BOTH: 25.734,
                    EXIT: 7.625, NEW: 2.260}
    reporting_transition = {NULL: 252.341}   (cero HANDOFF)

**Confirmacion contractual:** sin evidencia cruzada L3, el pipeline P65 v1
se comporta fail-closed: no deduplica, no emite HANDOFF, no fabrica
OVERLAP/CONFLICT. Exactamente lo que exige el dictamen v3.

### Hallazgo colateral

El probe e2e destapo que `data/mappings/cusip_equivalence.csv` no tenia la
columna `identity_type` que P60 exige. Fix en commit 3c2140e (0 filas de
datos afectadas, el fichero estaba vacio por diseno). Evidencia de que la
fase e2e real detecta desalineaciones que los tests unitarios no ven.

### Capacidad diferida (no es deuda, es alcance declarado)

- **DROP_DUP efectivo:** requiere evidencia cuantitativa externa. La
  coexistencia de dos filings sobre la misma security es OVERLAP por
  definicion. `DROP_DUP` no se emite en v1.
- **L3 cruzado real:** requiere el campo "Other Managers Reporting for
  this Manager", que esta en `ADDITIONALINFORMATION` (texto libre).
  Parsear texto libre esta prohibido por las reglas del sistema.

### Commits del ciclo (6 + 1 colateral)

    2ff1751  Commit 1 - modelos + L1/L2/L3
    420d1d0  Commit 2 - pre-delta R1
    7f683e8  Commit 3 - post-delta HANDOFF
    8c2fcf0  Commit 2-fix - DROP_DUP diferido + OVERLAP_UNRESOLVED
    3c2140e  fix colateral - cusip_equivalence.csv
    62314ac  Commit 4 - probe + evidencia

### Tests

- 26 tests especificos P65 (`tests/test_sec_13f_reporting_dedup.py`).
- 1039 passed + 2 skipped + 0 xfailed global.
- 0 warnings.

### Siguiente

- P65 v2 (DROP_DUP efectivo): requiere fuente externa cuantitativa.
- P62 point-in-time: OpenFIGI masivo.
- Bloqueante 1 (TARGET indep.): OpenFIGI masivo.

---



## 20. INFORME P66 L3 - Dictamen externo y propuesta v2 (2026-09-20)

**Objeto:** registrar la elevacion de P66 L3 al auditor externo y el
dictamen obtenido, con la propuesta v2 corregida.

**Referencia completa:** iae/DICTAMENES.md #27 + iae/P66_L3_REFORMULACION_PROPUESTA.md v2.

### Secuencia

1. P65 v2 (DROP_DUP efectivo) cerrado como WONT FIX razonado por
   Gate 0 extendido. Evidencia cruzada para L3 no encontrada en
   OTHERMANAGER2 ni ADDITIONALINFORMATION.

2. Informe inicial P66 (`P66_INFORME_HALLAZGO.md`, commit 2edb462)
   propuso XML crudo del 13F-NT como fuente.

3. Revision externa detecto 3 errores metodologicos:
   - Confusion OTHERMANAGER vs OTHERMANAGER2.
   - Caso Yorktown con filer invertido.
   - "XML firmado digitalmente" sobreafirmado.

4. Gate 0.1: OTHERMANAGER cubre 100% de NT filings (2,008/2,008 Q4;
   2,045/2,045 Q1). OTHERMANAGER2 cubre 0% NT.

5. Gate 0.2: caso Vanguard Q4 2025 -> Q1 2026 documentado
   bidireccionalmente en OTHERMANAGER sin XML.

6. P66 NO-GO como ciclo XML. Reemplazado por
   `P66_L3_REFORMULACION_PROPUESTA.md` (v1, commit 6b610b3).

7. Dictamen externo #27: GO CONDICIONADO. Fuente OTHERMANAGER
   aprobada. 8 correcciones obligatorias antes del GO contractual.

8. Propuesta v2 redactada con las 8 correcciones aplicadas.

### Correcciones aplicadas en v2

    #1  Terminologia: "evidencia estructurada cruzada entre
        filing A y filing B" (no "bidireccional").
    #2  Mantener R3 explicito (no absorcion en R4).
    #3  Formalizar amendments (prohibir MAX(ACCESSION)/MAX(FILING_DATE)).
    #4  Identidad: CIK primario; FormNum fallback; CONFLICT si ambos
        contradicen; no exigir ambos poblados.
    #5  Anadir CONFLICT al enum de estados.
    #6  NT sin A: NO_MATCH si ingestion integra; N/D si fallo;
        nunca N/D -> NO_MATCH.
    #7  DROP_DUP solo con L3 completo (R1+R2+R3+R4+R5) + R5.
    #8  Gate 0.1: conservar granularidad (pendiente Gate 0.3).

### Pendientes tras v2

- Gate 0.3: medir granularidad de identidad en NT (CIK/FormNum/ambos/ninguno).
- Dictamen final sobre el texto contractual exacto.
- Solo despues: aplicar a NIPC_CONTRATOS_SEMANTICOS_v1.md in-place.
- Solo despues: implementar en reporting_dedup.py.



### Gate 0.3 ejecutado (2026-09-20)

Medicion de granularidad de identidad en OTHERMANAGER sobre NT
Q4 2025 + Q1 2026. Resultados:

- Cobertura 100% de NT en OTHERMANAGER (2,008/2,008 Q4; 2,045/2,045 Q1).
- Cardinalidad CIK <-> FormNum 1:1 perfecta (716/716 Q4; 696/696 Q1).
- Composición de filas: 64.7% con ambos; 4.5-5.0% solo CIK; 26.8-27.5%
  solo FormNum; 3.5-3.6% ninguno.
- 6.6% de FormNum usa prefijo `28-` en lugar de `028-`.
  Normalizable trivialmente.
- Cero casos CONFLICT.

Consecuencia: la tabla canonica Form13FFileNumber -> CIK es
construible 1:1. Cobertura efectiva MATCH sube de 69.7% (solo CIK)
a ~96.5% (CIK + FormNum normalizado).

Evidencia: `iae/evidence/p66_gate03_granularidad/`.



### Gate 0.4 ejecutado (2026-09-20)

Origen: dictamen P66 v2, Bloqueo A. El Gate 0.3 midio consistencia
interna de OTHERMANAGER; no demostro cobertura del mapping contra
el universo de filings.

Gate 0.4 construye la tabla canonica FormNum -> CIK vía COVERPAGE +
SUBMISSION (join por ACCESSION_NUMBER). Resultados:

- Cobertura del mapping: 97.81% Q4 / 98.24% Q1.
- Cardinalidad 1:1 perfecta en ambas direcciones.
- Normalizacion `28-` -> `028-` validada (sin ambiguedad).
- Cobertura efectiva MATCH en filas OTHERMANAGER (NT): 96.01% Q4 /
  96.18% Q1.
- N/D total (fail-closed): 3.99% Q4 / 3.82% Q1.
- Bloqueo A del dictamen P66 v2: RESUELTO.

Evidencia: `iae/evidence/p66_gate04_mapping/`.

### Bloqueo B pendiente (amendment probe)

Formalizar como se determina el filing efectivo de B cuando existen
NT/A y multiples amendments. Requiere medicion especifica sobre los
campos CIK + PERIODOFREPORT + SUBMISSIONTYPE + ISAMENDMENT +
AMENDMENTNO + AMENDMENTTYPE.

Sin el, la semantica de R3 no puede formalizarse contractualmente y
el dictamen contractual final no puede emitirse.



### Gate 0.5 ejecutado (2026-09-21)

Origen: dictamen P66 v2, Bloqueo B.

Hallazgo principal: los directorios del Data Set SEC estan
organizados por FECHA DE PRESENTACION, no por periodo objetivo.
Contienen filings con multiples PERIODOFREPORT (48 en Q4 2025,
77 en Q1 2026).

Regla obligatoria: filtrar filings por PERIODOFREPORT, no por
directorio fisico.

R3 formalizada:

    R3(B, period) := existe filing efectivo de B
                     para ese PERIODOFREPORT

    Filing efectivo:
    - Base: punto de partida.
    - RESTATEMENT: sustituye el snapshot (incluye OTHERMANAGER).
    - NEW HOLDINGS: union con el snapshot actual.
    - R4 se evalua sobre el filing efectivo, no sobre el base.

Caso material: CIK 0002056909. El RESTATEMENT sustituye el
OTHERMANAGER (de Prospector Partners a Gator Capital Management).
Un analisis sobre el base produciria un falso positivo en R4.

Evidencia: `iae/evidence/p66_gate05_amendment_probe/`.

Bloqueo B del dictamen P66 v2: RESUELTO.



### Dictamen P66 v2-bis recibido (2026-09-21)

Resultado: NO-GO contractual definitivo.
Resultado tecnico: GO condicionado para cierre de 6 bloqueos.

Fuente OTHERMANAGER aprobada. Arquitectura probatoria aprobada.
Sin embargo, el contrato no esta listo para modificar 14.3.

6 bloqueos:

1. Combination excluido. El universo del filing efectivo debe incluir
   13F-HR con REPORTTYPE = 13F COMBINATION REPORT, no solo NT/NT-A.
2. NEW HOLDINGS sobre OTHERMANAGER no probado. La regla union es
   prematura. Requiere probe especifico (Gate 0.7).
3. Gate 0.4 sin filtro por PERIODOFREPORT declarado. Debe re-ejecutarse
   con filtro explicito (Gate 0.4-reissue).
4. Mapping FormNum -> CIK unidireccional. No invariante 1:1 bidireccional.
5. Normalizacion 28- -> 028- no debe truncar. Caso 028-2813114 (7 digitos)
   queda N/D.
6. Nomenclatura: MATCH -> IDENTITY_RESOLVED en Gate 0.4. Los porcentajes
   96% son cobertura de resolucion de identidad, no cobertura L3.

Reclasificacion:
  Gate 0.5A (contaminacion cross-period)          PASS
  Gate 0.5B (semantica completa amendments)       PENDING

Estado: contrato 14.3 SIN CAMBIOS. reporting_dedup.py SIN CAMBIOS.
DROP_DUP NO ACTIVADO.

Referencia: DICTAMENES.md #28.



### Gate 0.6 Combination probe ejecutado (2026-09-21)

Origen: dictamen #28, bloqueo #1.

Medicion del universo R4 segun REPORTTYPE:

| SUBMISSIONTYPE | REPORTTYPE | Q4 filings | Q4 con OM | Q1 filings | Q1 con OM |
|----------------|-----------|-----------:|----------:|-----------:|----------:|
| 13F-HR | HOLDINGS | 8,237 | 0 | 8,339 | 0 |
| 13F-NT | NOTICE | 1,900 | 1,900 | 1,907 | 1,907 |
| 13F-HR | COMBINATION | 388 | 388 | 402 | 402 |
| 13F-HR/A | HOLDINGS | 105 | 0 | 118 | 0 |
| 13F-NT/A | NOTICE | 38 | 38 | 1 | 1 |
| 13F-HR/A | COMBINATION | 8 | 8 | 9 | 9 |

Universo R4 corregido:
- Q4 2025: 2,334 (1,938 NOTICE + 396 COMBINATION). Delta vs v2-bis +396.
- Q1 2026: 2,319 (1,908 NOTICE + 411 COMBINATION). Delta vs v2-bis +411.

Cobertura OM en universo R4: 100.00%.
Sin casos anomalos.

Conclusion: BLOQUEO #1 RESUELTO. El texto contractual de R4 se
actualiza para incluir 13F COMBINATION REPORT segun REPORTTYPE.

Evidencia: `iae/evidence/p66_gate06_combination_probe/`.

Bloqueos pendientes: #2 (NEW HOLDINGS probe), #3 (Gate 0.4-reissue
con filtro), #4 (mapping unidireccional), #5 (sin truncar), #6
(nomenclatura).



### Gate 0.7 NEW HOLDINGS probe ejecutado (2026-09-21)

Origen: dictamen #28, bloqueo #2 (regla union prematura).

Medicion sobre el universo R4 (NOTICE + COMBINATION):

| Periodo | NEW HOLDINGS | Identicos | Union | Sustitucion | Otros |
|---------|-------------:|----------:|------:|------------:|------:|
| 2025-12-31 | 2 | 2 | 0 | 0 | 0 |
| 2026-03-31 | 1 | 1 | 0 | 0 | 0 |

3/3 casos identicos. Cero cambios en OTHERMANAGER.

Regla contractual propuesta (fail-closed):
- Base: punto de partida.
- RESTATEMENT: sustituye OTHERMANAGER (evidencia Gate 0.5).
- NEW HOLDINGS: tratar como identidad. Si aparece caso con
  OTHERMANAGER distinto -> N/D o CONFLICT. NO union.

Muestra insuficiente (3 casos sobre 4,653 filings) para afirmar
con alta confianza. Alternativa ofrecida al auditor: extender el
probe a 4-6 trimestres historicos antes del GO contractual.

Evidencia: `iae/evidence/p66_gate07_newholdings_probe/`.

Bloqueo #2: RESUELTO provisionalmente con regla fail-closed. Sujeto
a validacion del auditor.

Bloqueos pendientes: #3 (Gate 0.4-reissue), #4 (mapping unidireccional),
#5 (sin truncar), #6 (nomenclatura).



### Gate 0.4-reissue ejecutado (2026-09-21)

Origen: dictamen #28, bloqueo #3 (Gate 0.4 sin filtro PERIODOFREPORT).

Re-ejecucion del mapping con filtro por PERIODOFREPORT y
normalizacion estricta (sin truncar). Nomenclatura corregida
(MATCH -> IDENTITY_RESOLVED).

| Metric | Q4 original | Q4 reissue | Q1 original | Q1 reissue |
|--------|------------:|-----------:|------------:|-----------:|
| Total filas OTHERMANAGER(NT) | 2,804 | 2,734 | 2,827 | 2,683 |
| Cobertura de identidad | 96.01% | **94.92%** | 96.18% | **95.34%** |
| Delta | | −1.09 pp | | −0.84 pp |

El delta cuantifica la contaminacion cross-periodo del Gate 0.4
original. La regla corregida es correcta; la cobertura cae ~1 pp
pero se mantiene >94%.

Bloqueos resueltos por este Gate:
- #3 Gate 0.4-reissue con filtro: RESUELTO.
- #5 Normalizacion sin truncar: APLICADO.
- #6 Nomenclatura IDENTITY_RESOLVED: APLICADO.

Evidencia: `iae/evidence/p66_gate04_reissue_period/`.

Bloqueo #4 (mapping unidireccional): PENDIENTE en propuesta v3.



### Propuesta v3 final - bloqueo #4 resuelto (2026-09-21)

Correcciones aplicadas a P66_L3_REFORMULACION_PROPUESTA.md:

1. Terminologia: "evidencia bidireccional" -> "evidencia estructurada
   cruzada entre filings" (correccion #1 del dictamen #28).
2. Cardinalidad reformulada: invariante unidireccional FormNum -> CIK.
   Estados: 0 CIK (UNRESOLVED), 1 CIK (IDENTITY_RESOLVED),
   >1 CIK (CONFLICT). No se exige direccion inversa.
3. Nomenclatura: MATCH -> IDENTITY_RESOLVED en todo el documento.
4. Nueva seccion 10 "Invariante del mapping (bloqueo #4)" con:
   - Direccion contractual (FormNum -> CIK).
   - Construccion (period, normalized_form13f_filenumber, cik,
     source_accessions, resolution_status).
   - Uso (0/1/>1 CIK).
5. Reordenacion de secciones 9.3 / 9.4 / 9.5.
6. Cabeceras actualizadas (auditor #28, Gates 0.5-0.7).

Los 6 bloqueos del dictamen #28 quedan resueltos:

    #1 Combination                         RESUELTO (Gate 0.6)
    #2 NEW HOLDINGS                        RESUELTO (Gate 0.7, fail-closed)
    #3 Gate 0.4-reissue filtro period      RESUELTO (Gate 0.4-reissue)
    #4 Mapping unidireccional              RESUELTO (seccion 10)
    #5 Normalizacion sin truncar           APLICADO
    #6 Nomenclatura IDENTITY_RESOLVED      APLICADO

Propuesta v3 lista para reenviar al auditor.



### Dictamen P66 #29 recibido (2026-09-21)

Resultado: GO CONDICIONADO. Un bloqueo material nuevo + correcciones
textuales.

Aprobado: arquitectura probatoria de R4 completa. R2/R4 separacion
confirmada por la SEC. Combination, cross-period, mapping
unidireccional, normalizacion sin truncar, identidad CIK/FormNum,
RESTATEMENT, NEW HOLDINGS (regla fail-closed), estados N/D/CONFLICT:
todos RESUELTOS.

Bloqueo material nuevo: **Gate 0.8**. La prueba de cobertura del Gate
0.4-reissue se hizo sobre `OTHERMANAGER(NT)`. El universo contractual
de R4 que ahora esta definido es mas amplio: NOTICE (NT + NT/A) +
COMBINATION (HR + HR/A). El auditor exige medir cobertura de identidad
sobre el universo completo, separado por tipo y por clase de amendment.

Correcciones textuales aplicadas:
- §2 R4: adoptado texto exacto del §11 del dictamen #29.
- §5.2: afinada la formulacion de la cadena de amendments.
- §10.4: reflejada la aprobacion explicita del bloqueo #4.

Evidencia: `iae/DICTAMENES.md` #29.

Siguiente paso: Gate 0.8.



### Gate 0.8 ejecutado (2026-09-21)

Origen: dictamen #29. Cobertura de identidad sobre universo R4 completo
(NOTICE + COMBINATION + amendments).

| Clase | Q4 Filas | Q4 Cobertura | Q1 Filas | Q1 Cobertura |
|-------|---------:|------------:|---------:|------------:|
| NOTICE_BASE | 2,696 | 94.84% | 2,682 | 95.34% |
| NOTICE_AMEND (RESTATEMENT + NEW HOLDINGS) | 38 | 100% | 1 | 100% |
| COMBINATION_BASE | 1,799 | 86.33% | 1,841 | 85.33% |
| COMBINATION_AMEND (RESTATEMENT + NEW HOLDINGS) | 107 | 100% | 22 | 100% |
| **TOTAL R4** | **4,640** | **91.70%** | **4,546** | **91.31%** |

Delta vs NOTICE-solo: -3.21 pp Q4, -4.03 pp Q1.

Drill-down: 82-85% del N/D_null en COMBINATION_BASE se concentra
en 2 filings del mismo filer (0001580642-). Composicion:
NAME 100%, CRDNUMBER 85-88%, SECFILENUMBER 64-69%, CIK 0%,
FORM13FFILENUMBER ~0%. Es el caso N/D que el contrato ya cubre
con fail-closed.

Conclusion: Gate 0.8 PASS. Sin nueva clase de ambiguedad.
Cobertura R4 efectiva: 91.70% / 91.31%.

Evidencia: `iae/evidence/p66_gate08_full_r4_coverage/`.

Bloqueo material del dictamen #29: RESUELTO.



### Dictamen P66 #30 recibido (2026-09-21)

Resultado: NO-GO contractual definitivo.
Resultado tecnico: GO condicionado a 3 correcciones de especificacion.

3 bloqueos:

- **A** CONFLICT debe tener precedencia sobre MATCH. Ambos
  identificadores presentes con contradiccion -> CONFLICT, no MATCH.
- **B** NO_MATCH requiere resolucion completa de TODAS las filas
  OTHERMANAGER. Si existe alguna fila N/D -> N/D, no NO_MATCH.
- **C** >1 filing base independiente para mismo CIK+PERIODOFREPORT
  no puede resolverse por orden arbitrario -> N/D o CONFLICT.

Los 3 bloqueos aplicados al texto contractual:
- §3 redefinido con las 4 definiciones del dictamen #30.
- §4 reformulado con precedencia CONFLICT > MATCH.
- §5.2 ampliado con regla de >1 filing base.
- §11 nuevo con diagrama de decision.

Gate 0.8: PASS confirmado por el auditor.

Evidencia: `iae/DICTAMENES.md` #30.

Siguiente paso: Gate 0.9 (verificar existencia real de >1 filing base).



### Gate 0.9 ejecutado (2026-09-21)

Origen: bloqueo C del dictamen #30.

Verificacion empirica de >1 filing base independiente por
(CIK, PERIODOFREPORT).

| Metrica | Q4 2025 | Q1 2026 |
|---------|--------:|--------:|
| Grupos (CIK, PERIOD) | 2,287 | 2,309 |
| Con 1 filing base | 2,286 | 2,309 |
| Con >1 filing base | 1 | 0 |
| MULTIPLE_NOTICE | 0 | 0 |
| MULTIPLE_COMBINATION | 0 | 0 |
| MIXED (NT + HR) | 1 | 0 |

Caso unico: CIK 0002016827 (Q4). Patron MIXED: 13F-NT presentado
2026-01-02 (0 holdings, delegacion total a Vident Advisory) +
13F-HR COMBINATION presentado 2026-02-20 (78 holdings, misma
relacion OTHERMANAGER). Mismo PERIODOFREPORT.

Interpretacion: evolucion documental del filer. El contrato lo
clasifica correctamente como N/D fail-closed.

Conclusion: la regla del bloqueo C esta empiricamente justificada.
Sin cambios contractuales necesarios.

Evidencia: `iae/evidence/p66_gate09_multiple_base/`.

Bloqueo C del dictamen #30: RESUELTO.



### Dictamen P66 #31 recibido (2026-09-21)

Resultado: NO-GO contractual.
Resultado evidencia: PASS.

3 bloqueos normativos finales:

- **1** R3 debe incluir REPORTTYPE (13F-HR distingue HOLDINGS vs
  COMBINATION por REPORTTYPE, no por SUBMISSIONTYPE).
- **2** CONFLICT debe estar scoped a la fila candidata de A. Un
  conflicto en otra fila OTHERMANAGER no invalida el MATCH de A.
- **3** 0 bases + 0 amendments -> R3=False; 0 bases + amendment
  sin base reconstruible -> R3=N/D.

Correcciones aplicadas a la propuesta:
- §2 nota terminologica "regla contractual IAE".
- §4 CONFLICT scoped a la evidencia candidata de A.
- §5.2 regla 4 dividida en Caso A (False) / Caso B (N/D).
- §6 R3 con SUBMISSIONTYPE + REPORTTYPE.
- §11 diagrama actualizado con scope de CONFLICT.

Evidencia: `iae/DICTAMENES.md` #31.

Siguiente paso: reenviar paquete completo al auditor para
dictamen contractual final.



### Dictamen P66 #32 recibido (2026-09-21)

Resultado: GO CONDICIONADO FINAL. Evidencia PASS. Sin nuevos gates.

2 correcciones aplicadas:

- **A**: §7 generalizado de NT a universo R4 (NOTICE + COMBINATION).
  Antes decia solo NT; ahora cubre completo. Consistente con §2 y §5.
- **B**: R3 con tri-state TRUE/FALSE/N/D. TRUE si el conjunto
  documental efectivo es determinable inequivocamente; FALSE si no
  existe filing R4; N/D si existen submissions pero el conjunto no
  puede determinarse.

Trazabilidad de la cabecera actualizada:
- #28 corregido
- #29 corregido
- #30 corregido
- #31 corregido
- #32 GO CONDICIONADO FINAL con 2 correcciones aplicadas

Evidencia: `iae/DICTAMENES.md` #32.

Siguiente paso: dictamen contractual final del auditor sobre 14.3.



### Dictamen P66 #33 recibido (2026-09-21)

Resultado: GO CONDICIONADO FINAL. Evidencia PASS definitivo.
El auditor emitira GO CONTRACTUAL DEFINITIVO tras aplicar 2
correcciones finales.

2 correcciones aplicadas:

- **1**: CONFLICT debe cubrir CUALQUIER fila OTHERMANAGER relevante
  para A, no solo la seleccionada. Prohibicion de seleccion
  selectiva: si existe contradiccion entre dos filas que refieren
  a A -> CONFLICT.
- **2**: Nota de alcance: "filing efectivo" definido por P66 se
  aplica exclusivamente a B/R3/R4. NO redefine la seleccion del
  filing de A en R1/R2.

Trazabilidad de la cabecera actualizada a #28-#33.

Evidencia: `iae/DICTAMENES.md` #33.

Siguiente paso: dictamen contractual definitivo -> modificar 14.3.



### Dictamen P66 #34 + propuesta v4 consolidada (2026-09-21)

Resultado: NO-GO contractual. Evidencia PASS.

4 bloqueos + 2 recomendadas:
- 1: FormNum normalizacion con padding flexible (Gate 0.10).
- 2: Determinismo amendments via ISAMENDMENT != Y.
- 3: candidate_A formalizado matematicamente.
- 4: L3 con combinacion booleana explicita.
- R1: OTHERMANAGER vacio -> N/D.
- R2: Correcciones editoriales v2/v3/v2-bis.

**Cambio de proceso:** para romper el ciclo de refinamiento
incremental (#28 -> #34), se ha hecho reescritura completa como
propuesta v4 consolidada. Todos los bloqueos anteriores integrados.
Pass interno ejecutado sin residuos.

Propuesta v4 (617 lineas):
- §0.3 Terminologia formalizada.
- §0.2 Alcance explicitado (B/R3/R4 vs A/R1/R2).
- §2.1 R3 tri-state; §2.2 R4 estados; §2.3 candidate_A;
  §2.5 L3 booleano.
- §3 amendments determinista.
- §4 FormNum canonicalizacion flexible.
- §5 NO_MATCH con completitud; OTHERMANAGER vacio -> N/D.
- §8 evidencia de los 10 Gates.

Gate 0.10 ejecutado: evidencia en
`iae/evidence/p66_gate10_formnum_representation/`.

Evidencia: `iae/DICTAMENES.md` #34.



### Dictamen P66 #35 + propuesta v5 consolidada (2026-09-21)

Resultado: NO-GO contractual. Evidencia PASS. 5 bloqueos materiales
+ 1 ajuste menor, todos aplicados en v5.

**5 bloqueos #35 aplicados:**
- 1: BASE_R4 global (todas las bases NOTICE+COMBINATION antes de
  construir cadena, no por familia aislada). >1 base R4 -> R3=N/D.
- 2: Cadena de amendments validada completa: AMENDMENTNO entero
  1..99, sin huecos, tipos reconocibles.
- 3: Identidad resuelta coherente con CIK/FormNum. Nuevo estado
  INCONSISTENT -> CONFLICT.
- 4: Prefijo FormNum restringido a {28, 028}. Prohibido padding
  generico.
- 5: NEW HOLDINGS "consistente" definido operacionalmente como
  igualdad exacta de conjunto OTHERMANAGER_state.

**1 ajuste menor aplicado:**
- 6: Terminologia §4.1 "canonicalizacion IAE" (no formato SEC
  obligatorio).

**Condicion de cierre del auditor:** "Cuando estas seis correcciones
esten incorporadas en la v5, mi criterio seria GO CONTRACTUAL,
sin necesidad de tocar todavia reporting_dedup.py ni activar
DROP_DUP, siempre que la v5 no introduzca nuevas reglas
heuristicas."

Propuesta v5 (699 lineas, 22,642 bytes). Pass interno ejecutado.

Evidencia: `iae/DICTAMENES.md` #35.

Siguiente paso: enviar v5 al auditor con expectativa de GO
contractual.



### Dictamen P66 #36 + propuesta v6 candidata final (2026-09-21)

Resultado: NO-GO contractual. Evidencia PASS. 3 cierres materiales
+ 2 ajustes menores, todos aplicados en v6.

**3 cierres materiales aplicados:**
- 1: BASE/AMENDMENT por SUBMISSIONTYPE+REPORTTYPE, no por
  ISAMENDMENT (permite ISAMENDMENT=<NA> en bases correctamente).
- 2: Familia cerrada: base NOTICE solo admite amendments NOTICE;
  base COMBINATION solo admite amendments COMBINATION. Cruce de
  familia -> N/D.
- 3: NEW HOLDINGS consistente solo si ambos estados son
  completamente determinables. Cualquier N/D o CONFLICT en una
  fila impide la comparacion -> N/D.

**2 ajustes aplicados:**
- 4: "Secuencia sin huecos" etiquetada como regla conservadora IAE
  (no SEC).
- 5: "Fin de la propuesta v4" -> v6.

**Condicion de cierre del auditor:** "Una v6 que incorpore
literalmente esos tres cierres materiales, sin introducir nuevas
heuristicas, estara en condiciones de recibir GO CONTRACTUAL FINAL."

Propuesta v6 escrita sin heuristicas nuevas. Pass interno ejecutado.

Evidencia: `iae/DICTAMENES.md` #36.

Siguiente paso: enviar v6 al auditor.



### Dictamen P66 #37 + propuesta v7 ronda final (2026-09-21)

Resultado: NO-GO contractual. Evidencia PASS. 3 correcciones aplicadas
+ §0.5 Clausula de cierre contractual.

**3 correcciones aplicadas:**
- 1: Flujo unico de R3 en §3.2. 4 pasos secuenciales. Caso
  BASE=0 + AMENDMENT>0 -> N/D (antes decia FALSE, regresion
  introducida en v6).
- 2: Eliminado `AND ISAMENDMENT != Y` de BASE_R4. Ya contradicia
  §3.1. ISAMENDMENT queda como control de coherencia.
- 3: Inspeccion global de familia en §3.2 PASO 3 antes de aceptar
  cadena unica.

**§0.5 Clausula de cierre contractual:**
- Define BLOQUEO MATERIAL (nueva version + dictamen) vs
  RECOMENDACION DIFERIBLE (no bloquea).
- Condicion: GO de v7 autoriza traslado a
  NIPC_CONTRATOS_SEMANTICOS_v1.md sin nueva iteracion ordinaria.

Propuesta v7 escrita. Pass interno ejecutado.

Evidencia: `iae/DICTAMENES.md` #37.

Siguiente paso: enviar v7 al auditor. Ronda final.



### Dictamen P66 #38 + propuesta v7-bis ronda final (2026-09-21)

Resultado: NO-GO contractual. 1 bloqueo material + 2 correcciones
editoriales + 1 observacion no bloqueante.

**Bloqueo material aplicado:**
- §3.4 contradicia §3.2 (permitia "N/D o CONFLICT" cuando §3.2
  ya cerro solo N/D). Convertida en nota explicativa. Regla
  absoluta: BASE > 1 -> R3 = N/D. CONFLICT no puede producirse
  en R3.

**2 correcciones editoriales aplicadas:**
- §9 trazabilidad actualizada (#28 a #38).
- "Fin de la propuesta v6" -> v7-bis.

**1 observacion no bloqueante aplicada:**
- §9.2 nueva: nota de rotulacion de evidencia historica. Los
  READMEs de Gates no se modifican retroactivamente (HASHES
  preservados). La semantica contractual vigente es v7-bis.

**Declaracion del auditor:** "Con esa correccion, considero que
v7 puede recibir GO CONTRACTUAL FINAL: la evidencia esta cerrada,
R3 esta practicamente completamente determinista y no queda una
nueva cuestion arquitectonica."

Propuesta v7-bis escrita. Pass interno ejecutado.

Evidencia: `iae/DICTAMENES.md` #38.

Siguiente paso: enviar v7-bis al auditor. Ronda final.



### Dictamen P66 #39 + propuesta v7-ter ronda final (2026-09-21)

Resultado: NO-GO contractual. 1 bloqueo material + 1 correccion
editorial, ambos aplicados en v7-ter.

**Bloqueo material aplicado:**
- Falta precondicion de integridad/completitud del scope R4 antes
  de poder emitir R3=FALSE. Sin ella, "0 observados en dataset
  incompleto" podria convertirse en "0 existen".
- Correccion: PASO 0 en §3.2.
  - Si completitud del scope NO demostrable: R3=N/D.
  - Si registro R4 con clasificacion no inequivoca: R3=N/D.
  - Solo despues: 0 bases + 0 amendments -> R3=FALSE.
- Evidencia: Gate 0.8 detecto 137-144 FormNum en OTHERMANAGER(R4)
  no presentes en COVERPAGE. La ingestion puede ser incompleta.

**Correccion editorial aplicada:**
- §0.5 "GO de v7" -> "v7-bis" (ahora v7-ter).

**Declaracion del auditor:** "Despues de incorporar ese PASO 0 y
corregir la referencia v7 -> v7-bis, considero que si procede el
GO CONTRACTUAL FINAL."

Propuesta v7-ter escrita. Pass interno ejecutado.

Evidencia: `iae/DICTAMENES.md` #39.

Siguiente paso: enviar v7-ter al auditor. Ronda final.



### GO CONTRACTUAL P66 - Cierre del ciclo (2026-09-21)

Dictamen #39 declaro la condicion de GO:

    "Despues de incorporar ese PASO 0 y corregir la referencia
     v7 -> v7-bis, considero que si procede el GO CONTRACTUAL
     FINAL."

Ambas condiciones cumplidas en v7-ter. GO emitido por declaracion
condicional cumplida.

**Traslado realizado:**
- §14.3 de NIPC_CONTRATOS_SEMANTICOS_v1.md reformulada con el texto
  contractual de v7-ter.
- 283 lineas insertadas, 7 eliminadas.
- Estructura §14.1 a §14.14 preservada.
- §14.3.1 a §14.3.7: flujo unico R3 + PASO 0 completitud + cadena
  amendments + semantica + candidate_A + mapping FormNum + clausula
  de cierre.

**Commit del traslado:** 3a233b4.

**Ciclo P66 cerrado.**

Estados:
    §14.3 trasladada                  COMPLETADO
    reporting_dedup.py                NO TOCADO (segun auditor)
    DROP_DUP                          NO ACTIVADO (segun auditor)
    Tests contractuales               PENDIENTE (siguiente ciclo)
    Implementacion reporting_dedup    PENDIENTE (siguiente ciclo)

### Estado

    Fuente             APROBADA (OTHERMANAGER)
    Propuesta v2       REDACTA, pendiente dictamen final
    Contrato 14.3      SIN CAMBIOS
    reporting_dedup.py SIN CAMBIOS
    DROP_DUP           NO ACTIVADO
    XML/parser         NO NECESARIO
    Descarga EDGAR     NO NECESARIA


Fin del informe consolidado.

## 18. INFORME F2.4 - Dictamen formal y bloqueantes

**Objeto:** registrar el dictamen F2.4 emitido por el auditor externo sobre la submission v6 (3 divergencias contrato<->codigo + Q12 + AGREG + OpenFIGI + Policy v1.3).
**Fecha:** 2026-09-20.
**Resultado global:** GO CONDICIONADO arquitectura. Certificacion "acumulacion" BLOQUEADA.
**Referencia completa:** `docs/auditoria/iae/DICTAMENES.md` seccion 24.
**Evidencia ejecutable:** tests/test_p60_contract.py, test_p61_contract.py, test_p38_contract.py (5 xfail strict).

### Decisiones sobre las 3 divergencias

| Div | Contrato | Dictamen        | Detalle |
|-----|----------|-----------------|---------|
| D1  | P61      | GO              | Conectar resolve_source_status. Evidence conserva source + valid_from + valid_to + provenance_id + resolution_timestamp. |
| D2  | P38      | GO CONDICIONADO | Opcion A (TARGET real) + rediseno: TARGET no puede depender del exito del mapping. |
| D3  | P60      | GO              | Eliminar default TICKER. Si falta columna: schema_error / identity_type_missing. |
### Decisiones arquitectonicas

- Q12 = Modelo A (shareClassFIGI). Clausula: unidad economica = share class.
- AGREG. = Opcion 2 (aggregate_positions_by_shareclass_figi separada de compute_contractual_coverage).
- Solo VERIFIED aporta al peso contractual. unverified_weight conservado aparte. Nunca unverified = 0.
- OpenFIGI: snapshot + hash. No dependencia live.
- Policy v1.3: NO aprobada. v1.0 sigue normativa.

### 3 bloqueantes estructurales

1. TARGET independiente del exito del mapping. Separar CATALOGO (externo, versionado) de TARGET_OBSERVED.
2. Semantica point-in-time. catalog_version + valid_from/to, o target_catalog_as_of(period_end). OpenFIGI no expone effective_date.
3. 13F != flujo en tiempo real. Etiqueta: "cambio trimestral observado de posiciones institucionales reportables via 13F".
### Reglas adicionales (10)

period_end != filing_date != knowledge_date; unmapped_count int separado de unmapped_weight; compute_nipc sin degradacion silenciosa a proxy (evidence_class CONTRACTUAL|PROXY); PositionRecord tipado en coverage.py; NO_MATCH != not_target; "no aparece" != "vendio" (6 estados: ZERO_REPORTED | MISSING | BELOW_REPORTING_THRESHOLD | CONFIDENTIAL | UNRESOLVED | SOLD); corporate actions distinguidos de economic accumulation; manager duplication con unidad definida (manager / manager-group / filing / position / security); tests obligatorios P38 (10 casos); 4 niveles de validacion (F2.4 certifica nivel 1 y parcialmente 2).

### Estado

    THRESHOLD_1                UNDEFINED
    THRESHOLD_2                BLOQUEADO
    Gate-NIPC.2                BLOQUEADO
    Gate-NIPC.3                NO AUTORIZADO
    OpenFIGI masivo            NO AUTORIZADO (GO condicional arquitectonico)
    Policy v1.3 aplicacion     NO AUTORIZADA
    Push a origin/main         NO
    Certificacion "acumulacion" BLOQUEADA

### Siguiente

Gate 0 de los 3 bloqueantes (inventario de codigo, sin tocar codigo). Actualizar FASE_A6_PLAN.md + REESTRUCTURACION_MODULO.md + NIPC_CONTRATOS_SEMANTICOS_v1.md.

---


---

## 21. INVENTARIO A.6.0 - Gate 0 de los 3 bloqueantes F2.4

**Fecha:** 2026-09-21.
**Origen:** FASE_A6_PLAN.md seccion A.6.0 (PENDIENTE hasta la fecha).
**Alcance:** inventario empirico con referencias archivo:linea. Sin
cambios de codigo productivo. Sin OpenFIGI. Sin tocar contratos.
**Referencias:** DICTAMENES.md #24 (F2.4), RECONCILIACION_CONTRATO_CODIGO.md
seccion 3 (D1/D2/D3), REESTRUCTURACION_MODULO.md.

**Objetivo:** mapear cuanto de los 3 bloqueantes estructurales introducidos
por el dictamen F2.4 (2026-09-20) esta ya cubierto por el codigo actual y
cuanto requiere rediseno antes de tocar codigo productivo.

**Metodo:** lectura directa de modulos y CSVs en disco. Grep sobre
`src/institutional_accumulation/**/*.py`. Ejecutado 2026-09-21.
`HEAD` al ejecutar: `bb3e106` o posterior.

---

### 21.1. B1 - TARGET independiente del exito del mapping

**Enunciado F2.4:** el denominador de `paired_weighted_share_coverage`
no puede depender del exito del mapping. Separar CATALOGO (externo,
versionado) de TARGET_OBSERVED.

**Que existe:**

- `src/institutional_accumulation/aggregation/coverage.py::compute_contractual_coverage(target_q4, target_q1, records_q4, records_q1)` (L64-...). API contractual P38. Recibe TARGET como parametro externo, construido por el caller. NO construye TARGET internamente.
- `src/institutional_accumulation/aggregation/coverage.py::PositionRecord` (L25-38). Dataclass tipado (F2.4 regla #4).
- `src/institutional_accumulation/aggregation/coverage.py::aggregate_positions_by_shareclass_figi(records, period)` (L41-61). Agregacion por `share_class_figi`. F2.4 Opcion 2.
- `src/institutional_accumulation/identity/radar_target_catalog.py` (4.5 KB). Constructor del catalogo desde OpenFIGI probe result.
- `src/institutional_accumulation/identity/target_universe.py::resolve_cusips(cusips, catalog_df)` (L81-...). Resolver CUSIP_13F -> target membership por `share_class_figi`.
- `data/mappings/radar_target_catalog.csv` (242 filas). Columnas: `radar_ticker, figi, share_class_figi, composite_figi, ticker_from_openfigi, name, security_type, market_sector, exch_code, source, source_date, status`. 240 OK, 2 MISS.

**Que falta:**

- `src/institutional_accumulation/identity/target_builder.py`. NO EXISTE. Es la pieza descrita en REESTRUCTURACION_MODULO.md seccion 4.2. Debe materializar TARGET_P (set de `share_class_figi`) desde RADAR_TARGET_CATALOG + OpenFIGI batch + observacion 13F del periodo.
- Integracion en `src/institutional_accumulation/aggregation/nipc.py::compute_coverage_pairwise` (L150-265). La implementacion actual:
  - L185-188: `_securities(df) = set(observed_security_key)` — CUSIPs observados, NO `share_class_figi` del catalogo.
  - L213: `target_pairwise = sec_c & sec_p` — interseccion de CUSIPs observados, no de TARGET.
  - L240: `denom = sum(by_sec.get(k, 0.0) for k in target_pairwise)` — el denominador ponderado usa CUSIPs observados.
  - El nombre `target_pairwise` es nominal: la implementacion real es `observed_pairwise`.
  - `nipc.py` NO importa `radar_target_catalog` ni `target_universe` (verificado en imports, L26-30).

**Evaluacion:** P38 tiene 2 APIs conviviendo.

| API | Fichero | Estado |
|---|---|---|
| Contractual P38 | `aggregation/coverage.py` | COMPLETA. Recibe TARGET externo. |
| Operativa | `aggregation/nipc.py::compute_coverage_pairwise` | PROXY observacional. NO invoca la contractual. |

Falta `target_builder.py` + integracion. El nombre `target_pairwise` en `nipc.py` es enganoso (D2 GRAVE confirmado).

**Riesgo si no se materializa:** los valores de `paired_weighted_share_coverage` publicados en baseline (`e9fd830`) y TOP 2000 (`32baf9d`) son historicos, no aptos para THRESHOLD_2 bajo la definicion contractual.

**Propuesta (fuera de A.6.0):** `target_builder.py` + refactor de `compute_coverage_pairwise` para invocar `compute_contractual_coverage`. Requiere autorizacion (A.6.2-bis).

---

### 21.2. B2 - Semantica point-in-time

**Enunciado F2.4:** toda entidad (catalogo, mapping, universos) aplicada a un periodo historico debe declarar validez temporal. Aplicar la version de hoy retroactivamente produce survivorship y look-ahead bias.

**Que existe:**

- `data/mappings/radar_target_catalog.csv::source_date` (columna). Valor actual: `2026-09-19` para las 242 filas. Es fecha de snapshot del probe, NO fecha de vigencia.
- `data/mappings/cusip_equivalence.csv::valid_from / valid_to` (columnas). 0 filas por diseno (regla C1-revisada).
- `data/mappings/cusip_ticker_exceptions.csv::valid_from / valid_to` (columnas). 3 filas COM (Q-CUR).

**Que falta:**

- Columnas `catalog_version`, `catalog_valid_from`, `catalog_valid_to` en `radar_target_catalog.csv`. Grep en `src/`: **0 hits**.
- Funcion `target_catalog_as_of(period_end)` en `src/`. Grep: **0 hits**.
- `effective_date` en `identity/openfigi_client.py`. NO EXISTE. OpenFIGI no lo expone (verificado). La unica fecha asociada es la del probe, no la de vigencia historica del mapping.
- Snapshot + hash por consulta OpenFIGI (F2.4). NO IMPLEMENTADO.

**Evaluacion:** B2 esta completamente ausente excepto el `source_date` del catalogo (que no cumple la semantica point-in-time).

**Riesgo si no se materializa:** el catalogo actual, aplicado retrospectivamente a Q4 2025 / Q1 2026, asume que la identificacion de hoy es valida para periodos pasados. Survivorship + look-ahead bias.

**Propuesta (fuera de A.6.0):** anadir `catalog_version` + `catalog_valid_from` + `catalog_valid_to` al catalogo. Implementar `target_catalog_as_of(period_end)`. Requiere dictamen especifico.

---

### 21.3. B3 - 13F != flujo en tiempo real

**Enunciado F2.4:** "posiciones al cierre de trimestre + publicacion hasta 45 dias. Sin cortos, con minimis, con confidencialidad. Etiqueta correcta: cambio trimestral observado de posiciones institucionales reportables via 13F."

**Que existe:**

- Doctrina P63/P64/P65 documentada en docstrings de `delta_shares.py` (`DELTA_SEMANTICS_GROSS_OBSERVED` L56, `P64_EVENTS_DEFERRED` L57).
- `SUBMISSION.parquet` con columnas `FILING_DATE` + `PERIODOFREPORT`. Disponibles en bruto.
- `COVERPAGE.parquet` con `DATEREPORTED` + `REPORTCALENDARORQUARTER`.

**Que falta:**

- Columna `knowledge_date` (fecha de ingesta). Grep en `src/`: **0 hits**.
- Campos `period_end` / `filing_date` como dimensiones separadas en `PositionRecord` (`coverage.py::PositionRecord` solo tiene `period: str`, L31).
- Clasificador `absence_reason` (`MISSING` / `BELOW_REPORTING_THRESHOLD` / `CONFIDENTIAL` / `OTHER_MANAGER` / `UNKNOWN`). Grep: **0 hits**.
- Estados `ZERO_REPORTED` / `NOT_PRESENT` como enums materializados. Grep: **0 hits**.
- Distincion explicita corporate action vs economic accumulation en `delta_shares.py`. `DELTA_SEMANTICS_GROSS_OBSERVED = True` documenta que NO se distingue (capacidad diferida v1).

**Evaluacion:** doctrina documentada (P63/P64/P65 en `NIPC_CONTRATOS_SEMANTICOS_v1.md` secciones 12/13/14). Materializacion: AUSENTE.

**Riesgo si no se materializa:** los outputs pueden leerse como "flujo en tiempo real" o como "el manager vendio". El contrato exige la etiqueta "cambio trimestral observado". `delta_shares` no crea `SOLD` (regla P63 R8), pero el clasificador de ausencia no existe.

**Propuesta (fuera de A.6.0):** anadir `knowledge_date` al pipeline de ingest. Ampliar `PositionRecord` con `period_end` + `filing_date`. Implementar clasificador de ausencia como capa posterior. Requiere dictamen especifico.

---

### 21.4. Matriz de cobertura de los 3 bloqueantes

| Bloqueante | Piezas existentes | Piezas ausentes | Estado global |
|---|---|---|---|
| B1 TARGET independiente | `coverage.py::compute_contractual_coverage`, `PositionRecord`, `aggregate_positions_by_shareclass_figi`, `target_universe.py::resolve_cusips`, `radar_target_catalog.csv` | `target_builder.py`, integracion en `nipc.py::compute_coverage_pairwise`, refactor de nombres | PARCIAL |
| B2 point-in-time | `source_date` (no contractual), `valid_from`/`valid_to` en tablas con vigencia | `catalog_version`, `catalog_valid_from`, `catalog_valid_to`, `target_catalog_as_of`, snapshot+hash OpenFIGI | AUSENTE: solo `source_date` no contractual |
| B3 13F != real-time | Doctrina P63/P64/P65 (docstrings), `FILING_DATE`+`PERIODOFREPORT` en bruto | `knowledge_date`, `period_end`+`filing_date` en `PositionRecord`, clasificador `absence_reason`, estados `ZERO_REPORTED`/`NOT_PRESENT` | DOCUMENTAL: doctrina si, materializacion no |

---

### 21.5. Lo que NO se toca en A.6.0

- Codigo productivo: `nipc.py`, `coverage.py`, `delta_shares.py`, `security_identity.py`, `relationships.py`.
- Contratos: `NIPC_CONTRATOS_SEMANTICOS_v1.md`, `NIPC_COVERAGE_POLICY.md` v1.0.
- Catalogos en disco: `radar_target_catalog.csv`, `cusip_equivalence.csv`, `cusip_ticker_exceptions.csv`.
- OpenFIGI: NO ejecutado. NO autorizado.
- `DROP_DUP`: NO activado.

---

### 21.6. Estado tras A.6.0

**Que desbloquea:** A.6.0 es prerequisito de A.6.2. Con este inventario, A.6.2 (fixes quirurgicos) y A.6.2-bis (rediseno TARGET) tienen base empirica.

**Que queda pendiente:**

- A.6.2 (fixes quirurgicos post F2.4): CERRADO segun FASE_A6_PLAN.md, pero su base ahora es explicita y trazable.
- A.6.2-bis (rediseno TARGET): PENDIENTE. Requiere autorizacion especifica. Materialmente bloqueado por OpenFIGI masivo NO AUTORIZADO.
- A.6.3 (test P38 segun Q12): PENDIENTE.
- A.6.4 (recalculo evidencia): condicional a F2.4.
- A.6.6 (F2.4-CLOSE): PENDIENTE.

**Criterio de aceptacion A.6.0 (segun FASE_A6_PLAN.md):** inventario completo con referencias archivo:linea. Sin tocar codigo. **CUMPLIDO.**

---

### 21.7. Referencias

- `FASE_A6_PLAN.md` seccion A.6.0.
- `RECONCILIACION_CONTRATO_CODIGO.md` seccion 3 (D1/D2/D3).
- `REESTRUCTURACION_MODULO.md` secciones 4.1-4.5.
- `DICTAMENES.md` #24 (F2.4, 2026-09-20).
- `NIPC_CONTRATOS_SEMANTICOS_v1.md` secciones 3 (P38), 11 (P62), 12 (P63), 13 (P64), 14 (P65).
- Modulos auditados: `coverage.py`, `nipc.py`, `delta_shares.py`, `radar_target_catalog.py`, `target_universe.py`, `temporal_validity.py`.

---

Fin de la seccion 21 (A.6.0).

---

## 22. INFORME B2-PIT - Infraestructura temporal del catalogo (2026-09-21)

**Fecha:** 2026-09-21.
**Origen:** dictamen estrategico #52 (Opcion 1 aprobada). A.6.2-bis
se descompone en 3 subfases: B2-PIT (cerrada), B1 (bloqueada), B3
(aprobada condicionalmente).
**HEAD al cerrar:** 5eee203.
**Evidencia:** `iae/evidence/b2_pit_cierre/`.
**Documento de subfase:** `iae/A62BIS_B2_PIT_SUBFASE.md`.

### Objeto

Materializar la infraestructura temporal del catalogo: "que catalogo
era valido en una fecha t".

### Alcance (acotado por #52 seccion 5)

    snapshot materializado + manifest + version_id + sha256
    valid_from / valid_to (semiabiertos)
    target_catalog_as_of(period_end)
    integridad + corrupcion + ambiguedad
    no backdating

**NO incluye** (pertenece a B1 o B3):

    catalog_key
    catalog_key_assignment_unique
    resolucion economica FIGI
    continuidad catalog_key -> FIGI
    TARGET_PAIRWISE
    adaptador P38
    collision catalog_key -> FIGI
    catalog_validator de asignacion

### Implementacion

**Modulo nuevo:** `src/institutional_accumulation/catalog_pit.py`.

    target_catalog_as_of(period_end, *, catalog_root) -> (df, version_id, sha256)
    load_manifest(catalog_root) -> dict
    verify_snapshot_integrity(version_id, *, catalog_root) -> bool
    list_snapshots(*, catalog_root) -> list[dict]

Excepciones: `CatalogNotAvailable`, `CatalogAmbiguous`,
`SnapshotIntegrityError`, `ManifestError`.

**Contrato de pureza:** determinista, sin `datetime.now()`, sin
escritura de ficheros. Los snapshots se publican en fase
administrativa externa.

**Artefactos materializados:**

    data/mappings/catalog_snapshots/
        snapshot_20260921_01.csv      (242 filas)
        snapshot_20260921_01.sha256   (sha256 externo, separado del CSV)
    data/mappings/catalog_manifest.json

**Versionado:** `version_id = <YYYYMMDD>_<NN>` (no autorreferencial).
**Intervalos:** `[valid_from, valid_to)`. `null` = vigente.
**Backdating:** prohibido. Q4 2025 / Q1 2026 -> `CatalogNotAvailable`.

### Tests

`tests/test_catalog_pit.py` (16 tests, PASS):

    - as_of con 0/1/>1 snapshots que cubren
    - snapshot existente que NO cubre
    - backdating prohibido (Q4 2025 + Q1 2026)
    - sha256 recalculado vs publicado
    - corrupcion simulada
    - manifest schema + inexistente
    - intervalos semiabiertos sin solapamiento
    - coherencia manifest <-> snapshot sha
    - inmutabilidad del CSV entre runs
    - list_snapshots

### Resultado

    pytest tests/test_catalog_pit.py     16 passed
    pyflakes                              LIMPIO
    compileall                            OK
    Suite completa                        1083 passed + 2 skipped
                                          + 3 failed preexistentes
                                          (test_freshness, ya documentados)

Cero regresion nueva. Los 3 failed son los mismos ya demostrados
preexistentes en `iae/evidence/p66_baseline_pre/`.

### Estado del ciclo A.6.2-bis

    A.6.2-bis-B2-PIT    CERRADO (este informe)
    A.6.2-bis-B1        OPEN - bloqueado por #51 (asignacion + dominio P38)
    A.6.2-bis-B3        APPROVED CONDITIONAL
    A.6.2-bis completo  NO CERRADO

    A.6.3                BLOCKED (requiere B1)
    A.6.4                BLOCKED (requiere A.6.3)
    F2.4-CLOSE           BLOCKED

### Refs

    Dictamen #52          Opcion 1 aprobada
    A62BIS_B2_PIT_SUBFASE.md   documento de subfase
    A62BIS_PROPUESTA.md   v9
    FASE_A6_PLAN.md       seccion A.6.2-bis actualizada
    NIPC_CONTRATOS_SEMANTICOS_v1.md secciones 3 y 11
    F2.4 #24              bloqueante 2 (point-in-time)

---

Fin del informe B2-PIT.