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
