# IAE - INFORME CONSOLIDADO

**Generado:** 2026-09-20
**Consolida:** 17 informes del ciclo IAE

**Regla:** este fichero se actualiza in-place. Las versiones previas se archivan con fecha.

**Actualizacion 2026-09-20:** entrada en FASE A.5 (NIPC + contratos).
3 divergencias contrato<->codigo identificadas (D1 P61, D2 P38, D3 P60)
con evidencia ejecutable (6 tests xfail). Expediente F2.4 listo.
F2.4 PENDIENTE EXTERNO.

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


Fin del informe consolidado.
