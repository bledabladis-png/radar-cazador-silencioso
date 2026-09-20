# INFORME TECNICO - SISTEMA IAE (INSTITUTIONAL ACCUMULATION EVIDENCE)

**Objeto:** informe tecnico completo del sistema IAE para auditoria externa.
**HEAD al redactar:** 9c32c9b (o posterior, ver commit del propio informe).
**Ahead de origin/main:** 105+
**Tests locales:** 979 passed + 2 skipped.
**pyflakes:** 0 warnings. **compileall:** OK. **Working tree:** limpio.
**Push:** NO (regla local-first IAE).
**Fecha:** 2026-09-20.

---

## 1. RESUMEN EJECUTIVO

El modulo IAE implementa acumulacion institucional sobre el dataset SEC 13F.
Se compone de 5 fases:

- FA-1 (ingestion): 7 TSVs -> parquets + manifest 3 niveles hash.
- FA-2 (identidad + amendments): filtro temporal, CUSIP->ticker, relationships, canonical snapshot.
- NIPC (Net Institutional Position Change): 5 modulos, 91 tests.
- F2.x (coverage semantica): 3 contratos (P60/P61/P38) redactados e implementados.
- Revision estructural: 95 hallazgos, 5 fixes mecanicos + 1 guarda.

**Estado:** 979 tests passed. Motor NIPC completo. Falta materializar TARGET
real (requiere OpenFIGI masivo, NO AUTORIZADO).

---

## 2. ARQUITECTURA

### 2.1. Ficheros del modulo

    src/institutional_accumulation/
      __init__.py
      aggregation/
        __init__.py
        delta_shares.py           # unidades + delta
        nipc.py                   # NIPC + coverage pairwise
      identity/
        __init__.py
        openfigi_client.py
        radar_target_catalog.py
        target_universe.py
      sec_13f/
        __init__.py
        downloader.py
        parser.py
        schema.py
        storage.py
        manifest.py
        ingest.py
        identity/
          temporal_filter.py
          cusip_resolver.py
          relationships.py
          amendments.py
          sec13f_list.py
          security_identity.py
      temporal_validity.py        # NUEVO (P61)

### 2.2. Contratos temporales (pipeline principal)

10 contratos FU-021-5 + FU-021-3C-bis.
FSM: PENDING | OK | STALE | INSUFFICIENT | BLOCKED.
Autoridad: `temporal_meta` dict explicito. `df.attrs` es espejo auxiliar.

### 2.3. Cobertura principal (radar)

313/313 tickers (Yahoo 262 + Euronext 13 + Xetra 19 + BME 19 + OilPriceAPI 5).
Gate de validacion: 10/10.

---

## 3. CONTRATOS SEMANTICOS (P60 / P61 / P38)

### 3.1. P60 - Identity Contract

    identity_type obligatorio declarado por la fuente.

    TICKER -> equity:<ticker>  (kind CANONICAL_EQUIVALENCE)
    FIGI   -> figi:<FIGI>      (kind CANONICAL_FIGI)
    CUSIP  -> NULL             (kind OBSERVED_CUSIP_ONLY)
    ISIN   -> NULL             (kind UNRESOLVED)

    Prefijo explicito en el valor (equity:/figi:) gana sobre identity_type.
    Ausente o invalido -> ValueError.
    Prohibido inferir tipo desde valor desnudo.

Implementacion: `security_identity.py::_normalize_canonical(value, identity_type)`.

### 3.2. P61 - Temporal Validity Contract

Dos campos separados:

- `security_resolution_status` (identidad): 5 estados.
- `operational_mapping_status` (validez historica):
  VERIFIED | TEMPORAL_UNVERIFIED | UNRESOLVED | CONFLICT.

    Fuentes con vigencia declarada (cusip_equivalence, cusip_ticker_exceptions):
      cubre periodo -> VERIFIED
      no cubre      -> UNRESOLVED

    Fuentes sin vigencia (etf_holdings, openfigi):
      -> TEMPORAL_UNVERIFIED

    Agregacion Q7:
      VERIFIED + VERIFIED mismo valor -> VERIFIED
      VERIFIED + VERIFIED distinto   -> CONFLICT
      VERIFIED + TEMP mismo valor    -> VERIFIED
      VERIFIED + TEMP distinto       -> CONFLICT
      TEMP + TEMP mismo valor        -> TEMPORAL_UNVERIFIED
      TEMP + TEMP distinto           -> CONFLICT

Implementacion: modulo `temporal_validity.py` + integracion en
`resolve_security_identity`.

### 3.3. P38 - Coverage Contract

    TARGET_PAIRWISE = TARGET_Q4 INTERSECT TARGET_Q1
    denominador     = sum(w(s)) sobre TARGET_PAIRWISE
    numerador       = sum(w(s)) sobre PAIRED
    w(s) = max(SSHPRNAMT_Q4(s), SSHPRNAMT_Q1(s)) una vez por security.

    RESOLVED_P   = TARGET_P INTERSECT {security_resolution_status == CANONICAL}
    operacional  = RESOLVED_P INTERSECT {operational_mapping_status == VERIFIED}
    PAIRED       = operacional_Q4 INTERSECT operacional_Q1
                   con canonical_security comun.

    Si TARGET_PAIRWISE vacio o sum(w(s))=0:
      paired_weighted_share_coverage = None
      coverage_status = UNAVAILABLE

Implementacion: `nipc.py::compute_coverage_pairwise`.

---

## 4. METRICAS CONTRACTUALES (spec v1.4 sec 3.14)

| Metrica | Definicion |
|---|---|
| coverage_previous | RESOLVED_Q4 / TARGET_Q4 |
| coverage_current | RESOLVED_Q1 / TARGET_Q1 |
| paired_security_coverage | PAIRED / TARGET_PAIRWISE |
| paired_weighted_share_coverage | sum(w(s)) PAIRED / sum(w(s)) TARGET_PAIRWISE (o None) |
| coverage_status | VALID / UNAVAILABLE (Q5) |
| unmapped_count_previous | 1 - coverage_previous (Q6) |
| unmapped_count_current | 1 - coverage_current (Q6) |

---

## 5. IMPLEMENTACION - COMMITS CLAVE

### 5.1. Implementacion P60 / P61 / P38

| Commit | Contenido |
|---|---|
| 030f239 | P60: identity_type obligatorio en _normalize_canonical |
| 6198d60 | P61: modulo temporal_validity.py |
| fc12402 | P38: denominador TARGET_PAIRWISE + Q5 coverage_status |
| 2a13a6f | P61: integrado en resolver |
| 0f3ef8d | P61: end-to-end hasta _mapped_mask |
| 83e60c3 | Q6 unmapped_count_* + Q8 identity_type en CSV |

### 5.2. Fixes mecanicos previos

| Commit | Contenido |
|---|---|
| 0f1e5a4 | P51: linea corta -> anomalia |
| 714457e | P32: contador de descartes |
| eb687c4 | P14-BIS: fillna condicionado a _merge |
| e31d82e | P31: nipc_total_available |
| 54478c3 | P18/P26: project_root por env var |
| 1575d17 | P70: guarda explicita len==2 |

---

## 6. TESTS

| Bloque | Tests |
|---|---|
| Suite local total | 979 passed + 2 skipped |
| Modulo IAE | ~280 passed |
| Regresion acumulada | 0 |

Ficheros de tests del modulo (extracto):

    tests/test_sec_13f_list.py               26
    tests/test_sec_13f_delta_shares.py       22
    tests/test_sec_13f_nipc.py               24
    tests/test_sec_13f_manifest.py           13
    tests/test_sec_13f_amendments.py         31
    tests/test_sec_13f_security_identity.py  40
    tests/test_temporal_validity.py          15
    tests/test_sec_13f_* (resto)             ...

---

## 7. ESTADO DE BLOQUEOS

    THRESHOLD_1 / THRESHOLD_2         UNDEFINED
    Gate-NIPC.2                       BLOQUEADO
    Gate-NIPC.3                       NO AUTORIZADO
    OpenFIGI masivo (24.838 CUSIPs)   NO AUTORIZADO
    Policy v1.3 aplicacion            NO AUTORIZADA
    F2.4                              PENDIENTE EXTERNO

---

## 8. DOCUMENTOS DE TRABAJO

| Documento | Hash SHA-256 (prefijo) |
|---|---|
| NIPC_INFORME_ENTREGA_F24.md | 0763f16b... |
| NIPC_INFORME_ESTADO_POST_FIXES_F24.md | 49c397b7... |
| NIPC_CONTRATOS_SEMANTICOS_v1.md | c02d202f... |
| NIPC_COVERAGE_POLICY_V13_PROPUESTA.md | 933fcb2e... |
| NIPC_P70_DICTAMEN.md | e8d4e4e8... |
| NIPC_DICTAMEN_REVISION_ESTRUCTURAL_2026-09-20.md | 3d004540... |
| INSTITUTIONAL_ACCUMULATION_REVISION_ESTRUCTURAL_2026-09-20.md | 318f02e8... |

Preservados intactos:

| Documento | Hash |
|---|---|
| NIPC_COVERAGE_POLICY.md (v1.0) | 57f2d01f... |
| NIPC_COVERAGE_POLICY_V11_PROPUESTA.md | historico |
| NIPC_COVERAGE_POLICY_V12_PROPUESTA.md | historico |

---

## 9. HUECO PENDIENTE - TARGET REAL

El denominador P38 usa `observed_security_key` como proxy de TARGET.
La proyeccion real `CUSIP -> shareClassFIGI -> RADAR_TARGET_CATALOG`
requiere OpenFIGI sobre el universo 13F completo (24.838 CUSIPs).

**Bloqueo:** OpenFIGI masivo NO AUTORIZADO hasta F2.4.

**Alternativa conservadora** (sin OpenFIGI): filtrar units a aquellos
con `canonical_security` presente en `radar_target_catalog.csv`
(242 tickers). Es una aproximacion, no TARGET real.

---

## 10. PREGUNTAS AL AUDITOR

### 10.1. Decisiones bloqueantes

**D1.** Aprobacion del contrato semantico v1
(`NIPC_CONTRATOS_SEMANTICOS_v1.md`, hash c02d202f...) en los terminos
resultantes de las decisiones Q1-Q12.

**D2.** Aprobacion de `NIPC_COVERAGE_POLICY_V13_PROPUESTA.md`
(hash 933fcb2e...) como sustitucion funcional de v1.2.

**D3.** Confirmacion del cierre de P70 como GO CONDICIONADO.

**D4.** Autorizacion del paso 4 (tests especificos).

### 10.2. Preguntas contractuales (Q1-Q12)

Detalladas en `NIPC_INFORME_ESTADO_POST_FIXES_F24.md`
(hash 49c397b7...).

- Q1: RESOLVED subconjunto de TARGET.
- Q2: unidad de security + agregacion + consolidacion de estados.
- Q3: temporalidad RADAR_TARGET_CATALOG (CURRENT_RETROSPECTIVE).
- Q4: API normalize_identity().
- Q5: denominador P38 = 0 -> UNAVAILABLE.
- Q6: unmapped_count_* (implementado).
- Q7: agregacion multi-fuente.
- Q8: identity_type en CSV (implementado).
- Q9-Q11: correcciones documentales.
- Q12: clave contractual de PAIRED (shareClassFIGI vs canonical_security).

### 10.3. Pregunta adicional (post-implementacion)

**Q13.** Autorizacion de la materializacion del TARGET real mediante
OpenFIGI masivo (24.838 CUSIPs) sobre el universo 13F completo, o
mantenimiento del proxy actual hasta fijar el contrato definitivo.

---

## 11. LO QUE NO SE HA HECHO

- No push a origin/main (regla local-first IAE).
- No modificar v1.0, v1.1, v1.2.
- No ejecutar OpenFIGI masivo.
- No fijar THRESHOLD_1 ni THRESHOLD_2.
- No aplicar policy v1.3.
- No reconciliar NT <-> HR.
- No sumar SSHPRNAMT desde INFOTABLE.parquet crudo.

---

FIN DEL INFORME TECNICO
HEAD: 9c32c9b
Ahead: 105+ commits
Sin push (regla local-first IAE)
