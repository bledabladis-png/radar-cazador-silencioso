# NIPC_COVERAGE_POLICY_V13_PROPUESTA

**Objeto:** propuesta normativa de cobertura NIPC, derivada del contrato semantico habilitante.
**HEAD al redactar:** 77d4785.
**Estado:** BORRADOR. No aplicable. Sustituye funcionalmente a v1.2 (NO GO por F2.3).
**Contrato habilitante:** NIPC_CONTRATOS_SEMANTICOS_v1.md.
**Autor:** Ingeniero Supervisor (revision interna).
**Fecha:** 2026-09-20.

**Cadena documental preservada (INTACTA):**

    NIPC_COVERAGE_POLICY.md               v1.0  vigente
    NIPC_COVERAGE_POLICY_V11_PROPUESTA.md v1.1  historico (NO GO F2.2)
    NIPC_COVERAGE_POLICY_V12_PROPUESTA.md v1.2  historico (NO GO F2.3)
    NIPC_CONTRATOS_SEMANTICOS_v1.md       v1   contrato habilitante
    NIPC_COVERAGE_POLICY_V13_PROPUESTA.md v1.3  ESTE documento

**Prohibiciones vigentes:**

- NO modificar v1.0 (hash 57f2d01f...).
- NO modificar v1.1.
- NO modificar v1.2.
- NO modificar codigo productivo sin autorizacion.
- NO fijar THRESHOLD_1 ni THRESHOLD_2.
- NO reescribir baseline historico.

---

## 0. Resumen ejecutivo

Esta propuesta sustituye funcionalmente a v1.2. Se apoya en
NIPC_CONTRATOS_SEMANTICOS_v1.md, que congela tres contratos:

| Contrato | Bloqueo | Decision |
|---|---|---|
| Identity        | P60 | `identity_type` obligatorio, sin nuevos kinds |
| Temporal        | P61 | `operational_mapping_status` separado de `security_resolution_status` |
| Coverage        | P38 | Denominador pairwise = `TARGET_Q4 INTERSECT TARGET_Q1` |

Los tres deben leerse juntos. Esta policy los cita sin duplicarlos.

**THRESHOLD_1 y THRESHOLD_2 permanecen UNDEFINED.** Se fijaran tras
recalculo de evidencia bajo la formula contractual congelada.

---

## 1. Relacion con el contrato habilitante

Esta policy **no es autonomamente interpretable**. Cuando una seccion
dice "ver contrato seccion X", el texto contractual prevalece sobre
cualquier parafrasis aqui recogida.

Regla de conflicto:

    NIPC_CONTRATOS_SEMANTICOS_v1.md     >  esta policy
    NIPC_COVERAGE_POLICY.md (v1.0)      >  esta policy (mientras v1.0 siga vigente)

Cuando v1.3 sea aprobada y aplicada, sustituira a v1.0 como referente
normativo. Hasta entonces, v1.0 sigue vigente.
---

## 2. Definiciones base

Referencia normativa: NIPC_CONTRATOS_SEMANTICOS_v1.md secciones 3 y 4.

### 2.1. TARGET

    TARGET_P:
      securities con observacion 13F en P
      AND presentes en RADAR_TARGET_CATALOG por shareClassFIGI

TARGET_P **no depende del resolver evaluado**. Se construye desde
RADAR_TARGET_CATALOG (242 filas, sha256 11eabce8...) y la observacion
13F/SEC del periodo P.

### 2.2. RESOLVED

    RESOLVED_P:
      securities en TARGET_P
      con security_resolution_status == CANONICAL
      segun resolve_security_identity(cusip, period)

RESOLVED_P es subset de TARGET_P.

### 2.3. PAIRED

    PAIRED:
      securities en RESOLVED_Q4 INTERSECT RESOLVED_Q1
      con operational_mapping_status == VERIFIED en ambos periodos
      AND canonical_security(Q4) == canonical_security(Q1)

### 2.4. TARGET_PAIRWISE

    TARGET_PAIRWISE:
      TARGET_Q4 INTERSECT TARGET_Q1

Este es el denominador contractual de la metrica ponderada.

### 2.5. No-colapso

1. TARGET no se filtra por RESOLVED.
2. RESOLVED no se filtra por TARGET.
3. PAIRED requiere TARGET ∩ RESOLVED ∩ canonical_security comun ∩
   operational_mapping_status VERIFIED en ambos periodos.

---

## 3. Metricas pairwise obligatorias

Las seis metricas contractuales (spec v1.4 seccion 3.14):

| Metrica | Definicion contractual |
|---|---|
| coverage_previous | RESOLVED_Q4 / TARGET_Q4 |
| coverage_current | RESOLVED_Q1 / TARGET_Q1 |
| paired_security_coverage | PAIRED / TARGET_PAIRWISE |
| paired_weighted_share_coverage | ver seccion 4 |
| unmapped_weight_previous | 1 - coverage_previous |
| unmapped_weight_current | 1 - coverage_current |

Nota: las definiciones de coverage_previous/current **cambian** respecto
de v1.0 y v1.2. En v1.0 el denominador era "securities observadas"; en
v1.3 es "securities TARGET". La diferencia es material para
OPERATIONAL_EQUITY (donde coverage era 1.0 por construccion).
---

## 4. Formula contractual: paired_weighted_share_coverage (P38)

### 4.1. Formula

    paired_weighted_share_coverage
      = sum( w(s) para s en PAIRED )
        / sum( w(s) para s en TARGET_PAIRWISE )

    con:
      w(s) = max( SSHPRNAMT_Q4(s), SSHPRNAMT_Q1(s) )
      una vez por security (no por filing, no por manager)

### 4.2. Diferencia respecto de v1.2 y de la implementacion vigente

Implementacion vigente (`nipc.py::compute_coverage_pairwise`):

    denom = sum(by_sec.get(k, 0.0) for k in union_sec)

Contrato v1.3:

    denom = sum(by_sec.get(k, 0.0) for k in target_pairwise)
    con target_pairwise = TARGET_Q4 INTERSECT TARGET_Q1

**El cambio es de interseccion de TARGET, no de interseccion de
observadas.** El denominador deja de ser la union de observadas y pasa
a ser la interseccion de TARGET.

### 4.3. Consecuencia operativa

Los valores de `paired_weighted_share_coverage` publicados en:

- Baseline Fase A (commit e9fd830).
- TOP 2000 (commit 32baf9d).

Se declaran **HISTORICOS / NO APTOS PARA THRESHOLD_2** bajo la definicion
v1.3. Deben recalcularse tras aplicar el fix de codigo (fase 5 de la
secuencia del dictamen).

El resto del baseline (coverage_previous, coverage_current,
paired_security_coverage) tambien cambia si TARGET != observadas.
Recalculo completo requerido antes de fijar thresholds.

---

## 5. Identity Contract (P60) — referencia

Referencia normativa: NIPC_CONTRATOS_SEMANTICOS_v1.md seccion 1.

Resumen operativo:

| identity_type | canonical_security | canonical_security_kind |
|---|---|---|
| TICKER | `equity:<ticker>` | CANONICAL_EQUIVALENCE |
| FIGI | `figi:<FIGI>` | CANONICAL_FIGI |
| CUSIP | NULL | OBSERVED_CUSIP_ONLY (o UNRESOLVED) |
| ISIN | NULL | (fuera de alcance v1) |

**Prohibido:** inferir el prefijo desde un valor desnudo.

**Estado `canonical_security_kind`:** SIN CAMBIOS respecto de v1.2.
Los cuatro kinds vigentes se mantienen.

---

## 6. Temporal Validity Contract (P61) — referencia

Referencia normativa: NIPC_CONTRATOS_SEMANTICOS_v1.md seccion 2.

### 6.1. Separacion de campos

    security_resolution_status   (identidad)
      CANONICAL
      OBSERVED_ONLY
      UNRESOLVED
      AMBIGUOUS
      CONFLICT

    operational_mapping_status   (validez historica)
      VERIFIED
      TEMPORAL_UNVERIFIED
      UNRESOLVED
      CONFLICT

`TEMPORAL_UNVERIFIED` **no esta en `security_resolution_status`**. Es un
campo independiente.

### 6.2. Regla para fuentes sin vigencia

| Fuente | operational_mapping_status |
|---|---|
| cusip_ticker_exceptions.csv | VERIFIED si cubre el periodo |
| cusip_equivalence.csv | VERIFIED si cubre el periodo (0 filas hoy) |
| etf_holdings.csv | TEMPORAL_UNVERIFIED (sin vigencia declarada) |
| radar_target_catalog.csv | (ver seccion 2.1) |

### 6.3. Regla de entrada al conjunto operacional

Solo entran al conjunto operacional historico las securities con:

    security_resolution_status = CANONICAL
    AND
    operational_mapping_status = VERIFIED

`TEMPORAL_UNVERIFIED` queda fuera del conjunto operacional **sin alterar
`security_resolution_status`**. Es una consecuencia metodologica, no un
fallback.
---

## 7. Thresholds

    THRESHOLD_1 = UNDEFINED
    THRESHOLD_2 = UNDEFINED

Ninguno de los dos se fija en v1.3. Se fijaran tras:

1. Aplicar fix P38 en codigo (fase 5).
2. Aplicar fix P60 en codigo (fase 5).
3. Aplicar fix P61 en codigo (fase 5).
4. Recalcular baseline completo.
5. Recalcular TOP 2000.
6. Nueva evidencia = base para propuesta de thresholds.

El dictamen formal del auditor mantiene:

    Gate-NIPC.2 = BLOQUEADO
    Gate-NIPC.3 = NO AUTORIZADO

hasta que se complete la secuencia.

---

## 8. Baseline historico

### 8.1. Estado del baseline Fase A

El baseline (commit e9fd830, informe
INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_BASELINE_INFORME.md)
se declara:

- **Vigente como diagnostico** de la implementacion v1.0.
- **NO apto para THRESHOLD_2** en su metrica `paired_weighted_share_coverage`.
- **Sujeto a recalculo** para `coverage_*` y `paired_security_coverage` si
  TARGET difiere de observadas.

### 8.2. Valores historicos preservados

Los valores de baseline y TOP 2000 se preservan como evidencia historica
(no se reescriben). Cualquier recalculo produce NUEVOS ficheros con
sufijo de version, no sobrescribe.

### 8.3. TOP 2000

El piloto TOP 2000 (commit 3ef4d2d) tambien queda como historico. Su
metrica `paired_weighted_share_coverage` requiere recalculo bajo la
definicion v1.3.

---

## 9. Cambios respecto de v1.2

Resumen de cambios materiales v1.2 -> v1.3:

| Seccion | Cambio |
|---|---|
| 2.1 TARGET | Definicion independiente del resolver (explicita) |
| 2.2 RESOLVED | Definicion encadenada TARGET -> RESOLVED |
| 3 Metricas | Denominadores cambian a TARGET_P (no observadas) |
| 4 P38 | Denominador = TARGET_PAIRWISE (no union_sec) |
| 5 P60 | Anadida seccion identity_type (nueva) |
| 6 P61 | Anadida seccion operational_mapping_status (nueva) |
| 7 Thresholds | UNDEFINED explicito |
| 8 Baseline | Declarado historico para threshold |

### 9.1. Lo que NO cambia respecto de v1.2

- Convencion ponderada `w(s) = max(Q4, Q1)` una vez por security.
- No uso del crosswalk interno como fuente del catalogo radar.
- Filer continuity como diagnostico, no threshold.
- F2.3-bis (H1 CERRADO): sin circularidad.
- Estructura de enums NIPC (`ALL_NIPC_STATUSES`).

---

## 10. Anexos

### 10.1. Referencias

| Documento | Rol |
|---|---|
| NIPC_CONTRATOS_SEMANTICOS_v1.md | Contrato habilitante |
| NIPC_P70_DICTAMEN.md | Cierre P70 |
| NIPC_COVERAGE_POLICY.md (v1.0) | Policy vigente (hash 57f2d01f...) |
| NIPC_COVERAGE_POLICY_V11_PROPUESTA.md | Historico |
| NIPC_COVERAGE_POLICY_V12_PROPUESTA.md | Historico (NO GO F2.3) |
| INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md | Spec NIPC v1.4 |
| INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_BASELINE_INFORME.md | Baseline Fase A |
| INSTITUTIONAL_ACCUMULATION_OPENFIGI_TOP2000_INFORME.md | TOP 2000 |

### 10.2. Hashes

    NIPC_COVERAGE_POLICY.md (v1.0):              57f2d01feb68d8916dfa4fa0451224c2ee3dda7e4631a7863e671257b3e91f06
    radar_target_catalog.csv:                    11eabce8f8aaed1be6aa3c3557b5e392ad305757230333f84f37285c401b62b7
    NIPC_CONTRATOS_SEMANTICOS_v1.md:             2d5084b93af5c8e29d3cd2471c36e4ef3d52413ebe2733493ede2a4227cb6875
    NIPC_P70_DICTAMEN.md:                        e8d4e4e825fe6136b23573705b800108247eb07c9a46e2f62c2c06980f179092

### 10.3. Estado de bloqueos

    P38   CERRADO contractualmente (contrato sec. 3, 4)
    P60   CERRADO contractualmente (contrato sec. 1)
    P61   CERRADO contractualmente (contrato sec. 2)
    P70   CERRADO (NIPC_P70_DICTAMEN.md)

    THRESHOLD_1                          UNDEFINED
    THRESHOLD_2                          UNDEFINED
    Policy v1.0                          INTACTA
    Policy v1.1                          INTACTA
    Policy v1.2                          INTACTA
    Policy v1.3                          BORRADOR (este documento)
    Codigo productivo                    SIN CAMBIOS
    OpenFIGI masivo                      NO AUTORIZADO
    Gate-NIPC.2                          BLOQUEADO
    Gate-NIPC.3                          NO AUTORIZADO
    F2.4                                 NO AUTORIZADA

---

Fin de la propuesta v1.3.
Version 1.3-rc1 (2026-09-20). HEAD 77d4785.
Sustituye funcionalmente a v1.2, que permanece como historico.