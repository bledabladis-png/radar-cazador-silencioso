# Autorizacion TOP 2000 - validacion cuantitativa

**Fase:** F2.3-bis
**Estado:** GO
**Alcance:** exclusivamente piloto empirico sobre Q1 2026
**THRESHOLD_1 / THRESHOLD_2:** UNDEFINED
**F2.4:** NO AUTORIZADA

---

## 1. Tratamiento temporal aprobado

Queda aprobada para el piloto la siguiente convencion:

> "Este piloto aplica el radar actual (2026-09-19) retrospectivamente a
> las observaciones 13F del 2026Q1, para caracterizar que fraccion del
> universo 13F, por numero de securities y por peso SSHPRNAMT, esta
> cubierta por el radar actual. No se modela historical membership del
> radar."

Esta convencion responde al issue de temporalidad sin introducir una
decision estructural sobre la pertenencia historica al radar.

---
## 2. Limites contractuales

La convencion:

  - NO afirma que los 242 tickers pertenecieran al radar en Q1 2026.
  - NO reconstruye el radar historico de Q1 2026.
  - NO sustituye la definicion futura de historical membership.
  - NO permite llamar a la cobertura obtenida "cobertura historica del
    radar Q1 2026".
  - SI permite medir retrospectivamente la interseccion entre el universo
    13F Q1 2026 y el radar vigente a 2026-09-19.

La nomenclatura del informe debe reflejarlo inequivocamente:

    RADAR_SNAPSHOT_DATE = 2026-09-19
    13F_OBSERVATION_PERIOD = 2026-03-31
    RADAR_MEMBERSHIP_MODE = CURRENT_RETROSPECTIVE

---
## 3. TOP 2000 - metricas obligatorias

El piloto debera medir por count y peso SSHPRNAMT:

    target_true_count
    target_false_count
    no_id_count
    error_count

    target_true_shares
    target_false_shares
    no_id_shares
    error_shares

y sus porcentajes respectivos.

Ademas, debera reportar:

    duplicate_shareClassFIGI
    multiple_openfigi_hits
    conflicting_candidates

separando claramente COUNT IMPACT vs WEIGHT IMPACT.

---
## 4. Reglas de resolucion

Se mantiene:

    CUSIP_13F
        -> OpenFIGI ID_CUSIP
        -> shareClassFIGI
        -> RADAR_TARGET_CATALOG
        -> target_membership

El RADAR_TARGET_CATALOG sigue siendo independiente del crosswalk
interno.

Los MISS BRK-B y MOG-A no se convierten en equivalencias sin evidencia.

Los NO_ID y ERROR no se eliminan: se conservan y se ponderan por
SSHPRNAMT.

---

## 5. Que queda fuera

Este piloto NO autoriza:

    OpenFIGI full run
    modificacion del motor NIPC
    modificacion C2
    fijacion de thresholds
    aplicacion de policy v1.2
    Gate-NIPC.2
    historical radar membership
    reconciliacion economica NT-HR

---
## 6. Decision

    Fase 1 - preservar dictamen + README     GO
    Fase 2 - TOP 2000                        GO
    Fase 3 - informe al auditor              GO

    FULL OpenFIGI                            NO
    THRESHOLD_1/2                            UNDEFINED
    F2.4                                     NO
    Gate-NIPC.2                              BLOQUEADO

La medicion TOP 2000 queda autorizada unicamente bajo la convencion
temporal indicada y debera distinguir expresamente entre radar actual
aplicado retrospectivamente y membresia historica real del radar.

---

Fin de la autorizacion. Fase F2.3-bis. HEAD revisado: c6d3e0a.
Fecha autorizacion: 2026-09-19.