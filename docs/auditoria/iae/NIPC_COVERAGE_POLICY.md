# NIPC COVERAGE POLICY - Politica de umbrales de cobertura

Version: 1.0 (2026-09-19)
Dictamen habilitante: INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION_V11_DICTAMEN.md (Q-ESP-V11-9)
Especificacion: INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md (v1.2)
Estado: DOCUMENTO DE POLITICA. Thresholds UNDEFINED hasta dictamen del auditor.
Naturaleza: no normativo. Si hay conflicto con el Prompt Maestro, gana el Prompt Maestro.

---

## 1. Proposito

Fijar la metodologia y criterios para determinar los umbrales
contractuales de cobertura que autorizan la publicacion de NIPC
productivo. Los valores numericos concretos (THRESHOLD_1, THRESHOLD_2)
permanecen UNDEFINED hasta el dictamen especifico del auditor previo
a Gate-NIPC.2.

Origen (cita literal del dictamen Gate-NIPC.1 v1.1, Q-ESP-V11-9):

  "Los thresholds deben fijarse antes de Gate-NIPC.2, no durante la
   implementacion. El criterio debe considerar al menos:
   paired_security_coverage, paired_weighted_share_coverage,
   unmapped_weight_previous, unmapped_weight_current. No number of
   mapped securities only. Y evitaria tambien fijarlos por simple
   analogia con el actual 90.91%."

## 2. Metricas obligatorias (C3, seis simultaneas)

Se reportan siempre juntas. Nunca una sola cifra:

    coverage_previous              Q4 2025: mapped / total
    coverage_current               Q1 2026: mapped / total
    paired_security_coverage       securities con mapping en ambos periodos
    paired_weighted_share_coverage ponderado por SSHPRNAMT de securities emparejadas
    unmapped_weight_previous       1 - coverage ponderado Q4
    unmapped_weight_current        1 - coverage ponderado Q1

Las dos metricas que gobiernan el status READY son las pairwise:
paired_security_coverage y paired_weighted_share_coverage.

## 3. Regla de publicacion

NIPC productivo solo con status = READY.

READY requiere AMBOS controles simultaneos:

    (1) paired_security_coverage        >= THRESHOLD_1
    (2) paired_weighted_share_coverage  >= THRESHOLD_2

Los thresholds se evaluan sobre el operational_universe
(radar_equities ^ section13f_eligible ^ ticker_mapped ^ EQUITY),
nunca sobre el universo tecnico (13F SH + null) ni sobre el 13F
completo.

Un solo control por encima del umbral no autoriza READY.

## 4. Criterios de fijacion (obligatorios para el auditor)

Al fijar THRESHOLD_1 y THRESHOLD_2 deben considerarse:

  A. Universo operativo (radar ^ 13F eligible ^ mapped ^ EQUITY).
     NO universo 13F completo (24,838 CUSIPs). NO universo tecnico.

  B. Sesgo large-cap. El crosswalk interno actual esta sesgado a
     large caps; los umbrales deben evaluarse contra el universo
     radar real (313 tickers), no contra subconjuntos.

  C. Descomposicion de no mapeados:
       - no-equity legitimamente excluida (CONVERTIBLE BOND, MMF,
         PFD, warrants, rights)
       - equity genuinamente faltante
     Solo la segunda categoria justifica subir el umbral. La primera
     es exclusion por diseno, no gap de cobertura.

  D. Alcanzabilidad al primer intento. Un umbral aspiracional no
     alcanzable bloquea el producto sin beneficio funcional.

  E. Pairwise mas exigente que single-period. La cobertura pairwise
     es estructuralmente menor que coverage_current; THRESHOLD_1 y
     THRESHOLD_2 deben acomodarlo.

## 5. Criterios prohibidos

NO fijar THRESHOLD_1 / THRESHOLD_2 por:

  - Analogia con 90.91% (observado en un periodo aislado, Q-ESP-V11-9).
  - Analogia con 29.995% (observado en un periodo aislado).
  - Numero de securities mapeadas unicamente (ignora peso economico).
  - Numero de CUSIPs mapeadas / total 13F completo.
  - Umbrales "redondos" sin justificacion empirica.
  - Analogia con umbrales de otros modulos (coverage distinto).

## 6. Controles simultaneos (contrato Q3)

Contrato Q3 exige DOS controles. Se mantienen:

    THRESHOLD_1 -> paired_security_coverage (numero de securities)
    THRESHOLD_2 -> paired_weighted_share_coverage (peso por SSHPRNAMT)

Un control por si solo no autoriza READY. NIPC no puede publicar una
unica cifra de coverage (regla 3.14).

## 7. Valores actuales de los umbrales

    THRESHOLD_1 = UNDEFINED
    THRESHOLD_2 = UNDEFINED

Ambos seran fijados por dictamen especifico del auditor ANTES de
Gate-NIPC.2. Este documento se actualiza con los valores aprobados
y se marca con nueva version (v1.1 del documento).

## 8. Estado actual de cobertura (Q1 2026, crosswalk interno actual)

    mapping_coverage (radar USA):          90.91% (220/242)
    weighted_share_coverage (radar USA):   29.995%
    section13f_eligible (radar USA):       pendiente integracion SEC 13(f)
    paired_*:                              pendiente calculo

    NIPC status actual: INSUFFICIENT.

Estos valores son DATOS OBSERVADOS, no umbrales. Ver dictamenes
Gate 0 Mapping y Gate 0 SEC 13(f) mini-probe.

## 9. Regla de promocion

Cambio de INSUFFICIENT a READY:

  - Solo el auditor promueve.
  - Se realiza tras validacion formal en Gate-NIPC.3.
  - Queda registrado en FOLLOWUPS.md.
  - Requiere actualizacion del prompt maestro si aplica.

## 10. Relacion con el operational_universe

El operational_universe es el unico universo sobre el que se
publica NIPC productivo. Es la interseccion:

    radar_equities ^ section13f_eligible ^ ticker_mapped ^ EQUITY

Los thresholds se evaluan sobre esta interseccion. El universo
tecnico (13F SH + null) se usa para diagnostico del motor, no para
fijar umbrales.

## 10.bis. Filer continuity como control de integridad

Dictamen habilitante: INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_TOP50_DICTAMEN.md
(2026-09-19, Q-T50-1 a Q-T50-5). Especificacion: seccion 3.15 (v1.4).

### Principio

Filer continuity es CONTROL DE INTEGRIDAD / DIAGNOSTICO, NO threshold
cuantitativo. La cifra observada (top 50: 4/54 = 7.4%) NO se convierte
en threshold.

### Metricas de diagnostico (a reportar siempre)

  filer_discontinuity_count
  filer_discontinuity_pct
  filer_discontinuity_gross_shares
  filer_discontinuity_concentration

Estas metricas se reportan como OBSERVACION. No autorizan ni bloquean
READY por si solas. Se usan para detectar reorganizaciones anormales
que podrian distorsionar el NIPC observable.

### Thresholds

  THRESHOLD_3 = UNDEFINED (NO aplicable; filer continuity no es threshold)
  THRESHOLD_4 = UNDEFINED (NO aplicable)

Eliminados los THRESHOLD_3/THRESHOLD_4 que la version anterior (v1.3)
contemplaba. Ahora solo existen:

  THRESHOLD_1 (paired_security_coverage)
  THRESHOLD_2 (paired_weighted_share_coverage)

Ambos UNDEFINED hasta dictamen especifico.

### Regla de publicacion

  READY requiere:

    (1) paired_security_coverage        >= THRESHOLD_1
    (2) paired_weighted_share_coverage  >= THRESHOLD_2
    (3) filer_discontinuity_pct <= valor de control (a definir con thresholds)

No se fija el valor del control (3) en esta version. Se propone como
alerta: si filer_discontinuity_pct supera el N% (a definir), se marca
la observacion en el reporte pero no se bloquea READY automaticamente.

### Dimensiones separadas (Q-T50-2)

  filer_status           por presencia documental del filing
                         (CONTINUOUS_FILER | FILER_DISCONTINUITY | UNRESOLVED)
  position_mass_status   por SSHPRNAMT (HAS_SHARES | ZERO_SHARES | NO_CANONICAL_HOLDINGS)
  nt_to_hr_relation_observed (bool)
  nt_to_hr_relation_targets (list)

### Estado actual (Q4 2025 -> Q1 2026)

  top 20: 3/24 discontinuidades (12.5%), todas Vanguard.
  top 50: 4/54 discontinuidades (7.4%), todas Vanguard.

  Interpretacion: concentracion en una estructura corporativa concreta
  durante este periodo. NO extrapolable al universo 13F.

  STATUS global: INSUFFICIENT (por coverage < thresholds UNDEFINED).

### Prohibiciones

  - NO convertir filer_discontinuity_pct en threshold de publicacion.
  - NO inferir economic_owner por parent/child.
  - NO reconciliar NT <-> HR productivamente.
  - NO usar filer_status por SSHPRNAMT > 0.

---

## 11. Referencias

  - Dictamen Gate-NIPC.1 v1.1: INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION_V11_DICTAMEN.md
  - Spec NIPC v1.2: INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md
  - Dictamen Gate 0 Mapping: INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_MAPPING_DICTAMEN.md
  - Dictamen Gate 0 SEC 13(f) mini-probe: INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_SEC13F_MINIPROBE_DICTAMEN.md
  - Prompt Maestro v6.35: PROMPT_MAESTRO.md

---

Fin del documento. Version 1.0 (2026-09-19). Thresholds UNDEFINED.
Pendiente: dictamen especifico del auditor con valores numericos
antes de Gate-NIPC.2.
