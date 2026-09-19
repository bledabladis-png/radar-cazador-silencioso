# NIPC COVERAGE POLICY v1.2 - PROPUESTA (BORRADOR)

Version: 1.2-borrador (2026-09-19)
Estado: BORRADOR. NO vigente. NO sustituye a NIPC_COVERAGE_POLICY.md v1.0.
Dictamen habilitante: dictamen F2.2 (NO GO CON CAMBIOS OBLIGATORIOS, 2026-09-19).
Inventario origen F2.1: INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_CONTRACT_INVENTARIO.md
Inventario origen F2.1-bis: INSTITUTIONAL_ACCUMULATION_NIPC_F21BIS_FUENTES_IDENTIDAD.md
Naturaleza: no normativo. Si hay conflicto con el Prompt Maestro, gana el Prompt Maestro.
Alcance: propuesta de reescritura completa de la policy de cobertura NIPC.

---

## 0. Proposito y trazabilidad

### 0.1. Cadena de versiones

  v1.0 (2026-09-19)   vigente. Define operational_universe como
                      interseccion con ticker_mapped. Circular al
                      medir coverage sobre el mismo universo.
  v1.1 (2026-09-19)   propuesta NO GO. Separaba TARGET / RESOLVED /
                      PAIRED conceptualmente bien, pero definia TARGET
                      como interseccion sobre claves incompatibles.
  v1.2 (este doc)     propuesta actual. Aplica Camino B + los 8
                      cambios obligatorios del dictamen F2.2.

### 0.2. Cita literal del dictamen F2.2

  "La arquitectura conceptual es GO.
   La definicion matematica/operativa de TARGET es NO GO."

El dictamen F2.2 fue entregado como texto en sesion 2026-09-19.
Se preserva operativamente en:
  docs/auditoria/TRANSFER_SESION_2026-09-19_POST_F22.md (commit d2e03a1)
  docs/auditoria/PROMPT_MAESTRO.md v6.38, seccion 11.27
### 0.3. Los 8 cambios obligatorios (dictamen F2.2)

  1. RADAR_TARGET_REGISTRY con identidad estable independiente del
     resolver evaluado.
  2. TARGET_UNIVERSE sobre clave comun a CUSIP/13F.
  3. RESOLVED_UNIVERSE con enums v1.3: CANONICAL_FIGI /
     CANONICAL_EQUIVALENCE (no estado generico CANONICAL).
  4. PAIRED_UNIVERSE: canonical_security comun, sin
     "or observed_security_key".
  5. PAIRED_WEIGHTED_SHARE_COVERAGE: convencion Q4/Q1 inequivoca.
  6. Resolver contradiccion filer continuity (seccion 10.bis).
  7. Corregir referencia "Prompt Maestro v6.35" en policy L225.
  8. Trazabilidad explicita del cambio semantico v1.0 -> v1.1 -> v1.2.

---

## 1. Problema heredado (resumen de F2.1 y F2.1-bis)

### 1.1. Circularidad (F2.1)

La policy v1.0 define:

    operational_universe
      = radar_equities ^ section13f_eligible ^ ticker_mapped ^ EQUITY

y a la vez declara:

    coverage = mapped / total

Como ticker_mapped ya restringe el universo a las securities
resueltas, el cociente mapped / total es 1.0000 por construccion
sobre ese universo. El baseline (commit e9fd830) lo midio
empiricamente.
### 1.2. Camino B declarado (F2.1-bis)

No existe en disco una fuente que cumpla simultaneamente:
  (a) clave CUSIP/FIGI comparable con 13F,
  (b) ticker del radar USA,
  (c) independiente del crosswalk evaluado.

Declaracion formal (2026-09-19):

    RADAR_TARGET_REGISTRY = NOT AVAILABLE.

Fuente futura de construccion: OpenFIGI (o equivalente),
a dictamen del auditor.

---

## 2. Definiciones canonicas

### 2.1. TARGET_UNIVERSE (semantico)

Universo objetivo del NIPC. Que queremos cubrir.

    TARGET_UNIVERSE (semantica):
      { S : S elegible 13F (SEC Official List ACTIVE/ADDED)
            Y S tiene CUSIP o FIGI comparable con 13F
            Y S tiene ticker en radar USA (242 sin sufijo)
            Y get_instrument_class(ticker) == EQUITY }

Notas:
  - NO contiene ticker_mapped. El universo objetivo es previo
    al proceso de resolucion de identidad.
  - No es realizable hoy sin registry (Camino B).
  - El numero 242 corresponde a los tickers sin sufijo del universo
    radar. Se documenta para eliminar la contradiccion 242 vs 313
    de la v1.0.
### 2.2. RADAR_TARGET_REGISTRY

Fuente de identidad con las tres propiedades (a)+(b)+(c).

    Estado: NOT AVAILABLE (2026-09-19).
    Fuente de construccion futura: OpenFIGI (a dictamen del auditor).

Sin esta fuente, TARGET_UNIVERSE es solo semantico, no materializable.

### 2.3. RESOLVED_UNIVERSE

Subconjunto de TARGET que el sistema ha resuelto con identidad
canonica.

    RESOLVED_UNIVERSE:
      subset de TARGET_UNIVERSE donde
        security_resolution_status == CANONICAL
        Y canonical_security_kind in {CANONICAL_FIGI,
                                      CANONICAL_EQUIVALENCE}

(Cambio #3: se excluye el estado generico CANONICAL sin kind.)

Los enums v1.3 verificados en
src/institutional_accumulation/sec_13f/identity/security_identity.py:

    security_resolution_status (5):
      CANONICAL | OBSERVED_ONLY | UNRESOLVED | AMBIGUOUS | CONFLICT

    canonical_security_kind (4):
      CANONICAL_FIGI | CANONICAL_EQUIVALENCE
      | OBSERVED_CUSIP_ONLY | UNRESOLVED

Regla dura: canonical_security = NULL salvo status == CANONICAL.
ticker:<ticker> PROHIBIDO como canonical_security.
### 2.4. PAIRED_UNIVERSE

Subconjunto de RESOLVED que aparece emparejado entre los dos
periodos comparados.

    PAIRED_UNIVERSE:
      subset de RESOLVED_UNIVERSE donde
        canonical_security aparece en ambos periodos.

(Cambio #4: emparejamiento por canonical_security comun.
Prohibido "or observed_security_key". El observed_security_key
es clave tecnica por periodo, no identidad estable.)

### 2.5. Encadenamiento

    TARGET_UNIVERSE
           v   (resolucion de identidad)
    RESOLVED_UNIVERSE
           v   (emparejamiento interperiodo)
    PAIRED_UNIVERSE

Cada flecha tiene su propia metrica. Las metricas se reportan
siempre por separado, nunca colapsadas.

---

## 3. Metricas obligatorias (6)

Por cada periodo P en {Q4 2025, Q1 2026}:

    resolution_coverage_P
      = |RESOLVED_UNIVERSE_P| / |TARGET_UNIVERSE_P|

    resolution_weighted_P
      = sum(SSHPRNAMT de RESOLVED_UNIVERSE_P)
        / sum(SSHPRNAMT de TARGET_UNIVERSE_P)
Interperiodo (pairwise):

    paired_security_coverage
      = |PAIRED_UNIVERSE|
        / |TARGET_UNIVERSE_Q4 ^ TARGET_UNIVERSE_Q1|

    paired_weighted_share_coverage
      = sum(SSHPRNAMT de PAIRED_UNIVERSE, contado UNA vez)
        / sum(SSHPRNAMT de TARGET_UNIVERSE_Q4 ^ TARGET_UNIVERSE_Q1,
              contado UNA vez)

    Convencion Q4/Q1 (cambio #5, inequivoca):
      Para cada canonical_security en el conjunto, tomar
      valor = max(SSHPRNAMT_Q4, SSHPRNAMT_Q1). UNA sola vez por
      security, no suma bruta de ambos periodos. Es la convencion
      ya implementada en compute_coverage_pairwise() (nipc.py).
      Se congela aqui como contrato.

Diagnostico (obligatorio, informativo):

    unmapped_weight_previous
      = 1 - resolution_weighted_Q4
    unmapped_weight_current
      = 1 - resolution_weighted_Q1

Las 6 metricas se reportan juntas. Prohibido publicar una sola.

---

## 4. Estado actual post F2.1-bis

    TARGET_CUSIP_REGISTRY  = NOT AVAILABLE
    TARGET_UNIVERSE        = semanticamente definible,
                             no materializable
    coverage               = NOT_MEASURABLE
    NIPC status            = INSUFFICIENT
                             (por ausencia de denominador,
                              no por valores bajos)
Ninguna cifra del baseline (coverage_previous = 1.0000,
coverage_current = 1.0000, paired_security_coverage = 0.9909,
paired_weighted_share_coverage = 0.9864) puede utilizarse como
evidencia de cobertura suficiente del universo objetivo. Miden
estabilidad Q4->Q1 de un subconjunto ya seleccionado por el
propio proceso de mapping.

---

## 5. Regla de publicacion

READY requiere TODAS las condiciones simultaneas:

    (1) TARGET_CUSIP_REGISTRY != NOT_AVAILABLE
    (2) paired_security_coverage       >= THRESHOLD_1
    (3) paired_weighted_share_coverage >= THRESHOLD_2

Un solo control no autoriza READY.

THRESHOLD_1 / THRESHOLD_2: UNDEFINED hasta dictamen especifico
del auditor, previo a Gate-NIPC.2, tras medicion real sobre
TARGET_UNIVERSE independiente.

---

## 6. Reconciliacion con filer continuity (seccion 10.bis) - Cambio #6

La v1.0 contiene una contradiccion en la seccion 10.bis:

  - Texto principal: "Filer continuity es CONTROL DE INTEGRIDAD,
    NO threshold cuantitativo."
  - Regla de publicacion: "READY requiere ...
    (3) filer_discontinuity_pct <= valor de control
    (a definir con thresholds)."
La v1.2 elimina la condicion (3). Cita literal de la spec v1.4
(seccion 3.15):

  "Filer continuity NO es threshold. Es:
     - Control de integridad del NIPC observable.
     - Diagnostico del impacto de reorganizaciones de filer.
     - Alerta cuando la discontinuidad bruta supera un margen
       razonable."

Regla v1.2:

    filer_discontinuity_pct: metrica de diagnostico.
    NO es condicion de READY.
    NO se convierte en threshold cuantitativo.
    Se reporta siempre como OBSERVACION.

THRESHOLD_3 y THRESHOLD_4 no existen. Eliminados.

---

## 7. Correccion de referencia - Cambio #7

v1.0 L225: "Prompt Maestro v6.35: PROMPT_MAESTRO.md"

v1.2:      "Prompt Maestro: docs/auditoria/PROMPT_MAESTRO.md
            (referenciar sin version; la version vigente se consulta
             en el propio fichero y en git log)."

Correccion: se elimina la version hardcoded para no crear deuda
documental cada vez que avanza el prompt. Se cita la ruta canonica.

---

## 8. Reordenacion del pipeline (a dictamen F2.3)

Secuencia descartada (v1.0):
  baseline -> OpenFIGI -> thresholds

Secuencia NO GO (v1.1):
  baseline -> policy v1.1 -> OpenFIGI -> thresholds

Secuencia propuesta (v1.2, NO ejecutada):
  baseline
    -> RADAR_TARGET_REGISTRY (via OpenFIGI como fuente)
    -> policy v1.2 con TARGET independiente
    -> medicion real sobre TARGET independiente
    -> thresholds (previo dictamen del auditor)

Diferencia clave: OpenFIGI pasa de mejora posterior de coverage
a fuente primaria para construir el target registry, ANTES de
fijar thresholds.

Esta reordenacion NO se ejecuta en este ciclo. Solo se documenta
como propuesta a dictamen del auditor (F2.3).
---

## 9. Lo que NO hace v1.2

  - NO modifica NIPC_COVERAGE_POLICY.md v1.0
    (hash 57f2d01feb68d8916dfa4fa0451224c2ee3dda7e4631a7863e671257b3e91f06).
  - NO modifica NIPC_COVERAGE_POLICY_V11_PROPUESTA.md.
  - NO toca el motor:
      src/institutional_accumulation/aggregation/nipc.py
      src/institutional_accumulation/aggregation/delta_shares.py
      src/institutional_accumulation/sec_13f/identity/security_identity.py
  - NO modifica C2 (match key sin report_period).
  - NO ejecuta OpenFIGI.
  - NO fija THRESHOLD_1 ni THRESHOLD_2.
  - NO reescribe la evidencia del baseline
    (docs/auditoria/evidence/nipc_gate0_baseline/).
  - NO modifica la spec v1.4
    (INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md).

---

## 10. Estado de gates

    F2.3 (dictamen del auditor sobre v1.2)     PENDIENTE EXTERNO
    F2.4 (aplicacion de v1.2 sobre la policy)  NO AUTORIZADA
    THRESHOLD_1                                UNDEFINED
    THRESHOLD_2                                UNDEFINED
    OpenFIGI masivo                            NO AUTORIZADO
    Gate-NIPC.2                                BLOQUEADO
    Gate-NIPC.3                                NO AUTORIZADO
    NIPC_COVERAGE_POLICY.md v1.0               INTACTA

---

## 11. Referencias

  - Policy v1.0 (intacta):
      docs/auditoria/NIPC_COVERAGE_POLICY.md
      SHA256 57f2d01feb68d8916dfa4fa0451224c2ee3dda7e4631a7863e671257b3e91f06
  - Propuesta v1.1 (NO GO):
      docs/auditoria/NIPC_COVERAGE_POLICY_V11_PROPUESTA.md
  - Dictamen baseline Gate 0:
      docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_BASELINE_DICTAMEN.md
  - Inventario F2.1:
      docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_CONTRACT_INVENTARIO.md
  - Inventario F2.1-bis:
      docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_F21BIS_FUENTES_IDENTIDAD.md
  - Transfer post F2.2 NO GO:
      docs/auditoria/TRANSFER_SESION_2026-09-19_POST_F22.md
  - Spec NIPC v1.4 activa:
      docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md
  - Evidencia baseline (intacta):
      docs/auditoria/evidence/nipc_gate0_baseline/
  - Codigo de referencia verificado:
      src/institutional_accumulation/aggregation/nipc.py
      src/institutional_accumulation/aggregation/delta_shares.py
      src/institutional_accumulation/sec_13f/identity/security_identity.py

---

Fin del borrador v1.2. Version 1.2-borrador (2026-09-19).
No sustituye a NIPC_COVERAGE_POLICY.md v1.0.
Propuesta sujeta a dictamen del auditor (F2.3).
