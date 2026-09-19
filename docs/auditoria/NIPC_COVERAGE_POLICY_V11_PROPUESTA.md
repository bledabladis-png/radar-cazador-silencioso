# NIPC COVERAGE POLICY v1.1 - PROPUESTA (BORRADOR)

Version: 1.1-borrador (2026-09-19)
Estado: BORRADOR. NO vigente. NO sustituye a NIPC_COVERAGE_POLICY.md v1.0.
Dictamen habilitante: INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_BASELINE_DICTAMEN.md
Inventario origen: INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_CONTRACT_INVENTARIO.md
Naturaleza: no normativo. Si hay conflicto con el Prompt Maestro, gana el Prompt Maestro.
Alcance: propuesta de reescritura de apartados afectados (3, 4.B, 10) + seccion nueva.

---

## 0. Proposito de este borrador

El dictamen del auditor sobre el coverage baseline identifico una
circularidad documental: la policy v1.0 define `operational_universe`
incorporando `ticker_mapped` y despues lo usa para medir coverage. El
resultado es que `coverage` se vuelve 1.0000 por construccion sobre ese
universo, sin medir la cobertura efectiva del universo objetivo.

Este borrador propone:

  1. Separar tres conceptos que la v1.0 mezcla:
       TARGET UNIVERSE      que queremos cubrir
       RESOLVED UNIVERSE    que podemos identificar actualmente
       PAIRED UNIVERSE      que empareja entre Q4 y Q1

  2. Reescribir la regla de publicacion (seccion 3) para que los
     thresholds se evaluen sobre PAIRED medido contra TARGET, no contra
     un universo ya filtrado por mapping.

  3. Reescribir la relacion con el operational_universe (seccion 10).

  4. Aclarar la diferencia 242 vs 313 del apartado 4.B.

No toca la policy v1.0 (que permanece como documento historico).
No fija THRESHOLD_1 / THRESHOLD_2 (eso es fase posterior).

---

## 1. Problema a resolver (resumen del inventario)

Cita literal de la policy v1.0 que origina la circularidad:

  L51-54: "Los thresholds se evaluan sobre el operational_universe
           (radar_equities ^ section13f_eligible ^ ticker_mapped ^ EQUITY),
           nunca sobre el universo tecnico (13F SH + null) ni sobre el
           13F completo."

  L32-33: "coverage_previous   Q4 2025: mapped / total"
          "coverage_current    Q1 2026: mapped / total"

  L134-143: Seccion 10 completa.

Consecuencia medida (coverage baseline, commit e9fd830):

  coverage_previous = 1.0000  y  coverage_current = 1.0000
  en TECHNICAL_RADAR y OPERATIONAL_EQUITY.

Estos 1.0000 son consecuencia de la definicion, no evidencia de
cobertura suficiente. La distincion debe quedar congelada.

---

## 2. Tres conceptos separados (nuevo)

La v1.1 introduce tres universos con definiciones disjuntas y encadenadas.

### 2.1. TARGET_UNIVERSE

Universo objetivo del NIPC. Que queremos cubrir.

  TARGET_UNIVERSE
    = radar_equities
      ^ section13f_eligible
      ^ EQUITY

Notas:

  - NO incluye `ticker_mapped`. El universo objetivo es previo al
    proceso de resolucion de identidad.
  - `radar_equities` se define en el apartado 4.B (ver seccion 4 de
    este borrador).
  - `section13f_eligible` viene de SEC Official List (estados
    ACTIVE / ADDED).
  - `EQUITY` viene de get_instrument_class(ticker) == EQUITY.

### 2.2. RESOLVED_UNIVERSE

Subconjunto de TARGET_UNIVERSE que el sistema ha resuelto con
identidad canonica.

  RESOLVED_UNIVERSE
    = subset de TARGET_UNIVERSE
      donde security_resolution_status == CANONICAL

Notas:

  - Un elemento de TARGET puede estar en RESOLVED o no.
  - Los elementos de TARGET que NO estan en RESOLVED son el gap de
    resolucion (mapping), medido por `resolution_coverage`.

### 2.3. PAIRED_UNIVERSE

Subconjunto de RESOLVED_UNIVERSE que aparece emparejado entre los dos
periodos comparados.

  PAIRED_UNIVERSE
    = subset de RESOLVED_UNIVERSE
      donde canonical_security (o su observed_security_key) aparece
      en ambos periodos con posicion no nula

Notas:

  - El emparejamiento se realiza por canonical_security (identidad
    estable), NO por observed_security_key (tecnico por periodo).
  - Un elemento en RESOLVED puede estar en PAIRED o no. Los no paired
    son NEW o EXIT del periodo.

### 2.4. Encadenamiento

  TARGET_UNIVERSE
         v   (resolucion de identidad)
  RESOLVED_UNIVERSE
         v   (emparejamiento interperiodo)
  PAIRED_UNIVERSE

Cada flecha tiene su propia metrica. Las metricas se reportan siempre
por separado, nunca colapsadas.

---

## 3. Metricas obligatorias (nuevas)

Las 6 metricas de la v1.0 se sustituyen por 6 metricas equivalentes
pero con denominador explicito:

  Por cada periodo P en {Q4 2025, Q1 2026}:

    resolution_coverage_P    = |RESOLVED_UNIVERSE_P|
                               / |TARGET_UNIVERSE_P|

    resolution_weighted_P    = sum(SSHPRNAMT de RESOLVED_UNIVERSE_P)
                               / sum(SSHPRNAMT de TARGET_UNIVERSE_P)

  Interperiodo (pairwise):

    paired_security_coverage = |PAIRED_UNIVERSE|
                               / |TARGET_UNIVERSE_Q4 ^ TARGET_UNIVERSE_Q1|

    paired_weighted_share_coverage
                             = sum(SSHPRNAMT de PAIRED_UNIVERSE)
                               / sum(SSHPRNAMT de TARGET_UNIVERSE_Q4 ^ TARGET_UNIVERSE_Q1)

  Diagnostico (siguen obligatorios, informativos):

    unmapped_weight_previous = 1 - resolution_weighted_Q4
    unmapped_weight_current  = 1 - resolution_weighted_Q1

Las 4 metricas pairwise + resolution cubren las 6 obligatorias de la
v1.0 con denominadores declarados. Ninguna usa un universo filtrado por
su propio proceso.

---

## 4. Reescritura del apartado 4.B (aclaracion 242 vs 313)

La v1.0 mezcla dos cifras sin explicar la diferencia:

  L67:  "universo radar real (313 tickers)"
  L115: "mapping_coverage (radar USA): 90.91% (220/242)"

La v1.1 propone la siguiente redaccion para el apartado 4.B:

  4.B. Universo radar.

    El sistema opera sobre un universo radar de 313 tickers, que
    se descompone asi:

      313 total
       +--  51 con sufijo europeo y provider oficial
       +--  20 con sufijo .L sin provider oficial (limitacion conocida)
       +-- 242 sin sufijo (universo USA del micro-gate NIPC)

    El micro-gate NIPC opera sobre los 242 tickers sin sufijo. Esta es
    la definicion de `radar_equities` para TARGET_UNIVERSE.

    Cualquier cifra de cobertura reportada sobre el universo radar USA
    (242) debe declarar explicitamente su base. Las cifras sobre 313 se
    reservan para el reporte general del sistema, no para NIPC.

    Los 71 tickers con sufijo se excluyen del micro-gate NIPC por ser
    de mercados europeos (LSE, XETRA, BME, EURONEXT), fuera del
    universo 13F (SEC, solo USA).

---

## 5. Reescritura de la seccion 10 (relacion con el universo)

La v1.0 tenia una sola seccion 10 ("Relacion con el
operational_universe"). La v1.1 la sustituye por dos apartados.

### 5.1. Seccion 10 (nueva). Universos y su uso

  TARGET_UNIVERSE      Universo objetivo. Se usa como denominador de
                       resolution_coverage y paired_security_coverage.
                       NO contiene `ticker_mapped`.

  RESOLVED_UNIVERSE    Subconjunto de TARGET que el sistema ha
                       resuelto con identidad canonica.
                       Se usa como numerador de resolution_coverage.

  PAIRED_UNIVERSE      Subconjunto de RESOLVED que empareja entre
                       periodos. Se usa como numerador de
                       paired_security_coverage.

  Diagnostico: el universo tecnico (13F SH + null, sin filtros) se usa
  para diagnosticar el motor, nunca para fijar umbrales.

### 5.2. Seccion 10.bis (nueva). Regla de publicacion sobre PAIRED

  La regla de publicacion cambia asi:

  v1.0:
    (1) paired_security_coverage    >= THRESHOLD_1
    (2) paired_weighted_share_coverage >= THRESHOLD_2
    sobre operational_universe = TARGET ^ ticker_mapped

  v1.1:
    (1) paired_security_coverage    >= THRESHOLD_1
    (2) paired_weighted_share_coverage >= THRESHOLD_2
    sobre PAIRED_UNIVERSE / TARGET_UNIVERSE ^ (Q4, Q1)

  La diferencia: el denominador deja de incluir `ticker_mapped`. Los
  thresholds miden cuanta cobertura efectiva del universo objetivo se
  consigue emparejar entre periodos, no cuanta estabilidad tiene el
  subconjunto ya mapeado.

  El fallo de circularidad queda eliminado por construccion: no puede
  haber `coverage = 1.0000` sin que el universo objetivo este cubierto
  al 100%.

---

## 6. Secciones que NO cambian

Las siguientes secciones de la v1.0 se mantienen integras en la v1.1:

  Seccion 1  (Proposito)
  Seccion 2  (Metricas obligatorias, C3) -> reescrita en seccion 3 de
             este borrador, con denominadores explicitos
  Seccion 5  (Criterios prohibidos)
  Seccion 6  (Controles simultaneos)
  Seccion 7  (Valores actuales de los umbrales)
  Seccion 8  (Estado actual de cobertura) -> valores a actualizar tras
             proxima medicion, no en este borrador
  Seccion 9  (Regla de promocion)
  Seccion 10.bis (Filer continuity como control de integridad)
  Seccion 11 (Referencias) -> actualizar versiones de prompt y spec

---

## 7. Diff resumido frente a v1.0

  APARTADO                ACCION
  ----------------------  -----------------------------------------
  3 (Regla de publicacion)  REESCRITA. Denominador: PAIRED / TARGET
  4.B (242 vs 313)          ACLARADA. Descomposicion documentada
  10 (Relacion universo)    DESDOBLADA en 10 y 10.bis
  Nuevo: TARGET/RESOLVED/PAIRED  ANADIDO como conceptos separados
  Nuevo: resolution_coverage     ANADIDO como metrica explicita
  2 (Metricas)              ACTUALIZADA con denominadores explicitos
  7, 11 (referencias)       A ACTUALIZAR en version final
  1, 5, 6, 8, 9             SIN CAMBIO

Ningun cambio toca:

  - compute_delta_shares
  - compute_nipc_and_coverage
  - security_identity
  - C2 (match key)
  - Baseline evidencia (evidence/nipc_gate0_baseline/)

---

## 8. Estado y siguiente fase

  Este documento es BORRADOR. NO sustituye a NIPC_COVERAGE_POLICY.md
  v1.0.

  Fases pendientes:

    F2.3  Dictamen del auditor sobre esta propuesta.
          Si aprueba: F2.4 (aplicacion de v1.1 sobre la policy).
          Si rechaza o pide cambios: revisar y repetir F2.2.

    F2.4  Solo si aprobado: reescribir NIPC_COVERAGE_POLICY.md a v1.1
          con los cambios aqui propuestos.

  Thresholds THRESHOLD_1 / THRESHOLD_2 siguen UNDEFINED en este
  borrador. No se fijan hasta que la semantica este aprobada y exista
  evidencia medida sobre el nuevo denominador.

  Gate-NIPC.2 sigue BLOQUEADO.

---

Fin del borrador v1.1. Version 1.1-borrador (2026-09-19).
No sustituye a NIPC_COVERAGE_POLICY.md v1.0.
Propuesta sujeta a dictamen del auditor (F2.3).