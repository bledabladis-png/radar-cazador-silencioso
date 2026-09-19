# INVENTARIO - Gate 0 documental del contrato de cobertura NIPC

**Modulo:** IAE (Institutional Accumulation Evidence) - SEC 13F / NIPC
**Referencia:** prompt maestro v6.37
**Fecha:** 2026-09-19
**Fase:** F2.1 del micro-gate Coverage Contract Normalization
**Dictamen habilitante:** INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_BASELINE_DICTAMEN.md (2026-09-19)
**Fichero auditado:** docs/auditoria/NIPC_COVERAGE_POLICY.md (v1.0)

---

## 0. Proposito

Inventario linea-por-linea de los tres conceptos que el dictamen del
auditor (seccion 9) exige separar:

    A. TARGET UNIVERSE      - que queremos cubrir
    B. RESOLVED UNIVERSE    - que podemos identificar
    C. PAIRED (comparabilidad) - que proporcion empareja Q4 <-> Q1

Objetivo: localizar donde vive (o no vive) cada concepto en la policy
v1.0, con cita literal y numero de linea, para fundamentar la futura
propuesta v1.1 sin ambiguedad.

NO toca el motor. NO modifica la policy. NO fija thresholds.
Solo diagnostico documental.

---

## 1. Estado del fichero auditado

    Path:   docs/auditoria/NIPC_COVERAGE_POLICY.md
    Bytes:  8487
    Lineas: 231
    SHA256: 57f2d01feb68d8916dfa4fa0451224c2ee3dda7e4631a7863e671257b3e91f06

---

## 2. Concepto A - TARGET UNIVERSE (universo objetivo)

Busqueda literal de "TARGET" / "target" en la policy: 0 ocurrencias.

El concepto no existe como tal. La policy declara un unico universo
(`operational_universe`) que YA incorpora `ticker_mapped`.

Conceptos equivalentes que SI aparecen:

  L51-54  "Los thresholds se evaluan sobre el operational_universe
           (radar_equities ^ section13f_eligible ^ ticker_mapped ^ EQUITY),
           nunca sobre el universo tecnico (13F SH + null) ni sobre el 13F
           completo."

  L62-63  "A. Universo operativo (radar ^ 13F eligible ^ mapped ^ EQUITY).
           NO universo 13F completo (24,838 CUSIPs). NO universo tecnico."

  L134-143  Seccion 10 completa ("Relacion con el operational_universe").
            Repite la interseccion. No distingue objetivo de resuelto.

Verificacion: el concepto TARGET no existe como tal. Existe
`operational_universe`, que ya contiene `ticker_mapped` en su
definicion. Esa es la circularidad detectada por el auditor.

---

## 3. Concepto B - RESOLVED UNIVERSE (universo resuelto)

Busqueda literal de "RESOLVED" en la policy: 0 ocurrencias.

Conceptos equivalentes que SI aparecen:

  L32-33  "coverage_previous   Q4 2025: mapped / total"
          "coverage_current    Q1 2026: mapped / total"

          Define la metrica `mapped / total`. NO define ni `mapped`
          ni `total`.

  L115    "mapping_coverage (radar USA): 90.91% (220/242)"

          Unica cifra de mapping. No declara que universo es
          `radar USA`.

  L116    "weighted_share_coverage (radar USA): 29.995%"

          Idem.

Verificacion: el concepto RESOLVED no existe como tal. La policy usa
`mapped / total` sin definir el universo denominador. Es una de las
dos causas de la circularidad: no se puede distinguir "el total antes
de mapear" de "el total despues de mapear".

---

## 4. Concepto C - PAIRED (comparabilidad Q4 <-> Q1)

Busqueda literal de "paired" en la policy: 13 ocurrencias.

Listado completo con cita literal:

  L23     "paired_security_coverage, paired_weighted_share_coverage,"
          (cita del dictamen habilitante)

  L34     "paired_security_coverage    securities con mapping en ambos periodos"

  L35     "paired_weighted_share_coverage ponderado por SSHPRNAMT de
           securities emparejadas"

  L39-40  "Las dos metricas que gobiernan el status READY son las pairwise:
           paired_security_coverage y paired_weighted_share_coverage."

  L48-49  "(1) paired_security_coverage        >= THRESHOLD_1
           (2) paired_weighted_share_coverage  >= THRESHOLD_2"

  L79-81  "E. Pairwise mas exigente que single-period. La cobertura pairwise
           es estructuralmente menor que coverage_current"

  L98-99  "THRESHOLD_1 -> paired_security_coverage (numero de securities)
           THRESHOLD_2 -> paired_weighted_share_coverage (peso por SSHPRNAMT)"

  L118    "paired_*:                              pendiente calculo"

  L141    "Los thresholds se evaluan sobre esta interseccion."

  L175-176 "THRESHOLD_1 (paired_security_coverage)
            THRESHOLD_2 (paired_weighted_share_coverage)"

  L184-185 "(1) paired_security_coverage        >= THRESHOLD_1
            (2) paired_weighted_share_coverage  >= THRESHOLD_2"

Verificacion: PAIRED existe con 2 metricas definidas de manera vaga.
"Securities con mapping en ambos periodos" (L34) no declara:

  - Sobre que universo base se calcula.
  - Como se empareja la misma security entre periodos
    (por observed_security_key? por canonical_security? por ticker?).
  - Si el "mapping en ambos periodos" es mapping al radar o mapping
    a canonical.

La metrica pairwise hereda la misma circularidad que `coverage`:
no declara el universo base.

---

## 5. Tabla resumen: cobertura de los 3 conceptos

| Concepto              | Palabra literal | Concepto operativo | Definicion explicita |
|-----------------------|-----------------|--------------------|----------------------|
| TARGET UNIVERSE       | NO              | Parcial            | NO                   |
| RESOLVED UNIVERSE     | NO              | Parcial            | NO                   |
| PAIRED                | SI (13x)        | SI (2 metricas)    | Parcial (vaga)       |

Donde:

  TARGET UNIVERSE    aparece como `operational_universe` que ya incluye
                     `ticker_mapped`. No hay denominador objetivo
                     independiente.

  RESOLVED UNIVERSE  aparece como `mapped / total` sin definicion de
                     ninguno de los dos terminos.

  PAIRED             definido para las 2 metricas pairwise, pero sin
                     universo base ni criterio de emparejamiento
                     explicito.

---

## 6. Hallazgos verificables (con cita literal)

### H1 - Circularidad documental explicita

  L51-54: operational_universe = radar_equities ^ section13f_eligible
          ^ ticker_mapped ^ EQUITY.
  L32-33: coverage = mapped / total.

  Si el universo contiene `ticker_mapped`, el `total` de coverage ya
  esta restringido a `mapped`. No hay denominador "objetivo" declarado.

  Resultado medido en el baseline (commit e9fd830):
    coverage = 1.0000 en TECHNICAL_RADAR y OPERATIONAL_EQUITY.
  Coherente con la definicion circular.

### H2 - Contradiccion de universos

  L67:   "los umbrales deben evaluarse contra el universo radar real
          (313 tickers), no contra subconjuntos".
  L115:  "mapping_coverage (radar USA): 90.91% (220/242)".

  La policy usa 313 en L67 y 242 en L115 sin explicar la diferencia.

### H3 - Conceptos sin definicion

  `mapped`  (L32-33): usado, no definido.
  `total`   (L32-33): usado, no definido.
  `radar USA` (L115): usado, no definido.
  Universo base de las metricas pairwise: no declarado.
  Universo base de `mapping_coverage`: no declarado.

### H4 - Residuo de versionado

  L169-170: "THRESHOLD_3 = UNDEFINED (NO aplicable; filer continuity no
             es threshold)" / "THRESHOLD_4 = UNDEFINED (NO aplicable)".
  L172-173: "Eliminados los THRESHOLD_3/THRESHOLD_4 que la version
             anterior (v1.3) contemplaba."

  Sin impacto funcional. Residuo de versionado.

### H5 - Referencia desactualizada

  L225: "Prompt Maestro v6.35: PROMPT_MAESTRO.md"

  El prompt vigente es v6.37.

---

## 7. Fuera de alcance de este inventario

  - Analisis de si los 3 conceptos aparecen con nombres distintos
    en INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md v1.4.
  - Analisis de si compute_coverage_pairwise en nipc.py implementa
    literalmente la definicion vaga de L34-35.
  - Analisis de si el dictamen habilitante
    (INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION_V11_DICTAMEN.md)
    contiene definiciones mas precisas que la policy no heredo.

Estos 3 puntos van a F2.2 (redaccion de la propuesta v1.1) como
input adicional si el auditor lo requiere.

---

## 8. Uso previsto

Este inventario es la base documental de:

  F2.2  NIPC_COVERAGE_POLICY_V11_PROPUESTA.md
        Propuesta de reescritura de los apartados afectados
        (secciones 3, 4.B, 10) incorporando:

          TARGET UNIVERSE       (nuevo)
          RESOLVED UNIVERSE     (nuevo)
          PAIRED                (clarificado)
          aclaracion 242 vs 313 (H2)

  F2.3  Dictamen del auditor sobre la propuesta.

  F2.4  Aplicacion de v1.1 (solo si el auditor aprueba).

---

## 9. Referencias

  - NIPC_COVERAGE_POLICY.md v1.0 (auditada):
      docs/auditoria/NIPC_COVERAGE_POLICY.md
  - Dictamen habilitante:
      docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_BASELINE_DICTAMEN.md
  - Baseline (evidencia):
      docs/auditoria/evidence/nipc_gate0_baseline/
  - Informe del baseline:
      docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_BASELINE_INFORME.md

---

Fin del inventario. Version 1.0 (2026-09-19).
Sin modificacion de la policy auditada.
Siguiente fase autorizada: F2.2 (propuesta v1.1).