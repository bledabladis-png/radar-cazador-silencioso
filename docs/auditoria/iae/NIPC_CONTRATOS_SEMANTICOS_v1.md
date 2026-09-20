# NIPC_CONTRATOS_SEMANTICOS_v1

**Objeto:** contrato habilitante de identidad, validez temporal y cobertura pairwise para el modulo NIPC.
**HEAD al redactar:** 77d4785.
**Estado:** VIGENTE (2026-09-20). F2.4 EMITIDO (GO CONDICIONADO). Ver DICTAMENES.md #24.

**Precedencia transitoria:** mientras este contrato y NIPC_COVERAGE_POLICY_V13_PROPUESTA.md sean borradores, NIPC_COVERAGE_POLICY.md v1.0 permanece como referencia normativa vigente. Una vez aprobado formalmente este contrato y aplicada la policy v1.3, el contrato habilitante prevalece sobre cualquier parafrasis de la policy.
**Origen:** dictamen formal del auditor (2026-09-20), que clasifico P38/P60/P61 como NO GO con contrato requerido.
**Relacion:** cierra la seccion 2, 3 y 4 del dictamen. P70 cerrado por NIPC_P70_DICTAMEN.md.
**Autor:** Ingeniero Supervisor (revision interna).
**Fecha:** 2026-09-20.

---

## 0. Resumen ejecutivo

Tres bloqueos semanticos detectados durante la revision estructural del modulo IAE
(2026-09-20) requieren congelacion contractual antes de cualquier fijacion de
thresholds:

- **P60** — canonical_security se prefijaba por inferencia (`_normalize_canonical`).
  Viola el principio "NO INFERIR IDENTIDAD".
- **P61** — etf_holdings.csv se trataba como valido indefinidamente (sin vigencia
  temporal). Afecta a la validez historica operacional.
- **P38** — paired_weighted_share_coverage usaba `union_sec` como denominador
  mientras el docstring decia "en ambos". Bloquea THRESHOLD_2.

Este documento congela las tres decisiones como contrato coherente. Los tres
forman una sola cadena: **que es una security, cuando esta resuelta, como se
pondera en la cobertura**.

**Decisiones congeladas:**

| ID | Decision |
|---|---|
| T1/P60 | Identidad exige `identity_type` explicito. No se infiere. Enum de kinds sin cambios. |
| T2/P61 | Temporalidad separada de identidad. Nuevo campo `operational_mapping_status`. `security_resolution_status` no se amplia. |
| T3 | Orden documental: P60 → P61 → P38. |

**Estado tras este documento:** P60 y P61 CERRADOS contractualmente. P38 EN REDACCION.

**No se autoriza ningun cambio de codigo ni de policy todavia.**

---

**Actualizacion 2026-09-20 (F2.4):** decisiones confirmadas por el auditor externo.

    D1 P61       GO              Conectar resolve_source_status + evidence extendida.
    D2 P38       GO CONDICIONADO  TARGET real + rediseno independencia del mapping.
    D3 P60       GO              Fail-closed, sin default TICKER.
    Q12          Modelo A        shareClassFIGI (unidad = share class).
    AGREG.       Opcion 2        Funcion separada + solo VERIFIED al peso contractual.
    OpenFIGI     GO CONDICIONADO Snapshot + hash, no dependencia live.
    Policy v1.3  NO              v1.0 sigue normativa.

**Contratos nuevos formalizados por este dictamen:**

    P62  Point-in-time (ver seccion 11).
    P63  Missing != Sold (ver seccion 12).
    P64  Corporate Actions (ver seccion 13).
    P65  Manager Duplication (ver seccion 14).

**Reglas adicionales adoptadas:**

    - unmapped_count int separado de unmapped_weight float.
    - compute_nipc sin degradacion silenciosa (evidence_class CONTRACTUAL|PROXY).
    - PositionRecord tipado en coverage.py.
    - Solo VERIFIED al peso contractual; unverified_weight conservado aparte.
---

## 1. Identity Contract (P60)

### 1.1. Principio rector

Un valor de identidad (ticker, FIGI, CUSIP, ISIN, etc.) **no se interpreta**.
Su tipo debe venir declarado por la fuente que lo origina.

    prohibido:  valor desnudo -> inferir prefijo
    requerido:  (valor, identity_type) -> canonical_security

### 1.2. Enum de `identity_type`

    TICKER    ticker de mercado (Yahoo-equivalente)
    FIGI      Financial Instrument Global Identifier
    CUSIP     identificador observado 13F (NO es canonical por defecto)
    ISIN      identificador internacional (fuera de alcance v1)

### 1.3. Correspondencia contractual

| identity_type | canonical_security      | canonical_security_kind |
|---|---|---|
| TICKER        | `equity:<ticker>`        | CANONICAL_EQUIVALENCE   |
| FIGI          | `figi:<FIGI>`            | CANONICAL_FIGI          |
| CUSIP         | NULL                     | (no aplica)             |
| ISIN          | NULL (fuera de alcance)  | (no aplica)             |

Un valor con `identity_type = CUSIP` **no produce canonical_security**. El
CUSIP sigue siendo identificador observado de 13F, no identidad canonica
estable por defecto (regla C1-revisada vigente).

### 1.4. Enum `canonical_security_kind` — SIN CAMBIOS

    CANONICAL_FIGI
    CANONICAL_EQUIVALENCE
    OBSERVED_CUSIP_ONLY
    UNRESOLVED

P60 no introduce nuevos kinds. Endurece la **entrada** al normalizador.

### 1.5. Firma contractual del normalizador

    normalize_identity(value: str, identity_type: str) -> dict

    Devuelve:
      canonical_security       (str | None)
      canonical_security_kind  (enum)
    Raises:
      ValueError si identity_type no esta en el enum o es CUSIP/ISIN
      y se intenta producir canonical.

Un valor sin `identity_type` explicito **NO se convierte a equity:**.
Queda:

    canonical_security = NULL
    canonical_security_kind = OBSERVED_CUSIP_ONLY
    (o UNRESOLVED si el valor no es ni observado)
### 1.6. Fuentes declaradas hoy

| Fuente                  | identity_type | Notas |
|---|---|---|
| cusip_equivalence.csv   | (ver `reason`) | Vacia por diseno (0 filas). Si se puebla, cada fila debe declarar tipo. |
| cusip_ticker_exceptions | TICKER        | Tabla operativa Q-CUR. 3 filas COM. |
| etf_holdings.csv        | TICKER        | Sin vigencia temporal (ver P61). |
| openfigi (figi_lookup)  | FIGI          | No enchufado en produccion. |

### 1.7. Cambio requerido en codigo

`security_identity.py::_normalize_canonical` debe:

1. Recibir `identity_type` como parametro obligatorio.
2. Rechazar inferencia silenciosa.
3. Producir el prefijo segun la tabla 1.3.
4. Devolver NULL cuando `identity_type` es CUSIP/ISIN o no declarado.

Firma nueva propuesta:

    def _normalize_canonical(value, identity_type): ...

Los llamadores (`resolve_security_identity`) deben propagar el tipo desde
la fuente (`evidence["identity_type"]`).

### 1.8. Estado

**P60 = PROPUESTO / SOMETIDO A F2.4.**

- No nuevos kinds.
- No inferencia.
- `identity_type` obligatorio.
---

## 2. Temporal Validity Contract (P61)

### 2.1. Principio rector

    IDENTITY  !=  TEMPORAL VALIDITY

Dos preguntas distintas:

- `security_resolution_status` responde: **que identidad hemos resuelto**.
- `operational_mapping_status` responde: **tenemos evidencia de que esa
  relacion era operacionalmente valida para el periodo historico evaluado**.

### 2.2. Enum `security_resolution_status` — SIN CAMBIOS

    CANONICAL
    OBSERVED_ONLY
    UNRESOLVED
    AMBIGUOUS
    CONFLICT

**TEMPORAL_UNVERIFIED no entra aqui.** No es un fallo de identidad.

### 2.3. Enum `operational_mapping_status` (nuevo)

    VERIFIED             la relacion tiene vigencia declarada cubriendo el periodo
    TEMPORAL_UNVERIFIED  la relacion existe pero sin evidencia temporal
    UNRESOLVED           sin relacion declarada
    CONFLICT             multiples relaciones incompatibles

Este campo vive fuera de `security_identity.py` para preservar la separacion.
Ubicacion propuesta: `aggregation/` o un modulo `temporal_validity.py` nuevo.

### 2.4. Regla para `etf_holdings.csv`

    CUSIP -> ticker
        |
        v
    security_resolution_status = CANONICAL       (identidad tecnica valida)
    operational_mapping_status = TEMPORAL_UNVERIFIED  (sin vigencia Q4/Q1)

La identidad **no se degrada**. La validez historica queda marcada como
no verificada.

### 2.5. Regla operacional

Solo entran al conjunto operacional historico las securities con:

    security_resolution_status = CANONICAL
    AND
    operational_mapping_status = VERIFIED

`TEMPORAL_UNVERIFIED` queda fuera del conjunto operacional sin alterar
`security_resolution_status`. Es un resultado metodologico, no un fallback.
### 2.6. Fuentes con vigencia declarada hoy

| Fuente                        | Vigencia             | operational_mapping_status |
|---|---|---|
| cusip_ticker_exceptions.csv   | valid_from / valid_to | VERIFIED si cubre el periodo |
| cusip_equivalence.csv         | valid_from / valid_to | VERIFIED si cubre el periodo (0 filas hoy) |
| etf_holdings.csv              | sin vigencia         | TEMPORAL_UNVERIFIED |
| radar_target_catalog.csv      | source_date          | (ver seccion 4) |

### 2.7. Cambio requerido en codigo

Nuevo modulo `src/institutional_accumulation/temporal_validity.py`:

    operational_mapping_status(source: str, valid_from, valid_to, period) -> str

    Regla:
      - source con vigencia declarada + cubre periodo -> VERIFIED
      - source con vigencia declarada + no cubre -> UNRESOLVED
      - source sin vigencia -> TEMPORAL_UNVERIFIED
      - multiples fuentes incompatibles -> CONFLICT

`resolve_security_identity` debe devolver, ademas de los campos actuales,
`operational_mapping_status`. **No modifica** su logica de identidad.

### 2.8. Estado

**P61 = PROPUESTO / SOMETIDO A F2.4.**

- Identidad y temporalidad separadas.
- Nuevo campo fuera de `security_resolution_status`.
- `etf_holdings.csv` -> `TEMPORAL_UNVERIFIED`.
---

## 3. Coverage Contract (P38)

### 3.1. Principio rector

    TARGET  ->  RESOLVED  ->  operationally resolved  ->  PAIRED

TARGET no depende de RESOLVED. RESOLVED no depende de TARGET. PAIRED es la
interseccion de los dos, filtrada por validez temporal.

### 3.2. Definiciones formales

Para un periodo P (Q4 2025 o Q1 2026):

    TARGET_P
      = securities con observacion 13F en P
        AND presentes en RADAR_TARGET_CATALOG por shareClassFIGI
        (independiente del resolver evaluado)

    RESOLVED_P
      = securities en TARGET_P con security_resolution_status == CANONICAL

    OPERATIONALLY_RESOLVED_P
      = RESOLVED_P con operational_mapping_status == VERIFIED

    PAIRED
      = OPERATIONALLY_RESOLVED_Q4
        INTERSECT
        OPERATIONALLY_RESOLVED_Q1

    TARGET_PAIRWISE
      = TARGET_Q4 INTERSECT TARGET_Q1

### 3.3. Formula contractual de la metrica

    paired_weighted_share_coverage
      = sum( w(s) para s en PAIRED )
        / sum( w(s) para s en TARGET_PAIRWISE )

    con:
      w(s) = max( SSHPRNAMT_Q4(s), SSHPRNAMT_Q1(s) )
      una vez por security (no por filing, no por manager)

### 3.4. Correccion respecto de la implementacion actual

La implementacion vigente (`nipc.py::compute_coverage_pairwise`) usa:

    denom = sum(by_sec.get(k, 0.0) for k in union_sec)

El contrato exige:

    TARGET_PAIRWISE = TARGET_Q4 INTERSECT TARGET_Q1

    denom = sum( w(s) for s in TARGET_PAIRWISE )

Es decir: el denominador es la interseccion de TARGET, no la union de
observadas. La formula se expresa exclusivamente en terminos del universo
contractual; no se traduce a nombres de implementacion (`sec_c`, `sec_p`,
`union_sec`) en este documento. Esto cambia el valor de
`paired_weighted_share_coverage` y, por tanto, de THRESHOLD_2.

### 3.5. Baseline historico

Los valores de `paired_weighted_share_coverage` publicados en el baseline
(commit e9fd830) y en el TOP 2000 (commit 32baf9d) se declaran:

    HISTORICOS / NO APTOS PARA THRESHOLD_2

bajo la definicion anterior (union_sec). Deben recalcularse bajo 3.3 antes
de cualquier fijacion de threshold.

El resto de metricas del baseline (coverage_previous, coverage_current,
paired_security_coverage) no se ven afectadas por P38 — su definicion
contractual no cambia.

### 3.6. Estado

**P38 = GO CONDICIONADO (F2.4, 2026-09-20).** Denominador TARGET_PAIRWISE
confirmado. Implementacion con evidencia contractual en A.6.2 + A.6.2-bis.
---

## 4. TARGET / RESOLVED / PAIRED

### 4.1. Diagrama contractual

                          RADAR_TARGET_CATALOG
                                  |
                                  v
                             TARGET_P
                                  |
                     +------------+------------+
                     |                         |
                     v                         v
              IDENTITY RESOLUTION       TEMPORAL VALIDITY
                     |                         |
                     v                         v
              security_resolution       operational_mapping
                     |                         |
                     +------------+------------+
                                  |
                                  v
                             RESOLVED
                                  |
                             canonical_security
                             comun Q4 <-> Q1
                                  |
                                  v
                               PAIRED

### 4.2. Independencia de TARGET del resolver

TARGET_P se construye **exclusivamente** desde:

    RADAR_TARGET_CATALOG (242 filas, 240 OK, sha256 11eabce8...)
    +
    observacion 13F en el periodo P (SUBMISSION filtrado)

No se consulta `security_identity` ni `cusip_resolver` ni `target_universe`
para construir TARGET_P. La circularidad H1 (F2.1-bis) queda cerrada por
diseno.

### 4.3. Materializacion de TARGET_P

    TARGET_P:
      para cada CUSIP observado en 13F de P:
        resolved_scf = OpenFIGI(cusip).shareClassFIGI
        si resolved_scf in RADAR_TARGET_CATALOG:
          incluir en TARGET_P

El resultado se documenta con provenance: `cusip`, `share_class_figi`,
`radar_ticker`, `source_date`.

**Estado de materializacion:** esta seccion define TARGET_P a nivel
conceptual. La materializacion completa sobre el universo 13F requiere una
operacion de resolucion batch contra OpenFIGI no autorizada en este
contrato. OpenFIGI masivo (24.838 CUSIPs) sigue marcado como NO AUTORIZADO
hasta dictamen especifico. Los pilotos ejecutados (top 500, top 2000) son
evidencia metodologica, no materializacion completa del universo.

No confundir con `target_universe.py` operativo — ese es un resolver, TARGET_P es un universo contractual.

### 4.4. RESOLVED_P

    RESOLVED_P:
      securities en TARGET_P
      con security_resolution_status == CANONICAL
      segun `resolve_security_identity(cusip, period)`

Notese que RESOLVED_P es **subset de TARGET_P**. Una security no-target
puede estar CANONICAL sin entrar en RESOLVED_P.

### 4.5. PAIRED

    PAIRED:
      securities en RESOLVED_Q4 INTERSECT RESOLVED_Q1
      con operational_mapping_status == VERIFIED en ambos periodos

La identidad canonica debe ser comun: `canonical_security(Q4) == canonical_security(Q1)`.
No basta con coincidencia de CUSIP observado.
### 4.6. No-colapso de TARGET y RESOLVED

Reglas duras:

1. TARGET no se filtra por RESOLVED. Si un CUSIP esta en target y no
   resuelve, sigue siendo TARGET (afecta al denominador).
2. RESOLVED no se filtra por TARGET. Un CANONICAL fuera de target no
   participa en el pairing pero sigue siendo RESOLVED.
3. PAIRED requiere ambas condiciones + identidad canonica comun +
   operational_mapping_status VERIFIED en ambos periodos.

### 4.7. Filer continuity NO interviene

La continuidad del filer (patron Vanguard Q4->Q1) es diagnostico de
integridad, no dimension del pairing. No entra en TARGET ni en RESOLVED
ni en PAIRED. Se mantiene como control separado (ver seccion 5).

---

## 5. Filer continuity interface

### 5.1. Rol

Diagnostico de integridad del dataset. **NO es threshold.**

    filer_continuity_pct = (n_filers con HR->HR consistente)
                           / (n_filers con HR en cualquier periodo)

### 5.2. Prohibiciones

- NO reconciliacion NT <-> HR.
- NO conversion a threshold.
- NO uso como filtro de TARGET ni de RESOLVED.
- NO uso como sustituto de identity resolution.

### 5.3. Referencia

Dictamen TOP 50 (2026-09-19): 4/54 FILER_DISCONTINUITY concentradas en
Vanguard. CERRADO como caracterizacion, no como defecto.

---

## 6. Invariantes P70

P70 esta cerrado por NIPC_P70_DICTAMEN.md (documento propio, 2026-09-20).
La unica referencia contractual necesaria en este documento es:

    HR_PLUS_RESTATEMENT -> cardinalidad contractual observada == 2

Esta invariante esta garantizada por `_classify_strategy_from_types`.
El fix propuesto en NIPC_P70_DICTAMEN.md seccion 6 anade guarda explicita
sin modificar la logica. No es bloqueante para P38/P60/P61.
---

## 7. Impacto en policy v1.2

Cambios requeridos en `NIPC_COVERAGE_POLICY_V12_PROPUESTA.md` (borrador no
aplicado) y en la version final v1.2 antes de aprobacion:

1. **Seccion TARGET/RESOLVED/PAIRED** — reemplazar definiciones actuales por
   las de secciones 3 y 4 de este documento.
2. **Formula paired_weighted_share_coverage** — sustituir denominador union
   por interseccion (seccion 3.3).
3. **Convencion ponderada** — explicitar `w(s) = max(Q4, Q1)` una vez por
   security (ya estaba en propuesta v1.2, mantener).
4. **`identity_type` explicito** — anadir seccion contractual de identidad
   (seccion 1).
5. **`operational_mapping_status`** — anadir como campo contractual separado
   de `security_resolution_status` (seccion 2).
6. **Referencia a P70** — citar `NIPC_P70_DICTAMEN.md`.
7. **Baseline historico** — declarar `paired_weighted_share_coverage` como
   historico/no apto para threshold. Resto del baseline vigente.
8. **Trazabilidad** — anadir `NIPC_CONTRATOS_SEMANTICOS_v1.md` como
   documento habilitante.

**La policy v1.0 permanece INTACTA (hash 57f2d01f...).**

---

## 8. Impacto en codigo

Cambios requeridos (NO aplicar todavia; listado para fase 5):

### 8.1. `security_identity.py`

- `_normalize_canonical(value, identity_type)` — firma endurecida.
- `resolve_security_identity` propaga `identity_type` desde `evidence`.
- `evidence` incluye `identity_type` por fuente.
- Rechaza inferencia silenciosa.

### 8.2. Nuevo modulo `temporal_validity.py`

- `operational_mapping_status(source, valid_from, valid_to, period) -> str`.
- Enum: `VERIFIED | TEMPORAL_UNVERIFIED | UNRESOLVED | CONFLICT`.

### 8.3. `nipc.py`

- `compute_coverage_pairwise` recibe como entrada:
    - `target_q4`, `target_q1` (universos target).
    - `resolved_q4`, `resolved_q1` (universos resueltos + temporalmente validos).
- Calcula denominador como `target_q4 INTERSECT target_q1`.
- Calcula numerador como `resolved_q4 INTERSECT resolved_q1` con
  `canonical_security` comun.
- Documenta la distincion TARGET/RESOLVED/PAIRED en docstring.

### 8.4. Tests requeridos

- `test_sec_13f_security_identity.py` — casos identity_type TICKER/FIGI/CUSIP.
- `test_sec_13f_nipc.py` — casos denominador interseccion vs union.
- `test_temporal_validity.py` (nuevo) — 4 estados + casos borde.
- `test_sec_13f_amendments.py` — guarda P70 (invariante cardinalidad).

### 8.5. Sin cambios

- `aggregation/delta_shares.py` no se toca.
- `compute_delta_shares` no se toca.
- `match_key` no se toca.
- C2 no se toca.
---

## 9. Secuencia de implementacion

Autorizada por el dictamen formal del auditor (seccion 19):

    1. Dictamen P38/P60/P61/P70      <- este documento + NIPC_P70_DICTAMEN.md
              |
              v
    2. Modificar propuesta policy v1.2
              |
              v
    3. Aprobar contrato
              |
              v
    4. Tests especificos
              |
              v
    5. Fixes quirurgicos
              |
              v
    6. Nueva medicion de coverage
              |
              v
    7. OpenFIGI ampliado
              |
              v
    8. Threshold proposal
              |
              v
    9. Gate-NIPC.2

Estado actual: paso 1 completado con la entrega de este documento.
Pendiente de dictamen F2.4. Los pasos 2-9 no se inician hasta que F2.4
sea GO.

---

## 10. Anexos

### 10.1. Referencias

| Documento | Rol |
|---|---|
| NIPC_P70_DICTAMEN.md | Cierre P70 |
| NIPC_COVERAGE_POLICY.md (v1.0) | Policy vigente, INTACTA |
| NIPC_COVERAGE_POLICY_V12_PROPUESTA.md | Borrador v1.2 |
| INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_BASELINE_INFORME.md | Baseline Fase A |
| INSTITUTIONAL_ACCUMULATION_OPENFIGI_TOP2000_INFORME.md | TOP 2000 |
| INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_BASELINE_DICTAMEN.md | Dictamen baseline |
| INSTITUTIONAL_ACCUMULATION_NIPC_F21BIS_FUENTES_IDENTIDAD.md | F2.1-bis |
| docs/auditoria/iae/evidence/nipc_p70_probe/ | Evidencia P70 |

### 10.2. Hashes

    NIPC_COVERAGE_POLICY.md v1.0: 57f2d01feb68d8916dfa4fa0451224c2ee3dda7e4631a7863e671257b3e91f06
    radar_target_catalog.csv:     11eabce8f8aaed1be6aa3c3557b5e392ad305757230333f84f37285c401b62b7
    probe_p70_result.json:        809c51af47ae154c6c01bba1cf3983a6c0143fd3b0711958fa8b9e6e18bae96f
    probe_p70_summary.txt:        9e8c42d6a742d773be7e531782a7d8514a5c00aa64efc5080d81a718d6384fe5
    NIPC_P70_DICTAMEN.md:         e8d4e4e825fe6136b23573705b800108247eb07c9a46e2f62c2c06980f179092

### 10.3. Estado de bloqueos

    P38   GO CONDICIONADO (F2.4, 2026-09-20)
    P60   GO (F2.4, 2026-09-20)
    P61   GO (F2.4, 2026-09-20)
    P62   FORMALIZADO (F2.4, 2026-09-20)
    P63   FORMALIZADO (F2.4, 2026-09-20)
    P64   FORMALIZADO (F2.4, 2026-09-20)
    P65   FORMALIZADO (F2.4, 2026-09-20)
    P70   CERRADO (NIPC_P70_DICTAMEN.md)

    THRESHOLD_1                          UNDEFINED
    THRESHOLD_2                          BLOQUEADO
    Policy v1.0                          INTACTA (normativa vigente)
    Policy v1.3                          NO APROBADA
    Codigo productivo                    SIN CAMBIOS
    OpenFIGI masivo                      NO AUTORIZADO
    Gate-NIPC.2                          BLOQUEADO
    Gate-NIPC.3                          NO AUTORIZADO
    F2.4                                 EMITIDO 2026-09-20 (GO CONDICIONADO)
    Certificacion "acumulacion"          BLOQUEADA

---

## 11. Point-in-time Contract (P62)

**Origen:** bloqueante 2 del dictamen F2.4 (2026-09-20).

### 11.1. Principio rector

Toda entidad (catalogo, mapping, universos) que se aplique a un periodo
historico debe declarar explicitamente su validez temporal. Aplicar la
version de hoy retroactivamente introduce survivorship bias y look-ahead bias.

### 11.2. Tres timestamps obligatorios

Conservar separados:

    period_end        cierre del trimestre (13F: 31-mar, 30-jun...)
    filing_date       fecha del filing (SUBMISSION filing_date)
    knowledge_date    fecha en que el sistema conocio el dato (ingesta)

No son equivalentes. Un agregado con period_end=Q1 2026 pudo conocerse
semanas despues.

### 11.3. Catalogo versionado

`radar_target_catalog.csv` debe llevar:

    catalog_version
    catalog_valid_from
    catalog_valid_to    (NULL si vigente)

Alternativa admitida: target_catalog_as_of(period_end).

### 11.4. OpenFIGI no expone effective_date

Verificado. No asumir que una consulta actual constituye identificacion
historica. Snapshot + hash por consulta (ver seccion 13).

### 11.5. Estado

**P62 = FORMALIZADO.** Implementacion en A.6.7.

---

## 12. Missing != Sold Contract (P63)

**Origen:** dictamen F2.4 seccion 11.

### 12.1. Principio rector

"Una security no aparece" NO implica "fue vendida". La ausencia puede
deberse a: posicion bajo minimis, tratamiento confidencial, ausencia de
filing, o fallo de resolucion.

### 12.2. Seis estados obligatorios

    ZERO_REPORTED               filing existe, SHPRNAMT=0
    MISSING                     filing no incluye la security
    BELOW_REPORTING_THRESHOLD   bajo umbral minimis SEC
    CONFIDENTIAL                tratada como confidencial por la SEC
    UNRESOLVED                  identidad no resuelta
    SOLD                        evidencia directa de venta

En 13F, Q4=100 -> Q1=ausencia se etiqueta MISSING salvo evidencia adicional.

### 12.3. NO_MATCH (mapping) != NOT_TARGET (cobertura)

    NO_MATCH      OpenFIGI no resolvio el identificador.
    NOT_TARGET    la security no pertenece al universo contractual.

Prohibido transformar NO_MATCH en NOT_TARGET sin politica explicita.

### 12.4. Estado

**P63 = FORMALIZADO.** Distincion obligatoria en delta_shares.py.

---

## 13. Corporate Actions Contract (P64)

**Origen:** dictamen F2.4 seccion 12.

### 13.1. Principio rector

Un cambio de shares entre Q4 y Q1 puede ser economico (compra/venta) o
mecanico (corporate action). El sistema debe distinguirlos.

### 13.2. Eventos que rompen continuidad

    split
    reverse split
    spin-off
    merger
    share-class conversion
    CUSIP change (por reorganizacion corporativa)

### 13.3. Regla

`delta_shares.py` devuelve por observacion:

    delta_shares_economic      atribuible a compra/venta
    delta_shares_mechanical    atribuible a corporate action
    delta_shares_total         = economic + mechanical

Cuando la atribucion no es posible, delta_shares_total es el valor;
economic y mechanical quedan NULL con flag.

### 13.4. Fuentes de deteccion

Actuales: relationships.py + amendments.py.
Futuras: calendario de corporate actions (fuera de alcance v1).

### 13.5. Estado

**P64 = FORMALIZADO.**

---

## 14. Manager Duplication Contract (P65)

**Origen:** dictamen F2.4 seccion 13.

### 14.1. Principio rector

Form 13F admite estructura multi-manager. Una misma posicion economica
puede ser reportada por varios filings (other included manager, combination
report). Agregar sin resolver la estructura duplica peso.

### 14.2. Unidad obligatoria

Definir explicitamente la unidad de agregacion:

    manager            entidad legal (CIK)
    manager-group      grupo bajo una combinacion
    filing             accession individual
    position           una fila INFOTABLE
    security           CUSIP o canonical_security

La unidad por defecto del NIPC es `security`. Cuando el universo admite
multiples filings para la misma posicion economica, se aplica deduplicacion.

### 14.3. Regla

Prohibido agregar el universo 13F por CUSIP sin resolver previamente la
estructura del filing.

### 14.4. Estado

**P65 = FORMALIZADO.**


---

Fin del documento. F2.4 emitido 2026-09-20. Pendiente A.6.0 (Gate 0 bloqueantes).
