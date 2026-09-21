# NIPC_CONTRATOS_SEMANTICOS_v1

**Objeto:** contrato habilitante de identidad, validez temporal y cobertura pairwise para el modulo NIPC.
**Estado del sistema:** ver `ESTADO_SISTEMA.md` (hechos) + `ESTADO_DECLARADO.md` (declaraciones).
**Estado:** VIGENTE (2026-09-21). F2.4 EMITIDO (GO CONDICIONADO). A.6.0-A.6.4 cerrados (dictamenes #43-#75). H-73.1 corregido. Q12 Modelo A confirmado por test. Ver DICTAMENES.md #24, #72, #75.

**Precedencia transitoria:** mientras este contrato y NIPC_COVERAGE_POLICY_V13_PROPUESTA.md sean borradores, NIPC_COVERAGE_POLICY.md v1.0 permanece como referencia normativa vigente. Una vez aprobado formalmente este contrato y aplicada la policy v1.3, el contrato habilitante prevalece sobre cualquier parafrasis de la policy.
**Origen:** dictamen formal del auditor (2026-09-20), que clasifico P38/P60/P61 como NO GO con contrato requerido.
**Relacion:** cierra la seccion 2, 3 y 4 del dictamen. P70 cerrado por NIPC_P70_DICTAMEN.md.
**Autor:** Ingeniero Supervisor (revision interna).
**Fecha:** 2026-09-21.

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

**Estado tras este documento:** P60 GO. P61 GO. P38 GO CONDICIONADO.
Cierres aplicados: A.6.0-A.6.4 cerrados (dictamenes #43-#75). Q12 Modelo A
confirmado por test (dictamen #72). H-73.1 corregido en adapter B1<->P38
(dictamen #75). R-69.2 vigente.

**Implementacion de codigo y fixes quirurgicos autorizados por F2.4.**

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

**P60 = GO (F2.4, 2026-09-20) / IMPLEMENTADO / CERRADO (A.6.2).**

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

**P61 = GO (F2.4, 2026-09-20) / IMPLEMENTADO / CERRADO (A.6.2).**

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

- Q12 Modelo A confirmado por test (dictamen #72).
- A.6.4-v2 (dictamen #73): P38 aislado sobre TOP 2000 -> VALID 1.0/1.0/1.0/1.0.
- A.6.4 (dictamen #75): integracion end-to-end B1+P61+P38 -> VALID.
  Q1: 20 keys -> 18 FIGI -> 18 VERIFIED via adapter.
  Q4: 0 keys -> fail-closed (no se fabrica TARGET).
- H-73.1 corregido: adapter mapea `weight_status` a
  `operational_mapping_status`. Regresion en
  `tests/test_h731_adapter_p38_compat.py` (5 tests).
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

Estado actual: F2.4 GO (2026-09-20). Pasos 1-5 ejecutados:
contrato vigente, tests especificos (P60/P61/P38) y fixes quirurgicos
aplicados. Paso 6 parcial (A.6.4-v2 sobre TOP 2000). Pasos 7-9
bloqueados: OpenFIGI masivo NO AUTORIZADO, thresholds UNDEFINED.

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
    radar_target_catalog.csv:     e5d8f9c86b07c4d3981b2fd3c3a804f75d3559fa33fbb7731bfc76308a60b292
    probe_p70_result.json:        809c51af47ae154c6c01bba1cf3983a6c0143fd3b0711958fa8b9e6e18bae96f
    probe_p70_summary.txt:        9e8c42d6a742d773be7e531782a7d8514a5c00aa64efc5080d81a718d6384fe5
    NIPC_P70_DICTAMEN.md:         e8d4e4e825fe6136b23573705b800108247eb07c9a46e2f62c2c06980f179092 (historico; consolidado en DICTAMENES.md seccion 23)

Nota EOL (2026-09-21): radar_target_catalog.csv registra hash sobre
variante CRLF (pre-.gitattributes). El hash vigente corresponde a la
variante LF (normativa por .gitattributes). Contenido identico.

### 10.3. Estado de bloqueos

    P38   GO CONDICIONADO / A.6.4 CERRADO (dictamen #75)
    P60   GO / IMPLEMENTADO (A.6.2)
    P61   GO / IMPLEMENTADO (A.6.2)
    P62   FORMALIZADO (F2.4, 2026-09-20)
    P63   FORMALIZADO (F2.4, 2026-09-20)
    P64   FORMALIZADO (F2.4, 2026-09-20)
    P65   FORMALIZADO (F2.4, 2026-09-20)
    P70   CERRADO (NIPC_P70_DICTAMEN.md, consolidado en DICTAMENES.md seccion 23)

    A.6.0   CERRADO (dictamen #43)
    A.6.1   CERRADO (F2.4, 2026-09-20)
    A.6.2   CERRADO
    A.6.2-bis-B2-PIT  CERRADO (dictamen #53)
    A.6.2-bis-B1      CERRADO (dictamen #68)
    A.6.2-bis-B3      IMPLEMENTADO SPEC-SIDE
    A.6.3   CERRADO (dictamen #72)
    A.6.4-v2 CERRADO (dictamen #73)
    A.6.4   CERRADO (dictamen #75)
    A.6.5   CERRADA (dccf70d + 911e95b)
    A.6.6   PENDIENTE

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

**Origen:** dictamen F2.4 seccion 11 + dictamen P63 formal (2026-09-20,
ver DICTAMENES.md #25).

### 12.1. Principio rector

"Una security no aparece en el filing publico" NO implica "fue vendida".
La ausencia observada puede deberse a causas distintas (minimis,
confidencialidad, otro manager, omision). El canal 13F aislado NO
permite identificar la causa. Por eso hay que separar **hecho observado**
de **causa inferida/conocida**.

### 12.2. Distincion observacion vs causa (revisada por F2.4)

La ausencia observada es un HECHO DOCUMENTAL. La causa es una
CLASIFICACION. La dimension contractual se estructura asi:

    reporting_status (observado)
      PRESENT              la security aparece en el snapshot efectivo.
      ZERO_REPORTED        fila INFOTABLE con SSHPRNAMT=0 (no demostrado
                           como convencion universal; capacidad propuesta,
                           requiere fixture real SEC).
      NOT_PRESENT          la security no aparece en el snapshot efectivo.

    absence_reason (inferida/conocida, si NOT_PRESENT)
      MISSING              ausencia sin causa demostrable. Estado por
                           defecto. NO equivale a "vendio".
      BELOW_REPORTING_THRESHOLD   requiere evidencia. NO se infiere
                           solo por ausencia.
      CONFIDENTIAL         detectable a nivel filing via 13F-CTR / metadata
                           de tratamiento confidencial. Atribucion
                           security-level requiere evidencia suficiente.
      OTHER_MANAGER        el filing puede no incluir todas las posiciones
                           (Combination Report / 13F-NT). Requiere resolver
                           relaciones OTHERMANAGER antes de clasificar.
      UNKNOWN              sin informacion adicional.

    identity_status (ortogonal)
      RESOLVED | UNRESOLVED

    sale_evidence (independiente)
      NONE                 por defecto. 13F delta NO crea SOLD.
      DIRECT               requiere fuente externa que lo demuestre.

El estado `SOLD` NO existe como miembro directo del enum. Solo puede
alcanzarse via `sale_evidence = DIRECT` con fuente adicional.

### 12.3. NO_MATCH (mapping) != NOT_TARGET (cobertura)

    NO_MATCH      OpenFIGI no resolvio el identificador.
    NOT_TARGET    la security no pertenece al universo contractual.

Prohibido transformar NO_MATCH en NOT_TARGET sin politica explicita.

### 12.4. Ocho reglas del dictamen P63 formal

    R1  EXIT does not imply SOLD.
    R2  Absence from a public 13F Information Table does not by itself
        identify the cause of absence.
    R3  MISSING means unreported/absent with no stronger supported
        absence classification.
    R4  BELOW_REPORTING_THRESHOLD requires supporting evidence;
        it is not inferred merely from absence.
    R5  CONFIDENTIAL may be detected at filing level through 13F
        confidential-treatment metadata, but security-level attribution
        requires sufficient evidence.
    R6  13F amendments/restatements must be resolved before
        quarter-to-quarter delta computation.
    R7  13F-NT / Combination / Other Manager relationships must be
        resolved before interpreting absence at manager level.
    R8  SOLD requires direct external evidence.
        13F delta alone never creates SOLD.

### 12.5. Arquitectura recomendada por el dictamen

    SEC FILINGS
        |
        +--- base filing
        +--- amendments
        |
        v
    EFFECTIVE SNAPSHOT
        |
        +--- identity
        +--- reporting
        +--- metadata
        |
        v
    ABSENCE CLASSIFIER (capa posterior)
        |
        +--- UNKNOWN / MISSING
        +--- CONFIDENTIAL (con evidencia)
        +--- OTHER_MANAGER (con evidencia)
        |
        v
    SALE EVIDENCE
        NONE / DIRECT

`delta_shares.py` NO es un clasificador economico de ventas. Su funcion
es reconciliar MATCH_KEY entre periodos y producir delta. La clasificacion
de la causa vive en una capa posterior que conoce filing metadata,
amendments, 13F-CTR, other managers e identidad.

### 12.6. Evidencia empirica

Test contractual en `tests/test_sec_13f_delta_shares.py`:

    test_p63_amendment_evita_exit_falso
      Demuestra R6: Q4 con AAPL + Q1 base sin AAPL + Q1/A con NEW
      HOLDINGS -> snapshot efectivo incluye el amendment -> delta
      produce BOTH, NO EXIT. Si el sistema usara el filing original,
      produciria una venta falsa.

    test_p63_othermanager_produce_exit_mas_new
      Documenta R7: filing_manager_cik parent en Q4 + child en Q1
      produce EXIT parent + NEW child. NO es bug: implementa C2
      fielmente. La resolucion de relaciones queda como capacidad
      diferida v1 (filer continuity = caracterizacion, no threshold).

    test_p63_no_hay_estado_sold_fabricado (assert inline)
      Documenta R8: "SOLD" no existe en ALL_MATCH_STATUSES.

### 12.7. Matices del dictamen sobre la version previa

Dos afirmaciones de la version anterior del contrato requieren matiz:

    (a) "MISSING = filing no incluye la security". La version revisada
        define MISSING como "ausencia sin causa demostrable", NO como
        "toda ausencia". La equivalencia fuerte EXIT -> MISSING se
        rechaza.

    (b) "CONFIDENTIAL es completamente invisible en EDGAR". Desde
        2023-02-28, las solicitudes 13F-CTR son electronicas. El
        filing publico puede indicar que hubo posiciones omitidas.
        Deteccion filing-level: posible. Atribucion security-level:
        requiere evidencia.

### 12.8. Estado

**P63 = GO CONDICIONADO (F2.4 formal, 2026-09-20).**

    Implementado: docstring + 2 tests contractuales.
    Diferido v1: clasificacion automatica de absence_reason desde
                 metadata de filing (13F-CTR, OTHERMANAGER).
    Fuera de alcance v1: BELOW_REPORTING_THRESHOLD a nivel security
                 (requiere universo completo del manager).
    No aplicable a delta_shares.py: clasificacion economica de ventas.

---

## 13. Corporate Actions Contract (P64)

**Origen:** dictamen F2.4 seccion 12 + dictamen P64 formal (2026-09-20).
**Estado:** CERRADO 2026-09-20. Commit 2f8140a.

### 13.1. Definicion contractual vigente

    delta_shares  =  GROSS_OBSERVED_DELTA

NO es delta_economic. 13F aislado NO permite distinguir compra/venta
de split/spin-off/merger. La separacion economic/mechanical queda como
capacidad diferida v1.

### 13.2. Eventos fuera de alcance v1

    split
    reverse_split
    spin_off
    merger
    share_class_conversion

Declarados en constante P64_EVENTS_DEFERRED (delta_shares.py).

### 13.3. CUSIP change: identidad, no evento

CUSIP_A -> canonical_X y CUSIP_B -> canonical_X produce BOTH.
Esto demuestra continuidad de identidad; NO demuestra deteccion del
evento societario concreto que provoco el cambio.

    IDENTITY CONTINUITY  !=  CORPORATE ACTION DETECTION

### 13.4. Codigo

delta_shares.py:
- DELTA_SEMANTICS_GROSS_OBSERVED = True
- P64_EVENTS_DEFERRED = (split, reverse_split, spin_off, merger, share_class_conversion)
- Docstring: GROSS OBSERVED DELTA
- SIN columnas delta_shares_economic / delta_shares_mechanical

### 13.5. Tests contractuales

tests/test_sec_13f_delta_shares.py:
- test_p64_semantica_gross_observed_delta
- test_p64_delta_no_lleva_columnas_economic_mechanical
- test_p64_split_no_se_etiqueta_como_economico
- test_p64_cusip_change_es_identidad_no_corporate_action

### 13.6. Audit de consumidores

Audit realizado 2026-09-20: solo nipc.py::_sum_delta consume
delta_shares; lee como magnitud, no como compra/venta. Probes solo
imprimen. Cero consumidores que interpreten delta como economic.

### 13.7. Fuente externa requerida para v2

La materializacion de economic/mechanical requiere fuente externa
de eventos societarios (CRSP, Compustat, OpenFIGI events, equivalente).
NO heuristica. NO deteccion por ratios limpios.

---

## 14. Manager Duplication Contract (P65)

**Origen:** dictamen F2.4 seccion 13 + dictamenes P65 v1/v2/v3 (2026-09-20).
**Estado:** GO CONDICIONADO v3. 3 correcciones de cierre pendientes. Ver
iae/P64_P65_EXPEDIENTE.md seccion 2.

### 14.1. Regla transversal

    REPORTING RELATIONSHIP  !=  REPORTING NETWORK  !=  DEDUP AUTHORIZATION  !=  ECONOMIC OWNERSHIP

- Una relacion OTHERMANAGER resuelta prueba una relacion de reporting.
- NO prueba por si sola que dos observaciones sean duplicadas.
- NO prueba por si sola que un EXIT + NEW sea un handoff.
- economic_owner_cik PROHIBIDO. En relationships.py::FORBIDDEN_TERMS.
- Un HR normal puede contener holdings de otros managers incluidos.
- Managers bajo control comun pueden presentar 13F-HR separados.
- 13F-NT no tiene Information Table. Aporta L1; L3 solo por evidencia cruzada.

### 14.2. Los 3 niveles de evidencia

| Nivel | Nombre | Significado | Autoriza |
|---|---|---|---|
| L1 | REPORTING_EDGE | Relacion documental (Column 7 + OM2 resuelto) | Nada |
| L2 | POSITION_SCOPED_EDGE | Relacion vinculada a security concreta | Nada |
| L3 | DEDUP_AUTHORIZATION | Evidencia cruzada permite afirmar duplicidad | DROP_DUP |

### 14.3. L3 - evidencia cruzada entre filings

    L3(A, B, S, period) == True sii:
      R1(A, S, period) == True
      AND R2(A, B, period) == True
      AND R3(B, period) == TRUE
      AND R4(A, B) == MATCH
      AND R5(A, B, S, period) == True

Los estados N/D y CONFLICT de R3 y R4 se preservan. Nunca se
convierten internamente en False. Impiden L3 == True.

---

#### 14.3.1. R3(B, period) - determinacion del filing efectivo

Flujo unico en cuatro pasos secuenciales. No existen reglas
solapadas.

PASO 0. Verificacion de integridad y completitud del scope R4.

    Antes de poder afirmar cualquier resultado de R3 debe estar
    demostrada la completitud del scope CIK + PERIODOFREPORT:

        - ingestion completa y verificada;
        - disponibilidad integra de los registros necesarios para
          clasificar BASE / AMENDMENT;
        - ausencia de registros R4 con SUBMISSIONTYPE / REPORTTYPE
          ausentes, invalidos o no clasificables.

    Si la completitud del scope NO puede demostrarse:
        R3 = N/D.

    Si existe un registro potencialmente R4 cuya clasificacion no
    puede determinarse inequivocamente:
        R3 = N/D.

    Solo despues se procede con los pasos siguientes.

    Semantica de R3 = FALSE: significa "se ha demostrado que no
    existe ningun filing R4 en el scope definido", NO simplemente
    "no aparece ninguno en el dataset disponible".

        scope completo + 0 R4           -> R3 = FALSE
        scope no demostrable + 0 R4     -> R3 = N/D

PASO 1. Identificar todos los filings R4 del CIK + PERIODOFREPORT.

    R4_filings(B, period) = filings de B con:
        mismo CIK
        AND mismo PERIODOFREPORT
        AND (NOTICE family o COMBINATION family)

    NOTICE family:
        SUBMISSIONTYPE in {13F-NT, 13F-NT/A}
        AND REPORTTYPE = 13F NOTICE

    COMBINATION family:
        SUBMISSIONTYPE in {13F-HR, 13F-HR/A}
        AND REPORTTYPE = 13F COMBINATION REPORT

PASO 2. Clasificar cada filing por rol.

    BASE:
        SUBMISSIONTYPE in {13F-NT, 13F-HR}

    AMENDMENT:
        SUBMISSIONTYPE in {13F-NT/A, 13F-HR/A}

    ISAMENDMENT es SOLO control de coherencia, no criterio de
    seleccion. Si contradice SUBMISSIONTYPE: R3 = N/D.

PASO 3. Detectar pluralidad de familias documentales.

    Si entre BASE + AMENDMENT hay representacion de MAS DE UNA
    familia (NOTICE y COMBINATION simultaneamente):
        R3 = N/D.

    Deteccion GLOBAL sobre el conjunto R4 completo del
    CIK + PERIODOFREPORT. No se evalua por familia aislada.

PASO 4. Conteo y validacion.

    BASE = 0 AND AMENDMENT = 0      -> R3 = FALSE
    BASE = 0 AND AMENDMENT >= 1     -> R3 = N/D
    BASE = 1                        -> validar cadena (§14.3.2)
    BASE > 1                        -> R3 = N/D

    Prohibido como criterio de resolucion:
        - primero encontrado
        - ultimo encontrado
        - MAX(ACCESSION_NUMBER)
        - MAX(FILING_DATE)

    CONFLICT no puede producirse en R3 por pluralidad documental.
    CONFLICT pertenece exclusivamente a la evaluacion de identidad
    de A dentro de R4 (§14.3.4).

---

#### 14.3.2. Validacion de la cadena de amendments

Aplica solo cuando §14.3.1 PASO 3 confirma familia unica y PASO 4
confirma BASE = 1.

    AMENDMENT valido:
        SUBMISSIONTYPE in {13F-NT/A, 13F-HR/A}
        AND REPORTTYPE consistente con la familia de la base
        AND AMENDMENTNO entero en 1..99
        AND AMENDMENTTYPE in {RESTATEMENT, NEW_HOLDINGS}

    Si existen amendments aplicables a la base:

        - numeros unicos;
        - secuencia sin huecos desde 1 hasta N;
          (Regla conservadora contractual IAE. La SEC exige
          numeracion 1..99 y orden, pero no que la secuencia
          este fisicamente completa. Se conserva la regla por
          prudencia fail-closed.)
        - tipos reconocibles.

    Si cualquiera falla: R3 = N/D.

Casos que producen N/D:
    - huecos: base + amendment 1 + amendment 3 (falta el 2);
    - duplicados: base + amendment 1 + amendment 1;
    - amendment sin numero determinable;
    - amendment con tipo desconocido.

---

#### 14.3.3. Semantica de amendments

    RESTATEMENT:
        Sustituye la evidencia OTHERMANAGER anterior.

    NEW HOLDINGS:
        Conserva la evidencia cuando es consistente con el estado
        efectivo previo.

        La comparacion solo se realiza si AMBOS estados (previo y
        amendment) son completamente determinables. Cualquier N/D
        o CONFLICT en cualquier fila de cualquiera de los dos
        estados impide la comparacion -> N/D.

        Cuando ambos son determinables, "consistente" significa
        igualdad exacta de conjunto de identidades de managers
        obtenidas tras:
            - resolver CIK;
            - resolver FormNum;
            - canonicalizar FormNum (§14.3.6).

        Cambio de conjunto -> N/D. NO asumir union.
        NO asumir sustitucion.

---

#### 14.3.4. R4(A, B) - estados y candidate_A

Definiciones:

    MATCH
        Existe al menos una fila OTHERMANAGER que identifica
        inequivocamente a A mediante CIK o Form13FFileNumber.

    NO_MATCH
        El filing efectivo es determinable, la ingestion es integra,
        TODAS las filas OTHERMANAGER tienen identidad resuelta
        inequivocamente, y A no aparece.

    N/D
        El filing efectivo no es determinable, o existe al menos
        una fila OTHERMANAGER cuya identidad no puede resolverse
        inequivocamente y no existe evidencia positiva inequivoca
        de A.

    CONFLICT
        La evidencia candidata a identificar A contiene
        identificadores contradictorios o una resolucion no univoca.

Reglas duras:

    N/D != NO_MATCH.
    CONFLICT != NO_MATCH.
    CONFLICT PREVALECE sobre MATCH.
    N/D y CONFLICT son fail-closed respecto de DROP_DUP.

candidate_A(r):

    candidate_A(r) :=
        CIK(r) == CIK_A
        OR
        CIK_A ∈ resolved_ciks(FormNum(r))

    donde resolved_ciks(FormNum) es el conjunto de CIKs a los que
    resuelve el FormNum normalizado (§14.3.6) dentro del scope
    temporal.

CONFLICT > MATCH aplica a CUALQUIER fila con candidate_A(r) == True:

    - CIK(r) y FormNum(r) apuntan a CIK distintos; o
    - FormNum(r) resuelve a >1 CIK dentro del scope temporal; o
    - FormNum(r) contradice el CIK(r) explicitamente declarado.

Regla de no-seleccion selectiva: la implementacion NO puede elegir
la fila consistente e ignorar la contradictoria respecto de A. Si
existe contradiccion entre dos filas con candidate_A == True, el
resultado es CONFLICT, no MATCH.

Un conflicto perteneciente exclusivamente a otro manager (no
candidate_A) NO invalida por si mismo un MATCH de A.

Orden de evaluacion:

    1. ¿Filing efectivo determinable?
       NO -> N/D
    2. ¿CONFLICT en alguna fila con candidate_A(r) == True?
       SI -> CONFLICT
    3. ¿Existe fila que identifica inequivocamente a A?
       SI -> MATCH
    4. ¿Todas las filas OTHERMANAGER tienen identidad resuelta?
       NO -> N/D
    5. NO_MATCH

OTTOMANAGER ausente o vacio (filing efectivo R4, ingestion integra)
-> N/D. Nunca NO_MATCH.

---

#### 14.3.5. Mapping FormNum -> CIK

    FormNum + period:
        -> 0 CIK    = UNRESOLVED
        -> 1 CIK    = IDENTITY_RESOLVED
        -> >1 CIK   = CONFLICT

Construccion:

    COVERPAGE
        JOIN SUBMISSION ON ACCESSION_NUMBER
        Filtro: FORM13FFILENUMBER no-null AND CIK no-null
        Filtro: PERIODOFREPORT == period
        Agrupacion: FormNum normalizado -> CIK

Cardinalidad observada: 1:1 estricta (0 casos N:1). No se exige
la direccion inversa CIK -> FormNum.

---

#### 14.3.6. Canonicalizacion de Form13FFileNumber

Regla IAE basada en correspondencia observada con COVERPAGE. NO
es una afirmacion de formato SEC universal.

    entrada: <prefijo>-<sufijo>

    prefijo:
        in {"28", "028"}   -> canonicalizar a "028"
        cualquier otro     -> UNRESOLVED

    sufijo:
        strip_leading_zeros
        si len > 5         -> UNRESOLVED (no truncar)
        sino               -> padding a 5 digitos

---

#### 14.3.7. Clausula de cierre contractual

El GO contractual sobre este texto autoriza su traslado y mantiene
su vigencia sin nueva iteracion ordinaria. Cualquier observacion
posterior debera clasificarse expresamente como:

    A) BLOQUEO MATERIAL
       Cuando la correccion pueda cambiar el resultado contractual,
       modificar una condicion necesaria de L3/R3/R4/R5, alterar
       el criterio de evidencia, introducir una nueva semantica o
       dejar una ambiguedad que permita resultados distintos entre
       implementaciones conformes al contrato.
       -> requiere nueva version y nuevo dictamen.

    B) RECOMENDACION DIFERIBLE
       Cuando la observacion sea exclusivamente editorial,
       documental, de legibilidad o de implementacion y no pueda
       cambiar el resultado contractual definido por este texto.
       -> no bloquea el GO.

### 14.4. R1 - Intra-period dedup (7 requisitos)

    Para dos observaciones S1 (M1) y S2 (M2) del mismo periodo:
      1. S1.canonical_security == S2.canonical_security
      2. S1.discretion_type    == S2.discretion_type
      3. S1.filing_manager_cik != S2.filing_manager_cik
      4. Existe relacion documental explicita M1 -> M2 (Column 7 de S1)
      5. La relacion esta vinculada a la security S concreta
      6. Existe evidencia cruzada L3: M2 declara que M1 reporta por M2
      7. NO existe evidencia contradictoria ni overlap no resuelto

    Si las 7 se cumplen:
      dedup_decision = DROP_DUP
      dedup_reason = INTRA_PERIOD_DUP
      se conserva S1 (documented reporting representative)
      dedup_audit registra la eliminacion.

    Si falla cualquier requisito:
      dedup_decision = KEEP
      dedup_reason = NULL | DUPLICATION_UNRESOLVED |
                     REPORTING_OVERLAP_UNRESOLVED | REPORTING_CONFLICT

### 14.5. R2 - Reporting handoff inter-periodo

    Q4 unit A:  filing_manager = A, reporting_for_manager = B, security = S
    Q1 unit A': filing_manager = B, reporting_for_manager = B, security = S

    AND:
      - A y A' producen EXIT + NEW en delta_shares (C2 intacto)
      - reporting_for_manager_cik ESTABLE (B == B)
      - continuidad documental explicita de la misma relacion de reporting

      -> reporting_transition = HANDOFF
      -> dedup_reason = POSITION_SCOPED_HANDOFF
      -> match_status = EXIT + NEW (sin cambios)
      -> delta_shares (sin cambios)

    NO se anade STATUS_REPORTING_HANDOFF a delta_shares.py.

### 14.6. R3, R4, R5

    R3  Sin RESOLVED o sin included_manager_cik -> KEEP.
    R4  Amendments antes de relaciones. 2 tests obligatorios
        (RESTATEMENT + NEW HOLDINGS).
    R5  13F-NT: holdings=0, relacion preservada. NT aporta L1,
        no L3 por si solo.

### 14.7. Casos especiales

    REPORTING_OVERLAP_UNRESOLVED:
      A reports for B on X  +  B also reports X, sin evidencia de si
      portion A + portion B son complementarias o duplicadas.
      -> dedup_decision = KEEP
      -> dedup_reason = REPORTING_OVERLAP_UNRESOLVED

    REPORTING_CONFLICT (definicion estricta):
      B declara explicitamente una condicion incompatible con
      "A reports for B on X" (contradiccion documental explicita).
      NO por mera coincidencia de security.
      -> dedup_decision = KEEP
      -> dedup_reason = REPORTING_CONFLICT

### 14.8. reporting_for_manager_cik

    filing_manager_cik         quien presento el filing
    reporting_for_manager_cik  para que manager se reporta esta observacion
    economic_owner_cik         NO EXISTE. Prohibido.

    reporting_for_manager_cik NO se usa para dedup. Se usa solo en R2.

### 14.9. dedup_audit - trazabilidad obligatoria

Toda fila eliminada (DROP_DUP) debe quedar registrada con:

    source_line_id, period, filing_manager_cik, reporting_for_manager_cik,
    canonical_security, dedup_decision, dedup_reason, evidence_level,
    evidence_source, evidence_accession_representante,
    evidence_accession_representado, evidence_reference_seq

### 14.10. reporting_network_id

Solo diagnostico. NO participa en DROP_DUP ni HANDOFF.

### 14.11. Arquitectura aprobada

    SEC FILINGS -> AMENDMENTS -> EFFECTIVE SNAPSHOT
        -> REPORTING RELATIONSHIP RESOLUTION (L1/L2/L3)
        -> INTRA-PERIOD DEDUP (PRE-delta)
        -> effective_Q4/effective_Q1 -> delta_shares (C2 INTACTO)
        -> REPORTING TRANSITION ENRICHMENT (POST-delta)
        -> NIPC / coverage

    No toca: MATCH_KEY, C2, delta_shares.py, nipc.py, coverage.py,
    relationships.py.

### 14.12. Implementacion en 4 commits

    Commit 1   contrato + modelos + tests unitarios L1/L2/L3
    Commit 2   reporting_dedup pre-delta
    Commit 3   transition classification
    Commit 4   integracion e2e + evidencia

Justificacion metodologica: permite identificar que modificacion
introduce cualquier cambio de resultado.

### 14.13. Fuera de alcance v1

- Economic owner (prohibido inferir).
- Persistencia de reporting_network_id entre periodos.
- Deduplicacion cross-periodo automatica (solo HANDOFF con evidencia).
- Regla de precedencia para REPORTING_CONFLICT (diferida).

### 14.14. Tests obligatorios

15 tests en iae/P64_P65_EXPEDIENTE.md seccion 2.15.


---

Fin del documento. F2.4 emitido 2026-09-20. A.6.0-A.6.4 cerrados (dictamenes
#43-#75). A.6.5 CERRADA. Pendiente A.6.6 (F2.4-CLOSE).
