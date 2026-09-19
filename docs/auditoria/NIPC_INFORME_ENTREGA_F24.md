# INFORME DE ENTREGA AL AUDITOR - SOLICITUD DE F2.4

**Objeto:** entrega de la cadena documental y tecnica del ciclo post-F2.3-bis + solicitud de dictamen F2.4.
**HEAD al redactar:** a9bb1fb.
**Estado:** BORRADOR pendiente de envio.
**Autor:** Ingeniero Supervisor.
**Fecha:** 2026-09-20.
**Destinatario:** auditor externo (dictamen F2.4).

---

## 0. Resumen ejecutivo

Esta entrega cierra el ciclo abierto tras el dictamen F2.3-bis (PASS CONDICIONADO) y
solicita el dictamen F2.4.

**Trabajo ejecutado en este ciclo:**

- Curacion CUSIP Q1 2026 (3 filas COM): DD, HON, XOM.
- Coverage baseline Fase A (4 universos anidados, 6 metricas pairwise).
- F2.1 (inventario contrato coverage).
- F2.1-bis (fuentes de identidad, declaracion TARGET_CUSIP_REGISTRY = NOT AVAILABLE).
- F2.2 (propuesta v1.1, NO GO).
- F2.2-v2 (propuesta v1.2, BORRADOR rechazado por F2.3).
- F2.3 (dictamen NO GO v1.2).
- Implementacion de OpenFIGI + RADAR_TARGET_CATALOG + target_universe (7 commits funcionales).
- F2.3-bis (PASS CONDICIONADO, H1 CERRADO).
- TOP 2000 (validacion cuantitativa ponderada).
- Revision estructural completa del modulo IAE (95 hallazgos clasificados).
- P70 (invariante de classify_strategy, cerrado con probe Q4 2025 + Q1 2026).
- P38 / P60 / P61 (contratos semanticos congelados).
- Policy coverage v1.3 propuesta (borrador).

**Bloqueos resueltos en este ciclo:**

    P70   classify_strategy                CERRADO / GO CONDICIONADO
    P60   canonical_security prefijos      CERRADO contractualmente
    P61   etf_holdings sin vigencia        CERRADO contractualmente
    P38   paired_weighted_share_coverage   CERRADO contractualmente

**Estado de la cadena documental:**

    Policy v1.0                          INTACTA (hash 57f2d01f...)
    Policy v1.1                          INTACTA (historico, NO GO F2.2)
    Policy v1.2                          INTACTA (historico, NO GO F2.3)
    Contrato semantico v1                NUEVO (aprobado internamente)
    Policy v1.3 propuesta                NUEVO (borrador)
    Dictamen P70                         NUEVO (cerrado)
    Revision estructural IAE             NUEVO (diagnostico)

**Bloqueos vigentes (sin cambios):**

    THRESHOLD_1                          UNDEFINED
    THRESHOLD_2                          UNDEFINED
    Gate-NIPC.2                          BLOQUEADO
    Gate-NIPC.3                          NO AUTORIZADO
    F2.4                                 (esta entrega la solicita)
    OpenFIGI masivo (24.838 CUSIPs)      NO AUTORIZADO
    Policy v1.3 aplicacion                NO AUTORIZADA

---

## 1. Contexto

Al cierre del dictamen F2.3-bis (2026-09-19):

- H1 (circularidad crosswalk interno) CERRADO.
- RADAR_TARGET_CATALOG materializado (242 filas, 240 OK).
- TARGET_UNIVERSE resolver operativo.
- TOP 500 + TOP 2000 ejecutados.
- Pendiente: cierre de los 4 bloqueos semanticos antes de fijar thresholds.

**El presente ciclo cierra esos 4 bloqueos y entrega la propuesta normativa
derivada (v1.3).**

---

## 2. Cadena de trabajo ejecutada

### 2.1. Curacion CUSIP Q1 2026 (Q-CUR)

- 3 filas COM pobladas en `data/mappings/cusip_ticker_exceptions.csv`:
  DD (26614N102), HON (438516106), XOM (30231G102).
- Excluidos CALL/PUT (derivados) y HONA/FDXF (entidades post-Q1).
- Informe: INSTITUTIONAL_ACCUMULATION_CUSIP_CURATION_INFORME.md.
- Commits: e4a6576, e8d7e53, 9d4a81e.

### 2.2. Coverage baseline Fase A

- 4 universos anidados: RAW_13F, ELIGIBLE_SEC, TECHNICAL_RADAR, OPERATIONAL_EQUITY.
- 6 metricas pairwise sobre cada uno.
- Hallazgo H1: TECHNICAL_RADAR == OPERATIONAL_EQUITY en units (filtro EQUITY no-op).
- Hallazgo H2: coverage_previous == coverage_current == 1.0 en OPERATIONAL.
- Hallazgo H3: NIPC cambia signo entre RAW (+23.65M) y OP_EQUITY (-109.46M).
- Informe: INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_BASELINE_INFORME.md.
- Commit: e9fd830.

### 2.3. F2.1 - Inventario contrato coverage

- Inventario linea-por-linea de TARGET/RESOLVED/PAIRED en policy v1.0.
- Hallazgo: TARGET y RESOLVED con 0 ocurrencias literales.
- Commit: eda67ef.

### 2.4. F2.1-bis - Fuentes de identidad

- Verificacion de 6 fuentes en disco.
- Ninguna cumple (CUSIP + radar USA + independiente del crosswalk).
- Declaracion: `TARGET_CUSIP_REGISTRY = NOT AVAILABLE (2026-09-19)`.
- Confirmacion numerica de la circularidad (90.91% del etf_holdings.csv).
- Commit: 4659ce1.
### 2.5. F2.2 - Propuesta v1.1 (NO GO)

- Propuesta de TARGET/RESOLVED/PAIRED conceptualmente correcta.
- NO GO: claves incompatibles (ticker ^ CUSIP ^ ticker-por-class).
- 8 cambios obligatorios para v1.2.
- Commit: d4a926e.

### 2.6. F2.2-v2 - Propuesta v1.2 (BORRADOR)

- Aplicacion de los 8 cambios obligatorios.
- Estado: BORRADOR, NO vigente.
- Commit: 0518b96.

### 2.7. F2.3 - Dictamen NO GO v1.2

- Dictamen del auditor: NO GO.
- Autorizacion de OpenFIGI como capa de resolucion.
- Q1=GO N-PORT auxiliar, Q2=GO OpenFIGI, Q3=NO FIJAR, Q4=GO fix N-PORT, Q5=CERRADO.

### 2.8. OpenFIGI + RADAR_TARGET_CATALOG + target_universe

7 commits funcionales:

1. a130bf1 - fix N-PORT pivot por HOLDING_ID + SERIES_ID.
2. 3fccc93 - openfigi_client.py + 8 tests.
3. 7501825 - micro-probe identidad radar (240/242 con FIGI).
4. a530ee3 - radar_target_catalog.py + 8 tests.
5. ae6129b - radar_target_catalog.csv materializado (242 filas).
6. f81fc17 - target_universe.py + 8 tests.
7. 32baf9d - piloto top 500 CUSIPs 13F.

Resultado: `TARGET_CUSIP_REGISTRY` deja de ser NOT AVAILABLE. 240/242
resueltos por shareClassFIGI.

### 2.9. F2.3-bis - PASS CONDICIONADO

- H1 (circularidad crosswalk interno) CERRADO arquitectonicamente.
- GO TOP 2000 con condiciones (medir count + weight).
- GO OpenFIGI como capa de resolucion (no como autoridad).
- NO autorizado OpenFIGI masivo.
- 3 issues abiertos: temporalidad catalogo, peso de NO_ID/ERROR, dependencia de proveedor.
- Commit: 3ef4d2d.

### 2.10. TOP 2000 - validacion cuantitativa ponderada

- Convencion: RADAR_SNAPSHOT_DATE=2026-09-19, 13F_OBSERVATION_PERIOD=2026-03-31.
- RADAR_MEMBERSHIP_MODE = CURRENT_RETROSPECTIVE.
- Resultado RAW: 210 target (10.50% count, 32.81% weight).
- Resultado CORREGIDO: 212 target (10.60% count, 33.34% weight).
- Delta +0.5349 pp (2 canales: cusip_ticker_exceptions + ticker match).
- Hallazgos materiales: count/weight divergen 2.70x, shareClassFIGI cambia tras
  reorganizacion (XOM), HON fuera del radar actual, 221 NO_ID genuinos en prefijos
  no-USA.
- 4 deudas declaradas (D1-D4).
- Commit: 3ef4d2d.

### 2.11. Revision estructural del modulo IAE (95 hallazgos)

- Revision completa de 22 ficheros .py (2.456 LOC no-vacias), 17 tests,
  53 documentos.
- Resultado: 95 observaciones clasificadas (3 ALTA, 20 MEDIA, 23 BAJA,
  49 INFORMATIVO). 2 hipotesis iniciales cerradas como NO BUG (P14, P29).
- 4 candidatos criticos: P38, P60, P61, P70.
- Informe: INSTITUTIONAL_ACCUMULATION_REVISION_ESTRUCTURAL_2026-09-20.md (1180 lineas).
- Commit: f107b27.

### 2.12. P70 - cierre de classify_strategy

- Origen: asimetria iloc[1] vs iloc[-1] en classify_strategy.
- Probe empirico Q4 2025 + Q1 2026:
  - 195 grupos HR_PLUS_RESTATEMENT auditados (100 + 95).
  - Cero casos len > 2. Cero casos iloc[1] != iloc[-1]. Cero con filas intermedias.
- Confirmacion estructural: la invariante `len == 2` la garantiza el
  propio clasificador (`_classify_strategy_from_types`), no el dataset.
- Dictamen: GO CONDICIONADO (caso B del arbol A/B/C).
- Fix propuesto: guarda explicita, sin cambio de logica.
- Evidencia: docs/auditoria/evidence/nipc_p70_probe/.
- Commit: af0dcb6.

### 2.13. P38 + P60 + P61 - contratos semanticos

- **P60 (identity contract):** `identity_type` obligatorio. Tipos TICKER, FIGI,
  CUSIP, ISIN. CUSIP no produce canonical. Enum de kinds sin cambios.
- **P61 (temporal validity contract):** separacion explicita entre identidad
  (`security_resolution_status`) y validez historica (`operational_mapping_status`
  nuevo). `TEMPORAL_UNVERIFIED` no degrada identidad.
- **P38 (coverage contract):** denominador pairwise = `TARGET_Q4 INTERSECT
  TARGET_Q1`. TARGET independiente del resolver. Correccion respecto de la
  implementacion vigente (que usaba `union_sec`).
- Decisiones del auditor: T1=A, T2=B (arquitectura), T3=si (P60 -> P61 -> P38).
- Documento: NIPC_CONTRATOS_SEMANTICOS_v1.md (561 lineas).
- Commit: b88e207.

### 2.14. Policy coverage v1.3 propuesta

- Documento derivado del contrato semantico. Sustituye funcionalmente a v1.2.
- Diferencia clave: denominador pairwise = TARGET_PAIRWISE (no union_sec).
- Preserva v1.0, v1.1, v1.2 intactas como historico.
- Estado: BORRADOR. No aplicable sin F2.4.
- Documento: NIPC_COVERAGE_POLICY_V13_PROPUESTA.md (355 lineas).
- Commit: 9a1cab5.

### 2.15. Preservacion del probe P70

- Script `_probe_p70.py` movido a `docs/auditoria/evidence/nipc_p70_probe/`.
- Hash SHA-256 preservado byte a byte.
- README de trazabilidad creado.
- Scripts one-shot eliminados de raiz.
- Commit: a9bb1fb.
---

## 3. Estado del sistema al cierre

| Metrica | Valor |
|---|---:|
| HEAD local | a9bb1fb |
| origin/main | 9d4a81e |
| Ahead | 80 |
| Working tree | limpio |
| Tests locales (modulo IAE) | 937 passed + 2 skipped |
| pyflakes | 0 warnings |
| compileall | OK |
| Policy v1.0 hash | 57f2d01f... (intacta) |
| Contrato v1 hash | 2d5084b9... |
| P70 dictamen hash | e8d4e4e8... |
| RADAR_TARGET_CATALOG | 242 filas, 240 OK, sha256 11eabce8... |

**Modulo IAE - estado funcional:**

    FA-1                          CERRADO / pushed
    FA-2                          CERRADO / pushed
    NIPC                          implementado (spec v1.4)
    Filer continuity              CERRADO como caracterizacion
    Q-CUR                         CERRADO (3 COM)
    Coverage baseline Fase A      CERRADO
    F2.3-bis                      PASS CONDICIONADO
    P70                           CERRADO
    P60 / P61 / P38               CERRADOS contractualmente
    Policy v1.3                   BORRADOR

**Modulo IAE - bloqueos vigentes:**

    THRESHOLD_1                   UNDEFINED
    THRESHOLD_2                   UNDEFINED
    Gate-NIPC.2                   BLOQUEADO
    Gate-NIPC.3                   NO AUTORIZADO
    OpenFIGI masivo               NO AUTORIZADO
    Policy v1.3 aplicacion        NO AUTORIZADA
    Codigo productivo P60/P61/P38 SIN CAMBIOS

---

## 4. Bloqueos resueltos en este ciclo

| ID | Descripcion | Estado |
|---|---|---|
| P70 | Asimetria iloc[1]/iloc[-1] en classify_strategy | CERRADO / GO CONDICIONADO |
| P60 | Inferencia de prefijo en canonical_security | CERRADO contractualmente |
| P61 | etf_holdings sin vigencia temporal | CERRADO contractualmente |
| P38 | Denominador de paired_weighted_share_coverage | CERRADO contractualmente |

---

## 5. Documentos entregados

### 5.1. Documentos nuevos (5)

| Documento | Lineas | Hash SHA-256 |
|---|---:|---|
| INSTITUTIONAL_ACCUMULATION_REVISION_ESTRUCTURAL_2026-09-20.md | 1180 | 318f02e82ef179e28f9938c32491e1a827942bdb2dd53d5aa0d4aad8032b137a |
| NIPC_P70_DICTAMEN.md | 301 | e8d4e4e825fe6136b23573705b800108247eb07c9a46e2f62c2c06980f179092 |
| NIPC_CONTRATOS_SEMANTICOS_v1.md | 561 | 2d5084b93af5c8e29d3cd2471c36e4ef3d52413ebe2733493ede2a4227cb6875 |
| NIPC_COVERAGE_POLICY_V13_PROPUESTA.md | 355 | e8b616177528b87cbd2ef11327e1a0aa3d0929a257ef0ae836cfa0fd8c8d1c1d |
| docs/auditoria/evidence/nipc_p70_probe/README.md | 76 | 1370017c34300991c70f7ab33362542826ed02d18b12c62c04327ec81ffb190d |

### 5.2. Evidencia nueva

| Fichero | Bytes | Hash SHA-256 |
|---|---:|---|
| evidence/nipc_p70_probe/_probe_p70.py | 6602 | d1742a769f24b7b07e28d3493ee1264e7ce82accca29fddd52a7011dbb82cd50 |
| evidence/nipc_p70_probe/probe_p70_result.json | 814 | 809c51af47ae154c6c01bba1cf3983a6c0143fd3b0711958fa8b9e6e18bae96f |
| evidence/nipc_p70_probe/probe_p70_summary.txt | 984 | 9e8c42d6a742d773be7e531782a7d8514a5c00aa64efc5080d81a718d6384fe5 |

### 5.3. Documentos preservados intactos

| Documento | Estado |
|---|---|
| NIPC_COVERAGE_POLICY.md (v1.0) | INTACTO (hash 57f2d01f...) |
| NIPC_COVERAGE_POLICY_V11_PROPUESTA.md | INTACTO |
| NIPC_COVERAGE_POLICY_V12_PROPUESTA.md | INTACTO |

### 5.4. Cadena de trazabilidad

    NIPC_CONTRATOS_SEMANTICOS_v1.md (2d5084b9...)
        |
        +-- cita hash -> NIPC_P70_DICTAMEN.md (e8d4e4e8...)
        |                    |
        |                    +-- cita por nombre -> _probe_p70.py (d1742a76...)
        |                                            |
        |                                            +-- produce -> probe_p70_result.json (809c51af...)
        |                                            +-- produce -> probe_p70_summary.txt (9e8c42d6...)
        |                                            |
        |                                            +-- documentado en -> README.md
        |
        +-- referenciado por -> NIPC_COVERAGE_POLICY_V13_PROPUESTA.md
---

## 6. Peticion formal de F2.4

### 6.0. Instrucciones de lectura y decisiones solicitadas

#### 6.0.1. Material objeto de revision

Para el dictamen F2.4 se solicita revisar, en este orden:

1. `NIPC_CONTRATOS_SEMANTICOS_v1.md` - contrato semantico habilitante.
2. `NIPC_COVERAGE_POLICY_V13_PROPUESTA.md` - propuesta normativa derivada.
3. `NIPC_P70_DICTAMEN.md` - cierre documentado de P70.
4. `docs/auditoria/evidence/nipc_p70_probe/` - evidencia reproducible del probe P70.
5. `INSTITUTIONAL_ACCUMULATION_REVISION_ESTRUCTURAL_2026-09-20.md` - informe interno que origino los cuatro bloqueos.

Los cinco elementos estan materializados y preservados en el commit `a9bb1fb`. Los documentos historicos referenciados en esta entrega permanecen intactos.

#### 6.0.2. Decisiones solicitadas al auditor

El dictamen F2.4 debe resolver cuatro puntos:

**D1 - Contrato semantico v1.**
Confirmar la adopcion contractual de P60, P61 y P38, incluyendo:

- T1: P60 no modifica `canonical_security_kind`.
- T2: `operational_mapping_status` es independiente de `security_resolution_status`.
- T3: el orden contractual de aplicacion es P60 -> P61 -> P38.

**D2 - Policy v1.3.**
Confirmar si `NIPC_COVERAGE_POLICY_V13_PROPUESTA.md` puede constituir la nueva base normativa, sustituyendo funcionalmente a v1.2 cuando sea aplicada, manteniendo v1.0 vigente hasta ese momento.

Esta decision implica aceptar la definicion de `paired_weighted_share_coverage` con denominador `TARGET_Q4 INTERSECT TARGET_Q1` y, por tanto, considerar el baseline y el TOP 2000 historicos como no aptos para fijar thresholds bajo v1.3 hasta su recalculo.

**D3 - P70.**
Confirmar el cierre de P70 como `GO CONDICIONADO`, con la guarda explicita propuesta y sin modificacion de la logica de negocio.

**D4 - Continuacion tecnica.**
Autorizar exclusivamente el paso 4 de la secuencia posterior a F2.4: creacion de tests especificos para P38, P60, P61 y P70. Esta autorizacion no incluye todavia la aplicacion de fixes, el recalculo de coverage, la ampliacion masiva de OpenFIGI ni la fijacion de thresholds.

#### 6.0.3. Forma esperada del dictamen

El auditor puede emitir una de las siguientes resoluciones:

- **F2.4 GO** - los cuatro puntos quedan aprobados y se autoriza el paso 4.
- **F2.4 GO CONDICIONADO** - uno o varios puntos quedan aprobados sujetos a condiciones explicitas; la autorizacion del paso 4 queda limitada por dichas condiciones.
- **F2.4 NO GO** - uno o varios puntos no quedan aprobados y el paquete debe volver a revision antes de continuar.

En ningun caso esta solicitud pretende fijar `THRESHOLD_1` o `THRESHOLD_2`, aprobar cambios de codigo productivo, autorizar OpenFIGI masivo, aplicar la policy v1.3 ni modificar los artefactos historicos preservados.

### 6.1. Dictaminar sobre los contratos semanticos v1

Pregunta:

> ?Aprueba el auditor el contrato `NIPC_CONTRATOS_SEMANTICOS_v1.md` como
> base contractual para la implementacion de P38, P60, P61?

Sub-preguntas:

1. **T1** - ?Confirma el auditor que P60 no modifica `canonical_security_kind`?
2. **T2** - ?Confirma el auditor que `operational_mapping_status` debe ser un
   campo independiente de `security_resolution_status`?
3. **T3** - ?Confirma el auditor el orden P60 -> P61 -> P38?

### 6.2. Dictaminar sobre la propuesta de policy v1.3

Pregunta:

> ?Aprueba el auditor `NIPC_COVERAGE_POLICY_V13_PROPUESTA.md` como borrador
> normativo que sustituye funcionalmente a v1.2, manteniendo v1.0 vigente
> hasta su aplicacion?

### 6.3. Dictaminar sobre P70

Pregunta:

> ?Confirma el auditor el cierre de P70 como GO CONDICIONADO, con el fix
> propuesto (guarda explicita sin cambio de logica)?

### 6.4. Autorizar la continuacion

Si F2.4 es GO, la secuencia continuaria:

    Paso 4: tests especificos (P38, P60, P61, P70)
    Paso 5: fixes quirurgicos
    Paso 6: recalculo de coverage
    Paso 7: OpenFIGI ampliado (si autorizado)
    Paso 8: threshold proposal
    Paso 9: Gate-NIPC.2

**Solicitud explicita:** autorizar el paso 4 (tests especificos) tras F2.4 GO.

### 6.5. Confirmacion de bloqueos vigentes

    THRESHOLD_1                   UNDEFINED
    THRESHOLD_2                   UNDEFINED
    Gate-NIPC.2                   BLOQUEADO
    Gate-NIPC.3                   NO AUTORIZADO
    Policy v1.3 aplicacion        NO AUTORIZADA
    Codigo productivo P60/P61/P38 SIN CAMBIOS
    OpenFIGI masivo               NO AUTORIZADO

El presente ciclo NO fija thresholds, NO aplica policy, NO toca codigo
productivo. Solo entrega los contratos habilitantes y solicita su
validacion externa.

---

## 7. Anexos

### 7.1. Referencias cruzadas

| Documento | Rol |
|---|---|
| NIPC_CONTRATOS_SEMANTICOS_v1.md | Contrato habilitante de este ciclo |
| NIPC_P70_DICTAMEN.md | Cierre de P70 |
| NIPC_COVERAGE_POLICY_V13_PROPUESTA.md | Propuesta normativa derivada |
| INSTITUTIONAL_ACCUMULATION_REVISION_ESTRUCTURAL_2026-09-20.md | Informe origen de los 4 bloqueos |
| INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_BASELINE_INFORME.md | Baseline Fase A |
| INSTITUTIONAL_ACCUMULATION_OPENFIGI_TOP2000_INFORME.md | TOP 2000 |
| INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_BASELINE_DICTAMEN.md | Dictamen baseline |
| INSTITUTIONAL_ACCUMULATION_NIPC_F21BIS_FUENTES_IDENTIDAD.md | F2.1-bis |
| INSTITUTIONAL_ACCUMULATION_OPENFIGI_DICTAMEN_F23BIS.md | Dictamen F2.3-bis |
| INSTITUTIONAL_ACCUMULATION_TOP2000_AUTORIZACION.md | Autorizacion TOP 2000 |

### 7.2. Commits del ciclo

    f107b27  docs(iae): revision estructural del modulo IAE - 95 hallazgos
    af0dcb6  docs(iae): dictamen P70 - cierre classify_strategy
    b88e207  docs(iae): contratos semanticos v1 - P38/P60/P61
    9a1cab5  docs(iae): policy coverage v1.3 propuesta
    a9bb1fb  docs(iae): preservar probe P70 en evidence + README de trazabilidad

Los 5 commits son locales. No push (regla local-first IAE).

### 7.3. Estado de Gate-NIPC

    Gate-NIPC.1                          CERRADO (filer continuity)
    Gate-NIPC.2                          BLOQUEADO (THRESHOLD_1/2 UNDEFINED)
    Gate-NIPC.3                          NO AUTORIZADO
    F2.4                                 (esta entrega la solicita)

### 7.4. Prohibiciones confirmadas

Se confirma que este ciclo NO ha realizado:

- Modificacion de NIPC_COVERAGE_POLICY.md v1.0, v1.1, v1.2.
- Modificacion de codigo productivo.
- Fijacion de THRESHOLD_1 ni THRESHOLD_2.
- Ejecucion de OpenFIGI masivo.
- Aplicacion de NIPC_COVERAGE_POLICY_V13_PROPUESTA.md.
- Reescritura del baseline historico.
- Tocar compute_delta_shares, match_key, C2.

### 7.5. Documentos transfer

Los documentos de transferencia de sesion (capa complementaria, no normativa)
no forman parte de esta entrega. Si hay conflicto, gana NIPC_CONTRATOS_SEMANTICOS_v1.md.

---

## 8. Cierre

**El ciclo post-F2.3-bis queda cerrado a nivel documental.** La cadena completa
esta materializada en `docs/auditoria/` con hashes registrados. Los 4 bloqueos
semanticos estan congelados contractualmente.

**Solicitud:** dictamen F2.4 sobre los contratos semanticos v1, la propuesta
de policy v1.3, el cierre de P70, y autorizacion del paso 4 (tests especificos).

**Prohibicion de continuar sin F2.4 GO:** no se inician tests, fixes, recalculo
de coverage, ni ampliacion de OpenFIGI sin dictamen externo.

---

Fin del informe de entrega F2.4.
Version 1.0 (2026-09-20). HEAD a9bb1fb.