# Indice de docs/auditoria

Documento de navegacion. Ultima actualizacion: 2026-09-20.

## Estructura

    docs/auditoria/
      PROMPT_MAESTRO.md     Norma vigente del sistema
      FOLLOWUPS.md          Cronologia de decisiones (activo)
      README.md             Este documento
      iae/                  Modulo Institutional Accumulation Evidence
        evidence/           Probes y evidencia empirica (6 subdirs)
      radar/                Contratos temporales + informes del radar
      auditorias/           Auditorias estructurales y revisiones
      archive/              Archivado futuro (vacio)

## Regla de versionado

1 concepto = 1 fichero vivo. Al evolucionar, se edita in-place. Las
versiones previas se archivan con fecha o se borran si su contenido
esta embebido en la version vigente (git conserva el historico).

No se admiten ficheros `_v1.md`, `_V12_PROPUESTA.md`, `_DICTAMEN_C.md`.

**Excepcion documentada:** contratos cuyo sha256 ya esta en la cadena
autoritativa (NIPC_CONTRATOS_SEMANTICOS_v1.md, NIPC_COVERAGE_POLICY.md
v1.0) NO se modifican in-place. Se emite version nueva con seccion
"Deriva de vN" y se preserva la anterior.
Los dictamenes y informes historicos se consolidan en un unico fichero
por tema, con indice y resumen.

## Leer primero

Orden recomendado para entender el sistema actual:

    1. PROMPT_MAESTRO.md                            norma completa
    2. FOLLOWUPS.md                                 cronologia
    3. iae/PROPUESTA.md                             diseno fundacional IAE
    4. iae/CONTRATO.md                              contrato IAE
    5. iae/NIPC_ESPECIFICACION.md                   spec NIPC vigente
    6. iae/NIPC_CONTRATOS_SEMANTICOS.md             contrato P38/P60/P61
    7. iae/NIPC_COVERAGE_POLICY.md                  policy vigente
    8. iae/INFORME.md                               informe consolidado IAE
    9. iae/DICTAMENES.md                            registro de dictamenes
    10. radar/CONTRATO_TEMPORAL.md                  contrato temporal radar

## Estructura detallada

### iae/

    PROPUESTA.md                          diseno fundacional
    CONTRATO.md                           contrato IAE v1.1
    CONTRATO_ADDENDUM.md                  D1-D5
    NIPC_ESPECIFICACION.md                spec NIPC v1.4 (vive)
    NIPC_CONTRATOS_SEMANTICOS_v1.md       contrato P38/P60/P61
    NIPC_COVERAGE_POLICY.md               policy v1.0 vigente
    NIPC_COVERAGE_POLICY_V13_PROPUESTA.md policy propuesta
    INFORME.md                            consolidado de 17 informes
    DICTAMENES.md                         registro de 23 dictamenes
    RECONCILIACION_CONTRATO_CODIGO.md     divergencias contrato<->codigo
    REESTRUCTURACION_MODULO.md            plan arquitectonico
    FASE_A6_PLAN.md                       plan de ejecucion A.6

### tests/ (contractuales para F2.4)

    test_p60_contract.py     4 tests (3 pass + 1 xfail)
    test_p61_contract.py     5 tests (3 pass + 2 xfail)
    test_p38_contract.py     5 tests (2 pass + 3 xfail)

### radar/

    DEUDA.md                              deuda activa del radar (vivo)
    FU-021-5_ESPECIFICACION_CONTRATO_TEMPORAL.md
    FU-021-5_ESPECIFICACION_CONTRATO_TEMPORAL_PARTE_B.md
    FU-021-5_PLAN_IMPLEMENTACION.md
    FU-021-3B_INFORME_INDEX_EOD_RATE_YIELD.md
    FU-021-3B_ANEXO_VOLATILITY_INDEX.md
    FU-021-3C_INFORME_FUTURE_FX.md
    H1_INFORME_MUTABILIDAD_HISTORICA.md
    E5_INFORME_VIX3M.md

### auditorias/

    AUDITORIA_ESTRUCTURAL_2026-09-20.md
    INSTITUTIONAL_ACCUMULATION_REVISION_ESTRUCTURAL_2026-09-20.md
    NIPC_DICTAMEN_REVISION_ESTRUCTURAL_2026-09-20.md
    INSTITUTIONAL_ACCUMULATION_TOP2000_AUTORIZACION.md

### iae/evidence/

    nipc_gate0_baseline/                  probe coverage baseline
    nipc_gate0_openfigi/                  probe OpenFIGI
    nipc_gate0_probe/                     probes filer continuity
    nipc_gate0_target_identity/           probe radar identity
    nipc_gate0_target_identity_top2000/   probe TOP 2000
    nipc_p70_probe/                       probe P70

## Documentos por rol

AUDITOR EXTERNO:

    1. auditorias/NIPC_INFORME_TECNICO_AUDITORIA_EXTERNA.md
    2. iae/NIPC_CONTRATOS_SEMANTICOS_v1.md
    3. iae/NIPC_COVERAGE_POLICY.md
    4. iae/NIPC_COVERAGE_POLICY_V13_PROPUESTA.md
    5. auditorias/AUDITORIA_ESTRUCTURAL_2026-09-20.md

SUPERVISOR (retomar proyecto):

    1. PROMPT_MAESTRO.md (completo)
    2. FOLLOWUPS.md (ultimas 200 lineas)
    3. Este README.md
    4. iae/INFORME.md
    5. iae/DICTAMENES.md

## Reglas de uso

    - NO modificar PROMPT_MAESTRO.md sin bump de version.
    - NO modificar dictamenes cerrados (registro en iae/DICTAMENES.md).
    - NO modificar iae/NIPC_COVERAGE_POLICY.md v1.0.
    - NO reintroducir ficheros con sufijos _v1.md, _V12_PROPUESTA.md.
    - Git conserva el historico de lo archivado.

---

Fin del README. Mantener actualizado en cada limpieza.
