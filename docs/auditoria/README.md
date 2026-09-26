# README - docs/auditoria/

Navegacion de la documentacion de auditoria.

## Estructura

    PROMPT_MAESTRO.md    norma general (rol, metodologia, arquitectura)
    TRANSFER.md          guia de onboarding
    README.md            este documento
    iae/                 modulo IAE

## iae/

    IAE_MAESTRO.md                                  referencia unica del modulo IAE
    ESTADO_DECLARADO.md                             estado de fases y deuda
    ESTADO_SISTEMA.md                               hechos autogenerados
    NIPC_CONTRATOS_SEMANTICOS_v1.md                 contrato historico de diseno
    NIPC_COVERAGE_POLICY.md                         policy historica
    INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md  especificacion
    evidence/                                       evidencia empirica de probes

## Lectura recomendada

1. PROMPT_MAESTRO.md (completo)
2. TRANSFER.md (onboarding)
3. iae/IAE_MAESTRO.md (modulo IAE completo)
4. iae/ESTADO_SISTEMA.md (hechos)

## Reglas de operacion

- Push a origin/main autorizado tras validacion funcional completa
  (compileall + pyflakes + suite verde) y verificacion en CI cuando
  proceda. Regla vigente en PROMPT_MAESTRO.md seccion 2.
- No OpenFIGI masivo. Solo consultas dirigidas.
- No modificar codigo sin ciclo previo (reconocimiento -> diseno ->
  implementacion -> verificacion -> documentacion).
- No integracion a produccion sin validacion funcional.

Detalle en PROMPT_MAESTRO.md secciones 2 y 3.