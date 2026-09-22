# README - docs/auditoria/

Navegacion de la documentacion de auditoria.

## Estructura

    PROMPT_MAESTRO.md    norma vigente (rol, metodologia, arquitectura)
    TRANSFER.md          guia de onboarding (no es fuente de estado)
    README.md            este documento
    iae/                 modulo IAE (IAE_MAESTRO.md + evidencia)

## iae/

    IAE_MAESTRO.md                                  referencia unica del modulo IAE
    ESTADO_DECLARADO.md                             fases y prohibiciones
    ESTADO_SISTEMA.md                               hechos autogenerados
    NIPC_CONTRATOS_SEMANTICOS_v1.md                 contrato P60-P70 vigente
    NIPC_COVERAGE_POLICY.md                         policy normativa
    INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md  especificacion v1.4
    evidence/                                       evidencia empirica de probes

## Lectura recomendada

1. PROMPT_MAESTRO.md (completo)
2. TRANSFER.md (onboarding)
3. iae/IAE_MAESTRO.md (modulo IAE completo)
4. iae/ESTADO_DECLARADO.md (prohibiciones vigentes)
5. iae/ESTADO_SISTEMA.md (hechos)

## Prohibiciones vigentes (resumen)

- NO push a origin/main (local-first IAE).
- NO modificar contratos normativos sin dictamen.
- NO modificar codigo del IAE sin dictamen (nipc.py, delta_shares.py,
  security_identity.py, temporal_validity.py, relationships.py,
  target_universe.py).
- NO OpenFIGI masivo.
- NO retro-fechar snapshots.

Lista completa en iae/IAE_MAESTRO.md seccion 9.