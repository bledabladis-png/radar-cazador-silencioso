# TRANSFER - Guia de onboarding

**NO es fuente de estado.**
**Estado vivo:** `iae/ESTADO_SISTEMA.md` + `iae/ESTADO_DECLARADO.md`.
**Referencia unica del modulo IAE:** `iae/IAE_MAESTRO.md`.

---

## Rol

Ingeniero Supervisor del Radar de Rotacion Sectorial (IAE).
Reglas de personalidad y metodo: `PROMPT_MAESTRO.md` secciones 1 y 3.

---

## Estado al cierre de la consolidacion (2026-09-22)

    HEAD                 ce6a390 (verificar con git al arrancar)
    Ahead                348 commits locales
    Working tree         LIMPIO
    Tests IAE            641 passed
    Suite global         1421 passed + 2 skipped + 3 failed
                         (los 3 failed son test_freshness, ambientales)
    Push                 NO (local-first IAE)
    Cobertura IAE        110/110 funciones con llamada real (83% lineas)

---

## Documentos clave

    docs/auditoria/PROMPT_MAESTRO.md              v7.2
    docs/auditoria/TRANSFER.md                    este documento
    docs/auditoria/README.md                      navegacion

    docs/auditoria/iae/IAE_MAESTRO.md             referencia unica del modulo IAE
    docs/auditoria/iae/ESTADO_DECLARADO.md        estado de fases y deuda
    docs/auditoria/iae/ESTADO_SISTEMA.md          hechos autogenerados
    docs/auditoria/iae/evidence/                  evidencia empirica

Documentos historicos (no normativos): NIPC_CONTRATOS_SEMANTICOS_v1.md,
NIPC_COVERAGE_POLICY.md, INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md.

---

## Estado del modulo IAE en una frase

Implementado, testeado y verificado end-to-end sobre datos reales.
Con el crosswalk CUSIP extendido a 246 entradas:

- Q4 2025: 239 tickers operativos
- Q1 2026: 243 tickers operativos
- Delta radar: 553.321 filas (n_unresolved_identity = 0)
- NIPC radar: -4.316.734.936

Aislado del pipeline productivo. Entrega los datos necesarios para
el analisis de acumulacion institucional.

Detalle completo en `iae/IAE_MAESTRO.md`.

---

## Reglas de operacion

- No push a `origin/main` (local-first IAE).
- No OpenFIGI masivo.
- No modificar codigo sin ciclo previo.
- No integracion a produccion sin validacion funcional.

---

## Comandos de arranque

    Set-Location D:\Macro_Sectorial
    git log --oneline -5
    git status -sb
    py scripts\generate_estado_sistema.py
    py -m pytest tests/validation/ -q --tb=line
    py -m pyflakes . ; py -m compileall . -q

Esperado:

- HEAD ce6a390 o posterior
- ahead 348 o mas
- working tree limpio
- 1421 passed + 2 skipped + 3 failed (test_freshness, preexistentes)
- pyflakes silencio, compileall OK

---

## Confirmacion esperada

"Confirmado, contexto asimilado."

Estado que reconozco:

- HEAD 8b9b80f o posterior
- IAE implementado, testeado, aislado
- Crosswalk extendido a 246 CUSIPs
- Verificado end-to-end sobre 13F real Q4/Q1 (NIPC radar -4.32B)
- Prohibiciones respetadas
- Siguiente: auditoria externa de formulas (Fase 2)

Pregunta: "Que hacemos?"

---

FIN DE TRANSFER. 2026-09-22.