# TRANSFER - Guia de onboarding

**NO es fuente de estado.**
**Estado vivo:** `iae/ESTADO_SISTEMA.md` (hechos) + `iae/ESTADO_DECLARADO.md` (fases).

**Referencia unica del modulo IAE:** `iae/IAE_MAESTRO.md`.

---

## Como usar este documento

1. Pegar este documento + `PROMPT_MAESTRO.md` al asistente.
2. Esperar confirmacion de asimilacion antes de tocar nada.
3. Consultar `iae/IAE_MAESTRO.md` para estado, arquitectura,
   evidencia y consulta abierta del modulo IAE.

---

## Rol

Eres el Ingeniero Supervisor del Radar de Rotacion Sectorial (IAE).
Reglas de personalidad y metodo: `PROMPT_MAESTRO.md` secciones 1 y 3.

---

## Estado al cierre de la consolidacion documental (2026-09-22)

```
HEAD                  384a27f (verificar con git al arrancar)
Ahead                 ~337 commits locales
Working tree          LIMPIO (salvo untracked de outputs/)
Tests                 test_h731: 11 passed
                      suite global: 1421 passed + 2 skipped + 3 failed
                      (los 3 failed son test_freshness, ambientales:
                       parquets stale en local; en CI se skipean)
Push                  NO (local-first IAE, dictamenes #76/#77/#78)
Cobertura IAE         110/110 funciones con llamada real en tests
                      (cobertura de lineas: 83%)
```

---

## Documentos clave (vivos)

```
docs/auditoria/PROMPT_MAESTRO.md              v7.1 - normativa general
docs/auditoria/TRANSFER.md                    este documento (onboarding)
docs/auditoria/FOLLOWUPS.md                   cronologia de ciclos
docs/auditoria/README.md                      navegacion

docs/auditoria/iae/IAE_MAESTRO.md             REFERENCIA UNICA DEL MODULO IAE
docs/auditoria/iae/ESTADO_DECLARADO.md        fases, prohibiciones, hallazgos
docs/auditoria/iae/ESTADO_SISTEMA.md          hechos (autogenerado)
docs/auditoria/iae/DICTAMENES.md              indice #1-#78
docs/auditoria/iae/HISTORICO_IAE.md           registro de ciclos
docs/auditoria/iae/INFORME.md                 informes consolidados
docs/auditoria/iae/NIPC_CONTRATOS_SEMANTICOS_v1.md  contrato P60-P70
docs/auditoria/iae/NIPC_COVERAGE_POLICY.md          policy normativa
docs/auditoria/iae/INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md
docs/auditoria/iae/evidence/                  probes y evidencia empirica
```

---

## Estado del modulo IAE en una frase

Codigo implementado y aislado del pipeline productivo. Evidencia
empirica supera el test acido (TARGET_PAIRWISE = 239 sobre 13F
real). Certificacion A.6.6 bloqueada por cuestion semantica sobre
vigencia PIT del catalogo, no por defecto de implementacion.

Detalle completo en `iae/IAE_MAESTRO.md`.

---

## Prohibiciones vigentes (resumen)

- NO tocar nipc.py, delta_shares.py, security_identity.py,
  temporal_validity.py, relationships.py, target_universe.py.
- NO push a origin/main.
- NO OpenFIGI masivo.
- NO retro-fechar snapshots.
- NO fabricar overlap con exceptions ad-hoc.
- NO modificar contratos normativos sin dictamen.
- NO fijar THRESHOLD_1/2 sin propuesta razonada.

Lista completa en `iae/IAE_MAESTRO.md` seccion 9.

---

## Comandos de arranque

```
Set-Location D:\Macro_Sectorial
git log --oneline -5
git status -sb
py scripts\generate_estado_sistema.py
py -m pytest tests/ validation/ -q --tb=line
py -m pyflakes . ; py -m compileall . -q
```

Esperado:

- HEAD 384a27f o posterior
- ahead 337 o mas
- working tree limpio
- 1421 passed + 2 skipped + 3 failed (test_freshness, preexistentes)
- test_h731: 11 passed
- pyflakes silencio, compileall OK

---

## Confirmacion esperada

"Confirmado, contexto asimilado."

Estado que reconozco:

- HEAD 384a27f o posterior
- Dictamenes #76 (A2=GO, A.6.6=NO-GO) + #77 (B-02/B-03/B-04) +
  #78 (B-04 code-level cerrado, A.6.6=NO-GO)
- IAE implementado y aislado de produccion
- Test acido superado: TARGET_PAIRWISE = 239
- Consulta abierta al auditor (IAE_MAESTRO.md seccion 11)
- Prohibiciones vigentes respetadas

Pregunta: "Que hacemos?"

---

FIN DE TRANSFER. 2026-09-22.