# TRANSFER DE SESION - 2026-09-21 v9

Documento de onboarding. **NO es fuente de estado.**
Estado vivo: `iae/ESTADO_SISTEMA.md` (hechos) + `iae/ESTADO_DECLARADO.md` (fases).

**Como usarlo:**
1. Pegar este documento + `PROMPT_MAESTRO.md`.
2. Esperar confirmacion de asimilacion antes de tocar nada.

---

## 1. Rol

Eres el Ingeniero Supervisor del Radar de Rotacion Sectorial (IAE).
Reglas de personalidad y metodo: `PROMPT_MAESTRO.md` secciones 1 y 3.

## 2. Estado real al cierre de esta sesion

    HEAD          5877e9c (verificar con git al arrancar)
    Ahead         282 commits locales
    Working tree  LIMPIO (verificar)
    Tests         7 passed + 1 xfailed en test_h731;
                  suite global: 1301 passed + 2 skipped + 3 failed
                  (los 3 failed son test_freshness.py, preexistentes, no-regresion)
    Push          NO (local-first IAE)

## 3. Contexto: auditoria externa reciente

El auditor externo ha emitido, el 2026-09-21:

  **Dictamen bundle v3 (minimo):** NO-GO para A.6.6 / F2.4-CLOSE.
    Motivo: P38 no validado cuantitativamente, H-10.1 abierto,
    A.6.4 reclasificado como smoke test, THRESHOLD_2 bloqueado.

  **Dictamen consulta H-10.1:** GO para fix A2.
    Autoriza ciclo de correccion completo con propagacion SSHPRNAMT real.

**5 respuestas del auditor (fijas, no renegociables):**

    Q1  Alcance del fix: A2 (no A1). Incluye SSHPRNAMT real.
    Q2  coverage_previous/current = RESOLVED / TARGET (no observed).
    Q3  operational_mapping_status debe derivarse del estado operacional.
        El adapter PROPAGA, no fabrica VERIFIED.
    Q4  SSHPRNAMT del canonical_snapshot post-amendments, agregado por
        shareClassFIGI, luego max(Q4,Q1).
    Q5  Q4 vacio:
          coverage_previous              = UNAVAILABLE (TARGET_Q4=0)
          coverage_current               = calculable sobre TARGET_Q1
          paired_security_coverage       = UNAVAILABLE (TARGET_PAIRWISE=0)
          paired_weighted_share_coverage = UNAVAILABLE

## 4. TRABAJO PENDIENTE: fix A2 (H-05/H-06/H-07/H-10.1)

**Estado: empezando, 0 de 5 commits hechos.**

Plan autorizado, en orden estricto:

    Commit 1  period_state.py
              Anadir a PeriodState: sshprnamt (float|None) y
              operational_mapping_status (str, default UNRESOLVED).
              NO romper build_period_state.

    Commit 2  target_builder.py + probe
              Extraer SSHPRNAMT efectivo del canonical snapshot por key.
              Transportarlo hasta el state.

    Commit 3  catalog_p38_adapter.py
              Quitar operational_mapping_status="VERIFIED" hardcoded
              (L164) y weight=1.0 hardcoded (L165). PROPAGAR del state:
                operational_mapping_status = state[k].operational_mapping_status
                weight = float(state[k].sshprnamt) if not None else 0.0
              (cierra H-10.1 + H-07 en el mismo bloque _records).
              Adicionalmente: reescribir tests/test_h731_adapter_p38_compat.py
              (xfail A-12 -> positivo con 3 direcciones a/b/c; compat_a/b/c/d
              actualizados a propagacion; nuevo test weight). Motivo: regla
              3.1 PROMPT_MAESTRO (un commit = un veredicto verde) prima
              sobre el plan original que diferia el xfail a Commit 5.

    Commit 4  coverage.py
              coverage_previous/current con denominador TARGET_Q4/TARGET_Q1.
              UNAVAILABLE si TARGET del periodo = 0.

    Commit 5  probe + tests
              probe_integration_b1_p61_p38.py: Q4 y Q1 REALES, sin mock.
              Tests nuevos: max(Q4,Q1), agregacion FIGI.

**Reglas por commit:** suite completa + pyflakes + compileall antes del
siguiente. Si algo falla, parar y avisar.

## 5. Ficheros que el fix TOCA (autorizados)

    src/institutional_accumulation/identity/period_state.py
    src/institutional_accumulation/identity/target_builder.py
    src/institutional_accumulation/aggregation/catalog_p38_adapter.py
    src/institutional_accumulation/aggregation/coverage.py
    docs/auditoria/iae/evidence/a64_integration_b1_p61_p38/probe_integration_b1_p61_p38.py
    tests/test_p38_contract.py
    tests/test_h731_adapter_p38_compat.py

## 6. Ficheros PROHIBIDOS (no tocar sin dictamen)

    src/institutional_accumulation/aggregation/nipc.py
    src/institutional_accumulation/aggregation/delta_shares.py
    src/institutional_accumulation/sec_13f/identity/security_identity.py
    src/institutional_accumulation/temporal_validity.py
    src/institutional_accumulation/sec_13f/identity/relationships.py
    MATCH_KEY, C2
    Contratos normativos
    NIPC_COVERAGE_POLICY.md

## 7. Correcciones documentales aplicadas hoy (A-01/A-04/A-12)

    A-01  ESTADO_SISTEMA.md lista los 3 tests fallidos por nombre.
    A-04  Separadas metricas diagnosticas (18/242 = 0.0744) de las
          contractuales (RESOLVED/TARGET) en EXPEDIENTE_A64_FIX.md.
    A-12  test_h101_adapter_no_marca_verified_sin_evidencia_operacional
          marcado xfail(strict=True). Documenta el DEFECTO como
          expectativa futura, no como conformidad actual.

## 8. Documentos clave (vivos)

    docs/auditoria/PROMPT_MAESTRO.md         v7 - normativa general
    docs/auditoria/iae/ESTADO_DECLARADO.md   fases, prohibiciones, hallazgos
    docs/auditoria/iae/ESTADO_SISTEMA.md     hechos (autogenerado)
    docs/auditoria/iae/EXPEDIENTE_A64_FIX.md el expediente del fix
    docs/auditoria/iae/NIPC_CONTRATOS_SEMANTICOS_v1.md   contrato P38
    docs/auditoria/iae/DICTAMENES.md         indice #1-#75
    docs/auditoria/iae/HISTORICO_IAE.md      registro de ciclos
    docs/auditoria/iae/evidence/a64_integration_b1_p61_p38/README.md
                                             evidencia A.6.4 (smoke test)

## 9. Comandos de arranque

    Set-Location D:\Macro_Sectorial
    git log --oneline -5
    git status -sb
    py scripts\generate_estado_sistema.py     # regenera ESTADO_SISTEMA.md
    py -m pytest tests/ validation/ -q --tb=line
    py -m pyflakes . ; py -m compileall . -q

Esperado:
  - HEAD 5877e9c o posterior
  - ahead 282 o mas
  - working tree limpio
  - 1301 passed + 2 skipped + 3 failed (test_freshness, preexistentes)
  - test_h731: 7 passed + 1 xfailed
  - pyflakes silencio, compileall OK

## 10. Confirmacion esperada

    "Confirmado, contexto asimilado."

    Estado que reconozco:
      - HEAD 5877e9c, ahead 282
      - Auditor: NO-GO A.6.6 + GO fix A2
      - Fix A2: 0/5 commits, empezando por period_state.py
      - H-10.1 abierto; P38 GO CONDICIONADO; A.6.6 NO-GO
      - Prohibido tocar nipc/delta_shares/security_identity/temporal_validity
      - Push: NO

    Pregunta: "Que hacemos?"

---

FIN DEL TRANSFER v9. HEAD 5877e9c. Fecha: 2026-09-21.
