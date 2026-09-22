# TRANSFER DE SESION - 2026-09-22 v10.1

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

    HEAD          post-B-04 (verificar con git al arrancar)
    Ahead         333 commits locales
    Working tree  LIMPIO (verificar)
    Tests         test_h731: 11 passed;
                  suite global: 1421 passed + 2 skipped + 3 failed
                  (los 3 failed son test_freshness.py, ambientales:
                   parquets stale en local; en CI se skipean via
                   @pytest.mark.skipif -> suite CI = 0 failed)
    Push          NO (local-first IAE, dictamenes #76/#77)
    Cobertura     IAE 110/110 funciones publicas con test semantico
                  + 6 tests fixture pairwise (B-04)

## 3. Contexto: auditoria externa reciente

### 3.1. Cadena de dictamenes

    #76 (2026-09-21)  A2 = GO; A.6.6 = NO-GO. Bloqueos residuales:
                      B-02 (P62/PIT), B-03 (TARGET), B-04 (pairwise).
    #77 (2026-09-22)  Respuesta a la consulta A.6.7:
                      - B-02: GO CONDICIONADO. Prohibido retro-fechado.
                        Probe debe declarar PIT_UNAVAILABLE si no hay
                        snapshot que cubra el periodo. A.6.6 NO.
                      - B-03: GO PARCIAL. 2 re-queries OpenFIGI
                        (BRK-B, MOG-A). NO OpenFIGI masivo. NO cerrar
                        con 240/242 sin definir TARGET_P.
                      - B-04: NO-GO a fabricar overlap con exceptions.
                        GO a fixture sintetico NON-PRODUCTION o a
                        mappings Q4 con vigencia historica real.
                      - Punto contractual clave: CATALOG_UNIVERSE (242)
                        NO es TARGET_P (universo por periodo). TARGET
                        no se filtra por RESOLVED.
                      - Hallazgos nuevos: H-19 (HEAD ambiguo), H-20
                        (CI sin SHA/run verificable).
                      - Push: NO.

### 3.2. Fix A2 (H-05/H-06/H-07/H-10.1) - CERRADO

5 commits (2f98926..473ce06) ejecutados 2026-09-21. Estado:

    period_state       transporta sshprnamt + operational_mapping_status
    target_builder     extract_sshprnamt_by_figi agregado por FIGI
    adapter P38        propaga VERIFIED + weight desde el state
    coverage           denominador TARGET + Q5 fail-closed
    probe A.6.4        sin mock Q4=Q1; Q5 literal; cardinalidades

### 3.3. Saneamiento post-#76 - CERRADO

    B-01  Contradiccion documental post-fix (docs sincronizados)
    B-05  Precision: coverage_current sobre TARGET materializado
    B-06  Cadena end-to-end sin doble conteo (test + probe)
    B-07  Suite CI = 0 failed (verificado; skip en clone fresco)
    H-08  Test ortogonal identity=RESOLVED + weight=NOT_PRESENT

### 3.6. Ciclo B-04 (fixture pairwise) - CERRADO

Commit (proximo). Fixture NON-PRODUCTION ejercita rama pairwise:

    TARGET_Q4 = {A, B, D}; TARGET_Q1 = {A, C, D}
    TARGET_PAIRWISE = {A, D}; PAIRED = {A}
    paired_security_coverage         = 0.5
    paired_weighted_share_coverage   = 6/7
    coverage_status                  = VALID

Ficheros: `tests/test_p38_pairwise_fixture.py`,
`evidence/a64_integration_b1_p61_p38/probe_pairwise_fixture.py`,
`result_pairwise_fixture.json`. NO toca mappings productivos.

### 3.4. Ciclo B-02 (PIT_UNAVAILABLE) - CERRADO

Commit `ea672a8`. Probe A.6.4 emite `pit_status` explicito:

    strict-pit (default): target_catalog_as_of("2026-03-31") ->
      CatalogNotAvailable (snapshot vigente valid_from=2026-09-22
      no cubre Q1) -> PIT_UNAVAILABLE, coverage_current=None,
      coverage_contractual=False, coverage_status=PIT_UNAVAILABLE.
    --no-pit: bypass documentado con PIT_BYPASS_DOCUMENTED.

El numero tecnico pre-PIT se preserva bajo
`coverage_current_technical_pre_pit` para trazabilidad.

### 3.5. Ciclo B-03 (materializacion TARGET) - CERRADO

Commits `a8ad54b` + `7f314ef` + `9e04cb8` + `d18cf62`.

    BRK-B: CUSIP 084670702 -> OpenFIGI US -> scf BBG001S90346
    MOG-A: CUSIP 615394202 -> OpenFIGI US -> scf BBG001S5T922
    Catalogo: 240/242 -> 242/242 keys OK
    Snapshot B2-PIT: 20260922_01 (anterior 20260921_01 preservado)
    Membership: migrada, catalog_key inmutable, 2 predecessors
    Probe A.6.4: catalog_keys Q1 20 -> 22 (coverage tecnico 22/22)

Evidencia: `data/mappings/openfigi_requeries/` con input/raw/summary
y HASHES.txt.

## 4. TRABAJO PENDIENTE

### 4.1. Autorizado por #77

    B-04 (fixture pairwise NON-PRODUCTION):
      Disenar fixture sintetico con 2-5 FIGIs, overlap Q4<->Q1
      no vacio, marcado explicitamente NON_PRODUCTION /
      CONTRACT_TEST_FIXTURE. NO toca mappings productivos.
      Objetivo: ejercitar la rama pairwise (paired_security_coverage
      y paired_weighted_share_coverage con TARGET_PAIRWISE > 0).
      Alternativa aceptada por #77: mappings Q4 con evidencia
      historica real (NO fabricados para producir overlap).

### 4.2. Sub-deudas registradas (no bloqueantes)

    H-19 (MEDIA): bundle con multiples HEAD ambiguos.
      Requiere etiquetar HEAD_DEL_EXPEDIENTE vs HEAD_DEL_CODIGO_AUDITADO
      vs HEAD_DE_LA_EVIDENCIA antes del proximo bundle.
    H-20 (MEDIA): afirmacion CI 0 failed sin SHA/run.
      Requiere aportar CI run + commit SHA + resultado.
    Contrato NIPC L332: cita 18 FIGI -> 18 records; real ahora 22.
      Prohibido tocar sin dictamen (contrato normativo).
    OpenFIGI requeries raw.json = 121 KB (expandido en A66_DIFFS.txt).
      Considerar excluir del proximo bundle via git pathspec.

### 4.3. NO autorizado (esperar dictamen)

    A.6.6 / F2.4-CLOSE: NO-GO en #76 y #77.
    Push a origin/main: NO (local-first IAE).
    OpenFIGI masivo: NO.
    Fabricar overlap con exceptions Q4: NO.
    Retro-fechar snapshot: NO.
    Modificar contratos normativos sin dictamen: NO.

## 5. Ficheros que el trabajo pendiente TOCA (autorizados)

    docs/auditoria/iae/evidence/a64_integration_b1_p61_p38/
      probe_pairwise_fixture.py        (nuevo, B-04)
      result_pairwise_fixture.json     (nuevo, B-04)
      HASHES.txt                       (regenerar tras B-04)
    tests/test_p38_pairwise_fixture.py (nuevo, B-04)

## 6. Ficheros PROHIBIDOS (no tocar sin dictamen)

    src/institutional_accumulation/aggregation/nipc.py
    src/institutional_accumulation/aggregation/delta_shares.py
    src/institutional_accumulation/sec_13f/identity/security_identity.py
    src/institutional_accumulation/temporal_validity.py
    src/institutional_accumulation/sec_13f/identity/relationships.py
    MATCH_KEY, C2
    Contratos normativos (NIPC_CONTRATOS_SEMANTICOS_v1.md,
      NIPC_COVERAGE_POLICY.md, INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md)

## 7. Correcciones documentales aplicadas hoy (2026-09-22)

    A-01  ESTADO_SISTEMA lista los 3 tests fallidos por nombre
    A-04  Separadas metricas diagnosticas (18/242) de contractuales
          (RESOLVED/TARGET) en EXPEDIENTE_A64_FIX
    A-12  Test xfail(strict=True) H-10.1 -> reescrito como positivo
          en commit 3 (A2 cierra H-10.1)
    #77   Correcciones materiales a #76 aplicadas en 3 expedientes:
          B02_EXPEDIENTE, B034_EXPEDIENTE, A67_CONSULTA
    B-02  Probe declara PIT_UNAVAILABLE; coverage_current anulado
          como contractual (commit ea672a8)
    B-03  242/242 keys materializadas; snapshot 20260922_01
    H-05..H-10.1, H-08, B-01..B-07: CERRADOS

## 8. Documentos clave (vivos)

    docs/auditoria/PROMPT_MAESTRO.md         v7.1 - normativa general
    docs/auditoria/iae/ESTADO_DECLARADO.md   fases, prohibiciones, hallazgos
    docs/auditoria/iae/ESTADO_SISTEMA.md     hechos (autogenerado)
    docs/auditoria/iae/DICTAMENES.md         indice #1-#77
    docs/auditoria/iae/HISTORICO_IAE.md      registro de ciclos
    docs/auditoria/iae/NIPC_CONTRATOS_SEMANTICOS_v1.md  contrato P60-P70
    docs/auditoria/iae/NIPC_COVERAGE_POLICY.md          policy normativa
    docs/auditoria/iae/EXPEDIENTE_A64_FIX.md  expediente del fix A2
    docs/auditoria/iae/A66_BUNDLE.md          bundle previo #76
    docs/auditoria/iae/A67_CONSULTA.md        consulta #77 (larga)
    docs/auditoria/iae/A67_RESUMEN.md         resumen ejecutivo #77
    docs/auditoria/iae/B02_EXPEDIENTE.md      correccion P62
    docs/auditoria/iae/B034_EXPEDIENTE.md     correccion B-03/B-04
    docs/auditoria/iae/A66_DIFFS.txt          diffs completos del ciclo
    docs/auditoria/iae/evidence/a64_integration_b1_p61_p38/README.md

## 9. Comandos de arranque

    Set-Location D:\Macro_Sectorial
    git log --oneline -5
    git status -sb
    py scripts\generate_estado_sistema.py
    py -m pytest tests/ validation/ -q --tb=line
    py -m pyflakes . ; py -m compileall . -q

Esperado:
  - HEAD ac1edc3 o posterior
  - ahead 331 o mas
  - working tree limpio
  - 1415 passed + 2 skipped + 3 failed (test_freshness, preexistentes)
  - test_h731: 11 passed, 0 xfailed
  - pyflakes silencio, compileall OK

## 10. Confirmacion esperada

    "Confirmado, contexto asimilado."

    Estado que reconozco:
      - HEAD post-B-04, ahead 333
      - Dictamenes #76 (A2=GO, A.6.6=NO-GO) + #77 (B-02/B-03/B-04)
      - Fix A2 y saneamiento post-#76: CERRADOS
      - Ciclo B-02 (PIT_UNAVAILABLE) + B-03 (242/242): CERRADOS
      - B-04 (fixture pairwise NON-PRODUCTION) CERRADO
      - Prohibido: nipc/delta_shares/security_identity/temporal_validity,
        contratos normativos, push, OpenFIGI masivo, retro-fechado

    Pregunta: "Que hacemos?"

---

FIN DEL TRANSFER v10.1. 2026-09-22. Ciclos B-02, B-03 y B-04 CERRADOS.
