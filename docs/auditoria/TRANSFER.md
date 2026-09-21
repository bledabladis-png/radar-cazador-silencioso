# TRANSFER DE SESION - 2026-09-21 v7

Documento complementario al PROMPT_MAESTRO v6.53 (docs/auditoria/PROMPT_MAESTRO.md).
No normativo. Si hay conflicto, gana el prompt.

**Como usarlo:**
1. Pegar este documento como primer mensaje.
2. Si tienes acceso al repo, adjuntar tambien PROMPT_MAESTRO.md.
3. Esperar confirmacion de asimilacion antes de empezar.

---

## 1. Rol

Eres el Ingeniero Supervisor del Radar de Rotacion Sectorial. Sistema
determinista, descriptivo, auditable, sin ML predictivo. Entorno
Windows, PowerShell, Python via py. Repo: D:\Macro_Sectorial.

Reglas de personalidad y metodo: ver PROMPT_MAESTRO v6.53 secciones 1 y 3.

---

## 2. Objetivo real del proyecto

**El objetivo es IMPLEMENTAR el modulo IAE (Institutional Accumulation
Evidence) - analisis de posiciones institucionales via SEC 13F.**

NO es redactar documentos de diseño. NO es iterar propuestas ->
dictamenes -> propuestas.

**Regla nueva (aplicada con exito en B3 y B1):** cuando el diseño de
una subfase haya cerrado >=3 rondas de dictamen con patron "cierro N,
aparecen N nuevos", PARAR. Congelar diseño en la version vigente e
IMPLEMENTAR. Las dudas se resuelven escribiendo tests, no documentos.

Precedentes: B3 (4 commits, #57-#60) y B1 (4 commits, #68).

---

## 3. Estado actual verificado (2026-09-21)

| Metrica | Valor |
|---|---|
| HEAD local | 9f0d841 |
| origin/main | 9d4a81e |
| Ahead | 238 commits locales |
| Behind | 3 (bot CI) |
| Working tree | limpio |
| Push | NO (local-first IAE activo) |
| Prompt vigente | v6.53 |
| Tests locales | 1289 passed + 2 skipped + 3 failed preexistentes |
| pyflakes / compileall | LIMPIO / OK |
| Gate validacion | 10/10 |
| Cobertura radar | 313/313 |
| Contratos temporales | 10 |

**Los 3 failed de test_freshness.py son preexistentes** (parquets
locales desactualizados). No son regresion.

---

## 4. Estado del modulo IAE

| Bloque | Estado |
|---|---|
| FA-1 + FA-2 (ingestion + identity) | CERRADO / PUSHED |
| NIPC (5 modulos + 91 tests) | IMPLEMENTADO (usa proxy observacional) |
| P60/P61/P38/P63/P64/P65 | CERRADOS documentalmente |
| P66 (§14.3 + reporting_dedup) | CERRADO (codigo implementado) |
| A.6.0 Gate 0 | CERRADO |
| A.6.2-bis-B2-PIT | CERRADO (#53) |
| A.6.2-bis-B1 | **CERRADO (#68)** |
| A.6.2-bis-B3 | IMPLEMENTADO SPEC-SIDE (sin integracion) |
| Gap spec->codigo §5.1-§5.5 | **CERRADO (#61-#67)** |
| GHISALLO (+765%) | CERRADO (#60) |
| A.6.3 | BLOQUEADO (requiere dictamen especifico) |
| Gate-NIPC.2 | NOT READY |

---

## 5. Modulos IAE implementados (resumen)

### 5.1. Gap spec->codigo §5.1-§5.5

    src/institutional_accumulation/security_type.py          (Nivel A+B)
    src/institutional_accumulation/operational_universe.py   (§5.5)
    tests/test_security_type.py                              (55 tests)
    tests/test_operational_universe.py                       (19 tests)

Contratos: ticker_mapped = CANONICAL AND VERIFIED (#65); PIT
obligatorio (identity_period_iso, #66 H-66.1 / #67).

### 5.2. B1 (TARGET + adaptador P38)

    src/institutional_accumulation/identity/catalog_key.py           (304)
    src/institutional_accumulation/identity/target_builder.py        (115)
    src/institutional_accumulation/identity/period_state.py          (113)
    src/institutional_accumulation/aggregation/catalog_validator.py  (74)
    src/institutional_accumulation/aggregation/catalog_p38_adapter.py (119)
    scripts/build_catalog_csvs.py                                    (127)
    data/mappings/catalog_assignments.csv                            (242)
    data/mappings/catalog_membership.csv                             (242)
    tests: test_catalog_key, test_catalog_membership, test_b1_schema,
           test_target_builder, test_period_state,
           test_catalog_p38_adapter, test_p66_pipeline,
           test_build_catalog_csvs                                  (100)

### 5.3. B3 (semantica temporal 13F)

    src/institutional_accumulation/absence.py                (stub, P63)
    src/institutional_accumulation/timestamps.py             (derive + enrich)
    tests/test_absence.py, tests/test_timestamps.py          (26 tests)
    PositionRecord extendido con period_end/filing_date/knowledge_date.

### 5.4. Modulos con contrato PIT

    src/institutional_accumulation/temporal_validity.py
    src/institutional_accumulation/sec_13f/identity/security_identity.py
    src/institutional_accumulation/sec_13f/identity/sec13f_list.py

---

## 6. Deuda activa (no bloqueante)

- Integracion de los modulos IAE al pipeline productivo: **NADIE los
  invoca fuera de tests**. No hay script/workflow que ejecute
  `ingest_13f -> ... -> build_operational_universe`.
- `compute_nipc_contractual` sigue SIN CALLERS PRODUCTIVOS.
- Marcadores residuales §5.4 (SPONSORED ADR/ADS, ADR, SH BEN INT,
  FUND, ACT).
- Snapshots historicos Q4 2025 / Q1 2026 requieren OpenFIGI masivo
  (NO AUTORIZADO).
- Curacion crosswalk residual: ONB, SPCX, PTGX sin match 13F Q1 2026.
- `reporting_dedup.py` (P66) sin invocacion productiva.
- Gate-NIPC.2 BLOQUEADO por THRESHOLD_1/2 UNDEFINED.

---

## 7. Reglas criticas

### 7.1. Local-first IAE

NO push a main de codigo hasta validar funcionalidad + beneficio.
Todo el trabajo IAE de esta sesion esta en local.

### 7.2. Prohibiciones vigentes

    NO activar compute_nipc_contractual
    NO ejecutar OpenFIGI masivo
    NO activar DROP_DUP
    NO modificar contratos normativos
    NO reescribir snapshots publicados
    NO modificar coverage.py, nipc.py, delta_shares.py
    NO modificar security_identity.py ni temporal_validity.py
    NO push

### 7.3. Precedente metodologico

B1 estaba 3 NO-GO con 4 versiones. Se congelo v4 y se implemento en 4
commits (B1.0/B1.1/B1.2/B1.3) + dictamen #68 GO. Resultado: 100 tests,
cero regresiones.

Regla: si el patron "cierro N, aparecen N nuevos" se repite 3 veces,
PARAR e IMPLEMENTAR.

---

## 8. Ficheros clave

### 8.1. Contratos y specs

    docs/auditoria/PROMPT_MAESTRO.md
    docs/auditoria/iae/INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md
    docs/auditoria/iae/NIPC_CONTRATOS_SEMANTICOS_v1.md
    docs/auditoria/iae/NIPC_COVERAGE_POLICY.md

### 8.2. Diseños por subfase

    docs/auditoria/iae/A62BIS_B1_SUBFASE.md           v4 (CERRADO #68)
    docs/auditoria/iae/A62BIS_B2_PIT_SUBFASE.md       CERRADO #53
    docs/auditoria/iae/A62BIS_PROPUESTA.md            v9 base
    docs/auditoria/iae/FASE_A6_PLAN.md
    docs/auditoria/iae/A60_EXPEDIENTE_AUDITOR.md

### 8.3. Dictamenes y cronologia

    docs/auditoria/iae/DICTAMENES.md                  #1 a #68
    docs/auditoria/iae/INFORME.md
    docs/auditoria/FOLLOWUPS.md

### 8.4. Evidencia empirica

    docs/auditoria/iae/evidence/
      p66_gate0*/, p66_e2e_probe/, p66_baseline_pre/, b2_pit_cierre/

---

## 9. Comandos de arranque

    Set-Location D:\Macro_Sectorial
    git log --oneline -10
    git status -sb
    git rev-list --count origin/main..HEAD
    py -m compileall . -q
    py -m pyflakes . 2>&1
    py -m pytest tests/ validation/ -q --tb=line

Esperado:
- HEAD 9f0d841 o posterior
- ahead 238 o mas
- working tree limpio
- 1289 passed + 2 skipped + 3 failed (freshness preexistentes)
- pyflakes silencio
- compileall OK

---

## 10. Confirmacion esperada

    "Confirmado, contexto asimilado."

    Estado del sistema que reconozco:
      - HEAD 9f0d841, ahead 238
      - 1289 passed + 2 skipped + 3 failed preexistentes
      - Prompt v6.53
      - §5.1-§5.5 CERRADOS (#61-#67)
      - B1 CERRADO (#68), 4 commits, 100 tests
      - B3 IMPLEMENTADO SPEC-SIDE (sin integracion)
      - B2-PIT CERRADO (#53)
      - GHISALLO CERRADO (#60)
      - A.6.3 BLOQUEADO (requiere dictamen)
      - Gate-NIPC.2 NOT READY
      - Objetivo real: IMPLEMENTAR IAE
      - Proximo paso natural: orquestador productivo (integración)
        o nueva subfase autorizada por dictamen

    Pregunta final: "Que hacemos?"

No empieces a proponer tareas sin antes confirmar asimilacion.

---

FIN DEL TRANSFER
Version 7.0 (2026-09-21). HEAD 9f0d841.