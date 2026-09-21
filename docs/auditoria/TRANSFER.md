# TRANSFER DE SESION - 2026-09-21 v8

Documento complementario al PROMPT_MAESTRO v6.55 (docs/auditoria/PROMPT_MAESTRO.md).
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

Reglas de personalidad y metodo: ver PROMPT_MAESTRO v6.55 secciones 1 y 3.

---

## 2. Objetivo real del proyecto

**El objetivo es IMPLEMENTAR el modulo IAE (Institutional Accumulation
Evidence) - analisis de posiciones institucionales via SEC 13F.**

NO es redactar documentos de diseño. NO es iterar propuestas ->
dictamenes -> propuestas.

**Regla nueva (aplicada con exito en B3, B1, A.6.4):** cuando el
diseño de una subfase haya cerrado >=3 rondas de dictamen con patron
"cierro N, aparecen N nuevos", PARAR. Congelar diseño en la version
vigente e IMPLEMENTAR. Las dudas se resuelven escribiendo tests, no
documentos.

---

## 3. Estado actual verificado (2026-09-21)

| Metrica | Valor |
|---|---|
| HEAD local | da70f9e |
| origin/main | 9d4a81e |
| Ahead | 246 commits locales |
| Behind | 3 (bot CI) |
| Working tree | limpio |
| Push | NO (local-first IAE activo) |
| Prompt vigente | v6.55 |
| Tests locales | 1301 passed + 2 skipped + 3 failed preexistentes |
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
| B2-PIT | CERRADO (#53) |
| B1 | CERRADO (#68, 4 commits + 100 tests) |
| B3 (semantica temporal 13F) | IMPLEMENTADO SPEC-SIDE (sin integracion) |
| Gap spec->codigo §5.1-§5.5 | CERRADO (#61-#67) |
| H-69.1 (source id) | CERRADO (#70) |
| H-69.2 (precedencia temporal) | CERRADO (#72) |
| H-73.1 (adapter B1<->P38) | CERRADO (#75) |
| GHISALLO (+765%) | CERRADO (#60) |
| A.6.3 (test P38 Q12) | CERRADO (#72) |
| A.6.4-v2 (P38 aislado) | CERRADO (#73) |
| A.6.4 (integracion B1+P61+P38) | CERRADO (#75) |
| Orquestador `iae_pipeline.py` | IMPLEMENTADO + RATIFICADO |
| A.6.5 | AUTORIZADA, pendiente |
| A.6.6 (F2.4-CLOSE) | PENDIENTE |
| Baseline full | BLOQUEADO por OpenFIGI masivo |
| Gate-NIPC.2 | NOT READY (thresholds UNDEFINED) |

### 4.1. Modulos IAE clave

    src/institutional_accumulation/
      sec_13f/              ingestion + schema + parser + identity
      identity/             security_identity, catalog_key,
                            target_builder, period_state,
                            radar_target_catalog, target_universe,
                            openfigi_client
      aggregation/          delta_shares, nipc, coverage,
                            catalog_validator, catalog_p38_adapter,
                            reporting_dedup
      temporal_validity.py  P61 operational_mapping_status
      security_type.py      §3.3 + §5.4 Nivel A+B
      operational_universe.py §5.5
      timestamps.py         B3 derive + enrich
      absence.py            P63 stub
      catalog_pit.py        B2-PIT

    scripts/iae_pipeline.py         Orquestador productivo (ratificado)
    scripts/build_catalog_csvs.py   Generador CSV catalogo B1

---

## 5. Deuda activa (no bloqueante)

- Integracion de modulos IAE al pipeline productivo principal: los
  modulos estan implementados y ejecutables via `scripts/iae_pipeline.py`,
  pero el pipeline diario (`daily_run.yml`) NO los invoca.
- `compute_nipc_contractual` SIN CALLERS PRODUCTIVOS.
- Marcadores residuales §5.4 (SPONSORED ADR/ADS, ADR, SH BEN INT,
  FUND, ACT).
- Snapshots historicos Q4 2025 / Q1 2026 requieren OpenFIGI masivo
  (NO AUTORIZADO).
- Curacion crosswalk residual: ONB, SPCX, PTGX sin match 13F Q1 2026.
- `reporting_dedup.py` (P66) sin invocacion productiva.
- Gate-NIPC.2 BLOQUEADO por THRESHOLD_1/2 UNDEFINED.
- A.6.5 y A.6.6 pendientes.

---

## 6. Reglas criticas

### 6.1. Local-first IAE

NO push a main de codigo hasta validar funcionalidad + beneficio.

### 6.2. Prohibiciones vigentes

    NO activar compute_nipc_contractual
    NO ejecutar OpenFIGI masivo
    NO activar DROP_DUP
    NO modificar contratos normativos sin autorizacion
    NO reescribir snapshots publicados
    NO modificar coverage.py, nipc.py, delta_shares.py
    NO modificar security_identity.py ni temporal_validity.py
    NO push

### 6.3. Precedente metodologico

B1 estaba 3 NO-GO con 4 versiones. Se congelo v4 y se implemento en 4
commits + dictamen #68 GO. 100 tests, cero regresiones.

A.6.4 acumulo 2 partes (v2 + integracion) con H-73.1 como hallazgo
material. Corregido + regresion + revalidacion. Cerrado por #75.

Regla: si el patron "cierro N, aparecen N nuevos" se repite 3 veces,
PARAR e IMPLEMENTAR.

---

## 7. Ficheros clave

### 7.1. Contratos y specs

    docs/auditoria/PROMPT_MAESTRO.md
    docs/auditoria/iae/INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md
    docs/auditoria/iae/NIPC_CONTRATOS_SEMANTICOS_v1.md
    docs/auditoria/iae/NIPC_COVERAGE_POLICY.md

### 7.2. Diseños por subfase

    docs/auditoria/iae/A60_EXPEDIENTE_AUDITOR.md
    docs/auditoria/iae/A62BIS_B1_SUBFASE.md           v4 (CERRADO #68)
    docs/auditoria/iae/A62BIS_B2_PIT_SUBFASE.md       CERRADO #53
    docs/auditoria/iae/A62BIS_PROPUESTA.md            v9 base
    docs/auditoria/iae/FASE_A6_PLAN.md                estado A.6.x

### 7.3. Dictamenes y cronologia

    docs/auditoria/iae/DICTAMENES.md                  #1 a #75
    docs/auditoria/iae/INFORME.md
    docs/auditoria/FOLLOWUPS.md

### 7.4. Evidencia empirica

    docs/auditoria/iae/evidence/
      nipc_gate0_baseline/           coverage baseline
      nipc_gate0_target_identity_top2000/   piloto TOP 2000
      nipc_gate0_top2000_v2/         A.6.4-v2 (P38 aislado)
      a64_integration_b1_p61_p38/    A.6.4 (integracion completa)
      b2_pit_cierre/, p66_*, nipc_p65_probe/, nipc_p70_probe/

---

## 8. Comandos de arranque

    Set-Location D:\Macro_Sectorial
    git log --oneline -10
    git status -sb
    git rev-list --count origin/main..HEAD
    py -m compileall . -q
    py -m pyflakes . 2>&1
    py -m pytest tests/ validation/ -q --tb=line

Esperado:
- HEAD da70f9e o posterior
- ahead 246 o mas
- working tree limpio
- 1301 passed + 2 skipped + 3 failed (freshness preexistentes)
- pyflakes silencio
- compileall OK

---

## 9. Confirmacion esperada

    "Confirmado, contexto asimilado."

    Estado del sistema que reconozco:
      - HEAD da70f9e, ahead 246
      - 1301 passed + 2 skipped + 3 failed preexistentes
      - Prompt v6.55
      - B1 CERRADO (#68). B2-PIT CERRADO (#53).
      - Gap §5.1-§5.5 CERRADO (#61-#67).
      - H-69.1/H-69.2/H-73.1 CERRADOS (#70/#72/#75).
      - A.6.3, A.6.4-v2, A.6.4 CERRADOS.
      - A.6.5 AUTORIZADA. A.6.6 PENDIENTE.
      - Baseline full BLOQUEADO por OpenFIGI.
      - Gate-NIPC.2 NOT READY.
      - Objetivo real: IMPLEMENTAR IAE

    Pregunta final: "Que hacemos?"

No empieces a proponer tareas sin antes confirmar asimilacion.

---

FIN DEL TRANSFER
Version 8.0 (2026-09-21). HEAD da70f9e.