# HANDOFF - Radar de Rotacion Sectorial

Documento de traspaso entre sesiones. Autocontenido. Pegar al inicio
del nuevo chat junto con `Consolidacion_Documentos/00_ARRANQUE.md`.

**Este fichero se sobrescribe cada sesion.** El historico queda en git.
No crear variantes con fecha.

---

## 1. ESTADO AL CIERRE (2026-10-16 noche)

Rama activa:    main
HEAD:           consultar `git log --oneline -1` (no literal; el corpus no fija SHA para evitar deriva)
Sincronizacion: main == origin/main
Tests:          ver `Consolidacion_Documentos/ESTADO_SISTEMA.md` (fuente autoritativa).
                Referencia orientativa: ~3430 passed + 5 skipped (con filtro CI).
Validation Gate: 10/10
Working tree:   limpio (salvo data/ y outputs/ que el bot actualiza)
Corpus:         v3 (Consolidacion_Documentos/00-08 + ESTADO_SISTEMA)
Gobernanza:     docs/gobernanza/REGISTRO.yaml (fuente canonica de items auditables)

---

## 2. SESION 16-OCT (auditoria interna CI + dictamen externo + fases 1-7)

### 2.0. Contexto

Sesion larga en tres bloques:

**Bloque 1**: auditoria interna del workflow `daily_run.yml` tras el
cierre del 15-oct. Detectados 9 hallazgos (HD-1 a HD-9). Corregidos
los 8 accionables + precision de 02_ARQUITECTURA Â§6.

**Bloque 2**: informe al auditor externo + dictamen recibido.
Aprobados 7 frentes con condiciones (orden estricto, criterio triple
de calidad). Implementadas fases 1, 2 y 3 del dictamen.

**Bloque 3**: implementadas fases 4, 5, 6 y 7 del dictamen.
Sistema de gobernanza documental completo.

### 2.1. Auditoria workflow daily_run.yml (HD-1..HD-9)

- **HD-1**: drift test `pipeline_gate` vs `health_check` ampliado.
- **HD-2**: comentarios "4 slots" -> "5 slots" en `daily_run.yml` y
  `health_check.py`.
- **HD-3**: `resolve_target_session` naive-UTC vs Madrid. Verificado
  empiricamente que los 5 slots dan identico target_session. WONT FIX
  razonado en docstring.
- **HD-4**: `guard_coverage` y `pipeline_conclusion` hardcodeados en
  receipt. Resuelto en Fase 2 (ver Â§2.3).
- **HD-5**: `health_check` no distinguia Schedule vs Dispatch.
  Resuelto en Fase 1 (ver Â§2.3).
- **HD-6**: `pyflakes . || true` nunca bloqueaba CI. Resuelto en
  Fase 3 (ver Â§2.3).
- **HD-7**: mismo fix que HD-4.
- **HD-8**: `--limit 20` en `check_cron_slots` con comentario
  justificativo.
- **HD-9**: `_manifest_satisfies` verificado que NO es dead code
  (~8 tests + contrato).

### 2.2. Informe al auditor y dictamen

Informe enviado con 3 debilidades de diseno del CI (HD-4/HD-5/HD-6) y
4 consultas de metodo (P-A gobernanza HANDOFF, P-B anclas temporales,
P-C trazabilidad normativa, P-D poda bitacora). Dictamen: los 7
frentes aprobados con condiciones. Orden estricto de implementacion.
Criterio de calidad triple: estado inequivoco + evidencia verificable
+ test que detecte regresion.

### 2.3. Fases del dictamen (1-7 implementadas)

**Fase 1 - HD-5.** Commit `3e8d2828`. `check_workflows` refactorizado
en `_check_one_workflow` (pura) + helpers. Schedule Health separado de
Dispatch Health. Dispatch reciente NUNCA oculta ausencia de schedule.
8 tests.

**Fase 2 - HD-4.** Commit `433da326`. `run.py` escribe `written_at`
al VG result. `write_completion_receipt.py` valida estrictamente:
`passed=True, errors_n=0, checks_n>=10, written_at <=6h`. Fail-closed.
Campos anadidos: `success_path_reached`, `validation_gate_checks_n`,
`validation_gate_errors_n`. Compat schema v1. 9 tests.

**Fase 3 - HD-6.** Commit `66179b90`. `daily_run.yml`: eliminado
`|| true` del step pyflakes. Falla el build si hay warnings. Reporte
preservado en artifact `pyflakes-report` con `if: always()`. 5 tests.

**Fase 4 - P-A: registro canonico.** Fuente unica
`docs/gobernanza/REGISTRO.yaml` (38 items iniciales: H1-H18, HD-1..HD-9,
DICT-F1..F7, 4 frentes). Estados: OPEN, IN_PROGRESS, BLOCKED_EXTERNAL,
FIXED, WONT_FIX, FALSE_POSITIVE, SUPERSEDED. Verificador
`scripts/verifica_gobernanza.py` con 11 reglas. 14 tests.

**Fase 5 - P-C: indice H1-H18.** `docs/auditoria/radar/audits/
INDEX_H1_H18.md` generado desde REGISTRO por
`scripts/genera_indice_auditorias.py`. Vista derivada, nunca editada a
mano. 5 tests.

**Fase 6 - P-B: anclas temporales.** Catalogo
`docs/gobernanza/ANCLAS.yaml` con 6 anclas documentadas (SOFR,
DEFAULT_SINCE, CONFIRMED_INCOMPLETE_DATES, IAE_FULL_PERIOD,
IAE_PIPELINE_PERIODS, MACRO_MANUAL_START_DATE). Test selectivo sobre
literales ISO en asignaciones modulo y defaults de funcion. 6 tests.

**Fase 7 - P-D: poda bitacora.** `scripts/poda_bitacora.py` con modo
`--check` (exit 1 si hay que podar, NO muta). Sesion = fecha unica
(agrupa headings del mismo dia). 7 tests.

### 2.4. Commits de la sesion

  529f4ed2e1ab037940e88f65449cd82f2080a615  feat(gobernanza): fases 4-7 del dictamen 16-oct
  66179b90 fix(ci): HD-6 - pyflakes como gate real + artifact
  7b790b12 docs(corpus): cierre sesion 16-oct - HANDOFF + bitacora + ESTADO_SISTEMA
  433da326 fix(receipt): HD-4 - validar VG result antes de emitir receipt
  3e8d2828 fix(health_check): HD-5 - separar Schedule vs Dispatch
  083cfd39 docs(corpus): precisar MTE bonos secundarios y fallback SPY->^GSPC
  2dda7986 docs(ci)+test: auditoria workflow daily_run (HD-1..HD-8)
  8269d671 fix(tests)+docs(handoff): residuo pyflakes H7 + cierre deuda HANDOFF
  fd42010d docs(corpus): MTE + rs_abs_20d en 02_ARQUITECTURA Â§6
  c8a16f06 docs(corpus): HD-6 - poda bitacora + script poda_bitacora.py

---

## 3. FRENTES CERRADOS EN LA SESION

- Auditoria workflow `daily_run.yml`: HD-1 a HD-9.
- Residuo pyflakes H7.
- Dictamen externo 16-oct: Fases 1-7 implementadas y verificadas.
- Poda bitacora (21 -> 9 sesiones unicas; script `poda_bitacora.py`
  con `--check`).
- Precision `02_ARQUITECTURA Â§6` (bonos MTE secundarios, fallback
  SPY->^GSPC).
- Sistema de gobernanza documental (`docs/gobernanza/`).

---

## 4. PENDIENTES VIVOS

### 4.1. Frentes bloqueados por externos

- **C2 (920)**: discrepancia IAE par Q4-2025 -> Q1-2026. Bloqueado
  por auditor externo. Estado en `REGISTRO.yaml`.
- **SOW v6**: `docs/auditoria/wyckoff/48_sow_protocol_v6.md` pendiente
  de firma externa.
- **Wyckoff 5b.X v3**: bloqueado por datos (~2027).
- **H5.3**: verificacion del cron 13F del 20-nov-2026 06:17 UTC.

### 4.2. Verificacion del cron tras Fase 2

El cambio en `write_completion_receipt.py` es fail-closed. Verificar
el primer run post-commit que el receipt se sigue emitiendo. Ultimo
run confirmado: `38834597307` (16-oct 01:20 UTC).

### 4.3. Aviso GitHub

`ubuntu-latest` migra a Ubuntu 26 el 19-oct-2026.

### 4.4. Issue #5

`[health-check] WARN - discrepancias no criticas`. Deuda de proceso.

### 4.5. Verificacion del propio sistema de gobernanza

Anadir `py scripts/verifica_gobernanza.py` al workflow `daily_run.yml`
como gate de CI (comando de arranque ya lo incluye). Opcional:
decidir si va a health_check semanal o a daily_run.

---

## 5. COMO ARRANCAR

    Set-Location D:\Macro_Sectorial
    git log --oneline -5
    git status -sb
    py scripts\generate_estado_sistema.py
    py scripts\verifica_gobernanza.py
    py scripts\genera_indice_auditorias.py
    py scripts\poda_bitacora.py --check
    py -m pytest tests/ -q -n 12 --tb=line -m "not integration_real_data"
    py -m pyflakes . ; py -m compileall . -q

Esperado:
- HEAD = el que devuelva `git log --oneline -1`.
- ahead 0, behind 0 (o behind N si el bot ha pusheado).
- working tree limpio.
- ~3430 passed + 5 skipped.
- pyflakes silencio, compileall OK.
- Verificador: `[OK] Gobernanza coherente`.
- Indice: `[OK] indice actualizado`.
- Poda: `[OK] Bitacora dentro de la regla`.

---

## 6. NOTAS CRITICAS

- **H7 fue un bug real.** Test AST en
  `tests/test_audit_contracts_rolling_nan.py`. Si algun fix futuro
  necesita rolling sobre series con huecos, dropna previo.
- **H2 es falso positivo.** `rs_mom` (WLS) invariante a escala.
- **No usar run.py en horario de mercado USA** (13:30-20:00 UTC).
- **Receipt fail-closed.** Si el VG result no cumple el contrato
  estricto, no se emite receipt. Siguiente slot hara READY.
- **Schedule vs Dispatch.** Un dispatch reciente NUNCA oculta
  ausencia de schedule.
- **Gobernanza canonica.** Cualquier item de auditoria, frente o
  dictamen se registra en `docs/gobernanza/REGISTRO.yaml`. El HANDOFF,
  la bitacora y el indice son VISTAS; nunca fuente de estado.
- **Anclas temporales.** Cualquier fecha literal en asignacion modulo
  o default de funcion debe estar en `docs/gobernanza/ANCLAS.yaml`.
  El test `tests/test_anclas_temporales.py` lo exige.
- **EOL del corpus:** LF.

---

## 7. REFERENCIAS

Corpus normativo: `Consolidacion_Documentos/` (12 docs).
Gobernanza: `docs/gobernanza/` (README, REGISTRO.yaml, ANCLAS.yaml).
Expedientes: `docs/auditoria/wyckoff/`, `docs/auditoria/iae/golden/`,
`docs/auditoria/s03_deep/`, `docs/auditoria/radar/audits/`.
Contrato del gate: `docs/auditoria/daily_run_gate_contrato_v1.md`.
Indice auditoria externa: `docs/auditoria/radar/audits/INDEX_H1_H18.md`.
Sandbox N8: `outputs/audit/liq_backtest/`.

---

**Fin del handoff.**