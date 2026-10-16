# HANDOFF - Radar de Rotacion Sectorial

Documento de traspaso entre sesiones. Autocontenido. Pegar al inicio
del nuevo chat junto con `Consolidacion_Documentos/00_ARRANQUE.md`.

**Este fichero se sobrescribe cada sesion.** El historico queda en git.
No crear variantes con fecha.

---

## 1. ESTADO AL CIERRE (2026-10-16)

Rama activa:    main
HEAD:           consultar `git log --oneline -1` (no literal; el corpus no fija SHA para evitar deriva)
Sincronizacion: main == origin/main
Tests:          ver `Consolidacion_Documentos/ESTADO_SISTEMA.md` (fuente autoritativa).
                Referencia orientativa: ~3406 passed + 5 skipped (con filtro CI).
Validation Gate: 10/10
Working tree:   limpio (salvo data/ y outputs/ que el bot actualiza)
Corpus:         v3 (Consolidacion_Documentos/00-08 + ESTADO_SISTEMA)

---

## 2. SESION 16-OCT (auditoria interna CI + dictamen externo + fases 1-2)

### 2.0. Contexto

Dos bloques en la sesion del 16-oct:

**Bloque 1 (mañana/tarde temprano)**: auditoria interna del workflow
`daily_run.yml` tras el cierre del 15-oct. Detectados 9 hallazgos
(HD-1 a HD-9) y corregidos los 8 accionables. Emitido informe al
auditor externo con 3 debilidades de diseno + 4 consultas de metodo.

**Bloque 2 (tarde/noche)**: recibido dictamen del auditor externo.
Aprobados los 7 frentes con condiciones, con orden de implementacion
estricto (una fase por sesion). Implementadas Fases 1 y 2.

### 2.1. Auditoria workflow daily_run.yml (HD-1..HD-8)

- **HD-1**: drift test entre `pipeline_gate.py` y `daily_run.yml`
  existia, pero no cubria `health_check.py`. Ampliado con
  `test_drift_cron_slots_vs_health_check`.
- **HD-2**: 2 comentarios "4 slots" en `daily_run.yml` y
  `health_check.py` vs 5 reales. Corregidos.
- **HD-3**: `resolve_target_session` con naive-UTC interpretado como
  Madrid. Verificado empiricamente que los 5 slots dan el mismo
  target_session con naive-UTC o UTC-aware. WONT FIX razonado en
  docstring.
- **HD-4** (ver §2.3 abajo, Fase 2 del dictamen).
- **HD-5** (ver §2.3 abajo, Fase 1 del dictamen).
- **HD-6**: `pyflakes . || true` nunca bloquea CI. Recogido en
  dictamen como Fase 3 pendiente.
- **HD-7**: `guard_coverage: "OK"` y `pipeline_conclusion: "success"`
  hardcoded en `write_completion_receipt.py`. Resuelto en Fase 2.
- **HD-8**: `--limit 20` en check_cron_slots sin justificacion.
  Comentario anadido.
- **HD-9**: `_manifest_satisfies` sin consumidores. Verificado que SI
  tiene ~8 tests dedicados + contrato declarado. NO es dead code.

### 2.2. Informe al auditor externo y dictamen recibido

Informe enviado con HD-4/HD-5/HD-6 (debilidades de diseno del CI) +
P-A/P-B/P-C/P-D (decisiones de metodo). Dictamen del auditor:
aprobados los 7 frentes con condiciones, orden de implementacion
estricto, criterio de calidad triple (estado inequivoco + evidencia
verificable + test de regresion). Dictamen integro en el hilo de
conversacion del 16-oct; no se transcribe aqui.

### 2.3. Fases 1 y 2 del dictamen (implementadas)

**Fase 1 — HD-5: separar Schedule vs Dispatch en health_check.**
Commit `3e8d2828`. `check_workflows` refactorizado en
`_check_one_workflow` (pura) + helpers `_last_run` y `_delta_days`.
Nuevo comportamiento: si solo hay `workflow_dispatch` reciente y el
schedule no esta acreditado, WARN con `SCHEDULE no acreditado; ultimo
workflow_dispatch`. Si gh no responde, SKIP (no verificable). 8 tests
en `tests/test_health_check_workflow_schedule.py`.

**Fase 2 — HD-4: receipt con evidencia del Guard.**
Commit `433da326`. `run.py` escribe `written_at` (UTC ISO) al VG
result. `write_completion_receipt.py` valida estrictamente antes de
emitir: `passed is True`, `errors_n == 0`, `checks_n >= 10`,
`written_at` presente/no futuro/<= 6h. Si falla -> no se emite
receipt (fail-closed, exit 1). Anadidos campos informativos:
`success_path_reached: true`, `validation_gate_checks_n`,
`validation_gate_errors_n`. Compatible con schema v1 (no rompe
receipts ya emitidos). 9 tests en
`tests/test_write_completion_receipt.py`.

### 2.4. Commits de la sesion

  433da326 fix(receipt): HD-4 - validar VG result antes de emitir receipt
  3e8d2828 fix(health_check): HD-5 - separar Schedule vs Dispatch
  083cfd39 docs(corpus): precisar MTE bonos secundarios y fallback SPY->^GSPC
  2dda7986 docs(ci)+test: auditoria workflow daily_run (HD-1..HD-8)
  8269d671 fix(tests)+docs(handoff): residuo pyflakes H7 + cierre deuda HANDOFF
  fd42010d docs(corpus): MTE + rs_abs_20d en 02_ARQUITECTURA §6
  c8a16f06 docs(corpus): HD-6 - poda bitacora + script poda_bitacora.py

---

## 3. FRENTES CERRADOS EN LA SESION

- **Auditoria workflow daily_run.yml**: HD-1 a HD-8 cerrados
  (comentarios, drift test, docstrings).
- **Residuo pyflakes H7**: eliminado. `00_ARRANQUE` y HANDOFF dejan
  de declarar "pyflakes silencio" sin base.
- **HD-5** (dictamen Fase 1): separacion schedule/dispatch implementada.
- **HD-4** (dictamen Fase 2): receipt con evidencia del Guard.
- **HD-6** (poda bitacora): 05_BITACORA 75 KB -> 18 KB; script
  `poda_bitacora.py` creado.
- **Precision corpus**: `02_ARQUITECTURA §6` corregido (bonos MTE
  secundarios, fallback SPY->^GSPC).
- **HANDOFF §4.1**: los 15 items de deriva aparecen como RESUELTOS por
  Fase B (fix del 15-oct que el HANDOFF no reflejaba).

---

## 4. PENDIENTES VIVOS

### 4.1. Fases del dictamen externo pendientes

Orden estricto del auditor. Una fase por sesion.

- **Fase 3** — HD-6: convertir pyflakes en gate real de CI (3 etapas:
  definir alcance, bloquear regresiones, gate permanente).
- **Fase 4** — P-A: contrato de estados del HANDOFF + verificador que
  detecte reapertura de items `FIXED`/`WONT_FIX`.
- **Fase 5** — P-C: indice consolidado H1-H18 en
  `docs/auditoria/radar/audits/` con estado, evidencia, commit, test.
- **Fase 6** — P-B: registro y test selectivo de anclas temporales
  (`# ANCHOR: reason=...` no vacia).
- **Fase 7** — P-D: modo `--check` en `poda_bitacora.py`, sin borrado
  automatico.

Cada fase debe cumplir: estado inequivoco + evidencia verificable +
test que detecte regresion.

### 4.2. Verificacion del cron

Ultimo run exitoso: cron de la madrugada del 16-oct
(`38834597307`, HEAD `c8a16f06`). Patron confirmado: 1 pipeline
completo + 4 skip por completion receipt. Siguiente slot: 23:17 UTC
del 16-oct.

**Atencion tras Fase 2:** el cambio en `write_completion_receipt.py`
hace fail-closed si el VG result no cumple el nuevo contrato. Verificar
el primer run post-commit que el receipt se sigue emitiendo.

### 4.3. Item menor GitHub

Aviso del runner: `ubuntu-latest` migra a Ubuntu 26 el 19-oct-2026.
No urgente, pero conviene tenerlo presente.

### 4.4. Issue #5 abierta (health-check)

`[health-check] WARN - discrepancias no criticas`. Abierta 18 dias.
WARN recurrentes: `update_sec_13f` cancelado + cron_slots 3/5.
Deuda de proceso, no bug. Decidir si forzar el workflow o ajustar
criterio.

---

## 5. COMO ARRANCAR

    Set-Location D:\Macro_Sectorial
    git log --oneline -5
    git status -sb
    py scripts\generate_estado_sistema.py
    py -m pytest tests/ -q -n 12 --tb=line -m "not integration_real_data"
    py -m pyflakes . ; py -m compileall . -q

Esperado:
- HEAD = el que devuelva `git log --oneline -1` (no se fija literal).
- ahead 0, behind 0 (o behind N si el bot ha pusheado; `git pull --ff-only`).
- working tree limpio (salvo data/ y outputs/ que el bot actualiza).
- ~3406 passed + 5 skipped (con filtro CI).
- pyflakes silencio, compileall OK.

---

## 6. NOTAS CRITICAS

- **H7 fue un bug real.** El patron rolling(N, min_periods=N) sobre
  series con NaN dispersos devuelve NaN en toda la serie. Test AST
  en tests/test_audit_contracts_rolling_nan.py lo prohibe. Si algun
  fix futuro necesita rolling sobre series con huecos, dropna previo.
- **H2 es falso positivo.** rs_mom (WLS) es invariante a escala.
  No modificar a la ligera hacia un ratio nominal.
- **No usar run.py en horario de mercado USA** (13:30-20:00 UTC).
  Contamina el snapshot.
- **Receipt fail-closed (Fase 2).** El receipt solo se emite si el
  VG result cumple el contrato estricto (passed=True, errors_n=0,
  checks_n>=10, written_at reciente). Si falla, no se emite.
  El siguiente slot hara READY y volvera a correr. Fail-safe.
- **Schedule vs Dispatch (Fase 1).** El health check distingue
  Schedule Health de Dispatch Health. Un dispatch reciente NUNCA
  debe ocultar que el schedule no dispara.
- **EOL del corpus:** LF.

---

## 7. REFERENCIAS

Corpus normativo en `Consolidacion_Documentos/` (12 docs).
Expedientes: docs/auditoria/wyckoff/, docs/auditoria/iae/golden/,
docs/auditoria/s03_deep/, docs/auditoria/radar/audits/.
Contrato del gate: docs/auditoria/daily_run_gate_contrato_v1.md.
Sandbox N8: outputs/audit/liq_backtest/.

---

**Fin del handoff.**
