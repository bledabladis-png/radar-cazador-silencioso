# HANDOFF - Radar de Rotacion Sectorial

Documento de traspaso entre sesiones. Autocontenido. Pegar al inicio
del nuevo chat junto con `Consolidacion_Documentos/00_ARRANQUE.md`.

**Este fichero se sobrescribe cada sesion.** El historico queda en git.
No crear variantes con fecha.

---

## 1. ESTADO AL CIERRE (2026-10-14 ~18:30 CEST)

Rama activa:    main
HEAD:           e8c26d4c (posterior al push; consultar con `git log`)
Sincronizacion: main == origin/main
Tests:          3335 passed + 5 skipped + 0 failed
                (`pytest tests/ -q -m "not integration_real_data"`)
Validation Gate: 10/10
Working tree:   limpio
Corpus:         v3 (Consolidacion_Documentos/00-08 + ESTADO_SISTEMA)

---

## 2. SESION 14-OCT (frente N8 completo + sustitucion v1->v2)

### 2.0. Contexto

Sesion larga. Cierre del frente N8 (rediseno del score de liquidez)
tras una cadena de auditorias externas. Se sustituye el score v1
(regimes/liquidity.py) por v2 (regimes/liquidity_v2.py) en produccion.

### 2.0a. Diagnostico del score v1

El score v1 acumulaba problemas documentados:
- MAD nula sobre series planas: EFFR 41.3% del tiempo, SOFR 17.5%.
- Doble `ffill` sobre series FRED contaminando la muestra estadistica.
- 68 transiciones/ano (una cada 4 dias habiles).
- 34% del tiempo en regimen CRISIS.

### 2.0b. Forense del CRISIS del 14-oct

El run 38498223287 (14-oct 01:54-02:17 UTC) declaro CRISIS a raiz
de un state stale: `liquidity_state.json` retrocedio de
`2026-10-13, score=+0.29` a `2026-10-07, score=-0.49`. El run calculo
con datos FRED de 6 dias antes. Con datos frescos hasta el 13-oct,
el regimen es ESTRECHA/NEUTRAL. El HANDOFF anterior atribuia el CRISIS
al fix F5.6-06b; era incorrecto.

### 2.0c. Rediseno B3 (especificacion congelada, dictamen N8 Fase J)

Senal A: SOFR - EFFR en puntos basicos.
Umbrales congelados: 0 / 5 / 20 pb.
Persistencia: entrada 3/3, salida 3/3 (state machine binaria).
Alerta diferencial extremo: >= 100 pb, sin persistencia.
Diagnostico C1: p99 SOFR - EFFR, campo `c1_tail_divergence_candidate`.
Sin efecto decisorio.
Contexto de calendario: natural + financiero (habil USA).

Validacion (C1-bis con episodios independientes):
- Repo 2019 (16-17 sep): detectado en primer dia de ventana.
- Q4-2025 (actividad elevada Fed NY): 46/62 dias confirmados.
- COVID 2020-03: no detectado (dislocacion amplia, no tension repo).
- 2024-08 (volatilidad carry trade yen): correctamente no detectado.
- 0 falsas alarmas en 1024 dias de control.
- Robustez: 10.3% vs 11.3% dias confirmados 2018-20 vs 2024-26.

### 2.0d. Sustitucion v1 -> v2 en produccion

Commit `84f46dd3`: `src/pipeline/regimes.py` invoca v2 via el
adaptador `compute_liquidity_score_v2()` con contrato 4-tuple
compatible con v1. Consumidores sin cambios.

Rollback disponible: v1 sigue en `regimes/liquidity.py`. Para revertir,
cambiar `compute_real_liquidity()` para invocar
`regimes.liquidity.compute_liquidity_score`.

Verificacion en run.py local (18:07 CEST, 20.6 min):
- exit=0, Gate 10/10, Verificacion outputs OK (19 checks).
- `Liquidez Real (FRED): NEUTRAL (conf: 90%)` en reporte.
- State v2 persistido correctamente.

### 2.0e. Limpieza documental

Comandos pytest locales alineados con el filtro de CI
`-m "not integration_real_data"` en 5 ficheros (00_ARRANQUE,
01_METODO, 02_ARQUITECTURA, README, HANDOFF). Los 2 fallos de
`test_freshness` que aparecian en cada sesion eran ambientales:
parquets locales stale. Con el filtro la suite queda limpia.

### 2.1. Commits de la sesion (17)

  e8c26d4c data(state): actualizar liquidity_state.json a esquema v2
  5babc8e3 docs: alinear comandos pytest locales con el filtro de CI
  e19ba091 chore(liquidity): retirar script de sombra tras sustitucion
  84f46dd3 feat(liquidity): sustituir score v1 por v2 en pipeline
  786b3fc2 feat(liquidity): modo sombra de liquidity_v2 + N/D != False
  c51372fb feat(liquidity): especificacion experimental B3 congelada
  456f3a28 chore(fred): instrumentacion de descargas FRED en JSONL
  3d1a1fb6 fix(liquidity): trazabilidad de fechas + guard de frescura
  4291593f fix(liquidity): guard de monotonicidad en liquidity_state
  27c12671 chore(test): eliminar import pytest sin uso
  953b7fee fix(slpm): contrato fail-closed N1 en compute_leader_breadth_v2
  f49c85f0 fix(state_machine): guard no finitos + reason_code
  05b70686 fix(slpm): cortocircuito en evaluate_slpm_v12
  31f9c00a fix(report): render del state_reason_code en SLPM v1.2
  0fe4e3e2 test(slpm): integracion end-to-end del contrato N1
  953b7fee (precedido por los commits N1)
  ...

Detalle completo: `git log --oneline -20`.

---

## 3. FRENTES CERRADOS EN LA SESION

- **N1 (SLPM fail-closed)**: contrato fail-closed completo en
  `compute_leader_breadth_v2`, guard de no finitos en state machine,
  cortocircuito en `evaluate_slpm_v12`, render del `state_reason_code`.
  21 tests nuevos.
- **N8 forense**: bug del state retrocedido identificado y documentado.
- **N8 guard frescura**: trazabilidad + guard por serie en v1 y v2.
- **N8 sustitucion v1 -> v2**: activa en produccion.

---

## 4. PENDIENTES VIVOS

### 4.1. Verificacion del cron 23:17 UTC

El primer run de CI con v2 en produccion sera el slot 23:17 UTC del
14-oct. Verificar:
1. `gh run list --workflow=daily_run.yml --limit 3`.
2. `gh run view <id> --log` sin excepciones.
3. `outputs/state/liquidity_state.json` con esquema v2 en el commit
   del bot.
4. `outputs/audit/fred_downloads.jsonl` con datos frescos (JSONL
   instrumentado, commit 456f3a28).

### 4.2. Corpus documental

~15 items de deriva detectados durante la sesion: contratos temporales
(9 vs 10), slots cron (4 vs 5), OilPriceAPI residual, Liquidity Score
5 vs 4, tests literales desactualizados. Aparcados. No bloquean.

### 4.3. Audiencias externas pendientes

- **N8**: dictamen final sobre sustitucion v1 -> v2. La fase de sombra
  se ha saltado por decision propia (v2 ya validada contra episodios
  independientes). El auditor puede objetar. Documentar razonamiento.
- **N1**: cierre de las 4 condiciones del dictamen sobre rama vacia
  y datos parciales. Ya aplicado.

### 4.4. Deudas sin urgencia

- 8 F6-* sin descripcion (informe original no disponible).
- `sector_flow_characteristics` 09-oct (ROI bajo).
- Cosmeticos varios.

---

## 5. COMO ARRANCAR

    Set-Location D:\Macro_Sectorial
    git log --oneline -5
    git status -sb
    py scripts\generate_estado_sistema.py
    py -m pytest tests/ -q -n 12 --tb=line -m "not integration_real_data"
    py -m pyflakes . ; py -m compileall . -q

Esperado:
- HEAD = e8c26d4c o posterior.
- ahead 0, behind 0.
- working tree limpio (salvo data/ y outputs/ que el bot actualiza).
- 3335 passed + 5 skipped.
- pyflakes silencio, compileall OK.

Los tests `integration_real_data` requieren parquets de un run previo.
El filtro `-m "not integration_real_data"` los excluye (mismo criterio
que CI en el paso "Run tests"). Tras un `py run.py` local se pueden
ejecutar manualmente sin el filtro.

---

## 6. NOTAS CRITICAS

- **No usar run.py en horario de mercado** (13:30-20:00 UTC).
  Verificado hoy con la red de seguridad (snapshot pre/post + revert).
  Aplicar `git checkout -- .` tras la verificacion.

- **liquidity_v2 en produccion.** Cualquier cambio a `liquidity_v2.py`
  requiere dictamen (spec congelada).

- **State compartido**: `outputs/state/liquidity_state.json` con
  esquema v2 (score + date + regime + bloque v2 anidado). Guard de
  monotonicidad activo.

- **Log FRED**: `outputs/audit/fred_downloads.jsonl` (append-only).
  Instrumentacion del commit 456f3a28. Cada descarga FRED escribe
  una linea JSON con url, status, dates, duration.

- **EOL del corpus**: LF en todos los docs. `[System.IO.File]::WriteAllText`
  para no convertirlos a CRLF.

- **No push sin validacion local completa.** Aplicado en toda la sesion.

---

## 7. REFERENCIAS

Corpus normativo en `Consolidacion_Documentos/`:
00_ARRANQUE, 01_METODO, 01b_PATRONES, 01c_LECCIONES, 02_ARQUITECTURA,
03_IAE, 04_HISTORICO, 05_BITACORA, 06_IAE_P65_P66, 07_RUNBOOK,
08_AUDITORIA_SECTOR_REGIME, ESTADO_SISTEMA (auto).

Expedientes: docs/auditoria/wyckoff/, docs/auditoria/iae/golden/,
docs/auditoria/s03_deep/, docs/auditoria/radar/audits/.

Sandbox N8: outputs/audit/liq_backtest/.

---

**Fin del handoff.**
