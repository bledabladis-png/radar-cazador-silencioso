# HANDOFF - Radar de Rotacion Sectorial

Documento de traspaso entre sesiones. Autocontenido. Pegar al inicio
del nuevo chat junto con `Consolidacion_Documentos/00_ARRANQUE.md`.

**Este fichero se sobrescribe cada sesion.** El historico queda en git.
No crear variantes con fecha.

---

## 1. ESTADO AL CIERRE (2026-10-15 ~19:00 CEST)

Rama activa:    main
HEAD:           consultar `git log --oneline -1` (no literal; el corpus no fija SHA para evitar deriva)
Sincronizacion: main == origin/main
Tests:          ver `Consolidacion_Documentos/ESTADO_SISTEMA.md` (fuente autoritativa).
                Referencia orientativa: ~3390 passed + 5 skipped.
Validation Gate: 10/10
Working tree:   limpio
Corpus:         v3 (Consolidacion_Documentos/00-08 + ESTADO_SISTEMA)

---

## 2. SESION 15-OCT (auditoria externa H1-H18)

### 2.0. Contexto

Auditoria externa del reporte del 15-oct. 18 hallazgos clasificados
por el auditor en tres niveles (critico / alto / medio). Todos
atendidos. 17 commits de fix/docs + 3 commits previos de la sesion
14-oct (N8). 1 falso positivo verificado (H2), 1 bug real (H7).

### 2.1. Criticos cerrados

- **H1 (alerta HYG/LQD)**: el texto era literal fijo. Ahora depende
  del Z-score real de HYG/LQD. 3 ramas: positivo / negativo / N/D.
  Commit edabdb52.
- **H2 (RS nominal en WLS)**: FALSO POSITIVO. Verificado empiricamente
  que rs_mom = log(rs).diff(20) es invariante a escala (split 10:1 da
  resultado identico). El problema del rs nominal documentado como N2
  afecta SOLO a LIS, no a WLS. Commit 804968b1 + 4 tests.
- **H3 (frescura SEC)**: la tabla declaraba una sola fuente SEC (N-PORT
  posiciones) cuando el pipeline consume DOS (posiciones + N-PORT-P QQQ
  flujos). Separadas. Commit db0462c2.
- **H7 (NH/NL)**: BUG REAL. rolling(N, min_periods=N) sobre series con
  NaN dispersos devuelve NaN en TODA la serie. 289/329 tickers
  producian 0/0 NH/NL en el historico. Fix: rolling sobre serie
  dropna, cada ticker evaluado independientemente. 5 sitios:
  breadth_core, sector_breadth, stock_leader, breadth_equity,
  wyckoff_v1 + rolling_percentile en utils. Test AST de regresion.
  Commit 1189b7f8.

### 2.2. Altos cerrados

- **H4**: "Volatilidad LOW" (velocidad del VIX) ≠ "Lectura volatilidad
  media" (nivel). Nota en 2 sitios. Commit 3bd5f68a.
- **H5**: `Data Conf` se deriva SOLO de frescura (no cobertura).
  Publicados los umbrales FRESHNESS_*. Commit 47b34d82.
- **H6 + H10**: XLP con flujo 622.85M verificado real (7.4M shares x
  84.17 NAV). El Z=5.00 es el clip contractual. Fix: expuesto el z
  bruto sin clip con asterisco. Commit 4d905104.
- **H8**: "Sector Lider" -> "Candidato a lider (no confirmado)" cuando
  `state=UNRESOLVED` por falta de datos (NO_LEADERS, INPUT_UNAVAILABLE,
  INSUFFICIENT_*). Commit 0710db68.

### 2.3. Medios cerrados

- **H9**: correlacion entre evidencias cuantificada (24 filas, Pearson:
  solo price-breadth r=0.64; resto |r|<0.3). El Flow Proxy NO
  correlaciona con Precio. Commit 3540218b.
- **H11**: cubierto por H3.
- **H12**: ETF Primary Flow = estimacion operativa, no flujo del
  mercado secundario. Commit 0f990fdf.
- **H13**: benchmark SPY + formula de rs_abs_20d publicados. Commit
  7e621ea4.
- **H14**: NIPC 13F: unidad (acciones), formula, tratamiento NEW/EXIT,
  deduplicacion, significado de COMPLETE. Commit e9cefaeb.
- **H15**: pesos SCENARIO_WEIGHTS, umbrales y condiciones del MTE
  publicados. Commit bb7fd03b.
- **H16**: Momentum amplitud: unidad (pp) + reglas de Expansion/
  Deterioro. Commit 16e064f4.
- **H17**: desambiguadas las 3 "persistencias" del reporte (Structural
  Ranking, lideres por ticker, flujo primario). Commit fdbc7ccf.
- **H18**: "Desde inicio" QQQ/SPY: fechas reales (1999-03-10 / 1993-
  01-22) y advertencia de no comparabilidad. Commit fbcfe4cd.

### 2.4. Commits de la sesion (17)

  edabdb52 fix(alerts): H1
  804968b1 docs(H2): WLS usa rs_mom invariante a escala
  db0462c2 fix(H3): separar frescura SEC N-PORT y N-PORT-P QQQ
  4d905104 fix(H6,H10): exponer z-score bruto sin clip
  3bd5f68a docs(H4): velocidad vs nivel de volatilidad
  47b34d82 fix(H5): Data Conf solo de frescura + umbrales
  3540218b docs(H9): cuantificar correlacion evidencias
  0f990fdf docs(H12): ETF Primary Flow estimacion operativa
  7e621ea4 docs(H13): benchmark SPY + formula rs_abs_20d
  e9cefaeb docs(H14): NIPC unidad, formula y reglas
  bb7fd03b docs(H15): pesos y umbrales MTE
  16e064f4 docs(H16): unidad pp + reglas flags
  fdbc7ccf docs(H17): desambiguar persistencias
  fbcfe4cd docs(H18): QQQ/SPY Desde inicio
  9b438549 docs(corpus): cierre de sesion 15-oct (HANDOFF + bitacora)
  07e75fb0 docs(corpus): barrido de deriva documental (Fase B)

---

## 3. FRENTES CERRADOS EN LA SESION

- **Auditoria externa H1-H18** completa.
- **N8** (liquidity_v2) ya en produccion desde el 14-oct.
- **N1** (SLPM fail-closed) cerrado el 14-oct.
- **Pyflakes residual** (`import pytest` sin uso en
  `tests/test_audit_contracts_rolling_nan.py`, introducido
  por H7) corregido. HANDOFF §5 vuelve a poder declarar
  pyflakes limpio.

---

## 4. PENDIENTES VIVOS

### 4.1. Corpus documental — RESUELTO

Los 15 items de deriva detectados en la sesion 14-oct quedaron
resueltos en el barrido Fase B (commit 07e75fb0, 15-oct):
02_ARQUITECTURA §2/§4 (contratos 10->9), §8/§10 (slots 4->5),
§10/§12 (OilPriceAPI), §6 (Liquidity 5->4); 00_ARRANQUE §5/§3
(literales de tests -> remision a ESTADO_SISTEMA); 01_METODO §12
(referencia rota). Los literales de tests en 04_HISTORICO,
05_BITACORA y 08_AUDITORIA_SECTOR_REGIME se mantienen: son
cronologia, no deriva.

Los 8 F6-* figuran como WONT FIX razonado desde 2026-10-12
(TRIAGE_BAJAS_2026-10-12.md seccion 4). No son pendientes.

### 4.2. Verificacion del cron 23:17 UTC

Sin cambios respecto a la sesion anterior. El cron del 14-oct ejecuto
OK (run 38665838881). Siguiente 23:17 UTC.

### 4.3. Audiencias externas

Nada pendiente. La sesion 15-oct responde a la auditoria externa del
15-oct. Queda a discrecion enviar el anexo si el usuario quiere.

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
- ahead 0, behind 0.
- working tree limpio (salvo data/ y outputs/ que el bot actualiza).
- 3388 passed + 5 skipped (con filtro CI).
- pyflakes silencio, compileall OK.

---

## 6. NOTAS CRITICAS

- **H7 fue un bug real.** El patron rolling(N, min_periods=N) sobre
  series con NaN dispersos devuelve NaN en toda la serie. Test AST
  en tests/test_audit_contracts_rolling_nan.py lo prohibe. Si algun
  fix futuro necesita rolling sobre series con huecos, dropna previo.
- **H2 es falso positivo.** rs_mom (WLS) es invariante a escala.
  No modificar a la ligera hacia un ratio nominal.
- **No usar run.py en horario de mercado USA.**
- **Corpus desactualizado:** 15 items de deriva pendientes.
- **EOL del corpus:** LF.

---

## 7. REFERENCIAS

Corpus normativo en `Consolidacion_Documentos/` (12 docs).
Expedientes: docs/auditoria/wyckoff/, docs/auditoria/iae/golden/,
docs/auditoria/s03_deep/, docs/auditoria/radar/audits/.

Sandbox N8: outputs/audit/liq_backtest/.

---

**Fin del handoff.**
