# 05 - BITACORA

**Registro de sesiones recientes. NO normativo. NO se pega al arrancar.**
**Para sesiones anteriores, ver `git log`.**
**Se poda por antiguedad: mantener las ultimas 10 sesiones + las del dia en curso. Al anadir una nueva, la mas antigua fuera del dia se elimina o se resume en 04_HISTORICO.md. Ver seccion 5.**

---

## 1. PROPOSITO

Bitacora de las ultimas sesiones. Contiene:
- Que se hizo.
- Que quedo pendiente.
- Commits asociados.
- Proximo paso sugerido.

**No es 04_HISTORICO.md.** El historico es tematico y estable. La bitacora es cronologica y se poda.

---

## 2. FORMATO DE UNA ENTRADA

Cada sesion se registra asi:

    ### YYYY-MM-DD â€” Titulo corto

    **Objetivo.** Que se pretendia hacer.

    **Hecho.**
    - Bloque 1.
    - Bloque 2.

    **Commits.** <lista de SHAs> (o rango).

    **Pendiente.**
    - Item 1.
    - Item 2.

    **Proximo paso sugerido.** Que deberia hacerse a continuacion.

Al cerrar una sesion nueva, se anade arriba (las mas recientes primero). Si hay mas de 10 fuera del dia en curso, la mas antigua se elimina o se resume en el historico. Ver reglas completas en Â§5.

---

## 3. SESIONES

### 2026-10-15 - Auditoria externa H1-H18 del reporte

**Objetivo.** Atender los 18 hallazgos de la auditoria externa del
reporte del 15-oct. Tres niveles: critico / alto / medio.

**Hecho.**

- H1 (critico): alerta HYG/LQD dependiente del Z real.
- H2 (critico): FALSO POSITIVO. rs_mom (WLS) es invariante a escala
  (verificado con split 10:1 simulado). Solo afecta a LIS (N2).
- H3 (critico): separadas frescura SEC N-PORT (posiciones) y
  N-PORT-P QQQ (flujos).
- H7 (critico): BUG REAL. rolling(N, min_periods=N) roto con NaN
  dispersos -> 289/329 tickers con NH/NL=0. Fix en 5 sitios + AST.
- H4, H5, H6, H8, H10 (altos): documentacion + exposicion de valores.
- H9, H11-H18 (medios): documentacion, cuantificaciones, notas.
- 17 commits. Suite 3335 -> 3388 passed.

**Commits.** edabdb52, 804968b1, db0462c2, 4d905104, 3bd5f68a,
47b34d82, 3540218b, 0f990fdf, 7e621ea4, e9cefaeb, bb7fd03b,
16e064f4, fdbc7ccf, fbcfe4cd.

**Pendiente.**
- Corpus documental (~15 items de deriva).
- Verificacion del cron 23:17 UTC.

**Proximo paso sugerido.** Barrido de corpus en sesion propia. Los
15 items son deuda transversal que reaparecen en cada sesion.

### 2026-10-14 - N8 completo: forense + rediseno + sustitucion v1->v2

**Objetivo.** Cerrar el frente N8 (rediseno del score de liquidez)
tras varias rondas de auditoria externa. Sustituir v1 por v2 en
produccion.

**Hecho.**

- Forense del CRISIS 14-oct: state stale (retrocedio de 2026-10-13
  a 2026-10-07). Causa: run con datos FRED de 6 dias antes. No era
  artefacto del fix F5.6-06b como decia el HANDOFF anterior.
- Guard de monotonicidad en `liquidity_state.json` (4291593f).
- Trazabilidad de fechas + guard de frescura por serie (3d1a1fb6).
- Instrumentacion FRED en JSONL (456f3a28).
- Especificacion B3 congelada en `regimes/liquidity_v2.py` (c51372fb):
  senal A (SOFR-EFFR pb, umbrales 0/5/20), persistencia 3/3,
  alerta >=100pb, diagnostico C1, contexto calendario.
- C1-bis con episodios independientes: repo 2019 detectado, Q4-2025
  detectado (46/62), COVID y carry trade yen no detectados (correcto),
  0 falsas alarmas en 1024 dias.
- Sustitucion v1 -> v2 en pipeline (84f46dd3). Adaptador 4-tuple,
  consumidores sin cambios, rollback disponible.
- Verificacion en run.py local: exit=0, Gate 10/10, NEUTRAL (conf 90%).
- Cierre N1 (SLPM fail-closed) en 6 commits previos.
- Alineacion de comandos pytest documentados con el filtro CI.

**Commits (17).** 27c12671, 953b7fee, f49c85f0, 05b70686, 31f9c00a,
0fe4e3e2, 4291593f, 3d1a1fb6, 456f3a28, c51372fb, 786b3fc2,
84f46dd3, e19ba091, 5babc8e3, e8c26d4c.

**Pendiente.**
- Verificacion del cron 23:17 UTC con v2 en CI.
- Dictamen auditor sobre sustitucion (fase sombra saltada).
- Corpus documental (~15 items de deriva).
- 8 F6-* sin descripcion.

**Proximo paso sugerido.** Verificar el run del cron al dia siguiente.
Si v2 en CI va bien, el frente N8 queda cerrado tecnicamente.

### 2026-10-13 - Verificacion del contrato en prod + revision del reporte

**Objetivo.** Verificar el run de la madrugada (primer run con el
contrato de persistencia activo) + revision linea a linea del
reporte diario.

**Verificacion del contrato en prod (38342330518).** Pipeline
ejecuto completo (17m48s). `Verificacion de outputs: OK (19
checks)`. Target session 2026-10-12. Receipt emitido. Contrato
validado en produccion.

**Revision del reporte (10 hallazgos).**
- B1 (fix): FRED en `data/providers/fred.py` extendia el indice
  hasta hoy con ffill, contaminando la fecha del `liquidity_state.json`.
  Bug visible en el reporte (FRED decia '0 dias / CURRENT' cuando
  el dato real era 3 dias). Fix: `end=s.index[-1]` en vez de
  `end=today`.
- B2 (nota): 'Breadth de Mercado' cuenta ETFs sobre EMA; 'Sector
  Breadth & Health' cuenta componentes por sector. Distintas metricas.
- B3 (nota): 3 columnas de Matriz de Evidencia tienden a constante
  (Wyckoff, Credito, Volat). Explicito en el render.
- B4 (nota): z-score de flujos internacionales clipeado a [-5, +5].
- B6 (nota): lag de proveedores (BlackRock T+1, Amundi T+1/T+2).
- B8 (nota): `n_total` (universo completo ETF) vs `rs_internal`
  (top-20 por sector) miden cosas distintas. Documentado.
- B10 (nota): SSGA devuelve ETFs no sectoriales (FEZ). Aclarado.
- B5, B7, B9: descartados (falsos positivos verificados).

**Fix colateral (upstream).** `OutputSpec.date_source='upstream'`
para outputs con lag de proveedor (sector_flow_characteristics).
El verificador extendido ya no reporta gap cuando el CSV tiene la
ultima publicacion real aunque no coincida con la sesion.

**Verificacion.** Suite: 3251 -> 3253 passed. 5 skipped.
Pyflakes limpio.

**Commits.** 3542d29d, 89331321, 1ea34ac7, 43f10428, 8b441a7a.

**Pendiente.**
- Deudas bloqueadas por terceros/datos (C2, SOW v6, Wyckoff 5b.X).
- ~11 BAJA vivas de baja prioridad (IAE rendimiento, MTE limpieza).

**Proximo paso sugerido.** Monitorizar el proximo run del cron.
Confirmar que B1 mantiene FRED sin fecha sintetica.

### 2026-10-12 - Contrato en prod + triaje auditoria + higiene

**Objetivo.** Cerrar el frente del contrato (verify extendido),
triage completo de la auditoria externa 2026-09-26, y deudas de
higiene del corpus.

**Hecho.**
- `scripts/verify_outputs.py`: verificacion extendida de outputs
  (ultimas N sesiones). Gap real detectado:
  `sector_flow_characteristics` 09-oct.
- Triaje completo auditoria 2026-09-26:
  ALTA 6/6 cerradas. MEDIA 71/71 clasificadas (0 vivas).
  BAJA 53/61 triadas (8 F6-* sin informe original).
- Verificacion run domingo 11-oct: sin anomalia (gate skipeo
  por diseno).
- F2-02, F2-03, F2-04, F2-07, A5-57 anotados en `02_ARQUITECTURA`.
- Politica `outputs/audit/` en `01_METODO` 10.1.
- Poda `00_ARRANQUE` 11670 -> 10240 B (bajo umbral 11 KB).
- D-07 cerrado con notas cruzadas. D-04 sin parche (ya renombrado).

**Verificacion.** Suite: 3243 -> 3247 passed. 5 skipped. Pyflakes
limpio.

**Commits.** 026d182d, 45ec04de, d80cebf8, 0035be38, 06dd6cd6
  (+ f428bf51, 69506319 del bloque anterior).

**Pendiente.**
- Primer test del contrato en prod: cron 23:17 UTC hoy.
- 8 F6-* sin descripcion (informe original no disponible).
- ~11 BAJA vivas de baja prioridad.
- Regenerar `sector_flow_characteristics` 09-oct (ROI bajo).
- Cosmeticos: `render_calidad_datos` ubicacion, divergencia
  runtime/CSV, `test_render_freshness_d36` fragilidad.

**Proximo paso sugerido.** Verificar el primer run del cron con
el contrato activo. Si el verificador caza algo, investigar. Si
OK, el contrato queda validado en produccion.

### 2026-10-11 - Contrato de persistencia de outputs

**Objetivo.** Cerrar la clase entera de bugs "writer silencioso"
(el gap 09-oct fue el sintoma). No un parche por CSV.

**Hecho.**
- Reconocimiento del sistema (estructura, pipeline, 19 writers,
  contratos temporales).
- Deuda limpia: retirados 3 scripts one-shot, `sectorial.py.orig`,
  `outputs/audit` reducido (131 -> 66 MB).
- Diseno e implementacion del contrato de persistencia: registro +
  wrapper + verificador + integracion + AST test.
- 19 writers migrados en 6 commits por fichero.
- Test AST cazo un `to_csv` residual en `indices_intl.py`.

**Verificacion.** Suite: 3204 -> 3243 passed. 5 skipped. Pyflakes
limpio.

**Commits.** f0311cdb, 8596690c, c43a6b00, 2593ac91, ceecad70,
b1237083, ae752882, ab838dbf.

**Pendiente.**
- Push a origin.
- Politica de limpieza para `outputs/audit/pre_*` (apunte, sin
  codificar).

**Proximo paso sugerido.** Push. Cerrar sesion. La proxima sesion
arranca con el contrato activo: si un writer falla silenciosamente,
el run aborta con error explicito.

### 2026-10-10 - Retirada del modulo Wyckoff legacy (5e)

**Objetivo.** Cerrar el frente legacy: llevar a termino la retirada que el plan tenia bloqueada por ceremonia de fases.

**Hecho.**
- Portado el test de regresion I2 / K-INDEX-RANGE-01 a `test_wyckoff_v1_contract.py`.
- Borrados: `indicators/wyckoff.py`, `tests/test_build_ticker_df_i2.py`, `tests/test_index_phase_range.py`, `scripts/compare_wyckoff_legacy_v18.py`.
- Actualizados `00_ARRANQUE` y `02_ARQUITECTURA`.

**Verificacion.** Barrido de imports residuales: 0. Suite: 3204+5+0. CI con v1.8 confirmado 09-oct y 10-oct.

**Pendiente.** Verificar scheduled 10-oct 23:17 UTC (no debe notar el cambio: ya corria con v1.8).

**Proximo paso sugerido.** Ninguno urgente. H5.3 (20-nov-2026), C2 y SOW v6 bloqueados por terceros.

### 2026-10-09 (noche) - Regeneracion de CSVs contaminados por legacy Wyckoff

**Objetivo.** Tras detectar que el clasificador legacy seguia contaminando CSVs historicos, barrer todos los `outputs/history/*.csv` y regenerar los afectados.

**Detectado.** Columnas derivadas de wyckoff tenian dos metricas bajo el mismo nombre: legacy pre-08-oct, v1.8 post. Discontinuidad medida en XLK `wyckoff_median`: 07-oct=+0.0749 (legacy) -> 08-oct=+0.3776 (v1.8). Ratio 5.04.

**Regenerado.**
- sector_concentration.csv (80cb03f8). Ratio 07/08 post: 1.92. Continuo.
- leader_representativeness.csv (440750c1). 442 -> 115 filas, 10 -> 3 sectores.
- sector_leader_divergence.csv (440750c1). 82 -> 23 filas, 10 -> 3 sectores.

**Diagnosticado como limpio o no-bug.** rs_internal (RS, no wyckoff), sector_rank_history (no guarda fase), sector_breadth/_momentum/_persistence (breadth puro), sector_flow_characteristics (lag SSGA 1-2d), evidence_matrix y sector_regime_matrix (solo ultimo dia).

**Error reconocido.** Primera version del script `regenerate_leader_dfs.py` llamaba a `generate_leader_section` con `fase_dict={}`, dejando `leader_df` vacio. Resultado: CSVs con 5 y 1 filas. Restaurado desde snapshot. Script rehecho construyendo `fase_dict` desde `compute_sector_scores(df_market_cut)`, replicando el flujo real del pipeline.

**Commits.** 80cb03f8, 440750c1.

**Pendiente.** H5.3 (20-nov-2026). C2, SOW v6 (terceros). Wyckoff 5e (bloqueado por cadena 5c/5d).

**Proximo paso sugerido.** Verificar scheduled 2026-10-09 23:17 UTC con los nuevos CSVs.

### 2026-10-09 - Auditoria del reporte diario: 15 hallazgos clasificados

**Objetivo.** Revision bloque a bloque del reporte del 2026-10-09 buscando errores, incoherencias, ceros/NaN/N-D ocultos.

**Resultado.** 15 hallazgos. 2 corregidos, 13 no-bug / falsos positivos / deuda de diseno documentada.

**Corregidos.**
- H-07/H-08 (commit 35559c1d): notas de Representatividad del lider y Divergencia sector-lideres declaraban criterios falsos (cobertura / validos >=3). El filtro real es por fase Wyckoff (solo ACCUMULATION o MARKUP).
- H-09 (commit 4a604b21): `heterogeneity_type` se retiro del writer, del render y del test. Valor constante en el historico observado. Redundante con `dispersion_reading`.

**Falsos positivos / no-bug (verificados con codigo).**
- H-02: N/D en VIX3M/VIX se debe a falta de VIX, no de VIX3M. Fuente distinta.
- H-04: coberturas 18-25% son top-15/total-ETF. La metrica usa top-15 por diseno (`TOP_N_CANDIDATES`). Naming confuso.
- H-05: `signal_cons = (pers + agree) / 2`. 38% es matematica.
- H-11: Flujo Proxy en dos secciones mide cosas distintas (z-score vs flow_median). Naming confuso.
- H-12, H-13: cosméticos documentados.
- H-14: Q1 en N-PORT es calendario. Q2 llega en el cron del 20-oct.
- H-15: Indices en RANGE por umbrales (`struct > 0.30 AND t_norm > 0.30`). Ningun indice los supera hoy. No es fallback.
- H-03: `IPS = -0.39` e `IPI = 31` son la misma metrica en escalas [-1,+1] y [0,100]. No incoherencia.
- H-01, H-10: cosmeticos ya documentados.

**Deuda declarada.**
- H-06: `sector_flow_vs_price_div` huerfano en `compute_flow_divergence_v2`. Se documenta con nota en el render + entrada en 04_HISTORICO. WONT FIX hasta tener fuente.

**Commits.** 35559c1d, 4a604b21, este.

**Pendiente.** H5.3 (20-nov-2026). C2 y SOW v6 bloqueados por terceros. Wyckoff 5e tras CI estable.

**Proximo paso sugerido.** Ninguno urgente. Verificar H5.3 el 20-nov-2026.


### 2026-10-09 - Continuacion: verificacion E2E fix B + cierres documentales

**Objetivo.** Confirmar en produccion el fix B (`save_regime_history`) y cerrar los frentes abiertos del dia anterior.

**Hecho.**

- Fix B confirmado. El scheduled `37876944887` (2026-10-09 02:56 UTC, headSha `c9b600b1`) corrio `run.py` completo con el fix y produjo `macro_regime.csv` con fila `2026-10-08`, que es el `effective_date` correcto (ultima sesion USA cerrada a esa hora). Sin bug.
- Aclaracion de regla: el writer escribe con `effective_date`, no con `date.today()`. Nueva seccion en `01_METODO` seccion 4. Mi formulacion previa ("hoy no debe aparecer como fila") era incorrecta y genero una falsa alarma.
- Precisada la causa raiz del warm-up artefactual Wyckoff en `04_HISTORICO` (las dos ramas silenciosas del legacy: `len(close) < 200` y `score_clean.empty`).
- Corregida la cifra de la auditoria externa 2026-09-26: 205 -> 142 (6 ALTA / 71 MEDIA / 65 BAJA). Los informes existen en `bc656374`, borrados de main en `cdf47ad2`.
- Cerrado `sow-v4-validation` como WONT FIX razonado (88 ficheros unicos, 0 consumidores en main).
- Tests de funciones puras SEC/QQQ (`test_nport_sec_helpers.py`, 9 casos).
- Documentada la cadena de universos de tickers (`02_ARQUITECTURA` seccion 16).

**Commits.** c9b600b1, 55a45cf4, b5413e8b, 06bec8cc, 634d44d4, fd5f0e9a, 90127936, 815e113b, 6027a943, 0017bf4f, 8e8826f5, d495dc1e, c09a268c, 203b61c7, a356ae1d, mas el de esta sesion.

**Pendiente.**

- H5.3 (cron 13F, 20-nov-2026). Sin accion antes.
- C2 (920): bloqueado por auditor externo.
- SOW v6: pendiente firma externa.
- Wyckoff 5e (retirada legacy): tras 1 CI estable.
- `00_ARRANQUE` margen del umbral: 248 B. Sin urgencia.

**Proximo paso sugerido.** Ninguno urgente. Verificar H5.3 el 20-nov-2026.


### 2026-10-08 (tarde/noche) - Verificacion completa del sistema + 4 fixes

**Objetivo.** Bajo correccion expresa del usuario: revisar el estado real del sistema (pipeline, artefactos, frescura, providers) en lugar de seguir en higiene documental.

**Hecho.**

- Revertido un `run.py` manual ejecutado fuera de ventana CI (18:30 CEST). Sin contaminacion de fecha efectiva (`2026-10-07` en todos los CSV) pero con revision de fuente detectada en `evidence_matrix.csv` (XLRE/XLU).
- Frente workflows cron trimestrales: cerrado como fix-ya-aplicado. Los 3 schedule del 1-oct fallaron por `sys.path` + arrays desalineados; corregidos el mismo dia en 1bead8ad y 27248d35.
- Fix A: `check_iae_section` (STALE del N-PORT confundido con el de IAE). Commit c09a268c.
- Fix B: `save_regime_history` (fila huerfana cuando FRED publica fecha natural + effective es sesion anterior). Commit d495dc1e.
- Fix C: `check_workflows` (WARN permanente + cancelled mal clasificado). Commit 8e8826f5.
- Fix D: vigilancia de `etf_holdings.csv` e `index_holdings.csv`. Commit 0017bf4f.
- Corpus: cierre de 4 pendientes documentales (deuda huerfana auditoria externa, archivo gold-standard-v4, excepcion workflows datos externos, HEAD autorreferente del HANDOFF). Commit 203b61c7.
- Port de la evidencia IAE completa (53 ficheros) desde tag archivado. Commit a356ae1d.

**Commits.**

- a356ae1d docs(iae): portar evidencia IAE completa desde tag archivado
- 203b61c7 docs(corpus): cierre de pendientes documentales 2026-10-08
- c09a268c fix(health_check): aislar STALE de la seccion IAE
- d495dc1e fix(finalize): save_regime_history prioriza effective_date
- 8e8826f5 fix(health_check): check_workflows mira ultimo run y trata cancelled como WARN
- 0017bf4f fix(health_check): vigilancia de etf_holdings.csv e index_holdings.csv
- 6027a943 chore(test): eliminar import pytest sin uso en test_health_check_holdings
- (este commit) docs(corpus): cierre de la sesion de verificacion

**Pendiente.**

- E2E del fix B (finalize) en el scheduled 2026-10-08 23:17 UTC o proximo scheduled.
- Wyckoff 5e (retirada legacy) tras 1 run CI estable.
- H5.3 (cron trimestral 20-nov-2026 06:17 UTC). Sin accion antes.
- C2 (920): bloqueado por auditor externo.
- SOW v6: pendiente firma externa.
- `00_ARRANQUE` a 45 B del umbral 11 KB.

**Proximo paso sugerido.** Confirmar scheduled 23:17 UTC (E2E fix B). Valorar ROI de los 2 workflows de outputs derivados sin tests (`update_qqq_sec_flow`, `update_sec_nport`) declarados en 02_ARQUITECTURA. Cerrar bitacora si supera 10 entradas fuera del dia.

## 4. REFERENCIA A SESIONES ANTERIORES

Para sesiones anteriores al 2026-09-24, ver `git log --oneline`. Resumen tematico en `04_HISTORICO.md`.

---

## 5. REGLAS DE PODA

- La bitacora mantiene las ultimas **10 sesiones**. Ademas, siempre
  conserva las del dia en curso, aunque superen 10 (una jornada larga
  puede generar 6-7 entradas, y perder la de la manana mientras se
  trabaja la de la noche no aporta).
- Al anadir una nueva, si hay >10 fuera del dia en curso, la mas antigua
  se elimina. Nota: la regla era 'ultimas 5' hasta 2026-09-30. Nunca se
  aplico (12 entradas acumuladas). Se cambia al umbral real observado
  en lugar de fingir que se poda.
- **Antes de eliminar**, verificar que sus decisiones clave estan en `04_HISTORICO.md`. Si no, resumir primero.

---

**Fin de la bitacora.**