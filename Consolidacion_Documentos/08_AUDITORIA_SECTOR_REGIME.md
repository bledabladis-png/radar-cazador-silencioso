# 08 - AUDITORIA DEL SUBSISTEMA SECTOR_REGIME

**Expediente consolidado. Auditoria externa + fixes ejecutados.**
**Referencia:** HEAD `8dc7437` (2026-09-30 noche).

Este documento cubre `regimes/sector_regime.py::compute_sector_scores`:
estado tras auditoria, hallazgos H1-H10, decisiones D1-D4, los 6 commits
de fix, verificacion triple y end-to-end.

---

## 1. RESUMEN EJECUTIVO

El subsistema calcula el score de los 11 sectores USA a partir de 6
componentes. La aritmetica es determinista, reproducible y coherente.

La auditoria externa identifico una violacion material: cuando
`close_sector` es NaN, `trend` y `breadth` devolvian `-1.0` (senal
bajista maxima fabricada). Ademas, `last_scores` leia la posicion
fisica `.iloc[-1]`, y no habia umbral minimo de componentes validos.

Estado tras los fixes: 10 hallazgos cerrados o aceptados. Suite 2961
+ 2 skipped. Validation Gate 10/10.

---## 2. FORMULA VIGENTE
Etapa 1: 6 componentes en [-1, +1]
Etapa 2: score_series = sum(c_i * w_i) / sum(w_i) sobre no-NaN
Etapa 3: dispersion = std(ddof=0) sobre componentes no-NaN
penalty = clip(1 - 0.5 * dispersion, 0, 1)
Etapa 4: score = score_series * penalty (NaN si n_valid < 4)

**Componentes y pesos:**

| ID | Peso | Definicion | Rango |
|---|---:|---|---|
| rs_mom_20 | 0.25 | tanh_normalize(sharpe_20(rs_ret)) | (-1,+1) |
| rs_mom_50 | 0.15 | idem ventana 50 | (-1,+1) |
| rs_mom_126 | 0.10 | idem ventana 126 | (-1,+1) |
| trend | 0.15 | media (close > EMA20/50/100/200) | {-1,-0.5,0,0.5,1} |
| volatility_inv | 0.15 | -tanh_normalize(atr14) | (-1,+1) |
| breadth | 0.20 | (media(close > EMA20/50/200) - 0.5) * 2 | {-1,-1/3,1/3,1} |

**Configuracion:** `SECTOR_DISPERSION_PENALTY = 0.5`
(`config/weights.py`).

---
## 3. HALLAZGOS H1-H10

### H1 - Contrato n_valid < 4 sin definir (CERRADO)

Antes: score publicable para n_valid >= 2. La renormalizacion
convertia hasta 0.25 del peso original en el 100% del score. Sin
marca de degradacion.

Decision D2: umbral n_valid >= 4. Con 4 componentes, el peso minimo
representado es 0.10+0.15+0.15+0.15 = 0.55. Fix `e92f61c`.

### H2 - Semantica de dispersion no formalizada (ACEPTADO)

`std(ddof=0)` mide heterogeneidad estadistica, no contradiccion
direccional. Decision D3: mantener formula; documentar semantica.

### H3 - Semantica de volatility_inv no documentada (ACEPTADO)

Mide anomalia temporal del ATR propio (ventana 60). Decision D3:
mantener formula; documentar.

### H4 - Redundancia trend/breadth (ACEPTADO)

Comparten EMA20, EMA50, EMA200. Peso conjunto 0.35. Decision de
diseno documentada. No es bug.

### H5 - Test de min_periods estructural (ACEPTADO)

`test_sector_regime_min_periods.py` verifica presencia de literales.
No es funcional, pero cubre la regresion deseada (que los min_periods
no se relajen accidentalmente). Aceptado por diseno.
### H6 - Explicacion inicial de n_valid=2 incorrecta (CERRADO)

El primer informe atribuia las 1031 filas con n_valid=2 a
min_periods=200 de EMA200. Causa real verificada:

- 10 sectores regulares (53 filas c/u): warm-up de robust_zscore
  sobre rs_ret (arrastra NaN iniciales de ^GSPC.Close[0]).
- XLC (501 filas): instrumento inexistente antes de 2018-06-19.

Reconciliacion empirica de XLC:
- A-pre: 448 filas (close NaN, instrumento no existia).
- A-post: 2 filas (2018-07-04, 2018-09-03, gaps).
- B: 51 filas (warm-up con close disponible).

### H7 - Casos extremos de normalizacion (PARCIALMENTE AUDITADO)

MAD=0 en robust_zscore: z en {-5, 0, +5} segun posicion vs mediana.
rolling_std ~ 0: denominador (std + 1e-9), z explota y clip a
[-5, +5]. Controlado numericamente, semantica economica no
caracterizada.
### H8 - NaN de precio convertido en -1.0 (CERRADO)

Verificado: 1454 filas historicas con close=NaN. `trend_position` y
`breadth` devolvian `-1.0` exacto. Materialmente grave porque cuando
solo quedan esos dos componentes, el sistema producia
`score = -1` + `penalty = 1` (maxima conviccion bajista + ausencia
de descuento por dispersion), derivados exclusivamente de ausencia
de datos.

`missing observation != bearish observation`

Decision D1: propagar NaN desde la raiz.

Fix `ee24afc` (trend_position propaga NaN via `.mask(close.isna())`).
Fix `43a2fdb` (cierres excepcionales NYSE: 2018-12-05 Bush,
2025-01-09 Carter). Estos dos dias estaban causando filas vacias
que se leian como bursatiles.

### H9 - Historico XLC con mezcla H8 + H1 (CERRADO)

Las 501 filas de XLC pre-2018-09-03 combinan:
- A-pre (448): close NaN, instrumento no existia -> H8.
- A-post (2): 2018-07-04 y 2018-09-03, gaps de datos -> H8.
- B (51): close disponible, warm-up -> H1.

Fix H8 elimina la fabricacion. Fix H1 marca las 501 como N/D.

### H10 - Descripcion incorrecta de MAD=0 (CERRADO)

Corregida: MAD=0 + s=median -> z=0; MAD=0 + s != median -> clip a
+/-5.

---
## 4. DECISIONES DE DISENO

**D1 - Contrato de componente observable.** Propagar NaN desde la
raiz cuando `close_sector` es NaN. `trend_position` y `breadth` no
convierten ausencia de dato en senal numerica.

**D2 - Contrato de score publicable.** `n_valid < 4` -> score = NaN.
Con 4 componentes, peso minimo representado 0.55.

**D3 - Semanticas del modelo.** Sin cambios en formulas. Mantener:
- dispersion = std(ddof=0) (heterogeneidad estadistica).
- volatility_inv = -tanh_normalize(atr14) (anomalia temporal del ATR
  propio, ventana 60).
- trend/breadth con sus pesos actuales; documentar EMA compartidas.

**D4 - Tratamiento del historico.** Recalcular con el contrato nuevo
mediante ejecucion E2E. Evidencia pre/post con hash. Diff
identificable por H8 (close NaN) y H1 (n_valid<4).

**Alcance:** D1 resuelve integridad. D2 limita degradacion. Ninguna
cambia la logica financiera del score.

---
## 5. COMMITS DE FIX

| Commit | SHA | Hallazgo | Descripcion |
|---|---|---|---|
| 1 | `43a2fdb` | H8 | Cierres excepcionales NYSE (Bush, Carter) |
| 2 | `ee24afc` | H8 | trend_position propaga NaN donde close es NaN |
| 3 | `c98c22a` | R2 | effective_date resuelto antes de last_scores |
| 4 | `e92f61c` | H1 | Umbral n_valid >= 4 |
| 5 | `c798857` | - | append_dedup en 5 writers de historico de flujo |
| 6 | `7b0044d` | - | Normalizar Date en ssga antes de append_dedup |
| 7 | `8dc7437` | - | Regenerar outputs tras fixes |

**Verificacion triple en los 6 commits de codigo:** rojo sin fix ->
verde con fix -> rojo al revertir -> verde restaurado.

**Nota sobre commit 3 (R2):** `compute_sector_scores` usaba
`scores.iloc[-1]`. Si el df terminaba con un dia no bursatil USA
(festivo NYSE, cierre excepcional, fila multi-mercado donde solo
cotizaron IBEX/DAX/FX), `last_scores` leia el score fabricado.
Verificado con festivo 2025-09-01: last_scores devolvia XLC=-0.055
(fabricado) en vez de XLC=+0.325 (real del 2025-08-29).

Fix: resolver `effective_date` sobre el universo de los 11 sectores
con min_coverage=0.90 antes del calculo (patron FU-020 ya establecido
en leaders.py:39-42).
---

## 6. FIX D - BUG DESTAPADO POR EL E2E

El primer run end-to-end de validacion de los fixes H8/R2/H1 destapo
un bug preexistente no relacionado con el expediente:

**Sintoma:** `outputs/history/etf_primary_flow.csv` y
`amundi_lyxi_primary_flow.csv` perdieron la fila `2026-09-29` que ya
estaba en `origin/main`. El run local la borro.

**Causa:** 5 providers escribian su historico sobrescribiendo el CSV
con lo que devolvia el proveedor, sin leer el existente. Si el
proveedor o su cache devolvia menos filas (delay de publicacion,
cache stale, glitch), el CSV perodia esas filas de forma permanente.

**5 sitios:**

- `ssga_fund_data.py`: get_etf_primary_flow_data.
- `amundi_fund_data.py`: get_amundi_lyxi_primary_flow.
- `_blackrock_base.py`: base comun de DAX e ISF.L (2 providers).
- `qqq_nport_flow.py`: main().
- `cftc_data.py`: get_cftc_position_flow_data.

**Fix:** en cada uno, leer HISTORY si existe, `append_dedup(hist,
nuevo, key)`, escribir. Patron ya establecido en sectors_base.py,
flows_primary.py y los writers de outputs/history.

**Fix D2 (commit 7b0044d):** el primer fix D introdujo un bug
secundario en ssga. `append_dedup` normaliza solo columnas `date`
(minuscula). SSGA usa `Date` (mayuscula). El `drop_duplicates` no
detectaba coincidencias ('2026-09-25' vs '2026-09-25 00:00:00') y
duplicaba todo el historico (56093 -> 112173 filas). Detectado en el
segundo run E2E. Corregido normalizando `Date` a YYYY-MM-DD antes del
dedup. Test reforzado: ademas de no perder filas, verifica que no se
duplican.

**Leccion:** el run E2E detecta bugs que los tests unitarios no ven.
El test de Fix D original era ciego (verificaba "fila presente", no
"conteo correcto").
---

## 7. VERIFICACION END-TO-END

**Run 1 (12:16 min, 2026-09-30 tarde):** detecto Fix D. Outputs
revertidos.

**Run 2 (12:05 min, 2026-09-30 noche, tras Fix D2):** limpio.

| Metrica | Resultado |
|---|---|
| Validation Gate | 10/10 |
| NIPC | 8256882557 (par Q1-2026 -> Q2-2026) |
| effective_date | 2026-09-29 (coverage 100%) |
| Ranking publicado | XLK=0.719, XLV=0.339, XLC=-0.134 |
| Perdida de filas (6 CSV) | 0 |
| Duplicados (6 CSV) | 0 |
| Fila 2026-09-29 | Presente |
| Ranking igual pre/post | Si (fixes no alteran output actual) |
| Suite | 2961 passed + 2 skipped |

**Interpretacion:** los fixes H8/R2/H1 no alteran el output del dia
actual, como se diseño. Actuan sobre:
- H8: warm-up 2016 + XLC pre-2018 (fuera del CSV de rotacion,
  que retiene ~2 meses).
- R2: solo cuando el df termina en dia no bursatil (hoy no aplica).
- H1: XLC pre-2018 y warm-up inicial.

La cobertura de los fixes es por test unitario, no por E2E. El E2E
verifica que no rompen el output actual.

---

## 8. PENDIENTES

- **C2 (920):** discrepancia H1-B. Bloqueado por auditor externo.
- **H5.3:** verificacion cron trimestral noviembre 2026.
- **Regeneracion del historico de `sector_rank_history.csv`:**
  actualmente retiene ~2 meses. Si se quiere verificar H1/H8 sobre
  el historico completo (warm-up 2016, XLC pre-2018), requiere
  ampliar la ventana de retencion o consultar el calculo interno de
  `scores` (no el CSV de rotacion).

---

## 9. REFERENCIAS

- Expediente previo (informe + anexos + dictamenes): en el hilo de
  auditoria externa, 2026-09-30.
- Commits: `43a2fdb`, `ee24afc`, `c98c22a`, `e92f61c`, `c798857`,
  `7b0044d`, `8dc7437`.
- Tests: `test_trend_position_nan.py`,
  `test_sector_regime_effective_date.py`,
  `test_sector_regime_n_valid_threshold.py`,
  `test_market_calendar_exceptional.py`,
  `test_flows_primary_preserve_history.py`.

---

**Fin del expediente.**