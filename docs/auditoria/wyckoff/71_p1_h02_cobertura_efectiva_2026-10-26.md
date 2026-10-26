# Doc 71 - P1-H02: cobertura efectiva por grupo

**Fichero:** docs/auditoria/wyckoff/71_p1_h02_cobertura_efectiva_2026-10-26.md
**Estado:** para revision externa.
**Predecesor:** dictamen 2026-10-26, P1-H02 ("la cobertura efectiva todavia no esta demostrada").

---

## 0. Resumen ejecutivo

Cuantificada la cobertura efectiva por grupo (18 grupos: 11 sectores USA, 3 indices USA, 4 indices EU). Tres conclusiones:

1. **Universo real de tickers sin datos: 52** (no 8 como decia el handoff sesion 11). 9 grupos afectados. Mayor brecha: XLF (11 tickers, 9.05% peso), QQQ (6 tickers, 6.30%), XLV (6 tickers, 6.18%).
2. **Cobertura sobre seleccionado** (peso con datos / peso seleccionado): XLF 89.36%, QQQ 92.50%, XLV 92.73%, XLC 94.94%, SPY 95.58%. Resto >96% o 100%.
3. **Cobertura sobre grupo** (peso con datos / peso total del ETF): XLF 76.19%, QQQ 79.02%, XLV 79.08%, SPY 81.25%, XLK 81.85%, XLU 83.31%, XLY 84.62%, XLI 84.86%. EU: FEZ 100% del CSV pero solo 66.26% del ETF real; ISF.L 100% del CSV pero solo 60.80% del ETF real.

**`TOP20-FALTANTES-DATOS` subestima el universo ×6.5** (8 -> 52). Se actualiza en REGISTRO.

---

## 1. Definiciones formales

El dictamen (P1-H02) pide "definir formalmente la diferencia entre evaluable y publicado, y aportar la cobertura ponderada efectiva".

**Seleccionado**: los N tickers que devuelve `_select_target_holdings(target_weight=0.85, min_tickers=20, max_tickers=300)`.

**Con datos**: el ticker existe en `data/stock_prices.parquet` (parquet de produccion) con `n_bars >= MIN_BARS_THEORETICAL=200`. Es el filtro primario de `_evaluate_holding`: si no pasa este umbral, la fila no se emite.

**Evaluable**: fila en `compute_wyckoff_snapshot`. Requiere `classify_phase` exitoso (fase != INSUFFICIENT_DATA). **No calculado en este doc**: `classify_phase` cuelga con al menos un ticker (bug separado, ver seccion 5). Se reporta `con_datos` como proxy y se declara la limitacion en el CSV de evidencia (`evaluable_metodo`).

**Publicado**: fila en alguno de los 6 CSVs de `outputs/history/wyckoff_*.csv`. Filtro: `phase in ACC_PHASES | DIST_PHASES`. No mide cobertura, mide **regimen de fase direccional** en la fecha de observacion. Un sector puede tener 100% de cobertura efectiva y 5% de publicado si la mayoria de sus tickers estan en RANGE. Esta distincion corrige el error del doc 70-v2, que usaba "publicado" como si fuera cobertura.

**Cobertura ponderada**:
- `cov_vs_sel_pct = con_datos_peso / seleccionado_peso`. Mide cuanto del peso objetivo (85%) tiene datos.
- `cov_vs_grupo_pct = con_datos_peso / weight_total`. Mide cuanto del ETF real esta cubierto.

Ambas necesarias: `cov_vs_sel` mide fidelidad del selector, `cov_vs_grupo` mide cobertura real del ETF.

---

## 2. Tabla de cobertura efectiva (18 grupos)

| grupo | gt | estado | wt | sel_n | sel_peso | cd_n | cd_peso | cov_vs_sel | cov_vs_grupo |
|---|---|---|---|---|---|---|---|---|---|
| XLB | sector_usa | REACHED | 99.74 | 20 | 90.14 | 20 | 90.14 | 100.00 | 90.38 |
| XLC | sector_usa | REACHED | 99.79 | 20 | 97.66 | 19 | 92.71 | 94.94 | 92.91 |
| XLE | sector_usa | REACHED | 99.80 | 20 | 99.02 | 20 | 99.02 | 100.00 | 99.22 |
| XLF | sector_usa | REACHED | 99.82 | 34 | 85.11 | 23 | 76.05 | **89.36** | **76.19** |

> **Actualizacion 2026-10-26 (WYCKOFF-TICKER-NORMALIZATION):** la
> tabla anterior asumia `cd_n == ev_n`. Con el fix `4e94bf2e`, XLF
> recupera BRK.B (holdings usa `BRK.B`, parquet `BRK-B`). Pre-fix
> `ev_n=22`, `ev_peso=63.74`. Post-fix `ev_n=23`, `ev_peso=76.05`.
> Los valores de esta tabla son ya post-fix. En los otros 17 grupos
> `cd_n == ev_n` antes y despues del fix.
| XLI | sector_usa | REACHED | 99.85 | 45 | 85.66 | 44 | 84.73 | 98.92 | 84.86 |
| XLK | sector_usa | REACHED | 99.94 | 25 | 85.19 | 22 | 81.80 | 96.02 | 81.85 |
| XLP | sector_usa | REACHED | 99.62 | 20 | 86.75 | 20 | 86.75 | 100.00 | 87.08 |
| XLRE | sector_usa | REACHED | 99.56 | 20 | 86.66 | 20 | 86.66 | 100.00 | 87.05 |
| XLU | sector_usa | REACHED | 99.75 | 21 | 85.05 | 20 | 83.10 | 97.70 | 83.31 |
| XLV | sector_usa | REACHED | 99.74 | 26 | 85.06 | 20 | 78.87 | 92.73 | 79.08 |
| XLY | sector_usa | REACHED | 99.90 | 21 | 85.72 | 20 | 84.53 | 98.62 | 84.62 |
| DIA | index_usa | REACHED | 99.98 | 20 | 85.00 | 20 | 85.00 | 100.00 | 85.02 |
| QQQ | index_usa | REACHED | 98.30 | 41 | 83.98 | 35 | 77.67 | 92.50 | 79.02 |
| SPY | index_usa | REACHED | 99.97 | 167 | 84.98 | 145 | 81.22 | 95.58 | 81.25 |
| DAXEX | index_eu | REACHED | 78.43 | 20 | 78.43 | 20 | 78.43 | 100.00 | **100.00 del CSV** |
| FEZ | index_eu | REACHED | 66.26 | 20 | 66.26 | 20 | 66.26 | 100.00 | **100.00 del CSV** |
| ISF.L | index_eu | REACHED | 60.80 | 20 | 60.80 | 20 | 60.80 | 100.00 | **100.00 del CSV** |
| LYXI | index_eu | REACHED | 93.24 | 20 | 93.24 | 20 | 93.24 | 100.00 | **100.00 del CSV** |

Lectura EU: `cov_vs_grupo=100%` significa "100% del CSV de 20 filas", no "100% del ETF real". El `wt` real de FEZ es 66.26% (el CSV solo trae 66.26% de las posiciones del ETF). Por tanto, la cobertura efectiva de FEZ **sobre el ETF real** es 66.26%, no 100%.

**Nota adicional sobre EU**: DAXEX/FEZ/ISF.L/LYXI tienen `peso_total_csv < 100%` (78.43, 66.26, 60.80, 93.24). El selector cumple 85% del CSV pero no del ETF. Ya registrado en `TOP20-EU-HOLDINGS-INCOMPLETOS`.

---

## 3. Universo real de tickers sin datos (52, no 8)

El handoff sesion 11 y doc 70-v2 listaban 8 tickers (XLV: CI, MRNA, BSX, HCA, COR, CAH; XLU: CNP; XLY: NKE). El universo real, con `normalize_yahoo_ticker` aplicado, es **52 tickers en 9 grupos**.

| grupo | tickers | peso_faltante | sobre_grupo | sobre_seleccion |
|---|---|---|---|---|
| XLF | 11 | 9.05 | 9.1% | 10.6% |
| QQQ | 6 | 6.30 | 6.4% | 7.5% |
| XLV | 6 | 6.18 | 6.2% | 7.3% |
| XLC | 1 | 4.95 | 5.0% | 5.1% |
| SPY | 22 | 3.75 | 3.8% | 4.4% |
| XLK | 3 | 3.39 | 3.4% | 4.0% |
| XLU | 1 | 1.95 | 2.0% | 2.3% |
| XLY | 1 | 1.19 | 1.2% | 1.4% |
| XLI | 1 | 0.92 | 0.9% | 1.1% |

Detalle completo en `outputs/audit/p1_h02_tickers_sin_datos.csv` (gitignored, regenerable con `py scripts/audit/p1_h02_cobertura_efectiva.py`).

**Casos especiales**:
- **WBD (XLC, 4.95%)**: 1 barra en produccion. Preexistente (NaN masivo por IPO reciente). Sin cobertura efectiva.
- **SPCX (QQQ, 2.71%)**: 93 barras, por debajo del umbral 200. IPO 2025.
- **HONA (XLI, 0.92%)**: 92 barras. Mismo problema.
- **MRNA**: ausente en cualquier parquet (ya registrado en TOP20-FALTANTES-DATOS).

**Correccion respecto al handoff**: 7/8 tickers originales (BSX, CAH, CI, CNP, COR, HCA, NKE) siguen en `stock_prices_reserved_cohort.parquet` / `stock_prices_reserved_r3000.parquet` (holdout SOW). No se promueven. Ver `SOW-HOLDOUT-IMMUTABLE`. Los 44 restantes (QCOM, ADI, APH, ICE, etc.) no estan en ningun parquet conocido y requieren descarga nueva.

---

## 4. Implicacion para el dictamen

El dictamen P1-H02 pide "aportar la cobertura ponderada efectiva". Esta entregado:

- **Sectores USA**: 8 de 11 cumplen >95% sobre seleccion. XLF (89.36%), XLK (96.02%), XLV (92.73%) por debajo del 100%, pero el selector alcanzo 85% sobre el CSV. La brecha real es datos, no selector.
- **Indices USA**: DIA 100%, SPY 95.58%, QQQ 92.50% sobre seleccion. Bien.
- **Indices EU**: cobertura sobre CSV = 100%, pero cobertura sobre ETF real = 66-93%. El frente `TOP20-EU-HOLDINGS-INCOMPLETOS` es **bloqueante** para declarar cobertura EU.

**Sobre el objetivo del dictamen (85% cobertura)**: los 11 sectores USA cumplen 85% sobre el CSV, pero 4 de 11 (XLF, QQQ, XLV, SPY) no cumplen 85% sobre grupo. El dictamen es explicito: "El objetivo no debe validarse solo mediante recuentos de tickers". Aceptado: cobertura sobre seleccion > 85% no es suficiente para declarar A.2c cerrada; se necesita cobertura sobre grupo.

**Conclusion**: `TOP20-FALTANTES-DATOS` es la deuda principal del frente. Sin descargar los 44 tickers nuevos + los 7 reservados (que requieren decision de holdout), no se puede cumplir el objetivo del dictamen. Se actualiza el ticket con el universo real 52.

---

## 5. Limitaciones de esta auditoria

1. **~~`classify_phase` cuelga~~** — RESUELTO, falso positivo.
   `WYCKOFF-CLASSIFY-PHASE-HANG` cerrado como `FALSE_POSITIVE`
   (`842c9aaf`). Probe 360 tickers: 0 timeouts, 21.6s total. La
   limitacion real detectada fue distinta: **tickers con punto**
   (`BRK.B` en holdings vs `BRK-B` en parquet). `_evaluate_holding`
   no normalizaba y descartaba la fila silenciosamente. XLF perdia
   BRK.B (12.32% de peso). Fix `4e94bf2e` + test
   `test_dotted_ticker_normalized_before_parquet_lookup`. Solo 1
   ticker afectado en el universo actual. Este doc fue actualizado
   post-fix. Ticket abierto: `WYCKOFF-TICKER-NORMALIZATION`.

2. **`publicado` no es cobertura**. Refleja regimen de fase direccional. El doc 70-v2 lo trato como cobertura y genero confusion (FEZ 6 evaluables/16 publicados). Corregido aqui.

3. **Cross-check contra CSVs publicados**: no realizado en detalle. La tabla usa conteos del snapshot recalculado (via `_select_target_holdings`), no los CSVs reales. Si los CSVs tienen filas desactualizadas por el bug huerfanas previo a `f4528b49`, la comparacion seria erronea. Verificado `f4528b49` corregido y CSVs regenerados en `2026-10-26 17:35:52`, posterior al fix. Sin embargo, la trazabilidad snapshot-vs-CSV se verifica en P1-H03 (invariantes del backfill).

---

## 6. Lo que este documento NO hace

- No cierra `TOP20-FALTANTES-DATOS`. La expande de 8 a 52.
- No descarga tickers.
- No promueve tickers de `reserved_cohort` a produccion.
- No cierra `TOP20-EU-HOLDINGS-INCOMPLETOS`.
- No regenera el historico (1086 fechas). Pendiente commit aparte
  tras confirmar el fix con snapshot (528 filas).
- No verifica invariantes del backfill (eso es P1-H03).

---

## 7. Siguiente paso

`TOP20-FALTANTES-DATOS` actualizado con universo 52. Siguiente en orden del dictamen: P1-H04 (sesgo PIT visible en artefactos) o P1-H03 (invariantes del backfill). El auditor decide.

---

**Preparado 2026-10-26. Script reproducible:** `py scripts/audit/p1_h02_cobertura_efectiva.py`.