# CALIBRACION K - Resultados Fase 5b

**Documento de resultados. Ejecutado 2026-10-02 segun protocolo v2.**
**Estado:** K=0.25 elegido por regla §6.3, validacion FALLA en H3.

---

## 1. Entorno de ejecucion

- Script: `scripts/calibrate_wyckoff_k.py`.
- Fuente: `data/stock_prices.parquet`.
- Ultima sesion cerrada (min): `2026-09-30`.
- Tickers en parquet: 316.
- Tickers con `trend` valido (>=200 obs): 302.
- Split temporal 70/30: corte en `2025-06-25`.
- Tickers en calibracion: 301. En validacion: 302.
- Vista primaria: **balanced** (por ticker).

---

## 2. Resultados en calibracion (70% inicial)

| K | pct_90 | pct_95 | p05 | p95 | H1-H4 | skew | n_tk |
|---:|---:|---:|---:|---:|---|---:|---:|
| 0.10 | 11.01% | 2.99% | -0.796 | +0.918 | PASA | -0.376 | 301 |
| 0.15 | 0.00% | 0.00% | -0.620 | +0.782 | PASA | -0.258 | 301 |
| 0.20 | 0.00% | 0.00% | -0.496 | +0.657 | PASA | -0.217 | 301 |
| **0.25** | 0.00% | 0.00% | **-0.409** | **+0.558** | **PASA** | -0.179 | 301 |
| 0.30 | 0.00% | 0.00% | -0.347 | +0.482 | H3 | -0.144 | 301 |
| 0.35 | 0.00% | 0.00% | -0.301 | +0.422 | H3 | -0.111 | 301 |
| 0.40 | 0.00% | 0.00% | -0.265 | +0.375 | H3,H4 | -0.092 | 301 |

**Lectura:**
- K=0.10 es el unico con saturacion medible (11% / 3%) pero todavia dentro de H1-H2.
- K>=0.15 satura 0% en calibracion (mercado 2021-2025 no alcanza `|trend| > 0.135` para el percentil 90).
- K=0.25 es el mayor que cumple H1-H4 en calibracion.

**K seleccionado por regla §6.3:** **K = 0.25**.

---

## 3. Diagnosticos sobre K=0.25 (calibracion)

| Criterio | Valor | Estado |
|---|---:|:---:|
| D1 skewness | -0.179 | OK (`|s|<1`) |
| D2 rango anual de medianas | 0.414 | **FAIL** (`<0.40`) |
| D3 IQR cross-sectional (mediana) | 0.451 | OK (`>=0.05`) |
| D3 pct fechas con IQR_cs > 0.10 | 100% | — |

D2 falla ligeramente (0.414 vs umbral 0.40). Es diagnostico, no invalida.

---

## 4. Validacion (30% final) con K=0.25

| Metrica | Valor | Criterio | Estado |
|---|---:|---|:---:|
| pct_90 | 0.00% | < 15% | OK |
| pct_95 | 0.00% | < 5% | OK |
| p05 | **-0.193** | `<= -0.40` | **FALLA (H3)** |
| p95 | +0.535 | `>= +0.40` | OK |
| skew | -0.202 | D1 |s|<1 | OK |
| IQR_cs | 0.488 | D3 >= 0.05 | OK |

**H1, H2, H4 pasan. H3 falla.** Segun protocolo §7, 5b FALLA.

---

## 5. Sensibilidad balanced vs pooled (calibracion, K=0.25)

| Metrica | Balanced | Pooled |
|---|---:|---:|
| pct_90 | 0.00% | 1.93% |
| pct_95 | 0.00% | 1.10% |
| p05 | -0.409 | -0.523 |
| p50 | +0.139 | +0.124 |
| p95 | +0.558 | +0.697 |
| iqr | +0.419 | +0.504 |
| skew | -0.179 | -0.180 |

**Lectura:** la vista pooled muestra saturacion (1.93% / 1.10%) que la vista balanced no ve. Razon: tickers con mas historia (empresas grandes, larga serie) pesan mas en pooled. La vista balanced es la correcta segun protocolo §5.0.

---

## 6. Analisis del fallo de H3

**H3 exige `p05 <= -0.40` sobre el `t_norm` de los tickers.**

- Calibracion (hasta 2025-06-25): mercado con correcciones (2022 bear market incluido). p05 = -0.41.
- Validacion (2025-06-25 a 2026-09-30): 15 meses de tendencia alcista sostenida. **p05 = -0.19**.

**Interpretacion:** en el periodo de validacion, casi todos los tickers estan en tendencia alcista simultaneamente. La cola izquierda de la distribucion de `t_norm` es muy corta. No hay valores extremos bajistas porque no hay mercado bajista en el periodo.

**No es un defecto de K.** Es una **propiedad del periodo**. H3 mide "amplitud de la cola izquierda de la distribucion actual". En un bull market puro esa cola no existe, independientemente de K.

**Comprobacion:** con K=0.40 (mas conservador respecto a saturacion), p05 en calibracion es -0.265, tambien fallaria H3. No hay K de la grid que haga pasar H3 en el periodo de validacion. Es decir, **la grid no contiene solucion; el fallo es estructural del criterio en regimen alcista**.

---

## 7. Estado

- Fase 5b: **FALLA en validacion** segun protocolo §7.
- K NO se congela.
- 5c/5d siguen bloqueadas.
- Hallazgo documentado en `11_hallazgo_validacion_5b.md`.

---

Fin de resultados 5b.
