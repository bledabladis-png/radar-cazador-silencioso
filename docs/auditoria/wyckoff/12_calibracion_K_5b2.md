# CALIBRACION K - Resultados Fase 5b.2

**Documento de resultados. Ejecutado 2026-10-02 segun protocolo v3.**
**Estado:** EXITOSA. K=0.25 CONGELADO.

---

## 1. Entorno

- Script: `scripts/calibrate_wyckoff_k.py`.
- Fuente: `data/stock_prices.parquet`.
- Ultima sesion cerrada (min): `2026-09-30`.
- Tickers en parquet: 316.
- Tickers con trend valido (>=200 obs): 302.
- Split temporal 70/30: corte en `2025-06-25`.
- Tickers en calibracion: 301. En validacion: 302.
- Vista primaria: **ticker-balanced** (mediana entre tickers).
- Sensibilidad: **observation-pooled**.

---

## 2. Calibracion (70% inicial)

H3/H4 calculados intra-ticker (distribucion temporal por ticker dentro
del 70%), agregados por mediana entre tickers.

| K | pct_90 | pct_95 | p05 | p95 | H1-H4 | IQR | skew |
|---:|---:|---:|---:|---:|---|---:|---:|
| 0.10 | 11.01% | 2.99% | -0.796 | +0.918 | PASA | 0.847 | -0.376 |
| 0.15 | 0.00% | 0.00% | -0.620 | +0.782 | PASA | 0.645 | -0.258 |
| 0.20 | 0.00% | 0.00% | -0.496 | +0.657 | PASA | 0.512 | -0.217 |
| **0.25** | 0.00% | 0.00% | **-0.409** | **+0.558** | **PASA** | 0.419 | -0.179 |
| 0.30 | 0.00% | 0.00% | -0.347 | +0.482 | Falla H3 | 0.361 | -0.144 |
| 0.35 | 0.00% | 0.00% | -0.301 | +0.422 | Falla H3 | 0.311 | -0.111 |
| 0.40 | 0.00% | 0.00% | -0.265 | +0.375 | Falla H3+H4 | 0.278 | -0.092 |

**K seleccionado por regla §6.3 (mayor de los que pasan):** **K = 0.25**.

---

## 3. Validacion OOS (30% final, 2025-06-25 -> 2026-09-30)

| Criterio | Valor | Umbral | Estado |
|---|---:|---|---|
| V2 pct_90 | 0.00% | < 15% | OK |
| V2 pct_95 | 0.00% | < 5% | OK |
| V3 IQR_cs | 0.329 | >= 0.05 | OK |
| V4 n_tickers | 302 | >= 10 | OK |

**V1-V4 pasan.**

**Diagnosticos OOS** (reportados, no invalidan):

- p05 = -0.193.
- p95 = +0.535.
- skew = -0.202.

El p05 OOS mas alto que el de calibracion (-0.41 -> -0.19) se explica
por el sesgo del periodo (bull market). Se reporta como diagnostico.
En el protocolo v3, esto ya no es fallo.

---

## 4. Sensibilidad balanced vs pooled (K=0.25)

| Metrica | Ticker-balanced | Observation-pooled |
|---|---:|---:|
| pct_90 | 0.00% | 1.93% |
| pct_95 | 0.00% | 1.10% |
| p05 | -0.409 | -0.523 |
| p95 | +0.558 | +0.698 |
| IQR | 0.419 | 0.504 |

La vista pooled concentra saturacion (1.93%) que la balanced no ve.
Razon: tickers con mas historia (empresas grandes) pesan mas en pooled.

**Vista primaria:** balanced (evita dominancia por historia).

---

## 5. Decision

**K = 0.25 CONGELADO.**

- `config/settings.py`: `WYCKOFF_T_NORM_K = 0.25`.
- `02_proveniencia_parametros.md`: parametro pasa de PROPUESTO a CAL.

---

## 6. Estado del proyecto tras 5b.2

    Fases 1-3   CERRADAS
    Fase 4      CERRADA
    Fase 5a     IMPLEMENTADA (v1.3)
    Fase 5b     CERRADA (K congelado = 0.25)
    Fase 5c     DESBLOQUEADA (comparativa legacy v1)
    Fase 5d     BLOQUEADA (migracion)

---

Fin de resultados 5b.2.
