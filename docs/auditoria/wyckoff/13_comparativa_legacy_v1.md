# COMPARATIVA LEGACY v4.2 vs v1.3 - Resultados Fase 5c

**Documento de diagnostico. No migra consumidores.**
**Fecha:** 2026-10-02.
**Estado:** hallazgos elevables a dictamen externo.

---

## 1. Entorno

- Script: `scripts/compare_wyckoff_legacy_v13.py`.
- Fuente: `data/stock_prices.parquet`, 316 tickers.
- Legacy: `indicators/wyckoff.py` (v4.2, `robust_zscore(trend, w=60)`).
- v1.3: `indicators/wyckoff_v1.py` con K=0.25 (`tanh(trend/0.25)`).
- Fecha de ejecucion: 2026-10-02.

---

## 2. Distribucion global

| Fase | Legacy | v1.3 |
|---|---:|---:|
| ACCUMULATION | 119 (37.7%) | 21 (6.6%) |
| MARKUP | 65 (20.6%) | 77 (24.4%) |
| RANGE | 113 (35.8%) | **211 (66.8%)** |
| DISTRIBUTION | 18 (5.7%) | **0 (0.0%)** |
| MARKDOWN | 0 | 6 (1.9%) |
| INSUFFICIENT_DATA | 1 | 1 |
| **Cambios** | — | **210/316 (66.5%)** |

**Lectura critica:** v1.3 concentra dos tercios del universo en RANGE y
no clasifica ningun DISTRIBUTION. Es una distribucion muy distinta de
la esperada para un universo con tendencias mixtas.

---

## 3. Transiciones observadas (top)

| Legacy -> v1.3 | Casos |
|---|---:|
| ACCUMULATION -> RANGE | 77 |
| MARKUP -> RANGE | 37 |
| ACCUMULATION -> MARKUP | 32 |
| RANGE -> MARKUP | 25 |
| DISTRIBUTION -> RANGE | 17 |
| MARKUP -> ACCUMULATION | 8 |
| RANGE -> ACCUMULATION | 6 |
| ACCUMULATION -> MARKDOWN | 3 |
| RANGE -> MARKDOWN | 2 |
| DISTRIBUTION -> MARKUP | 1 |
| MARKUP -> MARKDOWN | 1 |

**Los dos grandes bloques son:**
- 77 tickers pasan de ACCUMULATION a RANGE (perdida de clasificacion).
- 37 tickers pasan de MARKUP a RANGE (perdida de clasificacion).

En ambos casos, v1.3 es **mas restrictivo** que legacy.

---

## 4. Score delta

Sobre 315 tickers con score en ambos modulos:

| Metrica | Valor |
|---|---:|
| media | +0.127 |
| mediana | +0.175 |
| std | 0.315 |
| min | -0.922 |
| max | +0.808 |
| `|delta| > 0.30` | 130 / 315 (41%) |

**Lectura:** 41% de tickers tienen un cambio de score > 0.30. Es un
cambio masivo. El signo es positivo en media (v1.3 suele subir score),
pero hay outliers fuertes en ambas direcciones.

---

## 5. Distribucion por sector SPDR

| Sector | Legacy AC/MK | v1.3 AC/MK | Cambios |
|---|---|---:|---:|
| XLB | 9 / 3 | 0 / 4 | 13 |
| XLC | 5 / 11 | 2 / 1 | 16 |
| XLE | 9 / 1 | 0 / 4 | 12 |
| XLF | 9 / 6 | 0 / 2 | 16 |
| XLI | 10 / 3 | 1 / 6 | 15 |
| XLK | 7 / 6 | 0 / 17 | 14 |
| XLP | 8 / 5 | 4 / 4 | 13 |
| XLRE | 4 / 0 | 0 / 2 | 6 |
| XLU | 8 / 2 | 2 / 0 | 11 |
| XLV | 6 / 11 | 0 / 6 | 14 |
| XLY | 8 / 3 | 5 / 3 | 14 |

**Patron:** v1.3 reduce drasticamente los ACCUMULATION en todos los
sectores (de 4-10 a 0-5). MARKUP varia: sube en XLK (6->17), XLI (3->6),
XLE (1->4); baja en XLV (11->6), XLC (11->1).

---

## 6. Casos extremos

### 6.1. Los 6 MARKDOWN de v1.3

| Ticker | Legacy | score_legacy | score_v13 | delta |
|---|---|---|---:|---:|---:|
| AZO | ACCUMULATION | +0.078 | -0.209 | -0.287 |
| BMW.DE | RANGE | -0.027 | -0.515 | -0.488 |
| CHTR | MARKUP | +0.376 | -0.489 | -0.866 |
| MBG.DE | ACCUMULATION | +0.242 | -0.383 | -0.624 |
| MCD | ACCUMULATION | +0.036 | -0.281 | -0.317 |
| ORCL | RANGE | -0.162 | -0.322 | -0.160 |

**CHTR es el caso mas dramatico:** legacy MARKUP (+0.376), v1.3
MARKDOWN (-0.489). Delta -0.87.

**Verificacion estructural:** todos estos tickers tienen `struct_score
< -0.30` y `t_norm < -0.30` en v1.3, con `c_norm < 0` (expansion). Es
decir: tendencia bajista confirmada con volatilidad en expansion. La
clasificacion de v1.3 es **estructuralmente coherente** con la
definicion de MARKDOWN. El legacy los veia alcistas porque media
**aceleracion** (desviacion del regimen), no nivel.

### 6.2. Los 8 MARKUP -> ACCUMULATION

Tickers: CVNA, DG, DIS, HD, ORLY, RHM.DE, RKT.L, ULVR.L.

Todos con `score_legacy > 0.35` (legacy los veia alcistas) y
`score_v13` entre -0.14 y +0.38.

**Interpretacion:** v1.3 los ve en transicion (no alcistas, no
bajistas) porque `t_norm` esta en banda baja. El legacy los veia
alcistas porque su tendencia **aceleraba**.

### 6.3. PM: DISTRIBUTION -> MARKUP

| Ticker | score_legacy | score_v13 | delta |
|---|---:|---:|---:|
| PM | -0.332 | +0.162 | +0.494 |

**Interpretacion:** PM tiene tendencia alcista actual (MA50/MA200 > 0).
El legacy lo veia en DISTRIBUTION porque la tendencia **perdia
velocidad**. v1.3 lo ve MARKUP porque la tendencia **es alcista ahora**.

**Caso semantico interesante.** PM probablemente esta en una fase
temprana de MARKUP (recien salido de un rango). Legacy lo penaliza por
desaceleracion; v1.3 lo premia por direccion.

### 6.4. ACCUMULATION -> MARKDOWN (3 casos)

AZO, MBG.DE, MCD. Todos tienen `score_legacy` positivo y `score_v13`
negativo.

**Interpretacion:** estos tickers estaban en base (legacy) y en v1.3
tienen tendencia bajista clara. Si MA50 < MA200 y `c_norm < 0`
(expansion), la lectura de v1.3 es coherente: son bajistas, no base.

---

## 7. Hallazgo principal: RANGE excesivo y DISTRIBUTION nula

### 7.1. RANGE en 66.8%

v1.3 es mas restrictivo que legacy en cada clasificacion:
- MARKUP exige `struct_score > 0.30 AND t_norm > 0.30`.
- MARKDOWN exige `struct_score < -0.30 AND t_norm < -0.30 AND c_norm < 0`.
- ACCUMULATION exige 4 condiciones conjuntas (trend_weak + base_band +
  compresion + precedente debil).
- DISTRIBUTION exige 4 condiciones conjuntas.
- Cualquier caso que no cumpla cae en RANGE.

**Efecto:** la mayoria del universo no cumple ninguna conjuncion.

### 7.2. DISTRIBUTION = 0 casos

Las condiciones de §5.4 v1.3:

    struct_score_t < STRUCT_DETERIORO (-0.10)
    AND t_norm_t > 0 OR |t_norm_t| < T_NORM_WEAK
    AND c_norm_t > C_NORM_COMPRESSION (0.30)
    AND struct_max > PREC_STRUCT_STRONG (0.30)

Son **casi contradictorias**:
- Deterioro actual (struct < -0.10) sugiere que la tendencia ya giro.
- Pero compresion alta (c > 0.30) es incompatible con giro de
  volatilidad: cuando el precio gira, la volatilidad se expande,
  `c_norm < 0`.
- Y precedente fuerte (struct_max > 0.30) que conviva con deterioro
  actual en ventana de 60 sesiones es raro.

Resultado: **DISTRIBUTION es practicamente inalcanzable.**

---

## 8. Diagnostico estructural

**v1.3 no es "peor" que legacy.** Es un clasificador **mas estricto**
con criterios mas explicitos:

- Legacy: clasifica por bandas de un score continuo (`combined`).
  Resultado: 4 estados por umbrales.
- v1.3: clasifica por **conjuncion estructural** de varios indicadores.
  Resultado: mas cobertura en RANGE, menos en categorias fuertes.

**Pero la distribucion actual (67% RANGE, 0% DISTRIBUTION) no parece
util para el objetivo del sistema.** El radar sectorial necesitaba
detectar lideres en ACCUMULATION y MARKUP. Con 6.6% + 24.4% = 31%
clasificados con direccion, el sistema tiene menos base para ranking.

---

## 9. Preguntas al auditor

1. **Sobre RANGE al 67%:** ¿es un resultado aceptable del contrato v1.3?
   ¿O sugiere que los umbrales de MARKUP/MARKDOWN/ACCUMULATION son
   demasiado estrictos?

2. **Sobre DISTRIBUTION = 0:** las condiciones §5.4 son casi
   contradictorias. ¿Deben relajarse? ¿Se acepta que DISTRIBUTION sea
   practicamente inalcanzable? (Nota: legacy daba 18 casos, 5.7%).

3. **Sobre el cambio de score en 41% de tickers (|delta| > 0.30):**
   coherente con el cambio semantico de `t_norm`, pero es un cambio
   masivo. ¿Se acepta como consecuencia esperada?

4. **Sobre casos dramaticos** (CHTR -0.87, MBG.DE -0.62, PM +0.49):
   ¿verificar 3-5 de ellos a mano para confirmar que v1.3 tiene razon
   estructural? Mi lectura: v1.3 tiene razon, legacy media aceleracion.

5. **Sobre migracion 5d:** con esta distribucion, migrar los 5
   consumidores cambiaria el top-5 de cada sector/indice
   significativamente. ¿Se procede? ¿O hay que ajustar v1.3 antes?

---

## 10. Estado

- Fase 5c: **ejecutada**. Hallazgos documentados.
- Fase 5d: bloqueada hasta respuesta del auditor.
- K=0.25: congelado.
- Legacy: intacto.

---

Fin de comparativa 5c.
