# 5c.2 - Re-ejecucion comparativa con v1.5

**Documento de diagnostico. Fase 5c.2 ejecutada.**
**Fecha:** 2026-10-02.
**Precedente:** dictamen O1 aprobada (`15_hallazgo_distribution_v14.md`).

---

## 1. Cambio v1.4 -> v1.5

§5.4 DISTRIBUTION reformulada segun O1 aprobada:

    struct_score_t < -0.10
    AND t_norm_t > -0.30
    AND struct_max > 0.30

- Eliminada la condicion `c_norm` (deterioro y compresion estaban
  correlacionadas negativamente en el universo).
- Simplificada la condicion de `t_norm` (`> 0 OR |t|<0.30` era
  redundante con `> -0.30`).
- Formalizacion historica de `struct_max`: `max(struct_score[t-W+1:t-1])`.

Tests nuevos: I26 (requires_prior_strength), I27 (boundary vs MARKDOWN),
I28 (no_lookahead con as_of).

---

## 2. Resultado 5c.2

### 2.1. Distribucion global

| Fase | Legacy | v1.5 |
|---|---:|---:|
| ACCUMULATION | 119 (37.7%) | 21 (6.6%) |
| MARKUP | 65 (20.6%) | 77 (24.4%) |
| RANGE | 113 (35.8%) | **201 (63.6%)** |
| DISTRIBUTION | 18 (5.7%) | **10 (3.2%)** |
| MARKDOWN | 0 | 6 (1.9%) |
| INSUFFICIENT_DATA | 1 | 1 |

**DISTRIBUTION ya no es 0.** 10 casos, coherente con equity.

**RANGE cede 10 tickers** (211 -> 201). Antes absorbia los que no
cumplian ninguna condicion; con v1.5 los que tienen `struct_max > 0.30`
+ `struct_t < -0.10` + `t_norm > -0.30` ya salen como DISTRIBUTION.

### 2.2. Los 10 DISTRIBUTION v1.5

| Ticker | Legacy | score_legacy | score_v1.5 |
|---|---|---:|---:|
| ACS.MC | DISTRIBUTION | -0.564 | -0.344 |
| ANA.MC | RANGE | -0.261 | +0.121 |
| BA | DISTRIBUTION | -0.497 | +0.110 |
| BKNG | RANGE | -0.099 | -0.109 |
| O | RANGE | -0.136 | +0.124 |
| PCG | DISTRIBUTION | -0.635 | -0.428 |
| SLB | ACCUMULATION | +0.001 | -0.067 |
| TEF.MC | RANGE | -0.100 | -0.121 |
| VMRK | DISTRIBUTION | -0.441 | -0.205 |
| WFC | ACCUMULATION | +0.158 | -0.079 |

**Legacy de estos 10:**
- DISTRIBUTION: 4 (ACS.MC, BA, PCG, VMRK).
- RANGE: 4 (ANA.MC, BKNG, O, TEF.MC).
- ACCUMULATION: 2 (SLB, WFC).

**Todos con `struct_max > 0.30` en los ultimos 60 dias.** Es decir:
fueron fuertes, ahora estan deteriorados, pero no bajistas establecidos.

### 2.3. Casos llamativos

- **SLB y WFC**: legacy los clasificaba ACCUMULATION. v1.5 los ve
  DISTRIBUTION. Razon: `struct_max` reciente > 0.30 (fueron fuertes
  hace poco), `struct_actual < -0.10` (deterioro). El legacy media
  aceleracion; veia mejoria en el deterioro. v1.5 lee nivel.

- **BA**: legacy DISTRIBUTION (-0.497) -> v1.5 DISTRIBUTION (+0.110
  score). Coincidencia en la fase, divergencia en el score.
  Explicacion: `combined` en v1.5 puede ser positivo si `t_norm` es
  alto, incluso en DISTRIBUTION. La fase no depende de `combined`.

### 2.4. Por sector SPDR

Cambios respecto a v1.4:
- XLK: 17 MARKUP (sin cambio).
- XLC: 2 ACC + 1 MK (sin cambio).
- XLV: 6 MK (sin cambio).

Los cambios estan fuera de los SPDR mayoritariamente (small caps,
europeos, mid caps). **Los 11 sectores SPDR no cambian
significativamente con v1.5.**

---

## 3. Verificacion de tests

- **32 tests previos**: sin cambios, todos verdes.
- **I26** (`test_distribution_requires_prior_strength`): verde.
- **I27** (`test_distribution_vs_markdown_boundary`): verde.
- **I28** (`test_distribution_no_lookahead`): **reescrito** con datos
  reales y `as_of`. Verde.

**Suite completa:** 3077 passed + 5 skipped.

---

## 4. Conclusion

**H1 resuelto.** DISTRIBUTION es alcanzable y da 10 casos reales.
La distribucion de fases queda:

    RANGE 63.6%, MARKUP 24.4%, ACCUMULATION 6.6%,
    DISTRIBUTION 3.2%, MARKDOWN 1.9%

Sin cuello de botella artificial. RANGE sigue alto pero genuino
(analisis H2 en dictamen previo).

**Fase 5c cerrada. Siguiente paso: revision del auditor + 5d si aprueba.**

---

Fin de 5c.2.
