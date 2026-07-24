# MACRO SECTORIAL - Reporte Diario
**Fecha:** 2026-07-24 23:16:04
**Modelo:** v3.15 | Pesos: v3 | Indicadores: v2

## Resumen de Regimenes
- **Macro:** MIXED (Score: -0.07, Signal Consistency: 50%)
- **Cond. Financieras:** ESTRECHA (Score: -0.08, Signal Consistency: 53%)
- **Liquidez Real (FRED):** ESTRECHA (Signal Consistency: 81%)
- **Volatilidad:** NORMAL (Z-Score: 0.00, Senhal neutra (sin desviacion significativa))
- **Sectores:** MIXED
*Nota: Signal Consistency mide la consistencia entre senhales, no una probabilidad estadistica calibrada. Data Conf mide la frescura y cobertura de los datos.*

### Data Freshness
| Fuente | Ultimo dato | Antiguedad | Estado | Data Conf |
|--------|-------------|------------|--------|----------|
| CBOE (Opciones) | 2026-07-23 | 1 dias | CURRENT | Alta |
| FINRA (Dark Pools) | 2026-06-29 | 25 dias | ARCHIVAL | Baja |
| FRED (Macro) | Semanal | Variable | RECENT | Alta |
| Yahoo Finance (Precios) | Diario | < 1 dia | CURRENT | Alta |

### Divergencias Detectadas
- **Breadth Divergence:** Breadth EMA200: 82%; Breadth EMA20: 55%. La amplitud de corto plazo es inferior a la de largo plazo.
- **Energy Price-Flow:** Precio fuerte sin confirmacion del Flow Proxy. El indicador no permite inferir directamente participacion institucional.

### WARN Cross-Module: DIVERGENCE
**Mensaje:** Estres financiero concentrado en 2 modulo(s). Presion inflacionaria detectada en 1 modulo(s). los módulos no presentan clasificación uniforme.
**Bloques:** Financial Stress: 2/5 | Inflation Pressure: 1/5

**Detalle por modulo:**
- macro: MIXED (Neutral)
- financial: ESTRECHA (Estres Financiero )
- volatility: NORMAL (Neutral)
- liquidity: ESTRECHA (Estres Financiero )
- mte: STAGFLATION (Presion Inflacionaria )

## Breadth de Mercado (11 sectores)
| Metrica | Valor |
|---------|-------|
| % sobre EMA20 | 54.55% |
| % sobre EMA50 | 54.55% |
| % sobre EMA200 | 81.82% |
| New Highs sectoriales (0/11) |
| New Lows sectoriales (1/11) |


## Momentum de Precio - Sectores (20 dias)
| # | Sector | Retorno 20d (%) |
|---|--------|------------------|
| 1 | Energy (XLE) | 9.78% |
| 2 | Financials (XLF) | 4.45% |
| 3 | Healthcare (XLV) | 3.73% |
| 4 | Real Estate (XLRE) | 0.81% |
| 5 | Utilities (XLU) | 0.74% |
| 6 | Communication Services (XLC) | -0.19% |
| 7 | Consumer Staples (XLP) | -0.87% |
| 8 | Industrials (XLI) | -1.18% |
| 9 | Materials (XLB) | -2.99% |
| 10 | Technology (XLK) | -3.32% |
| 11 | Consumer Discretionary (XLY) | -4.05% |

## Flujo Institucional - Sectores (Proxy)
| # | Sector | Flujo (z-score) |
|---|--------|------------------|
| 1 | Healthcare (XLV) | 0.42 |
| 2 | Industrials (XLI) | 0.30 |
| 3 | Utilities (XLU) | 0.11 |
| 4 | Energy (XLE) | 0.05 |
| 5 | Financials (XLF) | -0.09 |
| 6 | Materials (XLB) | -0.10 |
| 7 | Consumer Staples (XLP) | -0.33 |
| 8 | Technology (XLK) | -0.65 |
| 9 | Real Estate (XLRE) | -0.73 |
| 10 | Communication Services (XLC) | -1.06 |
| 11 | Consumer Discretionary (XLY) | -1.12 |
## Tactical Leaders (Momentum de corto plazo)
| # | Sector | Tactical | Structural | Retorno 20d | Flow Proxy (z) | Comm Corr |
|---|--------|----------|------------|-------------|----------------|------------|
| 1 | Energy (XLE) | +0.42 | +0.08 | 9.78% | +0.05 | HIGH (+0.64) |
| 2 | Financials (XLF) | +0.28 | -0.01 | 4.45% | -0.09 | MODERATE (-0.31) |
| 3 | Healthcare (XLV) | +0.19 | +0.03 | 3.73% | +0.42 | MODERATE (-0.33) |
| 4 | Utilities (XLU) | +0.05 | -0.02 | 0.74% | +0.11 | LOW (-0.13) |
| 5 | Real Estate (XLRE) | +0.04 | -0.00 | 0.81% | -0.73 | MODERATE (-0.31) |
| 6 | Communication Services (XLC) | -0.05 | -0.10 | -0.19% | -1.06 | LOW (-0.29) |
| 7 | Industrials (XLI) | -0.07 | +0.02 | -1.18% | +0.30 | MODERATE (-0.39) |
| 8 | Consumer Staples (XLP) | -0.07 | -0.04 | -0.87% | -0.33 | LOW (-0.24) |
| 9 | Technology (XLK) | -0.18 | +0.09 | -3.32% | -0.65 | LOW (-0.19) |
| 10 | Materials (XLB) | -0.23 | -0.02 | -2.99% | -0.10 | MODERATE (-0.30) |
| 11 | Consumer Discretionary (XLY) | -0.23 | -0.09 | -4.05% | -1.12 | MODERATE (-0.53) |

*Nota: Comm Corr mide la correlacion de 126 dias con ^SPGSCI. No implica causalidad.*


## Momentum de Precio - Otros Activos (20 dias)
| # | Activo | Retorno 20d (%) |
|---|--------|------------------|
| 1 | BZ=F | 33.40% |
| 2 | CL=F | 28.13% |
| 3 | ^SPGSCI | 13.72% |
| 4 | HG=F | 4.33% |
| 5 | QUAL | 1.29% |
| 6 | USDJPY=X | 1.27% |
| 7 | ^STOXX50E | 0.79% |
| 8 | ^GSPC | 0.69% |
| 9 | GC=F | 0.53% |
| 10 | BIL | 0.27% |
| 11 | EURUSD=X | 0.24% |
| 12 | SCHC | 0.04% |
| 13 | DX-Y.NYB | 0.01% |
| 14 | USDCNY=X | -0.34% |
| 15 | HYG | -0.35% |

## Flujo Institucional - Otros Activos (Proxy)
| # | Activo | Flujo (z-score) |
|---|--------|------------------|
| 1 | ^STOXX50E | 0.51 |
| 2 | CL=F | 0.43 |
| 3 | BZ=F | 0.38 |
| 4 | HG=F | 0.07 |
| 5 | SCHC | 0.05 |
| 6 | EEM | -0.15 |
| 7 | ELD | -0.15 |
| 8 | QUAL | -0.32 |
| 9 | ^NDX | -0.33 |
| 10 | LQD | -0.36 |
| 11 | NG=F | -0.41 |
| 12 | EWJ | -0.45 |
| 13 | EMB | -0.47 |
| 14 | GC=F | -0.49 |
| 15 | ^GSPC | -0.54 |
## Structural Ranking (Fortaleza de largo plazo)
| # | Sector | Structural | Tactical | Persist | Agreement | Signal Consistency |
|---|--------|------------|----------|---------|-----------|------------|
| 1 | Technology (XLK) | +0.09 | -0.18 | 0% | 75% BEARISH | 38% |
| 2 | Energy (XLE) | +0.08 | +0.42 | 75% | 100% BULLISH | 88% |
| 3 | Healthcare (XLV) | +0.03 | +0.19 | 100% | 100% BULLISH | 100% |
| 4 | Industrials (XLI) | +0.02 | -0.07 | 42% | 50% MIXED | 46% |
| 5 | Real Estate (XLRE) | -0.00 | +0.04 | 50% | 50% MIXED | 50% |
| 6 | Financials (XLF) | -0.01 | +0.28 | 100% | 50% MIXED | 75% |
| 7 | Utilities (XLU) | -0.02 | +0.05 | 83% | 75% BULLISH | 79% |
| 8 | Materials (XLB) | -0.02 | -0.23 | 0% | 100% BEARISH | 50% |
| 9 | Consumer Staples (XLP) | -0.04 | -0.07 | 33% | 100% BEARISH | 67% |
| 10 | Consumer Discretionary (XLY) | -0.09 | -0.23 | 8% | 100% BEARISH | 54% |
| 11 | Communication Services (XLC) | -0.10 | -0.05 | 58% | 100% BEARISH | 79% |

## Rankings Sectoriales (Score combinado original)
> *Nota: Este Score es el ranking historico del sistema (momentum, tendencia, volatilidad, breadth, Wyckoff). No es el Tactical ni el Structural Score.*

| # | Sector | Score | Tactical | Structural | Persist | Agreement | Comm Corr | Fase Wyckoff |
|---|--------|-------|----------|------------|---------|-----------|------------|---------------|
| 1 | Energy (XLE) | 0.47 | +0.42 | +0.08 | 75% | 100% BULLISH | HIGH (+0.64) | RANGE |
| 2 | Real Estate (XLRE) | 0.41 | +0.04 | -0.00 | 50% | 50% MIXED | MODERATE (-0.31) | RANGE |
| 3 | Financials (XLF) | 0.39 | +0.28 | -0.01 | 100% | 50% MIXED | MODERATE (-0.31) | ACCUMULATION |
| 4 | Utilities (XLU) | 0.21 | +0.05 | -0.02 | 83% | 75% BULLISH | LOW (-0.13) | RANGE |
| 5 | Industrials (XLI) | 0.20 | -0.07 | +0.02 | 42% | 50% MIXED | MODERATE (-0.39) | ACCUMULATION |
| 6 | Healthcare (XLV) | 0.07 | +0.19 | +0.03 | 100% | 100% BULLISH | MODERATE (-0.33) | ACCUMULATION |
| 7 | Technology (XLK) | -0.00 | -0.18 | +0.09 | 0% | 75% BEARISH | LOW (-0.19) | ACCUMULATION |
| 8 | Materials (XLB) | 0.00 | -0.23 | -0.02 | 0% | 100% BEARISH | MODERATE (-0.30) | RANGE |
| 9 | Consumer Staples (XLP) | -0.00 | -0.07 | -0.04 | 33% | 100% BEARISH | LOW (-0.24) | RANGE |
| 10 | Communication Services (XLC) | 0.00 | -0.05 | -0.10 | 58% | 100% BEARISH | LOW (-0.29) | DISTRIBUTION |
| 11 | Consumer Discretionary (XLY) | -0.13 | -0.23 | -0.09 | 8% | 100% BEARISH | MODERATE (-0.53) | RANGE |

## Opportunity Map (basado en medianas Tactical/Structural, independiente del SLPM)

*Umbrales del dia: Tactical mediana=-0.05, Structural mediana=-0.01*

| Cuadrante | Sectores | Signal Consistency |
|-----------|----------|------------|
| VERDE **Structural Strength** | Healthcare, Energy, Real Estate | 79% |
| AMARILLO **Tactical Correction** | Technology, Industrials | 42% |
| AZUL **Tactical Strength** | Financials, Utilities | 77% |
| ROJO **Structural Weakness** | Consumer Discretionary, Consumer Staples, Materials, Communication Services | 62% |
| GRIS **Transition** | -- | -- |

*Nota: 'Structural Strength' en Opportunity Map identifica posicion relativa en el eje Structural. No implica liderazgo confirmado por SLPM.*

*Cobertura de lideres SLPM: 3/5 (60%)*

## Structural Leadership (SLPM v1.2)
- **Sector Lider:** Energy
  - *Nota: El SLPM selecciona al lider combinando Tactical, Structural, LIS, Breadth y Persistence. No es simplemente el sector con mayor Structural Score.*
- **Estado:** UNRESOLVED -> Transition
  - *Ninguna condicion de estado se cumple. Senhales mixtas o insuficientes.*
- **Scores oficiales:** T=+0.42 | S=+0.08 | LIS=+0.25 | Eff Breadth=0.80 | Persist=75%

### Leader Breadth & Health
- **Leader Breadth (RS ratio > 1.0):** 100%
- **Leader Momentum Breadth:** 100%
- **Leader Flow Support:** 100%
- **Leader Wyckoff Health:** 0%
  - *Scoring Wyckoff: MARKUP=1.0, ACCUMULATION=0.75, RANGE=0.0, DISTRIBUTION=-0.75, MARKDOWN=-1.0*
- **Leader Health Composite (sin ajustar):** 80% (0.30xRS + 0.25xMom + 0.25xFlow + 0.20xWyckoff)
- **Effective Breadth (Health 80% x Cobertura 60%):** 80%
  - N analizado: 3/5
  - *Nota: La confianza ajustada reduce la senhal por baja cobertura. La calidad observada (Health Composite) es independiente de la cobertura.*

### Leader Integrity Score (LIS)
- **LIS:** +0.25 (n=3)
- *Formula: LIS_individual = 0.30*tanh((RS-1)*2) + 0.25*tanh(RS_mom*5) + 0.25*tanh(flow_z/2) + 0.20*Wyckoff_score. LIS = media.*
- *LIS mide la intensidad/calidad de la senhal de los lideres, no el % que cumple condiciones (eso es el Breadth).*

### Flow Divergence 2.0
- **Composite:** +0.357
  - Leader vs Sector: +0.509
  - Sector Flow vs Price: +0.000
  - Structural: +0.408
- *Nota: Flujo medido como Flow Proxy (retorno x volumen). No implica flujo institucional real.*


## Acciones Seleccionadas por el Modelo de Liderazgo Sectorial
> Acciones identificadas por el modelo dentro de sectores favorables.

## Sector: XLK (ACCUMULATION)
| Ticker | RS | RS Mom | Flujo (z) | WLS | Fase Wyckoff | Spring | SOS |
|--------|----|--------|-----------|-----|---------------|--------|-----|
| PLTR | 0.69 | 17.36% | -0.42 | 0.61 | DISTRIBUTION |  |  |
| AVGO | 2.20 | 6.89% | 0.17 | 0.47 | RANGE |  |  |
| NVDA | 1.17 | 9.81% | 0.19 | 0.07 | ACCUMULATION |  |  |

## Sector: XLF (ACCUMULATION)
| Ticker | RS | RS Mom | Flujo (z) | WLS | Fase Wyckoff | Spring | SOS |
|--------|----|--------|-----------|-----|---------------|--------|-----|
| BAC | 1.10 | 0.82% | 0.11 | 1.22 | MARKUP |  |  |
| JPM | 6.27 | 0.41% | 0.34 | 0.97 | ACCUMULATION |  |  |
| V | 6.30 | 1.83% | -0.14 | 0.73 | ACCUMULATION |  |  |

## Sector: XLV (ACCUMULATION)
| Ticker | RS | RS Mom | Flujo (z) | WLS | Fase Wyckoff | Spring | SOS |
|--------|----|--------|-----------|-----|---------------|--------|-----|
| TMO | 3.55 | 8.70% | 1.56 | 2.14 | ACCUMULATION |  | ✓ |
| ABT | 0.62 | 4.79% | 0.50 | 1.13 | ACCUMULATION |  |  |
| JNJ | 1.61 | 2.04% | 0.65 | 0.22 | RANGE |  |  |

## Sector: XLI (ACCUMULATION)
| Ticker | RS | RS Mom | Flujo (z) | WLS | Fase Wyckoff | Spring | SOS |
|--------|----|--------|-----------|-----|---------------|--------|-----|
| UNP | 1.67 | 14.00% | 1.52 | 2.22 | MARKUP |  | ✓ |
| RTX | 1.15 | 12.61% | 1.71 | 2.04 | ACCUMULATION |  | ✓ |
| PH | 5.42 | 0.74% | 0.39 | 0.34 | RANGE |  |  |

## Sentimiento de Opciones (OMS v2.0)
- **PCR Total:** 0.88 (EWMA(5): N/D - historial insuficiente)
- **PCR Indices:** 0.86 | **PCR Acciones:** 0.61 | **PCR ETP:** 1.28
- **PCR VIX:** 0.29 | **PCR SPX:** 1.09
- **Institutional Hedge Ratio:** 1.41 (Equilibrado, bandas: <1.2 Especulacion, 1.2-1.6 Equilibrado, >1.6 Cobertura institucional)
- **Volumen en Indices:** 47.4% del total
- **Put Share:** 46.9% | **Call Share:** 53.1%
- **Volume PCR (calculado):** 0.88 | **OI PCR:** 0.75
- **Ultimo dato:** 2026-07-23 (desfase: 1 dias)

*Fuente: CBOE Official Data. Timestamp: 2026-07-24 23:14:33.*

## Market Transition Engine (MTE v1.0)
- **Escenario (UNCONFIRMED):** STAGFLATION (Signal Consistency: 0) - *No se considera confirmado.*
*Nota: Signal Consistency representa la distancia a los umbrales y el consenso entre motores. No esta calibrada historicamente.*
- **Market Stress Index (MSI):** 34
- **Inflation Pressure Index (IPI):** 66
- **Sector Rotation Score:** -0.09
- **Safe Haven Score:** -0.01
- **Credit Stress Score:** 0.08 (orientacion: positivo = mayor estres crediticio)
- **Inflation Pressure Score:** 0.31

## Confirmation Data (Nivel 2)
> *Indicadores de confirmacion. No modifican el macro_score.*

- **10Y-3M Spread:** +0.76%
- **Realized Vol (21d):** 11.37%
- **Realized Vol (60d):** 13.10%
- **VRP Proxy (VIX - RV21):** +7.33%
- **VRP Proxy (VIX - RV60):** +5.60%
- **Funding & Liquidity Stress (FLS):** 69/100 (3/5 componentes en estres)
  - Desglose:
    OK SOFR: -0.52
    WARN WALCL: 1.00
    WARN RRP: 0.60
    WARN CP: 0.79
    OK Discount: 0.00
- **Advance/Decline Net:** -12 (49 avances / 61 descensos)
- **New Highs/Lows (mercado):** 6 maximos / 3 minimos (NH-NL: +3)
- **A/D Line (acumulada):** 5884

### Cross-Asset Ratios
| Ratio | Valor | Delta 20d | Z-Score (60d) |
|-------|-------|-----------|---------------|
| Copper/Gold | 0.0016 | +3.8% | +0.38 |
| TLT/IEF | 0.8957 | -2.8% | -1.06 |
| DXY/EEM | 1.5702 | +5.2% | +0.52 |
| HYG/LQD | 0.7456 | +2.3% | +1.86 |
| KRE/SPY | 0.1018 | -0.0% | +0.90 |
| SMH/SPY | 0.7859 | -9.4% | -0.06 |
| IYT/SPY | 0.1195 | +1.3% | +0.90 |
| XLE/SPY | 0.0804 | +9.2% | +0.70 |
| XLU/SPY | 0.0626 | +0.2% | +0.42 |
| XLV/SPY | 0.2187 | +3.2% | +0.75 |
| XLP/SPY | 0.1127 | -1.4% | -0.11 |

## Actividad en ATS - Dark Pools (FINRA v1.0)
**DATOS OBSOLETOS:** Ultimo dato con 25 dias de antiguedad. No se usa para clasificacion actual. Contexto historico solamente.

- **% Volumen en ATS medio:** 19.51% (144/144 tickers)
- **Z-Scores por ventana:**
  - 13w: Z=-4.66, Estado=Distribucion extrema
  - 26w: Z=-3.71, Estado=Distribucion extrema
- **Semana FINRA:** 2026-06-29

**Mayor % de volumen en ATS:**
| Ticker | % ATS | Vol ATS | Vol Total |
|--------|:-----:|:-------:|:---------:|
| ETN | 30.40% | 2,954,751 | 9,719,900 |
| NUE | 30.02% | 2,265,927 | 7,548,400 |
| VMC | 28.94% | 1,719,624 | 5,942,400 |
| VST | 28.89% | 4,720,553 | 16,341,000 |
| GEV | 28.41% | 3,124,306 | 10,996,600 |

*Nota: Un alto % de volumen en ATS NO implica acumulacion institucional. 'Distribucion extrema' se refiere al nivel de actividad ATS, no a distribucion Wyckoff.*

*Fuente: FINRA ATS Transparency Data.*

## Estado Actual — Síntesis de Señales

- **Régimen macro: MIXED** — score dentro del rango de clasificación MIXED.
- **Liderazgo no confirmado: Energy** (#1 del ranking, SLPM: UNRESOLVED).
- **Condiciones financieras: ESTRECHA** — señales financieras en territorio restrictivo.

*Esta sección describe únicamente estados observables del sistema. No interpreta causas ni sugiere acciones.*


*Esta interpretacion es descriptiva y no constituye una recomendacion de inversion.*

### Anti-Double-Counting Audit
*Advertencia: Algunas variables son utilizadas por multiples modulos. Esto no implica error, pero el gestor debe saber que estas senhales pueden estar correlacionadas.*

**Critico (2 variables compartidas por 4+ modulos):**
- **Relative Strength (RS)** (4 modulos): stock_leader.py, slpm_v12.py, tactical_engine.py, structural_engine.py
- **Flow Proxy** (4 modulos): stock_leader.py, slpm_v12.py, tactical_engine.py, structural_engine.py

**Alto (2 variables compartidas por 3 modulos):**
- **VIX** (3 modulos): financial_conditions.py, volatility_regime.py, mte.py
- **Credit Signal** (3 modulos): financial_conditions.py, macro_regime.py, mte.py

**Medio (8 variables compartidas por 2 modulos):**
- **HYG/LQD (credito)** (2 modulos): financial_conditions.py, mte.py
- **SOFR** (2 modulos): liquidity.py, fls.py
- **WALCL** (2 modulos): liquidity.py, fls.py
- **RRP** (2 modulos): liquidity.py, fls.py
- **Tactical Score** (2 modulos): opportunity_map, slpm_v12.py
- **Structural Score** (2 modulos): opportunity_map, slpm_v12.py
- **Persistence** (2 modulos): structural_engine.py, slpm_v12.py
- **Leader Breadth** (2 modulos): structural_engine.py, slpm_v12.py

**Dependencias indirectas detectadas:**
- **Persistence → Structural Score → SLPM**: La persistencia alimenta el Structural Score, que a su vez alimenta el SLPM.
- **Tactical Score → Opportunity Map + SLPM**: El Tactical Score se usa tanto en el Opportunity Map como en el SLPM.
- **Structural Score → Opportunity Map + SLPM**: El Structural Score se usa tanto en el Opportunity Map como en el SLPM.
- **WLS → LIS → SLPM**: El Wyckoff Leadership Score (stock_leader.py) selecciona lideres usando RS/Flow intra-sector. Luego el LIS (slpm_v12.py) vuelve a transformar RS y Flow con tanh. Canal de amplificacion documentado.

*Esta matriz es informativa. No modifica ningun calculo.*
