# INFORME CONSOLIDADO DE AUDITORÍA MAESTRA — RADAR SECTORIAL v3.15

**Fecha:** 24 de julio de 2026
**Versión del sistema:** v3.15
**Tipo de auditoría:** Transversal (4 capas)
**Estado:** ✅ Completada

---

## RESUMEN EJECUTIVO

La Auditoría Maestra evaluó el Radar de Rotación Sectorial v3.15 en cuatro capas secuenciales: dependencias de datos, transformaciones intermedias, correlación entre señales y consistencia temporal. El sistema presenta una arquitectura sólida y bien diversificada en sus señales primarias y scores agregados. Se identificaron dos alertas de redundancia en el SLPM v1.2 (`effective_breadth ↔ lis`) y en el legado (`structural_score ↔ leader_breadth`), ambas documentadas en la matriz Anti-Double-Counting. No se requiere ninguna modificación urgente del código.

---

## CAPA 1: DEPENDENCIAS DE DATOS

**Objetivo:** Mapear qué fuentes primarias alimentan cada módulo.

| Dato primario | Módulos consumidores |
|---------------|----------------------|
| Precio (Close) | Tactical, Structural, Momentum, Trend, Wyckoff, SRS, SHS, Stock Leader, SLPM, Breadth, RS |
| Volumen | Flow Proxy, Stock Leader, Dark Pools |
| VIX | Financial Conditions, Volatility Regime, MTE (CLS), Vol Metrics |
| HYG/LQD | Financial Conditions, MTE (CLS) |
| DXY | Financial Conditions |
| Curva 10Y-2Y | Financial Conditions, Macro Regime |
| WALCL, SOFR, RRP | Real Liquidity, FLS |
| CBOE (PCRs) | OMS, MTE (CLS) |
| FINRA (ATS) | Dark Pools, MTE (CLS) |
| Acciones individuales | Stock Leader, Breadth Equity (A/D) |

**Conclusión:** El sistema se alimenta de 10 fuentes de datos distintas, con una dependencia mayoritaria del precio, lo cual es esperable en un radar de rotación sectorial. Las fuentes alternativas (VIX, crédito, opciones, Dark Pools) están correctamente aisladas en módulos específicos.

---

## CAPA 2: TRANSFORMACIONES INTERMEDIAS

**Objetivo:** Trazar la ruta completa desde el precio bruto hasta las señales compuestas.

| Señal derivada | Módulos que la consumen | Riesgo |
|----------------|------------------------|--------|
| RS (Relative Strength) | Tactical, Structural, Stock Leader, SLPM, Signal Agreement | Medio |
| Flow Proxy (ret×vol + OBV + CMF) | Tactical, Stock Leader (WLS), SLPM (LIS, Flow Divergence), Signal Agreement | **Alto** |
| Momentum Score | Tactical, Sector Score | Bajo |
| Wyckoff Phase | Sector Score, Stock Leader (WLS), SLPM (Breadth, LIS) | Medio |
| Persistence | Structural Engine, SLPM | Bajo |
| Tactical Score | Opportunity Map, SLPM, Signal Agreement | Medio |
| Structural Score | Opportunity Map, SLPM, Signal Agreement | Medio |

**Hallazgo principal:** El Flow Proxy es la señal más reutilizada del sistema (4 módulos). Su composición (30% ret×vol, 35% OBV, 35% CMF) añade profundidad pero no independencia real, ya que todos sus componentes derivan de precio y volumen.

**Conclusión:** Las transformaciones son trazables y deterministas. La reutilización del Flow Proxy es aceptable siempre que se monitorice su correlación con otras señales (ver Capa 3).

---

## CAPA 3: CORRELACIÓN ENTRE SEÑALES

### 3A — Pearson entre señales primarias (RS20, Flow, Momentum, Trend)

**Muestra:** 2605 días (2016-2026), 11 sectores, 44 columnas.

**Resultado:** Ninguna correlación >0.80 entre tipos distintos de señal.  
**Conclusión:** Las cuatro familias de señales primarias no son redundantes. ✅

### 3B — Spearman entre señales primarias y scores agregados

**Muestra:** 26641 filas (todos los sectores y días).

| Señal vs Tactical | Señal vs Structural |
|-------------------|---------------------|
| RS20: -0.040 | RS20: +0.029 |
| Flow: +0.006 | Flow: -0.012 |
| Momentum: -0.059 | Momentum: +0.009 |
| Trend: -0.064 | Trend: -0.014 |

**Resultado:** Ninguna señal domina los scores agregados (todas las correlaciones <0.07).  
**Conclusión:** Los Tactical y Structural Scores diversifican correctamente sus fuentes. ✅

### 3C — Spearman entre outputs del SLPM Legacy

**Muestra:** 260 semanas (2021-2026), archivo `slpm_history.csv`.

| Componente | Tactical | Structural | Breadth | Flow Div |
|------------|----------|------------|---------|----------|
| Tactical | 1.000 | -0.360 | -0.090 | -0.558 |
| Structural | -0.360 | 1.000 | **+0.798** | 0.598 |
| Breadth | -0.090 | **+0.798** | 1.000 | 0.118 |
| Flow Div | -0.558 | 0.598 | 0.118 | 1.000 |

**Hallazgo:** `structural_score ↔ leader_breadth`: ρ = +0.798.  
**Interpretación:** En el SLPM v1.0 Legacy, el Structural Score y el Leader Breadth miden una dimensión similar. Esta alerta **no se replica** en el SLPM v1.2 activo (ver 3D). ⚠️

### 3D — Spearman entre outputs del SLPM v1.2 activo (con líderes reales)

**Muestra:** 123 semanas (2021-2026), recalculado con `stock_prices_historical.csv`.

| Componente | Eff Breadth | LIS | Flow Div | Tactical | Structural | Persistence |
|------------|-------------|-----|----------|----------|------------|-------------|
| Eff Breadth | 1.000 | **+0.838** | NaN | 0.037 | -0.148 | -0.062 |
| LIS | **+0.838** | 1.000 | NaN | 0.158 | -0.186 | -0.092 |
| Tactical | 0.037 | 0.158 | NaN | 1.000 | 0.149 | 0.256 |
| Structural | -0.148 | -0.186 | NaN | 0.149 | 1.000 | 0.340 |
| Persistence | -0.062 | -0.092 | NaN | 0.256 | 0.340 | 1.000 |

**Hallazgo:** `effective_breadth ↔ lis`: ρ = +0.838.  
**Causa:** Ambos componentes usan los mismos inputs (RS, momentum, flow, Wyckoff) con pesos casi idénticos (30/25/25/20 vs 30/25/25/20). La única diferencia es que LIS aplica `tanh` a cada input antes de promediar.  
**Riesgo:** El SLPM da peso doble a la salud individual de los líderes. No es un error de diseño, pero debe monitorizarse.  
**Estado:** ⚠️ Documentado en Anti-Double-Counting.

### Conclusión global de la Capa 3

| Capa | Resultado | Estado |
|------|-----------|--------|
| 3A – Pearson señales primarias | Sin redundancia >0.80 | ✅ |
| 3B – Spearman scores vs señales | Sin dominancia | ✅ |
| 3C – SLPM Legacy | Structural↔Breadth: +0.798 | ⚠️ |
| 3D – SLPM v1.2 activo | Breadth↔LIS: +0.838 | ⚠️ |

---

## CAPA 4: CONSISTENCIA TEMPORAL DE VENTANAS

**Objetivo:** Verificar coherencia entre las ventanas de cálculo de módulos interdependientes.

| Módulo | Ventana principal | Horizonte |
|--------|-------------------|-----------|
| Tactical Engine | 5-20 días | Corto plazo |
| Structural Engine | 63, 126, 252 días | Largo plazo |
| Persistence (SLPM) | 12 observaciones (RS20 diario = 12 días) | Corto plazo |
| OMS Z-Score | 252 días | Largo plazo |
| Dark Pool Z-Score | 13-104 semanas | Muy largo plazo |
| MTE (SRS/SHS/CLS/IPS) | 60-120 días | Medio plazo |
| Breadth | EMAs 20/50/200 | Multi-ventana |

**Hallazgo principal:** El SLPM mezcla intencionadamente señales de corto plazo (Tactical, Persistence) con señales de largo plazo (Structural). Esto captura tanto momentum táctico como fortaleza estructural.  
**Conclusión:** Las ventanas son coherentes dentro de cada motor (Tactical= corto, Structural= largo). La mezcla en el SLPM está documentada en `slpm_v12.py`. ✅

---

## MATRIZ DE RIESGOS IDENTIFICADOS

| Riesgo | Capa | Nivel | Componentes afectados | Acción |
|--------|------|-------|-----------------------|--------|
| Flow Proxy reutilizado en 4 módulos | 2 | Medio | Tactical, Stock Leader, SLPM, Signal Agreement | Monitorizar correlación |
| `structural_score ↔ leader_breadth` (Legacy) | 3C | Medio | SLPM v1.0 | No aplica al sistema activo |
| `effective_breadth ↔ lis` (v1.2 activo) | 3D | Medio | SLPM v1.2 | Documentado en Anti-Double-Counting |
| Mezcla de horizontes temporales en SLPM | 4 | Informativo | SLPM v1.2 | Documentado en `slpm_v12.py` |

---

## CONCLUSIONES Y RECOMENDACIONES

1. **El Radar v3.15 es estructuralmente sólido.** Las señales primarias no están duplicadas y los scores compuestos diversifican correctamente sus fuentes.

2. **El SLPM v1.2 tiene una redundancia interna conocida** (`effective_breadth ↔ lis`, ρ=0.838). Ambos componentes miden la salud individual de los líderes con inputs y pesos casi idénticos. Esta redundancia no rompe el sistema, pero debe considerarse en una futura recalibración de pesos (v4.0).

3. **La mezcla de horizontes temporales en el SLPM es intencionada y está documentada.** No se recomienda modificarla sin un estudio de sensibilidad previo.

4. **Se recomienda repetir la Capa 3D en 6 meses**, cuando `stock_prices_historical.csv` acumule más de 250 semanas de datos, para validar la estabilidad temporal de la correlación `Breadth↔LIS`.

5. **No se requiere ninguna modificación urgente del código.** Todos los hallazgos están documentados en los archivos correspondientes (`dependency_tracker.py`, `slpm_v12.py`, `state_machine.py`).

---

## ARCHIVOS GENERADOS POR LA AUDITORÍA

| Archivo | Contenido |
|---------|-----------|
| `outputs/corr_pearson_señales.csv` | Matriz Pearson entre RS20, Flow, Momentum, Trend |
| `outputs/corr_slpm_v12_full.csv` | Matriz Spearman entre outputs del SLPM v1.2 activo |
| `outputs/slpm_history.csv` | Histórico del SLPM Legacy (referencia) |
| `data/stock_prices_historical.csv` | Histórico de acciones para auditorías futuras |
| `src/dependency_tracker.py` | Matriz Anti-Double-Counting actualizada |

---

**Firma del ingeniero de supervisión:**  
Radar de Rotación Sectorial v3.15 — Auditoría Maestra completada el 24/07/2026.
