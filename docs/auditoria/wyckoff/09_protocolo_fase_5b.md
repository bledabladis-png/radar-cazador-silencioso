# PROTOCOLO FASE 5b - Calibracion de K

**Documento normativo. Define el procedimiento de calibracion del
parametro K de `t_norm = tanh(trend / K)`.**
**Fecha:** 2026-10-02.
**Version:** v3 (revisada por dictamen externo tras ejecucion 5b v2).
**Estado:** APROBADA PARA EJECUCION (5b.2).

---

## 0. Cambios respecto a v1 (dictamen externo)

El dictamen externo aprobo la arquitectura general pero exigio 5
correcciones + 2 metodologicas:

1. **Terminologia temporal:** 70/30 es `temporal holdout`, no
   `walk-forward`. Corregido en §3.3.
2. **"Hoy" -> "ultima sesion cerrada":** evitar incorporar barra
   parcial. Corregido en §3.2.
3. **Regla de seleccion invertida:** "el K mas pequeno" es
   matematicamente MENOS conservador (K menor -> mas saturacion).
   Nueva regla: **el mayor K que cumpla todos los criterios duros**.
   Corregido en §6.6.
4. **Metrica cross-sectional:** `% valores distintos` no discrimina
   con variable continua. Sustituido por IQR cross-sectional. §5.5.
5. **Separacion criterios duros/diagnosticos:** H1-H4 obligatorios,
   D1-D3 diagnosticos. §6.1-6.5.
6. **Ponderacion balanceada por ticker** ademas del pooled. §5.0.
7. **Registrar survivorship bias.** §3.4.


---

## 0bis. Correcciones v3 (dictamen post-ejecucion 5b v2)

Tras ejecutar 5b v2 (resultado: K=0.25 pasa calibracion, falla
validacion H3), el dictamen externo exigio 4 correcciones:

**C1. H3/H4 intra-ticker calculados POR SPLIT.**
En lugar de distribucion cross-sectional agregada, calcular por ticker
la distribucion temporal de t_norm DENTRO de cada split (cal 70% y
val 30% por separado). NO se usa la historia completa 2021-2026 para
seleccionar K.

**C2. H3/H4 fuera de validacion OOS.**
En validacion, H3/H4 dejan de ser hard criteria. Razon: exigen que el
periodo OOS reproduzca colas que dependen del regimen. Nuevos criterios
OOS obligatorios:
  V1 bounds
  V2 ausencia de saturacion extrema (pct_90, pct_95)
  V3 no degradacion grave de resolucion
  V4 ausencia de comportamiento patologico

Diagnosticos OOS: p05, p95, skewness (reportados, no fallo).

**C3. Monotonicidad como invariante.**
Anadida al contrato: trend_a < trend_b -> t_norm_a < t_norm_b.
Test permanente en suite.

**C4. Nomenclatura formal.**
  Ticker-balanced     -> primario
  Observation-pooled  -> sensibilidad obligatoria

---

## 1. Proposito

Determinar un valor de `K` para el parametro `WYCKOFF_T_NORM_K` que
produzca una representacion util, estable y no saturada del nivel de
tendencia sobre el universo y periodo definidos.

**No es:**
- Un ajuste para que los 4 tickers (MSFT/PLTR/INTC/AMD) queden "bien".
- Una maximizacion de clasificaciones de una fase concreta.
- Una optimizacion libre.

**Si es:**
- Un procedimiento predefinido, con criterios escritos antes de mirar
  resultados.
- Una calibracion sobre la distribucion de `trend`, no sobre las fases.
- Un paso bloqueante antes de 5c y 5d.

---

## 2. Principios

**P1. Separacion calibracion/validacion.**
**P2. Anclaje sobre `trend`:** K se calibra sobre la distribucion de
`trend = MA50/MA200 - 1`, no sobre la salida del clasificador.
**P3. Criterio ex-ante:** las metricas y umbrales de aceptacion se
fijan en este documento, no pueden modificarse a posteriori.
**P4. Grid predefinida.**
**P5. Congelacion:** K se congela tras aplicar §6. Cualquier cambio
posterior abre 5b.2.
**P6. Validacion en holdout temporal independiente.**

---

## 3. Entrada

### 3.1. Dataset

- **Fuente:** `data/stock_prices.parquet` (universo USA + Europa).
- **Campo:** `Close` por ticker.
- **Universo:** todos los tickers con >= 200 observaciones validas.
- **No se filtran** por sector ni por cap.

### 3.2. Periodo

- **Inicio:** 2021-10-01.
- **Fin:** **ultima sesion bursatil completamente cerrada y consolidada
  disponible para cada instrumento.** NO "hoy". No se incorporan barras
  parciales.
- **Razon:** evita incorporar la barra en curso del dia de ejecucion.

### 3.3. Split temporal

**Temporal holdout** (no walk-forward):

    70% inicial -> calibracion
    30% final   -> validacion fuera de muestra

- **Justificacion:** 70/30 es razonable para estimar una escala robusta
  de transformacion (no es prediccion, no requiere walk-forward con
  multiples ventanas).
- **Nota terminologica:** walk-forward implica multiples ventanas
  train/validate. Aqui solo hay un corte. Se llama `temporal holdout`.

### 3.4. Survivorship bias (registro)

El universo es **el actualmente presente en el parquet**, no el
universo historico completo en cada fecha desde 2021. Esto introduce
un sesgo de supervivencia conocido y **debe registrarse como tal** en
el informe final:

    universo = instrumentos actualmente presentes
    NO es: universo historico completo

No es bloqueante para calibrar una transformacion matematica, pero
debe quedar explicito.

---

## 4. Candidatos de K

Grid primaria predefinida:

    K in {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40}

**Grid 5b.2** (si la primaria falla):

    K in {0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80}

**Grid 5b.3** (si 5b.2 tambien falla):

    K in {0.90, 1.00, 1.10, 1.20, 1.30, 1.40}

**Prohibido:** anadir K fuera de las grids predefinidas. Cada grid es
cerrada.

---

## 5. Metricas

### 5.0. Dos vistas: balanced y pooled

Los resultados se calculan en **dos vistas**:

**Vista balanceada (primaria):**
Para cada ticker se calcula su serie de `t_norm`. Se agregan metricas
por ticker (mediana por ticker), y despues se agregan los tickers con
peso igual. **Cada ticker aporta lo mismo**, independientemente de su
historia.

**Vista pooled (sensibilidad):**
Todas las observaciones de todos los tickers concatenadas. Los tickers
con mas historia pesan mas.

Ambas se reportan. La seleccion de K se hace sobre la **vista
balanceada** (objetivo: K comun comparable entre activos).

### 5.1. Metricas de saturacion

- `pct_extreme_90`: % `|t_norm| > 0.90`.
- `pct_extreme_95`: % `|t_norm| > 0.95`.
- `max_abs_t_norm`.

### 5.2. Metricas de resolucion

- `p05, p25, p50, p75, p95` de `t_norm`.
- `IQR = p75 - p25`.
- `std_t_norm`.

### 5.3. Metricas de simetria

- `skewness`.
- `pct_positivos`.

### 5.4. Metricas de estabilidad temporal

- `std_por_anio` (desviacion estandar anual).
- `rango_p50_por_anio` (max-min de la mediana anual).

### 5.5. Metrica de comparabilidad cross-sectional

**IQR cross-sectional de `t_norm` entre tickers en una misma fecha.**

Para cada fecha valida `t` con N >= 5 tickers:

    IQR_cs(t) = p75(t_norm_tickers(t)) - p25(t_norm_tickers(t))

Metricas agregadas:

    mediana_IQR_cs     = median(IQR_cs(t) para todas las fechas)
    pct_fechas_IQR_alto = % fechas con IQR_cs > 0.10

**Sustituye** al "pct_valores_distintos" del protocolo v1. Razon: con
variable continua, `% valores distintos` no discrimina (seria >95%
para cualquier K).

---

## 6. Criterios v3

### 6.1. Calibracion (70% inicial) - hard criteria

**H3/H4 calculados intra-ticker:**

    para cada ticker:
        p05_ticker = p05 de su t_norm dentro del 70%
        p95_ticker = p95 de su t_norm dentro del 70%

**Agregados ticker-balanced** (mediana entre tickers):

    H1  mediana(pct_90_ticker) < 15%
    H2  mediana(pct_95_ticker) < 5%
    H3  mediana(p05_ticker) <= -0.40
    H4  mediana(p95_ticker) >= +0.40

Los 4 son obligatorios en calibracion.

### 6.2. Diagnostico (calibracion)

    D1  |skewness| < 1.0
    D2  rango_p50_por_anio < 0.40
    D3  mediana_IQR_cs >= 0.05

Se reportan. No invalidan.

### 6.3. Regla de seleccion

Entre los K que cumplen H1-H4 (intra-ticker, calibracion):

    seleccionar el MAYOR K.

Justificacion matematica: mayor K -> menor saturacion. Es el que
maximiza reduccion de saturacion sin perder resolucion exigida.

### 6.4. Si ningun K cumple

Grid 5b.2 (0.45-0.80) o 5b.3 (0.90-1.40), predefinidas.

---

## 7. Validacion holdout (30% final) - v3

### 7.1. Criterios obligatorios OOS

    V1 bounds       -> -1 < t_norm < 1 (por construccion)
    V2 saturacion   -> pct_90 < 15%, pct_95 < 5% (ticker-balanced)
    V3 resolucion   -> IQR_cs mediana >= 0.05
    V4 no-patologico -> sin NaN masivos, sin colapso a 0

**Todos obligatorios. Si alguno falla, 5b.2 FALLA.**

### 7.2. Diagnosticos OOS (reportados, no fallo)

    p05
    p95
    skewness

Su valor se reporta. Un p05 = -0.19 en OOS no significa que K falle,
significa que el periodo OOS tiene distribucion sesgada hacia positivo.

### 7.3. Criterio de exito

5b.2 tiene exito si:
- Calibracion: H1-H4 pasan para el K seleccionado (mayor que pasa).
- Validacion OOS: V1-V4 pasan.
- Diagnosticos D1-D3 y p05/p95/skew reportados.

**NO se exige que p05/p95 OOS coincidan con calibracion.**

---

## 8. Congelacion

Si 5b tiene exito (grid primaria + validacion OK):

1. `WYCKOFF_T_NORM_K` en `config/settings.py` -> valor final.
2. `02_proveniencia_parametros.md`: `PROPUESTO` -> `CAL` con:
   - valor final de K;
   - fecha de calibracion;
   - periodo;
   - grid evaluada;
   - metricas H1-H4 + D1-D3 en calibracion y validacion;
   - criterio de eleccion ("mayor K cumpliendo H1-H4");
   - aviso de survivorship bias.
3. Commit documental + codigo.

---

## 9. Sensibilidad exploratoria

Analisis exploratorio permitido (etiquetado como tal, NO es 5b):

- Rango amplio de K (0.05 a 0.50, pasos de 0.05).
- Sin criterios de aceptacion.
- Solo para observar curvas de saturacion vs K, percentiles,
  sensibilidad de fases.

**Prohibido:** usarlo para elegir K.

---

## 10. Analisis permitidos

- Distribucion de `trend` en el universo/periodo.
- Comparacion de los candidatos de K sobre §5.
- Analisis de sensibilidad de `t_norm` a K.

## 11. Analisis NO permitidos

- Elegir K maximizando el numero de MARKUP (o cualquier fase).
- Elegir K segun los 4 tickers MSFT/PLTR/INTC/AMD.
- Elegir K segun backtest de WLS.
- Modificar criterios ex-ante.

---

## 12. Salida

`docs/auditoria/wyckoff/10_calibracion_K_resultados.md` con:
- Tabla K vs H1-H4 + D1-D3, dos vistas (balanced + pooled).
- Aviso de survivorship bias.
- K elegido + justificacion (§6.3).
- Resultados de validacion H1-H4 (§7).
- Comparativa balanced vs pooled como sensibilidad.

---

## 13. Estado del proyecto tras 5b

Si exito:

    Fases 1-3   CERRADAS
    Fase 4      CERRADA
    Fase 5a     IMPLEMENTADA (v1.3)
    Fase 5b     CERRADA (K congelado)
    Fase 5c     DESBLOQUEADA (comparativa legacy v1)
    Fase 5d     BLOQUEADA (migracion)

---

## 14. Procedimiento paso a paso (v3)

    PASO 1  Cargar stock_prices.parquet
    PASO 2  Ultima sesion cerrada por ticker
    PASO 3  Filtrar tickers >= 200 obs validas
    PASO 4  Calcular trend por ticker
    PASO 5  Split temporal 70/30
    PASO 6  Para cada K:
              - Calcular t_norm sobre calibracion
              - Por ticker: p05, p95, pct_90, pct_95, IQR
              - Agregar ticker-balanced
              - Evaluar H1-H4
    PASO 7  Filtrar K con H1-H4 pass
    PASO 8  Seleccionar MAYOR K
    PASO 9  Validar sobre 30%: V1-V4 obligatorios
    PASO 10 Reportar diagnosticos (D1-D3 cal, p05/p95/skew OOS)
    PASO 11 Si V1-V4 OK y calibracion OK: congelar K
    PASO 12 Si falla: grid 5b.2

Vistas:
    Primaria: Ticker-balanced (mediana entre tickers)
    Sensibilidad: Observation-pooled (concatenado)

Fin del protocolo 5b v2.
