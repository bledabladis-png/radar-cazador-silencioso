# PROTOCOLO FASE 5b - Calibracion de K

**Documento normativo. Define el procedimiento de calibracion del
parametro K de `t_norm = tanh(trend / K)`.**
**Fecha:** 2026-10-02.
**Version:** v2 (revisada por dictamen externo 2026-10-02).
**Estado:** APROBADA PARA EJECUCION tras aplicar las correcciones del
dictamen.

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

## 6. Criterios

### 6.1. Criterios DUROS (H1-H4) - todos obligatorios

    H1  pct_extreme_90 < 15%
    H2  pct_extreme_95 < 5%
    H3  p05 <= -0.40
    H4  p95 >= +0.40

Forman el nucleo: no saturar excesivamente + mantener amplitud util.

### 6.2. Criterios DIAGNOSTICOS (D1-D3) - informativos

    D1  |skewness| < 1.0
    D2  rango_p50_por_anio < 0.40
    D3  mediana_IQR_cs >= 0.05

Se reportan. **NO invalidan K automaticamente.** Si fallan, se abre
observacion para investigacion, pero K sigue valido si cumple H1-H4.

Razon: pueden ser propiedades reales del universo observado, no
necesariamente defectos de K.

### 6.3. Regla de seleccion

**Entre los K que cumplan TODOS los criterios duros H1-H4:**

    seleccionar el MAYOR K.

**Justificacion matematica:** `K` mayor -> `trend/K` menor -> saturacion
menor. El "mayor K que aun cumple el rango minimo" es el que **maximiza
reduccion de saturacion** sin perder resolucion exigida. NO es "el mas
conservador".

Contraejemplo del protocolo v1: "el mas pequeno" seria **mas
saturador**, no mas conservador.

### 6.4. Si ningun K cumple H1-H4

Fase 5b falla. Se pasa a grid 5b.2 (predefinida). No se inventan
valores.

---

## 7. Validacion en holdout temporal (30% final)

Una vez elegido K sobre el 70%:

1. Se calcula `t_norm` sobre el 30% no usado.
2. Se aplican H1-H4 sobre la serie de validacion.
3. **H1-H4: TODOS obligatorios.** No se admite "4 de 5".
4. D1-D3: diagnosticos. Se reportan, no invalidan.

**Si H1, H2, H3 o H4 falla en validacion -> 5b FALLA.**

Razon: la escala debe generalizar. Si la saturacion solo se cumple en
calibracion, K esta sobreajustado al periodo.

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

## 14. Procedimiento paso a paso

    PASO 1  Cargar stock_prices.parquet
    PASO 2  Determinar ultima sesion cerrada por ticker
    PASO 3  Filtrar tickers con >= 200 obs validas
    PASO 4  Calcular trend para cada ticker
    PASO 5  Split temporal 70/30 (calibracion/validacion)
    PASO 6  Para cada K en grid primaria:
              - Calcular t_norm sobre calibracion
              - Calcular H1-H4 (balanced y pooled)
              - Calcular D1-D3
    PASO 7  Filtrar K que cumplen H1-H4 en balanced
    PASO 8  Seleccionar el MAYOR K
    PASO 9  Si ningun K: grid 5b.2
    PASO 10 Validar K elegido sobre 30% (H1-H4 obligatorios)
    PASO 11 Si falla validacion: 5b falla
    PASO 12 Congelar K + documentar + commit

---

Fin del protocolo 5b v2.
