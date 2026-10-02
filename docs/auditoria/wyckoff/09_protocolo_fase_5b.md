# PROTOCOLO FASE 5b - Calibracion de K

**Documento normativo. Define el procedimiento de calibracion del
parametro K de `t_norm = tanh(trend / K)`.**
**Fecha:** 2026-10-02.
**Aprobacion requerida:** auditor externo (aprobacion arquitectonica de
v1.3 emitida 2026-10-02). Este protocolo debe aprobarse ANTES de
ejecutar ningun experimento.
**Referencia:** dictamen P1 + revision c_norm/e_norm (§14 del dictamen).

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

## 2. Principios de calibracion

**P1. Separacion calibracion/validacion.**
La eleccion de K debe fijarse antes de observar que fases produce.
Si K cambia el numero de MARKUP/ACCUMULATION/etc., ese numero NO es
criterio de eleccion.

**P2. Anclaje sobre `trend`.**
K se calibra sobre la distribucion historica de `trend = MA50/MA200 - 1`
en el universo objetivo, no sobre la salida del clasificador.

**P3. Criterio ex-ante.**
Las metricas y umbrales de aceptacion se fijan en este documento. No
pueden modificarse a posteriori para justificar un K concreto.

**P4. Grid predefinida.**
Los candidatos de K son los listados en §4. No se anade un K ad-hoc
porque "queda bien".

**P5. Congelacion.**
K se congela tras aplicar §6. Cualquier cambio posterior abre un nuevo
ciclo de calibracion (5b.2).

**P6. Validacion independiente.**
El K elegido se valida sobre un periodo distinto al de calibracion
(§7). Si falla, vuelve a 5b.

---

## 3. Entrada

### 3.1. Dataset

- **Fuente:** `data/stock_prices.parquet` (universo USA + Europa).
- **Campo:** `Close` por ticker.
- **Universo:** todos los tickers con historia suficiente para calcular
  MA200 (>= 200 observaciones validas).
- **No se filtran por sector ni por cap.** El universo es el completo.

### 3.2. Periodo de calibracion

- **Inicio:** 2021-10-01 (fecha de inicio del parquet actual).
- **Fin:** se fija en el momento de la ejecucion (el "hoy" del pipeline).
- **Razon:** cobertura completa del parquet. No se inventa un corte
  arbitrario que pudiera sesgar la distribucion.

### 3.3. Periodo de validacion

- **Metodo:** walk-forward. Se reserva el 30% final de la serie para
  validacion. El 70% inicial es calibracion.
- **Justificacion:** 70/30 es un corte habitual en calibracion robusta
  sin optimizacion (suficiente para estimar distribucion; suficiente
  para validar estabilidad).
- **Nota:** si el corte 70/30 produce <200 sesiones de validacion, se
  usa 80/20.

---

## 4. Candidatos de K

Grid predefinida:

    K in {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40}

Siete valores. Rango razonable:

- K=0.10: agresivo, saturacion temprana.
- K=0.40: laxo, buena resolucion pero t_norm apenas alcanza valores
  extremos (bull markets muy fuertes).

**Prohibido:** anadir K fuera de esta grid. Si los resultados sugieren
que el optimo esta entre dos valores, se interpola en 5b.2 (nuevo ciclo).

---

## 5. Metricas

Para cada K, sobre la serie de `t_norm` (no sobre las fases):

### 5.1. Metricas de saturacion

- `pct_extreme_90`: % de observaciones con `|t_norm| > 0.90`.
- `pct_extreme_95`: % de observaciones con `|t_norm| > 0.95`.
- `max_abs_t_norm`: maximo absoluto observado.

### 5.2. Metricas de resolucion

- `p05, p25, p50, p75, p95`: percentiles de `t_norm`.
- `iqr`: rango intercuartilico (`p75 - p25`).
- `std_t_norm`: desviacion estandar.

### 5.3. Metricas de simetria

- `skewness`: asimetria.
- `pct_positivos`: % observaciones con `t_norm > 0`.

### 5.4. Metricas de estabilidad temporal

- `std_por_anio`: desviacion estandar de `t_norm` por anio calendario
  (mide si el regimen cambia de forma dependiente del anio).
- `rango_p50_por_anio`: max-min de la mediana anual.

### 5.5. Metricas de comparabilidad cross-sectional

- `resolucion_en_el_top`: la diferencia minima detectable entre dos
  tickers ordenados por `t_norm` en una fecha dada.
- `pct_valores_distintos`: % de valores unicos en el p50 de una fecha.

---

## 6. Criterios de aceptacion (ex-ante)

Un K se acepta si cumple **todos** los siguientes:

### 6.1. Saturacion acotada

    pct_extreme_90 < 15%
    pct_extreme_95 < 5%

Razon: si mas del 15% de las observaciones caen en `|t_norm| > 0.90`,
la senal pierde resolucion en la cola.

### 6.2. Rango util

    p05 <= -0.40  AND  p95 >= +0.40

Razon: la distribucion de `t_norm` debe cubrir un rango amplio, no
concentrarse en un intervalo estrecho.

### 6.3. Simetria razonable

    |skewness| < 1.0

Razon: `t_norm` deberia ser aproximadamente simetrico si el universo
tiene tanto tendencias alcistas como bajistas. Asimetria fuerte sugiere
sesgo del universo o del parametro.

### 6.4. Estabilidad temporal

    rango_p50_por_anio < 0.40

Razon: la mediana anual de `t_norm` no debe cambiar drasticamente. Un
cambio grande sugiere que K no es robusto entre regimenes.

### 6.5. Resolucion cross-sectional

    pct_valores_distintos > 30%

Razon: en una fecha dada, al menos el 30% de los tickers deben tener
valores de `t_norm` distintos a nivel de 2 decimales. Por debajo de
eso, la señal no discrimina.

### 6.6. Seleccion entre candidatos aceptados

Si varios K cumplen §6.1-6.5, se elige el **mas pequeno** (mas
conservador respecto a la sensibilidad de trend; más cerca de la escala
natural).

Si ningun K cumple, Fase 5b falla. Se abre ciclo 5b.2 con grid
extendida.

---

## 7. Validacion independiente

Una vez elegido K sobre el 70% de calibracion:

1. Se calcula `t_norm` sobre el 30% de validacion.
2. Se aplican §6.1-6.5 sobre la serie de validacion.
3. **Criterio:** al menos 4 de 5 criterios deben pasar.
4. Si pasan <4: el K no generaliza. Fase 5b falla.

**Nota:** este paso valida que K no esta sobreajustado al periodo de
calibracion.

---

## 8. Congelacion

Si 5b tiene exito:

1. `WYCKOFF_T_NORM_K` en `config/settings.py` se actualiza al valor
   elegido.
2. `02_proveniencia_parametros.md` pasa el parametro de `PROPUESTO` a
   `CAL` con la siguiente informacion:
   - valor final de K;
   - fecha de calibracion;
   - periodo de calibracion y validacion;
   - metricas obtenidas;
   - criterio de eleccion (el mas pequeno de los aceptados).
3. Commit documental + codigo en una sola operacion.

---

## 9. Sensibilidad exploratoria (permitida)

Ademas del protocolo estricto, se permite un analisis exploratorio
**etiquetado como tal** (no forma parte de 5b):

- Rango amplio de K (0.05 a 0.50, pasos de 0.05).
- Sin criterios de aceptacion.
- Solo para observar:
  - curvas de saturacion vs K;
  - percentiles por K;
  - sensibilidad de fases por K.

**Prohibiciones:**

- No puede usarse para elegir K.
- No puede llamarse 5b ni 5c.
- Debe quedar registrado en el expediente como analisis exploratorio.

Razon: separar "conocer el terreno" de "decidir".

---

## 10. Analisis permitidos durante 5b

- Distribucion de `trend` en el universo y periodo.
- Comparacion de los 7 candidatos de K sobre §5.
- Analisis de la sensibilidad de `t_norm` a cambios de K.

## 11. Analisis NO permitidos durante 5b

- Elegir K maximizando el numero de MARKUP (o de cualquier fase).
- Elegir K segun los 4 tickers MSFT/PLTR/INTC/AMD.
- Elegir K segun el resultado sobre un anio concreto.
- Elegir K segun backtest de WLS.
- Modificar criterios ex-ante tras ver resultados.

---

## 12. Salida esperada

Al cerrar 5b:

- `docs/auditoria/wyckoff/10_calibracion_K_resultados.md` con:
  - Tabla de K vs metricas §5.
  - K elegido.
  - Justificacion segun §6.
  - Resultado de validacion §7.
- Actualizacion de `config/settings.py`.
- Actualizacion de `02_proveniencia_parametros.md`.

---

## 13. Criterios de fallo de Fase 5b

5b falla si:

- Ningun K de la grid cumple §6.
- El K elegido no pasa §7 (validacion independiente).
- Los resultados sugieren que la formula `tanh(trend/K)` es
  insuficiente (p.ej. la distribucion de `trend` tiene una forma que
  ninguna K puede normalizar).

En caso de fallo, se abre ciclo 5b.2 (nueva grid) o se eleva al
auditor una revision de la formula.

---

## 14. Estado del proyecto tras 5b

Si 5b tiene exito:

    Fases 1-3      CERRADAS
    Fase 4         CERRADA
    Fase 5a        IMPLEMENTADA (v1.3)
    Fase 5b        CERRADA (K congelado)
    Fase 5c        DESBLOQUEADA (comparativa legacy v1.3)
    Fase 5d        BLOQUEADA (migracion)

Si 5b falla:

    Fases 1-3      CERRADAS
    Fase 4         CERRADA
    Fase 5a        IMPLEMENTADA (v1.3)
    Fase 5b        FALLIDA
    Fase 5b.2      ABIERTA (nueva grid o revision formula)
    Fase 5c        BLOQUEADA
    Fase 5d        BLOQUEADA

---

## 15. Anexo - Procedimiento paso a paso

    PASO 1  Cargar stock_prices.parquet
    PASO 2  Filtrar tickers con >= 200 obs validas
    PASO 3  Calcular trend para cada ticker (MA50/MA200 - 1)
    PASO 4  Construir la serie agregada de trend (stack de todos los
            tickers)
    PASO 5  Split 70/30 (calibracion/validacion)
    PASO 6  Para cada K en la grid:
              - Calcular t_norm sobre calibracion
              - Calcular metricas §5
              - Evaluar criterios §6
    PASO 7  Seleccionar K segun §6.6
    PASO 8  Validar sobre 30% (walk-forward) segun §7
    PASO 9  Congelar K segun §8
    PASO 10 Documentar resultados segun §12

---

Fin del protocolo 5b.
