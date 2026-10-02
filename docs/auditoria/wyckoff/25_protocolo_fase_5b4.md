# PROTOCOLO FASE 5b.4 - Landmark temporal (BORRADOR)

**Documento normativo en borrador. NO ejecutable hasta dictamen.**
**Fecha:** 2026-10-02.
**Precedente:** 24_dictamen_5b3_D3.md.
**Estado:** pendiente de validacion por auditor externo.

---

## 0. Contexto

5b.3 fallo formalmente (D3=0/240) y, al revisarla, se detecto
inmortal time bias en el anclaje del outcome. El dictamen 24 define
las correcciones obligatorias. Este protocolo las implementa.

Objetivo de 5b.4: determinar si el SOW anade informacion sobre
deterioro futuro **mas alla de candidate**, con anclaje temporal
simetrico y criterio de seleccion definido ex-ante.

---

## 1. Definiciones

### 1.1. Unidad de observacion

    candidate episode = (candidate_t=True) AND (candidate_{t-1}=False)
    por ticker

### 1.2. Landmark

    L = t0 + M

donde t0 es el indice de entrada al episodio (fecha del candidate).
M es el parametro de ventana (grid).

### 1.3. Grupo confirmed

    confirmed = existe SOW_t = 1 para algun t en [t0, L]

Donde SOW se calcula con la formula v1.7 (ATR ex-ante + X_ATR +
Y_VOL), contrato v1.8.

### 1.4. Grupo baseline

    baseline = NO existe SOW_t = 1 en [t0, L]

### 1.5. Outcome

**Ambos grupos miden el outcome desde L.**

    struct_deterioration_H{h} = struct[L+h] < struct[L]
    price_weakness_H{h}       = close[L+h] < close[L] * (1 - 0.02)
    below_support_H{h}        = close[L+h] < support[L]
    lower_low_H{h}            = min(close[L+1:L+h]) < min(close[L-N:L])

donde support[L] = rolling_min(Low, N).shift(1) evaluado en L.

Primario: `struct_deterioration_H20`.

### 1.6. Nota semantica

`struct_deterioration_H20` es comparacion punto-a-punto:
struct[L+20] < struct[L]. NO implica deterioro continuo. No
introducir metrica de continuidad en 5b.4.

---

## 2. Diferencia con 5b.3

    CONFIRMED:
    t0 -> ventana M -> L
    outcome = struct[L + H] < struct[L]

    BASELINE:
    t0 -> ventana M -> L
    outcome = struct[L + H] < struct[L]

Mismo reloj. Elimina el sesgo.

**Consecuencia:** ya no se exige M <= H. M=30 con H=20 es valido,
porque L = t0+30 y outcome = struct[L+20], siempre completamente
posterior a L sin leakage.

---

## 3. Grid propuesto

Los mismos 4 parametros que 5b.3. Ahora tambien con landmark:

    N     in {20, 30, 40, 60}
    M     in {5, 10, 15, 20, 30}
    X_ATR in {0.25, 0.50, 0.75, 1.00}
    Y_VOL in {1.10, 1.20, 1.50}

Grid total: 240 combinaciones. Mismo tamanio, diseno corregido.

---

## 4. Universo y split

- Fuente: `data/stock_prices.parquet`.
- Universo: 315 tickers con >= 200 obs Close no-NaN (SPCX fuera).
- Periodo: 2021-10-01 hasta ultima sesion cerrada.

### 4.1. Bloques temporales (nuevo diseno)

En lugar de calibracion/holdout 70/30, **division en K bloques
temporales** para sensibilidad por regimen (dictamen 15):

    bloque A  (mas antiguo)
    bloque B
    bloque C
    bloque D  (mas reciente)

Numero y fronteras exactas: **PROPUESTO** abajo, requiere validacion.

Propuesta:

    2021-10-01 -> 2022-12-31   bloque A  (mercado bajista 2022)
    2023-01-01 -> 2024-03-31   bloque B  (recuperacion)
    2024-04-01 -> 2025-08-31   bloque C  (tendencia)
    2025-09-01 -> 2026-10-01   bloque D  (post-recuperacion)

### 4.2. Holdout verdaderamente ciego

El holdout 2025-03-31 / 2026-10-01 de 5b.3 ya ha sido inspeccionado.
**NO se reutiliza como validacion final.**

Opciones para validacion final (requiere dictamen):

    a) Reservar bloque D (2025-09-01 -> 2026-10-01) como holdout ciego.
       No ha sido inspeccionado en 5b.3 (aunque la frontera 2025-03-31
       del split calibro parte de el).
    b) Esperar ventana futura (post 2026-10-02) para validacion
       completamente ciega. Coste temporal alto.
    c) Reservar bloque A+B+C como desarrollo, bloque D como validacion.
       Mismo problema: bloque D ya fue parcialmente visto en 5b.3.

**Recomendacion del equipo:** opcion (a), asumiendo que el bloque D
solo fue visto agregado en el top-10 holdout de 5b.3, no por sesion.
Requiere validacion del auditor.

---

## 5. Inferencia

Debido a multiples episodios por ticker, no tratar como observaciones
independientes.

Metodo propuesto:

    cluster bootstrap por ticker
    B = 1000 muestras
    IC 95% del lift_H20 = percentiles [2.5, 97.5] de la distribucion

Criterio:

    lower_CI(lift_H20) > 0

Requiere validacion del auditor:
- B (numero de iteraciones).
- Semilla fija para reproducibilidad.
- Tratamiento de tickers con multiples episodios (bootstrap por
  ticker completo vs por episodio con reweight).

---

## 6. Criterios de seleccion ex-ante

Antes de mirar resultados, definir:

### 6.1. Criterios duros propuestos

    D1: n_sow_raw >= 100
    D2: n_confirmed >= 20 y n_baseline >= 20 (ambos grupos con muestra)
    D3: lower_CI(lift_H20) > 0 (inferencia cluster)
    D4: numero de bloques con lift > 0 >= 3 de 4

### 6.2. D3_diagnostic (no puerta)

Se reporta pero no selecciona:

    D3_diag = pct_conf_struct_H20
    D3_base = pct_base_struct_H20
    lift_pp = (D3_diag - D3_base) * 100

### 6.3. Ranking entre aceptados

Antes de mirar:

    Paso 1: pasar D1-D4.
    Paso 2: mayor mediana de lift_H20 entre bloques.
    Paso 3: menor desviacion estandar de lift_H20 entre bloques.
    Paso 4 (desempate): mayor X_ATR > mayor Y_VOL > N ascendente >
                        M ascendente.

**Nota:** el criterio por mediana (no media) y por baja desviacion
refleja el dictamen 15 (estabilidad temporal > pico aislado).

---

## 7. Diagnostico H1/H3

Se mantiene la verificacion SLB/PCG: los falsos positivos marginales
del 5c.3 no deben confirmar.

---

## 8. Analisis prohibidos

- Optimizar por n_confirmed.
- Optimizar por numero de DISTRIBUTION.
- Seleccionar mirando el top de bloques sin criterio ex-ante.
- Reutilizar el holdout 2025-03-31/2026-10-01 como validacion final
  ciega.
- Relajar D3 retroactivamente.
- Modificar este protocolo tras ver resultados.
- Anclar confirmed y baseline en fechas distintas.
- Calcular outcome sin usar el mismo L para ambos grupos.

---

## 9. Salida esperada

- `outputs/audit/wyckoff_5b4_grid.csv` (240 filas).
- `outputs/audit/wyckoff_5b4_summary.json`.
- `26_calibracion_5b4_resultados.md`.
- `27_expediente_5b4_<tema>.md` si surgen dudas.

---

## 10. Criterios de exito

5b.4 tiene exito si:

- Al menos una combinacion pasa D1-D4.
- La combinacion elegida sobre bloque de desarrollo generaliza en el
  bloque de validacion (lower_CI(lift_H20) > 0 en validacion).
- Estabilidad entre bloques: >= 3 de 4 con lift > 0.
- SLB no confirma.

Si ninguna combinacion pasa: 5b.4 FAIL. Se abre 5b.5 con otra
hipotesis o se abandona la linea SOW como confirmador.

---

## 11. Huecos para el auditor

Los siguientes puntos son decisiones del auditor, no del equipo:

**H1.** Fronteras exactas de los 4 bloques (seccion 4.1).
**H2.** Eleccion del bloque de validacion ciega (seccion 4.2).
**H3.** Numero B de bootstrap y semilla (seccion 5).
**H4.** Tratamiento bootstrap por ticker con multiples episodios
(seccion 5).
**H5.** Umbral D4: 3 de 4 bloques con lift > 0. Alternativa: exigir
lower_CI > 0 en cada bloque individual.
**H6.** Ranking paso 2: mediana vs minimo vs media truncada.
**H7.** Confirmacion de los 4 criterios duros D1-D4.
**H8.** Cualquier modificacion adicional que el auditor considere
necesaria.

---

## 12. Estado

    v1.8              CONTRATO VIGENTE
    5b.3              FAIL (D3 = 0/240, sesgo temporal adicional)
    5b.4 (borrador)   ESTE DOCUMENTO, pendiente dictamen
    Parametros SOW    None en config
    5c.4, 5d          BLOQUEADAS
    Legacy            INTACTO

---

**Fin del borrador 25.**