# Re-acumulacion de continuacion - pre-registro v1

**Estado:** CONGELADO antes de codificar. No se ajusta tras ver
resultados.
**Fecha:** 2026-10-27.
**Sesion:** 13.
**Item:** DICT3-REACCUMULATION (REGISTRO.yaml, OPEN).
**Base:** doc 68 seccion re-acumulacion.

---

## 1. Proposito

El detector actual solo reconoce reversion:

- Acumulacion: SC -> AR -> Range -> SOS.
- Distribucion: BC -> AR -> Range -> SOW.

Ambas requieren capitulacion (SC/BC) previa. En tendencia alcista
sana sin capitulacion, el detector no marca acumulacion. Ejemplo:
XLI sube 2023-2026 sin SC confirmado -> 0 acumulaciones por diseno
Wyckoff clasico. Doc 68 lo documenta.

El objetivo es un detector de RE-ACUMULACION: consolidacion dentro
de tendencia alcista que se rompe al alza. Familia separada de las
cascadas existentes.

---

## 2. Definicion

Sea un ticker con serie de cierres C(t). Un episodio de
re-acumulacion en t = as_of cumple:

R1. Contexto previo UP: la tendencia en t-L_prev (L_prev=60 sesiones)
    es UP segun detect_trend_context. No se exige SC previo.

R2. Consolidacion: existe una ventana [t-W, t-1] (W=40) en la que el
    rango (max-min)/min <= RANGE_TOLERANCE_FRAC (mismo umbral que el
    detector de reversion: 0.05).

R3. Prueba de oferta agotada: dentro de la ventana, al menos un
    retroceso toca el soporte del rango (close <= support + tol)
    con volumen relativo v_norm <= -0.3 (mismo umbral que SC).

R4. Ruptura confirmada: en t, close > resistance + tol durante
    al menos SOS_CONFIRM_CLOSES=3 cierres consecutivos (mismo
    umbral que SOS de reversion).

R5. Volumen en ruptura: en el cierre de ruptura, v_norm >= +0.5
    (mismo umbral que SC, invertido).

La fase resultante es `REACCUMULATION_CONFIRMED`. No se mezcla con
`ACCUMULATION_CONFIRMED`: familia separada.

---

## 3. Criterios de aceptacion (pre-definidos, no ajustables)

Se corre sobre holdout: data/stock_prices_reserved_cohort.parquet
(268 tickers S&P 500), nunca usados para codificar el detector.

**C1 - Frecuencia no extrema.** El numero de REACCUMULATION_CONFIRMED
no debe ser ni 0 (detector roto) ni >30% de los tickers por ano
(detector laxo). Rango aceptable: 1-30% de tickers del holdout por
ano con al menos un episodio.

**C2 - Discriminacion.** Tasa de acierto a 3 meses: tras un
REACCUMULATION_CONFIRMED, el precio debe subir >= 5% en los 63
siguientes dias habiles en >=60% de los casos. Si acierta <=50%
(moneda al aire), el detector no aporta.

**C3 - No canibaliza.** El detector NO debe solaparse con
ACCUMULATION_CONFIRMED en mas del 10% de los episodios. Si
REACCUMULATION aparece donde ya hay ACCUMULATION, el nuevo detector
no aporta informacion nueva.

Aceptar <=> C1 ^ C2 ^ C3. Si falla cualquiera, se descarta.

---

## 4. Lo que NO hace

- No modifica las cascadas existentes (ACCUMULATION, DISTRIBUTION).
- No modifica classify_wyckoff_phase.
- No modifica el orquestador actual (wyckoff_orchestrator.py).
- No toca WYCKOFF_SOW_*.
- No se aplica al reporte hasta validacion completa.
- No usa `datetime.now()` para observacion.

---

## 5. Compromiso

Tras correr el holdout, no se ajustan los parametros. Si falla,
se documenta el fallo y el item DICT3-REACCUMULATION se cierra
como WONT_FIX con la razon medida. Si pasa, se aplica en una sesion
posterior con tests de contrato T1-T6 (uno por cada regla R1-R5).

---

## 6. Parametros

| Parametro | Valor | Justificacion |
|---|---|---|
| L_prev | 60 sesiones | Coherencia con SC_LOOKBACK |
| W | 40 sesiones | Coherencia con ST_MAX_WINDOW |
| RANGE_TOLERANCE_FRAC | 0.05 | Mismo que reversion |
| SOS_CONFIRM_CLOSES | 3 | Mismo que reversion |
| v_norm retroceso | <= -0.3 | Mismo que SC |
| v_norm ruptura | >= +0.5 | Mismo que SC (invertido) |
| Tasa de acierto | 3 meses (63 sesiones) | Ventana estandar de reversion |

Ninguno se optimiza. Cualquier cambio requiere v2 del pre-registro.

---

## 7. Estado normativo

Borrador congelado. Sin validez hasta que exista el detector en
codigo y el holdout se corra contra este pre-registro. Cualquier
desviacion entre el codigo y este documento invalida el resultado.

---

Fin del pre-registro 59 (congelado).