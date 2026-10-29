# PRE-REGISTRO DICT3-REACCUMULATION v2

**Estado:** v2. Pendiente de revision y firma del auditor externo.
**Fecha:** 2026-10-29.
**Ciclo:** sesion 16.
**Frente:** DICT3-REACCUMULATION (REGISTRO.yaml, OPEN).
**Base:** v1 (commit 0bff883f) + dictamen externo 2026-10-29 (chat).
**Regla:** los parametros de esta v2 quedan congelados antes de
ejecutar el experimento. No se ajustan tras ver resultados.

Responde a los 9 hallazgos del dictamen (P0-H01, P0-H02, P0-H03,
H-04, H-05, H-06, C1, C2, C3) y corrige la salida y los tests.

---

## 0. Objetivo y alcance

Detectar REACUMULACION de continuacion: consolidacion en tendencia
alcista sana que se rompe al alza, sin que el detector actual de
reversion (SC->AR->Range->SOS) la capture.

La etiqueta `REACCUMULATION_CONFIRMED` es salida del detector
experimental. NO es una fase nueva del sistema, NO modifica
`classify_wyckoff_phase` ni el orquestador, NO se publica como fase
en el reporte diario. Vive en un CSV experimental aislado.

---

## 1. P0-H01: estado del holdout (RESUELTO)

- `stock_prices_reserved_cohort.parquet` y `_r3000.parquet` fueron
  consumidos por SOW-V6 el 2026-10-06. No son muestra nueva.
- Se registran en `data/mappings/holdout_registry.json` como
  `CONSUMED_FOR_CANDIDATE(SOW-V6)` con `event_type =
  EXPLORATORY_DESIGN_INFLUENCE` (H-06 dictamen doc 76).
- La poblacion es composicion 2026Q4 del S&P 500 aplicada
  retrospectivamente. Sesgo de supervivencia conocido.
- Resultados sobre este holdout se clasifican como EXPLORATORIOS o
  evidencia secundaria, nunca confirmacion independiente.
- Para habilitar validacion confirmatoria y eventual incorporacion
  a produccion se reserva un periodo futuro: datos posteriores a
  la firma de esta v2, con protocolo congelado antes de abrir la
  ventana. Ese periodo se define en §10.

Restriccion ejecutable: `holdout_registry.json` ya bloquea
presentar un resultado de este holdout como confirmatorio. Cualquier
`capture_experiment_env --confirmatory` sobre estos parquets con la
candidata REACCUMULATION lo registrara como `EXPLORATORY`.

---

## 2. P0-H02: anclaje temporal de la ruptura (RESUELTO)

Sea `C(t)` la serie de cierres. Sea `W = 40` sesiones.

**Definicion.** Sea `b` la primera sesion que cierra por encima del
umbral de ruptura. Las magnitudes se congelan a partir de la
ventana anterior a `b` y NO se recalculan con los cierres de la
propia secuencia de ruptura.

Formulas:
    S_b = min(C(b-W), ..., C(b-1))
    R_b = max(C(b-W), ..., C(b-1))
    w_b = R_b - S_b
    tol_b = 0.05 * w_b
    ruptura_umbral_b = R_b + tol_b

**Confirmacion.** Se exige:
    C(b)   > ruptura_umbral_b
    C(b+1) > ruptura_umbral_b
    C(b+2) > ruptura_umbral_b

**Fecha de la etiqueta.** El evento solo existe a partir de `b+2`.
La fecha registrada en el CSV es `date_start = b`,
`date_confirmed = b+2`. La evaluacion de los 63 dias posteriores
(C2) se hace desde `date_confirmed`.

**Volumen de ruptura.** Se evalua en la sesion `b` (primera
ruptura). `v_norm(b)` se calcula sobre la media y desviacion de
volumen de la ventana [b-W, b-1].

**Volumen de retroceso (R3).** Ver H-05.

---

## 3. P0-H03: umbrales de volumen (RESUELTO)

Correccion de la justificacion (el dictamen detecta que v1 mentia
al presentarlos como heredados de SC):

- `v_norm <= -0.3` es un umbral NUEVO. NO es el umbral de volumen
  de SC. El valor -0.3 corresponde a `SC_C_NORM_THRESHOLD`
  (componente de compresion), no a `SC_V_NORM_THRESHOLD = 0.5`.
- `v_norm >= +0.5` coincide numericamente con `SC_V_NORM_THRESHOLD`,
  pero NO es "el umbral de SC invertido" (no hay inversion
  matematica: -0.3 != -0.5). Es coincidencia de magnitud.

Ambos valores son candidatos experimentales fijados aqui antes de
ejecutar. No se optimizan despues de ver resultados. Si el
experimento no funciona con estos valores, se documenta el fallo
sin recalibrar (regla del dictamen).

---

## 4. H-04: formulas de consolidacion (RESUELTO)

Dado el anclaje `b` de §2:

    S_b, R_b, w_b, tol_b   -> formulas de §2
    Consolidacion valida si:
        (R_b - S_b) / R_b <= 0.15    (anchura relativa max 15%)
        R_b > S_b > 0                (evita ventanas degeneradas)

Tratamiento de datos:
- `C(t)` es Close diario ajustado por splits (auto_adjust=False
  sobre Yahoo; ajuste manual de splits si el proveedor no lo aplica).
- Si alguna sesion de [b-W, b-1] tiene Close NaN o <= 0, el evento
  NO es evaluable. No se imputa.
- Si hay menos de `W` sesiones disponibles (historial insuficiente),
  el ticker no es evaluable en esa fecha.

Decision explicita: el detector usa Close, no rango High-Low
intradia. Es una eleccion de diseño, coherente con las cascadas
existentes que tambien cierran sobre Close.

---

## 5. H-05: pullback (RESUELTO)

R3 exige retroceso con volumen bajo. Definicion operativa:

Sea `S_b` el soporte congelado de §2. Un **pullback** es una sesion
`p` en el rango `[b+1, b+2]` que cumple simultaneamente:

    C(p) <= S_b + tol_b           (toca o queda bajo soporte + tol)
    v_norm(p) <= -0.3             (volumen relativo bajo)

**La condicion de contacto y la de volumen bajo deben ocurrir en la
MISMA sesion `p`.** No se admite contacto en `p` y volumen bajo en
`q != p`. Esto elimina la interpretacion discrecional.

Si no existe tal `p` en `[b+1, b+2]`, no hay pullback y el evento no
se etiqueta (aunque haya ruptura confirmada). R3 es obligatoria.

`v_norm(p)` se calcula con media/desv de volumen de [b-W, b-1]
(la misma ventana que §2, no recalculada).

---

## 6. H-06: contexto UP (RESUELTO)

R1 exige contexto de tendencia alcista. Definicion:

    contexto_up = detect_trend_context(C, as_of=b - L_prev)
    L_prev = 60

Se evalua en `b - 60`, es decir: **el contexto se calcula con
informacion estrictamente anterior al inicio de la ruptura**. Nunca
incluye `C(b)` ni posteriores.

Distincion de fechas en el CSV:
- `date_context = b - 60`   (fecha de evaluacion del contexto)
- `date_start = b`          (primera ruptura)
- `date_confirmed = b+2`    (confirmacion)
- La evaluacion C2 (63 sesiones posteriores) parte de `date_confirmed`.

No hay look-ahead: en `date_confirmed` toda la informacion usada ya
era conocida.

---

## 7. C1: frecuencia (RESUELTO)

- Universo: 268 tickers de cohort. Solo entran los tickers con
  historial suficiente para evaluar R1-R4 en la ventana del año.
- **Denominador por año:** numero de tickers EVALUABLES en ese año,
  no los 268 fijos.
- Años parciales (< 252 sesiones completas): se excluyen del computo
  de C1 y se marcan como `PARCIAL` en el informe.
- Regla del 1%-30%: se aplica sobre el denominador de tickers
  evaluables de cada año completo. Un año con 0 evaluables no
  cuenta. Un año con evaluables y 0 episodios sí cuenta como 0% y
  puede rechazar C1 si el resto tampoco compensa.

No se exige aparicion en todos los años. Se exige que la tasa anual
este en [1%, 30%] en al menos `N_anios_validos - 1` años, donde
`N_anios_validos` es el numero de años completos evaluados. Esto
evita que un año sin oportunidades rechace un detector valido, pero
evita que un unico año con detecciones sostenga el criterio.

---

## 8. C2: tasa de acierto (RESUELTO)

- **Precio de rentabilidad:** Close de `date_confirmed` como base,
  Close de `date_confirmed + 63 sesiones` como final. Splits
  ajustados. Dividendos no reinvertidos (Close price return, no
  total return).
- **Episodios evaluables:** solo los que tienen 63 sesiones futuras
  disponibles. Los que no, se excluyen del denominador.
- **Solapamiento:** dos episodios del mismo ticker con
  `date_confirmed` a menos de 63 sesiones de distancia se tratan
  como un unico episodio (se conserva el primero). Evita doble
  conteo.
- **Tamaño muestral minimo:** `N_episodios_evaluables >= 30`. Con
  menos, C2 = `INSUFFICIENT_SAMPLE`.
- **Umbral de exito:** tasa puntual `>= 0.60` Y limite inferior del
  IC95 (Wilson o bootstrap percentil, seed fija = 42) `> 0.50`.
- Si `N < 30`, el resultado se marca `INSUFFICIENT_SAMPLE` y NO
  cuenta como exito ni como fallo. Se requiere ampliar muestra o
  cerrar como no concluyente.

El valor 0.60 es umbral de relevancia practica. La condicion
`IC95_lower > 0.50` es la que distingue el resultado del azar.

---

## 9. C3: solapamiento con acumulacion (RESUELTO)

- **Numerador:** numero de episodios de reacumulacion cuya ventana
  `[date_start, date_confirmed]` intersecta con al menos un episodio
  `ACCUMULATION_CONFIRMED` del mismo ticker.
- **Denominador:** total de episodios de reacumulacion evaluables
  para esta comparacion (los que tienen acumulacion evaluable en el
  mismo periodo).
- **Regla de coincidencia:** interseccion temporal no vacia de
  intervalos, con el contrato vigente de acumulacion para delimitar
  el episodio existente (SC->AR->Range->SOS, fechas SC y SOS).
- **Umbral:** `C3 <= 0.10`.
- Si el denominador no es calculable (ningun episodio con
  acumulacion evaluable en el periodo), C3 = `NO_EVALUABLE` y NO
  se da por superado. No se asume cumplimiento por ausencia de
  datos.

---

## 10. Periodo futuro reservado (para confirmacion)

Para que un resultado favorable en cohort habilite validacion
confirmatoria, se reserva el periodo:

    desde 2026-11-01 hasta el cierre del año fiscal siguiente
    (2027-10-31), con protocolo congelado antes de abrir la
    ventana.

Condiciones:
- La candidata (parametros de §2-§6) queda congelada con la firma
  de esta v2.
- El periodo se declara `RESERVED_FUTURE` en `holdout_registry.json`
  cuando se abra, con `experiment_manifest` capturado al inicio de
  la ventana.
- Cualquier ajuste de parametros durante la ventana invalida la
  confirmacion y degrada el resultado a exploratorio.

Sin firma de esta v2 y sin la ventana abierta, no hay confirmacion
posible.

---

## 11. Salida y persistencia (corregido)

Salida del detector: CSV `outputs/audit/reaccumulation_events.csv`
con columnas:

    ticker, date_context, date_start, date_confirmed,
    S_b, R_b, w_b, tol_b, v_norm_b, v_norm_p, pullback_session,
    contexto_up, evaluable, reason_not_evaluable

No se publica en `outputs/history/*`. No modifica el reporte diario.
No modifica `classify_wyckoff_phase` ni el orquestador. Es un
detector experimental con salida aislada.

`REACCUMULATION_CONFIRMED` es una etiqueta del CSV experimental, no
una fase del sistema. Se elimina cualquier mencion a "fase
resultante" del v1.

---

## 12. Tests de contrato (T1-T6, corregido)

El v1 decia "un test por cada regla R1-R5" (5 tests) y luego
enumeraba T1-T6 (6 tests). Correccion:

- T1: R1 contexto UP en `b - 60`, con info estrictamente anterior.
- T2: R2 consolidacion sobre `[b-W, b-1]`, formulas congeladas.
- T3: R3 pullback en la MISMA sesion, contacto + volumen bajo.
- T4: R4 confirmacion en `b`, `b+1`, `b+2` sobre umbral congelado.
- T5: R5 volumen de ruptura en `b` con ventana `[b-W, b-1]`.
- T6: **invariante de aislamiento.** El detector no toca
  `classify_wyckoff_phase`, ni el orquestador, ni `outputs/history/*`,
  ni ningun CSV productivo. Test que verifica por ausencia de
  escritura y por AST (no importa modulos de produccion).

---

## 13. Secuencia aprobada (dictamen)

1. Revision de este diseño (v2) por el auditor externo.
2. Firma expresa de v2.
3. Implementacion de T1-T6 con tests verdes.
4. Implementacion experimental aislada del detector.
5. Ejecucion sobre cohort, resultado marcado como exploratorio.
6. Evaluacion C1-C3 con los denominadores de §7-§9.
7. Decision separada de produccion: solo si el resultado es
   `FUNCIONA` en la ventana futura de §10, con firma adicional.

Sin firma de v2, no hay paso 3.

---

## 14. Estado normativo

Borrador v2. Sin validez normativa hasta firma del auditor. La firma
de v2 no autoriza incorporacion a produccion; solo autoriza ejecutar
T1-T6 y el detector experimental sobre cohort como evidencia
exploratoria.

Fin del pre-registro DICT3-REACCUMULATION v2.
