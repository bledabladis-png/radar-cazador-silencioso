# UTAD_PENDING y SPRING_PENDING — pre-registro v1

**Estado:** CONGELADO antes de correr sobre 327 stocks.
**Fecha:** 2026-10-24

## Semantica

Estados intermedios entre "rango formado" y "ruptura confirmada".

### UTAD_PENDING (cascada distribucion)

Se emite cuando:
- `range_event` esta `RANGE_ACTIVE` (rango de distribucion formado por BC+AR).
- `detect_false_breakout_up` devuelve `CONFIRMED` (ruptura arriba rechazada).
- `detect_sow_in_range` **NO** ha confirmado (sow_status en NOT_OBSERVED / CANDIDATE).

Significado: el precio intento romper arriba (UTAD/UT) y fracaso. Se ha
rechazado la resistencia, pero el SOW (ruptura abajo) aun no ha ocurrido.
La distribucion se esta materializando pero no esta confirmada.

### SPRING_PENDING (cascada acumulacion)

Simetrico:
- `range_event` esta `RANGE_ACTIVE`.
- `detect_false_breakout_down` devuelve `CONFIRMED` (ruptura abajo rechazada).
- `detect_sos_in_range` **NO** ha confirmado.

Significado: el precio intento romper abajo (Spring) y fracaso. Se ha
rechazado el soporte, pero el SOS (ruptura arriba) aun no ha ocurrido.

## Regla de precedencia

Si hay SOW/SOS confirmado, ese gana siempre. UTAD_PENDING solo aplica si
NO hay SOW/SOS. Si UTAD_PENDING y SPRING_PENDING coexisten en el mismo
stock (falsas rupturas en ambos lados), se emite `RANGE` (sin direccion).

## Lo que NO hace

- No activa DISTRIBUTION ni ACCUMULATION confirmada.
- No modifica la regla simetrica de vigencia.
- No toca los estados MARKUP/MARKDOWN.

## Criterios de aceptacion (pre-definidos, no ajustables tras ver resultado)

- **A1 — No inflacion**: con la regla actual, `DISTRIBUTION` vigente = 16.
  Con UTAD_PENDING aplicado, el numero de DISTRIBUTION confirmadas no cambia
  (UTAD_PENDING es un estado distinto, no reemplaza DISTRIBUTION).
- **A2 — Coherencia**: ninguna UTAD_PENDING puede coexistir con SOW
  confirmado. Ningun SPRING_PENDING con SOS confirmado.
- **A3 — Simetria**: la diferencia entre |UTAD_PENDING| y |SPRING_PENDING|
  no debe ser superior a 5x. Si lo es, el detector es asimetrico y hay
  un bug de calibracion (pre-registro falla, se descarta y se documenta).

Si A1 ∧ A2 ∧ A3 → UTAD_PENDING/SPRING_PENDING se incorporan al output
del orquestador. Si no → se descarta la integracion, los detectores
quedan como evidencia pasiva (opcion A), y se documenta.

## Compromiso

Tras correr sobre los 327: no se ajustan umbrales. Si falla, se descarta.

---

## RESULTADO — 2026-10-24

Corrida sobre 327 stocks de stock_prices.parquet.

- UTAD_PENDING: **26** stocks (AAL.L, ANET, AVGO, CABK.MC, ..., WMB).
- SPRING_PENDING: **3** stocks (CHTR, DHR, DVN).

Criterios:
- A1 (DISTRIBUTION = 16): **PASA**.
- A2 (0 violaciones de coherencia): **PASA**.
- A3 (simetria ratio <= 5x): **FALLA** (26/3 = 8.67x).

**VEREDICTO: DESCARTADO.**

No se ajusta A3. No se re-prueba con otro umbral. La integracion se
revierte: `classify_distribution_at` y `classify_accumulation_at` vuelven
a devolver `RANGE_ACTIVE_NO_BREAK` cuando no hay SOW/SOS confirmado.

Los detectores `detect_false_breakout_up` y `detect_false_breakout_down`
se conservan como codigo. Pueden usarse como **evidencia pasiva**
(adjuntar a `evidence`) en el futuro, pero no modifican `classification`.

## Diagnostico de la asimetria

26 vs 3 no es un bug tecnico. Los dos detectores usan la misma logica
(mismo `max_window`, mismo `confirm_sessions`, misma tolerancia). La
asimetria es del DATO:

- En el S&P 500 alcista 2025-2026, hay muchos mas intentos de ruptura
  arriba rechazados que de ruptura abajo rechazados. Logico: el sesgo
  del mercado es alcista, los UT/UTAD son frecuentes, los Spring raros.
- Los 3 Spring (CHTR, DHR, DVN) estan en sectores defensivos o en
  tendencia bajista propia. Los 26 UTAD cubren tecnologia, energia,
  financieras, materiales — el espectro alcista del mercado.

Consecuencia: el detector es correcto. El pre-registro fallo porque
asumio simetria donde no la hay. Un nuevo pre-registro v2 tendria que
aceptar la asimetria como dato de mercado, no como bug.

## Lo que NO se hace

- No se reescribe A3 con otro umbral.
- No se descarta el detector.
- No se intenta "simetrizar" el detector forzando mas Springs.

El detector queda disponible. La integracion queda descartada hasta
nuevo pre-registro que parta de que UTAD y Spring son fenomenos con
distinta frecuencia en mercados alcistas.
