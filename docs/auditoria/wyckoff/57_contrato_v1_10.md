# CONTRATO WYCKOFF v1.10 - SOW-BEAR-CONFIRMATION (borrador para auditor)

**Estado:** BORRADOR. No implementado. Requiere revision y firma del
auditor externo antes de tocar codigo de produccion.
**Fecha:** 2026-10-27.
**Ciclo:** sesion 13.
**Frente:** SOW-BEAR-CONFIRMATION (REGISTRO.yaml, estado IN_PROGRESS).
**Base:** contrato v1.9 (docs/auditoria/wyckoff/01_contrato_semantico_v1_9.md).
**Evidencia empirica:** docs/auditoria/wyckoff/56_event_study_2026-10-24/.

---

## 1. Proposito

Definir una etiqueta informativa **SOW_BEAR_CONFIRMATION** que marque
sectores en los que concurren dos condiciones:

1. El mercado (market proxy) esta en regimen bear.
2. El detector SOW v2 ha disparado recientemente en el sector.

La etiqueta NO es una fase. NO sustituye a `classify_wyckoff_phase`.
NO activa DISTRIBUTION por si sola. Es una anotacion de confirmacion
de deterioro dentro de un bear ya identificado.

Motivacion empirica: el event study 2026-10-24 (doc 56) encontro que
el patron bear es real — 3/3 anos con >10% de dias bear funcionan
(2018, 2020, 2022; 2025 en el limite con 8.8%), 0/5 anos con <10%
de dias bear funcionan. El SOW universal (SOW-V6) se cerro como
WONT_FIX porque no es universal; el bear-confirmation es el subconjunto
que si muestra senal.

---

## 2. Alcance

**Dentro:**
- Definicion formal de la etiqueta.
- Regla de activacion.
- Contrato de salida (que campos anade al reporte y a los CSVs).
- Invariantes normativas.
- Tests de contrato requeridos.

**Fuera:**
- Implementacion (bloqueada hasta firma).
- Modificacion de `classify_wyckoff_phase`.
- Modificacion de `WYCKOFF_SOW_*` (los cuatro parametros siguen en
  `None` en produccion; ver v1.9 seccion 13).
- Cualquier uso de la etiqueta como senal de trading.

---
## 3. Definiciones

### 3.1. Market proxy

Indice USA de referencia para clasificar el regimen de mercado.
Candidato: SPY (o el equivalente en data/market_data.parquet).
Debe declararse explicitamente en la implementacion; no se asume.

### 3.2. Regimen bear

El market proxy esta en bear si y solo si se cumplen AMBAS:

- close < MA200(close) en la fecha de observacion.
- drawdown_252d < -10%, donde
  drawdown_252d = close / max(close, 252 sesiones) - 1.

Los umbrales (MA200, -10%, 252 sesiones) son los del plan del frente
en REGISTRO. No se optimizan.

### 3.3. SOW v2

El detector SOW v2 ya existente (detect_sow_v2 en wyckoff_v1.py),
sin cambios. Definido en v1.8/v1.9 seccion 7.3. Su semantica no se toca.

### 3.4. Ventana M

M = 5 sesiones. Coherente con la ventana C1 del sistema. No se optimiza.

---

### 3.5. SOW_BEAR_CONFIRMATION

Un sector s en la fecha t tiene la etiqueta si y solo si:

    bear(market_proxy, t) == True
    AND
    existe k en [0, M-1] tal que detect_sow_v2(s, t-k) == True

donde detect_sow_v2(s, t-k) es el disparo del SOW v2 en el sector
s en la sesion t-k. Si el sector no tiene datos suficientes para
evaluar detect_sow_v2 en ninguna de las M sesiones, la etiqueta es
False (no NaN), con motivo INSUFFICIENT_DATA registrado en el
campo sow_bear_reason.

---

## 4. Contrato de salida

### 4.1. Reporte diario

En la seccion de regimen sectorial se anade una linea por sector
etiquetado, con formato:

    SOW_BEAR_CONFIRMATION: <sector_etf> (<n_sow_hits>/<M> sesiones)

Solo aparece si n_sow_hits >= 1 Y bear == True. Si no hay
sectores etiquetados, la seccion se omite (no se imprime "vacio").

---

### 4.2. CSV sector_regime_matrix.csv

Se anaden tres columnas (schema v3):

- sow_bear_confirmation (bool).
- sow_bear_hits (int, 0..M).
- sow_bear_reason (str: OK, NOT_BEAR, NO_SOW_HITS, INSUFFICIENT_DATA).

No se anade ninguna columna derivada tipo sow_bear_score ni
bear_strength. La etiqueta es booleana por diseno.

### 4.3. Lo que NO se anade

- No se anade fase nueva.
- No se modifica wyckoff_phase (retirada del ranking en sesion 3).
- No se modifica structure_ratio.
- No se modifica n_estructuras.

---

## 5. Invariantes normativas

I1. El detector NO modifica classify_wyckoff_phase. La fase de
cada ticker sigue siendo la que dicta el orquestador. La etiqueta
es ortogonal.

I2. El detector NO modifica holdings_snapshots.csv ni los 6 CSVs
de holdings. La etiqueta vive solo en el reporte y en
sector_regime_matrix.csv.

I3. El detector NO modifica WYCKOFF_SOW_*. Los cuatro parametros
siguen en None. Si la implementacion necesita activarlos, es un
cambio de contrato que requiere nueva firma.

I4. Determinismo: dada la misma entrada (parquet + holdings), la
etiqueta es identica byte a byte. Sin aleatoriedad, sin datetime.now()
para observacion.

I5. Fail-closed: si falta el market proxy o faltan datos de un
sector, la etiqueta es False con sow_bear_reason explicito. No se
imputa.

I6. La etiqueta NO se propaga a sector_rankings.csv ni a ningun
ranking. Es informativa.

---

## 6. Tests de contrato requeridos

Antes de implementar, el auditor debe validar que existan tests para:

T1. bear == False -> etiqueta False con reason NOT_BEAR, aunque SOW v2
haya disparado.

T2. bear == True y sow_hits == 0 -> etiqueta False con reason
NO_SOW_HITS.

T3. bear == True y sow_hits >= 1 -> etiqueta True con reason OK.

T4. Datos insuficientes del sector -> etiqueta False con reason
INSUFFICIENT_DATA.

T5. Invariante I1: classify_wyckoff_phase produce la misma salida con
y sin el detector activo.

T6. Invariante I4: dos ejecuciones consecutivas con la misma entrada
producen la misma etiqueta byte a byte.

T7. sector_regime_matrix.csv cumple schema v3 (tres columnas nuevas,
tipos correctos).

Estos tests se escriben antes de implementar (TDD de contrato).

---

## 7. Parametros

| Parametro | Valor | Justificacion |
|---|---|---|
| MA200 | 200 sesiones | Estandar de mercado, no optimizado |
| DRAWDOWN_252D_THRESHOLD | -10% | Del plan del frente en REGISTRO |
| DRAWDOWN_252D_WINDOW | 252 sesiones | Ventana anual bursatil |
| M (ventana SOW) | 5 sesiones | Coherencia con C1 |

Ninguno de estos parametros se ajusta via backtesting. Son definiciones
a priori. Cualquier cambio requiere nueva version del contrato.

---

## 8. Diferencias con v1.9

- Anade la etiqueta SOW_BEAR_CONFIRMATION (seccion 3.5).
- Anade tres columnas a sector_regime_matrix.csv (schema v3).
- No cambia fases, estados, componentes primarios, precedente
  estructural, clasificacion, transiciones, eventos, contrato de
  entrada/salida, invariantes de v1.9, parametros de v1.9, ni la
  candidata SOW congelada (v1.9 seccion 13).
- No activa WYCKOFF_SOW_*. Siguen en None.

---

## 9. Condiciones de firma

El contrato se considera firmado si el auditor externo responde
explicitamente con:

- Aprobado: se implementa en una sesion posterior, con los tests
  T1-T7 escritos antes del codigo de produccion.
- Aprobado con cambios: se anotan los cambios, se versiona como
  v1.10.1 o v1.11, se re-revisa.
- Rechazado: el frente se cierra como WONT_FIX con la razon del
  auditor registrada en REGISTRO.yaml. La evidencia del doc 56 se
  conserva como referencia historica.

Sin respuesta explicita del auditor, el frente permanece IN_PROGRESS
y no se toca codigo.

---

## 10. Estado normativo

Borrador. Este documento no tiene validez normativa hasta firma
del auditor externo. Cualquier implementacion antes de la firma es
una violacion del contrato del sistema (REGISTRO, SOW-BEAR-CONFIRMATION,
dependencia: "No se implementa nada en produccion hasta firma del
contrato").

---

Fin del contrato v1.10 (borrador).

