# CONTRATO WYCKOFF v1.10.1 - SOW-BEAR-CONFIRMATION (borrador para auditor)

**Estado:** BORRADOR. No implementado. Requiere revision y firma
del auditor externo antes de tocar codigo de produccion.
**Fecha:** 2026-10-31.
**Ciclo:** sesion 22.
**Frente:** SOW-BEAR-CONFIRMATION (REGISTRO.yaml, IN_PROGRESS).
**Base:** contrato v1.10 (2026-10-27). Esta version corrige errores
de v1.10 (ver seccion 11). Sustituye a v1.10 como contrato vigente
para auditoria.

---

## 0. Cambios respecto a v1.10

v1.10 tenia tres defectos que invalidaban su implementacion:

1. **Detector incorrecto.** Se apoyaba en `detect_sow_v2`, que
   exige parametros `window/x_atr/y_vol` **sin defaults** por
   contrato (v1.8, dictamen 5b.3). `detect_sow_v2` lanza ValueError
   si no se le pasan. Calibracion 5b.3 cerrada como WONT_FIX.
   v1.10 era inimplementable.
   **v1.10.1 usa `detect_sow_in_range`**, que ya se ejecuta en cada
   `classify_phase` de cada holding en produccion. No requiere
   calibracion.

2. **Agregacion indefinida.** v1.10 no definia como pasar de "SOW
   en un holding" a "SOW en el sector". Especificaba solo el caso
   ETF (que no funciona: los ETFs sectoriales son planos respecto
   a los holdings individuales).
   **v1.10.1 define agregacion holdings-based con umbral N=3.**

3. **Market proxy sin fijar.** v1.10 dejaba SPY como "candidato".
   **v1.10.1 fija SPY.**

---

## 1. Proposito

Definir una etiqueta informativa **SOW_BEAR_CONFIRMATION** que marque
sectores en los que concurren tres condiciones:

1. El mercado USA (SPY) esta en regimen bear.
2. Al menos 3 holdings del sector tienen SOW CONFIRMED en las
   ultimas M=5 sesiones.
3. El sector no es INVALID_UNIVERSE (cobertura suficiente).

La etiqueta NO es una fase. NO sustituye a `classify_wyckoff_phase`.
NO activa DISTRIBUTION por si sola. Es una anotacion de confirmacion
de deterioro amplio (no de un solo valor) dentro de un bear ya
identificado.

Motivacion empirica: el event study 2026-10-24 (doc 56) encontro que
el patron bear es real. SOW universal (SOW-V6) se cerro WONT_FIX
porque no es universal. SOW-BEAR-CONFIRMATION es el subconjunto
que si muestra senal, medido sobre holdings (no sobre ETF).

---

## 2. Alcance

**Dentro:**
- Definicion formal de la etiqueta.
- Regla de activacion.
- Contrato de salida (3 columnas en `sector_regime_matrix.csv`,
  schema v3).
- Invariantes normativas.
- Tests de contrato requeridos.

**Fuera:**
- Implementacion (bloqueada hasta firma).
- Modificacion de `classify_wyckoff_phase`.
- Modificacion de `detect_sow_in_range`.
- Modificacion de `WYCKOFF_SOW_*` (siguen en None, no se tocan).
- Modificacion de los 6 CSVs de holdings Wyckoff.
- Uso de la etiqueta como senal de trading.

---

## 3. Definiciones

### 3.1. Market proxy

**SPY**. Declarado como constante en la implementacion:
`SOW_BEAR_MARKET_PROXY = "SPY"`. Presente en
`data/market_data.parquet`.

### 3.2. Regimen bear

SPY esta en bear en la fecha t si y solo si se cumplen AMBAS:

- `close_spy(t) < MA200(close_spy, t)`.
- `drawdown_252d(t) < -0.10`, donde
  `drawdown_252d(t) = close_spy(t) / max(close_spy, 252 sesiones) - 1`.

Umbrales (200, -0.10, 252) son definiciones a priori. No se
optimizan. Cualquier cambio requiere nueva version del contrato.

### 3.3. SOW por holding

Un holding h de un sector s tiene SOW en la fecha t si y solo si:

    dist_evidence["sow_status"] == "CONFIRMED"
    AND dist_evidence["sow_event_date"] esta en [t - M + 1, t]

donde M = 5 sesiones. `detect_sow_in_range` (ya en produccion) emite
estos campos en `FinalPhase.dist_evidence`. No se modifica.

Un holding con `sow_status=NOT_EVALUABLE` o `NOT_OBSERVED` no cuenta
como SOW. Un holding que no completa la cascada (BC/AR no
confirmados) tampoco cuenta; es `None` implicito y se descarta.

### 3.4. Ventana M

M = 5 sesiones. Coherente con ventana C1 del sistema. No se optimiza.

### 3.5. Agregacion sectorial

Un sector s tiene SOW_BEAR_CONFIRMATION en la fecha t si y solo si
se cumplen LAS TRES:

1. `bear(SPY, t) == True`.
2. `count(holdings h de s con SOW en [t-M+1, t]) >= 3`.
3. `n_holdings_validos(s, t) >= 5` (cobertura suficiente para que
   3 SOW sean informativos; no 3 de 3).

**Umbral N=3:** tres holdings distintos con SOW confirma deterioro
amplio, no ruido de un solo valor grande arrastrando al sector.
Es la leccion del ETF: XLF puede caer 1 dia por JPM (-12% de peso)
sin que el sector este deteriorado. Tres holdings es senal.

**Umbral cobertura 5:** si el sector solo tiene 3 holdings
evaluables, "3 de 3 con SOW" es fragil. Se exige base minima.

Si falla (3): `sow_bear_reason = "INSUFFICIENT_COVERAGE"`.
Si falla (2): `sow_bear_reason = "INSUFFICIENT_SOW_HITS"`.
Si falla (1): `sow_bear_reason = "NOT_BEAR"`.

### 3.6. SOW_BEAR_CONFIRMATION

Etiqueta booleana por sector y fecha. Coexiste con `wyckoff_phase`
del sector (que sigue siendo el agregado de fases de holdings).
Es ortogonal: un sector puede estar en fase RANGE dominante Y
tener SOW_BEAR_CONFIRMATION=True. La etiqueta no modifica la fase.

---

## 4. Contrato de salida

### 4.1. CSV `sector_regime_matrix.csv` (schema v3)

Tres columnas nuevas al final:

- `sow_bear_confirmation` (bool).
- `sow_bear_hits` (int, 0..n_holdings; cuenta cruda de holdings con
  SOW en la ventana, aunque no llegue a N=3).
- `sow_bear_reason` (str: `OK`, `NOT_BEAR`, `INSUFFICIENT_SOW_HITS`,
  `INSUFFICIENT_COVERAGE`, `NO_DATA_MARKET_PROXY`,
  `NO_DATA_SECTOR`).

`sow_bear_hits` siempre se publica (util para observabilidad). Las
otras dos, con el significado descrito.

### 4.2. Reporte diario

En la seccion de regimen sectorial, una linea adicional **solo si
hay al menos un sector etiquetado**:

    SOW_BEAR_CONFIRMATION: XLF (4/5), XLE (3/5)

Formato: `<sector> (<n_sow_hits>/<M> sesiones)`. Si no hay sectores
etiquetados, la seccion se omite (no se imprime "vacio").

### 4.3. Lo que NO se anade

- No se anade fase nueva.
- No se modifica `wyckoff_phase`.
- No se modifica `structure_ratio`.
- No se modifica `n_estructuras`.
- No se propaga a `sector_rankings.csv` ni a rankings.

---

## 5. Invariantes normativas

I1. `classify_wyckoff_phase` no se modifica. La fase de cada ticker
sigue siendo la que dicta el orquestador.

I2. Los 6 CSVs de holdings Wyckoff no se modifican. La etiqueta vive
solo en `sector_regime_matrix.csv` y en el reporte.

I3. `WYCKOFF_SOW_*` sigue en None. `detect_sow_v2` no se usa. Si
la implementacion necesitara `detect_sow_v2`, es violacion del
contrato.

I4. Determinismo: dada la misma entrada (parquet + holdings), la
etiqueta es identica byte a byte. Sin aleatoriedad, sin
datetime.now() para observacion.

I5. Fail-closed: si falta SPY en `market_data.parquet`, todos los
sectores devuelven `sow_bear_confirmation=False` con
`sow_bear_reason="NO_DATA_MARKET_PROXY"`. Si falta un sector,
`"NO_DATA_SECTOR"`. No se imputa.

I6. No se propaga a ningun ranking. Es informativa.

---

## 6. Tests de contrato requeridos (TDD antes de codigo)

T1. SPY bear=False -> `sow_bear_confirmation=False`, `reason=NOT_BEAR`,
aunque haya 5 holdings con SOW.

T2. SPY bear=True, 0 holdings con SOW -> `False`,
`reason=INSUFFICIENT_SOW_HITS`, `sow_bear_hits=0`.

T3. SPY bear=True, 2 holdings con SOW -> `False`,
`reason=INSUFFICIENT_SOW_HITS`, `sow_bear_hits=2`.

T4. SPY bear=True, 3 holdings con SOW, sector con 5+ holdings
validos -> `True`, `reason=OK`, `sow_bear_hits=3`.

T5. SPY bear=True, 3 holdings con SOW, pero sector solo tiene 3
holdings validos -> `False`, `reason=INSUFFICIENT_COVERAGE`,
`sow_bear_hits=3`.

T6. SPY no presente en market_data -> todos `False`,
`reason=NO_DATA_MARKET_PROXY`.

T7. Invariante I1: `classify_wyckoff_phase` produce la misma salida
con y sin el detector activo.

T8. Invariante I4: dos ejecuciones consecutivas, misma entrada,
misma etiqueta byte a byte.

T9. `sector_regime_matrix.csv` cumple schema v3 (3 columnas nuevas,
tipos correctos: bool, int, str).

T10. Regresion: si SOW expira fuera de la ventana M (event_date
t-M-1), no cuenta. Solo `[t-M+1, t]`.

---

## 7. Parametros

| Parametro | Valor | Justificacion |
|---|---|---|
| `SOW_BEAR_MARKET_PROXY` | "SPY" | Coherente con rankings; instrumento observable |
| `MA200_WINDOW` | 200 sesiones | Estandar, no optimizado |
| `DRAWDOWN_WINDOW` | 252 sesiones | Ventana anual bursatil |
| `DRAWDOWN_THRESHOLD` | -0.10 | Del plan del frente |
| `M_SOW_WINDOW` | 5 sesiones | Coherencia con C1 |
| `N_SOW_MIN_HITS` | 3 | Deterioro amplio, no 1 valor |
| `MIN_VALID_HOLDINGS` | 5 | Base minima para que 3 SOW sean informativos |

Ninguno se ajusta via backtesting. Definiciones a priori. Cualquier
cambio requiere nueva version del contrato.

---

## 8. Diferencias con v1.9

- Anade `SOW_BEAR_CONFIRMATION` (seccion 3.6).
- Anade 3 columnas a `sector_regime_matrix.csv` (schema v3).
- No cambia fases, estados, componentes primarios, precedente
  estructural, clasificacion, transiciones, eventos, contrato de
  entrada/salida, invariantes de v1.9, parametros de v1.9, ni la
  candidata SOW congelada (v1.9 seccion 13).
- No activa `WYCKOFF_SOW_*`. Siguen en None.
- No usa `detect_sow_v2` (a diferencia de v1.10).

---

## 9. Condiciones de firma

El contrato se considera firmado si el auditor externo responde
explicitamente con:

- **Aprobado:** se implementa en una sesion posterior, con los tests
  T1-T10 escritos antes del codigo de produccion.
- **Aprobado con cambios:** se anotan, se versiona como v1.10.2 o
  v1.11, se re-revisa.
- **Rechazado:** el frente se cierra como WONT_FIX con la razon del
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

## 11. Trazabilidad de errores de v1.10

v1.10 se commiteo en 2026-10-27 (caa0ec9e). Tres problemas detectados
2026-10-31 (sesion 22) durante preparacion de implementacion:

1. `detect_sow_v2` (v1.10 seccion 3.3) exige parametros sin defaults
   por dictamen 5b.3. Calibracion cerrada WONT_FIX. Detector
   inutilizable. **v1.10.1 usa `detect_sow_in_range`.**
2. Agregacion sectorial no definida (v1.10 seccion 3.5 solo
   contemplaba caso ETF). ETFs sectoriales son planos respecto a
   holdings: la senal SOW no se propaga. **v1.10.1 agrega por
   holdings con umbral N=3.**
3. Market proxy "candidato: SPY" (v1.10 seccion 3.1). **v1.10.1
   fija SPY como constante.**

Estos errores no se ocultaron: se descubrieron al intentar implementar
v1.10. Se sustituye por v1.10.1 como contrato vigente. v1.10 queda
en el repo como referencia historica del error.

---

Fin del contrato v1.10.1 (borrador).
