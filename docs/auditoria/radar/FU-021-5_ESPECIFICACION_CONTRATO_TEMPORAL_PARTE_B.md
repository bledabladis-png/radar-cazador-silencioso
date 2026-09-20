# FU-021-5 — Especificación del contrato temporal de `df_market`
## Parte B — Contratos por clase (semántica operativa)

**Fecha:** 2026-09-16
**HEAD de referencia:** e54bab8 (origin/main)
**Parte A:** FU-021-5_ESPECIFICACION_CONTRATO_TEMPORAL.md (61aa99f, corregida e54bab8)
**Informes empíricos que alimentan este documento:**
- FU-021-3B_INFORME_INDEX_EOD_RATE_YIELD.md (b645de4)
- FU-021-3B_ANEXO_VOLATILITY_INDEX.md (e54bab8)
- FU-021-3C_INFORME_FUTURE_FX.md (794f0e9)
**Estado:** borrador Parte B. Pendiente de dictamen.

---

## §B.0 — Alcance y delimitación Parte A / Parte B

### B.0.1. Qué cubre esta Parte B

La Parte A especificó la arquitectura contractual de `df_market` (responsabilidades, propagación, writers, MTE, darkpool) sin entrar en la semántica operativa de cada clase.

La Parte B cierra la semántica operativa de las **6 clases** identificadas empíricamente:

| Clase | Estado previo | Estado tras Parte B |
|---|---|---|
| EQUITY_EOD | Cerrada por FU-021-3A | Confirmada sin cambios |
| INDEX_EOD | Pendiente | Cerrada (padre + 3 subclases) |
| VOLATILITY_INDEX | Descubierta en 3B-bis | Cerrada (heredada) |
| RATE_YIELD | Pendiente | Cerrada |
| FUTURE_SETTLEMENT | Bloqueada | **Cerrada como BLOCKED** |
| FX_DAILY_CUT | Pendiente | Cerrada (activable) |

### B.0.2. Qué NO cubre esta Parte B

- No activa contratos.
- No modifica código.
- No retira `trim_to_last_valid_date` (A3.1 sigue bloqueada).
- No decide el orden de implementación.
- No corrige la Parte A (ya corregida en e54bab8).

### B.0.3. Dependencias empíricas

Toda semántica de esta Parte B está respaldada por medición empírica, no por diseño teórico. Los informes de referencia son:

- **FU-021-3B** (INDEX_EOD + RATE_YIELD): causa raíz del lag europeo, valores idénticos en día común.
- **FU-021-3B-bis** (VOLATILITY_INDEX): confirmación de herencia desde INDEX_EOD_USA.
- **FU-021-3C** (FUTURE + FX): descubrimiento de históricos no reproducibles en futuros.

---

## §B.1 — INDEX_EOD (padre + subclases)

### B.1.1. Contrato padre

`INDEX_EOD` es un contrato **parametrizado por mercado origen**. Agrupa todos los índices bursátiles cuya fecha efectiva es la última sesión cerrada del mercado correspondiente.

**Parámetros comunes:**
```
class INDEX_EOD:
eligible_universe : dict[subclase, list[ticker]]
session_calendar : por subclase
max_lag_days : por subclase
filter_function : no aplica (no es equity)
resolver : por subclase
outputs:
effective_date : date
expected_session : date
coverage : float en [0,1]
status : OK | STALE | INSUFFICIENT | BLOCKED
lag_days : int

```

**Semántica del `status`:**

| Status | Condición |
|---|---|
| `OK` | `effective_date == expected_session` |
| `STALE` | `0 < lag_days <= max_lag_days` |
| `INSUFFICIENT` | `lag_days > max_lag_days` o `coverage < min_coverage` |
| `BLOCKED` | clase no activable (no aplica a INDEX_EOD) |

**Invariantes:**

- `effective_date <= expected_session`.
- `lag_days = (expected_session - effective_date).days`.
- La subclase sólo se considera `OK` si TODOS sus tickers alcanzan la fecha esperada; si algunos van por detrás, la subclase es `STALE` con `lag_days` = máximo lag observado.

### B.1.2. Subclase INDEX_EOD_USA

**Miembros:** `^GSPC`, `^DJI`, `^NDX`, `^RUT`, `^SPGSCI`

**Semántica:**
- Calendario: NYSE.
- Cierre: 16:00 ET (20:00 UTC).
- Publicación Yahoo: ~21:00 UTC.
- Lag típico observado: 1 día.

**Parámetros:**
```
class INDEX_EOD_USA:
eligible_universe : [^GSPC, ^DJI, ^NDX, ^RUT, ^SPGSCI]
session_calendar : "NYSE"
max_lag_days : 1
min_coverage : 1.0 (todos los miembros deben estar)

```

**Evidencia empírica (FU-021-3B §3.1):**
- Todos los miembros con `last_date = 2026-09-14` en parquet.
- Yahoo devuelve `last_date = 2026-09-15` para todos.
- Valores Close en día común: idénticos (diff ≤ 0.003).
- Lag = 1 día uniforme.

**Estado esperado:** `OK` o `STALE` con `lag_days = 1`.

### B.1.3. Subclase INDEX_EOD_EUROPA

**Miembros:** `^FTSE`, `^GDAXI`, `^IBEX`, `^STOXX50E`

**Semántica:**
- Calendario: LSE (^FTSE), XETRA (^GDAXI), BME (^IBEX), Euronext (^STOXX50E).
- Cierre: 16:30-17:30 CET (14:30-15:30 UTC).
- Publicación Yahoo: variable por mercado.
- Lag típico observado: 3-4 días por artefacto de cache (Parte A §7). El lag "estructural" es 1 día.

**Parámetros:**
```
class INDEX_EOD_EUROPA:
eligible_universe : [^FTSE, ^GDAXI, ^IBEX, ^STOXX50E]
session_calendar : "LSE|XETRA|BME|EURONEXT" (por ticker)
max_lag_days : 5 (tolerancia fines de semana + festivos)
min_coverage : 1.0

```

**Evidencia empírica (FU-021-3B §3.1, §5.2):**

| Ticker | Parquet last | Yahoo last | Lag |
|---|---|---|---|
| ^FTSE | 2026-09-11 | 2026-09-15 | 4d |
| ^GDAXI | 2026-09-11 | 2026-09-14 | 3d |
| ^IBEX | 2026-09-11 | 2026-09-14 | 3d |
| ^STOXX50E | 2026-09-11 | 2026-09-14 | 3d |

**Hallazgo H-3B-10:** `^FTSE` publica 1 día antes que los otros 3. No asumir coherencia entre índices del mismo mercado. El contrato debe declarar `per_ticker_lag` si se necesita granularidad.

**Causa raíz del lag observado (FU-021-3B §6):** artefacto de la ventana de cache (`CACHE_HOURS=23`) combinada con la frecuencia del cron. No es bug del pipeline. Los valores Close son correctos cuando coinciden las fechas.

### B.1.4. Subclase INDEX_EOD_COMMODITY

**Miembros:** `DX-Y.NYB` (US Dollar Index, ICE)

**Semántica:**
- Calendario: ICE.
- Cierre: 17:00 ET (21:00 UTC).
- Lag típico observado: 1 día.

**Parámetros:**
```
class INDEX_EOD_COMMODITY:
eligible_universe : [DX-Y.NYB]
session_calendar : "ICE"
max_lag_days : 1
min_coverage : 1.0

```

**Evidencia empírica (FU-021-3B §3.1):**
- Parquet last = 2026-09-14.
- Yahoo last = 2026-09-15.
- Diff en día común: 0.041 (rel 0.04%). Mayor que el resto pero consistente con ajuste de composición intradía.

**Hallazgo H-3B-9:** diferencia de 0.041 en DX-Y.NYB frente a 0.002-0.003 de índices puros. Registrar como característica conocida del índice.

---

## §B.2 — VOLATILITY_INDEX (heredado de INDEX_EOD_USA)

**Miembros:** `^VIX`, `^VIX3M`, `^VXN`

**Semántica:**
- Calendario: CBOE (mismo calendario bursátil USA).
- Cierre: 16:15 ET.
- Hereda parámetros de INDEX_EOD_USA.

**Parámetros:**
```
class VOLATILITY_INDEX(INDEX_EOD_USA):
eligible_universe : [^VIX, ^VIX3M, ^VXN]
session_calendar : "NYSE" (heredado)
max_lag_days : 1 (heredado)
min_coverage : 1.0

```

**Evidencia empírica (FU-021-3B-bis §3):**

| Ticker | Parquet last | Yahoo last | Lag | Match |
|---|---|---|---|---|
| ^VIX | 2026-09-14 | 2026-09-15 | 1d | idéntico |
| ^VIX3M | 2026-09-14 | 2026-09-15 (n=1) | 1d | no verificable |
| ^VXN | 2026-09-14 | 2026-09-15 | 1d | idéntico |

**Nota especial ^VIX3M (H-3B-bis-2):** descarga individual `yfinance` con `period='15d'` devuelve 1 sola fila. La descarga en batch (que es la que usa el pipeline) trae el histórico completo. No es problema del pipeline.

**Estado esperado:** `OK` o `STALE` con `lag_days = 1`, idéntico a INDEX_EOD_USA.

---

## §B.3 — RATE_YIELD

**Miembros:** `^FVX` (5-year Treasury yield), `^TNX` (10-year Treasury yield)

**Semántica:**
- Fuente: US Treasury.
- Publicación: tras el cierre del mercado de bonos (~15:30 ET).
- Lag típico observado: 1 día.

**Parámetros:**
```
class RATE_YIELD:
eligible_universe : [^FVX, ^TNX]
session_calendar : "NYSE" (referencia bursátil)
max_lag_days : 1
min_coverage : 1.0
publication_source : "US_TREASURY"

```

**Evidencia empírica (FU-021-3B §3.1):**

| Ticker | Parquet last | Yahoo last | Lag | Match |
|---|---|---|---|---|
| ^FVX | 2026-09-14 | 2026-09-15 | 1d | idéntico |
| ^TNX | 2026-09-14 | 2026-09-15 | 1d | idéntico |

**Hallazgo H-3B-3:** el universo contiene solo 2 yields de los ~4 habituales. Faltan `^IRX` (13-week) y `^TYX` (30-year). No es bloqueante; es límite del universo actual.

**Nota sobre `effective_date` para yields:** si Yahoo devuelve el yield publicado por el Treasury en la fecha T, y el cierre bursátil USA también es el día T, `effective_date` coincide con INDEX_EOD_USA. No se necesita contrato temporal separado, solo declaración de `publication_source`.
---

## §B.4 — FUTURE_SETTLEMENT (clase BLOCKED)

### B.4.1. Miembros

`BZ=F` (Brent), `CL=F` (WTI), `GC=F` (Gold), `HG=F` (Copper), `NG=F` (Natural Gas)

### B.4.2. Estado: BLOCKED

**La clase no puede activarse con Yahoo continuo.** No es cuestión de provider pendiente; es que el ticker continuo de Yahoo **no es un identificador estable**.

### B.4.3. Evidencia empírica (FU-021-3C §3.1, §5.2)

| Ticker | Parquet 14/09 | Yahoo 14/09 (hoy) | Diff rel |
|---|---|---|---|
| BZ=F | 106.49 | 105.68 | 0.76% |
| CL=F | 102.10 | 101.39 | 0.70% |
| GC=F | 4327.50 | 4351.90 | 0.56% |
| HG=F | 6.389 | 6.330 | 0.92% |
| NG=F | 2.880 | 2.896 | 0.56% |

Comparativa con índices USA: diff era 0.002-0.003. Aquí es 100-1000× mayor. **La divergencia no es uniforme** (mezcla signos positivos y negativos).

### B.4.4. Hipótesis y descarte (FU-021-3C §4-§6)

| Hipótesis | Veredicto |
|---|---|
| Rollover retroactivo puro | Descartada |
| Ajuste por rango temporal | Descartada |
| Ajuste por `auto_adjust` | Descartada |
| Snapshot intradía (sesión asiática) | Descartada |
| Reescritura del histórico por Yahoo | **No verificable, posible** |

**Conclusión (FU-021-3C §6.5):** la investigación se cerró sin causa raíz exacta. Sea rollover, sea ajuste, sea corrección, el resultado operativo es el mismo: **los históricos del parquet no son reproducibles**.

### B.4.5. Contrato
```
class FUTURE_SETTLEMENT:
eligible_universe : [BZ=F, CL=F, GC=F, HG=F, NG=F]
session_calendar : "CME" (todos los futuros actuales son CME/ICE)
settlement_source : "CME|ICE" (según contrato)
max_lag_days : N/A
min_coverage : N/A
status : BLOCKED
activation_requirement:

provider que exponga el contrato específico (no el continuo)

settlement oficial, no close intradía

identificador estable por contrato

```

### B.4.6. Acciones requeridas para activación

Tres caminos posibles (decisión de otro ciclo, no de Parte B):

1. **Contratos explícitos con lógica de rollover declarada.**
   Yahoo soporta `GCZ26.CMX`, `CLV26.NYM`. Requiere lógica propia de gestión de rollover.
2. **Provider dedicado (CME directo, ICE).**
   Coste de integración alto, pero semántica limpia.
3. **Congelar la clase como `BLOCKED` permanentemente.**
   Aceptar que el pipeline no consume futuros con garantía temporal. Afecta a los 30-40% de consumidores que los usan.

**Recomendación de Parte B:** mantener `BLOCKED` hasta que el auditor decida entre las tres.

### B.4.7. Impacto en consumidores

**Hallazgo H-3C-12:** 30-40% del pipeline consume futuros. Afectados directos:

- `pipeline/flows_primary.py`
- `pipeline/mte_confirmation.py` (HG=F/GC=F)
- `pipeline/sectors_base.py` (CL=F, HG=F, GC=F)
- `regimes/macro_regime.py` (industrial_tickers + gold)
- `indicators/mte.py`

Mientras la clase esté `BLOCKED`, el contrato temporal no puede garantizar coherencia para estos consumidores.

---

## §B.5 — FX_DAILY_CUT (clase activable)

### B.5.1. Miembros

`EURUSD=X`, `USDCNY=X`, `USDJPY=X`

### B.5.2. Semántica

- Mercado: FX spot 24h, sin cierre único.
- Cutoff: 17:00 ET (21:00 UTC), estándar de mercado.
- Publicación Yahoo: variable por par.
- Lag típico observado: heterogéneo (0-1 día por par).

### B.5.3. Parámetros
```
class FX_DAILY_CUT:
eligible_universe : [EURUSD=X, USDCNY=X, USDJPY=X]
session_calendar : "FX_24h"
cutoff_time : "17:00 America/New_York"
max_lag_days : dict por par
min_coverage : 0.5 (al menos 2 de 3 pares deben estar al día)
per_pair_max_lag : {
"EURUSD=X": 0,
"USDJPY=X": 0,
"USDCNY=X": 1,
}

```

### B.5.4. Evidencia empírica (FU-021-3C §2.2, §3.2)

| Ticker | Parquet last | Yahoo last | Lag | Match |
|---|---|---|---|---|
| EURUSD=X | 2026-09-15 | 2026-09-15 | 0 | diff 3.46e-4 |
| USDCNY=X | 2026-09-14 | 2026-09-15 | 1 | idéntico en día común |
| USDJPY=X | 2026-09-15 | 2026-09-15 | 0 | diff 0.35% |

**Hallazgo H-3C-2:** FX tiene last_date asimétrico intra-clase. Requiere `per_pair_max_lag`.

**Hallazgo H-3C-4:** FX tiene ~89 filas más en el parquet por festivos USA que los mercados FX sí negocian.

**Hallazgo H-3C-8:** USDJPY tiene diff relativo de 0.35% (mayor que EURUSD). Pendiente de verificación adicional si se quiere entender la causa.

### B.5.5. Estado esperado

`OK` si todos los pares están al día (EURUSD, USDJPY con lag 0, USDCNY con lag ≤ 1).
`STALE` si USDCNY cae a lag 2+.
`INSUFFICIENT` si más de 1 par cae fuera.

### B.5.6. Decisión pendiente

El cutoff exacto (17:00 ET) es provisional. Requiere verificación empírica adicional si se necesita precisión. Por ahora se declara como documental.

---

## §B.6 — Criterios de activación por clase

Resumen del estado por clase tras Parte B:

| Clase | Estado Parte B | Activable |
|---|---|---|
| EQUITY_EOD | Cerrada (FU-021-3A) | ✅ Sí (ya activa) |
| INDEX_EOD_USA | Cerrada (Parte B §B.1.2) | ✅ Sí |
| INDEX_EOD_EUROPA | Cerrada (Parte B §B.1.3) | ✅ Sí |
| INDEX_EOD_COMMODITY | Cerrada (Parte B §B.1.4) | ✅ Sí |
| VOLATILITY_INDEX | Cerrada (Parte B §B.2) | ✅ Sí |
| RATE_YIELD | Cerrada (Parte B §B.3) | ✅ Sí |
| FUTURE_SETTLEMENT | Cerrada como BLOCKED (Parte B §B.4) | 🔴 No |
| FX_DAILY_CUT | Cerrada (Parte B §B.5) | ✅ Sí |

### B.6.1. Orden de activación propuesto

1. **INDEX_EOD + VOLATILITY_INDEX + RATE_YIELD** (grupo homogéneo: calendario USA + EU).
2. **FX_DAILY_CUT** (grupo pequeño: 3 tickers).
3. **FUTURE_SETTLEMENT** (bloqueado hasta decisión de provider).

### B.6.2. Criterio de desbloqueo de A3.1

Según el dictamen del contrato temporal, A3.1 se desbloquea cuando:
```
✅ contrato de cada clase relevante
✅ regla de consolidación de df_market
✅ propagación de effective metadata
✅ writers adaptados
✅ estado MTE compatible
✅ darkpool fuera de fuente paralela

```

**Con Parte B publicada, los dos primeros se cumplen. Los cuatro restantes son trabajo de implementación.**

---

## §B.7 — Compatibilidad con Parte A

### B.7.1. Correcciones de la Parte A aplicadas en e54bab8

- §A.1: 5 clases → 6 clases (añadida VOLATILITY_INDEX).
- Tabla de clases: fila nueva.
- Nota de verificación empírica.

### B.7.2. Cambios adicionales propuestos por Parte B

1. **`status` ampliado** con `STALE`. Parte A definía `OK | INSUFFICIENT | BLOCKED | PENDING`. Parte B añade `STALE` (lag positivo dentro de tolerancia).

2. **`per_pair_max_lag` para FX.** Parte A no contemplaba lag heterogéneo intra-clase. Parte B lo introduce para FX.

3. **`per_ticker_lag` para INDEX_EOD_EUROPA.** Mismo motivo: ^FTSE publica 1 día antes que el resto.

4. **`activation_requirement` para FUTURE_SETTLEMENT.** Campo nuevo que documenta las condiciones de desbloqueo.

**Estas propuestas requieren ratificación por dictamen.** No modifican la Parte A publicada; se acumulan para una eventual Parte A v2.

### B.7.3. Invariantes preservadas

- Contratos por clase (no agregado).
- Consolidación explícita (no imposición de fecha común).
- Fuente canónica única (`df_market` en memoria).
- Responsabilidad temporal única (productor).

---

## §B.8 — Referencias empíricas cruzadas

| Clase | Informe | Secciones de referencia |
|---|---|---|
| INDEX_EOD_USA | FU-021-3B | §3.1, §5 |
| INDEX_EOD_EUROPA | FU-021-3B | §3.1, §5.2, §6 |
| INDEX_EOD_COMMODITY | FU-021-3B | §3.1, §3.2 |
| VOLATILITY_INDEX | FU-021-3B-bis | §1, §2, §3 |
| RATE_YIELD | FU-021-3B | §3.1 |
| FUTURE_SETTLEMENT | FU-021-3C | §2, §3.1, §4, §5.2, §6 |
| FX_DAILY_CUT | FU-021-3C | §2.2, §3.2 |

---

## §B.9 — Lo que la Parte B NO decide

- No activa contratos.
- No implementa cambios.
- No retira `trim_to_last_valid_date` (A3.1 sigue bloqueada).
- No decide el orden de activación final.
- No decide el provider alternativo para FUTURE_SETTLEMENT.
- No decide el cutoff exacto de FX.
- No modifica la Parte A publicada (solo propone cambios para eventual v2).
- No decide si las modificaciones a Parte A (§B.7.2) se aceptan o se rechazan.

---

## §B.10 — Preguntas al auditor

### Q-B.1 — Aceptación del `status` ampliado

¿Aprueba el auditor el nuevo estado `STALE` (lag positivo dentro de tolerancia) como distinto de `OK` e `INSUFFICIENT`?

### Q-B.2 — `per_pair_max_lag` para FX

¿Aprueba el diseño de `per_pair_max_lag` como parámetro del contrato FX_DAILY_CUT, reconociendo que los pares no publican simultáneamente?

### Q-B.3 — `per_ticker_lag` para INDEX_EOD_EUROPA

¿Aprueba `per_ticker_lag` para declarar que ^FTSE publica 1 día antes que el resto de índices EU?

### Q-B.4 — FUTURE_SETTLEMENT como BLOCKED definitivo

¿Confirma el auditor que `FUTURE_SETTLEMENT` debe quedar como `BLOCKED` hasta decisión explícita sobre contratos explícitos vs provider dedicado?

### Q-B.5 — Modificaciones a Parte A (§B.7.2)

¿Aprueba las 4 modificaciones propuestas a la Parte A publicada, o prefiere diferirlas?

### Q-B.6 — Orden de activación (§B.6.1)

¿Aprueba el orden propuesto (INDEX_EOD + VOLATILITY + RATE → FX → FUTURE)?

### Q-B.7 — Impacto en consumidores

Con FUTURE_SETTLEMENT bloqueado, ¿cómo deben comportarse los 5 consumidores afectados? Opciones:
- (α) Consumir futuros igualmente, asumiendo el riesgo de históricos inestables.
- (β) Excluir futuros del cálculo hasta desbloqueo.
- (γ) Consumir futuros pero declarar `status=UNRELIABLE` en sus outputs.

### Q-B.8 — Desbloqueo de A3.1

Con Parte B publicada, ¿debe reconsiderarse el desbloqueo de A3.1 o mantenerlo bloqueado hasta completar las 4 condiciones restantes (propagación, writers, MTE, darkpool)?

---

**Fin de la Parte B.**
**HEAD de referencia:** e54bab8 (origin/main).
**Fecha:** 2026-09-16.
**Estado:** borrador para dictamen. Parte A publicada, Parte B propuesta.