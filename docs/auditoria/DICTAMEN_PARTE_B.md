# Dictamen externo — FU-021-5 Parte B

**Fecha:** 2026-09-16
**HEAD de referencia al recibir el dictamen:** 0eb7e4f (origin/main)
**Documento evaluado:** FU-021-5_ESPECIFICACION_CONTRATO_TEMPORAL_PARTE_B.md (468fc08)
**Documento base:** FU-021-5_ESPECIFICACION_CONTRATO_TEMPORAL.md Parte A (61aa99f)
**Estado:** dictamen completo. 8 preguntas Q-B resueltas. GO CONDICIONADO.

---

## 0. Resultado global

**GO CONDICIONADO.**

La evidencia presentada es suficiente para ratificar la arquitectura operativa de los contratos, pero no recomiendo publicar Parte A v2 todavía sin corregir dos inconsistencias documentales:

1. `DX-Y.NYB` aparece como `INDEX_EOD_COMMODITY`, cuando la clasificación previamente establecida es `CURRENCY_INDEX`.
2. El documento habla de **6 clases**, pero la tabla operativa contiene **8 contratos/subgrupos**. Esto debe expresarse como **familias + subclases/contratos**, no como seis clases.

Además, la afirmación de que los futuros afectan al "30-40% del pipeline" necesita reformularse: representan **5 de 562 instrumentos (~0,9%)**, aunque pueden influir en varios cálculos. El impacto de dependencia y el peso del instrumento no son la misma métrica.

Con esas correcciones, se dicta lo siguiente.

---

## 1. Q-B.1 — `STALE`

**Dictamen: 🟢 α — APROBADO.**

Debe existir el estado `STALE` como distinto de `OK` e `INSUFFICIENT`, con semántica inequívoca.

La distinción es importante:

- `coverage` correcta + retraso permitido → `STALE`.
- `coverage` insuficiente → `INSUFFICIENT`.

No deben colapsarse.

### Condición

`STALE` no significa "cualquier dato antiguo es aceptable". Debe definirse:

- `0 < lag_days <= max_lag_days` → `STALE`.
- `lag_days > max_lag_days` → `BLOCKED` / `INVALID` contractual según la FSM definitiva.

Además, el contrato debe conservar **la fecha efectiva real**, no solo el estado.

---

## 2. Q-B.2 — `per_pair_max_lag`

**Dictamen: 🟢 α — APROBADO.**

Para tres pares, un diccionario explícito es más auditable que una cifra global:

- `EURUSD=X` → ...
- `USDJPY=X` → ...
- `USDCNY=X` → ...

Pero no aprobaría que la tolerancia se convierta en una excusa para ocultar una caída de actualización. Debe existir:

- `expected publication behavior`
- `observed lag`
- `maximum tolerated lag`

Así se puede distinguir `USDCNY normalmente +1d` de `USDCNY +7d` sin cambiar el contrato.

---

## 3. Q-B.3 — `per_ticker_lag`

**Dictamen: 🟢 α — APROBADO, con una condición de diseño.**

Sí para `^FTSE`, `^GDAXI`, `^IBEX`, `^STOXX50E`, pero no aprobaría `max_lag_days=5` como sustituto de la especificación por ticker.

Debe quedar algo parecido a:

- `ticker`
- `expected lag`
- `maximum tolerated lag`
- `source behavior`

La razón es exactamente la observación de la auditoría: un retraso habitual del proveedor no es lo mismo que un retraso anormal. `per_ticker_lag` permite registrar ambas cosas.

---

## 4. Q-B.4 — FUTURE_SETTLEMENT

**Dictamen: 🟢 α — `BLOCKED`.**

Decisión importante, ratificada.

Los datos presentados no demuestran:

`Yahoo continuous future → official settlement reproducible`

Por el contrario, demuestran que la serie continua puede cambiar históricamente.

Por tanto `CL=F`, `BZ=F`, `NG=F`, `GC=F`, `HG=F` no deben declararse `FUTURE_SETTLEMENT` válido mientras no exista una fuente/contrato capaz de proporcionar realmente el **settlement** requerido.

### No aprobado β

No debe saltarse desde `Yahoo continuo no auditable` directamente a `GCZ26.CMX` sin resolver:

- elección del contrato;
- roll;
- expiración;
- continuidad histórica;
- tratamiento de gaps;
- fuente de settlement.

### No aprobado γ

No considero aceptable congelar permanentemente la clase sin necesidad. El estado correcto es `BLOCKED` + `activation_requirement` explícito.
---

## 5. Q-B.5 — Cambios a Parte A

**Dictamen: 🟢 α, pero después de corregir la taxonomía indicada arriba.**

Aprueba incorporar `STALE`, `per_pair_max_lag`, `per_ticker_lag`, `activation_requirement` en una **Parte A v2**. No dejaría Parte A y Parte B divergiendo indefinidamente.

### Pero Parte A v2 debe resolver primero: familia ≠ subclase ≠ contrato

Propuesta conceptual:

Familias: EQUITY, INDEX, RATE_YIELD, FUTURE, FX.

Contratos/subgrupos:

- EQUITY_EOD
- INDEX_EOD_USA
- INDEX_EOD_EUROPA
- INDEX_EOD_COMMODITY
- INDEX_EOD_CURRENCY
- VOLATILITY_INDEX
- RATE_YIELD
- FUTURE_SETTLEMENT
- FX_DAILY_CUT

Aunque luego se decida agrupar algunos subtipos, no llamaría "6 clases" a una tabla que tiene ocho unidades operativas.

Y especialmente: `DX-Y.NYB` → `CURRENCY_INDEX`, no `INDEX_EOD_COMMODITY`.

---

## 6. Q-B.6 — Orden de activación

**Dictamen: 🟢 α — aprobado en principio, pero FUTURE no entra en la activación.**

Orden:

1. INDEX_EOD_USA
2. INDEX_EOD_EUROPA
3. VOLATILITY_INDEX
4. RATE_YIELD
5. FX_DAILY_CUT
6. FUTURE_SETTLEMENT → permanece BLOCKED

Esto permite aislar progresivamente las clases. No recomiendo γ: activar simultáneamente todas las clases genera demasiadas variables de fallo.

### Precisión

Agruparía INDEX_EOD_USA + VOLATILITY_INDEX + RATE_YIELD solo si sus validadores y semántica temporal ya están suficientemente separados en código. No debemos confundir que compartan calendario con que tengan idéntica semántica económica.

---

## 7. Q-B.7 — Consumidores de FUTURE bloqueado

**Dictamen: 🟢 β como comportamiento por defecto, no γ.**

No considero suficiente `UNRELIABLE` si el componente sigue entrando numéricamente en el cálculo. Eso produce "dato no auditable + score aparentemente válido", que es precisamente lo que debemos evitar.

Mientras FUTURE está `BLOCKED`:

- `future component` → `excluded / unavailable`
- `consumer recalculates remaining evidence`

Y debe existir metadata explícita:

- `future_status=BLOCKED`
- `future_coverage=0`
- `future_contribution=excluded`

### Qué pasa con los consumidores

No deben romperse ni fingir que los futuros siguen siendo fiables. El cálculo debe degradarse explícitamente.

Para MTE, macro regime, etc.: `BLOCKED futures` → componente ausente → score según política de datos incompletos → estado/coverage documentado.

Esto es más robusto que inventar un `UNRELIABLE` genérico.

Si posteriormente se demuestra que un consumidor necesita obligatoriamente esos futuros para producir un score válido, entonces ese consumidor debe pasar a `INSUFFICIENT`, no fabricar el resultado.

---

## 8. Q-B.8 — A3.1

**Dictamen: 🔴 α — mantener bloqueada.**

No recomiendo desbloquear A3.1 simplemente porque ya tengamos definidas las clases en documento.

La situación es:

- contratos conceptuales: ✅
- implementación completa: ❌
- propagación: ❌
- writers: ❌
- MTE state: ❌
- darkpool: ❌

Por tanto `A3.1 → NO-GO`. La retirada del trim debe producirse **después** de que exista una ruta temporal contractual funcional.

### No aprobado β

Retirar primero para `EQUITY_EOD` deja el mismo `df_market` compartido parcialmente resuelto y parcialmente no resuelto. Eso vuelve a introducir un estado híbrido.

### No aprobado γ

Ya conocemos el coste de eliminar una protección antes de tener su sustituto.

---

## 9. Corrección conceptual importante: "contrato global"

El documento habla de "contratos por clase", pero la arquitectura aprobada anteriormente requiere algo más preciso:
df_market
├── contrato EQUITY_EOD
├── contrato INDEX_EOD_USA
├── contrato INDEX_EOD_EUROPA
├── contrato INDEX_EOD_CURRENCY
├── contrato VOLATILITY_INDEX
├── contrato RATE_YIELD
├── contrato FUTURE_SETTLEMENT
└── contrato FX_DAILY_CUT

text

Cada contrato debe poder producir, como mínimo:

- `effective_date`
- `expected_date`
- `lag_days`
- `coverage`
- `status`

Después debe existir una **regla de consolidación**.

Pero la consolidación no significa "todos deben acabar en una misma fecha". Significa: **el DataFrame conoce qué fecha efectiva corresponde a cada grupo y cada consumidor sabe qué contrato puede utilizar**.

Éste es el punto que debe quedar cristalino en Parte A v2.
---

## 10. Sobre `STALE` y `BLOCKED`

Recomiendo formalizar la máquina:

`PENDING → (resuelto) → OK | STALE | INSUFFICIENT | BLOCKED`

Con esta interpretación:

- `OK` → contrato satisfecho y dentro de frescura esperada.
- `STALE` → contrato satisfecho, pero con lag permitido.
- `INSUFFICIENT` → cobertura insuficiente.
- `BLOCKED` → contrato no auditable / no disponible de forma fiable.
- `PENDING` → contrato definido pero todavía no activado.

Esto es mucho mejor que usar `STALE` como una especie de "OK débil".

---

## 11. Decisión sobre Parte A v2

**Aprobada conceptualmente.**

Pero la versión 2 debe corregir:

- ❌ "6 clases"
- ❌ DX-Y.NYB como commodity
- ❌ confusión entre familia/subclase/contrato

Y dejar:

- ✅ STALE
- ✅ per_pair_lag
- ✅ per_ticker_lag
- ✅ activation_requirement
- ✅ FUTURE BLOCKED

---

## 12. Prioridad del siguiente trabajo

Después de este dictamen:

1. Parte A v2
2. FU-021-5 implementación contractual
3. propagación de metadata
4. writers
5. MTE state
6. darkpool
7. activación secuencial
8. A3.1

Y paralelamente, por ser una dependencia externa:

`FU-021-3C` → investigar proveedor dedicado para futures, sin activar la clase hasta resolverlo.

---

## 13. Dictamen final Q-B.1–Q-B.8

| Pregunta | Decisión |
|---|---|
| **Q-B.1** | 🟢 α — STALE |
| **Q-B.2** | 🟢 α — per_pair_max_lag |
| **Q-B.3** | 🟢 α — per_ticker_lag |
| **Q-B.4** | 🟢 α — FUTURE = BLOCKED |
| **Q-B.5** | 🟢 α — Parte A v2, tras corrección taxonómica |
| **Q-B.6** | 🟢 α — activación secuencial, excluyendo FUTURE mientras BLOCKED |
| **Q-B.7** | 🟢 β — excluir futuros del cálculo, con estado/coverage explícitos |
| **Q-B.8** | 🔴 α — A3.1 continúa bloqueada |

### Dictamen arquitectónico definitivo
FU-021-5 Parte B
↓
🟢 RATIFICADA CON CORRECCIONES DOCUMENTALES

FU-021-5 diseño
↓
🟢 AUTORIZADO

FU-021-5 implementación
↓
🟡 siguiente fase

FUTURE_SETTLEMENT
↓
🔴 BLOCKED

A3.1
↓
🔴 NO-GO

text

La conclusión más importante es ésta:

> **Ya no estamos diseñando "una fecha para `df_market`". Estamos diseñando un conjunto de contratos temporales por grupo homogéneo y una política explícita de cómo conviven dentro de un DataFrame mixto.**

Esa es la arquitectura que debe gobernar los writers, los consumidores y finalmente la retirada de `trim_to_last_valid_date`.

---

## 14. Aplicación de este dictamen

Los dos ajustes obligatorios (taxonomía y corrección de "6 clases") fueron aplicados:

- **`e54bab8`** — anexo VOLATILITY_INDEX + corrección Parte A (5→6 → familias+contratos).
- **`5eda93a`** — Parte A v2 con taxonomía ratificada (`DX-Y.NYB` → `INDEX_EOD_CURRENCY`, `^SPGSCI` → `INDEX_EOD_COMMODITY`).

Los cambios en el diseño de FUTURE_SETTLEMENT y las decisiones sobre Q-B.7 (β) y Q-B.8 (α) fueron reflejados en:

- **Parte A v2 §2.6** — FUTURE_SETTLEMENT como BLOCKED con `activation_requirement`.
- **Parte A v2 §6.4** — tratamiento de consumidores con FUTURE bloqueado (exclusión + metadata explícita).
- **Plan §0.5** — decisiones adicionales D1 y D2.

---

**Fin del dictamen de Parte B.**
**HEAD de referencia:** 0eb7e4f (origin/main).
**Fecha:** 2026-09-16.
---

## 10. Sobre `STALE` y `BLOCKED`

Recomiendo formalizar la máquina:

`PENDING → (resuelto) → OK | STALE | INSUFFICIENT | BLOCKED`

Con esta interpretación:

- `OK` → contrato satisfecho y dentro de frescura esperada.
- `STALE` → contrato satisfecho, pero con lag permitido.
- `INSUFFICIENT` → cobertura insuficiente.
- `BLOCKED` → contrato no auditable / no disponible de forma fiable.
- `PENDING` → contrato definido pero todavía no activado.

Esto es mucho mejor que usar `STALE` como una especie de "OK débil".

---

## 11. Decisión sobre Parte A v2

**Aprobada conceptualmente.**

Pero la versión 2 debe corregir:

- ❌ "6 clases"
- ❌ DX-Y.NYB como commodity
- ❌ confusión entre familia/subclase/contrato

Y dejar:

- ✅ STALE
- ✅ per_pair_lag
- ✅ per_ticker_lag
- ✅ activation_requirement
- ✅ FUTURE BLOCKED

---

## 12. Prioridad del siguiente trabajo

Después de este dictamen:

1. Parte A v2
2. FU-021-5 implementación contractual
3. propagación de metadata
4. writers
5. MTE state
6. darkpool
7. activación secuencial
8. A3.1

Y paralelamente, por ser una dependencia externa:

`FU-021-3C` → investigar proveedor dedicado para futures, sin activar la clase hasta resolverlo.

---

## 13. Dictamen final Q-B.1–Q-B.8

| Pregunta | Decisión |
|---|---|
| **Q-B.1** | 🟢 α — STALE |
| **Q-B.2** | 🟢 α — per_pair_max_lag |
| **Q-B.3** | 🟢 α — per_ticker_lag |
| **Q-B.4** | 🟢 α — FUTURE = BLOCKED |
| **Q-B.5** | 🟢 α — Parte A v2, tras corrección taxonómica |
| **Q-B.6** | 🟢 α — activación secuencial, excluyendo FUTURE mientras BLOCKED |
| **Q-B.7** | 🟢 β — excluir futuros del cálculo, con estado/coverage explícitos |
| **Q-B.8** | 🔴 α — A3.1 continúa bloqueada |

### Dictamen arquitectónico definitivo
FU-021-5 Parte B
↓
🟢 RATIFICADA CON CORRECCIONES DOCUMENTALES

FU-021-5 diseño
↓
🟢 AUTORIZADO

FU-021-5 implementación
↓
🟡 siguiente fase

FUTURE_SETTLEMENT
↓
🔴 BLOCKED

A3.1
↓
🔴 NO-GO

text

La conclusión más importante es ésta:

> **Ya no estamos diseñando "una fecha para `df_market`". Estamos diseñando un conjunto de contratos temporales por grupo homogéneo y una política explícita de cómo conviven dentro de un DataFrame mixto.**

Esa es la arquitectura que debe gobernar los writers, los consumidores y finalmente la retirada de `trim_to_last_valid_date`.

---

## 14. Aplicación de este dictamen

Los dos ajustes obligatorios (taxonomía y corrección de "6 clases") fueron aplicados:

- **`e54bab8`** — anexo VOLATILITY_INDEX + corrección Parte A (5→6 → familias+contratos).
- **`5eda93a`** — Parte A v2 con taxonomía ratificada (`DX-Y.NYB` → `INDEX_EOD_CURRENCY`, `^SPGSCI` → `INDEX_EOD_COMMODITY`).

Los cambios en el diseño de FUTURE_SETTLEMENT y las decisiones sobre Q-B.7 (β) y Q-B.8 (α) fueron reflejados en:

- **Parte A v2 §2.6** — FUTURE_SETTLEMENT como BLOCKED con `activation_requirement`.
- **Parte A v2 §6.4** — tratamiento de consumidores con FUTURE bloqueado (exclusión + metadata explícita).
- **Plan §0.5** — decisiones adicionales D1 y D2.

---

**Fin del dictamen de Parte B.**
**HEAD de referencia:** 0eb7e4f (origin/main).
**Fecha:** 2026-09-16.