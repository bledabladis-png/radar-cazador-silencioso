# Dictamen del auditor externo — Plan de implementación FU-021-5

**Fecha:** 2026-09-16
**HEAD de referencia al recibir el dictamen:** eb14bf3 (origin/main)
**Documento evaluado:** FU-021-5_PLAN_IMPLEMENTACION.md (eb14bf3)
**Estado:** dictamen completo. 8 preguntas Q-P resueltas. 2 ajustes obligatorios.

---

## 0. Resultado global

**GO para el plan**, con dos ajustes de diseño obligatorios antes de comenzar la implementación.

La secuencia general es correcta y conserva la disciplina de separación entre contrato, implementación, activación y A3.1.

Los dos ajustes son:

1. **Q-P.3:** no adoptar `df.attrs` como mecanismo canónico sin prueba de persistencia e integridad semántica durante las operaciones reales. Decisión: `dict` paralelo como contrato arquitectónico. `df.attrs` puede utilizarse internamente como caché auxiliar.
2. **Q-P.7:** `α`, pero con snapshots/checksums del histórico antes de cada migración. Preservar el CSV no significa asumir que sus filas son comparables sin etiquetar el cambio de metodología.

El resto queda aprobado.

---

## 1. Q-P.1 — Orden de las 9 fases

**Dictamen: 🟢 α — APROBADO.**

La secuencia es correcta:
infraestructura

contratos individuales

consolidación

propagación

writers

MTE

darkpool

activación

A3.1

```

Dependencia esencial fijada:

> **A3.1 es la última fase funcional del cambio, no una fase paralela.**

Dentro de la Fase 5, los writers no deben "interpretar" el contrato por su cuenta. Deben consumir metadata ya resuelta.

---

## 2. Q-P.2 — Ubicación de módulos

**Dictamen: 🟢 α — `src/temporal_contracts/`.**

Aprobado.

La estructura de directorio debe mantener clara la frontera:
src/temporal_contracts/
infrastructure
contracts
resolver
consolidation

```

No hace falta imponer esos subdirectorios ahora, pero no mezclar los contratos temporales con `market_hours.py`, `utils.py` o `pipeline/`.

---

## 3. Q-P.3 — `df.attrs` vs dict paralelo

**Dictamen: 🟡 β como contrato arquitectónico; α solo como mecanismo auxiliar sujeto a prueba.**

No aprobaría `df.attrs` como única fuente de verdad.

Razón: `df.attrs` depende de cómo pandas propague/copie atributos durante operaciones como:
.loc
.copy
concat
merge
groupby
transformaciones intermedias

```

Aunque un test demuestre que sobrevive hoy, eso no lo convierte en un contrato fuerte.

### Arquitectura aprobada
load/resolve
↓
df_market
+
temporal_meta
↓
consumidores

```

Donde `temporal_meta` es una estructura explícita retornada/transportada por la capa de pipeline.

`df.attrs['temporal_meta']` puede mantenerse como cache/conveniencia, pero **no como autoridad contractual**.

### Consecuencia

Fase 3 debe probar ambos mecanismos si se quieren usar ambos:
dict paralelo → autoridad
df.attrs → espejo opcional

```

Si solo se quiere implementar uno, elegir **β**.
---

## 4. Q-P.4 — `EQUITY_EOD`

**Dictamen: 🟢 α — wrapper sobre `resolve_effective_date`.**

Aprobado.

No debemos crear:
FU-021-5 EQUITY_EOD
+
FU-020 resolve_effective_date

```

como dos algoritmos.

La jerarquía debe ser:
resolve_effective_date()
↓
EQUITY_EOD wrapper
↓
metadata + estado contractual

```

Así se conserva una única lógica de cobertura.

**No autorizado γ ahora.** Un refactor de extracción solo tendría sentido si aparecen duplicaciones reales durante implementación.

---

## 5. Q-P.5 — Granularidad de commits

**Dictamen: 🟢 α — ~30 commits.**

Aprobado.

Regla añadida:

> **Un commit puede agrupar varios archivos solo cuando constituyen una unidad funcional que no tiene sentido validar por separado.**

No se quieren "30 commits artificiales"; se quieren **30 unidades reversibles y testeables**.

Ejemplo aceptable:
commit
infraestructura contrato
→ tests infraestructura

```

Ejemplo no aceptable:
commit
modifica 11 writers + MTE + darkpool

```

---

## 6. Q-P.6 — Activación secuencial

**Dictamen: 🟢 α — uno a uno.**

Aprobado.

El hecho de compartir calendario no implica equivalencia económica ni equivalencia de proveedor.

Orden:
INDEX_EOD_USA

INDEX_EOD_EUROPA

VOLATILITY_INDEX

RATE_YIELD

FX_DAILY_CUT

```

Y:
FUTURE_SETTLEMENT → BLOCKED

```

### Condición de cada activación

No basta con "el código pasa tests".

Debe existir:
pre-activation snapshot
→ workflow/E2E
→ manifest/metadata
→ consumer checks
→ post-activation comparison

```

Esto permitirá atribuir cualquier cambio.

---

## 7. Q-P.7 — Rollback de `darkpool`

**Dictamen: 🟢 α — preservar histórico, con protección adicional obligatoria.**

No restaurar/eliminar `darkpool_history.csv` durante rollback.

Antes de migrar:
SHA256 del histórico
+
nº filas
+
última fecha
+
backup/snapshot

```

Y el nuevo proceso debe poder distinguir:
datos históricos anteriores a migración
vs
datos producidos por nueva implementación

```

Por tanto:
CSV preserved
+
migration boundary documented

```

No es necesario `v2.csv` por ahora.
---

## 8. Q-P.8 — `FU-021-3C-bis`

**Dictamen: 🟢 α — paralelo.**

Aprobado.

Es independiente del código del pipeline y ataca directamente el único contrato todavía bloqueado:
FUTURE_SETTLEMENT

```

Mientras el equipo desarrolla:
EQUITY/INDEX/RATE/FX

```

puede investigarse:
provider dedicado
settlement oficial
contratos explícitos
rollover

```

Esto no debe convertirse en una rama de implementación simultánea dentro del mismo código hasta que exista una decisión de provider.

---

## 9. Dos decisiones adicionales fijadas

### D1 — No existe "fecha global" de `df_market`

El plan debe tratar siempre:
temporal_meta
├── per_contract
└── consolidated

```

No:
df_market.effective_date = una sola fecha económica

```

Puede existir una **fecha de consolidación técnica**, pero no sustituye las fechas por contrato.

### D2 — `BLOCKED` no equivale a `STALE`

Debe conservarse:
STALE
→ contrato válido, lag dentro de tolerancia

BLOCKED
→ contrato no auditable/activable

```

Especialmente importante para FUTURE.

---

## 10. Respuestas finales Q-P.1–Q-P.8

| Pregunta | Decisión |
|---|---|
| **Q-P.1** | 🟢 α — orden aprobado |
| **Q-P.2** | 🟢 α — `src/temporal_contracts/` |
| **Q-P.3** | 🟢 β como autoridad; `attrs` solo auxiliar |
| **Q-P.4** | 🟢 α — wrapper |
| **Q-P.5** | 🟢 α — ~30 commits, unidades funcionales |
| **Q-P.6** | 🟢 α — activación uno a uno |
| **Q-P.7** | 🟢 α — preservar histórico + snapshot |
| **Q-P.8** | 🟢 α — investigación paralela |

---

## 11. Estado de autorización
FU-021-5
├── Plan general 🟢 GO
├── Infraestructura 🟢 GO
├── Contratos 🟢 GO
├── Metadata 🟢 GO
├── Writers 🟢 GO
├── MTE 🟢 GO
├── darkpool 🟢 GO dentro de FU-021-5
├── Activación 🟢 GO secuencial
├── FUTURE_SETTLEMENT 🔴 BLOCKED
├── FU-021-3C-bis 🟢 GO paralelo
└── A3.1 🔴 NO-GO hasta completar fases 1–8

```

**El plan queda aprobado para pasar a implementación.**

La condición previa al primer patch es actualizar el diseño de la Fase 3 para que `temporal_meta` tenga autoridad en un transporte explícito y `df.attrs` no sea la única fuente de verdad. Después de esa corrección, queda autorizado el primer commit de infraestructura.

---

## 12. Aplicación al plan

Los dos ajustes obligatorios (Q-P.3 y Q-P.7) fueron aplicados al plan en el commit `698eac3`:

- **§2.5 reescrita:** `temporal_meta` como `dict` paralelo autorizado; `df.attrs` solo espejo auxiliar.
- **§0.5 insertada:** tabla de decisiones del dictamen.
- **Fase 3 ampliada:** test empírico de persistencia (referencia) + transporte explícito.
- **Fase 7 ampliada:** snapshot + migration boundary para `darkpool_history.csv`.

Las otras 6 ratificaciones (Q-P.1, Q-P.2, Q-P.4, Q-P.5, Q-P.6, Q-P.8) no requirieron cambios al plan. Quedan ratificadas por este dictamen.

---

**Fin del dictamen del plan de implementación.**
**HEAD de referencia:** eb14bf3 (origin/main).
**Fecha:** 2026-09-16.
