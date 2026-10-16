# Enmienda contractual v1.1 al contrato del gate del daily_run

**Contrato base:** `docs/auditoria/daily_run_gate_contrato_v1.md` (2026-10-01).
**Fecha:** 2026-10-16.
**Estado:** APROBADO por dictamen externo del 2026-10-16 (segunda ronda).
**Schema del receipt:** se mantiene en v1. No hay cambio estructural.

---

## 1. Alcance

### 1.1. Cambia

- `scripts/guard_coverage.py`: **deja de estar prohibido modificarlo.**
  Se anade validacion contractual compartida con el escritor del
  receipt. La logica de decision temporal unificada vive en
  `src/temporal_contracts/validation.py`.

- `scripts/write_completion_receipt.py`: incorpora la misma validacion
  contractual **como precondicion de emision**. El receipt solo se
  escribe si `validate_artifact_contracts()` devuelve `OK` o `WARN`.
  Si devuelve `BLOCKED` o `UNAVAILABLE`, el script aborta sin emitir
  receipt (fail-closed).

- Nuevo modulo: `src/temporal_contracts/validation.py`. Funcion publica
  `validate_artifact_contracts(manifest) -> ValidationResult`.

- `src/temporal_contracts/registry.py`: anadido campo `critical: bool`
  a cada entrada del catalogo. Los contratos marcados `critical=True`
  gobiernan el bloqueo de la finalizacion.

- Nuevo registro diagnostico: `outputs/state/completion_attempt.jsonl`.
  Persistido como artifact de cada ejecucion (no commiteado al repo).
  Documenta cada intento del pipeline, exitoso o no.

### 1.2. NO cambia

- Schema JSON del receipt: se mantiene en v1. Mismos campos.
- Fuente de CURRENT: `find_completion_receipt()` sigue siendo la
  autoridad. `_manifest_satisfies` sigue sin participar.
- Invariantes I1-I8 del contrato base: preservadas (ver §5).
- Retencion de artifacts (90 dias) y semantica de expiracion.
- Logica de `probe_panel` y de `resolve_slot_flags`.

### 1.3. Sustituye

La clausula 1.2 del contrato base:

> *"`scripts/guard_coverage.py`: sin tocar."*

Se sustituye por: el guard puede extenderse con validacion contractual
adicional, manteniendo su contrato original de cobertura y sha256.

---

## 2. Nuevo contrato de validacion

### 2.1. Estados

| Estado | Significado | Efecto |
|---|---|---|
| `OK` | Todos los contratos obligatorios cumplen | Receipt autorizado |
| `WARN` | Desviaciones permitidas | Receipt autorizado con aviso |
| `BLOCKED` | Contrato obligatorio incumplido | Sin receipt |
| `UNAVAILABLE` | Evidencia insuficiente para evaluar contrato obligatorio | Sin receipt |

`UNAVAILABLE` NO se convierte en `OK` por ausencia de errores
explicitos. Es un estado conservador.

### 2.2. Reglas de bloqueo

Para cada contrato marcado `critical=True` en el registro:

- `by_market.<mercado>.status == "INVALID"` -> `BLOCKED`.
- `by_market.<mercado>.coverage_at_session < min_coverage` -> `BLOCKED`,
  **aunque el status publicado diga otra cosa** (inconsistencia).
- `by_market.<mercado>.n == 0` (sin observaciones) -> `BLOCKED`.
- `by_market.<mercado>` ausente -> `UNAVAILABLE`.

Para contratos no criticos: los mismos casos generan `WARN`, no bloqueo.

### 2.3. Resolucion de `expected_session`

`expected_session` se resuelve por contrato, no global. Para `EQUITY_EOD`
coincide con la sesion NYSE esperada por `reference_date`. Para otros
contratos (FUTURE_SETTLEMENT, FX_DAILY_CUT) puede diferir.

### 2.4. Autoridad unica

`validate_artifact_contracts()` es la **unica** fuente de decision
temporal contractual. El guard y el escritor del receipt la consumen
desde el mismo modulo. Ninguna otra parte del sistema implementa
reglas temporales paralelas.

---

## 3. Registro de intentos incompletos

Fichero: `outputs/state/completion_attempt.jsonl` (una linea JSON por
intento). Persistido como artifact con nombre `completion-attempt-<run_id>`.

Campos minimos por entrada:

| Campo | Tipo | Finalidad |
|---|---|---|
| `run_id` | int | Identificador del run |
| `run_attempt` | int | Numero de intento del run |
| `target_session` | str ISO | Sesion objetivo |
| `attempt_timestamp` | str ISO | Cuando se realizo el intento |
| `outcome` | str | `INCOMPLETE` \| `BLOCKED` \| `SUCCESS` |
| `failed_contracts` | list[str] | Contratos incumplidos |
| `reasons` | list[str] | Explicacion |
| `manifest_hashes` | dict | sha256 de los manifests evaluados |

**No se commitea al repositorio.** Se sube como artifact al final de
cada ejecucion (exitoso o no), con `if: always()`.

El issue-manager puede abrir incidencia si los bloqueos se repiten,
pero no es requisito para demostrar el fallo.

---

## 4. Enmienda al contrato v1: campo `critical`

`registry.py` declara `critical: bool` por contrato. Valores iniciales:

| Contrato | critical | Justificacion |
|---|---|---|
| EQUITY_EOD | `True` | Universo principal del analisis sectorial (539 tickers) |
| INDEX_EOD_USA | `False` | Indices de referencia; fallo no invalida el analisis |
| INDEX_EOD_EUROPA | `False` | Idem |
| INDEX_EOD_COMMODITY | `False` | Idem |
| INDEX_EOD_CURRENCY | `False` | Idem |
| VOLATILITY_INDEX | `False` | Idem |
| RATE_YIELD | `False` | Idem |
| FUTURE_SETTLEMENT | `False` | Idem |
| FX_DAILY_CUT | `False` | Idem |

La criticidad es **explicita por contrato**, no por umbral dinamico de
volumen. El auditor externo del 2026-10-16 rechazo el criterio de
porcentaje (>= 50% de filas): la criticidad depende de la funcion del
universo en el pipeline, no de su tamano.

---

## 5. Reconciliacion con invariantes I1-I8

| Invariante | Estado |
|---|---|
| I1 — Nunca CURRENT por ausencia de parquet | Preservada |
| I2 — Nunca CURRENT por manifest aislado | Preservada |
| I3 — Receipt solo se genera tras ultimo check | Preservada (el nuevo control se anade antes) |
| I4 — Receipt inmutable e identifica ejecucion | Preservada (schema v1 sin cambios) |
| I5 — Expiration fail-open | Preservada |
| I6 — API failure != CURRENT | Preservada |
| I7 — Target session exacta | Preservada |
| I8 — Receipt no sustituye manifest | **Reforzada** (precondicion adicional) |

Ninguna invariante se rompe. I8 se refuerza en la direccion del
dictamen del 2026-10-16.

---

## 6. Rollback

Revertir los commits de la implementacion (identificados en la
bitacora) restaura el comportamiento anterior:

- Guard vuelve a solo cobertura + sha256.
- Receipt vuelve a solo VG 10/10 + written_at.
- `completion_attempt.jsonl` queda como diagnostico sin efecto.

El schema del receipt nunca cambio, por lo que no hay migracion de
artefactos previos.

---

## 7. Firma

**APROBADO 2026-10-16.** Fundamento: dictamen externo del 2026-10-16
(segunda ronda), que acepta el diseno con las precisiones:

1. Criticidad explicita por contrato, no umbral dinamico.
2. `src/temporal_contracts/validation.py` como ubicacion canonica.
3. Opcion (b) para registro de intentos: artifact por ejecucion.
4. Test R3 explicito para `sector_breadth` / `sector_breadth_momentum`.
5. Guard semantico STAGFLATION con condicion `IPS > 0.15 AND SRS > 0`.
6. Enmienda contractual publicada antes de tocar el guard.

**Fin de la enmienda.**