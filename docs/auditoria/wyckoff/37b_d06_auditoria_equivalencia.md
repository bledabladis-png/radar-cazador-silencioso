# 37b - Auditoria de equivalencia D-06 (post-freeze)

**Estado:** evidencia post-hoc. NO modifica el paquete FIRMADO.
**Fecha:** 2026-10-03.
**Paquete auditado:** `5b.X v3 FIRMADO / FROZEN` (commit `927d7c5`).
**Motivo:** §14 de `35_protocolo_5bX_v3.md` exige, antes del freeze,
"evidencia de cumplimiento" del mapeo 1:1 protocolo<->script. La
evidencia no se genero antes del freeze; se genera ahora como
documentacion historica. El paquete firmado no se toca.

---

## 1. Objetivo

Verificar que el diff `d324346..d0874f4` en
`scripts/validate_wyckoff_sow_5bX.py` cae integramente en los 6
cambios normativos autorizados por §14 del protocolo v3 y no
introduce ninguna modificacion semantica adicional.

Regla textual (auditor, 2026-10-03):

> el nuevo script es v2 unicamente respecto de los cambios
> normativos aprobados; todo lo demas permanece bit-equivalente
> o funcionalmente equivalente a la implementacion auditada de
> 5b.4-bis.

---

## 2. Cambios normativos autorizados (§14 v3)

| # | Cambio | Punto |
|---|---|---|
| 1 | Umbral: 20 -> 550 confirmed H20-complete | D-06.1 |
| 2 | Universo OOS: `L_date > cutoff` -> `t0_date >= 2026-10-02` | D-06.2 |
| 3 | Retirada de MIN_MONTHS=12 y MIN_N_EPISODES=50 como bloqueantes | D-06.3 |
| 4 | Clasificacion de escenarios por IC95%, no por lift_point | D-06.4 |
| 5 | Elegibilidad H20_complete: `L_idx + 20 < len(dates)` | D-06.5 |
| 6 | Status `BLOCKED_INSUFFICIENT_H20` + `scenario=D` antes de bootstrap | D-06.1 + §6.2 |

---

## 3. Diff: 7 hunks

`git diff d324346..d0874f4 -- scripts/validate_wyckoff_sow_5bX.py`
= 107 insertions, 64 deletions. 7 hunks.

### Hunk 1 - Docstring (lineas 2-13 diff)

    -Protocolo: docs/auditoria/wyckoff/35_protocolo_5bX.md
    +Protocolo: docs/auditoria/wyckoff/35_protocolo_5bX_v3.md

**Clasificacion:** derivado. Actualiza la referencia al protocolo
vigente. No cambia semantica de ejecucion.

### Hunk 2 - Constantes (lineas 14-40 diff)

    +OOS_T0_MIN = pd.Timestamp("2026-10-02")
    +MIN_N_CONFIRMED_H20 = 550
    +GUARDRAIL_MIN_MONTHS = 12
    +GUARDRAIL_MIN_STARTS_OOS = 50
    -MIN_MONTHS = 12
    -MIN_N_EPISODES = 50
    -MIN_N_CONFIRMED = 20

**Clasificacion:** puntos 1, 2, 3. Las tres constantes v1 bloqueantes
se retiran; las dos nuevas de guardrail no bloquean; el umbral sube a
550; el corte OOS pasa a ser fecha explicita.

### Hunk 3 - check_preconditions (lineas 42-119 diff)

- Docstring actualizado.
- Retirada la rama `if n_months < MIN_MONTHS: reasons.append(...)`.
- Retirada la rama `if n_starts_after < MIN_N_EPISODES: reasons.append(...)`.
- Contador pasa de `L_idx > cutoff` a `t0_idx >= OOS_T0_MIN`.
- Calculo `guardrails_ok` sin append a `reasons` (no bloqueante).
- Dict devuelve `oos_t0_min`, `n_candidate_starts_oos`,
  `guardrail_min_*`, `guardrails_ok`, `min_n_confirmed_h20_required`.

**Clasificacion:** puntos 2, 3. El unico bloqueo que queda en
precondiciones es `n_sessions == 0` (ausencia total de datos OOS),
que no es cambio normativo nuevo (no era una de las tres condiciones
v1 retiradas).

### Hunk 4 - classify_scenario + collect_validation_rows + prints (lineas 120-204 diff)

`classify_scenario`:

    -lift = boot["lift_point"]
     lower = boot["lower_ci"]
    -if lift is None or lower is None:
    -    return "INCONCLUSO", ...
    +upper = boot["upper_ci"]
    +if lower is None or upper is None:
    +    return "D", ...
     if lower > 0:
         return "A", ...
    -if lift > 0:
    -    return "B", ...
    -return "C", ...
    +if upper < 0:
    +    return "C", ...
    +return "B", ...

`collect_validation_rows`:

    -def collect_validation_rows(episodes, feats, N, cutoff):
    +def collect_validation_rows(episodes, feats, N, t0_min):
         ...
    -    if e["L_date"] <= cutoff:
    +    if e["t0_date"] < t0_min:
    +        n_dropped_oos += 1
             continue
    +    if e["L_idx"] + 20 >= len(feat["struct"]):
    +        n_dropped_h20 += 1
    +        continue
         m = episode_metrics(...)
         if m is None:
    +        n_dropped_h20 += 1
             continue
         ...
    -    return rows
    +    return rows, n_dropped_oos, n_dropped_h20

Prints de main (protocolo v3, cutoff desarrollo, universo OOS).

**Clasificacion:** puntos 2, 4, 5. Retorno de collect pasa a tupla
para reporte; es consecuencia directa de 2 y 5.

### Hunk 5 - summary BLOCKED (lineas 205-217 diff)

    -"protocolo": "35_protocolo_5bX.md",
    +"protocolo": "35_protocolo_5bX_v3.md",
     ...
    +"cutoff_date": str(CUTOFF_DATE.date()),
    +"oos_rule": f"t0 >= {OOS_T0_MIN.date()}",

**Clasificacion:** derivado. Plomeria de campos de salida.

### Hunk 6 - main, bloqueo por muestra (lineas 218-267 diff)

    -rows = collect_validation_rows(episodes, feats, CANDIDATA["N"],
    -                                CUTOFF_DATE)
    +rows, n_dropped_oos, n_dropped_h20 = collect_validation_rows(
    +    episodes, feats, CANDIDATA["N"], OOS_T0_MIN)
     ...
    -if n_conf < MIN_N_CONFIRMED:
    +if n_conf < MIN_N_CONFIRMED_H20:
     ...
    -"status": "BLOCKED_INSUFFICIENT_CONFIRMED",
    +"status": "BLOCKED_INSUFFICIENT_H20",
     ...
    +"scenario": {"code": "D", "motivo": ...}

**Clasificacion:** puntos 1, 2, 6.

### Hunk 7 - summary EXECUTED (lineas 268-295 diff)

    -"protocolo": "35_protocolo_5bX.md",
    +"protocolo": "35_protocolo_5bX_v3.md",
     ...
    +"oos_rule": f"t0 >= {OOS_T0_MIN.date()}",
     ...
    -"n_confirmed": boot["n_confirmed"],
    -"n_baseline": boot["n_baseline"],
    +"n_confirmed_H20_complete": boot["n_confirmed"],
    +"n_baseline_H20_complete": boot["n_baseline"],
     ...
    +"n_dropped_oos": n_dropped_oos,
    +"n_dropped_h20": n_dropped_h20,

**Clasificacion:** derivado. Renombrado de claves + plomeria. Los
valores subyacentes (`boot["n_confirmed"]`, `boot["n_baseline"]`) son
los mismos. `bootstrap_lift` no se modifica.

---

## 4. Declaracion de conformidad

**Cero cambios no autorizados.**

Cada modificacion del diff `d324346..d0874f4` cae en uno de los 6
puntos normativos autorizados (D-06.1..D-06.5 + §6.2) o en plomeria
derivada (renombrado de fichero de protocolo, adicion de campos de
reporte, cambios de mensajes de consola).

Funciones auditadas como inmutables:

- `find_candidate_starts` (sin cambios).
- `build_episodes_landmark` (sin cambios; se sigue importando del
  calibrador).
- `episode_metrics` (sin cambios; se sigue importando).
- `bootstrap_lift` (sin cambios; el diff no lo toca).
- `precompute_features` / `compute_sow_cache` / `first_sow_in_window`
  (sin cambios; importados).
- `BOOT_B`, `BOOT_SEED` (sin cambios: 2000 / 20261002).

---

## 5. Limitaciones

- Esta auditoria es **visual linea por linea** sobre el diff ya
  volcado a fichero. No es una prueba matematica de equivalencia.
- No sustituye ni modifica la firma del auditor (`927d7c5`). Es
  evidencia post-hoc anadida al expediente.
- No cambia el hash del script v3 (`2b130074...fe7bd8cd`), ni del
  protocolo v3 (`d4b570dc...529218ff`), ni de los tests
  (`215508b7...cf375cb`). El paquete FIRMADO / FROZEN permanece
  inmutable. T7/T7b lo verifican en cada ejecucion de la suite.

---

## 6. Estado

    5b.X v3                            FIRMADO / FROZEN (927d7c5)
    Auditoria de equivalencia (§14)    COMPLETADA (este documento)
    Cambios no autorizados             CERO
    Hash del paquete firmado           INMUTABLE

---

**Fin del informe 37b.**
