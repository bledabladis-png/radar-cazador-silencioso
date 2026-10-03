# Auditoria linea a linea - indicators/wyckoff_v1.py (v1.8)

- **Auditoria:** 2026-10-03.
- **Fichero:** `indicators/wyckoff_v1.py` (493 LOC tras el fix de W-01..W-03).
- **Version:** v1.8 (contrato `01_contrato_semantico_v1_8.md`).
- **Contrato de auditoria:** Fase 0 + barrido de patrones + revision
  linea a linea + cotejo invariantes.
- **Dictamen:** CERRADO CON FIX. Cero ALTA, cero MEDIA. 4 BAJA
  (W-01..W-03 corregidas, W-04 trazabilidad). 3 INFO.

## 1. Contexto

El dictamen externo `38_dictamen_final_5bX_v3.md` (§5) declara
literalmente:

> "No he inspeccionado fisicamente el diff `d324346..d0874f4`;
> por tanto, no voy a afirmar que lo he revisado linea por linea.
> Mi firma se emite sobre el paquete contractual y la evidencia
> presentada, no como certificacion de una inspeccion visual
> independiente del diff que no he realizado."

Esta auditoria cubre exactamente ese hueco: inspeccion linea a linea
del modulo v1.8 y cotejo contra el contrato.

Ademas, confirmacion de estado operativo: el pipeline del 2026-10-03
(run 37132181161, SHA `b7dd6a3`) ejecuto el modulo LEGACY
(`indicators/wyckoff.py`), no v1.8. Todos los consumidores directos
importan `from indicators.wyckoff import ...`. Esto no es bug: el
plan de migracion (`03_plan_migracion.md`) tiene el Frente D
(migracion de consumidores) BLOQUEADO hasta cierre 5b.X, prohibicion
vigente del dictamen §3.
## 2. Metodo

- **Fase 0** (read-only): inventario de funciones, constantes, docs,
  consumidores, tests.
- **Fase 1**: barrido de los 13 patrones de bug (`01b_PATRONES.md`).
- **Fase 2**: verificacion caso a caso de los hits del barrido.
- **Fase 3**: cotejo de invariantes I1-I36 contra tests por nombre.
- **Fase 4**: clasificacion y fix de BAJA/INFO.

Sin E2E: el modulo no esta integrado al pipeline (legacy activo). La
verificacion se limita a tests del contrato.

## 3. Barrido de los 13 patrones (Fase 1)

| # | Patron | Hits | Comentario |
|---|---|---|---|
| 1 | `except Exception` outer | 0 | Solo mencion en comentario L239 |
| 2 | Check muerto (siempre True/False) | 0 | -- |
| 3 | tz-naive vs tz-aware | 0 | -- |
| 4 | Aridad variable de retorno | 0 | -- |
| 5 | Bucle sin guard | 0 | -- |
| 6 | Variante local de funcion canonica | 0 | -- |
| 7 | Test anclado a numero de linea | 0 | -- |
| 8 | Dead code por refactor | 3 | W-01, W-02, W-03 |
| 9 | Fallback silencioso en provider | 0 | -- |
| 10 | Documento vs codigo | 0 | Los 6 parametros normativos coinciden |
| 11 | mtime como criterio de freshness | 0 | -- |
| 12 | Cache-hit valida forma, no calidad | 0 | -- |
| 13 | Consumidor accede por indice sin guard | 0 | 5 `.iloc[-1]` con `.empty` previo |

## 4. Hallazgos

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| W-01 | BAJA | L69 | `C_NORM_DISTR_MIN = -0.10`. Residuo de v1.4 (`09efc1a`). Superseded por v1.6. Sin uso en codigo, tests ni scripts. Solo mencionado en `01_contrato_semantico_v1_4.md` (historico). | ELIMINADO (`036c4ef`). |
| W-02 | BAJA | L74 | `PREC_STRUCT_NEG = -0.30`. Introducido en `45e1a80` como PROPUESTO. Nunca activado. Sin uso. Solo mencionado en `02_proveniencia_parametros.md` como propuesta. | ELIMINADO (`036c4ef`). |
| W-03 | BAJA | L83 | `ALL_FASES`. Tupla auxiliar introducida en `e4af0c2`. Sin uso en ningun sitio. | ELIMINADO (`036c4ef`). |
| W-04 | BAJA | tests | Invariantes I16, I17, I18 no mencionadas por nombre en los tests que las cubren funcionalmente. Cotejo daba 33/36 por nombre. | ANOTADO en docstrings (commit `de7d729`). Cero cambio funcional. |

## 5. Falsos positivos descartados

- **`T_NORM_WEAK = 0.30` igual a `T_NORM_STRONG = 0.30`.** El
  contrato §5.2 dice `t_norm_t < T_NORM_WEAK OR |t_norm_t| < T_NORM_WEAK`.
  El codigo implementa `(last_t < 0) or (abs(last_t) < T_NORM_WEAK)`.
  Ambas formulaciones dan la misma condicion para `T_NORM_WEAK = 0.30`:
  `t_norm_t < 0.30`. El contrato es redundante en su formulacion;
  el codigo aterriza a la union. INFO.
- **`build_ticker_df` fillna.** Contrato §8.3 lo documenta
  explicitamente: "Close autoritativo. Open/High/Low con fallback
  a Close. Volume con 0." Alineado. INFO.
- **`wyckoff_stability` sin consumidor en pipeline.** El contrato
  §8.2 lo declara API publica. No es dead code: es API expuesta
  pendiente de integracion. INFO.
- **5 `.iloc[-1]` sin guard explicito.** Todos precedidos por
  `.dropna().empty` o `.empty` que garantiza no-vacio.
- **`detect_sow` con `sow_params.get("max_age_m", 10)`** — default
  local distinto del contrato v1.9 (`M = 30`). Pero el contrato v1.8
  no fija M (fail-closed); el default 10 es defensivo. INFO.
## 6. Cotejo invariantes I1-I36 (Fase 3)

Antes: 33/36 mencionadas por nombre en `test_wyckoff_v1_contract.py`.
Despues de W-04: 36/36.

| Bloque | Invariantes | Estado |
|---|---|---|
| I1-I10 | Base (composicion, rangos, determinismo) | 10/10 |
| I11-I14 | Precedente (no look-ahead, no circular) | 4/4 |
| I15-I18 | v1.2 (MARKUP/MARKDOWN sin precedente, tact no veta, selectivo) | 4/4 (I16-I18 anotadas en W-04) |
| I19-I23 | v1.3 (t_norm signo, aceleracion, T1) | 5/5 |
| I24-I26 | v1.4/v1.5 (K, DISTRIBUTION alcanzable) | 3/3 |
| I27-I29 | v1.6 (candidate sin SOW, SOW sin candidate, flag) | 3/3 |
| I30-I33 | v1.7 (ATR ex-ante, break_depth, no ruptura, no look-ahead) | 4/4 |
| I34-I36 | v1.8 (fail-closed, config None, sin SOW no DISTRIBUTION) | 3/3 |

## 7. Verificacion

- `py -m pyflakes indicators\wyckoff_v1.py` -> silencio.
- `py -m pytest tests\test_wyckoff_v1_contract.py -q` -> 59 passed
  + 3 skipped (los 3 por fase 5b: `as_of` / precedente).
- `py -m pytest tests\ -q` -> 3175 passed + 5 skipped (0 regresion).

## 8. Estado operativo del modulo en el pipeline

**v1.8 NO esta integrado.** Todos los consumidores directos del
pipeline importan el legacy:

- `indicators/index_leaders.py:4`
- `indicators/index_phase.py:3`
- `indicators/sector_breadth.py:11`
- `indicators/sector_wyckoff_distribution.py:10`
- `indicators/stock_leader.py:4`
- `regimes/sector_regime.py:13`

Esto es coherente con el plan de migracion: el Frente D (migracion
de consumidores) esta BLOQUEADO hasta cierre de 5b.X. La prohibicion
"migracion de consumidores" es explicita en el dictamen final §3.

El run del 2026-10-03 (37132181161, SHA `b7dd6a3`) ejecuto el
pipeline con legacy. No es bug ni regresion; es el estado esperado
del proyecto.

## 9. Dictamen

CERRADO CON FIX. Cero ALTA, cero MEDIA. 4 BAJA (3 dead code
eliminadas, 1 trazabilidad anotada). 3 INFO WONT FIX razonado.

El modulo v1.8 queda auditado linea a linea. El contrato v1.8 tiene
cobertura nominal 36/36 de sus invariantes.

## 10. Proximo

El modulo no tiene trabajo pendiente hasta que cierre 5b.X. Los
frentes posteriores del plan (5c comparativa, 5d migracion,
5e retirada legacy) estan desbloqueables en el orden documentado
en `03_plan_migracion.md`.

**Nota sobre el estado del pipeline:** el run diario sigue con
legacy. Cualquier cambio de comportamiento observable en
`outputs/history/sector_wyckoff_distribution.csv` u otros outputs
no proviene de v1.8 (no integrado), sino del legacy.

---

**Fin de la auditoria v1.8.**