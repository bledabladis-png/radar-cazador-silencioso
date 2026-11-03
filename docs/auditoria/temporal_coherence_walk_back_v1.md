# temporal_coherence y estado WALK_BACK - contrato v1

**Fecha:** 2026-11-03.
**Estado:** APROBADO 2026-11-03 (revision de workflows, sesion 27).
**Motivado por:** run scheduled real `41575090209` (03-11 03:55 UTC,
HEAD `db8ed703`), publicado con `temporal_coherence: INCOHERENT` y
`validation_gate: 11/11 (INCOHERENT)`.
**Documentos relacionados:**
- `docs/auditoria/daily_run_gate_contrato_v1.md` (contrato del gate).
- `docs/auditoria/daily_run_gate_enmienda_v1_3.md` (enmienda derivada).
- `src/report/provenance.py` (implementacion).

---

## 1. Motivo

El run `41575090209` publico el reporte con:

    target_session:        2026-11-02
    effective_date_equity: 2026-10-30
    contract_max_lag_days: 0
    temporal_coherence:    INCOHERENT
    validation_gate:       11/11 (INCOHERENT)

El gate (`pipeline_gate`, `guard_coverage`, `write_completion_receipt`)
lo habia aceptado: el manifest `data/stock_prices.parquet.manifest.json`
declaraba `coverage_pct_effective=1.0` sobre `effective_session=2026-10-30`
tras walk-back desde `last_date=2026-11-02` (que tenia cobertura 0.7698,
por debajo del umbral 0.90). El reporte, en cambio, lo marco `INCOHERENT`
porque `provenance._temporal_coherence` usaba un criterio distinto:
`contract_max_lag_days=0`.

## 2. Causa raiz

**Dos criterios incompatibles sobre el mismo hecho.**

- `pipeline_gate` / `guard_coverage` / `write_completion_receipt`:
  aceptan walk-back si `coverage_pct_effective >= umbral` (0.90). Es la
  politica contractual del gate (`enmienda_v1_1`, `enmienda_v1_2`,
  `GUARD-COVERAGE-LAST-DATE-PARCIAL`, commit `a692fafb` 2026-10-31).
- `src/report/provenance.py::_temporal_coherence`:
  comparaba `(target - effective).days` contra `contract_max_lag_days=0`
  sin conocer que el walk-back se habia aplicado.

El walk-back sano (cobertura insuficiente en target, retroceso a la
ultima sesion con cobertura >= umbral, gate lo acepta) es un **evento
legitimo del pipeline**, no una incoherencia. El reporte no tenia forma
de expresarlo.

## 3. Semantica

`_temporal_coherence` produce exactamente uno de 6 estados, con este
orden de precedencia:

| # | Condicion | Estado | Significado |
|---|---|---|---|
| 1 | falta target o effective | `N/D` | Sin datos suficientes |
| 2 | target == effective | `OK` | Coherencia perfecta |
| 3 | effective > target | `ANOMALY` | Integridad (nunca ocurre por diseno) |
| 4 | walk-back explicito | `WALK_BACK` | Retroceso documentado por el pipeline |
| 5 | lag_days <= max_lag_days | `LAG_ADMITTED` | Fuente lenta por diseno |
| 6 | resto | `INCOHERENT` | Stale real, incoherencia no justificada |

**`WALK_BACK` no es `LAG_ADMITTED`.** `LAG_ADMITTED` cubre fuentes
intrinsecamente lentas (p.ej. FINRA con lag regulatorio) y su tolerancia
se declara como `max_lag_days`. `WALK_BACK` cubre un evento del pipeline:
hubo cobertura insuficiente en target y el sistema **decidio** retroceder
a la ultima sesion completa. Es eleccion, no lag inherente.

**`ANOMALY` gana siempre.** Si `effective > target`, el estado es
`ANOMALY` aunque el caller declare `effective_is_walk_back=True`. Es un
fallo de integridad que no se tapa con el flag.

## 4. Fuente de verdad

**`infer_walk_back(effective_meta)` en `src/report/provenance.py`:**

    True  si effective_meta["status"] == "OK" y lag_days > 0
    False en cualquier otro caso (None, dict vacio, status distinto,
                                  lag=0 o invalido)

`effective_meta` es el dict devuelto por
`src.effective_date.resolve_effective_date`, con claves:
`date`, `requested_date`, `lag_days`, `n_eligible`, `n_observed`,
`coverage`, `status`.

`run.py` cablea:

    _prov_walk_back = infer_walk_back(df_stocks_effective_meta)
    _provenance = build_provenance(..., effective_is_walk_back=_prov_walk_back)

## 5. Alineacion con el gate

Ambos criterios usan **la misma fuente**: `coverage_pct_effective` del
manifest `data/stock_prices.parquet.manifest.json`. Si el gate acepta el
walk-back (`coverage_pct_effective >= umbral`), `infer_walk_back` devuelve
`True` (via `status == "OK"` y `lag_days > 0`) y el reporte publica
`WALK_BACK`. Si el gate no lo acepta, no hay publicacion.

**No hay ampliacion de tolerancia.** `contract_max_lag_days` sigue siendo
`0`. La excepcion es un **estado nuevo con nombre propio**, no un numero
mas alto.

## 6. Lo que NO cambia

- `contract_max_lag_days` sigue siendo `0` por defecto.
- `INCOHERENT` sigue siendo detectable: sin walk-back explicito y con
  `lag > max_lag_days`, el estado es `INCOHERENT`. Los tests lo cubren
  (`test_coherencia_sin_flag_sigue_incoherent`).
- `ANOMALY` sin cambios.
- `validation_gate` sigue anadiendo `(INCOHERENT)` cuando hay incoherencia
  real con gate passed (contrato H8 intacto).
- El comportamiento del gate, del receipt y de `guard_coverage` no se toca.

## 7. Cobertura de tests

- `tests/test_report_provenance.py::test_coherencia_walk_back_es_estado_propio`
- `tests/test_report_provenance.py::test_coherencia_walk_back_no_aplica_si_effective_posterior`
- `tests/test_report_provenance.py::test_coherencia_walk_back_no_aplica_si_effective_igual`
- `tests/test_report_provenance.py::test_coherencia_sin_flag_sigue_incoherent`
- `tests/test_report_provenance.py::test_build_provenance_walk_back_no_marca_gate_incoherent`
- `tests/test_report_provenance.py::test_infer_walk_back_*` (6 casos)
- `tests/test_report_provenance.py::test_render_walk_back_incluye_nota_explicativa`
- `tests/test_report_provenance.py::test_render_ok_no_incluye_nota_walk_back`
- `tests/test_report_provenance.py::test_render_incoherent_no_incluye_nota_walk_back`
- `tests/test_report_provenance.py::test_render_lag_admitted_no_incluye_nota_walk_back`
- `tests/test_report_provenance.py::test_run_py_pasa_effective_is_walk_back`

## 8. Rollback

Los 4 commits de implementacion son independientes y reversibles:

- `3c488a22` feat(provenance): estado WALK_BACK + param en _temporal_coherence + build_provenance.
- `e25e051d` feat(pipeline): run.py cablea infer_walk_back.
- `1abddd73` feat(report): nota explicativa en render_provenance.

Revertir `3c488a22` restaura el comportamiento anterior (`INCOHERENT` en
casos de walk-back). Revertir `e25e051d` mantiene el estado pero desactiva
la propagacion (el flag queda `False`). Revertir `1abddd73` elimina la
nota pero conserva el estado.

## 9. Invariantes

- I1: `contract_max_lag_days` permanece `0` en todo el pipeline.
- I2: `INCOHERENT` solo aparece si `effective_is_walk_back == False` Y
  `lag > contract_max_lag_days` Y `effective <= target`.
- I3: `WALK_BACK` solo aparece si `effective_is_walk_back == True` Y
  `effective < target`.
- I4: `ANOMALY` gana sobre `WALK_BACK` si `effective > target`.
- I5: `build_provenance` publica `effective_is_walk_back` en el dict
  (trazabilidad; consumidores pueden auditarlo).
- I6: gate y reporte comparten `coverage_pct_effective` como fuente
  unica de decision sobre walk-back.

## 10. Firma

Diseno e implementacion: sesion 27 (2026-11-03).
Aprobacion: pendiente de dictamen externo si procede.
