# Enmienda v1.3 al contrato del gate

**Contrato base:** `docs/auditoria/daily_run_gate_contrato_v1.md` (2026-10-01).
**Enmiendas previas:** `daily_run_gate_enmienda_v1_1.md` (2026-10-16),
`daily_run_gate_enmienda_v1_2.md` (2026-10-17).
**Fecha:** 2026-11-03.
**Estado:** APROBADO 2026-11-03 (revision de workflows, sesion 27).
**Motivado por:** run `41575090209` (03-11 03:55 UTC, HEAD `db8ed703`),
publicado con `temporal_coherence: INCOHERENT` y `validation_gate: 11/11
(INCOHERENT)` pese a que el gate lo habia aceptado con
`coverage_pct_effective=1.0`.

---

## 1. Motivo

El contrato v1 no menciona `temporal_coherence`. El bloque de provenance
del reporte (H8, 2026-10-16) lo introdujo como campo informativo, pero sin
vincularlo al gate. Con el tiempo, `pipeline_gate` y
`provenance._temporal_coherence` empezaron a usar criterios distintos
sobre el mismo hecho (walk-back sano por cobertura parcial), produciendo
publicaciones con `validation_gate: 11/11 (INCOHERENT)`.

## 2. Cambio

Se declara que **el gate y el reporte comparten `coverage_pct_effective`**
como fuente unica de decision sobre walk-back. Se introduce el estado
`WALK_BACK` en `temporal_coherence` para expresar el walk-back sano
(contrato semantico completo: `docs/auditoria/temporal_coherence_walk_back_v1.md`).

Precedencia en `provenance._temporal_coherence`:

    N/D > OK > ANOMALY > WALK_BACK > LAG_ADMITTED > INCOHERENT

La fuente de verdad del flag es `infer_walk_back(effective_meta)` sobre
el dict de `src.effective_date.resolve_effective_date`.

## 3. Alcance

### 3.1. Cambia

- `src/report/provenance.py::_temporal_coherence`: nuevo parametro
  `effective_is_walk_back`, nuevo estado `WALK_BACK`.
- `src/report/provenance.py::build_provenance`: nuevo kwarg
  `effective_is_walk_back=False`, propagado al dict de salida
  (`"effective_is_walk_back": bool`) para trazabilidad.
- `src/report/provenance.py::infer_walk_back(effective_meta)`: nueva
  funcion publica (bool).
- `src/report/provenance.py::render_provenance`: nota explicativa cuando
  `temporal_coherence == "WALK_BACK"`.
- `run.py`: computa `_prov_walk_back = infer_walk_back(df_stocks_effective_meta)`
  y lo pasa a `build_provenance`.

### 3.2. No cambia

- Gate (`pipeline_gate`, `guard_coverage`, `write_completion_receipt`):
  sin cambios. Ya aceptaban walk-back sano con `coverage_pct_effective`.
- `contract_max_lag_days`: sigue `0`.
- Deteccion de `INCOHERENT` real: preservada.
- `ANOMALY`: sin cambios.
- Marca `(INCOHERENT)` en `validation_gate`: sin cambios cuando hay
  incoherencia real.
- Schema del receipt, manifests, `_probe_panel`: sin cambios.

### 3.3. Anade

- Contrato semantico `temporal_coherence_walk_back_v1.md`.
- Item `TEMPORAL-COHERENCE-WALK-BACK` en `docs/gobernanza/REGISTRO.yaml`.
- 18 tests nuevos en `tests/test_report_provenance.py`.

## 4. Justificacion

Sin este cambio, el reporte publica `INCOHERENT` en runs scheduled con
walk-back activo (frecuencia: cuando Yahoo entrega Close parcial al slot
del dia). El usuario ve `validation_gate: 11/11 (INCOHERENT)` en un run
que el gate considero correcto. Es una contradiccion visible en el
artefacto publico. El propio codigo de provenance lo senalaba como
"contradiccion: el gate no deberia haber pasado" (comentario H8,
2026-10-16) sin resolverla.

Alternativas descartadas:

- **Ampliar `contract_max_lag_days`**: relajaria la deteccion de
  `INCOHERENT` real. Rechazada.
- **Silenciar `INCOHERENT`**: perderia capacidad de deteccion. Rechazada.
- **Dejar el bug como WONT_FIX**: el reporte publica una contradiccion
  visible al usuario. No es cosmetico. Rechazada.

## 5. Rollback

Revertir los 3 commits de implementacion:

- `3c488a22` feat(provenance): estado WALK_BACK.
- `e25e051d` feat(pipeline): run.py cablea.
- `1abddd73` feat(report): nota en render.

Revertir `3c488a22` restaura el comportamiento previo. Ver contrato
semantico seccion 8 para detalle.

## 6. Invariantes

- I1: `contract_max_lag_days` permanece `0`.
- I2: `INCOHERENT` solo aparece sin walk-back explicito y con lag > max.
- I3: `WALK_BACK` solo aparece con walk-back explicito y `effective < target`.
- I4: `ANOMALY` gana sobre `WALK_BACK`.
- I5: `effective_is_walk_back` se publica en el dict de provenance.
- I6: gate y reporte comparten `coverage_pct_effective` como fuente unica.

## 7. Firma

Diseno e implementacion: sesion 27 (2026-11-03).
Aprobacion: pendiente de dictamen externo si procede.
