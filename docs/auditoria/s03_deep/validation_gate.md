# S-03-deep - src/pipeline/validation_gate.py

- **Auditoria:** 2026-10-03
- **HEAD:** ae165af
- **Fichero:** `src/pipeline/validation_gate.py` (188 LOC, 9.8 KB)
- **mtime:** 2026-10-03 (modificado en D-05, commit e0aacda)
- **Contrato:** Fase 0 (inventario) + revision linea a linea + contraste con tests.
- **Dictamen:** CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA.

## 1. Alcance

188 LOC. 10 checks secuenciales que acumulan en `validation_checks` (OK)
o `validation_errors` (fail). Extraido de `run.py` en refactor C2-11
(commit d97453f). Ultimo cambio funcional: D-05 (2026-10-03, commit
e0aacda) normaliza `reference_date` a naive antes de calcular edades.

Consumidor: `run.py` (raiz del repo) decide abortar si `passed` es False.
Receipt: `scripts/pipeline_gate.py:90` exige "10/10" en el
`completion_receipt.json` del `daily_run`.

## 2. Hallazgos

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| F-01 | BAJA | L191 + L201-204 | `audit_double_counting()` llamado 2 veces. Funcion pura (itera `DIRECT_DEPENDENCIES`, sin estado). Resultados identicos. | WONT FIX: redundancia, ROI < 1. |
| F-02 | INFO | L86-100 | Check 5 registra "OK N sectores" aunque haya errores por ticker (que van a `validation_errors`). Doble semantica: capacidad de calculo vs validez de valores. | WONT FIX: intencional, confirmado por `test_check5_tipo_invalido_no_crashea`. |
| F-03 | INFO | L43 | Fallback `datetime.now()` si `reference_date is None`. 7 de 8 tests dependen de el. | WONT FIX: contrato de API, no viola regla §2 (que aplica a metricas publicadas, no a calculos internos de age). |
| F-04 | BAJA | L55, L77, L85 | `pd.isna(val)` sin guarda de tipo previa. Con listas/arrays lanzaria `ValueError`. | WONT FIX: fuera de contrato (API acepta escalares). |
| F-05 | TRIVIAL | L144, L165 | `except (ValueError, TypeError, OSError)` - `OSError` residual, `pd.Timestamp` no lo lanza. | WONT FIX: sin efecto. |
| F-06 | INFO | L43-46 | Comentario D-05 cita "L129 y L147" que ya no coinciden con las lineas actuales. | WONT FIX: cosmetico. |
| F-07 | INFO | L173 | `lis_in_state_machine` con logica invertida respecto al nombre (True -> check falla). | WONT FIX: cosmetico. |

## 3. Contraste con tests

`tests/test_validation_gate.py` (8 tests, 4828 B, mtime 2026-10-03).

Cubren:
- Check 1: SLPM con `validation_errors` -> gate falla.
- Check 2: `pd.NA` en `total_pcr` -> gate falla sin crash.
- Check 3: `NaN` en `media_dark_pool` -> gate falla.
- Check 4: `pd.NA` en `msi` -> gate falla sin crash.
- Check 5: sin interseccion tactical/structural -> gate falla.
- Check 5: valor no numerico -> gate falla sin crash.
- D-05: `reference_date` tz-aware -> freshness reporta dias, no "sin fecha".
- Caso feliz: 10 checks OK, `passed=True`.

No cubren:
- F-01 (doble llamada): no mockea `audit_double_counting`.
- F-03 (fallback): 7 tests dependen de `_ref` por fallback.
- `dc_summary` solo se testea indirectamente.

## 4. Reclasificaciones respecto al primer pase

Informe inicial clasifico F-01 MEDIA, F-02 MEDIA, F-04 BAJA. Tras
contraste con `src/dependency_tracker.py` (funcion pura) y con los tests
(F-02 intencional), reclasificados:

- F-01: MEDIA -> BAJA (funcion pura, sin divergencia posible).
- F-02: MEDIA -> INFO (doble semantica intencional).
- F-04: sin cambio (BAJA).

Patron de error propio: clasificar antes de verificar. Mismo error que
seccion 1 de 01_METODO advierte ("Ver el contenido real antes del patch").

## 5. Falsos positivos descartados

- Check 1: `slpm_errors = None` cae en rama OK. Correcto.
- Check 4: doble guarda (None + isinstance + pd.isna). Igual que F-04.
- Check 10: `inspect.signature(classify_leadership_state)` con
  `ImportError`/`AttributeError` capturados. No hay bug.
- `_ref - d` con datetime naive menos Timestamp naive: funciona
  (Timestamp es subclase de datetime).

## 6. Dictamen

CERRADO SIN CAMBIOS. Cero hallazgos ALTA, cero MEDIA.
`validation_gate.py` queda auditado post-D-05. Ninguna accion pendiente.

## 7. Proximo

Siguiente fichero con mtime 10-03 en `src/pipeline/`:
`mte_confirmation.py` (146 LOC, 7.7 KB). Tambien tocado en D-05.
