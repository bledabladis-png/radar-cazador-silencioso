# S-03-deep - bloque D-05/D-04 (3 ficheros)

Auditoria agrupada de los 3 ficheros restantes con mtime 2026-10-03
(tocados por D-05 y D-04). Los otros dos del bloque
(`validation_gate.py`, `mte_confirmation.py`) se auditaron por separado.

- **Auditoria:** 2026-10-03
- **HEAD inicial:** f368fa9
- **Ficheros:**
  - `src/pipeline/breadth_metrics.py` (156 LOC, 8 KB)
  - `src/pipeline/engines.py` (129 LOC, 6.8 KB)
  - `src/pipeline/flows_secondary.py` (119 LOC, 6.2 KB)
- **Contrato:** Fase 0 + revision linea a linea + contraste con tests.
- **Dictamen:** CERRADO CON FIX (1 BAJA). Cero ALTA, cero MEDIA.

## 1. breadth_metrics.py

156 LOC. Modificado hoy por S-02c (tz-aware fallback). Fase 6c.

Funciones:
- `_compute_momentum_amplitud`: momentum sectorial.
- `_load_latest_valid_breadth_snapshot`: fallback C2-followup.
- `_compute_sector_breadth_health`: B2 + C2-followup.
- `compute_breadth_metrics`: orquestador.

### Hallazgos

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| BM-1 | INFO | rama except | La rama except devuelve (None, False, ERROR) sin intentar fallback al ultimo snapshot valido. Diverge del diseno C2-followup. | WONT FIX: fail-safe a None. |
| BM-2 | INFO | L~102 | datetime.now(Europe/Madrid) fallback cuando reference_date=None. | WONT FIX: documentado en S-02c. |

## 2. engines.py

129 LOC. Modificado hoy por S-02d (guard ranking vacio). Fase 8a.

Funciones:
- `_forzar_lideres_slpm`: guard ranking vacio -> return [], 0.0.
- `_compute_tactical_structural`: loop por sector con try/except inner.
- `_compute_persistence_and_save`: persistencia + CSV historico.
- `compute_engines`: orquestador.

### Hallazgos

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| EN-1 | INFO | L~60 | except Exception externo en _compute_tactical_structural. | WONT FIX: contrato defensivo. |
| EN-2 | INFO | L~95 | except Exception externo en _compute_persistence_and_save. | WONT FIX: idem. |

Fix S-02d verificado: guard de ranking vacio funciona, test lo cubre.

## 3. flows_secondary.py

119 LOC. Modificado hoy por D-04 (tz-normalize mtime). Fase 5.

### Hallazgos

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| FS-1 | BAJA | L~86 | datetime.fromtimestamp(mtime) sin tz usa TZ local del runner. En CI (UTC), _ref - mtime inflaba ~2h. Threshold 7 dias -> no material, pero incorrecto. | FIX APLICADO (commit 637d78a). |
| FS-2 | INFO | L~118 | except Exception externo. | WONT FIX. |
| FS-3 | INFO | varias | Rutas relativas outputs/history/*.csv. | WONT FIX: CWD raiz. |

### Fix FS-1

Import: anadido timezone.
mtime = datetime.fromtimestamp(perf_path.stat().st_mtime, tz=timezone.utc).replace(tzinfo=None)

Tests: tests/test_pipeline_market_data_flows.py 14/14 OK. Suite completa 3142+5.

## 4. Contraste con tests

- breadth_metrics.py: test_pipeline_breadth_metrics.py, test_c2f_breadth_fallback.py, test_c2_temporalidad.py.
- engines.py: test_pipeline_diagnostics_engines.py.
- flows_secondary.py: test_pipeline_market_data_flows.py (14 tests).

No cubren:
- BM-1: la rama except no se testea con excepcion forzada.
- FS-1: el test D-04 usa reference_date controlado; el mtime real en CI no se verifica.

## 5. Patron observado

Los 5 ficheros con mtime 2026-10-03 comparten commits origen (e0aacda D-05, ec9e5b2 D23, D-04). Todos tocados por el mismo tipo de fix (tz-aware -> naive). Solo flows_secondary.py dejo una imprecision residual (FS-1), ahora corregida.

## 6. Dictamen

- breadth_metrics.py: CERRADO SIN CAMBIOS.
- engines.py: CERRADO SIN CAMBIOS.
- flows_secondary.py: CERRADO CON FIX (FS-1).

Bloque D-05/D-04 completo: los 5 ficheros con mtime 10-03 auditados.

## 7. Proximo

Continuar S-03-deep por LOC descendente en src/pipeline/: iae_section.py (216), sectors_base.py (155), sector_metrics.py (154). Y src/report/: flows_international.py (191), helpers.py (157), synthesis.py (149).
