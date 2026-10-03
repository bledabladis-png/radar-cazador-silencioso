# S-03-deep - bloque src/report/ (slpm, leaders, rankings)

- **Auditoria:** 2026-10-03
- **HEAD inicial:** c54d201
- **Ficheros:**
  - `src/report/slpm.py` (137 LOC, 8.7 KB, mtime 2026-10-02)
  - `src/report/leaders.py` (128 LOC, 7.8 KB, mtime 2026-10-02)
  - `src/report/rankings.py` (116 LOC, 6.3 KB, mtime 2026-10-02)
- **Contrato:** Fase 0 + revision linea a linea + contraste con tests.
- **Dictamen:** CERRADO CON FIX (LD-1, RK-1). Cero ALTA, cero MEDIA.

## 1. slpm.py

137 LOC. 2 funciones de render:
- `render_slpm_v12(slpm_v12_data)`: SLPM v1.2 completo (structural
  leadership, breadth, integrity, flow divergence).
- `render_slpm_legacy(slpm_data)`: bloque Legacy v1.0 en <details>.

Consumidor: `src/report_generator.py`.

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| SL-1 | INFO | render_slpm_v12 | `expected_leaders=5` en `.get()`, coincide con `TOP_N_LEADERS` pero semantica distinta (leaders esperados por sector vs mostrados en reporte). | WONT FIX. |
| SL-2 | INFO | render_slpm_v12 | `breadth`/`coverage` reasignadas (L18/L74). Cosmetico. | WONT FIX. |
| SL-3 | INFO | docstring | Tildes en UTF-8 correctas, el `Ãndices` observado era artefacto consola. | WONT FIX. |

FU-003b (LIS con n=0 -> N/D) y FU-013 (breadth con n=0 -> N/D)
aplicados y testados.

## 2. leaders.py

128 LOC. 5 funciones de render:
- `render_momentum_sectores`: momentum de precio + flow.
- `render_tactical_leaders`: tabla Tactical Leaders.
- `render_momentum_otros`: momentum de otros activos.
- `render_structural_ranking`: tabla Structural Ranking.
- `render_acciones_seleccionadas`: Acciones Seleccionadas (SLPM).

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| LD-1 | BAJA | 4 sitios | `[:11]` hardcoded en render_momentum_sectores (2), render_tactical_leaders (1), render_structural_ranking (1). Duplica `EXPECTED_SECTOR_COUNT`. | **FIX APLICADO** (commit 4e5bf2a). |
| LD-2 | INFO | render_momentum_otros | `[:15]` coincide con `TOP_N_CANDIDATES` pero concepto distinto (otros activos vs candidatos de lideres). | WONT FIX. |

I1 (2026-09-18): notas aclaratorias del "Retorno 20d" presentes en
2 tablas. B2 (2026-09-18): nota de criterio de seleccion WLS.

## 3. rankings.py

116 LOC. 3 funciones de render:
- `render_rankings_sectoriales`: tabla de score combinado.
- `render_persistencia`: persistencia por sector.
- `render_opportunity_map`: Opportunity Map por cuadrantes.

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| RK-1 | BAJA | render_rankings_sectoriales | `[:11]` hardcoded, duplica `EXPECTED_SECTOR_COUNT`. | **FIX APLICADO** (commit 4e5bf2a). |

## 4. Contraste con tests

- `tests/test_i1_o1_notas.py`: I1 (notas aclaratorias) + O1 (fecha
  efectiva en fuente SSGA).
- `tests/test_report_leaders_criterio_b2.py`: B2 (nota de criterio
  WLS en Acciones Seleccionadas).
- `tests/test_report_leaders_nomenclatura.py`: nomenclatura.
- `tests/test_report_generator_helpers.py` (bloque leaders): helpers
  compartidos.
- `tests/test_fu003b_lis_nd.py`: FU-003b (LIS con n=0 -> N/D).
- `tests/test_render_modules_low_coverage_d29.py` (parcial).

No cubren:
- LD-1/RK-1: el slice `[:11]` no se testea con exactamente 11
  sectores. Los tests usan 1-2 tickers.

## 5. Patron: 11 hardcoded en src/report/

Cuatro ficheros de `src/report/` han mostrado el mismo patron
"11 hardcoded" (mismo A5-11 de `engines.py`, SS-2 de
`sectors_base.py`, HP-2 de `helpers.py`, LD-1/RK-1 aqui). Total:
6 ficheros, 12+ sitios corregidos en S-03-deep.

Candidato a barrido final sobre los 14 ficheros restantes de
`src/report/` para localizar el resto de ocurrencias.

## 6. Dictamen

CERRADO CON FIX. Cero ALTA, cero MEDIA. 2 BAJA corregidas
(LD-1, RK-1) en commit 4e5bf2a. 4 INFO WONT FIX razonado.

Los 3 ficheros del bloque quedan auditados.

## 7. Proximo

Continuar `src/report/` (5/20 auditados): `iae.py` (106),
`market_context.py` (105), `freshness.py` (101),
`sector_context.py` (97), `confirmation.py` (95),
`volatility_mte.py` (89), `sectorial.py` (85), `header.py` (71),
`alerts.py` (69), `darkpool.py` (62), `etf_flows.py` (61),
`sentiment.py` (51), `breadth.py` (30), `__init__.py` (2).
