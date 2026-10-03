# S-03-deep - barrido del patron "11 hardcoded"

Barrido transversal sobre todo el repo para localizar el patron
"11 hardcoded" (numero de sectores) y sustituirlo por
`EXPECTED_SECTOR_COUNT` de `config/settings.py:63`.

- **Auditoria:** 2026-10-03
- **HEAD inicial:** b3af02c
- **Alcance:** todo el repo (excluye tests y __pycache__).
- **Metodo:** grep con 3 patrones (`[:11]`, `* 11`/`/ 11`/`== 11`,
  `EXPECTED_SECTOR_COUNT`).
- **Dictamen:** CERRADO. 2 ficheros nuevos corregidos, 1 commit.

## 1. Resultado del barrido

Total ficheros corregidos por este patron en S-03-deep: **9**.

| # | Fichero | Sitios | Commit | Contexto |
|---|---|---:|---|---|
| 1 | `src/pipeline/engines.py` | 1 | A5-11 (2026-09-28) | constante duplicada |
| 2 | `src/pipeline/sectors_base.py` | 5 | SS-2 (2026-10-03) | breadth_values |
| 3 | `src/report/helpers.py` | 1 | HP-2 (2026-10-03) | sectores_total |
| 4 | `src/report/leaders.py` | 4 | LD-1 (2026-10-03) | slices [:11] |
| 5 | `src/report/rankings.py` | 1 | RK-1 (2026-10-03) | slice [:11] |
| 6 | `src/report/breadth.py` | 11 | BR-1 (2026-10-03) | titulo + 5 defaults + 5 f-strings |
| 7 | `indicators/data_quality.py` | 3 | edb251f (2026-10-03) | n_total sectorial |
| 8 | `indicators/sector_dispersion.py` | 1 | edb251f (2026-10-03) | n_total |
| | **Total** | **27** | | |

## 2. Los 2 ficheros nuevos (cierre del barrido)

### data_quality.py

3 sitios `n_total = 11` en `compute_data_quality`, ramas:
- `sectorial`: cobertura sobre `nunique()['sector']`.
- `sectorial_family`: cobertura sobre mean_corr notna.
- `sectorial_evidence`: cobertura sobre alignment_reading !=
  'EVIDENCIA INSUFICIENTE'.

Los otros `n_total` del mismo fichero:
- `flow_tickers`: `n_total = 12` (universo flujo, NO sectorial).
  WONT FIX: el 12 es correcto, distinto del 11.
- `rows`: `n_total = len(df)` (dinamico). Correcto.

### sector_dispersion.py

1 sitio `n_total = 11` en `compute_sector_dispersion`. Cobertura
sobre `price_rank_list` (retornos sectoriales).

## 3. Sitios NO corregidos (semanticamente distintos)

- `indicators/data_quality.py` `n_total = 12` (flow_tickers).
- `indicators/evidence_matrix.py:36` (comentario, no codigo).
- Tests (`n=11`, `window=11`, `seed=11`).
- Cualquier `11` sin semantica de sectores.

## 4. Verificacion

- `tests/test_data_quality_finra_no_walkback.py` OK.
- `tests/test_sector_regime_dispersion.py` OK.
- Suite completa: 3142 passed + 5 skipped.

## 5. Regla de estilo derivada

**Toda iteracion o conteo sobre los 11 sectores debe usar
`EXPECTED_SECTOR_COUNT` importado de `config.settings`.**

Candidato a documentar en `01_METODO.md` seccion 6 o similar.
No se aplica en este commit (decision de corpus).

## 6. Dictamen

CERRADO. 2 ficheros nuevos, 4 sitios, commit `edb251f`.

## 7. Proximo

Candidatos siguientes en S-03-deep:
- Barrido de docstrings desalineados en `src/` (3 casos detectados:
  LD-1 pipeline, II-1, DL-1).
- Caracterizacion de `macro_regime.py` (12 ramas sin test directo).
- `src/temporal_contracts/` (11 ficheros, A1 CERRADO).
- `src/institutional_accumulation/` (36 ficheros, B/6 CERRADOS).
