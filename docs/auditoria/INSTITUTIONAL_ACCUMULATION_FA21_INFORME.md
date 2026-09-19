# INFORME FA-2.1 - Filtro temporal canonico por PERIODOFREPORT

Version: 1.0 (2026-09-19)
Dictamen habilitante: INSTITUTIONAL_ACCUMULATION_FA2_GATE0_DICTAMEN.md
Commit: 1679639
Estado: CERRADO local. Sin push (regla local-first IAE).

---

## 1. Alcance

Filtro canonico de los 7 TSVs al periodo del trimestre segun
SUBMISSION.PERIODOFREPORT (Q-A dictaminado).

NO incluye: CUSIP resolver, reporting relationships, amendments,
NIPC, economic_owner, clasificacion (FA-2.2..FA-2.5).

## 2. Artefactos

  src/institutional_accumulation/sec_13f/identity/
    __init__.py          - re-exports
    temporal_filter.py   - filter_by_period + CANONICAL_PERIOD_FIELD

  tests/test_sec_13f_temporal_filter.py  - 13 tests

## 3. Contrato

    filter_by_period(dfs, period="2026-03-31",
                     *, return_stats=False) -> dict

  - Campo canonico: SUBMISSION.PERIODOFREPORT.
  - Propagacion por ACCESSION_NUMBER a los 6 TSVs derivados.
  - No muta el dict original.
  - Idempotente.
  - Sin imputacion.

## 4. Probe real contra parquets FA-1

Entrada: 11,761 filings (dataset completo).
Salida:  10,776 filings Q1 2026, 985 fuera del periodo.

| TSV | Completo | Q1 2026 | Delta |
|---|---|---|---|
| SUBMISSION | 11,761 | 10,776 | -985 |
| COVERPAGE | 11,761 | 10,776 | -985 |
| SUMMARYPAGE | 9,846 | 8,995 | -851 |
| OTHERMANAGER | 4,751 | 4,546 | -205 |
| OTHERMANAGER2 | 4,242 | 3,704 | -538 |
| SIGNATURE | 11,761 | 10,776 | -985 |
| INFOTABLE | 3,822,885 | 3,321,967 | -500,918 |

## 5. Verificaciones

  - filings_in_period = 10,776 (Gate 0 filtrado).
  - ciks_in_period = 10,648 (Gate 0).
  - submissiontype_counts exactos:
      13F-HR    8,741
      13F-NT    1,907
      13F-HR/A    127
      13F-NT/A      1
  - FK coherence: extra_vs_submission = 0 en los 6 TSVs derivados.
  - 13 tests verdes.
  - pyflakes 0, compileall OK.
  - Suite global 740 passed + 2 skipped (727 + 13).

## 6. P4 heredado

Dictamen: P4 (519 vs 501) pasa a requisito de FA-2.2. No bloquea FA-2.1.
Se aborda como primera tarea de FA-2.2 (reconciliacion de universos).

## 7. Notas

  - El filtro aplica tanto a filings sin holdings (13F-NT) como a
    holdings. Un 13F-NT dentro del periodo se conserva aunque no
    tenga INFOTABLE asociado.
  - Los ACCESSION_NUMBER sin filing en periodo se descartan de todos
    los TSVs derivados, manteniendo coherencia referencial.

---

Fin del informe. Version 1.0 (2026-09-19).
Referencia: prompt v6.34, commit 1679639.
