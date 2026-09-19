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

## 6. P4 resuelto - Reconciliacion de universos

Probe ejecutado sobre data/etf_holdings.csv + data/market_data.parquet.

### 6.1. Anatomia etf_holdings

| Categoria | Filas | Identifiers | Tickers |
|---|---|---|---|
| Equity real valido | 506 | 506 | 503 |
| Cash line (999USDZ92) | 8 | 1 | 1 (`-`) |
| CVR (436CVR021) | 1 | 1 | 1 (2602335D) |
| Residuales ADI* | 11 | 11 | 11 (IXAU6..XASU6) |
| TOTAL | 526 | 519 | 516 |

### 6.2. Cierre aritmetico

  Tickers residuales (no equity): 13.
  Tickers equity validos: 516 - 13 = 503.

  503 coincide EXACTAMENTE con el universo IAE del dictamen Q5
  (539 EQUITY - 36 ETFs = 503).

### 6.3. Naturaleza de los 13 residuales

  - `-` (identifier 999USDZ92): cash line USD. 8 ETFs la declaran.
    No es equity. Excluir del universo CUSIP.

  - `2602335D` (identifier 436CVR021): Contingent Value Right.
    Prefijo CVR en el identifier. No es equity. Excluir.

  - 11 tickers formato IX?U6/X??U6 con identifier ADI*: identificadores
    internos SSGA (no CUSIPs, 9 chars pero prefijo ADI). Pesos
    residuales (-0.0099 a +0.032). Derivados internos del ETF.
    Excluir del universo CUSIP.

### 6.4. Los 38 EQUITY sin CUSIP local

  36 ETFs (11 sectoriales + 25 globales/macro) + BF-B + BRK-B.
  Los 2 ultimos son normalizables a BF-B / BRK-B (formato ticker).
  Los 36 ETFs son exactamente los excluidos por Q5 del dictamen.

### 6.5. Diferencia residual 506 vs 503

  519 identifiers - 13 residuales = 506.
  506 - 503 = 3 identifiers que mapean al mismo ticker (posible
  duplicado SSGA + corporate actions). Reconciliacion fina en FA-2.2.

### 6.6. Conclusion P4

  P4 CERRADO empiricamente. El universo IAE = 503 tickers equity
  individuales coincide con el dictamen Q5. Los casos residuales
  (13 tickers no-equity + 3 identifiers por reconciliar) quedan
  identificados y se resuelven en el CUSIP resolver (FA-2.2).

## 7. Notas

  - El filtro aplica tanto a filings sin holdings (13F-NT) como a
    holdings. Un 13F-NT dentro del periodo se conserva aunque no
    tenga INFOTABLE asociado.
  - Los ACCESSION_NUMBER sin filing en periodo se descartan de todos
    los TSVs derivados, manteniendo coherencia referencial.

---

Fin del informe. Version 1.0 (2026-09-19).
Referencia: prompt v6.34, commit 1679639.
