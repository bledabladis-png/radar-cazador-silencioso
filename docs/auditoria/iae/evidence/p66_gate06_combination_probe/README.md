# IAE - P66 Gate 0.6 Combination probe

**Objeto:** medir el universo de filings aplicable a R4, incluyendo
`13F COMBINATION REPORT` (bloqueo #1 del dictamen #28).

**Origen:** dictamen #28, bloqueo #1 (Combination excluido).

**Fecha:** 2026-09-21.
**Autor:** Ingeniero Supervisor.
**Naturaleza:** evidencia directa, no normativa.

---

## 1. Metodologia

Script deterministico, sin `datetime.now()`. Filtrado por
`PERIODOFREPORT = periodo dominante` del directorio (2025-12-31 en
2025Q4; 2026-03-31 en 2026Q1). Aplicado tras Gate 0.5A que demostro
que los directorios son cross-periodo.

Cruce con `COVERPAGE` (REPORTTYPE) y `OTHERMANAGER` (filas).

---

## 2. Distribucion por tipo de filing

### 2025Q4 (PERIOD=2025-12-31)

| SUBMISSIONTYPE | REPORTTYPE | Filings | Con OM | % |
|----------------|-----------|--------:|-------:|---:|
| 13F-HR | 13F HOLDINGS REPORT | 8,237 | 0 | 0.0% |
| 13F-NT | 13F NOTICE | 1,900 | 1,900 | 100.0% |
| 13F-HR | 13F COMBINATION REPORT | 388 | 388 | 100.0% |
| 13F-HR/A | 13F HOLDINGS REPORT | 105 | 0 | 0.0% |
| 13F-NT/A | 13F NOTICE | 38 | 38 | 100.0% |
| 13F-HR/A | 13F COMBINATION REPORT | 8 | 8 | 100.0% |

### 2026Q1 (PERIOD=2026-03-31)

| SUBMISSIONTYPE | REPORTTYPE | Filings | Con OM | % |
|----------------|-----------|--------:|-------:|---:|
| 13F-HR | 13F HOLDINGS REPORT | 8,339 | 0 | 0.0% |
| 13F-NT | 13F NOTICE | 1,907 | 1,907 | 100.0% |
| 13F-HR | 13F COMBINATION REPORT | 402 | 402 | 100.0% |
| 13F-HR/A | 13F HOLDINGS REPORT | 118 | 0 | 0.0% |
| 13F-NT/A | 13F NOTICE | 1 | 1 | 100.0% |
| 13F-HR/A | 13F COMBINATION REPORT | 9 | 9 | 100.0% |

---

## 3. Hallazgos

1. **REPORTTYPE distingue limpiamente.**
   - 13F HOLDINGS REPORT: 0% OM.
   - 13F COMBINATION REPORT: 100% OM.
   - 13F NOTICE: 100% OM.

2. **SUBMISSIONTYPE por si solo NO basta.** `13F-HR` incluye
   Holdings Report (sin OM) y Combination Report (con OM). El
   dictamen #28 tenia razon.

3. **Universo R4 corregido** (segun el texto del dictamen #28):

   | Periodo | NOTICE | COMBINATION | Total R4 | v2-bis | Delta |
   |---------|-------:|------------:|---------:|-------:|------:|
   | 2025-12-31 | 1,938 | 396 | 2,334 | 1,938 | +396 (+17.0%) |
   | 2026-03-31 | 1,908 | 411 | 2,319 | 1,908 | +411 (+21.5%) |

4. **Cobertura OM en universo R4: 100.00%.** Sin excepciones.

5. **Sin casos anomalos.** Ninguna HR-normal con OM, ninguna
   Combination sin OM. La spec SEC se cumple estrictamente.

---

## 4. Implicacion para el contrato 14.3

El texto de R4 debe ampliarse:

    La evidencia es valida cuando el filing B es:
      - 13F-NOTICE (SUBMISSIONTYPE = 13F-NT / 13F-NT/A
                    y REPORTTYPE = 13F NOTICE), o
      - 13F-COMBINATION (SUBMISSIONTYPE = 13F-HR / 13F-HR/A
                         y REPORTTYPE = 13F COMBINATION REPORT).

La distincion depende de `REPORTTYPE`, no solo de `SUBMISSIONTYPE`.

---

## 5. Refs

- `iae/DICTAMENES.md` #28, bloqueo #1.
- SEC Form 13F (documentacion oficial).
- Gate 0.5A (contaminacion cross-periodo).

---

Fin del informe Gate 0.6.
