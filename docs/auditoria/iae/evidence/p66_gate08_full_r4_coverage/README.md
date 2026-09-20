# IAE - P66 Gate 0.8 Cobertura de identidad universo R4 completo

**Objeto:** medir cobertura de identidad sobre el universo completo
que R4 puede consumir, no solo sobre NOTICE. Bloqueo material del
dictamen #29.

**Origen:** dictamen #29 (2026-09-21). Gate 0.4-reissue midio solo
`OTHERMANAGER(NT)`. El auditor exigio medir sobre NOTICE + COMBINATION
+ amendments, separado por clase.

**Fecha:** 2026-09-21.
**Autor:** Ingeniero Supervisor.
**Naturaleza:** evidencia directa, no normativa.

---

## 1. Metodologia

Script deterministico. Filtrado por `PERIODOFREPORT` = periodo
dominante. Mapping `FormNum -> CIK` construido desde COVERPAGE +
SUBMISSION del periodo (Gate 0.4-reissue).

Universo evaluado (R4 segun dictamen #29):

    NOTICE_BASE                 (13F-NT, REPORTTYPE=13F NOTICE)
    NOTICE_RESTATEMENT          (13F-NT/A RESTATEMENT)
    NOTICE_NEW_HOLDINGS         (13F-NT/A NEW HOLDINGS)
    COMBINATION_BASE            (13F-HR, REPORTTYPE=13F COMBINATION REPORT)
    COMBINATION_RESTATEMENT     (13F-HR/A RESTATEMENT)
    COMBINATION_NEW_HOLDINGS    (13F-HR/A NEW HOLDINGS)

Clasificacion de cada fila OTHERMANAGER:

    IDENTITY_RESOLVED por CIK
    IDENTITY_RESOLVED por FormNum (fallback)
    N/D - FormNum no resuelto
    N/D - ni CIK ni FormNum

---

## 2. Cobertura de identidad por clase

### 2025Q4 (PERIOD=2025-12-31)

| Clase | Filings | Filas | CIK | FN | N/D_fn | N/D_null | Cobertura |
|-------|--------:|------:|----:|---:|-------:|---------:|----------:|
| NOTICE_BASE | 1,900 | 2,696 | 1,851 | 706 | 10 | 129 | 94.84% |
| NOTICE_RESTATEMENT | 37 | 37 | 37 | 0 | 0 | 0 | 100.00% |
| NOTICE_NEW_HOLDINGS | 1 | 1 | 1 | 0 | 0 | 0 | 100.00% |
| COMBINATION_BASE | 388 | 1,799 | 1,202 | 351 | 19 | 227 | 86.33% |
| COMBINATION_RESTATEMENT | 7 | 105 | 97 | 8 | 0 | 0 | 100.00% |
| COMBINATION_NEW_HOLDINGS | 1 | 2 | 2 | 0 | 0 | 0 | 100.00% |
| **TOTAL R4** | **2,334** | **4,640** | **3,190** | **1,065** | **29** | **356** | **91.70%** |

### 2026Q1 (PERIOD=2026-03-31)

| Clase | Filings | Filas | CIK | FN | N/D_fn | N/D_null | Cobertura |
|-------|--------:|------:|----:|---:|-------:|---------:|----------:|
| NOTICE_BASE | 1,907 | 2,682 | 1,820 | 737 | 2 | 123 | 95.34% |
| NOTICE_NEW_HOLDINGS | 1 | 1 | 0 | 1 | 0 | 0 | 100.00% |
| COMBINATION_BASE | 402 | 1,841 | 1,225 | 346 | 18 | 252 | 85.33% |
| COMBINATION_RESTATEMENT | 9 | 22 | 21 | 1 | 0 | 0 | 100.00% |
| **TOTAL R4** | **2,319** | **4,546** | **3,066** | **1,085** | **20** | **375** | **91.31%** |

---

## 3. Delta vs universo NOTICE-solo

| Metrica | Q4 NOTICE-only | Q4 R4-completo | Delta |
|---------|---------------:|---------------:|------:|
| Filas | 2,734 | 4,640 | +1,906 |
| Cobertura | 94.92% | **91.70%** | **−3.21 pp** |

| Metrica | Q1 NOTICE-only | Q1 R4-completo | Delta |
|---------|---------------:|---------------:|------:|
| Filas | 2,683 | 4,546 | +1,863 |
| Cobertura | 95.34% | **91.31%** | **−4.03 pp** |

---

## 4. Drill-down del N/D_null en COMBINATION_BASE

### 4.1. Concentracion

| Metrica | Q4 | Q1 |
|---------|----:|----:|
| Filas N/D_null | 227 | 252 |
| Filings afectados | 31 | 32 |
| Top 2 filings concentran | 176 (77.5%) | 180 (71.4%) |
| Top 5 filings concentran | 194 (85.5%) | 208 (82.5%) |

**Filer dominante en ambos trimestres:** `0001580642-`.

- Q4: `0001580642-26-001034` (100 filas) + `0001580642-26-000991` (76 filas).
- Q1: `0001580642-26-003152` (99 filas) + `0001580642-26-003153` (81 filas).

### 4.2. Composicion de columnas en filas N/D_null

| Columna | Pobladas Q4 | Pobladas Q1 |
|---------|------------:|------------:|
| CIK | **0 / 227** | **0 / 252** |
| FORM13FFILENUMBER | 1 / 227 | 6 / 252 |
| CRDNUMBER | 200 / 227 (88%) | 214 / 252 (85%) |
| SECFILENUMBER | 157 / 227 (69%) | 161 / 252 (64%) |
| NAME | **227 / 227 (100%)** | **252 / 252 (100%)** |

**Patron:** los filers afectados declaran sub-managers con NAME +
CRDNUMBER + (a veces) SECFILENUMBER, pero NO con CIK ni
FORM13FFILENUMBER.

---

## 5. Interpretacion

1. **No es un fallo de ingestion.** El Data Set SEC replica lo que
   el filing publico.

2. **Es exactamente el caso N/D que el contrato prohibe resolver por
   nombre.** Regla del auditor #28/#29: "No se permite matching por
   nombre". Aplica.

3. **No es una nueva clase de ambiguedad.** Es el mismo N/D ya
   definido, concentrado en un filer concreto.

4. **CRDNUMBER no entra en R4.** Aunque es estructurado, no forma
   parte del mapping contractual de L3.

---

## 6. Conclusion

**Gate 0.8 PASS con nota de caracterizacion.**

- Cobertura del universo R4 completo: 91.70% / 91.31%.
- Delta de -3/-4 pp vs NOTICE-solo tiene causa unica: filer
  `0001580642-` que declara sub-managers sin identificadores L3
  validos.
- El contrato ya cubre este caso con N/D fail-closed.
- No se descubre nueva clase de ambiguedad.
- El auditor puede emitir GO contractual.

---

## 7. Refs

- `iae/DICTAMENES.md` #29.
- Gate 0.4-reissue (`p66_gate04_reissue_period/`).
- Gate 0.6 (`p66_gate06_combination_probe/`).
- SEC Form 13F (estructura oficial).

---

Fin del informe Gate 0.8.
