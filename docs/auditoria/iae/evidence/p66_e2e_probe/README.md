# IAE - P66 e2e probe

**Objeto:** validar las funciones puras implementadas en `reporting_dedup.py`
para el ciclo P66 (§14.3.1 R3 tri-state + §14.3.4 R4) sobre datos reales
SEC 13F Q4 2025 y Q1 2026.

**Origen:** GO CONTRACTUAL #40 (P66 cerrado). Cierre del paso 3 del ciclo
(implementacion) mediante probe e2e sobre los parquets en
`data/sec_13f/processed/{2025Q4,2026Q1}/`.

**Fecha:** 2026-09-21.
**Autor:** Ingeniero Supervisor.
**Naturaleza:** evidencia directa, no normativa.

---

## 1. Metodologia

Probe deterministico, sin `datetime.now()`, sin IO de red. Carga
`SUBMISSION`, `COVERPAGE`, `OTHERMANAGER` de cada trimestre. Aplica:

- `resolve_r3(filings, period, scope_completeness_verified)`.
- `resolve_r4(a_cik, othermanager_rows, formnum_mapping)`.
- `build_formnum_cik_mapping(cov_full, period)` con join a SUBMISSION
  para obtener `CIK` y `PERIODOFREPORT`.
- `classify_filing_family` / `classify_filing_role`.

Periodo dominante por trimestre, filtro `PERIODOFREPORT` == dominante.

---

## 2. Resultados por trimestre

### 2025Q4 (PERIOD=2025-12-31)

| Metrica | Valor |
|---|---:|
| Filings totales | 10,676 |
| Filings R4 no clasificables | 0 |
| Grupos (CIK, PERIOD) | 10,524 |
| R3 = TRUE | 2,286 |
| R3 = FALSE | 8,235 |
| **R3 = N/D** | **3** |
| Mapping FormNum->CIK entradas | 10,524 |

### 2026Q1 (PERIOD=2026-03-31)

| Metrica | Valor |
|---|---:|
| Filings totales | 10,776 |
| Filings R4 no clasificables | 0 |
| Grupos (CIK, PERIOD) | 10,648 |
| R3 = TRUE | 2,309 |
| R3 = FALSE | 8,337 |
| **R3 = N/D** | **2** |
| Mapping FormNum->CIK entradas | 10,648 |

**Coincidencia con Gates:** R3=TRUE (2,286 / 2,309) coincide exactamente
con Gate 0.9 ("con 1 filing base": 2,286 / 2,309).

---

## 3. Casos N/D - drill-down

### 2025Q4 (3 casos)

| CIK | ACC base | ST base | RT base | ACC amd | ST amd | RT amd | Tipo |
|---|---|---|---|---|---|---|---|
| 0002056095 | ...000001 | 13F-HR | 13F HOLDINGS REPORT | ...000002 | 13F-HR/A | 13F COMBINATION REPORT | HOLDINGS+COMB |
| 0002016827 | ...000005 | 13F-HR | 13F COMBINATION REPORT | (sin amd) | ...000001 | 13F-NT | 13F NOTICE | MIXED NT+HR |
| 0001315059 | ...000001 | 13F-HR | 13F HOLDINGS REPORT | ...000002 | 13F-HR/A | 13F COMBINATION REPORT | HOLDINGS+COMB |

### 2026Q1 (2 casos)

| CIK | ACC base | ST base | RT base | ACC amd | ST amd | RT amd | Tipo |
|---|---|---|---|---|---|---|---|
| 0002089727 | ...000280 | 13F-HR | 13F HOLDINGS REPORT | ...000312 | 13F-HR/A | 13F COMBINATION REPORT | HOLDINGS+COMB |
| 0001572838 | ...000005 | 13F-HR | 13F HOLDINGS REPORT | ...000006 | 13F-HR/A | 13F COMBINATION REPORT | HOLDINGS+COMB |

---

## 4. Hallazgos

### H1. Implementacion literal de §14.3.1 PASO 1-4

La implementacion filtra primero a R4 (PASO 1) y luego clasifica por
rol (PASO 2). Un filing base con `REPORTTYPE = 13F HOLDINGS REPORT`
no pasa PASO 1, por lo que no cuenta como BASE para el grupo. El
amendment con `REPORTTYPE = 13F COMBINATION REPORT` si pasa y se
clasifica AMENDMENT. Resultado: `BASE=0 + AMENDMENT>=1 -> R3=N/D`.

### H2. Casos "base HR-holdings + amendment HR-COMBINATION" no documentados en Gate 0.9

Gate 0.9 excluyo explicitamente amendments (`ISAMENDMENT != Y`).
Contando solo bases puras, Q4 tiene 1 caso (CIK 0002016827, MIXED
documentado). La implementacion contractual (con amendments,
segun §14.3.2) detecta 2 casos adicionales en Q4 y 2 en Q1 con el
patron "HOLDINGS+COMB". Total: 5 casos sobre 2 trimestres.

### H3. Coincidencia de R3=TRUE con Gate 0.9

Los valores R3=TRUE (2,286 / 2,309) coinciden exactamente con Gate 0.9.
Los N/D adicionales son estrictamente amendments que Gate 0.9 excluyo.

### H4. Estados R4 observados en muestra

`resolve_r4` produce MATCH, NO_MATCH y N/D sobre datos reales.
CONFLICT no se observa en la muestra (necesita fixture especifico,
ya cubierto en tests unitarios).

---

## 5. Pregunta material al auditor

**§14.3.1 PASO 1 filtra a R4 antes de PASO 2.** Esto hace que un filing
base no-R4 (HR-holdings) no cuente como BASE del grupo. El resultado
fail-closed es `R3=N/D` aunque exista cadena documental completa.

Dos lecturas posibles:

**(A) Literal actual (implementada).** PASO 1 filtra antes de PASO 2.
Un HR-holdings base no-R4 no es BASE del conjunto R4 -> BASE=0 ->
N/D. Fail-closed.

**(B) Extendida.** La cadena documental del periodo se construye sobre
TODOS los filings (base + amendments), y solo el filing efectivo debe
ser R4. En "base HR-holdings + amendment HR/COMBINATION", el filing
efectivo (el amendment, que sustituye) es R4 -> R3=TRUE.

**Impacto material:** con (A), 2 CIKs Q4 + 2 CIKs Q1 quedan N/D por
esta razon. Con (B), pasarian a TRUE. La diferencia es 4 grupos sobre
21.172 (0.02%).

**Recomendacion tecnica:** la implementacion actual sigue (A) por
lectura literal. Si el auditor confirma (A), el probe cierra tal cual.
Si el auditor prefiere (B), requiere ajuste de `resolve_r3` en
PASO 1-4 + dictamen material.

---

## 6. Ficheros

    probe_p66_e2e.py    probe deterministico
    probe_output.txt    salida cruda (UTF-8 LF)
    HASHES.txt          SHA-256 del probe y del output
    README.md           este documento

---

## 7. Refs

- `NIPC_CONTRATOS_SEMANTICOS_v1.md` §14.3.
- `iae/DICTAMENES.md` #40 (GO CONTRACTUAL).
- `iae/evidence/p66_gate06_combination_probe/` (universo R4).
- `iae/evidence/p66_gate08_full_r4_coverage/` (cobertura identidad).
- `iae/evidence/p66_gate09_multiple_base/` (multiples bases).

---

Fin del README. Probe ejecutado 2026-09-21.

---

## 8. Post-dictamen #41 (2026-09-21)

**Resolucion del auditor:** LECTURA A adoptada.

La divergencia material elevada en la seccion 5 queda resuelta a favor de
la lectura literal (A). La implementacion actual de `resolve_r3` se conserva
sin cambios. Los 4 grupos afectados permanecen `R3 = N/D`.

Ver `iae/DICTAMENES.md` #41.

### Reserva de auditoria (suite global) - resuelta

El dictamen exigio evidencia directa de baseline pre-P66 para los 3 fallos
de `test_freshness.py`. Resuelto con evidencia en
`iae/evidence/p66_baseline_pre/`: worktree detached en `64b0637` con los
parquets actuales reproduce los 3 fallos identicos.

---

Fin del apendice post-dictamen.
