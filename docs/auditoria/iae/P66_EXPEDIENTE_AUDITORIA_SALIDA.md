# IAE - P66 Expediente para auditoria de salida

**Objeto:** expediente formal de entrega al auditor externo para auditoria
de salida del ciclo P66, que materializo la §14.3 reformulada del contrato
NIPC_CONTRATOS_SEMANTICOS_v1.

**Origen:** GO CONTRACTUAL #40 (2026-09-21). Cierre de los pasos 2, 3 y 4
del ciclo autorizado por GO #40.

**Fecha:** 2026-09-21.

**HEAD al redactar:** a00ee48.

**Commit del GO:** 6e698ca.

**Naturaleza:** documento de entrega. NO normativo. En caso de conflicto
con `NIPC_CONTRATOS_SEMANTICOS_v1.md` §14.3 o con los dictamenes #28-#40,
gana el contrato o el dictamen.

---

## 0. Resumen ejecutivo

(a) Los pasos 2, 3 y 4 del ciclo autorizado por GO #40 estan ejecutados.
(b) Existe una divergencia material en la interpretacion de §14.3.1
    PASO 1-4 que se eleva al auditor para dictamen (seccion 5).
(c) Las prohibiciones vigentes permanecen intactas (seccion 7).

Se solicita al auditor:

1. Dictamen sobre la divergencia material (seccion 5).
2. Confirmacion de cierre del paso 4 (probe e2e) y apertura del paso 5
   (auditoria formal de salida).

---

## 1. Alcance autorizado

GO #40 autorizo expresamente solo el paso 1 (traslado de §14.3 a
`NIPC_CONTRATOS_SEMANTICOS_v1.md`). Los pasos 2, 3 y 4 fueron autorizados
explicitamente por el usuario en sesion 2026-09-21:

- Paso 1: traslado §14.3. Autorizado por GO #40. Commit `3a233b4`.
- Paso 2: tests contractuales §14.3. Autorizado por usuario.
- Paso 3: implementacion en `reporting_dedup.py`. Autorizado por usuario.
- Paso 4: probe e2e sobre datos reales. Autorizado por usuario.
- Paso 5: auditoria de salida. Se abre con este expediente.
- Paso 6: activacion `DROP_DUP`. Requiere nuevo dictamen. NO AUTORIZADO.
---

## 2. Lo ejecutado en este ciclo

Cinco commits en `origin/main..HEAD` (todos locales, sin push):

| Commit | Fichero(s) | Contenido |
|---|---|---|
| `bff0945` | `tests/test_p66_contract.py` (nuevo) | Tests contractuales §14.3. Capa A (10), Capa B (3), Capa C (18). |
| `fe7c1dd` | `reporting_dedup.py` + tests | §14.3.6 canonicalizacion Form13FFileNumber. |
| `3be5465` | `reporting_dedup.py` + tests | §14.3.5 mapping FormNum -> CIK + §14.3.1 PASO 1-2 + §14.3.2 cadena amendments. |
| `ea50880` | `reporting_dedup.py` + tests | §14.3.1 R3 tri-state + §14.3.4 R4 (candidate_A, CONFLICT, N/D) + L3 booleano. |
| `a00ee48` | `docs/auditoria/iae/evidence/p66_e2e_probe/` (nuevo) | Probe e2e + README + output + HASHES. |

**Estado por paso del GO #40:**

| Paso | Descripcion | Estado |
|---|---|---|
| 1 | Traslado §14.3 | CERRADO (`3a233b4`) |
| 2 | Tests contractuales §14.3 | CERRADO (`bff0945`) |
| 3 | Implementacion `reporting_dedup.py` | CERRADO (`fe7c1dd` + `3be5465` + `ea50880`) |
| 4 | Probe e2e | CERRADO (`a00ee48`) — eleva divergencia material |
| 5 | Auditoria de salida | ABIERTO (este expediente) |
| 6 | Activacion `DROP_DUP` | NO AUTORIZADO |

---

## 3. Superficie de codigo nuevo

Todo el codigo nuevo vive en `src/institutional_accumulation/aggregation/reporting_dedup.py`.
Son funciones puras: sin IO, sin `datetime.now()`, sin side effects.

### 3.1. Constantes nuevas

- `R3_TRUE`, `R3_FALSE`, `R3_ND`.
- `R4_MATCH`, `R4_NO_MATCH`, `R4_ND`, `R4_CONFLICT`.
- `FORMNUM_STATUS_IDENTITY_RESOLVED`, `FORMNUM_STATUS_UNRESOLVED`, `FORMNUM_STATUS_CONFLICT`.
- `AMENDMENT_TYPE_RESTATEMENT`, `AMENDMENT_TYPE_NEW_HOLDINGS`, `ALL_AMENDMENT_TYPES`.
- `FILING_FAMILY_NOTICE`, `FILING_FAMILY_COMBINATION`, `ALL_FILING_FAMILIES`.
- `FILING_ROLE_BASE`, `FILING_ROLE_AMENDMENT`, `ALL_FILING_ROLES`.
- `FORMNUM_PREFIX_CANONICAL`, `FORMNUM_VALID_PREFIXES`, `FORMNUM_SUFFIX_WIDTH`.

### 3.2. Funciones nuevas

| Funcion | §14.3 | Rol |
|---|---|---|
| `canonicalize_form13f_filenumber` | §14.3.6 | Canonicalizacion FormNum (prefijo {28,028} -> 028, padding sufijo <=5). |
| `classify_filing_family` | §14.3.1 PASO 1 | NOTICE / COMBINATION via (SUBMISSIONTYPE, REPORTTYPE). |
| `classify_filing_role` | §14.3.1 PASO 2 | BASE / AMENDMENT via SUBMISSIONTYPE. |
| `build_formnum_cik_mapping` | §14.3.5 | Mapping FormNum normalizado -> CIK + accessions. |
| `resolve_formnum_to_cik` | §14.3.5 | Resolucion (IDENTITY_RESOLVED | UNRESOLVED | CONFLICT). |
| `validate_amendment_chain` | §14.3.2 | Valida numeros unicos + secuencia sin huecos + tipos reconocibles. |
| `resolve_r3` | §14.3.1 | R3 tri-state con PASO 0-4. |
| `resolve_r4` | §14.3.4 | R4 estados con candidate_A. |
| `_row_identity_state` | §14.3.4 | Helper interno: candidate_A + IDENTITY_RESOLVED + CONFLICT. |
| `evaluate_l3` | §14.3 | Booleano L3 = R1 ∧ R2 ∧ (R3=TRUE) ∧ (R4=MATCH) ∧ R5. |
---

## 4. Cobertura de tests

### 4.1. Fichero nuevo `tests/test_p66_contract.py`

31 tests, 3 capas:

| Capa | Tests | Rol |
|---|---:|---|
| A - integridad documental | 10 | Verifica literales contractuales en `NIPC_CONTRATOS_SEMANTICOS_v1.md` §14.3. |
| B - no regresion P65 | 3 | Constantes + FORBIDDEN_TERMS de `reporting_dedup.py` y `relationships.py`. |
| C - semantica ejecutable | 18 | Casos §14.3.1-14.3.6 sobre funciones puras. |

**Resultado:** 31 passed + 0 failed + 0 xfailed (los 12 xfail originales del commit `bff0945` fueron retirados conforme se materializaba cada subseccion).

### 4.2. No regresion P65

`tests/test_sec_13f_reporting_dedup.py`: **31/31 passed**.

### 4.3. Suite global

| Metrica | Valor |
|---|---:|
| Passed | 1067 |
| Skipped | 2 |
| Failed | 3 (preexistentes, no relacionados) |
| xfailed | 0 |

Los 3 fallos son `tests/test_freshness.py` (`test_market_data_fresh`, `test_stock_prices_fresh`, `test_european_tickers_recent`). Causa: `data/market_data.parquet` y `data/stock_prices.parquet` con fecha 2026-09-16 en disco (5 dias vs hoy 2026-09-21, MAX_DAILY_AGE=4). Cero relacion con este ciclo. Se autorregeneran con el proximo `daily_run.yml` (cron `0 4 * * *`).

`pyflakes` global: LIMPIO.
`compileall`: OK.
---

## 5. Probe e2e - resultados y divergencia material

Evidencia: `docs/auditoria/iae/evidence/p66_e2e_probe/`
(probe + output + README + HASHES; commit `a00ee48`).

### 5.1. Cifras verificadas

| Metrica | 2025Q4 | 2026Q1 |
|---|---:|---:|
| Filings totales | 10,676 | 10,776 |
| Filings R4 no clasificables | 0 | 0 |
| Grupos (CIK, PERIOD) | 10,524 | 10,648 |
| **R3 = TRUE** | **2,286** | **2,309** |
| R3 = FALSE | 8,235 | 8,337 |
| **R3 = N/D** | **3** | **2** |
| Mapping FormNum -> CIK entradas | 10,524 | 10,648 |

**Coincidencia con Gate 0.9:** R3=TRUE (2.286 / 2.309) coincide exactamente
con el conteo "con 1 filing base" del Gate 0.9.

### 5.2. Casos N/D - detalle

**2025Q4 (3 casos):**

| CIK | Base | Amendment | Patron |
|---|---|---|---|
| 0002056095 | 13F-HR + 13F HOLDINGS REPORT | 13F-HR/A + 13F COMBINATION REPORT (RESTATEMENT) | HOLDINGS+COMB |
| 0002016827 | 13F-HR + 13F COMBINATION REPORT | 13F-NT + 13F NOTICE | MIXED NT+HR (documentado Gate 0.9) |
| 0001315059 | 13F-HR + 13F HOLDINGS REPORT | 13F-HR/A + 13F COMBINATION REPORT (RESTATEMENT) | HOLDINGS+COMB |

**2026Q1 (2 casos):**

| CIK | Base | Amendment | Patron |
|---|---|---|---|
| 0002089727 | 13F-HR + 13F HOLDINGS REPORT | 13F-HR/A + 13F COMBINATION REPORT (RESTATEMENT) | HOLDINGS+COMB |
| 0001572838 | 13F-HR + 13F HOLDINGS REPORT | 13F-HR/A + 13F COMBINATION REPORT (RESTATEMENT) | HOLDINGS+COMB |
### 5.3. Divergencia material

**Origen:** §14.3.1 PASO 1 filtra los filings al universo R4 (NOTICE o
COMBINATION REPORT) **antes** de PASO 2 (clasificacion BASE / AMENDMENT).

**Consecuencia:** un filing base con `REPORTTYPE = 13F HOLDINGS REPORT`
(no-R4) NO cuenta como BASE del grupo. El amendment que lo sustituye
(`REPORTTYPE = 13F COMBINATION REPORT`, R4) SI pasa como AMENDMENT.
Resultado: `BASE=0 + AMENDMENT>=1 -> R3 = N/D` (fail-closed).

**Casos reales:** 2 en Q4 + 2 en Q1 = 4 grupos sobre 21.172 (0.02%).
Patron unico: HR-holdings base + HR/A-COMBINATION amendment RESTATEMENT.

**Impacto de cada lectura:**

| Lectura | Regla | R3 resultante | Grupos afectados |
|---|---|---|---|
| **(A)** Literal actual (implementada) | PASO 1 filtra a R4 antes de PASO 2; base no-R4 no cuenta. | N/D | 4 (0.02%) |
| **(B)** Extendida | Cadena documental construida sobre TODOS los filings; solo el filing efectivo debe ser R4. | TRUE | 4 pasarian a TRUE |

**Justificacion de la lectura (A):** §14.3.1 PASO 1 textualmente dice
"Identificar todos los filings R4 del CIK + PERIODOFREPORT". El filtro
es previo al conteo de bases. La lectura literal produce N/D fail-closed.

**Argumento para la lectura (B):** el filing efectivo, tras aplicar
§14.3.3 (RESTATEMENT sustituye), es el amendment HR-COMBINATION, que
pertenece al universo R4. Bajo esa lectura, el grupo tiene 1 filing
efectivo R4 y R3 deberia ser TRUE.

**Posicion del redactor:** la implementacion sigue (A) por lectura
literal. **La decision corresponde al auditor.**

**Nota:** Gate 0.9 excluia explicitamente amendments (`ISAMENDMENT != Y`)
por diseno. Por eso no contabilizo los 4 casos. La implementacion, que
los incluye segun §14.3.2, los detecta.

---

## 6. Preguntas al auditor

1. **§14.3.1 PASO 1-4:** confirmar lectura (A) literal o autorizar
   lectura (B) extendida.
   - Si (A): cerrar paso 4 sin cambios de codigo.
   - Si (B): autorizar ajuste de `resolve_r3` + nuevo commit + nueva
     iteracion de tests.

2. **Cierre del paso 4:** confirmar que el probe e2e (evidencia
   `p66_e2e_probe/`, commit `a00ee48`) es suficiente para cerrar el
   paso 4 del GO #40.

3. **Apertura del paso 5:** confirmar si el presente expediente es
   suficiente para emitir dictamen de cierre del ciclo P66 (paso 5),
   o si requiere evidencia adicional.

4. **Confirmacion de prohibiciones:** verificar que las prohibiciones
   vigentes (seccion 7) siguen aplicando.
---

## 7. Lo que NO se ha tocado

- `MATCH_KEY`: intacto.
- C2: intacto.
- `src/institutional_accumulation/aggregation/delta_shares.py`: intacto.
- `src/institutional_accumulation/aggregation/nipc.py`: intacto.
- `src/institutional_accumulation/aggregation/coverage.py`: intacto.
- `src/institutional_accumulation/sec_13f/identity/relationships.py`: intacto.
- `NIPC_COVERAGE_POLICY.md` v1.0: INTACTA (hash 57f2d01f...).
- `NIPC_CONTRATOS_SEMANTICOS_v1.md`: §14.3 NO modificada en este ciclo
  (solo el traslado `3a233b4` previo a este ciclo).
- `DROP_DUP`: NO activado.
- `pyproject.toml` / `requirements.txt`: sin cambios.
- Push a `origin/main`: NO.

---

## 8. Bloqueos vigentes

| Bloqueo | Estado |
|---|---|
| THRESHOLD_1 | UNDEFINED |
| THRESHOLD_2 | BLOQUEADO |
| Gate-NIPC.2 | BLOQUEADO |
| Gate-NIPC.3 | NO AUTORIZADO |
| OpenFIGI masivo | NO AUTORIZADO |
| Policy v1.3 aplicacion | NO AUTORIZADA |
| Push a `origin/main` | NO |
| Activacion `DROP_DUP` | NO AUTORIZADA |
| Certificacion "acumulacion" | BLOQUEADA |

---

## 9. Trazabilidad

### 9.1. Commits del ciclo

    3a233b4  docs(iae): traslado §14.3 reformulada   (previo a este ciclo)
    bff0945  test(p66): contrato §14.3 capa A/B/C
    fe7c1dd  feat(p66): 14.3.6 canonicalizacion FormNum
    3be5465  feat(p66): 14.3.5 mapping + 14.3.1 PASO 1-2 + 14.3.2 cadena
    ea50880  feat(p66): 14.3.1 R3 tri-state + 14.3.4 R4 + L3 booleano
    a00ee48  chore(p66): probe e2e (commit del HEAD al redactar)

### 9.2. Evidencia

    docs/auditoria/iae/evidence/p66_e2e_probe/
      probe_p66_e2e.py    sha256 3a5cd47c3dd0aa24fcd4fef643eaf90d57f654630bc1ae8b62e4ffb9eaa2bb29
      probe_output.txt    sha256 4475e64cdf526bfc7bbcfdb6e261c61ac05e29a4ccb26a706fbd021450d7595a
      README.md           sha256 8c029967f602b0af39cb72c0cc40d8584fdf2afb2ad69cffdfc9d233bb9d9884
      HASHES.txt

### 9.3. Documentos de referencia

    NIPC_CONTRATOS_SEMANTICOS_v1.md §14.3       (contrato reformulado)
    iae/DICTAMENES.md #28 a #40                 (dictamenes del ciclo P66)
    iae/P66_L3_REFORMULACION_PROPUESTA.md       (v7-ter, propuesta congelada)
    iae/evidence/p66_gate04_reissue_period/     (mapping FormNum)
    iae/evidence/p66_gate06_combination_probe/  (universo R4)
    iae/evidence/p66_gate08_full_r4_coverage/   (cobertura identidad R4)
    iae/evidence/p66_gate09_multiple_base/      (multiples bases)
    iae/evidence/p66_gate10_formnum_representation/  (FormNum longitudes)

---

## 10. Estado final

    HEAD local            a00ee48
    origin/main           9d4a81e
    Ahead                 191 commits locales
    Behind                3 (bot CI)
    Working tree          limpio
    Push                  NO

---

Fin del expediente. Redactado 2026-09-21. HEAD a00ee48.