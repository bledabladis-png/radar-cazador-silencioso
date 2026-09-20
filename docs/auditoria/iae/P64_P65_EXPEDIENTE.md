# IAE - Expediente P64 + P65 (Corporate Actions + Manager Duplication)

**Objeto:** registro consolidado del ciclo P64 + P65. Incluye informes,
dictamenes del auditor, correcciones aplicadas, arquitectura final y
decisiones.

**Origen:** F2.4 (2026-09-20), secciones 12 y 13 + dictamenes sucesivos
del auditor externo.

**Estado:** P64 CERRADO. P65 GO CONDICIONADO v3 (pendiente implementar
las 3 correcciones de cierre).

**Referencias autoritativas:**
- Contrato: `iae/NIPC_CONTRATOS_SEMANTICOS_v1.md` secciones 13 y 14.
- Dictamenes: `iae/DICTAMENES.md` (entradas #25 P63, #26 P65 v3).
- Codigo: `src/institutional_accumulation/aggregation/delta_shares.py` (P64),
  `reporting_dedup.py` (P65, PENDIENTE crear).

---

## 0. Frontera semantica (regla transversal)

```
REPORTING RELATIONSHIP  !=  REPORTING NETWORK  !=  DEDUP AUTHORIZATION  !=  ECONOMIC OWNERSHIP
```

Corolarios:
- Una relacion OTHERMANAGER resuelta prueba una relacion de reporting.
- NO prueba por si sola que dos observaciones sean duplicadas.
- NO prueba por si sola que un EXIT + NEW sea un handoff.
- economic_owner_cik PROHIBIDO. En relationships.py::FORBIDDEN_TERMS.
- Un HR normal puede contener holdings de otros managers incluidos.
- Managers bajo control comun pueden presentar 13F-HR separados.
- 13F-NT no tiene Information Table. Aporta L1; L3 solo por evidencia cruzada.

---


## 1. P64 - Corporate Actions Contract

### 1.1. Estado

**CERRADO 2026-09-20.** Commit 2f8140a (4 tests + constantes).

### 1.2. Definicion contractual vigente

```
delta_shares  =  GROSS_OBSERVED_DELTA
```

NO es delta_economic. La separacion economic / mechanical queda
como capacidad diferida v1. Requiere fuente externa de eventos
societarios (CRSP, Compustat, OpenFIGI events, o equivalente).

### 1.3. Eventos declarados fuera de alcance v1

```
split
reverse_split
spin_off
merger
share_class_conversion
```

Constante P64_EVENTS_DEFERRED en delta_shares.py.

### 1.4. CUSIP change: identidad, no evento

CUSIP_A -> canonical_X y CUSIP_B -> canonical_X produce BOTH. Esto
demuestra continuidad de identidad; NO demuestra deteccion del
evento societario concreto.

### 1.5. Codigo

delta_shares.py:
- DELTA_SEMANTICS_GROSS_OBSERVED = True
- P64_EVENTS_DEFERRED = (split, reverse_split, spin_off, merger, share_class_conversion)
- Docstring actualizado: GROSS OBSERVED DELTA.
- SIN columnas delta_shares_economic / delta_shares_mechanical.

### 1.6. Tests contractuales

tests/test_sec_13f_delta_shares.py:
- test_p64_semantica_gross_observed_delta
- test_p64_delta_no_lleva_columnas_economic_mechanical
- test_p64_split_no_se_etiqueta_como_economico
- test_p64_cusip_change_es_identidad_no_corporate_action

### 1.7. Riesgos declarados

- Implementacion: BAJO.
- Semantico downstream: ABIERTO hasta auditar consumidores de delta_shares.
  Audit realizado: solo nipc.py::_sum_delta consume; lee como magnitud,
  no como compra/venta. Probes solo imprimen. Cero consumidores que
  interpreten delta como economic.

---


## 2. P65 - Manager Duplication Contract

### 2.1. Estado

**GO CONDICIONADO (v3) 2026-09-20.** Pendiente aplicar 3 correcciones
de cierre antes de implementar.

### 2.2. Historico de iteraciones

| Version | Fecha | Resultado |
|---|---|---|
| Informe v1 | 2026-09-20 | NO-GO (RESOLVED como autorizacion) |
| Informe v2 | 2026-09-20 | NO-GO CONDICIONADO (L3 incompleto, filing-role incorrecto) |
| Informe v3 | 2026-09-20 | GO CONDICIONADO con 3 correcciones de cierre |

### 2.3. Correcciones de cierre (v3 -> implementable)

1. REPORTING_CONFLICT solo si contradiccion documental explicita.
   NO por mera coincidencia de security.
2. Anadir REPORTING_OVERLAP_UNRESOLVED:
   A reports B on X + B reports X sin evidencia de si son
   complementarias o duplicadas -> KEEP.
3. L3/R1 refinadas: DROP_DUP solo si L3(A,B,S) AND NOT overlap_unresolved.

### 2.4. Arquitectura aprobada (v3)

```
SEC FILINGS
    |
    v
AMENDMENTS (apply_amendments)
    |
    v
EFFECTIVE SNAPSHOT (canonical_snapshot)
    |
    v
REPORTING RELATIONSHIP RESOLUTION
    |
    +--- L1 REPORTING_EDGE
    +--- L2 POSITION_SCOPED_EDGE
    +--- L3 DEDUP_AUTHORIZATION
            |
            v
      INTRA-PERIOD DEDUP (PRE-delta)
            |
            +--- DROP_DUP
            +--- KEEP
            +--- OVERLAP_UNRESOLVED
            +--- REPORTING_CONFLICT
            |
            v
        effective_Q4 / effective_Q1
            |
            v
        delta_shares  (C2 INTACTO)
            |
            v
      REPORTING TRANSITION ENRICHMENT
            |
            +--- HANDOFF
            +--- NULL
            +--- DUPLICATE_REMOVED
            +--- DUPLICATION_UNRESOLVED
            |
            v
        NIPC / coverage
```

**No toca:** MATCH_KEY, C2, delta_shares.py, nipc.py, coverage.py,
relationships.py.

---


### 2.5. Los 3 niveles de evidencia

| Nivel | Nombre | Significado | Autoriza |
|---|---|---|---|
| L1 | REPORTING_EDGE | Relacion documental (Column 7 + OM2 resuelto) | Nada |
| L2 | POSITION_SCOPED_EDGE | Relacion vinculada a security concreta | Nada |
| L3 | DEDUP_AUTHORIZATION | Evidencia cruzada permite afirmar duplicidad | DROP_DUP |

### 2.6. L3 - evidencia cruzada entre filings

```
L3(A, B, S, period) == True sii:

  1. existe filing de A con linea L sobre security S
  2. Column 7(L) referencia a B (reference_status = RESOLVED)
  3. existe filing de B en el mismo periodo
  4. B declara explicitamente que A reporta por B
     (NT o Combination con A en Other Managers Reporting for this Manager)
  5. no existe evidencia contradictoria
     ni evidencia de reporting partition/overlap no resuelto
```

### 2.7. R1 (intra-period dedup) - 7 requisitos

```
Para dos observaciones S1 (M1) y S2 (M2) del mismo periodo:

  1. S1.canonical_security == S2.canonical_security
  2. S1.discretion_type    == S2.discretion_type
  3. S1.filing_manager_cik != S2.filing_manager_cik
  4. Existe relacion documental explicita M1 -> M2 (Column 7 de S1)
  5. La relacion esta vinculada a la security S concreta
  6. Existe evidencia cruzada L3: M2 declara que M1 reporta por M2
  7. NO existe evidencia contradictoria ni overlap no resuelto

Si las 7 se cumplen:
  dedup_decision = DROP_DUP
  dedup_reason = INTRA_PERIOD_DUP
  se conserva S1 (documented reporting representative)
  dedup_audit registra la eliminacion.

Si falla cualquier requisito:
  dedup_decision = KEEP
  dedup_reason = NULL | DUPLICATION_UNRESOLVED | REPORTING_OVERLAP_UNRESOLVED | REPORTING_CONFLICT
```

### 2.8. R2 (handoff inter-periodo)

```
Q4 unit A:
  filing_manager_cik          = A
  reporting_for_manager_cik   = B
  canonical_security          = S
  discretion_type             = D

Q1 unit A':
  filing_manager_cik          = B
  reporting_for_manager_cik   = B
  canonical_security          = S
  discretion_type             = D

AND:
  - A y A' producen EXIT + NEW en delta_shares (C2 intacto)
  - reporting_for_manager_cik es ESTABLE (B == B)
  - existe continuidad documental explicita de la misma relacion de reporting

  -> reporting_transition = HANDOFF
  -> dedup_reason = POSITION_SCOPED_HANDOFF
  -> match_status = EXIT + NEW (sin cambios)
  -> delta_shares (sin cambios)
```

**NO se anade STATUS_REPORTING_HANDOFF a delta_shares.py.**

### 2.9. Caso REPORTING_OVERLAP_UNRESOLVED

```
A reports for B on X
+
B also reports X
```

Sin evidencia de si portion A + portion B son complementarias o
duplicadas.

```
dedup_decision = KEEP
dedup_reason = REPORTING_OVERLAP_UNRESOLVED
```

Fail-closed.

### 2.10. Caso REPORTING_CONFLICT (definicion estricta)

Reservado a:

```
B declara explicitamente una condicion incompatible con
"A reports for B on X"
```

Ejemplo:

```
B: "all holdings are reported by A" (13F-NT)
PERO
B effective filing: X declarado como self-reported
SIN estructura Combination/partition que explique ambas cosas.
```

```
dedup_decision = KEEP
dedup_reason = REPORTING_CONFLICT
```

### 2.11. R3, R4, R5 (sin cambios)

- R3 - Sin RESOLVED o sin included_manager_cik -> KEEP.
- R4 - Amendments antes de relaciones. 2 tests obligatorios
  (RESTATEMENT + NEW HOLDINGS).
- R5 - 13F-NT: holdings=0, relacion preservada. NT aporta L1,
  no L3 por si solo. L3 se alcanza combinando con evidencia
  position-scoped del representante.

### 2.12. reporting_for_manager_cik (campo nuevo)

```
filing_manager_cik         quien presento el filing
reporting_for_manager_cik  para que manager se reporta esta observacion
economic_owner_cik         NO EXISTE. Prohibido.
```

reporting_for_manager_cik NO se usa para dedup. Se usa solo en R2.

### 2.13. dedup_audit - columnas obligatorias

```
source_line_id                    ACCESSION + INFOTABLE_SK
period
filing_manager_cik
reporting_for_manager_cik
canonical_security
dedup_decision                    KEEP | DROP_DUP
dedup_reason                      INTRA_PERIOD_DUP | REPORTING_OVERLAP_UNRESOLVED |
                                  REPORTING_CONFLICT | NULL
evidence_level                    L1 | L2 | L3
evidence_source                   accession_representante | accession_representado | both
evidence_accession_representante
evidence_accession_representado
evidence_reference_seq            SEQ de Column 7 (para reconstruir sin re-inferir)
```

### 2.14. reporting_network_id

Solo diagnostico. NO participa en DROP_DUP ni HANDOFF.

---


### 2.15. Tests obligatorios (13 + 2 nuevos)

| # | Test | Nivel |
|---|---|---|
| 1 | Column 7 -> manager correctamente resuelto | L1 |
| 2 | Column 7 inexistente / referencia invalida -> KEEP | - |
| 3 | HR/Combination + Column 7 + contraparte NT + misma security -> DEDUP_ELIGIBLE | L3 |
| 4 | same reporting_network + separate HR -> KEEP | L1 |
| 5 | relationship resolved sin position scope -> KEEP | L1 |
| 6 | NT -> 0 holdings | - |
| 7 | NT -> relationship preserved | L1 |
| 8 | RESTATEMENT -> before relationship resolution | R4 |
| 9 | NEW HOLDINGS -> before relationship resolution | R4 |
| 10 | economic owner never inferred (semantico) | - |
| 11 | EXIT + NEW + explicit reporting continuity -> HANDOFF | L3 |
| 12 | EXIT + NEW + network only -> NULL / KEEP | L1 |
| 13 | Column 7 -> B + B HR mismo security + overlap no resuelto -> KEEP | - |
| 14 | (nuevo) B filing declara condicion incompatible -> REPORTING_CONFLICT -> KEEP | - |
| 15 | (nuevo) same_security_cross_filing_partition_not_deduped -> KEEP | - |

Test 13 cambiado respecto a v2 (era REPORTING_CONFLICT, ahora es
REPORTING_OVERLAP_UNRESOLVED). Tests 14 y 15 nuevos.

### 2.16. Interfaces propuestas

aggregation/reporting_dedup.py (NUEVO):

```
def build_effective_reporting_snapshot(
    units_q4, units_q1,
    relationships_q4, relationships_q1,
    cross_filing_evidence,
    *,
    period_q4: str, period_q1: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Devuelve (effective_q4, effective_q1, dedup_audit)."""

def classify_reporting_transition(
    delta_df,
    effective_q4, effective_q1,
) -> pd.DataFrame:
    """Anade columnas reporting_transition + dedup_reason.
    NO modifica match_status ni delta_shares."""
```

Constantes:

```
TRANSITION_NULL                    = "NULL"
TRANSITION_HANDOFF                 = "HANDOFF"
TRANSITION_DUPLICATE_REMOVED       = "DUPLICATE_REMOVED"
TRANSITION_DUPLICATION_UNRESOLVED  = "DUPLICATION_UNRESOLVED"

DEDUP_REASON_INTRA_PERIOD_DUP      = "INTRA_PERIOD_DUP"
DEDUP_REASON_POSITION_HANDOFF      = "POSITION_SCOPED_HANDOFF"
DEDUP_REASON_REPORTING_CONFLICT    = "REPORTING_CONFLICT"
DEDUP_REASON_OVERLAP_UNRESOLVED    = "REPORTING_OVERLAP_UNRESOLVED"
```

### 2.17. Plan de implementacion (4 commits, no 1)

Orden metodologico exigido por el auditor:

```
Commit 1    contrato + modelos + tests unitarios L1/L2/L3
Commit 2    reporting_dedup pre-delta
Commit 3    transition classification
Commit 4    integracion e2e + evidencia
```

Justificacion: permite identificar que modificacion introduce cualquier
cambio de resultado.

---

## 3. Decisiones consolidadas F2.4

| Decision | Resultado |
|---|---|
| P64 gross observed delta | GO |
| P64 economic delta | DIFERIDO v1 |
| P64 mechanical delta | DIFERIDO v1 |
| P64 heuristica split | NO-GO |
| P65 L1/L2/L3 | GO |
| P65 R1 corregida (7 req.) | GO con overlap_unresolved |
| P65 R2 corregida (reporting_for) | GO |
| P65 R3, R4, R5 | GO |
| P65 REPORTING_CONFLICT | GO (definicion estricta) |
| P65 REPORTING_OVERLAP_UNRESOLVED | ANADIDO |
| P65 MATCH_KEY / C2 | INTACTOS |
| P65 delta_shares.py | INTACTO |
| P65 economic_owner | PROHIBIDO |
| P65 reporting_network_id | Diagnostico |
| P65 4 commits | OBLIGATORIO |

---

## 4. Estado global tras F2.4 (P60-P65)

| Fix | Commit | Estado |
|---|---|---|
| P60 fail-closed | 5d60d48 | CERRADO |
| P61 resolve_source_status | 1773cc9 | CERRADO |
| P38 coverage contractual | a48717a | CERRADO |
| P63 tests contractuales | a841cd6 | CERRADO |
| P63 contrato | 05f0a84 | CERRADO |
| P64 gross delta | 2f8140a | CERRADO |
| P65 v3 diseno | (este) | GO CONDICIONADO, 3 correcciones |
| P65 implementacion | PENDIENTE | 4 commits |
| P62 point-in-time | PENDIENTE | Requiere OpenFIGI |
| Bloqueante 1 (TARGET indep.) | PENDIENTE | Requiere OpenFIGI masivo |

---

## 2.18. Evidencia empirica (probe P65 e2e, 2026-09-20)

**Probe:** `docs/auditoria/iae/evidence/nipc_p65_probe/probe_p65_e2e_reduced.py`
**Ejecutado:** 200 CIKs (primeros por ACCESSION_NUMBER ascendente), Q4 2025 -> Q1 2026.
**Salida cruda:** `output.txt` (LF puro, 100 lineas).
**Hashes:** `HASHES.txt`.

### Pipeline validado

    filter_by_period
      -> apply_amendments (canonical_snapshot)
      -> resolve_batch_identities
      -> compute_reported_position_units
      -> build_effective_reporting_snapshot (dedup pre-delta)
      -> compute_delta_shares
      -> classify_reporting_transition (post-delta)

### Resultado

    Q4 2025: SUBMISSION 200 -> snapshot 125, INFOTABLE 325.546, units 151.222
    Q1 2026: SUBMISSION 200 -> snapshot 106, INFOTABLE 276.312, units 126.853

    DEDUP PRE-DELTA:
      effective_q4 = 151.222  (KEEP integral)
      effective_q1 = 126.853  (KEEP integral)
      dedup_audit rows = 0

    DELTA SHARES:
      delta rows = 252.341
      match_status = {UNRESOLVED_IDENTITY: 216.722,
                      BOTH: 25.734, EXIT: 7.625, NEW: 2.260}

    REPORTING TRANSITION (POST-DELTA):
      reporting_transition = {NULL: 252.341}  (100% NULL)

### Veredicto

El pipeline P65 v1 se comporta **fail-closed** sin evidencia cruzada L3:

- `dedup_decision = KEEP` en todas las unidades.
- `reporting_transition = NULL` en todas las filas delta.
- `dedup_audit` vacio (no hay decisiones que auditar).
- `match_status` intacto (C2 preservado): BOTH/EXIT/NEW/UNRESOLVED_IDENTITY.

Exactamente el comportamiento contractual exigido por el dictamen P65 v3.

### Hallazgo colateral

El probe destapo un dato de produccion desalineado con el contrato P60:
`data/mappings/cusip_equivalence.csv` no tenia la columna `identity_type`.
Fix en commit 3c2140e (cabecera actualizada, 0 filas de datos afectadas).

Evidencia de que la fase e2e real detecta desalineaciones que los tests
unitarios no ven (los tests construyen fixtures con la columna correcta).

### Limitaciones declaradas

1. `cross_filing_evidence` = vacio. La dedup real requiere evidencia L3
   cruzada entre filings. El campo "Other Managers Reporting for this
   Manager" esta en `ADDITIONALINFORMATION` (texto libre); parsear texto
   libre esta prohibido por las reglas del sistema.
2. `DROP_DUP` no se emite en v1 con 13F puro. La coexistencia de dos
   filings sobre la misma security es OVERLAP por definicion. Requiere
   evidencia cuantitativa externa (capacidad diferida v2).
3. Probe reducido a 200 CIKs. El completo (~10 min) es viable pero no
   necesario para la validacion contractual.

---

Fin del expediente P64 + P65. Version 1.0 (2026-09-20).
