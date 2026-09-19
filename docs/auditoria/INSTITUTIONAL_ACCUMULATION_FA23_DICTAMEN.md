# DICTAMEN AUDITOR - HALLAZGO FA-2.3 IAE

Version: 1.0 (2026-09-19)
Informe evaluado: dictamen de FA-2.3 enviado al auditor (chat, 2026-09-19)
HEAD revisado: 1b778c4 local ([ahead 8])
Estado: vinculante. FA-2.3 PAUSADA con causa identificada.
Naturaleza: no normativo. Si hay conflicto, gana el Prompt Maestro.

---

## 0. Veredicto

FA-2.3 PAUSADA CON CAUSA IDENTIFICADA. NO-GO para FA-2.3 definitiva.
GO para investigacion C (caracterizacion) y posterior fix quirurgico.
El commit 1b778c4 queda como baseline diagnosticado, sin modificar.

## 1. P-FK.1 - FK real

CONFIRMADO por especificacion oficial SEC:
  INFOTABLE.OTHERMANAGER -> OTHERMANAGER2.SEQUENCENUMBER.

NO: INFOTABLE.OTHERMANAGER -> OTHERMANAGER.OTHERMANAGER_SK.

El commit 1b778c4 tiene FK incorrecta real. No se reescribe.

## 2. P5 anterior queda SUPERSEDED

El dictamen P5 previo fue incorrecto en la identificacion fisica de
las tablas. Nuevo modelo:

    SUBMISSION.CIK -> filing_manager_cik
    INFOTABLE.OTHERMANAGER -> split / normalize -> OTHERMANAGER2.
    SEQUENCENUMBER -> CIK -> included_manager_cik.

    OTHERMANAGER -> reporting-for relation -> provenance/secundaria.

El anterior P5 no debe conservarse con descripcion ambigua.

## 3. B.1 - GO

Estructura de clave aprobada:
  (filing_manager_cik, included_manager_cik|null, discretion_type, period).

Semantica: canonical_reporting_relationship. NO economic_owner.

La SEC documenta una relacion de shared investment discretion,
no propiedad economica.

## 4. P-FK.3 - 106 sin match

UNMAPPED. NO fallback por CIK. Los datasets SEC son as-filed y
pueden contener inconsistencias.

  - 105 pares SEQ bajo (1, 2) sin accession en OM2: UNMAPPED.
  - 1 valor 1360533: INVALID / MALFORMED REFERENCE (SEQ=Number(3)).

## 5. P-FK.4 - 0 / NONE

GO condicionado. Normalizacion autorizada:
  0     -> NO_REFERENCE
  NONE  -> NO_REFERENCE

Con precondiciones:
  - Conservar raw_other_manager original.
  - Validar que no resuelve contra ningun SEQUENCENUMBER.
  - Contar por separado nonstandard_no_reference_0 y _NONE.

NO borrar.

## 6. P-FK.5 - Comma-separated

CONFIRMADO 1:N. Explotar relaciones, NO holdings.

Modelo:

    INFO LINE (source_line_id = ACCESSION + INFOTABLE_SK)
        | 1:N
        v
    RELATION (accession, infotable_sk, manager_sequence,
              included_manager_cik)

La linea original NO se divide en shares/value. La 13F no da regla
de reparto proporcional entre managers con shared discretion.

## 7. P-FK.6 - Decision

C: investigar primero. NO fix todavia.

Secuencia autorizada:
  C1  diagnostico 106 UNMAPPED
  C2  caracterizacion 0 / NONE
  C3  invariantes de cardinalidad
       |
       v
  dictamen interno
       |
       v
  fix quirurgico (commit separado: "resolve Column 7 through OTHERMANAGER2")
       |
       v
  nuevos tests

## 8. P-FK.7 - Metrica

Metrica propuesta anterior RECHAZADA (count groups canonical_key+CUSIP > 1).

Metricas correctas:

A. Unicidad de la relacion canonica:
   (ACCESSION, INFOTABLE_SK, INCLUDED_MANAGER_CIK, DISCRETION)
   duplicate_canonical_edges = 0

B. Integridad de la expansion:
   unique source_line_id conservado pre/post.
   relationship_edge_count > source_line_count es esperado (1:N).

C. Cobertura de referencias:
   resolved_reference_tokens / all_reference_tokens
   unmapped_reference_rate
   invalid_reference_rate
   no_reference_rate
   multi_manager_reference_rate

D. Doble conteo economico: NO inferible sin politica adicional.
   Distinguir: TECHNICAL DUPLICATION != SHARED-MANAGER MULTIPLICITY
   != ECONOMIC DOUBLE COUNTING.

## 9. Implicacion arquitectonica

No crear manager -> CUSIP -> value como si fuera titularidad
individual inequivoca. Conservar dimensiones separadas:
  filing_manager, included_manager, source_line, shared_discretion.

La 13F permite afirmar relacion reportada de investment discretion,
NO atribucion economica individual sin regla adicional.

## 10. Tabla de decisiones

| Punto | Dictamen |
|---|---|
| P-FK.1 | CONFIRMADO - FK -> OTHERMANAGER2.SEQUENCENUMBER |
| P-FK.2 | CONFIRMADO - P5 corregido |
| B.1 | GO |
| Semantica B.1 | reporting/discretion, no ownership |
| P-FK.3 | UNMAPPED; no fallback CIK |
| 1360533 | INVALID / MALFORMED REFERENCE |
| P-FK.4 | 0/NONE -> NO_REFERENCE, conservar raw |
| P-FK.5 | 1:N confirmado; explotar relaciones, no holdings |
| P-FK.6 | C - caracterizar antes de corregir |
| P-FK.7 | rechazada metrica propuesta |
| Nueva metrica primaria | unique canonical edges = 0 |
| Doble conteo economico | no inferible todavia |
| FA-2.3 | PAUSADA |
| FA-2.4 | NO-GO hasta fix FA-2.3 |

## 11. Orden operativo

  1. Caracterizar los 106 UNMAPPED.
  2. Caracterizar 0 / NONE.
  3. Definir invariantes de expansion 1:N.
  4. Corregir FK -> OTHERMANAGER2.SEQUENCENUMBER.
  5. Implementar split de Column 7.
  6. Mantener source_line_id.
  7. Tests de unicidad / cobertura / cardinalidad.
  8. Mini-gate FA-2.3.
  9. Solo entonces continuar con amendments (FA-2.4).

---

Fin del dictamen. Version 1.0 (2026-09-19).
