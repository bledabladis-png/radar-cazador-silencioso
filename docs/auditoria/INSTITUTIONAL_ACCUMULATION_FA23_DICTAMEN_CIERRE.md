# DICTAMEN AUDITOR - CIERRE FIX FA-2.3

Version: 1.0 (2026-09-19)
HEAD revisado: c0ba083 ([ahead 13])
Estado: vinculante. FA-2.3 CERRADA/PASS. FA-2.4 GO.
Naturaleza: no normativo. Si hay conflicto, gana el Prompt Maestro.

---

## 0. Veredicto

FA-2.3 CLOSED / PASS.
FA-2.4 AMENDMENTS + CANONICAL SNAPSHOT: GO.
Push: NO. NIPC: BLOQUEADO.

## 1. Invariantes confirmados

  INFOTABLE rows             3,321,967
  unique source_line_id      3,321,967
  invalid source_line_edges          0
  duplicate canonical edges          0

Implementacion respeta semantica correcta de Column 7 (SEC).

## 2. P-FIN.1 - Cierre FA-2.3

CERRADA. No reabrir para analisis adicionales.

## 3. P-FIN.2 - INVALID_OUT_OF_DOMAIN (1,147)

DOCUMENTAR. NO fallback heuristico.

Subclasificacion diagnostica autorizada (solo documental):
  CIK_LIKE
  SURROGATE_KEY_LIKE
  OTHER_OUT_OF_DOMAIN

Probe CIK-like (922 valores): comparar contra SUBMISSION.CIK.
NO bloqueante para FA-2.4.

## 4. P-FIN.3 - resolution_rate

98.84% ACEPTADO como baseline empirico del resolver.

NO fijar como umbral contractual permanente.

Conservar en futuras ejecuciones:
  resolved_rate
  invalid_rate
  unmapped_rate
  no_reference_rate

## 5. P-FIN.4 - Tokens repetidos

MEDIR, no ocultar. Anadir a metricas de auditoria:
  count_repeated_token_in_raw
  rows_with_repeated_manager_sequence
  repeated_token_occurrences

NO elevar a INVALID. Clasificar:
  SOURCE_ANOMALY / DUPLICATE_REFERENCE

## 6. Deduplicacion

Aprobada. Granularidad correcta:
  (ACCESSION, INFOTABLE_SK, manager_sequence)

No altera source_line_id ni divide holding economicamente.

## 7. duplicate_canonical_edges = 0

Aprobado. Precision:
  demuestra ausencia de duplicacion tecnica del edge.
  NO demuestra ausencia de doble conteo economico.

Separacion obligatoria:
  source_line_id != economic_position_id != reporting_edge_id

## 8. NO_REFERENCE

raw == "0" -> NO_REFERENCE.
raw == "NONE" -> NO_REFERENCE.

Solo para esa linea INFOTABLE. NO confundir con la lista
de Other Included Managers a nivel de filing.

## 9. H-alfa / H-beta

H-alfa (96 duplicados por tokens repetidos): CERRADO.
H-beta (1,147 OOD, 922 CIK-like): anomalia de fuente documentada.

## 10. Pico 25 edges

OBSERVACION ABIERTA NO BLOQUEANTE.

Probe ideal (cuando se documente):
  n_edges = 25 <-> OTHERINCLUDEDMANAGERSCOUNT <-> OM2 SEQs

No autorizar cambios de codigo para este punto.

## 11. P-FIN.5 - FA-2.4 GO

AUTORIZADA. Regla arquitectonica:

  amendment
      |
      v
  canonical snapshot
      |
      v
  reporting relationships definitivas

No asumir que el filing original permanece como snapshot final.

FA-2.4 debe resolver:
  13F-HR
  13F-HR/A
  13F-NT
  13F-NT/A

Y determinar el snapshot canonico ANTES de declarar definitivas las
relaciones manager-security.

## 12. Tabla final

| Elemento | Estado |
|---|---|
| FK Column 7 | CERRADA |
| Split multi-manager | CERRADO |
| Multi-edge | CONFIRMADO |
| source_line_id | INTEGRO |
| Deduplicacion | CERRADA |
| 0 exacto | NO_REFERENCE |
| NONE | NO_REFERENCE |
| 0 embebido | INVALID_REFERENCE_ZERO |
| nombres | INVALID_REFERENCE_NONNUMERIC |
| fuera de dominio | INVALID_OUT_OF_DOMAIN |
| SEQ ausente | UNMAPPED_MISSING_IN_OM2 |
| Fallback CIK/nombre | PROHIBIDO |
| Resolution rate | 98.84% baseline |
| Doble conteo tecnico | 0 |
| Doble conteo economico | NO demostrado / fuera FA-2.3 |
| Repeticiones raw | documentar metrica |
| 922 CIK-like | documentar; probe opcional |
| Pico 25 | observacion no bloqueante |
| FA-2.3 | CERRADA |
| FA-2.4 | GO |
| NIPC | BLOQUEADO |
| Push | NO |

---

Fin del dictamen. Version 1.0 (2026-09-19).
