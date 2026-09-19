# DICTAMEN AUDITOR - CARACTERIZACION C FA-2.3

Version: 1.0 (2026-09-19)
Informe evaluado: caracterizacion C (chat, 2026-09-19)
HEAD revisado: post-dictamen FK ([ahead 9])
Estado: vinculante. CARACTERIZACION APROBADA. FIX QUIRURGICO AUTORIZADO.
Naturaleza: no normativo. Si hay conflicto, gana el Prompt Maestro.

---

## 0. Veredicto

CARACTERIZACION APROBADA. FIX QUIRURGICO AUTORIZADO con criterios
congelados. Push NO. NIPC BLOQUEADO. FA-2.4 aun NO.

## 1. Reglas congeladas para el fix

  FK:
    INFOTABLE.OTHERMANAGER -> OTHERMANAGER2.SEQUENCENUMBER

  split: comma-separated (1:N)

  Clasificacion de tokens:
    raw == "0"                  -> NO_REFERENCE
    raw == "NONE"               -> NO_REFERENCE
    token "0" dentro de lista   -> INVALID_REFERENCE_ZERO
    token no numerico           -> INVALID_REFERENCE_NONNUMERIC
    SEQ positivo ausente de OM2 -> UNMAPPED_MISSING_IN_OM2
    SEQ numerico fuera dominio  -> INVALID_OUT_OF_DOMAIN
    token resuelto              -> RESOLVED

  source_line_id: preservado (accession, infotable_sk).
  Multi-manager: multi-edge. NO dividir holding economicamente.
  CIK fallback: NO.

## 2. Hallazgo INVALID 58,252

Los tokens INVALID son nombres en Column 7 (ej. "ARK Advisory").
La propia presentacion SEC del filing ARK Q1 2026 contiene ese
valor, pese a que la instruccion oficial exige solo numeros.

Clasificacion: INVALID_REFERENCE_NONNUMERIC.
NO es "corrupcion sistematica del dataset SEC". Es:
  "source-level non-conforming value in Column 7".

NO fallback por nombre. Preservar raw_othermanager.

## 3. Hallazgo UNMAPPED 17,492

SEPARAR dos patrones:
  - token "0" post-split -> INVALID_REFERENCE_ZERO
  - SEQ positivo ausente -> UNMAPPED_MISSING_IN_OM2

No fallback por CIK.

## 4. 0 / NONE

raw == "0" -> NO_REFERENCE.
raw == "NONE" -> NO_REFERENCE.

NO significa "sin managers en el accession". El filing puede
tener OM2 y aun asi la linea concreta no compartir discrecion.

Contadores separados:
  nonstandard_no_reference_0
  nonstandard_no_reference_NONE
Sin usarlos como evidencia de ausencia de managers del filing.

## 5. Pico 25 x 1,000

INVESTIGACION OBLIGATORIA ANTES DEL CIERRE DE FA-2.3.
No bloquea la arquitectura del resolver.

Prueba decisiva:
  edge_count_per_row = 25
  AND OTHERINCLUDEDMANAGERSCOUNT = 25
  AND OM2 count = 25

Si ocurre -> patron estructural legitimo del filing.
Si no -> anomalia de integridad.

## 6. Modelo multi-edge

CONFIRMADO. Estructura de dos niveles:

  Nivel 1 - edge de provenance:
    (accession, infotable_sk, manager_sequence)

  Nivel 2 - relacion canonica:
    (filing_manager_cik, included_manager_cik, discretion_type, period)

El sequence number pertenece a la evidencia fuente.
El CIK pertenece a la identidad resuelta.

INFOTABLE_SK -> manager_sequence -> OTHERMANAGER2 -> CIK
              -> canonical relationship.

## 7. Metrica de cobertura

La 94.74% actual es baseline diagnostica, NO threshold de calidad.
Mezcla referencias resolubles con anomalias.

Metricas aprobadas:

  Integridad de fuente:
    source_line_count
    unique_source_line_id  (== INFOTABLE rows)
    invalid_source_line_edges (== 0)

  Integridad de relacion:
    resolved_edges
    unmapped_missing_om2
    invalid_reference_non_numeric
    invalid_reference_zero

  Unicidad tecnica:
    (ACCESSION_NUMBER, INFOTABLE_SK, manager_sequence) unico

  Cobertura:
    raw token resolution rate
    valid-reference resolution rate

No fijar ahora un >=95% como Gate del resolver.

## 8. 1360533

INVALID_OUT_OF_DOMAIN. SEQUENCENUMBER es NUMBER(3).
No buscar por coincidencia con surrogate keys.

## 9. P-FK.6/C cierra

Ya no se requiere otra investigacion general.
Unica investigacion adicional: pico 25 x 1,000.

## 10. Tabla final

| Punto | Dictamen |
|---|---|
| FK real | CONFIRMADA |
| OTHERMANAGER2 | tabla directa para Column 7 |
| B.1 | GO |
| ARK Advisory | INVALID_REFERENCE_NONNUMERIC |
| Fallback por nombre | NO |
| 0 exacto | NO_REFERENCE |
| NONE | NO_REFERENCE |
| 0 embebido | INVALID_REFERENCE_ZERO |
| SEQ ausente de OM2 | UNMAPPED_MISSING_IN_OM2 |
| 1360533 | INVALID_OUT_OF_DOMAIN |
| Multi-edge | CONFIRMADO |
| Division economica | NO |
| source_line_id | preservar |
| 94.74% | baseline, no threshold |
| Pico 25 x 1,000 | INVESTIGAR |
| Fix quirurgico | AUTORIZADO |
| FA-2.4 | NO |
| Push | NO |
| NIPC | BLOQUEADO |

---

Fin del dictamen. Version 1.0 (2026-09-19).
