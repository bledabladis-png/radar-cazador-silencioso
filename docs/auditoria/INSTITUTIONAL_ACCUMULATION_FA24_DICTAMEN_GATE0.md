# DICTAMEN AUDITOR - GATE 0 FA-2.4 (AMENDMENTS)

Version: 1.0 (2026-09-19)
HEAD revisado: c0ba083 ([ahead 13] al dictaminar)
Estado: vinculante. GO CONDICIONADO a 2 correcciones documentales.
Naturaleza: no normativo. Si hay conflicto, gana el Prompt Maestro.

---

## 0. Veredicto

GATE 0 APROBADO CONDICIONADO.
Codigo amendments.py: NO autorizado hasta corregir 2 especificaciones.
Push: NO. NIPC: BLOQUEADO.

## 1. P-AM.1 - RESTATEMENT vs NEW HOLDINGS

APROBADO.
  RESTATEMENT  -> reemplaza snapshot anterior.
  NEW HOLDINGS -> anade entradas al snapshot vigente.

NO usar politica "ultimo filing gana" (contradice semantica SEC).

Secuencias no contempladas (ej. HR -> NEW -> RESTATEMENT):
devolver REVIEW_REQUIRED, no improvisar.

## 2. P-AM.2 - Orden de aplicacion (CORREGIDO)

CORRECCION: NO es FILING_DATE -> ACCESSION.
Orden correcto:
  period
  -> original vs amendment
  -> AMENDMENTNO
  -> FILING_DATE
  -> ACCESSION_NUMBER

Razon: SEC define AMENDMENTNO como orden de presentacion.

Invariante nuevo:
  unique (CIK, period, AMENDMENTNO).
Si viola -> AMBIGUOUS_AMENDMENT_ORDER.

## 3. P-AM.3 - 13F-NT/A + NEW HOLDINGS

GO. Excluir del snapshot, registrar anomalia.

  holding_snapshot         = NO
  relationship/provenance  = YES
  anomaly = SOURCE_ANOMALY_NT_AUGMENTED

No fabricar holdings. No convertir a HR.

## 4. P-AM.4 - 13F-HR con ISAMENDMENT=Y

GO. Tratar como original, registrar anomalia.

  canonical_type = 13F-HR
  anomaly = SOURCE_ANOMALY_HR_WITH_AMENDMENT_FLAGS

SUBMISSIONTYPE tiene jerarquia sobre ISAMENDMENT.
Preservar flags conflictivos para auditoria.

## 5. P-AM.5 - Snapshot composicional

CONFIRMADO. El snapshot NO es un unico accession.

Lineage obligatorio:
  applied_accessions = [orig, rest, new_h]
  con operacion: REPLACE | ADD

## 6. Contrato amendments.py (corregido)

  canonical_snapshot  datos finales
  lineage             accessions aplicados
  anomalies           anomalias detectadas
  per_cik_period      estado por grupo:
                        strategy
                        base_accession
                        applied_accessions
                        amendment_sequence
                        status

  status enum:
    CANONICAL
    CANONICAL_COMPOSITE
    NO_HOLDINGS
    SOURCE_ANOMALY
    REVIEW_REQUIRED

## 7. Correccion obligatoria: tabla de estrategias

El informe presentaba 101/22/1 que no reconciliaba con 128 amendments
y 124 grupos. La nomenclatura SINGLE / RESTATED / AUGMENTED / MIXED
no es autoexplicativa.

Reconciliacion correcta:
  119 grupos (HR, HR/A)     = 119 amendments
    4 grupos (HR, HR/A, HR/A) =   8 amendments
    1 grupo  (NT, NT/A)       =   1 amendment
    ----------------------------------------
  124 grupos                  = 128 amendments

Tabla definitiva en INSTITUTIONAL_ACCUMULATION_FA24_ESPECIFICACION.md:
    SINGLE_HR              8,617
    SINGLE_NOTICE          1,907
    HR_PLUS_RESTATEMENT      100
    HR_PLUS_NEW_HOLDINGS      19
    HR_CHAIN_RESTATEMENT       2
    HR_COMPOSITE               2
    NOTICE_AMENDED             1
    TOTAL                 10,648

## 8. Control obligatorio: base filing

No asumir "primer filing es base". Codigo debe comprobar:
  13F-HR -> candidato base.
  13F-NT -> no contiene holdings.
  >1 candidato base -> AMBIGUOUS_BASE.
  HR con flags contradictorios -> base + anomaly.

## 9. Separacion holdings vs graph

Dos productos distintos:
  CANONICAL HOLDINGS SNAPSHOT
  REPORTING / NOTICE LINEAGE

NT / NT/A NO contribuye al snapshot de holdings.
NT / NT/A SI contribuye al grafo.

## 10. Tabla final

| Pregunta | Dictamen |
|---|---|
| P-AM.1 | GO - RESTATEMENT reemplaza, NEW HOLDINGS complementa |
| P-AM.2 | GO - AMENDMENTNO primero; accession solo desempate |
| P-AM.3 | GO - NT/A + NEW HOLDINGS = anomalia |
| P-AM.4 | GO - HR + ISAMENDMENT=Y = anomalia, tratar como HR |
| P-AM.5 | GO - snapshot composicional + lineage |
| FA-2.4 | GO CONDICIONADO (2 correcciones antes de codigo) |
| Push | NO |
| NIPC | BLOQUEADO |

---

Fin del dictamen. Version 1.0 (2026-09-19).
