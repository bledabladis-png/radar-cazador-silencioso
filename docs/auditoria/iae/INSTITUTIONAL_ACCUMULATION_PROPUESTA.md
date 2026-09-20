# PROPUESTA DE MODULO - Institutional Accumulation Evidence (IAE)

Version: 1.0 (2026-09-19)
Prompt de referencia: v6.31 (HEAD 2e29645)
Autor: Ingeniero Supervisor + dictamen auditor externo (vinculante)
Estado: DOCUMENTO DE DISENO. NO IMPLEMENTADO. K-INSTITUTIONAL-ACCUMULATION-01 = MONITORED/BAJA.
Naturaleza: no normativo. Si hay conflicto con el Prompt Maestro, gana el Prompt.

**Nota de estado (2026-09-19):** el ciclo IAE Fase A ha cerrado. FA-1 (ingestion + schema + lineage) y FA-2 (filtro temporal + CUSIP resolver + reporting relationships + amendments) estan CERRADOS / PASS y pusheados. NIPC desbloqueado respecto de FA-2; pendiente su propio Gate. Este documento se preserva sin modificacion (regla: contratos IAE inmutables dentro de una version). Ver docs/auditoria/INSTITUTIONAL_ACCUMULATION_GATE_FA2_DICTAMEN.md.

---

## 1. Proposito

Anadir al Radar una capa de evidencia institucional que responda, con
datos publicos, si existe acumulacion observable de posiciones long
reportadas por institutional investment managers sujetos a Form 13F,
con amplitud, cobertura y persistencia suficientes, y con confirmacion
independiente cuando exista un ETF representativo valido.

NO responde a la pregunta "estan comprando las instituciones": no es
empiricamente defendible con informacion publica diaria.

## 2. Pregunta empirica defendible

> Existe evidencia publica, observable y persistente de aumento de
> posiciones atribuibles a participantes institucionales reportantes,
> acompanada o no por evidencia independiente de flujo primario?

La salida es una clasificacion, no un score unico:
NO_EVIDENCE | INSUFFICIENT | COMPATIBLE | CONFIRMED_INSTITUTIONAL | CONFIRMED_FLOW.

## 3. Alcance y limites

### Incluye
- 13F-HR + 13F-HR/A (SEC EDGAR).
- N-PORT (SEC, dataset publico mensual, retraso 60d post reformas 2024).
- ETF Primary Flow (SSGA + BlackRock + Amundi, ya operativo).
- CFTC TFF (contexto de derivados).
- FINRA ATS (contexto de microestructura).

### Excluye
- Flow Proxy (pertenece a FLOW_PROXY, no a IAE).
- RS, momentum, dark pool como evidencia de acumulacion.
- Superindicador unico.
- Prediccion, timing, buy/sell.
- Imputacion de datos faltantes.

## 4. Arquitectura general

    INSTITUTIONAL ACCUMULATION EVIDENCE
                  |
        +---------+---------+
        v         v         v
       13F      N-PORT   ETF Primary
        |         |         |
        +----+----+---------+
             v
        CROSS VALIDATION (comparability gate previo)
             |
        +----+----+
        v         v
     Breadth  Persistence
        +----+----+
             v
        COVERAGE GATE
             |
        +----+----+
        v         v
   CONFIRMED    COMPATIBLE
   INSTITUTIONAL
        |
        v (si ETF valido)
   CONFIRMED_FLOW

CFTC: contexto derivados (no confirmacion).
FINRA ATS: contexto microestructura (no direccion).
Flow Proxy: fuera del modulo.

## 5. Bloques de evidencia

### 5.1. 13F (SEC EDGAR) - fuente principal
Posiciones long reportadas por managers con discrecion sobre >=$100M
en Section 13(f) securities.
- Periodicidad: trimestral. Deadline: 45 dias post cierre trimestral.
- Campos: manager_cik, manager_name, cusip, shares, market_value,
  put_call, sshPrnamtType (SH/PRN), investment_discretion,
  sole/shared/none voting, report_period, filing_date, accession_number.
- Semantica exacta: "posicion long reportada por institutional
  investment managers sujetos a 13F". NO "ownership institucional total".
- Limitaciones oficiales: no incluye short positions; no es la cartera
  economica completa del manager; puede solaparse con N-PORT.
- Riesgo principal: deduplicacion de reporting relationships
  (parent/subholding, other included managers, combination reports).
- Unidad canonica: la relacion de reporte, NO el CIK aislado.

### 5.2. N-PORT - fuente secundaria
Holdings de fondos registrados.
- Periodicidad: mensual (reformas 2024, publico 60d post mes).
- Uso: corroboracion de 13F, NO sustitucion ni suma.
- Regla: NO sumar 13F shares + N-PORT shares. Solapamiento economico.
- Comparability gate previo obligatorio: same manager + same fund +
  same security + same economic scope.
- Sin umbral numerico de "no contradiccion material" aprobado.

### 5.3. ETF Primary Flow - ya operativo
- Fuentes: SSGA, BlackRock, Amundi. 13 ETFs sectoriales.
- Formula: PrimaryFlow_t = dSharesOutstanding_t * NAV_t.
- Normalizacion: FlowPctAUM = PrimaryFlow / AUM. Ventanas: flow_1d,
  flow_5d, flow_20d, flow_60d, flow_z20, flow_z60.
- Semantica: creacion/redencion primaria. Los Authorized Participants
  son intermediarios, no "compradores institucionales".
- Uso en IAE: confirmacion cuando exista mapping stock -> ETF
  explicito. NO se infiere dinamicamente.

### 5.4. CFTC TFF - contexto de derivados
- Ya operativo (cftc_position_flow.csv).
- Categorias: Dealer, Asset Manager, Leveraged Money, Other, Non-Rep.
- Uso: confirmacion contextual en futuros concretos. NO demuestra
  compra de acciones de un sector.

### 5.5. FINRA ATS - contexto de microestructura
- Ya operativo (marcado ARCHIVAL, retraso 2-4 semanas).
- Datos: volumen agregado + numero de operaciones por ATS/member.
- Uso en IAE: ninguno como evidencia de acumulacion. NO aporta
  direccion economica ni identidad de contrapartes.

## 6. Variables del modulo

### 6.1. DeltaShares canonico
Filtro obligatorio: sshPrnamtType == "SH" AND putCall is null.
Excluye: PRN (principal), Put, Call.
DeltaShares_i,t = Shares_i,t - Shares_i,t-1.
Market_value NO se usa para determinar acumulacion (el precio distorsiona).

### 6.2. Institutional Breadth (IB)
buyers = managers con DeltaShares > 0
sellers = managers con DeltaShares < 0
IB = buyers / (buyers + sellers)
Se reportan ademas: managers_observed, buyers_count, sellers_count,
new_positions, exited_positions. Amplitud y magnitud son dimensiones
separadas; no colapsar.

### 6.3. NIPC (Net Institutional Position Change)
NIPC = sum_i DeltaShares_i
Solo despues de resolver identidad de managers (5.1). Sin resolucion,
NIPC no se publica.

### 6.4. New / Exits
NEW: shares_prev == 0 AND shares_curr > 0
EXIT: shares_prev > 0 AND shares_curr == 0
Evidencias diferentes. No mezclar con aumentos de posiciones existentes.

### 6.5. Coverage
Coverage = acciones_con_dato / universo_elegible
Universo elegible = radar_equities AND section_13f_eligible AND mapped.
INSUFFICIENT si coverage < minimum_coverage. No imputar.

### 6.6. Persistence
13F: Q/Q, 2Q, 3Q, 4Q (trimestral).
N-PORT: M/M, 3M, 6M (mensual).
ETF Primary: 5d, 20d, 60d (diario).
Cada frecuencia conserva su semantica. No mezclar 20d ETF flow con
20d institutional ownership.

## 7. Clasificacion de evidencia

| Estado | Condicion |
|---|---|
| NO_EVIDENCE | cobertura suficiente + sin senal positiva observable |
| INSUFFICIENT | cobertura o temporalidad insuficiente para concluir |
| COMPATIBLE | evidencia positiva pero no cumple confirmacion |
| CONFIRMED_INSTITUTIONAL | 13F + breadth + coverage + persistencia |
| CONFIRMED_FLOW | CONFIRMED_INSTITUTIONAL + ETF Primary Flow valido |

Frase literal para NO_EVIDENCE:
"El dataset no muestra evidencia positiva suficiente en las variables
observadas." NO significa "instituciones no compran".

CONFIRMED_INSTITUTIONAL requiere:
  A. 13F Net DeltaShares > 0
  B. Institutional Breadth > 50%
  C. Coverage >= 60%
  D. al menos 2 periodos 13F consecutivos positivos (backfill permitido)
  E. N-PORT sin contradiccion material (si comparable)
No requiere ETF. No requiere CFTC. No requiere FINRA.

CONFIRMED_FLOW = CONFIRMED_INSTITUTIONAL + ETF Primary Flow positivo
con mapping stock -> ETF explicito y curado.

## 8. Contrato temporal

Cada observacion lleva: source, source_type, entity_id, security_id,
ticker, cusip, sector, observation_date, publication_date, value, units,
report_period, effective_date, coverage.

REGLA CRITICA:
  observation_date = quarter_end / month_end (dato observado)
  publication_date = fecha real del filing SEC
  NUNCA tratar el dato como conocido en observation_date.

Ejemplo: 13F observation_date=2026-06-30, publication_date=2026-08-14.

4 contratos temporales propuestos (alineados con FU-021-5):
SEC_13F_QUARTERLY, SEC_NPORT_MONTHLY, ETF_PRIMARY_DAILY, CFTC_WEEKLY.

## 9. Almacenamiento y lineage

data/sec_13f/ (gitignored)
  +-- 2024Q4/ ... 2026Q2/
  +-- <quarter>.parquet (raw, gitignored)
  +-- <quarter>.manifest.json (versionado)

Manifest por trimestre: accession_number (referencia primaria), CIK,
report_period, filing_date, sha256, source=EDGAR.
Excepcion explicita a FU-002: el artefacto grande no esta en Git.
Trazabilidad descansa en accession_number (identificador unico SEC).

Politica de enmiendas:
  manager + report_period -> 13F-HR + 13F-HR/A (chain)
  Ultima enmienda aceptada = snapshot canonico.
  Chain preservado como lineage. NO tratar enmiendas como
  observaciones adicionales.

## 10. Integracion con el Radar

- Nueva seccion en reporte: "INSTITUTIONAL ACCUMULATION EVIDENCE".
- NO sustituye secciones existentes.
- NO alimenta scores Tactical/Structural.
- NO alimenta SLPM (que se basa en leaders + breadth de precio).
- Capa independiente, descriptiva, auditable.

## 11. Prohibiciones explicitas

NO:
  RS -> institutional accumulation
  momentum -> institutional accumulation
  Flow Proxy -> institutional accumulation
  ATS volume -> institutional buying
  CFTC -> equity-sector institutional buying
  ETF inflow -> institutional buying
  13F market_value change -> accumulation
  deadline como publication_date
  sumar 13F + N-PORT shares
  inferir ETF representativo dinamicamente

## 12. Decisiones vinculantes del auditor

Q1  K-ID MONITORED/BAJA (no deuda activa)
Q2  Deduplicacion 13F = precondicion bloqueante
Q3  Gate 0 CUSIP mapping >=95% + control de peso no mapeado
Q4  DeltaShares = SH + putCall=null
Q5  Universo = radar ∩ section_13f_eligible ∩ mapped
Q6  data/sec_13f/ gitignored + manifest con accession_number
Q7  publication_date = filing_date real (no deadline+1)
Q8  Ultima 13F-HR/A canonica; lineage preservado
Q9  Sin umbral 13F<->N-PORT; comparability gate primero
Q10 ETF no obligatorio para CONFIRMED_INSTITUTIONAL
Q11 Backfill historico, NO esperar 6 meses
Q12 CONFIRMED_INSTITUTIONAL != CONFIRMED_FLOW
Q13 Eliminar "Evidence quality: HIGH"
Q14 Tabla NO_EVIDENCE/INSUFFICIENT/COMPATIBLE/CONFIRMED_* obligatoria
Q15 Siguiente paso: Opcion C (contrato + Gate 0 sobre 100 filings reales)
Q16 Implementacion por fases A->B->C->D->E, no bloque unico

## 13. Precondiciones bloqueantes

NO-GO para implementacion hasta:
  1. Resolver deduplicacion de reporting relationships (unidad = relacion).
  2. Mapping CUSIP->ticker >=95% + control de peso no mapeado.
  3. Contrato de datos cerrado (observation != publication).
  4. Comparability gate 13F<->N-PORT definido.
  5. Almacenamiento aprobado con acceso a accession_number.

## 14. Roadmap por fases

FASE A: 13F core -> Gate
FASE B: N-PORT -> Gate
FASE C: cross-validation -> Gate
FASE D: sector aggregation -> Gate
FASE E: reporte -> Gate

CFTC y FINRA fuera del camino critico inicial.

## 15. Condiciones de reapertura / WONT FIX

K-INSTITUTIONAL-ACCUMULATION-01 = MONITORED/BAJA.

Reabrir ciclo si:
  - se cierra el contrato de datos,
  - Gate 0 sobre 100 filings valida viabilidad,
  - el usuario requiere el modulo con evidencia de uso.

WONT FIX si:
  - deduplicacion no se puede resolver sin identificador externo,
  - coverage CUSIP<->ticker < 95% sin posibilidad de mejora,
  - el coste de mantenimiento supera el valor informativo.

## 16. Regla local-first especifica para IAE

Modulo nuevo, grande, no consolidado. Reglas:
  - NO PUSH a main hasta validar funcionalidad local completa.
  - Trabajo en rama local o stash.
  - Validar con 655 tests + Gate 10/10 sin regresion.
  - Solo tras validacion y verificacion de beneficio para el reporte,
    se pushea.

Razon: evitar contaminar main con codigo no probado o sin valor
funcional confirmado.

## 17. Referencias oficiales

- SEC Form 13F FAQ (division investment management).
- SEC Official List Section 13(f) Securities.
- SEC Form N-PORT Data Sets.
- SEC Press Release 2024-110 (reformas N-PORT).
- CFTC Commitments of Traders (publicreporting.cftc.gov).
- FINRA OTC Transparency (ATS + Non-ATS).
- Investor.gov - Updated Investor Bulletin: ETFs.

---

Fin del documento. Version 1.0 (2026-09-19).
Referencia: Prompt Maestro v6.31 (HEAD 2e29645).
Proximo paso autorizado: Opcion C (contrato + Gate 0 sobre 100 filings
13F reales).