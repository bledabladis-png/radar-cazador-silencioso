# INFORME PROBE END-TO-END NIPC - Q4 2025 -> Q1 2026

Version: 1.0 (2026-09-19)
HEAD revisado: b75c262
Estado: probe completado. Hallazgo estructural no cubierto por el contrato.
Autor: Ingeniero Supervisor.
Naturaleza: no normativo. Si hay conflicto con el Prompt Maestro, gana el
            Prompt Maestro.

---

## 1. Proposito

Ejecutar el pipeline NIPC completo (modulos S1-S5 del ciclo v1.2) sobre los
datasets reales Q4 2025 + Q1 2026, midiendo por primera vez:

- elegibilidad SEC Official List
- resolucion de identidad canonica (C1-revisada)
- DeltaShares por reported_position_unit
- NIPC total + breakdown por discretion
- coverage pairwise (6 metricas)
- status derivado

NO es Gate-NIPC.3 formal. Es un probe de caracterizacion previo al informe
de Gate-NIPC.3, para detectar hallazgos antes de invertir en umbrales o
en OpenFIGI masivo.

---

## 2. Setup empirico

| Item | Q4 2025 | Q1 2026 |
|---|---|---|
| Periodo canonico (PERIODOFREPORT) | 2025-12-31 | 2026-03-31 |
| Filings SUBMISSION filtrados | 11,372 | 11,761 |
| Filings SUBMISSION canonicos (post-amendments) | 8,635 | 8,762 |
| INFOTABLE crudo | 3,473,209 | 3,822,885 |
| INFOTABLE canonico | 3,211,193 | 3,239,273 |
| CUSIPs unicos con SH + PUTCALL NULL | 24,129 | 24,580 |
| SEC Official List - lineas | 12,282 | 24,641 |
| SEC Official List - CUSIPs unicos | 12,282 | 22,659 |

Crosswalk interno aplicado:
  - cusip_ticker_exceptions.csv: 3 filas (DD, HON, XOM)
  - etf_holdings.csv: 526 filas
  - cusip_equivalence.csv: 0 filas (tabla vacia por diseno)

---

## 3. Metricas del motor

### 3.1. Elegibilidad SEC Official List

| Metrica | Q4 2025 | Q1 2026 |
|---|---|---|
| n_total (CUSIPs SH+null) | 24,129 | 24,580 |
| eligible (ADDED + ACTIVE) | 11,693 | 12,940 |
| not_in_list | 12,281 | 11,500 |
| deleted | 155 | 139 |
| active | 10,961 | 12,212 |
| added | 732 | 728 |
| conflict | 0 | 1 |
| pct_eligible | 0.4846 | 0.5264 |

Cobertura coherente con Gate 0 SEC 13(f) PASS/CLOSED.

### 3.2. Resolucion de identidad canonica

| Metrica | Q4 2025 | Q1 2026 |
|---|---|---|
| n_total | 24,129 | 24,580 |
| CANONICAL | 498 | 502 |
| OBSERVED_ONLY | 23,631 | 24,078 |
| AMBIGUOUS | 0 | 0 |
| CONFLICT | 0 | 0 |
| UNRESOLVED | 0 | 0 |
| pct_canonical | 0.0206 | 0.0204 |

by_kind:
  Q4: CANONICAL_EQUIVALENCE=498, OBSERVED_CUSIP_ONLY=23,631
  Q1: CANONICAL_EQUIVALENCE=502, OBSERVED_CUSIP_ONLY=24,078

Interpretacion:
  - Los ~500 CUSIPs canonicos son los que resuelve el crosswalk actual
    (curados + etf_holdings). Coherente con Gate 0 Mapping
    (502/24,838 = 2.021% del universo 13F completo).
  - 98% de CUSIPs caen en OBSERVED_ONLY. Por contrato (C1-revisada),
    NO entran en el matching interperiodo.

### 3.3. NIPC observado

Scope ALL (todos los CUSIPs SH+null):

  Q4 units: 2,360,246    Q1 units: 2,420,326
  Delta filas: 4,082,327
  match_status:
    UNRESOLVED_IDENTITY  3,177,329
    BOTH                   698,245
    NEW                    113,710
    EXIT                    93,043

  NIPC total observable:  +23,651,586
  nipc_sole:             -41,161,904,992
  nipc_dfnd:             +41,246,135,898
  nipc_otr:                  -60,579,320

Scope ELIGIBLE (filtrado por SEC Official List):

  Q4 units: 2,335,031    Q1 units: 2,397,186
  Delta filas: 4,034,035
  NIPC total observable:  +52,570,648
  nipc_sole:             -41,160,312,485
  nipc_dfnd:             +41,276,572,445
  nipc_otr:                  -63,689,312

### 3.4. Coverage pairwise (6 metricas)

Scope ALL:

  coverage_previous                0.0206
  coverage_current                 0.0204
  paired_security_coverage         0.0181
  paired_weighted_share_coverage   0.2271
  unmapped_weight_previous         0.9794
  unmapped_weight_current          0.9796

Scope ELIGIBLE:

  coverage_previous                0.0424
  coverage_current                 0.0386
  paired_security_coverage         0.0368
  paired_weighted_share_coverage   0.2772
  unmapped_weight_previous         0.9576
  unmapped_weight_current          0.9614

### 3.5. Status derivado

Ambos scopes: INSUFFICIENT.

Motivo: thresholds UNDEFINED en NIPC_COVERAGE_POLICY.md.

Aun con thresholds fijados, paired_security_coverage < 0.04 bloquearia
READY en cualquier umbral realista.

---

## 4. Hallazgo estructural: reorganizacion de filer CIK (caso Vanguard)

### 4.1. Observacion

En el TOP 10 |delta| del scope ALL, los mayores movimientos no son
compras/ventas, son reasignaciones entre filers del mismo grupo:

    FM=0000102909  sec=equity:NVDA  SOLE  prev=2,026,728,493  curr=0  EXIT
    FM=0002100119  sec=equity:NVDA  DFND  prev=0  curr=1,538,550,382  NEW
    FM=0000102909  sec=equity:AAPL  SOLE  prev=1,279,051,701  curr=0  EXIT
    FM=0002100119  sec=equity:AAPL  DFND  prev=0  curr=953,847,648   NEW
    FM=0000102909  sec=equity:AMZN  SOLE  prev=749,020,454   curr=0  EXIT
    FM=0002100119  sec=equity:AMZN  DFND  prev=0  curr=631,176,205   NEW
    (y 4 mas con el mismo patron)

### 4.2. Evidencia: continuidad de filer

Query 1 - filings por CIK y periodo:

  Q4 2025:
    CIK 0000102909 (VANGUARD GROUP INC)  1 filing  13F-HR
    CIK 0002100119                        0 filings
    CIK 0002100121                        0 filings

  Q1 2026:
    CIK 0000102909 (VANGUARD GROUP INC)  1 filing  13F-NT
    CIK 0002100119 (VANGUARD CAPITAL)    3 filings 13F-HR + 2x13F-HR/A
    CIK 0002100121 (VANGUARD PORTFOLIO)  2 filings 13F-HR + 13F-HR/A

Query 2 - mismo CUSIP, ambos periodos:

  CUSIP 67066G104 (NVDA): presente en Q4 (9,699 filas) y Q1 (10,287 filas)
  CUSIP 037833100 (AAPL): presente en Q4 (9,763 filas) y Q1 (10,564 filas)
  CUSIP 023135106 (AMZN): presente en Q4 (9,764 filas) y Q1 (10,401 filas)

Los CUSIPs no cambiaron. La security es identica en ambos periodos.

Query 3 - COVERPAGE.FILINGMANAGER_NAME:

  0000102909-26-002707  VANGUARD GROUP INC
  0002100119-26-001306  VANGUARD CAPITAL MANAGEMENT LLC
  0002100119-26-001311  VANGUARD CAPITAL MANAGEMENT LLC
  0002100119-26-001313  VANGUARD CAPITAL MANAGEMENT LLC
  0002100121-26-000861  VANGUARD PORTFOLIO MANAGEMENT LLC
  0002100121-26-000865  VANGUARD PORTFOLIO MANAGEMENT LLC

### 4.3. Mecanismo

Vanguard reorganizo su estructura de filing entre Q4 2025 y Q1 2026:

  Q4 2025: un unico filer (VANGUARD GROUP INC) reporta holdings.
  Q1 2026: VANGUARD GROUP INC presenta 13F-NT (notice declarando que
           sus posiciones se reportan por otros managers); los holdings
           migran a VANGUARD CAPITAL MANAGEMENT LLC + VANGUARD PORTFOLIO
           MANAGEMENT LLC.

Es exactamente el escenario que la SEC documenta para 13F-NT: la entidad
padre declara que otro manager reporta sus posiciones.

### 4.4. Impacto cuantitativo

Con el match key contractado (C2, spec 4.7):

  match_key = (filing_manager_cik, canonical_security, discretion_type)

SIN report_period.

Consecuencia:
  - CIK viejo presente en Q4 y ausente en Q1 -> EXIT.
  - CIKs nuevos ausentes en Q4 y presentes en Q1 -> NEW.

Resultado:
  nipc_sole  = -41.16B
  nipc_dfnd  = +41.28B

Los dos valores se cancelan casi exactamente (+52M netos sobre ~82B
brutos). NO son flujos reales. Es la misma masa accionarial
reportandose por un filer distinto.

---

## 5. Analisis del ingeniero

### 5.1. El motor funciona segun contrato

NO es bug. El sistema implementa exactamente C2 (spec 4.7). El dictamen
v1.1 fijo la match key sin report_period precisamente para detectar
continuidad economica via manager CIK. El caso Vanguard no es continuidad
de manager: es discontinuidad de filer.

### 5.2. El contrato no cubre este caso

El dictamen v1.1 no contempla la posibilidad de que un mismo grupo
economico reorganice sus filings entre trimestres. Ni prohibe ni
autoriza reconciliacion via parent/child.

Prohibiciones vigentes (relationships.py FORBIDDEN_TERMS):
  - economic_owner_cik
  - owner_cik
  - beneficial_owner_cik

Es decir: el contrato no permite reconciliar por parent/child. El
ingeniero NO tiene autoridad para introducir esa reconciliacion.

### 5.3. Consecuencia para el producto NIPC

Con el dataset real:
  - NIPC observado (scope ALL)      ~ +24M   (cancelado)
  - NIPC observado (scope ELIGIBLE) ~ +53M   (cancelado)

La senal observable es practicamente cero una vez descontado el efecto
reorganizacion. NIPC seguiria en INSUFFICIENT aunque tuvieramos 100%
de mapping CUSIP->ticker: el gap estructural dominante no es la
cobertura de CUSIPs, es la continuidad de filer.

### 5.4. Indicio adicional

El filing 13F-NT de Q1 contiene, por diseno SEC, referencias a los
managers que reportan los holdings (Column 7 -> OTHERMANAGER2.SEQUENCENUMBER).
El sistema YA resuelve esas referencias via relationships.py. No se ha
verificado aun si en el caso Vanguard el NT apunta explicitamente a
los CIKs subsidiarios, o si esa informacion no esta presente.

Pendiente de probe adicional (no bloqueante para este informe).

---

## 6. Preguntas al auditor

Q-PROBE-1  El hallazgo se documenta como limitacion conocida del
           contrato (del tipo "NIPC asume continuidad de filer entre
           trimestres; no valida reorganizaciones de filing") o se
           considera trigger para diseno de una capa de reconciliacion
           NT <-> HR?

Q-PROBE-2  Si se autoriza reconciliacion: se permite via
           OTHERMANAGER2 pointers del 13F-NT (ya resueltos por FA-2.3)
           o requiere una fuente adicional?

Q-PROBE-3  Los thresholds de NIPC_COVERAGE_POLICY.md se fijan
           considerando que, aun con cobertura alta, el sesgo de
           reorganizacion de filer puede dominar la senal? Se propone
           fijar thresholds acompanados de un segundo control
           ("continuidad de filer") que descarte trimestres con alta
           discontinuidad. Confirmar o proponer alternativa.

Q-PROBE-4  El scope ALL tiene paired_security_coverage=0.0181 y el
           scope ELIGIBLE 0.0368. Ninguno de los dos es publicable.
           Se confirma que el motor queda en modo diagnostico hasta
           que el crosswalk crezca, o se autoriza un tercer scope
           ("canonical subset") para publicacion restringida?

Q-PROBE-5  El caso Vanguard puede ser sistemico (grandes gestoras
           reorganizando filers) o puntual. Se autoriza un mini-probe
           (top 20 filers por SSHPRNAMT total, comparar CIK entre
           Q4 y Q1) antes de fijar la politica?

---

## 7. Recomendacion del ingeniero

1. Aceptar el hallazgo. NO tocar codigo (C2 es correcto por contrato).
2. Documentar como limitacion conocida en NIPC_COVERAGE_POLICY.md y
   en la spec v1.3.
3. Ejecutar mini-probe Q-PROBE-5 antes de decidir si es caso puntual
   o sistemico.
4. Una vez cuantificado, fijar thresholds con el control adicional
   propuesto en Q-PROBE-3.
5. Autorizar (o rechazar) reconciliacion NT <-> HR segun Q-PROBE-2.

---

## 8. Estado del repo

  HEAD local: b75c262.
  origin/main: 9d4a81e.
  Ahead: 30 commits.
  Working tree: limpio.
  Push: NO (regla local-first IAE).

---

## 9. Anexo: definicion exacta del scope

  scope ALL       = todos los CUSIPs con SH + PUTCALL NULL del
                    canonical_snapshot del periodo.

  scope ELIGIBLE  = scope ALL filtrado a los CUSIPs con
                    section13f_eligible = (status in {ADDED, ACTIVE})
                    segun SEC Official List del periodo.

En ambos scopes, la resolucion de identidad usa el crosswalk interno
actual (3 curados + 526 ETFs) + cusip_equivalence.csv (vacio).

---

Fin del informe. Version 1.0 (2026-09-19). HEAD b75c262.
Pendiente: dictamen del auditor sobre Q-PROBE-1 a Q-PROBE-5.
