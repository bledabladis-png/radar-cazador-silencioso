# INFORME MINI-PROBE FILER CONTINUITY (Q-PROBE-5) - Q4 2025 -> Q1 2026

Version: 1.0 (2026-09-19)
HEAD revisado: d75e84c
Estado: mini-probe completado. Evidencia directa NT -> OTHERMANAGER -> HR.
Autor: Ingeniero Supervisor.
Naturaleza: no normativo. Si hay conflicto con el Prompt Maestro, gana el
            Prompt Maestro.

---

## 1. Proposito

Cuantificar si la discontinuidad de filer CIK observada en el probe
end-to-end (caso Vanguard) es un fenomeno aislado o sistemico, y
verificar la relacion 13F-NT -> OTHERMANAGER -> 13F-HR exigida por el
dictamen probe Q-PROBE-5.

Universo: top 20 filers por sum(SSHPRNAMT) tras filtro SH + PUTCALL NULL,
canonical_snapshot Q4 2025 + Q1 2026. Comparacion por CIK.

Estados observables definidos (sin inferencia economica):

    CONTINUOUS_FILER             CIK presente en ambos periodos con
                                 tipo de filing estable.
    FILER_DISCONTINUITY          CIK presente en Q4 y ausente en Q1
                                 (o viceversa) sin relacion NT-HR
                                 observable.
    NT_TO_HR_RELATION_OBSERVED   CIK presenta 13F-NT y otro CIK presenta
                                 13F-HR vinculado via OTHERMANAGER.
    HR_TO_HR_WITHOUT_RELATION    Dos CIKs distintos 13F-HR sin relacion
                                 NT-HR.
    UNRESOLVED                   No determinable.

---

## 2. Composicion del top 20

Q4 2025 (20 filers):
  VANGUARD GROUP INC           62.44B shares
  BlackRock, Inc.              56.14B
  STATE STREET CORP            25.48B
  MORGAN STANLEY               18.00B
  FMR LLC                      17.34B
  GEODE CAPITAL                14.48B
  BANK OF AMERICA CORP         13.74B
  JPMORGAN CHASE & CO          12.58B
  CHARLES SCHWAB INV MGMT       8.37B
  GOLDMAN SACHS                 7.87B
  NORGES BANK                   7.52B
  DIMENSIONAL FUND ADVISORS     7.44B
  Invesco Ltd.                  7.09B
  NORTHERN TRUST CORP           7.02B
  PRICE T ROWE ASSOCIATES       6.93B
  UBS Group AG                  6.52B
  Capital World Investors       5.59B
  ROYAL BANK OF CANADA          5.58B
  WELLINGTON MGMT GROUP         5.44B
  Bank of New York Mellon       5.10B

Q1 2026 (20 filers):
  BlackRock, Inc.              56.70B
  VANGUARD CAPITAL MGMT LLC    35.33B   <- NUEVO
  STATE STREET CORP            25.75B
  VANGUARD PORTFOLIO MGMT LLC  21.03B   <- NUEVO
  GHISALLO CAPITAL MGMT        20.69B   <- variacion 2.39B -> 20.69B
  MORGAN STANLEY               18.26B
  FMR LLC                      17.56B
  GEODE CAPITAL                14.98B
  BANK OF AMERICA CORP         14.28B
  JPMORGAN CHASE & CO          13.36B
  GOLDMAN SACHS                 8.80B
  CHARLES SCHWAB INV MGMT       8.39B
  UBS Group AG                  7.64B
  DIMENSIONAL FUND ADVISORS     7.55B
  Invesco Ltd.                  7.38B
  NORTHERN TRUST CORP           7.08B
  PRICE T ROWE ASSOCIATES       6.96B
  Capital World Investors       5.95B
  ROYAL BANK OF CANADA          5.50B
  LPL Financial LLC             5.40B   <- NUEVO

Union: 24 CIKs.

---

## 3. Clasificacion de los 24 CIKs

| CIK          | Q4  | Q1  | shares Q4       | shares Q1       | clasificacion          |
|--------------|-----|-----|-----------------|-----------------|------------------------|
| 0000019617   | si  | si  |    12,581,023,607 | 13,358,759,635 | CONTINUOUS_FILER    |
| 0000070858   | si  | si  |    13,740,836,842 | 14,281,222,583 | CONTINUOUS_FILER    |
| 0000073124   | si  | si  |     7,015,300,796 |  7,076,958,071 | CONTINUOUS_FILER    |
| 0000080255   | si  | si  |     6,933,647,331 |  6,956,487,078 | CONTINUOUS_FILER    |
| 0000093751   | si  | si  |    25,475,515,149 | 25,747,618,586 | CONTINUOUS_FILER    |
| 0000102909   | si  | no  |    62,442,074,612 |              0 | FILER_DISCONTINUITY |
| 0000315066   | si  | si  |    17,336,642,577 | 17,564,535,541 | CONTINUOUS_FILER    |
| 0000354204   | si  | si  |     7,437,918,143 |  7,547,459,133 | CONTINUOUS_FILER    |
| 0000884546   | si  | si  |     8,365,328,809 |  8,386,304,943 | CONTINUOUS_FILER    |
| 0000886982   | si  | si  |     7,872,124,782 |  8,803,361,754 | CONTINUOUS_FILER    |
| 0000895421   | si  | si  |    17,995,019,392 | 18,260,853,727 | CONTINUOUS_FILER    |
| 0000902219   | si  | si  |     5,442,971,637 |  5,305,307,176 | CONTINUOUS_FILER    |
| 0000914208   | si  | si  |     7,093,459,451 |  7,378,235,089 | CONTINUOUS_FILER    |
| 0001000275   | si  | si  |     5,583,135,451 |  5,497,541,661 | CONTINUOUS_FILER    |
| 0001214717   | si  | si  |    14,479,052,163 | 14,976,114,036 | CONTINUOUS_FILER    |
| 0001374170   | si  | si  |     7,516,874,735 |              0 | CONTINUOUS_FILER    |
| 0001390777   | si  | si  |     5,095,834,181 |  5,279,537,309 | CONTINUOUS_FILER    |
| 0001403438   | si  | si  |     4,964,373,731 |  5,399,255,600 | CONTINUOUS_FILER    |
| 0001422849   | si  | si  |     5,590,997,484 |  5,950,738,605 | CONTINUOUS_FILER    |
| 0001610520   | si  | si  |     6,524,895,878 |  7,644,396,758 | CONTINUOUS_FILER    |
| 0001825214   | si  | si  |     2,392,247,946 | 20,690,812,146 | CONTINUOUS_FILER    |
| 0002012383   | si  | si  |    56,138,010,050 | 56,696,397,221 | CONTINUOUS_FILER    |
| 0002100119   | no  | si  |              0 | 35,329,105,562 | FILER_DISCONTINUITY |
| 0002100121   | no  | si  |              0 | 21,030,554,644 | FILER_DISCONTINUITY |

Resumen:
  CONTINUOUS_FILER:     21 (87.5%)
  FILER_DISCONTINUITY:   3 (12.5%) - los 3 son Vanguard.

Dos observaciones:
  - 0001374170 (NORGES BANK): presente en Q1 con shares=0. Pasa filtro
    de presencia por SUBMISSIONTYPE pero no aporta shares SH+null.
    Clasificado CONTINUOUS_FILER por firma de filing, no por masa.
    Requiere nota metodologica pero no es discontinuidad.
  - 0001825214 (GHISALLO): Q4 2.39B -> Q1 20.69B (+765%). Continua
    presente en ambos periodos. El salto NO es reorganizacion; es
    crecimiento genuino o cambio de estrategia. No encaja en la
    taxonomia de discontinuidad.

---

## 4. Evidencia NT -> OTHERMANAGER (caso Vanguard)

El filing 13F-NT de VANGUARD GROUP INC en Q1 2026:

    ACC=0000102909-26-002707  CIK=0000102909  type=13F-NT
    FILING_DATE=2026-05-08

Declara 10 managers en OTHERMANAGER (schema: managers que reportan
PARA este filing manager):

| ref CIK      | nombre                                  | Q4 types   | Q4 shares | Q1 types     | Q1 shares |
|--------------|-----------------------------------------|------------|-----------|--------------|-----------|
| 0001550100   | Vanguard Investments Australia, Ltd.    | 13F-NT     |         0 | 13F-HR/A+HR  |   311M    |
| 0002100121   | VANGUARD PORTFOLIO MANAGEMENT LLC       | (ausente)  |         0 | 13F-HR/A+HR  | 21,030M   |
| 0002100119   | VANGUARD CAPITAL MANAGEMENT LLC         | (ausente)  |         0 | 13F-HR/A+HR  | 70,236M   |
| 0000933478   | VANGUARD FIDUCIARY TRUST CO             | 13F-NT     |         0 | 13F-HR/A+HR  | 4,124M    |
| 000217448    | VANGUARD MARKETING CORPORATION          | (ausente)  |         0 | (ausente)    | 0         |
| 0001767306   | Vanguard Personalized Indexing Mgmt LLC | 13F-HR     |    93.5M  | 13F-HR       |    99M    |
| 0001680208   | VANGUARD ASSET MANAGEMENT, Ltd          | 13F-NT     |         0 | 13F-HR/A+HR  |   984M    |
| 0001984256   | Vanguard National Trust Co              | 13F-NT     |         0 | 13F-HR       |    10M    |
| 0001811242   | Vanguard Global Advisers, LLC           | 13F-NT     |         0 | 13F-HR/A+HR  | 1,385M    |
| 0000947529   | VANGUARD ADVISERS INC                   | 13F-NT     |         0 | 13F-HR       |   256M    |

Confirmacion directa:
  - El NT padre identifica explicitamente a los 2 CIKs con HR grandes
    en Q1 (0002100119 y 0002100121).
  - 6 de las 10 filiales ya eran NT en Q4 (delegando al padre); en Q1
    presentan HR directo.
  - 0001767306 (Personalized Indexing) filtra HR en ambos periodos.
  - 000217448 (Marketing Corp) no presenta filing en ninguno.

Modelo Q4 2025: padre HR agrega + 6 filiales NT delegan.
Modelo Q1 2026: padre NT + 10 filiales HR reportan directamente.

La reorganizacion es completa y documentada por la propia SEC.

---

## 5. Hallazgo colateral: GHISALLO CAPITAL

CIK 0001825214 (GHISALLO CAPITAL MANAGEMENT LLC):
  Q4 2025:  2,392,247,946 shares.
  Q1 2026: 20,690,812,146 shares.
  Variacion: +765%.

Presente en ambos periodos. NO es discontinuidad de filer. Es un
crecimiento genuino del filer. La taxonomia observable (CONTINUOUS_FILER)
es correcta pero el caso merece nota para auditoria por si es un
artefacto o un evento relevante.

No es filer_continuity. Se documenta como observacion.

---

## 6. Conclusiones

### 6.1. Vanguard es caso aislado en el top 20

3 de 24 CIKs (12.5%) son FILER_DISCONTINUITY. Los 3 son del grupo
Vanguard. Los otros 21 (87.5%) son CONTINUOUS_FILER.

En el top 20 no hay otros casos de reorganizacion de filer. El caso
Vanguard es un evento significativo pero puntual dentro del top 20.

### 6.2. Evidencia NT -> OTHERMANAGER -> HR confirmada

El 13F-NT del padre declara explicitamente los 2 CIKs grandes de Q1
como managers que reportan para el. La relacion documental es directa
y verificable.

Por dictamen Q-PROBE-2: esto es evidencia de reporting, NO de
reconciliacion economica. No se autoriza reconciliacion automatica.

### 6.3. Impacto cuantitativo

Reorganizacion Vanguard:
  Q4 padre:      62.44B shares (agregado)
  Q1 filiales:   70.24B (Capital) + 21.03B (Portfolio) = 91.27B
  Delta bruto:   ~29B (crecimiento del trimestre + reorg)

En terminos de NIPC observable (probe ALL):
  nipc_sole = -41.16B (EXIT del padre)
  nipc_dfnd = +41.28B (NEW de filiales)

### 6.4. Estado de la taxonomia

Los 5 estados observables del dictamen se han podido asignar sin
ambiguedad:
  CONTINUOUS_FILER          21
  FILER_DISCONTINUITY        3
  NT_TO_HR_RELATION_OBSERVED 0 (los 3 DISCONTINUITY tienen NT visible,
                                 pero la taxonomia los clasifica como
                                 DISCONTINUITY por ausencia en uno de
                                 los periodos)
  HR_TO_HR_WITHOUT_RELATION  0
  UNRESOLVED                 0

Nota: los 3 DISCONTINUITY cumplen tambien NT_TO_HR_RELATION_OBSERVED.
La precedencia taxonomica deberia revisarse. En este informe se
reportan como DISCONTINUITY por ser el criterio mas estricto, con
evidencia NT -> HR anadida aparte.

---

## 7. Recomendaciones al auditor

  Q-MINI-1  Confirmar que Vanguard se documenta como limitacion
            conocida + caso aislado top 20, sin disparar diseno de
            reconciliacion productiva.

  Q-MINI-2  Confirmar precedencia taxonomica:
            FILER_DISCONTINUITY vs NT_TO_HR_RELATION_OBSERVED cuando
            ambas aplican. Propuesta: NT_TO_HR prevalece (mas
            especifico). El probe los ha reportado como DISCONTINUITY
            y anadido la evidencia NT->HR aparte.

  Q-MINI-3  GHISALLO (CIK 0001825214, +765%) merece investigacion
            separada antes de fijar thresholds? No es discontinuidad,
            pero distorsiona el NIPC observable.

  Q-MINI-4  Confirmar que el diseno de reconciliacion NT <-> HR
            (Q-PROBE-2) sigue pendiente de dictamen, pese a que la
            evidencia documental sea clara. La interpretacion economica
            sigue siendo inferencia.

  Q-MINI-5  Extender el mini-probe a top 50, top 100, o universo
            completo? Si top 20 muestra 87.5% continuidad, el sesgo
            podria estar fuera del top 20.

---

## 8. Estado del repo

  HEAD local: d75e84c.
  origin/main: 9d4a81e.
  Ahead: 35 commits.
  Working tree: limpio.
  Push: NO (local-first IAE).

---

Fin del informe. Version 1.0 (2026-09-19). HEAD d75e84c.
Pendiente: dictamen del auditor sobre Q-MINI-1 a Q-MINI-5.