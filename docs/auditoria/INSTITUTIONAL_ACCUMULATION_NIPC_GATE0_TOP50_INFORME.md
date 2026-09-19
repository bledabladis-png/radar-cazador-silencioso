# INFORME TOP 50 FILER CONTINUITY - Q-MINI-5

Version: 1.0 (2026-09-19)
HEAD revisado: 37d2879
Estado: TOP 50 completado. Vanguard confirmado como concentracion unica.
Autor: Ingeniero Supervisor.
Naturaleza: no normativo. Si hay conflicto con el Prompt Maestro, gana el
            Prompt Maestro.

---

## 1. Proposito

Extender el mini-probe de filer continuity del top 20 al top 50 filers
por sum(SSHPRNAMT), segun dictamen mini-probe (Q-MINI-5).

Metodologia identica al mini-probe top 20:
  - Filtro SH + PUTCALL NULL.
  - canonical_snapshot Q4 2025 + Q1 2026 via apply_amendments (FA-2.4).
  - Union de CIKs presentes en ambos periodos.
  - Clasificacion filer_status (Q-MINI-2): CONTINUOUS_FILER |
    FILER_DISCONTINUITY | UNRESOLVED.
  - NT_to_hr_relation_observed + targets.

Correccion metodologica incorporada (dictamen mini-probe, seccion 7):
TODOS los agregados SSHPRNAMT se calculan sobre canonical_snapshot,
nunca sobre INFOTABLE.parquet crudo. Esto evita doble conteo cuando
existen amendments RESTATEMENT.

---

## 2. Composicion top 50

Q4 2025 (top 50): rango SSHPRNAMT de 62.44B (Vanguard padre) a
2.03B (Principal Financial Group).

Q1 2026 (top 50): rango SSHPRNAMT de 56.70B (BlackRock) a
2.41B (Van Eck Associates).

Union de CIKs: 54 (hay 46 CIKs en el top 50 de ambos periodos,
4 CIKs entran/salen del top 50 sin ser discontinuidad — solo
desplazamiento por cambios de shares).

---

## 3. Clasificacion filer_status (54 CIKs)

    CONTINUOUS_FILER:     50 (92.6%)
    FILER_DISCONTINUITY:   4 (7.4%)
    UNRESOLVED:            0

pct_discontinuity = 0.0741.

Comparativa con top 20 (probe previo):
    top 20: 3/24 = 12.5%
    top 50: 4/54 =  7.4%

La discontinuidad baja al ampliar muestra. No aparecen casos nuevos
fuera del grupo Vanguard.

---

## 4. FILER_DISCONTINUITY - detalle

Los 4 casos son todos del grupo Vanguard:

### 4.1. CIK 0000102909 VANGUARD GROUP INC (padre)

    Q4 2025: presente  shares=62,442,074,612
    Q1 2026: ausente
    nt_to_hr_observed = True
    nt_targets (10 managers):
      0001550100 Vanguard Investments Australia
      0002100121 VANGUARD PORTFOLIO MANAGEMENT LLC
      0002100119 VANGUARD CAPITAL MANAGEMENT LLC
      0000933478 VANGUARD FIDUCIARY TRUST CO
      0000217448 VANGUARD MARKETING CORPORATION
      0001767306 Vanguard Personalized Indexing Management
      0001680208 VANGUARD ASSET MANAGEMENT, Ltd
      0001984256 Vanguard National Trust Co
      0001811242 Vanguard Global Advisers, LLC
      0000947529 VANGUARD ADVISERS INC

### 4.2. CIK 0000933478 VANGUARD FIDUCIARY TRUST CO

    Q4 2025: ausente
    Q1 2026: presente  shares=4,124,421,637
    nt_to_hr_observed = True
    nt_targets: ['0000102909'] (padre)

Nota: en Q4 era NT que delegaba al padre. En Q1 presenta HR directo.

### 4.3. CIK 0002100119 VANGUARD CAPITAL MANAGEMENT LLC

    Q4 2025: ausente
    Q1 2026: presente  shares=35,329,105,562
    nt_to_hr_observed = False (no presenta NT propio)
    target: no aplica

### 4.4. CIK 0002100121 VANGUARD PORTFOLIO MANAGEMENT LLC

    Q4 2025: ausente
    Q1 2026: presente  shares=21,030,554,644
    nt_to_hr_observed = False
    target: no aplica

---

## 5. NT_TO_HR observations (2 casos)

### 5.1. CIK 0000102909 - NT Q1 2026

    ACC = 0000102909-26-002707
    SUBMISSIONTYPE = 13F-NT
    FILING_DATE = 2026-05-08
    OTHERMANAGER declara 10 managers (ver seccion 4.1).

### 5.2. CIK 0000933478 - NT Q4 2025

    VANGUARD FIDUCIARY TRUST CO era NT en Q4 delegando al padre
    (target: 0000102909). En Q1 presenta 13F-HR directo.

---

## 6. Observaciones

### 6.1. Vanguard es el unico grupo con discontinuidad

En los 54 CIKs del top 50, solo 4 son FILER_DISCONTINUITY. Los 4
pertenecen a la estructura de filing de Vanguard. Ni BlackRock,
State Street, Fidelity, ni ningun otro gran gestor muestra el mismo
patron en este trimestre.

### 6.2. Continuidad estructural

Los 50 CIKs continuos mantienen el mismo filer CIK con filings HR en
ambos periodos. La composicion del top 50 es estable.

### 6.3. Caso NORGES BANK (CIK 0001374170)

Presente en ambos periodos (Q4_sh=7.52B, Q1_sh=0). El filing de Q1
existe pero no aporta shares SH+null al snapshot canonico. Clasificado
CONTINUOUS_FILER por firma de filing, no por masa. No es
discontinuidad de filer. Requiere nota metodologica (ya anotado en
informe mini-probe top 20).

### 6.4. GHISALLO CAPITAL (CIK 0001825214)

Continua presente en ambos periodos:
    Q4 2025:  2,392,247,946
    Q1 2026: 20,690,812,146
    Delta:   +765%

Pendiente probe especifico segun dictamen (Q-MINI-3). No es
discontinuidad de filer.

---

## 7. Sobre la tasa de discontinuidad

    top 20: 3/24 = 12.5%
    top 50: 4/54 =  7.4%

Esta diferencia NO debe interpretarse como "la discontinuidad baja al
ampliar muestra". Debe interpretarse como: Vanguard (que ocupa 2 filas
en top 20: padre + 1 filial, y 4 filas en top 50: padre + 3 filiales)
es la unica fuente de discontinuidad. Al aumentar el denominador con
50 CIKs que no son Vanguard, el porcentaje baja naturalmente.

El auditor ya advirtio (dictamen previo): no usar estos porcentajes
como umbral. Solo descripcion de la muestra.

---

## 8. Conclusion

### 8.1. Vanguard es el unico caso en top 50

La extension del probe a top 50 no ha revelado ningun caso adicional
de discontinuidad de filer. La concentracion observada en top 20 se
mantiene y se intensifica: no hay patrones alternativos.

### 8.2. Evidencia NT → HR documentada

Se han observado 2 NT_TO_HR: padre Vanguard Q1 y Fiduciary Trust Q4.
Los targets declarados en OTHERMANAGER confirman la estructura.

### 8.3. Estado del ciclo sin cambios estructurales

  C2: VIGENTE / SIN PATCH.
  Reconciliacion NT <-> HR: NO AUTORIZADA.
  Thresholds: UNDEFINED.
  Gate-NIPC.2: BLOQUEADO.
  Gate-NIPC.3: NO AUTORIZADO.

---

## 9. Recomendaciones

  Q-T50-1  Confirmar que top 50 cierra el ciclo de mini-probes filer
           continuity. ¿Se autoriza extension a top 100 o universo
           completo, o el resultado es suficiente?

  Q-T50-2  Sobre la metodologia de filer_status: considerar "presente
           por SUBMISSIONTYPE" vs "presente por SSHPRNAMT > 0". El
           caso NORGES BANK demuestra la diferencia. ¿Cual prevalece
           como criterio operativo?

  Q-T50-3  Sobre la observacion de Fiduciary Trust (Q4 NT -> Q1 HR,
           target padre): la relacion NT_to_hr observable es clara
           pero la reconciliacion economica sigue sin autorizar. ¿Se
           mantiene la prohibicion de reconciliacion?

  Q-T50-4  GHISALLO sigue pendiente de probe especifico (Q-MINI-3).
           ¿Se autoriza ahora con el resto del contexto top 50?

  Q-T50-5  Cierre del ciclo mini-probes: si top 50 es suficiente,
           ¿cual es el siguiente paso autorizado? ¿Coverage OpenFIGI
           masivo? ¿Diseño de reconciliacion NT-HR como investigacion
           acotada?

---

## 10. Estado del repo

  HEAD local: 37d2879.
  origin/main: 9d4a81e.
  Ahead: 39 commits.
  Working tree: limpio.
  Push: NO (local-first IAE).

---

Fin del informe. Version 1.0 (2026-09-19). HEAD 37d2879.
Pendiente: dictamen del auditor sobre Q-T50-1 a Q-T50-5.