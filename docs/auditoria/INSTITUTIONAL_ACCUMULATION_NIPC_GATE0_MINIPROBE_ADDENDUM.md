# ADDENDUM CORRECCION CUANTITATIVA - Informe mini-probe Q-PROBE-5

Version: 1.0 (2026-09-19)
Informe corregido: INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_MINIPROBE_INFORME.md
Dictamen habilitante: INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_MINIPROBE_DICTAMEN.md
                    (hallazgo cuantitativo obligatorio, seccion 7)
HEAD al redactar: ba1e6d1
Naturaleza: no normativo. Si hay conflicto con el Prompt Maestro, gana el
            Prompt Maestro.

---

## 1. Motivo

El auditor identifico una inconsistencia cuantitativa en el informe
mini-probe Q-PROBE-5 (seccion 7 del dictamen):

    Top 20 Q1:        CIK 0002100119 = 35.329B shares
    Seccion 4 Vanguard: CIK 0002100119 = 70.236B shares
    Cifra filiales Q1:  70.236B + 21.031B = 91.267B

Las dos cifras del mismo CIK no eran compatibles. El auditor ordeno
reconciliar contra el canonical_snapshot antes de reutilizar la cifra.

---

## 2. Diagnostico del error

El error del informe original fue sumar SSHPRNAMT desde
INFOTABLE.parquet FILTRADO POR PERIODO (todos los filings del rango),
en lugar de sumar desde el CANONICAL_SNAPSHOT producido por FA-2.4
(apply_amendments).

Esto viola la regla de canonicalizacion de amendments: el snapshot
canonico aplica RESTATEMENT -> REPLACE y NEW HOLDINGS -> ADD, y solo
los filings "applied" contribuyen al conteo. Sumar INFOTABLE crudo
duplica el HR original cuando existe un RESTATEMENT posterior.

Evidencia directa del probe de reconciliacion
(evidence/nipc_gate0_probe/probe_reconcile_vanguard.py):

  CIK 0002100119 (Vanguard Capital Management LLC):

    ACC=0002100119-26-001306  13F-HR        shares=34,906,823,405  SUPERSEDED
    ACC=0002100119-26-001311  13F-HR/A      shares=34,906,823,405  APPLIED (RESTATEMENT, AN=1)
    ACC=0002100119-26-001313  13F-HR/A      shares=   422,282,157  APPLIED (NEW HOLDINGS, AN=2)

    Suma raw (3 acc):     70,235,928,967
    Suma canonical (2):   35,329,105,562
    Ratio:                1.9880

El HR original fue SUPERSEDED por el RESTATEMENT posterior. La
canonicalizacion lo excluye correctamente. El raw lo contaba dos veces.

  CIK 0002100121 (Vanguard Portfolio Management LLC):

    ACC=0002100121-26-000861  13F-HR        shares=21,030,554,644  APPLIED
    ACC=0002100121-26-000865  13F-HR/A      shares=        0       APPLIED (NEW HOLDINGS, AN=1)

    Suma raw:         21,030,554,644
    Suma canonical:   21,030,554,644
    Ratio:            1.0000 (sin doble conteo)

---

## 3. Cifras corregidas

### 3.1. CIK 0002100119 (Vanguard Capital) Q1 2026

    Antes (informe mini-probe, incorrecto):   70,235,928,967
    Ahora (canonical_snapshot, correcto):     35,329,105,562
    Diferencia:                               34,906,823,405 (exactamente
                                              el HR superseded)

### 3.2. CIK 0002100121 (Vanguard Portfolio) Q1 2026

    Antes y ahora:                            21,030,554,644 (sin cambio)

### 3.3. Agregado filiales Q1 2026

    Antes (informe mini-probe, incorrecto):   91,267,483,611
    Ahora (canonical_snapshot, correcto):     56,359,660,206

### 3.4. Comparativa padre Q4 vs filiales Q1

    Padre Vanguard Q4 2025:                   62,442,074,612
    Filiales Vanguard Q1 2026 (canonical):    56,359,660,206
    Delta:                                    -6,082,414,406 (-9.74%)

    Antes el informe implicaba un crecimiento de ~+29B.
    La cifra correcta muestra una contraccion de -6.08B (-9.74%).

    Interpretacion: el cambio de modelo de filing (padre HR -> padre NT +
    filiales HR) se acompana de una ligera reduccion de la masa reportada.
    La reorganizacion administrativa no genero crecimiento ficticio:
    los ~82B brutos de cancelacion (nipc_sole + nipc_dfnd) reflejan
    reasignacion entre CIKs, no variacion economica.

---

## 4. Impacto en el NIPC observable

Las cifras del probe end-to-end (informe anterior) NO se ven afectadas
por este error, porque el motor NIPC usa el canonical_snapshot via
apply_amendments antes de calcular delta_shares y nipc.

Cifras confirmadas:

    NIPC observable scope ALL:      +23,651,586
    NIPC observable scope ELIGIBLE: +52,570,648

    match_status ALL:
      UNRESOLVED_IDENTITY   3,177,329
      BOTH                    698,245
      NEW                     113,710
      EXIT                     93,043

    nipc_sole:  -41,161,904,992
    nipc_dfnd:  +41,246,135,898
    nipc_otr:       -60,579,320

Estas cifras son correctas porque se calcularon sobre el
canonical_snapshot.

El error fue local: solo afecto a la seccion 4 y 5.1 del informe
mini-probe, donde se sumo SSHPRNAMT raw para describir el caso Vanguard
a nivel agregado. El motor de calculo NIPC siempre opero sobre
canonical.

---

## 5. Leccion metodologica

Regla congelada para futuros informes/probes:

    Todo agregado de SSHPRNAMT debe calcularse sobre el
    canonical_snapshot producido por apply_amendments.

    PROHIBIDO sumar SSHPRNAMT desde INFOTABLE.parquet filtrado por
    periodo cuando existan amendments RESTATEMENT. Duplica el HR
    original.

Excepcion: analisis de filings individuales (lineage, deteccion de
supersedings). Ahi si se recorre raw, pero nunca se presenta como
agregado.

---

## 6. Acciones correctivas

  1. Informe mini-probe original NO se modifica (historial
     preservado). Esta addendum lo corrige.

  2. Probe de reconciliacion anadido a evidence:
     docs/auditoria/evidence/nipc_gate0_probe/probe_reconcile_vanguard.py

  3. HASHES.txt regenerado.

  4. Regla metodologica incorporada al prompt maestro (v6.36 pendiente).

  5. Cifra 91.267B queda marcada como NO UTILIZABLE. La cifra correcta
     es 56.360B.

---

## 7. Estado del ciclo

  Dictamen mini-probe: GO CONDICIONADO.
  Correccion cuantitativa: APLICADA.
  C2: VIGENTE / SIN PATCH.
  Reconciliacion NT <-> HR: NO AUTORIZADA.
  Thresholds: UNDEFINED.
  Gate-NIPC.2: BLOQUEADO.
  Gate-NIPC.3: NO AUTORIZADO.

  Siguiente accion autorizada (dictamen seccion 12):
    (a) Reconciliacion cuantitativa Vanguard [COMPLETADA por este
        addendum].
    (b) TOP 50 filer continuity probe [PENDIENTE].

---

Fin del addendum. Version 1.0 (2026-09-19). HEAD al redactar: ba1e6d1.