# 38 - Dictamen final 5b.X v3 (FIRMADO)

**Estado:** FIRMADO / FROZEN. 2026-10-03.
**Precedente:** `37_d06_5bX_invalidada.md` (D-06) + `35_protocolo_5bX_v3.md`.
**Paquete firmado:**

- `35_protocolo_5bX_v3.md` (contrato vigente).
- `scripts/validate_wyckoff_sow_5bX.py` (script v3).
- `tests/test_wyckoff_sow_5bX_contract.py` (19 tests).
- `35_protocolo_5bX_v3.freeze.json` (manifiesto).

---

## 1. Resultado

    revision_auditor = FIRMADO
    estado           = FROZEN

C1 (equivalencia bootstrap) y C2 (guardrails no bloqueantes)
cerradas. No hay condicion normativa pendiente que impida el freeze.

---

## 2. Configuracion congelada

    N = 60
    M = 30
    X_ATR = 0.25
    Y_VOL = 1.10

    OOS: t0 >= 2026-10-02

    MIN_N_CONFIRMED_H20 = 550

    B_bootstrap = 2000
    seed_bootstrap = 20261002

    A: lower_CI95 > 0
    B: lower_CI95 <= 0 <= upper_CI95
    C: upper_CI95 < 0
    D: insufficient sample

---

## 3. Prohibiciones vigentes

    recalibracion
    grid
    reseleccion de candidata
    cambio de cutoff
    cambio de 550
    cambio del criterio de exito
    uso confirmatorio del bloque historico D
    activacion de SOW
    migracion de consumidores
    apertura de 5c.4 / 5d

---

## 4. Estado formal recomendado por el auditor

    5b.X original      INVALIDADA COMO EJECUTABLE
    5b.X v3            FIRMADO / FROZEN
    C1 bootstrap       CERRADA
    C2 guardrails      CERRADA
    550                CONGELADO
    Config SOW         None
    Fail-closed        INTACTO
    Validacion OOS     BLOQUEADA POR DATOS

---

## 5. Limites de la firma (nota de honestidad del auditor)

El auditor declara expresamente en su dictamen:

> **No he inspeccionado fisicamente el diff `d324346..d0874f4`; por
> tanto, no voy a afirmar que lo he revisado linea por linea.**

> Mi firma se emite **sobre el paquete contractual y la evidencia
> presentada**, no como certificacion de una inspeccion visual
> independiente del diff que no he realizado.

Esta limitacion debe quedar registrada para que futuros lectores no
asuman una revision visual linea por linea que no ocurrio. La firma
se apoya en:

- mapeo explicito de los 6 cambios normativos autorizados;
- tests contractuales (19 en el fichero);
- C1 bootstrap: equivalencia funcional demostrada por test;
- C2 guardrails: no bloqueantes cuando hay datos OOS;
- freeze criptografico (sha256 script/protocolo/tests).

---

## 6. Firma de auditoria (literal)

> **DICTAMEN: CONFORME - FIRMADO.**
> El paquete 5b.X v3 puede quedar formalmente congelado. No requiere
> una nueva revision previa a la ejecucion por cambios de diseno,
> puesto que las condiciones C1 y C2 han sido cerradas y los criterios
> normativos ya estan fijados ex ante. La futura ejecucion debera
> utilizar exclusivamente el protocolo, script, tests y hashes
> incluidos en el freeze.

---

## 7. Proximo evento valido

El siguiente evento valido **no es una modificacion metodologica**.
Es la llegada de muestra OOS suficiente:

    n_confirmed_H20_complete >= 550

y la ejecucion unica del paquete congelado.

---

**Fin del dictamen final.**
