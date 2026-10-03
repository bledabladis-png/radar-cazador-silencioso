# 37 - D-06: validador 5b.X v1 congelado contra protocolo v2

**Estado:** INVALIDACION REGISTRADA. 5b.X original NO EJECUTABLE.
**Fecha:** 2026-10-03.
**Dictamen externo:** 2026-10-03 (extractos literales en seccion 4).
**Afecta a:** `scripts/validate_wyckoff_sow_5bX.py`, `35_protocolo_5bX.md`,
`03_plan_migracion.md`.
**NO afecta a:** candidata v1.9, contrato v1.9, resultados historicos de
5b.3 / 5b.4 / 5b.4-bis.

---

## 1. Resumen

Al preparar los tests contractuales de 5b.X (frente P7 del handoff del
2026-10-03) se detecto que `scripts/validate_wyckoff_sow_5bX.py` implementa
la version v1 del diseno (umbrales 12m / 50ep / 20conf; inclusion por
`L > cutoff`; escenarios por `lift_point`). El protocolo
`35_protocolo_5bX.md` (commit `9ef7497`, v2) exige otra cosa: 550 confirmed
H20-complete, inclusion por `t0 >= cutoff`, escenarios por IC95%. El commit
`ccba3ab` congelo el sha256 del script v1 bajo mensaje "sincronizar plan v2".

El binomio protocolo v2 + script v1 no constituye un paquete ejecutable
valido. 5b.X original queda invalidada como unidad contractual. Se abre
5b.X-bis / v3. La candidata v1.9 y los resultados historicos no se tocan.

---

## 2. Desalineaciones

### D-06.1 - Umbral de muestra

- Protocolo §2.1: `n_confirmed_H20_complete >= 550`.
- Script v1: `MIN_N_CONFIRMED = 20` (linea 63, chequeo en linea 276).
- El valor 550 no aparece en el script.

### D-06.2 - Criterio OOS

- Protocolo §3.1: OOS si y solo si `t0 >= 2026-10-02`. Explicito: "No:
  `L >= 2026-10-02`".
- Script v1: linea 206 `if e["L_date"] <= cutoff: continue`.
- Diferencia material: un episodio iniciado antes del cutoff con landmark
  despues participa parcialmente en el historico de desarrollo. v1 lo
  cuenta; v2 lo excluye.

### D-06.3 - Precondiciones vigentes

- Protocolo changelog §13 (linea 428): el umbral 550 reemplaza
  `12m + 50ep + 20conf`.
- Script v1: `MIN_MONTHS=12`, `MIN_N_EPISODES=50`, `MIN_N_CONFIRMED=20`
  (lineas 61-63), las tres activas en `check_preconditions()`.

### D-06.4 - Escenarios por lift_point vs IC

- Protocolo §6.2: `lower_CI > 0 -> A`; `IC cruza 0 -> B`;
  `upper_CI < 0 -> C`.
- Script v1: lineas 190-196. `lower > 0 -> A`; `lift_point > 0 -> B`;
  `else -> C`. Clasifica por `lift_point`, no por IC.
- Divergen cuando `lift_point` y el signo del IC apuntan en direcciones
  opuestas. Ejemplo: `lift = -0.5, IC = [-1.2, +0.2]`. Protocolo: B.
  Script: C.

### D-06.5 - Elegibilidad H20_complete

- Protocolo §2.1: `confirmed` elegible exige poder calcular
  `struct_deterioration_H20`, es decir `L + 20 < len(dates)` (o
  equivalente: `t0 + M + H20 = t0 + 50` sesiones).
- Script v1: `check_preconditions` cuenta episodios con `L > cutoff`
  (lineas 96-107). No verifica `L + 20 < len(dates)`.
- Consecuencia: el umbral de muestra se cuenta sobre una poblacion que
  no es la declarada. Episodios cercanos al fin del dataset entrarian
  como confirmed estando censurados para H20.

---

## 3. Historia de la desalineacion

    d324346   feat(wyckoff): script validate 5b.X (out-of-sample, bloqueado
              hasta datos). Implementa v1 (20 conf, L > cutoff, 12m+50ep,
              escenarios por lift).

    9ef7497   docs(wyckoff): protocolo 5b.X v2 (dictamen externo + umbral
              550 congelado). El script no se adapta.

    ccba3ab   docs(wyckoff): congelar hash validate_5bX + sincronizar
              plan v2. Congela sha256 del script v1 bajo mensaje de
              sincronizacion v2. Autocontradictorio.

El script v1 no se ha modificado desde `d324346`. El sha256
`CC01A6ABE95E57D7810D77B94AE9E5F85181274403D439E4C8D4EB706AD90D39`
coincide con el registrado en `35_protocolo_5bX.md` §2.5.

---

## 4. Dictamen externo (extractos literales, 2026-10-03)

> El binomio `35_protocolo_5bX.md @ 9ef7497 + validate_wyckoff_sow_5bX.py
> @ sha256 CC01...` **NO constituye un paquete ejecutable valido de 5b.X**.
> El hash congelado certifica un ejecutable que **no implementa el contrato
> normativo contra el que deberia ejecutarse**.

Severidades: D-06.1 a D-06.4 todas CRITICAS. D-06.5 anadida por el auditor
en el mismo dictamen (elegibilidad H20).

Consultas:

- **5b.X-bis no es mero tramite interno.** Afecta al artefacto que queda
  formalmente congelado para producir el resultado OOS.
- **Revision del auditor: si, pero de conformidad**, no reapertura de la
  decision 550 / candidata.
- **Notificacion: ahora.** Registrar la invalidacion antes de la ventana
  OOS.

Lo que **no** se reabre: N=60, M=30, X_ATR=0.25, Y_VOL=1.10, umbral 550,
criterio `t0 >= 2026-10-02`, criterio primario por IC95%, ausencia de grid
y recalibracion.

Lo que **si** debe auditarse antes del nuevo freeze:

- que el nuevo ejecutable implemente el contrato, y
- que **no haya cambios colaterales** en: landmark, bloques, definicion
  SOW, bootstrap ticker-cluster, seed, censura, calculo del lift,
  outcomes.

Regla textual del auditor:

> **el nuevo script es v2 unicamente respecto de los cambios normativos
> aprobados; todo lo demas permanece bit-equivalente o funcionalmente
> equivalente a la implementacion auditada de 5b.4-bis.**

Sobre el hash `ccba3ab`: **no borrar ni sobrescribir**. Debe quedar en la
historia del repositorio como freeze historico no valido como paquete
ejecutable.

Sobre nomenclatura: preferencia del auditor por
`35_protocolo_5bX_v3.md` con cabecera "REEMPLAZA OPERATIVAMENTE A
35_protocolo_5bX.md". El script puede mantener nombre fisico, pero commit
y hash deben cambiar. Registrar `script_sha256 + protocol_sha256 +
git_commit` en un manifiesto de freeze.

---

## 5. Secuencia aprobada por el auditor (11 pasos)

    1.  Registrar D-06.1 ... D-06.5.
    2.  Declarar 5b.X original NO EJECUTABLE.
    3.  Redactar 5b.X v3 / bis.
    4.  Implementar el script conforme al v2/v3.
    5.  Anadir tests contractuales.
    6.  Verificar: 550, t0, H20_complete, escenarios por IC, bootstrap,
        seed, ausencia de grid.
    7.  Ejecutar tests completos.
    8.  Obtener nuevo SHA-256.
    9.  Emitir freeze contractual.
    10. Auditor revisa el diff y certifica conformidad.
    11. Permanecer bloqueado hasta disponer de datos OOS suficientes.
    12. Ejecutar una unica vez.

No aprobado: `script nuevo -> hash nuevo -> esperar a 2027 -> ejecutar
sin revision`.

---

## 6. Estado tras el dictamen

    5b.X original                 INVALIDADA / NO EJECUTABLE
    ccba3ab                       FREEZE HISTORICO, NO VALIDO
                                  COMO PAQUETE EJECUTABLE
    550                           CONGELADO
    t0 >= 2026-10-02              CONGELADO
    H20_complete                  OBLIGATORIO
    Escenarios                    BASADOS EN IC95%
    5b.X-bis / v3                 REQUERIDO
    Nuevo script                  REQUERIDO
    Nuevo SHA-256                 REQUERIDO
    Revision del auditor          SI, antes del freeze final
    Nueva seleccion/recalibracion NO
    Revision de N/M/X/Y           NO
    Ejecucion con script v1       PROHIBIDA
    Candidata v1.9                SIN CAMBIOS
    Contrato v1.9                 SIN CAMBIOS
    Resultados 5b.3/5b.4/5b.4-bis SIN CAMBIOS

---

## 7. Fase B - checklist tecnica (no abierta aun)

Requiere contrato de patch firmado y cabeza fresca. Sin abrir el
2026-10-03.

Cambios normativos exigidos en el script:

- `MIN_N_CONFIRMED = 550` (o variable dedicada `MIN_N_CONFIRMED_H20`).
- Inclusion OOS: `t0 >= 2026-10-02` (reemplaza `L_date > cutoff`).
- Retirar `MIN_MONTHS` y `MIN_N_EPISODES` como bloqueantes. Si se
  conservan, solo como guardrail informativo.
- `classify_scenario` por IC: `lower > 0 -> A`; `IC cruza 0 -> B`;
  `upper < 0 -> C`.
- Anadir escenario `D = INSUFFICIENT_SAMPLE` antes de bootstrap, si
  `n_confirmed_H20 < 550`.
- `H20_complete`: verificar `L_idx + 20 < len(dates)` antes de contar.
- Incluir `n_H20_complete` en el summary JSON.

Sin tocar: landmark, bloques, bootstrap, seed, censura, calculo del lift,
outcomes, candidata N/M/X/Y.

Tests contractuales exigidos:

- Rechazo con `t0 < cutoff`.
- Bloqueo con `n_confirmed < 550` (escenario D).
- Escenarios A/B/C por IC (forzar los tres).
- Elegibilidad H20: episodio con `L + 20 >= len(dates)` no cuenta como
  confirmed.
- Verificacion del hash pre-ejecucion (si el hash difiere, abortar).

---

## 8. Que NO se ha hecho el 2026-10-03

- No se ha modificado `scripts/validate_wyckoff_sow_5bX.py`.
- No se ha re-congelado el hash `CC01A6AB...`.
- No se han escrito tests contractuales (serian verdes certificando v1).
- No se han modificado parametros ni umbrales.
- No se ha alterado `config/settings.py` (SOW sigue a `None`).

---

**Fin del informe 37.**
