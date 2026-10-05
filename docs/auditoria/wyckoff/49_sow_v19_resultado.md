# 49 - SOW v1.9 - Resultado del protocolo v3 (FAIL confirmatorio)

Fecha: 2026-10-05
Rama: sow-v4-validation, HEAD a505b87
Protocolo: SOW_v19_PROTOCOL.json v3 (FIRMADO)
SHA-256 protocolo: c6dae718a258ac47e4c75326ec5278397a97f116a56f5070a22f8c8778f3a1f2
Artefacto: outputs/audit/sow_v19/bootstrap_v2fix_B20000.json

---

## 1. Dictamen

FAIL confirmatorio en desarrollo.

La ejecucion del protocolo SOW v1.9 v3, congelado y firmado, evaluo las 240
configuraciones predefinidas sobre el universo H20-complete, aplicando:

- anti-solapamiento obligatorio (min_gap_sessions=50),
- panel calendar block bootstrap (block_len=120, burn_in=200),
- recomputacion del pipeline por panel,
- MDE=5 pp,
- correccion de multiplicidad mediante fixed-SE studentized maxT.

Ninguna de las 240 configuraciones alcanzo `p_maxT <= 0.05`. El menor valor
observado fue 0.814009. La candidata historica v1.9 obtuvo p_maxT = 0.916154.

En consecuencia, **no existe evidencia confirmatoria suficiente para
seleccionar una configuracion SOW con efecto minimo de 5 pp en el conjunto
de desarrollo**.

---

## 2. Precision sobre la interpretacion

El protocolo NO prueba matematicamente que `theta_j <= 5 pp`. La formulacion
correcta es:

    NO EVIDENCIA SUFICIENTE DE EFECTO >= 5 pp

y no:

    PRUEBA DE AUSENCIA DE EFECTO >= 5 pp

Un `p_maxT > 0.05` es un fallo en rechazar H0, no una prueba de H0. La
distincion importa para la trazabilidad del expediente.

---

## 3. Ejecucion

    Comando:   py scripts/sow_v19_bootstrap_v2.py --phase 20000 --workers 12
    Exit:      0
    Tiempo:    188.9 s
    Semilla:   20261005
    B:         20000
    block_len: 120
    burn_in:   200
    H:         20
    min_gap:   50

Precomputacion: 315 tickers, 1297 sesiones. SOW 34.4 s. Episodios 57.6 s
(240 combos, 1255-1271 episodios por combo). Bootstrap 188.9 s.

---

## 4. Resultado agregado

    p_maxT minimo     = 0.814009
    p_maxT maximo     = 1.000000
    p_maxT medio      = 0.961326
    p_maxT <= 0.05    = 0/240
    p_maxT <= 0.10    = 0/240
    p_maxT <= 0.50    = 0/240

Resolucion del test: 1/(B+1) = 4.9998e-05.

---

## 5. Top 5 configuraciones por p_maxT ascendente

| N  | M  | X_ATR | Y_VOL | lift   | SE_j   | T_obs  | p_maxT   |
|----|----|-------|-------|--------|--------|--------|----------|
| 60 | 5  | 0.75  | 1.20  | +0.0935 | 0.0633 | +0.6872 | 0.814009 |
| 20 | 10 | 1.00  | 1.50  | +0.1022 | 0.0775 | +0.6731 | 0.819009 |
| 60 | 5  | 0.75  | 1.10  | +0.0910 | 0.0621 | +0.6592 | 0.823509 |
| 40 | 15 | 0.25  | 1.20  | +0.0919 | 0.0658 | +0.6370 | 0.830708 |
| 60 | 15 | 0.25  | 1.20  | +0.0890 | 0.0636 | +0.6133 | 0.838708 |

El mejor `T_obs` = 0.6872. Bajo el nulo, el maximo de 240 estadisticos
studentizados tiene mediana aproximada ~2.7. Ninguna configuracion alcanza
la mediana del maximo nulo.

---

## 6. Candidata congelada v1.9

    N=60, M=30, X_ATR=0.25, Y_VOL=1.10

    lift      = +0.0692
    SE_j      = 0.0576
    T_obs     = +0.3327
    p_maxT    = 0.916154

---

## 7. Nota metodologica sobre maxT

El procedimiento maxT proporciona control FWER bajo los supuestos necesarios
del procedimiento (adecuacion del remuestreo y estructura de dependencia).
Las garantias fuertes de procedimientos Westfall-Young dependen de supuestos
como subset pivotality o condiciones asintoticas apropiadas.

No es un bloqueo adicional, es una nota de alcance. El resultado observado
(0/240 con p_minimo=0.814) esta suficientemente alejado del umbral como
para que la conclusion no dependa de este detalle.

---

## 8. Trayectoria de la evidencia

    Analisis inicial:      +7.40 pp, IC95% positivo, p individual ~0.001
                           |
    Anti-solape:           +6.92 pp
                           |
    MDE 5 pp:              p_vs_mde = 0.17
                           |
    Dependencia + maxT:    p_maxT = 0.916 (candidata v1.9)
                           p_maxT = 0.814 (mejor de 240)

El efecto puntual positivo no desaparece. Lo que desaparece es la evidencia
estadistica suficiente para afirmar que el efecto es real y material frente
a la busqueda de 240 configuraciones.

Formulacion correcta: "La senal descriptiva observada no sobrevive al
protocolo confirmatorio."

---

## 9. Estado final

    Detector wyckoff_v1.py      SIN MODIFICAR
    Config SOW                  None (fail-closed)
    Legacy wyckoff.py           intacto
    Consumidores                5 directos, sin migrar
    Produccion                  BLOQUEADA
    5b.X (2027)                 Unico experimento confirmatorio pendiente

---

## 10. Prohibiciones vigentes

- No modificar MDE (5 pp congelado).
- No ampliar ni reducir el grid 2021-2026.
- No buscar parametros alternativos sobre el mismo desarrollo.
- No modificar la formula del detector para "mejorar" el resultado.
- No activar SOW en produccion con los datos actuales.
- No migrar consumidores.

---

## 11. Proximo paso

Ejecucion OOS 5b.X sobre datos posteriores al cutoff 2026-10-01, con la
candidata congelada v1.9. Requiere `n_confirmed_H20_complete >= 550`.
Estimacion: ~2027. Protocolo: 35_protocolo_5bX_v3.md (FIRMADO).

Los resultados anuales (2020, 2025, 2026 positivos) quedan clasificados
como analisis exploratorio generador de hipotesis. No tienen peso decisorio.
No se utilizan para rescatar v1.9.

---

## 12. Artefactos

- scripts/sow_v19_bootstrap_v2.py (commit a505b87)
- outputs/audit/sow_v19/bootstrap_v2fix_B20000.json
- outputs/audit/sow_v19/_bootstrap_v2fix_B20000.log
- docs/auditoria/wyckoff/SOW_v19_PROTOCOL.json (v3, hash c6dae718...)
- docs/auditoria/wyckoff/SOW_v19_PROTOCOL.freeze.json (FIRMADO)

---

**Fin del dictamen.**
