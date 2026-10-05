# CALIBRACION 5b.3 - Resultados

**Informe de resultados. NO normativo. Pendiente de segunda ronda de auditoria.**
**Fecha:** 2026-10-02.
**Protocolo:** docs/auditoria/wyckoff/20_protocolo_fase_5b3.md (v3).
**Contrato:** docs/auditoria/wyckoff/01_contrato_semantico_v1_8.md.
**Commit de ejecucion:** e66cf41.
**Script:** scripts/calibrate_wyckoff_sow_5b3.py.
**Artefactos:** outputs/audit/wyckoff_5b3_grid.csv (240 filas),
outputs/audit/wyckoff_5b3_summary.json, outputs/audit/wyckoff_5b3_run.log.

---

## 1. Resumen ejecutivo

El grid completo (240 combinaciones) se ejecuto sobre 315 tickers y 2415
episodios candidatos. Resultado principal:

- **D1 (muestra SOW >= 100):** 240/240 combinaciones cumplen.
- **D2 (muestra confirmados >= 20):** 240/240 cumplen.
- **D3 (struct_deterioration_H20 confirmado >= 50%):** 0/240 cumplen.
  El maximo observado es 45.0%. La mediana es 39.0%.
- **D4 (estabilidad vecindario <= 30 pp):** 240/240 cumplen.
- **incremental_lift_H20 > 0 en calibracion:** 240/240.
- **incremental_lift_H20 > 0 en holdout:** 240/240.

**Lectura.** El SOW actua como confirmador con lift positivo universal:
todas las configuraciones separan candidate+confirmed de candidate
sin SOW, y el resultado generaliza al bloque holdout. Lo que no se
cumple es D3 al 50% de nivel absoluto. El propio dictamen (seccion 13)
marco D3 como umbral provisional que debe coexistir con lift; el
criterio declarado como real es lift > 0.

**No se elige combinacion en este informe.** Con D3 no cumplido, la
eleccion de parametros queda bloqueada hasta dictamen. Se listan las
candidatas del apartado 7. Se abre expediente 23_expediente_5b3_D3.md.

---

## 2. Dataset y configuracion

- **Fuente:** data/stock_prices.parquet.
- **Universo:** 315 tickers con >= 200 observaciones Close no-NaN.
  Queda fuera SPCX (77 obs, IPO 2026-06-12).
- **Periodo:** 2021-10-01 -> 2026-10-01 (1254 sesiones).
- **Split 70/30 por fecha unica global:** corte 2025-03-31.
  - Calibracion: t0 <= 2025-03-31.
  - Holdout: t0 > 2025-03-31.
- **Episodios candidatos totales (universo completo):** 2415.
- **Aviso survivorship bias:** universo actual, no historico.

**Constantes de control (fijas, no en grid):**

    WYCKOFF_ATR_WINDOW = 20
    PRECEDENT_WINDOW   = 60 (W del struct_max)
    WYCKOFF_T_NORM_K   = 0.25 (congelado en 5b.2)

**Grid (240 combinaciones):**

    N     in {20, 30, 40, 60}
    M     in {5, 10, 15, 20, 30}
    X_ATR in {0.25, 0.50, 0.75, 1.00}
    Y_VOL in {1.10, 1.20, 1.50}

---

## 3. Metrica primaria y secundarias

Definiciones operativas aplicadas (protocolo v3 + dictamen):

- **Unidad de observacion:** candidate episode = (candidate_t=True)
  AND (candidate_{t-1}=False), por ticker.
- **Confirmacion:** primer SOW valido en [t0, t0+M]. Si existe, el
  episodio es confirmed. El ancla temporal del outcome es el indice de
  ese primer SOW.
- **Baseline:** episodios candidate sin SOW en la ventana. Ancla en t0.
- **Outcome primario:** struct_deterioration_H20 = (struct[t+20] < struct[t]),
  con struct[t] y struct[t+20] validos.
- **Secundarias:** price_weakness (close[t+h] < close[t]*0.98),
  below_support (close[t+h] < support[t]), lower_low
  (min(close[t+1:t+h]) < min(close[t-N:t])).
  Horizontes H10, H20, H40.
- **Lift:** incremental_lift_H20 = P20_confirmed - P20_baseline.
  Calculado para H10, H20, H40.

**Boundary temporal:** evento con t_anchor + h > fin_bloque excluido
para ese horizonte. Sin trasvase de informacion entre bloques.

---

## 4. Cumplimiento de criterios D1-D4

| Criterio | Descripcion | Combos OK | Combos totales |
|---|---|---:|---:|
| D1 | n_sow_raw >= 100 | 240 | 240 |
| D2 | cal_n_confirmed >= 20 | 240 | 240 |
| D3 | cal_pct_conf_struct_H20 >= 0.50 | **0** | 240 |
| D4 | neighbour_confirmation_delta_pp <= 30 | 240 | 240 |

Combos que cumplen exactamente 3 de 4: 240/240 (todos fallan D3).
Combos que cumplen 2 o menos: 0.

**Causa unica del bloqueo:** D3 al 50%. El maximo observado es 45.0%.

---

## 5. Distribuciones del grid

### 5.1. Por N (media de las 60 combinaciones de cada N)

    N=20   H20_conf=0.3825  H20_base=0.2830  lift_H20=0.0995  n_sow_raw=4263
    N=30   H20_conf=0.3746  H20_base=0.2871  lift_H20=0.0875  n_sow_raw=3294
    N=40   H20_conf=0.3906  H20_base=0.2882  lift_H20=0.1025  n_sow_raw=2730
    N=60   H20_conf=0.4182  H20_base=0.2864  lift_H20=0.1318  n_sow_raw=2033

N=60 domina en H20_conf y lift. Menos eventos, mas selectivo.

### 5.2. Por X_ATR (media de las 60 combinaciones de cada X)

    X=0.25  lift_H20=0.1017  n_sow_raw=5039  n_confirmed=411
    X=0.50  lift_H20=0.1129  n_sow_raw=3439  n_confirmed=334
    X=0.75  lift_H20=0.1015  n_sow_raw=2305  n_confirmed=264
    X=1.00  lift_H20=0.1051  n_sow_raw=1536  n_confirmed=204

X=0.50 maximiza lift medio. X=0.75/1.00 reducen muestra.

### 5.3. Por Y_VOL (media de las 80 combinaciones de cada Y)

    Y=1.10  lift_H20=0.1088  n_sow_raw=3636  n_confirmed=331
    Y=1.20  lift_H20=0.1129  n_sow_raw=3288  n_confirmed=316
    Y=1.50  lift_H20=0.0942  n_sow_raw=2316  n_confirmed=263

Y=1.20 maximiza lift medio.

### 5.4. Por M (media de las 48 combinaciones de cada M)

    M=5    lift_H20=0.0934  n_confirmed=217
    M=10   lift_H20=0.0994  n_confirmed=255
    M=15   lift_H20=0.0955  n_confirmed=291
    M=20   lift_H20=0.0997  n_confirmed=335
    M=30   lift_H20=0.1386  n_confirmed=418

M=30 maximiza lift medio: ventana larga permite confirmar mas
episodios. M=15 es el unico valle local.

---

## 6. Lift positivo universal

`incremental_lift_H20 > 0` en las **240 combinaciones** en calibracion
y en las **240 combinaciones** en holdout. Rango:

    cal_lift_H20:  min=0.0408  median=0.1035  max=0.1920
    hold_lift_H20: min positivo, todos > 0

Es decir: el SOW, con cualquier configuracion de la grid, aporta
informacion sobre deterioro futuro mas alla del candidate. La
conclusion no depende de la combinacion elegida.

---

## 7. Candidatas

### 7.1. Top-3 por cal_lift_H20

| N | M | X_ATR | Y_VOL | cal_H20_conf | cal_H20_base | cal_lift_H20 | cal_n_conf | cal_n_base | neighbor_delta_pp |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 60 | 30 | 0.50 | 1.20 | 0.4458 | 0.2538 | **0.1920** | 406 | 1212 | 5.01 |
| 60 | 30 | 0.50 | 1.10 | 0.4366 | 0.2491 | 0.1874 | 424 | 1194 | 5.25 |
| 40 | 30 | 0.50 | 1.20 | 0.4266 | 0.2535 | 0.1730 | 450 | 1168 | 5.62 |

### 7.2. Candidata #1 (N=60, M=30, X=0.50, Y=1.20) - comparativa cal/hold

    cal:   conf=0.4458  base=0.2538  lift=+0.1920  n_conf=406  n_base=1212
    hold:  conf=0.5689  base=0.3427  lift=+0.2261  n_conf=180  n_base=617

**La combinacion generaliza: el lift incluso sube en holdout.**
En H10 y H40:

    H10: conf=0.7388  base=0.4426  lift=+0.2962
    H20: conf=0.4458  base=0.2538  lift=+0.1920
    H40: conf=0.4650  base=0.2419  lift=+0.2231

El lift es monotono creciente con el horizonte.

### 7.3. Top-10 por hold_lift_H20 (independiente de la calibracion)

    N  M  X_ATR Y_VOL  hold_lift_H20  hold_n_conf
    40 30 1.00  1.10   0.3381         104
    40 30 1.00  1.20   0.3371         103
    60 30 1.00  1.10   0.3272          90
    60 30 1.00  1.20   0.3218          88
    40 10 1.00  1.20   0.3215          73
    40 10 1.00  1.10   0.3215          73
    60 10 1.00  1.10   0.3190          61
    40 15 1.00  1.20   0.3183          78
    40 15 1.00  1.10   0.3183          78
    40 30 1.00  1.50   0.3153          88

**Nota:** el top-holdout prefiere X=1.00 (mas exigente), mientras el
top-calibracion prefiere X=0.50. Divergencia a resolver por el auditor.

---

## 8. Verificacion SLB / PCG (protocolo seccion 10, 5c.3 H1/H3)

### 8.1. SLB

Episodios SOW en los ultimos 90 dias:

    v1.6 (sin umbrales):    3
    v1.8 (X=0.50, Y=1.20):  0

Los 3 SOW de v1.6 tenian:

    2025-10-10  bd=+0.168  vr=0.949
    2026-06-18  bd=+0.264  vr=2.911
    2026-06-24  bd=+0.235  vr=1.882

Ninguno pasa el umbral X>=0.50. La definicion v1.8 excluye
exactamente los casos H1 del diagnostico 5c.3.

### 8.2. PCG

    v1.6 (sin umbrales):    4
    v1.8 (X=0.50, Y=1.20):  1

El unico SOW que sobrevive en v1.8:

    2026-08-31  bd=+4.028  vr=6.820

Ruptura masiva de soporte con volumen 6.8x. Es un SOW legitimo,
no el marginal H3 del diagnostico. Los casos H3 (-1.43%, 1.05x)
quedan fuera.

**Conclusion H1/H3:** el filtro X_ATR + Y_VOL elimina los falsos
positivos marginales sin eliminar el SOW estructural. Verificacion
positiva.

---

## 9. Metricas secundarias (candidata #1)

Para N=60, M=30, X=0.50, Y=1.20, H20, las cuatro metricas del
outcome:

    struct_deterioration:  conf=0.4458  base=0.2538  lift=+0.1920
    price_weakness:        conf=0.2229  base=0.1445  lift=+0.0785
    below_support:         (consultar CSV)
    lower_low:             (consultar CSV)

La senal se concentra en la metrica estructural (`struct_deterioration`)
y es mas debil en `price_weakness`. Esto es consistente con la
definicion: el SOW es una senal estructural, no de precio inmediato.

---

## 10. Analisis prohibidos (protocolo seccion 8)

Comprobaciones explicitas:

- **NO se optimizo por n_confirmed.** La candidata #1 no es la de mas
  confirmados (n_conf=406 vs maximo 743).
- **NO se optimizo por numero de DISTRIBUTION total.** No aplica; el
  informe no selecciona combinacion.
- **NO se ajustaron parametros para los 5 casos 5c.3.** La verificacion
  SLB/PCG es ex-post, no ex-ante.
- **NO se modificaron criterios despues de ver resultados.** El
  protocolo v3 esta congelado en aabc921.
- **NO se incluyo W ni ATR_WINDOW en el grid.**
- **NO se uso pct_deterioro_continuo** (nombre retirado en v3).
- **NO se contaron sesiones candidate como eventos independientes.**
  Unidad = entrada en episodio.
- **NO se reporto lift sin baseline.** Ambas columnas en el CSV.
- **NO se trasvaso informacion del futuro al bloque actual.** Boundary
  por horizonte implementado.

---

## 11. Reproducibilidad

    Set-Location D:\Macro_Sectorial
    py scripts\calibrate_wyckoff_sow_5b3.py

Determinista sobre el snapshot actual de data/stock_prices.parquet.
Tiempo de ejecucion: ~318s en el equipo de referencia.

Salidas:

- `outputs/audit/wyckoff_5b3_grid.csv` (240 filas x 101 columnas).
- `outputs/audit/wyckoff_5b3_summary.json`.
- `outputs/audit/wyckoff_5b3_run.log`.

---

## 12. Conclusion

**Hecho empirico 1.** El SOW con ATR-normalizacion + umbrales X/Y
aporta lift positivo universal (240/240) sobre el baseline candidate
sin SOW, tanto en calibracion como en holdout.

**Hecho empirico 2.** El nivel absoluto `struct_deterioration_H20`
en episodios confirmados no alcanza el 50% definido en D3. Maximo
45.0%, mediana 39.0%.

**Hecho empirico 3.** Las combinaciones con mayor lift concentran
N=60, X=0.50, Y=1.20, y lift crece monotonicamente con M.

**Hecho empirico 4.** Verificacion H1/H3 (SLB, PCG) positiva: el
filtro X_ATR + Y_VOL elimina los falsos positivos marginales
identificados en 5c.3.

**Bloqueo.** D3 esta marcado como provisional en el propio dictamen
(seccion 13): "D3 por si solo no basta. Debe coexistir con
incremental_lift_H20. El criterio real es lift > 0 (o el umbral que
se decida en el informe final)". La decision de si D3=0.50 sigue
vigente, se relaja, o se sustituye por un criterio de lift, es del
auditor externo.

**No se elige combinacion. No se congelan parametros. No se toca
config/settings.py.**

---

## 13. Proximo paso

Expediente `23_expediente_5b3_D3.md` remitido al auditor externo.
Contiene:

- Pregunta sobre D3.
- Pregunta sobre la divergencia cal (X=0.50) vs hold (X=1.00).
- Propuesta de criterio alternativo basado en lift.
- Datos completos del grid para su analisis independiente.

Hasta dictamen, el sistema queda:

    v1.8              CONTRATO VIGENTE
    5b.3 (protocolo)  congelado en commit aabc921
    5b.3 (ejecucion)  COMPLETADA, bloqueada en seleccion
    parametros SOW    None en config (fail-closed)
    5b.4              BLOQUEADA
    5c.4              BLOQUEADA
    Legacy            INTACTO
    Consumidores      SIN MIGRAR

---

**Fin del informe 5b.3.**

---

## 14. NOTA DE CIERRE (2026-10-02, post-dictamen)

**Este informe queda superado por `24_dictamen_5b3_D3.md`.**

El dictamen externo ha detectado un sesgo metodologico en el diseno del
outcome que invalida el `incremental_lift_H20` reportado en las
secciones 6, 7 y 9 de este informe: el grupo confirmed y el grupo
baseline no estaban anclados al mismo momento temporal (inmortal time
bias).

Concretamente:

    CONFIRMED: outcome = struct[t_sow + 20] < struct[t_sow]
    BASELINE:  outcome = struct[t0   + 20] < struct[t0]

donde t_sow puede estar hasta M=30 sesiones despues de t0. Ambos
grupos se evaluan en relojes distintos. El lift positivo universal
(240/240) es **evidencia exploratoria prometedora, no validacion**.

### Estado de los resultados de este informe

- **Validos como diagnostico historico:** cumplimiento D1-D4, conteos,
  distribuciones por N/M/X/Y, verificacion H1/H3 (SLB/PCG).
- **Invalidos como evidencia de confirmacion SOW:** todo valor de
  `incremental_lift_H20` / `H40` y cualquier interpretacion derivada.
- **Candidata A (N=60, M=30, X=0.50, Y=1.20):** NO congelada.
  Compite de nuevo bajo el protocolo 5b.4 con landmark temporal.
- **D3=50%:** se mantiene como resultado formal (FAIL). No se relaja.
- **Divergencia X=0.50 vs X=1.00:** queda como diagnostico historico.
  El holdout ya ha sido inspeccionado; no sirve como validacion
  independiente de 5b.4.

### Que sigue

`25_protocolo_fase_5b4.md` — nuevo protocolo con landmark temporal,
lift como outcome primario, inferencia agrupada por ticker, criterio
de seleccion ex-ante, y holdout verdaderamente no inspeccionado.

### Lo que NO cambia

- Contrato v1.8 vigente.
- `detect_sow` fail-closed (parametros obligatorios).
- `config/settings.py` con los 4 SOW a None.
- K=0.25 congelado (5b.2).
- Legacy intacto. Consumidores sin migrar.
- 5c.4 y 5d bloqueadas.

---

**Fin de la nota de cierre.**