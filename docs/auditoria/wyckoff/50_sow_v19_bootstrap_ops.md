# 50 - Bootstrap v2 - Especificacion operativa (expediente tecnico)

Fecha: 2026-10-05
Rama: sow-v4-validation, HEAD a505b87
Fichero: scripts/sow_v19_bootstrap_v2.py
Motivo: cierre del expediente tecnico del bootstrap del protocolo v3 (auditor
2026-10-05, seccion 5). Documentar exactamente como se forma, concatena,
trata y descarta cada parte del panel bootstrap.

---

## 1. Dataset original

    data/stock_prices.parquet
    Universo: 315 tickers con >= 200 obs Close no-NaN.
    session_dates: union ordenada de todos los calendarios.
                   DatetimeIndex ordenado, unico. 1297 sesiones.

---

## 2. Formacion de bloques

    BLOCK_LEN  = 120
    BURN_IN    = 200
    T          = len(session_dates) = 1297
    n_blocks   = ceil(T / BLOCK_LEN) = ceil(1297/120) = 11

Por cada bloque bid en [0, n_blocks):

    s = rng.integers(BURN_IN, T - BLOCK_LEN)
         # s en [200, 1177]
    panel_slice = session_dates[s-BURN_IN : s+BLOCK_LEN]
         # 320 sesiones: 200 de burn-in + 120 activas
    active_slice = [False]*200 + [True]*120
    bid_slice   = [bid]*320

Cada bloque tiene 320 posiciones. Las 200 primeras son burn-in (active=False).
Las 120 ultimas son activas (active=True).

---

## 3. Concatenacion

    panel_dates  = concat(session_dates[s_i-BURN_IN : s_i+BLOCK_LEN] para i)
    panel_active = concat(active_slice_i)
    panel_bid    = concat(bid_slice_i)

Longitud total del panel: n_blocks * (BURN_IN + BLOCK_LEN) = 11 * 320 = 3520.

Los bloques pueden solaparse temporalmente: dos bloques pueden usar el mismo
rango de session_dates original. Es intencional (moving block bootstrap).

---

## 4. Tratamiento de fronteras

Problema: el pipeline usa ventanas rolling de hasta 200 sesiones (MA200,
robust_z window=200). Concatenar bloques sin contexto introduciria
discontinuidades artificiales en las propias variables a validar.

Solucion: BURN_IN = 200 sesiones de contexto antes de la zona activa de
cada bloque. Dentro de la zona activa, las features son identicas a las del
dataset original porque el contexto cubre todas las rolling windows.

    BURN_IN = 200 >= max(MA200, robust_z(200)) = 200

Precomputacion: los features del dataset original se calculan UNA VEZ
(precompute_features sobre session_dates completo). Cada panel reindexa
por fecha. Equivalencia bit-a-bit demostrada: obs_lifts v1 == v2fix
para las 240 configuraciones.

---

## 5. Manejo de fechas y tickers ausentes

    pos_of_date = {date: i for i, date in enumerate(session_dates)}
    active_bits[orig_idx, bid] = True  si la sesion orig_idx aparece
                                        en la zona activa del bloque bid

Tickers ausentes en una fecha (por calendario distinto, NaN):
- En reindex, se rellenan con NaN.
- SOW reindex devuelve NaN -> fillna(0).int.
- Episodios que requieran datos NaN se descartan via episode_metrics
  (struct_L o struct_H NaN -> None).

---

## 6. Descarte de burn-in y frontera

Un episodio (t0, L+H) es valido en el panel si:

    multiplicity(t0, L+H) = |{bid : active_bits[t0, bid]
                              AND active_bits[L+H, bid]}| > 0

Un episodio puede aparecer en multiplicidad > 1 si t0 y L+H caen ambos en
la zona activa de varios bloques (por solapamiento de bloques).

Ponderacion: cada episodio se pondera por su multiplicidad en el calculo
del lift de la replica.

    lift_r = sum(w_i * det_i para i confirmados) / sum(w_i para confirmados)
           - sum(w_j * det_j para j baseline)   / sum(w_j para baseline)

donde w_i = multiplicity del episodio i.

Esto es la semantica correcta del panel bootstrap: cada aparicion
independiente del mismo episodio cuenta como una observacion.

---

## 7. Calendario comun

El calendario es global (union de todos los tickers). Un panel bootstrap
reutiliza el MISMO indice temporal para todas las configuraciones y todos
los tickers. Esto preserva la dependencia cross-sectional de mercado:
shocks macro, regimen, VIX, etc. afectan simultaneamente a todos los tickers
en la misma fecha.

No se resamplea el tiempo de cada ticker de forma independiente.

---

## 8. Estadistico por replica

Por cada replica b:
- Se construye un panel (panel_dates, panel_active, panel_bid).
- Se calcula `active_bits` (T_orig x n_blocks).
- Se filtran los episodios precomputados por combo (240).
- Se calcula `lift_j,b` para las 240 configuraciones.
- Se calcula `T*_j,b = (lift_j,b - lift_j) / SE_j` con SE_j observado fijo.
- Se obtiene `maxT_b = max_j T*_j,b`.

---

## 9. Estadistico observado y p-valor

    SE_j = std({lift_j,b}_{b=1..B}, ddof=1)
    T_obs_j = (lift_j,obs - MDE) / SE_j    con MDE=0.05
    maxT_b = max_j T*_j,b
    p_maxT_j = (1 + #{b : maxT_b >= T_obs_j}) / (1 + B)

MDE = 0.05. B = 20000.

---

## 10. Determinismo

    seed = 20261005
    rng  = np.random.default_rng(SEED + b)

Cada replica b tiene su propio rng con semilla SEED + b. Ejecucion
determinista. Reproducible bit-a-bit con mismo seed, mismo dataset,
mismo commit.

---

## 11. Rendimiento

    B=100,     12 workers:  6.9 s
    B=20000,   12 workers:  188.9 s

Precomputacion (una vez por ejecucion): ~92 s (SOW 34.4s + episodios 57.6s).
Total por ejecucion B=20000: ~280 s.

85x mas rapido que la version sin optimizar (v1, B=100: 591 s).

---

## 12. Diferencia respecto a v1

v1 (sow_v19_bootstrap.py, referencia historica): recalculaba
find_candidate_starts sobre el panel concatenado. En las fronteras de
bloque, `candidate_{t-1}` apuntaba al burn-in del mismo bloque (200
sesiones antes), generando/pierdiendo candidate_starts artificialmente.

v2: candidate_starts se precomputa sobre el dataset original. Cada t0 es un
candidate_start real del calendario real. Sin artefactos de frontera.

Esta correccion cambia la distribucion de SE_j y por tanto p_maxT. La
evidencia de que v2 es correcta: obs_lifts v1 == obs_lifts v2 (240/240
bit-identicos), pero p_maxT difiere por el sesgo de frontera de v1.

---

## 13. Trazabilidad

- Codigo: scripts/sow_v19_bootstrap_v2.py, commit a505b87.
- Resultado: outputs/audit/sow_v19/bootstrap_v2fix_B20000.json.
- Log: outputs/audit/sow_v19/_bootstrap_v2fix_B20000.log.
- Protocolo: docs/auditoria/wyckoff/SOW_v19_PROTOCOL.json v3.
- Freeze: docs/auditoria/wyckoff/SOW_v19_PROTOCOL.freeze.json (FIRMADO).

---

**Fin de la especificacion operativa.**
