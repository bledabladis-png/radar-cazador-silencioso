# PRE-REGISTRO DICT3-REACCUMULATION v4

**Estado:** v4. Pendiente de revision y firma del auditor externo.
**Fecha:** 2026-10-29.
**Ciclo:** sesion 16.
**Frente:** DICT3-REACCUMULATION (REGISTRO.yaml, OPEN).
**Base:** v3 (commit e479eca6) + dictamen externo 2026-10-29 (chat).
**Regla:** parametros congelados antes de ejecutar. No se ajustan tras
ver resultados.

Responde a los 6 hallazgos del dictamen v3: P0-H01 historial R1 y T7;
P1-H02 algoritmo de busqueda; P1-H03 bootstrap de C2; P1-H04
tratamiento de datos insuficientes; P1-H05 T6 vs R1; P1-H06 warm-up
vs evaluacion.

---

## 0. Objetivo y alcance

Detectar REACUMULACION de continuacion: consolidacion en tendencia
alcista sana que se rompe al alza, sin que el detector actual de
reversion (SC->AR->Range->SOS) la capture.

La etiqueta REACCUMULATION_CONFIRMED es salida del detector
experimental. NO es una fase nueva del sistema, NO modifica
classify_wyckoff_phase ni el orquestador, NO se publica en el
reporte diario. Vive en un CSV experimental aislado.

---

## 1. Estado del holdout

- cohort y r3000 fueron consumidos por SOW-V6 el 2026-10-06.
- Registrados como CONSUMED_FOR_CANDIDATE(SOW-V6),
  event_type=EXPLORATORY_DESIGN_INFLUENCE.
- Poblacion: composicion 2026Q4 del S&P 500 aplicada retrospectivamente.
- Resultado sobre estos parquets: EXPLORATORIO o evidencia secundaria.
- Confirmacion real: periodo futuro reservado (§8).

---

## 2. Reglas R1-R5 (conjunto consolidado)

### 2.1. Notacion y parametros congelados

- W = 40 sesiones (ventana de consolidacion).
- W_v = 40 sesiones (ventana de normalizacion de volumen).
- L_prev = 60 sesiones (ventana de contexto UP).
- H_R1 = 400 sesiones (historial minimo ANTES de b-60 para evaluar R1,
  ver §2.3).
- Horizonte C2 = 63 sesiones.

Series: C(t) Close ajustado por splits (auto_adjust=False; ajuste
manual si el proveedor no lo aplica). V(t) volumen.

### 2.2. v_norm: formula unica y tratamiento de datos (H-04)

Para cualquier sesion t y ventana V_v = [t-W_v, t-1]:

    base(t)  = mean( V(s) para s en [t-W_v, t-1] )
    sd(t)    = std( V(s) para s en [t-W_v, t-1], ddof=0 )
    v_norm(t) = ( V(t) - base(t) ) / sd(t)

**Validez (regla unica, sin ambiguedad):**
La ventana [t-W_v, t-1] debe contener W_v = 40 sesiones COMPLETAS con
V(s) valido (> 0, no NaN). Si falta una sola: la ventana entera es
`INVALID_DATA`. No se permite ventana parcial. El umbral de "20
observaciones" del v3 desaparece: o hay 40 validas o no hay v_norm.

Ademas: si sd(t) == 0 (todos los volumenes iguales), `INVALID_DATA`.

**Tres estados de v_norm(t):**
- `INVALID_DATA`: alguna de las 40 sesiones con V(s) no valido, o sd=0.
- `VALID`: v_norm(t) calculado (float).
- (El valor numerico puede cumplir o no el umbral; eso se decide en
  R3/R5, no aqui.)

**Diferencia con Wyckoff existente:** el detector existente usa
robust_zscore sobre volumen (mediana/MAD); aqui v_norm es z-score
clasico sobre media/std. NO son matematicamente identicos. Se
declara la diferencia.

### 2.3. R1: contexto UP (H-01 corregido)

interfaz_efectiva:
  modulo: indicators.wyckoff_v1
  funcion: detect_trend_context
  firma: (df_ticker, ticker, as_of) -> str
  invocacion: detect_trend_context(df_ticker, ticker, as_of=b-60)

**Historial minimo (H-01):**
`detect_trend_context` depende internamente de `t_norm`, cuya media
movil lenta requiere `min_periods=200` cierres, y del propio contexto
que consume hasta `120` valores validos de `t_norm`. Por tanto:

    H_R1 = 200 (MA lenta) + 120 (contexto) + 80 (margen) = 400 sesiones

antes de la fecha de evaluacion `b-60`. Es decir: la sesion `b`
requiere al menos `400 + 60 = 460` sesiones de historial antes de
ella. Se congela H_R1 = 400 para tener margen ante pequeñas
variaciones del indicador. La implementacion efectiva debera confirmar
el minimo real (AST de `detect_trend_context` + test de calentamiento)
ANTES de codificar el detector. Si el minimo real es > 400, se
incrementa H_R1 sin recalibrar nada mas.

**Regla de evaluacion:**
- Si hay < H_R1 sesiones validas antes de `b-60`, el ticker es
  `NOT_EVALUABLE` en esa fecha (no se asume UP ni se asume otro
  contexto).
- Si `detect_trend_context` devuelve `UNKNOWN` o cualquier valor
  distinto de `UP`, R1 = `NOT_TRIGGERED` (evaluado, no cumple).
- Si devuelve `UP`, R1 = `PASS`.

R1 exige `PASS`. No se admite `UNKNOWN` como UP por defecto.

### 2.4. R2: consolidacion

Sobre [b-W, b-1]:

    S_b = min( C(b-W), ..., C(b-1) )
    R_b = max( C(b-W), ..., C(b-1) )
    w_b = R_b - S_b
    tol_b = 0.05 * w_b

R2 = PASS si:
    (a) S_b > 0, R_b > S_b, w_b > 0
    (b) w_b / R_b <= 0.15
    (c) todas las sesiones de [b-W, b-1] con C(s) > 0 y no NaN
    (d) existe al menos un pullback segun R3 (§2.5)

Si (a), (b) o (c) fallan: R2 = `NOT_EVALUABLE`.
Si (d) falla pero (a)(b)(c) pasan: R2 = `NOT_TRIGGERED` (ventana
valida, sin pullback).

### 2.5. R3: pullback en consolidacion

Un pullback es cualquier sesion p en [b-W, b-1] tal que:
    (i)  C(p) <= S_b + tol_b
    (ii) v_norm_p(p) = VALID y v_norm_p(p) <= -0.3
         (ventana [p-W_v, p-1] segun §2.2)
    (iii) C(p) y V(p) validos (no NaN, > 0)

R3 = PASS si existe al menos un p con (i)+(ii)+(iii).
R3 = `NOT_TRIGGERED` si hay al menos un p con (i) que tiene
    v_norm_p(p) = VALID y > -0.3, y ninguno que cumpla (ii).
R3 = `NOT_EVALUABLE` si TODOS los p con (i) tienen
    v_norm_p(p) = INVALID_DATA.

**Regla de busqueda de pullback (H-04):** se recorre [b-W, b-1] de
mas reciente a mas antiguo. Se aceptan como `TRIGGERED` los que
cumplan (i)+(ii)+(iii). Si se agotan todos los p con (i) sin
`TRIGGERED` y al menos uno de ellos era `NOT_TRIGGERED`, el
resultado es `NOT_TRIGGERED`. Si todos los candidatos eran
`INVALID_DATA`, el resultado es `NOT_EVALUABLE`.

### 2.6. R4: confirmacion de ruptura

Sea b la primera sesion que cumple `C(b) > R_b + tol_b` (definicion
operativa de `b` en §3). Se exige:

    C(b)   > R_b + tol_b    (por definicion de b)
    C(b+1) > R_b + tol_b
    C(b+2) > R_b + tol_b

Si C(b+1) o C(b+2) <= R_b + tol_b, R4 = `NOT_TRIGGERED` (ruptura
fallida; el algoritmo de busqueda §3 debe continuar buscando otras
`b` candidatas).

**La etiqueta REACCUMULATION_CONFIRMED solo se emite en b+2.**

### 2.7. R5: volumen de ruptura

R5 = PASS si:
    v_norm_b(b) = VALID y v_norm_b(b) >= +0.5
    (ventana [b-W, b-1] segun §2.2)

R5 = `NOT_TRIGGERED` si v_norm_b(b) = VALID y < +0.5.
R5 = `NOT_EVALUABLE` si v_norm_b(b) = INVALID_DATA.

### 2.8. Umbrales de volumen (justificacion congelada)

- -0.3 (pullback bajo): umbral NUEVO. NO es
  SC_V_NORM_THRESHOLD=0.5. Corresponde a SC_C_NORM_THRESHOLD=0.5
  en magnitud, pero es para compresion, no volumen.
- +0.5 (ruptura alto): coincide numericamente con
  SC_V_NORM_THRESHOLD, pero NO es su inversion (-0.3 != -0.5). Es
  coincidencia de magnitud.

Ambos congelados. No se optimizan tras ver resultados.

---

## 3. Algoritmo de busqueda de episodios (H-02)

**Definicion cronologica unica.** Para cada ticker, con serie
ordenada por fecha:

    t = primera sesion con historial suficiente para R1 (t >= fecha_inicio + H_R1 + L_prev)
    while t <= ultima_sesion - 2:                # -2 por R4 (b+2)
        b = t
        # 1. Filtrar por contexto previo (R1)
        R1_res = evalua_R1(ticker, b-60)          # PASS / NOT_TRIGGERED / NOT_EVALUABLE
        if R1_res != PASS:
            t = t + 1; continue
        # 2. Consolidacion y pullback previos (R2 + R3)
        R2_res, R3_res = evalua_R2_R3(b)         # §2.4, §2.5
        if R2_res != PASS or R3_res != PASS:
            t = t + 1; continue
        # 3. Ruptura (R5 sobre b, R4 sobre b+1, b+2)
        if C(b) <= R_b + tol_b:
            t = t + 1; continue                   # b no es candidata
        R5_res = evalua_R5(b)
        if R5_res != PASS:
            t = t + 1; continue
        R4_res = evalua_R4(b)                    # b+1, b+2 sobre umbral
        if R4_res != PASS:
            t = t + 1; continue                   # ruptura fallida; se
                                                  # continua escaneando
        # 4. Episodio confirmado en b+2
        emitir_episodio(ticker, date_start=b, date_confirmed=b+2)
        t = b + 3                                 # salto post-episodio

**Reglas:**
- `t` avanza SIEMPRE de uno en uno salvo tras emitir un episodio,
  donde salta a `b+3` (evita solapamientos inmediatos).
- Una ruptura que falla R4 en `b+1`/`b+2` NO consume el estado: se
  sigue escaneando con `t+1`.
- La busqueda es sobre toda la serie del ticker, no por ventanas.

**Deduplicacion (post-scan):**
Dos episodios del mismo ticker con intervalos `[date_start,
date_confirmed]` que se solapan temporalmente se colapsan en el de
menor `date_start`. Se aplica ANTES de C1, C2, C3. Regla congelada,
no condicionada a resultados.

---

## 4. Estados y elegibilidad (H-04 unificada)

Todo resultado parcial de R1..R5 toma uno de tres valores:
- `PASS`: condicion evaluada y cumple.
- `NOT_TRIGGERED`: condicion evaluada, no cumple.
- `NOT_EVALUABLE`: no se puede evaluar por datos insuficientes.

**Regla de un ticker por episodio candidato:**
- Si TODAS R1..R5 = PASS: episodio emitido.
- Si alguna = `NOT_EVALUABLE`: episodio no emitido; ticker se
  registra como `NOT_EVALUABLE` en esa fecha.
- Si alguna = `NOT_TRIGGERED` (y ninguna `NOT_EVALUABLE`): ticker
  evaluado, sin episodio.

**Regla de elegibilidad de C1 (§6.1):**
Un ticker es evaluable en un chunk si en AL MENOS una sesion del
chunk, R1..R5 son todas `PASS` o `NOT_TRIGGERED` (ninguna
`NOT_EVALUABLE`). Si en todas las sesiones del chunk hay al menos
una `NOT_EVALUABLE`, el ticker no cuenta para el denominador.

Un dato desconocido NO se interpreta como "sin evento". Se
interpreta como "este ticker no es evaluable en este chunk".

---

## 5. Criterios C1-C3 (corregidos)

### 5.1. C1: frecuencia por chunks

- Periodo de evaluacion dividido en chunks consecutivos NO
  solapados de 252 sesiones. Ultimo incompleto se descarta.
- Para cada chunk:
    n_eval      = tickers evaluables (§4) en el chunk
    n_episodios = tickers con >= 1 episodio cuyo date_start cae
                  en el chunk
    ratio       = n_episodios / n_eval
- C1 exige:
    (a) n_chunks_evaluados >= 2
    (b) para TODOS los chunks: ratio en [0.01, 0.30]
- Si n_chunks_evaluados < 2: C1 = INSUFFICIENT_SAMPLE.
- Asignacion del episodio al chunk: por `date_start`.

### 5.2. C2: tasa de acierto con bootstrap estacionario (H-03)

**Cambio v3 -> v4:** v3 usaba bootstrap por mes calendario. Eso
preserva dependencia intra-mes entre tickers pero NO preserva la
dependencia entre meses consecutivos cuando el horizonte de retorno
es 63 sesiones (~3 meses). Dos episodios separados por <63 sesiones
comparten parte de su ventana futura de retorno.

**Metodo congelado: bootstrap estacionario de Politis-Romano con
longitud media de bloque 63 sesiones.**

Definicion:
- Retorno de episodio: r(e) = C(date_confirmed + 63) / C(date_confirmed) - 1.
  Close price return, splits ajustados, dividendos no reinvertidos.
- Exito: r(e) >= 0.05.
- Evaluables: episodios con 63 sesiones futuras disponibles.
- Solapamiento intra-ticker: si dos episodios del mismo ticker
  tienen date_confirmed a <63 sesiones, colapsar en el primero.
- Orden: episodios ordenados por date_confirmed ascendente.
- Bloques: longitud L ~ Geometric(p = 1/63) + 1. Cada replica:
  se camina el array ordenado de episodios; en cada paso se toma un
  bloque de longitud L empezando por un episodio aleatorio con
  reemplazo; se concatenan hasta reconstruir N episodios. B=1000,
  seed=42.
- **Replica valida:** si al reconstruir una replica de tamaño N, se
  obtiene al menos 30 episodios con exito/fracaso observado (todos
  lo estan por construccion, ya son evaluables). Si <30 por alguna
  razon (no deberia), la replica se descarta.
- **Numero minimo de replicas validas:** si B_validas < 800 (80%),
  C2 = INSUFFICIENT_SAMPLE.
- p_hat = tasa puntual de exitos (todos los episodios, no
  remuestreados).
- IC95_lower = percentil 2.5 de las p_replica de las B_validas.
- **Minimo de bloques efectivos:** si N_episodios < 63 (numero de
  sesiones del horizonte), el bootstrap estacionario puede tener
  poca informacion: se requiere N_episodios >= 63 (no solo 30). Si
  N < 63: C2 = INSUFFICIENT_SAMPLE. Nota: 63 es el horizonte, no
  el minimo de "muestras independientes". Es un suelo de robustez.

C2 exige:
    (a) N_episodios_evaluables >= 63
    (b) B_validas >= 800
    (c) p_hat >= 0.60
    (d) IC95_lower > 0.50
- Si (a) o (b) fallan: C2 = INSUFFICIENT_SAMPLE.

Wilson rechazado explicitamente (no captura dependencia entre
episodios). Bootstrap por mes (v3) rechazado por no preservar
dependencia a 63 sesiones.

### 5.3. C3: solapamiento con acumulacion

- Episodio de reacumulacion: [date_start, date_confirmed].
- Acumulacion confirmada: [SC_date, SOS_date] segun contrato
  vigente.
- Solapa(e): interseccion temporal no vacia.
- **n_eval_C3** (H-04 del dictamen v1 y del dictamen v3): episodios
  de reacumulacion cuyo periodo [date_start, date_confirmed] cae
  dentro del alcance temporal del detector de acumulacion Y cuya
  evaluacion es posible (hay datos de OHLCV cubriendo el periodo).
  Incluye episodios sin acumulacion confirmada en el mismo periodo:
  si el detector de acumulacion pudo evaluarse y no encontro
  acumulacion, la tasa de solapamiento contribuye como 0 (exito).
- n_solapa = episodios con Solapa = True.
- ratio_C3 = n_solapa / n_eval_C3.

C3 exige:
    (a) n_eval_C3 >= 63
    (b) ratio_C3 <= 0.10
- Si n_eval_C3 < 63: C3 = NO_EVALUABLE.
- Si n_eval_C3 >= 63 y n_solapa = 0: ratio_C3 = 0. Exito.
- Si n_eval_C3 < 63: NO_EVALUABLE, no exito por defecto.

### 5.4. Aceptacion global

C1, C2 y C3 deben ser EVALUABLES y SUPERADOS. Un
INSUFFICIENT_SAMPLE o NO_EVALUABLE no cuenta como "pasa por
defecto". Sin los tres, el veredicto global es
INSUFFICIENT_SAMPLE o NO_EVALUABLE segun corresponda.

---

## 6. Periodo futuro reservado (H-06: warm-up vs evaluacion)

**Inicio de la ventana confirmatoria:** fecha de firma efectiva de
esta v4. Datos anteriores a la firma se usan EXCLUSIVAMENTE como
calentamiento de indicadores (t_norm, MA200, W_v) para evaluar
eventos cuyo date_start es >= firma. NO se cuentan como episodios
confirmatorios.

**Distincion operativa:**
- **Warm-up:** sesiones anteriores a la firma. Se usan para
  calcular `t_norm` en la fecha `b-60` de un episodio con
  `date_start >= firma`. NO se usan como `date_start` de episodio
  confirmatorio.
- **Evaluacion:** sesiones desde la firma hasta completar 504
  sesiones evaluadas. Solo episodios con `date_start >= firma` se
  cuentan como confirmatorios.

**Regla de no retroactividad:** un episodio con `date_start` anterior
a la firma NO es confirmatorio, aunque su `date_confirmed` o ventana
futura caiga dentro de la ventana. Se registra como exploratorio.

**Horizonte:** 504 sesiones de trading desde la firma. Se registran
sesiones no disponibles (festivos, fallos) y el periodo se extiende
hasta completar 504 sesiones evaluadas.

**Huella al abrir:** al firmar, se registra en
holdout_registry.json (event_type=CONFIRMATORY_EXECUTION) con
experiment_manifest capturado via capture_experiment_env
--confirmatory. Contiene: lista de tickers fijada + hash de la
lista, fuentes de precios, HEAD SHA, SHA del contrato v4.

**Retornos futuros (63 sesiones):** si un episodio con
`date_start >= firma` tiene `date_confirmed` dentro de la ventana
pero el retorno a 63 sesiones se completa despues del cierre, se
extiende la ventana hasta completarlo, sin que ello consuma mas
sesiones de la ventana de evaluacion.

**Cierre:** al alcanzar 504 sesiones evaluadas y completar los
retornos pendientes, se cierra la ventana y se evaluan C1-C3. Si
N_episodios < 63, el resultado es INSUFFICIENT_SAMPLE o INCONCLUSO.
No se extiende la ventana tras ver resultados sin firma adicional.

---

## 7. Salida y persistencia

CSV outputs/audit/reaccumulation_events.csv con columnas:
    ticker, date_context, date_start, date_confirmed,
    S_b, R_b, w_b, tol_b, pullback_session,
    v_norm_p_pullback, v_norm_b_breakout,
    contexto_up, r1_res, r2_res, r3_res, r4_res, r5_res,
    evaluable, reason_not_evaluable

No se publica en outputs/history/*. No modifica el reporte diario.
No modifica classify_wyckoff_phase ni el orquestador.

REACCUMULATION_CONFIRMED es etiqueta del CSV experimental, no fase
del sistema.

---

## 8. Tests de contrato (T1-T7, corregido H-05)

- T1: R1 contexto UP en b-60 con info estrictamente anterior.
  Verifica que datos de b-60 o posteriores NO se usan para R1.
  Ademas, verifica el minimo H_R1=400: con menos sesiones, R1 debe
  ser NOT_EVALUABLE.
- T2: R2 consolidacion §2.4. Anchos >15% -> NOT_EVALUABLE.
- T3: R3 pullback §2.5. Verifica (i)+(ii)+(iii) en la misma
  sesion p. Verifica los tres estados: PASS, NOT_TRIGGERED,
  NOT_EVALUABLE.
- T4: R4 confirmacion §2.6. Verifica que C(b+1) o C(b+2) <=
  umbral -> NOT_TRIGGERED (sin emitir episodio).
- T5: R5 volumen de ruptura §2.7.
- T6 (**H-05 corregido**): invariante de aislamiento.
  Permitido explicitamente:
    - llamada read-only a `indicators.wyckoff_v1.detect_trend_context`
      (R1).
  Prohibido por AST:
    - import de `src.pipeline.*` (orquestador, writers).
    - import de `indicators.wyckoff_orchestrator` ni de
      `classify_wyckoff_phase` directa.
    - escritura en cualquier path fuera de
      `outputs/audit/reaccumulation_events.csv`.
  Verificacion:
    - AST de imports (listas blancas y negras).
    - monkeypatch sobre `open` y `Path.write_*` durante ejecucion
      de T7, verificando que solo se toco el CSV de auditoria.
- T7 (**H-01 corregido**): end-to-end de un evento positivo.
  Serie sintetica con calentamiento suficiente (>= 500 sesiones):
    * 400 sesiones planas (calentamiento de indicadores).
    * 100 sesiones alcistas sostenidas (UP claro).
    * Consolidacion de 40 sesiones con width <= 15%, al menos una
      sesion p en [S_b, S_b+tol_b] con v_norm_p(p) <= -0.3
      -> R2 + R3.
    * b: C(b) > R_b + tol_b y v_norm_b(b) >= +0.5 -> R5.
    * b+1, b+2: C > R_b + tol_b -> R4.
  **Aserción previa obligatoria:** antes de probar el evento, se
  verifica explicitamente que `detect_trend_context` devuelve
  `UP` en `b-60`. Si no, el test falla por construccion del
  sintetico, no por el detector.
  Verifica:
    * Etiqueta emitida exactamente en b+2 (no antes).
    * Modificar datos en b+3.. no cambia la etiqueta en b+2.
    * Ninguna escritura fuera del CSV de auditoria.
  Determinista: seed fija.

---

## 9. Secuencia aprobada

1. Revision de esta v4 por el auditor externo.
2. Firma expresa de v4.
3. Implementacion de T1-T7 con tests verdes.
4. Implementacion experimental aislada del detector.
5. Registro del periodo futuro en holdout_registry al firmar
   (CONFIRMATORY_EXECUTION, experiment_manifest via
   capture_experiment_env --confirmatory).
6. Espera hasta completar 504 sesiones de trading post-firma (mas
   extensiones por retornos 63 sesiones pendientes).
7. Evaluacion C1-C3 con §5.
8. Decision separada de produccion: solo si FUNCIONA y con firma
   adicional.

Sin firma de v4, no hay paso 3.

---

## 10. Estado normativo

Borrador v4. Sin validez normativa hasta firma del auditor. La
firma de v4 no autoriza incorporacion a produccion; autoriza
ejecutar T1-T7, el detector experimental sobre cohort (marcado
EXPLORATORIO), y abrir la ventana futura de §6 con los datos
congelados en la firma.

---

## 11. Resumen cambios v3 -> v4

| Hallazgo | Cambio v4 | Seccion |
|---|---|---|
| P0-H01 historial R1 + T7 | H_R1=400 antes de b-60; T7 con 400+100+40 y assert UP en b-60 | §2.3, §8 |
| P1-H02 busqueda episodios | Algoritmo cronologico: t++ por defecto, t=b+3 tras emitir; dedup por date_start | §3 |
| P1-H03 bootstrap C2 | Estacionario Politis-Romano, bloques L~Geometric(1/63), B=1000, N>=63, B_validas>=800 | §5.2 |
| P1-H04 datos insuficientes | Tres estados PASS / NOT_TRIGGERED / NOT_EVALUABLE; ventana V_v debe tener 40 validas o INVALID_DATA | §2.2, §4 |
| P1-H05 T6 vs R1 | T6 permite read-only detect_trend_context; prohibe src.pipeline.* y escritura fuera del CSV | §8 |
| P1-H06 warm-up vs evaluacion | Warm-up = solo init; evaluacion = post-firma; episodios con date_start < firma NO confirmatorios | §6 |

Fin del pre-registro DICT3-REACCUMULATION v4.