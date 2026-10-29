# PRE-REGISTRO DICT3-REACCUMULATION v3

**Estado:** v3. Pendiente de revision y firma del auditor externo.
**Fecha:** 2026-10-29.
**Ciclo:** sesion 16.
**Frente:** DICT3-REACCUMULATION (REGISTRO.yaml, OPEN).
**Base:** v2 (commit 69baa2c2) + dictamen externo 2026-10-29 (chat).
**Regla:** los parametros de esta v3 quedan congelados antes de
ejecutar el experimento. No se ajustan tras ver resultados.

Responde a los 7 hallazgos del segundo dictamen (H-01 R3/R4
incompatibles; H-02 C1 inviable en ventana futura; H-03 C2 sin
metodo unico; H-04 v_norm incompleto; H-05 interfaz contexto;
H-06 elegibilidad C1 omite R5; H-07 falta T7 end-to-end).

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

## 1. Estado del holdout (H-01 dictamen v1, sin cambios)

- cohort y r3000 fueron consumidos por SOW-V6 el 2026-10-06.
- Registrados en holdout_registry.json como
  CONSUMED_FOR_CANDIDATE(SOW-V6), event_type=EXPLORATORY_DESIGN_INFLUENCE.
- Poblacion: composicion 2026Q4 del S&P 500 aplicada retrospectivamente.
- Resultado sobre estos parquets: EXPLORATORIO o evidencia secundaria,
  nunca confirmacion independiente.
- Confirmacion real requiere periodo futuro reservado (§10).

Restriccion ejecutable: capture_experiment_env --confirmatory sobre
estos parquets con la candidata REACCUMULATION se registra como
EXPLORATORY. No hay via para presentarlo como confirmatorio.

---

## 2. Reglas R1-R5 (consolidadas)

### 2.1. Notacion base

Sea C(t) la serie de cierres (Close ajustado por splits). Sea
V(t) el volumen. W = 40 sesiones (ventana de consolidacion).
W_v = 40 sesiones (ventana de normalizacion de volumen).
L_prev = 60 sesiones (ventana de contexto).

Sean S_b, R_b, w_b, tol_b definidos en §4 una vez fijado el anclaje
b de la ruptura.

### 2.2. Normalizacion de volumen (v_norm) — H-04

Definicion UNICA (v2 dejaba abierto el estimador y el caso de
varianza nula). Para cualquier sesion t y ventana V_v:

    base(t) = media( V(s) para s en [t-W_v, t-1] )
    sd(t)   = std( V(s) para s en [t-W_v, t-1], ddof=0 )
    v_norm(t) = ( V(t) - base(t) ) / sd(t)

Casos no evaluables (devuelven None, no un valor):
- Hay menos de W_v/2 = 20 sesiones validas en [t-W_v, t-1].
- Alguna de las sesiones de [t-W_v, t-1] tiene V(s) NaN o <= 0.
- sd(t) == 0 (todos los volumenes iguales). No se aproxima por
  1e-9; el evento no es evaluable.

La ventana V_v usada para pullback (R3) es [p-W_v, p-1]. La usada
para ruptura (R5) es [b-W, b-1]. Son ventanas diferentes, con la
misma formula. Distintas del detector Wyckoff existente: aqui v_norm
se calcula sobre log-return de volumen? No. Sobre V(t) bruto. El
detector existente usa robust_zscore sobre volumen; NO son
matematicamente identicas. Se declara la diferencia.

### 2.3. R1: contexto UP

interfaz_efectiva:
  modulo: indicators.wyckoff_v1
  funcion: detect_trend_context
  firma: (df_ticker, ticker, as_of) -> str
  invocacion: detect_trend_context(df_ticker, ticker, as_of=b-60)
  datos_usados: OHLCV del ticker con fecha ESTRICTAMENTE anterior
                a b-60 (t_norm se calcula internamente sobre ventana
                previa del propio modulo).
  valores: UP, DOWN, NEUTRAL, UNKNOWN.
  evaluable_si: hay al menos 120 sesiones de historial antes de b-60.
  no_evaluable_si: menos de 120 sesiones -> se excluye el ticker del
                   denominador de C1. NO se trata como "no UP".

R1 exige contexto == UP. No se admite "UNKNOWN" como UP por
defecto.

### 2.4. R2: consolidacion

Sea b el ancla de ruptura (§3). Sobre [b-W, b-1]:

    S_b = min( C(b-W), ..., C(b-1) )
    R_b = max( C(b-W), ..., C(b-1) )
    w_b = R_b - S_b
    tol_b = 0.05 * w_b

R2 exige:
    (a) S_b > 0, R_b > S_b, w_b > 0
    (b) w_b / R_b <= 0.15
    (c) todas las sesiones de [b-W, b-1] con C(s) > 0 y no NaN
    (d) existe al menos un pullback segun R3 (§2.5)

Si falla (a), (b) o (c): no evaluable. Si falla (d): no episodio
(R3 es obligatoria).

### 2.5. R3: pullback en consolidacion (CORREGIDO)

**Cambio v2 -> v3:** en v2, R3 exigia pullback en [b+1, b+2]. Eso es
incompatible con R4 (b+1, b+2 deben estar POR ENCIMA de R_b+tol_b;
S_b < R_b implica S_b+tol_b < R_b+tol_b, luego ningun cierre puede
satisfacer ambas). En v3, R3 se evalua ANTES de la ruptura, dentro
de la consolidacion.

Definicion:
Un pullback es cualquier sesion p en [b-W, b-1] tal que:
    (i)  C(p) <= S_b + tol_b
    (ii) v_norm_p(p) <= -0.3, donde v_norm_p usa ventana [p-W_v, p-1]
         segun §2.2.
    (iii) C(p) y V(p) validos (no NaN, > 0)

R3 exige que exista al menos un p asi.

No se exige que p sea distinto del argmin (S_b). Si el propio minimo
cayo con volumen bajo (v_norm <= -0.3), es un pullback valido: es la
prueba de oferta clasica "no hay oferta en el minimo".

Si no existe tal p: R3 falla, no hay episodio (aunque R2, R4 y R5 se
cumplan).

---

## 3. Anclaje de ruptura y confirmacion (H-01 dictamen v1, sin cambios)

Sea b la PRIMERA sesion tal que:
    C(b) > R_b + tol_b

A partir de b:
    C(b)   > R_b + tol_b   (por definicion de b)
    C(b+1) > R_b + tol_b
    C(b+2) > R_b + tol_b

**El evento NO existe antes de b+2.** La etiqueta
REACCUMULATION_CONFIRMED solo se emite cuando ya hay tres cierres
consecutivos por encima del umbral congelado.

Fechas:
    date_context   = b - 60      (fecha de evaluacion R1)
    date_start     = b           (primera ruptura)
    date_confirmed = b+2         (confirmacion)

C2 (§8) evalua desde date_confirmed.

---

## 4. R4: confirmacion de ruptura

R4 exige los tres cierres de §3. Si C(b+1) o C(b+2) caen por debajo
o igual a R_b + tol_b, no hay episodio (aunque b hubiera roto).

---

## 5. R5: volumen de ruptura

R5 exige:
    v_norm_b(b) >= +0.5, con v_norm_b usando ventana [b-W, b-1]
    segun §2.2.

Justificacion de umbrales (corregida, dictamen v1):
- -0.3 es un umbral NUEVO para "volumen bajo en pullback". NO es
  SC_V_NORM_THRESHOLD (que vale 0.5). Corresponde a
  SC_C_NORM_THRESHOLD (compresion), no a volumen.
- +0.5 coincide numericamente con SC_V_NORM_THRESHOLD, pero NO es
  su "inversion". Es coincidencia de magnitud.

Ambos valores congelados aqui. No se optimizan tras ver resultados.

---

## 6. Criterios C1-C3 (corregidos)

### 6.1. C1: frecuencia (H-02 corregido)

**Cambio v2 -> v3:** en v2, C1 por años naturales con exclusion de
años parciales. En v3, chunks no solapados de 252 sesiones. Permite
evaluar C1 en la ventana futura de 504 sesiones (§10) y en la
retrospectiva (9 años -> 9 chunks aprox).

Definicion:
- Periodo de evaluacion [t_ini, t_fin].
- Se divide en chunks consecutivos no solapados de 252 sesiones.
  El ultimo chunk incompleto se descarta.
- Para cada chunk:
    n_eval      = tickers con historial suficiente para evaluar
                  R1-R5 (§6.1.1)
    n_episodios = tickers con >= 1 episodio cuyo date_start cae
                  dentro del chunk
    ratio       = n_episodios / n_eval
- C1 exige:
    (a) n_chunks_evaluados >= 2
    (b) para TODOS los chunks: ratio en [0.01, 0.30]
- Si n_chunks_evaluados < 2: C1 = INSUFFICIENT_SAMPLE.

#### 6.1.1. Elegibilidad (H-06)

Un ticker es evaluable en un chunk si:
- Tiene historial suficiente para R1 (>= 120 sesiones pre-b-60) en
  al menos una de las sesiones del chunk.
- Tiene historial suficiente para R2/R3/R4/R5 (>= W + 3 sesiones)
  en esa misma sesion candidata.

**No basta con R1-R4.** Se exige R1-R5 (§2). Si un ticker podria
satisfacer R1-R4 pero no R5 (p.ej. sin volumen valido en [b-W, b-1]),
no cuenta como evaluable.

#### 6.1.2. Asignacion de año (H-06)

Si date_start y date_confirmed caen en chunks distintos, el
episodio se asigna al chunk de **date_start**. Misma regla para
todos los resultados.

### 6.2. C2: tasa de acierto (H-03 congelada)

**Cambio v2 -> v3:** en v2 se permitia Wilson O bootstrap (abierto).
En v3 se congela **bootstrap por mes-bloque**. Razon: episodios de
distintos tickers pueden compartir dias y por tanto movimientos
comunes; un IC por Wilson sobre episodios asume independencia que
no existe. El bootstrap por bloque (mes calendario) remuestrea
bloques completos, absorbiendo la dependencia intra-mes.

Definicion congelada:
- Retorno de un episodio: r(e) = C(date_confirmed + 63) / C(date_confirmed) - 1
  (Close price return, splits ajustados, dividendos no reinvertidos).
- Exito: r(e) >= 0.05.
- Episodios evaluables: los que tienen 63 sesiones futuras
  disponibles en el momento de la evaluacion.
- Solapamiento intra-ticker: si date_confirmed(e1) y date_confirmed(e2)
  del mismo ticker estan a < 63 sesiones, se colapsan en uno solo
  (se conserva el primero por date_confirmed).
- Bloque = mes calendario de date_confirmed.
- Bootstrap por bloque: B=1000, seed=42, remuestreo con reemplazo
  de bloques completos.
- p_hat = tasa puntual de exitos (episodios, no bloques).
- IC95_lower = percentil 2.5 del bootstrap de p_hat por bloques.

C2 exige:
    (a) N_episodios_evaluables >= 30
    (b) p_hat >= 0.60
    (c) IC95_lower > 0.50
- Si N < 30: C2 = INSUFFICIENT_SAMPLE.

Alternativa (Wilson) rechazada explicitamente: no captura la
dependencia entre tickers en el mismo periodo.

### 6.3. C3: solapamiento con acumulacion (corregido)

**Cambio v2 -> v3:** denominador incluye TODOS los episodios
evaluables para esta comparacion, aunque no exista acumulacion
confirmada en el periodo.

Definicion:
- Episodio de reacumulacion: [date_start, date_confirmed].
- Acumulacion confirmada en el mismo ticker: intervalo [SC_date, SOS_date]
  segun contrato vigente de acumulacion.
- Solapa(e): existe interseccion temporal no vacia entre los dos
  intervalos.
- n_eval_C3 = episodios de reacumulacion con acumulacion evaluable
  en el mismo periodo (es decir, podemos determinar si hay o no
  interseccion).
- n_solapa = episodios con Solapa(e) == True.
- ratio_C3 = n_solapa / n_eval_C3.

C3 exige:
    (a) n_eval_C3 >= 30
    (b) ratio_C3 <= 0.10
- Si n_eval_C3 < 30: C3 = NO_EVALUABLE.
- Si n_eval_C3 >= 30 y n_solapa = 0: ratio_C3 = 0. Es exito
  (no hay solapamiento). NO es lo mismo que "no evaluable".
- Si n_eval_C3 < 30 con episodios de reacumulacion existentes pero
  sin historial de acumulacion evaluable en el periodo: NO_EVALUABLE
  (no se asume 0).

### 6.4. Aceptacion global

C1, C2 y C3 deben ser EVALUABLES y SUPERADOS. Un
INSUFFICIENT_SAMPLE o NO_EVALUABLE no cuenta como "pasa por
defecto". Si alguno no es evaluable, el veredicto global es
INSUFFICIENT_SAMPLE o NO_EVALUABLE segun corresponda. No se
declara FUNCIONA sin los tres.

---

## 7. Periodo futuro reservado (H-02 + condiciones adicionales)

**Inicio:** la fecha de firma efectiva de esta v3. Datos anteriores
a la firma NO se incorporan retroactivamente como si hubieran estado
reservados desde antes.

**Duracion:** 504 sesiones de trading contadas desde la firma. Se
registran las sesiones no disponibles (festivos, fallos del sistema)
y el periodo se extiende hasta completar 504 sesiones evaluadas.

**Huella al abrir:** en la fecha de firma se registra en
holdout_registry.json (event_type=CONFIRMATORY_EXECUTION) con
experiment_manifest capturado ese dia. Contiene: lista de tickers
fijada, hash de la lista, fuentes de precios, version del codigo
(HEAD SHA), version del contrato (v3 + SHA del doc).

**Regla de no retroactividad:** el periodo confirmatorio no incluye
datos anteriores a la firma. Si la firma es 2027-03-15, la ventana
empieza 2027-03-16 y no usa datos de 2026.

**Cierre:** al alcanzar 504 sesiones evaluadas, se cierra la
ventana y se evaluan C1-C3. Si N_episodios < 30, el resultado es
INSUFFICIENT_SAMPLE o INCONCLUSO. No se extiende la ventana tras
ver resultados sin una nueva firma explicita.

---

## 8. Salida y persistencia

Salida del detector: CSV outputs/audit/reaccumulation_events.csv
con columnas:
    ticker, date_context, date_start, date_confirmed,
    S_b, R_b, w_b, tol_b, pullback_session,
    v_norm_p_pullback, v_norm_b_breakout,
    contexto_up, evaluable, reason_not_evaluable

No se publica en outputs/history/*. No modifica el reporte diario.
No modifica classify_wyckoff_phase ni el orquestador. CSV aislado.

REACCUMULATION_CONFIRMED es etiqueta del CSV experimental, no fase
del sistema.

---

## 9. Tests de contrato (T1-T7)

- T1: R1 contexto UP en b-60, con info estrictamente anterior.
  Verifica que datos de b-60 o posteriores NO se usan para R1.
- T2: R2 consolidacion sobre [b-W, b-1], formulas §2.4.
  Anchos > 15% -> no evaluable.
- T3: R3 pullback en [b-W, b-1] con v_norm §2.2.
  Verifica que la sesion p cumple C(p) <= S_b + tol_b y
  v_norm <= -0.3 simultaneamente.
- T4: R4 confirmacion en b, b+1, b+2 sobre umbral congelado.
  Cualquier cierre <= R_b + tol_b en b+1 o b+2 -> no episodio.
- T5: R5 volumen de ruptura en b con ventana [b-W, b-1],
  v_norm >= +0.5.
- T6: invariante de aislamiento. El detector no toca
  classify_wyckoff_phase, ni el orquestador, ni outputs/history/*,
  ni ningun CSV productivo. Verificacion por ausencia de escritura
  y por AST (no importa modulos de produccion).
- T7 (nuevo, H-07): **end-to-end de un evento positivo.**
  Serie sintetica que satisface R1-R5 simultaneamente:
    * 120 sesiones planas (contexto previo)
    * 60 sesiones con tendencia alcista clara -> R1 UP
    * 40 sesiones con consolidacion width <= 15%, con al menos una
      sesion p con C(p) en [S_b, S_b+tol_b] y v_norm_p(p) <= -0.3
      -> R2 y R3
    * Sesion b con C(b) > R_b + tol_b y v_norm_b(b) >= +0.5 -> R5
    * Sesiones b+1, b+2 con C > R_b + tol_b -> R4
  Verifica:
    * El detector emite REACCUMULATION_CONFIRMED exactamente en
      b+2.
    * NO emite nada en b, b+1, o antes.
    * Modificar datos en b+3, b+4, ... no cambia la etiqueta en b+2
      (sin look-ahead retroactivo).
  Determinista: seed fija.

---

## 10. Secuencia aprobada (dictamen)

1. Revision de esta v3 por el auditor externo.
2. Firma expresa de v3.
3. Implementacion de T1-T7 con tests verdes.
4. Implementacion experimental aislada del detector.
5. Registro del periodo futuro en holdout_registry al firmar
   (event_type=CONFIRMATORY_EXECUTION, experiment_manifest
   capturado con capture_experiment_env --confirmatory).
6. Espera hasta completar 504 sesiones de trading post-firma.
7. Evaluacion C1-C3 con los denominadores y metodos de §6.
8. Decision separada de produccion: solo si FUNCIONA y con firma
   adicional.

Sin firma de v3, no hay paso 3.

---

## 11. Estado normativo

Borrador v3. Sin validez normativa hasta firma del auditor. La
firma de v3 no autoriza incorporacion a produccion; autoriza ejecutar
T1-T7, el detector experimental sobre cohort (resultado marcado
EXPLORATORIO), y abrir la ventana futura de §7 con los datos
congelados en la firma.

---

## 12. Resumen cambios v2 -> v3

| Hallazgo | Cambio v3 | Seccion |
|---|---|---|
| H-01 R3/R4 incompatibles | R3 se mueve a [b-W, b-1]; b, b+1, b+2 reservados a R4 | §2.5 |
| H-02 C1 inviable en futuro | Chunks de 252 sesiones, >= 2 chunks, ventana futura = 504 sesiones | §6.1, §7 |
| H-03 C2 sin metodo unico | Congelado bootstrap por mes-bloque (B=1000, seed=42); rechaza Wilson por dependencia | §6.2 |
| H-04 v_norm incompleto | Formula con ventana, media, std, ddof=0, caso sd=0 y datos insuficientes como None | §2.2 |
| H-05 interfaz contexto | Declarada detect_trend_context(df_ticker, ticker, as_of=b-60); historial >= 120 sesiones o no evaluable | §2.3 |
| H-06 C1 elegibilidad omite R5 | Elegibilidad exige R1-R5; asignacion del episodio por date_start | §6.1.1, §6.1.2 |
| H-07 falta T7 end-to-end | Anadido T7 con serie sintetica que satisface R1-R5 y verifica etiqueta exacta en b+2 sin look-ahead | §9 |

Fin del pre-registro DICT3-REACCUMULATION v3.