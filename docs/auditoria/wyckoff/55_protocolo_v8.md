# 55 - Protocolo v8

Fecha: 2026-10-05
Rama: gold-standard-v3 (a crear desde gold-standard-v2)
Estado: GO METODOLOGICO del auditor. Pendiente materializar.

---

## §0. Encuadre

- v7 abortado por capacidad (12/27 escenarios con e_se > 0.07 a n_B=800).
- v8-r4 aprobado por el auditor en su arquitectura.
- Este documento es la revision 5, incorporando las cuatro correcciones
  de cierre (A, B, C, D) exigidas en el dictamen.
- Identidad de rama: gold-standard-v3. Base congelada: gold-standard-v2.

---

## §1. Universo elegible

    data/stock_prices.parquet
    build_ticker_df (contrato implicito del detector)
    240 filas en indice hasta t
    200 filas warmup

N_frame = 243435 episodios. N_tickers = 301.

---

## §2. Estratos

Cuatro celdas por sector x periodo:

    C1 = (D=1, ctx=1)
    C2 = (D=1, ctx=0)
    C3 = (D=0, ctx=1)
    C4 = (D=0, ctx=0)

D = detect_sow(FROZEN_V19_PARAMS).
ctx = §3.
Sector = 12 (11 GICS + UNKNOWN).
Periodo = 2 (colapso determinista).

Total estratos potenciales = 4 x 12 x 2 = 96.

---

## §3. Variable de sobre-muestreo (ctx)

    ctx_t = 1  si y solo si:
      (a) MA50(t-60) > MA200(t-60)
      AND
      (b) (max(Close,[t-59,t]) - Close_t) / max(Close,[t-59,t]) >= 0.15

- ctx se calcula exclusivamente sobre OHLCV.
- No importa wyckoff_v1, detect_sow, config ni ningun otro modulo.
- ctx no genera Y ni modifica el detector.
- ctx no se muestra al anotador.

---

## §4. Probabilidades de inclusion

    p_i = n_h / N_h   dentro de cada estrato h

Diseno: muestreo aleatorio simple sin reemplazo (SRSWOR) dentro de
cada estrato.

Tabla {h: (N_h, n_h, p_h)} congelada antes del muestreo.

---

## §5. Asignacion de n por estrato

Paso 0 - Factibilidad previa de minimos.

    Para cada celda Cj en {C1, C2, C3, C4}:
        para cada estrato h en Cj:
            n_h_min(h) = 5 si N_h >= 5
                       = N_h si N_h < 5   (censo)
        verificar: suma n_h_min(h) <= 200

    Si falla para alguna Cj: preflight FAIL / SUSPENSION

Paso 1. Aplicar n_h_min(h) a todo estrato.
Paso 2. Calcular remanente R_Cj = 200 - suma n_h_min(h) por celda.
Paso 3. Distribuir R_Cj proporcionalmente a N_h con largest remainder
        (Hamilton).
Paso 4. Verificar suma n_h = 200 exacto en cada celda Cj.
Paso 5. Verificar n_h <= N_h en todos los estratos.

Total esperado: C1 = C2 = C3 = C4 = 200. n_total = 800.

Si tras el colapso temporal un estrato queda con N_h < 5: censo
(n_h = N_h, peso 1). No se intenta bootstrap con menos de 5 unidades.
---

## §6. Pesos

    w_i = N_h(i) / n_h(i)

Sin ajustes post-hoc. Se aplican directamente en los estimadores §7-§10.

---

## §7-§10. Estimadores

Hajek ponderados:

    Se_hat  = sum w_i [D=1, Y=1] / sum w_i [Y=1]
    Sp_hat  = sum w_i [D=0, Y=0] / sum w_i [Y=0]
    PPV_hat = sum w_i [D=1, Y=1] / sum w_i [D=1]
    NPV_hat = sum w_i [D=0, Y=0] / sum w_i [D=0]

Y = etiqueta adjudicada del panel (mayoria 2/3).
IC95% por Rao-Wu-Yue rescaled bootstrap (§12).

---

## §11. Metodo de varianza

Rao-Wu-Yue rescaled bootstrap. SeedSequence por replica.

---

## §12. Bootstrap

### §12.1. Estratos no censales (N_h >= 5)

    m_h = n_h - 1
    f_h = n_h / N_h
    lambda_h = sqrt(m_h * (1 - f_h) / (n_h - 1))
    r_hi* = multiplicidad de seleccion
    w_hi* = [1 - lambda_h + lambda_h * (n_h/m_h) * r_hi*] * w_hi

### §12.2. Estratos censales (n_h = N_h)

    f_h = 1
    peso w_i = 1 (constante, no normalizado por n_h)
    NO se calcula lambda_h.
    Las unidades censadas permanecen SIN MODIFICACION en todas las
    replicas. Contribucion fija. No participan en el componente de
    variabilidad bootstrap.

Justificacion: con f_h = 1 no queda variabilidad de muestreo dentro
del estrato.

Regla obligatoria: si n_h == N_h (censo por cualquier motivo),
aplicar §12.2.

---

## §13. FPC

El FPC se incorpora exclusivamente mediante lambda_h en RWY (termino
(1 - f_h)).

No se aplica correccion FPC adicional sobre la varianza bootstrap
resultante.

---

## §14. Targets de error

    e(Se)  <= 0.07
    e(Sp)  <= 0.07
    e(PPV) <= 0.08
    e(NPV) <= 0.08

Con e(theta) = (UCI95(theta) - LCI95(theta)) / 2.

Terminologia (correccion C): PASS mide criterio de precision del
intervalo, no validacion de cobertura frecuentista. La palabra
"IC95%" se refiere al procedimiento nominal; el gate verifica
anchura/precision, no cobertura.
---

## §15. Escenarios de simulacion

### §15.1. Prevalencia observada de D

    q_D = 2559 / 243435 = 0.01051204630...

### §15.2. Restricciones matematicas

    P(D=1) = pi_Y * Se + (1 - pi_Y) * (1 - Sp)

Con P(D=1) = q_D fijado:
    (a) pi_Y <= q_D / Se
    (b) Sp >= 1 - q_D ~ 0.9895
    (c) Sp = 1 - (q_D - pi_Y * Se) / (1 - pi_Y), en [0, 1]

### §15.3. Gate principal - cuadricula factible ex ante

    pi_Y en {0.005, 0.008, 0.010, 0.012}
    Se   en {0.60, 0.70, 0.80}

|S_factible| = 12 > 0.

Justificacion: rango predefinido de prevalencias verdaderas compatible
con q_D bajo la hipotesis Se >= 0.60. No se describe como "rango
plausible" (no es medible).

Regla obligatoria:

    Si |S_factible| == 0: protocolo INVALIDO, preflight FAIL, SUSPENSION.

### §15.4. Stress test no-gate

    Entrada: (pi_Y, Se, Sp) de la cuadricula 45 combinaciones:
        pi_Y en {0.005, 0.008, 0.010, 0.012, 0.020, 0.050}
        Se   en {0.60, 0.70, 0.80}
        Sp   en {0.60, 0.70, 0.80}

    Del frame real:
        q_D = N_D / N
        r1  = P(ctx=1 | D=1) observada
        r0  = P(ctx=1 | D=0) observada

    Generacion de poblacion sintetica:
        N_syn = 100000 (fijo, no negociable)
        q_D_syn = pi_Y * Se + (1 - pi_Y) * (1 - Sp)

        N_C1 = q_D_syn * r1 * N_syn
        N_C2 = q_D_syn * (1 - r1) * N_syn
        N_C3 = (1 - q_D_syn) * r0 * N_syn
        N_C4 = (1 - q_D_syn) * (1 - r0) * N_syn

        Redondeo: largest remainder global sobre los 4 N_Cj hasta
        forzar suma = N_syn.

        Dentro de cada celda: subdividir por sector x periodo con
        proporciones del frame real. Redondeo: largest remainder
        por celda.

        Asignacion de Y (§17.2).

        Muestreo: cuotas n_h del gate principal.
                  Si n_h > N_Cj_syn para algun estrato: escenario
                  "no simulable", se reporta.

        Calculo de metricas ponderadas.
        B = 500 replicas bootstrap.

    Salida: tabla de errores empiricos por escenario, etiquetada
            STRESS_TEST_MODEL_BASED, SIN PASS/FAIL.

    No sustituye al gate principal.

---

## §16. Niveles de simulacion

### §16.1. Dos niveles

    Nivel 1 - Monte Carlo de diseno (R_MC):
        muestras independientes SRSWOR del mismo diseno.

    Nivel 2 - Bootstrap RWY (B):
        replicas dentro de cada muestra para estimar el IC.

### §16.2. Valores

    SCREENING:      R_MC = 200,  B = 500
    CONFIRMACION:   R_MC = 500,  B = 2000

Screening solo identifica candidatos de n. Confirmacion decide.

### §16.3. Estructura

    para cada escenario:
        para r = 1..R_MC:
            pob_syn_r = generar_poblacion(escenario)   # §17.2
            muestra_r = SRSWOR(pob_syn_r, cuotas n_h)
            estimadores_r = Hajek(muestra_r)
            para b = 1..B:
                replica_rb = RWY(muestra_r)
            e_theta_r = (q97.5 - q2.5) / 2 sobre replicas validas
        distribucion_MC(e_theta) = {e_theta_r}_{r=1..R_MC}

### §16.4. Seeds (correccion D)

    master_seed = 20261006 (congelada)

    SeedSequence(master_seed).spawn(3):
        stream_0: MC (seleccion de muestras SRSWOR)
        stream_1: bootstrap RWY (deteccion de replicas)
        stream_2: SRSWOR dentro de replicas bootstrap si aplica

    Ningun stream se reutiliza entre niveles.
    Semillas registradas en manifest del run de simulacion.
---

## §17. Criterio PASS/FAIL

### §17.1. Pre-condicion

    Si |S_factible| == 0: protocolo INVALIDO, SUSPENSION.

### §17.2. Generacion de Y (regla congelada)

Correccion A: conteos enteros deterministas con redondeo global y
largest remainder. NO round() por celda.

    Para D=1:
        total_TP = round(sum N_Cj para j en {C1,C2} * Se)
        distribuir total_TP entre C1 y C2 mediante largest remainder,
        respetando tamanos N_C1, N_C2.
        -> n_TP_C1, n_TP_C2

    Para D=0:
        total_FP = round(sum N_Cj para j en {C3,C4} * (1-Sp))
        distribuir total_FP entre C3 y C4 mediante largest remainder,
        respetando tamanos N_C3, N_C4.
        -> n_FP_C3, n_FP_C4

    En C1: n_Y1 = n_TP_C1, n_Y0 = N_C1 - n_TP_C1
    En C2: n_Y1 = n_TP_C2, n_Y0 = N_C2 - n_TP_C2
    En C3: n_Y1 = n_FP_C3, n_Y0 = N_C3 - n_FP_C3
    En C4: n_Y1 = n_FP_C4, n_Y0 = N_C4 - n_FP_C4

La palabra "exactos" se sustituye por "conteos enteros deterministas,
con redondeo global y asignacion largest remainder".

El implementador no puede elegir entre Bernoulli, round() por celda,
floor, ceiling u otro. La regla queda congelada.

### §17.3. Replicas bootstrap con denominador nulo

    Denominadores:
        Se:  sum w_i [Y=1]
        Sp:  sum w_i [Y=0]
        PPV: sum w_i [D=1]
        NPV: sum w_i [D=0]

    Si denominador == 0 en replica b: replica b INVALIDA para ese
    estimador.

    Registro obligatorio por escenario y estimador:
        n_replicas_validas
        n_replicas_invalidas
        invalid_rate = n_invalidas / B

    Reglas:
        NO imputar. NO sustituir por cero. NO eliminar silenciosamente.

    Umbral: invalid_rate <= 0.01 (B=2000 -> max 20 invalidas)
    Si invalid_rate > 0.01: escenario FAIL.

    Percentiles: SOLO sobre replicas validas, dejando constancia del
    n efectivo.

    Una replica invalida para Se no contamina PPV si PPV esta definida
    en esa misma replica.

### §17.4. PASS por escenario y estimador

    para cada r en {1..R_MC}:
        e_theta_r = (q97.5_r - q2.5_r) / 2

    q95_MC(e_theta) = percentil 95 de {e_theta_r}

    PASS por escenario y estimador:
        q95_MC(e_Se)  <= 0.07
        q95_MC(e_Sp)  <= 0.07
        q95_MC(e_PPV) <= 0.08
        q95_MC(e_NPV) <= 0.08

Equivale a: P(e_theta <= target) >= 0.95 estimado por Monte Carlo.

### §17.5. Busqueda de n_total (correccion B)

Regla congelada: no se modifica la muestra durante la anotacion.
El dimensionamiento es ex ante.

    1. Evaluar n_total = 800 (nominal, C1=C2=C3=C4=200).

    2. Si 800 PASS: diseno aprobado = 800.

    3. Si 800 FAIL:
         buscar el menor n_total <= 800 que satisface §17.4 para todos
         los escenarios.
         parametrizacion: n_total = 4K, K entero, 5 <= K <= 200.
         distribucion de n_h por celda segun §5 (misma regla
         determinista).

         Si n_min encontrado <= 800:
             redimensionar protocolo antes de anotacion.
             repetir confirmacion con R_MC=500, B=2000.
             si PASS: diseno aprobado = n_min.

         Si n_min > 800:
             SUSPENSION.

### §17.6. Criterio PASS global

    PASS global si:
        - |S_factible| > 0
        - Todos los escenarios de S_factible cumplen §17.4
        - invalid_rate <= 0.01 en todos los escenarios y estimadores
        - Factibilidad §5 Paso 0 verificada
        - Terminologia §14 respetada (precision, no cobertura)

### §17.7. Flujo completo

    1. Calcular q_D del frame real.
    2. Calcular |S_factible| segun §15.3. Si == 0: SUSPENSION.
    3. Verificar §5 Paso 0 en cada celda. Si falla: SUSPENSION.
    4. Para cada escenario de S_factible:
       a. R_MC muestras SRSWOR independientes (stream MC).
       b. Para cada muestra: B replicas bootstrap RWY (stream
          bootstrap).
       c. e_theta_r por muestra.
       d. q95_MC(e_theta) sobre las R_MC.
       e. invalid_rate acumulado.
    5. Si todos los escenarios cumplen §17.4 y §17.6: PASS.
    6. Si falla con n_total=800: buscar menor n_total <= 800 (§17.5).
       Si n_min > 800: SUSPENSION.

---

## §18. Techo maximo

    800 UNIDADES UNICAS por anotador.
    3 anotadores x 800 unidades = 2400 actos de anotacion.

Si n requerido > 800: SUSPENSION.
No se relaja ningun umbral.

---

## §19. Ceguera del anotador

Los anotadores son ciegos a:

- D (resultado del detector).
- ctx.
- Celda (C1/C2/C3/C4).
- Peso (w_i).
- Si la unidad provino de sobremuestreo o muestreo proporcional.
- Sector y periodo del caso individual.
- Cualquier etiqueta derivada que permita inferirlos indirectamente.

La interfaz de anotacion recibe unicamente la informacion necesaria
para adjudicar Y.
---

## §20. Cambios respecto a v7

| Aspecto | v7 | v8 |
|---|---|---|
| Muestra A | 200+200 enriquecida | Eliminada |
| Muestra B | 200 representativa | 800 estratificada 4 celdas |
| Estratificacion | sector x periodo | celda x sector x periodo |
| Escenarios gate | pi en {0.10,0.20,0.30} | condicionados a q_D |
| Terminologia | representativa | estratificada con p inclusion |
| FPC | no especificado | solo via lambda_h en RWY |
| n_h minimo | 2 | 5 (o censo si N_h < 5) |
| Censo si N_h < 5 | no | si |
| MC externo | no | R_MC = 200/500 |
| PASS terminologia | no especificada | criterio de precision |
| Streams aleatorios | no especificados | separados MC/bootstrap |

---

## §21. Prohibiciones vigentes

    NO tocar wyckoff_v1.py
    NO tocar legacy
    NO tocar config
    NO tocar protocolo v3 firmado
    NO tocar gold-standard-v1 (congelada)
    NO tocar gold-standard-v2 (snapshot aborto v7)
    NO migrar consumidores
    NO merge a main

---

## §22. Autorizacion del auditor

GO METODOLOGICO v8-r4 con cuatro correcciones de cierre incorporadas
en r5:

| # | Correccion | Estado en r5 |
|---|---|---|
| A | §17.2 redondeo global + largest remainder | Incorporada |
| B | §17.5 buscar menor n_total <= 800 | Incorporada |
| C | Terminologia: PASS = precision, no cobertura | Incorporada |
| D | §16.4 streams separados MC / bootstrap | Incorporada |

Tras r5: firmable sin nueva revision conceptual.

---

## §23. Plan de implementacion tras firma

    1. Crear rama gold-standard-v3 desde gold-standard-v2.
    2. Tag/anotacion de congelacion en gold-standard-v2.

    3. Modulos nuevos:
       scripts/gold_standard/ctx.py
       scripts/gold_standard/sampling_b_v8.py
       scripts/gold_standard/power_v8.py
       scripts/gold_standard/run_capa1.py (modificado, sin Muestra A)
       scripts/gold_standard/verify_paquetes_v8.py

    4. Tests por modulo.

    5. Preflight v8 sobre datos reales.

    6. Si preflight PASS: run_capa1.py --go.

    7. Requiere: 3 anotadores humanos disponibles.

---

Fin del protocolo v8.