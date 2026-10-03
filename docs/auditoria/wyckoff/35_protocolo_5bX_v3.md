# PROTOCOLO FASE 5b.X (v3) - Validacion out-of-sample de la candidata SOW

**Documento normativo. Define la validacion futura de la candidata.**
**Version:** v3 (2026-10-03). **Supersede operativamente:**
`35_protocolo_5bX.md` v2 (2026-10-03).
**Precedente:** dictamen externo 2026-10-03 sobre D-06
(`37_d06_5bX_invalidada.md`) + dictamen externo 2026-10-02 sobre 35 v1 +
QA 36 + dictamen `33_dictamen_5b4bis_v2.md` + contrato v1.9.
**Estado:** PENDIENTE DE IMPLEMENTACION Y EJECUCION. Bloqueado por dos
causas independientes: (a) datos posteriores al cutoff 2026-10-01 con
muestra suficiente; (b) script v3 conforme pendiente de implementar.

> **NOTA DE REEMPLAZO.** Este documento **reemplaza operativamente** a
> `35_protocolo_5bX.md` v2. No modifica los resultados historicos de
> 5b.3 / 5b.4 / 5b.4-bis. La candidata v1.9 y el contrato v1.9 no se
> tocan. El v2 queda archivado como referencia historica y su script
> congelado (`CC01A6AB...`) queda preservado como freeze historico no
> valido como paquete ejecutable.

---

## 0. Proposito

Validar out-of-sample la candidata SOW congelada en el contrato v1.9:

    N=60, M=30, X_ATR=0.25, Y_VOL=1.10

sobre datos temporalmente posteriores al cutoff 2026-10-01, con la misma
implementacion y sin recalibracion.

**Objetivo:** determinar si el lift +0.074 observado en 5b.4-bis
sobrevive fuera de la muestra de seleccion, o si es artefacto de una
busqueda sobre 240 configuraciones.

---

## 1. Estado de partida

    v1.8              CONTRATO PRODUCTIVO VIGENTE
    v1.9              CANDIDATA FROZEN_FOR_VALIDATION
    config SOW        None (fail-closed)
    5b.4-bis          PASS desarrollo / NO confirmado
    5b.X v2           INVALIDADA como paquete ejecutable (D-06)
    5b.X v3           ESTE DOCUMENTO (pendiente implementacion + datos)
    5c.4, 5d          BLOQUEADAS
    Legacy            INTACTO

**Cutoff temporal:** 2026-10-01 (ultima fecha del historico de
desarrollo). Universo OOS: `t0 >= 2026-10-02` (estrictamente posterior
al cutoff). Ver §3.1.

---

## 2. Condiciones previas a la ejecucion

5b.X v3 NO puede ejecutarse hasta que se cumplan las condiciones
siguientes, evaluadas de forma independiente. La mera existencia de
tiempo calendario NO es suficiente.

### 2.1. Umbral de muestra (CONGELADO)

    n_confirmed_H20_complete >= 550

**Definicion de "H20_complete":** episodio confirmado (SOW en
[t0+1, L]) cuyo landmark L permite calcular el outcome primario
struct_deterioration_H20, es decir, tal que:

    L_idx + 20 < len(dates) del ticker en el momento del corte de datos

Equivalente operativo, con M=30:

    t0_idx + M + 20 < len(dates)  =>  t0_idx + 50 < len(dates)

**No cuenta como confirmed elegible:** un episodio con SOW en ventana
pero cuyo H20 cae fuera de la serie disponible. La censura H20 se
aplica **antes** de contar el `n_confirmed`.

**Implementacion exigida (D-06.5):** el script v3 debe verificar
`H20_complete` en el filtro de elegibilidad, no solo `L existe`. El test
contractual correspondiente se exige en §16.

### 2.2. Justificacion del 550 (fuente congelada)

El umbral 550 proviene de `scripts/power_analysis_5bX.py`
(commit `344dae9`, schema `wyckoff_5bX_power_summary_v1`). El script:

- Reutiliza `load_dataset`, `precompute_features`, `compute_sow_cache`,
  `build_episodes_landmark`, `episode_metrics` y `bootstrap_lift_H20`
  del calibrador `calibrate_wyckoff_sow_5b4bis.py` sin tocar la logica.
- Remuestrea tickers con reemplazo (cluster bootstrap real) hasta
  n_conf_target.
- Asigna outcome simulado Bernoulli(p_base + lift_real) sobre
  confirmed, preservando la estructura real de clusters.
- Corre bootstrap_lift_H20 sobre la muestra simulada.
- Repite N_SIM=500 veces por celda.

**Supuestos congelados (ex-ante):**

    efecto de referencia (lift_real grid): {0.03, 0.05, 0.074, 0.10, 0.15}
    p_base: empirico del pool de desarrollo (0.4697)
    metodo: cluster bootstrap por ticker, remuestreo con reemplazo
    B interno de la simulacion: BOOT_B_SIM = 300
    seed de simulacion: SIM_SEED = 20261003
    criterio de potencia: P(IC95% lower > 0) >= 0.80
    definicion de IC de la potencia: IC95% sobre la proporcion de exitos

**Resultado congelado para lift_real = 0.074 (observado en desarrollo):**

    n_conf | power  | CI95 lower del poder
    -------|--------|--------------------
     400   | 0.672  | 0.631
     450   | 0.814  | 0.780
     500   | 0.808  | 0.773
     550   | 0.838  | 0.806   <- primer n con CI95 lower >= 0.80
     600   | 0.890  | 0.863

**Regla de decision aplicada:** el umbral normativo es el primer n donde
el **limite inferior del IC95% de la potencia** cruza 0.80. A 450 el
punto (0.814) cruza pero el lower (0.780) no. A 550 ambos cruzan.

### 2.3. Guardrail temporal (no criterio de potencia)

    >= 12 meses de sesiones bursatiles posteriores al cutoff 2026-10-01

**Proposito:** evitar que una validacion estadisticamente alcanzable
ocurra en una ventana demasiado estrecha y dominada por una unica
condicion de mercado. NO es sustituto del umbral de muestra.

**Razon:** el QA 36 (seccion 2) documenta que starts/mes varia de 0 a
~85.8 en el historico. Tiempo calendario y potencia no son equivalentes.

### 2.4. NO recalibrar

Los parametros N/M/X/Y estan congelados en v1.9. La ejecucion de
5b.X v3 **no explora grid**: aplica la candidata una sola vez.

### 2.5. Script de validacion (PENDIENTE DE IMPLEMENTACION)

El script de 5b.X v3 sera `scripts/validate_wyckoff_sow_5bX.py`
(nombre fisico conservado por compatibilidad con consumidores),
derivado de `calibrate_wyckoff_sow_5b4bis.py` **sin tocar la logica
del landmark, de los bloques, ni del bootstrap**. Ver §14 para la
regla de auditoria de equivalencia.

**Estado de congelacion (PENDIENTE, 2026-10-03):**

    Fichero:          scripts/validate_wyckoff_sow_5bX.py
    sha256 script:    PENDIENTE
    sha256 protocolo: PENDIENTE (se computa al cerrar este documento)
    commit script:    PENDIENTE
    python:           3.14
    seed:             20261002 (heredada del calibrador, BOOT_SEED)
    B:                2000 (heredada del calibrador, BOOT_B)

**El script v1 (`CC01A6AB...`, commit `d324346`) queda como freeze
historico NO VALIDO como paquete ejecutable de v3. Ver
`37_d06_5bX_invalidada.md`.**

**Verificacion obligatoria pre-ejecucion:**

    Set-Location D:\Macro_Sectorial
    $h = (Get-FileHash scripts\validate_wyckoff_sow_5bX.py -Algorithm SHA256).Hash
    if ($h -ne "<sha256 script v3, PENDIENTE>") {
        throw "ABORTAR: hash del validador no coincide con el congelado"
    }

Si el hash difiere, la validacion queda invalidada. Se abre 5b.X-ter
con nuevo protocolo y nuevo hash, declarando la presente abortada.

---

## 3. Definiciones

Heredadas del protocolo v3 (5b.4-bis) y del contrato v1.9, con una
precision nueva sobre la unidad temporal.

- **Unidad:** candidate episode (t0 = entrada, candidate_t=True AND
  candidate_{t-1}=False).
- **Landmark:** L = t0 + M, con M=30.
- **Confirmed:** SOW en [t0+1, L].
- **Baseline:** sin SOW en [t0+1, L].
- **Outcome primario:** struct_deterioration_H20 = struct[L+20] < struct[L].
- **Outcome secundarios:** price_weakness, below_support, lower_low,
  H10 y H40.
- **Censura:** elegible para H solo si L+H < len(dates) en el momento
  del corte de datos.
- **Lift:** P(deterioro | confirmed) - P(deterioro | baseline),
  episode-weighted.

### 3.1. Universo out-of-sample

Un episodio es OOS **si y solo si**:

    t0 >= 2026-10-02

No:

    L >= 2026-10-02

**Razon:** con M=30, un episodio iniciado antes del cutoff puede tener
landmark despues del cutoff. Ese episodio participa parcialmente en el
historico de desarrollo y no es una observacion OOS independiente.

**Claridad adicional:** la generacion del candidato en t0 puede usar
toda la informacion historica disponible hasta t0, incluida la anterior
al cutoff. El requisito se aplica a t0, no a la ventana retrospectiva
de construccion de la senal.

---

## 4. Bloques de validacion

Los bloques P1/P2/P3 del protocolo v3 son historicos y ya estan
"quemados" (seleccionados para el desarrollo). En 5b.X:

### 4.1. Bloque unico de validacion

Todo dato con t0 >= 2026-10-02 forma **un unico bloque de validacion**,
sin subdivisiones.

Razon: la subdivision en 3 bloques fue decision de diseno para el
desarrollo. En validacion no se subdivide porque no hay masa critica
para 3 bloques con suficiente muestra.

### 4.2. Reporte por sub-bloques (informativo)

Si el bloque unico de validacion tiene suficiente muestra
(>100 episodios confirmados H20-complete), se puede reportar
adicionalmente por sub-periodos de ~6 meses, **solo con caracter
informativo**, no como criterio de seleccion ni de decision.

---

## 5. Criterio de exito

### 5.1. Criterio primario

    IC95% lower del lift_H20 > 0

calculado con bootstrap ticker-cluster, B=2000, seed fija
(20261002), sobre el bloque unico de validacion.

**Se reporta obligatoriamente:**

- lift_point
- lower_CI, upper_CI
- n_confirmed_H20_complete, n_baseline_H20_complete
- n_tickers_total, n_tickers_confirmed, n_tickers_baseline
- concentracion top5 share y top10 share
- starts/mes en el bloque de validacion

**La cantidad de episodios no sustituye a la cantidad de clusters.**

### 5.2. Comparacion con el desarrollo

Para contexto, se reporta lado a lado:

    Desarrollo (5b.4-bis):
        lift = +0.0740  IC [+0.0234, +0.1242]  n_conf=635

    Validacion (5b.X v3):
        lift = ?         IC [?, ?]            n_conf=?

**NO se exige que el lift de validacion sea >= el de desarrollo.**
Se exige que el IC95% lower sea > 0.

### 5.3. Criterios secundarios (informativos)

- Comportamiento de los outcomes secundarios.
- Concentracion por ticker (top5/top10 share).
- Estabilidad por sub-bloques (si aplica, seccion 4.2).

---

## 6. Interpretacion de resultados

### 6.1. Escenarios (declarados ex-ante)

La clasificacion se basa **exclusivamente en el IC95%**, nunca en el
lift_point. Un corte de lift_point (como el +0.025 propuesto en el QA
36) queda expresamente descartado como frontera entre escenarios.

**Escenario A - Validacion confirma.**

    IC95% lower > 0

La candidata pasa a contrato productivo (v2.0). Se autoriza activacion
en config y migracion de consumidores.

**Escenario B - Inconcluyente.**

    IC95% lower <= 0 <= IC95% upper

El IC cruza 0. El resultado no confirma ni refuta. Se documenta como
"no concluyente". La decision posterior (ampliar ventana, abandonar,
abrir 5b.5) queda al auditor externo.

**Escenario C - Evidencia contraria.**

    IC95% upper < 0

El IC esta enteramente bajo 0. Existe evidencia de efecto negativo.
Se abandona la candidata. Se abre 5b.5 con otra hipotesis o se cierra
la linea SOW como confirmador.

### 6.2. Escenario D - INSUFFICIENT_SAMPLE (no es un resultado)

    D = INSUFFICIENT_SAMPLE

**D no es un cuarto resultado estadistico equivalente a A/B/C.** Es un
estado de **no ejecucion / no inferencia**:

> "No se ejecuta la validacion confirmatoria porque no se cumplen
>  sus precondiciones."

**Jerarquia correcta:**

    FASE 1: precondiciones (seccion 2)
        muestra < 550 H20-complete  -> D / NO EJECUTAR
        muestra >= 550              -> ir a FASE 2

    FASE 2: inferencia estadistica
        lower_CI > 0                -> A
        IC cruza 0                  -> B
        upper_CI < 0                -> C

**Prohibido:** reportar "resultado D" como si fuera una clasificacion
equivalente a A/B/C. D bloquea la ejecucion, no la clausura.

### 6.3. Implementacion de la clasificacion (D-06.4)

El script v3 implementara la clasificacion **por IC95%**, no por
lift_point. Orden exacto:

    if lower_ci > 0:             return "A"
    if upper_ci < 0:             return "C"
    return "B"   # IC cruza 0

**Prohibido en la implementacion:** usar `lift_point > 0` como frontera
entre B y C. El lift_point se reporta, no se clasifica.

El test contractual correspondiente se exige en §16.

---

## 7. Analisis prohibidos en 5b.X v3

- Recalibrar N/M/X/Y (estan congelados en v1.9).
- Explorar grid (no hay grid; una sola configuracion).
- Volver a seleccionar entre las 17 candidatas de 5b.4-bis.
- Subdividir el bloque de validacion en funcion de los resultados.
- Ampliar el grid si el lift cae.
- Usar los sub-bloques informativos como criterio de decision.
- Modificar el codigo entre freeze y validacion.
- Cambiar el cutoff temporal.
- **Cambiar el umbral de 550 confirmed H20-complete.**
- **Cambiar la definicion del universo OOS (t0 >= cutoff).**
- **Clasificar por lift_point en lugar de por IC95%.**
- **Tratar D como un resultado estadistico.**
- **Contar confirmed sin verificar H20_complete.**
- Cambiar el criterio de exito (5.1) tras ver resultados.
- Reportar solo el resultado favorable.

**Cualquier cambio en estas reglas invalida la validacion. Si hace
falta cambiar algo, se abre 5b.X-ter con nuevo protocolo y se declara
que la validacion presente ha sido abortada.**

---

## 8. Salida esperada

- `outputs/audit/wyckoff_5bX_bootstrap.csv` (una fila).
- `outputs/audit/wyckoff_5bX_summary.json`.
- `outputs/audit/wyckoff_5bX_run.log`.
- `docs/auditoria/wyckoff/38_validacion_5bX_resultados.md`.
  Nota: 36 ocupado por QA precondiciones; 37 ocupado por D-06.
  La validacion real usa el 38.

**Cambio respecto a v2 (§8):** se elimina `wyckoff_5bX_grid.csv`
(no hay grid). Se anade obligatoriamente `n_H20_complete` y
`n_confirmed_H20_complete` en el summary JSON. Campos minimos:

    {
      "schema":  "wyckoff_5bX_summary_v1",
      "status":  "EXECUTED" | "BLOCKED" | "BLOCKED_INSUFFICIENT_H20",
      "candidata": { N, M, X_ATR, Y_VOL },
      "cutoff_date": "2026-10-01",
      "oos_rule": "t0 >= 2026-10-02",
      "preconditions": { ... },
      "validation": {
        "n_episodes", "n_confirmed_H20_complete", "n_baseline_H20_complete",
        "n_tickers_total", "n_tickers_confirmed", "n_tickers_baseline"
      },
      "bootstrap": { "B", "seed", "lift_point", "lower_ci", "upper_ci" },
      "scenario": { "code": "A"|"B"|"C"|"D", "motivo" }
    }

---

## 9. Que NO se hace en 5b.X v3

- NO activar parametros en config.
- NO retirar fail-closed.
- NO migrar consumidores.
- NO abrir 5c.4 ni 5d.
- NO declarar exitosamente "SOW es confirmador" hasta que el informe
  de 5b.X v3 lo confirme con IC95% lower > 0 (Escenario A).

---

## 10. Estado

    v1.8              CONTRATO PRODUCTIVO VIGENTE
    v1.9              CANDIDATA FROZEN_FOR_VALIDATION
    5b.3              FAIL
    5b.4              FAIL seleccion / diagnostico valido
    5b.4-bis          PASS desarrollo / NO confirmado
    5b.X v2           INVALIDADA (D-06, 2026-10-03)
    5b.X v3           ESTE DOCUMENTO, pendiente implementacion + datos
    5c.4, 5d          BLOQUEADAS
    Legacy            INTACTO
    Config SOW        None (fail-closed)

---

## 11. Trazabilidad

- 5b.3 (`639fe92`): primera ejecucion con sesgo de anclaje.
- Dictamen 24: inmortal time bias detectado.
- 5b.4 (`04e0620`): landmark corregido. FAIL en seleccion (D3
  inalcanzable con 4 bloques).
- Dictamen 26: core landmark aprobado. 10 correcciones al protocolo.
- 5b.4-bis (`848ee23`): rediseno a 3 bloques. PASS desarrollo
  (17/240).
- Dictamen 33: freeze de validacion autorizado. Freeze productivo NO.
- QA 5b.4-bis (`dacc7b2`): IC por bloque. Solo P3 no cruza 0.
- Contrato v1.9: candidata congelada.
- 5b.X v1 (2026-10-02): protocolo inicial, umbrales 12m/50/20.
- QA 36 (`d82beea`): precondiciones 5b.X. Diagnostico aceptado;
  umbrales insuficientes para poder estadistico.
- Dictamen externo 2026-10-02 (sobre 35 v1 + 36):
  - universo OOS: `t0 >= cutoff` (no `L >= cutoff`).
  - muestra de potencia: H20-complete (no landmark-complete).
  - clasificacion A/B/C/D por IC, no por lift_point.
  - D = estado de no ejecucion, no resultado.
  - bloque D historico: solo diagnostico, sin valor confirmatorio.
  - script de validacion: congelar con hash+commit antes de OOS.
  - reescritura completa a v2 (no parche).
- Power analysis (`344dae9`, 2026-10-03): simulacion cluster
  bootstrap real. Umbral 550 confirmed H20-complete derivado del
  primer n cuyo CI95 lower de la potencia cruza 0.80.
- 5b.X v2 (`9ef7497`, 2026-10-03): reescritura normativa que
  incorpora las correcciones del dictamen.
- **Script congelado (`ccba3ab`, 2026-10-03): congelo el sha256 del
  script v1 (commit `d324346`) bajo mensaje de sincronizacion v2.
  Autocontradictorio.**
- **D-06 detectado (2026-10-03):** el script v1 no implementa v2. Ver
  `37_d06_5bX_invalidada.md` para detalle de D-06.1 a D-06.5.
- **Dictamen externo 2026-10-03 (sobre D-06):** 5b.X original
  INVALIDADA como paquete ejecutable. Se abre 5b.X-bis/v3 con nuevo
  script + tests + nuevo hash. Revision de conformidad del auditor
  antes del freeze final. No se reabre candidata ni umbrales.
- **5b.X v3 (este documento, 2026-10-03):** protocolo que reemplaza
  operativamente a v2 y preserva su contenido normativo. Anade
  §14 (auditoria de equivalencia), §15 (manifiesto de freeze), §16
  (tests contractuales exigidos).

---

## 12. Condiciones de activacion (post 5b.X v3 exitoso)

Si 5b.X v3 resulta en Escenario A (IC95% lower > 0):

1. Emitir contrato v2.0 con STATUS=PRODUCTION.
2. Actualizar `config/settings.py`:
       WYCKOFF_SOW_WINDOW_N  = 60
       WYCKOFF_SOW_MAX_AGE_M = 30
       WYCKOFF_SOW_X_ATR     = 0.25
       WYCKOFF_SOW_Y_VOL     = 1.10
3. Evaluar retirada del fail-closed (o mantenerlo como argumento
   explicito obligatorio).
4. Migrar consumidores de `wyckoff.py` legacy a `wyckoff_v1.py`
   (con E2E obligatorio).
5. Abrir 5c.4 y 5d.
6. Documentar en `07_RUNBOOK.md`.

**Ninguno de estos pasos se ejecuta hasta que 5b.X v3 cierre con
Escenario A.**

---

## 13. Changelog v2 -> v3

| Seccion | Cambio |
|---|---|
| Cabecera | "Reemplaza operativamente a v2". Nota de invalidacion de 5b.X v2. |
| 1 | Estado de partida: 5b.X v2 INVALIDADA; 5b.X v3 vigente. |
| 2.1 | Explicitada formula operativa `t0_idx + 50 < len(dates)` para H20_complete. Referencia a D-06.5. |
| 2.5 | Reescrita: script PENDIENTE, sha256 PENDIENTE, manifiesto de freeze en §15. El hash v1 se preserva como historico. |
| 3.1 | Aclaracion: cutoff 2026-10-01 = ultima fecha del historico; universo OOS = `t0 >= 2026-10-02`. |
| 6.3 | Nueva: implementacion de clasificacion por IC (pseudocodigo). Referencia a D-06.4. |
| 7 | Ampliados los prohibidos: clasificar por lift, contar confirmed sin H20_complete. |
| 8 | Eliminado `wyckoff_5bX_grid.csv`. Anadido `n_H20_complete`. Renumerado informe de resultados a 38. |
| 11 | Trazabilidad ampliada con D-06 y dictamen 2026-10-03. |
| 14 | Nueva: auditoria de equivalencia (regla del auditor). |
| 15 | Nueva: manifiesto de freeze. |
| 16 | Nueva: tests contractuales exigidos. |
| 13 | Este changelog. |

**No cambia respecto a v2:** proposito (§0), definiciones base (§3 salvo
3.1), bloques (§4), criterio primario (§5.1), comparacion desarrollo
(§5.2), criterios secundarios (§5.3), escenarios A/B/C/D (§6.1-6.2), que
no se hace (§9), condiciones de activacion (§12).

---

## 14. Auditoria de equivalencia (regla del auditor)

Regla textual del auditor, 2026-10-03:

> **el nuevo script es v2 unicamente respecto de los cambios
> normativos aprobados; todo lo demas permanece bit-equivalente o
> funcionalmente equivalente a la implementacion auditada de
> 5b.4-bis.**

**Cambios normativos autorizados en el script v3:**

1. Umbral de muestra: 20 -> 550 confirmed H20-complete.
2. Universo OOS: `L_date > cutoff` -> `t0_date >= 2026-10-02`.
3. Retirada de `MIN_MONTHS=12` y `MIN_N_EPISODES=50` como bloqueantes
   (si se conservan, solo como guardrail informativo, no bloqueante).
4. Clasificacion de escenarios: por IC95%, no por lift_point.
5. Elegibilidad H20: verificar `L_idx + 20 < len(dates)` antes de
   contar confirmed.
6. Anadido `status=BLOCKED_INSUFFICIENT_H20` y `scenario=D` antes
   de bootstrap si `n_confirmed_H20 < 550`.

**Todo lo demas debe permanecer bit-equivalente o funcionalmente
equivalente a la implementacion auditada de `calibrate_wyckoff_sow_5b4bis.py`.**
En particular, sin cambios en:

- `find_candidate_starts`
- `build_episodes_landmark`
- `episode_metrics` (calculo de `struct_deterioration_H20`, outcomes
  secundarios)
- Definicion de SOW (candidata N=60, M=30, X_ATR=0.25, Y_VOL=1.10)
- Bootstrap ticker-cluster (`bootstrap_lift` / `bootstrap_lift_H20`)
- `BOOT_B=2000`, `BOOT_SEED=20261002`
- Censura H20 en el calculo del outcome
- Calculo del lift (episode-weighted)

**Evidencia de cumplimiento exigida antes del freeze:**

- Diff `git diff` del script v3 contra `calibrate_wyckoff_sow_5b4bis.py`,
  anotando linea a linea que cada cambio cae en uno de los 6 puntos
  normativos.
- Declaracion explicita: "cero cambios no autorizados".

---

## 15. Manifiesto de freeze

**PENDIENTE.** Se rellena en el commit de freeze del script v3,
**despues** de los tests contractuales (§16) y de la revision de
conformidad del auditor.

    git_commit:        PENDIENTE
    script_sha256:     PENDIENTE
    protocol_sha256:   PENDIENTE (sha256 del fichero 35_protocolo_5bX_v3.md)
    python:            3.14
    seed_bootstrap:    20261002 (BOOT_SEED)
    B_bootstrap:       2000 (BOOT_B)
    fecha_freeze:      PENDIENTE
    revision_auditor:  PENDIENTE

**Regla:** la validacion v3 no se ejecuta hasta que este manifiesto
este completo y el auditor haya certificado conformidad del diff.

---

## 16. Tests contractuales exigidos

**PENDIENTE DE IMPLEMENTACION.** Se escriben en el mismo commit que
el script v3, antes del freeze final. Cubren como minimo:

1. **Rechazo por cutoff.** Episodio con `t0 < 2026-10-02` no entra en
   el bloque de validacion, aunque `L >= 2026-10-02`. (D-06.2)
2. **Bloqueo por muestra.** Con `n_confirmed_H20 < 550`, el validador
   no ejecuta bootstrap, escribe `status=BLOCKED_INSUFFICIENT_H20`,
   `scenario=D`. (D-06.1, §6.2)
3. **Escenario A.** Bootstrap sintetico con `lower_ci > 0` produce
   `scenario=A`.
4. **Escenario B.** Bootstrap sintetico con `IC cruza 0` produce
   `scenario=B`, **independientemente del signo de `lift_point`**.
   Caso testigo: `lift = -0.5`, `IC = [-1.2, +0.2]` -> B. (D-06.4)
5. **Escenario C.** Bootstrap sintetico con `upper_ci < 0` produce
   `scenario=C`.
6. **Elegibilidad H20.** Episodio confirmed con `L_idx + 20 >=
   len(dates)` NO cuenta como `n_confirmed_H20_complete`. (D-06.5)
7. **Verificacion de hash pre-ejecucion.** Si `sha256` del script no
   coincide con el manifiesto, la ejecucion aborta antes de leer
   datos. (§2.5)
8. **Sin grid.** El validador aplica la candidata una sola vez.
   Cero llamadas a exploracion de parametros. (§2.4)
9. **Determinismo.** Dos ejecuciones consecutivas con el mismo input
   producen el mismo `summary.json` (excluyendo timestamps).

**Los tests no reimplementan la logica del validador.** Se anclan a
los contratos declarados en este protocolo. Si la implementacion
cambia, los tests fallan: esa es la senal.

---

**Fin del protocolo 5b.X v3.**
