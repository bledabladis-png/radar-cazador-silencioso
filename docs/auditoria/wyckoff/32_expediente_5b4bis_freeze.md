# EXPEDIENTE 5b.4-bis - Congelacion de parametros

**Documento de consulta. NO normativo. Requiere dictamen externo.**
**Fecha:** 2026-10-02.
**Precedente:** 31_calibracion_5b4bis_resultados.md + protocolo v3 +
dictamen 29 + commit 4595676.
**Estado:** BLOQUEO de la congelacion hasta dictamen.

---

## 0. Proposito

5b.4-bis se ejecuto correctamente. **17/240 combinaciones pasan D1-D3
por primera vez en toda la secuencia.** El ranking v3 selecciona una
candidata sin ambiguedad.

Se solicita dictamen sobre:

- Autorizacion o no de congelar la candidata.
- Alcance de la congelacion.
- Modo de validacion en 5b.X.
- Cualquier ajuste previo a la congelacion.

**Cero cambios de codigo, config o contrato hasta dictamen.**

---

## 1. Candidata elegida por el ranking v3

    N=60, M=30, X_ATR=0.25, Y_VOL=1.10

Metricas (H20, struct_deterioration):

    n_confirmed = 635
    n_baseline  = 1753
    lift_point  = +0.0740
    lower_CI    = +0.0234   (bootstrap ticker-cluster, B=2000, seed=20261002)
    upper_CI    = +0.1242

    n_bloques_evaluables = 3
    n_bloques_positivos  = 3   (firma "+++")
    mediana_lift         = 0.0932
    delta_hetero         = 0.0556

Por bloque:

    P1 (2021-10 -> 2023-06): n_conf=32   n_base=226  lift=+0.093
    P2 (2023-07 -> 2025-03): n_conf=304  n_base=914  lift=+0.050
    P3 (2025-04 -> 2026-10): n_conf=293  n_base=591  lift=+0.106

La candidata gana el ranking v3 por:
1. 3/3 bloques evaluables con lift positivo (9 combinaciones empatadas).
2. Mayor mediana_lift dentro de ese grupo (0.0932 vs 0.0895 siguiente).
3. Menor delta_hetero (0.0556 vs 0.2394 y 0.2250 siguientes).

No hay desempate por lexicografico.

---

## 2. Las 9 candidatas con firma "+++"

Por si el auditor prefiere otra combinacion del grupo estable:

| N | M | X_ATR | Y_VOL | n_conf | n_base | lift | lower_CI | mediana | delta_het |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | 30 | 0.25 | 1.10 | 635 | 1753 | +0.0740 | +0.0234 | 0.0932 | 0.0556 |
| 60 | 20 | 0.25 | 1.10 | 506 | 1886 | +0.0587 | +0.0059 | 0.0895 | 0.2394 |
| 60 | 20 | 0.25 | 1.20 | 472 | 1920 | +0.0548 | +0.0003 | 0.0865 | 0.2250 |
| 40 | 30 | 0.25 | 1.10 | 703 | 1685 | +0.0690 | +0.0205 | 0.0851 | 0.0654 |
| 60 | 30 | 0.25 | 1.20 | 601 | 1787 | +0.0647 | +0.0131 | 0.0756 | 0.0425 |
| 40 | 30 | 0.25 | 1.20 | 671 | 1717 | +0.0577 | +0.0094 | 0.0668 | 0.0726 |
| 40 | 30 | 0.50 | 1.10 | 544 | 1844 | +0.0630 | +0.0099 | 0.0645 | 0.0116 |
| 30 | 30 | 0.25 | 1.10 | 794 | 1594 | +0.0474 | +0.0025 | 0.0610 | 0.0357 |
| 40 | 30 | 0.50 | 1.20 | 527 | 1861 | +0.0505 | +0.0000 | 0.0560 | 0.0057 |

**Observaciones:**

- Las 9 tienen firma "+++" y D1+D2+D3 OK.
- La candidata elegida (#1) tiene el mayor mediana_lift y el menor
  delta_hetero de su subgrupo inmediato.
- La #4 (40/30/0.25/1.10) tiene n_confirmed mayor (703) pero mediana
  ligeramente inferior.
- La #8 (30/30/0.25/1.10) tiene n_confirmed maximo (794) pero mediana
  baja (0.061).
- La #7 (40/30/0.50/1.10) tiene delta_hetero minimo (0.0116) pero
  mediana baja (0.0645).

**El equipo no propone alternativa a la #1. La lista se incluye para
transparencia y por si el auditor prefiere otro criterio.**

---

## 3. Preguntas al auditor externo

### P1. Autorizacion de freeze

La candidata pasa D1-D3 y el ranking v3 la selecciona sin ambiguedad.

**Pregunta:** autoriza congelar N=60, M=30, X_ATR=0.25, Y_VOL=1.10?

    a) Si, congelar la candidata #1 tal cual.
    b) No, mantener todos los parametros en None hasta validacion 5b.X.
    c) Congelar solo algunos parametros (p.ej. N y M), otros siguen
       en None.
    d) Otra.

### P2. Alcance de la congelacion

Si se autoriza congelar, el cambio en el codigo seria:

    config/settings.py:
        WYCKOFF_SOW_WINDOW_N = 60      (antes None)
        WYCKOFF_SOW_MAX_AGE_M = 30     (antes None)
        WYCKOFF_SOW_X_ATR = 0.25       (antes None)
        WYCKOFF_SOW_Y_VOL = 1.10       (antes None)

Y la retirada de la validacion fail-closed de `detect_sow` (o su
relajacion para aceptar los defaults de config).

**Pregunta:** este alcance es correcto, o prefiere otro?

    a) Congelar en config y retirar fail-closed.
    b) Congelar en config y mantener fail-closed (los defaults de
       config se pasan como argumentos explicitos desde los call
       sites).
    c) Congelar solo a nivel documental (contrato v1.9), sin tocar
       config.
    d) Otra.

### P3. Contrato v1.8 -> v1.9

El contrato v1.8 declara los 4 parametros como PROPUESTOS. La
congelacion implicaria crear contrato v1.9 con los valores fijados.

**Pregunta:** se abre v1.9 o se mantiene v1.8 con nota de freeze?

    a) v1.9 nuevo (mismo formato que v1.8 -> v1.9 con seccion de
       congelados).
    b) v1.8 con parche en la seccion 5.4 (fecha de freeze).
    c) Otra.

### P4. Validacion 5b.X

El dictamen 29 exige: validacion final sobre datos temporalmente
posteriores al freeze, con la misma implementacion, sin recalibracion.

**Pregunta:** como proceder a corto plazo, dado que no hay datos
nuevos?

    a) Congelar y esperar acumulacion de datos nuevos. 5b.X se abre
       cuando haya, digamos, 6-12 meses de sesiones nuevas.
    b) Congelar y ejecutar 5b.X sobre el bloque P3 actual como
       "diagnostico post-freeze" (marcado como NO ciego).
    c) Congelar sin fecha de validacion hasta que aparezca una
       ventana natural.
    d) Otra.

### P5. Riesgo de freeze prematuro

El dictamen 29 rechazo explicitamente congelar basandose en las 17 de
5b.4 v2 (todas con D3 fallido). Ahora hay 17 con D3 OK.

**Pregunta:** la mejora metodologica justifica el freeze ahora, o hay
que hacer algo mas antes?

    a) Freeze ahora. El protocolo v3 esta bien disenado y los
       criterios ex-ante se cumplen.
    b) Esperar a un analisis adicional (indicar cual).
    c) Abrir 5b.5 con otra hipotesis.
    d) Otra.

---

---

## 4. Anexo A - Criterios D1-D3 sobre el grid completo

    D1 (n_confirmed >= 20 AND n_baseline >= 20):      240/240
    D2 (lower_CI(lift_H20_ALL) > 0):                   17/240
    D3 (ceil(2/3) bloques evaluables con lift > 0):    69/240
    D1 AND D2 AND D3:                                  17/240

Combos por criterios cumplidos:

    3 de 3:   17
    2 de 3:   52
    1 de 3:  171
    0 de 3:    0

---

## 5. Anexo B - Comparativa 5b.4 v2 vs 5b.4-bis v3

    | | 5b.4 v2 | 5b.4-bis v3 |
    |---|---|---:|
    | Bloques | 4 (A/B/C/D) | 3 (P1/P2/P3) |
    | Bloque mas corto | A=15 meses | P1=21 meses |
    | Bloque mas antiguo n_conf max | 12 | 32 |
    | D1 | 240/240 | 240/240 |
    | D2 | 17/240 | 17/240 |
    | D3 | 0/240 | 69/240 |
    | D1+D2+D3 | 0/240 | 17/240 |
    | Firma elegida | n/a (ninguna) | +++ |
    | Congelacion | rechazada | pendiente |

D2 filtra igual en los dos disenos (17). La diferencia la marca D3,
que pasa de inalcanzable a alcanzable al redistribuir la muestra en
3 bloques con capacidad suficiente.

**No es un cambio de umbral. Es un cambio de diseno temporal.**

---

## 6. Anexo C - Metricas secundarias de la candidata elegida

Para P1 (detalle completo en summary JSON):

    struct_deterioration: conf=0.531  base=0.438  lift=+0.093
    price_weakness:       conf=0.188  base=0.385  lift=-0.197
    below_support:        conf=0.125  base=0.058  lift=+0.067
    lower_low:            (ver CSV)

La senal se concentra en struct_deterioration y below_support.
price_weakness tiene lift negativo en P1 (aunque menor peso). Esto
es coherente con la definicion de SOW como senal estructural.

---

## 7. Anexo D - Artefactos disponibles

    outputs/audit/wyckoff_5b4bis_grid.csv         240 filas
    outputs/audit/wyckoff_5b4bis_bootstrap.csv    240 filas
    outputs/audit/wyckoff_5b4bis_hetero.csv       240 filas
    outputs/audit/wyckoff_5b4bis_summary.json
    outputs/audit/wyckoff_5b4bis_run.log

Todos disponibles para analisis del auditor si los solicita.

---

## 8. Que NO se ha tocado

- Cero cambios en config/settings.py.
- Cero cambios en contrato v1.8.
- Cero cambios en codigo de produccion.
- Parametros SOW siguen en None.
- Legacy intacto.
- Consumidores sin migrar.
- 5c.4 y 5d bloqueadas.

El sistema sigue sin ningun cambio funcional respecto a 4595676
(salvo el script 5b.4-bis y el fix del titulo).

---

## 9. Propuesta de siguiente paso

### Si P1 = (a) freeze ahora:

1. Editar config/settings.py con los 4 valores de la candidata #1.
2. Actualizar contrato a v1.9.
3. Evaluar el alcance de la validacion fail-closed (P2).
4. Congelar el codigo en un commit.
5. Abrir 5b.X como fase de validacion futura.

### Si P1 = (b) no freeze:

1. Mantener parametros en None.
2. Documentar 5b.4-bis como PASS en seleccion pero FAIL en freeze.
3. Abrir 5b.5 con rediseno adicional (indicado por el auditor).

### Si P1 = (c) freeze parcial:

1. Congelar solo los subconjuntos indicados.
2. Resto sigue en None.

---

## 10. Estado al cierre de este expediente

    v1.8              CONTRATO VIGENTE
    5b.3              FAIL (sesgo temporal)
    5b.4              FAIL en seleccion, PASS en diagnostico
    5b.4-bis          PASS en seleccion (17/240), pendiente freeze
    5b.X              PENDIENTE
    Parametros SOW    None en config (fail-closed)
    5c.4, 5d          BLOQUEADAS
    Legacy            INTACTO

---

**Fin del expediente 32.**
