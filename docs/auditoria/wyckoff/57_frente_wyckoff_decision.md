# 57 - Frente Wyckoff - Decision solicitada al auditor

Fecha: 2026-10-05
Ramas: gold-standard-v3 (HEAD 6c4912f) + sow-v4-validation (HEAD a505b87)
Motivo: unificar el estado del frente para auditor rotativo.

## 1. Contexto

Este documento es autocontenido. El frente Wyckoff ha recibido
dictamenes de al menos cuatro auditores distintos con criterios
metodologicos incompatibles. Los expedientes 33-56 se citan solo
como trazabilidad, no como prerequisito de lectura.

Objeto: detector SOW. Tres claims separables:

    Capa 1 (semantica)    D=1 <=> debilidad estructural Wyckoff
    Capa 2 (fase)         SOW eleva RANGE -> DISTRIBUTION
    Capa 3 (predictiva)   D=1 anticipa deterioro H20

Cinco intentos de validacion. Cero confirmaciones. Produccion en legacy.

## 2. Detector

Implementacion: indicators/wyckoff_v1.py (v1.8, no produccion,
fail-closed).

    SHA-256 wyckoff_v1.py:
      FA9F30AD6EBE1DCBEECC3D058C82DE146F6AEB735A067B9752C772B97A4389AB

Legacy en produccion: indicators/wyckoff.py (6 consumidores).

Formula (detect_sow, lineas 426-486):

    support_t      = rolling_min(Low, N).shift(1)
    ATR_baseline_t = ATR(20).shift(1)
    vol_baseline_t = rolling_mean(Volume, N).shift(1)

    break_depth_t  = (support_t - Close_t) / ATR_baseline_t
    volume_ratio_t = Volume_t / vol_baseline_t

    SOW_t = (Close_t < support_t)
            AND (break_depth_t >= X_ATR)
            AND (volume_ratio_t >= Y_VOL)

Parametros:

    Parametro   v1.9 congelada   config actual
    ---------   --------------   -------------
    N           60               None
    M           30               None
    X_ATR       0.25             None
    Y_VOL       1.10             None

v1.9 STATUS = FROZEN_FOR_VALIDATION / NOT_PRODUCTION.
Contrato vigente: v1.8. Contrato candidato: v1.9.
Config WYCKOFF_SOW_* = None. Fail-closed.
## 3. Los cinco intentos

    #    Fecha   Rama    Metodo                       Resultado
    --   -----   -----   --------------------------   -------------------
    v3   10-05   sow-v4  maxT fixed-SE B=20000        0/240 p_min=0.814009
    v5   10-04   sow-v4  grounded outer (TEST=fit)    NO APROBADO N_eval=2
    v6   -       -       AIPW + Brier/LogLoss         autorizado no impl
    v8   10-05   gs-v3   Se/Sp Hajek + RWY + 4 celd   NO-GO 12/12 FAIL Se
    v9   -       -       rare-event enrichment        no viable (secc. 4)

Detalle minimo por intento:

    v3: dictamen 49. Formulacion correcta: NO EVIDENCIA SUFICIENTE DE
        EFECTO >= 5 pp. NO es PRUEBA DE AUSENCIA. Candidata v1.9:
        p_maxT = 0.916154. Mejor de 240: 0.814009. Protocolo
        SOW_v19_PROTOCOL.json SHA-256 c6dae718...

    v5: dictamen 46. Ajusta el modelo en OUTER TEST y evalua sobre el
        mismo OUTER TEST. No es out-of-sample. IC95 fold 1 anchura
        ~0.44. 5/7 folds no convergen.

    v6: dictamen 46. OUTER TRAIN -> m(Y|.) + e(SOW|.) -> OUTER TEST
        con Y observado -> AIPW + Brier + LogLoss + MBB + placebos
        homogeneos. delta_min = 5 pp. Pendiente de spec firmada.

    v8: dictamen 56. q_D = 1.0512%. ~584 FN en poblacion. Muestra
        D=0 ~400 -> ~0.97 FN esperados. Necesario ~164 FN ->
        n_D0 ~67500 -> factor ~42x sobre techo 800. 12/12 escenarios
        gate FAIL en e(Se).

    v9: no redactado. Dictamen 56 Ruta 2 exige demostrar factor de
        enriquecimiento ex ante sin etiquetas humanas. Los tres
        caminos admisibles violan independencia de Y o usan datos
        prohibidos. No viable.

## 4. Aritmetica del muro

    q_D       = 2559 / 243435 = 0.01051204630
    P(FN|D=0) = 584 / 240876  = 0.00242
    Necesario ~164 FN en muestra para e(Se) <= 0.07 con Se ~ 0.70

Reparto muestral y factor minimo de enriquecimiento:

    n_D1 / n_D0   n_D0   FN sin enriquecer   rho_min
    -----------   ----   -----------------   -------
    400 / 400     400    ~0.97               ~169x
    500 / 300     300    ~0.73               ~225x
    700 / 100     100    ~0.24               ~675x

Conclusion: con techo 800 y P(FN|D=0) = 0.242%, cualquier diseno de
enriquecimiento exige factor >= 150x. Ningun proxy OHLCV admisible
bajo independencia de Y ha sido argumentado para ese orden.

Techo vigente: 800 unidades unicas por anotador, 3 anotadores.
No se relaja.
## 5. Decision solicitada

Cinco preguntas cerradas. Respuesta: SI / NO / REFORMULAR.

    D1. Se aprueba tratar Capa 1 (semantica), Capa 2 (fase) y Capa 3
        (predictiva) como claims independientes, validables por
        metodos distintos y con estados distintos?

    D2. Se autoriza archivar Capas 1+2 como NO VALIDADAS Y NO
        VALIDABLES CON EL DISENO ACTUAL (techo 800/anotador, rareza
        0.242%), etiquetadas UNVALIDATED / INCONCLUSIVE DUE TO
        SAMPLE-CAPACITY CONSTRAINT, NO como NEGATIVE / ABSENCE OF
        EFFECT, dejando abierta la reapertura condicionada (ver D4)?

    D3. Se autoriza migrar produccion a wyckoff_v1.py v1.8-core
        (4 fases: MARKUP, ACCUMULATION, RANGE, MARKDOWN; sin
        DISTRIBUTION, sin SOW), previa cierre formal de 5c
        comparativa legacy vs v1, con el gate minimo de 7 puntos
        definido en seccion 6?

    D4. Se autoriza reabrir Capa 1 mediante metodo alternativo
        (expertos Wyckoff o eventos externos documentados), con
        protocolo aparte pre-registrado, ciego, independiente y
        sujeto a nueva firma?

    D5. Se confirman las prohibiciones de la seccion 7 hasta nueva
        orden?

Contexto D3: legacy wyckoff.py clasifica 20/20 sectores como RANGE
en el CSV del 3-oct. No discrimina fases utiles. v1.8-core (sin SOW)
clasifica MARKUP/MARKDOWN/ACCUMULATION/RANGE. La migracion core no
depende de SOW activado. DISTRIBUTION queda fuera de D3 porque
depende de Capas 1+2.

## 6. Gate 5c minimo (condicion de D3)

Antes de migrar consumidores a v1.8-core, debe quedar documentado:

    1. Que entradas recibe exactamente legacy y v1.
    2. Que diferencias de clasificacion aparecen.
    3. Que diferencias son esperadas por diseno.
    4. Que no existe dependencia residual de SOW/DISTRIBUTION.
    5. Que los consumidores no reciben campos semanticamente
       incompatibles.
    6. Que el cambio no introduce look-ahead ni fuga temporal.
    7. Que existe regresion reproducible del comportamiento del core.

Una mejora de discriminacion frente a legacy no constituye evidencia
de verdad de la clasificacion.
## 7. Prohibiciones vigentes

Unificadas de dictamenes 33, 46, 49, 56. Literales.

    NO tocar wyckoff_v1.py.
    NO tocar legacy wyckoff.py.
    NO tocar config/settings.py.
    NO tocar protocolo v3 firmado (SOW_v19_PROTOCOL.json).
    NO activar parametros SOW.
    NO reabrir grid 2021-2026.
    NO modificar MDE = 5 pp.
    NO reducir e(Se) ni eliminarlo del gate.
    NO cambiar q_D, pi_Y ni Y tras observar resultados.
    NO cambiar B / R_MC para conseguir PASS.
    NO declarar PASS por Sp/PPV/NPV sin Se.
    NO usar panel futuro para retroajustar power model.
    NO migrar consumidores sin 5c cerrada.
    NO merge a main sin dictamen favorable.

## Anexo. Trazabilidad

    Hash SHA-256:
        indicators/wyckoff_v1.py
            FA9F30AD6EBE1DCBEECC3D058C82DE146F6AEB735A067B9752C772B97A4389AB
        config/settings.py
            8343B393B5418E2A3B362DE7FD095F30D5D6EBE6DAC21E1465B310B8A1E6FECF
        docs/auditoria/wyckoff/SOW_v19_PROTOCOL.json
            c6dae718a258ac47e4c75326ec5278397a97f116a56f5070a22f8c8778f3a1f2
        outputs/audit/power_v8_screening_2026-10-05.json
            B113F50FD4885DC39B43919BA5E09A90FD11F2989DED10B8856235D5ED5B1F85

    Tags:
        v8-frozen-pre-abort
        v8-suspended-capacity-se

    Ramas:
        gold-standard-v3       HEAD 6c4912f
        sow-v4-validation      HEAD a505b87
        main                   8c811c1 (sin merge del frente Wyckoff)

    Fin del documento.