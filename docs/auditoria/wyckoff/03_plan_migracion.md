# PLAN DE MIGRACION - Modulo Wyckoff v1 (v2)

**Hoja de ruta actualizada del proyecto.**
**Fecha:** 2026-10-02 (madrugada del 3-oct).
**Supersede:** 03_plan_migracion.md v1 (2026-10-02 19:14), que reflejaba
el estado del proyecto antes de 5b.2/5b.3/5b.4/5b.4-bis/QA/contrato v1.9.
**Referencia normativa:** contrato v1.9 + protocolo 5b.X + dictamenes
externos 24, 26, 29, 33.

---

## 1. Principios (sin cambios respecto a v1)

- No tocar el legacy hasta cerrar v1: `indicators/wyckoff.py` permanece
  operativo durante todo el proyecto.
- Un cambio = una verificacion = un commit. Fase por fase.
- Contrato antes que codigo. Ninguna linea de implementacion se escribe
  sin haber cerrado la fase documental correspondiente.
- Golden tests antes que migracion. Los consumidores solo migran cuando
  los tests del contrato pasan.
- No mezclar fases.
- **Auditor externo antes de decisiones irreversibles** (activacion de
  parametros, migracion de consumidores, retirada de legacy).
- **Fronteras ex-ante.** Los criterios de seleccion se congelan antes
  de mirar resultados.

---

## 2. Estado actual (2026-10-02 noche)

### 2.1. Contratos vigentes

    v1.8    CONTRATO PRODUCTIVO VIGENTE
    v1.9    CANDIDATA SOW CONGELADA PARA VALIDACION
            (STATUS = FROZEN_FOR_VALIDATION / NOT_PRODUCTION)

Ver `01_contrato_semantico_v1_8.md` y `01_contrato_semantico_v1_9.md`.

### 2.2. Frentes

    Frente A - Nucleo v1 (trend, compression, effort, stability,
               clasificacion MARKUP/MARKDOWN/ACCUMULATION/RANGE)
        Estado: IMPLEMENTADO en wyckoff_v1.py v1.3.
        K = 0.25 congelado (5b.2).
        Componentes no-SOW: cubiertos por Frente F (F-01/F-02/F-03).

    Frente B - SOW (Sign of Weakness, confirmador de DISTRIBUTION)
        Estado: PASS DESARROLLO / NO CONFIRMADO.
        Candidata: N=60, M=30, X_ATR=0.25, Y_VOL=1.10.
        Pendiente: 5b.X (validacion out-of-sample).

    Frente C - Comparativa legacy vs v1
        Estado: PRIMERA PASADA con v1.8 (824ef72). Documento
        13_comparativa_legacy_v1.md actualizado; resultados
        identicos a v1.3 (componentes no-SOW no cambiaron).
        Pendiente: version v2 definitiva post-5b.X.

    Frente D - Migracion de consumidores
        Estado: BLOQUEADO.
        Desbloqueo: tras 5b.X exitoso + cierre 5c.
        Directos (5): index_phase, sector_wyckoff_distribution,
        sector_breadth, stock_leader, index_leaders.
        Indirectos (4, leen columnas derivadas): evidence_matrix,
        sector_concentration, sector_regime_matrix, slpm_v12.
        Incluidos en 5d con E2E pre/post. Ver seccion 6.

    Frente E - Retirada del legacy
        Estado: BLOQUEADO.
        Desbloqueo: tras 5d + 1 run CI estable.

    Frente F - QA interno del modulo v1
        Estado: CERRADO.
        SOW cubierto por 34 + 36.
        No-SOW: F-01 no-deuda, F-02 fix, F-03 cobertura
        (commits a00ee2c + aaf7923).

### 2.3. Configuracion productiva

    config/settings.py:
        WYCKOFF_SOW_WINDOW_N    = None
        WYCKOFF_SOW_MAX_AGE_M   = None
        WYCKOFF_SOW_X_ATR       = None
        WYCKOFF_SOW_Y_VOL       = None
        WYCKOFF_T_NORM_K        = 0.25 (congelado 5b.2)

    detect_sow: fail-closed (los cuatro parametros obligatorios).

### 2.4. Consumidores

Ningun consumidor del pipeline usa `wyckoff_v1.py`. Todos siguen con
`indicators/wyckoff.py` (legacy v4.2):

    indicators/index_phase.py
    indicators/sector_wyckoff_distribution.py
    indicators/sector_breadth.py
    indicators/stock_leader.py
    indicators/index_leaders.py

### 2.5. Bloqueos activos

    5b.X      BLOQUEADO hasta datos posteriores a 2026-10-01
    5c        DESBLOQUEADA desde 5b.2 (ver seccion 5)
    5d        BLOQUEADO hasta cierre 5b.X + 5c
    5e        BLOQUEADO hasta cierre 5d + 1 run CI estable
    5c.4      BLOQUEADO

---

## 3. Orden estricto de desbloqueo

    Frentes A-F
        |
        v
    [A] Nucleo v1    ------------ IMPLEMENTADO
    [F] QA interno   ------------ PARCIAL (SOW si, resto pendiente)
    [B] 5b.X         ------------ BLOQUEADO hasta datos
        |
        v (si 5b.X exitoso)
    [C] 5c comparativa legacy vs v1 con contrato v1.9
        |
        v (si 5c cierra)
    [D] 5d migracion 5 consumidores
        |
        v (si 5d cierra + 1 run CI verde)
    [E] 5e retirada legacy

**No hay atajos.** Cada frontera requiere dictamen externo antes de
proceder. En particular:

- No activar parametros SOW en config hasta 5b.X exitoso.
- No migrar consumidores hasta que 5c cierre.
- No retirar legacy hasta que 5d cierre + 1 run CI estable.

---

## 4. Fase B (SOW) - Detalle

### 4.1. Estado

    Fase 5b.2  CERRADA  -> K = 0.25 congelado
    Fase 5b.3  FAIL     -> primera ejecucion con sesgo de anclaje
    Dictamen 24         -> inmortal time bias detectado
    Fase 5b.4  FAIL sel -> landmark corregido. D3 inalcanzable.
    Dictamen 26         -> core landmark aprobado.
    Fase 5b.4-bis PASS  -> 3 bloques + delta_hetero. 17/240 pasan D1-D3.
    Dictamen 33         -> freeze de validacion autorizado.
    QA (34)             -> solo P3 con IC95 que no cruza 0.
    Contrato v1.9       -> candidata FROZEN_FOR_VALIDATION.
    5b.X                -> PENDIENTE.

### 4.2. Candidata congelada

    N     = 60
    M     = 30
    X_ATR = 0.25
    Y_VOL = 1.10

Documentada en `01_contrato_semantico_v1_9.md` seccion 13.

### 4.3. Evidencia de desarrollo

    lift_H20_ALL = +0.0740  IC95% [+0.0234, +0.1242]
    P1: +0.093  IC95 [-0.099, +0.296]  (cruza 0)
    P2: +0.050  IC95 [-0.024, +0.122]  (cruza 0)
    P3: +0.106  IC95 [+0.038, +0.178]  (no cruza 0)

Solo P3 tiene IC que no cruza 0. El agregado esta sostenido por P3.
Esto debilita cualquier lenguaje de "efecto robusto universal".

### 4.4. Protocolo 5b.X (35) - Estado

    Script validate_wyckoff_sow_5bX.py: LISTO.
    Estado actual: BLOCKED (sin datos post-2026-10-01).
    Umbrales propuestos: 12 meses + 50 ep + 20 conf.
    QA (36) demuestra que estos umbrales son insuficientes
    para poder estadistico.

### 4.5. Bloqueo real de 5b.X

5b.X NO puede ejecutarse hasta:

1. Existan datos posteriores a 2026-10-01.
2. Con suficiente muestra para tener poder estadistico
   (~200 confirmed segun QA 36).
3. El auditor resuelva P1-P5 del expediente 36 (umbrales,
   escenario D, umbral de lift).

El script esta preparado para las tres condiciones: chequea
muestra y bloquea si no es suficiente.

---

## 5. Fase C (comparativa legacy) - DESBLOQUEADA

**Estado: DESBLOQUEADA desde 5b.2** (plan v1 seccion 9, dictamen
externo). Lo unico bloqueado es 5c.4 y 5d, no la comparativa general.

Se puede ejecutar con el contrato v1.8 vigente (sin SOW activado,
porque fail-closed impide DISTRIBUTION sin sow_params). El
comportamiento de v1.8 sin SOW es funcional: clasifica MARKUP,
MARKDOWN, ACCUMULATION, RANGE y expone distribution_candidate via
meta.

**Nota de correccion (2026-10-02):** el plan v2 redactado antes de
este commit decia "se reescribira tras 5b.X". Era un error de
interpretacion. 5c nunca se bloqueo tras 5b.2.

Estado previsto:

- Comparar salidas de `indicators/wyckoff.py` (legacy v4.2) vs
  `indicators/wyckoff_v1.py` (contrato v1.9) sobre el universo
  y periodo actuales.
- Documentar diferencias:
  - Distribuciones de fases por sector/indice.
  - Diffs en top-5 de indices.
  - Diffs en `rws_z` (WLS).
  - Cambio en la composicion de `sector_wyckoff_distribution.csv`.
- Criterio de no-regresion: el sistema no cambia de forma traumatica
  la composicion del top-5 sin justificacion estructural.

**Documento de salida:** `13_comparativa_legacy_v1.md` reescrito como
v2, fechado post-5b.X.

---

## 6. Fase D (migracion) - Prevision

Consumidores directos (5) + indirectos (4). Orden por fan-out
creciente (empezar por el que menos propaga):

    1. indicators/sector_wyckoff_distribution.py (solo contadores)
    2. indicators/sector_breadth.py
    3. indicators/stock_leader.py
    4. indicators/index_phase.py (alimenta sector_regime_matrix)
    5. indicators/index_leaders.py

Indirectos (leen columnas derivadas, no importan el modulo).
Revalidar con E2E + snapshot pre/post en cada migracion:

    6. indicators/evidence_matrix.py        (wyckoff_evidence)
    7. indicators/sector_concentration.py   (wyckoff_median, coverage_wyckoff)
    8. indicators/sector_regime_matrix.py   (wyckoff_phase)
    9. indicators/slpm_v12.py               (wyckoff_phase, wyckoff_score)

**Cada migracion:** 1 commit + suite verde + verificacion E2E local
(obligatoria por tocar writers/readers del pipeline).

**Precondiciones:**

- 5b.X exitoso (contrato v1.9 confirmado).
- 5c cerrada.
- Dictamen externo autorizando migracion.
- Tests de contrato pasando sobre `wyckoff_v1.py`.

---

## 7. Fase E (retirada legacy) - Prevision

- Renombrar `indicators/wyckoff.py` a `indicators/wyckoff_legacy.py`
  (NO borrar; mantener por referencia).
- Ajustar imports en consumidores residuales.
- Actualizar `02_ARQUITECTURA.md` y `03_IAE.md`.
- Tras 1 run CI estable con v1.9 en produccion.

---

## 8. Analisis prohibidos en todas las fases

- Recalibrar parametros congelados sin abrir nueva fase + dictamen.
- Ampliar el grid de busqueda condicionada por resultados.
- Migrar consumidores sin 5c cerrada.
- Retirar legacy sin 5d + 1 run CI estable.
- Activar parametros en config sin 5b.X exitoso.
- Retirar fail-closed sin dictamen explicito.
- Cualquier modificacion cuyo objetivo implicito sea hacer pasar
  alguna configuracion.

---

## 9. Fuera de alcance

Este proyecto NO incluye:

- Rediseño del WLS.
- Rediseño del SLPM.
- Rediseño del pipeline de sectores.
- Cambios en `robust_zscore`.
- Cambios en `get_col`.

Si emergen durante alguna fase, se abren como sub-proyectos separados.

---

## 10. Bitacora del proyecto (actualizada)

| Fecha | Hito | Estado |
|---|---|---|
| 2026-10-02 | Fase 1 cerrada (semantica) | OK |
| 2026-10-02 | Fase 2 cerrada (estados) | OK |
| 2026-10-02 | Fase 3 cerrada (proveniencia) | OK |
| 2026-10-02 | Fase 4 cerrada (golden tests) | OK |
| 2026-10-02 | Fase 5a implementada (wyckoff_v1.py v1.3) | OK |
| 2026-10-02 | 5b.2 exitosa: K=0.25 congelado | OK |
| 2026-10-02 | 5b.3 ejecutada: FAIL por sesgo de anclaje | OK (diagnostico) |
| 2026-10-02 | Dictamen 24: inmortal time bias | OK |
| 2026-10-02 | 5b.4 ejecutada: FAIL por D3 inalcanzable | OK (diagnostico) |
| 2026-10-02 | Dictamen 26: core landmark aprobado | OK |
| 2026-10-02 | 5b.4-bis ejecutada: PASS desarrollo (17/240) | OK |
| 2026-10-02 | Dictamen 29: 5b.4-bis = PASS desarrollo | OK |
| 2026-10-02 | QA 5b.4-bis (34): solo P3 con IC positivo | OK |
| 2026-10-02 | Dictamen 33: freeze de validacion autorizado | OK |
| 2026-10-02 | Contrato v1.9 FROZEN_FOR_VALIDATION | OK |
| 2026-10-02 | Protocolo 5b.X (35) redactado | OK |
| 2026-10-02 | Script 5b.X redactado, bloqueado | OK |
| 2026-10-02 | QA precondiciones 5b.X (36): umbrales insuficientes | OK |
| PENDIENTE | Dictamen sobre umbrales de 5b.X (36) | - |
| PENDIENTE | 5b.X ejecucion (requiere datos post-2026-10-01) | - |
| PENDIENTE | 5c comparativa legacy vs v1 (post-5b.X) | - |
| PENDIENTE | 5d migracion 5 consumidores | - |
| PENDIENTE | 5e retirada legacy | - |

---

## 11. Frente F (QA interno) - Desglose

QA SOW ya cubierto (34, 36). Componentes no-SOW de
`indicators/wyckoff_v1.py` pendientes de revision:

- `_trend_component`: rolling MA50/MA200, formula trend.
  Contrato v1.3.
- `_atr_normalized`: ATR/Close, ventana 20.
  Contrato v1.3.
- `_volume_z`: robust_zscore(Volume, window=60, min_periods=20).
  Contrato v1.3.
- `_effort_vs_result`: tanh(effort_z - result_z).
  Contrato v1.3.
- `wyckoff_stability`: MAD rolling + tanh. K_STABILITY = 1.0
  (PROPUESTO, no calibrado).
- `classify_wyckoff_phase`: las 6 fases.

**Objetivo del frente F:** auditar cada uno como se hizo con el SOW:

1. Verificar que la implementacion coincide con el contrato v1.8.
2. Buscar `datetime.now()`, `np.median` sin `nanmedian`, y otros
   patrones de bug del catalogo.
3. Cobertura de tests: comprobar que cada componente tiene tests
   de contrato (no solo del flujo agregado).
4. Documentar hallazgos como expediente si son estructurales.

**Estado:** CERRADO (2026-10-02, commits a00ee2c + aaf7923).

Cobertura por componente:

- `_trend_component`: tests indirectos via robust_zscore (L353-369).
- `_atr_normalized`: `test_f03_atr_normalized_min_periods` +
  `test_f03_atr_normalized_valor_positivo_con_rango`.
- `_volume_z`: `test_f03_volume_z_delega_en_robust_zscore`.
- `_effort_vs_result`: `test_f03_effort_vs_result_en_rango`.
- `wyckoff_stability`: `test_i6_stability_in_range`.
- `classify_wyckoff_phase`: tests I36, I36b + flujo agregado.

El frente F completo (SOW + no-SOW) queda cerrado. Si emerge
deuda nueva, abrir frente nuevo; no reabrir F.

---

## 12. Criterio de exito global

El proyecto se considera cerrado cuando:

1. `indicators/wyckoff_v1.py` implementa el contrato vigente
   completo (v1.9 si 5b.X valida; v1.8 con la candidata en
   validacion si no).
2. Los golden tests de Fase 4 + tests I1-I36 pasan en verde.
3. 5b.X confirma o refuta la candidata con datos out-of-sample.
4. Los 5 consumidores migrados, suite verde (si 5b.X exitoso).
5. Un run CI verde con el nuevo modulo en produccion.
6. Documentos `13_comparativa_legacy_v1.md` (v2) y
   `02_ARQUITECTURA.md` actualizados.

**Si 5b.X refuta la candidata:** el proyecto se reorienta. Opciones:
abandonar SOW como confirmador, o abrir 5b.5 con nueva hipotesis.
Decision del auditor externo.

---

## 13. Riesgos identificados

| Riesgo | Mitigacion |
|---|---|
| La nueva clasificacion cambia el top-5 de sectores | Fase C (comparativa legacy vs v1). Documentar cada cambio. |
| El SOW no generaliza out-of-sample | 5b.X. Si refuta, se abandona o se rediseña. |
| Los umbrales de 5b.X son insuficientes para poder | QA 36. Auditor decide ajuste. |
| Retirada prematura del legacy rompe el pipeline | 5e solo tras 1 run CI estable con v1.9. |
| Consumidores con comportamientos implicitos no documentados | Tests de contrato en cada consumidor antes de migrar. |
| Performance: v1 mas costoso que legacy | Benchmark en 5a/5d. Si supera 2x, revisar. |
| Fronteras de bloques de validacion contaminadas | Cutoff 2026-10-01 explicito. Datos posteriores solo. |
| Presion por congelar antes de validacion | Contrato v1.9 declara explicitamente FROZEN_FOR_VALIDATION / NOT_PRODUCTION. |

---

## 14. Vigencias contractuales

- `01_contrato_semantico_v1_8.md`: contrato productivo vigente.
- `01_contrato_semantico_v1_9.md`: candidata SOW congelada para
  validacion. NO activacion productiva.
- `01_contrato_semantico_v1_7.md` y anteriores: historicos.

**La migracion de v1.8 a v1.9 en produccion solo ocurre si 5b.X es
exitoso y el auditor lo autoriza.**

---

**Fin del plan de migracion v2.**
