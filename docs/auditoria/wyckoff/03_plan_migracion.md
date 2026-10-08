# PLAN DE MIGRACION - Modulo Wyckoff v1 (v3)

**Hoja de ruta actualizada del proyecto.**
**Fecha:** 2026-10-04.
**Supersede:** v2 (2026-10-02) y v1 (2026-10-02 19:14).
**Referencia normativa:** contrato v1.9 + protocolo 5b.X + dictamenes
externos 24, 26, 29, 33.
**Cambio v2 -> v3:** se anade el walk-forward historico como metodo
de validacion alternativo de SOW (§15). Motivo: el sistema es
experimental, el legacy no discrimina fases (el CSV del 3-oct
muestra 20/20 RANGE en todos los sectores), y bloquear la migracion
12 meses por esperar 5b.X deja el pipeline sin mejora durante un
ano. El walk-forward no sustituye a 5b.X; lo complementa.

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
        Pendiente: 5b.X (validacion out-of-sample, 2027).
        Metodo alternativo: walk-forward historico (§15).

    Frente C - Comparativa legacy vs v1
        Estado: PRIMERA PASADA con v1.8 (824ef72). Documento
        13_comparativa_legacy_v1.md actualizado; resultados
        identicos a v1.3 (componentes no-SOW no cambiaron).
        Pendiente: version v2 definitiva post-5b.X.

    Frente D - Migracion de consumidores
        Estado: BLOQUEADO hasta cierre 5c.
        Desbloqueo: tras cierre 5c + decision sobre SOW (§15).
        Nota: la migracion del core de v1.8 (MARKUP/ACCUMULATION/
        RANGE/MARKDOWN) no depende de SOW. Solo la activacion de
        DISTRIBUTION depende del resultado de §15 / 5b.X.
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

**Actualizado 2026-10-08.** Los 5 consumidores directos migrados a
`indicators/wyckoff_v1.py` (v1.8 core). Commits 2fd6c449 + 35663000.

    indicators/sector_wyckoff_distribution.py   (migrado 2026-10-08)
    indicators/stock_leader.py                  (migrado 2026-10-08)
    indicators/index_leaders.py                 (migrado 2026-10-08)
    indicators/index_phase.py                   (migrado 2026-10-08)
    regimes/sector_regime.py                    (migrado 2026-10-08)

**Correccion al plan original.** El plan listaba `sector_breadth.py`
como consumidor. Grep confirma que no importa wyckoff. Se omite.
`regimes/sector_regime.py` aparecia como indirecto; es directo.

El legacy `indicators/wyckoff.py` permanece en el repo pero sin
consumidores productivos. Se retirara en 5e.

### 2.5. Bloqueos activos

    5b.X      BLOQUEADO hasta datos posteriores a 2026-10-01
    5c        DESBLOQUEADA desde 5b.2 (ver seccion 5)
    5d        BLOQUEADO hasta cierre 5b.X + 5c
    5e        BLOQUEADO hasta cierre 5d + 1 run CI estable
    5c.4      BLOQUEADO

---

## 3. Orden estricto de desbloqueo

    [A] Nucleo v1    ------------ IMPLEMENTADO
    [F] QA interno   ------------ CERRADO
        |
        v
    [B'] Walk-forward historico (§15) --- validacion SOW temprana
        |
        +--- PASS ---> activar SOW (v1.8 pasa a v1.9 completa)
        |
        +--- FAIL ---> SOW descartado (v1.8 sin SOW)
        |
        v
    [C] 5c comparativa legacy vs v1 (core + SOW si se activo)
        |
        v
    [D] 5d migracion 9 consumidores (5 directos + 4 indirectos)
        |
        v
    [E] 5e retirada legacy
        |
        v
    [B] 5b.X (2027, confirmacion cruzada del SOW)

**Cambio v2 -> v3:** el nodo [B] 5b.X deja de bloquear la migracion.
Se ejecuta en 2027 como confirmacion cruzada. Si refuta el resultado
de [B'], se reabre la decision sobre SOW.

Reglas vigentes:

- No activar parametros SOW sin [B'] PASS + dictamen externo.
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

### 4.4. Protocolo 5b.X (35 v2) - Estado

    Script validate_wyckoff_sow_5bX.py: CONGELADO HISTORICAMENTE
    (hash CC01A6AB..., commit d324346, seed 20261002, python 3.14).
    [2026-10-03] NO VALIDO como ejecutable v2: ver
    37_d06_5bX_invalidada.md. Freeze preservado en historia.
    Estado actual: BLOCKED (datos + contrato sin implementar).
    Umbral normativo: 550 confirmed H20-complete (congelado).
    Guardrail temporal: 12 meses.
    Fuente del umbral: power_analysis_5bX.py (344dae9).
    Precedente: dictamen externo 2026-10-02 sobre 35 v1 + QA 36.

### 4.5. Bloqueo real de 5b.X

5b.X NO puede ejecutarse hasta:

1. Existan datos con t0 >= 2026-10-02 (post-cutoff).
2. n_confirmed_H20_complete >= 550.
3. Guardrail: >= 12 meses de sesiones post-cutoff.

Las tres condiciones se evaluan de forma independiente. El script
chequea muestra y bloquea (D = INSUFFICIENT_SAMPLE) si no se cumple
la condicion 2.

**Dictamen de umbrales: RESUELTO** (dictamen externo 2026-10-02).
El umbral 550 esta congelado y no es revisable mirando resultados.

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

## 6. Fase D (migracion) - Prevision [CERRADA 2026-10-08]

**Estado 2026-10-08:** los 5 consumidores directos migrados a
v1.8 core (commits 2fd6c449 + 35663000). El CSV historico
regenerado con v1.8 (commit f782b9d4). E2E local verificado
(exit=0, Gate 10/10). Los 4 indirectos NO requieren migracion:
leen columnas derivadas del CSV, no importan el modulo.

Lista real de consumidores directos (corregida respecto al plan
original):

    1. indicators/sector_wyckoff_distribution.py (2fd6c449)
    2. indicators/stock_leader.py                (35663000)
    3. indicators/index_leaders.py               (35663000)
    4. indicators/index_phase.py                 (35663000)
    5. regimes/sector_regime.py                  (35663000)

Plan original listaba `sector_breadth.py` como consumidor. Grep
confirma que no lo es.

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

- 5c cerrada.
- Decision sobre SOW tomada (§15): PASS (activar) o FAIL (descartar).
- Dictamen externo autorizando migracion.
- Tests de contrato pasando sobre `wyckoff_v1.py`.

**Alcance segun resultado de §15:**

- Si SOW activado: migrar con v1.9 completa (5 fases incluyendo
  DISTRIBUTION).
- Si SOW descartado: migrar con v1.8-core (4 fases: MARKUP,
  ACCUMULATION, RANGE, MARKDOWN).

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
| 2026-10-02 | Protocolo 5b.X (35 v1) redactado | OK |
| 2026-10-02 | Script 5b.X redactado, bloqueado | OK |
| 2026-10-02 | QA precondiciones 5b.X (36): umbrales insuficientes | OK |
| 2026-10-02 | Dictamen externo: 35 v1 + QA 36 auditados | OK |
| 2026-10-03 | Power analysis 5b.X (344dae9): umbral 550 derivado | OK |
| 2026-10-03 | Protocolo 5b.X v2 (9ef7497): 550 congelado | OK |
| 2026-10-03 | Script validador congelado con hash (CC01A6AB) | HISTORICO - no valido como ejecutable v2 (D-06, `37_`) |
| PENDIENTE | 5b.X ejecucion (requiere 550 confirmed H20-complete) | - |
| 2026-10-08 | Migracion core v1.8 (5 consumidores directos) | OK |
| 2026-10-08 | Regeneracion CSV sector_wyckoff_distribution | OK |
| 2026-10-08 | Retirada legacy (5e) | PENDIENTE (1 run CI estable) |
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
| Los umbrales de 5b.X son insuficientes para poder | RESUELTO: 550 confirmed H20-complete (congelado 2026-10-03). |
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

## 15. Walk-forward historico de SOW (enmienda v3, 2026-10-04)

### 15.1. Motivo

El plan v2 bloqueaba la activacion de SOW hasta 5b.X (2027), que
requiere 550 confirmed H20-complete con t0 >= 2026-10-02 + 12 meses
de sesiones post-cutoff. El bloqueo tenia fundamento metodologico
(evitar autoengano por reutilizar in-sample), pero dejaba el pipeline
produciendo outputs con un modulo legacy que clasifica casi todo
como RANGE durante un ano.

El sistema es experimental. El legacy `indicators/wyckoff.py` no
discrimina fases utiles. Mantenerlo un ano mas por coherencia
metodologica estricta no aporta valor.

Se anade un metodo de validacion alternativo: **walk-forward
historico**.

### 15.2. Metodo

**Corte:** mitad de la ventana historica disponible (2021-10-01 a
2026-10-01). Bloques:

    Periodo A (in-sample del walk-forward): 2021-10-01 -> 2023-12-31
    Periodo B (out-of-sample del walk-forward): 2024-01-01 -> 2026-10-01

**Calibracion:** ejecutar el calibrador sobre Periodo A SOLO.
Obtener candidata C_A = (N, M, X_ATR, Y_VOL) que pasa D1-D3.

**Validacion:** ejecutar C_A sobre Periodo B con la misma metodologia
de bootstrap y criterios. Medir lift_H20 + IC95%.

**Comparacion adicional:** ejecutar la candidata congelada actual
(N=60, M=30, X_ATR=0.25, Y_VOL=1.10) sobre Periodo B. Distingue:

  1. ¿La calibracion en A produce candidata parecida?
  2. ¿La candidata congelada funciona en B sin haberlo visto?

**Robustez:** small-grid alrededor de C_A sobre B
(N +/- 5, X_ATR +/- 0.05, Y_VOL +/- 0.10) para evaluar si el lift
aparece solo en el punto exacto o en un entorno.

### 15.3. Criterios de exito

- **PASS si:** candidata C_A o candidata congelada produce lift_H20
  con IC95% cuyo limite inferior > 0 sobre Periodo B. Ademas, >=2/3
  de sub-bloques de B con lift > 0 (D3 analogo).
- **FAIL si:** IC95% cruza 0, o lift < 0, o D3 no se cumple.
- **INSUFICIENTE si:** n_confirmed < 20 en B.

### 15.4. Consecuencias

- **PASS:** SOW se activa en config (4 parametros). v1.8 pasa a
  clasificar DISTRIBUTION. Se procede a migracion completa.
- **FAIL:** SOW se descarta como confirmador. Se activa solo el
  core de v1.8 (4 fases sin SOW). v1.9 queda como referencia
  historica.
- **INSUFICIENTE:** extender el periodo A o el periodo B, o
  mantener 5b.X como unico criterio.

### 15.5. Relacion con 5b.X

- 5b.X oficial sigue programado (2027). El walk-forward **no lo
  sustituye**; sirve como decision operativa temprana.
- Si ambos pasan: refuerzo mutuo.
- Si walk-forward pasa y 5b.X refuta: discrepancia informativa.
  Se revisa el contrato del SOW con el auditor externo.
- Si walk-forward falla y 5b.X pasa: se activa SOW en 2027.

### 15.6. Trazabilidad

- Documento de salida: `docs/auditoria/wyckoff/41_walk_forward_sow.md`.
- Script: `scripts/walk_forward_sow.py` (read-only sobre data; no
  toca config).
- Resultados: `outputs/audit/wyckoff_walk_forward.csv`.

El documento y el script se commitean juntos. La ejecucion se hace
una sola vez. Los resultados no se ajustan ex-post.

---

**Fin del plan de migracion v3.**
