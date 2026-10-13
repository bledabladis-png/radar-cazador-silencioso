# Anexo de verificacion - Auditoria MACRO SECTORIAL v4.3

**Fecha:** 2026-10-13
**Objeto:** Anexo de verificacion solicitado por el auditor en su dictamen
de seguimiento del 13-oct-2026 (seccion 4, tabla de pruebas exigidas).
**Rango de commits:** 0413518e..6607a773 (todos en `main`).
**Suite al cierre:** 3273 passed + 5 skipped + 0 failed.

---

## 1. Pruebas por hallazgo

### P1.1 - Leader Breadth usa rs_ratio_20d

- **Commit:** 0413518e (fix) + 00b01f46 (N1 fail-closed).
- **Tests (en `tests/test_slpm_state_machine_d31.py`):**
  - `test_n1_sin_rs_ratio_20d_no_cae_a_criterio_nominal`: con `rs=5.0` y
    sin `rs_ratio_20d`, `rs_breadth = NaN` y status
    `INSUFFICIENT_RS_RATIO`.
  - `test_n1_con_rs_ratio_20d_parcial`: si solo algunos lideres tienen
    el campo, se calcula sobre los validos y el status lo declara.
  - `test_n1_rs_ratio_20d_negativo_no_es_positivo`: valor negativo no
    cuenta como positivo.
  - `test_breadth_v2_todos_favorables` y
    `test_breadth_v2_todos_desfavorables`: casos extremos.
- **Benchmark y horizonte:** `rs_ratio_20d = (1+ret20_accion)/(1+ret20_ETF) - 1`,
  mismo horizonte (20 sesiones) para ambos, precios ajustados via
  Yahoo Close.

### P1.2 - Retornos 20d

- **Commit:** efa68e9e.
- **Test:** `tests/test_i1_o1_notas.py::test_i1_momentum_sectores_incluye_nota`
  actualizado para validar la nota nueva ("retorno del ETF sectorial").
- **Verificacion cruzada:** `compute_price_momentum` aplica
  `close.pct_change(20)` sobre el ETF (no mediana de componentes).

### P1.3 - WLS centralizado

- **Commit:** d6a238bf (contrato) + 6b3466df (N3).
- **Tests (en `tests/test_wls_validation_n3.py`):**
  - `test_n3_componentes_declarados`: los 4 componentes coinciden con
    `WLS_COMPONENTS`.
  - `test_n3_wls_persistence_en_rango_ok`
  - `test_n3_wls_persistence_mayor_1_se_clipea` (persistence=70 -> 1.0
    + status PERSISTENCE_OUT_OF_RANGE + warning)
  - `test_n3_wls_persistence_negativa_se_clipea`
  - `test_n3_wls_componente_ausente_nan` (status
    INSUFFICIENT_COMPONENTS)
  - `test_n3_wls_persistence_nan_no_out_of_range`
  - `test_n3_wls_componente_ausente_y_persistencia_out_of_range`
  - `test_n3_wls_multiples_filas_estados_distintos`
- **Pesos leidos desde config:** `WLS_WEIGHTS` en `config/weights.py`.
  Los valores por defecto del calculo y los de la nota del reporte
  derivan del mismo dict.

### P1.4 - IWM no entra en Europa Primary Flow

- **Evidencia de codigo:** `src/pipeline/flows_secondary.py` lineas
  51-71. `european_flows` se construye SOLO con
  `blackrock_dax_flow`, `blackrock_isf_flow`, `amundi_lyxi_flow`.
  `blackrock_iwm_flow` no participa.
- **Cambio de codigo:** ninguno (el codigo es correcto).
- **Nota anadida:** 43f10428 (nota en render que separa los 4 fondos
  por region).

### P1.5 - Clasificacion en 3 niveles

- **Commit:** 189c1935.
- **Cambio:** `EVIDENCE_CLASSIFICATION` en `src/dependency_tracker.py`
  pasa de 2 niveles (INDEPENDIENTE / DERIVADA) a 3 (OBSERVABLE /
  DERIVADA-DIRECTA / DERIVADA-COMPUESTA).
- **Nota al pie:** "DERIVADA-DIRECTA no implica independencia
  estadistica. Breadth, RS y Flow Proxy comparten informacion del
  precio."
- **N4 ampliacion (commit 46bcb886):** nota en `render_matriz_evidencia`
  declarando que la agregacion es heuristica de signos (1 voto por
  variable, sin pesos).

### P1.6 - Data Conf vs frescura

- **Commits:** bbd0cef9 + 6607a773 (N6.1).
- **Cambio:** nota en `freshness.py` aclara que Data Conf es runtime y
  no se publica en la tabla completa (esa declara frescura por fuente).

### P2.1 - Wyckoff=+1 en 11/11

- **Commit:** 461af305.
- **Tests (en `tests/test_evidence_matrix.py`):**
  - `test_wyckoff_sign_favorable_mayor`
  - `test_wyckoff_sign_desfavorable_mayor`
  - `test_wyckoff_sign_empate_devuelve_cero`
  - `test_wyckoff_sign_todo_cero_devuelve_cero_no_mas_uno`
  - `test_wyckoff_sign_nan_devuelve_nan`
  - `test_wyckoff_sign_campos_ausentes_devuelve_nan`

### P2.2 - Cobertura baja Wyckoff

- **Commits:** 386e6af5 (constante) + fix de test.
- **Cambio:** constante `WYCKOFF_MIN_COVERAGE_PCT = 50.0` en
  `config/settings.py`. Marcador `[COB BAJA]` en el render para
  sectores por debajo del umbral.

### P2.3 - NH/NL = 0

- **Commit:** efa68e9e.
- **Verificacion:** `n_valid_nhnl = 20/20` en los 11 sectores (datos
  reales del ultimo run). Definicion: 252 sesiones previas.
- **Nota:** aclaracion explicita en el render.

### P2.4 / N6.2 - Percentiles PCR

- **Commits:** bbd0cef9 (nota inicial) + 6607a773 (correccion).
- **Correccion aplicada:** la nota inicial decia "252 sesiones" para
  Perc PCR. **Era incorrecto.** La columna `pcr_percentile_20d` usa
  ventana de 20 sesiones. Corregido. El "Percentil" de Sentimiento usa
  `min(252, len)`.

### P2.5 - STAGFLATION

- **Commits:** 189c1935 + 6603c774.
- **Nota actual:** sin umbrales hardcodeados. Referencia a
  `indicators/mte/scoring.py::score_scenarios` y `SCENARIO_WEIGHTS`.

### P2.6 - Titulo 13F

- **Commit:** 189c1935.
- **Cambio:** "Acumulacion Institucional (13F)" ->
  "Variacion neta de posiciones declaradas (13F)".

### P2.7 / N5 - Asincronia precio-flujo

- **Commits:** 6603c774 (nota inicial) + 1a7181cc (refuerzo).
- **Decision adoptada:** mantener la comparacion asincrona pero
  etiquetarla explicitamente con la fecha del flujo en el encabezado
  y nota reforzada. La lectura se declara "de posicion relativa, no de
  senal contemporanea".

### P2.8 - Coberturas identicas

- **Commit:** bbd0cef9.
- **Nota:** explicita que las 5 columnas derivan del mismo `dropna`
  comun (interseccion de tickers con todas las metricas).

### P3.1 - Redondeo QQQ

- **Commit:** bbd0cef9.
- **Nota:** aclaracion del calculo sobre valores sin redondear.

### P3.2 - VRP como pp

- **Commit:** bbd0cef9 + correccion posterior.
- **Cambio:** format string `{:+.2f}%` -> `{:+.2f}pp`.

---

## 2. N1 a N6 - Nuevos hallazgos del auditor

| ID | Commit | Evidencia |
|---|---|---|
| N1 | 00b01f46 | Fail-closed en `rs_ratio_20d`. 3 tests en `test_slpm_state_machine_d31.py`. |
| N2 | a3a601be | `lis_validated=False` en 3 returns de `compute_leader_integrity`. Nota provisional en render. 2 tests. |
| N3 | 6b3466df | `_validate_wls_inputs` funcion pura + 8 tests en `test_wls_validation_n3.py`. |
| N4 | 46bcb886 | Nota "heuristica descriptiva" en `render_matriz_evidencia`. |
| N5 | 1a7181cc | Fecha del flujo en encabezado + nota asincronia reforzada. |
| N6.1 | 6607a773 | Formula exacta de cobertura por `coverage_logic`. |
| N6.2 | 6607a773 | Correccion ventanas PCR (20 vs 252). |

---

## 3. Hallazgo N7 - Bug propio detectado durante esta auditoria

**Descripcion.** `evidence_matrix.csv` y `sector_regime_matrix.csv`
contenian **1 fila por fecha** en lugar de 11 (una por sector).

**Causa.** En el registro declarativo
(`src/pipeline/output_registry.py`) introducido el 2026-10-11, ambos
`OutputSpec` declaraban `merge_keys=('date',)`. `append_dedup` deduplica
por esas columnas: al no incluir `sector`, cada run conservaba 1 sector
y descartaba los otros 10.

**Alcance.** Bug **pre-existente** a la migracion (los writers
originales ya sobreescribian el CSV con 1 fila). La migracion no lo
introdujo pero tampoco lo corrigio: el registro copio el patron
incorrecto.

**Fix.** Commit f5db9f5f: `merge_keys=('date','sector')` en ambos
specs. Test permanente
`test_merge_keys_incluye_sector_si_csv_lo_tiene` en
`tests/test_output_registry.py` que recorre el registro y detecta
cualquier futura incoherencia entre columnas del CSV y `merge_keys`.

**Impacto historico.** Las filas perdidas (10 de 11 por fecha) son
**irrecuperables desde git**: el bug existe en el writer original, no
en la migracion. A partir del proximo run con el fix, cada fecha
tendra las 11 filas.

**Leccion.** La verificacion estatica de coherencia CSV vs registro
que faltaba en la migracion original. Ahora existe como test.

---

## 4. Regeneracion del reporte

**Estado:** PENDIENTE.

Por politica del corpus (`01_METODO` seccion 4): no usar `run.py` en
horario de mercado USA (13:30-20:00 UTC). El run de validacion se hara
con el cron programado de las **23:17 UTC** del 13-oct-2026, que
procesara la sesion del 13-oct con el contrato completo.

**Verificacion comprometida para el 14-oct:**
1. `gh run list --workflow=daily_run.yml --limit 3` para confirmar
   ejecucion.
2. `gh run view <id> --log | grep "Verificacion de outputs"` para
   confirmar `19 checks`.
3. Descargar el `daily-report` del run y comparar con el del 13-oct.
4. Verificar `evidence_matrix.csv` y `sector_regime_matrix.csv` con 11
   filas por fecha en el commit del bot.

---

**Fin del anexo.**
