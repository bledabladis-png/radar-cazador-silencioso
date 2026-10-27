# CONTRATO SOW-HOLDOUT-IMMUTABLE - Holdout reservado para validacion SOW

**Estado:** BORRADOR. Pendiente de firma del auditor externo.
**Fecha:** 2026-10-27.
**Ciclo:** sesion 13.
**Frente:** SOW-HOLDOUT-IMMUTABLE (REGISTRO.yaml, OPEN).
**Corrige:** la descripcion previa del ticket decia "7/8 tickers en
holdout". El analisis 2026-10-27 sobre los parquets actuales dice 35/40.

---

## 1. Proposito

Declarar formalmente que dos parquets de datos son HOLDOUT INMUTABLE
para la validacion SOW, y fijar las condiciones bajo las que (si acaso)
un ticker puede promoverse desde ellos a produccion.

Sin este contrato, promover tickers desde holdout a produccion es
ambiguo: no hay declaracion explicita de que los parquets sean holdout,
solo se infiere de su uso en scripts de validacion. Eso viola el
principio de que toda decision no obvia se justifica y se documenta.

---

## 2. Definiciones

### 2.1. Parquet holdout cohort

`data/stock_prices_reserved_cohort.parquet`

- Contenido: 268 tickers del S&P 500 que NO estan en
  `stock_prices_extended.parquet`.
- Rango: 2015-01-02 -> 2026-10-24.
- Generado por: `scripts/fetch_reserved_cohort.py`.
- Gitignored: `.gitignore:96`. Regenerable, no versionado.

### 2.2. Parquet holdout r3000

`data/stock_prices_reserved_r3000.parquet`

- Contenido: 3423 tickers del Russell 3000 que NO estan en
  `stock_prices_extended.parquet`.
- Rango: 2015-01-02 -> 2026-10-24.
- Generado por: `scripts/fetch_reserved_cohort_r3000.py`.
- Gitignored: `.gitignore:97`. Regenerable, no versionado.

### 2.3. Holdout inmutable

Un parquet es holdout inmutable si:

- Su contenido no se modifica por el pipeline diario.
- Su contenido no se modifica por scripts de regeneracion historica.
- Sus tickers NO se promueven a produccion sin decision firmada.
- Los scripts de validacion que lo consumen tienen pre-registro
  congelado (criterio de exito declarado ANTES de ejecutar).

---
## 3. Consumidores declarados

### 3.1. Generadores

- `scripts/fetch_reserved_cohort.py`: genera `reserved_cohort.parquet`.
- `scripts/fetch_reserved_cohort_r3000.py`: genera `reserved_r3000.parquet`.

Los generadores son read-only respecto a produccion. No tocan
`stock_prices*.parquet` ni `config/`.

### 3.2. Validadores con pre-registro congelado

- `scripts/purged_kfold_sow_c1_reserved.py`: consume `reserved_cohort`.
- `scripts/purged_kfold_sow_c1_r3000.py`: consume `reserved_r3000`.

Ambos declaran en su docstring:

- Candidata congelada: C1-M5 FROZEN, N=40, M=5, X_ATR=1.00, Y_VOL=1.10.
- Folds 2017..2025, embargo 50, purga H=20.
- Criterio FUNCIONA: mean(lift) > 0.05 AND IC95_lower > 0 AND
  folds_positive >= 7/9 AND folds_beat_placebo_B >= 7/9.
- Nota: "No modificar tras ver resultados".

El pre-registro congelado es la garantia de que el holdout no se ha
usado para optimizar. Cualquier cambio en esos scripts invalida el
holdout como tal y requiere nueva version del contrato.

### 3.3. Consumidores NO declarados

Cualquier otro script o modulo que lea de estos parquets debe:

- Declararse en este contrato antes de usarlo.
- Justificar por que necesita el holdout y no produccion.
- No modificar el parquet.

---

## 4. Reglas de inmutabilidad operativa

I1. El pipeline diario (`py run.py`) NO lee de los parquets holdout.

I2. Los scripts de regeneracion historica
(`regenerate_wyckoff_history.py`, `build_extended_dataset.py`) NO
leen de los parquets holdout.

I3. Los parquets holdout NO se versionan en git (gitignored). Su
contenido es reproducible con los generadores, pero su estado actual
no se recupera desde git. Si se pierden, hay que regenerarlos y
verificar que el rango y contenido siguen siendo compatibles con el
pre-registro congelado.

I4. Cualquier modificacion del contenido (anadir tickers, extender
rango, corregir OHLC) requiere version bump del contrato y re-firma
del auditor.

---

## 5. Estado real de la lista de tickers faltantes (2026-10-27)

La descripcion previa de `TOP20-FALTANTES-DATOS` decia "7 en
holdout, 1 ausente, 44 requieren descarga nueva". El analisis
2026-10-27 sobre los parquets actuales corrige esa cifra:

| Clasificacion | N | Tickers |
|---|---|---|
| Ya en produccion (pocas barras) | 3 | WBD (1), HONA (92), SPCX (93) |
| Solo en holdout (cohort o r3000) | 35 | ACN, ADBE, ADI, AFL, AJG, ALL, AON, APH, APO, BSX, CAH, CDNS, CI, CNP, COR, DELL, FTNT, GLW, HCA, HPE, ICE, INTU, KKR, LITE, MCO, MET, MRSH, MSI, NKE, NOW, QCOM, SHOP, SNPS, TFC, WDC |
| Nadie (ni produccion ni holdout) | 2 | DDOG, MRNA |

Total unicos: 40 (el CSV original tenia 52 filas por duplicacion
entre grupos: QCOM en XLK/QQQ/SPY, APH en XLK/SPY, etc.).

**Correccion respecto al ticket:**

- "7 en holdout" -> 35 reales.
- "44 requieren descarga nueva" -> 2 reales (DDOG, MRNA).
- "1 ausente total" -> 2 reales (DDOG, MRNA).
- El CSV `p1_h02_tickers_sin_datos.csv` es correcto en su definicion
  (ticker no en `stock_prices.parquet` con n_bars >= 200). El error
  fue interpretarlo como "no en ningun sitio".

---

## 6. Condiciones de promocion holdout -> produccion

Un ticker del holdout puede promoverse a produccion (anadirse a
`data/stock_prices.parquet` o a `data/universe_expansions.yaml`) si y
solo si se cumplen TODAS:

P1. El pre-registro congelado del validador correspondiente ya ha
    emitido su veredicto (FUNCIONA o NO_FUNCIONA). No se promueve
    antes de conocer el resultado, para no contaminar el holdout con
    informacion del sistema.

P2. El auditor externo ha firmado explicitamente la promocion. No
    basta con que el ticket pase a FIXED; requiere mencion explicita
    del ticker o del grupo de tickers.

P3. La promocion no invalida el holdout para futuros usos. Si el
    ticker se usa en produccion, deja de ser holdout para ese ticker
    concreto. El resto del parquet sigue siendo holdout.

P4. La promocion se registra en REGISTRO.yaml con SHA del commit que
    la aplica, lista de tickers promovidos y referencia al veredicto
    del validador.

---

## 7. Tests de contrato requeridos

Antes de considerar el contrato implementado, deben existir tests
para:

T1. El pipeline diario no lee de los parquets holdout. Verificable por
    AST o por grep controlado en `src/pipeline/` e `indicators/`.

T2. Los scripts de regeneracion no leen de los parquets holdout.
    Verificable por grep controlado en `scripts/`.

T3. Cualquier script que lea de los parquets holdout esta declarado en
    este contrato. Verificable por comparacion entre consumidores
    reales (grep) y consumidores declarados (seccion 3).

T4. Los parquets holdout siguen gitignored. Verificable por
    `git check-ignore`.

T5. El pre-registro congelado de los validadores no ha cambiado desde
    la firma del contrato. Verificable por hash SHA256 de los dos
    scripts `purged_kfold_sow_c1_*.py`.

---

## 8. Procedimiento de promocion (cuando se apruebe)

1. Confirmar P1 (veredicto emitido) y P2 (firma del auditor).
2. Anadir el ticker a `data/universe_expansions.yaml` en el bloque
   del grupo correspondiente, con `fecha` y `motivo`.
3. Ejecutar `py run.py` para descargar el ticker al parquet
   productivo.
4. Verificar cobertura efectiva del grupo con
   `py scripts/audit/p1_h02_cobertura_efectiva.py`.
5. Registrar en REGISTRO.yaml: tickers promovidos, SHA del commit,
   veredicto del validador citado.
6. Confirmar que el ticker desaparece de
   `outputs/audit/p1_h02_tickers_sin_datos.csv`.
7. No borrar el ticker del parquet holdout: deja de ser holdout para
   ese ticker, pero el parquet conserva la historia.

---

## 9. Condiciones de firma

El contrato se considera firmado si el auditor externo responde
explicitamente con:

- Aprobado: se aplica tal cual. `SOW-HOLDOUT-IMMUTABLE` pasa a FIXED
  con SHA del commit del contrato.
- Aprobado con cambios: se anotan los cambios, se versiona como
  v1.1, se re-revisa.
- Rechazado: se documenta la razon en REGISTRO.yaml. Los parquets
  holdout se siguen usando por los validadores existentes, pero no
  se declaran inmutables formalmente. `TOP20-FALTANTES-DATOS` queda
  bloqueado en su parte de holdout hasta que se resuelva.

Sin respuesta explicita, el ticket sigue OPEN y no se promueve
ningun ticker desde holdout.

---

## 10. Impacto en TOP20-FALTANTES-DATOS

Este contrato desbloquea el analisis, no la accion:

- **35 tickers en holdout:** pueden promoverse cuando se cumplan
  P1+P2. Mientras tanto, siguen fuera de produccion. No es deuda de
  descarga, es deuda de decision.
- **2 tickers nadie (DDOG, MRNA):** descarga real. No dependen de
  este contrato. Se pueden abordar ya.
- **3 tickers con pocas barras (WBD, HONA, SPCX):** problema de IPO
  reciente, no de holdout ni de descarga. Se resuelven solos con el
  tiempo (mas sesiones bursatiles).

Una vez firmado este contrato, `TOP20-FALTANTES-DATOS` puede
reformularse como:

- "35 pendientes de decision de promocion" (bloqueado por auditor).
- "2 pendientes de descarga" (DDOG, MRNA, sin bloqueo externo).
- "3 pendientes de historico suficiente" (WBD, HONA, SPCX, sin accion
  posible).

---

## 11. Estado normativo

Borrador. Sin validez normativa hasta firma del auditor externo.
Cualquier promocion de tickers desde holdout antes de la firma es una
violacion del contrato del sistema.

---

Fin del contrato SOW-HOLDOUT-IMMUTABLE (borrador).

