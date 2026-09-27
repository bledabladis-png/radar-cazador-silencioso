# Informe O1 - Ausencia de imputacion en el modulo IAE

**Fecha:** 2026-09-28
**Referencia:** hallazgo O1 del dictamen externo (`AUDITORIA_EXTERNA_2026-09-27.md` seccion 3.6)
**Estado:** CERRADO con un fix aplicado + evidencia empirica

---

## 1. Motivo

El auditor externo observo que la cobertura de tests (90%) no equivale
a prueba semantica. Pidio evidencia especifica de ausencia de
imputacion (`fillna`, `ffill`, `bfill`, `interpolate`, `pad`, `backfill`,
`combine_first`) en TODOS los caminos del modulo IAE.

---

## 2. Metodologia

Gate 0 read-only: barrido regex sobre `src/institutional_accumulation/**/*.py`
buscando los 8 patrones de imputacion. Clasificacion de cada hit por
contexto (linea +- 8).

---

## 3. Resultado del barrido

| Patron | Hits |
|---|---:|
| `fillna` | 4 |
| `ffill` | 0 |
| `bfill` | 0 |
| `interpolate` | 0 |
| `pad` | 0 |
| `backfill` | 0 |
| `combine_first` | 0 |
| `dropna` (no imputa) | 8 |
| **TOTAL imputaciones** | **4** |

---

## 4. Clasificacion de los 4 `fillna`

### 4.1. `pipeline_contractual.py:173` - PROBLEMATICO (CORREGIDO)

Antes:

    sshp = pd.to_numeric(op["SSHPRNAMT"], errors="coerce").fillna(0.0)

Imputaba 0.0 a filas con SSHPRNAMT no-parseable. Contradecia el
principio "no imputar" y era incoherente con
`delta_shares.py::_filter_canonical`, que descarta esas filas.

Despues (fix O1):

    op = op.assign(_sshp=pd.to_numeric(op["SSHPRNAMT"], errors="coerce"))
    op = op.dropna(subset=["_sshp"])
    sshp = op["_sshp"]

Alineado con la politica del modulo: descartar, no imputar.

**Impacto:** nulo sobre el NIPC observable (-4.264.449.012 sin cambios).
Las filas con SSHPRNAMT no-parseable no contribuian materialmente al
delta. El fix es de coherencia, no de magnitud.

### 4.2. `amendments.py:124` - LEGITIMO

    pd.to_numeric(out["_amendment_no"], errors="coerce").fillna(0).astype(int)

`_amendment_no` es una clave de sort, no un dato contractual.
Imputar 0 significa "sin numero de amendment" (equivale a filing base).
No entra en NIPC.

### 4.3. `delta_shares.py:130` - LEGITIMO

    _toc = df["TITLEOFCLASS"].fillna("").astype(str)

Uso local: transformar None en "" para aplicar una regex. La columna
original no se modifica. La fila descartada o conservada depende de si
la regex matchea; el "" no se propaga a ningun calculo.

### 4.4. `pipeline_contractual.py:71` - LEGITIMO

    x = series.fillna("").astype(str).str.strip()

Uso local en `ticker_of()`: extraer el ticker de `canonical_security`.
Si `canonical_security` es None, se devuelve None aguas abajo (via
`.where(ok, None)`). El "" es un valor intermedio, no se imputa.

---

## 5. Clasificacion de los 8 `dropna`

Todos legitimos: descartan filas sin dato, no imputan. Ningun caso de
"rellenar un hueco con un valor fabricado".

- `delta_shares.py:141`: descarta SSHPRNAMT_f NaN (filtro pre-agregacion).
- `nipc.py:70`: descarta deltas NaN para sumar NIPC.
- `nipc.py:188, 195`: set de observed_security_key no nulos.
- `pipeline_contractual.py:137`: set de tickers no nulos del catalogo.
- `amendments.py:289`: lista de amendment nos no nulos.
- `relationships.py:137`: descarta edges sin OTHERMANAGER_SK.
- `security_identity.py:322`: descarta filas sin canonical_security.

---

## 6. Conclusion

**Antes del fix:** 1 imputacion problematico (SSHPRNAMT=0.0).

**Despues del fix:** 0 imputaciones sobre datos contractuales. Los 3
`fillna` restantes son transformaciones locales (regex, extraccion de
ticker, clave de sort) que no modifican el valor contractual ni se
propagan a calculos del NIPC.

**Evidencia:** barrido reproducible con el mismo patron del Gate 0.
Reejecutable tras cualquier commit.

**Recomendacion:** mantener este barrido como check de CI tras cambios
en el modulo IAE. Hoy no esta automatizado.

---

**Fin del informe.**
