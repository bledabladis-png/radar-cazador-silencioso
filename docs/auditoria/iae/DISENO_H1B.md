# DISENO TECNICO DEL FIX H1-B

**Fecha:** 2026-09-27
**Estado:** BORRADOR - pendiente de aprobacion del auditor externo antes de implementar
**Referencia:** hallazgo H1-B del dictamen de auditoria externa (ver `AUDITORIA_EXTERNA_2026-09-27.md`)

---

## 1. Contexto

H1-B es el hallazgo bloqueante principal de la auditoria externa del IAE. El filtro actual del pipeline acepta filas de INFOTABLE con `PUTCALL=NULL` cuya security es, en realidad, una opcion CALL/PUT segun la Official List de la SEC. Dano medido: +53.228.375 sobre el NIPC actual.

**Instruccion del auditor:**

> Implementar el fix definitivo basado en identificacion por CUSIP + tipo de security temporalmente apropiado, usando la Official List como fuente primaria.

**Restricciones duras (del auditor):**
- Fuente primaria: Official List de la SEC (Option Indicator en posicion 10).
- Resolucion temporal: lista Q4 2025 para Q4 2025; lista Q1 2026 para Q1 2026; etc.
- Aplicacion en capa de identidad, no en `_filter_canonical`.
- Regex `TITLEOFCLASS` como defensa secundaria, no autoridad principal.
- Contadores de auditoria obligatorios.
- No congelar NIPC baseline hasta cerrar H1-B.

---

## 2. Causa raiz (verificada)

### 2.1. Datos

Los CUSIPs de opciones son **distintos** de los CUSIPs de la accion ordinaria subyacente:

| Ticker | CUSIP equity | CUSIP CALL | CUSIP PUT |
|---|---|---|---|
| AMZN | `023135106` | `023135906` | `023135956` |
| MU | `595112103` | `595112903` | `595112953` |

Los tres aparecen en la Official List con `issuer_description` explicito: `COM`, `CALL`, `PUT`.

### 2.2. El filtro actual

```python
# aggregation/delta_shares.py::_filter_canonical
mask_sh = df["SSHPRNAMTTYPE"].astype(str).str.strip() == "SH"
mask_null = df["PUTCALL"].isna()
df = df[mask_sh & mask_null]
```

El filtro opera sobre **INFOTABLE**, no sobre la Official List. Si el filer reporta una posicion CALL con `PUTCALL=NULL` (dato omitido), la fila pasa el filtro y se trata como si fuera del CUSIP equity.

### 2.3. Magnitud

| Trimestre | Filas CALL/OPTION con PUTCALL=NULL que pasan el filtro | CUSIPs del crosswalk afectados |
|---|---:|---:|
| 2025Q4 | 3.002 | 97 |
| 2026Q1 | 2.337 | 28 |

**Dano agregado sobre el NIPC actual:** +53.228.375.

---

## 3. Fuente primaria: Official List

### 3.1. Formato (verificado)

Fichero fixed-width de 80 caracteres. Layout confirmado en `data/sec_13f/official_list_13f/13flist_2026Q1.txt`:

```
Pos 01-09: CUSIP
Pos 10:    Option Indicator ('*' o espacio)
Pos 11-40: Issuer Name
Pos 41-67: Issuer Description (COM / CALL / PUT / SHS / ...)
Pos 68-70: STATUS ('*A*' add, '*D*' delete, blank no change)
Pos 71-79: Blank
Pos 80:    Misc / Unused
```

### 3.2. Evidencia empirica

```
023135106*AMAZON COM INC                COM                                    E
023135906 AMAZON COM INC                CALL                                   E
023135956 AMAZON COM INC                PUT                                    E
595112103*MICRON TECHNOLOGY INC         COM                                    E
595112903 MICRON TECHNOLOGY INC         CALL                                   E
```

Patron observado: `*` en posicion 10 → equity primaria; espacio → opcion CALL/PUT. El campo `issuer_description` tambien contiene explicito el tipo (`COM`, `CALL`, `PUT`).

### 3.3. Recurso ya existente

El parser `sec13f_list.py::parse_line` **ya extrae**:
- `cusip` (pos 1-9)
- `option_indicator` (pos 10)
- `issuer_description` (pos 41-67)

Y `resolve_eligibility(cusips, official_df)` devuelve `{status, option_indicator, eligible}` por CUSIP.

**El fix no requiere parser nuevo.** Solo requiere:
1. Exponer el `issuer_description` en `resolve_eligibility` (hoy no se propaga).
2. Una funcion `classify_instrument_type(description, option_indicator) -> str`.
3. Aplicar el filtro en `operational_universe.py::_apply_5_3`.
4. Exponer contadores de auditoria en `build_operational_universe`.

---

## 4. Diseno propuesto

### 4.1. Nueva funcion `classify_instrument_type`

**Ubicacion:** `src/institutional_accumulation/sec_13f/identity/sec13f_list.py`

**Firma:**
```python
def classify_instrument_type(description, option_indicator):
    """Clasifica el tipo de instrumento segun la Official List SEC 13(f).

    Args:
        description (str | None): issuer_description en pos 41-67.
        option_indicator (str | None): char en pos 10 ('*' o ' ').

    Returns:
        str: uno de {"EQUITY", "OPTION", "UNKNOWN"}.
    """
```

**Semantica (jerarquica):**

```python
DERIVATIVE_KEYWORDS = ("CALL", "PUT", "OPTION", "OPT", "WARRANT", "RIGHT", "FUTURE", "SWAP")
EQUITY_KEYWORDS = ("COM", "SHS", "SHARE", "STOCK", "NAMEN", "AKT", "ORD", "ORDINARY")

def classify_instrument_type(description, option_indicator):
    desc_upper = (description or "").upper().strip()
    # 1. Derivadas tienen prioridad (evita falsos negativos por substring)
    for kw in DERIVATIVE_KEYWORDS:
        if kw in desc_upper:
            return "OPTION"
    # 2. Equity por description
    for kw in EQUITY_KEYWORDS:
        if kw in desc_upper:
            return "EQUITY"
    # 3. Fallback por option_indicator (posicion 10)
    if option_indicator == "*":
        return "EQUITY"
    return "UNKNOWN"
```

**Nota sobre la prioridad DERIVATIVE > EQUITY:** evita clasificar erroneamente descripciones mixtas. Ejemplo: un campo description que contuviera "COM CALL" seria clasificado como OPTION (correcto).

**Nota sobre el fallback:** si la descripcion es ambigua (p.ej. `NAMEN AKT` de un emisor aleman), el indicador `*` en posicion 10 aporta la clasificacion.

### 4.2. Exponer `issuer_description` en `resolve_eligibility`

**Cambio minimo:** extender el dict de retorno (backward-compatible).

```python
# Antes:
resolved[c] = {
    "status": st,
    "option_indicator": opt,
    "eligible": st in ELIGIBLE_STATES,
}

# Despues (anadir instrument_type):
resolved[c] = {
    "status": st,
    "option_indicator": opt,
    "instrument_type": _resolve_instrument_type(sub),
    "eligible": st in ELIGIBLE_STATES and instrument_type == "EQUITY",
}
```

Donde `_resolve_instrument_type(sub)` itera las filas del CUSIP y aplica `classify_instrument_type` a cada una, devolviendo la clasificacion agregada (si todas coinciden → esa; si hay conflicto → "UNKNOWN").

### 4.3. Filtro en `operational_universe.py`

**Cambio minimo:** nuevo paso `_apply_5_3b` despues de `_apply_5_3`.

```python
def _apply_5_3b(df, official_df):
    """Filtro H1-B: excluir CUSIPs cuya clase de instrumento no sea EQUITY.

    Usa resolve_eligibility (Official List como fuente primaria).
    Contadores de auditoria expuestos via return_stats.
    """
    cusips = df["CUSIP"].astype(str).unique()
    elig = sl.resolve_eligibility(cusips, official_df)
    df = df.copy()
    df["_instrument_type"] = df["CUSIP"].astype(str).map(
        lambda c: elig.get(c, {}).get("instrument_type", "UNKNOWN")
    )
    mask = df["_instrument_type"] == "EQUITY"
    n_dropped = (~mask).sum()
    return df[mask].copy(), {
        "n_dropped_by_instrument_type": int(n_dropped),
        "dropped_by_type_breakdown": df[~mask]["_instrument_type"].value_counts().to_dict(),
    }
```

**Por que aqui y no en `_filter_canonical`:** el auditor lo exige. `_filter_canonical` opera sobre filas de INFOTABLE; el tipo de instrumento es una propiedad del CUSIP resuelta contra la Official List (capa de identidad). Aplicarlo aqui evita reconsultar la Official List por cada fila.

### 4.4. Contadores de auditoria

Extender `build_operational_universe` para devolver un dict `audit_stats` con:

```python
{
    "n_total_input": int,
    "n_after_5_1_sh_putcall_null": int,
    "n_after_5_3_official_list_eligible": int,
    "n_after_5_3b_instrument_type_equity": int,
    "n_after_5_4_security_type_resolved_equity": int,
    "n_after_5_5_canonical_verified": int,
    "n_dropped_by_instrument_type": int,
    "dropped_by_type_breakdown": {"OPTION": int, "UNKNOWN": int},
    "n_dropped_by_titleofclass": int,  # defensa secundaria
    "n_conflict_filer_vs_official_list": int,
    "unresolved_type": int,
}
```

Estos contadores son lo que el auditor pide explicitamente para auditar H1-B.

---

## 5. Defensa secundaria (regex TITLEOFCLASS)

Adicionalmente, en `_filter_canonical` (defensa en profundidad):

```python
import re
OPTION_HINTS = re.compile(r"CALL|PUT|OPTION|\bOPT\b", re.IGNORECASE)

def _filter_canonical(infotable_df):
    # ... filtro SH + PUTCALL NULL existente ...
    df = df[mask_sh & mask_null].copy()

    # H1-B defensa secundaria: descartar filas con TITLEOFCLASS sugerente
    # de opcion. No sustituye al filtro primario (Official List); solo
    # captura el residuo que la Official List no cubra.
    df["_toc_str"] = df["TITLEOFCLASS"].fillna("").astype(str)
    mask_clean = ~df["_toc_str"].str.contains(OPTION_HINTS, regex=True)
    n_dropped_by_toc = (~mask_clean).sum()
    df = df[mask_clean].drop(columns=["_toc_str"])
    stats["n_dropped_by_titleofclass"] = int(n_dropped_by_toc)
    # ... resto sin cambios ...
```

**Efecto medido** de la defensa secundaria sola: +53.228.375 de reduccion (medido empiricamente hoy). **Efecto esperado del fix completo:** mayor, porque el filtro primario captura CUSIPs de opciones con `TITLEOFCLASS='COM'` mal reportado por el filer.

---

## 6. Tests

### 6.1. Tests unitarios

`tests/test_sec_13f_list.py` (extender):
- `test_classify_instrument_type_common` — `"COM"` → `"EQUITY"`.
- `test_classify_instrument_type_call` — `"CALL"` → `"OPTION"`.
- `test_classify_instrument_type_put` — `"PUT"` → `"OPTION"`.
- `test_classify_instrument_type_mixed` — `"COM CALL"` → `"OPTION"` (prioridad derivadas).
- `test_classify_instrument_type_unknown` — `""` y `None` → `"UNKNOWN"`.
- `test_classify_instrument_type_option_indicator_fallback` — descripcion desconocida + `*` → `"EQUITY"`.
- `test_resolve_eligibility_returns_instrument_type` — extension backward-compatible.

### 6.2. Tests de integracion

`tests/test_operational_universe.py` (extender):
- `test_apply_5_3b_excluye_call` — CUSIP CALL con PUTCALL=NULL → excluido.
- `test_apply_5_3b_acepta_common` — CUSIP COM → incluido.
- `test_apply_5_3b_contadores` — contadores de auditoria poblados.

### 6.3. Test E2E sobre datos reales

`tests/test_h1b_e2e.py` (nuevo):
- Reconstruye la cadena contractual sobre los 3 trimestres reales.
- Verifica que `023135906` y `595112903` estan excluidos.
- Verifica que `023135106` y `595112103` estan incluidos.
- Verifica el conteo: < 553.319 filas del estado actual.
- Compara con un golden nuevo (a crear tras el fix).

### 6.4. Verificacion de no regresion

- E2E contractual: `py scripts/iae_contractual_nipc_e2e.py` — esperado FAIL contra §12.5 (porque el golden es de otro snapshot); rehacer reconciliacion.
- Reconciliacion B1: `py scripts/iae_reconciliation_b1.py` — esperado PASS (particion disjunta intacta).
- Unicidad shareClassFIGI: `py scripts/iae_identity_uniqueness_audit.py` — esperado PASS.

---

## 7. Plan de cierre de H1-B

Segun la instruccion del auditor:

1. **Gate 0** — verificar que las 3 Official List contienen `issuer_description` parseable.
2. **Implementar** el diseno de este documento.
3. **Rehacer** E2E + reconciliacion + cobertura + determinismo.
4. **Enviar al auditor** para revision antes de congelar.
5. **Congelar `current.json`** solo tras aprobacion.
6. **Actualizar** `TRASPASO_IAE.md` y `ESTADO_DECLARADO.md`.

**No congelar ningun NIPC baseline antes del paso 5.** Los numeros intermedios son `PRE_H1B_OBSERVATION` o `MITIGATION_RESULT`, nunca baseline contractual.

---

## 8. Rollback

Si el fix causa regresiones materiales (E2E con delta >1% inesperado, tests nuevos fallan, aislamiento arquitectonico roto):

1. `git revert <commit>` del fix.
2. Restaurar el estado previo de `operational_universe.py` y `sec13f_list.py`.
3. Documentar la regresion en `AUDITORIA_EXTERNA_2026-09-27.md`.
4. Reabrir H1-B con el auditor antes de reintentar.

El fix es **backward-compatible** en `resolve_eligibility` (extension, no reemplazo). Un rollback no rompe consumidores existentes.

---

## 9. Anexos

### 9.1. Ficheros afectados (estimacion)

| Fichero | Cambio |
|---|---|
| `src/institutional_accumulation/sec_13f/identity/sec13f_list.py` | +30 LOC (nueva funcion + extension resolve_eligibility) |
| `src/institutional_accumulation/operational_universe.py` | +50 LOC (nuevo _apply_5_3b + audit_stats) |
| `src/institutional_accumulation/aggregation/delta_shares.py` | +8 LOC (defensa secundaria regex) |
| `tests/test_sec_13f_list.py` | +60 LOC (7 tests) |
| `tests/test_operational_universe.py` | +40 LOC (3 tests) |
| `tests/test_h1b_e2e.py` | +80 LOC (nuevo) |
| **Total** | ~270 LOC |

### 9.2. Impacto esperado en el NIPC

- **NIPC actual (HEAD):** -4.317.678.307
- **Tras fix primario (Official List):** por determinar
- **Tras fix primario + secundario (regex):** por determinar
- **Diferencia estimada:** > 53.228.375 (el filtro primario captura mas casos)

El numero final se medira en el E2E tras implementar. **No se congela hasta aprobacion del auditor.**

### 9.3. Por que no se aplica solo el regex

El auditor lo rechazo explicito:

> Aprobaria la regex como mitigacion intermedia, pero no cerraria H1-B con ella.

Razon: hay filas con `TITLEOFCLASS='COM'` reportadas erroneamente por el filer para CUSIPs que SEC confirma como opciones. La regex no las captura. Solo el filtro por Official List (a nivel CUSIP) las detecta.

---

**Fin del diseno.**
Pendiente: aprobacion del auditor antes de implementar.
