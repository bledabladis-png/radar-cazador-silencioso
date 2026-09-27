# DISENO TECNICO DEL FIX H1-B

**Fecha:** 2026-09-27
**Estado:** DISENO REVISADO - pendiente de aprobacion del auditor externo
**Referencia:** hallazgo H1-B de `AUDITORIA_EXTERNA_2026-09-27.md`
**Revision:** v2, incorpora dictamen del auditor sobre Gate 0
(ver `CONSULTA_H1B.md` secciones 7 y 8)

---

## 1. Contexto

H1-B es el hallazgo bloqueante principal del dictamen externo. El filtro
actual acepta filas INFOTABLE con `PUTCALL=NULL` cuya security es, en
realidad, una opcion CALL/PUT. La fila pasa el filtro y se trata como
equity, contaminando el NIPC.

**Instruccion original del auditor:** usar la Official List SEC como
fuente primaria de tipo de instrumento, con resolucion temporal por
periodo.

**Correccion del auditor tras Gate 0 (2026-09-27):** la Official List
Q4 2025 existe en formato TXT, es oficial, es temporalmente aplicable,
y contiene opciones. Pero NO contiene los option CUSIPs problematicos
observados en filings Q4 (AMZN 023135906, MU 595112903). Por tanto
la Official List Q4 no es exhaustiva para clasificar todo CUSIP del
universo INFOTABLE. El diseno debe adaptarse a esa asimetria.

**Restricciones duras vigentes (del auditor):**
- Fuente primaria: Official List SEC, temporalmente resuelta por periodo.
- Aplicacion en capa de identidad, no en `_filter_canonical`.
- Regex `TITLEOFCLASS` como defensa secundaria, nunca autoridad.
- Contadores de auditoria obligatorios.
- Regla normativa: **no encontrado != equity**. CUSIP sin tipo
  certificable para el periodo -> UNRESOLVED -> excluir del NIPC.
- Aplicacion asimetrica declarada: Q1+ clasificacion externa;
  Q4 y anteriores tratamiento conservador.
- No congelar NIPC baseline hasta cierre de H1-B.

---

## 2. Causa raiz (verificada parcialmente)

### 2.1. CUSIPs de opciones son distintos de los CUSIPs equity

| Ticker | CUSIP equity | CUSIP CALL | CUSIP PUT |
|---|---|---|---|
| AMZN | `023135106` | `023135906` | `023135956` |
| MU | `595112103` | `595112903` | `595112953` |

Verificado empíricamente contra Official List Q1 y Q2 2026 (ambos CALL
y PUT presentes con `issuer_description` explicito). Ver
`CONSULTA_H1B.md` seccion 2.5.

### 2.2. El filtro actual

    # aggregation/delta_shares.py::_filter_canonical
    mask_sh = df["SSHPRNAMTTYPE"].astype(str).str.strip() == "SH"
    mask_null = df["PUTCALL"].isna()
    df = df[mask_sh & mask_null]

Opera sobre INFOTABLE. Si el filer reporta una posicion CALL con
`PUTCALL=NULL`, la fila pasa el filtro y se trata como equity.

### 2.3. Magnitud (declarada por el auditor, no reproducida localmente)

| Trimestre | Filas CALL con PUTCALL=NULL | CUSIPs afectados |
|---|---:|---:|
| 2025Q4 | 3.002 | 97 |
| 2026Q1 | 2.337 | 28 |

**Dano agregado declarado:** +53.228.375 sobre el NIPC actual.

**Estado de reproduccion.** Estas cifras provienen del dictamen del
auditor. NO son reproducibles con la Official List Q4 2025 como fuente
(esa lista no contiene los CUSIPs problematicos). El auditor ha pedido
explicitamente que se documente su procedencia exacta. Queda como
**deuda de trazabilidad** hasta que el auditor indique la metodologia
concreta empleada.

---
## 3. Fuente primaria: Official List SEC

### 3.1. Formato (verificado empíricamente)

Fichero fixed-width de 80 caracteres.

    Pos 01-09: CUSIP
    Pos 10:    Option Indicator ('*' o espacio)
    Pos 11-40: Issuer Name
    Pos 41-67: Issuer Description (COM / CALL / PUT / SHS / ...)
    Pos 68-70: STATUS ('*A*' add, '*D*' delete, blank no change)
    Pos 71-79: Blank
    Pos 80:    Campo no interpretado (homogéneo 'E' en Q1/Q2, heterogéneo en Q4)

**Nota sobre posicion 80.** En Q1 y Q2 2026 toma siempre el valor 'E'.
En Q4 2025 toma 8 valores distintos (N/E/A/P/U/S/%). No se ha
interpretado su semántica. Se declara como campo no utilizado.

### 3.2. Disponibilidad por trimestre (verificado)

| Trimestre | Lineas | CALL | PUT | Cobertura |
|---|---:|---:|---:|---|
| 2025Q4 | 12.282 | 11 | 7 | **NO exhaustiva** para option CUSIPs |
| 2026Q1 | 24.641 | 6.029 | 6.021 | Exhaustiva |
| 2026Q2 | 25.333 | 6.128 | 6.116 | Exhaustiva |

Los CUSIPs `023135906` (AMZN CALL) y `595112903` (MU CALL) NO aparecen
en Q4 2025 pero SI en Q1 y Q2 2026. Es la evidencia central del
dictamen: la Official List Q4 existe pero no contiene todos los option
CUSIPs observados en filings Q4.

### 3.3. La Official List como fuente de tipo (no de universo)

La Official List es autoridad de **tipo** para CUSIPs presentes en ella.
NO es autoridad de **universo completo**: puede no contener CUSIPs que
los filers reportan (evidencia Q4 2025). Este matiz es lo que cambia el
diseno respecto al borrador v1.

### 3.4. Recurso ya existente en el codigo

`sec13f_list.py::parse_line` ya extrae:
- `cusip` (pos 1-9)
- `option_indicator` (pos 10)
- `issuer_description` (pos 41-67)

`resolve_eligibility(cusips, official_df)` devuelve un dict por CUSIP.
El fix extiende ese dict (no lo redefine).

---

## 4. Contrato conceptual (fijado por el auditor)

    CUSIP observado en INFOTABLE
           |
           v
    Official List del periodo
           |
           +-- encontrado + descripcion de opcion   -> OPTION
           +-- encontrado + descripcion de equity   -> EQUITY
           +-- no encontrado / ambiguo / conflicto  -> UNRESOLVED

**Regla normativa:** `UNRESOLVED` NO se trata como equity. Se excluye
del NIPC por aplicacion del principio "no imputar".

**Tabla de tratamiento:**

| Estado | Tratamiento NIPC |
|---|---|
| EQUITY confirmado | Incluir |
| OPTION confirmado | Excluir |
| UNRESOLVED (no en lista) | Excluir |
| UNRESOLVED (conflicto entre fuentes) | Excluir |

**Aplicacion asimetrica por periodo:**

| Periodo | Tratamiento |
|---|---|
| Q1 2026 en adelante | Clasificacion por Official List (exhaustiva) |
| Q4 2025 y anteriores | Tratamiento conservador: sin clasificacion clara -> UNRESOLVED |

Esta asimetria se declara explicitamente como **limitacion del diseno**,
no se oculta. El objetivo no es "cobertura completa" del par Q4->Q1,
sino **no contaminar el NIPC con imputaciones**.

---
## 5. Diseno propuesto

### 5.1. Nueva funcion `classify_instrument_type`

**Ubicacion:** `src/institutional_accumulation/sec_13f/identity/sec13f_list.py`

    def classify_instrument_type(description):
        """Clasifica el tipo de instrumento segun issuer_description.

        Args:
            description (str | None): issuer_description en pos 41-67.

        Returns:
            str: "EQUITY" | "OPTION" | "UNKNOWN".
                 UNKNOWN -> tratar como UNRESOLVED por el caller.
        """

**Semantica (jerarquica, con word-boundary):**

    import re
    DERIVATIVE_RE = re.compile(r"\b(?:CALL|PUT|OPTION|OPT)\b", re.IGNORECASE)
    EQUITY_RE = re.compile(
        r"\b(?:COM|SHS|SHARE|STOCK|NAMEN|AKT|ORD|ORDINARY)\b",
        re.IGNORECASE,
    )

    def classify_instrument_type(description):
        desc = (description or "").strip()
        if not desc:
            return "UNKNOWN"
        if DERIVATIVE_RE.search(desc):
            return "OPTION"
        if EQUITY_RE.search(desc):
            return "EQUITY"
        return "UNKNOWN"

**Cambios respecto al borrador v1:**

- Eliminado el parametro `option_indicator`. No se usa como fallback.
  Razon: la semantica de `*` vs espacio no esta completamente verificada
  (en Q1/Q2 `*` coincide con CALL por conteo agregado; no se puede usar
  como autoridad). Se elimina hasta que se resuelva su semantica.
- Eliminados `WARRANT`, `RIGHT`, `FUTURE`, `SWAP` del patron. Fuera de
  alcance de H1-B (que es CALL/PUT/OPTION). Si aparecen, finding
  separado.
- Word-boundary `\b` en ambos patrones. Evita falsos positivos como
  `CALLON PETROLEUM` o `COMPUTERSHARE`.
- Retorno `"UNKNOWN"` en lugar de `"EQUITY"` como fallback. `UNKNOWN`
  se traduce a UNRESOLVED en el caller, que excluye.

### 5.2. Extension de `resolve_eligibility` (no redefinicion)

**Regla:** los campos existentes (`status`, `option_indicator`,
`eligible`) NO se tocan. Se añaden campos nuevos.

    # Antes (intacto):
    resolved[c] = {
        "status": st,
        "option_indicator": opt,
        "eligible": st in ELIGIBLE_STATES,
    }

    # Despues (extension aditiva):
    resolved[c] = {
        "status": st,                            # intacto
        "option_indicator": opt,                 # intacto
        "eligible": st in ELIGIBLE_STATES,       # intacto
        "instrument_type": _aggregate_instrument_type(sub),  # nuevo
        "eligible_for_nipc": (
            st in ELIGIBLE_STATES
            and _aggregate_instrument_type(sub) == "EQUITY"
        ),                                                     # nuevo
    }

Donde `_aggregate_instrument_type(sub)` itera las filas del CUSIP
en la Official List y agrega:

- Todas EQUITY -> EQUITY
- Todas OPTION -> OPTION
- Cualquier otra combinacion o sin filas -> UNKNOWN

**Por que no redefinir `eligible`:** el dictamen del auditor exige
separar `security_type` de `eligible_as_of_period`. Consumidores
existentes de `eligible` no deben romperse. El nuevo campo
`eligible_for_nipc` es opt-in.

### 5.3. Filtro en `operational_universe.py`

Nuevo paso `_apply_5_3b` DESPUES de `_apply_5_3`:

    def _apply_5_3b(df, official_df):
        """Filtro H1-B: excluir CUSIPs cuyo instrument_type no sea EQUITY.

        UNRESOLVED (incluye UNKNOWN) -> excluido. Regla: no imputar.
        """
        cusips = df["CUSIP"].astype(str).unique()
        elig = sl.resolve_eligibility(cusips, official_df)
        df = df.copy()
        df["_instrument_type"] = df["CUSIP"].astype(str).map(
            lambda c: elig.get(c, {}).get("instrument_type", "UNKNOWN")
        )
        mask_keep = df["_instrument_type"] == "EQUITY"
        stats = {
            "n_equity": int(mask_keep.sum()),
            "n_option": int((df["_instrument_type"] == "OPTION").sum()),
            "n_unresolved": int((df["_instrument_type"] == "UNKNOWN").sum()),
        }
        return df[mask_keep].copy(), stats

**Por que aqui y no en `_filter_canonical`:** el tipo es propiedad del
CUSIP resuelta contra la Official List (capa de identidad). Se calcula
una vez por CUSIP, no por fila.

### 5.4. Contadores de auditoria

Extender `build_operational_universe` para devolver `audit_stats`:

    {
        "n_total_input": int,
        "n_after_5_1_sh_putcall_null": int,
        "n_after_5_3_official_list_eligible": int,
        "n_after_5_3b_instrument_type_equity": int,
        "n_after_5_4_security_type_resolved_equity": int,
        "n_after_5_5_canonical_verified": int,
        "n_dropped_by_instrument_type": int,
        "n_equity": int,
        "n_option": int,
        "n_unresolved": int,
        "n_dropped_by_titleofclass": int,
    }

---
## 6. Defensa secundaria (regex TITLEOFCLASS)

En `_filter_canonical`, defensa en profundidad:

    import re
    OPTION_HINTS = re.compile(r"\b(?:CALL|PUT|OPTION|OPT)\b", re.IGNORECASE)

    def _filter_canonical(infotable_df):
        # ... filtro SH + PUTCALL NULL existente ...
        df = df[mask_sh & mask_null].copy()
        df["_toc_str"] = df["TITLEOFCLASS"].fillna("").astype(str)
        mask_clean = ~df["_toc_str"].str.contains(OPTION_HINTS, regex=True)
        n_dropped_by_toc = (~mask_clean).sum()
        df = df[mask_clean].drop(columns=["_toc_str"])
        stats["n_dropped_by_titleofclass"] = int(n_dropped_by_toc)
        # ... resto sin cambios ...

**Word-boundary obligatorio.** Sin `\b`, `CALLON PETROLEUM` (emisor
real) caeria en falso positivo.

**Efecto:** captura el residuo de filas con `TITLEOFCLASS` explicito de
opcion que el filtro primario no detecta (CUSIP no clasificable, o
conflicto filer vs Official List). No sustituye al filtro primario.

---

## 7. Tests

### 7.1. Unitarios (`tests/test_sec_13f_list.py`)

- `test_classify_instrument_type_common`: `"COM"` -> EQUITY.
- `test_classify_instrument_type_call`: `"CALL"` -> OPTION.
- `test_classify_instrument_type_put`: `"PUT"` -> OPTION.
- `test_classify_instrument_type_empty`: `""`/`None` -> UNKNOWN.
- `test_classify_instrument_type_word_boundary`: `"CALLON PETROLEUM"`
  -> UNKNOWN (no OPTION). Evita falso positivo por substring.
- `test_classify_instrument_type_mixed`: `"COM CALL"` -> OPTION.
- `test_resolve_eligibility_extension_backward_compat`: campos
  existentes (`status`, `option_indicator`, `eligible`) intactos en
  contenido y tipo.
- `test_resolve_eligibility_adds_instrument_type`: campo nuevo presente.
- `test_resolve_eligibility_eligible_for_nipc_unresolved_false`:
  CUSIP ausente -> UNKNOWN -> eligible_for_nipc=False.

### 7.2. Integracion (`tests/test_operational_universe.py`)

- `test_apply_5_3b_excluye_option`: CUSIP CALL -> excluido.
- `test_apply_5_3b_acepta_equity`: CUSIP COM -> incluido.
- `test_apply_5_3b_excluye_unresolved`: CUSIP no en lista -> excluido.
- `test_apply_5_3b_contadores`: n_equity/n_option/n_unresolved poblados.

### 7.3. E2E (`tests/test_h1b_e2e.py`, nuevo)

- Reconstruye cadena sobre Q4 2025 + Q1 2026 reales.
- `023135906` y `595112903` excluidos de Q1 (UNRESOLVED o OPTION).
- `023135106` y `595112103` incluidos en Q4 y Q1 (EQUITY).
- Conteo filas resultante < 553.319 (delta por exclusiones).
- Determinista: dos ejecuciones -> mismo output.

### 7.4. No regresion

- `py scripts/iae_contractual_nipc_e2e.py` -> FAIL esperado contra
  golden §12.5 (snapshot distinto). Se rehace golden tras fix.
- `py scripts/iae_reconciliation_b1.py` -> PASS esperado.
- `py scripts/iae_identity_uniqueness_audit.py` -> PASS esperado.

---
## 8. Plan de cierre (alineado con dictamen del auditor)

1. **Aprobacion del auditor** de este documento revisado.
2. **Implementacion** del diseno de §5 (chunk de codigo, tests, no push).
3. **Rehacer** E2E + reconciliacion B1 + unicidad + cobertura.
4. **Medir** el nuevo NIPC y documentarlo como `POST_H1B_OBSERVATION`.
5. **Enviar al auditor** los resultados + nuevo golden propuesto.
6. **Congelar `current.json`** solo tras aprobacion del paso 5.
7. **Actualizar** `TRASPASO_IAE.md`, `ESTADO_DECLARADO.md`, y este doc
   (mover a "CERRADO").

**Regla dura:** el valor actual `-4.317.678.307` sigue etiquetado como
`PRE_H1B_OBSERVATION`. No se convierte en baseline. No se congela nada
antes del paso 6.

---

## 9. Rollback

Si el fix causa regresiones materiales:

1. `git revert <commit>` del fix.
2. Documentar la regresion en `AUDITORIA_EXTERNA_2026-09-27.md`.
3. Reabrir H1-B con el auditor antes de reintentar.

El fix es aditivo: no rompe consumidores existentes de
`resolve_eligibility`. Rollback limpio.

---

## 10. Anexos

### 10.1. Ficheros afectados (estimacion v2)

| Fichero | Cambio |
|---|---|
| `src/institutional_accumulation/sec_13f/identity/sec13f_list.py` | +40 LOC (nueva funcion + extension) |
| `src/institutional_accumulation/operational_universe.py` | +55 LOC (_apply_5_3b + audit_stats) |
| `src/institutional_accumulation/aggregation/delta_shares.py` | +10 LOC (regex con `\b`) |
| `tests/test_sec_13f_list.py` | +90 LOC (9 tests) |
| `tests/test_operational_universe.py` | +50 LOC (4 tests) |
| `tests/test_h1b_e2e.py` | +90 LOC (nuevo) |
| **Total** | ~335 LOC |

### 10.2. Impacto esperado en el NIPC

- Actual (`PRE_H1B_OBSERVATION`): -4.317.678.307
- Tras fix: por determinar. Se mide en el paso 3 del plan.
- **No se congela hasta aprobacion del auditor.**

### 10.3. Por que no solo regex

El auditor lo rechazo explicito: la regex no captura CUSIPs con
`TITLEOFCLASS='COM'` mal reportado por el filer. Solo el filtro por
CUSIP (contra Official List) los detecta, y aun así deja UNRESOLVED
los que la lista no cubre.

---

## 11. Rectificacion registrada

El borrador v1 de este documento afirmo:

> "Los tres aparecen en la Official List con issuer_description
> explicito: COM, CALL, PUT."

Y, para Q4 2025, implicito: la lista contiene los CUSIPs de opciones.
**Incorrecto.** Los CUSIPs `023135906` y `595112903` NO aparecen en
la Official List Q4 2025. Solo en Q1 y Q2 2026.

Causa raiz del error: se asumio simetria entre trimestres sin
verificar la fuente Q4. El Gate 0 y el dictamen del auditor
corrigieron esa asuncion.

Seccion 3.2 refleja la disponibilidad real por trimestre.
Seccion 4 fija el contrato de 3 estados con UNRESOLVED explicito
para CUSIPs sin clasificacion en el periodo.

---

**Fin del diseno v2.**
Pendiente: aprobacion del auditor antes de implementar.