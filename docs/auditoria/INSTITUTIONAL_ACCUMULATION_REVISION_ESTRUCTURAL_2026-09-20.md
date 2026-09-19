# INFORME DE REVISIÓN ESTRUCTURAL DEL MÓDULO INSTITUTIONAL ACCUMULATION EVIDENCE (IAE)



**Fecha:** 2026-09-20

**Autor:** Ingeniero Supervisor (revisión interna)

**Destinatario:** Auditor externo

**Alcance:** `src/institutional_accumulation/` — 22 ficheros `.py` (2.456 LOC no-vacías), 17 ficheros de test, 53 documentos en `docs/auditoria/`

**HEAD al cierre:** `77d4785` (ahead 75 de `origin/main`)

**Estado:** BORRADOR INTERNO. Pendiente de validación por el responsable antes de envío externo.



\---



## 1. Resumen ejecutivo



Se ha realizado una revisión estructural completa del módulo IAE con **95 observaciones** clasificadas: **3 ALTA**, **20 MEDIA**, **23 BAJA**, **~49 INFORMATIVO**. Dos hipótesis iniciales se cerraron como NO BUG tras verificación empírica (P14, P29).



**Candidatos críticos que requieren dictamen del auditor (no fix mecánico):**



1\. **P38** — Ambigüedad semántica en el denominador de `paired_weighted_share_coverage` (unión vs intersección). Afecta directamente al `THRESHOLD_2` que sigue UNDEFINED. **Bloquea la fijación de thresholds**.

2\. **P60** — `_normalize_canonical` infiere prefijo `equity:` sin evidencia de formato. Posible violación del contrato C1-revisada ("no inferir").

3\. **P61** — `etf_holdings.csv` cargado como crosswalk sin vigencia temporal. Raíz estructural de la circularidad documentada por el propio auditor en F2.1-bis.

4\. **P70** — Asimetría en `classify_strategy` al decidir `HR_PLUS_RESTATEMENT` vs `HR_PLUS_NEW_HOLDINGS` (mira `iloc[1]` para un caso, `iloc[-1]` para otro).

5\. **P31** — `_sum_delta` promete devolver `None` en el docstring pero devuelve siempre `0.0`. No distingue "no hay datos" de "NIPC neto = 0".



**Candidatos de fix mecánico (una vez autorizado):**



1\. **P18 / P26** — Path absoluto `D:\\Macro_Sectorial` hardcoded en `manifest.py`. Rompe portabilidad (CI, clones, cualquier máquina no-Windows-D).

2\. **P14-BIS** — `fillna(0.0)` en `delta_shares` es correcto hoy pero no distingue "ausencia de fila" de "fila con NaN". Fix preventivo recomendado.

3\. **P32** — `_filter_canonical` descarta filas con `SSHPRNAMT` no parseable sin contador. Pérdida silenciosa de trazabilidad.

4\. **P11 / P1 / P2 / P12** — Desfases documentales en `__init__.py` (estado FA-1.1) y referencias a spec v1.2.



**Estado de contratos:** El módulo respeta las prohibiciones duras verificables (`ticker:` no usado, `datetime.now()` solo en `generated_at_utc`, sin IO en `aggregation/`, sin IDs hardcodeados en lógica). El aislamiento (`src/` y `scripts/` no consumen el módulo) es coherente con la regla local-first IAE.



\---



## 2. Metodología



Cada hallazgo se ha clasificado en cuatro niveles:



\- **ALTA:** Rompe portabilidad, produce resultados incorrectos, o viola contrato declarado de forma verificable.

\- **MEDIA:** Riesgo de inconsistencia futura, pérdida de trazabilidad, o ambigüedad semántica que requiere dictamen.

\- **BAJA:** Defecto cosmético, inconsistencia de estilo, o deuda declarada.

\- **INFORMATIVO:** Observación sin defecto, documentada para contexto del auditor.



**Regla aplicada:** cada afirmación se ancla a línea de código verificada. Las hipótesis no confirmadas se marcan explícitamente como tales. Los hallazgos cerrados durante la revisión se documentan en el Anexo C.



\---



## 3. Hallazgos ALTA



### P18 / P26 — Path absoluto `D:\\Macro_Sectorial` en `manifest.py`



**Evidencia:**

```python

# src/institutional_accumulation/sec_13f/manifest.py:21

DEFAULT_PROJECT_ROOT = Path(r"D:\\Macro_Sectorial")

```

`build_manifest(..., project_root=DEFAULT_PROJECT_ROOT)` es el default de la firma (línea 59). `ingest_13f(..., project_root=DEFAULT_PROJECT_ROOT)` propaga el acoplamiento (`ingest.py:38`).



**Impacto:**

\- Cualquier llamada desde CI, WSL, Mac, Linux o clon en ruta distinta produce rutas absolutas inválidas en el campo `file` de cada entrada del manifest (`_relpath_or_abs` hace fallback a absoluta si `relative_to` falla — línea 31).

\- Rompe portabilidad del módulo, declarado en el prompt como "local-first refactor" y "verificar en CI".

\- El README del módulo no declara esta restricción.



**Propuesta:**

```python

DEFAULT_PROJECT_ROOT = Path(os.environ.get(

&#x20;   "IAE_PROJECT_ROOT",

&#x20;   Path(__file__).resolve().parents[4]

))

```

O `None` explícito con resolución obligatoria por caller.



**Verificación:** `grep -r "D:\\\\\\\\Macro_Sectorial" src/` → 2 matches (manifest.py:21, schema.py:7 en docstring).



\---



## 4. Hallazgos MEDIA



### P38 — Denominador de `paired_weighted_share_coverage` (candidato estrella)



**Evidencia cruzada:**

```python

# src/institutional_accumulation/aggregation/nipc.py (docstring, líneas ~137-142)

paired_weighted_share_coverage

&#x20; = sum(SSHPRNAMT de S con mapping en ambos)

&#x20;   / sum(SSHPRNAMT de S con posicion en ambos)

```



```python

# src/institutional_accumulation/aggregation/nipc.py (código, líneas ~218-228)

union_sec = sec_c | sec_p

both_mapped_sec = mapped_c \& mapped_p

...

denom = sum(by_sec.get(k, 0.0) for k in union_sec)

numer = sum(by_sec.get(k, 0.0) for k in both_mapped_sec)

```



**El docstring dice "en ambos". El código usa la unión (`Q4 ∪ Q1`), no la intersección (`Q4 ∩ Q1`).** Una de las dos definiciones es incorrecta.



**Impacto:**

\- Si debe ser intersección, todas las cifras publicadas en el baseline (commit `e9fd830`) y en el TOP 2000 (`32baf9d`) están mal calculadas.

\- `THRESHOLD_2` depende directamente de este valor.

\- Es exactamente el tipo de ambigüedad que el micro-gate F2.2-v2 pretendía eliminar con "PAIRED_WEIGHTED_SHARE_COVERAGE: convencion Q4/Q1 inequivoca (max)". La convención "max por security" está aplicada en `by_sec`, pero el conjunto de claves del denominador no está fijado por el cambio obligatorio #5 del dictamen F2.2.



**Propuesta:** dictamen del auditor antes de cualquier fix. Documento afectado: `NIPC_COVERAGE_POLICY_V12_PROPUESTA.md` (borrador).



\---



### P60 — `_normalize_canonical` infiere prefijo `equity:` sin evidencia



**Evidencia:**

```python

# src/institutional_accumulation/sec_13f/identity/security_identity.py:60-81

def _normalize_canonical(value):

&#x20;   ...

&#x20;   if s.startswith("equity:") or s.startswith("figi:"):

&#x20;       return s

&#x20;   return f"equity:{s}"

```



**Impacto:**

\- Un `canonical_security = "BBG001S69V32"` (FIGI desnudo, 12 chars alfanuméricos) se convierte en `"equity:BBG001S69V32"` en vez de `"figi:BBG001S69V32"`.

\- El prefijo se **asigna** sin verificar el formato del valor. Viola nominalmente el contrato C1-revisada: *"NO inferir equivalence sin entry explicita"* y *"NO presentar `cusip:<CUSIP>` como identidad canonica estable sin evidencia"* — la misma lógica aplica a la dirección contraria.

\- `crosswalk_internal` (líneas 285-295) devuelve **tickers Yahoo** (no FIGI), que se prefijan como `equity:<ticker>`. Es correcto como convención, pero el código no distingue "ticker" de "FIGI" al prefijar.



**Propuesta:** dictamen. Opciones:

1\. Rechazar el prefijo `equity:` para valores con formato FIGI (regex `^BBG[A-Z0-9]{9}$`).

2\. Declarar formalmente que `canonical_security` es siempre `equity:<ticker>` o `figi:<FIGI>` y validar el formato.



\---



### P61 — `etf_holdings.csv` sin vigencia temporal en el crosswalk



**Evidencia:**

```python

# src/institutional_accumulation/sec_13f/identity/security_identity.py:198-217

def load_crosswalk_internal(exceptions_path=..., etf_holdings_path=...):

&#x20;   ...

&#x20;   if p2.exists():

&#x20;       h = pd.read_csv(p2, dtype=str)

&#x20;       for _, r in h.iterrows():

&#x20;           rows.append({

&#x20;               "CUSIP": str(r["identifier"]).strip(),

&#x20;               "ticker": str(r["ticker"]).strip(),

&#x20;               "valid_from": None,   # <-- siempre activo

&#x20;               "valid_to": None,

&#x20;               "source": "etf_holdings",

&#x20;           })

```



**Impacto:**

\- Los CUSIPs que provienen de `etf_holdings.csv` quedan con vigencia `(-∞, +∞)`. Si un ETF holding tiene un ticker obsoleto (post corporate action), el resolver lo devuelve como `CANONICAL` sin verificación.

\- **Es la raíz estructural documentada por el auditor en F2.1-bis** ("el 90.91% (220/242) que la policy v1.0 L115 cita como `mapping_coverage` sale exactamente de `etf_holdings.csv`, parte del crosswalk evaluado"). No es hallazgo nuevo, pero debe figurar en el informe como limitación cuantificada del resolver.

\- En el baseline Fase A, `coverage_previous == coverage_current == 1.0` sobre `OPERATIONAL_EQUITY` por construcción.



**Propuesta:** no es fix mecánico. Añadir sección explícita al informe de cobertura. Candidato al cambio obligatorio #8 del dictamen F2.2 ("trazabilidad explícita del cambio semántico") — declarar que `crosswalk_internal` **no tiene cobertura temporal verificable**.



\---



### P70 — Asimetría en `classify_strategy` (nuevo)



**Evidencia:**

```python

# src/institutional_accumulation/sec_13f/identity/amendments.py:181-198

if base == STRATEGY_HR_PLUS_RESTATEMENT:

&#x20;   if "AMENDMENTTYPE" in filings_group.columns:

&#x20;       at = _norm_str(filings_group.iloc[1].get("AMENDMENTTYPE"))   # <-- iloc[1] fijo

&#x20;       if at == "NEW HOLDINGS":

&#x20;           return STRATEGY_HR_PLUS_NEW_HOLDINGS

&#x20;   return STRATEGY_HR_PLUS_RESTATEMENT



if base == STRATEGY_HR_CHAIN_RESTATEMENT:

&#x20;   if "AMENDMENTTYPE" in filings_group.columns and len(filings_group) >= 2:

&#x20;       last = _norm_str(filings_group.iloc[-1].get("AMENDMENTTYPE"))  # <-- iloc[-1]

&#x20;       if last == "NEW HOLDINGS":

&#x20;           return STRATEGY_HR_COMPOSITE

&#x20;   return STRATEGY_HR_CHAIN_RESTATEMENT

```



Un caso mira **el segundo filing** (`iloc[1]`) y el otro **el último** (`iloc[-1]`). La lógica de negocio 13F suele ser: la cadena de amendments aplica en orden (`AMENDMENTNO` ascendente) y **el resultado final** determina la composición. Mirar `iloc[1]` para `HR_PLUS_RESTATEMENT` (que solo tiene 2 filings) es equivalente a mirar el último porque solo hay 2. Pero el código es frágil: si en el futuro se admite `HR + HR/A + HR/A` con el primero NEW HOLDINGS, `iloc[1]` decide prematuramente.



**Propuesta:** dictamen del auditor sobre la semántica correcta. Unificar la lógica a `iloc[-1]` para ambos casos, o documentar la asimetría explícitamente.



\---



### P31 — `_sum_delta` docstring miente sobre el retorno



**Evidencia:**

```python

# src/institutional_accumulation/aggregation/nipc.py:64-72

def _sum_delta(delta_df, mask=None):

&#x20;   """Suma delta_shares ignorando NaN. None si no hay filas validas."""

&#x20;   if delta_df is None or delta_df.empty:

&#x20;       return 0.0

&#x20;   ...

&#x20;   return float(s.sum()) if len(s) > 0 else 0.0

```



Devuelve **siempre** `0.0`, nunca `None`. Contradice el principio "No confundir VALID con UNAVAILABLE" del prompt.



**Impacto:**

\- `nipc_total = 0` puede significar "no hay datos" o "el neto es exactamente 0". Aunque `_derive_status` distingue por `n_total == 0`, el campo `nipc_total` aislado es ambiguo para un consumidor externo (informe, auditoría a posteriori, futuros readers).



**Propuesta:** mantener `0.0` (compatibilidad con el motor actual) y añadir `nipc_total_available: bool` o `n_observations: int` al dict. Requiere dictamen.



\---



### P14-BIS — `fillna(0.0)` sin distinguir origen de NaN



**Evidencia:**

```python

# src/institutional_accumulation/aggregation/delta_shares.py:252-259

merged["sshprnamt_current"] = (

&#x20;   pd.to_numeric(merged["sshprnamt_current"], errors="coerce")

&#x20;   .fillna(0.0).astype("Float64")

)

merged["sshprnamt_previous"] = (

&#x20;   pd.to_numeric(merged["sshprnamt_previous"], errors="coerce")

&#x20;   .fillna(0.0).astype("Float64")

)

```



**Análisis:** Es correcto hoy (los NaN provienen solo del FULL OUTER JOIN, no de datos corruptos, porque `_filter_canonical` ya hace `dropna(subset=["SSHPRNAMT_f"])`). **No imputa datos ficticios: la ausencia de fila es equivalente semánticamente a "cantidad 0" en el periodo.**



**Riesgo:** si en el futuro se relaja `_filter_canonical`, llega un parquet corrupto, o se reutiliza la función con otra fuente, el fillna convertiría un dato corrupto en `0.0` → EXIT falso → **contaminación de `nipc_total` sin traza**.



**Propuesta:** condicionar el fillna a `_merge` en lugar de al valor NaN:

```python

mask_new = merged["_merge"] == "left_only"

mask_exit = merged["_merge"] == "right_only"

merged.loc[mask_new, "sshprnamt_previous"] = 0.0

merged.loc[mask_exit, "sshprnamt_current"] = 0.0

bad = (merged["_merge"] == "both") \& (

&#x20;   merged["sshprnamt_current"].isna() | merged["sshprnamt_previous"].isna()

)

if bad.any():

&#x20;   raise ValueError(f"corrupcion: {int(bad.sum())} filas BOTH con NaN")

```



\---



### P32 — Dropna silencioso en `_filter_canonical`



**Evidencia:**

```python

# src/institutional_accumulation/aggregation/delta_shares.py:86-87

df["SSHPRNAMT_f"] = pd.to_numeric(df["SSHPRNAMT"], errors="coerce")

df = df.dropna(subset=["SSHPRNAMT_f"])

```



Sobre 3.3M filas de `INFOTABLE` Q1 2026, `errors="coerce"` puede convertir cadenas inválidas en NaN y luego drop silencioso. **El auditor externo no puede reconstruir el denominador real desde los artefactos.**



**Propuesta:** contador `n_dropped_invalid_sshprnamt` retornado o logueado.



\---



### P1 / P11 — `institutional_accumulation/__init__.py` desactualizado



**Evidencia:**

```python

"""Modulo Institutional Accumulation Evidence (IAE).

Estado: FA-1.1 - downloader + schema.

...

Regla: local-first. Sin push a main hasta Gate FA-1.

"""

```



La realidad es post-FA-2, NIPC implementado (spec v1.4), OpenFIGI integrado, `TARGET_UNIVERSE` operativo, 75 commits locales. Un auditor externo que abra el paquete por primera vez leería una descripción obsoleta.



\---



### P4 — Colisión de nombres `identity/` vs `sec_13f/identity/`



Dos directorios con el mismo nombre en capas distintas. Confirmado tras mapa de imports: no se cruzan, pero genera ambigüedad documental. Candidato a renombrar (`identity/` → `radar_identity/` o `openfigi_identity/`).



\---



### P12 / P2 — Referencias a spec v1.2 en docstrings



\- `delta_shares.py:4` → `(v1.2) secciones 4, 5, 6, 11`

\- `nipc.py:4` → `(v1.2) secciones 3.14, 10, 11`

\- `aggregation/__init__.py:4` → `(v1.2) seccion 11.1`



La spec vigente es **v1.4**. `security_identity.py:4` referencia `ESPECIFICACION_V11_DICTAMEN.md` — defendible porque apunta a un dictamen histórico, no a la spec.



\---



### P19 — `openfigi_client._post` sin retry en `URLError` / `socket.timeout`



**Evidencia:**

```python

# src/institutional_accumulation/identity/openfigi_client.py:71-78

except urllib.error.HTTPError as e:

&#x20;   if e.code in RETRYABLE_HTTP and attempt < max_retry:

&#x20;       ...

&#x20;       continue

&#x20;   raise RuntimeError(...)

```



Solo reintenta `HTTPError` con código `(429, 500, 503)`. `URLError` (DNS, timeout de conexión) y `socket.timeout` propagan. `map_identifiers` los captura con `except Exception` y marca **todo el chunk** como error (líneas 130-138). Coste: una caída de red de 5 segundos puede tirar un chunk de hasta 100 CUSIPs.



\---



### P20 — `load_radar_tickers` deriva el radar desde `stock_prices.parquet` con contrato implícito



**Evidencia:**

```python

# src/institutional_accumulation/identity/radar_target_catalog.py:55-65

sp = pd.read_parquet(stock_prices_path)

cols = sp.columns.get_level_values(-1).unique()

return sorted(c for c in cols if isinstance(c, str)

&#x20;             and "." not in c and not c.startswith("^") and not c.endswith("=X"))

```



\- `"." not in c` excluiría un hipotético ticker tipo `BRK.B` si apareciera. El radar actual usa `BRK-B` (guion), así que no afecta hoy, pero es un contrato implícito no declarado.

\- El docstring dice "242 por construccion actual" — si el radar cambia (IPO, delisting), cambia silenciosamente.



\---



### P21 — `_pick_row` duplicada entre `target_universe.py` y `extract_stable_identity`



**Evidencia:**

```python

# target_universe.py:66-72

def _pick_row(rows: list) -> dict | None:

&#x20;   if not isinstance(rows, list) or not rows: return None

&#x20;   for r in rows:

&#x20;       if r.get("exchCode") == "US": return r

&#x20;   return rows[0]



# openfigi_client.py:142-152

chosen = None

for r in rows:

&#x20;   if r.get("exchCode") == "US": chosen = r; break

if chosen is None: chosen = rows[0]

```



Misma regla implementada dos veces. Candidato a extraer en `openfigi_client` como helper público.



\---



### P24 — `resolve_cusips` con `exch_code="US"` hardcoded



**Evidencia:**

```python

# target_universe.py:88

raw = map_identifiers("ID_CUSIP", clean, exch_code="US", api_key=api_key)

```



Excluye listings no-USA por diseño. Es la causa raíz del hallazgo H4 del informe TOP 2000 (221 NO_ID genuinos concentrados en prefijos no-USA G/H/N/F/Y/D). Documentado en el informe pero no en el docstring del módulo.



\---



### P39 — `_derive_status` calcula `n_conflict`/`n_ambiguous` solo desde `units_current`



**Evidencia:**

```python

# nipc.py:296-306

if units_current is not None and not units_current.empty:

&#x20;   n_conflict = int((units_current["security_resolution_status"] == "CONFLICT").sum())

&#x20;   n_ambiguous = int((units_current["security_resolution_status"] == "AMBIGUOUS").sum())

```



Si el periodo previo (Q4) tuvo CONFLICT/AMBIGUOUS pero el actual (Q1) no, el status NIPC no lo refleja. Como el estado se deriva sobre el run actual, puede ser por diseño, pero el docstring de `_derive_status` no lo aclara.



\---



### P47 — `temporal_filter` no fuerza tz-naive



**Evidencia:**

```python

# temporal_filter.py:31-35

def _normalize_period(period):

&#x20;   """Convierte el periodo a Timestamp tz-naive normalizado."""

&#x20;   try:

&#x20;       ts = pd.Timestamp(period)

&#x20;   ...

&#x20;   return ts.normalize()

```



Docstring promete tz-naive pero código solo hace `.normalize()`. Si `PERIODOFREPORT` llegara tz-aware y `period` fuera tz-naive, `period_field.dt.normalize() == period_norm` (línea 78) devolvería `False` silenciosamente → 0 filings filtrados → pipeline produce output vacío sin error.



\---



### P51 — `parse_line` rellena líneas cortas sin validación inferior



**Evidencia:**

```python

# sec13f_list.py:109-110

if len(line) < LINE_WIDTH:

&#x20;   line = line.ljust(LINE_WIDTH)

```



Una línea basura de 10 chars se rellena a 80, y `POS_STATUS=(67,70)` devuelve espacios → `_classify_status("")` → `STATUS_ACTIVE`. Una línea malformada podría clasificarse como CUSIP activo.



**Propuesta:** si `len(line) < POS_STATUS[1]` → `status=None` (anomalía preservada).



\---



### P63 — `figi_lookup` no enchufado en producción



El pipeline NIPC opera con solo 2 de 4 niveles de la cadena de autoridad (`cusip_equivalence.csv` vacía por diseño + `crosswalk_internal`). `target_universe` sí usa OpenFIGI. Asimetría de fuente entre dos resolvers que operan sobre el mismo universo 13F. Candidato a documentar.



\---



### P64 — `AMBIGUOUS` (equivalence) vs `CONFLICT` (crosswalk) sin distinción documentada



**Evidencia:**

```python

# security_identity.py:262-269 (equivalence, multiples candidatos)

result["security_resolution_status"] = STATUS_AMBIGUOUS

...

# security_identity.py:283-291 (crosswalk, multiples candidatos)

result["security_resolution_status"] = STATUS_CONFLICT

```



Dos ramas devuelven estados distintos ante el mismo patrón. La spec no explica por qué. Candidato a docstring/dictamen.



\---



### P75 — `build_filing_manager_index` con `keep="first"` silencioso (nuevo)



**Evidencia:**

```python

# relationships.py:79-86

idx = submission_df[["ACCESSION_NUMBER", "CIK"]].copy()

idx = idx.rename(columns={"CIK": "filing_manager_cik"})

...

idx = idx.drop_duplicates(subset=["ACCESSION_NUMBER"], keep="first")

```



Mismo patrón que P33 (`delta_shares._attach_filing_manager`). Si SUBMISSION tuviera dos CIKs para un mismo accession (no debería, pero no hay guarda), se elige el primero silenciosamente. Candidato a `assert idx["ACCESSION_NUMBER"].is_unique`.



\---



## 5. Hallazgos BAJA



Se listan agrupados por fichero. Cada uno con referencia de línea.



### `delta_shares.py`

\- **P33** — `keep="first"` en `_attach_filing_manager` sin guarda de unicidad.

\- **P35** — `_attach_identity` con `identity_results` parcial devuelve `OBSERVED_ONLY` sin distinguir "no resuelto" de "dict incompleto".

\- **P37** — Constante `CANONICAL = "CANONICAL"` duplicada en `delta_shares.py:44` y `security_identity.py:43`.

\- **P42** — `_sum_delta` con `mask` Series no valida alineación de índice.

\- **P44** — `DISCRETION_TYPES` (nipc) y `VALID_DISCRETIONS` (delta_shares) duplican la misma tupla con nombres distintos.

\- **P45** — `_mapped_mask` usa string literal `"CANONICAL"` en vez de importar la constante.

\- **P67** — `unres_rows` con `0.0` opaco en `sshprnamt_current/previous` para filas `UNRESOLVED_IDENTITY`. Candidato a `pd.NA`.



### `nipc.py`

\- **P40** — `_derive_status` no distingue "no hay nada que medir" de "hay cosas pero no llegan al umbral" más allá de `n_total == 0`.



### `openfigi_client.py`

\- **P22** — `_batch_size` / `_sleep_seconds` hardcoded.

\- **P23** — `zip(chunk, parsed)` sin verificar longitudes. Si OpenFIGI devuelve menos respuestas, CUSIPs faltantes caen a `ERROR` genérico.

\- **P25** — Manifest con "Version: 2.0" en docstring sin constante consumible.



### `radar_target_catalog.py`

\- **P3** — `identity/__init__.py` (OpenFIGI) sin re-exports (1 LOC). Inconsistente con `aggregation/__init__.py` y `sec_13f/identity/__init__.py`.



### `security_identity.py`

\- **P66** — `resolve_batch_identities` con `str(None) == "None"` como clave del dict resultante.



### `manifest.py`

\- **P13** — Path absoluto `D:\\Macro_Sectorial` (fusionado con P18 como ALTA).



### `sec13f_list.py`

\- **P50** — `accessions_keep` sin `.str.strip()`.

\- **P78** — `_classify_raw_field` con `s == "0"` devuelve `NO_REFERENCE`, pero `classify_token("0")` devuelve `INVALID_ZERO`. Asimetría documentada pero inconsistente.



### `cusip_resolver.py`

\- **P59** — `resolve_cusip` no hace `.strip()` al ticker devuelto.

\- **P30** — `target_universe` no consulta `cusip_ticker_exceptions.csv` (deuda declarada D2).



### `amendments.py`

\- **P68** — `_parse_amendment_no` con `int(v)` falla con valores tipo `"1.0"` → None silencioso.

\- **P71** — `detect_base_filing` devuelve `"AMBIGUOUS_BASE"` como string literal en vez de `ANOMALY_AMBIGUOUS_BASE`.



### `temporal_filter.py`

\- **P46** — `_normalize_period` no fuerza tz-naive (relacionado con P47).

\- **P50** — `accessions_keep` sin strip.



### `relationships.py`

\- **P78** — Ver arriba.



\---



## 6. Hallazgos INFORMATIVO (relevantes)



\- **P5** — Contrato de pureza de `aggregation/` respetado (0 IO de ficheros).

\- **P6** — Sin cross-imports prohibidos entre capas.

\- **P7** — `manifest.py:99` con `datetime.now(timezone.utc)` legítimo para `generated_at_utc`.

\- **P8** — Resto de `datetime.now()` detectados son docstrings.

\- **P9** — `sec_13f/identity/__init__.py` (157 LOC) con re-exports + aliases correctos.

\- **P10** — 17/17 módulos funcionales con test.

\- **P16** — Sin `ticker:` prohibido ni IDs hardcodeados en lógica.

\- **P27** — `VALID_ID_TYPES` limitado (deliberado).

\- **P36** — `merged["observed_security_key"] = None` para filas canónicas (pérdida de trazabilidad canonical→observed por diseño).

\- **P41** — `compute_coverage_pairwise` early-return solo cuando ambos units vacíos.

\- **P43** — `nipc_total` no incluye unresolved por diseño.

\- **P53 / P80 / P81** — `resolve_eligibility` y `build_canonical_relationship._mk_key` son O(n²) pero aceptables una vez por run.

\- **P56** — Prohibición de `valid_to=NULL` seguido de otra fila es correcta.

\- **P76** — `classify_token` con `seq` presente pero `cik=None` devuelve `RESOLVED` (documentado).

\- **P82** — `build_om2_seq_index` maneja NaN en CIK correctamente.

\- **P90** — Cache-hit de downloader no verifica sha256 contra valor esperado (solo imprime).

\- **P91** — `extract_13f_zip` con `force=False` no detecta si el ZIP cambió y los TSV son viejos.

\- **P92** — `DEFAULT_USER_AGENT` con email del autor (SEC lo requiere así).

\- **P95** — `ingest.py` no llama a `filter_by_period`, `apply_amendments` ni NIPC. Coherente con FA-1.



\---



## 7. Anexo A — Ficheros revisados



| Categoría | Ficheros |

|---|---|

| Core sec_13f | downloader, schema, parser, storage, manifest, ingest |

| sec_13f/identity | temporal_filter, sec13f_list, cusip_resolver, relationships, amendments, security_identity |

| aggregation | delta_shares, nipc |

| identity (OpenFIGI) | openfigi_client, radar_target_catalog, target_universe |

| __init__.py | institutional_accumulation, aggregation, identity, sec_13f, sec_13f/identity |

| Tests (17) | test_sec_13f_* + test_openfigi_client + test_radar_target_catalog + test_target_universe |



\---



## 8. Anexo B — Comandos de verificación



```powershell

# Verificar P18 (path absoluto)

Select-String -Path src\\institutional_accumulation\\sec_13f\\manifest.py -Pattern 'D:\\\\Macro_Sectorial'



# Verificar P31 (docstring miente)

Select-String -Path src\\institutional_accumulation\\aggregation\\nipc.py -Pattern '_sum_delta' -Context 0,10



# Verificar P38 (denominador)

Select-String -Path src\\institutional_accumulation\\aggregation\\nipc.py -Pattern 'union_sec|both_mapped_sec'



# Verificar P60 (inferencia prefijo)

Select-String -Path src\\institutional_accumulation\\sec_13f\\identity\\security_identity.py -Pattern '_normalize_canonical' -Context 0,15



# Verificar P61 (sin vigencia)

Select-String -Path src\\institutional_accumulation\\sec_13f\\identity\\security_identity.py -Pattern 'etf_holdings'



# Verificar P14-BIS (fillna)

Select-String -Path src\\institutional_accumulation\\aggregation\\delta_shares.py -Pattern 'fillna'

```



\---



## 9. Anexo C — Hipótesis cerradas como NO BUG



\- **P14** — Los `fillna(0.0)` en `delta_shares.py:254/259` son correctos por semántica de FULL OUTER JOIN. Refinado a P14-BIS (defecto de robustez, no de lógica).

\- **P29** — `Path('sec_13f_2026Q1.json').with_suffix('.json.tmp')` funciona sin excepción. Verificado empíricamente.



\---



## 10. Recomendaciones por fase



### Fase 1 — Cierre del ciclo F2.3-bis actual (sin tocar código)

1\. **Dictamen P38** — bloquear `THRESHOLD_2` hasta resolución del denominador.

2\. **Dictamen P60** — definir contrato de prefijos de `canonical_security`.

3\. **Dictamen P70** — semántica de `HR_PLUS_RESTATEMENT` vs `HR_CHAIN_RESTATEMENT`.



### Fase 2 — Fixes mecánicos autorizables

1\. **P18 / P26** — path absoluto → env var + `Path(__file__)`.

2\. **P11 / P1 / P2 / P12** — actualizar `__init__.py` y docstrings a spec v1.4.

3\. **P14-BIS** — condicionar fillna a `_merge`.

4\. **P32** — emitir contador `n_dropped_invalid_sshprnamt`.

5\. **P51** — `parse_line` con guarda inferior.



### Fase 3 — Refactor arquitectónico (candidato a ciclo propio)

1\. **P4** — renombrar `identity/` para evitar colisión.

2\. **P21** — extraer `_pick_row` a helper compartido.

3\. **P37 / P44 / P45** — unificar constantes.

4\. **P75 / P33** — guardas de unicidad en índices.



### Fase 4 — Deuda declarada (mantener como tal)

\- **D1-D5** del transfer doc (multiple_openfigi_hits hardcoded, target_universe sin Q-CUR, validación CUSIP, BRK-B/MOG-A, EOL hygiene).

\- **P30 / P61 / P63** — asimetrías estructurales del resolver. Documentar, no arreglar sin dictamen.



\---



## 11. Cierre



El módulo IAE está **funcionalmente correcto en su núcleo** (NIPC, delta_shares, resolver de identidad) y **respeta las prohibiciones duras verificables**. Los hallazgos ALTA son de portabilidad, no de corrección de cálculo. Los candidatos a dictamen (P38, P60, P61, P70) son los únicos que pueden afectar a resultados publicados.



**Ningún fix se ha aplicado.** Este documento es un borrador interno pendiente de revisión por el responsable antes de envío al auditor externo.



\---



**Fin del informe. Pendiente de validación del usuario antes de:**



