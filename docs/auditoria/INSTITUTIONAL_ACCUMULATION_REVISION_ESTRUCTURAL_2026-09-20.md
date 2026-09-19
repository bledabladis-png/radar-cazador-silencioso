\# INFORME DE REVISIÓN ESTRUCTURAL DEL MÓDULO INSTITUTIONAL ACCUMULATION EVIDENCE (IAE)



\*\*Fecha:\*\* 2026-09-20

\*\*Autor:\*\* Ingeniero Supervisor (revisión interna)

\*\*Destinatario:\*\* Auditor externo

\*\*Alcance:\*\* `src/institutional\_accumulation/` — 22 ficheros `.py` (2.456 LOC no-vacías), 17 ficheros de test, 53 documentos en `docs/auditoria/`

\*\*HEAD al cierre:\*\* `77d4785` (ahead 75 de `origin/main`)

\*\*Estado:\*\* BORRADOR INTERNO. Pendiente de validación por el responsable antes de envío externo.



\---



\## 1. Resumen ejecutivo



Se ha realizado una revisión estructural completa del módulo IAE con \*\*95 observaciones\*\* clasificadas: \*\*3 ALTA\*\*, \*\*20 MEDIA\*\*, \*\*23 BAJA\*\*, \*\*\~49 INFORMATIVO\*\*. Dos hipótesis iniciales se cerraron como NO BUG tras verificación empírica (P14, P29).



\*\*Candidatos críticos que requieren dictamen del auditor (no fix mecánico):\*\*



1\. \*\*P38\*\* — Ambigüedad semántica en el denominador de `paired\_weighted\_share\_coverage` (unión vs intersección). Afecta directamente al `THRESHOLD\_2` que sigue UNDEFINED. \*\*Bloquea la fijación de thresholds\*\*.

2\. \*\*P60\*\* — `\_normalize\_canonical` infiere prefijo `equity:` sin evidencia de formato. Posible violación del contrato C1-revisada ("no inferir").

3\. \*\*P61\*\* — `etf\_holdings.csv` cargado como crosswalk sin vigencia temporal. Raíz estructural de la circularidad documentada por el propio auditor en F2.1-bis.

4\. \*\*P70\*\* — Asimetría en `classify\_strategy` al decidir `HR\_PLUS\_RESTATEMENT` vs `HR\_PLUS\_NEW\_HOLDINGS` (mira `iloc\[1]` para un caso, `iloc\[-1]` para otro).

5\. \*\*P31\*\* — `\_sum\_delta` promete devolver `None` en el docstring pero devuelve siempre `0.0`. No distingue "no hay datos" de "NIPC neto = 0".



\*\*Candidatos de fix mecánico (una vez autorizado):\*\*



1\. \*\*P18 / P26\*\* — Path absoluto `D:\\Macro\_Sectorial` hardcoded en `manifest.py`. Rompe portabilidad (CI, clones, cualquier máquina no-Windows-D).

2\. \*\*P14-BIS\*\* — `fillna(0.0)` en `delta\_shares` es correcto hoy pero no distingue "ausencia de fila" de "fila con NaN". Fix preventivo recomendado.

3\. \*\*P32\*\* — `\_filter\_canonical` descarta filas con `SSHPRNAMT` no parseable sin contador. Pérdida silenciosa de trazabilidad.

4\. \*\*P11 / P1 / P2 / P12\*\* — Desfases documentales en `\_\_init\_\_.py` (estado FA-1.1) y referencias a spec v1.2.



\*\*Estado de contratos:\*\* El módulo respeta las prohibiciones duras verificables (`ticker:` no usado, `datetime.now()` solo en `generated\_at\_utc`, sin IO en `aggregation/`, sin IDs hardcodeados en lógica). El aislamiento (`src/` y `scripts/` no consumen el módulo) es coherente con la regla local-first IAE.



\---



\## 2. Metodología



Cada hallazgo se ha clasificado en cuatro niveles:



\- \*\*ALTA:\*\* Rompe portabilidad, produce resultados incorrectos, o viola contrato declarado de forma verificable.

\- \*\*MEDIA:\*\* Riesgo de inconsistencia futura, pérdida de trazabilidad, o ambigüedad semántica que requiere dictamen.

\- \*\*BAJA:\*\* Defecto cosmético, inconsistencia de estilo, o deuda declarada.

\- \*\*INFORMATIVO:\*\* Observación sin defecto, documentada para contexto del auditor.



\*\*Regla aplicada:\*\* cada afirmación se ancla a línea de código verificada. Las hipótesis no confirmadas se marcan explícitamente como tales. Los hallazgos cerrados durante la revisión se documentan en el Anexo C.



\---



\## 3. Hallazgos ALTA



\### P18 / P26 — Path absoluto `D:\\Macro\_Sectorial` en `manifest.py`



\*\*Evidencia:\*\*

```python

\# src/institutional\_accumulation/sec\_13f/manifest.py:21

DEFAULT\_PROJECT\_ROOT = Path(r"D:\\Macro\_Sectorial")

```

`build\_manifest(..., project\_root=DEFAULT\_PROJECT\_ROOT)` es el default de la firma (línea 59). `ingest\_13f(..., project\_root=DEFAULT\_PROJECT\_ROOT)` propaga el acoplamiento (`ingest.py:38`).



\*\*Impacto:\*\*

\- Cualquier llamada desde CI, WSL, Mac, Linux o clon en ruta distinta produce rutas absolutas inválidas en el campo `file` de cada entrada del manifest (`\_relpath\_or\_abs` hace fallback a absoluta si `relative\_to` falla — línea 31).

\- Rompe portabilidad del módulo, declarado en el prompt como "local-first refactor" y "verificar en CI".

\- El README del módulo no declara esta restricción.



\*\*Propuesta:\*\*

```python

DEFAULT\_PROJECT\_ROOT = Path(os.environ.get(

&#x20;   "IAE\_PROJECT\_ROOT",

&#x20;   Path(\_\_file\_\_).resolve().parents\[4]

))

```

O `None` explícito con resolución obligatoria por caller.



\*\*Verificación:\*\* `grep -r "D:\\\\\\\\Macro\_Sectorial" src/` → 2 matches (manifest.py:21, schema.py:7 en docstring).



\---



\## 4. Hallazgos MEDIA



\### P38 — Denominador de `paired\_weighted\_share\_coverage` (candidato estrella)



\*\*Evidencia cruzada:\*\*

```python

\# src/institutional\_accumulation/aggregation/nipc.py (docstring, líneas \~137-142)

paired\_weighted\_share\_coverage

&#x20; = sum(SSHPRNAMT de S con mapping en ambos)

&#x20;   / sum(SSHPRNAMT de S con posicion en ambos)

```



```python

\# src/institutional\_accumulation/aggregation/nipc.py (código, líneas \~218-228)

union\_sec = sec\_c | sec\_p

both\_mapped\_sec = mapped\_c \& mapped\_p

...

denom = sum(by\_sec.get(k, 0.0) for k in union\_sec)

numer = sum(by\_sec.get(k, 0.0) for k in both\_mapped\_sec)

```



\*\*El docstring dice "en ambos". El código usa la unión (`Q4 ∪ Q1`), no la intersección (`Q4 ∩ Q1`).\*\* Una de las dos definiciones es incorrecta.



\*\*Impacto:\*\*

\- Si debe ser intersección, todas las cifras publicadas en el baseline (commit `e9fd830`) y en el TOP 2000 (`32baf9d`) están mal calculadas.

\- `THRESHOLD\_2` depende directamente de este valor.

\- Es exactamente el tipo de ambigüedad que el micro-gate F2.2-v2 pretendía eliminar con "PAIRED\_WEIGHTED\_SHARE\_COVERAGE: convencion Q4/Q1 inequivoca (max)". La convención "max por security" está aplicada en `by\_sec`, pero el conjunto de claves del denominador no está fijado por el cambio obligatorio #5 del dictamen F2.2.



\*\*Propuesta:\*\* dictamen del auditor antes de cualquier fix. Documento afectado: `NIPC\_COVERAGE\_POLICY\_V12\_PROPUESTA.md` (borrador).



\---



\### P60 — `\_normalize\_canonical` infiere prefijo `equity:` sin evidencia



\*\*Evidencia:\*\*

```python

\# src/institutional\_accumulation/sec\_13f/identity/security\_identity.py:60-81

def \_normalize\_canonical(value):

&#x20;   ...

&#x20;   if s.startswith("equity:") or s.startswith("figi:"):

&#x20;       return s

&#x20;   return f"equity:{s}"

```



\*\*Impacto:\*\*

\- Un `canonical\_security = "BBG001S69V32"` (FIGI desnudo, 12 chars alfanuméricos) se convierte en `"equity:BBG001S69V32"` en vez de `"figi:BBG001S69V32"`.

\- El prefijo se \*\*asigna\*\* sin verificar el formato del valor. Viola nominalmente el contrato C1-revisada: \*"NO inferir equivalence sin entry explicita"\* y \*"NO presentar `cusip:<CUSIP>` como identidad canonica estable sin evidencia"\* — la misma lógica aplica a la dirección contraria.

\- `crosswalk\_internal` (líneas 285-295) devuelve \*\*tickers Yahoo\*\* (no FIGI), que se prefijan como `equity:<ticker>`. Es correcto como convención, pero el código no distingue "ticker" de "FIGI" al prefijar.



\*\*Propuesta:\*\* dictamen. Opciones:

1\. Rechazar el prefijo `equity:` para valores con formato FIGI (regex `^BBG\[A-Z0-9]{9}$`).

2\. Declarar formalmente que `canonical\_security` es siempre `equity:<ticker>` o `figi:<FIGI>` y validar el formato.



\---



\### P61 — `etf\_holdings.csv` sin vigencia temporal en el crosswalk



\*\*Evidencia:\*\*

```python

\# src/institutional\_accumulation/sec\_13f/identity/security\_identity.py:198-217

def load\_crosswalk\_internal(exceptions\_path=..., etf\_holdings\_path=...):

&#x20;   ...

&#x20;   if p2.exists():

&#x20;       h = pd.read\_csv(p2, dtype=str)

&#x20;       for \_, r in h.iterrows():

&#x20;           rows.append({

&#x20;               "CUSIP": str(r\["identifier"]).strip(),

&#x20;               "ticker": str(r\["ticker"]).strip(),

&#x20;               "valid\_from": None,   # <-- siempre activo

&#x20;               "valid\_to": None,

&#x20;               "source": "etf\_holdings",

&#x20;           })

```



\*\*Impacto:\*\*

\- Los CUSIPs que provienen de `etf\_holdings.csv` quedan con vigencia `(-∞, +∞)`. Si un ETF holding tiene un ticker obsoleto (post corporate action), el resolver lo devuelve como `CANONICAL` sin verificación.

\- \*\*Es la raíz estructural documentada por el auditor en F2.1-bis\*\* ("el 90.91% (220/242) que la policy v1.0 L115 cita como `mapping\_coverage` sale exactamente de `etf\_holdings.csv`, parte del crosswalk evaluado"). No es hallazgo nuevo, pero debe figurar en el informe como limitación cuantificada del resolver.

\- En el baseline Fase A, `coverage\_previous == coverage\_current == 1.0` sobre `OPERATIONAL\_EQUITY` por construcción.



\*\*Propuesta:\*\* no es fix mecánico. Añadir sección explícita al informe de cobertura. Candidato al cambio obligatorio #8 del dictamen F2.2 ("trazabilidad explícita del cambio semántico") — declarar que `crosswalk\_internal` \*\*no tiene cobertura temporal verificable\*\*.



\---



\### P70 — Asimetría en `classify\_strategy` (nuevo)



\*\*Evidencia:\*\*

```python

\# src/institutional\_accumulation/sec\_13f/identity/amendments.py:181-198

if base == STRATEGY\_HR\_PLUS\_RESTATEMENT:

&#x20;   if "AMENDMENTTYPE" in filings\_group.columns:

&#x20;       at = \_norm\_str(filings\_group.iloc\[1].get("AMENDMENTTYPE"))   # <-- iloc\[1] fijo

&#x20;       if at == "NEW HOLDINGS":

&#x20;           return STRATEGY\_HR\_PLUS\_NEW\_HOLDINGS

&#x20;   return STRATEGY\_HR\_PLUS\_RESTATEMENT



if base == STRATEGY\_HR\_CHAIN\_RESTATEMENT:

&#x20;   if "AMENDMENTTYPE" in filings\_group.columns and len(filings\_group) >= 2:

&#x20;       last = \_norm\_str(filings\_group.iloc\[-1].get("AMENDMENTTYPE"))  # <-- iloc\[-1]

&#x20;       if last == "NEW HOLDINGS":

&#x20;           return STRATEGY\_HR\_COMPOSITE

&#x20;   return STRATEGY\_HR\_CHAIN\_RESTATEMENT

```



Un caso mira \*\*el segundo filing\*\* (`iloc\[1]`) y el otro \*\*el último\*\* (`iloc\[-1]`). La lógica de negocio 13F suele ser: la cadena de amendments aplica en orden (`AMENDMENTNO` ascendente) y \*\*el resultado final\*\* determina la composición. Mirar `iloc\[1]` para `HR\_PLUS\_RESTATEMENT` (que solo tiene 2 filings) es equivalente a mirar el último porque solo hay 2. Pero el código es frágil: si en el futuro se admite `HR + HR/A + HR/A` con el primero NEW HOLDINGS, `iloc\[1]` decide prematuramente.



\*\*Propuesta:\*\* dictamen del auditor sobre la semántica correcta. Unificar la lógica a `iloc\[-1]` para ambos casos, o documentar la asimetría explícitamente.



\---



\### P31 — `\_sum\_delta` docstring miente sobre el retorno



\*\*Evidencia:\*\*

```python

\# src/institutional\_accumulation/aggregation/nipc.py:64-72

def \_sum\_delta(delta\_df, mask=None):

&#x20;   """Suma delta\_shares ignorando NaN. None si no hay filas validas."""

&#x20;   if delta\_df is None or delta\_df.empty:

&#x20;       return 0.0

&#x20;   ...

&#x20;   return float(s.sum()) if len(s) > 0 else 0.0

```



Devuelve \*\*siempre\*\* `0.0`, nunca `None`. Contradice el principio "No confundir VALID con UNAVAILABLE" del prompt.



\*\*Impacto:\*\*

\- `nipc\_total = 0` puede significar "no hay datos" o "el neto es exactamente 0". Aunque `\_derive\_status` distingue por `n\_total == 0`, el campo `nipc\_total` aislado es ambiguo para un consumidor externo (informe, auditoría a posteriori, futuros readers).



\*\*Propuesta:\*\* mantener `0.0` (compatibilidad con el motor actual) y añadir `nipc\_total\_available: bool` o `n\_observations: int` al dict. Requiere dictamen.



\---



\### P14-BIS — `fillna(0.0)` sin distinguir origen de NaN



\*\*Evidencia:\*\*

```python

\# src/institutional\_accumulation/aggregation/delta\_shares.py:252-259

merged\["sshprnamt\_current"] = (

&#x20;   pd.to\_numeric(merged\["sshprnamt\_current"], errors="coerce")

&#x20;   .fillna(0.0).astype("Float64")

)

merged\["sshprnamt\_previous"] = (

&#x20;   pd.to\_numeric(merged\["sshprnamt\_previous"], errors="coerce")

&#x20;   .fillna(0.0).astype("Float64")

)

```



\*\*Análisis:\*\* Es correcto hoy (los NaN provienen solo del FULL OUTER JOIN, no de datos corruptos, porque `\_filter\_canonical` ya hace `dropna(subset=\["SSHPRNAMT\_f"])`). \*\*No imputa datos ficticios: la ausencia de fila es equivalente semánticamente a "cantidad 0" en el periodo.\*\*



\*\*Riesgo:\*\* si en el futuro se relaja `\_filter\_canonical`, llega un parquet corrupto, o se reutiliza la función con otra fuente, el fillna convertiría un dato corrupto en `0.0` → EXIT falso → \*\*contaminación de `nipc\_total` sin traza\*\*.



\*\*Propuesta:\*\* condicionar el fillna a `\_merge` en lugar de al valor NaN:

```python

mask\_new = merged\["\_merge"] == "left\_only"

mask\_exit = merged\["\_merge"] == "right\_only"

merged.loc\[mask\_new, "sshprnamt\_previous"] = 0.0

merged.loc\[mask\_exit, "sshprnamt\_current"] = 0.0

bad = (merged\["\_merge"] == "both") \& (

&#x20;   merged\["sshprnamt\_current"].isna() | merged\["sshprnamt\_previous"].isna()

)

if bad.any():

&#x20;   raise ValueError(f"corrupcion: {int(bad.sum())} filas BOTH con NaN")

```



\---



\### P32 — Dropna silencioso en `\_filter\_canonical`



\*\*Evidencia:\*\*

```python

\# src/institutional\_accumulation/aggregation/delta\_shares.py:86-87

df\["SSHPRNAMT\_f"] = pd.to\_numeric(df\["SSHPRNAMT"], errors="coerce")

df = df.dropna(subset=\["SSHPRNAMT\_f"])

```



Sobre 3.3M filas de `INFOTABLE` Q1 2026, `errors="coerce"` puede convertir cadenas inválidas en NaN y luego drop silencioso. \*\*El auditor externo no puede reconstruir el denominador real desde los artefactos.\*\*



\*\*Propuesta:\*\* contador `n\_dropped\_invalid\_sshprnamt` retornado o logueado.



\---



\### P1 / P11 — `institutional\_accumulation/\_\_init\_\_.py` desactualizado



\*\*Evidencia:\*\*

```python

"""Modulo Institutional Accumulation Evidence (IAE).

Estado: FA-1.1 - downloader + schema.

...

Regla: local-first. Sin push a main hasta Gate FA-1.

"""

```



La realidad es post-FA-2, NIPC implementado (spec v1.4), OpenFIGI integrado, `TARGET\_UNIVERSE` operativo, 75 commits locales. Un auditor externo que abra el paquete por primera vez leería una descripción obsoleta.



\---



\### P4 — Colisión de nombres `identity/` vs `sec\_13f/identity/`



Dos directorios con el mismo nombre en capas distintas. Confirmado tras mapa de imports: no se cruzan, pero genera ambigüedad documental. Candidato a renombrar (`identity/` → `radar\_identity/` o `openfigi\_identity/`).



\---



\### P12 / P2 — Referencias a spec v1.2 en docstrings



\- `delta\_shares.py:4` → `(v1.2) secciones 4, 5, 6, 11`

\- `nipc.py:4` → `(v1.2) secciones 3.14, 10, 11`

\- `aggregation/\_\_init\_\_.py:4` → `(v1.2) seccion 11.1`



La spec vigente es \*\*v1.4\*\*. `security\_identity.py:4` referencia `ESPECIFICACION\_V11\_DICTAMEN.md` — defendible porque apunta a un dictamen histórico, no a la spec.



\---



\### P19 — `openfigi\_client.\_post` sin retry en `URLError` / `socket.timeout`



\*\*Evidencia:\*\*

```python

\# src/institutional\_accumulation/identity/openfigi\_client.py:71-78

except urllib.error.HTTPError as e:

&#x20;   if e.code in RETRYABLE\_HTTP and attempt < max\_retry:

&#x20;       ...

&#x20;       continue

&#x20;   raise RuntimeError(...)

```



Solo reintenta `HTTPError` con código `(429, 500, 503)`. `URLError` (DNS, timeout de conexión) y `socket.timeout` propagan. `map\_identifiers` los captura con `except Exception` y marca \*\*todo el chunk\*\* como error (líneas 130-138). Coste: una caída de red de 5 segundos puede tirar un chunk de hasta 100 CUSIPs.



\---



\### P20 — `load\_radar\_tickers` deriva el radar desde `stock\_prices.parquet` con contrato implícito



\*\*Evidencia:\*\*

```python

\# src/institutional\_accumulation/identity/radar\_target\_catalog.py:55-65

sp = pd.read\_parquet(stock\_prices\_path)

cols = sp.columns.get\_level\_values(-1).unique()

return sorted(c for c in cols if isinstance(c, str)

&#x20;             and "." not in c and not c.startswith("^") and not c.endswith("=X"))

```



\- `"." not in c` excluiría un hipotético ticker tipo `BRK.B` si apareciera. El radar actual usa `BRK-B` (guion), así que no afecta hoy, pero es un contrato implícito no declarado.

\- El docstring dice "242 por construccion actual" — si el radar cambia (IPO, delisting), cambia silenciosamente.



\---



\### P21 — `\_pick\_row` duplicada entre `target\_universe.py` y `extract\_stable\_identity`



\*\*Evidencia:\*\*

```python

\# target\_universe.py:66-72

def \_pick\_row(rows: list) -> dict | None:

&#x20;   if not isinstance(rows, list) or not rows: return None

&#x20;   for r in rows:

&#x20;       if r.get("exchCode") == "US": return r

&#x20;   return rows\[0]



\# openfigi\_client.py:142-152

chosen = None

for r in rows:

&#x20;   if r.get("exchCode") == "US": chosen = r; break

if chosen is None: chosen = rows\[0]

```



Misma regla implementada dos veces. Candidato a extraer en `openfigi\_client` como helper público.



\---



\### P24 — `resolve\_cusips` con `exch\_code="US"` hardcoded



\*\*Evidencia:\*\*

```python

\# target\_universe.py:88

raw = map\_identifiers("ID\_CUSIP", clean, exch\_code="US", api\_key=api\_key)

```



Excluye listings no-USA por diseño. Es la causa raíz del hallazgo H4 del informe TOP 2000 (221 NO\_ID genuinos concentrados en prefijos no-USA G/H/N/F/Y/D). Documentado en el informe pero no en el docstring del módulo.



\---



\### P39 — `\_derive\_status` calcula `n\_conflict`/`n\_ambiguous` solo desde `units\_current`



\*\*Evidencia:\*\*

```python

\# nipc.py:296-306

if units\_current is not None and not units\_current.empty:

&#x20;   n\_conflict = int((units\_current\["security\_resolution\_status"] == "CONFLICT").sum())

&#x20;   n\_ambiguous = int((units\_current\["security\_resolution\_status"] == "AMBIGUOUS").sum())

```



Si el periodo previo (Q4) tuvo CONFLICT/AMBIGUOUS pero el actual (Q1) no, el status NIPC no lo refleja. Como el estado se deriva sobre el run actual, puede ser por diseño, pero el docstring de `\_derive\_status` no lo aclara.



\---



\### P47 — `temporal\_filter` no fuerza tz-naive



\*\*Evidencia:\*\*

```python

\# temporal\_filter.py:31-35

def \_normalize\_period(period):

&#x20;   """Convierte el periodo a Timestamp tz-naive normalizado."""

&#x20;   try:

&#x20;       ts = pd.Timestamp(period)

&#x20;   ...

&#x20;   return ts.normalize()

```



Docstring promete tz-naive pero código solo hace `.normalize()`. Si `PERIODOFREPORT` llegara tz-aware y `period` fuera tz-naive, `period\_field.dt.normalize() == period\_norm` (línea 78) devolvería `False` silenciosamente → 0 filings filtrados → pipeline produce output vacío sin error.



\---



\### P51 — `parse\_line` rellena líneas cortas sin validación inferior



\*\*Evidencia:\*\*

```python

\# sec13f\_list.py:109-110

if len(line) < LINE\_WIDTH:

&#x20;   line = line.ljust(LINE\_WIDTH)

```



Una línea basura de 10 chars se rellena a 80, y `POS\_STATUS=(67,70)` devuelve espacios → `\_classify\_status("")` → `STATUS\_ACTIVE`. Una línea malformada podría clasificarse como CUSIP activo.



\*\*Propuesta:\*\* si `len(line) < POS\_STATUS\[1]` → `status=None` (anomalía preservada).



\---



\### P63 — `figi\_lookup` no enchufado en producción



El pipeline NIPC opera con solo 2 de 4 niveles de la cadena de autoridad (`cusip\_equivalence.csv` vacía por diseño + `crosswalk\_internal`). `target\_universe` sí usa OpenFIGI. Asimetría de fuente entre dos resolvers que operan sobre el mismo universo 13F. Candidato a documentar.



\---



\### P64 — `AMBIGUOUS` (equivalence) vs `CONFLICT` (crosswalk) sin distinción documentada



\*\*Evidencia:\*\*

```python

\# security\_identity.py:262-269 (equivalence, multiples candidatos)

result\["security\_resolution\_status"] = STATUS\_AMBIGUOUS

...

\# security\_identity.py:283-291 (crosswalk, multiples candidatos)

result\["security\_resolution\_status"] = STATUS\_CONFLICT

```



Dos ramas devuelven estados distintos ante el mismo patrón. La spec no explica por qué. Candidato a docstring/dictamen.



\---



\### P75 — `build\_filing\_manager\_index` con `keep="first"` silencioso (nuevo)



\*\*Evidencia:\*\*

```python

\# relationships.py:79-86

idx = submission\_df\[\["ACCESSION\_NUMBER", "CIK"]].copy()

idx = idx.rename(columns={"CIK": "filing\_manager\_cik"})

...

idx = idx.drop\_duplicates(subset=\["ACCESSION\_NUMBER"], keep="first")

```



Mismo patrón que P33 (`delta\_shares.\_attach\_filing\_manager`). Si SUBMISSION tuviera dos CIKs para un mismo accession (no debería, pero no hay guarda), se elige el primero silenciosamente. Candidato a `assert idx\["ACCESSION\_NUMBER"].is\_unique`.



\---



\## 5. Hallazgos BAJA



Se listan agrupados por fichero. Cada uno con referencia de línea.



\### `delta\_shares.py`

\- \*\*P33\*\* — `keep="first"` en `\_attach\_filing\_manager` sin guarda de unicidad.

\- \*\*P35\*\* — `\_attach\_identity` con `identity\_results` parcial devuelve `OBSERVED\_ONLY` sin distinguir "no resuelto" de "dict incompleto".

\- \*\*P37\*\* — Constante `CANONICAL = "CANONICAL"` duplicada en `delta\_shares.py:44` y `security\_identity.py:43`.

\- \*\*P42\*\* — `\_sum\_delta` con `mask` Series no valida alineación de índice.

\- \*\*P44\*\* — `DISCRETION\_TYPES` (nipc) y `VALID\_DISCRETIONS` (delta\_shares) duplican la misma tupla con nombres distintos.

\- \*\*P45\*\* — `\_mapped\_mask` usa string literal `"CANONICAL"` en vez de importar la constante.

\- \*\*P67\*\* — `unres\_rows` con `0.0` opaco en `sshprnamt\_current/previous` para filas `UNRESOLVED\_IDENTITY`. Candidato a `pd.NA`.



\### `nipc.py`

\- \*\*P40\*\* — `\_derive\_status` no distingue "no hay nada que medir" de "hay cosas pero no llegan al umbral" más allá de `n\_total == 0`.



\### `openfigi\_client.py`

\- \*\*P22\*\* — `\_batch\_size` / `\_sleep\_seconds` hardcoded.

\- \*\*P23\*\* — `zip(chunk, parsed)` sin verificar longitudes. Si OpenFIGI devuelve menos respuestas, CUSIPs faltantes caen a `ERROR` genérico.

\- \*\*P25\*\* — Manifest con "Version: 2.0" en docstring sin constante consumible.



\### `radar\_target\_catalog.py`

\- \*\*P3\*\* — `identity/\_\_init\_\_.py` (OpenFIGI) sin re-exports (1 LOC). Inconsistente con `aggregation/\_\_init\_\_.py` y `sec\_13f/identity/\_\_init\_\_.py`.



\### `security\_identity.py`

\- \*\*P66\*\* — `resolve\_batch\_identities` con `str(None) == "None"` como clave del dict resultante.



\### `manifest.py`

\- \*\*P13\*\* — Path absoluto `D:\\Macro\_Sectorial` (fusionado con P18 como ALTA).



\### `sec13f\_list.py`

\- \*\*P50\*\* — `accessions\_keep` sin `.str.strip()`.

\- \*\*P78\*\* — `\_classify\_raw\_field` con `s == "0"` devuelve `NO\_REFERENCE`, pero `classify\_token("0")` devuelve `INVALID\_ZERO`. Asimetría documentada pero inconsistente.



\### `cusip\_resolver.py`

\- \*\*P59\*\* — `resolve\_cusip` no hace `.strip()` al ticker devuelto.

\- \*\*P30\*\* — `target\_universe` no consulta `cusip\_ticker\_exceptions.csv` (deuda declarada D2).



\### `amendments.py`

\- \*\*P68\*\* — `\_parse\_amendment\_no` con `int(v)` falla con valores tipo `"1.0"` → None silencioso.

\- \*\*P71\*\* — `detect\_base\_filing` devuelve `"AMBIGUOUS\_BASE"` como string literal en vez de `ANOMALY\_AMBIGUOUS\_BASE`.



\### `temporal\_filter.py`

\- \*\*P46\*\* — `\_normalize\_period` no fuerza tz-naive (relacionado con P47).

\- \*\*P50\*\* — `accessions\_keep` sin strip.



\### `relationships.py`

\- \*\*P78\*\* — Ver arriba.



\---



\## 6. Hallazgos INFORMATIVO (relevantes)



\- \*\*P5\*\* — Contrato de pureza de `aggregation/` respetado (0 IO de ficheros).

\- \*\*P6\*\* — Sin cross-imports prohibidos entre capas.

\- \*\*P7\*\* — `manifest.py:99` con `datetime.now(timezone.utc)` legítimo para `generated\_at\_utc`.

\- \*\*P8\*\* — Resto de `datetime.now()` detectados son docstrings.

\- \*\*P9\*\* — `sec\_13f/identity/\_\_init\_\_.py` (157 LOC) con re-exports + aliases correctos.

\- \*\*P10\*\* — 17/17 módulos funcionales con test.

\- \*\*P16\*\* — Sin `ticker:` prohibido ni IDs hardcodeados en lógica.

\- \*\*P27\*\* — `VALID\_ID\_TYPES` limitado (deliberado).

\- \*\*P36\*\* — `merged\["observed\_security\_key"] = None` para filas canónicas (pérdida de trazabilidad canonical→observed por diseño).

\- \*\*P41\*\* — `compute\_coverage\_pairwise` early-return solo cuando ambos units vacíos.

\- \*\*P43\*\* — `nipc\_total` no incluye unresolved por diseño.

\- \*\*P53 / P80 / P81\*\* — `resolve\_eligibility` y `build\_canonical\_relationship.\_mk\_key` son O(n²) pero aceptables una vez por run.

\- \*\*P56\*\* — Prohibición de `valid\_to=NULL` seguido de otra fila es correcta.

\- \*\*P76\*\* — `classify\_token` con `seq` presente pero `cik=None` devuelve `RESOLVED` (documentado).

\- \*\*P82\*\* — `build\_om2\_seq\_index` maneja NaN en CIK correctamente.

\- \*\*P90\*\* — Cache-hit de downloader no verifica sha256 contra valor esperado (solo imprime).

\- \*\*P91\*\* — `extract\_13f\_zip` con `force=False` no detecta si el ZIP cambió y los TSV son viejos.

\- \*\*P92\*\* — `DEFAULT\_USER\_AGENT` con email del autor (SEC lo requiere así).

\- \*\*P95\*\* — `ingest.py` no llama a `filter\_by\_period`, `apply\_amendments` ni NIPC. Coherente con FA-1.



\---



\## 7. Anexo A — Ficheros revisados



| Categoría | Ficheros |

|---|---|

| Core sec\_13f | downloader, schema, parser, storage, manifest, ingest |

| sec\_13f/identity | temporal\_filter, sec13f\_list, cusip\_resolver, relationships, amendments, security\_identity |

| aggregation | delta\_shares, nipc |

| identity (OpenFIGI) | openfigi\_client, radar\_target\_catalog, target\_universe |

| \_\_init\_\_.py | institutional\_accumulation, aggregation, identity, sec\_13f, sec\_13f/identity |

| Tests (17) | test\_sec\_13f\_\* + test\_openfigi\_client + test\_radar\_target\_catalog + test\_target\_universe |



\---



\## 8. Anexo B — Comandos de verificación



```powershell

\# Verificar P18 (path absoluto)

Select-String -Path src\\institutional\_accumulation\\sec\_13f\\manifest.py -Pattern 'D:\\\\Macro\_Sectorial'



\# Verificar P31 (docstring miente)

Select-String -Path src\\institutional\_accumulation\\aggregation\\nipc.py -Pattern '\_sum\_delta' -Context 0,10



\# Verificar P38 (denominador)

Select-String -Path src\\institutional\_accumulation\\aggregation\\nipc.py -Pattern 'union\_sec|both\_mapped\_sec'



\# Verificar P60 (inferencia prefijo)

Select-String -Path src\\institutional\_accumulation\\sec\_13f\\identity\\security\_identity.py -Pattern '\_normalize\_canonical' -Context 0,15



\# Verificar P61 (sin vigencia)

Select-String -Path src\\institutional\_accumulation\\sec\_13f\\identity\\security\_identity.py -Pattern 'etf\_holdings'



\# Verificar P14-BIS (fillna)

Select-String -Path src\\institutional\_accumulation\\aggregation\\delta\_shares.py -Pattern 'fillna'

```



\---



\## 9. Anexo C — Hipótesis cerradas como NO BUG



\- \*\*P14\*\* — Los `fillna(0.0)` en `delta\_shares.py:254/259` son correctos por semántica de FULL OUTER JOIN. Refinado a P14-BIS (defecto de robustez, no de lógica).

\- \*\*P29\*\* — `Path('sec\_13f\_2026Q1.json').with\_suffix('.json.tmp')` funciona sin excepción. Verificado empíricamente.



\---



\## 10. Recomendaciones por fase



\### Fase 1 — Cierre del ciclo F2.3-bis actual (sin tocar código)

1\. \*\*Dictamen P38\*\* — bloquear `THRESHOLD\_2` hasta resolución del denominador.

2\. \*\*Dictamen P60\*\* — definir contrato de prefijos de `canonical\_security`.

3\. \*\*Dictamen P70\*\* — semántica de `HR\_PLUS\_RESTATEMENT` vs `HR\_CHAIN\_RESTATEMENT`.



\### Fase 2 — Fixes mecánicos autorizables

1\. \*\*P18 / P26\*\* — path absoluto → env var + `Path(\_\_file\_\_)`.

2\. \*\*P11 / P1 / P2 / P12\*\* — actualizar `\_\_init\_\_.py` y docstrings a spec v1.4.

3\. \*\*P14-BIS\*\* — condicionar fillna a `\_merge`.

4\. \*\*P32\*\* — emitir contador `n\_dropped\_invalid\_sshprnamt`.

5\. \*\*P51\*\* — `parse\_line` con guarda inferior.



\### Fase 3 — Refactor arquitectónico (candidato a ciclo propio)

1\. \*\*P4\*\* — renombrar `identity/` para evitar colisión.

2\. \*\*P21\*\* — extraer `\_pick\_row` a helper compartido.

3\. \*\*P37 / P44 / P45\*\* — unificar constantes.

4\. \*\*P75 / P33\*\* — guardas de unicidad en índices.



\### Fase 4 — Deuda declarada (mantener como tal)

\- \*\*D1-D5\*\* del transfer doc (multiple\_openfigi\_hits hardcoded, target\_universe sin Q-CUR, validación CUSIP, BRK-B/MOG-A, EOL hygiene).

\- \*\*P30 / P61 / P63\*\* — asimetrías estructurales del resolver. Documentar, no arreglar sin dictamen.



\---



\## 11. Cierre



El módulo IAE está \*\*funcionalmente correcto en su núcleo\*\* (NIPC, delta\_shares, resolver de identidad) y \*\*respeta las prohibiciones duras verificables\*\*. Los hallazgos ALTA son de portabilidad, no de corrección de cálculo. Los candidatos a dictamen (P38, P60, P61, P70) son los únicos que pueden afectar a resultados publicados.



\*\*Ningún fix se ha aplicado.\*\* Este documento es un borrador interno pendiente de revisión por el responsable antes de envío al auditor externo.



\---



\*\*Fin del informe. Pendiente de validación del usuario antes de:\*\*



