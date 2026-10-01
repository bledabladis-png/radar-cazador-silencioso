\# Expediente — Crecimiento del snapshot del catálogo PIT rompe el binding ticker ↔ catalog\_key



\*\*Documento para auditor externo. Autocontenido. No normativo.\*\*

\*\*Fecha:\*\* 2026-10-01.

\*\*HEAD al redactar:\*\* `037c7ed`.

\*\*Naturaleza:\*\* expediente de diagnóstico. Solicita dictamen sobre opciones de diseño antes de tocar código.



\---



\## 0. Resumen ejecutivo



El subsistema catalog PIT tiene un fallo de diseño latente que se ha materializado hoy. El vínculo entre `radar\_ticker` y `catalog\_key` es \*\*posicional por orden alfabético\*\*, no está declarado en ningún contrato, y no tiene test que lo verifique. El contrato A1 v4 (`A62BIS\_B1\_SUBFASE.md`, commit `bfb3cd7`, borrado en `7ba266d`) define `catalog\_key` como identidad inmutable y `K -> assigned\_entity\_id` como 1:1, pero \*\*no dice nada sobre la relación ticker↔key\*\*.



Consecuencias verificadas:



1\. `build\_catalog\_csvs.py::build\_membership` revienta con `IndexError` cuando el snapshot crece (255) y `catalog\_assignments.csv` no (242).

2\. Los 4 tests anclados a estado (`test\_assignments\_242\_filas`, `test\_membership\_242\_filas`, `test\_predecessor\_solo\_en\_filas\_cambiadas`, `test\_justification\_por\_fila`) rompen con cualquier migración que añada filas.

3\. `validate\_membership` en `catalog\_key.py` es \*\*código muerto sobre los datos reales\*\*: espera membership acumulativo, el CSV solo contiene la versión vigente.

4\. Race condition en CI: `Run tests` corre antes de `regenerate\_radar\_catalog`, así que el próximo `daily\_run` fallará la suite con el IndexError ya commiteado en `origin/main`.



\*\*No hay fix sin decisión de auditor.\*\* Cinco puntos requieren dictamen: extensión de A1 v4, política de bajas, modo membership (acumulativo vs per-version), materialización del binding histórico, y destino de `validate\_membership`.



\---



\## 1. Síntoma observado



\### 1.1. Estado actual del repo



\- `data/mappings/catalog\_snapshots/snapshot\_2026-10-01\_01.csv` — \*\*255 filas\*\* (31616 B), commiteado el 2026-10-01 20:37 CEST por el step `Regenerar catalogo radar (IAE)` del `daily\_run`.

\- `data/mappings/radar\_target\_catalog.csv` — 255 filas (vista del snapshot vigente).

\- `data/mappings/catalog\_assignments.csv` — \*\*242 filas\*\* (1 cabecera + 242).

\- `data/mappings/catalog\_membership.csv` — \*\*242 filas\*\*.

\- `catalog\_manifest.json` — entrada vigente `version\_id=2026-10-01\_01`, `valid\_from=2026-10-01`, `valid\_to=null`.



\### 1.2. Fallo reproducido



text



```

py -m pytest tests/ -q --tb=line

1 failed, 3012 passed, 2 skipped

FAILED tests/test\_build\_catalog\_csvs.py::test\_idempotencia

&#x20; - IndexError: single positional indexer is out-of-bounds

&#x20;   (build\_catalog\_csvs.py:195)

```



svgsvg



El `IndexError` ocurre dentro de `gen.main()`. El test no llega a sus `assert`. El test es correcto (verifica idempotencia); el problema es que `main()` no es idempotente cuando snapshot y assignments divergen.



\### 1.3. Punto exacto del fallo



`scripts/build\_catalog\_csvs.py:184-199` en `main()`:



python



```

prev\_ordered = prev\_df.sort\_values("radar\_ticker").reset\_index(drop=True)

prev\_cols = list(prev\_df.columns)

assign\_ordered = assignments.reset\_index(drop=True)

for i, (\_, prow) in enumerate(prev\_ordered.iterrows()):

&#x20;   key = assign\_ordered.iloc\[i]\["catalog\_key"]   # <-- L195: IndexError en i=242

&#x20;   prev\_uid\_by\_key\[key] = compute\_snapshot\_row\_uid(prow, prev\_cols)

```



svgsvg



`prev\_ordered` tiene 242 filas (snapshot previo), `assign\_ordered` tiene 242 filas. El bucle en sí no revienta. El IndexError está en la \*\*otra iteración posicional\*\*, la que sí tiene 255:



`scripts/build\_catalog\_csvs.py:108-119` en `build\_membership()`:



python



```

ordered = df.sort\_values("radar\_ticker").reset\_index(drop=True)   # 255 filas

assign\_ordered = assignments\_df.reset\_index(drop=True)             # 242 filas

cols = list(df.columns)



for i, (\_, row) in enumerate(ordered.iterrows()):                  # i = 0..254

&#x20;   key = assign\_ordered.iloc\[i]\["catalog\_key"]                    # <-- IndexError i=242

```



svgsvg



`build\_membership` se invoca con el snapshot vigente (255) y `assignments` (242). Al llegar a `i=242`, `assign\_ordered` no tiene fila 242.



\### 1.4. Un bug más grave latente



El IndexError es la manifestación \*benigna\*. Si `assignments` también tuviera 255 filas (por ejemplo porque en un run previo se regeneró con un snapshot intermedio), el alineamiento posicional sería \*\*silenciosamente incorrecto\*\* con tickers nuevos intercalados alfabéticamente.



Ejemplo concreto sobre el snapshot actual:



| \*\*Posición\*\* | \*\*Ticker (orden alfabético, 255)\*\*           | \*\*Key asignada por posición (242 existentes + 13 nuevos)\*\*          |

| :----------- | :------------------------------------------- | :------------------------------------------------------------------ |

| 0..9         | AAPL, ABBV, ABT, ACGL, ACN, ADBE, ...        | `radar\_20260921\_0001..0010` (correcto si no hay intercalados antes) |

| …            | …                                            | …                                                                   |

| n            | \*\*ANET\*\* (nuevo, intercala entre AMZN y AON) | \*\*desplaza\*\* todas las keys posteriores                             |

| n+1          | AON                                          | recibe key de AMZN                                                  |

| …            | …                                            | …                                                                   |



El binding se corrompe sin que ningún test lo detecte. `test\_membership\_catalog\_keys\_en\_assignments` sólo verifica que los \*sets\* coincidan, no que el mapeo ticker→key sea estable.



\### 1.5. Race condition CI (materializada)



`tests/test\_build\_catalog\_csvs.py` corre \*\*antes\*\* que `scripts/regenerate\_radar\_catalog.py` en el `daily\_run`:



text



```

.github/workflows/daily\_run.yml (orden de steps):

&#x20; Run tests

&#x20; Run pyflakes

&#x20; Run Macro Sectorial

&#x20; Regenerar catalogo radar (IAE)   <- aquí se genera snapshot 255

&#x20; ...

&#x20; Verify leader selection

```



svgsvg



Hoy:



1\. Run del 2026-10-01 (paso `Run tests`): suite verde (snapshot y assignments ambos en 242).

2\. Mismo run, step posterior `Regenerar catalogo radar`: genera snapshot 255 y lo commitea.

3\. \*\*Próximo run del 2026-10-02 (paso\*\* `Run tests`\*\*): la suite fallará con el IndexError ya commiteado.\*\*



No es teórico: el estado en `origin/main` (HEAD `037c7ed`) es rojo.



\---



\## 2. Diagnóstico raíz



\### 2.1. El contrato A1 v4 no define el binding ticker ↔ key



Recuperado de `bfb3cd7:docs/auditoria/iae/A62BIS\_B1\_SUBFASE.md` (texto completo en `git show bfb3cd7:docs/auditoria/iae/A62BIS\_B1\_SUBFASE.md`).



\*\*§2.1 Semantica normativa (literal):\*\*



text



```

catalog\_key = identidad administrativa estable.

K -> assigned\_entity\_id es 1:1 para toda la vida de K.

K NO se reasigna jamas.

Cambio de entidad -> nueva key: K\_old -> A, K\_new -> B

```



svgsvg



\*\*§2.2 Formato (literal):\*\*



text



```

catalog\_key := "radar\_<YYYYMMDD\_alta>\_<NNNN>"

```



svgsvg



\*\*§2.3\*\* `catalog\_assignments.csv` \*\*(literal):\*\*



text



```

catalog\_key | assigned\_entity\_id | valid\_from | valid\_to | source | reason

EXACTAMENTE 1 fila por key.

```



svgsvg



Lo que A1 v4 \*\*no dice\*\*:



\- No dice que `catalog\_key` esté biyectada con `radar\_ticker`.

\- No dice que la numeración `<NNNN>` sea por ticker o por posición.

\- No dice qué ocurre cuando el universo crece.

\- No dice qué ocurre cuando el universo se reduce (ticker desaparecido).



El comentario en `build\_catalog\_csvs.py:7-12` reduce A1 al formato, sin decir nada sobre el binding:



python



```

"""Contrato (A62BIS\_B1\_SUBFASE.md v4 secciones 2-3):

&#x20; catalog\_key := "radar\_<YYYYMMDD\_alta>\_<NNNN>"    (0001-based, alfabetico)

&#x20; ...

```



svgsvg



`alfabetico` es una \*\*convención del generador\*\*, no un contrato.



\### 2.2. El binding es posicional, silente, y sin invariante



`build\_assignments` (`build\_catalog\_csvs.py:96-109`):



python



```

def build\_assignments(df, alta\_date):

&#x20;   """242 filas, orden alfabetico por radar\_ticker."""

&#x20;   ordered = df.sort\_values("radar\_ticker").reset\_index(drop=True)

&#x20;   rows = \[]

&#x20;   for i, \_ in ordered.iterrows():

&#x20;       suffix = \_numeric\_suffix(i)                 # i+1, zero-padded 4

&#x20;       rows.append({

&#x20;           "catalog\_key": CATALOG\_KEY\_PREFIX + "\_" + alta\_date + "\_" + suffix,

&#x20;           ...

```



svgsvg



El sufijo se calcula como `i + 1` sobre el \*\*índice de fila tras sort alfabético\*\*. La relación `ticker ↔ key` no se materializa en ninguna columna; sólo existe implícitamente en el orden.



`build\_membership` (`build\_catalog\_csvs.py:112-155`) replica el mismo patrón: recorre `snapshot.sort\_values("radar\_ticker")` y para cada fila obtiene `assign\_ordered.iloc\[i]\["catalog\_key"]`. Asume `len(assignments) == len(snapshot)` y que ambos están ordenados igual.



\*\*No hay ningún test que verifique que\*\* `ticker\_by\_key` \*\*es estable entre ejecuciones.\*\* El único test cercano es `test\_membership\_catalog\_keys\_en\_assignments`, que compara conjuntos, no mapeos.



\### 2.3. Los 13 nuevos confirmados; cero bajas



Comparación `snapshot\_20260922\_01.csv` (242) vs `snapshot\_2026-10-01\_01.csv` (255):



text



```

altas (en vigente, no en prev): ANET, EL, EMR, GRMN, HON, HOOD, HST, MRVL,

&#x20;                               PPL, RDDT, TGTX, TXG, USB

bajas: (ninguna)

```



svgsvg



Los 13 son empresas cotizadas USA con ticker estable. Ninguna es rebranding/rename de un ticker existente en el snapshot 242 (verificado contra el catálogo `radar\_target\_catalog.csv`: los 13 aparecen con `status=OK`, `market\_sector=Equity`, `source=openfigi:/v3/mapping`).



El caso simétrico (baja) no se ha ejercitado nunca. `regenerate\_radar\_catalog.\_build\_updated\_catalog` hace `pd.concat(\[catalog\_df, nuevas\_filas])` — \*\*mantiene todo\*\* `catalog\_df`, no filtra por radar actual. La política de bajas no existe formalmente.



\### 2.4. `validate\_membership` es código muerto sobre los datos reales



`src/institutional\_accumulation/identity/catalog\_key.py:300-345` (docstring):



python



```

def validate\_membership(membership\_df, assignments\_df, manifest, ...):

&#x20;   """..."""

```



svgsvg



El propio docstring del módulo (L11-30) lo declara:



text



```

ESTADO DE INTEGRACION (2026-09-29):

&#x20; - Productivos: check\_schema + compute\_snapshot\_row\_uid.

&#x20; - NO integrados: validate\_assignment, validate\_membership, y los

&#x20;   loaders asociados. Solo invocados en tests. Ademas,

&#x20;   validate\_membership espera un membership\_df acumulativo (todas

&#x20;   las versiones historicas) para resolver predecessor\_row\_uid

&#x20;   contra versiones previas; el CSV real (catalog\_membership.csv)

&#x20;   contiene solo la version vigente, producido asi por

&#x20;   build\_catalog\_csvs.py. La funcion no es usable sobre los datos

&#x20;   reales tal como estan.

```



svgsvg



A1 v4 §3.3 diseña `predecessor\_row\_uid` como \*\*cadena histórica explícita entre versiones\*\*. El CSV actual no acumula versiones: `build\_catalog\_csvs.py:212-216` hace `membership.to\_csv(MEMBERSHIP\_PATH, ...)` sin append ni merge. Cada ejecución \*\*sobrescribe\*\* el CSV con la versión vigente.



Contradicción entre contrato v4 (§3.3: acumulativo) y código (per-version).



\---



\## 3. Contratos afectados



| \*\*Documento\*\*           | \*\*Sección\*\* | \*\*Qué dice\*\*                                             | \*\*Cómo lo rompe el bug\*\*                                                                      |

| :---------------------- | :---------- | :------------------------------------------------------- | :-------------------------------------------------------------------------------------------- |

| A62BIS\_B1\_SUBFASE.md v4 | §2.1        | K inmutable, K↔entity 1:1                                | No dice nada de ticker. El bug no viola A1 literalmente, pero tampoco está cubierto.          |

| A62BIS\_B1\_SUBFASE.md v4 | §2.3        | EXACTAMENTE 1 fila por key                               | Se mantiene con opción (A): 242 + 13 = 255 filas únicas.                                      |

| A62BIS\_B1\_SUBFASE.md v4 | §3.2        | `snapshot\_row\_uid := sha256\_hex(fila\_canonica)`          | Estable, no afectado.                                                                         |

| A62BIS\_B1\_SUBFASE.md v4 | §3.3        | `predecessor\_row\_uid`: cadena histórica                  | \*\*Roto\*\*: el CSV no acumula versiones.                                                        |

| A62BIS\_B1\_SUBFASE.md v4 | §3.4        | Cobertura temporal completa (B1-NEW-6)                   | \*\*Potencialmente roto\*\* si algún assignment nuevo tuviera `valid\_to` no-null (hoy no ocurre). |

| A62BIS\_B1\_SUBFASE.md v4 | §8.1        | `B1\_REQUIRED\_COLUMNS = {radar\_ticker, share\_class\_figi}` | Intacto: la columna `radar\_ticker` sigue en el snapshot.                                      |

| A62BIS\_B1\_SUBFASE.md v4 | §1.3        | "NO modificar contratos normativos"                      | La opción (A) del usuario requiere extensión de §2.3 con columna nueva.                       |



\---



\## 4. Consumidores y tests afectados



\### 4.1. Consumidores de los CSV (grep verificado)



Ficheros que leen o mencionan `catalog\_assignments` / `catalog\_membership`:



| \*\*Fichero\*\*                                                      | \*\*Rol\*\*               | \*\*Impacto\*\*                                                    |

| :--------------------------------------------------------------- | :-------------------- | :------------------------------------------------------------- |

| `scripts/build\_catalog\_csvs.py`                                  | Productor             | \*\*Roto\*\* (IndexError)                                          |

| `src/institutional\_accumulation/pipeline\_contractual.py:147-149` | Consumidor            | Lee `mem` y `asg` con `pd.read\_csv`, los pasa a `build\_target` |

| `src/institutional\_accumulation/identity/catalog\_key.py`         | Contrato + validators | Define columnas vía `ASSIGNMENT\_COLUMNS`, `MEMBERSHIP\_COLUMNS` |

| `src/institutional\_accumulation/identity/target\_builder.py`      | Consumidor real       | `build\_target(snapshot, membership, assignments, ...)`         |

| `scripts/iae\_contractual\_coverage.py`                            | Consumidor            | Carga los CSV                                                  |

| `tests/test\_build\_catalog\_csvs.py`                               | Test                  | \*\*4 tests anclados rompen con opción (A)\*\*                     |

| `tests/test\_catalog\_key.py`                                      | Test de contrato      | 9 tests A1-a..i, no afectados                                  |

| `tests/test\_iae\_gap\_coverage.py`                                 | Test                  | 2 tests de loaders, no afectados                               |

| `docs/auditoria/iae/evidence/...`                                | Probes históricos     | No productivos                                                 |



Ficheros que mencionan `catalog\_manifest`:



| \*\*Fichero\*\*                                                           | \*\*Rol\*\*                               |

| :-------------------------------------------------------------------- | :------------------------------------ |

| `scripts/build\_catalog\_csvs.py`                                       | Lee `valid\_from` del snapshot vigente |

| `scripts/regenerate\_radar\_catalog.py`                                 | Escribe manifest                      |

| `src/institutional\_accumulation/catalog\_pit.py`                       | B2-PIT                                |

| `tests/test\_catalog\_pit.py`, `tests/test\_regenerate\_radar\_catalog.py` | Tests                                 |

| `.github/workflows/daily\_run.yml`                                     | Orquesta                              |



\### 4.2. Cómo lee el consumidor real (`target\_builder.py`)



`build\_target` (L97-153) itera el snapshot, calcula `row\_uid` por fila, y consulta `mem\_by\_uid\[uid]` para obtener `catalog\_key`. \*\*No usa posición.\*\* El bug no rompe `build\_target` \*por sí mismo\*; rompe porque `membership` no cubrirá los 13 nuevos `row\_uid` si `build\_membership` no los materializa. En la práctica:



\- Hoy: `build\_membership` revienta antes de escribir.

\- Tras un parche que sólo evite el crash: `build\_target` levantaría `BuildTargetError("membership no cubre row\_uid=...")` con los 13 nuevos.



\### 4.3. Tests anclados al estado actual (bloquean la migración)



`tests/test\_build\_catalog\_csvs.py`:



| \*\*Test\*\*                                   | \*\*Qué afirma\*\*                                                              | \*\*Qué pasa con opción (A)\*\*                                                                            |

| :----------------------------------------- | :-------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------- |

| `test\_assignments\_242\_filas`               | `len(df) == 242`                                                            | Rompe (255)                                                                                            |

| `test\_membership\_242\_filas`                | `len(df) == 242`                                                            | Rompe (255)                                                                                            |

| `test\_predecessor\_solo\_en\_filas\_cambiadas` | `n\_pred == 2` + keys literales `radar\_20260921\_0029`, `radar\_20260921\_0145` | Rompe (2 + N nuevos con predecessor, o 2 intactos si los 13 no tienen predecessor; depende del diseño) |

| `test\_justification\_por\_fila`              | `n\_init == 240`                                                             | Rompe (240 + 13 nuevos con otro justification)                                                         |

| `test\_idempotencia`                        | Snapshot estable antes/después de `main()`                                  | Rompe hoy (IndexError)                                                                                 |



Estos 4 tests hay que reescribirlos como \*\*invariantes\*\*:



\- `assignments` tiene 1 fila por `catalog\_key` único.

\- `membership` tiene 1 fila por `(version\_id, catalog\_key)` único.

\- Sets de `catalog\_key` coinciden entre ambos.

\- `predecessor\_row\_uid` es no-vacío sii el `snapshot\_row\_uid` difiere de la versión anterior para esa key.

\- Cada `(version\_id, catalog\_key)` del CSV vigente corresponde a un `snapshot\_row\_uid` presente en el snapshot vigente.



\### 4.4. Verificación de sha256



`grep sha256 tests/ | grep -i catalog\_assign|membership` → sin resultados. No hay golden de los CSV. \*\*Cambiar el esquema no rompe ningún test por hash.\*\* El manifest sólo guarda sha256 del CSV de cada snapshot (`snapshot\_YYYYMMDD\_NN.sha256`), no de `assignments` ni de `membership`.



\---



\## 5. Estado de partida del snapshot



\### 5.1. Snapshots disponibles



text



```

snapshot\_20260921\_01.csv    (242 filas, 29849 B)  24/09/2026 04:03

snapshot\_20260922\_01.csv    (242 filas, 29986 B)  24/09/2026 04:04

snapshot\_2026-10-01\_01.csv  (255 filas, 31616 B)  01/10/2026 20:37

```



svgsvg



\### 5.2. Correlación con assignments actuales



El `assignments` actual corresponde al snapshot `20260922\_01`:



\- Test `test\_predecessor\_solo\_en\_filas\_cambiadas` espera `n\_pred == 2` con keys `radar\_20260921\_0029` y `radar\_20260921\_0145`.

\- Esas dos keys corresponden a las dos filas cuya `snapshot\_row\_uid` cambió entre `20260921\_01` y `20260922\_01` (BRK-B y MOG-A, resueltas vía OpenFIGI 2026-09-22).



`git log -- data/mappings/catalog\_assignments.csv` → \*\*1 solo commit\*\*: `0a30212 feat(iae): B1.0 - generador CSV catalogo + assignments + membership`. Congelado desde entonces.

`git log -- data/mappings/catalog\_membership.csv` → 2 commits: `0a30212` y `7f314ef` (B-03, cierre BRK-B + MOG-A).



\### 5.3. Reconstrucción del binding histórico (hipótesis verificable)



El binding de los 242 keys existentes es reconstruible:



text



```

for i, ticker in enumerate(sorted(snapshot\_20260922\_01\["radar\_ticker"])):

&#x20;   key = f"radar\_20260921\_{i+1:04d}"

```



svgsvg



Verificación propuesta: para cada key de `assignments`, reconstruir el `radar\_ticker` vía esta fórmula sobre `snapshot\_20260922\_01.csv` y comprobar que los `snapshot\_row\_uid` correspondientes coinciden con los que hay en `catalog\_membership.csv` (mismo `version\_id=20260922\_01`).



Si la verificación es positiva, la migración puede materializar la columna `radar\_ticker` en `assignments` a partir de esta reconstrucción, con `snapshot pre/post` de los CSV.



\---



\## 6. Opciones de diseño



\### Opción (A) — append-only con columna `radar\_ticker` (recomendada por el usuario)



\*\*Descripción:\*\*



1\. Añadir columna `radar\_ticker` a `catalog\_assignments.csv` como fuente de verdad explícita.

2\. Preservar los 242 keys existentes.

3\. Añadir 13 filas nuevas al final: `radar\_20261001\_0001..0013`, con `alta\_date=20261001`, `valid\_from=20261001`, `source=radar\_target\_catalog`, `reason=snapshot expansion 2026-10-01`.

4\. `build\_assignments` deja de ser determinista por posición: se vuelve un merge que (i) detecta tickers presentes en `assignments` y los preserva con su key original, (ii) detecta tickers nuevos y les asigna la siguiente key, (iii) si hay tickers ausentes del snapshot, los deja con su key y `valid\_to=NULL` (append-only puro).



\*\*Ventajas:\*\*



\- Cumple A1 §2.1 (K no se reasigna jamás).

\- Binding ticker↔key explícito en el CSV.

\- Migración auditable: snapshot pre/post.

\- Idempotente: si el snapshot no cambia, `main()` no escribe nada.



\*\*Coste:\*\*



\- \*\*Requiere extensión de A1 v4 §2.3\*\*: añade una columna. §1.3 dice "NO modificar contratos normativos". Hay que aprobar A62BIS\_B1\_SUBFASE v5 o un amendment, con dictamen.

\- Los 4 tests anclados deben reescribirse como invariantes.

\- `build\_membership` debe alinear por `radar\_ticker` (o por `snapshot\_row\_uid` directamente), no por posición.



\*\*Sub-decisión A1\*\*: ¿bajada de ticker cuenta como baja de la key (retired con `valid\_to`) o se mantiene viva? A1 §2.1 sólo cubre cambio de entidad, no cambio de universo. La política debe escribirse.



\### Opción (B) — fichero de binding aparte



\*\*Descripción:\*\* mantener `catalog\_assignments.csv` intacto (schema A1 actual) y crear `catalog\_ticker\_binding.csv`:



text



```

radar\_ticker | catalog\_key | bound\_at

```



svgsvg



\*\*Ventajas:\*\*



\- No toca A1.

\- Un cuarto artefacto, pero aislado.



\*\*Coste:\*\*



\- Introduce una fuente de verdad paralela. Cualquier consumidor debe leer \*los dos\* ficheros para resolver el binding.

\- El riesgo de desincronización entre ambos es real (mismo problema que hoy con assignments vs membership).

\- A1 no cubre este fichero; hay que formalizarlo igual.



\*\*Veredicto:\*\* no reduce el trabajo documental (sigue requiriendo contrato) y añade complejidad operativa. Descartable.



\### Opción (C) — keys estables por hash de ticker



\*\*Descripción:\*\* `catalog\_key := "radar\_" + hash(ticker)\[:12]` (o similar). Determinista, sin posición.



\*\*Ventajas:\*\*



\- Independiente de orden, inmune a crecimiento/reducción.

\- No requiere migración de 242 keys si hasheamos el ticker y coinciden.



\*\*Coste:\*\*



\- \*\*Rompe A1 §2.2\*\* (formato `radar\_<YYYYMMDD\_alta>\_<NNNN>`).

\- Pierde la información de `alta\_date` en la propia key.

\- Requiere migración total de los 242 keys existentes, con snapshot pre/post y re-validación de todos los predecessors.

\- No aporta nada que la opción (A) no aporte ya, y destruye el formato firmado en A1.



\*\*Veredicto:\*\* descartable.



\### Opción (D) — membership acumulativo + binding por `snapshot\_row\_uid`



\*\*Descripción:\*\* sin columna `radar\_ticker`. El binding se resuelve porque `membership` acumula todas las versiones y cada versión liga `snapshot\_row\_uid` ↔ `catalog\_key`. El binding ticker↔key se reconstruye desde el snapshot de cada versión: `ticker = snapshot\[uid].radar\_ticker`.



\*\*Ventajas:\*\*



\- Cero cambios de schema.

\- Alinea con A1 §3.3 (membership acumulativo, predecessors verificables).

\- Activa `validate\_membership`.



\*\*Coste:\*\*



\- `membership` acumula 242 + 255 = 497 filas tras la migración. Es lineal con el tiempo.

\- El binding ticker↔key no es consultable en un solo CSV; requiere cargar el snapshot vigente.

\- La opción (A) ya resuelve esto si se combina con membership acumulativo.

\- \*\*No exime de extender A1\*\*, porque el problema raíz (el `build\_membership` posicional) sigue ahí; sólo cambia \*cómo\* se resuelve el binding en lectura.



\*\*Veredicto:\*\* complementaria a (A), no alternativa.



\---



\## 7. Recomendación (a validar por el auditor)



\*\*Combinación (A) + membership acumulativo.\*\*



1\. Extender A1 v4 con un amendment (v5) que:

&#x20;  - Añade columna `radar\_ticker` a `catalog\_assignments.csv`.

&#x20;  - Declara la biyección `radar\_ticker ↔ catalog\_key` como invariante verificable.

&#x20;  - Declara la política de bajas: ticker desaparecido ⇒ su key pasa a `valid\_to=<fecha\_snapshot>`; si reaparece, se le asigna \*\*nueva key\*\* (no se reactiva; coherente con A1 §2.1 "cambio de entidad → nueva key").

&#x20;  - Declara membership como \*\*acumulativo\*\* (una fila por `(version\_id, catalog\_key)` histórica), alineado con §3.3.

2\. Migración de los 242 keys existentes:

&#x20;  - Reconstruir el binding por la fórmula verificable del §5.3.

&#x20;  - Materializar la columna `radar\_ticker` en `assignments` con snapshot pre/post.

&#x20;  - Migrar `membership` al modo acumulativo: conservar la versión `20260922\_01` (242 filas) y añadir `2026-10-01\_01` (255 filas).

3\. Alta de los 13 nuevos:

&#x20;  - `catalog\_key = radar\_20261001\_0001..0013`.

&#x20;  - `valid\_from = 20261001`, `valid\_to = null`, `source = radar\_target\_catalog`, `reason = snapshot expansion 2026-10-01`.

4\. Reescribir `build\_assignments` y `build\_membership` como operaciones \*\*no posicionales\*\*:

&#x20;  - `build\_assignments`: merge por `radar\_ticker`. Si el ticker existe, preserva su key. Si no, asigna la siguiente. Nunca reordena.

&#x20;  - `build\_membership`: itera el snapshot y consulta `assignments\[radar\_ticker → catalog\_key]`. Si un `row\_uid` ya está en membership de una versión previa con la misma key, reusa (append idempotente); si no, añade fila.

5\. Reescribir los 4 tests anclados como invariantes (ver §4.3).

6\. Añadir tests nuevos:

&#x20;  - `test\_binding\_ticker\_key\_estable`: reordenar filas del snapshot no cambia el binding.

&#x20;  - `test\_snapshot\_crece\_sin\_reasignar`: simular snapshot 242 → 255 y verificar que los 242 keys existentes preservan su ticker.

&#x20;  - `test\_membership\_acumulativo`: 2 versiones → 497 filas, sin duplicados por `(version\_id, catalog\_key)`.

&#x20;  - `test\_baja\_de\_ticker\_marca\_valid\_to`: simular ticker desaparecido.

7\. Activar `validate\_membership` como PASO 0 del flujo B1 §9.



\---



\## 8. Contención inmediata (opcional, sin esperar dictamen)



El repo está rojo \*\*ahora mismo\*\* (`HEAD 037c7ed` en `origin/main`). Sin dictamen no hay fix limpio. Contención honesta:



\*\*Contención (h2) — revertir el snapshot vigente a 242.\*\*



\- Snapshot pre: `data/mappings/catalog\_manifest.json` + `data/mappings/radar\_target\_catalog.csv` + `data/mappings/catalog\_snapshots/snapshot\_2026-10-01\_01.{csv,sha256}`.

\- Acción: en `catalog\_manifest.json`, restaurar `valid\_to=null` a la entrada `20260922\_01` y eliminar la entrada `2026-10-01\_01` (o marcarla como `valid\_to=<fecha>`). En `radar\_target\_catalog.csv`, restaurar contenido de `snapshot\_20260922\_01.csv`.

\- Snapshot post: sha256 de los 3 ficheros antes y después.

\- Efecto: `build\_membership` vuelve a operar con 242 vs 242. CI verde.

\- Coste: los 13 nuevos vuelven a quedar "pendientes". El siguiente `regenerate\_radar\_catalog` los detectará de nuevo y volverá a generar snapshot 255. Habrá que decidir si desactivar temporalmente el step hasta que la migración (A) esté aprobada.

\- Riesgo: toca datos históricos. Requiere snapshot pre/post documentado.



\*\*Recomendación:\*\* no aplicar contención hasta tener dictamen, salvo que la ventana sin dictamen se extienda más allá de 24h. Mientras tanto, documentar el estado rojo en el corpus para que quede trazable.



\---



\## 9. Preguntas al auditor



1\. \*\*Extensión de A1 v4.\*\* ¿Se aprueba un amendment v5 que añada `radar\_ticker` a `catalog\_assignments.csv` como fuente de verdad explícita y que declare la biyección `radar\_ticker ↔ catalog\_key` como invariante? ¿O se prefiere un cuarto artefacto (opción B) con su propio contrato?

2\. \*\*Política de bajas.\*\* ¿Se aprueba: ticker desaparecido ⇒ key marcada con `valid\_to=<fecha>`; reaparición ⇒ nueva key (no reactivación)? ¿O se prefiere reactivación con la key original? §2.1 actual ("cambio de entidad → nueva key") sugiere la primera, pero no es explícito para bajas/altas de universo.

3\. \*\*Membership acumulativo vs per-version.\*\* A1 §3.3 asume acumulativo; código hace per-version. ¿Se aprueba migrar `catalog\_membership.csv` a acumulativo y activar `validate\_membership` como PASO 0 del flujo B1 §9? ¿O se aprueba el modo per-version y se retira `validate\_membership` formalmente?

4\. \*\*Materialización del binding histórico.\*\* ¿Se aprueba la reconstrucción del binding de los 242 keys existentes vía `sorted(snapshot\_20260922\_01\["radar\_ticker"])` indexado, con verificación cruzada contra `catalog\_membership.csv` (uid por key)? ¿O se prefiere otro método?

5\. \*\*Contención CI.\*\* ¿Se aprueba (h2) revertir el snapshot vigente a 242 mientras el dictamen llega? ¿O se prefiere dejar CI rojo y priorizar el fix completo?

6\. \*\*Alcance del fix.\*\* ¿El fix debe limitarse a la migración (A) + tests, o incluir también activar `validate\_membership` y reescribir el flujo B1 §9 PASO 0? Es decir: ¿el fix cierra deuda de hoy (crecimiento del snapshot) o también deuda arrastrada (membership no acumulativo, validator muerto)?

7\. \*\*Trazabilidad.\*\* ¿Se aprueba que la migración quede registrada en `04\_HISTORICO.md` como decisión de arquitectura (con referencias al dictamen v5), o se prefiere un expediente específico en `docs/auditoria/iae/`?



\---



\*\*Fin del expediente.\*\*

