# S-03-deep - barrido de docstrings desalineados (dict con keys)

Barrido AST sobre todo el repo para detectar funciones cuyo docstring
declara keys de un return dict que no coinciden con el return real.

- **Auditoria:** 2026-10-03
- **HEAD inicial:** a8e3736
- **Alcance:** todo el repo (excluye tests y __pycache__).
- **Metodo:** AST + parser de "Returns: dict con keys:".
- **Dictamen:** CERRADO. 2 ficheros corregidos.

## 1. Metodo

Script `ast` que:
1. Recorre FunctionDef/AsyncFunctionDef.
2. Extrae el bloque "Returns: dict con keys:" del docstring,
   tokenizando por comas + parentesis + dos puntos.
3. Extrae las keys del return{...} literal (ast.Dict).
4. Compara los dos sets.

Solo detecta returns con dict literal (no variables). Aplica a
modulos con docstring "dict con keys". Es un subconjunto de las
funciones del repo, pero captura exactamente el patron buscado.

## 2. Hallazgos corregidos

2 casos residuales tras la auditoria manual de S-03-deep:

| Fichero | Funcion | Falta en docstring | Commit |
|---|---|---|---|
| `src/pipeline/breadth_metrics.py` | compute_breadth_metrics | sector_breadth_stale_reason | (este commit) |
| `src/pipeline/iae_section.py` | compute_iae_section | catalog_coverage_warning | (este commit) |

Mismo patron que los 3 corregidos durante la auditoria manual:
- LD-1 (`leaders.py`): holiday_mode -> df_stocks_effective_meta.
- II-1 (`indices_intl.py`): index_data sobra.
- DL-1 (`data_load.py`): temporal_meta falta.

Total del patron: **5 ficheros corregidos**.

## 3. Verificacion

Tras el commit, el script AST se re-ejecuta y devuelve
"OK: sin desalineaciones detectadas".

## 4. Leccion de metodo

Los 2 casos residuales se escaparon en la auditoria manual de sus
ficheros (leidos linea a linea). El barrido AST los detecta de
inmediato. La leccion: la verificacion mecanica complementa a la
manual; ninguna de las dos es suficiente por si sola.

Referencia cruzada con 01_METODO.md seccion 7 (metodo de
auditoria). Candidato a documentar como "auditoria mecanica" en
proximos barridos.

## 5. Regla de estilo derivada

**El bloque "Returns: dict con keys:" debe coincidir exactamente
con las keys del return real.** Candidato a test contractual (no
aplicado en este commit).

## 6. Dictamen

CERRADO. 2 ficheros, 2 docstrings. Commit del fix + este audit doc.

## 7. Proximo

Candidatos siguientes:
- Caracterizacion de `macro_regime.py` (12 ramas sin test directo).
- `src/temporal_contracts/` (11 ficheros, A1 CERRADO).
- `src/institutional_accumulation/` (36 ficheros, B/6 CERRADOS).
- Test contractual del patron "dict con keys" (AST) para evitar
  regresiones.
