# 79 - HOLDINGS-NORMALIZACION-EN-ORIGEN v1

**Pre-registro. 2026-10-31, sesion 22.**
**Contrato firmado antes de tocar codigo.**

## 1. Motivacion

Clase de bug: "el sistema descarga X pero analiza Y". Dos instancias
cerradas hoy mismo:

- O2 (1cb15d52): cap TOP_N_CANDIDATES desincronizado.
- BRK-NORMALIZACION-RESIDUAL (e04b53ff): BRK.B vs BRK-B en
  rs_internal y sector_concentration.

Ambos cerraron el sintoma, no la causa raiz. Hoy cada consumidor de
`holdings_df` normaliza por su cuenta. Un modulo nuevo que olvide
normalizar pierde BRK.B silenciosamente. `01c_LECCIONES` (bfba1763)
registro la leccion; este pre-registro define el fix estructural.

## 2. Diseño

**Invariante nuevo.** `holdings_df` sale de `load_stocks_data`
(`src/pipeline/stocks_data.py`) con DOS columnas:

- `ticker`: original (`BRK.B`, `BF.B`, `MOGA`). Para output/reporte.
- `ticker_norm`: normalizado (`BRK-B`, `BF-B`, `MOG-A`). Para lookup
  en parquet.

**Contrato.** Ningun consumidor de `holdings_df` normaliza por su
cuenta. Acceden a `ticker_norm` para lookup, `ticker` para output.

**Assert defensivo** en `load_stocks_data`:
`assert holdings_df['ticker_norm'].apply(normalize_yahoo_ticker).equals(holdings_df['ticker_norm'])`.
Idempotencia verificada: `YAHOO_TICKER_MAP` no tiene inversos.

## 3. Consumidores identificados (5)

Auditoria read-only 2026-10-31:

| # | Modulo | Linea | Lookup actual | Output actual | Test ancla |
|---|---|---|---|---|---|
| 1 | indicators/rs_internal.py | 55 | normalize(t) | `ticker` (BRK.B) | test_rs_internal.py:50 |
| 2 | indicators/sector_concentration.py | 59 | normalize(t) | `ticker` (BRK.B) | test_sector_concentration.py:122 |
| 3 | indicators/stock_metrics.py | 295 | normalize(t) | `_tn` (BRK-B) | test_stock_metrics.py:206-208 |
| 4 | indicators/wyckoff_holdings_report.py | 280 | normalize(t) | `ticker` (BRK.B) | test_wyckoff_holdings_report.py:246 |
| 5 | indicators/sector_breadth.py | 63 | normalize(t) | agregado | ninguno directo |

Consistencia vigente: 3 emiten BRK.B, 1 emite BRK-B, 1 agrega.
Cada uno anclado por test.

## 4. Ficheros a tocar (10)

**Revisado 2026-10-31 tras intento de ejecucion (rollback):**

El conteo inicial de 10 ficheros era incompleto. Al aplicar el patch
y correr pytest, 16 tests fallaron con `KeyError: 'ticker_norm'`:

- test_sector_concentration.py: 2 tests
- test_stock_metrics.py: 2 tests
- test_wyckoff_holdings_report.py: 10 tests
- test_sector_breadth.py: 3 tests

Causa: los tests construyen holdings_df sintetico con solo `ticker`.
La fase 1 identifico los tests ancla de BRK.B, pero no los tests
genericos que pasan holdings_df a los consumidores.

Conteo real pendiente: 20+ tests. Y hay que verificar si hay tests
en otros ficheros (test_pipeline_*.py, test_report_*.py) que
construyan holdings_df o llamen a los 5 consumidores indirectamente.

## 4bis. Alternativa B' (helper centralizado)

Dado el hallazgo anterior, hay dos opciones para sesion 23:

**B (diseño original):** holdings_df siempre trae ticker y ticker_norm.
Consumidores acceden a `row['ticker_norm']` directamente. Tests
construyen holdings_df con ambas columnas. Contrato explicito.

**B' (helper defensivo):** helper unico en src/utils.py:

    def get_ticker_norm(row):
        if 'ticker_norm' in row.index:
            return row['ticker_norm']
        return normalize_yahoo_ticker(row['ticker'])

Consumidores llaman al helper. Tests existentes siguen pasando (el
helper calcula si falta). load_stocks_data sigue anadiendo
ticker_norm (rendimiento + assert de idempotencia). El helper es el
fallback, no el camino principal.

**Ventajas B':** menos tests a tocar, menos riesgo de romper contratos
existentes en sesion 23. **Desventajas B':** mantiene el patron
"si falta, lo arreglo yo" que origino la deuda. La causa raiz solo
se cierra parcialmente (load_stocks_data normaliza, pero un modulo
nuevo que use el helper no falla si holdings_df llega sin ticker_norm).

**Decision:** B' si el conteo real de tests afectados supera 25.
B si es menor. Se decide en sesion 23 tras el grep exhaustivo:

    Get-ChildItem tests -Recurse -Include *.py | Select-String "holdings_df|holdings ="

## 4ter. Tests ancla identificados en fase 1 (incompletos)


- `src/pipeline/stocks_data.py` (+3 lineas + assert).
- 5 consumidores (sustituir `normalize_yahoo_ticker(t)` por
  `row['ticker_norm']`, retirar import si queda huerfano).
- 4 tests ancla (anadir `"ticker_norm"` al `holdings_df` sintetico;
  no relajar asserts — el test refleja el contrato, no lo esquiva).
- `tests/test_holdings_contract.py` (nuevo, centinela AST).

## 5. Test centinela

`tests/test_holdings_contract.py`, 2 tests:

1. `test_load_stocks_data_emite_ticker_y_ticker_norm`: tras llamar
   a `load_stocks_data` con holdings_df sintetico, verificar ambas
   columnas y que `ticker_norm == normalize(ticker)` por fila.
2. `test_consumidores_no_normalizan_por_su_cuenta`: AST sobre
   `indicators/*.py` y `src/report/*.py`. Detecta
   `normalize_yahoo_ticker(...)` aplicado a un ticker de holdings
   sin pasar por `ticker_norm`. Falla si aparece un consumidor nuevo
   que normalice por su cuenta.

## 6. Criterios de aceptacion (E2E obligatorio, METODO seccion 4)

Snapshot pre de `outputs/history`, `outputs/state`, `outputs/report`
en `outputs/audit/pre_holdings_norm_<stamp>/`.

`py run.py` real (~15 min). Snapshot post en
`outputs/audit/post_holdings_norm_<stamp>/`.

Comparacion contra baseline run local `20261031_003442` (HANDOFF
seccion 2.3):

1. gap ret20 vs rs = 0 en los 11 sectores.
2. XLF 34/34, XLV 26/26, XLU 22/22, XLY 22/22, XLI 50/50, XLK 39/39.
3. Reporte muestra `BRK.B` (no `BRK-B`) en Concentracion
   (sector_concentration preserva original).
4. `stock_metrics` sigue emitiendo `BRK-B` (su contrato, test
   test_stock_metrics.py:206-208).
5. CSVs historicos: solo append de la fecha del run, 0 reescritura
   retroactiva.
6. Exit=0. Validation Gate 11/11.

Rollback de `outputs/history` y `outputs/state` al cerrar
(`git checkout -- outputs/history outputs/state`). Snapshots
`pre_*`/`post_*` en `outputs/audit/` (gitignored).

## 7. Rollback si E2E falla

- `git checkout -- <ficheros>` para revertir codigo.
- `git checkout -- outputs/history outputs/state` para revertir
  outputs.
- Anotar en este documento la fase que fallo y por que.
- Sesion 23 empieza con el diagnostico hecho, no desde cero.

## 8. Fuera de alcance

- Otros normalizadores (CUSIP, provider symbols). Solo BRK.B y
  afines en holdings_df.
- Refactor de `normalize_yahoo_ticker`. Se usa tal cual.
- Cap MAX_CANDIDATES_PER_SECTOR (ya resuelto en O2).
- `src/pipeline/stocks_data.py` mas alla de la linea 72 (donde se
  carga holdings_df).

## 9. Decisiones y trazabilidad

- **B sobre A** (normalizar en origen sin preservar original): B
  respeta `e04b53ff` y el contrato del reporte (BRK.B visible).
- **B sobre C** (helper + test sin tocar holdings_df): C no cierra
  la causa raiz, solo el sintoma. B la cierra.
- **Tests ancla actualizados, no relajados.** Regla METODO seccion 7.

## 10. Estimacion

**Actualizado 2026-10-31 tras rollback:**

Estimacion original (~1h40) subestimada por alcance incompleto.
Sesion 23 debe empezar con grep exhaustivo de tests que construyen
holdings_df, luego decidir B vs B', luego ejecutar con presupuesto
de 3h minimo (no 2h).

**Leccion:** el pre-registro identifico los 5 consumidores de codigo
pero no los ~20 tests que los ejercitan. "Fase 1 read-only" se
detuvo demasiado pronto. En refactors de contrato, el inventario de
tests es tan critico como el de codigo.

**Rollback realizado:** intento de B en sesion 22 (0004caec
pre-registro + patch no commiteado). Estado restaurado via
`git checkout --` sobre 10 ficheros. pytest 32/32 verde.

~1h40 con E2E. Cabe en sesion 22 (2h disponibles). Si se excede,
parar sin commit, rollback, sesion 23.
