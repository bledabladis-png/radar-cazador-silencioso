# 66 - Regeneracion historica de sector_wyckoff_distribution (Fase 3)

**Fecha:** 2026-10-25
**Commit codigo/datos:** ver git log (Fase 3)
**Estado:** CERRADA. Fase 3 de 6.

## 1. Motivacion

`sector_wyckoff_distribution.csv` arrastraba un historico con dos
esquemas mezclados (Fase 1 documento el incidente del commit
`415487ff`). El `.v1.csv` resultante (342 filas) tiene 330 filas v1 sin
FORMING/CONFLICT y 11 filas v2 con esas columnas. No comparable en el
tiempo.

Fase 1 añadio schema versioning. Fase 4 alineo el universo. Fase 3
regenera el historico completo con el codigo actual: una sola pasada de
`classify_phase` por ticker, universo coherente con el contrato doc 63
§2.6 (top-20, >=200 barras), 1086 fechas.

## 2. Script

`scripts/regenerate_wyckoff_history.py`. Paralelizado por ticker
(`ProcessPoolExecutor`, 12 workers). Checkpoint cada 25 tickers en
`_regen_checkpoint.json` (reanudable). Validacion obligatoria antes de
escribir: reproduce el `.csv` activo en la ultima fecha.

Comando: `py -u scripts/regenerate_wyckoff_history.py --full`
Duracion: 49.8 min (234.197 invocaciones classify_phase_at).

## 3. Resultado

- 11946 filas (1086 fechas x 11 sectores).
- Rango: 2022-08-11 -> 2026-10-23.
- n_total variable por sector: 18-20. Depende de que tickers del top-20
  tengan >=200 barras en cada fecha (WBD en XLC con 1 barra, alguno mas).
- n_valid_wyckoff = n_total en todas las filas (el script ya filtra
  INSUFFICIENT_DATA).
- Validacion contra .csv activo (2026-10-23): 11 filas, 20 columnas
  identicas.

## 4. Hallazgo critico: structure_positive = 0/11946

`structure_positive = (pct_accumulation + pct_markup) >= 50` da **False
en las 11946 filas**. 1086 fechas x 11 sectores, siempre False.

Causa: `pct_range` domina la distribucion. Media por sector en todo el
historico:

| sector | mean pct_range | min | max |
|---|---|---|---|
| XLB | 86.7 | 70.0 | 100.0 |
| XLC | 59.4 | 26.3 | 100.0 |
| XLE | 75.9 | 35.0 | 100.0 |
| XLF | 84.5 | 57.9 | 100.0 |
| XLI | 80.9 | 40.0 | 100.0 |
| XLK | 47.4 | 21.1 | 100.0 |
| XLP | 86.7 | 70.0 | 100.0 |
| XLRE | 86.9 | 75.0 | 100.0 |
| XLU | 91.6 | 75.0 | 100.0 |
| XLV | 87.1 | 60.0 | 100.0 |
| XLY | 88.5 | 65.0 | 100.0 |

Con `pct_range` entre 47% y 92% de media, el resto (acc + acc_forming +
markup + dist + dist_forming + markdown) se reparte sobre 8-53%. La
probabilidad de que `(pct_acc + pct_markup) >= 50` con ese reparto es
practicamente nula. **No es un problema de calibracion de umbral: la
metrica no mide lo que dice medir.**

`SECTOR-REGIME-MATRIX-STRUCTURE-POSITIVE` (OPEN) queda confirmado
empiricamente como roto. Fase 5 requiere rediseño, no ajustar el 50 a 30
o a 20 (cualquier umbral con este reparto sera trivialmente False o
trivialmente True segun la cola que se mire).

## 5. Cambios operativos

- `.csv` activo: 11946 filas (antes 11).
- `.v1.csv` (342 filas v1+v2 mix): backup no-comparable, gitignored.
- `.csv.v1_mixed` (11 filas del run de Fase 4, ya en git): eliminado.
- `.schema` sigue en 2.
- Pipeline no cambia: lee `.csv`, escribe append con `append_dedup`. El
  schema_version=2 sigue vigente.

## 6. Proximo (Fase 5)

Con 1086 fechas x 11 sectores coherentes:

- Medir si otras metricas derivadas discriminan (pct_markup solo,
  pct_acc + acc_forming, ratio dist/acc, etc.).
- Si ninguna metrica binaria funciona, rediseñar como score continuo
  sobre base real (magnitud de fase + confianza).
- Pre-registro obligatorio antes de fijar cualquier umbral.

Sin pre-registro no se toca `structure_positive`. Queda documentado como
deuda en REGISTRO.
