# 63 - Plan del sistema Wyckoff por holdings

**Fecha:** 2026-10-24
**Estado:** DISENO. Sin codigo. Aprobado.
**Sustituye a:** el sistema de lideres por sector basado en fase ETF (retirado).

## 1. Motivacion

El sistema previo aplicaba el detector Wyckoff al ETF sectorial y usaba la
fase resultante como gate para seleccionar los top-5 lideres por sector.
Verificado esta sesion: sobre 11 ETFs sectoriales, el orquestador devuelve
RANGE en 9. El ETF es un agregado suavizado; la senal de
acumulacion/distribucion no es detectable con firmeza.

Decision: la deteccion se aplica directamente a los tickers que componen
cada sector o indice.

## 2. Lo que se retira

**Modulos completos:**

- indicators/slpm_v12.py, state_machine.py, state_transition.py
- indicators/stock_leader.py, index_leaders.py, index_phase.py
- indicators/leader_representativeness.py, sector_leader_divergence.py
- src/pipeline/leaders.py, src/pipeline/indices_intl.py

**Piezas dentro de modulos supervivientes:**

- sector_regime.py: llamada a classify_wyckoff_phase sobre el ETF y 4a
  columna del ranking.
- wyckoff_v1.py: classify_wyckoff_phase, classify_wyckoff_phase_meta,
  _is_distribution_candidate, _compute_precedent, _has_sufficient_data.
- src/pipeline/engines.py: _forzar_lideres_slpm.
- src/report_generator.py: render_acciones_seleccionadas,
  render_indices_internacionales, secciones SLPM.
- config/weights.py: SLPM_WEIGHTS.
- config/settings.py: SLPM_EXPECTED_LEADERS.

**CSVs detenidos** (historicos no se borran):

- analisis_lideres.csv, analisis_lideres_internacionales.csv
- leader_representativeness.csv, sector_leader_divergence.csv
- Columna wyckoff_phase de sector_rankings.csv.

## 3. Lo que se construye

**Un modulo:** indicators/wyckoff_holdings_report.py

    def compute_wyckoff_holdings_report(
        df_stocks,           # parquet con OHLCV (stock_prices.parquet)
        usa_sectors_df,      # etf_holdings.csv
        usa_indices_df,      # index_holdings.csv filtrado a SPY/QQQ/DIA
        eu_indices_df,       # index_holdings.csv filtrado a FEZ/DAXEX/ISF.L/LYXI
    ) -> dict:
        # {'usa_sector_acc', 'usa_sector_dist',
        #  'usa_index_acc',  'usa_index_dist',
        #  'eu_index_acc',   'eu_index_dist'}

**Seis CSVs** en outputs/history/ con receipt via output_registry.py:

- wyckoff_usa_sector_accumulation.csv
- wyckoff_usa_sector_distribution.csv
- wyckoff_usa_index_accumulation.csv
- wyckoff_usa_index_distribution.csv
- wyckoff_eu_index_accumulation.csv
- wyckoff_eu_index_distribution.csv

**Dos secciones nuevas en el reporte:** "Tickers en acumulacion" y
"Tickers en distribucion". Cada una con tres sub-tablas (USA sectores,
USA indices, EU indices).

## 4. Esquema exacto de los CSVs

Columnas: date (ISO date, ultimo bar del df_stocks), group (sector XLB->XLY
o indice SPY/QQQ/DIA/DAXEX/FEZ/ISF.L/LYXI), group_type (sector_usa /
index_usa / index_eu), ticker (holdings), name (holdings; EU lo tiene, USA
usa ticker), weight (float, holdings), phase (classify_phase(...).phase),
event_phase (classify_phase(...).event_phase), phase_since (ISO date o
vacio), close (ultimo Close), basis (current_phase_basis),
coverage_weight_pct (peso acumulado del grupo representado).

Ordenacion: por group ascendente, dentro por phase (ACC -> ACC_FORMING ->
MARKUP para acumulacion; DIST -> DIST_FORMING -> MARKDOWN para
distribucion), dentro por weight descendente.

Criterio de inclusion (iii, decidido):

- Tabla ACC: phase in {ACCUMULATION, ACCUMULATION_FORMING, MARKUP}
- Tabla DIST: phase in {DISTRIBUTION, DISTRIBUTION_FORMING, MARKDOWN}

La columna phase distingue "dentro de estructura" (ACC/DIST) de
"consumada" (MARKUP/MARKDOWN). El usuario filtra visualmente.

## 5. Cobertura y criterios de fiabilidad (medidos)

**Criterio 1 - barras por ticker:** >=460 barras para score valido (MA200
+ robust_zscore(200, min_periods=60)). Entre 200-460: INSUFFICIENT_DATA,
incluido en CSV con basis = "warm_up_incompleto". Por debajo de 200:
excluido. Estado: 313/329 tickers con score valido.

**Criterio 2 - tickers validos por grupo:** >=5 (coherente con
sector_wyckoff_distribution.py:85). Todos los grupos cumplen con >=19/20.

**Criterio 3 - representatividad economica:** top-20 por peso con
coverage_weight_pct explicito en el CSV.

| Grupo | Peso top-20 | Tickers con >=460 barras | Veredicto |
|---|---|---|---|
| XLB | 90.1% | 20/20 | Solido |
| XLC | 97.7% | 19/20 | Solido |
| XLE | 99.0% | 20/20 | Solido |
| XLF | 72.8% | 19/20 | Aceptable |
| XLI | 57.2% | 20/20 | Aceptable |
| XLK | 79.6% | 19/20 | Solido |
| XLP | 86.7% | 20/20 | Solido |
| XLRE | 86.7% | 20/20 | Solido |
| XLU | 83.1% | 20/20 | Solido |
| XLV | 78.9% | 20/20 | Solido |
| XLY | 84.5% | 20/20 | Solido |
| SPY | 49.9% | 20/20 | Aceptable |
| QQQ | 66.8% | 20/20 | Aceptable |
| DIA | 85.0% | 20/20 | Solido |
| FEZ | 66.3% | 20/20 | Aceptable |
| DAXEX | 78.4% | 20/20 | Aceptable |
| ISF.L | 60.8% | 20/20 | Aceptable |
| LYXI | 93.2% | 20/20 | Solido |

**IWM retirado.** Top-20 representa solo 6.2% del peso de 1.977 holdings.
Clasificar 20 microcaps y llamarlo "Russell 2000" seria enganoso.
Sub-fase futura (seccion 10).

## 6. Protocolo obligatorio para cualquier descarga futura

Cualquier ampliacion del parquet (IWM, sectores incompletos, historia
corta) debe pasar por los 6 pasos. Saltarlos reintroduce la contaminacion
por festivos que se tardo meses en cerrar:

1. Descargar OHLCV del proveedor.
2. Filtrar por calendario bursatil del mercado del ticker (NYSE, LSE,
   BME, Xetra).
3. Verificar 0 filas H=L=C en las nuevas.
4. Pasar por resolve_effective_date() si alimenta metricas agregadas.
5. Registrar el writer en output_registry.py para que el receipt lo cubra.
6. Test que verifique que no hay fechas no bursatiles en las filas nuevas.

## 7. Los 7 commits

**C1.** Crear indicators/wyckoff_holdings_report.py + tests unitarios con
fixtures sinteticos. Sin cablear.

**C2.** Cablear: output_registry.py (6 specs), run.py,
src/report_generator.py (2 secciones). Secciones nuevas + viejas
conviviendo.

**C3.** Retirar SLPM.

**C4.** Retirar lideres por sector/indice + indices_intl.py.

**C5.** Retirar fase ETF del ranking + classify_wyckoff_phase de
wyckoff_v1.py.

**C6.** Cierre documental (HANDOFF, 00_ARRANQUE, 04_HISTORICO, 62,
REGISTRO.yaml, correccion 15->16 detectores, caso TWST, ACCUMULATION=0,
Validation Gate, borrar rama feature/wyckoff-cascade).

**C7.** (si aparece) Correcciones de tests que fallan solo en CI.

Cada commit deja el pipeline funcional con suite verde.

## 8. Criterios de aceptacion

- Al final de C2: reporte diario contiene las 2 secciones nuevas con
  datos reales. Los 6 CSVs en outputs/history/ con receipt.
- Al final de C5: grep -r "classify_wyckoff_phase|slpm|stock_leader|
  index_leaders|index_phase" en src/, indicators/, regimes/ sin
  resultados productivos.
- Suite completa verde despues de cada commit.
- py scripts/verifica_gobernanza.py OK al final de C6.

## 9. Riesgos y no-objetivos

Riesgos: C2 anade 2 secciones mientras las viejas siguen (transitorio,
1-3 commits). C3-C4 reducen el conteo de tests - documentado en cada
commit. El reporte pierde "Indices (USA + Europa)" hasta que C2 cablee
las nuevas tablas.

No-objetivos: no se reescribe main. No se borran CSVs historicos con
receipt. No se toca wyckoff_distribution.py, wyckoff_accumulation.py,
wyckoff_orchestrator.py, sector_wyckoff_distribution.py. No se anade
INSUFFICIENT_DATA ni CONFLICT a las tablas (no son fases utiles para el
usuario).

## 10. Deuda abierta tras la sesion

- IWM - sub-fase de descarga (top-200/300 por peso, con protocolo de 6
  pasos).
- Sectores USA - 273 holdings de etf_holdings.csv no estan en parquet.
  Ampliacion futura con protocolo.
- Historia corta - 14 tickers entre 200-460 barras. Completar historico.
- Correccion 15->16 detectores (doc 62 + HANDOFF).
- Caso TWST en CONFLICT.
- ACCUMULATION=0 en produccion (estado transitorio por diseno).
- Validation Gate no reproducible standalone.
- Rama feature/wyckoff-cascade borrable.
- current_phase_basis cerrado en este sistema (antes deuda).
