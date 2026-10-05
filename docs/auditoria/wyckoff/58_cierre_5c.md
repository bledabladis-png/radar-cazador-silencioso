# 58 - Cierre 5c (gate dictamen 57b)

Fecha: 2026-10-05
Rama: gold-standard-v3
Objeto: cerrar 5c comparativa legacy vs v1.8-core con el gate de 7
        puntos exigido por dictamen 57b D3.
Estado: 4/6 consumidores compatibles. 2/6 requieren decision.
        Gate 5c NO se cierra como 6/6.

## 1. Gate 5c - 7 puntos

Punto 1 - Entradas exactas legacy vs v1: DOCUMENTADO.

    Legacy build_ticker_df (wyckoff.py:355):
      dropna() completo sobre OHLCV. Puede perder filas si Volume NaN.
    v1.8 build_ticker_df (wyckoff_v1.py:493):
      dropna(subset=['Close']) + fillna(Open/High/Low con Close)
      + Volume.fillna(0.0). Conserva filas.

Diferencia semantica real. Afecta a los 6 consumidores.

Punto 2 - Diferencias de clasificacion: DOCUMENTADO.
    Referencia: 13_comparativa_legacy_v1.md.
    210/316 tickers cambian de fase. v1.8 concentra 67% en RANGE,
    0% DISTRIBUTION (fail-closed sin sow_params).

Punto 3 - Diferencias esperadas por diseno: DOCUMENTADO.
    Referencia: 14_diagnostico_5c.md (H1/H2/H3) +
                19_verificacion_manual_5casos_v16.md (dictamen O1-R).

Punto 4 - Sin dependencia residual de SOW/DISTRIBUTION: DOCUMENTADO.
    Referencia: 13 bloque post-v1.8.
    v1.8 sin sow_params devuelve RANGE + distribution_candidate=True.
    DISTRIBUTION=0 por fail-closed, no por algebra.

Punto 5 - Consumidores sin campos incompatibles: PARCIAL.
    Ver seccion 2.

Punto 6 - No look-ahead del core: VERIFICADO.
    Tests: test_v12_t4_no_lookahead_datos_futuros,
           test_v13_no_lookahead_equivalente_a_T4 (I23).
    I9/I11/I12 declarados skip por falta de API as_of.

Punto 7 - Regresion reproducible: CONGELADO en este doc.
    Script: scripts/compare_wyckoff_legacy_v18.py
    SHA-256:
      53E482EA03FF33DE1B6AE09D30B1B60BA8C8BA1754E1CE8AE2F4BCA56C418A7E
## 2. Consumidores - analisis

6 consumidores directos del legacy (plan seccion 2.4 declaraba 5):

    #  Consumidor                              Importa
    1  indicators/index_leaders.py             wyckoff_score,
                                               classify_wyckoff_phase,
                                               detect_spring, detect_sos,
                                               build_ticker_df
    2  indicators/index_phase.py               wyckoff_structure_core,
                                               build_ticker_df
    3  indicators/sector_breadth.py            classify_wyckoff_phase,
                                               build_ticker_df
    4  indicators/sector_wyckoff_distribution.py
                                               classify_wyckoff_phase,
                                               build_ticker_df
    5  indicators/stock_leader.py              wyckoff_score,
                                               classify_wyckoff_phase,
                                               detect_spring, detect_sos,
                                               build_ticker_df
    6  regimes/sector_regime.py                wyckoff_structure_core,
                                               build_ticker_df

Compatibilidad con v1.8:

    #  Compatible  Notas
    1  SI          Todas las funciones expuestas por v1.8.
    2  NO          wyckoff_structure_core no existe en v1.8.
    3  SI
    4  SI
    5  SI
    6  NO          wyckoff_structure_core no existe en v1.8.

## 3. Incompatibilidades detectadas

I-01. wyckoff_structure_core (legacy:301) no existe en v1.8.
      Es el clasificador por bandas de combined del legacy.
      2 consumidores lo usan.

      Opciones:
      (a) Anadir wyckoff_structure_core a v1.8.
          PROHIBIDO por dictamen 57b seccion 6.
      (b) Adaptar los 2 consumidores a classify_wyckoff_phase.
          Cambia su comportamiento (bandas -> conjunciones).
      (c) Mantener los 2 consumidores en legacy.
          Migracion parcial 4/6.

I-02. build_ticker_df semantica distinta (ver Punto 1).
      Afecta a los 6 consumidores. Los numeros de 13 reflejan el
      comportamiento de v1.8.

I-03. Cadena indirecta no listada en plan seccion 6:
      regimes/sector_regime.py -> sector_regime_matrix
      -> src/pipeline/engines.py ('wyckoff_phase')
      -> src/report/*.py.
      El plan seccion 6 lista 4 indirectos sin incluir engines.py.
## 4. Estado del gate

    Punto 1  DOCUMENTADO
    Punto 2  DOCUMENTADO
    Punto 3  DOCUMENTADO
    Punto 4  DOCUMENTADO
    Punto 5  PARCIAL (4/6 compatibles; I-01 requiere decision)
    Punto 6  VERIFICADO (T4, I23)
    Punto 7  CONGELADO en este documento (53E482EA...)

Gate 5c cierra como "4/6 migrables sin decision adicional; 2/6
requieren decision explicita". No cierra como 6/6.

## 5. Decision solicitada

    E1. Autoriza migrar los 4 consumidores compatibles
        (index_leaders, sector_breadth, sector_wyckoff_distribution,
        stock_leader) a v1.8-core tras commit de este cierre?

    E2. Para los 2 consumidores incompatibles (index_phase,
        sector_regime), elige:
          (b) adaptar a classify_wyckoff_phase
          (c) mantener en legacy
          (a) descartada (prohibida por 57b seccion 6)

    E3. Confirma que engines.py (indirecto no listado en plan
        seccion 6) se incluye en el alcance de 5d?

## 6. Prohibiciones vigentes

Las de 57 seccion 7. Sin cambios: no tocar wyckoff_v1.py, legacy,
config, protocolo v3; no activar SOW; no reabrir grid; no modificar
MDE = 5 pp; no migrar consumidores sin 5c cerrada; no merge a main.

## Anexo. Hashes

    scripts/compare_wyckoff_legacy_v18.py
        53E482EA03FF33DE1B6AE09D30B1B60BA8C8BA1754E1CE8AE2F4BCA56C418A7E
    indicators/wyckoff.py
        E4469595C28C8D44DB94F36F1AEC02093D30FAA74A4C3BFF8EFB186871CCD044
    indicators/wyckoff_v1.py
        FA9F30AD6EBE1DCBEECC3D058C82DE146F6AEB735A067B9752C772B97A4389AB

Fin del cierre 5c.