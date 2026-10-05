# -*- coding: utf-8 -*-
"""Paquete Gold Standard — Capa 1 y Capa 2 del estudio SOW v1.9.

Estado: BORRADOR — pendiente firma del auditor (protocolo v7).

Este paquete NO modifica:
  - indicators/wyckoff_v1.py
  - config/settings.py
  - protocolo v3 firmado (SOW_v19_PROTOCOL.json)
  - MDE, grid, consumidores
  - main

Alcance: scripts de muestreo, renderizado, adjudicacion, metricas
y orquestacion para el estudio de reference standard por panel experto.

Estructura:
  constants           constantes congeladas del estudio
  sampling_frame      universo elegible (240 sesiones + OHLCV completo)
  sampling_a          Muestra A enriquecida (200+200, min_gap=120)
  sampling_b          Muestra B representativa (33 estratos)
  power               simulacion del diseno para dimensionar n_B
  render              PNG ciego
  annotation          CSVs vacios + mapping
  adjudication        mayoria binaria + technical_ineligible
  kappa               Fleiss + Cohen
  metrics_simple      Wilson, proporciones por celda
  metrics_weighted    PPV/NPV/Se/Sp ponderadas
  bootstrap_rwy       Rao-Wu-Yue rescaled bootstrap
  bootstrap_cluster   Cluster bootstrap global de ticker (Capa 2)
  classify_as_of      Wrapper temporal/as-of
  run_capa1           Orquestador Capa 1
  run_capa2           Orquestador Capa 2
"""
from __future__ import annotations

__version__ = "0.0.1-draft"
__status__ = "BORRADOR - pendiente firma auditor SOW v1.9 v7"