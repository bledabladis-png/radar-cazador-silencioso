"""Submodulo SEC 13F Data Set.

FA-1.1 alcance:
  - downloader.py: descarga ZIP trimestral + extraccion + verificacion.
  - schema.py: contrato declarativo de los 7 TSVs.

NO incluye (fases posteriores):
  - parser: interpretacion de campos.
  - identity: resolucion de reporting relationships.
  - aggregation: NIPC, breadth, clasificacion.
"""