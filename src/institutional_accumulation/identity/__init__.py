"""Identity layer - catalogos de identidad + catalogo administrativo B1.

Modulos:
  openfigi_client.py        cliente OpenFIGI /v3/mapping
  radar_target_catalog.py   builder del catalogo radar (242 filas)
  target_universe.py        resolver CUSIP_13F -> target_membership

  catalog_key.py            B1: catalog_key + validadores A1/membership
  target_builder.py         B1: TargetUniverse (vinculacion por row_uid)
  period_state.py           B1: estados por periodo (identity + weight)

Estado: B1 CERRADO (#68). Todos los modulos con tests.
"""