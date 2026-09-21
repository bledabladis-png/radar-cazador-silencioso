"""Submodulo SEC 13F Data Set.

Cubre el pipeline completo de ingestion + identity (FA-1 + FA-2,
cerrados y pusheados):

  downloader.py    descarga ZIP trimestral + extraccion + verificacion
  schema.py        contrato declarativo de los 7 TSVs
  parser.py        interpretacion de campos
  storage.py       escritura atomica de parquets
  manifest.py      lineage de 3 niveles (source + TSV + parquet)
  ingest.py        orquestador FA-1

  identity/
    temporal_filter.py    filtro canonico por PERIODOFREPORT
    cusip_resolver.py     CUSIP -> ticker con vigencia temporal
    relationships.py      Column 7 -> OTHERMANAGER2.SEQUENCENUMBER
    amendments.py         canonical snapshot composicional
    sec13f_list.py        SEC Official List 13(f) parser
    security_identity.py  canonical_security + operational_mapping_status

Estado: FA-1 + FA-2 CERRADOS / PUSHED.
"""