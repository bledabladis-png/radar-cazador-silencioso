# Evidence: probe P70

## Objetivo

Verificar la invariante de cardinalidad de HR_PLUS_RESTATEMENT en
classify_strategy (amendments.py) sobre Q4 2025 y Q1 2026.

## Ficheros

    _probe_p70.py             script de diagnostico (preservado byte a byte)
    probe_p70_result.json     resultado estructurado
    probe_p70_summary.txt     resumen legible

## Hashes SHA-256

    _probe_p70.py             d1742a769f24b7b07e28d3493ee1264e7ce82accca29fddd52a7011dbb82cd50  (6602 bytes)
    probe_p70_result.json     809c51af47ae154c6c01bba1cf3983a6c0143fd3b0711958fa8b9e6e18bae96f
    probe_p70_summary.txt     9e8c42d6a742d773be7e531782a7d8514a5c00aa64efc5080d81a718d6384fe5

## Reubicacion

- Ubicacion original: raiz del proyecto (D:\Macro_Sectorial\_probe_p70.py).
- Ubicacion actual:   este directorio.
- Motivo: preservar el script como evidencia reproducible del dictamen
  NIPC_P70_DICTAMEN.md, que en su seccion 8.2 lo cita por su nombre
  historico.

El script se ha movido byte a byte. El hash SHA-256 no ha cambiado.

## Relacion con NIPC_P70_DICTAMEN.md

El dictamen docs/auditoria/NIPC_P70_DICTAMEN.md (hash e8d4e4e8...) cita el
script por su nombre historico _probe_p70.py con la ubicacion original en
la raiz del proyecto.

La divergencia entre la ruta citada en el dictamen y la ruta actual de
este directorio es intencional:

- El dictamen NO se ha modificado para no invalidar su hash contractual
  (referenciado en NIPC_CONTRATOS_SEMANTICOS_v1.md seccion 10.2).
- El hash del script SI se ha preservado byte a byte en la reubicacion.

Cadena de trazabilidad:

    NIPC_CONTRATOS_SEMANTICOS_v1.md
        -> hash del dictamen (e8d4e4e8...)
        -> NIPC_P70_DICTAMEN.md (inmutable)
        -> cita _probe_p70.py (nombre historico)
        -> reubicacion documentada aqui.

## Reproduccion

Requisitos:

- Los parquets Q1 2026 en data/sec_13f/processed/2026Q1/.
- Los parquets Q4 2025 en data/sec_13f/processed/2025Q4/.
- Ejecutar desde la raiz del proyecto D:\Macro_Sectorial.

Comando (documentado; primera ejecucion desde la nueva ruta pendiente):

    py docs/auditoria/evidence/nipc_p70_probe/_probe_p70.py

Estado de verificacion: NO EJECUTADO desde la nueva ruta al 2026-09-20.
La ruta se documenta para reproducibilidad futura. La primera ejecucion
desde este directorio confirmara que la reproduccion funciona tal cual.

## Datos generados por el probe

El script escribe su salida en:

    docs/auditoria/evidence/nipc_p70_probe/probe_p70_result.json
    docs/auditoria/evidence/nipc_p70_probe/probe_p70_summary.txt

Sobrescribe los ficheros existentes en cada ejecucion. Los hashes
registrados en la seccion anterior corresponden al resultado de la
ejecucion de cierre (2026-09-20).