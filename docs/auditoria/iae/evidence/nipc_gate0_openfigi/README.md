# Evidencia OpenFIGI - NIPC Gate 0 OpenFIGI

Preservacion de la evidencia empirica del probe OpenFIGI ejecutado el
2026-09-19 sobre muestra estratificada de CUSIPs de 13F Q1 2026.

## Contexto

- Ciclo: IAE NIPC / Gate 0 OpenFIGI.
- Dictamen asociado: `docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_OPENFIGI_DICTAMEN.md`.
- Commit preservacion: ver `git log --oneline docs/auditoria/iae/evidence/nipc_gate0_openfigi/`.
- Entrada FOLLOWUPS: "IAE NIPC - Gate 0 OpenFIGI".

## Contenido

| Fichero            | Contenido                                                  |
|--------------------|------------------------------------------------------------|
| `README.md`        | Este documento.                                            |
| `build_sample.py`  | Script reproducible: genera `sample.json` + `cusips.txt`.  |
| `run_openfigi.py`  | Script reproducible: consulta OpenFIGI, escribe `result.json`. Sin API key embebida. |
| `sample.json`      | Muestra estratificada (5 estratos, CUSIPs unicos).         |
| `cusips.txt`       | Lista plana de CUSIPs consultados.                         |
| `result.json`      | Respuesta cruda de OpenFIGI.                               |
| `HASHES.txt`       | SHA-256 de cada artefacto.                                 |

## Metodologia del probe

- Endpoint: `https://api.openfigi.com/v3/mapping` (API v3).
- Tipo de consulta: `{"idType": "ID_CUSIP", "idValue": "<CUSIP>", "exchCode": "US"}`.
- Sin API key: batches de 5 jobs, sleep 2.5s (~25 req/min).
- Duracion registrada: ~89.5s. Sin HTTP 429.
- User-Agent: `Macro_Sectorial-research/1.0`.
- Fecha de ejecucion: 2026-09-19 (timestamp UTC en commit).

## Estratos

- A (radar USA mapped): 50 CUSIPs. Ground truth disponible (crosswalk interno).
- B (radar USA unmapped tickers): 22 tickers sin CUSIP conocido. NO consultados.
- C (top SSHPRNAMT unmapped): 50 CUSIPs.
- D (mid/low SSHPRNAMT, p40-p60): 50 CUSIPs, muestreo aleatorio seed=42.
- E (problematicos conocidos): 6 CUSIPs (NIPST, BRK-A/B, GOOG/GOOGL).

## Reproducibilidad

    cd docs/auditoria/iae/evidence/nipc_gate0_openfigi
    py build_sample.py
    py run_openfigi.py

`build_sample.py` requiere acceso al probe local
`D:/13f_probe/processed/2026Q1/` y a `D:/Macro_Sectorial/data/`.
Determinista: sin estado aleatorio no fijado (seed=42 en estrato D).

`run_openfigi.py` sin `OPENFIGI_API_KEY`: ~90s. Con la env var: batches
de 100 y sleep 1s.

## API key

No hay API key embebida en ningun artefacto. Si se escala con
autenticacion, debe usarse la variable de entorno `OPENFIGI_API_KEY`,
nunca hardcoded. La API v3 es la version vigente (v2 deprecada por
OpenFIGI ~junio 2026).

## Restricciones de licencia

Los identificadores FIGI estan dedicados al dominio publico. Existen
restricciones de licencia sobre identificadores propietarios de terceros
que la API no devuelve. La preservacion de `sample.json` y `result.json`
como evidencia del probe es compatible con dichas restricciones.

## Hashes SHA-256

Ver `HASHES.txt`.
