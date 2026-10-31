# B6 - Checklist de auditoria (reporte)

Plantilla de B6. Una fila por modulo. Firma = [x] + fecha + SHA (si fix).
Sesion 1 (read-only) rellena: leido / 18 patrones / consumidores / tests / hallazgos.
Sesion 2 (fixes) rellena: firma / commit.

Modulos detectados por glob: 21

## 18 patrones

### 13 de 01b_PATRONES.md
- [ ] except desnudo / except Exception sin justificar
- [ ] hardcode fecha
- [ ] hardcode ticker
- [ ] hardcode lista sectores
- [ ] threshold sin constante nombrada
- [ ] datetime.now() sin justificar
- [ ] TODO/FIXME sin ticket
- [ ] fuente de verdad duplicada (viejo)
- [ ] ruta fichero en string
- [ ] nombre columna en string
- [ ] literal fragil en mensaje
- [ ] dead code
- [ ] cascada abortada (viejo)

### 5 nuevos (Fase 2)
- [ ] dead code funcional
- [ ] cascada abortada (funcion devuelve antes)
- [ ] literal fragil (numero/test/fecha)
- [ ] fuente de verdad duplicada
- [ ] fallback silencioso sin WARN

## Modulos

| # | Modulo | Leido | 18 patrones | Consumidores | Tests cubren | Hallazgos | Firma | Commit |
|---|---|---|---|---|---|---|---|---|
| 1 | `src/report/alerts.py` | [ ] | [ ] | [ ] | [ ] | | | |
| 2 | `src/report/breadth.py` | [ ] | [ ] | [ ] | [ ] | | | |
| 3 | `src/report/confirmation.py` | [ ] | [ ] | [ ] | [ ] | | | |
| 4 | `src/report/darkpool.py` | [ ] | [ ] | [ ] | [ ] | | | |
| 5 | `src/report/etf_flows.py` | [ ] | [ ] | [ ] | [ ] | | | |
| 6 | `src/report/flows_international.py` | [ ] | [ ] | [ ] | [ ] | | | |
| 7 | `src/report/freshness.py` | [ ] | [ ] | [ ] | [ ] | | | |
| 8 | `src/report/header.py` | [ ] | [ ] | [ ] | [ ] | | | |
| 9 | `src/report/helpers.py` | [ ] | [ ] | [ ] | [ ] | | | |
| 10 | `src/report/iae.py` | [ ] | [ ] | [ ] | [ ] | | | |
| 11 | `src/report/leaders.py` | [ ] | [ ] | [ ] | [ ] | | | |
| 12 | `src/report/market_context.py` | [ ] | [ ] | [ ] | [ ] | | | |
| 13 | `src/report/provenance.py` | [ ] | [ ] | [ ] | [ ] | | | |
| 14 | `src/report/rankings.py` | [ ] | [ ] | [ ] | [ ] | | | |
| 15 | `src/report/sector_context.py` | [ ] | [ ] | [ ] | [ ] | | | |
| 16 | `src/report/sectorial.py` | [ ] | [ ] | [ ] | [ ] | | | |
| 17 | `src/report/sentiment.py` | [ ] | [ ] | [ ] | [ ] | | | |
| 18 | `src/report/synthesis.py` | [ ] | [ ] | [ ] | [ ] | | | |
| 19 | `src/report/volatility_mte.py` | [ ] | [ ] | [ ] | [ ] | | | |
| 20 | `src/report/wyckoff_holdings.py` | [ ] | [ ] | [ ] | [ ] | | | |
| 21 | `src/report_generator.py` | [ ] | [ ] | [ ] | [ ] | | | |
