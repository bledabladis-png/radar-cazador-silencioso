# Radar de Rotacion Sectorial

Sistema determinista, descriptivo y auditable de analisis macro-sectorial.
Sin ML predictivo. Sin senales de trading. Solo diagnostico.

---

## Arranque

Si eres un asistente entrante, lee primero:

**[Consolidacion_Documentos/00_ARRANQUE.md](Consolidacion_Documentos/00_ARRANQUE.md)**

Ese documento es autosuficiente para empezar. Referencia on-demand:
`01_METODO.md` (metodo), `02_ARQUITECTURA.md` (mapa), `03_IAE.md`
(subsistema IAE), `04_HISTORICO.md` (cronologia),
`05_BITACORA.md` (sesiones).

---

## Estado del sistema

Snapshot autoritativo: `Consolidacion_Documentos/ESTADO_SISTEMA.md`
(regenerable con `py scripts/generate_estado_sistema.py`).

Resumen:
- Tests: 2267 passed + 2 skipped.
- Validation Gate: 10/10.
- Cobertura configurada: 313 tickers.
- Working tree limpio.

---

## Comandos de arranque

    Set-Location D:\Macro_Sectorial
    git log --oneline -5
    git status -sb
    py scripts\generate_estado_sistema.py
    py -m pytest tests/ -q --tb=line
    py -m pyflakes src\ scripts\ ; py -m compileall . -q

Ejecucion completa del pipeline (~10-15 min):

    py run.py

Salida: `outputs/report/reporte_diario.md`.

---

## Estructura

| Directorio | Contenido |
|---|---|
| `config/` | Parametros, tickers, pesos |
| `regimes/` | 4 regimenes macro (financial, liquidity, volatility, macro) |
| `indicators/` | 50 modulos de indicadores (incluye paquete `mte/`) |
| `src/` | Orquestador, loaders, nucleo temporal, report generator |
| `src/report/` | 20 modulos de render |
| `src/pipeline/` | 18 modulos (12 fases + IAE) |
| `src/temporal_contracts/` | 10 contratos temporales (FU-021-5) |
| `src/institutional_accumulation/` | Modulo IAE (SEC 13F), 36 ficheros |
| `data/providers/` | 26 providers (Yahoo, Euronext, Xetra, BME, LSE, FRED, SEC...) |
| `scripts/` | 34 scripts + 3 en `scripts/audit/` |
| `validation/` | 7 modulos de validacion |
| `tests/` | 182 ficheros locales (~22900 LOC) |
| `docs/` | `automatica/` (auto-generados) + `auditoria/` (evidencia IAE) |
| `outputs/` | Reportes, historicos, estado |

---

## Filosofia

- Determinista, no predictivo.
- Descriptivo, no prescriptivo.
- Auditable: cada score es trazable a sus fuentes.
- Sin datos ficticios: `N/D` antes que imputar.
- Sin mezcla de capas de flujo (ETF_PRIMARY_FLOW, CFTC_POSITION_FLOW,
  SEC_POSITION_FLOW, FLOW_PROXY, QQQ NPORT-P FLOW, QQQ SEC FLOW).
- Sin superindicadores.

---

## Fuentes de datos (resumen)

| Fuente | Tickers | Metodo |
|---|:---:|---|
| Yahoo | 262 | yfinance |
| Euronext | 13 | API AJAX publica (AES-256-CBC) |
| Xetra | 19 | WebSocket MDS + JWT |
| BME | 19 | API REST publica (JSON) |
| LSE | 20 | Scraper privado (Refinitiv Widgets) |
| OilPriceAPI | 3 | Spot GC/HG/NG |
| CBOE | - | Opciones diarias + ^VIX3M |
| FRED | - | Liquidez y yields |
| CFTC / FINRA / SEC | - | Position flow / Dark pools / 13F + N-PORT |

---

*Determinista. Descriptivo. Auditable.*