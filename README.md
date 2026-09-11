# Radar de Rotación Sectorial

Sistema determinista y descriptivo de análisis macro-sectorial.
Sin ML predictivo. Sin señales de trading. Solo diagnóstico.

## Estado actual

- **Cobertura**: 313/313 tickers (100%)
- **Validation Gate**: 10/10
- **Tests**: 64/64 passed
- **Fuentes europeas**: Euronext (13) + Xetra (19) + BME (19)

## Filosofía

- Determinista, no predictivo.
- Descriptivo, no prescriptivo.
- Auditable: cada score es trazable a sus fuentes.
- Sin datos ficticios: si no hay dato, se omite (N/D).
- Sin mezcla de capas de flujo.

## Estructura

| Directorio | Contenido |
|------------|-----------|
| `config/` | Parámetros, tickers, pesos |
| `indicators/` | 30+ módulos de indicadores sectoriales |
| `regimes/` | 5 regímenes macro |
| `data/providers/` | 28 proveedores (Yahoo, Euronext, Xetra, BME, FRED, SEC...) |
| `src/` | Orquestador, loaders, report generator |
| `validation/` | 61 auditorías y validaciones |
| `tests/` | 22 archivos / 64 tests |
| `docs/` | Documentación completa |
| `outputs/` | Reportes, históricos, estado |

## Ejecución

```
pip install -r requirements.txt
py run.py
```

El pipeline:

1. Descarga datos de mercado (Yahoo + Euronext + Xetra + BME).
2. Calcula regímenes macro (Financial Conditions, Liquidity, Volatility, Macro).
3. Ejecuta motores táctico y estructural por sector.
4. Calcula 30+ indicadores (momentum, breadth, wyckoff, MTE, dark pool...).
5. Genera reporte diario en `outputs/report/reporte_diario.md`.

## Documentación

- [Índice completo](docs/automatica/README.md) - Arquitectura, módulos, fórmulas
- [Auditorías](docs/auditoria/) - Dictámenes y validaciones externas

## Fuentes de datos

| Fuente | Tickers | Método |
|--------|:-------:|--------|
| Yahoo | 262 | yfinance |
| Euronext | 13 | API AJAX pública (AES-256-CBC) |
| Xetra | 19 | WebSocket MDS + JWT |
| BME | 19 | API REST pública (JSON) |

## Reglas de operación

- No mezclar capas de flujo (ETF_PRIMARY_FLOW, CFTC_POSITION_FLOW, SEC_POSITION_FLOW, FLOW_PROXY).
- No construir superindicadores predictivos.
- No imputar valores artificiales.
- Documentar limitaciones descriptivas.

---

*Determinista. Descriptivo. Auditable.*
