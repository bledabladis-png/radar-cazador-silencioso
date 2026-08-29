## Proposito
Documenta los modulos de flujo primario de ETFs (ETF Primary Flow), que estiman creaciones/redenciones de participaciones usando los cambios en Shares Outstanding multiplicados por el NAV.

## Arquitectura

- data/providers/ssga_fund_data.py: ETFs SPDR USA (11 sectores) + FEZ.
- data/providers/blackrock_fund_data.py: iShares DAXEX (DAX 40).
- data/providers/blackrock_isf_fund_data.py: iShares ISF.L (FTSE 100).
- data/providers/amundi_fund_data.py: Amundi LYXI (Ibex 35).
- Todos guardan históricos en outputs/history/ y se integran en run.py.


## Formulas

- **ETF Primary Flow:** ΔSharesOutstanding × NAV.
- **Flow % Assets:** EstimatedFlow / AUM (o TotalNetAssets), en decimal.
- **Flow Z-Score:** rolling z-score (120 días) sobre Flow % Assets.
- **Flow 5d/20d:** medias móviles del flujo estimado.


## Salidas

- outputs/history/etf_primary_flow.csv
- outputs/history/blackrock_dax_primary_flow.csv
- outputs/history/blackrock_isf_primary_flow.csv
- outputs/history/amundi_lyxi_primary_flow.csv
- Secciones en el reporte: '## Flujo Primario ETF (SPDR)', '## Flujo Primario DAXEX', '## Flujo Primario ISF.L', '## Flujo Primario LYXI'.


## Limitaciones Conocidas
El flujo primario estimado no es flujo institucional directo; solo refleja cambios en participaciones. Amundi requiere dos fechas efectivas para calcular el flujo.
