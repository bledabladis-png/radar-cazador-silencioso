## Proposito
Descripcion de los proveedores de datos utilizados por el radar.

## Arquitectura

| Fuente | Proveedor | Archivo | Actualizacion |
|--------|-----------|---------|---------------|
| Precios | Yahoo Finance | data/providers/yahoo.py | Diaria |
| Opciones | CBOE | data/providers/cboe.py | Diaria |
| Dark Pools | FINRA | data/providers/finra.py | Semanal |
| Macro | FRED / manual | data/providers/fred.py, data/macro_manual/ | Diaria |

### Mapeo FRED -> CSVs (data/macro_manual/)

| CSV | Columna | Serie FRED |
|-----|---------|------------|
| 10y3m.csv | T10Y3M | T10Y3M |
| actividad.csv | Industrial_Production_Total | INDPRO |
| actividad.csv | Industrial_Production_Manufacturing | IPMAN |
| actividad.csv | Retail_Sales | RSAFS |
| commercial_paper.csv | COMPOUT | COMPOUT |
| credit_oas.csv | CreditOAS | BAMLC0A0CM |
| discount_rate.csv | DPRIME | DPRIME |
| empleo.csv | NonFarm_Payrolls | PAYEMS |
| empleo.csv | Unemployment_Rate | UNRATE |
| empleo.csv | Initial_Claims | ICSA |
| empleo.csv | Continuing_Claims | CCSA |
| empleo.csv | Avg_Hourly_Earnings | CES0500000003 |
| empleo.csv | Total_Private_Employees | USPRIV |
| empleo.csv | Manufacturing_Employees | MANEMP |
| inflacion.csv | CPI | CPIAUCSL |
| inflacion.csv | Core_CPI | CPILFESL |
| inflacion.csv | PCE | PCEPI |
| inflacion.csv | Core_PCE | PCEPILFE |
| inflacion.csv | Inf_Expect_5Y | T5YIFR |
| inflacion.csv | Breakeven_10Y | T10YIE |
| iorb.csv | IORB | IORB |
| nfci.csv | NFCI | NFCI |
| rrpp.csv | RRPONTSYD | RRPONTSYD |
| sofr.csv | SOFR | SOFR |
| walcl.csv | WALCL | WALCL |

*Generado automaticamente desde MACRO_MANUAL_MAP en scripts/update_macro_manual.py.*


## Formulas
No aplica.

## Salidas
DataFrames de OHLCV, datos de opciones, datos ATS y series macroeconomicas.
