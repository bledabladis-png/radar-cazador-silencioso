# Auditoria Fase 4 - Holdings y Lideres
**Fecha:** 2026-08-12 18:05

## Sectorial (ETF holdings)
Registros totales: 526
Columnas: ['etf', 'ticker']
Duplicados: 0
Tickers nulos/vacios: 0
Tickers en minuscula: False
Conteo por ETF/indice:
 etf  count
 XLB     27
 XLC     25
 XLE     23
 XLF     78
 XLI     85
 XLK     75
 XLP     36
XLRE     33
 XLU     33
 XLV     62
 XLY     49

## Internacional (Index holdings)
Registros totales: 95
Columnas: ['etf', 'ticker']
Duplicados: 0
Tickers nulos/vacios: 0
Tickers en minuscula: False
Conteo por ETF/indice:
  etf  count
DAXEX     10
  DIA     10
  FEZ     10
ISF.L     10
  IWM     10
 LYXI     10
  QQQ     15
  SPY     20

## Coherencia Sectorial
Tickers en líderes no presentes en holdings: 0
Grupos en holdings: ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
Grupos en líderes: ['XLB', 'XLF', 'XLI', 'XLV']

## Coherencia Internacional
Tickers en líderes no presentes en holdings: 0
Grupos en holdings: ['DAXEX', 'DIA', 'FEZ', 'ISF.L', 'IWM', 'LYXI', 'QQQ', 'SPY']
Grupos en líderes: ['Dow Jones', 'FTSE 100', 'Russell 2000']

## Persistencia y WLS Sectorial
Rango persistencia: [0.30, 0.80] (esperado 0-1)
  OK
WLS: media=-0.056, min=-1.948, max=1.753
Lider #1 por grupo y su wyckoff_score:
  XLB: NEM (wls=1.75, wyckoff=0.10)
  XLF: JPM (wls=0.67, wyckoff=0.50)
  XLI: UBER (wls=1.69, wyckoff=0.44)
  XLV: TMO (wls=1.22, wyckoff=0.56)

## Persistencia y WLS Internacional
Rango persistencia: [0.20, 0.80] (esperado 0-1)
  OK
WLS: media=0.931, min=0.157, max=2.439
Lider #1 por grupo y su wyckoff_score:
  Dow Jones: GS (wls=1.05, wyckoff=0.10)
  FTSE 100: HSBA.L (wls=2.44, wyckoff=0.40)
  Russell 2000: SMCI (wls=1.54, wyckoff=0.45)

## Conclusion
Auditoria de holdings y lideres completada. Ver detalles anteriores.
