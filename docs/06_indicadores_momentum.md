# Indicadores: Momentum y Flujo

## Flow Proxy
```
Calcula el Flow Proxy compuesto para un ticker.
    Formula: 0.30*flow_smooth + 0.35*obv_z + 0.35*cmf_z
    donde:
      flow_smooth = EWMA(10) de robust_zscore(ret*dollar_vol, window=60)
      obv_z = robust_zscore(OBV.pct_change(), window=60)
      cmf_z = robust_zscore(CMF(20), window=60)
    Retorna una Serie temporal con el Flow Proxy compuesto.
```

## Constantes de Flujo
| Constante | Valor |
|---|---|
| FLOW_CMF_WINDOW | 20 |
| FLOW_EWM_SPAN | 10 |
| FLOW_ZSCORE_WINDOW | 60 |
