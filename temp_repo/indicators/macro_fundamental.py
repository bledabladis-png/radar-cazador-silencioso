import pandas as pd
import numpy as np
from src.utils import tanh_normalize
from datetime import datetime

def fundamental_signals(df_macro):
    if df_macro is None or df_macro.empty:
        return None

    end_date = datetime.now().strftime('%Y-%m-%d')
    daily_index = pd.date_range(start='2000-01-01', end=end_date, freq='D')

    base = pd.DataFrame(index=daily_index)
    base.index.name = 'date'

    df_macro['date'] = pd.to_datetime(df_macro['date'])
    df_macro = df_macro.set_index('date').sort_index()
    base = base.join(df_macro, how='left')
    base = base.interpolate(method='linear', limit_direction='both').ffill().bfill()

    signals = pd.DataFrame(index=base.index)

    # Inflacion (niveles)
    inflation_cols = [c for c in base.columns if any(k in c.lower() for k in ['cpi','pce','inflacion','breakeven','expect'])]
    if inflation_cols:
        inflation_series = base[inflation_cols].mean(axis=1)
        signals['inflation'] = tanh_normalize(inflation_series)

    # Empleo (niveles)
    employment_cols = [c for c in base.columns if any(k in c.lower() for k in ['empleo','nfp','unemployment','payroll','claims','earnings','private','manufacturing'])]
    if employment_cols:
        emp_series = base[employment_cols].mean(axis=1)
        signals['employment'] = tanh_normalize(emp_series)

    # Actividad: usar cambio porcentual anual para capturar dinamica
    activity_cols = [c for c in base.columns if any(k in c.lower() for k in ['industrial_production_total','industrial_production_manufacturing','retail_sales'])]
    if activity_cols:
        # Calcular cambio porcentual respecto a 252 dÃ­as hÃ¡biles (1 aÃ±o)
        act_change = base[activity_cols].pct_change(252).mean(axis=1)
        signals['activity'] = tanh_normalize(act_change)
    else:
        # Fallback: si no hay columnas, usar Leading_Index si existe
        if 'actividad_Leading_Index' in base.columns:
            signals['activity'] = tanh_normalize(base['actividad_Leading_Index'])

    return signals
