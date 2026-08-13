import pandas as pd

def validate_market_data(df):
    """
    Verifica integridad de los datos de mercado.
    Asume que df tiene un MultiIndex con niveles ['Price', 'Ticker'].
    Retorna (lista de tickers vÃ¡lidos, diccionario de issues).
    """
    issues = {}
    valid_cols = []

    if not isinstance(df.columns, pd.MultiIndex):
        issues['global'] = 'El DataFrame no tiene MultiIndex en columnas'
        return valid_cols, issues

    # Los niveles son Price (Close, High, etc.) y Ticker (sÃ­mbolo)
    if 'Ticker' in df.columns.names:
        ticker_level = df.columns.names.index('Ticker')
        tickers = df.columns.levels[ticker_level].tolist()
    else:
        # fallback: segundo nivel
        tickers = df.columns.levels[1].tolist()

    for t in tickers:
        try:
            close = df[('Close', t)]
        except KeyError:
            issues[t] = 'DATA ISSUE (missing Close)'
            continue

        if close.isna().sum() / len(close) > 0.1:
            issues[t] = 'DATA ISSUE (NaNs >10%)'
        elif (close <= 0).any():
            issues[t] = 'DATA ISSUE (close <= 0)'
        elif len(close.dropna()) < 40:
            issues[t] = 'DATA ISSUE (insufficient history)'
        else:
            valid_cols.append(t)

    return valid_cols, issues

def validate_macro_manual(df):
    """Chequea que los CSVs manuales tengan columna 'date' y no estÃ©n vacÃ­os."""
    issues = []
    if df is None or df.empty:
        return False, ['Empty DataFrame']
    if 'date' not in df.columns:
        return False, ['Missing "date" column']
    if df['date'].isna().any():
        issues.append('NaNs in date column')
    if len(df) < 2:
        issues.append('Very few rows')
    return len(issues) == 0, issues

