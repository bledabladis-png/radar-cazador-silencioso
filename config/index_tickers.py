# -*- coding: utf-8 -*-
# Configuracion de indices internacionales - Fase 1b (EE.UU. + Europa)

INDEX_CONFIG = {
    # EE.UU.
    'S&P 500': {
        'index_ticker': '^GSPC',
        'etf_ticker': 'SPY',
        'max_companies': 20
    },
    'Dow Jones': {
        'index_ticker': '^DJI',
        'etf_ticker': 'DIA',
        'max_companies': 20
    },
    'Nasdaq-100': {
        'index_ticker': '^NDX',
        'etf_ticker': 'QQQ',
        'max_companies': 20
    },
    'Russell 2000': {
        'index_ticker': '^RUT',
        'etf_ticker': 'IWM',
        'max_companies': 20
    },
    # Europa
    'Euro Stoxx 50': {
        'index_ticker': '^STOXX50E',
        'etf_ticker': 'FEZ',
        'max_companies': 20
    },
    'Ibex 35': {
        'index_ticker': '^IBEX',
        'etf_ticker': 'LYXI',
        'max_companies': 20
    },
    'DAX 40': {
        'index_ticker': 'DAXEX',
        'etf_ticker': 'DAXEX',
        'max_companies': 20
    },
    'FTSE 100': {
        'index_ticker': '^FTSE',
        'etf_ticker': 'ISF.L',
        'max_companies': 20
    }
}
