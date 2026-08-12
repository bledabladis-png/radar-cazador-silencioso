# -*- coding: utf-8 -*-
# Configuracion de indices internacionales - Fase 1 (EE.UU.)

INDEX_CONFIG = {
    'S&P 500': {
        'index_ticker': '^GSPC',
        'etf_ticker': 'SPY',
        'max_companies': 20
    },
    'Dow Jones': {
        'index_ticker': '^DJI',
        'etf_ticker': 'DIA',
        'max_companies': 10
    },
    'Nasdaq-100': {
        'index_ticker': '^NDX',
        'etf_ticker': 'QQQ',
        'max_companies': 15
    },
    'Russell 2000': {
        'index_ticker': '^RUT',
        'etf_ticker': 'IWM',
        'max_companies': 10
    }
}
