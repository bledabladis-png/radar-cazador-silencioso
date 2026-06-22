import re
with open('regimes/sector_regime.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = """    assets = {
        'Sectores': ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC'],
        'Indices': ['^GSPC', '^NDX', '^RUT', '^STOXX50E', 'EEM'],
        'Bonos': ['^TNX', '^FVX', '^TYX'],
        'Credito': ['HYG', 'LQD'],
        'Materias Primas': ['^SPGSCI', 'GC=F', 'HG=F', 'CL=F', 'BZ=F'],
        'Divisas': ['DX-Y.NYB', 'EURUSD=X', 'USDJPY=X', 'USDCNY=X'],
    }"""

new = """    assets = {
        'Sectores': ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC'],
        'Indices': ['^GSPC', '^NDX', '^RUT', '^STOXX50E', 'EEM', 'EWJ'],
        'Bonos': ['BIL', 'IEF', 'TLT'],
        'Credito': ['HYG', 'LQD'],
        'Factores': ['VLUE', 'MTUM', 'QUAL'],
        'Small Caps Intl': ['SCHC', 'EWX'],
        'Bonos Emergentes': ['EMB', 'ELD'],
        'Materias Primas': ['^SPGSCI', 'GC=F', 'HG=F', 'CL=F', 'BZ=F', 'NG=F'],
        'Divisas': ['DX-Y.NYB', 'EURUSD=X', 'USDJPY=X', 'USDCNY=X'],
    }"""

content = content.replace(old, new)
with open('regimes/sector_regime.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('assets actualizado con los nuevos activos.')
