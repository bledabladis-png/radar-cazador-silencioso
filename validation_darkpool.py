import requests, base64, os, pandas as pd, io
from dotenv import load_dotenv
load_dotenv()
cid = os.getenv('FINRA_CLIENT_ID')
csc = os.getenv('FINRA_CLIENT_SECRET')

# --- Autenticacion FINRA ---
credentials = f'{cid}:{csc}'
encoded = base64.b64encode(credentials.encode()).decode()
token_resp = requests.post('https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token?grant_type=client_credentials', headers={'Authorization': f'Basic {encoded}'})
token = token_resp.json()['access_token']

# --- Descargar datos ATS de la semana 28 ---
url = 'https://api.finra.org/data/group/otcMarket/name/weeklySummary'
headers = {'Authorization': f'Bearer {token}', 'Accept': 'text/plain'}
params = {'weekStartDate': '2026-W28'}
resp = requests.get(url, headers=headers, params=params)
df_ats = pd.read_csv(io.StringIO(resp.text), sep=',', on_bad_lines='skip', low_memory=False)
df_ats = df_ats[df_ats['summaryTypeCode'] == 'ATS_W_SMBL_FIRM']
ats_vol = df_ats.groupby('issueSymbolIdentifier')['totalWeeklyShareQuantity'].sum()

# --- Cargar volumen total de Yahoo Finance para la misma semana ---
df_market = pd.read_csv('data/market_data.csv', header=[0,1], index_col=0, parse_dates=True)
start = '2026-07-06'
end = '2026-07-10'
week_data = df_market.loc[start:end]

# --- Calcular Dark Pool % para tickers de nuestro universo ---
tickers_prueba = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'XLK', 'XLV', 'TLT', 'HYG', 'LQD', 'EEM', 'GLD', 'SLV', 'USO', 'UNG']
resultados = []
for t in tickers_prueba:
    try:
        vol_total = week_data[('Volume', t)].sum() if ('Volume', t) in week_data.columns else 0
        vol_ats = ats_vol.get(t, 0)
        if vol_total > 0 and vol_ats > 0:
            dark_pool_pct = vol_ats / vol_total * 100
            resultados.append({'Ticker': t, 'Vol ATS': vol_ats, 'Vol Total': vol_total, 'Dark Pool %': dark_pool_pct})
    except:
        pass

df_res = pd.DataFrame(resultados).sort_values('Dark Pool %', ascending=False)
print('=== DARK POOL % POR TICKER (Semana 28, 2026) ===')
print(df_res.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
if len(df_res) > 0:
    media = df_res['Dark Pool %'].mean()
    mediana = df_res['Dark Pool %'].median()
    print(f'Media Dark Pool %: {media:.2f}%')
    print(f'Mediana Dark Pool %: {mediana:.2f}%')
