import requests
import json
import re
from datetime import datetime
from .base import MarketDataProvider

class CboeProvider(MarketDataProvider):
    def __init__(self):
        self.name = "CBOE Official"
        self.url = "https://www.cboe.com/markets/us/options/market-statistics/daily/"

    def get_name(self) -> str:
        return self.name

    def is_available(self) -> bool:
        try:
            resp = requests.get(self.url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            return resp.status_code == 200
        except:
            return False

    def _extract_json(self):
        html = requests.get(self.url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
        pattern = r'\[1,"(.*?)"\]'
        matches = re.findall(pattern, html)
        if not matches:
            return None
        combined = "".join(matches)
        combined = combined.encode().decode('unicode_escape')
        combined = combined.replace('\\"', '"')
        start = combined.find('"optionsData"')
        if start == -1:
            return None
        start = combined.rfind('{', 0, start)
        if start == -1:
            return None
        count = 0
        end = -1
        for i in range(start, len(combined)):
            if combined[i] == '{':
                count += 1
            elif combined[i] == '}':
                count -= 1
                if count == 0:
                    end = i + 1
                    break
        if end == -1:
            return None
        json_str = combined[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    def get_options_data(self):
        data = self._extract_json()
        if not data:
            return None
        result = {}
        if 'selectedDate' in data:
            result['date'] = data['selectedDate']
        if 'optionsData' in data and 'ratios' in data['optionsData']:
            # Indexar por nombre para evitar dependencia del orden
            ratios = {}
            for r in data['optionsData']['ratios']:
                name = r.get('name', '')
                value = r.get('value', None)
                if value is not None:
                    try:
                        value = float(value)
                    except:
                        pass
                ratios[name] = value
            # Mapeo directo por nombre (robusto ante cambios de orden)
            result['total_pcr'] = ratios.get('TOTAL PUT/CALL RATIO', None)
            result['index_pcr'] = ratios.get('INDEX PUT/CALL RATIO', None)
            result['etp_pcr'] = ratios.get('EXCHANGE TRADED PRODUCTS PUT/CALL RATIO', None)
            result['equity_pcr'] = ratios.get('EQUITY PUT/CALL RATIO', None)
            result['vix_pcr'] = ratios.get('CBOE VOLATILITY INDEX (VIX) PUT/CALL RATIO', None)
            result['spx_pcr'] = ratios.get('SPX + SPXW PUT/CALL RATIO', None)
        return result

    def get_prices(self, tickers, start=None, end=None, period=None):
        raise NotImplementedError
    def get_treasury_yields(self, maturities=None, index=None):
        raise NotImplementedError
    def get_fed_data(self, index=None):
        raise NotImplementedError
