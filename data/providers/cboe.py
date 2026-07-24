import requests, json, re, sys
from .base import MarketDataProvider

URL = "https://www.cboe.com/markets/us/options/market-statistics/daily/"

class CboeProvider(MarketDataProvider):
    def __init__(self):
        self.name = "CBOE Official"

    def get_name(self):
        return self.name

    def is_available(self):
        try:
            resp = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            return resp.status_code == 200
        except:
            return False

    def _extract_json(self):
        """Extrae el JSON de optionsData del HTML del CBOE."""
        html = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
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
        brace_start = combined.rfind('{', 0, start)
        if brace_start == -1:
            return None
        depth = 0
        brace_end = -1
        for i in range(brace_start, len(combined)):
            if combined[i] == '{':
                depth += 1
            elif combined[i] == '}':
                depth -= 1
                if depth == 0:
                    brace_end = i + 1
                    break
        if brace_end == -1:
            return None
        json_str = combined[brace_start:brace_end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    def _ratio(self, data, name):
        for item in data.get("ratios", []):
            if item.get("name") == name:
                return float(item["value"])
        return None

    def _section(self, data, name):
        rows = data.get(name, [])
        out = {}
        for r in rows:
            if r.get("name") == "VOLUME":
                out["call_volume"] = r.get("call")
                out["put_volume"] = r.get("put")
                out["total_volume"] = r.get("total")
            elif r.get("name") == "OPEN INTEREST":
                out["call_oi"] = r.get("call")
                out["put_oi"] = r.get("put")
                out["total_oi"] = r.get("total")
        return out

    def get_options_data(self):
        """Devuelve diccionario plano con todos los datos disponibles."""
        data = self._extract_json()
        if not data:
            return None

        od = data.get("optionsData", {})
        total = self._section(od, "SUM OF ALL PRODUCTS")
        index = self._section(od, "INDEX OPTIONS")
        equity = self._section(od, "EQUITY OPTIONS")
        etp   = self._section(od, "EXCHANGE TRADED PRODUCTS")
        spx   = self._section(od, "SPX + SPXW")
        vix   = self._section(od, "CBOE VOLATILITY INDEX (VIX)")

        result = {
            "date": data.get("selectedDate"),

            # Ratios
            "total_pcr": self._ratio(od, "TOTAL PUT/CALL RATIO"),
            "index_pcr": self._ratio(od, "INDEX PUT/CALL RATIO"),
            "equity_pcr": self._ratio(od, "EQUITY PUT/CALL RATIO"),
            "etp_pcr": self._ratio(od, "EXCHANGE TRADED PRODUCTS PUT/CALL RATIO"),
            "vix_pcr": self._ratio(od, "CBOE VOLATILITY INDEX (VIX) PUT/CALL RATIO"),
            "spx_pcr": self._ratio(od, "SPX + SPXW PUT/CALL RATIO"),

            # Total
            "total_call_volume": total.get("call_volume"),
            "total_put_volume": total.get("put_volume"),
            "total_volume": total.get("total_volume"),
            "total_call_oi": total.get("call_oi"),
            "total_put_oi": total.get("put_oi"),
            "total_oi": total.get("total_oi"),

            # Index
            "index_call_volume": index.get("call_volume"),
            "index_put_volume": index.get("put_volume"),
            "index_volume": index.get("total_volume"),
            "index_call_oi": index.get("call_oi"),
            "index_put_oi": index.get("put_oi"),
            "index_oi": index.get("total_oi"),

            # Equity
            "equity_call_volume": equity.get("call_volume"),
            "equity_put_volume": equity.get("put_volume"),
            "equity_volume": equity.get("total_volume"),
            "equity_call_oi": equity.get("call_oi"),
            "equity_put_oi": equity.get("put_oi"),
            "equity_oi": equity.get("total_oi"),

            # ETP
            "etp_call_volume": etp.get("call_volume"),
            "etp_put_volume": etp.get("put_volume"),
            "etp_volume": etp.get("total_volume"),

            # SPX
            "spx_call_volume": spx.get("call_volume"),
            "spx_put_volume": spx.get("put_volume"),
            "spx_volume": spx.get("total_volume"),

            # VIX
            "vix_call_volume": vix.get("call_volume"),
            "vix_put_volume": vix.get("put_volume"),
            "vix_volume": vix.get("total_volume"),
        }
        return result

    # Métodos no implementados (interfaz)
    def get_prices(self, tickers, start=None, end=None, period=None):
        raise NotImplementedError
    def get_treasury_yields(self, maturities=None, index=None):
        raise NotImplementedError
    def get_fed_data(self, index=None):
        raise NotImplementedError
