import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from io import StringIO
from .base import MarketDataProvider

BASE_URL = "https://api.finra.org/data/group/otcMarket/name"

class FinraProvider(MarketDataProvider):
    def __init__(self):
        self.name = "FINRA ATS"
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "text/plain",
            "Content-Type": "application/json",
            "Origin": "https://otctransparency.finra.org",
            "Referer": "https://otctransparency.finra.org/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
        })

    def get_name(self) -> str:
        return self.name

    def is_available(self) -> bool:
        try:
            week = self.get_latest_week()
            return week is not None
        except:
            return False

    # ------------------------------------------------------------
    # FUNCIONES PRIVADAS (GENÉRICAS)
    # ------------------------------------------------------------
    def _post(self, endpoint, payload):
        try:
            resp = self._session.post(f"{BASE_URL}/{endpoint}", json=payload, timeout=60)
            resp.raise_for_status()
            return resp
        except Exception as e:
            print(f"ERROR en {endpoint}: {e}")
            if 'resp' in locals():
                print(f"Status: {resp.status_code}")
                print(resp.text[:500])
            return None

    def _paginated_request(self, endpoint, payload):
        offset = 0
        frames = []
        while True:
            payload["offset"] = offset
            payload["limit"] = 5000
            resp = self._post(endpoint, payload)
            if resp is None:
                break
            # Leer CSV protegiendonos contra respuesta vacia
            try:
                df = pd.read_csv(StringIO(resp.text), sep="|", on_bad_lines="skip")
            except pd.errors.EmptyDataError:
                break
            if df.empty:
                break
            # Quedarnos solo con las columnas que nos interesan
            if "issueSymbolIdentifier" in df.columns and "totalWeeklyShareQuantity" in df.columns:
                df = df[["issueSymbolIdentifier", "totalWeeklyShareQuantity"]]
            frames.append(df)
            total = int(resp.headers.get("record-total", 0))
            offset += 5000
            if offset >= total:
                break
            time.sleep(2)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ------------------------------------------------------------
    # MÉTODOS PÚBLICOS
    # ------------------------------------------------------------
    def get_archive_index(self):
        url = "https://otctransparency.finra.org/otctransparency/assets/archives/atsdownload/index.json"
        try:
            return requests.get(url).json()
        except:
            return []

    def get_available_weeks(self):
        resp = self._post("weeklyDownloadDetail", {})
        if resp is not None:
            return resp.json()
        return []

    def get_latest_week(self):
        for i in range(6):
            test_date = (datetime.now() - timedelta(weeks=i))
            monday = test_date - timedelta(days=test_date.weekday())
            monday_str = monday.strftime('%Y-%m-%d')
            data = self.get_week_summary(monday_str)
            if not data.empty:
                return monday_str
        return None

    def get_week_summary(self, week, tier="T1"):
        payload = {
            "quoteValues": False,
            "delimiter": "|",
            "limit": 5000,
            "fields": [
                "tierDescription", "issueSymbolIdentifier", "issueName",
                "marketParticipantName", "MPID", "totalWeeklyShareQuantity",
                "totalWeeklyTradeCount", "lastUpdateDate"
            ],
            "sortFields": ["issueSymbolIdentifier", "MPID"],
            "compareFilters": [
                {"fieldName": "summaryTypeCode", "fieldValue": "ATS_W_SMBL_FIRM", "compareType": "EQUAL"},
                {"fieldName": "weekStartDate", "fieldValue": week, "compareType": "EQUAL"},
                {"fieldName": "tierIdentifier", "fieldValue": tier, "compareType": "EQUAL"}
            ]
        }
        return self._paginated_request("weeklySummary", payload)

    def get_symbol(self, symbol, week, tier="T1"):
        payload = {
            "quoteValues": False,
            "delimiter": "|",
            "limit": 5000,
            "sortFields": ["-totalWeeklyShareQuantity"],
            "compareFilters": [
                {"fieldName": "summaryTypeCode", "fieldValue": "ATS_W_SMBL_FIRM", "compareType": "EQUAL"},
                {"fieldName": "issueSymbolIdentifier", "fieldValue": symbol, "compareType": "EQUAL"},
                {"fieldName": "weekStartDate", "fieldValue": week, "compareType": "EQUAL"},
                {"fieldName": "tierIdentifier", "fieldValue": tier, "compareType": "EQUAL"}
            ]
        }
        return self._paginated_request("weeklySummary", payload)

    def get_all_tiers(self, week):
        frames = []
        for tier in ["T1", "T2", "OTCE"]:
            df = self.get_week_summary(week, tier)
            if not df.empty:
                frames.append(df)
            time.sleep(2)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ------------------------------------------------------------
    # MÉTODOS NO IMPLEMENTADOS (interfaz)
    # ------------------------------------------------------------
    def get_prices(self, tickers, start=None, end=None, period=None):
        raise NotImplementedError("FINRA no proporciona precios")
    def get_treasury_yields(self, maturities=None, index=None):
        raise NotImplementedError("FINRA no proporciona yields")
    def get_fed_data(self, index=None):
        raise NotImplementedError("FINRA no proporciona datos de la Fed")
    def get_options_data(self, index=None):
        raise NotImplementedError("FINRA no proporciona datos de opciones")
