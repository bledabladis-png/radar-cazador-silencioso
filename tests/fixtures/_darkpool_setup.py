"""DT3 Fase 0: setup compartido para caracterizar darkpool.

NO es codigo productivo. Lo usan el generador y el test de caracterizacion.
"""
import numpy as np
import pandas as pd

SEED = 42
LATEST_WEEK = "2026-08-24"
N_HIST_WEEKS = 110
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
# Volumen semanal sintetico por ticker
BASE_WEEKLY = {
    "AAPL": 20_000_000,
    "MSFT": 25_000_000,
    "GOOGL": 30_000_000,
    "AMZN": 35_000_000,
    "META": 40_000_000,
}
ATS_FRACTION = 0.30


def make_hist():
    rng = np.random.RandomState(SEED)
    weeks = pd.date_range(end=LATEST_WEEK, periods=N_HIST_WEEKS, freq="W-MON")
    ratios = rng.uniform(0.15, 0.30, N_HIST_WEEKS)
    return pd.DataFrame({"week": weeks, "ratio": ratios})


def make_market_df():
    dates = pd.date_range("2025-01-01", "2026-08-28", freq="B")
    data = {}
    for t in TICKERS:
        vol = np.zeros(len(dates))
        mask = (dates >= LATEST_WEEK) & (dates <= "2026-08-28")
        vol[mask] = BASE_WEEKLY[t] / 5
        data[("Close", t)] = 100.0 + np.arange(len(dates)) * 0.01
        data[("Volume", t)] = vol
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def make_ats_df():
    return pd.DataFrame({
        "issueSymbolIdentifier": list(TICKERS),
        "totalWeeklyShareQuantity": [
            int(BASE_WEEKLY[t] * ATS_FRACTION) for t in TICKERS
        ],
    })


def make_finra():
    ats_df = make_ats_df()

    class FakeFinra:
        def get_latest_week(self):
            return LATEST_WEEK

        def get_all_tiers(self, week):
            return ats_df

    return FakeFinra


def setup_and_run(monkeypatch, tmp_path):
    """Configura entorno y llama compute_darkpool_signals.

    Returns: dict devuelto por la funcion.
    """
    monkeypatch.chdir(tmp_path)

    out_hist = tmp_path / "outputs" / "history"
    out_hist.mkdir(parents=True, exist_ok=True)
    make_hist().to_csv(out_hist / "darkpool_history.csv", index=False)

    # Interceptar read_parquet para no depender de data/*.parquet
    def fake_read_parquet(path, *a, **kw):
        raise FileNotFoundError(f"fake: {path}")

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    # Interceptar FinraProvider
    import indicators.darkpool as dp
    monkeypatch.setattr(dp, "FinraProvider", make_finra())

    df_market = make_market_df()
    return dp.compute_darkpool_signals(df_market=df_market, df_stocks=None)