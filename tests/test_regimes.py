import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')
from regimes.financial_conditions import compute_financial_conditions

def make_df(tickers_dict):
    cols = []
    data = {}
    for t in tickers_dict:
        cols.append(('Close', t))
        data[('Close', t)] = tickers_dict[t]
    return pd.DataFrame(data, index=range(len(data[cols[0]])))

def test_liquidity_crisis():
    n = 120
    # Escenario de estrés realista que produce HIGH_STRESS en v4.3
    vix = [15 + i*(25/119) for i in range(n)]
    hyg = [80 - i*(20/119) for i in range(n)]
    lqd = [120]*n
    dxy = [100 + i*(10/119) for i in range(n)]
    tnx = [2 - i*(1/119) for i in range(n)]
    fvx = [3 - i*(0.5/119) for i in range(n)]
    df = make_df({
        '^VIX': vix,
        'HYG': hyg,
        'LQD': lqd,
        'DX-Y.NYB': dxy,
        '^TNX': tnx,
        '^FVX': fvx,
    })
    score, regime, conf = compute_financial_conditions(df)
    assert regime == 'HIGH_STRESS'

    # C19: confidence = disagreement entre componentes via rango.
    # No imponemos un threshold arbitrario; verificamos que el valor
    # coincide exactamente con la formula C19 aplicada a los
    # componentes del ultimo dia.
    from src.utils import robust_zscore, get_col
    from config.settings import CONFIDENCE_RANGE_DIVISOR

    components = {}
    try:
        vix_s = get_col(df, '^VIX', 'Close')
        components['vix'] = -np.tanh(robust_zscore(vix_s, 60)).iloc[-1]
    except KeyError:
        pass
    try:
        hyg_s = get_col(df, 'HYG', 'Close')
        lqd_s = get_col(df, 'LQD', 'Close')
        components['credit'] = np.tanh(robust_zscore(hyg_s/lqd_s, 60)).iloc[-1]
    except KeyError:
        pass
    try:
        dxy_s = get_col(df, 'DX-Y.NYB', 'Close')
        components['dollar'] = -np.tanh(robust_zscore(dxy_s.pct_change(fill_method=None), 60)).iloc[-1]
    except KeyError:
        pass
    try:
        tnx_s = get_col(df, '^TNX', 'Close')
        fvx_s = get_col(df, '^FVX', 'Close')
        components['curve'] = np.tanh(robust_zscore(tnx_s-fvx_s, 60)).iloc[-1]
    except KeyError:
        pass

    vals = [v for v in components.values() if not np.isnan(v)]
    if len(vals) >= 2:
        rng = max(vals) - min(vals)
        expected = max(0.0, min(1.0, 1.0 - rng / CONFIDENCE_RANGE_DIVISOR))
        assert abs(conf - expected) < 1e-9, f'conf={conf} != expected={expected}'

    # Invariante
    assert 0.0 <= conf <= 1.0

def test_liquidity_returns_valid_types():
    n = 200
    np.random.seed(123)
    noise = lambda: np.random.randn(n).cumsum() * 0.1
    df = make_df({
        '^VIX': 20 + noise(),
        'HYG': 80 + noise(),
        'LQD': 120 + noise(),
        'DX-Y.NYB': 100 + noise(),
        '^TNX': 3 + noise(),
        '^FVX': 2.8 + noise(),
    })
    score, regime, conf = compute_financial_conditions(df)
    assert isinstance(regime, str)
    assert isinstance(conf, float)
    assert not np.isnan(conf)


# ============================================================
# N8 (2026-10-14): guard de monotonicidad del liquidity_state
# ============================================================
def test_liquidity_state_no_retrocede_por_dato_stale(tmp_path, monkeypatch):
    """Si fed_data termina ANTES que el state persistido, no se sobreescribe.
    Bug observado: state fe98c29a (date=2026-10-13) sobreescrito por
    5e58197a (date=2026-10-07). El run del 14-oct tenia datos FRED stale."""
    import json
    import pandas as pd
    from regimes import liquidity as liq_mod

    # Preparar estructura de directorio y state persistido
    (tmp_path / "outputs" / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "outputs" / "state" / "liquidity_state.json").write_text(
        json.dumps({"score": 0.29, "date": "2026-10-13"}))

    # Router fake: datos stale hasta 2026-10-07
    class FakeRouter:
        def get_fed_data(self, index=None):
            idx = pd.date_range("2026-08-01", "2026-10-07", freq="D")
            return pd.DataFrame({
                "fed_balance": [6700000.0] * len(idx),
                "reverse_repo": [300.0] * len(idx),
                "sofr": [3.90] * len(idx),
                "fed_funds": [3.63] * len(idx),
            }, index=idx)

    monkeypatch.setattr(liq_mod, "DataRouter", FakeRouter)
    monkeypatch.chdir(tmp_path)

    liq_mod.compute_liquidity_score()

    written = json.loads((tmp_path / "outputs" / "state" / "liquidity_state.json").read_text())
    assert written["date"] == "2026-10-13", f"state no debe retroceder: {written}"
    assert written["score"] == 0.29



# ============================================================
# N8 (2026-10-14): trazabilidad de fechas y guard de frescura
# ============================================================
def test_liquidity_state_persiste_trazabilidad(tmp_path, monkeypatch):
    """El state extendido incluye computed_at, components con
    observation_date y flag stale."""
    import json
    import pandas as pd
    from regimes import liquidity as liq_mod

    (tmp_path / "outputs" / "state").mkdir(parents=True, exist_ok=True)

    today = pd.Timestamp.now().normalize()
    idx = pd.date_range(today - pd.Timedelta(days=90), today, freq="D")
    class FakeRouter:
        def get_fed_data(self, index=None):
            return pd.DataFrame({
                "fed_balance": [6700000.0 + i for i in range(len(idx))],
                "reverse_repo": [300.0 + i * 0.1 for i in range(len(idx))],
                "sofr": [3.90 + i * 0.001 for i in range(len(idx))],
                "fed_funds": [3.63 + i * 0.001 for i in range(len(idx))],
            }, index=idx)

    monkeypatch.setattr(liq_mod, "DataRouter", FakeRouter)
    monkeypatch.chdir(tmp_path)

    liq_mod.compute_liquidity_score()

    written = json.loads((tmp_path / "outputs" / "state" / "liquidity_state.json").read_text())
    assert "score" in written
    assert "date" in written
    assert "computed_at" in written
    assert "components" in written
    assert "stale" in written
    # Cada componente con observation_date
    for name in ["fed_balance", "reverse_repo", "sofr", "fed_funds"]:
        if name in written["components"]:
            assert "observation_date" in written["components"][name]
            assert "value" in written["components"][name]


def test_liquidity_state_no_sobreescribe_fresco_con_stale(tmp_path, monkeypatch):
    """Guard 2: si el nuevo calculo tiene componentes stale y el
    persistido no, no sobreescribir (aunque la fecha sea >=)."""
    import json
    import pandas as pd
    from regimes import liquidity as liq_mod

    (tmp_path / "outputs" / "state").mkdir(parents=True, exist_ok=True)

    # State previo: score OK, sin stale, con fecha de hace 30 dias
    old_date = (pd.Timestamp.now().normalize() - pd.Timedelta(days=30))
    (tmp_path / "outputs" / "state" / "liquidity_state.json").write_text(
        json.dumps({"score": 0.5, "date": str(old_date.date()),
                    "stale": False, "components": {}}))

    # Datos de entrada: hasta 30 dias antes de hoy -> componentes stale
    # (SOFR/RRPONTSYD maximo 4 dias)
    stale_end = pd.Timestamp.now().normalize() - pd.Timedelta(days=30)
    idx = pd.date_range(stale_end - pd.Timedelta(days=90), stale_end, freq="D")
    class FakeRouter:
        def get_fed_data(self, index=None):
            return pd.DataFrame({
                "fed_balance": [6700000.0 + i for i in range(len(idx))],
                "reverse_repo": [300.0 + i * 0.1 for i in range(len(idx))],
                "sofr": [3.90 + i * 0.001 for i in range(len(idx))],
                "fed_funds": [3.63 + i * 0.001 for i in range(len(idx))],
            }, index=idx)

    monkeypatch.setattr(liq_mod, "DataRouter", FakeRouter)
    monkeypatch.chdir(tmp_path)

    liq_mod.compute_liquidity_score()

    written = json.loads((tmp_path / "outputs" / "state" / "liquidity_state.json").read_text())
    # El state debe seguir siendo el anterior (no sobreescrito)
    assert written["score"] == 0.5, f"state no debe sobreescribirse: {written}"
    assert written["date"] == str(old_date.date())
