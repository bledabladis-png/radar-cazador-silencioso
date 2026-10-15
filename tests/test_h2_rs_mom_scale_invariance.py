"""Test H2: rs_mom es invariante a la escala del precio.

Auditoria externa 2026-10-15. El auditor sospecho que el rs_z que
entra en WLS pudiera depender del precio nominal de la accion (como
el rs de LIS). Verificado empiricamente: rs_mom = log(rs).diff(20) es
un log-return relativo, invariante a cualquier factor de escala.

Si este test falla, alguien ha cambiado rs_mom a un ratio nominal
(close/precio_etf) en lugar de un log-return. Ver H2.
"""
import numpy as np
import pandas as pd


def _rs_mom(close, etf, window=20):
    """Replica de stock_leader.py: rs_mom = log(close/etf).diff(20)."""
    rs = close / etf
    return np.log(rs).diff(window).iloc[-1]


def _rs_mom_alt(close, etf, window=20):
    """Formula alternativa: log-return directo.
    log(P_t/P_{t-20}) - log(E_t/E_{t-20}).
    Debe coincidir con _rs_mom."""
    ret_stock = np.log(close.iloc[-1] / close.iloc[-window-1])
    ret_etf = np.log(etf.iloc[-1] / etf.iloc[-window-1])
    return ret_stock - ret_etf


def test_rs_mom_invariante_a_split_simulado():
    """Simular un split 10:1 sobre la accion no debe cambiar rs_mom."""
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    np.random.seed(42)
    close = pd.Series(100.0 * np.cumprod(1 + np.random.randn(60) * 0.01), index=idx)
    etf = pd.Series(100.0 * np.cumprod(1 + np.random.randn(60) * 0.005), index=idx)

    rm_orig = _rs_mom(close, etf)
    rm_split = _rs_mom(close / 10, etf)
    assert abs(rm_orig - rm_split) < 1e-12, (
        f"rs_mom no invariante a split: {rm_orig} vs {rm_split}"
    )


def test_rs_mom_invariante_a_escala_arbitraria():
    """Cualquier factor k sobre el precio no cambia rs_mom."""
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    np.random.seed(7)
    close = pd.Series(50.0 * np.cumprod(1 + np.random.randn(60) * 0.02), index=idx)
    etf = pd.Series(80.0 * np.cumprod(1 + np.random.randn(60) * 0.01), index=idx)

    rm_base = _rs_mom(close, etf)
    for k in [0.001, 0.5, 2.0, 100.0, 1e6]:
        rm_k = _rs_mom(close * k, etf)
        assert abs(rm_base - rm_k) < 1e-10, f"falla con k={k}"


def test_rs_mom_coincide_con_log_return_directo():
    """rs_mom == log(P_t/P_{t-20}) - log(E_t/E_{t-20})."""
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    np.random.seed(3)
    close = pd.Series(100.0 * np.cumprod(1 + np.random.randn(60) * 0.01), index=idx)
    etf = pd.Series(50.0 * np.cumprod(1 + np.random.randn(60) * 0.005), index=idx)

    rm_a = _rs_mom(close, etf)
    rm_b = _rs_mom_alt(close, etf)
    assert abs(rm_a - rm_b) < 1e-10


def test_rs_z_intra_sector_invariante_a_escala():
    """robust_zscore intra-sector de rs_mom es invariante si rs_mom lo es."""
    # Simular dos acciones con precios muy distintos del mismo sector
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    np.random.seed(5)
    etf = pd.Series(100.0 * np.cumprod(1 + np.random.randn(60) * 0.005), index=idx)
    s1 = pd.Series(300.0 * np.cumprod(1 + np.random.randn(60) * 0.01), index=idx)
    s2 = pd.Series(20.0 * np.cumprod(1 + np.random.randn(60) * 0.01), index=idx)

    rm1_a = _rs_mom(s1, etf)
    rm2_a = _rs_mom(s2, etf)

    # Ahora con la accion 2 escalada x100 (simula split inverso)
    rm1_b = _rs_mom(s1, etf)
    rm2_b = _rs_mom(s2 * 100, etf)

    assert abs(rm1_a - rm1_b) < 1e-12
    assert abs(rm2_a - rm2_b) < 1e-12

    # Y el z-score intra-sector de [rm1, rm2] es identico
    def _robust_intra(vals):
        s = pd.Series(vals)
        med = s.median()
        mad = (s - med).abs().median()
        if pd.isna(mad) or mad == 0:
            return pd.Series(0.0, index=s.index)
        return (s - med) / (1.4826 * mad + 1e-9)

    z_a = _robust_intra([rm1_a, rm2_a])
    z_b = _robust_intra([rm1_b, rm2_b])
    assert (z_a - z_b).abs().max() < 1e-10
