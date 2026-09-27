"""Tests pipeline: flows_primary (fase 4).

Cobertura de orquestador con 7 providers externos. Los providers
individuales ya estan cubiertos por sus propios tests. Aqui se verifica:
  - Contrato: 8 keys siempre.
  - Propagacion de exito por provider.
  - Degradacion individual (fallo o vacio -> None).
  - Sector Flow Characteristics solo si etf_primary_flow_data presente.

Estrategia de mocks: parchear los nombres importados en el modulo
(no los subyacentes) y sustituir retry_call por un passthrough.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import flows_primary as fp


EXPECTED_KEYS = {
    "etf_primary_flow_data", "sector_flow_characteristics_df",
    "blackrock_dax_flow", "blackrock_isf_flow", "blackrock_iwm_flow",
    "amundi_lyxi_flow", "qqq_sec_flow", "cftc_position_flow_data",
}


def _df():
    return pd.DataFrame({"x": [1.0, 2.0]})


def _passthrough_retry(f, *a, **kw):
    return f(*a, **kw)


def _all_none(tmp_path, monkeypatch):
    """Setup: cwd=tmp, todos los providers devuelven None."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    return {
        "src.pipeline.flows_primary.retry_call": _passthrough_retry,
        "src.pipeline.flows_primary.get_etf_primary_flow_data": None,
        "src.pipeline.flows_primary.get_blackrock_dax_primary_flow": None,
        "src.pipeline.flows_primary.get_blackrock_isf_primary_flow": None,
        "src.pipeline.flows_primary.get_blackrock_iwm_primary_flow": None,
        "src.pipeline.flows_primary.get_amundi_lyxi_primary_flow": None,
        "src.pipeline.flows_primary.get_qqq_sec_primary_flow": None,
        "src.pipeline.flows_primary.get_cftc_position_flow_data": None,
    }


def test_flows_primary_contract_8_keys(tmp_path, monkeypatch):
    _all_none(tmp_path, monkeypatch)
    with patch("src.pipeline.flows_primary.retry_call",
               side_effect=_passthrough_retry), \
         patch("src.pipeline.flows_primary.get_etf_primary_flow_data", return_value=None), \
         patch("src.pipeline.flows_primary.get_blackrock_dax_primary_flow", return_value=None), \
         patch("src.pipeline.flows_primary.get_blackrock_isf_primary_flow", return_value=None), \
         patch("src.pipeline.flows_primary.get_blackrock_iwm_primary_flow", return_value=None), \
         patch("src.pipeline.flows_primary.get_amundi_lyxi_primary_flow", return_value=None), \
         patch("src.pipeline.flows_primary.get_qqq_sec_primary_flow", return_value=None), \
         patch("src.pipeline.flows_primary.get_cftc_position_flow_data", return_value=None):
        out = fp.compute_flows_primary(pd.DataFrame())
    assert set(out.keys()) == EXPECTED_KEYS
    for k, v in out.items():
        assert v is None, f"{k} deberia ser None pero es {type(v)}"


def test_flows_primary_propaga_df_de_cada_provider(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    with patch("src.pipeline.flows_primary.retry_call",
               side_effect=_passthrough_retry), \
         patch("src.pipeline.flows_primary.get_etf_primary_flow_data", return_value=_df()), \
         patch("src.pipeline.flows_primary.get_blackrock_dax_primary_flow", return_value=_df()), \
         patch("src.pipeline.flows_primary.get_blackrock_isf_primary_flow", return_value=_df()), \
         patch("src.pipeline.flows_primary.get_blackrock_iwm_primary_flow", return_value=_df()), \
         patch("src.pipeline.flows_primary.get_amundi_lyxi_primary_flow", return_value=_df()), \
         patch("src.pipeline.flows_primary.get_qqq_sec_primary_flow", return_value=_df()), \
         patch("src.pipeline.flows_primary.get_cftc_position_flow_data", return_value=_df()), \
         patch("src.pipeline.flows_primary.compute_sector_flow_characteristics",
               return_value=_df()):
        out = fp.compute_flows_primary(pd.DataFrame())
    for k in EXPECTED_KEYS:
        assert out[k] is not None, f"{k} no propagado"


def test_flows_primary_fallo_dax_no_rompe_otros(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    with patch("src.pipeline.flows_primary.retry_call",
               side_effect=_passthrough_retry), \
         patch("src.pipeline.flows_primary.get_etf_primary_flow_data", return_value=_df()), \
         patch("src.pipeline.flows_primary.get_blackrock_dax_primary_flow",
               side_effect=RuntimeError("boom")), \
         patch("src.pipeline.flows_primary.get_blackrock_isf_primary_flow", return_value=_df()), \
         patch("src.pipeline.flows_primary.get_blackrock_iwm_primary_flow", return_value=_df()), \
         patch("src.pipeline.flows_primary.get_amundi_lyxi_primary_flow", return_value=_df()), \
         patch("src.pipeline.flows_primary.get_qqq_sec_primary_flow", return_value=_df()), \
         patch("src.pipeline.flows_primary.get_cftc_position_flow_data", return_value=_df()), \
         patch("src.pipeline.flows_primary.compute_sector_flow_characteristics",
               return_value=_df()):
        out = fp.compute_flows_primary(pd.DataFrame())
    assert out["blackrock_dax_flow"] is None
    assert out["blackrock_isf_flow"] is not None
    assert out["amundi_lyxi_flow"] is not None


def test_flows_primary_empty_df_se_convierte_a_none(tmp_path, monkeypatch):
    """Si un provider devuelve df.empty, se guarda como None."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    empty = pd.DataFrame()
    with patch("src.pipeline.flows_primary.retry_call",
               side_effect=_passthrough_retry), \
         patch("src.pipeline.flows_primary.get_etf_primary_flow_data", return_value=empty), \
         patch("src.pipeline.flows_primary.get_blackrock_dax_primary_flow", return_value=empty), \
         patch("src.pipeline.flows_primary.get_blackrock_isf_primary_flow", return_value=empty), \
         patch("src.pipeline.flows_primary.get_blackrock_iwm_primary_flow", return_value=empty), \
         patch("src.pipeline.flows_primary.get_amundi_lyxi_primary_flow", return_value=empty), \
         patch("src.pipeline.flows_primary.get_qqq_sec_primary_flow", return_value=empty), \
         patch("src.pipeline.flows_primary.get_cftc_position_flow_data", return_value=empty), \
         patch("src.pipeline.flows_primary.compute_sector_flow_characteristics",
               return_value=empty):
        out = fp.compute_flows_primary(pd.DataFrame())
    for k, v in out.items():
        assert v is None, f"{k} deberia ser None por df.empty"


def test_flows_primary_sector_char_no_se_calcula_sin_etf_flow(tmp_path, monkeypatch):
    """compute_sector_flow_characteristics NO debe invocarse si
    etf_primary_flow_data es None."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    with patch("src.pipeline.flows_primary.retry_call",
               side_effect=_passthrough_retry), \
         patch("src.pipeline.flows_primary.get_etf_primary_flow_data", return_value=None), \
         patch("src.pipeline.flows_primary.get_blackrock_dax_primary_flow", return_value=None), \
         patch("src.pipeline.flows_primary.get_blackrock_isf_primary_flow", return_value=None), \
         patch("src.pipeline.flows_primary.get_blackrock_iwm_primary_flow", return_value=None), \
         patch("src.pipeline.flows_primary.get_amundi_lyxi_primary_flow", return_value=None), \
         patch("src.pipeline.flows_primary.get_qqq_sec_primary_flow", return_value=None), \
         patch("src.pipeline.flows_primary.get_cftc_position_flow_data", return_value=None), \
         patch("src.pipeline.flows_primary.compute_sector_flow_characteristics") as mock_sfc:
        fp.compute_flows_primary(pd.DataFrame())
    mock_sfc.assert_not_called()