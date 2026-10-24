from __future__ import annotations

from pathlib import Path

import pytest
import pandas as pd

from indicators.wyckoff_distribution import (
    ClassifiedBreak, classify_distribution_at, classify_distribution,
)


VALID_CLASSIFICATIONS = {
    "DISTRIBUTION_CONFIRMED",
    "REDISTRIBUTION_CONFIRMED",
    "BREAK_WITHOUT_DISTRIBUTION_CONTEXT",
    "BREAK_PENDING",
    "RANGE_ACTIVE_NO_BREAK",
    "NOT_EVALUABLE",
}

VALID_STRUCTURAL = {
    "RANGE_ACTIVE", "RANGE_BROKEN", "NOT_EVALUABLE",
}


def _check_invariants(r: ClassifiedBreak):
    """Verifica invariantes estructurales del ClassifiedBreak. Devuelve
    lista de violaciones (vacia si todo OK)."""
    v = []

    if r.classification not in VALID_CLASSIFICATIONS:
        v.append(f"classification invalida: {r.classification}")
    if r.structural_state not in VALID_STRUCTURAL:
        v.append(f"structural_state invalido: {r.structural_state}")

    if r.classification == "DISTRIBUTION_CONFIRMED":
        if r.range_type != "DISTRIBUTION_CANDIDATE":
            v.append(f"DIST_CONFIRMED con range_type={r.range_type}")
        if r.trend_direction != "UP":
            v.append(f"DIST_CONFIRMED con trend={r.trend_direction}")
        if r.structural_state != "RANGE_BROKEN":
            v.append(f"DIST_CONFIRMED con structural={r.structural_state}")
        if r.break_direction != "DOWN":
            v.append(f"DIST_CONFIRMED con direction={r.break_direction}")
        if r.coherence_error is not None:
            v.append(f"DIST_CONFIRMED con coherence_error={r.coherence_error}")
        if r.sow_status != "CONFIRMED":
            v.append(f"DIST_CONFIRMED con sow_status={r.sow_status}")

    if r.classification == "REDISTRIBUTION_CONFIRMED":
        if r.range_type != "REDISTRIBUTION_CANDIDATE":
            v.append(f"REDIST_CONFIRMED con range_type={r.range_type}")
        if r.trend_direction != "DOWN":
            v.append(f"REDIST_CONFIRMED con trend={r.trend_direction}")
        if r.structural_state != "RANGE_BROKEN":
            v.append(f"REDIST_CONFIRMED con structural={r.structural_state}")
        if r.break_direction != "DOWN":
            v.append(f"REDIST_CONFIRMED con direction={r.break_direction}")
        if r.coherence_error is not None:
            v.append(f"REDIST_CONFIRMED con coherence_error={r.coherence_error}")

    if r.classification == "BREAK_WITHOUT_DISTRIBUTION_CONTEXT":
        if r.structural_state != "RANGE_BROKEN":
            v.append(f"BREAK_WITHOUT con structural={r.structural_state}")
        if r.break_direction != "DOWN":
            v.append(f"BREAK_WITHOUT con direction={r.break_direction}")
        if r.sow_status != "CONFIRMED":
            v.append(f"BREAK_WITHOUT con sow_status={r.sow_status}")
        # Viene de dos causas: (a) UNCLASSIFIED sin coherencia rota,
        # (b) contradiccion range_type/trend con coherence_error
        if r.range_type == "UNCLASSIFIED":
            if r.coherence_error is not None:
                v.append(f"BREAK_WITHOUT con UNCLASSIFIED y coherence_error={r.coherence_error}")
        else:
            # range_type DISTRIBUTION/REDISTRIBUTION pero trend incoherente
            if r.coherence_error is None:
                v.append(f"BREAK_WITHOUT con range_type={r.range_type} sin coherence_error")

    if r.classification == "BREAK_PENDING":
        if r.structural_state != "RANGE_ACTIVE":
            v.append(f"BREAK_PENDING con structural={r.structural_state}")
        if r.sow_status != "CANDIDATE":
            v.append(f"BREAK_PENDING con sow_status={r.sow_status}")

    if r.classification == "RANGE_ACTIVE_NO_BREAK":
        if r.structural_state != "RANGE_ACTIVE":
            v.append(f"RANGE_ACTIVE_NO_BREAK con structural={r.structural_state}")
        if r.sow_status not in (None, "NOT_OBSERVED", "REJECTED", "EXPIRED", "NOT_EVALUABLE"):
            v.append(f"RANGE_ACTIVE_NO_BREAK con sow_status={r.sow_status}")

    return v


@pytest.mark.integration_real_data
def test_invariants_on_real_315_tickers():
    p = Path("data/stock_prices_extended.parquet")
    if not p.exists():
        pytest.skip("parquet no disponible")
    df = pd.read_parquet(p)
    tickers = sorted(df.columns.get_level_values(1).unique().tolist())

    violations = []
    n_checked = 0
    for tk in tickers:
        try:
            close = df[("Close", tk)].dropna()
        except (KeyError, ValueError):
            continue
        if len(close) < 300:
            continue
        as_of = close.index[-1]
        try:
            r = classify_distribution_at(df, tk, as_of)
        except Exception as exc:
            violations.append((tk, f"exception: {type(exc).__name__}: {exc}"))
            continue
        vs = _check_invariants(r)
        if vs:
            violations.append((tk, "; ".join(vs)))
        n_checked += 1

    assert n_checked >= 300, f"solo {n_checked} tickers auditados"
    assert not violations, (
        f"{len(violations)} tickers violan invariantes:\n" +
        "\n".join(f"  {tk}: {msg}" for tk, msg in violations[:20])
    )


@pytest.mark.integration_real_data
def test_reproducibility_real_acs_mc():
    """Dos llamadas identicas producen el mismo output."""
    p = Path("data/stock_prices_extended.parquet")
    if not p.exists():
        pytest.skip("parquet no disponible")
    df = pd.read_parquet(p)
    if "ACS.MC" not in df.columns.get_level_values(1):
        pytest.skip("ACS.MC no en parquet")
    close = df[("Close", "ACS.MC")].dropna()
    as_of = close.index[-1]

    r1 = classify_distribution_at(df, "ACS.MC", as_of)
    r2 = classify_distribution_at(df, "ACS.MC", as_of)

    assert r1.classification == r2.classification
    assert r1.structural_state == r2.structural_state
    assert r1.range_type == r2.range_type
    assert r1.trend_direction == r2.trend_direction
    assert r1.sow_status == r2.sow_status
    assert r1.coherence_error == r2.coherence_error
    # evidence puede contener floats; comparar los campos clave
    for k in ("bc_date", "ar_date", "range_active_at",
              "sow_event_date", "trend_direction"):
        assert r1.evidence.get(k) == r2.evidence.get(k), f"difiere en {k}"


@pytest.mark.integration_real_data
def test_causality_bc_stable_over_time():
    """El BC detectado en t1 debe ser el mismo en t2>t1 mientras no aparezca
    un BC mas reciente posterior a t1."""
    p = Path("data/stock_prices_extended.parquet")
    if not p.exists():
        pytest.skip("parquet no disponible")
    df = pd.read_parquet(p)
    ticker = "ACS.MC"
    if ticker not in df.columns.get_level_values(1):
        pytest.skip(f"{ticker} no en parquet")
    close = df[("Close", ticker)].dropna()

    # Tomar posicion ~250 barras antes del final, y el final
    pos_a = len(close) - 250
    pos_b = len(close) - 1
    as_of_a = close.index[pos_a]
    as_of_b = close.index[pos_b]

    r_a = classify_distribution_at(df, ticker, as_of_a)
    r_b = classify_distribution_at(df, ticker, as_of_b)

    # Si ambos tienen BC, y no aparecio uno nuevo entre a y b, el BC debe coincidir
    bc_a = r_a.evidence.get("bc_date")
    bc_b = r_b.evidence.get("bc_date")
    if bc_a is None or bc_b is None:
        pytest.skip(f"BC no detectado en a o b (a={r_a.classification}, b={r_b.classification})")
    if bc_a == bc_b:
        # OK: BC estable
        return
    # BC cambio: debe ser porque aparecio uno mas reciente DESPUES de as_of_a
    assert bc_b > bc_a, f"BC en b={bc_b} es anterior a BC en a={bc_a}: retroceso temporal"


def test_snapshot_vs_wrapper_same_result():
    """classify_distribution(df, tk) == classify_distribution_at(df, tk, df.index[-1])."""
    df_p = Path("data/stock_prices_extended.parquet")
    if not df_p.exists():
        pytest.skip("parquet no disponible")
    df = pd.read_parquet(df_p)
    if "ACS.MC" not in df.columns.get_level_values(1):
        pytest.skip("ACS.MC no en parquet")
    r1 = classify_distribution(df, "ACS.MC")
    r2 = classify_distribution_at(df, "ACS.MC", df.index[-1])
    assert r1.classification == r2.classification
    assert r1.as_of == r2.as_of