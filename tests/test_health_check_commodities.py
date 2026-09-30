# -*- coding: utf-8 -*-
"""Tests de check_commodities_staleness (F3-05-bis, 2026-09-30).

Contexto: commodities_futures lleva congelado desde 2026-09-21 por
bloqueo del plan OilPriceAPI. Yahoo cubre BZ=F/CL=F frescos en
market_data. El check debe emitir SKIP (no WARN) mientras siga la
condicion, y volver a OK si el parquet se descongela.

Sin red. Manifests sinteticos en tmp_path.
"""
import json
from datetime import date

import scripts.health_check as hc
from scripts.health_check import (
    OK,
    SKIP,
    WARN,
    check_commodities_staleness,
)


def _write_manifest(root, name, last_date_iso):
    d = root / "data"
    d.mkdir(exist_ok=True)
    m = {"quality": {"last_date": last_date_iso}}
    (d / f"{name}.parquet.manifest.json").write_text(json.dumps(m))


def _patch_env(tmp_path, monkeypatch, today):
    monkeypatch.setattr(hc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "src.market_calendar.last_expected_market_date",
        lambda *a, **kw: today,
    )


def test_spot_stale_emite_warn(tmp_path, monkeypatch):
    """Spot stale -> WARN (no bloqueado). Invariante del comportamiento
    existente: el fix no debe tocar el caso no-bloqueado.
    """
    _patch_env(tmp_path, monkeypatch, date(2026, 9, 30))
    _write_manifest(tmp_path, "commodities_spot", "2026-09-15")   # 15d
    _write_manifest(tmp_path, "commodities_futures", "2026-09-29")  # OK
    results = check_commodities_staleness()
    spot = [r for r in results if r.name == "staleness:commodities_spot"][0]
    assert spot.status == WARN


def test_futures_stale_blocked_emite_skip(tmp_path, monkeypatch):
    """Futures stale + blocked_by_plan -> SKIP (no WARN).

    Test fuerte: sin el fix, este caso emite WARN.
    """
    _patch_env(tmp_path, monkeypatch, date(2026, 9, 30))
    _write_manifest(tmp_path, "commodities_spot", "2026-09-29")
    _write_manifest(tmp_path, "commodities_futures", "2026-09-21")  # 9d
    results = check_commodities_staleness()
    fut = [r for r in results if r.name == "staleness:commodities_futures"][0]
    assert fut.status == SKIP
    assert "BLOCKED" in fut.message


def test_futures_fresco_emite_ok_aunque_blocked(tmp_path, monkeypatch):
    """Futures fresco -> OK aunque blocked_by_plan=True.

    Invariante contra auto-ceguera: si el parquet se descongela, el
    check lo ve y no queda permanentemente en SKIP.
    """
    _patch_env(tmp_path, monkeypatch, date(2026, 9, 30))
    _write_manifest(tmp_path, "commodities_spot", "2026-09-29")
    _write_manifest(tmp_path, "commodities_futures", "2026-09-29")  # 1d
    results = check_commodities_staleness()
    fut = [r for r in results if r.name == "staleness:commodities_futures"][0]
    assert fut.status == OK
