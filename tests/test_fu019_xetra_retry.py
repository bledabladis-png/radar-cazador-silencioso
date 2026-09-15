# -*- coding: utf-8 -*-
"""Tests FU-019: retry en _query_one."""
import json
from unittest.mock import MagicMock

import pytest

from data.providers.xetra_provider import XetraProvider


class FakeTime:
    """Reloj simulado. Reemplaza a `time` dentro del modulo xetra_provider
    sin tocar el modulo `time` global del proceso."""
    def __init__(self, step=0.6):
        self.t = 0.0
        self.step = step
    def time(self):
        v = self.t
        self.t += self.step
        return v
    def sleep(self, secs):
        self.t += secs


@pytest.fixture
def provider():
    return XetraProvider()


def test_query_one_returns_on_first_attempt(provider):
    ws = MagicMock()
    ws.recv.side_effect = [
        '{"requestId": "q1_a1", "dataTimeseries": [{"date": "2026-09-14"}]}',
        '{"requestId": "q1_a1", "isComplete": true}',
    ]
    out = provider._query_one(ws, "q1", "DE000BAY0017",
                              "2026-09-01T00:00:00Z", "2026-09-15T23:59:00Z",
                              max_attempts=2)
    assert len(out) == 1
    assert out[0]["date"] == "2026-09-14"
    assert ws.send.call_count == 1


def test_query_one_retries_on_empty_first_attempt(provider, monkeypatch):
    """Primer intento agota deadline; segundo OK."""
    ws = MagicMock()
    state = {"a2_count": 0}

    def recv_side_effect():
        rid = json.loads(ws.send.call_args_list[-1][0][0])["requestId"]
        if rid == "q1_a1":
            raise Exception("simulated timeout")
        if rid == "q1_a2":
            state["a2_count"] += 1
            if state["a2_count"] == 1:
                return ('{"requestId": "q1_a2", '
                        '"dataTimeseries": [{"date": "2026-09-14"}]}')
            return '{"requestId": "q1_a2", "isComplete": true}'
        raise Exception(f"unexpected rid {rid}")

    ws.recv.side_effect = recv_side_effect

    import data.providers.xetra_provider as xp
    fake = FakeTime(step=0.6)
    monkeypatch.setattr(xp, "time", fake)
    monkeypatch.setattr(xp, "WS_QUERY_TIMEOUT", 1)

    out = provider._query_one(ws, "q1", "DE000BAY0017",
                              "2026-09-01T00:00:00Z", "2026-09-15T23:59:00Z",
                              max_attempts=2)
    assert len(out) == 1
    assert out[0]["date"] == "2026-09-14"
    assert ws.send.call_count == 2


def test_query_one_exhausts_attempts(provider, monkeypatch):
    """Ambos intentos agotan deadline -> lista vacia."""
    ws = MagicMock()
    ws.recv.side_effect = Exception("always timeout")

    import data.providers.xetra_provider as xp
    fake = FakeTime(step=0.6)
    monkeypatch.setattr(xp, "time", fake)
    monkeypatch.setattr(xp, "WS_QUERY_TIMEOUT", 1)

    out = provider._query_one(ws, "q1", "DE000BAY0017",
                              "2026-09-01T00:00:00Z", "2026-09-15T23:59:00Z",
                              max_attempts=2)
    assert out == []


def test_query_one_uses_unique_request_ids(provider, monkeypatch):
    """Cada intento usa requestId distinto."""
    ws = MagicMock()
    ws.recv.side_effect = Exception("timeout")

    import data.providers.xetra_provider as xp
    fake = FakeTime(step=0.6)
    monkeypatch.setattr(xp, "time", fake)
    monkeypatch.setattr(xp, "WS_QUERY_TIMEOUT", 1)

    provider._query_one(ws, "q1", "ISIN", "start", "end", max_attempts=2)

    sent_ids = [json.loads(c[0][0])["requestId"]
                for c in ws.send.call_args_list]
    assert sent_ids == ["q1_a1", "q1_a2"]
