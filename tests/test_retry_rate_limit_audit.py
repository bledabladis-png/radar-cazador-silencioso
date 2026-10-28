"""Tests del fix de rate limiting (rate-limit-audit 2026-10-28).

2 bugs:

1. `retry_utils.retry_call` con `time.sleep(backoff * attempt)` daba
   sleep(0) en el primer retry (attempt=0). Primer reintento inmediato.
   Con un 429 de Yahoo, agrava el bloqueo. Fix: backoff * (attempt + 1).

2. `stock_data_loader.py` retry individual sin sleep entre tickers
   fallidos. Si 100 tickers fallan en batch, 100 requests single-ticker
   back-to-back. Rate-limit Yahoo garantizado. Fix: time.sleep(delay)
   al final de cada iteracion del retry loop.
"""
from __future__ import annotations

import ast
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.providers.retry_utils import retry_call


def test_retry_call_primer_sleep_no_es_cero(monkeypatch):
    """El primer reintento NO debe dormir 0s. backoff*(0+1)=backoff."""
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    # Funcion que falla 2 veces y luego devuelve 42
    calls = {"n": 0}
    def _flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError(f"fallo {calls['n']}")
        return 42

    out = retry_call(_flaky, retries=3, backoff=2)
    assert out == 42
    # 2 sleeps: tras intento 1 y 2. El 3 tiene exito, no duerme.
    assert len(sleeps) == 2, f"esperados 2 sleeps, obtenidos {sleeps}"
    # backoff=2, retries=3: sleep en attempt=0 -> 2*(0+1)=2;
    # sleep en attempt=1 -> 2*(1+1)=4.
    assert sleeps[0] == 2, f"primer sleep deberia ser 2, es {sleeps[0]}"
    assert sleeps[1] == 4, f"segundo sleep deberia ser 4, es {sleeps[1]}"


def test_retry_call_ultimo_error_propaga(monkeypatch):
    """Si todos los intentos fallan, se propaga la ultima excepcion."""
    monkeypatch.setattr(time, "sleep", lambda s: None)

    def _siempre_falla():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        retry_call(_siempre_falla, retries=2, backoff=1)


def test_retry_call_no_duerme_ultimo_intento(monkeypatch):
    """A2.1-10: el ultimo intento no debe dormir (no hay siguiente)."""
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    def _siempre_falla():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        retry_call(_siempre_falla, retries=3, backoff=1)
    # 3 intentos, 3 fallos, solo 2 sleeps (tras 1 y 2).
    assert len(sleeps) == 2, f"esperados 2 sleeps, obtenidos {len(sleeps)}"


def test_stock_data_loader_sleep_entre_retries_individuales():
    """AST: el retry loop individual de stock_data_loader.py tiene
    time.sleep(delay) al final de cada iteracion (en finally, para
    cubrir tambien el continue)."""
    src = (ROOT / "src" / "stock_data_loader.py").read_text(
        encoding="utf-8-sig", errors="replace"
    )
    tree = ast.parse(src)
    # Localizar EXACTAMENTE `for ticker in failed_tickers[:]` o
    # `for ticker in failed_tickers`. No confundir con metodos que
    # contienen "failed_tickers" en el nombre.
    encontrado = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        # iter puede ser Subscript(failed_tickers, [:]) o Name(failed_tickers)
        target_iter = node.iter
        es_failed = False
        if isinstance(target_iter, ast.Subscript):
            base = target_iter.value
            if isinstance(base, ast.Name) and base.id == "failed_tickers":
                es_failed = True
        elif isinstance(target_iter, ast.Name):
            if target_iter.id == "failed_tickers":
                es_failed = True
        if not es_failed:
            continue
        # Buscar time.sleep dentro del cuerpo del for (incluye finally)
        tiene_sleep = False
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                if (isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "sleep"
                        and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id == "time"):
                    tiene_sleep = True
                    break
        assert tiene_sleep, (
            f"stock_data_loader.py:{node.lineno}: bucle sobre "
            f"failed_tickers SIN time.sleep. Rate-limit Yahoo."
        )
        encontrado = True
    assert encontrado, "no se encontro el bucle sobre failed_tickers"
