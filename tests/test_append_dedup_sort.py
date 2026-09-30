# -*- coding: utf-8 -*-
"""Fix E: append_dedup debe ordenar tras el dedup.

Bug detectado 2026-09-30: cuando el CSV historico tiene una fila
mas reciente que el proveedor (lag de publicacion), append_dedup
deja esa fila ARRIBA y el resto debajo. Esto rompe consumidores
que leen .iloc[-1].

Caso real: amundi_lyxi_primary_flow.csv. CSV con fila 2026-09-29,
JSON del proveedor sin esa fecha. Resultado: 2026-09-29 en posicion 0.
El render hace .iloc[-1] y publica 2026-09-28.
"""
import pandas as pd

from src.utils import append_dedup


def test_append_dedup_sin_sort_preserva_comportamiento():
    """Sin sort_by, el comportamiento no cambia (backward-compat)."""
    hist = pd.DataFrame({
        "date": ["2018-01-02", "2026-09-28", "2026-09-29"],
        "v": [1, 2, 3],
    })
    # Proveedor sin la fila 29
    nuevo = pd.DataFrame({
        "date": ["2018-01-02", "2026-09-28"],
        "v": [1, 2],
    })
    out = append_dedup(hist, nuevo, ["date"])
    # El comportamiento actual: la fila 29 queda arriba (bug)
    # Este test documenta el comportamiento por defecto
    assert len(out) == 3


def test_append_dedup_con_sort_by_ordena_ascendente():
    """Con sort_by, el resultado sale ordenado por esa columna."""
    hist = pd.DataFrame({
        "date": ["2018-01-02", "2026-09-28", "2026-09-29"],
        "v": [1, 2, 3],
    })
    nuevo = pd.DataFrame({
        "date": ["2018-01-02", "2026-09-28"],
        "v": [1, 2],
    })
    out = append_dedup(hist, nuevo, ["date"], sort_by=["date"])
    assert out["date"].is_monotonic_increasing, (
        f"Esperado orden ascendente. Obtenido: {out['date'].tolist()}"
    )
    assert out["date"].iloc[-1] == "2026-09-29", (
        f"La fila mas reciente debe ser la ultima. "
        f"Ultimas 3: {out['date'].tail(3).tolist()}"
    )


def test_append_dedup_sort_by_multiples_columnas():
    """sort_by con varias columnas (ticker, Date)."""
    hist = pd.DataFrame({
        "ticker": ["XLK", "XLK", "XLB"],
        "Date": ["2026-09-25", "2026-09-29", "2026-09-28"],
        "v": [1, 2, 3],
    })
    nuevo = pd.DataFrame({
        "ticker": ["XLK"],
        "Date": ["2026-09-29"],
        "v": [99],
    })
    out = append_dedup(hist, nuevo, ["ticker", "Date"], sort_by=["ticker", "Date"])
    # Verificar orden por (ticker, Date)
    assert out["ticker"].tolist() == ["XLB", "XLK", "XLK"]
    assert out["Date"].tolist() == ["2026-09-28", "2026-09-25", "2026-09-29"]
    # La fila del XLK/2026-09-29 debe ser la ultima
    assert out.iloc[-1]["Date"] == "2026-09-29"