# -*- coding: utf-8 -*-
"""D-03: compute_cmf con volumen 0 en ventana -> division por cero."""
import numpy as np
import pandas as pd

from indicators.momentum import compute_cmf


def test_compute_cmf_volumen_cero_no_produce_nan_ni_inf():
    """Con volumen 0 en toda la ventana, el denominador es 0.

    Bug: mfv.rolling(w).sum() / volume.rolling(w).sum() sin guard
    -> 0/0 = NaN silencioso (o +-inf). El resto del fichero usa
    +1e-9 (momentum_score:28). Fix: +1e-9 en denominador -> cmf=0.
    """
    n = 30
    df = pd.DataFrame({
        "High":  np.linspace(100, 101, n),
        "Low":   np.linspace(99, 100, n),
        "Close": np.linspace(99.5, 100.5, n),
        "Volume": np.zeros(n),
    })
    cmf = compute_cmf(df, "SYNTH", window=20)
    finitos = cmf.dropna()
    assert len(finitos) > 0, (
        "cmf es todo NaN cuando volumen=0; el consumidor no recibe senal "
        "de disponibilidad (deberia ser 0.0 con guard +1e-9)"
    )
    assert np.all(np.isfinite(finitos)), (
        f"cmf contiene inf/-inf: {finitos[finitos.apply(lambda x: not np.isfinite(x))].to_dict()}"
    )
