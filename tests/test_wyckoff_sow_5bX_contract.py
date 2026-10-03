# -*- coding: utf-8 -*-
"""Tests contractuales del validador 5b.X v3.

Contrato: docs/auditoria/wyckoff/35_protocolo_5bX_v3.md seccion 16.
Contexto: D-06 (docs/auditoria/wyckoff/37_d06_5bX_invalidada.md).

Estos tests verifican INVARIANTES del contrato, no resultados plausibles.
Los 9 puntos exigidos por §16 del v3:

    T1  Rechazo por cutoff (t0 < OOS_T0_MIN, no L >= OOS_T0_MIN).
    T2  Bloqueo por muestra (n_confirmed_H20 < 550 -> escenario D).
    T3  Escenario A (lower_ci > 0).
    T4  Escenario B (IC cruza 0), incluido el caso testigo D-06.4
        con lift_point < 0.
    T5  Escenario C (upper_ci < 0).
    T6  Elegibilidad H20_complete (L_idx + 20 < len(struct)).
    T7  Verificacion de hash pre-ejecucion (skip: manifiesto PENDIENTE).
    T8  Sin grid (AST sobre el script).
    T9  Determinismo del bootstrap (misma seed -> mismo output).

Sin datos reales. Sin red. DataFrame sintetico con seed fijo.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import validate_wyckoff_sow_5bX as v5bX


# =====================================================================
# Fixtures sinteticas
# =====================================================================

def _make_feats(n=300, tk="AAA"):
    """feats dict con estructura minima para episode_metrics.

    No reproduce el calibrador; solo provee los campos que
    collect_validation_rows y episode_metrics leen.
    """
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    struct = pd.Series(np.linspace(0.6, 0.4, n), index=dates)
    close = pd.Series(np.linspace(100.0, 80.0, n), index=dates)
    low = close - 0.5
    volume = pd.Series(1_000_000.0, index=dates)
    return {tk: {
        "dates": dates, "struct": struct, "close": close,
        "low": low, "volume": volume,
    }}


def _make_ep(tk, t0_date_str, L_idx, is_conf=True):
    """Episodio minimo. t0_date explicito (independiente de L_idx)."""
    t0 = pd.Timestamp(t0_date_str)
    return {
        "ticker": tk,
        "t0_idx": 0,
        "t0_date": t0,
        "L_idx": L_idx,
        "L_date": t0 + pd.Timedelta(days=45),
        "sow_idx": None,
        "is_confirmed": is_conf,
        "block_L": "P1",
    }


def _make_boot(lift_point, lower_ci, upper_ci):
    return {
        "lift_point": lift_point,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
    }


# =====================================================================
# Constantes normativas (guardia contra cambio de umbrales)
# =====================================================================

def test_constantes_normativas_v3():
    """Los umbrales del contrato v3 estan congelados y no son revisables."""
    assert v5bX.MIN_N_CONFIRMED_H20 == 550
    assert v5bX.OOS_T0_MIN == pd.Timestamp("2026-10-02")
    assert v5bX.CUTOFF_DATE == pd.Timestamp("2026-10-01")


# =====================================================================
# T1 - Rechazo por cutoff (D-06.2)
# =====================================================================

def test_t1_rechazo_por_cutoff_pre_oos():
    """t0 < OOS_T0_MIN descarta el episodio aunque L sea posterior.

    Caso material de D-06.2: v1 usaba L_date > cutoff, permitiendo
    episodios iniciados antes del cutoff cuya ventana retrospectiva
    participa del desarrollo. v3 exige t0 >= OOS_T0_MIN.
    """
    feats = _make_feats(300)
    ep_pre = _make_ep("AAA", "2026-09-15", L_idx=50, is_conf=True)
    ep_oos = _make_ep("AAA", "2026-10-15", L_idx=100, is_conf=True)
    rows, n_oos, n_h20 = v5bX.collect_validation_rows(
        [ep_pre, ep_oos], feats, N=60, t0_min=v5bX.OOS_T0_MIN)
    assert len(rows) == 1, "pre-OOS deberia descartarse"
    assert n_oos == 1
    assert n_h20 == 0


def test_t1bis_episodio_exactamente_en_cutoff():
    """t0 == OOS_T0_MIN cuenta como OOS (>=, no >)."""
    feats = _make_feats(300)
    ep = _make_ep("AAA", "2026-10-02", L_idx=100, is_conf=True)
    rows, n_oos, _ = v5bX.collect_validation_rows(
        [ep], feats, N=60, t0_min=v5bX.OOS_T0_MIN)
    assert len(rows) == 1
    assert n_oos == 0


# =====================================================================
# T6 - Elegibilidad H20_complete (D-06.5)
# =====================================================================

def test_t6_h20_censurado_no_cuenta():
    """L_idx + 20 >= len(struct) descarta el episodio.

    Caso material de D-06.5: v1 contaba cualquier episodio con
    landmark existente. v3 exige H20 completo.
    """
    feats = _make_feats(300)
    ep_cens = _make_ep("AAA", "2026-10-15", L_idx=290, is_conf=True)
    ep_ok = _make_ep("AAA", "2026-10-15", L_idx=100, is_conf=True)
    rows, n_oos, n_h20 = v5bX.collect_validation_rows(
        [ep_cens, ep_ok], feats, N=60, t0_min=v5bX.OOS_T0_MIN)
    assert len(rows) == 1
    assert n_h20 == 1
    assert n_oos == 0


# =====================================================================
# T2 - Bloqueo por muestra (n_confirmed_H20 < 550 -> D)
# =====================================================================

def test_t2_bootstrap_vacio_devuelve_d():
    """Sin muestra, el IC no es calculable -> escenario D."""
    boot = v5bX.bootstrap_lift(list())
    esc, _ = v5bX.classify_scenario(boot)
    assert esc == "D"


# =====================================================================
# T3 - Escenario A
# =====================================================================

def test_t3_escenario_a():
    boot = _make_boot(0.1, 0.02, 0.2)
    esc, _ = v5bX.classify_scenario(boot)
    assert esc == "A"


# =====================================================================
# T4 - Escenario B (IC cruza 0)
# =====================================================================

def test_t4_escenario_b_ic_cruza_cero():
    boot = _make_boot(0.05, -0.02, 0.1)
    assert v5bX.classify_scenario(boot)[0] == "B"


def test_t4_d06_caso_testigo_lift_negativo_ic_cruza():
    """D-06.4: v1 clasificaba como C; v3 debe devolver B.

    Caso testigo del protocolo v3 §6.3:
        lift_point = -0.5, IC95% = [-1.2, +0.2]
    El IC cruza 0 -> B. El signo de lift_point es irrelevante.
    """
    boot = _make_boot(-0.5, -1.2, 0.2)
    assert v5bX.classify_scenario(boot)[0] == "B"


# =====================================================================
# T5 - Escenario C
# =====================================================================

def test_t5_escenario_c():
    boot = _make_boot(-0.05, -0.15, -0.01)
    assert v5bX.classify_scenario(boot)[0] == "C"


# =====================================================================
# T7 - Verificacion de hash pre-ejecucion
# =====================================================================

def test_t7_hash_freeze_script_coincide():
    """El sha256 del script coincide con el manifiesto de freeze.

    §2.5 del v3: la ejecucion aborta si el hash del script difiere
    del congelado. Este test verifica la coherencia. Si el script se
    modifica sin actualizar el manifiesto, este test falla: esa es la
    senal de que el freeze queda invalidado.
    """
    import hashlib
    import json
    mf_path = (ROOT / "docs" / "auditoria" / "wyckoff"
               / "35_protocolo_5bX_v3.freeze.json")
    if not mf_path.exists():
        pytest.skip("Manifiesto de freeze aun no existe")
    mf = json.loads(mf_path.read_text(encoding="utf-8"))
    script_path = ROOT / "scripts" / "validate_wyckoff_sow_5bX.py"
    h = hashlib.sha256(script_path.read_bytes()).hexdigest()
    assert h == mf["script"]["sha256"].lower(), (
        "sha256 del script no coincide con el manifiesto\n"
        f"  script:     {h}\n"
        f"  manifiesto: {mf['script']['sha256']}"
    )


def test_t7b_revision_auditor_valor_valido():
    """El campo revision_auditor solo admite PENDIENTE o FIRMADO.

    §15 del v3: la validacion no se ejecuta hasta que sea FIRMADO.
    Este test garantiza que no se cambia silenciosamente a otro valor
    ni se olvida actualizar el manifiesto.
    """
    import json
    mf_path = (ROOT / "docs" / "auditoria" / "wyckoff"
               / "35_protocolo_5bX_v3.freeze.json")
    if not mf_path.exists():
        pytest.skip("Manifiesto de freeze aun no existe")
    mf = json.loads(mf_path.read_text(encoding="utf-8"))
    assert mf["revision_auditor"] in ("PENDIENTE", "FIRMADO"), (
        f"valor inesperado: {mf['revision_auditor']}"
    )


# =====================================================================
# T8 - Sin grid
# =====================================================================

def test_t8_sin_grid():
    """El validador aplica la candidata una sola vez; cero grid.

    §2.4 del v3: no se exploran parametros. Este test verifica que
    el script no referencia grid como identificador, atributo o
    import. Los strings (p. ej. "NO explora grid" en el docstring)
    no cuentan: son texto, no codigo.

    Señales que harian fallar este test:
      - `run_grid(...)` o cualquier funcion/metodo con "grid" en el
        nombre.
      - `from scripts.calibrate_wyckoff_sow_5b4bis import run_grid`.
      - `itertools.product(...)` sobre parametros N/M/X/Y.
    """
    path = ROOT / "scripts" / "validate_wyckoff_sow_5bX.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    refs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and "grid" in node.id.lower():
            refs.append(("Name", node.id, node.lineno))
        if isinstance(node, ast.Attribute) and "grid" in node.attr.lower():
            refs.append(("Attribute", node.attr, node.lineno))
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if "grid" in alias.name.lower():
                    refs.append(("ImportFrom", alias.name, node.lineno))
    assert not refs, f"referencias a grid: {refs}"


# =====================================================================
# T9 - Determinismo
# =====================================================================

def test_t9_bootstrap_determinista():
    """Misma seed -> mismo output. Requisito de reproducibilidad."""
    rows = [
        ("AAA", True, {"struct_deterioration": 1}),
        ("AAA", False, {"struct_deterioration": 0}),
        ("BBB", True, {"struct_deterioration": 1}),
        ("BBB", False, {"struct_deterioration": 1}),
        ("CCC", True, {"struct_deterioration": 0}),
        ("CCC", False, {"struct_deterioration": 0}),
    ]
    b1 = v5bX.bootstrap_lift(rows, B=200, seed=20261002)
    b2 = v5bX.bootstrap_lift(rows, B=200, seed=20261002)
    assert b1 == b2
    assert b1["lift_point"] is not None
    assert b1["lower_ci"] is not None


def test_t9_classify_determinista():
    """classify_scenario es puro: mismo input -> mismo output."""
    boot = _make_boot(0.03, -0.01, 0.07)
    assert v5bX.classify_scenario(boot) == v5bX.classify_scenario(boot)


# =====================================================================
# Invariante derivada: la clasificacion D no es resultado
# =====================================================================

def test_d_no_es_a_b_c():
    """D nunca se devuelve con un IC calculable (§6.2 v3)."""
    for lift in (-1.0, -0.1, 0.0, 0.1, 1.0):
        for lo, hi in [(-0.5, 0.5), (-0.2, -0.1), (0.1, 0.2)]:
            boot = _make_boot(lift, lo, hi)
            esc = v5bX.classify_scenario(boot)[0]
            assert esc in ("A", "B", "C"), (
                f"lift={lift}, IC=[{lo},{hi}] -> {esc}"
            )



# =====================================================================
# C1 - Equivalencia funcional del bootstrap local vs calibrador
# (condicion de firma del auditor, 2026-10-03)
# =====================================================================

def test_c1_bootstrap_equivalencia_funcional():
    """El bootstrap local y el del calibrador producen el mismo output.

    Condicion C1 del dictamen de conformidad: para una fixture
    determinista comun (mismos episodios, misma B, misma seed),
    bootstrap_lift local == bootstrap_lift_H20 del calibrador en
    lift_point, lower_ci, upper_ci, n_confirmed, n_baseline, n_boot.

    No se importa el del calibrador como sustituto: se compara.
    """
    from scripts.calibrate_wyckoff_sow_5b4bis import bootstrap_lift_H20

    rows = [
        ("AAA", True, {"struct_deterioration": 1}),
        ("AAA", False, {"struct_deterioration": 0}),
        ("AAA", True, {"struct_deterioration": 1}),
        ("BBB", True, {"struct_deterioration": 0}),
        ("BBB", False, {"struct_deterioration": 1}),
        ("CCC", True, {"struct_deterioration": 0}),
        ("CCC", False, {"struct_deterioration": 0}),
        ("CCC", False, {"struct_deterioration": 1}),
        ("DDD", True, {"struct_deterioration": 1}),
        ("DDD", False, {"struct_deterioration": 0}),
        ("EEE", True, {"struct_deterioration": 1}),
        ("EEE", False, {"struct_deterioration": 1}),
    ]
    B = 500
    seed = 20261002

    ref = bootstrap_lift_H20(rows, B=B, seed=seed)
    loc = v5bX.bootstrap_lift(rows, B=B, seed=seed)

    assert ref["lift_point"] == loc["lift_point"], (
        f"lift_point: ref={ref['lift_point']} loc={loc['lift_point']}"
    )
    assert ref["lower_ci"] == loc["lower_ci"], (
        f"lower_ci: ref={ref['lower_ci']} loc={loc['lower_ci']}"
    )
    assert ref["upper_ci"] == loc["upper_ci"], (
        f"upper_ci: ref={ref['upper_ci']} loc={loc['upper_ci']}"
    )
    assert ref["n_confirmed"] == loc["n_confirmed"]
    assert ref["n_baseline"] == loc["n_baseline"]
    assert ref["n_boot_validos"] == loc["n_boot"]


# =====================================================================
# C2 - Guardrails informativos no bloquean
# (condicion de firma del auditor, 2026-10-03)
# =====================================================================

def _make_feats_con_candidate(dates, tk="AAA", candidate_value=False):
    """feats dict con candidate constante (todo False por defecto)."""
    candidate = pd.Series(candidate_value, index=dates)
    struct = pd.Series(0.5, index=dates)
    close = pd.Series(100.0, index=dates)
    low = close - 0.5
    return {tk: {
        "dates": dates, "struct": struct, "close": close,
        "low": low, "volume": pd.Series(1000.0, index=dates),
        "candidate": candidate,
    }}


def test_c2a_menos_12_meses_y_sin_starts_no_bloquea():
    """3 meses post-cutoff + 0 candidate starts -> ok=True.

    Condicion C2 del dictamen: 12m y 50 starts son informativos,
    no bloqueantes. Con datos OOS existentes, no se bloquea aunque
    no se cumplan los guardrails.
    """
    dates = pd.date_range("2026-10-02", periods=60, freq="B")
    df = pd.DataFrame({"Close": 100.0}, index=dates)
    feats = _make_feats_con_candidate(dates, candidate_value=False)
    precond = v5bX.check_preconditions(df, ["AAA"], feats)
    assert precond["guardrails_ok"] is False, "se esperaba guardrails no cumplidos"
    assert precond["n_candidate_starts_oos"] == 0
    assert precond["ok"] is True, (
        "12m y 50 starts NO deben bloquear. "
        f"reasons={precond['reasons']}"
    )


def test_c2b_mas_12_meses_y_sin_starts_no_bloquea():
    """15 meses post-cutoff + 0 starts -> ok=True. Solo falla el guardrail
    de starts, y eso no bloquea.
    """
    dates = pd.date_range("2026-10-02", periods=320, freq="B")
    df = pd.DataFrame({"Close": 100.0}, index=dates)
    feats = _make_feats_con_candidate(dates, candidate_value=False)
    precond = v5bX.check_preconditions(df, ["AAA"], feats)
    assert precond["guardrails_ok"] is False
    assert precond["ok"] is True


def test_c2c_sin_sesiones_oos_si_bloquea():
    """Sin sesiones post-cutoff -> ok=False. Unico bloqueo valido por
    precondiciones (fuera del 550, que se comprueba despues).
    """
    dates = pd.date_range("2026-01-01", "2026-09-30", freq="B")
    df = pd.DataFrame({"Close": 100.0}, index=dates)
    feats = _make_feats_con_candidate(dates, candidate_value=False)
    precond = v5bX.check_preconditions(df, ["AAA"], feats)
    assert precond["ok"] is False
    assert any("sesiones" in r for r in precond["reasons"])