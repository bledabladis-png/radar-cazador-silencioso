# -*- coding: utf-8 -*-
"""Manifiesto y heuristica de contratos dict productor-consumidor.

Cierra el plan de cierre de la clase 1fb791ab (2026-10-30). Los 9
pares cubiertos por contrato AST o schema estan en DICT_CONTRACT_PAIRS.
Los pares identificados pero no cubiertos estan en
DICT_CONTRACT_WHITELIST con su razon.

Tests:
1. test_manifest_pares_tienen_test_correspondiente: cada entrada del
   manifiesto tiene un test en su fichero de contrato. Falla si
   alguien anade un par al manifiesto sin test.
2. test_whitelist_justificada: la whitelist no crece sin justificar.
3. test_heuristico_render_dict_detecta_nuevos_pares: enumera los
   render_*(X_data|section|result|synthesis|conflict) con clasificador
   df_attrs (distingue DataFrame de dict real). Falla si aparece un
   dict real sin manifiesto ni whitelist. DataFrames disfrazados no
   dan falsos positivos.

Limites conocidos:
- La heuristica no detecta pares donde el producer no es funcion
  local (API, JSON externo). Esos se whitelistean con razon.
- La clasificacion df_attrs es heuristica. Si aparece un falso
  positivo nuevo, anadir el atributo a DF_ATTRS en el clasificador.
"""
import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Manifiesto: cada par tiene test en tests/test_<X>_contract.py.
DICT_CONTRACT_PAIRS = {
    ("indicators/mte/engine.py", "compute_mte",
     "src/report/volatility_mte.py", "render_mte"):
        "tests/test_mte_return_contract.py",
    ("regimes/liquidity_v2.py", "compute_liquidity_score_v2",
     "src/report/header.py", "render_regimenes"):
        "tests/test_liquidity_state_contract.py",
    ("regimes/liquidity_v2.py", "compute_liquidity_score_v2",
     "src/report/freshness.py", "render_data_freshness"):
        "tests/test_liquidity_state_contract.py",
    ("src/pipeline/iae_section.py", "compute_iae_section",
     "src/report/iae.py", "render_iae_section"):
        "tests/test_iae_section_contract.py",
    ("indicators/darkpool.py", "compute_darkpool_signals",
     "src/report/darkpool.py", "render_darkpool"):
        "tests/test_darkpool_contract.py",
    ("src/pipeline/flows_secondary.py", "compute_flows_secondary",
     "src/report/flows_international.py", "render_flujo_sintesis"):
        "tests/test_flows_confirmation_contract.py",
    ("src/pipeline/mte_confirmation.py", "_compute_confirmation",
     "src/report/confirmation.py", "render_confirmation"):
        "tests/test_flows_confirmation_contract.py",
    ("src/utils.py", "detect_cross_module_conflict",
     "src/report/alerts.py", "render_cross_module"):
        "tests/test_cross_module_contract.py",
    # 2o consumidor del mismo confirmation_data. Lee solo ratios (ya
    # en CONFIRMATION_DATA_SCHEMA), para la alerta "Financial Stress
    # vs Credit" (HYG/LQD z-score). Cubierto por test en
    # test_flows_confirmation_contract.py.
    ("src/pipeline/mte_confirmation.py", "_compute_confirmation",
     "src/report/alerts.py", "render_alerts"):
        "tests/test_flows_confirmation_contract.py",
    # Cross-module: producer en indicators/options.py::compute_pcr_signals.
    # Antes en whitelist; cubierto por test_pcr_contract.py (2026-10-31).
    ("indicators/options.py", "compute_pcr_signals",
     "src/report/sentiment.py", "render_sentimiento_opciones"):
        "tests/test_pcr_contract.py",
}

# Pares identificados pero NO cubiertos en el plan 2026-10-30. Razon:
# productor en modulo distinto (indicators/options.py,
# indicators/sector_rank_score.py) que no ha pasado por el analisis
# de AST/schema del plan. Cubrir en sesion 20 si ROI.
DICT_CONTRACT_WHITELIST = frozenset({
    ("src/pipeline/sectors_base.py", "compute_sectors_base",
     "src/report/rankings.py", "render_ranking_sectorial_unificado",
     "sector_results nested de compute_sector_rank_score; no analizado "
     "en plan 2026-10-30"),
})

# Atributos que indican DataFrame (no dict). Distingue df.empty (attr)
# de df.empty() (call). Incluye los mas frecuentes; si un falso positivo
# aparece, anadir aqui.
DF_ATTRS = frozenset({
    "iterrows", "itertuples", "iloc", "loc", "empty", "columns",
    "dropna", "sort_values", "head", "tail", "copy", "notna",
    "groupby", "index", "values", "shape", "dtypes", "reset_index",
    "to_dict", "to_csv", "pivot", "unstack", "merge", "astype",
})


def test_manifest_pares_tienen_test_correspondiente():
    """Cada par del manifiesto tiene su fichero de test existente."""
    for pair, test_file in DICT_CONTRACT_PAIRS.items():
        p = REPO_ROOT / test_file
        assert p.exists(), (
            f"Manifiesto declara {pair} -> {test_file} pero el "
            f"fichero no existe. Anadir el test o retirar del "
            f"manifiesto."
        )


def test_whitelist_justificada():
    """La whitelist no crece sin justificacion (anti whitelist creep)."""
    # Si crece, actualizar este numero y justificar en commit.
    assert len(DICT_CONTRACT_WHITELIST) == 1, (
        f"La whitelist tiene {len(DICT_CONTRACT_WHITELIST)} pares "
        f"(esperado 1). Si es una decision consciente, actualizar "
        f"este test con el nuevo conteo y justificar en el mensaje "
        f"del commit. Si no, revisar si el nuevo par debe estar "
        f"cubierto por contrato AST/schema."
    )


def _find_function(tree, name):
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name):
            return node
    return None


def _classify_var(module_file, fn_name, var_name):
    """Clasifica el var como dict, df o desconocido.

    Regresa 'dict' si solo usa .get() y/o subscripts. 'df' si usa
    DF_ATTRS. 'mixto' si ambos. 'desconocido' si no accede.
    """
    p = REPO_ROOT / module_file
    if not p.exists():
        return "no_encontrado"
    tree = ast.parse(p.read_text(encoding="utf-8-sig"))
    fn = _find_function(tree, fn_name)
    if fn is None:
        return "fn_no_encontrada"
    dict_uses = 0
    df_uses = 0
    subscripts = 0
    for node in ast.walk(fn):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == var_name):
            if node.attr in DF_ATTRS:
                df_uses += 1
            if node.attr == "get":
                dict_uses += 1
        if (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == var_name
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            subscripts += 1
    if df_uses > 0 and dict_uses == 0:
        return "df"
    if dict_uses > 0 and df_uses == 0:
        return "dict"
    if df_uses > 0 and dict_uses > 0:
        return "mixto"
    if subscripts > 0:
        return "dict"
    return "desconocido"


def test_heuristico_render_dict_detecta_nuevos_pares():
    """Detecta render_*(X) donde X es dict real sin manifiesto.

    Enumera los render_* de src/report/ con firma X_data, X_section,
    X_result, X_synthesis, X_conflict. Clasifica cada X. Falla si un
    dict real (o mixto) sin manifiesto ni whitelist aparece.
    """
    report_dir = REPO_ROOT / "src" / "report"
    pares_manifiesto = {(p[2], p[3]) for p in DICT_CONTRACT_PAIRS}
    pares_whitelist = {
        (w[2], w[3]) for w in DICT_CONTRACT_WHITELIST
    }

    sin_contrato = []
    for py in sorted(report_dir.glob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.name.startswith("render_"):
                continue
            for arg in node.args.args:
                aname = arg.arg
                if not any(aname.endswith(s) for s in (
                        "_data", "_section", "_result", "_synthesis",
                        "_conflict", "_conflict_data")):
                    continue
                if aname in ("reference_date", "is_stale",
                             "stale_reason"):
                    continue
                rel = f"src/report/{py.name}"
                kind = _classify_var(rel, node.name, aname)
                if kind not in ("dict", "mixto"):
                    continue  # df puro o desconocido: no es dict
                key = (rel, node.name)
                if key in pares_manifiesto or key in pares_whitelist:
                    continue
                sin_contrato.append((rel, node.name, aname, kind))

    if sin_contrato:
        detalle = "\n".join(
            f"  - {f}::{fn}({v}) [{k}]"
            for f, fn, v, k in sin_contrato
        )
        raise AssertionError(
            f"Aparecieron {len(sin_contrato)} render_* con dict real "
            f"sin contrato ni whitelist:\n{detalle}\n"
            f"Anadir al manifiesto (con test) o a la whitelist "
            f"(con razon). No ignorar: esta heuristica detecta la "
            f"clase 1fb791ab (consumer lee X, producer no lo emite)."
        )
