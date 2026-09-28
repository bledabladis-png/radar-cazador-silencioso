"""Caracterizacion de indicators/mte/scoring.py (F2.4-01, F2.4-02, F2.4-03).

F2.4-01: eliminacion del bonus cls > 0.7 duplicado en CRISIS.
F2.4-02: except acotado en sector_rotation_score (SRS defensive_beating_spy).
F2.4-03: SCENARIO_WEIGHTS["IPS"]["weight"] usado (antes hardcode +3).
"""
import ast
from pathlib import Path

import pytest

from indicators.mte.scoring import SCENARIO_WEIGHTS, score_scenarios


SCORING_PATH = Path("indicators/mte/scoring.py")


class TestScoreScenariosContrato:
    """Contrato observable: dict con 6 claves siempre."""

    @pytest.mark.parametrize("srs,shs,cls,ips,esperado", [
        # Todo cero. STAGFLATION=3 por cls < 0.3 (CLS weight=3). MIXED=2 por defecto.
        (0.0, 0.0, 0.0, 0.0,
         {"CRISIS": 0, "RECESSION": 0, "STAGFLATION": 3,
          "SOFT LANDING": 0, "EXPANSION": 0, "MIXED": 2}),
        # cls=0.7 exacto: no dispara el bonus severo (>)
        (0.0, 0.0, 0.7, 0.0,
         {"CRISIS": 6, "RECESSION": 3, "STAGFLATION": 0,
          "SOFT LANDING": 0, "EXPANSION": 0, "MIXED": 2}),
        # cls=0.71: F2.4-01 -> CRISIS=9 (antes +2 duplicado daba 11)
        (0.0, 0.0, 0.71, 0.0,
         {"CRISIS": 9, "RECESSION": 3, "STAGFLATION": 0,
          "SOFT LANDING": 0, "EXPANSION": 0, "MIXED": 2}),
        # cls=0.86: 6+3+3 = 12
        (0.0, 0.0, 0.86, 0.0,
         {"CRISIS": 12, "RECESSION": 3, "STAGFLATION": 0,
          "SOFT LANDING": 0, "EXPANSION": 0, "MIXED": 2}),
        # ips=0.2: STAGFLATION = IPS(3) + CLS(3) = 6
        (0.0, 0.0, 0.0, 0.2,
         {"CRISIS": 0, "RECESSION": 0, "STAGFLATION": 6,
          "SOFT LANDING": 0, "EXPANSION": 0, "MIXED": 2}),
        # shs=0.4
        (0.0, 0.4, 0.0, 0.0,
         {"CRISIS": 2, "RECESSION": 2, "STAGFLATION": 3,
          "SOFT LANDING": 2, "EXPANSION": 0, "MIXED": 2}),
        # srs=0.4
        (0.4, 0.0, 0.0, 0.0,
         {"CRISIS": 2, "RECESSION": 2, "STAGFLATION": 5,
          "SOFT LANDING": 2, "EXPANSION": 0, "MIXED": 2}),
    ])
    def test_score_scenarios_valores(self, srs, shs, cls, ips, esperado):
        assert score_scenarios(srs, shs, cls, ips) == esperado

    def test_cls_nan_bloquea_crisis(self):
        r = score_scenarios(0.0, 0.0, float("nan"), 0.0)
        assert r["CRISIS"] == -999

class TestF24_01_NoDuplicado:
    """F2.4-01: el bonus cls > 0.7 aparece UNA sola vez en CRISIS."""

    def test_una_sola_condicion_cls_07(self):
        tree = ast.parse(SCORING_PATH.read_text(encoding="utf-8-sig"))
        for fn in tree.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == "score_scenarios":
                hits = []
                for node in ast.walk(fn):
                    if isinstance(node, ast.Compare):
                        if (isinstance(node.left, ast.Name)
                                and node.left.id == "cls"
                                and len(node.comparators) == 1
                                and isinstance(node.comparators[0], ast.Constant)
                                and node.comparators[0].value == 0.7):
                            hits.append(node.lineno)
                assert len(hits) == 1, (
                    f"cls > 0.7 aparece {len(hits)} veces en score_scenarios "
                    f"(lineas: {hits}); debe ser 1"
                )
                return
        pytest.fail("score_scenarios no encontrado")


class TestF24_02_SinBareExcept:
    """F2.4-02: no debe haber ningun except desnudo en scoring.py."""

    def test_sin_bare_except(self):
        tree = ast.parse(SCORING_PATH.read_text(encoding="utf-8-sig"))
        bare = [n.lineno for n in ast.walk(tree)
                if isinstance(n, ast.ExceptHandler) and n.type is None]
        assert bare == [], f"except desnudo en lineas: {bare}"

    def test_srs_except_tupla_especifica(self):
        """El except de sector_rotation_score captura los 4 tipos reales."""
        tree = ast.parse(SCORING_PATH.read_text(encoding="utf-8-sig"))
        for fn in tree.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == "sector_rotation_score":
                for node in ast.walk(fn):
                    if isinstance(node, ast.ExceptHandler):
                        assert isinstance(node.type, ast.Tuple), (
                            "sector_rotation_score except debe ser una tupla"
                        )
                        names = sorted(
                            elt.id for elt in node.type.elts
                            if isinstance(elt, ast.Name)
                        )
                        assert names == ["AttributeError", "KeyError",
                                         "TypeError", "ValueError"], (
                            f"tupla inesperada: {names}"
                        )
                        return
        pytest.fail("sector_rotation_score sin ExceptHandler")

class TestF24_03_IPSUsesConstante:
    """F2.4-03: SCENARIO_WEIGHTS["IPS"]["weight"] = 3 y se usa en STAGFLATION."""

    def test_ips_weight_valor(self):
        assert SCENARIO_WEIGHTS["IPS"]["weight"] == 3

    def test_ips_contribuye_a_stagflation(self):
        # ips=0.2 sin srs, sin cls alto -> STAGFLATION incluye IPS weight
        r = score_scenarios(0.0, 0.0, 0.0, 0.2)
        # 3 (IPS) + 3 (CLS < 0.3) = 6
        assert r["STAGFLATION"] == 6

    def test_ips_bajo_umbral_no_contribuye(self):
        # ips=0.1 no dispara (> 0.15)
        r = score_scenarios(0.0, 0.0, 0.0, 0.1)
        # solo cls < 0.3: +3
        assert r["STAGFLATION"] == 3

class TestF24_11_SinBareExceptEngine:
    """F2.4-11: no debe haber except desnudo en engine.py."""

    def test_sin_bare_except_engine(self):
        tree = ast.parse(Path("indicators/mte/engine.py").read_text(encoding="utf-8-sig"))
        bare = [n.lineno for n in ast.walk(tree)
                if isinstance(n, ast.ExceptHandler) and n.type is None]
        assert bare == [], f"except desnudo en engine.py lineas: {bare}"

    def test_engine_vix_term_except_tupla(self):
        """El except del bloque vix_term captura los 4 tipos reales.

        A3.3-13 (2026-09-28): antes el test anclaba node.lineno == 89.
        Cualquier refactor que desplace esa linea (whitespace collapse,
        nuevo import arriba, comentario) rompia el test SIN que el
        contrato cambiara. Reescrito para buscar por estructura: la
        tupla exacta (AttributeError, KeyError, TypeError, ValueError)
        debe aparecer exactamente una vez en engine.py.
        """
        tree = ast.parse(Path("indicators/mte/engine.py").read_text(encoding="utf-8-sig"))
        expected = ["AttributeError", "KeyError", "TypeError", "ValueError"]
        candidates = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Tuple):
                names = sorted(
                    elt.id for elt in node.type.elts
                    if isinstance(elt, ast.Name)
                )
                candidates.append((node.lineno, names))
        matches = [ln for ln, names in candidates if names == expected]
        assert len(matches) == 1, (
            f"esperaba exactamente 1 except con tupla {expected}; "
            f"candidatos: {candidates}"
        )
