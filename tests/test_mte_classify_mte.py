"""Tests de classify_mte (F2.4-10 + F2.4-12 ii)."""
from indicators.mte.decision import classify_mte


def _patch_prev(monkeypatch, prev, pending, reset=False):
    monkeypatch.setattr(
        "indicators.mte.decision.load_previous_scenario",
        lambda: (prev, pending, reset),
    )


def _patch_score(monkeypatch, scores_dict):
    monkeypatch.setattr(
        "indicators.mte.decision.score_scenarios",
        lambda *a, **kw: scores_dict,
    )


def _patch_validate(monkeypatch, allowed=True):
    monkeypatch.setattr(
        "indicators.mte.decision.validate_transition",
        lambda *a, **kw: allowed,
    )


def _patch_confidence(monkeypatch, value=0.7):
    monkeypatch.setattr(
        "indicators.mte.decision.compute_confidence",
        lambda *a, **kw: value,
    )


class TestClassifyMteBranches:

    def test_branch1_cls_alto(self, monkeypatch):
        _patch_prev(monkeypatch, prev="MIXED", pending=None)
        _patch_score(monkeypatch, {"MIXED": 0, "RECESSION": 5})
        _patch_validate(monkeypatch, True)
        _patch_confidence(monkeypatch, 0.7)

        scenario, _, pending_out, reset = classify_mte(
            srs=0, shs=0, cls=0.9, ips=0)

        assert scenario == "RECESSION"
        assert pending_out is None
        assert reset is False

    def test_branch2a_pending_none(self, monkeypatch):
        _patch_prev(monkeypatch, prev="MIXED", pending=None)
        _patch_score(monkeypatch, {"MIXED": 0, "RECESSION": 5})
        _patch_validate(monkeypatch, True)
        _patch_confidence(monkeypatch, 0.7)

        scenario, _, pending_out, reset = classify_mte(
            srs=0, shs=0, cls=0.5, ips=0)

        assert scenario == "MIXED"
        assert pending_out == "RECESSION"
        assert reset is False

    def test_branch2b_pending_confirma(self, monkeypatch):
        _patch_prev(monkeypatch, prev="MIXED", pending="RECESSION")
        _patch_score(monkeypatch, {"MIXED": 0, "RECESSION": 5})
        _patch_validate(monkeypatch, True)
        _patch_confidence(monkeypatch, 0.7)

        scenario, _, pending_out, reset = classify_mte(
            srs=0, shs=0, cls=0.5, ips=0)

        assert scenario == "RECESSION"
        assert pending_out is None
        assert reset is False

    def test_branch2c_pending_descarta(self, monkeypatch):
        _patch_prev(monkeypatch, prev="MIXED", pending="STAGFLATION")
        _patch_score(monkeypatch, {"MIXED": 0, "RECESSION": 5})
        _patch_validate(monkeypatch, True)
        _patch_confidence(monkeypatch, 0.7)

        scenario, _, pending_out, reset = classify_mte(
            srs=0, shs=0, cls=0.5, ips=0)

        assert scenario == "MIXED"
        assert pending_out is None
        assert reset is False

    def test_branch3_new_igual_prev(self, monkeypatch):
        _patch_prev(monkeypatch, prev="MIXED", pending="RECESSION")
        _patch_score(monkeypatch, {"MIXED": 5, "RECESSION": 0})
        _patch_validate(monkeypatch, True)
        _patch_confidence(monkeypatch, 0.7)

        scenario, _, pending_out, reset = classify_mte(
            srs=0, shs=0, cls=0.3, ips=0)

        assert scenario == "MIXED"
        assert pending_out is None
        assert reset is False


class TestHysteresisAcrossRuns:

    def test_two_runs_confirm_candidate(self, monkeypatch):
        _patch_score(monkeypatch, {"MIXED": 0, "RECESSION": 5})
        _patch_validate(monkeypatch, True)
        _patch_confidence(monkeypatch, 0.7)

        run1_prev = ("MIXED", None, False)
        monkeypatch.setattr(
            "indicators.mte.decision.load_previous_scenario",
            lambda: run1_prev,
        )
        s1, _, p1, r1 = classify_mte(srs=0, shs=0, cls=0.5, ips=0)
        assert s1 == "MIXED"
        assert p1 == "RECESSION"
        assert r1 is False

        run2_prev = (s1, p1, False)
        monkeypatch.setattr(
            "indicators.mte.decision.load_previous_scenario",
            lambda: run2_prev,
        )
        s2, _, p2, r2 = classify_mte(srs=0, shs=0, cls=0.5, ips=0)
        assert s2 == "RECESSION"
        assert p2 is None
        assert r2 is False


class TestTemporalReset:

    def test_reset_preserves_current_result(self, monkeypatch):
        _patch_score(monkeypatch, {"MIXED": 0, "RECESSION": 5})
        _patch_validate(monkeypatch, True)
        _patch_confidence(monkeypatch, 0.7)

        monkeypatch.setattr(
            "indicators.mte.decision.load_previous_scenario",
            lambda: ("MIXED", None, True),
        )

        scenario, _, pending_out, reset = classify_mte(
            srs=0, shs=0, cls=0.9, ips=0)

        assert reset is True
        assert scenario == "RECESSION"
        assert pending_out is None

    def test_reset_clears_pending_in_memory(self, monkeypatch):
        _patch_score(monkeypatch, {"MIXED": 0, "STAGFLATION": 5})
        _patch_validate(monkeypatch, True)
        _patch_confidence(monkeypatch, 0.7)

        monkeypatch.setattr(
            "indicators.mte.decision.load_previous_scenario",
            lambda: ("MIXED", None, True),
        )

        scenario, _, pending_out, reset = classify_mte(
            srs=0, shs=0, cls=0.3, ips=0.3)

        assert reset is True
        assert scenario == "MIXED"
        assert pending_out == "STAGFLATION"
