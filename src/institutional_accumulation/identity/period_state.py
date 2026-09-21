"""B1 - Estados por periodo (A62BIS_B1_SUBFASE.md v4 §4-§6).

Para cada K in TARGET_Q4 UNION TARGET_Q1, define:
  identity_status   (RESOLVED | UNRESOLVED | CONFLICT | AMBIGUOUS | NOT_PRESENT)
  weight_status     (RESOLVED_OBSERVED | ZERO_REPORTED | NOT_PRESENT)
  figi              unico FIGI o None
  ticker            ticker o None

Regla B4 (feasible_state): identity_status == RESOLVED
  AND weight_status in {RESOLVED_OBSERVED, ZERO_REPORTED}.

Contrato de pureza: no lee/escribe ficheros, no datetime.now(), determinista.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# --- identity_status ---
IDENTITY_RESOLVED = "RESOLVED"
IDENTITY_UNRESOLVED = "UNRESOLVED"
IDENTITY_CONFLICT = "CONFLICT"
IDENTITY_AMBIGUOUS = "AMBIGUOUS"
IDENTITY_NOT_PRESENT = "NOT_PRESENT"

ALL_IDENTITY_STATUSES = (
    IDENTITY_RESOLVED,
    IDENTITY_UNRESOLVED,
    IDENTITY_CONFLICT,
    IDENTITY_AMBIGUOUS,
    IDENTITY_NOT_PRESENT,
)


# --- weight_status ---
WEIGHT_RESOLVED_OBSERVED = "RESOLVED_OBSERVED"
WEIGHT_ZERO_REPORTED = "ZERO_REPORTED"
WEIGHT_NOT_PRESENT = "NOT_PRESENT"

ALL_WEIGHT_STATUSES = (
    WEIGHT_RESOLVED_OBSERVED,
    WEIGHT_ZERO_REPORTED,
    WEIGHT_NOT_PRESENT,
)


FEASIBLE_WEIGHT_STATUSES = frozenset({
    WEIGHT_RESOLVED_OBSERVED,
    WEIGHT_ZERO_REPORTED,
})


# --- operational_mapping_status (P61 seccion 2.3) ---
OPERATIONAL_VERIFIED = "VERIFIED"
OPERATIONAL_TEMPORAL_UNVERIFIED = "TEMPORAL_UNVERIFIED"
OPERATIONAL_UNRESOLVED = "UNRESOLVED"
OPERATIONAL_CONFLICT = "CONFLICT"

ALL_OPERATIONAL_MAPPING_STATUSES = (
    OPERATIONAL_VERIFIED,
    OPERATIONAL_TEMPORAL_UNVERIFIED,
    OPERATIONAL_UNRESOLVED,
    OPERATIONAL_CONFLICT,
)


@dataclass(frozen=True)
class PeriodState:
    """Estado de una catalog_key en un periodo.

    Campos B1 (A62BIS_B1_SUBFASE.md v4 secciones 4-6):
      identity_status            ver ALL_IDENTITY_STATUSES.
      weight_status              ver ALL_WEIGHT_STATUSES.
      figi                       shareClassFIGI unico o None.
      ticker                     ticker o None.

    Campos A2 (fix H-10.1, 2026-09-21):
      sshprnamt                  SSHPRNAMT efectivo agregado por
                                 shareClassFIGI desde el
                                 canonical_snapshot post-amendments.
                                 None si no hay evidencia observada.
                                 Default None (backward compat).
      operational_mapping_status estado P61 seccion 2.3:
                                 VERIFIED | TEMPORAL_UNVERIFIED |
                                 UNRESOLVED | CONFLICT.
                                 Default UNRESOLVED (fail-closed).
                                 NO se infiere VERIFIED.
    """
    identity_status: str
    weight_status: str
    figi: Optional[str]
    ticker: Optional[str]
    sshprnamt: Optional[float] = None
    operational_mapping_status: str = OPERATIONAL_UNRESOLVED

    def __post_init__(self):
        if self.identity_status not in ALL_IDENTITY_STATUSES:
            raise ValueError(
                "identity_status invalido: " + repr(self.identity_status)
            )
        if self.weight_status not in ALL_WEIGHT_STATUSES:
            raise ValueError(
                "weight_status invalido: " + repr(self.weight_status)
            )
        if self.operational_mapping_status not in ALL_OPERATIONAL_MAPPING_STATUSES:
            raise ValueError(
                "operational_mapping_status invalido: "
                + repr(self.operational_mapping_status)
            )
        if self.sshprnamt is not None:
            if not isinstance(self.sshprnamt, (int, float)):
                raise ValueError(
                    "sshprnamt debe ser numerico o None: "
                    + repr(self.sshprnamt)
                )
            if self.sshprnamt < 0:
                raise ValueError(
                    "sshprnamt debe ser >= 0: " + repr(self.sshprnamt)
                )


class StateBuildError(ValueError):
    """Error en la construccion de estados."""


def build_period_state(universe, *, figi_evidence=None, weight_evidence=None,
                       operational_evidence=None, sshprnamt_evidence=None):
    """Construye state[K] para K in universe.declared_keys.

    Parametros:
      universe              TargetUniverse.
      figi_evidence         dict {catalog_key: {"status": RESOLVED|...,
                            "figi": str|None}}.
                            Por defecto: cada key con figi no vacio ->
                            RESOLVED.
      weight_evidence       dict {catalog_key: weight_status}.
                            Por defecto: RESOLVED_OBSERVED.
      operational_evidence  dict {catalog_key: operational_mapping_status}.
                            Por defecto: OPERATIONAL_UNRESOLVED.
                            NO se infiere VERIFIED.
      sshprnamt_evidence    dict {catalog_key: float|None}.
                            SSHPRNAMT efectivo del canonical_snapshot
                            post-amendments, agregado por shareClassFIGI.
                            Por defecto: None.

    Devuelve dict {catalog_key: PeriodState}.
    """
    out = {}
    for k in sorted(universe.declared_keys):
        if figi_evidence is not None and k in figi_evidence:
            ev = figi_evidence[k]
            identity_status = ev.get("status", IDENTITY_UNRESOLVED)
            figi = ev.get("figi")
        else:
            f = universe.figi_by_key.get(k)
            if f:
                identity_status = IDENTITY_RESOLVED
                figi = f
            else:
                identity_status = IDENTITY_UNRESOLVED
                figi = None

        if weight_evidence is not None and k in weight_evidence:
            weight_status = weight_evidence[k]
        else:
            weight_status = WEIGHT_RESOLVED_OBSERVED

        if (operational_evidence is not None
                and k in operational_evidence):
            operational_mapping_status = operational_evidence[k]
        else:
            operational_mapping_status = OPERATIONAL_UNRESOLVED

        if sshprnamt_evidence is not None and k in sshprnamt_evidence:
            sshprnamt = sshprnamt_evidence[k]
        else:
            sshprnamt = None

        ticker = universe.ticker_by_key.get(k)
        out[k] = PeriodState(
            identity_status=identity_status,
            weight_status=weight_status,
            figi=figi,
            ticker=ticker,
            sshprnamt=sshprnamt,
            operational_mapping_status=operational_mapping_status,
        )
    return out


def feasible_state(state):
    """B4: identity_status == RESOLVED AND weight_status in
    {RESOLVED_OBSERVED, ZERO_REPORTED}."""
    if state is None:
        return False
    return (
        state.identity_status == IDENTITY_RESOLVED
        and state.weight_status in FEASIBLE_WEIGHT_STATUSES
    )


def unique_figi(state):
    """Devuelve el FIGI unico si identity_status==RESOLVED y figi no vacio.
    Si no, devuelve None."""
    if state is None:
        return None
    if state.identity_status != IDENTITY_RESOLVED:
        return None
    if not state.figi:
        return None
    return state.figi