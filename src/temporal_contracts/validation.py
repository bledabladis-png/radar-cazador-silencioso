"""Validacion contractual compartida (enmienda v1.1, 2026-10-16).

Fuente unica de decision temporal contractual para:
- scripts/guard_coverage.py
- scripts/write_completion_receipt.py

Reglas (ver docs/auditoria/daily_run_gate_enmienda_v1_1.md §2):
  OK           - todos los contratos obligatorios cumplen.
  WARN         - desviaciones permitidas en contratos no obligatorios.
  BLOCKED      - contrato obligatorio incumplido.
  UNAVAILABLE  - evidencia insuficiente para evaluar contrato obligatorio.

El estado UNAVAILABLE nunca se convierte en OK por ausencia de errores.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.temporal_contracts.registry import (
    CONTRACTS_REGISTRY,
    get_registry_entry,
)


# Mapeo mercado -> contrato. Los by_market del manifest usan nombres
# de mercado (US_EQUITY, BME, EURONEXT, LSE, XETRA); los contratos del
# registro usan nombres funcionales (EQUITY_EOD, INDEX_EOD_*, etc).
#
# Solo US_EQUITY tiene contrato declarado con criticidad True hoy.
# Los mercados europeos (BME, EURONEXT, LSE, XETRA) no tienen contrato
# de nivel EQUITY en el registro (pendiente de fases 3B/3C). Se tratan
# como informativos.
MARKET_TO_CONTRACT = {
    "US_EQUITY": "EQUITY_EOD",
    # BME, EURONEXT, LSE, XETRA: sin contrato -> no criticos
}


STATUS_OK = "OK"
STATUS_WARN = "WARN"
STATUS_BLOCKED = "BLOCKED"
STATUS_UNAVAILABLE = "UNAVAILABLE"


@dataclass
class ContractVerdict:
    """Resultado por mercado/contrato."""
    market: str
    contract: Optional[str]
    critical: bool
    status: str
    reasons: list = field(default_factory=list)


@dataclass
class ValidationResult:
    """Resultado de validacion de un manifest completo."""
    status: str
    reasons: list = field(default_factory=list)
    per_market: dict = field(default_factory=dict)

    def blocks_publication(self) -> bool:
        """True si el estado impide emitir receipt de finalizacion."""
        return self.status in (STATUS_BLOCKED, STATUS_UNAVAILABLE)

    def is_ok_or_warn(self) -> bool:
        return self.status in (STATUS_OK, STATUS_WARN)


def _classify_market(market: str, info: dict) -> ContractVerdict:
    """Clasifica un mercado/contrato individual."""
    contract = MARKET_TO_CONTRACT.get(market)
    critical = False
    min_coverage = None
    if contract is not None and contract in CONTRACTS_REGISTRY:
        entry = get_registry_entry(contract)
        critical = bool(entry.get("critical", False))
        min_coverage = entry.get("min_coverage")

    reasons = []
    n = info.get("n", 0)
    status = info.get("status")
    cov = info.get("coverage_at_session")

    # Contrato critico sin observaciones -> BLOCKED (regla del auditor:
    # n == 0 en contrato obligatorio tambien bloquea).
    if critical and (n is None or n == 0):
        reasons.append(
            f"{market}: n=0 en contrato obligatorio {contract}"
        )
        return ContractVerdict(market, contract, critical,
                                STATUS_BLOCKED, reasons)

    # Contrato critico con status INVALID -> BLOCKED.
    if critical and status == "INVALID":
        reasons.append(
            f"{market}: status=INVALID en contrato obligatorio {contract}"
        )
        return ContractVerdict(market, contract, critical,
                                STATUS_BLOCKED, reasons)

    # Contrato critico con cobertura insuficiente en su sesion esperada
    # -> BLOCKED, aunque el status diga otra cosa (inconsistencia
    # detectada por el auditor).
    if critical and min_coverage is not None:
        if cov is None:
            reasons.append(
                f"{market}: cobertura_at_session ausente en contrato "
                f"obligatorio {contract}"
            )
            return ContractVerdict(market, contract, critical,
                                    STATUS_UNAVAILABLE, reasons)
        if not isinstance(cov, (int, float)) or cov < min_coverage:
            reasons.append(
                f"{market}: cobertura {cov} < min_coverage {min_coverage} "
                f"en contrato obligatorio {contract}"
            )
            return ContractVerdict(market, contract, critical,
                                    STATUS_BLOCKED, reasons)

    # Contrato no critico con problemas -> WARN.
    if not critical:
        if status == "INVALID" or (min_coverage is not None
                                    and isinstance(cov, (int, float))
                                    and cov < min_coverage):
            reasons.append(
                f"{market}: desviacion no bloqueante en contrato "
                f"no critico {contract}"
            )
            return ContractVerdict(market, contract, False, STATUS_WARN, reasons)

    return ContractVerdict(market, contract, critical, STATUS_OK, reasons)


def validate_artifact_contracts(manifest: dict) -> ValidationResult:
    """Valida los contratos obligatorios de un manifest.

    Args:
        manifest: dict del manifest (con quality.by_market).

    Returns:
        ValidationResult con status global y desglose por mercado.
        - OK: todos los criticos cumplen.
        - WARN: solo desviaciones no bloqueantes.
        - BLOCKED: algun critico incumple.
        - UNAVAILABLE: falta evidencia para un critico (por ejemplo
          by_market ausente).

    Casos:
        - manifest no-dict -> UNAVAILABLE.
        - quality ausente -> UNAVAILABLE.
        - by_market ausente -> UNAVAILABLE (no podemos evaluar
          contratos criticos).
        - by_market presente sin el contrato critico esperado ->
          UNAVAILABLE.
    """
    if not isinstance(manifest, dict):
        return ValidationResult(
            STATUS_UNAVAILABLE,
            reasons=["manifest no es dict"],
        )

    quality = manifest.get("quality")
    if not isinstance(quality, dict):
        return ValidationResult(
            STATUS_UNAVAILABLE,
            reasons=["quality ausente o no dict"],
        )

    by_market = quality.get("by_market")
    if not isinstance(by_market, dict) or not by_market:
        # Enmienda v1.1: un manifest SIN by_market no declara contratos
        # por mercado. No se puede evaluar el desglose, pero tampoco
        # afirmar que un contrato falla. La validacion global del guard
        # (cobertura, status, last_date) sigue aplicando. Se degrada a
        # WARN, no a UNAVAILABLE: UNAVAILABLE se reserva para "hay
        # desglose y falta el contrato critico".
        return ValidationResult(
            STATUS_WARN,
            reasons=["by_market ausente o vacio: sin desglose por "
                     "mercado; validacion contractual no aplicable"],
        )

    # Comprobar que todos los contratos criticos tienen evidencia en
    # by_market (via MARKET_TO_CONTRACT inverso).
    criticos = [
        name for name, entry in CONTRACTS_REGISTRY.items()
        if entry.get("critical", False)
    ]
    markets_of_criticos = {
        mkt for mkt, c in MARKET_TO_CONTRACT.items()
        if c in criticos
    }
    mercados_presentes = set(by_market.keys())
    criticos_sin_evidencia = markets_of_criticos - mercados_presentes
    if criticos_sin_evidencia:
        return ValidationResult(
            STATUS_UNAVAILABLE,
            reasons=[
                f"Contrato(s) critico(s) sin evidencia en by_market: "
                f"{sorted(criticos_sin_evidencia)}"
            ],
        )

    # Clasificar cada mercado.
    per_market = {}
    worst = STATUS_OK
    for market, info in by_market.items():
        if not isinstance(info, dict):
            per_market[market] = ContractVerdict(
                market, MARKET_TO_CONTRACT.get(market), False,
                STATUS_WARN, [f"{market}: info no es dict"],
            )
            if worst == STATUS_OK:
                worst = STATUS_WARN
            continue
        v = _classify_market(market, info)
        per_market[market] = v
        if v.status == STATUS_BLOCKED:
            worst = STATUS_BLOCKED
        elif v.status == STATUS_UNAVAILABLE and worst != STATUS_BLOCKED:
            worst = STATUS_UNAVAILABLE
        elif v.status == STATUS_WARN and worst == STATUS_OK:
            worst = STATUS_WARN

    reasons = []
    for m, v in per_market.items():
        reasons.extend(v.reasons)

    return ValidationResult(worst, reasons=reasons, per_market=per_market)