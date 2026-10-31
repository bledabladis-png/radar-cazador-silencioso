from src.utils import resolve_observation_date
# -*- coding: utf-8 -*-
"""Fase 11 del pipeline: Validation Gate (10 comprobaciones) + audit
double-counting.

Extraido de run.py (refactor C2, fase C2-11).

IMPORTANTE: el original hacia sys.exit(1) si habia errores. Aqui se
devuelve un dict con 'passed'. run.py decide si abortar.
"""


import numpy as np
import pandas as pd

from src.dependency_tracker import audit_double_counting


def _add_check(checks, errors, nombre, ok=True, detalle=""):
    if ok:
        checks.append(f"{nombre}: OK {detalle}".strip())
    else:
        errors.append(f"{nombre}: {detalle}".strip())


def run_validation_gate(wyckoff_holdings_data, pcr_data, darkpool_data, mte_result,
                        tactical_scores, structural_scores,
                        reference_date=None):
    """Ejecuta las 10 comprobaciones del Validation Gate.

    Returns:
        dict con keys:
            passed (bool): True si no hay errores
            checks (list[str]): comprobaciones OK
            errors (list[str]): errores detectados
            dc_summary (str): resumen double-counting para el reporte
    """
    # D23 (2026-09-30): _ref para los calculos de age.
    # D-05 (2026-10-03): normalizar tzinfo. run.py:43 construye
    # reference_date tz-aware (ZoneInfo Europe/Madrid). pd.Timestamp(week)
    # y pd.Timestamp(last_date) son naive. Sin replace(tzinfo=None),
    # _ref - d lanza TypeError, capturado por los except de L129 y L147.
    _ref = resolve_observation_date(reference_date)
    print("Ejecutando Validation Gate...")
    validation_errors = []
    validation_checks = []

    # 1. Wyckoff holdings CSVs (C3.2: sustituye check SLPM v1.2)
    if wyckoff_holdings_data:
        n_tables = sum(1 for df in wyckoff_holdings_data.values() if df is not None and not df.empty)
        n_rows = sum(len(df) for df in wyckoff_holdings_data.values() if df is not None and not df.empty)
        if n_tables > 0 and n_rows > 0:
            _add_check(validation_checks, validation_errors, "Wyckoff holdings CSVs", True, f"{n_tables} tablas con datos, {n_rows} filas")
        else:
            _add_check(validation_checks, validation_errors, "Wyckoff holdings CSVs", True, "0 tablas con datos")
    else:
        _add_check(validation_checks, validation_errors, "Wyckoff holdings CSVs", True, "no disponible")

    # 2. PCR Total
    if pcr_data:
        val = pcr_data.get('total_pcr', np.nan)
        if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
            _add_check(validation_checks, validation_errors, "PCR Total", False, "NaN")
        else:
            _add_check(validation_checks, validation_errors, "PCR Total", True, f"{val:.2f}")
    else:
        _add_check(validation_checks, validation_errors, "PCR Total", True, "sin datos")

    # 3. Dark Pool medio
    if darkpool_data:
        val = darkpool_data.get('media_dark_pool', np.nan)
        if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
            _add_check(validation_checks, validation_errors, "Dark Pool medio", False, "NaN")
        else:
            _add_check(validation_checks, validation_errors, "Dark Pool medio", True, f"{val:.2f}")
    else:
        _add_check(validation_checks, validation_errors, "Dark Pool medio", True, "sin datos")

    # 4. MTE (MSI/IPI)
    if mte_result:
        msi = mte_result.get('msi', np.nan)
        ipi = mte_result.get('ipi', np.nan)
        if (msi is None or (isinstance(msi, float) and np.isnan(msi)) or pd.isna(msi) or
            ipi is None or (isinstance(ipi, float) and np.isnan(ipi)) or pd.isna(ipi)):
            _add_check(validation_checks, validation_errors, "MTE", False, "NaN en MSI/IPI")
        else:
            _add_check(validation_checks, validation_errors, "MTE", True, f"MSI={msi:.2f}, IPI={ipi:.2f}")
    else:
        _add_check(validation_checks, validation_errors, "MTE", True, "sin datos")

    # 5. Rangos tacticos/estructurales
    if tactical_scores and structural_scores:
        sectors_checked = 0
        for ticker in tactical_scores:
            if ticker in structural_scores:
                t = tactical_scores[ticker]
                s = structural_scores[ticker]
                if not isinstance(t, (int, float)) or not isinstance(s, (int, float)) or pd.isna(t) or pd.isna(s):
                    validation_errors.append(f"{ticker}: Tactical/Structural no numerico.")
                    continue
                if abs(t) > 1.0:
                    validation_errors.append(f"{ticker}: Tactical Score fuera de rango ({t:+.2f}).")
                if abs(s) > 1.0:
                    validation_errors.append(f"{ticker}: Structural Score fuera de rango ({s:+.2f}).")
                sectors_checked += 1
        if sectors_checked == 0:
            _add_check(validation_checks, validation_errors, "Rangos tacticos/estructurales", False, "sin sectores coincidentes")
        else:
            _add_check(validation_checks, validation_errors, "Rangos tacticos/estructurales", True, f"{sectors_checked} sectores")
    else:
        _add_check(validation_checks, validation_errors, "Rangos tacticos/estructurales", True, "sin datos")

    # 6. Cobertura holdings (C3.2: sustituye check Opportunity Map)
    if wyckoff_holdings_data:
        groups_counts = {}
        for _key, _df in wyckoff_holdings_data.items():
            if _df is None or _df.empty:
                continue
            if 'group' in _df.columns:
                for _g, _cnt in _df.groupby('group').size().items():
                    groups_counts[_g] = groups_counts.get(_g, 0) + int(_cnt)
        good_groups = [g for g, c in groups_counts.items() if c >= 5]
        if good_groups:
            _add_check(validation_checks, validation_errors, "Cobertura holdings", True, f"{len(good_groups)} grupos con >=5 tickers")
        elif groups_counts:
            _add_check(validation_checks, validation_errors, "Cobertura holdings", True, f"0 grupos con >=5 tickers ({len(groups_counts)} con datos)")
        else:
            _add_check(validation_checks, validation_errors, "Cobertura holdings", True, "sin grupos con datos")
    else:
        _add_check(validation_checks, validation_errors, "Cobertura holdings", True, "no disponible")

    # 7. Data Freshness Dark Pool
    if darkpool_data:
        week = darkpool_data.get('week', '')
        if week:
            try:
                d = pd.Timestamp(week)
                age = (_ref - d).days
                if age > 14:
                    _add_check(validation_checks, validation_errors, "Freshness Dark Pool", True, f"obsoleto {age} dias (advertencia)")
                else:
                    _add_check(validation_checks, validation_errors, "Freshness Dark Pool", True, f"{age} dias")
            except (ValueError, TypeError, OSError):
                _add_check(validation_checks, validation_errors, "Freshness Dark Pool", True, "sin fecha")
        else:
            _add_check(validation_checks, validation_errors, "Freshness Dark Pool", True, "sin fecha")
    else:
        _add_check(validation_checks, validation_errors, "Freshness Dark Pool", True, "sin datos")

    # 8. Data Freshness PCR
    if pcr_data:
        last_date = pcr_data.get('last_date', '')
        if last_date and last_date != 'N/A':
            try:
                d = pd.Timestamp(last_date)
                age = (_ref - d).days
                if age > 5:
                    _add_check(validation_checks, validation_errors, "Freshness PCR", True, f"desactualizado {age} dias (advertencia)")
                else:
                    _add_check(validation_checks, validation_errors, "Freshness PCR", True, f"{age} dias")
            except (ValueError, TypeError, OSError):
                _add_check(validation_checks, validation_errors, "Freshness PCR", True, "sin fecha")
        else:
            _add_check(validation_checks, validation_errors, "Freshness PCR", True, "sin fecha")
    else:
        _add_check(validation_checks, validation_errors, "Freshness PCR", True, "sin datos")

    # 9. Configuracion de pesos
    try:
        from config.weights import validate_weights
        validate_weights()
        _add_check(validation_checks, validation_errors, "Config pesos", True, "validados")
    except (ImportError, KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
        _add_check(validation_checks, validation_errors, "Config pesos", False, str(e))

    # 10. Sanity holdings (C3.2: sustituye check Anti-Double-Counting)
    if wyckoff_holdings_data:
        issues = []
        for _key, _df in wyckoff_holdings_data.items():
            if _df is None or _df.empty:
                continue
            if 'close' in _df.columns:
                _bad_close = pd.to_numeric(_df['close'], errors='coerce')
                _n_bad = int((_bad_close <= 0).sum())
                if _n_bad > 0:
                    issues.append(f"{_key}.close: {_n_bad} valores <= 0")
            if 'weight' in _df.columns:
                _bad_w = pd.to_numeric(_df['weight'], errors='coerce')
                _n_bad = int((_bad_w < 0).sum())
                if _n_bad > 0:
                    issues.append(f"{_key}.weight: {_n_bad} valores negativos")
        if issues:
            for _iss in issues:
                validation_errors.append(f"Sanity holdings: {_iss}")
            _add_check(validation_checks, validation_errors, "Sanity holdings", False, f"{len(issues)} problemas")
        else:
            _add_check(validation_checks, validation_errors, "Sanity holdings", True, "valores coherentes")
    else:
        _add_check(validation_checks, validation_errors, "Sanity holdings", True, "no disponible")

    # 11. Schema versioning (Fase 1): todos los outputs temporales con
    #     .schema deben coincidir con spec.schema_version. Si hay .v*.csv
    #     al lado, se reporta MIGRATED (informativo, no falla).
    try:
        from src.pipeline.output_registry import REGISTRY
        from pathlib import Path as _P
        _schema_issues = []
        _migrated_count = 0
        _checked = 0
        for _spec in REGISTRY:
            if getattr(_spec, "kind", None) != "temporal":
                continue
            _csv = _P(_spec.path)
            _schema_f = _csv.with_suffix(_csv.suffix + ".schema")
            _expected = getattr(_spec, "schema_version", 1)
            _checked += 1
            if _schema_f.exists():
                try:
                    _actual = int(_schema_f.read_text(encoding="utf-8").strip())
                except (OSError, ValueError):
                    _actual = None
                if _actual is not None and _actual != _expected:
                    _schema_issues.append((_spec.name, _actual, _expected))
            # Buscar .v*.csv al lado
            for _sib in _csv.parent.glob(_csv.stem + ".v*" + _csv.suffix):
                _migrated_count += 1
                break
        # Bloque 2 auditoria externa (2026-10-25): distinguir migracion
        # pendiente (spec > actual: el CSV aun no se ha reescrito con el
        # nuevo schema, se hara en el siguiente persist() de ese output)
        # de regresion (spec < actual: el codigo espera un schema mas
        # antiguo que el CSV). Migracion = WARN, regresion = BLOCKED.
        _pending = []
        _regressions = []
        for _iss_tuple in _schema_issues:
            _name, _actual, _expected = _iss_tuple
            if _expected > _actual:
                _pending.append((_name, _actual, _expected))
            else:
                _regressions.append((_name, _actual, _expected))
        if _regressions:
            for _name, _actual, _expected in _regressions:
                validation_errors.append(
                    f"Schema regression: {_name} espera v{_expected} pero CSV es v{_actual}"
                )
            _add_check(
                validation_checks, validation_errors, "Schema versioning",
                False, f"{len(_regressions)} regresiones",
            )
        elif _pending:
            # Migracion pendiente: WARN, no bloquea. Se completa en el
            # siguiente persist() del output.
            _detail = f"{_checked} outputs v, {len(_pending)} migracion pendiente"
            if _migrated_count:
                _detail += f", {_migrated_count} MIGRATED"
            print(f"    [WARN] Schema versioning: {len(_pending)} migraciones pendientes (no bloquea)")
            _add_check(
                validation_checks, validation_errors, "Schema versioning",
                True, _detail,
            )
        else:
            _detail = f"{_checked} outputs v"
            if _migrated_count:
                _detail += f", {_migrated_count} MIGRATED"
            _add_check(
                validation_checks, validation_errors, "Schema versioning",
                True, _detail,
            )
    except (ImportError, OSError, AttributeError, KeyError, TypeError) as _e:
        # FAIL-CLOSED (2026-10-31, Fase 2 B5.3): antes except Exception
        # reportaba True "no evaluable". Un check que existe pero no
        # verifica es peor que no tenerlo: un import roto, un REGISTRY
        # vacio por bug o un OSError en el glob dejaba pasar el gate sin
        # auditar los schemas. Misma clase que GUARD-COVERAGE-LAST-DATE-
        # PARCIAL y que la clase 1fb791ab.
        _add_check(
            validation_checks, validation_errors, "Schema versioning",
            False, f"no evaluable: {_e}",
        )

    # Resultado
    passed = not validation_errors
    if passed:
        print(f"    VALIDATION GATE: Sin errores ({len(validation_checks)} comprobaciones OK)")
    else:
        print(f"    VALIDATION GATE: {len(validation_errors)} errores, {len(validation_checks)} comprobaciones")
        for err in validation_errors:
            print(f"      {err}")

    # Generar resumen de double-counting para el reporte
    dc_summary = ""
    try:
        dc_audit = audit_double_counting()
        dc_summary = dc_audit.get('summary', '')
    except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
        print(f"  [WARN] audit_double_counting: {e}")

    return {
        'passed': passed,
        'checks': validation_checks,
        'errors': validation_errors,
        'dc_summary': dc_summary,
    }
