# -*- coding: utf-8 -*-
"""Fase 11 del pipeline: Validation Gate (10 comprobaciones) + audit
double-counting.

Extraido de run.py (refactor C2, fase C2-11).

IMPORTANTE: el original hacia sys.exit(1) si habia errores. Aqui se
devuelve un dict con 'passed'. run.py decide si abortar.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from src.dependency_tracker import audit_double_counting


def _add_check(checks, errors, nombre, ok=True, detalle=""):
    if ok:
        checks.append(f"{nombre}: OK {detalle}".strip())
    else:
        errors.append(f"{nombre}: {detalle}".strip())


def run_validation_gate(slpm_v12_data, pcr_data, darkpool_data, mte_result,
                        tactical_scores, structural_scores):
    """Ejecuta las 10 comprobaciones del Validation Gate.

    Returns:
        dict con keys:
            passed (bool): True si no hay errores
            checks (list[str]): comprobaciones OK
            errors (list[str]): errores detectados
            dc_summary (str): resumen double-counting para el reporte
    """
    print("Ejecutando Validation Gate...")
    validation_errors = []
    validation_checks = []

    # 1. SLPM v1.2
    if slpm_v12_data:
        slpm_errors = slpm_v12_data.get('validation_errors', [])
        if slpm_errors:
            validation_errors.extend(slpm_errors)
            _add_check(validation_checks, validation_errors, "SLPM v1.2", False, f"{len(slpm_errors)} errores SLPM")
        else:
            _add_check(validation_checks, validation_errors, "SLPM v1.2", True, "estado validado")
    else:
        _add_check(validation_checks, validation_errors, "SLPM v1.2", True, "no disponible")

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

    # 6. Opportunity Map
    if slpm_v12_data and tactical_scores and structural_scores:
        leader_etf = slpm_v12_data.get('sector_etf', '')
        if leader_etf and leader_etf in tactical_scores:
            slpm_quadrant = slpm_v12_data.get('opportunity_quadrant', '')
            if slpm_v12_data.get('state') == 'UNRESOLVED' and slpm_quadrant != 'Transition':
                _add_check(validation_checks, validation_errors, "Opportunity Map", False, f"inconsistente {slpm_quadrant}")
            else:
                _add_check(validation_checks, validation_errors, "Opportunity Map", True, f"{slpm_v12_data.get('sector', '')} -> {slpm_quadrant}")
        else:
            _add_check(validation_checks, validation_errors, "Opportunity Map", True, "sin leader_etf")
    else:
        _add_check(validation_checks, validation_errors, "Opportunity Map", True, "sin datos")

    # 7. Data Freshness Dark Pool
    if darkpool_data:
        week = darkpool_data.get('week', '')
        if week:
            try:
                d = pd.Timestamp(week)
                age = (datetime.now() - d).days
                if age > 14:
                    _add_check(validation_checks, validation_errors, "Freshness Dark Pool", True, f"obsoleto {age} dias (advertencia)")
                else:
                    _add_check(validation_checks, validation_errors, "Freshness Dark Pool", True, f"{age} dias")
            except Exception:
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
                age = (datetime.now() - d).days
                if age > 5:
                    _add_check(validation_checks, validation_errors, "Freshness PCR", True, f"desactualizado {age} dias (advertencia)")
                else:
                    _add_check(validation_checks, validation_errors, "Freshness PCR", True, f"{age} dias")
            except Exception:
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
    except Exception as e:
        _add_check(validation_checks, validation_errors, "Config pesos", False, str(e))

    # 10. Anti-Double-Counting
    try:
        import inspect
        from indicators.state_machine import classify_leadership_state
        sig = inspect.signature(classify_leadership_state)
        lis_in_state_machine = 'lis' in sig.parameters

        dc_audit = audit_double_counting()
        critical_vars = len(dc_audit.get('critical', []))
        high_vars = len(dc_audit.get('high', []))

        if lis_in_state_machine:
            _add_check(validation_checks, validation_errors, "Anti-Double-Counting", False, "LIS aun en State Machine")
        elif critical_vars > 0:
            _add_check(validation_checks, validation_errors, "Anti-Double-Counting", True, f"correccion LIS activa, {critical_vars} criticas, {high_vars} altas")
        else:
            _add_check(validation_checks, validation_errors, "Anti-Double-Counting", True, f"correccion LIS activa, sin criticas, {high_vars} compartidas")
    except Exception as e:
        _add_check(validation_checks, validation_errors, "Anti-Double-Counting", False, str(e))

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
    except Exception as e:
        print(f"  [WARN] audit_double_counting: {e}")

    return {
        'passed': passed,
        'checks': validation_checks,
        'errors': validation_errors,
        'dc_summary': dc_summary,
    }
