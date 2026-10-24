# -*- coding: utf-8 -*-
"""SLPM v1.2 y Legacy v1.0 (Structural Leadership).

Extraidos de src/report_generator.py (refactor C1, fase C1-6d1).
"""

import math

from config.weights import SLPM_WEIGHTS
from src.report.helpers import _fmt_signed


def render_slpm_v12(slpm_v12_data):
    """Renderiza la seccion SLPM v1.2.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if slpm_v12_data:
        breadth = slpm_v12_data.get('leader_breadth_v2', {})
        if breadth:
            total = breadth.get('expected_leaders', 5)
            n = breadth.get('n_used', 0)
            coverage = breadth.get('coverage', 0)
            out.append(f"*Cobertura de lideres SLPM: {n}/{total} ({coverage:.0%})*")
            if breadth.get('coverage_warning', False):
                out.append(" - ADVERTENCIA: Cobertura baja, resultados con incertidumbre elevada.")
            out.append("\n\n")

        out.append("## Structural Leadership (SLPM v1.2)\n")
        sector = slpm_v12_data.get('sector', 'N/A')
        state_v12 = slpm_v12_data.get('state', 'N/A')
        reason = slpm_v12_data.get('state_reason', '')
        reason_code = slpm_v12_data.get('state_reason_code', '')
        quadrant = slpm_v12_data.get('opportunity_quadrant', 'N/A')
        # H8 (2026-10-15, auditoria externa): no presentar "Sector Lider"
        # como confirmado cuando el estado es UNRESOLVED por falta de datos.
        # Cuando el reason_code indica insuficiencia, la etiqueta cambia a
        # "Candidato a lider" con nota explicita.
        _no_confirmado = (
            state_v12 == 'UNRESOLVED'
            and reason_code in (
                'NO_LEADERS', 'INPUT_UNAVAILABLE',
                'INSUFFICIENT_DATA_RS', 'INSUFFICIENT_COMPONENT_DATA',
                'INSUFFICIENT_COVERAGE', 'INSUFFICIENT_DATA_BREADTH',
            )
        )
        if _no_confirmado:
            out.append(f"- **Candidato a líder (no confirmado):** {sector}\n")
            # H3 (2026-10-16, auditoria): la frase previa afirmaba que
            # el sector mostrado era "el de mayor Structural Score".
            # Empiricamente no se sostiene (XLP con Structural -0.02 vs
            # XLE con +0.20). El sector mostrado es el #1 del ranking
            # estructural general (combinando distintos motores). Se
            # reformula sin afirmar un criterio que no se cumple.
            # H3 (2026-10-16, auditor): el texto anterior afirmaba
            # "el de mayor Structural Score". Empiricamente no se
            # sostiene (XLP -0.02 vs XLE +0.20). Ademas, sin acceso al
            # ranking completo desde este render, no podemos afirmar
            # la posicion relativa. Texto honesto: candidato #1 del
            # ranking compuesto, sin prometer el Structural maximo.
            out.append(f"  - *El SLPM no confirma liderazgo: {reason_code}. El sector mostrado es el candidato #1 del ranking compuesto general en este snapshot; su Structural Score puede no ser el maximo del universo. Se presenta como candidato, no como liderazgo validado.*\n")
        else:
            out.append(f"- **Sector Líder:** {sector}\n")
            out.append("  - *Nota: El SLPM selecciona al líder combinando Structural, Breadth y Persistence. Tactical y LIS son métricas diagnósticas. No es simplemente el sector con mayor Structural Score.*\n")
        out.append(f"- **Estado:** {state_v12}")
        if quadrant:
            out.append(f" -> {quadrant}")
        out.append("\n")
        if reason_code and reason_code not in ('', 'UNKNOWN'):
            out.append(f"  - *Motivo: {reason_code}*\n")
        if reason:
            out.append(f"  - *{reason}*\n")
        
        inputs = slpm_v12_data.get('input_scores', {})
        if inputs:
            eff_breadth = inputs.get('effective_breadth', inputs.get('breadth', 0))
            pers_val = inputs.get('persistence')
            pers_str = f"{pers_val:.0%}" if pers_val is not None else "N/A"
            tact_val = inputs.get("tactical", 0)
            flow_val = slpm_v12_data.get("flow_divergence_v2", {}).get("composite", 0) if slpm_v12_data else 0
            struct_val = inputs.get("structural", 0)
            # FU-003b (2026-09-15): con n=0 el valor +0.00 es enganoso;
            # la lectura correcta es N/D. Extiende FU-013 a LIS, Eff Breadth
            # y Cf de la linea compacta "Scores oficiales".
            integrity = slpm_v12_data.get("leader_integrity", {}) if slpm_v12_data else {}
            lis_val = integrity.get("lis", 0)
            n_lis = integrity.get("n_leaders", 0)
            lis_str = _fmt_signed(lis_val, '{:+.2f}', '{:.2f}') if n_lis > 0 else "N/D"
            n_breadth = breadth.get("n_used", 0) if breadth else 0
            eff_breadth_str = f"{eff_breadth:.2f}" if n_breadth > 0 else "N/D"
            out.append(f"- **Scores oficiales:** T={_fmt_signed(tact_val, '{:+.2f}', '{:.2f}')} | S={_fmt_signed(struct_val, '{:+.2f}', '{:.2f}')} | LIS={lis_str} | Eff Breadth={eff_breadth_str} | Persist={pers_str} | LQ: P={_fmt_signed(tact_val, '{:+.2f}', '{:.2f}')} C={_fmt_signed(flow_val, '{:+.3f}', '{:.3f}')} S={_fmt_signed(struct_val, '{:+.2f}', '{:.2f}')} Cf={lis_str}\n")
        
        errors = slpm_v12_data.get('validation_errors', [])
        if errors:
            out.append("\nERRORES DE VALIDACION:\n")
            for e in errors:
                out.append(f"  - {e}\n")
        
        breadth = slpm_v12_data.get('leader_breadth_v2', {})
        if breadth:
            out.append("\n### Leader Breadth & Health\n")

            def _is_finite(v):
                if v is None:
                    return False
                try:
                    return math.isfinite(v)
                except TypeError:
                    return False

            def _pct(v):
                return f"{v*100:.0f}%" if _is_finite(v) else "N/D"

            lb_status = breadth.get('leader_breadth_status', 'UNKNOWN')
            rs_missing = breadth.get('rs_breadth_missing', 0)
            n = breadth.get('n_used', 0)
            total = breadth.get('expected_leaders', 5)
            coverage = breadth.get('coverage', 0) * 100

            if lb_status == 'OK':
                out.append(f"- **Leader Breadth (rs_ratio_20d > 0):** {_pct(breadth.get('rs_breadth'))}\n")
                out.append(f"- **Leader Momentum Breadth:** {_pct(breadth.get('momentum_breadth'))}\n")
                out.append(f"- **Leader Flow Support:** {_pct(breadth.get('flow_breadth'))}\n")
                out.append(f"- **Leader Wyckoff Health:** {_pct(breadth.get('wyckoff_breadth'))}\n")
            else:
                if lb_status == 'INPUT_UNAVAILABLE':
                    motivo = "entrada no disponible"
                elif lb_status == 'NO_LEADERS':
                    motivo = "sin lideres evaluables"
                elif lb_status == 'INSUFFICIENT_RS_RATIO':
                    motivo = f"rs_ratio_20d ausente en {rs_missing} lideres"
                elif lb_status == 'INSUFFICIENT_COMPONENT_DATA':
                    motivo = "componentes de breadth incompletos"
                else:
                    motivo = f"status={lb_status}"
                out.append(f"- **Leader Breadth (rs_ratio_20d > 0):** N/D ({motivo})\n")
                out.append(f"- **Leader Momentum Breadth:** {_pct(breadth.get('momentum_breadth'))}\n")
                out.append(f"- **Leader Flow Support:** {_pct(breadth.get('flow_breadth'))}\n")
                out.append(f"- **Leader Wyckoff Health:** {_pct(breadth.get('wyckoff_breadth'))}\n")

            out.append("  - *Scoring Wyckoff: MARKUP=1.0, ACCUMULATION=0.75, RANGE=0.0, DISTRIBUTION=-0.75, MARKDOWN=-1.0, DISTRIBUTION_FORMING=0.0, ACCUMULATION_FORMING=0.0, CONFLICT=0.0, INSUFFICIENT_DATA=0.0*\n")

            comp_v = breadth.get('composite')
            comp_valid = (lb_status == 'OK' and _is_finite(comp_v))
            if comp_valid:
                out.append(f"- **Leader Health Composite (sin ajustar):** {_pct(comp_v)} ")
            else:
                out.append("- **Leader Health Composite (sin ajustar):** N/D ")
            out.append(f"({SLPM_WEIGHTS['leader_breadth']['rs']:.2f}xRS + {SLPM_WEIGHTS['leader_breadth']['momentum']:.2f}xMom + {SLPM_WEIGHTS['leader_breadth']['flow']:.2f}xFlow + {SLPM_WEIGHTS['leader_breadth']['wyckoff']:.2f}xWyckoff)\n")

            if comp_valid:
                out.append(f"- **Effective Breadth:** {_pct(breadth.get('effective_composite'))} (Health Composite: {_pct(comp_v)}, Cobertura: {coverage:.0f}%) - Regla: si cobertura >= 50% no se aplica penalizacion\n")
            else:
                out.append(f"- **Effective Breadth:** N/D (Cobertura: {n}/{total})\n")
            out.append(f"  - N analizado: {n}/{total}\n")
            out.append("  - *Nota: Effective Breadth = Health Composite (sin ajuste cuando cobertura >= 50%). La penalizacion por cobertura solo se aplica cuando la cobertura es inferior al 50%. La calidad observada (Health Composite) es independiente de la cobertura.*\n")
        
        integrity = slpm_v12_data.get('leader_integrity', {})
        if integrity:
            lis = integrity.get('lis', 0)
            n_leaders = integrity.get('n_leaders', 0)
            out.append("\n### Leader Integrity Score (LIS)\n")
            # FU-003b (2026-09-15): n=0 -> N/D (extiende FU-013 a LIS).
            if n_leaders == 0:
                out.append("- **LIS:** N/D (n=0)\n")
            else:
                out.append(f"- **LIS:** {_fmt_signed(lis, '{:+.2f}', '{:.2f}')} (n={n_leaders})\n")
            out.append(f"- *Formula: LIS_individual = {SLPM_WEIGHTS['lis']['rs']:.2f}*tanh((RS-1)*2) + {SLPM_WEIGHTS['lis']['momentum']:.2f}*tanh(RS_mom*5) + {SLPM_WEIGHTS['lis']['flow']:.2f}*tanh(flow_proxy_z/2) + {SLPM_WEIGHTS['lis']['wyckoff']:.2f}*Wyckoff_score. LIS = media.*\n")
            out.append("- *LIS mide la intensidad/calidad de la señal de los lideres, no el % que cumple condiciones (eso es el Breadth).*\n")
            # N2 (2026-10-13, auditor): LIS declarado provisional.
            out.append("- *N2 auditoria 2026-10-13: LIS usa `rs` "
                       "(ratio precio nominal accion/ETF) como input, no "
                       "`rs_ratio_20d`. PROVISIONAL, no validado como "
                       "senal decisoria. Requiere recalibracion. Ver "
                       "TRIAGE del auditor 2026-10-13.*\n")
        
        flow_div = slpm_v12_data.get('flow_divergence_v2', {})
        if flow_div:
            out.append("\n### Flow Divergence 2.0\n")
            out.append(f"- **Composite:** {_fmt_signed(flow_div.get('composite'), '{:+.3f}', '{:.3f}')}\n")
            out.append(f"  - Leader vs Sector: {_fmt_signed(flow_div.get('leader_flow_div'), '{:+.3f}', '{:.3f}')}\n")
            out.append(f"  - Sector Flow vs Price: {_fmt_signed(flow_div.get('sector_flow_vs_price_div'), '{:+.3f}', '{:.3f}')}\n")
            out.append(f"  - Structural: {_fmt_signed(flow_div.get('structural_flow_div'), '{:+.3f}', '{:.3f}')}\n")
            out.append("- *Nota: el composite pondera 3 componentes (leader_flow_div, sector_flow_vs_price_div, structural_flow_div). El segundo no tiene fuente en el pipeline actual (slpm_v12.py:266 pasa None). Mientras no se cablee, el composite sale N/D por diseno. No es error de calculo.*\n")
            out.append("- *Nota: Flujo medido como Flow Proxy (retorno x volumen). No implica flujo institucional real.*\n")
        out.append("\n")
    return out


def render_slpm_legacy(slpm_data):
    """Renderiza el bloque Legacy SLPM v1.0 (dentro de <details>).

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if slpm_data:
        out.append("<details>\n<summary><b>Legacy SLPM v1.0 (referencia historica)</b></summary>\n\n")
        state = slpm_data.get('state', 'N/A')
        out.append(f"- **Sector Líder:** {slpm_data.get('sector', 'N/A')} ({slpm_data.get('sector_etf', '')})\n")
        out.append(f"- **Estado:** {state}\n")
        out.append(f"- **Structural RS:** {_fmt_signed(slpm_data.get('struct_rs', 0), '{:+.3f}', '{:.3f}')}\n")
        out.append(f"- **Leader Breadth:** {slpm_data.get('leader_breadth', 0)*100:.0f}%\n")
        out.append(f"- **Flow Divergence:** {_fmt_signed(slpm_data.get('flow_divergence', 0), '{:+.3f}', '{:.3f}')}\n")
        out.append(f"- **Tactical Score (legacy):** {_fmt_signed(slpm_data.get('tactical_score', 0), '{:+.3f}', '{:.3f}')}\n")
        out.append(f"- **Structural Score (legacy):** {_fmt_signed(slpm_data.get('structural_score', 0), '{:+.3f}', '{:.3f}')}\n")
        out.append("\n</details>\n\n")
    return out
