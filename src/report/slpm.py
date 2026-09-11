# -*- coding: utf-8 -*-
"""SLPM v1.2 y Legacy v1.0 (Structural Leadership).

Extraidos de src/report_generator.py (refactor C1, fase C1-6d1).
"""

from config.weights import SLPM_WEIGHTS
from src.report.helpers import _fmt_num


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
        quadrant = slpm_v12_data.get('opportunity_quadrant', 'N/A')
        out.append(f"- **Sector Líder:** {sector}\n")
        out.append("  - *Nota: El SLPM selecciona al líder combinando Structural, Breadth y Persistence. Tactical y LIS son métricas diagnósticas. No es simplemente el sector con mayor Structural Score.*\n")
        out.append(f"- **Estado:** {state_v12}")
        if quadrant:
            out.append(f" -> {quadrant}")
        out.append("\n")
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
            lis_val = slpm_v12_data.get("leader_integrity", {}).get("lis", 0) if slpm_v12_data else 0
            out.append(f"- **Scores oficiales:** T={tact_val:+.2f} | S={struct_val:+.2f} | LIS={lis_val:+.2f} | Eff Breadth={eff_breadth:.2f} | Persist={pers_str} | LQ: P={tact_val:+.2f} C={_fmt_num(flow_val, '{:+.3f}')} S={struct_val:+.2f} Cf={lis_val:+.2f}\n")
        
        errors = slpm_v12_data.get('validation_errors', [])
        if errors:
            out.append("\nERRORES DE VALIDACION:\n")
            for e in errors:
                out.append(f"  - {e}\n")
        
        breadth = slpm_v12_data.get('leader_breadth_v2', {})
        if breadth:
            out.append("\n### Leader Breadth & Health\n")
            rs_b = breadth.get('rs_breadth', 0)*100
            mom_b = breadth.get('momentum_breadth', 0)*100
            flow_b = breadth.get('flow_breadth', 0)*100
            wyck_b = breadth.get('wyckoff_breadth', 0)*100
            comp = breadth.get('composite', 0)*100
            effective = breadth.get('effective_composite', 0)*100
            n = breadth.get('n_used', 0)
            total = breadth.get('expected_leaders', 5)
            coverage = breadth.get('coverage', 0)*100
            out.append(f"- **Leader Breadth (RS ratio > 1.0):** {rs_b:.0f}%\n")
            out.append(f"- **Leader Momentum Breadth:** {mom_b:.0f}%\n")
            out.append(f"- **Leader Flow Support:** {flow_b:.0f}%\n")
            out.append(f"- **Leader Wyckoff Health:** {wyck_b:.0f}%\n")
            out.append("  - *Scoring Wyckoff: MARKUP=1.0, ACCUMULATION=0.75, RANGE=0.0, DISTRIBUTION=-0.75, MARKDOWN=-1.0*\n")
            out.append(f"- **Leader Health Composite (sin ajustar):** {comp:.0f}% ")
            out.append(f"({SLPM_WEIGHTS['leader_breadth']['rs']:.2f}xRS + {SLPM_WEIGHTS['leader_breadth']['momentum']:.2f}xMom + {SLPM_WEIGHTS['leader_breadth']['flow']:.2f}xFlow + {SLPM_WEIGHTS['leader_breadth']['wyckoff']:.2f}xWyckoff)\n")
            out.append(f"- **Effective Breadth:** {effective:.0f}% (Health Composite: {comp:.0f}%, Cobertura: {coverage:.0f}%) — Regla: si cobertura >= 50% no se aplica penalización\n")
            out.append(f"  - N analizado: {n}/{total}\n")
            out.append("  - *Nota: Effective Breadth = Health Composite (sin ajuste cuando cobertura >= 50%). La penalización por cobertura solo se aplica cuando la cobertura es inferior al 50%. La calidad observada (Health Composite) es independiente de la cobertura.*\n")
        
        integrity = slpm_v12_data.get('leader_integrity', {})
        if integrity:
            lis = integrity.get('lis', 0)
            n_leaders = integrity.get('n_leaders', 0)
            out.append("\n### Leader Integrity Score (LIS)\n")
            out.append(f"- **LIS:** {lis:+.2f} (n={n_leaders})\n")
            out.append(f"- *Formula: LIS_individual = {SLPM_WEIGHTS['lis']['rs']:.2f}*tanh((RS-1)*2) + {SLPM_WEIGHTS['lis']['momentum']:.2f}*tanh(RS_mom*5) + {SLPM_WEIGHTS['lis']['flow']:.2f}*tanh(flow_proxy_z/2) + {SLPM_WEIGHTS['lis']['wyckoff']:.2f}*Wyckoff_score. LIS = media.*\n")
            out.append("- *LIS mide la intensidad/calidad de la señal de los lideres, no el % que cumple condiciones (eso es el Breadth).*\n")
        
        flow_div = slpm_v12_data.get('flow_divergence_v2', {})
        if flow_div:
            out.append("\n### Flow Divergence 2.0\n")
            out.append(f"- **Composite:** {_fmt_num(flow_div.get('composite'), '{:+.3f}')}\n")
            out.append(f"  - Leader vs Sector: {_fmt_num(flow_div.get('leader_flow_div'), '{:+.3f}')}\n")
            out.append(f"  - Sector Flow vs Price: {_fmt_num(flow_div.get('sector_flow_vs_price_div'), '{:+.3f}')}\n")
            out.append(f"  - Structural: {_fmt_num(flow_div.get('structural_flow_div'), '{:+.3f}')}\n")
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
        out.append(f"- **Structural RS:** {slpm_data.get('struct_rs', 0):+.3f}\n")
        out.append(f"- **Leader Breadth:** {slpm_data.get('leader_breadth', 0)*100:.0f}%\n")
        out.append(f"- **Flow Divergence:** {slpm_data.get('flow_divergence', 0):+.3f}\n")
        out.append(f"- **Tactical Score (legacy):** {slpm_data.get('tactical_score', 0):+.3f}\n")
        out.append(f"- **Structural Score (legacy):** {slpm_data.get('structural_score', 0):+.3f}\n")
        out.append("\n</details>\n\n")
    return out
