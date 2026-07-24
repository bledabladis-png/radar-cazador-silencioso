# Añadir acuerdo Wyckoff en la sección de líderes sectoriales
if leader_lines and sector_results:
    for i, (ticker, name, score, wyckoff) in enumerate(sector_results['ranking'][:11], 1):
        # Contar líderes en fase favorable
        leaders_in_favor = sum(1 for line in leader_lines if f'Sector: {ticker}' in line)
        if leaders_in_favor > 0:
            # Buscar fases de los líderes
            leader_phases = []
            capture = False
            for line in leader_lines:
                if f'Sector: {ticker}' in line:
                    capture = True
                    continue
                if capture and line.startswith('|') and '---' not in line:
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    if len(parts) >= 6 and parts[0] not in ('Ticker', ''):
                        leader_phases.append(parts[5] if len(parts) > 5 else '')
                if capture and line.strip() == '':
                    break
            if leader_phases:
                favorable = sum(1 for p in leader_phases if p in ('ACCUMULATION', 'MARKUP'))
                agreement = favorable / len(leader_phases)
                lines.append(f"- **{name} ({ticker}):** Wyckoff Agreement = {agreement:.0%} ({favorable}/{len(leader_phases)} líderes en fase favorable)\n")
