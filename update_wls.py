with open('indicators/stock_leader.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = "    df['sector_rank_pct'] = df.groupby('sector')['wls'].rank(pct=True)\n    return df.sort_values('wls', ascending=False)"

new = """    df['sector_rank_pct'] = df.groupby('sector')['wls'].rank(pct=True)

    # Rank Stability (leader_confidence)
    df['leader_confidence'] = np.nan
    for sector in df['sector'].unique():
        mask = df['sector'] == sector
        sector_df = df[mask]
        if len(sector_df) >= 20:
            current_rank = sector_df['wls'].rank()
            historical_rank = sector_df['wls'].iloc[:-20].rank()
            common_idx = current_rank.index.intersection(historical_rank.index)
            if len(common_idx) > 5:
                rho = current_rank.loc[common_idx].corr(historical_rank.loc[common_idx], method='spearman')
                df.loc[mask, 'leader_confidence'] = (rho + 1) / 2

    return df.sort_values('wls', ascending=False)"""

content = content.replace(old, new)
with open('indicators/stock_leader.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Rank Stability añadido.')
