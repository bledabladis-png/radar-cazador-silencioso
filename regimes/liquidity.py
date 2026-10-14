import pandas as pd






from data.providers.router import DataRouter



from src.utils import safe_std, robust_zscore, tanh_normalize







def compute_liquidity_score():



    router = DataRouter()



    fed_data = router.get_fed_data()



    if fed_data is None or fed_data.empty:
        # A3.1-03 (2026-09-28): aridad uniforme 4-tuple (None, None, None, None)
        # para que el caller no dependa de un contrato implicito 3-vs-4.
        return None, None, None, None







    # Rellenar hacia adelante para evitar que NaN en la ultima fila anule las senales



    # fed_data = fed_data.ffill()  # imputación eliminada: datos reales







    signals = {}

    if 'fed_balance' in fed_data.columns:
        z = robust_zscore(fed_data['fed_balance'], window=60)
        t = tanh_normalize(z)
        last_val = t.dropna().iloc[-1] if t.dropna().size > 0 else float('nan')
        if pd.notna(last_val):
            signals['fed_balance'] = last_val

    if 'reverse_repo' in fed_data.columns:
        z = robust_zscore(fed_data['reverse_repo'], window=60)
        t = -tanh_normalize(z)
        last_val = t.dropna().iloc[-1] if t.dropna().size > 0 else float('nan')
        if pd.notna(last_val):
            signals['reverse_repo'] = last_val

    if 'sofr' in fed_data.columns:
        z = robust_zscore(fed_data['sofr'], window=60)
        t = -tanh_normalize(z)
        last_val = t.dropna().iloc[-1] if t.dropna().size > 0 else float('nan')
        if pd.notna(last_val):
            signals['sofr'] = last_val

    if 'fed_funds' in fed_data.columns:
        z = robust_zscore(fed_data['fed_funds'], window=60)
        t = -tanh_normalize(z)
        last_val = t.dropna().iloc[-1] if t.dropna().size > 0 else float('nan')
        if pd.notna(last_val):
            signals['fed_funds'] = last_val

    if not signals:
        return None, None, None, None







    weights = {'fed_balance': 0.35, 'reverse_repo': 0.25, 'sofr': 0.20, 'fed_funds': 0.20}



    available = [k for k in weights if k in signals]



    if not available:
        return None, None, None, None







    w_sum = sum(weights[k] for k in available)



    score = sum(signals[k] * weights[k] / w_sum for k in available)







    if score > 0.3:



        regime = 'ABUNDANTE'



    elif score > 0:



        regime = 'NEUTRAL'



    elif score > -0.3:



        regime = 'ESTRECHA'



    else:



        regime = 'CRISIS'







    sig_vals = [signals[k] for k in available]



    if len(sig_vals) > 1:



        confidence = 1 - safe_std(sig_vals) / 2



    elif len(sig_vals) == 1:



        confidence = 0.5  # una sola senal -> confianza neutra



    else:



        confidence = 0.0







    # Guardar score para calculo de delta en la siguiente ejecucion



    previous_score = None
    previous_date = None



    try:
        import json, os
        delta_file = 'outputs/state/liquidity_state.json'
        if os.path.exists(delta_file):
            with open(delta_file, 'r') as f:
                prev = json.load(f)
                previous_score = prev.get('score', None)
                previous_date = prev.get('date', None)
        # N8 fix (2026-10-14): guard de monotonicidad. Si el run actual
        # calcula con datos MAS ANTIGUOS que el persistido (cache stale,
        # latencia FRED, parquets con lag), NO sobreescribir.
        # Bug observado: fe98c29a (13-oct) state date=2026-10-13;
        #               5e58197a (14-oct) state date=2026-10-07.
        last_idx = fed_data.dropna(how='all').index[-1] if len(fed_data.dropna(how='all')) > 0 else fed_data.index[-1]
        if previous_date is not None:
            try:
                prev_ts = pd.Timestamp(previous_date)
                if last_idx < prev_ts:
                    print(f"  [WARN] liquidity_state: fecha calculada {last_idx.date()} < persistida {prev_ts.date()}. NO se sobreescribe. Posible dato stale.")
                    return pd.Series(score, index=[last_idx]), regime, confidence, previous_score
            except (ValueError, TypeError) as e:
                print(f"  [WARN] liquidity_state: previous_date invalida ({previous_date!r}): {e}")
        with open(delta_file, 'w') as f:
            json.dump({'score': float(score), 'date': str(last_idx.date())}, f)
    except (json.JSONDecodeError, OSError, ValueError, TypeError, KeyError, IndexError) as e:
        # A3.1-01: fallo de persistencia de state no debe tumbar el pipeline,
        # pero tampoco debe ser invisible. previous_score queda en None.
        print(f"  [WARN] liquidity_state no actualizado: {e}")







    last_idx = fed_data.dropna(how='all').index[-1] if len(fed_data.dropna(how='all')) > 0 else fed_data.index[-1]
    return pd.Series(score, index=[last_idx]), regime, confidence, previous_score



