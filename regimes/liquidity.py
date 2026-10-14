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







    # N8 fix (2026-10-14, dictamen auditor): tambien guardamos la
    # observation_date (nativa) por componente para trazabilidad y
    # para validar frescura individual. Distinta de as_of_date.
    signals = {}
    signals_obs = {}

    if 'fed_balance' in fed_data.columns:
        z = robust_zscore(fed_data['fed_balance'], window=60)
        t = tanh_normalize(z)
        t_nn = t.dropna()
        if t_nn.size > 0 and pd.notna(t_nn.iloc[-1]):
            signals['fed_balance'] = t_nn.iloc[-1]
            signals_obs['fed_balance'] = t_nn.index[-1]

    if 'reverse_repo' in fed_data.columns:
        z = robust_zscore(fed_data['reverse_repo'], window=60)
        t = -tanh_normalize(z)
        t_nn = t.dropna()
        if t_nn.size > 0 and pd.notna(t_nn.iloc[-1]):
            signals['reverse_repo'] = t_nn.iloc[-1]
            signals_obs['reverse_repo'] = t_nn.index[-1]

    if 'sofr' in fed_data.columns:
        z = robust_zscore(fed_data['sofr'], window=60)
        t = -tanh_normalize(z)
        t_nn = t.dropna()
        if t_nn.size > 0 and pd.notna(t_nn.iloc[-1]):
            signals['sofr'] = t_nn.iloc[-1]
            signals_obs['sofr'] = t_nn.index[-1]

    if 'fed_funds' in fed_data.columns:
        z = robust_zscore(fed_data['fed_funds'], window=60)
        t = -tanh_normalize(z)
        t_nn = t.dropna()
        if t_nn.size > 0 and pd.notna(t_nn.iloc[-1]):
            signals['fed_funds'] = t_nn.iloc[-1]
            signals_obs['fed_funds'] = t_nn.index[-1]

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







    # N8 fix (2026-10-14, dictamen auditor A-D): persistencia con
    # trazabilidad de fechas, umbrales de frescura por serie y guards
    # de monotonicidad + calidad.
    previous_score = None
    previous_date = None
    previous_stale = False

    # Umbrales de frescura por frecuencia de publicacion FRED (dias
    # calendario). No universal: cada serie con su calendario.
    # SOFR/RRPONTSYD diarias T+1 -> 4 dias (cubre fin de semana + margen).
    # WALCL semanal (miercoles, publicado jueves) -> 10 dias.
    # FEDFUNDS mensual (primer dia del mes siguiente) -> 45 dias.
    FRESHNESS_MAX_DAYS = {
        'sofr': 4,
        'reverse_repo': 4,
        'fed_balance': 10,
        'fed_funds': 45,
    }

    # Fecha de calculo del score (wall clock). Distinta de observation_date.
    computed_at = pd.Timestamp.now().normalize()

    # Diagnostico de frescura por componente.
    stale_components = []
    for _name, _obs in signals_obs.items():
        _max_days = FRESHNESS_MAX_DAYS.get(_name, 7)
        _age = (computed_at - pd.Timestamp(_obs).normalize()).days
        if _age > _max_days:
            stale_components.append((_name, str(_obs.date()), _age, _max_days))

    stale = len(stale_components) > 0
    stale_reason = None
    if stale:
        _parts = [f"{n} ({d}, {a}d>{m}d)" for n, d, a, m in stale_components]
        stale_reason = "componentes stale: " + ", ".join(_parts)

    # Fecha as_of del score: max observation_date entre componentes con
    # dato valido. Distinta de computed_at y de observation_date por
    # componente.
    if signals_obs:
        as_of_date = max(signals_obs.values())
    else:
        as_of_date = fed_data.index[-1]

    try:
        import json, os
        delta_file = 'outputs/state/liquidity_state.json'
        if os.path.exists(delta_file):
            with open(delta_file, 'r') as f:
                prev = json.load(f)
                previous_score = prev.get('score', None)
                previous_date = prev.get('date', None)
                previous_stale = prev.get('stale', False)

        last_idx = fed_data.dropna(how='all').index[-1] if len(fed_data.dropna(how='all')) > 0 else fed_data.index[-1]

        skip_persist = False
        skip_reason = None

        # Guard 1 (monotonicidad): fecha calculada < fecha persistida.
        # Bug observado: fe98c29a (13-oct) state date=2026-10-13;
        #               5e58197a (14-oct) state date=2026-10-07.
        if previous_date is not None:
            try:
                prev_ts = pd.Timestamp(previous_date)
                if last_idx < prev_ts:
                    skip_persist = True
                    skip_reason = f"fecha calculada {last_idx.date()} < persistida {prev_ts.date()}"
            except (ValueError, TypeError) as e:
                print(f"  [WARN] liquidity_state: previous_date invalida ({previous_date!r}): {e}")

        # Guard 2 (frescura): el calculo actual tiene componentes stale
        # y el persistido no. No sobreescribir un estado fresco con uno
        # de calidad inferior aunque la fecha sea >=.
        if stale and not previous_stale and previous_date is not None:
            skip_persist = True
            skip_reason = stale_reason

        if skip_persist:
            print(f"  [WARN] liquidity_state: NO se sobreescribe. Motivo: {skip_reason}")
            return pd.Series(score, index=[as_of_date]), regime, confidence, previous_score

        # Persistencia extendida con trazabilidad.
        state = {
            'score': float(score),
            'date': str(as_of_date.date()),
            'computed_at': str(computed_at.date()),
            'components': {
                name: {
                    'observation_date': str(pd.Timestamp(obs).date()),
                    'value': float(signals[name]),
                }
                for name, obs in signals_obs.items()
            },
            'stale': stale,
            'stale_reason': stale_reason,
        }
        with open(delta_file, 'w') as f:
            json.dump(state, f)
    except (json.JSONDecodeError, OSError, ValueError, TypeError, KeyError, IndexError) as e:
        # A3.1-01: fallo de persistencia de state no debe tumbar el pipeline,
        # pero tampoco debe ser invisible. previous_score queda en None.
        print(f"  [WARN] liquidity_state no actualizado: {e}")







    return pd.Series(score, index=[as_of_date]), regime, confidence, previous_score



