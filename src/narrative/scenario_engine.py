import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

class ScenarioEngine:
    """
    Genera escenarios macro basados en el estado actual del radar y calibración histórica.
    Busca situaciones pasadas similares y calcula probabilidades empíricas.
    """

    def __init__(self, history_df=None, feature_cols=None, config=None):
        """
        Args:
            history_df: DataFrame con todo el historial (debe incluir las métricas y 'fase_ciclo')
            feature_cols: lista de columnas a usar para la comparación
        """
        self.weights = {
            'base': {'score_global': 1.0, 'neg_stress': 1.0, 'breadth': 0.8, 'liquidity': 0.5},
            'alternativo': {'dispersion': 1.0, 'divergencia': 1.5, 'pend_5d': 0.5},
            'adverso': {'neg_stress': 2.0, 'riesgo_sistemico': 1.5, 'neg_liquidity': 1.0, 'score_global_neg': 1.0}
        }
        self.percentile_boost = {
            'score_global': {'umbral': 90, 'factor': 1.5, 'escenario': 'base'},
            'stress': {'umbral': 90, 'factor': 2.0, 'escenario': 'adverso'},
            'breadth': {'umbral': 10, 'factor': 1.5, 'escenario': 'alternativo'},
            'dispersion': {'umbral': 90, 'factor': 1.5, 'escenario': 'alternativo'},
            'riesgo_sistemico': {'umbral': 90, 'factor': 2.0, 'escenario': 'adverso'}
        }
        self.history_df = history_df
        if feature_cols is None:
            self.feature_cols = ['score_global', 'score_stress', 'score_breadth', 
                                 'score_liquidity', 'score_riesgo_sistemico', 'dispersion']
        else:
            self.feature_cols = feature_cols
        # Escalador para normalizar antes de calcular distancias
        self.scaler = StandardScaler()

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _find_similar_periods(self, snapshot, top_n=5, lookforward_days=30):
        """
        Busca los top_n periodos históricos más similares al snapshot actual.
        Retorna una lista de diccionarios con fecha, distancia, fase posterior y evolución.
        """
        if self.history_df is None or len(self.history_df) < 100:
            return []  # No hay suficiente histórico

        # Extraer vector actual (solo las features numéricas)
        current_vec = []
        for col in self.feature_cols:
            val = snapshot.get(col, 0)
            if col == 'dispersion' and col not in snapshot:
                val = snapshot.get('dispersion', 0)
            current_vec.append(val)
        current_vec = np.array(current_vec).reshape(1, -1)

        # Preparar matriz histórica (excluyendo los últimos 30 días para evitar superposición)
        df_history = self.history_df.iloc[:-30].copy()
        hist_vectors = df_history[self.feature_cols].fillna(0).values

        # Normalizar (ajustar scaler con histórico)
        self.scaler.fit(hist_vectors)
        hist_norm = self.scaler.transform(hist_vectors)
        current_norm = self.scaler.transform(current_vec)

        # Calcular distancias euclídeas
        distances = cdist(current_norm, hist_norm, metric='euclidean').flatten()

        # Obtener índices de los más cercanos
        closest_indices = np.argsort(distances)[:top_n]

        similar = []
        for idx in closest_indices:
            fecha = df_history.iloc[idx]['fecha']
            fase_actual = df_history.iloc[idx]['fase_ciclo']
            dist = distances[idx]

            # Calcular fase posterior (la más frecuente en los siguientes 'lookforward_days')
            if idx + lookforward_days < len(self.history_df):
                fases_futuras = self.history_df.iloc[idx+1:idx+lookforward_days+1]['fase_ciclo'].value_counts()
                fase_futura = fases_futuras.idxmax() if not fases_futuras.empty else "DESCONOCIDA"
                # Score global promedio futuro
                score_futuro = self.history_df.iloc[idx+1:idx+lookforward_days+1]['score_global'].mean()
            else:
                fase_futura = "DESCONOCIDA"
                score_futuro = np.nan

            similar.append({
                'fecha': fecha.strftime('%Y-%m-%d') if hasattr(fecha, 'strftime') else str(fecha),
                'distancia': dist,
                'fase_actual': fase_actual,
                'fase_posterior': fase_futura,
                'score_posterior': score_futuro
            })
        return similar

    def _empirical_probabilities(self, similar_periods):
        """
        Calcula probabilidades de escenarios base/alternativo/adverso basadas en los periodos similares.
        Mapeo simplificado: fase_posterior -> escenario.
        """
        if not similar_periods:
            return None

        # Mapeo de fase a escenario
        fase_to_escenario = {
            'EXPANSION': 'base',
            'ACUMULACION': 'base',
            'EUFORIA': 'base',
            'NEUTRAL': 'alternativo',
            'LATE_CYCLE': 'alternativo',
            'DISTRIBUCION': 'adverso',
            'CONTRACCION': 'adverso',
            'CAPITULACION': 'adverso'
        }
        conteo = {'base': 0, 'alternativo': 0, 'adverso': 0}
        for p in similar_periods:
            esc = fase_to_escenario.get(p['fase_posterior'], 'alternativo')
            conteo[esc] += 1

        total = sum(conteo.values())
        if total == 0:
            return None
        probs = {k: v/total for k, v in conteo.items()}
        return probs

    def generate_scenarios(self, snapshot, return_similar=False):
        """
        Genera escenarios con probabilidades combinadas:
        - 50% del softmax original (Fase 4)
        - 50% de las probabilidades empíricas (si hay histórico)
        Si return_similar=True, además devuelve la lista de periodos similares.
        """
        # Softmax original (Fase 4)
        score = snapshot.get('score_global', 0)
        stress = snapshot.get('score_stress', 0)
        breadth = snapshot.get('score_breadth', 0)
        liquidity = snapshot.get('score_liquidity', 0)
        riesgo = snapshot.get('score_riesgo_sistemico', 0)
        dispersion = snapshot.get('dispersion', 0)
        pend_5d = snapshot.get('pend_5d', 0)

        neg_stress = -stress if stress < 0 else 0
        neg_liquidity = -liquidity if liquidity < 0 else 0
        score_global_neg = -score if score < 0 else 0
        divergencia = max(0, score) * max(0, -breadth)

        base_score = (self.weights['base']['score_global'] * max(0, score) +
                      self.weights['base']['neg_stress'] * neg_stress +
                      self.weights['base']['breadth'] * max(0, breadth) +
                      self.weights['base']['liquidity'] * max(0, liquidity))
        alt_score = (self.weights['alternativo']['dispersion'] * dispersion +
                     self.weights['alternativo']['divergencia'] * divergencia +
                     self.weights['alternativo']['pend_5d'] * max(0, pend_5d))
        adv_score = (self.weights['adverso']['neg_stress'] * neg_stress +
                     self.weights['adverso']['riesgo_sistemico'] * max(0, riesgo) +
                     self.weights['adverso']['neg_liquidity'] * neg_liquidity +
                     self.weights['adverso']['score_global_neg'] * score_global_neg)

        # Ajuste por percentiles
        for metric, cfg in self.percentile_boost.items():
            pct_key = f"{metric}_percentile"
            if pct_key in snapshot and snapshot[pct_key] >= cfg['umbral']:
                if cfg['escenario'] == 'base':
                    base_score *= cfg['factor']
                elif cfg['escenario'] == 'alternativo':
                    alt_score *= cfg['factor']
                elif cfg['escenario'] == 'adverso':
                    adv_score *= cfg['factor']

        scores = np.array([base_score, alt_score, adv_score])
        probs_soft = self._softmax(scores)

        # Probabilidades empíricas
        similar = self._find_similar_periods(snapshot, top_n=10)
        probs_emp = self._empirical_probabilities(similar)
        if probs_emp:
            # Combinar: 50% soft, 50% empírico
            probs = {
                'base': 0.5 * probs_soft[0] + 0.5 * probs_emp.get('base', 0),
                'alternativo': 0.5 * probs_soft[1] + 0.5 * probs_emp.get('alternativo', 0),
                'adverso': 0.5 * probs_soft[2] + 0.5 * probs_emp.get('adverso', 0)
            }
            # Renormalizar (por si acaso)
            total = sum(probs.values())
            probs = {k: v/total for k, v in probs.items()}
        else:
            probs = {
                'base': probs_soft[0],
                'alternativo': probs_soft[1],
                'adverso': probs_soft[2]
            }

        # Narrativa base (similar a la original)
        if base_score > alt_score and base_score > adv_score:
            if score > 0.3:
                base_desc = "Expansión robusta: flujo positivo, estrés controlado, amplitud saludable."
            elif score > 0:
                base_desc = "Crecimiento moderado: flujo positivo pero sin excesos."
            else:
                base_desc = "Fase defensiva: flujo negativo pero estabilizado."
        else:
            base_desc = "Escenario base coherente con el estado actual."

        if alt_score > base_score or divergencia > 0.5:
            alt_desc = "Rally frágil o rotación: flujo positivo estrecho o divergencias. Podría derivar en distribución."
        else:
            alt_desc = "Escenario alternativo por divergencias o alta dispersión."

        if adv_score > 0.5 or stress < -0.4:
            adv_desc = "Riesgo de crisis: estrés extremo o riesgo sistémico elevado."
        else:
            adv_desc = "Escenario adverso por estrés o riesgo sistémico."

        result = {
            'base': {'descripcion': base_desc, 'probabilidad': probs['base']},
            'alternativo': {'descripcion': alt_desc, 'probabilidad': probs['alternativo']},
            'adverso': {'descripcion': adv_desc, 'probabilidad': probs['adverso']}
        }

        if return_similar:
            return result, similar
        return result

    def format_scenarios(self, scenarios):
        lines = []
        for key, sc in scenarios.items():
            lines.append(f"- **{key.capitalize()}** ({sc['probabilidad']*100:.1f}%): {sc['descripcion']}")
        return "\n".join(lines)

    def format_similar_periods(self, similar_periods):
        """Formatea la lista de periodos similares para el informe."""
        if not similar_periods:
            return "No hay suficiente histórico para encontrar situaciones comparables."
        lines = ["\n#### 📅 Situaciones históricas comparables:\n"]
        for p in similar_periods[:5]:  # Mostrar solo los 5 más cercanos
            score_fmt = f" (score futuro: {p['score_posterior']:.2f})" if not np.isnan(p['score_posterior']) else ""
            lines.append(f"- **{p['fecha']}** (distancia {p['distancia']:.3f}): fase actual {p['fase_actual']} → fase posterior {p['fase_posterior']}{score_fmt}")
        return "\n".join(lines)
