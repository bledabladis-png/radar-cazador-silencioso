# src/interpretation/phase_probability.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class PhaseProbability:
    """
    Calcula la probabilidad de cada fase del ciclo basándose en la distancia a centroides históricos.
    """
    def __init__(self, history_df, feature_columns):
        """
        Args:
            history_df: DataFrame con todo el historial (debe incluir las columnas de características)
            feature_columns: lista de nombres de columnas a usar como características (ej. ['score_global', 'score_breadth', ...])
        """
        self.feature_columns = feature_columns
        self.history = history_df[feature_columns].dropna().copy()
        if len(self.history) == 0:
            raise ValueError("No hay datos históricos suficientes para calcular probabilidades.")
        
        # Normalizar características
        self.scaler = StandardScaler()
        self.history_scaled = self.scaler.fit_transform(self.history)
        
        # Definir centroides de fase (aproximados, se pueden ajustar empíricamente)
        # Formato: [score_global, score_breadth, score_stress, ...] en el mismo orden que feature_columns
        # Estos son valores de ejemplo; conviene calcularlos a partir de datos históricos etiquetados.
        self.centroids = {
            'CONTRACCION': np.array([-0.6, -0.5, -0.4]),
            'CAPITULACION': np.array([-0.8, -0.7, -0.8]),
            'ACUMULACION': np.array([-0.2, -0.1, 0.0]),
            'EXPANSION': np.array([0.3, 0.4, 0.2]),
            'EUFORIA': np.array([0.7, 0.6, -0.2]),
            'LATE_CYCLE': np.array([0.2, -0.3, -0.1]),
            'NEUTRAL': np.array([0.0, 0.0, 0.0])
        }
        # Asegurar que los centroides tengan la misma dimensión que las características
        # Si no, se ajustan a la longitud correcta (rellenando con ceros o tomando los primeros)
        for phase in self.centroids:
            if len(self.centroids[phase]) < len(feature_columns):
                # Rellenar con ceros
                pad = np.zeros(len(feature_columns) - len(self.centroids[phase]))
                self.centroids[phase] = np.concatenate([self.centroids[phase], pad])
            elif len(self.centroids[phase]) > len(feature_columns):
                # Recortar
                self.centroids[phase] = self.centroids[phase][:len(feature_columns)]

    def distances_to_centroids(self, sample):
        """
        Calcula la distancia euclidiana (normalizada) desde la muestra actual a cada centroide.
        sample: dict o array con los valores de las características en el mismo orden que feature_columns.
        Retorna dict {fase: distancia}
        """
        if isinstance(sample, dict):
            # Convertir a array en el orden correcto
            x = np.array([sample[col] for col in self.feature_columns])
        else:
            x = np.array(sample)
        # Normalizar con el scaler
        x_scaled = self.scaler.transform(x.reshape(1, -1)).flatten()
        dist = {}
        for phase, centroid in self.centroids.items():
            # centroid ya debería estar en el espacio original, normalizamos
            centroid_scaled = self.scaler.transform(centroid.reshape(1, -1)).flatten()
            dist[phase] = np.linalg.norm(x_scaled - centroid_scaled)
        return dist

    def softmax(self, distances, temperature=1.0):
        """
        Convierte distancias en probabilidades usando softmax negativo.
        """
        d = np.array(list(distances.values()))
        # Negativo porque menor distancia = mayor probabilidad
        exp_neg = np.exp(-d / temperature)
        probs = exp_neg / exp_neg.sum()
        return dict(zip(distances.keys(), probs))

    def get_probabilities(self, sample, temperature=1.0):
        """
        Devuelve un dict {fase: probabilidad} para la muestra actual.
        """
        dist = self.distances_to_centroids(sample)
        return self.softmax(dist, temperature)

    def get_top_phases(self, sample, n=3, temperature=1.0):
        """
        Devuelve las n fases con mayor probabilidad.
        """
        probs = self.get_probabilities(sample, temperature)
        sorted_phases = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_phases[:n]

    def format_probabilities(self, probs, decimals=1):
        """
        Formatea las probabilidades para el informe.
        """
        lines = []
        for phase, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {phase}: {prob*100:.{decimals}f}%")
        return "\n".join(lines)
