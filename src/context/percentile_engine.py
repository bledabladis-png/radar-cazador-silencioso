import pandas as pd
import numpy as np
from scipy.stats import percentileofscore

class PercentileEngine:
    """
    Calcula percentiles históricos para las principales métricas del radar.
    Permite contextualizar los valores actuales dentro de la distribución histórica.
    Soporta ventanas móviles (rolling).
    """
    
    def __init__(self, history_df, thresholds=None):
        """
        Args:
            history_df: DataFrame con todo el historial (debe incluir las columnas)
            thresholds: dict con umbrales para clasificación (ej. {'extremo_alto': 90, 'alto': 75, ...})
                       Si es None, se usan valores por defecto.
        """
        self.history = history_df.copy()
        self.metrics = {}
        
        # Umbrales por defecto (coinciden con los originales de describe)
        self.thresholds = thresholds or {
            'extremo_alto': 90,
            'alto': 75,
            'bajo': 25,
            'extremo_bajo': 10
        }
        
    def add_metric(self, name, series=None):
        """
        Almacena una serie histórica para una métrica.
        Si no se proporciona, se intenta extraer del DataFrame.
        """
        if series is None:
            if name in self.history.columns:
                series = self.history[name].dropna()
            else:
                raise ValueError(f"La métrica '{name}' no está en el historial.")
        self.metrics[name] = series
        
    def percentile(self, metric_name, value, window=None):
        """
        Devuelve el percentil (0-100) del valor actual dentro de la serie histórica.
        Si se especifica window, se usa solo las últimas 'window' observaciones.
        Si el valor es NaN o la serie está vacía, devuelve NaN.
        """
        if metric_name not in self.metrics:
            raise ValueError(f"Métrica '{metric_name}' no cargada.")
        series = self.metrics[metric_name]
        if pd.isna(value) or len(series) == 0:
            return np.nan
        if window is not None:
            # Usar las últimas window observaciones (asumiendo que están ordenadas)
            series = series.iloc[-window:]
        return percentileofscore(series, value, kind='mean')
    
    def percentile_str(self, metric_name, value, decimals=0, window=None):
        """Versión formateada para informes, con opción de ventana."""
        pct = self.percentile(metric_name, value, window=window)
        if np.isnan(pct):
            return "N/A"
        return f"{pct:.{decimals}f}%"
    
    def classify(self, percentile_value):
        """
        Clasifica un percentil en categoría semántica según los umbrales.
        """
        if pd.isna(percentile_value):
            return "desconocido"
        if percentile_value >= self.thresholds['extremo_alto']:
            return "extremadamente alto"
        elif percentile_value >= self.thresholds['alto']:
            return "alto"
        elif percentile_value >= self.thresholds['bajo']:
            return "normal"
        elif percentile_value >= self.thresholds['extremo_bajo']:
            return "bajo"
        else:
            return "extremadamente bajo"
    
    def get_narrative(self, metric_name, display_name, value, window=None):
        """
        Genera una frase narrativa completa para una métrica, opcionalmente con ventana.
        Ejemplo: "El score global está en el percentil 82 (alto)"
        """
        pct = self.percentile(metric_name, value, window=window)
        if np.isnan(pct):
            return f"{display_name}: sin datos suficientes para percentil."
        category = self.classify(pct)
        if window:
            return f"{display_name} (últimos {window} días): percentil {pct:.1f} ({category})"
        else:
            return f"{display_name}: percentil {pct:.1f} ({category})"
    
    # Mantener compatibilidad con el método describe original
    def describe(self, metric_name, value):
        """
        (Método legacy) Devuelve una descripción cualitativa basada en el percentil.
        """
        pct = self.percentile(metric_name, value)
        return self.classify(pct)