# cycle_engine.py - Clasificador dinámico del ciclo institucional
# Utiliza las nuevas métricas dinámicas (pendientes, aceleración, motores_mejorando)
# para clasificar el mercado en fases reales: Contracción, Capitulación, Acumulación, Expansión, Euforia.

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CycleEngine:
    """
    Clasifica el mercado en fases dinámicas del ciclo institucional.
    Requiere las siguientes columnas en la fila:
        - score_global (o flujo_puro)
        - stress (score_stress)
        - credit (score_bonds)
        - liquidity (score_liquidity)
        - regime (score_regime)
        - leadership (score_leadership)
        - geographic (score_geographic)
        - pend_3d, pend_5d, pend_10d
        - aceleracion
        - motores_mejorando
        - dispersion (opcional)
    """

    def __init__(self, thresholds=None):
        # Umbrales configurables (ajustar según experiencia)
        self.thresholds = thresholds or {
            'score_muy_bajo': -0.4,
            'score_bajo': -0.2,
            'score_alto': 0.2,
            'score_muy_alto': 0.5,
            'pend_positiva': 0.01,
            'pend_fuerte': 0.03,
            'acel_positiva': 0.01,
            'acel_fuerte': 0.03,
            'motores_min': 2,
            'motores_muchos': 4,
            'stress_alto': -0.3,
            'credit_deterioro': -0.05,  # cambio en crédito para detectar deterioro
            'dispersion_alta': 0.6,
        }

    def clasificar(self, fila, fila_anterior=None):
        """
        fila: pd.Series con los valores actuales (debe incluir todas las columnas necesarias).
        fila_anterior: opcional, para detectar cambios (deterioro, mejora).
        Retorna:
            - fase: str (CONTRACCION, CAPITULACION, ACUMULACION, EXPANSION, EUFORIA, NEUTRAL)
            - descripcion: str explicativa
        """
        # Extraer valores con get (por si alguna columna falta)
        score = fila.get('score_global', fila.get('flujo_puro', 0))
        stress = fila.get('stress', 0)
        credit = fila.get('bonds', 0)
        liquidity = fila.get('liquidity', 0)
        regime = fila.get('regime', 0)
        leadership = fila.get('leadership', 0)
        geographic = fila.get('geo', 0)
        pend_3d = fila.get('pend_3d', 0)
        pend_5d = fila.get('pend_5d', 0)
        pend_10d = fila.get('pend_10d', 0)
        aceleracion = fila.get('aceleracion', 0)
        motores = fila.get('motores_mejorando', 0)
        dispersion = fila.get('dispersion', 0)

        # Si tenemos fila anterior, calcular cambios en crédito (para detectar deterioro)
        credit_deterioro = False
        if fila_anterior is not None:
            credit_prev = fila_anterior.get('bonds', 0)
            if credit - credit_prev < self.thresholds['credit_deterioro']:
                credit_deterioro = True

        # --- Reglas de clasificación (ordenadas de más específica a más general) ---

        # 1. EUFORIA (score muy alto, pendiente fuerte, muchos motores mejorando)
        if (score > self.thresholds['score_muy_alto'] and
            pend_5d > self.thresholds['pend_fuerte'] and
            motores >= self.thresholds['motores_muchos']):
            return ("EUFORIA",
                    "Mercado en euforia: score muy alto, pendiente fuerte y amplia participación. Vigilar sobrecalentamiento.")

        # 2. EXPANSION (score positivo, pendiente positiva, aceleración positiva o neutra)
        if (score > self.thresholds['score_bajo'] and
            pend_5d > self.thresholds['pend_positiva'] and
            aceleracion > -self.thresholds['acel_positiva']):  # no negativa
            return ("EXPANSION",
                    "Expansión: score positivo, pendiente positiva y aceleración estable. Crecimiento económico percibido.")

        # 3. ACUMULACION (score aún negativo o bajo, pero pendiente y aceleración positivas)
        if (score < self.thresholds['score_bajo'] and
            pend_5d > self.thresholds['pend_positiva'] and
            aceleracion > self.thresholds['acel_positiva'] and
            motores >= self.thresholds['motores_min']):
            return ("ACUMULACION",
                    "Acumulación institucional: score aún bajo pero pendiente y aceleración positivas. Capital entrando temprano.")

        # 4. CAPITULACION (score muy bajo, pendiente negativa, estrés alto)
        if (score < self.thresholds['score_muy_bajo'] and
            pend_5d < -self.thresholds['pend_positiva'] and
            stress < self.thresholds['stress_alto']):
            return ("CAPITULACION",
                    "Capitulación: score muy bajo, pendiente negativa y estrés alto. Venta masiva, posible suelo cercano.")

        # 5. CONTRACCION (score negativo, pendiente negativa o neutra, sin señales de mejora)
        if (score < self.thresholds['score_bajo'] and
            pend_5d <= self.thresholds['pend_positiva']):
            return ("CONTRACCION",
                    "Contracción: score negativo y pendiente no positiva. Mercado débil, sin señales de giro.")

        # 6. LATE CYCLE (score positivo pero deterioro en crédito o alta dispersión)
        if (score > self.thresholds['score_bajo'] and
            (credit_deterioro or dispersion > self.thresholds['dispersion_alta'])):
            return ("LATE_CYCLE",
                    "Ciclo tardío: score positivo pero señales de deterioro (crédito bajando o alta dispersión). Reducir exposición.")

        # 7. NEUTRAL (cualquier otro caso)
        return ("NEUTRAL",
                "Fase no definida. Mercado sin señales claras o en transición.")