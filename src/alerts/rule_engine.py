# src/alerts/rule_engine.py

import yaml
import os

class RuleEngine:
    """
    Carga reglas desde un archivo YAML y evalúa condiciones.
    """
    def __init__(self, rules_path='config/alert_rules.yaml'):
        if not os.path.exists(rules_path):
            raise FileNotFoundError(f"No se encuentra el archivo de reglas: {rules_path}")
        with open(rules_path, 'r', encoding='utf-8') as f:
            self.rules = yaml.safe_load(f)

    def evaluate_rule(self, rule_name, rule_data, snapshot):
        """
        Evalúa una regla contra una fila de datos (snapshot).
        Retorna una tupla (cumple, intensidad) donde intensidad es un float 0-1.
        """
        conditions = rule_data.get('conditions', {})
        cumple = True
        drivers = {}
        
        for var, condition in conditions.items():
            if var not in snapshot:
                cumple = False
                break
            val = snapshot[var]
            # Condición puede ser string como "> 0.4" o "improving"
            if isinstance(condition, str):
                if condition == 'improving':
                    # Necesitamos histórico para detectar mejora; por ahora simple
                    # Asumimos que si el valor actual es mayor que el anterior, mejora
                    # Esto requeriría pasar también la fila anterior. Lo simplificamos.
                    # Mejor lo dejamos para después.
                    cumple = False  # Placeholder
                elif condition.startswith('>'):
                    threshold = float(condition[1:].strip())
                    if not (val > threshold):
                        cumple = False
                        break
                elif condition.startswith('<'):
                    threshold = float(condition[1:].strip())
                    if not (val < threshold):
                        cumple = False
                        break
                elif condition.startswith('>='):
                    threshold = float(condition[2:].strip())
                    if not (val >= threshold):
                        cumple = False
                        break
                elif condition.startswith('<='):
                    threshold = float(condition[2:].strip())
                    if not (val <= threshold):
                        cumple = False
                        break
                else:
                    # comparación exacta
                    try:
                        if val != float(condition):
                            cumple = False
                            break
                    except:
                        if val != condition:
                            cumple = False
                            break
            else:
                # Si no es string, asumimos que es un valor a comparar directamente
                if val != condition:
                    cumple = False
                    break
            drivers[var] = val
        
        # Calcular intensidad (si hay umbrales, usamos la distancia)
        intensity = 0.0
        if cumple and 'threshold' in rule_data:
            # Por ejemplo, si la condición es stress > 0.4, la intensidad podría ser (valor - 0.4)/(1-0.4)
            # Esto es mejorable
            threshold = rule_data.get('threshold', 0.0)
            max_val = rule_data.get('max', 1.0)
            # Tomamos la primera variable relevante
            for var in conditions:
                if var in snapshot:
                    val = snapshot[var]
                    if threshold < max_val:
                        intensity = min(1.0, max(0.0, (val - threshold) / (max_val - threshold)))
                    break
        return cumple, intensity, drivers

    def evaluate_all(self, snapshot):
        """
        Evalúa todas las reglas contra el snapshot y devuelve lista de alertas (sin deduplicar).
        """
        from .alert import Alert
        alerts = []
        for name, rule in self.rules.items():
            cumple, intensity, drivers = self.evaluate_rule(name, rule, snapshot)
            if cumple:
                alert = Alert(
                    alert_id=name,
                    name=rule.get('name', name),
                    level=rule.get('level', 'INFO'),
                    message=rule.get('message', ''),
                    score=intensity,
                    drivers=drivers
                )
                alerts.append(alert)
        return alerts
