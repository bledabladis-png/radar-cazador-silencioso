# src/alerts/alert_engine.py

from .alert import Alert
from .rule_engine import RuleEngine

class AlertEngine:
    """
    Orquesta la generación de alertas: evalúa reglas, deduplica y ordena.
    """
    LEVEL_ORDER = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3, 'INFO': 4}

    def __init__(self, rules_path='config/alert_rules.yaml'):
        self.rule_engine = RuleEngine(rules_path)
        # Grupos de alertas para deduplicación
        self.alert_groups = {
            'ACCUMULATION': ['SMART_MONEY_ACCUMULATION', 'MACRO_INFLECTION', 'EARLY_ACCUMULATION'],
            'DISTRIBUTION': ['INSTITUTIONAL_DISTRIBUTION', 'LATE_CYCLE_DETERIORATION'],
            'RISK': ['SYSTEMIC_RISK', 'VOLATILITY_SHOCK', 'CREDIT_STRESS'],
        }

    def generate(self, snapshot, previous_snapshot=None):
        """
        Genera alertas a partir del snapshot actual (y opcionalmente el anterior).
        Retorna lista de alertas deduplicadas y ordenadas por prioridad.
        """
        # Obtener alertas crudas
        raw_alerts = self.rule_engine.evaluate_all(snapshot)
        
        # Deduplicar por grupos
        grouped_ids = set()
        deduped = []
        for alert in raw_alerts:
            # Verificar si pertenece a algún grupo
            grouped = False
            for group_name, members in self.alert_groups.items():
                if alert.id in members:
                    if group_name not in grouped_ids:
                        grouped_ids.add(group_name)
                        # Crear alerta de grupo (podría tener su propio mensaje)
                        # Por ahora, nos quedamos con la de mayor intensidad del grupo
                        # Esto se podría mejorar
                        deduped.append(alert)
                    grouped = True
                    break
            if not grouped:
                deduped.append(alert)
        
        # Ordenar por nivel e intensidad
        deduped.sort(key=lambda a: (self.LEVEL_ORDER.get(a.level, 99), -a.score))
        return deduped
