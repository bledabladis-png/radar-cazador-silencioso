# src/alerts/alert.py

class Alert:
    """
    Representa una alerta generada por el sistema.
    """
    def __init__(self, alert_id, name, level, message, score=0.0, drivers=None):
        self.id = alert_id
        self.name = name
        self.level = level  # CRITICAL, HIGH, MEDIUM, LOW
        self.message = message
        self.score = score  # intensidad (0-1)
        self.drivers = drivers or {}  # factores que la provocaron

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'level': self.level,
            'message': self.message,
            'score': self.score,
            'drivers': self.drivers
        }

    def __repr__(self):
        return f"[{self.level}] {self.name}: {self.message} (score: {self.score:.2f})"
