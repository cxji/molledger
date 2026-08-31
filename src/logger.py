import json
from pathlib import Path


class MetricLogger:
    """Logs metrics to a JSON file."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.history = []  # list of {epoch, metric: value, ...}
        self.summary = {}  # scalar run-level stats (best val, test MAE, etc.)

    def log(self, metrics: dict):
        """Log a dict of metrics for one step/epoch."""
        self.history.append(metrics)
        self._save()

    def set_summary(self, key: str, value: float):
        self.summary[key] = value
        self._save()

    def _save(self):
        with open(self.path, "w") as f:
            json.dump({"history": self.history, "summary": self.summary}, f, indent=2)
