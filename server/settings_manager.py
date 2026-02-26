import json
import os
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

DEFAULT_SETTINGS: Dict[str, Any] = {
    "symbol": "BTCUSDc",
    "timeframe": "M1",
    "bars_count": 100,
    "prediction_count": 5,
    "ollama_endpoint": "http://localhost:11434",
    "ollama_model": "llama3.2",
    "ollama_autostart": True,
    "mt5_autostart": True,
    "mt5_terminal_path": "",
    "ui": {
        "pred_gap": 3,
        "right_padding": 2
    },
    "autotrader": {
        "enabled": False,
        "strategy": "rsi",
        "rsi_period": 4,
        "lot": 0.01,
        "tp_buy": 0,
        "tp_sell": 0,
        "mode": "both",
        "timeframe_analysis": "M1",
        "trailing_stop": 0,
        "trailing_step": 0,
        "profit_limit": 0,
        "loss_limit": 0,
        "max_orders": 1,
        "time_between_orders": 5
    }
}

class SettingsManager:
    def __init__(self, filename: str = "settings.json"):
        self.filename: str = filename
        self.settings: Dict[str, Any] = self.load()

    def get(self, key: str, default: Any = None) -> Any:
        return self.settings.get(key, default)

    def load(self) -> Dict[str, Any]:
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    settings = json.load(f)
                    def merge(d: Dict[str, Any], s: Dict[str, Any]) -> None:
                        for k, v in s.items():
                            if k in d and isinstance(d[k], dict) and isinstance(v, dict):
                                merge(d[k], v)
                            else:
                                d[k] = v
                    merged = DEFAULT_SETTINGS.copy()
                    merge(merged, settings)
                    return merged
            except Exception as e:
                logger.error(f"Error loading settings: {e}")
                return DEFAULT_SETTINGS.copy()
        return DEFAULT_SETTINGS.copy()

    def save(self):
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.settings, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            return False
