from typing import List, Dict

def bollinger_bands(closes: List[float], period: int = 20, mult: float = 2.0) -> Dict[str, float]:
    if len(closes) < period:
        return {"middle": 0.0, "upper": 0.0, "lower": 0.0}
    window = closes[-period:]
    sma = sum(window) / period
    var = sum((x - sma) ** 2 for x in window) / period
    sd = var ** 0.5
    upper = sma + mult * sd
    lower = sma - mult * sd
    return {"middle": sma, "upper": upper, "lower": lower}
