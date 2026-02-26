from typing import List, Dict

def compute_rsi(closes: List[float], period: int = 14) -> Dict[str, float | bool]:
    if len(closes) < period + 1:
        return {"value": 50.0, "overbought": False, "oversold": False}
    changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [c if c > 0 else 0 for c in changes]
    losses = [-c if c < 0 else 0 for c in changes]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return {"value": 100.0, "overbought": True, "oversold": False}
    rs = avg_gain / avg_loss
    val = 100 - (100 / (1 + rs))
    return {"value": round(val, 2), "overbought": val > 70, "oversold": val < 30}
