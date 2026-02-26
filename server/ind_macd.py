from typing import List, Dict

def ema_series(data: List[float], p: int) -> List[float]:
    m = 2 / (p + 1)
    ema = [sum(data[:p]) / p]
    for x in data[p:]:
        ema.append((x - ema[-1]) * m + ema[-1])
    return ema

def compute_macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
    if len(closes) < slow + signal:
        return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}
    fast_ema = ema_series(closes, fast)
    slow_ema = ema_series(closes, slow)
    diff = len(fast_ema) - len(slow_ema)
    macd_line = [fast_ema[i + diff] - slow_ema[i] for i in range(len(slow_ema))]
    if len(macd_line) < signal:
        return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}
    signal_line = ema_series(macd_line, signal)
    hist = macd_line[-1] - signal_line[-1]
    return {"macd": round(macd_line[-1], 5), "signal": round(signal_line[-1], 5), "histogram": round(hist, 5)}
