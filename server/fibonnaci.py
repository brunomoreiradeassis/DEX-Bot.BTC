from typing import List, Dict, Any

class FibonacciTool:
    def compute(self, candles: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not candles:
            return {"high": 0, "low": 0, "levels": [], "midlevels": []}
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        hi = max(highs)
        lo = min(lows)
        if hi == lo:
            return {"high": hi, "low": lo, "levels": [], "midlevels": []}
        percents = [0.0, 23.6, 38.2, 50.0, 61.8, 78.6, 100.0]
        levels = []
        for p in percents:
            price = lo + (hi - lo) * (p / 100.0)
            levels.append({"pct": p, "price": float(price)})
        midlevels = []
        for i in range(len(levels) - 1):
            p1 = levels[i]
            p2 = levels[i + 1]
            mp = (p1["pct"] + p2["pct"]) / 2.0
            mprice = (p1["price"] + p2["price"]) / 2.0
            midlevels.append({"pct": mp, "price": float(mprice)})
        return {"high": float(hi), "low": float(lo), "levels": levels, "midlevels": midlevels}