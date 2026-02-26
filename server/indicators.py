from typing import List, Dict, Optional, Any
from .ind_rsi import compute_rsi
from .ind_macd import compute_macd
from .ind_bollinger import bollinger_bands

class TechnicalIndicators:
    
    @staticmethod
    def rsi(closes: List[float], period: int = 14) -> Dict[str, float | bool]:
        return compute_rsi(closes, period)

    @staticmethod
    def macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        return compute_macd(closes, fast, slow, signal)

    @staticmethod
    def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Dict[str, float]:
        if len(closes) < period + 1: return {'value': 0.0, 'percentage': 0.0}
        tr_list: List[float] = []
        for i in range(1, len(closes)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            tr_list.append(tr)
        atr_val: float = sum(tr_list[-period:]) / period
        return {'value': round(atr_val, 2), 'percentage': round((atr_val / closes[-1]) * 100, 4)}

    @staticmethod
    def support_resistance(candles: List[Dict[str, Any]], lookback: int = 20) -> Dict[str, float]:
        if len(candles) < lookback: return {'nearest_support': 0, 'nearest_resistance': 0}
        recent = candles[-lookback:]
        return {
            'nearest_support': min([c['low'] for c in recent]),
            'nearest_resistance': max([c['high'] for c in recent])
        }

    @staticmethod
    def volume_profile(candles: List[Dict[str, Any]], rows: int = 20) -> List[Dict[str, Any]]:
        if not candles: return []
        
        prices = [c['low'] for c in candles] + [c['high'] for c in candles]
        min_price = min(prices)
        max_price = max(prices)
        price_step = (max_price - min_price) / rows
        
        if price_step == 0: return []
        
        profile: Dict[int, Dict[str, int]] = {}
        
        for c in candles:
            idx = int((c['close'] - min_price) / price_step)
            idx = max(0, min(rows - 1, idx))
            
            vol = c.get('volume', c.get('tick_volume', 0))
            slot = profile.setdefault(idx, {'volume': 0, 'buy_vol': 0, 'sell_vol': 0})
            slot['volume'] += int(vol)
            
            if c['close'] >= c['open']:
                slot['buy_vol'] += int(vol)
            else:
                slot['sell_vol'] += int(vol)
                
        result: List[Dict[str, Any]] = []
        for i in range(rows):
            price_level: float = min_price + (i * price_step)
            data = profile.get(i, {'volume': 0, 'buy_vol': 0, 'sell_vol': 0})
            if int(data['volume']) > 0:
                result.append({
                    'price': round(price_level, 2),
                    'volume': int(data['volume']),
                    'buy_vol': int(data['buy_vol']),
                    'sell_vol': int(data['sell_vol'])
                })
        return result

    def calculate_all(self, candles: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not candles: return {}
        closes = [c['close'] for c in candles]
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]
        
        out: Dict[str, Any] = {
            'rsi': self.rsi(closes),
            'macd': self.macd(closes),
            'ema9': self.ema(closes, 9) if len(closes) >= 9 else 0,
            'ema21': self.ema(closes, 21) if len(closes) >= 21 else 0,
            'atr': self.atr(highs, lows, closes),
            'support_resistance': self.support_resistance(candles),
            'trend': 'bullish' if (self.ema(closes, 9) or 0) > (self.ema(closes, 21) or 0) else 'bearish',
            'volume_profile': self.volume_profile(candles)
        }
        try:
            bb = bollinger_bands(closes)
            out['bollinger'] = bb
        except Exception:
            out['bollinger'] = {"middle": 0.0, "upper": 0.0, "lower": 0.0}
        return out
    
    @staticmethod
    def ema(data: List[float], period: int) -> Optional[float]:
        if len(data) < period: return None
        multiplier = 2 / (period + 1)
        ema_val = sum(data[:period]) / period
        for price in data[period:]:
            ema_val = (price - ema_val) * multiplier + ema_val
        return ema_val
