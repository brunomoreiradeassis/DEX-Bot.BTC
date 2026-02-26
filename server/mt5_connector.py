import MetaTrader5 as mt5
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

class MT5Connector:
    def __init__(self):
        self._connected = False

    def connect(self) -> bool:
        if not mt5.initialize():
            logger.error(f"MT5 initialize() failed: {mt5.last_error()}")
            self._connected = False
            return False
        self._connected = True
        logger.info("MT5 connected successfully")
        return True

    def disconnect(self):
        mt5.shutdown()
        self._connected = False
        logger.info("MT5 disconnected")

    def is_connected(self) -> bool:
        return self._connected

    def ensure_symbol_selected(self, symbol: str) -> bool:
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Failed to select symbol {symbol}")
            return False
        return True

    def get_tick(self, symbol: str) -> Optional[Dict]:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return {
            'bid': float(tick.bid),
            'ask': float(tick.ask),
            'last': float(tick.last),
            'time': int(tick.time)
        }

    def get_positions(self, symbol: str = None) -> List[Dict]:
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        result = []
        if positions:
            for p in positions:
                result.append({
                    'ticket': p.ticket,
                    'symbol': p.symbol,
                    'type': 'BUY' if p.type == mt5.POSITION_TYPE_BUY else 'SELL',
                    'volume': float(p.volume),
                    'price_open': float(p.price_open),
                    'price_current': float(p.price_current),
                    'profit': float(p.profit),
                    'sl': float(p.sl),
                    'tp': float(p.tp)
                })
        return result