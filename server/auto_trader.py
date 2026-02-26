import MetaTrader5 as mt5
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AutoTrader:
    def __init__(self, config: Dict):
        self.config = config
        self.is_running = False
        self.last_order_time = None

    def update_config(self, config: Dict):
        self.config = config

    def execute_order(self, symbol: str, signal: str, reason: str) -> Dict:
        if not mt5.initialize():
            return {"success": False, "error": "MT5 not initialized"}

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return {"success": False, "error": "Symbol not found"}

        lot = float(self.config.get('lot', 0.01))
        order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if signal == "BUY" else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": reason,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.comment}")
            return {"success": False, "error": result.comment}

        logger.info(f"Order executed: {signal} {symbol}")
        self.last_order_time = datetime.now()
        return {"success": True, "order": result.order}

    def apply_trailing_stop(self, symbol: str):
        # Lógica de Trailing Stop simplificada
        pass

    def get_remaining_time(self) -> int:
        try:
            gap = int(self.config.get('time_between_orders', 0))
        except Exception:
            gap = 0
        if not self.last_order_time or gap <= 0:
            return 0
        elapsed = (datetime.now() - self.last_order_time).total_seconds()
        remaining = max(0, gap - int(elapsed))
        return remaining
