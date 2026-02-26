from typing import Any, Optional

TIMEFRAME_M1: int
TIMEFRAME_M5: int
TIMEFRAME_M15: int
TIMEFRAME_M30: int
TIMEFRAME_H1: int
TIMEFRAME_H4: int
TIMEFRAME_D1: int
TIMEFRAME_W1: int
TIMEFRAME_MN1: int

POSITION_TYPE_BUY: int
POSITION_TYPE_SELL: int

def initialize() -> bool: ...
def shutdown() -> None: ...
def symbol_select(symbol: str, enable: bool) -> bool: ...
def copy_rates_from_pos(symbol: str, timeframe: int, start_pos: int, count: int) -> Any: ...
def symbol_info_tick(symbol: str) -> Any: ...
def positions_get(symbol: Optional[str] = ...) -> Any: ...
def account_info() -> Any: ...

class _Info:
    balance: float
    equity: float
    profit: float
    margin_free: float

class _Position:
    ticket: int
    symbol: str
    type: int
    volume: float
    price_open: float
    price_current: float
    profit: float
    time: int
