import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, cast
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
import MetaTrader5 as _mt5
import httpx
import os
import subprocess
import shutil
import re
import base64
from contextlib import asynccontextmanager

mt5: Any = _mt5
from .settings_manager import SettingsManager
from .indicators import TechnicalIndicators
from .patterns import PatternDetector
from .ollama_analyzer import OllamaAnalyzer
from .auto_trader import AutoTrader
from .system_state import system_state, state_persister
from .fibonnaci import FibonacciTool

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
for name, level in [
    ("uvicorn", logging.INFO),
    ("uvicorn.error", logging.INFO),
    ("uvicorn.access", logging.INFO),
    ("uvicorn.asgi", logging.INFO),
    ("websockets", logging.WARNING),
    ("websockets.server", logging.WARNING),
    ("websockets.client", logging.WARNING),
]:
    try:
        logging.getLogger(name).setLevel(level)
    except Exception:
        pass

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global auto_trader
    auto_trader.config = settings_manager.get('autotrader', {})
    mt5.initialize()
    t1: asyncio.Task[Any] = asyncio.create_task(connection_monitor())
    t2: asyncio.Task[Any] = asyncio.create_task(data_streamer())
    t3: asyncio.Task[Any] = asyncio.create_task(account_monitor())
    t4: asyncio.Task[Any] = asyncio.create_task(state_persister())
    t5: asyncio.Task[Any] = asyncio.create_task(fib_alert_monitor())
    t6: asyncio.Task[Any] = asyncio.create_task(auto_trader_loop())
    t7: asyncio.Task[Any] = asyncio.create_task(ensure_services())
    tasks: List[asyncio.Task[Any]] = [t1, t2, t3, t4, t5, t6, t7]
    try:
        try:
            system_state.save()
        except Exception:
            pass
        yield
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass
        for t in tasks:
            try:
                t.cancel()
            except Exception:
                pass

app = FastAPI(title="CryptoVision Pro Server", lifespan=lifespan)

# --- Gerenciadores e Estado Global ---
settings_manager = SettingsManager()
indicators = TechnicalIndicators()
pattern_detector = PatternDetector()
ollama_analyzer = OllamaAnalyzer(
    endpoint=cast(str, settings_manager.get('ollama_endpoint', 'http://localhost:11434')),
    model=cast(str, settings_manager.get('ollama_model', 'llama3.2'))
)
auto_trader = AutoTrader(cast(Dict[str, Any], settings_manager.get('autotrader', {})))
fibo_tool = FibonacciTool()

active_connections: List[WebSocket] = []
last_fetch_info: Dict[str, Any] = {'symbol': None, 'timeframe': None, 'rates_count': 0, 'ok': False}
last_streamer_error: Optional[str] = None
last_broadcast_ts: Optional[str] = None

# Configuração atual (Runtime)
current_config: Dict[str, Any] = {
    'symbol': cast(str, settings_manager.get('symbol', 'BTCUSDc')),
    'timeframe': cast(str, settings_manager.get('timeframe', 'M1')),
    'bars_count': cast(int, settings_manager.get('bars_count', 100)),
    'prediction_count': cast(int, settings_manager.get('prediction_count', 5)),
    'fib_tf': 'D1'
}

# Caminhos estáticos (absolutos, baseados neste arquivo)
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "static"))
ANALISE_FILE = os.path.join(BASE_DIR, "analise.json")
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
IMAGES_DIR = os.path.join(PROJECT_DIR, "Imagens")
INDICADORES_FILE = os.path.join(BASE_DIR, "indicadores.json")
PINE_DIR = os.path.join(PROJECT_DIR, "Pine Scripts")
PINE_INDEX = os.path.join(PINE_DIR, "index.json")

def _ensure_pine_dir():
    try:
        os.makedirs(PINE_DIR, exist_ok=True)
    except Exception:
        pass

def _safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9 _.-]", "", name or "Sem titulo").strip()
    if not s:
        s = "script"
    return s[:80]

def _load_pine_index() -> List[Dict[str, Any]]:
    _ensure_pine_dir()
    if os.path.exists(PINE_INDEX):
        try:
            with open(PINE_INDEX, "r", encoding="utf-8") as f:
                data: Any = json.load(f)
                if isinstance(data, list):
                    return cast(List[Dict[str, Any]], data)
        except Exception:
            pass
    items: List[Dict[str, Any]] = []
    try:
        for fn in os.listdir(PINE_DIR):
            if fn.lower().endswith(".pine"):
                p = os.path.join(PINE_DIR, fn)
                try:
                    st = os.stat(p)
                    base = fn[:-5]
                    mid = base.split("_", 1)[0]
                    nm = base.split("_", 1)[1] if "_" in base else base
                    items.append({"id": mid, "name": nm, "filename": fn, "updatedAt": int(st.st_mtime)})
                except Exception:
                    pass
    except Exception:
        pass
    return items

def _save_pine_index(items: List[Dict[str, Any]]) -> None:
    _ensure_pine_dir()
    tmp = PINE_INDEX + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    try:
        os.replace(tmp, PINE_INDEX)
    except Exception:
        pass

def save_analise_snapshot(payload: Dict[str, Any]) -> None:
    try:
        tmp = ANALISE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        try:
            os.replace(tmp, ANALISE_FILE)
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Erro ao salvar analise.json: {e}")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Cliente conectado. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

# --- Funções Auxiliares ---
def get_mt5_timeframe(tf_str: str) -> int:
    mapping = {
        'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1, 'MN': mt5.TIMEFRAME_MN1
    }
    return mapping.get(tf_str, mt5.TIMEFRAME_M1)

async def fetch_real_data(symbol: str, timeframe: str, count: int) -> Optional[List[Dict[str, Any]]]:
    if not mt5.initialize():
        logger.error("MT5 initialization failed in fetch_real_data")
        last_fetch_info.update({'symbol': symbol, 'timeframe': timeframe, 'rates_count': 0, 'ok': False})
        return None
    if not mt5.symbol_select(symbol, True):
        logger.warning(f"Symbol {symbol} not selected or not found")
    tf_const = get_mt5_timeframe(timeframe)
    rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, count)
    if rates is None or len(rates) == 0:
        try:
            candidates: List[str] = []
            base = re.sub(r"[^A-Za-zA-Z]", "", symbol).upper() or "BTCUSD"
            candidates.extend([symbol, base, "BTCUSD", "BTCUSDT", "XBTUSD"])
            sym_get = getattr(mt5, "symbols_get", None)
            if sym_get:
                try:
                    res_any = sym_get(f"{base}*") or sym_get("BTC*")
                    names: List[str] = []
                    if res_any:
                        res_list = cast(List[Any], res_any)
                        for it in res_list:
                            name = getattr(it, "name", None)
                            if name:
                                names.append(f"{name}")
                            elif isinstance(it, (list, tuple)):
                                it_list = cast(Any, it)
                                if len(it_list) > 0:
                                    names.append(str(it_list[0]))
                            elif isinstance(it, str):
                                names.append(it)
                    for nm in names:
                        if nm and nm not in candidates:
                            candidates.append(nm)
                except Exception:
                    pass
            found_symbol = None
            for nm in candidates:
                if not nm:
                    continue
                try:
                    mt5.symbol_select(nm, True)
                    rr = mt5.copy_rates_from_pos(nm, tf_const, 0, count)
                    if rr is not None and len(rr) > 0:
                        symbol = nm
                        rates = rr
                        found_symbol = nm
                        break
                except Exception:
                    continue
            if not rates:
                logger.warning(f"No rates found for {symbol} {timeframe}")
                last_fetch_info.update({'symbol': symbol, 'timeframe': timeframe, 'rates_count': 0, 'ok': False})
                return None
            else:
                last_fetch_info.update({'symbol': found_symbol or symbol, 'timeframe': timeframe, 'rates_count': len(rates), 'ok': True})
        except Exception:
            last_fetch_info.update({'symbol': symbol, 'timeframe': timeframe, 'rates_count': 0, 'ok': False})
            return None
    else:
        last_fetch_info.update({'symbol': symbol, 'timeframe': timeframe, 'rates_count': len(rates), 'ok': True})
    return [
        {'time': int(r['time']), 'open': float(r['open']), 'high': float(r['high']), 
         'low': float(r['low']), 'close': float(r['close']), 
         'volume': int(r['tick_volume']), 'spread': int(r['spread'])} 
        for r in rates
    ]

# --- Tarefas em Segundo Plano ---

async def connection_monitor():
    """Monitora conexões e tenta reconectar se perder."""
    while True:
        initialized = mt5.initialize()
        if not initialized:
            logger.warning("MT5 disconnected. Retrying...")
            await manager.broadcast({'type': 'status', 'mt5': False})
        else:
            await manager.broadcast({'type': 'status', 'mt5': True})
        
        # Ollama Check
        try:
            is_ollama_ok = await ollama_analyzer.check_health()
            await manager.broadcast({'type': 'status', 'ollama': is_ollama_ok})
        except:
            await manager.broadcast({'type': 'status', 'ollama': False})
        
        await asyncio.sleep(5)

async def data_streamer():
    """Envia dados de mercado em tempo real."""
    while True:
        try:
            if len(active_connections) > 0:
                candles = await fetch_real_data(
                    current_config['symbol'], 
                    current_config['timeframe'], 
                    current_config['bars_count']
                )
                
                # Somente envia se tiver dados válidos
                if candles:
                    inds: Dict[str, Any] = indicators.calculate_all(candles)
                    patterns_data: Dict[str, Any] = cast(Dict[str, Any], cast(Any, pattern_detector).detect_all(candles))
                    tick = mt5.symbol_info_tick(current_config['symbol'])
                    
                    # Update System State (para salvamento em sistema.json)
                    system_state.update('market', {
                        'symbol': current_config['symbol'],
                        'price': candles[-1]['close'],
                        'indicators': inds,
                        'patterns': patterns_data['list']
                    })
                    
                    await manager.broadcast({
                        'type': 'market_data',
                        'symbol': current_config['symbol'],
                        'timeframe': current_config['timeframe'],
                        'candles': candles,
                        'indicators': inds,
                        'patterns': patterns_data['list'],
                        'pattern_drawings': patterns_data['drawings'],
                        'current_price': {
                            'bid': float(tick.bid) if tick else candles[-1]['close'],
                            'ask': float(tick.ask) if tick else candles[-1]['close'],
                            'last': float(tick.last) if tick else candles[-1]['close']
                        }
                    })
                    globals()['last_broadcast_ts'] = datetime.now(timezone.utc).isoformat()
        except Exception as e:
            logger.error(f"Data streamer error: {e}")
            globals()['last_streamer_error'] = str(e)
        await asyncio.sleep(1)

async def account_monitor():
    """Monitora conta, ordens e status do AutoTrader."""
    while True:
        try:
            if mt5.initialize():
                info = mt5.account_info()
                if info:
                    await manager.broadcast({
                        'type': 'account_info',
                        'balance': float(info.balance),
                        'equity': float(info.equity),
                        'profit': float(info.profit),
                        'free_margin': float(info.margin_free)
                    })
                
                positions = mt5.positions_get()
                orders: List[Dict[str, Any]] = []
                if positions:
                    for p in positions:
                        orders.append({
                            'ticket': p.ticket, 'symbol': p.symbol,
                            'type': 'BUY' if p.type == mt5.POSITION_TYPE_BUY else 'SELL',
                            'volume': float(p.volume), 'price_open': float(p.price_open),
                            'price_current': float(p.price_current), 'profit': float(p.profit),
                            'time': int(p.time)
                        })
                await manager.broadcast({'type': 'open_orders', 'orders': orders})

                # Envia status do AutoTrader (contador e estado)
                await manager.broadcast({
                    'type': 'autotrader_status',
                    'is_running': auto_trader.is_running,
                    'remaining_time': auto_trader.get_remaining_time()
                })
        except Exception as e:
            logger.error(f"Account monitor error: {e}")
        await asyncio.sleep(2)

# --- Inicialização de Serviços (Ollama e MT5) ---
async def ensure_services():
    # Ollama
    try:
        ok = False
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{cast(str, settings_manager.get('ollama_endpoint', 'http://localhost:11434'))}/api/tags")
                ok = r.status_code == 200
        except:
            ok = False
        if not ok and settings_manager.get('ollama_autostart', True):
            exe = shutil.which("ollama")
            if exe:
                subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # aguarda até subir
            for _ in range(10):
                try:
                    async with httpx.AsyncClient(timeout=2.0) as client:
                        r = await client.get(f"{settings_manager.get('ollama_endpoint')}/api/tags")
                        ok = r.status_code == 200
                        if ok: break
                except: pass
                await asyncio.sleep(1.5)
    except Exception as e:
        logger.error(f"Ollama autostart error: {e}")
    # MT5
    try:
        inited = mt5.initialize()
        if not inited and settings_manager.get('mt5_autostart', True):
            term_path: str = cast(str, settings_manager.get('mt5_terminal_path', "")) or ""
            candidates: List[str] = [
                term_path,
                r"C:\Program Files\MetaTrader 5\terminal64.exe",
                r"C:\Program Files\MetaTrader 5\terminal.exe",
                r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
                r"C:\Program Files (x86)\MetaTrader 5\terminal.exe"
            ]
            for p in candidates:
                if p and os.path.exists(p):
                    try:
                        subprocess.Popen([p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        await asyncio.sleep(3)
                        break
                    except: 
                        pass
            # tenta inicializar repetidamente
            for _ in range(10):
                if mt5.initialize(): break
                await asyncio.sleep(1.5)
    except Exception as e:
        logger.error(f"MT5 autostart error: {e}")

# --- Eventos da Aplicação ---

# Eventos migrados para lifespan (startup/shutdown)

# --- Rotas HTTP ---

@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/@vite/client")
async def vite_client_placeholder():
    return PlainTextResponse("", status_code=204)

@app.get("/favicon.ico")
async def favicon():
    p = os.path.join(STATIC_DIR, "favicon.ico")
    if os.path.exists(p):
        return FileResponse(p, media_type="image/x-icon")
    png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAn8B9f7M8TQAAAAASUVORK5CYII="
    return PlainTextResponse(base64.b64decode(png_b64), media_type="image/png")

@app.get("/api/settings")
async def get_settings():
    return settings_manager.settings

@app.post("/api/settings")
async def update_settings(data: Dict[str, Any]) -> Dict[str, Any]:
    global current_config
    
    # 1. Atualiza configurações gerais do gráfico
    if 'symbol' in data: 
        current_config['symbol'] = data['symbol']
        settings_manager.settings['symbol'] = data['symbol']
    if 'timeframe' in data: 
        current_config['timeframe'] = data['timeframe']
        settings_manager.settings['timeframe'] = data['timeframe']
    if 'bars_count' in data: 
        current_config['bars_count'] = int(data['bars_count'])
        settings_manager.settings['bars_count'] = int(data['bars_count'])
    if 'prediction_count' in data: 
        current_config['prediction_count'] = int(data['prediction_count'])
        settings_manager.settings['prediction_count'] = int(data['prediction_count'])
    
    # 2. Atualiza Ollama
    if 'ollama_endpoint' in data:
        settings_manager.settings['ollama_endpoint'] = data['ollama_endpoint']
        ollama_analyzer.endpoint = data['ollama_endpoint']
    if 'ollama_model' in data:
        settings_manager.settings['ollama_model'] = data['ollama_model']
        ollama_analyzer.model = data['ollama_model']
    
    # 3. Atualiza AutoTrader (Deep Merge)
    if 'autotrader' in data:
        if 'autotrader' not in settings_manager.settings:
            settings_manager.settings['autotrader'] = {}
            
        at_data = data['autotrader']
        at_settings: Dict[str, Any] = cast(Dict[str, Any], settings_manager.settings['autotrader'])
        
        # Lista de chaves esperadas
        valid_keys = [
            'strategy', 'rsi_period', 'lot', 'tp_buy', 'tp_sell', 'mode', 
            'timeframe_analysis', 'trailing_stop', 'trailing_step', 
            'profit_limit', 'loss_limit', 'max_orders', 'time_between_orders'
        ]
        
        for key in valid_keys:
            if key in at_data:
                at_settings[key] = at_data[key]
        
        # Atualiza a instância em execução
        cast(Any, auto_trader).update_config(at_settings)
    
    # 4. Atualiza configurações de UI (gap de previsão e padding direito)
    if 'ui' in data:
        if 'ui' not in settings_manager.settings:
            settings_manager.settings['ui'] = {}
        ui_data = data['ui']
        ui_settings: Dict[str, Any] = cast(Dict[str, Any], settings_manager.settings['ui'])
        for key in ['pred_gap', 'right_padding']:
            if key in ui_data:
                ui_settings[key] = int(ui_data[key])
        
    settings_manager.save()
    return {"status": "saved", "settings": settings_manager.settings}

@app.get("/api/ollama/models")
async def get_ollama_models() -> Dict[str, Any]:
    try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{cast(str, settings_manager.get('ollama_endpoint', 'http://localhost:11434'))}/api/tags")
            if resp.status_code == 200:
                return {"models": [m['name'] for m in resp.json().get('models', [])]}
    except:
        pass
    return {"models": []}

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
try:
    app.mount("/imagens", StaticFiles(directory=IMAGES_DIR), name="imagens")
except Exception:
    pass

@app.get("/api/diagnostics")
async def diagnostics() -> Dict[str, Any]:
    return {
        "active_ws": len(active_connections),
        "current_config": current_config,
        "last_fetch": last_fetch_info,
        "last_broadcast_ts": last_broadcast_ts,
        "last_streamer_error": last_streamer_error
    }

@app.get("/api/pine/list")
async def pine_list() -> Dict[str, Any]:
    try:
        items = _load_pine_index()
        return {"items": items}
    except Exception as e:
        return {"items": [], "error": str(e)}

@app.get("/api/pine/get")
async def pine_get(id: str) -> Dict[str, Any]:
    try:
        items = _load_pine_index()
        it = next((x for x in items if x.get("id") == id), None)
        if not it:
            return {"error": True, "message": "not found"}
        p = os.path.join(PINE_DIR, it["filename"])
        if not os.path.exists(p):
            return {"error": True, "message": "file missing"}
        with open(p, "r", encoding="utf-8") as f:
            code = f.read()
        return {"id": it["id"], "name": it["name"], "code": code, "updatedAt": it.get("updatedAt")}
    except Exception as e:
        return {"error": True, "message": str(e)}

@app.post("/api/pine/save")
async def pine_save(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        _ensure_pine_dir()
        name = payload.get("name") or "Sem titulo"
        code = payload.get("code") or ""
        pid = payload.get("id")
        safe = _safe_name(name).replace(" ", "_")
        if not pid:
            pid = f"{int(datetime.now(timezone.utc).timestamp()*1000)}"
        filename = f"{pid}_{safe}.pine"
        path = os.path.join(PINE_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        items = _load_pine_index()
        now = int(datetime.now(timezone.utc).timestamp())
        existing = next((x for x in items if x.get("id") == pid), None)
        if existing:
            existing["name"] = name
            existing["filename"] = filename
            existing["updatedAt"] = now
        else:
            items.insert(0, {"id": pid, "name": name, "filename": filename, "updatedAt": now})
        _save_pine_index(items)
        return {"ok": True, "id": pid, "name": name, "filename": filename}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/api/pine/delete")
async def pine_delete(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        pid = str(payload.get("id") or "")
        if not pid:
            return {"ok": False, "error": "missing id"}
        items = _load_pine_index()
        it = next((x for x in items if x.get("id") == pid), None)
        if not it:
            return {"ok": False, "error": "not found"}
        path = os.path.join(PINE_DIR, it["filename"])
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        items = [x for x in items if x.get("id") != pid]
        _save_pine_index(items)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _default_ema_config() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "emas": [
            {"period": 9, "color": "#00ff88", "width": 1.4, "visible": True},
            {"period": 21, "color": "#ffa502", "width": 1.4, "visible": True},
            {"period": 100, "color": "#7c3aed", "width": 2.6, "visible": True},
            {"period": 200, "color": "#3b82f6", "width": 2.6, "visible": True}
        ]
    }

@app.get("/api/indicadores")
async def get_indicadores() -> Dict[str, Any]:
    try:
        if os.path.exists(INDICADORES_FILE):
            with open(INDICADORES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                base: Dict[str, Any] = _default_ema_config()
                base.update(cast(Dict[str, Any], data) if isinstance(data, dict) else {})
                return base
    except Exception as e:
        logger.error(f"Erro ao ler indicadores.json: {e}")
    return _default_ema_config()

@app.post("/api/indicadores")
async def post_indicadores(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        new_data: Dict[str, Any] = payload or {}
        # merge with existing file if present
        merged: Dict[str, Any] = {}
        if os.path.exists(INDICADORES_FILE):
            try:
                with open(INDICADORES_FILE, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    if isinstance(existing, dict):
                        merged.update(cast(Dict[str, Any], existing))
            except Exception:
                pass
        merged.update(new_data)
        if "emas" not in merged or not isinstance(merged.get("emas"), list):
            base: Dict[str, Any] = _default_ema_config()
            base.update({k: v for k, v in merged.items() if k != "emas"})
            merged = base
        tmp = INDICADORES_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        try:
            os.replace(tmp, INDICADORES_FILE)
        except Exception:
            pass
        return {"ok": True}
    except Exception as e:
        logger.error(f"Erro ao salvar indicadores.json: {e}")
        return {"ok": False, "error": str(e)}

@app.get("/health")
async def health() -> Dict[str, Any]:
    mt5_ok = mt5.initialize()
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{settings_manager.get('ollama_endpoint')}/api/tags")
            ollama_ok = r.status_code == 200
    except:
        ollama_ok = False
    status_val = "ok" if (ollama_ok or mt5_ok) else "degraded"
    return {"status": status_val, "mt5": mt5_ok, "ollama": ollama_ok}

@app.get("/api/analise")
async def get_analise() -> Dict[str, Any] | Any:
    try:
        if os.path.exists(ANALISE_FILE):
            with open(ANALISE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Erro ao ler analise.json: {e}")
    return {"error": True, "message": "analise.json não encontrado"}

# --- WebSocket ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    active_connections.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                
                # Salva estado do frontend
                if msg.get('type') == 'frontend_state':
                    data = msg.get('data', {})
                    system_state.update('frontend', data)
                    try:
                        ui_settings = settings_manager.settings.get('ui', {})
                        if 'predGap' in data: ui_settings['pred_gap'] = int(data['predGap'])
                        if 'axisPadBars' in data: ui_settings['right_padding'] = int(data['axisPadBars'])
                        settings_manager.settings['ui'] = ui_settings
                        if 'overlays' in data and isinstance(data['overlays'], dict):
                            settings_manager.settings['overlays'] = data['overlays']
                        if 'fib_tf' in data:
                            settings_manager.settings['fib_tf'] = data['fib_tf']
                            current_config['fib_tf'] = data['fib_tf']
                        settings_manager.save()
                    except Exception as e:
                        logger.error(f"frontend_state save error: {e}")

                # Análise IA (Botão Analisar)
                elif msg.get('type') == 'analyze':
                    try:
                        # Limpa analise.json
                        try:
                            save_analise_snapshot({})
                        except Exception:
                            pass
                        await manager.broadcast({'type': 'analysis_progress', 'pct': 5, 'text': 'Coletando dados iniciais...'})
                        # Coleta dados atuais
                        candles = await fetch_real_data(
                            current_config['symbol'], 
                            current_config['timeframe'], 
                            current_config['bars_count']
                        )
                        
                        # Verifica se candles existe
                        if not candles:
                            logger.warning("Analyze cancelled: No candle data")
                            await websocket.send_json({'type': 'error', 'message': 'Não foi possível obter dados do MT5.'})
                            continue

                        system_state.reset()

                        # Coleta dados múltiplos timeframes (Micro, Medio, Macro)
                        tf_list = ['M1', 'M5', 'H1', 'H4', 'D1', 'W1'] 
                        multi_tf_data: Dict[str, Any] = {}
                        
                        for tf in tf_list:
                            try:
                                c = await fetch_real_data(current_config['symbol'], tf, 100)
                                if c:
                                    multi_tf_data[tf] = {
                                        'indicators': indicators.calculate_all(c),
                                        'patterns': cast(Dict[str, Any], cast(Any, pattern_detector).detect_all(c)).get('list', []),
                                        'last_close': c[-1]['close']
                                    }
                            except Exception as e:
                                logger.error(f"Error fetching {tf} data for analysis: {e}")
                        await manager.broadcast({'type': 'analysis_progress', 'pct': 20, 'text': 'Consolidando múltiplos timeframes...'})
                        
                        # 1. Deep Analysis (Tendências M/M/M)
                        deep_analysis: Dict[str, Any] = {"micro_trend": "NEUTRAL", "medium_trend": "NEUTRAL", "macro_trend": "NEUTRAL", "action": "HOLD", "confidence": 0}
                        try:
                            deep_analysis = cast(Dict[str, Any], await cast(Any, ollama_analyzer).deep_analysis(multi_tf_data, current_config['symbol']))
                        except Exception as e:
                            logger.error(f"Deep analysis error: {e}")
                        await manager.broadcast({'type': 'analysis_progress', 'pct': 35, 'text': 'Análise de tendências concluída'})
                        
                        # 1.1 Padrões via LLM (conforme guia do usuário)
                        llm_patterns: Dict[str, Any] = {"patterns": [], "decision": {"action": "HOLD", "confidence": 0, "reason": "Error"}}
                        try:
                            llm_patterns = cast(Dict[str, Any], await cast(Any, ollama_analyzer).detect_patterns_llm(candles))
                        except Exception as e:
                            logger.error(f"LLM patterns error: {e}")
                        await manager.broadcast({'type': 'analysis_progress', 'pct': 50, 'text': 'Padrões IA identificados'})
                        
                        # 2. Previsões por fonte (Indicadores, Padrões, IA, Notícias)
                        inds: Dict[str, Any] = indicators.calculate_all(candles)
                        
                        preds_ind: List[Dict[str, Any]] = []
                        preds_pat: List[Dict[str, Any]] = []
                        preds_ai: List[Dict[str, Any]] = []
                        preds_news: List[Dict[str, Any]] = []
                        
                        try:
                            preds_ind = cast(List[Dict[str, Any]], cast(Any, ollama_analyzer).predict_next_bars_real(candles, inds, current_config['prediction_count']))
                            preds_pat = cast(List[Dict[str, Any]], cast(Any, ollama_analyzer).predict_patterns_based(candles, inds, current_config['prediction_count']))
                            preds_ai = cast(List[Dict[str, Any]], cast(Any, ollama_analyzer).predict_ai_based(candles, inds, deep_analysis, current_config['prediction_count']))
                            preds_news = cast(List[Dict[str, Any]], cast(Any, ollama_analyzer).predict_news_based(candles, current_config['prediction_count']))
                        except Exception as e:
                            logger.error(f"Predictions generation error: {e}")
                        await manager.broadcast({'type': 'analysis_progress', 'pct': 65, 'text': 'Gerando previsões'})
                        
                        # 2.1 Fibonacci detalhada no TF configurado para Fib
                        fib: Dict[str, Any] = {}
                        fib_tf = current_config.get('fib_tf', 'D1')
                        try:
                            fib_candles = await fetch_real_data(current_config['symbol'], fib_tf, 500)
                            fib = fibo_tool.compute(fib_candles or candles)
                        except Exception as e:
                            logger.error(f"Fibonacci error: {e}")
                        await manager.broadcast({'type': 'analysis_progress', 'pct': 75, 'text': 'Calculando Fibonacci'})
                        
                        # 3. Ordens abertas (preço de abertura e P/L)
                        open_orders: List[Dict[str, Any]] = []
                        try:
                            poss = mt5.positions_get()
                            if poss:
                                for p in poss:
                                    open_orders.append({
                                        'ticket': p.ticket,
                                        'symbol': p.symbol,
                                        'type': 'BUY' if p.type == mt5.POSITION_TYPE_BUY else 'SELL',
                                        'volume': float(p.volume),
                                        'price_open': float(p.price_open),
                                        'price_current': float(p.price_current),
                                        'profit': float(p.profit),
                                        'time': int(p.time)
                                    })
                        except Exception as e:
                            logger.error(f"Open orders error: {e}")
                        # 3.1 Info da conta
                        account: Dict[str, Any] = {}
                        try:
                            info = mt5.account_info()
                            if info:
                                account = {
                                    'balance': float(info.balance),
                                    'equity': float(info.equity),
                                    'profit': float(info.profit),
                                    'free_margin': float(info.margin_free)
                                }
                        except Exception:
                            pass
                        await manager.broadcast({'type': 'analysis_progress', 'pct': 80, 'text': 'Coletando dados de conta/ordens'})
                        
                        # 4. Métricas atuais claras
                        ema_cross = 'ALTA' if (inds.get('ema9') or 0) > (inds.get('ema21') or 0) else 'BAIXA'
                        metrics: Dict[str, Any] = {
                            'current_price': float(candles[-1]['close']),
                            'volatility_pct': float(inds.get('atr', {}).get('percentage', 0)),
                            'rsi_value': float(inds.get('rsi', {}).get('value', 0)),
                            'ema_cross': ema_cross
                        }
                        
                        # 5. Lucro/Prejuízo por timeframes da variável Tempo (lista padrão)
                        tf_full = ['M1','M5','M15','M30','H1','H4','D1','W1','MN']
                        tf_profit = {}
                        for tf in tf_full:
                            try:
                                ctf = await fetch_real_data(current_config['symbol'], tf, 120)
                                if ctf and len(ctf) > 1:
                                    start = ctf[0]['close']
                                    end = ctf[-1]['close']
                                    chg = ((end - start) / start) * 100 if start != 0 else 0
                                    tf_profit[tf] = {'start_close': float(start), 'end_close': float(end), 'change_pct': float(round(chg, 4))}
                            except:
                                tf_profit[tf] = {'error': True}
                        await manager.broadcast({'type': 'analysis_progress', 'pct': 85, 'text': 'Consolidando métricas e variações por TF'})
                        
                        # Salva análise no sistema.json
                        patterns_detected: Dict[str, Any] = {"list": [], "drawings": []}
                        try:
                            patterns_detected = cast(Dict[str, Any], cast(Any, pattern_detector).detect_all(candles))
                        except Exception as e:
                            logger.error(f"Pattern detector error: {e}")

                        system_state.update('analysis', {
                            'symbol': current_config['symbol'],
                            'timeframe': current_config['timeframe'],
                            'deep': deep_analysis,
                            'candles': candles[-200:],
                            'indicators': inds,
                            'patterns': patterns_detected.get('list', []),
                            'pattern_drawings': patterns_detected.get('drawings', []),
                            'predictions': {
                                'indicator': preds_ind, 
                                'pattern': preds_pat, 
                                'ai': preds_ai, 
                                'news': preds_news
                            },
                            'patterns_ai': llm_patterns.get('patterns', []),
                            'ai_decision': llm_patterns.get('decision', {}),
                            'multi_tf': multi_tf_data,
                            'fib': {'timeframe': fib_tf, **fib},
                            'metrics': metrics,
                            'open_orders': open_orders,
                            'tf_profit': tf_profit
                        })
                        system_state.save()
                        
                        final_summary: Dict[str, Any] = {"action": "HOLD", "confidence": 0, "reason": "Error"}
                        try:
                            final_summary = cast(Dict[str, Any], await cast(Any, ollama_analyzer).final_analysis_from_file())
                        except Exception as e:
                            logger.error(f"Final summary error: {e}")
                        await manager.broadcast({'type': 'analysis_progress', 'pct': 92, 'text': 'Resumo final da análise'})
                        
                        system_state.update('analysis_summary', final_summary)
                        system_state.save()
                        
                        final_text = "Resumo indisponível"
                        try:
                            final_text = await ollama_analyzer.narrative_summary_from_file()
                        except Exception as e:
                            logger.error(f"Narrative summary error: {e}")

                        system_state.update('analysis_summary_text', final_text)
                        system_state.save()
                        await manager.broadcast({'type': 'analysis_progress', 'pct': 96, 'text': 'Preparando entrega dos resultados'})

                        # 6. Contexto e Plano de ação
                        action_plan: Dict[str, Any] = {}
                        try:
                            action_plan = await ollama_analyzer.action_plan_from_file()
                        except Exception as e:
                            logger.error(f"Action plan error: {e}")

                        snapshot: Dict[str, Any] = {
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'symbol': current_config['symbol'],
                            'timeframe': current_config['timeframe'],
                            'bars_analyzed': int(current_config.get('bars_count', 0)),
                            'account': account,
                            'metrics': metrics,
                            'indicators': inds,
                            'deep': deep_analysis,
                            'ai_decision': llm_patterns.get('decision', {}),
                            'final_analysis': final_summary,
                            'final_text': final_text,
                            'context': action_plan.get('contexto'),
                            'action_plan': action_plan.get('plano'),
                            'patterns': patterns_detected.get('list', []),
                            'patterns_ai': llm_patterns.get('patterns', []),
                            'open_orders': open_orders,
                            'open_orders_count': len(open_orders),
                            'tf_profit': tf_profit,
                            'predictions': {
                                'indicator': preds_ind,
                                'pattern': preds_pat,
                                'ai': preds_ai,
                                'news': preds_news
                            }
                        }
                        save_analise_snapshot(snapshot)
                        await manager.broadcast({'type': 'analysis_progress', 'pct': 100, 'text': 'Análise concluída'})

                        # Envia para o frontend
                        await websocket.send_json({
                            'type': 'predictions_full',
                            'indicator': preds_ind,
                            'pattern': preds_pat,
                            'ai': preds_ai,
                            'news': preds_news,
                            'deep_analysis': deep_analysis,
                            'final_analysis': final_summary,
                            'final_analysis_text': final_text,
                            'patterns_ai': llm_patterns.get('patterns', []),
                            'ai_decision': llm_patterns.get('decision', {})
                        })
                    except Exception as e:
                        logger.error(f"General analysis error: {e}")
                        await websocket.send_json({'type': 'error', 'message': f'Erro durante a análise: {str(e)}'})

                elif msg.get('type') == 'fib_request':
                    tf = msg.get('timeframe') or 'D1'
                    current_config['fib_tf'] = tf
                    fib_candles = await fetch_real_data(current_config['symbol'], tf, 500)
                    if fib_candles:
                        fib = fibo_tool.compute(fib_candles)
                        await websocket.send_json({'type': 'fib_response', 'timeframe': tf, 'fib': fib})

                # Atualização de config vinda do frontend
                elif msg.get('type') == 'config_update':
                    if 'symbol' in msg: current_config['symbol'] = msg['symbol']
                    if 'timeframe' in msg: current_config['timeframe'] = msg['timeframe']
                    if 'bars_count' in msg: current_config['bars_count'] = int(msg['bars_count'])
                    if 'prediction_count' in msg: current_config['prediction_count'] = int(msg['prediction_count'])
                    
                    # Salva no arquivo (reutiliza lógica do POST)
                    await update_settings(msg)
                
                # Notas manuais de padrões (salvar no estado)
                elif msg.get('type') == 'pattern_notes_save':
                    notes = msg.get('data', {})
                    system_state.update('pattern_notes', notes)
                    await websocket.send_json({'type': 'pattern_notes_saved', 'ok': True})

                # Ordens Manuais
                elif msg.get('type') == 'manual_order':
                    res: Dict[str, Any] = cast(Dict[str, Any], cast(Any, auto_trader).execute_order(msg.get('symbol'), msg.get('signal'), "Manual"))
                    await websocket.send_json({'type': 'order_result', 'result': res})
                
                # Controle AutoTrader
                elif msg.get('type') == 'autotrader_start':
                    auto_trader.is_running = True
                    logger.info("AutoTrader Started")
                
                elif msg.get('type') == 'autotrader_stop':
                    auto_trader.is_running = False
                    logger.info("AutoTrader Stopped")

            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        active_connections.remove(websocket)
        
# --- Monitor Fibonacci e Alertas ---
last_fib_alert_ts = {'up': None, 'down': None}

last_fib_alert_ts: Dict[str, Optional[datetime]] = {'up': None, 'down': None}

def build_predictions_to_targets(last_close: float, targets: Dict[str, Optional[float]], count: int = 5) -> Dict[str, List[Dict[str, float]]]:
    preds: Dict[str, List[Dict[str, float]]] = {}
    for key, tgt in targets.items():
        if tgt is None:
            preds[key] = []
            continue
        steps = max(2, count)
        series: List[Dict[str, float]] = []
        cur = last_close
        for i in range(steps):
            next_close = cur + (tgt - last_close) * ((i + 1) / steps)
            o = cur
            c = next_close
            h = max(o, c) * 1.001
            l = min(o, c) * 0.999
            series.append({'open': float(o), 'close': float(c), 'high': float(h), 'low': float(l)})
            cur = next_close
        preds[key] = series
    return preds

async def fib_alert_monitor():
    """Verifica proximidade do preço às linhas de Fibonacci e envia alertas/predições."""
    global last_fib_alert_ts
    while True:
        try:
            tf = current_config.get('fib_tf', 'D1')
            candles = await fetch_real_data(current_config['symbol'], tf, 300)
            if candles and len(candles) > 0:
                fib = fibo_tool.compute(candles)
                last_close = candles[-1]['close']
                levels = [lv['price'] for lv in fib.get('levels', [])]
                up_levels = sorted([p for p in levels if p >= last_close])
                down_levels = sorted([p for p in levels if p <= last_close], reverse=True)
                nearest_up = up_levels[0] if up_levels else None
                nearest_down = down_levels[0] if down_levels else None
                dist_up_pct = abs(nearest_up - last_close) / last_close * 100 if nearest_up else None
                dist_down_pct = abs(last_close - nearest_down) / last_close * 100 if nearest_down else None
                
                # Atualiza sistema.json
                system_state.update('fib', {
                    'timeframe': tf,
                    'last_close': float(last_close),
                    'nearest_up': float(nearest_up) if nearest_up else None,
                    'nearest_down': float(nearest_down) if nearest_down else None,
                    'dist_up_pct': float(dist_up_pct) if dist_up_pct is not None else None,
                    'dist_down_pct': float(dist_down_pct) if dist_down_pct is not None else None
                })
                
                preds = build_predictions_to_targets(last_close, {
                    'up': nearest_up, 'down': nearest_down
                }, current_config.get('prediction_count', 5))
                
                system_state.update('fib_predictions', preds)
                
                # Alerta se <= 10% com rate limit 15 minutos
                now = datetime.now(timezone.utc)
                if nearest_up is not None and dist_up_pct is not None and dist_up_pct <= 10:
                    if not last_fib_alert_ts['up'] or (now - last_fib_alert_ts['up']).total_seconds() >= 15*60:
                        await manager.broadcast({'type': 'fib_alert', 'side': 'UP', 'distance_pct': float(dist_up_pct), 'target': float(nearest_up)})
                        await manager.broadcast({'type': 'fib_predictions', 'side': 'UP', 'predictions': preds['up']})
                        last_fib_alert_ts['up'] = now
                if nearest_down is not None and dist_down_pct is not None and dist_down_pct <= 10:
                    if not last_fib_alert_ts['down'] or (now - last_fib_alert_ts['down']).total_seconds() >= 15*60:
                        await manager.broadcast({'type': 'fib_alert', 'side': 'DOWN', 'distance_pct': float(dist_down_pct), 'target': float(nearest_down)})
                        await manager.broadcast({'type': 'fib_predictions', 'side': 'DOWN', 'predictions': preds['down']})
                        last_fib_alert_ts['down'] = now
        except Exception as e:
            logger.error(f"Fib alert monitor error: {e}")
        await asyncio.sleep(60)

# --- AutoTrader Loop ---
def get_open_positions_count(symbol: str) -> int:
    try:
        ps = mt5.positions_get(symbol=symbol)
        return len(ps) if ps else 0
    except:
        return 0

def decide_signal(candles: List[Dict[str, Any]], inds: Dict[str, Any], config: Dict[str, Any]) -> Optional[str]:
    if not candles or not inds: 
        return None
    last = candles[-1]
    strat = (config.get('strategy') or 'rsi').lower()
    mode = (config.get('mode') or 'both').lower()
    ema9: float = float(inds.get('ema9') or 0)
    ema21: float = float(inds.get('ema21') or 0)
    # RSI recalculado com período do usuário
    try:
        rsi_period = int(config.get('rsi_period', 14))
    except Exception:
        rsi_period = 14
    closes = [c['close'] for c in candles]
    rsi_dict: Dict[str, Any] = cast(Dict[str, Any], indicators.rsi(closes, rsi_period) or {})
    rsi: float = float(rsi_dict.get('value', 50))
    macd_dict: Dict[str, Any] = cast(Dict[str, Any], inds.get('macd') or {})
    macd_hist: float = float(macd_dict.get('histogram', 0) or 0)
    sr: Dict[str, Any] = cast(Dict[str, Any], inds.get('support_resistance') or {})
    atr_dict: Dict[str, Any] = cast(Dict[str, Any], inds.get('atr') or {})
    atr: float = float(atr_dict.get('value', 0) or 0)
    close: float = float(last['close'])
    signal = None
    if strat == 'rsi':
        if rsi < 30: 
            signal = 'BUY'
        elif rsi > 70: 
            signal = 'SELL'
    elif strat == 'scalping':
        if ema9 and ema21:
            if ema9 > ema21 and last['close'] >= last['open']: signal = 'BUY'
            elif ema9 < ema21 and last['close'] < last['open']: signal = 'SELL'
    elif strat == 'day_trade':
        if macd_hist > 0: signal = 'BUY'
        elif macd_hist < 0: signal = 'SELL'
    elif strat == 'swing_trade':
        sup: Optional[float] = cast(Optional[float], sr.get('nearest_support'))
        res: Optional[float] = cast(Optional[float], sr.get('nearest_resistance'))
        if sup is not None and abs(close - sup) <= max(atr, 0.001) * 0.5: signal = 'BUY'
        elif res is not None and abs(res - close) <= max(atr, 0.001) * 0.5: signal = 'SELL'
    elif strat == 'ai':
        tr = inds.get('trend')
        if tr == 'bullish': signal = 'BUY'
        elif tr == 'bearish': signal = 'SELL'
    if signal and mode != 'both':
        if mode == 'buy_only' and signal != 'BUY': signal = None
        if mode == 'sell_only' and signal != 'SELL': signal = None
    return signal

def _rsi_zone(val: float) -> str:
    if val < 30: return 'below'
    if val > 70: return 'above'
    return 'between'

async def auto_trader_loop():
    """Loop que verifica sinais e executa ordens quando AutoTrading estiver ativo."""
    while True:
        try:
            if auto_trader.is_running:
                symbol = current_config['symbol']
                cfg: Dict[str, Any] = cast(Dict[str, Any], getattr(auto_trader, 'config', {}))
                tf: str = cast(str, cfg.get('timeframe_analysis', current_config.get('timeframe', 'M1')))
                candles = await fetch_real_data(symbol, tf, max(200, current_config.get('bars_count', 100)))
                if candles:
                    # RSI conforme período do usuário
                    try:
                        rsi_period = int(cfg.get('rsi_period', 14))
                    except Exception:
                        rsi_period = 14
                    closes = [c['close'] for c in candles]
                    rsi_val = (indicators.rsi(closes, rsi_period) or {}).get('value', 50)
                    zone = _rsi_zone(rsi_val)
                    prev_zone = cast(Any, auto_trader).last_rsi_zone
                    cast(Any, auto_trader).last_rsi_zone = zone
                    signal = None
                    mode: str = str(cfg.get('mode') or 'both').lower()
                    if prev_zone != zone:
                        if zone == 'below' and (mode in ['both','buy_only']):
                            signal = 'BUY'
                        elif zone == 'above' and (mode in ['both','sell_only']):
                            signal = 'SELL'
                        # reset quando volta para between
                        if zone == 'between':
                            cast(Any, auto_trader).last_signal = None
                    # Respeita limites e espaçamento
                    open_cnt = get_open_positions_count(symbol)
                    max_orders = int(cfg.get('max_orders', 1))
                    remaining = auto_trader.get_remaining_time()
                    if signal and open_cnt < max_orders and remaining == 0:
                        res: Dict[str, Any] = cast(Dict[str, Any], cast(Any, auto_trader).execute_order(symbol, signal, f"AutoTrader:{cfg.get('strategy','rsi')}"))
                        await manager.broadcast({'type': 'order_result', 'result': res})
                        # após abrir, envia status para iniciar contagem no frontend
                        await manager.broadcast({
                            'type': 'autotrader_status',
                            'is_running': auto_trader.is_running,
                            'remaining_time': auto_trader.get_remaining_time()
                        })
                        cast(Any, auto_trader).last_signal = signal
                # Atualiza status
                await manager.broadcast({
                    'type': 'autotrader_status',
                    'is_running': auto_trader.is_running,
                    'remaining_time': auto_trader.get_remaining_time()
                })
        except Exception as e:
            logger.error(f"AutoTrader loop error: {e}")
        await asyncio.sleep(5)
