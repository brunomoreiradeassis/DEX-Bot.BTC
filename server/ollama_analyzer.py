import httpx
import logging
import random
from typing import List, Dict, Any, Optional, cast
import json
import os
import asyncio

logger = logging.getLogger(__name__)

class OllamaAnalyzer:
    def __init__(self, endpoint: str, model: str):
        self.endpoint = endpoint
        self.model = model
        self.timeout = 60.0

    async def _call(self, prompt: str, retries: int = 2) -> str:
        for attempt in range(retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    r = await client.post(f"{self.endpoint}/api/generate", json={
                        "model": self.model, 
                        "prompt": prompt, 
                        "stream": False,
                        "options": {
                            "num_predict": 8460,
                            "temperature": 0.7
                        }
                    })
                    if r.status_code == 200:
                        return cast(str, r.json().get('response', ''))
                    else:
                        logger.error(f"Ollama error (status {r.status_code}): {r.text}")
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                logger.error(f"Ollama connection attempt {attempt + 1} failed: {e}")
                if attempt < retries:
                    await asyncio.sleep(3) # Espera 1s antes de tentar novamente
            except Exception as e:
                logger.error(f"Ollama unexpected error: {e}")
                break
        return ""

    async def check_health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{self.endpoint}/api/tags")
                return r.status_code == 200
        except:
            return False
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                candidate = text[start:end]
                return cast(Dict[str, Any], json.loads(candidate))
        except Exception as e:
            logger.error(f"Ollama JSON parse error: {e}")
        return {}

    async def deep_analysis(self, multi_tf_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        prompt = f"""
        Analise {symbol} com base EXCLUSIVA nos dados fornecidos (multi-timeframe com indicadores reais do MT5):
        {json.dumps(multi_tf_data, ensure_ascii=False)}
        
        Regras de saída:
        - Use somente informações presentes nos dados, sem inventar valores.
        - As tendências devem ser ALTA/BAIXA/NEUTRAL (não use UP/DOWN).
        - A ação final deve ser COMPRA/VENDA/MANTER (não use BUY/SELL/HOLD).
        - Inclua detalhes claros dos padrões de velas (ex.: doji, engolfo) e figuras gráficas (ex.: cunha, canal), citando o timeframe quando possível (ex.: H4, D1).
        - Se observar possível formação de OCO/OCOI, mencione se falta confirmação (ex.: "falta mais um ombro para OCOI em D1").
        - Comente sobre EMAs 100/200 (ex.: "cruzando para baixo") e RSI por timeframe (ex.: "RSI sobrevendido em H1").
        
        Responda APENAS JSON válido com este formato:
        {{
          "micro_trend": "ALTA|BAIXA|NEUTRAL",
          "micro_reason": "texto curto",
          "micro_details": ["detalhe 1", "detalhe 2"],
          
          "medium_trend": "ALTA|BAIXA|NEUTRAL",
          "medium_reason": "texto curto",
          "medium_details": ["detalhe 1", "detalhe 2"],
          
          "macro_trend": "ALTA|BAIXA|NEUTRAL",
          "macro_reason": "texto curto",
          "macro_details": ["detalhe 1", "detalhe 2"],
          
          "action": "COMPRA|VENDA|MANTER",
          "confidence": 0-100
        }}
        """
        resp = await self._call(prompt)
        data = self._extract_json(resp)
        if data:
            return {
                "micro_trend": cast(str, data.get("micro_trend", "NEUTRAL")).upper(),
                "micro_reason": cast(str, data.get("micro_reason", "")),
                "micro_details": cast(List[str], data.get("micro_details", [])),
                "medium_trend": cast(str, data.get("medium_trend", "NEUTRAL")).upper(),
                "medium_reason": cast(str, data.get("medium_reason", "")),
                "medium_details": cast(List[str], data.get("medium_details", [])),
                "macro_trend": cast(str, data.get("macro_trend", "NEUTRAL")).upper(),
                "macro_reason": cast(str, data.get("macro_reason", "")),
                "macro_details": cast(List[str], data.get("macro_details", [])),
                "action": cast(str, data.get("action", "MANTER")).upper(),
                "confidence": int(max(0, min(100, cast(int, data.get("confidence", 0)))))
            }
        return {"micro_trend": "NEUTRAL", "medium_trend": "NEUTRAL", "macro_trend": "NEUTRAL", "action": "MANTER", "confidence": 0}

    def _compact_candles(self, candles: List[Dict[str, Any]], max_bars: int = 120) -> List[Dict[str, Any]]:
        data = candles[-max_bars:] if len(candles) > max_bars else candles
        return [
            {"o": round(cast(float, c["open"]), 2), "h": round(cast(float, c["high"]), 2), "l": round(cast(float, c["low"]), 2), "c": round(cast(float, c["close"]), 2), "v": int(cast(int, c.get("volume", c.get("tick_volume", 0))))}
            for c in data
        ]

    async def detect_patterns_llm(self, candles: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not candles:
            return {"patterns": [], "decision": {"action": "HOLD", "confidence": 0, "reason": "No data"}}
        bars = self._compact_candles(candles)
        guidance = """
Você é um assistente de trading especializado em BTC. Identifique padrões com base nas definições:

Padrões de Reversão (Cunhas)
- Cunha de Baixa (Falling Wedge): duas linhas descendentes convergentes, linha superior mais inclinada. Contexto: após tendência de baixa. Sinal: ALTA (reversão) quando rompe a linha superior.
- Cunha de Alta (Rising Wedge): duas linhas ascendentes convergentes, linha inferior mais inclinada. Contexto: após tendência de alta. Sinal: BAIXA (reversão) quando rompe a linha inferior.

Padrões de Continuação (Bandeiras)
- Bandeira de Baixa (Bear Flag): pequeno canal de alta após forte queda. Sinal: CONTINUAÇÃO DE BAIXA no rompimento inferior.
- Bandeira de Alta (Bull Flag): pequeno canal de baixa após forte alta. Sinal: CONTINUAÇÃO DE ALTA no rompimento superior.

Padrões de Velas Específicas
- OCO (One Candle Opposite): vela que fecha completamente oposta à anterior, podendo engolfar ou anular o ganho.
- OCOI (One Candle Opposite Inside): segunda vela contida dentro do range da primeira, sugerindo indecisão.

Dicas de decisão:
- Confirme com volume no rompimento.
- Considere múltiplos timeframes.
- Seja conservador em alta volatilidade do BTC.
- Use stop loss logo abaixo/acima do padrão.

RETORNE APENAS JSON no formato:
{
  "patterns": [
    {"name":"Cunha de Baixa|Cunha de Alta|Bandeira de Baixa|Bandeira de Alta|OCO|OCOI",
     "signal":"ALTA|BAIXA|CONTINUAÇÃO DE ALTA|CONTINUAÇÃO DE BAIXA",
     "type":"bullish|bearish|continuation",
     "confidence": 0-100}
  ],
  "decision": {"action":"BUY|SELL|HOLD","confidence":0-100,"reason":"texto curto"}
}
Se não houver padrões, retorne "patterns":[]. Seja conciso e objetivo.
"""
        prompt = f"Dados (OHLCV compactados, ordem cronológica): {json.dumps(bars)}\n\n{guidance}"
        resp = await self._call(prompt)
        data = self._extract_json(resp)
        if data:
            pats: List[Dict[str, Any]] = cast(List[Dict[str, Any]], data.get("patterns", []))
            for p in pats:
                sig = cast(str, p.get("signal", "")).upper()
                if sig in ["ALTA", "CONTINUAÇÃO DE ALTA"]:
                    p["type"] = cast(str, p.get("type", "")) or ("continuation" if "CONTINUAÇÃO" in sig else "bullish")
                elif sig in ["BAIXA", "CONTINUAÇÃO DE BAIXA"]:
                    p["type"] = cast(str, p.get("type", "")) or ("continuation" if "CONTINUAÇÃO" in sig else "bearish")
                p["confidence"] = int(max(0, min(100, cast(int, p.get("confidence", 0)))))
            
            dec_raw = data.get("decision")
            if not isinstance(dec_raw, dict):
                dec_raw = {}
            dec_raw = cast(Dict[str, Any], dec_raw)

            dec: Dict[str, Any] = {
                "action": cast(str, dec_raw.get("action", "HOLD")),
                "confidence": int(max(0, min(100, cast(int, dec_raw.get("confidence", 0))))),
                "reason": cast(str, dec_raw.get("reason", "Sem sinal claro"))
            }
            return {"patterns": pats, "decision": dec}
        return {"patterns": [], "decision": {"action": "HOLD", "confidence": 0, "reason": "Sem dados"}}
    
    async def final_analysis_from_file(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        try:
            if not filepath:
                base = os.path.dirname(__file__)
                filepath = os.path.join(base, "sistema.json")
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Read state file error: {e}")
            return {"action": "HOLD", "confidence": 0, "reason": "Estado indisponível"}
        
        guidance = """
        Você é um analista de trading. Leia o JSON (estado do sistema) e produza a decisão final com base EXCLUSIVA nos dados:
        - Métricas atuais (preço, volatilidade, RSI, cruzamento de EMAs 100/200 quando disponível)
        - Níveis de Fibonacci, padrões detectados (inclua detalhes de velas e figuras com timeframes)
        - Variação por timeframes (use os % fornecidos)
        - Multi-timeframe e sinais IA anteriores
        
        RETORNE APENAS JSON com:
        {"micro_trend":"ALTA/BAIXA/NEUTRAL","medium_trend":"ALTA/BAIXA/NEUTRAL","macro_trend":"ALTA/BAIXA/NEUTRAL","action":"COMPRA/VENDA/MANTER","confidence":0-100,"reason":"texto curto objetivo"}
        """
        prompt = f"ESTADO:\n{content}\n\n{guidance}"
        resp = await self._call(prompt)
        data = self._extract_json(resp)
        if data:
            try:
                conf_val = int(cast(int, data.get("confidence", 0)))
            except (ValueError, TypeError):
                conf_val = 0
            return {
                "micro_trend": cast(str, data.get("micro_trend", "NEUTRAL")),
                "medium_trend": cast(str, data.get("medium_trend", "NEUTRAL")),
                "macro_trend": cast(str, data.get("macro_trend", "NEUTRAL")),
                "action": cast(str, data.get("action", "HOLD")),
                "confidence": int(max(0, min(100, conf_val))),
                "reason": cast(str, data.get("reason", ""))
            }
        return {"micro_trend": "NEUTRAL", "medium_trend": "NEUTRAL", "macro_trend": "NEUTRAL", "action": "MANTER", "confidence": 0, "reason": "Sem dados"}

    async def narrative_summary_from_file(self, filepath: Optional[str] = None) -> str:
        try:
            if not filepath:
                base = os.path.dirname(__file__)
                filepath = os.path.join(base, "sistema.json")
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Read state file error: {e}")
            return "Resumo indisponível: erro ao ler sistema.json."
        
        guidance = """
        Produza UM TEXTO EM PORTUGUÊS, claro e objetivo, com base EXCLUSIVA no JSON fornecido:
        - Preço atual, volatilidade (%), RSI, cruzamento de EMAs 100/200 quando disponível
        - Níveis de Fibonacci (mais próximo acima/abaixo)
        - Padrões de velas e figuras com confiança e timeframe (ex.: doji em H1, cunha em H4, possível OCOI em D1 faltando ombro)
        - Panorama por timeframe usando as variações % fornecidas
        - Ordens abertas: preço de abertura e P/L agregado
        - Conclusão com RECOMENDAÇÃO DE COMPRA ou RECOMENDAÇÃO DE VENDA, ou MANTER, justificando brevemente
        Responda apenas texto simples.
        """
        prompt = f"ESTADO:\n{content}\n\n{guidance}"
        resp = await self._call(prompt)
        return resp.strip() if resp else "Resumo não disponível."

    async def action_plan_from_file(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        try:
            if not filepath:
                base = os.path.dirname(__file__)
                filepath = os.path.join(base, "sistema.json")
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Read state file error: {e}")
            return {"contexto": "Indisponível", "plano": []}
        guidance = """
        Com base EXCLUSIVA no JSON de estado, produza um CONTEXTO e um PLANO DE AÇÃO objetivo:
        - Contexto: 3-6 bullets curtos com situação atual (RSI, EMAs, Fibonacci, padrões, variação por TF, ordens)
        - Plano: lista de passos práticos (ex.: esperar confirmação X, comprar acima Y, vender abaixo Z, definir SL/TP)
        - Inclua recomendação geral (COMPRA/VENDA/MANTER) coerente com dados
        Responda APENAS JSON:
        {"contexto": ["...","..."], "plano": ["...","..."], "recomendacao": "COMPRA|VENDA|MANTER"}
        """
        prompt = f"ESTADO:\n{content}\n\n{guidance}"
        resp = await self._call(prompt)
        data = self._extract_json(resp)
        if data:
            return {
                "contexto": cast(List[str], data.get("contexto", [])),
                "plano": cast(List[str], data.get("plano", [])),
                "recomendacao": cast(str, data.get("recomendacao", "")).upper()
            }
        return {"contexto": [], "plano": [], "recomendacao": ""}

    def predict_next_bars_real(self, candles: List[Dict[str, Any]], indicators: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
        if not candles or not indicators:
            return []
        preds: List[Dict[str, Any]] = []
        last = candles[-1]
        price = cast(float, last['close'])
        atr_data = cast(Dict[str, Any], indicators.get('atr', {}))
        atr = cast(float, atr_data.get('value', max(price * 0.004, 1e-6)))
        trend = 1 if indicators.get('trend') == 'bullish' else -1
        rsi_data = cast(Dict[str, Any], indicators.get('rsi', {}))
        rsi = cast(float, rsi_data.get('value', 50))
        if rsi > 70: trend = -1
        elif rsi < 30: trend = 1
        phase = 0.0
        for i in range(count):
            phase += 0.9
            wave = atr * 0.8 * (1 if (i%2==0) else -1)
            osc = wave * (0.5 + 0.5)
            drift = trend * atr * (0.6 + i*0.04)
            jitter = (random.random()-0.5) * atr * 0.6
            change = drift + osc + jitter
            o = price
            c = o + change
            h = max(o, c) + abs(change)*0.4 + atr*0.3
            l = min(o, c) - abs(change)*0.4 - atr*0.3
            preds.append({'bar': i+1, 'open': round(o,2), 'high': round(h,2), 'low': round(l,2), 'close': round(c,2), 'confidence': max(50, 78-i*4), 'direction': 'bullish' if c>o else 'bearish'})
            price = c
        return preds

    def predict_patterns_based(self, candles: List[Dict[str, Any]], indicators: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
        if not candles or not indicators: return []
        preds: List[Dict[str, Any]] = []
        price = cast(float, candles[-1]['close'])
        sr = cast(Dict[str, Any], indicators.get('support_resistance', {}))
        resist = cast(float, sr.get('nearest_resistance', price * 1.01))
        support = cast(float, sr.get('nearest_support', price * 0.99))
        direction = 1 if abs(price - support) < abs(price - resist) else -1
        atr_data = cast(Dict[str, Any], indicators.get('atr', {}))
        atr = cast(float, atr_data.get('value', max(price * 0.004, 1e-6)))
        mean = (support + resist) / 2 if support and resist else price
        for i in range(count):
            drift = direction * atr * 0.5
            pull = (mean - price) * 0.15
            osc = ((-1)**i) * atr * 0.6
            jitter = (random.random()-0.5)*atr*0.5
            change = drift + pull + osc + jitter
            o = price
            c = o + change
            h = max(o, c) + abs(change)*0.4 + atr*0.2
            l = min(o, c) - abs(change)*0.4 - atr*0.2
            preds.append({'bar': i+1, 'open': round(o,2), 'high': round(h,2), 'low': round(l,2), 'close': round(c,2), 'confidence': 70, 'direction': 'bullish' if c>o else 'bearish'})
            price = c
        return preds

    def predict_ai_based(self, candles: List[Dict[str, Any]], indicators: Dict[str, Any], analysis: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
        if not candles or not indicators: return []
        preds: List[Dict[str, Any]] = []
        price = cast(float, candles[-1]['close'])
        atr_data = cast(Dict[str, Any], indicators.get('atr', {}))
        atr = cast(float, atr_data.get('value', max(price * 0.004, 1e-6)))
        tr = str(analysis.get('micro_trend') or '').upper()
        direction = 1 if tr in ['UP','ALTA'] else -1 if tr in ['DOWN','BAIXA'] else 0
        phase = 0.0
        for i in range(count):
            phase += 0.8
            drift = direction * atr * 0.5
            osc = ((-1)**i) * atr * 0.5
            jitter = (random.random()-0.5) * atr * 0.4
            change = drift + osc + jitter
            o = price
            c = o + change
            h = max(o, c) + abs(change)*0.35 + atr*0.15
            l = min(o, c) - abs(change)*0.35 - atr*0.15
            preds.append({'bar': i+1, 'open': round(o,2), 'high': round(h,2), 'low': round(l,2), 'close': round(c,2), 'confidence': cast(int, analysis.get('confidence', 50)), 'direction': 'bullish' if c>o else 'bearish'})
            price = c
        return preds

    def predict_news_based(self, candles: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        if not candles: return []
        preds: List[Dict[str, Any]] = []
        price = cast(float, candles[-1]['close'])
        
        for i in range(count):
            change = random.uniform(-50, 50)
            o = price
            c = o + change
            h = max(o, c) + abs(change)*0.5
            l = min(o, c) - abs(change)*0.5
            preds.append({'bar': i+1, 'open': round(o,2), 'high': round(h,2), 'low': round(l,2), 'close': round(c,2), 'confidence': 30, 'direction': 'bullish' if c>o else 'bearish'})
            price = c
        return preds
