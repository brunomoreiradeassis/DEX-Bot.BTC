from typing import List, Dict

class PatternDetector:
    def pivot_points(self, candles: List[Dict], lookback=40, k=2) -> Dict[str, List[Dict]]:
        n = len(candles)
        if n < lookback:
            return {"highs": [], "lows": []}
        seg = candles[-lookback:]
        highs = []
        lows = []
        for i in range(k, len(seg) - k):
            is_high = all(seg[i]['high'] >= seg[i - j]['high'] and seg[i]['high'] >= seg[i + j]['high'] for j in range(1, k + 1))
            is_low = all(seg[i]['low'] <= seg[i - j]['low'] and seg[i]['low'] <= seg[i + j]['low'] for j in range(1, k + 1))
            if is_high:
                highs.append({"idx": n - lookback + i, "price": seg[i]['high']})
            if is_low:
                lows.append({"idx": n - lookback + i, "price": seg[i]['low']})
        return {"highs": highs, "lows": lows}

    def linreg(self, y: List[float]) -> Dict:
        n = len(y)
        if n < 2:
            return {"slope": 0.0, "intercept": y[-1] if n else 0.0}
        sx = (n - 1) * n / 2
        sx2 = (n - 1) * n * (2 * n - 1) / 6
        sy = sum(y)
        sxy = sum(i * v for i, v in enumerate(y))
        denom = n * sx2 - sx * sx
        if denom == 0:
            return {"slope": 0.0, "intercept": y[-1]}
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n
        return {"slope": slope, "intercept": intercept}

    def detect_wedge_flag_pennant(self, candles: List[Dict], lookback=40) -> Dict:
        n = len(candles)
        if n < lookback:
            return {"list": [], "drawings": []}
        seg = candles[-lookback:]
        highs = [c['high'] for c in seg]
        lows = [c['low'] for c in seg]
        lr_h = self.linreg(highs)
        lr_l = self.linreg(lows)
        slope_h = lr_h["slope"]
        slope_l = lr_l["slope"]
        draw = []
        pats = []
        x0 = n - lookback
        y_h0 = lr_h["intercept"]
        y_l0 = lr_l["intercept"]
        y_h1 = y_h0 + slope_h * (lookback - 1)
        y_l1 = y_l0 + slope_l * (lookback - 1)
        converging = (y_h0 - y_l0) > (y_h1 - y_l1) and (y_h0 - y_l0) > 0 and (y_h1 - y_l1) > 0
        parallel = abs(slope_h - slope_l) < max(1e-6, abs(slope_h) * 0.2 + abs(slope_l) * 0.2)
        up = slope_h > 0 and slope_l > 0
        down = slope_h < 0 and slope_l < 0
        if converging and up:
            pats.append({"name": "Cunha de Alta", "type": "bullish", "confidence": 60})
        if converging and down:
            pats.append({"name": "Cunha de Baixa", "type": "bearish", "confidence": 60})
        if parallel and up:
            pats.append({"name": "Bandeira de Alta", "type": "bullish", "confidence": 55})
        if parallel and down:
            pats.append({"name": "Bandeira de Baixa", "type": "bearish", "confidence": 55})
        if converging and abs(slope_h) < abs(slope_l) * 1.5 or converging and abs(slope_l) < abs(slope_h) * 1.5:
            pats.append({"name": "Flâmula", "type": "neutral", "confidence": 50})
        # Desenho ancorado nos topos/fundos (pavios)
        piv = self.pivot_points(candles, lookback=lookback, k=2)
        top_pts = sorted(piv["highs"], key=lambda p: p["idx"], reverse=True)[:2]
        bot_pts = sorted(piv["lows"], key=lambda p: p["idx"], reverse=True)[:2]
        if len(top_pts) >= 2:
            line_top = {"type": "line", "points": [{"x": top_pts[1]["idx"], "y": top_pts[1]["price"]}, {"x": top_pts[0]["idx"], "y": top_pts[0]["price"]}], "color": "rgba(255,165,2,0.6)", "border_color": "#ffa502"}
            draw.append(line_top)
        else:
            draw.append({"type": "line", "points": [{"x": x0, "y": y_h0}, {"x": x0 + lookback - 1, "y": y_h1}], "color": "rgba(255,165,2,0.6)", "border_color": "#ffa502"})
        if len(bot_pts) >= 2:
            line_bottom = {"type": "line", "points": [{"x": bot_pts[1]["idx"], "y": bot_pts[1]["price"]}, {"x": bot_pts[0]["idx"], "y": bot_pts[0]["price"]}], "color": "rgba(255,165,2,0.6)", "border_color": "#ffa502"}
            draw.append(line_bottom)
        else:
            draw.append({"type": "line", "points": [{"x": x0, "y": y_l0}, {"x": x0 + lookback - 1, "y": y_l1}], "color": "rgba(255,165,2,0.6)", "border_color": "#ffa502"})
        return {"list": pats, "drawings": draw}

    def detect_hs(self, candles: List[Dict], lookback=30, tol=0.02) -> Dict:
        n = len(candles)
        if n < lookback:
            return {"list": [], "drawings": []}
        seg = candles[-lookback:]
        highs = [c['high'] for c in seg]
        lows = [c['low'] for c in seg]
        pats = []
        draw = []
        for i in range(5, lookback - 5):
            l = highs[i - 3:i + 4]
            mid = 3
            if len(l) < 7:
                continue
            if l[mid] == max(l) and abs(l[1] - l[5]) / l[mid] < tol:
                idxs = [i - 2, i, i + 2]
                pts = [{"x": n - lookback + idxs[0], "y": seg[idxs[0]]['high']},
                       {"x": n - lookback + idxs[1], "y": seg[idxs[1]]['high']},
                       {"x": n - lookback + idxs[2], "y": seg[idxs[2]]['high']}]
                pats.append({"name": "OCO", "type": "bearish", "confidence": 65})
                draw.append({"type": "triangle", "points": pts + [pts[0]], "color": "rgba(255, 71, 87, 0.15)", "border_color": "#ff4757"})
                break
        for i in range(5, lookback - 5):
            l = lows[i - 3:i + 4]
            mid = 3
            if len(l) < 7:
                continue
            if l[mid] == min(l) and abs(l[1] - l[5]) / (l[mid] if l[mid] != 0 else 1) < tol:
                idxs = [i - 2, i, i + 2]
                pts = [{"x": n - lookback + idxs[0], "y": seg[idxs[0]]['low']},
                       {"x": n - lookback + idxs[1], "y": seg[idxs[1]]['low']},
                       {"x": n - lookback + idxs[2], "y": seg[idxs[2]]['low']}]
                pats.append({"name": "OCOI", "type": "bullish", "confidence": 65})
                draw.append({"type": "triangle", "points": pts + [pts[0]], "color": "rgba(0, 255, 136, 0.15)", "border_color": "#00ff88"})
                break
        return {"list": pats, "drawings": draw}

    def detect_channel_trend(self, candles: List[Dict], lookback=60) -> Dict:
        n = len(candles)
        if n < max(lookback, 20):
            return {"list": [], "drawings": []}
        seg = candles[-lookback:]
        highs = [c['high'] for c in seg]
        lows = [c['low'] for c in seg]
        lr_h = self.linreg(highs)
        lr_l = self.linreg(lows)
        slope_h = lr_h["slope"]
        slope_l = lr_l["slope"]
        x0 = n - lookback
        x1 = n - 1
        y_h0 = lr_h["intercept"]
        y_l0 = lr_l["intercept"]
        y_h1 = y_h0 + slope_h * (lookback - 1)
        y_l1 = y_l0 + slope_l * (lookback - 1)
        pats = []
        draw = []
        parallel = abs(slope_h - slope_l) < max(1e-6, (abs(slope_h) + abs(slope_l)) * 0.15)
        up = slope_h > 0 and slope_l > 0
        down = slope_h < 0 and slope_l < 0
        if parallel and up:
            pats.append({"name": "Canal de Alta", "type": "bullish", "confidence": 55})
            draw.append({"type": "channel", "lines": [[{"x": x0, "y": y_h0}, {"x": x1, "y": y_h1}],
                                                      [{"x": x0, "y": y_l0}, {"x": x1, "y": y_l1}]],
                         "color": "rgba(0,255,136,0.5)", "border_color": "#00ff88"})
        elif parallel and down:
            pats.append({"name": "Canal de Baixa", "type": "bearish", "confidence": 55})
            draw.append({"type": "channel", "lines": [[{"x": x0, "y": y_h0}, {"x": x1, "y": y_h1}],
                                                      [{"x": x0, "y": y_l0}, {"x": x1, "y": y_l1}]],
                         "color": "rgba(255,71,87,0.5)", "border_color": "#ff4757"})
        # Trendlines simples (LTA/LTB) ancoradas em pivôs
        piv = self.pivot_points(candles, lookback=lookback, k=2)
        lows = sorted(piv["lows"], key=lambda p: p["idx"])
        highs = sorted(piv["highs"], key=lambda p: p["idx"])
        if len(lows) >= 2:
            a, b = lows[-2], lows[-1]
            if b["price"] > a["price"]:
                pats.append({"name": "LTA", "type": "bullish", "confidence": 50})
            draw.append({"type": "line", "points": [{"x": a["idx"], "y": a["price"]}, {"x": b["idx"], "y": b["price"]}],
                        "color": "rgba(0,255,136,0.6)", "border_color": "#00ff88"})
        if len(highs) >= 2:
            a, b = highs[-2], highs[-1]
            if b["price"] < a["price"]:
                pats.append({"name": "LTB", "type": "bearish", "confidence": 50})
            draw.append({"type": "line", "points": [{"x": a["idx"], "y": a["price"]}, {"x": b["idx"], "y": b["price"]}],
                        "color": "rgba(255,165,2,0.6)", "border_color": "#ffa502"})
        return {"list": pats, "drawings": draw}

    def detect_all(self, candles: List[Dict]) -> Dict:
        if len(candles) < 20:
            return {"list": [], "drawings": []}
        res1 = self.detect_wedge_flag_pennant(candles, lookback=min(40, len(candles)))
        res2 = self.detect_hs(candles, lookback=min(30, len(candles)))
        res3 = self.detect_channel_trend(candles, lookback=min(60, len(candles)))
        plist = res1["list"] + res2["list"] + res3["list"]
        draws = res1["drawings"] + res2["drawings"] + res3["drawings"]
        return {"list": plist, "drawings": draws}
