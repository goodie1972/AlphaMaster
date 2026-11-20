"""US_Range_Fade — 美盘 H4 区间 M5 假突破回归策略：纯逻辑回测 + MT5 数据拉取 + CLI。

规则（用户批准的设计，服务器时间口径：bar epoch 即服务器墙钟 UTC+2/+3）：
- 参照棒 = 每日服务器 16:00–20:00 H4；合格 = 实体 >= body_ratio×(H−L) 且
  |实体中点−区间中点| <= center_tol×(H−L)。不合格当日不交易。
- 上轨=H、下轨=L；M5 监控窗 = 服务器 20:00–24:00。
- 空：M5 收盘>上轨 → 标记（跟踪标记以来最高价，含标记棒）；后续收盘<=上轨 →
  该收盘价做空；SL=标记后最高价；TP=入场−rr×(SL−入场)。多：镜像。
- 每轨每日仅首次突破标记一次；开仓后该轨当日作废；ADX 挡掉的回归同样作废该轨
  （保守一次性）；不回归不开仓；最多 1 笔持仓；持仓期间跳过信号逻辑
  （另一轨的极值跟踪也暂停）；双触 SL/TP 同根 bar → 先止损；
  窗口最后一根 M5（23:55）收盘强平未触 SL/TP 的持仓。
- 两轨同根回归 → 空头优先（确定性约定）。入场棒本身不查 SL/TP（收盘成交）。
- ADX 门 = 入场棒 M5 Wilder ADX(14)；None=关；warmup 缺值按挡处理。
- 成本 cost_dist = 2×cost_rate×entry；net_R = (gross_dist−cost_dist)/risk_dist；
  权益复利 eq ×= (1 + risk_per_trade×net_R)。

注意：默认 body_ratio=0.5 时居中判据数学上冗余（|实体中点−区间中点| <=
(区间−实体)/2 <= 区间/4），两判据仍都保留，参数改小时居中判据才独立生效。
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import datetime, timezone

REF_HOUR = 16
_WINDOW_START = 4 * 3600
_WINDOW_END = 8 * 3600


@dataclass
class Params:
    body_ratio: float = 0.5
    center_tol: float = 0.25
    rr: float = 2.0
    adx_max: float | None = None
    cost_rate: float = 0.0003


def ref_eligible(ref: dict, p: Params) -> bool:
    rng = ref["high"] - ref["low"]
    if rng <= 0:
        return False
    body = abs(ref["close"] - ref["open"])
    if body < p.body_ratio * rng:
        return False
    dev = abs((ref["open"] + ref["close"]) / 2 - (ref["high"] + ref["low"]) / 2)
    return dev <= p.center_tol * rng


def _check_exit(pos: dict, b: dict) -> tuple[float | None, str | None]:
    if pos["direction"] == "short":
        hit_sl = b["high"] >= pos["sl"]
        hit_tp = b["low"] <= pos["tp"]
    else:
        hit_sl = b["low"] <= pos["sl"]
        hit_tp = b["high"] >= pos["tp"]
    if hit_sl:
        return pos["sl"], "sl"
    if hit_tp:
        return pos["tp"], "tp"
    return None, None


def _close_trade(pos: dict, exit_time: int, exit_price: float, reason: str, p: Params) -> dict:
    entry = pos["entry_price"]
    gross = entry - exit_price if pos["direction"] == "short" else exit_price - entry
    cost = 2 * p.cost_rate * entry
    return {
        "direction": pos["direction"],
        "mark_time": pos["mark_time"],
        "entry_time": pos["entry_time"],
        "entry_price": entry,
        "sl": pos["sl"],
        "tp": pos["tp"],
        "exit_time": exit_time,
        "exit_price": exit_price,
        "exit_reason": reason,
        "risk_dist": pos["risk_dist"],
        "gross_dist": gross,
        "cost_dist": cost,
        "net_R": (gross - cost) / pos["risk_dist"],
    }


def simulate_day(ref: dict, m5_bars: list[dict], adx_by_time: dict, p: Params) -> list[dict]:
    """单日状态机。m5_bars 可含窗外棒（内部过滤）；adx_by_time: {bar_time: ADX|None}。"""
    if not ref_eligible(ref, p):
        return []
    upper, lower = ref["high"], ref["low"]
    start = ref["time"] + _WINDOW_START
    end = ref["time"] + _WINDOW_END
    bars = sorted((b for b in m5_bars if start <= b["time"] < end), key=lambda b: b["time"])
    if not bars:
        return []
    last_time = bars[-1]["time"]

    marks: dict = {"short": None, "long": None}
    consumed = {"short": False, "long": False}
    pos = None
    trades: list[dict] = []

    for b in bars:
        if pos is not None:
            exit_price, reason = _check_exit(pos, b)
            if exit_price is None and b["time"] == last_time:
                exit_price, reason = b["close"], "eod"
            if exit_price is not None:
                trades.append(_close_trade(pos, b["time"], exit_price, reason, p))
                pos = None
            continue

        entered = False
        for d in ("short", "long"):
            mk = marks[d]
            if mk is None:
                continue
            if d == "short":
                mk["extreme"] = max(mk["extreme"], b["high"])
                returned = b["close"] <= upper
            else:
                mk["extreme"] = min(mk["extreme"], b["low"])
                returned = b["close"] >= lower
            if not returned:
                continue
            marks[d] = None
            adx = adx_by_time.get(b["time"])
            if p.adx_max is not None and (adx is None or adx >= p.adx_max):
                consumed[d] = True
                continue
            entry = b["close"]
            sl = mk["extreme"]
            risk = sl - entry if d == "short" else entry - sl
            if risk <= 0:
                consumed[d] = True
                continue
            tp = entry - p.rr * risk if d == "short" else entry + p.rr * risk
            pos = {
                "direction": d,
                "mark_time": mk["mark_time"],
                "entry_time": b["time"],
                "entry_price": entry,
                "sl": sl,
                "tp": tp,
                "risk_dist": risk,
            }
            consumed[d] = True
            entered = True
            break
        if entered:
            continue

        if not consumed["short"] and marks["short"] is None and b["close"] > upper:
            marks["short"] = {"extreme": b["high"], "mark_time": b["time"]}
        if not consumed["long"] and marks["long"] is None and b["close"] < lower:
            marks["long"] = {"extreme": b["low"], "mark_time": b["time"]}

    return trades


def wilder_adx(bars: list[dict], period: int = 14) -> list[float | None]:
    """Wilder ADX：TR/±DM → RMA 平滑 → DI → DX → RMA(DX)。前 2p−2 根为 None。"""
    n = len(bars)
    out: list[float | None] = [None] * n
    first = 2 * period - 2
    if n <= first:
        return out
    tr = [0.0] * n
    pdm = [0.0] * n
    ndm = [0.0] * n
    tr[0] = bars[0]["high"] - bars[0]["low"]
    for i in range(1, n):
        h, l, pc = bars[i]["high"], bars[i]["low"], bars[i - 1]["close"]
        ph, pl = bars[i - 1]["high"], bars[i - 1]["low"]
        tr[i] = max(h - l, abs(h - pc), abs(l - pc))
        up, dn = h - ph, pl - l
        pdm[i] = up if (up > dn and up > 0) else 0.0
        ndm[i] = dn if (dn > up and dn > 0) else 0.0
    atr = sum(tr[:period])
    ap = sum(pdm[:period])
    an = sum(ndm[:period])
    dx = [0.0] * n
    for i in range(period - 1, n):
        if i > period - 1:
            atr = atr - atr / period + tr[i]
            ap = ap - ap / period + pdm[i]
            an = an - an / period + ndm[i]
        pdi = 100 * ap / atr if atr > 0 else 0.0
        ndi = 100 * an / atr if atr > 0 else 0.0
        s = pdi + ndi
        dx[i] = 100 * abs(pdi - ndi) / s if s > 0 else 0.0
    adx = sum(dx[period - 1:first + 1]) / period
    out[first] = adx
    for i in range(first + 1, n):
        adx = (adx * (period - 1) + dx[i]) / period
        out[i] = adx
    return out


def summarize(trades: list[dict], initial_equity: float = 5000.0,
              risk_per_trade: float = 0.01) -> dict:
    eq = initial_equity
    curve = [eq]
    wins = 0
    win_sum = 0.0
    loss_sum = 0.0
    peak = eq
    mdd = 0.0
    for t in trades:
        pnl = eq * risk_per_trade * t["net_R"]
        eq += pnl
        curve.append(eq)
        if t["net_R"] > 0:
            wins += 1
            win_sum += pnl
        else:
            loss_sum += pnl
        peak = max(peak, eq)
        if peak > 0:
            mdd = max(mdd, (peak - eq) / peak)
    n = len(trades)
    return {
        "n_trades": n,
        "wins": wins,
        "win_rate": wins / n if n else None,
        "profit_factor": win_sum / abs(loss_sum) if loss_sum < 0 else None,
        "final_equity": eq,
        "equity_curve": curve,
        "total_return": eq / initial_equity - 1,
        "max_drawdown": mdd,
    }


def run_backtest(ref_bars: list[dict], m5_bars: list[dict], adx_by_time: dict,
                 p: Params, initial_equity: float = 5000.0,
                 risk_per_trade: float = 0.01) -> dict:
    m5_sorted = sorted(m5_bars, key=lambda b: b["time"])
    times = [b["time"] for b in m5_sorted]
    trades: list[dict] = []
    for ref in sorted(ref_bars, key=lambda b: b["time"]):
        if datetime.fromtimestamp(ref["time"], tz=timezone.utc).hour != REF_HOUR:
            continue
        lo = bisect.bisect_left(times, ref["time"] + _WINDOW_START)
        hi = bisect.bisect_left(times, ref["time"] + _WINDOW_END)
        trades.extend(simulate_day(ref, m5_sorted[lo:hi], adx_by_time, p))
    trades.sort(key=lambda t: t["entry_time"])
    by_year: dict = {}
    for t in trades:
        y = datetime.fromtimestamp(t["entry_time"], tz=timezone.utc).year
        by_year.setdefault(y, []).append(t)
    return {
        "trades": trades,
        "summary": summarize(trades, initial_equity, risk_per_trade),
        "by_year": {y: summarize(ts_, initial_equity, risk_per_trade)
                    for y, ts_ in sorted(by_year.items())},
    }