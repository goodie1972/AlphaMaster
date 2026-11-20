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


# ---------- MT5 数据拉取 / CLI ----------

def _rates_to_bars(rates) -> list[dict]:
    return [{"time": int(r["time"]), "open": float(r["open"]), "high": float(r["high"]),
             "low": float(r["low"]), "close": float(r["close"])} for r in rates]


def fetch_data(symbol: str = "XAUUSD", years: float = 2.0) -> dict:
    """分块 copy_rates_range 拉 M5（单次请求超终端 max-bars 会报 Invalid params），
    H4 用 from_pos。入参按本机北京时刻解释：服务器墙钟 +5h（夏）/+6h（冬），
    两端各留 1 天余量，再按 epoch 过滤回服务器区间。"""
    import MetaTrader5 as mt5
    from datetime import timedelta
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize 失败: {mt5.last_error()}")
    try:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"品种 {symbol} 不可用: {mt5.last_error()}")
        tick = mt5.symbol_info_tick(symbol)
        if tick is None or tick.time == 0:
            raise RuntimeError(f"无 tick: {mt5.last_error()}")
        now_srv = int(tick.time)
        start_srv = now_srv - int(years * 365) * 86400
        lo = datetime.utcfromtimestamp(start_srv) + timedelta(hours=5, days=-1)
        hi = datetime.utcfromtimestamp(now_srv) + timedelta(hours=6, days=1)
        seen: dict = {}
        cur = lo
        while cur < hi:
            nxt = min(cur + timedelta(days=85), hi)
            r = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, cur, nxt)
            got = 0 if r is None else len(r)
            if r is not None:
                for row in r:
                    t = int(row["time"])
                    if start_srv <= t and t + 300 <= now_srv:  # 只要已收盘 bar
                        seen[t] = row
            print(f"  chunk {cur:%Y-%m-%d}..{nxt:%Y-%m-%d} -> {got} 根")
            cur = nxt
        h4 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 1, 4000)
        if h4 is None or len(h4) == 0:
            raise RuntimeError(f"H4 拉取为空: {mt5.last_error()}")
    finally:
        mt5.shutdown()
    m5 = [_rates_to_bars([seen[t]])[0] for t in sorted(seen)]
    if not m5:
        raise RuntimeError("M5 合并后为空")
    return {"symbol": symbol, "m5": m5, "h4": _rates_to_bars(h4),
            "fetched_at": int(datetime.now(timezone.utc).timestamp()),
            "requested_start": start_srv, "requested_end": now_srv}


def _srv(ts_: int) -> str:
    return datetime.fromtimestamp(ts_, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _cache_path() -> "pathlib.Path":
    import pathlib
    return pathlib.Path(__file__).resolve().parent.parent / "tmp" / "urf_cache.json"


def _load_cache() -> dict:
    import json
    path = _cache_path()
    if not path.exists():
        raise SystemExit("无数据缓存，先跑: python mt5/backtest_us_range_fade.py fetch")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_pct(v) -> str:
    return "—" if v is None else f"{v * 100:.2f}%"


def _fmt_num(v, nd: int = 2) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="US_Range_Fade 回测")
    ap.add_argument("cmd", choices=["fetch", "backtest"])
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--equity", type=float, default=5000.0)
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--cost-rate", type=float, default=0.0003,
                    help="单边成本率（默认=台账口径 bt_commission+bt_slippage=0.0003）")
    args = ap.parse_args(argv)

    if args.cmd == "fetch":
        data = fetch_data(args.symbol, args.years)
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        m5, h4 = data["m5"], data["h4"]
        refs = [b for b in h4 if datetime.fromtimestamp(b["time"], tz=timezone.utc).hour == REF_HOUR]
        print(f"品种 {data['symbol']} · 缓存 {path}")
        print(f"请求区间（服务器钟）: {_srv(data['requested_start'])} → {_srv(data['requested_end'])}")
        print(f"M5: {len(m5)} 根 · {_srv(m5[0]['time'])} → {_srv(m5[-1]['time'])}")
        print(f"H4: {len(h4)} 根 · {_srv(h4[0]['time'])} → {_srv(h4[-1]['time'])}")
        print(f"16:00 参照棒: {len(refs)} 根")
        return

    data = _load_cache()
    m5, h4 = data["m5"], data["h4"]
    print(f"数据: {data['symbol']} M5 {len(m5)} 根 {_srv(m5[0]['time'])} → {_srv(m5[-1]['time'])}")
    adx_vals = wilder_adx(m5, 14)
    adx_by_time = {b["time"]: v for b, v in zip(m5, adx_vals)}
    variants = [("off", None), ("<20", 20.0), ("<25", 25.0), ("<30", 30.0)]
    report = {"symbol": data["symbol"],
              "data_range": [_srv(m5[0]["time"]), _srv(m5[-1]["time"])],
              "m5_bars": len(m5), "initial_equity": args.equity,
              "risk_per_trade": args.risk, "cost_rate": args.cost_rate,
              "variants": {}}
    print(f"成本率（单边）: {args.cost_rate}")
    print(f"\n{'ADX':<6}{'笔数':>6}{'胜率':>9}{'PF':>8}{'净收益':>10}{'MDD':>9}{'终值$':>11}")
    for name, thr in variants:
        p = Params(adx_max=thr, cost_rate=args.cost_rate)
        res = run_backtest(h4, m5, adx_by_time, p, args.equity, args.risk)
        s = res["summary"]
        reasons = {}
        for t in res["trades"]:
            reasons[t["exit_reason"]] = reasons.get(t["exit_reason"], 0) + 1
        risks = [t["risk_dist"] for t in res["trades"]]
        report["variants"][name] = {
            "summary": {k: v for k, v in s.items() if k != "equity_curve"},
            "by_year": {y: {k: v for k, v in ys.items() if k != "equity_curve"}
                        for y, ys in res["by_year"].items()},
            "exit_reasons": reasons,
            "avg_risk_dist": sum(risks) / len(risks) if risks else None,
            "trades": res["trades"],
        }
        print(f"{name:<6}{s['n_trades']:>6}{_fmt_pct(s['win_rate']):>9}"
              f"{_fmt_num(s['profit_factor']):>8}{_fmt_pct(s['total_return']):>10}"
              f"{_fmt_pct(s['max_drawdown']):>9}{s['final_equity']:>11.2f}")
    print("\n分年（净收益 / 笔数 / 胜率 / MDD）:")
    for name, _ in variants:
        v = report["variants"][name]
        parts = [f"{y}: {_fmt_pct(ys['total_return'])} / {ys['n_trades']}笔 / "
                 f"{_fmt_pct(ys['win_rate'])} / {_fmt_pct(ys['max_drawdown'])}"
                 for y, ys in v["by_year"].items()]
        print(f"  ADX {name:<4} " + " · ".join(parts))
    print("\n出场构成（tp/sl/eod）与平均止损距离（美元/盎司）:")
    for name, _ in variants:
        v = report["variants"][name]
        print(f"  ADX {name:<4} {v['exit_reasons']} · 平均SL距离 {_fmt_num(v['avg_risk_dist'])}")
    out = _cache_path().parent / f"urf_report_cost{args.cost_rate:g}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\n报告: {out}")


if __name__ == "__main__":
    main()