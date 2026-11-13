"""MT5 上 AlphaMaster EA 的「最近 24 小时」只读汇总。

用途：定时任务每个交易日跑一次，把 EA 的实盘（演示账户）跟踪打成一段人能直接读的文本。

三条口径是从实测里钉死的，改动前先重跑探针：
1. MT5 返回的 time 字段是**服务器钟当 UTC** 编码的；日志里的 bar= 则已经是服务器墙钟。
   两者都要换算到本机时刻，时差用实时 tick **现算**（measure_offset），不写死常量 ——
   Dukascopy 随欧美夏令时挪 GMT 偏移，写死的值换季那天会整体错一小时。
2. 账户币种是 CNH，`profit` 是 CNH 而 `price` 是 USD —— 汇总里的钱一律带币种，
   不能让人当美元读。
3. 同一个账户里混着 magic=0 的手工单和 magic=20250101 的 EA 单。已实现盈亏必须
   按 magic 分开结算，且净利含 commission+swap；手工平掉 EA 仓要单独点名，
   否则"样本外跟踪"会被悄悄污染成一份好看的业绩。

本模块**只读**：只用 account_info / positions_get / history_deals_get / terminal_info
和读日志文件，绝不调用任何下单、平仓、改仓接口。
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys

EA_MAGIC = 20250101
TF_MINUTES = 30
SYMBOL = "XAUUSD"
TRADE_MODES = {0: "DEMO", 1: "CONTEST", 2: "REAL"}

_SIG_RE = re.compile(
    r"bar=(\d{4})\.(\d{2})\.(\d{2}) (\d{2}):(\d{2})\s+\S*\s*"
    r"z=([+-]?[\d.]+)\s+tanh=([+-]?[\d.]+)\s+dir=([+-]?\d+)"
)


# ── 纯函数层（可单测，不依赖 MetaTrader5）──────────────────────────
def parse_signal_lines(text: str) -> list[dict]:
    """从 EA 的 Expert 日志文本里抽出信号行；非信号行（警告、别的 EA）丢掉。"""
    out = []
    for m in _SIG_RE.finditer(text):
        y, mo, d, h, mi, z, tanh, dirn = m.groups()
        out.append({
            "bar": dt.datetime(int(y), int(mo), int(d), int(h), int(mi)),
            "z": float(z), "tanh": float(tanh), "dir": int(dirn),
        })
    return out


def measure_offset(now_local: dt.datetime, tick_epoch: float) -> dt.timedelta:
    """本机时刻 − 服务器墙钟。拿实时 tick 现算，别写死 6 小时：
    Dukascopy 跟着欧美夏令时挪 GMT 偏移，写死的常量换季那天会整体错一小时。

    结果取整到分钟：原始差里混着「t1 被抹掉秒」和「tick 是几十秒前的成交价」两笔噪声，
    而真实时差是整分钟量。不取整会让汇总里每笔成交比终端少 1 分钟。"""
    raw = now_local - dt.datetime.utcfromtimestamp(tick_epoch)
    return dt.timedelta(seconds=round(raw.total_seconds() / 60) * 60)


def to_local(epoch: float, offset: dt.timedelta) -> dt.datetime:
    """MT5 的 time/time_msc 字段是「服务器墙钟当 UTC」编码：先 utcfromtimestamp
    还原服务器墙钟，再加实测时差。"""
    return dt.datetime.utcfromtimestamp(epoch) + offset


def to_local_wall(wall: dt.datetime, offset: dt.timedelta) -> dt.datetime:
    """日志里 TimeToString 印出来的已经是服务器墙钟，只需要加时差。"""
    return wall + offset


def summarize_signals(sigs: list[dict], tf_minutes: int = TF_MINUTES) -> dict:
    if not sigs:
        return {"n": 0, "flips": 0, "missing_bars": 0, "expected_bars": 0,
                "dir_counts": {"long": 0, "short": 0, "flat": 0},
                "z_min": None, "z_max": None, "first_bar": None, "last_bar": None,
                "last_dir": None}
    bars = [s["bar"] for s in sigs]
    dirs = [s["dir"] for s in sigs]
    zs = [s["z"] for s in sigs]
    step = dt.timedelta(minutes=tf_minutes)
    expected = int((bars[-1] - bars[0]) / step) + 1
    return {
        "n": len(sigs),
        "dir_counts": {"long": sum(1 for d in dirs if d > 0),
                       "short": sum(1 for d in dirs if d < 0),
                       "flat": sum(1 for d in dirs if d == 0)},
        "flips": sum(1 for a, b in zip(dirs, dirs[1:]) if a != b),
        "z_min": min(zs), "z_max": max(zs),
        "first_bar": bars[0], "last_bar": bars[-1],
        "last_dir": dirs[-1],
        "expected_bars": expected,
        "missing_bars": max(0, expected - len(sigs)),
    }


def _net(deals: list[dict]) -> float:
    return sum(d["profit"] + d["commission"] + d["swap"] for d in deals)


def classify_deals(deals: list[dict], magic: int = EA_MAGIC) -> dict:
    ea = [d for d in deals if d["magic"] == magic]
    manual = [d for d in deals if d["magic"] != magic]
    return {
        "ea": ea, "manual": manual,
        "ea_open_n": sum(1 for d in ea if d["entry"] == 0),
        "ea_close_n": sum(1 for d in ea if d["entry"] == 1),
        "ea_net": _net(ea),
        "manual_net": _net(manual),
        "commission": sum(d["commission"] for d in deals),
        "swap": sum(d["swap"] for d in deals),
    }


def detect_interference(deals: list[dict], magic: int = EA_MAGIC) -> list[str]:
    """找出"钱算在 EA 头上、动作却不是 EA 做的"那几笔。"""
    opener = {d["position_id"]: d["magic"] for d in deals if d["entry"] == 0}
    hits = []
    for d in deals:
        if d["entry"] != 1:
            continue
        is_ea_position = opener.get(d["position_id"]) == magic
        if not is_ea_position:
            continue
        if d["magic"] != magic:
            hits.append(f"仓 #{d['position_id']} 由手工平仓（magic={d['magic']}），"
                        f"盈亏 {d['profit']:+.2f} 不该记进 EA 业绩")
        elif d["comment"].startswith("[tp") or d["comment"].startswith("[sl"):
            hits.append(f"仓 #{d['position_id']} 被手工挂的单子打掉：{d['comment'].strip()}")
    return hits


def open_positions(positions: list[dict], magic: int = EA_MAGIC) -> dict:
    ea = [p for p in positions if p["magic"] == magic]
    net = sum(p["volume"] * (1 if p["type"] == 0 else -1) for p in ea)
    return {
        "n": len(positions), "ea_n": len(ea),
        "side": "long" if net > 0 else "short" if net < 0 else "flat",
        "floating": sum(p["profit"] for p in ea),
        "naked": any(p["sl"] == 0.0 and p["tp"] == 0.0 for p in ea),
        "rows": ea,
    }


# ── 报告成型 ─────────────────────────────────────────────────
def _ts(x) -> str:
    return "—" if x is None else x.strftime("%Y-%m-%d %H:%M")


def _hms(off) -> str:
    """+6:00 / +5:30 / −1:00 —— 只到分钟，够看时差了。"""
    secs = int(off.total_seconds())
    sign = "-" if secs < 0 else "+"
    secs = abs(secs)
    return f"{sign}{secs // 3600}:{secs % 3600 // 60:02d}"


def format_report(ctx: dict) -> str:
    acc, win, sig, deals, pos = (ctx["account"], ctx["window"], ctx["signals"],
                                 ctx["deals"], ctx["positions"])
    cur = acc.get("currency", "")
    span_h = int(round((win["to"] - win["from"]).total_seconds() / 3600))
    lines = [
        f"MT5 EA {span_h} 小时汇总 · {_ts(win['from'])} → {_ts(win['to'])}（本机时刻）",
        f"账户 {acc.get('login')} @ {acc.get('server')} [{acc.get('trade_mode')}] 币种 {cur}",
        "",
        "── 账户 ──",
        f"余额 {acc.get('balance'):,.2f} {cur} / 净值 {acc.get('equity'):,.2f}"
        f" / 已用保证金 {acc.get('margin'):,.2f} / 浮动 {acc.get('floating'):,.2f} {cur}",
    ]

    lines += ["", "── EA 信号（XAUUSD M30）──"]
    if sig["n"] == 0:
        lines.append("窗口内无信号：休市、终端未连接或 EA 未在跑 —— 不要当成「零交易的一天」")
    else:
        dc = sig["dir_counts"]
        lines.append(
            f"信号 {sig['n']} 根（应为 {sig['expected_bars']}，缺 {sig['missing_bars']}）"
            f"，覆盖 {_ts(sig['first_bar'])} → {_ts(sig['last_bar'])}")
        lines.append(f"方向分布 多 {dc['long']} / 空 {dc['short']} / 空仓 {dc['flat']}，"
                     f"翻转 {sig['flips']} 次")
        lines.append(f"z 区间 [{sig['z_min']:+.4f}, {sig['z_max']:+.4f}]")

    lines += ["", "── EA 成交（magic " + str(EA_MAGIC) + "）──"]
    if not deals["ea"]:
        lines.append("窗口内无成交")
    else:
        lines.append(f"开 {deals['ea_open_n']} 笔 / 平 {deals['ea_close_n']} 笔，"
                     f"已实现净 {deals['ea_net']:+,.2f} {cur}"
                     f"（佣金 {deals['commission']:+,.2f}、隔夜费 {deals['swap']:+,.2f}）")
        for d in deals["ea"]:
            act = "开" if d["entry"] == 0 else "平"
            side = "多" if d["type"] == 0 else "空"
            lines.append(f"  {act}{side} {d['volume']} {d['symbol']}"
                         f" @ {d['price']:.3f}  {d['profit']:+,.2f} {cur}"
                         f"  {_ts(d.get('time_local'))}")

    lines += ["", "── 手工单（magic 0）──"]
    if not deals["manual"]:
        lines.append("窗口内无手工成交")
    else:
        lines.append(f"{len(deals['manual'])} 笔，已实现净 {deals['manual_net']:+,.2f} {cur}"
                     f"（不计入 EA 业绩）")

    lines += ["", "── 当前持仓 ──"]
    if pos["ea_n"] == 0:
        lines.append("EA 空仓" + (f"（账户另有 {pos['n']} 笔非 EA 持仓）" if pos["n"] else ""))
    else:
        word = {"long": "多头", "short": "空头"}[pos["side"]]
        lines.append(f"EA {pos['ea_n']} 笔 {word}，浮动 {pos['floating']:+,.2f} {cur}"
                     + ("，无 SL/TP ⇒ 裸仓" if pos["naked"] else ""))
        for p in pos["rows"]:
            lines.append(f"  #{p['ticket']} {p['symbol']} {'多' if p['type'] == 0 else '空'}"
                         f" {p['volume']} @ {p['price_open']:.3f}"
                         f" 现价 {p.get('price_current', 0):.3f}"
                         f" SL {p['sl']:.3f} TP {p['tp']:.3f} 开于 {_ts(p.get('time_open'))}")

    if ctx.get("interference"):
        lines += ["", "── ⚠ 手工干预（业绩归属存疑）──"]
        lines += [f"  {h}" for h in ctx["interference"]]

    h = ctx.get("health", {})
    lines += ["", "── 运行状况 ──"]
    auto = "开" if h.get("trade_allowed", True) else "关（EA 不会下单）"
    off = f" / 时钟差 {_hms(h['clock_offset'])}（实测，用于换算下面所有时刻）" \
        if h.get("clock_offset") is not None else ""
    lines.append(f"终端连接 {'正常' if h.get('terminal_running') else '不可用'}"
                 f" / 自动交易 {auto}"
                 f" / 断线 {h.get('connection_lost', 0)} 次"
                 f" / 日志信号行 {h.get('log_lines', 0)}" + off)
    for w in h.get("warnings", []):
        lines.append(f"  ⚠ {w}")

    lines += ["", "注：演示账户的成交用于检验策略是否成立，不构成可上实盘的依据。"
                  "该公式在离线对抗审计中未通过（二进制无止损基线：回撤 43%、2× 成本转亏）。"]
    return "\n".join(lines)


# ── 采集层（真机；只读）───────────────────────────────────────
def _ea_log_dir(data_folder: str) -> str:
    return os.path.join(data_folder, "MQL5", "Logs")


def _terminal_log_dir(data_folder: str) -> str:
    return os.path.join(data_folder, "logs")


def read_logs(dir_path: str, since: dt.date) -> str:
    """读窗口涉及日期的 UTF-16 日志并拼接；文件按 YYYYMMDD.log 命名。"""
    chunks = []
    d = since
    while d <= dt.date.today():
        p = os.path.join(dir_path, d.strftime("%Y%m%d") + ".log")
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    raw = f.read()
                chunks.append(raw.decode("utf-16-le", errors="ignore") if raw[:2] in (b"\xff\xfe", b"\xfe\xff")
                             or b"\x00" in raw[:24] else raw.decode("utf-8", errors="ignore"))
            except OSError:
                pass
        d += dt.timedelta(days=1)
    return "\n".join(chunks)


def collect(hours: int = 24) -> dict:
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize 失败：{mt5.last_error()}（终端在跑吗？）")
    try:
        acc = mt5.account_info()
        term = mt5.terminal_info()
        if acc is None or term is None:
            raise SystemExit("account_info/terminal_info 返回空：终端未连接服务器")

        now = dt.datetime.now()
        t1 = now.replace(second=0, microsecond=0)
        t0 = t1 - dt.timedelta(hours=hours)
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            raise SystemExit(f"symbol_info_tick({SYMBOL}) 为空：拿不到服务器钟，"
                             f"没法把时间换算到本机")
        # 用未抹秒的 now 配对 tick：用 t1 会让估计少 0~59 秒，取整后整份报告的时刻齐刷刷错一分钟
        offset = measure_offset(now, tick.time)

        deals_raw = mt5.history_deals_get(t0 - dt.timedelta(days=1),
                                         t1 + dt.timedelta(days=1)) or []
        deals = []
        for d in deals_raw:
            local = to_local(d.time, offset)
            if t0 <= local < t1:
                deals.append({"ticket": d.ticket, "magic": d.magic, "entry": d.entry,
                              "type": d.type, "price": d.price, "profit": d.profit,
                              "commission": d.commission, "swap": d.swap,
                              "volume": d.volume, "position_id": d.position_id,
                              "comment": d.comment or "", "symbol": d.symbol,
                              "time_local": local})
        positions = [{"magic": p.magic, "symbol": p.symbol, "type": p.type,
                      "volume": p.volume, "price_open": p.price_open,
                      "price_current": p.price_current, "profit": p.profit,
                      "sl": p.sl, "tp": p.tp, "ticket": p.ticket,
                      "time_open": to_local(p.time, offset)}
                     for p in (mt5.positions_get() or [])]

        text = read_logs(_ea_log_dir(term.data_path), t0.date())
        sigs = []
        for s in parse_signal_lines(text):
            s["bar"] = to_local_wall(s["bar"], offset)
            if t0 <= s["bar"] < t1:
                sigs.append(s)
        conn_text = read_logs(_terminal_log_dir(term.data_path), t0.date())

        mode = TRADE_MODES.get(acc.trade_mode, str(acc.trade_mode))
        expected_bars = max(1, int(hours * 60 / TF_MINUTES))
        warnings = []
        if mode != "DEMO":
            warnings.append(f"账户类型是 {mode}，不是演示 —— 以下数字是真钱")
        if 0 < len(sigs) < expected_bars // 2:
            warnings.append(f"信号只有 {len(sigs)} 根，窗口理论应有约 {expected_bars} 根"
                            f"（休市时段除外）")
        if positions and not any(p["magic"] == EA_MAGIC for p in positions):
            warnings.append(f"账户有 {len(positions)} 笔非 EA 持仓")
        if not term.trade_allowed:
            warnings.append("终端 AutoTrading 处于关闭状态，EA 只算信号不下单")

        return {
            "account": {"login": acc.login, "server": acc.server, "trade_mode": mode,
                        "currency": acc.currency, "balance": acc.balance,
                        "equity": acc.equity, "margin": acc.margin, "floating": acc.profit},
            "window": {"from": t0, "to": t1},
            "signals": summarize_signals(sigs),
            "deals": classify_deals(deals),
            "positions": open_positions(positions),
            "interference": detect_interference(deals),
            "health": {"terminal_running": bool(term.connected),
                       "trade_allowed": bool(term.trade_allowed),
                       "connection_lost": conn_text.count("connection to"),
                       "log_lines": len(sigs), "warnings": warnings,
                       "clock_offset": offset,
                       "last_tick_local": to_local(tick.time, offset),
                       "data_path": term.data_path},
        }
    finally:
        mt5.shutdown()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MT5 AlphaMaster EA 最近 N 小时只读汇总")
    ap.add_argument("--hours", type=int, default=24)
    args = ap.parse_args(argv)
    try:
        sys.stdout.write(format_report(collect(args.hours)) + "\n")
    except SystemExit as e:
        sys.stdout.write(f"汇总不可用：{e}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
