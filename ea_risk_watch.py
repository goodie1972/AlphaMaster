"""EA 持仓只读告警：判定层（纯函数）+ 采集接线 + 状态去重 + CLI。

判定层吃 plain dict / 标量，吐中文告警字符串列表；采集层只经 ea_daily_digest.collect()
只读取数，绝不下单/平仓/改仓。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re

import ea_daily_digest
# EA_MAGIC / open_positions 一并转手给采集层（C2），口径只有 ea_daily_digest 一份
from ea_daily_digest import EA_MAGIC, TF_MINUTES, open_positions, read_logs, _ea_log_dir


def check_should_flat(dir_now: int, pos: dict) -> list[str]:
    # 先验方向再判空仓：非法 dir 必须炸出来，不能被「空仓 ⇒ []」这条路静默当成 0
    if dir_now not in (-1, 0, 1):
        raise ValueError(f"dir_now 只能是 -1/0/1，收到 {dir_now!r}")
    n = pos["ea_n"]
    if n == 0:
        return []
    side = pos["side"]
    if dir_now == 0:
        return [f"信号已归零，EA 仍持 {n} 笔 {side} —— 该平没平"]
    if side == "flat":
        return [f"EA 多空并存净平但仍持 {n} 笔 —— 该清没清"]
    want = "long" if dir_now == 1 else "short"
    if side != want:
        return [f"信号要求反向（dir={dir_now} 要 {want}），EA 实际 {n} 笔 {side}"]
    return []


def check_terminal(pos: dict, health: dict) -> list[str]:
    n = pos["ea_n"]
    if n == 0:
        return []
    out = []
    # 缺字段就 KeyError：体检数据拿不到时不能默认当成健康
    if not health["terminal_running"]:
        out.append(f"终端未连接，EA 的 {n} 笔仓在裸奔 —— 断连期间没人平仓")
    if not health["trade_allowed"]:
        out.append(f"AutoTrading 已关闭，EA 只算信号不会平掉这 {n} 笔仓")
    return out


def check_stop_attached(pos: dict, expected_sl: dict[int, float], tol: float) -> list[str]:
    out = []
    for p in pos["rows"]:
        ticket, sl = p["ticket"], p["sl"]
        if sl == 0:
            out.append(f"仓 #{ticket} SL 未挂（sl=0）—— 没有任何兜底")
            continue
        if ticket not in expected_sl:
            # 核不了 ≠ 核过了：期望值缺失时放行，等于把手工挪票洗白
            out.append(f"仓 #{ticket} 期望 SL 未知，无法核对（实际 sl={sl:.3f}）")
            continue
        exp = expected_sl[ticket]
        if abs(sl - exp) > tol:
            out.append(f"仓 #{ticket} SL 被挪动：实际 {sl:.3f}，期望 {exp:.3f}（容差 {tol}）")
    return out


def check_hold_age(pos: dict, now_local: dt.datetime, max_bars: int,
                   tf_minutes: int = TF_MINUTES) -> list[str]:
    out = []
    for p in pos["rows"]:
        ticket, opened = p["ticket"], p.get("time_open")
        if opened is None:
            out.append(f"仓 #{ticket} 开仓时刻缺失，无法核算持仓时长")
            continue
        age_bars = (now_local - opened).total_seconds() / (tf_minutes * 60)
        if age_bars > max_bars:
            out.append(f"仓 #{ticket} 已持 {age_bars:.1f} 根 bar（上限 {max_bars} 根）—— 持仓过久")
    return out


def check_mixed_hedge(pos: dict) -> list[str]:
    rows = pos["rows"]
    kinds = {p["type"] for p in rows}
    if 0 in kinds and 1 in kinds:
        # EA 的 CurrentDirection() 在 hasBuy&&hasSell 时一律返回 9 并清仓，不看净手数
        return [f"EA 多空并存（{len(rows)} 笔）—— EA 会把它当混合对冲清仓"]
    return []


def _safe(name: str, fn):
    # Q7：单个检查失败转成一条「告警判定失败」，不让它吞掉同轮其它独立检查
    try:
        return fn()
    except (ValueError, KeyError) as exc:
        return [f"告警判定失败：{name}：{exc}"]


def collect_alerts(dir_now, pos, health, expected_sl, now_local, *,
                   max_bars, tol, tf_minutes=TF_MINUTES) -> list[str]:
    out = []
    out += _safe("方向", lambda: check_should_flat(dir_now, pos))
    out += _safe("终端", lambda: check_terminal(pos, health))
    out += _safe("SL", lambda: check_stop_attached(pos, expected_sl, tol))
    out += _safe("持仓时长", lambda: check_hold_age(pos, now_local, max_bars, tf_minutes))
    out += _safe("对冲", lambda: check_mixed_hedge(pos))
    return out


def fingerprint(alerts: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(alerts)).encode("utf-8")).hexdigest()[:16]


def should_emit(prev_fp: str | None, prev_ts: float | None, now_ts: float,
                fp: str, cooldown_s: float) -> bool:
    # 空告警的指纹是个确定值，必须先拦：否则冷却到期会把「一切正常」也播一遍
    if fp == fingerprint([]):
        return False
    if prev_fp is None or prev_ts is None:
        return True
    if fp != prev_fp:
        return True
    return now_ts - prev_ts >= cooldown_s


# ── 采集接线层（C2）──────────────────────────────────────────
_SL_ATTACH_RE = re.compile(r"sl_attach(?:\s+FAILED)?\s+ticket=(\d+)[^\n]*?\bsl=([+-]?\d+(?:\.\d+)?)")


def parse_sl_attach(text: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for m in _SL_ATTACH_RE.finditer(text):
        out[int(m.group(1))] = float(m.group(2))
    return out


def check_signal_stale(last_signal_bar, last_tick_local, max_stale_bars: int,
                       tf_minutes: int = TF_MINUTES) -> list[str]:
    if last_signal_bar is None or last_tick_local is None:
        return ["信号/行情时刻缺失，无法核算停更"]
    # 用「行情时刻 − 信号时刻」而非「现在 − 信号时刻」：周末休市两者都旧、差值仍小 ⇒ 不误报
    diff_bars = (last_tick_local - last_signal_bar).total_seconds() / (tf_minutes * 60)
    if diff_bars < 0:
        return ["信号时刻晚于行情时刻，时钟口径异常"]
    if diff_bars > max_stale_bars:
        return [f"EA 信号停更：行情已走 {diff_bars:.1f} 根 bar 而信号未更新"
                f"（上限 {max_stale_bars} 根）"]
    return []


def alerts_from_ctx(ctx: dict, *, max_bars: int, tol: float, max_stale_bars: int,
                    now_local: dt.datetime, ea_log_text: str = "") -> list[str]:
    sig, health, pos = ctx["signals"], ctx["health"], ctx["positions"]
    dir_now = sig["last_dir"]
    expected_sl = parse_sl_attach(ea_log_text)
    out: list[str] = []
    if dir_now is None:
        out.append("窗口内没有 EA 信号行，无法判定该不该持仓")
        # 信号缺失不代表其余检查也该静默：每个 dir 无关检查各自兜（Q7）
        out += _safe("终端", lambda: check_terminal(pos, health))
        out += _safe("SL", lambda: check_stop_attached(pos, expected_sl, tol))
        out += _safe("持仓时长", lambda: check_hold_age(pos, now_local, max_bars))
        out += _safe("对冲", lambda: check_mixed_hedge(pos))
    else:
        out += collect_alerts(dir_now, pos, health, expected_sl, now_local,
                              max_bars=max_bars, tol=tol)
    # Q8：畸形 ctx 的停更检查也在 try 之外，单独兜住不让 traceback
    out += _safe("停更", lambda: check_signal_stale(sig["last_bar"],
                                                    health.get("last_tick_local"),
                                                    max_stale_bars))
    return out


def load_state(path: str) -> tuple[str | None, float | None]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d["fp"], d["ts"]
    except (OSError, ValueError, KeyError, TypeError):
        return None, None


def save_state(path: str, fp: str, ts: float) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    # 原子写：进程被杀不会留下半截 state.json，下次 load 当成第一次而非崩溃
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"fp": fp, "ts": ts}, f)
    os.replace(tmp, path)


def _print_alerts(ctx: dict, alerts: list[str], now_local: dt.datetime) -> None:
    acc = ctx["account"]
    header = (f"⚠ EA 持仓告警 · {now_local.strftime('%Y-%m-%d %H:%M')} · "
              f"账户 {acc['login']} @ {acc['server']} [{acc['trade_mode']}]")
    lines = [header] + [f"- {a}" for a in alerts]
    lines.append("（只读告警，不会平仓；EA 每根 M30 收盘自行 reconcile）")
    print("\n".join(lines))


def run_watch(*, hours=24, max_bars=192, tol=0.05, max_stale_bars=4,
              cooldown_s=21600.0, state_path="var/ea_risk_watch_state.json",
              dry_run=False) -> int:
    try:
        ctx = ea_daily_digest.collect(hours)
    except (SystemExit, Exception) as exc:
        print(f"告警采集失败：{exc}")
        return 1
    ea_log_text = read_logs(_ea_log_dir(ctx["health"]["data_path"]),
                            ctx["window"]["from"].date())
    now_local = dt.datetime.now()
    alerts = alerts_from_ctx(ctx, max_bars=max_bars, tol=tol,
                             max_stale_bars=max_stale_bars, now_local=now_local,
                             ea_log_text=ea_log_text)
    if dry_run:
        if alerts:
            _print_alerts(ctx, alerts, now_local)
        return 0
    fp = fingerprint(alerts)
    prev_fp, prev_ts = load_state(state_path)
    if not should_emit(prev_fp, prev_ts, now_local.timestamp(), fp, cooldown_s):
        return 0
    _print_alerts(ctx, alerts, now_local)
    save_state(state_path, fp, now_local.timestamp())
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="EA 持仓只读告警（不改 EA，只读取数）")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--max-bars", type=int, default=192)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--max-stale-bars", type=int, default=4)
    ap.add_argument("--cooldown-s", type=float, default=21600.0)
    ap.add_argument("--state", default="var/ea_risk_watch_state.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    return run_watch(hours=args.hours, max_bars=args.max_bars, tol=args.tol,
                     max_stale_bars=args.max_stale_bars, cooldown_s=args.cooldown_s,
                     state_path=args.state, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())