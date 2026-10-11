# tests/unit/test_derive_trades.py
"""Task C1：`derive_trades` / `trade_stats` 纯函数（brief §1/§4）。

合成行序列：手拼 derive_rows 形状的 dict（不依赖真实行情），逐行口径与
derive_rows 严格一致（行 k 配对行 k+1：相邻且字段可算才非 missing；末行恒 missing）。
覆盖：①单进单单平仓 ②多头两笔+中间空仓 ③反手 ④缺口→broken ⑤末尾→open
⑥同向加/减仓同一笔 ⑦恒等式 Σnet==Σpnl(非missing行)（①②③⑥类，序列首行平仓位）
⑧ trade_stats 空序列全 None/0 结构 ⑨胜率/盈亏比手算对照。
"""
from __future__ import annotations

import math

import pytest

import web.ledger_stats as LS

H = 3600
COST = 0.001


def _rows(closes, positions, gaps=(), tf_s=H):
    """按 derive_rows 同一口径合成行序列。

    gaps: 行号集合——该行与下一行的 bar_ts 差 2×tf_s（数据缺口），使其 missing=True。
    末行恒 missing（无 next，与 derive_rows 一致）。行字段与 derive_rows 产出完全一致：
    {bar_ts, close, position, ret, turnover, pnl, missing}。
    """
    n = len(closes)
    tss, ts = [], 0
    for i in range(n):
        tss.append(ts)
        ts += tf_s * (2 if i in gaps else 1)
    rows = []
    for i in range(n):
        row = {"bar_ts": tss[i], "close": float(closes[i]),
               "position": float(positions[i]), "ret": None, "turnover": None,
               "pnl": None, "missing": True}
        if i + 1 < n and i not in gaps:
            pos = float(positions[i])
            ret = float(closes[i + 1]) / float(closes[i]) - 1.0
            turnover = abs(float(positions[i + 1]) - pos)
            row.update(ret=ret, turnover=turnover,
                       pnl=pos * ret - turnover * COST, missing=False)
        rows.append(row)
    return rows


def _sum_pnl(rows):
    return sum(float(r["pnl"]) for r in rows if not r["missing"])


# ── ① 单进单单平仓 ──────────────────────────────────────────────────
def test_single_entry_single_exit():
    rows = _rows([100, 100, 101, 101], [0.0, 0.5, 0.5, 0.0])
    trades = LS.derive_trades(rows, COST)
    assert len(trades) == 1
    t = trades[0]
    assert t["direction"] == "LONG"
    assert t["status"] == "closed"
    assert t["entry_ts"] == rows[1]["bar_ts"]
    assert t["entry_close"] == pytest.approx(100.0)
    assert t["exit_ts"] == rows[3]["bar_ts"]
    assert t["exit_close"] == pytest.approx(101.0)
    assert t["entry_position"] == pytest.approx(0.5)
    assert t["bars_held"] == 2                       # 行1..行2
    gross = 0.5 * (100.0 / 100.0 - 1.0) + 0.5 * (101.0 / 100.0 - 1.0)
    assert t["gross"] == pytest.approx(gross)
    assert t["cost"] == pytest.approx(0.5 * COST + 0.0 + 0.5 * COST)
    assert t["net"] == pytest.approx(gross - COST)
    # 交易 dict 字段全集（防字段漂移）
    assert set(t) == {"direction", "entry_ts", "entry_close", "exit_ts", "exit_close",
                      "entry_position", "bars_held", "gross", "cost", "net", "status"}


# ── ② 多头两笔 + 中间空仓 ───────────────────────────────────────────
def test_two_long_trades_with_flat_between():
    rows = _rows([100, 100, 101, 101, 102, 102, 103],
                 [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    trades = LS.derive_trades(rows, COST)
    assert len(trades) == 2
    assert [t["bars_held"] for t in trades] == [2, 2]
    assert all(t["status"] == "closed" for t in trades)
    assert trades[0]["exit_ts"] == rows[3]["bar_ts"]
    assert trades[1]["entry_ts"] == rows[4]["bar_ts"]
    assert trades[1]["exit_ts"] == rows[6]["bar_ts"]
    # ⑦ 恒等式
    assert sum(t["net"] for t in trades) == pytest.approx(_sum_pnl(rows))


# ── ③ 反手（LONG→SHORT 相邻）──────────────────────────────────────
def test_reversal_long_to_short():
    rows = _rows([100, 100, 102, 101, 101], [0.0, 1.0, -1.0, 0.0, 0.0])
    trades = LS.derive_trades(rows, COST)
    assert len(trades) == 2
    old, new = trades
    assert old["direction"] == "LONG" and old["status"] == "closed"
    assert old["entry_ts"] == rows[1]["bar_ts"]
    # 出场=状态变化行（反手行），同一行也是新交易进场
    assert old["exit_ts"] == rows[2]["bar_ts"] and old["exit_close"] == pytest.approx(102.0)
    assert new["direction"] == "SHORT" and new["status"] == "closed"
    assert new["entry_ts"] == rows[2]["bar_ts"]
    assert new["exit_ts"] == rows[3]["bar_ts"]
    # 旧交易 gross 含反手行 ret（pos_h×ret_h，与 derive_rows 行口径一致）
    assert old["gross"] == pytest.approx(1.0 * (102.0 / 100.0 - 1.0))
    assert new["gross"] == pytest.approx(-1.0 * (101.0 / 102.0 - 1.0))
    # 反手行成本：旧出场 |1| + 新进场 |−1| = 行 turnover |−1−1|=2
    assert old["cost"] == pytest.approx(1.0 * COST + 1.0 * COST)
    assert new["cost"] == pytest.approx(1.0 * COST + 1.0 * COST)
    # ⑦ 恒等式
    assert sum(t["net"] for t in trades) == pytest.approx(_sum_pnl(rows))


# ── ④ 缺口→broken ─────────────────────────────────────────────────
def test_gap_breaks_trade():
    # 行号 2 与行号 3 之间 bar_ts 不相邻 → 行 2 missing（持仓但断档）
    rows = _rows([100, 100, 101, 102, 102, 103],
                 [0.0, 1.0, 1.0, 1.0, 1.0, 0.0], gaps={2})
    assert rows[2]["missing"] is True and rows[3]["missing"] is False
    trades = LS.derive_trades(rows, COST)
    assert len(trades) == 2
    broken, after = trades
    assert broken["status"] == "broken"
    assert broken["exit_ts"] is None and broken["exit_close"] is None
    # net 只含到最后一根有效持仓 bar（行 2 的 ret 不计）
    assert broken["gross"] == pytest.approx(1.0 * (101.0 / 100.0 - 1.0))
    # 不收出场费：进场 1.0 + 内部 |1−1|=0（行 2 缺失也照常按行字段算）
    assert broken["cost"] == pytest.approx(1.0 * COST)
    assert after["status"] == "closed"
    assert after["entry_ts"] == rows[3]["bar_ts"]


# ── ⑤ 到序列末仍持仓→open，浮盈不含末 bar ─────────────────────────
def test_open_trade_at_sequence_end():
    rows = _rows([100, 100, 101, 102], [0.0, 1.0, 1.0, 1.0])
    assert rows[2]["missing"] is False and rows[3]["missing"] is True  # 末行恒 missing
    trades = LS.derive_trades(rows, COST)
    assert len(trades) == 1
    t = trades[0]
    assert t["status"] == "open"
    assert t["exit_ts"] is None and t["exit_close"] is None
    # gross = 行1+行2 的 pos×ret；末行（行3）ret=None 天然不含 → 「浮盈不含末 bar」
    expect_gross = 1.0 * (101.0 / 100.0 - 1.0) + 1.0 * (102.0 / 101.0 - 1.0)
    assert t["gross"] == pytest.approx(expect_gross)
    # open 不收出场费
    assert t["cost"] == pytest.approx(1.0 * COST)
    assert t["net"] == pytest.approx(expect_gross - COST)
    assert t["bars_held"] == 3


# ── ⑥ 同向加仓/减仓归入同一笔 ─────────────────────────────────────
def test_scale_in_and_out_stay_one_trade():
    # 0 → 0.5 → 1.0（加仓）→ 0.3（减仓）→ 0（平仓）
    rows = _rows([100, 100, 101, 102, 102], [0.0, 0.5, 1.0, 0.3, 0.0])
    trades = LS.derive_trades(rows, COST)
    assert len(trades) == 1
    t = trades[0]
    assert t["status"] == "closed" and t["bars_held"] == 3
    assert t["cost"] == pytest.approx(
        (0.5 + abs(1.0 - 0.5) + abs(0.3 - 1.0) + 0.3) * COST)
    # ⑦ 恒等式
    assert t["net"] == pytest.approx(_sum_pnl(rows))


# ── ⑦ 恒等式（独立重言断言：①②③⑥ 各形态）────────────────────────
def test_identity_sum_net_equals_sum_pnl():
    cases = [
        _rows([100, 100, 101, 101], [0.0, 0.5, 0.5, 0.0]),                       # ①
        _rows([100, 100, 101, 101, 102, 102, 103], [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]),  # ②
        _rows([100, 100, 102, 101, 101], [0.0, 1.0, -1.0, 0.0, 0.0]),            # ③
        _rows([100, 100, 101, 102, 102], [0.0, 0.5, 1.0, 0.3, 0.0]),             # ⑥加仓
        _rows([100, 100, 101, 100.5, 100.5, 100.5, 101],                         # 减仓+空仓+SHORT 再进
              [0.0, 1.0, 0.4, 0.0, -0.6, -0.6, 0.0]),
    ]
    for i, rows in enumerate(cases):
        trades = LS.derive_trades(rows, COST)
        assert sum(t["net"] for t in trades) == pytest.approx(_sum_pnl(rows)), f"case {i}"
        # 非 missing 行 pnl 总和的绝对值应可分解（sanity：无 NaN 泄漏）
        assert all(math.isfinite(t["net"]) for t in trades), f"case {i}"


# ── 空序列 / 全平仓序列不抛 ────────────────────────────────────────
def test_no_holdings_yields_no_trades():
    rows = _rows([100, 101, 102], [0.0, 0.0, 0.0])
    assert LS.derive_trades(rows, COST) == []
    assert LS.derive_trades([], COST) == []


# ── ⑧ trade_stats 空序列：全 None/0 结构，绝不抛 ──────────────────
def test_trade_stats_empty():
    s = LS.trade_stats([])
    assert s["trades_total"] == 0
    assert s["trades_closed"] == 0
    assert s["open_count"] == 0
    assert s["win_rate"] is None
    assert s["profit_factor"] is None
    assert s["avg_net"] is None
    assert s["cum_net"] is None
    assert s["max_drawdown"] == pytest.approx(0.0)


# ── ⑨ 胜率/盈亏比手算对照 ─────────────────────────────────────────
def test_trade_stats_numbers():
    t_closed = [
        {"status": "closed", "net": 0.02},
        {"status": "closed", "net": -0.01},
        {"status": "closed", "net": 0.03},
        {"status": "open", "net": 0.05},
        {"status": "broken", "net": -0.04},
    ]
    s = LS.trade_stats(t_closed)
    assert s["trades_total"] == 5
    assert s["trades_closed"] == 3
    assert s["open_count"] == 1
    assert s["win_rate"] == pytest.approx(2 / 3)                 # closed 中 net>0 占比
    # profit_factor = Σ盈/|Σ亏|（closed 口径：(0.02+0.03)/0.01）
    assert s["profit_factor"] == pytest.approx(5.0)
    assert s["avg_net"] == pytest.approx((0.02 - 0.01 + 0.03) / 3)
    assert s["cum_net"] == pytest.approx(0.02 - 0.01 + 0.03 + 0.05 - 0.04)
    # max_drawdown：按交易序列 cum 峰谷差（复用 _max_drawdown）
    expect = LS._max_drawdown([0.02, -0.01, 0.03, 0.05, -0.04])
    assert s["max_drawdown"] == pytest.approx(expect)


def test_trade_stats_no_closed_no_losses():
    # 无 closed → win_rate None；closed 无亏损 → profit_factor None
    s = LS.trade_stats([{"status": "open", "net": 0.1}])
    assert s["win_rate"] is None and s["profit_factor"] is None
    s2 = LS.trade_stats([{"status": "closed", "net": 0.1},
                         {"status": "closed", "net": 0.0}])
    assert s2["profit_factor"] is None
    assert s2["win_rate"] == pytest.approx(0.5)   # net==0 不算赢（严格 >0）
    # 全亏：profit_factor=0.0（有亏无盈）
    s3 = LS.trade_stats([{"status": "closed", "net": -0.1}])
    assert s3["profit_factor"] == 0.0
    assert s3["avg_net"] == pytest.approx(-0.1)
