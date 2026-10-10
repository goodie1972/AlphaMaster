# tests/unit/test_ledger_stats.py
import math

from web.ledger_stats import derive_rows, compute_stats
from web.live_ledger import LedgerRecord


def R(bar_ts, pos, close):
    return LedgerRecord(ts=0.0, bar_ts=bar_ts, source="okx", symbol="BTCUSDT",
                        timeframe="1h", strategy_sha="x", factor_value=pos,
                        position=pos, direction="up" if pos > 0 else "down",
                        strength=abs(pos), close=close)


def test_adjacency_and_costs():
    recs = [R(0, 0.5, 100.0), R(3600, 0.5, 102.0), R(10800, 0.0, 101.0)]  # 第二段起不相邻
    rows = derive_rows(recs, tf="1h", cost_rate=0.001)
    assert rows[0]["missing"] is False and abs(rows[0]["ret"] - 0.02) < 1e-12
    assert abs(rows[0]["pnl"] - (0.5 * 0.02)) < 1e-12          # 无换手，不扣成本
    assert rows[1]["missing"] is True                           # 7200s ≠ 3600s
    s = compute_stats(rows, cost_rate=0.001, tf="1h")
    assert s["win_rate"] == 1.0                                 # 唯一有效样本：0.5×0.02>0
    assert s["profit_factor"] is None                           # 无亏损样本


def test_turnover_cost():
    rows = derive_rows([R(0, 0.5, 100.0), R(3600, -0.5, 100.0)], tf="1h", cost_rate=0.001)
    assert abs(rows[0]["pnl"] - (0.5 * 0.0 - 1.0 * 0.001)) < 1e-12  # ret=0, 换手=1.0


def test_empty_rows_returns_none_stats():
    assert derive_rows([], tf="1h", cost_rate=0.001) == []
    s = compute_stats([], cost_rate=0.001, tf="1h")
    for k in ("win_rate", "profit_factor", "annualized", "sortino", "max_drawdown"):
        assert s[k] is None
    assert s["live_bars"] == 0
    assert s["theoretical_bars"] is None


def test_sortino_matches_backtest_formula():
    # 镜像 model_core/backtest.py _sortino：mean / max(下行pop std, 0.2×全pop std, 1e-8)
    # × sqrt(ppy)，clip ±20。手算期望，不 import torch。
    pnls = [0.002, -0.003, 0.004, -0.005, 0.007]
    rows = [{"bar_ts": i * 86400, "close": 100.0, "position": 0.5,
             "ret": 0.0, "turnover": 0.0, "pnl": v, "missing": False}
            for i, v in enumerate(pnls)]

    def pstd(xs):  # population std（torch unbiased=False 同口径）
        if not xs:
            return 0.0
        m = sum(xs) / len(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    ppy = 365.25 * 86400 / 86400  # tf="1d" → 365.25
    mean = sum(pnls) / len(pnls)
    downside = [v for v in pnls if v < 0]
    full_std = max(pstd(pnls), 1e-8)
    dstd = max(pstd(downside), max(0.2 * full_std, 1e-8))
    expected = max(-20.0, min(20.0, mean / dstd * math.sqrt(ppy)))

    s = compute_stats(rows, cost_rate=0.001, tf="1d")
    assert abs(s["sortino"] - expected) < 1e-9
    assert s["sortino"] < 20.0  # 样例不触发 clip，公式差异真实可比


def test_nonadjacent_rows_last_record_missing():
    recs = [R(0, 0.5, 100.0), R(3600, 0.5, 102.0), R(14400, 0.2, 103.0)]
    rows = derive_rows(recs, tf="1h", cost_rate=0.001)
    assert rows[0]["missing"] is False
    assert rows[1]["missing"] is True          # 3600→14400 断档 10800s
    assert rows[1]["ret"] is None and rows[1]["pnl"] is None
    assert rows[1]["turnover"] is None          # 不相邻不计换手
    assert rows[2]["missing"] is True           # 末条无 next bar
    assert rows[2]["ret"] is None and rows[2]["turnover"] is None
