# tests/unit/test_ledger_stats.py
import math

import pytest

from web.ledger_stats import derive_rows, compute_stats
from web.live_ledger import LedgerRecord

_SPY = 365.25 * 86400.0  # 年秒数，同 backtest._SECONDS_PER_YEAR


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
    rows = [{"bar_ts": i * 2 * 86400, "close": 100.0, "position": 0.5,
             "ret": 0.0, "turnover": 0.0, "pnl": v, "missing": False}
            for i, v in enumerate(pnls)]

    def pstd(xs):  # population std（torch unbiased=False 同口径）
        if not xs:
            return 0.0
        m = sum(xs) / len(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    # 密度 ppy（镜像 estimate_periods_per_year）：5 行、span=4×2d
    ppy = 5 * _SPY / (4 * 2 * 86400.0)
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


def V(bar_ts, pnl, pos=0.5):
    """手工构造有效行（绕过 derive_rows 以精确控制 span/pnl）。"""
    return {"bar_ts": bar_ts, "close": 100.0, "position": pos, "ret": 0.0,
            "turnover": 0.0, "pnl": pnl, "missing": False}


def test_annualized_density_ppy_contiguous_crypto():
    # 25 条连续 1h 行：span=24h，密度 ppy = 25*SPY/86400 = 9131.25（≈日历 8766 量级）
    recs = [R(i * 3600, 0.5, 100.0 * 1.02 ** i) for i in range(25)]
    rows = derive_rows(recs, tf="1h", cost_rate=0.001)
    s = compute_stats(rows, cost_rate=0.001, tf="1h")
    ppy = 25 * _SPY / (24 * 3600.0)             # 9131.25
    assert 8766.0 * 0.95 < ppy < 8766.0 * 1.05  # sanity：连续样本密度≈日历值
    assert s["annualized"] == pytest.approx(0.01 * ppy, rel=1e-6)


def test_annualized_density_ppy_weekend_gapped_fx():
    # 外汇 H1、仅周一有数据（周末闭市）：密度 ppy « 日历 8766
    rows = [V(d * 86400 + h * 3600, 0.01) for d in (0, 7, 14, 21) for h in (0, 1)]
    s = compute_stats(rows, cost_rate=0.001, tf="1h")
    ppy = 8 * _SPY / (21 * 86400.0 + 3600.0)    # ≈138.9
    assert ppy < 8766.0 / 10                    # 密度远低于日历口径
    assert s["annualized"] == pytest.approx(0.01 * ppy, rel=1e-9)


def test_ppy_span_includes_missing_rows():
    # missing 行同样是真实观测：计入行数与 span
    rows = [V(0, 0.01), V(3600, 0.01),
            {"bar_ts": 10 * 3600, "close": 100.0, "position": 0.5, "ret": None,
             "turnover": None, "pnl": None, "missing": True}]
    s = compute_stats(rows, cost_rate=0.001, tf="1h")
    ppy = 3 * _SPY / (10 * 3600.0)
    assert s["annualized"] == pytest.approx(0.01 * ppy, rel=1e-9)


def test_ppy_falls_back_to_calendar_when_span_zero():
    rows = [V(0, 0.01), V(0, 0.01)]             # 同一 bar_ts → span 0 → 回退日历
    s = compute_stats(rows, cost_rate=0.001, tf="1h")
    assert s["annualized"] == pytest.approx(0.01 * (_SPY / 3600.0), rel=1e-9)


def test_ppy_falls_back_with_single_row():
    s = compute_stats([V(0, 0.01)], cost_rate=0.001, tf="1h")
    assert s["annualized"] == pytest.approx(0.01 * (_SPY / 3600.0), rel=1e-9)


def test_unknown_tf_raises():
    with pytest.raises(ValueError):
        compute_stats([], cost_rate=0.001, tf="3x")
    with pytest.raises(ValueError):
        derive_rows([], tf="3x", cost_rate=0.001)


def test_zero_multiple_tf_raises():
    with pytest.raises(ValueError):   # H0/M0 数字部分 <1 → tf_s=0 会除零
        compute_stats([], cost_rate=0.001, tf="H0")


def test_theoretical_bars_explicit_ts():
    s = compute_stats([], cost_rate=0.001, tf="1h", first_ts=0, last_ts=7200)
    assert s["theoretical_bars"] == 3       # (7200-0)//3600 + 1


def test_profit_factor_zero_when_only_losses():
    s = compute_stats([V(0, -0.01), V(3600, -0.02)], cost_rate=0.001, tf="1h")
    assert s["profit_factor"] == 0.0        # 有亏损无盈利 → 0.0（区别于 None）


def test_close_le_zero_marks_missing():
    rows = derive_rows([R(0, 0.5, 0.0), R(3600, 0.5, 100.0)], tf="1h", cost_rate=0.001)
    assert rows[0]["missing"] is True and rows[0]["pnl"] is None


def test_sortino_clipped_at_pm20():
    s = compute_stats([V(0, 5.0), V(3600, 4.0)], cost_rate=0.001, tf="1h")
    assert s["sortino"] == 20.0             # 大均值/小dstd → 上裁剪
    s = compute_stats([V(0, -5.0), V(3600, -4.0)], cost_rate=0.001, tf="1h")
    assert s["sortino"] == -20.0            # 下裁剪


def test_unsorted_input_is_sorted():
    rows = derive_rows([R(3600, 0.5, 102.0), R(0, 0.5, 100.0)], tf="1h", cost_rate=0.001)
    assert rows[0]["bar_ts"] == 0 and rows[0]["missing"] is False
    assert rows[1]["bar_ts"] == 3600 and rows[1]["missing"] is True
