"""US_Range_Fade — 美盘 H4 区间 M5 假突破回归策略（纯逻辑层 TDD）。

规则（用户批准的设计）：
- 参照棒 = 每日服务器 16:00–20:00 H4；合格 = 实体 >= 50%×(H−L) 且
  |实体中点−区间中点| <= 25%×(H−L)。不合格当日不交易。
- 上轨=H、下轨=L；M5 监控窗 = 服务器 20:00–24:00。
- 空：M5 收盘>上轨 → 标记（跟踪标记以来最高价，含标记棒）；
  后续收盘<=上轨 → 该收盘价做空；SL=标记后最高价；TP=入场−2×(SL−入场)。
- 多：镜像。
- 每轨每日仅首次突破标记一次；开仓后该轨当日作废；ADX 挡掉的回归同样作废该轨。
- 不回归不开仓；最多 1 笔持仓；持仓期间跳过信号逻辑；
  双触 SL/TP 同根 bar → 先止损；24:00（窗口最后一根 M5 收盘）强平。
- ADX 门 = 入场棒 M5 Wilder ADX(14)，None=关；warmup 缺值按挡处理。
- 成本 cost_dist = 2×cost_rate×entry；net_R = (gross_dist−cost_dist)/risk_dist。
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from datetime import datetime, timedelta, timezone

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_MOD_PATH = ROOT / "mt5" / "backtest_us_range_fade.py"
_spec = importlib.util.spec_from_file_location("backtest_us_range_fade", _MOD_PATH)
urf = importlib.util.module_from_spec(_spec)
sys.modules["backtest_us_range_fade"] = urf
_spec.loader.exec_module(urf)

# 服务器墙钟 = utcfromtimestamp(bar epoch)，故用 UTC datetime 构造 epoch
DAY0 = datetime(2026, 1, 5, tzinfo=timezone.utc)  # 周一


def ts(h: int, m: int = 0, day: int = 0) -> int:
    return int((DAY0.replace(hour=h, minute=m) + timedelta(days=day)).timestamp())


def bar(t: int, o: float, h: float, l: float, c: float) -> dict:
    return {"time": t, "open": o, "high": h, "low": l, "close": c}


# 合格参照棒：body=10, range=14, ratio≈0.714, 中点偏差 0。上轨 112 下轨 98
REF = bar(ts(16), 100.0, 112.0, 98.0, 110.0)


def m5_window(day: int = 0, overrides: dict | None = None) -> list[dict]:
    """20:00..23:55 共 48 根 M5，默认平价 100（轨内），overrides 键 (h,m)→(o,h,l,c)。"""
    overrides = overrides or {}
    out = []
    t = ts(20, 0, day)
    for i in range(48):
        h, m = 20 + (i * 5) // 60, (i * 5) % 60
        o, hh, ll, c = overrides.get((h, m), (100.0, 100.0, 100.0, 100.0))
        out.append(bar(t + i * 300, o, hh, ll, c))
    return out


def safe_after(entry_hm: tuple[int, int], until_hm: tuple[int, int] = (23, 55)) -> dict:
    """(103,115) 区间内的安全棒（不触 SL115/TP103），用于持仓期填充。"""
    out = {}
    start = entry_hm[0] * 60 + entry_hm[1]
    end = until_hm[0] * 60 + until_hm[1]
    for mm in range(start, end + 1, 5):
        out[(mm // 60, mm % 60)] = (110.0, 111.0, 109.0, 110.0)
    return out


P = urf.Params()


# ---------- 参照棒合格判定 ----------

def test_ref_eligible_boundaries():
    # body 恰 50%（10/20）→ 合格；中点偏差 3 <= 5
    assert urf.ref_eligible(bar(ts(16), 100.0, 118.0, 98.0, 110.0), P) is True
    # body 9.9/20 < 50% → 不合格
    assert urf.ref_eligible(bar(ts(16), 100.0, 118.0, 98.0, 109.9), P) is False
    # 零区间 → 不合格
    assert urf.ref_eligible(bar(ts(16), 100.0, 100.0, 100.0, 100.0), P) is False


def test_ref_center_criterion_with_custom_params():
    # 默认参数下 body>=50% 数学上蕴含偏差<=25%，故用宽 body 窄容差参数单独测居中判据
    p = urf.Params(body_ratio=0.3, center_tol=0.1)
    # body 8/20=0.4>=0.3，但实体中点 114 偏离区间中点 108 达 6 > 2 → 不合格
    assert urf.ref_eligible(bar(ts(16), 110.0, 118.0, 98.0, 118.0), p) is False
    # 偏差恰等于容差 → 合格：body 12/20=0.6, mid 112 vs 108 偏差 4, tol=0.2×20=4
    p2 = urf.Params(body_ratio=0.3, center_tol=0.2)
    assert urf.ref_eligible(bar(ts(16), 106.0, 118.0, 98.0, 118.0), p2) is True


def test_ineligible_ref_no_trades():
    ov = {(20, 0): (112.5, 114.0, 112.0, 113.0), (20, 5): (113.0, 113.5, 110.5, 111.0)}
    weak = bar(ts(16), 100.0, 112.0, 98.0, 101.0)  # body 1/14
    assert urf.simulate_day(weak, m5_window(0, ov), {}, P) == []


# ---------- 空头全流程 ----------

def test_short_full_flow_tp():
    ov = {
        (20, 0): (112.5, 114.0, 112.0, 113.0),   # 收盘 113>112 标记，极值 114
        (20, 5): (113.0, 115.0, 112.8, 114.0),   # 仍轨上，极值抬到 115
        (20, 10): (114.0, 114.5, 110.5, 111.0),  # 收盘 111<=112 回归 → 做空@111
        (20, 15): (111.0, 111.5, 104.0, 105.0),  # 未触 TP103
        (20, 20): (105.0, 105.5, 102.0, 102.5),  # low<=103 → TP@103
    }
    trades = urf.simulate_day(REF, m5_window(0, ov), {}, P)
    assert len(trades) == 1
    t = trades[0]
    assert t["direction"] == "short"
    assert t["mark_time"] == ts(20, 0)
    assert t["entry_time"] == ts(20, 10)
    assert t["entry_price"] == 111.0
    assert t["sl"] == 115.0
    assert t["tp"] == pytest.approx(103.0)
    assert t["exit_time"] == ts(20, 20)
    assert t["exit_price"] == pytest.approx(103.0)
    assert t["exit_reason"] == "tp"
    assert t["risk_dist"] == pytest.approx(4.0)
    assert t["gross_dist"] == pytest.approx(8.0)
    assert t["cost_dist"] == pytest.approx(2 * 0.0003 * 111.0)
    assert t["net_R"] == pytest.approx((8.0 - 0.0666) / 4.0)


def test_short_mark_bar_extreme_included():
    ov = {
        (20, 0): (112.5, 116.0, 112.0, 113.0),   # 标记棒最高 116 是全程最高
        (20, 5): (113.0, 113.5, 112.6, 113.2),
        (20, 10): (113.0, 113.4, 110.5, 111.0),  # 回归 → SL 必须是 116
        (20, 15): (111.0, 111.5, 104.0, 105.0),
        (20, 20): (105.0, 105.5, 100.5, 101.5),  # TP=111−2×5=101
    }
    trades = urf.simulate_day(REF, m5_window(0, ov), {}, P)
    assert len(trades) == 1
    assert trades[0]["sl"] == 116.0
    assert trades[0]["tp"] == pytest.approx(101.0)
    assert trades[0]["exit_reason"] == "tp"


def test_no_return_no_trade():
    ov = {(20, 0): (112.5, 114.0, 112.0, 113.0)}
    for mm in range(20 * 60 + 5, 24 * 60, 5):
        ov[(mm // 60, mm % 60)] = (113.0, 114.0, 112.5, 113.5)  # 全程不回归
    assert urf.simulate_day(REF, m5_window(0, ov), {}, P) == []


def test_double_touch_sl_first():
    ov = {
        (20, 0): (112.5, 114.0, 112.0, 113.0),
        (20, 5): (113.0, 115.0, 112.8, 114.0),
        (20, 10): (114.0, 114.5, 110.5, 111.0),  # 空@111 SL115 TP103
        (20, 15): (111.0, 116.0, 102.0, 108.0),  # 同根双触 → 先止损
    }
    trades = urf.simulate_day(REF, m5_window(0, ov), {}, P)
    assert len(trades) == 1
    t = trades[0]
    assert t["exit_reason"] == "sl"
    assert t["exit_price"] == pytest.approx(115.0)
    assert t["gross_dist"] == pytest.approx(-4.0)
    assert t["net_R"] == pytest.approx((-4.0 - 0.0666) / 4.0)


def test_rail_consumed_after_entry():
    ov = {
        (20, 0): (112.5, 114.0, 112.0, 113.0),
        (20, 5): (113.0, 113.5, 110.5, 111.0),   # 回归 → 空@111 SL114 TP105
        (20, 10): (111.0, 111.5, 104.0, 104.5),  # TP@105
        (20, 15): (105.0, 113.5, 105.0, 113.0),  # 再次突破上轨 → 该轨已作废
        (20, 20): (113.0, 113.5, 110.5, 111.0),  # 再次回归 → 不得开仓
    }
    trades = urf.simulate_day(REF, m5_window(0, ov), {}, P)
    assert len(trades) == 1


# ---------- 多头镜像 ----------

def test_long_full_flow_tp():
    ov = {
        (20, 0): (97.5, 98.0, 96.0, 97.0),      # 收盘 97<98 标记，极值 96
        (20, 5): (97.0, 97.2, 95.0, 96.5),      # 极值 95
        (20, 10): (96.5, 99.5, 96.0, 99.0),     # 收盘 99>=98 回归 → 做多@99 SL95 TP107
        (20, 15): (99.0, 106.0, 98.5, 105.0),   # 未触 TP107
        (20, 20): (105.0, 108.0, 104.5, 107.5), # high>=107 → TP@107
    }
    trades = urf.simulate_day(REF, m5_window(0, ov), {}, P)
    assert len(trades) == 1
    t = trades[0]
    assert t["direction"] == "long"
    assert t["entry_price"] == 99.0
    assert t["sl"] == 95.0
    assert t["tp"] == pytest.approx(107.0)
    assert t["exit_reason"] == "tp"
    assert t["exit_price"] == pytest.approx(107.0)
    assert t["net_R"] == pytest.approx((8.0 - 2 * 0.0003 * 99.0) / 4.0)


def test_long_double_touch_sl_first():
    ov = {
        (20, 0): (97.5, 98.0, 96.0, 97.0),
        (20, 5): (97.0, 97.2, 95.0, 96.5),
        (20, 10): (96.5, 99.5, 96.0, 99.0),     # 多@99 SL95 TP107
        (20, 15): (99.0, 108.0, 94.0, 100.0),   # 双触 → 先止损@95
    }
    trades = urf.simulate_day(REF, m5_window(0, ov), {}, P)
    assert len(trades) == 1
    assert trades[0]["exit_reason"] == "sl"
    assert trades[0]["exit_price"] == pytest.approx(95.0)


# ---------- ADX 门 ----------

def test_adx_gate_blocks_and_consumes_rail():
    ov = {
        (20, 0): (112.5, 114.0, 112.0, 113.0),
        (20, 5): (113.0, 113.5, 112.5, 113.0),   # 仍轨上，回归发生在 20:10
        (20, 10): (113.0, 113.5, 110.5, 111.0),  # 回归但 ADX=30 >= 25 → 挡 + 作废
        (20, 15): (111.0, 111.5, 110.5, 111.0),  # 再回归 ADX=10 也不得开仓
    }
    adx = {ts(20, 10): 30.0, ts(20, 15): 10.0}
    p = urf.Params(adx_max=25.0)
    assert urf.simulate_day(REF, m5_window(0, ov), adx, p) == []
    # 门关（None）同样数据 → 正常开仓
    trades = urf.simulate_day(REF, m5_window(0, ov), adx, P)
    assert len(trades) == 1


def test_adx_gate_pass_below_threshold():
    ov = {
        (20, 0): (112.5, 114.0, 112.0, 113.0),
        (20, 5): (113.0, 113.5, 112.5, 113.0),
        (20, 10): (113.0, 113.5, 110.5, 111.0),
        (20, 15): (111.0, 111.5, 104.0, 105.0),
        (20, 20): (105.0, 105.5, 102.0, 102.5),
    }
    adx = {ts(20, 10): 24.9}
    p = urf.Params(adx_max=25.0)
    trades = urf.simulate_day(REF, m5_window(0, ov), adx, p)
    assert len(trades) == 1


def test_adx_missing_value_blocks():
    ov = {
        (20, 0): (112.5, 114.0, 112.0, 113.0),
        (20, 5): (113.0, 113.5, 112.5, 113.0),
        (20, 10): (113.0, 113.5, 110.5, 111.0),  # 回归但 adx_map 无值 → 保守挡
    }
    p = urf.Params(adx_max=25.0)
    assert urf.simulate_day(REF, m5_window(0, ov), {}, p) == []


# ---------- 24:00 强平 / 窗口 ----------

def test_eod_force_close():
    ov = {
        (20, 0): (112.5, 114.0, 112.0, 113.0),
        (20, 5): (113.0, 115.0, 112.8, 114.0),
        (20, 10): (114.0, 114.5, 110.5, 111.0),  # 空@111 SL115 TP103
    }
    ov.update(safe_after((20, 15), (23, 50)))
    ov[(23, 55)] = (109.0, 109.5, 107.5, 108.0)  # 最后一根收盘 108 → 强平
    trades = urf.simulate_day(REF, m5_window(0, ov), {}, P)
    assert len(trades) == 1
    t = trades[0]
    assert t["exit_reason"] == "eod"
    assert t["exit_time"] == ts(23, 55)
    assert t["exit_price"] == pytest.approx(108.0)
    assert t["gross_dist"] == pytest.approx(3.0)


def test_sl_on_last_bar_beats_eod():
    ov = {
        (20, 0): (112.5, 114.0, 112.0, 113.0),
        (20, 5): (113.0, 115.0, 112.8, 114.0),
        (20, 10): (114.0, 114.5, 110.5, 111.0),
    }
    ov.update(safe_after((20, 15), (23, 50)))
    ov[(23, 55)] = (110.0, 116.0, 109.0, 115.5)  # 最后一根触 SL → reason=sl
    trades = urf.simulate_day(REF, m5_window(0, ov), {}, P)
    assert trades[0]["exit_reason"] == "sl"


def test_bars_outside_window_ignored():
    # 19:55 突破上轨不标记（窗外）；窗内无突破 → 无交易
    early = bar(ts(19, 55), 112.5, 114.0, 112.0, 113.0)
    late = bar(ts(0, 0, 1), 112.5, 114.0, 112.0, 113.0)  # 次日 00:00 已出窗
    assert urf.simulate_day(REF, [early] + m5_window(0) + [late], {}, P) == []


# ---------- 持仓期间跳过信号 / 平仓后另一轨仍可标记 ----------

def test_hold_skips_signals_then_other_rail_marks():
    ov = {
        (20, 0): (112.5, 114.0, 112.0, 113.0),
        (20, 5): (113.0, 113.5, 110.5, 111.0),   # 空@111 SL114 TP105
        (20, 10): (110.0, 110.5, 97.0, 97.5),    # 持仓中 low<=105 → TP@105；收盘<98 不得标记多
        (20, 15): (97.5, 98.0, 96.0, 97.0),      # 平仓后首次跌破下轨 → 标记多，极值 96
        (20, 20): (97.0, 99.5, 96.5, 99.0),      # 回归 → 多@99 SL96 TP105
        (20, 25): (99.0, 106.0, 98.5, 105.5),    # high>=105 → TP@105
    }
    trades = urf.simulate_day(REF, m5_window(0, ov), {}, P)
    assert len(trades) == 2
    assert trades[0]["direction"] == "short"
    assert trades[0]["exit_reason"] == "tp"
    assert trades[1]["direction"] == "long"
    assert trades[1]["mark_time"] == ts(20, 15)  # 不是持仓中的 20:10
    assert trades[1]["sl"] == 96.0
    assert trades[1]["exit_reason"] == "tp"


# ---------- Wilder ADX ----------

def test_adx_warmup_and_bounds():
    bars = m5_window(0)
    adx = urf.wilder_adx(bars, period=14)
    assert len(adx) == len(bars)
    assert all(v is None for v in adx[:26])
    assert adx[26] is not None
    assert all(v is None or 0.0 <= v <= 100.0 for v in adx)


def test_adx_flat_market_zero():
    adx = urf.wilder_adx(m5_window(0), period=14)
    assert all(v == 0.0 for v in adx if v is not None)


def test_adx_strong_uptrend_high():
    bars = []
    for i in range(80):
        o, c = 100.0 + i, 101.0 + i
        bars.append(bar(ts(20) + i * 300, o, 102.0 + i, 99.5 + i, c))
    adx = urf.wilder_adx(bars, period=14)
    assert adx[-1] > 90.0


# ---------- 汇总 / 编排 ----------

def _mk_trade(net_r: float, entry_time: int) -> dict:
    return {"net_R": net_r, "entry_time": entry_time, "direction": "short"}


def test_summarize_equity_and_metrics():
    trades = [_mk_trade(2.0, ts(20, 10)), _mk_trade(-1.0, ts(20, 10, 1))]
    s = urf.summarize(trades, initial_equity=5000.0, risk_per_trade=0.01)
    assert s["n_trades"] == 2
    assert s["wins"] == 1
    assert s["win_rate"] == pytest.approx(0.5)
    assert s["final_equity"] == pytest.approx(5049.0)
    assert s["equity_curve"] == pytest.approx([5000.0, 5100.0, 5049.0])
    assert s["profit_factor"] == pytest.approx(100.0 / 51.0)
    assert s["total_return"] == pytest.approx(0.0098)
    assert s["max_drawdown"] == pytest.approx(51.0 / 5100.0)


def test_summarize_empty():
    s = urf.summarize([], initial_equity=5000.0)
    assert s["n_trades"] == 0
    assert s["final_equity"] == 5000.0
    assert s["total_return"] == 0.0
    assert s["max_drawdown"] == 0.0
    assert s["win_rate"] is None
    assert s["profit_factor"] is None


def test_run_backtest_day_reset_and_ref_hour_filter():
    def day_ov(day):
        return {
            (20, 0): (112.5, 114.0, 112.0, 113.0),
            (20, 5): (113.0, 113.5, 110.5, 111.0),  # 空@111 SL114 TP105
            (20, 10): (111.0, 111.5, 104.0, 104.5),  # TP@105
        }

    refs = [
        bar(ts(12), 100.0, 112.0, 98.0, 110.0),  # 12:00 H4 → 忽略
        REF,
        bar(ts(16, 0, 1), 100.0, 112.0, 98.0, 110.0),
    ]
    m5 = m5_window(0, day_ov(0)) + m5_window(1, day_ov(1))
    res = urf.run_backtest(refs, m5, {}, P)
    assert len(res["trades"]) == 2
    assert res["trades"][0]["entry_time"] == ts(20, 5)
    assert res["trades"][1]["entry_time"] == ts(20, 5, 1)
    assert res["summary"]["n_trades"] == 2
    assert "by_year" in res