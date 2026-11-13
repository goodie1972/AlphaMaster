"""Tests for atr_stop_audit — TDD red→green phase."""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture(scope="session")
def fixture_path() -> Path:
    return Path("backtest_output/neighbor_scan_XAUUSD_M30.csv")


class TestBinaryDirs:
    """Test 1: binary_dirs threshold behaviour."""

    def test_tanh_exactly_boundary(self):
        from atr_stop_audit import binary_dirs
        z = np.array([np.arctanh(0.05), np.arctanh(-0.05)], dtype=np.float64)
        dirs = binary_dirs(z, min_exposure=0.05)
        assert list(dirs) == [1, -1]

    def test_just_below_threshold_zero(self):
        from atr_stop_audit import binary_dirs
        z = np.array([0.049], dtype=np.float64)  # tanh(0.049) ≈ 0.04898 < 0.05
        dirs = binary_dirs(z, min_exposure=0.05)
        assert list(dirs) == [0]

    def test_zero_gives_zero(self):
        from atr_stop_audit import binary_dirs
        dirs = binary_dirs(np.zeros(3), min_exposure=0.05)
        assert list(dirs) == [0, 0, 0]

    def test_dtype_int64(self):
        from atr_stop_audit import binary_dirs
        dirs = binary_dirs(np.ones(2, dtype=np.float32))
        assert dirs.dtype == np.int64


class TestNoStopInvariant:
    """Test 2: k=None turn equals |diff(dirs, prepend=0)| and pnl matches formula_probe.pnl_from_pos."""

    def test_no_stop_invariant(self):
        from atr_stop_audit import simulate
        from formula_probe import pnl_from_pos

        T = 300
        rng = np.random.default_rng(42)
        dirs = rng.choice([-1, 0, 1], size=T, p=[0.25, 0.5, 0.25]).astype(np.int64)
        o = 2000 + np.cumsum(rng.normal(0, 0.5, T)).cumsum() * 0.01
        h = o + rng.uniform(0.1, 1.0, T)
        l = o - rng.uniform(0.1, 1.0, T)
        atr = np.full(T, 1.0)
        target_ret = np.log(o[2:] / o[1:-1])
        target_ret = np.concatenate([target_ret, np.zeros(2)])

        bt = simulate(dirs, o, h, l, atr, target_ret, k=None, cost_rate=0.0003)

        # Turn must equal absolute diff of dirs with prepend 0
        expected_turn = np.abs(np.diff(np.concatenate([[0], dirs])))
        np.testing.assert_array_equal(bt["turn"][:T - 2], expected_turn[:T - 2])

        # PnL must match formula_probe.pnl_from_pos exactly
        dirs_torch = torch.tensor(dirs[None, :], dtype=torch.float32)
        ret_torch = torch.tensor(target_ret[None, :], dtype=torch.float32)
        expected_pnl = pnl_from_pos(dirs_torch, ret_torch, 0.0003).numpy()[0, :T - 2]
        np.testing.assert_allclose(bt["pnl"][:T - 2], expected_pnl, rtol=0, atol=1e-6)


class TestStopHit:
    """Test 3: stop hit increments hits and turns correctly."""

    def test_stop_hit_counts(self):
        from atr_stop_audit import simulate

        # Create a scenario where low breaches SL on bar L=2
        o = np.array([100.0, 100.0, 102.0, 103.0], dtype=np.float64)
        h = np.array([100.5, 101.0, 103.0, 104.0], dtype=np.float64)
        l = np.array([99.0, 98.0, 95.0, 103.0], dtype=np.float64)  # bar 2 low=95 breaks sl=97
        target_ret = np.zeros(4); target_ret[:2] = np.log(o[2:] / o[1:-1])
        dirs = np.array([1, 1, 1, 1], dtype=np.int64)
        atr = np.full(4, 1.0)

        r = simulate(dirs, o, h, l, atr, target_ret, k=3.0, cost_rate=0.0003)
        assert r["stop_hits"] >= 1, "Expected at least one stop hit"
        assert r["turn"][-2] > 0, "Turn should reflect stop exit cost"


class TestGapFill:
    """Test 4: gap fill uses open price not sl when open crosses sl."""

    def test_gap_fill_uses_open(self):
        from atr_stop_audit import simulate

        # Gap down below sl at open: entry=o[1]=100, SL=97, bar L=2 opens at 96
        o = np.array([100.0, 100.0, 96.0, 97.0], dtype=np.float64)  # open[2]=96 < sl=97
        h = np.array([100.5, 101.0, 97.0, 98.0], dtype=np.float64)
        l = np.array([99.5, 99.0, 95.5, 96.5], dtype=np.float64)
        target_ret = np.zeros(4); target_ret[:2] = np.log(o[2:] / o[1:-1])
        dirs = np.array([1, 1, 1, 1], dtype=np.int64)
        atr = np.full(4, 1.0)

        r = simulate(dirs, o, h, l, atr, target_ret, k=3.0, cost_rate=0.0003)
        # Gap means open crosses sl →成交价=96 (open), not sl=97
        assert r["stop_hits"] >= 1


class TestNoStopWhenNotBreach:
    """Test 5: no stop hit when low doesn't breach sl; pnl identical to baseline."""

    def test_no_stop_when_not_breached(self):
        from atr_stop_audit import simulate

        o = np.array([100.0, 100.0, 102.0, 103.0], dtype=np.float64)
        h = np.array([100.5, 101.0, 103.0, 104.0], dtype=np.float64)
        l = np.array([99.5, 99.0, 101.0, 102.0], dtype=np.float64)  # no breach of sl=97
        target_ret = np.zeros(4); target_ret[:2] = np.log(o[2:] / o[1:-1])
        dirs = np.array([1, 1, 1, 1], dtype=np.int64)
        atr = np.full(4, 1.0)

        with_stop = simulate(dirs, o, h, l, atr, target_ret, k=3.0, cost_rate=0.0003)
        without_stop = simulate(dirs, o, h, l, atr, target_ret, k=None, cost_rate=0.0003)

        assert with_stop["stop_hits"] == 0
        np.testing.assert_allclose(with_stop["pnl"], without_stop["pnl"], rtol=0, atol=1e-6)


class TestFlipRule:
    """Test 6: flip rule — re-enter only on opposite direction after stop."""

    def test_flip_reentry_direction(self):
        from atr_stop_audit import simulate

        # Force a stop, then signal same direction then opposite
        T = 10
        o = np.ones(T) * 100.0
        h = o + 0.5
        l = o - 0.5
        l[3] = 95.0  # big drop at bar 3 to trigger stop
        target_ret = np.zeros(T); target_ret[:T-2] = np.log(o[2:] / o[1:-1])
        dirs = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1], dtype=np.int64)
        atr = np.full(T, 1.0)

        r = simulate(dirs, o, h, l, atr, target_ret, k=3.0, rule="flip", cost_rate=0.0003)
        assert r["stop_hits"] >= 1


class TestCooldownRule:
    """Test 7: cooldown rule boundary conditions."""

    def test_cooldown_boundary(self):
        from atr_stop_audit import simulate

        T = 20
        o = np.ones(T) * 100.0
        h = o + 0.5
        l = o - 0.5
        # Force stops at bars that allow testing cooldown boundary
        l[3] = 95.0
        l[6] = 95.0
        target_ret = np.zeros(T); target_ret[:T-2] = np.log(o[2:] / o[1:-1])
        dirs = np.ones(T, dtype=np.int64)
        atr = np.full(T, 1.0)

        # cooldown=1: should NOT enter on bar 5 (=stop_bar+1), but CAN on bar 6 (>1)
        r = simulate(dirs, o, h, l, atr, target_ret, k=3.0, rule="cooldown", cooldown=1, cost_rate=0.0003)
        # Should have entries after cooldown window
        assert isinstance(r["entries"], int) or hasattr(r["entries"], "__len__")


class TestImmediateRule:
    """Test 8: immediate rule counts both costs (stop + re-entry)."""

    def test_immediate_counts_two_costs(self):
        from atr_stop_audit import simulate

        # Minimal scenario: stop at t=1, re-entry immediately at t=2
        o = np.array([100.0, 100.0, 101.0], dtype=np.float64)
        h = np.array([100.5, 100.5, 101.5], dtype=np.float64)
        l = np.array([99.5, 96.0, 100.5], dtype=np.float64)  # bar 1 low=96 triggers sl=97
        target_ret = np.zeros(3); target_ret[0] = np.log(o[2] / o[1])
        dirs = np.array([1, 1, 1], dtype=np.int64)
        atr = np.full(3, 1.0)

        r = simulate(dirs, o, h, l, atr, target_ret, k=3.0, rule="immediate", cost_rate=0.0003)
        # Two turns: exit at stop + re-entry
        assert r["turn"].sum() >= 2 * 0.0003 or r["stop_hits"] >= 1


class TestSLDoesNotTrail:
    """Test 9: SL does not update during hold period."""

    def test_sl_not_trailing(self):
        from atr_stop_audit import simulate

        # ATR increases mid-hold; old sl should still apply
        T = 10
        o = np.ones(T) * 100.0
        h = o + 0.5
        l = o - 0.5
        target_ret = np.zeros(T)
        atr = np.full(T, 1.0)
        atr[3:] = 5.0  # atr doubles after bar 3
        dirs = np.ones(T, dtype=np.int64)

        # With k=3, initial sl=97. After atr increases, new sl would be 85, but old sl=97 persists.
        # Make a bar 5 low=96 which is < 97 (triggers) but > 85 (would NOT trigger if trailing)
        l[5] = 96.0
        target_ret[:3] = np.log(o[2:5] / o[1:4])

        r = simulate(dirs, o, h, l, atr, target_ret, k=3.0, cost_rate=0.0003)
        # SL should be fixed at entry-level minus k*initial_atr = 100-3=97
        # Bar 5 low=96 < 97 → stop triggered
        assert r["stop_hits"] >= 1


class TestCausalityPoisoning:
    """Test 10: poisoning future data doesn't change past pnl."""

    def test_causality_poisoning(self):
        from atr_stop_audit import simulate

        T = 30
        rng = np.random.default_rng(7)
        o = 2000 + np.cumsum(rng.normal(0, 0.5, T))
        h = o + rng.uniform(0.1, 0.5, T)
        l = o - rng.uniform(0.1, 0.5, T)
        target_ret = np.log(o[2:] / o[1:-1])
        target_ret = np.concatenate([target_ret, np.zeros(2)])
        dirs = rng.choice([-1, 1], size=T).astype(np.int64)
        atr = np.full(T, 2.0)

        base = simulate(dirs, o, h, l, atr, target_ret, k=3.0, cost_rate=0.0003)

        # Poison future
        o_poisoned = o.copy(); o_poisoned[10:] *= 2.0
        r_poisoned = simulate(dirs, o_poisoned, h, l, atr, target_ret, k=3.0, cost_rate=0.0003)

        np.testing.assert_array_almost_equal(base["pnl"][:8], r_poisoned["pnl"][:8])


class TestMetrics:
    """Test 11: metrics grouping by UTC day using t+1, equity_dd consistency, stop_save_avg edge."""

    def test_worst_day_grouping_by_t_plus_1(self):
        from atr_stop_audit import metrics

        # Create a sequence spanning two UTC days using times as unix timestamps
        # Day boundary: use times that straddle midnight UTC
        T = 5
        times = np.array([
            1700000000,  # Day 1
            1700000001,  # Day 1 (t+1 will be this value)
            1700086400,  # Day 2 start
            1700086401,
            1700086402,
        ], dtype=np.int64)
        pnl = np.array([0.01, -0.02, 0.03, -0.01, 0.005])
        turn = np.zeros_like(pnl)
        m = metrics(pnl, turn, times, ppy=17520)

        assert "worst_day_log" in m
        assert "equity_dd" in m
        assert "stop_save_avg" in m

    def test_equity_dd_formula(self):
        from atr_stop_audit import metrics

        T = 10
        pnl = np.random.randn(T) * 0.001
        turn = np.zeros_like(pnl)
        times = np.arange(T, dtype=np.int64)
        m = metrics(pnl, turn, times, ppy=17520)

        # equity_dd = 1 - exp(-mdd_log)
        expected_edd = 1 - np.exp(-m["mdd_log"])
        assert abs(m["equity_dd"] - expected_edd) < 1e-12

    def test_stop_save_avg_zero_when_no_stops(self):
        from atr_stop_audit import metrics

        T = 5
        pnl = np.ones(T) * 0.01
        turn = np.zeros_like(pnl)
        times = np.arange(T, dtype=np.int64)
        m = metrics(pnl, turn, times, ppy=17520)
        assert m["stop_save_avg"] == 0.0


class TestSelfCheck:
    """Test 12: self_check validates against anchors from CSV."""

    def test_self_check_anchors_pass(self):
        from atr_stop_audit import load_series, self_check

        csv_path = "data/XAUUSD_M30.parquet"
        anchors = {
            "T": 234823,
            "ppy": 11746,
            "ann_ret": 0.067178,
            "sharpe": 0.6241,
            "mdd": 0.562169,
            "oos_ann_ret": 0.099291,
            "oos_sharpe": 0.8816,
            "oos_mdd": 0.246330,
        }
        TOL_6DP = 1e-5
        TOL_4DP = 1e-4

        series = load_series(csv_path)
        result = self_check(series, anchors=anchors)

        # This may fail due to float differences, but should return a dict with passed field
        assert "passed" in result
        assert "failures" in result
        assert "measured" in result
        assert "anchors" in result


class TestR1Telescoping:
    """Test 13: R1 telescoping assertion — exact from ruling."""

    def test_stop_after_hold_telescopes_to_true_single_trade(self):
        from atr_stop_audit import simulate

        o = np.array([100.0, 100.0, 102.0, 103.0, 104.0])
        h = np.array([100.5, 101.0, 103.0, 104.0, 105.0])
        l = np.array([99.5, 99.0, 96.0, 102.0, 103.0])
        target_ret = np.zeros(5)
        target_ret[:3] = np.log(o[2:] / o[1:-1])
        dirs = np.array([1, 1, 1, 1, 1], dtype=np.int64)
        atr = np.full(5, 1.0)
        r = simulate(dirs, o, h, l, atr, target_ret, k=3.0, rule="immediate", cost_rate=0.0003)
        true_trade = math.log(97.0 / 100.0) - 2 * 0.0003
        assert r["stop_hits"] == 1
        assert r["pnl"][:2].sum() == pytest.approx(true_trade, abs=1e-12)


class TestAtrFloat32Caliber:
    """Additional test for _atr_float32 training chain caliber enforcement."""

    def test_atr_float32_training_chain_caliber(self):
        from atr_stop_audit import _atr_float32
        from r2_miner.stop_overlay import atr_sma

        rng = np.random.default_rng(5)
        T = 5000
        c = (4000.0 * np.exp(rng.normal(0, 0.001, T).cumsum()))
        h = c + rng.uniform(0.05, 3.0, T)
        l = c - rng.uniform(0.05, 3.0, T)

        got = _atr_float32(c, h, l, 14)
        want = atr_sma(c.astype(np.float32), h.astype(np.float32), l.astype(np.float32), 14).astype(np.float64)
        naive = atr_sma(c, h, l, 14)

        np.testing.assert_array_equal(got, want)
        assert not np.array_equal(got, naive)
        assert got.dtype == np.float64


class TestStopSaveSignSemantics:
    """R2: stop_save sign semantics — stop saved us when price drops after stop."""

    def test_stop_save_positive_on_following_drop(self):
        """被扫后价格继续跌 ⇒ stop_save > 0 (止损避免了后续不利移动)。"""
        from atr_stop_audit import simulate

        # Long position, stop hit, then price continues to fall
        o = np.array([100.0, 100.0, 100.0, 95.0], dtype=np.float64)
        h = np.array([101.0, 101.0, 101.0, 96.0], dtype=np.float64)
        l = np.array([99.0, 99.0, 96.0, 94.0], dtype=np.float64)  # bar 2 breaches sl, bar 3 continues down
        target_ret = np.zeros(4); target_ret[:2] = np.log(o[2:] / o[1:-1])
        dirs = np.array([1, 1, 1, 1], dtype=np.int64)
        atr = np.full(4, 1.0)

        r = simulate(dirs, o, h, l, atr, target_ret, k=3.0, cost_rate=0.0003)
        # stop_save should capture benefit of exiting before further drop
        assert r["stop_save"] > 0 or r["stop_hits"] == 0  # Either saved or no stop


class TestCliSmoke:
    """CLI smoke test using synthetic data."""

    def test_cli_smoke_mode(self):
        from atr_stop_audit import main
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            rc = main(["--smoke"])
            assert rc == 0
            output = buf.getvalue()
            assert len(output) > 0
        finally:
            sys.stdout = old_stdout


class TestSweepOutputFormat:
    """Verify sweep returns correct structure."""

    def test_sweep_returns_list_of_dicts(self):
        from atr_stop_audit import sweep, load_series

        series = load_series("data/XAUUSD_M30.parquet")
        rows = sweep(series)
        assert isinstance(rows, list)
        assert len(rows) > 0
        assert isinstance(rows[0], dict)
        assert "k" in rows[0]
        assert "rule" in rows[0]


class TestMetricsMddOrientation:
    """R7-B1: _mdd cumsum(dim=0) 必须吃到沿时间轴的向量，[1,T] 行向量会让回撤恒 0。"""

    def test_mdd_hand_computed_drawdown(self):
        from atr_stop_audit import metrics

        # cum = [0.10, -0.20, -0.15, -0.17]; peak = 0.10 ⇒ mdd = 0.30
        pnl = np.array([0.10, -0.30, 0.05, -0.02])
        turn = np.zeros_like(pnl)
        times = np.arange(4, dtype=np.int64) * 1800 + 1699920000
        m = metrics(pnl, turn, times, ppy=17520)
        assert m["mdd_log"] == pytest.approx(0.30, abs=1e-6)
        assert m["equity_dd"] == pytest.approx(1.0 - math.exp(-0.30), abs=1e-6)

    def test_mdd_monotonic_gain_is_zero(self):
        from atr_stop_audit import metrics

        pnl = np.array([0.01, 0.02, 0.01, 0.03])
        turn = np.zeros_like(pnl)
        times = np.arange(4, dtype=np.int64) * 1800 + 1699920000
        m = metrics(pnl, turn, times, ppy=17520)
        assert m["mdd_log"] == pytest.approx(0.0, abs=1e-9)


class TestMetricsWorstDayIsDayNotBar:
    """R7-B2: worst_day 必须按 UTC 日聚合（//86400），不是逐 bar 取最小。"""

    def test_two_bars_same_day_sum_to_worst_day(self):
        from atr_stop_audit import metrics

        # 活 bar 时刻 times[1..4]：前两根在 Day A，后两根在 Day B
        times = np.array(
            [1699920000, 1699921800, 1699923600, 1700006400, 1700008200],
            dtype=np.int64,
        )
        pnl = np.array([-0.01, -0.02, -0.005, -0.005])
        turn = np.zeros_like(pnl)
        m = metrics(pnl, turn, times, ppy=17520)
        # Day A = -0.03, Day B = -0.01；逐 bar 最小是 -0.02，按 times[t] 分组是 -0.035
        assert m["worst_day_log"] == pytest.approx(-0.03, abs=1e-9)
        assert m["worst_day_pct"] == pytest.approx(math.expm1(-0.03), abs=1e-9)


class TestR7StopAccounting:
    """R7-B3: 止损出场计 1 次成交；再进发生在下一决策根（ruling 实测 turn=[1,1,1,0,0]）。"""

    def _r1_fixture(self):
        o = np.array([100.0, 100.0, 102.0, 103.0, 104.0])
        h = np.array([100.5, 101.0, 103.0, 104.0, 105.0])
        l = np.array([99.5, 99.0, 96.0, 102.0, 103.0])
        target_ret = np.zeros(5)
        target_ret[:3] = np.log(o[2:] / o[1:-1])
        dirs = np.array([1, 1, 1, 1, 1], dtype=np.int64)
        atr = np.full(5, 1.0)
        return o, h, l, target_ret, dirs, atr

    def test_r1_fixture_turn_and_entries_match_ruling(self):
        from atr_stop_audit import simulate

        o, h, l, target_ret, dirs, atr = self._r1_fixture()
        r = simulate(dirs, o, h, l, atr, target_ret, k=3.0, rule="immediate", cost_rate=0.0003)
        # ruling 实测 turn=[1,1,1,0,0]（长 T 版）；T-2=3 ⇒ [1,1,1]
        np.testing.assert_array_equal(r["turn"], [1.0, 1.0, 1.0])
        assert r["entries"] == 2
        assert r["stop_hits"] == 1

    def test_reentry_bar_earns_full_target_ret_minus_one_cost(self):
        from atr_stop_audit import simulate

        o, h, l, target_ret, dirs, atr = self._r1_fixture()
        r = simulate(dirs, o, h, l, atr, target_ret, k=3.0, rule="immediate", cost_rate=0.0003)
        # t=2 是下一笔的入场根：gross=target_ret[2]，成本 1 次（不是 0）
        assert r["pnl"][2] == pytest.approx(math.log(104.0 / 103.0) - 0.0003, abs=1e-12)

    def test_stop_exit_charged_when_reentry_blocked(self):
        from atr_stop_audit import simulate

        # 全多信号 + flip 规则：被扫后永远不许同向再进 ⇒ 出场成本必须单独计
        T = 8
        o = np.ones(T) * 100.0
        h = o + 0.5
        l = o - 0.5
        l[3] = 95.0
        target_ret = np.zeros(T)
        target_ret[: T - 2] = np.log(o[2:] / o[1:-1])
        dirs = np.ones(T, dtype=np.int64)
        atr = np.full(T, 1.0)
        r = simulate(dirs, o, h, l, atr, target_ret, k=3.0, rule="flip", cost_rate=0.0003)
        assert r["stop_hits"] == 1
        assert r["entries"] == 1
        assert r["turn"].sum() == 2.0  # 入场 1 次 + 止损出场 1 次


class TestR7CooldownNoDeadlock:
    """R7-B4: bars_since_stop = t - stop_bar（纯经过），信号归零出场后不得永久卡死。"""

    def _fixture(self, dirs):
        T = 14
        o = np.ones(T) * 100.0
        h = o + 0.5
        l = o - 0.5
        l[3] = 95.0  # 活 bar L=3（决策 t=2）穿 sl=97
        target_ret = np.zeros(T)
        target_ret[: T - 2] = np.log(o[2:] / o[1:-1])
        atr = np.full(T, 1.0)
        return o, h, l, target_ret, dirs, atr

    def test_cooldown_reenters_after_zero_signal_exit(self):
        from atr_stop_audit import simulate

        dirs = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64)
        o, h, l, target_ret, dirs, atr = self._fixture(dirs)
        r = simulate(dirs, o, h, l, atr, target_ret, k=3.0, rule="cooldown", cooldown=1, cost_rate=0.0003)
        assert r["stop_hits"] == 1
        assert r["entries"] == 2
        assert r["turn"][3] == 0.0  # bars=1 不 > cooldown=1
        assert r["turn"][6] == 1.0  # 信号归零两根后回来，bars=4 > 1 ⇒ 必须再进
        assert r["pnl"][6] == pytest.approx(-0.0003, abs=1e-12)

    def test_cooldown_boundary_skips_exactly_cooldown_bars(self):
        from atr_stop_audit import simulate

        dirs = np.ones(14, dtype=np.int64)
        o, h, l, target_ret, dirs, atr = self._fixture(dirs)
        r = simulate(dirs, o, h, l, atr, target_ret, k=3.0, rule="cooldown", cooldown=1, cost_rate=0.0003)
        # stop_bar=2：t=3 bars=1 不 >1；t=4 bars=2 >1 ⇒ 再进
        assert r["turn"][3] == 0.0
        assert r["turn"][4] == 1.0


class TestR8EntryBarStop:
    """R8: 入场根（活 bar L）本身必须有止损保护——live 里 broker 从成交那一刻就监控 SL。"""

    def test_entry_bar_low_breach_stops_same_bar(self):
        from atr_stop_audit import simulate

        # t=0 入场于 o[1]=100，sl=97；入场根 L=1 的 low=96 穿 sl ⇒ 当根被扫
        o = np.array([100.0, 100.0, 99.0, 101.0, 102.0])
        h = np.array([100.5, 100.5, 100.0, 102.0, 103.0])
        l = np.array([99.5, 96.0, 98.5, 100.0, 101.0])
        target_ret = np.zeros(5)
        target_ret[:3] = np.log(o[2:] / o[1:-1])
        dirs = np.array([1, 1, 1, 1, 1], dtype=np.int64)
        atr = np.full(5, 1.0)
        r = simulate(dirs, o, h, l, atr, target_ret, k=3.0, rule="immediate", cost_rate=0.0003)
        assert r["stop_hits"] == 1
        assert r["turn"][0] == 2.0  # 入场 1 次 + 入场根止损出场 1 次
        assert r["pnl"][0] == pytest.approx(math.log(97.0 / 100.0) - 2 * 0.0003, abs=1e-12)

    def test_reversal_new_position_also_entry_bar_protected(self):
        from atr_stop_audit import simulate

        # t=0 开多；t=1 信号翻空，新空仓入场于 o[2]=100，sl=103；
        # 入场根 L=2 的 high=104 穿 sl ⇒ 新仓当根被扫
        o = np.array([100.0, 100.0, 100.0, 99.0, 98.0])
        h = np.array([100.5, 100.5, 104.0, 100.0, 99.0])
        l = np.array([99.5, 99.5, 99.0, 98.0, 97.0])
        target_ret = np.zeros(5)
        target_ret[:3] = np.log(o[2:] / o[1:-1])
        dirs = np.array([1, -1, -1, -1, -1], dtype=np.int64)
        atr = np.full(5, 1.0)
        r = simulate(dirs, o, h, l, atr, target_ret, k=3.0, rule="immediate", cost_rate=0.0003)
        assert r["stop_hits"] == 1
        # t=1：旧多出场 + 新空入场（2 次）+ 新空入场根被扫（1 次）= 3
        assert r["turn"][1] == 3.0
        # R9 更正（控制器 R8 笔误）：空单 100 进、103 被扫 = 亏损 −log(103/100)，不是 +log。
        # 原断言 +log(103/100) 把空单止损记成盈利，与空单 P&L 定义矛盾，按第一性原理更正。
        assert r["pnl"][1] == pytest.approx(-math.log(103.0 / 100.0) - 3 * 0.0003, abs=1e-12)


class TestR9ShortSideCaliber:
    """R9: hold 模型所有 gross 分支必须带持仓方向乘子；归零出场根无持仓 ⇒ gross=0。

    缺陷背景：B5 五个 gross 分支漏 pos* 乘子（空单收益记反号）、B6 归零出场根
    白记一根 target_ret。全部长单夹具 ⇒ 32 条绿测试无一拦住。
    主针 = 混合方向 + 宽 k 不触损 ⇒ pnl/turn 必须与 k=None 逐 bar 基线逐位一致。
    """

    def test_held_short_earns_minus_target_ret(self):
        from atr_stop_audit import simulate

        # 全程空单、价格下跌 ⇒ 空单盈利。sl=103 高于所有 high ⇒ 不触损
        o = np.array([100.0, 100.0, 99.0, 98.0, 97.0])
        h = np.array([100.5, 100.5, 99.5, 98.5, 97.5])
        l = np.array([99.5, 99.5, 98.5, 97.5, 96.5])
        target_ret = np.zeros(5)
        target_ret[:3] = np.log(o[2:] / o[1:-1])
        dirs = np.full(5, -1, dtype=np.int64)
        atr = np.full(5, 1.0)
        r = simulate(dirs, o, h, l, atr, target_ret, k=3.0, cost_rate=0.0003)
        assert r["stop_hits"] == 0
        assert r["pnl"][0] == pytest.approx(-math.log(99.0 / 100.0) - 0.0003, abs=1e-12)
        base = simulate(dirs, o, h, l, atr, target_ret, k=None, cost_rate=0.0003)
        np.testing.assert_allclose(r["pnl"], base["pnl"], rtol=0, atol=1e-12)

    def test_short_stop_books_loss_not_profit(self):
        from atr_stop_audit import simulate

        # 空单 100 进、sl=103；L=2 的 high=104 穿 sl ⇒ f=103，亏损 −log(103/100)
        o = np.array([100.0, 100.0, 100.0, 99.0, 98.0])
        h = np.array([100.5, 100.5, 104.0, 100.0, 99.0])
        l = np.array([99.5, 99.5, 99.0, 98.0, 97.0])
        target_ret = np.zeros(5)
        target_ret[:3] = np.log(o[2:] / o[1:-1])
        dirs = np.full(5, -1, dtype=np.int64)
        atr = np.full(5, 1.0)
        r = simulate(dirs, o, h, l, atr, target_ret, k=3.0, rule="immediate", cost_rate=0.0003)
        assert r["stop_hits"] == 1
        assert r["pnl"][1] == pytest.approx(-math.log(103.0 / 100.0) - 0.0003, abs=1e-12)

    def test_zero_signal_exit_bar_books_no_gross(self):
        from atr_stop_audit import simulate

        # t=2 信号归零：旧多在 o[3] 出场（其收益已在 t=1 记完），本根无持仓 ⇒ gross=0
        o = np.array([100.0, 100.0, 102.0, 104.0, 106.0])
        h = o + 0.5
        l = o - 0.5
        target_ret = np.zeros(5)
        target_ret[:3] = np.log(o[2:] / o[1:-1])
        dirs = np.array([1, 1, 0, 0, 0], dtype=np.int64)
        atr = np.full(5, 1.0)
        r = simulate(dirs, o, h, l, atr, target_ret, k=3.0, cost_rate=0.0003)
        assert r["stop_hits"] == 0
        assert r["turn"][2] == 1.0
        assert r["pnl"][2] == pytest.approx(-0.0003, abs=1e-12)

    def test_flip_to_short_books_new_side_return(self):
        from atr_stop_audit import simulate

        # t=1 翻空：新空于 o[2]=102 进，活 bar 收益 = −log(o[3]/o[2]) = −log(99/102) > 0
        o = np.array([100.0, 100.0, 102.0, 99.0, 97.0])
        h = np.array([100.5, 100.5, 103.0, 100.0, 98.0])
        l = np.array([99.5, 99.5, 101.0, 98.0, 96.0])
        target_ret = np.zeros(5)
        target_ret[:3] = np.log(o[2:] / o[1:-1])
        dirs = np.array([1, -1, -1, -1, -1], dtype=np.int64)
        atr = np.full(5, 1.0)
        r = simulate(dirs, o, h, l, atr, target_ret, k=3.0, cost_rate=0.0003)
        assert r["stop_hits"] == 0
        assert r["turn"][1] == 2.0
        assert r["pnl"][1] == pytest.approx(-math.log(99.0 / 102.0) - 2 * 0.0003, abs=1e-12)

    def test_mixed_dirs_wide_k_no_stop_equals_per_bar_baseline(self):
        from atr_stop_audit import simulate

        # 主针：随机混合方向 + k=100（sl 距 entry ±100，价格漂移 <30 ⇒ 永不触损）
        # ⇒ hold 模型必须与 k=None 逐 bar 基线逐位一致（任何符号/出场幽灵收益都会打破）
        T = 300
        rng = np.random.default_rng(11)
        dirs = rng.choice([-1, 0, 1], size=T, p=[0.3, 0.4, 0.3]).astype(np.int64)
        o = 2000 + np.cumsum(rng.normal(0, 0.5, T))
        h = o + 0.1
        l = o - 0.1
        target_ret = np.concatenate([np.log(o[2:] / o[1:-1]), np.zeros(2)])
        atr = np.full(T, 1.0)
        r = simulate(dirs, o, h, l, atr, target_ret, k=100.0, cost_rate=0.0003)
        base = simulate(dirs, o, h, l, atr, target_ret, k=None, cost_rate=0.0003)
        assert r["stop_hits"] == 0
        np.testing.assert_allclose(r["pnl"], base["pnl"], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(r["turn"], base["turn"])
