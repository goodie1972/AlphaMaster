"""A1 stop-overlay primitives: caliber pins against the training-chain ATR."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from model_core.features import MT5FeatureEngineer
from r2_miner.stop_overlay import atr_sma, reentry_allowed, stop_fill, stop_price


def _synth_ohlc_f32(T: int = 64, seed: int = 12345):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.01, T).cumsum()
    close = (1.0 + steps).astype(np.float32)
    high = (close + rng.uniform(0.001, 0.02, T)).astype(np.float32)
    low = (close - rng.uniform(0.001, 0.02, T)).astype(np.float32)
    return close, high, low


# ── atr_sma ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("w", [1, 5, 14])
def test_atr_sma_matches_torch_bitcaliber_including_warmup(w):
    close, high, low = _synth_ohlc_f32()
    ct = torch.from_numpy(close[None, :])
    ht = torch.from_numpy(high[None, :])
    lt = torch.from_numpy(low[None, :])
    ref = MT5FeatureEngineer._atr(ct, ht, lt, w).numpy()[0]
    got = atr_sma(close, high, low, w)
    assert got.shape == ref.shape == (close.shape[0],)
    assert got.dtype == close.dtype
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, ref, atol=1e-6, rtol=0)


def test_atr_sma_matches_torch_bitwise_at_xauusd_magnitude_and_length():
    # A7（cumsum 退回输入 dtype）在 T=64/base=1.0 夹具上只差 5.6e-09，atol=1e-6 看不见；
    # 在真量级长序列上差 3.4e-02 ⇒ 这根针用 array_equal 把它钉死。
    T, base, w = 200000, 4000.0, 14
    r = np.random.default_rng(5)
    close = (base * np.exp(r.normal(0, 0.001, T).cumsum())).astype(np.float32)
    high = (close + r.uniform(0.05, 3.0, T)).astype(np.float32)
    low = (close - r.uniform(0.05, 3.0, T)).astype(np.float32)
    ref = MT5FeatureEngineer._atr(
        torch.from_numpy(close[None, :]), torch.from_numpy(high[None, :]),
        torch.from_numpy(low[None, :]), w).numpy()[0]
    got = atr_sma(close, high, low, w)
    assert got.dtype == np.float32
    assert np.array_equal(got, ref)


def test_atr_sma_output_dtype_follows_input_dtype_float64():
    close, high, low = _synth_ohlc_f32()
    c64, h64, l64 = close.astype(np.float64), high.astype(np.float64), low.astype(np.float64)
    ref = MT5FeatureEngineer._atr(
        torch.from_numpy(c64[None, :]), torch.from_numpy(h64[None, :]),
        torch.from_numpy(l64[None, :]), 14).numpy()[0]
    got = atr_sma(c64, h64, l64, 14)
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=0)


def test_atr_sma_warmup_is_zero_pad_partial_mean_not_nan_not_wilder():
    # TR forced to [1,2,3,4,5]; w=3 zero-pad mean = [1/3, 1, 2, 3, 4].
    close = np.zeros(5, dtype=np.float64)
    high = np.arange(1.0, 6.0, dtype=np.float64)
    low = np.zeros(5, dtype=np.float64)
    got = atr_sma(close, high, low, 3)
    expected = np.array([1.0 / 3.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(got, expected, atol=1e-12, rtol=0)
    assert np.all(np.isfinite(got))
    assert got[0] != pytest.approx(1.0)  # not Wilder seed / not full-window


def test_atr_sma_first_value_equals_tr0_over_w():
    close, high, low = _synth_ohlc_f32()
    w = 14
    pc0 = close[0]
    tr0 = max(high[0] - low[0], abs(high[0] - pc0), abs(low[0] - pc0))
    got = atr_sma(close, high, low, w)
    assert got[0] == pytest.approx(float(tr0) / w, abs=1e-6)


def test_atr_sma_constant_price_all_zero():
    for dtype in (np.float32, np.float64):
        c = np.full(40, 100.0, dtype=dtype)
        got = atr_sma(c, c, c, 14)
        assert np.all(got == 0.0)
        assert np.all(np.isfinite(got))


def test_atr_sma_w1_equals_true_range():
    close, high, low = _synth_ohlc_f32()
    pc = np.concatenate([close[:1], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))
    got = atr_sma(close, high, low, 1)
    np.testing.assert_allclose(got, tr, atol=1e-6, rtol=0)


def test_atr_sma_short_series_below_window():
    close = np.zeros(2, dtype=np.float64)
    high = np.array([2.0, 4.0])
    low = np.zeros(2, dtype=np.float64)
    got = atr_sma(close, high, low, 14)
    np.testing.assert_allclose(got, np.array([2.0 / 14.0, 6.0 / 14.0]), atol=1e-12, rtol=0)


# ── stop_price ─────────────────────────────────────────────────────────────

def test_stop_price_long():
    assert stop_price(100.0, 1, 2.0, 1.5) == pytest.approx(97.0)


def test_stop_price_short():
    assert stop_price(100.0, -1, 2.0, 1.5) == pytest.approx(103.0)


@pytest.mark.parametrize("side,k,atr", [
    (0, 1.5, 2.0),    # no position
    (1, 0.0, 2.0),    # k == 0
    (1, -1.0, 2.0),   # k < 0
    (1, 1.5, 0.0),    # atr == 0
    (1, 1.5, -2.0),   # atr < 0
    (-1, 0.0, 2.0),   # short, k == 0
    (-1, 1.5, 0.0),   # short, atr == 0
])
def test_stop_price_invalid_raises(side, k, atr):
    with pytest.raises(ValueError):
        stop_price(100.0, side, atr, k)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_stop_price_rejects_non_finite_atr(bad):
    # NaN 放行会让 stop_fill 永远返回 None ⇒ 灾难止损静默失效，表上显示「加止损毫无损失」
    with pytest.raises(ValueError):
        stop_price(4000.0, 1, bad, 3)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_stop_price_rejects_non_finite_entry(bad):
    with pytest.raises(ValueError):
        stop_price(bad, 1, 8.0, 3)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_stop_price_rejects_non_finite_k(bad):
    with pytest.raises(ValueError):
        stop_price(4000.0, 1, 8.0, bad)


# ── stop_fill ──────────────────────────────────────────────────────────────

def test_stop_fill_long_no_hit():
    assert stop_fill(97.0, 1, 100.0, 105.0, 98.0) is None


def test_stop_fill_short_no_hit():
    assert stop_fill(103.0, -1, 100.0, 102.0, 95.0) is None


def test_stop_fill_long_inbar_returns_sl():
    assert stop_fill(97.0, 1, 100.0, 105.0, 96.0) == pytest.approx(97.0)


def test_stop_fill_short_inbar_returns_sl():
    assert stop_fill(103.0, -1, 100.0, 104.0, 95.0) == pytest.approx(103.0)


def test_stop_fill_long_gap_returns_open_not_sl():
    # open already below sl AND low below sl -> gap wins, fill at open.
    assert stop_fill(97.0, 1, 95.0, 99.0, 94.0) == pytest.approx(95.0)


def test_stop_fill_short_gap_returns_open_not_sl():
    assert stop_fill(103.0, -1, 105.0, 106.0, 100.0) == pytest.approx(105.0)


def test_stop_fill_long_exact_low_equals_sl_is_hit():
    assert stop_fill(97.0, 1, 100.0, 105.0, 97.0) == pytest.approx(97.0)


def test_stop_fill_short_exact_high_equals_sl_is_hit():
    assert stop_fill(103.0, -1, 100.0, 103.0, 95.0) == pytest.approx(103.0)


def test_stop_fill_long_exact_open_equals_sl_is_hit_not_none():
    assert stop_fill(97.0, 1, 97.0, 100.0, 96.0) == pytest.approx(97.0)


def test_stop_fill_gap_priority_long_both_conditions():
    # open<=sl and low<=sl simultaneously; must return open (worse fill), not sl.
    assert stop_fill(97.0, 1, 90.0, 100.0, 89.0) == pytest.approx(90.0)


def test_stop_fill_gap_priority_short_both_conditions():
    assert stop_fill(103.0, -1, 110.0, 111.0, 100.0) == pytest.approx(110.0)


def test_stop_fill_rejects_side_zero():
    # 0 = 无仓位，不该有止损成交价；简报外自加的防御，报告声称过就要钉住
    with pytest.raises(ValueError):
        stop_fill(97.0, 0, 100.0, 101.0, 96.0)


# ── reentry_allowed ────────────────────────────────────────────────────────

def test_reentry_flip_same_dir_false():
    assert reentry_allowed("flip", 5, 1, 1) is False
    assert reentry_allowed("flip", 5, -1, -1) is False


def test_reentry_flip_reverse_true():
    assert reentry_allowed("flip", 1, -1, 1) is True
    assert reentry_allowed("flip", 1, 1, -1) is True


def test_reentry_flip_dir_now_zero_false():
    assert reentry_allowed("flip", 3, 0, 1) is False
    assert reentry_allowed("flip", 3, 0, -1) is False
    assert reentry_allowed("flip", 3, 0, 0) is False


@pytest.mark.parametrize("bars,dir_now,dir_at_stop", [
    (0, 0, 0), (0, 1, 1), (1, -1, 1), (99, 0, -1),
])
def test_reentry_immediate_always_true(bars, dir_now, dir_at_stop):
    assert reentry_allowed("immediate", bars, dir_now, dir_at_stop) is True


@pytest.mark.parametrize("bars,expected", [
    (0, False), (2, False), (3, False), (4, True), (10, True),
])
def test_reentry_cooldown_boundary(bars, expected):
    assert reentry_allowed("cooldown", bars, 1, 1, cooldown=3) is expected


def test_reentry_cooldown_zero_default():
    assert reentry_allowed("cooldown", 0, 1, 1) is False
    assert reentry_allowed("cooldown", 1, 1, 1) is True


def test_reentry_cooldown_is_keyword_only():
    with pytest.raises(TypeError):
        reentry_allowed("cooldown", 5, 1, 1, 3)


@pytest.mark.parametrize("rule", ["bogus", "", "FLIP", "cooldown_n"])
def test_reentry_unknown_rule_raises(rule):
    with pytest.raises(ValueError):
        reentry_allowed(rule, 5, 1, 1)


def test_reentry_allowed_returns_python_bool_for_numpy_scalars():
    # A2 逐 bar 循环里传的是 numpy 标量；报告声称 bool() 包装保证返回 Python bool
    assert reentry_allowed("flip", np.int64(0), np.int64(1), np.int64(-1)) is True
    assert reentry_allowed("flip", np.int64(0), np.int64(1), np.int64(1)) is False
    assert reentry_allowed("cooldown", np.int64(0), np.int64(1), np.int64(1), cooldown=np.int64(4)) is False
    assert reentry_allowed("cooldown", np.int64(5), np.int64(1), np.int64(1), cooldown=np.int64(4)) is True