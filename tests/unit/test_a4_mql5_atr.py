"""A4 — 把 mt5/AlphaMaster_XAUUSD.mq5 里 FeatureAtrSma 的 MQL5 逻辑逐行 transliterate 成
Python mql5_atr_sma(h,l,c,w)，并在真 data/XAUUSD_M30.parquet 上与独立 oracle
（sliding_window_view 开窗均值）对拉。只测这段 ATR 算法，不碰 .mq5 文件本身。

口径契约（A4 §2 / A1 stop_overlay）：
- ATR = TR 的 w 期**简单均值** + **零填充 warmup**（分母恒为 w，缺的根按 0 计）。
- 禁 iATR（Wilder 平滑），与训练链路口径不同。
- mql5_atr_sma 用显式双层循环、float64、升序累加、分母恒为 w —— 逐行对应 brief 的 MQL5 片段。
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
W = 14
PARQUET = ROOT / "data" / "XAUUSD_M30.parquet"


def mql5_atr_sma(h, l, c, w):
    """逐行 transliterate mt5/AlphaMaster_XAUUSD.mq5 的 FeatureAtrSma：
    显式双层循环、float64、升序累加、分母恒为 w（零填充 warmup）。
    """
    n = len(h)
    atr = [0.0] * n
    if n <= 0 or w <= 0:
        return np.array(atr, dtype=np.float64)
    tr = [0.0] * n
    for i in range(n):
        pc = c[0] if i == 0 else c[i - 1]
        a = h[i] - l[i]
        b = abs(h[i] - pc)
        d = abs(l[i] - pc)
        tr[i] = max(a, max(b, d))
    for i in range(n):
        s = 0.0
        frm = i - w + 1
        if frm < 0:
            frm = 0
        for j in range(frm, i + 1):
            s += tr[j]
        atr[i] = s / w
    return np.array(atr, dtype=np.float64)


def _load_tail(n):
    """真 parquet 的末 n 根已收盘 bar（float64 OHLC），与 EA LoadClosedBars 同序（时间升序）。"""
    df = pd.read_parquet(PARQUET)
    c = df["close"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    s = slice(len(c) - n, len(c))
    return c[s], h[s], l[s], df


def _oracle(c, h, l, w=W):
    """独立 oracle：零填充 + sliding_window_view 开窗均值。
    与 mql5_atr_sma 是**不同算法**（向量化开窗 vs 显式双循环），真独立复核。
    """
    pc = np.concatenate([c[:1], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    padded = np.concatenate([np.zeros(w - 1, dtype=tr.dtype), tr])
    win = np.lib.stride_tricks.sliding_window_view(padded, w)  # shape (T, w)
    return win.mean(axis=1, dtype=np.float64).astype(tr.dtype)


@pytest.mark.parametrize("n", [2000, 800])
def test_mql5_atr_sma_matches_independent_oracle_on_real_parquet(n):
    c, h, l, _ = _load_tail(n)
    got = mql5_atr_sma(h, l, c, W)
    ref = _oracle(c, h, l, W)
    assert got.shape == ref.shape == (n,)
    assert got.dtype == np.float64
    diff = np.abs(got - ref)
    print(f"n={n}  max|mql5_atr_sma - oracle| = {diff.max():.10e}")
    assert diff.max() < 1e-8, f"n={n} max|Δ|={diff.max():.10e} >= 1e-8"


def test_mql5_atr_sma_hand_fixture_zero_pad_warmup():
    # TR 强制为 [1,2,3,4,5]；w=3 零填充均值 = [1/3, 1, 2, 3, 4]（不是 Wilder 种子 / 不是完整窗口）
    h = np.arange(1.0, 6.0, dtype=np.float64)
    l = np.zeros(5, dtype=np.float64)
    c = np.zeros(5, dtype=np.float64)
    got = mql5_atr_sma(h, l, c, 3)
    want = np.array([1.0 / 3.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(got, want, atol=1e-12, rtol=0)
    assert got[0] != pytest.approx(1.0)  # 不是 Wilder 种子 / 不是完整窗口


def test_mql5_atr_sma_last_10_anchor_values_print():
    # A4 联调对拉锚点：末 10 根 ATR，double 全精度（与控制器 ctrl_a4_mql5_atr.txt 一致）。
    n = 2000
    c, h, l, df = _load_tail(n)
    got = mql5_atr_sma(h, l, c, W)
    T = len(df)
    print("== 末 10 根 ATR（double 全精度，A4 对拉锚点）==")
    for i in range(n - 10, n):
        print(f"    idx={T - n + i}  time={int(df['time'].iloc[T - n + i])}  "
              f"close={c[i]:.4f}  atr={got[i]:.12f}")
    # 锚点硬钉：末根 ATR（double 全精度，与控制器 ctrl_a4_mql5_atr.txt 复算一致）。
    # 末 10 根（idx 234813..234822）ATR 逐值见上面打印，逐位与控制器 oracle 相同。
    assert got[-1] == pytest.approx(6.240000000000107, abs=1e-12)
    assert got[n - 2] == pytest.approx(6.608571428571, abs=1e-12)   # idx 234821
    assert got[n - 3] == pytest.approx(6.512857142857, abs=1e-12)   # idx 234820
    assert got[n - 6] == pytest.approx(7.480000000000, abs=1e-12)   # idx 234817