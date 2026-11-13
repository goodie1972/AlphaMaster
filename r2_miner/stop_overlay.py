"""r2_miner/stop_overlay.py — 灾难止损（broker 侧 ATR SL）的离线原语层。

纯函数、无状态、无数组级循环（逐 bar 循环归 atr_stop_audit.py / A2）。

口径契约（A2/A4 消费前必读）：
- ATR 口径 = SMA + 零填充 w-1 warmup，不是 Wilder；逐行对齐 model_core/features.py:35,57
  的训练链口径 ⇒ MQL5 侧禁用 iATR（那是 Wilder 平滑）。
- warmup 首 w-1 根是「含 0 的偏小均值」，不是 NaN、不是完整窗口。
- 调用方必须显式 .astype(np.float32)：parquet 列是 float64，直接喂会落到另一套口径，
  真数据实测差 2.08e-4（相对 3e-4），k=3 时约 0.062 个 XAUUSD 点。
- 「与 torch 逐位相同」的适用范围：XAUUSD 量级（base≈2000–4000）、float32 OHLC、w≤200
  时逐位相同（真数据 234823 根 × w∈{1,5,14,20,50} 全部 0.0/0 不等，控制器已独立复核）；
  其余情形 ≤1–2 ULP（相对 ≤3e-7）——实测 base=1.0 宽振幅分叉 1.79e-7、
  XAUUSD-like 但 w=2000 分叉 9.54e-7。机理：torch 的 float32 mean 在 float32 里累加；
  真 OHLC 的 TR 是价格 ULP 的整数倍且尾数很短，w·TR ≲ 2·price 时窗口和精确，
  两条路径才撞到同一个 float32。这是数据性质，不是结构保证。
- atr_sma 只吃 1-D；w 会被 int() 截断（w=14.9 等价 w=14），调用方必须传 int。
- NaN 传播：前缀和一旦被 NaN 污染就污染此后全部（torch 的 unfold-mean 只污染 w 根），
  这是前缀和结构的必然，本层不修；护栏在 stop_price（非有限 entry/atr/k → ValueError），
  因为 NaN 放行会让 stop_fill 永远返回 None ⇒ 灾难止损静默失效。
- stop_fill 单侧假设：只加 SL 不加 TP ⇒ 无「先止盈还是先止损」歧义 ⇒ 不需 M1 路径；跳空优先。
"""
import numpy as np


def atr_sma(close: np.ndarray, high: np.ndarray, low: np.ndarray, w: int = 14) -> np.ndarray:
    close = np.asarray(close)
    high = np.asarray(high)
    low = np.asarray(low)
    w = int(w)
    pc = np.concatenate([close[:1], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))
    # 零填充 w-1 复刻训练链路 _rolling_mean：warmup 是「含 0 的偏小均值」，非 NaN/完整窗口/Wilder。
    padded = np.concatenate([np.zeros(w - 1, dtype=tr.dtype), tr])
    cs = np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(padded, dtype=np.float64)])
    T = tr.shape[0]
    sums = cs[w:w + T] - cs[:T]
    out_dtype = tr.dtype if np.issubdtype(tr.dtype, np.floating) else np.float64
    return (sums / w).astype(out_dtype)


def stop_price(entry: float, side: int, atr: float, k: float) -> float:
    if not (np.isfinite(k) and np.isfinite(atr) and np.isfinite(entry)):
        raise ValueError("entry/atr/k must be finite (NaN/Inf would silently disable the stop)")
    if k <= 0 or atr <= 0:
        raise ValueError("k and atr must be > 0 (zero/negative distance is immediate exit, not a stop)")
    if side == 1:
        return entry - k * atr
    if side == -1:
        return entry + k * atr
    raise ValueError("side must be +1 or -1 (0 = no position, no stop)")


def stop_fill(sl: float, side: int, bar_open: float, bar_high: float, bar_low: float) -> float | None:
    """只加 SL 不加 TP ⇒ 单侧出场，bar 内无「先止盈还是先止损」歧义，故不需 M1 路径；Low/High 破位即命中。
    跳空优先：开盘已越过 sl 时只能成交在更差的 open，不可能成交在 sl。"""
    if side == 1:
        if bar_open <= sl:
            return bar_open
        if bar_low <= sl:
            return sl
        return None
    if side == -1:
        if bar_open >= sl:
            return bar_open
        if bar_high >= sl:
            return sl
        return None
    raise ValueError("side must be +1 or -1")


def reentry_allowed(rule: str, bars_since_stop: int, dir_now: int, dir_at_stop: int, *, cooldown: int = 0) -> bool:
    if rule == "flip":
        return bool(dir_now == -dir_at_stop and dir_now != 0)
    if rule == "immediate":
        return True
    if rule == "cooldown":
        return bool(bars_since_stop > cooldown)
    raise ValueError(f"unknown reentry rule: {rule!r}")