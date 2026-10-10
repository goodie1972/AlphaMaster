"""实盘台账统计纯函数（spec §5.5）。

derive_rows：把 LedgerRecord 序列按相邻性规则展开为逐 bar 收益行。
compute_stats：对行序列聚合胜率/盈亏比/年化/Sortino/MDD 等统计。

约束：纯函数、无 IO；不 import torch（Sortino 公式镜像
model_core/backtest.py MT5Backtest._sortino，用纯 Python population std 实现）。
周期字符串只做内部查表归一，绝不改写传出的 tf 值（各域原样约束）。
"""
from __future__ import annotations

import math

from strategy_manager.live_signal import min_exposure

# realtime 域记法（数字在前）；"1M"=月 与 "1m"=分钟 精确区分，不做整体 lower()。
_TF_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800,
    "1M": 2678400,  # 近似 31 天
}
# parquet 域记法（字母在前，大写 H1/M15 等）：单位字母内部 lower 查表，仅此用途。
_UNIT_SECONDS: dict[str, int] = {"m": 60, "h": 3600, "d": 86400, "w": 604800}

_SORTINO_CLIP = 20.0   # 同 backtest._SORTINO_CLIP
_EPS = 1e-8             # 同 backtest _sortino(eps=1e-8)
_SECONDS_PER_YEAR = 365.25 * 86400.0


def _tf_seconds(tf: str) -> int:
    """周期字符串 → bar 秒数。未知周期 raise ValueError。"""
    exact = _TF_SECONDS.get(tf)
    if exact is not None:
        return exact
    # parquet 域 "H1"/"M15"/"D1"/"W1"：字母+数字；数字部分 <1（H0/M0）拒绝，防 tf_s=0
    if len(tf) >= 2 and tf[0].isalpha() and tf[1:].isdigit():
        mult = int(tf[1:])
        unit = _UNIT_SECONDS.get(tf[0].lower())
        if unit is not None and mult >= 1:
            return unit * mult
    raise ValueError(f"未知周期: {tf!r}")


def _density_ppy(rows, tf_s: int) -> float:
    """年化因子：镜像 estimate_periods_per_year（model_core/backtest.py:33-56）口径。

    ppy = 行数 ÷ ((末 bar_ts − 首 bar_ts) 秒 / 年秒数)，用账本行自身时间戳估计
    每年 bar 密度；全部行参与（含 missing——它们同样是真实观测）。
    回测数据只含交易时段 bar，密度口径自动适应外汇/股票闭市空档
    （日历口径对 FX H1 会虚高：8766 vs 实际 ~6240）。
    行数 <2、span<=0 或 ppy 异常（<=0/非有限）→ 回退日历 365.25*86400/tf_s，
    对应 backtest 估计失败时的默认值回退（此处回退到该 tf 的日历 ppy）。
    """
    bar_tss = [int(r["bar_ts"]) for r in rows if r.get("bar_ts") is not None]
    if len(bar_tss) >= 2:
        span = float(max(bar_tss) - min(bar_tss))
        if span > 0:
            ppy = len(bar_tss) / (span / _SECONDS_PER_YEAR)
            if ppy > 0 and math.isfinite(ppy):
                return ppy
    return _SECONDS_PER_YEAR / tf_s


def derive_rows(records, *, tf: str, cost_rate: float) -> list[dict]:
    """按 spec §5.5 相邻性规则展开逐 bar 收益行。

    行字段：{bar_ts, close, position, ret, turnover, pnl, missing}。
    行 i 配对记录 i+1：仅当 bar_ts[i+1]-bar_ts[i]==tf_seconds 时
    ret=close_next/close-1、turnover=|pos_next-pos|、pnl=pos*ret-turnover*cost；
    否则（含末条无 next）missing=True，ret/turnover/pnl 均 None。
    close<=0 或非有限视为不可计算，同样 missing=True（防除零）。
    """
    tf_s = _tf_seconds(tf)
    recs = sorted(records, key=lambda r: int(r.bar_ts))
    n = len(recs)
    rows: list[dict] = []
    for i, cur in enumerate(recs):
        row = {"bar_ts": int(cur.bar_ts), "close": cur.close,
               "position": cur.position, "ret": None, "turnover": None,
               "pnl": None, "missing": True}
        if i + 1 < n:
            nxt = recs[i + 1]
            if int(nxt.bar_ts) - int(cur.bar_ts) == tf_s:
                c0, c1 = float(cur.close), float(nxt.close)
                if c0 > 0.0 and math.isfinite(c0) and math.isfinite(c1):
                    pos = float(cur.position)
                    ret = c1 / c0 - 1.0
                    turnover = abs(float(nxt.position) - pos)
                    row.update(ret=ret, turnover=turnover,
                               pnl=pos * ret - turnover * cost_rate,
                               missing=False)
        rows.append(row)
    return rows


def _pop_std(vals: list[float]) -> float:
    """population std（对齐 torch std(unbiased=False)），空序列返回 0.0。"""
    n = len(vals)
    if n == 0:
        return 0.0
    mean = sum(vals) / n
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / n)


def _sortino(pnls: list[float], ppy: float) -> float:
    """镜像 model_core/backtest.py MT5Backtest._sortino：
    mean / max(下行std, 0.2×全std地板, eps) × sqrt(ppy)，clip ±20。
    全序列单值时 std=0 → clamp 到 eps，与 torch 行为一致。
    """
    mean = sum(pnls) / len(pnls)
    downside = [v for v in pnls if v < 0]
    raw_std = _pop_std(downside)                     # 空下行 → 0.0
    full_std = max(_pop_std(pnls), _EPS)             # clamp(min=eps)
    floor = max(full_std * 0.2, _EPS)                # P0b 地板：防稀疏刷分
    downside_std = max(raw_std, floor)
    val = mean / downside_std * math.sqrt(ppy)
    return max(-_SORTINO_CLIP, min(_SORTINO_CLIP, val))


def _max_drawdown(pnls: list[float]) -> float:
    """累计 pnl（log 口径）曲线的最大回撤，正数 float。
    对齐 torch 侧 cum=cumsum、peak=cummax(cum)、mdd=max(peak-cum)。
    """
    cum = 0.0
    peak = None
    mdd = 0.0
    for v in pnls:
        cum += v
        if peak is None or cum > peak:
            peak = cum
        mdd = max(mdd, peak - cum)
    return mdd


def compute_stats(rows, cost_rate, *, tf: str,
                  first_ts=None, last_ts=None) -> dict:
    """聚合统计（spec §5.5）。

    cost_rate 保留在签名中（retained per API contract, B5 passes it）；
    本函数内 pnl 已在 derive_rows 净额化，参数在函数体中不再使用。统计口径：
    - win_rate/profit_factor：非 missing 且 |position| >= min_exposure() 的行；
      胜率纯方向（pos*ret>0，不含成本）；盈亏比 = 盈利均值 ÷ |亏损均值|（含成本），
      pnl==0 的行被两组均值同时排除（严格 >0 / <0）；
      无亏损行 → None，无盈利行（但有亏损）→ 0.0。
    - annualized/sortino/max_drawdown：全部非 missing 行的 pnl（不过滤低敞口）。
    - 年化因子（annualized 与 sortino 共用）为密度口径，镜像
      model_core/backtest.py 的 estimate_periods_per_year：ppy = 行数 ÷
      (首尾 bar_ts 跨度年数)，含 missing 行；估计失败回退日历 365.25*86400/tf_s。
    - live_bars=行数；theoretical_bars=[first_ts,last_ts] 内应有 bar 数，
      ts 缺省时用非 missing 行 bar_ts 的 min/max；不可确定 → None。
      完整率（记录数÷源 bar 数）由上层 API 计算（源 bar 数可能不可得）。
    """
    tf_s = _tf_seconds(tf)          # tf_s 仅用于 theoretical_bars
    ppy = _density_ppy(rows, tf_s)  # 年化因子：密度口径，镜像 backtest 估计

    valid = [r for r in rows if not r["missing"] and r["pnl"] is not None]
    min_exp = min_exposure()
    qual = [r for r in valid if abs(float(r["position"])) >= min_exp]

    win_rate = None
    if qual:
        wins = sum(1 for r in qual if float(r["position"]) * float(r["ret"]) > 0)
        win_rate = wins / len(qual)

    profits = [float(r["pnl"]) for r in qual if r["pnl"] > 0]
    losses = [float(r["pnl"]) for r in qual if r["pnl"] < 0]
    if not losses:
        profit_factor = None            # 无亏损样本：盈亏比无定义
    elif not profits:
        profit_factor = 0.0             # 只亏不赚
    else:
        profit_factor = (sum(profits) / len(profits)) / abs(sum(losses) / len(losses))

    pnls = [float(r["pnl"]) for r in valid]
    if pnls:
        annualized = (sum(pnls) / len(pnls)) * ppy
        sortino = _sortino(pnls, ppy)
        max_drawdown = _max_drawdown(pnls)
    else:
        annualized = sortino = max_drawdown = None

    if first_ts is None or last_ts is None:
        bar_ts = [int(r["bar_ts"]) for r in valid]
        if bar_ts:
            if first_ts is None:
                first_ts = min(bar_ts)
            if last_ts is None:
                last_ts = max(bar_ts)
    theoretical_bars = None
    if first_ts is not None and last_ts is not None and int(last_ts) >= int(first_ts):
        theoretical_bars = (int(last_ts) - int(first_ts)) // tf_s + 1

    return {"win_rate": win_rate, "profit_factor": profit_factor,
            "annualized": annualized, "sortino": sortino,
            "max_drawdown": max_drawdown, "live_bars": len(rows),
            "theoretical_bars": theoretical_bars}
