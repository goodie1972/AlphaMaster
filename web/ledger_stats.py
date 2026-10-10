"""实盘台账统计纯函数（spec §5.5）+ 理论曲线懒算（spec §5.4）。

derive_rows：把 LedgerRecord 序列按相邻性规则展开为逐 bar 收益行。
compute_stats：对行序列聚合胜率/盈亏比/年化/Sortino/MDD 等统计。
recompute_theo：同段离线重算理论仓位序列并经 derive_rows 展开（复用同一套
收益数学）；theo_is_fresh/load_theo_cache 管理 .theo.jsonl 派生缓存。

性能告诫（spec §5.4 决策⑥）：整段重算是 O(窗口 bar 数 × _THEO_LOOKBACK bar
VM 运行) 的懒算，缓存按账本 mtime 失效 —— 调用方（B5 API 层）必须限制请求
时段跨度；本模块不做增量优化。

约束：统计部分纯函数、无 IO；theo 部分做数据源拉取与缓存 IO，但任何失败
都只回 reason 字符串、绝不抛出（台账失败不得中断信号评估主链路的延伸原则）。
不 import torch 本体（evaluate_signal/bars_to_raw_dict 的 torch 依赖是既有链路）。
周期字符串只做内部查表归一，绝不改写传出的 tf 值（各域原样约束）。
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import web.live_ledger as LL
from strategy_manager.live_signal import evaluate_signal, min_exposure, _min_bars
from web.data_sources.base import bars_to_raw_dict
from web.data_sources.factory import get_source  # noqa: F401 —— 测试 monkeypatch 本模块属性

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
    position 为 None（理论重算中 state!="ok" 的 bar）→ 该行 missing=True，
    绝不参与 ret/pnl 数学。
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
            if (int(nxt.bar_ts) - int(cur.bar_ts) == tf_s
                    and cur.position is not None and nxt.position is not None):
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


# ---------------------------------------------------------------------------
# 理论曲线懒算 + 缓存（spec §5.4）
# ---------------------------------------------------------------------------

# 理论重算的单 bar 评估回看上限 = 1000，取自 realtime_manager._N_BARS 的同值口径：
# 特征 warm-up 200（_NORM_WINDOW）+ VM 滚动归一化窗口 500 + 余量（实盘评估拉 1000 根、
# evaluate_signal 要求 >=800 根才出 state=ok）。短窗口直接评估会因 warm-up 不足失真，
# 故重算窗口前必须先多带 _THEO_LOOKBACK 根历史。
_THEO_LOOKBACK = 1000

# live/theo 仓位分叉判定的最小差（浮点噪声容差）。
_DIVERGENCE_EPS = 1e-6


def _theo_path_for(ledger: Path) -> Path:
    """账本路径 → 理论曲线缓存路径 `{stem}.theo.jsonl`（派生物，可删）。"""
    p = Path(ledger)
    return p.with_name(p.stem + ".theo.jsonl")


def _write_theo_cache(path: Path, rows: list[dict], cost_rate: float) -> None:
    """原子写缓存（tmp+os.replace）；每行 {bar_ts, position, close, cost_rate}。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"bar_ts": r["bar_ts"], "position": r["position"],
                                    "close": r["close"], "cost_rate": cost_rate},
                                   ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def recompute_theo(source, symbol, tf, formula, *,
                   start_ts, end_ts, cost_rate,
                   ledger_dir=None) -> tuple[list[dict], str | None]:
    """离线重算 [start_ts, end_ts] 的理论仓位序列（spec §5.4）。

    一次 fetch `span_bars + _THEO_LOOKBACK + 2` 根历史（保证窗口首根也有满
    LOOKBACK  warm-up），对窗口内每根 bar 在切片 bars[k-LOOKBACK:k+1] 上调用
    真实 evaluate_signal（与实盘同一计算链 → 口径恒等式成立）；state!="ok"
    的行 position=None。产出先合成 LedgerRecord，再原样过 derive_rows ——
    收益数学与实盘账本共用一份实现，绝不重复。

    返回 (rows, None)；rows 结构同 derive_rows。任何失败（源不可用/拉取异常/
    未知周期等）返回 ([], reason 字符串)，绝不抛出、绝不写缓存。源静默截断
    历史（首根窗口切片 < live_signal._min_bars()）同样按失败拒绝，防止退化
    缓存污染新鲜度判定。成功后写
    `{stem}.theo.jsonl` 原子缓存；缓存写失败仅降级为「下次重算」，不影响返回。

    性能：O(窗口 bar 数 × LOOKBACK bar VM)，调用方（B5）必须限制请求跨度。
    """
    try:
        tf_s = _tf_seconds(tf)
        formula = [int(t) for t in formula]
        start_ts, end_ts = int(start_ts), int(end_ts)
        if end_ts < start_ts:
            raise ValueError("end_ts 早于 start_ts")
        src = get_source(str(source))
        need = (end_ts - start_ts) // tf_s + _THEO_LOOKBACK + 2
        bars = src.fetch_bars(str(symbol), tf, need, drop_forming=True)
        if not bars:
            raise RuntimeError("未获取到 K 线")
        # 数据源可能静默截断响应（如 okx_source 上限 300 根）：fetch 数 < need 时
        # 窗口首根切片不足 _min_bars() 根 warm-up，evaluate_signal 对每根都回
        # state!="ok" → 会写出全 position None 的退化缓存并被 theo_is_fresh 永久
        # 当作新鲜。故先校验首根切片长度，不足则拒绝重算且绝不写缓存（口径同
        # 下方逐根切片公式 bars[max(0,idx-LOOKBACK):idx+1]）。
        first_idx = next((i for i, b in enumerate(bars)
                          if start_ts <= int(b.ts) <= end_ts), None)
        if first_idx is not None:
            warm = min(first_idx + 1, _THEO_LOOKBACK + 1)
            required = _min_bars()
            if warm < required:
                return [], (f"数据源历史 bar 不足（{warm}/{required}）："
                            f"窗口首根可用切片仅 {warm} 根，无法 warm-up，"
                            f"已放弃重算且不写缓存")
        sha = LL.formula_sha(formula)
        recs: list[LL.LedgerRecord] = []
        for idx, b in enumerate(bars):
            ts = int(b.ts)
            if ts < start_ts or ts > end_ts:
                continue
            window = bars[max(0, idx - _THEO_LOOKBACK): idx + 1]
            res = evaluate_signal(formula, bars_to_raw_dict(window),
                                  require_closed=False)
            ok = res.get("state") == "ok"
            recs.append(LL.LedgerRecord(
                ts=float(ts), bar_ts=ts, source=str(source), symbol=str(symbol),
                timeframe=tf, strategy_sha=sha,
                factor_value=res.get("factor_value") if ok else None,
                position=res.get("position") if ok else None,
                direction=res.get("direction", "") if ok else "",
                strength=res.get("strength", 0.0) if ok else 0.0,
                close=float(b.close)))
        if not recs:
            raise RuntimeError(f"窗口 {start_ts}..{end_ts} 内无可用 bar")
        rows = derive_rows(recs, tf=tf, cost_rate=cost_rate)
        lp = LL.ledger_path(source, symbol, tf, ledger_dir=ledger_dir)
        try:
            _write_theo_cache(_theo_path_for(lp), rows, cost_rate)
        except OSError:
            pass  # 缓存是派生物：写失败不回退结果，下次按 mtime 判定重算
        return rows, None
    except Exception as exc:  # noqa: BLE001 —— spec §6：任何故障只回原因
        return [], f"数据源拉取失败: {exc}"


def load_theo_cache(path) -> list[dict] | None:
    """容错读取理论缓存：文件缺失/任一行损坏 → None（调用方触发重算）。"""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    rows: list[dict] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            return None
        if not isinstance(obj, dict) or "bar_ts" not in obj:
            return None
        rows.append(obj)
    return rows


def theo_is_fresh(ledger_path, theo_path, cost_rate) -> bool:
    """缓存新鲜判定：theo mtime >= ledger mtime 且首行 cost_rate 与请求一致。

    跨度覆盖检查归调用方（B5）；账本不存在时 mtime 视为 0（theo 在即新鲜）。
    """
    lp, tp = Path(ledger_path), Path(theo_path)
    try:
        theo_mtime = tp.stat().st_mtime
        ledger_mtime = lp.stat().st_mtime if lp.exists() else 0.0
    except OSError:
        return False
    if theo_mtime < ledger_mtime:
        return False
    rows = load_theo_cache(tp)
    if not rows:
        return False
    try:
        return float(rows[0].get("cost_rate")) == float(cost_rate)
    except (TypeError, ValueError):
        return False


def align_divergence(live_rows, theo_rows) -> list[int]:
    """按 bar_ts 对齐 live/theo 行序列，返回仓位分叉（|Δposition| > 1e-6）的 bar_ts。

    任一侧 position 为 None → 不参与分叉；单边缺失的 bar_ts 是数据缺口
    （由完整率/theo_missing 处理），不算分叉。
    """
    theo_by_ts: dict[int, object] = {}
    for r in theo_rows or []:
        if r.get("bar_ts") is not None:
            theo_by_ts[int(r["bar_ts"])] = r.get("position")
    out: list[int] = []
    for r in live_rows or []:
        if r.get("bar_ts") is None:
            continue
        ts = int(r["bar_ts"])
        lp = r.get("position")
        tp = theo_by_ts.get(ts)
        if lp is None or ts not in theo_by_ts or tp is None:
            continue
        if abs(float(lp) - float(tp)) > _DIVERGENCE_EPS:
            out.append(ts)
    return sorted(out)
