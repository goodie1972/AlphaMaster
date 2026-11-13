"""ATR stop audit — 灾难止损审计驱动的实现层。

import r2_miner.stop_overlay 的四个纯函数：
  atr_sma / stop_price / stop_fill / reentry_allowed
复用 formula_probe 的 independent_backtest / _ann / _sharpe / _mdd / _sortino。
复用 verify_mq5_port.py 的 port_pipeline / FORMULA（self_check z_port 对拉）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from formula_probe import _ann, _mdd, _sharpe, independent_backtest, pnl_from_pos
from model_core.vm import StackVM
from r2_miner.stop_overlay import atr_sma, reentry_allowed, stop_fill, stop_price
from verify_mq5_port import FORMULA, port_pipeline


# ── ANCHORS（钉死，不许改）──────────────────────────────────────────────────
ANCHORS = {
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


# ── 私有帮助：训练链口径 ATR ───────────────────────────────────────────────
def _atr_float32(c: np.ndarray, h: np.ndarray, l: np.ndarray, w: int = 14) -> np.ndarray:
    """先降 float32 再算 atr_sma，回 float64；对齐训练链 feat_tensor 口径。"""
    return atr_sma(
        c.astype(np.float32),
        h.astype(np.float32),
        l.astype(np.float32),
        w,
    ).astype(np.float64)


# ── load_series ───────────────────────────────────────────────────────────
def load_series(parquet_path: str = "data/XAUUSD_M30.parquet") -> dict:
    """从 parquet 加载 OHLCV + target_ret + feature + atr。

    返回 dict 含 o/h/l/c/v/times/target_ret(ret_torch)/feat(attr)/atr/ppy/T/path。
    atr 走 _atr_float32 保证与训练链口径一致。
    """
    from data_pipeline.parquet_manager import ParquetDataManager

    mgr = ParquetDataManager(parquet_path)
    mgr.load()
    raw = mgr.raw_dict
    T = int(raw["open"].shape[1])
    ppy = ANCHORS["ppy"]

    o = raw["open"].numpy()[0]
    h = raw["high"].numpy()[0]
    l = raw["low"].numpy()[0]
    c = raw["close"].numpy()[0]
    v = raw["volume"].numpy()[0]
    times = raw["time"].numpy()[0].astype(np.int64)
    ret_torch = mgr.target_ret  # [1,T] float32

    # target_ret numpy 版：用于 simulate
    target_ret_np = ret_torch.numpy()[0].astype(np.float64)

    feat = mgr.feat_tensor  # [1, F, T] float32

    # ATR：必须 float32 口径
    atr = _atr_float32(c, h, l, w=14)

    return {
        "o": o,
        "h": h,
        "l": l,
        "c": c,
        "v": v,
        "times": times,
        "target_ret": target_ret_np,
        "ret_torch": ret_torch,
        "feat": feat,
        "atr": atr,
        "ppy": ppy,
        "T": T,
        "path": parquet_path,
    }


# ── binary_dirs ───────────────────────────────────────────────────────────
def binary_dirs(z: np.ndarray, min_exposure: float = 0.05) -> np.ndarray:
    pos = np.tanh(z)
    dirs = np.where(
        pos >= min_exposure,
        1,
        np.where(pos <= -min_exposure, -1, 0),
    )
    return dirs.astype(np.int64)


# ── simulate ──────────────────────────────────────────────────────────────
def simulate(
    dirs: np.ndarray,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    atr: np.ndarray,
    target_ret: np.ndarray,
    *,
    k: float | None = None,
    rule: str = "immediate",
    cooldown: int = 0,
    cost_rate: float = 0.0003,
) -> dict:
    """逐 bar 离线模拟 ATR 止损策略。

    决策时刻 t（收盘），活 bar L=t+1 进 open[L]，正常出场 open[L+1]。
    止损成交 f = stop_fill(sl, side, o[L], h[L], l[L])。
    cost base = o[L]（活 bar 开盘作基准），避免重复计账。
    turn 显式计数 += abs(new_pos - old_pos)，不用 |Δpos|。
    stop_save += -side * log(o[t+2] / f)（>=0 表示止损避免了后续不利移动）。
    SL 不追踪：entry/sl 在 hold 期间固定不变。
    causal：t+2 之后的数据不影响 [:t+1] 的输出。
    """
    T = len(o)

    # k=None：逐 bar 模型（与 pnl_from_pos 同口径，float32 精度对齐）
    if k is None:
        # 逐 bar 模型：position = dirs[t]，与 pnl_from_pos 同口径但 float64 精度
        prev = np.concatenate([[0], dirs[:-1]]).astype(np.float64)
        turn_full = np.abs(dirs.astype(np.float64) - prev)
        pnl_full = dirs.astype(np.float64) * target_ret.astype(np.float64) - turn_full * cost_rate
        return {
            "pnl": pnl_full[:T - 2].copy(),
            "turn": turn_full[:T - 2].copy(),
            "stop_hits": 0,
            "entries": int((turn_full[:T - 2] > 0).sum()),
            "stop_save": 0.0,
            "save_rate": 0.0,
            "cost_total": float(turn_full[:T - 2].sum()),
        }

    # k!=None：持仓模型（hold across bars, check stops）
    pnl = np.zeros(T - 2, dtype=np.float64)
    turn = np.zeros(T - 2, dtype=np.float64)

    pos = 0  # 当前持仓方向 (+1/-1/0)
    entry = 0.0
    sl = 0.0
    side_at_stop = 0
    stop_bar = -1  # 最近一次止损的决策索引；bars_since_stop = t - stop_bar（纯经过，不归零）
    stop_hits = 0
    stop_save = 0.0
    entries = 0

    for t in range(T - 2):
        L = t + 1  # 活 bar
        R = t + 2  # 出场 bar

        current_dir = int(dirs[t])
        gross = 0.0  # 本根 gross（不含成本），在最后统一减 turn[t]*cost_rate
        just_entered = False

        if pos != 0:
            # ── 有仓：先查止损 ──
            f = stop_fill(sl, pos, float(o[L]), float(h[L]), float(l[L]))
            if f is not None:
                # 止损成交：gross = pos*log(f / o[L])（R1：活 bar 开盘作基准，telescoping；
                # R9：必须带方向乘子，空单 f>o[L] 是亏损）
                # 止损出场本身是一次成交，计 1（R7）；再进在下一决策根，不在本根
                gross = pos * math.log(f / float(o[L]))
                stop_save += -pos * math.log(float(o[R]) / f)
                stop_hits += 1
                turn[t] += 1
                side_at_stop = pos
                pos = 0
                stop_bar = t
            elif current_dir == pos:
                # 同向持有：记录本根收益（带方向），不退出，不计 turn
                gross = pos * float(target_ret[t])
            elif current_dir == -pos:
                # 信号反转：旧仓出场 + 新仓同 o[L] 进场（2 次成交）；本根收益归新仓方向
                gross = current_dir * float(target_ret[t])
                turn[t] += 2
                entry = float(o[L])
                pos = current_dir
                sl = stop_price(entry, pos, float(atr[t]), k)
                entries += 1
                just_entered = True
            else:
                # 信号归零：仅出场（1 次成交）。旧仓收益已在 t-1 记完，本根无持仓 ⇒ gross=0
                gross = 0.0
                turn[t] += 1
                pos = 0
        elif current_dir != 0:
            # ── 无仓：再进闸门（首次开仓不受限；被扫后 bars_since_stop = t - stop_bar）──
            allowed = stop_hits == 0 or reentry_allowed(
                rule, t - stop_bar, current_dir, side_at_stop, cooldown=cooldown
            )
            if allowed:
                entry = float(o[L])
                pos = current_dir
                sl = stop_price(entry, pos, float(atr[t]), k)
                turn[t] = 1
                entries += 1
                gross = current_dir * float(target_ret[t])
                just_entered = True

        # R8：入场根本身也要查止损（live 里 broker 从成交那一刻监控 SL）
        if just_entered:
            f2 = stop_fill(sl, pos, float(o[L]), float(h[L]), float(l[L]))
            if f2 is not None:
                gross = pos * math.log(f2 / float(o[L]))
                stop_save += -pos * math.log(float(o[R]) / f2)
                stop_hits += 1
                turn[t] += 1
                side_at_stop = pos
                pos = 0
                stop_bar = t

        # pnl = gross - turn*cost_rate（turn 已全部计完）
        pnl[t] = gross - turn[t] * cost_rate


    result = {
        "pnl": pnl,
        "turn": turn,
        "hits": stop_hits,
        "entries": entries,
        "stop_save": stop_save,
        "save_rate": stop_save / stop_hits if stop_hits > 0 else 0.0,
        "cost_total": float(turn.sum()),
        "stop_hits": stop_hits,
    }
    return result


# ── metrics ───────────────────────────────────────────────────────────────
def metrics(
    pnl: np.ndarray,
    turn: np.ndarray,
    times: np.ndarray,
    ppy: int,
    *,
    cost_rate: float = 0.0003,
    stop_save: float = 0.0,
    stop_hits: int = 0,
) -> dict:
    """计算组合指标。worst_day 按活 bar 时刻 times[t+1] 的 UTC 日分组。"""
    T_eff = len(pnl)
    # 1-D：_mdd 沿 dim=0 cumsum，[1,T] 行向量会让 cumsum 成恒等变换 ⇒ 回撤恒 0
    pnl_torch = torch.tensor(pnl, dtype=torch.float32)
    ann_ret = _ann(pnl_torch, float(ppy))
    sharpe = _sharpe(pnl_torch, float(ppy))
    mdd_log = _mdd(pnl_torch)
    equity_dd = 1.0 - math.exp(-mdd_log)

    # worst_day 按活 bar 时刻 times[t+1] 的 UTC 日分组（⚠ 用 t+1 不是 t；//86400 才是日不是 bar）
    # t ∈ [0, T_eff-1] → 活 bar 索引为 t+1 ∈ [1, T_eff]
    n_day = min(T_eff, len(times) - 1) if len(times) > 1 else T_eff
    day_keys = times[1:n_day + 1] // 86400 if len(times) > 1 else np.zeros(T_eff, dtype=np.int64)
    if len(day_keys) < T_eff:
        day_keys = np.concatenate([day_keys, np.zeros(T_eff - len(day_keys), dtype=np.int64)])
    df_daily = pd.DataFrame({"day": day_keys, "pnl": pnl})
    daily_pnl = df_daily.groupby("day")["pnl"].sum()
    worst_day_log = float(daily_pnl.min()) if len(daily_pnl) > 0 else 0.0
    worst_day_pct = math.expm1(worst_day_log) if worst_day_log != 0 else 0.0

    total_cost = float((turn * cost_rate).sum())
    stop_save_avg = stop_save / stop_hits if stop_hits > 0 else 0.0

    return {
        "ann_ret": ann_ret,
        "sharpe": sharpe,
        "mdd_log": mdd_log,
        "equity_dd": equity_dd,
        "worst_day_log": worst_day_log,
        "worst_day_pct": worst_day_pct,
        "total_cost": total_cost,
        "stop_hits": stop_hits,
        "entries": int(np.count_nonzero(turn > 0)),
        "stop_save_avg": stop_save_avg,
    }


# ── self_check ────────────────────────────────────────────────────────────
def self_check(series: dict, anchors: dict | None = None) -> dict:
    failures = []
    measured = {}
    anchors_used = anchors or ANCHORS

    T = series["T"]
    ppy = series["ppy"]
    feat = series["feat"]  # [1,F,T]
    ret = series["ret_torch"]  # [1,T]

    # z_ref
    z_ref = StackVM().execute(FORMULA, feat)[0]  # [1,T]
    if z_ref is None:
        failures.append("StackVM.execute returned None")
        return {"passed": False, "failures": failures, "measured": {}, "anchors": anchors_used}

    # 全段独立回测
    full = independent_backtest(z_ref[None, :], ret, 0.0003, ppy, 0.05)
    measured["full_ann_ret"] = full["ann_ret"]
    measured["full_sharpe"] = full["sharpe"]
    measured["full_mdd"] = full["mdd"]

    # 对比锚点
    tol_r = lambda akey, t: TOL_6DP if akey in ("ann_ret", "mdd", "oos_ann_ret", "oos_mdd") else TOL_4DP
    for akey, am in [("ann_ret", "ann_ret"), ("sharpe", "sharpe"), ("mdd", "mdd"),
                      ("oos_ann_ret", "oos_ann_ret"), ("oos_sharpe", "oos_sharpe"), ("oos_mdd", "oos_mdd")]:
        av = anchors_used.get(akey)
        mv = full.get(am)
        if av is not None and mv is not None:
            diff = abs(mv - av)
            tol = TOL_6DP if akey in ("ann_ret", "mdd", "oos_ann_ret", "oos_mdd") else TOL_4DP
            if diff > tol:
                failures.append(f"{akey}: measured={mv:.8f} vs anchor={av:.6f} diff={diff:.2e} > tol={tol:.0e}")

    # T/ppy 核对
    if T != anchors_used["T"]:
        failures.append(f"T mismatch: {T} vs {anchors_used['T']}")
    # ppy 不在 full 里直接比，跳过

    # 后 1/3 段独立对比
    split = int(T * 2 / 3)
    if split < T:
        z_oos = z_ref[split:]
        ret_oos = ret[:, split:]
        oos = independent_backtest(z_oos[None, :], ret_oos, 0.0003, ppy, 0.05)
        measured["oos_ann_ret"] = oos["ann_ret"]
        measured["oos_sharpe"] = oos["sharpe"]
        measured["oos_mdd"] = oos["mdd"]

        for akey, mkey in [("oos_ann_ret", "ann_ret"), ("oos_sharpe", "sharpe"), ("oos_mdd", "mdd")]:
            av = anchors_used.get(akey)
            mv = oos.get(mkey)
            if av is not None and mv is not None:
                diff = abs(mv - av)
                tol = TOL_6DP if akey in ("oos_ann_ret", "oos_mdd") else TOL_4DP
                if diff > tol:
                    failures.append(f"{akey}: measured={mv:.8f} vs anchor={av:.6f} diff={diff:.2e} > tol={tol:.0e}")

    # cost_stress 2.0x 转亏
    cs = full.get("cost_stress", {})
    cs_2x = cs.get("2.0x", {})
    if cs_2x.get("ann_ret", 0.0) >= 0:
        failures.append(f"cost_stress 2.0x ann_ret={cs_2x.get('ann_ret')} >= 0, should be < 0")

    # z_port 对拉（bar>=500 处 max|z_port-z_ref| < 1e-3）
    o = series["o"]
    h = series["h"]
    l = series["l"]
    c = series["c"]
    v = series["v"]
    zp = port_pipeline(o, h, l, c, v)
    zr = z_ref.numpy() if hasattr(z_ref, "numpy") else np.asarray(z_ref)
    mask = np.arange(len(zp)) >= 500
    max_diff = float(np.abs(zp[mask] - zr[mask]).max())
    measured["z_port_max_diff"] = max_diff
    if max_diff >= 1e-3:
        failures.append(f"z_port vs z_ref max diff at bar>=500: {max_diff:.6f} >= 1e-3")

    passed = len(failures) == 0
    return {
        "passed": passed,
        "failures": failures,
        "measured": measured,
        "anchors": anchors_used,
    }


# ── sweep ─────────────────────────────────────────────────────────────────
def sweep(
    series: dict,
    *,
    ks=(2, 3, 4, 6, 8, 12),
    rules=(("flip", 0), ("immediate", 0), ("cooldown", 1), ("cooldown", 4)),
    cost_mults=(1.0, 2.0),
) -> list[dict]:
    """48 种组合（6×4×2）全扫描，每行 delta 对二值无止损基线。"""
    o = series["o"]
    h = series["h"]
    l = series["l"]
    atr = series["atr"]
    target_ret = series["target_ret"]
    times = series["times"]
    # 计算 z 信号：优先用 series["z"]（smoke 模式），否则走 StackVM
    if "z" in series:
        z = series["z"]
    else:
        z = StackVM().execute(FORMULA, series["feat"])[0]
    dirs = binary_dirs(z.numpy() if hasattr(z, 'numpy') else z)
    ppy = series["ppy"]

    # 先跑基线 k=None
    base = simulate(dirs, o, h, l, atr, target_ret, k=None, cost_rate=0.0003)
    base_metrics = metrics(base["pnl"], base["turn"], times, ppy, cost_rate=0.0003)

    rows = []
    for k in ks:
        for rule_name, cooldown_val in rules:
            for cm in cost_mults:
                cr = 0.0003 * cm
                r = simulate(dirs, o, h, l, atr, target_ret, k=k, rule=rule_name,
                             cooldown=cooldown_val, cost_rate=cr)
                m = metrics(r["pnl"], r["turn"], times, ppy, cost_rate=cr)
                row = {
                    "k": k,
                    "rule": rule_name,
                    "cooldown": cooldown_val,
                    "cost_mult": cm,
                    "ann_ret": m["ann_ret"],
                    "sharpe": m["sharpe"],
                    "mdd_log": m["mdd_log"],
                    "equity_dd": m["equity_dd"],
                    "worst_day_log": m["worst_day_log"],
                    "worst_day_pct": m["worst_day_pct"],
                    "total_cost": m["total_cost"],
                    "entries": r["entries"],
                    "stop_hits": r["stop_hits"],
                    "stop_save_avg": r["save_rate"],
                    # delta vs baseline
                    "delta_ann_ret": m["ann_ret"] - base_metrics["ann_ret"],
                    "delta_equity_dd": m["equity_dd"] - base_metrics["equity_dd"],
                    "delta_worst_day_pct": m["worst_day_pct"] - base_metrics["worst_day_pct"],
                }
                rows.append(row)
    return rows


# ── CLI ───────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ATR stop audit CLI")
    parser.add_argument("--parquet", default="data/XAUUSD_M30.parquet")
    parser.add_argument("--self-check-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-dir", default="backtest_output")
    args = parser.parse_args(argv)

    if args.smoke:
        return _cli_smoke(args)

    series = load_series(args.parquet)

    if args.self_check_only:
        sc = self_check(series)
        print(json.dumps(sc, indent=2, ensure_ascii=False))
        if not sc["passed"]:
            print("self_check FAIL:")
            for f in sc["failures"]:
                print(f"  - {f}")
            sys.exit(2)
        return 0

    sc = self_check(series)
    if not sc["passed"]:
        print("self_check FAIL:")
        for f in sc["failures"]:
            print(f"  - {f}")
        sys.exit(2)

    rows = sweep(series)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    symbol = Path(args.parquet).stem.replace(".parquet", "")
    csv_path = out / f"atr_stop_sweep_{symbol}.csv"
    json_path = out / f"atr_stop_sweep_{symbol}.json"

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    payload = {
        "anchors": ANCHORS,
        "self_check": sc,
        "sweep": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote {csv_path} ({len(rows)} rows)")
    print(f"Wrote {json_path}")
    return 0


def _cli_smoke(args: argparse.Namespace) -> int:
    """合成数据 smoke 测试，证明链路通。"""
    rng = np.random.default_rng(123)
    T = 300
    o = 2000 + np.cumsum(rng.normal(0, 0.5, T))
    h = o + rng.uniform(0.1, 1.0, T)
    l = o - rng.uniform(0.1, 1.0, T)
    c = o + rng.uniform(0.1, 1.0, T)
    v = np.ones(T) * 100.0
    times = np.arange(T, dtype=np.int64) * 1800
    target_ret = np.log(o[2:] / o[1:-1])
    target_ret = np.concatenate([target_ret, np.zeros(2)])
    atr = np.full(T, 2.0)
    feat = torch.randn(1, 10, T, dtype=torch.float32)

    series = {
        "o": o, "h": h, "l": l, "c": c, "v": v,
        "times": times, "target_ret": target_ret,
        "ret_torch": torch.tensor(target_ret[None, :], dtype=torch.float32),
        "feat": feat, "atr": atr, "z": rng.standard_normal(T).astype(np.float32),
        "ppy": 17520, "T": T, "path": "<smoke>",
    }

    rows = sweep(series)
    print(f"Smoke sweep: {len(rows)} rows")
    print(f"First row ann_ret={rows[0]['ann_ret']:.6f}, sharpe={rows[0]['sharpe']:.4f}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "atr_stop_sweep_smoke.csv"
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
