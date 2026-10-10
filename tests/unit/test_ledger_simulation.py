# tests/unit/test_ledger_simulation.py
"""模拟测试：纸面交易全生命周期端到端（合成行情 + 真实计算链 + 真实台账 IO/API）。

约定（同 test_ledger_api.py / test_ledger_adversarial.py）：
- TestClient(app) 不进上下文管理器 —— @app.on_event("startup") 绝不触发。
- 一切 IO 锁进 tmp_path：autouse fixture monkeypatch web.live_ledger.LEDGER_DIR；
  数据源一律 fake（autouse 拦掉 LS.get_source 真实入口，逐用例再换成合成序列）。
  真实 live_ledger/、端口 8765、web_settings.json 绝不触碰。
- 计算链不 mock：live 仓位由真实 evaluate_signal 在 bars[k-1000:k+1] 切片上逐根
  评估（与 realtime_manager._evaluate_task 每 tick 喂入的切片形状一致）。
- 规模说明：brief 中「bar 100..149 / 1200 根重算 800..899」在 warm-up 预检下
  物理不可行（recompute_theo 要求窗口首根在 fetch 切片内 ≥ _min_bars()=800 根
  历史，见 ledger_stats.py:295-303）——凡涉及 theo 重算的场景，账本窗口一律放在
  合成序列末端，使 live 切片与 theo 切片同段同形，恒等式才有意义。
  1h 窗口 × ~0.3s/评估太贵，涉及 API 重算的场景用 1d 周期压缩懒算成本（S1 保留
  1h+days=1 的真实窗口形状）。
"""
from __future__ import annotations

import json
import math

import pytest
from fastapi.testclient import TestClient

import web.ledger_stats as LS
import web.live_ledger as LL
from web.app import app
from web.data_sources.base import bars_to_raw_dict
from strategy_manager.live_signal import evaluate_signal

client = TestClient(app)

# F1 = strategies/best_XAUUSD.json 真实公式；F2 = 同组合末位 MIN(115) 换成 MAX(116)
# （两者均为词表内真实 token，实测 state=ok 且仓位不同）。
F1 = [25, 81, 61, 103, 93, 104, 31, 115]
F2 = [25, 81, 61, 103, 93, 104, 31, 116]
SHA1, SHA2 = LL.formula_sha(F1), LL.formula_sha(F2)
# autouse 把 web.app.load_settings 钉成 0.02/0.01 → cost_rate=(0.02+0.01)/100
COST = 0.0003


class _B:
    def __init__(s, ts, c):
        s.ts = ts
        s.open = s.high = s.low = s.close = c
        s.volume = 1.0


class _FakeSource:
    def __init__(s, bars):
        s.bars = bars
        s.fetches = 0

    def fetch_bars(s, symbol, tf, n, drop_forming=True):
        s.fetches += 1
        return s.bars[-n:]


class _BoomSource:
    def __init__(s, msg):
        s.msg = msg

    def fetch_bars(s, *a, **k):
        raise RuntimeError(s.msg)


def _series(count, step, drift=0.02):
    """合成升序序列（sine + 线性 drift），保证仓位非退化且 close>0。"""
    return [_B(i * step, 100.0 + math.sin(i / 5.0) + drift * i) for i in range(count)]


def _eval_live(formula, bars, k):
    """realtime_manager._evaluate_task 同形状切片：bars[k-1000:k+1] 末位评估。"""
    sl = bars[max(0, k - LS._THEO_LOOKBACK):k + 1]
    res = evaluate_signal(formula, bars_to_raw_dict(sl), require_closed=False)
    assert res["state"] == "ok", res
    return res


def _append_eval_record(ledger_dir, formula, bars, k, *, source, symbol, tf):
    """按 _evaluate_task 的字段映射写一条 live 台账记录。"""
    res = _eval_live(formula, bars, k)
    rec = LL.LedgerRecord(
        ts=float(bars[k].ts), bar_ts=int(bars[k].ts), source=source, symbol=symbol,
        timeframe=tf, strategy_sha=LL.formula_sha(formula),
        factor_value=float(res["factor_value"]), position=float(res["position"]),
        direction=str(res["direction"]), strength=float(res["strength"]),
        close=float(bars[k].close))
    assert LL.append_record(rec, ledger_dir=ledger_dir)


def _hand_rec(bar_ts, pos, close, sha, *, source="fake", symbol="BTCUSDT", tf="1d"):
    return LL.LedgerRecord(ts=float(bar_ts), bar_ts=int(bar_ts), source=source,
                           symbol=symbol, timeframe=tf, strategy_sha=sha,
                           factor_value=float(pos), position=float(pos),
                           direction="LONG" if pos > 0 else "SHORT",
                           strength=abs(float(pos)), close=float(close))


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    monkeypatch.setattr("web.app.load_settings",
                        lambda: {"bt_commission_pct": 0.02, "bt_slippage_pct": 0.01})

    def _boom(kind):
        raise AssertionError(f"测试禁止真实数据源: {kind}")
    monkeypatch.setattr(LS, "get_source", _boom)


# ── S1 双策略同对互不干扰 ─────────────────────────────────────────────
# 账本内 live 记录/统计按 sha 隔离正确；theo 缓存现已按「账本文件 + sha」键控
# （ledger_stats._theo_path_for → {stem}.{sha}.theo.jsonl）：默认段（F2）重算写
# 缓存后，sha=F1 的 curve/summary 使用 F1 自己的缓存，同输入恒等式不被跨策略污染。
def test_s1_two_formulas_interleaved_same_ledger(tmp_path, monkeypatch):
    bars = _series(1050, 3600)                    # 0..1049，账本窗口贴末端（见模块 docstring）
    monkeypatch.setattr(LS, "get_source", lambda kind: _FakeSource(bars))
    for k in range(1025, 1050):                   # 同 bar 交错：先 F1 后 F2
        _append_eval_record(tmp_path, F1, bars, k, source="fake", symbol="BTCUSDT", tf="1h")
        _append_eval_record(tmp_path, F2, bars, k, source="fake", symbol="BTCUSDT", tf="1h")
    recs, corrupt = LL.read_records(LL.ledger_path("fake", "BTCUSDT", "1h", ledger_dir=tmp_path))
    assert corrupt == 0 and len(recs) == 50 and recs[-1].strategy_sha == SHA2

    f1 = [r for r in recs if r.strategy_sha == SHA1]
    # 直链恒等式（缓存写进隔离子目录，不污染 API 缓存）：同源同链 → 零分叉
    theo1, why = LS.recompute_theo("fake", "BTCUSDT", "1h", F1, start_ts=1045 * 3600,
                                   end_ts=1049 * 3600, cost_rate=COST,
                                   ledger_dir=tmp_path / "direct")
    assert why is None and len(theo1) == 5
    live1 = LS.derive_rows([r for r in f1 if r.bar_ts >= 1045 * 3600], tf="1h", cost_rate=COST)
    assert LS.align_divergence(live1, theo1) == []

    LL.ensure_strategy_index(F1, "best_XAUUSD_1h.json", "v1", ledger_dir=tmp_path)
    LL.ensure_strategy_index(F2, "best_XAUUSD_M30.json", "v1", ledger_dir=tmp_path)
    base = {"watch_id": "fake_BTCUSDT_1h", "days": 1}
    d = client.get("/api/ledger/summary", params=base).json()
    assert d["sha"] == SHA2 and d["records"] == 25 and d["theo_available"] is True
    assert d["divergence_count"] == 0             # F2 live vs F2 theo：同源零分叉
    s1 = client.get("/api/ledger/summary", params={**base, "sha": SHA1}).json()
    assert s1["sha"] == SHA1 and s1["records"] == 25
    man = LS.compute_stats(LS.derive_rows(f1, tf="1h", cost_rate=COST), COST, tf="1h")
    assert s1["win_rate"] == man["win_rate"] and s1["live_bars"] == man["live_bars"]
    page = client.get("/api/ledger/records",
                      params={"watch_id": "fake_BTCUSDT_1h", "limit": 500}).json()
    assert page["total"] == 50                    # 明细不按 sha 过滤：交错流水完整可见
    assert page["records"][0]["bar_ts"] == page["records"][1]["bar_ts"] == 1049 * 3600
    assert {page["records"][0]["strategy_sha"], page["records"][1]["strategy_sha"]} == {SHA1, SHA2}

    # API 侧 sha=F1 的理论曲线必须也是 F1 的（同链恒等 → cum_theo == cum_live）
    c = client.get("/api/ledger/curve", params={**base, "sha": SHA1}).json()
    assert all(p["cum_theo"] is not None for p in c["curve"])
    assert max(abs(p["cum_theo"] - p["cum_live"]) for p in c["curve"]) < 1e-9
    assert s1["divergence_count"] == 0


# ── S2 重启恢复与断档 ────────────────────────────────────────────────
def test_s2_restart_anchor_and_gap_completeness(tmp_path, monkeypatch):
    import web.realtime_manager as RM
    day = 86400
    for d in range(1000, 1009):                   # F1 写到 1008d
        assert LL.append_record(_hand_rec(d * day, 0.5, 100.0 + d % 3, SHA1,
                                          source="tradingview"), ledger_dir=tmp_path)
    for d in range(1000, 1011):                   # F2 写到 1010d → 文件末行是 F2
        assert LL.append_record(_hand_rec(d * day, -0.5, 100.0 + d % 3, SHA2,
                                          source="tradingview"), ledger_dir=tmp_path)
    path = LL.ledger_path("tradingview", "BTCUSDT", "1d", ledger_dir=tmp_path)
    assert LL.last_bar_ts(path) == 1010 * day     # 全文件末位（对 F1 是错误锚点）

    sf = tmp_path / "btc_1d.json"
    sf.write_text(json.dumps({"formula": F1, "vocab_version": "v1"}), encoding="utf-8")

    class _TVSrc:
        label = "TV"

        def supported_timeframes(self):
            return ["1d"]

    monkeypatch.setattr(RM, "get_source", lambda kind: _TVSrc())
    task = RM.RealtimeManager()._add_task_internal(
        "tradingview", "BTCUSDT", "1d", str(sf), persist=False)   # 模拟进程重启
    assert task.last_ledger_bar_ts == 1008 * day  # per-sha 锚点 = F1 自己的末 bar

    for d in range(1009, 1019):                   # 恢复：>锚点才补；1011..1015 停机断档
        if d * day > task.last_ledger_bar_ts and d not in range(1011, 1016):
            assert LL.append_record(_hand_rec(d * day, 0.5, 100.0 + d % 3, SHA1,
                                              source="tradingview"), ledger_dir=tmp_path)
    recs, _ = LL.read_records(path)
    pairs = [(r.bar_ts, r.strategy_sha) for r in recs]
    assert len(pairs) == len(set(pairs))          # 断点续写不产生同 (bar_ts, sha) 重复

    monkeypatch.setattr(LS, "get_source", lambda kind: _FakeSource(_series(1020, day)))
    LL.ensure_strategy_index(F1, "s1.json", "v1", ledger_dir=tmp_path)
    LL.ensure_strategy_index(F2, "s2.json", "v1", ledger_dir=tmp_path)
    s = client.get("/api/ledger/summary",
                   params={"watch_id": "tradingview_BTCUSDT_1d", "days": 10}).json()
    assert s["sha"] == SHA1 and s["records"] == 6 and s["theo_available"] is True
    assert s["theo_bars"] == 11                   # 源有全部 11 根收盘 bar，账本只有 6
    assert s["completeness"] == pytest.approx(6 / 11) and s["completeness"] < 1.0
    rows = LS.derive_rows([r for r in recs if r.strategy_sha == SHA1
                           and r.bar_ts >= 1008 * day], tf="1d", cost_rate=COST)
    assert {r["bar_ts"] for r in rows if r["missing"]} == {1010 * day, 1018 * day}


# ── S3 理论=实盘恒等式跨全链 ─────────────────────────────────────────
def test_s3_identity_live_vs_theo_every_bar(tmp_path, monkeypatch):
    step = 3600
    bars = _series(1200, step)                    # 账本放 1170..1199：切片形状与 live 全等
    monkeypatch.setattr(LS, "get_source", lambda kind: _FakeSource(bars))
    for k in range(1170, 1200):
        _append_eval_record(tmp_path, F1, bars, k, source="fake", symbol="BTCUSDT", tf="1h")
    recs, corrupt = LL.read_records(LL.ledger_path("fake", "BTCUSDT", "1h", ledger_dir=tmp_path))
    assert corrupt == 0 and len(recs) == 30
    assert len({r.position for r in recs}) > 5    # 非退化仓位序列

    cost = 0.001
    theo, why = LS.recompute_theo("fake", "BTCUSDT", "1h", F1, start_ts=1170 * step,
                                  end_ts=1199 * step, cost_rate=cost, ledger_dir=tmp_path)
    assert why is None and len(theo) == 30
    by_ts = {r["bar_ts"]: r for r in theo}
    for r in recs:                                # 恒等式：逐根 |live - theo| < 1e-6
        assert abs(float(r.position) - float(by_ts[int(r.bar_ts)]["position"])) < 1e-6
    live_rows = LS.derive_rows(recs, tf="1h", cost_rate=cost)
    for a, b in zip(live_rows, theo):             # pnl 口径两侧全等（同一 derive_rows 数学）
        assert a["bar_ts"] == b["bar_ts"] and a["missing"] == b["missing"]
        if a["pnl"] is not None:
            assert a["pnl"] == pytest.approx(b["pnl"], abs=1e-12)

    ledger = LL.ledger_path("fake", "BTCUSDT", "1h", ledger_dir=tmp_path)
    cache = LS._theo_path_for(ledger, SHA1)         # sha 键控缓存（DEFECT-3 修复后）
    assert cache.exists() and LS.theo_is_fresh(ledger, cache, cost) is True
    assert LS.theo_is_fresh(ledger, cache, 0.002) is False      # 成本口径变化即失效
    cached = LS.load_theo_cache(cache)
    assert len(cached) == 30 and set(cached[0]) == {"bar_ts", "position", "close", "cost_rate"}
    for c0, t0 in zip(cached, theo):              # 缓存往返 == 返回行
        assert (c0["bar_ts"], c0["position"]) == (t0["bar_ts"], t0["position"])
        assert c0["close"] == pytest.approx(t0["close"]) and c0["cost_rate"] == cost


# ── S4 源故障降级故事 ────────────────────────────────────────────────
def test_s4_source_outage_degrades_then_recovers(tmp_path, monkeypatch):
    day = 86400
    for i, d in enumerate((1016, 1017, 1018)):
        assert LL.append_record(_hand_rec(d * day, 0.6, 101.0 + i, SHA1, tf="1d"),
                                ledger_dir=tmp_path)
    LL.ensure_strategy_index(F1, "s.json", "v1", ledger_dir=tmp_path)
    monkeypatch.setattr(LS, "get_source", lambda kind: _BoomSource("连接超时"))
    base = {"watch_id": "fake_BTCUSDT_1d", "days": 1}

    r1 = client.get("/api/ledger/summary", params=base)
    assert r1.status_code == 200                  # 源故障绝不 500
    p1 = r1.json()
    assert p1["theo_available"] is False and "连接超时" in p1["theo_reason"]
    assert p1["completeness"] is None and p1["records"] == 2 and p1["win_rate"] == 1.0
    r1c = client.get("/api/ledger/curve", params=base)
    assert r1c.status_code == 200 and r1c.json()["theo_available"] is False
    assert all(p["theo_missing"] and p["cum_theo"] is None for p in r1c.json()["curve"])

    monkeypatch.setattr(LS, "get_source", lambda kind: _FakeSource(_series(1020, day)))
    r2 = client.get("/api/ledger/summary", params=base)
    assert r2.status_code == 200
    p2 = r2.json()
    assert p2["theo_available"] is True and p2["theo_reason"] is None
    assert p2["completeness"] == pytest.approx(1.0) and p2["theo_bars"] == 2
    assert set(p1) == set(p2)                     # 两阶段字段集稳定
    assert (tmp_path / f"fake_BTCUSDT_1d.{SHA1}.theo.jsonl").exists()


# ── S5 成本口径联动 ──────────────────────────────────────────────────
def test_s5_cost_settings_flip_pf_and_invalidate_cache(tmp_path, monkeypatch):
    day = 86400
    src = _FakeSource(_series(1020, day))
    monkeypatch.setattr(LS, "get_source", lambda kind: src)
    # 行A pos=1.0 ret≈+0.1% 换手0.15 → pnl≈0.001-0.15c（低成本盈、高成本亏）；
    # 行B pos=0.85 ret≈-0.1% 换手0.15 → 恒亏。c=0.0003 时 PF>1，c=0.01 时全亏 PF=0。
    spec = [(1015, 1.0, 100.0), (1016, 0.85, 100.1), (1017, 1.0, 99.9999)]
    for d, pos, close in spec:
        assert LL.append_record(_hand_rec(d * day, pos, close, SHA1, tf="1d"),
                                ledger_dir=tmp_path)
    LL.ensure_strategy_index(F1, "s.json", "v1", ledger_dir=tmp_path)
    base = {"watch_id": "fake_BTCUSDT_1d", "days": 3}

    r_a, r_b = 100.1 / 100.0 - 1.0, 99.9999 / 100.1 - 1.0
    a_lo = 1.0 * r_a - 0.15 * 0.0003          # c=(0.02+0.01)/100=0.0003 → 行A 为正
    b_lo = 0.85 * r_b - 0.15 * 0.0003         # 行B 恒亏
    assert a_lo > 0.0 and 1.0 * r_a - 0.15 * 0.01 < 0.0   # 0.5/0.5 口径下 A 由盈转亏
    s_low = client.get("/api/ledger/summary", params=base).json()
    assert s_low["profit_factor"] == pytest.approx(a_lo / abs(b_lo))
    assert s_low["profit_factor"] > 1.0 and s_low["win_rate"] == 0.5
    cached = LS.load_theo_cache(tmp_path / f"fake_BTCUSDT_1d.{SHA1}.theo.jsonl")
    assert cached and cached[0]["cost_rate"] == 0.0003

    monkeypatch.setattr("web.app.load_settings",
                        lambda: {"bt_commission_pct": 0.5, "bt_slippage_pct": 0.5})
    s_hi = client.get("/api/ledger/summary", params=base).json()    # c=0.01 → 行A 由盈转亏
    assert s_hi["profit_factor"] == 0.0           # 全亏 → PF 翻转归零（无盈利样本）
    assert s_hi["win_rate"] == 0.5                # 胜率纯方向（不含成本）→ 不随成本变
    cached = LS.load_theo_cache(tmp_path / f"fake_BTCUSDT_1d.{SHA1}.theo.jsonl")
    assert cached[0]["cost_rate"] == 0.01         # 旧缓存被按新口径重算并覆写
    assert src.fetches == 2                       # cost_rate 不一致 → 恰好触发一次重算
