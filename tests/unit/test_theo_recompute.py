# tests/unit/test_theo_recompute.py
"""Task B4：理论曲线懒算重算 + mtime 缓存 + 口径恒等式（spec §5.4 / §8.3）。

恒等式核心：同一合成序列，「逐根 evaluate_signal 记账」（live 风格切片评估）
与「整段重算 recompute_theo」的 position 逐 bar 全等 —— 只 fake 数据源，
计算链走真实 evaluate_signal（不 mock 任何计算）。
"""
import math
import os
import time

import web.ledger_stats as LS
from web.data_sources.base import bars_to_raw_dict
from strategy_manager.live_signal import evaluate_signal

# 真实已验证公式：strategies/best_XAUUSD.json 的 "formula" 字段，
# 全部 token 在 model_core.vocab 词表内；合成 sine bar（1001 根，>=800 warm-up）
# 上 evaluate_signal 实测 state=="ok"、position 非退化值。
FORMULA = [25, 81, 61, 103, 93, 104, 31, 115]


class _B:
    def __init__(s, ts, c):
        s.ts = ts
        s.open = s.high = s.low = s.close = c
        s.volume = 1.0


class _FakeSource:
    def __init__(s, bars):
        s.bars = bars

    def fetch_bars(s, symbol, tf, n, drop_forming=True):
        return s.bars[-n:]


class _BoomSource:
    def fetch_bars(self, *a, **k):
        raise RuntimeError("boom")


def _sine_bars(count):
    return [_B(i * 3600, 100.0 + math.sin(i / 5)) for i in range(count)]


def test_identity_recompute_matches_live_formula(tmp_path, monkeypatch):
    bars = _sine_bars(1001)
    monkeypatch.setattr("web.ledger_stats.get_source", lambda kind: _FakeSource(bars))
    theo, why = LS.recompute_theo("fake", "BTCUSDT", "1h", FORMULA,
                                  start_ts=995 * 3600, end_ts=1000 * 3600,
                                  cost_rate=0.001, ledger_dir=tmp_path)
    assert why is None and theo
    assert len(theo) == 6
    # 行结构与 derive_rows 完全一致（复用同一数学，不另起口径）
    assert set(theo[0]) == {"bar_ts", "close", "position", "ret", "turnover", "pnl", "missing"}

    # 末位重算 position == live 逐根评估（同一 LOOKBACK 切片）。
    # 说明：brief 原文 bars[-5:] 改为完整切片 —— 真实 evaluate_signal 要求
    # >=800 bar warm-up（5 bar 会 state=insufficient、无 position 键），恒等式
    # 只能钉在「live 评估实际喂入的同一根切片」上，这才是 spec §8.3 的本意。
    live_slice = bars[max(0, 1000 - LS._THEO_LOOKBACK):1001]
    live = evaluate_signal(FORMULA, bars_to_raw_dict(live_slice), require_closed=False)
    assert live["state"] == "ok"
    assert abs(theo[-1]["position"] - live["position"]) < 1e-9

    # 更强恒等式：模拟 live 风格逐根 append 循环 —— 对窗口内多个末位下标 k，
    # 在切片 [k-LOOKBACK, k] 上评估，与该 bar_ts 的 theo position 全等。
    for k in (996, 998, 1000):
        sl = bars[max(0, k - LS._THEO_LOOKBACK):k + 1]
        r = evaluate_signal(FORMULA, bars_to_raw_dict(sl), require_closed=False)
        assert r["state"] == "ok"
        row = next(x for x in theo if x["bar_ts"] == k * 3600)
        assert abs(row["position"] - r["position"]) < 1e-9


def test_state_not_ok_yields_none_position(tmp_path, monkeypatch):
    # 300 根 < live_signal 最低 800 根：全部 state=insufficient → position None，
    # 重算本身成功（不 raise），且 derive_rows 遇 None position 必须安全（missing）。
    bars = _sine_bars(300)
    monkeypatch.setattr("web.ledger_stats.get_source", lambda kind: _FakeSource(bars))
    theo, why = LS.recompute_theo("fake", "BTCUSDT", "1h", FORMULA,
                                  start_ts=290 * 3600, end_ts=299 * 3600,
                                  cost_rate=0.001, ledger_dir=tmp_path)
    assert why is None
    assert len(theo) == 10
    assert all(r["position"] is None for r in theo)
    assert all(r["missing"] is True and r["pnl"] is None for r in theo)


def _row(bar_ts, pos):
    return {"bar_ts": bar_ts, "close": 100.0, "position": pos, "ret": 0.0,
            "turnover": 0.0, "pnl": 0.0, "missing": False}


def test_align_divergence_injects_and_detects():
    live = [_row(0, 0.5), _row(3600, -0.4), _row(7200, 0.2), _row(14400, 0.3), _row(18000, None)]
    theo = [_row(0, 0.5), _row(3600, -0.4 + 0.1), _row(7200, 0.2 + 5e-7),
            _row(10800, 0.9), _row(18000, 0.4)]
    # 3600：注入 ±0.1 分叉 → 命中；7200：差 5e-7 <= 1e-6 → 不算；
    # 10800/14400 单边缺失 = 缺口非分叉；18000 live 位 None → 跳过。
    assert LS.align_divergence(live, theo) == [3600]


def test_theo_cache_roundtrip_and_freshness(tmp_path, monkeypatch):
    ledger = tmp_path / "fake_BTCUSDT_1h.jsonl"
    ledger.write_text('{"bar_ts": 1}\n', encoding="utf-8")
    old = time.time() - 3600.0
    os.utime(ledger, (old, old))  # 账本 mtime 定在 1 小时前

    bars = _sine_bars(300)        # 轻量重算（全部 insufficient，不影响缓存往返）
    monkeypatch.setattr("web.ledger_stats.get_source", lambda kind: _FakeSource(bars))
    theo, why = LS.recompute_theo("fake", "BTCUSDT", "1h", FORMULA,
                                  start_ts=290 * 3600, end_ts=299 * 3600,
                                  cost_rate=0.001, ledger_dir=tmp_path)
    assert why is None

    theo_path = tmp_path / "fake_BTCUSDT_1h.theo.jsonl"  # {stem}.theo.jsonl
    assert theo_path.exists()
    assert LS.theo_is_fresh(ledger, theo_path, 0.001) is True
    assert LS.theo_is_fresh(ledger, theo_path, 0.002) is False  # 成本口径变了 → 失效

    rows = LS.load_theo_cache(theo_path)
    assert rows is not None and len(rows) == 10
    assert set(rows[0]) == {"bar_ts", "position", "close", "cost_rate"}
    assert all(r["cost_rate"] == 0.001 for r in rows)

    newer = time.time() + 60.0
    os.utime(ledger, (newer, newer))  # 账本更新 → 缓存过期
    assert LS.theo_is_fresh(ledger, theo_path, 0.001) is False


def test_load_theo_cache_tolerant(tmp_path):
    p = tmp_path / "x.theo.jsonl"
    assert LS.load_theo_cache(p) is None                       # 缺失 → None
    p.write_text('{"bar_ts": 1}\nnot-json\n', encoding="utf-8")
    assert LS.load_theo_cache(p) is None                       # 半行损坏 → None
    p.write_text('{"bar_ts": 1, "position": 0.5, "close": 100.0, "cost_rate": 0.001}\n',
                 encoding="utf-8")
    assert LS.load_theo_cache(p) == [{"bar_ts": 1, "position": 0.5,
                                      "close": 100.0, "cost_rate": 0.001}]
    # theo 文件不存在 → 永不 fresh
    missing = tmp_path / "missing.theo.jsonl"
    assert LS.theo_is_fresh(tmp_path / "ledger.jsonl", missing, 0.001) is False
    ledger2 = tmp_path / "ledger.jsonl"
    ledger2.write_text("", encoding="utf-8")
    # 账本 mtime 更新（os.utime 拉开严格时差，免文件系统粒度平局）→ 过期
    os.utime(ledger2, (p.stat().st_mtime + 10, p.stat().st_mtime + 10))
    assert LS.theo_is_fresh(ledger2, p, 0.001) is False
    # 账本 mtime 更早 → 新鲜；cost_rate 不一致 → 过期
    os.utime(ledger2, (p.stat().st_mtime - 10, p.stat().st_mtime - 10))
    assert LS.theo_is_fresh(ledger2, p, 0.001) is True
    assert LS.theo_is_fresh(ledger2, p, 0.002) is False


def test_recompute_failure_returns_reason_and_no_cache(tmp_path, monkeypatch):
    theo_path = tmp_path / "fake_BTCUSDT_1h.theo.jsonl"

    def boom(kind):
        raise RuntimeError("源不可用")

    monkeypatch.setattr("web.ledger_stats.get_source", boom)
    rows, why = LS.recompute_theo("fake", "BTCUSDT", "1h", FORMULA,
                                  start_ts=0, end_ts=5 * 3600,
                                  cost_rate=0.001, ledger_dir=tmp_path)
    assert rows == [] and why is not None and "源不可用" in why
    assert not theo_path.exists()

    monkeypatch.setattr("web.ledger_stats.get_source", lambda kind: _BoomSource())
    rows, why = LS.recompute_theo("fake", "BTCUSDT", "1h", FORMULA,
                                  start_ts=0, end_ts=5 * 3600,
                                  cost_rate=0.001, ledger_dir=tmp_path)
    assert rows == [] and why is not None and "boom" in why
    assert not theo_path.exists()


def test_unknown_tf_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("web.ledger_stats.get_source", lambda kind: _FakeSource(_sine_bars(10)))
    rows, why = LS.recompute_theo("fake", "BTCUSDT", "3x", FORMULA,
                                  start_ts=0, end_ts=100,
                                  cost_rate=0.001, ledger_dir=tmp_path)
    assert rows == [] and why is not None
