# tests/unit/test_ledger_wiring.py
import time  # noqa: F401  (与 brief 保持一致；实现侧使用 time)
import torch  # noqa: F401  (与 brief 保持一致；realtime 链路依赖 torch 环境)
from pathlib import Path  # noqa: F401
import web.live_ledger as LL
from web.realtime_manager import RealtimeManager, WatchTask


class _B:
    def __init__(self, ts):
        self.ts = ts
        self.open = self.high = self.low = self.close = 100.0
        self.volume = 1.0


def _mk_mgr(tmp_path, monkeypatch):
    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    mgr = RealtimeManager()
    task = WatchTask(id="okx:BTCUSDT:1h:x", source="okx", symbol="BTCUSDT",
                     timeframe="1h", strategy_file="s.json", strategy_name="s",
                     formula=[1], vocab_version="v", strategy_symbol=None,
                     strategy_timeframe=None, best_score=None, cadence_s=60)
    mgr._tasks[task.id] = task
    return mgr, task


def last_bar_ts_count(tmp_path):
    recs, _ = LL.read_records(LL.ledger_path("okx", "BTCUSDT", "1h", ledger_dir=tmp_path))
    return len(recs)


def test_one_row_per_closed_bar(tmp_path, monkeypatch):
    mgr, task = _mk_mgr(tmp_path, monkeypatch)
    bars = [_B(1000), _B(4600)]        # 1h 间隔
    monkeypatch.setattr(mgr, "_get_bars", lambda *a: bars)
    monkeypatch.setattr("web.realtime_manager.evaluate_signal",
                        lambda f, raw, **k: {"state": "ok", "direction": "up", "strength": 0.5,
                                             "position": 0.5, "factor_value": 1.0,
                                             "bars_used": len(raw['close'][0])})
    try:
        mgr._evaluate_task(task)
        mgr._evaluate_task(task)       # 同一最后一根 bar 评估两次 → 只写 1 行
        assert last_bar_ts_count(tmp_path) == 1
        bars.append(_B(8200))          # 新收盘 bar
        mgr._evaluate_task(task)
        assert last_bar_ts_count(tmp_path) == 2
        assert task.last_ledger_bar_ts == 8200
    finally:
        mgr.stop()
        mgr._executor.shutdown(wait=False)


def test_restart_recovers_last_ledger_bar_ts(tmp_path, monkeypatch):
    """watch 重建（进程重启）时从账本末行恢复 last_ledger_bar_ts，同一 bar 不重复写。"""
    import json as _json

    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    # 预置账本：已有 bar_ts=8200 的记录（sha 与本 watch 公式一致，锚点按 sha 归组）
    rec = LL.LedgerRecord(ts=1.0, bar_ts=8200, source="tradingview", symbol="BTCUSDT",
                          timeframe="1h", strategy_sha=LL.formula_sha([1]),
                          factor_value=1.0,
                          position=0.5, direction="up", strength=0.5, close=100.0)
    assert LL.append_record(rec) is True

    sf = tmp_path / "btc_1h.json"
    sf.write_text(_json.dumps({"formula": [1], "vocab_version": "v"}), encoding="utf-8")

    class _FakeSrc:
        label = "TV"

        def supported_timeframes(self):
            return ["1h"]

    monkeypatch.setattr("web.realtime_manager.get_source", lambda kind: _FakeSrc())
    mgr = RealtimeManager()
    try:
        task = mgr._add_task_internal("tradingview", "BTCUSDT", "1h", str(sf), persist=False)
        assert task.last_ledger_bar_ts == 8200
    finally:
        mgr.stop()
        mgr._executor.shutdown(wait=False)


def test_restart_anchor_is_per_sha(tmp_path, monkeypatch):
    """终评 F1①：账本文件被同 pair+tf 的多策略共享（spec §5.2 按 sha 去重）。
    恢复锚点只能取「本 sha」记录的最大 bar_ts——文件末行属于别的 sha 时不能被继承，
    否则后加策略会永久跳过自己尚未写过的 bar。"""
    import json as _json

    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    sha_b = LL.formula_sha([1])   # 正在重建的 watch 的策略
    sha_a = LL.formula_sha([2])   # 同 pair+tf 的另一个策略

    def _r(bar_ts, sha):
        return LL.LedgerRecord(ts=float(bar_ts), bar_ts=bar_ts, source="tradingview",
                               symbol="BTCUSDT", timeframe="1h", strategy_sha=sha,
                               factor_value=1.0, position=0.5, direction="up",
                               strength=0.5, close=100.0)

    # 交错写入：文件末行是 A 的 7200；B 自己的最大 bar_ts 只有 3600
    for rec in (_r(0, sha_b), _r(3600, sha_b), _r(7200, sha_a)):
        assert LL.append_record(rec) is True

    sf = tmp_path / "btc_1h.json"
    sf.write_text(_json.dumps({"formula": [1], "vocab_version": "v"}), encoding="utf-8")

    class _FakeSrc:
        label = "TV"

        def supported_timeframes(self):
            return ["1h"]

    monkeypatch.setattr("web.realtime_manager.get_source", lambda kind: _FakeSrc())
    mgr = RealtimeManager()
    try:
        task = mgr._add_task_internal("tradingview", "BTCUSDT", "1h", str(sf), persist=False)
        assert task.last_ledger_bar_ts == 3600      # B 的锚点，而非全文件末行 7200
    finally:
        mgr.stop()
        mgr._executor.shutdown(wait=False)
