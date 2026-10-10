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
    # 预置账本：已有 bar_ts=8200 的记录
    rec = LL.LedgerRecord(ts=1.0, bar_ts=8200, source="tradingview", symbol="BTCUSDT",
                          timeframe="1h", strategy_sha="abcdef", factor_value=1.0,
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
