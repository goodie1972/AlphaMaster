# tests/unit/test_watch_validation.py
import json
from pathlib import Path

import pytest

from web.strategy_file import inspect_strategy_file


def test_legacy_file_rejected(tmp_path):
    p = tmp_path / "best_XAUUSD.json"
    p.write_text(json.dumps({"symbol": "XAUUSD", "formula": [1]}), encoding="utf-8")
    info = inspect_strategy_file(str(p))
    assert info.get("error") == "legacy_naming"   # 新约定：返回错误码而非静默接受


def test_new_named_file_inspects_without_error(tmp_path):
    p = tmp_path / "best_XAUUSD_H1.json"
    p.write_text(
        json.dumps({"symbol": "XAUUSD", "timeframe": "H1", "formula": [1]}),
        encoding="utf-8",
    )
    info = inspect_strategy_file(str(p))
    assert info.get("error") is None
    assert info["valid"] is True
    assert info["symbol"] == "XAUUSD"
    assert info["timeframe"] == "H1"


LEGACY_MSG = "旧命名策略不可用：文件名需含周期，请用新命名重新导出"


def test_add_watch_rejects_legacy_naming(tmp_path):
    from web.realtime_manager import RealtimeManager

    p = tmp_path / "best_XAUUSD.json"
    p.write_text(json.dumps({"symbol": "XAUUSD", "formula": [1]}), encoding="utf-8")
    mgr = RealtimeManager()
    try:
        with pytest.raises(ValueError) as exc:
            mgr.add_watch("tradingview", "XAUUSD", "1h", str(p))
        assert str(exc.value) == LEGACY_MSG
        assert mgr.status()["count"] == 0
    finally:
        mgr.stop()
        mgr._executor.shutdown(wait=False)


def test_add_watch_new_named_timeframe_mismatch_warns(tmp_path):
    from web.realtime_manager import RealtimeManager

    p = tmp_path / "best_XAUUSD_5m.json"
    p.write_text(
        json.dumps({"symbol": "XAUUSD", "timeframe": "5m", "formula": [1]}),
        encoding="utf-8",
    )
    mgr = RealtimeManager()
    try:
        # 不启动调度线程：直接走内部添加（persist=False 不写设置）
        task = mgr._add_task_internal("tradingview", "XAUUSD", "1h", str(p), persist=False)
        assert "5m" in task.warn and "1h" in task.warn
    finally:
        mgr.clear()
        mgr.stop()
        mgr._executor.shutdown(wait=False)


def test_add_watch_new_named_matching_tf_no_warn(tmp_path):
    from web.realtime_manager import RealtimeManager

    p = tmp_path / "best_XAUUSD_5m.json"
    p.write_text(
        json.dumps({"symbol": "XAUUSD", "timeframe": "5m", "formula": [1]}),
        encoding="utf-8",
    )
    mgr = RealtimeManager()
    try:
        task = mgr._add_task_internal("tradingview", "XAUUSD", "5m", str(p), persist=False)
        assert task.warn == ""
    finally:
        mgr.clear()
        mgr.stop()
        mgr._executor.shutdown(wait=False)


# ── 终评 F2：tf 不一致告警按「秒数等价」判定，两种记法域（H1/M30… 与 1h/30m…）互认 ──

def _mk_mgr_with_fake_src(monkeypatch, tfs):
    from web.realtime_manager import RealtimeManager

    class _FakeSrc:
        label = "TV"

        def supported_timeframes(self):
            return list(tfs)

    monkeypatch.setattr("web.realtime_manager.get_source", lambda kind: _FakeSrc())
    return RealtimeManager()


def test_strategy_H1_watch_1h_is_equivalent_no_warn(tmp_path, monkeypatch):
    """主流真实场景：策略按 parquet 记法标 H1、监控按 realtime 记法配 1h → 同周期，不应告警。"""
    p = tmp_path / "best_XAUUSD_H1.json"
    p.write_text(
        json.dumps({"symbol": "XAUUSD", "timeframe": "H1", "formula": [1]}),
        encoding="utf-8",
    )
    mgr = _mk_mgr_with_fake_src(monkeypatch, ["1h", "4h"])
    try:
        task = mgr._add_task_internal("tradingview", "XAUUSD", "1h", str(p), persist=False)
        assert task.warn == ""
    finally:
        mgr.clear()
        mgr.stop()
        mgr._executor.shutdown(wait=False)


def test_true_tf_mismatch_warns_with_original_strings(tmp_path, monkeypatch):
    """H1(3600s) vs 4h(14400s) 真不一致 → 告警，且文案原样打印两侧原始字符串。"""
    p = tmp_path / "best_XAUUSD_H1.json"
    p.write_text(
        json.dumps({"symbol": "XAUUSD", "timeframe": "H1", "formula": [1]}),
        encoding="utf-8",
    )
    mgr = _mk_mgr_with_fake_src(monkeypatch, ["1h", "4h"])
    try:
        task = mgr._add_task_internal("tradingview", "XAUUSD", "4h", str(p), persist=False)
        assert task.warn != ""
        assert "H1" in task.warn and "4h" in task.warn   # 原样，不做大小写/记法转换
    finally:
        mgr.clear()
        mgr.stop()
        mgr._executor.shutdown(wait=False)


def test_unparseable_tf_warns(tmp_path, monkeypatch):
    """任一侧记法无法折算秒数 → 保守告警（宁亮不误撤）。"""
    p = tmp_path / "best_XAUUSD_xyz.json"
    p.write_text(
        json.dumps({"symbol": "XAUUSD", "timeframe": "xyz", "formula": [1]}),
        encoding="utf-8",
    )
    mgr = _mk_mgr_with_fake_src(monkeypatch, ["1h"])
    try:
        task = mgr._add_task_internal("tradingview", "XAUUSD", "1h", str(p), persist=False)
        assert task.warn != ""
        assert "xyz" in task.warn and "1h" in task.warn
    finally:
        mgr.clear()
        mgr.stop()
        mgr._executor.shutdown(wait=False)
