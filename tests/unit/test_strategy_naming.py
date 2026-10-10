# tests/unit/test_strategy_naming.py
import json
from pathlib import Path

import pytest

from web import strategy_file as sf
from web.strategy_file import (
    parse_best_stem,
    strategy_path_for_symbol,
    symbol_from_strategy_path,
    tf_from_best_name,
)

def test_parse_best_stem_ok():
    assert parse_best_stem("best_XAUUSD_H1") == ("XAUUSD", "H1")
    assert parse_best_stem("best_metals_comm_1h") == ("metals_comm", "1h")
    assert parse_best_stem("best_US30.cash_H4") == ("US30.cash", "H4")

def test_parse_best_stem_legacy_rejected():
    assert parse_best_stem("best_XAUUSD") is None      # 旧命名
    assert parse_best_stem("strategy_XAUUSD_step0060") is None

def test_strategy_path_requires_tf():
    p = strategy_path_for_symbol("XAUUSD", "H1")
    assert p.name == "best_XAUUSD_H1.json"
    with pytest.raises(ValueError):
        strategy_path_for_symbol("XAUUSD", "")

def test_strategy_path_rejects_overlong_tf():
    # 周期超过 4 个字符视为非法
    with pytest.raises(ValueError):
        strategy_path_for_symbol("XAUUSD", "12345")


def test_tf_from_best_name():
    assert tf_from_best_name(Path("best_XAUUSD_H1.json")) == "H1"
    assert tf_from_best_name(Path("best_metals_comm_1h.json")) == "1h"

def test_tf_from_best_name_none_for_legacy_and_export():
    assert tf_from_best_name(Path("best_XAUUSD.json")) is None
    assert tf_from_best_name(Path("strategy_XAUUSD_step0060.json")) is None
    assert tf_from_best_name(Path("random.json")) is None


def test_symbol_from_strategy_path_returns_tuple():
    assert symbol_from_strategy_path(Path("best_XAUUSD_H4.json")) == ("XAUUSD", "H4")
    assert symbol_from_strategy_path(Path("best_metals_comm_1h.json")) == ("metals_comm", "1h")
    assert symbol_from_strategy_path(Path("strategy_XAUUSD_step0060.json")) == ("XAUUSD", None)
    assert symbol_from_strategy_path(Path("random.json")) is None


@pytest.fixture
def tmp_strategies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """仅隔离 strategy_file 的目录与设置来源，不 mock 被测函数本身。"""
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    import web.progress as progress_mod
    import web.settings as settings_mod

    monkeypatch.setattr(progress_mod, "STRATEGIES_DIR", strategies)
    monkeypatch.setattr(sf, "STRATEGIES_DIR", strategies)
    monkeypatch.setattr(sf, "checkpoint_glob", lambda _symbol: [])
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", tmp_path / "web_settings.json")
    monkeypatch.setattr(settings_mod, "STRATEGIES_DIR", strategies)
    monkeypatch.chdir(tmp_path)
    return strategies


def test_inspect_list_file_uses_filename_symbol_and_tf(tmp_strategies: Path):
    # findings 2：list 形态文件过去把元组直接当 symbol 消费 → AttributeError
    p = tmp_strategies / "best_XAUUSD_H1.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    info = sf.inspect_strategy_file(str(p))
    assert info["symbol"] == "XAUUSD"
    assert info["timeframe"] == "H1"


def test_inspect_dict_without_symbol_falls_back_to_filename(tmp_strategies: Path):
    p = tmp_strategies / "best_US30.cash_M15.json"
    p.write_text(json.dumps({"formula": [1, 2], "best_score": 0.5}), encoding="utf-8")
    info = sf.inspect_strategy_file(str(p))
    assert info["symbol"] == "US30.cash"
    assert info["timeframe"] == "M15"


def test_sync_uses_explicit_timeframe(tmp_strategies: Path):
    # findings 1：sync 需要 timeframe 入参，写入 best_{symbol}_{tf}.json
    # （A3 起 _load_strategy 只认新命名，种子文件同步用新命名）
    seed = tmp_strategies / "best_XAUUSD_M30.json"
    seed.write_text(
        json.dumps({"symbol": "XAUUSD", "formula": [1, 2, 3], "best_score": 1.5}),
        encoding="utf-8",
    )
    info = sf.sync_best_strategy_for_symbol("XAUUSD", "M30")
    assert info is not None
    out = tmp_strategies / "best_XAUUSD_M30.json"
    assert out.exists()
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["timeframe"] == "M30"
    assert info["symbol"] == "XAUUSD"
    assert info["timeframe"] == "M30"


def test_sync_falls_back_to_payload_timeframe(tmp_strategies: Path):
    # 无显式 timeframe 时，sync 从既有新命名策略文件解析周期（旧命名不参与）
    seed = tmp_strategies / "best_XAUUSD_H1.json"
    seed.write_text(
        json.dumps(
            {"symbol": "XAUUSD", "formula": [1, 2, 3], "best_score": 1.5, "timeframe": "H1"}
        ),
        encoding="utf-8",
    )
    info = sf.sync_best_strategy_for_symbol("XAUUSD")
    assert info is not None
    assert (tmp_strategies / "best_XAUUSD_H1.json").exists()


def test_sync_without_any_timeframe_raises(tmp_strategies: Path):
    seed = tmp_strategies / "best_XAUUSD.json"
    seed.write_text(
        json.dumps({"symbol": "XAUUSD", "formula": [1, 2, 3], "best_score": 1.5}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        sf.sync_best_strategy_for_symbol("XAUUSD")


def test_resolve_strategy_file_falls_back_to_tf_named_default(tmp_strategies: Path):
    default = tmp_strategies / "best_XAUUSD_H1.json"
    default.write_text(json.dumps({"symbol": "XAUUSD", "formula": [1]}), encoding="utf-8")
    resolved = sf.resolve_strategy_file("", train_symbol="XAUUSD", timeframe="H1")
    assert resolved == str(default.resolve())
    # 无法确定周期时拒绝回退到旧命名
    with pytest.raises(ValueError):
        sf.resolve_strategy_file("", train_symbol="XAUUSD")
