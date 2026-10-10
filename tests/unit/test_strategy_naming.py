# tests/unit/test_strategy_naming.py
from web.strategy_file import parse_best_stem, strategy_path_for_symbol
import pytest

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
