# tests/unit/test_engine_strategy_path.py
from model_core.engine import _strategy_file_for_symbol

def test_path_includes_tf():
    assert _strategy_file_for_symbol("XAUUSD", "H1").endswith("best_XAUUSD_H1.json")

def test_missing_tf_returns_none():
    assert _strategy_file_for_symbol("XAUUSD", None) is None
    assert _strategy_file_for_symbol(None, "H1") is None


def test_illegal_tf_returns_none():
    # 与 A1 相同规则：tf 必须匹配 [A-Za-z0-9]{1,4}，否则拒绝落盘
    assert _strategy_file_for_symbol("XAUUSD", "H1X2ZZ") is None   # 超过 4 字符
    assert _strategy_file_for_symbol("XAUUSD", "H;1") is None      # 路径/非法字符
    assert _strategy_file_for_symbol("XAUUSD", "../evil") is None  # 路径穿越
    assert _strategy_file_for_symbol("XAUUSD", "") is None
    assert _strategy_file_for_symbol("XAUUSD", ["H1"]) is None     # 非字符串（岛模式列表）
