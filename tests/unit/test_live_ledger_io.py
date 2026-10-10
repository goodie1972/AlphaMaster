# tests/unit/test_live_ledger_io.py
import json
from web.live_ledger import (LedgerRecord, append_record, read_records,
                             last_bar_ts, formula_sha, ensure_strategy_index, load_strategy_index)

def _rec(bar_ts, **kw):
    base = dict(ts=0.0, bar_ts=bar_ts, source="okx", symbol="BTCUSDT",
                timeframe="1h", strategy_sha="abc123", factor_value=1.0,
                position=0.5, direction="up", strength=0.5, close=100.0)
    return LedgerRecord(**{**base, **kw})

def test_append_read_roundtrip(tmp_path):
    assert append_record(_rec(1000), ledger_dir=tmp_path)
    assert append_record(_rec(2000), ledger_dir=tmp_path)
    path = last_good(tmp_path)
    recs, corrupt = read_records(path)
    assert [(r.bar_ts, r.position, r.close) for r in recs] == [(1000, 0.5, 100.0), (2000, 0.5, 100.0)]
    assert corrupt == 0

def test_corrupt_line_counted(tmp_path):
    append_record(_rec(1000), ledger_dir=tmp_path)
    last_good(tmp_path).open("a", encoding="utf-8").write('{"broken\n')
    recs, corrupt = read_records(last_good(tmp_path))
    assert len(recs) == 1 and corrupt == 1

def test_last_bar_ts_and_empty(tmp_path):
    assert last_bar_ts(last_good(tmp_path)) is None
    append_record(_rec(1000), ledger_dir=tmp_path)
    assert last_bar_ts(last_good(tmp_path)) == 1000

def test_strategy_index_dedupe(tmp_path):
    sha1 = ensure_strategy_index([1, 2, 3], "s.json", "v9", ledger_dir=tmp_path)
    idx = load_strategy_index(tmp_path)
    assert sha1 == formula_sha([1, 2, 3]) and len(idx) == 1
    # spec §5.1：索引文件名 strategy_index.json，条目含 formula + first_seen
    assert (tmp_path / "strategy_index.json").is_file()
    entry = idx[sha1]
    assert entry["formula"] == [1, 2, 3]
    assert entry["strategy_file"] == "s.json" and entry["vocab_version"] == "v9"
    first_seen = entry["first_seen"]
    assert isinstance(first_seen, (int, float))
    # 幂等：同 sha 再登记只 1 条，且保留原 first_seen
    ensure_strategy_index([1, 2, 3], "s.json", "v9", ledger_dir=tmp_path)
    idx = load_strategy_index(tmp_path)
    assert len(idx) == 1 and idx[sha1]["first_seen"] == first_seen
    # 元数据更新（忽略 first_seen 判等）：覆盖新值但仍保留原 first_seen
    ensure_strategy_index([1, 2, 3], "s2.json", "v10", ledger_dir=tmp_path)
    idx = load_strategy_index(tmp_path)
    assert len(idx) == 1
    assert idx[sha1]["strategy_file"] == "s2.json" and idx[sha1]["vocab_version"] == "v10"
    assert idx[sha1]["first_seen"] == first_seen

def test_strategy_index_two_formulas(tmp_path):
    ensure_strategy_index([1, 2, 3], "s.json", "v9", ledger_dir=tmp_path)
    ensure_strategy_index([4, 5, 6], "t.json", "v9", ledger_dir=tmp_path)
    idx = load_strategy_index(tmp_path)
    assert len(idx) == 2
    assert set(idx) == {formula_sha([1, 2, 3]), formula_sha([4, 5, 6])}

def last_good(tmp_path):
    from web.live_ledger import ledger_path
    return ledger_path("okx", "BTCUSDT", "1h", ledger_dir=tmp_path)


# --- extras per controller rulings: whitelist on load + safe() sanitization ---

def test_unknown_field_in_raw_line_ignored(tmp_path):
    append_record(_rec(1000), ledger_dir=tmp_path)
    raw = dict(ts=0.0, bar_ts=2000, source="okx", symbol="BTCUSDT",
               timeframe="1h", strategy_sha="abc123", factor_value=1.0,
               position=0.5, direction="up", strength=0.5, close=100.0,
               mystery_field=42)
    last_good(tmp_path).open("a", encoding="utf-8").write(json.dumps(raw) + "\n")
    recs, corrupt = read_records(last_good(tmp_path))
    assert len(recs) == 2 and corrupt == 0
    assert not hasattr(recs[1], "mystery_field")

def test_ledger_path_sanitizes_unsafe_chars(tmp_path):
    import re
    from web.live_ledger import ledger_path
    name = ledger_path("ok,x", "BTC/USDT", "1 h", ledger_dir=tmp_path).name
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", name)
