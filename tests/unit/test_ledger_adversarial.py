# tests/unit/test_ledger_adversarial.py
"""对抗性测试：malformed / malicious 输入攻击台账 API、IO 与命名层。

约定（同 test_ledger_api.py / test_live_ledger_io.py）：
- TestClient(app) 不进上下文管理器 —— startup 事件绝不触发。
- 一切文件 IO 都锁在 tmp_path：`led` autouse fixture monkeypatch
  web.live_ledger.LEDGER_DIR；命名层测试另用 `strategies` fixture patch
  STRATEGIES_DIR / checkpoint_glob / settings。真实 live_ledger/、strategies/、
  web_settings.json、端口 8765 绝不触碰。
- autouse 拦掉真实数据源（LS.get_source → raise）与真实 settings 读取
  （web.app.load_settings → {}，cost_rate 确定性回退 Config.COST_RATE=0.0003）。
- 断言的是「契约上正确」的行为；确认的真实缺陷保留为
  @pytest.mark.xfail(strict=False, reason="DEFECT: ...") 以便持续可见。
- FastAPI 对不可 int 转换的查询参数回 422 属已知接受行为：断言 200 或 422 并注释。
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import threading

import pytest
from fastapi.testclient import TestClient

import web.ledger_stats as LS
import web.live_ledger as LL
from web import strategy_file as sf
from web.app import app

client = TestClient(app)

FORMULA = [25, 81, 61, 103, 93, 104, 31, 115]
SHA = LL.formula_sha(FORMULA)
FALLBACK_COST = 0.0003          # load_settings→{} 时 _ledger_cost_rate 兜底口径
WATCH = "okx_BTCUSDT_1h"
STD_FIELDS = ("ts", "bar_ts", "source", "symbol", "timeframe", "strategy_sha",
              "factor_value", "position", "direction", "strength", "close")


def _rec(bar_ts, pos=0.5, close=100.0, *, sha=SHA, tf="1h",
         source="okx", symbol="BTCUSDT", direction="LONG"):
    return LL.LedgerRecord(
        ts=float(bar_ts), bar_ts=int(bar_ts), source=source, symbol=symbol,
        timeframe=tf, strategy_sha=sha, factor_value=float(pos), position=float(pos),
        direction=direction, strength=abs(float(pos)), close=float(close))


def _rec_line(bar_ts, pos=0.5, close=100.0, *, sha=SHA, tf="1h",
              source="okx", symbol="BTCUSDT", ts=None, **extra):
    """账本原始行文本；显式字段可被覆盖成 dataclass 不校验的垃圾类型，
    extra 键模拟注入未知字段。"""
    d = {"ts": 0.0, "bar_ts": bar_ts, "source": source, "symbol": symbol,
         "timeframe": tf, "strategy_sha": sha, "factor_value": 1.0,
         "position": pos, "direction": "LONG", "strength": 0.5, "close": close}
    if ts is not None:
        d["ts"] = ts
    d.update(extra)
    return json.dumps(d, ensure_ascii=False)


def _write(led, recs):
    for r in recs:
        assert LL.append_record(r, ledger_dir=led)


def _append_lines(led, lines, name=f"{WATCH}.jsonl"):
    with (led / name).open("a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln if ln.endswith("\n") else ln + "\n")


def _files(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def _err_only(r):
    """watch_id 类攻击的统一期望：HTTP 200 + 纯 {"error"}（无任何数据字段外泄）。"""
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, dict) and set(body) == {"error"}, body
    assert body["error"], body
    return body["error"]


class _B:
    def __init__(self, ts, c):
        self.ts = self.open = self.high = self.low = self.close = c
        self.volume = 1.0


class _FakeSource:
    def __init__(self, bars):
        self.bars = bars

    def fetch_bars(self, symbol, tf, n, drop_forming=True):
        return self.bars[-n:]


def _sine_bars(count, step=3600):
    return [_B(i * step, 100.0 + math.sin(i / 5)) for i in range(count)]


@pytest.fixture(autouse=True)
def led(monkeypatch, tmp_path):
    """每个测试：LEDGER_DIR → tmp_path/led；禁真实源；禁真实 settings。"""
    d = tmp_path / "led"
    d.mkdir()
    monkeypatch.setattr(LL, "LEDGER_DIR", d)

    def _boom(kind):
        raise AssertionError(f"测试禁止真实数据源: {kind}")
    monkeypatch.setattr(LS, "get_source", _boom)
    monkeypatch.setattr("web.app.load_settings", lambda: {})
    return d


@pytest.fixture
def strategies(tmp_path, monkeypatch):
    sd = tmp_path / "strategies"
    sd.mkdir()
    import web.progress as pm
    import web.settings as sm
    monkeypatch.setattr(pm, "STRATEGIES_DIR", sd)
    monkeypatch.setattr(sf, "STRATEGIES_DIR", sd)
    monkeypatch.setattr(sf, "checkpoint_glob", lambda _symbol: [])
    monkeypatch.setattr(sm, "SETTINGS_PATH", tmp_path / "web_settings.json")
    monkeypatch.setattr(sf, "load_settings", lambda: {})
    return sd


# ════════════════════════════════════════════════════════════════════
# 1. watch_id 穿越 / 恶意字符：一律 200 + {"error"}，绝不越出台账目录
# ════════════════════════════════════════════════════════════════════

ENDPOINTS = ("/api/ledger/summary", "/api/ledger/curve",
             "/api/ledger/records", "/api/ledger/export")


def test_watchid_traversal_and_absolute_paths_rejected(led, tmp_path):
    _write(led, [_rec(0)])
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.jsonl").write_text(_rec_line(0) + "\n", encoding="utf-8")
    before = _files(tmp_path)
    bad = ["../outside/secret", "../live", "..\\..\\config", "C:/Windows/winini",
           "C:\\Windows\\winini", "/etc/passwd", "..\\led\\secret"]
    for ep in ENDPOINTS:
        for wid in bad:
            _err_only(client.get(ep, params={"watch_id": wid}))
    assert (outside / "secret.jsonl").read_text(encoding="utf-8") == _rec_line(0) + "\n"
    assert _files(tmp_path) == before            # 零新建/零删除/零读出


def test_watchid_dot_dotspace_empty_variants(led, tmp_path):
    before = _files(tmp_path)
    for wid in (".", "..", "", "   ", "\t\n", "...", "...."):
        _err_only(client.get("/api/ledger/summary", params={"watch_id": wid}))
    assert _files(tmp_path) == before


def test_watchid_url_encoded_traversal(led, tmp_path):
    """raw query 会被 starlette 解码（%2e%2e%2f → "../"）→ 分隔符检查拦截；
    params dict 传字面量 "%2e%2e%2f..." → 不匹配目录清单 → 拒。两条路都必须死。"""
    _write(led, [_rec(0)])
    before = _files(tmp_path)
    _err_only(client.get("/api/ledger/summary?watch_id=%2e%2e%2foutside%2fsecret"))
    _err_only(client.get("/api/ledger/summary", params={"watch_id": "%2e%2e%2flive"}))
    _err_only(client.get("/api/ledger/summary", params={"watch_id": "..%2f..%2fconfig"}))
    assert _files(tmp_path) == before


def test_watchid_nullbyte_and_newline(led, tmp_path):
    _write(led, [_rec(0)])
    before = _files(tmp_path)
    for wid in ("a\x00b", "okx_BTCUSDT_1h\x00", "mt5\nXAUUSD", "okx_BTCUSDT_1h\n.jsonl"):
        _err_only(client.get("/api/ledger/summary", params={"watch_id": wid}))
    # 尾部换行被 strip 后命中真实账本属良性归一化（只允许读到账本自己）
    r = client.get("/api/ledger/summary", params={"watch_id": "okx_BTCUSDT_1h\n"})
    assert r.status_code == 200 and r.json().get("watch_id") == WATCH
    assert _files(tmp_path) == before


def test_watchid_overlong_10k_and_homoglyph(led, tmp_path):
    _write(led, [_rec(0)])
    before = _files(tmp_path)
    for wid in ("x" * 10000, "ｍｔ5_XAUUSD_1h", "оkx_BTCUSDT_1h", "okx_BTCUSDT_１h"):
        _err_only(client.get("/api/ledger/summary", params={"watch_id": wid}))
    assert _files(tmp_path) == before


# ════════════════════════════════════════════════════════════════════
# 2. 查询参数滥用：200+error / 合理钳位 / 空结果，绝不 500
# ════════════════════════════════════════════════════════════════════

def test_days_out_of_int_range_is_200_error(led):
    _write(led, [_rec(i * 3600) for i in range(3)])
    for bad in (0, -1, -10**9, 3651, 10**12, "9" * 40):
        r = client.get("/api/ledger/summary", params={"watch_id": WATCH, "days": bad})
        assert r.status_code == 200, bad
        assert "days" in r.json().get("error", ""), bad
    # 边界内合法：days=3650 → 正常数据（theo 因索引缺失降级，而非 error）
    r = client.get("/api/ledger/summary", params={"watch_id": WATCH, "days": 3650})
    assert r.status_code == 200 and "records" in r.json()


@pytest.mark.parametrize("ep", ["/api/ledger/summary", "/api/ledger/curve",
                                "/api/ledger/records"])
def test_non_coercible_params_422_accepted(led, ep):
    """已知接受行为：days="abc"/"1e18" 无法 coerce 成 int → FastAPI 输入层 422。
    台账契约只承诺「绝不 500」；422 属框架层拒绝，接受，不与之搏斗。"""
    _write(led, [_rec(0)])
    for bad in ("abc", "1e18"):
        r = client.get(ep, params={"watch_id": WATCH, "days": bad})
        assert r.status_code in (200, 422), (ep, bad, r.status_code)
        if r.status_code == 200:
            assert "error" in r.json()


def test_limit_offset_clamped_not_crashed(led):
    _write(led, [_rec(i * 3600) for i in range(5)])
    base = {"watch_id": WATCH}
    for limit, expect in ((0, 1), (-5, 1), (99999999, 5000)):
        r = client.get("/api/ledger/records", params={**base, "limit": limit})
        assert r.status_code == 200, limit
        assert r.json()["limit"] == expect, (limit, r.json()["limit"])
    r = client.get("/api/ledger/records", params={**base, "offset": -1})
    assert r.status_code == 200 and r.json()["offset"] == 0
    # offset 越界 → 空页而不是错误
    r = client.get("/api/ledger/records", params={**base, "offset": 10**6})
    assert r.status_code == 200 and r.json()["records"] == []


def test_direction_and_sha_injection_strings_degrade(led):
    _write(led, [_rec(i * 3600) for i in range(3)])
    r = client.get("/api/ledger/records",
                   params={"watch_id": WATCH, "direction": "DROP TABLE"})
    assert r.status_code == 200 and r.json()["total"] == 0    # 字面匹配不到 → 空
    r = client.get("/api/ledger/records",
                   params={"watch_id": WATCH, "direction": "' OR 1=1 --"})
    assert r.status_code == 200 and r.json()["total"] == 0
    r = client.get("/api/ledger/summary",
                   params={"watch_id": WATCH, "sha": "<script>alert(1)</script>"})
    assert r.status_code == 200
    body = r.json()
    assert body.get("records") == 0 and body.get("theo_available") is False
    assert body.get("win_rate") is None                       # 无错误统计产出
    r = client.get("/api/ledger/export", params={"watch_id": WATCH, "sha": "%00%0d%0a"})
    assert r.status_code == 200                               # 无匹配段：空 CSV 或 error
    assert r.headers["content-type"].startswith("text/csv") or "error" in r.json()


# ════════════════════════════════════════════════════════════════════
# 3. 损坏账本文件：read_records corrupt 计数语义 + 端点降级
# ════════════════════════════════════════════════════════════════════

def test_read_records_malformed_lines_counted_exactly(led):
    p = led / f"{WATCH}.jsonl"
    lines = [
        _rec_line(0), _rec_line(3600),                     # 2 条合法
        '{"bar_ts": 7200, "so',                            # 半行截断 → corrupt
        'not json at all!!',                               # 垃圾文本 → corrupt
        '[1, 2, 3]',                                       # JSON list → corrupt
        '"just a string"',                                 # JSON str → corrupt
        '42',                                              # JSON number → corrupt
        'null',                                            # JSON null → corrupt
        '{}',                                              # 缺全部必填 → corrupt(TypeError)
        '',                                                # 空行 → 忽略不计数
        _rec_line(7200, mystery={"x": 1}),                 # 未知字段白名单过滤 → 合法
    ]
    _append_lines(led, lines)
    recs, corrupt = LL.read_records(p)
    assert corrupt == 7, corrupt
    assert len(recs) == 3 and not hasattr(recs[2], "mystery")
    body = client.get("/api/ledger/summary", params={"watch_id": WATCH}).json()
    assert body["corrupt_lines"] == 7 and body["records"] == 3   # 降级可见，不崩


def test_read_records_binary_garbage_never_raises(led):
    """契约：read_records「缺失或损坏内容降级为空结果，绝不抛出」（模块 docstring 原文）。
    非 UTF-8 字节按 errors="replace" 解码后逐行走 corrupt 计数（DEFECT 已修复）。"""
    p = led / f"{WATCH}.jsonl"
    p.write_bytes(b"\xff\xfe\x00\x01binary garbage \x80\x81")
    recs, corrupt = LL.read_records(p)                     # 不抛，降级计数
    assert recs == [] and corrupt >= 1


def test_read_records_binary_endpoint_degrades_to_200_error(led):
    """修复后契约：非 UTF-8 字节按 errors="replace" 解码 → 逐行 corrupt 计数，
    端点 200 空统计（records=0 + corrupt_lines≥1），绝不 500、绝不抛出。
    （旧行为是解码异常穿透 catch-all 回 {"error"}；修复后降级为可见的 corrupt 计数。）"""
    p = led / f"{WATCH}.jsonl"
    p.write_bytes(b"\xff\xfe\x00\x01garbage")
    r = client.get("/api/ledger/summary", params={"watch_id": WATCH})
    assert r.status_code == 200
    body = r.json()
    assert "error" not in body and body["records"] == 0 and body["corrupt_lines"] >= 1


def test_wrong_typed_valid_json_records_degrade(led):
    """bar_ts="abc" / position=[1,2] / ts={} —— 修复后契约：读侧逐字段类型校验，
    毒行计入 corrupt 并跳过，合法行不受牵连；close=None 属允许值（float 字段可为
    None），该行入库但配对行记 missing。端点一律 200 出真实统计，绝不 error/500。"""
    p = led / f"{WATCH}.jsonl"
    p.write_text(_rec_line(0) + "\n" + _rec_line("abc") + "\n", encoding="utf-8")
    body = client.get("/api/ledger/summary", params={"watch_id": WATCH}).json()
    assert "error" not in body and body["records"] == 1 and body["corrupt_lines"] == 1
    p.write_text(_rec_line(0) + "\n" + _rec_line(3600, pos=[1, 2]) + "\n", encoding="utf-8")
    body = client.get("/api/ledger/summary", params={"watch_id": WATCH}).json()
    assert "error" not in body and body["records"] == 1 and body["corrupt_lines"] == 1
    p.write_text(_rec_line(0) + "\n" + _rec_line(3600, close=None) + "\n", encoding="utf-8")
    body = client.get("/api/ledger/summary", params={"watch_id": WATCH}).json()
    assert "error" not in body and body["records"] == 2 and body["corrupt_lines"] == 0
    # close=None 行仍入库（records=2）但不可配对出收益 → 相邻行 win_rate 无定义
    assert body["win_rate"] is None
    p.write_text(_rec_line(0, ts={}) + "\n", encoding="utf-8")
    body = client.get("/api/ledger/summary", params={"watch_id": WATCH}).json()
    assert "error" not in body and body.get("records") == 0   # ts 非 float → 整行 corrupt
    assert body.get("corrupt_lines") == 1


def test_one_wrong_typed_line_should_not_poison_whole_ledger(led):
    p = led / f"{WATCH}.jsonl"
    p.write_text(_rec_line(0) + "\n" + _rec_line(3600) + "\n" + _rec_line("abc") + "\n",
                 encoding="utf-8")
    body = client.get("/api/ledger/summary", params={"watch_id": WATCH}).json()
    # 契约上正确：毒行计入 corrupt，其余 2 条照常统计
    assert body.get("records") == 2 and body.get("corrupt_lines") == 1


def test_absurd_1mb_line_parsed_without_crash(led):
    _append_lines(led, [_rec_line(0, pad="x" * 1_000_000)])
    recs, corrupt = LL.read_records(led / f"{WATCH}.jsonl")
    assert corrupt == 0 and len(recs) == 1                 # 未知巨字段被过滤
    body = client.get("/api/ledger/summary", params={"watch_id": WATCH}).json()
    assert body.get("records") == 1 and body.get("corrupt_lines") == 0


def test_duplicate_bar_ts_records_no_wrong_stats(led):
    _write(led, [_rec(0, pos=1.0, close=100.0), _rec(0, pos=1.0, close=200.0)])
    body = client.get("/api/ledger/summary", params={"watch_id": WATCH}).json()
    assert body["records"] == 2
    # 重复 bar（Δ=0 ≠ tf 秒）相邻不成立 → missing，绝不产出虚假收益统计
    assert body["win_rate"] is None and body["annualized"] is None
    assert body["corrupt_lines"] == 0


def test_unknown_source_and_timeframe_records_degrade(led):
    p = led / f"{WATCH}.jsonl"
    p.write_text(_rec_line(0, source="👿", symbol="💣") + "\n", encoding="utf-8")
    body = client.get("/api/ledger/summary", params={"watch_id": WATCH}).json()
    assert body.get("records") == 1 and body.get("source") == "👿"   # 仅展示，无数学
    w = client.get("/api/ledger/watches").json()["watches"]
    assert w[0]["source"] == "👿" and w[0]["watch_id"] == WATCH
    # 毒 timeframe：_tf_seconds ValueError → 200 error，绝不按文件名误算
    p.write_text(_rec_line(0, tf="9x;rm -rf") + "\n", encoding="utf-8")
    r = client.get("/api/ledger/summary", params={"watch_id": WATCH})
    assert r.status_code == 200 and "error" in r.json()


def test_nan_bar_ts_in_ledger_degrades(led):
    """json.loads 接受 NaN 字面量：修复后 bar_ts=NaN 被读侧逐字段校验拒绝
    （int() 化前先 isfinite）→ 整行计入 corrupt 并跳过，绝不静默入库；
    端点 200 空统计，不 500。"""
    p = led / f"{WATCH}.jsonl"
    p.write_text('{"ts": 0.0, "bar_ts": NaN, "source": "okx", "symbol": "BTCUSDT",'
                 ' "timeframe": "1h", "strategy_sha": "abc", "factor_value": 1.0,'
                 ' "position": 0.5, "direction": "LONG", "strength": 0.5, "close": 100.0}\n',
                 encoding="utf-8")
    recs, corrupt = LL.read_records(p)
    assert recs == [] and corrupt == 1                   # NaN bar_ts → corrupt 计数
    r = client.get("/api/ledger/summary", params={"watch_id": WATCH})
    assert r.status_code == 200 and r.json().get("corrupt_lines") == 1
    assert r.json().get("records") == 0


def test_nan_position_does_not_leak_into_stats(led):
    """攻击：position=NaN（json 合法字面量）。修复后读侧拒 NaN float 字段 →
    毒行计 corrupt 跳过，剩余合法行独立成统计；契约：不 500、整行 error 不再
    出现、非标准 JSON 的 NaN 字面量绝不外泄。"""
    p = led / f"{WATCH}.jsonl"
    p.write_text('{"ts": 0.0, "bar_ts": 0, "source": "okx", "symbol": "BTCUSDT",'
                 ' "timeframe": "1h", "strategy_sha": "abc", "factor_value": 1.0,'
                 ' "position": NaN, "direction": "LONG", "strength": 0.5, "close": 100.0}\n'
                 + _rec_line(3600, pos=0.5, sha="abc") + "\n", encoding="utf-8")
    r = client.get("/api/ledger/summary", params={"watch_id": WATCH})
    assert r.status_code == 200
    assert "NaN" not in r.text, r.text[:200]               # 非标准 JSON 字面量不外泄
    assert "error" not in r.json()                         # 修复后：毒行隔离，不再整行 error
    body = r.json()
    assert body["records"] == 1 and body["corrupt_lines"] == 1
    for key in ("annualized", "sortino", "max_drawdown"):
        assert body.get(key) is None, (key, body.get(key))  # 绝无 NaN/假 ±20 统计外发


# ════════════════════════════════════════════════════════════════════
# 4. strategy_index.json 攻击：recompute 路径只回 ([], reason)，绝不执行 token
# ════════════════════════════════════════════════════════════════════

def test_index_missing_corrupt_or_list_degrades(led):
    _write(led, [_rec(0, pos=1.0, sha="dead01"),
                 _rec(3600, close=101.0, pos=1.0, sha="dead01")])
    idx = led / "strategy_index.json"
    cases = [None, "not json{", '["top level list"]',
             '{"dead01": "entry not dict"}', '{"dead01": {"formula": null}}']
    for case in cases:
        if case is None:
            idx.unlink(missing_ok=True)
        else:
            idx.write_text(case, encoding="utf-8")
        body = client.get("/api/ledger/summary",
                          params={"watch_id": WATCH, "sha": "dead01"}).json()
        assert "error" not in body, case                   # 实盘统计照常
        assert body["theo_available"] is False, case
        assert body["theo_reason"], case                   # 有原因，不是 500
        assert body["win_rate"] == 1.0, case               # 统计未被索引异常污染


def test_index_formula_shell_tokens_never_executes(led, tmp_path):
    _write(led, [_rec(0), _rec(3600, close=101.0, sha="dead01")])
    evil = ["1;import os", "os.system('touch pwned.txt')", "__import__('os')"]
    (led / "strategy_index.json").write_text(
        json.dumps({"dead01": {"formula": evil}}), encoding="utf-8")
    body = client.get("/api/ledger/summary",
                      params={"watch_id": WATCH, "sha": "dead01"}).json()
    assert body["theo_available"] is False
    assert "数据源拉取失败" in body["theo_reason"] or "公式索引缺失" in body["theo_reason"]
    assert not list(tmp_path.rglob("pwned*"))              # token 从未被执行
    # 直调 recompute_theo：非 int token → ([], reason)，不抛
    rows, why = LS.recompute_theo("okx", "BTCUSDT", "1h", evil,
                                  start_ts=0, end_ts=3600, cost_rate=FALLBACK_COST,
                                  ledger_dir=led)
    assert rows == [] and why and "拉取失败" in why
    assert not list(tmp_path.rglob("pwned*"))
    assert not list(led.glob(f"{WATCH}.*.theo.jsonl"))     # 失败绝不写缓存（任何 sha 键控名）


def test_index_formula_nonint_container_tokens_degrade(led):
    rows, why = LS.recompute_theo("okx", "BTCUSDT", "1h", {"a": 1},
                                  start_ts=0, end_ts=3600, cost_rate=FALLBACK_COST,
                                  ledger_dir=led)
    assert rows == [] and why is not None                  # int("a") ValueError → reason


def test_vocab_out_of_range_tokens_no_crash(tmp_path, monkeypatch):
    """999999 / -5 越界 token：能过 int() 但在词表外。给足历史进真实 VM 链，
    要求要么拒绝（reason），要么全行无仓位 —— 绝不抛出、不对垃圾 token
    给出可信仓位。"""
    monkeypatch.setattr(LS, "get_source", lambda kind: _FakeSource(_sine_bars(1120)))
    for formula in ([999999], [-5], [999999, -5, 25, 81, 61, 103, 93, 104]):
        rows, why = LS.recompute_theo("okx", "BTCUSDT", "1h", formula,
                                      start_ts=1110 * 3600, end_ts=1119 * 3600,
                                      cost_rate=FALLBACK_COST, ledger_dir=tmp_path)
        assert isinstance(rows, list) and isinstance(why, (str, type(None))), formula
        if why is None:
            assert all(r["position"] is None for r in rows), formula


def test_huge_formula_100k_tokens_indexed_but_never_vmstorm(led, tmp_path):
    big = list(range(100000))
    sha = LL.ensure_strategy_index(big, "best_X_1h.json", "v1", ledger_dir=led)
    assert sha == LL.formula_sha(big)
    _write(led, [_rec(0, sha=sha), _rec(3600, close=101.0, sha=sha)])
    body = client.get("/api/ledger/summary", params={"watch_id": WATCH}).json()
    assert body["theo_available"] is False                 # boom 源即时拒绝
    assert "数据源拉取失败" in body["theo_reason"]
    assert not list(tmp_path.rglob("*.theo.jsonl"))
    # warm-up 预检路径：足量 100k token 序列在源截断时于 VM 前即被拒
    monkey_seen = []
    def _few(kind):
        monkey_seen.append(kind)
        return _FakeSource(_sine_bars(300))
    orig = LS.get_source
    try:
        LS.get_source = _few
        rows, why = LS.recompute_theo("okx", "BTCUSDT", "1h", big,
                                      start_ts=0, end_ts=9 * 3600,
                                      cost_rate=FALLBACK_COST, ledger_dir=tmp_path)
    finally:
        LS.get_source = orig
    assert rows == [] and why and "历史 bar 不足" in why   # 未进逐根 VM 风暴
    assert not list(tmp_path.rglob("*.theo.jsonl"))        # 拒绝路径零缓存落盘（任何 sha 名）


# ════════════════════════════════════════════════════════════════════
# 5. theo 缓存投毒：损坏 .theo.jsonl 必须判为不可用 → 重算，不 500 不外泄 NaN
# ════════════════════════════════════════════════════════════════════

def _poison_env(led, monkeypatch):
    LL.ensure_strategy_index(FORMULA, "s.json", "v1", ledger_dir=led)
    _write(led, [_rec(0, pos=1.0, close=100.0), _rec(3600, pos=1.0, close=101.0)])
    calls = []

    def _spy(*a, **k):
        calls.append(1)
        return [], "sentinel-recompute"
    monkeypatch.setattr(LS, "recompute_theo", _spy)
    return calls


def test_theo_poison_cost_rate_string_triggers_recompute(led, monkeypatch):
    calls = _poison_env(led, monkeypatch)
    theo = led / f"{WATCH}.theo.jsonl"
    ledger = led / f"{WATCH}.jsonl"
    theo.write_text(json.dumps({"bar_ts": 0, "position": 1.0, "close": 100.0,
                                 "cost_rate": "abc"}) + "\n", encoding="utf-8")
    assert LS.theo_is_fresh(ledger, theo, FALLBACK_COST) is False
    body = client.get("/api/ledger/summary", params={"watch_id": WATCH}).json()
    assert body["theo_available"] is False and "sentinel" in body["theo_reason"]
    assert calls, "毒缓存必须落到重算路径，绝不能拿毒行当新鲜缓存出数"


def test_theo_poison_missing_bar_ts_and_null_line_unusable(led, monkeypatch):
    calls = _poison_env(led, monkeypatch)
    theo = led / f"{WATCH}.theo.jsonl"
    ledger = led / f"{WATCH}.jsonl"
    theo.write_text(json.dumps({"position": 1.0, "close": 100.0,
                                 "cost_rate": FALLBACK_COST}) + "\n", encoding="utf-8")
    assert LS.load_theo_cache(theo) is None                # 缺 bar_ts → 整份作废
    assert LS.theo_is_fresh(ledger, theo, FALLBACK_COST) is False
    body = client.get("/api/ledger/summary", params={"watch_id": WATCH}).json()
    assert body["theo_available"] is False and calls
    calls.clear()
    theo.write_text("null\n" + json.dumps(
        {"bar_ts": 3600, "position": 1.0, "close": 101.0,
         "cost_rate": FALLBACK_COST}) + "\n", encoding="utf-8")
    assert LS.load_theo_cache(theo) is None                # JSON null 行 → 作废
    r = client.get("/api/ledger/curve", params={"watch_id": WATCH})
    assert r.status_code == 200 and r.json()["theo_available"] is False
    assert calls


def test_theo_poison_nan_bar_ts_treated_unusable(led, monkeypatch):
    calls = _poison_env(led, monkeypatch)
    theo = led / f"{WATCH}.theo.jsonl"
    theo.write_text('{"bar_ts": NaN, "position": 1.0, "close": 100.0, "cost_rate": '
                    + repr(FALLBACK_COST) + "}\n", encoding="utf-8")
    body = client.get("/api/ledger/summary", params={"watch_id": WATCH})
    assert body.status_code == 200
    j = body.json()
    assert j["theo_available"] is False and "sentinel" in j["theo_reason"]
    assert calls, "NaN bar_ts 行不可参与 int()/排序 → 必须触发重算而非出数"
    assert "NaN" not in body.text                          # 投毒 NaN 不外泄
    assert j["completeness"] is None


# ════════════════════════════════════════════════════════════════════
# 6. LedgerRecord / append_record IO 攻击
# ════════════════════════════════════════════════════════════════════

def test_append_ledger_dir_pointing_to_a_file(tmp_path):
    f = tmp_path / "afile.txt"
    f.write_text("do not clobber", encoding="utf-8")
    assert LL.append_record(_rec(1), ledger_dir=f) is False  # 目录位是文件 → False 不抛
    assert LL.append_record(_rec(1), ledger_dir=f / "sub") is False
    assert f.read_text(encoding="utf-8") == "do not clobber"
    assert not (tmp_path / "sub").exists()


def test_append_fields_with_path_chars_stay_inside(led, tmp_path):
    r = _rec(1, source="a/b", symbol="../../evil", tf="..\\..\\x")
    r.strategy_sha = "../../pwned\x00"
    assert LL.append_record(r, ledger_dir=led) is True
    files = list(led.iterdir())
    assert len(files) == 1 and files[0].parent == led
    expect = LL.ledger_path("a/b", "../../evil", "..\\..\\x", ledger_dir=led)
    assert files[0] == expect and expect.parent == led
    import re as _re
    assert _re.fullmatch(r"[A-Za-z0-9_.-]+", files[0].name), files[0].name
    recs, corrupt = LL.read_records(files[0])
    assert corrupt == 0 and recs[0].strategy_sha == "../../pwned\x00"  # 内容原样存证
    assert not list(tmp_path.rglob("evil*")) and not list(tmp_path.rglob("pwned*"))


def test_concurrent_appends_8x50_no_interleaving(led):
    def _worker(t):
        for i in range(50):
            LL.append_record(_rec(t * 100000 + i, pos=t / 8.0), ledger_dir=led)
    threads = [threading.Thread(target=_worker, args=(t,)) for t in range(8)]
    [x.start() for x in threads]
    [x.join() for x in threads]
    p = led / f"{WATCH}.jsonl"
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == 400
    seen = set()
    for ln in lines:
        seen.add(json.loads(ln)["bar_ts"])                 # 每行完整 JSON（锁保证）
    assert len(seen) == 400
    recs, corrupt = LL.read_records(p)
    assert len(recs) == 400 and corrupt == 0


# ════════════════════════════════════════════════════════════════════
# 7. CSV 公式注入：/export 单元格里裸 '='/'@' 前缀
# ════════════════════════════════════════════════════════════════════

def _csv_injection_ledger(led):
    _write(led, [_rec(0, sha="csv123", source="csv", symbol="XAUUSD"),
                 _rec(3600, sha="csv123", source="csv", symbol="XAUUSD", close=101.0)])
    with (led / "csv_XAUUSD_1h.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": 7200.0, "bar_ts": 7200, "source": "csv", "symbol": "=SUM(A1)",
            "timeframe": "1h", "strategy_sha": "csv123", "factor_value": 1.0,
            "position": 0.5, "direction": '@HYPERLINK("http://evil")',
            "strength": 0.5, "close": 102.0}) + "\n")


def test_export_injection_endpoint_never_500(led):
    _csv_injection_ledger(led)
    r = client.get("/api/ledger/export", params={"watch_id": "csv_XAUUSD_1h"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert len(r.text.strip().splitlines()) == 4           # 表头 + 3 行


def test_export_cells_should_be_neutralized_for_spreadsheets(led):
    _csv_injection_ledger(led)
    text = client.get("/api/ledger/export",
                      params={"watch_id": "csv_XAUUSD_1h"}).text
    # 契约（OWASP prefix-' mitigation）：危险前缀单元格必须带单引号前缀落地，
    # 电子表格打开时只会显示字面文本而非执行公式。注：原 xfail 断言用的是
    # 裸子串 `=SUM(A1) not in text`，在 prefix-' 方案下 "=SUM(A1)" 仍是
    # "'=SUM(A1)" 的子串，无法作为判据 → 钉在中和后的完整单元格上。
    assert "'=SUM(A1)" in text, text                      # 公式已中和（' 前缀）
    assert "'@HYPERLINK" in text, text                    # 命令已中和（' 前缀）
    # 逐单元格底线：剥掉中和前缀前，任何单元格都不得以裸 =/+/@ 直达（csv 解析后）
    import csv as _csv
    import io as _io
    for row in _csv.reader(_io.StringIO(text)):
        for cell in row:
            if cell.startswith("'"):
                continue
            assert not cell.startswith(("=", "+", "@")), cell


# ════════════════════════════════════════════════════════════════════
# 8. 命名层攻击（web/strategy_file.py）
# ════════════════════════════════════════════════════════════════════

def test_parse_best_stem_overlong_tf_and_degenerate_names():
    # tf 段 8 字符 > {1,4}：唯一下划线后无法收尾 → 整体不可解析（None），
    # 绝不出现 "XX_1234"+"5678" 这类错位切分。
    assert sf.parse_best_stem("best_XX_12345678") is None
    assert sf.tf_from_best_name(pathlib.Path("best_XX_12345678.json")) is None
    assert sf.parse_best_stem("best_") is None             # "best_.json" 主干
    assert sf.parse_best_stem("best_X_") is None           # "best_X_.json" 主干
    assert sf.parse_best_stem("best_X_H11111") is None     # tf 6 字符
    assert sf.parse_best_stem("best_X_H1") == ("X", "H1")  # 对照：合法
    assert sf.is_legacy_best_name("best_XX_12345678.json") is True


def test_strategy_path_for_symbol_attacks(strategies):
    # 斜杠/空格 symbol：现契约不抛（tf 才被校验），底线是产物必须落在 strategies 内
    p = sf.strategy_path_for_symbol("XAU/USD", "H1")
    assert str(p.resolve()).startswith(str(strategies.resolve())), p   # best_XAU 子目录，未逃逸
    p = sf.strategy_path_for_symbol("XAU USD", "H1")
    assert p.parent == strategies and " " in p.name        # 空格保留（cosmetic）
    with pytest.raises(ValueError):
        sf.strategy_path_for_symbol("XAU", "H11111")       # tf 5 字符 → 拒绝
    with pytest.raises(ValueError):
        sf.strategy_path_for_symbol("XAU", "   ")
    # [A-Za-z0-9]{1,4} 大小写都放行 —— "h1" 合法（契约以正则为准，非「拒绝小写」）
    assert sf.strategy_path_for_symbol("XAU", "h1").name == "best_XAU_h1.json"


def test_strategy_path_symbol_dotdot_stays_contained(strategies):
    """对抗结论（修正 brief 假设）：`f"best_{symbol}_{tf}.json"` 的 best_ 前缀使
    "../" 永远无法成为独立路径段（pathlib 只折叠纯 ".." 段），resolve 仍落在
    strategies 内；绝对路径 symbol 同理被 "best_" 粘住不成绝对路径。
    真正的残留问题仅是会造出 "best_.." / "best_" 这类怪目录层级（cosmetic）。"""
    for sym in ("../escapee", "../../x", "/etc/passwd", ".."):
        p = sf.strategy_path_for_symbol(sym, "H1")
        resolved = p.resolve()
        assert str(resolved).startswith(str(strategies.resolve()) + os.sep), (sym, resolved)
        assert not (strategies.parent / "escapee_H1.json").exists()


def test_sync_payload_tf_mismatch_explicit_wins(strategies):
    # 文件名 H1 但 payload 内 timeframe 写 M30：显式 timeframe 必须赢，
    # 落盘 payload 的 timeframe 归一为 H1，绝不静默保留错配值。
    seed = strategies / "best_XAUUSD_H1.json"
    seed.write_text(json.dumps({"symbol": "XAUUSD", "formula": [1, 2, 3],
                                "best_score": 1.5, "timeframe": "M30"}), encoding="utf-8")
    info = sf.sync_best_strategy_for_symbol("XAUUSD", "H1")
    assert info is not None
    saved = json.loads(seed.read_text(encoding="utf-8"))
    assert saved["timeframe"] == "H1" == info["timeframe"]


def test_sync_tf_mismatch_other_tf_seed_not_copied(strategies):
    # 只有 M30 种子而请求 H1：不得把 M30 载荷伪装成 H1 写出（返回 None、目录不变）。
    seed = strategies / "best_XAUUSD_M30.json"
    seed.write_text(json.dumps({"symbol": "XAUUSD", "formula": [1, 2, 3],
                                "best_score": 1.5, "timeframe": "M30"}), encoding="utf-8")
    before = _files(strategies)
    assert sf.sync_best_strategy_for_symbol("XAUUSD", "H1") is None
    assert not (strategies / "best_XAUUSD_H1.json").exists()
    assert _files(strategies) == before
    assert '"M30"' in seed.read_text(encoding="utf-8")     # 种子未被改写


# ════════════════════════════════════════════════════════════════════
# 9. 双重编码 / 分隔符混淆 watch_id
# ════════════════════════════════════════════════════════════════════

def test_watchid_with_suffix_never_double_suffixed(led):
    _write(led, [_rec(0)])
    for wid in (f"{WATCH}.jsonl", f"{WATCH}.jsonl.jsonl", f"{WATCH}.theo",
                f"{WATCH}.jsonl.theo", f"{WATCH}.json"):
        assert "error" in client.get("/api/ledger/summary",
                                     params={"watch_id": wid}).json(), wid
    # 对照：正确 watch_id 能出数据 → 证明后缀拒绝不是误杀
    assert client.get("/api/ledger/summary", params={"watch_id": WATCH}).json().get("records") == 1


def test_watchid_separator_confusion(led, tmp_path):
    _write(led, [_rec(0)])
    before = _files(tmp_path)
    # 真实账本的等价「路径写法」必须全被拒，而不是被归一化命中
    for wid in ("okx/BTCUSDT/1h", "okx\\BTCUSDT\\1h", "okx//BTCUSDT//1h",
                "okx/BTCUSDT/1h.jsonl", "./okx_BTCUSDT_1h", ".//okx_BTCUSDT_1h",
                "okx_/BTCUSDT_1h", "%6fnames"):
        _err_only(client.get("/api/ledger/summary", params={"watch_id": wid}))
    assert _files(tmp_path) == before
