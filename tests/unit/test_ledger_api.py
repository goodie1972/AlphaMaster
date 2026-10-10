# tests/unit/test_ledger_api.py
"""Task B5：`/api/ledger/*` 五端点（spec §5.6）。

约定：
- 不进入 TestClient 上下文管理器：`@app.on_event("startup")`（web/app.py:1043）会调
  realtime_manager.load_persisted() 起真实网络 watch 线程，测试里绝不能触发。
- 端点必须在调用时读 `LL.LEDGER_DIR`（B1 延迟解析约定），故本文件一律
  monkeypatch 模块属性 + `append_record(..., ledger_dir=tmp_path)`。
- autouse fixture 把 `LS.get_source` 换成会抛异常的假源：recompute_theo 任何
  真实网络入口都被拦住（生产允许、测试禁止）。
"""
from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

import web.ledger_stats as LS
import web.live_ledger as LL
from web.app import app

client = TestClient(app)

# 真实可评估公式（同 tests/unit/test_theo_recompute.py）：sha 要与账本
# strategy_sha 一致，理论重算才会从索引取到 formula。
FORMULA = [25, 81, 61, 103, 93, 104, 31, 115]
SHA = LL.formula_sha(FORMULA)
_OLD_SHA = "aaaa11"


def _rec(bar_ts, pos=0.5, close=100.0, *, sha=SHA, tf="1h",
         source="okx", symbol="BTCUSDT", direction=None):
    return LL.LedgerRecord(
        ts=float(bar_ts), bar_ts=int(bar_ts), source=source, symbol=symbol,
        timeframe=tf, strategy_sha=sha, factor_value=float(pos), position=float(pos),
        direction=direction or ("LONG" if pos > 0 else "FLAT"),
        strength=abs(float(pos)), close=float(close))


def _write(ledger_dir, recs):
    for r in recs:
        assert LL.append_record(r, ledger_dir=ledger_dir)


class _B:
    """最小 bar（同 tests/unit/test_theo_recompute.py 的 fake 源形状）。"""

    def __init__(self, ts, c):
        self.ts = ts
        self.open = self.high = self.low = self.close = c
        self.volume = 1.0


class _FakeSource:
    def __init__(self, bars):
        self.bars = bars

    def fetch_bars(self, symbol, tf, n, drop_forming=True):
        return self.bars[-n:]


def _sine_bars(count, step):
    return [_B(i * step, 100.0 + math.sin(i / 5)) for i in range(count)]


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch):
    """台账端点的理论重算默认走真实数据源 —— 测试里禁止。"""
    def _boom(kind):
        raise AssertionError(f"测试禁止真实数据源: {kind}")
    monkeypatch.setattr(LS, "get_source", _boom)


# ── /api/ledger/watches ──────────────────────────────────────────────
def test_watches_lists_ledgers(tmp_path, monkeypatch):
    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    _write(tmp_path, [_rec(0)])
    rows = client.get("/api/ledger/watches").json()["watches"]
    assert rows[0]["watch_id"] == "okx_BTCUSDT_1h" and rows[0]["records"] == 1


def test_watches_skips_theo_cache_and_reports_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    _write(tmp_path, [_rec(0), _rec(3600, close=101.0)])
    (tmp_path / "okx_BTCUSDT_1h.theo.jsonl").write_text('{"bar_ts": 0}\n', encoding="utf-8")
    (tmp_path / "strategy_index.json").write_text("{}", encoding="utf-8")
    rows = client.get("/api/ledger/watches").json()["watches"]
    assert [r["watch_id"] for r in rows] == ["okx_BTCUSDT_1h"]   # theo/索引不入清单
    assert rows[0]["records"] == 2 and rows[0]["last_bar_ts"] == 3600
    assert rows[0]["source"] == "okx" and rows[0]["timeframe"] == "1h"


def test_watches_missing_dir_is_empty_list(monkeypatch, tmp_path):
    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path / "never-created")
    r = client.get("/api/ledger/watches")
    assert r.status_code == 200 and r.json()["watches"] == []


# ── /api/ledger/summary ──────────────────────────────────────────────
def test_summary_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    for i in range(3):
        _write(tmp_path, [_rec(i * 3600, close=100 + i)])
    r = client.get("/api/ledger/summary",
                   params={"watch_id": "okx_BTCUSDT_1h", "days": 1})
    assert r.status_code == 200
    body = r.json()
    assert {"records", "win_rate", "profit_factor", "annualized", "sortino",
            "max_drawdown", "completeness", "corrupt_lines", "theo_available"} <= body.keys()


def test_summary_days_window_is_anchored_at_last_bar(tmp_path, monkeypatch):
    """截窗以末条 bar_ts 为锚：days=1 → 只留末 24h（含锚点）= 25 条。"""
    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    _write(tmp_path, [_rec(i * 3600, close=100.0 + (i % 3)) for i in range(72)])
    one = client.get("/api/ledger/summary",
                     params={"watch_id": "okx_BTCUSDT_1h", "days": 1}).json()
    three = client.get("/api/ledger/summary",
                       params={"watch_id": "okx_BTCUSDT_1h", "days": 3}).json()
    assert one["records"] == 25 and one["window_start_ts"] == 47 * 3600
    assert one["window_end_ts"] == 71 * 3600
    assert three["records"] == 72
    # 旧记录被排除：30 天窗口与 3 天窗口统计口径一致
    assert client.get("/api/ledger/summary", params={"watch_id": "okx_BTCUSDT_1h"}).json()["records"] == 72


def test_summary_sha_defaults_to_newest_segment(tmp_path, monkeypatch):
    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    old = [_rec(i * 3600, pos=1.0, close=100.0 + i, sha=_OLD_SHA) for i in range(4)]
    new = [_rec((4 + i) * 3600, pos=1.0, close=110.0 - i, sha=SHA) for i in range(4)]
    _write(tmp_path, old + new)
    base = {"watch_id": "okx_BTCUSDT_1h", "days": 1}
    default = client.get("/api/ledger/summary", params=base).json()
    explicit = client.get("/api/ledger/summary", params={**base, "sha": _OLD_SHA}).json()
    assert default["sha"] == SHA and default["records"] == 4
    assert default["win_rate"] == 0.0            # 新段：多头遇跌 → 全亏
    assert explicit["sha"] == _OLD_SHA and explicit["win_rate"] == 1.0
    assert explicit["records"] == 4


def test_summary_corrupt_lines_surfaced(tmp_path, monkeypatch):
    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    _write(tmp_path, [_rec(0), _rec(3600, close=101.0)])
    with (tmp_path / "okx_BTCUSDT_1h.jsonl").open("a", encoding="utf-8") as f:
        f.write("半行损坏 not-json\n")
    body = client.get("/api/ledger/summary",
                      params={"watch_id": "okx_BTCUSDT_1h", "days": 1}).json()
    assert body["corrupt_lines"] == 1 and body["records"] == 2


def test_summary_theo_unavailable_reason_and_completeness_null(tmp_path, monkeypatch):
    """源不可用：theo_available=false + reason 回传，completeness=null（不是 0）。"""
    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    # 先登记索引，让 sha 有 formula → 走到重算（被 fake 的 recompute_theo 接管）
    LL.ensure_strategy_index(FORMULA, "best_BTCUSDT_1h.json", "v1", ledger_dir=tmp_path)
    _write(tmp_path, [_rec(i * 3600, close=100.0 + i) for i in range(4)])
    monkeypatch.setattr(LS, "recompute_theo",
                        lambda *a, **k: ([], "test: 源不可用"))
    body = client.get("/api/ledger/summary",
                      params={"watch_id": "okx_BTCUSDT_1h", "days": 1}).json()
    assert body["theo_available"] is False
    assert "源不可用" in body["theo_reason"]
    assert body["completeness"] is None
    assert body["win_rate"] == 1.0               # 实盘统计照常（spec §6）


def test_summary_missing_sha_index_entry_marks_theo_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    _write(tmp_path, [_rec(i * 3600, close=100.0 + i, sha="dead01") for i in range(3)])
    body = client.get("/api/ledger/summary",
                      params={"watch_id": "okx_BTCUSDT_1h", "days": 1}).json()
    assert body["theo_available"] is False
    assert "公式索引缺失" in body["theo_reason"]
    assert body["completeness"] is None


# ── 终评 F3：cost_rate 必须与 02 回测同源 = (bt_commission_pct + bt_slippage_pct)/100 ──
def test_summary_cost_rate_sourced_from_settings(tmp_path, monkeypatch):
    """settings 有键 → (0.05+0.02)/100=0.0007；缺键 → 兜底 Config.COST_RATE=0.0003。
    选换手 1.0 的两根 bar，让 pnl（及 annualized）对 cost_rate 敏感、两口径可区分。"""
    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    # 行0：pos=1.0→0.0（turnover=1.0），ret=101/100-1=0.01 → pnl=0.01-cost
    _write(tmp_path, [_rec(0, pos=1.0, close=100.0), _rec(3600, pos=0.0, close=101.0)])
    monkeypatch.setattr("web.app.load_settings",
                        lambda: {"bt_commission_pct": 0.05, "bt_slippage_pct": 0.02})
    s = client.get("/api/ledger/summary",
                   params={"watch_id": "okx_BTCUSDT_1h", "days": 1}).json()
    monkeypatch.setattr("web.app.load_settings", lambda: {})
    f = client.get("/api/ledger/summary",
                   params={"watch_id": "okx_BTCUSDT_1h", "days": 1}).json()
    ppy = 2 / (3600 / (365.25 * 86400.0))          # 密度口径：2 行 / 1h 跨度
    assert s["annualized"] == pytest.approx((0.01 - 0.0007) * ppy, rel=1e-6)
    assert f["annualized"] == pytest.approx((0.01 - 0.0003) * ppy, rel=1e-6)
    assert s["annualized"] != f["annualized"]


def test_summary_cost_rate_falls_back_on_unparseable_settings(tmp_path, monkeypatch):
    """键存在但不可解析 → 兜底 Config.COST_RATE，绝不抛出、绝不打印 settings。"""
    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    _write(tmp_path, [_rec(0, pos=1.0, close=100.0), _rec(3600, pos=0.0, close=101.0)])
    monkeypatch.setattr("web.app.load_settings",
                        lambda: {"bt_commission_pct": "abc", "bt_slippage_pct": 0.02})
    body = client.get("/api/ledger/summary",
                      params={"watch_id": "okx_BTCUSDT_1h", "days": 1}).json()
    ppy = 2 / (3600 / (365.25 * 86400.0))
    assert body["annualized"] == pytest.approx((0.01 - 0.0003) * ppy, rel=1e-6)


# ── 终评 F5：theo 有行但跨度覆盖不足（源截断/退市）→ completeness=null 而非 1.0 ──
def test_summary_completeness_null_on_partial_theo_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    LL.ensure_strategy_index(FORMULA, "best_BTCUSDT_1h.json", "v1", ledger_dir=tmp_path)
    _write(tmp_path, [_rec(0, pos=1.0, close=100.0), _rec(3600, pos=1.0, close=101.0)])
    # 假重算：只回窗口末根一根 theo（首根 3600 ≫ start+tf_s → 跨度覆盖不足）
    partial = [{"bar_ts": 3600, "close": 101.0, "position": 0.5, "ret": None,
                "turnover": None, "pnl": None, "missing": True}]
    monkeypatch.setattr(LS, "recompute_theo", lambda *a, **k: (partial, None))
    body = client.get("/api/ledger/summary",
                      params={"watch_id": "okx_BTCUSDT_1h", "days": 1}).json()
    assert body["theo_available"] is True and body["theo_bars"] == 1
    assert body["completeness"] is None            # 覆盖不足 → null（此前恒为 1.0）


def test_summary_days_range_is_200_error(tmp_path, monkeypatch):
    """跨度上限保护：theo 重算是 O(窗口 bar × 1001 bar VM) 懒算（B4 性能告诫）。"""
    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    _write(tmp_path, [_rec(0)])
    for bad in (0, -1, 99999):
        r = client.get("/api/ledger/summary",
                       params={"watch_id": "okx_BTCUSDT_1h", "days": bad})
        assert r.status_code == 200 and "days" in r.json().get("error", ""), bad


def test_summary_unknown_or_traversal_watch_id_is_200_error(tmp_path, monkeypatch):
    live = tmp_path / "live"
    live.mkdir()
    monkeypatch.setattr(LL, "LEDGER_DIR", live)
    # 派生缓存真实存在，但仍不能当账本被访问
    (live / "okx_BTCUSDT_1h.theo.jsonl").write_text('{"bar_ts": 0}\n', encoding="utf-8")
    (tmp_path / "secret.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    for bad in ("nope", "a/../b", "../secret", "okx_BTCUSDT_1h.theo", "", "okx/../secret"):
        r = client.get("/api/ledger/summary", params={"watch_id": bad})
        assert r.status_code == 200, bad
        assert "error" in r.json(), bad
        assert (tmp_path / "secret.jsonl").read_text(encoding="utf-8") == '{"x": 1}\n'


# ── /api/ledger/curve ────────────────────────────────────────────────
def test_curve_shape_and_cumulative(tmp_path, monkeypatch):
    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    LL.ensure_strategy_index(FORMULA, "best_BTCUSDT_1h.json", "v1", ledger_dir=tmp_path)
    _write(tmp_path, [_rec(i * 3600, pos=1.0, close=100.0 + i) for i in range(4)])
    monkeypatch.setattr(LS, "recompute_theo", lambda *a, **k: ([], "test: 源不可用"))
    body = client.get("/api/ledger/curve",
                      params={"watch_id": "okx_BTCUSDT_1h", "days": 1}).json()
    pts = body["curve"]
    assert len(pts) == 4
    assert set(pts[0]) == {"bar_ts", "cum_live", "cum_theo", "theo_missing"}
    assert body["theo_available"] is False and "源不可用" in body["theo_reason"]
    assert all(p["theo_missing"] is True and p["cum_theo"] is None for p in pts)
    assert pts[0]["cum_live"] == pytest.approx(0.01)          # 1.0×(101/100-1)
    expect = 0.01 + (102.0 / 101.0 - 1.0) + (103.0 / 102.0 - 1.0)
    assert pts[2]["cum_live"] == pytest.approx(expect)
    assert pts[3]["cum_live"] == pytest.approx(expect)        # 末条 missing → 累计不前移


def test_curve_theo_real_chain_joins_by_bar_ts(tmp_path, monkeypatch):
    """一条真实链路（fake 源 + 真 evaluate_signal）：theo 侧按 bar_ts 对齐。"""
    day = 86400
    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    monkeypatch.setattr(LS, "get_source", lambda kind: _FakeSource(_sine_bars(1120, day)))
    LL.ensure_strategy_index(FORMULA, "best_BTCUSDT_1d.json", "v1", ledger_dir=tmp_path)
    _write(tmp_path, [_rec(i * day, tf="1d", sha=SHA) for i in range(1117, 1120)])

    summary = client.get("/api/ledger/summary",
                         params={"watch_id": "okx_BTCUSDT_1d", "days": 1}).json()
    assert summary["theo_available"] is True and summary["theo_reason"] is None
    assert summary["records"] == 2
    assert summary["completeness"] == pytest.approx(1.0)   # 2 条 ÷ 源 2 根收盘 bar
    assert (tmp_path / f"okx_BTCUSDT_1d.{SHA}.theo.jsonl").exists()

    curve = client.get("/api/ledger/curve",
                       params={"watch_id": "okx_BTCUSDT_1d", "days": 1}).json()["curve"]
    assert [p["bar_ts"] for p in curve] == [1118 * day, 1119 * day]
    assert all(p["theo_missing"] is False and p["cum_theo"] is not None for p in curve)

    # 第二次请求必须命中 .theo.jsonl（跨度覆盖检查通过）：把重算打成「绝不该被调用」
    monkeypatch.setattr(LS, "recompute_theo", lambda *a, **k: ([], "test: 不该重算"))
    again = client.get("/api/ledger/curve",
                       params={"watch_id": "okx_BTCUSDT_1d", "days": 1}).json()
    assert again["theo_available"] is True and again["theo_reason"] is None
    assert [p["cum_theo"] for p in again["curve"]] == [p["cum_theo"] for p in curve]


# ── /api/ledger/records ─────────────────────────────────────────────
def test_records_pagination_and_direction_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    recs = [_rec(i * 3600, pos=p, close=100.0 + i,
                 direction=("LONG" if p > 0 else "SHORT"))
            for i, p in enumerate([1.0, 1.0, -1.0, 1.0, -1.0, -1.0])]
    _write(tmp_path, recs)
    base = {"watch_id": "okx_BTCUSDT_1h"}

    page = client.get("/api/ledger/records", params={**base, "limit": 2}).json()
    assert page["total"] == 6 and len(page["records"]) == 2
    assert [r["bar_ts"] for r in page["records"]] == [5 * 3600, 4 * 3600]  # 新→旧
    assert {"strategy_sha", "position", "close", "pnl", "direction"} <= page["records"][0].keys()

    page1 = client.get("/api/ledger/records", params={**base, "limit": 2, "offset": 2}).json()
    assert [r["bar_ts"] for r in page1["records"]] == [3 * 3600, 2 * 3600]

    shorts = client.get("/api/ledger/records", params={**base, "direction": "SHORT"}).json()
    assert shorts["total"] == 3
    assert {r["direction"] for r in shorts["records"]} == {"SHORT"}
    # 派生 pnl 与 derive_rows 同口径（相邻才计）
    assert page["records"][0]["pnl"] is None      # 末条无 next → missing
    assert page["records"][1]["pnl"] == pytest.approx(-1.0 * (105.0 / 104.0 - 1.0))


def test_records_days_window_optional(tmp_path, monkeypatch):
    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    _write(tmp_path, [_rec(i * 3600) for i in range(100)])   # 跨度 99h > 1d
    base = {"watch_id": "okx_BTCUSDT_1h", "limit": 500}
    assert client.get("/api/ledger/records", params=base).json()["total"] == 100
    assert client.get("/api/ledger/records",
                      params={**base, "days": 1}).json()["total"] == 25


# ── /api/ledger/export ──────────────────────────────────────────────
def test_export_csv_header_and_disposition(tmp_path, monkeypatch):
    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    _write(tmp_path, [_rec(i * 3600, pos=1.0, close=100.0 + i) for i in range(3)])
    r = client.get("/api/ledger/export",
                   params={"watch_id": "okx_BTCUSDT_1h", "days": 1})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    disp = r.headers["content-disposition"]
    assert 'attachment' in disp and 'filename="okx_BTCUSDT_1h_1d.csv"' in disp
    lines = r.text.strip().splitlines()
    assert lines[0] == ("ts,bar_ts,source,symbol,timeframe,strategy_sha,"
                        "factor_value,position,direction,strength,close,pnl")
    assert len(lines) == 4                        # 表头 + 窗口内 3 行
    assert lines[1].startswith("0.0,0,okx,BTCUSDT,1h,")


def test_export_unknown_watch_is_200_json_error(tmp_path, monkeypatch):
    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    r = client.get("/api/ledger/export", params={"watch_id": "nope", "days": 1})
    assert r.status_code == 200 and "error" in r.json()
