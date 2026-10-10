# tests/unit/test_strategy_list.py
import json
from pathlib import Path
from web import progress
from web.progress import list_strategies

def test_list_only_new_names(tmp_path, monkeypatch):
    (tmp_path / "best_XAUUSD_H1.json").write_text(json.dumps(
        {"symbol": "XAUUSD", "timeframe": "H1", "formula": [1, 2], "best_score": 0.5}), encoding="utf-8")
    (tmp_path / "best_XAUUSD.json").write_text(json.dumps(
        {"symbol": "XAUUSD", "formula": [1, 2], "best_score": 0.5}), encoding="utf-8")
    monkeypatch.setattr(progress, "STRATEGIES_DIR", tmp_path)
    rows = list_strategies()
    assert len(rows) == 1
    assert rows[0]["timeframe"] == "H1"
    assert rows[0]["symbol"] == "XAUUSD"


def _seed(tmp_path: Path) -> None:
    (tmp_path / "best_Y_H1.json").write_text(json.dumps(
        {"symbol": "Y", "timeframe": "H1", "formula": [1, 2], "best_score": 0.5}), encoding="utf-8")
    (tmp_path / "best_Y.json").write_text(json.dumps(
        {"symbol": "Y", "formula": [1, 2], "best_score": 0.5}), encoding="utf-8")


def test_api_strategies_excludes_legacy(tmp_path, monkeypatch):
    """GET /api/strategies 处理函数可用：只返回新命名行，且行内含 timeframe。

    直接调用端点函数（FastAPI 路由处理器即普通函数），避免依赖 TestClient 所需的
    httpx（本仓库不引入新第三方依赖）；真正的 live 冒烟按控制要求顺延至 Task B6。
    """
    _seed(tmp_path)
    monkeypatch.setattr(progress, "STRATEGIES_DIR", tmp_path)
    from web.app import api_strategies

    rows = api_strategies()["strategies"]
    names = [r["file"] for r in rows]
    assert "best_Y_H1.json" in names
    assert "best_Y.json" not in names
    for r in rows:
        if r["file"] == "best_Y_H1.json":
            assert r["timeframe"] == "H1"


def test_http_client_strategies(tmp_path, monkeypatch):
    """TestClient 版 GET /api/strategies（环境缺 httpx 时跳过，不新增依赖）。"""
    _seed(tmp_path)
    monkeypatch.setattr(progress, "STRATEGIES_DIR", tmp_path)
    try:
        from fastapi.testclient import TestClient
        from web.app import app
    except Exception:
        import pytest
        pytest.skip("TestClient 需要 httpx，本环境未安装且不允许新增依赖")
    try:
        client = TestClient(app)
    except RuntimeError as exc:  # starlette 在实例化时才报缺 httpx
        import pytest
        pytest.skip(f"TestClient 不可用: {exc}")
    resp = client.get("/api/strategies")
    assert resp.status_code == 200
    names = [r["file"] for r in resp.json()["strategies"]]
    assert "best_Y_H1.json" in names
    assert "best_Y.json" not in names
