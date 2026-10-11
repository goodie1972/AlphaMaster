"""首页 HTML 必须禁缓存：?v=NN 资源版本号升级后，若 index.html 本身被浏览器
启发式缓存（重启浏览器不清磁盘缓存，普通导航直接吃缓存），用户会永远加载旧
app.js —— 表现为「新版已上线但界面还是旧版」。no-store 保证每次导航都拉新 HTML，
新 HTML 里的 ?v= 再驱动 JS 缓存失效。"""

from web.app import app
from fastapi.testclient import TestClient

client = TestClient(app)   # 裸用，不进 startup


def test_index_html_should_forbid_browser_caching():
    r = client.get("/")
    assert r.status_code == 200
    cc = (r.headers.get("cache-control") or "").lower()
    assert "no-store" in cc or "no-cache" in cc, f"Cache-Control 缺失: {cc!r}"


def test_index_html_should_carry_build_stamp():
    # 「看到的是不是新构建」必须一条命令可证：GET / 回 X-AM-Build（git 短 sha，
    # 拿不到 git 时为 "dev"，但绝不允许缺失）。
    r = client.get("/")
    b = r.headers.get("x-am-build")
    assert b, f"X-AM-Build 缺失: {b!r}"
    assert b == "dev" or (4 <= len(b) <= 40 and b.isalnum()), f"X-AM-Build 异常: {b!r}"
