# AlphaMaster 策略文件名带周期 + 实盘信号台账 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 策略 JSON 文件名带上训练周期（根治跨周期同名覆盖），并新增实盘信号台账（逐 bar 落盘、同回测口径统计、同段离线重算理论曲线、04 对照页）。

**Architecture:** Part A 改造 best_{symbol}.json 的命名/解析/读写全链路为 best_{symbol}_{tf}.json，只认新命名。Part B 在 realtime_manager 评估循环内嵌追加式 jsonl 台账（每 watch 一文件、bar_ts 去重、重启从末行恢复），统计与理论曲线全部懒算：`web/live_ledger.py` 只做 IO/索引，`web/ledger_stats.py` 做衍生统计与重算缓存，FastAPI 新增 `/api/ledger/*`，前端加 04 页。

**Tech Stack:** Python 3.11（`.venv/Scripts/python.exe`）、FastAPI、pytest（tests/unit 已有骨架）、零新依赖。

**Spec:** `docs/superpowers/specs/2026-10-09-live-signal-ledger-design.md`（Part B）；Part A 依据 2026-10-08 已批准设计（本文档 §Part A 头部复述决议）。

## Global Constraints

- 一切测试用 `.venv/Scripts/python.exe -m pytest`（在仓库根目录运行）。
- **台账失败绝不允许中断信号评估主链路**（spec §6 总原则）。
- 不引入任何新第三方依赖；不使用 sqlite。
- Part A「不兼容旧命名」：旧 `best_{symbol}.json` 不出现在列表、watch 引用即报错；**不加任何回退读取**。
- 对齐/去重只用 `bar_ts`（源时间戳），本地 `ts` 仅审计。
- 记账收益相邻性规则：`bar_ts_next − bar_ts == tf_seconds` 才计 pnl，否则记 missing（spec §5.5）。
- 周期字符串保持各域原样（parquet 域 `H1`，realtime 域 `1h`），**不做大小写转换**。
- 本仓库当前**不是 git 库**。Task 0 需征得用户同意后 `git init`；用户不同意则把所有 Commit 步骤替换为「跑通该任务全部测试」作为检查点。

---

### Task 0: 仓库基线（git init + 忽略项）

**Files:**
- Create: `.gitignore`（若不存在则新建，存在则追加下列条目）

- [ ] **Step 1: 征得用户同意初始化 git**（一次询问，原话引用 Global Constraints 中说明）
- [ ] **Step 2: 初始化并提交现状**

```bash
git init
cat >> .gitignore <<'EOF'
.venv/
__pycache__/
*.pyc
.env
live_ledger/
web_settings.json
checkpoints/
backtest_output/
logs/
EOF
git add -A && git commit -m "chore: baseline before tf-naming + live ledger work"
```

预期：`git status` 干净；`.venv`、`live_ledger/` 未被跟踪。

---

## Part A：策略文件名带周期（已批准设计，摘要）

命名 `best_{symbol}_{tf}.json`，tf 按来源域原样（H1/1h…），JSON 增加 `"timeframe"` 字段；写方两处（model_core/engine.py 训练直写、web/strategy_file.py sync）；只认新命名。ckpt/training_history 不在本次范围。

### Task A1: 命名与解析工具（web/strategy_file.py）

**Files:**
- Modify: `web/strategy_file.py`（`_BEST_NAME_RE`、`strategy_path_for_symbol`、`symbol_from_strategy_path`，约 :19-40）
- Test: `tests/unit/test_strategy_naming.py`

**Interfaces:**
- Produces:
  - `parse_best_stem(stem: str) -> tuple[str, str] | None`（返回 (symbol, tf)，不匹配返回 None）
  - `strategy_path_for_symbol(symbol: str, timeframe: str) -> Path`（tf 空或非法 → `ValueError`）
  - `tf_from_best_name(path: Path) -> str | None`

- [ ] **Step 1: 写失败测试**

```python
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
```

- [ ] **Step 2: 跑测试确认失败** — `.venv/Scripts/python.exe -m pytest tests/unit/test_strategy_naming.py -v`，预期 ImportError 或断言失败
- [ ] **Step 3: 最小实现**

```python
_TF_PART_RE = re.compile(r"^(?P<sym>.+)_(?P<tf>[A-Za-z0-9]{1,4})$")

def parse_best_stem(stem: str) -> tuple[str, str] | None:
    if not stem.lower().startswith("best_"):
        return None
    m = _TF_PART_RE.match(stem[len("best_"):])
    return (m.group("sym"), m.group("tf")) if m else None

def strategy_path_for_symbol(symbol: str, timeframe: str) -> Path:
    tf = (timeframe or "").strip()
    if not tf or not re.fullmatch(r"[A-Za-z0-9]{1,4}", tf):
        raise ValueError(f"无效周期，拒绝写无周期策略名: {timeframe!r}")
    return STRATEGIES_DIR / f"best_{symbol}_{tf}.json"
```

同时把 `_BEST_NAME_RE` 改为只匹配 `^best_(.+)_[A-Za-z0-9]{1,4}\.json$`（大小写不敏感），`symbol_from_strategy_path` 复用 `parse_best_stem` 返回 (symbol, tf)。

- [ ] **Step 4: 跑测试确认通过** — 同 Step 2，预期 3 passed
- [ ] **Step 5: Commit** — `git add web/strategy_file.py tests/unit/test_strategy_naming.py && git commit -m "feat(strategies): require timeframe in best_ file names"`

### Task A2: 训练引擎按周期写策略（model_core/engine.py）

**Files:**
- Modify: `model_core/engine.py:56-64`（`_strategy_file_for_symbol`）、`:1249-1268`（strategy_data 组装）
- Test: `tests/unit/test_engine_strategy_path.py`

**Interfaces:**
- Consumes: A1 的命名规则（字符串层面一致，不 import web 层）
- Produces: `_strategy_file_for_symbol(symbol, timeframe)`；tf 解析失败时**跳过落盘并 warning，训练继续**

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_engine_strategy_path.py
from model_core.engine import _strategy_file_for_symbol

def test_path_includes_tf():
    assert _strategy_file_for_symbol("XAUUSD", "H1").endswith("best_XAUUSD_H1.json")

def test_missing_tf_returns_none():
    assert _strategy_file_for_symbol("XAUUSD", None) is None
    assert _strategy_file_for_symbol(None, "H1") is None
```

- [ ] **Step 2: 跑测试确认失败** — `.venv/Scripts/python.exe -m pytest tests/unit/test_engine_strategy_path.py -v`
- [ ] **Step 3: 实现**：`_strategy_file_for_symbol` 加第二个参数，`(symbol and tf)` 才返回路径，否则 None。在 strategy_data 组装处（:1249 附近，`tf = getattr(self, "timeframe", None) or strategy_data.get("timeframe") or _fallback_data_file_for_symbol(...)[1]`）：

```python
save_path = _strategy_file_for_symbol(self.target_symbol, tf)
if save_path is None:
    logger.warning("[engine] 无法确定周期，跳过 best_ 策略落盘（checkpoint 不受影响）")
else:
    strategy_data["timeframe"] = tf
    # 原有原子写逻辑不变，写入 save_path
```

（`logger` 若 engine 内没有，用 `print` 前缀 `[engine]`，与文件现状一致；实施时 grep engine.py 内既有日志方式并沿用。）

- [ ] **Step 4: 跑测试确认通过** + `.venv/Scripts/python.exe -m pytest tests/unit -v` 全绿
- [ ] **Step 5: Commit** — `git commit -am "feat(engine): write best_{symbol}_{tf}.json, skip when tf unknown"`

### Task A3: Web 读写链路适配（progress / settings / app / training_package）

**Files:**
- Modify: `web/progress.py`（`:108` 策略读取、`:224` list_strategies、get_symbol_progress 增 timeframe 参数、`:233` 读出 tf）
- Modify: `web/settings.py:99`（浏览默认列表 glob 过滤新命名）
- Modify: `web/app.py`（`/api/strategies` 增 timeframe、sync/训练启动调用点传 tf、`:1024` strategy_path_for_symbol 调用）
- Modify: `web/strategy_file.py`（sync 写 tf；tf 未知 → raise ValueError）
- Modify: `web/training_package.py:79`（导出清单/导入路径按新命名）
- Test: `tests/unit/test_strategy_list.py`

**Interfaces:**
- Consumes: A1 的 `parse_best_stem`/`strategy_path_for_symbol`
- Produces: `get_symbol_progress(symbol: str, timeframe: str | None = None, ...)`；list_strategies 行内含 `"timeframe"`

- [ ] **Step 1: 写失败测试**

```python
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
```

- [ ] **Step 2: 跑测试确认失败** — 预期 rows 有 2 条或 KeyError "timeframe"
- [ ] **Step 3: 实现**：`list_strategies` 内用 `parse_best_stem(path.stem)` 过滤并注入 timeframe；`get_symbol_progress` 增加 `timeframe` 参数，内部策略读取改 `strategy_path_for_symbol(symbol, timeframe)`（tf 为 None 时策略字段置空而非读旧文件）；`strategy_file.py` 的 sync：payload 增 `"timeframe": tf`，tf 来自 ckpt meta `timeframe` → 数据文件解析（`_resolve_data_file_for_symbol` 已返回 tf），拿不到则 raise；`settings.py` glob 过滤 `parse_best_stem(...) is not None`；`app.py` 调用点从数据文件/请求参数解析 tf 后传入（数据文件名 `{symbol}_{tf}.parquet`，用 `inspect_parquet_file` 的 timeframe）；`training_package.py:79` 改 `f"strategies/best_{symbol}_{tf}.json"`（tf 从 ckpt meta 读，缺失 raise）。
- [ ] **Step 4: 跑测试确认通过** — `.venv/Scripts/python.exe -m pytest tests/unit -v` 全绿
- [ ] **Step 5: 手工冒烟** — 重启 Web（`.venv/Scripts/python.exe run_web.py --port 8765`），`curl -s http://127.0.0.1:8765/api/strategies` 应返回 `[]`（旧文件被排除，属预期）；页面 01 不报错
- [ ] **Step 6: Commit** — `git commit -am "feat(web): strategies keyed by symbol+timeframe, legacy names excluded"`

### Task A4: 实时 watch 校验与前端标签（含旧文件处理）

**Files:**
- Modify: `web/realtime_manager.py`（`_load_strategy` 校验新命名；`add_watch` 拒绝旧名并报清晰错误）
- Modify: `web/static/index.html` / `web/static/app.js`（策略下拉、watch 卡片显示「品种 · 周期」；实施时 `grep -n "strategy" web/static/app.js | head -30` 找渲染点）
- Test: `tests/unit/test_watch_validation.py`

**Interfaces:**
- Consumes: A1 `parse_best_stem`
- Produces: `add_watch(...)` 对旧命名 raise `ValueError("旧命名策略不可用：文件名需含周期，请用新命名重新导出")`

- [ ] **Step 1: 写失败测试**

```python
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
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**：`inspect_strategy_file` 开头按 stem 调 `parse_best_stem`，None → `{"error": "legacy_naming", ...}`；`add_watch`/`_load_strategy` 收到该 error 即 raise ValueError（文案见 Interfaces）；前端两处列表 label 改 `${symbol} · ${timeframe}`（timeframe 来自 A3 的 list 字段），卡片同名字段回退显示 `—`。
- [ ] **Step 4: 跑测试确认通过** + 全套 `.venv/Scripts/python.exe -m pytest tests/unit -v`
- [ ] **Step 5: 旧资产处置（询问用户后执行，二选一）**
  - a) 用户同意手工迁移：读 `logs/train_XAUUSD_*.log` 确认周期（当前在跑 H1），把 `strategies/best_XAUUSD.json` 改名 `best_XAUUSD_H1.json` 并在 JSON 补 `"timeframe": "H1"`；或
  - b) 不动，等下轮训练 sync 自然产出新文件（旧实时中断可接受）。
- [ ] **Step 6: Commit** — `git commit -am "feat(realtime): reject legacy strategy naming; show timeframe in UI"`

---

## Part B：实盘信号台账（实现 spec §5–§8）

**Interfaces 总览（B 部分共享契约）：**
- `web/live_ledger.py`：IO 层（LedgerRecord / append / read / last_bar_ts / strategy_index）
- `web/ledger_stats.py`：统计 + 理论重算 + 缓存（spec §5.1 的单模块按单一职责拆成两个文件，API 与格式不变）
- realtime_manager 只调 live_ledger；app.py 只调 ledger_stats

### Task B1: 台账 IO 核心（web/live_ledger.py）

**Files:**
- Create: `web/live_ledger.py`
- Test: `tests/unit/test_live_ledger_io.py`

**Interfaces:**
- Produces:
  - `LedgerRecord`（dataclass，字段=spec §5.2：ts,bar_ts,source,symbol,timeframe,strategy_sha,factor_value,position,direction,strength,close）
  - `formula_sha(formula: list[int]) -> str`
  - `ledger_path(source, symbol, timeframe, *, ledger_dir=LEDGER_DIR) -> Path`
  - `append_record(record, *, ledger_dir=LEDGER_DIR) -> bool`
  - `read_records(path) -> tuple[list[LedgerRecord], int]`（第二项 corrupt_lines）
  - `last_bar_ts(path) -> int | None`
  - `ensure_strategy_index(formula, strategy_file, vocab_version, *, ledger_dir=LEDGER_DIR) -> str`（返回 sha）
  - `load_strategy_index(ledger_dir=LEDGER_DIR) -> dict`

- [ ] **Step 1: 写失败测试**

```python
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
    ensure_strategy_index([1, 2, 3], "s.json", "v9", ledger_dir=tmp_path)
    idx = load_strategy_index(tmp_path)
    assert sha1 == formula_sha([1, 2, 3]) and len(idx) == 1

def last_good(tmp_path):
    from web.live_ledger import ledger_path
    return ledger_path("okx", "BTCUSDT", "1h", ledger_dir=tmp_path)
```

- [ ] **Step 2: 跑测试确认失败** — `.venv/Scripts/python.exe -m pytest tests/unit/test_live_ledger_io.py -v` 预期 ModuleNotFoundError
- [ ] **Step 3: 实现** — 按 Interfaces 写 `web/live_ledger.py`；关键代码已在设计段确认：`append_record` 用 `threading.Lock` + 追加模式整行 `json.dumps(asdict(record), ensure_ascii=False) + "\n"`，`except OSError: return False`；`read_records` 逐行 try/except 计 corrupt，字段用 `__dataclass_fields__` 白名单取；`ledger_path` 文件名 `{safe(source)}_{safe(symbol)}_{safe(timeframe)}.jsonl`，`safe = re.sub(r"[^A-Za-z0-9_.-]", "_", s)`；`formula_sha = hashlib.sha1(",".join(map(str, formula)).encode()).hexdigest()[:6]`；index 更新用 tmp+replace 原子写。`LEDGER_DIR = Path(__file__).resolve().parents[1] / "live_ledger"`。
- [ ] **Step 4: 跑测试确认通过** — 5 passed
- [ ] **Step 5: Commit** — `git add web/live_ledger.py tests/unit/test_live_ledger_io.py && git commit -m "feat(ledger): jsonl ledger io with strategy index"`

### Task B2: realtime_manager 接入（去重 + 重启恢复）

**Files:**
- Modify: `web/realtime_manager.py`（WatchTask 字段 ~:149；`_evaluate_task` ok 分支 ~:377；`_add_task_internal` ~:235 恢复）
- Test: `tests/unit/test_ledger_wiring.py`

**Interfaces:**
- Consumes: B1 `append_record/read_records/last_bar_ts/ensure_strategy_index/formula_sha`
- Produces: `WatchTask.last_ledger_bar_ts: int | None`；评估成功且 bar 前进时账本 +1 行

- [ ] **Step 1: 写失败测试**（不起线程，直接调 `_evaluate_task`，monkeypatch `_get_bars` 喂合成 bars）

```python
# tests/unit/test_ledger_wiring.py
import time, torch
from pathlib import Path
import web.live_ledger as LL
from web.realtime_manager import RealtimeManager, WatchTask

class _B:
    def __init__(self, ts): self.ts=ts; self.open=self.high=self.low=self.close=100.0; self.volume=1.0

def _mk_mgr(tmp_path, monkeypatch):
    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    mgr = RealtimeManager()
    task = WatchTask(id="okx:BTCUSDT:1h:x", source="okx", symbol="BTCUSDT",
                     timeframe="1h", strategy_file="s.json", strategy_name="s",
                     formula=[1], vocab_version="v", strategy_symbol=None,
                     strategy_timeframe=None, best_score=None, cadence_s=60)
    mgr._tasks[task.id] = task
    return mgr, task

def test_one_row_per_closed_bar(tmp_path, monkeypatch):
    mgr, task = _mk_mgr(tmp_path, monkeypatch)
    bars = [_B(1000), _B(4600)]        # 1h 间隔
    monkeypatch.setattr(mgr, "_get_bars", lambda *a: bars)
    monkeypatch.setattr("web.realtime_manager.evaluate_signal",
                        lambda f, raw, **k: {"state":"ok","direction":"up","strength":0.5,
                                              "position":0.5,"factor_value":1.0,"bars_used":len(raw['close'][0])})
    mgr._evaluate_task(task); mgr._evaluate_task(task)     # 同一最后一根 bar 评估两次
    assert last_bar_ts_count(tmp_path) == 1
    bars.append(_B(8200))                                   # 新收盘 bar
    mgr._evaluate_task(task)
    assert last_bar_ts_count(tmp_path) == 2
    assert task.last_ledger_bar_ts == 8200

def last_bar_ts_count(tmp_path):
    recs, _ = LL.read_records(LL.ledger_path("okx", "BTCUSDT", "1h", ledger_dir=tmp_path))
    return len(recs)
```

- [ ] **Step 2: 跑测试确认失败**（账本 0 行）
- [ ] **Step 3: 实现**：WatchTask 加 `last_ledger_bar_ts: int | None = None`；ok 分支末尾：

```python
if task.last_ledger_bar_ts is None or last_ts > task.last_ledger_bar_ts:
    sha = LL.ensure_strategy_index(task.formula, task.strategy_file, task.vocab_version)
    rec = LL.LedgerRecord(ts=time.time(), bar_ts=int(last_ts), source=task.source,
                          symbol=task.symbol, timeframe=task.timeframe, strategy_sha=sha,
                          factor_value=float(result["factor_value"]), position=float(result["position"]),
                          direction=str(result["direction"]), strength=float(result["strength"]),
                          close=float(bars[-1].close))
    if LL.append_record(rec):
        task.last_ledger_bar_ts = int(last_ts)
    else:
        logger.warning("[ledger] 写入失败 bar_ts=%s（下个 tick 重试）", last_ts)
```

整段包 `except Exception: logger.warning(...)`（原则：不打断信号）。`_add_task_internal` 建 task 后：`task.last_ledger_bar_ts = LL.last_bar_ts(LL.ledger_path(source, symbol, timeframe))`。
- [ ] **Step 4: 跑测试确认通过** — 2 passed；再全套回归
- [ ] **Step 5: Commit** — `git commit -am "feat(realtime): append ledger record per closed bar"`

### Task B3: 统计纯函数（web/ledger_stats.py）

**Files:**
- Create: `web/ledger_stats.py`
- Test: `tests/unit/test_ledger_stats.py`

**Interfaces:**
- Consumes: B1 `LedgerRecord`；`strategy_manager.live_signal.min_exposure`
- Produces:
  - `derive_rows(records: list[LedgerRecord], tf: str) -> list[dict]`，每行 `{bar_ts, close, position, ret, turnover, pnl, missing: bool}`（口径：`ret=close_next/close-1`，`turnover=|pos_next-pos|`，`pnl=pos*ret-turnover*cost`，cost 由参数传入；不相邻 → missing=True，ret/pnl=None）
  - `compute_stats(rows, cost_rate, *, tf, first_ts, last_ts) -> dict`：`win_rate`（`|pos|>=min_exp` 且非 missing 的 `pos*ret>0` 占比）、`profit_factor`（含成本 |盈利均值|÷|亏损均值|，无亏损返回 None）、`annualized`、`sortino`（镜像 model_core/backtest.py:147 公式：下行 std 地板=0.2×全 std，clip ±20）、`max_drawdown`（累计 log pnl）、`live_bars`、`theoretical_bars`（供完整率分子）

- [ ] **Step 1: 写失败测试**（手算样例钉死边界）

```python
# tests/unit/test_ledger_stats.py
from web.ledger_stats import derive_rows, compute_stats
from web.live_ledger import LedgerRecord

def R(bar_ts, pos, close):
    return LedgerRecord(ts=0.0, bar_ts=bar_ts, source="okx", symbol="BTCUSDT",
                        timeframe="1h", strategy_sha="x", factor_value=pos,
                        position=pos, direction="up" if pos > 0 else "down",
                        strength=abs(pos), close=close)

def test_adjacency_and_costs():
    recs = [R(0, 0.5, 100.0), R(3600, 0.5, 102.0), R(10800, 0.0, 101.0)]  # 第二段起不相邻
    rows = derive_rows(recs, tf="1h", cost_rate=0.001)
    assert rows[0]["missing"] is False and abs(rows[0]["ret"] - 0.02) < 1e-12
    assert abs(rows[0]["pnl"] - (0.5 * 0.02)) < 1e-12          # 无换手，不扣成本
    assert rows[1]["missing"] is True                           # 7200s ≠ 3600s
    s = compute_stats(rows, cost_rate=0.001, tf="1h")
    assert s["win_rate"] == 1.0                                 # 唯一有效样本：0.5×0.02>0
    assert s["profit_factor"] is None                           # 无亏损样本

def test_turnover_cost():
    rows = derive_rows([R(0, 0.5, 100.0), R(3600, -0.5, 100.0)], tf="1h", cost_rate=0.001)
    assert abs(rows[0]["pnl"] - (0.5 * 0.0 - 1.0 * 0.001)) < 1e-12  # ret=0, 换手=1.0
```

（`derive_rows` 签名含 `cost_rate`——成本属于 pnl 定义，测试即规格。）

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现** — `derive_rows` 按 bar_ts 升序、`_TF_SECONDS` 字典取周期秒（"1m":60…"1h":3600…"4h":14400，"1d":86400，"1w":604800；未知 tf raise ValueError）；`compute_stats` 用纯 Python+math，ppy=`365.25*86400/tf_seconds`。
- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: Commit** — `git commit -am "feat(ledger): backtest-consistent stats (win rate, PF, sortino, mdd)"`

### Task B4: 理论曲线重算 + 缓存 + 口径恒等式

**Files:**
- Modify: `web/ledger_stats.py`（追加 theo 部分）
- Test: `tests/unit/test_theo_recompute.py`

**Interfaces:**
- Consumes: `web.data_sources.factory.get_source`、`web.realtime_manager.bars_to_raw_dict`（实施时确认该函数所在模块并按 import 使用）、`strategy_manager.live_signal.evaluate_signal(formula, raw, require_closed=False)`（签名见 live_signal.py:65）、B3 `derive_rows`
- Produces: `recompute_theo(source, symbol, tf, formula, *, start_ts, end_ts, cost_rate, ledger_dir) -> tuple[list[dict], str | None]`（rows 同 derive_rows 结构；第二项 unavailable 原因或 None）；`theo_is_fresh(ledger_path, theo_path, cost_rate) -> bool`；缓存文件 `{stem}.theo.jsonl` 每行 `{bar_ts, position, close, cost_rate}`

- [ ] **Step 1: 写失败测试**（恒等式：同一合成序列，「逐根 evaluate_signal 记账」与「整段重算」position 全等）

```python
# tests/unit/test_theo_recompute.py
import math
import web.ledger_stats as LS

class _B:
    def __init__(s, ts, c): s.ts=ts; s.open=s.high=s.low=s.close=c; s.volume=1.0

class _FakeSource:
    def __init__(s, bars): s.bars=bars
    def fetch_bars(s, symbol, tf, n, drop_forming=True): return s.bars[-n:]

def test_identity_recompute_matches_live_formula(tmp_path, monkeypatch):
    bars = [_B(i*3600, 100.0 + math.sin(i/5)) for i in range(60)]
    formula = [31, 55]  # 词表内合法 token（实施时用 model_core.vocab 校验过的两条 token）
    monkeypatch.setattr("web.ledger_stats.get_source", lambda kind: _FakeSource(bars))
    theo, why = LS.recompute_theo("fake", "BTCUSDT", "1h", formula,
                                  start_ts=40*3600, end_ts=59*3600,
                                  cost_rate=0.001, ledger_dir=tmp_path)
    assert why is None and theo
    # 用同一公式直接对最后一根评估，与重算序列的末位一致
    from web.realtime_manager import bars_to_raw_dict
    from strategy_manager.live_signal import evaluate_signal
    live = evaluate_signal(formula, bars_to_raw_dict(bars[-5:]), require_closed=False)
    assert abs(theo[-1]["position"] - live["position"]) < 1e-9
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现** — `recompute_theo`：`need = span_bars + LOOKBACK + 1`（LOOKBACK 用 realtime_manager 的 `_N_BARS` 常量值），fetch 一次；对窗口内每个下标 k：`window = bars[max(0,k-LOOKBACK):k+1]`，`res = evaluate_signal(formula, bars_to_raw_dict(window), require_closed=False)`，state!="ok" 的行 position=None；产出 rows 过 `derive_rows`（合成 LedgerRecord 复用）。`theo_is_fresh`：缓存文件 mtime ≥ 账本 mtime 且首行 `cost_rate` 相同。异常/源不可用 → `([], f"数据源拉取失败: {exc}")`，不写缓存。
- [ ] **Step 4: 跑测试确认通过** + 注入差异回归：把 theo 某行 position ±0.1 后 `compute_stats` 分叉标记（curve API 在 B5 体现，此处先加纯函数 `align_divergence(live_rows, theo_rows) -> list[bar_ts]`，单测一行：注入后返回该 bar_ts）
- [ ] **Step 5: Commit** — `git commit -am "feat(ledger): lazy theoretical recompute with mtime-based cache"`

### Task B5: `/api/ledger/*` 端点（web/app.py）

**Files:**
- Modify: `web/app.py`（文件尾部 realtime 端点后追加）
- Test: `tests/unit/test_ledger_api.py`

**Interfaces:**
- Consumes: B1 `read_records/ledger_path/load_strategy_index`、B3 `derive_rows/compute_stats`、B4 `recompute_theo/theo_is_fresh/align_divergence`
- Produces: 5 个 GET 端点（路径/参数/返回字段严格按 spec §5.6 表；`days` 默认 30；`watch_id` 即账本文件名去后缀）

- [ ] **Step 1: 写失败测试**（fastapi TestClient，直接向账本目录灌合成记录）

```python
# tests/unit/test_ledger_api.py
from fastapi.testclient import TestClient
import web.live_ledger as LL
from web.app import app

client = TestClient(app)

def test_watches_lists_ledgers(tmp_path, monkeypatch):
    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    LL.append_record(_rec(0), ledger_dir=tmp_path)
    rows = client.get("/api/ledger/watches").json()["watches"]
    assert rows[0]["watch_id"] == "okx_BTCUSDT_1h" and rows[0]["records"] == 1

def test_summary_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(LL, "LEDGER_DIR", tmp_path)
    for i in range(3): LL.append_record(_rec(i*3600, close=100+i), ledger_dir=tmp_path)
    r = client.get("/api/ledger/summary", params={"watch_id": "okx_BTCUSDT_1h", "days": 1})
    assert r.status_code == 200
    body = r.json()
    assert {"records","win_rate","profit_factor","annualized","sortino",
            "max_drawdown","completeness","corrupt_lines","theo_available"} <= body.keys()
```

（`_rec` 同 B3 测试定义，可提公共 conftest。）

- [ ] **Step 2: 跑测试确认失败** — 404
- [ ] **Step 3: 实现** — 端点逻辑：定位账本→read_records→按 `days` 截窗（以末条 bar_ts 为锚）→sha 过滤（默认最新段）→derive_rows→若 `theo_is_fresh` 失败且有源记录则调 `recompute_theo`（结果喂 derive_rows 得 theo 曲线）→组装。`completeness` = 记录数 ÷ max(记录数, 重算源窗内收盘 bar 数)，源不可用返回 None（不是 0）。`/records` 分页返回原字段+`pnl`；`/export` 用 `io.StringIO` 写 CSV（表头+全部行），`Response(content=..., media_type="text/csv")` 带 `Content-Disposition`。所有 except → `{"error": str(exc)}` + 200（前端可展示原因，不 500）。
- [ ] **Step 4: 跑测试确认通过** + 全套回归
- [ ] **Step 5: Commit** — `git commit -am "feat(api): /api/ledger endpoints (summary/curve/records/export/watches)"`

### Task B6: 04 前端页 + 端到端验证

**Files:**
- Modify: `web/static/index.html`（tab 导航加「04 实盘台账」；`grep -n 'class="tab' web/static/index.html` 找现有结构照排）
- Modify: `web/static/app.js`（新增 `initLedger()/loadLedger()/drawLedgerChart()`，tab 切换处调用 `initLedger`）

**Interfaces:** Consumes B5 全部端点。Produces：无下游。

- [ ] **Step 1: HTML** — section 内：watch `<select id="ledgerWatch">`、days `<select>`（7/30/90）、统计卡容器 `<div id="ledgerStats">`、`<canvas id="ledgerChart" width="900" height="320">`、明细 `<table id="ledgerTable">`、导出按钮 `<a id="ledgerCsv">`、空态文案「台账自添加 watch 起积累，服务重启不丢」
- [ ] **Step 2: JS** — `loadLedger()` 并发拉 summary/curve/records，统计卡渲染（null→「—」，注明理论不可用原因）；`drawLedgerChart(canvas, curve)` 自绘两条线（`beginPath` 双序列 + 颜色实盘#4f8ef7/理论#f7a34f，y 轴 cum，x 轴 bar_ts；不引外部库）；`ledgerCsv.href = '/api/ledger/export?...'`
- [ ] **Step 3: 端到端（真实网络）** — 重启 Web；03 页添加 watch：源 okx / 品种 BTCUSDT / 周期 1h / 新命名策略文件；等待下一根 1h 收盘（或临时用 1m 品种加速）；确认 `live_ledger/okx_BTCUSDT_1h.jsonl` 出现 ≥1 行；04 页出统计与叠图；**杀进程重启** → 确认账本无重复行、`last_ledger_bar_ts` 从末行恢复
- [ ] **Step 4: 回归** — `.venv/Scripts/python.exe -m pytest tests/unit -v` 全绿
- [ ] **Step 5: Commit** — `git commit -am "feat(web): ledger page 04 with live-vs-theoretical curve"`

---

## Self-Review 结论

- spec 覆盖：§5.1–5.7 → B1–B6；§6 错误处理内嵌于各步（写失败重试、corrupt 计数、索引缺失降级、源失败单线、bar 不齐 missing 排除）；§7 fills 接口本期无任务（明确不实现）；§8 测试 1↔B1、2↔B2、3↔B4、4↔B3、5↔B1、6↔B6。
- 类型一致性：`LedgerRecord.close`、`derive_rows(cost_rate=)`、`recompute_theo` 返回 `(rows, reason)` 全文统一。
- 与 spec 的两处偏差（已在文中声明）：模块拆分为 live_ledger/ledger_stats 两文件；测试断言口径细节以各任务内代码为准。
