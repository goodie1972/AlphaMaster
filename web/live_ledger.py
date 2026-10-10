"""实盘信号台账 IO 核心（spec §5.1/§5.2）。

每 watch 一个 jsonl 账本文件，逐行追加 LedgerRecord；
strategy 索引 sha -> {strategy_file, vocab_version}，原子写。

原则：台账失败绝不中断信号评估主链路 —— append_record 遇 OSError 返回 False，
read/load 对缺失或损坏内容降级为空结果。仅用标准库。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

# 台账根目录；测试通过 monkeypatch 本模块属性替换。
LEDGER_DIR = Path(__file__).resolve().parents[1] / "live_ledger"

_APPEND_LOCK = threading.Lock()

_INDEX_NAME = "strategies_index.json"


@dataclass
class LedgerRecord:
    """账本每行一条记录，字段与 spec §5.2 完全一致。"""

    ts: float               # 本地写入墙钟，仅审计
    bar_ts: int             # 信号 bar 开盘时间（源给），对齐/去重只用它
    source: str
    symbol: str
    timeframe: str
    strategy_sha: str       # 公式数组短哈希（sha1 前 6 位）
    factor_value: float
    position: float
    direction: str
    strength: float
    close: float            # 信号 bar 收盘价（算相邻收益必需）


def formula_sha(formula: list[int]) -> str:
    """公式数组短哈希：逗号连接后 sha1，取前 6 位十六进制。"""
    return hashlib.sha1(",".join(map(str, formula)).encode()).hexdigest()[:6]


def _safe(s: str) -> str:
    """文件名安全化（规则同 strategy_file.py）：仅保留 [A-Za-z0-9_.-]。"""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def ledger_path(source, symbol, timeframe, *, ledger_dir: Path | None = None) -> Path:
    if ledger_dir is None:
        ledger_dir = LEDGER_DIR
    return Path(ledger_dir) / f"{_safe(str(source))}_{_safe(str(symbol))}_{_safe(str(timeframe))}.jsonl"


def append_record(record: LedgerRecord, *, ledger_dir: Path | None = None) -> bool:
    """整行追加一条记录。成功 True；任何 OSError False（绝不抛出）。"""
    if ledger_dir is None:
        ledger_dir = LEDGER_DIR
    path = ledger_path(record.source, record.symbol, record.timeframe, ledger_dir=ledger_dir)
    line = json.dumps(asdict(record), ensure_ascii=False) + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _APPEND_LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
    except OSError:
        return False
    return True


def read_records(path) -> tuple[list[LedgerRecord], int]:
    """逐行容错读取：返回 (记录列表, corrupt_lines)。

    未知字段经 __dataclass_fields__ 白名单过滤；缺字段/坏 JSON 行计入 corrupt。
    文件不存在或整体读失败时返回 ([], 0)，绝不抛出。
    """
    fields = set(LedgerRecord.__dataclass_fields__)
    recs: list[LedgerRecord] = []
    corrupt = 0
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return recs, corrupt
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            corrupt += 1
            continue
        if not isinstance(obj, dict):
            corrupt += 1
            continue
        try:
            recs.append(LedgerRecord(**{k: v for k, v in obj.items() if k in fields}))
        except TypeError:  # 缺必填字段等
            corrupt += 1
    return recs, corrupt


def last_bar_ts(path) -> int | None:
    """账本末条记录的 bar_ts；空账本/读失败返回 None。"""
    recs, _ = read_records(path)
    return int(recs[-1].bar_ts) if recs else None


def _index_path(ledger_dir: Path) -> Path:
    return Path(ledger_dir) / _INDEX_NAME


def load_strategy_index(ledger_dir: Path | None = None) -> dict:
    """读策略索引；缺失/损坏降级为空 dict。"""
    if ledger_dir is None:
        ledger_dir = LEDGER_DIR
    try:
        data = json.loads(_index_path(ledger_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def ensure_strategy_index(formula, strategy_file, vocab_version, *, ledger_dir: Path | None = None) -> str:
    """登记公式 sha -> {strategy_file, vocab_version}；幂等，返回 sha。

    同公式同元数据只保留一条；不同公式产生新 sha 条目。更新用 tmp+os.replace 原子写。
    """
    if ledger_dir is None:
        ledger_dir = LEDGER_DIR
    sha = formula_sha(list(formula))
    idx = load_strategy_index(ledger_dir)
    entry = {"strategy_file": strategy_file, "vocab_version": vocab_version}
    if idx.get(sha) != entry:
        idx[sha] = entry
        path = _index_path(ledger_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    return sha
