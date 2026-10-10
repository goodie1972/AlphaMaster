"""实盘信号台账 IO 核心（spec §5.1/§5.2）。

每 watch 一个 jsonl 账本文件，逐行追加 LedgerRecord；
strategy 索引 sha -> {formula, strategy_file, vocab_version, first_seen}，原子写。

原则：台账失败绝不中断信号评估主链路 —— append_record 遇 OSError 返回 False，
read/load 对缺失或损坏内容降级为空结果。仅用标准库。
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# 台账根目录；测试通过 monkeypatch 本模块属性替换。
LEDGER_DIR = Path(__file__).resolve().parents[1] / "live_ledger"

_APPEND_LOCK = threading.Lock()
# 索引读-改-写全程互斥（load -> merge -> tmp -> os.replace），防并发 watch 丢更新。
_INDEX_LOCK = threading.Lock()
# tmp 文件名去重计数：跨进程用 pid、进程内用计数避免固定 tmp 名相互覆盖。
_TMP_COUNTER = itertools.count()

_INDEX_NAME = "strategy_index.json"


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


# 逐字段类型白名单（spec §5.2）：读侧校验/强转，任一线失败即整行计入 corrupt 并跳过，
# 防止一行错类型/NaN 数据（json.loads 接受 NaN/Infinity 字面量）毒化全部下游统计。
_FLOAT_FIELDS = ("ts", "close", "factor_value", "position", "strength")  # 有限 float 或 None
_STR_FIELDS = ("source", "symbol", "timeframe", "strategy_sha", "direction")  # str 直存，标量强转


def _coerce_bar_ts(v):
    """bar_ts 必须可从有限数值 int() 化：bool/NaN/inf/非数值 str/容器 → None（拒）。"""
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        return None
    try:
        f = float(v)
    except ValueError:  # 非数值字符串
        return None
    if not math.isfinite(f):  # NaN / ±inf 字面量
        return None
    return int(f)


def _coerce_float(v):
    """float 字段：None 原样放行；有限数值（含数值字符串）→ float；其余 → (拒, False)。"""
    if v is None:
        return None, True
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        return None, False
    try:
        f = float(v)
    except ValueError:
        return None, False
    if not math.isfinite(f):
        return None, False
    return f, True


def _coerce_str(v):
    """字符串字段：str 原样；非 bool 数值标量 str() 强转（如 strategy_file 派生数字段）；
    其余（dict/list/None/bool）→ None（拒）。内容原样存证，不做语义校验。"""
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(v)
    return None


def _line_to_record(obj) -> "LedgerRecord | None":
    """整行字段级校验：任一必填字段缺失/类型不合 → None（调用方计 corrupt）。"""
    if not isinstance(obj, dict):
        return None
    fields = LedgerRecord.__dataclass_fields__
    if not set(fields) <= set(obj):  # 缺必填字段 → 与原 TypeError 语义一致
        return None
    kwargs = {}
    for k in _STR_FIELDS:
        s = _coerce_str(obj.get(k))
        if s is None:
            return None
        kwargs[k] = s
    for k in _FLOAT_FIELDS:
        f, ok = _coerce_float(obj.get(k))
        if not ok:
            return None
        kwargs[k] = f
    bar_ts = _coerce_bar_ts(obj.get("bar_ts"))
    if bar_ts is None:
        return None
    kwargs["bar_ts"] = bar_ts
    return LedgerRecord(**kwargs)


def read_records(path: Path) -> tuple[list[LedgerRecord], int]:
    """逐行容错读取：返回 (记录列表, corrupt_lines)。

    未知字段经 __dataclass_fields__ 白名单过滤；缺字段/坏 JSON/字段类型不合法
    （如 bar_ts="abc"、position=NaN —— dataclass 无运行时校验，读侧单点补齐）
    一律计入 corrupt 并跳过该行，绝不入库。
    文件不存在或整体读失败时返回 ([], 0)，绝不抛出；非 UTF-8 字节按
    errors="replace" 解码后交给逐行 corrupt 计数处理（契约：永不抛）。
    注：读者可能把写者半刷新的末行瞬时计为 corrupt（spec §6 半行损坏=容错设计）；
    追加模式整行写入保证下次读取该行完整可见。
    """
    recs: list[LedgerRecord] = []
    corrupt = 0
    p = Path(path)
    try:
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = p.read_bytes().decode("utf-8", errors="replace")
    except (OSError, UnicodeError):
        return recs, corrupt
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:  # json.JSONDecodeError 是 ValueError 子类
            corrupt += 1
            continue
        rec = _line_to_record(obj)
        if rec is None:
            corrupt += 1
            continue
        recs.append(rec)
    return recs, corrupt


def last_bar_ts(path: Path) -> int | None:
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
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def ensure_strategy_index(formula: list[int], strategy_file: str, vocab_version: str,
                          *, ledger_dir: Path | None = None) -> str:
    """登记公式 sha -> {formula, strategy_file, vocab_version, first_seen}；返回 sha。

    同公式同元数据幂等不重写；first_seen 只在首次登记时写入，后续更新保留原值
    （判等只看 formula/strategy_file/vocab_version，忽略 first_seen）。
    不同公式产生新 sha 条目。整个读-改-写由 _INDEX_LOCK 保护，tmp 名带 pid+计数
    避免多写者竞争同一固定 tmp 文件，更新用 tmp+os.replace 原子替换。
    """
    if ledger_dir is None:
        ledger_dir = LEDGER_DIR
    formula = list(formula)
    sha = formula_sha(formula)
    new_entry = {"formula": formula, "strategy_file": strategy_file, "vocab_version": vocab_version}
    with _INDEX_LOCK:
        idx = load_strategy_index(ledger_dir)
        entry = idx.get(sha)
        core = {k: entry.get(k) for k in new_entry} if isinstance(entry, dict) else None
        if core == new_entry:  # 幂等：内容一致（first_seen 不参与判等），不写盘
            return sha
        entry = dict(new_entry)
        old = idx.get(sha)
        if isinstance(old, dict) and "first_seen" in old:
            entry["first_seen"] = old["first_seen"]  # 更新元数据也保留原始 first_seen
        entry.setdefault("first_seen", time.time())  # 首次登记才写入
        idx[sha] = entry
        path = _index_path(ledger_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{next(_TMP_COUNTER)}")
        tmp.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    return sha
