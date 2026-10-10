"""Strategy JSON inspection for the web UI."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from data_pipeline.parquet_manager import inspect_parquet_file
from model_core.vocab import VOCAB_VERSION
from web.progress import (
    STRATEGIES_DIR,
    _decode_formula,
    _load_checkpoint_meta,
    _load_strategy,
    checkpoint_glob,
    strategy_files_for_symbol,
)
from web.settings import load_settings

_BEST_NAME_RE = re.compile(r"^best_(.+)_[A-Za-z0-9]{1,4}\.json$", re.IGNORECASE)
_STRATEGY_EXPORT_RE = re.compile(
    r"^strategy_(.+)_step(\d+)(?:_score([\d.]+))?\.json$",
    re.IGNORECASE,
)
_TF_PART_RE = re.compile(r"^(?P<sym>.+)_(?P<tf>[A-Za-z0-9]{1,4})$")


def parse_best_stem(stem: str) -> tuple[str, str] | None:
    """解析 best_{symbol}_{tf} 文件名主干；旧命名（无周期）返回 None。"""
    if not stem.lower().startswith("best_"):
        return None
    m = _TF_PART_RE.match(stem[len("best_"):])
    return (m.group("sym"), m.group("tf")) if m else None


def strategy_path_for_symbol(symbol: str, timeframe: str) -> Path:
    tf = (timeframe or "").strip()
    if not tf or not re.fullmatch(r"[A-Za-z0-9]{1,4}", tf):
        raise ValueError(f"无效周期，拒绝写无周期策略名: {timeframe!r}")
    return STRATEGIES_DIR / f"best_{symbol}_{tf}.json"


def tf_from_best_name(path: Path) -> str | None:
    """从 best_{symbol}_{tf}.json 文件名取周期；旧命名或非 best 文件返回 None。"""
    m = _BEST_NAME_RE.match(path.name)
    if not m:
        return None
    parsed = parse_best_stem(path.stem)
    return parsed[1] if parsed else None


def symbol_from_strategy_path(path: Path) -> tuple[str, str | None] | None:
    """best_ 文件返回 (symbol, tf)；strategy_ 导出返回 (symbol, None)；否则 None。"""
    parsed = parse_best_stem(path.stem)
    if parsed:
        return parsed
    m = _STRATEGY_EXPORT_RE.match(path.name)
    if m:
        return m.group(1), None
    return None


def is_legacy_best_name(path: Path | str) -> bool:
    """旧命名 best_{symbol}.json（best_ 前缀但主干不含周期）返回 True。

    仅针对 best_ 前缀文件；strategy_ 导出与任意名文件不属于「旧命名」，返回 False。
    """
    stem = Path(path).stem
    return stem.lower().startswith("best_") and parse_best_stem(stem) is None


def _resolve_data_file_for_symbol(
    symbol: str,
    data_file: str | None,
    *,
    data_file_hint: str | None = None,
) -> tuple[str | None, str | None]:
    """Fill missing strategy data_file from training settings / job hint."""
    sym = (symbol or "").strip()
    if data_file:
        p = Path(str(data_file))
        if p.exists():
            try:
                info = inspect_parquet_file(str(p.resolve()))
                if not sym or info.get("symbol") == sym:
                    return str(p.resolve()), info.get("timeframe")
            except Exception:
                return str(p.resolve()), None

    for candidate in (data_file_hint, load_settings().get("last_data_file") or ""):
        path = str(candidate or "").strip()
        if not path:
            continue
        p = Path(path)
        if not p.exists():
            continue
        try:
            info = inspect_parquet_file(str(p.resolve()))
        except Exception:
            continue
        if sym and info.get("symbol") != sym:
            continue
        return str(p.resolve()), info.get("timeframe")
    return data_file, None


def _apply_data_file_fallback(
    payload: dict[str, Any],
    *,
    data_file_hint: str | None = None,
) -> dict[str, Any]:
    symbol = str(payload.get("symbol") or "").strip()
    data_file, timeframe = _resolve_data_file_for_symbol(
        symbol,
        payload.get("data_file"),
        data_file_hint=data_file_hint,
    )
    if data_file:
        payload["data_file"] = data_file
    if timeframe and not payload.get("timeframe"):
        payload["timeframe"] = timeframe
    if data_file and not payload.get("mode"):
        payload["mode"] = "parquet_file"
    return payload


def inspect_strategy_file(
    path: str,
    *,
    data_file_hint: str | None = None,
) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")

    if is_legacy_best_name(p):
        # 旧命名 best_{symbol}.json：不再兼容，返回错误码而非静默接受
        return {
            "error": "legacy_naming",
            "strategy_file": str(p.resolve()),
            "filename": p.name,
            "symbol": "",
            "timeframe": None,
            "data_file": None,
            "data_file_exists": False,
            "mode": None,
            "best_score": None,
            "vocab_version": None,
            "formula_decoded": None,
            "valid": False,
            "message": "旧命名策略不可用：文件名需含周期（best_品种_周期.json），请用新命名重新导出",
        }

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"无法解析策略 JSON: {exc}") from exc

    parsed = symbol_from_strategy_path(p)
    fname_symbol, fname_tf = parsed if parsed else (None, None)

    if isinstance(data, list):
        formula = data
        symbol = fname_symbol
        best_score = None
        vocab_version = "legacy"
    elif isinstance(data, dict):
        formula = data.get("formula")
        symbol = data.get("symbol") or fname_symbol
        best_score = data.get("best_score")
        vocab_version = data.get("vocab_version")
    else:
        raise ValueError("策略文件格式无效")

    if not formula:
        raise ValueError("策略文件缺少 formula 字段")

    formula_decoded = None
    timeframe = None
    data_file = None
    mode = None
    if isinstance(data, dict):
        formula_decoded = data.get("formula_decoded") or _decode_formula(formula)
        timeframe = data.get("timeframe")
        data_file = data.get("data_file")
        mode = data.get("mode")

    data_file, tf_fallback = _resolve_data_file_for_symbol(
        symbol or "",
        data_file,
        data_file_hint=data_file_hint,
    )
    if tf_fallback and not timeframe:
        timeframe = tf_fallback
    if not timeframe:
        timeframe = fname_tf

    data_file_exists = bool(data_file) and Path(str(data_file)).exists()

    return {
        "strategy_file": str(p.resolve()),
        "filename": p.name,
        "symbol": symbol or "",
        "timeframe": timeframe,
        "data_file": data_file,
        "data_file_exists": data_file_exists,
        "mode": mode,
        "best_score": best_score,
        "vocab_version": vocab_version,
        "formula_decoded": formula_decoded,
        "valid": True,
        "message": "",
    }


def resolve_strategy_file(
    saved_path: str,
    train_symbol: str | None = None,
    timeframe: str | None = None,
) -> str:
    """优先使用已保存路径；否则回退到训练品种/周期对应的 best_{symbol}_{tf}.json。

    周期优先取显式 timeframe 参数；缺省时尝试从已保存文件名解析；
    仍无法确定则抛 ValueError（不兼容旧的无周期命名）。
    """
    if saved_path:
        p = Path(saved_path)
        if p.exists():
            return str(p.resolve())

    if train_symbol:
        tf = (timeframe or "").strip()
        if not tf and saved_path:
            tf = tf_from_best_name(Path(saved_path)) or ""
        if not tf:
            raise ValueError(f"缺少周期，无法解析 {train_symbol!r} 的策略文件")
        default = strategy_path_for_symbol(train_symbol, tf)
        if default.exists():
            return str(default.resolve())

    return saved_path or ""


def _step_from_export_name(path: Path) -> int:
    m = _STRATEGY_EXPORT_RE.match(path.name)
    if not m:
        return 0
    try:
        return int(m.group(2))
    except (TypeError, ValueError):
        return 0


def sync_best_strategy_for_symbol(
    symbol: str,
    timeframe: str | None = None,
    *,
    data_file_hint: str | None = None,
) -> dict[str, Any] | None:
    """在策略文件与检查点中选出最高分策略，写入 strategies/best_{symbol}_{tf}.json。

    周期解析链（全部失败则抛 ValueError，绝不写无周期文件名）：
    显式 timeframe 参数 → checkpoint meta 的 timeframe → 数据文件解析
    （_resolve_data_file_for_symbol 已返回 tf）→ 该品种唯一的新命名策略文件。
    旧命名 best_{symbol}.json 不作为任何读取来源。
    """
    tf = (timeframe or "").strip()
    if not tf:
        # checkpoint_glob 按 mtime 升序；倒序遍历，优先取最新 checkpoint 的周期
        for ckpt_path in reversed(checkpoint_glob(symbol)):
            try:
                meta_tf = _load_checkpoint_meta(ckpt_path).get("timeframe")
            except Exception:  # 损坏的 ckpt 不应阻断 sync，跳过
                meta_tf = None
            if meta_tf:
                tf = str(meta_tf).strip()
                break
    if not tf:
        _, data_tf = _resolve_data_file_for_symbol(
            symbol, None, data_file_hint=data_file_hint
        )
        tf = str(data_tf or "").strip()
    if not tf:
        tfs = {tf_from_best_name(p) for p in strategy_files_for_symbol(symbol)}
        tfs.discard(None)
        if len(tfs) == 1:
            tf = tfs.pop()
    if not tf:
        raise ValueError(
            f"无法确定周期，拒绝写入无周期策略名: {symbol!r}"
            "（显式 timeframe / ckpt meta / 数据文件 / 既有新命名文件均未提供周期）"
        )

    strat = _load_strategy(symbol, tf)

    candidates: list[tuple[float, list[int], int]] = []

    if strat and strat.get("formula") and strat.get("best_score") is not None:
        step = int(strat.get("train_step") or strat.get("current_step") or 0)
        candidates.append((float(strat["best_score"]), strat["formula"], step))

    for ckpt_path in checkpoint_glob(symbol):
        meta = _load_checkpoint_meta(ckpt_path)
        score = meta.get("best_score")
        formula = meta.get("best_formula")
        if score is None or not formula:
            continue
        candidates.append((float(score), formula, int(meta.get("step") or 0)))

    safe = symbol.replace(".", "_")
    for path in STRATEGIES_DIR.glob(f"strategy_{safe}_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        formula = data.get("formula")
        score = data.get("best_score")
        if not formula or score is None:
            continue
        candidates.append((float(score), formula, _step_from_export_name(path)))

    if not candidates:
        existing = strategy_path_for_symbol(symbol, tf)
        if existing.exists():
            return inspect_strategy_file(str(existing.resolve()))
        return None

    best_score, best_formula, best_step = max(candidates, key=lambda row: row[0])
    out_path = strategy_path_for_symbol(symbol, tf)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vocab_version": VOCAB_VERSION,
        "symbol": symbol,
        "timeframe": tf,
        "formula": best_formula,
        "best_score": best_score,
        "formula_decoded": _decode_formula(best_formula),
        "train_step": best_step,
    }
    if strat:
        for key in ("data_file", "mode", "train_steps"):
            if strat.get(key) is not None:
                payload[key] = strat[key]
    payload = _apply_data_file_fallback(payload, data_file_hint=data_file_hint)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return inspect_strategy_file(str(out_path.resolve()), data_file_hint=data_file_hint)
