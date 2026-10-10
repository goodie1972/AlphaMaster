"""Export / import training checkpoints as portable zip packages."""
from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from model_core.vocab import FORMULA_VOCAB
from web.progress import (
    CHECKPOINT_DIR,
    PROJECT_ROOT,
    _decode_formula,
    checkpoint_glob,
    invalidate_checkpoint_cache,
    strategy_files_for_symbol,
)
from web.strategy_file import strategy_path_for_symbol, tf_from_best_name

_CKPT_NAME_RE = re.compile(r"^ckpt_(.+)_step_(\d+)\.pt$", re.IGNORECASE)


def _history_path(symbol: str) -> Path:
    return PROJECT_ROOT / f"training_history_{symbol}.json"


def _load_export_strategy(
    symbol: str,
    timeframe: str | None,
) -> tuple[dict | None, Path | None]:
    """选择导出用策略文件，只认新命名 best_{symbol}_{tf}.json。

    显式 timeframe → 该精确文件（非法周期由 strategy_path_for_symbol 抛 ValueError）；
    缺省时若该品种恰好只有一个新命名策略文件则用它，多个则抛 ValueError 要求显式
    指定，没有则不导出策略。绝不读取旧命名 best_{symbol}.json。
    """
    tf = (timeframe or "").strip()
    if tf:
        path = strategy_path_for_symbol(symbol, tf)
    else:
        files = strategy_files_for_symbol(symbol)
        if len(files) > 1:
            tfs = "/".join(str(tf_from_best_name(p)) for p in files)
            raise ValueError(
                f"{symbol} 存在多个周期的策略文件（{tfs}），导出前需显式指定 timeframe"
            )
        if not files:
            return None, None
        path = files[0]
    if not path.exists():
        return None, None
    try:
        return json.loads(path.read_text(encoding="utf-8")), path
    except (json.JSONDecodeError, OSError):
        return None, None


def _symbol_from_ckpt_name(name: str) -> str | None:
    m = _CKPT_NAME_RE.match(Path(name).name)
    return m.group(1) if m else None


def _validate_checkpoint_file(path: Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    artifact_version = ckpt.get("vocab_version")
    if artifact_version is None:
        raise ValueError(
            f"检查点 {path.name} 过旧（无 vocab_version），"
            f"当前词表 {FORMULA_VOCAB.version!r}，请重新训练"
        )
    FORMULA_VOCAB.verify(artifact_version)
    step = int(ckpt.get("step", 0))
    symbol = ckpt.get("symbol")
    return {"step": step, "symbol": symbol, "best_score": ckpt.get("best_score")}


def build_training_export_zip(symbol: str, timeframe: str | None = None) -> tuple[bytes, str]:
    ckpts = checkpoint_glob(symbol)
    if not ckpts:
        raise FileNotFoundError(f"未找到 {symbol} 的训练检查点，请先训练并保存 checkpoint")

    latest = ckpts[-1]
    _validate_checkpoint_file(latest)

    strategy, strategy_file = _load_export_strategy(symbol, timeframe)
    history_path = _history_path(symbol)

    step = int(re.search(r"_step_(\d+)\.pt$", latest.name).group(1))
    safe = symbol.replace(".", "_")
    zip_name = f"training_{safe}_step{step:04d}.zip"

    manifest = {
        "format": "alphamaster_training_v1",
        "symbol": symbol,
        "step": step,
        "checkpoint": f"checkpoints/{latest.name}",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "files": [f"checkpoints/{latest.name}"],
    }
    if strategy_file is not None:
        manifest["timeframe"] = tf_from_best_name(strategy_file)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(latest, f"checkpoints/{latest.name}")

        if strategy:
            # 新命名契约：成员路径与实际文件名一致，best_{symbol}_{tf}.json
            strat_name = f"strategies/{strategy_file.name}"
            payload = dict(strategy)
            if payload.get("formula") and not payload.get("formula_decoded"):
                payload["formula_decoded"] = _decode_formula(payload["formula"])
            zf.writestr(strat_name, json.dumps(payload, ensure_ascii=False, indent=2))
            manifest["files"].append(strat_name)

        if history_path.exists():
            hist_name = f"training_history_{symbol}.json"
            zf.write(history_path, hist_name)
            manifest["files"].append(hist_name)

        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    return buf.getvalue(), zip_name


def _remove_symbol_checkpoints(symbol: str) -> int:
    removed = 0
    for path in checkpoint_glob(symbol):
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def import_training_package(
    content: bytes,
    filename: str,
    expected_symbol: str | None = None,
) -> dict[str, Any]:
    name = Path(filename).name.lower()
    installed: list[str] = []
    symbol: str | None = None
    step: int | None = None

    if name.endswith(".zip"):
        result = _import_zip(content, expected_symbol)
        symbol = result["symbol"]
        step = result["step"]
        installed = result["installed"]
    elif name.endswith(".pt"):
        result = _import_pt(content, filename, expected_symbol)
        symbol = result["symbol"]
        step = result["step"]
        installed = result["installed"]
    else:
        raise ValueError("仅支持 .zip 训练包或 .pt 检查点文件")

    invalidate_checkpoint_cache()
    return {
        "ok": True,
        "symbol": symbol,
        "step": step,
        "installed": installed,
        "message": f"已导入 {symbol} 的训练文件（step {step}），下次训练将从断点续训",
    }


def _import_zip(content: bytes, expected_symbol: str | None) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if "manifest.json" not in names:
            raise ValueError("训练包缺少 manifest.json")

        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        if manifest.get("format") != "alphamaster_training_v1":
            raise ValueError("不支持的训练包格式")

        symbol = manifest.get("symbol") or ""
        ckpt_rel = manifest.get("checkpoint", "")
        if not symbol or not ckpt_rel:
            raise ValueError("训练包 manifest 不完整")

        if expected_symbol and symbol != expected_symbol:
            raise ValueError(
                f"训练包品种为 {symbol}，与当前选择的 {expected_symbol} 不一致"
            )

        _remove_symbol_checkpoints(symbol)
        installed: list[str] = []
        imported_history = False

        for member in names:
            if member == "manifest.json":
                continue
            dest = PROJECT_ROOT / member.replace("/", "\\") if "\\" in str(PROJECT_ROOT) else PROJECT_ROOT / member
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(member))
            installed.append(str(dest.relative_to(PROJECT_ROOT)).replace("\\", "/"))

            if member.endswith(".pt"):
                _validate_checkpoint_file(dest)
            if member == f"training_history_{symbol}.json":
                imported_history = True

        # 若 zip 不包含训练曲线，项目中可能残留更“新”的 history，导致 UI 显示步数不变。
        # 这类 zip 通常只有 checkpoint + strategy（例如只想迁移断点），因此应清掉旧曲线以与 checkpoint 对齐。
        if not imported_history:
            try:
                _history_path(symbol).unlink(missing_ok=True)
            except OSError:
                pass

        step = int(manifest.get("step") or 0)
        return {"symbol": symbol, "step": step, "installed": installed}


def _import_pt(content: bytes, filename: str, expected_symbol: str | None) -> dict[str, Any]:
    ckpt_name = Path(filename).name
    symbol = _symbol_from_ckpt_name(ckpt_name)
    if not symbol:
        raise ValueError(f"无法从文件名识别品种: {ckpt_name}（须为 ckpt_品种_step_XXXX.pt）")

    if expected_symbol and symbol != expected_symbol:
        raise ValueError(
            f"检查点品种为 {symbol}，与当前选择的 {expected_symbol} 不一致"
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    dest = CHECKPOINT_DIR / ckpt_name

    # 只导入 .pt 时，项目里可能还残留更“新”的 training_history_{symbol}.json。
    # Web 进度会优先使用该曲线文件的最后一步，导致显示步数大于 checkpoint。
    # 因此这里主动清掉旧曲线，避免「导入 60 步却显示 84」。
    try:
        _history_path(symbol).unlink(missing_ok=True)
    except OSError:
        pass

    _remove_symbol_checkpoints(symbol)
    dest.write_bytes(content)
    meta = _validate_checkpoint_file(dest)

    return {
        "symbol": symbol,
        "step": meta["step"],
        "installed": [str(dest.relative_to(PROJECT_ROOT)).replace("\\", "/")],
    }
