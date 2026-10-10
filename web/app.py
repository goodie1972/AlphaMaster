"""FastAPI application for AlphaMaster training UI."""
from __future__ import annotations

import csv
import io
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.parquet_manager import inspect_parquet_file
from model_core.config import ModelConfig
from web.file_dialog import pick_parquet_file, pick_strategy_file
from web.progress import (
    get_symbol_progress,
    get_strategy_for_export,
    invalidate_checkpoint_cache,
    list_strategies,
    build_strategy_export_filename,
)
from web.server_log import (
    debug_snapshot,
    get_logger,
    is_debug_mode,
    log_error,
    set_debug_mode,
    setup_logging,
)
from web.settings import load_settings, save_settings
from web.strategy_file import (
    inspect_strategy_file,
    resolve_strategy_file,
    strategy_path_for_symbol,
    sync_best_strategy_for_symbol,
)
from web.training_manager import training_manager
from web.training_time import get_training_time_summary
from web.training_package import build_training_export_zip, import_training_package
from web.backtest_manager import backtest_manager
from web.realtime_manager import realtime_manager
from web.data_sources.factory import list_sources
from strategy_manager.live_signal import min_exposure
import web.ledger_stats as LS
import web.live_ledger as LL

STATIC_DIR = Path(__file__).resolve().parent / "static"
BACKTEST_OUTPUT_DIR = ROOT / "backtest_output"

setup_logging()
logger = get_logger()

app = FastAPI(title="AlphaMaster Training", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartTrainingRequest(BaseModel):
    data_file: str
    from_scratch: bool = False


class ClientLogRequest(BaseModel):
    level: str = "error"
    message: str
    context: dict[str, Any] | None = None


class SettingsRequest(BaseModel):
    last_data_file: str | None = None
    last_strategy_file: str | None = None
    debug_mode: bool | None = None
    ai_provider: str | None = None
    ai_api_key: str | None = None
    ai_base_url: str | None = None
    ai_model: str | None = None
    bt_commission_pct: float | None = None
    bt_slippage_pct: float | None = None


class AnalyzeTrainingRequest(BaseModel):
    provider: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    symbol: str | None = None


class StartBacktestRequest(BaseModel):
    strategy_file: str
    commission_pct: float | None = None
    slippage_pct: float | None = None


class AddWatchRequest(BaseModel):
    source: str
    symbol: str
    timeframe: str
    strategy_file: str


class RemoveWatchRequest(BaseModel):
    id: str


class FeishuSettingsRequest(BaseModel):
    enabled: bool | None = None
    webhook_url: str | None = None
    secret: str | None = None


class FeishuTestRequest(BaseModel):
    webhook_url: str | None = None
    secret: str | None = None


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        log_error(f"{request.method} {request.url.path} unhandled", exc)
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000
    if is_debug_mode():
        logger.info(
            "%s %s -> %s (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
    if response.status_code >= 400:
        log_error(f"{request.method} {request.url.path} -> HTTP {response.status_code}")
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    log_error(f"{request.method} {request.url.path} HTTP {exc.status_code}: {exc.detail}")
    detail = exc.detail
    if not isinstance(detail, str):
        detail = str(detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log_error(f"{request.method} {request.url.path} crashed", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "traceback": traceback.format_exc()},
    )


def _inspect_or_http(path: str) -> dict[str, Any]:
    try:
        return inspect_parquet_file(path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def _browse_data_file() -> dict[str, Any]:
    if is_debug_mode():
        logger.info("Opening native file picker")
    try:
        path = pick_parquet_file()
    except Exception as exc:
        log_error("File picker failed", exc)
        raise HTTPException(500, f"文件选择失败: {exc}") from exc

    if not path:
        if is_debug_mode():
            logger.info("File picker cancelled")
        return {"ok": False, "cancelled": True}

    if is_debug_mode():
        logger.info("Selected file: %s", path)
    info = _inspect_or_http(path)
    save_settings({"last_data_file": info["data_file"]})
    return {"ok": True, "cancelled": False, **info}


def _strategy_context() -> dict[str, Any]:
    settings = load_settings()
    data_file = settings.get("last_data_file") or ""
    train_symbol = None
    train_tf = None
    if data_file:
        try:
            info = inspect_parquet_file(data_file)
            train_symbol = info.get("symbol")
            train_tf = info.get("timeframe")
        except Exception:
            pass

    saved = settings.get("last_strategy_file") or ""
    try:
        resolved = resolve_strategy_file(saved, train_symbol, train_tf)
    except ValueError as exc:
        # 周期无法确定：不回退无周期命名，降级为已保存路径并让错误以消息形式浮出
        log_error("resolve_strategy_file: timeframe unresolvable", exc)
        resolved = saved
    strategy_info = None
    if resolved:
        try:
            strategy_info = inspect_strategy_file(
                resolved,
                data_file_hint=settings.get("last_data_file") or None,
            )
        except Exception as e:
            strategy_info = {
                "strategy_file": resolved,
                "valid": False,
                "message": str(e),
            }
    return {
        "last_strategy_file": resolved,
        "strategy_file": strategy_info,
        "train_symbol": train_symbol,
    }


def _browse_strategy_file() -> dict[str, Any]:
    if is_debug_mode():
        logger.info("Opening strategy file picker")
    try:
        path = pick_strategy_file()
    except Exception as exc:
        log_error("Strategy file picker failed", exc)
        raise HTTPException(500, f"文件选择失败: {exc}") from exc

    if not path:
        if is_debug_mode():
            logger.info("Strategy file picker cancelled")
        return {"ok": False, "cancelled": True}

    if is_debug_mode():
        logger.info("Selected strategy: %s", path)
    info = _inspect_strategy_or_http(path)
    save_settings({"last_strategy_file": info["strategy_file"]})
    return {"ok": True, "cancelled": False, **info}


def _inspect_strategy_or_http(path: str) -> dict[str, Any]:
    try:
        return inspect_strategy_file(path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def _resolve_train_symbol(symbol: str | None = None) -> str | None:
    if symbol:
        return symbol.strip() or None
    settings = load_settings()
    data_file = settings.get("last_data_file") or ""
    if not data_file:
        return None
    try:
        return inspect_parquet_file(data_file).get("symbol")
    except Exception:
        return None


def _resolve_train_timeframe(
    timeframe: str | None = None,
    symbol: str | None = None,
) -> str | None:
    """解析调用点周期：显式参数 → 运行中训练 job（同品种）→ 设置中的数据文件。"""
    tf = str(timeframe or "").strip()
    if tf:
        return tf
    job = training_manager.status().get("job") or {}
    job_tf = str(job.get("timeframe") or "").strip()
    if job_tf and (not symbol or str(job.get("symbol") or "") == symbol):
        return job_tf
    data_file = load_settings().get("last_data_file") or ""
    if data_file:
        try:
            info = inspect_parquet_file(data_file)
            if not symbol or info.get("symbol") == symbol:
                tf = str(info.get("timeframe") or "").strip()
                if tf:
                    return tf
        except Exception:
            pass
    return None


def _wait_training_idle(timeout_s: float = 5.0) -> None:
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        training_manager.status()
        if not training_manager.status().get("active"):
            return
        time.sleep(0.2)


def _sync_and_persist_best_strategy(
    symbol: str,
    *,
    timeframe: str | None = None,
    data_file_hint: str | None = None,
) -> dict[str, Any] | None:
    """同步 best_{symbol}_{tf}.json。

    timeframe 可为 None：由 sync 内部解析链（ckpt meta → 数据文件 → 唯一新命名
    文件）兜底；全部失败时抛 ValueError，由调用方转成用户可见的错误消息。
    """
    invalidate_checkpoint_cache()
    hint = data_file_hint
    if not hint:
        job = training_manager.status().get("job") or {}
        if str(job.get("symbol") or "") == symbol:
            hint = job.get("data_file") or None
    if not hint:
        hint = load_settings().get("last_data_file") or None
    info = sync_best_strategy_for_symbol(symbol, timeframe, data_file_hint=hint)
    if info:
        save_settings({"last_strategy_file": info["strategy_file"]})
    return info


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "1.1.0"}


@app.get("/api/routes")
def api_routes() -> dict[str, Any]:
    routes = []
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if path and methods:
            routes.append({"path": path, "methods": sorted(methods)})
    return {"routes": sorted(routes, key=lambda r: r["path"])}


@app.get("/api/debug/logs")
def api_debug_logs(lines: int = 200) -> dict[str, Any]:
    return debug_snapshot(lines)


@app.post("/api/debug/client-log")
def api_client_log(req: ClientLogRequest) -> dict[str, bool]:
    msg = req.message
    if req.context:
        msg = f"{msg} | context={req.context}"
    if req.level == "error":
        log_error(f"[client] {msg}")
    elif is_debug_mode():
        logger.info("[client] %s", msg)
    return {"ok": True}


@app.get("/api/settings")
def api_get_settings() -> dict[str, Any]:
    return load_settings()


@app.put("/api/settings")
def api_put_settings(req: SettingsRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if req.last_data_file is not None:
        payload["last_data_file"] = req.last_data_file
    if req.last_strategy_file is not None:
        payload["last_strategy_file"] = req.last_strategy_file
    if req.debug_mode is not None:
        payload["debug_mode"] = req.debug_mode
    if req.ai_provider is not None:
        payload["ai_provider"] = req.ai_provider
    if req.ai_api_key is not None:
        payload["ai_api_key"] = req.ai_api_key
    if req.ai_base_url is not None:
        payload["ai_base_url"] = req.ai_base_url
    if req.ai_model is not None:
        payload["ai_model"] = req.ai_model
    if req.bt_commission_pct is not None:
        payload["bt_commission_pct"] = req.bt_commission_pct
    if req.bt_slippage_pct is not None:
        payload["bt_slippage_pct"] = req.bt_slippage_pct
    saved = save_settings(payload)
    if req.debug_mode is not None:
        set_debug_mode(req.debug_mode)
    return {"ok": True, **saved}


@app.get("/api/config")
def api_config() -> dict[str, Any]:
    settings = load_settings()
    data_file = settings.get("last_data_file") or ""
    file_info = None
    if data_file:
        try:
            file_info = inspect_parquet_file(data_file)
        except Exception as e:
            file_info = {
                "data_file": data_file,
                "valid": False,
                "message": str(e),
            }
    snap = debug_snapshot(1)
    strat_ctx = _strategy_context()
    return {
        "train_steps": ModelConfig.TRAIN_STEPS,
        "batch_size": ModelConfig.BATCH_SIZE,
        "reward_mode": ModelConfig.REWARD_MODE,
        "max_formula_len": ModelConfig.MAX_FORMULA_LEN,
        "device": str(ModelConfig.DEVICE),
        "last_data_file": data_file,
        "data_file": file_info,
        "last_strategy_file": strat_ctx["last_strategy_file"],
        "strategy_file": strat_ctx["strategy_file"],
        "debug_mode": load_settings().get("debug_mode", False),
        "ai_provider": settings.get("ai_provider", "deepseek"),
        "ai_api_key": settings.get("ai_api_key", ""),
        "ai_base_url": settings.get("ai_base_url", "https://api.deepseek.com"),
        "ai_model": settings.get("ai_model", "deepseek-v4-flash"),
        "bt_commission_pct": settings.get("bt_commission_pct", 0.02),
        "bt_slippage_pct": settings.get("bt_slippage_pct", 0.01),
        "server_log": snap["server_log"],
        "error_log": snap["error_log"],
    }


@app.get("/api/ai/providers")
def api_ai_providers() -> dict[str, Any]:
    from web.ai_providers import provider_status

    status = provider_status()
    settings = load_settings()
    status["selected"] = settings.get("ai_provider", "deepseek")
    status["has_api_key"] = bool(settings.get("ai_api_key"))
    return status


@app.post("/api/ai/analyze-training")
def api_ai_analyze_training(req: AnalyzeTrainingRequest):
    from fastapi.responses import StreamingResponse

    from web.ai_analyze import analyze_training_stream

    settings = load_settings()
    raw_key = req.api_key if req.api_key is not None else settings.get("ai_api_key") or ""
    key_lower = str(raw_key).strip().lower()
    base_url = (
        req.base_url
        if req.base_url is not None
        else settings.get("ai_base_url") or ""
    )
    model = req.model if req.model is not None else settings.get("ai_model") or ""

    # openclaw_wb 必须先于 openclaw 判断
    if key_lower in ("openclaw_wb",) or key_lower.startswith("openclaw_wb/"):
        provider = "openclaw_wb"
    elif key_lower in ("openclaw",) or key_lower.startswith("openclaw/"):
        provider = "openclaw"
    else:
        provider = (req.provider or settings.get("ai_provider") or "deepseek").strip()

    save_settings({
        "ai_provider": provider,
        "ai_api_key": str(raw_key).strip(),
        "ai_base_url": str(base_url).strip(),
        "ai_model": str(model).strip(),
    })

    def event_gen():
        try:
            for event in analyze_training_stream(
                provider=provider,
                api_key=str(raw_key).strip() or None,
                base_url=str(base_url).strip() or None,
                model=str(model).strip() or None,
                symbol=req.symbol,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/data-file/browse")
@app.get("/api/data-file/browse")
def api_browse_data_file() -> dict[str, Any]:
    return _browse_data_file()


@app.post("/api/strategy-file/browse")
@app.get("/api/strategy-file/browse")
def api_browse_strategy_file() -> dict[str, Any]:
    return _browse_strategy_file()


@app.post("/api/strategy-file/sync-best")
@app.get("/api/strategy-file/sync-best")
def api_sync_best_strategy(
    symbol: str | None = None,
    timeframe: str | None = None,
) -> dict[str, Any]:
    sym = _resolve_train_symbol(symbol)
    if not sym:
        raise HTTPException(400, "请先选择训练数据文件或指定品种")
    tf = _resolve_train_timeframe(timeframe, sym)
    try:
        info = _sync_and_persist_best_strategy(sym, timeframe=tf)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not info:
        raise HTTPException(404, f"未找到 {sym} 的可用策略")
    return {"ok": True, **info}


def _progress_with_live_step(
    symbol: str,
    active: bool,
    timeframe: str | None = None,
) -> dict[str, Any]:
    p = get_symbol_progress(symbol, timeframe)
    current_step = p.current_step
    if active:
        live = training_manager.parse_step_from_log()
        if live is not None:
            current_step = max(current_step, live)
    train_steps = p.train_steps
    progress_pct = min(100.0, 100.0 * current_step / train_steps) if train_steps > 0 else 0.0
    val_score = None
    hist = p.history or {}
    vals = hist.get("val_score") or []
    if vals:
        try:
            val_score = float(vals[-1])
        except (TypeError, ValueError):
            val_score = None
    return {
        "symbol": p.symbol,
        "current_step": current_step,
        "train_steps": train_steps,
        "progress_pct": round(progress_pct, 1),
        "best_score": p.best_score,
        "val_score": val_score,
        "formula_decoded": p.formula_decoded,
        "status": p.status,
        "history": p.history,
        "has_checkpoint": bool(p.checkpoint_path),
        "has_strategy": p.has_strategy,
    }


def _attach_training_time(
    row: dict[str, Any] | None,
    *,
    symbol: str | None,
    job: dict[str, Any] | None,
    active: bool,
) -> dict[str, Any] | None:
    if not row or not symbol:
        return row
    summary = get_training_time_summary(symbol, job=job, active=active)
    row = dict(row)
    row["session_seconds"] = summary.session_seconds
    row["history_total_seconds"] = summary.history_total_seconds
    return row


@app.get("/api/overview")
def api_overview() -> dict[str, Any]:
    settings = load_settings()
    data_file = settings.get("last_data_file") or ""
    file_info = None
    progress = None

    training = training_manager.status()
    job = training.get("job")
    active = bool(training.get("active"))

    if data_file:
        try:
            file_info = inspect_parquet_file(data_file)
            sym = file_info.get("symbol")
            row = _progress_with_live_step(
                sym, active=False, timeframe=file_info.get("timeframe")
            )
            progress = {
                "symbol": row["symbol"],
                "status": row["status"],
                "current_step": row["current_step"],
                "train_steps": row["train_steps"],
                "progress_pct": row["progress_pct"],
                "best_score": row["best_score"],
                "val_score": row.get("val_score"),
                "formula_decoded": row["formula_decoded"],
                "has_checkpoint": row.get("has_checkpoint", False),
                "has_strategy": row.get("has_strategy", False),
            }
            progress = _attach_training_time(
                progress, symbol=sym, job=job, active=active and job and job.get("symbol") == sym
            )
        except Exception as e:
            file_info = {"data_file": data_file, "valid": False, "message": str(e)}

    if job and job.get("symbol") and active:
        sym = job["symbol"]
        row = _progress_with_live_step(sym, active=True, timeframe=job.get("timeframe"))
        progress = {
            "symbol": row["symbol"],
            "status": "running_job",
            "current_step": row["current_step"],
            "train_steps": row["train_steps"],
            "progress_pct": row["progress_pct"],
            "best_score": row["best_score"],
            "val_score": row.get("val_score"),
            "formula_decoded": row["formula_decoded"],
            "has_checkpoint": row.get("has_checkpoint", False),
            "has_strategy": row.get("has_strategy", False),
        }
        progress = _attach_training_time(progress, symbol=sym, job=job, active=True)

    return {
        "data_file": file_info,
        "progress": progress,
        "training": training,
    }


@app.get("/api/symbols/{symbol}")
def api_symbol(symbol: str, timeframe: str | None = None) -> dict[str, Any]:
    tf = _resolve_train_timeframe(timeframe, symbol)
    p = get_symbol_progress(symbol, tf)
    return {
        "symbol": p.symbol,
        "status": p.status,
        "current_step": p.current_step,
        "train_steps": p.train_steps,
        "progress_pct": round(p.progress_pct, 1),
        "best_score": p.best_score,
        "best_formula": p.best_formula,
        "formula_decoded": p.formula_decoded,
        "has_strategy": p.has_strategy,
        "strategy_score": p.strategy_score,
        "checkpoint_path": p.checkpoint_path,
        "history": p.history,
    }


@app.get("/api/strategies")
def api_strategies() -> dict[str, Any]:
    return {"strategies": list_strategies()}


@app.get("/api/strategies/{symbol}/export")
def api_export_strategy(symbol: str, timeframe: str | None = None):
    import json

    from fastapi.responses import Response

    tf = _resolve_train_timeframe(timeframe, symbol)
    try:
        payload = get_strategy_for_export(symbol, tf)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    progress = get_symbol_progress(symbol, tf)
    step = progress.current_step
    score = payload.get("best_score")
    if score is None:
        score = progress.strategy_score if progress.strategy_score is not None else progress.best_score
    filename = build_strategy_export_filename(symbol, step, score)
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    return Response(
        content=body,
        media_type="application/json; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@app.get("/api/training/{symbol}/export")
def api_export_training(symbol: str, timeframe: str | None = None):
    from fastapi.responses import Response

    tf = _resolve_train_timeframe(timeframe, symbol)
    try:
        body, zip_name = build_training_export_zip(symbol, tf)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(
        content=body,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )


@app.post("/api/training/import")
async def api_import_training(
    file: UploadFile = File(...),
    symbol: str | None = Query(None, description="当前选择的品种，用于校验导入包是否一致"),
) -> dict[str, Any]:
    if training_manager.status().get("active"):
        raise HTTPException(409, "训练进行中，请先停止再导入")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "上传文件为空")

    try:
        return import_training_package(
            raw,
            file.filename or "upload.zip",
            expected_symbol=symbol or None,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/training/status")
def api_training_status() -> dict[str, Any]:
    status = training_manager.status()
    status["log_tail"] = training_manager.tail_log(150)
    return status


@app.post("/api/training/start")
def api_training_start(req: StartTrainingRequest) -> dict[str, Any]:
    info = _inspect_or_http(req.data_file)
    save_settings({"last_data_file": info["data_file"]})
    try:
        job = training_manager.start(
            data_file=info["data_file"],
            symbol=info["symbol"],
            timeframe=info["timeframe"],
            mode="ftmo",
            from_scratch=bool(req.from_scratch),
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    if req.from_scratch:
        invalidate_checkpoint_cache()
    return {
        "ok": True,
        "job": job.to_dict(),
        "data_file": info,
        "from_scratch": bool(req.from_scratch),
    }


@app.post("/api/training/stop")
def api_training_stop() -> dict[str, Any]:
    job = training_manager.status().get("job") or {}
    symbol = job.get("symbol")
    data_file_hint = job.get("data_file")
    job_timeframe = job.get("timeframe")
    stopped = training_manager.stop()
    strategy_file = None
    sync_error = None
    if symbol:
        _wait_training_idle()
        try:
            strategy_file = _sync_and_persist_best_strategy(
                symbol,
                timeframe=job_timeframe,
                data_file_hint=data_file_hint,
            )
        except ValueError as exc:
            # 周期解析失败以消息返回，不中断 stop 接口
            log_error("sync best strategy on stop failed", exc)
            sync_error = str(exc)
    return {
        "ok": stopped,
        "training": training_manager.status(),
        "strategy_file": strategy_file,
        "sync_error": sync_error,
    }


# ─────────────────────────────────────────────────────────────────────
# 回测 API
# ─────────────────────────────────────────────────────────────────────

_METRIC_KEYS = (
    "total_return", "sharpe", "sortino", "profit_loss_ratio",
    "n_trades", "win_rate", "avg_hold_bars",
)


def _load_backtest_report() -> dict[str, Any] | None:
    import json

    report_path = BACKTEST_OUTPUT_DIR / "multi_factor_report.json"
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _backtest_focus_symbol(symbol: str | None = None) -> str | None:
    """Resolve the symbol used to filter backtest charts/report for the web UI."""
    if symbol:
        return symbol.strip() or None

    job = backtest_manager.status().get("job") or {}
    if job.get("symbol"):
        return str(job["symbol"])

    strat = _strategy_context().get("strategy_file") or {}
    if strat.get("symbol"):
        return str(strat["symbol"])

    report = _load_backtest_report()
    if report:
        keys = list((report.get("symbols") or {}).keys())
        if len(keys) == 1:
            return keys[0]
    return None


def _filter_report_for_symbol(report: dict[str, Any], symbol: str) -> dict[str, Any]:
    symbols = report.get("symbols") or {}
    if symbol not in symbols:
        return report

    sym_data = symbols[symbol]
    return {
        **report,
        "focus_symbol": symbol,
        "symbols": {symbol: sym_data},
        "portfolio": {
            "total_return": sym_data.get("total_return"),
            "sharpe": sym_data.get("sharpe"),
            "sortino": sym_data.get("sortino"),
            "profit_loss_ratio": sym_data.get("profit_loss_ratio"),
            "n_trades": sym_data.get("n_trades"),
            "win_rate": sym_data.get("win_rate"),
        },
    }


def _list_backtest_charts(symbol: str | None = None) -> list[dict[str, str]]:
    """列出回测输出目录下的图表；单品种模式只返回该品种相关文件。"""
    if not BACKTEST_OUTPUT_DIR.exists():
        return []

    if symbol:
        charts: list[dict[str, str]] = []
        equity = BACKTEST_OUTPUT_DIR / "portfolio_equity.png"
        if equity.exists():
            charts.append(
                {"name": equity.name, "label": f"{symbol} 资金曲线", "kind": "equity"}
            )
        return charts

    charts = []
    portfolio = BACKTEST_OUTPUT_DIR / "portfolio_equity.png"
    if portfolio.exists():
        charts.append({"name": "portfolio_equity.png", "label": "组合资金曲线", "kind": "portfolio"})
    for path in sorted(BACKTEST_OUTPUT_DIR.glob("equity_*.png")):
        sym = path.stem.replace("equity_", "", 1)
        charts.append({"name": path.name, "label": f"{sym} 资金曲线", "kind": "symbol"})
    return charts


@app.get("/api/backtest/status")
def api_backtest_status() -> dict[str, Any]:
    status = backtest_manager.status()
    status["log_tail"] = backtest_manager.tail_log(200)
    return status


@app.post("/api/backtest/start")
def api_backtest_start(req: StartBacktestRequest) -> dict[str, Any]:
    info = _inspect_strategy_or_http(req.strategy_file)
    settings = load_settings()
    commission = (
        float(req.commission_pct)
        if req.commission_pct is not None
        else float(settings.get("bt_commission_pct", 0.02))
    )
    slippage = (
        float(req.slippage_pct)
        if req.slippage_pct is not None
        else float(settings.get("bt_slippage_pct", 0.01))
    )
    if commission < 0 or slippage < 0:
        raise HTTPException(400, "手续费和滑点不能为负数")

    save_settings({
        "last_strategy_file": info["strategy_file"],
        "bt_commission_pct": commission,
        "bt_slippage_pct": slippage,
    })

    data_file: str | None = None
    # 1) 优先用策略 JSON 里记录的训练数据路径
    strat_data = (info.get("data_file") or "").strip()
    if strat_data:
        try:
            pf = inspect_parquet_file(strat_data)
            if pf.get("valid") is False:
                raise HTTPException(
                    400,
                    f"策略记录的数据文件无效: {pf.get('message') or strat_data}",
                )
            data_file = pf["data_file"]
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                400,
                f"策略记录的数据文件无法加载: {strat_data}\n{e}",
            ) from e
    else:
        # 2) 回退：训练页最近选择的、同品种 Parquet
        last_data = settings.get("last_data_file") or ""
        if last_data:
            try:
                pf = inspect_parquet_file(last_data)
                if pf.get("symbol") == info.get("symbol") and pf.get("valid") is not False:
                    data_file = pf["data_file"]
            except Exception:
                pass

    if not data_file:
        raise HTTPException(
            400,
            "该策略未记录数据文件路径（data_file），且当前也没有同品种的 Parquet。"
            "请先在「模型训练」页选择对应品种的 Parquet 再回测；"
            "或使用本软件训练/导出、且包含 data_file 字段的策略文件。",
        )

    save_settings({"last_data_file": data_file})

    try:
        job = backtest_manager.start(
            strategy_file=info["strategy_file"],
            data_file=data_file,
            commission_pct=commission,
            slippage_pct=slippage,
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return {"ok": True, "job": job.to_dict(), "strategy_file": info, "data_file": data_file}


@app.post("/api/backtest/stop")
def api_backtest_stop() -> dict[str, Any]:
    stopped = backtest_manager.stop()
    return {"ok": stopped, "backtest": backtest_manager.status()}


@app.get("/api/backtest/report")
def api_backtest_report(symbol: str | None = None) -> dict[str, Any]:
    report = _load_backtest_report()
    focus = _backtest_focus_symbol(symbol)
    if report and focus:
        report = _filter_report_for_symbol(report, focus)
    return {
        "available": report is not None,
        "report": report,
        "charts": _list_backtest_charts(focus),
        "focus_symbol": focus,
    }


@app.get("/api/backtest/equity")
def api_backtest_equity(symbol: str | None = None) -> dict[str, Any]:
    """资金曲线原始数据（供前端渲染交互式 HTML 图表）。"""
    import json

    path = BACKTEST_OUTPUT_DIR / "equity_curve.json"
    focus = _backtest_focus_symbol(symbol)
    if not path.exists():
        return {"available": False, "focus_symbol": focus, "data": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"available": False, "focus_symbol": focus, "data": None}

    # 单品种模式：只保留聚焦品种，去掉无关序列
    if focus and isinstance(data.get("symbols"), dict) and focus in data["symbols"]:
        data = {
            **data,
            "symbols": {focus: data["symbols"][focus]},
        }
        data.pop("portfolio", None)

    return {"available": True, "focus_symbol": focus, "data": data}


@app.get("/api/backtest/chart/{name}")
def api_backtest_chart(name: str):
    # 防止路径穿越：仅允许输出目录内的 png 文件
    if "/" in name or "\\" in name or ".." in name or not name.lower().endswith(".png"):
        raise HTTPException(400, "非法文件名")
    path = (BACKTEST_OUTPUT_DIR / name).resolve()
    try:
        path.relative_to(BACKTEST_OUTPUT_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "非法路径") from None
    if not path.exists():
        raise HTTPException(404, "图表不存在")
    return FileResponse(path, media_type="image/png")


# ─────────────────────────────────────────────────────────────────────
# 实时行情分析 API
# ─────────────────────────────────────────────────────────────────────


@app.on_event("startup")
def _startup_realtime() -> None:
    try:
        realtime_manager.load_persisted()
    except Exception as exc:  # noqa: BLE001
        log_error("realtime load_persisted failed", exc)


@app.get("/api/realtime/sources")
def api_realtime_sources() -> dict[str, Any]:
    return {"sources": list_sources(), "min_exposure": min_exposure()}


@app.post("/api/realtime/tradingview/probe")
def api_realtime_tradingview_probe() -> dict[str, Any]:
    """Probe TradingView reachability (same behavior as PA_Agent before fetch)."""
    from web.data_sources.tradingview_connectivity import (
        TV_CLOUD_SERVER_WIKI_URL,
        TV_CONNECTIVITY_MESSAGE,
        check_tradingview_connectivity,
    )

    ok, detail = check_tradingview_connectivity(
        timeout_s=15.0, max_attempts=2, retry_delay_s=2.0
    )
    return {
        "ok": ok,
        "detail": detail,
        "blocked": not ok,
        "title": "无法使用 TradingView",
        "message": None if ok else TV_CONNECTIVITY_MESSAGE,
        "wiki_url": TV_CLOUD_SERVER_WIKI_URL,
    }


@app.get("/api/realtime/strategies")
def api_realtime_strategies() -> dict[str, Any]:
    """已保存的新命名 best_{{symbol}}_{{tf}}.json 策略，供因子来源下拉。"""
    rows = []
    for s in list_strategies():
        sym = s.get("symbol")
        tf = s.get("timeframe")
        if not sym or not tf:
            continue
        try:
            path = strategy_path_for_symbol(sym, tf)
        except ValueError:
            continue  # 周期非法：跳过而非 500
        if not path.exists():
            continue
        rows.append(
            {
                "symbol": sym,
                "timeframe": tf,
                "best_score": s.get("best_score"),
                "formula_decoded": s.get("formula_decoded"),
                "strategy_file": str(path.resolve()),
            }
        )
    return {"strategies": rows}


@app.get("/api/realtime/status")
def api_realtime_status() -> dict[str, Any]:
    return realtime_manager.status()


@app.post("/api/realtime/watch")
def api_realtime_watch(req: AddWatchRequest) -> dict[str, Any]:
    try:
        watch = realtime_manager.add_watch(
            req.source, req.symbol, req.timeframe, req.strategy_file
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "watch": watch}


@app.post("/api/realtime/unwatch")
def api_realtime_unwatch(req: RemoveWatchRequest) -> dict[str, Any]:
    removed = realtime_manager.remove_watch(req.id)
    return {"ok": removed}


@app.post("/api/realtime/start")
def api_realtime_start() -> dict[str, Any]:
    realtime_manager.start()
    return {"ok": True, **realtime_manager.status()}


@app.post("/api/realtime/stop")
def api_realtime_stop() -> dict[str, Any]:
    realtime_manager.stop()
    return {"ok": True, "running": False}


@app.get("/api/realtime/feishu")
def api_realtime_feishu_get() -> dict[str, Any]:
    s = load_settings()
    return {
        "enabled": bool(s.get("feishu_enabled")),
        "webhook_url": s.get("feishu_webhook_url") or "",
        "secret": s.get("feishu_secret") or "",
    }


@app.put("/api/realtime/feishu")
def api_realtime_feishu_put(req: FeishuSettingsRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if req.enabled is not None:
        payload["feishu_enabled"] = bool(req.enabled)
    if req.webhook_url is not None:
        payload["feishu_webhook_url"] = req.webhook_url
    if req.secret is not None:
        payload["feishu_secret"] = req.secret
    saved = save_settings(payload)
    return {
        "ok": True,
        "enabled": bool(saved.get("feishu_enabled")),
        "webhook_url": saved.get("feishu_webhook_url") or "",
        "secret": saved.get("feishu_secret") or "",
    }


@app.post("/api/realtime/feishu/test")
def api_realtime_feishu_test(req: FeishuTestRequest) -> dict[str, Any]:
    from web.feishu_notify import send_text

    url = (req.webhook_url or "").strip()
    if not url:
        url = (load_settings().get("feishu_webhook_url") or "").strip()
    if not url:
        raise HTTPException(400, "请先填写 Webhook URL")
    secret = req.secret
    if secret is None:
        secret = load_settings().get("feishu_secret") or ""
    ok, msg = send_text(
        "✅ AlphaMaster 飞书通知测试：配置正常。信号方向转折时会推送提醒。",
        webhook_url=url,
        secret=secret or "",
    )
    if not ok:
        raise HTTPException(400, msg)
    return {"ok": True, "message": msg}


# ─────────────────────────────────────────────────────────────────────
# 实盘信号台账 API（spec §5.6）
#
# 总原则（spec §6）：台账是观测者 —— 这五个端点任何故障一律 200 + {"error": 原因}，
# 绝不 500，前端直接把原因显示成角标/提示。
# ─────────────────────────────────────────────────────────────────────

_LEDGER_DAYS_MIN = 1
_LEDGER_DAYS_MAX = 3650        # 跨度上限：理论重算是 O(窗口 bar × 1001 bar VM) 懒算
_LEDGER_LIMIT_MAX = 5000
_LEDGER_CSV_COLUMNS = (
    "ts", "bar_ts", "source", "symbol", "timeframe", "strategy_sha",
    "factor_value", "position", "direction", "strength", "close", "pnl",
)
_DEFAULT_COST_RATE = 0.0003    # 与 config.Config.COST_RATE 同值，读不到时兜底


def _ledger_fail(op: str, exc: BaseException) -> dict[str, Any]:
    """台账端点统一失败出口：HTTP 200 + error 文本（spec §6 不 500）。

    ValueError 是用户可触发的输入/目录问题，只 warn；其余按异常记错误日志。
    """
    if isinstance(exc, ValueError):
        logger.warning("[ledger] %s 输入/数据不可用: %s", op, exc)
    else:
        log_error(f"[ledger] {op} failed", exc)
    return {"error": str(exc)}


def _ledger_cost_rate() -> float:
    """调用时取成本口径（spec §5.4）：web settings 的 bt_commission_pct + bt_slippage_pct，
    口径与 02 回测链同源（api_backtest_start 读同名键，百分数 → /100）。
    键缺失/不可解析才兜底 Config.COST_RATE。绝不打印 settings 内容（文件含凭据）。"""
    try:
        settings = load_settings()
        commission = float(settings.get("bt_commission_pct"))
        slippage = float(settings.get("bt_slippage_pct"))
        return (commission + slippage) / 100.0
    except (TypeError, ValueError):
        pass
    try:
        from config import Config

        return float(getattr(Config, "COST_RATE", _DEFAULT_COST_RATE))
    except Exception:  # noqa: BLE001
        return _DEFAULT_COST_RATE


def _ledger_root() -> Path:
    """LEDGER_DIR 必须每次调用时读模块属性（测试 monkeypatch 才生效）。"""
    return Path(LL.LEDGER_DIR)


def _ledger_stem_meta(stem: str) -> tuple[str, str, str]:
    """账本文件名 stem → (source, symbol, tf)：首段 source、末段 tf、中间 symbol。

    只在账本没有任何记录（取不到记录字段）时兜底用；有记录时一律以记录字段为准。
    """
    parts = stem.split("_")
    if len(parts) < 3:
        return "", "", ""
    return parts[0], "_".join(parts[1:-1]), parts[-1]


def _ledger_resolve(watch_id: str) -> Path:
    """watch_id → 账本路径。

    watch_id 是会被拼成文件名的用户输入，因此不做拼接式猜测：先拒绝路径分隔与穿越，
    再与 LEDGER_DIR 目录清单里的真实文件名精确比对（只有实际存在的 `{watch_id}.jsonl`
    才可用）。`.theo.jsonl` 是派生缓存、`strategy_index.json` 是索引，都不作为账本暴露。
    """
    wid = (watch_id or "").strip()
    if not wid or "/" in wid or "\\" in wid or ".." in wid:
        raise ValueError(f"非法 watch_id: {watch_id!r}")
    if wid.endswith(".theo"):
        raise ValueError(f"非法 watch_id（理论曲线缓存不是账本）: {watch_id!r}")
    root = _ledger_root()
    try:
        names = {p.name for p in root.iterdir() if p.is_file()}
    except OSError as exc:
        raise ValueError(f"台账目录不可用: {root}（{exc.strerror or exc}）") from None
    name = f"{wid}.jsonl"
    if name not in names:
        raise ValueError(f"账本不存在: {wid}")
    return root / name


def _ledger_scope(watch_id: str, days: int | None, sha: str | None) -> dict[str, Any]:
    """定位账本 → days 截窗（以末条 bar_ts 为锚）→ sha 段过滤，返回共用统计上下文。

    days=None 表示不截窗（/records 全量分页）；sha 缺省取窗口内最新段的 sha。
    """
    path = _ledger_resolve(watch_id)
    if days is not None:
        days = int(days)
        if days < _LEDGER_DAYS_MIN or days > _LEDGER_DAYS_MAX:
            raise ValueError(f"days 需在 {_LEDGER_DAYS_MIN}..{_LEDGER_DAYS_MAX} 之间")
    recs, corrupt = LL.read_records(path)
    recs.sort(key=lambda r: int(r.bar_ts))        # 读的是写入顺序，统计按 bar_ts 时序
    source, symbol, tf = _ledger_stem_meta(path.stem)
    anchor = None
    if recs:
        source, symbol, tf = recs[-1].source, recs[-1].symbol, recs[-1].timeframe
        anchor = int(recs[-1].bar_ts)
    start = anchor - days * 86400 if anchor is not None and days is not None else None
    window = recs if start is None else [r for r in recs if int(r.bar_ts) >= start]
    active_sha = sha or (window[-1].strategy_sha if window else None)
    # sha 过滤在 derive_rows 之前：跨段相邻本就无收益意义（spec §5.5 相邻性规则），
    # 代价是段边界那一根的 pnl 记 missing —— spec §5.5「分段统计」明确接受。
    seg = [r for r in window if active_sha is None or r.strategy_sha == active_sha]
    return {
        "path": path, "watch_id": path.stem, "source": source, "symbol": symbol,
        "timeframe": tf, "corrupt_lines": corrupt, "records_all": len(recs),
        "window": window, "seg": seg, "sha": active_sha,
        "start_ts": start, "anchor_ts": anchor, "days": days,
    }


def _theo_rows_from_cache(cache_rows: list[dict], scope: dict[str, Any],
                          cost_rate: float) -> list[dict]:
    """`.theo.jsonl` 行（bar_ts/position/close/cost_rate）→ 复用 derive_rows 出 pnl。

    缓存只存仓位与收盘价（B4 口径），收益数学仍由 ledger_stats 单点实现，
    这里只做形状还原，绝不另起一套 pnl 公式。
    """
    recs = [
        LL.LedgerRecord(
            ts=float(r["bar_ts"]), bar_ts=int(r["bar_ts"]),
            source=scope["source"], symbol=scope["symbol"],
            timeframe=scope["timeframe"], strategy_sha="",
            factor_value=0.0,
            position=None if r.get("position") is None else float(r["position"]),
            direction="", strength=0.0, close=float(r["close"]),
        )
        for r in cache_rows
    ]
    return LS.derive_rows(recs, tf=scope["timeframe"], cost_rate=cost_rate)


def _ledger_theo_rows(scope: dict[str, Any], cost_rate: float) -> tuple[list[dict], str | None]:
    """理论曲线行（spec §5.4 懒算）：新鲜且覆盖窗口的缓存优先，否则按需重算。

    返回 (窗口内 theo 行, None) 或 ([], reason)。theo_is_fresh 只看 mtime+cost_rate，
    跨度覆盖由本调用方检查（B4 ruling）；缓存形状异常按不新鲜处理。
    """
    start, anchor = scope["start_ts"], scope["anchor_ts"]
    if not scope["seg"]:
        return [], "窗口内无实盘记录"
    if start is None or anchor is None:
        return [], "账本为空"
    ledger = scope["path"]
    theo = LS._theo_path_for(ledger)
    if LS.theo_is_fresh(ledger, theo, cost_rate):
        cached = LS.load_theo_cache(theo)
        rows: list[dict] = []
        if cached is not None:
            try:
                rows = _theo_rows_from_cache(cached, scope, cost_rate)
            except (TypeError, ValueError, KeyError):
                rows = []           # 缓存形状不对 → 落到重算分支
        # 跨度覆盖检查（B4 ruling 留给调用方）：缓存首根必须就在窗口起点那一根上
        # （窗口按 days*86400 截，tf 整除一天时天然对齐；容差一整个 bar 覆盖
        #  起点落在两根之间的周期，如 1w/1M）。首根更晚 = 上一次请求的窗口更短，
        #  缓存不覆盖本窗口 → 重算。
        tf_s = LS._tf_seconds(scope["timeframe"])
        rows = [r for r in rows if start <= int(r["bar_ts"]) <= anchor]
        if rows and int(rows[0]["bar_ts"]) < start + tf_s:
            return rows, None
    entry = LL.load_strategy_index(_ledger_root()).get(scope["sha"] or "")
    formula = entry.get("formula") if isinstance(entry, dict) else None
    if not formula:
        return [], "公式索引缺失"
    fresh, reason = LS.recompute_theo(
        scope["source"], scope["symbol"], scope["timeframe"], formula,
        start_ts=start, end_ts=anchor, cost_rate=cost_rate,
        ledger_dir=_ledger_root(),
    )
    if reason is not None:
        return [], reason
    return [r for r in fresh if start <= int(r["bar_ts"]) <= anchor], None


@app.get("/api/ledger/watches")
def api_ledger_watches() -> dict[str, Any]:
    """账本文件清单（含已删 watch 的历史账，spec §5.6/§6）。"""
    try:
        root = _ledger_root()
        try:
            files = sorted(p for p in root.iterdir() if p.is_file())
        except OSError:
            files = []             # 目录还没建：空清单，不是错误
        out: list[dict[str, Any]] = []
        for p in files:
            if not p.name.endswith(".jsonl") or p.name.endswith(".theo.jsonl"):
                continue
            recs, corrupt = LL.read_records(p)
            source, symbol, tf = _ledger_stem_meta(p.stem)
            last = recs[-1] if recs else None
            if last is not None:
                source, symbol, tf = last.source, last.symbol, last.timeframe
            out.append({
                "watch_id": p.stem,
                "source": source,
                "symbol": symbol,
                "timeframe": tf,
                "records": len(recs),
                "corrupt_lines": corrupt,
                "last_bar_ts": int(last.bar_ts) if last is not None else None,
                "has_theo_cache": LS._theo_path_for(p).exists(),
                "size_bytes": p.stat().st_size,
            })
        return {"watches": out, "ledger_dir": str(root)}
    except Exception as exc:  # noqa: BLE001
        return _ledger_fail("watches", exc)


@app.get("/api/ledger/summary")
def api_ledger_summary(watch_id: str, days: int = 30, sha: str | None = None) -> dict[str, Any]:
    """统计卡数据：records/win_rate/profit_factor/annualized/sortino/max_drawdown
    + corrupt_lines + 完整率 + 理论可用状态（spec §5.5/§5.6）。

    completeness = 窗口记录数 ÷ max(记录数, 同窗口源收盘 bar 数)，源不可用 → null。
    """
    try:
        cost_rate = _ledger_cost_rate()
        scope = _ledger_scope(watch_id, days, sha)
        live_rows = LS.derive_rows(scope["seg"], tf=scope["timeframe"], cost_rate=cost_rate)
        stats = LS.compute_stats(live_rows, cost_rate, tf=scope["timeframe"])
        theo_rows, theo_reason = _ledger_theo_rows(scope, cost_rate)
        n = len(scope["seg"])
        completeness = None
        if theo_reason is None and n:
            # 跨度覆盖判定沿用 _ledger_theo_rows 缓存分支同一口径：theo 首根必须落在
            # 窗口起点一根 bar 容差内。theo 有行但起步晚于窗口（源截断/退市）→ 覆盖
            # 不足，完整率报 null 而非 1.0（记录数÷max 的分母口径此时失真）。
            covered = False
            if theo_rows and scope["start_ts"] is not None:
                tf_s = LS._tf_seconds(scope["timeframe"])
                theo_first = min(int(r["bar_ts"]) for r in theo_rows)
                covered = theo_first < int(scope["start_ts"]) + tf_s
            else:
                covered = not theo_rows
            if covered:
                completeness = n / max(n, len(theo_rows))
        divergence = LS.align_divergence(live_rows, theo_rows) if theo_reason is None else []
        return {
            "watch_id": scope["watch_id"],
            "source": scope["source"],
            "symbol": scope["symbol"],
            "timeframe": scope["timeframe"],
            "days": days,
            "sha": scope["sha"],
            "window_start_ts": scope["start_ts"],
            "window_end_ts": scope["anchor_ts"],
            "records": n,
            "corrupt_lines": scope["corrupt_lines"],
            "completeness": completeness,
            "theo_available": theo_reason is None,
            "theo_reason": theo_reason,
            "theo_bars": len(theo_rows),
            "divergence_count": len(divergence),
            "divergence_bars": divergence[:200],
            **stats,
        }
    except Exception as exc:  # noqa: BLE001
        return _ledger_fail("summary", exc)


@app.get("/api/ledger/curve")
def api_ledger_curve(watch_id: str, days: int = 30, sha: str | None = None) -> dict[str, Any]:
    """实盘/理论累计曲线叠图序列：[{bar_ts, cum_live, cum_theo, theo_missing}]。

    cum_* 为 derive_rows 的 pnl 前缀和（missing 行不前移）；theo 侧按 bar_ts join，
    缺口记 theo_missing（spec §5.4/§6）。
    """
    try:
        cost_rate = _ledger_cost_rate()
        scope = _ledger_scope(watch_id, days, sha)
        live_rows = LS.derive_rows(scope["seg"], tf=scope["timeframe"], cost_rate=cost_rate)
        theo_rows, theo_reason = _ledger_theo_rows(scope, cost_rate)
        theo_cum: dict[int, float] = {}
        acc = 0.0
        for r in sorted(theo_rows, key=lambda x: int(x["bar_ts"])):
            if not r.get("missing") and r.get("pnl") is not None:
                acc += float(r["pnl"])
            theo_cum[int(r["bar_ts"])] = acc
        curve: list[dict[str, Any]] = []
        acc = 0.0
        for r in live_rows:                # derive_rows 已按 bar_ts 升序
            if not r["missing"] and r["pnl"] is not None:
                acc += float(r["pnl"])
            ts = int(r["bar_ts"])
            curve.append({
                "bar_ts": ts,
                "cum_live": acc,
                "cum_theo": theo_cum.get(ts),
                "theo_missing": ts not in theo_cum,
            })
        return {
            "watch_id": scope["watch_id"],
            "timeframe": scope["timeframe"],
            "days": days,
            "sha": scope["sha"],
            "theo_available": theo_reason is None,
            "theo_reason": theo_reason,
            "curve": curve,
        }
    except Exception as exc:  # noqa: BLE001
        return _ledger_fail("curve", exc)


@app.get("/api/ledger/records")
def api_ledger_records(
    watch_id: str,
    days: int | None = None,
    direction: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """明细分页：账本原字段 + 派生 pnl（相邻才有值）。

    days 省略即全量历史；本端点不按 sha 过滤（明细要看完整流水），
    排序新→旧，便于前端表格直接展示。
    """
    try:
        cost_rate = _ledger_cost_rate()
        scope = _ledger_scope(watch_id, days, None)
        recs = scope["window"]
        rows = LS.derive_rows(recs, tf=scope["timeframe"], cost_rate=cost_rate)
        # derive_rows 内部 stable sort 同键同序 → 与已排序的 recs 逐下标对齐
        items = [
            {**asdict(r), "pnl": row["pnl"], "missing": row["missing"]}
            for r, row in zip(recs, rows)
        ]
        if direction:
            want = str(direction).upper()
            items = [x for x in items if str(x.get("direction") or "").upper() == want]
        total = len(items)
        lim = max(1, min(int(limit), _LEDGER_LIMIT_MAX))
        off = max(0, int(offset))
        page = list(reversed(items))[off:off + lim]
        return {
            "watch_id": scope["watch_id"],
            "days": days,
            "direction": direction,
            "total": total,
            "limit": lim,
            "offset": off,
            "records": page,
        }
    except Exception as exc:  # noqa: BLE001
        return _ledger_fail("records", exc)


@app.get("/api/ledger/export")
def api_ledger_export(watch_id: str, days: int = 30, sha: str | None = None):
    """CSV 下载：账本原始行 + 派生 pnl 列（spec §5.6）。失败同样 200 + error。"""
    from fastapi.responses import Response

    try:
        cost_rate = _ledger_cost_rate()
        scope = _ledger_scope(watch_id, days, sha)
        recs = scope["seg"]
        rows = LS.derive_rows(recs, tf=scope["timeframe"], cost_rate=cost_rate)
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(_LEDGER_CSV_COLUMNS)
        for r, row in zip(recs, rows):
            data = asdict(r)
            data["pnl"] = row["pnl"]
            writer.writerow([data[c] for c in _LEDGER_CSV_COLUMNS])
        filename = f'{scope["watch_id"]}_{days}d.csv'
        return Response(
            content=buf.getvalue().encode("utf-8"),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:  # noqa: BLE001
        return _ledger_fail("export", exc)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")



app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
