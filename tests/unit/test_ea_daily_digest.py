# tests/unit/test_ea_daily_digest.py
"""ea_daily_digest 的纯函数层：不碰 MetaTrader5 包，全部吃合成 dict。

覆盖的六条都是汇总里"会说谎"的地方，所以逐条钉死：
① EA 日志行的解析（制表符分列 + bar=/z=/tanh=/dir=），非信号行必须被丢掉；
② MT5 返回的是服务器钟(UTC+2)，转本机时刻必须 −7200，否则汇总里的时间全错两小时；
③ dir 分布与翻转次数（"今天翻了几次"是换手成本的直接读数）；
④ 缺失 bar 检测（周末/断线留下的洞，不能当成"EA 判断空仓"）；
⑤ magic 分组：EA 单 magic=20250101、手工单 magic=0，两者混在一个账户里，
   已实现盈亏必须各自结算，且净利含 commission+swap（只加 profit 会虚高）；
⑥ 手工干预检测：magic=0 平掉 EA 开的仓、或 EA 仓被手工挂的 tp/sl 打掉——
   这会让"样本外跟踪"被污染，必须在报告里点名而不是悄悄算进业绩。
"""
from __future__ import annotations

import datetime as dt

import pytest

import ea_daily_digest as D


# ── ① 日志解析 ────────────────────────────────────────────────
def test_parse_signal_lines_reads_bar_z_dir_and_skips_noise():
    text = (
        "NJ\t0\t11:30:00.142\tAlphaMaster_XAUUSD (XAUUSD,M30)\t"
        "[AlphaMaster] bar=2026.11.06 05:00 closedBars=2000 z=-0.08899 tanh=-0.08875 dir=-1\n"
        "QM\t0\t14:00:00.179\tAlphaMaster_XAUUSD (XAUUSD,M30)\t"
        "[AlphaMaster] bar=2026.11.06 07:30 closedBars=2000 z=-0.37328 tanh=-0.35686 dir=-1\n"
        "XX\t0\t14:00:01.000\tSomeOtherEA (EURUSD,H1)\twhatever\n"
        "YY\t0\t14:30:00.000\tAlphaMaster_XAUUSD (XAUUSD,M30)\t"
        "[AlphaMaster] 无足够 bar，跳过\n"
    )

    sigs = D.parse_signal_lines(text)

    assert len(sigs) == 2, "只有带 bar=/z=/dir= 的行算信号"
    assert sigs[0]["bar"] == dt.datetime(2026, 11, 6, 5, 0)
    assert sigs[0]["z"] == pytest.approx(-0.08899)
    assert sigs[0]["dir"] == -1
    assert sigs[1]["bar"] == dt.datetime(2026, 11, 6, 7, 30)
    assert sigs[1]["dir"] == -1


def test_parse_signal_lines_accepts_positive_z_and_dir():
    """真机踩过的坑：dir=1 的行写成 z=+0.64091 tanh=+0.56552。正则漏掉正号会把
    所有多头信号静默丢掉（11-04/11-05 两个整日日志解析出 0 根），汇总只剩空头那天。"""
    text = (
        "AA\t0\t00:00:00.206\tAlphaMaster_XAUUSD (XAUUSD,M30)\t"
        "[AlphaMaster] bar=2026.11.05 17:30 closedBars=2000 z=+0.64091 tanh=+0.56552 dir=1\n"
        "BB\t0\t00:30:00.246\tAlphaMaster_XAUUSD (XAUUSD,M30)\t"
        "[AlphaMaster] bar=2026.11.05 18:00 closedBars=2000 z=+0.64091 tanh=+0.56552 dir=1\n"
    )

    sigs = D.parse_signal_lines(text)

    assert len(sigs) == 2
    assert sigs[0]["z"] == pytest.approx(0.64091)
    assert sigs[0]["tanh"] == pytest.approx(0.56552)
    assert sigs[0]["dir"] == 1


# ── ② 时间基准 ────────────────────────────────────────────────
def test_to_local_uses_measured_server_offset():
    """MT5 的 epoch 字段是「服务器墙上钟当 UTC」编码，实测对拉：
    utcfromtimestamp 给服务器墙钟，再加本机与服务器的时差才是本机时刻。"""
    epoch = dt.datetime(2026, 11, 6, 5, 30).replace(
        tzinfo=dt.timezone.utc).timestamp()

    local = D.to_local(epoch, dt.timedelta(hours=6))

    assert local == dt.datetime(2026, 11, 6, 11, 30)


def test_measure_offset_from_live_tick():
    """本机(now) − 服务器墙钟(tick) 的原始差里混着 tick 年龄噪声；时差本身是整分钟量，
    所以取整到分钟。不取整的话 24 小时汇总里每笔成交都少 1 分钟（10:59 / 11:29），
    跟终端日志和 EA 打印的时刻对不上。"""
    tick_epoch = dt.datetime(2026, 11, 6, 5, 30).replace(
        tzinfo=dt.timezone.utc).timestamp()

    off = D.measure_offset(dt.datetime(2026, 11, 6, 11, 30, 5), tick_epoch)

    assert off == dt.timedelta(hours=6)


def test_measure_offset_survives_truncated_now_and_stale_tick():
    """窗口右端把秒抹成 0（t1），tick 又可能几十秒没更新——两个误差方向相反，
    取整到分钟后仍要得到真实的 6 小时时差。"""
    tick_epoch = dt.datetime(2026, 11, 6, 8, 50, 29).replace(
        tzinfo=dt.timezone.utc).timestamp()

    off = D.measure_offset(dt.datetime(2026, 11, 6, 14, 50), tick_epoch)

    assert off == dt.timedelta(hours=6)


def test_measure_offset_tracks_dst_shift():
    """Dukascopy 换季把 GMT+2 挪成 +1 时，同一台本机北京(+8)的时差应是 5 小时，
    不是写死的 6 小时。"""
    tick_epoch = dt.datetime(2026, 3, 30, 4, 0).replace(
        tzinfo=dt.timezone.utc).timestamp()

    off = D.measure_offset(dt.datetime(2026, 3, 30, 9, 0), tick_epoch)

    assert off == dt.timedelta(hours=5)


def test_to_local_wall_shifts_only_the_offset():
    """日志里 TimeToString 印的已经是服务器墙钟，不能再走 epoch 还原。"""
    assert D.to_local_wall(dt.datetime(2026, 11, 5, 17, 30),
                          dt.timedelta(hours=6)) == dt.datetime(2026, 11, 5, 23, 30)


# ── ③④ 信号统计 ──────────────────────────────────────────────
def _sigs(bars, dirs, z=-0.1):
    return [{"bar": b, "z": z, "tanh": z, "dir": d}
            for b, d in zip(bars, dirs)]


def test_summarize_signals_counts_dirs_and_flips():
    base = dt.datetime(2026, 11, 6, 0, 0)
    bars = [base + dt.timedelta(minutes=30 * i) for i in range(6)]
    sigs = _sigs(bars, [1, 1, 0, -1, -1, 1])

    s = D.summarize_signals(sigs)

    assert s["n"] == 6
    assert s["dir_counts"] == {"long": 3, "short": 2, "flat": 1}
    assert s["flips"] == 3, "1→0、0→-1、-1→1 共三次翻转"
    assert s["first_bar"] == bars[0] and s["last_bar"] == bars[-1]
    assert s["missing_bars"] == 0


def test_summarize_signals_flags_bar_gaps():
    bars = [dt.datetime(2026, 11, 6, 0, 0), dt.datetime(2026, 11, 6, 0, 30),
            dt.datetime(2026, 11, 6, 2, 0)]  # 01:00/01:30 两根缺

    s = D.summarize_signals(_sigs(bars, [1, 1, -1]))

    assert s["missing_bars"] == 2, "相邻信号间隔 90 分钟 ⇒ 缺 2 根"
    assert s["expected_bars"] == 5


def test_summarize_signals_on_empty_input_is_not_a_crash():
    s = D.summarize_signals([])

    assert s["n"] == 0 and s["flips"] == 0 and s["missing_bars"] == 0
    assert s["dir_counts"] == {"long": 0, "short": 0, "flat": 0}
    assert s["z_min"] is None and s["z_max"] is None


# ── ⑤ magic 分组与净利口径 ───────────────────────────────────
def _deal(ticket, magic, entry, typ, price, profit=0.0, comm=0.0, swap=0.0,
          pos=None, comment="", vol=0.01):
    return {"ticket": ticket, "magic": magic, "entry": entry, "type": typ,
            "price": price, "profit": profit, "commission": comm, "swap": swap,
            "volume": vol, "position_id": pos if pos is not None else ticket,
            "comment": comment, "symbol": "XAUUSD"}


def test_classify_deals_separates_ea_from_manual_and_nets_costs():
    deals = [
        _deal(1, D.EA_MAGIC, 0, 0, 3918.805, comm=-1.31, pos=100),
        _deal(2, D.EA_MAGIC, 1, 1, 3950.015, profit=208.18, comm=-1.32, swap=-3.97, pos=100),
        _deal(3, 0, 1, 0, 3920.386, profit=-9.94, comm=-1.31, pos=200),
    ]

    c = D.classify_deals(deals)

    assert [d["ticket"] for d in c["ea"]] == [1, 2]
    assert [d["ticket"] for d in c["manual"]] == [3]
    assert c["ea_open_n"] == 1 and c["ea_close_n"] == 1
    assert c["ea_net"] == pytest.approx(208.18 - 1.31 - 1.32 - 3.97), "净利必须含佣金与隔夜费"
    assert c["manual_net"] == pytest.approx(-9.94 - 1.31)
    assert c["commission"] == pytest.approx(-1.31 - 1.32 - 1.31)
    assert c["swap"] == pytest.approx(-3.97)


def test_classify_deals_on_empty_window():
    c = D.classify_deals([])

    assert c["ea"] == [] and c["manual"] == []
    assert c["ea_net"] == 0.0 and c["manual_net"] == 0.0
    assert c["ea_open_n"] == 0 and c["ea_close_n"] == 0


# ── ⑥ 手工干预检测 ──────────────────────────────────────────
def test_detect_interference_names_manual_close_of_ea_position():
    # EA 开仓(magic 20250101, position_id 100) → 被 magic=0 平掉
    deals = [_deal(1, D.EA_MAGIC, 0, 0, 3954.135, pos=100),
             _deal(2, 0, 1, 1, 4038.875, profit=565.2, comm=-1.34, pos=100)]

    hits = D.detect_interference(deals)

    assert len(hits) == 1
    assert "100" in hits[0] and "手工" in hits[0]


def test_detect_interference_names_manual_tp_sl_on_ea_position():
    deals = [_deal(1, D.EA_MAGIC, 0, 0, 3918.805, pos=100),
             _deal(2, D.EA_MAGIC, 1, 1, 3950.015, profit=208.18, pos=100,
                   comment="[tp 3950.000]")]

    hits = D.detect_interference(deals)

    assert len(hits) == 1 and "3950.000" in hits[0]


def test_detect_interference_is_silent_for_pure_ea_roundtrip():
    deals = [_deal(1, D.EA_MAGIC, 0, 0, 3918.805, pos=100),
             _deal(2, D.EA_MAGIC, 1, 1, 3950.015, profit=208.18, pos=100,
                   comment="AlphaMaster_XAUUSD")]

    assert D.detect_interference(deals) == []


# ── 持仓读数 ─────────────────────────────────────────────────
def test_open_positions_reports_direction_floating_and_naked_flag():
    positions = [{"magic": D.EA_MAGIC, "symbol": "XAUUSD", "type": 1,
                  "volume": 0.01, "price_open": 4049.215, "price_current": 4048.123,
                  "profit": 7.33, "sl": 0.0, "tp": 0.0, "ticket": 9,
                  "time_open": dt.datetime(2026, 11, 6, 11, 30)}]

    o = D.open_positions(positions)

    assert o["n"] == 1 and o["side"] == "short" and o["floating"] == pytest.approx(7.33)
    assert o["naked"] is True, "sl==tp==0 必须标裸仓"
    assert o["ea_n"] == 1


def test_open_positions_flat_when_no_ea_position():
    other = [{"magic": 0, "symbol": "XAUUSD", "type": 0, "volume": 0.05,
              "price_open": 4000.0, "price_current": 4001.0, "profit": 5.0,
              "sl": 3990.0, "tp": 4010.0, "ticket": 8,
              "time_open": dt.datetime(2026, 11, 6, 1, 0)}]

    o = D.open_positions(other + [{"magic": D.EA_MAGIC, "symbol": "XAUUSD", "type": 0,
                                   "volume": 0.01, "price_open": 1.0,
                                   "price_current": 1.0, "profit": 0.0,
                                   "sl": 0.5, "tp": 1.5, "ticket": 7,
                                   "time_open": dt.datetime(2026, 11, 6, 2, 0)}])

    assert o["n"] == 2 and o["ea_n"] == 1
    assert o["side"] == "long" and o["naked"] is False


# ── 报告成型 ─────────────────────────────────────────────────
def _ctx(**kw):
    base = {
        "account": {"login": 222139413, "server": "Dukascopy-demo-mt5-1",
                    "trade_mode": "DEMO", "currency": "CNH",
                    "balance": 100909.35, "equity": 100916.68,
                    "margin": 539.8, "floating": 7.33},
        "window": {"from": dt.datetime(2026, 11, 5, 11, 0),
                   "to": dt.datetime(2026, 11, 6, 11, 0)},
        "signals": {"n": 44, "flips": 6, "missing_bars": 0, "expected_bars": 44,
                    "dir_counts": {"long": 20, "short": 18, "flat": 6},
                    "z_min": -0.9, "z_max": 0.8,
                    "first_bar": dt.datetime(2026, 11, 5, 11, 0),
                    "last_bar": dt.datetime(2026, 11, 6, 10, 30)},
        "deals": {"ea": [_deal(1, D.EA_MAGIC, 1, 1, 4060.465, profit=177.82,
                               comm=-1.36, swap=-15.97, pos=102,
                               comment="AlphaMaster_XAUUSD")],
                  "manual": [], "ea_open_n": 1, "ea_close_n": 1,
                  "ea_net": 177.82, "manual_net": 0.0, "commission": -2.67,
                  "swap": -15.97},
        "positions": {"n": 1, "ea_n": 1, "side": "short", "floating": 7.33,
                      "naked": True, "rows": []},
        "interference": ["仓 #100 由手工平仓（magic=0）"],
        "health": {"terminal_running": True, "log_lines": 44,
                   "connection_lost": 2, "trade_allowed": True, "warnings": []},
    }
    base.update(kw)
    return base


def test_format_report_shows_account_window_pnl_and_caveats():
    txt = D.format_report(_ctx())

    assert "DEMO" in txt and "222139413" in txt
    assert "CNH" in txt, "盈亏币种必须标明，不能让人当美元读"
    assert "177.82" in txt and "-15.97" in txt
    assert "裸仓" in txt, "sl==tp==0 要在报告里可见"
    assert "手工" in txt and "仓 #100" in txt
    assert "2026-11-05 11:00" in txt and "2026-11-06 11:00" in txt


def test_report_title_uses_the_window_length():
    """--hours 48（周一补周五下午）时标题不能还写「24 小时」。"""
    ctx = _ctx(window={"from": dt.datetime(2026, 10, 30, 11),
                       "to": dt.datetime(2026, 11, 1, 11)})

    assert "48 小时汇总" in D.format_report(ctx)


def test_format_report_shows_measured_clock_offset():
    """所有时刻都是「服务器墙钟 + 实测时差」换算出来的。一旦时差估错（比如拿抹掉秒的
    窗口右端去减 tick），报告里每个时间都整齐地错一分钟，肉眼看不出来。
    所以把实测时差本身印出来。"""
    txt = D.format_report(_ctx(health=dict(_ctx()["health"],
                                          clock_offset=dt.timedelta(hours=6))))
    assert "时钟差 +6:00" in txt

    txt2 = D.format_report(_ctx(health=dict(_ctx()["health"],
                                           clock_offset=dt.timedelta(hours=5, minutes=30))))
    assert "时钟差 +5:30" in txt2, "半小时时区不能印成 +5:300 或 +5.5"


def test_format_report_shouts_when_autotrading_is_off():
    """AutoTrading 关了，"今天没成交"的意思完全变了 —— 必须看得见。"""
    ctx = _ctx()
    ctx["health"] = dict(ctx["health"], trade_allowed=False)
    ctx["deals"] = {"ea": [], "manual": [], "ea_open_n": 0, "ea_close_n": 0,
                    "ea_net": 0.0, "manual_net": 0.0, "commission": 0.0, "swap": 0.0}

    txt = D.format_report(ctx)

    assert "自动交易 关" in txt
    assert "窗口内无成交" in txt


def test_format_report_says_no_data_instead_of_fake_zero_performance():
    """空窗口不能被排成"已实现 0.00"——那会被读成"今天没赚没亏"，而事实是压根没数据。"""
    ctx = _ctx()
    ctx["signals"] = {"n": 0, "flips": 0, "missing_bars": 0, "expected_bars": 0,
                      "dir_counts": {"long": 0, "short": 0, "flat": 0},
                      "z_min": None, "z_max": None,
                      "first_bar": None, "last_bar": None}
    ctx["deals"] = {"ea": [], "manual": [], "ea_open_n": 0, "ea_close_n": 0,
                    "ea_net": 0.0, "manual_net": 0.0, "commission": 0.0, "swap": 0.0}
    ctx["positions"] = {"n": 0, "ea_n": 0, "side": "flat", "floating": 0.0,
                        "naked": False, "rows": []}
    ctx["window"] = {"from": dt.datetime(2026, 11, 7, 11, 0),
                     "to": dt.datetime(2026, 11, 8, 11, 0)}

    txt = D.format_report(ctx)

    assert "无信号" in txt
    assert "窗口内无成交" in txt
    assert "空仓" in txt


# ── ⑦ last_dir（C2 接线需要的最后一根方向）─────────────────────
def test_summarize_signals_last_dir_takes_final_bar():
    base = dt.datetime(2026, 11, 6, 0, 0)
    bars = [base + dt.timedelta(minutes=30 * i) for i in range(4)]

    # 首末不同向：dirs[-1] 与 dirs[0] 必须能区分（C2 审查 Q1：旧夹具 [1,-1,-1,1] 首末同值，
    # 把实现改成取第一根也不会红）
    s = D.summarize_signals(_sigs(bars, [1, -1, -1, -1]))

    assert s["last_dir"] == -1, "多根取最后一根的方向（首末不同向，取错第一根会变红）"
    assert isinstance(s["last_dir"], int)


def test_summarize_signals_last_dir_single_bar():
    s = D.summarize_signals(_sigs([dt.datetime(2026, 11, 6, 0, 0)], [-1]))

    assert s["last_dir"] == -1


def test_summarize_signals_last_dir_empty_is_none():
    s = D.summarize_signals([])

    assert s["last_dir"] is None, "空仓分支也要有这个键，否则 C2 会 KeyError"
