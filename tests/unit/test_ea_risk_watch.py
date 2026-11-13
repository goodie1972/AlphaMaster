# tests/unit/test_ea_risk_watch.py
"""ea_risk_watch 判定层：吃 plain dict / 标量，吐中文告警；不碰 MetaTrader5、不读文件。

被每 30 分钟的只读定时任务消费，所以两条语义逐条钉死：
① 无命中必须静默 —— 空仓 / 终端健康 / SL 在容差内 / 持仓未超时 ⇒ []，且空告警永不 emit；
② 核不了 ≠ 核过了 —— 期望 SL 缺失、开仓时刻缺失都要告警，放行等于把手工改票洗白。
夹具一律经 ea_daily_digest.open_positions 生成，与采集层交付的真实形状一致。
"""
from __future__ import annotations

import datetime as dt

import pytest

import ea_daily_digest as D
import ea_risk_watch as W

NOW = dt.datetime(2026, 11, 6, 12, 0)
HEALTH_OK = {"terminal_running": True, "trade_allowed": True}
FP = "0123456789abcdef"
OTHER_FP = "fedcba9876543210"


def _row(ticket, *, typ=0, sl=0.0, tp=0.0, time_open=NOW, magic=D.EA_MAGIC,
         volume=0.01):
    return {"magic": magic, "symbol": "XAUUSD", "type": typ, "volume": volume,
            "price_open": 4000.0, "price_current": 4001.0, "profit": 7.33,
            "sl": sl, "tp": tp, "ticket": ticket, "time_open": time_open}


def _pos(*rows):
    return D.open_positions(list(rows))


# ── ① 该平没平 ────────────────────────────────────────────────
def test_should_flat_shouts_when_signal_zeroed_but_ea_still_holds():
    pos = _pos(_row(11), _row(12), _row(13))

    alerts = W.check_should_flat(0, pos)

    assert len(alerts) == 1
    assert "信号已归零" in alerts[0]
    assert "3" in alerts[0], "文案要带仓数，否则读的人不知道有几笔裸着"


def test_should_flat_shouts_when_signal_flips_against_long_book():
    alerts = W.check_should_flat(-1, _pos(_row(11, typ=0)))

    assert len(alerts) == 1 and "信号要求反向" in alerts[0]


def test_should_flat_shouts_when_signal_flips_against_short_book():
    alerts = W.check_should_flat(1, _pos(_row(11, typ=1)))

    assert len(alerts) == 1 and "信号要求反向" in alerts[0]


@pytest.mark.parametrize("dir_now,typ", [(1, 0), (-1, 1)])
def test_should_flat_silent_when_signal_matches_book(dir_now, typ):
    assert W.check_should_flat(dir_now, _pos(_row(11, typ=typ))) == []


def test_should_flat_silent_when_no_ea_position():
    assert W.check_should_flat(0, _pos()) == []


def test_should_flat_ignores_manual_positions():
    """magic=0 的手工单不是 EA 的仓，不能拿它触发「该平没平」。"""
    assert W.check_should_flat(0, _pos(_row(99, magic=0))) == []


@pytest.mark.parametrize("bad", [2, -2, 99])
def test_should_flat_rejects_out_of_range_dir(bad):
    with pytest.raises(ValueError):
        W.check_should_flat(bad, _pos(_row(11)))


def test_check_should_flat_illegal_dir_raises_even_when_flat():
    """校验必须在空仓早退之前，否则坏信号源会被静默当成 0。"""
    with pytest.raises(ValueError):
        W.check_should_flat(2, _pos())


@pytest.mark.parametrize("dir_now", [1, -1])
def test_should_flat_net_zero_hedge_gets_own_wording(dir_now):
    """净敞口是零不是反向：文案不许误导读的人去查反向持仓。"""
    pos = _pos(_row(11, typ=0), _row(12, typ=1))
    assert pos["side"] == "flat" and pos["ea_n"] == 2

    alerts = W.check_should_flat(dir_now, pos)

    assert len(alerts) == 1
    assert "多空并存净平" in alerts[0]
    assert "信号要求反向" not in alerts[0]


# ── ② 有仓但兜底没了 ──────────────────────────────────────────
def test_terminal_shouts_when_disconnected_with_open_ea_book():
    health = {"terminal_running": False, "trade_allowed": True}

    alerts = W.check_terminal(_pos(_row(11)), health)

    assert len(alerts) == 1 and "终端未连接" in alerts[0]


def test_terminal_shouts_when_autotrading_off_with_open_ea_book():
    health = {"terminal_running": True, "trade_allowed": False}

    alerts = W.check_terminal(_pos(_row(11)), health)

    assert len(alerts) == 1 and "AutoTrading" in alerts[0]


def test_terminal_reports_both_failures_independently():
    health = {"terminal_running": False, "trade_allowed": False}

    alerts = W.check_terminal(_pos(_row(11)), health)

    assert len(alerts) == 2, "两个兜底各自失效，压成一条会掩盖第二个"
    assert "终端未连接" in alerts[0] and "AutoTrading" in alerts[1]


def test_terminal_silent_when_healthy_and_holding():
    assert W.check_terminal(_pos(_row(11)), HEALTH_OK) == []


def test_terminal_silent_when_flat_even_if_terminal_down():
    """空仓时终端关不关无所谓，报了就是每 30 分钟一次的噪音。"""
    health = {"terminal_running": False, "trade_allowed": False}

    assert W.check_terminal(_pos(), health) == []


def test_terminal_ignores_manual_positions():
    """只有手工单时 EA 是空仓，拿 pos['n'] 当 ea_n 会每 30 分钟误报一次。"""
    health = {"terminal_running": False, "trade_allowed": False}

    assert W.check_terminal(_pos(_row(99, magic=0)), health) == []


def test_check_terminal_missing_health_keys_raise():
    """有 EA 仓时体检数据整个拿不到 ⇒ KeyError，不许静默当成健康。"""
    with pytest.raises(KeyError):
        W.check_terminal(_pos(_row(11)), {})


def test_check_terminal_half_health_raises():
    """不许改成 .get(k, True)：那会把体检失败洗成健康。"""
    with pytest.raises(KeyError):
        W.check_terminal(_pos(_row(11)), {"terminal_running": True})
    with pytest.raises(KeyError):
        W.check_terminal(_pos(_row(11)), {"trade_allowed": True})


# ── ③ SL 没挂 / 被手工挪远 ────────────────────────────────────
def test_stop_attached_shouts_when_sl_missing():
    alerts = W.check_stop_attached(_pos(_row(11, sl=0.0)), {11: 3990.0}, tol=0.5)

    assert len(alerts) == 1
    assert "SL 未挂" in alerts[0] and "11" in alerts[0]


def test_stop_attached_shouts_when_expected_sl_unknown():
    alerts = W.check_stop_attached(_pos(_row(11, sl=3990.0)), {}, tol=0.5)

    assert len(alerts) == 1
    assert "期望 SL 未知" in alerts[0] and "无法核对" in alerts[0] and "11" in alerts[0]


def test_stop_attached_shouts_when_sl_moved_beyond_tol():
    alerts = W.check_stop_attached(_pos(_row(11, sl=3900.0)), {11: 3990.0}, tol=0.5)

    assert len(alerts) == 1
    assert "SL 被挪动" in alerts[0]
    assert "11" in alerts[0]
    assert "3900.000" in alerts[0] and "3990.000" in alerts[0], "实际值与期望值都要印出来"


def test_stop_attached_silent_at_exactly_tol():
    assert W.check_stop_attached(_pos(_row(11, sl=3990.5)), {11: 3990.0}, tol=0.5) == []


def test_stop_attached_silent_for_small_drift_inside_tol():
    assert W.check_stop_attached(_pos(_row(11, sl=3990.2)), {11: 3990.0}, tol=0.5) == []


def test_stop_attached_silent_when_sl_matches_expected():
    assert W.check_stop_attached(_pos(_row(11, sl=3990.0)), {11: 3990.0}, tol=0.5) == []


def test_stop_attached_silent_when_no_ea_position():
    assert W.check_stop_attached(_pos(), {}, tol=0.5) == []


def test_stop_attached_ignores_manual_positions():
    assert W.check_stop_attached(_pos(_row(99, magic=0, sl=0.0)), {}, tol=0.5) == []


def test_stop_attached_checks_every_row_not_just_the_first():
    pos = _pos(_row(11, sl=3990.0), _row(12, sl=0.0), _row(13, sl=3800.0))
    expected = {11: 3990.0, 12: 3990.0, 13: 3990.0}

    alerts = W.check_stop_attached(pos, expected, tol=0.5)

    assert len(alerts) == 2
    assert "12" in alerts[0] and "SL 未挂" in alerts[0]
    assert "13" in alerts[1] and "SL 被挪动" in alerts[1]


# ── ④ 持仓太久 ────────────────────────────────────────────────
def test_hold_age_shouts_past_max_bars_with_one_decimal():
    # 2026-11-04 11:45 → 11-06 12:00 = 2895 分钟 / 30 = 96.5 根
    pos = _pos(_row(11, time_open=dt.datetime(2026, 11, 4, 11, 45)))

    alerts = W.check_hold_age(pos, NOW, max_bars=48)

    assert len(alerts) == 1
    assert "11" in alerts[0] and "96.5" in alerts[0] and "48" in alerts[0]


def test_hold_age_silent_at_exactly_max_bars():
    # 11-05 12:00 → 11-06 12:00 = 1440 分钟 / 30 = 48.0 根，恰好压在上限上
    pos = _pos(_row(11, time_open=dt.datetime(2026, 11, 5, 12, 0)))

    assert W.check_hold_age(pos, NOW, max_bars=48) == []


def test_hold_age_silent_for_young_position():
    pos = _pos(_row(11, time_open=dt.datetime(2026, 11, 6, 11, 30)))

    assert W.check_hold_age(pos, NOW, max_bars=48) == []


def test_hold_age_shouts_when_open_time_missing():
    alerts = W.check_hold_age(_pos(_row(11, time_open=None)), NOW, max_bars=48)

    assert len(alerts) == 1
    assert "开仓时刻缺失" in alerts[0] and "无法核算持仓时长" in alerts[0]
    assert "11" in alerts[0]


def test_hold_age_shouts_when_open_time_key_absent():
    """键整个丢了和值为 None 一样核不了，不能靠 KeyError 把整轮告警打断。"""
    row = _row(11)
    del row["time_open"]

    alerts = W.check_hold_age(_pos(row), NOW, max_bars=48)

    assert len(alerts) == 1 and "开仓时刻缺失" in alerts[0] and "11" in alerts[0]


def test_hold_age_silent_when_no_ea_position():
    assert W.check_hold_age(_pos(), NOW, max_bars=0) == []


def test_hold_age_honours_tf_minutes():
    pos = _pos(_row(11, time_open=dt.datetime(2026, 11, 4, 12, 0)))   # 整 48 小时

    assert len(W.check_hold_age(pos, NOW, max_bars=48, tf_minutes=30)) == 1   # 96 根
    assert W.check_hold_age(pos, NOW, max_bars=48, tf_minutes=60) == []       # 48 根，恰好等于上限


def test_hold_age_names_only_the_stale_ticket():
    pos = _pos(_row(11, time_open=NOW),
               _row(12, time_open=dt.datetime(2026, 11, 4, 11, 45)))

    alerts = W.check_hold_age(pos, NOW, max_bars=48)

    assert len(alerts) == 1 and "12" in alerts[0] and "96.5" in alerts[0]


# ── ⑤ 聚合 ────────────────────────────────────────────────────
def test_collect_alerts_concatenates_four_checks_in_order():
    pos = _pos(_row(11, sl=0.0, time_open=dt.datetime(2026, 11, 4, 11, 45)))
    health = {"terminal_running": False, "trade_allowed": False}

    alerts = W.collect_alerts(0, pos, health, {}, NOW, max_bars=48, tol=0.5)

    assert len(alerts) == 5, "1 归零 + 2 兜底 + 1 SL + 1 时长，聚合不许吞也不许加"
    assert "信号已归零" in alerts[0]
    assert "终端未连接" in alerts[1]
    assert "AutoTrading" in alerts[2]
    assert "SL 未挂" in alerts[3]
    assert "96.5" in alerts[4]


def test_collect_alerts_silent_on_clean_book():
    pos = _pos(_row(11, sl=3990.0, time_open=dt.datetime(2026, 11, 6, 11, 30)))

    assert W.collect_alerts(1, pos, HEALTH_OK, {11: 3990.0}, NOW,
                            max_bars=48, tol=0.5) == []


def test_collect_alerts_forwards_tf_minutes():
    pos = _pos(_row(11, sl=3990.0, time_open=dt.datetime(2026, 11, 4, 12, 0)))
    args = (1, pos, HEALTH_OK, {11: 3990.0}, NOW)

    assert len(W.collect_alerts(*args, max_bars=48, tol=0.5)) == 1
    assert W.collect_alerts(*args, max_bars=48, tol=0.5, tf_minutes=60) == []


def test_collect_alerts_forwards_tol():
    """同一份输入只改 tol：聚合层把 tol 写死成常量必须变红。"""
    pos = _pos(_row(11, sl=3990.1, time_open=dt.datetime(2026, 11, 6, 11, 30)))
    args = (1, pos, HEALTH_OK, {11: 3990.0}, NOW)

    loose = W.collect_alerts(*args, max_bars=48, tol=0.5)
    tight = W.collect_alerts(*args, max_bars=48, tol=0.05)

    assert not any("SL 被挪动" in a for a in loose)
    assert any("SL 被挪动" in a for a in tight)


def test_collect_alerts_forwards_max_bars():
    """同一份输入只改 max_bars：聚合层把 max_bars 写死成常量必须变红。"""
    # 11-06 00:00 → 12:00 = 720 分钟 / 30 = 24 根
    pos = _pos(_row(11, sl=3990.0, time_open=dt.datetime(2026, 11, 6, 0, 0)))
    args = (1, pos, HEALTH_OK, {11: 3990.0}, NOW)

    loose = W.collect_alerts(*args, max_bars=48, tol=0.5)
    tight = W.collect_alerts(*args, max_bars=12, tol=0.5)

    assert not any("持仓过久" in a for a in loose)
    assert any("持仓过久" in a for a in tight)


# ── ⑤b 多空并存（F3）─────────────────────────────────────────
def test_mixed_hedge_shouts_on_equal_lots():
    pos = _pos(_row(11, typ=0, volume=0.01), _row(12, typ=1, volume=0.01))
    assert pos["side"] == "flat"

    alerts = W.check_mixed_hedge(pos)

    assert len(alerts) == 1
    assert "多空并存" in alerts[0] and "2" in alerts[0] and "清仓" in alerts[0]


def test_mixed_hedge_shouts_on_unequal_lots_even_if_net_long():
    """EA 的 CurrentDirection 看 hasBuy&&hasSell，不看净手数：净 long 也必须响。"""
    pos = _pos(_row(11, typ=0, volume=0.02), _row(12, typ=1, volume=0.01))
    assert pos["side"] == "long"

    alerts = W.check_mixed_hedge(pos)

    assert len(alerts) == 1
    assert "多空并存" in alerts[0] and "2" in alerts[0] and "清仓" in alerts[0]


def test_mixed_hedge_silent_on_pure_long():
    assert W.check_mixed_hedge(_pos(_row(11, typ=0), _row(12, typ=0))) == []


def test_mixed_hedge_silent_on_pure_short():
    assert W.check_mixed_hedge(_pos(_row(11, typ=1), _row(12, typ=1))) == []


def test_mixed_hedge_silent_when_no_position():
    assert W.check_mixed_hedge(_pos()) == []


def test_collect_alerts_appends_mixed_hedge_last():
    """多空并存接在第 5 位：既有 1→2→3→4 顺序不动，新告警排在时长之后。"""
    pos = _pos(_row(11, typ=0, sl=0.0, time_open=dt.datetime(2026, 11, 4, 11, 45)),
               _row(12, typ=1, sl=0.0, time_open=dt.datetime(2026, 11, 4, 11, 45)))
    health = {"terminal_running": False, "trade_allowed": False}

    alerts = W.collect_alerts(0, pos, health, {}, NOW, max_bars=48, tol=0.5)

    assert len(alerts) == 8, "1 归零 + 2 兜底 + 2 SL + 2 时长 + 1 多空并存"
    assert "信号已归零" in alerts[0]
    assert "终端未连接" in alerts[1]
    assert "AutoTrading" in alerts[2]
    assert "SL 未挂" in alerts[3] and "SL 未挂" in alerts[4]
    assert "96.5" in alerts[5] and "96.5" in alerts[6]
    assert "多空并存" in alerts[7] and "清仓" in alerts[7]


# ── ⑥ 指纹去重 ────────────────────────────────────────────────
def test_fingerprint_ignores_alert_order():
    assert W.fingerprint(["信号已归零", "SL 未挂"]) == W.fingerprint(["SL 未挂", "信号已归零"])


def test_fingerprint_changes_when_content_changes():
    assert W.fingerprint(["信号已归零"]) != W.fingerprint(["SL 未挂"])


def test_fingerprint_changes_when_count_changes():
    assert W.fingerprint(["a"]) != W.fingerprint(["a", "a"]), "去重成 set 会把两笔仓洗成一笔"


def test_fingerprint_of_empty_list_is_stable_and_distinct():
    empty = W.fingerprint([])

    assert empty == W.fingerprint([])
    assert isinstance(empty, str) and len(empty) == 16
    assert empty != W.fingerprint(["信号已归零"])


def test_fingerprint_does_not_reorder_caller_list():
    alerts = ["b", "a"]

    W.fingerprint(alerts)

    assert alerts == ["b", "a"], "就地 sort 会破坏 collect_alerts 的顺序契约"


# ── ⑦ 冷却与静默 ──────────────────────────────────────────────
def test_should_emit_true_on_first_run():
    assert W.should_emit(None, None, 1000.0, FP, 1800.0) is True


def test_should_emit_true_when_prev_timestamp_missing():
    assert W.should_emit(FP, None, 1000.0, FP, 1800.0) is True


def test_should_emit_true_when_prev_fingerprint_missing():
    assert W.should_emit(None, 0.0, 1000.0, FP, 1800.0) is True


def test_should_emit_false_for_same_fingerprint_inside_cooldown():
    assert W.should_emit(FP, 1_000_000.0, 1_001_799.0, FP, 1800.0) is False


def test_should_emit_true_for_same_fingerprint_at_cooldown_boundary():
    assert W.should_emit(FP, 1_000_000.0, 1_001_800.0, FP, 1800.0) is True


def test_should_emit_true_for_same_fingerprint_after_cooldown():
    assert W.should_emit(FP, 1_000_000.0, 1_003_600.0, FP, 1800.0) is True


def test_should_emit_true_immediately_when_fingerprint_changes():
    assert W.should_emit(FP, 1_000_000.0, 1_000_001.0, OTHER_FP, 1800.0) is True


def test_should_emit_false_for_empty_alerts_on_first_run():
    assert W.should_emit(None, None, 1000.0, W.fingerprint([]), 1800.0) is False


def test_should_emit_false_for_empty_alerts_even_after_cooldown():
    empty = W.fingerprint([])

    assert W.should_emit(empty, 0.0, 1e9, empty, 1800.0) is False, "无命中必须静默，冷却到期也不能说话"


def test_should_emit_true_when_alerts_appear_after_a_clean_run():
    assert W.should_emit(W.fingerprint([]), 1_000_000.0, 1_000_001.0, FP, 1800.0) is True


# ── ⑧ 多空并存合成账本（钉死 C1 复评存活变异 N3/N5/N8）──────────
def _mixed_book():
    """EA 两多一空（净平）+ 一笔手工单 ⇒ pos['n']=4 而 len(rows)=3。
    前两行同向、第三行反向：对冲检测必须看全部 rows，不能只看前两行。"""
    return _pos(_row(11, typ=0, volume=0.01),
                _row(12, typ=0, volume=0.01),
                _row(13, typ=1, volume=0.02),
                _row(99, typ=0, volume=0.05, magic=0))


def test_mixed_hedge_count_uses_ea_rows_not_account_total():
    pos = _mixed_book()
    assert pos["n"] == 4 and len(pos["rows"]) == 3 and pos["ea_n"] == 3

    alerts = W.check_mixed_hedge(pos)

    assert len(alerts) == 1 and "多空并存" in alerts[0]
    assert "3" in alerts[0], "根数必须取 len(rows)=3"
    assert "4" not in alerts[0], "不许用 pos['n']=4（含手工单）—— N3"


def test_should_flat_net_flat_count_uses_ea_rows_not_account_total():
    pos = _mixed_book()
    assert pos["side"] == "flat"

    alerts = W.check_should_flat(1, pos)

    assert len(alerts) == 1 and "多空并存净平" in alerts[0]
    assert "3" in alerts[0], "净平文案根数取 len(rows)=3"
    assert "4" not in alerts[0], "不许用 pos['n']=4 —— N5"


def test_should_flat_zeroed_count_uses_ea_rows_not_account_total():
    pos = _mixed_book()

    alerts = W.check_should_flat(0, pos)

    assert len(alerts) == 1 and "信号已归零" in alerts[0]
    assert "3" in alerts[0] and "4" not in alerts[0]


def test_mixed_hedge_sees_opposite_third_row_when_first_two_agree():
    pos = _mixed_book()
    kinds = [p["type"] for p in pos["rows"]]
    assert kinds[0] == kinds[1] == 0 and kinds[2] == 1, "前两行同向、第三行反向"

    alerts = W.check_mixed_hedge(pos)

    assert len(alerts) == 1, "只看前两行会漏掉第三行反向 —— N8"
    assert "多空并存" in alerts[0] and "清仓" in alerts[0]


# ── ⑨ parse_sl_attach ─────────────────────────────────────────
_SL_LINE = ("[AlphaMaster] sl_attach ticket=161549988 side=1 entry=4066.930 "
            "atr=12.34567 k=4.0 sl=4017.54732")


def test_parse_sl_attach_single_line():
    assert W.parse_sl_attach(_SL_LINE) == {161549988: 4017.54732}


def test_parse_sl_attach_same_ticket_last_wins():
    text = ("[AlphaMaster] sl_attach ticket=11 side=1 entry=4000.0 atr=10.0 k=4.0 sl=3990.0\n"
            "[AlphaMaster] sl_attach ticket=11 side=1 entry=4000.0 atr=10.0 k=4.0 sl=3985.5")

    assert W.parse_sl_attach(text) == {11: 3985.5}, "init 校正过 ⇒ 取最后一次"


def test_parse_sl_attach_two_different_tickets():
    text = ("[AlphaMaster] sl_attach ticket=11 side=1 entry=4000.0 atr=10.0 k=4.0 sl=3990.0\n"
            "[AlphaMaster] sl_attach ticket=12 side=0 entry=4000.0 atr=10.0 k=4.0 sl=4010.0")

    assert W.parse_sl_attach(text) == {11: 3990.0, 12: 4010.0}


def test_parse_sl_attach_ignores_signal_and_other_ea_lines():
    text = ("[AlphaMaster] bar=2026.11.06 05:00 closedBars=2000 z=-0.08 tanh=-0.08 dir=-1\n"
            + _SL_LINE + "\n"
            "SomeOtherEA (EURUSD,H1) whatever ticket=999 sl=123.0\n"
            "[AlphaMaster] 无足够 bar，跳过")

    assert W.parse_sl_attach(text) == {161549988: 4017.54732}


def test_parse_sl_attach_empty_text():
    assert W.parse_sl_attach("") == {}


def test_parse_sl_attach_no_match():
    assert W.parse_sl_attach("random log line\nnothing to see here") == {}


def test_parse_sl_attach_failed_line_still_parses_sl():
    # A4 的 sl_attach FAILED 行（PositionModify 失败）仍携带 sl=，让 C2 知道「期望该 SL 却没挂上」
    text = ("[AlphaMaster] sl_attach FAILED ticket=161631408 side=1 entry=4066.930 "
            "atr=12.34567 k=20.0 sl=4017.54732 retcode=10016")
    assert W.parse_sl_attach(text) == {161631408: 4017.54732}, "FAILED 行也要解析出期望 SL"


def test_parse_sl_attach_malformed_missing_sl_not_collected():
    text = "[AlphaMaster] sl_attach ticket=11 side=1 entry=4000.0 atr=10.0 k=4.0"

    assert W.parse_sl_attach(text) == {}, "缺 sl= 不许崩也不许收进字典"


def test_parse_sl_attach_malformed_missing_ticket_not_collected():
    text = "[AlphaMaster] sl_attach side=1 entry=4000.0 atr=10.0 k=4.0 sl=3990.0"

    assert W.parse_sl_attach(text) == {}


# ── ⑩ check_signal_stale ──────────────────────────────────────
def test_signal_stale_silent_one_bar_behind():
    bar = NOW - dt.timedelta(minutes=30)

    assert W.check_signal_stale(bar, NOW, max_stale_bars=4) == []


def test_signal_stale_shouts_over_threshold_with_one_decimal():
    bar = NOW - dt.timedelta(minutes=30 * 5)

    alerts = W.check_signal_stale(bar, NOW, max_stale_bars=4)

    assert len(alerts) == 1
    assert "EA 信号停更" in alerts[0] and "5.0" in alerts[0]


def test_signal_stale_silent_at_exactly_threshold():
    bar = NOW - dt.timedelta(minutes=30 * 4)

    assert W.check_signal_stale(bar, NOW, max_stale_bars=4) == [], "严格 >，等于不报"


def test_signal_stale_silent_on_weekend_when_both_old():
    old_tick = dt.datetime(2026, 11, 1, 12, 0)
    old_bar = old_tick - dt.timedelta(minutes=30)

    assert W.check_signal_stale(old_bar, old_tick, max_stale_bars=4) == [], \
        "周末两者都旧但差值小 ⇒ 不误报"


def test_signal_stale_shouts_when_signal_bar_none():
    alerts = W.check_signal_stale(None, NOW, max_stale_bars=4)

    assert len(alerts) == 1 and "无法核算停更" in alerts[0]


def test_signal_stale_shouts_when_tick_none():
    alerts = W.check_signal_stale(NOW - dt.timedelta(minutes=30), None, max_stale_bars=4)

    assert len(alerts) == 1 and "无法核算停更" in alerts[0]


def test_signal_stale_shouts_on_negative_diff_clock_anomaly():
    bar = NOW + dt.timedelta(minutes=30)

    alerts = W.check_signal_stale(bar, NOW, max_stale_bars=4)

    assert len(alerts) == 1
    assert "时钟口径异常" in alerts[0]
    assert "EA 信号停更" not in alerts[0], "负差不是停更，是时钟异常"


def test_signal_stale_honours_tf_minutes():
    bar = NOW - dt.timedelta(minutes=60 * 5)

    assert len(W.check_signal_stale(bar, NOW, max_stale_bars=4, tf_minutes=30)) == 1
    assert W.check_signal_stale(bar, NOW, max_stale_bars=4, tf_minutes=120) == []


# ── ⑪ alerts_from_ctx ─────────────────────────────────────────
def _watch_ctx(**kw):
    base = {
        "account": {"login": 222139413, "server": "Dukascopy-demo-mt5-1",
                    "trade_mode": "DEMO"},
        "window": {"from": dt.datetime(2026, 11, 5, 12, 0), "to": NOW},
        "signals": {"last_dir": 1, "last_bar": NOW - dt.timedelta(minutes=30), "n": 1},
        "positions": D.open_positions([]),
        "health": {"terminal_running": True, "trade_allowed": True,
                   "last_tick_local": NOW, "data_path": "Z:/fake"},
    }
    base.update(kw)
    return base


def test_alerts_from_ctx_clean_is_empty():
    ctx = _watch_ctx()

    assert W.alerts_from_ctx(ctx, max_bars=192, tol=0.05, max_stale_bars=4,
                             now_local=NOW) == []


def test_alerts_from_ctx_no_signal_still_checks_sl():
    pos = D.open_positions([_row(11, sl=0.0)])
    ctx = _watch_ctx(positions=pos,
                     signals={"last_dir": None, "last_bar": NOW - dt.timedelta(minutes=30),
                              "n": 0})

    alerts = W.alerts_from_ctx(ctx, max_bars=192, tol=0.05, max_stale_bars=4, now_local=NOW)

    assert any("没有 EA 信号行" in a for a in alerts)
    assert any("SL 未挂" in a for a in alerts), "信号缺失不能把 SL 检查也静默掉"


def test_alerts_from_ctx_bad_dir_reports_failure_and_continues():
    ctx = _watch_ctx(signals={"last_dir": 99, "last_bar": NOW - dt.timedelta(minutes=30 * 10),
                              "n": 1})

    alerts = W.alerts_from_ctx(ctx, max_bars=192, tol=0.05, max_stale_bars=4, now_local=NOW)

    assert any("告警判定失败" in a for a in alerts)
    assert any("EA 信号停更" in a for a in alerts), "判定失败后仍要跑停更检查（F4）"


def test_alerts_from_ctx_missing_health_keys_no_crash():
    pos = D.open_positions([_row(11, sl=3990.0)])
    ctx = _watch_ctx(positions=pos, health={"last_tick_local": NOW})

    alerts = W.alerts_from_ctx(ctx, max_bars=192, tol=0.05, max_stale_bars=4, now_local=NOW)

    assert any("告警判定失败" in a for a in alerts)


def test_alerts_from_ctx_health_failure_does_not_swallow_sl_alert():
    # Q7：check_terminal 缺 health 键失败，不能把独立于 health 的「SL 未挂」也一起吞掉
    pos = D.open_positions([_row(11, sl=0.0)])
    ctx = _watch_ctx(positions=pos, health={"last_tick_local": NOW})

    alerts = W.alerts_from_ctx(ctx, max_bars=192, tol=0.05, max_stale_bars=4, now_local=NOW)

    assert any("告警判定失败" in a for a in alerts), "terminal 检查失败要转成一条告警"
    assert any("SL 未挂" in a for a in alerts), "terminal 失败不能吞掉 SL 未挂（本 feature 最重要）"


def test_alerts_from_ctx_sl_attach_text_detects_moved_sl():
    pos = D.open_positions([_row(11, sl=3900.0)])
    ctx = _watch_ctx(positions=pos)
    log = "[AlphaMaster] sl_attach ticket=11 side=1 entry=4000.0 atr=10.0 k=4.0 sl=3990.0"

    alerts = W.alerts_from_ctx(ctx, max_bars=192, tol=0.05, max_stale_bars=4,
                               now_local=NOW, ea_log_text=log)

    assert any("SL 被挪动" in a for a in alerts)


def test_alerts_from_ctx_stale_over_threshold():
    ctx = _watch_ctx(signals={"last_dir": 1, "last_bar": NOW - dt.timedelta(minutes=30 * 10),
                              "n": 1})

    alerts = W.alerts_from_ctx(ctx, max_bars=192, tol=0.05, max_stale_bars=4, now_local=NOW)

    assert any("EA 信号停更" in a for a in alerts)


# ── ⑫ load_state / save_state ─────────────────────────────────
def test_state_round_trip(tmp_path):
    p = str(tmp_path / "state.json")

    W.save_state(p, "abcd1234", 1000.0)

    assert W.load_state(p) == ("abcd1234", 1000.0)


def test_load_state_missing_file(tmp_path):
    assert W.load_state(str(tmp_path / "nope.json")) == (None, None)


def test_load_state_bad_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")

    assert W.load_state(str(p)) == (None, None), "半截文件当成第一次"


def test_save_state_atomic_leaves_no_tmp(tmp_path):
    p = str(tmp_path / "state.json")

    W.save_state(p, "fp", 1.0)

    assert (tmp_path / "state.json").exists()
    assert not (tmp_path / "state.json.tmp").exists()


def test_save_state_creates_parent_dir(tmp_path):
    p = str(tmp_path / "nested" / "deep" / "state.json")

    W.save_state(p, "fp", 2.0)

    assert W.load_state(p) == ("fp", 2.0)


# ── ⑬ run_watch ───────────────────────────────────────────────
def test_run_watch_dry_run_prints_alerts_and_writes_no_state(tmp_path, monkeypatch, capsys):
    ctx = _watch_ctx(positions=D.open_positions([_row(11, sl=0.0)]))
    monkeypatch.setattr(W.ea_daily_digest, "collect", lambda hours: ctx)
    state = str(tmp_path / "state.json")

    rc = W.run_watch(state_path=state, max_bars=10**6, dry_run=True)

    out = capsys.readouterr().out
    assert rc == 0
    assert "EA 持仓告警" in out and "SL 未挂" in out
    assert "只读告警，不会平仓" in out
    assert not (tmp_path / "state.json").exists(), "dry_run 不写状态"


def test_run_watch_dry_run_clean_prints_nothing(tmp_path, monkeypatch, capsys):
    ctx = _watch_ctx()
    monkeypatch.setattr(W.ea_daily_digest, "collect", lambda hours: ctx)

    rc = W.run_watch(state_path=str(tmp_path / "s.json"), max_bars=10**6, dry_run=True)

    assert rc == 0
    assert capsys.readouterr().out == ""


def test_run_watch_collect_failure_returns_1(monkeypatch, capsys):
    def boom(hours):
        raise SystemExit("mt5.initialize 失败")
    monkeypatch.setattr(W.ea_daily_digest, "collect", boom)

    rc = W.run_watch(dry_run=True)

    assert rc == 1
    assert "告警采集失败" in capsys.readouterr().out


def test_run_watch_first_run_emits_and_saves_state(tmp_path, monkeypatch, capsys):
    ctx = _watch_ctx(positions=D.open_positions([_row(11, sl=0.0)]))
    monkeypatch.setattr(W.ea_daily_digest, "collect", lambda hours: ctx)
    state = str(tmp_path / "state.json")

    rc = W.run_watch(state_path=state, max_bars=10**6)

    assert rc == 0
    assert "EA 持仓告警" in capsys.readouterr().out
    fp, ts = W.load_state(state)
    assert fp is not None and ts is not None


def test_run_watch_same_alerts_within_cooldown_silent(tmp_path, monkeypatch, capsys):
    ctx = _watch_ctx(positions=D.open_positions([_row(11, sl=0.0)]))
    monkeypatch.setattr(W.ea_daily_digest, "collect", lambda hours: ctx)
    state = str(tmp_path / "state.json")
    W.run_watch(state_path=state, max_bars=10**6)
    capsys.readouterr()

    rc = W.run_watch(state_path=state, max_bars=10**6, cooldown_s=21600.0)

    assert rc == 0
    assert capsys.readouterr().out == "", "冷却内同指纹必须静默"


def test_run_watch_fingerprint_change_emits_again(tmp_path, monkeypatch, capsys):
    ctx = _watch_ctx(positions=D.open_positions([_row(11, sl=0.0)]))
    monkeypatch.setattr(W.ea_daily_digest, "collect", lambda hours: ctx)
    state = str(tmp_path / "state.json")
    W.run_watch(state_path=state, max_bars=10**6)
    capsys.readouterr()
    ctx2 = _watch_ctx(positions=D.open_positions([_row(12, sl=0.0)]))
    monkeypatch.setattr(W.ea_daily_digest, "collect", lambda hours: ctx2)

    rc = W.run_watch(state_path=state, max_bars=10**6)

    assert rc == 0
    assert "EA 持仓告警" in capsys.readouterr().out, "情况一变立刻报"


# ── ⑭ main / CLI ──────────────────────────────────────────────
def test_main_dry_run_clean_returns_0(tmp_path, monkeypatch, capsys):
    ctx = _watch_ctx()
    monkeypatch.setattr(W.ea_daily_digest, "collect", lambda hours: ctx)

    rc = W.main(["--dry-run", "--state", str(tmp_path / "s.json")])

    assert rc == 0
    assert capsys.readouterr().out == ""


def test_main_collect_failure_returns_1(monkeypatch, capsys):
    def boom(hours):
        raise SystemExit("no terminal")
    monkeypatch.setattr(W.ea_daily_digest, "collect", boom)

    rc = W.main(["--dry-run"])

    assert rc == 1
    assert "告警采集失败" in capsys.readouterr().out


def test_main_passes_args_to_run_watch(monkeypatch):
    captured = {}
    monkeypatch.setattr(W, "run_watch", lambda **kw: captured.update(kw) or 0)

    rc = W.main(["--hours", "48", "--max-bars", "10", "--tol", "0.1",
                 "--max-stale-bars", "2", "--cooldown-s", "60",
                 "--state", "X", "--dry-run"])

    assert rc == 0
    assert captured == {"hours": 48, "max_bars": 10, "tol": 0.1, "max_stale_bars": 2,
                        "cooldown_s": 60.0, "state_path": "X", "dry_run": True}