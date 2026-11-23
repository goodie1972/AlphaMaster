//+------------------------------------------------------------------+
//| US_Range_Fade_EA.mq5 — H4 区间 M5 假突破回归（美盘/欧盘可切）        |
//| 参照棒 = 每日服务器 InpRefHour:00 起 H4；实体>=50%区间且居中(偏差<=25%)|
//| 上轨=H 下轨=L；M5 监控窗 = 参照棒开盘 +4h ~ +8h（服务器钟）          |
//| 收盘突破上轨→标记跟踪最高价；收盘回归上轨→做空，SL=最高价，TP=2R    |
//| 下轨镜像做多。每轨每日一次；门挡掉的回归同样作废该轨；                |
//| 最多1笔持仓；窗口末强平；双触 SL/TP 由 broker tick 顺序决定          |
//| 逻辑与 mt5/backtest_us_range_fade.py 逐条镜像（34 项单元测试锁定）   |
//| 默认参数 = 美盘推荐组：ref16 / H4 ADX 20-30 / 只做空 / 区间>=1%      |
//+------------------------------------------------------------------+
#property copyright "AlphaMaster"
#property version   "1.10"

#include <Trade\Trade.mqh>

enum EnumDirMode
{
   DIR_BOTH    = 0,   // 双向
   DIR_SHORT   = 1,   // 只做空
   DIR_LONG    = 2,   // 只做多
   DIR_COUNTER = 3    // 参照棒反向（阳线只做空、阴线只做多）
};

input double InpRiskPercent = 1.0;    // 每笔风险%（按权益反算手数）
input double InpBodyRatio   = 0.5;    // 参照棒实体/区间 下限
input double InpCenterTol   = 0.25;   // 实体中点偏离区间中点 上限（×区间）
input double InpRR          = 2.0;    // 止盈 = RR × 止损距离
input int    InpRefHour     = 16;     // 参照 H4 开盘小时（服务器钟）16=美盘 12=欧盘尾段
input EnumDirMode InpDirMode = DIR_SHORT;  // 方向模式
input bool   InpUseH4Adx    = true;   // H4 ADX 区间门 开关
input int    InpAdxPeriod   = 14;     // ADX 周期（14，Wilder）
input double InpH4AdxMin    = 20.0;   // H4 ADX 下限（< 挡当日）
input double InpH4AdxMax    = 30.0;   // H4 ADX 上限（>= 挡当日）
input double InpMinRange    = 0.0;    // 参照区间宽度下限（USD，0=关闭）
input double InpMinRangePct = 1.0;    // 参照区间宽度下限（占参照棒收盘价 %，0=关闭）
input bool   InpUseAdxM5    = false;  // M5 ADX 门 开关（回测未采用，默认关）
input double InpAdxMax      = 25.0;   // M5 ADX >= 此值挡单（仅 InpUseAdxM5=true 时生效）
input ulong  InpMagic       = 20261120;
input ulong  InpDeviation   = 20;     // 市价单允许滑点（points）

CTrade   g_trade;
int      g_adx_handle = INVALID_HANDLE;
int      g_h4_adx_handle = INVALID_HANDLE;
datetime g_last_bar   = 0;
int      g_day_key    = 0;
datetime g_ref_time   = 0;
datetime g_win_start  = 0;
datetime g_win_end    = 0;
bool     g_ref_loaded = false;
bool     g_eligible   = false;
double   g_upper = 0, g_lower = 0;
bool     g_allow_short = true;
bool     g_allow_long  = false;
bool     g_consumed[2] = {false, false};   // 0=short 1=long
bool     g_marked[2]   = {false, false};
double   g_extreme[2]  = {0, 0};
datetime g_mark_time[2]= {0, 0};
bool     g_holding  = false;
int      g_pos_dir  = 0;
double   g_pos_sl   = 0, g_pos_tp = 0;

//+------------------------------------------------------------------+
int DayKey(datetime t)
{
   MqlDateTime st; TimeToStruct(t, st);
   return st.year * 10000 + st.mon * 100 + st.day;
}

void SetDirs(double o, double c)
{
   switch(InpDirMode)
   {
      case DIR_BOTH:    g_allow_short = true;  g_allow_long = true;         break;
      case DIR_SHORT:   g_allow_short = true;  g_allow_long = false;        break;
      case DIR_LONG:    g_allow_short = false; g_allow_long = true;         break;
      case DIR_COUNTER: g_allow_short = (c >= o); g_allow_long = (c < o);    break;
      default:          g_allow_short = true;  g_allow_long = true;         break;
   }
}

void ResetDay(datetime t)
{
   g_day_key = DayKey(t);
   MqlDateTime st; TimeToStruct(t, st);
   st.hour = InpRefHour; st.min = 0; st.sec = 0;
   g_ref_time  = StructToTime(st);
   g_win_start = g_ref_time + 4 * 3600;
   g_win_end   = g_ref_time + 8 * 3600;
   g_ref_loaded = false;
   g_eligible   = false;
   g_allow_short = (InpDirMode != DIR_LONG);
   g_allow_long  = (InpDirMode != DIR_SHORT);
   for(int d = 0; d < 2; d++)
   {
      g_consumed[d] = false;
      g_marked[d]   = false;
      g_extreme[d]  = 0;
      g_mark_time[d]= 0;
   }
}

double H4AdxAt(datetime t)
{
   if(g_h4_adx_handle == INVALID_HANDLE) return -1;
   int sh = iBarShift(_Symbol, PERIOD_H4, t, true);
   if(sh < 0) return -1;
   double buf[1];
   if(CopyBuffer(g_h4_adx_handle, 0, sh, 1, buf) != 1) return -1;
   if(!MathIsValidNumber(buf[0])) return -1;
   return buf[0];
}

void LoadRef()
{
   int sh = iBarShift(_Symbol, PERIOD_H4, g_ref_time, true);
   if(sh < 0) return;                       // 参照棒还没出现
   double o = iOpen(_Symbol, PERIOD_H4, sh);
   double h = iHigh(_Symbol, PERIOD_H4, sh);
   double l = iLow(_Symbol, PERIOD_H4, sh);
   double c = iClose(_Symbol, PERIOD_H4, sh);
   g_ref_loaded = true;
   double rng = h - l;
   if(rng <= 0) { g_eligible = false; return; }
   if(InpMinRange > 0 && rng < InpMinRange) { g_eligible = false; return; }
   if(InpMinRangePct > 0 && rng < InpMinRangePct / 100.0 * c) { g_eligible = false; return; }
   double body = MathAbs(c - o);
   double dev  = MathAbs((o + c) / 2 - (h + l) / 2);
   if(!(body >= InpBodyRatio * rng && dev <= InpCenterTol * rng)) { g_eligible = false; return; }
   if(InpUseH4Adx)
   {
      double adx = H4AdxAt(g_ref_time);
      if(adx < 0 || adx < InpH4AdxMin || adx >= InpH4AdxMax) { g_eligible = false; return; }
   }
   g_eligible = true;
   g_upper = h; g_lower = l;
   SetDirs(o, c);
}

double AdxAt(int shift)
{
   if(!InpUseAdxM5) return -1;
   double buf[1];
   if(CopyBuffer(g_adx_handle, 0, shift, 1, buf) != 1) return -1;
   if(!MathIsValidNumber(buf[0])) return -1;
   return buf[0];
}

bool HasPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
         PositionGetInteger(POSITION_MAGIC) == (long)InpMagic)
         return true;
   }
   return false;
}

bool SendOrder(int dir, double entry, double sl, double tp)
{
   double tick_size  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   if(tick_size <= 0 || tick_value <= 0) return false;
   double risk_money = AccountInfoDouble(ACCOUNT_EQUITY) * InpRiskPercent / 100.0;
   double sl_dist    = MathAbs(entry - sl);
   if(sl_dist <= 0) return false;
   double lots = risk_money / (sl_dist / tick_size * tick_value);
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minl = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxl = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   if(step > 0) lots = MathFloor(lots / step) * step;
   if(lots < minl)
   {
      Print("US_Range_Fade: 计算手数 ", lots, " < 最小手数 ", minl, "，跳过该信号");
      return false;
   }
   if(lots > maxl) lots = maxl;
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   sl = NormalizeDouble(sl, digits);
   tp = NormalizeDouble(tp, digits);
   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetDeviationInPoints(InpDeviation);
   bool ok = (dir == 0)
      ? g_trade.Sell(lots, _Symbol, 0, sl, tp, "US_Range_Fade")
      : g_trade.Buy(lots, _Symbol, 0, sl, tp, "US_Range_Fade");
   if(ok)
   {
      g_pos_dir = dir; g_pos_sl = sl; g_pos_tp = tp;
      Print("US_Range_Fade: ", (dir == 0 ? "SELL" : "BUY"), " ", lots,
            " 手 @", entry, " SL=", sl, " TP=", tp);
   }
   else
      Print("US_Range_Fade: 下单失败 retcode=", g_trade.ResultRetcode(),
            " ", g_trade.ResultRetcodeDescription());
   return ok;
}

void CloseEOD()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
         PositionGetInteger(POSITION_MAGIC) == (long)InpMagic)
      {
         g_trade.SetExpertMagicNumber(InpMagic);
         g_trade.PositionClose(ticket);
         Print("US_Range_Fade: 窗口末强平 ticket=", ticket);
      }
   }
}

// 信号状态机（空仓时调用）：先处理已有标记的回归（空头优先），再处理新突破
void ProcessSignals(datetime bt, double h, double l, double c, double adx, bool live)
{
   for(int d = 0; d < 2; d++)
   {
      if(d == 0 && !g_allow_short) continue;
      if(d == 1 && !g_allow_long)  continue;
      if(!g_marked[d]) continue;
      bool returned;
      if(d == 0)
      {
         g_extreme[0] = MathMax(g_extreme[0], h);
         returned = (c <= g_upper);
      }
      else
      {
         g_extreme[1] = MathMin(g_extreme[1], l);
         returned = (c >= g_lower);
      }
      if(!returned) continue;
      g_marked[d] = false;
      if(InpUseAdxM5 && (adx < 0 || adx >= InpAdxMax))
      {
         g_consumed[d] = true;              // ADX 挡单 → 该轨当日作废
         Print("US_Range_Fade: M5 ADX=", adx, " 挡掉回归，轨道 ", d, " 当日作废");
         continue;
      }
      double entry = c, sl = g_extreme[d];
      double risk = (d == 0) ? sl - entry : entry - sl;
      if(risk <= 0) { g_consumed[d] = true; continue; }
      double tp = (d == 0) ? entry - InpRR * risk : entry + InpRR * risk;
      g_consumed[d] = true;
      if(live)
      {
         if(SendOrder(d, entry, sl, tp)) g_holding = true;
      }
      else
      {
         g_holding = true; g_pos_dir = d; g_pos_sl = sl; g_pos_tp = tp;
      }
      return;
   }
   if(g_allow_short && !g_consumed[0] && !g_marked[0] && c > g_upper)
   {
      g_marked[0] = true; g_extreme[0] = h; g_mark_time[0] = bt;
   }
   if(g_allow_long && !g_consumed[1] && !g_marked[1] && c < g_lower)
   {
      g_marked[1] = true; g_extreme[1] = l; g_mark_time[1] = bt;
   }
}

// 回放（dry）：含 SL/TP 模拟，双触先止损；用于 OnInit 重建当日状态
void ReplayBar(datetime bt, double h, double l, double c, double adx)
{
   if(g_holding)
   {
      bool hit_sl, hit_tp;
      if(g_pos_dir == 0) { hit_sl = (h >= g_pos_sl); hit_tp = (l <= g_pos_tp); }
      else               { hit_sl = (l <= g_pos_sl); hit_tp = (h >= g_pos_tp); }
      if(hit_sl || hit_tp) g_holding = false;
      return;
   }
   ProcessSignals(bt, h, l, c, adx, false);
}

//+------------------------------------------------------------------+
int OnInit()
{
   g_adx_handle = iADX(_Symbol, PERIOD_M5, InpAdxPeriod);
   if(g_adx_handle == INVALID_HANDLE)
   {
      Print("US_Range_Fade: iADX 创建失败");
      return INIT_FAILED;
   }
   g_h4_adx_handle = iADX(_Symbol, PERIOD_H4, InpAdxPeriod);
   if(g_h4_adx_handle == INVALID_HANDLE)
   {
      Print("US_Range_Fade: iADX(H4) 创建失败");
      return INIT_FAILED;
   }
   g_trade.SetExpertMagicNumber(InpMagic);
   g_holding = false;
   ResetDay(TimeCurrent());
   if(TimeCurrent() >= g_ref_time + 4 * 3600) LoadRef();
   if(g_eligible && TimeCurrent() >= g_win_start)
   {
      int start_sh = iBarShift(_Symbol, PERIOD_M5, g_win_start, true);
      for(int s = start_sh; s >= 1; s--)
      {
         datetime bt = iTime(_Symbol, PERIOD_M5, s);
         if(bt < g_win_start || bt >= g_win_end) continue;
         ReplayBar(bt, iHigh(_Symbol, PERIOD_M5, s), iLow(_Symbol, PERIOD_M5, s),
                   iClose(_Symbol, PERIOD_M5, s), AdxAt(s));
      }
   }
   if(HasPosition())                        // 重启接管：以真实持仓为准
   {
      g_holding = true;
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         ulong ticket = PositionGetTicket(i);
         if(ticket == 0 || !PositionSelectByTicket(ticket)) continue;
         if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
            PositionGetInteger(POSITION_MAGIC) == (long)InpMagic)
         {
            g_pos_dir = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) ? 0 : 1;
            g_pos_sl  = PositionGetDouble(POSITION_SL);
            g_pos_tp  = PositionGetDouble(POSITION_TP);
            break;
         }
      }
   }
   else
      g_holding = false;                    // 回放出的入场若已不在 → 不追溯补单
   g_last_bar = iTime(_Symbol, PERIOD_M5, 1);
   Print("US_Range_Fade: 初始化完成 day=", g_day_key,
         " eligible=", g_eligible, " holding=", g_holding,
         " mode=", (int)InpDirMode, " ref_hour=", InpRefHour);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(g_adx_handle != INVALID_HANDLE) IndicatorRelease(g_adx_handle);
   if(g_h4_adx_handle != INVALID_HANDLE) IndicatorRelease(g_h4_adx_handle);
   Comment("");
}

//+------------------------------------------------------------------+
void OnTick()
{
   datetime t1 = iTime(_Symbol, PERIOD_M5, 1);
   if(t1 == 0 || t1 == g_last_bar) return;  // 只处理新收盘的 M5 bar

   if(g_holding)
   {
      if(!HasPosition())
         g_holding = false;                 // SL/TP 已被 broker 执行
      else if(t1 >= g_win_end)
      {
         CloseEOD();                        // 窗口最后一根收盘后强平
         g_holding = false;
      }
      if(DayKey(t1) != g_day_key) ResetDay(t1);
      g_last_bar = t1;
      return;                              // 持仓棒跳过信号（与回测一致）
   }

   if(DayKey(t1) != g_day_key) ResetDay(t1);
   if(!g_ref_loaded && TimeCurrent() >= g_ref_time + 4 * 3600) LoadRef();

   if(g_eligible && t1 >= g_win_start && t1 < g_win_end)
   {
      double h = iHigh(_Symbol, PERIOD_M5, 1);
      double l = iLow(_Symbol, PERIOD_M5, 1);
      double c = iClose(_Symbol, PERIOD_M5, 1);
      ProcessSignals(t1, h, l, c, AdxAt(1), true);
   }
   g_last_bar = t1;

   Comment("US_Range_Fade  day=", g_day_key,
           "  ref=", (g_ref_loaded ? (g_eligible ? "合格" : "不合格") : "待形成"),
           "  上轨=", DoubleToString(g_upper, 2), " 下轨=", DoubleToString(g_lower, 2),
           "\n空轨: ", (g_consumed[0] ? "已作废" : (g_marked[0] ? "已标记" : "等待")),
           "  多轨: ", (g_consumed[1] ? "已作废" : (g_marked[1] ? "已标记" : "等待")),
           "  持仓=", g_holding);
}
//+------------------------------------------------------------------+
