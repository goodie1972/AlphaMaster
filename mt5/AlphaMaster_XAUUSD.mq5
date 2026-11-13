//+------------------------------------------------------------------+
//| AlphaMaster_XAUUSD.mq5                                            |
//| Auto-generated port of strategies/best_XAUUSD.json               |
//| formula tokens [25,81,61,103,93,104,31,115] on M30               |
//+------------------------------------------------------------------+
#property copyright "AlphaMaster port"
#property version   "1.00"

#include <Trade\Trade.mqh>

//--- 策略参数
input long     InpMagicNumber   = 20250101;   // 订单 magic number
input double   InpLot           = 0.01;       // 固定手数（对标 FIXED_LOT_BY_SYMBOL）
input double   InpMinExposure   = 0.05;       // |tanh(z)| 低于此值视为空仓
input int      InpHistoryBars   = 2000;       // 参与计算的已收盘 bar 数
input int      InpMinBars       = 800;        // 低于此根数不出信号（warm-up 保护）
input bool     InpTradeEnabled  = true;       // false = 只计算信号不下单
input int      InpDeviation     = 30;         // 允许滑点（points）
input string   InpComment       = "AlphaMaster_XAUUSD";
input double   InpStopAtrMult   = 20.0;       // 灾难止损 = k * ATR(14) 距离开仓价；0 = 禁用（不开 SL）
input string   InpRequiredSymbol = "XAUUSD";   // 只允许该品种（公式仅对 XAUUSD 拟合）
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_M30;  // 计算周期（本策略训练于 M30）

//--- 与 Python 侧一致的常量
#define EPS_FEAT     1e-9     // MT5FeatureEngineer._EPS
#define NORM_WINDOW  200      // _NORM_WINDOW（因果 rolling robust 归一化）
#define CLIP_BOUND   5.0      // _CLIP_BOUND
#define ZROLL_WINDOW 500      // vm._normalize_output 滚动 z-score 窗口
#define Z_CLIP       3.0

CTrade trade;

bool     g_symbol_ok    = true;
datetime g_lastSignalBar = 0;
double   g_lastZ          = 0.0;
double   g_lastPosition   = 0.0;
int      g_lastDirection  = 0;
string   g_status         = "init";

//+------------------------------------------------------------------+
//| 基础工具                                                          |
//+------------------------------------------------------------------+
bool IsBad(const double v)
{
   return !MathIsValidNumber(v);
}

//--- 特征层 _clean：nan/inf -> 0
void CleanFeat(double &a[], const int n)
{
   for(int i = 0; i < n; i++)
      if(IsBad(a[i])) a[i] = 0.0;
}

//--- VM 层 nan_to_num：nan->0, +inf->1, -inf->-1
void CleanVM(double &a[], const int n)
{
   for(int i = 0; i < n; i++)
   {
      if(a[i] != a[i])                     a[i] = 0.0;              // NaN -> 0
      else if(!MathIsValidNumber(a[i]))    a[i] = (a[i] > 0.0 ? 1.0 : -1.0);
   }
}

//--- 因果左 pad 0 取值：index<0 或 >=n 视为 0
double PadVal(const double &a[], const int idx, const int n)
{
   if(idx < 0 || idx >= n) return 0.0;
   return a[idx];
}

//--- MFI 的前一期典型价：t==0 时取自身（使首根 pos=neg=0）
double PrevVal(const double &a[], const int t)
{
   if(t == 0) return a[0];
   return a[t - 1];
}

//--- 下中位数（与 torch.median 对偶数长度取较小中间值一致）
double LowerMedian(double &buf[], const int n)
{
   ArraySort(buf);              // 升序
   if(n <= 0) return 0.0;
   return buf[(n - 1) / 2];
}

//+------------------------------------------------------------------+
//| 特征 MFI14：带量版 RSI，输出直接 clamp 到 [-1,1]，不做 robust 归一 |
//+------------------------------------------------------------------+
void FeatureMFI14(const double &o[], const double &h[], const double &l[],
                  const double &c[], const double &v[], const int n, double &out[])
{
   double typ[], mf[], pos[], neg[];
   ArrayResize(typ, n); ArrayResize(mf, n);
   ArrayResize(pos, n); ArrayResize(neg, n);

   for(int t = 0; t < n; t++)
   {
      typ[t] = (h[t] + l[t] + c[t]) / 3.0;
      mf[t]  = typ[t] * v[t];
      double pc = PrevVal(typ, t);   // t==0 时 pc=typ[0] -> pos=neg=0
      pos[t] = (typ[t] > pc) ? mf[t] : 0.0;
      neg[t] = (typ[t] < pc) ? mf[t] : 0.0;
   }

   ArrayResize(out, n);
   const int w = 14;
   for(int t = 0; t < n; t++)
   {
      double psum = 0.0, nsum = 0.0;
      for(int k = 0; k < w; k++)          // _rolling_mean*w == 因果窗口求和
      {
         int idx = t - (w - 1) + k;
         if(idx < 0) continue;
         psum += pos[idx];
         nsum += neg[idx];
      }
      double mfr = psum / (nsum + EPS_FEAT);
      double mfi = 100.0 - (100.0 / (1.0 + mfr));
      double val = (mfi - 50.0) / 50.0;
      if(val >  1.0) val =  1.0;
      if(val < -1.0) val = -1.0;
      out[t] = val;
   }
   CleanFeat(out, n);
}

//+------------------------------------------------------------------+
//| 特征 AC2：对数收益的二阶自相关（窗口 20、lag 2），clamp [-1,1]     |
//+------------------------------------------------------------------+
void FeatureAC2(const double &c[], const int n, double &out[])
{
   double ret[];
   ArrayResize(ret, n);
   ret[0] = 0.0;
   for(int t = 1; t < n; t++)
      ret[t] = MathLog(c[t] / (c[t - 1] + EPS_FEAT));

   ArrayResize(out, n);
   const int w = 20, lag = 2;
   for(int t = 0; t < n; t++)
   {
      double xm = 0.0, ym = 0.0;
      for(int k = 0; k < w; k++)
      {
         xm += PadVal(ret, t - (w + lag - 1) + k, n);
         ym += PadVal(ret, t - (w - 1) + k, n);
      }
      xm /= w; ym /= w;

      double cov = 0.0, vx = 0.0, vy = 0.0;
      for(int k = 0; k < w; k++)
      {
         double xv = PadVal(ret, t - (w + lag - 1) + k, n) - xm;
         double yv = PadVal(ret, t - (w - 1) + k, n) - ym;
         cov += xv * yv;
         vx  += xv * xv;
         vy  += yv * yv;
      }
      cov /= w; vx /= w; vy /= w;
      double corr = cov / (MathSqrt(vx) * MathSqrt(vy) + 1e-8);
      if(corr >  1.0) corr =  1.0;
      if(corr < -1.0) corr = -1.0;
      out[t] = corr;
   }
   CleanFeat(out, n);
}

//+------------------------------------------------------------------+
//| 特征 TREND_STRENGTH_50：50 期 SLOPE*R2，再做 robust 归一化        |
//+------------------------------------------------------------------+
void FeatureTrendStrength50(const double &c[], const int n, double &out[])
{
   double raw[];
   ArrayResize(raw, n);
   const int w = 50;
   const double tmean = (w - 1) / 2.0;
   double tvar = 0.0;
   for(int k = 0; k < w; k++)
   {
      double tc = k - tmean;
      tvar += tc * tc;
   }

   for(int t = 0; t < n; t++)
   {
      double xm = 0.0;
      for(int k = 0; k < w; k++) xm += PadVal(c, t - (w - 1) + k, n);
      xm /= w;

      double num = 0.0, ss_tot = 0.0;
      double slope = 0.0;
      for(int k = 0; k < w; k++)
      {
         double tc = k - tmean;
         double dv = PadVal(c, t - (w - 1) + k, n) - xm;
         num    += dv * tc;
         ss_tot += dv * dv;
      }
      slope = num / (tvar + EPS_FEAT);

      double ss_res = 0.0;
      for(int k = 0; k < w; k++)
      {
         double tc  = k - tmean;
         double wnd = PadVal(c, t - (w - 1) + k, n);
         double pr  = xm + slope * tc;
         double dv  = wnd - pr;
         ss_res += dv * dv;
      }
      double r2 = 1.0 - ss_res / (ss_tot + EPS_FEAT);
      if(r2 >  1.0) r2 =  1.0;
      if(r2 <  0.0) r2 =  0.0;

      double slope_norm = slope / (xm + EPS_FEAT);
      raw[t] = slope_norm * r2;
   }
   CleanFeat(raw, n);
   RobustNorm(raw, n, out);
   CleanFeat(out, n);
}

//+------------------------------------------------------------------+
//| 因果 rolling robust 归一化（median/MAD，窗口 200，warm-up 输出 0） |
//+------------------------------------------------------------------+
void RobustNorm(const double &x[], const int n, double &out[])
{
   ArrayResize(out, n);
   const int w = NORM_WINDOW;
   if(n < w)                     // 不足一个窗口：全部视为 warm-up 输出 0
   {
      for(int i = 0; i < n; i++) out[i] = 0.0;
      return;
   }
   double buf[];
   ArrayResize(buf, w);
   for(int t = 0; t < n; t++)
   {
      if(t < w - 1) { out[t] = 0.0; continue; }
      for(int k = 0; k < w; k++) buf[k] = PadVal(x, t - (w - 1) + k, n);
      double med = LowerMedian(buf, w);
      for(int k = 0; k < w; k++) buf[k] = MathAbs(PadVal(x, t - (w - 1) + k, n) - med);
      double mad = LowerMedian(buf, w) + 1e-6;
      double val = (x[t] - med) / mad;
      if(val >  CLIP_BOUND) val =  CLIP_BOUND;
      if(val < -CLIP_BOUND) val = -CLIP_BOUND;
      out[t] = val;
   }
   CleanFeat(out, n);
}

//+------------------------------------------------------------------+
//| 算子层                                                            |
//+------------------------------------------------------------------+
//--- TS_STD_10：ddof=0 因果滚动标准差，下界 1e-6
void OpTsStd10(const double &x[], const int n, double &out[])
{
   ArrayResize(out, n);
   const int d = 10;
   for(int t = 0; t < n; t++)
   {
      double m = 0.0;
      for(int k = 0; k < d; k++) m += PadVal(x, t - (d - 1) + k, n);
      m /= d;
      double v = 0.0;
      for(int k = 0; k < d; k++)
      {
         double dv = PadVal(x, t - (d - 1) + k, n) - m;
         v += dv * dv;
      }
      v /= d;
      out[t] = MathSqrt(v) + 1e-6;
   }
   for(int i = 0; i < n; i++) if(out[i] != out[i]) out[i] = 0.0;
}

//--- SCALE：因果累积 L1 归一 scale[t] = x[t] / (sum|x[0..t]| + 1e-6)
void OpScale(const double &x[], const int n, double &out[])
{
   ArrayResize(out, n);
   double cum = 0.0;
   for(int t = 0; t < n; t++)
   {
      cum += MathAbs(x[t]);
      out[t] = x[t] / (cum + 1e-6);
   }
}

//--- EMA_5：exact 递推，alpha = 2/(span+1)，out[0] = x[0]
void OpEma5(const double &x[], const int n, double &out[])
{
   ArrayResize(out, n);
   if(n <= 0) return;
   const double alpha = 2.0 / (5.0 + 1.0);
   out[0] = x[0];
   for(int t = 1; t < n; t++)
      out[t] = alpha * x[t] + (1.0 - alpha) * out[t - 1];
}

//--- COVARIANCE_10：10 期因果滑动协方差
void OpTsCov10(const double &x[], const double &y[], const int n, double &out[])
{
   ArrayResize(out, n);
   const int d = 10;
   for(int t = 0; t < n; t++)
   {
      double mx = 0.0, my = 0.0;
      for(int k = 0; k < d; k++)
      {
         mx += PadVal(x, t - (d - 1) + k, n);
         my += PadVal(y, t - (d - 1) + k, n);
      }
      mx /= d; my /= d;
      double cov = 0.0;
      for(int k = 0; k < d; k++)
         cov += (PadVal(x, t - (d - 1) + k, n) - mx) * (PadVal(y, t - (d - 1) + k, n) - my);
      out[t] = cov / d;
   }
   for(int i = 0; i < n; i++) if(out[i] != out[i]) out[i] = 0.0;
}

//--- MIN：逐元素取小
void OpMin(const double &a[], const double &b[], const int n, double &out[])
{
   ArrayResize(out, n);
   for(int t = 0; t < n; t++)
      out[t] = (a[t] < b[t]) ? a[t] : b[t];
}

//+------------------------------------------------------------------+
//| ATR：与 Python model_core.features._atr 同口径                    |
//|   = TR 的 w 期**简单均值** + **零填充 warmup**                     |
//|   不用 iATR —— iATR 是 Wilder 平滑，与训练链路口径不同，禁止使用    |
//|   warmup 期分母恒为 w，缺的根按 0 计（不是 NaN、不是完整窗口）      |
//+------------------------------------------------------------------+
void FeatureAtrSma(const double &h[], const double &l[], const double &c[],
                   const int n, const int w, double &atr[])
{
   ArrayResize(atr, n);
   if(n <= 0 || w <= 0) return;

   double tr[];
   ArrayResize(tr, n);
   for(int i = 0; i < n; i++)
   {
      double pc = (i == 0) ? c[0] : c[i - 1];
      double a = h[i] - l[i];
      double b = MathAbs(h[i] - pc);
      double d = MathAbs(l[i] - pc);
      tr[i] = MathMax(a, MathMax(b, d));
   }

   for(int i = 0; i < n; i++)
   {
      double sum = 0.0;
      int from = i - w + 1;
      if(from < 0) from = 0;
      for(int j = from; j <= i; j++) sum += tr[j];
      atr[i] = sum / w;
   }
}

//+------------------------------------------------------------------+
//| 因子输出标准化：滚动 z-score（窗口 500，ddof=1），clip +-3         |
//+------------------------------------------------------------------+
void NormalizeOutput(const double &x[], const int n, double &out[])
{
   ArrayResize(out, n);

   //--- 全局常数检测（torch.std 默认 ddof=1）
   double mean = 0.0;
   for(int i = 0; i < n; i++) mean += x[i];
   mean /= n;
   double vs = 0.0;
   for(int i = 0; i < n; i++) vs += (x[i] - mean) * (x[i] - mean);
   double gstd = (n > 1) ? MathSqrt(vs / (n - 1)) : 0.0;
   if(gstd < 1e-6)
   {
      for(int i = 0; i < n; i++) out[i] = x[i];   // 常数因子，原值返回
      return;
   }

   const int w = ZROLL_WINDOW;
   double buf[];
   ArrayResize(buf, w);
   for(int t = 0; t < n; t++)
   {
      if(t < w - 1) { out[t] = 0.0; continue; }   // warm-up 输出 0
      double s = 0.0, ss = 0.0;
      for(int k = 0; k < w; k++)
      {
         double v = PadVal(x, t - (w - 1) + k, n);
         s  += v;
         ss += v * v;
      }
      double m  = s / w;
      double vr = (ss - s * s / w) / (w - 1);     // 无偏方差
      if(vr < 0.0) vr = 0.0;
      double sd = MathSqrt(vr);
      if(sd < 1e-8) sd = 1e-8;
      double z = (x[t] - m) / sd;
      if(z >  Z_CLIP) z =  Z_CLIP;
      if(z < -Z_CLIP) z = -Z_CLIP;
      out[t] = z;
   }
   CleanFeat(out, n);
}

//+------------------------------------------------------------------+
//| 计算因子：tokens [25,81,61,103,93,104,31,115] 按 VM 栈序执行       |
//|   push MFI14 ; TS_STD_10 ;                                       |
//|   push AC2 ; SCALE ; EMA_5 ;                                     |
//|   COVARIANCE_10 ;                                                |
//|   push TREND_STRENGTH_50 ; MIN                                   |
//+------------------------------------------------------------------+
bool ComputeSignal(const double &o[], const double &h[], const double &l[],
                   const double &c[], const double &v[], const int n,
                   double &zLast, double &position, int &direction)
{
   double mfi[], ac2[], ts50[];
   FeatureMFI14(o, h, l, c, v, n, mfi);
   FeatureAC2(c, n, ac2);
   FeatureTrendStrength50(c, n, ts50);

   double s1[], sc[], e5[], cov[], fmin[];
   OpTsStd10(mfi, n, s1);  CleanVM(s1, n);
   OpScale(ac2, n, sc);    CleanVM(sc, n);
   OpEma5(sc, n, e5);      CleanVM(e5, n);
   OpTsCov10(s1, e5, n, cov); CleanVM(cov, n);
   OpMin(cov, ts50, n, fmin); CleanVM(fmin, n);

   double z[];
   NormalizeOutput(fmin, n, z);

   zLast = z[n - 1];
   if(IsBad(zLast)) { zLast = 0.0; }
   position = MathTanh(zLast);
   if(position >=  InpMinExposure)      direction =  1;
   else if(position <= -InpMinExposure) direction = -1;
   else                                 direction =  0;
   return true;
}

//+------------------------------------------------------------------+
//| 取已收盘 bar（shift 从 1 开始，排除当前未收盘 bar）                 |
//+------------------------------------------------------------------+
bool LoadClosedBars(double &o[], double &h[], double &l[], double &c[],
                    double &v[], datetime &t[], int &n)
{
   int avail = Bars(_Symbol, InpTimeframe);
   if(avail < InpMinBars) { n = 0; return false; }

   int need = (InpHistoryBars < InpMinBars) ? InpMinBars : InpHistoryBars;
   if(need > avail - 1) need = avail - 1;

   MqlRates rates[];
   ArraySetAsSeries(rates, false);       // 复制后 index 0 = 最旧，直接时间升序
   int got = CopyRates(_Symbol, InpTimeframe, 1, need, rates);
   if(got <= 0) { n = 0; return false; }

   n = ArraySize(rates);
   if(n < InpMinBars) { n = 0; return false; }   // 样本不足会退化成全 0 信号
   ArrayResize(o, n); ArrayResize(h, n); ArrayResize(l, n);
   ArrayResize(c, n); ArrayResize(v, n); ArrayResize(t, n);
   for(int i = 0; i < n; i++)
   {
      o[i] = rates[i].open;
      h[i] = rates[i].high;
      l[i] = rates[i].low;
      c[i] = rates[i].close;
      v[i] = (double)rates[i].tick_volume;   // Python 侧 volume = tick_volume
      t[i] = rates[i].time;
   }
   return true;
}

//+------------------------------------------------------------------+
//| 当前该 magic 下的净方向                                            |
//+------------------------------------------------------------------+
int CurrentDirection()
{
   double net = 0.0;
   bool hasBuy = false, hasSell = false;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      long typ = PositionGetInteger(POSITION_TYPE);
      double vol = PositionGetDouble(POSITION_VOLUME);
      if(typ == POSITION_TYPE_BUY)  { hasBuy = true;  net += vol; }
      if(typ == POSITION_TYPE_SELL) { hasSell = true; net -= vol; }
   }
   if(hasBuy && hasSell) return 9;       // 多空并存，需要先清理
   if(net > 0.0) return 1;
   if(net < 0.0) return -1;
   return 0;
}

void CloseAllMine()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      trade.PositionClose(tk);
   }
}

//+------------------------------------------------------------------+
//| 对一笔 EA 持仓挂灾难止损（broker 侧 ATR SL）。                       |
//|   SL = entry ∓ k*ATR(14)；atrVal<=0 或 k<=0 ⇒ 不开（0=禁用/数据异常）。|
//|   成功返回 true 并打印 sl_attach 心跳；失败打印 FAILED（C2 当「SL 未挂」告警）。|
//+------------------------------------------------------------------+
bool AttachStopLoss(const ulong ticket, const int side, const double atrVal,
                    const double entry, const double k, double &outSl)
{
   if(atrVal <= 0.0 || k <= 0.0) return false;

   double sl = (side == 1) ? entry - k * atrVal : entry + k * atrVal;
   outSl = sl;

   // 保留既有 TP（现在为 0，但不覆盖别处设的 TP）；只加 SL 不加 TP
   double tp = 0.0;
   if(PositionSelectByTicket(ticket)) tp = PositionGetDouble(POSITION_TP);

   if(!trade.PositionModify(ticket, sl, tp))
   {
      PrintFormat("[AlphaMaster] sl_attach FAILED ticket=%llu side=%d entry=%.3f "
                  "atr=%.5f k=%.1f sl=%.5f retcode=%u",
                  ticket, side, entry, atrVal, k, sl, trade.ResultRetcode());
      return false;
   }
   PrintFormat("[AlphaMaster] sl_attach ticket=%llu side=%d entry=%.3f "
               "atr=%.5f k=%.1f sl=%.5f",
               ticket, side, entry, atrVal, k, sl);
   return true;
}

void OpenDir(const int dir, const double atrLast)
{
   bool ok = false;
   if(dir == 1)      ok = trade.Buy(InpLot, _Symbol, 0.0, 0.0, 0.0, InpComment);
   else if(dir == -1) ok = trade.Sell(InpLot, _Symbol, 0.0, 0.0, 0.0, InpComment);
   if(!ok) return;

   // 用实际成交价作 entry（不是请求价，避免滑点让 SL 贴在错价位）；市场单的 order ticket = position ticket
   ulong tk = trade.ResultOrder();
   if(tk == 0) return;

   double entry = 0.0, sl = 0.0;
   if(PositionSelectByTicket(tk))
   {
      entry = PositionGetDouble(POSITION_PRICE_OPEN);
      long typ = PositionGetInteger(POSITION_TYPE);
      int side = (typ == POSITION_TYPE_BUY) ? 1 : -1;
      AttachStopLoss(tk, side, atrLast, entry, InpStopAtrMult, sl);
   }
}

//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(InpDeviation);
   trade.SetTypeFillingBySymbol(_Symbol);
   Print("AlphaMaster_XAUUSD ready. symbol=", _Symbol,
         " bars=", Bars(_Symbol, InpTimeframe),
         " trade=", InpTradeEnabled);

   if(_Symbol != InpRequiredSymbol)
   {
      g_symbol_ok = false;
      g_status = StringFormat("禁用：本公式仅拟合 %s，当前图表是 %s",
                              InpRequiredSymbol, _Symbol);
      Comment(g_status);
      Print("[AlphaMaster] ", g_status, " —— 不计算不下单");
      return INIT_SUCCEEDED;
   }

   //--- 与 Python runner 一致：启动即按最后一根已收盘 bar 调仓一次，
   //--- 否则周末/无 tick 时段挂载后 EA 不会产出任何信号
   EvaluateAndReconcile(true);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) { Comment(""); }

//+------------------------------------------------------------------+
void OnTick()
{
   EvaluateAndReconcile(false);
}

void EvaluateAndReconcile(const bool force)
{
   if(!g_symbol_ok) return;

   datetime barTime = iTime(_Symbol, InpTimeframe, 1);   // 最近已收盘 bar 的开盘时间
   if(barTime == 0) return;

   //--- 未收盘不计算，避免信号随 tick 抖动
   if(!force && g_lastSignalBar == barTime) return;

   double o[], h[], l[], c[], v[];
   datetime t[];
   int n = 0;
   if(!LoadClosedBars(o, h, l, c, v, t, n))
   {
      g_status = "waiting for bars";
      return;
   }

   g_lastSignalBar = barTime;

   // 每根已收盘 bar 算 ATR(14)（与 Python 训练链同口径，FeatureAtrSma），开仓/修复 SL 都要用
   double atr[];
   FeatureAtrSma(h, l, c, n, 14, atr);
   double atrLast = atr[n - 1];

   double z, pos;
   int dir;
   ComputeSignal(o, h, l, c, v, n, z, pos, dir);
   g_lastZ = z; g_lastPosition = pos; g_lastDirection = dir;

   PrintFormat("[AlphaMaster] bar=%s closedBars=%d z=%+.5f tanh=%+.5f dir=%d atr=%.5f",
               TimeToString(t[n - 1]), n, z, pos, dir, atrLast);

   if(!InpTradeEnabled)
   {
      g_status = "signal only";
      ShowPanel();
      return;
   }

   int cur = CurrentDirection();
   if(cur == 9)
   {
      CloseAllMine();
      g_status = "cleaned mixed hedge";
      ShowPanel();
      return;
   }
   if(cur == dir) g_status = "hold";
   else if(dir == 0) { CloseAllMine(); g_status = "flat (below threshold)"; }
   else              { if(cur != 0) CloseAllMine();
                       OpenDir(dir, atrLast);
                       g_status = (dir == 1 ? "long" : "short"); }

   // 修复 pass：每轮 reconcile（含 init）把本 EA 持仓的 SL 挂回/校正，
   // 这样手工挪走或进程重启后丢的 SL 会在下根收盘自动挂回。
   double atrNow = atrLast;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      long typ = PositionGetInteger(POSITION_TYPE);
      if(typ != POSITION_TYPE_BUY && typ != POSITION_TYPE_SELL) continue;
      int side = (typ == POSITION_TYPE_BUY) ? 1 : -1;
      double entry = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl = 0.0;
      AttachStopLoss(tk, side, atrNow, entry, InpStopAtrMult, sl);
   }

   ShowPanel();
}

void ShowPanel()
{
   Comment(StringFormat("AlphaMaster XAUUSD " + EnumToString(InpTimeframe) + "\nz=%+.4f  tanh=%+.4f  dir=%d\n%s",
                        g_lastZ, g_lastPosition, g_lastDirection, g_status));
}
//+------------------------------------------------------------------+
