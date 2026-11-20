//+------------------------------------------------------------------+
//| ST_QQE_Recipe_EA.mq5                                             |
//| ST+QQE champion recipe EA (places real orders when enabled).     |
//+------------------------------------------------------------------+
// Recipe (backtested on XAUUSD M30 2006-2026, cost 0.02%/side,
// bar-close decisions, no look-ahead; IS 2006-2018 net +6.69%/y,
// OOS 2019-2026 net +14.74%/y sharpe 0.57 mdd 26.4%; probes passed:
// 1-bar-delay execution, direction poison, OOS half-split):
//   trigger : slow QQE MOD (RSI len 20, smoothing 12, both lines),
//             up/dn signal ONSET bar (signal valid now, invalid prev)
//   confirm : SuperTrend AI (default params) trend direction must
//             agree with the signal direction
//   gate    : efficiency ratio ER96 = |C-C[96]| / sum|ret| >= 0.35
//   exit    : opposite QQE onset -> close; if it also passes the
//             confirm+gate -> reverse instead
//   sizing  : fixed lot (default) or risk-% of equity against the
//             disaster stop distance; broker-side ATR disaster SL
// Computation code is copied verbatim from the validated standalone
// ports QQE_MOD.mq5 / SuperTrend_AI.mq5 (Python-oracle checked).
// NOTE: the backtest had no stop-loss; the disaster SL is a safety
// overlay (default 20xATR(14), rarely hit). Set 0 to disable.
//+------------------------------------------------------------------+
#property copyright "Personal use"
#property version   "1.00"

#include <Trade\Trade.mqh>

input group "Trading"
input long     InpMagicNumber   = 20261118;      // 订单 magic number
input bool     InpTradeEnabled  = true;          // false = 只算信号不下单
input bool     InpRiskSizing    = false;         // true = 按风险%反算手数（需止损>0）
input double   InpRiskPercent   = 1.0;           // 每笔风险占净值 %
input double   InpLot           = 0.01;          // 固定手数（InpRiskSizing=false 时用）
input double   InpStopAtrMult   = 20.0;          // 灾难止损 = k*ATR(14)；0 = 禁用
input int      InpDeviation     = 30;            // 允许滑点（points）
input string   InpComment       = "ST_QQE_Recipe";
input string   InpRequiredSymbol= "XAUUSD";      // 只允许该品种（配方仅回测于此）
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_M30; // 计算周期（配方回测于 M30）

input group "Slow QQE trigger (recipe: 20 / 12)"
input int    InpRsiLenP = 20;     // Primary RSI Length
input int    InpSmP     = 12;     // Primary RSI Smoothing
input double InpFacP    = 3.0;    // Primary QQE Factor
input int    InpRsiLenS = 20;     // Secondary RSI Length
input int    InpSmS     = 12;     // Secondary RSI Smoothing
input double InpFacS    = 1.61;   // Secondary QQE Factor
input double InpThrS    = 3.0;    // Secondary Threshold
input int    InpBollLen  = 50;    // Bollinger Length
input double InpBollMult = 0.35;  // Bollinger Multiplier

input group "SuperTrend AI confirm (defaults)"
input int    InpLength     = 10;    // ATR Length
input int    InpMinMult    = 1;     // Factor Range (min)
input int    InpMaxMult    = 5;     // Factor Range (max)
input double InpStep       = 0.5;   // Step
input double InpPerfAlpha  = 10;    // Performance Memory
input int    InpFromCluster= 2;     // From Cluster (0=Worst 1=Average 2=Best)
input int    InpMaxIter    = 1000;  // k-means Maximum Iteration Steps

input group "Regime gate / data"
input int    InpErLen       = 96;    // Efficiency Ratio window
input double InpErThreshold = 0.35;  // ER minimum to accept a signal
input int    InpHistoryBars = 5000;  // 参与计算的已收盘 bar 数
input int    InpMinBars     = 400;   // 低于此根数不交易（warm-up 保护）

CTrade trade;

bool     g_symbol_ok     = true;
datetime g_lastSignalBar = 0;
int      g_lastTarget    = 0;
int      g_lastEvent     = 0;
double   g_lastEr        = 0.0;
int      g_lastStDir     = 0;
string   g_status        = "init";

#define NA EMPTY_VALUE
bool IsNa(const double v){ return(v==EMPTY_VALUE); }
double nz(const double v,const double d=0.0){ return(IsNa(v)?d:v); }
double sgn(const double v){ if(IsNa(v)) return(NA); return(v>0?1.0:(v<0?-1.0:0.0)); }
double PMin(const double a,const double b){ return(IsNa(a)||IsNa(b))?NA:MathMin(a,b); }
double PMax(const double a,const double b){ return(IsNa(a)||IsNa(b))?NA:MathMax(a,b); }

//+------------------------------------------------------------------+
//| Shared math helpers (verbatim from the validated indicator ports)|
//+------------------------------------------------------------------+
void CalcSMA(const double &x[],const int n,const int len,double &out[])
  {
   ArrayResize(out,n);
   double s=0.0; int bad=0;
   for(int i=0;i<n;i++)
     {
      double v=x[i];
      if(i>=len){ double old=x[i-len]; if(IsNa(old)) bad--; else s-=old; }
      if(IsNa(v)) bad++; else s+=v;
      out[i]=(i>=len-1 && bad==0)? s/len : NA;
     }
  }
void CalcEMA(const double &x[],const int n,const int len,double &out[])
  {
   ArrayResize(out,n);
   double a=2.0/(len+1), prev=NA;
   for(int i=0;i<n;i++)
     {
      double v=x[i];
      if(IsNa(v)){ out[i]=NA; prev=NA; continue; }
      out[i]=IsNa(prev)? v : a*v+(1.0-a)*prev;
      prev=out[i];
     }
  }
void CalcRMA(const double &x[],const int n,const int len,double &out[])
  {
   ArrayResize(out,n);
   double a=1.0/len, prev=NA, sm[];
   CalcSMA(x,n,len,sm);
   for(int i=0;i<n;i++)
     {
      if(IsNa(prev)) out[i]=sm[i];
      else { double v=x[i]; out[i]=IsNa(v)?NA:a*v+(1.0-a)*prev; }
      prev=out[i];
     }
  }
void CalcRSI(const double &x[],const int n,const int len,double &out[])
  {
   ArrayResize(out,n);
   double u[],d[]; ArrayResize(u,n); ArrayResize(d,n);
   u[0]=NA; d[0]=NA;
   for(int i=1;i<n;i++){ double ch=x[i]-x[i-1]; u[i]=MathMax(ch,0.0); d[i]=MathMax(-ch,0.0); }
   double ru[],rd[]; CalcRMA(u,n,len,ru); CalcRMA(d,n,len,rd);
   for(int i=0;i<n;i++)
     {
      if(IsNa(ru[i])||IsNa(rd[i])||rd[i]==0.0){ out[i]=NA; continue; }
      double rs=ru[i]/rd[i];
      out[i]=100.0-100.0/(1.0+rs);
     }
  }
void CalcStdev(const double &x[],const int n,const int len,double &out[])
  {
   ArrayResize(out,n);
   double m[]; CalcSMA(x,n,len,m);
   for(int i=0;i<n;i++)
     {
      out[i]=NA;
      if(i<len-1||IsNa(m[i])) continue;
      double mu=m[i], s=0.0; bool anyna=false;
      for(int j=i-len+1;j<=i;j++){ if(IsNa(x[j])){anyna=true;break;} double dv=x[j]-mu; s+=dv*dv; }
      if(!anyna) out[i]=MathSqrt(s/len);
     }
  }
void CalcATR(const double &h[],const double &l[],const double &c[],const int n,const int len,double &out[])
  {
   double tr[]; ArrayResize(tr,n);
   tr[0]=h[0]-l[0];
   for(int i=1;i<n;i++)
      tr[i]=MathMax(MathMax(h[i]-l[i],MathAbs(h[i]-c[i-1])),MathAbs(l[i]-c[i-1]));
   CalcRMA(tr,n,len,out);
  }
bool CrossAt(const double &a[],const double &b[],const int i,const int sa,const int sb)
  {
   if(i-1-sa<0||i-1-sb<0||i-sa<0||i-sb<0) return(false);
   double A=a[i-sa],A1=a[i-1-sa],B=b[i-sb],B1=b[i-1-sb];
   if(IsNa(A)||IsNa(A1)||IsNa(B)||IsNa(B1)) return(false);
   bool co=(A1<=B1)&&(A>B);
   bool cu=(A1>=B1)&&(A<B);
   return(co||cu);
  }

//+------------------------------------------------------------------+
//| QQE MOD (verbatim from QQE_MOD.mq5)                              |
//+------------------------------------------------------------------+
void CalculateQQE(const double &c[],const int n,const int rsiLen,const int sm,const double fac,
                  double &trend[],double &srsi[])
  {
   double r[]; CalcRSI(c,n,rsiLen,r);
   CalcEMA(r,n,sm,srsi);
   double atrRsi[]; ArrayResize(atrRsi,n); atrRsi[0]=NA;
   for(int i=1;i<n;i++)
      atrRsi[i]=(IsNa(srsi[i])||IsNa(srsi[i-1]))?NA:MathAbs(srsi[i-1]-srsi[i]);
   int wild=rsiLen*2-1;
   double sAtr[]; CalcEMA(atrRsi,n,wild,sAtr);
   double dyn[]; ArrayResize(dyn,n);
   for(int i=0;i<n;i++) dyn[i]=IsNa(sAtr[i])?NA:sAtr[i]*fac;

   double lb[],sb[],td[]; ArrayResize(lb,n); ArrayResize(sb,n); ArrayResize(td,n); ArrayResize(trend,n);
   for(int i=0;i<n;i++)
     {
      double sr=srsi[i];
      double sr1=(i>0)?srsi[i-1]:NA;
      double lb1=(i>0)?lb[i-1]:NA;
      double sb1=(i>0)?sb[i-1]:NA;
      double ad=dyn[i];
      double newShort=(IsNa(sr)||IsNa(ad))?NA:sr+ad;
      double newLong =(IsNa(sr)||IsNa(ad))?NA:sr-ad;
      bool condL=(!IsNa(sr1)&&!IsNa(lb1)&&sr1>lb1)&&(!IsNa(sr)&&sr>lb1);
      lb[i]=condL?PMax(lb1,newLong):newLong;
      bool condS=(!IsNa(sr1)&&!IsNa(sb1)&&sr1<sb1)&&(!IsNa(sr)&&sr<sb1);
      sb[i]=condS?PMin(sb1,newShort):newShort;
     }
   for(int i=0;i<n;i++)
     {
      bool lbc=CrossAt(lb,srsi,i,1,0);
      bool cs =CrossAt(srsi,sb,i,0,1);
      if(cs) td[i]=1.0; else if(lbc) td[i]=-1.0; else td[i]=(i>0)?td[i-1]:NA;
      trend[i]=(td[i]==1.0)?lb[i]:sb[i];
     }
  }

//+------------------------------------------------------------------+
//| SuperTrend AI core (verbatim from SuperTrend_AI.mq5); outputs    |
//| the oscillator direction series osArr (1=up trend, 0=down).      |
//+------------------------------------------------------------------+
double PercentileLinear(const double &data[],const int n,const double p)
  {
   double s[]; ArrayResize(s,0);
   for(int i=0;i<n;i++) if(!IsNa(data[i])){ int k=ArraySize(s); ArrayResize(s,k+1); s[k]=data[i]; }
   int cnt=ArraySize(s);
   if(cnt==0) return(NA);
   if(cnt==1) return(s[0]);
   ArraySort(s);
   double idx=(cnt-1)*p/100.0;
   int lo=(int)MathFloor(idx); int hi=(int)MathMin(lo+1,cnt-1); double frac=idx-lo;
   return(s[lo]+frac*(s[hi]-s[lo]));
  }
void KMeans1D(const double &data[],const int K,const double &factors[],const int fromIdx,
              const int maxIter,double &tfOut,double &pmOut)
  {
   double centroids[3];
   centroids[0]=PercentileLinear(data,K,25);
   centroids[1]=PercentileLinear(data,K,50);
   centroids[2]=PercentileLinear(data,K,75);
   int label[]; ArrayResize(label,K);
   for(int j=0;j<K;j++) label[j]=0;
   for(int it=0;it<=maxIter;it++)
     {
      int cnt[3]={0,0,0}; double sum[3]={0.0,0.0,0.0};
      for(int j=0;j<K;j++)
        {
         double best=NA; int idx=0;
         for(int t=0;t<3;t++)
           {
            double dd=IsNa(centroids[t])?NA:MathAbs(data[j]-centroids[t]);
            if(IsNa(dd)) continue;
            if(IsNa(best)||dd<best){ best=dd; idx=t; }
           }
         label[j]=idx; cnt[idx]++; sum[idx]+=data[j];
        }
      double newc[3];
      for(int t=0;t<3;t++) newc[t]=(cnt[t]>0)?sum[t]/cnt[t]:NA;
      bool same=true;
      for(int t=0;t<3;t++)
         if(IsNa(newc[t])||IsNa(centroids[t])||newc[t]!=centroids[t]) same=false;
      for(int t=0;t<3;t++) centroids[t]=newc[t];
      if(same) break;
     }
   double sf=0.0,sp=0.0; int cf=0;
   for(int j=0;j<K;j++) if(label[j]==fromIdx){ sf+=factors[j]; sp+=data[j]; cf++; }
   tfOut=(cf>0)?sf/cf:NA;
   pmOut=(cf>0)?sp/cf:NA;
  }
void CalcSuperTrendOs(const double &h[],const double &l[],const double &c[],const int n,int &osArr[])
  {
   ArrayResize(osArr,n);
   int nf=(int)((InpMaxMult-InpMinMult)/InpStep);
   int K=nf+1;
   double factors[]; ArrayResize(factors,K);
   for(int i=0;i<K;i++) factors[i]=InpMinMult+i*InpStep;

   double atr[]; CalcATR(h,l,c,n,InpLength,atr);
   double absdiff[]; ArrayResize(absdiff,n); absdiff[0]=NA;
   for(int i=1;i<n;i++) absdiff[i]=MathAbs(c[i]-c[i-1]);
   double den[]; CalcEMA(absdiff,n,(int)InpPerfAlpha,den);

   double hl2_0=(h[0]+l[0])/2.0;
   double upper[],lower[],output[],perf[]; int trend[];
   ArrayResize(upper,K); ArrayResize(lower,K); ArrayResize(output,K); ArrayResize(perf,K); ArrayResize(trend,K);
   for(int k=0;k<K;k++){ upper[k]=hl2_0; lower[k]=hl2_0; trend[k]=0; output[k]=NA; perf[k]=0.0; }
   double alpha=2.0/(InpPerfAlpha+1.0);
   double f_upper=hl2_0, f_lower=hl2_0, target_factor=NA;
   int os=0;

   for(int i=0;i<n;i++)
     {
      double c1=(i>0)?c[i-1]:NA;
      double ai=atr[i];
      double hl2i=(h[i]+l[i])/2.0;
      for(int k=0;k<K;k++)
        {
         double up=IsNa(ai)?NA:hl2i+ai*factors[k];
         double dn=IsNa(ai)?NA:hl2i-ai*factors[k];
         if(!IsNa(upper[k])&&c[i]>upper[k]) trend[k]=1;
         else if(!IsNa(lower[k])&&c[i]<lower[k]) trend[k]=0;
         upper[k]=(!IsNa(c1)&&!IsNa(upper[k])&&c1<upper[k])?PMin(up,upper[k]):up;
         lower[k]=(!IsNa(c1)&&!IsNa(lower[k])&&c1>lower[k])?PMax(dn,lower[k]):dn;
         double diff=nz(sgn((!IsNa(c1)&&!IsNa(output[k]))?c1-output[k]:NA));
         double chg=nz(!IsNa(c1)?c[i]-c1:NA);
         perf[k]+=alpha*(chg*diff-perf[k]);
         output[k]=(trend[k]==1)?lower[k]:upper[k];
        }
      // full-history k-means window (backtest used maxData=1e9)
      double data[]; ArrayResize(data,K);
      for(int k=0;k<K;k++) data[k]=perf[k];
      double tf,pm; KMeans1D(data,K,factors,InpFromCluster,InpMaxIter,tf,pm);
      target_factor=nz(tf,target_factor);
      double up=(!IsNa(ai)&&!IsNa(target_factor))?hl2i+ai*target_factor:NA;
      double dn=(!IsNa(ai)&&!IsNa(target_factor))?hl2i-ai*target_factor:NA;
      f_upper=(!IsNa(c1)&&!IsNa(f_upper)&&c1<f_upper)?PMin(up,f_upper):up;
      f_lower=(!IsNa(c1)&&!IsNa(f_lower)&&c1>f_lower)?PMax(dn,f_lower):dn;
      if(!IsNa(f_upper)&&c[i]>f_upper) os=1;
      else if(!IsNa(f_lower)&&c[i]<f_lower) os=0;
      osArr[i]=os;
     }
  }

//+------------------------------------------------------------------+
//| ATR(14) simple-mean with zero-padded warmup (existing EA caliber)|
//+------------------------------------------------------------------+
void FeatureAtrSma(const double &h[],const double &l[],const double &c[],
                   const int n,const int w,double &atr[])
  {
   ArrayResize(atr,n);
   double tr[]; ArrayResize(tr,n);
   for(int i=0;i<n;i++)
     {
      double pc=(i==0)?c[0]:c[i-1];
      tr[i]=MathMax(h[i]-l[i],MathMax(MathAbs(h[i]-pc),MathAbs(l[i]-pc)));
     }
   for(int i=0;i<n;i++)
     {
      double sum=0.0;
      int from=i-w+1; if(from<0) from=0;
      for(int j=from;j<=i;j++) sum+=tr[j];
      atr[i]=sum/w;
     }
  }

//+------------------------------------------------------------------+
//| Load closed bars (oldest-first, forming bar excluded)            |
//+------------------------------------------------------------------+
bool LoadClosedBars(double &o[],double &h[],double &l[],double &c[],datetime &t[],int &n)
  {
   MqlRates rates[];
   ArraySetAsSeries(rates,true);
   int copied=CopyRates(_Symbol,InpTimeframe,1,InpHistoryBars,rates);
   if(copied<InpMinBars) return(false);
   n=copied;
   ArrayResize(o,n); ArrayResize(h,n); ArrayResize(l,n); ArrayResize(c,n); ArrayResize(t,n);
   for(int i=0;i<n;i++)
     {
      int j=n-1-i;   // rates[] is series (newest first) -> flip to oldest-first
      o[i]=rates[j].open; h[i]=rates[j].high; l[i]=rates[j].low;
      c[i]=rates[j].close; t[i]=rates[j].time;
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| Recipe decision on the last closed bar.                          |
//|   event: +1 up onset, -1 down onset, 0 none                      |
//|   target: desired position after this bar (-1/0/+1)              |
//+------------------------------------------------------------------+
int Decide(const double &h[],const double &l[],const double &c[],const int n,
           int &eventOut,double &erOut,int &stDirOut)
  {
   eventOut=0; erOut=0.0; stDirOut=0;
   int i=n-1;

   //--- slow QQE up/dn validity at bars i and i-1
   double ptl[],srsiP[],stl[],srsiS[];
   CalculateQQE(c,n,InpRsiLenP,InpSmP,InpFacP,ptl,srsiP);
   CalculateQQE(c,n,InpRsiLenS,InpSmS,InpFacS,stl,srsiS);
   double ptl50[]; ArrayResize(ptl50,n);
   for(int k=0;k<n;k++) ptl50[k]=IsNa(ptl[k])?NA:ptl[k]-50.0;
   double basis[],dev[];
   CalcSMA(ptl50,n,InpBollLen,basis);
   CalcStdev(ptl50,n,InpBollLen,dev);

   bool upNow=false,dnNow=false,upPrev=false,dnPrev=false;
   for(int k=i-1;k<=i;k++)
     {
      if(k<0) continue;
      bool okB=!IsNa(basis[k])&&!IsNa(dev[k]);
      double bU=okB?basis[k]+InpBollMult*dev[k]:NA;
      double bL=okB?basis[k]-InpBollMult*dev[k]:NA;
      double secRSI=IsNa(srsiS[k])?NA:srsiS[k]-50.0;
      double primRSI=IsNa(srsiP[k])?NA:srsiP[k]-50.0;
      bool u=(!IsNa(secRSI)&&!IsNa(primRSI)&&!IsNa(bU)&&secRSI>InpThrS&&primRSI>bU);
      bool d=(!IsNa(secRSI)&&!IsNa(primRSI)&&!IsNa(bL)&&secRSI<-InpThrS&&primRSI<bL);
      if(k==i){ upNow=u; dnNow=d; } else { upPrev=u; dnPrev=d; }
     }
   int event=0;
   if(upNow&&!upPrev) event=1;
   if(dnNow&&!dnPrev) event=-1;   // down wins if both fire (cannot co-occur: secRSI sign)
   eventOut=event;

   //--- SuperTrend AI direction
   int osArr[]; CalcSuperTrendOs(h,l,c,n,osArr);
   int stDir=(osArr[i]==1)?1:-1;
   stDirOut=stDir;

   //--- efficiency ratio ER (causal, bars i-InpErLen..i)
   double er=NA;
   if(i>=InpErLen)
     {
      double path=0.0;
      for(int k=i-InpErLen+1;k<=i;k++) path+=MathAbs(c[k]/c[k-1]-1.0);
      er=(path>0.0)?MathAbs(c[i]-c[i-InpErLen])/path:0.0;
     }
   erOut=nz(er,0.0);

   if(event==0) return(g_lastTarget);   // no onset -> hold whatever we target

   bool gate=(stDir==event)&&(!IsNa(er))&&er>=InpErThreshold;
   if(event==g_lastTarget) return(g_lastTarget);   // already in that direction
   if(gate) return(event);              // open or reverse
   return(0);                           // onset fails gate -> flat (opposite onset exits)
  }

//+------------------------------------------------------------------+
//| Position helpers (conventions from AlphaMaster_XAUUSD.mq5)       |
//+------------------------------------------------------------------+
int CurrentDirection()
  {
   double net=0.0; bool hasBuy=false,hasSell=false;
   for(int i=PositionsTotal()-1;i>=0;i--)
     {
      ulong tk=PositionGetTicket(i);
      if(tk==0) continue;
      if(PositionGetString(POSITION_SYMBOL)!=_Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC)!=InpMagicNumber) continue;
      long typ=PositionGetInteger(POSITION_TYPE);
      double vol=PositionGetDouble(POSITION_VOLUME);
      if(typ==POSITION_TYPE_BUY){ hasBuy=true; net+=vol; }
      if(typ==POSITION_TYPE_SELL){ hasSell=true; net-=vol; }
     }
   if(hasBuy&&hasSell) return(9);
   if(net>0.0) return(1);
   if(net<0.0) return(-1);
   return(0);
  }
void CloseAllMine()
  {
   for(int i=PositionsTotal()-1;i>=0;i--)
     {
      ulong tk=PositionGetTicket(i);
      if(tk==0) continue;
      if(PositionGetString(POSITION_SYMBOL)!=_Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC)!=InpMagicNumber) continue;
      trade.PositionClose(tk);
     }
  }
bool AttachStopLoss(const ulong ticket,const int side,const double atrVal,
                    const double entry,const double k,double &outSl)
  {
   if(atrVal<=0.0||k<=0.0) return(false);
   double sl=(side==1)?entry-k*atrVal:entry+k*atrVal;
   outSl=sl;
   double tp=0.0;
   if(PositionSelectByTicket(ticket)) tp=PositionGetDouble(POSITION_TP);
   if(!trade.PositionModify(ticket,sl,tp))
     {
      PrintFormat("[ST_QQE] sl_attach FAILED ticket=%llu side=%d entry=%.3f atr=%.5f k=%.1f sl=%.5f retcode=%u",
                  ticket,side,entry,atrVal,k,sl,trade.ResultRetcode());
      return(false);
     }
   PrintFormat("[ST_QQE] sl_attach ticket=%llu side=%d entry=%.3f atr=%.5f k=%.1f sl=%.5f",
               ticket,side,entry,atrVal,k,sl);
   return(true);
  }

//--- risk-% sizing against the disaster stop distance; falls back to fixed lot
double ComputeLots(const double atrVal)
  {
   double lots=InpLot;
   if(InpRiskSizing && InpStopAtrMult>0.0 && atrVal>0.0)
     {
      double riskMoney=AccountInfoDouble(ACCOUNT_EQUITY)*InpRiskPercent/100.0;
      double tickValue=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_VALUE);
      double tickSize=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE);
      if(tickValue>0.0 && tickSize>0.0)
        {
         double moneyPerLot=InpStopAtrMult*atrVal/tickSize*tickValue;
         if(moneyPerLot>0.0) lots=riskMoney/moneyPerLot;
        }
     }
   double minLot=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN);
   double maxLot=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MAX);
   double step=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
   if(step>0.0) lots=MathFloor(lots/step)*step;
   if(lots<minLot) lots=minLot;
   if(lots>maxLot) lots=maxLot;
   return(lots);
  }

void OpenDir(const int dir,const double atrLast)
  {
   double lots=ComputeLots(atrLast);
   bool ok=false;
   if(dir==1)      ok=trade.Buy(lots,_Symbol,0.0,0.0,0.0,InpComment);
   else if(dir==-1) ok=trade.Sell(lots,_Symbol,0.0,0.0,0.0,InpComment);
   if(!ok)
     {
      PrintFormat("[ST_QQE] open FAILED dir=%d lots=%.2f retcode=%u",dir,lots,trade.ResultRetcode());
      return;
     }
   ulong tk=trade.ResultOrder();
   if(tk==0) return;
   if(PositionSelectByTicket(tk))
     {
      double entry=PositionGetDouble(POSITION_PRICE_OPEN);
      long typ=PositionGetInteger(POSITION_TYPE);
      int side=(typ==POSITION_TYPE_BUY)?1:-1;
      double sl=0.0;
      AttachStopLoss(tk,side,atrLast,entry,InpStopAtrMult,sl);
     }
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(InpDeviation);
   trade.SetTypeFillingBySymbol(_Symbol);
   if(InpMinMult>InpMaxMult || InpStep<=0)
     {
      Print("[ST_QQE] bad SuperTrend factor range");
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(_Symbol!=InpRequiredSymbol)
     {
      g_symbol_ok=false;
      g_status=StringFormat("禁用：配方仅回测于 %s，当前图表是 %s",InpRequiredSymbol,_Symbol);
      Comment(g_status);
      Print("[ST_QQE] ",g_status," —— 不计算不下单");
      return(INIT_SUCCEEDED);
     }
   Print("[ST_QQE] ready. symbol=",_Symbol," bars=",Bars(_Symbol,InpTimeframe),
         " trade=",InpTradeEnabled," riskSizing=",InpRiskSizing);
   EvaluateAndReconcile(true);
   return(INIT_SUCCEEDED);
  }
void OnDeinit(const int reason){ Comment(""); }
void OnTick(){ EvaluateAndReconcile(false); }

//+------------------------------------------------------------------+
void EvaluateAndReconcile(const bool force)
  {
   if(!g_symbol_ok) return;
   datetime barTime=iTime(_Symbol,InpTimeframe,1);
   if(barTime==0) return;
   if(!force && g_lastSignalBar==barTime) return;

   double o[],h[],l[],c[]; datetime t[]; int n=0;
   if(!LoadClosedBars(o,h,l,c,t,n))
     {
      g_status="waiting for bars";
      ShowPanel();
      return;
     }
   g_lastSignalBar=barTime;

   // sync state with reality BEFORE deciding: if SL/manual closed us between
   // bars, "no onset" must mean stay flat, not re-open the stale target
   if(InpTradeEnabled)
     {
      int cur0=CurrentDirection();
      if(cur0!=9) g_lastTarget=cur0;
     }

   int event=0,stDir=0; double er=0.0;
   int target=Decide(h,l,c,n,event,er,stDir);
   g_lastTarget=target; g_lastEvent=event; g_lastEr=er; g_lastStDir=stDir;

   double atr[]; FeatureAtrSma(h,l,c,n,14,atr);
   double atrLast=atr[n-1];

   PrintFormat("[ST_QQE] bar=%s n=%d event=%d stDir=%d er=%.3f target=%d atr=%.5f",
               TimeToString(t[n-1]),n,event,stDir,er,target,atrLast);

   if(!InpTradeEnabled)
     {
      g_status="signal only";
      ShowPanel();
      return;
     }

   int cur=CurrentDirection();
   if(cur==9)
     {
      CloseAllMine();
      g_status="cleaned mixed hedge";
      ShowPanel();
      return;
     }
   // sync state with reality: if SL/manual closed us between bars, stay flat
   // until a fresh onset instead of blindly re-opening the stale target
   if(cur==target) g_status="hold";
   else if(target==0){ CloseAllMine(); g_status="flat (exit / gate rejected)"; }
   else
     {
      if(cur!=0) CloseAllMine();
      OpenDir(target,atrLast);
      g_status=(target==1?"long":"short");
     }

   //--- SL repair pass (same convention as the existing EA)
   for(int i=PositionsTotal()-1;i>=0;i--)
     {
      ulong tk=PositionGetTicket(i);
      if(tk==0) continue;
      if(PositionGetString(POSITION_SYMBOL)!=_Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC)!=InpMagicNumber) continue;
      long typ=PositionGetInteger(POSITION_TYPE);
      if(typ!=POSITION_TYPE_BUY&&typ!=POSITION_TYPE_SELL) continue;
      int side=(typ==POSITION_TYPE_BUY)?1:-1;
      double entry=PositionGetDouble(POSITION_PRICE_OPEN);
      double sl=0.0;
      AttachStopLoss(tk,side,atrLast,entry,InpStopAtrMult,sl);
     }
   ShowPanel();
  }

void ShowPanel()
  {
   Comment(StringFormat("ST+QQE Recipe %s\nQQE r%d/s%d  ER%d>=%.2f  STdir=%d\nevent=%d target=%d er=%.3f\n%s",
                        EnumToString(InpTimeframe),InpRsiLenP,InpSmP,InpErLen,InpErThreshold,
                        g_lastStDir,g_lastEvent,g_lastTarget,g_lastEr,g_status));
  }
//+------------------------------------------------------------------+