//+------------------------------------------------------------------+
//| SuperTrend_AI.mq5                                                |
//| Faithful MT5 port of "SuperTrend AI (Clustering) [LuxAlgo]"      |
//| Pine v5 (overlay=true -> main chart). Standalone, no AlphaMaster.|
//| Original (c) LuxAlgo, CC BY-NC-SA 4.0. Core algorithm ported.    |
//+------------------------------------------------------------------+
// Scope (user-approved): full core algorithm = per-factor SuperTrend +
// performance EMA, 3-cluster k-means factor selection (quartile-seeded,
// R-7 percentile), adaptive trailing stop, performance AMA, signal
// labels. SKIPPED: dashboard table and candle gradient coloring (no
// MQL5 equivalent / heavy object code).
// Fidelity (validated vs an independent Python oracle on XAUUSD_M30):
//   ta.atr = Wilder RMA of TR (SMA seed); ta.ema = first-value seed;
//   percentile_linear_interpolation = R-7; perf EMA alpha=2/(perfAlpha+1).
// Cosmetic-only: AMA/stop transparency drawn solid; runtime.error on
// minMult>maxMult -> Print + INIT_PARAMETERS_INCORRECT.
//+------------------------------------------------------------------+
#property copyright "Ported for personal use"
#property version   "1.00"
#property indicator_chart_window
#property indicator_buffers 6
#property indicator_plots   4

//--- plot 0: Trailing Stop (teal up / red down, gap on flip)
#property indicator_label1  "Trailing Stop"
#property indicator_type1   DRAW_COLOR_LINE
#property indicator_color1  clrRed,clrTeal
#property indicator_width1  2
//--- plot 1: Trailing Stop AMA (teal up / red down, gap on cross)
#property indicator_label2  "Trailing Stop AMA"
#property indicator_type2   DRAW_COLOR_LINE
#property indicator_color2  clrRed,clrTeal
#property indicator_width2  1
//--- plot 2: Up signal arrow (color editable in the Colors tab)
#property indicator_label3  "Up Arrow"
#property indicator_type3   DRAW_ARROW
#property indicator_color3  C'0,100,100'
#property indicator_width3  2
//--- plot 3: Down signal arrow (color editable in the Colors tab)
#property indicator_label4  "Down Arrow"
#property indicator_type4   DRAW_ARROW
#property indicator_color4  C'180,0,0'
#property indicator_width4  2

enum ENUM_FROM_CLUSTER
  {
   Worst   = 0,  // Worst
   Average = 1,  // Average
   Best    = 2   // Best
  };

input group "Settings"
input int    InpLength     = 10;    // ATR Length
input int    InpMinMult    = 1;     // Factor Range (min)
input int    InpMaxMult    = 5;     // Factor Range (max)
input double InpStep       = 0.5;   // Step
input double InpPerfAlpha  = 10;    // Performance Memory
input ENUM_FROM_CLUSTER InpFromCluster = Best; // From Cluster
input group "Optimization"
input int    InpMaxIter    = 1000;  // Maximum Iteration Steps
input int    InpMaxData    = 10000; // Historical Bars Calculation
input group "Style"
input bool   InpShowSignals= true;  // Show Signals
input bool   InpShowAMA    = true;  // Show AMA
input group "Signal Markers"
input color  InpTextUpColor  = clrTeal;  // Up Number Color
input color  InpTextDnColor  = clrRed;   // Down Number Color
input string InpSigFont      = "Arial";  // Number Font
input int    InpSigFontSize  = 8;        // Number Font Size
input bool   InpSigBold      = false;    // Number Bold
input bool   InpSigUnderline = true;     // Number Underline
input double InpOffsetATR    = 0.5;      // Marker Offset (ATR x)

double BufTS[];
double BufTSColor[];
double BufAMA[];
double BufAMAColor[];
double BufUpArrow[];
double BufDnArrow[];

//--- signal event collectors
int    sigBar[];
int    sigDir[];
int    sigText[];
double sigPrice[];
double sigLine[];

#define NA EMPTY_VALUE
#define SIG_PREFIX "STAI_"
bool IsNa(const double v){ return(v==EMPTY_VALUE); }
double nz(const double v,const double d=0.0){ return(IsNa(v)?d:v); }
double sgn(const double v){ if(IsNa(v)) return(NA); return(v>0?1.0:(v<0?-1.0:0.0)); }
double PMin(const double a,const double b){ return(IsNa(a)||IsNa(b))?NA:MathMin(a,b); }
double PMax(const double a,const double b){ return(IsNa(a)||IsNa(b))?NA:MathMax(a,b); }

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
//+------------------------------------------------------------------+
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
//+------------------------------------------------------------------+
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
//+------------------------------------------------------------------+
void CalcATR(const double &h[],const double &l[],const double &c[],const int n,const int len,double &out[])
  {
   double tr[]; ArrayResize(tr,n);
   tr[0]=h[0]-l[0];
   for(int i=1;i<n;i++)
      tr[i]=MathMax(MathMax(h[i]-l[i],MathAbs(h[i]-c[i-1])),MathAbs(l[i]-c[i-1]));
   CalcRMA(tr,n,len,out);
  }
//+------------------------------------------------------------------+
//| R-7 percentile (linear interpolation between closest ranks)      |
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
//+------------------------------------------------------------------+
//| 1-D k-means, 3 clusters, quartile seeds; returns target mean     |
//| factor (tf) and mean perf (pm) of the fromIdx cluster.           |
//+------------------------------------------------------------------+
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
//+------------------------------------------------------------------+
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
int OnInit()
  {
   if(InpMinMult>InpMaxMult)
     {
      Print("SuperTrend AI: Minimum factor is greater than maximum factor in the range");
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(InpStep<=0)
     {
      Print("SuperTrend AI: Step must be > 0");
      return(INIT_PARAMETERS_INCORRECT);
     }
   SetIndexBuffer(0,BufTS,INDICATOR_DATA);
   SetIndexBuffer(1,BufTSColor,INDICATOR_COLOR_INDEX);
   SetIndexBuffer(2,BufAMA,INDICATOR_DATA);
   SetIndexBuffer(3,BufAMAColor,INDICATOR_COLOR_INDEX);
   SetIndexBuffer(4,BufUpArrow,INDICATOR_DATA);
   SetIndexBuffer(5,BufDnArrow,INDICATOR_DATA);
   ArraySetAsSeries(BufTS,false);
   ArraySetAsSeries(BufTSColor,false);
   ArraySetAsSeries(BufAMA,false);
   ArraySetAsSeries(BufAMAColor,false);
   ArraySetAsSeries(BufUpArrow,false);
   ArraySetAsSeries(BufDnArrow,false);
   PlotIndexSetInteger(0,PLOT_COLOR_INDEXES,2);
   PlotIndexSetInteger(0,PLOT_LINE_COLOR,0,clrRed);
   PlotIndexSetInteger(0,PLOT_LINE_COLOR,1,clrTeal);
   PlotIndexSetInteger(1,PLOT_COLOR_INDEXES,2);
   PlotIndexSetInteger(1,PLOT_LINE_COLOR,0,clrRed);
   PlotIndexSetInteger(1,PLOT_LINE_COLOR,1,clrTeal);
   PlotIndexSetDouble(0,PLOT_EMPTY_VALUE,EMPTY_VALUE);
   PlotIndexSetDouble(1,PLOT_EMPTY_VALUE,EMPTY_VALUE);
   PlotIndexSetInteger(2,PLOT_ARROW,233);
   PlotIndexSetInteger(3,PLOT_ARROW,234);
   PlotIndexSetDouble(2,PLOT_EMPTY_VALUE,EMPTY_VALUE);
   PlotIndexSetDouble(3,PLOT_EMPTY_VALUE,EMPTY_VALUE);
   if(!InpShowAMA) PlotIndexSetInteger(1,PLOT_DRAW_TYPE,DRAW_NONE);
   if(!InpShowSignals)
     {
      PlotIndexSetInteger(2,PLOT_DRAW_TYPE,DRAW_NONE);
      PlotIndexSetInteger(3,PLOT_DRAW_TYPE,DRAW_NONE);
     }
   IndicatorSetString(INDICATOR_SHORTNAME,"SuperTrend AI");
   IndicatorSetInteger(INDICATOR_DIGITS,_Digits);
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   ObjectsDeleteAll(0,SIG_PREFIX);
   ChartRedraw(0);
  }
//+------------------------------------------------------------------+
void BuildSignals(const datetime &time[])
  {
   ObjectsDeleteAll(0,SIG_PREFIX);
   if(!InpShowSignals) return;
   int total=ArraySize(sigBar);
   int start=(int)MathMax(0,total-500);
   string font=InpSigBold?InpSigFont+" Bold":InpSigFont;
   int hw=(int)(0.35*PeriodSeconds());
   for(int s=start;s<total;s++)
     {
      int i=sigBar[s];
      bool up=(sigDir[s]>0);
      color textCol=up?InpTextUpColor:InpTextDnColor;
      string tname=SIG_PREFIX+"T"+IntegerToString(i);
      if(ObjectCreate(0,tname,OBJ_TEXT,0,time[i],sigPrice[s]))
        {
         ObjectSetString(0,tname,OBJPROP_TEXT,IntegerToString(sigText[s]));
         ObjectSetInteger(0,tname,OBJPROP_COLOR,textCol);
         ObjectSetInteger(0,tname,OBJPROP_ANCHOR,up?ANCHOR_UPPER:ANCHOR_LOWER);
         ObjectSetInteger(0,tname,OBJPROP_FONTSIZE,InpSigFontSize);
         ObjectSetString(0,tname,OBJPROP_FONT,font);
         ObjectSetInteger(0,tname,OBJPROP_SELECTABLE,false);
         ObjectSetInteger(0,tname,OBJPROP_HIDDEN,true);
        }
      if(InpSigUnderline)
        {
         string uname=SIG_PREFIX+"U"+IntegerToString(i);
         if(ObjectCreate(0,uname,OBJ_TREND,0,time[i]-hw,sigLine[s],time[i]+hw,sigLine[s]))
           {
            ObjectSetInteger(0,uname,OBJPROP_COLOR,textCol);
            ObjectSetInteger(0,uname,OBJPROP_WIDTH,1);
            ObjectSetInteger(0,uname,OBJPROP_STYLE,STYLE_SOLID);
            ObjectSetInteger(0,uname,OBJPROP_RAY_LEFT,false);
            ObjectSetInteger(0,uname,OBJPROP_RAY_RIGHT,false);
            ObjectSetInteger(0,uname,OBJPROP_SELECTABLE,false);
            ObjectSetInteger(0,uname,OBJPROP_HIDDEN,true);
           }
        }
     }
   ChartRedraw(0);
  }
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
   if(rates_total<2) return(0);
   ArraySetAsSeries(close,false);
   ArraySetAsSeries(high,false);
   ArraySetAsSeries(low,false);
   ArraySetAsSeries(time,false);
   int n=rates_total;

   int nf=(int)((InpMaxMult-InpMinMult)/InpStep);
   int K=nf+1;
   double factors[]; ArrayResize(factors,K);
   for(int i=0;i<K;i++) factors[i]=InpMinMult+i*InpStep;
   int fromIdx=(int)InpFromCluster;

   double atr[]; CalcATR(high,low,close,n,InpLength,atr);
   double absdiff[]; ArrayResize(absdiff,n); absdiff[0]=NA;
   for(int i=1;i<n;i++) absdiff[i]=MathAbs(close[i]-close[i-1]);
   double den[]; CalcEMA(absdiff,n,(int)InpPerfAlpha,den);

   double hl2_0=(high[0]+low[0])/2.0;
   double upper[],lower[],output[],perf[]; int trend[];
   ArrayResize(upper,K); ArrayResize(lower,K); ArrayResize(output,K); ArrayResize(perf,K); ArrayResize(trend,K);
   for(int k=0;k<K;k++){ upper[k]=hl2_0; lower[k]=hl2_0; trend[k]=0; output[k]=NA; perf[k]=0.0; }
   double alpha=2.0/(InpPerfAlpha+1.0);
   double f_upper=hl2_0, f_lower=hl2_0, perf_ama=NA, target_factor=NA, perf_idx=NA;
   int os=0;

   double tsArr[],amaArr[]; int osArr[];
   ArrayResize(tsArr,n); ArrayResize(amaArr,n); ArrayResize(osArr,n);
   ArrayResize(sigBar,0); ArrayResize(sigDir,0); ArrayResize(sigText,0); ArrayResize(sigPrice,0);

   int last_bar=n-1;
   for(int i=0;i<n;i++)
     {
      double c1=(i>0)?close[i-1]:NA;
      double ai=atr[i];
      double hl2i=(high[i]+low[i])/2.0;
      for(int k=0;k<K;k++)
        {
         double up=IsNa(ai)?NA:hl2i+ai*factors[k];
         double dn=IsNa(ai)?NA:hl2i-ai*factors[k];
         if(!IsNa(upper[k])&&close[i]>upper[k]) trend[k]=1;
         else if(!IsNa(lower[k])&&close[i]<lower[k]) trend[k]=0;
         upper[k]=(!IsNa(c1)&&!IsNa(upper[k])&&c1<upper[k])?PMin(up,upper[k]):up;
         lower[k]=(!IsNa(c1)&&!IsNa(lower[k])&&c1>lower[k])?PMax(dn,lower[k]):dn;
         double diff=nz(sgn((!IsNa(c1)&&!IsNa(output[k]))?c1-output[k]:NA));
         double chg=nz(!IsNa(c1)?close[i]-c1:NA);
         perf[k]+=alpha*(chg*diff-perf[k]);
         output[k]=(trend[k]==1)?lower[k]:upper[k];
        }
      if((last_bar-i)<=InpMaxData)
        {
         double data[]; ArrayResize(data,K);
         for(int k=0;k<K;k++) data[k]=perf[k];
         double tf,pm; KMeans1D(data,K,factors,fromIdx,InpMaxIter,tf,pm);
         target_factor=nz(tf,target_factor);
         perf_idx=(!IsNa(den[i])&&den[i]!=0.0)?MathMax(nz(pm,0.0),0.0)/den[i]:NA;
        }
      double up=(!IsNa(ai)&&!IsNa(target_factor))?hl2i+ai*target_factor:NA;
      double dn=(!IsNa(ai)&&!IsNa(target_factor))?hl2i-ai*target_factor:NA;
      f_upper=(!IsNa(c1)&&!IsNa(f_upper)&&c1<f_upper)?PMin(up,f_upper):up;
      f_lower=(!IsNa(c1)&&!IsNa(f_lower)&&c1>f_lower)?PMax(dn,f_lower):dn;
      if(!IsNa(f_upper)&&close[i]>f_upper) os=1;
      else if(!IsNa(f_lower)&&close[i]<f_lower) os=0;
      double ts=(os==1)?f_lower:f_upper;
      double ts1=(i>0)?tsArr[i-1]:NA;
      if(IsNa(ts1)&&!IsNa(ts)) perf_ama=ts;
      else
        {
         if(!IsNa(perf_ama)&&!IsNa(perf_idx)&&!IsNa(ts)) perf_ama=perf_ama+perf_idx*(ts-perf_ama);
         else perf_ama=NA;
        }
      tsArr[i]=ts; amaArr[i]=perf_ama; osArr[i]=os;

      int os1=(i>0)?osArr[i-1]:-999;
      BufTS[i]=(i>0 && os==os1)?ts:NA;
      BufTSColor[i]=(os==1)?1:0;
      bool cr=CrossAt(close,amaArr,i,0,0);
      BufAMA[i]=cr?NA:perf_ama;
      BufAMAColor[i]=(!IsNa(perf_ama)&&close[i]>perf_ama)?1:0;
      BufUpArrow[i]=EMPTY_VALUE;
      BufDnArrow[i]=EMPTY_VALUE;

      if(i>0 && !IsNa(ts) && !IsNa(perf_idx) && os!=os1)
        {
         int dir=(os>os1)?1:-1;
         double atrI=nz(atr[i],0.0);
         double off=InpOffsetATR*atrI;
         double base=(dir>0)?low[i]:high[i];
         double out=(dir>0)?-1.0:1.0;
         double pArrow=base+out*off;
         double pText =base+out*2.0*off;
         double pLine =pText-0.3*off;
         if(dir>0) BufUpArrow[i]=pArrow; else BufDnArrow[i]=pArrow;
         int sc=ArraySize(sigBar);
         ArrayResize(sigBar,sc+1); ArrayResize(sigDir,sc+1); ArrayResize(sigText,sc+1); ArrayResize(sigPrice,sc+1); ArrayResize(sigLine,sc+1);
         sigBar[sc]=i; sigDir[sc]=dir; sigText[sc]=(int)(perf_idx*10); sigPrice[sc]=pText; sigLine[sc]=pLine;
        }
     }

   static int lastSigBars=-1;
   if(rates_total!=lastSigBars)
     {
      BuildSignals(time);
      lastSigBars=rates_total;
     }
   return(rates_total);
  }
//+------------------------------------------------------------------+