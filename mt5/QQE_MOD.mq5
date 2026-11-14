//+------------------------------------------------------------------+
//| QQE MOD.mq5                                                      |
//| Faithful MT5 port of the TradingView "QQE MOD" Pine v6 indicator |
//| (overlay=false -> separate window). Standalone, no AlphaMaster.  |
//+------------------------------------------------------------------+
// Pine -> MQL5 fidelity notes (validated against an independent Python
// oracle on real XAUUSD_M30 data):
//   ta.ema  : alpha=2/(len+1), seeded with the FIRST VALID source value
//   ta.rsi  : Wilder RSI; rma seeded with SMA of first len values
//   ta.stdev: population (biased=true, divide by len)
//   primaryRSI / secondaryRSI == the returned smoothedRsi (EMA of RSI)
//   Bands use previous-bar values (state machine), exactly as Pine [1].
// Cosmetic-only differences vs TradingView: plot transparency is not
// supported by MQL5 buffers (grey/cyan/magenta drawn solid); the dotted
// zero line is reproduced with a STYLE_DOT line buffer at 0.
//+------------------------------------------------------------------+
#property copyright "Ported for personal use"
#property version   "1.00"
#property indicator_separate_window
#property indicator_buffers 5
#property indicator_plots   5

//--- plot 0: Secondary QQE Trend Line (white line, width 2)
#property indicator_label1  "Secondary QQE Trend Line"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrWhite
#property indicator_width1  2
//--- plot 1: Secondary RSI Histogram (grey columns)
#property indicator_label2  "Secondary RSI Histogram"
#property indicator_type2   DRAW_HISTOGRAM
#property indicator_color2  C'112,112,112'
//--- plot 2: QQE Up Signal (cyan columns)
#property indicator_label3  "QQE Up Signal"
#property indicator_type3   DRAW_HISTOGRAM
#property indicator_color3  C'0,195,255'
//--- plot 3: QQE Down Signal (magenta columns)
#property indicator_label4  "QQE Down Signal"
#property indicator_type4   DRAW_HISTOGRAM
#property indicator_color4  C'255,0,98'
//--- plot 4: Zero Line (white dotted)
#property indicator_label5  "Zero Line"
#property indicator_type5   DRAW_LINE
#property indicator_color5  clrWhite
#property indicator_style5  STYLE_DOT
#property indicator_width5  1

//--- inputs (defaults verbatim from the Pine script)
input group "Primary QQE Settings"
input int    InpRsiLenP = 6;     // RSI Length
input int    InpSmP     = 5;     // RSI Smoothing
input double InpFacP    = 3.0;   // QQE Factor
input double InpThrP    = 3.0;   // Threshold (unused by original plots)
input group "Secondary QQE Settings"
input int    InpRsiLenS = 6;     // RSI Length
input int    InpSmS     = 5;     // RSI Smoothing
input double InpFacS    = 1.61;  // QQE Factor
input double InpThrS    = 3.0;   // Threshold
input group "Bollinger Bands Settings"
input int    InpBollLen  = 50;   // Length
input double InpBollMult = 0.35; // Multiplier
input group "Alerts"
input bool   InpAlerts  = true;  // Alert on Primary RSI cross of 50

//--- indicator buffers
double BufSecTrend[];
double BufHist[];
double BufUp[];
double BufDn[];
double BufZero[];

#define NA EMPTY_VALUE
bool IsNa(const double v){ return(v==EMPTY_VALUE); }
double PMin(const double a,const double b){ return(IsNa(a)||IsNa(b))?NA:MathMin(a,b); }
double PMax(const double a,const double b){ return(IsNa(a)||IsNa(b))?NA:MathMax(a,b); }

//+------------------------------------------------------------------+
//| Simple moving average (na until a full valid window)             |
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
//| EMA: alpha=2/(len+1), first-valid-value seed (Pine ta.ema)       |
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
//| RMA / Wilder: alpha=1/len, SMA seed (Pine ta.rma)                |
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
//| Wilder RSI (Pine ta.rsi)                                         |
//+------------------------------------------------------------------+
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
//+------------------------------------------------------------------+
//| Population standard deviation (Pine ta.stdev, biased=true)       |
//+------------------------------------------------------------------+
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
//+------------------------------------------------------------------+
//| cross of series A=a[shift sa] and B=b[shift sb] at bar i         |
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
//| calculateQQE: returns trend line and smoothedRsi                 |
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
      bool lbc=CrossAt(lb,srsi,i,1,0);   // ta.cross(longBand[1], smoothedRsi)
      bool cs =CrossAt(srsi,sb,i,0,1);   // ta.cross(smoothedRsi, shortBand[1])
      if(cs) td[i]=1.0; else if(lbc) td[i]=-1.0; else td[i]=(i>0)?td[i-1]:NA;
      trend[i]=(td[i]==1.0)?lb[i]:sb[i];
     }
  }
//+------------------------------------------------------------------+
int OnInit()
  {
   SetIndexBuffer(0,BufSecTrend,INDICATOR_DATA);
   SetIndexBuffer(1,BufHist,INDICATOR_DATA);
   SetIndexBuffer(2,BufUp,INDICATOR_DATA);
   SetIndexBuffer(3,BufDn,INDICATOR_DATA);
   SetIndexBuffer(4,BufZero,INDICATOR_DATA);
   ArraySetAsSeries(BufSecTrend,false);
   ArraySetAsSeries(BufHist,false);
   ArraySetAsSeries(BufUp,false);
   ArraySetAsSeries(BufDn,false);
   ArraySetAsSeries(BufZero,false);
   for(int p=0;p<5;p++) PlotIndexSetDouble(p,PLOT_EMPTY_VALUE,EMPTY_VALUE);
   IndicatorSetString(INDICATOR_SHORTNAME,"QQE MOD");
   IndicatorSetInteger(INDICATOR_DIGITS,_Digits);
   return(INIT_SUCCEEDED);
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
   ArraySetAsSeries(time,false);
   int n=rates_total;

   double ptl[],srsiP[],stl[],srsiS[];
   CalculateQQE(close,n,InpRsiLenP,InpSmP,InpFacP,ptl,srsiP);
   CalculateQQE(close,n,InpRsiLenS,InpSmS,InpFacS,stl,srsiS);

   double ptl50[]; ArrayResize(ptl50,n);
   for(int i=0;i<n;i++) ptl50[i]=IsNa(ptl[i])?NA:ptl[i]-50.0;
   double basis[],dev[];
   CalcSMA(ptl50,n,InpBollLen,basis);
   CalcStdev(ptl50,n,InpBollLen,dev);

   for(int i=0;i<n;i++)
     {
      double bU=NA,bL=NA;
      if(!IsNa(basis[i])&&!IsNa(dev[i]))
        { bU=basis[i]+InpBollMult*dev[i]; bL=basis[i]-InpBollMult*dev[i]; }
      double secTrend=IsNa(stl[i])?NA:stl[i]-50.0;
      double secRSI  =IsNa(srsiS[i])?NA:srsiS[i]-50.0;
      double primRSI =IsNa(srsiP[i])?NA:srsiP[i]-50.0;

      BufSecTrend[i]=secTrend;
      BufZero[i]=0.0;
      BufHist[i]=(!IsNa(secRSI)&&(secRSI>InpThrS||secRSI<-InpThrS))?secRSI:NA;
      BufUp[i]  =(!IsNa(secRSI)&&!IsNa(primRSI)&&!IsNa(bU)&&secRSI>InpThrS&&primRSI>bU)?secRSI:NA;
      BufDn[i]  =(!IsNa(secRSI)&&!IsNa(primRSI)&&!IsNa(bL)&&secRSI<-InpThrS&&primRSI<bL)?secRSI:NA;
     }

   //--- alerts: primaryRSI (=smoothedRsi) cross of 50, once per new bar
   static datetime lastAlertTime=0;
   if(InpAlerts && time[n-1]!=lastAlertTime)
     {
      int i=n-1;
      if(!IsNa(srsiP[i])&&!IsNa(srsiP[i-1]))
        {
         if(srsiP[i-1]<=50.0 && srsiP[i]>50.0) Alert("QQE MOD: Primary RSI crossed above zero");
         if(srsiP[i-1]>=50.0 && srsiP[i]<50.0) Alert("QQE MOD: Primary RSI crossed below zero");
        }
      lastAlertTime=time[n-1];
     }
   return(rates_total);
  }
//+------------------------------------------------------------------+