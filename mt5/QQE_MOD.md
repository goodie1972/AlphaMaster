# QQE MOD 指标说明（mt5/QQE_MOD.mq5）

TradingView "QQE MOD"（Pine v6）的忠实 MT5 移植，独立副图指标，不依赖 AlphaMaster 任何组件。已用独立 Python 参照实现在 XAUUSD M30 真实数据上逐 bar 校验。

## 安装

1. 将 `QQE_MOD.mq5` 放入 MT5 数据目录的 `MQL5/Indicators/`（或直接在 MetaEditor 中打开）
2. MetaEditor 编译（F7），无错误后回到 MT5
3. 插入 → 指标 → 自定义 → QQE MOD，挂在任意品种/周期（配方用法见下）

## 显示内容（5 条绘图）

| 绘图 | 样式 | 含义 |
|---|---|---|
| Secondary QQE Trend Line | 白色实线（宽2） | 副 QQE 趋势线（stl−50） |
| Secondary RSI Histogram | 灰色柱 | \|副RSI\| 超过阈值时的动量柱 |
| QQE Up Signal | 青色柱 C'0,195,255' | 多头动量信号 |
| QQE Down Signal | 品红柱 C'255,0,98' | 空头动量信号 |
| Zero Line | 白色点线 | 0 轴 |

## 信号定义（与原版逐字一致）

记 secRSI = 副平滑RSI−50，primRSI = 主平滑RSI−50，bU/bL = 主QQE趋势线−50 的布林上下轨（长度50、倍数0.35）：

- **Up 信号**：secRSI > 3.0 且 primRSI > bU
- **Down 信号**：secRSI < −3.0 且 primRSI < bL
- **灰色动量柱**：secRSI > 3.0 或 < −3.0（不要求主RSI过轨）
- **提醒**：主平滑RSI 上穿/下穿 50 时弹窗（可关）

信号"出现"（onset）= 当根有效、前一根无效。配方 EA 用的就是这个 onset。

## 参数表（默认 = Pine 原版）

| 输入 | 默认 | 说明 |
|---|---|---|
| Primary RSI Length | 6 | 主RSI长度 |
| Primary RSI Smoothing | 5 | 主RSI的EMA平滑 |
| Primary QQE Factor | 3.0 | 主QQE带宽因子 |
| Primary Threshold | 3.0 | 原版绘图未使用，保留兼容 |
| Secondary RSI Length | 6 | 副RSI长度 |
| Secondary RSI Smoothing | 5 | 副RSI平滑 |
| Secondary QQE Factor | 1.61 | 副QQE带宽因子 |
| Secondary Threshold | 3.0 | 动量阈值（信号判定用） |
| Bollinger Length | 50 | 主趋势线布林长度 |
| Bollinger Multiplier | 0.35 | 布林倍数 |
| Alerts | true | 主RSI穿50提醒 |

## 配方用法（ST_QQE_Recipe_EA 的触发器）

把 **主/副两条的 RSI Length 都改成 20、Smoothing 都改成 12**，其余默认。慢化后信号明显变稀（约 166 次/年），这是回测验证过的冠军触发器参数；EA 内部已内置同一套计算，挂这个指标只是为了肉眼核对信号。

## 数值口径（已对 Python oracle 校验）

- ta.ema：alpha=2/(len+1)，用**第一个有效值**播种
- ta.rsi：Wilder RSI，rma 用前 len 个值的 SMA 播种
- ta.stdev：**总体标准差**（除以 len，不是 len−1）
- 布林带用当根 basis/dev 值；QQE 长短带状态机与 Pine 的 [1] 引用逐字一致

## 已知差异（仅外观）

- MT5 缓冲不支持透明度：原版灰/青/品红的半透明效果在这里是实色
- 原版点状零线用 STYLE_DOT 线缓冲复刻

## 相关文件

- 源码：`mt5/QQE_MOD.mq5`
- 策略 EA：`mt5/ST_QQE_Recipe_EA.mq5`（说明见 ST_QQE_Recipe_EA.md）
- Python 参照：`tmp/ref_indicators.py` 的 `qqe_mod()`（研究用，非交付件）