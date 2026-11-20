# SuperTrend AI 指标说明（mt5/SuperTrend_AI.mq5）

TradingView "SuperTrend AI (Clustering) [LuxAlgo]"（Pine v5）核心算法的忠实 MT5 移植，主图指标。原版 (c) LuxAlgo，CC BY-NC-SA 4.0。已用独立 Python 参照实现在 XAUUSD M30 真实数据上逐 bar 校验。

**范围**：完整核心算法 = 逐因子 SuperTrend + 性能 EMA + 3 簇 k-means 因子选择（四分位播种、R-7 百分位）+ 自适应追踪止损 + 性能 AMA + 信号标记。**跳过**：原版右上角仪表盘表格、K线渐变色（MQL5 无对应物/重对象代码）。

## 安装

1. 将 `SuperTrend_AI.mq5` 放入 `MQL5/Indicators/`，MetaEditor 编译（F7）
2. 插入 → 指标 → 自定义 → SuperTrend AI，挂主图

## 显示内容（4 条绘图 + 图表对象）

| 元素 | 样式 | 含义 |
|---|---|---|
| Trailing Stop | 青色=多头趋势 / 红色=空头趋势，翻转处断线 | 自适应追踪止损线（k-means 选出的因子） |
| Trailing Stop AMA | 同色细线，价格穿越处断线 | 按性能指数平滑的止损线 |
| Up Arrow | 箭头码 233，颜色可在"颜色"页改 | 趋势翻多信号 |
| Down Arrow | 箭头码 234，颜色可在"颜色"页改 | 趋势翻空信号 |
| 数字+横线 | OBJ_TEXT/OBJ_TREND 对象，前缀 STAI_ | 信号强度标注（见下） |

**信号定义**：osc（价格相对自适应带的位置）翻转、且当根 ts 与 perf_idx 有效时出箭头。翻多=青色箭头+数字在K线下方，翻空=红色箭头+数字在K线上方。

**数字含义**：int(perf_idx×10)，即 k-means 选中簇的平均性能 ÷ 平均绝对波动，可粗略读作"这次翻转的质量分"（越大越强，常见 0~9）。

**标记几何**（ATR 自适应，移出K线避免遮挡）：
- 箭头：low/high 外侧 0.5×ATR（InpOffsetATR 可调）
- 数字：2×偏移处；横线：数字内侧 0.25×偏移、半宽 0.15×bar秒数
- 卸载指标时全部 STAI_ 对象自动清理

## 参数表

| 输入 | 默认 | 说明 |
|---|---|---|
| ATR Length | 10 | SuperTrend 的 ATR 周期（Wilder） |
| Factor Range min / max | 1 / 5 | 候选带宽因子范围 |
| Step | 0.5 | 因子步进（共 9 个候选） |
| Performance Memory | 10 | 性能 EMA 记忆长度 |
| From Cluster | Best | k-means 取哪个簇（Best/Average/Worst） |
| Maximum Iteration Steps | 1000 | k-means 迭代上限 |
| Historical Bars Calculation | 10000 | **只对最近 N 根算 k-means**（见下方陷阱） |
| Show Signals / Show AMA | true / true | 显示开关 |
| Up/Down Number Color | teal / red | 数字与横线颜色 |
| Number Font / Size | Arial / 8 | 数字字体字号 |
| Number Bold / Underline | false / true | 加粗 / 横线 |
| Marker Offset (ATR x) | 0.5 | 标记离K线的 ATR 倍数 |

## ⚠ maxData 陷阱（重要）

`Historical Bars Calculation=10000` 是原版默认，意思是 **k-means 只对最近 10000 根 bar 计算**——M30 上约 208 天。更早的历史没有 perf_idx，**不出箭头**。这是屏幕显示优化，不是 bug：

- 看最近行情：默认即可
- 全历史研究/回测：必须调大（研究脚本用 10⁹）；配方 EA 内部对加载的全部历史计算，不受此限

## 数值口径（已对 Python oracle 校验）

- ta.atr = TR 的 Wilder RMA（SMA 播种）；ta.ema 首值播种
- 百分位 = R-7 线性插值；k-means 三簇、四分位播种
- 性能 EMA alpha=2/(perfAlpha+1)；perf_idx = max(簇均值,0)/den

## 与配方 EA 的关系

EA 用**默认参数**的 ST 趋势方向做入场确认（不读箭头，读内部 osc 状态）。图上核对方法：EA 开多时，ST 线应为青色（多头趋势）。

## 相关文件

- 源码：`mt5/SuperTrend_AI.mq5`
- 策略 EA：`mt5/ST_QQE_Recipe_EA.mq5`（说明见 ST_QQE_Recipe_EA.md）
- Python 参照：`tmp/ref_indicators.py` 的 `supertrend_ai()`（研究用，非交付件）