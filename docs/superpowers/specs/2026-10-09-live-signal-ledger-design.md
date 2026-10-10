# 实盘信号台账与对照系统 — 设计规格

- 日期：2026-10-09
- 状态：待用户审阅
- 范围：新增实盘台账子系统（方案 A）；不改动训练/回测链路
- 关联决议：策略文件名带周期（另行实现，见 [[project-alphamaster-design-decisions]]）

## 1. 背景与问题

AlphaMaster 的 03 实时分析页在每根 K 线收盘后重算信号（方向/把握度），但记录只存在
`WatchTask.history` 的 `deque(maxlen=60)` 内存里（web/realtime_manager.py:46），重启即丢，
也没有任何「信号发出后行情是否兑现」的留痕。回测指标（Sharpe/MDD）是历史模拟，
实盘信号是前向断言，两者之间缺一本对账的账。

## 2. 目标

1. 每根收盘 bar 的信号全量落盘，重启不丢不重
2. 由台账衍生：胜率、盈亏比、实盘等效年化/Sortino/MDD、完整率
3. 同段离线重算「理论曲线」，与实盘曲线叠图；分叉即数据/计算链路异常告警
4. 明细可筛选、分页、导出 CSV
5. 成交执行对照（MT5 实填 vs 目标仓位）：本 spec 只定义数据接口（§7），不实现

## 3. 非目标

- 不自动回补停机段（决策⑤：台账=程序当时真实所见，断档如实呈现）
- 不改训练引擎、评分函数、回测页
- 不做数据库/新依赖

## 4. 访谈决议（已对齐，不再讨论）

| # | 决议 |
|---|---|
| 1 | 每根收盘 bar 全量记录（非仅转折） |
| 2 | 主口径 = 与回测相同的 `position=tanh(因子)` 连续仓位 + 换手成本；胜率等为同一账本的衍生视图 |
| 3 | 对照基准 = 台账同段同数据源离线重算的理论曲线 |
| 4 | 前端为独立「04 实盘台账」页 |
| 5 | 停机断档：接受，不补写，用「完整率」指标呈现 |
| 6 | 架构 = 方案 A：评估循环内嵌写 jsonl + 理论曲线懒算缓存 |

## 5. 架构与组件

### 5.1 文件布局

```
live_ledger/                         # 台账目录，加入 .gitignore
  {source}_{safe_symbol}_{tf}.jsonl  # 每 watch 一个账本（safe 化规则同 strategy_file.py）
  strategy_index.json                # {strategy_sha: {formula, strategy_file, vocab_version, first_seen}}
  {source}_{safe_symbol}_{tf}.theo.jsonl  # 理论曲线缓存（派生物，可删，按账本 mtime 失效）
web/live_ledger.py                   # 新模块：写入、读取、统计纯函数、理论重算
```

### 5.2 记录格式（账本每行一条 JSON）

```json
{
  "ts": 1762400000.0,          // 本地写入墙钟，仅审计
  "bar_ts": 1762396800,        // 信号 bar 开盘时间（源给），所有对齐/去重只用它
  "source": "okx", "symbol": "BTCUSDT", "timeframe": "H1",
  "strategy_sha": "a1b2c3",    // 公式数组短哈希（sha1 前 6 位）
  "factor_value": 0.83, "position": 0.68,
  "direction": "up", "strength": 0.68,
  "close": 67890.5             // 信号 bar 收盘价（算相邻收益必需）
}
```

`strategy_index.json` 首次见新 sha 时追加；跨机拷贝账本若索引缺失，理论曲线降级为
unavailable，实盘统计不受影响。

### 5.3 写入路径（零新线程）

- 钩子点：`RealtimeManager._evaluate_task` 的 `state == "ok"` 分支内，调用
  `live_ledger.append(task, result)`（web/realtime_manager.py 现约 :377）
- 去重：`WatchTask` 增字段 `last_ledger_bar_ts: int | None`；仅当本次 `bar_ts` 更大才写。
- 恢复：`add_watch`/启动加载 watch 时，从账本末行读回 `last_ledger_bar_ts`
  （读不到=空账本，None）。重启不丢不重由此保证。
- 每账本天然单写者（该 watch 的评估线程），追加模式整行写入。

### 5.4 理论曲线懒算

- 触发：`/api/ledger/summary|curve` 请求且缓存失效（账本 mtime 变化或无缓存）。
- 步骤：按台账时段 `fetch_bars(source, symbol, tf, 时长 + 最大算子回看冗余)` →
  用 sha 对应公式经 StackVM 整段向量重算因子序列 → 逐根 bar 理论 position=tanh(因子) →
  `theo_pnl_t = position_t × ret_{t→t+1} − turnover_t × cost_rate`；
  `cost_rate` = `web_settings.bt_commission_pct + bt_slippage_pct`（与 02 回测页同源）。
- 与账本按 `bar_ts` join；实盘有、理论无的 bar 记 `theo_missing`，不参与分叉统计。
- 结果写 `.theo.jsonl`；任何一步失败返回 `unavailable` 原因，不写坏缓存。

### 5.5 统计纯函数（由账本衍生）

- 逐 bar 实盘 pnl 与理论同式（用账本内 position）；|position| < `min_exposure()`
  （strategy_manager.live_signal）的 bar 从胜率/盈亏比分母中剔除。
  bar 间收益用账本相邻行的 `close` 计算：`ret = close_{next}/close − 1`，且要求
  `bar_ts_{next} − bar_ts == 周期秒数`；不相邻（中间断档）则该 bar 的 pnl 记 missing，
  不参与收益/曲线（完整率单独统计）。
- 胜率 = `position_t × ret_{t→t+1} > 0` 的占比（方向判定，不含成本）；
  盈亏比 = 含成本 pnl 的均值盈利 ÷ |均值亏损|。
- 实盘等效年化/Sortino/MDD：对累计 log pnl 序列计算，年化 bar 数沿用
  `estimate_periods_per_year`（model_core/backtest.py:33）。
- 完整率 = 账本记录数 ÷ 同段数据源收盘 bar 数（源不可用时返回 null，不是 0）。
- 多 strategy_sha 段：分段统计，API 默认返回最新 sha 段，可显式指定。

### 5.6 API

| 端点 | 参数 | 返回 |
|---|---|---|
| `GET /api/ledger/watches` | — | 账本文件清单（含已删 watch 的历史账） |
| `GET /api/ledger/summary` | `watch_id, days=30, sha?` | 统计卡数据 + `corrupt_lines` + 完整率 + 理论可用状态 |
| `GET /api/ledger/curve` | `watch_id, days=30, sha?` | `[{bar_ts, cum_live, cum_theo, theo_missing}]` 叠图序列 |
| `GET /api/ledger/records` | `watch_id, days?, direction?, limit, offset` | 明细分页 |
| `GET /api/ledger/export` | `watch_id, days, sha?` | CSV 下载（账本原始行 + 派生 pnl 列） |

### 5.7 前端 04 页

- 布局沿用现有三步页风格，新增「04 实盘台账」：
  watch 下拉（来自 `/api/ledger/watches`）→ 统计卡（完整率/胜率/盈亏比/Sortino/MDD/
  实盘 vs 理论相关性）→ 叠图（复用 02 页资金曲线组件）→ 明细表（分页、方向筛选、CSV 导出）。
- 空账本状态：引导文案「台账自添加 watch 起积累，服务重启不丢」，非白屏。
- 理论线不可用：角标说明原因，图仍画实盘单线。

## 6. 错误处理

| 故障 | 行为 |
|---|---|
| 写盘失败 | try/except + `logger.warning`，绝不影响信号评估；连续失败只告警一次 |
| 半行损坏 | 读侧逐行解析，坏行跳过并计入 `corrupt_lines`，API 如实回传 |
| sha 索引缺失 | 理论曲线 `unavailable:"公式索引缺失"`；实盘统计照常 |
| 源历史拉取失败 | 叠图退化为实盘单线 + 原因角标；不写坏缓存 |
| bar 对不齐 | 按 `bar_ts` join，缺口记 `theo_missing` 并从分叉统计排除 |
| watch 删除 | 账本文件保留（历史资产），04 页仍可查 |
| 时钟漂移 | 对齐/去重只用 `bar_ts`；`ts` 仅审计 |

总原则：台账是观测者，任何台账故障不得中断实时信号主链路。

## 7. 成交执行对照接口（只定义，不实现）

预留 `live_ledger/{source}_{safe_symbol}_{tf}.fills.jsonl`，每行：
`{ts, bar_ts, action, target_position, filled_volume, fill_price, slippage, fee, order_id}`。
MT5 接入后由 execution 层写入；统计层届时以「目标 vs 实际」替换纯信号 pnl。
本期内核与 UI 不出现该文件的读写。

## 8. 测试计划（tests/unit，pytest 骨架已有）

1. 读写往返：合成 task/result → append → 读回逐字段相等；混入坏行 → `corrupt_lines==1`
2. 去重/续写：同 bar_ts 重复评估 5 次只写 1 行；从末行恢复 `last_ledger_bar_ts` 后重启不重写旧 bar
3. 口径恒等式：同一合成 bar 序列「逐根实时评估」与「整段懒算」position 序列逐 bar 全等；
   注入单根差异后叠图能指认该 bar
4. 边界函数：胜率/盈亏比/完整率用 10 行手算样例钉死（空仓剔除、theo_missing 排除、首根无 ret）
5. sha 索引：新 sha 落盘、重复不重写、索引缺失 → API unavailable 不 500
6. E2E（手动）：起 Web 加 OKX BTCUSDT_H1 watch → 账本出文件 → 04 页出数 → 杀进程重启 → 续写不重

## 9. 实施顺序建议（供后续拆票）

1. `web/live_ledger.py` 纯函数核心（格式/读写/统计）+ 单测 1/4/5
2. realtime_manager 接入（append + 恢复 + 去重）+ 单测 2
3. 理论懒算 + 缓存 + API + 单测 3
4. 前端 04 页
5. E2E 验证 + `.gitignore`/文档收尾
