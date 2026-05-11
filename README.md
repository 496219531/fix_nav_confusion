# fix_nav_confusion

基金净值异常修复项目，目标是修正 `UnitNAV`（单位净值）与 `AccuNAV`（累计净值）混淆、共享值误填、短期孤点、累计分红链冲突等问题，并输出修复后的逐行对比结果与异常汇总。

## 数据背景

核心关系：

- `AccuDividends = AccuNAV - UnitNAV`
- `Dividends = diff(AccuDividends)`

当前修复与验证主要围绕三类约束：

- 负分红不能出现，允许绝对容忍度 `tol`
- 分红不能过于频繁，默认 `20` 天内最多 `1` 次
- 净值绝对变动不能超过阈值

## 主要文件

- [fix_nav_confusion.py](/Users/hkk/Documents/净值监控/fix_nav_confusion.py)：主修复脚本
- [thresholds.json](/Users/hkk/Documents/净值监控/thresholds.json)：参数配置
- [CODEX_CONTEXT.md](/Users/hkk/Documents/净值监控/CODEX_CONTEXT.md)：项目上下文与最新运行状态
- [净值修复逻辑说明.md](/Users/hkk/Documents/净值监控/净值修复逻辑说明.md)：算法说明文档
- [CASES.md](/Users/hkk/Documents/净值监控/CASES.md)：关键案例说明
- [具体异常净值.xlsx](/Users/hkk/Documents/净值监控/具体异常净值.xlsx)：原始总表
- [按基金拆分](/Users/hkk/Documents/净值监控/按基金拆分)：单基金输入文件目录

## 运行环境

- Python `3.10+`
- 依赖见 [requirements.txt](/Users/hkk/Documents/净值监控/requirements.txt)

安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

## 当前算法概览

当前脚本已经包含这些处理层：

- 先按 `FundID + TradingDay` 去重
- 预处理：如果 `AccuNAV < UnitNAV`，先交换两列
- 用动态规划在多种候选修复路径里选整体最自洽的序列
- 对 `UnitNAV == AccuNAV` 的共享值段做上下文判断与纠正
- 用上下文累计分红平台识别短期误填和混用段
- 识别并剔除孤点块、分红尖刺块，再重跑验证

当前关键上下文规则包括：

- `surrounding_div_match`
- `future_dividend_backfill`
- `stable_dividend_revert`
- `shared_zero_after_positive`
- `repeated_nonzero_bridge`

## 目录结构

```text
净值监控/
├── fix_nav_confusion.py
├── thresholds.json
├── requirements.txt
├── README.md
├── CODEX_CONTEXT.md
├── 净值修复逻辑说明.md
├── 具体异常净值.xlsx
├── 按基金拆分/
├── outputs/
└── outputs_* /
```

- `outputs/`：当前全量运行结果
- `outputs_*`：关键单基金调试结果快照

## 运行方式

全量运行：

```bash
python3 fix_nav_confusion.py --input '按基金拆分' --output-dir 'outputs' --config 'thresholds.json'
```

全量运行并输出标准汇总：

```bash
python3 scripts/run_full_and_report.py
```

如果要先重跑全量再输出汇总：

```bash
python3 scripts/run_full_and_report.py --rerun
```

单基金调试：

```bash
python3 fix_nav_confusion.py --input '按基金拆分/393514.xlsx' --output-dir 'outputs_393514' --config 'thresholds.json'
```

如果你使用的是 Codex 桌面环境，也可以继续使用项目上下文文档里记录的运行时 Python 路径。

关键案例回归检查：

```bash
python3 scripts/check_regressions.py
```

这个脚本会单独重跑几只关键基金，并检查当前已确认的业务现象是否仍然成立。

当前全量基线保存在：

- [baselines/full_run_baseline.json](/Users/hkk/Documents/净值监控/baselines/full_run_baseline.json)

## 当前参数

以 [thresholds.json](/Users/hkk/Documents/净值监控/thresholds.json) 为准，当前默认值：

- `tol = 0.003`
- `dividend_window_days = 20`
- `nav_change_threshold = 0.2`
- `max_outlier_span = 10`
- `context_run_min_len = 2`
- `context_neighbor_max_gap = 12`
- `stable_dividend_run_days = 5`
- `dividend_spike_max_span = 5`

## 输出文件

全量输出在 [outputs](/Users/hkk/Documents/净值监控/outputs)：

- [具体异常净值_修正结果.xlsx](/Users/hkk/Documents/净值监控/outputs/具体异常净值_修正结果.xlsx)
- [修复汇总.csv](/Users/hkk/Documents/净值监控/outputs/修复汇总.csv)
- [修复失败明细.csv](/Users/hkk/Documents/净值监控/outputs/修复失败明细.csv)

其中 `具体异常净值_修正结果.xlsx` 主要包含：

- `compare_raw_fixed`：原始值与修正值同一行对比
- `repaired_data`：修复后的明细数据
- `repair_summary`：按基金汇总
- `issue_details`：剩余异常明细

## 最新全量结果

最近一次全量运行结果：

- `total_funds = 52`
- `success_funds = 41`
- `failed_funds = 11`
- `remaining_neg_dividend_rows = 5`
- `remaining_return_anomaly_rows = 19`
- `remaining_dividend_frequency_rows = 9`

当前仍失败的基金：

- `942574`
- `394162`
- `942775`
- `943386`
- `1648828`
- `943458`
- `404802`
- `494282`
- `944765`
- `944791`
- `1550846`

## 备注

- 目录中的 `.~*.xlsx` 与 `~$*.xlsx` 属于 Excel 临时文件，脚本会自动忽略。
- 当前仓库保留了若干单基金调试输出目录，方便回看关键案例。
