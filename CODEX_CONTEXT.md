# Codex Long-Term Context

## Project Goal
Repair fund NAV data quality issues in Excel files, mainly confusion between:
- `UnitNAV` (单位净值)
- `AccuNAV` (累计净值)

Core relationship:
- `AccuDividends = AccuNAV - UnitNAV`
- `Dividends = diff(AccuDividends)`

## Strong / Validation Rules
- Strong rule:
  - dividends must not be negative beyond tolerance
  - `AccuNAV - UnitNAV` must not be negative beyond tolerance
- Validation rules:
  - negative dividend remaining
  - dividend frequency anomaly (more than one positive dividend within `dividend_window_days`)
  - return anomaly

## Current Script
- Main script: `fix_nav_confusion.py`
- Config file: `thresholds.json`
- Default input path: `按基金拆分`
- Default output path: `outputs`

## Current Threshold Config
Use `thresholds.json` as the canonical parameter source.

Current values:
- `tol = 0.003`
- `dividend_window_days = 20`
- `nav_change_threshold = 0.2`
- `max_outlier_span = 10`
- `context_run_min_len = 2`
- `context_neighbor_max_gap = 12`
- `stable_dividend_run_days = 5`
- `dividend_spike_max_span = 5`

## Important Data Handling Decisions
- Always deduplicate by `FundID + TradingDay` before repair.
- If `AccuNAV < UnitNAV`, swap them first in preprocessing.
- Ignore Excel temp files like `.~*.xlsx` and `~$*.xlsx` when reading a folder input.
- Keep raw/fixed values on the same row in output (`compare_raw_fixed` sheet).
- Do not silently delete rows; mark outliers explicitly:
  - `exclude_as_outlier = True`
  - `exclusion_reason` populated

## Current Repair Layers
Current repair is no longer just simple swap detection. It includes:
- preprocessing swap for `AccuNAV < UnitNAV`
- candidate-path dynamic programming
- context dividend inference
- shared-value repair when `UnitNAV == AccuNAV`
- forced short spike revert
- generic outlier block exclusion
- rerun after outlier exclusion

## Key Context-Dividend Rules
Important context sources currently used:
- `surrounding_div_match`
- `future_dividend_backfill`
- `stable_dividend_revert`
- `shared_zero_after_positive`
- `repeated_nonzero_bridge`

Meaning:
- `surrounding_div_match`:
  use matching nearby dividend platforms to fill a conflicting middle run
- `future_dividend_backfill`:
  move a future dividend platform earlier if it removes a return anomaly
- `stable_dividend_revert`:
  when a short run spikes away from stable platforms before and after, revert it
- `shared_zero_after_positive`:
  if a short shared-value zero-dividend run appears right after a positive-dividend platform, inherit the previous positive dividend
- `repeated_nonzero_bridge`:
  if a shared zero-dividend run is bracketed by repeated same nonzero platforms, allow bridging back to that dividend regime

## Shared-Value Logic
Current shared-value logic focuses on rows where `UnitNAV == AccuNAV`.

Main principles:
- if zero-dividend shared rows are bracketed by the same nonzero dividend regime, they are likely mixed/shared values
- decide repair direction by continuity to surrounding anchors
- for some high-confidence cases, lock direction and force context repair
- for lower-confidence bridge cases, keep them as strong candidates but not always forced

## Outlier Logic
Two outlier paths exist:
1. Generic isolated jump block detection (`isolated_jump_block`)
2. Forced dividend spike revert block (`dividend_spike_revert_block`)

Typical use cases:
- `393514` around `2025-04-30` to `2025-05-04`
- `893822` around `2025-11-14` to `2025-11-20`

## Latest Full Run
Latest full run used:
```bash
/Users/hkk/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 fix_nav_confusion.py --input '按基金拆分' --output-dir 'outputs' --config 'thresholds.json'
```

Latest full-run summary:
- `total_funds = 52`
- `success_funds = 41`
- `failed_funds = 11`
- `remaining_neg_dividend_rows = 5`
- `remaining_return_anomaly_rows = 19`
- `remaining_dividend_frequency_rows = 9`

Current failed funds:
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

## Current Fund Notes
- `393514`: now passes after outlier handling
- `893822`: now passes after raising `max_outlier_span` to 10
- `943386`: mostly improved; currently only one negative dividend remains (`2022-01-28`)
- `943458`: the shared-value dates around `2025-01-14/15` are already being repaired; remaining issue is later return-anomaly continuity
- `404802`: still looks like two competing dividend chains; likely needs explicit latest-to-oldest chain selection

## Standard Run Commands
Full run:
```bash
/Users/hkk/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 fix_nav_confusion.py --input '按基金拆分' --output-dir 'outputs' --config 'thresholds.json'
```

Single-fund debug:
```bash
/Users/hkk/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 fix_nav_confusion.py --input '按基金拆分/393514.xlsx' --output-dir 'outputs_393514' --config 'thresholds.json'
```

## Output Files
- `outputs/具体异常净值_修正结果.xlsx`
  - `compare_raw_fixed`
  - `repaired_data`
  - `repair_summary`
  - `issue_details`
- `outputs/修复汇总.csv`
- `outputs/修复失败明细.csv`

## Collaboration Convention
When user asks for latest results:
1. rerun script first
2. then read output files
3. report full summary and specific fund status if requested
