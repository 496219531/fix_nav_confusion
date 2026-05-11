from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "fix_nav_confusion.py"
CONFIG = ROOT / "thresholds.json"
INPUT_DIR = ROOT / "按基金拆分"
TMP_ROOT = ROOT / ".regression_tmp"


@dataclass
class CaseResult:
    name: str
    passed: bool
    details: list[str]


def run_case(fund_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_path = INPUT_DIR / f"{fund_id}.xlsx"
    output_dir = TMP_ROOT / fund_id

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--config",
        str(CONFIG),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    summary_df = pd.read_csv(output_dir / "修复汇总.csv")
    compare_df = pd.read_excel(output_dir / "具体异常净值_修正结果.xlsx", sheet_name="compare_raw_fixed")
    compare_df["TradingDay"] = pd.to_datetime(compare_df["TradingDay"]).dt.strftime("%Y-%m-%d")
    return summary_df, compare_df


def summary_row(summary_df: pd.DataFrame, fund_id: str) -> pd.Series:
    matched = summary_df.loc[summary_df["FundID"].astype(str) == fund_id]
    if matched.empty:
        raise AssertionError(f"summary missing fund {fund_id}")
    return matched.iloc[0]


def require(condition: bool, message: str, details: list[str]) -> None:
    if not condition:
        details.append(message)


def check_393514() -> CaseResult:
    summary_df, compare_df = run_case("393514")
    row = summary_row(summary_df, "393514")
    details: list[str] = []

    require(not bool(row["repair_failed"]), "393514 应该修复成功", details)
    require(int(row["excluded_rows"]) == 2, "393514 应该剔除 2 行异常点", details)

    for day in ("2025-04-30", "2025-05-04"):
        day_row = compare_df.loc[compare_df["TradingDay"] == day]
        require(not day_row.empty, f"393514 缺少日期 {day}", details)
        if day_row.empty:
            continue
        require(bool(day_row.iloc[0]["exclude_as_outlier"]), f"393514 {day} 应标记为异常剔除", details)
        require(
            day_row.iloc[0]["exclusion_reason"] == "dividend_spike_revert_block",
            f"393514 {day} 应为 dividend_spike_revert_block",
            details,
        )

    return CaseResult("393514", not details, details)


def check_893822() -> CaseResult:
    summary_df, compare_df = run_case("893822")
    row = summary_row(summary_df, "893822")
    details: list[str] = []

    require(not bool(row["repair_failed"]), "893822 应该修复成功", details)
    require(int(row["excluded_rows"]) == 6, "893822 应该剔除 6 行异常块", details)

    expected_days = {"2025-11-14", "2025-11-16", "2025-11-17", "2025-11-18", "2025-11-19", "2025-11-20"}
    block = compare_df.loc[compare_df["TradingDay"].isin(expected_days)]
    require(len(block) == 6, "893822 应覆盖 6 个关键日期", details)
    if len(block) == 6:
        require(bool(block["exclude_as_outlier"].all()), "893822 关键日期都应标记为异常剔除", details)
        require(
            bool((block["exclusion_reason"] == "dividend_spike_revert_block").all()),
            "893822 关键日期都应为 dividend_spike_revert_block",
            details,
        )

    return CaseResult("893822", not details, details)


def check_942574() -> CaseResult:
    summary_df, compare_df = run_case("942574")
    row = summary_row(summary_df, "942574")
    details: list[str] = []

    require(bool(row["repair_failed"]), "942574 当前应仍为失败案例", details)
    require(int(row["neg_dividend_rows"]) == 2, "942574 当前应剩余 2 条负分红", details)

    for day in ("2025-06-27", "2025-06-29"):
        day_row = compare_df.loc[compare_df["TradingDay"] == day]
        require(not day_row.empty, f"942574 缺少日期 {day}", details)
        if day_row.empty:
            continue
        require(
            str(day_row.iloc[0]["repair_action"]).startswith("shared_zero_after_positive_"),
            f"942574 {day} 应触发 shared_zero_after_positive 规则",
            details,
        )
        require(
            float(day_row.iloc[0]["fixed_AccuNAV"]) > float(day_row.iloc[0]["fixed_UnitNAV"]),
            f"942574 {day} 修复后应满足 AccuNAV > UnitNAV",
            details,
        )

    return CaseResult("942574", not details, details)


def check_943386() -> CaseResult:
    summary_df, _ = run_case("943386")
    row = summary_row(summary_df, "943386")
    details: list[str] = []

    require(bool(row["repair_failed"]), "943386 当前应仍为失败案例", details)
    require(int(row["neg_dividend_rows"]) == 1, "943386 当前应只剩 1 条负分红", details)
    require(int(row["return_anomaly_rows"]) == 0, "943386 当前不应再剩收益率异常", details)
    require(
        row["failure_reason"] == "negative_dividend_remaining",
        "943386 当前失败原因应为 negative_dividend_remaining",
        details,
    )

    return CaseResult("943386", not details, details)


def check_943458() -> CaseResult:
    summary_df, compare_df = run_case("943458")
    row = summary_row(summary_df, "943458")
    details: list[str] = []

    require(bool(row["repair_failed"]), "943458 当前应仍为失败案例", details)
    require(int(row["return_anomaly_rows"]) == 3, "943458 当前应剩余 3 条收益率异常", details)

    expected_actions = {
        "2025-01-14": "repeated_nonzero_bridge_accu_as_accu",
        "2025-01-15": "prev_div_accu_as_accu",
    }
    for day, action in expected_actions.items():
        day_row = compare_df.loc[compare_df["TradingDay"] == day]
        require(not day_row.empty, f"943458 缺少日期 {day}", details)
        if day_row.empty:
            continue
        require(day_row.iloc[0]["repair_action"] == action, f"943458 {day} 应为 {action}", details)

    return CaseResult("943458", not details, details)


def main() -> None:
    checks = [
        check_393514,
        check_893822,
        check_942574,
        check_943386,
        check_943458,
    ]

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    results = [check() for check in checks]

    failed = [result for result in results if not result.passed]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}")
        for detail in result.details:
            print(f"  - {detail}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
