from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "fix_nav_confusion.py"
CONFIG = ROOT / "thresholds.json"
DEFAULT_INPUT = ROOT / "按基金拆分"
DEFAULT_OUTPUT = ROOT / "outputs"
BASELINE_PATH = ROOT / "baselines" / "full_run_baseline.json"


def run_full() -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--input",
        str(DEFAULT_INPUT),
        "--output-dir",
        str(DEFAULT_OUTPUT),
        "--config",
        str(CONFIG),
    ]
    return subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)


def build_summary(summary_df: pd.DataFrame) -> dict:
    total_funds = int(summary_df["FundID"].nunique())
    failed_funds = int(summary_df["repair_failed"].sum())
    success_funds = total_funds - failed_funds
    return {
        "total_funds": total_funds,
        "success_funds": success_funds,
        "failed_funds": failed_funds,
        "remaining_neg_dividend_rows": int(summary_df["neg_dividend_rows"].sum()),
        "remaining_return_anomaly_rows": int(summary_df["return_anomaly_rows"].sum()),
        "remaining_dividend_frequency_rows": int(summary_df["dividend_frequency_rows"].sum()),
    }


def build_failed_map(summary_df: pd.DataFrame) -> dict[str, dict]:
    failed_df = summary_df.loc[summary_df["repair_failed"]].copy()
    failed_df["FundID"] = failed_df["FundID"].astype(str)
    result: dict[str, dict] = {}
    for _, row in failed_df.sort_values("FundID").iterrows():
        result[str(row["FundID"])] = {
            "neg_dividend_rows": int(row["neg_dividend_rows"]),
            "return_anomaly_rows": int(row["return_anomaly_rows"]),
            "dividend_frequency_rows": int(row["dividend_frequency_rows"]),
            "failure_reason": "" if pd.isna(row["failure_reason"]) else str(row["failure_reason"]),
        }
    return result


def load_baseline() -> dict | None:
    if not BASELINE_PATH.exists():
        return None
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def compare_with_baseline(current_summary: dict, current_failed: dict[str, dict], baseline: dict | None) -> list[str]:
    if baseline is None:
        return ["baseline_missing"]

    notes: list[str] = []
    baseline_summary = baseline.get("summary", {})
    baseline_failed = baseline.get("failed_funds", {})

    for key, value in current_summary.items():
        base_value = baseline_summary.get(key)
        if base_value != value:
            notes.append(f"summary_changed:{key}:{base_value}->{value}")

    current_failed_keys = set(current_failed)
    baseline_failed_keys = set(baseline_failed)

    for fund_id in sorted(current_failed_keys - baseline_failed_keys):
        notes.append(f"new_failed_fund:{fund_id}")
    for fund_id in sorted(baseline_failed_keys - current_failed_keys):
        notes.append(f"resolved_failed_fund:{fund_id}")

    for fund_id in sorted(current_failed_keys & baseline_failed_keys):
        current_info = current_failed[fund_id]
        baseline_info = baseline_failed[fund_id]
        for key in ("neg_dividend_rows", "return_anomaly_rows", "dividend_frequency_rows", "failure_reason"):
            if current_info.get(key) != baseline_info.get(key):
                notes.append(
                    f"failed_fund_changed:{fund_id}:{key}:{baseline_info.get(key)}->{current_info.get(key)}"
                )

    return notes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true", help="rerun full repair before reading outputs")
    args = parser.parse_args()

    if args.rerun:
        completed = run_full()
        print(completed.stdout.strip())
        print("")

    summary_df = pd.read_csv(DEFAULT_OUTPUT / "修复汇总.csv")
    current_summary = build_summary(summary_df)
    current_failed = build_failed_map(summary_df)
    baseline = load_baseline()
    baseline_notes = compare_with_baseline(current_summary, current_failed, baseline)

    print("")
    print("normalized_summary=" + json.dumps(current_summary, ensure_ascii=False))
    print("failed_funds_detail=" + json.dumps(current_failed, ensure_ascii=False, sort_keys=True))
    print("baseline_comparison=" + json.dumps(baseline_notes, ensure_ascii=False))


if __name__ == "__main__":
    main()
