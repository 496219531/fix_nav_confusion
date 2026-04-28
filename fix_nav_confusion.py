from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


STRONG_PENALTY = 1_000_000.0
NEG_ACC_DIV_PENALTY = 50_000.0
RETURN_PENALTY = 2_000.0
FREQUENCY_PENALTY = 800.0
SWITCH_PENALTY = 0.6
EDIT_PENALTY = 5.0


@dataclass
class Config:
    tol: float = 0.003
    dividend_window_days: int = 20
    nav_change_threshold: float = 0.20
    max_outlier_span: int = 10
    context_run_min_len: int = 2
    context_neighbor_max_gap: int = 12
    stable_dividend_run_days: int = 5
    dividend_spike_max_span: int = 5


def normalize_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def deduplicate_fund_day(df: pd.DataFrame) -> pd.DataFrame:
    work = df.sort_values(["FundID", "TradingDay", "_row_id"]).copy()
    dedup = (
        work.groupby(["FundID", "TradingDay"], as_index=False)
        .agg(
            UnitNAV=("UnitNAV", "first"),
            AccuNAV=("AccuNAV", "first"),
            AccuDividends=("AccuDividends", "first"),
            Dividends=("Dividends", "first"),
            dividend_cleaned=("dividend_cleaned", "first"),
            year=("year", "first"),
            source_row_count=("_row_id", "size"),
            source_row_id_min=("_row_id", "min"),
        )
        .rename(columns={"source_row_id_min": "_row_id"})
    )
    return dedup


def preprocess_nav_order(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    swap_mask = (
        out["UnitNAV"].notna()
        & out["AccuNAV"].notna()
        & (out["AccuNAV"] < out["UnitNAV"])
    )
    out["preprocess_swapped"] = swap_mask
    out["preprocess_reason"] = np.where(swap_mask, "accunav_lt_unitnav", "")
    if swap_mask.any():
        unit_values = out.loc[swap_mask, "UnitNAV"].copy()
        accu_values = out.loc[swap_mask, "AccuNAV"].copy()
        out.loc[swap_mask, "UnitNAV"] = accu_values.values
        out.loc[swap_mask, "AccuNAV"] = unit_values.values
    return out


def calc_return_rate(current_value: float, previous_value: float) -> float:
    if pd.isna(current_value) or pd.isna(previous_value):
        return math.nan
    if abs(previous_value) < 1e-12:
        return math.nan
    return float(current_value / previous_value - 1.0)


def is_return_anomaly(current_value: float, previous_value: float, threshold: float) -> tuple[bool, float]:
    rate = calc_return_rate(current_value, previous_value)
    if pd.isna(rate):
        return False, rate
    return abs(rate) > threshold, rate


def is_total_return_anomaly(
    current_unit_nav: float,
    previous_unit_nav: float,
    dividend_amount: float,
    threshold: float,
) -> tuple[bool, float]:
    adjusted_current = current_unit_nav + (0.0 if pd.isna(dividend_amount) else dividend_amount)
    return is_return_anomaly(adjusted_current, previous_unit_nav, threshold)


def build_dividend_runs(dividends: pd.Series, tol: float) -> list[dict]:
    runs: list[dict] = []
    if len(dividends) == 0:
        return runs

    start = 0
    run_values = [float(dividends.iloc[0])]
    for i in range(1, len(dividends)):
        value = float(dividends.iloc[i])
        run_mean = float(np.mean(run_values))
        if abs(value - run_mean) <= tol:
            run_values.append(value)
            continue

        runs.append(
            {
                "start": start,
                "end": i - 1,
                "value": float(np.mean(run_values)),
                "length": i - start,
            }
        )
        start = i
        run_values = [value]

    runs.append(
        {
            "start": start,
            "end": len(dividends) - 1,
            "value": float(np.mean(run_values)),
            "length": len(dividends) - start,
        }
    )
    return runs


def annotate_context_dividends(work: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = work.copy()
    out["raw_accu_div"] = out["AccuNAV"] - out["UnitNAV"]
    out["context_dividend"] = np.nan
    out["context_dividend_source"] = ""
    out["context_mode_hint"] = ""
    out["context_mode_locked"] = False
    out["force_context_repair"] = False

    runs = build_dividend_runs(out["raw_accu_div"], cfg.tol)
    if not runs:
        return out

    valid_run_ids = [i for i, run in enumerate(runs) if run["length"] >= cfg.context_run_min_len]
    if not valid_run_ids:
        valid_run_ids = list(range(len(runs)))

    isolated_positive_spike_ids: set[int] = set()
    for run_idx, run in enumerate(runs):
        if run_idx == 0 or run_idx == len(runs) - 1:
            continue
        if abs(run["value"]) <= cfg.tol:
            continue
        prev_valid = next((idx for idx in reversed(valid_run_ids) if idx < run_idx), None)
        next_valid = next((idx for idx in valid_run_ids if idx > run_idx), None)
        if prev_valid is None or next_valid is None:
            continue
        if runs[prev_valid]["length"] < cfg.stable_dividend_run_days or runs[next_valid]["length"] < cfg.stable_dividend_run_days:
            continue
        prev_div = runs[prev_valid]["value"]
        next_div = runs[next_valid]["value"]
        if abs(prev_div - next_div) > cfg.tol:
            continue
        if abs(prev_div) > cfg.tol:
            continue
        if run["length"] > cfg.dividend_spike_max_span:
            continue
        future_same_count = sum(
            1
            for idx in range(run_idx + 1, len(runs))
            if abs(runs[idx]["value"] - run["value"]) <= cfg.tol and abs(runs[idx]["value"]) > cfg.tol
        )
        if future_same_count < 2:
            isolated_positive_spike_ids.add(run_idx)

    def calc_shared_ratio(start: int, end: int) -> float:
        block = out.loc[start:end, ["UnitNAV", "AccuNAV"]]
        return float((block["UnitNAV"] - block["AccuNAV"]).abs().le(cfg.tol).mean())

    def boundary_score(current_value: float, reference_value: float) -> float:
        flag, rate = is_return_anomaly(current_value, reference_value, cfg.nav_change_threshold)
        if pd.isna(rate):
            return 0.0
        return abs(rate) + (10.0 if flag else 0.0)

    def choose_anchor_mode_hint(
        start: int,
        end: int,
        target_div: float,
        prev_anchor_run: dict | None,
        next_anchor_run: dict | None,
    ) -> str:
        start_shared = float(out.at[start, "UnitNAV"])
        end_shared = float(out.at[end, "UnitNAV"])

        candidates = {
            "unit_as_unit": {
                "start_unit": start_shared,
                "start_accu": start_shared + target_div,
                "end_unit": end_shared,
                "end_accu": end_shared + target_div,
            },
            "accu_as_accu": {
                "start_unit": start_shared - target_div,
                "start_accu": start_shared,
                "end_unit": end_shared - target_div,
                "end_accu": end_shared,
            },
        }

        scores = {}
        for mode, values in candidates.items():
            score = 0.0
            if prev_anchor_run is not None:
                prev_anchor = out.iloc[prev_anchor_run["end"]]
                score += boundary_score(values["start_unit"], float(prev_anchor["UnitNAV"]))
                score += boundary_score(values["start_accu"], float(prev_anchor["AccuNAV"]))
            if next_anchor_run is not None:
                next_anchor = out.iloc[next_anchor_run["start"]]
                score += boundary_score(float(next_anchor["UnitNAV"]), values["end_unit"])
                score += boundary_score(float(next_anchor["AccuNAV"]), values["end_accu"])
            scores[mode] = score

        return min(scores, key=scores.get)

    def assign_shared_value_mode_hint(start: int, end: int, prefer_closer_side: bool = False) -> None:
        if start <= 0 or end >= len(out) - 1:
            return
        prev_unit_ref = float(out.at[start - 1, "UnitNAV"])
        next_unit_ref = float(out.at[end + 1, "UnitNAV"])
        prev_accu_ref = float(out.at[start - 1, "AccuNAV"])
        next_accu_ref = float(out.at[end + 1, "AccuNAV"])

        for idx in range(start, end + 1):
            shared_value = float(out.at[idx, "UnitNAV"])
            if abs(shared_value - float(out.at[idx, "AccuNAV"])) > cfg.tol:
                continue
            unit_distance = min(abs(shared_value - prev_unit_ref), abs(shared_value - next_unit_ref))
            accu_distance = min(abs(shared_value - prev_accu_ref), abs(shared_value - next_accu_ref))
            if prefer_closer_side and accu_distance != unit_distance:
                if accu_distance < unit_distance:
                    out.at[idx, "context_mode_hint"] = "accu_as_accu"
                    out.at[idx, "context_mode_locked"] = True
                else:
                    out.at[idx, "context_mode_hint"] = "unit_as_unit"
                    out.at[idx, "context_mode_locked"] = True
            elif accu_distance + cfg.tol < unit_distance:
                out.at[idx, "context_mode_hint"] = "accu_as_accu"
                out.at[idx, "context_mode_locked"] = True
            elif unit_distance + cfg.tol < accu_distance:
                out.at[idx, "context_mode_hint"] = "unit_as_unit"
                out.at[idx, "context_mode_locked"] = True

    def assign_shared_plateau_mode_hints(
        start: int,
        end: int,
        target_div: float,
        prev_anchor_run: dict | None,
        next_anchor_run: dict | None,
        prefer_closer_side: bool = False,
    ) -> None:
        idx = start
        while idx <= end:
            is_shared = abs(float(out.at[idx, "UnitNAV"]) - float(out.at[idx, "AccuNAV"])) <= cfg.tol
            if not is_shared:
                idx += 1
                continue

            plateau_start = idx
            plateau_values = [float(out.at[idx, "UnitNAV"])]
            idx += 1
            while idx <= end:
                next_is_shared = abs(float(out.at[idx, "UnitNAV"]) - float(out.at[idx, "AccuNAV"])) <= cfg.tol
                if not next_is_shared:
                    break
                value = float(out.at[idx, "UnitNAV"])
                plateau_mean = float(np.mean(plateau_values))
                if abs(value - plateau_mean) > cfg.tol:
                    break
                plateau_values.append(value)
                idx += 1

            plateau_end = idx - 1
            if prev_anchor_run is not None or next_anchor_run is not None:
                anchor_mode_hint = choose_anchor_mode_hint(
                    plateau_start,
                    plateau_end,
                    target_div,
                    prev_anchor_run,
                    next_anchor_run,
                )
                out.loc[plateau_start:plateau_end, "context_mode_hint"] = anchor_mode_hint
                out.loc[plateau_start:plateau_end, "context_mode_locked"] = True
            else:
                assign_shared_value_mode_hint(
                    plateau_start,
                    plateau_end,
                    prefer_closer_side=prefer_closer_side,
                )

    for run_idx, run in enumerate(runs):
        prev_valid = next((idx for idx in reversed(valid_run_ids) if idx < run_idx), None)
        next_valid = next((idx for idx in valid_run_ids if idx > run_idx), None)
        prev_nonzero_any = next(
            (
                idx
                for idx in range(run_idx - 1, -1, -1)
                if abs(runs[idx]["value"]) > cfg.tol and idx not in isolated_positive_spike_ids
            ),
            None,
        )
        next_nonzero_any = next(
            (
                idx
                for idx in range(run_idx + 1, len(runs))
                if abs(runs[idx]["value"]) > cfg.tol and idx not in isolated_positive_spike_ids
            ),
            None,
        )
        prev_div = runs[prev_valid]["value"] if prev_valid is not None else math.nan
        next_div = runs[next_valid]["value"] if next_valid is not None else math.nan
        prev_gap = run["start"] - runs[prev_valid]["end"] if prev_valid is not None else None
        next_gap = runs[next_valid]["start"] - run["end"] if next_valid is not None else None
        target_div = math.nan
        source = ""
        mode_hint = ""
        shared_ratio = calc_shared_ratio(run["start"], run["end"])
        force_context_run = False

        repeated_bridge_value = math.nan
        if (
            abs(run["value"]) <= cfg.tol
            and shared_ratio >= 0.8
            and prev_nonzero_any is not None
            and next_nonzero_any is not None
            and abs(runs[prev_nonzero_any]["value"] - runs[next_nonzero_any]["value"]) <= cfg.tol
        ):
            repeated_bridge_value = 0.5 * (runs[prev_nonzero_any]["value"] + runs[next_nonzero_any]["value"])
            future_same_count = sum(
                1
                for idx in range(run_idx + 1, len(runs))
                if abs(runs[idx]["value"] - repeated_bridge_value) <= cfg.tol and abs(runs[idx]["value"]) > cfg.tol
            )
            if future_same_count >= 2:
                target_div = repeated_bridge_value
                source = "repeated_nonzero_bridge"

        if pd.isna(target_div) and (
            prev_valid is not None
            and next_valid is not None
            and runs[prev_valid]["length"] >= cfg.stable_dividend_run_days
            and runs[next_valid]["length"] >= cfg.stable_dividend_run_days
            and run["length"] <= cfg.dividend_spike_max_span
            and abs(prev_div - next_div) <= cfg.tol
            and abs(run["value"] - prev_div) > cfg.tol
            and run["start"] > 0
            and run["end"] < len(out) - 1
        ):
            prev_unit = float(out.at[run["start"] - 1, "UnitNAV"])
            next_unit = float(out.at[run["end"] + 1, "UnitNAV"])
            run_unit_start = float(out.at[run["start"], "UnitNAV"])
            run_unit_end = float(out.at[run["end"], "UnitNAV"])
            prev_accu = float(out.at[run["start"] - 1, "AccuNAV"])
            next_accu = float(out.at[run["end"] + 1, "AccuNAV"])
            run_accu_start = float(out.at[run["start"], "AccuNAV"])
            run_accu_end = float(out.at[run["end"], "AccuNAV"])

            unit_jump = abs(run_unit_start - prev_unit) + abs(next_unit - run_unit_end)
            accu_jump = abs(run_accu_start - prev_accu) + abs(next_accu - run_accu_end)

            target_div = prev_div
            source = "stable_dividend_revert"
            mode_hint = "unit_as_unit" if unit_jump <= accu_jump else "accu_as_accu"
            future_same_count = sum(
                1
                for idx in range(run_idx + 1, len(runs))
                if abs(runs[idx]["value"] - run["value"]) <= cfg.tol and abs(runs[idx]["value"]) > cfg.tol
            )
            if abs(run["value"]) > cfg.tol and abs(target_div) <= cfg.tol and future_same_count < 2:
                force_context_run = True

        elif pd.isna(target_div) and (
            pd.notna(prev_div)
            and pd.notna(next_div)
            and prev_gap is not None
            and next_gap is not None
            and prev_gap <= cfg.context_neighbor_max_gap
            and next_gap <= cfg.context_neighbor_max_gap
            and abs(prev_div - next_div) <= cfg.tol
            and abs(run["value"] - prev_div) > cfg.tol
        ):
            proposed_div = 0.5 * (prev_div + next_div)
            # Keep nonzero, non-shared anchor rows intact when they are surrounded by
            # zero-dividend shared-value rows. In this pattern the zero/shared rows are
            # usually the confused ones, not the nonzero anchor rows.
            if not (
                abs(proposed_div) <= cfg.tol
                and abs(run["value"]) > cfg.tol
                and shared_ratio < 0.8
            ):
                target_div = proposed_div
                source = "surrounding_div_match"
        elif pd.isna(target_div) and (
            pd.notna(prev_div)
            and pd.notna(next_div)
            and next_gap is not None
            and next_gap <= cfg.context_neighbor_max_gap
            and next_div > prev_div + cfg.tol
            and run["start"] > 0
        ):
            prev_unit = float(out.at[run["start"] - 1, "UnitNAV"])
            curr_unit = float(out.at[run["start"], "UnitNAV"])
            implied_dividend = next_div - prev_div
            future_ok, _ = is_total_return_anomaly(
                curr_unit,
                prev_unit,
                implied_dividend,
                cfg.nav_change_threshold,
            )
            if not future_ok:
                target_div = next_div
                source = "future_dividend_backfill"
        elif pd.isna(target_div) and (
            pd.notna(prev_div)
            and prev_div > cfg.tol
            and run["length"] <= cfg.dividend_spike_max_span
            and abs(run["value"]) <= cfg.tol
            and shared_ratio >= 0.8
        ):
            target_div = prev_div
            source = "shared_zero_after_positive"

        if pd.notna(target_div):
            out.loc[run["start"] : run["end"], "context_dividend"] = target_div
            out.loc[run["start"] : run["end"], "context_dividend_source"] = source
            if mode_hint:
                out.loc[run["start"] : run["end"], "context_mode_hint"] = mode_hint
                out.loc[run["start"] : run["end"], "context_mode_locked"] = True
                if force_context_run:
                    out.loc[run["start"] : run["end"], "force_context_repair"] = True
            elif (
                source in {"surrounding_div_match", "shared_zero_after_positive", "repeated_nonzero_bridge"}
                and abs(run["value"]) <= cfg.tol
                and abs(target_div) > cfg.tol
            ):
                prev_same = next(
                    (
                        runs[idx]
                        for idx in range(run_idx - 1, -1, -1)
                        if abs(runs[idx]["value"] - target_div) <= cfg.tol and abs(runs[idx]["value"]) > cfg.tol
                    ),
                    None,
                )
                next_same = next(
                    (
                        runs[idx]
                        for idx in range(run_idx + 1, len(runs))
                        if abs(runs[idx]["value"] - target_div) <= cfg.tol and abs(runs[idx]["value"]) > cfg.tol
                    ),
                    None,
                )
                if prev_same is not None or next_same is not None:
                    assign_shared_plateau_mode_hints(
                        run["start"],
                        run["end"],
                        float(target_div),
                        prev_same,
                        next_same,
                        prefer_closer_side=(source == "shared_zero_after_positive"),
                    )
                else:
                    assign_shared_value_mode_hint(
                        run["start"],
                        run["end"],
                        prefer_closer_side=(source == "shared_zero_after_positive"),
                    )
                run_slice = slice(run["start"], run["end"] + 1)
                shared_mask = (
                    out.loc[run_slice, "UnitNAV"] - out.loc[run_slice, "AccuNAV"]
                ).abs().le(cfg.tol)
                locked_mask = out.loc[run_slice, "context_mode_locked"].fillna(False)
                if (
                    source == "repeated_nonzero_bridge"
                    and prev_same is not None
                    and next_same is not None
                    and prev_same == run_idx - 1
                    and next_same == run_idx + 1
                ):
                    out.loc[run_slice, "force_context_repair"] = shared_mask & locked_mask
                elif source != "repeated_nonzero_bridge":
                    out.loc[run_slice, "force_context_repair"] = shared_mask & locked_mask

        if (
            pd.notna(next_div)
            and next_gap is not None
            and next_gap <= cfg.context_neighbor_max_gap
            and next_div > run["value"] + cfg.tol
        ):
            event_start = None
            implied_dividend = next_div - run["value"]
            for idx in range(run["start"], run["end"] + 1):
                if idx == 0:
                    continue
                prev_unit = float(out.at[idx - 1, "UnitNAV"])
                curr_unit = float(out.at[idx, "UnitNAV"])
                raw_flag, _ = is_total_return_anomaly(curr_unit, prev_unit, 0.0, cfg.nav_change_threshold)
                future_flag, _ = is_total_return_anomaly(
                    curr_unit,
                    prev_unit,
                    implied_dividend,
                    cfg.nav_change_threshold,
                )
                if raw_flag and not future_flag:
                    event_start = idx
                    break

            if event_start is not None:
                out.loc[event_start : run["end"], "context_dividend"] = next_div
                out.loc[event_start : run["end"], "context_dividend_source"] = "future_dividend_backfill"

    return out


def mark_forced_outlier_blocks(work: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = work.copy()
    out["force_exclude_outlier"] = False
    out["force_exclusion_reason"] = ""
    if "raw_accu_div" not in out.columns:
        out["raw_accu_div"] = out["AccuNAV"] - out["UnitNAV"]

    runs = build_dividend_runs(out["raw_accu_div"], cfg.tol)
    if len(runs) < 3:
        return out

    for i in range(1, len(runs) - 1):
        prev_run = runs[i - 1]
        curr_run = runs[i]
        next_run = runs[i + 1]
        if curr_run["length"] > cfg.max_outlier_span:
            continue
        if prev_run["length"] < cfg.stable_dividend_run_days or next_run["length"] < cfg.stable_dividend_run_days:
            continue
        if abs(prev_run["value"] - next_run["value"]) > cfg.tol:
            continue
        if abs(curr_run["value"] - prev_run["value"]) <= cfg.tol:
            continue
        if curr_run["start"] <= 0 or curr_run["end"] >= len(out) - 1:
            continue

        prev_row = out.iloc[curr_run["start"] - 1]
        next_row = out.iloc[curr_run["end"] + 1]
        all_far = True
        for idx in range(curr_run["start"], curr_run["end"] + 1):
            row = out.iloc[idx]
            unit_far_prev, _ = is_return_anomaly(row["UnitNAV"], prev_row["UnitNAV"], cfg.nav_change_threshold)
            unit_far_next, _ = is_return_anomaly(row["UnitNAV"], next_row["UnitNAV"], cfg.nav_change_threshold)
            accu_far_prev, _ = is_return_anomaly(row["AccuNAV"], prev_row["AccuNAV"], cfg.nav_change_threshold)
            accu_far_next, _ = is_return_anomaly(row["AccuNAV"], next_row["AccuNAV"], cfg.nav_change_threshold)
            if not (unit_far_prev and unit_far_next and accu_far_prev and accu_far_next):
                all_far = False
                break

        if all_far:
            out.loc[curr_run["start"] : curr_run["end"], "force_exclude_outlier"] = True
            out.loc[curr_run["start"] : curr_run["end"], "force_exclusion_reason"] = "dividend_spike_revert_block"

    return out


def build_single_dividend_candidate(
    unit: float,
    accu: float,
    target_div: float,
    source_label: str,
    mode_hint: str,
) -> dict:
    candidate_map = {
        "unit_as_unit": {
            "mode": f"{source_label}_unit_as_unit",
            "action": f"{source_label}_unit_as_unit",
            "swapped": False,
            "unit_nav": unit,
            "accu_nav": unit + target_div,
        },
        "accu_as_accu": {
            "mode": f"{source_label}_accu_as_accu",
            "action": f"{source_label}_accu_as_accu",
            "swapped": False,
            "unit_nav": accu - target_div,
            "accu_nav": accu,
        },
    }
    candidate = candidate_map[mode_hint].copy()
    candidate["raw_unit_nav"] = unit
    candidate["raw_accu_nav"] = accu
    return candidate


def build_dividend_candidates(
    unit: float,
    accu: float,
    target_div: float,
    source_label: str,
) -> list[dict]:
    return [
        {
            "mode": f"{source_label}_unit_as_unit",
            "action": f"{source_label}_unit_as_unit",
            "swapped": False,
            "unit_nav": unit,
            "accu_nav": unit + target_div,
            "raw_unit_nav": unit,
            "raw_accu_nav": accu,
        },
        {
            "mode": f"{source_label}_accu_as_accu",
            "action": f"{source_label}_accu_as_accu",
            "swapped": False,
            "unit_nav": accu - target_div,
            "accu_nav": accu,
            "raw_unit_nav": unit,
            "raw_accu_nav": accu,
        },
        {
            "mode": f"{source_label}_accu_as_unit",
            "action": f"{source_label}_accu_as_unit",
            "swapped": False,
            "unit_nav": accu,
            "accu_nav": accu + target_div,
            "raw_unit_nav": unit,
            "raw_accu_nav": accu,
        },
        {
            "mode": f"{source_label}_unit_as_accu",
            "action": f"{source_label}_unit_as_accu",
            "swapped": False,
            "unit_nav": unit - target_div,
            "accu_nav": unit,
            "raw_unit_nav": unit,
            "raw_accu_nav": accu,
        },
    ]


def evaluate_transition(
    prev_state: dict | None,
    curr_state: dict,
    changed_state: bool,
    trading_day: pd.Timestamp,
    cfg: Config,
) -> tuple[float, dict]:
    unit_nav = curr_state["unit_nav"]
    accu_nav = curr_state["accu_nav"]
    accu_dividends = accu_nav - unit_nav

    cost = 0.0
    event = {
        "accu_dividends": accu_dividends,
        "dividends": math.nan,
        "dividend_cleaned": 0.0,
        "neg_dividend_flag": False,
        "return_anomaly_flag": False,
        "dividend_frequency_flag": False,
        "last_dividend_day": None,
    }

    if pd.isna(unit_nav) or pd.isna(accu_nav):
        cost += STRONG_PENALTY
        return cost, event

    if unit_nav <= 0 or accu_nav <= 0:
        cost += STRONG_PENALTY
        return cost, event

    if accu_dividends < -cfg.tol:
        cost += STRONG_PENALTY + NEG_ACC_DIV_PENALTY + abs(accu_dividends) * 10_000
        return cost, event

    if prev_state is not None:
        prev_accu_dividends = prev_state["accu_dividends"]
        dividends = accu_dividends - prev_accu_dividends
        event["dividends"] = dividends
        cleaned = 0.0 if abs(dividends) <= cfg.tol else dividends
        event["dividend_cleaned"] = cleaned

        if dividends < -cfg.tol:
            event["neg_dividend_flag"] = True
            cost += STRONG_PENALTY + abs(dividends) * 10_000

        prev_unit_nav = prev_state["unit_nav"]
        prev_accu_nav = prev_state["accu_nav"]
        unit_flag, unit_return_rate = is_total_return_anomaly(
            unit_nav,
            prev_unit_nav,
            cleaned,
            cfg.nav_change_threshold,
        )
        if unit_flag:
            event["return_anomaly_flag"] = True
            cost += RETURN_PENALTY + (abs(unit_return_rate) - cfg.nav_change_threshold) * 20_000

        accu_flag, accu_return_rate = is_return_anomaly(accu_nav, prev_accu_nav, cfg.nav_change_threshold)
        if accu_flag:
            event["return_anomaly_flag"] = True
            cost += RETURN_PENALTY + (abs(accu_return_rate) - cfg.nav_change_threshold) * 20_000

        if cleaned > 0 and prev_state.get("last_dividend_day") is not None:
            delta_days = (trading_day - prev_state["last_dividend_day"]).days
            if delta_days <= cfg.dividend_window_days:
                event["dividend_frequency_flag"] = True
                cost += FREQUENCY_PENALTY + (cfg.dividend_window_days - delta_days + 1) * 20

    if changed_state:
        cost += SWITCH_PENALTY

    unit_delta = abs(curr_state["unit_nav"] - curr_state["raw_unit_nav"])
    accu_delta = abs(curr_state["accu_nav"] - curr_state["raw_accu_nav"])
    cost += (unit_delta + accu_delta) * 1_000 * EDIT_PENALTY
    if curr_state["action"] != "original":
        cost += 1.0

    if event["dividend_cleaned"] > 0:
        event["last_dividend_day"] = trading_day
    elif prev_state is not None:
        event["last_dividend_day"] = prev_state.get("last_dividend_day")
    else:
        event["last_dividend_day"] = None

    return cost, event


def generate_candidates(row: pd.Series, prev_state: dict | None) -> list[dict]:
    unit = row["UnitNAV"]
    accu = row["AccuNAV"]
    context_div = row.get("context_dividend", math.nan)
    context_mode_hint = row.get("context_mode_hint", "")
    context_source = str(row.get("context_dividend_source", "context_div")).strip() or "context_div"
    context_mode_locked = bool(row.get("context_mode_locked", False))
    force_context_repair = bool(row.get("force_context_repair", False))
    forced_context_candidate: dict | None = None

    if pd.notna(context_div) and context_mode_hint in {"unit_as_unit", "accu_as_accu"}:
        forced_context_candidate = build_single_dividend_candidate(
            unit,
            accu,
            float(context_div),
            context_source,
            context_mode_hint,
        )
        if force_context_repair:
            return [forced_context_candidate]

    candidates = [
        {
            "mode": "original",
            "action": "original",
            "swapped": False,
            "unit_nav": unit,
            "accu_nav": accu,
            "raw_unit_nav": unit,
            "raw_accu_nav": accu,
        },
        {
            "mode": "swap",
            "action": "swap",
            "swapped": True,
            "unit_nav": accu,
            "accu_nav": unit,
            "raw_unit_nav": unit,
            "raw_accu_nav": accu,
        },
    ]

    if prev_state is not None and pd.notna(prev_state.get("accu_dividends")):
        prev_div = prev_state["accu_dividends"]
        candidates.extend(build_dividend_candidates(unit, accu, prev_div, "prev_div"))

    if pd.notna(context_div):
        if forced_context_candidate is not None:
            candidates.append(forced_context_candidate)
        if not context_mode_locked:
            candidates.extend(build_dividend_candidates(unit, accu, float(context_div), "context_div"))

    unique: dict[tuple, dict] = {}
    for candidate in candidates:
        key = (
            candidate["mode"],
            round(float(candidate["unit_nav"]), 10) if pd.notna(candidate["unit_nav"]) else None,
            round(float(candidate["accu_nav"]), 10) if pd.notna(candidate["accu_nav"]) else None,
        )
        unique[key] = candidate
    return list(unique.values())


def is_outlier_block(df: pd.DataFrame, start: int, end: int, cfg: Config) -> bool:
    if start <= 0 or end >= len(df) - 1:
        return False
    if end - start + 1 > cfg.max_outlier_span:
        return False

    prev_row = df.iloc[start - 1]
    next_row = df.iloc[end + 1]
    prev_next_unit_flag, _ = is_return_anomaly(next_row["UnitNAV"], prev_row["UnitNAV"], cfg.nav_change_threshold)
    prev_next_accu_flag, _ = is_return_anomaly(next_row["AccuNAV"], prev_row["AccuNAV"], cfg.nav_change_threshold)
    if prev_next_unit_flag or prev_next_accu_flag:
        return False

    block = df.iloc[start : end + 1]

    unit_far_from_prev = block["UnitNAV"].apply(
        lambda value: is_return_anomaly(value, prev_row["UnitNAV"], cfg.nav_change_threshold)[0]
    )
    unit_far_from_next = block["UnitNAV"].apply(
        lambda value: is_return_anomaly(value, next_row["UnitNAV"], cfg.nav_change_threshold)[0]
    )
    accu_far_from_prev = block["AccuNAV"].apply(
        lambda value: is_return_anomaly(value, prev_row["AccuNAV"], cfg.nav_change_threshold)[0]
    )
    accu_far_from_next = block["AccuNAV"].apply(
        lambda value: is_return_anomaly(value, next_row["AccuNAV"], cfg.nav_change_threshold)[0]
    )

    unit_far = unit_far_from_prev & unit_far_from_next
    accu_far = accu_far_from_prev & accu_far_from_next
    return bool((unit_far & accu_far).all())


def mark_outlier_blocks(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    force_mask = out.get("force_exclude_outlier", pd.Series(False, index=out.index)).fillna(False)
    force_reason = out.get("force_exclusion_reason", pd.Series("", index=out.index)).fillna("")
    out["exclude_as_outlier"] = force_mask.astype(bool)
    out["exclusion_reason"] = np.where(force_mask, force_reason, "")

    i = 1
    while i < len(out) - 1:
        if bool(out.at[i, "exclude_as_outlier"]):
            i += 1
            continue
        found = False
        max_end = min(len(out) - 2, i + cfg.max_outlier_span - 1)
        for end in range(max_end, i - 1, -1):
            if bool(out.loc[i:end, "exclude_as_outlier"].any()):
                continue
            if is_outlier_block(out, i, end, cfg):
                out.loc[out.index[i : end + 1], "exclude_as_outlier"] = True
                out.loc[out.index[i : end + 1], "exclusion_reason"] = "isolated_jump_block"
                i = end + 1
                found = True
                break
        if not found:
            i += 1

    return out


def recompute_validation_flags(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    out["AccuDividends"] = out["AccuNAV"] - out["UnitNAV"]
    out["Dividends"] = np.nan
    out["dividend_cleaned"] = 0.0
    out["neg_dividend_flag"] = False
    out["return_anomaly_flag"] = False
    out["dividend_frequency_flag"] = False

    kept = out.index[~out["exclude_as_outlier"]].tolist()
    prev_idx = None
    last_dividend_day = None
    for idx in kept:
        if prev_idx is None:
            prev_idx = idx
            continue

        dividends = out.at[idx, "AccuDividends"] - out.at[prev_idx, "AccuDividends"]
        out.at[idx, "Dividends"] = dividends
        cleaned = 0.0 if abs(dividends) <= cfg.tol else dividends
        out.at[idx, "dividend_cleaned"] = cleaned
        if dividends < -cfg.tol:
            out.at[idx, "neg_dividend_flag"] = True

        unit_flag, _ = is_total_return_anomaly(
            out.at[idx, "UnitNAV"],
            out.at[prev_idx, "UnitNAV"],
            cleaned,
            cfg.nav_change_threshold,
        )
        accu_flag, _ = is_return_anomaly(out.at[idx, "AccuNAV"], out.at[prev_idx, "AccuNAV"], cfg.nav_change_threshold)
        if unit_flag or accu_flag:
            out.at[idx, "return_anomaly_flag"] = True

        if cleaned > 0:
            if last_dividend_day is not None:
                delta_days = (out.at[idx, "TradingDay"] - last_dividend_day).days
                if delta_days <= cfg.dividend_window_days:
                    out.at[idx, "dividend_frequency_flag"] = True
            last_dividend_day = out.at[idx, "TradingDay"]

        prev_idx = idx

    return out


def solve_repair_path(work: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, int]:
    n = len(work)
    dp_cost: list[list[float]] = []
    prev_choice: list[list[int]] = []
    dp_event: list[list[dict]] = []
    dp_state: list[list[dict]] = []

    first_candidates = generate_candidates(work.iloc[0], None)
    first_costs = []
    first_prev = []
    first_events = []
    first_states = []
    for candidate in first_candidates:
        cost, event = evaluate_transition(
            prev_state=None,
            curr_state=candidate,
            changed_state=(candidate["mode"] != "original"),
            trading_day=work.at[0, "TradingDay"],
            cfg=cfg,
        )
        first_costs.append(cost)
        first_prev.append(-1)
        first_events.append(event)
        first_states.append(
            {
                **candidate,
                "accu_dividends": event["accu_dividends"],
                "last_dividend_day": event["last_dividend_day"],
            }
        )
    dp_cost.append(first_costs)
    prev_choice.append(first_prev)
    dp_event.append(first_events)
    dp_state.append(first_states)

    for i in range(1, n):
        best_by_mode: dict[str, tuple[float, int, dict, dict]] = {}
        for prev_idx, prev_state in enumerate(dp_state[i - 1]):
            for candidate in generate_candidates(work.iloc[i], prev_state):
                cost, event = evaluate_transition(
                    prev_state=prev_state,
                    curr_state=candidate,
                    changed_state=(prev_state["mode"] != candidate["mode"]),
                    trading_day=work.at[i, "TradingDay"],
                    cfg=cfg,
                )
                total_cost = dp_cost[i - 1][prev_idx] + cost
                mode = candidate["mode"]
                if mode not in best_by_mode or total_cost < best_by_mode[mode][0]:
                    best_by_mode[mode] = (
                        total_cost,
                        prev_idx,
                        event,
                        {
                            **candidate,
                            "accu_dividends": event["accu_dividends"],
                            "last_dividend_day": event["last_dividend_day"],
                        },
                    )

        row_costs = []
        row_prev = []
        row_events = []
        row_states = []
        for _, best_values in best_by_mode.items():
            best_cost, best_prev, best_event, best_state = best_values
            row_costs.append(best_cost)
            row_prev.append(best_prev)
            row_events.append(best_event)
            row_states.append(best_state)
        dp_cost.append(row_costs)
        prev_choice.append(row_prev)
        dp_event.append(row_events)
        dp_state.append(row_states)

    final_state = int(np.argmin(dp_cost[n - 1]))
    chosen = [0] * n
    chosen[n - 1] = final_state
    for i in range(n - 1, 0, -1):
        chosen[i - 1] = prev_choice[i][chosen[i]]

    repaired = work.copy()
    swap_flags = []
    action_labels = []
    accu_dividends = []
    dividends = []
    dividend_cleaned = []
    neg_flags = []
    ret_flags = []
    freq_flags = []
    total_switches = 0
    for i, state in enumerate(chosen):
        event = dp_event[i][state]
        candidate = dp_state[i][state]
        repaired.at[i, "UnitNAV"] = candidate["unit_nav"]
        repaired.at[i, "AccuNAV"] = candidate["accu_nav"]
        swap_flags.append(candidate["swapped"])
        action_labels.append(candidate["action"])
        accu_dividends.append(event["accu_dividends"])
        dividends.append(event["dividends"])
        dividend_cleaned.append(event["dividend_cleaned"])
        neg_flags.append(event["neg_dividend_flag"])
        ret_flags.append(event["return_anomaly_flag"])
        freq_flags.append(event["dividend_frequency_flag"])
        if i > 0 and chosen[i] != chosen[i - 1]:
            total_switches += 1

    repaired["AccuDividends"] = accu_dividends
    repaired["Dividends"] = dividends
    repaired["dividend_cleaned"] = dividend_cleaned
    repaired["swap_flag"] = swap_flags
    repaired["repair_action"] = action_labels
    repaired["neg_dividend_flag"] = neg_flags
    repaired["return_anomaly_flag"] = ret_flags
    repaired["dividend_frequency_flag"] = freq_flags
    return repaired, total_switches


def rerun_without_outliers(
    original_work: pd.DataFrame,
    current_repaired: pd.DataFrame,
    cfg: Config,
    max_rounds: int = 3,
) -> tuple[pd.DataFrame, int]:
    repaired = current_repaired.copy()
    total_switches = count_mode_switches(repaired["repair_action"].tolist())
    prev_excluded = None

    for _ in range(max_rounds):
        repaired = mark_outlier_blocks(repaired, cfg)
        repaired = recompute_validation_flags(repaired, cfg)
        excluded_mask = repaired["exclude_as_outlier"].fillna(False).to_numpy()
        if prev_excluded is not None and np.array_equal(excluded_mask, prev_excluded):
            break
        if not excluded_mask.any():
            break

        active_work = original_work.loc[~excluded_mask].copy().reset_index(drop=True)
        rerun_repaired, total_switches = solve_repair_path(active_work, cfg)
        repair_cols = [
            "UnitNAV",
            "AccuNAV",
            "AccuDividends",
            "Dividends",
            "dividend_cleaned",
            "swap_flag",
            "repair_action",
            "neg_dividend_flag",
            "return_anomaly_flag",
            "dividend_frequency_flag",
        ]
        rerun_indexed = rerun_repaired.set_index("_base_pos")
        active_positions = repaired.loc[~excluded_mask, "_base_pos"]
        for col in repair_cols:
            repaired.loc[~excluded_mask, col] = active_positions.map(rerun_indexed[col])
        prev_excluded = excluded_mask.copy()

    repaired = mark_outlier_blocks(repaired, cfg)
    repaired = recompute_validation_flags(repaired, cfg)
    return repaired, total_switches


def count_mode_switches(action_labels: list[str]) -> int:
    total_switches = 0
    for i in range(1, len(action_labels)):
        if action_labels[i] != action_labels[i - 1]:
            total_switches += 1
    return total_switches


def repair_fund(group: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    work = group.sort_values(["TradingDay", "_row_id"]).reset_index(drop=True).copy()
    work["_base_pos"] = np.arange(len(work))
    work = annotate_context_dividends(work, cfg)
    work = mark_forced_outlier_blocks(work, cfg)
    repaired, total_switches = solve_repair_path(work, cfg)
    repaired, total_switches = rerun_without_outliers(work, repaired, cfg)
    n = len(repaired)

    consecutive_runs = []
    action_labels = repaired["repair_action"].tolist()
    run_start = 0
    for i in range(1, n + 1):
        if i == n or action_labels[i] != action_labels[run_start]:
            if action_labels[run_start] != "original":
                consecutive_runs.append(
                    {
                        "action": action_labels[run_start],
                        "start_day": repaired.at[run_start, "TradingDay"],
                        "end_day": repaired.at[i - 1, "TradingDay"],
                        "rows": i - run_start,
                    }
                )
            run_start = i

    active_rows = repaired.loc[~repaired["exclude_as_outlier"]]
    has_neg = int(active_rows["neg_dividend_flag"].sum())
    has_ret = int(active_rows["return_anomaly_flag"].sum())
    has_freq = int(active_rows["dividend_frequency_flag"].sum())
    repair_failed = bool(has_neg > 0 or has_ret > 0 or has_freq > 0)

    stats = {
        "FundID": repaired.at[0, "FundID"],
        "rows": n,
        "excluded_rows": int(repaired["exclude_as_outlier"].sum()),
        "swapped_rows": int(repaired["swap_flag"].fillna(False).sum()),
        "edited_rows": int(sum(action != "original" for action in action_labels)),
        "swap_switches": total_switches,
        "neg_dividend_rows": has_neg,
        "return_anomaly_rows": has_ret,
        "dividend_frequency_rows": has_freq,
        "repair_failed": repair_failed,
        "failure_reason": "",
        "swap_segments": consecutive_runs,
    }

    reasons = []
    if stats["neg_dividend_rows"] > 0:
        reasons.append("negative_dividend_remaining")
    if stats["return_anomaly_rows"] > 0:
        reasons.append("return_anomaly_remaining")
    if stats["dividend_frequency_rows"] > 0:
        reasons.append("dividend_too_frequent_remaining")
    stats["failure_reason"] = ",".join(reasons)
    return repaired, stats


def build_failure_detail(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df["neg_dividend_flag"].fillna(False)
        | df["return_anomaly_flag"].fillna(False)
        | df["dividend_frequency_flag"].fillna(False)
    )
    detail = df.loc[
        mask,
        [
            "FundID",
            "TradingDay",
            "UnitNAV",
            "AccuNAV",
            "AccuDividends",
            "Dividends",
            "dividend_cleaned",
            "swap_flag",
            "exclude_as_outlier",
            "exclusion_reason",
            "neg_dividend_flag",
            "return_anomaly_flag",
            "dividend_frequency_flag",
        ],
    ].copy()
    detail["issue_types"] = detail.apply(
        lambda row: ",".join(
            [
                name
                for name, flag in (
                    ("negative_dividend", row["neg_dividend_flag"]),
                    ("return_anomaly", row["return_anomaly_flag"]),
                    ("dividend_too_frequent", row["dividend_frequency_flag"]),
                )
                if bool(flag)
            ]
        ),
        axis=1,
    )
    return detail


def build_comparison_view(raw_df: pd.DataFrame, repaired_df: pd.DataFrame) -> pd.DataFrame:
    raw_keep = [
        "_row_id",
        "FundID",
        "TradingDay",
        "UnitNAV",
        "AccuNAV",
        "AccuDividends",
        "Dividends",
        "dividend_cleaned",
        "year",
        "source_row_count",
    ]
    if "source_file" in raw_df.columns:
        raw_keep.append("source_file")
    raw_view = raw_df[raw_keep].copy()
    repaired_view = repaired_df.copy()

    rename_raw = {
        "UnitNAV": "raw_UnitNAV",
        "AccuNAV": "raw_AccuNAV",
        "AccuDividends": "raw_AccuDividends",
        "Dividends": "raw_Dividends",
        "dividend_cleaned": "raw_dividend_cleaned",
    }
    rename_repaired = {
        "UnitNAV": "fixed_UnitNAV",
        "AccuNAV": "fixed_AccuNAV",
        "AccuDividends": "fixed_AccuDividends",
        "Dividends": "fixed_Dividends",
        "dividend_cleaned": "fixed_dividend_cleaned",
    }

    raw_view = raw_view.rename(columns=rename_raw)
    repaired_view = repaired_view.rename(columns=rename_repaired)

    repaired_keep = [
        "_row_id",
        "FundID",
        "TradingDay",
        "fixed_UnitNAV",
        "fixed_AccuNAV",
        "fixed_AccuDividends",
        "fixed_Dividends",
        "fixed_dividend_cleaned",
        "preprocess_swapped",
        "preprocess_reason",
        "swap_flag",
        "repair_action",
        "exclude_as_outlier",
        "exclusion_reason",
        "neg_dividend_flag",
        "return_anomaly_flag",
        "dividend_frequency_flag",
    ]
    compare_df = raw_view.merge(
        repaired_view[repaired_keep],
        on=["_row_id", "FundID", "TradingDay"],
        how="left",
        suffixes=("", "_dup"),
    )
    compare_df["unit_changed"] = compare_df["raw_UnitNAV"] != compare_df["fixed_UnitNAV"]
    compare_df["accu_changed"] = compare_df["raw_AccuNAV"] != compare_df["fixed_AccuNAV"]
    return compare_df


def load_input_dataframe(input_path: Path) -> pd.DataFrame:
    if input_path.is_dir():
        files = sorted(
            [
                path
                for path in input_path.iterdir()
                if path.is_file() and path.suffix.lower() in {".xlsx", ".xls"}
                and not path.name.startswith(".~")
                and not path.name.startswith("~$")
            ]
        )
        if not files:
            raise FileNotFoundError(f"no excel files found under {input_path}")
        frames = []
        for path in files:
            frame = pd.read_excel(path)
            frame["source_file"] = path.name
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)
    return pd.read_excel(input_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="按基金拆分")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--config", default="thresholds.json")
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--dividend-window-days", type=int, default=None)
    parser.add_argument("--nav-change-threshold", type=float, default=None)
    parser.add_argument("--max-outlier-span", type=int, default=None)
    parser.add_argument("--context-run-min-len", type=int, default=None)
    parser.add_argument("--context-neighbor-max-gap", type=int, default=None)
    parser.add_argument("--stable-dividend-run-days", type=int, default=None)
    parser.add_argument("--dividend-spike-max-span", type=int, default=None)
    args = parser.parse_args()

    cfg_values = {
        "tol": 0.003,
        "dividend_window_days": 20,
        "nav_change_threshold": 0.20,
        "max_outlier_span": 10,
        "context_run_min_len": 2,
        "context_neighbor_max_gap": 12,
        "stable_dividend_run_days": 5,
        "dividend_spike_max_span": 5,
    }
    config_path = Path(args.config)
    if config_path.exists():
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            for key in cfg_values:
                if key in loaded:
                    cfg_values[key] = loaded[key]

    arg_overrides = {
        "tol": args.tol,
        "dividend_window_days": args.dividend_window_days,
        "nav_change_threshold": args.nav_change_threshold,
        "max_outlier_span": args.max_outlier_span,
        "context_run_min_len": args.context_run_min_len,
        "context_neighbor_max_gap": args.context_neighbor_max_gap,
        "stable_dividend_run_days": args.stable_dividend_run_days,
        "dividend_spike_max_span": args.dividend_spike_max_span,
    }
    for key, value in arg_overrides.items():
        if value is not None:
            cfg_values[key] = value

    cfg = Config(**cfg_values)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_input_dataframe(input_path)
    df["TradingDay"] = pd.to_datetime(df["TradingDay"])
    df = normalize_numeric(df, ["UnitNAV", "AccuNAV", "AccuDividends", "Dividends", "dividend_cleaned"])
    df["_row_id"] = np.arange(len(df))
    dedup_df = deduplicate_fund_day(df)
    prepared_df = preprocess_nav_order(dedup_df)

    repaired_frames = []
    summaries = []
    fund_groups = list(prepared_df.groupby("FundID", sort=True))
    total_groups = len(fund_groups)
    for idx, (_, group) in enumerate(fund_groups, start=1):
        fund_id = group["FundID"].iloc[0]
        print(f"processing_fund={fund_id} ({idx}/{total_groups})", flush=True)
        repaired, stats = repair_fund(group, cfg)
        repaired_frames.append(repaired)
        summaries.append(stats)

    repaired_df = pd.concat(repaired_frames, ignore_index=True).sort_values("_row_id")
    summary_df = pd.DataFrame(summaries).sort_values(["repair_failed", "neg_dividend_rows", "return_anomaly_rows"], ascending=[False, False, False])
    comparison_df = build_comparison_view(dedup_df, repaired_df)
    repaired_export_df = repaired_df.drop(columns="_row_id")

    failure_detail_df = build_failure_detail(repaired_df)

    output_xlsx = output_dir / "具体异常净值_修正结果.xlsx"
    summary_csv = output_dir / "修复汇总.csv"
    failure_csv = output_dir / "修复失败明细.csv"

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        comparison_df.to_excel(writer, sheet_name="compare_raw_fixed", index=False)
        repaired_export_df.to_excel(writer, sheet_name="repaired_data", index=False)
        summary_df.to_excel(writer, sheet_name="repair_summary", index=False)
        failure_detail_df.to_excel(writer, sheet_name="issue_details", index=False)

    summary_df.to_csv(summary_csv, index=False)
    failure_detail_df.to_csv(failure_csv, index=False)

    total_funds = summary_df["FundID"].nunique()
    edited_funds = int((summary_df["edited_rows"] > 0).sum())
    failed_funds = int(summary_df["repair_failed"].sum())
    success_funds = total_funds - failed_funds
    print(f"output_xlsx={output_xlsx}")
    print(f"summary_csv={summary_csv}")
    print(f"failure_csv={failure_csv}")
    print(f"total_funds={total_funds}")
    print(f"edited_funds={edited_funds}")
    print(f"success_funds={success_funds}")
    print(f"failed_funds={failed_funds}")
    print(f"remaining_neg_dividend_rows={int(repaired_df['neg_dividend_flag'].sum())}")
    print(f"remaining_return_anomaly_rows={int(repaired_df['return_anomaly_flag'].sum())}")
    print(f"remaining_dividend_frequency_rows={int(repaired_df['dividend_frequency_flag'].sum())}")
    print(f"config_path={config_path}")
    print(f"config_values={json.dumps(cfg_values, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
