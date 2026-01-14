"""Pipeline entrypoint for Retention Guard."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from retention_guard.data import SampleConfig, generate_sample, load_csv
from retention_guard.model import ModelResult, score, train_model


def _infer_risk_band(score_value: float) -> str:
    if score_value >= 0.7:
        return "High"
    if score_value >= 0.4:
        return "Medium"
    return "Low"


def _top_driver_summary(importances: dict[str, float]) -> str:
    if not importances:
        return "insufficient data"
    top = max(importances, key=importances.get)
    return top.replace("cat__", "").replace("num__", "")


def _prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    if "exit_flag" in df.columns:
        return df

    # Create a synthetic label using a simple heuristic for demo purposes.
    # Forking tip: replace this with your real attrition label when available.
    heuristic = (
        (df["engagement_score"] < 60).astype(int)
        + (df["overtime_hours_month"] > 12).astype(int)
        + (df["absenteeism_days_month"] > 3).astype(int)
        + (df["last_promotion_months"] > 36).astype(int)
    )
    df = df.copy()
    df["exit_flag"] = (heuristic >= 2).astype(int)
    return df


def run_pipeline(
    input_path: Optional[Path],
    output_path: Path,
    generate_sample_flag: bool,
    rows: int,
) -> ModelResult:
    """Run the full pipeline and write the scored output CSV.

    Forking tip: add your own feature engineering before training and scoring.
    """
    if generate_sample_flag:
        df = generate_sample(SampleConfig(rows=rows))
    elif input_path is not None:
        df = load_csv(input_path)
    else:
        raise ValueError("Provide --input or --generate-sample.")

    training_df = _prepare_training_data(df)
    result = train_model(training_df)
    risk_scores = score(result.model, df)

    output_columns = [
        "employee_id",
        "dept",
        "tenure_months",
        "last_promotion_months",
        "engagement_score",
        "overtime_hours_month",
        "absenteeism_days_month",
        "performance_score",
        "peer_turnover_rate",
        "internal_mobility",
    ]
    output = df[output_columns].copy()
    output["risk_score"] = risk_scores.round(3)
    output["risk_band"] = output["risk_score"].apply(_infer_risk_band)
    output["top_driver"] = _top_driver_summary(result.feature_importance)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Retention Guard pipeline.")
    parser.add_argument("--input", type=Path, help="Path to input CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate a synthetic dataset for demo use.",
    )
    parser.add_argument("--rows", type=int, default=200, help="Rows for sample data.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = run_pipeline(args.input, args.output, args.generate_sample, args.rows)
    print(f"Model AUC (train): {result.auc:.3f}")
    print("Top drivers:")
    for name, weight in result.feature_importance.items():
        print(f"- {name}: {weight:.3f}")


if __name__ == "__main__":
    main()
