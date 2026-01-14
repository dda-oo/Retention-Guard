"""Data loading and synthetic data generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "employee_id",
    "dept",
    "tenure_months",
    "last_promotion_months",
    "salary_band",
    "manager_span",
    "overtime_hours_month",
    "engagement_score",
    "absenteeism_days_month",
    "peer_turnover_rate",
    "performance_score",
    "internal_mobility",
]


@dataclass
class SampleConfig:
    rows: int = 200
    seed: int = 42


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV and ensure required columns are present.

    Forking tip: expand REQUIRED_COLUMNS to match your HR data model.
    """
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return df


def generate_sample(config: Optional[SampleConfig] = None) -> pd.DataFrame:
    """Generate a synthetic HR dataset for demo usage.

    Forking tip: tune distributions to resemble your industry (e.g., tenure mix).
    """
    if config is None:
        config = SampleConfig()

    rng = np.random.default_rng(config.seed)
    rows = config.rows

    dept = rng.choice(["HR", "Sales", "Engineering", "Finance", "Ops"], size=rows)
    tenure = rng.integers(3, 120, size=rows)
    last_promo = np.maximum(0, tenure - rng.integers(0, 60, size=rows))
    salary_band = rng.choice(["A", "B", "C", "D"], size=rows, p=[0.2, 0.4, 0.3, 0.1])
    manager_span = rng.integers(3, 15, size=rows)
    overtime = rng.normal(8, 6, size=rows).clip(0)
    engagement = rng.normal(72, 12, size=rows).clip(30, 98)
    absenteeism = rng.normal(1.5, 1.0, size=rows).clip(0, 6)
    peer_turnover = rng.uniform(0.02, 0.35, size=rows)
    performance = rng.normal(74, 10, size=rows).clip(40, 98)
    mobility = rng.choice([0, 1], size=rows, p=[0.7, 0.3])

    data = pd.DataFrame(
        {
            "employee_id": [f"E{1000 + idx}" for idx in range(rows)],
            "dept": dept,
            "tenure_months": tenure,
            "last_promotion_months": last_promo,
            "salary_band": salary_band,
            "manager_span": manager_span,
            "overtime_hours_month": overtime.round(1),
            "engagement_score": engagement.round(1),
            "absenteeism_days_month": absenteeism.round(1),
            "peer_turnover_rate": peer_turnover.round(3),
            "performance_score": performance.round(1),
            "internal_mobility": mobility,
        }
    )
    return data
