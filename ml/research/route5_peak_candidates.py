"""Route 5 hour-shape sensitivity candidates based on S26.

The claimed exact hourly counts for route 5 are not independently verified.
These files test a commute-peak hypothesis while preserving each day's total.
"""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "ml/submissions/S26_MATH_ROUTE5_OFFICIAL.csv"
TEMPLATE = ROOT / "dataset/test_submission.csv"
PEAKS = {7: 0.5, 8: 1.0, 9: 0.5, 16: 0.5, 17: 1.0, 18: 1.0}


def redistribute(values: np.ndarray, hours: np.ndarray, strength: float) -> np.ndarray:
    weights = np.array([1.0 + strength * PEAKS.get(int(h), 0.0) for h in hours])
    scaled = values * weights
    scaled *= values.sum() / scaled.sum()
    base = np.floor(scaled).astype(int)
    remainder = int(values.sum() - base.sum())
    if remainder:
        order = np.argsort(-(scaled - base), kind="stable")
        base[order[:remainder]] += 1
    return base


def main() -> None:
    original = pd.read_csv(SOURCE, sep=";")
    template = pd.read_csv(TEMPLATE, sep=";")
    assert original[["route", "date", "hour"]].equals(template[["route", "date", "hour"]])
    dates = pd.to_datetime(original.date)
    route5_weekday = original.route.eq(5) & dates.ge("2025-12-16") & dates.dt.dayofweek.lt(5)
    variants = (
        ("S27_ROUTE5_PEAK15", 0.15),
        ("S28_ROUTE5_PEAK30", 0.30),
        ("S29_ROUTE5_FLAT15", -0.15),
        ("S30_ROUTE5_FLAT30", -0.30),
    )
    for name, strength in variants:
        candidate = original.copy()
        for _, group in original.loc[route5_weekday].groupby("date", sort=False):
            candidate.loc[group.index, "prediction"] = redistribute(
                group.prediction.to_numpy(), group.hour.to_numpy(), strength
            )
        assert candidate.prediction.ge(0).all()
        assert candidate.groupby(["route", "date"]).prediction.sum().equals(
            original.groupby(["route", "date"]).prediction.sum()
        )
        output = SOURCE.with_name(name + ".csv")
        candidate.to_csv(output, sep=";", index=False)
        print(output, "changed cells:", int(candidate.prediction.ne(original.prediction).sum()))


if __name__ == "__main__":
    main()
