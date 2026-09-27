"""Adjust S13's new-route scale using the official first-week trip count."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_math_raw_tail_submission import observed_lower_bounds  # noqa: E402


S13 = Path("ml/submissions/S13.csv")
OUT = Path("ml/submissions/S26_MATH_ROUTE5_OFFICIAL.csv")
REPORT = Path("ml/research/math_route5_official_submission.json")


def main() -> None:
    baseline = pd.read_csv(S13, sep=";")
    candidate = baseline.copy()
    route5 = candidate.route.eq(5) & candidate.date.ge("2025-12-16")
    candidate.loc[route5, "prediction"] = np.rint(
        candidate.loc[route5, "prediction"] * (1.5 / 1.6)
    ).astype(int)

    for (route, hour), observed in observed_lower_bounds().items():
        mask = candidate.route.eq(route) & candidate.date.eq("2025-11-01") & candidate.hour.eq(hour)
        if mask.sum() != 1:
            raise ValueError(f"Missing forecast cell for route {route}, hour {hour}")
        candidate.loc[mask, "prediction"] = max(int(candidate.loc[mask, "prediction"].iloc[0]), observed)

    first_week = baseline.route.eq(5) & baseline.date.between("2025-12-16", "2025-12-22")
    report = {
        "submission": str(OUT),
        "starting_point": "S13 (LB 0.89316)",
        "official_first_week_trips_approx": 40000,
        "official_source": "https://transport.mos.ru/mostrans/all_news/127782",
        "s13_first_week_prediction": int(baseline.loc[first_week, "prediction"].sum()),
        "candidate_first_week_prediction": int(candidate.loc[first_week, "prediction"].sum()),
        "route5_factor_relative_to_original": 1.5,
        "nov1_raw_target_lower_bound": True,
        "changed_cells": int((candidate.prediction != baseline.prediction).sum()),
        "total_count_change": int((candidate.prediction - baseline.prediction).sum()),
        "leaderboard_score": None,
    }
    template = pd.read_csv("dataset/test_submission.csv", sep=";")
    assert len(candidate) == 14640 and candidate[["route", "date", "hour"]].equals(
        template[["route", "date", "hour"]]
    )
    assert not candidate.duplicated(["route", "date", "hour"]).any()
    assert candidate.prediction.notna().all() and (candidate.prediction >= 0).all()
    candidate.to_csv(OUT, sep=";", index=False)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
