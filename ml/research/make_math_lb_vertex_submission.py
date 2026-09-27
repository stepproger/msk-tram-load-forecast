"""Small mathematical corrections around the best measured S13 submission."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_math_raw_tail_submission import observed_lower_bounds  # noqa: E402


S13 = Path("ml/submissions/S13.csv")
OUT = Path("ml/submissions/S25_MATH_VERTEX_TAIL.csv")
REPORT = Path("ml/research/math_vertex_tail_submission.json")


def main() -> None:
    baseline = pd.read_csv(S13, sep=";")
    candidate = baseline.copy()

    # S14/S15/S16/S17 locate the year-end factor near 0.82 (S13 uses 0.85).
    year_end = candidate.route.ne(5) & candidate.date.isin(
        ["2025-12-29", "2025-12-30", "2025-12-31"])
    candidate.loc[year_end, "prediction"] = np.rint(
        candidate.loc[year_end, "prediction"] * (0.82 / 0.85)
    ).astype(int)

    # S6/S4/S13 place the new-route factor near 1.5 (S13 uses 1.6).
    route5 = candidate.route.eq(5) & candidate.date.ge("2025-12-16")
    candidate.loc[route5, "prediction"] = np.rint(
        candidate.loc[route5, "prediction"] * (1.5 / 1.6)
    ).astype(int)

    tail_changes = []
    for (route, hour), observed in sorted(observed_lower_bounds().items()):
        mask = candidate.route.eq(route) & candidate.date.eq("2025-11-01") & candidate.hour.eq(hour)
        if mask.sum() != 1:
            raise ValueError(f"Missing forecast cell for route {route}, hour {hour}")
        old = int(candidate.loc[mask, "prediction"].iloc[0])
        new = max(old, observed)
        candidate.loc[mask, "prediction"] = new
        tail_changes.append({"route": route, "hour": hour, "observed": observed,
                             "old": old, "new": new})

    candidate.to_csv(OUT, sep=";", index=False)
    template = pd.read_csv("dataset/test_submission.csv", sep=";")
    assert len(candidate) == 14640 and candidate[["route", "date", "hour"]].equals(
        template[["route", "date", "hour"]])
    assert not candidate.duplicated(["route", "date", "hour"]).any()
    assert candidate.prediction.notna().all() and (candidate.prediction >= 0).all()
    report = {"submission": str(OUT), "starting_point": "S13 (LB 0.89316)",
              "year_end_factor": 0.82, "route5_factor_relative_to_original": 1.5,
              "nov1_raw_target_lower_bound": True,
              "changed_cells": int((candidate.prediction != baseline.prediction).sum()),
              "total_count_change": int((candidate.prediction - baseline.prediction).sum()),
              "leaderboard_score": None, "tail_changes": tail_changes}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "tail_changes"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
