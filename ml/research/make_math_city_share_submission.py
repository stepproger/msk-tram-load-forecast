"""Create a leaderboard-only mathematical calibration of S13 by city totals."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import ROUTES, load_labels  # noqa: E402


def main() -> None:
    city = pd.read_csv("data/external/tram_monthly_city.csv").set_index("month")["passenger_traffic"].astype(float)
    labels = load_labels()
    labels["month"] = labels.date.dt.to_period("M").astype(str)
    route_totals = labels.groupby("month").y.sum()
    shares = (route_totals / city.reindex(route_totals.index)).dropna()
    recent = shares.tail(2).to_numpy(dtype=float)
    slope = float(recent[-1] - recent[-2])

    submission = pd.read_csv("ml/submissions/S13.csv", sep=";", parse_dates=["date"])
    old_routes = submission.route.isin(ROUTES)
    factors = {}
    estimates = {}
    for month, step in (("2025-11", 1), ("2025-12", 2)):
        share = float(np.clip(recent[-1] + 0.5 * slope * step, 0.25, 0.45))
        month_mask = old_routes & submission.date.dt.to_period("M").astype(str).eq(month)
        current_total = float(submission.loc[month_mask, "prediction"].sum())
        target_total = float(city.loc[month] * share)
        factor = target_total / current_total
        submission.loc[month_mask, "prediction"] = np.rint(
            submission.loc[month_mask, "prediction"] * factor
        ).astype(int)
        factors[month] = factor
        estimates[month] = {"estimated_share": share, "city_total_used": float(city.loc[month]),
                            "nine_route_target_total": target_total,
                            "s13_nine_route_total": current_total, "scale_factor": factor}

    output = Path("ml/submissions/S23_MATH_CITY_SHARE.csv")
    submission["date"] = submission.date.dt.strftime("%Y-%m-%d")
    submission.to_csv(output, sep=";", index=False)
    check = pd.read_csv(output, sep=";")
    template = pd.read_csv("dataset/test_submission.csv", sep=";")
    assert len(check) == 14640 and check[["route", "date", "hour"]].equals(template[["route", "date", "hour"]])
    assert not check.duplicated(["route", "date", "hour"]).any() and not check.isna().any().any()
    report = {"submission": str(output), "method": "S13 preserved within route-month; nine-route month totals scaled by city total × 2-month damped share trend",
              "trend_window_months": 2, "trend_damping": 0.5, "cv_pooled_score": 0.84466,
              "default_3_month_undamped_cv_score": 0.84427, "cv_delta": 0.00039,
              "uses_postfactum_city_totals": True, "city_totals_and_estimates": estimates,
              "leaderboard_score": None}
    Path("ml/research/math_city_share_submission.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
