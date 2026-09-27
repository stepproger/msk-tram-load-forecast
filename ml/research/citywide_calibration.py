"""Out-of-time проверки калибровки 9 маршрутов по месячному итогу Москвы.

Реализует те же правила, что core.citywide_monthly_calibration. Значение
общегородского потока целевого месяца является постфактум-признаком (утечкой)
и подходит только для ретроспективной оценки hackathon submission.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from core import (  # noqa: E402
    ROUTES,
    base_forecast,
    citywide_monthly_calibration,
    fit_profile,
    load_labels,
    wape_score,
)


def main():
    y = load_labels()
    rows = []
    for month in range(4, 11):
        target_month = pd.Period(f"2025-{month:02d}", freq="M")
        cut = target_month.start_time - pd.Timedelta(days=1)
        dates = pd.date_range(target_month.start_time, target_month.end_time)
        level, shape = fit_profile(y, cut)
        baseline = base_forecast(level, shape, dates, routes=ROUTES)
        known = y[y.date <= cut]
        calibrated = citywide_monthly_calibration(known, baseline)
        actual = y[y.date.isin(dates)][["route", "date", "hour", "y"]]
        b = baseline.merge(actual, on=["route", "date", "hour"])
        c = calibrated.merge(actual, on=["route", "date", "hour"])
        rows.append({
            "month": str(target_month),
            "target": float(b.y.sum()),
            "baseline_error": float(np.abs(b.y - b.pred).sum()),
            "calibrated_error": float(np.abs(c.y - c.pred).sum()),
            "baseline_score": wape_score(b.y, b.pred),
            "calibrated_score": wape_score(c.y, c.pred),
        })

    result = pd.DataFrame(rows)
    target = result.target.sum()
    base_score = 1 - result.baseline_error.sum() / target
    calibrated_score = 1 - result.calibrated_error.sum() / target
    print(result[["month", "baseline_score", "calibrated_score"]].round(5).to_string(index=False))
    print(f"Pooled Apr–Oct WAPE-score: {base_score:.5f} -> {calibrated_score:.5f} "
          f"(delta {calibrated_score - base_score:+.5f})")


if __name__ == "__main__":
    main()
