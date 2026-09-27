"""Tune mathematical monthly-share extrapolation for citywide calibration."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import ROUTES, base_forecast, fit_profile, load_labels, wape_score  # noqa: E402


def estimate_share(history: pd.Series, method: str, n: int, damp: float) -> float:
    values = history.tail(n).to_numpy(dtype=float)
    if method == "last":
        return float(values[-1])
    if method == "mean":
        return float(values.mean())
    x = np.arange(len(values), dtype=float)
    slope = np.polyfit(x, values, 1)[0] if len(values) >= 2 else 0.0
    return float(values[-1] + damp * slope)


def main() -> None:
    labels = load_labels()
    labels["month"] = labels.date.dt.to_period("M").astype(str)
    monthly_route_total = labels.groupby("month").y.sum()
    city = pd.read_csv("data/external/tram_monthly_city.csv").set_index("month").passenger_traffic.astype(float)
    shares = (monthly_route_total / city.reindex(monthly_route_total.index)).dropna()
    route_month = labels.groupby(["month", "route"]).y.sum().unstack("route")
    route_shares = route_month.div(city.reindex(route_month.index), axis=0)

    methods = [("last", 1, 0.0)]
    methods += [("mean", n, 0.0) for n in range(2, 7)]
    methods += [("trend", n, damp) for n in range(2, 7) for damp in (0.0, 0.25, 0.5, 0.75, 1.0)]
    route_methods = [("route_trend", n, damp) for n in range(1, 7) for damp in (0.0, 0.25, 0.5, 0.75, 1.0)]
    folds = []
    errors = {f"{m}:{n}:{d:.2f}": [0.0, 0.0] for m, n, d in methods}
    errors.update({f"{m}:{n}:{d:.2f}": [0.0, 0.0] for m, n, d in route_methods})
    for month in range(4, 11):
        target_month = pd.Period(f"2025-{month:02d}", freq="M")
        cutoff = target_month.start_time - pd.Timedelta(days=1)
        dates = pd.date_range(target_month.start_time, target_month.end_time)
        level, shape = fit_profile(labels, cutoff)
        pred = base_forecast(level, shape, dates, routes=ROUTES)
        actual = labels[labels.date.isin(dates)][["route", "date", "hour", "y"]]
        merged = pred.merge(actual, on=["route", "date", "hour"], validate="one_to_one")
        observed = shares[shares.index < str(target_month)]
        target_city = float(city.loc[str(target_month)])
        fold_scores = {}
        for method, n, damp in methods:
            key = f"{method}:{n}:{damp:.2f}"
            share = np.clip(estimate_share(observed, method, n, damp), 0.25, 0.45)
            target_total = target_city * share
            calibrated = merged.pred.to_numpy() * (target_total / max(merged.pred.sum(), 1))
            y = merged.y.to_numpy()
            error = float(np.abs(y - calibrated).sum())
            denom = float(y.sum())
            errors[key][0] += error
            errors[key][1] += denom
            fold_scores[key] = float(max(0.0, 1 - error / denom))
        # More flexible alternative: forecast each route's share of the city
        # total separately, then scale that route's profile within the month.
        for n in range(1, 7):
            for damp in (0.0, 0.25, 0.5, 0.75, 1.0):
                key = f"route_trend:{n}:{damp:.2f}"
                route_targets = {}
                for route in ROUTES:
                    route_history = route_shares.loc[route_shares.index < str(target_month), route].dropna()
                    values = route_history.tail(n).to_numpy(dtype=float)
                    if len(values) == 0:
                        route_targets[route] = 0.0
                    elif len(values) == 1:
                        route_targets[route] = float(values[-1])
                    else:
                        slope = float(np.polyfit(np.arange(len(values)), values, 1)[0])
                        route_targets[route] = float(max(0.0, values[-1] + damp * slope))
                route_pred = merged.pred.to_numpy().copy()
                for route in ROUTES:
                    mask = merged.route.to_numpy() == route
                    current = float(route_pred[mask].sum())
                    target_total = float(city.loc[str(target_month)] * route_targets[route])
                    route_pred[mask] *= target_total / max(current, 1)
                error = float(np.abs(merged.y.to_numpy() - route_pred).sum())
                denom = float(merged.y.sum())
                errors.setdefault(key, [0.0, 0.0])
                errors[key][0] += error
                errors[key][1] += denom
                fold_scores[key] = float(max(0.0, 1 - error / denom))

        folds.append({"month": str(target_month), "fold_scores": fold_scores})

    ranked = []
    for method, n, damp in methods + route_methods:
        key = f"{method}:{n}:{damp:.2f}"
        error, denom = errors[key]
        ranked.append({"method": method, "window_months": n, "trend_damping": damp,
                       "pooled_score": max(0.0, 1 - error / denom), "total_abs_error": error})
    ranked.sort(key=lambda x: x["pooled_score"], reverse=True)
    out = {"metric": "score=1-WAPE; higher is better", "folds": folds, "ranked": ranked}
    Path("ml/research/citywide_calibration_tune.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(ranked).head(15).round(5).to_string(index=False))
    print("Best per fold:")
    for fold in folds:
        top = max(fold["fold_scores"].items(), key=lambda kv: kv[1])
        print(fold["month"], top[0], round(top[1], 5))


if __name__ == "__main__":
    main()
