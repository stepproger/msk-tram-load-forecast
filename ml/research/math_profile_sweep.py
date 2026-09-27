"""Sweep mathematical route/day/hour profile estimators on 2-month folds."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import load_labels, TYPE_OF_DOW  # noqa: E402


LEVEL_WEEKS = (1, 2, 3, 4, 6)
SHAPE_WEEKS = (2, 3, 4, 6, 8)
METHODS = ("median", "mean")
DAY_GROUPS = ("five", "seven")
CUTOFFS = ("2025-03-31", "2025-04-30", "2025-05-31",
           "2025-06-30", "2025-07-31", "2025-08-31")


def day_type(dates: pd.Series, code: np.ndarray, group: str) -> np.ndarray:
    dow = dates.dt.dayofweek.to_numpy()
    dtype = dow.copy() if group == "seven" else np.array([TYPE_OF_DOW[d] for d in dow])
    dtype = np.where((code == 1) & (dow < 5), 6, dtype)
    return np.where((code == 2) & (dow == 5), 4, dtype)


def fit_parts(hist: pd.DataFrame, grid: pd.DataFrame, cutoff: pd.Timestamp,
              group: str) -> tuple[dict, dict]:
    h = hist.copy()
    h["dtype"] = day_type(h.date, h.code.to_numpy(), group)
    g = grid.copy()
    g["dtype"] = day_type(g.date, g.code.to_numpy(), group)
    level_arrays = {}
    shape_arrays = {}
    for weeks, method in itertools.product(LEVEL_WEEKS, METHODS):
        sample = h[(h.date > cutoff - pd.Timedelta(weeks=weeks)) & (h.date <= cutoff)]
        daily = sample.groupby(["route", "dtype", "date"]).y.sum()
        level = daily.groupby(level=[0, 1]).agg(method).rename("level")
        level_arrays[(weeks, method)] = g.join(level, on=["route", "dtype"]).level.fillna(0).to_numpy()
    for weeks, method in itertools.product(SHAPE_WEEKS, METHODS):
        sample = h[(h.date > cutoff - pd.Timedelta(weeks=weeks)) & (h.date <= cutoff)].copy()
        total = sample.groupby(["route", "date"]).y.transform("sum")
        sample["share"] = sample.y / total.replace(0, np.nan)
        shape = sample.groupby(["route", "dtype", "hour"]).share.agg(method).fillna(0)
        shape = (shape / shape.groupby(level=[0, 1]).transform("sum").replace(0, np.nan)).fillna(0).rename("shape")
        shape_arrays[(weeks, method)] = g.join(shape, on=["route", "dtype", "hour"])["shape"].fillna(0).to_numpy()
    return level_arrays, shape_arrays


def main() -> None:
    labels = load_labels()
    calendar = pd.read_csv("data/external/calendar_ru_2025.csv", parse_dates=["date"])
    labels = labels.merge(calendar, on="date", validate="many_to_one")
    city = pd.read_csv("data/external/tram_monthly_city.csv").set_index("month").passenger_traffic.astype(float)
    totals = labels.assign(month=labels.date.dt.to_period("M").astype(str)).groupby("month").y.sum()
    shares = (totals / city.reindex(totals.index)).dropna()
    metrics = {}
    fold_details = []

    for cutoff_text in CUTOFFS:
        cutoff = pd.Timestamp(cutoff_text)
        target_end = (cutoff.to_period("M") + 2).end_time.normalize()
        hist = labels[labels.date <= cutoff]
        grid = labels[(labels.date > cutoff) & (labels.date <= target_end)].copy()
        actual = grid.y.to_numpy()
        months = grid.date.dt.to_period("M").astype(str).to_numpy()
        unique_months = list(dict.fromkeys(months))
        known = shares[shares.index <= cutoff.strftime("%Y-%m")].tail(2)
        slope = float(known.iloc[-1] - known.iloc[-2])
        target_totals = {}
        month_masks = {}
        for month in unique_months:
            months_ahead = pd.Period(month).ordinal - pd.Period(known.index[-1]).ordinal
            target_share = float(np.clip(known.iloc[-1] + 0.5 * slope * months_ahead, 0.25, 0.45))
            target_totals[month] = float(city.loc[month] * target_share)
            month_masks[month] = months == month
        fold_scores = {}
        for group in DAY_GROUPS:
            levels, shapes = fit_parts(hist, grid, cutoff, group)
            for lw, sw, lm, sm in itertools.product(LEVEL_WEEKS, SHAPE_WEEKS, METHODS, METHODS):
                key = f"{group}:L{lw}:{lm}:S{sw}:{sm}"
                prediction = levels[(lw, lm)] * shapes[(sw, sm)]
                base_error = float(np.abs(actual - prediction).sum())
                adjusted = prediction.copy()
                for month in unique_months:
                    mask = month_masks[month]
                    adjusted[mask] *= target_totals[month] / max(float(adjusted[mask].sum()), 1)
                city_error = float(np.abs(actual - adjusted).sum())
                denom = float(actual.sum())
                metrics.setdefault(key, {"base_error": 0.0, "city_error": 0.0, "target": 0.0,
                                         "per_fold": []})
                metrics[key]["base_error"] += base_error
                metrics[key]["city_error"] += city_error
                metrics[key]["target"] += denom
                metrics[key]["per_fold"].append({"cutoff": cutoff_text,
                                                 "base_score": 1 - base_error / denom,
                                                 "city_score": 1 - city_error / denom})
                fold_scores[key] = 1 - city_error / denom
        winner = max(fold_scores, key=fold_scores.get)
        fold_details.append({"cutoff": cutoff_text, "target_end": str(target_end.date()),
                             "best_city_config": winner, "best_city_score": fold_scores[winner],
                             "baseline_city_score": fold_scores["five:L2:median:S4:median"]})
        print(fold_details[-1], flush=True)

    ranking = []
    for key, values in metrics.items():
        ranking.append({"config": key, "base_score": 1 - values["base_error"] / values["target"],
                        "city_score": 1 - values["city_error"] / values["target"],
                        "folds": values["per_fold"]})
    ranking.sort(key=lambda item: item["city_score"], reverse=True)
    output = {"score": "1-WAPE; higher is better", "folds": fold_details, "ranking": ranking}
    Path("ml/research/math_profile_sweep.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(pd.DataFrame(ranking)[["config", "base_score", "city_score"]].head(15).round(5).to_string(index=False))


if __name__ == "__main__":
    main()
