"""Reconstruct a documented proxy for the new tram route 5 in December 2025.

This is an estimate, not recovered ticket-validation records.  Route 5 and
its predecessors 9/c510 have no usable hourly labels in the supplied archive.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DONORS = (7, 50)
SOURCE = ROOT / "ml/submissions/S26_MATH_ROUTE5_OFFICIAL.csv"
ESTIMATES = ROOT / "data/derived/route5_hourly_estimates_2025.csv"
REPORT = ROOT / "ml/research/route5_reconstruction_report.json"
START = "2025-12-16"
FIRST_WEEK_END = "2025-12-22"
END = "2025-12-31"


def stop_names(route: int) -> set[str]:
    data = json.loads((ROOT / f"data/external/osm/tram_{route}.json").read_text(encoding="utf-8"))
    return {
        element["tags"]["name"].casefold().strip()
        for element in data["elements"]
        if element.get("tags", {}).get("railway") == "tram_stop"
        and element.get("tags", {}).get("name")
    }


def main() -> None:
    target = stop_names(5)
    overlaps = {}
    for route in (1, 7, 11, 12, 17, 25, 26, 28, 50):
        common = target & stop_names(route)
        overlaps[route] = len(common)

    labels = pd.concat(
        [pd.read_csv(path, sep=";") for path in (ROOT / "dataset/labels").glob("labels_day_*.csv")],
        ignore_index=True,
    )
    labels["date"] = pd.to_datetime(labels.date)
    labels["weekday"] = labels.date.dt.dayofweek.lt(5)
    daily = labels.groupby(["route", "date"]).boardings.transform("sum")
    labels["share"] = labels.boardings / daily.replace(0, np.nan)
    fit = labels[labels.date.between("2025-08-01", "2025-08-31") & labels.weekday]
    hold = labels[labels.date.between("2025-09-01", "2025-09-30") & labels.weekday]
    holdouts = {}
    for route, other in ((7, 50), (50, 7)):
        actual = hold[hold.route.eq(route)].copy()
        for label, source in (("neighbor", fit[fit.route.eq(other)]), ("network", fit[~fit.route.eq(route)])):
            shape = source.groupby("hour").share.mean()
            predicted = actual.groupby("date").boardings.transform("sum") * actual.hour.map(shape / shape.sum())
            error = np.abs(predicted - actual.boardings).sum() / actual.boardings.sum()
            holdouts[f"route_{route}_{label}"] = round(float(error), 4)

    baseline = pd.read_csv(SOURCE, sep=";")
    template = pd.read_csv(ROOT / "dataset/test_submission.csv", sep=";")
    assert baseline[["route", "date", "hour"]].equals(template[["route", "date", "hour"]])
    route5 = baseline[baseline.route.eq(5)].copy()
    first_week = route5.date.between(START, FIRST_WEEK_END)
    late_december = route5.date.between("2025-12-23", END)
    first_week_total = int(route5.loc[first_week, "prediction"].sum())
    assert first_week_total == 39866
    scenarios = {"S31_ROUTE5_LATE90": 0.90, "S32_ROUTE5_LATE80": 0.80}
    scenario_totals = {}
    for name, factor in scenarios.items():
        candidate = baseline.copy()
        mask = candidate.route.eq(5) & candidate.date.between("2025-12-23", END)
        candidate.loc[mask, "prediction"] = np.rint(candidate.loc[mask, "prediction"] * factor).astype(int)
        assert candidate.loc[~mask, "prediction"].equals(baseline.loc[~mask, "prediction"])
        path = ROOT / f"ml/submissions/{name}.csv"
        candidate.to_csv(path, sep=";", index=False)
        scenario_totals[name] = int(candidate.loc[candidate.route.eq(5), "prediction"].sum())

    route5["scenario_low"] = np.where(late_december, np.rint(route5.prediction * .8), route5.prediction).astype(int)
    route5["scenario_high"] = np.where(late_december, route5.prediction, route5.prediction).astype(int)
    route5["basis"] = np.select(
        [route5.date.lt(START), first_week, late_december],
        ["not_in_service", "first_week_aggregate_calibrated", "unobserved_extrapolation"],
        default="unknown",
    )
    route5 = route5.rename(columns={"prediction": "central_estimate"})
    ESTIMATES.parent.mkdir(parents=True, exist_ok=True)
    route5[["route", "date", "hour", "central_estimate", "scenario_low", "scenario_high", "basis"]].to_csv(
        ESTIMATES, sep=";", index=False
    )

    report = {
        "data_type": "modeled hourly estimates, not actual validations",
        "official_start": START,
        "official_first_week_trips_approx": 40000,
        "s26_first_week_estimate": first_week_total,
        "s26_december_total_estimate": int(route5.central_estimate.sum()),
        "scenario_december_totals": scenario_totals,
        "osm_shared_stops": overlaps,
        "historical_shape_transfer_wape_with_known_daily_totals": holdouts,
        "sources": [
            "https://transport.mos.ru/mostrans/all_news/127652",
            "https://transport.mos.ru/mostrans/all_news/127782",
            "https://t.me/DtRoad/56578",
        ],
        "note": "The >160k trips and >7.5k runs announced after one month include January; they cannot determine December's exact hourly or daily counts. Late-December factors are sensitivity scenarios, not inferred observations.",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
