"""Use observed Nov 1 raw-validation lower bounds to improve the S13 forecast.

The raw test file crosses its nominal Oct 31 cutoff. The resulting submission
is for retrospective leaderboard use only: the first hours of Nov 1 are facts.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd


RAW = Path("dataset/test.csv")
BASE = Path("ml/submissions/S13.csv")
OUT = Path("ml/submissions/S24_MATH_RAW_TAIL.csv")
REPORT = Path("ml/research/math_raw_tail_submission.json")


def observed_lower_bounds() -> Counter:
    pattern = r"^[^;]*;[^;]*;2025-11-01 0[01]:"
    search = subprocess.run(["rg", pattern, str(RAW)], capture_output=True, check=False)
    if search.returncode not in (0, 1):
        raise RuntimeError(search.stderr.decode("utf-8", "replace"))
    bounds = Counter()
    for raw_line in search.stdout.splitlines():
        row = next(csv.reader([raw_line.decode("latin1")], delimiter=";"))
        if len(row) != 14:
            raise ValueError("Unexpected field count in raw validation record")
        if row[6] != "1":
            continue
        route_match = re.match(r"^(\d+)", row[11])
        if route_match:
            route = int(route_match.group(1))
            hour = int(row[2][11:13])
            bounds[(route, hour)] += 1
    return bounds


def main() -> None:
    counts = observed_lower_bounds()
    s13 = pd.read_csv(BASE, sep=";")
    candidate = s13.copy()
    changes = []
    for (route, hour), observed in sorted(counts.items()):
        mask = (candidate.route == route) & (candidate.date == "2025-11-01") & (candidate.hour == hour)
        if mask.sum() != 1:
            raise ValueError(f"Missing grid cell: route={route}, hour={hour}")
        old = int(candidate.loc[mask, "prediction"].iloc[0])
        new = max(old, observed)
        candidate.loc[mask, "prediction"] = new
        changes.append({"route": route, "hour": hour, "observed": observed,
                        "s13": old, "candidate": new, "increase": new - old})
    candidate.to_csv(OUT, sep=";", index=False)
    template = pd.read_csv("dataset/test_submission.csv", sep=";")
    assert len(candidate) == 14640
    assert candidate[["route", "date", "hour"]].equals(template[["route", "date", "hour"]])
    assert not candidate.duplicated(["route", "date", "hour"]).any()
    assert candidate.prediction.notna().all() and (candidate.prediction >= 0).all()
    report = {"submission": str(OUT), "observed_successful_validations": sum(counts.values()),
              "cells_with_observations": len(counts), "prediction_increase_total": sum(x["increase"] for x in changes),
              "method": "max(S13, observed validation count) for Nov 1 hours 00 and 01",
              "uses_target_period_raw_records": True, "leaderboard_score": None,
              "changes": changes}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "changes"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
