"""Compare profile shapes while fixing each route-month total to baseline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from math_profile_sweep import CUTOFFS, LEVEL_WEEKS, SHAPE_WEEKS, METHODS, DAY_GROUPS, fit_parts

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import load_labels  # noqa: E402


def main() -> None:
    labels = load_labels()
    calendar = pd.read_csv("data/external/calendar_ru_2025.csv", parse_dates=["date"])
    labels = labels.merge(calendar, on="date", validate="many_to_one")
    results = {}
    for cutoff_text in CUTOFFS:
        cutoff = pd.Timestamp(cutoff_text)
        end = (cutoff.to_period("M") + 2).end_time.normalize()
        grid = labels[(labels.date > cutoff) & (labels.date <= end)].copy()
        hist = labels[labels.date <= cutoff]
        actual = grid.y.to_numpy()
        route = grid.route.to_numpy()
        month = grid.date.dt.to_period("M").astype(str).to_numpy()
        parts = {group: fit_parts(hist, grid, cutoff, group) for group in DAY_GROUPS}
        level, shape = parts["five"]
        baseline = level[(2, "median")] * shape[(4, "median")]
        for group in DAY_GROUPS:
            levels, shapes = parts[group]
            for lw in LEVEL_WEEKS:
                for lm in METHODS:
                    for sw in SHAPE_WEEKS:
                        for sm in METHODS:
                            key = f"{group}:L{lw}:{lm}:S{sw}:{sm}"
                            candidate = levels[(lw, lm)] * shapes[(sw, sm)]
                            candidate = candidate.copy()
                            for r in np.unique(route):
                                for m in np.unique(month):
                                    mask = (route == r) & (month == m)
                                    candidate[mask] *= baseline[mask].sum() / max(candidate[mask].sum(), 1)
                            score = 1 - float(np.abs(actual - candidate).sum()) / float(actual.sum())
                            by_route = {str(r): float(1 - np.abs(actual[route == r] - candidate[route == r]).sum() /
                                                     max(actual[route == r].sum(), 1)) for r in np.unique(route)}
                            results.setdefault(key, []).append({"cutoff": cutoff_text, "score": score,
                                                                 "by_route": by_route,
                                                                 "error": float(np.abs(actual - candidate).sum()),
                                                                 "target": float(actual.sum())})
        print(cutoff_text, "done", flush=True)

    ranking = []
    for config, folds in results.items():
        total_error = sum(f["error"] for f in folds)
        total_target = sum(f["target"] for f in folds)
        ranking.append({"config": config, "pooled_score": 1 - total_error / total_target,
                        "latest_score": folds[-1]["score"], "folds": folds})
    ranking.sort(key=lambda r: r["pooled_score"], reverse=True)
    output = {"metric": "1-WAPE; higher is better", "normalization": "fixed baseline total per route-month",
              "ranking": ranking}
    Path("ml/research/math_shape_sweep.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(pd.DataFrame(ranking)[["config", "pooled_score", "latest_score"]].head(20).round(5).to_string(index=False))
    control = next(r for r in ranking if r["config"] == "five:L2:median:S4:median")
    print("control", control["pooled_score"], control["latest_score"])


if __name__ == "__main__":
    main()
