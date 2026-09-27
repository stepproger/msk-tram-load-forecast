"""Проверки воспроизводимости и контрактов. Запуск из корня:
uv run --python 3.12 --with-requirements ml/requirements.txt --with pytest pytest ml/tests -q
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

SUB = Path("ml/submissions")
ART = Path("artifacts/forecast")
BEST = "HONEST2.csv"   # релиз: честная версия (ml/honest.py), LB 0.90257 (ml/release.py → FINAL.csv)


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def best():
    return pd.read_csv(SUB / BEST, sep=";", parse_dates=["date"])


def test_submission_grid(best):
    assert list(best.columns) == ["route", "date", "hour", "prediction"]
    assert len(best) == 14640 and not best.duplicated(["route", "date", "hour"]).any()
    assert set(best.route) == {1, 5, 7, 11, 12, 17, 25, 26, 28, 50}
    assert best.date.min() == pd.Timestamp("2025-11-01") and best.date.max() == pd.Timestamp("2025-12-31")
    assert (best.prediction >= 0).all() and best.prediction.dtype.kind == "i"


def test_network_events(best):
    r5 = best[best.route == 5].groupby("date").prediction.sum()
    assert (r5[r5.index < "2025-12-16"] == 0).all() and (r5[r5.index >= "2025-12-16"] > 0).all()
    r50 = best[best.route == 50].groupby("date").prediction.sum()
    assert r50[pd.Timestamp("2025-12-06")] > 2 * r50[pd.Timestamp("2025-11-08")]   # выходные 50: ремонт → восстановлено (E7: бустинг частично «размывает» ремонт)


def test_calendar(best):
    day = best[best.route != 5].groupby("date").prediction.sum()
    assert day[pd.Timestamp("2025-11-04")] < 0.7 * day[pd.Timestamp("2025-11-11")]   # праздник (вт) ≈ воскресенье
    assert day[pd.Timestamp("2025-11-01")] > day[pd.Timestamp("2025-11-08")]         # рабочая суббота > обычной


def test_pipeline_reproduces_best():
    final = SUB / "FINAL.csv"
    if not final.exists():
        pytest.skip("запустите ml/run_pipeline.py")
    assert md5(final) == md5(SUB / BEST)


def test_artifacts_contract():
    h = pd.read_parquet(ART / "hourly.parquet")
    assert list(h.columns)[:6] == ["route", "date", "hour", "yhat", "q10", "q90"] and len(h) == 14640
    assert "weather_factor" in h.columns and h.weather_factor.between(0.8, 1.1).all()
    assert (h.q10 <= h.yhat + 1e-6).all() and (h.yhat <= h.q90 + 1e-6).all()
    a = pd.read_parquet(ART / "actuals_hourly.parquet")
    assert {"route", "date", "hour", "boardings", "trams"} <= set(a.columns)
    s = pd.read_parquet(ART / "stop_shares.parquet").groupby(["route", "hour"]).share.sum()
    assert ((s - 1).abs() < 1e-3).all()
    factors = json.loads((ART / "factors.json").read_text(encoding="utf-8"))
    assert all({"id", "label", "default", "min", "max", "source"} <= set(f) for f in factors)


def test_recommendations_keep_fleet_per_hour():
    r = pd.read_parquet(ART / "recommendations.parquet")
    g = r.groupby(["date", "hour"])[["trams_plan", "trams_needed"]].sum()
    assert (g.trams_needed <= g.trams_plan + 1e-9).all()     # парк в каждый час не растёт
    assert ((r.trams_needed >= 0.6 * r.trams_plan - 1) | (r.trams_plan == 0)).all()


def test_geo_covers_all_routes():
    st = pd.read_parquet("artifacts/geo/stops.parquet")
    assert set(st.route) == {1, 5, 7, 11, 12, 17, 25, 26, 28, 50}
    assert st.lat.between(55.4, 56.1).all() and st.lon.between(37.2, 38.0).all()


def test_target_matches_labels():
    p = Path("data/interim/target_hourly.parquet")
    if not p.exists():
        pytest.skip("запустите ml/ingest/build_target.py")
    ours = pd.read_parquet(p)
    lab = pd.concat([pd.read_csv(f, sep=";") for f in Path("dataset/labels").glob("labels_day_*.csv")])
    assert abs(ours.boardings.sum() - lab.boardings.sum()) <= 5   # 59.67 млн посадок, расхождение ≤ 5


def test_components_sum_to_final():
    c = pd.read_parquet(ART / "components.parquet")
    parts = c[["profile", "calendar", "network", "season", "calibration", "ml"]].sum(axis=1)
    assert (parts - c.final).abs().max() < 1e-2
    f = pd.read_csv(SUB / "FINAL.csv", sep=";")
    assert abs(c.final.sum() - f.prediction.sum()) < 0.01 * f.prediction.sum()
