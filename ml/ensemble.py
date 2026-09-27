"""Ансамбль: профиль (с событиями сети и калибровкой) + LightGBM L1 на всей истории (календарь + погода).

CV (ml/research/ensemble_cv.py, 5 временных фолдов по 1–2 мес.): вариант «уровень от профиля, структура от бустинга»
с весом 0.3 лучше профиля во ВСЕХ пяти фолдах (в среднем 0.8556 против 0.8505).

Кандидаты (база — лучший LB-сабмит, маршрут 5 не трогаем — у бустинга нет его истории):
  E1 — 0.7·база + 0.3·бустинг, бустинг перемасштабирован к сумме базы по (маршрут, месяц, выходной/будний)
  E2 — то же с весом 0.5
  E3 — 0.7·база + 0.3·бустинг без перемасштабирования (бустинг влияет и на уровень)

  uv run --python 3.12 --with-requirements ml/requirements.txt --with lightgbm,scikit-learn python ml/ensemble.py [BASE.csv]
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import events as ev  # noqa: E402
from core import load_labels, load_weather, write_submission  # noqa: E402

FEATS = ["route", "hour", "dow", "dt5", "hol", "t", "prcp", "snow"]
HIST_HOL = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08",
                           "2025-05-01", "2025-05-02", "2025-05-08", "2025-05-09", "2025-06-12", "2025-06-13"])
DT5 = {0: 0, 1: 1, 2: 1, 3: 1, 4: 4, 5: 5, 6: 6}


def daylight(dates: pd.Series, hours: pd.Series, lat=55.75, lon=37.62):
    """Долгота дня (ч) и «темно в этот час» — астрономия, известна на любой горизонт."""
    doy = dates.dt.dayofyear.values
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    day_len = 2 * np.degrees(np.arccos(np.clip(-np.tan(np.radians(lat)) * np.tan(decl), -1, 1))) / 15
    noon = 12 - lon / 15 + 3
    h = hours.values + 0.5
    return day_len, ((h < noon - day_len / 2) | (h > noon + day_len / 2)).astype(int)


EXTRA = ["day_len", "dark", "pre_hol"]


def climatology_weather(grid: pd.DataFrame) -> pd.DataFrame:
    """Погода прогнозного периода = климатическая норма 2015–2024 по (месяц, день, час) — без фактов будущего."""
    c = pd.read_csv("data/external/weather_climatology_2015_2024.csv")
    g = grid[["date", "hour"]].drop_duplicates().assign(month=lambda d: d.date.dt.month, day=lambda d: d.date.dt.day)
    return g.merge(c, on=["month", "day", "hour"], how="left")[["date", "hour", "t", "prcp", "snow"]]


def gbm_forecast(grid: pd.DataFrame, extra: bool = False, clim: bool = False) -> np.ndarray:
    w = load_weather()[["date", "hour", "t", "prcp", "snow"]]
    y = load_labels().merge(w, on=["date", "hour"], how="left")
    y["hol"], y["dow"] = y.date.isin(HIST_HOL).astype(int), y.date.dt.dayofweek
    y["dt5"] = np.where(y.hol == 1, 6, y.dtype)
    hol_all = set(HIST_HOL) | set(ev.HOLIDAYS)
    feats = FEATS + EXTRA if extra else FEATS
    y["day_len"], y["dark"] = daylight(y.date, y.hour)
    y["pre_hol"] = y.date.add(pd.Timedelta(days=1)).isin(hol_all).astype(int)
    m = lgb.LGBMRegressor(objective="l1", n_estimators=600, learning_rate=0.05, num_leaves=63, verbose=-1)
    m.fit(y[feats], y.y, categorical_feature=["route"])
    g = grid.merge(climatology_weather(grid) if clim else w, on=["date", "hour"], how="left")
    g["hol"] = g.date.isin(ev.HOLIDAYS).astype(int)
    g["dow"] = np.where(g.date.isin(ev.WORKING_SATURDAYS), 4, g.date.dt.dayofweek)
    g["dt5"] = np.where(g.hol == 1, 6, g.dow.map(DT5))
    g["day_len"], g["dark"] = daylight(g.date, g.hour)
    g["pre_hol"] = g.date.add(pd.Timedelta(days=1)).isin(hol_all).astype(int)
    return np.clip(m.predict(g[feats]), 0, None)


if __name__ == "__main__":
    base_file = sys.argv[1] if len(sys.argv) > 1 else "ml/submissions/S31_ROUTE5_LATE90.csv"
    b = pd.read_csv(base_file, sep=";", parse_dates=["date"])
    b["gbm"] = gbm_forecast(b[["route", "date", "hour"]])
    off = b.date.isin(ev.HOLIDAYS) | ((b.date.dt.dayofweek >= 5) & ~b.date.isin(ev.WORKING_SATURDAYS))
    key = [b.route, b.date.dt.month, off]
    scale = b.groupby(key).prediction.transform("sum") / b.groupby(key).gbm.transform("sum").replace(0, np.nan)
    b["gbm_scaled"] = b.gbm * scale.fillna(0)
    old = b.route != 5
    variants = [("E1", "gbm_scaled", 0.3), ("E2", "gbm_scaled", 0.5), ("E3", "gbm", 0.3),
                ("E4", "gbm_scaled", 0.7), ("E5", "gbm_scaled", 1.0), ("E6", "gbm", 0.5), ("E7", "mix", 0.5)]
    b["mix"] = 0.5 * b.gbm_scaled + 0.5 * b.gbm      # половина уровня от бустинга
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    for name, col, wgt in variants:
        if only and name not in only:
            continue
        pred = np.where(old, (1 - wgt) * b.prediction + wgt * b[col], b.prediction)
        write_submission(pd.DataFrame({"route": b.route, "date": b.date, "hour": b.hour, "pred": pred}), name)
        diff = np.abs(np.round(pred) - b.prediction).sum()
        print(f"{name}: total {pred.sum():,.0f} (база {b.prediction.sum():,}), Σ|Δ| = {diff:,.0f}")
