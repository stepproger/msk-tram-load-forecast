"""ML-модель дрейфа: LightGBM учится, как меняется отношение «факт / профиль» с горизонтом прогноза.

Идея. Профиль (уровень × форма суток на момент среза) хорошо описывает «сегодня», но за 61 день уровень дрейфует
(сезон, тренд, праздники, погода). Раньше дрейф закрывали ручные множители (сезон ×1.022, конец года ×0.85).
Здесь дрейф выучивается: по десяткам исторических срезов строим пары (прогноз профиля на h дней вперёд, факт)
и учим LightGBM предсказывать r = факт / профиль по признакам, известным в момент прогноза:
горизонт, маршрут, час, тип дня, праздник, погода целевого часа, тренд уровня маршрута на срезе.
Итог: ŷ = профиль × r̂ — прямой (direct) многошаговый прогноз без рекурсии.

  uv run --python 3.12 --with-requirements ml/requirements.txt --with lightgbm,scikit-learn python ml/drift_model.py cv
  ... python ml/drift_model.py submit D1
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import events as ev  # noqa: E402
from core import TYPE_OF_DOW, fit_profile, load_labels, load_weather, wape_score  # noqa: E402

H = 61
HIST_HOL = set(pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08",
                               "2025-05-01", "2025-05-02", "2025-05-08", "2025-05-09", "2025-06-12", "2025-06-13"]))
ALL_HOL = HIST_HOL | set(ev.HOLIDAYS)
FEATS = ["route", "hour", "dt5", "dow", "hol", "h_days", "t", "prcp", "snow", "trend_2_6", "trend_1_4"]
PARAMS = dict(objective="l1", n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)

y = load_labels()
y["dt5"] = np.where(y.date.isin(HIST_HOL), 6, y.dtype)
W = load_weather()[["date", "hour", "t", "prcp", "snow"]]


def profile_windows(hist: pd.DataFrame, cut: pd.Timestamp, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Прогноз профиля со среза cut на даты dates (праздник → тип «Вс») + признаки тренда на срезе."""
    lv, sh = fit_profile(hist.assign(dtype=hist.dt5), cut)
    g = pd.MultiIndex.from_product([sorted(hist.route.unique()), dates, range(24)],
                                   names=["route", "date", "hour"]).to_frame(index=False)
    g["hol"] = g.date.isin(ALL_HOL).astype(int)
    g["dow"] = g.date.dt.dayofweek
    g["dt5"] = np.where(g.hol == 1, 6, g.dow.map(TYPE_OF_DOW))
    g = g.join(lv.rename_axis(["route", "dt5"]), on=["route", "dt5"]).join(sh.rename_axis(["route", "dt5", "hour"]),
                                                                          on=["route", "dt5", "hour"])
    g["base"] = g.level.fillna(0) * g["shape"].fillna(0)
    g["h_days"] = (g.date - cut).dt.days
    d = hist[hist.date <= cut].groupby(["route", "date"]).y.sum().reset_index()
    wd = d[d.date.dt.dayofweek < 5]
    def lvl(weeks):
        return wd[wd.date > cut - pd.Timedelta(weeks=weeks)].groupby("route").y.median()
    g["trend_2_6"] = g.route.map(lvl(2) / lvl(6))
    g["trend_1_4"] = g.route.map(lvl(1) / lvl(4))
    return g.merge(W, on=["date", "hour"], how="left")


def training_set(last_target: pd.Timestamp, first_cut="2025-02-02") -> pd.DataFrame:
    parts = []
    for cut in pd.date_range(first_cut, last_target - pd.Timedelta(days=7), freq="W-SUN"):
        end = min(cut + pd.Timedelta(days=H), last_target)
        g = profile_windows(y[y.date <= cut], cut, pd.date_range(cut + pd.Timedelta(days=1), end))
        parts.append(g.merge(y[["route", "date", "hour", "y"]], on=["route", "date", "hour"]))
    t = pd.concat(parts)
    return t[t.base > 20]


def fit(train: pd.DataFrame) -> lgb.LGBMRegressor:
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(train[FEATS], (train.y / train.base).clip(0, 3), sample_weight=train.base, categorical_feature=["route"])
    return m


def cv():
    rows = []
    for cut, end in [("2025-06-29", "2025-08-31"), ("2025-07-27", "2025-09-30"), ("2025-08-31", "2025-10-31"),
                     ("2025-09-28", "2025-10-31")]:
        cut, end = pd.Timestamp(cut), pd.Timestamp(end)
        m = fit(training_set(cut))
        te = profile_windows(y[y.date <= cut], cut, pd.date_range(cut + pd.Timedelta(days=1), end))
        te = te.merge(y[["route", "date", "hour", "y"]], on=["route", "date", "hour"])
        r = np.clip(m.predict(te[FEATS]), 0.3, 2.0)
        rows.append(dict(fold=f"{cut:%d.%m}→{end:%d.%m}", profile=wape_score(te.y, te.base),
                         profile_x103=wape_score(te.y, te.base * 1.03), drift_ml=wape_score(te.y, te.base * r),
                         mean_ratio=float(np.average(r, weights=te.base))))
        print(rows[-1], flush=True)
    r = pd.DataFrame(rows)
    print(r.round(4).to_string(index=False))
    print("mean:", r[["profile", "profile_x103", "drift_ml"]].mean().round(4).to_dict())
    return r


def ratio_nov_dec(forecast: pd.DataFrame) -> np.ndarray:
    """Обучение на всех срезах с фактами до 31.10 → множитель r̂ для каждой ячейки ноября–декабря."""
    m = fit(training_set(y.date.max()))
    cut = y.date.max()
    g = profile_windows(y, cut, pd.date_range("2025-11-01", "2025-12-31"))
    g["r"] = np.clip(m.predict(g[FEATS]), 0.3, 2.0)
    imp = pd.Series(m.feature_importances_, FEATS).sort_values(ascending=False)
    print("importance:", imp.to_dict())
    return forecast.merge(g[["route", "date", "hour", "r"]], on=["route", "date", "hour"], how="left").r.fillna(1.0).values


if __name__ == "__main__":
    if sys.argv[1] == "cv":
        cv()
