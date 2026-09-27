"""Локальный прокси лидерборда: полный рецепт «профиль + LightGBM» на исторических срезах без утечек.

Зачем: загрузок на платформу мало. Кандидат грузим, только если прокси показывает прирост.
Проверка прокси: ранжирует ли он уже известные сабмиты так же, как лидерборд
(LB: E7 0.90017 > E6 0.89956 > E2 0.89939 > E3 0.89916 > E4 0.89860 > E1 0.89840 > E5 0.89405 > база S31 0.89329).

  uv run --python 3.12 --with-requirements ml/requirements.txt python ml/research/proxy_lb.py [features]
"""
import json
import sys

sys.path.insert(0, "ml")
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from core import fit_profile, load_labels, load_weather, wape_score  # noqa: E402

HOL = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08",
                      "2025-05-01", "2025-05-02", "2025-05-08", "2025-05-09", "2025-06-12", "2025-06-13"])
FOLDS = [("2025-03-02", "2025-04-30"), ("2025-04-30", "2025-06-30"), ("2025-06-30", "2025-08-31"),
         ("2025-08-31", "2025-10-31"), ("2025-09-28", "2025-10-31")]
LB = {"E7": 0.90017, "E6": 0.89956, "E2": 0.89939, "E3": 0.89916, "E4": 0.89860, "E1": 0.89840, "E5": 0.89405,
      "base": 0.89329}
BASE_FEATS = ["route", "hour", "dow", "dt5", "hol", "t", "prcp", "snow"]


def daylight(dates: pd.Series, hours: pd.Series, lat=55.75, lon=37.62) -> tuple[np.ndarray, np.ndarray]:
    """Долгота дня (ч) и признак «темно в этот час» по астрономической формуле (известны на любой горизонт)."""
    doy = dates.dt.dayofyear.values
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    cosw = -np.tan(np.radians(lat)) * np.tan(decl)
    day_len = 2 * np.degrees(np.arccos(np.clip(cosw, -1, 1))) / 15
    noon = 12 - lon / 15 + 3          # солнечный полдень по МСК (UTC+3), без уравнения времени
    sunrise, sunset = noon - day_len / 2, noon + day_len / 2
    h = hours.values + 0.5
    return day_len, ((h < sunrise) | (h > sunset)).astype(int)


def load():
    y = load_labels().merge(load_weather()[["date", "hour", "t", "prcp", "snow"]], on=["date", "hour"], how="left")
    y["hol"], y["dow"] = y.date.isin(HOL).astype(int), y.date.dt.dayofweek
    y["dt5"] = np.where(y.hol == 1, 6, y.dtype)
    y["day_len"], y["dark"] = daylight(y.date, y.hour)
    y["pre_hol"] = y.date.add(pd.Timedelta(days=1)).isin(HOL).astype(int)
    return y


def run(feats=BASE_FEATS, params=None, seeds=(0,), verbose=True, clim=False):
    params = params or dict(objective="l1", n_estimators=600, learning_rate=0.05, num_leaves=63, verbose=-1)
    y, rows = load(), []
    for cut, end in FOLDS:
        cut, end = pd.Timestamp(cut), pd.Timestamp(end)
        tr, te = y[y.date <= cut], y[(y.date > cut) & (y.date <= end)].copy()
        if clim:   # погода прогнозного окна = климатическая норма 2015–2024, а не факт
            c = pd.read_csv("data/external/weather_climatology_2015_2024.csv")
            te = te.drop(columns=["t", "prcp", "snow"]).assign(month=te.date.dt.month, day=te.date.dt.day).merge(
                c, on=["month", "day", "hour"], how="left").drop(columns=["month", "day"])
        lv, sh = fit_profile(tr.assign(dtype=tr.dt5), cut)
        te["p"] = (te.join(lv, on=["route", "dt5"]).level.fillna(0)
                   * te.join(sh, on=["route", "dt5", "hour"])["shape"].fillna(0)).values
        g = np.zeros(len(te))
        for s in seeds:
            m = lgb.LGBMRegressor(**params, random_state=s).fit(tr[feats], tr.y, categorical_feature=["route"])
            g += np.clip(m.predict(te[feats]), 0, None) / len(seeds)
        te["g"] = g
        off = te.dt5 >= 5
        key = [te.route, te.date.dt.month, off]
        te["gs"] = te.g * (te.groupby(key).p.transform("sum") / te.groupby(key).g.transform("sum").replace(0, np.nan)).fillna(0)
        v = {"base": te.p, "E1": .7 * te.p + .3 * te.gs, "E2": .5 * te.p + .5 * te.gs, "E3": .7 * te.p + .3 * te.g,
             "E4": .3 * te.p + .7 * te.gs, "E5": te.gs, "E6": .5 * te.p + .5 * te.g,
             "E7": .5 * te.p + .25 * te.gs + .25 * te.g}
        r = {"fold": f"{cut:%d.%m}→{end:%d.%m}", **{k: wape_score(te.y, x) for k, x in v.items()}}
        rows.append(r)
    d = pd.DataFrame(rows).set_index("fold")
    d.loc["среднее"] = d.mean()
    rho = {f: spearmanr([LB[k] for k in LB], [d.loc[f, k] for k in LB]).statistic for f in d.index}
    if verbose:
        print(d.T.round(4).to_string())
        print("Spearman с LB:", {k: round(v, 2) for k, v in rho.items()})
    return d, rho


if __name__ == "__main__":
    feats = sys.argv[1].split(",") if len(sys.argv) > 1 else BASE_FEATS
    d, rho = run(feats)
    json.dump({"scores": d.round(5).reset_index().to_dict("records"), "spearman": rho},
              open("ml/research/proxy_lb.json", "w"), ensure_ascii=False, indent=2, default=float)
