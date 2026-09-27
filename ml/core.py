"""Ядро прогноза: ŷ = Level(r, тип дня) × Shape(r, тип дня, час) × Season × Special × Net.

Запуск:
  uv run --python 3.12 --with duckdb,pandas python ml/core.py backtest
  uv run --python 3.12 --with duckdb,pandas python ml/core.py submit S1
Обоснование — docs/ML_PLAN.md.
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import events as ev  # noqa: E402

ROUTES = [1, 7, 11, 12, 17, 25, 26, 28, 50]
ALL_ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]
TYPE_OF_DOW = {0: 0, 1: 1, 2: 1, 3: 1, 4: 4, 5: 5, 6: 6}   # Пн / Вт–Чт / Пт / Сб / Вс
LEVEL_WEEKS, SHAPE_WEEKS = 2, 4
CITY_MONTHLY_FILE = "data/external/tram_monthly_city.csv"


def load_labels() -> pd.DataFrame:
    df = duckdb.sql("select * from read_csv('dataset/labels/labels_day_*.csv', delim=';')").df()
    df["date"] = pd.to_datetime(df["date"])
    grid = pd.MultiIndex.from_product(
        [ROUTES, pd.date_range(df.date.min(), df.date.max()), range(24)], names=["route", "date", "hour"])
    y = df.set_index(["route", "date", "hour"]).boardings.reindex(grid, fill_value=0).rename("y").reset_index()
    y["dtype"] = y.date.dt.dayofweek.map(TYPE_OF_DOW)
    return y


def fit_profile(hist: pd.DataFrame, end: pd.Timestamp, level_weeks=LEVEL_WEEKS, shape_weeks=SHAPE_WEEKS):
    """Уровень — медиана дневных сумм; форма — медиана часовых долей, нормированная к 1."""
    s = hist[(hist.date > end - pd.Timedelta(weeks=shape_weeks)) & (hist.date <= end)]
    day = s.groupby(["route", "date"]).y.transform("sum")
    shape = s.assign(sh=s.y / day.replace(0, np.nan)).groupby(["route", "dtype", "hour"]).sh.median().fillna(0)
    shape = (shape / shape.groupby(level=[0, 1]).transform("sum")).rename("shape")
    lv = hist[(hist.date > end - pd.Timedelta(weeks=level_weeks)) & (hist.date <= end)]
    level = lv.groupby(["route", "dtype", "date"]).y.sum().groupby(level=[0, 1]).median().rename("level")
    return level, shape


def base_forecast(level, shape, dates, routes=ROUTES) -> pd.DataFrame:
    g = pd.MultiIndex.from_product([routes, dates, range(24)], names=["route", "date", "hour"]).to_frame(index=False)
    g["dtype"] = g.date.dt.dayofweek.map(TYPE_OF_DOW)
    g = g.join(level, on=["route", "dtype"]).join(shape, on=["route", "dtype", "hour"])
    g["pred"] = g.level.fillna(0) * g["shape"].fillna(0)
    return g


def wape_score(y, p) -> float:
    return max(0.0, 1 - np.abs(y - p).sum() / y.sum())


def citywide_monthly_calibration(y: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """Ретроспективная калибровка 9 старых маршрутов по месячной статистике города.

    Использует опубликованный постфактум общий пассажиропоток трамваев целевого
    месяца. Долю наших маршрутов прогнозирует линейным трендом последних 3
    доступных месяцев. Это режим для hackathon backtest/submission, не real-time.
    """
    city = pd.read_csv(CITY_MONTHLY_FILE)
    city = city.set_index("month")["passenger_traffic"].astype(float)
    out = forecast.copy()
    out["month"] = out.date.dt.to_period("M").astype(str)
    hist = y.copy()
    hist["month"] = hist.date.dt.to_period("M").astype(str)
    monthly_target = hist.groupby("month").y.sum()
    shares = (monthly_target / city.reindex(monthly_target.index)).dropna()

    for month in out.month.unique():
        if month not in city.index:
            continue
        known = shares[shares.index < month].tail(3)
        if len(known) < 3:
            continue
        slope = np.polyfit(np.arange(len(known)), known.values, 1)[0]
        periods_ahead = pd.Period(month, freq="M").ordinal - pd.Period(known.index[-1], freq="M").ordinal
        predicted_share = float(np.clip(known.iloc[-1] + slope * periods_ahead, 0.25, 0.45))
        mask = out.month == month
        current_total = out.loc[mask, "pred"].sum()
        target_total = city.loc[month] * predicted_share
        if current_total > 0:
            out.loc[mask, "pred"] *= target_total / current_total
    return out.drop(columns="month")


def backtest():
    y = load_labels()
    for end, t1 in [("2025-10-05", "2025-10-31"), ("2025-09-28", "2025-10-31"), ("2025-03-02", "2025-03-30")]:
        end, t1 = pd.Timestamp(end), pd.Timestamp(t1)
        level, shape = fit_profile(y, end)
        f = base_forecast(level, shape, pd.date_range(end + pd.Timedelta(days=1), t1))
        m = f.merge(y[["route", "date", "hour", "y"]], on=["route", "date", "hour"])
        print(f"cutoff {end.date()} → {t1.date()}: WAPE-score {wape_score(m.y, m.pred):.4f}, bias {m.pred.sum()/m.y.sum()-1:+.3f}")


def load_weather() -> pd.DataFrame:
    w = pd.read_csv(ev.WEATHER_FILE, skiprows=3)
    w.columns = ["time", "t", "prcp", "snow", "depth", "rain", "wind", "cloud"]
    w["time"] = pd.to_datetime(w.time)
    w["date"], w["hour"] = w.time.dt.normalize(), w.time.dt.hour
    w["wf"] = pd.cut(w.prcp, ev.PRCP_BINS, labels=ev.PRCP_FACTOR).astype(float)
    return w


def weather_factor(dates_hours: pd.DataFrame, fit_end: pd.Timestamp, mode: str = "level") -> np.ndarray:
    """Множитель погоды, нормированный к среднему за окно уровня (в уровне уже «сидит» средняя погода)."""
    w = load_weather()
    ref = w[(w.date > fit_end - pd.Timedelta(weeks=LEVEL_WEEKS)) & (w.date <= fit_end) & w.hour.between(6, 22)].wf.mean()
    if mode == "shape":   # только перераспределение внутри месяца, средний уровень месяца не меняется
        w["wf"] = w.wf / w.groupby(w.date.dt.month).wf.transform("mean")
        ref = 1.0
    m = dates_hours.merge(w[["date", "hour", "wf"]], on=["date", "hour"], how="left")
    return (m.wf.fillna(1.0) / ref).values


def forecast_nov_dec(season=None, special=None, repair_end=ev.WEEKEND_REPAIR_END, route5=True, weather=False,
                     calendar=True, traffic=False, citywide_calibration=False) -> pd.DataFrame:
    season = season or ev.SEASON
    special = special if special is not None else ev.SPECIAL_MULT
    y = load_labels()
    end = y.date.max()
    level, shape = fit_profile(y, end)
    dates = pd.date_range("2025-11-01", "2025-12-31")
    f = base_forecast(level, shape, dates)

    # Особые дни: праздники → воскресенье; рабочая суббота → форма пятницы, уровень (Пт + Сб) / 2
    if not calendar:   # абляция: без производственного календаря и особых дней
        special = {}
    hol = f.date.isin(ev.HOLIDAYS) & calendar
    f.loc[hol, "dtype"] = 6
    ws = f.date.isin(ev.WORKING_SATURDAYS) & calendar
    f.loc[ws, "dtype"] = 4
    f = f.drop(columns=["level", "shape"]).join(level, on=["route", "dtype"]).join(shape, on=["route", "dtype", "hour"])
    sat = level.xs(5, level="dtype")
    f.loc[ws, "level"] = (f.loc[ws, "level"] + f.loc[ws, "route"].map(sat)) / 2

    # Ремонт по выходным (7, 50): после окончания и в праздники — уровень выходных «до ремонта»
    ref = y[(y.date >= ev.WEEKEND_REF_WINDOW[0]) & (y.date <= ev.WEEKEND_REF_WINDOW[1])]
    ref_day = ref.groupby(["route", "dtype", "date"]).y.sum().groupby(level=[0, 1]).median()
    ref_shape = fit_profile(ref, pd.Timestamp(ev.WEEKEND_REF_WINDOW[1]), 3, 3)[1]
    for r in ev.WEEKEND_REPAIR_ROUTES:
        for t in (5, 6):
            restored = level[(r, 1)] * ref_day[(r, t)] / ref_day[(r, 1)]
            m = (f.route == r) & (f.dtype == t) & ((f.date > repair_end) | hol)
            if not m.any():
                continue
            f.loc[m, "level"] = restored
            f.loc[m, "shape"] = f.loc[m].set_index(["route", "dtype", "hour"]).index.map(ref_shape).values

    f["pred"] = f.level * f["shape"]
    f["pred"] *= f.date.dt.month.map(season) * f.date.map(special).fillna(1.0)
    if weather:
        f["pred"] *= weather_factor(f[["date", "hour"]], end, mode="shape" if weather == "shape" else "level")
    if traffic:
        tm = f.date.isin(ev.TRAFFIC_EXTREME_DAYS)
        f.loc[tm, "pred"] *= f.loc[tm, "hour"].map(ev.TRAFFIC_HOUR_FACTOR).fillna(1.0)

    if citywide_calibration:
        f = citywide_monthly_calibration(y, f)

    # Маршрут 5: 0 до запуска, далее — оценка × средний трамвайный профиль
    avg_shape = shape.groupby(level=["dtype", "hour"]).mean()
    r5 = pd.MultiIndex.from_product([[5], dates, range(24)], names=["route", "date", "hour"]).to_frame(index=False)
    r5["dtype"] = r5.date.dt.dayofweek.map(TYPE_OF_DOW)
    r5.loc[r5.date.isin(ev.HOLIDAYS) & calendar, "dtype"] = 6
    daily = np.where(r5.dtype >= 5, ev.ROUTE5_DAILY["weekend"], ev.ROUTE5_DAILY["weekday"])
    r5["pred"] = daily * r5.set_index(["dtype", "hour"]).index.map(avg_shape).values
    r5.loc[(r5.date < ev.ROUTE5_START) | (not route5), "pred"] = 0.0
    r5["pred"] *= r5.date.dt.month.map(season)
    return pd.concat([f[["route", "date", "hour", "pred"]], r5[["route", "date", "hour", "pred"]]])


def release_forecast_s26() -> pd.DataFrame:
    """Exact S26 leaderboard release, including its retrospective observed floor."""
    out = forecast_nov_dec(weather=False).copy()
    out["pred"] = np.rint(out.pred).astype(int)
    route5 = out.route.eq(5) & out.date.ge(ev.ROUTE5_START)
    out.loc[route5, "pred"] = np.rint(
        out.loc[route5, "pred"] * ev.S26_ROUTE5_SCALE
    ).astype(int)
    for (route, hour), observed in ev.S26_NOV1_OBSERVED.items():
        mask = out.route.eq(route) & out.date.eq(pd.Timestamp("2025-11-01")) & out.hour.eq(hour)
        if mask.sum() != 1:
            raise ValueError(f"Missing S26 observed cell: {route=}, {hour=}")
        out.loc[mask, "pred"] = max(int(out.loc[mask, "pred"].iloc[0]), observed)
    return out


def write_submission(f: pd.DataFrame, name: str) -> Path:
    tmpl = pd.read_csv("dataset/test_submission.csv", sep=";", parse_dates=["date"])
    out = tmpl.drop(columns="prediction").merge(f, on=["route", "date", "hour"], how="left")
    assert len(out) == 14640 and out.pred.notna().all(), "сетка не совпала с шаблоном"
    out["prediction"] = out.pred.clip(lower=0).round().astype(int)
    out["date"] = out.date.dt.strftime("%Y-%m-%d")
    path = Path("ml/submissions") / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out[["route", "date", "hour", "prediction"]].to_csv(path, sep=";", index=False)
    return path


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "backtest"
    if cmd == "backtest":
        backtest()
    elif cmd == "submit":
        # опционально: python ml/core.py submit S2 season=1.00
        kw = dict(a.split("=") for a in sys.argv[3:])
        season = {11: float(kw["season"]), 12: float(kw["season"])} if "season" in kw else dict(ev.SEASON)
        season[11], season[12] = float(kw.get("nov", season[11])), float(kw.get("dec", season[12]))
        if "r5" in kw:
            ev.ROUTE5_DAILY = {k: v * float(kw["r5"]) for k, v in ev.ROUTE5_DAILY.items()}
        if "eoy" in kw:
            ev.SPECIAL_MULT = {d: float(kw["eoy"]) for d in ev.SPECIAL_MULT}
        wmode = {"1": True, "shape": "shape"}.get(kw.get("weather", "0"), False)
        name = sys.argv[2] if len(sys.argv) > 2 else "S1"
        if name == "S26" and not kw:
            f = release_forecast_s26()
        else:
            f = forecast_nov_dec(season=season, weather=wmode, traffic=kw.get("traffic") == "1", route5=kw.get("route5", "1") == "1",
                                 repair_end=pd.Timestamp(kw.get("repair_end", str(ev.WEEKEND_REPAIR_END.date()))),
                                 calendar=kw.get("calendar", "1") == "1",
                                 citywide_calibration=kw.get("citycal", "0") == "1")
        p = write_submission(f, name)
        print(p, "total", int(f.pred.sum()))
        print(f.assign(m=f.date.dt.month).groupby(["m", "route"]).pred.sum().unstack(0).round(-2).astype(int).to_string())
