"""Горизонт «год»: дневной прогноз посадок по маршрутам на 01.11.2025–31.10.2026 и его честная проверка.

Схема:  ŷ(маршрут, день) = Город(месяц) × Доля9(месяц) × Разбиение(маршрут, тип дня)  [+ маршрут 5 константой]
  1. Город — месячный поток трамваев Москвы (data.mos.ru, набор 62521). Варианты:
       naive       — тот же месяц год назад (бейзлайн);
       snaive_g    — тот же месяц год назад × рост скользящих 12 мес. (г/г);
       seas_trend  — мультипликативная сезонность (2019, 2022–2025; 2020–21 исключены) × уровень × лог-линейный тренд;
       seas_damped — то же с демпфированным трендом (φ = 0.9 в месяц).
     Выбор — по бэктесту на городских данных (прогноз 2024 и 2025 из октября и декабря предыдущего года).
  2. Доля 9 маршрутов в городе (по labels 2025): last / mean3 / damped-trend — выбор по бэктесту внутри 2025.
  3. Разбиение месяца на дни и маршруты: средние дневные посадки маршрута по типу дня (Пн, Вт–Чт, Пт, Сб,
     Вс/праздник по производственному календарю) за последние 8 недель истории, масштаб — к итогу месяца.
     Выходные маршрутов 7/50 — из окна до ремонта путей (events.WEEKEND_REF_WINDOW), в 2026 ремонт окончен.
  Ноябрь–декабрь 2025 берутся из ml/submissions/FINAL.csv (сумма часов за день); год-модель — 01.01–31.10.2026.
  Out-of-sample: прогноз города на янв–авг 2026 только по данным до 12.2025 против официальных фактов 2026.

Запуск (из корня):
  uv run --python 3.12 --with-requirements ml/requirements.txt python ml/year_horizon.py
Выход: ml/research/year_horizon_report.json, artifacts/forecast/daily_year_candidate.parquet
"""
import glob
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import events as ev  # noqa: E402
from core import ROUTES, TYPE_OF_DOW, load_labels  # noqa: E402

CITY_GLOB = "data/external/mosdata/62521/*.csv"
CAL_FILES = {2025: "data/external/calendar_ru_2025.csv", 2026: "data/external/calendar_ru_2026.csv"}
FINAL_FILE = "ml/submissions/FINAL.csv"
REPORT = Path("ml/research/year_horizon_report.json")
OUT = Path("artifacts/forecast/daily_year_candidate.parquet")
MONTHS_RU = ["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь", "Июль", "Август",
             "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"]
COVID_YEARS = {2020, 2021}
PHI = 0.9                      # демпфирование тренда, в месяц (не подбирается — фиксировано априори)
TREND_MONTHS = 24              # окно оценки тренда (без ковидных месяцев)
PROFILE_WEEKS = 8              # окно дневного профиля маршрутов
Z90 = 1.2816                   # квантиль N(0,1) для q10/q90
FORECAST_START, FORECAST_END = pd.Timestamp("2025-11-01"), pd.Timestamp("2026-10-31")
ROUTE5_START = pd.Timestamp("2025-12-16")


# ---------------------------------------------------------------- данные
def load_city() -> pd.Series:
    """Месячный поток трамваев, индекс — Period[M]."""
    df = pd.concat(pd.read_csv(f, sep=";", skiprows=[1]) for f in sorted(glob.glob(CITY_GLOB)))
    df = df[df["Type of transport"] == "Трамвай"]
    month = df.Month.map({m: i + 1 for i, m in enumerate(MONTHS_RU)})
    idx = pd.PeriodIndex([pd.Period(year=y, month=m, freq="M") for y, m in zip(df.Year, month)])
    s = pd.Series(df["Passenger traffic"].astype(float).values, index=idx).groupby(level=0).last().sort_index()
    return s


def load_calendar() -> pd.DataFrame:
    """Производственный календарь 2025–2026; 2026 скачивается с isdayoff.ru при отсутствии."""
    p26 = Path(CAL_FILES[2026])
    if not p26.exists():
        codes = urllib.request.urlopen("https://isdayoff.ru/api/getdata?year=2026&pre=1", timeout=60).read().decode()
        assert len(codes) == 365 and set(codes) <= set("012"), "неожиданный ответ isdayoff.ru"
        pd.DataFrame({"date": pd.date_range("2026-01-01", "2026-12-31").strftime("%Y-%m-%d"),
                      "code": list(codes)}).to_csv(p26, index=False)
    cal = pd.concat(pd.read_csv(f, dtype={"code": int}) for f in CAL_FILES.values())
    cal["date"] = pd.to_datetime(cal.date)
    return cal.set_index("date").code


def day_type(dates: pd.DatetimeIndex, cal: pd.Series) -> np.ndarray:
    """Тип дня: 0 Пн, 1 Вт–Чт, 4 Пт, 5 Сб, 6 Вс/праздник. Выходной по календарю в будни → 6, рабочие выходные → 1."""
    t = pd.Series(dates.dayofweek, index=dates).map(TYPE_OF_DOW)
    code = cal.reindex(dates).values
    t[(code == 1) & (dates.dayofweek < 5)] = 6
    t[(code != 1) & (dates.dayofweek >= 5) & ~np.isnan(code.astype(float))] = 1
    return t.values


# ---------------------------------------------------------------- модель города
def seasonal_index(hist: pd.Series) -> pd.Series:
    """Средний индекс месяца (месяц / среднее года) по полным не-ковидным годам."""
    df = hist.to_frame("y").assign(year=hist.index.year, m=hist.index.month)
    full = df.groupby("year").y.count()
    df = df[df.year.isin(full[full == 12].index) & ~df.year.isin(COVID_YEARS)]
    idx = (df.y / df.groupby("year").y.transform("mean")).groupby(df.m).mean()
    return idx / idx.mean()


def last_same_month(hist: pd.Series, target: pd.Period) -> tuple[float, int]:
    """Последнее наблюдение того же календарного месяца (не ковид) и сколько лет назад."""
    for k in range(1, 8):
        p = target - 12 * k
        if p in hist.index and p.year not in COVID_YEARS:
            return hist[p], k
    raise ValueError(target)


def forecast_city(hist: pd.Series, targets: list, variant: str) -> pd.Series:
    """Прогноз месячного потока на targets по истории hist (только прошлое)."""
    origin = hist.index.max()
    out = {}
    if variant in ("naive", "snaive_g"):
        g = hist.iloc[-12:].sum() / hist.iloc[-24:-12].sum() if variant == "snaive_g" else 1.0
        for t in targets:
            v, k = last_same_month(hist, t)
            out[t] = v * g ** k
        return pd.Series(out)
    S = seasonal_index(hist)
    d = hist / S.reindex(hist.index.month).values                       # десезонированный ряд
    d = d[~d.index.year.isin(COVID_YEARS)].iloc[-TREND_MONTHS:]
    x = np.array([(p - origin).n for p in d.index], dtype=float)       # 0 = origin, в месяцах
    b, a = np.polyfit(x, np.log(d.values), 1)
    level = np.exp(a)                                                   # сглаженный уровень в точке origin
    for t in targets:
        h = (t - origin).n
        steps = h if variant == "seas_trend" else sum(PHI ** j for j in range(1, h + 1))
        out[t] = level * np.exp(b * steps) * S[t.month]
    return pd.Series(out)


CITY_VARIANTS = ["naive", "snaive_g", "seas_trend", "seas_damped"]


def mape(fc: pd.Series, fact: pd.Series) -> float:
    fact = fact.reindex(fc.index)
    return float((np.abs(fc - fact) / fact).mean() * 100)


def city_backtest(city: pd.Series) -> tuple[dict, pd.DataFrame]:
    """Бэктесты: origin 10.2023→2024, 10.2024→2025 (как в задаче) и 12.2023→янв–окт 2024, 12.2024→янв–окт 2025."""
    setups = [("2023-10", "2024-01", "2024-12"), ("2024-10", "2025-01", "2025-12"),
              ("2023-12", "2024-01", "2024-10"), ("2024-12", "2025-01", "2025-10")]
    rows = []
    res = {}
    for o, t0, t1 in setups:
        o = pd.Period(o, "M")
        hist = city[city.index <= o]
        targets = list(pd.period_range(t0, t1, freq="M"))
        key = f"origin {o} → {t0}..{t1}"
        res[key] = {}
        for v in CITY_VARIANTS:
            fc = forecast_city(hist, targets, v)
            res[key][v] = round(mape(fc, city), 2)
            for t in targets:
                rows.append(dict(setup=key, variant=v, month=str(t), h=(t - o).n, fc=fc[t], fact=city[t]))
    err = pd.DataFrame(rows)
    err["log_err"] = np.log(err.fact / err.fc)
    res["mean_all_setups"] = {v: round(float(np.mean([res[k][v] for k in res if k.startswith("origin")])), 2)
                              for v in CITY_VARIANTS}
    return res, err


# ---------------------------------------------------------------- доля 9 маршрутов
def forecast_share(sh: pd.Series, targets: list, variant: str) -> pd.Series:
    origin = sh.index.max()
    if variant == "last":
        return pd.Series(sh.iloc[-1], index=targets)
    if variant == "mean3":
        return pd.Series(sh.iloc[-3:].mean(), index=targets)
    # damped: наклон по последним 6 мес., демпфирование φ
    x = np.array([(p - origin).n for p in sh.index[-6:]], dtype=float)
    b, a = np.polyfit(x, sh.values[-6:], 1)
    return pd.Series({t: a + b * sum(PHI ** j for j in range(1, (t - origin).n + 1)) for t in targets})


SHARE_VARIANTS = ["last", "mean3", "damped"]


def share_backtest(share: pd.Series) -> tuple[dict, pd.DataFrame]:
    rows = []
    for o in ["2025-05", "2025-06", "2025-07", "2025-08"]:
        o = pd.Period(o, "M")
        hist = share[share.index <= o]
        targets = [p for p in share.index if p > o]
        for v in SHARE_VARIANTS:
            fc = forecast_share(hist, targets, v)
            for t in targets:
                rows.append(dict(origin=str(o), variant=v, month=str(t), h=(t - o).n, fc=fc[t], fact=share[t]))
    err = pd.DataFrame(rows)
    err["ape"] = np.abs(err.fc / err.fact - 1) * 100
    err["log_err"] = np.log(err.fact / err.fc)
    res = {v: round(float(err[err.variant == v].ape.mean()), 2) for v in SHARE_VARIANTS}
    return res, err


# ---------------------------------------------------------------- разбиение на дни и маршруты
def route_profile(daily: pd.DataFrame, end: pd.Timestamp, cal: pd.Series) -> pd.Series:
    """Средние дневные посадки маршрута по типу дня за PROFILE_WEEKS недель до end (включительно)."""
    w = daily[(daily.date > end - pd.Timedelta(weeks=PROFILE_WEEKS)) & (daily.date <= end)].copy()
    w["dt"] = day_type(pd.DatetimeIndex(w.date), cal)
    return w.groupby(["route", "dt"]).y.mean()


def undo_weekend_repair(prof: pd.Series, daily: pd.DataFrame, cal: pd.Series) -> pd.Series:
    """Маршруты 7/50: осенью 2025 по выходным ремонт (ml/config/events.py), в 2026 он окончен.
    Выходные уровни = (выходные / Вт–Чт в окне до ремонта) × текущий уровень Вт–Чт."""
    prof = prof.copy()
    a, b = map(pd.Timestamp, ev.WEEKEND_REF_WINDOW)
    ref = daily[(daily.date >= a) & (daily.date <= b)].copy()
    ref["dt"] = day_type(pd.DatetimeIndex(ref.date), cal)
    ref = ref.groupby(["route", "dt"]).y.mean()
    for r in ev.WEEKEND_REPAIR_ROUTES:
        for dt in (5, 6):
            prof[(r, dt)] = ref[(r, dt)] / ref[(r, 1)] * prof[(r, 1)]
    return prof


def split_month(total: float, dates: pd.DatetimeIndex, prof: pd.Series, cal: pd.Series) -> pd.DataFrame:
    g = pd.MultiIndex.from_product([prof.index.get_level_values(0).unique(), dates], names=["route", "date"])
    g = g.to_frame(index=False)
    g["dt"] = np.tile(day_type(dates, cal), len(g) // len(dates))
    g["raw"] = prof.reindex(pd.MultiIndex.from_frame(g[["route", "dt"]])).fillna(0).values
    g["yhat"] = g.raw * total / g.raw.sum()
    return g


def split_backtest(daily: pd.DataFrame, cal: pd.Series) -> tuple[dict, float]:
    """Ошибка одного лишь разбиения: итог месяца 9 маршрутов известен, профиль — из 8 недель до месяца."""
    res, logs = {}, []
    for m in pd.period_range("2025-03", "2025-10", freq="M"):
        dates = pd.date_range(m.start_time, m.end_time.normalize())
        fact = daily[daily.date.isin(dates)]
        prof = route_profile(daily, m.start_time - pd.Timedelta(days=1), cal)
        g = split_month(fact.y.sum(), dates, prof, cal).merge(fact, on=["route", "date"])
        res[str(m)] = round(float(np.abs(g.y - g.yhat).sum() / g.y.sum() * 100), 2)
        ok = (g.y > 0) & (g.yhat > 0)
        logs.append(np.log(g.y[ok] / g.yhat[ok]))
    logs = pd.concat(logs)
    sigma = float(1.4826 * np.median(np.abs(logs)))   # робастно: хвост из дней-сбоев (≈0 посадок) не раздувает σ
    return res, sigma


# ---------------------------------------------------------------- main
def main():
    city = load_city()
    cal = load_calendar()
    y = load_labels()
    daily = y.groupby(["route", "date"]).y.sum().reset_index()
    monthly9 = daily.groupby(daily.date.dt.to_period("M")).y.sum()
    share = (monthly9 / city.reindex(monthly9.index)).rename("share")

    # 1. город: бэктест и выбор
    bt, city_err = city_backtest(city)
    best_city = min(CITY_VARIANTS, key=lambda v: bt["mean_all_setups"][v])

    # 1б. out-of-sample 2026: только данные до 12.2025 (и строгий вариант — до 10.2025, конец labels)
    oos = {}
    oos_table = []
    for o in ["2025-12", "2025-10"]:
        o = pd.Period(o, "M")
        hist = city[city.index <= o]
        targets = [p for p in city.index if p.year == 2026]
        oos[f"origin {o}"] = {}
        for v in CITY_VARIANTS:
            fc = forecast_city(hist, targets, v)
            oos[f"origin {o}"][v] = round(mape(fc, city), 2)
            if str(o) == "2025-12":
                for t in targets:
                    oos_table.append(dict(month=str(t), variant=v, forecast=round(fc[t]), fact=round(city[t]),
                                          err_pct=round((fc[t] / city[t] - 1) * 100, 2)))

    # 2. доля
    share_bt, share_err = share_backtest(share)
    best_share = min(SHARE_VARIANTS, key=lambda v: share_bt[v])

    # 3. разбиение
    split_bt, sigma_day = split_backtest(daily, cal)

    # ---- ноябрь–декабрь 2025 — из FINAL.csv (включая маршрут 5); доля 9 маршрутов, неявная в FINAL
    fin = pd.read_csv(FINAL_FILE, sep=";", parse_dates=["date"])
    fin = fin.groupby(["route", "date"]).prediction.sum().rename("yhat").reset_index()
    fin["sigma"] = sigma_day
    fin9 = fin[fin.route.isin(ROUTES)]
    nd = pd.period_range("2025-11", "2025-12", freq="M")
    implied = fin9.groupby(fin9.date.dt.to_period("M")).yhat.sum() / city.reindex(nd)

    # ---- прогноз на 2026-01..2026-10: город от origin 12.2025; доля — выбранное правило по ряду
    # «факт янв–окт 2025 + неявная доля FINAL ноя–дек» (FINAL — лучший прогноз, подтверждён LB); профиль — 8 недель labels
    targets = list(pd.period_range("2026-01", "2026-10", freq="M"))
    city_fc = forecast_city(city[city.index <= pd.Period("2025-12", "M")], targets, best_city)
    share_ext = pd.concat([share.dropna(), implied])
    share_fc = forecast_share(share_ext, targets, best_share)
    share_fc_labels_only = forecast_share(share.dropna(), targets, best_share)
    prof = undo_weekend_repair(route_profile(daily, daily.date.max(), cal), daily, cal)
    parts = []
    # разброс ошибок: город по горизонту (бэктест выбранного варианта), доля, разбиение — независимы в логе
    ce = city_err[city_err.variant == best_city]
    sig_city = {h: float(np.sqrt(np.mean(ce[(ce.h >= h - 1) & (ce.h <= h + 1)].log_err ** 2))) for h in range(1, 11)}
    se = share_err[share_err.variant == best_share]
    sig_share = float(np.sqrt(np.mean(se.log_err ** 2)))
    for t in targets:
        dates = pd.date_range(t.start_time, t.end_time.normalize())
        g = split_month(city_fc[t] * share_fc[t], dates, prof, cal)
        g["sigma"] = np.sqrt(sig_city[(t - pd.Period("2025-12", "M")).n] ** 2 + sig_share ** 2 + sigma_day ** 2)
        parts.append(g[["route", "date", "yhat", "sigma"]])
    year = pd.concat(parts)

    # покрытие 80%-интервала города на фактах 2026 (выбранный вариант)
    obs26 = [t for t in targets if t in city.index]
    cover = float(np.mean([abs(np.log(city[t] / city_fc[t])) <= Z90 * sig_city[(t - pd.Period("2025-12", "M")).n]
                           for t in obs26]))

    # маршрут 5 в 2026 — константный уровень по типу дня из FINAL за 16–31.12.2025
    r5 = fin[(fin.route == 5) & (fin.date >= ROUTE5_START)].copy()
    r5["dt"] = day_type(pd.DatetimeIndex(r5.date), cal)
    r5_level = r5.groupby("dt").yhat.mean()
    d26 = pd.date_range("2026-01-01", FORECAST_END)
    r5_year = pd.DataFrame({"route": 5, "date": d26,
                            "yhat": r5_level.reindex(day_type(d26, cal)).values, "sigma": sigma_day})

    out = pd.concat([fin, year, r5_year]).sort_values(["route", "date"]).reset_index(drop=True)
    out = out[(out.date >= FORECAST_START) & (out.date <= FORECAST_END)]
    out = pd.DataFrame({
        "route": out.route.astype("int32"),
        "date": out.date.dt.date,
        "yhat": out.yhat.astype("float32"),
        "q10": (out.yhat * np.exp(-Z90 * out.sigma)).astype("float32"),
        "q90": (out.yhat * np.exp(Z90 * out.sigma)).astype("float32"),
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    report = {
        "description": "Горизонт «год»: город (62521) × доля 9 маршрутов × разбиение по типам дня; ноя–дек 2025 из FINAL.csv",
        "city_variants": {
            "naive": "тот же месяц год назад",
            "snaive_g": "тот же месяц год назад × рост скользящих 12 мес. г/г",
            "seas_trend": "сезонность (2019, 2022–2025, полные годы) × уровень × лог-линейный тренд (24 мес.)",
            "seas_damped": f"то же, тренд демпфирован φ={PHI}/мес.",
        },
        "city_backtest_mape_pct": bt,
        "city_selected": best_city,
        "oos_2026_mape_pct_jan_aug": oos,
        "oos_2026_table_origin_2025_12": oos_table,
        "share_history_2025": {str(k): round(float(v), 4) for k, v in share.dropna().items()},
        "share_backtest_mape_pct": share_bt,
        "share_selected": best_share,
        "share_forecast_2026": round(float(share_fc.iloc[0]), 4),
        "share_forecast_2026_labels_only": round(float(share_fc_labels_only.iloc[0]), 4),
        "share_implied_by_FINAL": {str(k): round(float(v), 4) for k, v in implied.items()},
        "split_backtest_wape_pct_route_day": split_bt,
        "sigma_log": {"city_by_h": {h: round(s, 4) for h, s in sig_city.items()},
                      "share": round(sig_share, 4), "day_split": round(sigma_day, 4)},
        "city_80pct_interval_coverage_2026_jan_aug": round(cover, 3),
        "route5_2026_level_by_daytype": {int(k): round(float(v)) for k, v in r5_level.items()},
        "city_forecast_2026": {str(k): round(float(v)) for k, v in city_fc.items()},
        "routes9_monthly_forecast_2026": {str(t): round(float(city_fc[t] * share_fc[t])) for t in targets},
        "output": str(OUT),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print("Бэктест города, MAPE %:"); print(pd.DataFrame({k: v for k, v in bt.items()}).T.to_string())
    print("Out-of-sample 2026 (янв–авг), MAPE %:"); print(pd.DataFrame(oos).T.to_string())
    print(f"Выбран город: {best_city}; доля: {best_share} {share_bt}; разбиение WAPE: {split_bt}")
    print(pd.DataFrame(oos_table).pivot(index="month", columns="variant", values="err_pct").to_string())
    print(f"Доля неявная в FINAL: {implied.round(4).to_dict()}, прогноз доли 2026: {share_fc.iloc[0]:.4f} "
          f"(только labels: {share_fc_labels_only.iloc[0]:.4f}); покрытие 80% города 2026: {cover:.2f}")
    print(f"→ {OUT}: {len(out)} строк, {out.date.min()}..{out.date.max()}; → {REPORT}")


if __name__ == "__main__":
    main()
