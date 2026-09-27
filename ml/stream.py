"""Потоковый режим модели HONEST: данные поступают → модель переобучается → прогноз вперёд.

Рецепт (как в ml/honest.py, без ручных калибровок): прогноз = 0.5·Профиль + 0.25·LightGBM_в_масштабе_профиля + 0.25·LightGBM.
  • Профиль — ml/core.py fit_profile: уровень = медиана дневных сумм за 2 недели, форма суток = медиана часовых
    долей за 4 недели, по 5 типам дня (Пн / Вт–Чт / Пт / Сб / Вс); праздник → тип «Вс».
  • LightGBM — L1, 600 деревьев, признаки route, hour, dow, dt5, hol, t, prcp, snow (как ml/ensemble.py);
    перемасштабирование к профилю по (маршрут, месяц, выходной/будний) внутри горизонта прогноза.
  • Обучение строго на данных ≤ asof. Календарь: праздники 2025 (история) + ml/config/events.py HOLIDAYS.
  • Погода: Open-Meteo ERA5 (реанализ) — в реальной эксплуатации на её место встаёт прогноз погоды на сутки/неделю.
  • Маршрут 5 в поток не входит: у него нет истории до 16.12 (в релизе — аналог маршрута 25).

Команды (из корня репозитория):
  uv run --python 3.12 --with-requirements ml/requirements.txt --with matplotlib python ml/stream.py \
      forecast --asof 2025-10-15 --horizon 7 --out /tmp/forecast.csv
  uv run --python 3.12 --with-requirements ml/requirements.txt --with matplotlib python ml/stream.py \
      replay --start 2025-09-01 --end 2025-10-31 [--gbm-every 7]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import events as ev  # noqa: E402
from core import TYPE_OF_DOW, fit_profile, load_labels, load_weather, wape_score  # noqa: E402

FEATS = ["route", "hour", "dow", "dt5", "hol", "t", "prcp", "snow"]
GBM_PARAMS = dict(objective="l1", n_estimators=600, learning_rate=0.05, num_leaves=63, verbose=-1)
HIST_HOL = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08",
                           "2025-05-01", "2025-05-02", "2025-05-08", "2025-05-09", "2025-06-12", "2025-06-13"])
HOLIDAYS = set(HIST_HOL) | set(ev.HOLIDAYS)
KEY = ["route", "date", "hour"]

_WEATHER = None


def weather() -> pd.DataFrame:
    global _WEATHER
    if _WEATHER is None:
        _WEATHER = load_weather()[["date", "hour", "t", "prcp", "snow"]]
    return _WEATHER


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Календарь + погода для строк route×date×hour (одинаково для истории и прогноза)."""
    g = df.merge(weather(), on=["date", "hour"], how="left")
    g["hol"] = g.date.isin(HOLIDAYS).astype(int)
    g["dow"] = np.where(g.date.isin(ev.WORKING_SATURDAYS), 4, g.date.dt.dayofweek)
    g["dt5"] = np.where(g.hol == 1, 6, g.dow.map(TYPE_OF_DOW))
    return g


def normalize_hist(y_hist: pd.DataFrame) -> pd.DataFrame:
    y = y_hist.rename(columns={"boardings": "y"})[KEY + ["y"]].copy()
    y["date"] = pd.to_datetime(y.date)
    return features(y)


def train_profile(hist: pd.DataFrame, asof: pd.Timestamp):
    tr = hist[hist.date <= asof]
    return fit_profile(tr.assign(dtype=tr.dt5), asof)


def train_gbm(hist: pd.DataFrame, asof: pd.Timestamp) -> lgb.LGBMRegressor:
    tr = hist[hist.date <= asof]
    return lgb.LGBMRegressor(**GBM_PARAMS).fit(tr[FEATS], tr.y, categorical_feature=["route"])


def predict(profile, model, routes, dates) -> pd.DataFrame:
    """Смешивание профиля и бустинга на сетке routes × dates × 24 ч."""
    level, shape = profile
    g = features(pd.MultiIndex.from_product([routes, dates, range(24)], names=KEY).to_frame(index=False))
    g["p"] = (g.join(level, on=["route", "dt5"]).level.fillna(0)
              * g.join(shape, on=["route", "dt5", "hour"])["shape"].fillna(0)).values
    g["g"] = np.clip(model.predict(g[FEATS]), 0, None)
    key = [g.route, g.date.dt.month, g.dt5 >= 5]
    scale = g.groupby(key).p.transform("sum") / g.groupby(key).g.transform("sum").replace(0, np.nan)
    g["prediction"] = 0.5 * g.p + 0.25 * g.g * scale.fillna(0) + 0.25 * g.g
    return g[KEY + ["prediction"]]


def forecast_asof(y_hist: pd.DataFrame, asof, horizon_days: int) -> pd.DataFrame:
    """Почасовой прогноз маршрут×дата×час на horizon_days суток после asof; обучение только на данных ≤ asof."""
    asof = pd.Timestamp(asof).normalize()
    hist = normalize_hist(y_hist)
    hist = hist[hist.date <= asof]
    dates = pd.date_range(asof + pd.Timedelta(days=1), periods=horizon_days)
    return predict(train_profile(hist, asof), train_gbm(hist, asof), sorted(hist.route.unique()), dates)


def cmd_forecast(a):
    y = load_labels()
    asof = pd.Timestamp(a.asof)
    y = y[y.date <= asof]                      # имитация: поступили только данные до конца дня asof
    t0 = time.perf_counter()
    f = forecast_asof(y, asof, a.horizon)
    dt = time.perf_counter() - t0
    f["prediction"] = f.prediction.clip(lower=0).round().astype(int)
    f["date"] = f.date.dt.strftime("%Y-%m-%d")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(a.out, sep=";", index=False)
    print(f"asof {asof.date()} (история {y.date.min().date()}…{y.date.max().date()}), горизонт {a.horizon} сут → "
          f"{a.out}: {len(f)} строк, сумма {f.prediction.sum():,}; обучение+инференс {dt:.1f} с")


def cmd_replay(a):
    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    y_all = features(load_labels()[KEY + ["y"]])
    routes = sorted(y_all.route.unique())
    truth = y_all.set_index(KEY).y
    t_prof, t_gbm, t_inf, day_rows, week_rows = [], [], [], [], []
    model = None
    for i, d in enumerate(pd.date_range(start, end)):
        asof = d - pd.Timedelta(days=1)          # к вечеру D−1 поступили все валидации по D−1 включительно
        hist = y_all[y_all.date <= asof]          # модель физически не видит данных после asof
        t0 = time.perf_counter()
        prof = train_profile(hist, asof)
        t_prof.append(time.perf_counter() - t0)
        if model is None or i % a.gbm_every == 0:
            t0 = time.perf_counter()
            model = train_gbm(hist, asof)
            t_gbm.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        da = predict(prof, model, routes, pd.DatetimeIndex([d]))
        t_inf.append(time.perf_counter() - t0)
        wk = predict(prof, model, routes, pd.date_range(d, periods=7))
        wk = wk[wk.date <= end]
        for rows, f in ((day_rows, da), (week_rows, wk)):
            f = f.assign(origin=d)
            rows.append(f)
        print(f"{d.date()}: профиль {t_prof[-1]*1e3:.0f} мс" + (f", LightGBM {t_gbm[-1]:.1f} с" if i % a.gbm_every == 0 else "")
              + f", инференс {t_inf[-1]*1e3:.0f} мс", flush=True)

    def attach(rows):
        f = pd.concat(rows, ignore_index=True)
        idx = pd.MultiIndex.from_frame(f[KEY])
        f["y"] = truth.reindex(idx).values
        f["base"] = truth.reindex(pd.MultiIndex.from_arrays(
            [f.route, f.date - pd.Timedelta(days=7), f.hour])).values   # тот же день недели неделю назад (всегда ≤ asof)
        f["week"] = f.date.dt.to_period("W-SUN").dt.start_time
        return f

    day, week = attach(day_rows), attach(week_rows)
    weeks = []
    for w, g in day.groupby("week"):
        gw = week[week.week == w]
        weeks.append({"week_start": str(w.date()), "days": int(g.date.nunique()),
                      "day_ahead": wape_score(g.y, g.prediction), "baseline_day_ahead": wape_score(g.y, g.base),
                      "week_ahead": wape_score(gw.y, gw.prediction), "baseline_week_ahead": wape_score(gw.y, gw.base)})
    by_h = week.assign(h=(week.date - week.origin).dt.days + 1).groupby("h").apply(
        lambda g: wape_score(g.y, g.prediction), include_groups=False)
    res = {
        "period": [str(start.date()), str(end.date())],
        "recipe": "0.5·Профиль + 0.25·LightGBM×масштаб профиля (маршрут, месяц, выходной) + 0.25·LightGBM",
        "protocol": ("день за днём: к вечеру D−1 поступили данные по D−1; профиль переобучается ежедневно, "
                     f"LightGBM — полностью раз в {a.gbm_every} сут (экономия времени; между переобучениями "
                     "обученная модель получает новые признаки календаря/погоды); прогноз на D (day-ahead) и D..D+6 "
                     "(week-ahead, цели обрезаны концом периода). Погода — реанализ ERA5 вместо прогноза погоды."),
        "routes": [int(r) for r in routes],
        "metric": "WAPE-score = 1 − Σ|y−ŷ|/Σy по всем ячейкам маршрут×дата×час",
        "overall": {
            "day_ahead": wape_score(day.y, day.prediction),
            "week_ahead": wape_score(week.y, week.prediction),
            "baseline_day_ahead": wape_score(day.y, day.base),
            "baseline_week_ahead": wape_score(week.y, week.base),
            "bias_day_ahead": float(day.prediction.sum() / day.y.sum() - 1),
            "n_forecasts": len(day_rows),
            "n_cells_day_ahead": len(day), "n_cells_week_ahead": len(week),
        },
        "week_ahead_by_horizon_day": {int(k): float(v) for k, v in by_h.items()},
        "weekly": weeks,
        "timing": {
            "profile_fit_ms_median": float(np.median(t_prof) * 1e3),
            "gbm_fit_s_median": float(np.median(t_gbm)), "gbm_fits": len(t_gbm),
            "inference_one_day_ms_median": float(np.median(t_inf) * 1e3),
            "train_rows_last_gbm": int((y_all.date < end).sum()),
            "hardware": "Apple M4 Pro, LightGBM n_jobs по умолчанию (все ядра)",
        },
    }
    Path("ml/research/stream_replay.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    plot(weeks, res, "docs/img/stream_replay.png")
    o, t = res["overall"], res["timing"]
    print(f"\nWAPE-score {start.date()}…{end.date()}: day-ahead {o['day_ahead']:.4f}, week-ahead {o['week_ahead']:.4f}, "
          f"бейзлайн (−7 сут) {o['baseline_day_ahead']:.4f} / {o['baseline_week_ahead']:.4f}")
    print(f"Время: профиль {t['profile_fit_ms_median']:.0f} мс (ежедневно), LightGBM {t['gbm_fit_s_median']:.1f} с "
          f"(раз в {a.gbm_every} сут, {t['gbm_fits']} раз), инференс на сутки {t['inference_one_day_ms_median']:.0f} мс")
    print(pd.DataFrame(weeks).round(4).to_string(index=False))


def plot(weeks, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    ours, base = "#2a78d6", "#eb6834"
    w = pd.DataFrame(weeks)
    x = np.arange(len(w))
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    ax.plot(x, w.day_ahead, color=ours, lw=2, marker="o", ms=6, label="Наша модель, прогноз на сутки вперёд")
    ax.plot(x, w.baseline_day_ahead, color=base, lw=2, marker="s", ms=6,
            label="Бейзлайн: тот же день недели неделю назад")
    for xi, v, b in zip(x, w.day_ahead, w.baseline_day_ahead):
        dy = 8 if v >= b else -14                  # подпись с той стороны, где нет линии бейзлайна
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=8, color=ink)
    labels = [f"{pd.Timestamp(s):%d.%m}" + (f"\n({n} дн.)" if n < 7 else "") for s, n in zip(w.week_start, w.days)]
    ax.set_xticks(x, labels, color=ink2, fontsize=9)
    ax.set_xlabel("Неделя (с понедельника)", color=ink2)
    ax.set_ylabel("WAPE-score (1 − WAPE), выше — лучше", color=ink2)
    lo = min(w.day_ahead.min(), w.baseline_day_ahead.min())
    ax.set_ylim(max(0, lo - 0.05), min(1, max(w.day_ahead.max(), w.baseline_day_ahead.max()) + 0.04))
    ax.grid(axis="y", color=grid, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=ink2)
    o = res["overall"]
    ax.set_title(f"Потоковый реплей {res['period'][0]} … {res['period'][1]}: профиль — ежедневно, LightGBM — раз в неделю\n"
                 f"итог: модель {o['day_ahead']:.3f}, бейзлайн {o['baseline_day_ahead']:.3f}; "
                 f"на неделю вперёд модель {o['week_ahead']:.3f}", color=ink, fontsize=11, loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=ink)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=surface)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("forecast", help="инференс по потоку: обучение на данных ≤ asof, прогноз вперёд")
    f.add_argument("--asof", required=True)
    f.add_argument("--horizon", type=int, default=7)
    f.add_argument("--out", required=True)
    r = sub.add_parser("replay", help="реплей потока день за днём с метриками")
    r.add_argument("--start", default="2025-09-01")
    r.add_argument("--end", default="2025-10-31")
    r.add_argument("--gbm-every", type=int, default=7, help="полное переобучение LightGBM раз в N суток")
    a = ap.parse_args()
    cmd_forecast(a) if a.cmd == "forecast" else cmd_replay(a)
