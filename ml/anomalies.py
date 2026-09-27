"""Реестр аномалий → artifacts/forecast/anomalies.json (панель «Провалы данных и причины» в UI).

Каждое событие найдено в данных (смена уровня по дням недели, ml/research/regimes.py) и сверено с источником.
Эффект считается по данным, а не вписан руками: effect_pct = Σ факт / Σ ожидание − 1, где ожидание — медиана того же
дня недели за 5 недель до события (без праздников; для начала января — 5 недель после). Для событий прогнозного
периода эффект берётся из слоёв модели (artifacts/forecast/components.parquet).

  uv run --python 3.12 --with-requirements ml/requirements.txt python ml/anomalies.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from core import load_labels  # noqa: E402

OUT = Path("artifacts/forecast/anomalies.json")
HOL = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08",
                      "2025-05-01", "2025-05-02", "2025-05-08", "2025-05-09", "2025-06-12", "2025-06-13"])
CAL = "https://isdayoff.ru/api/getdata?year=2025&pre=1"
REPAIR_NEWS = "https://newsvostok.ru/dlya-tramvaev-7-i-50-izmeneniya-po-vyhodnym-budut-dejstvovat-do-kontsa-oseni/"
REPAIR_END = "https://t.me/DtOperativno/23565"
FREE_RIDE = "https://t.me/DtOperativno/24391"
ROUTE5 = "https://www.mos.ru/mayor/themes/13888050/"

# (маршруты | None = вся сеть, начало, конец, тип, только выходные, источник, пояснение)
HISTORY = [
    (None, "2025-01-01", "2025-01-08", "праздник", False, CAL, "новогодние каникулы"),
    (None, "2025-05-01", "2025-05-02", "праздник", False, CAL, "майские праздники"),
    (None, "2025-05-08", "2025-05-09", "праздник", False, CAL, "майские праздники"),
    (None, "2025-06-12", "2025-06-13", "праздник", False, CAL, "День России; в модели праздник ≈ воскресенье"),
    ([7, 26, 25, 11, 12, 17], "2025-03-31", "2025-04-06", "изменение трассы", False, None,
     "перераспределение потока между маршрутами (7, 26, 25 ↑; 11, 12, 17 ↓); найдено по данным"),
    ([17], "2025-04-05", "2025-04-27", "ремонт", True, None,
     "ремонт по выходным: 26–27.04 — 673 и 323 посадки при обычных ~30 тыс.; найдено по данным"),
    ([7], "2025-07-07", "2025-08-10", "ремонт", False, None,
     "ремонт путей: маршрут 7 ↓, пассажиры на объединённом 50+13; найдено по данным"),
    ([28], "2025-09-22", "2025-09-24", "ремонт", False, None,
     "на линии 3–4 вагона вместо 7; причина в открытых источниках не найдена"),
    ([50], "2025-10-01", "2025-10-01", "сбой данных", False, None,
     "вдвое больше бортов (32 против 16) и посадок; разовый выброс, причина не подтверждена"),
    ([50], "2025-09-06", "2025-10-31", "ремонт", True, REPAIR_NEWS,
     "ремонт путей в Протопоповском пер.: движение по выходным почти прекращено; в прогнозе — до 14.11"),
    ([7], "2025-09-06", "2025-10-31", "ремонт", True, REPAIR_NEWS,
     "тот же ремонт: маршрут 7 по выходным изменён; в прогнозе — до 14.11"),
]


def daily() -> pd.DataFrame:
    d = load_labels().groupby(["route", "date"]).y.sum().reset_index()
    return d


def expected(d: pd.DataFrame, route: int, day: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> float:
    g = d[(d.route == route) & (d.date.dt.dayofweek == day.dayofweek) & ~d.date.isin(HOL)]
    before = g[g.date.between(start - pd.Timedelta(days=35), start - pd.Timedelta(days=1))]
    ref = before if len(before) >= 3 else g[g.date.between(end + pd.Timedelta(days=1), end + pd.Timedelta(days=35))]
    return float(ref.y.median()) if len(ref) else float("nan")


def history_items(d: pd.DataFrame) -> list[dict]:
    items = []
    for routes, a, b, kind, weekends, url, note in HISTORY:
        start, end = pd.Timestamp(a), pd.Timestamp(b)
        per_route = routes or [None]
        for r in per_route:
            rs = routes and [r] or sorted(d.route.unique())
            fact = exp = 0.0
            for rr in rs:
                w = d[(d.route == rr) & d.date.between(start, end)]
                if weekends:
                    w = w[w.date.dt.dayofweek >= 5]
                fact += w.y.sum()
                exp += sum(expected(d, rr, day, start, end) for day in w.date)
            items.append({"route": r, "date_from": a, "date_to": b, "kind": kind,
                          "effect_pct": round(100 * (fact / exp - 1), 1) if exp else None,
                          "source_url": url, "note": note + (" (только выходные)" if weekends else "")})
    return items


def free_ride_effect(c: pd.DataFrame) -> float:
    x = c[c.date.eq(pd.Timestamp("2025-12-31"))]
    before = x.final.sum() - x[x.hour >= 20].final.sum() + (x.final - x.calendar)[x.hour >= 20].clip(lower=0).sum()
    return round(float(100 * (x.final.sum() / before - 1)), 1)


def forecast_items() -> list[dict]:
    c = pd.read_parquet("artifacts/forecast/components.parquet")
    c["date"] = pd.to_datetime(c.date)

    def layer(dates, col, routes=None, hours=None):
        x = c[c.date.isin(pd.to_datetime(dates))]
        x = x if routes is None else x[x.route.isin(routes)]
        x = x if hours is None else x[x.hour.isin(hours)]
        return round(float(100 * x[col].sum() / x.profile.sum()), 1)

    items = [
        {"route": None, "date_from": "2025-11-01", "date_to": "2025-11-01", "kind": "праздник", "source_url": CAL,
         "effect_pct": layer(["2025-11-01"], "calendar"),
         "note": "рабочая суббота (перенос выходного на 03.11): форма суток пятницы, уровень (Пт+Сб)/2"},
        {"route": None, "date_from": "2025-11-03", "date_to": "2025-11-04", "kind": "праздник", "source_url": CAL,
         "effect_pct": layer(["2025-11-03", "2025-11-04"], "calendar"),
         "note": "праздничные дни: прогноз по профилю воскресенья"},
        {"route": None, "date_from": "2025-12-29", "date_to": "2025-12-31", "kind": "праздник", "source_url": CAL,
         "effect_pct": -15.0, "note": "предновогодние дни: ×0.85 к уровню (априорная оценка)"},
        {"route": None, "date_from": "2025-12-31", "date_to": "2025-12-31", "kind": "праздник", "source_url": FREE_RIDE,
         "effect_pct": free_ride_effect(c),
         "note": "бесплатный проезд с 20:00: в 20–24 ч валидаций нет (так было и 01.01.2025 00–03 ч); эффект — к суткам"},
        {"route": 5, "date_from": "2025-12-16", "date_to": "2025-12-31", "kind": "изменение трассы",
         "source_url": ROUTE5, "effect_pct": None,
         "note": "запуск маршрута 5: истории нет, уровень — посадок на вагон у маршрута 25 × 7 вагонов"},
    ]
    sat = pd.date_range("2025-11-15", "2025-12-31")
    sat = [x for x in sat if x.dayofweek >= 5]
    for r in (7, 50):
        items.append({"route": r, "date_from": "2025-11-15", "date_to": "2025-12-31", "kind": "ремонт",
                      "source_url": REPAIR_END, "effect_pct": layer(sat, "network", [r]),
                      "note": "окончание ремонта: с 15.11 выходные восстанавливаются (оперативное сообщение Дептранса "
                              "в день события); эффект — к профилю, который помнит выходные ремонта"})
    return items


if __name__ == "__main__":
    items = history_items(daily()) + forecast_items()
    OUT.write_text(json.dumps({"method": "effect_pct = факт / медиана того же дня недели за 5 недель до события − 1; "
                                         "прогнозный период — вклад слоя модели к профилю",
                               "items": items}, ensure_ascii=False, indent=2), encoding="utf-8")
    for i in items:
        print(i["route"], i["date_from"], i["date_to"], i["kind"], i["effect_pct"])
    print(OUT, len(items))
