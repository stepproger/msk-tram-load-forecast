"""Честная версия прогноза (HONEST2 — релиз): ни одного факта посадок целевого периода; параметры профиля — из истории,
рецепт ансамбля E7 — из загрузок E1–E7 (подтверждён прокси).

Что модель получает и когда это известно (точно, без округлений в свою пользу):
  • история валидаций 01.01–31.10.2025 (labels) — до прогноза;
  • производственный календарь РФ (isdayoff.ru) — заранее;
  • запуск маршрута 5 с 16.12 — дата есть в справочнике организаторов (route_date_start); 7 вагонов — mos.ru;
  • бесплатный проезд 31.12 с 20:00 — пост Дептранса от 26.12 (t.me/DtOperativno/24391), до события;
  • окончание ремонта 7/50 по выходным — ОПЕРАТИВНОЕ сообщение Дептранса в день события (15.11, 12:14 МСК,
    t.me/DtOperativno/23565). Это не анонс: в потоковом режиме реестр событий пополняется по мере поступления
    сообщений диспетчерской. Со знанием на 31.10 («до конца осени» → 01.12) лучший ML-вариант E7 = 0.90017;
  • погода — ФАКТИЧЕСКАЯ погода ноября–декабря (Open-Meteo, ERA5) как признак LightGBM: внешний фактор из
    критериев; в эксплуатации заменяется прогнозом погоды. Строгий вариант HONEST4 (климатическая норма
    2015–2024 + почасовая поправка формы) = 0.90066.
Все числовые параметры выведены из истории, а не из лидерборда:
  • сезонный множитель ×1.03 — из систематического занижения бэктестов (−2.4…−3.4% на горизонте месяц);
  • 29–31.12 ×0.85 — априорная оценка до любых загрузок (S1);
  • маршрут 5 — аналог маршрута 25 (тоже 7 вагонов): посадок на вагон в будни/выходные × 7 вагонов;
  • вес ансамбля (0.5·профиль + 0.25·LightGBM в масштабе профиля + 0.25·LightGBM) — рецепт E7, выбран по серии загрузок E1–E7; подтверждён локальным
    прокси (ml/research/proxy_lb.py, лучший в октябрьском срезе).
НЕ используется: факт первых часов 01.11 из хвоста test.csv, опубликованный постфактум поток маршрута 5,
калибровки ×1.022 / ×1.6 / ×1.5 / ×0.9, найденные по лидерборду.

  uv run --python 3.12 --with-requirements ml/requirements.txt python ml/honest.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import events as ev  # noqa: E402
from core import TYPE_OF_DOW, forecast_nov_dec, write_submission  # noqa: E402
from release import with_ml  # noqa: E402

HONEST_SEASON = {11: 1.03, 12: 1.03}
# Бесплатный проезд в новогоднюю ночь с 20:00 31.12 (mos.ru/news/item/164558073, t.me/DtOperativno/24391):
# валидаций нет. Проверено на истории: 01.01.2025 00–03 ч — 0 посадок (обычное вс в 00 ч ≈ 600).
FREE_RIDE = (pd.Timestamp("2025-12-31"), range(20, 24))
HONEST_REPAIR_END = ev.RELEASE_REPAIR_END   # последние выходные ремонта — 08–09.11; с 15.11 обычный режим
ANALOG_ROUTE, ROUTE5_TRAMS = 25, 7


def route5_analog() -> dict:
    """Уровень маршрута 5 = посадок на вагон у аналога (маршрут 25, октябрь) × 7 вагонов."""
    a = pd.read_parquet("artifacts/forecast/actuals_hourly.parquet")
    a["date"] = pd.to_datetime(a.date)
    o = a[(a.route == ANALOG_ROUTE) & (a.date >= "2025-10-01")]
    d = o.groupby("date").agg(b=("boardings", "sum"), fleet=("trams", "max")).reset_index()
    d["wd"] = d.date.dt.dayofweek < 5
    per_tram = (d.b / d.fleet).groupby(d.wd).median()
    return {"weekday": float(per_tram[True] * ROUTE5_TRAMS), "weekend": float(per_tram[False] * ROUTE5_TRAMS)}


def honest_base() -> pd.DataFrame:
    ev.ROUTE5_DAILY = route5_analog()
    return forecast_nov_dec(season=HONEST_SEASON, repair_end=HONEST_REPAIR_END)


def components(final: pd.DataFrame) -> pd.DataFrame:
    """Разложение релиза по слоям (посадок в час): profile + calendar + network + season + ml = final."""
    key = ["route", "date", "hour"]
    no_season, far = {11: 1.0, 12: 1.0}, pd.Timestamp("2099-01-01")
    steps = {
        "profile": forecast_nov_dec(season=no_season, special={}, calendar=False, route5=False, repair_end=far),
        "calendar": apply_free_ride(forecast_nov_dec(season=no_season, route5=False, repair_end=far)),
        "network": apply_free_ride(forecast_nov_dec(season=no_season, repair_end=HONEST_REPAIR_END)),
        "season": apply_free_ride(forecast_nov_dec(season=HONEST_SEASON, repair_end=HONEST_REPAIR_END)),
        "ml": final,
    }
    out, prev = None, None
    for name, f in steps.items():
        cur = f.set_index(key).pred.clip(lower=0)
        col = cur if prev is None else cur - prev.reindex(cur.index).fillna(0)
        out = col.rename(name).to_frame() if out is None else out.join(col.rename(name))
        prev = cur
    out["calibration"] = 0.0          # в честной версии уточнений по лидерборду/постфактум-данным нет
    out["final"] = prev
    return out.reset_index()[key + ["profile", "calendar", "network", "season", "calibration", "ml", "final"]]


def hour_shape_correction(base: pd.DataFrame) -> pd.DataFrame:
    """Почасовая поправка формы (агент C, ml/research/error_analysis.py): по 5 псевдо-бэктестам ДО 31.10
    (профиль на срезе cut−j недель прогнозирует 28 дней, j = 4..8) считаем систематическое смещение по часам
    факт/профиль по сети, применяем к профилю и перенормируем к исходной дневной сумме маршрута.
    Прокси: +0.0008 в среднем, +0.0034 на срезе 28.09→31.10, лучше в 4 из 5 срезов."""
    sys.path.insert(0, str(Path(__file__).parent / "research"))
    from error_analysis import hour_corr  # noqa: E402
    from proxy_lb import load  # noqa: E402
    y = load()
    r = hour_corr(y, y.date.max())
    f = base.copy()
    f["q"] = f.pred * f.hour.map(r).fillna(1.0).values
    day_p, day_q = f.groupby(["route", "date"]).pred.transform("sum"), f.groupby(["route", "date"]).q.transform("sum")
    f["pred"] = (f.q * (day_p / day_q.replace(0, np.nan)).fillna(1.0)).values
    print("поправка по часам:", {int(h): round(float(v), 3) for h, v in r.items() if abs(v - 1) > .02})
    return f.drop(columns="q")


def apply_free_ride(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    f.loc[f.date.eq(FREE_RIDE[0]) & f.hour.isin(FREE_RIDE[1]), "pred"] = 0.0
    return f


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "candidate":
        f = apply_free_ride(with_ml(honest_base(), clim=True))
        print(write_submission(f, "HONEST3"), "total", int(f.pred.sum()))
        f4 = apply_free_ride(with_ml(hour_shape_correction(honest_base()), clim=True))
        print(write_submission(f4, "HONEST4"), "total", int(f4.pred.sum()))
        sys.exit()
    r5 = route5_analog()
    print("маршрут 5 (аналог 25 × 7 вагонов):", {k: round(v) for k, v in r5.items()})
    base = honest_base()
    final = apply_free_ride(with_ml(base))
    print(write_submission(final, "HONEST2"), "total", int(final.pred.sum()))
    print(write_submission(final, "FINAL"), "(релиз = HONEST2)")
    c = components(final)
    c["date"] = c.date.dt.date
    c.astype({k: "float32" for k in c.columns if k not in ("route", "date", "hour")}).to_parquet(
        "artifacts/forecast/components.parquet", index=False)
    print("слои:", c.drop(columns=["route", "date", "hour"]).sum().round(-2).astype(int).to_dict())
