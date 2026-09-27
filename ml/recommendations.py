"""Рекомендации по двум рычагам диспетчера (ответ организаторов): число ТС на линии и время стоянки.

Принцип — перераспределение выпуска между маршрутами БЕЗ роста парка и затрат:
в каждый час суммарное число вагонов на линии по сети = текущему (ограничение парка в пик).
  bpt          = прогноз посадок / вагонов на линии (посадок на вагон в час)
  B*(час)      = единый норматив посадок на вагон-час, подобранный так, что Σ_маршруты trams_needed = Σ trams_plan
  trams_needed = clip(ceil(yhat / B*), пол, потолок): пол = 60% текущего выпуска (стандарт интервала),
                 потолок = 150% (реально перебросить вагоны между маршрутами за час)
  dwell_extra_sec — доп. стоянка на средней остановке относительно типичного часа маршрута.
Выпуск (trams_plan) = медиана вагонов на линии в октябре по (маршрут, тип дня, час) из данных валидаций
(для 7/50 в дни после ремонта — из периода до ремонта; для маршрута 5 — 7 вагонов по mos.ru).
Оценка наполняемости peak_load_pct — справочно, параметры-допущения в ml/config/events.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import events as ev  # noqa: E402
from core import TYPE_OF_DOW  # noqa: E402

A = Path("artifacts/forecast")
MIN_SERVICE_SHARE = 0.6   # не сокращаем выпуск часа больше чем на 40% (стандарт интервала движения)
MAX_SERVICE_SHARE = 1.5   # не добавляем маршруту больше +50% за час (реальная переброска вагонов)
ADD_IF_LOAD = 60          # добавляем вагоны, только если оценка загрузки макс. участка ≥ 60% номинала
REMOVE_IF_LOAD = 40       # снимаем, только если ≤ 40% — пустые вагоны не «выравниваются» по сети

f = pd.read_parquet(A / "hourly.parquet")
act = pd.read_parquet(A / "actuals_hourly.parquet")
f["date"], act["date"] = pd.to_datetime(f.date), pd.to_datetime(act.date)
act["dtype"] = act.date.dt.dayofweek.map(TYPE_OF_DOW)

hol = f.date.isin(ev.HOLIDAYS)
f["dtype"] = np.where(hol, 6, np.where(f.date.isin(ev.WORKING_SATURDAYS), 4, f.date.dt.dayofweek.map(TYPE_OF_DOW)))
plan = act[act.date >= "2025-10-01"].groupby(["route", "dtype", "hour"]).trams.median().rename("trams_plan")
ref = act[act.date.between(*ev.WEEKEND_REF_WINDOW)].groupby(["route", "dtype", "hour"]).trams.median().rename("ref")
f = f.join(plan, on=["route", "dtype", "hour"]).join(ref, on=["route", "dtype", "hour"])
restored = f.route.isin(ev.WEEKEND_REPAIR_ROUTES) & (f.dtype >= 5) & ((f.date > ev.RELEASE_REPAIR_END) | hol)
f.loc[restored, "trams_plan"] = f.loc[restored, "ref"]
f.loc[f.route == 5, "trams_plan"] = np.where(f.loc[f.route == 5, "yhat"] > 30, ev.ROUTE5_TRAMS, 0)
f["trams_plan"] = f.trams_plan.fillna(0).round()
f.loc[f.yhat < 30, "trams_plan"] = np.minimum(f.loc[f.yhat < 30, "trams_plan"], 1)


K_LOAD = ev.RIDE_SHARE * ev.PEAK_SEGMENT * ev.DIRECTION_IMBALANCE


def solve_hour(g: pd.DataFrame) -> pd.DataFrame:
    """Норматив B* (бисекция) задаёт направление; переброска — только от недогруженных к перегруженным, парк часа не растёт."""
    active = g.trams_plan > 0
    floor, cap = np.ceil(MIN_SERVICE_SHARE * g.trams_plan), np.maximum(g.trams_plan + 1, np.floor(MAX_SERVICE_SHARE * g.trams_plan))
    budget = g.trams_plan.sum()
    need_at = lambda b: np.where(active, np.clip(np.ceil(g.yhat / b), floor, cap), 0)
    lo, hi = 1.0, 5000.0
    for _ in range(40):
        b = (lo + hi) / 2
        lo, hi = (b, hi) if need_at(b).sum() > budget else (lo, b)
    g = g.copy()
    g["norm_bpt"] = hi
    need = need_at(hi)
    load = g.yhat / g.trams_plan.replace(0, np.nan) * K_LOAD / ev.CAPACITY_NOMINAL * 100
    # Доноры: норматив предлагает снять вагоны и загрузка низкая. Получатели: предлагает добавить и загрузка высокая.
    give = np.where((need < g.trams_plan) & (load <= REMOVE_IF_LOAD), g.trams_plan - need, 0)
    want = np.where((need > g.trams_plan) & (load >= ADD_IF_LOAD), need - g.trams_plan, 0)
    pool, add = give.sum(), np.zeros(len(g))
    for i in np.argsort(-load.fillna(0).values):      # сначала самые загруженные
        take = min(want[i], pool)
        add[i], pool = take, pool - take
    used = give.sum() - pool                             # снимаем ровно столько, сколько отдали
    order = np.argsort(load.fillna(np.inf).values)       # снимаем с самых пустых
    remove, left = np.zeros(len(g)), used
    for i in order:
        r = min(give[i], left)
        remove[i], left = r, left - r
    g["trams_needed"] = g.trams_plan + add - remove
    return g


f = pd.concat([solve_hour(g) for _, g in f.groupby(["date", "hour"])])
f["trams_delta"] = f.trams_needed - f.trams_plan
f["boardings_per_tram"] = f.yhat / f.trams_plan.replace(0, np.nan)
k = ev.RIDE_SHARE * ev.PEAK_SEGMENT * ev.DIRECTION_IMBALANCE
f["peak_load_pct"] = f.boardings_per_tram * k / ev.CAPACITY_NOMINAL * 100
stops = pd.read_parquet(A / "stop_shares.parquet").groupby("route").stop_id.nunique() / 2
f["board_per_stop_per_trip"] = f.boardings_per_tram / f.route.map(stops).fillna(stops.median())
typ = f.groupby("route").board_per_stop_per_trip.transform("median")
f["dwell_extra_sec"] = ((f.board_per_stop_per_trip - typ) * ev.BOARD_SEC_PER_PAX).clip(lower=0).round()
ratio = f.boardings_per_tram / f.norm_bpt
f["status"] = np.select([ratio > 1.3, ratio > 1.1, ratio < 0.6], ["перегруз", "высокая", "недогруз"], "норма")
f.loc[f.trams_plan == 0, "status"] = "нет движения"

# Риск переполнения: P(фактическая загрузка макс. участка > номинала 188) — из эмпирического распределения
# отношения факт / прогноз по часам (скользящий бэктест «неделя вперёд», backtest_hourly × actuals_hourly).
bt = pd.read_parquet(A / "backtest_hourly.parquet").merge(act[["route", "date", "hour", "boardings"]].assign(
    date=lambda d: d.date.dt.date), on=["route", "date", "hour"])
bt = bt[bt.yhat > 20]
ratios = {h: np.sort((g.boardings / g.yhat).values) for h, g in bt.groupby("hour")}
need_ratio = 100 / f.peak_load_pct.replace(0, np.nan)          # во сколько раз факт должен превысить прогноз
f["overload_risk_pct"] = [
    0.0 if not np.isfinite(r) else 100 * (1 - np.searchsorted(ratios.get(h, np.array([1.0])), r) / len(ratios.get(h, [1])))
    for h, r in zip(f.hour, need_ratio)]
f["overload_risk_pct"] = f.overload_risk_pct.round(1)

out = f[["route", "date", "hour", "yhat", "trams_plan", "trams_needed", "trams_delta", "boardings_per_tram",
         "norm_bpt", "peak_load_pct", "dwell_extra_sec", "status", "overload_risk_pct"]].copy()
out["date"] = out.date.dt.date
out.round(2).to_parquet(A / "recommendations.parquet", index=False)

if __name__ == "__main__":
    day = f[f.hour.between(6, 22) & (f.trams_plan > 0)]
    print("Статусы (6–22 ч):", day.status.value_counts(normalize=True).round(3).to_dict())
    print(f"Вагоно-часов: план {int(f.trams_plan.sum())} → рекомендовано {int(f.trams_needed.sum())} (бюджет сохранён)")
    wd = day[day.date == "2025-11-12"]
    t = wd.groupby("route").agg(plan=("trams_plan", "sum"), need=("trams_needed", "sum")).astype(int)
    t["Δ"] = t.need - t.plan
    print("Будний день 12.11, вагоно-часы по маршрутам:\n", t.to_string())
    pk = wd[wd.hour.isin([8, 18])][["route", "hour", "yhat", "trams_plan", "trams_needed", "boardings_per_tram", "dwell_extra_sec", "status"]]
    print(pk.round(0).to_string(index=False))
    print("Часы с риском переполнения > 20%:", int((day.overload_risk_pct > 20).sum()),
          day.sort_values("overload_risk_pct", ascending=False)[["route", "date", "hour", "peak_load_pct", "overload_risk_pct"]].head(5).to_string(index=False))
    over = day[day.status == "перегруз"]
    print("Пассажиров в часах «перегруз»:", int(over.yhat.sum()), f"({over.yhat.sum() / day.yhat.sum():.1%} потока)")
