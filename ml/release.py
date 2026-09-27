"""Эксперимент E7 (LB 0.90017, НЕ релиз; релиз — ml/honest.py → FINAL.csv): E7.csv + разложение прогноза по шагам для UI «почему такой прогноз».

Цепочка (каждый шаг — отдельный слой модели, вклад = разница соседних шагов, посадок в час):
  profile     — уровень (2 нед.) × форма суток (4 нед.) по 5 типам дня, без календаря и событий
  calendar    — производственный календарь: праздники → «Вс», рабочая суббота, 29–31.12
  network     — события сети: окончание ремонта 7/50 по выходным с 01.12, запуск маршрута 5 с 16.12
  season      — сезонный множитель ×1.022 (осень → зима)
  calibration — уточнения по опубликованным данным: поток маршрута 5, факт 01.11 00–02 ч, конец декабря №5 ×0.9
  ml          — LightGBM на всей истории (календарь, погода, час, маршрут), вес 0.5: половина — структура при уровне профиля,
                половина — собственный уровень бустинга
  final       — итог = сумма всех слоёв

  uv run --python 3.12 --with-requirements ml/requirements.txt python ml/release.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import events as ev  # noqa: E402
from core import forecast_nov_dec, release_forecast_s26, write_submission  # noqa: E402
from ensemble import gbm_forecast  # noqa: E402

ML_WEIGHT = 0.5
KEY = ["route", "date", "hour"]


def s31() -> pd.DataFrame:
    f = release_forecast_s26()
    late = f.route.eq(5) & f.date.between(pd.Timestamp("2025-12-23"), pd.Timestamp("2025-12-31"))
    f.loc[late, "pred"] = np.rint(f.loc[late, "pred"] * 0.90)
    return f


def with_ml(base: pd.DataFrame, extra: bool = False, clim: bool = False) -> pd.DataFrame:
    """Ансамбль E7: 0.5·база + 0.25·LightGBM (перемасштабирован к базе по маршрут × месяц × выходной) + 0.25·LightGBM как есть."""
    b = base.rename(columns={"pred": "prediction"}).copy()
    b["prediction"] = b.prediction.clip(lower=0).round()
    b["gbm"] = gbm_forecast(b[KEY], extra=extra, clim=clim)
    off = b.date.isin(ev.HOLIDAYS) | ((b.date.dt.dayofweek >= 5) & ~b.date.isin(ev.WORKING_SATURDAYS))
    key = [b.route, b.date.dt.month, off]
    scale = b.groupby(key).prediction.transform("sum") / b.groupby(key).gbm.transform("sum").replace(0, np.nan)
    ml = 0.5 * b.gbm * scale.fillna(0) + 0.5 * b.gbm
    b["pred"] = np.where(b.route != 5, (1 - ML_WEIGHT) * b.prediction + ML_WEIGHT * ml, b.prediction)
    return b[KEY + ["pred"]]


def components(final: pd.DataFrame) -> pd.DataFrame:
    no_season = {11: 1.0, 12: 1.0}
    steps = {
        "profile": forecast_nov_dec(season=no_season, special={}, calendar=False, route5=False,
                                    repair_end=pd.Timestamp("2099-01-01")),
        "calendar": forecast_nov_dec(season=no_season, route5=False, repair_end=pd.Timestamp("2099-01-01")),
        "network": forecast_nov_dec(season=no_season),
        "season": forecast_nov_dec(),
        "calibration": s31(),
        "ml": final,
    }
    out, prev = None, None
    for name, f in steps.items():
        cur = f.set_index(KEY).pred.clip(lower=0)
        col = cur if prev is None else cur - prev.reindex(cur.index).fillna(0)
        out = col.rename(name).to_frame() if out is None else out.join(col.rename(name))
        prev = cur
    out["final"] = prev
    return out.reset_index()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "candidate":      # кандидат E8: + световой день и предпраздничные дни
        print(write_submission(with_ml(s31(), extra=True), "E8"))
        sys.exit()
    final = with_ml(s31())
    path = write_submission(final, "E7")
    print(path)
    c = components(final)
    c["date"] = c.date.dt.date
    c.astype({c_: "float32" for c_ in c.columns if c_ not in KEY}).to_parquet("ml/research/components_e7.parquet", index=False)
    tot = c.drop(columns=KEY).sum()
    print("Вклад слоёв за ноябрь–декабрь, посадок:", tot.round(-2).astype(int).to_dict())
    print("Доля |ML| в сумме |вкладов| по ячейкам:", round(c.ml.abs().sum() / c[["calendar", "network", "season", "calibration", "ml"]].abs().sum().sum(), 3))
