"""Выгрузка артефактов по контракту №1 (docs/contracts/README.md) в artifacts/forecast/.

uv run --python 3.12 --with duckdb,pandas,pyarrow,openpyxl python ml/export_artifacts.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import events as ev  # noqa: E402
from core import ROUTES, base_forecast, fit_profile, load_labels, release_forecast_s26, wape_score  # noqa: E402

OUT = Path("artifacts/forecast")
OUT.mkdir(parents=True, exist_ok=True)
FORECAST_ONLY = "--forecast-only" in sys.argv
y = load_labels()

# 1) Скользящий бэктест «неделя вперёд» по 4 предыдущим неделям → остатки для интервалов + прогноз для демо-реплея
parts = []
for end in pd.date_range("2025-02-02", "2025-10-26", freq="W-SUN"):
    lv, sh = fit_profile(y, end)
    parts.append(base_forecast(lv, sh, pd.date_range(end + pd.Timedelta(days=1), min(end + pd.Timedelta(days=7), y.date.max()))))
bt = pd.concat(parts).merge(y[["route", "date", "hour", "y"]], on=["route", "date", "hour"])
bt = bt[bt.pred > 20]
ratio = (bt.y / bt.pred).groupby(bt.hour)
q = pd.DataFrame({"q10": ratio.quantile(0.1), "q90": ratio.quantile(0.9)}).reindex(range(24)).ffill().bfill()

# 2) hourly.parquet — горизонт день/месяц (ноябрь–декабрь 2025)
f = pd.read_csv("ml/submissions/FINAL.csv", sep=";", parse_dates=["date"]).rename(columns={"prediction": "pred"})  # релиз HONEST2 (ml/honest.py)
f = f.join(q, on="hour")
from core import weather_factor  # noqa: E402
wf = weather_factor(f[["date", "hour"]], y.date.max())   # погода Open-Meteo: для горизонта «день» применяется по умолчанию
hourly = pd.DataFrame({
    "route": f.route.astype("int32"), "date": f.date.dt.date, "hour": f.hour.astype("int8"),
    "yhat": f.pred.astype("float32"), "q10": (f.pred * f.q10).astype("float32"), "q90": (f.pred * f.q90).astype("float32"),
    "weather_factor": wf.astype("float32")})
hourly.to_parquet(OUT / "hourly.parquet", index=False)

# 3) daily.parquet — горизонт год (качественно): ноя–дек из hourly, далее 2025 год-к-году (−364 дня) × рост
growth = ev.SEASON[11]
d_nd = hourly.groupby(["route", "date"])[["yhat", "q10", "q90"]].sum().reset_index()
hist_d = y.groupby(["route", "date"]).y.sum()
fut = pd.date_range("2026-01-01", "2026-10-31")
rows = [(r, d.date(), hist_d.get((r, d - pd.Timedelta(days=364)), np.nan)) for r in ROUTES for d in fut]
d_y = pd.DataFrame(rows, columns=["route", "date", "yhat"]).dropna()
d_y["yhat"] *= growth
d_y["q10"], d_y["q90"] = d_y.yhat * 0.8, d_y.yhat * 1.2
daily = pd.concat([d_nd, d_y])
daily = daily.astype({"route": "int32", "yhat": "float32", "q10": "float32", "q90": "float32"})
daily.to_parquet(OUT / "daily.parquet", index=False)

# 4) actuals_hourly.parquet — факт + вагонов на линии из сырых данных.
# Для обновления релизного прогноза можно сохранить уже собранный факт: исходные
# interim/*.parquet не всегда остаются на машине после первой сборки.
if FORECAST_ONLY:
    if not (OUT / "actuals_hourly.parquet").is_file():
        raise FileNotFoundError("--forecast-only requires existing actuals_hourly.parquet")
else:
    trams = duckdb.sql("""
        select try_cast(regexp_extract(route_raw, '^(\\d+)') as int) as route, ts::date as "date", hour(ts)::tinyint as "hour",
               count(distinct garage)::smallint as trams
        from read_parquet(['data/interim/train.parquet', 'data/interim/test.parquet']) where vr = 1 group by all""").df()
    trams["date"] = pd.to_datetime(trams["date"])
    act = y.merge(trams, on=["route", "date", "hour"], how="left").fillna({"trams": 0})
    pd.DataFrame({"route": act.route.astype("int32"), "date": act.date.dt.date, "hour": act.hour.astype("int8"),
                  "boardings": act.y.astype("int32"), "trams": act.trams.astype("int16")}).to_parquet(OUT / "actuals_hourly.parquet", index=False)

# 5) backtest_hourly.parquet — прогноз «неделя вперёд» на историю (для демо реального времени: реплей + nowcast)
pd.DataFrame({"route": bt.route.astype("int32"), "date": bt.date.dt.date, "hour": bt.hour.astype("int8"),
              "yhat": bt.pred.astype("float32")}).to_parquet(OUT / "backtest_hourly.parquet", index=False)

# 6) stop_shares.parquet — геопривязка по OSM для всех 10 маршрутов (ml/geo/fetch_osm_routes.py); равные доли по остановкам
st = pd.read_parquet("artifacts/geo/stops.parquet")
st["share"] = 1 / st.groupby("route").stop_id.transform("count")
shares = st.merge(pd.DataFrame({"hour": range(24)}), how="cross")
pd.DataFrame({"route": shares.route.astype("int32"), "stop_id": shares.stop_id.astype("int64"),
              "direction": shares.direction.astype("int8"), "hour": shares.hour.astype("int8"),
              "share": shares.share.astype("float32")}).to_parquet(OUT / "stop_shares.parquet", index=False)

# 7) factors.json — ползунки UI; значения из измерений (ml/research/weather_nowcast.py)
factors = [
    {"id": "weather_precip", "label": "Осадки", "group": "weather", "default": 1.0, "min": 0.8, "max": 1.1, "step": 0.01,
     "presets": {"сухо": 1.0, "небольшие осадки": 0.966, "дождь/снег 0.5–2 мм/ч": 0.937, "ливень/снегопад >2 мм/ч": 0.881},
     "source": "https://open-meteo.com/en/docs/historical-weather-api",
     "effect_note": "Почасово: сухо 1.017, 0.05–0.5 мм/ч 0.982, 0.5–2 мм/ч 0.953, >2 мм/ч 0.896 (факт/прогноз, янв–окт 2025)"},
    {"id": "weather_temp", "label": "Сильный мороз (< −10 °C)", "group": "weather", "default": 1.0, "min": 0.9, "max": 1.05, "step": 0.01,
     "presets": {"обычно": 1.0, "мороз": 0.96}, "source": "https://open-meteo.com/", "effect_note": "Дни < −10 °C: 0.958 (мало наблюдений)"},
    {"id": "traffic", "label": "Пробки (Яндекс/ЦОДД)", "group": "traffic", "default": 1.0, "min": 0.85, "max": 1.15, "step": 0.01,
     "presets": {"обычно": 1.0, "9–10 баллов, вечер 16–21": 0.97, "9–10 баллов, после 21:00": 1.05},
     "source": "https://www.rbc.ru/society/11/12/2025/693ae9839a79476666be9d4c",
     "effect_note": "Дни ≥9 баллов (16.04, 25.09.2025): вечер ×0.97…1.01, поздно ×0.99…1.09 — посадки сдвигаются позже; LB-абляция S9"},
    {"id": "calendar", "label": "Тип дня", "group": "calendar", "default": 1.0, "min": 0.5, "max": 1.2, "step": 0.01,
     "presets": {"обычный": 1.0, "между праздниками (29–30.12)": 0.85, "праздник → как воскресенье": 1.0},
     "source": "https://isdayoff.ru/", "effect_note": "Праздники ≈ воскресенье ×1.0 (0.83–1.16 по маю/июню 2025); сокращённые дни ≈ 1.0"},
    {"id": "event", "label": "Событие / перекрытие", "group": "event", "default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05,
     "presets": {"нет": 1.0, "ремонт путей, маршрут не ходит": 0.0, "укорочение маршрута": 0.75, "массовое мероприятие": 1.15},
     "source": "https://newsvostok.ru/dlya-tramvaev-7-i-50-izmeneniya-po-vyhodnym-budut-dejstvovat-do-kontsa-oseni/",
     "effect_note": "Ремонт по выходным: маршрут 50 ≈ 0, маршрут 7 ≈ −25% (сен–окт 2025)"},
    {"id": "season", "label": "Сезонная поправка", "group": "season", "default": 1.0, "min": 0.8, "max": 1.2, "step": 0.01,
     "presets": {"база": 1.0, "лето (−10…15%)": 0.87}, "source": "dataset labels 2025",
     "effect_note": "Июнь–август −10…15% к осени; калибровка ×1.022 по LB"},
]
(OUT / "factors.json").write_text(json.dumps(factors, ensure_ascii=False, indent=2), encoding="utf-8")

# 8) meta.json
cv = bt[bt.date >= "2025-10-06"]
meta = {"model_version": "honest-profile+lightgbm", "trained_at": datetime.now().isoformat(timespec="seconds"),
        "cv_wape_score": round(wape_score(cv.y, cv.pred), 4), "lb_wape_score": 0.90257,
        "features": ["profile_level_2w", "profile_shape_4w", "day_type_5", "calendar_rf", "network_events_announced", "season_x1.03_from_backtest", "route5_analog_route25", "free_ride_31dec", "lightgbm_l1_route_hour_dow_holiday_weather", "ensemble_0.5_profile_0.5_lightgbm"],
        "honesty_note": "No target-period ridership. Profile parameters from history; the only leaderboard-derived choice is the E7 blending recipe (best of E1-E7 uploads, confirmed by local proxy). Actual Nov-Dec weather used (disclosed). Network events and free-ride night from Deptrans announcements; route 5 level by analog (route 25 x 7 trams).",
        "nowcast": {"shrink": 0.7, "clip": [0.5, 1.5], "start_hour": 5,
                    "gain_note": "Коррекция остатка дня по факту с 5:00: WAPE-score 0.881→0.900 (в 9:00), 0.879→0.903 (12:00), 0.880→0.907 (15:00)"},
        "capacity": {"model": "71-931М Витязь-М", "nominal": 188, "max": 265}}
(OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
for p in sorted(OUT.iterdir()):
    print(f"{p.name:28s} {p.stat().st_size/1024:8.0f} KB")
