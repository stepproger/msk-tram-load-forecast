"""Климатическая норма погоды (без фактов прогнозного периода): Open-Meteo Historical (ERA5), 2015–2024.
Норма по (месяц, день, час): среднее температуры, осадков, снега. Используется как погода LightGBM
в прогнозном периоде вместо фактической — так прогноз строится только на том, что известно на 31.10.2025.
https://open-meteo.com/en/docs/historical-weather-api
"""
import io
import urllib.request
from pathlib import Path

import pandas as pd

URL = ("https://archive-api.open-meteo.com/v1/archive?latitude=55.75&longitude=37.62"
       "&start_date=2015-01-01&end_date=2024-12-31&hourly=temperature_2m,precipitation,snowfall"
       "&timezone=Europe%2FMoscow&format=csv")
raw = Path("data/external/weather_moscow_2015_2024_hourly.csv")
if not raw.exists():
    raw.write_bytes(urllib.request.urlopen(URL, timeout=180).read())
w = pd.read_csv(raw, skiprows=3)
w.columns = ["time", "t", "prcp", "snow"]
w["time"] = pd.to_datetime(w.time)
w["month"], w["day"], w["hour"] = w.time.dt.month, w.time.dt.day, w.time.dt.hour
clim = w.groupby(["month", "day", "hour"])[["t", "prcp", "snow"]].mean().reset_index()
clim.to_csv("data/external/weather_climatology_2015_2024.csv", index=False)
print(len(w), "часов →", len(clim), "строк нормы; ноябрь t =", round(clim[clim.month == 11].t.mean(), 1),
      "декабрь t =", round(clim[clim.month == 12].t.mean(), 1))
