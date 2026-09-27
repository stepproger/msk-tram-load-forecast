"""Погода Москвы почасово за 2025 (Open-Meteo Historical, ERA5). https://open-meteo.com/en/docs/historical-weather-api"""
import urllib.request
from pathlib import Path

URL = ("https://archive-api.open-meteo.com/v1/archive?latitude=55.75&longitude=37.62"
       "&start_date=2025-01-01&end_date=2025-12-31"
       "&hourly=temperature_2m,precipitation,snowfall,snow_depth,rain,wind_speed_10m,cloud_cover"
       "&timezone=Europe%2FMoscow&format=csv")
import sys
out = Path("data/external/weather_moscow_2025_hourly.csv")
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists() and "--refresh" not in sys.argv:   # файл в git: воспроизводимо и без сети; --refresh — скачать заново
    print(out, "уже есть (скачать заново: --refresh)")
    sys.exit()
out.write_bytes(urllib.request.urlopen(URL, timeout=60).read())
print(out, out.stat().st_size, "bytes")
