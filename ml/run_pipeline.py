"""Воспроизводимый пайплайн (Windows / Linux / macOS одинаково): сырые данные → внешние данные → бэктест → релиз →
артефакты сервиса → рекомендации → реестр аномалий.

  uv run --python 3.12 --with-requirements ml/requirements.txt python ml/run_pipeline.py

Нужен распакованный dataset.zip в папке dataset/ (train.csv, test.csv, labels/, spravochniki/).
Внешние данные (погода, календарь, OSM) уже лежат в git — сеть не нужна; скачать заново: fetch_*.py --refresh.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ("ml/ingest/convert_raw.py", [], lambda: not all((ROOT / f"data/interim/{n}.parquet").exists() for n in ("train", "test"))),  # ~1 мин, 62 млн строк
    ("ml/ingest/build_target.py", [], None),          # target из сырых данных + сверка с labels
    ("ml/external/fetch_weather.py", [], None),
    ("ml/external/fetch_calendar.py", [], None),
    ("ml/geo/fetch_osm_routes.py", [], lambda: not (ROOT / "artifacts/geo/stops.parquet").exists()),
    ("ml/core.py", ["backtest"], None),
    ("ml/export_artifacts.py", [], None),             # факт + вагоны на линии (нужны аналогу маршрута 5)
    ("ml/honest.py", [], None),                       # релиз: FINAL.csv + components.parquet
    ("ml/export_artifacts.py", ["--forecast-only"], None),
    ("ml/recommendations.py", [], None),
    ("ml/anomalies.py", [], None),
]

if __name__ == "__main__":
    os.chdir(ROOT)
    missing = [p for p in ("dataset/train.csv", "dataset/test.csv", "dataset/labels") if not Path(p).exists()]
    if missing and not Path("data/interim/train.parquet").exists():
        sys.exit(f"Нет {', '.join(missing)}: распакуйте dataset.zip в папку dataset/ в корне репозитория.")
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    for script, args, need in STEPS:
        if need is not None and not need():
            print(f"· {script}: уже готово, пропуск", flush=True)
            continue
        t = time.time()
        print(f"▶ {script} {' '.join(args)}", flush=True)
        subprocess.run([sys.executable, script, *args], check=True, env=env)
        print(f"  {time.time() - t:.0f} с", flush=True)
    print("Готово: ml/submissions/FINAL.csv, artifacts/forecast/")
