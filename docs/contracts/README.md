# Контракты между ML (Степан) и Продуктом (Кирилл)

Это единственная точка связи двух треков. Меняем контракт только по договорённости: правка в этом файле плюс сообщение напарнику. Не согласованные изменения ломают работу другого.

Стек (рекомендованный организаторами, берём как есть):
- **ML:** Python 3.12, uv, polars/duckdb, LightGBM (+ опционально экспорт в ONNX).
- **Backend:** Java 21, Spring Boot 4.1.x, WebFlux (Netty), Gradle.
- **Frontend:** React 19, TypeScript 5, Vite, Leaflet + OpenStreetMap, ECharts.
- **Infra:** Docker, Docker Compose. Нагрузочный тест — k6.

---

## Контракт №1. Артефакты ML → Backend

Папка `artifacts/forecast/` (в git, файлы небольшие). Backend читает её при старте, путь задаётся переменной окружения `FORECAST_DIR`.
Пока ML не выдал настоящие файлы, реальные файлы уже поставлены в `artifacts/forecast/`; генератор синтетического прогноза не используется.

| Файл | Колонки | Покрытие | Горизонт |
|---|---|---|---|
| `hourly.parquet` | `route:int32, date:date, hour:int8, yhat:float32, q10:float32, q90:float32, weather_factor:float32` (для горизонта «день» по умолчанию `yhat × weather_factor`) | 2025-11-01 … 2025-12-31, все 10 маршрутов × 24 ч | **день** (по часам) и **месяц** (агрегация по дням) |
| `daily.parquet` | `route:int32, date:date, yhat:float32, q10:float32, q90:float32` | 2025-11-01 … 2026-10-31 | **год** (агрегация по дням и месяцам) |
| `actuals_hourly.parquet` | `route:int32, date:date, hour:int8, boardings:int32, trams:int16` | 2025-01-01 … 2025-10-31 | история для графиков «факт vs прогноз» и расчёта рекомендаций (`trams` — уникальные вагоны за час) |
| `backtest_hourly.parquet` | `route:int32, date:date, hour:int8, yhat:float32` | 2025-02-03 … 2025-10-31 | прогноз «неделя вперёд» на историю: для **демо реального времени** (реплей факта против прогноза + nowcast) |
| `recommendations.parquet` | `route, date, hour, yhat, trams_plan, trams_needed, trams_delta, boardings_per_tram, norm_bpt, peak_load_pct, dwell_extra_sec, status` | ноябрь–декабрь 2025 | **готовые рекомендации** для `/recommendations`: backend отдаёт как есть (формулы — `ml/recommendations.py`) |
| `stop_shares.parquet` | `route:int32, stop_id:int64, direction:int8, hour:int8, share:float32` | маршруты со справочником | разнесение маршрутного прогноза по остановкам; сумма `share` по (route, hour) = 1 |
| `factors.json` | см. ниже | — | поправочные коэффициенты для ползунков UI |
| `meta.json` | `{"model_version": "...", "trained_at": "...", "cv_wape_score": 0.0, "features": [...]}` | — | отображается в UI и README |

`factors.json`:
```json
[
  {"id": "weather_precip", "label": "Осадки", "group": "weather",
   "default": 1.0, "min": 0.8, "max": 1.2, "step": 0.01,
   "presets": {"сильный снег": 0.93, "дождь": 0.97},
   "source": "https://open-meteo.com/", "effect_note": "−3% при осадках > 2 мм/ч (абляция: WAPE +0.01)"}
]
```
Backend применяет коэффициенты мультипликативно: `yhat_adj = yhat × Π factor_i` (то же для q10 и q90).
Пресеты и `effect_note` ML заполняет из реальных абляций. Это идёт в критерий 2в.

---

## Контракт №2. REST API (Backend → Frontend)

Базовый путь — `/api/v1`. Формальная спецификация будет в `docs/contracts/openapi.yaml` (её ведёт Кирилл, фронт генерирует типы из неё). Ниже — согласованная суть.

| Метод | Параметры | Ответ |
|---|---|---|
| `GET /routes` | — | `[{route, name, has_geo, color}]` |
| `GET /routes/{route}/geometry` | — | GeoJSON: линия маршрута и точки остановок (`stop_id`, `name`, `seq`, `direction`) |
| `GET /forecast` | `route` (можно несколько), `stop_id?`, `from`, `to`, `horizon=day\|month\|year`, `granularity=hour\|day\|month`, `factor.<id>=<value>`… | `{meta, points:[{ts, route, stop_id?, yhat, q10, q90, actual?}]}` |
| `GET /map/snapshot` | `datetime`, `factor.*` | загрузка по всем остановкам и маршрутам на час → раскраска карты |
| `GET /recommendations` | `route`, `date`, `factor.*` | по часам: `{hour, yhat, trams_now, trams_needed, load_pct, dwell_extra_sec}` |
| `GET /stream/replay` (SSE) | `date` (из истории, напр. 2025-10-31), `speed` (×60) | поток событий `{ts, route, hour, boardings_so_far}` из `actuals_hourly` — имитация приёма валидаций |
| `GET /nowcast` | `route`, `date`, `now_hour` | прогноз остатка дня, скорректированный по факту: `k = clip(Σфакт/Σпрогноз с 5:00 до now_hour, 0.5, 1.5)`, `yhat_adj = yhat × (1 + 0.7·(k−1))`; параметры в `meta.json → nowcast` |
| `GET /factors` | — | содержимое `factors.json` |
| `GET /export` | те же параметры, что у `/forecast`, + `format=csv\|xlsx` | файл |
| `GET /health` | — | `{status, model_version}` |

Ошибки: `400` при невалидных параметрах, тело `{error, message}` с понятным текстом на русском. Это требование надёжности из ТЗ.

Рекомендации (логика на backend, константы в конфиге):
- Вместимость Витязь-М: 188 номинально (5 чел/м²), 265 максимум.
- `trams_needed = ceil(yhat × turnover_factor / (capacity × target_load))`. `turnover_factor` — доля посадок, одновременно находящихся в вагоне; начальное значение 0.35, затем калибруем.
- `dwell_extra_sec ≈ (посадок на остановку за рейс − норма) × 1.5 с/пасс.`

**Статус (25.09):** реальные артефакты уже лежат в `artifacts/forecast/` (генерация: `ml/export_artifacts.py`). Фейковый генератор больше не нужен.
Эвристики, которые нужно честно подписать в UI: `stop_shares` — равные доли; `daily` 2026 — «год к году ×1.022» (качественный горизонт).

**Геоданные (25.09):** `artifacts/geo/routes.geojson` — трассы всех 10 маршрутов по направлениям (MultiLineString, свойства `route, direction, name`); `artifacts/geo/stops.parquet` — `route, direction, seq, stop_id (OSM node), name, lat, lon`. `stop_shares.parquet` теперь покрывает все 10 маршрутов (stop_id — OSM).

**Обновление 27.09 (релиз HONEST, LB 0.90174):**
- `hourly.parquet` строится из `ml/submissions/FINAL.csv` (= HONEST): профиль + LightGBM с общим весом 0.5. `meta.json → model_version = honest-profile+lightgbm`. Конкурсный E7 (0.90017) хранится отдельно.
- **Новый** `components.parquet` — «почему такой прогноз»: `route, date, hour, profile, calendar, network, season, calibration, ml, final` (посадок в час; слои складываются: `profile + calendar + network + season + calibration + ml = final`). Для UI: водопад по выбранному маршруту и дню/часу.
- `recommendations.parquet`: **новая колонка** `overload_risk_pct` — вероятность (%) того, что загрузка максимального участка превысит номинал 188 чел. Для карты «где будет давка».
- План/факт по прошедшим дням: `backtest_hourly.parquet` (прогноз «неделя вперёд») + `actuals_hourly.parquet` (факт), февраль–октябрь 2025.

**Обновление 27.09 (сервис, вечер):**
- Весь `/api/v1/**`, кроме `/health`, — HTTP Basic (`dispatcher` / `tram2025` по умолчанию); backend → сервис данных — заголовок `X-Internal-Token`.
- `/health` отдаёт `lb_wape_score`, `cv_wape_score`, `trained_at` из `meta.json` — UI показывает их в шапке, не хардкодить.
- Новые эндпоинты: `/components`, `/plan-fact`, `/anomalies`, `/risk`, `POST /events`, `DELETE /events/{id}` (см. `openapi.yaml`).
- `/forecast` (почасовой) и `/recommendations` отдают `no_service` (trams_plan = 0 или прогноз < 5); `/recommendations` — `overload_risk_pct`.
- `artifacts/forecast/anomalies.json` (ML, `ml/anomalies.py`) — `{method, items: [{route|null, date_from, date_to, kind: ремонт|праздник|изменение трассы|сбой данных, effect_pct, source_url|null, note}]}`. 23 события: история (эффект = факт / медиана того же дня недели за 5 недель до события − 1) и прогнозный период (эффект = вклад слоя модели к профилю).
