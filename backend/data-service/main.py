from __future__ import annotations

import hmac
import json
import uuid
import math
import os
from statistics import median
from collections import OrderedDict, defaultdict
from contextlib import asynccontextmanager
from datetime import date, datetime, time
from io import BytesIO
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
from fastapi import FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, ORJSONResponse


ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]
# Hours with no scheduled trams or under 5 predicted boardings are "no service", never "zero load".
NO_SERVICE_THRESHOLD = 5
ROUTE_COLORS = {
    1: "#16A6A1", 5: "#8659E8", 7: "#E6A535", 11: "#F16B5B", 12: "#5188E8",
    17: "#D64E81", 25: "#56A96A", 26: "#DB7841", 28: "#6878CE", 50: "#52A6C6",
}


def load_route_geometry(path: Path) -> tuple[dict[int, list[dict[str, Any]]], dict[int, dict[str, Any]]]:
    geo_dir = Path(os.getenv("GEO_DIR", "/data/geo"))
    routes_path = geo_dir / "routes.geojson"
    stops_path = geo_dir / "stops.parquet"
    if routes_path.is_file() and stops_path.is_file():
        route_features: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for feature in json.loads(routes_path.read_text(encoding="utf-8"))["features"]:
            route = int(feature["properties"]["route"])
            if route in ROUTES:
                route_features[route].append(feature)
        for stop in pl.read_parquet(stops_path).iter_rows(named=True):
            route = int(stop["route"])
            if route not in ROUTES:
                continue
            route_features[route].append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(stop["lon"]), float(stop["lat"])]},
                "properties": {
                    "route": route, "stop_id": int(stop["stop_id"]), "name": stop["name"],
                    "seq": int(stop["seq"]), "direction": int(stop["direction"]), "kind": "stop",
                },
            })
        route_info = {
            route: {"route": route, "name": f"Трамвай {route}", "has_geo": route in route_features,
                    "color": ROUTE_COLORS[route]}
            for route in ROUTES
        }
        return dict(route_features), route_info

    if not path.is_file():
        return {}, {}
    try:
        frame = pl.read_excel(path, sheet_name="Порядок_с_координатами")
    except Exception:
        return {}, {}

    normalized = {name.strip().casefold().replace(" ", "_"): name for name in frame.columns}

    def column(*candidates: str) -> str | None:
        return next((normalized[name] for name in candidates if name in normalized), None)

    route_col = column("route_short_name", "route", "route_id")
    stop_col = column("stop_id", "id_остановки")
    direction_col = column("direction_id", "direction", "направление")
    sequence_col = column("stop_sequence", "seq", "sequence", "порядок")
    name_col = column("stop_name", "name", "название_остановки")
    lat_col = column("stop_lat", "latitude", "lat", "широта")
    lon_col = column("stop_lon", "longitude", "lon", "долгота")
    if not all((route_col, stop_col, direction_col, sequence_col, name_col, lat_col, lon_col)):
        return {}, {}

    rows = frame.select(
        pl.col(route_col).cast(pl.Int32, strict=False).alias("route"),
        pl.col(stop_col).cast(pl.Int64, strict=False).alias("stop_id"),
        pl.col(direction_col).cast(pl.Int8, strict=False).alias("direction"),
        pl.col(sequence_col).cast(pl.Int32, strict=False).alias("seq"),
        pl.col(name_col).cast(pl.String, strict=False).alias("name"),
        pl.col(lat_col).cast(pl.Float64, strict=False).alias("lat"),
        pl.col(lon_col).cast(pl.Float64, strict=False).alias("lon"),
    ).drop_nulls().filter(pl.col("route").is_in(ROUTES)).to_dicts()

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["route"], row["direction"])].append(row)
    features_by_route: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (route, direction), stops in grouped.items():
        stops.sort(key=lambda stop: stop["seq"])
        features_by_route[route].append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[stop["lon"], stop["lat"]] for stop in stops]},
            "properties": {"route": route, "direction": direction, "kind": "route"},
        })
        for stop in stops:
            features_by_route[route].append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [stop["lon"], stop["lat"]]},
                "properties": {
                    "route": route, "stop_id": stop["stop_id"], "name": stop["name"],
                    "seq": stop["seq"], "direction": direction, "kind": "stop",
                },
            })
    route_info = {
        route: {"route": route, "name": f"Трамвай {route}", "has_geo": route in features_by_route,
                "color": ROUTE_COLORS[route]}
        for route in ROUTES
    }
    return dict(features_by_route), route_info


def estimate_stop_shares(shares: pl.DataFrame, geometry: dict[int, list[dict[str, Any]]]) -> pl.DataFrame:
    """Spatial prior when validation records have no stop identifier."""
    by_direction: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    routes_by_name: dict[str, set[int]] = defaultdict(set)
    for route, features in geometry.items():
        for feature in features:
            props = feature["properties"]
            if props.get("kind") != "stop":
                continue
            direction = int(props["direction"])
            by_direction[(route, direction)].append(feature)
            name = str(props.get("name") or "").strip().casefold()
            if name:
                routes_by_name[name].add(route)

    spacing_by_stop: dict[tuple[int, int, int], float] = {}
    names_by_stop: dict[tuple[int, int, int], str] = {}
    for (route, direction), features in by_direction.items():
        ordered = sorted(features, key=lambda feature: int(feature["properties"]["seq"]))
        for index, feature in enumerate(ordered):
            lon, lat = feature["geometry"]["coordinates"]
            gaps = []
            for neighbor_index in (index - 1, index + 1):
                if 0 <= neighbor_index < len(ordered):
                    other_lon, other_lat = ordered[neighbor_index]["geometry"]["coordinates"]
                    gaps.append(math.hypot((lon - other_lon) * math.cos(math.radians(lat)), lat - other_lat))
            key = (route, int(feature["properties"]["stop_id"]), direction)
            spacing_by_stop[key] = sum(gaps) / len(gaps) if gaps else 0.0
            names_by_stop[key] = str(feature["properties"].get("name") or "").strip().casefold()

    route_medians: dict[int, float] = {}
    for route in geometry:
        values = [value for key, value in spacing_by_stop.items() if key[0] == route and value > 0]
        route_medians[route] = median(values) if values else 1.0

    weights: dict[tuple[int, int, int], float] = {}
    for key, spacing in spacing_by_stop.items():
        route = key[0]
        relative_spacing = min(2.5, max(0.4, spacing / route_medians[route]))
        connections = len(routes_by_name.get(names_by_stop[key], {route}))
        weights[key] = (0.7 + 0.3 * math.sqrt(relative_spacing)) * (1 + 0.22 * min(3, connections - 1))

    rows = shares.to_dicts()
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["route"]), int(row["hour"]))].append(row)
    for group in grouped.values():
        if len({round(float(row["share"]), 7) for row in group}) > 1:
            continue
        total = sum(weights.get((int(row["route"]), int(row["stop_id"]), int(row["direction"])), 1.0)
                    for row in group)
        for row in group:
            key = (int(row["route"]), int(row["stop_id"]), int(row["direction"]))
            row["share"] = weights.get(key, 1.0) / total
    return pl.DataFrame(rows).select(shares.columns)


COMPONENT_LAYERS = [
    ("calendar", "Календарь", "Праздники РФ, переносы, 29–31.12."),
    ("network", "События сети", "Окончание ремонта 7/50, запуск маршрута 5 с 16.12."),
    ("season", "Сезон", "Сезонный множитель ×1.022."),
    ("calibration", "Уточнения", "Опубликованные данные по маршруту 5 и 01.11."),
    ("ml", "ML (LightGBM)", "LightGBM по всей истории: почасовая структура, календарь, погода."),
]
PROFILE_NOTE = "Уровень за 2 нед. × форма суток за 4 нед."


def check_route(route: int | None) -> None:
    if route is not None and route not in ROUTES:
        raise HTTPException(
            status_code=400,
            detail=f"Маршрут {route} не найден. Доступные: {', '.join(map(str, ROUTES))}",
        )


def load_components(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    frame = pl.read_parquet(path).with_columns(pl.col("date").cast(pl.Date))
    adjustments = {layer: float(frame[layer].abs().sum()) for layer, _, _ in COMPONENT_LAYERS}
    total = sum(adjustments.values())
    return {
        "frame": frame,
        "from": frame["date"].min(), "to": frame["date"].max(),
        # Share of all absolute corrections to the profile that comes from each layer, whole horizon.
        "shares": {layer: (value / total * 100 if total else 0.0) for layer, value in adjustments.items()},
    }


def load_dataset(directory: Path) -> dict[str, Any]:
    required = ["actuals_hourly.parquet", "backtest_hourly.parquet", "hourly.parquet", "daily.parquet", "stop_shares.parquet", "meta.json"]
    for name in required:
        if not (directory / name).is_file():
            raise RuntimeError(f"Required forecast artifact is missing: {name}")

    actuals = pl.read_parquet(directory / "actuals_hourly.parquet").with_columns(pl.col("date").cast(pl.Date))
    backtest = pl.read_parquet(directory / "backtest_hourly.parquet").with_columns(pl.col("date").cast(pl.Date))
    hourly = pl.read_parquet(directory / "hourly.parquet").with_columns(pl.col("date").cast(pl.Date))
    daily = pl.read_parquet(directory / "daily.parquet").with_columns(pl.col("date").cast(pl.Date))
    shares = pl.read_parquet(directory / "stop_shares.parquet")
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    factors_path = directory / "factors.json"
    factors = json.loads(factors_path.read_text(encoding="utf-8")) if factors_path.is_file() else []
    events_path = directory / "events.json"
    events = json.loads(events_path.read_text(encoding="utf-8")) if events_path.is_file() else []
    recommendations_path = directory / "recommendations.parquet"
    recommendations = pl.read_parquet(recommendations_path).with_columns(
        pl.col("date").cast(pl.Date)
    ) if recommendations_path.is_file() else None

    residuals = (
        backtest.join(actuals.select("route", "date", "hour", "boardings"),
                      on=["route", "date", "hour"], how="inner")
        .filter(pl.col("yhat") > 20)
        .with_columns((pl.col("boardings") / pl.col("yhat")).alias("ratio"))
    )
    intervals = {
        int(row["hour"]): (float(row["q10"]), float(row["q90"]))
        for row in residuals.group_by("hour").agg(
            pl.col("ratio").quantile(0.1).alias("q10"),
            pl.col("ratio").quantile(0.9).alias("q90"),
        ).iter_rows(named=True)
    }
    route_intervals = {
        (int(row["route"]), int(row["hour"])): (float(row["q10"]), float(row["q90"]))
        for row in residuals.group_by("route", "hour").agg(
            pl.col("ratio").quantile(0.1).alias("q10"),
            pl.col("ratio").quantile(0.9).alias("q90"),
        ).iter_rows(named=True)
    }
    network_residuals = (
        residuals.group_by("date", "hour").agg(pl.col("yhat").sum(), pl.col("boardings").sum())
        .with_columns((pl.col("boardings") / pl.col("yhat")).alias("ratio"))
    )
    route_intervals.update({
        (0, int(row["hour"])): (float(row["q10"]), float(row["q90"]))
        for row in network_residuals.group_by("hour").agg(
            pl.col("ratio").quantile(0.1).alias("q10"),
            pl.col("ratio").quantile(0.9).alias("q90"),
        ).iter_rows(named=True)
    })
    service_parts = [actuals.select(
        "route", "date", "hour", pl.col("trams").cast(pl.Float64).alias("trams_scheduled"))]
    if recommendations is not None:
        service_parts.append(recommendations.select(
            pl.col("route").cast(pl.Int32), "date", pl.col("hour").cast(pl.Int8),
            pl.col("trams_plan").cast(pl.Float64).alias("trams_scheduled")))
    service = pl.concat(service_parts, how="vertical_relaxed").unique(["route", "date", "hour"], keep="last")
    anomalies_path = directory / "anomalies.json"
    anomalies = json.loads(anomalies_path.read_text(encoding="utf-8")) if anomalies_path.is_file() else None
    if isinstance(anomalies, dict):
        anomalies = anomalies.get("items", [])
    reference = Path(os.getenv("REFERENCE_XLSX_PATH", "/data/spravochniki/Хакатон_справочники_трамвай_10_маршрутов.xlsx"))
    geometry, route_info = load_route_geometry(reference)
    shares = estimate_stop_shares(shares, geometry)
    geometry_stops = {
        (route, int(feature["properties"]["stop_id"]), int(feature["properties"]["direction"])): feature
        for route, features in geometry.items()
        for feature in features
        if feature["properties"].get("kind") == "stop"
    }
    shares_by_hour = {
        int(hour): frame.drop("hour")
        for (hour,), frame in shares.partition_by("hour", as_dict=True).items()
    }
    aggregate_shares = shares.group_by("route", "stop_id", "direction").agg(pl.col("share").mean())
    historical = backtest.with_columns(
        (pl.col("yhat") * 0.8).alias("q10"),
        (pl.col("yhat") * 1.2).alias("q90"),
    )
    historical_daily = historical.group_by("route", "date").agg(
        pl.col("yhat").sum(), pl.col("q10").sum(), pl.col("q90").sum()
    )
    return {
        "actuals": actuals, "backtest": backtest,
        "hourly": pl.concat([historical, hourly], how="diagonal_relaxed"),
        "daily": pl.concat([historical_daily, daily], how="diagonal_relaxed"), "shares": shares,
        "meta": meta, "factors": factors, "intervals": intervals,
        "events": events, "recommendations": recommendations,
        "geometry": geometry, "geometry_stops": geometry_stops,
        "shares_by_hour": shares_by_hour, "aggregate_shares": aggregate_shares,
        "routes": route_info,
        "components": load_components(directory / "components.parquet"),
        "route_intervals": route_intervals, "anomalies": anomalies, "service": service,
        "backtest_from": backtest["date"].min(), "backtest_to": backtest["date"].max(),
    }


INTERNAL_TOKEN_HEADER = "X-Internal-Token"
# Dispatcher events are shared by all uvicorn workers through one file; its mtime versions the cache.
USER_EVENTS_FILE = Path(os.getenv("USER_EVENTS_FILE", "/tmp/tram-user-events.json"))


def sync_user_events(state) -> None:
    try:
        mtime = USER_EVENTS_FILE.stat().st_mtime_ns
    except FileNotFoundError:
        mtime = None
    if mtime != state.events_mtime:
        state.user_events = json.loads(USER_EVENTS_FILE.read_text(encoding="utf-8")) if mtime else {}
        state.events_mtime = mtime


def save_user_events(state) -> None:
    temporary = USER_EVENTS_FILE.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state.user_events, ensure_ascii=False), encoding="utf-8")
    temporary.replace(USER_EVENTS_FILE)
    sync_user_events(state)


@asynccontextmanager
async def lifespan(app: FastAPI):
    token = os.getenv("INTERNAL_TOKEN", "")
    if not token:
        raise RuntimeError("INTERNAL_TOKEN must be set: the data service accepts only authenticated backend calls")
    app.state.internal_token = token.encode("utf-8")
    directory = Path(os.getenv("FORECAST_DIR", "/data/forecast"))
    app.state.dataset = load_dataset(directory)
    app.state.user_events = {}
    app.state.events_mtime = None
    sync_user_events(app.state)
    yield
    app.state.dataset = None


app = FastAPI(title="Forecast data adapter", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan,
              default_response_class=ORJSONResponse)

# Forecasts are precomputed, so a GET response depends only on its URL and the dispatcher events.
CACHE_MAX_BYTES = int(os.getenv("CACHE_MAX_MB", "64")) * 1024 * 1024  # per worker
CACHE_MAX_BODY = 2 * 1024 * 1024
UNCACHED_PREFIXES = ("/internal/health", "/internal/replay-events")


class InternalGateway:
    """Pure ASGI middleware: internal token check and an LRU cache of successful GET responses."""

    def __init__(self, app):
        self.app = app
        self.cache: OrderedDict[tuple, tuple[list, bytes]] = OrderedDict()
        self.cache_bytes = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        state = scope["app"].state
        supplied = next((value for name, value in scope["headers"] if name == b"x-internal-token"), b"")
        if not hmac.compare_digest(supplied, state.internal_token):
            response = JSONResponse(status_code=401, content={"detail": "Нет доступа: нужен межсервисный токен X-Internal-Token"})
            return await response(scope, receive, send)
        sync_user_events(state)
        path = scope["path"]
        if scope["method"] != "GET" or path.startswith(UNCACHED_PREFIXES):
            return await self.app(scope, receive, send)
        key = (path, scope["query_string"], state.events_mtime)
        hit = self.cache.get(key)
        if hit is not None:
            self.cache.move_to_end(key)
            await send({"type": "http.response.start", "status": 200, "headers": hit[0]})
            return await send({"type": "http.response.body", "body": hit[1]})
        captured: dict[str, Any] = {"chunks": [], "size": 0}

        async def capture(message):
            if message["type"] == "http.response.start":
                captured["start"] = message
            elif message["type"] == "http.response.body":
                captured["size"] += len(message.get("body", b""))
                if captured["size"] <= CACHE_MAX_BODY:
                    captured["chunks"].append(message.get("body", b""))
                if not message.get("more_body") and captured["start"]["status"] == 200 and captured["size"] <= CACHE_MAX_BODY:
                    body = b"".join(captured["chunks"])
                    self.cache[key] = (captured["start"]["headers"], body)
                    self.cache_bytes += len(body)
                    while self.cache_bytes > CACHE_MAX_BYTES and self.cache:
                        self.cache_bytes -= len(self.cache.popitem(last=False)[1][1])
            await send(message)

        await self.app(scope, receive, capture)


app.add_middleware(InternalGateway)


@app.exception_handler(RequestValidationError)
async def invalid_parameters(_request: Request, error: RequestValidationError) -> JSONResponse:
    names = sorted({str(item["loc"][-1]) for item in error.errors() if item.get("loc")})
    detail = f"Некорректный или отсутствующий параметр: {', '.join(names)}" if names else "Некорректные параметры запроса"
    return JSONResponse(status_code=400, content={"detail": detail})


def dataset(request: Request) -> dict[str, Any]:
    return request.app.state.dataset


def factor_multiplier(request: Request) -> float:
    data = dataset(request)
    by_id = {factor["id"]: factor for factor in data["factors"]}
    multiplier = 1.0
    for key, raw_value in request.query_params.items():
        if not key.startswith("factor."):
            continue
        factor_id = key.removeprefix("factor.")
        factor = by_id.get(factor_id)
        if factor is None:
            raise HTTPException(status_code=400, detail=f"Неизвестный коэффициент: {factor_id}")
        try:
            value = float(raw_value)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=f"Некорректное значение коэффициента {factor_id}") from error
        if value < float(factor["min"]) or value > float(factor["max"]):
            raise HTTPException(status_code=400, detail=f"Коэффициент {factor_id} вне допустимого диапазона")
        multiplier *= value
    return multiplier


class UserEvent(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    start: date
    end: date
    routes: list[int] = Field(default_factory=list)
    category: str = Field(default="event", pattern="^(repair|launch|event)$")
    effect_pct: float = Field(ge=-90, le=200)


EVENT_CATEGORY_LABELS = {"repair": "Ремонт", "launch": "Запуск / изменение трассы", "event": "Мероприятие"}


def event_factor(request: Request) -> pl.Expr | None:
    """Multiplier for rows with route/date columns from dispatcher-added events."""
    events = list(request.app.state.user_events.values())
    if not events:
        return None
    factor = pl.lit(1.0)
    for event in events:
        in_period = pl.col("date").is_between(date.fromisoformat(event["start"]), date.fromisoformat(event["end"]), closed="both")
        if event["routes"]:
            in_period = in_period & pl.col("route").is_in(event["routes"])
        factor = factor * pl.when(in_period).then(1 + event["effect_pct"] / 100).otherwise(1.0)
    return factor


def with_user_events(request: Request, frame: pl.DataFrame) -> pl.DataFrame:
    factor = event_factor(request)
    if factor is None or frame.is_empty():
        return frame
    columns = [name for name in ("yhat", "q10", "q90") if name in frame.columns]
    return frame.with_columns(factor.alias("_event")).with_columns(
        (pl.col(name) * pl.col("_event")).alias(name) for name in columns).drop("_event")


def route_day_event_factor(request: Request, route: int, day: date) -> float:
    factor = 1.0
    for event in request.app.state.user_events.values():
        if event["start"] <= day.isoformat() <= event["end"] and (not event["routes"] or route in event["routes"]):
            factor *= 1 + event["effect_pct"] / 100
    return factor


@app.get("/internal/health")
def health(request: Request) -> dict[str, Any]:
    meta = dataset(request)["meta"]
    return {
        "status": "ok", "model_version": str(meta.get("model_version", "unknown")),
        "lb_wape_score": meta.get("lb_wape_score"), "cv_wape_score": meta.get("cv_wape_score"),
        "trained_at": meta.get("trained_at"),
    }


@app.get("/internal/routes")
def routes(request: Request) -> list[dict[str, Any]]:
    return list(dataset(request)["routes"].values())


@app.get("/internal/routes/{route}/geometry")
def route_geometry(route: int, request: Request) -> dict[str, Any]:
    check_route(route)
    features = dataset(request)["geometry"].get(route)
    if not features:
        raise HTTPException(status_code=404, detail=f"Для маршрута {route} нет геометрии в справочнике")
    return {"type": "FeatureCollection", "features": features}


@app.get("/internal/map/snapshot")
def map_snapshot(
    request: Request,
    snapshot_at: datetime = Query(alias="datetime"),
    period: str = Query(default="hour", pattern="^(hour|day|month)$"),
) -> dict[str, Any]:
    data = dataset(request)
    day = snapshot_at.date()
    hour = snapshot_at.hour
    if period == "hour":
        predicted = with_user_events(request, data["hourly"].filter((pl.col("date") == day) & (pl.col("hour") == hour)))
        facts = data["actuals"].filter((pl.col("date") == day) & (pl.col("hour") == hour))
        shares = data["shares_by_hour"].get(hour)
        if shares is None:
            shares = data["shares"].head(0).drop("hour")
    elif period == "day":
        predicted = with_user_events(request, data["hourly"].filter(pl.col("date") == day)).group_by("route").agg(
            pl.col("yhat").sum(), pl.col("q10").sum(), pl.col("q90").sum()
        )
        if predicted.is_empty():
            predicted = with_user_events(request, data["daily"].filter(pl.col("date") == day)).select("route", "yhat", "q10", "q90")
        facts = data["actuals"].filter(pl.col("date") == day)
        shares = data["aggregate_shares"]
    else:
        year, month = day.year, day.month
        predicted = with_user_events(request, data["daily"].filter(
            (pl.col("date").dt.year() == year) & (pl.col("date").dt.month() == month)
        )).group_by("route").agg(
            pl.col("yhat").sum(), pl.col("q10").sum(), pl.col("q90").sum()
        )
        facts = data["actuals"].filter(
            (pl.col("date").dt.year() == year) & (pl.col("date").dt.month() == month)
        )
        shares = data["aggregate_shares"]

    predictions = {int(row["route"]): row for row in predicted.iter_rows(named=True)}
    actuals = {
        int(row["route"]): int(row["boardings"])
        for row in facts.group_by("route").agg(pl.col("boardings").sum()).iter_rows(named=True)
    }
    if not predictions and not actuals:
        return {"type": "FeatureCollection", "features": []}

    factor = factor_multiplier(request)
    features = []
    for share in shares.iter_rows(named=True):
        route = int(share["route"])
        stop_id = int(share["stop_id"])
        direction = int(share["direction"])
        source = data["geometry_stops"].get((route, stop_id, direction))
        prediction = predictions.get(route)
        actual = actuals.get(route)
        if source is None or (prediction is None and actual is None):
            continue
        share_value = float(share["share"])
        is_fact = prediction is None
        base = float(actual) if is_fact else float(prediction["yhat"]) * factor
        q10 = None if is_fact else max(0.0, float(prediction["q10"]) * share_value * factor)
        q90 = None if is_fact else max(q10, float(prediction["q90"]) * share_value * factor)
        props = {
            "route": route, "stop_id": stop_id, "direction": direction,
            "yhat": max(0.0, base * share_value),
            "q10": q10, "q90": q90,
            "actual": None if actual is None else round(actual * share_value),
            "estimated_stop_load": True,
            "source": "actual" if is_fact else "forecast",
            "period": period,
        }
        features.append({"type": "Feature", "geometry": source["geometry"], "properties": props})
    return {"type": "FeatureCollection", "features": features}


@app.get("/internal/factors")
def get_factors(request: Request) -> list[dict[str, Any]]:
    return dataset(request)["factors"]


@app.get("/internal/events")
def get_events(request: Request) -> list[dict[str, Any]]:
    return [*dataset(request)["events"], *request.app.state.user_events.values()]


@app.post("/internal/events", status_code=201)
def add_event(event: UserEvent, request: Request) -> dict[str, Any]:
    if event.start > event.end:
        raise HTTPException(status_code=400, detail="Дата начала события позже даты окончания")
    for route in event.routes:
        check_route(route)
    if len(request.app.state.user_events) >= 50:
        raise HTTPException(status_code=400, detail="Слишком много пользовательских событий (максимум 50)")
    event_id = f"user-{uuid.uuid4().hex[:10]}"
    sign = "+" if event.effect_pct >= 0 else "−"
    stored = {
        "id": event_id, "title": event.title, "start": event.start.isoformat(), "end": event.end.isoformat(),
        "routes": event.routes, "category": event.category, "effect_pct": event.effect_pct,
        "effect": f"{EVENT_CATEGORY_LABELS[event.category]}: {sign}{abs(event.effect_pct):g}% к прогнозу посадок",
        "source_label": "Добавлено диспетчером", "source_url": "", "user": True,
    }
    request.app.state.user_events[event_id] = stored
    save_user_events(request.app.state)
    return stored


@app.delete("/internal/events/{event_id}")
def delete_event(event_id: str, request: Request) -> dict[str, str]:
    if request.app.state.user_events.pop(event_id, None) is None:
        raise HTTPException(status_code=404, detail="Событие не найдено или не является пользовательским")
    save_user_events(request.app.state)
    return {"deleted": event_id}




@app.get("/internal/components")
def components(
    request: Request,
    component_date: date = Query(alias="date"),
    route: int | None = None,
    hour: int | None = Query(default=None, ge=0, le=23),
) -> dict[str, Any]:
    check_route(route)
    source = dataset(request)["components"]
    if source is None:
        raise HTTPException(status_code=404, detail="Разложение прогноза (components.parquet) не загружено")
    if not source["from"] <= component_date <= source["to"]:
        raise HTTPException(
            status_code=400,
            detail=f"Разложение прогноза доступно с {source['from']:%d.%m.%Y} по {source['to']:%d.%m.%Y}",
        )
    frame = source["frame"].filter(pl.col("date") == component_date)
    if route is not None:
        frame = frame.filter(pl.col("route") == route)
    if hour is not None:
        frame = frame.filter(pl.col("hour") == hour)
    names = ["profile", *(layer for layer, _, _ in COMPONENT_LAYERS), "final"]
    totals = frame.select(pl.col(name).sum() for name in names).row(0, named=True)
    return {
        "route": route, "date": component_date.isoformat(), "hour": hour,
        "profile": {"value": float(totals["profile"] or 0), "label": "Профиль", "note": PROFILE_NOTE},
        "layers": [
            {"id": layer, "label": label, "note": note, "value": float(totals[layer] or 0),
             "share_of_corrections_pct": round(source["shares"][layer], 1)}
            for layer, label, note in COMPONENT_LAYERS
        ],
        "final": float(totals["final"] or 0),
    }


@app.get("/internal/plan-fact")
def plan_fact(
    request: Request,
    from_date: date = Query(alias="from"),
    to_date: date = Query(alias="to"),
    route: int | None = None,
) -> dict[str, Any]:
    """Week-ahead backtest plan vs actual validations with the empirical q10–q90 band."""
    check_route(route)
    data = dataset(request)
    first, last = data["backtest_from"], data["backtest_to"]
    if from_date > to_date:
        raise HTTPException(status_code=400, detail="Параметр from не может быть позже to")
    if from_date < first or to_date > last:
        raise HTTPException(status_code=400, detail=f"План/факт доступен с {first:%d.%m.%Y} по {last:%d.%m.%Y}")
    if (to_date - from_date).days > 61:
        raise HTTPException(status_code=400, detail="Интервал план/факт — не больше 62 дней")
    plan = data["backtest"].filter(pl.col("date").is_between(from_date, to_date, closed="both"))
    fact = data["actuals"].filter(pl.col("date").is_between(from_date, to_date, closed="both"))
    if route is not None:
        plan = plan.filter(pl.col("route") == route)
        fact = fact.filter(pl.col("route") == route)
    plan = plan.group_by("date", "hour").agg(pl.col("yhat").sum())
    fact = fact.group_by("date", "hour").agg(pl.col("boardings").sum(), pl.col("trams").sum())
    rows = plan.join(fact, on=["date", "hour"], how="full", coalesce=True).sort("date", "hour")
    intervals = data["route_intervals"]
    points = []
    covered = counted = 0
    abs_error = fact_total = 0.0
    for row in rows.iter_rows(named=True):
        yhat, actual, trams = row["yhat"], row["boardings"], row["trams"]
        hour = int(row["hour"])
        no_service = (actual is None or actual < NO_SERVICE_THRESHOLD) and (
            trams == 0 or yhat is None or yhat < NO_SERVICE_THRESHOLD)
        q10_ratio, q90_ratio = intervals.get((route or 0, hour), (0.8, 1.2))
        q10 = None if yhat is None else max(0.0, float(yhat) * q10_ratio)
        q90 = None if yhat is None else max(0.0, float(yhat) * q90_ratio)
        outside = None
        if not no_service and yhat is not None and actual is not None:
            outside = "above" if actual > q90 else "below" if actual < q10 else None
            counted += 1
            covered += outside is None
            abs_error += abs(float(yhat) - float(actual))
            fact_total += float(actual)
        points.append({
            "ts": f"{row['date'].isoformat()}T{hour:02d}:00:00+03:00",
            "plan": None if yhat is None else float(yhat), "fact": actual,
            "q10": q10, "q90": q90, "trams": trams,
            "no_service": no_service, "outside": outside,
        })
    return {
        "route": route, "from": from_date.isoformat(), "to": to_date.isoformat(),
        "coverage_pct": round(covered / counted * 100, 1) if counted else None,
        "hours_with_service": counted,
        "wape_score": round(1 - abs_error / fact_total, 4) if fact_total else None,
        "band_note": "q10–q90 отношения факт/план за февраль–октябрь 2025 по часу суток",
        "points": points,
    }


@app.get("/internal/anomalies")
def anomalies(request: Request, route: int | None = None) -> dict[str, Any]:
    check_route(route)
    items = dataset(request)["anomalies"]
    if items is None:
        return {"available": False, "items": []}
    if route is not None:
        items = [item for item in items if item.get("route") in (None, route)]
    return {"available": True, "items": sorted(items, key=lambda item: str(item.get("date_from", "")))}


@app.get("/internal/replay-events")
def replay_events(request: Request, replay_date: date = Query(alias="date")) -> list[dict[str, Any]]:
    data = dataset(request)
    day = data["actuals"].filter(pl.col("date") == replay_date)
    if day.is_empty():
        raise HTTPException(status_code=404, detail="Для этой даты нет исторических фактов")

    counts = {
        (int(row["route"]), int(row["hour"])): int(row["boardings"])
        for row in day.select("route", "hour", "boardings").iter_rows(named=True)
    }
    routes_for_day = sorted(set(ROUTES).intersection(int(v) for v in day.get_column("route").unique().to_list()))
    totals = {route: 0 for route in routes_for_day}
    events: list[dict[str, Any]] = []
    for hour in range(24):
        for route in routes_for_day:
            totals[route] += counts.get((route, hour), 0)
            timestamp = datetime.combine(replay_date, time(hour=hour), tzinfo=ZoneInfo("Europe/Moscow")).isoformat()
            events.append({"ts": timestamp, "route": route, "hour": hour, "boardings_so_far": totals[route]})
    return events


@app.get("/internal/nowcast-inputs")
def nowcast_inputs(
    request: Request,
    route: int,
    forecast_date: date = Query(alias="date"),
) -> dict[str, Any]:
    check_route(route)
    data = dataset(request)
    actual_rows = data["actuals"].filter((pl.col("route") == route) & (pl.col("date") == forecast_date))
    baseline_rows = data["backtest"].filter((pl.col("route") == route) & (pl.col("date") == forecast_date))
    if actual_rows.is_empty() or baseline_rows.is_empty():
        raise HTTPException(status_code=404, detail="Для маршрута и даты нет факта или базового прогноза")

    actuals = {
        int(row["hour"]): int(row["boardings"])
        for row in actual_rows.select("hour", "boardings").iter_rows(named=True)
    }
    baseline_by_hour = {
        int(row["hour"]): float(row["yhat"])
        for row in baseline_rows.select("hour", "yhat").iter_rows(named=True)
    }
    intervals = data["intervals"]
    baseline = []
    for hour in range(24):
        q10, q90 = intervals.get(hour, (0.8, 1.2))
        baseline.append({"hour": hour, "yhat": baseline_by_hour.get(hour, 0.0),
                         "q10_ratio": q10, "q90_ratio": q90})
    return {
        "route": route,
        "date": forecast_date.isoformat(),
        "actuals": [{"hour": hour, "boardings": actuals.get(hour, 0)} for hour in range(24)],
        "baseline": baseline,
        "nowcast": data["meta"].get("nowcast", {}),
    }

def is_no_service(yhat: float | None, actual: float | None, trams: float | None) -> bool:
    if actual is not None and actual >= NO_SERVICE_THRESHOLD:
        return False
    return trams == 0 or (yhat is not None and yhat < NO_SERVICE_THRESHOLD) or (yhat is None and actual is not None)


@app.get("/internal/forecast")
def forecast(
    request: Request,
    from_date: date = Query(alias="from"),
    to_date: date = Query(alias="to"),
    route: list[int] | None = Query(default=None),
    stop_id: int | None = None,
    horizon: str = Query(default="day", pattern="^(day|month|year)$"),
    granularity: str = Query(default="hour", pattern="^(hour|day|month)$"),
    from_hour: int = Query(default=0, ge=0, le=23),
    to_hour: int = Query(default=23, ge=0, le=23),
    segment_direction: int | None = None,
    segment_start_seq: int | None = None,
    segment_end_seq: int | None = None,
) -> dict[str, Any]:
    for value in route or []:
        check_route(value)
    if from_date > to_date:
        raise HTTPException(status_code=400, detail="Параметр from не может быть позже to")
    if granularity == "hour" and horizon != "day":
        raise HTTPException(status_code=400, detail="Почасовой прогноз доступен только на горизонте день")
    if granularity == "hour" and from_date == to_date and from_hour > to_hour:
        raise HTTPException(status_code=400, detail="Начальный час не может быть позже конечного")
    segment_values = (segment_direction, segment_start_seq, segment_end_seq)
    has_segment = any(value is not None for value in segment_values)
    if has_segment and (any(value is None for value in segment_values) or stop_id is not None or not route or len(route) != 1):
        raise HTTPException(status_code=400, detail="Для участка нужны направление, две остановки и один маршрут")
    if has_segment and segment_start_seq >= segment_end_seq:
        raise HTTPException(status_code=400, detail="Начало участка должно быть раньше конечной остановки")

    data = dataset(request)
    source = data["hourly"] if horizon == "day" else data["daily"]
    frame = with_user_events(request, source.filter(pl.col("date").is_between(from_date, to_date, closed="both")))
    actuals = data["actuals"].filter(pl.col("date").is_between(from_date, to_date, closed="both"))
    if route:
        frame = frame.filter(pl.col("route").is_in(route))
        actuals = actuals.filter(pl.col("route").is_in(route))
    if granularity == "hour":
        frame = frame.filter(
            ((pl.col("date") > from_date) | (pl.col("hour") >= from_hour))
            & ((pl.col("date") < to_date) | (pl.col("hour") <= to_hour))
        )
        actuals = actuals.filter(
            ((pl.col("date") > from_date) | (pl.col("hour") >= from_hour))
            & ((pl.col("date") < to_date) | (pl.col("hour") <= to_hour))
        )

    if horizon == "year" or granularity in ("day", "month"):
        period = "1mo" if granularity == "month" else "1d"
        frame = frame.sort("date").group_by_dynamic(
            "date", every=period, group_by="route", closed="left"
        ).agg(pl.col("yhat").sum(), pl.col("q10").sum(), pl.col("q90").sum())
        actuals = actuals.sort("date").group_by_dynamic(
            "date", every=period, group_by="route", closed="left"
        ).agg(pl.col("boardings").sum())
    else:
        actuals = actuals.group_by("route", "date", "hour").agg(pl.col("boardings").sum())

    if stop_id is not None or has_segment:
        shares = data["shares"]
        if has_segment:
            stops = [
                int(feature["properties"]["stop_id"])
                for feature in data["geometry"].get(route[0], [])
                if feature["properties"].get("kind") == "stop"
                and int(feature["properties"]["direction"]) == segment_direction
                and segment_start_seq <= int(feature["properties"]["seq"]) < segment_end_seq
            ]
            if not stops:
                raise HTTPException(status_code=400, detail="Для выбранного участка не найдены остановки")
            shares = shares.filter(
                (pl.col("route") == route[0])
                & (pl.col("direction") == segment_direction)
                & (pl.col("stop_id").is_in(stops))
            )
        else:
            shares = shares.filter(pl.col("stop_id") == stop_id)
            if route:
                shares = shares.filter(pl.col("route").is_in(route))
        stop_share = shares.group_by("route", "direction", "stop_id").agg(pl.col("share").mean())
        stop_share = stop_share.group_by("route").agg(pl.col("share").sum())
        frame = frame.join(stop_share, on="route", how="inner").with_columns(
            (pl.col("yhat") * pl.col("share")).alias("yhat"),
            (pl.col("q10") * pl.col("share")).alias("q10"),
            (pl.col("q90") * pl.col("share")).alias("q90"),
        ).drop("share")
        actuals = actuals.join(stop_share, on="route", how="inner").with_columns(
            (pl.col("boardings") * pl.col("share")).alias("boardings")
        ).drop("share")

    keys = ["route", "date"] + (["hour"] if granularity == "hour" else [])
    combined = frame.join(actuals, on=keys, how="full", coalesce=True)
    if granularity == "hour":
        combined = combined.join(data["service"], on=keys, how="left")
    multiplier = factor_multiplier(request)
    points = []
    for row in combined.sort(keys).iter_rows(named=True):
        ts = row["date"].isoformat()
        if granularity == "hour":
            ts += f"T{int(row['hour']):02d}:00:00+03:00"
        yhat = row.get("yhat")
        q10 = row.get("q10")
        q90 = row.get("q90")
        points.append({
            "ts": ts, "route": int(row["route"]),
            **({"stop_id": stop_id} if stop_id is not None else {}),
            **({"segment_direction": segment_direction, "segment_start_seq": segment_start_seq,
                "segment_end_seq": segment_end_seq} if has_segment else {}),
            "yhat": None if yhat is None else max(0.0, float(yhat) * multiplier),
            "q10": None if q10 is None else max(0.0, float(q10) * multiplier),
            "q90": None if q90 is None else max(0.0, float(q90) * multiplier),
            "actual": row.get("boardings"),
            # Stop and segment values are shares of the route, so only the route schedule decides "no service".
            **({"no_service": row.get("trams_scheduled") == 0 if stop_id is not None or has_segment
                else is_no_service(yhat, row.get("boardings"), row.get("trams_scheduled"))}
               if granularity == "hour" else {}),
        })
    return {"meta": data["meta"], "points": points}


@app.get("/internal/recommendations")
def recommendations(request: Request, route: int, forecast_date: date = Query(alias="date")) -> list[dict[str, Any]]:
    check_route(route)
    data = dataset(request)
    multiplier = factor_multiplier(request) * route_day_event_factor(request, route, forecast_date)
    model_recs = data["recommendations"]
    if model_recs is not None:
        model_rows = model_recs.filter(
            (pl.col("route") == route) & (pl.col("date") == forecast_date)
        ).sort("hour")
        if not model_rows.is_empty():
            result = []
            for row in model_rows.iter_rows(named=True):
                plan = max(0, int(round(row["trams_plan"] or 0)))
                baseline_needed = max(0, int(round(row["trams_needed"] or 0)))
                needed = min(int(math.ceil(plan * 1.5)), int(math.ceil(baseline_needed * multiplier))) if plan else 0
                raw_load = row["peak_load_pct"]
                raw_dwell = row["dwell_extra_sec"]
                result.append({
                    "hour": int(row["hour"]),
                    "yhat": max(0.0, float(row["yhat"]) * multiplier),
                    "trams_now": plan,
                    "trams_needed": needed,
                    "load_pct": None if raw_load is None else max(0.0, min(100.0, float(raw_load) * multiplier)),
                    "dwell_extra_sec": None if raw_dwell is None else max(0, int(round(float(raw_dwell) * multiplier))),
                    "overload_risk_pct": row.get("overload_risk_pct"),
                    "no_service": plan == 0 or float(row["yhat"]) * multiplier < NO_SERVICE_THRESHOLD,
                })
            return result
    baseline = data["hourly"].filter((pl.col("route") == route) & (pl.col("date") == forecast_date))
    if baseline.is_empty():
        baseline = data["backtest"].filter((pl.col("route") == route) & (pl.col("date") == forecast_date))
    if baseline.is_empty():
        return []
    actual = data["actuals"].filter((pl.col("route") == route) & (pl.col("date") == forecast_date))
    trams_by_hour = {int(r["hour"]): int(r["trams"]) for r in actual.select("hour", "trams").iter_rows(named=True)} if "trams" in actual.columns else {}
    result = []
    for row in baseline.sort("hour").iter_rows(named=True):
        yhat = max(0.0, float(row["yhat"]) * multiplier)
        trams_now = trams_by_hour.get(int(row["hour"]))
        needed = max(1, int((yhat * 0.35 / (188 * 0.8)) + 0.999))
        effective_fleet = max(trams_now or 0, needed)
        result.append({"hour": int(row["hour"]), "yhat": yhat, "trams_now": trams_now,
                       "trams_needed": needed, "load_pct": min(100.0, yhat * 0.35 / (effective_fleet * 188) * 100),
                       "dwell_extra_sec": None, "overload_risk_pct": None,
                       "no_service": trams_now == 0 or yhat < NO_SERVICE_THRESHOLD})
    return result

@app.get("/internal/risk")
def overload_risk(request: Request, risk_date: date = Query(alias="date"), hour: int = Query(ge=0, le=23)) -> dict[str, Any]:
    """Per-route risk that the busiest segment exceeds the nominal tram capacity in the given hour."""
    data = dataset(request)
    capacity = data["meta"].get("capacity", {})
    result = {"date": risk_date.isoformat(), "hour": hour, "capacity_nominal": capacity.get("nominal", 188),
              "capacity_model": capacity.get("model"), "available": False, "routes": []}
    recs = data["recommendations"]
    if recs is None:
        return result
    rows = recs.filter((pl.col("date") == risk_date) & (pl.col("hour") == hour)).sort("route")
    result["available"] = not rows.is_empty()
    for row in rows.iter_rows(named=True):
        plan = max(0, int(round(row["trams_plan"] or 0)))
        # Dispatcher events scale demand; the model's overload probability is kept as published.
        factor = route_day_event_factor(request, int(row["route"]), risk_date)
        yhat = float(row["yhat"] or 0) * factor
        needed = max(0, int(round(row["trams_needed"] or 0)))
        load = row["peak_load_pct"]
        result["routes"].append({
            "route": int(row["route"]), "yhat": yhat,
            "trams_now": plan,
            "trams_needed": needed if factor == 1 else min(math.ceil(plan * 1.5), math.ceil(needed * factor)),
            "load_pct": None if load is None else min(100.0, float(load) * factor),
            "overload_risk_pct": row["overload_risk_pct"],
            "no_service": plan == 0 or yhat < NO_SERVICE_THRESHOLD,
        })
    return result


@app.get("/internal/export")
def export_forecast(
    request: Request,
    from_date: date = Query(alias="from"),
    to_date: date = Query(alias="to"),
    route: list[int] | None = Query(default=None),
    stop_id: int | None = None,
    horizon: str = Query(default="day", pattern="^(day|month|year)$"),
    granularity: str = Query(default="hour", pattern="^(hour|day|month)$"),
    format: str = Query(default="csv", pattern="^(csv|xlsx)$"),
    from_hour: int = Query(default=0, ge=0, le=23),
    to_hour: int = Query(default=23, ge=0, le=23),
    segment_direction: int | None = None,
    segment_start_seq: int | None = None,
    segment_end_seq: int | None = None,
):
    result = forecast(request, from_date, to_date, route, stop_id, horizon, granularity, from_hour, to_hour,
                      segment_direction, segment_start_seq, segment_end_seq)
    frame = pl.DataFrame(result["points"])
    if format == "csv":
        return Response(content=frame.write_csv(separator=";"), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": "attachment; filename=tram-forecast.csv"})
    output = BytesIO()
    import xlsxwriter
    with xlsxwriter.Workbook(output, {"in_memory": True}) as workbook:
        frame.write_excel(workbook=workbook, worksheet="Прогноз")
    return Response(content=output.getvalue(),
                     media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                     headers={"Content-Disposition": "attachment; filename=tram-forecast.xlsx"})
