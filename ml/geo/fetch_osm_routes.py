"""Геопривязка: трассы и остановки всех 10 маршрутов из OpenStreetMap (Overpass API).
Справочник организаторов покрывает только 1, 5, 7, 11, 12 — остальные маршруты без OSM не попадут на карту.
Источник: https://www.openstreetmap.org (ODbL), API: https://overpass-api.de
Выход: artifacts/geo/routes.geojson (линии по направлениям), artifacts/geo/stops.parquet (упорядоченные остановки).
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]
BBOX = "(55.55,37.35,55.95,37.85)"
OUT = Path("artifacts/geo")
RAW = Path("data/external/osm")
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)


def overpass(q: str, tries: int = 6) -> dict:
    data = urllib.parse.urlencode({"data": q}).encode()
    for i in range(tries):
        try:
            req = urllib.request.Request("https://overpass-api.de/api/interpreter", data=data,
                                        headers={"User-Agent": "msk-trans-hackathon/1.0"})
            return json.loads(urllib.request.urlopen(req, timeout=120).read())
        except Exception as e:  # 429/504 — сервер перегружен, ждём
            print(f"  retry {i + 1}: {e}")
            time.sleep(15 * (i + 1))
    raise RuntimeError("Overpass недоступен")


features, stops = [], []
for ref in ROUTES:
    cache = RAW / f"tram_{ref}.json"
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
    else:
        d = overpass(f'[out:json][timeout:90];rel["route"="tram"]["ref"="{ref}"]{BBOX};out geom;node(r);out;')
        cache.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    nodes = {e["id"]: e for e in d["elements"] if e["type"] == "node"}
    rels = [e for e in d["elements"] if e["type"] == "relation"]
    rels = sorted(rels, key=lambda r: -len(r.get("members", [])))[:2]     # два направления, самые полные варианты
    for direction, rel in enumerate(rels):
        lines = [[[p["lon"], p["lat"]] for p in m["geometry"]] for m in rel["members"]
                 if m["type"] == "way" and m.get("role", "") == "" and "geometry" in m]
        features.append({"type": "Feature", "geometry": {"type": "MultiLineString", "coordinates": lines},
                         "properties": {"route": ref, "direction": direction, "name": rel["tags"].get("name", ""),
                                        "osm_relation": rel["id"]}})
        seq = 0
        for m in rel["members"]:
            if m["type"] == "node" and m.get("role", "").startswith(("stop", "platform")):
                n = nodes.get(m["ref"])
                if not n or "lat" not in n:
                    continue
                name = n.get("tags", {}).get("name", "")
                if stops and stops[-1]["route"] == ref and stops[-1]["direction"] == direction and stops[-1]["name"] == name:
                    continue  # stop + platform одной остановки
                seq += 1
                stops.append(dict(route=ref, direction=direction, seq=seq, stop_id=m["ref"], name=name,
                                  lat=n["lat"], lon=n["lon"], role=m["role"]))
    print(ref, "relations:", len(rels), "stops:", sum(1 for s in stops if s["route"] == ref), flush=True)
    time.sleep(3)

(OUT / "routes.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False), encoding="utf-8")
pd.DataFrame(stops).to_parquet(OUT / "stops.parquet", index=False)
print("routes.geojson:", len(features), "features; stops.parquet:", len(stops), "rows")
