import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';

type Feature = { type: string; geometry: { type: string; coordinates: any }; properties?: Record<string, any> };
type Segment = { direction: number; startSeq: number; endSeq: number };
export type RouteRisk = { risk: number | null; noService: boolean };
type Props = { route: number; dateTime: string; features: Feature[]; routeFeatures: Feature[]; segment?: Segment | null; selectedStop?: number | null; onStop?: (route: number, id: number) => void; onRoute?: (route: number) => void;
  risk?: Map<number, RouteRisk> | null; capacity?: number };

const OSM_TILES = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';
const ROUTE_COLORS = ['#5b7ce8', '#4f75db', '#d77b42', '#9b61bd', '#c45579', '#aa8737', '#438eaa', '#ad74b8', '#d2634e', '#6d70b4'];
export const routeLineColor = (id: number) => ROUTE_COLORS[[1, 5, 7, 11, 12, 17, 25, 26, 28, 50].indexOf(id)] ?? '#5b7ce8';
export const RISK_COLORS = { none: '#63718d', low: '#8da8ff', mid: '#f2c14e', high: '#ef6262' };
export const riskColor = (value: RouteRisk | undefined) => !value || value.noService || value.risk == null ? RISK_COLORS.none
  : value.risk > 10 ? RISK_COLORS.high : value.risk >= 5 ? RISK_COLORS.mid : RISK_COLORS.low;
const escapeHtml = (value: unknown) => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');

export default function LeafletMap({ route, dateTime, features, routeFeatures, segment, selectedStop, onStop, onRoute, risk, capacity = 188 }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const map = useRef<L.Map | null>(null);
  const overlays = useRef<L.LayerGroup | null>(null);
  const [ready, setReady] = useState(false);
  const [tileError, setTileError] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(10);

  useEffect(() => {
    if (!host.current) return;
    const instance = L.map(host.current, { center: [55.7558, 37.6176], zoom: 10, zoomControl: true, attributionControl: true });
    instance.attributionControl.setPrefix(false);
    const tiles = L.tileLayer(OSM_TILES, { maxZoom: 19, attribution: 'Подложка, трассы и остановки © <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a>' });
    tiles.on('tileerror', () => setTileError(true));
    tiles.on('tileload', () => setTileError(false));
    tiles.addTo(instance);
    const layer = L.layerGroup().addTo(instance);
    instance.on('zoomend', () => setZoomLevel(instance.getZoom()));
    map.current = instance;
    overlays.current = layer;
    setReady(true);
    const resize = window.setTimeout(() => instance.invalidateSize(), 0);

    return () => {
      window.clearTimeout(resize);
      setReady(false);
      overlays.current = null;
      map.current = null;
      instance.remove();
    };
  }, []);

  useEffect(() => {
    const instance = map.current;
    if (!ready || !instance) return;
    if (route === 0) { instance.setView([55.7558, 37.6176], 10, { animate: false }); return; }
    const selected = routeFeatures.filter((feature) => Number(feature.properties?.route) === route);
    if (!selected.length) return;
    const bounds = L.geoJSON({ type: 'FeatureCollection', features: selected } as any).getBounds();
    if (bounds.isValid()) instance.fitBounds(bounds.pad(.14), { maxZoom: 13, animate: false });
  }, [ready, route, routeFeatures]);

  useEffect(() => {
    const layer = overlays.current;
    if (!ready || !layer) return;

    layer.clearLayers();
    const routeColor = ['#5b7ce8', '#8964c8', '#eea33a', '#df6581'][route % 4];
    const visiblePoints = route === 0 ? features : features.filter((feature) => Number(feature.properties?.route) === route);
    const loads = new Map(visiblePoints.map((feature) => [
      `${feature.properties?.route}:${feature.properties?.stop_id}:${feature.properties?.direction}`, feature.properties,
    ]));
    const mapFeatures = routeFeatures.length
      ? routeFeatures.map((feature) => {
          if (feature.geometry.type !== 'Point') return feature;
          const load = loads.get(`${feature.properties?.route}:${feature.properties?.stop_id}:${feature.properties?.direction}`);
          return load ? { ...feature, properties: { ...feature.properties, ...load } } : feature;
        })
      : visiblePoints;
    const collection = { type: 'FeatureCollection' as const, features: mapFeatures.filter((feature) => feature.geometry.type !== 'Point' || (route === 0 ? zoomLevel >= 12 : Number(feature.properties?.route) === route)) as any[] };

    const lineFeatures = collection.features.filter((feature) => feature.geometry.type === 'LineString' || feature.geometry.type === 'MultiLineString');
    if (risk) {
      L.geoJSON({ type: 'FeatureCollection', features: lineFeatures } as any, {
        interactive: false,
        style: (feature) => {
          const lineRoute = Number(feature?.properties?.route ?? route);
          const value = risk.get(lineRoute);
          const highlighted = value && !value.noService && value.risk != null && value.risk >= 5;
          return { color: highlighted ? riskColor(value) : '#111523', weight: highlighted ? 10 : route === lineRoute ? 9 : 6, opacity: highlighted ? route === 0 || route === lineRoute ? .9 : .25 : route === 0 || route === lineRoute ? .8 : .15 };
        },
      }).addTo(layer);
    }
    const drawn = L.geoJSON(collection as any, {
      onEachFeature: (feature, layer) => {
        if (feature.geometry.type === 'Point') return;
        const lineRoute = Number(feature.properties?.route ?? route);
        const value = risk?.get(lineRoute);
        const riskText = !risk ? '' : !value || value.risk == null ? ' · риск не рассчитан'
          : value.noService ? ' · нет движения' : ` · вероятность превышения вместимости ${capacity} чел.: ${value.risk.toFixed(1).replace('.', ',')}%`;
        layer.bindTooltip(`Маршрут ${lineRoute}${riskText}`, { sticky: true });
        if (onRoute) layer.on('click', () => onRoute(lineRoute));
      },
      style: (feature) => {
        const properties = feature?.properties ?? {};
        const isLine = feature?.geometry?.type === 'LineString' || feature?.geometry?.type === 'MultiLineString';
        // Stop markers keep the stroke set in pointToLayer; style() only paints route lines.
        if (feature?.geometry?.type === 'Point') return {};
        const lineRoute = Number(properties.route ?? route);
        const color = isLine ? routeLineColor(lineRoute) : properties.color || routeColor;
        return { color, weight: route === 0 || route === lineRoute ? route === 0 ? 4 : 6 : 2.5, opacity: route === 0 || route === lineRoute ? .98 : .23, fillColor: routeColor, fillOpacity: 0.12 };
      },
      pointToLayer: (feature, latLng) => {
        const properties = feature.properties ?? {};
        const isStop = properties.stop_id != null;
        const load = Number(properties.yhat ?? 0);
        const period = properties.period ?? 'hour';
        const reference = period === 'hour' ? 60 : period === 'day' ? 900 : 18000;
        const ratio = load / reference;
        const fillColor = !isStop ? routeColor : routeLineColor(Number(properties.route ?? route));
        const onSegment = !segment || (Number(properties.direction) === segment.direction
          && Number(properties.seq) >= segment.startSeq && Number(properties.seq) <= segment.endSeq);
        const chosen = isStop && selectedStop === Number(properties.stop_id) && (route === 0 || route === Number(properties.route));
        const baseRadius = chosen ? 10 : isStop ? route === 0 ? 3.5 : Math.min(7, 4.5 + Math.sqrt(Math.max(0, ratio)) * 2) : 5;
        const marker = L.circleMarker(latLng, {
          radius: baseRadius,
          color: chosen ? '#fff' : '#e8edff',
          weight: chosen ? 3 : 1.5,
          fillColor: chosen ? '#e9bd62' : onSegment ? fillColor : '#73809a',
          fillOpacity: onSegment ? 1 : .45,
        });
        if (isStop) {
          marker.bindTooltip(`<b>${escapeHtml(properties.name ?? 'Остановка')}</b><br/>Маршрут ${Number(properties.route)} · ${properties.yhat == null ? 'нет оценки' : `≈${Math.round(load)} посадок за ${period === 'hour' ? 'час' : period === 'day' ? 'день' : 'месяц'}`}<br/><small>Оценка по данным маршрута</small>`, { direction: 'top', offset: L.point(0, -8) });
          if (onStop) marker.on('click', () => onStop(Number(properties.route), Number(properties.stop_id)));
          marker.on('mouseover', () => { marker.setStyle({ weight: 3, color: '#fff' }); marker.bringToFront(); });
          marker.on('mouseout', () => { if (!chosen) marker.setStyle({ weight: 1.5, color: '#e8edff' }); });
        } else {
          marker.bindTooltip(`Маршрут ${properties.route ?? route}`, { direction: 'top' });
        }
        return marker;
      },
    });
    drawn.addTo(layer);
    if (segment) {
      const selectedStops = routeFeatures
        .filter((feature) => feature.geometry.type === 'Point'
          && Number(feature.properties?.direction) === segment.direction
          && Number(feature.properties?.seq) >= segment.startSeq
          && Number(feature.properties?.seq) <= segment.endSeq)
        .sort((a, b) => Number(a.properties?.seq) - Number(b.properties?.seq));
      const line = selectedStops.map((feature) => {
        const [lon, lat] = feature.geometry.coordinates;
        return L.latLng(Number(lat), Number(lon));
      });
      if (line.length >= 2) {
        L.polyline(line, { color: '#e88735', weight: 7, opacity: .85, interactive: false }).addTo(layer);
        map.current?.fitBounds(L.latLngBounds(line).pad(.2), { maxZoom: 14 });
      }
    }
  }, [ready, route, dateTime, features, routeFeatures, segment?.direction, segment?.startSeq, segment?.endSeq, selectedStop, onStop, onRoute, risk, capacity, zoomLevel]);

  return <div className="map-wrap"><div className="map-canvas" ref={host} />{tileError && <div className="map-message map-message-error"><b>Карта OpenStreetMap временно недоступна</b><span>Проверьте подключение к интернету и попробуйте обновить страницу</span></div>}</div>;
}
