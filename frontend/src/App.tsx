import { useEffect, useMemo, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import ReactECharts from 'echarts-for-react';
import LeafletMap, { routeLineColor } from './LeafletMap';
import WhyPanel from './WhyPanel';
import HistoryPanel from './HistoryPanel';
import EventForm from './EventForm';
import ActionPanel from './ActionPanel';
import type { Recommendation, RiskSnapshot } from './ActionPanel';
import type { RouteRisk } from './LeafletMap';
import { apiFetch, authUser, download, get, logout, streamEvents } from './api';

type Route = { route: number; name: string; has_geo: boolean; color: string };
type Point = { ts: string; route: number; yhat: number | null; q10: number | null; q90: number | null; actual?: number | null; no_service?: boolean };
type Factor = { id: string; label: string; default: number; min: number; max: number; step: number; effect_note?: string };
type Feature = { type: string; geometry: { type: string; coordinates: any }; properties?: Record<string, any> };
type EventRow = { ts: string; route: number; hour: number; boardings_so_far: number };
type Segment = { direction: number; startSeq: number; endSeq: number };
type RegistryEvent = { id: string; title: string; start: string; end: string; routes: number[]; category: string; effect: string; source_label: string; source_url: string; user?: boolean };
type NowcastResult = { route: number; date: string; now_hour: number; correction_factor: number; points: { hour: number; baseline: number; yhat_adj: number; q10: number; q90: number }[] };
const fmt = new Intl.NumberFormat('ru-RU');
const defaultDate = '2025-11-03';
const demoDate = '2025-10-31';
const minDate = '2025-01-01';
const maxDate = '2026-10-31';
function riskLabel(value: number) { return value > 10 ? 'Высокий' : value >= 5 ? 'Требует внимания' : 'Низкий'; }
function RiskReadout({ value, capacity, network }: { value: number; capacity: number; network: boolean }) {
  const position = Math.max(2, Math.min(98, value / 20 * 100));
  return <div className="tp-risk-readout">
    <div className="tp-risk-heading"><span>{network ? 'Наибольший риск в сети' : 'Риск на маршруте'}</span><strong className={value > 10 ? 'high' : value >= 5 ? 'medium' : 'low'}>{value.toFixed(1).replace('.', ',')}% <small>{riskLabel(value)}</small></strong></div>
    <div className="tp-risk-scale" role="img" aria-label={`Риск ${value.toFixed(1).replace('.', ',')} процента. ${riskLabel(value)}`}><i style={{ left: `${position}%` }}/></div>
    <div className="tp-risk-ticks"><span>0%</span><span>5%</span><span>10%</span><span>20%+</span></div>
    <p>Вероятность, что на самом загруженном участке пассажиров окажется больше вместимости вагона ({capacity} чел.).</p>
  </div>;
}
function factorParams(factors: Factor[], values: Record<string, number>) {
  const p = new URLSearchParams(); factors.forEach((f) => p.set(`factor.${f.id}`, String(values[f.id] ?? f.default))); return p.toString();
}
function dateRange(date: string, horizon: string) {
  if (horizon === 'month') return { from: `${date.slice(0, 7)}-01`, to: new Date(Date.UTC(+date.slice(0, 4), +date.slice(5, 7), 0)).toISOString().slice(0, 10), granularity: 'day' };
  if (horizon === 'year') return { from: `${date.slice(0, 4)}-01-01`, to: date.startsWith('2026') ? '2026-10-31' : `${date.slice(0, 4)}-12-31`, granularity: 'month' };
  return { from: date, to: date, granularity: 'hour' };
}
function combineNetworkPoints(rows: Point[]): Point[] {
  const groups = new Map<string, Point[]>();
  rows.forEach((point) => groups.set(point.ts, [...(groups.get(point.ts) ?? []), point]));
  const sum = (items: Point[], key: 'yhat' | 'q10' | 'q90' | 'actual') => {
    const present = items.map((item) => item[key]).filter((value): value is number => value != null);
    return present.length ? present.reduce((total, value) => total + value, 0) : null;
  };
  return [...groups.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([ts, items]) => ({
    ts, route: 0, yhat: sum(items, 'yhat'), q10: sum(items, 'q10'), q90: sum(items, 'q90'), actual: sum(items, 'actual'),
    no_service: items.every((item) => item.no_service),
  }));
}
function combineNetworkRecommendations(groups: any[][]): any[] {
  const byHour = new Map<number, any[]>();
  groups.flat().forEach((row) => byHour.set(Number(row.hour), [...(byHour.get(Number(row.hour)) ?? []), row]));
  return [...byHour.entries()].sort(([a], [b]) => a - b).map(([hour, rows]) => {
    const tramsNow = rows.reduce((sum, row) => sum + Number(row.trams_now ?? 0), 0);
    const tramsNeeded = rows.reduce((sum, row) => sum + Number(row.trams_needed ?? 0), 0);
    const weightedLoad = rows.reduce((sum, row) => sum + Number(row.load_pct ?? 0) * Number(row.trams_now ?? 0), 0);
    return { hour, trams_now: tramsNow, trams_needed: tramsNeeded,
      load_pct: tramsNow ? weightedLoad / tramsNow : 0, dwell_extra_sec: null };
  });
}
function pointLabel(ts: string) { const d = new Date(ts); return ts.includes('T') ? d.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }) : d.toLocaleDateString('ru-RU', { day: 'numeric', month: 'short' }); }

function sliderFill(value: number, min: number, max: number): CSSProperties {
  return { '--fill': `${max > min ? Math.max(0, Math.min(100, (value - min) / (max - min) * 100)) : 0}%` } as CSSProperties;
}

export default function App() {
  const [routes, setRoutes] = useState<Route[]>([]), [factors, setFactors] = useState<Factor[]>([]);
  const [baselinePoints, setBaselinePoints] = useState<Point[]>([]);
  const [screen, setScreen] = useState<'map' | 'planning'>('map');
  const [detail, setDetail] = useState<'overview' | 'why' | 'scenario' | 'events'>('overview');
  const [routeSearch, setRouteSearch] = useState('');
  const [aboutOpen, setAboutOpen] = useState(false);
  const [activityCollapsed, setActivityCollapsed] = useState(false), [factorsCollapsed, setFactorsCollapsed] = useState(false);
  const [leftRailCollapsed, setLeftRailCollapsed] = useState(false), [rightRailCollapsed, setRightRailCollapsed] = useState(false);
  const [values, setValues] = useState<Record<string, number>>({}), [route, setRoute] = useState(0), [date, setDate] = useState(defaultDate);
  const [timeIndex, setTimeIndex] = useState(8), [horizon, setHorizon] = useState('day'), [points, setPoints] = useState<Point[]>([]), [selectedStop, setSelectedStop] = useState<number | null>(null);
  const [intervalFrom, setIntervalFrom] = useState(defaultDate), [intervalTo, setIntervalTo] = useState(defaultDate);
  const [fromHour, setFromHour] = useState(0), [toHour, setToHour] = useState(23);
  const [mapPoints, setMapPoints] = useState<Feature[]>([]), [geometry, setGeometry] = useState<Feature[]>([]), [recs, setRecs] = useState<any[]>([]);
  const [segmentDirection, setSegmentDirection] = useState<number | null>(null);
  const [segmentStart, setSegmentStart] = useState<number | null>(null), [segmentEnd, setSegmentEnd] = useState<number | null>(null);
  const [events, setEvents] = useState<EventRow[]>([]), [nowcast, setNowcast] = useState<NowcastResult | null>(null);
  const [demoMode, setDemoMode] = useState(false);
  const [demoHours, setDemoHours] = useState<Record<number, Record<number, number>>>({});
  const [replayState, setReplayState] = useState<'idle' | 'loading' | 'playing' | 'paused' | 'complete'>('idle');
  const [replayHour, setReplayHour] = useState(-1);
  const [replaySpeed, setReplaySpeed] = useState(1);
  const replayRows = useRef<EventRow[]>([]);
  const beforeReplay = useRef<{ route: number; date: string; horizon: string; time: number } | null>(null);
  const replaySource = useRef<(() => void) | null>(null);
  const [registry, setRegistry] = useState<RegistryEvent[]>([]), [registryRouteOnly, setRegistryRouteOnly] = useState(true);
  const requestSeq = useRef(0);
  const routeMenuRef = useRef<HTMLDivElement>(null);
  const [routeMenuOpen, setRouteMenuOpen] = useState(false);
  useEffect(() => {
    if (!routeMenuOpen) return;
    const close = (event: PointerEvent) => { if (!routeMenuRef.current?.contains(event.target as Node)) setRouteMenuOpen(false); };
    const escape = (event: KeyboardEvent) => { if (event.key === 'Escape') setRouteMenuOpen(false); };
    document.addEventListener('pointerdown', close);
    document.addEventListener('keydown', escape);
    return () => { document.removeEventListener('pointerdown', close); document.removeEventListener('keydown', escape); };
  }, [routeMenuOpen]);
  const [dataVersion, setDataVersion] = useState(0);
  const [risk, setRisk] = useState<RiskSnapshot | null>(null), [showRisk, setShowRisk] = useState(true);
  const [model, setModel] = useState<{ model_version?: string; lb_wape_score?: number | null } | null>(null);
  const [health, setHealth] = useState('Подключаемся'), [error, setError] = useState(''), [loading, setLoading] = useState(false);
  const replaying = replayState === 'playing';
  useEffect(() => {
    let ready = false;
    const initialize = () => Promise.all([get<Route[]>("/routes"), get<Factor[]>("/factors"), get<any>("/health"), get<RegistryEvent[]>("/events").catch(() => [])]).then(([r, f, h, eventRegistry]) => {
      setRoutes(r); setFactors(f); setRegistry(eventRegistry); setValues(Object.fromEntries(f.map((x) => [x.id, x.default])));
      setHealth(h.status === "ok" ? "Данные обновлены" : "Ожидаем данные"); setModel(h);
      if (r.length) setRoute(0);
      ready = r.length > 0;
    }).catch((e) => { setError(e.message); setHealth("Нет связи"); });
    void initialize();
    const retry = window.setInterval(() => { if (!ready) void initialize(); }, 10000);
    return () => window.clearInterval(retry);
  }, []);
  const range = useMemo(() => ({ from: intervalFrom, to: intervalTo, granularity: horizon === 'day' ? 'hour' : horizon === 'month' ? 'day' : 'month' }), [intervalFrom, intervalTo, horizon]);
  const intervalError = !intervalFrom || !intervalTo || intervalFrom < minDate || intervalTo > maxDate || intervalFrom > intervalTo
    ? 'Укажите даты в порядке от начала к концу (01.01.2025–31.10.2026).'
    : horizon === 'day' && intervalFrom === intervalTo && fromHour > toHour
      ? 'Начальный час должен быть не позже конечного.' : '';
  useEffect(() => {
    if (intervalError) {
      requestSeq.current += 1;
      setPoints([]);
      setRecs([]);
    }
  }, [intervalError]);
  const monthDays = new Date(+date.slice(0, 4), +date.slice(5, 7), 0).getDate();
  const timeMax = horizon === 'day' ? 23 : horizon === 'month' ? monthDays : date.startsWith('2026') ? 10 : 12;
  const boundedTime = Math.min(timeIndex, timeMax);
  const snapshotPeriod = horizon === 'day' ? 'hour' : horizon === 'month' ? 'day' : 'month';
  const snapshotDate = horizon === 'day' ? date
    : horizon === 'month' ? `${date.slice(0, 7)}-${String(boundedTime).padStart(2, '0')}`
    : `${date.slice(0, 4)}-${String(boundedTime).padStart(2, '0')}-01`;
  const timeLabel = horizon === 'day' ? `${String(boundedTime).padStart(2, '0')}:00`
    : horizon === 'month' ? new Date(`${snapshotDate}T12:00:00`).toLocaleDateString('ru-RU', { day: 'numeric', month: 'long' })
    : new Date(`${snapshotDate}T12:00:00`).toLocaleDateString('ru-RU', { month: 'long', year: 'numeric' });
  const routeStops = geometry.filter((feature) => feature.geometry.type === 'Point' && Number(feature.properties?.route) === route);
  const directions = [...new Set(routeStops.map((feature) => Number(feature.properties?.direction)))].sort((a, b) => a - b);
  const directionStops = routeStops.filter((feature) => Number(feature.properties?.direction) === segmentDirection)
    .sort((a, b) => Number(a.properties?.seq) - Number(b.properties?.seq));
  const segment: Segment | null = segmentDirection != null && segmentStart != null && segmentEnd != null && segmentStart < segmentEnd
    ? { direction: segmentDirection, startSeq: segmentStart, endSeq: segmentEnd } : null;
  const segmentParams = segment ? `&segment_direction=${segment.direction}&segment_start_seq=${segment.startSeq}&segment_end_seq=${segment.endSeq}` : '';
  const segmentError = segmentDirection != null && !segment ? 'Конечная остановка должна идти после начальной.' : '';
  const filterError = intervalError || segmentError;
  useEffect(() => {
    if (segmentError) {
      requestSeq.current += 1;
      setPoints([]);
      setRecs([]);
    }
  }, [segmentError]);
  const params = useMemo(() => `${route === 0 ? '' : `route=${route}&`}${selectedStop ? `&stop_id=${selectedStop}` : ""}${segmentParams}&from=${range.from}&to=${range.to}&horizon=${horizon}&granularity=${range.granularity}${horizon === 'day' ? `&from_hour=${fromHour}&to_hour=${toHour}` : ''}&${factorParams(factors, values)}`, [route, selectedStop, segmentParams, range, horizon, factors, values, fromHour, toHour]);
  const refresh = async () => {
    if (filterError) return;
    const seq = ++requestSeq.current;
    setLoading(true); setError('');
    try {
      const [forecast, map, geo, recommendation] = await Promise.all([
        get<{points: Point[]}>(`/forecast?${params}`),
        get<{features: Feature[]}>(`/map/snapshot?datetime=${encodeURIComponent(`${snapshotDate}T${String(horizon === 'day' ? boundedTime : 12).padStart(2, '0')}:00:00+03:00`)}&period=${snapshotPeriod}&${factorParams(factors, values)}`),
        route === 0
          ? Promise.all(routes.filter((item) => item.has_geo).map((item) => get<{features: Feature[]}>(`/routes/${item.route}/geometry`).catch(() => ({features: [] as Feature[]}))))
              .then((groups) => ({features: groups.flatMap((group) => group.features)}))
          : routes.find((item) => item.route === route)?.has_geo
            ? get<{features: Feature[]}>(`/routes/${route}/geometry`)
            : Promise.resolve({features: [] as Feature[]}),
        route === 0
          ? Promise.all(routes.map((item) => get<any[]>(`/recommendations?route=${item.route}&date=${snapshotDate}&${factorParams(factors, values)}`).catch(() => [])))
              .then(combineNetworkRecommendations)
          : get<any[]>(`/recommendations?route=${route}&date=${snapshotDate}&${factorParams(factors, values)}`).catch(() => []),
      ]);
      if (seq !== requestSeq.current) return;
      setPoints(route === 0 ? combineNetworkPoints(forecast.points ?? []) : forecast.points ?? []); setMapPoints(map.features ?? []); setGeometry(geo.features ?? []); setRecs(segment ? [] : recommendation);
    } catch (e: any) { if (seq === requestSeq.current) setError(e.message || 'Не удалось получить прогноз'); }
    finally { if (seq === requestSeq.current) setLoading(false); }
  };
  useEffect(() => { if (!routes.length || filterError) return; void refresh(); const id = window.setInterval(() => void refresh(), 30000); return () => window.clearInterval(id); }, [routes.length, params, snapshotDate, snapshotPeriod, boundedTime, filterError]);
  const stopReplay = (clearMode = false, restore = false) => {
    replaySource.current?.();
    replaySource.current = null;
    if (clearMode) {
      setReplayState('idle'); setDemoMode(false); setReplayHour(-1);
      setDemoHours({}); setEvents([]);
      if (restore && beforeReplay.current) {
        const previous = beforeReplay.current;
        setRoute(previous.route); setDate(previous.date); setHorizon(previous.horizon);
        setTimeIndex(previous.time);
        const previousRange = dateRange(previous.date, previous.horizon);
        setIntervalFrom(previousRange.from); setIntervalTo(previousRange.to);
      }
      beforeReplay.current = null;
    } else setReplayState('paused');
  };
  useEffect(() => () => replaySource.current?.(), []);
  useEffect(() => {
    if (demoMode && (date !== demoDate || horizon !== 'day' || selectedStop != null || segmentDirection != null)) stopReplay(true);
  }, [demoMode, date, horizon, selectedStop, segmentDirection]);
  const replay = () => {
    if (replayState === 'playing') { setReplayState('paused'); return; }
    if (replayState === 'paused') { setReplayState('playing'); return; }
    if (replayState === 'loading') return;
    if (replayState === 'complete') { setReplayHour(-1); setReplayState('playing'); return; }
    beforeReplay.current = { route, date, horizon, time: boundedTime };
    replayRows.current = [];
    setEvents([]); setDemoHours({}); setDemoMode(true); setReplayState('loading'); setReplayHour(-1); setNowcast(null);
    setDate(demoDate); setHorizon('day'); if (route === 5) setRoute(17); setIntervalFrom(demoDate); setIntervalTo(demoDate);
    setFromHour(0); setToHour(23); setTimeIndex(0); setSelectedStop(null);
    setSegmentDirection(null); setSegmentStart(null); setSegmentEnd(null);
    replaySource.current = streamEvents(`/stream/replay?date=${demoDate}&speed=3600`, (event, data) => {
      if (event === 'validation') replayRows.current.push(JSON.parse(data) as EventRow);
    }, (streamError) => {
      replaySource.current = null;
      if (streamError || !replayRows.current.length) {
        setError('Не удалось загрузить исторический день. Попробуйте снова.');
        stopReplay(true);
      } else setReplayState('playing');
    });
  };
  useEffect(() => {
    if (replayState !== 'playing') return;
    const timer = window.setInterval(() => setReplayHour((hour) => Math.min(23, hour + 1)), replaySpeed === 1 ? 800 : 400);
    return () => window.clearInterval(timer);
  }, [replayState, replaySpeed]);
  useEffect(() => {
    if (replayHour < 0) return;
    const cumulative: Record<number, number> = {};
    const hourly: Record<number, Record<number, number>> = {};
    for (const row of replayRows.current) {
      const amount = Math.max(0, row.boardings_so_far - (cumulative[row.route] ?? 0));
      cumulative[row.route] = row.boardings_so_far;
      if (row.hour > replayHour) continue;
      hourly[row.route] ??= {};
      hourly[row.route][row.hour] = amount;
    }
    setDemoHours(hourly);
    setEvents(replayRows.current.filter((row) => row.hour === replayHour).slice(0, 6));
    setTimeIndex(replayHour);
    if (replayHour === 23) setReplayState('complete');
  }, [replayHour]);
  useEffect(() => { setNowcast(null); }, [route, date, horizon, selectedStop]);
  const runNowcast = async () => {
    if (route === 0) return;
    try { setNowcast(await get<NowcastResult>(`/nowcast?route=${route}&date=${date}&now_hour=${boundedTime}`)); }
    catch (e:any) { setError(e.message); }
  };
  const exportForecast = (format: 'csv' | 'xlsx') => {
    if (!filterError) download(`/export?${params}&format=${format}`, `tram-forecast.${format}`).catch((e) => setError(e.message));
  };
  const demoActive = demoMode && date === demoDate && horizon === 'day' && selectedStop == null && !segment;
  const demoCount = (hour: number) => route === 0
    ? Object.values(demoHours).reduce((sum, hourly) => sum + (hourly[hour] ?? 0), 0)
    : demoHours[route]?.[hour] ?? 0;
  const viewPoints = demoActive ? points.map((point) => point.ts.startsWith(demoDate)
    ? { ...point, actual: route === 0 ? (Object.values(demoHours).some((hourly) => hourly[Number(point.ts.slice(11, 13))] != null) ? demoCount(Number(point.ts.slice(11, 13))) : null) : demoHours[route]?.[Number(point.ts.slice(11, 13))] ?? null } : point) : points;
  const observedHour = demoCount(boundedTime);
  const viewMapPoints = demoActive ? mapPoints.map((feature) => {
    const featureRoute = Number(feature.properties?.route);
    if (route !== 0 && featureRoute !== route) return feature;
    const routeTotal = mapPoints.filter((item) => Number(item.properties?.route) === featureRoute)
      .reduce((sum, item) => sum + Number(item.properties?.yhat ?? 0), 0);
    return { ...feature, properties: { ...feature.properties, yhat: routeTotal > 0
      ? Number(feature.properties?.yhat ?? 0) * (demoHours[featureRoute]?.[boundedTime] ?? 0) / routeTotal : 0, source: 'demo' } };
  }) : mapPoints;
  const peak = recs.reduce((a, b) => !a || Number(b.load_pct ?? -1) > Number(a.load_pct ?? -1) ? b : a, null as any);
  const visibleRegistry = registry.filter((event) => !registryRouteOnly || route === 0 || !event.routes.length || event.routes.includes(route));
  const currentNowcast = route !== 0 && horizon === 'day' && !selectedStop && nowcast?.route === route && nowcast.date === date ? nowcast : null;
  const adjustedByHour = new Map(currentNowcast?.points.map((point) => [point.hour, point]) ?? []);
  const canNowcast = route !== 0 && horizon === 'day' && !selectedStop && boundedTime < 23
    && viewPoints.some((point) => point.ts.startsWith(date) && point.yhat != null && point.actual != null);
  const selectedPoint = viewPoints.find((point) => horizon === 'day'
    ? point.ts.slice(0, 13) === `${date}T${String(boundedTime).padStart(2, '0')}`
    : point.ts.slice(0, horizon === 'month' ? 10 : 7) === snapshotDate.slice(0, horizon === 'month' ? 10 : 7));
  const selectedHour = selectedPoint?.ts.includes('T') ? Number(selectedPoint.ts.slice(11, 13)) : -1;
  const adjustedPoint = adjustedByHour.get(selectedHour);
  const selectedValue = adjustedPoint?.yhat_adj ?? selectedPoint?.yhat ?? selectedPoint?.actual;
  const sparkSource = viewPoints.filter((point) => point.yhat != null || point.actual != null);
  const sparkIndex = sparkSource.findIndex((point) => point.ts === selectedPoint?.ts);
  const sparkStart = Math.max(0, Math.min(sparkIndex < 0 ? sparkSource.length - 9 : sparkIndex - 4, sparkSource.length - 9));
  const sparkPoints = sparkSource.slice(sparkStart, sparkStart + 9);
  const sparkMax = Math.max(1, ...sparkPoints.map((point) => point.yhat ?? point.actual ?? 0));
  const intervalForecast = viewPoints.reduce((sum, point) => sum + (point.yhat ?? 0), 0);
  const intervalActual = viewPoints.reduce((sum, point) => sum + (point.actual ?? 0), 0);
  const hasForecast = viewPoints.some((point) => point.yhat != null);
  const hasActual = viewPoints.some((point) => point.actual != null);
  const hasBounds = viewPoints.some((point) => point.q10 != null && point.q90 != null);
  const selectedBounds = adjustedPoint
    ? `${fmt.format(Math.round(adjustedPoint.q10))}–${fmt.format(Math.round(adjustedPoint.q90))}`
    : selectedPoint?.q10 != null && selectedPoint?.q90 != null
      ? `${fmt.format(Math.round(selectedPoint.q10))}–${fmt.format(Math.round(selectedPoint.q90))}` : null;
  const selected = routes.find((r) => r.route === route);
  const riskHour = horizon === 'day' ? boundedTime : 8;
  useEffect(() => {
    let active = true;
    get<RiskSnapshot>(`/risk?date=${snapshotDate}&hour=${riskHour}`)
      .then((result) => { if (active) setRisk(result); })
      .catch(() => { if (active) setRisk(null); });
    return () => { active = false; };
  }, [snapshotDate, riskHour, dataVersion]);
  const riskMap = useMemo(() => showRisk && risk?.available
    ? new Map<number, RouteRisk>(risk.routes.map((row) => [row.route, { risk: row.overload_risk_pct, noService: row.no_service }])) : null, [risk, showRisk]);
  const reloadEvents = () => {
    get<RegistryEvent[]>('/events').then(setRegistry).catch(() => {});
    setDataVersion((value) => value + 1);
    void refresh();
  };
  const deleteEvent = (id: string) => apiFetch(`/events/${id}`, { method: 'DELETE' }).then(reloadEvents).catch((e) => setError(e.message));
  const selectRoute = (next: number) => { setRoute(next); setSelectedStop(null); setSegmentDirection(null); setSegmentStart(null); setSegmentEnd(null); };
  const selectHour = (next: number) => {
    if (horizon !== 'day') { setDate(snapshotDate); setHorizon('day'); setIntervalFrom(snapshotDate); setIntervalTo(snapshotDate); setFromHour(0); setToHour(23); }
    setTimeIndex(next);
  };
  const selectedStopFeature = selectedStop == null ? undefined : routeStops.find((feature) => Number(feature.properties?.stop_id) === selectedStop);
  const contextDirection = segment ? `направление ${segment.direction}, участок` : selectedStopFeature ? `ост. «${selectedStopFeature.properties?.name}»` : 'оба направления';
  const contextDate = horizon === 'year'
    ? new Date(`${snapshotDate}T12:00:00`).toLocaleDateString('ru-RU', { month: 'long', year: 'numeric' })
    : new Date(`${snapshotDate}T12:00:00`).toLocaleDateString('ru-RU', { weekday: 'short', day: 'numeric', month: 'long', year: 'numeric' });
  const contextTime = horizon === 'day' ? `${String(boundedTime).padStart(2, '0')}:00–${String(boundedTime + 1).padStart(2, '0')}:00` : horizon === 'month' ? 'сутки' : 'месяц';
  const isDark = true;
  const noServiceAreas: object[][] = [];
  viewPoints.forEach((point, index) => {
    if (!point.no_service) return;
    const last = noServiceAreas[noServiceAreas.length - 1] as { xAxis: number }[] | undefined;
    if (last && last[1].xAxis === index - 1) last[1].xAxis = index;
    else noServiceAreas.push([{ xAxis: index, name: 'нет движения' } as object, { xAxis: index }]);
  });
  const chart = { color: isDark ? ['#7592ff', '#d8b9ff'] : ['#4d6fe2', '#8964c8'], textStyle: { fontSize: 13.2 },
    tooltip: { trigger: 'axis', confine: true, extraCssText: 'max-width:calc(100% - 16px);white-space:normal;overflow-wrap:anywhere;', position: (point: number[], _params: unknown, _dom: HTMLElement, _rect: unknown, size: { contentSize: number[]; viewSize: number[] }) => {
      const margin = 8;
      const width = Math.min(size.contentSize[0], size.viewSize[0] - margin * 2);
      const height = Math.min(size.contentSize[1], size.viewSize[1] - margin * 2);
      const x = Math.max(margin, Math.min(point[0] + 12, size.viewSize[0] - width - margin));
      const y = Math.max(margin, Math.min(point[1] - height - 12, size.viewSize[1] - height - margin));
      return [x, y];
    }, formatter: (entries: any[]) => { const item = viewPoints[entries[0]?.dataIndex]; if (!item) return ''; const label = horizon === 'day' && range.from !== range.to ? `${item.ts.slice(0, 10)} ${pointLabel(item.ts)}` : horizon === 'year' ? item.ts.slice(0, 7) : pointLabel(item.ts); if (item.no_service) return `<b>${label}</b><br/>Нет движения`; return `<b>${label}</b><br/>${item.yhat == null ? '' : `Прогноз: ${fmt.format(Math.round(item.yhat))}<br/>`}${item.q10 == null || item.q90 == null ? '' : `Диапазон q10–q90: ${fmt.format(Math.round(item.q10))}–${fmt.format(Math.round(item.q90))}<br/>`}${item.actual == null ? '' : `Факт: ${fmt.format(Math.round(item.actual))}<br/>`}${horizon === 'day' && item.ts.startsWith(date) && adjustedByHour.has(Number(item.ts.slice(11, 13))) ? `Уточнённый: ${fmt.format(Math.round(adjustedByHour.get(Number(item.ts.slice(11, 13)))!.yhat_adj))}` : ''}`; }, backgroundColor: isDark ? '#292342' : '#fff',
      borderColor: isDark ? '#46436b' : '#e5e9f4',
      textStyle: { color: isDark ? '#f2f1ff' : '#272743', fontSize: 12 } }, grid: { left: 45, right: 18, top: 22, bottom: 25 }, xAxis: { type: 'category', data: viewPoints.map((p) => horizon === 'year' ? new Date(`${p.ts.slice(0, 7)}-01T12:00:00`).toLocaleDateString('ru-RU', { month: 'short', year: range.from.slice(0, 4) === range.to.slice(0, 4) ? undefined : '2-digit' }) : horizon === 'day' && range.from !== range.to ? `${p.ts.slice(5, 10)} ${pointLabel(p.ts)}` : pointLabel(p.ts)), axisLabel: { color: isDark ? '#b8b5d3' : '#777991', fontSize: 13.2, hideOverlap: true } }, yAxis: { type: 'value', splitLine: { lineStyle: { color: isDark ? '#393554' : '#ebedf5' } }, axisLabel: { color: isDark ? '#b8b5d3' : '#777991', fontSize: 13.2 } }, series: [
    { name: 'Нижняя граница', type: 'line', stack: 'prediction-band', symbol: 'none', silent: true, tooltip: { show: false }, data: viewPoints.map((p) => p.no_service || p.q10 == null || p.q90 == null ? null : Math.round(p.q10)), lineStyle: { opacity: 0 }, areaStyle: { opacity: 0 } },
    { name: 'Диапазон q10–q90', type: 'line', stack: 'prediction-band', symbol: 'none', silent: true, tooltip: { show: false }, data: viewPoints.map((p) => p.no_service || p.q10 == null || p.q90 == null ? null : Math.max(0, Math.round(p.q90 - p.q10))), lineStyle: { opacity: 0 }, areaStyle: { color: isDark ? 'rgba(117,146,255,.24)' : 'rgba(77,111,226,.16)' } },
    { name: 'Прогноз', type: 'line', smooth: .35, symbol: 'none', data: viewPoints.map((p) => p.no_service || p.yhat == null ? null : Math.round(p.yhat)), markArea: { silent: true, itemStyle: { color: isDark ? 'rgba(160,160,190,.12)' : 'rgba(120,120,140,.10)' }, label: { show: true, position: 'insideTop', color: isDark ? '#8f98ba' : '#9aa0b8', fontSize: 10 }, data: noServiceAreas }, lineStyle: { width: 3 } },
    { name: 'Факт', type: 'line', smooth: .25, symbol: 'circle', symbolSize: 5, data: viewPoints.map((p) => p.no_service ? null : p.actual ?? null), lineStyle: { type: 'dashed', width: 2 } },
    { name: 'Уточнённый прогноз', type: 'line', smooth: .25, symbol: 'circle', symbolSize: 4, data: viewPoints.map((p) => p.ts.startsWith(date) ? (adjustedByHour.get(Number(p.ts.slice(11, 13)))?.yhat_adj ?? null) : null), lineStyle: { width: 3, color: isDark ? '#ffca83' : '#c77a24' }, itemStyle: { color: isDark ? '#ffca83' : '#c77a24' }, connectNulls: false },
    { name: 'Выбранный период', type: 'scatter', symbolSize: 13, data: viewPoints.map((p) => p.ts === selectedPoint?.ts ? (p.yhat ?? p.actual ?? null) : null), itemStyle: { color: isDark ? '#f1edff' : '#252747', borderColor: isDark ? '#3a355b' : '#fff', borderWidth: 2 } },
  ] };
  const scenarioActive = factors.some((factor) => Math.abs((values[factor.id] ?? factor.default) - factor.default) > .001);
  useEffect(() => {
    if (!scenarioActive) { setBaselinePoints([]); return; }
    let active = true;
    const baselineParams = new URLSearchParams(params);
    factors.forEach((factor) => baselineParams.set(`factor.${factor.id}`, String(factor.default)));
    get<{points: Point[]}>(`/forecast?${baselineParams.toString()}`)
      .then((response) => { if (active) setBaselinePoints(route === 0 ? combineNetworkPoints(response.points ?? []) : response.points ?? []); })
      .catch(() => { if (active) setBaselinePoints([]); });
    return () => { active = false; };
  }, [scenarioActive, params, route, factors]);
  const baselinePoint = baselinePoints.find((point) => point.ts === selectedPoint?.ts);
  const baselineValue = baselinePoint?.yhat ?? baselinePoint?.actual ?? null;
  const riskRows = risk?.available ? risk.routes : [];
  const selectedRisk = riskRows.find((item) => item.route === route);
  const selectedRec = horizon === 'day' && !selectedStop && !segment ? recs.find((item) => Number(item.hour) === boundedTime) as Recommendation | undefined : undefined;
  const planned = horizon !== 'day' || selectedStop || segment ? null : route === 0 ? riskRows.length ? riskRows.reduce((sum, row) => sum + row.trams_now, 0) : null : selectedRec?.trams_now ?? null;
  const needed = horizon !== 'day' || selectedStop || segment ? null : route === 0 ? riskRows.length ? riskRows.reduce((sum, row) => sum + row.trams_needed, 0) : null : selectedRec?.trams_needed ?? null;
  const delta = planned == null || needed == null ? null : needed - planned;
  const riskValue = scenarioActive ? null : route === 0 ? riskRows.length ? Math.max(...riskRows.filter((item) => !item.no_service).map((item) => item.overload_risk_pct ?? 0), 0) : null : selectedRisk?.overload_risk_pct ?? null;
  const actionText = delta == null ? 'Для этого выбора расчёт выпуска недоступен' : delta > 0 ? `Добавить ${delta} ваг. к выпуску` : riskValue != null && riskValue > 10 ? 'Высокий риск: проверьте выпуск и интервал' : 'Расчётный выпуск достаточен';
  const rankedRoutes = routes.filter((item) => `${item.route} ${item.name}`.toLowerCase().includes(routeSearch.toLowerCase().trim())).sort((a, b) => {
    const first = riskRows.find((row) => row.route === a.route);
    const second = riskRows.find((row) => row.route === b.route);
    return (second?.overload_risk_pct ?? -1) - (first?.overload_risk_pct ?? -1) || a.route - b.route;
  });
  const changeDate = (next: string) => {
    if (!next) return;
    setDate(next);
    const nextRange = dateRange(next, horizon);
    setIntervalFrom(nextRange.from); setIntervalTo(nextRange.to);
    setFromHour(0); setToHour(23);
    setTimeIndex(horizon === 'day' ? 8 : horizon === 'month' ? +next.slice(8, 10) : +next.slice(5, 7));
  };
  const changeHorizon = (next: string) => {
    setHorizon(next);
    const nextRange = dateRange(date, next);
    setIntervalFrom(nextRange.from); setIntervalTo(nextRange.to);
    setFromHour(0); setToHour(23);
    setTimeIndex(next === 'day' ? 8 : next === 'month' ? +date.slice(8, 10) : +date.slice(5, 7));
  };
  const openMap = () => { setScreen('map'); if (horizon !== 'day') changeHorizon('day'); };
  const openPlanning = () => { setScreen('planning'); if (horizon === 'day') changeHorizon('month'); };
  const selectionLabel = selectedStopFeature ? selectedStopFeature.properties?.name ?? `Остановка ${selectedStop}`
    : segment ? `Участок маршрута ${route}` : route === 0 ? 'Вся сеть' : `Маршрут ${route}`;
  const periodLabel = horizon === 'day' ? `${String(boundedTime).padStart(2, '0')}:00–${String(boundedTime + 1).padStart(2, '0')}:00`
    : timeLabel;

  return <div className={`tp-app${leftRailCollapsed ? ' left-collapsed' : ''}${rightRailCollapsed ? ' right-collapsed' : ''}`}>
    <header className="tp-header">
      <div className="tp-brand"><span className="tp-brand-icon" aria-hidden="true"><i/><i/><i/></span><span>Ход города</span></div>
      <nav className="tp-nav" aria-label="Разделы"><button type="button" className={screen === 'map' ? 'active' : ''} onClick={openMap}>Карта</button><button type="button" className={screen === 'planning' ? 'active' : ''} onClick={openPlanning}>Планирование</button></nav>
      <div className="tp-header-end"><label className="tp-date">Дата <input type="date" value={date} min={minDate} max={maxDate} onChange={(event) => changeDate(event.target.value)} /></label><button className="tp-about-button" type="button" onClick={() => setAboutOpen(!aboutOpen)} aria-expanded={aboutOpen}>О прогнозе</button><button className="tp-logout" type="button" onClick={logout}>Выйти</button></div>
    </header>
    {aboutOpen && <div className="tp-about"><b>О прогнозе</b><button type="button" onClick={() => setAboutOpen(false)} aria-label="Закрыть">×</button><p>Прогноз посадок по маршрутам Москвы. Показатели остановок и участков оценены из данных маршрута. Рекомендация по выпуску не является командой на изменение движения.</p><p>Метод: профиль + LightGBM · WAPE-score {model?.lb_wape_score == null ? '—' : model.lb_wape_score.toFixed(3)} · версия {model?.model_version ?? '—'}</p></div>}
    <div className="tp-context"><div><span className="tp-overline">{screen === 'map' ? 'ОПЕРАТИВНАЯ КАРТА' : 'АНАЛИЗ И ВЫГРУЗКА'}</span><h1>{selectionLabel}</h1><p>{contextDate} · {periodLabel}{date <= '2025-10-31' ? ' · архив' : ' · прогноз'}{scenarioActive ? ' · сценарий' : ''}</p></div>{health === 'Нет связи' && <span className="tp-offline">Нет связи с данными</span>}</div>
    {error && <div className="tp-error" role="alert">{error}<button type="button" onClick={() => setError('')} aria-label="Закрыть сообщение">×</button></div>}
    <div className="tp-workspace">
      <aside className={`tp-routes${leftRailCollapsed ? ' is-collapsed' : ''}`} aria-label="Маршруты"><button className="tp-panel-toggle" type="button" aria-label={leftRailCollapsed ? 'Показать маршруты' : 'Скрыть маршруты'} aria-expanded={!leftRailCollapsed} onClick={() => setLeftRailCollapsed((value) => !value)}><span aria-hidden="true">{leftRailCollapsed ? '›' : '‹'}</span><small>{leftRailCollapsed ? 'Маршруты' : ''}</small></button><div className="tp-rail-heading"><span className="tp-overline">СЕТЬ</span><h2>Маршруты</h2></div><label className="tp-search"><span className="sr-only">Найти маршрут</span><input value={routeSearch} onChange={(event) => setRouteSearch(event.target.value)} placeholder="Найти маршрут" /></label><button className={`tp-route-item tp-network ${route === 0 ? 'selected' : ''}`} type="button" onClick={() => { selectRoute(0); setDetail('overview'); }}><span className="tp-route-number">●</span><span><b>Вся сеть</b><small>{routes.length} маршрутов</small></span></button><div className="tp-route-list">{rankedRoutes.map((item) => {
        const row = scenarioActive ? undefined : riskRows.find((riskRow) => riskRow.route === item.route);
        const level = row?.overload_risk_pct == null || row.no_service ? 'unknown' : row.overload_risk_pct > 10 ? 'high' : row.overload_risk_pct >= 5 ? 'medium' : 'low';
        return <button type="button" key={item.route} aria-label={`Маршрут ${item.route}`} className={`tp-route-item ${route === item.route ? 'selected' : ''}`} onClick={() => { selectRoute(item.route); setDetail('overview'); }}><span className="tp-route-number" style={{ '--route-accent': routeLineColor(item.route) } as CSSProperties}>{item.route}</span><span><b>Трамвай {item.route}</b><small>{row?.no_service ? 'Нет движения' : row?.overload_risk_pct == null ? 'Риск не рассчитан' : level === 'high' ? 'Высокий риск' : level === 'medium' ? 'Требует внимания' : 'В пределах нормы'}</small></span><i className={`tp-risk-dot ${level}`} aria-hidden="true"/></button>;
      })}</div><div className="tp-rail-foot"><span className="tp-risk-dot high"/> Высокий риск <span className="tp-risk-dot medium"/> Внимание</div></aside>
      <main className="tp-center">
        {screen === 'map' ? <>
          <div className="tp-map-toolbar"><div><span className="tp-overline">КАРТА МАРШРУТОВ</span><strong>{route === 0 ? 'Москва' : `Маршрут ${route}`}</strong></div><label className="tp-layer-toggle"><input type="checkbox" checked={showRisk} onChange={(event) => setShowRisk(event.target.checked)}/> Показывать риск</label></div>
          {showRisk && !scenarioActive && risk?.available && <div className="tp-map-legend"><b>Линия — маршрут</b><span><i className="medium"/>от 5%</span><span><i className="high"/>свыше 10%</span><small>Кайма показывает риск</small></div>}
          <div className="tp-map"><LeafletMap route={route} dateTime={`${snapshotDate}T${boundedTime}`} features={viewMapPoints} routeFeatures={geometry} segment={segment} selectedStop={selectedStop} risk={scenarioActive ? null : riskMap} capacity={risk?.capacity_nominal} onRoute={(id) => { selectRoute(id); setDetail('overview'); }} onStop={(stopRoute, id) => { setRoute(stopRoute); setSegmentDirection(null); setSegmentStart(null); setSegmentEnd(null); setSelectedStop(id); setDetail('overview'); }}/>{scenarioActive && <div className="tp-map-note">Карта показывает сценарный спрос. Риск переполнения для сценария не рассчитан.</div>}{selected && !selected.has_geo && <div className="tp-map-note">Для этого маршрута нет координат остановок.</div>}</div>
          {rightRailCollapsed && route !== 0 && <div className="tp-quick-card" style={{ '--route-accent': routeLineColor(route) } as CSSProperties}><div className="tp-quick-head"><span className="tp-quick-number">{route}</span><div><b>{selectionLabel}</b><small>{periodLabel}</small></div></div><p>{selectedStop ? 'Оценка посадок на остановке' : 'Прогноз посадок'} <strong>{selectedValue == null ? '—' : fmt.format(Math.round(selectedValue))}</strong></p>{!selectedStop && riskValue != null && <p>Риск превысить вместимость <strong>{riskValue.toFixed(1).replace('.', ',')}% · {riskLabel(riskValue).toLowerCase()}</strong></p>}<button type="button" onClick={() => { setRightRailCollapsed(false); setDetail('overview'); }}>Открыть подробности →</button></div>}
          <div className="tp-timeline"><div className="tp-timeline-head"><b>Время</b><strong>{periodLabel}</strong></div><input type="range" aria-label="Час на карте" min={0} max={23} value={boundedTime} onChange={(event) => setTimeIndex(Number(event.target.value))}/><div className="tp-time-ticks"><span>00:00</span><span>06:00</span><span>12:00</span><span>18:00</span><span>23:00</span></div></div>
          <div className="tp-chart"><div className="tp-chart-head"><div><span className="tp-overline">ДИНАМИКА</span><h2>Факт и прогноз</h2></div><span>{demoActive ? 'Воспроизведение истории' : hasActual ? 'Факт доступен' : 'Прогноз'}</span></div>{viewPoints.length ? <ReactECharts option={chart} style={{height:160}} notMerge/> : <div className="tp-empty">{loading ? 'Загружаем данные…' : 'Данных на этот период нет'}</div>}</div>
        </> : <div className="tp-planning"><div className="tp-planning-head"><div><span className="tp-overline">ПЛАНИРОВАНИЕ</span><h2>Прогноз по времени</h2></div><div className="tp-horizons">{([['day','День'],['month','Месяц'],['year','Год']] as const).map(([value,label]) => <button type="button" key={value} className={horizon === value ? 'active' : ''} onClick={() => changeHorizon(value)}>{label}</button>)}</div></div><div className="tp-planning-controls"><label>С <input type="date" value={intervalFrom} min={minDate} max={maxDate} onChange={(event) => setIntervalFrom(event.target.value)}/></label><label>По <input type="date" value={intervalTo} min={minDate} max={maxDate} onChange={(event) => setIntervalTo(event.target.value)}/></label><button type="button" onClick={() => { const next = dateRange(date,horizon); setIntervalFrom(next.from); setIntervalTo(next.to); }}>Весь период</button></div>{filterError && <div className="tp-field-error" role="alert">{filterError}</div>}<div className="tp-planning-chart">{viewPoints.length ? <ReactECharts option={chart} style={{height:320}} notMerge/> : <div className="tp-empty">{loading ? 'Загружаем данные…' : 'Данных на этот период нет'}</div>}</div><div className="tp-planning-summary"><span>Прогноз за интервал <b>{hasForecast ? fmt.format(Math.round(intervalForecast)) : '—'}</b></span><span>Факт <b>{hasActual ? fmt.format(Math.round(intervalActual)) : '—'}</b></span></div><div className="tp-export"><span>Выгрузить выбранный период</span><button type="button" disabled={!!filterError} onClick={() => exportForecast('csv')}>CSV ↓</button><button type="button" disabled={!!filterError} onClick={() => exportForecast('xlsx')}>XLSX ↓</button></div><HistoryPanel route={route} date={date} isDark={true}/></div>}
      </main>
      <aside className={`tp-inspector${rightRailCollapsed ? ' is-collapsed' : ''}`} aria-label="Анализ выбранного объекта"><button className="tp-panel-toggle" type="button" aria-label={rightRailCollapsed ? 'Показать анализ' : 'Скрыть анализ'} aria-expanded={!rightRailCollapsed} onClick={() => setRightRailCollapsed((value) => !value)}><span aria-hidden="true">{rightRailCollapsed ? '‹' : '›'}</span><small>{rightRailCollapsed ? 'Анализ' : ''}</small></button><div className="tp-inspector-heading"><span className="tp-overline">ВЫБРАННЫЙ ОБЪЕКТ</span><h2>{selectionLabel}</h2><p>{periodLabel}</p></div><div className="tp-tabs" role="tablist" aria-label="Подробности"><button type="button" role="tab" aria-selected={detail === 'overview'} onClick={() => setDetail('overview')}>Обзор</button><button type="button" role="tab" aria-selected={detail === 'why'} onClick={() => setDetail('why')}>Причины</button><button type="button" role="tab" aria-selected={detail === 'scenario'} onClick={() => setDetail('scenario')}>Сценарий</button><button type="button" role="tab" aria-selected={detail === 'events'} onClick={() => setDetail('events')}>События</button></div>
        <div className="tp-inspector-body">
          {detail === 'overview' && <><div className="tp-primary-stat"><span>{demoActive && selectedPoint?.actual != null ? 'Факт посадок' : selectedPoint?.actual != null && selectedPoint?.yhat == null ? 'Факт посадок' : adjustedPoint ? 'Уточнённый прогноз' : 'Прогноз посадок'}</span><strong>{selectedPoint?.no_service ? 'Нет движения' : demoActive && selectedPoint?.actual != null ? fmt.format(Math.round(selectedPoint.actual)) : selectedValue == null ? '—' : fmt.format(Math.round(selectedValue))}</strong><small>за {horizon === 'day' ? 'выбранный час' : horizon === 'month' ? 'выбранный день' : 'выбранный месяц'}{selectedStop || segment ? ' · оценка для выбранного объекта' : ''}</small>{demoActive && selectedPoint?.yhat != null && <small>Прогноз на этот час: {fmt.format(Math.round(selectedPoint.yhat))}</small>}{!demoActive && selectedBounds && <small>Ожидаемый диапазон: {selectedBounds}</small>}</div>{demoActive && selectedPoint?.actual != null && selectedPoint?.yhat != null && <div className="tp-action"><span className="tp-overline">СРАВНЕНИЕ С ПРОГНОЗОМ</span><h3>{selectedPoint.actual >= selectedPoint.yhat ? 'Выше' : 'Ниже'} на {fmt.format(Math.round(Math.abs(selectedPoint.actual - selectedPoint.yhat)))} посадок</h3></div>}{horizon === 'day' && !demoActive && <div className="tp-action"><span className="tp-overline">ВЫПУСК В ЭТОТ ЧАС</span><h3>{actionText}</h3>{planned != null && needed != null && <div className="tp-fleet"><span><small>По плану</small><b>{planned}</b></span><span aria-hidden="true">→</span><span><small>Расчёт</small><b>{needed}</b></span><span>вагонов</span></div>}{riskValue != null && !scenarioActive ? <RiskReadout value={riskValue} capacity={risk?.capacity_nominal ?? 188} network={route === 0}/> : <p>{scenarioActive ? 'Риск для сценария не рассчитан.' : 'Риск для этой даты не рассчитан.'}</p>}{planned != null && <small>«По плану» — запланированный выпуск, не число вагонов в реальном времени.</small>}</div>}{route === 0 && !demoActive && !scenarioActive && riskRows.length > 0 && <div className="tp-priorities"><span className="tp-overline">ТРЕБУЮТ ВНИМАНИЯ</span>{[...riskRows].filter((row) => !row.no_service).sort((a,b) => (b.overload_risk_pct ?? -1) - (a.overload_risk_pct ?? -1)).slice(0,3).map((row) => <button type="button" key={row.route} onClick={() => selectRoute(row.route)}><b>№ {row.route}</b><span>{row.overload_risk_pct == null ? 'Риск не рассчитан' : `Риск ${row.overload_risk_pct.toFixed(1).replace('.', ',')}%`}</span><span aria-hidden="true">↗</span></button>)}</div>}{route !== 0 && !selectedStop && <div className="tp-segment"><span className="tp-overline">УЧАСТОК МАРШРУТА</span><label>Направление<select value={segmentDirection ?? ''} onChange={(event) => { const direction = event.target.value === '' ? null : Number(event.target.value); const stops = routeStops.filter((feature) => Number(feature.properties?.direction) === direction).sort((a,b) => Number(a.properties?.seq) - Number(b.properties?.seq)); setSegmentDirection(direction); setSegmentStart(stops.length >= 2 ? Number(stops[0].properties?.seq) : null); setSegmentEnd(stops.length >= 2 ? Number(stops[stops.length - 1].properties?.seq) : null); }}><option value="">Весь маршрут</option>{directions.map((direction) => <option value={direction} key={direction}>Направление {direction}</option>)}</select></label>{segmentDirection != null && <><label>От<select value={segmentStart ?? ''} onChange={(event) => setSegmentStart(Number(event.target.value))}>{directionStops.slice(0,-1).map((feature) => <option key={feature.properties?.seq} value={Number(feature.properties?.seq)}>{feature.properties?.name}</option>)}</select></label><label>До<select value={segmentEnd ?? ''} onChange={(event) => setSegmentEnd(Number(event.target.value))}>{directionStops.slice(1).map((feature) => <option key={feature.properties?.seq} value={Number(feature.properties?.seq)}>{feature.properties?.name}</option>)}</select></label></>}{segmentError && <p className="tp-field-error">{segmentError}</p>}</div>}{selectedStop && <button type="button" className="tp-clear-stop" onClick={() => setSelectedStop(null)}>← К маршруту</button>}{canNowcast && route !== 0 && <button type="button" className="tp-nowcast" onClick={runNowcast}>Уточнить остаток дня ↗</button>}{currentNowcast && <p className="tp-nowcast-note">После {String(currentNowcast.now_hour).padStart(2,'0')}:00 прогноз уточнён по факту. Новая линия показана на графике.</p>}</>}
          {detail === 'why' && <WhyPanel route={route} date={snapshotDate} hour={boundedTime} isDark={true}/>}
          {detail === 'scenario' && <div className="tp-scenario"><h3>Что, если условия изменятся?</h3><p>Сравните прогноз при других условиях. Изменения применяются сразу.</p>{factors.map((factor) => <label key={factor.id}><span><b>{factor.label}</b><strong>{Math.round(((values[factor.id] ?? factor.default)-1)*100)}%</strong></span><input type="range" min={factor.min} max={factor.max} step={factor.step} value={values[factor.id] ?? factor.default} onChange={(event) => setValues({...values, [factor.id]: Number(event.target.value)})}/>{factor.effect_note && <small>{factor.effect_note}</small>}</label>)}{scenarioActive && <div className="tp-scenario-result"><span>Исходный прогноз <b>{baselineValue == null ? '—' : fmt.format(Math.round(baselineValue))}</b></span><span>При новых условиях <b>{selectedValue == null ? '—' : fmt.format(Math.round(selectedValue))}</b></span>{baselineValue != null && selectedValue != null && <strong>{selectedValue >= baselineValue ? '+' : '−'}{fmt.format(Math.abs(Math.round(selectedValue - baselineValue)))} посадок</strong>}</div>}<button type="button" onClick={() => setValues(Object.fromEntries(factors.map((factor) => [factor.id,factor.default])))}>Сбросить условия</button>{scenarioActive && <p className="tp-scenario-warning">Прогноз и расчёт выпуска обновлены. Вероятность переполнения для сценария не рассчитывается.</p>}</div>}
          {detail === 'events' && <div className="tp-events"><h3>События сети</h3><p>События для выбранного маршрута учитываются в прогнозе.</p><EventForm date={snapshotDate} route={route} routes={routes.map((item) => item.route)} onChanged={reloadEvents} onError={setError}/><div className="tp-event-list">{visibleRegistry.map((event) => <div key={event.id} className="tp-event"><time>{event.start === event.end ? event.start : `${event.start} — ${event.end}`}</time><b>{event.title}</b><small>{event.effect}</small>{event.user ? <button type="button" onClick={() => void deleteEvent(event.id)}>Удалить</button> : <a href={event.source_url} target="_blank" rel="noreferrer">Источник ↗</a>}</div>)}</div></div>}
        </div>
      </aside>
    </div>
    <div className="tp-replay-bar"><span>{!demoMode ? 'Как меняется прогноз в течение дня' : replayState === 'loading' ? 'Загружаем историю за 31 октября 2025…' : replayHour < 0 ? 'Исторический день · 31 октября 2025' : `31 октября 2025 · ${String(replayHour).padStart(2,'0')}:00 · ${fmt.format(observedHour)} посадок за час`}</span>{demoMode && replayHour >= 0 && <div className="tp-replay-progress" aria-label={`Пройдено ${replayHour + 1} часов из 24`}><i style={{width:`${(replayHour + 1) / 24 * 100}%`}}/></div>}<button type="button" onClick={replay} disabled={replayState === 'loading'}>{replayState === 'playing' ? 'Пауза' : replayState === 'paused' ? 'Продолжить' : replayState === 'complete' ? 'Повторить' : replayState === 'loading' ? 'Загрузка…' : 'Посмотреть день в движении'} <span aria-hidden="true">{replayState === 'playing' ? 'Ⅱ' : '↗'}</span></button>{demoMode && <><label className="tp-replay-speed">Скорость<select value={replaySpeed} onChange={(event) => setReplaySpeed(Number(event.target.value))}><option value={1}>1×</option><option value={2}>2×</option></select></label><button type="button" className="tp-replay-exit" onClick={() => stopReplay(true, true)}>Выйти</button></>}</div>
  </div>;
}
