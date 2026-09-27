import { useEffect, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { get } from './api';

type PlanFactPoint = { ts: string; plan: number | null; fact: number | null; q10: number | null; q90: number | null; trams: number | null; no_service: boolean; outside: 'above' | 'below' | null };
type PlanFact = { from: string; to: string; coverage_pct: number | null; hours_with_service: number; wape_score: number | null; band_note: string; points: PlanFactPoint[] };
export type Anomaly = { route: number | null; date_from: string; date_to: string; kind: string; effect_pct: number | null; source_url?: string; note?: string };

// Range of the week-ahead backtest shipped in artifacts/forecast/backtest_hourly.parquet.
const HISTORY_FROM = '2025-02-03';
const HISTORY_TO = '2025-10-31';
const fmt = new Intl.NumberFormat('ru-RU');
const shiftDate = (iso: string, days: number) => new Date(Date.parse(`${iso}T00:00:00Z`) + days * 86400000).toISOString().slice(0, 10);
const ruDate = (iso: string) => `${iso.slice(8, 10)}.${iso.slice(5, 7)}`;
const KIND_CLASS: Record<string, string> = { 'ремонт': 'repair', 'праздник': 'holiday', 'изменение трассы': 'reroute', 'сбой данных': 'glitch' };

function runs(points: PlanFactPoint[], test: (point: PlanFactPoint) => boolean) {
  const areas: [number, number][] = [];
  points.forEach((point, index) => {
    if (!test(point)) return;
    const last = areas[areas.length - 1];
    if (last && last[1] === index - 1) last[1] = index; else areas.push([index, index]);
  });
  return areas;
}

export default function HistoryPanel({ route, date, isDark }: { route: number; date: string; isDark: boolean }) {
  const [days, setDays] = useState(7);
  const [data, setData] = useState<PlanFact | null>(null);
  const [anomalies, setAnomalies] = useState<{ available: boolean; items: Anomaly[] } | null>(null);
  const [message, setMessage] = useState('');
  const to = date > HISTORY_TO ? HISTORY_TO : date < HISTORY_FROM ? shiftDate(HISTORY_FROM, days - 1) : date;
  const from = shiftDate(to, -(days - 1)) < HISTORY_FROM ? HISTORY_FROM : shiftDate(to, -(days - 1));
  useEffect(() => {
    let active = true;
    get<PlanFact>(`/plan-fact?from=${from}&to=${to}${route ? `&route=${route}` : ''}`)
      .then((result) => { if (active) { setData(result); setMessage(''); } })
      .catch((e) => { if (active) { setData(null); setMessage(e.message); } });
    return () => { active = false; };
  }, [route, from, to]);
  useEffect(() => {
    let active = true;
    get<{ available: boolean; items: Anomaly[] }>(`/anomalies${route ? `?route=${route}` : ''}`)
      .then((result) => { if (active) setAnomalies(result); })
      .catch(() => { if (active) setAnomalies(null); });
    return () => { active = false; };
  }, [route]);

  const points = data?.points ?? [];
  const inPeriod = (anomalies?.items ?? []).filter((item) => item.date_from <= to && item.date_to >= from);
  const c = isDark
    ? { plan: '#7592ff', fact: '#f1f3ff', band: 'rgba(117,146,255,.22)', out: '#ff6b6b', text: '#aeb8d7', grid: '#393554', idle: 'rgba(160,160,190,.10)', anomaly: 'rgba(255,176,72,.18)', tipBg: '#292342', tipBorder: '#46436b', tipText: '#f2f1ff' }
    : { plan: '#4d6fe2', fact: '#272743', band: 'rgba(77,111,226,.15)', out: '#d43d3d', text: '#6d7190', grid: '#ebedf5', idle: 'rgba(120,120,140,.10)', anomaly: 'rgba(230,140,30,.16)', tipBg: '#fff', tipBorder: '#e5e9f4', tipText: '#272743' };
  const anomalyAreas = inPeriod.map((item) => {
    const start = points.findIndex((point) => point.ts.slice(0, 10) >= item.date_from);
    let end = -1;
    points.forEach((point, index) => { if (point.ts.slice(0, 10) <= item.date_to) end = index; });
    return start >= 0 && end >= start ? [{ xAxis: start, name: item.kind }, { xAxis: end }] : null;
  }).filter(Boolean);
  const outside = points.filter((point) => point.outside);
  const option = {
    grid: { left: 40, right: 8, top: 14, bottom: 22 },
    tooltip: {
      trigger: 'axis', confine: true, backgroundColor: c.tipBg, borderColor: c.tipBorder, textStyle: { color: c.tipText, fontSize: 12 },
      formatter: (entries: { dataIndex: number }[]) => {
        const point = points[entries[0]?.dataIndex];
        if (!point) return '';
        const day = point.ts.slice(0, 10);
        const reasons = inPeriod.filter((item) => item.date_from <= day && day <= item.date_to)
          .map((item) => `<br/><b>⚑ ${item.kind}</b>${item.route ? ` (маршрут ${item.route})` : ''}${item.effect_pct != null ? `: ${item.effect_pct > 0 ? '+' : ''}${Math.round(item.effect_pct)}%` : ''}${item.note ? ` — ${item.note}` : ''}`).join('');
        const head = `<b>${ruDate(day)} ${point.ts.slice(11, 16)}</b>`;
        if (point.no_service) return `${head}<br/>Нет движения${reasons}`;
        return `${head}<br/>План: ${point.plan == null ? '—' : fmt.format(Math.round(point.plan))}<br/>Факт: ${point.fact == null ? '—' : fmt.format(point.fact)}`
          + `${point.q10 != null && point.q90 != null ? `<br/>Интервал q10–q90: ${fmt.format(Math.round(point.q10))}–${fmt.format(Math.round(point.q90))}` : ''}`
          + `${point.outside ? `<br/><b style="color:${c.out}">Факт ${point.outside === 'above' ? 'выше' : 'ниже'} интервала</b>` : ''}${reasons}`;
      },
    },
    xAxis: { type: 'category', data: points.map((point) => `${ruDate(point.ts)} ${point.ts.slice(11, 13)}`), axisLabel: { color: c.text, fontSize: 10, interval: (index: number) => points[index]?.ts.slice(11, 13) === '12' && Math.floor(index / 24) % (days > 7 ? 5 : 1) === 0, formatter: (value: string) => value.slice(0, 5) }, axisTick: { show: false } },
    yAxis: { type: 'value', splitLine: { lineStyle: { color: c.grid } }, axisLabel: { color: c.text, fontSize: 10, formatter: (value: number) => value >= 1000 ? `${value / 1000}k` : String(value) } },
    series: [
      { type: 'line', stack: 'band', symbol: 'none', silent: true, lineStyle: { opacity: 0 }, data: points.map((point) => point.no_service ? null : point.q10) },
      { type: 'line', stack: 'band', symbol: 'none', silent: true, lineStyle: { opacity: 0 }, areaStyle: { color: c.band }, data: points.map((point) => point.no_service || point.q10 == null || point.q90 == null ? null : point.q90 - point.q10) },
      { name: 'План', type: 'line', symbol: 'none', lineStyle: { width: 2, color: c.plan }, itemStyle: { color: c.plan }, data: points.map((point) => point.no_service ? null : point.plan),
        markArea: { silent: true, itemStyle: { color: c.idle }, data: runs(points, (point) => point.no_service).map(([start, end]) => [{ xAxis: start }, { xAxis: end }]) } },
      { name: 'Факт', type: 'line', symbol: 'none', lineStyle: { width: 1.5, color: c.fact, type: 'dashed' }, itemStyle: { color: c.fact }, data: points.map((point) => point.no_service ? null : point.fact),
        markArea: { silent: true, itemStyle: { color: c.anomaly }, label: { show: true, position: 'insideTop', color: c.text, fontSize: 10 }, data: anomalyAreas } },
      { name: 'Вне интервала', type: 'scatter', symbolSize: 6, itemStyle: { color: c.out }, data: points.map((point) => point.outside ? point.fact : null) },
    ],
  };

  return <article className="history-panel panel" aria-label="План и факт по прошедшим дням">
    <div className="section-heading compact"><div><div className="eyebrow">СЛОЖНЫЕ СИТУАЦИИ</div><h2>План / факт и пробои интервала</h2></div>
      <div className="segmented small">{[7, 30].map((value) => <button key={value} type="button" className={days === value ? 'selected' : ''} onClick={() => setDays(value)}>{value} дн.</button>)}</div></div>
    <div className="history-period">{ruDate(from)}–{ruDate(to)}.{date > HISTORY_TO ? ' Для прогнозного периода факта ещё нет — показаны последние дни истории.' : ''} План — прогноз на неделю вперёд.</div>
    <div className="history-legend"><span><i className="plan" />План</span><span><i className="fact" />Факт</span><span><i className="band" />q10–q90</span><span><i className="out" />Вне интервала</span><span><i className="idle" />Нет движения</span>{anomalyAreas.length > 0 && <span><i className="anomaly" />Аномалия</span>}</div>
    {message ? <div className="why-empty">{message}</div> : <ReactECharts option={option} style={{ height: 190 }} notMerge />}
    {data && <div className="history-summary">
      <span>Факт внутри интервала: <b>{data.coverage_pct == null ? '—' : `${Math.round(data.coverage_pct)}%`}</b> часов (ожидаемо ~80%)</span>
      <span>WAPE-score за период: <b>{data.wape_score == null ? '—' : data.wape_score.toFixed(3)}</b></span>
      <span>Пробоев: <b>{outside.length}</b> (выше {outside.filter((point) => point.outside === 'above').length}, ниже {outside.filter((point) => point.outside === 'below').length})</span>
    </div>}
    {data && <div className="history-note">Полоса — {data.band_note}. Часы без движения исключены из оценки.</div>}
    <div className="anomaly-block">
      <b className="anomaly-title">Провалы данных и причины</b>
      {anomalies == null || !anomalies.available
        ? <div className="why-empty">Реестр аномалий ещё не выгружен ML-частью (artifacts/forecast/anomalies.json). После выгрузки здесь появятся ремонты, праздники, изменения трасс и сбои данных — с оценкой эффекта и ссылкой на источник, а на графике — затенённые интервалы.</div>
        : anomalies.items.length === 0 ? <div className="why-empty">Для этого маршрута аномалий не найдено.</div>
        : <div className="anomaly-list">{anomalies.items.map((item, index) => <div className={`anomaly-row ${item.date_from <= to && item.date_to >= from ? 'in-period' : ''}`} key={`${item.date_from}-${item.route}-${index}`}>
            <span className={`anomaly-kind ${KIND_CLASS[item.kind] ?? ''}`}>{item.kind}</span>
            <span className="anomaly-dates">{item.date_from === item.date_to ? ruDate(item.date_from) : `${ruDate(item.date_from)}–${ruDate(item.date_to)}`}{item.date_from.slice(0, 4) !== '2025' ? ` ${item.date_from.slice(0, 4)}` : ''}</span>
            <span>{item.route ? `М${item.route}` : 'сеть'}</span>
            <b className={item.effect_pct != null && item.effect_pct < 0 ? 'down' : 'up'}>{item.effect_pct == null ? '—' : `${item.effect_pct > 0 ? '+' : ''}${Math.round(item.effect_pct)}%`}</b>
            {item.source_url ? <a href={item.source_url} target="_blank" rel="noreferrer">Источник ↗</a> : <span />}
            {item.note && <small>{item.note}</small>}
          </div>)}</div>}
    </div>
  </article>;
}
