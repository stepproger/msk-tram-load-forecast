import { useEffect, useRef } from 'react';
import { riskColor } from './LeafletMap';

export type Recommendation = { hour: number; yhat: number; trams_now: number | null; trams_needed: number; load_pct: number | null; dwell_extra_sec: number | null; overload_risk_pct?: number | null; no_service?: boolean };
export type RouteRiskRow = { route: number; yhat: number; trams_now: number; trams_needed: number; load_pct: number | null; overload_risk_pct: number | null; no_service: boolean };
export type RiskSnapshot = { date: string; hour: number; capacity_nominal: number; capacity_model?: string; available: boolean; routes: RouteRiskRow[] };

const fmt = new Intl.NumberFormat('ru-RU');
const pct = (value: number | null | undefined) => value == null ? '—' : `${value < 10 ? value.toFixed(1).replace('.', ',') : Math.round(value)}%`;
const hh = (hour: number) => `${String(hour).padStart(2, '0')}:00`;

type Props = {
  route: number; hour: number; recs: Recommendation[]; risk: RiskSnapshot | null;
  onSelectHour: (hour: number) => void; onSelectRoute: (route: number) => void;
};

/** Main control element: fleet "now → needed" for the selected route and hour, with overload risk. */
export default function ActionPanel({ route, hour, recs, risk, onSelectHour, onSelectRoute }: Props) {
  const table = useRef<HTMLDivElement>(null);
  useEffect(() => {
    // Keep the selected hour visible inside the table without scrolling the whole rail.
    const box = table.current, row = box?.querySelector<HTMLElement>('.action-row.selected');
    if (box && row) box.scrollTop = row.offsetTop - box.clientHeight / 2 + row.offsetHeight / 2;
  }, [hour, route, recs]);
  const capacity = risk?.capacity_nominal ?? 188;
  const networkRows = risk?.routes ?? [];
  const current = route === 0
    ? networkRows.length ? {
        yhat: networkRows.reduce((sum, row) => sum + row.yhat, 0),
        now: networkRows.reduce((sum, row) => sum + row.trams_now, 0),
        needed: networkRows.reduce((sum, row) => sum + row.trams_needed, 0),
        risk: Math.max(...networkRows.map((row) => row.no_service ? 0 : row.overload_risk_pct ?? 0)),
        noService: networkRows.every((row) => row.no_service),
      } : null
    : (() => {
        const rec = recs.find((row) => row.hour === hour);
        return rec ? { yhat: rec.yhat, now: rec.trams_now, needed: rec.trams_needed, risk: rec.overload_risk_pct ?? null, noService: !!rec.no_service } : null;
      })();
  const riskiest = route === 0 ? [...networkRows].filter((row) => !row.no_service).sort((a, b) => (b.overload_risk_pct ?? 0) - (a.overload_risk_pct ?? 0))[0] : undefined;
  const delta = current && current.now != null ? current.needed - current.now : null;
  const tone = !current || current.noService ? 'idle' : (current.risk ?? 0) > 10 || (delta ?? 0) > 0 ? 'alert' : (current.risk ?? 0) >= 5 ? 'warn' : 'ok';
  const verdict = !current ? 'Нет расчёта выпуска на эту дату'
    : current.noService ? 'Нет движения — выпуск не требуется'
    : delta == null ? `Нужно ${current.needed} ваг.; фактический выпуск не загружен`
    : delta > 0 ? `Добавить ${delta} ваг. к выпуску`
    : (current.risk ?? 0) > 10 ? `Высокий риск переполнения${riskiest ? ` на маршруте ${riskiest.route}` : ''} — усилить выпуск или сократить интервал`
    : delta < 0 ? `Можно снять ${-delta} ваг.` : 'Выпуск достаточен';

  return <article className={`action-panel panel tone-${tone}`} aria-label="Рекомендация диспетчеру">
    <div className="eyebrow">РЕКОМЕНДАЦИЯ ДИСПЕТЧЕРУ · {route === 0 ? 'ВСЯ СЕТЬ' : `МАРШРУТ ${route}`} · {hh(hour)}–{hh(hour + 1)}</div>
    {current && !current.noService ? <div className="action-fleet"><span className="now">{current.now ?? '—'}</span><span className="arrow">→</span><span className="needed">{current.needed}</span><small>вагонов<br />сейчас → нужно</small></div>
      : <div className="action-fleet idle">{current ? 'нет движения' : '—'}</div>}
    <div className="action-verdict">{verdict}</div>
    {current && !current.noService && <div className="action-risk">
      <i style={{ background: riskColor({ risk: current.risk, noService: false }) }} />
      {current.risk == null ? 'Риск переполнения для этой даты не рассчитан'
        : <>Риск превысить номинал вагона ({capacity} чел.): <b>{pct(current.risk)}</b>{route === 0 && riskiest ? <> · выше всего у маршрута {riskiest.route}</> : null}</>}
    </div>}
    {current && !current.noService && <div className="action-meta">Прогноз: {fmt.format(Math.round(current.yhat))} посадок за час{risk?.capacity_model ? ` · вагон ${risk.capacity_model}` : ''}</div>}

    <div className="action-table" ref={table} role="table" aria-label={route === 0 ? 'Маршруты в выбранный час' : 'Часы выбранного маршрута'}>
      <div className="action-row head" role="row"><span>{route === 0 ? 'Маршрут' : 'Час'}</span><span>Прогноз</span><span>Вагоны</span><span>Загр.</span><span>Риск</span></div>
      {route === 0
        ? networkRows.map((row) => <button type="button" role="row" key={row.route} className={`action-row ${row.no_service ? 'no-service' : ''}`} onClick={() => onSelectRoute(row.route)} title="Открыть маршрут">
            <span>№ {row.route}</span>
            {row.no_service ? <span className="idle-cell">нет движения</span> : <><span>{fmt.format(Math.round(row.yhat))}</span><span>{row.trams_now} → <b className={row.trams_needed > row.trams_now ? 'more' : ''}>{row.trams_needed}</b></span><span>{pct(row.load_pct)}</span><span className="risk-cell"><i style={{ background: riskColor({ risk: row.overload_risk_pct, noService: false }) }} />{pct(row.overload_risk_pct)}</span></>}
          </button>)
        : recs.map((row) => <button type="button" role="row" key={row.hour} className={`action-row ${row.no_service ? 'no-service' : ''} ${row.hour === hour ? 'selected' : ''}`} onClick={() => onSelectHour(row.hour)}>
            <span>{hh(row.hour)}</span>
            {row.no_service ? <span className="idle-cell">нет движения</span> : <><span>{fmt.format(Math.round(row.yhat))}</span><span>{row.trams_now ?? '—'} → <b className={row.trams_now != null && row.trams_needed > row.trams_now ? 'more' : ''}>{row.trams_needed}</b></span><span>{pct(row.load_pct)}</span><span className="risk-cell"><i style={{ background: riskColor({ risk: row.overload_risk_pct ?? null, noService: false }) }} />{pct(row.overload_risk_pct)}</span></>}
          </button>)}
      {(route === 0 ? !networkRows.length : !recs.length) && <div className="action-empty">Рекомендации рассчитаны на 01.11–31.12.2025 и на прошедшие дни с данными выпуска.</div>}
    </div>
  </article>;
}
