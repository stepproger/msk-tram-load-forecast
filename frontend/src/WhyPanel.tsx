import { useEffect, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { get } from './api';

type Layer = { id: string; label: string; note: string; value: number; share_of_corrections_pct: number };
type Components = { route: number | null; date: string; hour: number | null; profile: { value: number; label: string; note: string }; layers: Layer[]; final: number };

const fmt = new Intl.NumberFormat('ru-RU');
const signed = (value: number) => `${value > 0 ? '+' : value < 0 ? '−' : ''}${fmt.format(Math.abs(Math.round(value)))}`;

export default function WhyPanel({ route, date, hour, isDark }: { route: number; date: string; hour: number; isDark: boolean }) {
  const [scope, setScope] = useState<'hour' | 'day'>('hour');
  const [data, setData] = useState<Components | null>(null);
  const [message, setMessage] = useState('');
  useEffect(() => {
    let active = true;
    const query = `/components?date=${date}${route ? `&route=${route}` : ''}${scope === 'hour' ? `&hour=${hour}` : ''}`;
    get<Components>(query)
      .then((result) => { if (active) { setData(result); setMessage(''); } })
      .catch((e) => { if (active) { setData(null); setMessage(e.message); } });
    return () => { active = false; };
  }, [route, date, hour, scope]);

  const colors = { base: isDark ? '#7592ff' : '#4d6fe2', up: isDark ? '#4fd49a' : '#1f9d6b', down: isDark ? '#ff7b7b' : '#d44b4b', text: isDark ? '#c4cbe3' : '#5d6278', grid: isDark ? '#393554' : '#ebedf5' };
  const ml = data?.layers.find((layer) => layer.id === 'ml');
  const tiny = data ? Math.max(data.profile.value, data.final) < 5 : false;
  let option: object | null = null;
  if (data && !tiny) {
    const names = [data.profile.label, ...data.layers.map((layer) => layer.label), 'Прогноз'];
    const base: (number | string)[] = [0];
    const bars: { value: number; itemStyle: { color: string } }[] = [{ value: data.profile.value, itemStyle: { color: colors.base } }];
    let level = data.profile.value;
    data.layers.forEach((layer) => {
      const next = level + layer.value;
      base.push(Math.min(level, next));
      bars.push({ value: Math.abs(layer.value), itemStyle: { color: layer.value >= 0 ? colors.up : colors.down } });
      level = next;
    });
    base.push(0);
    bars.push({ value: data.final, itemStyle: { color: colors.base } });
    const deltas = [data.profile.value, ...data.layers.map((layer) => layer.value), data.final];
    option = {
      grid: { left: 4, right: 52, top: 4, bottom: 4, containLabel: true },
      tooltip: { show: false },
      yAxis: { type: 'category', inverse: true, data: names, axisLabel: { color: colors.text, fontSize: 11, formatter: (name: string) => name.replace(' (LightGBM)', '') }, axisTick: { show: false } },
      xAxis: { type: 'value', splitLine: { lineStyle: { color: colors.grid } }, axisLabel: { show: false } },
      series: [
        { type: 'bar', stack: 'waterfall', silent: true, itemStyle: { color: 'transparent' }, data: base },
        { type: 'bar', stack: 'waterfall', data: bars, barMaxWidth: 16,
          label: { show: true, position: 'right', color: colors.text, fontSize: 10,
            formatter: (p: { dataIndex: number }) => p.dataIndex === 0 || p.dataIndex === deltas.length - 1 ? fmt.format(Math.round(deltas[p.dataIndex])) : signed(deltas[p.dataIndex]) } },
      ],
    };
  }

  return <article className="why-panel panel" aria-label="Почему такой прогноз">
    <div className="section-heading compact"><div><div className="eyebrow">ОБЪЯСНЕНИЕ ПРОГНОЗА</div><h2>Почему такой прогноз</h2></div>
      <div className="segmented small">{([['hour', `${String(hour).padStart(2, '0')}:00`], ['day', 'Сутки']] as const).map(([value, label]) => <button key={value} type="button" className={scope === value ? 'selected' : ''} onClick={() => setScope(value)}>{label}</button>)}</div></div>
    {ml && <div className="ml-badge" title="Доля модуля всех поправок к профилю по всему горизонту прогноза">ML-компонент: ≈{Math.round(ml.share_of_corrections_pct)}% всех поправок к профилю</div>}
    {message && <div className="why-empty">{message}</div>}
    {data && tiny && <div className="why-empty no-service">Нет движения: прогноз меньше 5 посадок — раскладывать нечего.</div>}
    {option && <ReactECharts option={option} style={{ height: 200 }} notMerge />}
    {data && <ul className="why-layers">
      <li><span className="why-swatch base" /><b>{data.profile.label}</b><em>{fmt.format(Math.round(data.profile.value))}</em><small>{data.profile.note}</small></li>
      {data.layers.map((layer) => <li key={layer.id} className={Math.abs(layer.value) < 0.5 ? 'is-zero' : ''}><span className={`why-swatch ${layer.value >= 0 ? 'up' : 'down'}`} /><b>{layer.label}</b><em className={layer.value > 0.5 ? 'up' : layer.value < -0.5 ? 'down' : ''}>{Math.abs(layer.value) < 0.5 ? '0' : signed(layer.value)}</em><small>{layer.note}</small></li>)}
      <li className="why-total"><span className="why-swatch base" /><b>Прогноз</b><em>{fmt.format(Math.round(data.final))}</em><small>Сумма слоёв, посадок за {scope === 'hour' ? 'час' : 'сутки'}</small></li>
    </ul>}
  </article>;
}
