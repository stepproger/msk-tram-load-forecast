import { useState } from 'react';
import type { FormEvent } from 'react';
import { apiFetch } from './api';

type Props = { date: string; route: number; routes: number[]; onChanged: () => void; onError: (message: string) => void };

/** Dispatcher adds a network event (repair, launch, mass event); the forecast is rescaled for its dates and routes. */
export default function EventForm({ date, route, routes, onChanged, onError }: Props) {
  const [open, setOpen] = useState(false);
  const [title, setTitle] = useState('');
  const [category, setCategory] = useState('event');
  const [start, setStart] = useState(date), [end, setEnd] = useState(date);
  const [scope, setScope] = useState<number>(route);
  const [effect, setEffect] = useState(15);
  const [busy, setBusy] = useState(false);
  const toggle = () => { setOpen(!open); setStart(date); setEnd(date); setScope(route); };
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!title.trim()) { onError('Укажите название события'); return; }
    setBusy(true);
    try {
      await apiFetch('/events', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: title.trim(), category, start, end, routes: scope ? [scope] : [], effect_pct: effect }) });
      setOpen(false); setTitle(''); onChanged();
    } catch (e: any) { onError(e.message); } finally { setBusy(false); }
  };
  if (!open) return <button type="button" className="event-add" onClick={toggle}>+ Добавить событие</button>;
  return <form className="event-form" onSubmit={submit}>
    <label className="wide">Название<input value={title} maxLength={120} placeholder="Например: концерт в Лужниках" onChange={(e) => setTitle(e.target.value)} /></label>
    <label>Тип<select value={category} onChange={(e) => { setCategory(e.target.value); setEffect(e.target.value === 'repair' ? -40 : e.target.value === 'launch' ? 20 : 15); }}>
      <option value="event">Мероприятие</option><option value="repair">Ремонт</option><option value="launch">Запуск / изменение трассы</option></select></label>
    <label>Маршрут<select value={scope} onChange={(e) => setScope(Number(e.target.value))}><option value={0}>Все</option>{routes.map((item) => <option key={item} value={item}>№ {item}</option>)}</select></label>
    <label>С<input type="date" value={start} onChange={(e) => setStart(e.target.value)} /></label>
    <label>По<input type="date" value={end} onChange={(e) => setEnd(e.target.value)} /></label>
    <label>Эффект, %<input type="number" min={-90} max={200} step={5} value={effect} onChange={(e) => setEffect(Number(e.target.value))} /></label>
    <div className="event-form-actions"><button type="submit" disabled={busy}>{busy ? 'Сохраняем…' : 'Применить к прогнозу'}</button><button type="button" className="ghost" onClick={() => setOpen(false)}>Отмена</button></div>
  </form>;
}
