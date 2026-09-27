"""Строгая валидация без утечек (всё обучение — только на данных до среза).
1) Кривая точности по горизонту (1–8 недель) → область определения.
2) Погода на горизонте «неделя»: множители осадков оцениваются только по прошлым неделям.
3) Холодный старт маршрута (адаптация): 7/14 дней истории, форма — от других маршрутов.
Результаты → ml/research/validation_results.json (используется в docs/MODEL.md и графиках).
"""
import json, sys
sys.path.insert(0, 'ml')
import numpy as np, pandas as pd
from core import load_labels, fit_profile, base_forecast, wape_score, load_weather, ROUTES, TYPE_OF_DOW
from config import events as ev

y = load_labels()
OFF = pd.to_datetime(['2025-01-01','2025-01-02','2025-01-03','2025-01-06','2025-01-07','2025-01-08',
                      '2025-05-01','2025-05-02','2025-05-08','2025-05-09','2025-06-12','2025-06-13'])
res = {}

# 1) Горизонт: срезы каждое воскресенье, прогноз на недели 1..8 вперёд
rows = []
for cut in pd.date_range('2025-02-02', '2025-09-07', freq='W-SUN'):
    lv, sh = fit_profile(y, cut)
    f = base_forecast(lv, sh, pd.date_range(cut + pd.Timedelta(days=1), min(cut + pd.Timedelta(weeks=8), y.date.max())))
    f = f.merge(y[['route', 'date', 'hour', 'y']], on=['route', 'date', 'hour'])
    f = f[~f.date.isin(OFF)]
    f['wk'] = ((f.date - cut).dt.days - 1) // 7 + 1
    for wk, g in f.groupby('wk'):
        rows.append(dict(cut=str(cut.date()), wk=int(wk), score=wape_score(g.y, g.pred), bias=g.pred.sum() / g.y.sum() - 1))
h = pd.DataFrame(rows)
curve = h.groupby('wk').agg(median=('score', 'median'), p10=('score', lambda s: s.quantile(.1)), bias=('bias', 'median'))
print('Горизонт (недели) → WAPE-score медиана / p10 / смещение:\n', curve.round(3).to_string())
res['horizon_curve'] = curve.round(4).reset_index().to_dict('records')

# 2) Погода на горизонте «неделя вперёд», множители — только по прошлому
w = load_weather()
wk_rows = []
for cut in pd.date_range('2025-02-02', '2025-10-26', freq='W-SUN'):
    lv, sh = fit_profile(y, cut)
    f = base_forecast(lv, sh, pd.date_range(cut + pd.Timedelta(days=1), min(cut + pd.Timedelta(days=7), y.date.max())))
    f['cut'] = cut
    wk_rows.append(f)
b = pd.concat(wk_rows).merge(y[['route', 'date', 'hour', 'y']], on=['route', 'date', 'hour'])
b = b[~b.date.isin(OFF)].merge(w[['date', 'hour', 'prcp']], on=['date', 'hour'])
b['bin'] = pd.cut(b.prcp, ev.PRCP_BINS, labels=False)
out = []
for cut, g in b.groupby('cut'):
    past = b[(b.date <= cut) & b.hour.between(6, 22)]
    if past.date.nunique() < 42:      # минимум 6 недель истории для оценки множителей
        continue
    r = past.groupby('bin').apply(lambda s: s.y.sum() / s.pred.sum())
    fac = (r / r.get(0, 1.0)).reindex(range(4)).fillna(1.0)
    lvl_w = past[past.date > cut - pd.Timedelta(weeks=2)]
    ref = lvl_w.bin.map(fac).mean()   # уровень уже содержит среднюю погоду окна
    adj = g.pred * g.bin.map(fac).values / ref
    out.append(dict(cut=cut, y=g.y.sum(), e0=(g.y - g.pred).abs().sum(), e1=(g.y - adj).abs().sum(),
                    wet=(g.prcp > 0.5).sum()))
o = pd.DataFrame(out)
s0, s1 = 1 - o.e0.sum() / o.y.sum(), 1 - o.e1.sum() / o.y.sum()
wet = o[o.wet >= o.wet.median()]
wins = (o.e1 < o.e0).mean()
print(f'\nПогода, неделя вперёд ({len(o)} недель, апр–окт): без {s0:.4f} → с погодой {s1:.4f} (Δ {s1-s0:+.4f}); '
      f'дождливые недели: {1-wet.e0.sum()/wet.y.sum():.4f} → {1-wet.e1.sum()/wet.y.sum():.4f}; погода лучше в {wins:.0%} недель')
res['weather_week_ahead'] = dict(weeks=len(o), score_base=round(s0, 4), score_weather=round(s1, 4), delta=round(s1 - s0, 4),
                                 wet_weeks_base=round(1 - wet.e0.sum() / wet.y.sum(), 4), wet_weeks_weather=round(1 - wet.e1.sum() / wet.y.sum(), 4),
                                 share_weeks_improved=round(wins, 3))

# 3) Холодный старт: маршрут с N днями истории; форма = средняя форма других маршрутов
cut, t0, t1 = pd.Timestamp('2025-10-05'), '2025-10-06', '2025-10-31'
lv_full, sh_full = fit_profile(y, cut)
test = y[(y.date >= t0) & (y.date <= t1)]
cs = []
for r in ROUTES:
    other = sh_full.drop(r, level='route').groupby(level=['dtype', 'hour']).mean()
    for n in (7, 14):
        hist = y[(y.route == r) & (y.date > cut - pd.Timedelta(days=n)) & (y.date <= cut)]
        lv_r = hist.groupby(['dtype', 'date']).y.sum().groupby(level=0).median()
        wk_level = lv_r.reindex([0, 1, 4, 5, 6])
        if wk_level.isna().any():          # при 7 днях есть все типы; иначе — от будней × средние доли
            wk_level = wk_level.fillna(wk_level.mean())
        g = test[test.route == r].copy()
        g['pred'] = g.dtype.map(wk_level).values * g.set_index(['dtype', 'hour']).index.map(other).values
        cs.append(dict(route=r, days=n, score=wape_score(g.y, g.pred)))
    g = test[test.route == r].merge(base_forecast(lv_full, sh_full, pd.date_range(t0, t1), routes=[r])[['route', 'date', 'hour', 'pred']])
    cs.append(dict(route=r, days='полная (4 нед.)', score=wape_score(g.y, g.pred)))
c = pd.DataFrame(cs).pivot(index='route', columns='days', values='score')
print('\nХолодный старт (октябрь), WAPE-score по маршрутам:\n', c.round(3).to_string())
tot = pd.DataFrame(cs).groupby('days').score.mean()
print(tot.round(4).to_string())
res['cold_start'] = {str(k): round(v, 4) for k, v in tot.items()}
json.dump(res, open('ml/research/validation_results.json', 'w'), ensure_ascii=False, indent=2, default=str)

# 2b) Погода в дождливые часы (out-of-sample, те же множители из прошлого)
hr = []
for cut, g in b.groupby('cut'):
    past = b[(b.date <= cut) & b.hour.between(6, 22)]
    if past.date.nunique() < 42:
        continue
    r = past.groupby('bin').apply(lambda s: s.y.sum() / s.pred.sum())
    fac = (r / r.get(0, 1.0)).reindex(range(4)).fillna(1.0)
    ref = past[past.date > cut - pd.Timedelta(weeks=2)].bin.map(fac).mean()
    hr.append(g.assign(adj=g.pred * g.bin.map(fac).values / ref))
hr = pd.concat(hr)
rain = {}
for lo, hi, name in [(0.5, 2, '0.5–2 мм/ч'), (2, 1000, '>2 мм/ч')]:
    s = hr[(hr.prcp > lo) & (hr.prcp <= hi) & hr.hour.between(6, 22)]
    rain[name] = dict(hours=int(s[['date', 'hour']].drop_duplicates().shape[0]), base=round(wape_score(s.y, s.pred), 4),
                      weather=round(wape_score(s.y, s.adj), 4), bias_base=round(s.pred.sum() / s.y.sum() - 1, 3),
                      bias_weather=round(s.adj.sum() / s.y.sum() - 1, 3))
print('\nДождливые часы (out-of-sample):', rain)
res['weather_rain_hours'] = rain
json.dump(res, open('ml/research/validation_results.json', 'w'), ensure_ascii=False, indent=2, default=str)
