"""Выбор модели на реальных данных: наш профиль против CatBoost / LightGBM (в лоб и гибрид) и наивных бейзлайнов.
Протокол как в задаче: обучение до среза, прогноз на 2 месяца вперёд без фактов периода. Праздники одинаково
известны всем моделям (признак is_holiday / тип дня «Вс»). Сезонный скаляр НЕ применяется (он калибруется по LB).
Также меряем время обучения, инференс полной сетки 14 640 строк и одного запроса (маршрут × день = 24 часа).
"""
import json, sys, time
sys.path.insert(0, 'ml')
import numpy as np, pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from core import load_labels, fit_profile, base_forecast, wape_score, load_weather

y = load_labels()
w = load_weather()[['date', 'hour', 't', 'prcp', 'snow']]
HOL = pd.to_datetime(['2025-01-01','2025-01-02','2025-01-03','2025-01-06','2025-01-07','2025-01-08',
                      '2025-05-01','2025-05-02','2025-05-08','2025-05-09','2025-06-12','2025-06-13'])
y = y.merge(w, on=['date', 'hour'], how='left')
y['hol'] = y.date.isin(HOL).astype(int)
y['dow'] = y.date.dt.dayofweek
y['dt5'] = np.where(y.hol == 1, 6, y.dtype)
FEATS = ['route', 'hour', 'dow', 'dt5', 'hol', 't', 'prcp', 'snow']
FOLDS = [('2025-03-02', '2025-04-30'), ('2025-04-30', '2025-06-30'), ('2025-06-30', '2025-08-31'),
         ('2025-08-31', '2025-10-31'), ('2025-09-28', '2025-10-31')]


def profile_pred(hist, cut, te):
    lv, sh = fit_profile(hist.assign(dtype=hist.dt5), cut)
    g = te[['route', 'date', 'hour', 'dt5']].rename(columns={'dt5': 'dtype'})
    g = g.join(lv, on=['route', 'dtype']).join(sh, on=['route', 'dtype', 'hour'])
    return (g.level.fillna(0) * g['shape'].fillna(0)).values


def rolling_base(hist, cut):
    """Прогноз «неделя вперёд» на историю до среза — для обучения гибрида на остатках."""
    parts = []
    for c in pd.date_range(hist.date.min() + pd.Timedelta(weeks=4), cut - pd.Timedelta(days=7), freq='W-SUN'):
        tr = hist[(hist.date > c) & (hist.date <= c + pd.Timedelta(days=7))]
        parts.append(tr.assign(base=profile_pred(hist[hist.date <= c], c, tr)))
    return pd.concat(parts)


rows, timing = [], {}
for cut, end in FOLDS:
    cut, end = pd.Timestamp(cut), pd.Timestamp(end)
    tr, te = y[y.date <= cut], y[(y.date > cut) & (y.date <= end)].copy()
    res = {}
    # наивные
    last = tr[tr.date > cut - pd.Timedelta(weeks=1)].groupby(['route', 'dow', 'hour']).y.mean()
    res['Сезонный наивный (та же неделя −1)'] = te.join(last.rename('p'), on=['route', 'dow', 'hour']).p.fillna(0).values
    res['Среднее по маршруту × часу (всё прошлое)'] = te.join(tr.groupby(['route', 'hour']).y.mean().rename('p'), on=['route', 'hour']).p.values
    # наш профиль
    t0 = time.perf_counter(); p = profile_pred(tr, cut, te); timing.setdefault('profile_fit', []).append(time.perf_counter() - t0)
    res['Наш профиль (уровень 2 нед. × форма 4 нед.)'] = p
    # GBM в лоб
    X, Xt = tr[FEATS], te[FEATS]
    t0 = time.perf_counter()
    m = lgb.LGBMRegressor(objective='l1', n_estimators=600, learning_rate=0.05, num_leaves=63, verbose=-1).fit(X, tr.y, categorical_feature=['route'])
    timing.setdefault('lgb_fit', []).append(time.perf_counter() - t0)
    res['LightGBM L1, всё прошлое'] = m.predict(Xt)
    t0 = time.perf_counter()
    cb = CatBoostRegressor(loss_function='MAE', iterations=800, learning_rate=0.08, depth=8, verbose=0, cat_features=['route'])
    cb.fit(X.astype({'route': str}), tr.y)
    timing.setdefault('cb_fit', []).append(time.perf_counter() - t0)
    res['CatBoost MAE, всё прошлое'] = cb.predict(Xt.astype({'route': str}))
    # GBM только на последних 8 неделях (ближе к уровню)
    tr8 = tr[tr.date > cut - pd.Timedelta(weeks=8)]
    cb8 = CatBoostRegressor(loss_function='MAE', iterations=800, learning_rate=0.08, depth=8, verbose=0, cat_features=['route'])
    cb8.fit(tr8[FEATS].astype({'route': str}), tr8.y)
    res['CatBoost MAE, последние 8 недель'] = cb8.predict(Xt.astype({'route': str}))
    # гибрид: CatBoost на отношении факт/профиль (вес = профиль), признаки — погода и календарь
    rb = rolling_base(tr, cut); rb = rb[rb.base > 20]
    hyb = CatBoostRegressor(loss_function='MAE', iterations=500, learning_rate=0.05, depth=6, verbose=0, cat_features=['route'])
    hyb.fit(rb[FEATS].astype({'route': str}), rb.y / rb.base, sample_weight=rb.base)
    res['Гибрид: профиль × CatBoost(остаток)'] = p * hyb.predict(Xt.astype({'route': str}))
    # CatBoost с профилем как признаком (знает свежий уровень и может его поправить)
    tb = rb.copy()
    fb = FEATS + ['base']
    cbp = CatBoostRegressor(loss_function='MAE', iterations=600, learning_rate=0.05, depth=6, verbose=0, cat_features=['route'])
    cbp.fit(tb[fb].astype({'route': str}), tb.y)
    res['CatBoost + профиль как признак'] = cbp.predict(te.assign(base=p)[fb].astype({'route': str}))
    res['Ансамбль 50/50: профиль + LightGBM'] = 0.5 * p + 0.5 * np.clip(res['LightGBM L1, всё прошлое'], 0, None)
    for k, v in res.items():
        rows.append(dict(fold=f'{cut:%d.%m}→{end:%d.%m}', model=k, score=wape_score(te.y.values, np.clip(v, 0, None))))
    print('fold', cut.date(), 'done', flush=True)

r = pd.DataFrame(rows).pivot(index='model', columns='fold', values='score')
r['среднее'] = r.mean(axis=1)
r['худший'] = r.drop(columns='среднее').min(axis=1)
r = r.sort_values('среднее', ascending=False)
print(r.round(4).to_string())

# Инференс: полная сетка и один запрос
grid = y[(y.date > '2025-08-31')].head(14640)
Xg = grid[FEATS]
lv, sh = fit_profile(y.assign(dtype=y.dt5), pd.Timestamp('2025-10-31'))
lk = {(k[0], k[1], h): lv[(k[0], k[1])] * sh.get((k[0], k[1], h), 0) for k in lv.index for h in range(24)}
def bench(f, n):
    t0 = time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter() - t0) / n * 1000
one = Xg.iloc[:24]
inf = {
    'Наш профиль (поиск в словаре)': (bench(lambda: [lk.get((r_, d_, h_), 0) for r_, d_, h_ in zip(grid.route, grid.dt5, grid.hour)], 5),
                                     bench(lambda: [lk.get((1, 1, h_), 0) for h_ in range(24)], 2000)),
    'LightGBM': (bench(lambda: m.predict(Xg), 5), bench(lambda: m.predict(one), 200)),
    'CatBoost': (bench(lambda: cb.predict(Xg.astype({'route': str})), 5), bench(lambda: cb.predict(one.astype({'route': str})), 200)),
}
ti = pd.DataFrame(inf, index=['сетка 14 640 строк, мс', 'один запрос 24 ч, мс']).T
ti['обучение, с'] = [np.mean(timing['profile_fit']), np.mean(timing['lgb_fit']), np.mean(timing['cb_fit'])]
print(ti.round(3).to_string())
json.dump({'scores': r.round(4).reset_index().to_dict('records'), 'inference': ti.round(4).reset_index().to_dict('records')},
          open('ml/research/model_selection.json', 'w'), ensure_ascii=False, indent=2)
