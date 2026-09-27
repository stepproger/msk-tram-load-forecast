"""CV ансамблей профиль + LightGBM (вся история): веса, перемасштабирование уровня, модель на долях суток."""
import sys
sys.path.insert(0, 'ml')
import numpy as np, pandas as pd, lightgbm as lgb
from core import load_labels, fit_profile, wape_score, load_weather

y = load_labels()
HOL = pd.to_datetime(['2025-01-01','2025-01-02','2025-01-03','2025-01-06','2025-01-07','2025-01-08',
                      '2025-05-01','2025-05-02','2025-05-08','2025-05-09','2025-06-12','2025-06-13'])
y = y.merge(load_weather()[['date', 'hour', 't', 'prcp', 'snow']], on=['date', 'hour'], how='left')
y['hol'] = y.date.isin(HOL).astype(int); y['dow'] = y.date.dt.dayofweek
y['dt5'] = np.where(y.hol == 1, 6, y.dtype)
F = ['route', 'hour', 'dow', 'dt5', 'hol', 't', 'prcp', 'snow']
FOLDS = [('2025-03-02', '2025-04-30'), ('2025-04-30', '2025-06-30'), ('2025-06-30', '2025-08-31'),
         ('2025-08-31', '2025-10-31'), ('2025-09-28', '2025-10-31')]
res = []
for cut, end in FOLDS:
    cut, end = pd.Timestamp(cut), pd.Timestamp(end)
    tr, te = y[y.date <= cut], y[(y.date > cut) & (y.date <= end)].copy()
    lv, sh = fit_profile(tr.assign(dtype=tr.dt5), cut)
    te['p'] = (te.join(lv, on=['route', 'dt5']).level.fillna(0) * te.join(sh, on=['route', 'dt5', 'hour'])['shape'].fillna(0)).values
    m = lgb.LGBMRegressor(objective='l1', n_estimators=600, learning_rate=0.05, num_leaves=63, verbose=-1).fit(tr[F], tr.y, categorical_feature=['route'])
    te['g'] = np.clip(m.predict(te[F]), 0, None)
    # гбм на долях суток (форма), признаки — те же; target = доля часа в дне
    day = tr.groupby(['route', 'date']).y.transform('sum')
    trs = tr[day > 0].assign(share=tr.y / day)
    ms = lgb.LGBMRegressor(objective='l1', n_estimators=400, learning_rate=0.05, num_leaves=63, verbose=-1).fit(trs[F], trs.share, categorical_feature=['route'])
    te['s'] = np.clip(ms.predict(te[F]), 0, None)
    te['s'] = te.s / te.groupby(['route', 'date']).s.transform('sum')
    te['m'] = te.date.dt.to_period('M')
    sc = te.groupby(['route', 'm']).p.transform('sum') / te.groupby(['route', 'm']).g.transform('sum').replace(0, np.nan)
    te['g_scaled'] = te.g * sc.fillna(0)
    te['off'] = (te.dt5 >= 5)
    sc2 = te.groupby(['route', 'm', 'off']).p.transform('sum') / te.groupby(['route', 'm', 'off']).g.transform('sum').replace(0, np.nan)
    te['g_scaled2'] = te.g * sc2.fillna(0)
    daytot = te.groupby(['route', 'date']).p.transform('sum')
    te['p_gshape'] = daytot * te.s
    r = dict(fold=f'{cut:%d.%m}→{end:%d.%m}', profile=wape_score(te.y, te.p), gbm=wape_score(te.y, te.g))
    for w in (0.3, 0.5, 0.7):
        r[f'ens{w}'] = wape_score(te.y, (1 - w) * te.p + w * te.g)
        r[f'ens_scaled{w}'] = wape_score(te.y, (1 - w) * te.p + w * te.g_scaled)
    for w in (0.2, 0.3, 0.4, 0.5):
        r[f'E_off{w}'] = wape_score(te.y, (1 - w) * te.p + w * te.g_scaled2)
    r['shape_gbm'] = wape_score(te.y, te.p_gshape)
    r['shape_mix'] = wape_score(te.y, 0.5 * te.p + 0.5 * te.p_gshape)
    res.append(r); print(r['fold'], 'done', flush=True)
d = pd.DataFrame(res).set_index('fold')
print(d.T.round(4).to_string())
print(d.mean().round(4).sort_values(ascending=False).to_string())
