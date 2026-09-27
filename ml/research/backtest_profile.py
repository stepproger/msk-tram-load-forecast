"""Где сидит ошибка: профиль (форма суток) vs уровень. Бэктесты на labels."""
import duckdb, pandas as pd, numpy as np
L=duckdb.sql("select * from read_csv('dataset/labels/labels_day_*.csv',delim=';')").df()
L['date']=pd.to_datetime(L['date'])
routes=[1,7,11,12,17,25,26,28,50]
full=pd.MultiIndex.from_product([routes,pd.date_range('2025-01-01','2025-10-31'),range(24)],names=['route','date','hour'])
y=L.set_index(['route','date','hour']).boardings.reindex(full,fill_value=0).rename('y').reset_index()
off={'2025-01-01','2025-01-02','2025-01-03','2025-01-06','2025-01-07','2025-01-08','2025-05-01','2025-05-02','2025-05-08','2025-05-09','2025-06-12','2025-06-13'}
y['dt']=np.where(y.date.dt.strftime('%Y-%m-%d').isin(off),6,y.date.dt.dayofweek)  # holiday -> Sunday type
def wape(a,p): return np.abs(a-p).sum()/a.sum()
def run(tr0,tr1,te0,te1,agg='median'):
    tr=y[(y.date>=tr0)&(y.date<=tr1)]; te=y[(y.date>=te0)&(y.date<=te1)].copy()
    prof=tr.groupby(['route','dt','hour']).y.agg(agg).rename('p')
    te=te.join(prof,on=['route','dt','hour'])
    # oracle day total: rescale predicted profile to true daily total
    s=te.groupby(['route','date'])[['y','p']].transform('sum')
    te['p_or']=te.p*s.y/s.p.replace(0,np.nan)
    # oracle route-period level (month): rescale by route total
    r=te.groupby('route')[['y','p']].transform('sum'); te['p_lvl']=te.p*r.y/r.p
    return dict(fold=f'{tr0[5:]}..{tr1[5:]} -> {te0[5:]}..{te1[5:]}', naive=1-wape(te.y,te.p), level_oracle=1-wape(te.y,te.p_lvl), day_oracle=1-wape(te.y,te.p_or.fillna(0)), bias=te.p.sum()/te.y.sum()-1)
res=[run('2025-09-08','2025-10-05','2025-10-06','2025-10-31'),
     run('2025-09-01','2025-09-30','2025-10-01','2025-10-31'),
     run('2025-02-03','2025-03-02','2025-03-03','2025-03-31'),
     run('2025-02-03','2025-03-02','2025-03-03','2025-04-30'),
     run('2025-08-04','2025-08-31','2025-09-01','2025-10-31'),
     run('2025-09-08','2025-10-05','2025-10-06','2025-10-31','mean')]
print(pd.DataFrame(res).round(3).to_string())
# per-route error share for Oct fold
tr=y[(y.date>='2025-09-08')&(y.date<='2025-10-05')]; te=y[(y.date>='2025-10-06')].copy()
te=te.join(tr.groupby(['route','dt','hour']).y.median().rename('p'),on=['route','dt','hour'])
te['e']=(te.y-te.p).abs()
g=te.groupby('route').agg(y=('y','sum'),p=('p','sum'),e=('e','sum')); g['share_of_err']=g.e/g.e.sum(); g['route_wape']=g.e/g.y; g['bias']=g.p/g.y-1
print(g.round(3).to_string())
h=te.groupby('hour').agg(y=('y','sum'),e=('e','sum')); h['share_err']=(h.e/h.e.sum()).round(3); print(h.share_err.to_dict())
