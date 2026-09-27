"""1) Эффект погоды на остаток от профиля. 2) Внутридневная коррекция (nowcast): даёт ли прирост."""
import sys; sys.path.insert(0,'ml')
import pandas as pd, numpy as np
from core import load_labels, fit_profile, base_forecast, wape_score
y=load_labels()
w=pd.read_csv('data/external/weather_moscow_2025_hourly.csv',skiprows=3); w.columns=['time','t','prcp','snow','depth','rain','wind','cloud']
w['time']=pd.to_datetime(w.time); w['date']=w.time.dt.normalize(); w['hour']=w.time.dt.hour
# rolling baseline: для каждой недели — профиль по 4 предыдущим неделям (скользящий бэктест 1 неделя вперёд)
parts=[]
for end in pd.date_range('2025-02-02','2025-10-26',freq='W-SUN'):
    lv,sh=fit_profile(y,end); f=base_forecast(lv,sh,pd.date_range(end+pd.Timedelta(days=1),end+pd.Timedelta(days=7))); parts.append(f)
f=pd.concat(parts).merge(y[['route','date','hour','y']],on=['route','date','hour'])
off=pd.to_datetime(['2025-05-01','2025-05-02','2025-05-08','2025-05-09','2025-06-12','2025-06-13','2025-03-07','2025-04-30'])
f=f[~f.date.isin(off)]
d=f.groupby('date')[['y','pred']].sum(); d['r']=d.y/d.pred
dw=w[(w.hour>=6)&(w.hour<=22)].groupby('date').agg(t=('t','mean'),prcp=('prcp','sum'),snow=('snow','sum'),rain=('rain','sum'),wind=('wind','mean'))
d=d.join(dw); d['r_dm']=d.r/d.r.rolling(15,center=True,min_periods=5).median()  # убрать медленный тренд уровня
print('corr(daily ratio, weather):'); print(d[['r_dm','t','prcp','snow','rain','wind']].corr().r_dm.round(3).to_string())
for col,bins in [('prcp',[-.1,0.5,3,8,100]),('snow',[-.1,0.01,1,3,100]),('t',[-40,-10,0,10,20,40])]:
    print(col, d.groupby(pd.cut(d[col],bins),observed=True).r_dm.agg(['mean','count']).round(3).to_dict('index'))
# hourly: rain in the hour
h=f.merge(w[['date','hour','prcp','t','snow']],on=['date','hour']); h=h[(h.hour>=7)&(h.hour<=21)]
h['r']=h.y/h.pred.replace(0,np.nan)
print('hourly by prcp:',h.groupby(pd.cut(h.prcp,[-.1,0.05,0.5,2,50]),observed=True).apply(lambda g:g.y.sum()/g.pred.sum()).round(3).to_dict())
# 2) nowcast: в час H знаем факт 5..H-1, корректируем остаток дня коэффициентом fact/pred (с усадкой)
for H in (9,12,15):
    base=[];nc=[]
    for (r,dt),g in f.groupby(['route','date']):
        g=g.sort_values('hour'); past=g[(g.hour>=5)&(g.hour<H)]; fut=g[g.hour>=H]
        k=past.y.sum()/max(past.pred.sum(),1); k=1+0.7*(np.clip(k,0.5,1.5)-1)
        base.append(fut[['y','pred']].assign(adj=fut.pred*k))
    b=pd.concat(base); print(f'nowcast at {H}:00 → rest of day WAPE-score base {wape_score(b.y,b.pred):.4f} vs corrected {wape_score(b.y,b.adj):.4f}')
