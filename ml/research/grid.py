"""Выбор оценщика: окно формы, окно уровня, агрегат. Горизонт 4-8 недель."""
import duckdb, pandas as pd, numpy as np, itertools
L=duckdb.sql("select * from read_csv('dataset/labels/labels_day_*.csv',delim=';')").df(); L['date']=pd.to_datetime(L['date'])
routes=[1,7,11,12,17,25,26,28,50]
full=pd.MultiIndex.from_product([routes,pd.date_range('2025-01-01','2025-10-31'),range(24)],names=['route','date','hour'])
y=L.set_index(['route','date','hour']).boardings.reindex(full,fill_value=0).rename('y').reset_index()
y['dt']=y.date.dt.dayofweek
def wape(a,p): return np.abs(a-p).sum()/a.sum()
def predict(tr, te, shape_w, lvl_w, agg, grp):
    end=tr.date.max()
    m={'dow':lambda d:d,'5type':lambda d:d.map({0:0,1:1,2:1,3:1,4:4,5:5,6:6})}[grp]
    tr=tr.assign(g=m(tr.dt)); te=te.assign(g=m(te.dt))
    s=tr[tr.date>end-pd.Timedelta(weeks=shape_w)]
    day=s.groupby(['route','date','g']).y.transform('sum'); s=s.assign(sh=s.y/day.replace(0,np.nan))
    shape=s.groupby(['route','g','hour']).sh.agg(agg).rename('sh')
    shape=shape/shape.groupby(level=[0,1]).transform('sum')
    l=tr[tr.date>end-pd.Timedelta(weeks=lvl_w)].groupby(['route','date','g']).y.sum().reset_index()
    lvl=l.groupby(['route','g']).y.agg(agg).rename('lv')
    te=te.join(shape,on=['route','g','hour']).join(lvl,on=['route','g'])
    return te.sh.fillna(0)*te.lv.fillna(0)
folds=[('2025-10-05','2025-10-06','2025-10-31'),('2025-03-02','2025-03-03','2025-03-30'),('2025-02-23','2025-02-24','2025-03-30'),('2025-10-12','2025-10-13','2025-10-31')]
rows=[]
for sw,lw,agg,grp in itertools.product([4,6,8],[2,3,4],['median','mean'],['dow','5type']):
    sc=[]
    for tr_end,t0,t1 in folds:
        tr=y[y.date<=tr_end]; te=y[(y.date>=t0)&(y.date<=t1)]
        sc.append(1-wape(te.y.values,predict(tr,te,sw,lw,agg,grp).values))
    rows.append(dict(shape_w=sw,lvl_w=lw,agg=agg,grp=grp,**{f'f{i}':round(v,4) for i,v in enumerate(sc)},mean=round(np.mean(sc),4)))
r=pd.DataFrame(rows).sort_values('mean',ascending=False); print(r.head(12).to_string()); print(r.tail(3).to_string())
