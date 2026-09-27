import duckdb, pandas as pd, numpy as np, urllib.request
pd.set_option('display.width',250); pd.set_option('display.max_rows',300); pd.set_option('display.max_columns',40)
L=duckdb.sql("select * from read_csv('dataset/labels/labels_day_*.csv',delim=';')").df()
L['date']=pd.to_datetime(L['date'])
D=L.pivot_table(index='date',columns='route',values='boardings',aggfunc='sum').fillna(0)
wk=D[D.index.dayofweek>=5]
print("WEEKEND daily totals Aug 15 - Oct 31:"); print(wk.loc['2025-08-15':].astype(int).to_string())
# calendar
s=urllib.request.urlopen("https://isdayoff.ru/api/getdata?year=2025&pre=1").read().decode()
cal=pd.Series(list(s),index=pd.date_range('2025-01-01','2025-12-31'))
dow=cal.index.dayofweek
print("\nNon-standard days 2025 (0=work,1=off,2=short):")
odd=cal[((dow<5)&(cal!='0'))|((dow>=5)&(cal!='1'))]
print(odd.to_string())
