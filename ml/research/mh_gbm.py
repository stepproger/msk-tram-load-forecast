"""Многогоризонтный LightGBM, стэкинг и walk-forward веса E7 — сравнение с E7 на прокси (clim=True).

Всё, что использует данные, — строго до среза. Для среза C:
  * MH-модель учится на парах (срез c — воскресенье, целевой день d ≤ C), признаки известны на c;
  * стэкинг/веса учатся на прогнозах P/G/Gs прошлых срезов c с целевыми днями d ≤ C.

  uv run --python 3.12 --with-requirements ml/requirements.txt --with scipy python ml/research/mh_gbm.py
"""
import itertools
import json
import os
import pickle
import sys
import time

sys.path.insert(0, "ml")
sys.path.insert(0, "ml/research")
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core import fit_profile, wape_score  # noqa: E402
from proxy_lb import BASE_FEATS, FOLDS, load  # noqa: E402

H = 61
CACHE = os.environ.get("MH_CACHE", "/tmp/mh_gbm_cache.pkl")
T0 = time.time()
G_PARAMS = dict(objective="l1", n_estimators=600, learning_rate=0.05, num_leaves=63, verbose=-1)

y = load()
DMAX = y.date.max()
clim = pd.read_csv("data/external/weather_climatology_2015_2024.csv")
daily = y.groupby(["route", "date"]).agg(ysum=("y", "sum"), dt5=("dt5", "first")).reset_index()


def with_clim(df):
    return df.drop(columns=["t", "prcp", "snow"]).assign(month=df.date.dt.month, day=df.date.dt.day).merge(
        clim, on=["month", "day", "hour"], how="left").drop(columns=["month", "day"])


def window(c, end=None):
    end = min(end or c + pd.Timedelta(days=H), DMAX)
    return y[(y.date > c) & (y.date <= end)].copy()


def cut_feats(c, te):
    """Признаки, известные на срезе c, для целевых строк te (погода — климат)."""
    L = {}
    for w in (1, 2, 4, 8):
        d = daily[(daily.date > c - pd.Timedelta(weeks=w)) & (daily.date <= c)]
        L[f"lv{w}"] = d.groupby(["route", "dt5"]).ysum.median()
    L = pd.DataFrame(L)
    m = lambda w: daily[(daily.date > c - pd.Timedelta(weeks=w)) & (daily.date <= c)].groupby("route").ysum.mean()
    trend = (m(2) / m(6)).rename("trend")
    lv, sh = fit_profile(y[y.date > c - pd.Timedelta(weeks=9)].assign(dtype=lambda d: d.dt5), c)
    te = with_clim(te).join(L, on=["route", "dt5"]).join(trend, on="route")
    te["shape"] = te.join(sh, on=["route", "dt5", "hour"])["shape"].values
    te["p"] = (te.join(lv, on=["route", "dt5"]).level.fillna(0) * te["shape"].fillna(0)).values
    for w in (1, 4, 8):
        te[f"r{w}"] = te[f"lv{w}"] / te.lv2
    te["h"] = (te.date - c).dt.days
    te["cut"] = c
    return te


def g_preds(c, te):
    """G и Gs как в proxy_lb (G учится на всей истории до c, погода окна — климат)."""
    tr = y[y.date <= c]
    mdl = lgb.LGBMRegressor(**G_PARAMS, random_state=0).fit(tr[BASE_FEATS], tr.y, categorical_feature=["route"])
    te["g"] = np.clip(mdl.predict(te[BASE_FEATS]), 0, None)
    key = [te.route, te.date.dt.month, te.dt5 >= 5]
    te["gs"] = te.g * (te.groupby(key).p.transform("sum") / te.groupby(key).g.transform("sum").replace(0, np.nan)).fillna(0)
    return te


# ---------- 1. выборка по воскресным срезам + прогнозы P/G/Gs на каждом (кэш) ----------
sundays = pd.date_range("2025-01-12", DMAX - pd.Timedelta(days=7), freq="W-SUN")
if os.path.exists(CACHE):
    S, TE = pickle.load(open(CACHE, "rb"))
else:
    S = [g_preds(c, cut_feats(c, window(c))) for c in sundays]
    TE = {}
    for cut, end in FOLDS:
        cut, end = pd.Timestamp(cut), pd.Timestamp(end)
        TE[cut] = g_preds(cut, cut_feats(cut, window(cut, end)))
    pickle.dump((S, TE), open(CACHE, "wb"))
S = pd.concat(S, ignore_index=True)
print(f"выборка {len(S):,} строк, {S.cut.nunique()} срезов, {time.time()-T0:.0f}s", flush=True)

# sanity: P и E7 на срезах совпадают с proxy_lb (clim=True)
for cut, te in TE.items():
    print(f"  {cut:%d.%m}: P {wape_score(te.y, te.p):.4f}  E7 {wape_score(te.y, .5*te.p+.25*te.gs+.25*te.g):.4f}")

CAL = ["route", "hour", "dow", "dt5", "hol", "h"]
W = ["t", "prcp", "snow"]
FEATS_Y = CAL + ["p", "lv1", "lv2", "lv4", "lv8", "shape", "trend"]
FEATS_R = CAL + ["r1", "r4", "r8", "shape", "trend"]
MH_PARAMS = dict(objective="l1", n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def fit_mh(tr, te, feats, target, months=None, seeds=(0, 1)):
    if months:
        tr = tr[tr.cut > tr.cut.max() - pd.DateOffset(months=months)]
    out = np.zeros(len(te))
    for s in seeds:
        m = lgb.LGBMRegressor(**MH_PARAMS, random_state=s)
        if target == "y":
            m.fit(tr[feats], tr.y, categorical_feature=["route"])
            out += np.clip(m.predict(te[feats]), 0, None)
        else:   # отношение y/P с весом P → ŷ = P·ratio
            ok = tr.p > 0
            m.fit(tr.loc[ok, feats], (tr.y / tr.p)[ok].clip(0, 5), sample_weight=tr.p[ok], categorical_feature=["route"])
            out += te.p.values * np.clip(m.predict(te[feats]), 0, 5)
    return out / len(seeds)


def best_weights(d, cols, step=0.05):
    """Веса на симплексе, минимизирующие Σ|y−ŷ| на прошлых срезах."""
    X, yy = d[cols].values, d.y.values
    grid = np.arange(0, 1 + 1e-9, step)
    best, bw = np.inf, None
    for w in itertools.product(grid, repeat=len(cols) - 1):
        if sum(w) > 1 + 1e-9:
            continue
        w = np.array(list(w) + [1 - sum(w)])
        e = np.abs(yy - X @ w).sum()
        if e < best:
            best, bw = e, w
    return bw


rows, wlog = [], {}
for cut, end in FOLDS:
    cut, end = pd.Timestamp(cut), pd.Timestamp(end)
    te = TE[cut].copy()
    past = S[(S.date <= cut)]          # целевые дни прошлых срезов — строго до среза
    v = {"P": te.p, "E7": .5 * te.p + .25 * te.gs + .25 * te.g}
    # --- 1. многогоризонтный GBM ---
    for name, feats, tgt, mo in [("MH_y", FEATS_Y, "y", None), ("MH_y_W", FEATS_Y + W, "y", None),
                                 ("MH_y_3m", FEATS_Y, "y", 3), ("MH_r", FEATS_R, "r", None),
                                 ("MH_r_3m", FEATS_R, "r", 3), ("MH_r_W", FEATS_R + W, "r", None)]:
        if past.cut.nunique() < 3:
            continue
        m = fit_mh(past, te, feats, tgt, mo)
        v[name] = pd.Series(m, index=te.index)
        v[name + "+E7"] = .5 * v[name] + .5 * v["E7"]
        # только форма: ML перераспределяет день, дневная сумма = P
        k = [te.route, te.date]
        mm = v[name]
        v[name + "_shape"] = mm * (te.p.groupby(k).transform("sum") / mm.groupby(k).transform("sum").replace(0, np.nan)).fillna(0)
        v[name + "_shape+E7"] = .5 * v[name + "_shape"] + .5 * v["E7"]
        for a in (.2, .3):   # более осторожная доля MH в смеси
            v[f"{name}@{a}+E7"] = a * v[name] + (1 - a) * v["E7"]
            v[f"{name}_shape@{a}+E7"] = a * v[name + "_shape"] + (1 - a) * v["E7"]
    # --- 2/3. walk-forward веса и стэкинг поверх P, G, Gs ---
    if past.cut.nunique() >= 3:
        for tag, pp in [("all", past), ("3m", past[past.cut > cut - pd.DateOffset(months=3)])]:
            w = best_weights(pp, ["p", "gs", "g"])
            wlog[f"{cut:%d.%m}_{tag}"] = [round(float(x), 2) for x in w]
            v[f"WF_{tag}"] = te[["p", "gs", "g"]].values @ w
        sf = ["p", "g", "gs", "hour", "dt5", "h"]
        st = lgb.LGBMRegressor(objective="l1", n_estimators=200, learning_rate=0.05, num_leaves=15,
                               min_child_samples=500, verbose=-1).fit(past[sf], past.y)
        v["STACK_gbm"] = np.clip(st.predict(te[sf]), 0, None)
        v["STACK_gbm+E7"] = .5 * v["STACK_gbm"] + .5 * v["E7"]
    rows.append({"fold": f"{cut:%d.%m}→{end:%d.%m}", **{k: wape_score(te.y, x) for k, x in v.items()}})
    print(rows[-1]["fold"], f"{time.time()-T0:.0f}s", flush=True)

d = pd.DataFrame(rows).set_index("fold")
d.loc["среднее(5)"] = d.mean()
d.loc["среднее(2–5)"] = d.iloc[1:5].mean()
t = d.T
t["Δср_vs_E7"] = t["среднее(5)"] - t.loc["E7", "среднее(5)"]
t["Δ28.09_vs_E7"] = t["28.09→31.10"] - t.loc["E7", "28.09→31.10"]
t["побед_из_5"] = [(d.iloc[:5][k] > d.iloc[:5]["E7"]).sum() for k in t.index]
t = t.sort_values("среднее(5)", ascending=False)
pd.set_option("display.width", 250)
print(t.round(4).to_string())
print("walk-forward веса (P, Gs, G):", wlog)
json.dump({"table": t.round(5).reset_index().rename(columns={"index": "variant"}).to_dict("records"),
           "wf_weights": wlog}, open("ml/research/mh_gbm.json", "w"), ensure_ascii=False, indent=2, default=float)
