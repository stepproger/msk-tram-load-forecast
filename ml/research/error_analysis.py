"""Разбор ошибок E7 (честный режим clim=True) и точечные исправления профиля/смеси на 5 срезах proxy_lb.

E7 = 0.5·p + 0.25·gs + 0.25·g (как в proxy_lb.run). Всё, что оценивается в исправлениях, — только по данным ≤ среза.
  uv run --python 3.12 --with-requirements ml/requirements.txt --with scipy python ml/research/error_analysis.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "ml")
sys.path.insert(0, "ml/research")
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import trim_mean  # noqa: E402

from core import load_weather, wape_score  # noqa: E402
from proxy_lb import BASE_FEATS, FOLDS, load  # noqa: E402

PARAMS = dict(objective="l1", n_estimators=600, learning_rate=0.05, num_leaves=63, verbose=-1, n_jobs=8)
CLIM = pd.read_csv("data/external/weather_climatology_2015_2024.csv")
CACHE = Path("data/interim/ea_g.pkl")


def clim(te):
    return te.drop(columns=["t", "prcp", "snow"]).assign(month=te.date.dt.month, day=te.date.dt.day).merge(
        CLIM, on=["month", "day", "hour"], how="left").drop(columns=["month", "day"])


def gbm(tr, te):
    m = lgb.LGBMRegressor(**PARAMS, random_state=0).fit(tr[BASE_FEATS], tr.y, categorical_feature=["route"])
    return np.clip(m.predict(te[BASE_FEATS]), 0, None)


# ---------------- профиль с опциями ----------------
def shape_tab(s, dcol):
    day = s.groupby(["route", "date"]).y.transform("sum")
    sh = s.assign(sh=s.y / day.replace(0, np.nan)).groupby(["route", dcol, "hour"]).sh.median().fillna(0)
    return sh / sh.groupby(level=[0, 1]).transform("sum")


def profile(tr, cut, te, dcol="dt5", lw=2, sw=4, lvl="median", shrink=0.0, to="net", lvl_dcol=None, shp_dcol=None,
            lw_off=None, drop_anom=0.0):
    ld, sd = lvl_dcol or dcol, shp_dcol or dcol
    if drop_anom:   # выкинуть маршрут-дни-аномалии (сбой/закрытие): сумма < drop_anom × медианы того же типа дня за 8 нед
        h8 = tr[tr.date > cut - pd.Timedelta(weeks=8)]
        ds = h8.groupby(["route", dcol, "date"]).y.sum()
        med = ds.groupby(level=[0, 1]).transform("median")
        bad = ds[ds < drop_anom * med].reset_index()[["route", "date"]].assign(bad=1)
        tr = tr.merge(bad, on=["route", "date"], how="left")
        tr = tr[tr.bad.isna()].drop(columns="bad")
    win = lambda w: tr[(tr.date > cut - pd.Timedelta(weeks=w)) & (tr.date <= cut)]  # noqa: E731
    sh = shape_tab(win(sw), sd).rename("shape")
    if shrink:
        if to == "net":
            ref = sh.groupby(level=[1, 2]).mean()
            ref = ref / ref.groupby(level=0).transform("sum")
            refv = sh.reset_index().join(ref.rename("r"), on=[sd, "hour"]).r.values
        else:  # форма за 8 недель
            ref = shape_tab(win(8), sd)
            refv = ref.reindex(sh.index).fillna(0).values
        sh = pd.Series((1 - shrink) * sh.values + shrink * refv, sh.index, name="shape")
    d = win(lw).groupby(["route", ld, "date"]).y.sum()
    if lw_off:   # выходные: уровень по более длинному окну (в 2 нед всего 2 наблюдения)
        d2 = win(lw_off).groupby(["route", ld, "date"]).y.sum()
        d = pd.concat([d[d.index.get_level_values(1) < 5], d2[d2.index.get_level_values(1) >= 5]])
    f = {"median": "median", "mean": "mean", "trim": lambda x: trim_mean(x, 0.25)}[lvl]
    lv = d.groupby(level=[0, 1]).agg(f).rename("level")
    return (te.join(lv, on=["route", ld]).level.fillna(0) * te.join(sh, on=["route", sd, "hour"])["shape"].fillna(0)).values


def add_gs(te, p):
    off = te.dt5 >= 5
    key = [te.route, te.date.dt.month, off]
    t = te.assign(p=p)
    return (t.g * (t.groupby(key).p.transform("sum") / t.groupby(key).g.transform("sum").replace(0, np.nan)).fillna(0)).values


def e7(te, p, w=(.5, .25, .25)):
    return w[0] * p + w[1] * add_gs(te, p) + w[2] * te.g.values


def hour_corr(tr, cut, dcol="dt5", edges=None, per_route=False, k=1.0):
    """Сдвиг формы по часам из псевдо-бэктестов до среза: профиль на cut-7j прогнозирует следующие 28 дней (≤cut)."""
    rows = []
    for j in range(4, 9):
        c = cut - pd.Timedelta(weeks=j)
        h, f = tr[tr.date <= c], tr[(tr.date > c) & (tr.date <= c + pd.Timedelta(days=28))].copy()
        f["pp"] = profile(h, c, f, dcol)
        rows.append(f)
    f = pd.concat(rows)
    key = ["route", "hour"] if per_route else ["hour"]
    a = f.groupby(key)[["y", "pp"]].sum()
    tot = f.y.sum() / f.pp.sum()
    r = ((a.y / a.pp.replace(0, np.nan)) / tot).fillna(1).clip(0.7, 1.3)
    r = 1 + k * (r - 1)
    if edges is not None:
        r[~r.index.get_level_values("hour").isin(edges)] = 1.0
    return r


def apply_hc(te, p, r):
    key = ["route", "hour"] if r.index.nlevels == 2 else ["hour"]
    m = te.join(r.rename("m"), on=key).m.fillna(1).values
    q = p * m   # перенормировать к исходной дневной сумме профиля
    t = te.assign(p=p, q=q)
    s = (t.groupby(["route", "date"]).p.transform("sum") / t.groupby(["route", "date"]).q.transform("sum").replace(0, np.nan)).fillna(1)
    return q * s.values


GROUPS = {"ночь/утро 0-9": list(range(0, 10)), "день 10-15": list(range(10, 16)), "вечер 16-23": list(range(16, 24))}
GRID = [(a, b, 1 - a - b) for a in np.arange(0, 1.01, .125) for b in np.arange(0, 1.01, .125) if a + b <= 1.0001]


def fit_weights(y, cut):
    """Веса смеси по группам часов на внутреннем срезе cut-61д → cut (только данные ≤ cut)."""
    c0 = max(cut - pd.Timedelta(days=61), y.date.min() + pd.Timedelta(days=35))   # срез 1: данных мало → короче
    tr, te = y[y.date <= c0], clim(y[(y.date > c0) & (y.date <= cut)].copy())
    te["g"] = gbm(tr, te)
    p = profile(tr, c0, te)
    gs = add_gs(te, p)
    out = {}
    for gname, hrs in GROUPS.items():
        m = te.hour.isin(hrs).values
        best = min(GRID, key=lambda w: np.abs(te.y.values[m] - (w[0] * p + w[1] * gs + w[2] * te.g.values)[m]).sum())
        out[gname] = tuple(round(float(x), 3) for x in best)
    allb = min(GRID, key=lambda w: np.abs(te.y.values - (w[0] * p + w[1] * gs + w[2] * te.g.values)).sum())
    out["все"] = tuple(round(float(x), 3) for x in allb)
    return out


def main():
    y = load()
    y["dt7"] = np.where(y.hol == 1, 6, y.dow)
    w25 = load_weather()[["date", "hour", "t", "prcp", "snow"]].rename(columns={"t": "t_fact", "prcp": "prcp_fact", "snow": "snow_fact"})
    folds = []
    cache = pd.read_pickle(CACHE) if CACHE.exists() else {}
    for cut, end in FOLDS:
        cut, end = pd.Timestamp(cut), pd.Timestamp(end)
        tr, te = y[y.date <= cut], clim(y[(y.date > cut) & (y.date <= end)].copy())
        if (cut, end) not in cache:
            cache[(cut, end)] = gbm(tr, te)
        te["g"] = cache[(cut, end)]
        te["p"] = profile(tr, cut, te)
        te["e7"] = e7(te, te.p.values)
        folds.append((cut, end, tr, te))
    pd.to_pickle(cache, CACHE)
    name = lambda c, e: f"{c:%d.%m}→{e:%d.%m}"  # noqa: E731
    res = {"E7": {name(c, e): wape_score(te.y, te.e7) for c, e, _, te in folds}}
    print("E7 по срезам:", {k: round(v, 4) for k, v in res["E7"].items()})

    # ---------------- 1. Разбор ошибки ----------------
    an = {}
    for c, e, tr, te in folds:
        k = name(c, e)
        te["ae"] = (te.y - te.e7).abs()
        S, E = te.y.sum(), te.ae.sum()
        d = te.groupby(["route", "date"])
        orc = te.e7 * (d.y.transform("sum") / d.e7.transform("sum").replace(0, np.nan)).fillna(0)
        orc_net = te.e7 * (te.groupby("date").y.transform("sum") / te.groupby("date").e7.transform("sum"))
        daily = te.groupby("date").agg(y=("y", "sum"), p=("e7", "sum"), ae=("ae", "sum"), dow=("dow", "first"), hol=("hol", "first"))
        daily["err_share"] = daily.ae / E
        daily["bias"] = daily.p / daily.y - 1
        wf = te[["date", "hour"]].merge(w25, on=["date", "hour"], how="left")
        wx = te.assign(tf=wf.t_fact.values, pf=wf.prcp_fact.values, sf=wf.snow_fact.values).groupby("date").agg(
            t_clim=("t", "mean"), t_fact=("tf", "mean"), prcp_fact=("pf", "sum"), snow_fact=("sf", "sum"))
        daily = daily.join(wx)
        worst = daily.sort_values("ae", ascending=False).head(8)
        rd = te.groupby(["route", "date"]).agg(y=("y", "sum"), p=("e7", "sum"), ae=("ae", "sum"))
        rd_w = rd.sort_values("ae", ascending=False).head(8)
        excess = lambda g: (g.ae.sum() / E, g.ae.sum() / g.y.sum() if g.y.sum() else None)  # noqa: E731
        an[k] = {
            "score": 1 - E / S, "bias_total": te.e7.sum() / S - 1,
            "оракул_дневная_сумма_маршрута": {"score": wape_score(te.y, orc),
                                               "доля_ошибки_уровня": 1 - (te.y - orc).abs().sum() / E},
            "оракул_дневная_сумма_сети": {"score": wape_score(te.y, orc_net),
                                           "доля_ошибки": 1 - (te.y - orc_net).abs().sum() / E},
            "маршруты": {int(r): {"доля_ошибки": round(a, 3), "wape": round(b, 3), "bias": round(g.e7.sum() / g.y.sum() - 1, 3)}
                         for r, g in te.groupby("route") for a, b in [excess(g)]},
            "часы": {int(h): {"доля_ошибки": round(a, 3), "wape": round(b, 3) if b else None,
                              "bias": round(g.e7.sum() / g.y.sum() - 1, 3) if g.y.sum() else None}
                     for h, g in te.groupby("hour") for a, b in [excess(g)]},
            "дни_недели(0=Пн)": {int(t): {"доля_ошибки": round(a, 3), "wape": round(b, 3), "bias": round(g.e7.sum() / g.y.sum() - 1, 3)}
                                 for t, g in te.groupby("dt7") for a, b in [excess(g)]},
            "худшие_дни": [{"date": f"{i:%Y-%m-%d}", **{c2: (round(float(v), 3) if pd.notna(v) else None) for c2, v in r.items()}}
                           for i, r in worst.iterrows()],
            "худшие_маршрут_дни": [{"route": int(i[0]), "date": f"{i[1]:%Y-%m-%d}", "y": int(r.y), "pred": round(r.p), "ae": round(r.ae)}
                                   for i, r in rd_w.iterrows()],
            "дневная_ошибка_квантили": daily.err_share.describe().round(4).to_dict(),
            "по_неделям_bias": {f"{w:%m-%d}": round(float(v), 3) for w, v in
                                daily.groupby(pd.Grouper(freq="W-SUN")).apply(lambda x: x.p.sum() / x.y.sum() - 1).items()},
        }
        # ошибка компонент
        an[k]["компоненты"] = {"p": wape_score(te.y, te.p), "g": wape_score(te.y, te.g),
                               "gs": wape_score(te.y, add_gs(te, te.p.values)), "E7": 1 - E / S}
    res["анализ"] = an

    # ---------------- 2. Исправления ----------------
    def variant(fn):
        return {name(c, e): wape_score(te.y, fn(c, tr, te)) for c, e, tr, te in folds}

    V = {}
    V["E7"] = res["E7"]
    for a in (0.2, 0.4, 0.6):
        V[f"a_shrink_net_{a}"] = variant(lambda c, tr, te, a=a: e7(te, profile(tr, c, te, shrink=a, to="net")))
        V[f"a_shrink_8w_{a}"] = variant(lambda c, tr, te, a=a: e7(te, profile(tr, c, te, shrink=a, to="8w")))
    V["a_shape_6w"] = variant(lambda c, tr, te: e7(te, profile(tr, c, te, sw=6)))
    V["a_shape_3w"] = variant(lambda c, tr, te: e7(te, profile(tr, c, te, sw=3)))
    V["b_7types"] = variant(lambda c, tr, te: e7(te, profile(tr, c, te, dcol="dt7")))
    V["b_7types_shape_only"] = variant(lambda c, tr, te: e7(te, profile(tr, c, te, lvl_dcol="dt5", shp_dcol="dt7")))
    V["b_7types_level_only"] = variant(lambda c, tr, te: e7(te, profile(tr, c, te, lvl_dcol="dt7", shp_dcol="dt5")))
    V["b_7types_lw4"] = variant(lambda c, tr, te: e7(te, profile(tr, c, te, dcol="dt7", lw=4)))
    for lvl in ("mean", "trim"):
        for lw in (2, 3, 4):
            V[f"c_{lvl}_lw{lw}"] = variant(lambda c, tr, te, lvl=lvl, lw=lw: e7(te, profile(tr, c, te, lw=lw, lvl=lvl)))
    for lw in (3, 4):
        V[f"c_median_lw{lw}"] = variant(lambda c, tr, te, lw=lw: e7(te, profile(tr, c, te, lw=lw)))
    hc = {}
    for c, e, tr, te in folds:
        hc[c] = {"net": hour_corr(tr, c), "route": hour_corr(tr, c, per_route=True),
                 "edges": hour_corr(tr, c, edges=[0, 5, 6, 23]), "edges_route": hour_corr(tr, c, edges=[0, 5, 6, 23], per_route=True)}
    for kk in ("edges", "edges_route", "net", "route"):
        V[f"d_hourcorr_{kk}"] = variant(lambda c, tr, te, kk=kk: e7(te, apply_hc(te, te.p.values, hc[c][kk])))
    V["d_hourcorr_route_half"] = variant(lambda c, tr, te: e7(te, apply_hc(te, te.p.values, 1 + 0.5 * (hc[c]["route"] - 1))))
    V["d_hourcorr_net_half"] = variant(lambda c, tr, te: e7(te, apply_hc(te, te.p.values, 1 + 0.5 * (hc[c]["net"] - 1))))
    V["f_weekend_lw3"] = variant(lambda c, tr, te: e7(te, profile(tr, c, te, lw_off=3)))
    V["f_weekend_lw4"] = variant(lambda c, tr, te: e7(te, profile(tr, c, te, lw_off=4)))
    for a in (0.3, 0.5):
        V[f"f_drop_anom_{a}"] = variant(lambda c, tr, te, a=a: e7(te, profile(tr, c, te, drop_anom=a)))
    V["f_shape3w+hourcorr_net"] = variant(lambda c, tr, te: e7(te, apply_hc(te, profile(tr, c, te, sw=3), hc[c]["net"])))
    V["f_trim_lw2+hourcorr_net"] = variant(lambda c, tr, te: e7(te, apply_hc(te, profile(tr, c, te, lvl="trim"), hc[c]["net"])))
    res["поправки_часов_(сеть)"] = {name(c, e): {int(h): round(float(v), 3) for h, v in hc[c]["net"].items() if abs(v - 1) > .02}
                                   for c, e, _, _ in folds}

    W = {c: fit_weights(y, c) for c, e, _, _ in folds}
    res["веса_по_внутр_срезу"] = {name(c, e): W[c] for c, e, _, _ in folds}

    def by_hour_w(c, te, glob=False):
        p = te.p.values
        gs, g = add_gs(te, p), te.g.values
        out = np.zeros(len(te))
        for gname, hrs in GROUPS.items():
            w = W[c]["все"] if glob else W[c][gname]
            m = te.hour.isin(hrs).values
            out[m] = (w[0] * p + w[1] * gs + w[2] * g)[m]
        return out
    V["e_weights_by_hourgroup"] = variant(lambda c, tr, te: by_hour_w(c, te))
    V["e_weights_global_refit"] = variant(lambda c, tr, te: by_hour_w(c, te, glob=True))

    # сводка
    tab = pd.DataFrame(V).T
    tab["среднее"] = tab.mean(axis=1)
    base = tab.loc["E7"]
    delta = tab - base
    tab["Δсреднее"] = delta["среднее"]
    tab["срезов_лучше"] = (delta.drop(columns="среднее") > 0).sum(axis=1)
    last = name(pd.Timestamp(FOLDS[-1][0]), pd.Timestamp(FOLDS[-1][1]))
    tab["Δ28.09"] = delta[last]
    print(tab.round(4).sort_values("Δсреднее", ascending=False).to_string())
    res["исправления"] = tab.round(5).to_dict("index")
    json.dump(res, open("ml/research/error_analysis.json", "w"), ensure_ascii=False, indent=1, default=float)
    for k, a in an.items():
        print(k, "score", round(a["score"], 4), "bias", round(a["bias_total"], 3), "оракул маршрут-день",
              round(a["оракул_дневная_сумма_маршрута"]["score"], 4), "доля уровня", round(a["оракул_дневная_сумма_маршрута"]["доля_ошибки_уровня"], 3),
              "оракул сеть-день", round(a["оракул_дневная_сумма_сети"]["score"], 4), a["компоненты"])


if __name__ == "__main__":
    main()
