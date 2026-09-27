"""Нейросеть как ТРЕТИЙ член ансамбля E7 (0.5·профиль + 0.25·LGB_перемасштаб + 0.25·LGB) на прокси лидерборда.

Те же 5 срезов, тот же код метрики и признаков, что в proxy_lb.run(clim=True): погода окна = климатическая норма.
Варианты нейросети:
  mlp   — ResNet-MLP (PyTorch, L1), эмбеддинги route/hour/dow/dt5 + hol/pre_hol/погода/долгота дня, 3 seed-а.
  mlpw  — то же, но веса обучения затухают по давности (полураспад 30 дней).
  nhits — N-HiTS (neuralforecast) на дневных суммах × часовая форма профиля (как в neural_comparison.py).
Для каждой: «_s» — перемасштаб к уровню профиля по (маршрут, месяц, выходной), как gs в E7.
Смеси: (1−w)·E7 + w·NN, w ∈ {0.1..0.4}.

  uv run --python 3.12 --with-requirements ml/research/requirements-neural.txt --with lightgbm,scipy \
    python ml/research/nn_ensemble.py [--no-nhits]
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, "ml")
sys.path.insert(0, "ml/research")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

if "--lgb-only" not in sys.argv:   # torch и lightgbm в одном процессе на macOS виснут (два libomp) — LGB в подпроцессе
    import torch  # noqa: E402
    import torch.nn as nn  # noqa: E402
else:
    import lightgbm as lgb  # noqa: E402
    nn = type("nn", (), {"Module": object})

from core import fit_profile, wape_score  # noqa: E402
from proxy_lb import BASE_FEATS, FOLDS, load  # noqa: E402

SEED = int(os.environ.get("NN_SEED", 42))
CAT = ["route", "hour", "dow", "dt5"]
NUM = ["hol", "pre_hol", "t", "prcp", "snow", "day_len", "dark"]
WEIGHTS = (0.1, 0.2, 0.3, 0.4)


class ResMLP(nn.Module):
    def __init__(self, cards, n_num, width=128, blocks=3, emb=8):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, emb) for c in cards])
        self.inp = nn.Linear(emb * len(cards) + n_num, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(),
                                                   nn.Dropout(0.1), nn.Linear(width * 2, width)) for _ in range(blocks)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, xc, xn):
        h = self.inp(torch.cat([e(xc[:, i]) for i, e in enumerate(self.embs)] + [xn], 1))
        for b in self.blocks:
            h = h + b(h)
        return nn.functional.softplus(self.out(h).squeeze(1))


def mlp_predict(tr, te, seeds=(0, 1, 2), half_life=None, epochs=30, bs=512, device="cpu"):
    cards = [9, 24, 7, 7]
    rmap = {r: i for i, r in enumerate(sorted(tr.route.unique()))}

    def enc(d):
        xc = np.stack([d.route.map(rmap).values, d.hour.values, d.dow.values, d.dt5.values], 1).astype(np.int64)
        return xc, d[NUM].astype(float).values
    xc_tr, xn_tr = enc(tr)
    xc_te, xn_te = enc(te)
    mu, sd = xn_tr.mean(0), xn_tr.std(0) + 1e-6
    xn_tr, xn_te = (xn_tr - mu) / sd, (xn_te - mu) / sd
    scale = 1000.0
    yt = tr.y.values / scale
    w = np.ones(len(tr)) if half_life is None else 0.5 ** ((tr.date.max() - tr.date).dt.days.values / half_life)
    w = w / w.mean()
    T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)  # noqa: E731
    Xc, Xn, Y, W = T(xc_tr, torch.long), T(xn_tr), T(yt), T(w)
    Xc_te, Xn_te = T(xc_te, torch.long), T(xn_te)
    pred = np.zeros(len(te))
    for s in seeds:
        torch.manual_seed(SEED + s)
        np.random.seed(SEED + s)
        m = ResMLP(cards, Xn.shape[1]).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
        steps = epochs * int(np.ceil(len(Y) / bs))
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
        g = torch.Generator(device="cpu").manual_seed(SEED + s)
        m.train()
        for _ in range(epochs):
            perm = torch.randperm(len(Y), generator=g)
            for i in range(0, len(Y), bs):
                ix = perm[i:i + bs]
                loss = (W[ix] * (m(Xc[ix], Xn[ix]) - Y[ix]).abs()).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sch.step()
        m.eval()
        with torch.no_grad():
            pred += m(Xc_te, Xn_te).cpu().numpy() * scale / len(seeds)
    return np.clip(pred, 0, None)


def nhits_predict(tr, te, steps=500):
    """Дневные суммы N-HiTS (конфиг из neural_comparison.py, календарь proxy_lb) × часовая форма профиля (te.shape)."""
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS

    def cal(d):
        dow, mon = d.ds.dt.dayofweek, d.ds.dt.month
        return d.assign(dow_sin=np.sin(2 * np.pi * dow / 7), dow_cos=np.cos(2 * np.pi * dow / 7),
                        month_sin=np.sin(2 * np.pi * mon / 12), month_cos=np.cos(2 * np.pi * mon / 12))
    ex = ["dow_sin", "dow_cos", "month_sin", "month_cos", "hol", "pre_hol"]
    daily = tr.groupby(["route", "date"]).agg(y=("y", "sum"), hol=("hol", "first"), pre_hol=("pre_hol", "first")).reset_index()
    daily = cal(daily.rename(columns={"route": "unique_id", "date": "ds"}).astype({"unique_id": str}))
    fut = te.groupby(["route", "date"]).agg(hol=("hol", "first"), pre_hol=("pre_hol", "first")).reset_index()
    fut = cal(fut.rename(columns={"route": "unique_id", "date": "ds"}).astype({"unique_id": str}))
    h = fut.ds.nunique()
    model = NHITS(h=h, input_size=28, futr_exog_list=ex, max_steps=steps, batch_size=9, windows_batch_size=128,
                  scaler_type="robust", learning_rate=1e-3, random_seed=SEED, accelerator="cpu", devices=1,
                  logger=False, enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                  stack_types=["identity", "identity"], n_blocks=[1, 1], mlp_units=[[64, 64], [64, 64]],
                  n_pool_kernel_size=[2, 1], n_freq_downsample=[4, 1])
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=daily[["unique_id", "ds", "y", *ex]])
    p = nf.predict(futr_df=fut[["unique_id", "ds", *ex]]).reset_index(drop=False)
    p["route"] = p.unique_id.astype(int)
    d = te[["route", "date"]].merge(p.rename(columns={"ds": "date"})[["route", "date", "NHITS"]],
                                    on=["route", "date"], how="left")
    return np.clip(d.NHITS.fillna(0).values, 0, None) * te["shape"].fillna(0).values


def lgb_stage(out):
    """Профиль + LightGBM ровно как proxy_lb.run(clim=True, seeds=(0,)); сохраняет te по срезам."""
    y, res = load(), []
    c = pd.read_csv("data/external/weather_climatology_2015_2024.csv")
    params = dict(objective="l1", n_estimators=600, learning_rate=0.05, num_leaves=63, verbose=-1)
    for cut, end in FOLDS:
        cut, end = pd.Timestamp(cut), pd.Timestamp(end)
        tr, te = y[y.date <= cut], y[(y.date > cut) & (y.date <= end)].copy()
        te = te.drop(columns=["t", "prcp", "snow"]).assign(month=te.date.dt.month, day=te.date.dt.day).merge(
            c, on=["month", "day", "hour"], how="left").drop(columns=["month", "day"])
        lv, sh = fit_profile(tr.assign(dtype=tr.dt5), cut)
        te["shape"] = te.join(sh, on=["route", "dt5", "hour"])["shape"].values
        te["p"] = (te.join(lv, on=["route", "dt5"]).level.fillna(0) * te["shape"].fillna(0)).values
        t0 = time.time()
        m = lgb.LGBMRegressor(**params, random_state=0).fit(tr[BASE_FEATS], tr.y, categorical_feature=["route"])
        te["g"] = np.clip(m.predict(te[BASE_FEATS]), 0, None)
        res.append((cut, end, te, time.time() - t0))
    pickle.dump(res, open(out, "wb"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-nhits", action="store_true")
    ap.add_argument("--nhits-steps", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lgb-only")
    args = ap.parse_args()
    if args.lgb_only:
        return lgb_stage(args.lgb_only)
    cache = os.path.join(tempfile.gettempdir(), "nn_ensemble_lgb.pkl")
    subprocess.run([sys.executable, __file__, "--lgb-only", cache], check=True)
    torch.set_num_threads(8)
    y, rows, times = load(), [], {}
    for cut, end, te, lgb_sec in pickle.load(open(cache, "rb")):
        tr = y[y.date <= cut]
        times.setdefault("lgb", []).append(lgb_sec)
        off = te.dt5 >= 5
        key = [te.route, te.date.dt.month, off]

        def resc(col):
            return te[col] * (te.groupby(key).p.transform("sum")
                              / te.groupby(key)[col].transform("sum").replace(0, np.nan)).fillna(0)
        te["gs"] = resc("g")
        te["E7"] = .5 * te.p + .25 * te.gs + .25 * te.g
        nets = {}
        t0 = time.time()
        te["mlp"] = mlp_predict(tr, te, epochs=args.epochs)
        times.setdefault("mlp", []).append(time.time() - t0)
        nets["mlp"] = 1
        t0 = time.time()
        te["mlpw"] = mlp_predict(tr, te, half_life=30, epochs=args.epochs)
        times.setdefault("mlpw", []).append(time.time() - t0)
        nets["mlpw"] = 1
        if not args.no_nhits:
            t0 = time.time()
            te["nhits"] = nhits_predict(tr, te, args.nhits_steps)
            times.setdefault("nhits", []).append(time.time() - t0)
            nets["nhits"] = 1
        v = {"base": te.p, "E7": te.E7}
        for n in nets:
            te[n + "_s"] = resc(n)
            for col in (n, n + "_s"):
                v[col] = te[col]
                for w in WEIGHTS:
                    v[f"E7+{w:.1f}·{col}"] = (1 - w) * te.E7 + w * te[col]
            # вариант «внутри» E7: 0.5p + 0.25gs + 0.25 смеси g и NN
            v[f"E7[g→½g+½{n}_s]"] = .5 * te.p + .25 * te.gs + .125 * te.g + .125 * te[n + "_s"]
        r = {"fold": f"{cut:%d.%m}→{end:%d.%m}", **{k: wape_score(te.y, x) for k, x in v.items()}}
        rows.append(r)
        print(r["fold"], {k: round(r[k], 4) for k in ["base", "E7", *[n for n in nets], *[n + "_s" for n in nets]]},
              {k: round(np.mean(t), 1) for k, t in times.items()}, flush=True)
    d = pd.DataFrame(rows).set_index("fold")
    d.loc["среднее"] = d.mean()
    d = d.T
    d["Δсреднее_vs_E7"] = d["среднее"] - d.loc["E7", "среднее"]
    d["Δ28.09_vs_E7"] = d["28.09→31.10"] - d.loc["E7", "28.09→31.10"]
    pd.set_option("display.width", 250)
    print(d.sort_values("среднее", ascending=False).round(4).to_string())
    json.dump({"clim": True, "seed": SEED, "train_seconds_per_fold": {k: float(np.mean(t)) for k, t in times.items()},
               "scores": d.round(5).reset_index(names="variant").to_dict("records")},
              open("ml/research/nn_ensemble.json", "w"), ensure_ascii=False, indent=2, default=float)


if __name__ == "__main__":
    main()
