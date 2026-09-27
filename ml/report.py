"""Графики для docs/MODEL.md и питча → docs/img/*.png.
uv run --python 3.12 --with-requirements ml/requirements.txt python ml/report.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("docs/img")
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, GRAY = "#2a78d6", "#eb6834", "#a3a29c"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "savefig.dpi": 160, "savefig.bbox": "tight"})
val = json.load(open("ml/research/validation_results.json", encoding="utf-8"))


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print(OUT / name)


# 1) Абляции на скрытом эталоне (LB): вклад каждого внешнего источника
abl = [("Производственный календарь", 0.0325), ("Окончание ремонта 7/50", 0.0053), ("Запуск маршрута 5", 0.0037),
       ("Бесплатный проезд 31.12", 0.0008), ("Пробки ≥9 баллов", 0.0), ("Множитель осадков в профиле", -0.0002)]
fig, ax = plt.subplots(figsize=(8, 3.4))
names, vals = [a for a, _ in abl][::-1], [b for _, b in abl][::-1]
ax.barh(names, vals, color=[S1 if v > 0 else GRAY for v in vals], height=0.55)
for i, v in enumerate(vals):
    ax.text(max(v, 0) + 0.0006, i, f"{v:+.4f}", va="center", ha="left", color=INK2)
ax.axvline(0, color=INK2, lw=1)
ax.set_xlim(-0.004, 0.038)
ax.grid(axis="y", visible=False)
ax.set_title("Вклад внешних данных на скрытом эталоне (абляции: профильная модель,\nбесплатный проезд — на релизе HONEST2)", loc="left", fontsize=12)
ax.set_xlabel("Δ WAPE-score = скор с источником − скор без него")
save(fig, "ablation_lb.png")

# 2) Кривая горизонта: где модель валидна
h = pd.DataFrame(val["horizon_curve"])
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.fill_between(h.wk, h.p10, h["median"], color=S1, alpha=0.12, lw=0)
ax.plot(h.wk, h["median"], color=S1, lw=2, marker="o", ms=5, label="Базовый профиль, медиана по 32 срезам")
ax.plot(h.wk, h.p10, color=S1, lw=1, ls="--", label="Худшие 10% срезов")
ax.scatter([8.7], [0.90257], color=S2, s=70, zorder=5, label="Релиз HONEST2 на лидерборде (61 день)")
ax.annotate("0.903 — релиз: профиль + календарь + события\n+ сезон + LightGBM", (8.7, 0.90257), (4.3, 0.918),
            color=INK2, fontsize=10, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.axhline(0.88, color=INK2, lw=1, ls=":")
ax.text(2.1, 0.872, "0.88 — порог максимального балла", color=INK2, fontsize=9)
ax.set_xlabel("Горизонт прогноза, недель")
ax.set_ylabel("WAPE-score")
ax.set_ylim(0.68, 0.94)
ax.set_xlim(0.7, 9.1)
ax.set_title("Точность профиля падает с горизонтом — явные слои и ML компенсируют", loc="left", fontsize=12)
ax.legend(frameon=False, loc="lower left", fontsize=9)
save(fig, "horizon_curve.png")

# 3) Погода в дождливые часы (out-of-sample)
r = val["weather_rain_hours"]
fig, ax = plt.subplots(figsize=(6.5, 3.4))
bins = list(r.keys())
x = range(len(bins))
b0 = [r[b]["bias_base"] * 100 for b in bins]
b1 = [r[b]["bias_weather"] * 100 for b in bins]
ax.bar([i - 0.18 for i in x], b0, width=0.34, color=GRAY, label="Без погоды")
ax.bar([i + 0.18 for i in x], b1, width=0.34, color=S1, label="С погодой")
for i in x:
    ax.text(i - 0.18, b0[i] + 0.3, f"{b0[i]:+.1f}%", ha="center", color=INK2, fontsize=10)
    ax.text(i + 0.18, max(b1[i], 0) + 0.3, f"{b1[i]:+.1f}%", ha="center", color=INK2, fontsize=10)
ax.set_xticks(list(x), [f"Осадки {b}\n({r[b]['hours']} ч)" for b in bins])
ax.axhline(0, color=INK2, lw=1)
ax.set_ylabel("Смещение прогноза, %")
ax.grid(axis="x", visible=False)
ax.set_title("Погода убирает завышение прогноза в дождь\n(проверка «неделя вперёд», без утечек)", loc="left", fontsize=12)
ax.legend(frameon=False)
save(fig, "weather_rain_hours.png")

# 4) Посадок на вагон в 8:00 будни (октябрь) — где не хватает вагонов
a = pd.read_parquet("artifacts/forecast/actuals_hourly.parquet")
a["date"] = pd.to_datetime(a.date)
p = a[(a.hour == 8) & (a.date.dt.dayofweek < 5) & (a.date.dt.month == 10) & (a.trams > 0)]
bpt = (p.groupby("route").boardings.sum() / p.groupby("route").trams.sum()).sort_values()
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.barh([f"№ {i}" for i in bpt.index], bpt.values, color=[S1 if i == 17 else GRAY for i in bpt.index], height=0.6)
for i, v in enumerate(bpt.values):
    ax.text(v + 2, i, f"{v:.0f}", va="center", color=INK2, fontsize=10)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Посадок на вагон в час, 8:00, будни октября")
ax.set_title("Маршрут 17 недообеспечен вагонами в утренний пик", loc="left", fontsize=12)
save(fig, "route_load_8am.png")

# 5) Прогноз против факта: неделя, маршрут 17 (скользящий бэктест «неделя вперёд»)
bt = pd.read_parquet("artifacts/forecast/backtest_hourly.parquet")
bt["date"] = pd.to_datetime(bt.date)
m = bt.merge(a, on=["route", "date", "hour"])
m = m[(m.route == 17) & m.date.between("2025-10-20", "2025-10-26")].sort_values(["date", "hour"])
ts = m.date + pd.to_timedelta(m.hour, unit="h")
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.plot(ts, m.boardings, color=GRAY, lw=2, label="Факт")
ax.plot(ts, m.yhat, color=S1, lw=2, label="Прогноз (неделя вперёд)")
ax.set_ylabel("Посадок в час")
ax.set_title("Маршрут 17, 20–26 октября: прогноз против факта", loc="left", fontsize=12)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
ax.legend(frameon=False, loc="upper right", ncol=2)
save(fig, "forecast_vs_actual_route17.png")

# ---------- графики для PDF-отчёта ----------
import numpy as np  # noqa: E402
import sys  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
DFMT = mdates.DateFormatter("%d.%m")
sys.path.insert(0, "ml")
from core import load_labels  # noqa: E402

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
y = load_labels()

# 6) Форма суток по типам дня, маршрут 17 (октябрь)
o = y[(y.route == 17) & (y.date >= "2025-10-01")]
sh = o.groupby(["dtype", "hour"]).y.median().unstack(0)
fig, ax = plt.subplots(figsize=(8, 3.6))
names = {0: "Пн", 1: "Вт–Чт", 4: "Пт", 5: "Сб", 6: "Вс"}
for i, t in enumerate([0, 1, 4, 5, 6]):
    ax.plot(sh.index, sh[t], color=CAT[i], lw=2, label=names[t])
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel("Час")
ax.set_ylabel("Посадок в час (медиана)")
ax.set_title("Маршрут 17: форма суток по типам дня (октябрь 2025)", loc="left", fontsize=12)
ax.legend(frameon=False, ncol=5, loc="upper left")
save(fig, "profile_shapes.png")

# 7) Режим выходных маршрута 50 (ремонт путей)
d50 = y[y.route == 50].groupby("date").y.sum()
d50 = d50["2025-08-01":"2025-10-31"]
fig, ax = plt.subplots(figsize=(9, 3.2))
wk = d50.index.dayofweek >= 5
ax.bar(d50.index[~wk], d50[~wk], color=GRAY, width=0.8, label="Будни")
ax.bar(d50.index[wk], d50[wk], color=S2, width=0.8, label="Выходные")
ax.axvline(pd.Timestamp("2025-09-05"), color=INK2, lw=1, ls=":")
ax.text(pd.Timestamp("2025-09-07"), d50.max() * 0.95, "с 06.09 — ремонт путей по выходным", color=INK2, fontsize=9)
ax.set_ylabel("Посадок в день")
ax.set_title("Маршрут 50: выходные «исчезают» — событие сети из новостей подтверждается данными", loc="left", fontsize=12)
ax.xaxis.set_major_formatter(DFMT)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
save(fig, "route50_weekends.png")

# 8) Калибровка сезонного множителя по трём сабмитам
kx, ky = np.array([1.00, 1.03, 1.06]), np.array([0.89083, 0.89249, 0.88734])
a2, b2, c2 = np.polyfit(kx, ky, 2)
xx = np.linspace(0.99, 1.07, 100)
fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.plot(xx, np.polyval([a2, b2, c2], xx), color=S1, lw=2, label="Парабола по S2, S1, S3")
ax.scatter(kx, ky, color=GRAY, s=50, zorder=5, label="Сабмиты S2 / S1 / S3")
ax.scatter([1.022], [0.89275], color=S2, s=70, zorder=6, label="S4 = ×1.022 (факт 0.89275)")
ax.set_xlabel("Сезонный множитель ноябрь–декабрь")
ax.set_ylabel("WAPE-score (LB)")
ax.set_title("Калибровка уровня: 3 точки → оптимум ×1.022", loc="left", fontsize=12)
ax.legend(frameon=False, fontsize=9, loc="lower center")
save(fig, "calibration_season.png")

# 9) Nowcast
fig, ax = plt.subplots(figsize=(6.5, 3.2))
hrs, b0, b1 = ["с 9:00", "с 12:00", "с 15:00"], [0.8814, 0.8787, 0.8801], [0.8996, 0.9031, 0.9072]
x = np.arange(3)
ax.bar(x - 0.18, b0, 0.34, color=GRAY, label="Прогноз утром")
ax.bar(x + 0.18, b1, 0.34, color=S1, label="Уточнён по факту")
for i in x:
    ax.text(i - 0.18, b0[i] + 0.002, f"{b0[i]:.3f}", ha="center", color=INK2, fontsize=9)
    ax.text(i + 0.18, b1[i] + 0.002, f"{b1[i]:.3f}", ha="center", color=INK2, fontsize=9)
ax.set_xticks(x, hrs)
ax.set_ylim(0.85, 0.92)
ax.set_ylabel("WAPE-score остатка дня")
ax.grid(axis="x", visible=False)
ax.set_title("Nowcast: прогноз уточняется по потоку валидаций", loc="left", fontsize=12)
ax.legend(frameon=False, loc="upper left")
save(fig, "nowcast.png")

# 10) Холодный старт
cs = pd.DataFrame({"route": [1, 7, 11, 12, 17, 25, 26, 28, 50],
                   "d7": [0.864, 0.878, 0.880, 0.893, 0.915, 0.817, 0.889, 0.863, 0.856],
                   "full": [0.919, 0.903, 0.904, 0.907, 0.941, 0.859, 0.924, 0.852, 0.893]})
fig, ax = plt.subplots(figsize=(8, 3.2))
x = np.arange(len(cs))
ax.bar(x - 0.18, cs.d7, 0.34, color=S1, label="7 дней истории + форма суток других маршрутов")
ax.bar(x + 0.18, cs.full, 0.34, color=GRAY, label="Полная история (4 недели)")
ax.set_xticks(x, [f"№ {r}" for r in cs.route])
ax.set_ylim(0.78, 0.96)
ax.set_ylabel("WAPE-score, октябрь")
ax.grid(axis="x", visible=False)
ax.set_title("Адаптация: новый маршрут выходит на рабочее качество за неделю", loc="left", fontsize=12)
ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
save(fig, "cold_start.png")

# 11) Прогноз ноябрь–декабрь в контексте истории (сумма по сети)
hist = y[y.date >= "2025-09-01"].groupby("date").y.sum()
hf = pd.read_parquet("artifacts/forecast/hourly.parquet")
hf["date"] = pd.to_datetime(hf.date)
fd = hf.groupby("date")[["yhat", "q10", "q90"]].sum()
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(hist.index, hist.values / 1000, color=GRAY, lw=1.8, label="Факт (сен–окт)")
ax.fill_between(fd.index, fd.q10 / 1000, fd.q90 / 1000, color=S1, alpha=0.15, lw=0, label="Интервал q10–q90")
ax.plot(fd.index, fd.yhat / 1000, color=S1, lw=1.8, label="Прогноз (ноя–дек)")
for d, t in [("2025-11-04", "4 ноября"), ("2025-12-16", "запуск №5"), ("2025-12-31", "31.12")]:
    ax.annotate(t, (pd.Timestamp(d), fd.yhat.get(pd.Timestamp(d), 0) / 1000), (0, -28), textcoords="offset points",
                ha="center", fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.set_ylabel("Посадок в день, тыс.")
ax.set_title("Прогноз сети на ноябрь–декабрь 2025 в продолжение истории", loc="left", fontsize=12)
ax.xaxis.set_major_formatter(DFMT)
ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
save(fig, "forecast_novdec.png")

# 12) Выбор модели: средний и худший WAPE-score по 5 фолдам
ms = json.load(open("ml/research/model_selection.json", encoding="utf-8"))
sc = pd.DataFrame(ms["scores"]).sort_values("среднее")
fig, ax = plt.subplots(figsize=(9, 4.2))
yy = np.arange(len(sc))
ours = sc.model.str.startswith("Наш") | sc.model.str.startswith("Ансамбль")
ax.barh(yy, sc["среднее"], color=[S1 if o else GRAY for o in ours], height=0.55, label="Среднее по 5 фолдам")
ax.scatter(sc["худший"], yy, color=S2, s=36, zorder=5, label="Худший фолд")
for i, (m_, w_) in enumerate(zip(sc["среднее"], sc["худший"])):
    ax.text(m_ + 0.003, i, f"{m_:.3f}", va="center", color=INK2, fontsize=9)
ax.set_yticks(yy, sc.model, fontsize=9)
ax.set_xlim(0.68, 0.9)
ax.grid(axis="y", visible=False)
ax.set_xlabel("WAPE-score (прогноз на 1–2 месяца вперёд)")
ax.set_title("Сравнение моделей на реальных данных, 5 временных фолдов", loc="left", fontsize=12)
ax.legend(frameon=False, loc="lower right", fontsize=9)
save(fig, "model_selection.png")

# ---------- графики для HTML-отчёта ----------
# 13) История лидерборда
lb = [("S1", .89249, "calib"), ("S2", .89083, "calib"), ("S3", .88734, "calib"), ("S4", .89275, "calib"), ("S5", .89252, "abl"),
      ("S6", .88910, "abl"), ("S7", .88742, "abl"), ("S8", .86026, "abl"), ("S9", .89275, "abl"), ("S10", .89254, "abl"),
      ("S11", .89084, "calib"), ("S12", .89167, "calib"), ("S13", .89316, "calib"), ("S14", .89161, "calib"), ("S15", .89033, "calib"),
      ("S16", .89217, "calib"), ("S21", .88542, "abl"), ("S26", .89325, "retro"), ("S31", .89329, "retro"), ("E1", .89840, "ml"),
      ("E2", .89939, "ml"), ("E3", .89916, "ml"), ("E4", .89860, "ml"), ("E5", .89405, "ml"), ("E6", .89956, "ml"), ("E7", .90017, "ml"),
      ("HONEST", .90174, "honest"), ("HONEST2", .90257, "honest"), ("HONEST4", .90066, "honest")]
col = {"calib": GRAY, "abl": "#c9c8c2", "retro": "#b3a36b", "ml": S1, "honest": S2}
lab = {"calib": "профиль, калибровка по LB", "abl": "абляции источников", "retro": "с данными задним числом",
       "ml": "профиль + LightGBM (подбор по LB)", "honest": "честные релизы"}
fig, ax = plt.subplots(figsize=(10, 3.8))
for k in col:
    idx = [i for i, r in enumerate(lb) if r[2] == k]
    ax.scatter(idx, [lb[i][1] for i in idx], color=col[k], s=34, zorder=3, label=lab[k])
ax.plot(range(len(lb)), [r[1] for r in lb], color=GRID, lw=1, zorder=1)
ax.axhline(0.88, color=INK2, lw=1, ls=":")
ax.text(0.2, 0.8806, "0.88 — порог максимального балла", color=INK2, fontsize=9)
for i, r in enumerate(lb):
    if r[0] in ("S1", "S8", "S13", "E2", "E7", "HONEST2", "HONEST4"):
        ax.annotate(f"{r[0]}\n{r[1]:.4f}", (i, r[1]), (0, -26 if r[0] in ("S8", "HONEST4") else 10), textcoords="offset points", ha="center", fontsize=8, color=INK2)
ax.set_xticks(range(len(lb)), [r[0] for r in lb], rotation=90, fontsize=8)
ax.set_ylim(0.855, 0.908)
ax.set_ylabel("WAPE-score (лидерборд)")
ax.set_title("Все загрузки по порядку: абляции, калибровки, ML-ансамбль, честные релизы", loc="left", fontsize=12)
ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=2)
save(fig, "lb_history.png")

# 14) Эксперименты на локальном прокси (честный режим, климат)
ex = [("Профиль (база)", .8505, .9023), ("E7: профиль + LightGBM", .8568, .9052), ("+ почасовая поправка формы", .8576, .9086),
      ("+ MLP 10%", .8574, .9060), ("+ N-HiTS 10%", .8574, .9060), ("+ многогоризонтный GBM 20%", .8577, .9064),
      ("Стэкинг поверх P, G, Gs", .8530, .9017), ("Веса по прошлым срезам", .8537, .9000), ("Модель дрейфа", .8222, .8940)]
fig, ax = plt.subplots(figsize=(9, 4.4))
yy = np.arange(len(ex))[::-1]
ax.barh(yy + .18, [e[1] for e in ex], .34, color=S1, label="Среднее по 5 срезам")
ax.barh(yy - .18, [e[2] for e in ex], .34, color=GRAY, label="Срез 28.09→31.10 (ближе всего к задаче)")
for y_, e in zip(yy, ex):
    ax.text(e[1] + .001, y_ + .18, f"{e[1]:.4f}", va="center", fontsize=8, color=INK2)
    ax.text(e[2] + .001, y_ - .18, f"{e[2]:.4f}", va="center", fontsize=8, color=INK2)
ax.set_yticks(yy, [e[0] for e in ex], fontsize=9)
ax.set_xlim(0.8, 0.925)
ax.grid(axis="y", visible=False)
ax.set_xlabel("WAPE-score на локальном прокси (погода окна = климатическая норма)")
ax.set_title("Что проверяли: всё сверх E7 — в пределах шума, кроме почасовой поправки", loc="left", fontsize=12, pad=26)
ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
save(fig, "experiments_proxy.png")

# 15) Почасовая поправка формы (как применена в HONEST4)
hc = {0: .944, 1: 1.157, 2: 1.3, 4: .967, 5: .94, 6: .946, 7: 1.0, 8: 1.036, 9: 1.024, 10: 1.024, 11: 1.022, 12: 1.023, 13: 1.027,
      14: 1.032, 15: 1.027, 16: 1.036, 17: 1.0, 18: .972, 19: .92, 20: .907, 21: .925, 22: .884, 23: .935}
hh = list(range(4, 24))
fig, ax = plt.subplots(figsize=(8, 3.2))
v = [hc.get(h, 1.0) for h in hh]
ax.bar(hh, [x - 1 for x in v], color=[S1 if x >= 1 else S2 for x in v], width=0.7)
ax.axhline(0, color=INK2, lw=1)
ax.set_xticks(hh)
ax.set_xlabel("Час")
ax.set_ylabel("Поправка к профилю")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.grid(axis="x", visible=False)
ax.set_title("Осенью днём ездят больше, вечером меньше: поправка формы из псевдо-бэктестов до 31.10", loc="left", fontsize=12)
save(fig, "hour_correction.png")

# 16) Горизонт «год»: прогноз городского потока трамваев на 2026 против факта data.mos.ru
yh = json.load(open("ml/research/year_horizon_report.json", encoding="utf-8", encoding="utf-8"))
t = pd.DataFrame(yh["oos_2026_table_origin_2025_12"])
sel, nv = t[t.variant == yh["city_selected"]].reset_index(drop=True), t[t.variant == "naive"].reset_index(drop=True)
MON = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]
x = np.arange(len(sel))
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.bar(x - .2, sel.fact / 1e6, .38, color=GRAY, label="Факт data.mos.ru")
ax.bar(x + .2, sel.forecast / 1e6, .38, color=S1, label=f"Наш прогноз (MAPE {yh['oos_2026_mape_pct_jan_aug']['origin 2025-12'][yh['city_selected']]}%)")
ax.plot(x, nv.forecast / 1e6, color=S2, lw=1.5, marker="o", ms=4,
        label=f"Наивный «тот же месяц год назад» (MAPE {yh['oos_2026_mape_pct_jan_aug']['origin 2025-12']['naive']}%)")
for i, r in sel.iterrows():
    ax.text(i + .2, r.forecast / 1e6 + .3, f"{r.err_pct:+.1f}%", ha="center", fontsize=8, color=INK2)
ax.set_xticks(x, [MON[int(m[5:]) - 1] + " 2026" for m in sel.month], fontsize=9)
ax.set_ylim(0, max(sel.fact.max(), sel.forecast.max()) / 1e6 * 1.25)
ax.set_ylabel("Поездок на трамвае за месяц, млн")
ax.grid(axis="x", visible=False)
ax.set_title("Горизонт «год»: прогноз на 2026 по данным до декабря 2025 против факта", loc="left", fontsize=12)
ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=1)
save(fig, "year_2026.png")
