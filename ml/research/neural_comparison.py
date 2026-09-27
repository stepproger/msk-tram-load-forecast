"""Walk-forward comparison of daily neural forecasts with the hourly profile.

Run from the repository root, for example::

    uv run --python 3.12 --with-requirements ml/research/requirements-neural.txt \
      python ml/research/neural_comparison.py --steps 1000

All models predict 61 daily totals directly. The same hourly shape, fitted only
on dates before each cutoff, converts daily forecasts to the target grid.
Only calendar fields known at the cutoff are passed as future covariates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, NHITS, TiDE

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import fit_profile, load_labels, wape_score  # noqa: E402


HORIZON = 61
INPUT_SIZE = 28
CUTOFFS = ("2025-04-30", "2025-06-30", "2025-08-31")
EXOG = ["dow_sin", "dow_cos", "month_sin", "month_cos", "holiday", "work_sat"]
MODEL_NAMES = ("LSTM", "NHITS", "TiDE")


def add_calendar(frame: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    frame = frame.merge(calendar, left_on="ds", right_on="date", how="left").drop(columns="date")
    if frame.code.isna().any():
        raise ValueError("The calendar does not cover the forecast dates")
    dow = frame.ds.dt.dayofweek
    month = frame.ds.dt.month
    frame["holiday"] = ((frame.code == 1) & (dow < 5)).astype("float32")
    frame["work_sat"] = ((frame.code != 1) & (dow == 5)).astype("float32")
    frame["day_type"] = (frame.code == 1).astype("float32")  # 0=workday, 1=calendar day off
    frame["dow_sin"] = np.sin(2 * np.pi * dow / 7).astype("float32")
    frame["dow_cos"] = np.cos(2 * np.pi * dow / 7).astype("float32")
    frame["month_sin"] = np.sin(2 * np.pi * month / 12).astype("float32")
    frame["month_cos"] = np.cos(2 * np.pi * month / 12).astype("float32")
    return frame.drop(columns="code")


def make_model(name: str, steps: int, accelerator: str):
    common = dict(
        h=HORIZON,
        input_size=INPUT_SIZE,
        futr_exog_list=EXOG,
        max_steps=steps,
        batch_size=9,
        windows_batch_size=128,
        scaler_type="robust",
        learning_rate=0.001,
        random_seed=42,
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    if name == "LSTM":
        return LSTM(
            **common,
            encoder_n_layers=1,
            encoder_hidden_size=32,
            decoder_hidden_size=64,
            decoder_layers=1,
            h_train=HORIZON,
            recurrent=False,
        )
    if name == "NHITS":
        return NHITS(
            **common,
            stack_types=["identity", "identity"],
            n_blocks=[1, 1],
            mlp_units=[[64, 64], [64, 64]],
            n_pool_kernel_size=[2, 1],
            n_freq_downsample=[4, 1],
        )
    if name == "TiDE":
        return TiDE(
            **common,
            hidden_size=64,
            decoder_output_dim=16,
            temporal_decoder_dim=32,
            dropout=0.1,
        )
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--cutoffs", nargs="+", default=list(CUTOFFS))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--output", type=Path, default=Path("ml/research/neural_comparison.json"))
    args = parser.parse_args()

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(42)
    np.random.seed(42)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    y = load_labels()
    calendar = pd.read_csv("data/external/calendar_ru_2025.csv", parse_dates=["date"])
    daily = y.groupby(["route", "date"], as_index=False).y.sum()
    panel = daily.rename(columns={"route": "unique_id", "date": "ds"})
    panel["unique_id"] = panel.unique_id.astype(str)
    panel = add_calendar(panel, calendar).rename(columns={"y": "y"})
    panel["y"] = panel.y.astype("float32")
    rows = []

    for cutoff_text in args.cutoffs:
        cutoff = pd.Timestamp(cutoff_text)
        evaluation_end = min(cutoff + pd.Timedelta(days=HORIZON), y.date.max())
        if cutoff + pd.Timedelta(days=HORIZON) > y.date.max():
            raise ValueError(f"No full {HORIZON}-day label window after {cutoff_text}")
        train = panel[panel.ds <= cutoff].copy()
        futr_dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=HORIZON)
        future = pd.MultiIndex.from_product(
            [sorted(panel.unique_id.unique()), futr_dates], names=["unique_id", "ds"]
        ).to_frame(index=False)
        future = add_calendar(future, calendar)

        history = y[y.date <= cutoff].copy()
        future_actual = y[(y.date > cutoff) & (y.date <= evaluation_end)].copy()
        history_calendar = add_calendar(
            history.rename(columns={"date": "ds"})[["route", "ds", "hour", "y", "dtype"]], calendar
        )
        history_calendar["date"] = history_calendar.ds
        history_calendar["dtype"] = np.where(
            history_calendar.holiday == 1,
            6,
            np.where(history_calendar.work_sat == 1, 4, history_calendar.dtype),
        )
        level, shape = fit_profile(history_calendar, cutoff)
        profile = add_calendar(
            future_actual.rename(columns={"date": "ds"})[["route", "ds", "hour", "y", "dtype"]], calendar
        )
        profile["date"] = profile.ds
        profile["dtype"] = np.where(
            profile.holiday == 1,
            6,
            np.where(profile.work_sat == 1, 4, profile.dtype),
        )
        profile = profile.join(level, on=["route", "dtype"]).join(shape, on=["route", "dtype", "hour"])
        profile["profile"] = profile.level.fillna(0) * profile["shape"].fillna(0)
        profile["unique_id"] = profile.route.astype(str)
        actual = profile.y.to_numpy()
        profile_prediction = profile["profile"].to_numpy()
        score = wape_score(actual, profile_prediction)
        rows.append(dict(cutoff=cutoff_text, model="profile", wape_score=score,
                         training_seconds=0.0, inference_seconds=0.0))
        print(f"{cutoff_text} profile: score={score:.4f}", flush=True)

        for name in args.models:
            model = make_model(name, args.steps, accelerator)
            nf = NeuralForecast(models=[model], freq="D")
            started = perf_counter()
            nf.fit(df=train[["unique_id", "ds", "y", *EXOG]])
            training_seconds = perf_counter() - started
            started = perf_counter()
            predicted = nf.predict(futr_df=future[["unique_id", "ds", *EXOG]]).reset_index(drop=False)
            inference_seconds = perf_counter() - started
            if "unique_id" not in predicted.columns:
                raise RuntimeError(f"Missing route IDs in {name} predictions")
            predicted["unique_id"] = predicted.unique_id.astype(str)
            hourly = profile.merge(predicted[["unique_id", "ds", name]], on=["unique_id", "ds"], how="left", validate="many_to_one")
            if hourly[name].isna().any():
                raise RuntimeError(f"Missing daily predictions from {name}")
            forecast = np.maximum(0, hourly[name].to_numpy()) * hourly["shape"].fillna(0).to_numpy()
            model_score = wape_score(hourly.y.to_numpy(), forecast)
            blend_score = wape_score(hourly.y.to_numpy(), 0.5 * profile_prediction + 0.5 * forecast)
            rows.append(dict(cutoff=cutoff_text, model=name, wape_score=model_score,
                             blend_50_score=blend_score, training_seconds=training_seconds,
                             inference_seconds=inference_seconds))
            print(f"{cutoff_text} {name}: score={model_score:.4f}, blend={blend_score:.4f}, "
                  f"train={training_seconds:.1f}s, infer={inference_seconds:.3f}s", flush=True)

    result = dict(
        run_at=datetime.now(timezone.utc).isoformat(),
        device=accelerator,
        torch_version=torch.__version__,
        horizon_days=HORIZON,
        input_days=INPUT_SIZE,
        steps=args.steps,
        future_covariates=EXOG,
        fold_results=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
