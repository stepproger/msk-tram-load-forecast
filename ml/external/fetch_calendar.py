"""Производственный календарь РФ 2025 (isdayoff.ru) + сверка с ml/config/events.py. https://isdayoff.ru/"""
import sys
import urllib.request
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from config import events as ev  # noqa: E402

out = Path("data/external/calendar_ru_2025.csv")
parser = argparse.ArgumentParser()
parser.add_argument("--check-local", action="store_true", help="validate existing CSV without network or writes")
parser.add_argument("--refresh", action="store_true", help="download again even if the CSV is already in git")
args = parser.parse_args()
if args.check_local or (out.exists() and not args.refresh):
    cal = pd.read_csv(out, parse_dates=["date"], dtype={"code": str})
else:
    codes = urllib.request.urlopen("https://isdayoff.ru/api/getdata?year=2025&pre=1", timeout=60).read().decode()
    cal = pd.DataFrame({"date": pd.date_range("2025-01-01", "2025-12-31"), "code": list(codes)})  # 0 раб, 1 вых, 2 сокр
    out.parent.mkdir(parents=True, exist_ok=True)
    cal.to_csv(out, index=False)
assert len(cal) == 365 and cal.date.is_unique and cal.code.notna().all(), "календарь неполон или содержит дубли"
assert set(cal.code.astype(str)) <= {"0", "1", "2"}, f"неизвестный код календаря: {set(cal.code)}"

# Cross-check the full year against Government Resolution No. 1335 of
# 04.10.2024 and the official five-day 2025 production calendar.
# 0 and 2 are working days; 1 is a non-working day; 2 is shortened.
expected_off_weekdays = {pd.Timestamp(d) for d in [
    "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08",
    "2025-05-01", "2025-05-02", "2025-05-08", "2025-05-09",
    "2025-06-12", "2025-06-13", "2025-11-03", "2025-11-04", "2025-12-31",
]}
expected_work_weekends = {pd.Timestamp("2025-11-01")}
expected_shortened = {pd.Timestamp(d) for d in [
    "2025-03-07", "2025-04-30", "2025-06-11", "2025-11-01",
]}
off_weekdays = set(cal[(cal.code == "1") & (cal.date.dt.dayofweek < 5)].date)
work_weekends = set(cal[(cal.code != "1") & (cal.date.dt.dayofweek >= 5)].date)
shortened = set(cal[cal.code == "2"].date)
assert off_weekdays == expected_off_weekdays, f"нерабочие будни разошлись: {off_weekdays ^ expected_off_weekdays}"
assert work_weekends == expected_work_weekends, f"рабочие выходные разошлись: {work_weekends ^ expected_work_weekends}"
assert shortened == expected_shortened, f"сокращённые дни разошлись: {shortened ^ expected_shortened}"
assert int(cal.code.isin(["0", "2"]).sum()) == 247
assert int((cal.code == "1").sum()) == 118
assert off_weekdays.intersection(ev.HOLIDAYS) == ev.HOLIDAYS
assert work_weekends.intersection(ev.WORKING_SATURDAYS) == ev.WORKING_SATURDAYS
print(out, "OK: 365 dates; 247 working (including shortened), 118 non-working; official calendar matches")
