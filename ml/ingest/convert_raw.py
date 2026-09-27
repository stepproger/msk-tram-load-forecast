"""Конвертация сырых dataset/{train,test}.csv (~10 ГБ) в data/interim/*.parquet (~1 мин).

Запуск из корня репо:  uv run --python 3.12 --with duckdb python ml/ingest/convert_raw.py
"""
from pathlib import Path

import duckdb

Path("data/interim").mkdir(parents=True, exist_ok=True)

COLS = {c: "VARCHAR" for c in [
    "tran_no", "device_no", "tran_date_time", "begin_date_time", "input_date_time",
    "crd_hashcode", "validation_result", "tran_type_id", "place_id", "good_type",
    "pass_route", "ngpt_route", "bus_exit_no", "garage_number",
]}

con = duckdb.connect(config={"memory_limit": "3GB", "temp_directory": "data/interim/duckdb_tmp", "preserve_insertion_order": False})
for name in ["train", "test"]:
    src = (f"read_csv('dataset/{name}.csv', delim=';', header=true, quote='', escape='', "
           f"columns={COLS}, strict_mode=false, parallel=false)")
    con.sql(f"""
        COPY (
            SELECT tran_no, device_no,
                   try_cast(tran_date_time AS TIMESTAMP) AS ts,
                   crd_hashcode AS card,
                   try_cast(validation_result AS INT) AS vr,
                   tran_type_id AS tt, place_id, good_type,
                   ngpt_route AS route_raw, bus_exit_no AS exit_no, garage_number AS garage
            FROM {src}
        ) TO 'data/interim/{name}.parquet' (FORMAT parquet, COMPRESSION zstd)
    """)
    print(name, con.sql(f"SELECT count(*) FROM 'data/interim/{name}.parquet'").fetchone()[0])
